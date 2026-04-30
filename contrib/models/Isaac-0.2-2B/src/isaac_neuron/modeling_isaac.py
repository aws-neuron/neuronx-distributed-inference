# Copyright 2025 © Amazon.com and Affiliates
"""Isaac NxDI orchestrator: VLM model combining vision encoder and Qwen3 text decoder.

Isaac-0.2-2B-Preview architecture:
- Vision: SigLIP2 (27 layers) -> pixel shuffle (2x2) -> 2-layer MLP projector
- Text: Qwen3 (28 layers, 2048 hidden, GQA 16/8)
- mRoPE: interleaved, section=(2,1,1) weighting -> ~[32,16,16]
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import copy  # noqa: E402
import logging  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.nn.utils.rnn as rnn_utils  # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402

import neuronx_distributed_inference.modules.autobucketing as autobucketing  # noqa: E402
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.image_to_text_model_base import (  # noqa: E402
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (  # noqa: E402
    ImageToTextModelWrapper,
    IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (  # noqa: E402
    pad_vision_embeddings,
)
from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    VISION_ENCODER_MODEL_TAG,
)
from neuronx_distributed_inference.modules.flashdecode.utils import (  # noqa: E402
    calculate_num_cores_per_group,
)

from isaac_neuron.modeling_isaac_text import NeuronIsaacTextModel  # noqa: E402
from isaac_neuron.modeling_isaac_vision import (  # noqa: E402
    NeuronIsaacVisionModel,
    IsaacVisionModelWrapper,
)
from isaac_neuron.utils import convert_state_dict_to_fused_qkv, StateDict  # noqa: E402

logger = logging.getLogger("Neuron")


class IsaacInferenceConfig(ImageToTextInferenceConfig):
    """Isaac-specific inference configuration.

    Extends ImageToTextInferenceConfig with:
    - pixel_shuffle_scale from model config
    - projector_intermediate_size from model config
    - Isaac-specific required attributes
    """

    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

        # Isaac uses hidden_act for the text model MLP (SiLU)
        if not hasattr(self.text_config, "hidden_act"):
            self.text_config.hidden_act = "silu"

        # Isaac's SigLIP2 encoder does NOT use a pooling head
        # (no head weights in the checkpoint; features go to pixel shuffle + MLP projector)
        if not hasattr(self.vision_config, "vision_use_head"):
            self.vision_config.vision_use_head = False

        # Extract Isaac-specific config values
        # pixel_shuffle_scale is in the vision_config or top-level config
        if not hasattr(self, "pixel_shuffle_scale"):
            self.pixel_shuffle_scale = getattr(
                self.vision_config, "pixel_shuffle_scale", 2
            )

        # Projector intermediate size
        if not hasattr(self, "projector_intermediate_size"):
            vision_hidden = self.vision_config.hidden_size  # 1152
            self.projector_intermediate_size = (
                vision_hidden * (self.pixel_shuffle_scale**2) * 4
            )  # 18432

        # Validation
        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Isaac does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Isaac does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Isaac does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Isaac does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Isaac does not yet support fused speculation.")

        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads = self.text_config.num_attention_heads
            num_kv_heads = self.text_config.num_key_value_heads
            num_attn_heads = (
                num_attn_heads // self.neuron_config.tp_degree + 1
            ) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.head_dim",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "vision_config.hidden_size",
            "vision_config.image_size",
            "vision_config.num_attention_heads",
            "vision_config.num_hidden_layers",
            "vision_config.patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronIsaacForConditionalGeneration(NeuronBaseForImageToText):
    """Isaac VLM orchestrator for NxDI.

    Combines:
    - NeuronIsaacVisionModel (SigLIP2 + pixel shuffle + MLP projector)
    - NeuronIsaacTextModel (Qwen3 decoder)
    - ImageToTextModelWrapper (text model tracing wrapper)
    - IsaacVisionModelWrapper (vision model tracing wrapper)
    """

    # Model classes
    text_model_cls = NeuronIsaacTextModel
    vision_model_cls = NeuronIsaacVisionModel

    # Model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = IsaacVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls):
        return IsaacInferenceConfig

    def enable_vision_encoder(
        self, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        """Enable and configure the vision encoder for compilation."""
        self.compile_tag = VISION_ENCODER_MODEL_TAG

        new_config = copy.deepcopy(self.config)
        if new_config.vision_config.neuron_config.enable_bucketing:
            if (
                new_config.vision_config.neuron_config.buckets
                == [new_config.vision_config.neuron_config.seq_len]
                or new_config.vision_config.neuron_config.buckets is None
            ):
                if new_config.vision_config.neuron_config.seq_len > 1024:
                    new_config.vision_config.neuron_config.buckets = (
                        autobucketing.generate_buckets(
                            1024, new_config.vision_config.neuron_config.seq_len
                        )
                    )
                else:
                    new_config.vision_config.neuron_config.buckets = [
                        new_config.vision_config.neuron_config.seq_len
                    ]

        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True,
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: StateDict) -> None:
        """Isaac ties embed_tokens and lm_head weights."""
        try:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        except KeyError:
            state_dict["embed_tokens.weight"] = state_dict["lm_head.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: StateDict, inference_config: InferenceConfig
    ) -> StateDict:
        """Convert HuggingFace Isaac state dict to NxDI format.

        NOTE: The base class ApplicationBase.get_state_dict strips the leading
        "model." prefix BEFORE calling this method. So incoming keys are:
        - text_model.embed_tokens.weight (was model.text_model.embed_tokens.weight)
        - text_model.layers.{i}.* (was model.text_model.layers.{i}.*)
        - text_model.norm.weight (was model.text_model.norm.weight)
        - lm_head.weight (unchanged)
        - vision_embedding.0.* (was model.vision_embedding.0.*)
        - vision_embedding.1.weight (was model.vision_embedding.1.weight)
        - vision_embedding.3.weight (was model.vision_embedding.3.weight)
        - rotary_emb.* (was model.rotary_emb.*)

        Key mappings applied here:
        - text_model.* -> * (strip text_model prefix)
        - vision_embedding.0.* -> vision_encoder.vision_encoder.vision_model.*
        - vision_embedding.1.weight -> vision_encoder.multi_modal_projector.fc1.weight
        - vision_embedding.3.weight -> vision_encoder.multi_modal_projector.fc2.weight
        - rotary_emb.* -> skipped

        Also renames attention keys for NxDI format:
        - .self_attn.q_proj. -> .self_attn.qkv_proj.q_proj.
        - .self_attn.k_proj. -> .self_attn.qkv_proj.k_proj.
        - .self_attn.v_proj. -> .self_attn.qkv_proj.v_proj.
        - .self_attn.o_proj. -> .self_attn.o_proj.o_proj.
        - .self_attn.q_norm. -> .self_attn.q_layernorm.
        - .self_attn.k_norm. -> .self_attn.k_layernorm.
        """
        neuron_config = inference_config.neuron_config

        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
            ".self_attn.out_proj.": ".self_attn.o_proj.o_proj.",  # for siglip
            ".self_attn.q_norm.": ".self_attn.q_layernorm.",
            ".self_attn.k_norm.": ".self_attn.k_layernorm.",
        }

        new_state_dict = {}
        for key, weights in state_dict.items():
            new_key = key

            # Text model weights: text_model.* -> *
            # (base class already stripped leading "model." prefix)
            if new_key.startswith("text_model."):
                new_key = new_key.replace("text_model.", "", 1)
                # Rename attention keys
                for attn_key, replacement in attention_keys.items():
                    if attn_key in new_key:
                        new_key = new_key.replace(attn_key, replacement)
                        break

            # LM head: lm_head.weight -> lm_head.weight (no change)
            # (already handled by tied weights)

            # Vision encoder: vision_embedding.0.* -> vision_encoder.vision_model.*
            # NeuronIsaacVisionModel.vision_encoder = NeuronSiglipVisionModel
            # NeuronSiglipVisionModel.vision_model = NeuronSiglipVisionTransformer
            elif new_key.startswith("vision_embedding.0."):
                new_key = new_key.replace(
                    "vision_embedding.0.",
                    "vision_encoder.vision_model.",
                    1,
                )
                # Rename attention keys for vision encoder
                for attn_key, replacement in attention_keys.items():
                    if attn_key in new_key:
                        new_key = new_key.replace(attn_key, replacement)
                        break

            # MLP projector fc1: vision_embedding.1.weight
            elif new_key == "vision_embedding.1.weight":
                new_key = "multi_modal_projector.fc1.weight"

            # MLP projector fc2: vision_embedding.3.weight
            elif new_key == "vision_embedding.3.weight":
                new_key = "multi_modal_projector.fc2.weight"

            # Skip rotary_emb (handled by NxDI internally)
            elif new_key.startswith("rotary_emb"):
                continue

            new_state_dict[new_key] = weights

        # Reshape patch_embedding weight from HF 2D [out_ch, in_ch*kH*kW] to Conv2d 4D
        patch_key = "vision_encoder.vision_model.embeddings.patch_embedding.weight"
        if patch_key in new_state_dict:
            w = new_state_dict[patch_key]
            if w.dim() == 2:
                patch_size = inference_config.vision_config.patch_size
                num_channels = inference_config.vision_config.num_channels
                out_channels = w.shape[0]
                new_state_dict[patch_key] = w.reshape(
                    out_channels, num_channels, patch_size, patch_size
                )

        # Add lm_head.bias if needed for LNC > 1
        if (
            "lm_head.bias" not in new_state_dict
            and inference_config.neuron_config.lm_head_pad
        ):
            new_state_dict["lm_head.bias"] = torch.zeros(
                new_state_dict["embed_tokens.weight"].shape[0],
                dtype=torch.float32,
            )

        # Fuse QKV for text model
        if inference_config.text_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict,
                num_layers=inference_config.text_config.num_hidden_layers,
                neuron_config=inference_config.text_config.neuron_config,
                prefix="layers.{layer_num}.self_attn",
            )

        # Fuse QKV for vision model
        if inference_config.vision_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict,
                num_layers=inference_config.vision_config.num_hidden_layers,
                neuron_config=inference_config.vision_config.neuron_config,
                prefix="vision_encoder.vision_model.encoder.layers.{layer_num}.self_attn",
            )

        # Add rank utilities
        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        tp_degree = neuron_config.tp_degree
        for i in range(inference_config.text_config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return new_state_dict

    @staticmethod
    def _convert_input_dict_to_ordered_tuple(input_dict: Dict[str, Any]):
        """Convert input dictionary to ordered tuple for model wrapper."""
        args = []
        for key in IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS:
            if key in input_dict and input_dict[key] is not None:
                arg = input_dict[key]
            else:
                arg = torch.empty(0)
            args.append(arg)
        return tuple(args)

    def _select_buckets_for_padding_length(self, position_ids):
        """Select appropriate buckets based on whether prefill or decode."""
        neuron_config = self.config.neuron_config
        context_encoding_buckets = (
            neuron_config.context_encoding_buckets
            if neuron_config.context_encoding_buckets is not None
            else neuron_config.buckets
        )
        token_generation_buckets = (
            neuron_config.token_generation_buckets
            if neuron_config.token_generation_buckets is not None
            else neuron_config.buckets
        )

        if self._is_prefill(position_ids):
            return context_encoding_buckets
        return token_generation_buckets

    @staticmethod
    def get_padding_length(buckets, position_ids):
        """Find the smallest bucket that fits the input."""
        max_position_id = torch.max(position_ids).item()
        for val in buckets:
            if val > max_position_id:
                return val
        raise ValueError("No bucket found for provided input_ids!")

    @staticmethod
    def get_required_kwargs() -> List[str]:
        """Additional kwargs for HuggingFaceGenerationAdapter."""
        return [
            "pixel_values",
            "vision_mask",
        ]

    @staticmethod
    def generate_positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """Generate position indices from a boolean vision mask."""
        if mask.dim() == 1:
            return torch.nonzero(mask).squeeze()
        else:
            rows, cols = torch.nonzero(mask, as_tuple=True)
            row_counts = torch.bincount(rows, minlength=mask.shape[0])
            cols_per_row = torch.split(cols, row_counts.tolist())
            return rnn_utils.pad_sequence(
                cols_per_row, batch_first=True, padding_value=0
            )

    @staticmethod
    def pad_positions(
        positions: torch.LongTensor, target_size: int, fill_value: float
    ) -> torch.LongTensor:
        """Pad positions tensor to target size."""
        positions_2d = positions.unsqueeze(0) if positions.dim() == 1 else positions
        padding_size = target_size - positions_2d.shape[1]
        assert padding_size >= 0, (
            "Text model sequence length is not enough to handle all vision embeddings"
        )
        positions_padded = F.pad(positions_2d, (0, padding_size), value=fill_value)
        return positions_padded.unsqueeze(-1)

    @staticmethod
    def _create_position_ids(
        attention_mask_2d: torch.LongTensor, is_prefill: bool
    ) -> torch.LongTensor:
        """Create position IDs from attention mask."""
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)
        if is_prefill:
            return position_ids
        else:
            return torch.amax(position_ids, dim=1, keepdim=True) + 1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass combining vision encoder and text decoder."""
        is_prefill = input_ids.shape[-1] > 1
        include_images = (
            pixel_values is not None
            and vision_mask is not None
            and pixel_values.sum() != 0
        )

        if position_ids is None:
            position_ids = self._create_position_ids(
                attention_mask_2d=attention_mask, is_prefill=is_prefill
            )

        buckets = self._select_buckets_for_padding_length(position_ids=position_ids)
        pad_target_size = self.get_padding_length(
            buckets=buckets, position_ids=position_ids
        )
        pad_fill_value = pad_target_size - 1

        if is_prefill and include_images:
            assert vision_mask.dtype == torch.bool, (
                f"vision_mask must be bool, got {vision_mask.dtype}"
            )

            # Run vision encoder
            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
            ).to(self.text_config.neuron_config.torch_dtype)

            # Flatten vision embeddings for multi-image support
            batch_sz = 1 if vision_mask.dim() == 1 else vision_mask.shape[0]
            num_images, seq_len, embedding_dim = vision_embeddings.shape
            img_per_sample = num_images // batch_sz
            vision_embeddings = vision_embeddings.view(
                batch_sz, img_per_sample * seq_len, embedding_dim
            )

            # Pad to bucket size
            vision_embeddings = pad_vision_embeddings(
                vision_embeddings=vision_embeddings, pad_limit=pad_target_size
            )

            # Create scatter positions from vision mask
            vision_mask = self.generate_positions_from_mask(mask=vision_mask.squeeze())
            vision_mask = self.pad_positions(
                positions=vision_mask,
                target_size=pad_target_size,
                fill_value=pad_fill_value,
            )
        else:
            # Text-only or token generation -> dummy vision inputs
            vision_embeddings, vision_mask = (
                self.context_encoding_model.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_target_size,
                    fill_value=pad_fill_value,
                )
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def get_compiler_args(self) -> str:
        """Get compiler arguments based on compilation phase."""
        logical_nc_config = self.text_config.neuron_config.logical_nc_config

        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O2"
        elif self.compile_tag == VISION_ENCODER_MODEL_TAG:
            return (
                f"-O1 --model-type=transformer "
                f"--tensorizer-options='--enable-ccop-compute-overlap' "
                f"--auto-cast=none --lnc={logical_nc_config}"
            )
        else:
            raise ValueError(
                f"get_compiler_args() Invalid compile tag: {self.compile_tag}"
            )

        args = (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor=1 --vectorize-strided-dma "
            f"--enable-scalar-dge-vectorization' "
            f"--lnc={logical_nc_config} {optimization_level} "
        )
        return args

    def _get_constructed_outputs(self, outputs, is_run_on_neuron):
        """Process model outputs into the expected format."""
        if (
            self.on_device_sampling
            and self.text_config.neuron_config.output_logits
            and not (
                self.text_config.neuron_config.enable_fused_speculation
                or self.text_config.neuron_config.is_medusa
            )
        ):
            logits_or_next_tokens = outputs[:2]
            constructed_outputs = self._construct_output_with_tokens_and_logits(
                next_tokens=logits_or_next_tokens[0],
                logits=logits_or_next_tokens[1],
            )
        else:
            if is_run_on_neuron:
                logits_or_next_tokens = (
                    outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                )
            else:
                logits_or_next_tokens, *_ = outputs
            constructed_outputs = self._construct_output(logits_or_next_tokens)

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug("---output---")
            logging.debug(
                f"{'tokens' if self.on_device_sampling else 'logits'} = %s",
                logits_or_next_tokens,
            )

        return constructed_outputs

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the HuggingFace Isaac model for weight extraction."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        ).eval()
        return model
