import copy
import math
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

import torch
import torch.nn.utils.rnn as rnn_utils
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed.parallel_layers.parallel_state import (
    destroy_model_parallel,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from neuronx_distributed.quantization.quantization_utils import convert_qint8_to_int8_state_dict
from neuronx_distributed.trace.trace import get_sharded_checkpoint

import neuronx_distributed_inference.modules.autobucketing as autobucketing
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig, 
    NeuronBaseForImageToText
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper, 
    IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import pad_vision_embeddings
from neuronx_distributed_inference.models.pixtral.modeling_pixtral import NeuronPixtralForCausalLM
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    VISION_ENCODER_MODEL_TAG
)
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.models.application_base import (
    COMPILED_MODEL_FILE_NAME,
    normalize_path,
)

from gemma3_vision.modeling_gemma3_text import NeuronGemma3TextModel
from gemma3_vision.modeling_gemma3_vision import NeuronGemma3VisionModel, Gemma3VisionModelWrapper
from gemma3_vision.utils import convert_state_dict_to_fused_qkv, StateDict

logger = logging.getLogger("Neuron")


class Gemma3InferenceConfig(ImageToTextInferenceConfig):
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

        # NeuronLlamaMLP expects the activation type to be at text_config.hidden_act
        # Enable to fully reuse NeuronLlamaMLP
        if not hasattr(self.text_config, "hidden_act"):
            self.text_config.hidden_act = self.text_config.hidden_activation
            del self.text_config.hidden_activation
        
        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Gemma3 does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Gemma3 does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Gemma3 does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Gemma3 does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Gemma3 does not yet support fused speculation.")

        if self.neuron_config.flash_decoding_enabled:
            # Following pixtral implementation, we use REPLICATE_TO_TP_DEGREE as the sharding_strategy
            # Hence attn_heads are padded to become divisible by tp_degree
            num_attn_heads, num_kv_heads = self.text_config.num_attention_heads, self.text_config.num_key_value_heads
            num_attn_heads = (num_attn_heads // self.neuron_config.tp_degree + 1) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.head_dim", # for gemma3, head_dim != hidden_size // num_attention_heads
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.query_pre_attn_scalar",
            "text_config.rope_scaling",
            "text_config.sliding_window",
            "vision_config.hidden_size",
            "vision_config.image_size",
            "vision_config.num_attention_heads",
            "vision_config.num_hidden_layers",
            "vision_config.patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronGemma3ForCausalLM(NeuronBaseForImageToText):
    # model cls
    text_model_cls = NeuronGemma3TextModel
    vision_model_cls = NeuronGemma3VisionModel

    # model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = Gemma3VisionModelWrapper

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
        return Gemma3InferenceConfig

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return f"--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return f"--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        if new_config.vision_config.neuron_config.enable_bucketing:
            # neuron_config.buckets default to neuron_config.seq_len is not given. For vision we want to do auto-bucketing here
            if new_config.vision_config.neuron_config.buckets == [new_config.vision_config.neuron_config.seq_len] or \
                    new_config.vision_config.neuron_config.buckets is None:
                # 1024 vision seq len corresponds to a single 512x512 image. Smaller bucket size does not make sense in real life.
                if new_config.vision_config.neuron_config.seq_len > 1024:
                    new_config.vision_config.neuron_config.buckets = autobucketing.generate_buckets(
                        1024, new_config.vision_config.neuron_config.seq_len
                    )
                else:
                    new_config.vision_config.neuron_config.buckets = [new_config.vision_config.neuron_config.seq_len]
        # This should not be needed as in vision modeling code we should always use vision_config.neuron_config as vision model's neuron config
        # added this line just to add insurance to avoid mix-up
        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=False, # TODO: True for opimization?
            return_ranked_to_cpu=True
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: StateDict) -> None:
        try: 
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        except KeyError:
            state_dict["embed_tokens.weight"] = state_dict["lm_head.weight"].clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: StateDict, inference_config: InferenceConfig) -> StateDict:
        neuron_config = inference_config.neuron_config
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
            ".self_attn.out_proj.": ".self_attn.o_proj.o_proj.", # for siglip
            ".self_attn.q_norm.": ".self_attn.q_layernorm.",
            ".self_attn.k_norm.": ".self_attn.k_layernorm.",
        }

        # At the time of writing, NxDI (Neuron 2.26) attention layer does not provide a simple way to use a custom 
        # scaling factor for raw attention scores (QK^T) while ensuring all optimizations (e.g. kernels) remain available 
        # To work around this, we fuse the scaling factor into the weights (knowing that  the attention layer will use the 
        # default math.sqrt(inference_config.head_dim) value)
        default_qk_scaling_factor_inv = math.sqrt(float(inference_config.text_config.query_pre_attn_scalar))
        gemma_qk_scaling_factor = 1.0 / math.sqrt(float(inference_config.text_config.head_dim))
        gamma = math.sqrt(gemma_qk_scaling_factor * default_qk_scaling_factor_inv)

        new_state_dict = {}
        for key, weights in state_dict.items():
            if 'language_model.model.' in key:
                key = key.replace('language_model.model.', "")
                for atten_key in attention_keys:
                    if atten_key in key:
                        replacement_atten_key = attention_keys[atten_key]
                        key = key.replace(atten_key, replacement_atten_key)
                        break
                if key.endswith((".q_proj.weight", ".k_proj.weight")):
                    orig_dtype = weights.dtype
                    weights = (weights.to(dtype=torch.float32) * gamma).to(dtype=orig_dtype)  
            if 'language_model.lm_head.' in key:
                key = key.replace('language_model.', "")
            if 'vision_tower.' in key:
                key = key.replace('vision_tower.', 'vision_encoder.')
                for atten_key in attention_keys:
                    if atten_key in key:
                        replacement_atten_key = attention_keys[atten_key]
                        key = key.replace(atten_key, replacement_atten_key)
                        break
            new_state_dict[key] = weights

        # If LNC > 1, model requires lm_head.bias which is equivalent to lm_head_pad
        if "language_model.lm_head.bias" not in state_dict and inference_config.neuron_config.lm_head_pad:
            # Use embed_tokens.weight instead of lm_head.weight as lm_head.weight is tied to embed_tokens.weight in Gemma3
            new_state_dict["lm_head.bias"] = torch.zeros(new_state_dict["embed_tokens.weight"].shape[0], dtype=torch.float32)
            
        if inference_config.text_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict, 
                num_layers=inference_config.text_config.num_hidden_layers,
                neuron_config=inference_config.text_config.neuron_config,
                prefix="layers.{layer_num}.self_attn"
                )
            
        if inference_config.vision_config.neuron_config.fused_qkv:
            new_state_dict = convert_state_dict_to_fused_qkv(
                state_dict=new_state_dict, 
                num_layers=inference_config.vision_config.num_hidden_layers,
                neuron_config=inference_config.vision_config.neuron_config,
                prefix="vision_encoder.vision_model.encoder.layers.{layer_num}.self_attn"
                )
            
        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        tp_degree = neuron_config.tp_degree
        for i in range(inference_config.text_config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return new_state_dict

    def _convert_input_dict_to_ordered_tuple(self, input_dict: Dict[str, Any]):
        """
        Utility function to convert input dictionary to ordered tuple
        based on outputs of _get_model_outputs
        """
        args = []

        for key in IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS:
            if key in input_dict and input_dict[key] is not None:
                arg = input_dict[key]
            else:
                arg = torch.empty(0)
            args.append(arg)

        return tuple(args)
    
    def _select_buckets_for_padding_length(self, position_ids):
        neuron_config = self.config.neuron_config
        context_encoding_buckets = neuron_config.context_encoding_buckets if neuron_config.context_encoding_buckets is not None \
            else neuron_config.buckets
        token_generation_buckets = neuron_config.token_generation_buckets if neuron_config.token_generation_buckets is not None \
            else neuron_config.buckets

        selected_buckets = token_generation_buckets
        if self._is_prefill(position_ids):
            selected_buckets = context_encoding_buckets

        return selected_buckets
    
    def get_padding_length(self, buckets, position_ids):
        max_position_id = torch.max(position_ids).item()
        for val in buckets:
            if val > max_position_id:
                return val
        raise ValueError("No bucket found for provided input_ids!")

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
            "image_sizes",
        ]

    def concat_causal_lm_outputs(self, outputs_list):
        concatenated_logits = []
        concatenated_hidden_states = []
        concatenated_tokens = []
        for output in outputs_list:
            if isinstance(output.logits, torch.Tensor):
                concatenated_logits.append(output.logits)
            if isinstance(output.hidden_states, torch.Tensor):
                concatenated_hidden_states.append(output.hidden_states)
            elif isinstance(output.hidden_states, list):
                concatenated_hidden_states.extend(output.hidden_states)
            if hasattr(output, 'tokens') and isinstance(output.tokens, torch.Tensor):
                concatenated_tokens.append(output.tokens)
        concatenated_logits = torch.cat(concatenated_logits, dim=0) if len(concatenated_logits) > 0 else None
        concatenated_tokens = torch.cat(concatenated_tokens, dim=0) if len(concatenated_tokens) else None

        concatentated_output = CausalLMOutputWithPast(
            logits=concatenated_logits,
            hidden_states=concatenated_hidden_states,
        )
        if concatenated_tokens is not None:
            concatentated_output.tokens = concatenated_tokens
        return concatentated_output

    def generate_positions_from_mask(self, mask):
        """
        Generate position indices from a boolean mask.
        Compared to generate_positions_from_mask() of models/llama4/utils/encoder_utils.py,
        this function can generate 1D or 2D masks to support batch size > 1.

        Args:
        mask (torch.Tensor): A 1D or 2D boolean tensor

        Returns:
        torch.Tensor: A 1D or 2D tensor containing the indices where the mask is True
        """
        if mask.dim() == 1:
            return torch.nonzero(mask).squeeze()
        else:
            rows, cols = torch.nonzero(mask, as_tuple=True)
            row_counts = torch.bincount(rows, minlength=mask.shape[0])
            cols_per_row = torch.split(cols, row_counts.tolist())
            return rnn_utils.pad_sequence(cols_per_row, batch_first=True, padding_value=0)

    def pad_positions(self, positions, target_size, fill_value):
        """
        Pad the positions tensor to a target size.
        Compared to pad_positions() of models/llama4/utils/encoder_utils.py,
        this function can support batch size > 1.
        
        Args:
        positions (torch.Tensor): A 1D or 2D tensor containing position indices
        target_size (int): The desired size of the padded tensor
        fill_value (int): The value used for padding

        Returns:
        torch.Tensor: A 3D tensor of shape (batch_size, target_size, 1) containing padded position indices
        """
        if positions.dim() == 1:
            # Handle 1D case (original behavior)
            padding_size = target_size - len(positions)
            if padding_size > 0:
                padding = torch.full(
                    (padding_size,), fill_value, dtype=positions.dtype, device=positions.device
                )
                positions_padded = torch.cat([positions, padding])
            elif padding_size < 0:
                raise RuntimeError("Text model sequence length is not enough to handle all vision embeddings")
            return positions_padded.unsqueeze(0).unsqueeze(-1)  # Shape: [1, x, 1]
        else:
            # Handle 2D case [batch_size, position_indices]
            padding_size = target_size - positions.shape[1]
            if padding_size > 0:
                padding = torch.full(
                    (positions.shape[0], padding_size), fill_value, dtype=positions.dtype, device=positions.device
                )
                positions_padded = torch.cat([positions, padding], dim=1)
            elif padding_size < 0:
                raise RuntimeError("Text model sequence length is not enough to handle all vision embeddings")
            return positions_padded.unsqueeze(-1)  # Shape: [batch_size, target_size, 1]

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
        buckets = self._select_buckets_for_padding_length(position_ids)
        pad_limit = self.get_padding_length(buckets, position_ids)
        if (
            (pixel_values is not None)
            and (vision_mask is not None)
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):  # call vision encoder
            assert (
                vision_mask.dtype == torch.bool
            ), f"Parameter `vision_mask` must be of type bool, recieved {vision_mask.dtype}"

            logger.info("pixel_values provided, using vision embeddings")

            vision_mask = self.generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = self.pad_positions(
                vision_mask, pad_limit, (pad_limit - 1) # pad_limit = 512
            )

            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
            ).to(self.text_config.neuron_config.torch_dtype)

            # flatten vision embeddings
            # embedding_dim = vision_embeddings.shape[-1]
            # vision_embeddings = vision_embeddings.view(-1, embedding_dim).unsqueeze(0)

            vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)
        else:
            vision_embeddings, vision_mask = self.context_encoding_model.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1)
            )

        output_token = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )
        return output_token

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma3ForConditionalGeneration
        return Gemma3ForConditionalGeneration.from_pretrained(model_path, **kwargs)  # nosec B615

    def to_cpu(self):
        """
        Initialize CPU versions of both text and vision models with different parallelism configurations,
        shard and load their weights, and assign to respective model wrappers.
        This function as of now only supports TP DEGREE of 1 in vision and text.
        """
        os.environ["NXD_CPU_MODE"] = "1"

        # Validation checks
        if self.neuron_config.torch_dtype == torch.bfloat16 and (
            self.neuron_config.tp_degree > 1 or self.neuron_config.ve_tp_degree > 1
        ):
            raise NotImplementedError(
                "The gloo backend does not natively support bfloat16, please proceed with float32 dtype instead."
            )
        if self.neuron_config.speculation_length > 0:
            raise NotImplementedError("Speculation is not yet supported for CPU inference.")

        # destroy distributed process if already started
        if model_parallel_is_initialized():
            destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Initialize distributed processing
        if "WORLD_SIZE" in os.environ:
            assert (
                int(os.environ["WORLD_SIZE"]) == self.neuron_config.world_size
            ), "Total number of processes does not match implied world size from NeuronConfig inputs."
            torch.distributed.init_process_group("gloo")
        if not torch.distributed.is_initialized():
            if self.neuron_config.world_size == 1:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
                torch.distributed.init_process_group(
                    backend="gloo",
                    world_size=1,
                    rank=0,
                )
            else:
                raise RuntimeError("Please initialize parallel processing via 'torchrun'.")

        # Initialize model parallel for vision and text model. We only support TP Degree 1 at this point.
        initialize_model_parallel(
            tensor_model_parallel_size=self.neuron_config.tp_degree,
            pipeline_model_parallel_size=1,  # No pipeline parallelism for vision encoder
            expert_model_parallel_size=1,  # No expert parallelism for vision encoder
            skip_collective_init=True,
        )

        # Initialize and load vision model with vision-specific config
        vision_base_model = self.vision_model_cls(self.config)
        vision_base_model = vision_base_model.to(
            self.vision_config.neuron_config.torch_dtype
        )

        vision_model_sd = (
            self.checkpoint_loader_fn()
        )  # You might need a separate loader for vision weights
        if self.vision_config.neuron_config.tp_degree > 1:
            get_sharded_checkpoint(
                vision_model_sd,
                vision_base_model,
                torch.distributed.get_rank(),
                self.vision_config.neuron_config.tp_degree,
            )

        vision_base_model.load_state_dict(vision_model_sd, strict=False)

        # Initialize and load text model with text-specific config
        text_base_model = self.text_model_cls(self.config.text_config)
        text_base_model = text_base_model.to(self.config.text_config.neuron_config.torch_dtype)

        text_model_sd = self.checkpoint_loader_fn()
        if self.neuron_config.tp_degree > 1:
            get_sharded_checkpoint(
                text_model_sd,
                text_base_model,
                torch.distributed.get_rank(),
                self.neuron_config.tp_degree,
            )
        text_base_model.load_state_dict(text_model_sd, strict=False)

        # Assign models to their respective wrappers
        for model_wrapper in self.text_models:
            model_wrapper.model = text_base_model

        for model_wrapper in self.vision_models:
            model_wrapper.model = vision_base_model

        self.eval()

    # Wraps NeuronBaseForCausalLM.enable_context_encoding() to add compile_tag.
    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    # Wraps NeuronBaseForCausalLM.enable_token_generation() to add compile_tag.
    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self) -> str:
        logical_nc_config = self.text_config.neuron_config.logical_nc_config

        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O2"
        elif self.compile_tag == VISION_ENCODER_MODEL_TAG:
            return f"-O1 --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap' " \
                   f"--auto-cast=none --lnc={logical_nc_config}"
        else:
            raise ValueError(f"get_compiler_args() Invalid compile tag encountered: {self.compile_tag}")

        args = f"--auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap " \
               f"--cc-pipeline-tiling-factor=1 --vectorize-strided-dma --enable-scalar-dge-vectorization' " \
               f"--lnc={logical_nc_config} {optimization_level} "
        return args

    def load(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False
    ):
        # Fixed broken path creation (Neuron 2.26)
        compiled_model_path = normalize_path(compiled_model_path)
        text_compiled_model_path = normalize_path(compiled_model_path) + "text_model/"
        vision_compiled_model_path = normalize_path(compiled_model_path) + "vision_model/"

        """Loads the compiled model checkpoint to the Neuron device."""
        self.text_traced_model = torch.jit.load(text_compiled_model_path + COMPILED_MODEL_FILE_NAME)  # nosec B614
        self.vision_traced_model = torch.jit.load(  # nosec B614
            vision_compiled_model_path + COMPILED_MODEL_FILE_NAME
        )

        self.load_weights(
            text_compiled_model_path,
            vision_compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
        )

        for model_wrapper in self.text_models:
            model_wrapper.model = self.text_traced_model

        for model_wrapper in self.vision_models:
            model_wrapper.model = self.vision_traced_model

        self.is_loaded_to_neuron = True

        if not self.neuron_config.skip_warmup and not skip_warmup:
            self.warmup()  # warmup will be executed only if both flags are false
        else:
            logger.info("Skipping model warmup")

    @classmethod
    def prepare_quantized_state_dict(cls, hf_model_quant):
        # Default assumes text-only model structure and breaks (AttributeError on hf_model_quant.model.state_dict())
        model_quant_sd = hf_model_quant.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        return model_quant_sd
