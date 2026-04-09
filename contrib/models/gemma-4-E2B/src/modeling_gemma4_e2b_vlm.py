# coding=utf-8
# Copyright 2026 Google Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemma4 E2B VLM (Vision-Language Model) for NeuronX Distributed Inference

Top-level VLM orchestrator that compiles and runs separate vision and text NEFFs.
- Vision encoder: NeuronGemma4VisionModel (modeling_gemma4_vision.py) — reused from 31B
- Text decoder: NeuronGemma4E2BTextModel (modeling_gemma4_e2b.py) — E2B with PLE + KV sharing
- State dict conversion handles both text and vision weights
"""

from ndxi_patch import apply_patch

apply_patch()

import copy  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.nn.utils.rnn as rnn_utils  # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.image_to_text_model_base import (  # noqa: E402
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (  # noqa: E402
    ImageToTextModelWrapper,
    IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS,
)
from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import (  # noqa: E402
    Llama4VisionModelWrapper,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (  # noqa: E402
    pad_vision_embeddings,
)
from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    VISION_ENCODER_MODEL_TAG,
)

from modeling_gemma4_e2b import (  # noqa: E402
    NeuronGemma4E2BTextModel,
    Gemma4E2BInferenceConfig,
    Gemma4E2BNeuronConfig,
)
from modeling_gemma4_vision import NeuronGemma4VisionModel  # noqa: E402

logger = logging.getLogger("Neuron")


# ====================================================================================
# Config loader
# ====================================================================================

import os  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from transformers import AutoConfig, PretrainedConfig  # noqa: E402
from neuronx_distributed_inference.models.config import to_torch_dtype  # noqa: E402


def load_pretrained_config(model_path):
    """
    Return a load_config callable for InferenceConfig.__init__.

    Falls back to manual JSON loading if AutoConfig doesn't recognize
    the model_type (e.g. 'gemma4' on older transformers versions).
    """
    import json as _json

    def load_config(self):
        config = None
        config_dict = None

        try:
            config = AutoConfig.from_pretrained(model_path)
            config_dict = config.to_dict()
        except (ValueError, KeyError):
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = _json.load(f)

        transformers_version = None
        if config is not None and hasattr(config, "transformers_version"):
            transformers_version = config.transformers_version
        elif "transformers_version" in config_dict:
            transformers_version = config_dict["transformers_version"]

        hf_dtype = config_dict.get("dtype", config_dict.get("torch_dtype", None))
        if hf_dtype is not None:
            if (
                self.neuron_config is not None
                and not self.neuron_config.overrides_torch_dtype
            ):
                self.neuron_config.torch_dtype = hf_dtype
                if isinstance(self.neuron_config.torch_dtype, str):
                    self.neuron_config.torch_dtype = to_torch_dtype(
                        self.neuron_config.torch_dtype
                    )
            config_dict.pop("dtype", None)
            config_dict.pop("torch_dtype", None)

        for k, v in config_dict.items():
            if isinstance(v, dict):
                config_dict[k] = SimpleNamespace(**v)
                if transformers_version is not None:
                    config_dict[k].transformers_version = transformers_version
            elif config is not None and isinstance(
                getattr(config, k, None), PretrainedConfig
            ):
                config_dict[k] = SimpleNamespace(**v)
                if transformers_version is not None:
                    config_dict[k].transformers_version = transformers_version

        self.__dict__.update(config_dict)

        if not hasattr(self, "_name_or_path"):
            self._name_or_path = str(model_path)

        if config is not None and hasattr(config, "attribute_map"):
            self.attribute_map = config.attribute_map

    return load_config


# ====================================================================================
# VLM Inference Config
# ====================================================================================


class Gemma4E2BVLMInferenceConfig(ImageToTextInferenceConfig):
    """Configuration for Gemma4 E2B VLM (text + vision)."""

    def __init__(
        self,
        text_neuron_config: NeuronConfig,
        vision_neuron_config: NeuronConfig,
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

        # ---- E2B-specific text_config enrichment ----
        # The VLM base class creates text_config from the HF config.json's
        # "text_config" sub-dict. The E2B text decoder expects derived attributes
        # that Gemma4E2BInferenceConfig.__init__ normally computes. We replicate
        # that logic here so text_config has everything get_updated_configs() needs.

        tc = self.text_config

        # Gemma4 uses hidden_activation; NeuronLlamaMLP expects hidden_act
        if hasattr(tc, "hidden_activation") and not hasattr(tc, "hidden_act"):
            tc.hidden_act = tc.hidden_activation
            del tc.hidden_activation

        # E2B defaults
        if not hasattr(tc, "hidden_size_per_layer_input"):
            tc.hidden_size_per_layer_input = 256
        if not hasattr(tc, "num_kv_shared_layers"):
            tc.num_kv_shared_layers = 20
        if not hasattr(tc, "global_head_dim"):
            tc.global_head_dim = 512
        if not hasattr(tc, "vocab_size_per_layer_input"):
            tc.vocab_size_per_layer_input = getattr(tc, "vocab_size", 262144)

        # E2B does not use attention_k_eq_v
        tc.attention_k_eq_v = False
        tc.num_global_key_value_heads = getattr(tc, "num_key_value_heads", 1)

        # Compute per-layer intermediate sizes (same logic as Gemma4E2BInferenceConfig)
        # Layers 0..(first_kv_shared-1) use base intermediate_size,
        # layers first_kv_shared..end use 2x intermediate_size.
        full_num_layers = 35  # E2B always has 35 layers
        first_kv_shared = full_num_layers - tc.num_kv_shared_layers  # 35 - 20 = 15
        base_intermediate = getattr(tc, "intermediate_size", 6144)
        num_hidden_layers = getattr(tc, "num_hidden_layers", 35)
        tc.layer_intermediate_sizes = []
        for i in range(num_hidden_layers):
            if i < first_kv_shared:
                tc.layer_intermediate_sizes.append(base_intermediate)
            else:
                tc.layer_intermediate_sizes.append(base_intermediate * 2)

        # Ensure rope_parameters is a plain dict (may be SimpleNamespace from config loading)
        if hasattr(tc, "rope_parameters"):
            rp = tc.rope_parameters
            if hasattr(rp, "__dict__") and not isinstance(rp, dict):
                # Convert SimpleNamespace tree to nested dict
                rp_dict = {}
                for k, v in vars(rp).items():
                    if hasattr(v, "__dict__") and not isinstance(v, dict):
                        rp_dict[k] = vars(v)
                    else:
                        rp_dict[k] = v
                tc.rope_parameters = rp_dict

        for attr, default in [
            ("output_attentions", False),
            ("output_hidden_states", False),
            ("use_return_dict", True),
            ("attention_bias", False),
        ]:
            if not hasattr(tc, attr):
                setattr(tc, attr, default)
            if not hasattr(self, attr):
                setattr(self, attr, default)

        if not hasattr(tc, "pad_token_id"):
            tc.pad_token_id = getattr(self, "pad_token_id", 0)
        if not hasattr(tc, "tie_word_embeddings"):
            tc.tie_word_embeddings = getattr(self, "tie_word_embeddings", True)

        # Validate unsupported features
        if tc.neuron_config.is_block_kv_layout:
            raise ValueError("Gemma4 E2B does not yet support block_kv_layout.")
        if tc.neuron_config.is_prefix_caching:
            raise ValueError("Gemma4 E2B does not yet support prefix_caching.")
        if tc.neuron_config.is_chunked_prefill:
            raise ValueError("Gemma4 E2B does not yet support chunked_prefill.")
        if tc.neuron_config.is_medusa:
            raise ValueError("Gemma4 E2B does not yet support medusa.")
        if tc.neuron_config.enable_fused_speculation:
            raise ValueError("Gemma4 E2B does not yet support fused speculation.")

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.head_dim",
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.sliding_window",
            "vision_config.hidden_size",
            "vision_config.num_attention_heads",
            "vision_config.num_hidden_layers",
            "vision_config.patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ====================================================================================
# Vision Model Wrapper (for tracing/compilation)
# ====================================================================================


class Gemma4E2BVisionModelWrapper(Llama4VisionModelWrapper):
    """
    Neuron ModelWrapper for Gemma4 E2B's vision encoder.

    Identical to the 31B wrapper — Gemma4 vision takes two inputs:
      1. pixel_values: [B, num_patches, 3*patch_size^2]
      2. pixel_position_ids: [B, num_patches, 2]
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = True,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate sample inputs for tracing the vision encoder."""
        vision_config = self.config.vision_config
        patch_size = vision_config.patch_size
        image_size = getattr(vision_config, "image_size", 384)
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side * num_patches_per_side
        batch_size = self.neuron_config.batch_size

        inputs = []
        for bucket in self.neuron_config.buckets:
            pixel_values = torch.ones(
                [batch_size, num_patches, 3 * patch_size * patch_size],
                dtype=self.config.neuron_config.torch_dtype,
            )
            position_ids = torch.zeros(batch_size, num_patches, 2, dtype=torch.long)
            for i in range(num_patches_per_side):
                for j in range(num_patches_per_side):
                    idx = i * num_patches_per_side + j
                    position_ids[:, idx, 0] = j
                    position_ids[:, idx, 1] = i
            inputs.append((pixel_values, position_ids))

        return inputs

    def forward(self, *args):
        """Override forward to handle two inputs (pixel_values, position_ids)."""
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() first."
            )

        pixel_values = args[0]
        input_batch_size = pixel_values.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
            output = self._forward(*args)
            return output

        cur_batch = 0
        outputs = []

        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                batch_args = [
                    arg[cur_batch : cur_batch + self.neuron_config.batch_size]
                    for arg in args
                ]
                output = self._forward(*batch_args)
            else:
                output = self._forward_with_pad(
                    *[arg[cur_batch:input_batch_size] for arg in args]
                )
            outputs.append(output)
            cur_batch += self.neuron_config.batch_size

        return output

    def _forward_with_pad(self, *args):
        """Pad inputs to compiled batch size."""
        pixel_values = args[0]
        orig_batch_size = pixel_values.shape[0]

        padded_args = []
        for arg in args:
            if arg.shape[0] == self.neuron_config.batch_size:
                padded_args.append(arg)
            else:
                padded_shape = list(arg.shape)
                padded_shape[0] = self.neuron_config.batch_size
                padded = arg[0].unsqueeze(0).expand(padded_shape).clone()
                padded[: arg.shape[0]] = arg
                padded_args.append(padded)

        outputs = self._forward(*padded_args)
        return outputs[:orig_batch_size]


# ====================================================================================
# Top-Level VLM Model
# ====================================================================================


class NeuronGemma4E2BForConditionalGeneration(NeuronBaseForImageToText):
    """
    Gemma4 E2B VLM: vision encoder + E2B text decoder (with PLE + KV sharing).

    Orchestrates compilation and inference of separate vision and text NEFFs.
    """

    text_model_cls = NeuronGemma4E2BTextModel
    vision_model_cls = NeuronGemma4VisionModel

    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = Gemma4E2BVisionModelWrapper

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
        return Gemma4E2BVLMInferenceConfig

    def enable_vision_encoder(
        self, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        """Configure and create the vision encoder model wrapper."""
        self.compile_tag = VISION_ENCODER_MODEL_TAG

        new_config = copy.deepcopy(self.config)
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

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def get_compiler_args(self) -> str:
        """Return compiler args based on current compilation target."""
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
            f"--tensorizer-options='--enable-ccop-compute-overlap' "
            f"--lnc={logical_nc_config} {optimization_level} "
        )
        return args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Handle tied weights: embed_tokens -> lm_head."""
        embed_key = None
        if "embed_tokens.embedding.weight" in state_dict:
            embed_key = "embed_tokens.embedding.weight"
        elif "embed_tokens.weight" in state_dict:
            embed_key = "embed_tokens.weight"

        if embed_key is not None:
            weight = state_dict[embed_key].clone()
            state_dict["lm_head.weight"] = weight
            state_dict["lm_head.linear.weight"] = weight.clone()

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        inference_config: InferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert HF Gemma4 E2B state dict to NeuronX format for both text and vision.

        Text transformations (E2B-specific):
        1. Strip 'language_model.model.' / 'model.language_model.' prefixes
        2. Remap embed_tokens -> embed_tokens.embedding
        3. Remap PLE keys into ple_module.* namespace
        4. Remap q_norm/k_norm -> q_layernorm/k_layernorm
        5. Fuse QK scaling into q_layernorm weights
        6. Cast PLE embedding to bf16
        7. Add rank_util tensors
        8. No attention_k_eq_v (E2B doesn't use it)

        Vision transformations:
        1. 'model.vision_tower.' -> 'vision_model.'
        2. 'model.embed_vision.' -> 'embed_vision.'
        3. Strip '.linear.weight' -> '.weight' for parallel layers
        4. Skip quantization metadata (input_max/min, output_max/min from use_clipped_linears)
        """
        neuron_config = inference_config.neuron_config
        tp_degree = neuron_config.tp_degree
        text_config = inference_config.text_config
        new_state_dict = {}

        for key, weight in state_dict.items():
            new_key = key

            # --- Skip quantization metadata from use_clipped_linears ---
            if any(
                suffix in new_key
                for suffix in [".input_max", ".input_min", ".output_max", ".output_min"]
            ):
                continue

            # --- VISION WEIGHTS ---
            if "vision_tower" in key or "embed_vision" in key:
                if "model.vision_tower." in new_key:
                    new_key = new_key.replace("model.vision_tower.", "vision_model.")
                elif "vision_tower." in new_key:
                    new_key = new_key.replace("vision_tower.", "vision_model.")

                if "model.embed_vision." in new_key:
                    new_key = new_key.replace("model.embed_vision.", "embed_vision.")

                # ColumnParallelLinear/RowParallelLinear: .linear.weight -> .weight
                new_key = new_key.replace(".linear.weight", ".weight")

                new_state_dict[new_key] = weight.detach().clone()
                continue

            # --- TEXT WEIGHTS ---
            if (
                "language_model" in new_key
                and "vision_tower" not in new_key
                and "embed_vision" not in new_key
            ):
                # Strip HF prefixes
                if new_key.startswith("model.language_model.model."):
                    new_key = new_key[len("model.language_model.model.") :]
                elif new_key.startswith("model.language_model."):
                    new_key = new_key[len("model.language_model.") :]
                elif new_key.startswith("language_model.model."):
                    new_key = new_key[len("language_model.model.") :]
                elif new_key.startswith("language_model."):
                    new_key = new_key[len("language_model.") :]
            else:
                # Handle bare keys without language_model prefix
                pass

            # Skip audio/vision keys that aren't already handled
            if any(
                skip in new_key
                for skip in [
                    "vision_tower.",
                    "multi_modal_projector.",
                    "embed_vision.",
                    "audio_tower.",
                    "embed_audio.",
                ]
            ):
                continue

            # Remap embedding key
            if new_key == "embed_tokens.weight":
                new_key = "embed_tokens.embedding.weight"

            # Remap PLE keys into ple_module
            if new_key == "embed_tokens_per_layer.weight":
                new_key = "ple_module.embed_tokens_per_layer.weight"
            if new_key == "per_layer_model_projection.weight":
                new_key = "ple_module.per_layer_model_projection.weight"
            if new_key == "per_layer_projection_norm.weight":
                new_key = "ple_module.per_layer_projection_norm.weight"

            # Remap QK norm keys
            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            new_state_dict[new_key] = weight.detach().clone()

        # --- Cast PLE embedding to bf16 ---
        ple_embed_key = "ple_module.embed_tokens_per_layer.weight"
        if ple_embed_key in new_state_dict:
            new_state_dict[ple_embed_key] = new_state_dict[ple_embed_key].to(
                torch.bfloat16
            )

        # --- PER-LAYER TEXT TRANSFORMATIONS ---
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            layer_types = getattr(inference_config, "layer_types", None)

        num_layers = text_config.num_hidden_layers
        for i in range(num_layers):
            layer_type = layer_types[i] if layer_types else "sliding_attention"
            is_global = layer_type == "full_attention"

            if is_global:
                hd = text_config.global_head_dim
            else:
                hd = text_config.head_dim

            prefix = f"layers.{i}.self_attn"

            # QK scaling: scale q_layernorm.weight by sqrt(head_dim)
            q_norm_key = f"{prefix}.q_layernorm.weight"
            if q_norm_key in new_state_dict:
                scaling_factor = math.sqrt(float(hd))
                orig_dtype = new_state_dict[q_norm_key].dtype
                new_state_dict[q_norm_key] = (
                    new_state_dict[q_norm_key].to(torch.float32) * scaling_factor
                ).to(orig_dtype)

            # rank_util for TP
            new_state_dict[f"{prefix}.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # Vocabulary parallelism rank
        if neuron_config.vocab_parallel:
            new_state_dict["embed_tokens.embedding.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # Base model rank
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

    @staticmethod
    def get_required_kwargs() -> List[str]:
        """Additional kwargs for HuggingFaceGenerationAdapter."""
        return [
            "pixel_values",
            "pixel_position_ids",
            "vision_mask",
        ]

    @staticmethod
    def generate_positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """Generate position indices from a boolean mask."""
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
        """Pad position tensor to target size."""
        positions_2d = positions.unsqueeze(0) if positions.dim() == 1 else positions
        padding_size = target_size - positions_2d.shape[1]
        assert padding_size >= 0, (
            "Text model sequence length not enough for vision embeddings"
        )
        positions_padded = F.pad(positions_2d, (0, padding_size), value=fill_value)
        return positions_padded.unsqueeze(-1)

    @staticmethod
    def _create_position_ids(
        attention_mask_2d: torch.LongTensor, is_prefill: bool
    ) -> torch.LongTensor:
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)
        if is_prefill:
            return position_ids
        else:
            return torch.amax(position_ids, dim=1, keepdim=True) + 1

    def _select_buckets_for_padding_length(self, position_ids):
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
        selected_buckets = token_generation_buckets
        if self._is_prefill(position_ids):
            selected_buckets = context_encoding_buckets
        return selected_buckets

    @staticmethod
    def get_padding_length(buckets, position_ids):
        max_position_id = torch.max(position_ids).item()
        for val in buckets:
            if val > max_position_id:
                return val
        raise ValueError("No bucket found for provided input_ids!")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for E2B VLM inference.

        During prefill with images:
        1. Run vision encoder on pixel_values + pixel_position_ids
        2. Scatter vision embeddings into text sequence at vision_mask positions
        3. Run text decoder (with PLE + KV sharing)

        During token generation or text-only prefill:
        - Provide dummy vision inputs to the text model
        """
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
                pixel_position_ids,
            ).to(self.text_config.neuron_config.torch_dtype)

            if vision_embeddings.dim() == 2:
                num_images = pixel_values.shape[0]
                vision_embeddings = vision_embeddings.view(
                    num_images, -1, vision_embeddings.shape[-1]
                )

            batch_sz = 1 if vision_mask.dim() == 1 else vision_mask.shape[0]
            num_images, seq_len, embedding_dim = vision_embeddings.shape
            img_per_sample = num_images // batch_sz
            vision_embeddings = vision_embeddings.view(
                batch_sz, img_per_sample * seq_len, embedding_dim
            )

            vision_embeddings = pad_vision_embeddings(
                vision_embeddings=vision_embeddings, pad_limit=pad_target_size
            )

            vision_mask = self.generate_positions_from_mask(mask=vision_mask.squeeze())
            vision_mask = self.pad_positions(
                positions=vision_mask,
                target_size=pad_target_size,
                fill_value=pad_fill_value,
            )
        else:
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

    def _get_constructed_outputs(self, outputs, is_run_on_neuron):
        """Process model outputs into standard format."""
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

        return constructed_outputs

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration.from_pretrained(model_path, **kwargs)
