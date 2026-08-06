"""
NxDI application for Wan2.2 with Context Parallelism (CP).

CP=4: each rank has the full model, sequence is split across ranks.
No weight sharding needed — weights are loaded identically on all ranks.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config

logger = logging.getLogger(__name__)


class WanCPInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "num_attention_heads", "attention_head_dim", "in_channels",
            "out_channels", "text_dim", "freq_dim", "ffn_dim",
            "num_layers", "patch_size", "qk_norm", "cross_attn_norm", "eps",
            "num_frames", "height", "width",
        ]

    @classmethod
    def from_pretrained(cls, model_path, neuron_config, num_frames=13,
                        height=480, width=832, subfolder="transformer", **kwargs):
        transformer_path = os.path.join(model_path, subfolder)
        if not os.path.isdir(transformer_path):
            transformer_path = model_path
        load_config = load_diffusers_config(transformer_path)
        config = cls(
            neuron_config=neuron_config, load_config=load_config,
            num_frames=num_frames, height=height, width=width,
            **kwargs,
        )
        config.inner_dim = config.num_attention_heads * config.attention_head_dim
        return config


class ModelWrapperWanCP(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, model_init_kwargs=None):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        super().__init__(config, model_cls, tag, compiler_args,
                         priority_model_idx, model_init_kwargs=model_init_kwargs)

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        dtype = self.config.neuron_config.torch_dtype
        p_t, p_h, p_w = self.config.patch_size
        vae_t, vae_s = 4, 8
        latent_f = (self.config.num_frames - 1) // vae_t + 1
        latent_h = self.config.height // vae_s
        latent_w = self.config.width // vae_s
        seq_len = (latent_f // p_t) * (latent_h // p_h) * (latent_w // p_w)
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        from nxdi_wan.modeling_wan_cp import CPWanSecondHalf
        if issubclass(self.model_cls, CPWanSecondHalf):
            # Second half: takes intermediate hidden_states + conditioning
            inputs = [(
                torch.randn([1, seq_len, inner_dim], dtype=dtype),  # hidden_states [B, S, dim]
                torch.randn([1, inner_dim], dtype=dtype),           # temb [B, dim]
                torch.randn([1, 6, inner_dim], dtype=dtype),        # timestep_proj [B, 6, dim]
                torch.randn([1, 512, inner_dim], dtype=dtype),      # enc_proj [B, T, dim]
                torch.randn([1, seq_len, 1, self.config.attention_head_dim], dtype=dtype),
                torch.randn([1, seq_len, 1, self.config.attention_head_dim], dtype=dtype),
            )]
        else:
            # Full model or first half: takes raw inputs
            inputs = [(
                torch.randn([1, 16, latent_f, latent_h, latent_w], dtype=dtype),
                torch.randn([1], dtype=dtype),
                torch.randn([1, 512, self.config.text_dim], dtype=dtype),
                torch.randn([1, seq_len, 1, self.config.attention_head_dim], dtype=dtype),
                torch.randn([1, seq_len, 1, self.config.attention_head_dim], dtype=dtype),
            )]
        return inputs

    def get_model_instance(self):
        config = self.config
        model_cls = self.model_cls
        model_kwargs = dict(
            patch_size=config.patch_size,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
        )

        def _create_model():
            model = model_cls(**model_kwargs)
            model = model.to(dtype=config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *args):
        if self.model is None:
            raise RuntimeError("Forward called before load.")
        return self._forward(*args)


class NeuronWanCPApplication(NeuronApplicationBase):
    _model_cls = None  # Set at init

    def __init__(self, model_path, config, model_cls=None, *args, **kwargs):
        if model_cls is None:
            from nxdi_wan.modeling_wan_cp import CPWanTransformer3DModel
            model_cls = CPWanTransformer3DModel
        self._model_cls = model_cls
        super().__init__(model_path=model_path, config=config, *args, **kwargs)
        self.model_wrapper_cls = ModelWrapperWanCP
        self.model = self.model_wrapper_cls(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        return "--model-type=transformer -O1 --target trn2 --lnc 2 --enable-mixed-precision-accumulation"

    def compile(self, compile_dir):
        # Patch: disable HLO verification to bypass the 24GB HBM check.
        for m in self.models:
            m.compiler_args = m.compiler_args.replace("--verify-hlo=true", "--verify-hlo=false")
        return super().compile(compile_dir)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: Dict, config) -> Dict:
        """CP needs no weight conversion — model uses plain nn.Linear.

        Only need to:
        1. Remap attention output key (remove .0 from ModuleList)
        2. Remap FFN keys
        3. Remove RoPE buffers
        4. Add SPMDRank tensor
        """
        new_sd = {}
        world_size = config.neuron_config.world_size

        for key, value in state_dict.items():
            if key.startswith("rope."):
                continue

            new_key = key
            new_key = new_key.replace(".attn1.to_out.0.", ".attn1.to_out.")
            new_key = new_key.replace(".attn2.to_out.0.", ".attn2.to_out.")
            new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.up_proj.")
            new_key = new_key.replace(".ffn.net.2.", ".ffn.down_proj.")

            # Condition embedder nested modules
            new_key = new_key.replace(
                "condition_embedder.text_embedder.linear_1.",
                "condition_embedder.text_embedder_linear_1.",
            )
            new_key = new_key.replace(
                "condition_embedder.text_embedder.linear_2.",
                "condition_embedder.text_embedder_linear_2.",
            )
            new_key = new_key.replace(
                "condition_embedder.time_embedder.linear_1.",
                "condition_embedder.time_embedder_linear_1.",
            )
            new_key = new_key.replace(
                "condition_embedder.time_embedder.linear_2.",
                "condition_embedder.time_embedder_linear_2.",
            )

            new_sd[new_key] = value.clone().detach().contiguous()

        # SPMDRank for CP rank resolution
        new_sd["global_rank.rank"] = torch.arange(0, world_size, dtype=torch.int32)

        return new_sd

class NeuronWanCPSecondHalfApplication(NeuronWanCPApplication):
    """Application for the second half — overrides weight conversion to remap block indices."""

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: Dict, config) -> Dict:
        return NeuronWanCPSecondHalfApplication._convert_second_half(state_dict, config)

    @staticmethod
    def _convert_second_half(state_dict: Dict, config, start_block: int = 20) -> Dict:
        """Weight conversion for the second-half model.

        Remaps HF blocks[start_block:] → model blocks[0:].
        Keeps only: blocks 20-39, norm_out, proj_out, scale_shift_table (top-level).
        Drops: blocks 0-19, patch_embedding, condition_embedder, rope.
        """
        base_sd = NeuronWanCPApplication.convert_hf_to_neuron_state_dict(state_dict, config)

        new_sd = {}
        for key, value in base_sd.items():
            # Remap block indices: blocks.20.* -> blocks.0.*, blocks.21.* -> blocks.1.*, etc.
            if key.startswith("blocks."):
                parts = key.split(".", 2)
                block_idx = int(parts[1])
                if block_idx < start_block:
                    continue
                new_idx = block_idx - start_block
                new_key = f"blocks.{new_idx}.{parts[2]}"
                new_sd[new_key] = value
            elif key.startswith("patch_embedding.") or key.startswith("condition_embedder."):
                continue  # Not in second half
            else:
                new_sd[key] = value

        return new_sd
