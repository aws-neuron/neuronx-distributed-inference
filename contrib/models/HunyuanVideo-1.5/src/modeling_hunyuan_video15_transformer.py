"""
HunyuanVideo-1.5 Transformer Backbone for NxDI

Defines the 54-block MMDiT transformer backbone with TP=2 support.
Includes time embedding, patch embedding, 3D RoPE, ByT5/image projections,
condition type embedding, token reordering, and output unpatchify.

Usage:
    # Compile
    from modeling_hunyuan_video15_transformer import (
        HunyuanVideo15TransformerConfig, NeuronHunyuanVideo15Transformer
    )
    config = HunyuanVideo15TransformerConfig.from_pretrained(model_path, tp_degree=2)
    model = NeuronHunyuanVideo15Transformer(model_path, config=config)
    model.compile(save_path)

    # Load and run
    model.load(save_path)
    output = model.forward(video_latent, mllm_refined, byt5_raw, image_embeds,
                           timestep, reorder_idx, zero_mask, attn_mask)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.models.diffusers.embeddings import Timesteps, get_1d_rotary_pos_embed
# Import TP submodules from the validated original implementation
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent / "03_transformer_backbone" / "03_transformer_block"))
from neuron_full_backbone import TPTransformerBlock, TPFFN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class HunyuanVideo15TransformerConfig(InferenceConfig):
    """Configuration for the HunyuanVideo-1.5 transformer backbone."""

    # 480p 33-frame defaults
    NUM_LAYERS = 54
    HIDDEN_SIZE = 2048
    NUM_HEADS = 16
    HEAD_DIM = 128
    MLP_RATIO = 4.0
    IN_CHANNELS = 65   # 32 latent + 32 cond + 1 mask
    OUT_CHANNELS = 32
    T_LAT, H_LAT, W_LAT = 9, 30, 40
    N_ENCODER = 1985   # 1000 MLLM + 256 ByT5 + 729 image
    TEXT_EMBED_2_DIM = 1472  # ByT5 hidden size
    IMAGE_EMBED_DIM = 1152

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = {
            "num_layers": self.NUM_LAYERS, "hidden_size": self.HIDDEN_SIZE,
            "num_heads": self.NUM_HEADS, "head_dim": self.HEAD_DIM,
            "mlp_ratio": self.MLP_RATIO, "in_channels": self.IN_CHANNELS,
            "out_channels": self.OUT_CHANNELS, "t_lat": self.T_LAT,
            "h_lat": self.H_LAT, "w_lat": self.W_LAT, "n_encoder": self.N_ENCODER,
            "text_embed_2_dim": self.TEXT_EMBED_2_DIM, "image_embed_dim": self.IMAGE_EMBED_DIM,
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)
        self.n_latent = self.t_lat * self.h_lat * self.w_lat

    def get_required_attributes(self) -> List[str]:
        return ["num_layers", "hidden_size"]

    @classmethod
    def from_pretrained(cls, model_path: str, tp_degree: int = 2, **kwargs):
        neuron_config = NeuronConfig(
            tp_degree=tp_degree, torch_dtype=torch.bfloat16, batch_size=1,
            seq_len=cls.T_LAT * cls.H_LAT * cls.W_LAT + cls.N_ENCODER,
            save_sharded_checkpoint=True,
        )
        defaults = {
            "num_layers": cls.NUM_LAYERS, "hidden_size": cls.HIDDEN_SIZE,
            "num_heads": cls.NUM_HEADS, "head_dim": cls.HEAD_DIM,
            "mlp_ratio": cls.MLP_RATIO, "in_channels": cls.IN_CHANNELS,
            "out_channels": cls.OUT_CHANNELS, "t_lat": cls.T_LAT,
            "h_lat": cls.H_LAT, "w_lat": cls.W_LAT, "n_encoder": cls.N_ENCODER,
            "text_embed_2_dim": cls.TEXT_EMBED_2_DIM, "image_embed_dim": cls.IMAGE_EMBED_DIM,
        }
        defaults.update(kwargs)
        return cls(neuron_config=neuron_config, **defaults)


# ---------------------------------------------------------------------------
# TP Submodules — imported from validated original
# ---------------------------------------------------------------------------
# TPTransformerBlock and TPFFN are imported above from neuron_full_backbone.py


# ---------------------------------------------------------------------------
# Full Backbone Model
# ---------------------------------------------------------------------------
class HunyuanVideo15TransformerModel(nn.Module):
    """
    HunyuanVideo-1.5 transformer backbone.

    Inputs:
        video_latent: [B, 65, T, H, W] — concatenated latent + cond + mask
        mllm_refined: [B, 1000, 2048] — refined MLLM embeddings + cond_type(0)
        byt5_raw: [B, 256, 1472] — raw ByT5 encoder output
        image_embeds: [B, 729, 1152] — image embeddings (zeros for t2v)
        timestep: [B] — diffusion timestep
        reorder_idx: [1985] — token reorder indices
        zero_mask: [1985] — valid token mask
        attn_mask: [B, 1, N_total, N_total] — attention mask

    Output:
        [B, 32, T, H, W] — denoised prediction
    """

    def __init__(self, config: HunyuanVideo15TransformerConfig):
        super().__init__()
        dim = config.hidden_size

        # Time embedding
        self.time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_linear_1 = nn.Linear(256, dim)
        self.time_silu = nn.SiLU()
        self.time_linear_2 = nn.Linear(dim, dim)

        # Patch embedding
        self.patch_embed = nn.Conv3d(config.in_channels, dim, kernel_size=(1, 1, 1))

        # ByT5 text projection
        self.byt5_norm = nn.LayerNorm(config.text_embed_2_dim)
        self.byt5_linear_1 = nn.Linear(config.text_embed_2_dim, 2048)
        self.byt5_linear_2 = nn.Linear(2048, 2048)
        self.byt5_linear_3 = nn.Linear(2048, dim)
        self.byt5_act = nn.GELU()

        # Image projection
        self.img_norm_in = nn.LayerNorm(config.image_embed_dim)
        self.img_linear_1 = nn.Linear(config.image_embed_dim, config.image_embed_dim)
        self.img_act = nn.GELU()
        self.img_linear_2 = nn.Linear(config.image_embed_dim, dim)
        self.img_norm_out = nn.LayerNorm(dim)

        # Condition type embedding
        self.cond_type_embed = nn.Embedding(3, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TPTransformerBlock(dim, config.num_heads, config.head_dim, config.mlp_ratio)
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.norm_silu = nn.SiLU()
        self.norm_linear = nn.Linear(dim, dim * 2)
        self.norm_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(dim, config.out_channels)

        # Spatial dimensions for unpatchify
        self.t_lat = config.t_lat
        self.h_lat = config.h_lat
        self.w_lat = config.w_lat

        # Pre-compute 3D RoPE as buffer (constant for fixed resolution)
        self._init_rope(config)

    def _init_rope(self, config):
        grid = torch.meshgrid(
            *[torch.arange(s, dtype=torch.float32) for s in [config.t_lat, config.h_lat, config.w_lat]],
            indexing="ij",
        )
        grid = torch.stack(grid, dim=0)
        rope_dims = [16, 56, 56]
        freqs = [get_1d_rotary_pos_embed(rope_dims[i], grid[i].reshape(-1), 256.0, use_real=True) for i in range(3)]
        self.register_buffer("rope_cos", torch.cat([f[0] for f in freqs], dim=1))
        self.register_buffer("rope_sin", torch.cat([f[1] for f in freqs], dim=1))

    def forward(self, video_latent, mllm_refined, byt5_raw, image_embeds,
                timestep, reorder_idx, zero_mask, attn_mask):
        # Time embedding
        temb = self.time_linear_2(self.time_silu(self.time_linear_1(
            self.time_proj(timestep).to(timestep.dtype))))

        # Patch embedding
        hs = self.patch_embed(video_latent).flatten(2).transpose(1, 2)

        # ByT5 projection + cond_type(1)
        byt5 = self.byt5_act(self.byt5_linear_1(self.byt5_norm(byt5_raw)))
        byt5 = self.byt5_act(self.byt5_linear_2(byt5))
        byt5 = self.byt5_linear_3(byt5)
        byt5 = byt5 + self.cond_type_embed(torch.ones(1, 256, dtype=torch.long, device=byt5.device))

        # Image projection + cond_type(2), zeroed for t2v
        img = self.img_norm_out(self.img_linear_2(self.img_act(
            self.img_linear_1(self.img_norm_in(image_embeds)))))
        img = img * 0.0
        img = img + self.cond_type_embed(2 * torch.ones(1, 729, dtype=torch.long, device=img.device))

        # Concatenate and reorder encoder tokens
        all_enc = torch.cat([mllm_refined, byt5, img], dim=1)
        encoder_hs = torch.index_select(all_enc, 1, reorder_idx) * zero_mask.unsqueeze(0).unsqueeze(-1)

        # 54 transformer blocks
        for block in self.blocks:
            hs, encoder_hs = block(
                hs, encoder_hs, temb, attn_mask,
                self.rope_cos.to(hs.dtype), self.rope_sin.to(hs.dtype),
            )

        # Output: AdaLayerNorm + linear + unpatchify
        emb = self.norm_linear(self.norm_silu(temb))
        sc, sh = emb.chunk(2, dim=1)
        hs = self.norm_ln(hs) * (1 + sc)[:, None] + sh[:, None]
        hs = self.proj_out(hs)
        B = hs.shape[0]
        hs = hs.reshape(B, self.t_lat, self.h_lat, self.w_lat, -1, 1, 1, 1)
        hs = hs.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


# ---------------------------------------------------------------------------
# NxDI Wrapper and Application
# ---------------------------------------------------------------------------
class HunyuanVideo15TransformerWrapper(ModelWrapper):
    """ModelWrapper for compilation and loading."""

    def __init__(self, config, model_cls, **kwargs):
        super().__init__(config, model_cls, **kwargs)
        self.bucket_config = None

    def input_generator(self):
        c = self.config
        d = c.neuron_config.torch_dtype
        n = c.t_lat * c.h_lat * c.w_lat
        t = n + c.n_encoder
        return [(
            torch.randn(1, c.in_channels, c.t_lat, c.h_lat, c.w_lat, dtype=d),
            torch.randn(1, 1000, c.hidden_size, dtype=d),
            torch.randn(1, 256, c.text_embed_2_dim, dtype=d),
            torch.zeros(1, 729, c.image_embed_dim, dtype=d),
            torch.tensor([1000.0], dtype=d),
            torch.arange(c.n_encoder, dtype=torch.long),
            torch.ones(c.n_encoder, dtype=d),
            torch.zeros(1, 1, t, t, dtype=d),
        )]

    def get_model_instance(self):
        def _create():
            m = self.model_cls(self.config)
            m.to(self.config.neuron_config.torch_dtype).eval()
            return m
        return BaseModelInstance(module_cls=_create, input_output_aliases={})

    def forward(self, *args):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._forward(*args)


class NeuronHunyuanVideo15Transformer(NeuronApplicationBase):
    """
    HunyuanVideo-1.5 Transformer for NxDI.

    This class follows the NxDI pattern:
        model = NeuronHunyuanVideo15Transformer(model_path, config=config)
        model.compile(save_path)
        model.load(save_path)
        output = model.forward(...)
    """

    _model_cls = HunyuanVideo15TransformerModel

    def __init__(self, model_path, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)
        self.model = HunyuanVideo15TransformerWrapper(
            config=self.config, model_cls=self._model_cls,
            tag="HunyuanVideo15Transformer",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)

    def forward(self, *args):
        return self.models[0](*args)

    def get_compiler_args(self):
        return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def load_hf_model(model_path):
        from diffusers import HunyuanVideo15Transformer3DModel
        return HunyuanVideo15Transformer3DModel.from_pretrained(
            model_path.rstrip("/"), subfolder="transformer",
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """
        Convert HuggingFace HunyuanVideo15Transformer3DModel state dict to Neuron format.

        Key mapping:
            x_embedder.proj.          -> patch_embed.
            time_embed.timestep_embedder.linear_{1,2}. -> time_linear_{1,2}.
            context_embedder_2.norm.  -> byt5_norm.
            context_embedder_2.linear_{1,2,3}. -> byt5_linear_{1,2,3}.
            image_embedder.*          -> img_*
            cond_type_embed.          -> cond_type_embed.
            transformer_blocks.       -> blocks.
            norm_out.linear.          -> norm_linear.
            proj_out.                 -> proj_out.
            attn.to_out.0.            -> attn.to_out.
            .ff.net.0.proj.           -> .ff.proj.
            .ff.net.2.               -> .ff.out.
            .ff_context.net.0.proj.   -> .ff_context.proj.
            .ff_context.net.2.       -> .ff_context.out.
        """
        KEY_MAP = {
            "x_embedder.proj.": "patch_embed.",
            "time_embed.timestep_embedder.linear_1.": "time_linear_1.",
            "time_embed.timestep_embedder.linear_2.": "time_linear_2.",
            "context_embedder_2.norm.": "byt5_norm.",
            "context_embedder_2.linear_1.": "byt5_linear_1.",
            "context_embedder_2.linear_2.": "byt5_linear_2.",
            "context_embedder_2.linear_3.": "byt5_linear_3.",
            "image_embedder.norm_in.": "img_norm_in.",
            "image_embedder.linear_1.": "img_linear_1.",
            "image_embedder.linear_2.": "img_linear_2.",
            "image_embedder.norm_out.": "img_norm_out.",
            "cond_type_embed.": "cond_type_embed.",
            "transformer_blocks.": "blocks.",
            "norm_out.linear.": "norm_linear.",
            "proj_out.": "proj_out.",
        }
        VALID_PREFIXES = [
            "patch_embed.", "time_linear_", "byt5_", "img_", "cond_type_embed.",
            "blocks.", "norm_", "proj_out.",
        ]

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old, new in KEY_MAP.items():
                if new_key.startswith(old):
                    new_key = new + new_key[len(old):]
                    break
            # Attention and FFN key remapping
            new_key = new_key.replace("attn.to_out.0.", "attn.to_out.")
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj.")
            new_key = new_key.replace(".ff.net.2.", ".ff.out.")
            new_key = new_key.replace(".ff_context.net.0.proj.", ".ff_context.proj.")
            new_key = new_key.replace(".ff_context.net.2.", ".ff_context.out.")

            if any(new_key.startswith(p) for p in VALID_PREFIXES):
                new_state_dict[new_key] = value.clone().detach().contiguous()

        return new_state_dict
