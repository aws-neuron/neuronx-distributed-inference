# Meta Llama 3 is licensed under the Meta Llama 3 Community License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE in the
# # current directory, mllama/.

import logging
import math
from typing import Callable

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import repeat_kv

from .aspect_ratio_utils import convert_aspect_ratios_to_ids
from .encoder_utils import (
    build_attention_mask_gen_vectors,
    contract_num_tokens_from_mult8,
    expand_num_tokens_to_mult8,
    get_aspect_ratio_mask,
)
from .hf_embeddings import PrecomputedAspectRatioEmbedding, PrecomputedPositionEmbedding
from .utils import HF_CHECKPOINT, META_CHECKPOINT, get_negative_inf_value, to_2tuple

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402

from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

_flash_fwd_call = nki_jit()(attention_isa_kernel)

logger = logging.getLogger(__name__)


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: int,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def _unfold(self, x):
        k0, k1 = self.kernel_size[0], self.kernel_size[1]
        assert k0 == self.stride
        assert k1 == self.stride

        bsz, nc, r, c = x.shape
        x = x.reshape(bsz, nc, r // k0, k0, c // k1, k1)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(bsz, (r // k0) * (c // k1), nc * k0 * k1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        # x = F.linear(x, self._linear.weight)
        x = self._linear(x)
        return x


# Image encoder for inference
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class ImageFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        sequence_parallel_enabled: bool,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        # layers
        self.c_fc = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=True,
            gather_output=False,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=1 if sequence_parallel_enabled else None,
        )
        self.c_proj = RowParallelLinear(
            hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=1 if sequence_parallel_enabled else None,
        )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x):
        hidden = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = self.c_proj(hidden)
        return hidden


class NeuronImageAttention(NeuronAttentionBase):
    def __init__(
        self, config: InferenceConfig, hidden_size, num_attention_heads, sequence_parallel_enabled
    ):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.padding_side = self.neuron_config.padding_side
        self.torch_dtype = self.neuron_config.torch_dtype
        self.num_cores_per_group = config.num_cores_per_group

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1
        self.fused_qkv = getattr(self.neuron_config, "fused_qkv", False)
        self.clip_qkv = None
        self.attn_kernel_enabled = True

        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_dimension = 1 if sequence_parallel_enabled else None

        self.init_gqa_properties()

    @staticmethod
    def perform_maskless_sdpa(Q, K, V, mask_gen_vectors, use_flash_attention, dtype):
        """
        Define as static method for easier unit testing.
        """
        bsz, num_heads, q_len, head_dim = Q.shape
        # Scale using original head_dim
        scale = 1.0 / math.sqrt(head_dim)

        Q_cat = torch.cat([Q, mask_gen_vectors.unsqueeze(3).to(Q.dtype)], dim=3)
        K_cat_v = mask_gen_vectors.unsqueeze(3).to(
            K.dtype
        ).detach().clone() * get_negative_inf_value(K.dtype)
        K_cat = torch.cat([K, K_cat_v], dim=3)
        if use_flash_attention:
            # Append to V to have uniform head_dim across QKV, to avoid error within kernel
            V_cat = torch.cat(
                [
                    V,
                    torch.zeros(
                        V.shape[0], V.shape[1], V.shape[2], 1, device=V.device, dtype=V.dtype
                    ),
                ],
                dim=3,
            )
            Q, K, V = Q_cat, K_cat, V_cat
            new_head_dim = head_dim + 1
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # original Q shape: batch, num_heads, seqlen, d_head
            Q = (
                Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
                .reshape((bsz * num_heads, new_head_dim, q_len))
                .to(dtype)
            )
            Q = Q * scale
            K = K.permute(0, 1, 3, 2).reshape((bsz * num_heads, new_head_dim, q_len)).to(dtype)
            V = V.reshape((bsz * num_heads, q_len, new_head_dim)).to(dtype)
            # shape: (B*H)DS
            attn_output = torch.zeros(
                bsz * num_heads, new_head_dim, q_len, dtype=Q.dtype, device=Q.device
            )
            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K.shape}")
            logger.debug(f"V input shape {V.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")
            # Use non-causal flash attention kernel
            _flash_fwd_call(
                Q,
                K,
                V,
                1.0,
                attn_output,
                kernel_name="AttentionMMSoftmaxMMWithoutSwap",
            )
            # shape: BHDS
            attn_output = attn_output.reshape((bsz, num_heads, new_head_dim, q_len))
            attn_output = attn_output[:, :, :head_dim, :]
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            # shape: BHSD
            attn_output = F.scaled_dot_product_attention(
                Q_cat, K_cat, V, attn_mask=None, scale=scale
            )
        return attn_output

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        self.attn_kernel_enabled = (
            (self.attn_kernel_enabled or q_len >= 4096)
            and (Q.shape == K_active.shape == V_active.shape)
            and (self.padding_side == "right" or bsz == 1)
        )
        assert int(self.logical_neuron_cores) == 1
        attn_output = self.perform_maskless_sdpa(
            Q=Q,
            K=K_active,
            V=V_active,
            mask_gen_vectors=attention_mask,
            use_flash_attention=self.attn_kernel_enabled,
            dtype=self.torch_dtype,
        )

        flash_attention_strategy = (
            FlashAttentionStrategy.UNSHARDED_KERNEL
            if self.attn_kernel_enabled
            else FlashAttentionStrategy.NONE
        )
        return attn_output, flash_attention_strategy


class ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        n_head: int,
        sequence_parallel_enabled: bool,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads
        self.attn = NeuronImageAttention(
            config=config,
            hidden_size=d_model,
            num_attention_heads=self.n_heads,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = ImageFeedForward(
            dim=d_model,
            hidden_dim=int(mlp_ratio * d_model),
            dropout=0.0,
            sequence_parallel_enabled=sequence_parallel_enabled,
            act_layer=act_layer,
        )
        self.ln_2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        x = x + _gate_attn * self.attn(self.ln_1(x), mask)[0]
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class ImageTransformer(nn.Module):
    def __init__(
        self,
        config,
        width: int,
        layers: int,
        heads: int,
        sequence_parallel_enabled: bool,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        self.config = config
        self.width = width
        self.layers = layers
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_dimension = 1
        self.dont_modify_sequence_parallel_enabled = True

        self.resblocks = nn.ModuleList(
            [
                ImageTransformerBlock(
                    config=config,
                    d_model=width,
                    n_head=heads,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    gated=gated,
                )
                for _ in range(self.layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_intermediate=None, mask=None):
        if self.sequence_parallel_enabled:
            # Enter sequence parallel
            # TODO: Replace with rank-specific scatter
            x = _reduce_scatter_along_dim(x, self.sequence_dimension, computation=xm.REDUCE_MAX)

        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                if self.sequence_parallel_enabled:
                    full_x = gather_from_sequence_parallel_region(x, self.sequence_dimension)
                else:
                    full_x = x
                out.append(full_x)
            x = r(x, mask=mask)

        if self.sequence_parallel_enabled:
            # Exit sequence parallel
            x = gather_from_sequence_parallel_region(x, self.sequence_dimension)

        if return_intermediate is not None:
            # Stack on dim=2 to have better layout for compiler, then transpose with last dim
            int_x = torch.stack(out, dim=2)
            int_x = int_x.transpose(2, -1)
            return x, int_x
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        config,
        vision_config,
        max_num_tiles: int,
        ckpt_path: str = None,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1280,
        layers: int = 32,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        in_channels: int = 3,
        load_ckpt: bool = False,
        n_global_layers: int = 2,
        global_model: bool = False,
        return_intermediate=None,
        sequence_parallel_enabled=True,
    ):
        super().__init__()
        self.config = config
        self.vision_config = vision_config
        self.global_model = global_model
        self.return_intermediate = return_intermediate
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.conv1 = ColumnParallelConv2dPatch(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.transformer = ImageTransformer(
            config=config,
            width=width,
            layers=layers,
            heads=heads,
            sequence_parallel_enabled=sequence_parallel_enabled,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
        )
        self.global_transformer = ImageTransformer(
            config=config,
            width=width,
            layers=n_global_layers,
            heads=heads,
            sequence_parallel_enabled=sequence_parallel_enabled,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            gated=True,
        )

        self._set_tile_pos_embedding(width)
        self._set_positional_embedding(scale, width)

    def apply_positional_embedding(self, x, ar, ar_ids):
        if self.config.checkpoint == HF_CHECKPOINT:
            return self.gated_positional_embedding(x, ar_ids)
        elif self.config.checkpoint == META_CHECKPOINT:
            # apply regular position embedding
            bsz, num_chunks, num_tokens, dim = x.shape
            x = x.view(bsz * num_chunks, num_tokens, dim)
            x = x + self.positional_embedding * (1 - self.gated_positional_embedding_gate.tanh())
            x = x.view(bsz, num_chunks, num_tokens, dim)

            embed = self.gated_positional_embedding.view(
                self.max_num_tiles**2,
                self.gated_positional_embedding.shape[2] * self.gated_positional_embedding.shape[3],
            )
            for idx in range(ar.shape[0]):
                arx = ar[idx]
                ar_mask = get_aspect_ratio_mask(arx, self.max_num_tiles).to(dtype=embed.dtype)
                _pos_embed = ar_mask @ embed  # (T, T^2) @ (T^2, N*W) -> (T, N*W)
                _pos_embed = _pos_embed.view(
                    self.max_num_tiles,
                    self.gated_positional_embedding.shape[2],
                    self.gated_positional_embedding.shape[3],
                )
                x[idx] += _pos_embed * self.gated_positional_embedding_gate.tanh()
            return x

    def apply_class_embedding(self, x):
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x

    def forward(self, images: torch.Tensor, ar: torch.Tensor, ar_ids: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar = ar.reshape(bsz * num_concurrent_media, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.conv1(x)  # shape = [*, width, grid ** 2]

        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        if self.config.checkpoint == HF_CHECKPOINT:
            x = self.pre_tile_pos_embed(x, ar_ids)
        elif self.config.checkpoint == META_CHECKPOINT:
            x = self.pre_tile_pos_embed(x, ar)

        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.apply_positional_embedding(x, ar, ar_ids)

        x = self.ln_pre(x)
        npad, attn_mask = 0, None
        x, npad = expand_num_tokens_to_mult8(x)
        # Pass in the attention_mask_gen_vectors as the attn_mask
        attn_mask = build_attention_mask_gen_vectors(x, ar, ntok, num_chunks, 1)
        x = x.view(bsz * num_concurrent_media, -1, dim)
        x, int_x = self.transformer(x, return_intermediate=self.return_intermediate, mask=attn_mask)

        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        if self.config.checkpoint == HF_CHECKPOINT:
            x = self.post_tile_pos_embed(x, ar_ids)
        elif self.config.checkpoint == META_CHECKPOINT:
            x = self.post_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = self.global_transformer(x, mask=attn_mask)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = contract_num_tokens_from_mult8(x, npad)

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        int_x = contract_num_tokens_from_mult8(int_x, npad)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = torch.cat([x, int_x], dim=-1)
        return x

    def _set_positional_embedding(self, scale, width):
        if self.config.checkpoint == HF_CHECKPOINT:
            self.gated_positional_embedding = PrecomputedPositionEmbedding(
                config=self.vision_config
            )
        elif self.config.checkpoint == META_CHECKPOINT:
            self.gated_positional_embedding = nn.Parameter(
                scale
                * torch.randn(
                    self.max_num_tiles,
                    self.max_num_tiles,
                    self.grid_size[0] * self.grid_size[1] + 1,
                    width,
                )
            )
            self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))

    def _set_tile_pos_embedding(self, width):
        # pre and post tile position embedding
        if self.config.checkpoint == META_CHECKPOINT:
            self.pre_tile_pos_embed = TilePositionEmbedding(
                num_tiles=self.max_num_tiles,
                width=width,
                gated=True,
            )
            self.post_tile_pos_embed = TilePositionEmbedding(
                num_tiles=self.max_num_tiles,
                width=width,
                gated=True,
            )
        elif self.config.checkpoint == HF_CHECKPOINT:
            self.pre_tile_pos_embed = PrecomputedAspectRatioEmbedding(
                config=self.vision_config,
            )
            self.post_tile_pos_embed = PrecomputedAspectRatioEmbedding(
                config=self.vision_config,
            )


class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        width: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = nn.Parameter(
            torch.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width)
        )
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        assert num_tiles is None or num_tiles == self.num_tiles, "Unsupported config on Neuron"
        embed = self.embedding.view(-1, self.width)  # (T^2, W)

        out_pos_embed = torch.zeros(
            x.shape[0], self.num_tiles, self.width, device=x.device, dtype=x.dtype
        )  # (B, T, W)
        for idx in range(ar.shape[0]):
            ar_mask = get_aspect_ratio_mask(ar[idx], self.num_tiles).to(dtype=embed.dtype)
            out_pos_embed[idx] = ar_mask @ embed  # (T, T^2) @ (T^2, W) -> (T, W)

        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed.unsqueeze(2)
        return x


class NeuronMllamaVisionModel(nn.Module):
    """
    The neuron version of CrossAttentionTransformerVision of llama multimodal model
    """

    def __init__(self, config: InferenceConfig, vision_config) -> None:
        super().__init__()
        return_intermediate = vision_config.intermediate_layers_indices
        self.config = config
        self.vision_input_dim = vision_config.hidden_size
        self.image_res = vision_config.image_size
        self.max_num_chunks = vision_config.max_num_tiles
        if return_intermediate is not None:
            self.vision_input_dim = (len(return_intermediate) + 1) * self.vision_input_dim
        self.patch_size = vision_config.patch_size
        self.vision_encoder = VisionEncoder(
            config=config,
            vision_config=vision_config,
            max_num_tiles=vision_config.max_num_tiles,
            image_size=vision_config.image_size,
            patch_size=self.patch_size,
            width=vision_config.hidden_size,
            layers=vision_config.num_hidden_layers,
            heads=vision_config.attention_heads,
            in_channels=vision_config.num_channels,
            n_global_layers=vision_config.num_global_layers,
            global_model=True,
            return_intermediate=return_intermediate,
        )
        # vision token projection
        self.vision_projection = ColumnParallelLinear(
            self.vision_input_dim,
            config.hidden_size,
            bias=True,
            gather_output=True,
        )

    def forward(self, images: torch.Tensor, aspect_ratios: torch.Tensor) -> torch.Tensor:
        # return vision_tokens shape: (batch_size, num_image_per_prompt, vision_max_num_chunks, num_vision_tokens, hidden_size)

        if self.config.checkpoint == HF_CHECKPOINT:
            aspect_ratios_ids = convert_aspect_ratios_to_ids(aspect_ratios, self.max_num_chunks).to(
                "xla"
            )
        else:
            aspect_ratios_ids = None
        vision_tokens = self.vision_encoder(images, aspect_ratios, aspect_ratios_ids)
        vision_tokens = self.vision_projection(vision_tokens)

        return vision_tokens
