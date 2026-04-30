# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    OutputChannelParallelConv2d,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
    scatter_to_process_group_spmd,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_tensor_model_parallel_size,
)
from torch import Tensor, einsum, nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_llama4_vision_encoder_buckets,
    pad_image_tensor,
)
from neuronx_distributed_inference.models.mllama.modeling_mllama_vision import (
    ImageFeedForward,
    LayerNorm,
)
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)
from neuronx_distributed.modules.attention.utils import apply_rotary_polar_compatible
from neuronx_distributed_inference.modules.attention.utils import move_heads_front
from neuronx_distributed_inference.modules.autobucketing import generate_buckets
from neuronx_distributed_inference.utils.distributed import get_dp_rank_spmd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# The maximum number of chunks that can be sent as input to vision model
# Vision model is supposed to handle 5 images, each of 16 chunks + 1 global chunk
# Total number of chunks = 5 * (16 + 1) = 85.
# We add 3 dummy chunks to make number of chunks divisible by dp (dp=4)
VISION_MAX_NUM_CHUNKS = 88
VISION_MLP_RATIO = 4


class NeuronLlama4ImageAttention(NeuronAttentionBase):
    """
    Compared with NeuronLlamaAttention, this class just
    1. use bias in linear layers
    2. use config.vision_config
    """

    def __init__(self, config: InferenceConfig):
        super().__init__(
            config=config,
            hidden_size=config.vision_config.hidden_size,
            num_attention_heads=config.vision_config.num_attention_heads,
            num_key_value_heads=config.vision_config.num_attention_heads,
            head_dim=config.vision_config.hidden_size // config.vision_config.num_attention_heads,
            # TODO: this is a dummy rotary_emb layer
            # will be replaced with Llama4VisionRotaryEmbedding layer with HF support
            rotary_emb=LlamaRotaryEmbedding(config.text_config),
            rms_norm_eps=1e-6,
            o_bias=True,
            qkv_bias=True,
        )

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        rotary_freqs=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """
        Override NeuronAttentionBase.prep_qkv_tensors() to use apply_rotary_polar_compatible
        to match Llama3.2 MM Pytorch implementation
        """
        cos_cache, sin_cache = None, None
        Q, K, V, _ = self.qkv_proj(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids
        )

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if self.rotary_emb is not None:
            assert rotary_freqs is not None, "rotary_emb is initilazed but rotary_freqs is None"
            Q, K = apply_rotary_polar_compatible(Q.transpose(1, 2), K.transpose(1, 2), rotary_freqs)
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
        return Q, K, V, cos_cache, sin_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        rotary_freqs: Optional[torch.LongTensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Override NeuronAttentionBase.forward() to use apply_rotary_polar_compatible
        to match Llama3.2&4 MM Pytorch implementation
        """
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            rotary_freqs=rotary_freqs,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        flash_attn_strategy = FlashAttentionStrategy.NONE
        if past_key_value is None:
            attn_output, flash_attn_strategy = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        else:
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, attention_mask, active_mask
            )

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
        else:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output, adapter_ids=adapter_ids)

        past_key_value: Tuple[Tensor, Tensor] = (K, V)

        return attn_output, past_key_value, cos_cache, sin_cache


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        config: InferenceConfig,
    ):
        self.config = config
        d_model = config.vision_config.hidden_size
        n_head = config.vision_config.num_attention_heads
        act_layer = nn.GELU
        gated = False

        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads

        self.attn = NeuronLlama4ImageAttention(config)
        self.ln_1 = LayerNorm(d_model, dtype=config.neuron_config.torch_dtype)
        self.mlp = ImageFeedForward(
            dim=d_model,
            hidden_dim=int(VISION_MLP_RATIO * d_model),
            dropout=0.0,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            act_layer=act_layer,
            dtype=config.neuron_config.torch_dtype,
        )
        self.ln_2 = LayerNorm(d_model, dtype=config.neuron_config.torch_dtype)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

        seq_len = 1 + (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        self.position_ids = torch.arange(0, 0 + seq_len, dtype=torch.long)

    def attention(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor] = None,
    ):
        attn_output = self.attn(
            hidden_states=x, position_ids=self.position_ids, rotary_freqs=freq_cis
        )
        return attn_output[0]

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor] = None,
    ):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()

        x = x + _gate_attn * self.attention(self.ln_1(x), freq_cis=freq_cis)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    def __init__(
        self,
        config: InferenceConfig,
    ):
        super().__init__()
        self.sequence_parallel_enabled = getattr(
            config.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.resblocks = nn.ModuleList(
            [
                _TransformerBlock(
                    config,
                )
                for _ in range(config.vision_config.num_hidden_layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_intermediate=None, mask=None, freq_cis=None):
        if self.sequence_parallel_enabled:
            # Enter sequence parallel
            # TODO: Replace with rank-specific scatter
            x = _reduce_scatter_along_dim(x, self.sequence_dimension, computation=xm.REDUCE_MAX)

        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, freq_cis=freq_cis)

        if self.sequence_parallel_enabled:
            # Exit sequence parallel
            x = gather_from_sequence_parallel_region(x, self.sequence_dimension)

        if return_intermediate is not None:
            return x, torch.stack(out, dim=-1)
        return x


class PackingIndex:
    Z = 0  # Z (time) coordinate of the token in the original sample
    Y = 1  # Y (height) coordinate of the token in the original sample
    X = 2  # X (width) coordinate of the token in the original sample
    TIME = 3  # Total number of time units (frames) in the original sample
    HEIGHT = 4  # Height of the original sample
    WIDTH = 5  # Width of the original sample
    # USE INDEX TO CHECK THE TYPE OF THE TOKEN (see ID fields below)
    IDX = 6  # Full index of the token in the original sample (x + y * w + z * w * h)
    BATCH_IDX = 7  # Which batch element this token belongs to. Note the batch idx of padding tokens is BATCH_SIZE

    # Total size of the enum, remember to update this!
    NUM_METADATA = 8

    # Note: For padding tokens IDX = -1
    #       For cls tokens,    IDX = -2
    ID_CLS_TOKEN = -2
    ID_PAD_TOKEN = -1


def get_hw(size):
    from types import SimpleNamespace

    if isinstance(size, dict):
        height = size["height"]
        width = size["width"]
    elif isinstance(size, SimpleNamespace):
        height = size.height
        width = size.width
    elif isinstance(size, int):
        height = width = size
    else:
        raise TypeError(f"Size is of invalid type {type(size)}")
    return height, width


class NeuronConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
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
        bias=False,
        dtype=torch.float16,
    ) -> None:
        super().__init__()
        self.conv = OutputChannelParallelConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Reshape to match the original format
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


class VisionEncoder(nn.Module):
    def __init__(
        self,
        config: InferenceConfig,
    ):
        super().__init__()
        self.config = config
        h, w = get_hw(config.vision_config.image_size)
        ph, pw = get_hw(config.vision_config.patch_size)
        assert (
            h == w and ph == pw
        ), f"Non-square Image (h, w) ({h}, {w}) or Patch (ph, pw) ({ph}, {pw})"
        config.vision_config.image_size = h
        config.vision_config.patch_size = ph
        image_size = [h, w]
        patch_size = [ph, pw]

        dim = config.vision_config.hidden_size
        heads = config.vision_config.num_attention_heads
        in_channels = (
            config.vision_config.num_channels
            if hasattr(config.vision_config, "num_channels")
            else 3
        )

        if not hasattr(config.vision_config, "hidden_size"):
            setattr(config.vision_config, "hidden_size", config.hidden_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.conv1 = NeuronConv2dPatch(
            in_channels=in_channels,  # 3
            out_channels=dim,  # 1408
            kernel_size=patch_size[0],  # 14
            stride=patch_size[0],  # 14
            bias=False,
            dtype=config.neuron_config.torch_dtype,
        )
        scale = dim**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(dim))

        self.positional_embedding_vlm = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, dim)
        )

        self.layernorm_pre = LayerNorm(dim, dtype=config.neuron_config.torch_dtype)
        self.layernorm_post = LayerNorm(dim, dtype=config.neuron_config.torch_dtype)
        self.transformer = _Transformer(config)

        # NOTE: hack for the fixed res
        image_h, image_w = self.image_size
        patch_h, patch_w = self.patch_size
        idx_h, idx_w = image_h // patch_h, image_w // patch_w
        img_idx = torch.arange(image_h * image_w // (patch_h * patch_w), dtype=torch.int32)
        img_idx = img_idx.reshape(idx_h * idx_w, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = PackingIndex.ID_CLS_TOKEN

        # If we have a 2x2 grid of patches (idx_w=2, idx_h=2):
        # Each patch would have metadata like this:
        # [
        #   [x_pos, y_pos, height, width, index],
        #   [0,     0,     2,      2,     0],    # Top-left patch
        #   [1,     0,     2,      2,     1],    # Top-right patch
        #   [0,     1,     2,      2,     2],    # Bottom-left patch
        #   [1,     1,     2,      2,     3],    # Bottom-right patch
        # ]
        packed_img_idx = torch.empty(
            img_idx.shape[0],
            img_idx.shape[1],
            PackingIndex.NUM_METADATA - 1,
            dtype=torch.int32,
        )
        packed_img_idx[:, :, PackingIndex.Y] = img_idx // idx_w
        packed_img_idx[:, :, PackingIndex.X] = img_idx % idx_w
        packed_img_idx[:, :, PackingIndex.HEIGHT].fill_(idx_h)
        packed_img_idx[:, :, PackingIndex.WIDTH].fill_(idx_w)
        packed_img_idx[:, :, PackingIndex.IDX] = img_idx
        packed_img_idx = packed_img_idx.reshape(1, -1, PackingIndex.NUM_METADATA - 1)
        self.packed_img_idx = packed_img_idx  # for positional embedding load hook

        # compute rope freqs
        rope_freq = self.get_rope_freqs(dim // heads // 2)

        freqs_x = self.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.X] + 1)
        freqs_y = self.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.Y] + 1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]

        # disable RoPE for padding and cls tokens
        self.freq_cis = freqs.masked_fill(packed_img_idx[:, :, PackingIndex.IDX, None] < 0, 0)
        logger.info(f"in Neuron VisionEncoder self.freq_cis {self.freq_cis.dtype}")

        self.n_heads = heads // get_tensor_model_parallel_size()

        self._register_load_state_dict_pre_hook(self.load_hook)

    @classmethod  # Use class method for ease of testing
    def get_rope_freqs(cls, dim, theta=10000):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    @classmethod  # Use class method for ease of testing
    def compute_rope_freqs(cls, freqs, t):
        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: List[str] = None,
        unexpected_keys: List[str] = None,
        error_msgs: List[str] = None,
        return_state_dict: bool = False,
    ) -> None:
        orig_pos_embed = state_dict.get(prefix + "positional_embedding")
        if (
            orig_pos_embed is not None
            and orig_pos_embed.shape[-2:] != self.positional_embedding_vlm.shape[-2:]
        ):
            raise ValueError(
                f"Positional embedding shape {orig_pos_embed.shape} does not match expected shape {self.positional_embedding_vlm.shape}"
            )

        batch_size, token_per_image, _ = self.packed_img_idx.shape
        # Input points for idx are [x, y, w, h]
        idx = self.packed_img_idx.reshape(batch_size * token_per_image, 1, -1)
        total_windows, window_size, _ = idx.shape

        # coordinate normalization to convert spatial coordinates from pixel space to normalized space (-1 to 1)
        # Grid values are [-1, 1] and coords are w, h
        grid = (
            (
                idx[:, :, [PackingIndex.X, PackingIndex.Y]]
                / idx[:, :, [PackingIndex.WIDTH, PackingIndex.HEIGHT]]
            )
            * 2
            - 1
        )[None, ...]

        # In this mode, cls token has no position embedding
        if orig_pos_embed is not None:
            posemb = (
                orig_pos_embed[1:]
                .view(1, self.grid_size[0], self.grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            posemb = posemb.to(device=grid.device, dtype=grid.dtype)
            sample = F.grid_sample(
                posemb, grid, padding_mode="zeros"
            )  # padding tokens / class token will get zero for posemb
            sample = sample.view(-1, total_windows, window_size).permute(1, 2, 0).contiguous()
            sample = torch.where(
                idx[:, :, PackingIndex.IDX, None] == PackingIndex.ID_CLS_TOKEN,
                orig_pos_embed[0].view(1, 1, -1).to(device=sample.device, dtype=sample.dtype),
                sample,
            )

            new_pos_embed = sample.reshape(batch_size, token_per_image, -1)

            state_dict[prefix + "positional_embedding_vlm"] = new_pos_embed.squeeze(0)

        if return_state_dict:
            return state_dict

    def apply_class_embedding(self, x):
        x = torch.cat(
            [
                x,
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images shape: [X, 3, 336, 336], where X is num_images * num_tiles
        logger.info(
            f"VisionEncoder input images.shape {images.shape}"
        )  # torch.Size([5, 3, 336, 336])
        # NOTE: in Llama4 bsz=bsz*num_tiles, num_chunks=1
        assert images.ndim == 4, f"Llama4 vision encoder expects pixel_values of shape \
            [bsz*num_chunks, num_channel, h, w], but received {images.shape}"
        bsz, nch, h, w = images.shape
        num_concurrent_media = 1
        num_chunks = 1

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, h, w)
        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, h, w)
        x = self.conv1(x)  # output shape is [*, h//14=336//14=24 * 24, dim=1408]
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        if self.positional_embedding_vlm is not None:
            x = x + self.positional_embedding_vlm.to(x.dtype)

        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        x = self.layernorm_pre(x)
        # x shape is [bsz * num_concurrent_media, num_chunks * ntok, dim]
        x = x.view(bsz * num_concurrent_media, -1, dim)
        freq_cis = self.freq_cis.to(images.device)

        tf_output = self.transformer(
            x,
            freq_cis=freq_cis,
        )
        logger.info(
            f"after transformer tf_output.shape {tf_output.shape}"
        )  # torch.Size([5, 577, 1408])

        int_x = None
        if isinstance(tf_output, tuple):
            x, int_x = tf_output
        else:
            x = tf_output
        x = self.layernorm_post(x)

        # remove cls token output
        x = x[:, :-1, :]

        # add and output x + int_x features
        if int_x is not None:
            int_x = int_x[:, :-1, :, :]
            int_x = int_x.reshape(bsz * num_concurrent_media, ntok - 1, -1)
            x = torch.cat([x, int_x], dim=-1)

        logger.info(f"VisionEncoder returning x.shape {x.shape}")  # torch.Size([5, 576, 1408])
        return x


class PixelShuffle(nn.Module):
    def __init__(self, ps_ratio):
        super().__init__()
        self.ps_ratio = ps_ratio  # 0.5

    def forward(self, x):
        # x: [B, N, C], N = number of patches
        assert self.ps_ratio is not None, "ps_ratio is required for pixel shuffle"
        assert x.dim() == 3, "pixel shuffle requires encoded patches [B, N, C]"
        hh = ww = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], hh, ww, -1)
        # x shape is [B, H, W, C]
        x = pixel_shuffle_op(x, ps_ratio=self.ps_ratio)
        # x shape is [B, H*ps_ratio * W*ps_ratio, C/ps_ratio^2]
        pixel_shuffle_patches = x.reshape(x.shape[0], -1, x.shape[-1])
        return pixel_shuffle_patches


def pixel_shuffle_op(input_x, ps_ratio):
    """
    upsamples the spatial dimensions (width and height) while reducing the number of channels
    since ps_ratio < 1 here, it downsamples the spatial dims and increasing the number of channels
    """
    n, w, h, c = input_x.size()
    input_x = input_x.view(n, w, int(h * ps_ratio), int(c / ps_ratio))
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    input_x = input_x.view(
        n,
        int(h * ps_ratio),
        int(w * ps_ratio),
        int(c / (ps_ratio * ps_ratio)),
    )
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    return input_x


class SimpleMLP(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        act_layer: Callable = nn.GELU,
        dtype=torch.float32,
    ):
        super().__init__()
        # layers
        self.fc1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=bias,
            gather_output=False,
            dtype=dtype,
        )
        self.fc2 = RowParallelLinear(
            hidden_dim,
            hidden_dim,
            bias=bias,
            input_is_parallel=True,
            dtype=dtype,
            reduce_dtype=dtype,
        )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.non_linearity(self.fc2(hidden))


class PixelShuffleMLP(torch.nn.Module):
    def __init__(
        self,
        config: InferenceConfig,
    ):
        super().__init__()
        ps_ratio = config.vision_config.pixel_shuffle_ratio
        input_dim = config.vision_config.hidden_size  # 1408
        output_dim = config.vision_config.vision_output_dim  # 4096
        add_fc = False

        self.pixel_shuffle = PixelShuffle(ps_ratio)
        self.mlp = SimpleMLP(
            int(input_dim // (ps_ratio**2)),  # 1408 / 0.25 = 5632
            output_dim,  # 4096
            bias=False,
            dropout=0.0,
            act_layer=nn.GELU,
            dtype=config.neuron_config.torch_dtype,
        )
        self.fc = nn.Identity()
        if add_fc:
            self.fc = ColumnParallelLinear(
                output_dim, output_dim, bias=False, dtype=config.neuron_config.torch_dtype
            )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        # input shape torch.Size([5, 576, 1408])
        # downsample the spatial dimensions (width and height) while increasing the number of channels
        encoded_patches = self.pixel_shuffle(encoded_patches)
        logger.info(
            f"after pixel_shuffle encoded_patches {encoded_patches.shape} "
        )  # torch.Size([5, 144, 5632])
        out = self.mlp(encoded_patches)
        logger.info(f"after pixel_shuffle mlp {out.shape} ")  # torch.Size([5, 144, 4096])
        out = self.fc(out)
        logger.info(f"after pixel_shuffle fc {out.shape} ")  # torch.Size([5, 144, 4096])
        return out


class NeuronLlama4VisionEmbeddings(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = self.config.neuron_config
        self.vision_config = config.vision_config
        self.global_rank = SPMDRank(world_size=self.neuron_config.world_size)
        assert (
            self.neuron_config.world_size % self.neuron_config.tp_degree == 0
        ), "Invalid parallel config. world_size should be a multiple of tp_degree"
        self.dp_degree = self.neuron_config.world_size // self.neuron_config.tp_degree
        self.data_parallel_enabled = self.dp_degree > 1
        self.data_parallel_group = get_data_parallel_group()

        self.vision_encoder = VisionEncoder(config)
        self.vision_adapter = PixelShuffleMLP(config)
        self.vision_projection = nn.Linear(
            in_features=self.vision_config.vision_output_dim,   # 4096
            out_features=self.config.text_config.hidden_size,   # 5120
            bias=False,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )
        self.output_dim = self.vision_config.vision_output_dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: List[str] = None,
        unexpected_keys: List[str] = None,
        error_msgs: List[str] = None,
        return_state_dict: bool = False,
    ) -> None:
        """
        it reshapes the empty tensor to match the shape of the corresponding parameter in the original state dictionary
        """
        original_sd = self.state_dict()
        for k in state_dict:
            if (
                k.startswith(prefix)
                and len(state_dict[k].shape) == 1
                and state_dict[k].shape[0] == 0
            ):
                state_dict[k] = state_dict[k].reshape(original_sd[k[len(prefix) :]].shape)

    def _get_empty_sequence(self, h):
        return torch.zeros(
            h.shape[0],
            h.shape[1],
            self.output_dim,
            device=h.device,
            dtype=h.dtype,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate vision embeddings from flattened pixel values.

        This function handles dynamic image shapes as well as multiple images by splitting each image
        into a number of fixed-size chunks. Afterwards, all chunks are stacked together on the batch dimension (dim=0)

        Args:
        pixel_values (Tensor): Vision pixel values of shape [num_chunks, 1(constant), num_chunnels, image_size, image_size]

        Returns:
        vision embeddings (Tensor): Vision embeddings (after projection) padded to the nearest bucket size.

        """

        if self.data_parallel_enabled:
            dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.neuron_config.tp_degree)
            # split inputs along batch dim
            pixel_values = scatter_to_process_group_spmd(
                pixel_values,
                partition_dim=0,
                rank=dp_rank,
                process_group=self.data_parallel_group,
            )
        embedding = self.vision_encoder(pixel_values)

        logger.info(f"embedding.shape {embedding.shape}")  # torch.Size([5, 576, 1408])
        projected_embedding = self.vision_adapter(embedding)
        logger.info(
            f"projected_embedding.shape {projected_embedding.shape}"
        )  # torch.Size([5, 144, 4096])

        h_image_proj = self.vision_projection(projected_embedding)

        if self.data_parallel_enabled:
            h_image_proj = gather_from_tensor_model_parallel_region_with_dim(
                h_image_proj, gather_dim=0, process_group=self.data_parallel_group
            )

        logger.info(f"h_image.shape {h_image_proj.shape}")

        return h_image_proj


class Llama4VisionModelWrapper(ModelWrapper):
    """
    Neuron ModelWrapper class for NeuronLlama4VisionEmbeddings.
    Generates input shapes for trace and compilation. Disables bucketing.
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
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs
        )
        self.bucket_config = None  # TODO: add bucketing

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override ModelWrapper.input_generator().
        Generate a list of valid sample inputs containing one input list for each bucket.
        Different model may have a different set of input args.

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
        """
        inputs = []
        for bucket in self.neuron_config.buckets:
            num_chunks = bucket
            pixel_values = torch.ones(
                [
                    num_chunks,
                    self.config.vision_config.num_channels,
                    self.config.vision_config.image_size,
                    self.config.vision_config.image_size,
                ],
                dtype=self.config.neuron_config.torch_dtype
            )
            inputs.append((pixel_values,))

        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def forward(self, *args):
        """
        Override ModelWrapper.forward().
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        args = self.pad_inputs(*args)

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)
        output = self._forward(*args)

        return output

    def pad_inputs(self, *args):
        target_bucket = self.get_target_bucket(*args)
        logger.debug(f"Selected bucket {target_bucket} for vision encoder")
        padded_pixel_values = pad_image_tensor(args[0], target_bucket)
        args = (padded_pixel_values,)
        return args

    def get_target_bucket(self, *args, strategy="first_fit"):
        num_chunks = args[0].shape[0]
        for bucket in self.neuron_config.buckets:
            if num_chunks <= bucket:
                return bucket

        largest_bucket = self.neuron_config.buckets[-1]
        raise ValueError(
            f"num image chunks: {num_chunks} is larger than the largest bucket: {largest_bucket}"
        )


def mask_to_fixed_positions(image_mask, fixed_size=2448):
    """
    Convert binary image mask to fixed-size positions tensor with padding.

    Args:
        image_mask: Binary mask tensor of shape [1, sequence_length, 1]
        fixed_size: Fixed size for positions tensor (e.g., 7200)

    Returns:
        positions: Padded tensor of shape [1, fixed_size] where valid positions are actual indices
                and padding positions are filled with a safe padding value (e.g., 0)
    """
    batch_size = image_mask.size(0)
    device = image_mask.device

    # Initialize output tensor with padding value
    padded_positions = torch.zeros((batch_size, fixed_size), dtype=torch.long, device=device)

    # Get actual positions
    positions = image_mask.squeeze(-1).nonzero()[:, 1]  # Only take the sequence dimension indices
    actual_size = positions.size(0)

    # Fill actual positions
    padded_positions[0, :actual_size] = positions

    # Create valid_mask to track which positions are actually used
    valid_mask = torch.zeros((batch_size, fixed_size), dtype=torch.bool, device=device)
    valid_mask[0, :actual_size] = True

    return padded_positions, valid_mask


def scatter_embeddings_on_index(
    vision_chunks_bucket, h_image, encoded_patches_proj, positions, valid_mask
):
    """
    Scatter embeddings using fixed-size positions tensor and valid mask for Neuron compatibility.

    Args:
        image_flattened: Flattened image chunks
        h_image: Target tensor to scatter embeddings into
        encoded_patches_proj: Encoded image patches to be scattered
        positions: Fixed-size tensor of positions [batch_size, fixed_size]
        valid_mask: Boolean tensor indicating valid positions [batch_size, fixed_size]
    """
    num_images_per_sequence = [vision_chunks_bucket]

    assert not torch.isnan(encoded_patches_proj).any()
    assert sum(num_images_per_sequence) == encoded_patches_proj.size(0)

    encoded_patches_list = encoded_patches_proj.split(num_images_per_sequence, dim=0)

    for index in range(h_image.size(0)):
        encoded_patches_per_sample = encoded_patches_list[index]

        if encoded_patches_per_sample.numel() == 0:
            continue

        encoded_patches_per_sample = encoded_patches_per_sample.contiguous().view(
            -1, encoded_patches_per_sample.size(-1)
        )

        # Use the provided positions and valid mask
        current_positions = positions[index]
        current_valid_mask = valid_mask[index]

        # Only use positions where valid_mask is True
        valid_positions = current_positions[current_valid_mask]
        n_tokens_to_fill = len(valid_positions)

        assert n_tokens_to_fill <= encoded_patches_per_sample.size(0)

        h_image[index].scatter_(
            0,
            valid_positions.unsqueeze(-1).expand(-1, h_image.size(-1)),
            encoded_patches_per_sample[:n_tokens_to_fill],
        )

    return h_image


class NeuronLlama4ForImageEncoding(NeuronApplicationBase):
    """
    Neuron Application class for ViT image encoding case.
    Wraps NeuronViTModel with Neuron specific functionalities such as compile and load.
    """

    _model_cls = NeuronLlama4VisionEmbeddings

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        if self.neuron_config.enable_bucketing:
            if self.neuron_config.buckets is None:
                self.neuron_config.buckets = generate_llama4_vision_encoder_buckets(
                    self.neuron_config.dp_degree, VISION_MAX_NUM_CHUNKS
                )
        else:
            self.neuron_config.buckets = generate_buckets(
                VISION_MAX_NUM_CHUNKS, VISION_MAX_NUM_CHUNKS
            )

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        # will only have one model one tag
        # after compilation, in /tmp/nxd_model,
        # you should only see one folder called f"self._model_cls.__name__"
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return Llama4VisionModelWrapper

    def forward(self, pixel_values):
        return self.models[0](pixel_values)

    def get_compiler_args(self):
        # Flag for model type
        compiler_args = "-O1 --model-type=transformer"
        # Add flags for cc-overlap
        compiler_args += (
            # no --cc-pipeline-tiling-factor since the sequence length is an odd number 577
            " --tensorizer-options='--enable-ccop-compute-overlap'"
        )
        # Always prevent auto-casting
        compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
        logger.info(f"{self._model_cls.__name__} compiler_args: {compiler_args}")
        return compiler_args

    @staticmethod
    def load_hf_model(model_path):
        from transformers import Llama4VisionModel

        return Llama4VisionModel.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        if hasattr(inference_config, "vision_config"):
            inference_config = inference_config.vision_config

        # vision projection mapping:
        if "multi_modal_projector.linear_1.weight" in state_dict:
            state_dict["vision_projection.weight"] = state_dict.pop(
                "multi_modal_projector.linear_1.weight"
            )

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vision_model."):
                # TODO: fix layer norm name mapping
                vision_model_keys_mapping = {
                    ".patch_embedding.linear.": ".conv1._linear.",
                }
                for pattern, replacement in vision_model_keys_mapping.items():
                    if pattern in key:
                        key = key.replace(pattern, replacement)
                if ".vision_adapter." in key:
                    key = key.replace("vision_model.", "")
                else:
                    key = key.replace("vision_model.", "vision_encoder.")
                new_state_dict[key] = (
                    value.clone()
                    .detach()
                    .contiguous()
                    .to(inference_config.neuron_config.torch_dtype)
                )
            else:
                new_state_dict[key] = value.clone().detach().contiguous()
        new_state_dict["global_rank.rank"] = torch.arange(
            0, inference_config.neuron_config.world_size, dtype=torch.int32
        )

        del state_dict

        for layer in range(inference_config.num_hidden_layers):
            prefix = f"vision_encoder.model.layers.{layer}."
            new_prefix = f"vision_encoder.transformer.resblocks.{layer}."
            if inference_config.neuron_config.fused_qkv:
                q_weight = new_state_dict.pop(f"{prefix}self_attn.q_proj.weight")
                k_weight = new_state_dict.pop(f"{prefix}self_attn.k_proj.weight")
                v_weight = new_state_dict.pop(f"{prefix}self_attn.v_proj.weight")
                # the shape of weight matrix is 90-degree rotated, so here torch.cat along dim 0 instead of dim 1
                new_state_dict[f"{new_prefix}attn.qkv_proj.Wqkv.weight"] = torch.cat(
                    [q_weight, k_weight, v_weight], dim=0
                )
                q_bias = new_state_dict.pop(f"{prefix}self_attn.q_proj.bias")
                k_bias = new_state_dict.pop(f"{prefix}self_attn.k_proj.bias")
                v_bias = new_state_dict.pop(f"{prefix}self_attn.v_proj.bias")
                new_state_dict[f"{new_prefix}attn.qkv_proj.Wqkv.bias"] = torch.cat(
                    [q_bias, k_bias, v_bias], dim=0
                )
            else:
                new_state_dict[f"{new_prefix}attn.qkv_proj.q_proj.weight"] = new_state_dict.pop(
                    f"{prefix}self_attn.q_proj.weight"
                )
                new_state_dict[f"{new_prefix}attn.qkv_proj.k_proj.weight"] = new_state_dict.pop(
                    f"{prefix}self_attn.k_proj.weight"
                )
                new_state_dict[f"{new_prefix}attn.qkv_proj.v_proj.weight"] = new_state_dict.pop(
                    f"{prefix}self_attn.v_proj.weight"
                )
                new_state_dict[f"{new_prefix}attn.qkv_proj.q_proj.bias"] = new_state_dict.pop(
                    f"{prefix}self_attn.q_proj.bias"
                )
                new_state_dict[f"{new_prefix}attn.qkv_proj.k_proj.bias"] = new_state_dict.pop(
                    f"{prefix}self_attn.k_proj.bias"
                )
                new_state_dict[f"{new_prefix}attn.qkv_proj.v_proj.bias"] = new_state_dict.pop(
                    f"{prefix}self_attn.v_proj.bias"
                )

            # they will be renamed to {prefix}attn.o_proj.o_proj.* in GroupQueryAttention_O#preshard_hook
            new_state_dict[f"{new_prefix}attn.o_proj.weight"] = new_state_dict.pop(
                f"{prefix}self_attn.o_proj.weight"
            )
            new_state_dict[f"{new_prefix}attn.o_proj.bias"] = new_state_dict.pop(
                f"{prefix}self_attn.o_proj.bias"
            )

            # ln_1 and ln_2 mapping
            if f"{prefix}input_layernorm.weight" in new_state_dict:
                new_state_dict[f"{new_prefix}ln_1.weight"] = new_state_dict.pop(
                    f"{prefix}input_layernorm.weight"
                )
            if f"{prefix}input_layernorm.bias" in new_state_dict:
                new_state_dict[f"{new_prefix}ln_1.bias"] = new_state_dict.pop(
                    f"{prefix}input_layernorm.bias"
                )
            if f"{prefix}post_attention_layernorm.weight" in new_state_dict:
                new_state_dict[f"{new_prefix}ln_2.weight"] = new_state_dict.pop(
                    f"{prefix}post_attention_layernorm.weight"
                )
            if f"{prefix}post_attention_layernorm.bias" in new_state_dict:
                new_state_dict[f"{new_prefix}ln_2.bias"] = new_state_dict.pop(
                    f"{prefix}post_attention_layernorm.bias"
                )

            # mlp mapping: fc1 -> c_fc, fc2 -> c_proj
            if f"{prefix}mlp.fc1.weight" in new_state_dict:
                new_state_dict[f"{new_prefix}mlp.c_fc.weight"] = new_state_dict.pop(
                    f"{prefix}mlp.fc1.weight"
                )
            if f"{prefix}mlp.fc1.bias" in new_state_dict:
                new_state_dict[f"{new_prefix}mlp.c_fc.bias"] = new_state_dict.pop(
                    f"{prefix}mlp.fc1.bias"
                )
            if f"{prefix}mlp.fc2.weight" in new_state_dict:
                new_state_dict[f"{new_prefix}mlp.c_proj.weight"] = new_state_dict.pop(
                    f"{prefix}mlp.fc2.weight"
                )
            if f"{prefix}mlp.fc2.bias" in new_state_dict:
                new_state_dict[f"{new_prefix}mlp.c_proj.bias"] = new_state_dict.pop(
                    f"{prefix}mlp.fc2.bias"
                )

        def get_hw(size):
            from types import SimpleNamespace

            if isinstance(size, dict):
                height = size["height"]
                width = size["width"]
            elif isinstance(size, SimpleNamespace):
                height = size.height
                width = size.width
            elif isinstance(size, int):
                height = width = size
            else:
                raise TypeError(f"Size is of invalid type {type(size)}")
            return height, width

        # Convert unfold+linear to conv2d
        kernel_size = get_hw(inference_config.patch_size)
        conv1_linear_weight = new_state_dict.pop("vision_encoder.conv1._linear.weight")
        in_channels = conv1_linear_weight.shape[1] // (kernel_size[0] * kernel_size[1])
        new_shape = (-1, in_channels, kernel_size[0], kernel_size[1])
        new_state_dict["vision_encoder.conv1.conv.weight"] = conv1_linear_weight.reshape(new_shape)

        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        from neuronx_distributed_inference.models.llama4.modeling_llama4 import Llama4InferenceConfig
        return Llama4InferenceConfig
