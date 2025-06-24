import enum
import logging
from typing import Optional, Tuple

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads
from neuronx_distributed.quantization.quantization_layers import BaseQuantizeParallelLinear
from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_kernel, rmsnorm_qkv_isa_fused_add_kernel
from neuronxcc.nki.compiler.backends.neuron.dimensions import CCPipeline  # noqa: N813
from neuronxcc.nki.language import nc
from torch import nn
from torch.distributed import ProcessGroup
from torch.nn import functional as F
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

from neuronx_distributed_inference.modules.attention.utils import transpose_parallel_linear_layer
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module

logger = logging.getLogger("Neuron")

_traced_qkv_kernel = nki_jit()(rmsnorm_qkv_isa_kernel)
_traced_qkv_kernel_fused_add = nki_jit()(rmsnorm_qkv_isa_fused_add_kernel)

try:
    from neuronxcc.nki._pre_prod_kernels.output_proj import output_proj_kernel
    _traced_o_proj_kernel = nki_jit()(output_proj_kernel)
except ImportError:
    logger.warning(
        "Use a more recent neuron compiler version to enable output projection kernel"
    )
    _traced_o_proj_kernel = None


class GQA(enum.Enum):
    # This transforms a GQA attention mechanism into a traditional MHA mechanism
    # by replicating the K/V heads to evenly match the corresponding Q heads.
    # This consumes more memory than would otherwise be used with other sharding
    # mechanisms but works in all cases.
    # Example:
    # tp_degree = 32
    # num_attention_heads: 56 -> 64
    # num_kev_value_heads: 8  -> 56
    # adding 8 padding ranks, (inclusive) from 57 to 64
    # | KV1 KV1 | KV1 KV1 | ... | KV8 KV8 | Pad1 Pad1 | ... | Pad8 Pad8 |
    # | Q1  Q2  | Q3  Q4  | ... | Q55 Q56 | Pad1 Pad1 | ... | Pad8 Pad8 |
    CONVERT_TO_MHA = "convert-to-mha"

    # This transforms a GQA attention mechanism such that there is exactly
    # one K/V head per tp_degree through replication e.g. 8 K/V heads with
    # tp_degree=32 results in 32 K/V heads. This is more memory efficient but
    # does not work for all configurations since
    # tp_degree % initial_num_kev_value_heads != 0 can only be padded at the end
    # Q heads are padded interleaved to retain correct alignment between Q and K/V heads.
    # Example:
    # tp_degree = 32
    # num_attention_heads: 56 -> 64
    # num_kev_value_heads: 8  -> 32
    # adding 8 padding ranks, one every 8th rank
    # | KV1   | KV1   | KV1   | KV1     | KV2   | ... | KV2   | | KV8     |
    # | Q1 Q2 | Q3 Q4 | Q5 Q6 | Q7 Pad1 | Q8 Q9 | ... | Q5 Q6 | | Q7 Pad8 |
    REPLICATE_TO_TP_DEGREE = "replicate-to-tp-degree"


def determine_sharding_strategy(
    tp_degree: int, source_key_value_heads: int, desired_sharding_strategy: Optional[GQA] = None
) -> GQA:
    sharding_strategy = (
        desired_sharding_strategy if desired_sharding_strategy else GQA.REPLICATE_TO_TP_DEGREE
    )

    if sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE and (
        tp_degree % source_key_value_heads != 0
    ):
        logger.warning(f"TP degree ({tp_degree}) and KV heads ({source_key_value_heads}) are not divisible. Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!")
        sharding_strategy = GQA.CONVERT_TO_MHA

    return sharding_strategy


def get_shardable_head_counts(
    tp_degree: int, num_attention_heads: int, num_key_value_heads: int, sharding_strategy: GQA
) -> Tuple[int, int]:
    # Pad attention heads
    updated_num_attention_heads = num_attention_heads + get_number_of_extra_heads(
        num_attention_heads, tp_degree
    )

    # Replicate and pad K/V heads
    updated_num_key_value_heads = num_key_value_heads
    if num_attention_heads == num_key_value_heads:  # MHA
        updated_num_key_value_heads = updated_num_attention_heads
    else:  # GQA / MQA
        if (num_key_value_heads < tp_degree) or (num_key_value_heads % tp_degree != 0):
            if sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
                assert (
                    tp_degree % num_key_value_heads == 0
                ), "GQA.REPLICATE_TO_TP_DEGREE requires tp_degree to be divisible by num_key_value_heads"
                updated_num_key_value_heads = tp_degree
            elif sharding_strategy == GQA.CONVERT_TO_MHA:
                updated_num_key_value_heads = updated_num_attention_heads

    return updated_num_attention_heads, updated_num_key_value_heads


def is_per_channel(scale: torch.Tensor) -> bool:
    """See if the scale is per channel"""
    if scale.shape == (1,):
        return False
    return True


def get_tensor_per_channel_scale_axis(scale: torch.Tensor) -> int:
    """Get the channel axis for the per channel scale"""
    scale_shape = scale.shape
    # Only one dimension would have scale values
    for i, dim_length in enumerate(scale_shape):
        if dim_length > 1:
            return i
    raise RuntimeError(f"Cannot get channel axis for the scale: {scale}")


def should_pad_scale(tensor_scale: torch.Tensor, pad_dim: int) -> bool:
    """Should scale be padded"""
    if (
        (tensor_scale is not None)
        and (is_per_channel(tensor_scale))
        and (get_tensor_per_channel_scale_axis(tensor_scale) == pad_dim)
    ):
        return True
    return False


def verify_scale_dimension(tensor: torch.Tensor, tensor_scale: torch.Tensor):
    if is_per_channel(tensor_scale):
        channel_axis = get_tensor_per_channel_scale_axis(scale=tensor_scale)
        assert tensor_scale.shape[channel_axis] == tensor.shape[channel_axis]


def maybe_pad_interleaved(
    tensor,
    pad_dim: int,
    source_heads: int,
    target_heads: int,
    source_group_size: int,
    tensor_scale: torch.Tensor = None,
):
    tensor = _maybe_pad_interleaved(tensor, pad_dim, source_heads, target_heads, source_group_size)
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=pad_dim):
        tensor_scale = _maybe_pad_interleaved(
            tensor_scale, pad_dim, source_heads, target_heads, source_group_size
        )

    return tensor, tensor_scale


def _maybe_pad_interleaved(
    tensor, pad_dim: int, source_heads: int, target_heads: int, source_group_size: int
):
    if tensor is None:
        return tensor

    # Why we convert FP8 tensor to bfloat16?
    # Torch does not support torch.cat, or torch.zeros (for large dimensions) for f8e4m3/f8e5m2
    # So we cast it to bfloat16, perform padding, and then recast back to f8e4m3/f8e5m2
    recast_dtype = None
    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        recast_dtype = tensor.dtype
        tensor = tensor.to(torch.bfloat16)

    shape = (
        tensor.shape[:pad_dim]
        + (source_heads, tensor.shape[pad_dim] // source_heads)
        + tensor.shape[pad_dim + 1 :]
    )
    tensor = tensor.view(shape)

    splits = torch.split(tensor, source_group_size, dim=pad_dim)

    pad_size = list(splits[0].size())
    pad_size[pad_dim] = (target_heads - source_heads) // (source_heads // source_group_size)
    pads = [torch.zeros(pad_size, dtype=tensor.dtype)] * len(splits)

    interleaved = [t for pair in zip(splits, pads) for t in pair]
    tensor = torch.cat(interleaved, dim=pad_dim)

    shape = (
        tensor.shape[:pad_dim]
        + (tensor.shape[pad_dim] * tensor.shape[pad_dim + 1],)
        + tensor.shape[pad_dim + 2 :]
    )

    if recast_dtype is not None:
        tensor = tensor.to(recast_dtype)

    return tensor.view(shape)


def maybe_pad_tail(tensor, source_heads: int, target_heads: int, pad_dim: int, tensor_scale=None):
    tensor = _maybe_pad_tail(tensor, source_heads, target_heads, pad_dim)
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=pad_dim):
        tensor_scale = _maybe_pad_tail(tensor_scale, source_heads, target_heads, pad_dim)
    return tensor, tensor_scale


def _maybe_pad_tail(tensor, source_heads: int, target_heads: int, pad_dim: int):
    if tensor is None:
        return tensor
    size_to_pad = int(
        (tensor.shape[pad_dim] // source_heads) * target_heads - tensor.shape[pad_dim]
    )

    dims_after_pad_dim = len(tensor.size()) - pad_dim
    pad_length = dims_after_pad_dim * 2
    pad = (0,) * (pad_length - 1) + (size_to_pad,)

    return F.pad(tensor, pad)


def replicate_kv(tensor, source_heads: int, repeats: int, head_dim=0, tensor_scale=None):
    tensor = _replicate_kv(
        tensor=tensor, source_heads=source_heads, repeats=repeats, head_dim=head_dim
    )
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=head_dim):
        tensor_scale = _replicate_kv(
            tensor=tensor_scale, source_heads=source_heads, repeats=repeats, head_dim=head_dim
        )
    return tensor, tensor_scale


def _replicate_kv(tensor, source_heads: int, repeats: int, head_dim=0):
    if tensor is None:
        return tensor
    shape = (
        tensor.shape[:head_dim]
        + (source_heads, tensor.shape[head_dim] // source_heads)
        + tensor.shape[head_dim + 1 :]
    )
    tensor = tensor.view(shape)
    tensor = torch.repeat_interleave(tensor, repeats=repeats, dim=head_dim)
    shape = (
        tensor.shape[:head_dim]
        + (tensor.shape[head_dim] * tensor.shape[head_dim + 1],)
        + tensor.shape[head_dim + 2 :]
    )
    return tensor.view(shape)


class BaseGroupQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ):
        super().__init__()

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
        elif parallel_state.model_parallel_is_initialized():
            self.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.tensor_model_parallel_group = None

        if tensor_model_parallel_group:
            if tp_degree == 1:
                # update default value
                tp_degree = tensor_model_parallel_group.size()
            else:
                assert (
                    tp_degree == self.tensor_model_parallel_group.size()
                ), f"TP Degree {tp_degree} and tensor model parallel group size {self.tensor_model_parallel_group.size()} does not match"

        self.hidden_size = hidden_size
        self.tp_degree = tp_degree
        self.head_dim = head_dim
        self.dtype = dtype
        self.bias = bias
        self._src_num_attention_heads = num_attention_heads
        self._src_num_key_value_heads = num_key_value_heads

        self.sharding_strategy = determine_sharding_strategy(
            tp_degree,
            self._src_num_key_value_heads,
            desired_sharding_strategy=desired_sharding_strategy,
        )
        self.num_attention_heads, self.num_key_value_heads = get_shardable_head_counts(
            tp_degree,
            self._src_num_attention_heads,
            self._src_num_key_value_heads,
            self.sharding_strategy,
        )

    def get_sharding_strategy(self) -> GQA:
        return self.sharding_strategy

    def get_num_attention_heads(self) -> int:
        return self.num_attention_heads

    def get_num_key_value_heads(self) -> int:
        return self.num_key_value_heads

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        raise NotImplementedError

    def replace_prefixes(self, old_prefix, new_prefix, model_state_dict):
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix)
                new_keys.append(new_key)
                old_keys.append(key)

        for key_index in range(len(old_keys)):
            model_state_dict[new_keys[key_index]] = model_state_dict[old_keys[key_index]]


class GroupQueryAttention_QKV(BaseGroupQueryAttention):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        gather_output: bool = True,
        fused_qkv: bool = False,
        clip_qkv: Optional[float] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        rms_norm_eps: float = 1e-6,
        qkv_kernel_enabled: bool = False,
        fused_rmsnorm_skip_gamma: bool = False,
        tiling_factor: int = 1,
        seq_len_threshold_for_cc_tiling: int = 16834,
        logical_nc_config: int = 1,
        qkv_kernel_nbsd_layout: bool = False,
        on_cpu: bool = False,
        rank_ordering: dict = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            tp_degree=tp_degree,
            dtype=dtype,
            bias=bias,
            desired_sharding_strategy=desired_sharding_strategy,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )
        if fused_qkv and gather_output:
            raise ValueError(
                "Gathering states followed by fused qkv is not allowed as it has a different weight sharding scheme."
            )

        self.gather_output = gather_output
        self.fused_qkv = fused_qkv
        self.clip_qkv = clip_qkv

        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_dimension = sequence_dimension
        self.rms_norm_eps = rms_norm_eps
        self.qkv_kernel_enabled = qkv_kernel_enabled
        self.fused_rmsnorm = not self.sequence_parallel_enabled
        self.fused_rmsnorm_skip_gamma = fused_rmsnorm_skip_gamma and self.fused_rmsnorm
        self.tiling_factor = tiling_factor
        self.seq_len_threshold_for_cc_tiling = seq_len_threshold_for_cc_tiling
        self.logical_nc_config = logical_nc_config
        self.qkv_kernel_nbsd_layout = qkv_kernel_nbsd_layout
        self.rank_ordering = rank_ordering

        if self.tensor_model_parallel_group is not None:
            if self.fused_qkv:
                self.Wqkv = ColumnParallelLinear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                    rank_ordering=rank_ordering,
                )
                if self.qkv_kernel_enabled:
                    # we need to transpose the weights on the CPU side to avoid
                    # needing to transpose on the device when using QKV kernel
                    self.Wqkv.weight = transpose_parallel_linear_layer(self.Wqkv.weight)

                # Set heads info as weight parameter attributes to be used in weights sharding
                setattr(self.Wqkv.weight, "fused_qkv", True)
                setattr(self.Wqkv.weight, "num_attention_heads", self.num_attention_heads)
                setattr(self.Wqkv.weight, "num_key_value_heads", self.num_key_value_heads)
                setattr(self.Wqkv.weight, "head_dim", self.head_dim)
                if self.bias:
                    setattr(self.Wqkv.bias, "fused_qkv", True)
                    setattr(self.Wqkv.bias, "num_attention_heads", self.num_attention_heads)
                    setattr(self.Wqkv.bias, "num_key_value_heads", self.num_key_value_heads)
                    setattr(self.Wqkv.bias, "head_dim", self.head_dim)

            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_attention_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                    rank_ordering=rank_ordering,
                )
                self.k_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                    rank_ordering=rank_ordering,
                )
                self.v_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                    sequence_parallel_enabled=False,
                    tensor_model_parallel_group=self.tensor_model_parallel_group,
                    rank_ordering=rank_ordering,
                )
        else:
            if self.fused_qkv:
                self.Wqkv = nn.Linear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                )
            else:
                self.q_proj = nn.Linear(
                    self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.bias
                )
                self.k_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )
                self.v_proj = nn.Linear(
                    self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias
                )

    def forward(self, hidden_states: torch.Tensor, rmsnorm=None, adapter_ids=None, residual=None):
        if self.sequence_parallel_enabled and self.tensor_model_parallel_group is not None:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
                tile_cc=self.tiling_factor > 1,
            )

        if self.qkv_kernel_enabled:
            assert self.fused_qkv, "QKV kernel only supported when fused_qkv is TRUE"
            return self._kernel_qkv_forward(hidden_states, rmsnorm, residual)
        else:
            Q, K, V = self._native_qkv_forward(hidden_states, adapter_ids)
        return Q, K, V, residual

    def _native_qkv_forward(self, hidden_states: torch.Tensor, adapter_ids=None):
        if self.fused_qkv:
            logger.debug("QKV: native compiler")
            QKV = (
                self.Wqkv(hidden_states)
                if not is_lora_module(self.Wqkv)
                else self.Wqkv(hidden_states, adapter_ids)
            )
            return self._split_fused_qkv(QKV)
        else:
            Q = (
                self.q_proj(hidden_states)
                if not is_lora_module(self.q_proj)
                else self.q_proj(hidden_states, adapter_ids)
            )
            K = (
                self.k_proj(hidden_states)
                if not is_lora_module(self.k_proj)
                else self.k_proj(hidden_states, adapter_ids)
            )
            V = (
                self.v_proj(hidden_states)
                if not is_lora_module(self.v_proj)
                else self.v_proj(hidden_states, adapter_ids)
            )
            if self.clip_qkv is not None:
                Q = Q.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                K = K.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                V = V.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            return Q, K, V

    def _split_fused_qkv(self, QKV):
        logger.debug(f"Fused QKV tensor has shape {QKV.shape}")
        if self.clip_qkv is not None:
            QKV = QKV.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        # shape of QKV is [batch, seqlen, fused_qkv_size]
        # we split the fused QKV (dim=2) into Q, K, V
        # for example:
        #   for 405B, TP=128, num_att_heads=128
        #   LNC=2/TP=64 will split QKV from [batch, seqlen, 512] into:
        #   Q [batch, seqlen, 256]
        #   K [batch, seqlen, 128]
        #   V [batch, seqlen, 128]
        # torch.split has accuracy issue and leads to more reshapes in hlo.
        # Using torch.tensor_split here. NAPP-3145
        q_end_index = self.num_attention_heads * self.head_dim // self.tp_degree
        k_end_index = q_end_index + self.num_key_value_heads * self.head_dim // self.tp_degree
        Q, K, V = torch.tensor_split(
            QKV,
            (
                q_end_index,
                k_end_index,
                # rest of the QKV will go to V output
            ),
            dim=2,
        )
        logger.debug(f"QKV shape before tensor_split: {QKV.shape}")
        logger.debug(f"Q shape after tensor_split: {Q.shape}")
        logger.debug(f"K shape after tensor_split: {K.shape}")
        logger.debug(f"V shape after tensor_split: {V.shape}")
        return Q, K, V

    def _kernel_qkv_forward(self, hidden_states, rmsnorm, residual):
        logger.debug(
            f"QKV kernel: fused_rmsnorm={self.fused_rmsnorm}, skip_gamma={self.fused_rmsnorm_skip_gamma} logical_nc_config={self.logical_nc_config}"
        )
        bs, seqlen, h = hidden_states.shape

        padded_seqlen = seqlen
        # The QKV kernel we are calling here is a unified kernel,
        # underneath there are two separate kernel implementations: TKG & CTE
        # We only want to pad the sequence length for the CTE, which has batch * original seqlen > 64
        if bs * seqlen > 64 and seqlen % 2 != 0:
            logger.debug("For the CTE QKV kernel pad the sequence length to the next even number")
            hidden_states = F.pad(
                hidden_states,
                pad=(
                    0,
                    0,  # hidden_dim: no padding
                    0,
                    1,  # seq_dim: pad 1 at end
                    0,
                    0,
                ),  # batch_dim: no padding
                value=1.0,  # pad non-zeros to avoid possible numerical issues
            )
            padded_seqlen = seqlen + 1

        h2, fused_qkv_size = self.Wqkv.weight.shape
        logger.debug(
            f"fused QKV projection weight - shape: {self.Wqkv.weight.shape}, dtype: {self.Wqkv.weight.dtype}"
        )

        # shape checks
        n = (self.num_attention_heads + 2 * self.num_key_value_heads) // self.tp_degree
        assert fused_qkv_size == n * self.head_dim
        assert h == h2

        fused_rmsnorm = self.fused_rmsnorm and rmsnorm is not None

        norm_weights = None
        if fused_rmsnorm:
            # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
            norm_weights = rmsnorm.weight.unsqueeze(0)
            assert norm_weights.shape == (1, h)

        if seqlen <= self.seq_len_threshold_for_cc_tiling:  # Keep regular grid for TKG. Messes up the impl
            grid = (nc(self.logical_nc_config),)
        else:  # Add CC pipelining dim for CTE kernel grid
            grid = (CCPipeline(self.tiling_factor) * nc(self.logical_nc_config),)

        if self.qkv_kernel_nbsd_layout:
            QKV = torch.zeros(
                n,
                bs,
                padded_seqlen,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            if seqlen <= self.seq_len_threshold_for_cc_tiling:  # Keep regular grid for TKG. Messes up the impl
                grid = (nc(self.logical_nc_config),)
            else:  # Add CC pipelining dim for CTE kernel grid
                grid = (CCPipeline(self.tiling_factor) * nc(self.logical_nc_config),)

            if residual is not None:
                # attn_out is set to zeros becauses we getting the residual from fused-add-MLP directly
                # TODO: remove this useless field from qkv kenrel
                zeros = torch.zeros(
                    residual.shape,
                    dtype=residual.dtype,
                    device=residual.device,
                )
                # the residual result before QKV is stored back to hidden_states (input0) such that it
                # can be used by fused-add-MLP later
                _traced_qkv_kernel_fused_add[grid](
                    hidden_states,
                    residual,
                    zeros,
                    self.Wqkv.weight,
                    (
                        norm_weights
                        if norm_weights is not None
                        # cannot pass None to this kernel API
                        else torch.ones((1, h), device=hidden_states.device)
                    ),
                    QKV,
                    eps=self.rms_norm_eps,
                    kernel_name="QKV",
                    # Run RMSNorm inside the kernel if NOT using SP norm
                    fused_rmsnorm=fused_rmsnorm,
                    # required shape: [1, hidden(sharded)]
                    bias=self.Wqkv.bias.unsqueeze(0) if self.bias else None,
                    skip_gamma=self.fused_rmsnorm_skip_gamma,
                )
                residual = hidden_states  # store residual for MLP
            else:
                _traced_qkv_kernel[grid](
                    hidden_states,
                    self.Wqkv.weight,
                    (
                        norm_weights
                        if norm_weights is not None
                        # cannot pass None to this kernel API
                        else torch.ones((1, h), device=hidden_states.device)
                    ),
                    QKV,
                    kernel_name="QKV",
                    eps=self.rms_norm_eps,
                    # Run RMSNorm inside the kernel if NOT using SP norm
                    fused_rmsnorm=fused_rmsnorm,
                    # required shape: [1, hidden(sharded)]
                    bias=self.Wqkv.bias.unsqueeze(0) if self.bias else None,
                    skip_gamma=self.fused_rmsnorm_skip_gamma,
                    use_dma_transpose=True,
                )

            assert QKV.shape == (n, bs, padded_seqlen, self.head_dim)

            # switch from:
            #   output layout: [n, b, s, d]
            #             dim:  0  1  2  3
            # back to original layout:
            #   output layout: [b, s, n*d]
            QKV = (
                QKV.permute(1, 2, 0, 3)  # after permute: batch, padded_seqlen, num_heads, d_head
                .reshape(bs, padded_seqlen, fused_qkv_size)
                .to(hidden_states.dtype)
            )

        else:
            assert residual is None, "fused_add_qkv only support for nbsd layout"
            QKV = torch.zeros(
                bs,
                padded_seqlen,
                fused_qkv_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # the QKV kernel will automatically switch to the TKG QKV if seqlen==1
            _traced_qkv_kernel[grid](
                hidden_states,
                self.Wqkv.weight,
                (
                    norm_weights
                    if norm_weights is not None
                    # cannot pass None to this kernel API
                    else torch.ones((1, h), device=hidden_states.device)
                ),
                QKV,
                kernel_name="QKV",
                eps=self.rms_norm_eps,
                # Run RMSNorm inside the kernel if NOT using SP norm
                fused_rmsnorm=fused_rmsnorm,
                # required shape: [1, hidden(sharded)]
                bias=self.Wqkv.bias.unsqueeze(0) if self.bias else None,
                skip_gamma=self.fused_rmsnorm_skip_gamma,
            )

        # unpad the last token if it's needed
        if padded_seqlen != seqlen:
            QKV = QKV[:, :-1, :]
        assert QKV.shape == (bs, seqlen, fused_qkv_size)

        return (*self._split_fused_qkv(QKV), residual)

    def get_weight(
        self, prefix: str, layer: torch.nn.Module, layer_name, model_state_dict: dict
    ) -> Tuple[torch.Tensor]:
        if hasattr(layer, "get_weight_from_state_dict"):
            weight = layer.get_weight_from_state_dict(
                prefix=f"{prefix}.{layer_name}.", state_dict=model_state_dict
            )
            if isinstance(layer, BaseQuantizeParallelLinear):
                scale = layer.get_scale_from_state_dict(
                    prefix=f"{prefix}.{layer_name}.", state_dict=model_state_dict
                )
            else:
                scale = None
        else:
            weight = model_state_dict[f"{prefix}.{layer_name}.weight"]
            if isinstance(layer, BaseQuantizeParallelLinear):
                scale = model_state_dict[f"{prefix}.{layer_name}.scale"]
            else:
                scale = None
        return weight, scale

    def get_bias(
        self, prefix: str, layer: torch.nn.Module, layer_name: str, model_state_dict: dict
    ) -> Tuple[torch.Tensor]:
        if hasattr(layer, "get_bias_from_state_dict"):
            bias = layer.get_bias_from_state_dict(
                prefix=f"{prefix}.{layer_name}.", state_dict=model_state_dict
            )
        else:
            bias = model_state_dict.get(f"{prefix}.{layer_name}.bias")
        return bias

    def set_weight(
        self,
        tensor: torch.Tensor,
        prefix: str,
        layer: torch.nn.Module,
        layer_name,
        model_state_dict: dict,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        # TODO: set weight to state dict support is pending.
        model_state_dict[f"{prefix}.{layer_name}.weight"] = tensor
        if scale is not None:
            model_state_dict[f"{prefix}.{layer_name}.scale"] = scale
            verify_scale_dimension(tensor=tensor, tensor_scale=scale)

    def set_bias(
        self,
        tensor: torch.Tensor,
        prefix: str,
        layer: torch.nn.Module,
        layer_name: str,
        model_state_dict: dict,
    ) -> Tuple[torch.Tensor]:
        if hasattr(layer, "set_bias_to_state_dict"):
            layer.set_bias_to_state_dict(
                prefix=f"{prefix}.{layer_name}.", tensor=tensor, state_dict=model_state_dict
            )
        else:
            model_state_dict[f"{prefix}.{layer_name}.bias"] = tensor

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        prefix_parts = prefix.split(".")
        prefix = ".".join(prefix_parts[:-1])
        hf_prefix = ".".join(prefix_parts[:-2])
        if self.fused_qkv:
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.Wqkv",
                new_prefix=f"{prefix}.Wqkv",
                model_state_dict=model_state_dict,
            )
            qkv_weight, qkv_scale = self.get_weight(
                prefix=prefix, layer=self.Wqkv, layer_name="Wqkv", model_state_dict=model_state_dict
            )
            q_proj_weight, k_proj_weight, v_proj_weight = qkv_weight.split(
                [
                    self._src_num_attention_heads * self.head_dim,
                    self._src_num_key_value_heads * self.head_dim,
                    self._src_num_key_value_heads * self.head_dim,
                ],
                dim=0,
            )

            if qkv_scale is not None:
                q_proj_scale, k_proj_scale, v_proj_scale = qkv_scale.split(
                    [
                        self._src_num_attention_heads * self.head_dim,
                        self._src_num_key_value_heads * self.head_dim,
                        self._src_num_key_value_heads * self.head_dim,
                    ],
                    dim=0,
                )
            else:
                q_proj_scale, k_proj_scale, v_proj_scale = None, None, None

            qkv_bias = self.get_bias(
                prefix=prefix, layer=self.Wqkv, layer_name="Wqkv", model_state_dict=model_state_dict
            )
            if qkv_bias is not None:
                q_proj_bias, k_proj_bias, v_proj_bias = qkv_bias.split(
                    [
                        self._src_num_attention_heads * self.head_dim,
                        self._src_num_key_value_heads * self.head_dim,
                        self._src_num_key_value_heads * self.head_dim,
                    ],
                    dim=0,
                )
            else:
                q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.q_proj",
                new_prefix=f"{prefix}.q_proj",
                model_state_dict=model_state_dict,
            )
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.k_proj",
                new_prefix=f"{prefix}.k_proj",
                model_state_dict=model_state_dict,
            )
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.v_proj",
                new_prefix=f"{prefix}.v_proj",
                model_state_dict=model_state_dict,
            )

            q_proj_weight, q_proj_scale = self.get_weight(
                prefix=prefix,
                layer=self.q_proj,
                layer_name="q_proj",
                model_state_dict=model_state_dict,
            )
            k_proj_weight, k_proj_scale = self.get_weight(
                prefix=prefix,
                layer=self.k_proj,
                layer_name="k_proj",
                model_state_dict=model_state_dict,
            )
            v_proj_weight, v_proj_scale = self.get_weight(
                prefix=prefix,
                layer=self.v_proj,
                layer_name="v_proj",
                model_state_dict=model_state_dict,
            )

            q_proj_bias = self.get_bias(
                prefix=prefix,
                layer=self.q_proj,
                layer_name="q_proj",
                model_state_dict=model_state_dict,
            )
            k_proj_bias = self.get_bias(
                prefix=prefix,
                layer=self.k_proj,
                layer_name="k_proj",
                model_state_dict=model_state_dict,
            )
            v_proj_bias = self.get_bias(
                prefix=prefix,
                layer=self.v_proj,
                layer_name="v_proj",
                model_state_dict=model_state_dict,
            )

        if self.num_key_value_heads != self._src_num_key_value_heads:
            if self.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
                repeats = self.tp_degree // self._src_num_key_value_heads
            elif self.sharding_strategy == GQA.CONVERT_TO_MHA:
                repeats = self._src_num_attention_heads // self._src_num_key_value_heads
            k_proj_weight, k_proj_scale = replicate_kv(
                k_proj_weight,
                source_heads=self._src_num_key_value_heads,
                repeats=repeats,
                head_dim=0,
                tensor_scale=k_proj_scale,
            )
            k_proj_bias, _ = replicate_kv(
                k_proj_bias, source_heads=self._src_num_key_value_heads, repeats=repeats, head_dim=0
            )
            v_proj_weight, v_proj_scale = replicate_kv(
                v_proj_weight,
                source_heads=self._src_num_key_value_heads,
                repeats=repeats,
                head_dim=0,
                tensor_scale=v_proj_scale,
            )
            v_proj_bias, _ = replicate_kv(
                v_proj_bias, source_heads=self._src_num_key_value_heads, repeats=repeats, head_dim=0
            )

        if self.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
            q_proj_weight, q_proj_scale = maybe_pad_interleaved(
                q_proj_weight,
                pad_dim=0,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                source_group_size=self._src_num_attention_heads // self._src_num_key_value_heads,
                tensor_scale=q_proj_scale,
            )
            q_proj_bias, _ = maybe_pad_interleaved(
                q_proj_bias,
                pad_dim=0,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                source_group_size=self._src_num_attention_heads // self._src_num_key_value_heads,
            )

        if self.sharding_strategy == GQA.CONVERT_TO_MHA:
            q_proj_weight, q_proj_scale = maybe_pad_tail(
                q_proj_weight,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                pad_dim=0,
                tensor_scale=q_proj_scale,
            )
            q_proj_bias, _ = maybe_pad_tail(
                q_proj_bias,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                pad_dim=0,
            )
            k_proj_weight, k_proj_scale = maybe_pad_tail(
                k_proj_weight,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
                tensor_scale=k_proj_scale,
            )
            k_proj_bias, _ = maybe_pad_tail(
                k_proj_bias,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
            )
            v_proj_weight, v_proj_scale = maybe_pad_tail(
                v_proj_weight,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
                tensor_scale=v_proj_scale,
            )
            v_proj_bias, _ = maybe_pad_tail(
                v_proj_bias,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
            )

        if self.fused_qkv:
            qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
            qkv_scale = None
            if all(scale is not None for scale in (q_proj_scale, k_proj_scale, v_proj_scale)):
                qkv_scale = torch.cat([q_proj_scale, k_proj_scale, v_proj_scale], dim=0)

            # Set heads info as weight parameter attributes to be used in weights sharding
            fused_qkv_params = (
                [self.Wqkv.weight, self.Wqkv.scale] if qkv_scale is not None else [self.Wqkv.weight]
            )
            for param in fused_qkv_params:
                setattr(param, "fused_qkv", True)
                setattr(param, "num_attention_heads", self.num_attention_heads)
                setattr(param, "num_key_value_heads", self.num_key_value_heads)
                setattr(param, "head_dim", self.head_dim)

            self.set_weight(
                tensor=qkv_weight,
                prefix=prefix,
                layer=self.Wqkv,
                layer_name="Wqkv",
                model_state_dict=model_state_dict,
                scale=qkv_scale,
            )
            if self.bias:
                qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
                self.set_bias(
                    tensor=qkv_bias,
                    prefix=prefix,
                    layer=self.Wqkv,
                    layer_name="Wqkv",
                    model_state_dict=model_state_dict,
                )
        else:
            self.set_weight(
                tensor=q_proj_weight,
                prefix=prefix,
                layer=self.q_proj,
                layer_name="q_proj",
                model_state_dict=model_state_dict,
                scale=q_proj_scale,
            )
            self.set_weight(
                tensor=k_proj_weight,
                prefix=prefix,
                layer=self.k_proj,
                layer_name="k_proj",
                model_state_dict=model_state_dict,
                scale=k_proj_scale,
            )
            self.set_weight(
                tensor=v_proj_weight,
                prefix=prefix,
                layer=self.v_proj,
                layer_name="v_proj",
                model_state_dict=model_state_dict,
                scale=v_proj_scale,
            )

            if self.bias:
                self.set_bias(
                    tensor=q_proj_bias,
                    prefix=prefix,
                    layer=self.q_proj,
                    layer_name="q_proj",
                    model_state_dict=model_state_dict,
                )
                self.set_bias(
                    tensor=k_proj_bias,
                    prefix=prefix,
                    layer=self.k_proj,
                    layer_name="k_proj",
                    model_state_dict=model_state_dict,
                )
                self.set_bias(
                    tensor=v_proj_bias,
                    prefix=prefix,
                    layer=self.v_proj,
                    layer_name="v_proj",
                    model_state_dict=model_state_dict,
                )

        return True


class GroupQueryAttention_O(BaseGroupQueryAttention):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        input_is_parallel: bool = False,
        layer_name: str = "o_proj",
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        rpl_reduce_dtype: torch.dtype = None,
        out_proj_kernel_enabled: bool = False,
        logical_nc_config: int = 1,
        rank_ordering: dict = None,
        tiling_factor: int = 1,
    ):
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            tp_degree=tp_degree,
            dtype=dtype,
            bias=bias,
            desired_sharding_strategy=desired_sharding_strategy,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )
        self.tiling_factor = tiling_factor

        self.input_is_parallel = input_is_parallel
        self.out_proj_kernel_enabled = out_proj_kernel_enabled
        self.logical_nc_config = logical_nc_config
        self.rpl_reduce_dtype = rpl_reduce_dtype
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.rank_ordering = rank_ordering

        if self.tensor_model_parallel_group is not None:
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias=self.bias,
                input_is_parallel=self.input_is_parallel,
                dtype=self.dtype,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                tensor_model_parallel_group=self.tensor_model_parallel_group,
                reduce_dtype=rpl_reduce_dtype,
                rank_ordering=rank_ordering,
                tile_cc=self.tiling_factor > 1,
            )
            if self.out_proj_kernel_enabled:
                # we need to transpose the weights on the CPU side to avoid
                # needing to transpose on the device when using out proj kernel
                self.o_proj.weight = transpose_parallel_linear_layer(self.o_proj.weight)
        else:
            self.o_proj = nn.Linear(
                self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.bias
            )

        # Prepared for changing "o_proj" to the corresponding name in model_state_dict
        # For example, in CLIP vision model, we use "out_proj"
        self.layer_name = layer_name

    def _kernel_o_proj(self, attention_output):
        logger.debug(f"Output projection kernel: logical_nc_config={self.logical_nc_config}")
        logger.debug(
            f"attention_output.shape: {attention_output.shape}"
            f"Output projection weight - shape: {self.o_proj.weight.shape}, dtype: {self.o_proj.weight.dtype}"
        )
        # The compute is: out(B, S, H) = attention_output(B, S, n, d) @ out_proj_weight(n * d, H)
        nd, H = self.o_proj.weight.shape
        B, S, nd = attention_output.shape
        heads_per_core = self.num_attention_heads // self.tp_degree
        assert (
            nd == heads_per_core * self.head_dim
        ), f"attention_output.shape = {attention_output.shape}, heads_per_core = {heads_per_core}, head_dim = {self.head_dim}"

        # Kernel wants BndS layout for input.
        attention_output = attention_output.reshape(B, S, heads_per_core, self.head_dim)
        kernel_attn_in = attention_output.permute(0, 2, 3, 1)

        out = torch.zeros(B, S, H, dtype=attention_output.dtype, device=attention_output.device)
        _traced_o_proj_kernel[(nc(self.logical_nc_config),)](
            active=kernel_attn_in,
            weight=self.o_proj.weight,
            out=out,
        )

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        original_dtype = out.dtype
        out = out.to(self.rpl_reduce_dtype)

        if self.sequence_parallel_enabled:
            out = reduce_scatter_to_sequence_parallel_region(
                out, 1, process_group=self.tensor_model_parallel_group
            )
        else:
            out = reduce_from_tensor_model_parallel_region(
                out, process_group=self.tensor_model_parallel_group
            )

        out = out.to(original_dtype)

        return out

    def forward(self, attention_output: torch.Tensor, adapter_ids=None):
        if self.out_proj_kernel_enabled:
            return self._kernel_o_proj(attention_output)

        return (
            self.o_proj(attention_output)
            if not is_lora_module(self.o_proj)
            else self.o_proj(attention_output, adapter_ids)
        )

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        prefix_parts = prefix.split(".")
        prefix = ".".join(prefix_parts[:-1])
        hf_prefix = ".".join(prefix_parts[:-2])

        self.replace_prefixes(
            old_prefix=f"{hf_prefix}.{self.layer_name}",
            new_prefix=f"{prefix}.o_proj",
            model_state_dict=model_state_dict,
        )
        o_proj_weight = model_state_dict[f"{prefix}.o_proj.weight"]
        o_proj_scale = model_state_dict.get(f"{prefix}.o_proj.scale", None)

        if self.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
            o_proj_weight, o_proj_scale = maybe_pad_interleaved(
                o_proj_weight,
                pad_dim=1,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                source_group_size=self._src_num_attention_heads // self._src_num_key_value_heads,
                tensor_scale=o_proj_scale,
            )

        if self.sharding_strategy == GQA.CONVERT_TO_MHA:
            o_proj_weight, o_proj_scale = maybe_pad_tail(
                o_proj_weight,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                pad_dim=1,
                tensor_scale=o_proj_scale,
            )

        model_state_dict[f"{prefix}.o_proj.weight"] = o_proj_weight
        if o_proj_scale is not None:
            model_state_dict[f"{prefix}.o_proj.scale"] = o_proj_scale
            verify_scale_dimension(tensor=o_proj_weight, tensor_scale=o_proj_scale)

        return True
