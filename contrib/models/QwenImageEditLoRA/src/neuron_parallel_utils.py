import torch
from torch import nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads, pad_model
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils


class ShardedRMSNorm(nn.Module):
    """RMSNorm that works with sharded hidden dimensions."""
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        # RMSNorm computation - normalize over last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.weight is not None:
            return x_normed * self.weight
        return x_normed


def get_sharded_data(data, dim):
    """Shard data across tensor parallel ranks."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    s = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()


def shard_rmsnorm(orig_norm, new_dim):
    """Create a sharded RMSNorm from an original RMSNorm."""
    eps = orig_norm.eps if hasattr(orig_norm, 'eps') else 1e-6
    elementwise_affine = hasattr(orig_norm, 'weight') and orig_norm.weight is not None

    new_norm = ShardedRMSNorm(new_dim, eps=eps, elementwise_affine=elementwise_affine)

    if elementwise_affine and orig_norm.weight is not None:
        new_norm.weight.data = get_sharded_data(orig_norm.weight.data, 0)

    return new_norm


def shard_qwen_attention(tp_degree: int, attn: Attention):
    """
    Shard QwenImage attention module for tensor parallelism.
    This handles both image attention (to_q/k/v) and text attention (add_q/k/v_proj).
    """
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    assert orig_inner_dim % attn.heads == 0
    orig_num_heads = attn.heads
    total_padded_heads = attn.heads + get_number_of_extra_heads(attn.heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
    attn.sliceable_head_dim = attn.heads
    new_inner_dim = dim_head * attn.heads
    attn.inner_dim = new_inner_dim

    # Shard image attention projections (to_q, to_k, to_v)
    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        attn.to_q.out_features,
        bias=(attn.to_q.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if attn.to_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del orig_q

    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        attn.to_k.out_features,
        bias=(attn.to_k.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if attn.to_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        attn.to_v.out_features,
        bias=(attn.to_v.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if attn.to_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard output projection
    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        attn.to_out[0].in_features,
        attn.to_out[0].out_features,
        bias=(attn.to_out[0].bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if attn.to_out[0].bias is not None:
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del orig_out

    # Shard text attention projections (add_q_proj, add_k_proj, add_v_proj)
    if hasattr(attn, 'add_q_proj') and attn.add_q_proj is not None:
        orig_add_q = attn.add_q_proj
        attn.add_q_proj = ColumnParallelLinear(
            orig_add_q.in_features,
            orig_add_q.out_features,
            bias=(orig_add_q.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16)
        attn.add_q_proj.weight.data = get_sharded_data(orig_add_q.weight.data, 0)
        if orig_add_q.bias is not None:
            attn.add_q_proj.bias.data = get_sharded_data(orig_add_q.bias.data, 0)
        del orig_add_q

    if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
        orig_add_k = attn.add_k_proj
        attn.add_k_proj = ColumnParallelLinear(
            orig_add_k.in_features,
            orig_add_k.out_features,
            bias=(orig_add_k.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16)
        attn.add_k_proj.weight.data = get_sharded_data(orig_add_k.weight.data, 0)
        if orig_add_k.bias is not None:
            attn.add_k_proj.bias.data = get_sharded_data(orig_add_k.bias.data, 0)
        del orig_add_k

    if hasattr(attn, 'add_v_proj') and attn.add_v_proj is not None:
        orig_add_v = attn.add_v_proj
        attn.add_v_proj = ColumnParallelLinear(
            orig_add_v.in_features,
            orig_add_v.out_features,
            bias=(orig_add_v.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16)
        attn.add_v_proj.weight.data = get_sharded_data(orig_add_v.weight.data, 0)
        if orig_add_v.bias is not None:
            attn.add_v_proj.bias.data = get_sharded_data(orig_add_v.bias.data, 0)
        del orig_add_v

    # Shard to_add_out
    if hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
        orig_add_out = attn.to_add_out
        attn.to_add_out = RowParallelLinear(
            orig_add_out.in_features,
            orig_add_out.out_features,
            bias=(orig_add_out.bias is not None),
            input_is_parallel=True,
            dtype=torch.bfloat16)
        attn.to_add_out.weight.data = get_sharded_data(orig_add_out.weight.data, 1)
        if orig_add_out.bias is not None:
            attn.to_add_out.bias.data = orig_add_out.bias.data.detach()
        del orig_add_out

    # Note: RMSNorm layers (norm_q, norm_k, norm_added_q, norm_added_k) should NOT be sharded!
    # They operate on head_dim (128) which doesn't change with tensor parallelism.
    # The norms are applied AFTER unflatten to [batch, seq, heads, head_dim],
    # so they normalize over head_dim, not inner_dim.

    # Note: pad_model is not needed when heads are evenly divisible by tp_degree
    # For QwenImage: 24 heads / 4 = 6 heads per rank (evenly divisible)
    return attn


def shard_feedforward(ff: FeedForward) -> FeedForward:
    """Shard FeedForward module for tensor parallelism."""
    # Shard the first linear layer (GELU projection)
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        ff.net[0].proj.in_features,
        ff.net[0].proj.out_features,
        bias=(ff.net[0].proj.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if ff.net[0].proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del orig_proj

    # Shard the output linear layer
    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        ff.net[2].in_features,
        ff.net[2].out_features,
        bias=(ff.net[2].bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if ff.net[2].bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()
    del orig_linear
    return ff


def shard_modulation(mod: nn.Sequential) -> nn.Sequential:
    """
    Shard modulation layer (img_mod, txt_mod) for tensor parallelism.

    Modulation layers are Sequential(SiLU, Linear) with shape [18432, 3072].
    18432 = 6 * 3072 (for 6 modulation outputs: shift, scale for 3 different targets)

    We shard the output dimension (18432) across TP ranks.

    IMPORTANT: When gather_output=True, the output is gathered to full size BEFORE
    adding the bias. So we must NOT shard the bias - it needs to be full size (18432).
    """
    # mod[0] is SiLU (no weights)
    # mod[1] is Linear(3072, 18432)
    orig_linear = mod[1]

    mod[1] = ColumnParallelLinear(
        orig_linear.in_features,
        orig_linear.out_features,
        bias=(orig_linear.bias is not None),
        gather_output=True,  # Need to gather for modulation to work correctly
        dtype=torch.bfloat16)
    # Shard weights across output dimension
    mod[1].weight.data = get_sharded_data(orig_linear.weight.data, 0)
    # IMPORTANT: Do NOT shard bias when gather_output=True!
    # The bias is added after gathering, so it needs full size
    if orig_linear.bias is not None:
        mod[1].bias.data = orig_linear.bias.data.clone().to(torch.bfloat16)
    del orig_linear

    return mod


def get_sharded_data_with_replication(data, dim, num_heads, tp_degree):
    """
    Shard data with head replication when num_heads < tp_degree.

    For GQA models where num_kv_heads < tp_degree, we replicate KV heads
    so each rank gets a copy. E.g., with 4 KV heads and TP=8:
    - Heads are replicated 2x to make 8 virtual heads
    - Each rank gets 1 virtual head (which is a copy of the original)
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()

    if num_heads >= tp_size:
        # Normal sharding
        return get_sharded_data(data, dim)
    else:
        # Replication mode: num_heads < tp_size
        # Each head is replicated (tp_size // num_heads) times
        replication_factor = tp_size // num_heads
        # Map tp_rank to the original head index
        original_head_idx = tp_rank // replication_factor

        head_dim = data.shape[dim] // num_heads
        if dim == 0:
            start = original_head_idx * head_dim
            end = (original_head_idx + 1) * head_dim
            return data[start:end].clone()
        elif dim == 1:
            start = original_head_idx * head_dim
            end = (original_head_idx + 1) * head_dim
            return data[:, start:end].clone()


def shard_qwen2_attention(tp_degree: int, self_attn):
    """
    Shard Qwen2/Qwen2.5-VL self attention module (used in text encoder).

    Handles GQA (Grouped Query Attention) where num_key_value_heads < num_heads.
    For Qwen2.5-VL: num_heads=28, num_key_value_heads=4

    Supports two modes:
    1. tp_degree <= num_kv_heads: Standard sharding (each rank gets subset of KV heads)
    2. tp_degree > num_kv_heads: KV head replication (each rank gets replicated KV heads)

    With tp_degree=8 and num_kv_heads=4:
    - Q heads: 28 -> padded to 32 -> 4 per rank
    - KV heads: 4 -> replicated to 8 -> 1 per rank (each pair of ranks shares same KV head)
    """
    # Get original dimensions
    orig_q = self_attn.q_proj
    orig_k = self_attn.k_proj
    orig_v = self_attn.v_proj
    orig_o = self_attn.o_proj

    # Get KV head count
    num_kv_heads = getattr(self_attn, 'num_key_value_heads', self_attn.num_heads)
    num_q_heads = self_attn.num_heads

    # Check if KV replication is needed
    kv_replicate_mode = num_kv_heads < tp_degree
    if kv_replicate_mode:
        # Replication mode: tp_degree must be divisible by num_kv_heads
        if tp_degree % num_kv_heads != 0:
            raise ValueError(
                f"For KV head replication, tp_degree ({tp_degree}) must be divisible by "
                f"num_key_value_heads ({num_kv_heads})")
        print(f"  Using KV head replication mode: {num_kv_heads} KV heads replicated across {tp_degree} ranks")

    # Calculate padded heads for Q
    extra_q_heads = get_number_of_extra_heads(num_q_heads, tp_degree)
    total_padded_q_heads = num_q_heads + extra_q_heads
    q_head_dim = orig_q.out_features // num_q_heads  # 3584 / 28 = 128
    padded_q_out_features = total_padded_q_heads * q_head_dim  # 32 * 128 = 4096

    print(f"  Q heads: {num_q_heads} -> padded to {total_padded_q_heads}, "
          f"out_features: {orig_q.out_features} -> {padded_q_out_features}")

    # Update number of heads per rank
    self_attn.num_heads = neuronx_dist_utils.divide(total_padded_q_heads, tp_degree)
    if hasattr(self_attn, 'num_key_value_heads'):
        if kv_replicate_mode:
            # In replication mode, each rank effectively has 1 KV head (replicated)
            self_attn.num_key_value_heads = 1
        else:
            self_attn.num_key_value_heads = self_attn.num_key_value_heads // tp_degree

    # CRITICAL: Update num_key_value_groups!
    # This is used by repeat_kv() in attention forward to expand KV heads
    if hasattr(self_attn, 'num_key_value_groups'):
        self_attn.num_key_value_groups = self_attn.num_heads // self_attn.num_key_value_heads
        print(f"  Updated num_key_value_groups: {self_attn.num_key_value_groups}")

    # Shard Q projection (with padding if needed)
    # Need to pad weights before sharding when num_heads is not divisible by tp_degree
    q_weight_padded = orig_q.weight.data
    q_bias_padded = orig_q.bias.data if orig_q.bias is not None else None

    if extra_q_heads > 0:
        # Pad Q weights with zeros for extra heads
        padding_size = extra_q_heads * q_head_dim
        q_weight_padding = torch.zeros(
            (padding_size, orig_q.in_features),
            dtype=orig_q.weight.dtype,
            device=orig_q.weight.device)
        q_weight_padded = torch.cat([orig_q.weight.data, q_weight_padding], dim=0)

        if orig_q.bias is not None:
            q_bias_padding = torch.zeros(
                padding_size,
                dtype=orig_q.bias.dtype,
                device=orig_q.bias.device)
            q_bias_padded = torch.cat([orig_q.bias.data, q_bias_padding], dim=0)

    # Now create ColumnParallelLinear with padded dimensions
    self_attn.q_proj = ColumnParallelLinear(
        orig_q.in_features,
        padded_q_out_features,  # Use padded out_features
        bias=(orig_q.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    self_attn.q_proj.weight.data = get_sharded_data(q_weight_padded, 0)
    if orig_q.bias is not None:
        self_attn.q_proj.bias.data = get_sharded_data(q_bias_padded, 0)
    del orig_q

    # Shard K projection (replicated if kv_replicate_mode)
    # Get head_dim for KV
    kv_head_dim = orig_k.out_features // num_kv_heads  # 512 / 4 = 128

    if kv_replicate_mode:
        # In replication mode, use regular nn.Linear (not ColumnParallelLinear)
        # because we want each rank to have 1 full KV head, not a fraction
        # Each rank gets 1 KV head = head_dim features
        kv_out_features_per_rank = kv_head_dim  # 128

        self_attn.k_proj = nn.Linear(
            orig_k.in_features,
            kv_out_features_per_rank,
            bias=(orig_k.bias is not None),
            dtype=torch.bfloat16)
        self_attn.k_proj.weight.data = get_sharded_data_with_replication(
            orig_k.weight.data, 0, num_kv_heads, tp_degree)
        if orig_k.bias is not None:
            self_attn.k_proj.bias.data = get_sharded_data_with_replication(
                orig_k.bias.data, 0, num_kv_heads, tp_degree)
    else:
        self_attn.k_proj = ColumnParallelLinear(
            orig_k.in_features,
            orig_k.out_features,
            bias=(orig_k.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16)
        self_attn.k_proj.weight.data = get_sharded_data(orig_k.weight.data, 0)
        if orig_k.bias is not None:
            self_attn.k_proj.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del orig_k

    # Shard V projection (replicated if kv_replicate_mode)
    if kv_replicate_mode:
        # Same as K: use regular nn.Linear with replicated weights
        kv_out_features_per_rank = kv_head_dim  # 128

        self_attn.v_proj = nn.Linear(
            orig_v.in_features,
            kv_out_features_per_rank,
            bias=(orig_v.bias is not None),
            dtype=torch.bfloat16)
        self_attn.v_proj.weight.data = get_sharded_data_with_replication(
            orig_v.weight.data, 0, num_kv_heads, tp_degree)
        if orig_v.bias is not None:
            self_attn.v_proj.bias.data = get_sharded_data_with_replication(
                orig_v.bias.data, 0, num_kv_heads, tp_degree)
    else:
        self_attn.v_proj = ColumnParallelLinear(
            orig_v.in_features,
            orig_v.out_features,
            bias=(orig_v.bias is not None),
            gather_output=False,
            dtype=torch.bfloat16)
        self_attn.v_proj.weight.data = get_sharded_data(orig_v.weight.data, 0)
        if orig_v.bias is not None:
            self_attn.v_proj.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del orig_v

    # Shard O projection (always sharded based on Q heads)
    # O projection input comes from attention output, which has padded_q_out_features
    # We need to pad the O weight's input dimension to match

    o_weight_padded = orig_o.weight.data

    if extra_q_heads > 0:
        # Original O weight: (out_features, in_features) = (3584, 3584)
        # Need to pad input dimension to padded_q_out_features = 4096
        padding_size = extra_q_heads * q_head_dim
        o_weight_padding = torch.zeros(
            (orig_o.out_features, padding_size),
            dtype=orig_o.weight.dtype,
            device=orig_o.weight.device)
        o_weight_padded = torch.cat([orig_o.weight.data, o_weight_padding], dim=1)

    self_attn.o_proj = RowParallelLinear(
        padded_q_out_features,  # Use padded in_features
        orig_o.out_features,
        bias=(orig_o.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    self_attn.o_proj.weight.data = get_sharded_data(o_weight_padded, 1)
    if orig_o.bias is not None:
        self_attn.o_proj.bias.data = orig_o.bias.data.detach()
    del orig_o

    return self_attn


def shard_vision_attention(tp_degree: int, attn):
    """
    Shard Qwen2.5-VL Vision Encoder attention module.

    Vision attention uses fused QKV projection:
    - qkv: (in_features, 3 * in_features) -> splits into Q, K, V
    - proj: output projection
    """
    orig_qkv = attn.qkv
    orig_proj = attn.proj

    # Shard fused QKV projection
    attn.qkv = ColumnParallelLinear(
        orig_qkv.in_features,
        orig_qkv.out_features,
        bias=(orig_qkv.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    attn.qkv.weight.data = get_sharded_data(orig_qkv.weight.data, 0)
    if orig_qkv.bias is not None:
        attn.qkv.bias.data = get_sharded_data(orig_qkv.bias.data, 0)
    del orig_qkv

    # Shard output projection
    attn.proj = RowParallelLinear(
        orig_proj.in_features,
        orig_proj.out_features,
        bias=(orig_proj.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    attn.proj.weight.data = get_sharded_data(orig_proj.weight.data, 1)
    if orig_proj.bias is not None:
        attn.proj.bias.data = orig_proj.bias.data.detach()
    del orig_proj

    return attn


def shard_vision_mlp(mlp):
    """
    Shard Qwen2.5-VL Vision Encoder MLP module.

    Uses gate_proj, up_proj, down_proj like Qwen2 MLP.
    """
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    # Shard gate projection
    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features,
        orig_gate.out_features,
        bias=(orig_gate.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    # Shard up projection
    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features,
        orig_up.out_features,
        bias=(orig_up.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    # Shard down projection
    mlp.down_proj = RowParallelLinear(
        orig_down.in_features,
        orig_down.out_features,
        bias=(orig_down.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp


def shard_qwen2_mlp(mlp):
    """
    Shard Qwen2 MLP module (used in text encoder).
    """
    orig_gate = mlp.gate_proj
    orig_up = mlp.up_proj
    orig_down = mlp.down_proj

    # Shard gate projection
    mlp.gate_proj = ColumnParallelLinear(
        orig_gate.in_features,
        orig_gate.out_features,
        bias=(orig_gate.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    mlp.gate_proj.weight.data = get_sharded_data(orig_gate.weight.data, 0)
    if orig_gate.bias is not None:
        mlp.gate_proj.bias.data = get_sharded_data(orig_gate.bias.data, 0)
    del orig_gate

    # Shard up projection
    mlp.up_proj = ColumnParallelLinear(
        orig_up.in_features,
        orig_up.out_features,
        bias=(orig_up.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    mlp.up_proj.weight.data = get_sharded_data(orig_up.weight.data, 0)
    if orig_up.bias is not None:
        mlp.up_proj.bias.data = get_sharded_data(orig_up.bias.data, 0)
    del orig_up

    # Shard down projection
    mlp.down_proj = RowParallelLinear(
        orig_down.in_features,
        orig_down.out_features,
        bias=(orig_down.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    mlp.down_proj.weight.data = get_sharded_data(orig_down.weight.data, 1)
    if orig_down.bias is not None:
        mlp.down_proj.bias.data = orig_down.bias.data.detach()
    del orig_down

    return mlp
