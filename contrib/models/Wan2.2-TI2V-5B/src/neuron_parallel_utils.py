from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerFF
from transformers.models.umt5.modeling_umt5 import UMT5Attention, UMT5LayerFF
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads, pad_model
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
import torch
from torch import nn

# 暂时禁用DistributedRMSNorm，因为all_reduce在编译时有问题
from distributed_rmsnorm import DistributedRMSNorm
# DistributedRMSNorm = RMSNorm  # 暂时使用标准RMSNorm

def get_sharded_data(data, dim):
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    s = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[s * tp_rank : s * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()

def shard_t5_self_attention(tp_degree: int, selfAttention: T5Attention):
    orig_inner_dim = selfAttention.q.out_features
    dim_head = orig_inner_dim // selfAttention.n_heads
    original_nheads = selfAttention.n_heads
    selfAttention.n_heads = selfAttention.n_heads // tp_degree
    selfAttention.inner_dim = dim_head * selfAttention.n_heads
    orig_q = selfAttention.q
    selfAttention.q = ColumnParallelLinear(
        selfAttention.q.in_features,
        selfAttention.q.out_features,
        bias=False, 
        gather_output=False)
    selfAttention.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    del(orig_q)
    orig_k = selfAttention.k
    selfAttention.k = ColumnParallelLinear(
        selfAttention.k.in_features, 
        selfAttention.k.out_features, 
        bias=(selfAttention.k.bias is not None),
        gather_output=False)
    selfAttention.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    del(orig_k)
    orig_v = selfAttention.v
    selfAttention.v = ColumnParallelLinear(
        selfAttention.v.in_features, 
        selfAttention.v.out_features, 
        bias=(selfAttention.v.bias is not None),
        gather_output=False)
    selfAttention.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    del(orig_v)
    orig_out = selfAttention.o
    selfAttention.o = RowParallelLinear(
        selfAttention.o.in_features,
        selfAttention.o.out_features,
        bias=(selfAttention.o.bias is not None),
        input_is_parallel=True)
    selfAttention.o.weight.data = get_sharded_data(orig_out.weight.data, 1)
    del(orig_out)
    return selfAttention

def shard_t5_ff(ff: T5LayerFF):
    orig_wi_0 = ff.DenseReluDense.wi_0
    ff.DenseReluDense.wi_0 = ColumnParallelLinear(
        orig_wi_0.in_features,
        orig_wi_0.out_features,
        bias=False,
        gather_output=False)
    ff.DenseReluDense.wi_0.weight.data = get_sharded_data(orig_wi_0.weight.data, 0)
    orig_wi_1 = ff.DenseReluDense.wi_1
    ff.DenseReluDense.wi_1 = ColumnParallelLinear(
        orig_wi_1.in_features,
        orig_wi_1.out_features,
        bias=False,
        gather_output=False)
    ff.DenseReluDense.wi_1.weight.data = get_sharded_data(orig_wi_1.weight.data, 0)
    orig_wo = ff.DenseReluDense.wo
    ff.DenseReluDense.wo = RowParallelLinear(
        orig_wo.in_features,
        orig_wo.out_features,
        bias=False,
        input_is_parallel=True)
    ff.DenseReluDense.wo.weight.data = get_sharded_data(orig_wo.weight.data, 1)
    ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")
    return ff

def shard_umt5_self_attention(tp_degree: int, selfAttention: UMT5Attention):
    orig_inner_dim = selfAttention.q.out_features
    original_nheads = selfAttention.n_heads
    dim_head = orig_inner_dim // original_nheads
    selfAttention.n_heads = original_nheads // tp_degree
    selfAttention.inner_dim = dim_head * selfAttention.n_heads
    orig_q = selfAttention.q
    selfAttention.q = ColumnParallelLinear(
        selfAttention.q.in_features,
        selfAttention.q.out_features,
        bias=False,
        gather_output=False,
        dtype=torch.bfloat16)
    selfAttention.q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    del(orig_q)
    orig_k = selfAttention.k
    selfAttention.k = ColumnParallelLinear(
        selfAttention.k.in_features,
        selfAttention.k.out_features,
        bias=(selfAttention.k.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    selfAttention.k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    del(orig_k)
    orig_v = selfAttention.v
    selfAttention.v = ColumnParallelLinear(
        selfAttention.v.in_features,
        selfAttention.v.out_features,
        bias=(selfAttention.v.bias is not None),
        gather_output=False,
        dtype=torch.bfloat16)
    selfAttention.v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    del(orig_v)
    orig_out = selfAttention.o
    selfAttention.o = RowParallelLinear(
        selfAttention.o.in_features,
        selfAttention.o.out_features,
        bias=(selfAttention.o.bias is not None),
        input_is_parallel=True,
        dtype=torch.bfloat16)
    selfAttention.o.weight.data = get_sharded_data(orig_out.weight.data, 1)
    del(orig_out)
    return selfAttention

def shard_umt5_ff(ff: UMT5LayerFF):
    orig_wi_0 = ff.DenseReluDense.wi_0
    ff.DenseReluDense.wi_0 = ColumnParallelLinear(
        orig_wi_0.in_features,
        orig_wi_0.out_features,
        bias=False,
        gather_output=False,
        dtype=torch.bfloat16)
    ff.DenseReluDense.wi_0.weight.data = get_sharded_data(orig_wi_0.weight.data, 0)
    orig_wi_1 = ff.DenseReluDense.wi_1
    ff.DenseReluDense.wi_1 = ColumnParallelLinear(
        orig_wi_1.in_features,
        orig_wi_1.out_features,
        bias=False,
        gather_output=False,
        dtype=torch.bfloat16)
    ff.DenseReluDense.wi_1.weight.data = get_sharded_data(orig_wi_1.weight.data, 0)
    orig_wo = ff.DenseReluDense.wo
    ff.DenseReluDense.wo = RowParallelLinear(
        orig_wo.in_features,
        orig_wo.out_features,
        bias=False,
        input_is_parallel=True,
        dtype=torch.bfloat16)
    ff.DenseReluDense.wo.weight.data = get_sharded_data(orig_wo.weight.data, 1)
    ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")  # Replace NewGELUActivation()
    return ff

def shard_transformer_attn(tp_degree: int, attn: Attention):
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    assert orig_inner_dim % attn.heads == 0
    orig_num_heads = attn.heads
    total_padded_heads = attn.heads + get_number_of_extra_heads(attn.heads, tp_degree)
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
    attn.sliceable_head_dim = attn.heads
    new_inner_dim = dim_head * attn.heads
    attn.inner_dim = new_inner_dim
    assert attn.to_q.out_features == attn.to_k.out_features and attn.to_q.out_features == attn.to_v.out_features

    orig_q = attn.to_q
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        attn.to_q.out_features,
        bias=(attn.to_q.bias is not None),
        gather_output=False)
    attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
    if attn.to_q.bias is not None:
        attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
    del(orig_q)

    orig_k = attn.to_k
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        attn.to_k.out_features,
        bias=(attn.to_k.bias is not None),
        gather_output=False)
    attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
    if attn.to_k.bias is not None:
        attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
    del(orig_k)

    orig_v = attn.to_v
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        attn.to_v.out_features,
        bias=(attn.to_v.bias is not None),
        gather_output=False)
    attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
    if attn.to_v.bias is not None:
        attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
    del(orig_v)

    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        attn.to_out[0].in_features,
        attn.to_out[0].out_features,
        bias=(attn.to_out[0].bias is not None),
        input_is_parallel=True)
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if attn.to_out[0].bias is not None: 
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del(orig_out)
    pad_model(attn, tp_degree, orig_num_heads, wrapped_classes=(Attention,))
    return attn


def shard_transformer_feedforward(ff: FeedForward) -> FeedForward:
    orig_proj = ff.net[0].proj
    ff.net[0].proj = ColumnParallelLinear(
        ff.net[0].proj.in_features,
        ff.net[0].proj.out_features,
        bias=(ff.net[0].proj.bias is not None),
        gather_output=False)
    ff.net[0].proj.weight.data = get_sharded_data(orig_proj.weight.data, 0)
    if ff.net[0].proj.bias is not None:
        ff.net[0].proj.bias.data = get_sharded_data(orig_proj.bias.data, 0)
    del(orig_proj)
    
    orig_linear = ff.net[2]
    ff.net[2] = RowParallelLinear(
        ff.net[2].in_features,
        ff.net[2].out_features,
        bias=(ff.net[2].bias is not None),
        input_is_parallel=True)
    ff.net[2].weight.data = get_sharded_data(orig_linear.weight.data, 1)
    if ff.net[2].bias is not None:
        ff.net[2].bias.data = orig_linear.bias.data.detach()
    del(orig_linear)
    return ff

def shard_transformer3d_attn_no_padding(tp_degree: int, attn: Attention, orig_num_heads: int):
    """当不需要padding时的简化版本（如TP=4时）"""
    
    # 获取维度信息
    orig_inner_dim = attn.to_q.out_features  # 1536/3072
    dim_head = orig_inner_dim // orig_num_heads  # 128
    new_inner_dim = attn.inner_dim  # 已经被更新为384 (1536/4, 3072/8)
    
    print(f"In no_padding: orig_inner_dim={orig_inner_dim}, new_inner_dim={new_inner_dim}, dim_head={dim_head}")
    
    # 分片Q/K/V - 重要：由于norm是在投影之后应用的，我们需要gather_output=True
    # 或者修改norm的处理方式
    
    # 方案1：使用gather_output=True（会增加通信开销）
    use_gather = False  # 暂时禁用，因为与rotary embedding不兼容
    
    # 当使用gather时，需要保存原始的heads数量用于unflatten
    if use_gather:
        attn._orig_heads = orig_num_heads  # 保存原始heads数量
    
    if use_gather:
        # 使用gather_output=True，这样norm看到的是完整维度
        orig_q = attn.to_q
        print('orig_q.in_features:', orig_q.in_features, 'orig_q.out_features:', orig_q.out_features)
        # 注意：ColumnParallelLinear不支持同时使用bias和gather_output=True
        # 所以我们禁用bias，稍后手动添加
        attn.to_q = ColumnParallelLinear(
            orig_q.in_features,
            orig_q.out_features,
            bias=False,  # 禁用bias以避免维度不匹配
            gather_output=True)  # 注意这里改为True
        attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
        # 保存原始bias以便后续使用
        if orig_q.bias is not None:
            attn.to_q._orig_bias = orig_q.bias.data.detach()
        print('attn.to_q.weight.data:', attn.to_q.weight.data.shape)
        del(orig_q)
        
        # 类似处理K和V
        orig_k = attn.to_k
        print('orig_k.in_features:', orig_k.in_features, 'orig_k.out_features:', orig_k.out_features)
        attn.to_k = ColumnParallelLinear(
            orig_k.in_features,
            orig_k.out_features,
            bias=False,  # 禁用bias
            gather_output=True)
        attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
        if orig_k.bias is not None:
            attn.to_k._orig_bias = orig_k.bias.data.detach()
        print('attn.to_k.weight.data:', attn.to_k.weight.data.shape)
        del(orig_k)
        
        orig_v = attn.to_v
        print('orig_v.in_features:', orig_v.in_features, 'orig_v.out_features:', orig_v.out_features)
        attn.to_v = ColumnParallelLinear(
            orig_v.in_features,
            orig_v.out_features,
            bias=False,  # 禁用bias
            gather_output=True)
        attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
        if orig_v.bias is not None:
            attn.to_v._orig_bias = orig_v.bias.data.detach()
        print('attn.to_v.weight.data:', attn.to_v.weight.data.shape)
        del(orig_v)
        
        # norm保持原始维度（因为gather_output=True）
        # 不需要修改norm
        
    else:
        # 方案2：不使用gather，修改norm以适应分片维度
        orig_q = attn.to_q
        attn.to_q = ColumnParallelLinear(
            orig_q.in_features,
            orig_q.out_features,
            bias=(orig_q.bias is not None),
            gather_output=False)
        attn.to_q.weight.data = get_sharded_data(orig_q.weight.data, 0)
        if orig_q.bias is not None:
            attn.to_q.bias.data = get_sharded_data(orig_q.bias.data, 0)
        del(orig_q)
        
        orig_k = attn.to_k
        attn.to_k = ColumnParallelLinear(
            orig_k.in_features,
            orig_k.out_features,
            bias=(orig_k.bias is not None),
            gather_output=False)
        attn.to_k.weight.data = get_sharded_data(orig_k.weight.data, 0)
        if orig_k.bias is not None:
            attn.to_k.bias.data = get_sharded_data(orig_k.bias.data, 0)
        del(orig_k)
        
        orig_v = attn.to_v
        attn.to_v = ColumnParallelLinear(
            orig_v.in_features,
            orig_v.out_features,
            bias=(orig_v.bias is not None),
            gather_output=False)
        attn.to_v.weight.data = get_sharded_data(orig_v.weight.data, 0)
        if orig_v.bias is not None:
            attn.to_v.bias.data = get_sharded_data(orig_v.bias.data, 0)
        del(orig_v)
        
        # 修改norm以适应分片后的维度
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            orig_norm_q = attn.norm_q
            old_eps = orig_norm_q.eps if hasattr(orig_norm_q, 'eps') else 1e-5
            old_elementwise_affine = orig_norm_q.elementwise_affine if hasattr(orig_norm_q, 'elementwise_affine') else True
            
            # 创建新的DistributedRMSNorm，使用分片后的维度
            attn.norm_q = DistributedRMSNorm(new_inner_dim, eps=old_eps, elementwise_affine=old_elementwise_affine)
            
            # 分片norm的weight
            if hasattr(orig_norm_q, 'weight') and orig_norm_q.weight is not None:
                attn.norm_q.weight.data = get_sharded_data(orig_norm_q.weight.data, 0)
        
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            orig_norm_k = attn.norm_k
            old_eps = orig_norm_k.eps if hasattr(orig_norm_k, 'eps') else 1e-5
            old_elementwise_affine = orig_norm_k.elementwise_affine if hasattr(orig_norm_k, 'elementwise_affine') else True
            
            attn.norm_k = DistributedRMSNorm(new_inner_dim, eps=old_eps, elementwise_affine=old_elementwise_affine)
            
            if hasattr(orig_norm_k, 'weight') and orig_norm_k.weight is not None:
                attn.norm_k.weight.data = get_sharded_data(orig_norm_k.weight.data, 0)
    
    # 对于I2V任务，处理额外的投影层
    if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
        orig_add_k = attn.add_k_proj
        attn.add_k_proj = ColumnParallelLinear(
            orig_add_k.in_features,
            orig_add_k.out_features,
            bias=(orig_add_k.bias is not None),
            gather_output=use_gather)  # 与Q/K/V保持一致
        attn.add_k_proj.weight.data = get_sharded_data(orig_add_k.weight.data, 0)
        if orig_add_k.bias is not None:
            attn.add_k_proj.bias.data = get_sharded_data(orig_add_k.bias.data, 0)
        del(orig_add_k)
    
    if hasattr(attn, 'add_v_proj') and attn.add_v_proj is not None:
        orig_add_v = attn.add_v_proj
        attn.add_v_proj = ColumnParallelLinear(
            orig_add_v.in_features,
            orig_add_v.out_features,
            bias=(orig_add_v.bias is not None),
            gather_output=use_gather)
        attn.add_v_proj.weight.data = get_sharded_data(orig_add_v.weight.data, 0)
        if orig_add_v.bias is not None:
            attn.add_v_proj.bias.data = get_sharded_data(orig_add_v.bias.data, 0)
        del(orig_add_v)
    
    # 处理norm_added_k
    if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
        if not use_gather:
            orig_norm_added_k = attn.norm_added_k
            old_eps = orig_norm_added_k.eps if hasattr(orig_norm_added_k, 'eps') else 1e-5
            
            attn.norm_added_k = DistributedRMSNorm(new_inner_dim, eps=old_eps, elementwise_affine=False)
    
    # 分片to_out
    orig_out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(
        orig_out.in_features,
        orig_out.out_features,
        bias=(orig_out.bias is not None),
        input_is_parallel=not use_gather)  # 如果使用gather，输入不是并行的
    attn.to_out[0].weight.data = get_sharded_data(orig_out.weight.data, 1)
    if orig_out.bias is not None:
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    del(orig_out)
    
    # 使用pad_model
    pad_model(attn, tp_degree, orig_num_heads, wrapped_classes=(Attention,))
    return attn

def shard_transformer3d_attn(tp_degree: int, attn: Attention):    
    orig_inner_dim = attn.to_q.out_features
    dim_head = orig_inner_dim // attn.heads
    assert orig_inner_dim % attn.heads == 0, f"inner_dim {orig_inner_dim} not divisible by heads {attn.heads}"
    orig_num_heads = attn.heads
    
    # 检查是否需要padding
    extra_heads = get_number_of_extra_heads(attn.heads, tp_degree)
    
    print(f"Original heads: {orig_num_heads}, Extra heads needed: {extra_heads}")
    print(f"Original inner_dim: {orig_inner_dim}, dim_head: {dim_head}")
    
    # 如果不需要padding（如TP=4, heads=12时），使用简化版本
    if extra_heads == 0:
        print(f"No padding needed for {orig_num_heads} heads with TP={tp_degree}")
        
        # 更新head数量（无padding）
        attn.heads = orig_num_heads // tp_degree
        attn.sliceable_head_dim = attn.heads
        attn.inner_dim = dim_head * attn.heads
        
        # 调用no_padding版本
        return shard_transformer3d_attn_no_padding(tp_degree, attn, orig_num_heads)
    
    # 需要padding的情况
    total_padded_heads = attn.heads + extra_heads
    print(f"Padding needed: {orig_num_heads} -> {total_padded_heads} heads")
    
    # 更新head数量（有padding）
    attn.heads = neuronx_dist_utils.divide(total_padded_heads, tp_degree)
    attn.sliceable_head_dim = attn.heads
    new_inner_dim = dim_head * attn.heads
    attn.inner_dim = new_inner_dim
    
    # 完整padded维度
    total_padded_dim = total_padded_heads * dim_head
    
    # 需要padding的情况（保留原有逻辑）
    # 分片 to_q, to_k, to_v
    orig_q = attn.to_q
    
    # Padding原始权重到完整的padded维度
    padded_q_weight = torch.zeros(total_padded_dim, orig_q.weight.shape[1], 
                                   dtype=orig_q.weight.dtype, device=orig_q.weight.device)
    padded_q_weight[:orig_inner_dim] = orig_q.weight.data
    
    # 创建新的ColumnParallelLinear
    attn.to_q = ColumnParallelLinear(
        attn.to_q.in_features,
        new_inner_dim,  # 使用padded后每个rank的维度 (256)
        bias=(attn.to_q.bias is not None),
        gather_output=False)
    
    # 使用修改后的get_sharded_data来分片padded权重
    attn.to_q.weight.data = get_sharded_data(padded_q_weight, 0)
    
    if attn.to_q.bias is not None:
        padded_q_bias = torch.zeros(total_padded_dim, dtype=orig_q.bias.dtype, device=orig_q.bias.device)
        padded_q_bias[:orig_inner_dim] = orig_q.bias.data
        attn.to_q.bias.data = get_sharded_data(padded_q_bias, 0)
    
    del(orig_q)

    # 同样处理to_k
    orig_k = attn.to_k
    
    # Padding K权重
    padded_k_weight = torch.zeros(total_padded_dim, orig_k.weight.shape[1],
                                   dtype=orig_k.weight.dtype, device=orig_k.weight.device)
    padded_k_weight[:orig_inner_dim] = orig_k.weight.data
    
    attn.to_k = ColumnParallelLinear(
        attn.to_k.in_features,
        new_inner_dim,  # 使用padded后每个rank的维度
        bias=(attn.to_k.bias is not None),
        gather_output=False)
    
    attn.to_k.weight.data = get_sharded_data(padded_k_weight, 0)
    
    if attn.to_k.bias is not None:
        padded_k_bias = torch.zeros(total_padded_dim, dtype=orig_k.bias.dtype, device=orig_k.bias.device)
        padded_k_bias[:orig_inner_dim] = orig_k.bias.data
        attn.to_k.bias.data = get_sharded_data(padded_k_bias, 0)
    
    del(orig_k)

    # 同样处理to_v
    orig_v = attn.to_v
    
    # Padding V权重
    padded_v_weight = torch.zeros(total_padded_dim, orig_v.weight.shape[1],
                                   dtype=orig_v.weight.dtype, device=orig_v.weight.device)
    padded_v_weight[:orig_inner_dim] = orig_v.weight.data
    
    attn.to_v = ColumnParallelLinear(
        attn.to_v.in_features,
        new_inner_dim,  # 使用padded后每个rank的维度
        bias=(attn.to_v.bias is not None),
        gather_output=False)
    
    attn.to_v.weight.data = get_sharded_data(padded_v_weight, 0)
    
    if attn.to_v.bias is not None:
        padded_v_bias = torch.zeros(total_padded_dim, dtype=orig_v.bias.dtype, device=orig_v.bias.device)
        padded_v_bias[:orig_inner_dim] = orig_v.bias.data
        attn.to_v.bias.data = get_sharded_data(padded_v_bias, 0)
    
    del(orig_v)

    # 修复norm层 - 需要匹配padding后的维度
    if hasattr(attn, 'norm_q') and attn.norm_q is not None:
        old_eps = attn.norm_q.eps if hasattr(attn.norm_q, 'eps') else 1e-5
        old_elementwise_affine = attn.norm_q.elementwise_affine if hasattr(attn.norm_q, 'elementwise_affine') else True
        
        # 保存原始weight
        orig_weight = None
        if hasattr(attn.norm_q, 'weight') and attn.norm_q.weight is not None:
            orig_weight = attn.norm_q.weight.data
        
        # 创建新的DistributedRMSNorm，使用padding后的维度
        attn.norm_q = DistributedRMSNorm(new_inner_dim, eps=old_eps, elementwise_affine=old_elementwise_affine)  # 使用256
        
        # 设置weight - 先padding原始权重再分片
        if orig_weight is not None and old_elementwise_affine:
            if orig_weight.shape[0] == orig_inner_dim:
                # 先padding原始权重到完整维度
                padded_norm_weight = torch.ones(total_padded_dim, dtype=orig_weight.dtype, device=orig_weight.device)
                padded_norm_weight[:orig_inner_dim] = orig_weight
                # 然后分片
                attn.norm_q.weight.data = get_sharded_data(padded_norm_weight, 0)
            else:
                # 默认值
                pass
    
    # 类似处理norm_k
    if hasattr(attn, 'norm_k') and attn.norm_k is not None:
        old_eps = attn.norm_k.eps if hasattr(attn.norm_k, 'eps') else 1e-5
        old_elementwise_affine = attn.norm_k.elementwise_affine if hasattr(attn.norm_k, 'elementwise_affine') else True
        
        orig_weight = None
        if hasattr(attn.norm_k, 'weight') and attn.norm_k.weight is not None:
            orig_weight = attn.norm_k.weight.data
        
        attn.norm_k = DistributedRMSNorm(new_inner_dim, eps=old_eps, elementwise_affine=old_elementwise_affine)  # 使用256
        
        if orig_weight is not None and old_elementwise_affine:
            if orig_weight.shape[0] == orig_inner_dim:
                # 先padding原始权重到完整维度
                padded_norm_weight = torch.ones(total_padded_dim, dtype=orig_weight.dtype, device=orig_weight.device)
                padded_norm_weight[:orig_inner_dim] = orig_weight
                # 然后分片
                attn.norm_k.weight.data = get_sharded_data(padded_norm_weight, 0)

    # 处理I2V相关层
    if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
        orig_add_k = attn.add_k_proj
        attn.add_k_proj = ColumnParallelLinear(
            orig_add_k.in_features,
            actual_output_dim,  # 使用实际分片后的维度
            bias=(orig_add_k.bias is not None),
            gather_output=False)
        attn.add_k_proj.weight.data = get_sharded_data(orig_add_k.weight.data, 0)
        if orig_add_k.bias is not None:
            attn.add_k_proj.bias.data = get_sharded_data(orig_add_k.bias.data, 0)
        del(orig_add_k)
    
    if hasattr(attn, 'add_v_proj') and attn.add_v_proj is not None:
        orig_add_v = attn.add_v_proj
        attn.add_v_proj = ColumnParallelLinear(
            orig_add_v.in_features,
            actual_output_dim,  # 使用实际分片后的维度
            bias=(orig_add_v.bias is not None),
            gather_output=False)
        attn.add_v_proj.weight.data = get_sharded_data(orig_add_v.weight.data, 0)
        if orig_add_v.bias is not None:
            attn.add_v_proj.bias.data = get_sharded_data(orig_add_v.bias.data, 0)
        del(orig_add_v)
    
    # 处理norm_added_k
    if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
        old_eps = attn.norm_added_k.eps if hasattr(attn.norm_added_k, 'eps') else 1e-5
        old_elementwise_affine = attn.norm_added_k.elementwise_affine if hasattr(attn.norm_added_k, 'elementwise_affine') else True
        
        orig_weight = None
        if hasattr(attn.norm_added_k, 'weight') and attn.norm_added_k.weight is not None:
            orig_weight = attn.norm_added_k.weight.data
        
        attn.norm_added_k = DistributedRMSNorm(new_inner_dim, eps=old_eps, elementwise_affine=old_elementwise_affine)  # 使用256
        
        if orig_weight is not None and old_elementwise_affine:
            if orig_weight.shape[0] == orig_inner_dim:
                sharded_weight = get_sharded_data(orig_weight, 0)
                # Padding到256维
                padded_weight = torch.ones(new_inner_dim, dtype=sharded_weight.dtype, device=sharded_weight.device)
                padded_weight[:actual_output_dim] = sharded_weight
                attn.norm_added_k.weight.data = padded_weight

    # 分片 to_out
    # to_out的权重也需要先padding再分片
    orig_out = attn.to_out[0]
    
    # Padding to_out权重 (注意这是RowParallel，所以padding在dim=1)
    padded_out_weight = torch.zeros(orig_out.weight.shape[0], total_padded_dim,
                                     dtype=orig_out.weight.dtype, device=orig_out.weight.device)
    padded_out_weight[:, :orig_inner_dim] = orig_out.weight.data
    
    attn.to_out[0] = RowParallelLinear(
        new_inner_dim,  # 输入维度是padded后的维度
        attn.to_out[0].out_features,
        bias=(attn.to_out[0].bias is not None),
        input_is_parallel=True)
    
    attn.to_out[0].weight.data = get_sharded_data(padded_out_weight, 1)
    
    if attn.to_out[0].bias is not None: 
        attn.to_out[0].bias.data = orig_out.bias.data.detach()
    
    del(orig_out)
    
    # 不再需要pad_model，因为我们已经手动padding了权重
    # pad_model(attn, tp_degree, orig_num_heads, wrapped_classes=(Attention,))
    return attn
