# flake8: noqa: E226

# Original copyright (c) Meta Platforms, Inc. and affiliates.
# Please check the LICENSE under the same director for details.
#
# Modified by Amazon


import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file

from neuronx_distributed_inference.experimental.models.config import Config
import neuronx_distributed_inference.experimental.functional as NF

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from torch_neuronx.xla_impl.ops import RmsNorm

from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)


def precompute_rope(device, theta, head_dim, seq_len):
    # Refer: https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32
    theta = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim))

    seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()

    # cache includes both the cos and sin components and so the output shape is
    # [max_seq_len, dim // 2, 2]
    return torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)


def rope(x, start_pos, cache):
    # input tensor has shape [b, s, n_h, h_d]
    bs, input_len, _, _ = x.shape

    # expand cache to batch size
    rope_cache = cache.unsqueeze(0).expand(bs, *cache.shape)

    if input_len == 1:
        # We are in decode mode, so we have a q & k of size 1.
        # But the positions of these tokens are not necessarily the same, so
        # we gather the RoPE cos and sine values by position id
        index = start_pos.view(bs, 1, 1, 1).expand(bs, 1, cache.shape[-2], cache.shape[-1])
        rope_cache = torch.gather(rope_cache, dim=1, index=index.to(torch.int64))

    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # reshape the cache for broadcasting
    # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
    # otherwise has shape [1, s, 1, h_d // 2, 2]
    rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

    # tensor has shape [b, s, n_h, h_d // 2, 2]
    x_out = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    # tensor has shape [b, s, n_h, h_d]
    x_out = x_out.flatten(3)
    return x_out.type_as(x)


def rmsnorm(x, eps):
    def _norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.eps = cfg.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=cfg.dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if x.is_cpu:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
        else:
            return RmsNorm.apply(x, self.weight, self.eps, len(x.shape) - 1).to(x.dtype)


class Attention(nn.Module):

    def __init__(self, cfg: Config, batch_size: int, seq_len: int, use_rope: bool, use_qk_norm:bool):
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = cfg.rms_norm_eps
        self.head_dim = cfg.head_dim

        self.tp_degree = parallel_state.get_tensor_model_parallel_group().size()
        self.n_heads = cfg.n_heads

        if self.n_heads % self.tp_degree != 0:
            print("n_heads not evenly divisible by tp degree. Padding...")
            # pad to closest tp_degree multiple
            self.n_heads = ((self.n_heads + self.tp_degree -1) // self.tp_degree) * self.tp_degree

        # we want atleast 1 kv head on a core
        self.n_heads = self.n_heads // self.tp_degree
        self.n_kv_heads = max(cfg.n_kv_heads // self.tp_degree, 1)

        self.wqkv = ColumnParallelLinear(
            cfg.hidden_size,
            (self.n_heads + 2 * self.n_kv_heads) * self.head_dim * self.tp_degree,
            bias=False,
            gather_output=False,
            dtype=cfg.dtype
        )
        self.o_proj = RowParallelLinear(
            self.n_heads * self.tp_degree * self.head_dim,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=cfg.dtype,
        )

        self.register_buffer('cache_k', torch.zeros(batch_size, seq_len, self.n_kv_heads, self.head_dim, requires_grad=False, dtype=cfg.dtype))
        self.register_buffer('cache_v', torch.zeros(batch_size, seq_len, self.n_kv_heads, self.head_dim, requires_grad=False, dtype=cfg.dtype))

        self.n_rep = self.n_heads // self.n_kv_heads

    def _qkv_proj(self, wQKV, hidden, n_heads, n_kv_heads, head_dim):
        qkv = wQKV(hidden)
        q_end_index = n_heads * head_dim
        k_end_index = q_end_index + n_kv_heads * head_dim

        Q, K, V = torch.tensor_split(
            qkv,
            (
                q_end_index,
                k_end_index,
                # rest of the QKV will go to V output
            ),
            dim=2,
        )

        return Q, K, V

    def forward(
        self,
        x: torch.Tensor,
        last_pos: torch.Tensor,
        rope_cache: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_prefill: bool,
    ):
        # x (Batch, Sequence, Hidden)
        bsz, seq_len, _ = x.shape

        # BSH
        if is_prefill:
            q, k, v = NF.qkv_kernel(
                hidden_states=x,
                w_qkv=self.wqkv.weight.T, # Transpose the weights for QKV kernel
                rmsnorm=None,
                num_attention_heads=self.n_heads,
                num_key_value_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            q, k, v = self._qkv_proj(self.wqkv, x, self.n_heads, self.n_kv_heads, self.head_dim)

        # BSNH
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        if seq_len > 1:
            start_pos = torch.zeros_like(last_pos)
        else:
            start_pos = last_pos

        if self.use_rope:
            q, k = rope(q, start_pos, rope_cache), rope(k, start_pos, rope_cache)

        if self.use_qk_norm:
            q = rmsnorm(q, self.rms_norm_eps)
            k = rmsnorm(k, self.rms_norm_eps)

        # Save KV Cache
        if seq_len > 1:
            indices = torch.arange(start=0, end=seq_len, dtype=torch.int64, device=q.device)
            indices = indices.view(1, seq_len, 1, 1)
            indices = indices.expand(bsz, seq_len, self.n_kv_heads, self.head_dim)
        else:
            indices = last_pos.view(bsz, 1, 1, 1).expand_as(k).to(torch.int64)

        # Update KV cache
        self.cache_k = torch.scatter(self.cache_k, 1, indices, k)
        self.cache_v = torch.scatter(self.cache_v, 1, indices, v)

        # Note: We cannot just slice the cache to the current position. If we slice, we would change the
        # compute shape for every decode run. On Neuron we compile for fixed shapes. So the alternative is to
        # operate on a fixed sequence length per compilation. In this example we compute the attention
        # for the full preallocated sequence length. We just 'read' the output at the right index.

        # keys = self.cache_k[:bsz, : start_pos + inp_len]  X wrong
        # values = self.cache_v[:bsz, : start_pos + ]       X wrong

        # Yes fixed shape will cause us to waste compute. The way to work around that is to 'bucket' - compile for many
        # shapes. One way is to bucket along sequence length. Here we can slice the KV cache when you bucket along
        # sequence length. This is an easy optimization you could do. Note, this example does not bucket.
        keys = self.cache_k
        values = self.cache_v

        # With GQA, k/v heads are shared amond different q heads
        # repeat k/v heads to match q heads
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.n_rep)

        # bs, seqlen, head, head_dim -> bs, head, seqlen, head_dim
        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if is_prefill:
            output = NF.scaled_dot_product_attention_kernel(Q=q, K=keys, V=values, is_causal=True)
        else:
            # Q.K^T/√d
            scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                # You cannot use -inf well on Neuron, you will run into 1003 errors
                scores = torch.where(mask, scores, torch.finfo(scores.dtype).min)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, values)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # TODO: Integrate o_proj kernel
        return self.o_proj(output)


class Experts(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.num_local_experts = cfg.num_local_experts
        self.hidden_size = cfg.hidden_size
        self.gate_up_stride = 2

        # Initialize expert weights
        self.gate_up_proj = ExpertFusedColumnParallelLinear(
            num_experts=self.num_local_experts,
            input_size=self.hidden_size,
            output_size=cfg.intermediate_size*2,
            dtype=cfg.dtype,
            stride=self.gate_up_stride,
        )
        self.down_proj = ExpertFusedRowParallelLinear(
            num_experts=self.num_local_experts,
            input_size=cfg.intermediate_size,
            output_size=self.hidden_size,
            dtype=cfg.dtype
        )

    def forward(self, routed_tokens):
        # routed_tokens shape: [num_experts * tokens_per_expert, hidden_dim]
        e = self.num_local_experts
        D = self.hidden_size

        x = routed_tokens.view(e, -1, D)  # [num_experts, tokens_per_expert, hidden_dim]

        gate_up = self.gate_up_proj(x)  # [num_experts, tokens_per_expert, intermediate_size*2]

        # Split gate and up projections
        gate, up = gate_up.chunk(2, dim=-1)  # Each: [num_experts, tokens_per_expert, intermediate_size]

        # Apply SwiGLU activation
        hidden = F.silu(gate) * up  # [num_experts, tokens_per_expert, intermediate_size]

        out = self.down_proj(hidden)  # [num_experts, tokens_per_expert, hidden_dim]

        return out.view(-1, D) # [num_experts * tokens_per_expert, hidden_dim]


class SharedExperts(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.gate_proj = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=cfg.dtype
        )
        self.up_proj = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=cfg.dtype
        )
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=cfg.dtype
        )

    def forward(self, hidden_states):
        hidden = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden)


class Router(nn.Module):
    def __init__(
        self,
        cfg,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.num_experts = cfg.num_local_experts
        self.top_k = cfg.num_experts_per_tok
        self.hidden_size = cfg.hidden_size

        # Router weights
        self.linear_router = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch*seq_len, hidden_dim]

        Returns:
            router_logits: Raw routing scores [num_experts, batch*seq_len]
            router_indices: Expert assignments [batch*seq_len, top_k]
            router_probs: Routing probabilities [batch*seq_len, top_k]
        """
        batch_seq_len = hidden_states.shape[0]  # batch_size * seq_len

        # Get router logits
        router_logits = self.linear_router(hidden_states.float())  # [batch*seq_len, num_experts]
        router_logits = router_logits.transpose(0, 1)  # [num_experts, batch*seq_len]
        
        # Get top-k experts and scores
        router_top_value, router_indices = torch.topk(
            router_logits.transpose(0, 1).to(torch.float64), 
            k=self.top_k, 
            dim=1
        )

        # Create routing scores
        router_scores = torch.full_like(
            router_logits.transpose(0, 1), 
            float('-inf'),
            dtype=torch.float64
        ).scatter_(
            1, 
            router_indices, 
            router_top_value
        ).transpose(0, 1)

        # Create indices for gathering
        router_indices = torch.arange(
            batch_seq_len, 
            device=hidden_states.device
        ).view(1, -1).expand(router_scores.size(0), -1)

        # Convert to probabilities
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        # Prepare gather indices for the full hidden dimension
        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_states.shape[-1])

        # Prepare router probabilities for scaling
        router_probs = router_scores.reshape(-1, 1)

        return router_scores, router_indices, router_probs


class MOE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.router_dtype = torch.float32

        # Router
        self.router = Router(
            cfg,
            dtype=self.router_dtype
        )

        # Expert networks
        self.experts = Experts(cfg)

        # Shared expert
        self.shared_expert = SharedExperts(cfg)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Keep original shape for shared expert
        shared_output = self.shared_expert(hidden_states)  # [B, S, H]
        
        # Flatten for expert routing
        flat_hidden = hidden_states.view(-1, hidden_dim)  # [B*S, H]
        
        # Get routing information
        router_scores, router_indices, router_probs = self.router(flat_hidden)
        
        # Prepare expert inputs
        expert_inputs = torch.gather(
            input=flat_hidden,
            dim=0,
            index=router_indices,
        )
        expert_inputs = expert_inputs * router_probs

        # Get expert outputs
        expert_outputs = self.experts(expert_inputs)  # [B*S, H]

        # Initialize combined output with shared expert output
        combined_output = torch.zeros_like(flat_hidden)  # [B*S, H]
        combined_output.copy_(shared_output.view(-1, hidden_dim))

        # Combine outputs
        combined_output = combined_output.scatter_add_(
            dim=0,
            index=router_indices,
            src=expert_outputs
        )

        # All-reduce
        if parallel_state.model_parallel_is_initialized():
            group = parallel_state.get_tensor_model_parallel_group()
            torch.distributed.all_reduce(
                combined_output,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )

        return combined_output.view(batch_size, seq_len, hidden_dim)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.gate_proj = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.intermediate_size_mlp,
            bias=False,
            gather_output=False,
            dtype=cfg.dtype,
        )
        self.up_proj = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.intermediate_size_mlp,
            bias=False,
            gather_output=False,
            dtype=cfg.dtype,
        )
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size_mlp,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=cfg.dtype,
        )

    def forward(self, x):
        output = NF.gated_mlp_kernel_unreduced(w_up=self.up_proj, w_gate=self.gate_proj, w_down=self.down_proj, hidden=x)

        # Reduce output manually since the kernel above does not perform a reduction
        if parallel_state.model_parallel_is_initialized():
            output = reduce_from_tensor_model_parallel_region(
                output, process_group=parallel_state.get_tensor_model_parallel_group(as_list=False),
            )

        return output


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config, batch_size: int, seq_len: int, layer_id: int):
        super().__init__()

        self.is_nope_layer = cfg.nope_layer_interval is not None and (layer_id + 1) % cfg.nope_layer_interval == 0

        use_rope = not self.is_nope_layer
        use_qk_norm = cfg.use_qk_norm and not self.is_nope_layer

        self.self_attn = Attention(cfg, batch_size, seq_len, use_rope=use_rope, use_qk_norm=use_qk_norm)

        if (layer_id + 1) % cfg.interleave_moe_layer_step == 0:
            self.feed_forward = MOE(cfg)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg)
        self.post_attention_layernorm = RMSNorm(cfg)

    def forward(
        self,
        x: torch.Tensor,
        last_pos: torch.Tensor,
        rope_cache: torch.Tensor,
        global_attn_mask: Optional[torch.Tensor],
        local_attn_mask: Optional[torch.Tensor],
        is_prefill: bool,
    ):
        # The iRoPE architecture uses global attention mask for NoPE layers or
        # if chunked local attention is not used
        if self.is_nope_layer or local_attn_mask is None:
            mask = global_attn_mask
        else:
            mask = local_attn_mask

        h = x + self.self_attn(self.input_layernorm(x), last_pos, mask, rope_cache, is_prefill)

        out = h + self.feed_forward(self.post_attention_layernorm(h))

        return out


class Transformer(nn.Module):
    def __init__(self, cfg: Config, batch_size: int, seq_len: int):
        super().__init__()

        self.embedding = ParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, shard_across_embedding=True, dtype=cfg.dtype
        )
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, gather_output=True, dtype=cfg.dtype
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(cfg.n_layers):
            self.layers.append(TransformerBlock(cfg, batch_size=batch_size, seq_len=seq_len, layer_id=layer_id))

        self.bs = batch_size
        self.seq_len = seq_len

        self.norm = RMSNorm(cfg)
        self.rope_theta = cfg.rope_theta
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.attention_chunk_size = cfg.attention_chunk_size

    def forward(self, tokens: torch.Tensor, last_pos: torch.Tensor, attention_mask: torch.Tensor):
        # TODO: Refactor forward logic into forward_prefill and forward_decode
        is_prefill = tokens.shape[1] > 1

        # Embedding
        h = self.embedding(tokens)

        global_attn_mask, local_attn_mask = None, None

        self.rope_cache = precompute_rope(
            device=h.device, theta=self.rope_theta, head_dim=self.head_dim, seq_len=self.seq_len
        )

        # Calculate attention mask
        if is_prefill:
            mask = torch.full((self.seq_len, self.seq_len), True, device=attention_mask.device).tril(diagonal=0)
            input_mask = attention_mask[:, None, None, :].expand(self.bs, 1, self.seq_len, self.seq_len).to(torch.bool)
            mask = torch.logical_and(mask, input_mask)
        else:
            mask = attention_mask[:, None, None, :].expand(self.bs, 1, 1, self.seq_len).to(torch.bool)

        global_attn_mask = mask
        local_attn_mask = mask

        # Transformer Decoder blocks
        for layer in self.layers:
            h = layer(h, last_pos, global_attn_mask, local_attn_mask, self.rope_cache, is_prefill)

        if is_prefill:
            last_pos = last_pos.view(self.bs, 1, 1).expand(self.bs, 1, self.hidden_size)
            h = torch.gather(h, dim=1, index=last_pos.to(torch.int64))

        # Final RMS Norm
        h = self.norm(h)
        # lm head
        output = self.lm_head(h).float()

        return output


def load_llama_checkpoint(cfg: Config, model_path: str, tp_degree=1):
    with torch.no_grad():
        state_dict = load_file(model_path, device="cpu")

        # Why do we change the weight keys?
        # The modeling code does not exactly match meta's llama code. This just
        # corrects the state dict so it can be loaded to the modeling code above.

        def replace(state_dict, old_key, new_key):
            return {k.replace(old_key, new_key): v for k, v in state_dict.items()}

        # Replace weights
        state_dict = replace(state_dict, "language_model.", "")
        state_dict = replace(state_dict, "model.", "")
        state_dict = replace(state_dict, "embed_tokens", "embedding")
        state_dict = replace(state_dict, "feed_forward.router", "feed_forward.router.linear_router")
        state_dict = replace(state_dict, "feed_forward.experts.gate_up_proj", "feed_forward.experts.gate_up_proj.weight")
        state_dict = replace(state_dict, "feed_forward.experts.down_proj", "feed_forward.experts.down_proj.weight")


        # Cast weights to cfg dtype
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(cfg.dtype)

        for lay in range(cfg.n_layers):
            # Upcast router weights to fp32
            router_key = f"layers.{lay}.feed_forward.router.linear_router.weight"
            if router_key in state_dict:
                state_dict[router_key] = state_dict[router_key].to(torch.float32)

            wq = state_dict[f"layers.{lay}.self_attn.q_proj.weight"]
            wk = state_dict[f"layers.{lay}.self_attn.k_proj.weight"]
            wv = state_dict[f"layers.{lay}.self_attn.v_proj.weight"]
            wo = state_dict[f"layers.{lay}.self_attn.o_proj.weight"]

            if tp_degree > 1:
                n_heads = cfg.n_heads
                n_kv_heads = cfg.n_kv_heads
                head_dim = cfg.head_dim

                # Calculate padding for Q heads
                padded_n_heads = ((n_heads + tp_degree - 1) // tp_degree) * tp_degree
                n_heads_to_pad = padded_n_heads - n_heads

                # For KV heads, we need at least 1 head per rank
                n_kv_repeat = tp_degree // n_kv_heads

                # GQA
                # Interleaved padding for Q heads - REPLCATE_TO_TP_DEGREE strategy
                wq = wq.view(n_heads, head_dim, cfg.hidden_size)
                if n_heads_to_pad > 0:
                    padding = torch.zeros(
                        n_heads_to_pad,
                        head_dim,
                        cfg.hidden_size,
                        dtype=wq.dtype,
                        device=wq.device
                    )
                    # Pad interleaved
                    source_group_size = n_heads // n_kv_heads
                    splits = torch.split(wq, source_group_size, dim=0)
                    pad_size = padding.shape[0] // len(splits)
                    pads = [padding[i*pad_size:(i+1)*pad_size] for i in range(len(splits))]
                    
                    interleaved = [t for pair in zip(splits, pads) for t in pair]
                    wq = torch.cat(interleaved, dim=0)

                wq = wq.reshape(-1, cfg.hidden_size)

                # Handle K, V projections
                wk = wk.view(n_kv_heads, head_dim, cfg.hidden_size)
                wv = wv.view(n_kv_heads, head_dim, cfg.hidden_size)

                wk = torch.repeat_interleave(wk, repeats=n_kv_repeat, dim=0)
                wv = torch.repeat_interleave(wv, repeats=n_kv_repeat, dim=0)

                wk = wk.reshape(-1, cfg.hidden_size)
                wv = wv.reshape(-1, cfg.hidden_size)

                # Pad output projection with interleaved padding
                wo = wo.view(cfg.hidden_size, n_heads, head_dim)  # [hidden_size, n_heads, head_dim]
                if n_heads_to_pad > 0:
                    padding = torch.zeros(
                        cfg.hidden_size,
                        n_heads_to_pad,
                        head_dim,
                        dtype=wo.dtype,
                        device=wo.device
                    )
                    # Pad interleaved
                    source_group_size = n_heads // n_kv_heads
                    splits = torch.split(wo, source_group_size, dim=1)  # Split along n_heads dimension
                    pad_size = padding.shape[1] // len(splits)  # padding heads per group
                    pads = [padding[:, i*pad_size:(i+1)*pad_size] for i in range(len(splits))]

                    # Interleave the splits and padding
                    interleaved = [t for pair in zip(splits, pads) for t in pair]
                    wo = torch.cat(interleaved, dim=1)

                wo = wo.reshape(cfg.hidden_size, -1)  # [hidden_size, padded_n_heads * head_dim]

                # Verify shapes
                assert wq.size(0) % tp_degree == 0, f"Q weight of shape {wq.shape[0]} should be divisible by tp_degree."
                assert wk.size(0) % tp_degree == 0, f"K weight of shape {wk.shape[0]} should be divisible by tp_degree."
                assert wv.size(0) % tp_degree == 0, f"V weight of shape {wv.shape[0]} should be divisible by tp_degree."
                assert wo.size(1) % tp_degree == 0, f"O weight of shape {wo.shape[1]} should be divisible by tp_degree."

                # Split into chunks for each rank
                wq_list = wq.chunk(tp_degree, dim=0)
                wk_list = wk.chunk(tp_degree, dim=0)
                wv_list = wv.chunk(tp_degree, dim=0)

                # Concatenate QKV for each rank
                wqkv_list = [
                    torch.cat([wq_list[i], wk_list[i], wv_list[i]], dim=0)
                    for i in range(tp_degree)
                ]

                wqkv = torch.cat(wqkv_list, dim=0)
            else:
                # For tp_degree=1, just concatenate Q,K,V weights
                wqkv = torch.cat([wq, wk, wv], dim=0)

            state_dict[f"layers.{lay}.self_attn.wqkv.weight"] = wqkv
            state_dict[f"layers.{lay}.self_attn.o_proj.weight"] = wo

            # Clean up individual weights
            del state_dict[f"layers.{lay}.self_attn.q_proj.weight"]
            del state_dict[f"layers.{lay}.self_attn.k_proj.weight"]
            del state_dict[f"layers.{lay}.self_attn.v_proj.weight"]

    return state_dict
