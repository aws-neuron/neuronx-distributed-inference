# flake8: noqa

# Original copyright (c) Meta Platforms, Inc. and affiliates.
# Please check the LICENSE under the same director for details.
#
# Modified by Amazon


import math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_tensor_model_parallel_region_with_dim,
)
from torch import nn
from torch_neuronx.xla_impl.ops import RmsNorm
from neuronx_distributed_inference.experimental.core.config import AttentionConfig, BuildConfig

import neuronx_distributed_inference.experimental.functional as NF


SEQUENCE_DIM = 1


@dataclass
class Llama3ModelConfig:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    dtype: torch.dtype
    rms_norm_eps: int
    rope_theta: int
    pad_token: int


@dataclass
class Llama3Config:
    model: Llama3ModelConfig
    attention: AttentionConfig
    build: BuildConfig


def precompute_rope(device, theta, head_dim, seq_len):
    # Refer: https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32
    theta = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim)
    )

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


class RMSNorm(nn.Module):
    """
    ref : https://github.com/meta-llama/llama3/blob/main/llama/model.py
    In a production usecase, from torch_neuronx.xla_impl.ops import RmsNorm
    """

    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.eps = cfg.model.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(cfg.model.hidden_size, dtype=cfg.model.dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if x.is_cpu:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
        else:
            return RmsNorm.apply(x, self.weight, self.eps, len(x.shape) - 1).to(x.dtype)

# TODO
class Llama3Attention(nn.Module):

    def __init__(self, cfg: Llama3Config, batch_size: int, seq_len: int):
        super().__init__()

        self.world_size = cfg.build.world_size
        self.cp_degree = cfg.attention.cp_degree
        self.tp_degree = self.world_size // self.cp_degree

        NF.initialize_context_parallel_process_groups(self.world_size, self.cp_degree)
        self.tensor_parallel_group = NF.get_context_parallel_tp_group()
        self.context_parallel_group = NF.get_context_parallel_cp_group()

        if cfg.model.n_heads % self.tp_degree != 0:
            print("n_heads not evenly divisible by tp degree. Padding...")
            # pad to closest tp_degree multiple
            self.n_heads = ((self.n_heads + self.tp_degree -1) // self.tp_degree) * self.tp_degree

        # we want atleast 1 kv head on a core
        self.n_heads = cfg.model.n_heads // self.tp_degree
        self.n_kv_heads = max(cfg.model.n_kv_heads // self.tp_degree, 1)

        self.wqkv = ColumnParallelLinear(
            cfg.model.hidden_size,
            (self.n_heads + 2 * self.n_kv_heads) * cfg.model.head_dim * self.tp_degree,
            bias=False,
            gather_output=False,
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )
        self.wo = RowParallelLinear(
            self.n_heads * self.tp_degree * cfg.model.head_dim,
            cfg.model.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_output=False,
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )

        self.register_buffer(
            "cache_k",
            torch.zeros(
                batch_size,
                seq_len,
                self.n_kv_heads,
                cfg.model.head_dim,
                requires_grad=False,
                dtype=cfg.model.dtype,
            ),
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                batch_size,
                seq_len,
                self.n_kv_heads,
                cfg.model.head_dim,
                requires_grad=False,
                dtype=cfg.model.dtype,
            ),
        )

        self.rank_util = SPMDRank(self.world_size)

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.model.head_dim

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
        mask: Optional[torch.Tensor],
        rope_cache: torch.tensor,
        is_prefill: bool,
    ):
        if is_prefill:
            x = gather_from_tensor_model_parallel_region_with_dim(
                x, gather_dim=SEQUENCE_DIM, process_group=self.tensor_parallel_group
            )

        # x (Batch, Sequence, Hidden)
        bsz, seq_len, hidden_dim = x.shape

        is_context_parallel = seq_len > 1 and self.cp_degree > 1

        if is_context_parallel:
            mask = NF.split_input_for_context_parallel(
                mask, dim=2, world_size=self.world_size, cp_degree=self.cp_degree, rank_util=self.rank_util,
            )
            rope_cache = NF.split_input_for_context_parallel(
                rope_cache, dim=0, world_size=self.world_size, cp_degree=self.cp_degree, rank_util=self.rank_util,
            )

        # BSH

        # NOTE: NF.qkv_proj() assertions are incorrect
        # q, k, v = NF.qkv_proj(
        #     self.wqkv, x, self.n_heads, self.n_kv_heads, self.head_dim, self.tp_degree
        # )

        # TODO: flip to NF.qkv_proj() after the assertions are fixed
        q, k, v = self._qkv_proj(self.wqkv, x, self.n_heads, self.n_kv_heads, self.head_dim)

        # BSNH
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        if seq_len > 1:
            start_pos = torch.zeros_like(last_pos)
            q, k = rope(q, start_pos, rope_cache), rope(k, start_pos, rope_cache)
        else:
            start_pos = last_pos
            q, k = rope(q, start_pos, rope_cache), rope(k, start_pos, rope_cache)

        # Gather KV along sequence dimension if running CP attention
        if is_context_parallel:
            k, v = NF.gather_kv_context_parallel(k, v, dim=1, process_group=self.context_parallel_group)

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
        # operate on a fixed sequence length per compilation.

        # keys = self.cache_k[:bsz, start_pos : start_pos + inp_len]   X wrong
        # values = self.cache_v[:bsz, start_pos : start_pos + inp_len] X wrong

        # Yes fixed shape will cause us to waste compute. The way to work around that is to 'bucket' - compile for many
        # shapes. One way is to bucket along sequence length. Here we can slice the KV cache when you bucket along
        # sequence length. This is an easy optimization you could do.

        # TODO: improve this part in the attention & KV cache workstream
        kv_len_bucket = mask.shape[-1]
        keys = self.cache_k[:, :kv_len_bucket, :, :]
        values = self.cache_v[:, :kv_len_bucket, :, :]

        # With GQA, k/v heads are shared amond different q heads
        # repeat k/v heads to match q heads
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.n_rep)

        # bs, seqlen, head, head_dim -> bs, head, seqlen, head_dim
        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Q.K^T/√d
        scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # You cannot use -inf well on Neuron, you will run into 1003 errors
            scores = torch.where(mask, scores, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        output = self.wo(output)

        # Llama3 has some accuracy issues when doing reduction in lower precision, so we upcast here
        original_dtype = output.dtype
        output = output.to(torch.float32)
        if is_prefill:
            output = reduce_scatter_to_tensor_model_parallel_region_with_dim(
                output, SEQUENCE_DIM, self.tensor_parallel_group
                )
        else:
            output = reduce_from_tensor_model_parallel_region(
                output, process_group=self.tensor_parallel_group
            )
        output = output.to(original_dtype)

        return output


class Llama3MLP(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()

        self.tensor_parallel_group = parallel_state.get_tensor_model_parallel_group()

        self.gate_proj = ColumnParallelLinear(
            cfg.model.hidden_size,
            cfg.model.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )
        self.up_proj = ColumnParallelLinear(
            cfg.model.hidden_size,
            cfg.model.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )
        self.down_proj = RowParallelLinear(
            cfg.model.intermediate_size,
            cfg.model.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )

    def forward(self, x: torch.Tensor, is_prefill: bool):
        # TODO: Remove this branching when adding support for CPU distributed llama. 
        # Requires us to add an a new API such as gated_mlp_unreduced()
        if x.device.type == "cpu":
            return NF.gated_mlp(
                w_up=self.up_proj,
                w_gate=self.gate_proj,
                w_down=self.down_proj,
                hidden=x,
                tensor_parallel_group=self.tensor_parallel_group,
            )

        # We use sequence parallel during prefill, so we need to gather the input
        if is_prefill:
            x = gather_from_tensor_model_parallel_region_with_dim(
                x, gather_dim=SEQUENCE_DIM, process_group=self.tensor_parallel_group
            )

        output = NF.gated_mlp_kernel_unreduced(w_up=self.up_proj, w_gate=self.gate_proj, w_down=self.down_proj, hidden=x)

        if is_prefill:
            output = reduce_scatter_to_tensor_model_parallel_region_with_dim(
                output, partition_dim=SEQUENCE_DIM, process_group=self.tensor_parallel_group,
            )
        else:
            output = reduce_from_tensor_model_parallel_region(
                output, process_group=self.tensor_parallel_group,
            )

        return output


class Llama3TransformerBlock(torch.nn.Module):
    def __init__(self, cfg: Llama3Config, batch_size: int, seq_len: int):
        super().__init__()
        self.attention = Llama3Attention(cfg, batch_size, seq_len)
        self.mlp = Llama3MLP(cfg)
        self.attention_norm = RMSNorm(cfg)
        self.mlp_norm = RMSNorm(cfg)

    def forward(
        self,
        x: torch.Tensor,
        last_pos: torch.Tensor,
        mask: torch.Tensor,
        rope_cache: torch.Tensor,
        is_prefill: bool,
    ):

        norm_h = self.attention_norm(x)
        attn_h = self.attention(norm_h, last_pos, mask, rope_cache, is_prefill)
        attn_h = x + attn_h

        norm_h = self.mlp_norm(attn_h)
        mlp_h = self.mlp(norm_h, is_prefill)
        out = attn_h + mlp_h

        return out


class Llama3Transformer(torch.nn.Module):

    def __init__(self, cfg: Llama3Config, batch_size: int, seq_len: int):
        super().__init__()

        self.tensor_parallel_group = parallel_state.get_tensor_model_parallel_group()

        # We turn off collectives in this layer to allow flexibility in forward pass
        # to use sequence parallel for prefill
        self.embedding = ParallelEmbedding(
            num_embeddings=cfg.model.vocab_size,
            embedding_dim=cfg.model.hidden_size,
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
            shard_across_embedding=False,
            use_spmd_rank=True,
            collect_output=False,
        )

        self.output = ColumnParallelLinear(
            cfg.model.hidden_size, 
            cfg.model.vocab_size, 
            bias=False, 
            gather_output=True, 
            dtype=cfg.model.dtype,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )

        self.layers = torch.nn.ModuleList()
        for _ in range(cfg.model.n_layers):
            self.layers.append(Llama3TransformerBlock(cfg, batch_size=batch_size, seq_len=seq_len))

        self.bs = batch_size
        self.seq_len = seq_len

        self.norm = RMSNorm(cfg)
        self.rope_theta = cfg.model.rope_theta
        self.head_dim = cfg.model.head_dim
        self.hidden_size = cfg.model.hidden_size
        self.vocab_size = cfg.model.vocab_size

    def forward(self, tokens: torch.Tensor, last_pos: torch.Tensor, attention_mask: torch.Tensor):
        is_prefill = tokens.shape[1] > 1

        h = self.embedding(tokens)
        if is_prefill:
            # We use sequence parallel (SP) during prefill
            h = reduce_scatter_to_tensor_model_parallel_region_with_dim(
                h, partition_dim=SEQUENCE_DIM, process_group=self.tensor_parallel_group,
            )
        else:
            h = reduce_from_tensor_model_parallel_region(
                h, process_group=self.tensor_parallel_group,
            )

        # q_len is just the len of current input tokens (prompt_len for
        # prefill and 1 for decode). kv_len is the length of both current
        # input tokens and previous computed tokens.
        # kv_len = q_len + len_computed_tokens
        q_len = tokens.shape[1]
        kv_len = attention_mask.shape[1]

        self.rope_cache = precompute_rope(
            device=h.device, theta=self.rope_theta, head_dim=self.head_dim, seq_len=kv_len
        )

        mask = None
        if is_prefill:
            assert q_len == kv_len
            mask = torch.full(
                (q_len, q_len), True, device=attention_mask.device
            ).tril(diagonal=0)
            input_mask = (
                attention_mask[:, None, None, :]
                .expand(self.bs, 1, q_len, q_len)
                .to(torch.bool)
            )
            mask = torch.logical_and(mask, input_mask)
        else:
            mask = (
                attention_mask[:, None, None, :].expand(self.bs, 1, 1, kv_len).to(torch.bool)
            )

        for layer in self.layers:
            h = layer(h, last_pos, mask, self.rope_cache, is_prefill)

        h = self.norm(h)

        # We need to gather the hidden states since we are running sequence parallel
        if is_prefill:
            h = gather_from_tensor_model_parallel_region_with_dim(
                h, gather_dim=SEQUENCE_DIM, process_group=self.tensor_parallel_group,
            )

        output = self.output(h).float()

        # We return the logits for the last token per batch.
        # This is a simple optimization to stop moving sequence length long
        # logits back from device to CPU for prefil.
        if is_prefill:
            last_pos = last_pos.view(self.bs, 1, 1).expand(self.bs, 1, self.vocab_size)
            output = torch.gather(output, dim=1, index=last_pos.to(torch.int64))
        # Note: We are returning K and V caches. The order in which the tensors
        # are returned is important as you will need to register the alias when
        # tracing the model.

        if output.is_cpu:
            return output
        else:
            return output


def load_llama_checkpoint(cfg: Llama3Config, model_path: str):

    with torch.no_grad():
        # Download model from : https://www.llama.com/llama-downloads/
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)

        # Why do we change the weight keys?
        # The modeling code does not exactly match meta's llama code. This just
        # corrects the state dict so it can be loaded to the modeling code above.

        def replace(state_dict, old_key, new_key):
            return {k.replace(old_key, new_key): v for k, v in state_dict.items()}

        state_dict = replace(state_dict, "tok_embeddings", "embedding")
        state_dict = replace(state_dict, "feed_forward", "mlp")
        state_dict = replace(state_dict, "ffn_norm", "mlp_norm")
        state_dict = replace(state_dict, "mlp.w1", "mlp.gate_proj")
        state_dict = replace(state_dict, "mlp.w3", "mlp.up_proj")
        state_dict = replace(state_dict, "mlp.w2", "mlp.down_proj")

        # We use SPMDRank within the embedding layer
        state_dict[f"embedding.rank_util.rank"] = torch.arange(0, cfg.build.world_size, dtype=torch.int32)

        # The embedding and modeling head outputs are tied. We just clone because
        # model builder's sharding logic does not take care of tied weights yet.
        state_dict["output.weight"] = state_dict["embedding.weight"].clone().detach()

        # Calculate attention TP degree based on world size and CP degree
        attention_tp_degree = cfg.build.world_size // cfg.attention.cp_degree

        # fused QKV weghts
        for lay in range(cfg.model.n_layers):
            wq = state_dict[f"layers.{lay}.attention.wq.weight"]
            wk = state_dict[f"layers.{lay}.attention.wk.weight"]
            wv = state_dict[f"layers.{lay}.attention.wv.weight"]
            wo = state_dict[f"layers.{lay}.attention.wo.weight"]

            if attention_tp_degree > 1:
                n_heads = cfg.model.n_heads
                n_kv_heads = cfg.model.n_kv_heads
                head_dim = cfg.model.head_dim

                # Calculate padding for Q heads
                padded_n_heads = ((n_heads + attention_tp_degree - 1) // attention_tp_degree) * attention_tp_degree
                n_heads_to_pad = padded_n_heads - n_heads

                # For KV heads, we need at least 1 head per rank
                n_kv_repeat = attention_tp_degree // n_kv_heads

                # GQA
                # Interleaved padding for Q heads - REPLCATE_TO_TP_DEGREE strategy
                wq = wq.view(n_heads, head_dim, cfg.model.hidden_size)
                if n_heads_to_pad > 0:
                    padding = torch.zeros(
                        n_heads_to_pad,
                        head_dim,
                        cfg.model.hidden_size,
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

                wq = wq.reshape(-1, cfg.model.hidden_size)

                # Handle K, V projections
                wk = wk.view(n_kv_heads, head_dim, cfg.model.hidden_size)
                wv = wv.view(n_kv_heads, head_dim, cfg.model.hidden_size)

                wk = torch.repeat_interleave(wk, repeats=n_kv_repeat, dim=0)
                wv = torch.repeat_interleave(wv, repeats=n_kv_repeat, dim=0)

                wk = wk.reshape(-1, cfg.model.hidden_size)
                wv = wv.reshape(-1, cfg.model.hidden_size)

                # Pad output projection with interleaved padding
                wo = wo.view(cfg.model.hidden_size, n_heads, head_dim)  # [hidden_size, n_heads, head_dim]
                if n_heads_to_pad > 0:
                    padding = torch.zeros(
                        cfg.model.hidden_size,
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

                wo = wo.reshape(cfg.model.hidden_size, -1)  # [hidden_size, padded_n_heads * head_dim]

                # Verify shapes
                assert wq.size(0) % attention_tp_degree == 0, f"Q weight of shape {wq.shape[0]} should be divisible by tp_degree."
                assert wk.size(0) % attention_tp_degree == 0, f"K weight of shape {wk.shape[0]} should be divisible by tp_degree."
                assert wv.size(0) % attention_tp_degree == 0, f"V weight of shape {wv.shape[0]} should be divisible by tp_degree."
                assert wo.size(1) % attention_tp_degree == 0, f"O weight of shape {wo.shape[1]} should be divisible by tp_degree."

                wq_list = wq.chunk(attention_tp_degree, dim=0)
                wk_list = wk.chunk(attention_tp_degree, dim=0)
                wv_list = wv.chunk(attention_tp_degree, dim=0)

                # Concatenate QKV for each rank
                wqkv_list = [
                    torch.cat([wq_list[i], wk_list[i], wv_list[i]], dim=0)
                    for i in range(attention_tp_degree)
                ]

                wqkv = torch.cat(wqkv_list, dim=0)
            else:
                # For tp_degree=1, just concatenate Q,K,V weights
                wqkv = torch.cat([wq, wk, wv], dim=0)

            state_dict[f"layers.{lay}.attention.wqkv.weight"] = wqkv
            state_dict[f"layers.{lay}.attention.wo.weight"] = wo

            # Clean up individual weights
            del state_dict[f"layers.{lay}.attention.wq.weight"]
            del state_dict[f"layers.{lay}.attention.wk.weight"]
            del state_dict[f"layers.{lay}.attention.wv.weight"]

            # For accessing rank
            state_dict[f"layers.{lay}.attention.rank_util.rank"] = torch.arange(0, cfg.build.world_size, dtype=torch.int32)

    return state_dict
