"""Qwen3-Omni Talker Code Predictor on Neuron.

The code predictor runs once per talker decode step:
  1. Prefill with 2 tokens (past_hidden, last_id_hidden) → code 0 logits + KV
  2. 14 decode steps, each consuming the previous argmax'd code, producing
     codes 1..14 and per-step hidden states.

Total 15 residual codes are produced; only 14 decode hidden states are
consumed by the talker (mid_residual_hiddens). The prefill's KV cache has
length 2; decode extends it up to length 16.

We compile a single NEFF: a 16-token-long causal self-attention over a fixed
input buffer, driven by a runtime state machine external to the NEFF (the
host Python code does the greedy argmax + embedding lookup between NEFF
invocations). The NEFF is invoked 15 times per talker step:
  - Invocation 0: prefill (2 "valid" positions, 14 masked)
  - Invocation 1..14: decode (i+1 valid positions)

This avoids having to trace a KV-cache scatter op or multiple NEFFs.

Architecture (from config.talker_config.code_predictor_config):
  - hidden_size=1024, num_hidden_layers=5, dense GQA
  - num_attention_heads=16, num_key_value_heads=8, head_dim=128
  - intermediate_size=3072, SwiGLU MLP
  - q_norm + k_norm (per-head-dim RMSNorm)
  - Plain 1D RoPE (no MRoPE), theta=1e6
  - 15 codec_embedding tables + 15 lm_heads

TP sharding at TP=8: 2 Q heads/rank, 1 KV head/rank, dense MLP sharded
as ColumnParallel (3072/8=384) + RowParallel.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

logger = logging.getLogger("Neuron")


# Constants from config.talker_config.code_predictor_config
HIDDEN_SIZE = 1024
NUM_LAYERS = 5
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 3072
VOCAB_SIZE = 2048
NUM_CODE_GROUPS = 16
NUM_EMBED_TABLES = NUM_CODE_GROUPS - 1  # 15
NUM_LM_HEADS = NUM_CODE_GROUPS - 1       # 15
# Total positions across a full prefill+decode cycle = 2 + 14 = 16.
MAX_SEQ_LEN = 16
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32 * self.weight).to(dtype)


def _compute_rope(dim: int, max_pos: int, base: float = ROPE_THETA):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CPAttention(nn.Module):
    def __init__(self, tp_degree: int, dtype=torch.bfloat16):
        super().__init__()
        self.tp_degree = tp_degree
        self.num_heads_per_rank = NUM_ATTN_HEADS // tp_degree
        self.num_kv_heads_per_rank = max(NUM_KV_HEADS // tp_degree, 1)
        self.num_key_value_groups = self.num_heads_per_rank // self.num_kv_heads_per_rank
        self.scaling = HEAD_DIM ** -0.5

        self.q_proj = ColumnParallelLinear(
            HIDDEN_SIZE, NUM_ATTN_HEADS * HEAD_DIM, bias=False,
            gather_output=False, dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False,
            gather_output=False, dtype=dtype,
        )
        self.v_proj = ColumnParallelLinear(
            HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False,
            gather_output=False, dtype=dtype,
        )
        self.o_proj = RowParallelLinear(
            NUM_ATTN_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False,
            input_is_parallel=True, dtype=dtype,
        )
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, x, cos, sin, causal_mask):
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, S, self.num_heads_per_rank, HEAD_DIM)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, S, self.num_kv_heads_per_rank, HEAD_DIM)).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads_per_rank, HEAD_DIM).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        if self.num_key_value_groups > 1:
            k_r = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v_r = v.repeat_interleave(self.num_key_value_groups, dim=1)
        else:
            k_r = k
            v_r = v

        scores = torch.matmul(q, k_r.transpose(-2, -1)) * self.scaling
        scores = scores + causal_mask
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v_r).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class CPMLP(nn.Module):
    def __init__(self, tp_degree: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            HIDDEN_SIZE, INTERMEDIATE, bias=False, gather_output=False, dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            HIDDEN_SIZE, INTERMEDIATE, bias=False, gather_output=False, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            INTERMEDIATE, HIDDEN_SIZE, bias=False, input_is_parallel=True, dtype=dtype,
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CPLayer(nn.Module):
    def __init__(self, tp_degree: int, dtype=torch.bfloat16):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN_SIZE)
        self.self_attn = CPAttention(tp_degree, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE)
        self.mlp = CPMLP(tp_degree, dtype=dtype)

    def forward(self, x, cos, sin, causal_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, causal_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NeuronCodePredictor(nn.Module):
    """One NEFF that re-runs the full causal self-attention over a
    MAX_SEQ_LEN=16 input buffer each step.

    Inputs (all fixed shape):
      inputs_embeds:  [1, MAX_SEQ_LEN, HIDDEN]
      position_ids:   [1, MAX_SEQ_LEN]
      mask_1d:        [1, MAX_SEQ_LEN]  — 1 for valid, 0 for masked

    Internally builds a [1,1,MAX,MAX] causal mask that also masks out the
    invalid suffix (set to -inf where mask_1d==0).

    Output:
      hidden:         [1, MAX_SEQ_LEN, HIDDEN]  (pre-lm_head; caller picks
                                                  the last-valid position)
    """

    def __init__(self, tp_degree: int, dtype=torch.bfloat16):
        super().__init__()
        self.tp_degree = tp_degree
        self.dtype = dtype
        self.layers = nn.ModuleList([CPLayer(tp_degree, dtype=dtype) for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN_SIZE)

        cos, sin = _compute_rope(HEAD_DIM, MAX_SEQ_LEN)
        self.register_buffer("cos_cache", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cache", sin.to(dtype), persistent=False)

    def forward(self, inputs_embeds, position_ids, mask_1d):
        # Build causal + validity mask [1, 1, MAX, MAX]
        # Use a "small" negative value (not finfo.min) to avoid -inf overflow
        # when causal and key masks overlap, which on bf16/Neuron can produce NaN.
        MASK_VAL = -1e4
        S = MAX_SEQ_LEN
        base_dtype = inputs_embeds.dtype
        causal = torch.triu(
            torch.full((S, S), MASK_VAL, dtype=base_dtype),
            diagonal=1,
        ).view(1, 1, S, S)
        # mask out invalid keys (k masked columns)
        key_mask = (mask_1d == 0).view(1, 1, 1, S).to(base_dtype) * MASK_VAL
        attn_mask = causal + key_mask  # broadcasted, min value 2*MASK_VAL is fine

        cos = self.cos_cache[position_ids[0]]
        sin = self.sin_cache[position_ids[0]]

        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# UnifiedNeuronCodePredictor: runs all 15 steps in one NEFF call.
# ---------------------------------------------------------------------------

class UnifiedNeuronCodePredictor(nn.Module):
    """Single-NEFF unrolled code predictor.

    Inputs:
      past_hidden:     [1, 1, 1024]  (talker's last hidden)
      last_id_hidden:  [1, 1, 1024]  (embedding of last predicted talker token)

    Outputs:
      codes:           [1, 15]       (int32 residual codes)
      mid_hiddens:     [1, 14, 1024] (hidden states from decode steps 1..14)

    Internally:
      - Builds a 16-token buffer [past_hidden, last_id_hidden, z, z, ..., z]
      - Runs 15 unrolled rounds. Round 0 predicts code[0] from the last
        valid position. Rounds 1..14 embed the previous code via
        codec_embedding[gs-1], put it at slot (1+gs), extend mask, rerun the
        5-layer attention, and apply lm_head[gs].
      - This uses the same "rerun full attention" pattern as the non-unified
        predictor (no KV cache). 15 × full 16-pos attention is still small
        since each layer is 5 × 1024-dim × 16 positions.

    The codec_embedding and lm_head tensors are replicated on every rank
    (not TP-sharded) because they are small (15 × 2048 × 1024 = 31 MB) and
    used inside a tight loop.
    """

    def __init__(self, tp_degree: int, dtype=torch.bfloat16):
        super().__init__()
        self.tp_degree = tp_degree
        self.dtype = dtype
        self.layers = nn.ModuleList([CPLayer(tp_degree, dtype=dtype) for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN_SIZE)

        cos, sin = _compute_rope(HEAD_DIM, MAX_SEQ_LEN)
        self.register_buffer("cos_cache", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cache", sin.to(dtype), persistent=False)

        # Stacked codec_embedding tables: [15, VOCAB, HIDDEN]
        self.codec_embedding_stacked = nn.Parameter(
            torch.zeros(NUM_EMBED_TABLES, VOCAB_SIZE, HIDDEN_SIZE, dtype=dtype),
            requires_grad=False,
        )
        # Stacked lm_head weights: [15, VOCAB, HIDDEN]
        self.lm_head_stacked = nn.Parameter(
            torch.zeros(NUM_LM_HEADS, VOCAB_SIZE, HIDDEN_SIZE, dtype=dtype),
            requires_grad=False,
        )

    def _attention_mask(self, mask_1d, base_dtype):
        """Build [1, 1, MAX, MAX] causal + validity mask."""
        MASK_VAL = -1e4
        S = MAX_SEQ_LEN
        causal = torch.triu(
            torch.full((S, S), MASK_VAL, dtype=base_dtype),
            diagonal=1,
        ).view(1, 1, S, S)
        key_mask = (mask_1d == 0).view(1, 1, 1, S).to(base_dtype) * MASK_VAL
        return causal + key_mask

    def _run_layers(self, buf, attn_mask, cos, sin):
        x = buf
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        return self.norm(x)

    def forward(self, past_hidden, last_id_hidden):
        """Unroll 15 code-prediction steps in one NEFF call.

        past_hidden / last_id_hidden: [1, 1, HIDDEN], dtype=bfloat16.
        """
        dtype = past_hidden.dtype
        device = past_hidden.device
        B = past_hidden.shape[0]  # always 1 for our pipeline

        # Fixed-shape 16-position buffer
        buf = torch.zeros(B, MAX_SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device=device)
        # Slot 0 = past_hidden, slot 1 = last_id_hidden
        buf = buf.clone()  # ensure writable in trace
        buf[:, 0:1, :] = past_hidden
        buf[:, 1:2, :] = last_id_hidden

        # Validity mask: both prefill positions are valid
        mask_1d = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.int32, device=device)
        mask_1d[:, 0] = 1
        mask_1d[:, 1] = 1

        position_ids = torch.arange(MAX_SEQ_LEN, dtype=torch.int32, device=device).unsqueeze(0)
        cos = self.cos_cache[position_ids[0]]  # [MAX, HEAD_DIM]
        sin = self.sin_cache[position_ids[0]]

        # Round 0: prefill, predict code[0] with lm_head[0] from position 1
        attn_mask = self._attention_mask(mask_1d, dtype)
        hidden = self._run_layers(buf, attn_mask, cos, sin)  # [B, MAX, H]
        last_valid_hidden = hidden[:, 1:2, :]  # [B, 1, H]
        logits0 = last_valid_hidden @ self.lm_head_stacked[0].transpose(-1, -2).to(dtype)
        # argmax doesn't have bf16 → int on Neuron; use float()
        code = logits0.float().argmax(dim=-1)  # [B, 1], int64
        code = code.to(torch.int32)

        codes_list = [code]
        mid_hiddens_list = []

        # Rounds 1..14: decode (produces codes 1..14, plus mid_hiddens 1..14)
        # Total codes = 1 (prefill) + 14 (decode) = 15. slot goes 2..15.
        for gs in range(1, NUM_EMBED_TABLES):
            # Embed the previously-predicted code using codec_embedding[gs-1]
            # code shape [B, 1], flatten to [B] then index_select
            emb_table = self.codec_embedding_stacked[gs - 1]  # [VOCAB, HIDDEN]
            new_embed = F.embedding(code, emb_table)  # [B, 1, HIDDEN]

            # Write into buf at position (1 + gs)
            slot = 1 + gs
            # Explicit clone to keep NEFF happy about mutable buffer writes
            buf = buf.clone()
            buf[:, slot:slot + 1, :] = new_embed

            # Extend validity mask
            mask_1d = mask_1d.clone()
            mask_1d[:, slot] = 1

            attn_mask = self._attention_mask(mask_1d, dtype)
            hidden = self._run_layers(buf, attn_mask, cos, sin)
            # Take the newly-computed position's hidden as "mid residual"
            this_hidden = hidden[:, slot:slot + 1, :]  # [B, 1, H]
            mid_hiddens_list.append(this_hidden)

            # Predict next code via lm_head[gs]
            head_w = self.lm_head_stacked[gs].transpose(-1, -2).to(dtype)
            next_logits = this_hidden @ head_w  # [B, 1, V]
            code = next_logits.float().argmax(dim=-1).to(torch.int32)  # [B, 1]
            codes_list.append(code)

        # Stack outputs
        codes_out = torch.cat(codes_list, dim=1)  # [B, 15]
        mid_hiddens_out = torch.cat(mid_hiddens_list, dim=1)  # [B, 14, H]
        return codes_out, mid_hiddens_out
