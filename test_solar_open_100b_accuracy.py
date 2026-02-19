"""
Accuracy test for Solar Open 100B MoE NXD inference vs CPU reference.

Tests a 2-layer random model with the upstage/Solar-Open-100B architecture:
- hidden_size=4096, n_routed_experts=128, num_experts_per_tok=8
- YaRN RoPE scaling (factor=2.0, original_max_position_embeddings=65536)
- Per-expert weight format (matching actual HF checkpoint)
- tp_degree=4, moe_tp_degree=4, moe_ep_degree=2

The CPU reference model (SolarOpen100BReferenceModel) loads per-expert weights
from the safetensors checkpoint and runs a pure-PyTorch forward pass.
With greedy decoding (top_k=1) and identical weights, the Neuron model output
must match the CPU reference exactly.

Usage:
    # Create random model first:
    python create_solar_open_100b_random.py

    # Compile and test:
    python test_solar_open_100b_accuracy.py --compile

    # Test only (assumes model is already compiled):
    python test_solar_open_100b_accuracy.py
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load

# ============================================================================
# YaRN RoPE (CPU reference implementation)
# ============================================================================


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = max(
        math.floor(
            _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        ),
        0,
    )
    high = min(
        math.ceil(
            _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        ),
        dim - 1,
    )
    return low, high


def _yarn_linear_ramp_mask(low, high, dim):
    if low == high:
        high += 0.001  # avoid division by zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class YarnRotaryEmbedding(nn.Module):
    """CPU reference YaRN RoPE matching DeepseekV3YarnRotaryEmbedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        dim = self.dim
        base = self.base
        scaling_factor = self.scaling_factor
        original_max = self.original_max_position_embeddings

        freq_extra = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        freq_inter = 1.0 / (
            scaling_factor
            * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        low, high = _yarn_find_correction_range(
            self.beta_slow, self.beta_fast, dim, base, original_max
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer = lambda *a, **kw: None  # no-op for plain nn.Module
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._cached_len = seq_len

    def forward(self, position_ids: torch.Tensor):
        max_pos = int(position_ids.max().item()) + 1
        if max_pos > self._cached_len:
            self._build_cache(max_pos)
        cos = self._cos[position_ids]  # [B, S, dim]
        sin = self._sin[position_ids]
        return cos, sin


# ============================================================================
# Standard RMSNorm
# ============================================================================


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)


# ============================================================================
# RoPE application
# ============================================================================


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ============================================================================
# Attention (full RoPE, YaRN-aware)
# ============================================================================


class SolarOpen100BAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        rope_scaling = config.get("rope_scaling")
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            self.rotary_emb = YarnRotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config["max_position_embeddings"],
                base=config["rope_theta"],
                scaling_factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling[
                    "original_max_position_embeddings"
                ],
            )
        else:
            # Standard RoPE fallback
            inv_freq = 1.0 / (
                config["rope_theta"]
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                    / self.head_dim
                )
            )
            t = torch.arange(config["max_position_embeddings"], dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
            self.rotary_emb = None

    def _get_cos_sin(self, position_ids):
        if self.rotary_emb is not None:
            return self.rotary_emb(position_ids)
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        return cos, sin

    def forward(self, hidden_states, position_ids, attention_mask=None):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_cos_sin(position_ids)
        cos = cos.unsqueeze(1)  # [B, 1, S, D]
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.full((S, S), float("-inf"), device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_output)


# ============================================================================
# MoE block (per-expert format)
# ============================================================================


class SolarOpen100BMoE(nn.Module):
    """Solar Open MoE block that loads per-expert weights (matching actual HF format)."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["moe_intermediate_size"]
        self.n_experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.n_group = config.get("n_group", 1)
        self.topk_group = config.get("topk_group", 1)
        self.norm_topk_prob = config["norm_topk_prob"]
        self.routed_scaling_factor = config["routed_scaling_factor"]

        # Router gate
        self.gate_weight = nn.Parameter(torch.zeros(self.n_experts, self.hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_experts, dtype=torch.float32), requires_grad=False
        )

        # Per-expert weights: stored as stacked tensors for efficiency
        # gate_up_proj: [E, I, H] (gate) and [E, I, H] (up) → stored as [E, 2*I, H] fused
        # down_proj: [E, H, I]
        # We store per-expert as two large tensors to match the load path
        self.experts_gate_up = nn.Parameter(
            torch.zeros(self.n_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.experts_down = nn.Parameter(
            torch.zeros(self.n_experts, self.hidden_size, self.intermediate_size)
        )

        # Shared experts
        n_shared = config.get("n_shared_experts", 0)
        shared_intermediate = self.intermediate_size * n_shared
        self.shared_gate_proj = nn.Linear(
            self.hidden_size, shared_intermediate, bias=False
        )
        self.shared_up_proj = nn.Linear(
            self.hidden_size, shared_intermediate, bias=False
        )
        self.shared_down_proj = nn.Linear(
            shared_intermediate, self.hidden_size, bias=False
        )

    def forward(self, x):
        B, S, H = x.shape
        x_flat = x.view(-1, H)
        T = x_flat.shape[0]

        # Router: sigmoid + group selection + bias correction
        router_logits = F.linear(
            x_flat.to(torch.float32), self.gate_weight.to(torch.float32)
        )
        scores = torch.sigmoid(router_logits)

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        if self.n_group <= 1:
            _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        else:
            E = self.n_experts
            group_size = E // self.n_group
            scores_grouped = scores_for_choice.view(T, self.n_group, group_size)
            group_scores = scores_grouped.max(dim=-1).values
            _, group_top_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)
            group_mask = torch.zeros(T, self.n_group, device=x.device, dtype=torch.bool)
            group_mask.scatter_(1, group_top_idx, True)
            score_mask = (
                group_mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(T, E)
            )
            masked_scores = scores_for_choice.masked_fill(~score_mask, 0.0)
            _, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1)

        topk_weights = scores.gather(1, topk_idx)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * self.routed_scaling_factor
        topk_weights = topk_weights.to(x_flat.dtype)

        # Routed expert computation
        routed_output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_ids = topk_idx[:, i]
            weights_i = topk_weights[:, i]
            for e in range(self.n_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                x_e = x_flat[mask]
                gate_up_w = self.experts_gate_up[e]  # [2*I, H]
                down_w = self.experts_down[e]  # [H, I]
                gate_w = gate_up_w[: self.intermediate_size]
                up_w = gate_up_w[self.intermediate_size :]
                gate_out = F.silu(F.linear(x_e, gate_w))
                up_out = F.linear(x_e, up_w)
                hidden = gate_out * up_out
                out_e = F.linear(hidden, down_w)
                routed_output[mask] += weights_i[mask].unsqueeze(-1) * out_e

        # Shared expert
        shared_gate = F.silu(self.shared_gate_proj(x_flat))
        shared_up = self.shared_up_proj(x_flat)
        shared_out = self.shared_down_proj(shared_gate * shared_up)

        output = routed_output + shared_out
        return output.view(B, S, H)


# ============================================================================
# Decoder layer
# ============================================================================


class SolarOpen100BDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SolarOpen100BAttention(config)
        self.mlp = SolarOpen100BMoE(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )

    def forward(self, hidden_states, position_ids, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Full reference model
# ============================================================================


class SolarOpen100BReferenceModel(nn.Module):
    """
    Pure PyTorch CPU reference for Solar Open 100B architecture.
    Loads per-expert weights from safetensors checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [
                SolarOpen100BDecoderLayer(config)
                for _ in range(config["num_hidden_layers"])
            ]
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids):
        B, S = input_ids.shape
        position_ids = (
            torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load from safetensors with per-expert weight format."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        print(
            f"  Config: hidden_size={config['hidden_size']}, "
            f"n_routed_experts={config['n_routed_experts']}, "
            f"rope_scaling={config.get('rope_scaling')}"
        )

        model = cls(config)

        # Support both single-file and sharded safetensors (e.g. upstage/Solar-Open-100B has 42 shards)
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        safetensor_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(index_path):
            print(f"  Found sharded safetensors index: {index_path}")
            with open(index_path) as _f:
                _index = json.load(_f)
            shard_files = sorted(set(_index["weight_map"].values()))
            print(f"  Loading {len(shard_files)} shards...")
            state_dict = {}
            for i, shard_file in enumerate(shard_files, 1):
                print(f"  [{i}/{len(shard_files)}] {shard_file}", flush=True)
                shard_dict = safetensors_load(os.path.join(model_path, shard_file))
                state_dict.update(shard_dict)
        elif os.path.exists(safetensor_path):
            print(f"  Loading safetensors from {safetensor_path}...")
            state_dict = safetensors_load(safetensor_path)
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.safetensors.index.json found in {model_path}"
            )

        n_experts = config["n_routed_experts"]
        intermediate_size = config["moe_intermediate_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_hidden_layers"]

        new_state_dict = {}

        for k, v in state_dict.items():
            # Strip "model." prefix
            if k.startswith("model."):
                k_strip = k[len("model.") :]
            else:
                k_strip = k

            # Per-expert gate/up/down weights: fuse into stacked tensors
            # We collect them below
            if k_strip.startswith("lm_head."):
                new_state_dict[k_strip] = v
            elif ".mlp.experts." in k_strip:
                pass  # handled in per-layer loop below
            elif ".mlp.gate.weight" in k_strip:
                new_k = k_strip.replace(".mlp.gate.weight", ".mlp.gate_weight")
                new_state_dict[new_k] = v
            elif ".mlp.gate.e_score_correction_bias" in k_strip:
                new_k = k_strip.replace(
                    ".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"
                )
                new_state_dict[new_k] = v
            elif ".mlp.shared_experts.gate_proj.weight" in k_strip:
                new_k = k_strip.replace(
                    ".mlp.shared_experts.gate_proj.weight",
                    ".mlp.shared_gate_proj.weight",
                )
                new_state_dict[new_k] = v
            elif ".mlp.shared_experts.up_proj.weight" in k_strip:
                new_k = k_strip.replace(
                    ".mlp.shared_experts.up_proj.weight", ".mlp.shared_up_proj.weight"
                )
                new_state_dict[new_k] = v
            elif ".mlp.shared_experts.down_proj.weight" in k_strip:
                new_k = k_strip.replace(
                    ".mlp.shared_experts.down_proj.weight",
                    ".mlp.shared_down_proj.weight",
                )
                new_state_dict[new_k] = v
            else:
                new_state_dict[k_strip] = v

        # Fuse per-expert weights into stacked tensors per layer
        print(
            f"  Fusing per-expert weights for {num_layers} layers x {n_experts} experts..."
        )
        for l in range(num_layers):
            # Collect all experts' gate/up/down
            gate_list = []
            up_list = []
            down_list = []
            for e in range(n_experts):
                g_key = f"layers.{l}.mlp.experts.{e}.gate_proj.weight"
                u_key = f"layers.{l}.mlp.experts.{e}.up_proj.weight"
                d_key = f"layers.{l}.mlp.experts.{e}.down_proj.weight"
                # These are in state_dict (with "model." stripped already handled above)
                # But we kept them in state_dict (raw), so look in the original
                raw_g = state_dict.get(f"model.{g_key}", state_dict.get(g_key))
                raw_u = state_dict.get(f"model.{u_key}", state_dict.get(u_key))
                raw_d = state_dict.get(f"model.{d_key}", state_dict.get(d_key))
                if raw_g is None:
                    raise KeyError(f"Missing key: model.{g_key} in checkpoint")
                gate_list.append(raw_g)  # [I, H]
                up_list.append(raw_u)  # [I, H]
                down_list.append(raw_d)  # [H, I]

            # Stack: gate_up = [E, 2*I, H], down = [E, H, I]
            gate_stacked = torch.stack(gate_list, dim=0)  # [E, I, H]
            up_stacked = torch.stack(up_list, dim=0)  # [E, I, H]
            down_stacked = torch.stack(down_list, dim=0)  # [E, H, I]

            gate_up_stacked = torch.cat(
                [gate_stacked, up_stacked], dim=1
            )  # [E, 2*I, H]

            new_state_dict[f"layers.{l}.mlp.experts_gate_up"] = gate_up_stacked
            new_state_dict[f"layers.{l}.mlp.experts_down"] = down_stacked

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  WARNING: Missing keys: {missing[:5]}")
        if unexpected:
            print(f"  WARNING: Unexpected keys: {unexpected[:5]}")

        return model

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 10
    ) -> torch.Tensor:
        """Greedy generation."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ============================================================================
# Neuron model generation
# ============================================================================


def generate_with_neuron(
    model_path: str, traced_model_path: str, input_ids: torch.Tensor
):
    """Run generation with the Neuron-compiled Solar Open 100B model."""
    from neuronx_distributed_inference.models.solar_open.modeling_solar_open import (
        NeuronSolarOpenForCausalLM,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    from transformers import GenerationConfig

    model = NeuronSolarOpenForCausalLM(traced_model_path)
    model.load(traced_model_path)

    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        generation_config = GenerationConfig(do_sample=False, top_k=1)

    generation_model = HuggingFaceGenerationAdapter(model)
    attention_mask = torch.ones_like(input_ids)
    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    return outputs


# ============================================================================
# Slack notification
# ============================================================================


def send_slack(webhook_url: str, message: str):
    import urllib.request
    import json as _json

    payload = _json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Slack notification failed: {e}")
        return False


# ============================================================================
# Main test
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Solar Open 100B accuracy test")
    parser.add_argument(
        "--model-path",
        default="/home/ubuntu/model_hf/Solar-Open-100B",
        help="Path to upstage/Solar-Open-100B HuggingFace checkpoint",
    )
    parser.add_argument(
        "--traced-model-path",
        default="solar_open_100b_traced",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model before testing"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--slack-webhook",
        default="",
        help="Slack webhook URL for notifications (optional)",
    )
    args = parser.parse_args()

    def notify(msg):
        print(msg)
        if args.slack_webhook:
            send_slack(args.slack_webhook, f"[Solar Open 100B Accuracy Test] {msg}")

    notify("🚀 Starting Solar Open 100B accuracy test (upstage/Solar-Open-100B)...")
    notify(
        f"  Architecture: hidden_size=4096, n_layers=48, n_experts=128, topk=8, YaRN RoPE"
    )
    notify(
        f"  Sharding: tp_degree=4, moe_tp_degree=4, moe_ep_degree=1 (EP disabled for library compatibility)"
    )

    # ---- CPU Reference ----
    print("\n" + "=" * 60)
    print("Loading CPU reference model (100B architecture, 2 layers)...")
    print("=" * 60)
    try:
        ref_model = SolarOpen100BReferenceModel.from_pretrained(args.model_path)
        ref_model.eval()
        notify("✅ CPU reference model loaded.")
    except Exception as e:
        notify(f"❌ FAILED to load CPU reference model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ---- Compile if requested ----
    if args.compile:
        print("\n" + "=" * 60)
        print("Compiling Neuron model...")
        print("=" * 60)
        notify("⚙️  Compiling Solar Open 100B Neuron model (tp=4, moe_tp=4, ep=1)...")
        try:
            import importlib.util
            import importlib.machinery

            # Import generation demo (path relative to this test file)
            _demo_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "examples",
                "generation_solar_open_100b_demo.py",
            )
            loader = importlib.machinery.SourceFileLoader(
                "generation_solar_open_100b_demo",
                _demo_path,
            )
            spec = importlib.util.spec_from_loader(
                "generation_solar_open_100b_demo", loader
            )
            demo_mod = importlib.util.module_from_spec(spec)
            loader.exec_module(demo_mod)
            demo_mod.generate(
                args.model_path, args.traced_model_path, skip_compile=False
            )
            notify("✅ Compilation succeeded.")
        except Exception as e:
            notify(f"❌ Compilation FAILED: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # ---- Test inputs ----
    torch.manual_seed(42)
    input_ids = torch.tensor([[1, 100, 200, 300, 400]], dtype=torch.long)
    max_new_tokens = args.max_new_tokens

    # ---- CPU Reference generation ----
    print("\n" + "=" * 60)
    print("Running CPU reference generation...")
    print("=" * 60)
    notify("📊 Running CPU reference generation...")
    with torch.no_grad():
        ref_output = ref_model.generate(
            input_ids.clone(), max_new_tokens=max_new_tokens
        )
    ref_new_tokens = ref_output[:, input_ids.shape[1] :]
    print(f"Reference input_ids:  {input_ids.tolist()}")
    print(f"Reference new tokens: {ref_new_tokens.tolist()}")
    notify(f"  CPU ref new tokens: {ref_new_tokens.tolist()}")

    # ---- Neuron model generation ----
    print("\n" + "=" * 60)
    print("Running Neuron model generation...")
    print("=" * 60)
    notify("⚡ Running Neuron model generation...")
    try:
        neuron_output = generate_with_neuron(
            args.model_path, args.traced_model_path, input_ids.clone()
        )
        neuron_new_tokens = neuron_output[:, input_ids.shape[1] :]
        print(f"Neuron input_ids:    {input_ids.tolist()}")
        print(f"Neuron new tokens:   {neuron_new_tokens.tolist()}")
        notify(f"  Neuron new tokens:  {neuron_new_tokens.tolist()}")
    except Exception as e:
        notify(f"❌ Neuron generation FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("Comparing outputs...")
    print("=" * 60)

    min_new = min(ref_new_tokens.shape[1], neuron_new_tokens.shape[1])
    ref_cmp = ref_new_tokens[:, :min_new]
    neuron_cmp = neuron_new_tokens[:, :min_new]

    match = torch.all(ref_cmp == neuron_cmp).item()

    if match:
        msg = (
            f"✅ PASSED: Neuron output matches CPU reference!\n"
            f"  Generated {min_new} tokens, all match.\n"
            f"  Reference: {ref_cmp.tolist()}\n"
            f"  Neuron:    {neuron_cmp.tolist()}"
        )
        notify(msg)
        print("\n" + "=" * 60)
        print("TEST PASSED ✅")
        print("=" * 60)
        sys.exit(0)
    else:
        mismatches = (ref_cmp != neuron_cmp).nonzero().tolist()
        msg = (
            f"❌ FAILED: Neuron output does NOT match CPU reference!\n"
            f"  Mismatches at positions: {mismatches}\n"
            f"  Reference: {ref_cmp.tolist()}\n"
            f"  Neuron:    {neuron_cmp.tolist()}"
        )
        notify(msg)
        print("\n" + "=" * 60)
        print("TEST FAILED ❌")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
