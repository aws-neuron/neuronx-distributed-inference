"""Integration test utilities for Solar Open MoE.

Solar Open is NOT in transformers — this module provides:
- create_tiny_solar_open_model(): writes a minimal safetensors checkpoint
- get_neuron_config(): returns MoENeuronConfig for integration tests
- SolarOpenReferenceModel: pure PyTorch CPU reference for logit accuracy checks
- check_logit_accuracy(): runs CPU ref + Neuron model and compares logits
"""

import json
import os
import sys
import math
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add contrib src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from solar_open.modeling_solar_open import (
    NeuronSolarOpenForCausalLM,
    SolarOpenInferenceConfig,
    load_solar_open_config,
)

from neuronx_distributed_inference.models.config import MoENeuronConfig


# ---------------------------------------------------------------------------
# Neuron config for integration tests
# ---------------------------------------------------------------------------


def get_neuron_config() -> MoENeuronConfig:
    """Return MoENeuronConfig for Solar Open integration tests.

    Uses tp_degree=2, moe_tp_degree=2, moe_ep_degree=1 — compatible with
    trn2.3xlarge (2 NeuronCores) and smaller test instances.
    """
    return MoENeuronConfig(
        tp_degree=2,
        moe_tp_degree=2,
        moe_ep_degree=1,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=128,
        max_context_length=112,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
        flash_decoding_enabled=False,
        output_logits=True,
        enable_bucketing=False,
        sequence_parallel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )


# ---------------------------------------------------------------------------
# Tiny random model factory
# ---------------------------------------------------------------------------


def create_tiny_solar_open_model(model_dir: str, config_json_path: str) -> None:
    """Create a tiny random-weight Solar Open checkpoint for testing.

    Writes model.safetensors + config.json to model_dir. Uses Format B
    (pre-fused 3D tensors) for the expert weights:
      mlp.experts.gate_up_proj: [E, 2*I, H]
      mlp.experts.down_proj:    [E, H, I]

    This format is auto-detected by convert_solar_open_hf_to_neuron_state_dict.

    Args:
        model_dir: Directory to write the checkpoint to.
        config_json_path: Path to the config JSON (e.g. config_solar_open_2layers.json).
    """
    from safetensors.torch import save_file

    os.makedirs(model_dir, exist_ok=True)

    # Load config
    with open(config_json_path) as f:
        cfg = json.load(f)

    H = cfg["hidden_size"]
    N_LAYERS = cfg["num_hidden_layers"]
    N_HEADS = cfg["num_attention_heads"]
    N_KV_HEADS = cfg["num_key_value_heads"]
    HEAD_DIM = cfg["head_dim"]
    I = cfg["moe_intermediate_size"]
    E = cfg["n_routed_experts"]
    N_SHARED = cfg["n_shared_experts"]
    VOCAB = cfg["vocab_size"]

    torch.manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.02

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.bfloat16)

    state_dict = {}

    # Embedding
    state_dict["model.embed_tokens.weight"] = rand(VOCAB, H)

    for l in range(N_LAYERS):
        # Layer norms
        state_dict[f"model.layers.{l}.input_layernorm.weight"] = ones(H)
        state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = ones(H)

        # Attention projections — no bias (attention_bias=False)
        q_dim = N_HEADS * HEAD_DIM
        kv_dim = N_KV_HEADS * HEAD_DIM
        state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = rand(q_dim, H)
        state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = rand(kv_dim, H)
        state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = rand(kv_dim, H)
        state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = rand(H, q_dim)

        # Router: gate weight + e_score_correction_bias
        state_dict[f"model.layers.{l}.mlp.gate.weight"] = rand(E, H)
        state_dict[f"model.layers.{l}.mlp.gate.e_score_correction_bias"] = torch.zeros(
            E, dtype=torch.float32
        )

        # Routed experts: Format B (pre-fused 3D tensors, no .weight suffix)
        # gate_up_proj: [E, 2*I, H]
        state_dict[f"model.layers.{l}.mlp.experts.gate_up_proj"] = rand(E, 2 * I, H)
        # down_proj: [E, H, I]
        state_dict[f"model.layers.{l}.mlp.experts.down_proj"] = rand(E, H, I)

        # Shared experts (always-on dense MLP, uses moe_intermediate_size)
        shared_I = I * N_SHARED
        state_dict[f"model.layers.{l}.mlp.shared_experts.gate_proj.weight"] = rand(
            shared_I, H
        )
        state_dict[f"model.layers.{l}.mlp.shared_experts.up_proj.weight"] = rand(
            shared_I, H
        )
        state_dict[f"model.layers.{l}.mlp.shared_experts.down_proj.weight"] = rand(
            H, shared_I
        )

    # Final norm
    state_dict["model.norm.weight"] = ones(H)

    # LM head (no "model." prefix in HF Solar Open format)
    state_dict["lm_head.weight"] = rand(VOCAB, H)

    # Save safetensors
    from safetensors.torch import save_file

    save_file(state_dict, os.path.join(model_dir, "model.safetensors"))

    # Copy config.json
    with open(config_json_path) as f:
        config_data = json.load(f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)


# ---------------------------------------------------------------------------
# Pure PyTorch CPU reference model (copied from test_solar_open_accuracy.py)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Minimal RMSNorm for CPU reference."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(q, k, cos, sin):
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, position_ids):
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class _SolarOpenAttention(nn.Module):
    """CPU reference attention (no bias, full RoPE)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.hidden_size = cfg["hidden_size"]
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

        self.rotary_emb = _RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=cfg["max_position_embeddings"],
            base=cfg["rope_theta"],
        )

    def forward(self, hidden_states, position_ids, attention_mask=None):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q, k = _apply_rotary_emb(q, k, cos, sin)

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


class _SolarOpenMoE(nn.Module):
    """CPU reference MoE block: routed experts + shared experts."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["moe_intermediate_size"]
        self.n_experts = cfg["n_routed_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.n_group = cfg.get("n_group", 1)
        self.topk_group = cfg.get("topk_group", 1)
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]

        # Router
        self.gate_weight = nn.Parameter(torch.zeros(self.n_experts, self.hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_experts, dtype=torch.float32), requires_grad=False
        )

        # Pre-fused 3D routed expert weights (Format B)
        self.experts_gate_up = nn.Parameter(
            torch.zeros(self.n_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.experts_down = nn.Parameter(
            torch.zeros(self.n_experts, self.hidden_size, self.intermediate_size)
        )

        # Shared experts
        n_shared = cfg.get("n_shared_experts", 0)
        shared_I = self.intermediate_size * n_shared
        self.shared_gate_proj = nn.Linear(self.hidden_size, shared_I, bias=False)
        self.shared_up_proj = nn.Linear(self.hidden_size, shared_I, bias=False)
        self.shared_down_proj = nn.Linear(shared_I, self.hidden_size, bias=False)

    def forward(self, x):
        B, S, H = x.shape
        x_flat = x.view(-1, H)
        T = x_flat.shape[0]

        # Router: sigmoid + bias correction + group selection
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

        # Weights from original sigmoid scores (not bias-corrected)
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

        # Shared expert computation
        shared_gate = F.silu(self.shared_gate_proj(x_flat))
        shared_up = self.shared_up_proj(x_flat)
        shared_out = self.shared_down_proj(shared_gate * shared_up)

        return (routed_output + shared_out).view(B, S, H)


class _SolarOpenDecoderLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.self_attn = _SolarOpenAttention(cfg)
        self.mlp = _SolarOpenMoE(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

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


class SolarOpenReferenceModel(nn.Module):
    """Pure PyTorch CPU reference for Solar Open MoE.

    Loads weights from safetensors checkpoint for logit accuracy comparison
    against the NeuronX compiled model.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(
            [_SolarOpenDecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        )
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        B, S = input_ids.shape
        position_ids = (
            torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "SolarOpenReferenceModel":
        """Load from safetensors checkpoint at model_dir."""
        from safetensors.torch import load_file as safetensors_load

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        model = cls(cfg)

        safetensor_path = os.path.join(model_dir, "model.safetensors")
        state_dict = safetensors_load(safetensor_path)

        # Map HF keys → reference model keys
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model.") :]

            if ".mlp.experts.gate_up_proj" in k:
                # [E, 2*I, H] → store as experts_gate_up
                new_sd[
                    k.replace(".mlp.experts.gate_up_proj", ".mlp.experts_gate_up")
                ] = v
            elif ".mlp.experts.down_proj" in k:
                # [E, H, I] → store as experts_down
                new_sd[k.replace(".mlp.experts.down_proj", ".mlp.experts_down")] = v
            elif ".mlp.gate.weight" in k:
                new_sd[k.replace(".mlp.gate.weight", ".mlp.gate_weight")] = v
            elif ".mlp.gate.e_score_correction_bias" in k:
                new_sd[
                    k.replace(
                        ".mlp.gate.e_score_correction_bias",
                        ".mlp.e_score_correction_bias",
                    )
                ] = v
            elif ".mlp.shared_experts.gate_proj.weight" in k:
                new_sd[
                    k.replace(
                        ".mlp.shared_experts.gate_proj.weight",
                        ".mlp.shared_gate_proj.weight",
                    )
                ] = v
            elif ".mlp.shared_experts.up_proj.weight" in k:
                new_sd[
                    k.replace(
                        ".mlp.shared_experts.up_proj.weight",
                        ".mlp.shared_up_proj.weight",
                    )
                ] = v
            elif ".mlp.shared_experts.down_proj.weight" in k:
                new_sd[
                    k.replace(
                        ".mlp.shared_experts.down_proj.weight",
                        ".mlp.shared_down_proj.weight",
                    )
                ] = v
            elif k.startswith("lm_head."):
                new_sd[k] = v
            else:
                new_sd[k] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  [ref] Missing keys: {missing[:3]}")
        if unexpected:
            print(f"  [ref] Unexpected keys: {unexpected[:3]}")
        return model


# ---------------------------------------------------------------------------
# Logit accuracy check
# ---------------------------------------------------------------------------


def check_logit_accuracy(
    model_dir: str,
    traced_dir: str,
    neuron_config: MoENeuronConfig,
    tol: float = 0.05,
) -> bool:
    """Compare logits from CPU reference vs compiled Neuron model.

    Args:
        model_dir: Path to the tiny random model checkpoint.
        traced_dir: Path to the compiled Neuron model.
        neuron_config: MoENeuronConfig used for compilation.
        tol: Maximum allowed mean absolute error on logits.

    Returns:
        True if logits match within tolerance, False otherwise.
    """
    import shutil

    torch.manual_seed(0)

    # --- CPU Reference ---
    ref_model = SolarOpenReferenceModel.from_pretrained(model_dir)
    ref_model.eval()

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    vocab_size = cfg["vocab_size"]
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, 10), dtype=torch.long)

    with torch.no_grad():
        ref_logits = ref_model(input_ids).float()  # [1, seq, vocab]

    # --- Neuron model ---
    # Copy model weights to traced_dir so load() can find safetensors
    src = os.path.join(model_dir, "model.safetensors")
    dst = os.path.join(traced_dir, "model.safetensors")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

    neuron_model = NeuronSolarOpenForCausalLM(traced_dir)
    neuron_model.load(traced_dir)

    with torch.no_grad():
        # NeuronSolarOpenForCausalLM forward: context encoding on full input
        neuron_logits = neuron_model(input_ids).logits.float()  # [1, seq, vocab]

    # Compare last-token logits (most stable)
    ref_last = ref_logits[:, -1, :]  # [1, vocab]
    neuron_last = neuron_logits[:, -1, :]  # [1, vocab]

    mae = (ref_last - neuron_last).abs().mean().item()
    print(f"  Logit MAE (last token): {mae:.6f} (tol={tol})")
    return mae < tol
