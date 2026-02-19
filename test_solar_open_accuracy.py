"""
Accuracy test for Solar Open MoE NXD inference vs CPU reference.

Since solar_open is NOT in transformers, this script implements a pure PyTorch
CPU reference model (SolarOpenReferenceModel) that loads the same safetensors
weights and runs a forward pass.

The test compares generated token IDs from the Neuron model vs the CPU reference.
With random weights and greedy decoding (top_k=1), they should be identical.

Usage:
    # First compile if needed:
    python examples/generation_solar_open_demo.py

    # Run accuracy test (assumes model is already compiled):
    python test_solar_open_accuracy.py

    # Compile then test:
    python test_solar_open_accuracy.py --compile
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
# Pure PyTorch CPU Reference Model
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


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, position_ids):
        cos = self.cos_cached[position_ids]  # [B, S, D]
        sin = self.sin_cached[position_ids]
        return cos, sin


class SolarOpenAttention(nn.Module):
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

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config["max_position_embeddings"],
            base=config["rope_theta"],
        )

    def forward(self, hidden_states, position_ids, attention_mask=None):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # [B, Hkv, S, D]
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # [B, Hkv, S, D]

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.unsqueeze(1)  # [B, 1, S, D]
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Repeat KV for grouped query attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

        # Causal mask
        causal_mask = torch.full((S, S), float("-inf"), device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_output)


class SolarOpenMoE(nn.Module):
    """Solar Open MoE block: routed experts + shared experts."""

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

        # Routed expert weights (pre-fused 3D tensors, as in HF solar_open)
        # gate_up_proj: [E, 2*I, H]
        self.experts_gate_up = nn.Parameter(
            torch.zeros(self.n_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        # down_proj: [E, H, I]
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
        x_flat = x.view(-1, H)  # [B*S, H]
        T = x_flat.shape[0]

        # Router: sigmoid + group selection + bias correction
        router_logits = F.linear(
            x_flat.to(torch.float32), self.gate_weight.to(torch.float32)
        )
        scores = torch.sigmoid(router_logits)  # [T, E]

        # e_score_correction_bias for routing decision
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Group-based selection (simplified for n_group=1 → standard topk)
        if self.n_group <= 1:
            _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        else:
            E = self.n_experts
            group_size = E // self.n_group
            scores_grouped = scores_for_choice.view(T, self.n_group, group_size)
            group_scores = scores_grouped.max(dim=-1).values  # [T, n_group]
            _, group_top_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1
            )  # [T, topk_group]
            group_mask = torch.zeros(T, self.n_group, device=x.device, dtype=torch.bool)
            group_mask.scatter_(1, group_top_idx, True)
            score_mask = (
                group_mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(T, E)
            )
            masked_scores = scores_for_choice.masked_fill(~score_mask, 0.0)
            _, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1)

        # Get weights from original sigmoid scores
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
            expert_ids = topk_idx[:, i]  # [T]
            weights_i = topk_weights[:, i]  # [T]

            for e in range(self.n_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                x_e = x_flat[mask]  # [n_e, H]

                # gate_up: [2*I, H], down: [H, I]
                gate_up_w = self.experts_gate_up[e]  # [2*I, H]
                down_w = self.experts_down[e]  # [H, I]

                gate_w = gate_up_w[: self.intermediate_size]  # [I, H]
                up_w = gate_up_w[self.intermediate_size :]  # [I, H]

                gate_out = F.silu(F.linear(x_e, gate_w))  # [n_e, I]
                up_out = F.linear(x_e, up_w)  # [n_e, I]
                hidden = gate_out * up_out  # [n_e, I]

                # down_w: [H, I], F.linear(x, W) = x @ W.T → [n_e, I] @ [I, H] = [n_e, H]
                out_e = F.linear(hidden, down_w)  # [n_e, H]

                routed_output[mask] += weights_i[mask].unsqueeze(-1) * out_e

        # Shared expert computation
        shared_gate = F.silu(self.shared_gate_proj(x_flat))
        shared_up = self.shared_up_proj(x_flat)
        shared_out = self.shared_down_proj(shared_gate * shared_up)

        output = routed_output + shared_out
        return output.view(B, S, H)


class SolarOpenDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SolarOpenAttention(config)
        self.mlp = SolarOpenMoE(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )

    def forward(self, hidden_states, position_ids, attention_mask=None):
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SolarOpenReferenceModel(nn.Module):
    """
    Pure PyTorch CPU reference implementation of Solar Open MoE.
    Loads weights from safetensors checkpoint for accuracy comparison.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [SolarOpenDecoderLayer(config) for _ in range(config["num_hidden_layers"])]
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
        """Load model from safetensors checkpoint."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        model = cls(config)

        # Load weights
        safetensor_path = os.path.join(model_path, "model.safetensors")
        state_dict = safetensors_load(safetensor_path)

        # Map HF state dict keys to our reference model structure
        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip "model." prefix
            if k.startswith("model."):
                k = k[len("model.") :]

            # Map layer keys
            if ".mlp.experts.gate_up_proj" in k:
                # [E, 2*I, H] → store as-is (our ref model handles the layout)
                new_k = k.replace(".mlp.experts.gate_up_proj", ".mlp.experts_gate_up")
                new_state_dict[new_k] = v
            elif ".mlp.experts.down_proj" in k:
                # [E, H, I] → store as-is
                new_k = k.replace(".mlp.experts.down_proj", ".mlp.experts_down")
                new_state_dict[new_k] = v
            elif ".mlp.gate.weight" in k:
                new_k = k.replace(".mlp.gate.weight", ".mlp.gate_weight")
                new_state_dict[new_k] = v
            elif ".mlp.gate.e_score_correction_bias" in k:
                new_k = k.replace(
                    ".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"
                )
                new_state_dict[new_k] = v
            elif ".mlp.shared_experts.gate_proj.weight" in k:
                new_k = k.replace(
                    ".mlp.shared_experts.gate_proj.weight",
                    ".mlp.shared_gate_proj.weight",
                )
                new_state_dict[new_k] = v
            elif ".mlp.shared_experts.up_proj.weight" in k:
                new_k = k.replace(
                    ".mlp.shared_experts.up_proj.weight", ".mlp.shared_up_proj.weight"
                )
                new_state_dict[new_k] = v
            elif ".mlp.shared_experts.down_proj.weight" in k:
                new_k = k.replace(
                    ".mlp.shared_experts.down_proj.weight",
                    ".mlp.shared_down_proj.weight",
                )
                new_state_dict[new_k] = v
            elif k.startswith("lm_head."):
                new_state_dict[k] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  WARNING: Missing keys in reference model: {missing[:5]}...")
        if unexpected:
            print(f"  WARNING: Unexpected keys in reference model: {unexpected[:5]}...")

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
    """Run generation with the Neuron-compiled Solar Open model."""
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
# Main test
# ============================================================================


def send_slack(webhook_url: str, message: str):
    """Send a Slack notification."""
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


def main():
    parser = argparse.ArgumentParser(description="Solar Open accuracy test")
    parser.add_argument(
        "--model-path",
        default="solar_open_tiny_random",
    )
    parser.add_argument(
        "--traced-model-path",
        default="solar_open_tiny_random_traced",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model before testing",
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
            send_slack(args.slack_webhook, f"[Solar Open Accuracy Test] {msg}")

    notify("Starting Solar Open accuracy test...")

    # ---- CPU Reference ----
    print("\n" + "=" * 60)
    print("Loading CPU reference model...")
    print("=" * 60)
    try:
        ref_model = SolarOpenReferenceModel.from_pretrained(args.model_path)
        ref_model.eval()
        print("CPU reference model loaded successfully.")
    except Exception as e:
        notify(f"❌ FAILED to load CPU reference model: {e}")
        sys.exit(1)

    # ---- Compile if requested ----
    if args.compile:
        print("\n" + "=" * 60)
        print("Compiling Neuron model...")
        print("=" * 60)
        notify("Compiling Solar Open Neuron model...")
        try:
            from examples.generation_solar_open_demo import generate as demo_generate

            demo_generate(args.model_path, args.traced_model_path, skip_compile=False)
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
    with torch.no_grad():
        ref_output = ref_model.generate(
            input_ids.clone(), max_new_tokens=max_new_tokens
        )
    ref_new_tokens = ref_output[:, input_ids.shape[1] :]
    print(f"Reference input_ids:  {input_ids.tolist()}")
    print(f"Reference new tokens: {ref_new_tokens.tolist()}")
    print(f"Reference output:     {ref_output.tolist()}")

    # ---- Neuron model generation ----
    print("\n" + "=" * 60)
    print("Running Neuron model generation...")
    print("=" * 60)
    notify("Running Neuron model generation...")
    try:
        neuron_output = generate_with_neuron(
            args.model_path, args.traced_model_path, input_ids.clone()
        )
        neuron_new_tokens = neuron_output[:, input_ids.shape[1] :]
        print(f"Neuron input_ids:    {input_ids.tolist()}")
        print(f"Neuron new tokens:   {neuron_new_tokens.tolist()}")
        print(f"Neuron output:       {neuron_output.tolist()}")
    except Exception as e:
        notify(f"❌ Neuron generation FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("Comparing outputs...")
    print("=" * 60)

    # Align lengths (neuron may generate up to max_length)
    min_new = min(ref_new_tokens.shape[1], neuron_new_tokens.shape[1])
    ref_cmp = ref_new_tokens[:, :min_new]
    neuron_cmp = neuron_new_tokens[:, :min_new]

    match = torch.all(ref_cmp == neuron_cmp).item()

    if match:
        msg = (
            f"✅ PASSED: Neuron output matches CPU reference!\n"
            f"  Generated {min_new} tokens, all match.\n"
            f"  Reference tokens: {ref_cmp.tolist()}\n"
            f"  Neuron tokens:    {neuron_cmp.tolist()}"
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
            f"  Reference tokens: {ref_cmp.tolist()}\n"
            f"  Neuron tokens:    {neuron_cmp.tolist()}"
        )
        notify(msg)
        print("\n" + "=" * 60)
        print("TEST FAILED ❌")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
