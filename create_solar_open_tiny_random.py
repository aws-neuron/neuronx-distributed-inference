"""
Create a tiny random solar_open model for testing neuronx-distributed-inference.

Solar Open (SolarOpenForCausalLM) state dict structure:
- Expert weights are PRE-FUSED as 3D tensors (unlike GLM-4.5):
    mlp.experts.gate_up_proj [n_experts, 2*moe_intermediate_size, hidden_size]
    mlp.experts.down_proj    [n_experts, hidden_size, moe_intermediate_size]
- No per-expert separate weights
- No attention bias (attention_bias=False)
- No QK norm (use_qk_norm=False)
- No dense layers (first_k_dense_replace=0 → all layers are MoE)
- Full RoPE (partial_rotary_factor=1.0)

Usage:
    python create_solar_open_tiny_random.py
"""

import json
import os
import torch
from safetensors.torch import save_file

MODEL_PATH = "solar_open_tiny_random"
os.makedirs(MODEL_PATH, exist_ok=True)

torch.manual_seed(42)

# ---- Tiny model dimensions ----
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_HEADS = 4  # num_attention_heads
NUM_KV_HEADS = 2  # num_key_value_heads
HEAD_DIM = 8  # head_dim → q_proj output = 4*8=32 = hidden_size
MOE_INTERMEDIATE = 8  # moe_intermediate_size
N_EXPERTS = 8  # n_routed_experts
N_SHARED = 1  # n_shared_experts
TOPK = 4  # num_experts_per_tok
VOCAB_SIZE = 196608  # keep original vocab_size


def rand(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16) * 0.02


def ones(*shape):
    return torch.ones(*shape, dtype=torch.bfloat16)


state_dict = {}

# Embedding
state_dict["model.embed_tokens.weight"] = rand(VOCAB_SIZE, HIDDEN_SIZE)

for l in range(NUM_LAYERS):
    # Layer norms
    state_dict[f"model.layers.{l}.input_layernorm.weight"] = ones(HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = ones(HIDDEN_SIZE)

    # Attention projections (no bias in solar_open)
    q_dim = NUM_HEADS * HEAD_DIM  # 4 * 8 = 32
    kv_dim = NUM_KV_HEADS * HEAD_DIM  # 2 * 8 = 16
    state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = rand(q_dim, HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = rand(kv_dim, HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = rand(kv_dim, HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = rand(HIDDEN_SIZE, q_dim)

    # --- MoE block (ALL layers, first_k_dense_replace=0) ---

    # Router gate (sigmoid-based, no softmax)
    state_dict[f"model.layers.{l}.mlp.gate.weight"] = rand(N_EXPERTS, HIDDEN_SIZE)
    # e_score_correction_bias is a buffer (float32)
    state_dict[f"model.layers.{l}.mlp.gate.e_score_correction_bias"] = torch.zeros(
        N_EXPERTS, dtype=torch.float32
    )

    # Routed experts: PRE-FUSED 3D tensors (key HF solar_open difference from GLM-4.5)
    # gate_up_proj: [n_experts, 2*moe_intermediate, hidden]  (nn.Parameter, no .weight suffix)
    state_dict[f"model.layers.{l}.mlp.experts.gate_up_proj"] = rand(
        N_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN_SIZE
    )
    # down_proj: [n_experts, hidden, moe_intermediate]  (nn.Parameter, no .weight suffix)
    state_dict[f"model.layers.{l}.mlp.experts.down_proj"] = rand(
        N_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE
    )

    # Shared expert (always-on dense MLP alongside routed experts)
    # Uses moe_intermediate_size * n_shared_experts
    shared_intermediate = MOE_INTERMEDIATE * N_SHARED
    state_dict[f"model.layers.{l}.mlp.shared_experts.gate_proj.weight"] = rand(
        shared_intermediate, HIDDEN_SIZE
    )
    state_dict[f"model.layers.{l}.mlp.shared_experts.up_proj.weight"] = rand(
        shared_intermediate, HIDDEN_SIZE
    )
    state_dict[f"model.layers.{l}.mlp.shared_experts.down_proj.weight"] = rand(
        HIDDEN_SIZE, shared_intermediate
    )

# Final norm
state_dict["model.norm.weight"] = ones(HIDDEN_SIZE)

# LM head (note: no "model." prefix - it's a direct attribute of ForCausalLM)
state_dict["lm_head.weight"] = rand(VOCAB_SIZE, HIDDEN_SIZE)

# Print state dict summary
print("State dict keys and shapes:")
for k, v in sorted(state_dict.items()):
    print(f"  {k}: {list(v.shape)} {v.dtype}")
print(f"\nTotal parameters: {sum(v.numel() for v in state_dict.values()):,}")

# Save as safetensors (loaded directly by neuronx load_state_dict)
save_file(state_dict, os.path.join(MODEL_PATH, "model.safetensors"))
print(f"\nSaved to {MODEL_PATH}/model.safetensors")

# Save config.json
config = {
    "model_type": "solar_open",
    "architectures": ["SolarOpenForCausalLM"],
    "hidden_size": HIDDEN_SIZE,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": NUM_HEADS,
    "num_key_value_heads": NUM_KV_HEADS,
    "head_dim": HEAD_DIM,
    "intermediate_size": 64,  # kept for backward compat; overridden in InferenceConfig
    "moe_intermediate_size": MOE_INTERMEDIATE,
    "n_routed_experts": N_EXPERTS,
    "n_shared_experts": N_SHARED,
    "num_experts_per_tok": TOPK,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": 131072,
    "first_k_dense_replace": 0,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "rope_scaling": None,  # plain RoPE for tiny test (no YaRN params)
    "partial_rotary_factor": 1.0,
    "attention_bias": False,
    "use_qk_norm": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 2,
    "transformers_version": "4.57.1",
}
with open(os.path.join(MODEL_PATH, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved config.json")
print("\nDone! Tiny solar_open random model created.")
