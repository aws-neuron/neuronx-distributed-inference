"""
Create a 2-layer random Solar Open model with the exact 100B architecture config.

This model uses the per-expert weight format that matches the actual upstage/Solar-Open-100B
HuggingFace checkpoint, so the weight conversion pipeline in modeling_solar_open.py can be
tested end-to-end before loading the real 205 GB model.

Architecture reference: upstage/Solar-Open-100B (config.json)
- hidden_size: 4096
- num_hidden_layers: 48 (reduced to 2 here for fast compilation)
- num_attention_heads: 64
- head_dim: 128
- num_key_value_heads: 8
- vocab_size: 196608
- n_routed_experts: 128
- n_shared_experts: 1
- num_experts_per_tok: 8
- moe_intermediate_size: 1280
- rope_scaling: {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536}

Expert weight format (per-expert, same as actual HF checkpoint):
  model.layers.{l}.mlp.experts.{e}.gate_proj.weight  [moe_intermediate_size, hidden_size]
  model.layers.{l}.mlp.experts.{e}.up_proj.weight    [moe_intermediate_size, hidden_size]
  model.layers.{l}.mlp.experts.{e}.down_proj.weight  [hidden_size, moe_intermediate_size]
  model.layers.{l}.mlp.gate.weight                   [n_routed_experts, hidden_size]
  model.layers.{l}.mlp.gate.e_score_correction_bias  [n_routed_experts]
  model.layers.{l}.mlp.shared_experts.gate_proj.weight  [shared_intermediate, hidden_size]
  model.layers.{l}.mlp.shared_experts.up_proj.weight    [shared_intermediate, hidden_size]
  model.layers.{l}.mlp.shared_experts.down_proj.weight  [hidden_size, shared_intermediate]

Usage:
    python create_solar_open_100b_random.py
"""

import json
import os
import torch
from safetensors.torch import save_file

MODEL_PATH = "solar_open_100b_random"
os.makedirs(MODEL_PATH, exist_ok=True)

torch.manual_seed(42)

# ---- 100B architecture dimensions (2-layer for fast testing) ----
HIDDEN_SIZE = 4096
NUM_LAYERS = 2  # Reduced from 48 for fast compilation
NUM_HEADS = 64  # num_attention_heads
NUM_KV_HEADS = 8  # num_key_value_heads
HEAD_DIM = 128  # head_dim
MOE_INTERMEDIATE = 1280  # moe_intermediate_size
N_EXPERTS = 128  # n_routed_experts
N_SHARED = 1  # n_shared_experts
TOPK = 8  # num_experts_per_tok
VOCAB_SIZE = 196608  # same as tiny model
N_GROUP = 1  # not in 100B config, default
TOPK_GROUP = 1  # not in 100B config, default
INTERMEDIATE_SIZE = 10240  # dense intermediate (unused; all layers are MoE)


def rand(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16) * 0.02


def ones(*shape):
    return torch.ones(*shape, dtype=torch.bfloat16)


state_dict = {}

# Embedding
state_dict["model.embed_tokens.weight"] = rand(VOCAB_SIZE, HIDDEN_SIZE)

SHARED_INTERMEDIATE = MOE_INTERMEDIATE * N_SHARED  # 1280 * 1 = 1280

for l in range(NUM_LAYERS):
    # Layer norms
    state_dict[f"model.layers.{l}.input_layernorm.weight"] = ones(HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = ones(HIDDEN_SIZE)

    # Attention projections (no bias)
    q_dim = NUM_HEADS * HEAD_DIM  # 64 * 128 = 8192
    kv_dim = NUM_KV_HEADS * HEAD_DIM  # 8 * 128 = 1024
    state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = rand(q_dim, HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = rand(kv_dim, HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = rand(kv_dim, HIDDEN_SIZE)
    state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = rand(HIDDEN_SIZE, q_dim)

    # --- MoE block (per-expert format, matching actual HF checkpoint) ---

    # Router gate
    state_dict[f"model.layers.{l}.mlp.gate.weight"] = rand(N_EXPERTS, HIDDEN_SIZE)
    # e_score_correction_bias is float32
    state_dict[f"model.layers.{l}.mlp.gate.e_score_correction_bias"] = torch.zeros(
        N_EXPERTS, dtype=torch.float32
    )

    # Routed experts: per-expert separate weights (matches upstage/Solar-Open-100B HF format)
    print(f"  Layer {l}: creating {N_EXPERTS} routed experts...", flush=True)
    for e in range(N_EXPERTS):
        state_dict[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = rand(
            MOE_INTERMEDIATE, HIDDEN_SIZE
        )
        state_dict[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = rand(
            MOE_INTERMEDIATE, HIDDEN_SIZE
        )
        state_dict[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = rand(
            HIDDEN_SIZE, MOE_INTERMEDIATE
        )

    # Shared expert
    state_dict[f"model.layers.{l}.mlp.shared_experts.gate_proj.weight"] = rand(
        SHARED_INTERMEDIATE, HIDDEN_SIZE
    )
    state_dict[f"model.layers.{l}.mlp.shared_experts.up_proj.weight"] = rand(
        SHARED_INTERMEDIATE, HIDDEN_SIZE
    )
    state_dict[f"model.layers.{l}.mlp.shared_experts.down_proj.weight"] = rand(
        HIDDEN_SIZE, SHARED_INTERMEDIATE
    )

# Final norm
state_dict["model.norm.weight"] = ones(HIDDEN_SIZE)

# LM head
state_dict["lm_head.weight"] = rand(VOCAB_SIZE, HIDDEN_SIZE)

# Print state dict summary (skip per-expert keys to avoid wall of text)
print("\nState dict summary (non-expert keys):")
expert_count = 0
total_params = 0
for k, v in sorted(state_dict.items()):
    total_params += v.numel()
    if ".mlp.experts." in k and ".gate_proj." in k and ".0." not in k:
        expert_count += 1
        continue  # only print expert 0 as sample
    print(f"  {k}: {list(v.shape)} {v.dtype}")

print(f"\nTotal parameters: {total_params:,}")
print(f"Total keys: {len(state_dict)}")

# Save as safetensors
print(f"\nSaving to {MODEL_PATH}/model.safetensors ...")
save_file(state_dict, os.path.join(MODEL_PATH, "model.safetensors"))
print("Saved model.safetensors")

# Save config.json matching actual upstage/Solar-Open-100B config
config = {
    "model_type": "solar_open",
    "architectures": ["SolarOpenForCausalLM"],
    "hidden_size": HIDDEN_SIZE,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": NUM_HEADS,
    "head_dim": HEAD_DIM,
    "num_key_value_heads": NUM_KV_HEADS,
    "vocab_size": VOCAB_SIZE,
    "intermediate_size": INTERMEDIATE_SIZE,  # dense MLP size (unused; all layers MoE)
    "moe_intermediate_size": MOE_INTERMEDIATE,
    "n_routed_experts": N_EXPERTS,
    "n_shared_experts": N_SHARED,
    "num_experts_per_tok": TOPK,
    "n_group": N_GROUP,
    "topk_group": TOPK_GROUP,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 0,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 65536,
    },
    "max_position_embeddings": 131072,
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
print("\nDone! Solar Open 100B random model (2 layers, per-expert format) created.")
