"""
Script to create a tiny random GLM-4.5 MoE model for testing.
"""

import os
import json
import torch
from transformers import AutoTokenizer

MODEL_SAVE_PATH = "glm4_moe_tiny_random"

config_dict = {
    "architectures": ["Glm4MoeForCausalLM"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "eos_token_id": [151329, 151336, 151338],
    "first_k_dense_replace": 1,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 16,
    "initializer_range": 0.02,
    "intermediate_size": 64,
    "max_position_embeddings": 131072,
    "model_type": "glm4_moe",
    "moe_intermediate_size": 64,
    "n_group": 1,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "num_nextn_predict_layers": 1,
    "pad_token_id": 151329,
    "partial_rotary_factor": 0.5,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "routed_scaling_factor": 2.5,
    "tie_word_embeddings": False,
    "topk_group": 1,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.55.0.dev0",
    "use_cache": True,
    "use_qk_norm": True,
    "vocab_size": 151552,
}


def create_tiny_random_model():
    print(f"Creating tiny random GLM-4.5 MoE model at {MODEL_SAVE_PATH}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Save config
    config_path = os.path.join(MODEL_SAVE_PATH, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")

    # Create model with random weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_SAVE_PATH)

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16)

    # Save model
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saved random model to {MODEL_SAVE_PATH}")

    # Check model state dict keys
    sd = model.state_dict()
    print("\nSample state dict keys:")
    for i, (k, v) in enumerate(sd.items()):
        if i < 30:
            print(f"  {k}: {v.shape} {v.dtype}")
    print(f"  ... total {len(sd)} keys")

    # Try to load tokenizer from a known GLM model
    # Use GLM-4's tokenizer if available, else create a simple one
    try:
        from transformers import AutoTokenizer

        # Check if there's a GLM-4 tokenizer available
        tokenizer_path = "/shared/cache/checkpoints/glm/glm-4-9b"
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            print(f"Saved tokenizer from {tokenizer_path}")
        else:
            # Create minimal tokenizer config
            tokenizer_config = {
                "bos_token_id": 151329,
                "eos_token_id": [151329, 151336, 151338],
                "model_max_length": 131072,
                "padding_side": "left",
                "tokenizer_class": "PreTrainedTokenizerFast",
            }
            with open(os.path.join(MODEL_SAVE_PATH, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            print("Created minimal tokenizer config (no full tokenizer available)")
    except Exception as e:
        print(f"Warning: could not save tokenizer: {e}")

    return model, config


if __name__ == "__main__":
    model, config = create_tiny_random_model()
    print("\nModel created successfully!")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  n_routed_experts: {config.n_routed_experts}")
    print(f"  n_shared_experts: {config.n_shared_experts}")
    print(f"  num_experts_per_tok: {config.num_experts_per_tok}")
    print(f"  first_k_dense_replace: {config.first_k_dense_replace}")
    print(f"  partial_rotary_factor: {config.partial_rotary_factor}")
    print(f"  attention_bias: {config.attention_bias}")
    print(f"  use_qk_norm: {config.use_qk_norm}")
