"""
Logit matching test for DeepSeek V3 NXDI vs HF reference.

Creates a mini DeepSeek V3 model (1 dense + 1 MoE layer, vocab=32000),
runs forward passes through both HF (CPU, FP32) and NXDI (Neuron, BF16),
and compares the first predicted token.

Requirements:
  - Neuron device (trn1/trn2 instance with at least 2 NeuronCores)
  - A tokenizer source (uses LLaMA tokenizer from HuggingFace by default)

Usage:
  python examples/test_logit_matching.py
  python examples/test_logit_matching.py --tokenizer-path /path/to/tokenizer
  python examples/test_logit_matching.py --tp-degree 2 --seq-len 128
"""
import argparse
import gc
import json
import os
import shutil
import tempfile

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3InferenceConfig,
    NeuronDeepseekV3ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

torch.manual_seed(42)

# Mini model config: 1 dense + 1 MoE layer, small but realistic dimensions
MINI_CONFIG = {
    "_name_or_path": "deepseek-ai/DeepSeek-V3",
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "deepseek-ai/DeepSeek-V3--configuration_deepseek.DeepseekV3Config",
        "AutoModel": "deepseek-ai/DeepSeek-V3--modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "deepseek-ai/DeepSeek-V3--modeling_deepseek.DeepseekV3ForCausalLM",
    },
    "aux_loss_alpha": 0.001,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "ep_size": 1,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "kv_lora_rank": 512,
    "max_position_embeddings": 4096,
    "model_type": "deepseek_v3",
    "moe_intermediate_size": 1408,
    "moe_layer_freq": 1,
    "n_group": 8,
    "n_routed_experts": 64,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 16,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 16,
    "num_nextn_predict_layers": 0,
    "pretraining_tp": 1,
    "q_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32, "beta_slow": 1, "factor": 40,
        "mscale": 1.0, "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096, "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "seq_aux": True,
    "tie_word_embeddings": False,
    "topk_group": 4,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.38.2",
    "use_cache": True,
    "v_head_dim": 128,
    "vocab_size": 32000,
}


def create_mini_model(model_dir, tokenizer_path=None):
    """Create a mini DeepSeek V3 model with random weights."""
    os.makedirs(model_dir, exist_ok=True)

    cfg = MINI_CONFIG
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    hidden = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]
    moe_intermediate = cfg["moe_intermediate_size"]
    vocab = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    kv_lora_rank = cfg["kv_lora_rank"]
    q_lora_rank = cfg["q_lora_rank"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    n_experts = cfg["n_routed_experts"]
    n_layers = cfg["num_hidden_layers"]
    first_k_dense = cfg["first_k_dense_replace"]

    sd = {}
    sd["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.02
    for i in range(n_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.q_a_proj.weight"] = torch.randn(q_lora_rank, hidden, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.q_a_layernorm.weight"] = torch.ones(q_lora_rank, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.q_b_proj.weight"] = torch.randn(n_heads * (qk_nope + qk_rope), q_lora_rank, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(kv_lora_rank + qk_rope, hidden, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.kv_a_layernorm.weight"] = torch.ones(kv_lora_rank, dtype=torch.bfloat16)
        sd[f"{p}.self_attn.kv_b_proj.weight"] = torch.randn(n_heads * (qk_nope + v_head), kv_lora_rank, dtype=torch.bfloat16) * 0.02
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, n_heads * v_head, dtype=torch.bfloat16) * 0.02
        if i < first_k_dense:
            sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.up_proj.weight"] = torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02
        else:
            sd[f"{p}.mlp.gate.weight"] = torch.randn(n_experts, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.gate.e_score_correction_bias"] = torch.randn(n_experts, dtype=torch.bfloat16) * 0.01
            for e in range(n_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(moe_intermediate, hidden, dtype=torch.bfloat16) * 0.02
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(moe_intermediate, hidden, dtype=torch.bfloat16) * 0.02
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(hidden, moe_intermediate, dtype=torch.bfloat16) * 0.02
            shared_int = moe_intermediate * cfg["n_shared_experts"]
            sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(shared_int, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(shared_int, hidden, dtype=torch.bfloat16) * 0.02
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(hidden, shared_int, dtype=torch.bfloat16) * 0.02

    sd["model.norm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.02
    save_file(sd, os.path.join(model_dir, "model.safetensors"))

    # Copy tokenizer from source or create minimal one
    if tokenizer_path and os.path.exists(tokenizer_path):
        for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
            src = os.path.join(tokenizer_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(model_dir, fname))
    else:
        # Download a small tokenizer (LLaMA-2 compatible)
        print("Downloading tokenizer from HuggingFace...")
        tok = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        tok.save_pretrained(model_dir)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3 logit matching test")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to existing tokenizer (copies files from here)")
    parser.add_argument("--model-dir", type=str, default="/tmp/deepseek-v3-logit-test",
                        help="Directory for mini model weights")
    parser.add_argument("--traced-path", type=str, default="/tmp/deepseek_v3_logit_traced",
                        help="Directory for compiled model")
    parser.add_argument("--tp-degree", type=int, default=2, help="Tensor parallelism degree")
    parser.add_argument("--seq-len", type=int, default=128, help="Max sequence length")
    args = parser.parse_args()

    # Step 0: Create mini model
    print("Creating mini model...")
    create_mini_model(args.model_dir, args.tokenizer_path)

    # Step 1: HF reference
    print("\n" + "=" * 60)
    print("HF REFERENCE (CPU, FP32)")
    print("=" * 60)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

    with torch.no_grad():
        hf_out = hf_model(input_ids, use_cache=False)
    hf_last_logits = hf_out.logits[0, -1, :].float()
    hf_top1 = hf_last_logits.argmax().item()
    hf_top5 = hf_last_logits.topk(5).indices.tolist()
    print(f"HF top-1: {hf_top1} ({tokenizer.decode([hf_top1])})")
    print(f"HF top-5: {hf_top5}")

    del hf_model
    gc.collect()

    # Step 2: NXDI compilation and generation
    print("\n" + "=" * 60)
    print("NXDI MODEL (Neuron, BF16)")
    print("=" * 60)

    neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
        seq_len=args.seq_len, torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False, flash_decoding_enabled=False, logical_nc_config=2,
        blockwise_matmul_config={"block_size": 999999},
    )
    inf_config = DeepseekV3InferenceConfig(
        neuron_config, load_config=load_pretrained_config(args.model_dir),
    )

    print("Compiling...")
    model = NeuronDeepseekV3ForCausalLM(args.model_dir, inf_config)
    model.compile(args.traced_path)
    del model
    gc.collect()

    print("Loading...")
    model = NeuronDeepseekV3ForCausalLM(args.traced_path)
    model.load(args.traced_path)
    tokenizer.save_pretrained(args.traced_path)

    print("Generating...")
    generation_config = GenerationConfig(
        do_sample=True, top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        input_ids, generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=input_ids.shape[1] + 5,
    )
    nxdi_tokens = outputs[0].tolist()
    nxdi_text = tokenizer.decode(nxdi_tokens, skip_special_tokens=True)
    print(f"NXDI output: {nxdi_text}")

    # Step 3: Compare
    nxdi_first = nxdi_tokens[input_ids.shape[1]]
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  HF  first token: {hf_top1} ({tokenizer.decode([hf_top1])})")
    print(f"  NXDI first token: {nxdi_first} ({tokenizer.decode([nxdi_first])})")

    if nxdi_first == hf_top1:
        print("  STATUS: PASS (exact match)")
    elif nxdi_first in hf_top5:
        pos = hf_top5.index(nxdi_first) + 1
        print(f"  STATUS: CLOSE (in HF top-5, position {pos})")
    else:
        print("  STATUS: FAIL (not in HF top-5)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
