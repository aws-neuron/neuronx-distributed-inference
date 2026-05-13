#!/usr/bin/env python3
"""Kernel sweep for Isaac-0.2-2B: test TKG attention block, MLP, out_proj, and combos.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:/mnt/models/neuronx-distributed-inference/src:$PYTHONPATH \
        python3 kernel_sweep.py
"""

import os
import sys
import time
import json
import torch
import traceback

# Ensure the correct paths
NXDI_ROOT = "/mnt/models/neuronx-distributed-inference"
sys.path.insert(0, f"{NXDI_ROOT}/contrib/models/Isaac-0.2-2B/src")
sys.path.insert(0, f"{NXDI_ROOT}/src")

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

from transformers import AutoConfig, AutoTokenizer
from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)
from isaac_neuron.modeling_isaac import (
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

MODEL_PATH = "/mnt/models/Isaac-0.2-2B-Preview"
COMPILED_BASE = "/mnt/models/traced_model/Isaac-0.2-2B"

# Kernel configurations to test
CONFIGS = {
    "baseline": {
        "desc": "No kernels (reference)",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": False,
            "mlp_kernel_enabled": False,
            "fused_qkv": False,
        },
    },
    "cte_flash_only": {
        "desc": "CTE flash attention only (current production config)",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": True,
            "mlp_kernel_enabled": False,
            "fused_qkv": False,
        },
    },
    "mlp_tp1": {
        "desc": "MLP kernel at TP=1 (nkilib production, NOT experimental)",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": True,
            "mlp_kernel_enabled": True,
            "fused_qkv": False,
        },
    },
    "tkg_block": {
        "desc": "TKG attention block kernel (fuses RMSNorm+QKV+QKnorm+RoPE+Attn+Oproj)",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": True,
            "mlp_kernel_enabled": False,
            "fused_qkv": True,
            "qkv_kernel_enabled": True,
            "attn_block_tkg_nki_kernel_enabled": True,
        },
    },
    "tkg_block_plus_mlp": {
        "desc": "TKG block + MLP kernel (full TKG optimization)",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": True,
            "mlp_kernel_enabled": True,
            "fused_qkv": True,
            "qkv_kernel_enabled": True,
            "attn_block_tkg_nki_kernel_enabled": True,
        },
    },
    "out_proj": {
        "desc": "CTE flash + out_proj kernel",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": True,
            "mlp_kernel_enabled": False,
            "fused_qkv": False,
            "out_proj_kernel_enabled": True,
        },
    },
    "tkg_block_mlp_outproj": {
        "desc": "TKG block + MLP + out_proj (maximum kernel coverage)",
        "tp": 1,
        "flags": {
            "attn_kernel_enabled": True,
            "mlp_kernel_enabled": True,
            "fused_qkv": True,
            "qkv_kernel_enabled": True,
            "attn_block_tkg_nki_kernel_enabled": True,
            "out_proj_kernel_enabled": True,
        },
    },
}


def build_config(config_name, flags, tp_degree=1, seq_len=1024):
    """Build IsaacInferenceConfig with specified kernel flags."""
    compiled_dir = f"{COMPILED_BASE}/kernel_sweep_{config_name}"

    text_config = NeuronConfig(
        batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        tp_degree=tp_degree,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[seq_len],
        token_generation_buckets=[seq_len],
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=True,
            do_sample=True,
            deterministic=True,
            top_k=1,
            global_topk=256,
            top_k_kernel_enabled=True,
        ),
        output_logits=True,
        save_sharded_checkpoint=True,
        **flags,
    )

    vision_config = NeuronConfig(
        batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        tp_degree=tp_degree,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        buckets=[1],
        save_sharded_checkpoint=True,
        fused_qkv=False,
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    inference_config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    # Override save/compiled paths
    inference_config.save_path = compiled_dir
    inference_config.compiled_model_path = compiled_dir

    return inference_config


def test_config(config_name, config_info):
    """Test a single kernel configuration: compile, load, generate, benchmark."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {config_name}")
    print(f"  {config_info['desc']}")
    print(f"  Flags: {config_info['flags']}")
    print(f"{'=' * 70}")

    tp = config_info["tp"]
    flags = config_info["flags"]

    try:
        inference_config = build_config(config_name, flags, tp_degree=tp)
        compiled_dir = f"{COMPILED_BASE}/kernel_sweep_{config_name}"

        # Compile
        t0 = time.time()
        print(f"  Compiling...")
        model = NeuronIsaacForConditionalGeneration(MODEL_PATH, inference_config)
        model.compile(compiled_dir, debug=False)
        compile_time = time.time() - t0
        print(f"  Compile time: {compile_time:.1f}s")

        # Load
        print(f"  Loading compiled model...")
        model.load(compiled_dir, skip_warmup=True)

        # Generate text-only
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        prompt = "What is the capital of France?"
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        attention_mask = torch.ones_like(input_ids)

        generation_model = HuggingFaceGenerationAdapter(model)
        sampling_params = prepare_sampling_params(
            batch_size=1,
            top_k=[1],
            top_p=[1.0],
            temperature=[0.0],
        )
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            max_new_tokens=50,
        )

        # Warmup
        print(f"  Warmup (3 runs)...")
        for _ in range(3):
            out = generation_model.generate(input_ids, **gen_kwargs)

        # Benchmark (10 runs)
        print(f"  Benchmarking (10 runs, 50 tokens each)...")
        times = []
        for _ in range(10):
            t0 = time.time()
            out = generation_model.generate(input_ids, **gen_kwargs)
            times.append(time.time() - t0)

        output_text = tokenizer.decode(
            out[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        avg_time = sum(times) / len(times)
        tok_per_sec = 50 / avg_time
        tpot_ms = (avg_time / 50) * 1000

        result = {
            "status": "SUCCESS",
            "compile_time_s": compile_time,
            "avg_time_s": avg_time,
            "tok_per_sec": tok_per_sec,
            "tpot_ms": tpot_ms,
            "output_preview": output_text[:100],
        }
        print(f"  tok/s: {tok_per_sec:.1f}")
        print(f"  TPOT:  {tpot_ms:.2f} ms")
        print(f"  Output: {output_text[:80]}...")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"  FAILED: {e}")
        print(f"  {tb[-500:]}")
        return {
            "status": "FAILED",
            "error": str(e),
            "traceback": tb[-500:],
        }


def main():
    # Parse args
    configs_to_test = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())

    print(f"Isaac Kernel Sweep")
    print(f"Configs to test: {configs_to_test}")
    print(f"Model: {MODEL_PATH}")

    results = {}
    for name in configs_to_test:
        if name not in CONFIGS:
            print(f"Unknown config: {name}, skipping")
            continue
        results[name] = test_config(name, CONFIGS[name])

    # Summary
    print(f"\n{'=' * 80}")
    print(f"KERNEL SWEEP SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<25} {'Status':<10} {'tok/s':>8} {'TPOT ms':>10} {'Compile':>10}")
    print("-" * 70)
    for name, r in results.items():
        if r["status"] == "SUCCESS":
            print(
                f"{name:<25} {'OK':<10} {r['tok_per_sec']:>8.1f} {r['tpot_ms']:>10.2f} {r['compile_time_s']:>10.1f}s"
            )
        else:
            print(f"{name:<25} {'FAIL':<10} {'—':>8} {'—':>10} {'—':>10}")

    # Save results
    out_path = "/mnt/models/kernel_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
