# Copyright 2025 © Amazon.com and Affiliates
"""Test NKI kernel enablement for Isaac at TP=1.

Incrementally enables kernels and validates:
1. Compilation succeeds
2. Accuracy matches baseline (cosine vs CPU reference)
3. Throughput improvement

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python test_kernels.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, GenerationConfig  # noqa: E402

from neuronx_distributed_inference.models.config import (  # noqa: E402
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: E402
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import (  # noqa: E402
    prepare_sampling_params,
)

from isaac_neuron.modeling_isaac import (  # noqa: E402
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")
REFERENCE_DIR = f"{DATA_PATH}/reference_outputs"
MODEL_PATH = f"{DATA_PATH}/Isaac-0.2-2B-Preview"

os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)

# Kernel configurations to test (incremental enablement)
KERNEL_CONFIGS = {
    "baseline": {
        "description": "No kernels (current default)",
        "text_config": {
            "fused_qkv": False,
            "attn_kernel_enabled": False,
            "attn_tkg_nki_kernel_enabled": False,
            "attn_tkg_builtin_kernel_enabled": False,
            "qkv_kernel_enabled": False,
            "mlp_kernel_enabled": False,
        },
    },
    "cte_flash_attn": {
        "description": "CTE flash attention only",
        "text_config": {
            "fused_qkv": False,
            "attn_kernel_enabled": True,
            "attn_tkg_nki_kernel_enabled": False,
            "attn_tkg_builtin_kernel_enabled": False,
            "qkv_kernel_enabled": False,
            "mlp_kernel_enabled": False,
        },
    },
    "mlp_kernel": {
        "description": "MLP kernel only",
        "text_config": {
            "fused_qkv": False,
            "attn_kernel_enabled": False,
            "attn_tkg_nki_kernel_enabled": False,
            "attn_tkg_builtin_kernel_enabled": False,
            "qkv_kernel_enabled": False,
            "mlp_kernel_enabled": True,
        },
    },
    "qkv_kernel": {
        "description": "QKV kernel (requires fused_qkv)",
        "text_config": {
            "fused_qkv": True,
            "attn_kernel_enabled": False,
            "attn_tkg_nki_kernel_enabled": False,
            "attn_tkg_builtin_kernel_enabled": False,
            "qkv_kernel_enabled": True,
            "qkv_nki_kernel_enabled": True,
            "mlp_kernel_enabled": False,
        },
    },
    "cte_flash_plus_mlp": {
        "description": "CTE flash attention + MLP kernel",
        "text_config": {
            "fused_qkv": False,
            "attn_kernel_enabled": True,
            "attn_tkg_nki_kernel_enabled": False,
            "attn_tkg_builtin_kernel_enabled": False,
            "qkv_kernel_enabled": False,
            "mlp_kernel_enabled": True,
        },
    },
    "full_suite": {
        "description": "All kernels: CTE flash + QKV + MLP + fused residual",
        "text_config": {
            "fused_qkv": True,
            "attn_kernel_enabled": True,
            "attn_tkg_nki_kernel_enabled": False,
            "attn_tkg_builtin_kernel_enabled": False,
            "qkv_kernel_enabled": True,
            "qkv_nki_kernel_enabled": True,
            "mlp_kernel_enabled": True,
            "mlp_kernel_fuse_residual_add": True,
            "qkv_kernel_fuse_residual_add": True,
            "out_proj_kernel_enabled": True,
        },
    },
}

PROMPTS = [
    "The capital of France is",
    "Explain quantum entanglement in simple terms:",
]


def create_config(kernel_name, kernel_cfg):
    """Create config with specified kernel settings."""
    traced_path = f"{DATA_PATH}/traced_model/Isaac-0.2-2B-kernel-{kernel_name}"

    text_overrides = kernel_cfg["text_config"]

    text_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[1024],
        token_generation_buckets=[1024],
        async_mode=False,
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=True,
            do_sample=True,
            deterministic=True,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            global_topk=256,
            top_k_kernel_enabled=True,
        ),
        output_logits=True,
        sequence_parallel_enabled=False,
        **text_overrides,
    )

    vision_config = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        world_size=1,
        save_sharded_checkpoint=True,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        buckets=[1],
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    config.image_token_index = 151655

    return config, traced_path


def test_kernel_config(kernel_name, kernel_cfg, tokenizer):
    """Test a single kernel configuration."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {kernel_name} — {kernel_cfg['description']}")
    print(f"{'=' * 70}")

    config, traced_path = create_config(kernel_name, kernel_cfg)
    result = {
        "name": kernel_name,
        "description": kernel_cfg["description"],
        "compiled": False,
        "accuracy_pass": False,
        "prompts": [],
        "compile_time": None,
        "error": None,
    }

    # Clean and compile
    if os.path.exists(traced_path):
        shutil.rmtree(traced_path)

    try:
        t0 = time.time()
        model = NeuronIsaacForConditionalGeneration(MODEL_PATH, config)
        model.compile(traced_path, debug=False)
        tokenizer.save_pretrained(traced_path)
        compile_time = time.time() - t0
        model.load(traced_path, skip_warmup=True)
        result["compiled"] = True
        result["compile_time"] = compile_time
        print(f"  Compiled in {compile_time:.1f}s")
    except Exception as e:
        result["error"] = str(e)
        print(f"  COMPILATION FAILED: {e}")
        traceback.print_exc()
        return result

    # Validate accuracy
    generation_model = HuggingFaceGenerationAdapter(model)
    all_passed = True

    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        attention_mask = torch.ones_like(input_ids)

        sampling_params = prepare_sampling_params(
            batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0]
        )
        gen_config = GenerationConfig(
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
        )

        t0 = time.time()
        outputs = generation_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=50,
        )
        elapsed = time.time() - t0

        generated = outputs.sequences[0, input_ids.shape[1] :]
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        n_tokens = len(generated)
        tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0

        # Compare first-token logits
        neuron_logits = outputs.scores[0][0].float().cpu()
        ref_path = os.path.join(REFERENCE_DIR, f"text_logits_{i:03d}.pt")
        cosine = -1.0
        if os.path.exists(ref_path):
            ref_logits = torch.load(ref_path, map_location="cpu")
            cosine = F.cosine_similarity(
                neuron_logits.unsqueeze(0), ref_logits.unsqueeze(0)
            ).item()

        top1_match = neuron_logits.argmax().item() == 151667
        passed = cosine >= 0.99 and top1_match
        if not passed:
            all_passed = False

        prompt_result = {
            "prompt": prompt,
            "cosine": cosine,
            "top1_match": top1_match,
            "passed": passed,
            "text": gen_text[:200],
            "n_tokens": n_tokens,
            "tok_per_sec": tok_per_sec,
            "elapsed": elapsed,
        }
        result["prompts"].append(prompt_result)
        print(
            f"  Prompt {i}: cosine={cosine:.6f}, top1={'OK' if top1_match else 'MISS'}, "
            f"{n_tokens} tok, {tok_per_sec:.1f} tok/s | {gen_text[:60]!r}"
        )

    result["accuracy_pass"] = all_passed

    # Cleanup model to free NeuronCores
    del model
    del generation_model
    import gc

    gc.collect()

    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    for name, cfg in KERNEL_CONFIGS.items():
        r = test_kernel_config(name, cfg, tokenizer)
        results.append(r)

    # Summary table
    print(f"\n{'=' * 70}")
    print("KERNEL TEST SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Config':<25} {'Compiled':>10} {'Accuracy':>10} {'Compile(s)':>12} {'tok/s (avg)':>12}"
    )
    print("-" * 70)
    for r in results:
        compiled = "YES" if r["compiled"] else "FAIL"
        accuracy = "PASS" if r["accuracy_pass"] else "FAIL"
        compile_t = f"{r['compile_time']:.1f}" if r["compile_time"] else "N/A"
        avg_tps = "N/A"
        if r["prompts"]:
            tps_vals = [p["tok_per_sec"] for p in r["prompts"] if p["tok_per_sec"] > 0]
            if tps_vals:
                avg_tps = f"{sum(tps_vals) / len(tps_vals):.1f}"
        print(
            f"{r['name']:<25} {compiled:>10} {accuracy:>10} {compile_t:>12} {avg_tps:>12}"
        )

    # Save results
    out_path = os.path.join(REFERENCE_DIR, "kernel_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
