# Copyright 2025 © Amazon.com and Affiliates
"""Test Isaac scaling: sequence length and batch size.

Tests compilation and throughput at various seq_len and batch_size.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH

    # Test single config
    python test_scaling.py --seq-len 2048 --batch-size 1

    # Test all configs (sequential)
    python test_scaling.py --sweep
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
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


def get_hbm_usage():
    """Get current HBM usage from neuron-ls."""
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for device in data:
                mem = device.get("neuron_device", {}).get("memory", {})
                used = mem.get("used_bytes", 0)
                total = mem.get("total_bytes", 0)
                return used / 1e9, total / 1e9  # GB
    except Exception:
        pass
    return None, None


def create_config(seq_len, batch_size, tp=1):
    """Create configs for a given seq_len and batch_size."""
    traced_path = f"{DATA_PATH}/traced_model/Isaac-2B-s{seq_len}-b{batch_size}-tp{tp}"

    # Build bucketing: CTE uses the seq_len bucket, TKG uses same
    cte_buckets = [seq_len]
    tkg_buckets = [seq_len]

    text_config = NeuronConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        tp_degree=tp,
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        is_continuous_batching=True,
        ctx_batch_size=batch_size,
        enable_bucketing=True,
        context_encoding_buckets=cte_buckets,
        token_generation_buckets=tkg_buckets,
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
        fused_qkv=False,
        sequence_parallel_enabled=False,
        # Enable CTE flash attention (verified working)
        attn_kernel_enabled=True,
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    vision_config = NeuronConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        tp_degree=tp,
        world_size=tp,
        save_sharded_checkpoint=True,
        is_continuous_batching=True,
        ctx_batch_size=batch_size,
        enable_bucketing=True,
        buckets=[batch_size],
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


def test_config(seq_len, batch_size, tp=1, force_recompile=True):
    """Test a single seq_len + batch_size configuration."""
    print(f"\n{'=' * 70}")
    print(f"Testing: seq_len={seq_len}, batch_size={batch_size}, TP={tp}")
    print(f"{'=' * 70}")

    result = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "tp": tp,
        "compiled": False,
        "inference_ok": False,
        "compile_time": None,
        "hbm_used_gb": None,
        "hbm_total_gb": None,
        "ttft_ms": None,
        "tkg_tok_per_sec": None,
        "error": None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    config, traced_path = create_config(seq_len, batch_size, tp)

    if force_recompile and os.path.exists(traced_path):
        shutil.rmtree(traced_path)

    # Compile
    try:
        t0 = time.time()
        model = NeuronIsaacForConditionalGeneration(MODEL_PATH, config)
        model.compile(traced_path, debug=False)
        tokenizer.save_pretrained(traced_path)
        compile_time = time.time() - t0
        result["compiled"] = True
        result["compile_time"] = compile_time
        print(f"  Compiled in {compile_time:.1f}s")
    except Exception as e:
        result["error"] = str(e)[:500]
        print(f"  COMPILATION FAILED: {str(e)[:200]}")
        traceback.print_exc()
        return result

    # Load
    try:
        model.load(traced_path, skip_warmup=True)
    except Exception as e:
        result["error"] = f"Load failed: {str(e)[:400]}"
        print(f"  LOAD FAILED: {str(e)[:200]}")
        return result

    # HBM usage
    hbm_used, hbm_total = get_hbm_usage()
    result["hbm_used_gb"] = hbm_used
    result["hbm_total_gb"] = hbm_total
    if hbm_used:
        print(f"  HBM: {hbm_used:.1f} / {hbm_total:.1f} GB")

    # Inference test
    generation_model = HuggingFaceGenerationAdapter(model)
    prompt = "Explain the theory of relativity in detail, covering both special and general relativity:"
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    # For BS > 1, replicate input
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)

    attention_mask = torch.ones_like(input_ids)

    sampling_params = prepare_sampling_params(
        batch_size=batch_size,
        top_k=[1] * batch_size,
        top_p=[1.0] * batch_size,
        temperature=[1.0] * batch_size,
    )
    gen_config = GenerationConfig(
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
    )

    try:
        # TTFT: first token time
        t0 = time.time()
        outputs = generation_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=model.config.neuron_config.max_length,
            sampling_params=sampling_params,
            generation_config=gen_config,
            max_new_tokens=50,
        )
        total_time = time.time() - t0

        generated = outputs.sequences[0, input_ids.shape[1] :]
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        n_tokens = len(generated)

        # TTFT approximation (first score is first token)
        if hasattr(outputs, "scores") and len(outputs.scores) > 0:
            # Rough: total_time / n_tokens gives TPOT, TTFT ≈ total_time - (n_tokens-1)*TPOT
            tpot = total_time / n_tokens if n_tokens > 1 else total_time
            ttft = total_time - (n_tokens - 1) * tpot if n_tokens > 1 else total_time
        else:
            ttft = total_time
            tpot = total_time / n_tokens if n_tokens > 0 else 0

        tok_per_sec = (n_tokens * batch_size) / total_time if total_time > 0 else 0

        result["inference_ok"] = True
        result["ttft_ms"] = ttft * 1000
        result["tkg_tok_per_sec"] = tok_per_sec
        result["tpot_ms"] = tpot * 1000
        result["n_tokens"] = n_tokens
        result["text_preview"] = gen_text[:100]

        print(f"  Generated: {n_tokens} tokens in {total_time:.3f}s")
        print(f"  TTFT: ~{ttft * 1000:.1f}ms, TPOT: ~{tpot * 1000:.1f}ms")
        print(f"  Throughput: {tok_per_sec:.1f} tok/s (total across batch)")
        print(f"  Text: {gen_text[:80]!r}")

    except Exception as e:
        result["error"] = f"Inference failed: {str(e)[:400]}"
        print(f"  INFERENCE FAILED: {str(e)[:200]}")
        traceback.print_exc()

    # Cleanup
    del model
    del generation_model
    import gc

    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sweep", action="store_true", help="Run full sweep")
    parser.add_argument("--no-recompile", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        # Sweep configurations: seq_len first, then batch_size
        configs = [
            # Seq len sweep (BS=1)
            (1024, 1),  # baseline
            (2048, 1),
            (4096, 1),
            (8192, 1),
            # Batch size sweep (seq_len=1024)
            (1024, 2),
            (1024, 4),
            (1024, 8),
        ]

        results = []
        for sl, bs in configs:
            r = test_config(sl, bs, tp=args.tp, force_recompile=not args.no_recompile)
            results.append(r)

        # Summary
        print(f"\n{'=' * 80}")
        print("SCALING TEST SUMMARY")
        print(f"{'=' * 80}")
        print(
            f"{'seq_len':>8} {'BS':>4} {'Compiled':>10} {'CompileT':>10} "
            f"{'HBM(GB)':>10} {'TTFT(ms)':>10} {'tok/s':>10} {'TPOT(ms)':>10}"
        )
        print("-" * 80)
        for r in results:
            comp = "YES" if r["compiled"] else "FAIL"
            ct = f"{r['compile_time']:.0f}" if r["compile_time"] else "N/A"
            hbm = f"{r['hbm_used_gb']:.1f}" if r["hbm_used_gb"] else "N/A"
            ttft = f"{r['ttft_ms']:.1f}" if r["ttft_ms"] else "N/A"
            tps = f"{r['tkg_tok_per_sec']:.1f}" if r["tkg_tok_per_sec"] else "N/A"
            tpot = f"{r.get('tpot_ms', 0):.1f}" if r.get("tpot_ms") else "N/A"
            print(
                f"{r['seq_len']:>8} {r['batch_size']:>4} {comp:>10} {ct:>10} "
                f"{hbm:>10} {ttft:>10} {tps:>10} {tpot:>10}"
            )

        # Save
        out_path = os.path.join(REFERENCE_DIR, "scaling_test_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    else:
        r = test_config(
            args.seq_len,
            args.batch_size,
            tp=args.tp,
            force_recompile=not args.no_recompile,
        )
        print(f"\nResult: {json.dumps(r, indent=2, default=str)}")


if __name__ == "__main__":
    main()
