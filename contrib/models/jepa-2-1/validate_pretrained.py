"""
Pretrained weight validation for V-JEPA 2.1 on Neuron.

Validates that pretrained ViT-B and ViT-L produce correct features on Neuron
by comparing BF16 Neuron output against FP32 CPU reference.

Usage (on trn2.3xlarge):
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python validate_pretrained.py

Outputs:
    - Cosine similarity (BF16 Neuron vs FP32 CPU)
    - Feature statistics (mean, std, min, max)
    - Latency benchmarks (single NC, 100 iterations)
    - DataParallel throughput (2 NCs)
"""

import sys
import os
import time
import json

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from modeling_jepa21 import build_vjepa21_encoder

try:
    import torch_neuronx
    HAS_NEURON = True
except ImportError:
    HAS_NEURON = False


def cosine_similarity(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def feature_stats(t):
    t = t.float()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "norm": t.norm().item(),
        "has_nan": bool(t.isnan().any()),
        "has_inf": bool(t.isinf().any()),
    }


def benchmark_latency(traced_model, example, warmup=10, iterations=100):
    for _ in range(warmup):
        traced_model(example)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        traced_model(example)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "p5_ms": float(np.percentile(times, 5)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def validate_arch(arch, input_shapes):
    """Validate a single architecture with pretrained weights."""
    print(f"\n{'='*60}")
    print(f"Validating {arch} (pretrained=True)")
    print(f"{'='*60}")

    # Build CPU reference (FP32)
    print(f"Loading pretrained {arch} weights...")
    encoder_cpu = build_vjepa21_encoder(
        arch=arch, img_size=384, num_frames=16,
        pretrained=True, use_sdpa=False,
    )
    encoder_cpu.eval()

    results = {"arch": arch, "inputs": []}

    for name, shape in input_shapes:
        print(f"\n--- Input: {name} {shape} ---")
        torch.manual_seed(42)
        x = torch.randn(*shape)

        # CPU FP32 reference
        with torch.no_grad():
            cpu_out = encoder_cpu(x)
        cpu_stats = feature_stats(cpu_out)
        print(f"  CPU FP32 output: shape={cpu_out.shape}, mean={cpu_stats['mean']:.4f}, std={cpu_stats['std']:.4f}, norm={cpu_stats['norm']:.1f}")

        if not HAS_NEURON:
            print("  [SKIP] No Neuron hardware — CPU-only validation")
            results["inputs"].append({
                "name": name, "shape": list(shape),
                "cpu_stats": cpu_stats, "neuron": "skipped",
            })
            continue

        # Trace for Neuron (BF16)
        encoder_bf16 = build_vjepa21_encoder(
            arch=arch, img_size=384, num_frames=16,
            pretrained=True, use_sdpa=False,
        )
        # Load same weights
        encoder_bf16.load_state_dict(encoder_cpu.state_dict())
        encoder_bf16.eval().bfloat16()

        x_bf16 = x.bfloat16()
        print(f"  Tracing for Neuron...")
        traced = torch_neuronx.trace(encoder_bf16, x_bf16, compiler_args=["--auto-cast", "none"])

        neuron_out = traced(x_bf16)
        neuron_stats = feature_stats(neuron_out)
        cos_sim = cosine_similarity(cpu_out, neuron_out)

        print(f"  Neuron BF16 output: shape={neuron_out.shape}, mean={neuron_stats['mean']:.4f}, std={neuron_stats['std']:.4f}, norm={neuron_stats['norm']:.1f}")
        print(f"  Cosine similarity (BF16 Neuron vs FP32 CPU): {cos_sim:.6f}")
        print(f"  NaN: {neuron_stats['has_nan']}, Inf: {neuron_stats['has_inf']}")

        # Benchmark
        print(f"  Benchmarking (100 iterations)...")
        latency = benchmark_latency(traced, x_bf16)
        print(f"  Latency: median={latency['median_ms']:.1f}ms, p5={latency['p5_ms']:.1f}ms, p95={latency['p95_ms']:.1f}ms")

        # DataParallel
        print(f"  DataParallel (2 NCs)...")
        model_dp = torch_neuronx.DataParallel(traced)
        x_dp = x_bf16.expand(2, -1, -1, -1, -1).contiguous()  # batch=2
        dp_latency = benchmark_latency(model_dp, x_dp, warmup=5, iterations=50)
        dp_throughput = 2.0 / (dp_latency["median_ms"] / 1000.0)
        print(f"  DataParallel: median={dp_latency['median_ms']:.1f}ms for 2 clips, throughput={dp_throughput:.1f} clips/sec")

        input_result = {
            "name": name,
            "shape": list(shape),
            "output_shape": list(neuron_out.shape),
            "cosine_similarity": cos_sim,
            "cpu_stats": cpu_stats,
            "neuron_stats": neuron_stats,
            "latency": latency,
            "dataparallel_latency": dp_latency,
            "dataparallel_throughput": dp_throughput,
        }
        results["inputs"].append(input_result)

        # Save traced model
        save_name = f"vjepa21_{arch.replace('vit_', 'vit')}_pretrained_{name}.pt"
        traced.save(save_name)
        print(f"  Saved: {save_name}")

        # Cleanup to free memory
        del traced, model_dp, encoder_bf16
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    archs = [
        ("vit_base", [
            ("image_1f", (1, 3, 1, 384, 384)),
            ("video_16f", (1, 3, 16, 384, 384)),
        ]),
        ("vit_large", [
            ("image_1f", (1, 3, 1, 384, 384)),
            ("video_16f", (1, 3, 16, 384, 384)),
        ]),
    ]

    all_results = []
    for arch, inputs in archs:
        result = validate_arch(arch, inputs)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        for inp in r["inputs"]:
            if isinstance(inp.get("neuron"), str):
                print(f"  {r['arch']} / {inp['name']}: SKIPPED (no Neuron)")
            else:
                cos = inp["cosine_similarity"]
                lat = inp["latency"]["median_ms"]
                status = "PASS" if cos > 0.999 else "WARN" if cos > 0.99 else "FAIL"
                print(f"  {r['arch']} / {inp['name']}: cos_sim={cos:.6f} latency={lat:.1f}ms [{status}]")

    # Save results
    out_file = "validation_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
