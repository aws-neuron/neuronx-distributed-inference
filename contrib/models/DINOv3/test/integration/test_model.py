"""
Integration tests for DINOv3 vision models on Neuron.

Tests accuracy (cosine similarity), compilation, inference, DataParallel,
and performance for DINOv3 ViT and ConvNeXt models compiled with torch_neuronx.trace().

Requires:
  - Neuron instance (trn2.3xlarge recommended, inf2.xlarge for small models)
  - Cloned dinov3 repository at /mnt/models/dinov3

Run:
    python -m pytest contrib/models/DINOv3/test/integration/test_model.py -v
    python contrib/models/DINOv3/test/integration/test_model.py  # standalone
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
import torch_neuronx

# Add src to path for modeling imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "src"))
sys.path.insert(0, _SRC_DIR)

from modeling_dinov3 import (
    COMPILER_ARGS_CONVNEXT,
    COMPILER_ARGS_VIT,
    MODEL_REGISTRY,
    benchmark_dataparallel,
    benchmark_model,
    load_dinov3_model,
    trace_dinov3,
    validate_accuracy,
)

# --- Constants ---

REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")
SAVED_DIR = os.environ.get(
    "DINOV3_SAVED_DIR",
    os.path.join(os.path.normpath(os.path.join(_THIS_DIR, "..", "..")), "saved_models"),
)
IMG_SIZE = 224

# Models to test (subset for fast CI; full set for manual runs)
# ViT-B and ConvNeXt-Tiny are representative of both architecture families
TEST_MODELS = {
    "vit_b": {"hub_name": "dinov3_vitb16", "is_convnext": False},
    "convnext_tiny": {"hub_name": "dinov3_convnext_tiny", "is_convnext": True},
}

# Extended model set for thorough testing
ALL_TRACE_MODELS = {
    "vit_s": {"hub_name": "dinov3_vits16", "is_convnext": False},
    "vit_b": {"hub_name": "dinov3_vitb16", "is_convnext": False},
    "vit_l": {"hub_name": "dinov3_vitl16", "is_convnext": False},
    "convnext_tiny": {"hub_name": "dinov3_convnext_tiny", "is_convnext": True},
    "convnext_base": {"hub_name": "dinov3_convnext_base", "is_convnext": True},
}

# Accuracy thresholds (validated: ViT achieves 1.000000, ConvNeXt achieves 0.999989)
COSINE_SIM_THRESHOLD_VIT = 0.9999
COSINE_SIM_THRESHOLD_CONVNEXT = 0.9998
MAX_DIFF_THRESHOLD = 0.05

# Performance thresholds (conservative, for trn2.3xlarge single core)
MIN_THROUGHPUT_VIT_B = 150  # img/s (achieved: 222)
MIN_THROUGHPUT_CONVNEXT_T = 100  # img/s (achieved: 183)
MIN_THROUGHPUT_DP4 = 300  # img/s, DP=4 for ViT-B (achieved: 439)
MAX_P50_LATENCY_MS = 15.0  # ms, single core


# --- Helpers ---


def get_neuron_core_count():
    """Detect number of NeuronCores available."""
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=10
        )
        info = json.loads(result.stdout)
        return sum(d["nc_count"] for d in info)
    except Exception:
        return 0


def compile_and_cache(cpu_model, model_key, config):
    """Compile a DINOv3 model for Neuron, tracing from the given CPU model.

    IMPORTANT: The cpu_model must be the same instance used for accuracy
    validation. Since pretrained=False gives different random weights on
    each call, we must trace the exact model we compare against.
    """
    os.makedirs(SAVED_DIR, exist_ok=True)
    save_path = os.path.join(SAVED_DIR, f"dinov3_{model_key}_bs1.pt")

    # Do NOT use cached NEFFs -- they were traced from a different model instance.
    # Always re-trace from the provided cpu_model to ensure weight consistency.
    model_neuron = trace_dinov3(
        cpu_model,
        is_convnext=config["is_convnext"],
        save_path=save_path,
    )
    return model_neuron


# --- Fixtures ---


@pytest.fixture(scope="module")
def n_cores():
    return get_neuron_core_count()


@pytest.fixture(scope="module")
def vit_b_cpu():
    return load_dinov3_model("dinov3_vitb16", repo_dir=REPO_DIR)


@pytest.fixture(scope="module")
def vit_b_neuron(vit_b_cpu):
    return compile_and_cache(vit_b_cpu, "vit_b", TEST_MODELS["vit_b"])


@pytest.fixture(scope="module")
def convnext_tiny_cpu():
    return load_dinov3_model("dinov3_convnext_tiny", repo_dir=REPO_DIR)


@pytest.fixture(scope="module")
def convnext_tiny_neuron(convnext_tiny_cpu):
    return compile_and_cache(
        convnext_tiny_cpu, "convnext_tiny", TEST_MODELS["convnext_tiny"]
    )


# --- Test Classes ---


class TestModelLoads:
    """Smoke tests: model loads, traces, and produces correct output shape."""

    def test_vit_b_loads(self, vit_b_neuron):
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        out = vit_b_neuron(example)
        # ViT-B output: CLS token embedding [1, 768]
        assert out.shape == (1, 768), f"Expected (1, 768), got {out.shape}"

    def test_convnext_tiny_loads(self, convnext_tiny_neuron):
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        out = convnext_tiny_neuron(example)
        # ConvNeXt-Tiny output shape depends on head configuration
        assert out.ndim >= 1, f"Expected at least 1D output, got {out.ndim}D"

    def test_output_is_finite(self, vit_b_neuron):
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        out = vit_b_neuron(example)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf values"

    def test_deterministic_output(self, vit_b_neuron):
        """Same input should produce identical output (no stochastic ops in eval)."""
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        out1 = vit_b_neuron(example)
        out2 = vit_b_neuron(example)
        assert torch.equal(out1, out2), "Model is not deterministic"


class TestAccuracy:
    """Accuracy tests: Neuron vs CPU cosine similarity."""

    def test_vit_b_cosine_similarity(self, vit_b_cpu, vit_b_neuron):
        metrics = validate_accuracy(vit_b_cpu, vit_b_neuron)
        assert metrics["cosine_sim"] >= COSINE_SIM_THRESHOLD_VIT, (
            f"ViT-B cosine sim {metrics['cosine_sim']:.6f} < {COSINE_SIM_THRESHOLD_VIT}"
        )

    def test_vit_b_max_diff(self, vit_b_cpu, vit_b_neuron):
        metrics = validate_accuracy(vit_b_cpu, vit_b_neuron)
        assert metrics["max_diff"] <= MAX_DIFF_THRESHOLD, (
            f"ViT-B max diff {metrics['max_diff']:.6f} > {MAX_DIFF_THRESHOLD}"
        )

    def test_convnext_tiny_cosine_similarity(
        self, convnext_tiny_cpu, convnext_tiny_neuron
    ):
        metrics = validate_accuracy(convnext_tiny_cpu, convnext_tiny_neuron)
        assert metrics["cosine_sim"] >= COSINE_SIM_THRESHOLD_CONVNEXT, (
            f"ConvNeXt-Tiny cosine sim {metrics['cosine_sim']:.6f} < {COSINE_SIM_THRESHOLD_CONVNEXT}"
        )

    def test_convnext_tiny_max_diff(self, convnext_tiny_cpu, convnext_tiny_neuron):
        metrics = validate_accuracy(convnext_tiny_cpu, convnext_tiny_neuron)
        assert metrics["max_diff"] <= MAX_DIFF_THRESHOLD, (
            f"ConvNeXt-Tiny max diff {metrics['max_diff']:.6f} > {MAX_DIFF_THRESHOLD}"
        )

    def test_vit_b_multiple_inputs(self, vit_b_cpu, vit_b_neuron):
        """Accuracy holds across multiple random inputs."""
        for i in range(5):
            torch.manual_seed(i * 42)
            metrics = validate_accuracy(vit_b_cpu, vit_b_neuron)
            assert metrics["cosine_sim"] >= COSINE_SIM_THRESHOLD_VIT, (
                f"ViT-B input {i}: cosine sim {metrics['cosine_sim']:.6f} < {COSINE_SIM_THRESHOLD_VIT}"
            )


class TestDataParallel:
    """DataParallel tests: verify multi-core scaling."""

    def test_dp_runs(self, vit_b_neuron, n_cores):
        if n_cores < 2:
            pytest.skip("Need >= 2 NeuronCores for DataParallel test")

        model_dp = torch_neuronx.DataParallel(
            vit_b_neuron,
            device_ids=list(range(min(n_cores, 4))),
            dim=0,
        )
        dp_cores = min(n_cores, 4)
        dp_input = torch.randn(dp_cores, 3, IMG_SIZE, IMG_SIZE)
        out = model_dp(dp_input)
        assert out.shape[0] == dp_cores, (
            f"Expected batch={dp_cores}, got {out.shape[0]}"
        )

    def test_dp_speedup(self, vit_b_neuron, n_cores):
        if n_cores < 4:
            pytest.skip("Need >= 4 NeuronCores for DP speedup test")

        # Single-core benchmark
        single_metrics = benchmark_model(vit_b_neuron, bench_iters=30)

        # DP=4 benchmark
        dp_results = benchmark_dataparallel(
            vit_b_neuron, num_cores=4, batch_sizes=[4], bench_iters=30
        )

        dp_throughput = dp_results[4]["throughput_img_s"]
        single_throughput = single_metrics["throughput_img_s"]
        speedup = dp_throughput / single_throughput

        assert speedup > 1.5, (
            f"DP=4 speedup {speedup:.2f}x < 1.5x "
            f"(single: {single_throughput:.1f}, DP: {dp_throughput:.1f} img/s)"
        )


class TestPerformance:
    """Performance tests: throughput and latency thresholds."""

    def test_vit_b_throughput(self, vit_b_neuron):
        metrics = benchmark_model(vit_b_neuron, bench_iters=30)
        assert metrics["throughput_img_s"] >= MIN_THROUGHPUT_VIT_B, (
            f"ViT-B throughput {metrics['throughput_img_s']:.1f} img/s < {MIN_THROUGHPUT_VIT_B}"
        )

    def test_convnext_tiny_throughput(self, convnext_tiny_neuron):
        metrics = benchmark_model(convnext_tiny_neuron, bench_iters=30)
        assert metrics["throughput_img_s"] >= MIN_THROUGHPUT_CONVNEXT_T, (
            f"ConvNeXt-Tiny throughput {metrics['throughput_img_s']:.1f} img/s < {MIN_THROUGHPUT_CONVNEXT_T}"
        )

    def test_vit_b_latency(self, vit_b_neuron):
        metrics = benchmark_model(vit_b_neuron, bench_iters=30)
        assert metrics["median_latency_ms"] <= MAX_P50_LATENCY_MS, (
            f"ViT-B P50 latency {metrics['median_latency_ms']:.2f}ms > {MAX_P50_LATENCY_MS}ms"
        )

    def test_dp4_throughput(self, vit_b_neuron, n_cores):
        if n_cores < 4:
            pytest.skip("Need >= 4 NeuronCores for DP throughput test")

        dp_results = benchmark_dataparallel(
            vit_b_neuron, num_cores=4, batch_sizes=[8], bench_iters=30
        )
        throughput = dp_results[8]["throughput_img_s"]
        assert throughput >= MIN_THROUGHPUT_DP4, (
            f"DP=4 throughput {throughput:.1f} img/s < {MIN_THROUGHPUT_DP4}"
        )


# --- Standalone Runner ---

if __name__ == "__main__":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    print("=" * 60)
    print("DINOv3 Neuron Integration Tests (standalone)")
    print("=" * 60)

    n_cores = get_neuron_core_count()
    print(f"\nNeuronCores detected: {n_cores}")
    print(f"DINOv3 repo: {REPO_DIR}")
    print(f"Save directory: {SAVED_DIR}")

    all_pass = True

    for model_key, config in TEST_MODELS.items():
        is_convnext = config["is_convnext"]
        arch = "ConvNeXt" if is_convnext else "ViT"
        threshold = (
            COSINE_SIM_THRESHOLD_CONVNEXT if is_convnext else COSINE_SIM_THRESHOLD_VIT
        )

        print(f"\n--- {model_key} ({arch}) ---")

        # 1. Load CPU model
        print(f"[1] Loading CPU model...")
        cpu_model = load_dinov3_model(config["hub_name"], repo_dir=REPO_DIR)
        n_params = sum(p.numel() for p in cpu_model.parameters()) / 1e6
        print(f"  Parameters: {n_params:.1f}M")

        # 2. CPU reference
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            cpu_out = cpu_model(example)
        print(f"[2] CPU output shape: {cpu_out.shape}")

        # 3. Compile (from the same cpu_model for weight consistency)
        print(f"[3] Compiling for Neuron...")
        neuron_model = compile_and_cache(cpu_model, model_key, config)

        # 4. Smoke test
        neuron_out = neuron_model(example)
        print(f"[4] Neuron output shape: {neuron_out.shape}")

        # 5. Accuracy
        metrics = validate_accuracy(cpu_model, neuron_model)
        status = "PASS" if metrics["cosine_sim"] >= threshold else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"[5] Accuracy [{status}]: cosine={metrics['cosine_sim']:.6f}, "
            f"max_diff={metrics['max_diff']:.6f}, l2_rel={metrics['l2_rel_error']:.6f}"
        )

        # 6. Performance (single core)
        perf = benchmark_model(neuron_model, bench_iters=50)
        print(
            f"[6] Performance: {perf['throughput_img_s']:.1f} img/s, "
            f"P50={perf['median_latency_ms']:.2f}ms, P99={perf['p99_latency_ms']:.2f}ms"
        )

        # 7. DataParallel (if enough cores)
        if n_cores >= 4:
            dp_results = benchmark_dataparallel(neuron_model, num_cores=4)
            print(f"[7] DataParallel (DP=4):")
            for bs, r in dp_results.items():
                print(
                    f"    BS={bs}: {r['throughput_img_s']:.1f} img/s, P50={r['median_latency_ms']:.2f}ms"
                )
        else:
            print(f"[7] DataParallel: SKIPPED (need >= 4 cores, have {n_cores})")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 60}")
