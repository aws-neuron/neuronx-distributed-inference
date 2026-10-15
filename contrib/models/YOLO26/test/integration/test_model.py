"""YOLO26 Neuron Integration Tests.

Self-contained integration tests for YOLO26 detection models on Neuron.
All helpers are local to this file (tests do not import from src/).

Usage
-----
With pytest::

    pytest test/integration/test_model.py -v

Standalone::

    python test/integration/test_model.py

Prerequisites
-------------
- Neuron instance (trn2.3xlarge or inf2)
- ``pip install ultralytics``
- YOLO26 weights auto-download on first run
"""

import json
import os
import subprocess
import time

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ["n", "s"]  # Test small variants for speed; m/l/x follow same pattern
VARIANT_DTYPES = {
    "n": torch.float32,
    "s": torch.float32,
    "m": torch.bfloat16,
    "l": torch.bfloat16,
    "x": torch.bfloat16,
}
INPUT_SHAPE = (3, 640, 640)
COSINE_SIM_THRESHOLD = 0.98
CACHE_DIR = "/tmp/yolo26_test_cache"
WARMUP_ITERS = 10
BENCHMARK_ITERS = 50


# ---------------------------------------------------------------------------
# Helpers (self-contained, no src/ imports)
# ---------------------------------------------------------------------------


def get_neuron_core_count() -> int:
    """Detect available NeuronCores via neuron-ls."""
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        return sum(dev.get("nc_count", 0) for dev in data)
    except Exception:
        return 0


def prepare_yolo26(weight_path: str, dtype: torch.dtype = torch.float32) -> nn.Module:
    """Prepare YOLO26 for tracing (same recipe as src/modeling_yolo26.py)."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import C2f

    model = YOLO(weight_path)
    pytorch_model = model.model.eval()

    detect = pytorch_model.model[-1]
    detect.end2end = False

    pytorch_model = pytorch_model.fuse(verbose=False)

    for m in pytorch_model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "dynamic"):
            m.dynamic = False
        if hasattr(m, "format"):
            m.format = "torchscript"
        if hasattr(m, "shape"):
            m.shape = None
        if isinstance(m, C2f):
            m.forward = m.forward_split

    if dtype != torch.float32:
        pytorch_model = pytorch_model.to(dtype)

    return pytorch_model


def compile_neuron_model(
    weight_path: str, batch_size: int = 1, dtype: torch.dtype = torch.float32
) -> torch.jit.ScriptModule:
    """Compile a YOLO26 model for Neuron, with caching."""
    import torch_neuronx

    variant = os.path.basename(weight_path).replace("yolo26", "").replace(".pt", "")
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp32"
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    cache_path = os.path.join(
        CACHE_DIR, f"yolo26{variant}_{dtype_name}_bs{batch_size}_lnc{lnc}.pt"
    )

    if os.path.exists(cache_path):
        return torch.jit.load(cache_path)

    model = prepare_yolo26(weight_path, dtype=dtype)
    dummy = torch.randn(batch_size, *INPUT_SHAPE, dtype=dtype)
    with torch.no_grad():
        _ = model(dummy)

    compiler_args = []
    if lnc == 1:
        compiler_args = ["--lnc", "1"]

    traced = torch_neuronx.trace(model, dummy, compiler_args=compiler_args)
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.jit.save(traced, cache_path)
    return traced


# ---------------------------------------------------------------------------
# Fixtures (module-scoped — compile once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request):
    return request.param


@pytest.fixture(scope="module")
def cpu_model_n():
    return prepare_yolo26("yolo26n.pt", dtype=torch.float32)


@pytest.fixture(scope="module")
def cpu_model_s():
    return prepare_yolo26("yolo26s.pt", dtype=torch.float32)


@pytest.fixture(scope="module")
def neuron_model_n():
    return compile_neuron_model("yolo26n.pt", batch_size=1, dtype=torch.float32)


@pytest.fixture(scope="module")
def neuron_model_s():
    return compile_neuron_model("yolo26s.pt", batch_size=1, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Tests: Model Loading & Compilation
# ---------------------------------------------------------------------------


class TestModelCompiles:
    """Smoke tests: models compile and produce output of expected shape."""

    def test_yolo26n_compiles(self, neuron_model_n):
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            out = neuron_model_n(dummy)
        assert out.shape == (1, 84, 8400), (
            f"Expected [1, 84, 8400], got {list(out.shape)}"
        )

    def test_yolo26s_compiles(self, neuron_model_s):
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            out = neuron_model_s(dummy)
        assert out.shape == (1, 84, 8400), (
            f"Expected [1, 84, 8400], got {list(out.shape)}"
        )

    def test_output_is_finite(self, neuron_model_n):
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            out = neuron_model_n(dummy)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Tests: Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    """Cosine similarity between CPU and Neuron outputs."""

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_yolo26n_cosine_similarity(self, cpu_model_n, neuron_model_n, seed):
        torch.manual_seed(seed)
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)

        with torch.no_grad():
            cpu_out = cpu_model_n(dummy)
            nrn_out = neuron_model_n(dummy)

        cossim = torch.nn.functional.cosine_similarity(
            cpu_out.flatten().float().unsqueeze(0),
            nrn_out.flatten().float().unsqueeze(0),
        ).item()

        assert cossim >= COSINE_SIM_THRESHOLD, (
            f"CosSim {cossim:.6f} below threshold {COSINE_SIM_THRESHOLD} (seed={seed})"
        )

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_yolo26s_cosine_similarity(self, cpu_model_s, neuron_model_s, seed):
        torch.manual_seed(seed)
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)

        with torch.no_grad():
            cpu_out = cpu_model_s(dummy)
            nrn_out = neuron_model_s(dummy)

        cossim = torch.nn.functional.cosine_similarity(
            cpu_out.flatten().float().unsqueeze(0),
            nrn_out.flatten().float().unsqueeze(0),
        ).item()

        assert cossim >= COSINE_SIM_THRESHOLD, (
            f"CosSim {cossim:.6f} below threshold {COSINE_SIM_THRESHOLD} (seed={seed})"
        )


# ---------------------------------------------------------------------------
# Tests: Data Parallel
# ---------------------------------------------------------------------------


class TestDataParallel:
    """Multi-core data parallel execution."""

    def test_dp_runs(self, neuron_model_n):
        import torch_neuronx

        num_cores = get_neuron_core_count()
        if num_cores < 2:
            pytest.skip("Only 1 NeuronCore available")

        model_dp = torch_neuronx.DataParallel(
            neuron_model_n,
            device_ids=list(range(num_cores)),
            dim=0,
        )

        total_bs = num_cores  # BS=1 per core
        dummy = torch.randn(total_bs, *INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            out = model_dp(dummy)

        assert out.shape[0] == total_bs, (
            f"Expected batch dim {total_bs}, got {out.shape[0]}"
        )
        assert out.shape[1:] == (84, 8400)

    def test_dp_speedup(self, neuron_model_n):
        import torch_neuronx

        num_cores = get_neuron_core_count()
        if num_cores < 2:
            pytest.skip("Only 1 NeuronCore available")

        # Single core baseline
        dummy_1 = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                neuron_model_n(dummy_1)

        single_lats = []
        for _ in range(BENCHMARK_ITERS):
            t0 = time.time()
            with torch.no_grad():
                neuron_model_n(dummy_1)
            single_lats.append(time.time() - t0)
        single_tput = 1.0 / np.median(single_lats)

        # Multi-core
        model_dp = torch_neuronx.DataParallel(
            neuron_model_n,
            device_ids=list(range(num_cores)),
            dim=0,
        )
        total_bs = num_cores
        dummy_dp = torch.randn(total_bs, *INPUT_SHAPE, dtype=torch.float32)

        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                model_dp(dummy_dp)

        dp_lats = []
        for _ in range(BENCHMARK_ITERS):
            t0 = time.time()
            with torch.no_grad():
                model_dp(dummy_dp)
            dp_lats.append(time.time() - t0)
        dp_tput = total_bs / np.median(dp_lats)

        speedup = dp_tput / single_tput
        assert speedup > 1.5, (
            f"DP speedup {speedup:.2f}x too low (expected >1.5x with {num_cores} cores)"
        )


# ---------------------------------------------------------------------------
# Tests: Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Throughput thresholds for detection variants."""

    def test_yolo26n_throughput(self, neuron_model_n):
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)

        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                neuron_model_n(dummy)

        latencies = []
        for _ in range(BENCHMARK_ITERS):
            t0 = time.time()
            with torch.no_grad():
                neuron_model_n(dummy)
            latencies.append((time.time() - t0) * 1000)

        p50 = np.median(latencies)
        throughput = 1000.0 / p50
        # Single core yolo26n should be at least 25 img/s
        assert throughput >= 25.0, (
            f"Throughput {throughput:.1f} img/s below minimum 25.0 (p50={p50:.1f}ms)"
        )

    def test_yolo26s_throughput(self, neuron_model_s):
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32)

        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                neuron_model_s(dummy)

        latencies = []
        for _ in range(BENCHMARK_ITERS):
            t0 = time.time()
            with torch.no_grad():
                neuron_model_s(dummy)
            latencies.append((time.time() - t0) * 1000)

        p50 = np.median(latencies)
        throughput = 1000.0 / p50
        # Single core yolo26s should be at least 50 img/s
        assert throughput >= 50.0, (
            f"Throughput {throughput:.1f} img/s below minimum 50.0 (p50={p50:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO26 Neuron Integration Tests")
    print("=" * 60)

    num_cores = get_neuron_core_count()
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")
    print(f"NeuronCores: {num_cores}, LNC: {lnc}")

    # [1/6] Compile n
    print("\n[1/6] Compiling yolo26n (FP32, BS=1)...")
    t0 = time.time()
    neuron_n = compile_neuron_model("yolo26n.pt", batch_size=1, dtype=torch.float32)
    print(f"  Done in {time.time() - t0:.1f}s")

    # [2/6] Compile s
    print("\n[2/6] Compiling yolo26s (FP32, BS=1)...")
    t0 = time.time()
    neuron_s = compile_neuron_model("yolo26s.pt", batch_size=1, dtype=torch.float32)
    print(f"  Done in {time.time() - t0:.1f}s")

    # [3/6] Shape test
    print("\n[3/6] Testing output shapes...")
    for name, model, dtype in [
        ("n", neuron_n, torch.float32),
        ("s", neuron_s, torch.float32),
    ]:
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=dtype)
        with torch.no_grad():
            out = model(dummy)
        print(f"  yolo26{name}: {list(out.shape)}, NaN={torch.isnan(out).any().item()}")
        assert out.shape == (1, 84, 8400)

    # [4/6] Accuracy
    print("\n[4/6] Accuracy validation...")
    for name, neuron_model, dtype in [
        ("n", neuron_n, torch.float32),
        ("s", neuron_s, torch.float32),
    ]:
        cpu_model = prepare_yolo26(f"yolo26{name}.pt", dtype=dtype)
        torch.manual_seed(42)
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=dtype)
        with torch.no_grad():
            cpu_out = cpu_model(dummy)
            nrn_out = neuron_model(dummy)
        cossim = torch.nn.functional.cosine_similarity(
            cpu_out.flatten().float().unsqueeze(0),
            nrn_out.flatten().float().unsqueeze(0),
        ).item()
        print(
            f"  yolo26{name}: CosSim = {cossim:.6f} {'PASS' if cossim >= COSINE_SIM_THRESHOLD else 'FAIL'}"
        )

    # [5/6] Throughput
    print("\n[5/6] Throughput benchmark (single core)...")
    for name, model, dtype in [
        ("n", neuron_n, torch.float32),
        ("s", neuron_s, torch.float32),
    ]:
        dummy = torch.randn(1, *INPUT_SHAPE, dtype=dtype)
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                model(dummy)
        lats = []
        for _ in range(BENCHMARK_ITERS):
            t0 = time.time()
            with torch.no_grad():
                model(dummy)
            lats.append((time.time() - t0) * 1000)
        p50 = np.median(lats)
        print(f"  yolo26{name}: {1000 / p50:.1f} img/s (p50={p50:.1f}ms)")

    # [6/6] Data Parallel
    print("\n[6/6] Data Parallel test...")
    if num_cores >= 2:
        import torch_neuronx

        model_dp = torch_neuronx.DataParallel(
            neuron_n, device_ids=list(range(num_cores)), dim=0
        )
        dummy_dp = torch.randn(num_cores, *INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            out_dp = model_dp(dummy_dp)
        print(f"  DP={num_cores}: output shape {list(out_dp.shape)}")
    else:
        print("  Skipped (only 1 NeuronCore)")

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
