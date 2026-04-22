"""
Integration tests for LaughterSegmentation on Neuron.

Tests compile and run the Wav2Vec2-based laughter detection model using
torch_neuronx.trace() on Inferentia2 / Trainium2. Validates accuracy by
comparing Neuron output against CPU reference using cosine similarity.

Usage:
    # Run with pytest
    pytest test_model.py --capture=tee-sys -v

    # Run standalone
    python test_model.py

Prerequisites:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    pip install safetensors
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch
import torch.nn.utils.parametrize as parametrize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "omine-me/LaughterSegmentation"
BASE_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
MODEL_DIR = "/home/ubuntu/models/LaughterSegmentation"
COMPILED_DIR = "/home/ubuntu/neuron_models/LaughterSegmentation"

INPUT_SEC = 7
SAMPLE_RATE = 16000
INPUT_SAMPLES = INPUT_SEC * SAMPLE_RATE  # 112,000
BATCH_SIZE = 1

COMPILER_ARGS = [
    "--model-type",
    "transformer",
    "--optlevel",
    "2",
    "--auto-cast",
    "matmult",
]

# Accuracy thresholds
COSINE_SIM_THRESHOLD = 0.999
FRAME_AGREEMENT_THRESHOLD = 0.99

# Performance thresholds
THROUGHPUT_THRESHOLD = 40.0  # windows/sec minimum on single core
LATENCY_THRESHOLD_MS = 50.0  # max p50 latency at BS=1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_weights():
    """Download model weights from HuggingFace if not present."""
    weights_path = os.path.join(MODEL_DIR, "model.safetensors")
    if not os.path.exists(weights_path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        subprocess.check_call(
            ["hf", "download", MODEL_ID, "model.safetensors", "--local-dir", MODEL_DIR]
        )
    return weights_path


def load_cpu_model():
    """Load the LaughterSegmentation model on CPU with parametrizations removed."""
    import safetensors.torch
    from transformers import Wav2Vec2Config, Wav2Vec2ForAudioFrameClassification

    weights_path = download_weights()

    config = Wav2Vec2Config.from_pretrained(BASE_MODEL_ID)
    config.num_labels = 1
    config.problem_type = "single_label_classification"
    model = Wav2Vec2ForAudioFrameClassification(config)

    # Load weights, stripping "audio_model." prefix
    state_dict = safetensors.torch.load_file(weights_path, device="cpu")
    prefix = "audio_model."
    stripped = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(stripped)
    model.eval()

    # Remove weight_norm parametrizations (required for SDK 2.28+)
    for name, module in model.named_modules():
        if hasattr(module, "parametrizations"):
            for param_name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, param_name)

    return model


def compile_neuron_model(cpu_model, batch_size=BATCH_SIZE):
    """Compile the model for Neuron and cache the result."""
    import torch_neuronx

    save_path = os.path.join(COMPILED_DIR, f"laughter_bs{batch_size}.pt")

    if os.path.exists(save_path):
        return torch.jit.load(save_path)

    os.makedirs(COMPILED_DIR, exist_ok=True)
    example_input = torch.randn(batch_size, INPUT_SAMPLES)

    model_neuron = torch_neuronx.trace(
        cpu_model,
        example_input,
        compiler_args=COMPILER_ARGS,
        inline_weights_to_neff=True,
    )

    torch.jit.save(model_neuron, save_path)
    return model_neuron


def get_neuron_core_count():
    """Detect available NeuronCores via neuron-ls."""
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            devices = json.loads(result.stdout)
            return sum(d["nc_count"] for d in devices)
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------
# Fixtures (module-scoped so compile happens once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_model():
    """Load the CPU reference model."""
    return load_cpu_model()


@pytest.fixture(scope="module")
def neuron_model(cpu_model):
    """Compile and load the Neuron model (BS=1)."""
    return compile_neuron_model(cpu_model, batch_size=BATCH_SIZE)


@pytest.fixture(scope="module")
def neuron_model_bs2(cpu_model):
    """Compile and load the Neuron model (BS=2) for DataParallel."""
    return compile_neuron_model(cpu_model, batch_size=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelLoads:
    """Smoke tests that the model compiles and loads on Neuron."""

    def test_neuron_model_loads(self, neuron_model):
        """Model compiles and loads successfully."""
        assert neuron_model is not None

    def test_neuron_model_runs(self, neuron_model):
        """Model produces output of expected shape."""
        x = torch.randn(BATCH_SIZE, INPUT_SAMPLES)
        out = neuron_model(x)
        # Output is a tensor of shape [batch, 349, 1] or [batch, 349]
        if isinstance(out, dict):
            logits = out["logits"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        assert logits.shape[0] == BATCH_SIZE
        assert logits.shape[1] == 349  # expected frame count for 7s at 16kHz


class TestAccuracy:
    """Validate Neuron output against CPU reference."""

    @pytest.mark.parametrize(
        "input_name,input_fn",
        [
            ("random_normal", lambda: torch.randn(1, INPUT_SAMPLES)),
            ("quiet_noise", lambda: torch.randn(1, INPUT_SAMPLES) * 0.01),
            ("loud_signal", lambda: torch.randn(1, INPUT_SAMPLES) * 5.0),
            (
                "sine_440hz",
                lambda: torch.sin(
                    2 * torch.pi * 440 * torch.linspace(0, INPUT_SEC, INPUT_SAMPLES)
                ).unsqueeze(0),
            ),
            ("silence", lambda: torch.zeros(1, INPUT_SAMPLES)),
        ],
    )
    def test_cosine_similarity(self, cpu_model, neuron_model, input_name, input_fn):
        """Cosine similarity between CPU and Neuron logits exceeds threshold."""
        torch.manual_seed(42)
        x = input_fn()

        # CPU reference
        with torch.no_grad():
            cpu_out = cpu_model(input_values=x)
        cpu_logits = cpu_out.logits.squeeze(-1).flatten().float()

        # Neuron
        neuron_out = neuron_model(x)
        if isinstance(neuron_out, dict):
            neuron_logits = neuron_out["logits"]
        elif isinstance(neuron_out, (tuple, list)):
            neuron_logits = neuron_out[0]
        else:
            neuron_logits = neuron_out
        neuron_logits = neuron_logits.squeeze(-1).flatten().float()

        cosine = torch.nn.functional.cosine_similarity(
            cpu_logits.unsqueeze(0), neuron_logits.unsqueeze(0)
        ).item()

        print(f"  {input_name}: cosine_sim={cosine:.6f}")
        assert cosine >= COSINE_SIM_THRESHOLD, (
            f"Cosine similarity {cosine:.6f} below threshold {COSINE_SIM_THRESHOLD}"
        )

    def test_frame_agreement(self, cpu_model, neuron_model):
        """Frame-level prediction agreement on random input."""
        torch.manual_seed(42)
        x = torch.randn(1, INPUT_SAMPLES)

        with torch.no_grad():
            cpu_out = cpu_model(input_values=x)
        cpu_preds = (torch.sigmoid(cpu_out.logits.squeeze(-1)) >= 0.5).int()

        neuron_out = neuron_model(x)
        if isinstance(neuron_out, dict):
            neuron_logits = neuron_out["logits"]
        elif isinstance(neuron_out, (tuple, list)):
            neuron_logits = neuron_out[0]
        else:
            neuron_logits = neuron_out
        neuron_preds = (torch.sigmoid(neuron_logits.squeeze(-1)) >= 0.5).int()

        agreement = (cpu_preds == neuron_preds).float().mean().item()
        print(f"  Frame agreement: {agreement * 100:.2f}%")
        assert agreement >= FRAME_AGREEMENT_THRESHOLD, (
            f"Frame agreement {agreement:.4f} below threshold {FRAME_AGREEMENT_THRESHOLD}"
        )


class TestDataParallel:
    """Test DataParallel for full-instance throughput."""

    def test_data_parallel_runs(self, neuron_model_bs2):
        """DataParallel loads and produces correct output shape."""
        import torch_neuronx

        num_cores = get_neuron_core_count()
        if num_cores < 2:
            pytest.skip("Only 1 NeuronCore available, skipping DataParallel test")

        model_dp = torch_neuronx.DataParallel(neuron_model_bs2)
        model_dp.num_workers = num_cores

        dp_total_batch = 2 * num_cores
        x = torch.randn(dp_total_batch, INPUT_SAMPLES)

        out = model_dp(x)
        if isinstance(out, dict):
            logits = out["logits"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        assert logits.shape[0] == dp_total_batch
        assert logits.shape[1] == 349
        print(f"  DataParallel OK: {num_cores} cores, output shape {logits.shape}")

    def test_data_parallel_speedup(self, neuron_model_bs2):
        """DataParallel achieves meaningful speedup over single core."""
        import torch_neuronx
        import numpy as np

        num_cores = get_neuron_core_count()
        if num_cores < 2:
            pytest.skip("Only 1 NeuronCore available, skipping DataParallel test")

        # Single core baseline
        x_single = torch.randn(2, INPUT_SAMPLES)
        for _ in range(5):
            neuron_model_bs2(x_single)
        single_times = []
        for _ in range(20):
            t0 = time.time()
            neuron_model_bs2(x_single)
            single_times.append(time.time() - t0)
        single_throughput = 2 / np.mean(single_times)

        # DataParallel
        model_dp = torch_neuronx.DataParallel(neuron_model_bs2)
        model_dp.num_workers = num_cores
        dp_total_batch = 2 * num_cores
        x_dp = torch.randn(dp_total_batch, INPUT_SAMPLES)
        for _ in range(5):
            model_dp(x_dp)
        dp_times = []
        for _ in range(20):
            t0 = time.time()
            model_dp(x_dp)
            dp_times.append(time.time() - t0)
        dp_throughput = dp_total_batch / np.mean(dp_times)

        speedup = dp_throughput / single_throughput
        print(
            f"  Single: {single_throughput:.1f} W/s, DP: {dp_throughput:.1f} W/s, Speedup: {speedup:.2f}x"
        )
        assert speedup > 1.3, (
            f"DataParallel speedup {speedup:.2f}x too low (expected >1.3x)"
        )


class TestPerformance:
    """Benchmark throughput and latency."""

    def test_throughput(self, neuron_model):
        """Single-core throughput exceeds minimum threshold."""
        import numpy as np

        x = torch.randn(BATCH_SIZE, INPUT_SAMPLES)

        # Warmup
        for _ in range(10):
            neuron_model(x)

        # Benchmark
        latencies = []
        for _ in range(50):
            t0 = time.time()
            neuron_model(x)
            latencies.append((time.time() - t0) * 1000)

        lat = np.array(latencies)
        throughput = BATCH_SIZE / (lat.mean() / 1000)
        p50 = np.percentile(lat, 50)

        print(f"  Throughput: {throughput:.1f} W/s, p50: {p50:.2f} ms")
        assert throughput >= THROUGHPUT_THRESHOLD, (
            f"Throughput {throughput:.1f} below threshold {THROUGHPUT_THRESHOLD}"
        )

    def test_latency(self, neuron_model):
        """p50 latency is below threshold at BS=1."""
        import numpy as np

        x = torch.randn(BATCH_SIZE, INPUT_SAMPLES)

        for _ in range(10):
            neuron_model(x)

        latencies = []
        for _ in range(50):
            t0 = time.time()
            neuron_model(x)
            latencies.append((time.time() - t0) * 1000)

        p50 = np.percentile(latencies, 50)
        print(f"  p50 latency: {p50:.2f} ms")
        assert p50 <= LATENCY_THRESHOLD_MS, (
            f"p50 latency {p50:.2f} ms exceeds threshold {LATENCY_THRESHOLD_MS} ms"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LaughterSegmentation Integration Tests")
    print("=" * 60)

    print("\n[1/7] Loading CPU model...")
    model_cpu = load_cpu_model()
    print(
        f"  Model loaded: {sum(p.numel() for p in model_cpu.parameters()) / 1e6:.1f}M params"
    )

    print("\n[2/7] Compiling for Neuron (BS=1)...")
    model_neuron = compile_neuron_model(model_cpu, batch_size=1)
    print("  Compile OK")

    print("\n[3/7] Testing model runs...")
    x = torch.randn(1, INPUT_SAMPLES)
    out = model_neuron(x)
    logits = (
        out[0]
        if isinstance(out, (tuple, list))
        else (out["logits"] if isinstance(out, dict) else out)
    )
    print(f"  Output shape: {logits.shape}")

    print("\n[4/7] Accuracy validation (cosine similarity)...")
    torch.manual_seed(42)
    test_inputs = {
        "random_normal": torch.randn(1, INPUT_SAMPLES),
        "quiet_noise": torch.randn(1, INPUT_SAMPLES) * 0.01,
        "loud_signal": torch.randn(1, INPUT_SAMPLES) * 5.0,
        "silence": torch.zeros(1, INPUT_SAMPLES),
    }
    all_pass = True
    for name, inp in test_inputs.items():
        with torch.no_grad():
            cpu_out = model_cpu(input_values=inp)
        cpu_logits = cpu_out.logits.squeeze(-1).flatten().float()
        nrn_out = model_neuron(inp)
        nrn_logits = (
            nrn_out[0]
            if isinstance(nrn_out, (tuple, list))
            else (nrn_out["logits"] if isinstance(nrn_out, dict) else nrn_out)
        )
        nrn_logits = nrn_logits.squeeze(-1).flatten().float()
        cosine = torch.nn.functional.cosine_similarity(
            cpu_logits.unsqueeze(0), nrn_logits.unsqueeze(0)
        ).item()
        status = "PASS" if cosine >= COSINE_SIM_THRESHOLD else "FAIL"
        if cosine < COSINE_SIM_THRESHOLD:
            all_pass = False
        print(f"  {name}: cosine={cosine:.6f} [{status}]")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    print("\n[5/7] Performance benchmark (BS=1)...")
    import numpy as np

    x = torch.randn(1, INPUT_SAMPLES)
    for _ in range(10):
        model_neuron(x)
    latencies = []
    for _ in range(50):
        t0 = time.time()
        model_neuron(x)
        latencies.append((time.time() - t0) * 1000)
    lat = np.array(latencies)
    print(f"  Throughput: {1 / (lat.mean() / 1000):.1f} W/s")
    print(
        f"  p50: {np.percentile(lat, 50):.2f} ms, p99: {np.percentile(lat, 99):.2f} ms"
    )

    print("\n[6/7] Compiling BS=2 for DataParallel...")
    model_bs2 = compile_neuron_model(model_cpu, batch_size=2)
    print("  Compile OK")

    print("\n[7/7] DataParallel test...")
    num_cores = get_neuron_core_count()
    if num_cores >= 2:
        import torch_neuronx

        model_dp = torch_neuronx.DataParallel(model_bs2)
        model_dp.num_workers = num_cores
        dp_batch = 2 * num_cores
        x_dp = torch.randn(dp_batch, INPUT_SAMPLES)
        for _ in range(5):
            model_dp(x_dp)
        dp_times = []
        for _ in range(20):
            t0 = time.time()
            model_dp(x_dp)
            dp_times.append(time.time() - t0)
        dp_throughput = dp_batch / np.mean(dp_times)
        print(f"  DataParallel ({num_cores} cores): {dp_throughput:.1f} W/s")
    else:
        print("  Skipped: only 1 NeuronCore")

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
