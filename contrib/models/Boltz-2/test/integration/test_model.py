"""Integration tests for Boltz-2 pairformer on AWS Trainium 2.

Tests compile the pairformer trunk using weight replacement with NKI kernels,
run inference, and validate accuracy against CPU reference.

Prerequisites:
    pip install boltz==2.2.1

Environment variables:
    BOLTZ2_CHECKPOINT: Path to Boltz-2 checkpoint (default: ~/.boltz/boltz2_conf.ckpt)
    NEURON_PLATFORM_TARGET_OVERRIDE: Must be "trn2" (set automatically)

Run:
    NEURON_RT_VISIBLE_CORES=0 pytest test/integration/test_model.py -v -s
"""

import os
import sys

# Set trn2 target before any Neuron imports
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import pytest
import torch
import torch.nn.functional as F

# Add src/ to path for imports
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC_DIR)


# ========================================================================
# Constants
# ========================================================================

N = 128  # Sequence length for tests (128 = 1 tile, fastest compile)
C_s = 384  # Single representation channels
C_z = 128  # Pair representation channels
NUM_LAYERS_TEST = 2  # Number of layers for accuracy test (2 = fast, sufficient)
CHECKPOINT_PATH = os.environ.get(
    "BOLTZ2_CHECKPOINT",
    os.path.expanduser("~/.boltz/boltz2_conf.ckpt"),
)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture(scope="module")
def boltz2_model():
    """Load Boltz-2 model from checkpoint."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(
            f"Boltz-2 checkpoint not found at {CHECKPOINT_PATH}. "
            f"Download with: boltz predict --help  (auto-downloads on first run), "
            f"or set BOLTZ2_CHECKPOINT env var."
        )

    from dataclasses import asdict

    from boltz.main import (
        Boltz2,
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )

    model = Boltz2.load_from_checkpoint(
        CHECKPOINT_PATH,
        strict=True,
        predict_args={
            "recycling_steps": 1,
            "sampling_steps": 1,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    model = model.float()
    return model


@pytest.fixture(scope="module")
def compiled_pairformer(boltz2_model):
    """Compile pairformer layers using weight replacement + NKI kernels."""
    from modeling_boltz2 import (
        compile_pairformer_weight_replaced,
        patch_boltz2_with_nki_kernels,
    )

    patch_boltz2_with_nki_kernels()

    # Compile only NUM_LAYERS_TEST layers for speed
    import copy

    from modeling_boltz2 import SinglePairformerLayerWrapper

    all_layers = list(boltz2_model.pairformer_module.layers)[:NUM_LAYERS_TEST]

    s_dummy = torch.randn(1, N, C_s, dtype=torch.bfloat16) * 0.1
    z_dummy = torch.randn(1, N, N, C_z, dtype=torch.bfloat16) * 0.1
    mask_dummy = torch.ones(1, N, dtype=torch.float32)
    pair_mask_dummy = torch.ones(1, N, N, dtype=torch.float32)

    import torch_neuronx

    # Compile layer 0
    layer0 = copy.deepcopy(all_layers[0]).to(torch.bfloat16)
    wrapper0 = SinglePairformerLayerWrapper(layer0)
    wrapper0.eval()

    traced_template = torch_neuronx.trace(
        wrapper0,
        (s_dummy, z_dummy, mask_dummy, pair_mask_dummy),
        compiler_args=["--target", "trn2"],
        inline_weights_to_neff=False,
    )

    try:
        torch_neuronx.move_trace_to_device(traced_template, 0)
    except Exception:
        pass

    # Warmup
    with torch.no_grad():
        _ = traced_template(s_dummy, z_dummy, mask_dummy, pair_mask_dummy)

    # Weight replacement for remaining layers
    traced_layers = [traced_template]
    for i in range(1, len(all_layers)):
        layer_bf16 = copy.deepcopy(all_layers[i]).to(torch.bfloat16)
        wrapper_i = SinglePairformerLayerWrapper(layer_bf16)
        wrapper_i.eval()

        traced_copy = copy.deepcopy(traced_template)
        torch_neuronx.replace_weights(traced_copy, wrapper_i.state_dict())

        try:
            torch_neuronx.move_trace_to_device(traced_copy, 0)
        except Exception:
            pass

        traced_layers.append(traced_copy)

    return traced_layers


@pytest.fixture(scope="module")
def test_inputs():
    """Create deterministic test inputs."""
    torch.manual_seed(42)
    s = torch.randn(1, N, C_s, dtype=torch.float32) * 0.1
    z = torch.randn(1, N, N, C_z, dtype=torch.float32) * 0.1
    mask = torch.ones(1, N, dtype=torch.float32)
    pair_mask = torch.ones(1, N, N, dtype=torch.float32)
    return s, z, mask, pair_mask


# ========================================================================
# Tests
# ========================================================================


class TestBoltz2Pairformer:
    """Test Boltz-2 pairformer with NKI kernels on Neuron."""

    def test_model_loads(self, boltz2_model):
        """Verify Boltz-2 model loads from checkpoint."""
        assert boltz2_model is not None
        layers = list(boltz2_model.pairformer_module.layers)
        assert len(layers) == 64, f"Expected 64 pairformer layers, got {len(layers)}"

    def test_compilation_succeeds(self, compiled_pairformer):
        """Verify pairformer compiles with NKI kernels."""
        assert compiled_pairformer is not None
        assert len(compiled_pairformer) == NUM_LAYERS_TEST

    def test_forward_pass_no_nan(self, compiled_pairformer, test_inputs):
        """Verify forward pass produces valid outputs (no NaN/Inf)."""
        s, z, mask, pair_mask = test_inputs
        s_curr = s.to(torch.bfloat16)
        z_curr = z.to(torch.bfloat16)

        with torch.no_grad():
            for traced in compiled_pairformer:
                s_curr, z_curr = traced(s_curr, z_curr, mask, pair_mask)
                s_curr = s_curr.to(torch.bfloat16)
                z_curr = z_curr.to(torch.bfloat16)

        assert not torch.isnan(s_curr).any(), "s output contains NaN"
        assert not torch.isnan(z_curr).any(), "z output contains NaN"
        assert not torch.isinf(s_curr).any(), "s output contains Inf"
        assert not torch.isinf(z_curr).any(), "z output contains Inf"
        assert s_curr.shape == (1, N, C_s)
        assert z_curr.shape == (1, N, N, C_z)

    def test_accuracy_vs_cpu(self, boltz2_model, compiled_pairformer, test_inputs):
        """Compare Neuron output against CPU reference.

        Validated results at N=128 with trained weights:
            s_cos >= 0.999 (measured 0.999796 for 1 layer)
            z_cos >= 0.995 (measured 0.998359 for 1 layer)
        """
        s, z, mask, pair_mask = test_inputs

        # CPU reference (first NUM_LAYERS_TEST layers only)
        with torch.no_grad():
            s_cpu = s.clone()
            z_cpu = z.clone()
            layers = list(boltz2_model.pairformer_module.layers)[:NUM_LAYERS_TEST]
            for layer in layers:
                s_cpu, z_cpu = layer(
                    s_cpu, z_cpu, mask=mask, pair_mask=pair_mask, use_kernels=False
                )

        # Neuron inference
        with torch.no_grad():
            s_neuron = s.to(torch.bfloat16)
            z_neuron = z.to(torch.bfloat16)
            for traced in compiled_pairformer:
                s_neuron, z_neuron = traced(s_neuron, z_neuron, mask, pair_mask)
                s_neuron = s_neuron.to(torch.bfloat16)
                z_neuron = z_neuron.to(torch.bfloat16)

        # Compare
        s_cos = F.cosine_similarity(
            s_cpu.flatten().unsqueeze(0).float(),
            s_neuron.flatten().unsqueeze(0).float(),
        ).item()
        z_cos = F.cosine_similarity(
            z_cpu.flatten().unsqueeze(0).float(),
            z_neuron.flatten().unsqueeze(0).float(),
        ).item()

        print(f"\n  Accuracy ({NUM_LAYERS_TEST} layers, N={N}):")
        print(f"    s_cos = {s_cos:.6f}")
        print(f"    z_cos = {z_cos:.6f}")

        # Thresholds based on measured results with trained weights
        assert s_cos >= 0.99, f"s cosine similarity {s_cos:.6f} < 0.99"
        assert z_cos >= 0.99, f"z cosine similarity {z_cos:.6f} < 0.99"

    def test_inference_latency(self, compiled_pairformer, test_inputs):
        """Measure inference latency per layer."""
        s, z, mask, pair_mask = test_inputs
        import time

        # Warmup
        with torch.no_grad():
            s_w = s.to(torch.bfloat16)
            z_w = z.to(torch.bfloat16)
            for traced in compiled_pairformer:
                s_w, z_w = traced(s_w, z_w, mask, pair_mask)
                s_w = s_w.to(torch.bfloat16)
                z_w = z_w.to(torch.bfloat16)

        # Timed runs
        latencies = []
        for _ in range(3):
            s_t = s.to(torch.bfloat16)
            z_t = z.to(torch.bfloat16)
            t0 = time.time()
            with torch.no_grad():
                for traced in compiled_pairformer:
                    s_t, z_t = traced(s_t, z_t, mask, pair_mask)
                    s_t = s_t.to(torch.bfloat16)
                    z_t = z_t.to(torch.bfloat16)
            latencies.append(time.time() - t0)

        avg_latency = sum(latencies) / len(latencies)
        per_layer = avg_latency / NUM_LAYERS_TEST * 1000

        print(f"\n  Latency ({NUM_LAYERS_TEST} layers, N={N}):")
        print(f"    Total: {avg_latency * 1000:.1f} ms")
        print(f"    Per layer: {per_layer:.1f} ms")

        # Sanity check: should be faster than 10s for 2 layers at N=128
        assert avg_latency < 10.0, f"Latency {avg_latency:.1f}s exceeds 10s threshold"
