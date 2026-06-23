"""Integration test for Evo2-7B on Neuron.

Tests NeuronFFT accuracy, prefill compilation, and decode block compilation
against CPU reference. Requires a trn2.3xlarge instance with Neuron SDK 2.28,
evo2 and vortex repositories installed.

Usage:
    # On trn2.3xlarge with Neuron SDK 2.28
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

    # Set paths
    export EVO2_SRC=/home/ubuntu/evo2-repo
    export VORTEX_SRC=/home/ubuntu/vortex-repo
    export EVO2_WEIGHTS=~/.cache/huggingface/hub/models--arcinstitute--evo2_7b_base/snapshots/.../evo2_7b_base.pt
    export NEURON_RT_VISIBLE_CORES=0

    cd contrib/models/Evo2-7B
    PYTHONPATH=src:$EVO2_SRC:$VORTEX_SRC:$PYTHONPATH \
        pytest test/integration/test_model.py -v -s
"""

import os
import sys
import gc
import math
import time

import pytest
import torch
import torch.nn as nn

# Check for Neuron hardware (only needed for prefill/decode tests, not FFT CPU tests)
try:
    import torch_neuronx

    HAS_NEURONX = True
except ImportError:
    HAS_NEURONX = False

requires_neuron = pytest.mark.skipif(
    not HAS_NEURONX, reason="torch_neuronx not available (requires Neuron hardware)"
)


# ============================================================================
# Constants
# ============================================================================

EVO2_SRC = os.environ.get("EVO2_SRC", "/home/ubuntu/evo2-repo")
VORTEX_SRC = os.environ.get("VORTEX_SRC", "/home/ubuntu/vortex-repo")
EVO2_WEIGHTS = os.environ.get("EVO2_WEIGHTS", None)

# Tolerances
COSINE_SIM_THRESHOLD = 0.999  # CPU vs Neuron cosine similarity
ATOL_PREFILL = 1.0  # Absolute tolerance for prefill logits
RTOL_PREFILL = 0.05

# Test sequence length (128 for faster compilation in tests)
TEST_SEQ_LEN = 128
TEST_HIDDEN_SIZE = 4096

COMPILER_ARGS = ["--auto-cast", "matmult", "--model-type=transformer"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def model_and_config():
    """Load Evo2 model (shared across all tests in module)."""
    # Add repos to path
    if EVO2_SRC not in sys.path:
        sys.path.insert(0, EVO2_SRC)
    if VORTEX_SRC not in sys.path:
        sys.path.insert(0, VORTEX_SRC)

    from modeling_evo2 import (
        _install_cuda_mocks,
        patch_vortex_for_neuron,
        load_config,
        build_model,
        install_neuron_fft_patch,
    )

    _install_cuda_mocks()
    patch_vortex_for_neuron()
    config = load_config()
    model, config = build_model(config, EVO2_WEIGHTS)
    install_neuron_fft_patch(model, TEST_SEQ_LEN)
    yield model, config
    del model
    gc.collect()


# ============================================================================
# NeuronFFT Tests (CPU-only, no Neuron hardware needed for these)
# ============================================================================


class TestNeuronFFT:
    """Test NeuronFFT accuracy against torch.fft."""

    def test_fft_accuracy_small(self):
        """NeuronFFT matches torch.fft.fft at n=256."""
        from modeling_evo2 import NeuronFFT

        n = 256
        fft_mod = NeuronFFT(n)
        x = torch.randn(4, n, dtype=torch.float32)
        x_imag = torch.zeros_like(x)

        y_r, y_i = fft_mod(x, x_imag)
        y_ref = torch.fft.fft(x, n=n)

        err = max(
            (y_r - y_ref.real).abs().max().item(),
            (y_i - y_ref.imag).abs().max().item(),
        )
        assert err < 1e-3, f"FFT error {err} exceeds threshold"

    def test_rfft_accuracy(self):
        """neuron_rfft matches torch.fft.rfft at n=1024."""
        from modeling_evo2 import NeuronFFT, neuron_rfft

        n = 1024
        fft_mod = NeuronFFT(n)
        x = torch.randn(2, 4096, 512, dtype=torch.float32)

        r_real, r_imag = neuron_rfft(x, n, fft_mod)
        r_ref = torch.fft.rfft(x, n=n)

        err = max(
            (r_real - r_ref.real).abs().max().item(),
            (r_imag - r_ref.imag).abs().max().item(),
        )
        assert err < 0.01, f"rfft error {err} exceeds threshold"

    def test_irfft_roundtrip(self):
        """rfft -> irfft roundtrip preserves signal."""
        from modeling_evo2 import NeuronFFT, neuron_rfft, neuron_irfft

        n = 512
        fft_mod = NeuronFFT(n)
        x = torch.randn(4, n, dtype=torch.float32)

        y_r, y_i = neuron_rfft(x, n, fft_mod)
        x_reconstructed = neuron_irfft(y_r, y_i, n, fft_mod)

        err = (x - x_reconstructed).abs().max().item()
        assert err < 1e-3, f"Roundtrip error {err} exceeds threshold"

    def test_no_registered_buffers(self):
        """NeuronFFT stores twiddle factors as plain attributes, not buffers."""
        from modeling_evo2 import NeuronFFT

        fft_mod = NeuronFFT(256)
        buf_names = [name for name, _ in fft_mod.named_buffers()]
        assert len(buf_names) == 0, (
            f"NeuronFFT has registered buffers: {buf_names}. "
            "These must be plain attributes to avoid NeuronModule dtype mismatch."
        )


# ============================================================================
# Prefill Tests (require Neuron hardware + model weights)
# ============================================================================


@requires_neuron
class TestPrefill:
    """Test prefill block compilation and accuracy."""

    @pytest.mark.skipif(EVO2_WEIGHTS is None, reason="EVO2_WEIGHTS not set")
    def test_single_block_compiles(self, model_and_config):
        """A single HCS block (block 0) compiles on Neuron."""
        from modeling_evo2 import BlockWrapper

        model, config = model_and_config
        wrapper = BlockWrapper(model.blocks[0])
        wrapper.eval()

        with torch.no_grad():
            for param in wrapper.parameters():
                if param.is_floating_point() and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16).contiguous()
                elif not param.data.is_contiguous():
                    param.data = param.data.contiguous()

        test_input = torch.randn(
            1,
            TEST_SEQ_LEN,
            TEST_HIDDEN_SIZE,
            dtype=torch.bfloat16,
        )

        traced = torch_neuronx.trace(
            wrapper,
            test_input,
            compiler_args=COMPILER_ARGS,
        )
        output = traced(test_input)

        assert output.shape == test_input.shape
        assert not torch.isnan(output).any(), "NaN in output"

    @pytest.mark.skipif(EVO2_WEIGHTS is None, reason="EVO2_WEIGHTS not set")
    def test_all_block_types_compile(self, model_and_config):
        """All 4 block types (HCS, HCM, HCL, ATT) compile on Neuron."""
        from modeling_evo2 import BlockWrapper, REPRESENTATIVE_BLOCKS

        model, config = model_and_config
        test_input = torch.randn(
            1,
            TEST_SEQ_LEN,
            TEST_HIDDEN_SIZE,
            dtype=torch.bfloat16,
        )

        for block_type, block_idx in REPRESENTATIVE_BLOCKS.items():
            wrapper = BlockWrapper(model.blocks[block_idx])
            wrapper.eval()

            with torch.no_grad():
                for param in wrapper.parameters():
                    if param.is_floating_point() and param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16).contiguous()
                    elif not param.data.is_contiguous():
                        param.data = param.data.contiguous()

            traced = torch_neuronx.trace(
                wrapper,
                test_input,
                compiler_args=COMPILER_ARGS,
            )
            output = traced(test_input)

            assert output.shape == test_input.shape, (
                f"Block {block_idx} ({block_type}): shape mismatch"
            )
            assert not torch.isnan(output).any(), (
                f"Block {block_idx} ({block_type}): NaN in output"
            )

    @pytest.mark.skipif(EVO2_WEIGHTS is None, reason="EVO2_WEIGHTS not set")
    def test_prefill_accuracy(self, model_and_config):
        """Full 32-block prefill matches CPU reference (cosine sim > 0.999)."""
        from modeling_evo2 import Evo2PrefillPipeline

        model, config = model_and_config

        # CPU reference
        input_ids = torch.randint(0, 512, (1, TEST_SEQ_LEN), dtype=torch.long)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits_cpu, _ = model(input_ids)
        logits_cpu = logits_cpu.float()

        # Neuron
        pipeline = Evo2PrefillPipeline(model, config, TEST_SEQ_LEN, COMPILER_ARGS)
        ok = pipeline.compile()
        assert ok, "Prefill compilation failed"

        logits_neuron = pipeline.forward(input_ids).float()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            logits_cpu.reshape(1, -1),
            logits_neuron.reshape(1, -1),
        ).item()

        assert cos_sim > COSINE_SIM_THRESHOLD, (
            f"Cosine similarity {cos_sim:.6f} < threshold {COSINE_SIM_THRESHOLD}"
        )

        # Token match
        tokens_cpu = logits_cpu.argmax(-1)
        tokens_neuron = logits_neuron.argmax(-1)
        match_rate = (tokens_cpu == tokens_neuron).float().mean().item()

        # At seq_len=128, expect > 90% match
        assert match_rate > 0.85, f"Token match rate {match_rate:.1%} too low"


# ============================================================================
# Decode Tests (require Neuron hardware + model weights)
# ============================================================================


@requires_neuron
class TestDecode:
    """Test decode block compilation."""

    @pytest.mark.skipif(EVO2_WEIGHTS is None, reason="EVO2_WEIGHTS not set")
    def test_decode_block_compiles(self, model_and_config):
        """A single ATT decode block compiles with input_output_aliases."""
        from modeling_evo2 import Evo2DecodeBlock

        model, config = model_and_config

        decode_block = Evo2DecodeBlock(
            model.blocks[3],
            block_idx=3,
            block_type="ATT",
            max_seqlen=384,
            batch_size=1,
        )
        decode_block.eval()

        with torch.no_grad():
            for param in decode_block.parameters():
                if param.is_floating_point() and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16).contiguous()
                elif not param.data.is_contiguous():
                    param.data = param.data.contiguous()

        aliases = decode_block.get_input_output_aliases()
        trace_inputs = decode_block.get_trace_inputs()

        traced = torch_neuronx.trace(
            decode_block,
            trace_inputs,
            compiler_args=COMPILER_ARGS,
            input_output_aliases=aliases,
            inline_weights_to_neff=False,
        )

        outputs = traced(*trace_inputs)
        x_out = outputs[0]
        assert x_out.shape == (1, 1, TEST_HIDDEN_SIZE)
        assert not torch.isnan(x_out).any(), "NaN in decode output"

    @pytest.mark.skipif(EVO2_WEIGHTS is None, reason="EVO2_WEIGHTS not set")
    def test_batch_mega_decode_compiles(self, model_and_config):
        """BatchMegaDecode (all 32 blocks) compiles with BS=1."""
        from modeling_evo2 import BatchMegaDecode

        model, config = model_and_config

        mega = BatchMegaDecode(model.blocks, max_seqlen=384, batch_size=1)
        mega.eval()

        with torch.no_grad():
            for param in mega.parameters():
                if param.is_floating_point() and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16).contiguous()
                elif not param.data.is_contiguous():
                    param.data = param.data.contiguous()

        aliases = mega.get_input_output_aliases()
        test_x = torch.randn(1, 1, TEST_HIDDEN_SIZE, dtype=torch.bfloat16)
        test_pos = torch.tensor([0], dtype=torch.int64)

        traced = torch_neuronx.trace(
            mega,
            (test_x, test_pos),
            compiler_args=COMPILER_ARGS,
            input_output_aliases=aliases,
            inline_weights_to_neff=False,
        )

        outputs = traced(test_x, test_pos)
        x_out = outputs[0]
        assert x_out.shape == (1, 1, TEST_HIDDEN_SIZE)
        # 64 state outputs + 1 x_out = 65 total
        assert len(outputs) == 65, f"Expected 65 outputs, got {len(outputs)}"
