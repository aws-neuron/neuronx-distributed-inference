"""Integration test for RosettaFold3 on Neuron.

Tests compilation and accuracy of the 5 compiled blocks against CPU reference.
Requires a trn2.3xlarge instance with Neuron SDK 2.28 and the RF3 Foundry
model installed.

Usage:
    # On trn2.3xlarge with Neuron SDK 2.28
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    pip install lightning

    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    export NEURON_RT_VISIBLE_CORES=0

    # RF3 source must be on PYTHONPATH:
    export PYTHONPATH=/mnt/models/foundry/models/rf3/src:/mnt/models/foundry/src:$PYTHONPATH

    cd contrib/models/RosettaFold3
    PYTHONPATH=src:$PYTHONPATH pytest test/integration/test_model.py -v -s
"""

import os
import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# Skip entire module if not on Neuron hardware
try:
    import torch_neuronx

    HAS_NEURONX = True
except ImportError:
    HAS_NEURONX = False

pytestmark = pytest.mark.skipif(
    not HAS_NEURONX, reason="torch_neuronx not available (requires Neuron hardware)"
)


# ============================================================================
# Constants
# ============================================================================

# Use small I for faster test compilation
TEST_I = 128
TEST_I_PADDED = 128  # Already aligned to 8
TEST_D = 1
TEST_C_S = 384
TEST_C_Z = 128
TEST_C_TOKEN = 768
TEST_C_Z_TEMPL = 64
TEST_C_ATOM = 128
TEST_C_ATOMPAIR = 16

CKPT_PATH = os.environ.get(
    "RF3_CKPT_PATH", "/mnt/models/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt"
)

# Accuracy thresholds
COS_SIM_THRESHOLD_SINGLE_BLOCK = 0.999
COS_SIM_THRESHOLD_MULTI_BLOCK = 0.99
COS_SIM_THRESHOLD_E2E = 0.98


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def rf3_model():
    """Load RF3 model (shared across all tests in this module)."""
    # Check that RF3 is importable
    try:
        from rf3.inference_engines.rf3 import RF3InferenceEngine
    except ImportError:
        pytest.skip(
            "RF3 not importable. Ensure RF3 source is on PYTHONPATH: "
            "export PYTHONPATH=/mnt/models/foundry/models/rf3/src:"
            "/mnt/models/foundry/src:$PYTHONPATH"
        )

    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"RF3 checkpoint not found at {CKPT_PATH}")

    engine = RF3InferenceEngine(
        ckpt_path=CKPT_PATH,
        n_recycles=1,
        diffusion_batch_size=TEST_D,
        num_steps=2,
        seed=42,
        verbose=False,
    )
    engine.initialize()

    ema = engine.trainer.state["model"]._forward_module
    rf3_with_conf = ema.shadow

    # Disable CUDA-specific paths
    for _, m in rf3_with_conf.named_modules():
        if hasattr(m, "use_cuequivariance"):
            m.use_cuequivariance = False
        if hasattr(m, "use_deepspeed_evo"):
            m.use_deepspeed_evo = False
    for _, m in ema.model.named_modules():
        if hasattr(m, "use_cuequivariance"):
            m.use_cuequivariance = False
        if hasattr(m, "use_deepspeed_evo"):
            m.use_deepspeed_evo = False

    return rf3_with_conf


# ============================================================================
# Tests: Individual Block Compilation
# ============================================================================


class TestPairformerBlock:
    """Test compilation and accuracy of a single PairformerBlock."""

    def test_compile_and_accuracy(self, rf3_model):
        """Compile one pairformer block and validate against CPU reference."""
        from modeling_rf3 import FullPairformerBlock

        pf_stack = rf3_model.recycler.pairformer_stack
        block = FullPairformerBlock(pf_stack[0]).eval().to(torch.bfloat16)

        # CPU reference
        S_in = torch.randn(1, TEST_I_PADDED, TEST_C_S, dtype=torch.bfloat16)
        Z_in = torch.randn(
            1, TEST_I_PADDED, TEST_I_PADDED, TEST_C_Z, dtype=torch.bfloat16
        )

        with torch.no_grad():
            S_ref, Z_ref = block(S_in.clone(), Z_in.clone())

        # Compile
        t0 = time.time()
        compiled = torch_neuronx.trace(
            block,
            (S_in, Z_in),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  PairformerBlock compile time: {compile_time:.1f}s")

        # Neuron inference
        with torch.no_grad():
            S_neu, Z_neu = compiled(S_in.clone(), Z_in.clone())

        # Accuracy
        cos_S = F.cosine_similarity(
            S_ref.float().flatten().unsqueeze(0),
            S_neu.float().flatten().unsqueeze(0),
        ).item()
        cos_Z = F.cosine_similarity(
            Z_ref.float().flatten().unsqueeze(0),
            Z_neu.float().flatten().unsqueeze(0),
        ).item()

        print(f"  S cos_sim: {cos_S:.6f}")
        print(f"  Z cos_sim: {cos_Z:.6f}")

        assert cos_S > COS_SIM_THRESHOLD_SINGLE_BLOCK, (
            f"S cos_sim {cos_S} below threshold {COS_SIM_THRESHOLD_SINGLE_BLOCK}"
        )
        assert cos_Z > COS_SIM_THRESHOLD_SINGLE_BLOCK, (
            f"Z cos_sim {cos_Z} below threshold {COS_SIM_THRESHOLD_SINGLE_BLOCK}"
        )


class TestPairformerWeightReplacement:
    """Test weight replacement across multiple pairformer layers."""

    def test_two_layer_weight_replacement(self, rf3_model):
        """Compile one block and validate weight replacement with a second block."""
        from modeling_rf3 import FullPairformerBlock

        pf_stack = rf3_model.recycler.pairformer_stack
        block0 = FullPairformerBlock(pf_stack[0]).eval().to(torch.bfloat16)
        block1 = FullPairformerBlock(pf_stack[1]).eval().to(torch.bfloat16)

        S_in = torch.randn(1, TEST_I_PADDED, TEST_C_S, dtype=torch.bfloat16)
        Z_in = torch.randn(
            1, TEST_I_PADDED, TEST_I_PADDED, TEST_C_Z, dtype=torch.bfloat16
        )

        # CPU reference: block0 then block1
        with torch.no_grad():
            S_mid, Z_mid = block0(S_in.clone(), Z_in.clone())
            S_ref, Z_ref = block1(S_mid.clone(), Z_mid.clone())

        # Compile block0
        compiled = torch_neuronx.trace(
            block0,
            (S_in, Z_in),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )

        # Run block0 on Neuron
        with torch.no_grad():
            S_mid_neu, Z_mid_neu = compiled(S_in.clone(), Z_in.clone())

        # Replace weights to block1
        torch_neuronx.replace_weights(compiled, block1.state_dict())

        # Run block1 on Neuron
        with torch.no_grad():
            S_neu, Z_neu = compiled(S_mid_neu, Z_mid_neu)

        cos_S = F.cosine_similarity(
            S_ref.float().flatten().unsqueeze(0),
            S_neu.float().flatten().unsqueeze(0),
        ).item()
        cos_Z = F.cosine_similarity(
            Z_ref.float().flatten().unsqueeze(0),
            Z_neu.float().flatten().unsqueeze(0),
        ).item()

        print(
            f"\n  2-layer weight replacement: S cos_sim={cos_S:.6f}, Z cos_sim={cos_Z:.6f}"
        )

        assert cos_S > COS_SIM_THRESHOLD_MULTI_BLOCK, (
            f"S cos_sim {cos_S} below threshold"
        )
        assert cos_Z > COS_SIM_THRESHOLD_MULTI_BLOCK, (
            f"Z cos_sim {cos_Z} below threshold"
        )


class TestDiffTransformerBlock:
    """Test compilation and accuracy of a single DiffTransformerBlock."""

    def test_compile_and_accuracy(self, rf3_model):
        """Compile one diff transformer block and validate against CPU reference."""
        from modeling_rf3 import DiffTransformerBlock

        diff_transformer = rf3_model.diffusion_module.diffusion_transformer
        block = (
            DiffTransformerBlock(diff_transformer.blocks[0]).eval().to(torch.bfloat16)
        )

        A_in = torch.randn(TEST_D, TEST_I_PADDED, TEST_C_TOKEN, dtype=torch.bfloat16)
        S_in = torch.randn(TEST_D, TEST_I_PADDED, TEST_C_S, dtype=torch.bfloat16)
        Z_in = torch.randn(
            TEST_D, TEST_I_PADDED, TEST_I_PADDED, TEST_C_Z, dtype=torch.bfloat16
        )

        with torch.no_grad():
            A_ref = block(A_in.clone(), S_in.clone(), Z_in.clone())

        # Compile
        t0 = time.time()
        compiled = torch_neuronx.trace(
            block,
            (A_in, S_in, Z_in),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  DiffTransformerBlock compile time: {compile_time:.1f}s")

        with torch.no_grad():
            A_neu = compiled(A_in.clone(), S_in.clone(), Z_in.clone())

        cos_A = F.cosine_similarity(
            A_ref.float().flatten().unsqueeze(0),
            A_neu.float().flatten().unsqueeze(0),
        ).item()
        print(f"  A cos_sim: {cos_A:.6f}")

        assert cos_A > COS_SIM_THRESHOLD_SINGLE_BLOCK, (
            f"A cos_sim {cos_A} below threshold {COS_SIM_THRESHOLD_SINGLE_BLOCK}"
        )


class TestAtomAttentionBlock:
    """Test compilation and accuracy of StaticWindowedAttnBlock."""

    def test_compile_and_accuracy(self, rf3_model):
        """Compile one atom attention block and validate against CPU reference."""
        from modeling_rf3 import StaticWindowedAttnBlock, precompute_window_indices

        L_test = 512
        enc_dt = rf3_model.diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer

        block = (
            StaticWindowedAttnBlock(enc_dt.blocks[0], 4, 32, 32, 128)
            .eval()
            .to(torch.bfloat16)
        )

        # CPU reference using original block
        orig_block = enc_dt.blocks[0].eval().to(torch.bfloat16)
        A_in = torch.randn(TEST_D, L_test, TEST_C_ATOM, dtype=torch.bfloat16)
        S_in = torch.randn(TEST_D, L_test, TEST_C_ATOM, dtype=torch.bfloat16)
        P_in = torch.randn(L_test, L_test, TEST_C_ATOMPAIR, dtype=torch.bfloat16)

        with torch.no_grad():
            A_ref = orig_block(A_in.clone(), S_in.clone(), P_in.clone(), True)

        # Static wrapper reference (CPU)
        indicesQ, indicesK, maskQ, maskK, nq = precompute_window_indices(
            L_test, 32, 128
        )
        maskQ_bf = maskQ.float().to(torch.bfloat16)
        maskK_bf = maskK.float().to(torch.bfloat16)
        Z_4d = P_in.unsqueeze(0)

        with torch.no_grad():
            A_wrap = block(
                A_in.clone(),
                S_in.clone(),
                Z_4d.clone(),
                indicesQ,
                indicesK,
                maskQ_bf,
                maskK_bf,
            )

        cos_wrap = F.cosine_similarity(
            A_ref.float().flatten().unsqueeze(0),
            A_wrap.float().flatten().unsqueeze(0),
        ).item()
        print(f"\n  Static wrapper vs original (CPU): cos_sim={cos_wrap:.6f}")

        # Compile
        trace_inputs = (
            A_in,
            S_in,
            Z_4d,
            indicesQ.long(),
            indicesK.long(),
            maskQ_bf,
            maskK_bf,
        )
        t0 = time.time()
        compiled = torch_neuronx.trace(
            block,
            trace_inputs,
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"  AtomAttention compile time: {compile_time:.1f}s")

        with torch.no_grad():
            A_neu = compiled(*trace_inputs)

        cos_neu = F.cosine_similarity(
            A_ref.float().flatten().unsqueeze(0),
            A_neu.float().flatten().unsqueeze(0),
        ).item()
        print(f"  Neuron vs original: cos_sim={cos_neu:.6f}")

        assert cos_neu > COS_SIM_THRESHOLD_SINGLE_BLOCK, (
            f"cos_sim {cos_neu} below threshold {COS_SIM_THRESHOLD_SINGLE_BLOCK}"
        )


class TestBucketing:
    """Test bucketing utilities (CPU-only, no Neuron hardware required)."""

    def test_bucket_selection(self):
        """Test that bucket selection returns correct sizes."""
        from rf3_bucketing import BucketConfig

        config = BucketConfig()

        assert config.select_I_bucket(58) == 128
        assert config.select_I_bucket(128) == 128
        assert config.select_I_bucket(240) == 256
        assert config.select_I_bucket(448) == 448

        with pytest.raises(ValueError):
            config.select_I_bucket(500)

    def test_pad_unpad_roundtrip(self):
        """Test that padding and unpadding is lossless."""
        from rf3_bucketing import pad_single, unpad_single, pad_pair, unpad_pair

        I_actual, I_bucket = 240, 256

        S = torch.randn(I_actual, 384)
        S_padded = pad_single(S, I_bucket)
        S_back = unpad_single(S_padded, I_actual)
        assert S_back.shape == S.shape
        assert torch.equal(S_back, S)

        Z = torch.randn(I_actual, I_actual, 128)
        Z_padded = pad_pair(Z, I_bucket)
        Z_back = unpad_pair(Z_padded, I_actual)
        assert Z_back.shape == Z.shape
        assert torch.equal(Z_back, Z)
