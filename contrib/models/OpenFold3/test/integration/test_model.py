"""Integration test for OpenFold3 on Neuron.

Tests compilation and accuracy of the 5 compiled blocks against CPU
reference using neuron_allclose with numerical tolerances. Requires a
trn2.3xlarge instance with Neuron SDK 2.28 and OpenFold3 installed.

Usage:
    # On trn2.3xlarge with Neuron SDK 2.28
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    pip install -e /home/ubuntu/openfold-3/'[predict]' --no-deps
    pip install ml-collections biopython modelcif dm-tree biotite gemmi \
        pytorch-lightning rdkit func-timeout wandb

    export NEURON_RT_VISIBLE_CORES=0

    cd contrib/models/OpenFold3
    PYTHONPATH=src:/home/ubuntu/openfold-3:$PYTHONPATH \
        pytest test/integration/test_model.py -v -s
"""

import os
import sys
import gc
import time
from pathlib import Path

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

# Use N=128 for faster test compilation (N=256 also works but slower to compile)
TEST_N_TOKEN = 128
TEST_N_ATOM = 128
TEST_N_MSA = 4
TEST_N_TEMPL = 1
TEST_C_S = 384
TEST_C_Z = 128
TEST_C_M = 64
TEST_C_TOKEN = 768

OPENFOLD3_SRC = os.environ.get("OPENFOLD3_SRC", "/home/ubuntu/openfold-3")
CKPT_PATH = os.environ.get(
    "OPENFOLD3_CKPT",
    os.path.expanduser("~/.openfold3/of3-p2-155k.pt"),
)

# Tolerances for neuron_allclose on individual blocks
# Single block: very tight (BF16 round-trip only)
ATOL_SINGLE_BLOCK = 0.05
RTOL_SINGLE_BLOCK = 0.02

# Multi-block (weight replacement chain): slightly relaxed
ATOL_MULTI_BLOCK = 0.1
RTOL_MULTI_BLOCK = 0.05


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def openfold3_model():
    """Load OpenFold3 model (shared across all tests in this module)."""
    try:
        if OPENFOLD3_SRC not in sys.path:
            sys.path.insert(0, OPENFOLD3_SRC)
        from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
        from openfold3.projects.of3_all_atom.model import OpenFold3
    except ImportError:
        pytest.skip(
            f"OpenFold3 not importable. Ensure OpenFold3 source is accessible: "
            f"OPENFOLD3_SRC={OPENFOLD3_SRC}"
        )

    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"OpenFold3 checkpoint not found at {CKPT_PATH}")

    # Apply source patches
    from modeling_openfold3 import patch_openfold3_source

    openfold3_pkg = os.path.join(OPENFOLD3_SRC, "openfold3")
    patch_openfold3_source(openfold3_pkg)

    # Reload modules after patching (patches modify files on disk)
    from openfold3.core.utils.checkpoint_loading_utils import (
        load_checkpoint,
        get_state_dict_from_checkpoint,
    )

    project = OF3ProjectEntry()
    config = project.get_model_config_with_presets(presets=["predict", "pae_enabled"])
    config.settings.memory.eval.use_deepspeed_evo_attention = False
    config.settings.memory.eval.use_cueq_triangle_kernels = False

    model = OpenFold3(config)
    model.eval()

    ckpt = load_checkpoint(Path(CKPT_PATH))
    state_dict, _ = get_state_dict_from_checkpoint(ckpt, init_from_ema_weights=True)
    bare_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(bare_state_dict, strict=False)
    del ckpt, state_dict, bare_state_dict
    gc.collect()

    return model


# ============================================================================
# Tests: Individual Block Compilation + Accuracy
# ============================================================================


class TestPairFormerBlock:
    """Test compilation and accuracy of a single PairFormerBlock."""

    def test_compile_and_accuracy(self, openfold3_model):
        """Compile one PairFormer block and validate against CPU reference."""
        from modeling_openfold3 import PairFormerBlockWrapper

        block = openfold3_model.pairformer_stack.blocks[0]
        wrapper = PairFormerBlockWrapper(block)
        wrapper.eval()

        # Inputs
        s_in = torch.randn(1, TEST_N_TOKEN, TEST_C_S)
        z_in = torch.randn(1, TEST_N_TOKEN, TEST_N_TOKEN, TEST_C_Z)
        single_mask = torch.ones(1, TEST_N_TOKEN)
        pair_mask = torch.ones(1, TEST_N_TOKEN, TEST_N_TOKEN)

        # CPU reference
        with torch.no_grad():
            s_ref, z_ref = wrapper(
                s_in.clone(), z_in.clone(), single_mask.clone(), pair_mask.clone()
            )

        # Compile
        t0 = time.time()
        compiled = torch_neuronx.trace(
            wrapper,
            (s_in, z_in, single_mask, pair_mask),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  PairFormerBlock compile time: {compile_time:.1f}s")

        # Neuron inference
        with torch.no_grad():
            s_neu, z_neu = compiled(
                s_in.clone(), z_in.clone(), single_mask.clone(), pair_mask.clone()
            )

        # Validate with neuron_allclose
        result_s = torch_neuronx.testing.neuron_allclose(
            s_ref, s_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )
        result_z = torch_neuronx.testing.neuron_allclose(
            z_ref, z_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )

        # Also report cosine similarity for diagnostics
        cos_s = F.cosine_similarity(
            s_ref.flatten().unsqueeze(0), s_neu.flatten().unsqueeze(0)
        ).item()
        cos_z = F.cosine_similarity(
            z_ref.flatten().unsqueeze(0), z_neu.flatten().unsqueeze(0)
        ).item()
        print(f"  S neuron_allclose: {result_s}, cos_sim: {cos_s:.6f}")
        print(f"  Z neuron_allclose: {result_z}, cos_sim: {cos_z:.6f}")

        assert result_s, (
            f"PairFormerBlock S output failed neuron_allclose "
            f"(rtol={RTOL_SINGLE_BLOCK}, atol={ATOL_SINGLE_BLOCK})"
        )
        assert result_z, (
            f"PairFormerBlock Z output failed neuron_allclose "
            f"(rtol={RTOL_SINGLE_BLOCK}, atol={ATOL_SINGLE_BLOCK})"
        )


class TestPairFormerWeightReplacement:
    """Test weight replacement across multiple PairFormer layers."""

    def test_two_layer_weight_replacement(self, openfold3_model):
        """Compile one block, replace weights, validate 2-layer chain."""
        from modeling_openfold3 import PairFormerBlockWrapper

        pf_stack = openfold3_model.pairformer_stack.blocks
        wrapper0 = PairFormerBlockWrapper(pf_stack[0])
        wrapper1 = PairFormerBlockWrapper(pf_stack[1])
        wrapper0.eval()
        wrapper1.eval()

        s_in = torch.randn(1, TEST_N_TOKEN, TEST_C_S)
        z_in = torch.randn(1, TEST_N_TOKEN, TEST_N_TOKEN, TEST_C_Z)
        single_mask = torch.ones(1, TEST_N_TOKEN)
        pair_mask = torch.ones(1, TEST_N_TOKEN, TEST_N_TOKEN)

        # CPU reference: block0 then block1
        with torch.no_grad():
            s_mid, z_mid = wrapper0(
                s_in.clone(), z_in.clone(), single_mask.clone(), pair_mask.clone()
            )
            s_ref, z_ref = wrapper1(
                s_mid.clone(), z_mid.clone(), single_mask.clone(), pair_mask.clone()
            )

        # Compile block0
        compiled = torch_neuronx.trace(
            wrapper0,
            (s_in, z_in, single_mask, pair_mask),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )

        # Run block0 on Neuron
        with torch.no_grad():
            s_mid_neu, z_mid_neu = compiled(
                s_in.clone(), z_in.clone(), single_mask.clone(), pair_mask.clone()
            )

        # Replace weights to block1
        torch_neuronx.replace_weights(compiled, wrapper1.state_dict())

        # Run block1 on Neuron
        with torch.no_grad():
            s_neu, z_neu = compiled(
                s_mid_neu, z_mid_neu, single_mask.clone(), pair_mask.clone()
            )

        result_s = torch_neuronx.testing.neuron_allclose(
            s_ref, s_neu, rtol=RTOL_MULTI_BLOCK, atol=ATOL_MULTI_BLOCK
        )
        result_z = torch_neuronx.testing.neuron_allclose(
            z_ref, z_neu, rtol=RTOL_MULTI_BLOCK, atol=ATOL_MULTI_BLOCK
        )

        cos_s = F.cosine_similarity(
            s_ref.flatten().unsqueeze(0), s_neu.flatten().unsqueeze(0)
        ).item()
        cos_z = F.cosine_similarity(
            z_ref.flatten().unsqueeze(0), z_neu.flatten().unsqueeze(0)
        ).item()
        print(
            f"\n  2-layer weight replacement: "
            f"S allclose={result_s} cos={cos_s:.6f}, "
            f"Z allclose={result_z} cos={cos_z:.6f}"
        )

        assert result_s, "PairFormer 2-layer S output failed neuron_allclose"
        assert result_z, "PairFormer 2-layer Z output failed neuron_allclose"


class TestMSABlockTypeA:
    """Test compilation and accuracy of MSA block type A (blocks 0-2)."""

    def test_compile_and_accuracy(self, openfold3_model):
        """Compile one MSA type-A block and validate against CPU reference."""
        from modeling_openfold3 import MSABlockWrapper

        block = openfold3_model.msa_module.blocks[0]
        wrapper = MSABlockWrapper(block)
        wrapper.eval()

        m_in = torch.randn(1, TEST_N_MSA, TEST_N_TOKEN, TEST_C_M)
        z_in = torch.randn(1, TEST_N_TOKEN, TEST_N_TOKEN, TEST_C_Z)
        msa_mask = torch.ones(1, TEST_N_MSA, TEST_N_TOKEN)
        pair_mask = torch.ones(1, TEST_N_TOKEN, TEST_N_TOKEN)

        with torch.no_grad():
            m_ref, z_ref = wrapper(
                m_in.clone(), z_in.clone(), msa_mask.clone(), pair_mask.clone()
            )

        t0 = time.time()
        compiled = torch_neuronx.trace(
            wrapper,
            (m_in, z_in, msa_mask, pair_mask),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  MSA type-A compile time: {compile_time:.1f}s")

        with torch.no_grad():
            m_neu, z_neu = compiled(
                m_in.clone(), z_in.clone(), msa_mask.clone(), pair_mask.clone()
            )

        result_m = torch_neuronx.testing.neuron_allclose(
            m_ref, m_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )
        result_z = torch_neuronx.testing.neuron_allclose(
            z_ref, z_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )

        cos_m = F.cosine_similarity(
            m_ref.flatten().unsqueeze(0), m_neu.flatten().unsqueeze(0)
        ).item()
        cos_z = F.cosine_similarity(
            z_ref.flatten().unsqueeze(0), z_neu.flatten().unsqueeze(0)
        ).item()
        print(f"  M neuron_allclose: {result_m}, cos_sim: {cos_m:.6f}")
        print(f"  Z neuron_allclose: {result_z}, cos_sim: {cos_z:.6f}")

        assert result_m, "MSA type-A M output failed neuron_allclose"
        assert result_z, "MSA type-A Z output failed neuron_allclose"


class TestMSABlockTypeB:
    """Test compilation and accuracy of MSA block type B (last block)."""

    def test_compile_and_accuracy(self, openfold3_model):
        """Compile MSA type-B block (different structure) and validate."""
        from modeling_openfold3 import MSABlockWrapper

        num_blocks = len(openfold3_model.msa_module.blocks)
        block = openfold3_model.msa_module.blocks[num_blocks - 1]
        wrapper = MSABlockWrapper(block)
        wrapper.eval()

        m_in = torch.randn(1, TEST_N_MSA, TEST_N_TOKEN, TEST_C_M)
        z_in = torch.randn(1, TEST_N_TOKEN, TEST_N_TOKEN, TEST_C_Z)
        msa_mask = torch.ones(1, TEST_N_MSA, TEST_N_TOKEN)
        pair_mask = torch.ones(1, TEST_N_TOKEN, TEST_N_TOKEN)

        with torch.no_grad():
            m_ref, z_ref = wrapper(
                m_in.clone(), z_in.clone(), msa_mask.clone(), pair_mask.clone()
            )

        t0 = time.time()
        compiled = torch_neuronx.trace(
            wrapper,
            (m_in, z_in, msa_mask, pair_mask),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  MSA type-B compile time: {compile_time:.1f}s")

        with torch.no_grad():
            m_neu, z_neu = compiled(
                m_in.clone(), z_in.clone(), msa_mask.clone(), pair_mask.clone()
            )

        result_m = torch_neuronx.testing.neuron_allclose(
            m_ref, m_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )
        result_z = torch_neuronx.testing.neuron_allclose(
            z_ref, z_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )

        cos_m = F.cosine_similarity(
            m_ref.flatten().unsqueeze(0), m_neu.flatten().unsqueeze(0)
        ).item()
        cos_z = F.cosine_similarity(
            z_ref.flatten().unsqueeze(0), z_neu.flatten().unsqueeze(0)
        ).item()
        print(f"  M neuron_allclose: {result_m}, cos_sim: {cos_m:.6f}")
        print(f"  Z neuron_allclose: {result_z}, cos_sim: {cos_z:.6f}")

        assert result_m, "MSA type-B M output failed neuron_allclose"
        assert result_z, "MSA type-B Z output failed neuron_allclose"


class TestTemplateBlock:
    """Test compilation and accuracy of TemplatePairBlock."""

    def test_compile_and_accuracy(self, openfold3_model):
        """Compile one template block and validate against CPU reference."""
        from modeling_openfold3 import TemplatePairBlockWrapper

        block = openfold3_model.template_embedder.template_pair_stack.blocks[0]
        wrapper = TemplatePairBlockWrapper(block)
        wrapper.eval()

        t_in = torch.randn(1, TEST_N_TEMPL, TEST_N_TOKEN, TEST_N_TOKEN, 64)
        t_mask = torch.ones(1, TEST_N_TEMPL, TEST_N_TOKEN, TEST_N_TOKEN)

        with torch.no_grad():
            t_ref = wrapper(t_in.clone(), t_mask.clone())

        t0 = time.time()
        compiled = torch_neuronx.trace(
            wrapper,
            (t_in, t_mask),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  TemplateBlock compile time: {compile_time:.1f}s")

        with torch.no_grad():
            t_neu = compiled(t_in.clone(), t_mask.clone())

        result = torch_neuronx.testing.neuron_allclose(
            t_ref, t_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )

        cos = F.cosine_similarity(
            t_ref.flatten().unsqueeze(0), t_neu.flatten().unsqueeze(0)
        ).item()
        print(f"  T neuron_allclose: {result}, cos_sim: {cos:.6f}")

        assert result, "TemplateBlock output failed neuron_allclose"


class TestDiffCondForward:
    """Test compilation and accuracy of DiffusionConditioning._forward()."""

    def test_compile_and_accuracy(self, openfold3_model):
        """Compile DiffCond._forward() and validate against CPU reference."""
        from modeling_openfold3 import DiffCondForwardWrapper

        dc = openfold3_model.diffusion_module.diffusion_conditioning
        wrapper = DiffCondForwardWrapper(dc)
        wrapper.eval()

        s_in = torch.randn(1, TEST_N_TOKEN, TEST_C_S)
        z_in = torch.randn(1, TEST_N_TOKEN, TEST_N_TOKEN, TEST_C_Z)
        mask = torch.ones(1, TEST_N_TOKEN)

        with torch.no_grad():
            s_ref, z_ref = wrapper(s_in.clone(), z_in.clone(), mask.clone())

        t0 = time.time()
        compiled = torch_neuronx.trace(
            wrapper,
            (s_in, z_in, mask),
            compiler_args=["--target", "trn2"],
            inline_weights_to_neff=False,
        )
        compile_time = time.time() - t0
        print(f"\n  DiffCond._forward() compile time: {compile_time:.1f}s")

        with torch.no_grad():
            s_neu, z_neu = compiled(s_in.clone(), z_in.clone(), mask.clone())

        result_s = torch_neuronx.testing.neuron_allclose(
            s_ref, s_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )
        result_z = torch_neuronx.testing.neuron_allclose(
            z_ref, z_neu, rtol=RTOL_SINGLE_BLOCK, atol=ATOL_SINGLE_BLOCK
        )

        cos_s = F.cosine_similarity(
            s_ref.flatten().unsqueeze(0), s_neu.flatten().unsqueeze(0)
        ).item()
        cos_z = F.cosine_similarity(
            z_ref.flatten().unsqueeze(0), z_neu.flatten().unsqueeze(0)
        ).item()
        print(f"  S neuron_allclose: {result_s}, cos_sim: {cos_s:.6f}")
        print(f"  Z neuron_allclose: {result_z}, cos_sim: {cos_z:.6f}")

        assert result_s, "DiffCond S output failed neuron_allclose"
        assert result_z, "DiffCond Z output failed neuron_allclose"
