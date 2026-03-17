"""OpenFold3 Neuron acceleration module.

Provides wrapper classes and monkey-patching utilities for running
OpenFold3 (AlphaFold3 reproduction, ~330M params) on AWS Trainium 2
hardware. Uses vanilla torch_neuronx.trace() compilation (no NKI
kernels) with the replace_weights pattern for multi-layer stacks.

Five model components are compiled:
  1. PairFormerBlock (48 layers) - main trunk
  2. MSA block type A (3 blocks) - full MSA blocks 0-2
  3. MSA block type B (1 block) - last MSA block (different structure)
  4. TemplatePairBlock (2 blocks) - template embedder
  5. DiffusionConditioning._forward() (1 block, shared weights) - conditioning

Architecture constants (from OpenFold3):
  C_S = 384 (single representation)
  C_Z = 128 (pair representation)
  C_M = 64 (MSA representation)
  C_TOKEN = 768 (token/atom representation)
"""

import os
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# Architecture Constants
# ============================================================================

C_S = 384
C_Z = 128
C_M = 64
C_TOKEN = 768


# ============================================================================
# Wrapper Modules for Neuron Compilation
# ============================================================================


class PairFormerBlockWrapper(nn.Module):
    """Wrapper for a single OpenFold3 PairFormerBlock.

    The PairFormerBlock contains triangle multiplicative updates,
    triangle attention, pair transition, and single attention.
    This wrapper exposes (s, z, single_mask, pair_mask) -> (s, z)
    for tracing, fixing boolean kwargs to evaluation defaults.

    Args:
        block: A PairFormerBlock from model.pairformer_stack.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one pairformer layer.

        Args:
            s: [1, N, C_S] single representation
            z: [1, N, N, C_Z] pair representation
            single_mask: [1, N] mask
            pair_mask: [1, N, N] mask

        Returns:
            (s, z): updated representations
        """
        s_out, z_out = self.block(
            s=s,
            z=z,
            single_mask=single_mask,
            pair_mask=pair_mask,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
        )
        return s_out, z_out


class MSABlockWrapper(nn.Module):
    """Wrapper for a single OpenFold3 MSA block.

    OpenFold3 has two MSA block types:
      - Type A (blocks 0-2): Full blocks with msa_att_row, msa_transition,
        outer_product_mean, and pair_stack (60 params each).
      - Type B (block 3): Reduced block with only outer_product_mean and
        pair_stack (47 params). No msa_att_row or msa_transition.

    Both types accept the same interface: (m, z, msa_mask, pair_mask) -> (m, z).

    Args:
        block: An MSA block from model.msa_module.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one MSA block.

        Args:
            m: [1, N_msa, N, C_M] MSA representation
            z: [1, N, N, C_Z] pair representation
            msa_mask: [1, N_msa, N] mask
            pair_mask: [1, N, N] mask

        Returns:
            (m, z): updated representations
        """
        m_out, z_out = self.block(
            m=m,
            z=z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            inplace_safe=False,
        )
        return m_out, z_out


class TemplatePairBlockWrapper(nn.Module):
    """Wrapper for a single OpenFold3 TemplatePairBlock.

    The template embedder contains 2 pairformer-style blocks that
    operate on template pair representations with c_t=64 (smaller
    than the main pairformer's c_z=128).

    Args:
        block: A TemplatePairBlock from
               model.template_embedder.template_pair_stack.blocks[i]
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run one template pairformer layer.

        Args:
            t: [1, N_templ, N, N, 64] template pair representation
            mask: [1, N_templ, N, N] template mask

        Returns:
            t: updated template pair representation
        """
        t_out = self.block(
            t=t,
            mask=mask,
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=False,
            use_lma=False,
            inplace_safe=False,
        )
        return t_out


class DiffCondForwardWrapper(nn.Module):
    """Wrapper for DiffusionConditioning._forward().

    The outer DiffusionConditioning forward() uses batch dict inputs
    (not traceable), but the inner _forward() is pure tensor math:
    transition layers applied to (si, zij, token_mask). This wrapper
    extracts just those layers for tracing.

    Args:
        diff_cond: The DiffusionConditioning module from
                   model.diffusion_module.diffusion_conditioning
    """

    def __init__(self, diff_cond):
        super().__init__()
        self.transition_z = diff_cond.transition_z
        self.transition_s = diff_cond.transition_s

    def forward(
        self,
        si: torch.Tensor,
        zij: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run diffusion conditioning transitions.

        Args:
            si: [1, N, C_S] single conditioning
            zij: [1, N, N, C_Z] pair conditioning
            token_mask: [1, N] mask

        Returns:
            (si, zij): updated conditioning tensors
        """
        pair_token_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_token_mask)
        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)
        return si, zij


# ============================================================================
# Source Code Patches
# ============================================================================


def patch_openfold3_source(openfold3_path: str) -> List[str]:
    """Apply Neuron compatibility patches to OpenFold3 source code.

    OpenFold3 contains CUDA-specific code that must be replaced for
    Neuron compatibility. This function patches the source files
    in-place. It is idempotent -- running it multiple times is safe.

    Patches applied:
      1. autocast("cuda") -> autocast("cpu") in 5 files (13 occurrences)
      2. device_type="cuda" -> device_type="cpu" in 3 files
      3. torch.cuda.empty_cache() -> pass in 6 files
      4. torch.cuda.synchronize() -> pass in callbacks.py
      5. torch.cuda.manual_seed_all() -> pass in callbacks.py
      6. use_deepspeed_evo_attention: True -> False in model_config.py

    Args:
        openfold3_path: Path to the openfold3 package directory
            (e.g., '/home/ubuntu/openfold-3/openfold3')

    Returns:
        List of patch descriptions applied
    """
    patches = []
    base = openfold3_path

    # Patch 1: autocast("cuda") -> autocast("cpu")
    autocast_files = [
        "core/model/primitives/attention.py",
        "core/model/primitives/linear.py",
        "core/model/primitives/normalization.py",
        "core/utils/geometry/kabsch_alignment.py",
        "core/loss/diffusion.py",
    ]
    for f in autocast_files:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            content = fh.read()
        original = content
        content = content.replace(
            'torch.amp.autocast("cuda"', 'torch.amp.autocast("cpu"'
        )
        if content != original:
            with open(path, "w") as fh:
                fh.write(content)
            count = original.count('torch.amp.autocast("cuda"')
            patches.append(f"{f}: replaced {count} autocast('cuda') -> autocast('cpu')")

    # Patch 2: device_type="cuda" -> device_type="cpu"
    device_type_files = [
        "projects/of3_all_atom/model.py",
        "core/model/heads/prediction_heads.py",
        "core/model/feature_embedders/input_embedders.py",
    ]
    for f in device_type_files:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            content = fh.read()
        original = content
        content = content.replace('device_type="cuda"', 'device_type="cpu"')
        if content != original:
            with open(path, "w") as fh:
                fh.write(content)
            count = original.count('device_type="cuda"')
            patches.append(
                f"{f}: replaced {count} device_type='cuda' -> device_type='cpu'"
            )

    # Patch 3: torch.cuda.empty_cache() -> pass
    empty_cache_files = [
        "projects/of3_all_atom/runner.py",
        "projects/of3_all_atom/model.py",
        "core/model/latent/base_stacks.py",
        "core/model/latent/pairformer.py",
        "core/model/latent/msa_module.py",
        "core/model/latent/evoformer.py",
    ]
    for f in empty_cache_files:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            content = fh.read()
        original = content
        content = content.replace(
            "torch.cuda.empty_cache()", "pass  # empty_cache removed for Neuron"
        )
        if content != original:
            with open(path, "w") as fh:
                fh.write(content)
            count = original.count("torch.cuda.empty_cache()")
            patches.append(f"{f}: replaced {count} empty_cache() -> pass")

    # Patch 4: callbacks.py
    callbacks_path = os.path.join(base, "core/utils/callbacks.py")
    if os.path.exists(callbacks_path):
        with open(callbacks_path) as fh:
            content = fh.read()
        original = content
        content = content.replace(
            "torch.cuda.synchronize()", "pass  # synchronize removed for Neuron"
        )
        content = content.replace(
            "torch.cuda.manual_seed_all(rank_specific_seed)",
            "pass  # manual_seed_all removed for Neuron",
        )
        if content != original:
            with open(callbacks_path, "w") as fh:
                fh.write(content)
            patches.append(
                "core/utils/callbacks.py: replaced synchronize() and manual_seed_all()"
            )

    # Patch 5: model_config.py — disable deepspeed evo for eval
    config_path = os.path.join(base, "projects/of3_all_atom/config/model_config.py")
    if os.path.exists(config_path):
        with open(config_path) as fh:
            lines = fh.readlines()
        modified = False
        for i, line in enumerate(lines):
            if '"use_deepspeed_evo_attention": True,' in line:
                lines[i] = line.replace(
                    '"use_deepspeed_evo_attention": True,',
                    '"use_deepspeed_evo_attention": False,  # Neuron: disabled',
                )
                modified = True
        if modified:
            with open(config_path, "w") as fh:
                fh.writelines(lines)
            patches.append("model_config.py: set use_deepspeed_evo_attention=False")

    return patches


# ============================================================================
# Dummy Batch Creation
# ============================================================================


def create_dummy_batch(
    n_token: int = 256,
    n_atom: int = 256,
    n_msa: int = 4,
    n_templ: int = 1,
    seed: int = 42,
) -> dict:
    """Create a dummy input batch for OpenFold3 inference.

    The batch format matches what OpenFold3.forward() expects. Uses
    is_protein=0, is_atomized=1 for a 1-atom-per-token setup to avoid
    CB atom index OOB errors.

    Args:
        n_token: Number of tokens (max 256 on Neuron)
        n_atom: Number of atoms (set equal to n_token for 1:1 mapping)
        n_msa: Number of MSA sequences
        n_templ: Number of templates
        seed: Random seed for reproducibility

    Returns:
        dict: Input batch dictionary with all required keys
    """
    torch.manual_seed(seed)

    batch = {
        # Token-level
        "residue_index": torch.arange(n_token).long().unsqueeze(0),
        "token_index": torch.arange(n_token).long().unsqueeze(0),
        "asym_id": torch.zeros(1, n_token, dtype=torch.long),
        "entity_id": torch.zeros(1, n_token, dtype=torch.long),
        "sym_id": torch.zeros(1, n_token, dtype=torch.long),
        "restype": torch.nn.functional.one_hot(
            torch.zeros(1, n_token, dtype=torch.long), 32
        ).float(),
        "is_protein": torch.zeros(1, n_token),
        "is_rna": torch.zeros(1, n_token),
        "is_dna": torch.zeros(1, n_token),
        "is_ligand": torch.zeros(1, n_token),
        "is_atomized": torch.ones(1, n_token),
        "token_bonds": torch.zeros(1, n_token, n_token),
        "token_mask": torch.ones(1, n_token),
        "num_atoms_per_token": torch.ones(1, n_token, dtype=torch.long),
        "start_atom_index": torch.arange(n_token).long().unsqueeze(0),
        "profile": torch.zeros(1, n_token, 32),
        "deletion_mean": torch.zeros(1, n_token),
        # Atom-level
        "ref_pos": torch.randn(1, n_atom, 3),
        "ref_mask": torch.ones(1, n_atom),
        "ref_element": torch.zeros(1, n_atom, 119),  # 119 element types
        "ref_charge": torch.zeros(1, n_atom),
        "ref_atom_name_chars": torch.zeros(1, n_atom, 4, 64),
        "ref_space_uid": torch.arange(n_atom).long().unsqueeze(0),
        "atom_mask": torch.ones(1, n_atom),
        "atom_to_token_index": torch.arange(n_atom).long().unsqueeze(0),
        # MSA
        "msa": torch.zeros(1, n_msa, n_token, 32),
        "has_deletion": torch.zeros(1, n_msa, n_token),
        "deletion_value": torch.zeros(1, n_msa, n_token),
        "msa_mask": torch.ones(1, n_msa, n_token),
        "num_paired_seqs": torch.tensor([0], dtype=torch.long),
        # Template
        "template_restype": torch.zeros(1, n_templ, n_token, 32),
        "template_pseudo_beta_mask": torch.zeros(1, n_templ, n_token),
        "template_backbone_frame_mask": torch.zeros(1, n_templ, n_token),
        "template_distogram": torch.zeros(1, n_templ, n_token, n_token, 39),
        "template_unit_vector": torch.zeros(1, n_templ, n_token, n_token, 3),
    }
    return batch


# ============================================================================
# OpenFold3 Neuron Pipeline
# ============================================================================


class OpenFold3NeuronPipeline:
    """End-to-end pipeline for OpenFold3 inference on Neuron.

    Handles model loading, source patching, compilation of 5 block types,
    weight replacement, monkey-patching, and inference. The pipeline
    patches the original model's forward() path so all orchestration
    (recycling, diffusion loop, confidence heads) stays on CPU while
    compute-heavy blocks run on Neuron.

    Compiled blocks:
      - PairFormerBlock: 48 layers, 1 NEFF, 47 weight swaps
      - MSA type A: 3 blocks, 1 NEFF, 2 weight swaps
      - MSA type B: 1 block, 1 NEFF (different structure)
      - TemplatePairBlock: 2 blocks, 1 NEFF, 1 weight swap
      - DiffusionConditioning._forward(): 1 block, shared weights

    Not compiled (run on CPU):
      - AtomAttentionEncoder/Decoder: batch dict inputs, data-dependent ops
      - InputEmbedder: batch dict inputs
      - DiffTransformerBlock: weight replacement overhead exceeds compute
        at N<=256 (1.3ms overhead x 4800 calls = 6.2s net slower)
      - Confidence heads' 4-block PairFormer: minor runtime contribution

    Usage::

        from modeling_openfold3 import OpenFold3NeuronPipeline

        pipeline = OpenFold3NeuronPipeline(
            openfold3_src_path="/home/ubuntu/openfold-3",
            checkpoint_path="~/.openfold3/of3-p2-155k.pt",
            n_token=256,
        )
        pipeline.load_model()
        pipeline.compile_all()
        pipeline.patch_model()
        output = pipeline.run_inference(num_recycles=3, diff_steps=200)

    Args:
        openfold3_src_path: Path to OpenFold3 repository root
        checkpoint_path: Path to model checkpoint
        n_token: Sequence length (max 256)
        n_atom: Number of atoms (default: same as n_token)
        n_msa: Number of MSA sequences (default: 4)
        n_templ: Number of templates (default: 1)
        compiler_args: Neuron compiler arguments
    """

    def __init__(
        self,
        openfold3_src_path: str = "/home/ubuntu/openfold-3",
        checkpoint_path: str = "~/.openfold3/of3-p2-155k.pt",
        n_token: int = 256,
        n_atom: Optional[int] = None,
        n_msa: int = 4,
        n_templ: int = 1,
        compiler_args: Optional[List[str]] = None,
    ):
        self.openfold3_src_path = openfold3_src_path
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.n_token = n_token
        self.n_atom = n_atom or n_token
        self.n_msa = n_msa
        self.n_templ = n_templ
        self.compiler_args = compiler_args or ["--target", "trn2"]

        # Model (populated by load_model)
        self.model = None

        # Compiled blocks (populated by compile_all)
        self.traced_pf = None
        self.traced_msa_a = None
        self.traced_msa_b = None
        self.traced_tmpl = None
        self.traced_dc = None

        # Compilation times
        self.compile_times: Dict[str, float] = {}

    def load_model(self) -> None:
        """Load and configure OpenFold3 model.

        Applies source patches, imports the model, loads checkpoint
        weights, and configures for evaluation.
        """
        import sys
        import gc

        # Add OpenFold3 to path
        if self.openfold3_src_path not in sys.path:
            sys.path.insert(0, self.openfold3_src_path)

        # Apply source patches
        openfold3_pkg = os.path.join(self.openfold3_src_path, "openfold3")
        patches = patch_openfold3_source(openfold3_pkg)
        print(f"Applied {len(patches)} source patches")

        # Import and create model
        from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
        from openfold3.projects.of3_all_atom.model import OpenFold3
        from openfold3.core.utils.checkpoint_loading_utils import (
            load_checkpoint,
            get_state_dict_from_checkpoint,
        )

        project = OF3ProjectEntry()
        config = project.get_model_config_with_presets(
            presets=["predict", "pae_enabled"]
        )
        config.settings.memory.eval.use_deepspeed_evo_attention = False
        config.settings.memory.eval.use_cueq_triangle_kernels = False

        self.model = OpenFold3(config)
        self.model.eval()

        # Load weights
        ckpt = load_checkpoint(Path(self.checkpoint_path))
        state_dict, _ = get_state_dict_from_checkpoint(ckpt, init_from_ema_weights=True)
        bare_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        self.model.load_state_dict(bare_state_dict, strict=False)
        del ckpt, state_dict, bare_state_dict
        gc.collect()

        print("Model loaded successfully")

    def compile_all(self) -> Dict[str, float]:
        """Compile all Neuron blocks.

        Returns:
            dict mapping block names to compilation times in seconds
        """
        import torch_neuronx

        assert self.model is not None, "Call load_model() first"
        N = self.n_token
        N_MSA = self.n_msa
        N_TEMPL = self.n_templ

        # Trace inputs
        s_dummy = torch.randn(1, N, C_S)
        z_dummy = torch.randn(1, N, N, C_Z)
        single_mask = torch.ones(1, N)
        pair_mask = torch.ones(1, N, N)
        m_dummy = torch.randn(1, N_MSA, N, C_M)
        msa_mask_dummy = torch.ones(1, N_MSA, N)
        t_dummy = torch.randn(1, N_TEMPL, N, N, 64)
        t_mask_dummy = torch.ones(1, N_TEMPL, N, N)

        # --- PairFormer ---
        print("  Compiling PairFormerBlock...")
        pf_wrapper = PairFormerBlockWrapper(self.model.pairformer_stack.blocks[0])
        pf_wrapper.eval()
        t0 = time.time()
        self.traced_pf = torch_neuronx.trace(
            pf_wrapper,
            (s_dummy, z_dummy, single_mask, pair_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["pairformer"] = time.time() - t0
        print(f"    Done in {self.compile_times['pairformer']:.1f}s")

        # --- MSA type A (blocks 0-2) ---
        print("  Compiling MSA block (type A)...")
        msa_a_wrapper = MSABlockWrapper(self.model.msa_module.blocks[0])
        msa_a_wrapper.eval()
        t0 = time.time()
        self.traced_msa_a = torch_neuronx.trace(
            msa_a_wrapper,
            (m_dummy, z_dummy, msa_mask_dummy, pair_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["msa_type_a"] = time.time() - t0
        print(f"    Done in {self.compile_times['msa_type_a']:.1f}s")

        # --- MSA type B (last block) ---
        num_msa_blocks = len(self.model.msa_module.blocks)
        print("  Compiling MSA block (type B - last)...")
        msa_b_wrapper = MSABlockWrapper(
            self.model.msa_module.blocks[num_msa_blocks - 1]
        )
        msa_b_wrapper.eval()
        t0 = time.time()
        self.traced_msa_b = torch_neuronx.trace(
            msa_b_wrapper,
            (m_dummy, z_dummy, msa_mask_dummy, pair_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["msa_type_b"] = time.time() - t0
        print(f"    Done in {self.compile_times['msa_type_b']:.1f}s")

        # --- Template ---
        print("  Compiling Template block...")
        tmpl_wrapper = TemplatePairBlockWrapper(
            self.model.template_embedder.template_pair_stack.blocks[0]
        )
        tmpl_wrapper.eval()
        t0 = time.time()
        self.traced_tmpl = torch_neuronx.trace(
            tmpl_wrapper,
            (t_dummy, t_mask_dummy),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["template"] = time.time() - t0
        print(f"    Done in {self.compile_times['template']:.1f}s")

        # --- DiffusionConditioning._forward() ---
        print("  Compiling DiffusionConditioning._forward()...")
        dc_wrapper = DiffCondForwardWrapper(
            self.model.diffusion_module.diffusion_conditioning
        )
        dc_wrapper.eval()
        t0 = time.time()
        self.traced_dc = torch_neuronx.trace(
            dc_wrapper,
            (s_dummy, z_dummy, single_mask),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
        )
        self.compile_times["diff_cond"] = time.time() - t0
        print(f"    Done in {self.compile_times['diff_cond']:.1f}s")

        # Warmup
        print("  Warming up traced models...")
        for _ in range(2):
            self.traced_pf(s_dummy, z_dummy, single_mask, pair_mask)
            self.traced_msa_a(m_dummy, z_dummy, msa_mask_dummy, pair_mask)
            self.traced_msa_b(m_dummy, z_dummy, msa_mask_dummy, pair_mask)
            self.traced_tmpl(t_dummy, t_mask_dummy)
            self.traced_dc(s_dummy, z_dummy, single_mask)
        print("  All blocks compiled and warmed up.")

        return self.compile_times

    def patch_model(self) -> None:
        """Monkey-patch the OpenFold3 model to use compiled Neuron blocks.

        After calling this, the model's forward() will route PairFormer,
        MSA, Template, and DiffCond blocks through Neuron hardware while
        keeping the orchestration (recycling, diffusion, confidence) on CPU.
        """
        import torch_neuronx

        assert self.model is not None, "Call load_model() first"
        assert self.traced_pf is not None, "Call compile_all() first"

        traced_pf = self.traced_pf
        traced_msa_a = self.traced_msa_a
        traced_msa_b = self.traced_msa_b
        traced_tmpl = self.traced_tmpl
        traced_dc = self.traced_dc

        # --- PairFormer blocks ---
        num_pf_blocks = len(self.model.pairformer_stack.blocks)
        for i in range(num_pf_blocks):
            block = self.model.pairformer_stack.blocks[i]

            def make_pf_forward(block_idx, original_block):
                def neuron_forward(s, z, single_mask, pair_mask, **kwargs):
                    w = PairFormerBlockWrapper(original_block)
                    torch_neuronx.replace_weights(traced_pf, w.state_dict())
                    return traced_pf(s, z, single_mask, pair_mask)

                return neuron_forward

            block.forward = make_pf_forward(i, block)
        print(f"  Patched {num_pf_blocks} PairFormer blocks")

        # --- MSA blocks ---
        num_msa_blocks = len(self.model.msa_module.blocks)
        for i in range(num_msa_blocks):
            block = self.model.msa_module.blocks[i]

            if i < num_msa_blocks - 1:

                def make_msa_a_forward(block_idx, original_block):
                    def neuron_forward(m, z, msa_mask, pair_mask, **kwargs):
                        w = MSABlockWrapper(original_block)
                        torch_neuronx.replace_weights(traced_msa_a, w.state_dict())
                        return traced_msa_a(m, z, msa_mask, pair_mask)

                    return neuron_forward

                block.forward = make_msa_a_forward(i, block)
            else:

                def make_msa_b_forward(original_block):
                    def neuron_forward(m, z, msa_mask, pair_mask, **kwargs):
                        w = MSABlockWrapper(original_block)
                        torch_neuronx.replace_weights(traced_msa_b, w.state_dict())
                        return traced_msa_b(m, z, msa_mask, pair_mask)

                    return neuron_forward

                block.forward = make_msa_b_forward(block)
        print(f"  Patched {num_msa_blocks} MSA blocks")

        # --- Template blocks ---
        num_tmpl_blocks = len(self.model.template_embedder.template_pair_stack.blocks)
        for i in range(num_tmpl_blocks):
            block = self.model.template_embedder.template_pair_stack.blocks[i]

            def make_tmpl_forward(block_idx, original_block):
                def neuron_forward(t, mask, **kwargs):
                    w = TemplatePairBlockWrapper(original_block)
                    torch_neuronx.replace_weights(traced_tmpl, w.state_dict())
                    return traced_tmpl(t, mask)

                return neuron_forward

            block.forward = make_tmpl_forward(i, block)
        print(f"  Patched {num_tmpl_blocks} Template blocks")

        # --- DiffusionConditioning._forward() ---
        dc = self.model.diffusion_module.diffusion_conditioning

        def neuron_dc_forward(si, zij, token_mask, chunk_size=None):
            orig_shape_si = si.shape
            n_tok = orig_shape_si[-2]
            leading = orig_shape_si[:-2]
            si_flat = si.reshape(1, n_tok, -1)
            zij_flat = zij.reshape(1, n_tok, n_tok, -1)
            mask_flat = token_mask.reshape(1, n_tok)
            si_out, zij_out = traced_dc(si_flat, zij_flat, mask_flat)
            return (
                si_out.reshape(*leading, n_tok, -1),
                zij_out.reshape(*leading, n_tok, n_tok, -1),
            )

        dc._forward = neuron_dc_forward
        print("  Patched DiffusionConditioning._forward()")
        print("  All blocks monkey-patched.")

    def run_inference(
        self,
        batch: Optional[dict] = None,
        num_recycles: int = 3,
        diff_steps: int = 200,
        diff_samples: int = 1,
    ) -> Tuple[dict, dict]:
        """Run OpenFold3 inference.

        Args:
            batch: Input batch dict (if None, creates a dummy batch)
            num_recycles: Number of recycling iterations (default: 3)
            diff_steps: Number of diffusion steps (default: 200)
            diff_samples: Number of diffusion samples (default: 1)

        Returns:
            Tuple of (updated_batch, output_dict)
        """
        import copy

        assert self.model is not None, "Call load_model() first"

        if batch is None:
            batch = create_dummy_batch(
                n_token=self.n_token,
                n_atom=self.n_atom,
                n_msa=self.n_msa,
                n_templ=self.n_templ,
            )

        # Configure model
        self.model.shared.num_recycles = num_recycles
        self.model.shared.diffusion.no_full_rollout_steps = diff_steps
        self.model.shared.diffusion.no_full_rollout_samples = diff_samples

        batch_copy = copy.deepcopy(batch)

        with torch.no_grad():
            t0 = time.time()
            batch_out, output = self.model(batch_copy)
            elapsed = time.time() - t0

        print(f"  Inference completed in {elapsed:.1f}s")
        return batch_out, output
