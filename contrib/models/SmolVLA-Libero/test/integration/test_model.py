#!/usr/bin/env python3
"""
Integration tests for the SmolVLA-Libero NeuronX port.

Tests:

    * ``test_smoke_synthetic_chunk`` — one forward pass through all three
      NEFFs with synthetic inputs; checks shape, finiteness, non-zero variance.
    * ``test_warm_latency`` — light p50 latency sanity bound.
    * ``test_lerobot_cpu_neuron_parity`` — Neuron vs upstream lerobot CPU
      action-chunk parity. Loads ``lerobot.SmolVLAPolicy`` from the same HF
      checkpoint, runs a CPU forward with identical inputs and identical
      seeded initial noise, and asserts cos-sim ≥ 0.99 against the Neuron
      output. This is the SmolVLA equivalent of the logit validation NxDI
      uses for CausalLM contrib models — it validates that the Neuron port
      reproduces the reference implementation, not just self-consistency.
      Skipped automatically if ``lerobot`` is not installed.

Required environment variables:

    SMOLVLA_CKPT  Path to the HuggingFaceVLA/smolvla_libero snapshot directory.
    SMOLVLA_NEFF  Output directory for the three compiled NEFFs. If it does
                  not yet contain compiled artifacts, the test will compile
                  them (≈ 90 s on trn3pd98.3xlarge).

Run:

    pytest contrib/models/SmolVLA-Libero/test/integration/test_model.py --capture=tee-sys

or directly:

    cd contrib/models/SmolVLA-Libero
    python test/integration/test_model.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Make ``src/`` importable without a package install.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config_constants as C  # noqa: E402
from modeling_smolvla import (  # noqa: E402
    DENOISE_NEFF_SUBDIR,
    PREFIX_NEFF_SUBDIR,
    VISION_NEFF_SUBDIR,
    SmolVLAPolicy,
)
from neuron_action_head_base import COMPILED_MODEL_FILE_NAME  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration via env vars
# ---------------------------------------------------------------------------

CKPT_ENV = "SMOLVLA_CKPT"
NEFF_ENV = "SMOLVLA_NEFF"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set; skipping SmolVLA integration test")
    return value


def _all_neffs_present(neff_root: str) -> bool:
    for sub in (VISION_NEFF_SUBDIR, PREFIX_NEFF_SUBDIR, DENOISE_NEFF_SUBDIR):
        if not (Path(neff_root) / sub / COMPILED_MODEL_FILE_NAME).is_file():
            return False
    return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_checkpoint() -> str:
    return _require_env(CKPT_ENV)


@pytest.fixture(scope="module")
def neff_dir() -> str:
    return _require_env(NEFF_ENV)


@pytest.fixture(scope="module")
def policy(hf_checkpoint: str, neff_dir: str) -> SmolVLAPolicy:
    p = SmolVLAPolicy(
        hf_checkpoint_dir=hf_checkpoint,
        tp_degree=C.DEFAULT_TP_DEGREE,
        batch_size=C.BATCH_SIZE,
    )
    if not _all_neffs_present(neff_dir):
        t0 = time.monotonic()
        p.compile(neff_dir)
        print(f"[smolvla] compile: {time.monotonic() - t0:.1f}s -> {neff_dir}")
    t0 = time.monotonic()
    p.load(neff_dir)
    print(f"[smolvla] load: {time.monotonic() - t0:.1f}s")
    return p


# ---------------------------------------------------------------------------
# Synthetic inputs (deterministic)
# ---------------------------------------------------------------------------


def _make_synthetic_inputs(batch_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = [
        torch.randn(
            batch_size,
            3,
            C.VISION_IMAGE_SIZE,
            C.VISION_IMAGE_SIZE,
            generator=g,
            dtype=torch.bfloat16,
        )
        for _ in range(C.NUM_CAMERAS)
    ]
    lang = torch.randint(
        0,
        C.VLM_VOCAB_SIZE,
        (batch_size, C.NUM_TEXT_TOKENS),
        generator=g,
        dtype=torch.int32,
    )
    state = torch.zeros(batch_size, C.MAX_STATE_DIM, dtype=torch.float32)
    return images, lang, state


# ---------------------------------------------------------------------------
# CPU reference (lerobot)
# ---------------------------------------------------------------------------


def _load_lerobot_reference_policy(hf_checkpoint: str):
    """Load the upstream ``lerobot`` SmolVLA policy from the same checkpoint.

    This is the *reference implementation* — the model the NxDI port is
    expected to match. We compare the Neuron output against this CPU
    forward pass, which is the SmolVLA equivalent of the logit validation
    NxDI uses for CausalLM contrib models.
    """
    lerobot = pytest.importorskip(
        "lerobot",
        reason="lerobot is required for the CPU reference accuracy test",
    )
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy as LerobotSmolVLAPolicy,
    )

    pol = LerobotSmolVLAPolicy.from_pretrained(hf_checkpoint).cpu().eval()
    return pol


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().to(torch.float32)
    bf = b.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(af, bf, dim=0).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smoke_synthetic_chunk(policy: SmolVLAPolicy):
    """One forward pass through all three NEFFs with synthetic inputs."""
    images, lang, state = _make_synthetic_inputs(batch_size=C.BATCH_SIZE)
    chunk = policy.generate(images, lang, state)

    expected = (C.BATCH_SIZE, C.ACTION_CHUNK_SIZE, C.MAX_ACTION_DIM)
    assert tuple(chunk.shape) == expected, (
        f"Expected action chunk shape {expected}, got {tuple(chunk.shape)}"
    )

    assert torch.isfinite(chunk).all(), "action chunk contains NaN or Inf"
    assert chunk.std().item() > 0.0, "action chunk has zero variance — graph likely failed silently"


def test_warm_latency(policy: SmolVLAPolicy):
    """Light p50 latency check — sanity bound, not a benchmark."""
    images, lang, state = _make_synthetic_inputs(batch_size=C.BATCH_SIZE, seed=1)

    # Warm-up
    policy.generate(images, lang, state)

    timings_ms = []
    for _ in range(5):
        t0 = time.monotonic()
        policy.generate(images, lang, state)
        timings_ms.append((time.monotonic() - t0) * 1000.0)
    timings_ms.sort()
    p50 = timings_ms[len(timings_ms) // 2]
    print(f"[smolvla] warm p50 latency: {p50:.1f} ms over {len(timings_ms)} iters")

    # Generous upper bound — the full pipeline runs in ~65 ms warm on
    # trn3pd98.3xlarge. 1 s catches "something is dramatically wrong" without
    # being flaky on slower hardware or under load.
    assert p50 < 1000.0, f"warm p50 latency unexpectedly high: {p50:.1f} ms"


def test_lerobot_cpu_neuron_parity(policy: SmolVLAPolicy, hf_checkpoint: str):
    """Neuron vs upstream lerobot CPU action-chunk parity (NxDI accuracy check).

    The lerobot ``SmolVLAPolicy`` is the reference implementation the NxDI
    port targets. We load it from the same checkpoint, run a CPU forward
    with identical inputs and identical initial noise, and assert the Neuron
    output matches via cosine similarity and mean abs diff.
    """
    B = C.BATCH_SIZE

    # Synthetic inputs — fp32 floats for image pixels so they feed both paths
    # cleanly. Seeded ``Generator`` so each test run is reproducible.
    g = torch.Generator().manual_seed(2)
    images_fp32 = [
        torch.randn(
            B, 3, C.VISION_IMAGE_SIZE, C.VISION_IMAGE_SIZE,
            generator=g, dtype=torch.float32,
        )
        for _ in range(C.NUM_CAMERAS)
    ]
    lang = torch.randint(
        0, C.VLM_VOCAB_SIZE, (B, C.NUM_TEXT_TOKENS),
        generator=g, dtype=torch.long,
    )
    lang_mask = torch.ones(B, C.NUM_TEXT_TOKENS, dtype=torch.bool)
    state = torch.zeros(B, C.MAX_STATE_DIM, dtype=torch.float32)

    # Shared initial noise — fed to both paths so the only numerical
    # difference is the model implementation, not the random starting point.
    noise = torch.randn(
        B, C.ACTION_CHUNK_SIZE, C.MAX_ACTION_DIM,
        generator=torch.Generator().manual_seed(123), dtype=torch.float32,
    )

    # --- Reference: lerobot CPU ---
    lerobot_pol = _load_lerobot_reference_policy(hf_checkpoint)
    img_masks = [torch.ones(B, dtype=torch.bool) for _ in images_fp32]
    with torch.no_grad():
        chunk_cpu = lerobot_pol.model.sample_actions(
            images=images_fp32,
            img_masks=img_masks,
            lang_tokens=lang,
            lang_masks=lang_mask,
            state=state,
            noise=noise,
        )

    # --- Neuron port ---
    images_neuron = [img.to(torch.bfloat16) for img in images_fp32]
    chunk_neuron = policy.generate(
        images_neuron,
        lang.to(torch.int32),
        state,
        lang_mask=lang_mask,
        noise=noise,
    ).cpu()

    cos = _cos_sim(chunk_neuron, chunk_cpu)
    max_abs = (chunk_neuron - chunk_cpu).abs().max().item()
    mean_abs = (chunk_neuron - chunk_cpu).abs().mean().item()
    print(
        f"[smolvla] Neuron vs lerobot CPU parity: "
        f"cos_sim={cos:.6f} max_abs={max_abs:.4f} mean_abs={mean_abs:.4f}"
    )

    # Cosine similarity is the primary acceptance criterion: it is invariant
    # to bf16 magnitude noise and accumulated rounding across the 10 Euler
    # steps. The README documents 0.9999 vs lerobot CPU on real LIBERO
    # inputs; we use a slightly looser bound (0.99) for synthetic inputs
    # which can amplify low-magnitude divergence. Mean abs diff catches
    # systemic divergence.
    assert cos >= 0.99, f"Neuron vs lerobot cos_sim too low: {cos:.6f}"
    assert mean_abs < 0.05, f"Neuron vs lerobot mean abs diff too high: {mean_abs:.4f}"


# ---------------------------------------------------------------------------
# Allow `python test/integration/test_model.py` invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--capture=tee-sys"]))
