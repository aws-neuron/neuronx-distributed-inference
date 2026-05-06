#!/usr/bin/env python3
"""
Integration tests for WAN 2.1 T2V 1.3B NeuronX implementation.
"""

import pytest
import torch
import torch_neuronx  # must import before torch.jit.load for Neuron models
import torch.nn.functional as F
import json
import os
import time
from pathlib import Path

MODEL_PATH = os.environ.get("WAN_MODEL_DIR", "/home/ubuntu/models/wan2.1-t2v-1.3b/")
COMPILED_PATH = os.environ.get("WAN_COMPILED_DIR", "/home/ubuntu/neuron_models/wan2.1-t2v-1.3b/")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_wan import (
    WanBackboneWrapper, T5Wrapper,
    ConvInCached, BlockCached, NormConvOutCached, NoCacheWrapper,
    VAE_BLOCK_ORDER, make_decoder_xla_compatible,
)


@pytest.fixture(scope="module")
def vae():
    from diffusers import AutoencoderKLWan
    v = AutoencoderKLWan.from_pretrained(f"{MODEL_PATH}/vae", torch_dtype=torch.bfloat16)
    v.eval()
    make_decoder_xla_compatible(v.decoder)
    return v


@pytest.fixture(scope="module")
def traced_backbone():
    p = Path(COMPILED_PATH) / "traced_wan_480x832.pt"
    if not p.exists():
        pytest.skip(f"Traced backbone not found: {p}")
    return torch.jit.load(str(p))


@pytest.fixture(scope="module")
def traced_vae_blocks():
    d = Path(COMPILED_PATH) / "vae_blocks_cached"
    if not d.exists():
        pytest.skip(f"VAE blocks not found: {d}")
    return {f.stem: torch.jit.load(str(f)) for f in sorted(d.glob("*.pt"))}


@pytest.fixture(scope="module")
def rope():
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    m = WanTransformer3DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
    rc, rs = m.rope(torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16))
    del m
    return rc, rs


def test_backbone_loads(traced_backbone):
    """Traced backbone loads successfully."""
    assert traced_backbone is not None


def test_backbone_output_shape(traced_backbone, rope):
    """Backbone produces correct output shape."""
    h = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    t = torch.tensor([999], dtype=torch.int64)
    e = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    rc, rs = rope
    out = traced_backbone(h, t, e, rc, rs)
    assert out.shape == (1, 16, 4, 60, 104)


def test_backbone_cosine_vs_cpu(traced_backbone, rope):
    """Backbone output matches CPU reference (cosine > 0.999)."""
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
    model.eval()
    wrapper = WanBackboneWrapper(model)

    h = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    t = torch.tensor([999], dtype=torch.int64)
    e = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    rc, rs = rope

    with torch.no_grad():
        cpu_out = wrapper(h, t, e, rc, rs)
    neuron_out = traced_backbone(h, t, e, rc, rs)

    cos = F.cosine_similarity(cpu_out.flatten().float(), neuron_out.flatten().float(), dim=0)
    assert cos.item() > 0.999, f"Backbone cosine {cos.item():.6f} < 0.999"


def test_vae_blocks_load(traced_vae_blocks):
    """All 18 VAE blocks load."""
    assert len(traced_vae_blocks) == 18


def test_vae_hybrid_decode_cosine(vae, traced_vae_blocks):
    """Hybrid VAE decode matches CPU reference (cosine > 0.999)."""
    z = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    lm = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    x = vae.post_quant_conv(z / ls + lm)

    # CPU reference
    vae.clear_cache()
    cpu_frames = []
    for i in range(4):
        vae._conv_idx = [0]
        kw = dict(first_chunk=True) if i == 0 else {}
        with torch.no_grad():
            cpu_frames.append(vae.decoder(x[:, :, i:i+1, :, :],
                              feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kw))
    cpu_video = torch.cat(cpu_frames, dim=2)

    # Hybrid: CPU 0+1, Neuron 2+3
    vae.clear_cache()
    frames = []
    for i in range(2):
        vae._conv_idx = [0]
        kw = dict(first_chunk=True) if i == 0 else {}
        with torch.no_grad():
            frames.append(vae.decoder(x[:, :, i:i+1, :, :],
                          feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kw))

    cache = [v.clone() if isinstance(v, torch.Tensor) else v for v in vae._feat_map]
    for fi in range(2, 4):
        h = x[:, :, fi:fi+1, :, :]
        for name, cs, cc in VAE_BLOCK_ORDER:
            c_in = tuple(cache[cs + i] for i in range(cc))
            result = traced_vae_blocks[name](h, *c_in)
            if cc > 0:
                h = result[0]
                for i in range(cc):
                    cache[cs + i] = result[1 + i]
            else:
                h = result
        frames.append(h)

    neuron_video = torch.cat(frames, dim=2)
    cos = F.cosine_similarity(cpu_video.flatten().float(), neuron_video.flatten().float(), dim=0)
    assert cos.item() > 0.999, f"VAE cosine {cos.item():.6f} < 0.999"


def test_vae_output_frame_count(vae, traced_vae_blocks):
    """Hybrid decode produces 13 output frames from 4 latent frames."""
    z = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    lm = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(torch.bfloat16)
    x = vae.post_quant_conv(z / ls + lm)

    vae.clear_cache()
    frames = []
    for i in range(2):
        vae._conv_idx = [0]
        kw = dict(first_chunk=True) if i == 0 else {}
        with torch.no_grad():
            frames.append(vae.decoder(x[:, :, i:i+1, :, :],
                          feat_cache=vae._feat_map, feat_idx=vae._conv_idx, **kw))

    cache = [v.clone() if isinstance(v, torch.Tensor) else v for v in vae._feat_map]
    for fi in range(2, 4):
        h = x[:, :, fi:fi+1, :, :]
        for name, cs, cc in VAE_BLOCK_ORDER:
            c_in = tuple(cache[cs + i] for i in range(cc))
            result = traced_vae_blocks[name](h, *c_in)
            if cc > 0:
                h = result[0]
                for i in range(cc):
                    cache[cs + i] = result[1 + i]
            else:
                h = result
        frames.append(h)

    video = torch.cat(frames, dim=2)
    assert video.shape == (1, 3, 13, 480, 832), f"Expected (1,3,13,480,832), got {video.shape}"


if __name__ == "__main__":
    print("Run with: pytest", __file__, "--capture=tee-sys")
