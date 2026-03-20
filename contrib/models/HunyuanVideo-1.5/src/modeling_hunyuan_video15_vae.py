"""
HunyuanVideo-1.5 VAE Decoder for Neuron

Compiles the 871M-param VAE decoder as 24 individually-traced shards,
then chains them at runtime for full decode on Neuron hardware.

Three model rewrites enable compilation:
  1. CausalConv3d replicate pad -> constant pad (avoids SBUF-busting XLA concat)
  2. 8D DCAE rearrange -> decomposed 6D ops (avoids XLA 8D tensor crash)
  3. Full self-attention -> spatial-pooled attention (reduces 24M -> <10M instructions)

Usage:
    # Compile
    compile_vae_shards(save_dir, hf_model_path)

    # Load and run
    vae = NeuronVAEDecoder(save_dir)
    video = vae.decode(latents)  # [1,32,9,30,40] -> [1,3,33,480,640]
"""
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Neuron-compatible patches
# ---------------------------------------------------------------------------
def patch_vae_for_neuron():
    """Monkey-patch diffusers VAE classes to be Neuron-compatible."""
    import diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 as hv15

    # Fix 1: constant pad instead of replicate (avoids XLA concat exceeding SBUF)
    def _neuron_causal_forward(self, hidden_states):
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode='constant', value=0)
        return self.conv(hidden_states)
    hv15.HunyuanVideo15CausalConv3d.forward = _neuron_causal_forward

    # Fix 2: decompose 8D rearrange to sequential 6D ops
    def _safe_rearrange(tensor, r1=1, r2=2, r3=2):
        b, packed_c, f, h, w = tensor.shape
        c = packed_c // (r1 * r2 * r3)
        t = tensor.view(b, r1, r2 * r3 * c, f, h, w)
        t = t.permute(0, 2, 3, 1, 4, 5).reshape(b, r2 * r3 * c, f * r1, h, w)
        t = t.view(b, r2, r3 * c, f * r1, h, w)
        t = t.permute(0, 2, 3, 4, 1, 5).reshape(b, r3 * c, f * r1, h * r2, w)
        t = t.view(b, r3, c, f * r1, h * r2, w)
        t = t.permute(0, 2, 3, 4, 5, 1).reshape(b, c, f * r1, h * r2, w * r3)
        return t
    hv15.HunyuanVideo15Upsample._dcae_upsample_rearrange = staticmethod(_safe_rearrange)

    # Fix 3: spatial-pooled attention (10,800 tokens -> 2,700 tokens)
    def _neuron_attn_forward(self, x):
        identity = x
        x = self.norm(x)
        b, c, f, h, w = x.shape
        x_down = x.reshape(b * f, c, h, w)
        x_down = F.avg_pool2d(x_down, kernel_size=2, stride=2)
        h2, w2 = x_down.shape[2], x_down.shape[3]
        x_down = x_down.reshape(b, c, f, h2, w2)
        query = self.to_q(x_down)
        key = self.to_k(x_down)
        value = self.to_v(x_down)
        n = f * h2 * w2
        query = query.reshape(b, c, n).permute(0, 2, 1).unsqueeze(1).contiguous()
        key = key.reshape(b, c, n).permute(0, 2, 1).unsqueeze(1).contiguous()
        value = value.reshape(b, c, n).permute(0, 2, 1).unsqueeze(1).contiguous()
        attn_mask = self.prepare_causal_attention_mask(f, h2 * w2, query.dtype, query.device, batch_size=b)
        attn_mask = attn_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        out = out.squeeze(1).reshape(b, f, h2, w2, c).permute(0, 4, 1, 2, 3)
        out = out.reshape(b * f, c, h2, w2)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        out = out.reshape(b, c, f, h, w)
        out = self.proj_out(out)
        return out + identity
    hv15.HunyuanVideo15AttnBlock.forward = _neuron_attn_forward


# ---------------------------------------------------------------------------
# Shard compilation
# ---------------------------------------------------------------------------
HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

SHARD_NAMES = [
    "conv_in",
    "mid_resnet0", "mid_attn0", "mid_resnet1",
    "up0_resnet0", "up0_resnet1", "up0_resnet2", "up0_upsample",
    "up1_resnet0", "up1_resnet1", "up1_resnet2", "up1_upsample",
    "up2_resnet0", "up2_resnet1", "up2_resnet2", "up2_upsample",
    "up3_resnet0", "up3_resnet1", "up3_resnet2", "up3_upsample",
    "up4_resnet0", "up4_resnet1", "up4_resnet2",
    "norm_conv_out",
]


def compile_vae_shards(save_dir: str, hf_model_path: str = HF_MODEL_ID):
    """Compile VAE decoder as 24 individually-traced shards."""
    import torch_neuronx

    os.environ.setdefault("NEURON_CC_FLAGS", "--model-type=unet-inference -O1 --auto-cast=none")
    patch_vae_for_neuron()

    from diffusers import AutoencoderKLHunyuanVideo15
    print(f"[{time.strftime('%H:%M:%S')}] Loading VAE...", flush=True)
    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        hf_model_path, subfolder="vae", torch_dtype=torch.bfloat16).eval()
    dec = vae.decoder
    d = torch.bfloat16
    os.makedirs(save_dir, exist_ok=True)

    # Compute intermediate shapes via CPU forward
    shapes = {}
    with torch.no_grad():
        h = dec.conv_in(torch.randn(1, 32, 9, 30, 40, dtype=d))
        h = h + torch.randn(1, 32, 9, 30, 40, dtype=d).repeat_interleave(repeats=dec.repeat, dim=1)
        shapes["after_conv_in"] = list(h.shape)
        h = dec.mid_block(h)
        for i, block in enumerate(dec.up_blocks):
            shapes[f"before_up{i}"] = list(h.shape)
            h_pre = h
            for r in block.resnets:
                h_pre = r(h_pre)
            shapes[f"after_up{i}_resnets"] = list(h_pre.shape)
            h = block(h)
        shapes["before_out"] = list(h.shape)

    def _compile(name, module, example):
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Compiling {name} {list(example.shape)}...", flush=True)
        traced = torch_neuronx.trace(module, (example,))
        torch.jit.save(traced, f"{save_dir}/vae_{name}.pt")
        print(f"[{time.strftime('%H:%M:%S')}] ✓ {name} ({time.time()-t0:.0f}s)", flush=True)
        del traced; gc.collect()

    _compile("conv_in", dec.conv_in, torch.randn(1, 32, 9, 30, 40, dtype=d))
    h_mid = torch.randn(*shapes["after_conv_in"], dtype=d)
    _compile("mid_resnet0", dec.mid_block.resnets[0], h_mid)
    _compile("mid_attn0", dec.mid_block.attentions[0], h_mid)
    _compile("mid_resnet1", dec.mid_block.resnets[1], h_mid)

    for i in range(5):
        h_shape = shapes[f"before_up{i}"]
        block = dec.up_blocks[i]
        for j, resnet in enumerate(block.resnets):
            _compile(f"up{i}_resnet{j}", resnet, torch.randn(*h_shape, dtype=d))
        if block.upsamplers:
            _compile(f"up{i}_upsample", block.upsamplers[0], torch.randn(*shapes[f"after_up{i}_resnets"], dtype=d))

    class NormConvOut(nn.Module):
        def __init__(self, norm, act, conv):
            super().__init__()
            self.norm, self.act, self.conv = norm, act, conv
        def forward(self, x):
            return self.conv(self.act(self.norm(x)))

    _compile("norm_conv_out",
             NormConvOut(dec.norm_out, dec.conv_act, dec.conv_out).bfloat16().eval(),
             torch.randn(*shapes["before_out"], dtype=d))

    print(f"All 24 VAE shards compiled to {save_dir}/")


# ---------------------------------------------------------------------------
# Runtime: sharded VAE decoder
# ---------------------------------------------------------------------------
class NeuronVAEDecoder:
    """Runs VAE decoder on Neuron using 24 pre-compiled shards."""

    REPEAT = 32  # decoder.repeat = in_channels (1024) // latent_channels (32)

    def __init__(self, shard_dir: str, scaling_factor: float = 0.476986):
        self.scaling_factor = scaling_factor
        self.shards = {}
        shard_dir = Path(shard_dir)
        for name in SHARD_NAMES:
            path = shard_dir / f"vae_{name}.pt"
            self.shards[name] = torch.jit.load(str(path))
        print(f"Loaded {len(self.shards)} VAE shards from {shard_dir}")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents [1,32,9,30,40] -> video [1,3,33,480,640]."""
        z = latents / self.scaling_factor
        h = self.shards["conv_in"](z)
        h = h + z.repeat_interleave(repeats=self.REPEAT, dim=1)

        h = self.shards["mid_resnet0"](h)
        h = self.shards["mid_attn0"](h)
        h = self.shards["mid_resnet1"](h)

        for i in range(5):
            for j in range(3):
                h = self.shards[f"up{i}_resnet{j}"](h)
            if i < 4:
                h = self.shards[f"up{i}_upsample"](h)

        return self.shards["norm_conv_out"](h)


def load_traced_vae(shard_dir: str, scaling_factor: float = 0.476986) -> NeuronVAEDecoder:
    """Load and return a NeuronVAEDecoder."""
    return NeuronVAEDecoder(shard_dir, scaling_factor)
