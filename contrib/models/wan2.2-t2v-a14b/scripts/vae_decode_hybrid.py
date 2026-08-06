"""
Hybrid VAE decode: frames 0+1 on CPU, frames 2+3 on Neuron.

Usage (subprocess):
  NEURON_RT_VISIBLE_CORES=20-21 python vae_decode_hybrid.py <latents_path> <output_path>
"""
import torch, torch_neuronx, os, sys, time
import torch.nn.functional as F
import diffusers.models.autoencoders.autoencoder_kl_wan as vm

# Patches
_op = F.pad
def _sp(i, p, mode='constant', value=0):
    """Patch F.pad to avoid unsupported replicate mode on 5D tensors."""
    if mode == 'replicate' and i.dim() == 5: mode = 'constant'
    return _op(i, p, mode=mode, value=value)
F.pad = _sp
_oi = torch.nn.functional.interpolate
def _si(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
        recompute_scale_factor=None, antialias=False):
    """Patch F.interpolate to replace nearest-exact with nearest."""
    if mode == 'nearest-exact': mode = 'nearest'
    return _oi(input, size=size, scale_factor=scale_factor, mode=mode,
               align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)
F.interpolate = _si
vm.CACHE_T = 1

from diffusers import AutoencoderKLWan


def main():
    """Decode latents using hybrid CPU+Neuron VAE pipeline."""
    latents_path = sys.argv[1]
    output_path = sys.argv[2]

    MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    CACHE_DIR = os.environ.get("CACHE_DIR", "/mnt/work/.cache")
    compiled_path = os.environ.get("COMPILED_VAE_PATH", "/mnt/work/wan2.2-lint-fix/compiled_vae_new/decoder_cached.pt")

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae",
        torch_dtype=torch.float32, cache_dir=CACHE_DIR)
    vae.eval()

    traced = torch.jit.load(compiled_path)

    latents = torch.load(latents_path, weights_only=True)
    lv = latents.to(torch.float32)
    lm = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    z = lv / ls + lm

    # Frames 0+1 on CPU
    z_pqc = vae.post_quant_conv(z)
    vae.clear_cache()
    vae._conv_idx = [0]
    with torch.no_grad():
        out = vae.decoder(z_pqc[:,:,0:1,:,:], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)
    vae._conv_idx = [0]
    with torch.no_grad():
        out = torch.cat([out, vae.decoder(z_pqc[:,:,1:2,:,:], feat_cache=vae._feat_map, feat_idx=vae._conv_idx)], 2)

    # Frames 2+3 on Neuron
    caches = [c.clone() for c in vae._feat_map[:32]]
    r2 = traced(z[:,:,2:3,:,:], *caches)
    out = torch.cat([out, r2[0]], 2)
    r3 = traced(z[:,:,3:4,:,:], *list(r2[1:]))
    out = torch.cat([out, r3[0]], 2)

    torch.save(out, output_path)
    print(f"DONE: {out.shape}")


if __name__ == '__main__':
    main()
