"""
Trace VAE decoder blocks with explicit cache I/O for the cached decode path.

Each block is wrapped to take (x, cache_0, ..., cache_n) as inputs
and return (output, updated_cache_0, ..., updated_cache_n) as outputs.

Two variants per block:
- frame0: cache inputs are zeros (first frame, no prior context)
- frame_n: cache inputs are [1, C, 2, H, W] (subsequent frames)
"""
import torch, torch_neuronx, os, time, sys
import torch.nn.functional as F
import diffusers.models.autoencoders.autoencoder_kl_wan as vm
vm.CACHE_T = 1  # Reduce temporal padding to avoid compiler "value out of range" error
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

os.environ["NEURON_RT_VISIBLE_CORES"] = "20-21"
sys.path.insert(0, os.environ.get("WAN_PORT_DIR", "/mnt/work/wan2.2-port"))

# Patches
_orig_pad = F.pad
def _safe_pad(input, pad, mode='constant', value=0):
    """Patch F.pad to avoid unsupported replicate mode on 5D tensors."""
    if mode == 'replicate' and input.dim() == 5: mode = 'constant'
    return _orig_pad(input, pad, mode=mode, value=value)
F.pad = _safe_pad

_orig_interp = torch.nn.functional.interpolate
def _safe_interp(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                 recompute_scale_factor=None, antialias=False):
    """Patch F.interpolate to replace nearest-exact with nearest."""
    if mode == 'nearest-exact': mode = 'nearest'
    return _orig_interp(input, size=size, scale_factor=scale_factor, mode=mode,
                        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                        antialias=antialias)
F.interpolate = _safe_interp


class CachedDecoder(torch.nn.Module):
    """Full decoder with explicit cache I/O for single-frame input."""
    def __init__(self, decoder, post_quant_conv):
        """Initialize with decoder and post-quantization convolution modules."""
        super().__init__()
        self.decoder = decoder
        self.pqc = post_quant_conv

    def forward(self, z_frame, *caches):
        """Run decoder on a single frame with explicit cache tensors."""
        # z_frame: [1, 16, 1, H, W]
        # caches: 32 tensors, each [1, C, 2, H, W]
        feat_cache = list(caches)
        feat_idx = [0]
        x = self.pqc(z_frame)
        out = self.decoder(x, feat_cache=feat_cache, feat_idx=feat_idx)
        # Return output + all 32 updated caches
        return (out,) + tuple(feat_cache[:32])


def main():
    """Trace the VAE cached decoder and save the compiled model."""
    cache_dir = os.environ.get("CACHE_DIR", "/mnt/work/.cache")
    save_dir = os.environ.get("VAE_SAVE_DIR", "/mnt/work/wan2.2-port/compiled_vae_cached")

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        subfolder="vae", torch_dtype=torch.float32, cache_dir=cache_dir)
    vae.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Get cache shapes after frame 1 (stable state: all [1,C,2,H,W])
    z = torch.randn(1, 16, 4, 60, 104)
    z_pqc = vae.post_quant_conv(z)
    vae.clear_cache()
    vae._conv_idx = [0]
    with torch.no_grad():
        vae.decoder(z_pqc[:,:,0:1,:,:], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)
        vae._conv_idx = [0]
        vae.decoder(z_pqc[:,:,1:2,:,:], feat_cache=vae._feat_map, feat_idx=vae._conv_idx)

    # Now all caches are [1,C,2,H,W]
    cache_shapes = []
    for c in vae._feat_map:
        if isinstance(c, torch.Tensor):
            cache_shapes.append(tuple(c.shape))
        else:
            cache_shapes.append(None)

    # Block definitions: (name, module, input_shape, cache_indices)
    # We trace for the "frame_n" variant (2-frame cache, stable state)
    block_defs = [
        ("conv_in", vae.decoder.conv_in, (1,16,1,60,104), [0]),
        ("mid_block", vae.decoder.mid_block, (1,384,1,60,104), list(range(1,5))),
        ("up_block_0", vae.decoder.up_blocks[0], (1,384,1,60,104), list(range(5,12))),
        ("up_block_1", vae.decoder.up_blocks[1], (1,192,1,120,208), list(range(12,19))),
        ("up_block_2", vae.decoder.up_blocks[2], (1,192,1,240,416), list(range(19,25))),
        ("up_block_3", vae.decoder.up_blocks[3], (1,96,1,480,832), list(range(25,31))),
        ("conv_out", vae.decoder.conv_out, (1,96,1,480,832), [31]),
    ]

    wrapper = CachedDecoder(vae.decoder, vae.post_quant_conv)

    # Build example inputs: z_frame + 32 cache tensors
    example_inputs = [torch.randn(1, 16, 1, 60, 104, dtype=torch.float32)]
    for i in range(32):
        if cache_shapes[i] is not None:
            example_inputs.append(torch.randn(*cache_shapes[i], dtype=torch.float32))
        else:
            # Cache 32 is None/unused, use a dummy
            example_inputs.append(torch.zeros(1, 1, 1, 1, 1, dtype=torch.float32))

    # Test on CPU first
    print("Testing cached decoder on CPU...", flush=True)
    with torch.no_grad():
        result = wrapper(*example_inputs)
    print(f"Output: {result[0].shape}, caches returned: {len(result)-1}")

    # Trace
    print("Tracing cached decoder (33 inputs)...", flush=True)
    t0 = time.time()
    try:
        traced = torch_neuronx.trace(wrapper, tuple(example_inputs),
            compiler_args="--model-type=unet-inference -O1")
        print(f"SUCCESS in {time.time()-t0:.0f}s!")
        torch.jit.save(traced, os.path.join(save_dir, "decoder_cached.pt"))
    except Exception as e:
        print(f"FAILED in {time.time()-t0:.0f}s: {str(e)[:300]}")


if __name__ == '__main__':
    main()
