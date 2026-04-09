"""
LongCat-Image-Edit Inference Script for AWS Trainium2

Runs the LongCat-Image-Edit model ENTIRELY on Neuron devices.
All components (Text Encoder, FLUX Transformer, VAE) run on Trainium2.

Components:
- Text Encoder (Qwen2.5-VL): Vision encoder + Language model (TP=4)
- Transformer: LongCatImageTransformer2DModel (FLUX-style, TP=4, CP=2)
- VAE: 2D AutoencoderKL (single device)

Usage:
    # Single image editing:
    NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
        --image input.jpg --prompt "change the sky to sunset"

    # With warmup:
    NEURON_RT_NUM_CORES=8 python run_longcat_image_edit.py \
        --image input.jpg --prompt "make it look like a painting" --warmup
"""

import os

# ============================================================================
# CRITICAL: Set Neuron environment variables BEFORE any other imports
# ============================================================================
# TP_DEGREE controls NxD world size. Use 4 for TP-only, 8 for TP+CP.
TP_DEGREE = int(os.environ.get("LONGCAT_WORLD_SIZE", "4"))

os.environ["LOCAL_WORLD_SIZE"] = str(TP_DEGREE)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

print(f"Neuron runtime configured: world_size={TP_DEGREE}, LNC=2")

import argparse
import contextlib
import json
import random
import time

import numpy as np
import torch
import torch_neuronx
import neuronx_distributed
from PIL import Image

from diffusers import LongCatImageEditPipeline
from diffusers.utils import load_image

# Patch xm.mark_step() to no-op: the diffusers pipeline calls it inside the
# denoising loop, which attempts to synchronize ALL 64 NeuronCores on the
# machine. Since we only use a subset (e.g. 4 or 8), this hangs.
# The NxDModel handles its own synchronization internally.
try:
    import torch_xla.core.xla_model as xm
    xm.mark_step = lambda *args, **kwargs: None
except ImportError:
    pass

from neuron_commons import NeuronTextEncoderWrapper

# Import NxDModel for NxDModel API loading
try:
    from neuronx_distributed.trace.nxd_model.nxd_model import NxDModel
    NXD_MODEL_AVAILABLE = True
except ImportError:
    NXD_MODEL_AVAILABLE = False
    print("WARNING: NxDModel not available.")

# Constants
COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed set to: {seed}")


class NeuronTransformerWrapper(torch.nn.Module):
    """
    Wrapper for Compiled compiled LongCat FLUX transformer on Trainium2.

    Handles:
    - Accepting txt_ids/img_ids from pipeline for RoPE computation
    - Padding hidden_states to expected_img_seq
    - Padding encoder_hidden_states to expected_txt_seq
    - Extracting target image patches from output
    """
    def __init__(self, original_transformer, nxd_model,
                 pos_embed, patch_h, patch_w,
                 expected_img_patches=8192, expected_txt_seq=1024,
                 target_patches=4096, batch_size=1):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.nxd_model = nxd_model

        # Keep pos_embed for RoPE computation from pipeline-provided position IDs
        self.pos_embed = pos_embed
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.expected_img_patches = expected_img_patches
        self.expected_txt_seq = expected_txt_seq
        self.target_patches = target_patches
        self.compiled_batch_size = batch_size

        # Cache RoPE keyed by (txt_len, img_len) to avoid recomputing
        self._rope_cache = {}

    @contextlib.contextmanager
    def cache_context(self, name: str):
        yield

    def _compute_rope_from_ids(self, txt_ids, img_ids):
        """
        Compute RoPE from pipeline-provided position IDs.

        Args:
            txt_ids: [txt_seq, 3] text position IDs (modality, row, col)
            img_ids: [img_seq, 3] image position IDs (modality, row, col)

        Returns:
            (txt_cos, txt_sin, img_cos, img_sin) padded to compiled sizes
        """
        actual_txt = txt_ids.shape[0]
        actual_img = img_ids.shape[0]
        cache_key = (actual_txt, actual_img)

        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        # Pad txt_ids to expected_txt_seq
        if actual_txt < self.expected_txt_seq:
            # Pad with continuing text positions (modality=0, incrementing row/col)
            pad_len = self.expected_txt_seq - actual_txt
            pad_ids = torch.zeros(pad_len, 3, dtype=txt_ids.dtype, device=txt_ids.device)
            last_row = txt_ids[-1, 1].item() if actual_txt > 0 else 0
            for i in range(pad_len):
                pad_ids[i, 0] = 0  # modality = text
                pad_ids[i, 1] = last_row + 1 + i
                pad_ids[i, 2] = last_row + 1 + i
            txt_ids_padded = torch.cat([txt_ids, pad_ids], dim=0)
        else:
            txt_ids_padded = txt_ids[:self.expected_txt_seq]

        # Pad img_ids to expected_img_patches
        if actual_img < self.expected_img_patches:
            pad_n = self.expected_img_patches - actual_img
            img_ids_padded = torch.cat(
                [img_ids, img_ids[-1:].expand(pad_n, -1)], dim=0)
        else:
            img_ids_padded = img_ids[:self.expected_img_patches]

        with torch.no_grad():
            txt_cos, txt_sin = self.pos_embed(txt_ids_padded)
            img_cos, img_sin = self.pos_embed(img_ids_padded)

        rope = (
            txt_cos.to(torch.bfloat16),
            txt_sin.to(torch.bfloat16),
            img_cos.to(torch.bfloat16),
            img_sin.to(torch.bfloat16),
        )
        self._rope_cache[cache_key] = rope
        return rope

    def _compute_rope_fallback(self, actual_txt_len):
        """Fallback RoPE computation when txt_ids/img_ids not provided."""
        cache_key = ("fallback", actual_txt_len)
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids

        text_ids = prepare_pos_ids(
            modality_id=0, type="text", num_token=self.expected_txt_seq)
        target_ids = prepare_pos_ids(
            modality_id=1, type="image",
            start=(actual_txt_len, actual_txt_len),
            height=self.patch_h, width=self.patch_w)
        source_ids = prepare_pos_ids(
            modality_id=2, type="image",
            start=(actual_txt_len, actual_txt_len),
            height=self.patch_h, width=self.patch_w)
        img_ids = torch.cat([target_ids, source_ids], dim=0)

        return self._compute_rope_from_ids(text_ids, img_ids)

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, txt_ids=None, img_ids=None,
                return_dict=False, **kwargs):
        """
        Forward pass using compiled Compiled transformer.

        hidden_states: [B, img_patches, 64] -- packed latents for target+source
        encoder_hidden_states: [B, txt_seq, 3584] -- text embeddings
        timestep: [B] -- denoising timestep
        txt_ids: [txt_seq, 3] -- text position IDs from pipeline (optional)
        img_ids: [img_seq, 3] -- image position IDs from pipeline (optional)
        """
        batch_size = hidden_states.shape[0]
        actual_txt_len = encoder_hidden_states.shape[1]

        # Compute RoPE from pipeline-provided position IDs or fallback
        if txt_ids is not None and img_ids is not None:
            txt_cos, txt_sin, img_cos, img_sin = self._compute_rope_from_ids(txt_ids, img_ids)
        else:
            txt_cos, txt_sin, img_cos, img_sin = self._compute_rope_fallback(actual_txt_len)

        # Pad hidden_states (image patches)
        actual_img = hidden_states.shape[1]
        if actual_img < self.expected_img_patches:
            pad = torch.zeros(
                (batch_size, self.expected_img_patches - actual_img, hidden_states.shape[2]),
                dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, pad], dim=1)
        elif actual_img > self.expected_img_patches:
            hidden_states = hidden_states[:, :self.expected_img_patches, :]

        # Pad encoder_hidden_states (text)
        if actual_txt_len < self.expected_txt_seq:
            pad = torch.zeros(
                (batch_size, self.expected_txt_seq - actual_txt_len, encoder_hidden_states.shape[2]),
                dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
            encoder_hidden_states = torch.cat([encoder_hidden_states, pad], dim=1)
        elif actual_txt_len > self.expected_txt_seq:
            encoder_hidden_states = encoder_hidden_states[:, :self.expected_txt_seq, :]

        # Batch padding
        if batch_size < self.compiled_batch_size:
            pad_batch = self.compiled_batch_size - batch_size
            hidden_states = torch.cat([
                hidden_states,
                torch.zeros((pad_batch,) + hidden_states.shape[1:],
                           dtype=hidden_states.dtype, device=hidden_states.device)
            ], dim=0)
            encoder_hidden_states = torch.cat([
                encoder_hidden_states,
                torch.zeros((pad_batch,) + encoder_hidden_states.shape[1:],
                           dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
            ], dim=0)
            timestep = torch.cat([
                timestep,
                torch.zeros(pad_batch, dtype=timestep.dtype, device=timestep.device)
            ], dim=0)

        timestep = timestep.to(torch.float32)

        # Run Compiled model
        output = self.nxd_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
        )

        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Remove batch padding
        if batch_size < self.compiled_batch_size:
            output_tensor = output_tensor[:batch_size]

        # Extract target image patches (first target_patches from output)
        output_tensor = output_tensor[:, :self.target_patches, :]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output_tensor)
        return (output_tensor,)


class NeuronTransformerWrapperCFG(torch.nn.Module):
    """
    Wrapper for CFG Parallel compiled LongCat FLUX transformer on Trainium2.

    Similar to NeuronTransformerWrapper but expects batch_size=2 input
    (negative + positive prompt embeddings batched together).
    No batch padding needed since CFG always uses exactly 2 batch items.
    """
    def __init__(self, original_transformer, nxd_model,
                 pos_embed, patch_h, patch_w,
                 expected_img_patches=8192, expected_txt_seq=1024,
                 target_patches=4096):
        super().__init__()
        self.config = original_transformer.config
        self.dtype = original_transformer.dtype
        self.device = original_transformer.device
        self.nxd_model = nxd_model

        self.pos_embed = pos_embed
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.expected_img_patches = expected_img_patches
        self.expected_txt_seq = expected_txt_seq
        self.target_patches = target_patches
        self.compiled_batch_size = 2  # Always 2 for CFG

        self._rope_cache = {}

    @contextlib.contextmanager
    def cache_context(self, name: str):
        yield

    def _compute_rope_from_ids(self, txt_ids, img_ids):
        """Compute RoPE from pipeline-provided position IDs."""
        actual_txt = txt_ids.shape[0]
        actual_img = img_ids.shape[0]
        cache_key = (actual_txt, actual_img)

        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        if actual_txt < self.expected_txt_seq:
            pad_len = self.expected_txt_seq - actual_txt
            pad_ids = torch.zeros(pad_len, 3, dtype=txt_ids.dtype, device=txt_ids.device)
            last_row = txt_ids[-1, 1].item() if actual_txt > 0 else 0
            for i in range(pad_len):
                pad_ids[i, 0] = 0
                pad_ids[i, 1] = last_row + 1 + i
                pad_ids[i, 2] = last_row + 1 + i
            txt_ids_padded = torch.cat([txt_ids, pad_ids], dim=0)
        else:
            txt_ids_padded = txt_ids[:self.expected_txt_seq]

        if actual_img < self.expected_img_patches:
            pad_n = self.expected_img_patches - actual_img
            img_ids_padded = torch.cat(
                [img_ids, img_ids[-1:].expand(pad_n, -1)], dim=0)
        else:
            img_ids_padded = img_ids[:self.expected_img_patches]

        with torch.no_grad():
            txt_cos, txt_sin = self.pos_embed(txt_ids_padded)
            img_cos, img_sin = self.pos_embed(img_ids_padded)

        rope = (
            txt_cos.to(torch.bfloat16),
            txt_sin.to(torch.bfloat16),
            img_cos.to(torch.bfloat16),
            img_sin.to(torch.bfloat16),
        )
        self._rope_cache[cache_key] = rope
        return rope

    def _compute_rope_fallback(self, actual_txt_len):
        """Fallback RoPE computation when txt_ids/img_ids not provided."""
        cache_key = ("fallback", actual_txt_len)
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids

        text_ids = prepare_pos_ids(
            modality_id=0, type="text", num_token=self.expected_txt_seq)
        target_ids = prepare_pos_ids(
            modality_id=1, type="image",
            start=(actual_txt_len, actual_txt_len),
            height=self.patch_h, width=self.patch_w)
        source_ids = prepare_pos_ids(
            modality_id=2, type="image",
            start=(actual_txt_len, actual_txt_len),
            height=self.patch_h, width=self.patch_w)
        img_ids = torch.cat([target_ids, source_ids], dim=0)

        return self._compute_rope_from_ids(text_ids, img_ids)

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, txt_ids=None, img_ids=None,
                return_dict=False, **kwargs):
        """
        Forward pass for CFG parallel transformer.

        hidden_states: [2, img_patches, 64] -- batched neg+pos packed latents
        encoder_hidden_states: [2, txt_seq, 3584] -- batched neg+pos text embeddings
        timestep: [2] -- denoising timestep for both batch items
        """
        batch_size = hidden_states.shape[0]
        actual_txt_len = encoder_hidden_states.shape[1]

        # Compute RoPE (same for both batch items)
        if txt_ids is not None and img_ids is not None:
            txt_cos, txt_sin, img_cos, img_sin = self._compute_rope_from_ids(txt_ids, img_ids)
        else:
            txt_cos, txt_sin, img_cos, img_sin = self._compute_rope_fallback(actual_txt_len)

        # Pad hidden_states (image patches)
        actual_img = hidden_states.shape[1]
        if actual_img < self.expected_img_patches:
            pad = torch.zeros(
                (batch_size, self.expected_img_patches - actual_img, hidden_states.shape[2]),
                dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, pad], dim=1)
        elif actual_img > self.expected_img_patches:
            hidden_states = hidden_states[:, :self.expected_img_patches, :]

        # Pad encoder_hidden_states (text)
        if actual_txt_len < self.expected_txt_seq:
            pad = torch.zeros(
                (batch_size, self.expected_txt_seq - actual_txt_len, encoder_hidden_states.shape[2]),
                dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
            encoder_hidden_states = torch.cat([encoder_hidden_states, pad], dim=1)
        elif actual_txt_len > self.expected_txt_seq:
            encoder_hidden_states = encoder_hidden_states[:, :self.expected_txt_seq, :]

        timestep = timestep.to(torch.float32)

        # Run compiled model (batch_size=2, no batch padding needed)
        output = self.nxd_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
        )

        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        # Extract target image patches for both batch items
        output_tensor = output_tensor[:, :self.target_patches, :]

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output_tensor)
        return (output_tensor,)


class SimpleLatentDistribution:
    """Minimal latent distribution matching DiagonalGaussianDistribution interface."""
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        return self.mean  # Deterministic for compiled models


class SimpleEncoderOutput:
    """Minimal encoder output matching AutoencoderKLOutput interface."""
    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class NeuronVAEWrapper:
    """
    Wrapper for compiled 2D AutoencoderKL matching the pipeline interface.

    IMPORTANT: Scaling (shift_factor, scaling_factor) is handled by the PIPELINE,
    NOT by this wrapper. The pipeline applies:
      encode: latents = (latents - shift_factor) * scaling_factor
      decode: latents = latents / scaling_factor + shift_factor

    This wrapper provides:
    - encode(x) -> returns object with .latent_dist.mode()/.sample()
    - decode(z, return_dict=False) -> returns (decoded_tensor,)
    - .config -> original VAE config (attribute-accessible)
    - Tiled processing for images larger than compiled tile size
    """
    def __init__(self, compiled_encoder, compiled_decoder, original_vae,
                 tile_h=512, tile_w=512):
        self.compiled_encoder = compiled_encoder
        self.compiled_decoder = compiled_decoder
        # Keep original VAE config for pipeline attribute access
        # (e.g., self.vae.config.scaling_factor, self.vae.config.shift_factor)
        self.config = original_vae.config
        self.dtype = original_vae.dtype
        self.device = original_vae.device
        self.tile_h = tile_h
        self.tile_w = tile_w

    def encode(self, x, return_dict=True):
        """
        Encode image to latent space with tiled processing.

        Returns AutoencoderKLOutput-compatible object.
        Pipeline calls: retrieve_latents(self.vae.encode(image))
        which calls .latent_dist.mode() or .latent_dist.sample()
        """
        B, C, H, W = x.shape

        if H <= self.tile_h and W <= self.tile_w:
            moments = self.compiled_encoder(x)
        else:
            moments = self._tiled_encode(x)

        mean, logvar = torch.chunk(moments, 2, dim=1)
        dist = SimpleLatentDistribution(mean)

        if not return_dict:
            return (dist,)
        return SimpleEncoderOutput(dist)

    def decode(self, z, return_dict=False):
        """
        Decode latents to image with tiled processing.

        Pipeline calls: self.vae.decode(latents, return_dict=False)[0]
        """
        latent_h = z.shape[2]
        latent_w = z.shape[3]
        tile_latent_h = self.tile_h // 8
        tile_latent_w = self.tile_w // 8

        if latent_h <= tile_latent_h and latent_w <= tile_latent_w:
            decoded = self.compiled_decoder(z)
        else:
            decoded = self._tiled_decode(z)

        if return_dict:
            return type('DecoderOutput', (), {'sample': decoded})()
        return (decoded,)

    def _tiled_encode(self, x):
        """Tiled encoding for large images (no overlap to ensure exact output size)."""
        B, C, H, W = x.shape
        tile_h, tile_w = self.tile_h, self.tile_w

        latent_tiles = []
        for y in range(0, H, tile_h):
            row_tiles = []
            for x_start in range(0, W, tile_w):
                y_end = min(y + tile_h, H)
                x_end = min(x_start + tile_w, W)
                tile = x[:, :, y:y_end, x_start:x_end]

                # Pad to tile size if needed
                if tile.shape[2] < tile_h or tile.shape[3] < tile_w:
                    padded = torch.zeros(B, C, tile_h, tile_w, dtype=tile.dtype, device=tile.device)
                    padded[:, :, :tile.shape[2], :tile.shape[3]] = tile
                    tile = padded

                moments = self.compiled_encoder(tile)
                mean, logvar = torch.chunk(moments, 2, dim=1)
                # Trim to actual latent size (in case of padding)
                latent_h = (y_end - y) // 8
                latent_w = (x_end - x_start) // 8
                row_tiles.append(mean[:, :, :latent_h, :latent_w])
            latent_tiles.append(row_tiles)

        rows = [torch.cat(row, dim=3) for row in latent_tiles]
        full_mean = torch.cat(rows, dim=2)
        full_logvar = torch.zeros_like(full_mean)
        return torch.cat([full_mean, full_logvar], dim=1)

    def _tiled_decode(self, z):
        """Tiled decoding for large latents (no overlap to ensure exact output size)."""
        B, C, H, W = z.shape
        tile_h = self.tile_h // 8
        tile_w = self.tile_w // 8

        pixel_tiles = []
        for y in range(0, H, tile_h):
            row_tiles = []
            for x_start in range(0, W, tile_w):
                y_end = min(y + tile_h, H)
                x_end = min(x_start + tile_w, W)
                tile = z[:, :, y:y_end, x_start:x_end]

                if tile.shape[2] < tile_h or tile.shape[3] < tile_w:
                    padded = torch.zeros(B, C, tile_h, tile_w, dtype=tile.dtype, device=tile.device)
                    padded[:, :, :tile.shape[2], :tile.shape[3]] = tile
                    tile = padded

                decoded = self.compiled_decoder(tile)
                pixel_h = (y_end - y) * 8
                pixel_w = (x_end - x_start) * 8
                row_tiles.append(decoded[:, :, :pixel_h, :pixel_w])
            pixel_tiles.append(row_tiles)

        rows = [torch.cat(row, dim=3) for row in pixel_tiles]
        return torch.cat(rows, dim=2)


def load_transformer(compiled_models_dir, pipe, args):
    """Load compiled transformer model."""
    model_path = f"{compiled_models_dir}/transformer"
    nxd_model_path = f"{model_path}/nxd_model.pt"
    weights_path = f"{model_path}/weights"
    rope_cache_path = f"{model_path}/rope_cache.pt"
    config_path = f"{model_path}/config.json"

    for p, name in [(nxd_model_path, "model"), (weights_path, "weights"),
                     (rope_cache_path, "RoPE cache"), (config_path, "config")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Compiled {name} not found at {p}")

    with open(config_path, "r") as f:
        config = json.load(f)

    expected_img_patches = config["num_img_patches_padded"]
    expected_txt_seq = config["text_seq_len"]
    target_patches = config["num_img_patches"] // 2  # Only target image patches
    compiled_batch_size = config.get("batch_size", 1)
    patch_h = config["patch_h"]
    patch_w = config["patch_w"]

    print(f"  Compiled config: img_patches={expected_img_patches}, txt_seq={expected_txt_seq}")
    print(f"  Target patches: {target_patches}, batch_size={compiled_batch_size}")
    print(f"  Patch grid: {patch_h}x{patch_w}")

    # Load NxDModel
    print(f"  Loading Compiled model...")
    nxd_model = NxDModel.load(nxd_model_path)

    # Load sharded weights
    # NxDModel expects one checkpoint per world_rank.
    # For CP: ranks within the same TP group share weights.
    # Layout: ranks [0..tp-1] = TP group 0 (CP=0), ranks [tp..2*tp-1] = TP group 1 (CP=1)
    from safetensors.torch import load_file
    tp_degree = config.get("tp_degree", 4)
    world_size = config.get("world_size", 8)

    # Load base TP checkpoints
    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)
        print(f"    Loaded tp{rank}: {len(ckpt)} tensors")

    # Duplicate for all world ranks (CP ranks share TP weights)
    # CRITICAL: Each world rank must have the correct global_rank.rank value
    # for SPMDRank to work. Without this, all ranks think they are rank 0
    # and the CP scatter always takes the first half of the sequence.
    import copy
    sharded_checkpoints = []
    for world_rank in range(world_size):
        tp_rank = world_rank % tp_degree
        ckpt_copy = dict(tp_checkpoints[tp_rank])  # shallow copy
        # Set the correct world rank for SPMDRank
        rank_key = "transformer.global_rank.rank"
        if rank_key in ckpt_copy:
            ckpt_copy[rank_key] = torch.tensor([world_rank], dtype=torch.int32)
        sharded_checkpoints.append(ckpt_copy)
    print(f"  Prepared {len(sharded_checkpoints)} weight shards for world_size={world_size}")

    nxd_model.set_weights(sharded_checkpoints)
    print("  Weights set, loading to Neuron...")
    nxd_model.to_neuron()
    print("  Compiled model initialized on Neuron!")

    wrapper = NeuronTransformerWrapper(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        pos_embed=pipe.transformer.pos_embed,
        patch_h=patch_h,
        patch_w=patch_w,
        expected_img_patches=expected_img_patches,
        expected_txt_seq=expected_txt_seq,
        target_patches=target_patches,
        batch_size=compiled_batch_size,
    )
    return wrapper


def load_transformer_cfg(compiled_models_dir, pipe, args):
    """Load CFG parallel compiled transformer model."""
    model_path = f"{compiled_models_dir}/transformer_cfg"
    nxd_model_path = f"{model_path}/nxd_model.pt"
    weights_path = f"{model_path}/weights"
    rope_cache_path = f"{model_path}/rope_cache.pt"
    config_path = f"{model_path}/config.json"

    for p, name in [(nxd_model_path, "model"), (weights_path, "weights"),
                     (rope_cache_path, "RoPE cache"), (config_path, "config")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Compiled CFG {name} not found at {p}")

    with open(config_path, "r") as f:
        config = json.load(f)

    expected_img_patches = config["num_img_patches_padded"]
    expected_txt_seq = config["text_seq_len"]
    target_patches = config["num_img_patches"] // 2
    patch_h = config["patch_h"]
    patch_w = config["patch_w"]

    print(f"  CFG config: img_patches={expected_img_patches}, txt_seq={expected_txt_seq}")
    print(f"  Target patches: {target_patches}, batch_size=2 (CFG)")
    print(f"  Patch grid: {patch_h}x{patch_w}")

    # Load NxDModel
    print(f"  Loading CFG compiled model...")
    nxd_model = NxDModel.load(nxd_model_path)

    from safetensors.torch import load_file
    tp_degree = config.get("tp_degree", 4)
    world_size = config.get("world_size", 8)

    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors"
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)
        print(f"    Loaded tp{rank}: {len(ckpt)} tensors")

    # Duplicate for all world ranks (DP ranks share TP weights)
    import copy
    sharded_checkpoints = []
    for world_rank in range(world_size):
        tp_rank = world_rank % tp_degree
        ckpt_copy = dict(tp_checkpoints[tp_rank])
        rank_key = "transformer.global_rank.rank"
        if rank_key in ckpt_copy:
            ckpt_copy[rank_key] = torch.tensor([world_rank], dtype=torch.int32)
        sharded_checkpoints.append(ckpt_copy)
    print(f"  Prepared {len(sharded_checkpoints)} weight shards for world_size={world_size}")

    nxd_model.set_weights(sharded_checkpoints)
    print("  Weights set, loading to Neuron...")
    nxd_model.to_neuron()
    print("  CFG compiled model initialized on Neuron!")

    wrapper = NeuronTransformerWrapperCFG(
        original_transformer=pipe.transformer,
        nxd_model=nxd_model,
        pos_embed=pipe.transformer.pos_embed,
        patch_h=patch_h,
        patch_w=patch_w,
        expected_img_patches=expected_img_patches,
        expected_txt_seq=expected_txt_seq,
        target_patches=target_patches,
    )
    return wrapper


def patch_pipeline_for_cfg_parallel(pipe):
    """
    Monkey-patch pipeline for CFG parallel inference.

    The LongCat pipeline calls transformer twice per denoising step when
    guidance_scale > 1 (positive first, then negative). This patch:
    1. Captures negative prompt embeddings from encode_prompt
    2. Replaces transformer with a proxy that batches both calls into one
    3. On positive call: runs batched [neg, pos], returns positive result
    4. On negative call: returns cached negative result (no computation)
    """
    real_transformer = pipe.transformer
    neg_state = {"embeds": None, "txt_ids": None}

    # Monkey-patch encode_prompt to capture negative embeddings
    original_encode = pipe.encode_prompt
    encode_call_count = [0]

    def capturing_encode_prompt(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        encode_call_count[0] += 1
        # Second encode call is the negative prompt
        if encode_call_count[0] % 2 == 0:
            neg_state["embeds"] = result[0]   # negative_prompt_embeds
            neg_state["txt_ids"] = result[1]  # negative_text_ids
        return result

    pipe.encode_prompt = capturing_encode_prompt

    # Create CFG batching proxy
    class CFGBatchingProxy:
        """
        Proxy that batches two sequential CFG transformer calls into one.

        Pipeline call order per step:
        1. noise_pred_text = transformer(latents, pos_embeds, t, ...)  [positive]
        2. noise_pred_uncond = transformer(latents, neg_embeds, t, ...) [negative]

        Proxy behavior:
        1. On positive call: batch [neg, pos] using stored neg_embeds, run ONCE
        2. On negative call: return cached result (zero compute)
        """
        def __init__(self, real_tf):
            self._real_tf = real_tf
            self.config = real_tf.config
            self.dtype = real_tf.dtype
            self.device = real_tf.device
            self._call_idx = 0  # 0=positive, 1=negative per step
            self._cached_neg_result = None

        def cache_context(self, name):
            return self._real_tf.cache_context(name)

        def __call__(self, hidden_states, timestep, encoder_hidden_states,
                     txt_ids=None, img_ids=None, return_dict=False, **kw):
            if self._call_idx == 0 and neg_state["embeds"] is not None:
                # Positive call: batch with stored negative, run once
                # hidden_states and timestep are the same for both batch items
                batched_hs = torch.cat([hidden_states, hidden_states], dim=0)
                batched_enc = torch.cat([neg_state["embeds"], encoder_hidden_states], dim=0)
                batched_t = torch.cat([timestep, timestep], dim=0)

                result = self._real_tf(
                    hidden_states=batched_hs,
                    encoder_hidden_states=batched_enc,
                    timestep=batched_t,
                    txt_ids=txt_ids,  # Same RoPE for both batch items
                    img_ids=img_ids,
                    return_dict=False,
                )

                output = result[0]  # [2, target_patches, 64]
                self._cached_neg_result = output[0:1]  # neg result
                self._call_idx = 1
                return (output[1:2],)  # pos result

            elif self._call_idx == 1 and self._cached_neg_result is not None:
                # Negative call: return cached result (no computation)
                result = self._cached_neg_result
                self._cached_neg_result = None
                self._call_idx = 0
                return (result,)

            else:
                # Fallback: run normally (non-CFG or neg_embeds not captured)
                self._call_idx = 0
                return self._real_tf(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=return_dict,
                    **kw,
                )

        def __getattr__(self, name):
            return getattr(self._real_tf, name)

    proxy = CFGBatchingProxy(real_transformer)
    pipe.transformer = proxy
    pipe._cfg_parallel_enabled = True

    return pipe


def load_text_encoder(compiled_models_dir, pipe, args, use_cpu_ve=False):
    """Load compiled text encoder (vision encoder + language model)."""
    # Load vision encoder
    ve_path = f"{compiled_models_dir}/vision_encoder"
    ve_model_path = f"{ve_path}/nxd_model.pt"

    compiled_ve = None
    cpu_ve = None
    if use_cpu_ve:
        print("  Using CPU vision encoder (for accuracy)")
        cpu_ve = pipe.text_encoder.model.visual
    elif os.path.exists(ve_model_path):
        from safetensors.torch import load_file
        with open(f"{ve_path}/config.json") as f:
            ve_config = json.load(f)

        print("  Loading compiled vision encoder...")
        ve_nxd = NxDModel.load(ve_model_path)
        tp_degree = ve_config.get("tp_degree", 4)
        ve_world_size = ve_config.get("world_size", ve_nxd.world_size)
        ve_tp_checkpoints = []
        for rank in range(tp_degree):
            ckpt = load_file(f"{ve_path}/weights/tp{rank}_sharded_checkpoint.safetensors")
            ve_tp_checkpoints.append(ckpt)
        # Duplicate for all world ranks (CP ranks share TP weights)
        ve_checkpoints = [ve_tp_checkpoints[r % tp_degree] for r in range(ve_world_size)]
        print(f"  VE: {tp_degree} TP checkpoints -> {len(ve_checkpoints)} world ranks")
        ve_nxd.set_weights(ve_checkpoints)
        ve_nxd.to_neuron()
        compiled_ve = ve_nxd
        print("  compiled vision encoder loaded!")
    else:
        print("  WARNING: compiled vision encoder not found, using CPU vision encoder")
        cpu_ve = pipe.text_encoder.model.visual

    # Load language model
    lm_path = f"{compiled_models_dir}/language_model"
    lm_model_path = f"{lm_path}/nxd_model.pt"

    compiled_lm = None
    cpu_lm = None

    if os.path.exists(lm_model_path):
        from safetensors.torch import load_file
        with open(f"{lm_path}/config.json") as f:
            lm_config = json.load(f)

        print("  Loading compiled language model...")
        lm_nxd = NxDModel.load(lm_model_path)
        tp_degree = lm_config.get("tp_degree", 4)
        lm_world_size = lm_config.get("world_size", lm_nxd.world_size)
        lm_tp_checkpoints = []
        for rank in range(tp_degree):
            ckpt = load_file(f"{lm_path}/weights/tp{rank}_sharded_checkpoint.safetensors")
            lm_tp_checkpoints.append(ckpt)
        # Duplicate for all world ranks (CP ranks share TP weights)
        lm_checkpoints = [lm_tp_checkpoints[r % tp_degree] for r in range(lm_world_size)]
        print(f"  LM: {tp_degree} TP checkpoints -> {len(lm_checkpoints)} world ranks")
        lm_nxd.set_weights(lm_checkpoints)
        lm_nxd.to_neuron()
        compiled_lm = lm_nxd
        max_seq_len = lm_config.get("max_sequence_length", 512)
        lm_batch_size = lm_config.get("batch_size", 1)
        print("  compiled language model loaded!")
    else:
        print("  compiled language model not found, using CPU fallback")
        cpu_lm = pipe.text_encoder.model.language_model
        max_seq_len = 512
        lm_batch_size = 1

    # Create wrapper
    wrapper = NeuronTextEncoderWrapper(
        original_text_encoder=pipe.text_encoder,
        compiled_vision_encoder=compiled_ve,
        compiled_language_model=compiled_lm,
        cpu_language_model=cpu_lm,
        cpu_vision_encoder=cpu_ve,
        image_size=args.image_size,
        max_seq_len=max_seq_len,
        language_model_batch_size=lm_batch_size,
    )
    return wrapper


def load_vae(compiled_models_dir, pipe, use_compiled=True):
    """Load compiled VAE or use original CPU VAE."""
    if not use_compiled:
        print("  Using original CPU VAE (compiled VAE skipped)")
        return pipe.vae

    encoder_path = f"{compiled_models_dir}/vae_encoder/model.pt"
    decoder_path = f"{compiled_models_dir}/vae_decoder/model.pt"
    config_path = f"{compiled_models_dir}/vae_config.json"

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("  WARNING: Compiled VAE not found, using CPU VAE")
        return pipe.vae

    with open(config_path) as f:
        vae_config = json.load(f)

    tile_h = vae_config.get("height", 512)
    tile_w = vae_config.get("width", 512)
    print(f"  Loading compiled VAE (tile: {tile_h}x{tile_w})")

    compiled_encoder = torch.jit.load(encoder_path)
    compiled_decoder = torch.jit.load(decoder_path)

    wrapper = NeuronVAEWrapper(
        compiled_encoder=compiled_encoder,
        compiled_decoder=compiled_decoder,
        original_vae=pipe.vae,
        tile_h=tile_h,
        tile_w=tile_w,
    )
    return wrapper


def main():
    parser = argparse.ArgumentParser(description="LongCat-Image-Edit Inference on Trainium2")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Edit instruction")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Negative prompt")
    parser.add_argument("--output", type=str, default="output_edited.png", help="Output path")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--image_size", type=int, default=448, help="Vision encoder image size")
    parser.add_argument("--warmup", action="store_true", help="Run warmup inference")
    parser.add_argument("--skip_compiled_vae", action="store_true", help="Use CPU VAE instead of compiled")
    parser.add_argument("--skip_compiled_text_encoder", action="store_true",
                        help="Use CPU text encoder instead of compiled")
    parser.add_argument("--cpu_vision_encoder", action="store_true",
                        help="Use CPU vision encoder for accuracy (Neuron LM still used)")
    parser.add_argument("--use_cfg_parallel", action="store_true",
                        help="Use CFG Parallel transformer (batches neg+pos prompts, ~2x denoising speedup). "
                             "Requires: ./compile.sh cfg")
    parser.add_argument("--compiled_models_dir", type=str, default=COMPILED_MODELS_DIR)
    parser.add_argument("--transformer_dir", type=str, default=None,
                        help="Override transformer compiled dir (default: <compiled_models_dir>)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load pipeline
    print("\n[Step 1/4] Loading LongCat pipeline...")
    t0 = time.perf_counter()
    load_kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if HUGGINGFACE_CACHE_DIR:
        load_kwargs["cache_dir"] = HUGGINGFACE_CACHE_DIR
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)
    print(f"  Pipeline loaded in {time.perf_counter() - t0:.1f}s")

    # Configure image processor
    # When using CPU VE, use default resolution (matching HuggingFace/H100 behavior)
    # When using compiled VE, force fixed resolution to match compiled model
    if not getattr(args, "cpu_vision_encoder", False):
        target_pixels = args.image_size * args.image_size
        print(f"  Configuring image processor: min_pixels=max_pixels={target_pixels} (compiled VE)")
        pipe.image_processor_vl.min_pixels = target_pixels
        pipe.image_processor_vl.max_pixels = target_pixels
    else:
        print(f"  Using default image processor resolution (CPU VE, matching HuggingFace defaults)")

    # Load compiled components
    print("\n[Step 2/4] Loading compiled Neuron models...")

    # Transformer
    transformer_dir = args.transformer_dir or args.compiled_models_dir
    if args.use_cfg_parallel:
        print(f"Loading CFG Parallel transformer from {transformer_dir}...")
        neuron_transformer = load_transformer_cfg(transformer_dir, pipe, args)
    else:
        print(f"Loading CP transformer from {transformer_dir}...")
        neuron_transformer = load_transformer(transformer_dir, pipe, args)

    # Text encoder
    if args.skip_compiled_text_encoder:
        print("Using original CPU text encoder (compiled text encoder skipped)")
        neuron_text_encoder = pipe.text_encoder
    else:
        print("Loading text encoder...")
        neuron_text_encoder = load_text_encoder(args.compiled_models_dir, pipe, args, use_cpu_ve=getattr(args, "cpu_vision_encoder", False))

    # VAE
    print("Loading VAE...")
    neuron_vae = load_vae(args.compiled_models_dir, pipe, use_compiled=not args.skip_compiled_vae)

    # Replace pipeline components
    pipe.transformer = neuron_transformer
    pipe.text_encoder = neuron_text_encoder
    pipe.vae = neuron_vae

    # Apply CFG parallel pipeline patching
    if args.use_cfg_parallel:
        print("Applying CFG parallel pipeline patch...")
        patch_pipeline_for_cfg_parallel(pipe)
        print("  CFG parallel enabled: batched neg+pos transformer calls")

    # Delete original weights to save memory
    import gc
    gc.collect()

    # Load image
    print("\n[Step 3/4] Loading input image...")
    source_image = Image.open(args.image).convert("RGB")
    print(f"  Input image: {source_image.size}")

    # Run inference
    print(f"\n[Step 4/4] Running inference ({args.num_inference_steps} steps)...")

    if args.warmup:
        print("  Warmup run...")
        with torch.inference_mode():
            _ = pipe(
                image=source_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.manual_seed(args.seed),
            )
        print("  Warmup complete!")

    # Timed run
    set_seed(args.seed)
    t_start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            image=source_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.manual_seed(args.seed),
        )
    t_end = time.perf_counter()

    # Save output
    output_image = result.images[0]
    output_image.save(args.output)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Output saved to: {os.path.abspath(args.output)}")
    print(f"  Output size: {output_image.size}")
    print(f"  Total time: {t_end - t_start:.2f}s")
    print(f"  Steps/sec: {args.num_inference_steps / (t_end - t_start):.2f}")


if __name__ == "__main__":
    main()
