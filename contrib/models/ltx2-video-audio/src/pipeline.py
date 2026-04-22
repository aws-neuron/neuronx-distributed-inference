"""
NxDI LTX-2 Pipeline
====================
Neuron-aware pipeline that wraps the Diffusers LTX2Pipeline with
Neuron-compiled transformer backbone.

The pipeline works by:
1. Using the stock Diffusers LTX2Pipeline for text encoding, VAE, vocoder
2. Intercepting transformer calls and routing them through:
   a. CPU preprocessing (proj_in, time_embed, RoPE, caption_projection)
   b. Neuron compiled backbone (48 blocks + output layers)
3. Returning the pipeline output (frames + audio)

This approach avoids reimplementing the complex LTX2Pipeline scheduling,
CFG handling, VAE decoding, etc. We only replace the transformer forward().
"""

import gc
import logging
import os
import time
from typing import Optional

import torch
import torch.nn as nn

try:
    from .modeling_ltx2 import replace_sdpa_with_bmm
    from .tiled_vae_decode import load_compiled_vae, tiled_decode
except ImportError:
    from modeling_ltx2 import replace_sdpa_with_bmm
    from tiled_vae_decode import load_compiled_vae, tiled_decode

logger = logging.getLogger(__name__)


class NeuronTransformerWrapper(nn.Module):
    """Drop-in replacement for LTX2VideoTransformer3DModel in the Diffusers pipeline.

    The pipeline calls:
        self.transformer(
            hidden_states=..., audio_hidden_states=...,
            encoder_hidden_states=..., audio_encoder_hidden_states=...,
            timestep=..., encoder_attention_mask=...,
            audio_encoder_attention_mask=...,
            num_frames=..., height=..., width=..., fps=...,
            audio_num_frames=..., video_coords=..., audio_coords=...,
            return_dict=False,
        )

    This wrapper:
    1. Keeps a CPU copy of the transformer's preprocessing layers
       (proj_in, time_embed, caption_projection, rope, etc.)
    2. Replicates the preprocessing from LTX2VideoTransformer3DModel.forward()
    3. Converts encoder_attention_mask to additive bias format
    4. Calls the compiled Neuron model with 22 positional tensor args
    5. Returns (video_output, audio_output) as the pipeline expects
    """

    def __init__(self, compiled_backbone, cpu_transformer, text_seq=1024):
        """
        Args:
            compiled_backbone: The NeuronLTX2BackboneApplication or
                              TensorParallelNeuronModel (loaded Neuron model)
            cpu_transformer: The original LTX2VideoTransformer3DModel (for preprocessing layers)
            text_seq: Maximum text sequence length (must match compile-time)
        """
        super().__init__()
        self.compiled_backbone = compiled_backbone
        self.text_seq = text_seq

        # Copy config and attributes the pipeline expects
        self.config = cpu_transformer.config
        self.dtype = cpu_transformer.dtype
        self.device = cpu_transformer.device
        if hasattr(cpu_transformer, "cache_context"):
            self.cache_context = cpu_transformer.cache_context

        # Keep CPU preprocessing layers (NOT compiled, NOT in the Neuron model)
        self.proj_in = cpu_transformer.proj_in
        self.audio_proj_in = cpu_transformer.audio_proj_in
        self.time_embed = cpu_transformer.time_embed
        self.audio_time_embed = cpu_transformer.audio_time_embed
        self.av_cross_attn_video_scale_shift = (
            cpu_transformer.av_cross_attn_video_scale_shift
        )
        self.av_cross_attn_video_a2v_gate = cpu_transformer.av_cross_attn_video_a2v_gate
        self.av_cross_attn_audio_scale_shift = (
            cpu_transformer.av_cross_attn_audio_scale_shift
        )
        self.av_cross_attn_audio_v2a_gate = cpu_transformer.av_cross_attn_audio_v2a_gate
        self.caption_projection = cpu_transformer.caption_projection
        self.audio_caption_projection = cpu_transformer.audio_caption_projection
        self.rope = cpu_transformer.rope
        self.audio_rope = cpu_transformer.audio_rope
        self.cross_attn_rope = cpu_transformer.cross_attn_rope
        self.cross_attn_audio_rope = cpu_transformer.cross_attn_audio_rope

    def _compute_step_invariant(
        self,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        video_coords,
        audio_coords,
        batch_size,
        inner_dim,
        audio_inner_dim,
        dtype,
    ):
        """Compute and cache step-invariant preprocessing (caption proj, RoPE, masks).

        These values depend only on the prompt and spatial layout, not the timestep
        or noisy latents, so they are identical across all denoising steps.
        Computing them once saves ~0.2-0.4s of CPU time per step.
        """
        with torch.no_grad():
            # Caption projection (CPU)
            enc_hs = self.caption_projection(encoder_hidden_states)
            enc_hs = enc_hs.view(batch_size, -1, inner_dim)
            audio_enc_hs = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_enc_hs = audio_enc_hs.view(batch_size, -1, audio_inner_dim)

            # RoPE (CPU) — compute from coords
            video_rotary_emb = self.rope(video_coords, device="cpu")
            audio_rotary_emb = self.audio_rope(audio_coords, device="cpu")
            video_cross_rotary_emb = self.cross_attn_rope(
                video_coords[:, 0:1, :], device="cpu"
            )
            audio_cross_rotary_emb = self.cross_attn_audio_rope(
                audio_coords[:, 0:1, :], device="cpu"
            )

        # RoPE modules return float32 for precision; cast to bfloat16 for Neuron
        video_rotary_emb = (
            video_rotary_emb[0].to(dtype),
            video_rotary_emb[1].to(dtype),
        )
        audio_rotary_emb = (
            audio_rotary_emb[0].to(dtype),
            audio_rotary_emb[1].to(dtype),
        )
        video_cross_rotary_emb = (
            video_cross_rotary_emb[0].to(dtype),
            video_cross_rotary_emb[1].to(dtype),
        )
        audio_cross_rotary_emb = (
            audio_cross_rotary_emb[0].to(dtype),
            audio_cross_rotary_emb[1].to(dtype),
        )

        # Attention masks — convert from binary (B, text_seq) to additive bias
        with torch.no_grad():
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                enc_mask = (1 - encoder_attention_mask.to(dtype)) * -10000.0
                enc_mask = enc_mask.unsqueeze(1)  # (B, 1, text_seq)
            else:
                enc_mask = encoder_attention_mask

            if (
                audio_encoder_attention_mask is not None
                and audio_encoder_attention_mask.ndim == 2
            ):
                audio_enc_mask = (1 - audio_encoder_attention_mask.to(dtype)) * -10000.0
                audio_enc_mask = audio_enc_mask.unsqueeze(1)
            else:
                audio_enc_mask = audio_encoder_attention_mask

            if enc_mask is None:
                enc_mask = torch.zeros(batch_size, 1, self.text_seq, dtype=dtype)
            if audio_enc_mask is None:
                audio_enc_mask = torch.zeros(batch_size, 1, self.text_seq, dtype=dtype)

        return (
            enc_hs,
            audio_enc_hs,
            video_rotary_emb,
            audio_rotary_emb,
            video_cross_rotary_emb,
            audio_cross_rotary_emb,
            enc_mask,
            audio_enc_mask,
        )

    def forward(
        self,
        hidden_states,
        audio_hidden_states=None,
        encoder_hidden_states=None,
        audio_encoder_hidden_states=None,
        timestep=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        num_frames=None,
        height=None,
        width=None,
        fps=None,
        audio_num_frames=None,
        video_coords=None,
        audio_coords=None,
        return_dict=False,
        **kwargs,
    ):
        """Preprocess on CPU, run 48 blocks on Neuron, return results.

        Step-invariant computations (caption projection, RoPE, attention masks) are
        cached after the first call and reused for subsequent denoising steps. The
        cache is invalidated when encoder_hidden_states changes (new prompt).
        """
        batch_size = hidden_states.shape[0]
        dtype = torch.bfloat16

        with torch.no_grad():
            # 1. Project inputs (CPU) — step-varying (latents change each step)
            hs = self.proj_in(hidden_states)
            ahs = self.audio_proj_in(audio_hidden_states)

            # 2. Time embeddings (CPU) — step-varying (timestep changes each step)
            temb, embedded_ts = self.time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_ts = embedded_ts.view(batch_size, -1, embedded_ts.size(-1))

            temb_audio, audio_embedded_ts = self.audio_time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
            audio_embedded_ts = audio_embedded_ts.view(
                batch_size, -1, audio_embedded_ts.size(-1)
            )

            # 3. Cross-attention conditioning (CPU) — step-varying (timestep-dependent)
            ts_scale = (
                self.config.cross_attn_timestep_scale_multiplier
                / self.config.timestep_scale_multiplier
            )

            video_ca_ss, _ = self.av_cross_attn_video_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            video_ca_gate, _ = self.av_cross_attn_video_a2v_gate(
                timestep.flatten() * ts_scale, batch_size=batch_size, hidden_dtype=dtype
            )
            video_ca_ss = video_ca_ss.view(batch_size, -1, video_ca_ss.shape[-1])
            video_ca_gate = video_ca_gate.view(batch_size, -1, video_ca_gate.shape[-1])

            audio_ca_ss, _ = self.av_cross_attn_audio_scale_shift(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=dtype
            )
            audio_ca_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
                timestep.flatten() * ts_scale, batch_size=batch_size, hidden_dtype=dtype
            )
            audio_ca_ss = audio_ca_ss.view(batch_size, -1, audio_ca_ss.shape[-1])
            audio_ca_v2a_gate = audio_ca_v2a_gate.view(
                batch_size, -1, audio_ca_v2a_gate.shape[-1]
            )

        # 4-6. Step-invariant: caption projection, RoPE, attention masks (cached)
        # Use data_ptr() of encoder_hidden_states as cache key — changes per prompt,
        # constant across denoising steps within a single generation.
        cache_key = encoder_hidden_states.data_ptr()
        if not hasattr(self, "_step_cache") or self._step_cache_key != cache_key:
            (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            ) = self._compute_step_invariant(
                encoder_hidden_states,
                audio_encoder_hidden_states,
                encoder_attention_mask,
                audio_encoder_attention_mask,
                video_coords,
                audio_coords,
                batch_size,
                hs.size(-1),
                ahs.size(-1),
                dtype,
            )
            self._step_cache = (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            )
            self._step_cache_key = cache_key
        else:
            (
                enc_hs,
                audio_enc_hs,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_rotary_emb,
                audio_cross_rotary_emb,
                enc_mask,
                audio_enc_mask,
            ) = self._step_cache

        # 7. Call compiled Neuron model (22 positional args)
        # The Neuron backbone is compiled for batch_size=2 (CFG mode).
        # When CFG is active (guidance_scale > 1), the pipeline doubles the batch
        # to 2 (uncond + cond) and the backbone processes both in a single pass.
        backbone_args = (
            hs,
            ahs,
            enc_hs,
            audio_enc_hs,
            temb,
            temb_audio,
            embedded_ts,
            audio_embedded_ts,
            video_ca_ss,
            audio_ca_ss,
            video_ca_gate,
            audio_ca_v2a_gate,
            video_rotary_emb[0],  # cos
            video_rotary_emb[1],  # sin
            audio_rotary_emb[0],
            audio_rotary_emb[1],
            video_cross_rotary_emb[0],
            video_cross_rotary_emb[1],
            audio_cross_rotary_emb[0],
            audio_cross_rotary_emb[1],
            enc_mask,
            audio_enc_mask,
        )

        video_output, audio_output = self.compiled_backbone(*backbone_args)

        return video_output, audio_output


class NeuronTiledVAEDecoder(nn.Module):
    """Drop-in replacement for the Diffusers LTX2VideoDecoder3d.

    This is swapped into pipe.vae.decoder so that the stock Diffusers
    AutoencoderKLLTX2Video.decode() calls our Neuron-tiled decode path
    transparently.

    The outer VAE calls:
        self.decoder(hidden_states, temb=None, causal=False)

    For LTX-2: timestep_conditioning=False, so temb is always None.
    causal=False for non-causal inference.

    This wrapper:
    1. Loads the compiled TP VAE decoder from disk
    2. On forward(), calls tiled_decode() to spatially tile the latent
    3. Returns the full-resolution decoded tensor

    Attributes copied from original decoder:
        patch_size, patch_size_t — needed by the outer VAE's unpatchify logic
        (Actually, unpatchify is handled inside our compiled model, but we
         expose these for compatibility in case the outer VAE checks them.)
    """

    def __init__(
        self,
        compiled_dir,
        tile_latent_h=4,
        tile_latent_w=16,
        overlap_latent_h=1,
        overlap_latent_w=0,
        original_decoder=None,
    ):
        """
        Args:
            compiled_dir: Path to directory with compiled TP model (tp_0.pt, etc.)
            tile_latent_h: Tile height in latent pixels (default 4)
            tile_latent_w: Tile width in latent pixels (default 16)
            overlap_latent_h: Overlap in latent H (default 1)
            overlap_latent_w: Overlap in latent W (default 0)
            original_decoder: The original LTX2VideoDecoder3d (for attribute copying)
        """
        super().__init__()
        self.tile_latent_h = tile_latent_h
        self.tile_latent_w = tile_latent_w
        self.overlap_latent_h = overlap_latent_h
        self.overlap_latent_w = overlap_latent_w

        # Load compiled model
        logger.info("Loading compiled VAE from %s", compiled_dir)
        t0 = time.time()
        self.compiled_model = load_compiled_vae(compiled_dir)
        logger.info("VAE loaded in %.1fs", time.time() - t0)

        # Copy attributes the outer VAE might access
        if original_decoder is not None:
            self.patch_size = getattr(original_decoder, "patch_size", 4)
            self.patch_size_t = getattr(original_decoder, "patch_size_t", 1)
            self.is_causal = getattr(original_decoder, "is_causal", False)
        else:
            self.patch_size = 4
            self.patch_size_t = 1
            self.is_causal = False

        self._warmed_up = False

    def warmup(self, num_frames=121):
        """Run 2 warmup iterations to prime the Neuron model."""
        if self._warmed_up:
            return
        logger.info("Warming up VAE decoder...")
        latent_t = (num_frames - 1) // 8 + 1
        dummy = torch.randn(
            1,
            128,
            latent_t,
            self.tile_latent_h,
            self.tile_latent_w,
            dtype=torch.float32,
        )
        for _ in range(2):
            with torch.no_grad():
                self.compiled_model(dummy)
        self._warmed_up = True
        logger.info("VAE warmup done")

    def forward(self, hidden_states, temb=None, causal=None):
        """Decode latent tensor using Neuron tiled decode.

        Args:
            hidden_states: [B, C, T, H, W] latent tensor (from VAE encoder/denormalize)
            temb: Time embedding (always None for LTX-2, timestep_conditioning=False)
            causal: Causal mode flag (always False for inference)

        Returns:
            Decoded tensor [B, 3, T_out, H_out, W_out]

        Note: The compiled Neuron model includes unpatchify (48 -> 3 channels),
        but the outer AutoencoderKLLTX2Video.decode() does NOT call unpatchify
        separately — it's part of the decoder. So our output is the final RGB.
        """
        if not self._warmed_up:
            num_frames_approx = (hidden_states.shape[2] - 1) * 8 + 1
            self.warmup(num_frames=num_frames_approx)

        # tiled_decode expects float32 input (Neuron compiles in fp32)
        latent_fp32 = hidden_states.float()

        output = tiled_decode(
            latent_fp32,
            self.compiled_model,
            tile_latent_h=self.tile_latent_h,
            tile_latent_w=self.tile_latent_w,
            overlap_latent_h=self.overlap_latent_h,
            overlap_latent_w=self.overlap_latent_w,
            spatial_scale=32,
            verbose=True,
        )

        return output


class NeuronLTX2Pipeline:
    """Wrapper around the Diffusers LTX2Pipeline that supports Neuron acceleration.

    This class manages:
    1. Loading the stock Diffusers pipeline (text encoder, VAEs, vocoder on CPU)
    2. Holding a reference to the NeuronLTX2BackboneApplication
    3. Swapping the transformer with NeuronTransformerWrapper after loading
    4. Optionally swapping the VAE decoder with NeuronTiledVAEDecoder

    Usage:
        pipe = NeuronLTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
        pipe.neuron_backbone = NeuronLTX2BackboneApplication(...)
        pipe.neuron_backbone.compile(path)
        pipe.neuron_backbone.load(path)
        pipe._swap_transformer_to_neuron()
        pipe._swap_vae_to_neuron("/home/ubuntu/ltx2_vae_tp4_128x512")
        output = pipe(prompt="...", height=384, width=512, num_frames=25)
    """

    def __init__(self, diffusers_pipe, text_seq=1024):
        """
        Args:
            diffusers_pipe: A loaded Diffusers LTX2Pipeline
            text_seq: Maximum text sequence length (must match compile-time)
        """
        self.pipe = diffusers_pipe
        self.text_seq = text_seq
        self.neuron_backbone = None
        self._original_transformer = None
        self._original_vae_decoder = None

    @classmethod
    def from_pretrained(
        cls, model_path, torch_dtype=torch.bfloat16, text_seq=1024, **kwargs
    ):
        """Load the Diffusers LTX2Pipeline from HuggingFace.

        Args:
            model_path: HuggingFace model ID or local path (e.g., "Lightricks/LTX-2")
            torch_dtype: Dtype for CPU components (default bfloat16)
            text_seq: Maximum text sequence length
            **kwargs: Additional args passed to LTX2Pipeline.from_pretrained()
        """
        from diffusers import LTX2Pipeline

        logger.info("Loading Diffusers LTX2Pipeline from %s", model_path)
        diffusers_pipe = LTX2Pipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype, **kwargs
        )
        logger.info(
            "Pipeline components: %s",
            ", ".join(k for k, v in diffusers_pipe.components.items() if v is not None),
        )

        return cls(diffusers_pipe, text_seq=text_seq)

    def _swap_transformer_to_neuron(self):
        """Replace the pipeline's CPU transformer with the Neuron-compiled version.

        Must be called after neuron_backbone.load() succeeds.
        Preserves the CPU preprocessing layers (proj_in, time_embed, etc.)
        and frees the heavy transformer blocks (~38 GB).
        """
        if self.neuron_backbone is None:
            raise RuntimeError("neuron_backbone must be set and loaded before swapping")

        cpu_transformer = self.pipe.transformer
        cpu_transformer.eval()

        # Create the wrapper that adapts pipeline calls to 22 positional args
        wrapper = NeuronTransformerWrapper(
            compiled_backbone=self.neuron_backbone,
            cpu_transformer=cpu_transformer,
            text_seq=self.text_seq,
        )

        # Free the heavy transformer blocks (keep preprocessing layers via wrapper refs)
        self._original_transformer = cpu_transformer
        del cpu_transformer.transformer_blocks
        del cpu_transformer.norm_out, cpu_transformer.proj_out
        del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
        gc.collect()

        # Hot-swap
        self.pipe.transformer = wrapper
        logger.info("Transformer swapped to Neuron backbone")

    def _swap_vae_to_neuron(
        self,
        compiled_dir,
        tile_latent_h=4,
        tile_latent_w=16,
        overlap_latent_h=1,
        overlap_latent_w=0,
        warmup_frames=None,
    ):
        """Replace the pipeline's CPU VAE decoder with Neuron tiled decoder.

        Swaps pipe.vae.decoder with a NeuronTiledVAEDecoder that loads the
        compiled TP model and uses spatial tiling with overlap blending.

        The outer AutoencoderKLLTX2Video.decode() calls:
            self.decoder(hidden_states, temb=None, causal=False)
        Our NeuronTiledVAEDecoder matches this interface.

        Args:
            compiled_dir: Path to compiled TP VAE model (contains tp_0.pt, etc.)
            tile_latent_h: Tile height in latent pixels (default 4)
            tile_latent_w: Tile width in latent pixels (default 16)
            overlap_latent_h: Overlap in latent H (default 1)
            overlap_latent_w: Overlap in latent W (default 0)
            warmup_frames: If set, run warmup immediately with this frame count
        """
        original_decoder = self.pipe.vae.decoder
        self._original_vae_decoder = original_decoder

        neuron_decoder = NeuronTiledVAEDecoder(
            compiled_dir=compiled_dir,
            tile_latent_h=tile_latent_h,
            tile_latent_w=tile_latent_w,
            overlap_latent_h=overlap_latent_h,
            overlap_latent_w=overlap_latent_w,
            original_decoder=original_decoder,
        )

        # Free the original decoder's heavy layers
        del original_decoder
        gc.collect()

        # Hot-swap
        self.pipe.vae.decoder = neuron_decoder
        logger.info("VAE decoder swapped to Neuron tiled decoder")

        if warmup_frames is not None:
            neuron_decoder.warmup(num_frames=warmup_frames)

    def __call__(self, *args, **kwargs):
        """Run the full LTX-2 pipeline (text encoding + denoising + VAE decode)."""
        return self.pipe(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying Diffusers pipeline."""
        if name in (
            "pipe",
            "text_seq",
            "neuron_backbone",
            "_original_transformer",
            "_original_vae_decoder",
        ):
            raise AttributeError(name)
        return getattr(self.pipe, name)
