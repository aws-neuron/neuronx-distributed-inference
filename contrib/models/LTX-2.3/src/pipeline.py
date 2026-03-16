"""
NxDI LTX-2.3 Pipeline
=====================
Neuron-aware pipeline for the LTX-2.3 22B audio-video diffusion model.

Unlike LTX-2 which wrapped the Diffusers LTX2Pipeline, LTX-2.3 uses native
ltx-core components directly (no Diffusers pipeline exists for 2.3).

The pipeline handles:
1. CPU preprocessing via native ltx-core TransformerArgsPreprocessor
2. Routing the 24 flat tensors through the compiled Neuron backbone
3. CPU postprocessing (unpatchify, VAE decode)
4. Euler denoising loop with flow matching scheduler

Text encoding (Gemma 3 12B) and VAE decoding stay on CPU.
Only the DiT transformer backbone runs on Neuron.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NeuronTransformerWrapper(nn.Module):
    """Wraps the compiled Neuron backbone with CPU preprocessing.

    The native ltx-core LTXModel.forward() takes Modality objects and runs:
    1. Preprocessing (patchify, adaln, rope, connector) -> TransformerArgs
    2. 48 transformer blocks
    3. Output projection (norm, scale/shift, linear)

    This wrapper keeps step 1 on CPU (using the original LTXModel's preprocessors)
    and routes steps 2-3 through the compiled Neuron backbone.

    The backbone expects 24 flat tensors (see modeling_ltx23.py forward signature).

    Performance optimizations:
    - AdaLN deduplication: when all tokens share the same sigma (T2V mode),
      the AdaLN MLP is computed once instead of N times. Saves ~47ms/step
      (from 54ms to 7ms for the AdaLN component).
    - Step-invariant caching: RoPE embeddings and context projection are
      computed once on the first step and reused for subsequent steps.
      Saves ~57ms/step (RoPE ~1.6ms + context projection ~55ms).
    """

    def __init__(
        self,
        compiled_backbone,
        cpu_ltx_model,
        text_seq=256,
        mask_4d=False,
        compiled_text_seq=None,
    ):
        """
        Args:
            compiled_backbone: Compiled Neuron model (TensorParallelNeuronModel
                or callable that takes 24 positional tensor args)
            cpu_ltx_model: The full unsharded LTXModel on CPU (for preprocessors)
            text_seq: Maximum text sequence length for preprocessing
            mask_4d: If True, keep attention masks as 4D (B,1,1,seq) for
                Application-compiled NEFFs. Standalone-compiled NEFFs expect
                2D (B,seq) masks.
            compiled_text_seq: Text sequence length the NEFF was compiled with.
                If different from text_seq, context tensors are padded to match.
                Defaults to text_seq if None.
        """
        super().__init__()
        self.compiled_backbone = compiled_backbone
        self.text_seq = text_seq
        self.compiled_text_seq = compiled_text_seq or text_seq
        self.dtype = torch.bfloat16
        self.mask_4d = mask_4d

        # Keep CPU preprocessors from the native model
        self.video_args_preprocessor = cpu_ltx_model.video_args_preprocessor
        self.audio_args_preprocessor = cpu_ltx_model.audio_args_preprocessor

        # Apply AdaLN optimization by default
        self._patch_adaln_dedup()

        # Apply step-invariant caching (RoPE + context projection)
        self._step_cache = {}
        self._patch_step_invariant_cache()

    def _patch_adaln_dedup(self):
        """Monkey-patch the preprocessor's _prepare_timestep to deduplicate AdaLN.

        In T2V mode, all video tokens share the same sigma value, so the AdaLN
        MLP (sinusoidal → Linear 256→4096 → SiLU → Linear 4096→36864) processes
        N identical values. This patch detects uniform timesteps and computes
        the MLP once, then expands. Measured speedup: 54ms → 7ms (8.3x) for
        768 video tokens.

        In I2V mode, there are typically 2 unique sigma values (sigma=0 for
        the conditioning frame, sigma for the rest). The patch handles this
        by computing only the unique values.

        The patch targets the inner TransformerArgsPreprocessor._prepare_timestep
        which is used by both the video and audio preprocessors.
        """

        def _dedup_prepare_timestep(original_fn, preprocessor):
            """Create a deduplicated version of _prepare_timestep."""

            def patched(timestep, adaln, batch_size, hidden_dtype):
                # timestep shape: (B, seq_len) or (B, 1)
                # In T2V: all values identical. In I2V: typically 2 unique values.
                flat = timestep.flatten()
                unique_vals = flat.unique()

                if unique_vals.numel() == flat.numel():
                    # All different — no dedup possible, use original
                    return original_fn(
                        preprocessor, timestep, adaln, batch_size, hidden_dtype
                    )

                # Compute AdaLN MLP only for unique values
                scaled_unique = unique_vals * preprocessor.timestep_scale_multiplier
                ts_unique, et_unique = adaln(scaled_unique, hidden_dtype=hidden_dtype)
                # ts_unique: (n_unique, adaln_dim), et_unique: (n_unique, embed_dim)

                if unique_vals.numel() == 1:
                    # All tokens share same sigma — most common T2V case
                    ts = ts_unique.expand(flat.shape[0], -1)
                    et = et_unique.expand(flat.shape[0], -1)
                else:
                    # Multiple unique values (I2V case) — scatter back
                    # Build index map: for each flat element, which unique index?
                    sorted_unique, sort_idx = unique_vals.sort()
                    bucket = torch.bucketize(flat, sorted_unique)
                    bucket = bucket.clamp(max=len(sorted_unique) - 1)
                    # Map through sort index to get original unique ordering
                    inv_sort = torch.empty_like(sort_idx)
                    inv_sort[sort_idx] = torch.arange(
                        len(sort_idx), device=sort_idx.device
                    )
                    idx = inv_sort[bucket]
                    ts = ts_unique[idx]
                    et = et_unique[idx]

                ts = ts.view(batch_size, -1, ts.shape[-1])
                et = et.view(batch_size, -1, et.shape[-1])
                return ts, et

            return patched

        # Patch the video preprocessor's inner simple_preprocessor
        video_inner = self.video_args_preprocessor.simple_preprocessor
        orig_video = type(video_inner)._prepare_timestep
        video_inner._prepare_timestep = _dedup_prepare_timestep(orig_video, video_inner)

        # Patch the audio preprocessor's inner simple_preprocessor
        audio_inner = self.audio_args_preprocessor.simple_preprocessor
        orig_audio = type(audio_inner)._prepare_timestep
        audio_inner._prepare_timestep = _dedup_prepare_timestep(orig_audio, audio_inner)

        logger.info(
            "AdaLN deduplication patch applied to video and audio preprocessors"
        )

    def _patch_step_invariant_cache(self):
        """Cache RoPE embeddings and context projections across denoising steps.

        Several preprocessing components depend only on positions and context
        (not on the diffusion timestep or latent sample), so they produce
        identical results on every denoising step:

        1. RoPE (positional embeddings): depends on spatial/temporal positions
           which are constant across steps. ~1.6ms × 4 calls = ~1.6ms total.
        2. Context projection (caption_projection Linear): depends on text
           encoder output which is constant across steps. ~55ms for video.
        3. Cross-modal RoPE: same positions, constant. ~0.4ms.
        4. Attention mask preparation: depends on context_mask, constant. ~0.02ms.

        Total savings: ~57ms/step (from ~67ms to ~10ms preprocessing before
        AdaLN dedup, or from ~48ms to ~14ms after AdaLN dedup is already applied).

        The cache is keyed by a string identifier for each call site. Call
        clear_step_cache() between generations with different prompts/resolutions.
        """
        cache = self._step_cache

        def _make_cached_fn(original_fn, preprocessor, cache_prefix):
            """Wrap _prepare_positional_embeddings with caching."""

            def cached_pe(
                positions,
                inner_dim,
                max_pos,
                use_middle_indices_grid,
                num_attention_heads,
                x_dtype,
            ):
                key = f"{cache_prefix}_pe_{inner_dim}_{positions.shape}"
                if key in cache:
                    return cache[key]
                result = original_fn(
                    preprocessor,
                    positions,
                    inner_dim,
                    max_pos,
                    use_middle_indices_grid,
                    num_attention_heads,
                    x_dtype,
                )
                cache[key] = result
                return result

            return cached_pe

        def _make_cached_context(original_fn, preprocessor, cache_prefix):
            """Wrap _prepare_context with caching."""

            def cached_ctx(context, x):
                key = f"{cache_prefix}_ctx"
                if key in cache:
                    return cache[key]
                result = original_fn(preprocessor, context, x)
                cache[key] = result
                return result

            return cached_ctx

        def _make_cached_mask(original_fn, preprocessor, cache_prefix):
            """Wrap _prepare_attention_mask with caching."""

            def cached_mask(context_mask, dtype):
                key = f"{cache_prefix}_mask"
                if key in cache:
                    return cache[key]
                result = original_fn(preprocessor, context_mask, dtype)
                cache[key] = result
                return result

            return cached_mask

        # Patch video preprocessor
        video_inner = self.video_args_preprocessor.simple_preprocessor
        video_cls = type(video_inner)
        video_inner._prepare_positional_embeddings = _make_cached_fn(
            video_cls._prepare_positional_embeddings, video_inner, "video"
        )
        video_inner._prepare_context = _make_cached_context(
            video_cls._prepare_context, video_inner, "video"
        )
        video_inner._prepare_attention_mask = _make_cached_mask(
            video_cls._prepare_attention_mask, video_inner, "video"
        )

        # Patch audio preprocessor
        audio_inner = self.audio_args_preprocessor.simple_preprocessor
        audio_cls = type(audio_inner)
        audio_inner._prepare_positional_embeddings = _make_cached_fn(
            audio_cls._prepare_positional_embeddings, audio_inner, "audio"
        )
        audio_inner._prepare_context = _make_cached_context(
            audio_cls._prepare_context, audio_inner, "audio"
        )
        audio_inner._prepare_attention_mask = _make_cached_mask(
            audio_cls._prepare_attention_mask, audio_inner, "audio"
        )

        logger.info(
            "Step-invariant cache applied (RoPE, context projection, attention mask)"
        )

    def clear_step_cache(self):
        """Clear the step-invariant cache.

        Call this between generations with different prompts, resolutions,
        or frame counts. Not needed between denoising steps (that's the point).
        """
        self._step_cache.clear()

    def preprocess(self, video_modality, audio_modality):
        """Run CPU preprocessing to produce the 24 flat tensors.

        Args:
            video_modality: ltx_core Modality for video
            audio_modality: ltx_core Modality for audio

        Returns:
            Tuple of 24 tensors matching the backbone forward signature.
        """
        with torch.no_grad():
            va = self.video_args_preprocessor.prepare(video_modality, audio_modality)
            aa = self.audio_args_preprocessor.prepare(audio_modality, video_modality)

        dtype = self.dtype

        # Extract RoPE tuples
        video_pe_cos, video_pe_sin = va.positional_embeddings
        audio_pe_cos, audio_pe_sin = aa.positional_embeddings
        ca_video_pe_cos, ca_video_pe_sin = va.cross_positional_embeddings
        ca_audio_pe_cos, ca_audio_pe_sin = aa.cross_positional_embeddings

        # Build the 24 flat tensors
        # Note: context_mask tensors must be distinct objects (different data_ptr)
        # to avoid flattener layout assertion errors in the Neuron JIT wrapper.
        v_mask = va.context_mask
        a_mask = aa.context_mask

        # Attention mask pipeline:
        # - If Modality.context_mask was int64 (from real text encoder or random
        #   with int64 mask), the preprocessor's _prepare_attention_mask converts
        #   it to 4D additive format: (B,1,1,seq) with 0=attend, -max=ignore.
        #   We squeeze to 2D and it's already correct — no further conversion.
        # - If Modality.context_mask was bf16 (e.g., from older code paths),
        #   the preprocessor passes it through unchanged as 2D bf16 binary {0,1}.
        #   We must convert binary→additive: 1→0 (attend), 0→-max (ignore).
        already_additive = False

        if v_mask is not None and v_mask.ndim == 4:
            if self.mask_4d:
                # Application-compiled NEFFs expect 4D masks (B,1,1,seq).
                # Keep as-is; already in additive format from preprocessor.
                already_additive = True
            else:
                # Standalone-compiled NEFFs expect 2D masks (B,seq).
                v_mask = v_mask.squeeze(1).squeeze(1)  # (B,1,1,seq) -> (B,seq)
                already_additive = True
        if a_mask is not None and a_mask.ndim == 4:
            if not self.mask_4d:
                a_mask = a_mask.squeeze(1).squeeze(1)

        # Convert to bf16
        if v_mask is not None:
            v_mask = v_mask.to(dtype)
        if a_mask is not None:
            a_mask = a_mask.to(dtype)

        # Only convert if the mask is still binary (bf16 input case)
        if v_mask is not None and not already_additive:
            if v_mask.ndim == 2:
                # Mask is bf16 binary {0, 1}: convert to additive format
                finfo = torch.finfo(dtype)
                v_mask = torch.where(
                    v_mask > 0.5,
                    torch.zeros_like(v_mask),
                    torch.full_like(v_mask, finfo.min),
                )
                a_mask = torch.where(
                    a_mask > 0.5,
                    torch.zeros_like(a_mask),
                    torch.full_like(a_mask, finfo.min),
                )
                if self.mask_4d:
                    # Expand to 4D for Application path
                    v_mask = v_mask.unsqueeze(1).unsqueeze(1)  # (B,seq) -> (B,1,1,seq)
                    a_mask = a_mask.unsqueeze(1).unsqueeze(1)

        inputs = (
            va.x.to(dtype),
            aa.x.to(dtype),
            va.context.to(dtype),
            aa.context.to(dtype),
            va.timesteps.to(dtype),
            aa.timesteps.to(dtype),
            va.embedded_timestep.to(dtype),
            aa.embedded_timestep.to(dtype),
            va.cross_scale_shift_timestep.to(dtype),
            aa.cross_scale_shift_timestep.to(dtype),
            va.cross_gate_timestep.to(dtype),
            aa.cross_gate_timestep.to(dtype),
            video_pe_cos.to(dtype),
            video_pe_sin.to(dtype),
            audio_pe_cos.to(dtype),
            audio_pe_sin.to(dtype),
            ca_video_pe_cos.to(dtype),
            ca_video_pe_sin.to(dtype),
            ca_audio_pe_cos.to(dtype),
            ca_audio_pe_sin.to(dtype),
            v_mask,
            a_mask.clone(),  # must be distinct tensor object
            va.prompt_timestep.to(dtype),
            aa.prompt_timestep.to(dtype),
        )

        # Pad context/mask tensors if compiled_text_seq > actual text_seq
        if self.compiled_text_seq != self.text_seq:
            inputs = self._pad_context_tensors(inputs)

        return inputs, va, aa

    def _pad_context_tensors(self, inputs):
        """Pad context and mask tensors to match compiled_text_seq.

        The NEFF was compiled with compiled_text_seq context tokens, but the
        actual text encoder output may have fewer tokens. Pad with zeros
        (context) and -inf (masks) so the extra positions are ignored.

        Tensor indices in the 24-input tuple:
            [2]  encoder_hidden_states    (B, text_seq, inner_dim)
            [3]  audio_encoder_hidden_states (B, text_seq, audio_inner_dim)
            [20] encoder_attention_mask   (B, text_seq) or (B,1,1,text_seq)
            [21] audio_encoder_attention_mask (same shape)
        """
        actual_seq = inputs[2].shape[1]
        target_seq = self.compiled_text_seq
        if actual_seq >= target_seq:
            return inputs

        pad_len = target_seq - actual_seq
        dtype = self.dtype
        inputs = list(inputs)

        # Pad context with zeros
        for idx in (2, 3):
            ctx = inputs[idx]
            pad = torch.zeros(ctx.shape[0], pad_len, ctx.shape[2], dtype=dtype)
            inputs[idx] = torch.cat([ctx, pad], dim=1)

        # Pad masks with -inf (masked out)
        finfo = torch.finfo(dtype)
        for idx in (20, 21):
            mask = inputs[idx]
            if mask.ndim == 4:
                # (B, 1, 1, seq) -> pad last dim
                pad = torch.full((mask.shape[0], 1, 1, pad_len), finfo.min, dtype=dtype)
                inputs[idx] = torch.cat([mask, pad], dim=-1)
            elif mask.ndim == 2:
                # (B, seq) -> pad last dim
                pad = torch.full((mask.shape[0], pad_len), finfo.min, dtype=dtype)
                inputs[idx] = torch.cat([mask, pad], dim=-1)

        return tuple(inputs)

    def forward(self, video_modality, audio_modality):
        """Preprocess on CPU, run backbone on Neuron, return (video_out, audio_out).

        Args:
            video_modality: ltx_core Modality for video
            audio_modality: ltx_core Modality for audio

        Returns:
            (video_output, audio_output) tensors from the backbone
        """
        inputs, va, aa = self.preprocess(video_modality, audio_modality)
        video_output, audio_output = self.compiled_backbone(*inputs)
        return video_output, audio_output


class NeuronLTX23Pipeline:
    """Self-contained pipeline for LTX-2.3 on Neuron.

    Orchestrates:
    1. Text encoding (Gemma 3 12B on CPU)
    2. Noise scheduling (flow matching with Euler steps)
    3. Denoising loop (Neuron backbone via NeuronTransformerWrapper)
    4. VAE decoding (video VAE + audio VAE + vocoder on CPU)

    Usage:
        pipe = NeuronLTX23Pipeline(
            ltx_model=cpu_ltx_model,  # full native ltx-core model
            neuron_backbone=compiled_backbone,
            text_encoder=gemma_model,
            embeddings_processor=embeddings_proc,
            tokenizer=tokenizer,
            video_vae=video_vae,
            audio_vae=audio_vae,
            vocoder=vocoder,
        )
        video, audio = pipe(
            prompt="A dog playing in a meadow",
            height=384, width=512, num_frames=25,
            num_inference_steps=8,
        )
    """

    def __init__(
        self,
        ltx_model,
        neuron_backbone,
        text_encoder=None,
        embeddings_processor=None,
        tokenizer=None,
        video_vae=None,
        audio_vae=None,
        vocoder=None,
        text_seq=256,
        mask_4d=False,
        compiled_text_seq=None,
    ):
        """
        Args:
            ltx_model: Native ltx-core LTXModel (for preprocessors, patchify, etc.)
            neuron_backbone: Compiled Neuron backbone (callable with 24 inputs)
            text_encoder: Gemma 3 12B model (on CPU)
            embeddings_processor: EmbeddingsProcessor (feature extractor + connectors)
            tokenizer: LTXVGemmaTokenizer
            video_vae: Video VAE decoder
            audio_vae: Audio VAE decoder
            vocoder: Audio vocoder
            text_seq: Maximum text sequence length
            mask_4d: If True, keep attention masks as 4D for Application-compiled NEFFs
            compiled_text_seq: Text seq len the NEFF was compiled with (defaults to text_seq)
        """
        self.ltx_model = ltx_model
        self.wrapper = NeuronTransformerWrapper(
            compiled_backbone=neuron_backbone,
            cpu_ltx_model=ltx_model,
            text_seq=text_seq,
            mask_4d=mask_4d,
            compiled_text_seq=compiled_text_seq,
        )
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.tokenizer = tokenizer
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.text_seq = text_seq
        self.dtype = torch.bfloat16

    def encode_text(self, prompt, device="cpu"):
        """Encode text prompt using Gemma 3 12B + embeddings processor.

        Returns:
            (video_context, audio_context, context_mask) tensors
        """
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("Text encoder and tokenizer must be set")

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            max_length=self.text_seq,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        # Run Gemma 3 12B
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

        # Run embeddings processor (feature extractor + connectors)
        with torch.no_grad():
            result = self.embeddings_processor.process_hidden_states(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        return result.video_encoding, result.audio_encoding, result.attention_mask

    def velocity_to_denoised(self, sample, velocity, sigma):
        """Convert velocity prediction to denoised sample (flow matching).

        The LTX-2.3 backbone (LTXModel) outputs velocity v, where:
            denoised = sample - v * sigma
        """
        return (sample.to(torch.float32) - velocity.to(torch.float32) * sigma).to(
            sample.dtype
        )

    def denoise_step(self, sample, velocity, sigma, sigma_next):
        """Single Euler step for flow matching diffusion.

        Takes the velocity output from the backbone (NOT denoised).
        Computes: next_sample = sample + velocity * (sigma_next - sigma)
        """
        dt = sigma_next - sigma
        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(
            sample.dtype
        )

    def __call__(
        self,
        prompt: str = "",
        video_context: Optional[torch.Tensor] = None,
        audio_context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        height: int = 384,
        width: int = 512,
        num_frames: int = 25,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        fps: float = 24.0,
        audio_num_frames: int = 26,
        decode_video: bool = True,
        decode_audio: bool = True,
    ):
        """Run the full LTX-2.3 pipeline.

        Either provide pre-computed context tensors or a text prompt.
        If both are provided, the pre-computed tensors are used.

        Returns:
            dict with keys 'video_latent', 'audio_latent', and optionally
            'video' (decoded frames) and 'audio' (decoded waveform).
        """
        # Text encoding (if context not pre-computed)
        if video_context is None:
            if not prompt:
                raise ValueError(
                    "Either prompt or pre-computed context must be provided"
                )
            video_context, audio_context, context_mask = self.encode_text(prompt)

        if context_mask is None:
            context_mask = torch.ones(1, self.text_seq, dtype=self.dtype)

        # Import ltx-core tools for latent creation and scheduling
        from ltx_core.tools import (
            VideoLatentTools,
            VideoLatentPatchifier,
            VideoLatentShape,
            AudioLatentTools,
            AudioPatchifier,
            AudioLatentShape,
            SpatioTemporalScaleFactors,
        )
        from ltx_core.model.transformer.modality import Modality

        # Compute latent dimensions
        # LTX-2.3 VAE downsamples spatially by 32x (not 16x)
        # For 384x512 -> height=12, width=16 latent grid
        latent_h = height // 32
        latent_w = width // 32
        latent_f = (num_frames - 1) // 8 + 1  # temporal downsampling

        video_shape = VideoLatentShape(
            batch=1, channels=128, frames=latent_f, height=latent_h, width=latent_w
        )
        v_patchifier = VideoLatentPatchifier(patch_size=1)
        v_scale = SpatioTemporalScaleFactors.default()  # time=8, height=32, width=32
        video_tools = VideoLatentTools(
            target_shape=video_shape,
            patchifier=v_patchifier,
            scale_factors=v_scale,
            causal_fix=False,
            fps=fps,
        )

        audio_shape = AudioLatentShape(
            batch=1, channels=8, frames=audio_num_frames, mel_bins=16
        )
        a_patchifier = AudioPatchifier(patch_size=16)
        audio_tools = AudioLatentTools(
            patchifier=a_patchifier, target_shape=audio_shape
        )

        # Create initial noise
        video_state = video_tools.create_initial_state(device="cpu", dtype=self.dtype)
        audio_state = audio_tools.create_initial_state(device="cpu", dtype=self.dtype)

        # Initialize with noise
        if generator is not None:
            video_noise = torch.randn(
                video_state.latent.shape,
                dtype=self.dtype,
                generator=generator,
            )
            audio_noise = torch.randn(
                audio_state.latent.shape,
                dtype=self.dtype,
                generator=generator,
            )
        else:
            video_noise = torch.randn_like(video_state.latent)
            audio_noise = torch.randn_like(audio_state.latent)

        # Sigma schedule — use distilled values for the distilled model
        # The distilled model was trained with these exact sigma values
        # See: ltx_pipelines/utils/constants.py DISTILLED_SIGMA_VALUES
        DISTILLED_SIGMA_VALUES = [
            1.0,
            0.99375,
            0.9875,
            0.98125,
            0.975,
            0.909375,
            0.725,
            0.421875,
            0.0,
        ]
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32)
        assert len(sigmas) == num_inference_steps + 1, (
            f"Distilled sigma values have {len(sigmas)} entries "
            f"but {num_inference_steps} steps require {num_inference_steps + 1}"
        )

        # Start from pure noise (sigma=1)
        video_sample = video_noise.clone()
        audio_sample = audio_noise.clone()

        # Clear step-invariant cache from any previous generation
        self.wrapper.clear_step_cache()

        logger.info(
            "Starting denoising: %d steps, sigmas=%s",
            num_inference_steps,
            [f"{s:.4f}" for s in sigmas.tolist()],
        )

        # Denoising loop
        for step_idx in range(num_inference_steps):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]

            # Per-token sigma for the backbone
            video_seq_len = video_state.latent.shape[1]
            audio_seq_len = audio_state.latent.shape[1]
            v_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, video_seq_len)
            a_ts = sigma.unsqueeze(0).unsqueeze(0).expand(1, audio_seq_len)

            # Build Modality objects
            video_mod = Modality(
                latent=video_sample,
                sigma=sigma.unsqueeze(0),
                timesteps=v_ts,
                positions=video_state.positions,
                context=video_context,
                enabled=True,
                context_mask=context_mask,
                attention_mask=None,
            )
            audio_mod = Modality(
                latent=audio_sample,
                sigma=sigma.unsqueeze(0),
                timesteps=a_ts,
                positions=audio_state.positions,
                context=audio_context,
                enabled=True,
                context_mask=context_mask.clone(),  # distinct object
                attention_mask=None,
            )

            # Forward through Neuron backbone (returns VELOCITY, not denoised)
            video_velocity, audio_velocity = self.wrapper(video_mod, audio_mod)

            # Euler step using velocity directly
            video_sample = self.denoise_step(
                video_sample, video_velocity, sigma, sigma_next
            )
            audio_sample = self.denoise_step(
                audio_sample, audio_velocity, sigma, sigma_next
            )

            logger.info(
                "  Step %d/%d: sigma=%.4f -> %.4f",
                step_idx + 1,
                num_inference_steps,
                sigma.item(),
                sigma_next.item(),
            )

        result = {
            "video_latent": video_sample,
            "audio_latent": audio_sample,
        }

        # VAE decode (optional, on CPU)
        if decode_video and self.video_vae is not None:
            with torch.no_grad():
                video_frames = self.video_vae.decode(video_sample)
            result["video"] = video_frames

        if decode_audio and self.audio_vae is not None and self.vocoder is not None:
            with torch.no_grad():
                audio_mel = self.audio_vae.decode(audio_sample)
                audio_waveform = self.vocoder(audio_mel)
            result["audio"] = audio_waveform

        return result
