# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni Token2Wav for NXD inference.
#
# This file contains TWO implementations:
#
# 1. NeuronQwen25OmniToken2Wav (CPU wrapper)
#    - Wraps HF's Qwen2_5OmniToken2WavModel entirely on CPU in float32
#    - Suitable for quick testing or when Neuron resources are limited
#
# 2. NeuronQwen25OmniToken2WavWithNeuronDiT (Neuron-accelerated)
#    - Compiles the DiT (22 transformer blocks) on Neuron via torch_neuronx.trace()
#    - ODE solver loop stays on CPU (inherently sequential, 10-50 steps)
#    - BigVGAN vocoder stays on CPU (convolutional, ~10-20M params)
#    - Speaker encoder (ECAPA-TDNN) stays on CPU (~small)
#    - DiT is the compute bottleneck: 22 blocks × 10 ODE steps = 220 forward passes
#
# Architecture:
#   - DiT (Diffusion Transformer): 22 blocks, dim=1024, 16 heads
#     - ECAPA-TDNN speaker encoder for speaker conditioning
#     - Codec embedding + RoPE + AdaLayerNorm
#     - ODE sampling (Runge-Kutta 4) for mel spectrogram generation
#   - BigVGAN vocoder: mel spectrogram -> waveform
#     - conv_pre(80->1536) + 6 upsample stages + AMPBlock residuals
#     - Snake activation, conv_post(24->1)
#
# Runs on CPU in float32 (required for ODE solver precision).
# Token2Wav has ~809 state dict keys total.

"""Qwen2.5-Omni Token2Wav model for NXD inference."""

import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: CPU-based Token2Wav (HF wrapper)
# =============================================================================


class NeuronQwen25OmniToken2Wav:
    """Wrapper around HF's Qwen2_5OmniToken2WavModel.

    Token2Wav converts codec tokens into audio waveforms through:
      1. DiT model: codec tokens + speaker embedding -> mel spectrogram
         (via ODE sampling with classifier-free guidance)
      2. BigVGAN vocoder: mel spectrogram -> waveform

    Speaker conditioning requires a speaker dict (spk_dict.pt) containing
    per-speaker 'cond' (conditioning) and 'ref_mel' (reference mel) tensors,
    plus 'bos_token' for the Talker.

    This wrapper:
      1. Instantiates the HF Token2Wav from config
      2. Loads weights from converted state dict
      3. Exposes waveform generation API
    """

    def __init__(self, token2wav_config):
        """Initialize Token2Wav.

        Args:
            token2wav_config: Token2Wav config (dict or HF config object).
                Must contain dit_config and bigvgan_config sub-configs.
        """
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniToken2WavConfig,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniToken2WavModel,
        )

        if isinstance(token2wav_config, dict):
            token2wav_config = Qwen2_5OmniToken2WavConfig(**token2wav_config)

        self.model = Qwen2_5OmniToken2WavModel(token2wav_config)
        # Token2Wav must run in float32 for ODE solver precision
        self.model.float()
        self.model.eval()
        self.config = token2wav_config

    @property
    def dtype(self):
        """Return dtype of the underlying HF model (for HF generate compatibility)."""
        return next(self.model.parameters()).dtype

    def float(self):
        """Cast underlying model to float32 (for HF generate compatibility)."""
        self.model.float()
        return self

    def load_state_dict(self, state_dict, strict=True):
        """Load converted state dict into the HF Token2Wav model."""
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def __call__(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """Generate waveform from codec tokens.

        Args:
            code: (batch, seq_len) codec token IDs from the Talker
            conditioning: (batch, mel_len, enc_dim) speaker conditioning
                (from spk_dict.pt 'cond' key)
            reference_mel: (batch, mel_len, mel_dim) reference mel spectrogram
                (from spk_dict.pt 'ref_mel' key)
            num_steps: Number of ODE solver steps (default 10)
            guidance_scale: Classifier-free guidance scale (default 0.5)
            sway_coefficient: Time schedule sway (default -1.0)
            **kwargs: Additional kwargs passed to Token2Wav

        Returns:
            waveform: (samples,) audio waveform tensor on CPU
        """
        return self.model(
            code=code,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
            **kwargs,
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict) -> dict:
        """Convert HF state dict to Token2Wav format.

        Strips 'token2wav.' prefix from keys. Non-token2wav keys are passed through.

        Args:
            state_dict: Full or partial state dict with token2wav.* keys.

        Returns:
            State dict with token2wav prefix stripped.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("token2wav."):
                new_state_dict[key[len("token2wav."):]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def load_speaker_dict(speaker_dict_path):
        """Load speaker dictionary from spk_dict.pt.

        Args:
            speaker_dict_path: Path to spk_dict.pt

        Returns:
            dict: Speaker name -> {cond, ref_mel, bos_token}
        """
        return torch.load(speaker_dict_path, weights_only=True)

    @classmethod
    def from_pretrained_state_dict(cls, token2wav_config, state_dict):
        """Create Token2Wav and load weights from converted state dict.

        Args:
            token2wav_config: Token2Wav config (dict or HF config object)
            state_dict: Already-converted state dict (token2wav keys only)

        Returns:
            Initialized NeuronQwen25OmniToken2Wav
        """
        token2wav = cls(token2wav_config)

        # Filter to only token2wav keys (skip non-token2wav prefixes)
        t2w_keys = {}
        for key, value in state_dict.items():
            if any(
                key.startswith(p)
                for p in [
                    "lm_head.", "visual.", "audio_tower.",
                    "thinker.", "talker.", "token2wav.",
                ]
            ):
                continue
            t2w_keys[key] = value

        missing, unexpected = token2wav.load_state_dict(t2w_keys, strict=False)
        if missing:
            logger.warning("Token2Wav missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Token2Wav unexpected keys: %s", unexpected[:10])
        logger.info("Loaded %d weights into Token2Wav", len(t2w_keys))

        return token2wav


# =============================================================================
# Part 2: Neuron-accelerated Token2Wav (DiT on Neuron)
# =============================================================================


def _monkeypatch_dit_attention_for_neuron(dit_module):
    """Replace DiTAttention.forward with XLA-traceable version.

    Fixes two XLA-incompatible operations in DiTAttention:
    1. In-place slice assignment: query[:, :1], key[:, :1] = ...
       → replaced with torch.cat-based reassembly
    2. ALL_ATTENTION_FUNCTIONS dispatch → explicit matmul attention

    Args:
        dit_module: Qwen2_5OmniToken2WavDiTModel instance
    """
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        apply_rotary_pos_emb,
    )

    for block in dit_module.transformer_blocks:
        attn = block.attn

        def _make_patched(a):
            def forward(hidden_states, position_embeddings=None, attention_mask=None):
                batch_size = hidden_states.shape[0]
                query = a.to_q(hidden_states)
                key = a.to_k(hidden_states)
                value = a.to_v(hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // a.heads
                query = query.view(batch_size, -1, a.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, a.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, a.heads, head_dim).transpose(1, 2)

                # Apply RoPE to first head only (matches HF behavior)
                # FIX: use torch.cat instead of in-place slice assignment
                cos, sin = position_embeddings
                q_rope, k_rope = apply_rotary_pos_emb(
                    query[:, :1], key[:, :1], cos, sin
                )
                query = torch.cat([q_rope, query[:, 1:]], dim=1)
                key = torch.cat([k_rope, key[:, 1:]], dim=1)

                # Explicit matmul attention (XLA-safe, no SDPA dispatch)
                scale = head_dim ** -0.5
                attn_weights = torch.matmul(
                    query, key.transpose(-2, -1)
                ) * scale
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1
                )
                attn_output = torch.matmul(attn_weights, value)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(
                    batch_size, -1, a.heads * head_dim
                )
                attn_output = attn_output.to(query.dtype)

                attn_output = a.to_out[0](attn_output)
                attn_output = a.to_out[1](attn_output)

                return attn_output

            return forward

        attn.forward = _make_patched(attn)


# File path used by the picklable builder `_build_dit_core_for_trace`.
# parallel_model_trace spawns child processes via torch.multiprocessing (start
# method = 'spawn'), so the builder must be importable by fully qualified name,
# and its only dependency — the DiT state dict + original module — has to be
# recoverable inside the child. We stash it on disk (torch.save) and let the
# child reload it: the Qwen2.5-Omni-7B repo weights are already on disk anyway,
# so an extra ~350MB pickle round-trip is acceptable.
_DIT_BUILDER_STASH_PATH: str = ""


def _build_dit_core_for_trace():
    """Module-level, picklable builder for parallel_model_trace.

    Reads the stashed pickle (see `_DIT_BUILDER_STASH_PATH`) that
    `NeuronQwen25OmniToken2WavWithNeuronDiT.compile_dit` wrote before invoking
    parallel_model_trace. The file is a plain ``torch.save`` of
    ``{"dit_module": ..., "state_dict": ..., "block_mask_idx": ...}``.
    """
    import os as _os
    import torch as _torch

    stash_path = _os.environ.get("_QWEN25_OMNI_DIT_STASH", "")
    if not stash_path or not _os.path.isfile(stash_path):
        raise RuntimeError(
            "Builder stash not found. Expected _QWEN25_OMNI_DIT_STASH env var "
            "to point at a torch.save()'d dict written by compile_dit()."
        )

    payload = _torch.load(stash_path, weights_only=False, map_location="cpu")
    dit = payload["dit_module"]
    state_dict = payload["state_dict"]
    block_mask_idx = payload["block_mask_idx"]

    fresh = _NeuronDiTCore(dit)
    fresh.float()
    fresh.eval()
    fresh._block_mask_idx = block_mask_idx
    fresh.load_state_dict(state_dict)
    return fresh


class _NeuronDiTCore(torch.nn.Module):
    """Traced wrapper for DiT transformer blocks + norm_out + proj_out.

    Only the compute-heavy transformer core is compiled on Neuron.
    All preprocessing (time_embed, text_embed, input_embed with ECAPA-TDNN,
    rotary_embed, block_diff) stays on CPU to avoid XLA tracing issues with:
    - ECAPA-TDNN Conv1d(padding="same", padding_mode="reflect")
    - AttentiveStatisticsPooling (dynamic masks, masked_fill(-inf))
    - DiTCodecEmbedding (torch.repeat_interleave)
    - DiTInputEmbedding (2D/3D tensor cat mismatch)
    - Rotary embedding (torch.autocast context manager)

    Each block may have a different attention pattern (look_backward/look_ahead).
    Three per-block masks are pre-computed on CPU:
    - mask_local: look_backward=0, look_ahead=0 (most blocks)
    - mask_backward: look_backward=1, look_ahead=0 (blocks 0, 20)
    - mask_ahead: look_backward=0, look_ahead=1 (block 10)
    """

    def __init__(self, dit_module):
        super().__init__()
        self.transformer_blocks = dit_module.transformer_blocks
        self.norm_out = dit_module.norm_out
        self.proj_out = dit_module.proj_out
        # Build per-block mask selection (Python list for static trace)
        # 0 = mask_local (0,0), 1 = mask_backward (1,0), 2 = mask_ahead (0,1)
        self._block_mask_idx = []
        for block in dit_module.transformer_blocks:
            lb = block.look_backward_block
            la = block.look_ahead_block
            if lb == 0 and la == 0:
                self._block_mask_idx.append(0)
            elif lb >= 1 and la == 0:
                self._block_mask_idx.append(1)
            else:  # la >= 1
                self._block_mask_idx.append(2)

    def forward(self, hidden_states, time_embedding, cos, sin,
                mask_local, mask_backward, mask_ahead):
        """
        Args:
            hidden_states: (batch, seq_len, dim) from input_embed
            time_embedding: (time_batch, dim) from time_embed (broadcasts)
            cos: (batch, seq_len, head_dim) from rotary_embed
            sin: (batch, seq_len, head_dim) from rotary_embed
            mask_local: (batch, 1, seq_len, seq_len) float mask (0,0)
            mask_backward: (batch, 1, seq_len, seq_len) float mask (1,0)
            mask_ahead: (batch, 1, seq_len, seq_len) float mask (0,1)
        """
        position_embeddings = (cos, sin)
        masks = [mask_local, mask_backward, mask_ahead]

        for i, block in enumerate(self.transformer_blocks):
            # Select per-block mask (static Python int index → traced as constant)
            attention_mask = masks[self._block_mask_idx[i]]

            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(
                hidden_states, emb=time_embedding
            )
            attn_output = block.attn(
                hidden_states=norm,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
            norm = (
                block.ff_norm(hidden_states) * (1 + scale_mlp[:, None])
                + shift_mlp[:, None]
            )
            ff_output = block.ff(norm)
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)
        return output


class NeuronQwen25OmniToken2WavWithNeuronDiT(NeuronQwen25OmniToken2Wav):
    """Token2Wav with DiT transformer core compiled on Neuron.

    The DiT is the compute bottleneck in Token2Wav:
      - 22 transformer blocks × 10 ODE steps = 220 forward passes per generation
      - Each block: self-attention (dim=1024, 16 heads) + FFN
      - Total ~85M params

    Split architecture (CPU preprocessing + Neuron transformer core):
      CPU: time_embed(time_step) → time embedding
      CPU: text_embed(quantized_code) → codec features
      CPU: input_embed(hidden_states, speaker_embedding, condition_vector, code_embed)
           → includes ECAPA-TDNN speaker encoder, CFG batch doubling
      CPU: rotary_embed(hidden_states) → cos, sin
      CPU: _create_block_diff(hidden_states) → attention mask
      Neuron: 22 transformer blocks + norm_out + proj_out (the compute hotspot)
      CPU: ODE solver loop (Runge-Kutta 4, 10 steps)
      CPU: BigVGAN vocoder → waveform

    Usage:
      t2w = NeuronQwen25OmniToken2WavWithNeuronDiT(token2wav_config)
      t2w.load_state_dict(state_dict)
      t2w.compile_dit("compiled_dit/", max_mel_len=2048)
      # Subsequent runs:
      t2w.load_dit("compiled_dit/")
      waveform = t2w(code, conditioning, reference_mel)
    """

    def __init__(self, token2wav_config):
        super().__init__(token2wav_config)
        self._neuron_dit_core = None
        self._dit_compiled_path = None
        self._dit_max_mel_len = None
        self._dit_batch_size = None

    def _get_dit_module(self):
        """Extract the DiT sub-module from the HF Token2Wav model."""
        for attr_name in [
            "code2wav_dit_model", "dit", "flow_model", "transformer",
        ]:
            if hasattr(self.model, attr_name):
                return getattr(self.model, attr_name)
        for name, module in self.model.named_children():
            if "dit" in name.lower() or "flow" in name.lower():
                return module
        return None

    def compile_dit(
        self,
        compiled_path,
        max_mel_len=2048,
        batch_size=2,
        tp_degree=4,
    ):
        """Compile the DiT transformer core on Neuron using TP=4.

        Only the 22 transformer blocks + norm_out + proj_out are compiled.
        Preprocessing (ECAPA-TDNN, codec embedding, input embedding, rotary)
        stays on CPU to avoid XLA tracing issues.

        Uses ``neuronx_distributed.trace.parallel_model_trace`` (replicated,
        not sharded) so the DiT lives on the same NeuronCore group (0..tp_degree-1)
        as the Thinker and Talker. This matters when all three models share one
        Python process: a single-device ``torch_neuronx.trace`` NEFF gets
        placed on a separate core group and pays a cross-group scheduling
        penalty (~4s per DiT forward on trn2.48xlarge). Replicating onto
        the same TP group makes the NeuronCore runtime treat all three as
        peers on the same logical device set.

        DiT itself is small (~85M params) so there is no memory win from
        sharding the linears; the win is purely co-location.

        Args:
            compiled_path: Directory to save compiled model.
            max_mel_len: Maximum mel spectrogram length (covers ~24s audio).
                Shorter inputs are padded; longer inputs fall back to CPU.
            batch_size: Batch size for compilation. Use 2 for standard
                inference with classifier-free guidance (CFG doubles batch).
            tp_degree: Replication degree; should match the Thinker/Talker
                ``tp_degree`` so all three live on the same NeuronCore group.
        """
        try:
            from neuronx_distributed.trace import parallel_model_trace
        except ImportError:
            raise ImportError(
                "neuronx_distributed required for DiT compilation. "
                "Run on a Neuron instance with the NxDI venv active."
            )

        os.makedirs(compiled_path, exist_ok=True)

        dit = self._get_dit_module()
        if dit is None:
            raise RuntimeError("Could not extract DiT module from Token2Wav.")

        # Get model dimensions
        dit_cfg = getattr(dit, "config", None)
        dim = getattr(dit_cfg, "dim", 1024)
        num_heads = getattr(dit_cfg, "num_attention_heads", 16)
        head_dim = dim // num_heads

        logger.info(
            "Compiling DiT core: batch=%d, mel_len=%d, dim=%d, heads=%d, tp=%d",
            batch_size, max_mel_len, dim, num_heads, tp_degree,
        )

        # Monkeypatch DiTAttention to fix in-place slice assignment
        _monkeypatch_dit_attention_for_neuron(dit)

        # Capture state dict so parallel_model_trace's builder can
        # reconstruct _NeuronDiTCore on the XLA device with the right weights.
        # The builder must be a module-level function (see _build_dit_core_for_trace
        # above) because parallel_model_trace pickles the builder across spawn'd
        # processes. Since spawn'd children don't inherit globals, we write the
        # inputs to a temp file and point the child at it via an env var.
        core_template = _NeuronDiTCore(dit)
        core_template.float()
        core_template.eval()

        stash_path = os.path.join(compiled_path, "_dit_builder_stash.pt")
        torch.save(
            {
                "dit_module": dit,
                "state_dict": core_template.state_dict(),
                "block_mask_idx": core_template._block_mask_idx,
            },
            stash_path,
        )
        os.environ["_QWEN25_OMNI_DIT_STASH"] = stash_path

        # Create example inputs
        # time_embedding uses batch=1 (broadcasts to hidden_states batch)
        hidden_states = torch.randn(
            batch_size, max_mel_len, dim, dtype=torch.float32
        )
        time_embedding = torch.randn(1, dim, dtype=torch.float32)
        cos = torch.randn(
            batch_size, max_mel_len, head_dim, dtype=torch.float32
        )
        sin = torch.randn(
            batch_size, max_mel_len, head_dim, dtype=torch.float32
        )
        # Three per-block attention masks (local, backward, ahead)
        mask_local = torch.zeros(
            batch_size, 1, max_mel_len, max_mel_len, dtype=torch.float32
        )
        mask_backward = torch.zeros(
            batch_size, 1, max_mel_len, max_mel_len, dtype=torch.float32
        )
        mask_ahead = torch.zeros(
            batch_size, 1, max_mel_len, max_mel_len, dtype=torch.float32
        )

        try:
            compiled = parallel_model_trace(
                _build_dit_core_for_trace,
                (hidden_states, time_embedding, cos, sin,
                 mask_local, mask_backward, mask_ahead),
                tp_degree=tp_degree,
                compiler_args=[
                    "--auto-cast=none",
                    "--model-type=transformer",
                    "-O1",
                ],
            )
        finally:
            os.environ.pop("_QWEN25_OMNI_DIT_STASH", None)
            try:
                os.remove(stash_path)
            except OSError:
                pass

        # parallel_model_trace produces a ParallelModel that serializes as a
        # directory (multiple per-rank artifacts), not a single .pt file.
        save_dir = os.path.join(compiled_path, "dit_core_parallel")
        os.makedirs(save_dir, exist_ok=True)
        from neuronx_distributed.trace import parallel_model_save
        parallel_model_save(compiled, save_dir)

        # Save metadata for load
        meta = {
            "max_mel_len": max_mel_len,
            "batch_size": batch_size,
            "dim": dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "tp_degree": tp_degree,
        }
        with open(os.path.join(compiled_path, "dit_core_meta.json"), "w") as f:
            json.dump(meta, f)

        logger.info("Compiled DiT core (TP=%d) saved to %s", tp_degree, save_dir)

        self._neuron_dit_core = compiled
        self._dit_compiled_path = compiled_path
        self._dit_max_mel_len = max_mel_len
        self._dit_batch_size = batch_size

    def load_dit(self, compiled_path):
        """Load a previously compiled DiT core model.

        Supports both the old single-device ``torch.jit`` artifact (filename
        ``dit_core_neuron.pt``) and the new TP-replicated ``parallel_model``
        artifact (directory ``dit_core_parallel/``). Loading a legacy
        single-device artifact will work but pays the cross-core-group
        scheduling penalty described in ``compile_dit``.
        """
        meta_path = os.path.join(compiled_path, "dit_core_meta.json")
        parallel_dir = os.path.join(compiled_path, "dit_core_parallel")
        legacy_path = os.path.join(compiled_path, "dit_core_neuron.pt")

        if os.path.isdir(parallel_dir):
            from neuronx_distributed.trace import parallel_model_load
            self._neuron_dit_core = parallel_model_load(parallel_dir)
            logger.info("Loaded TP-replicated DiT core from %s", parallel_dir)
        elif os.path.exists(legacy_path):
            self._neuron_dit_core = torch.jit.load(legacy_path)
            logger.warning(
                "Loaded legacy single-device DiT core from %s; recompile with "
                "compile_dit() to get the TP=4 replicated artifact and avoid "
                "cross-core-group scheduling overhead when running alongside "
                "the Thinker and Talker.",
                legacy_path,
            )
        else:
            raise FileNotFoundError(
                f"Compiled DiT core not found at {parallel_dir} or {legacy_path}"
            )

        self._dit_compiled_path = compiled_path

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._dit_max_mel_len = meta["max_mel_len"]
            self._dit_batch_size = meta["batch_size"]

    def _build_attention_masks(self, block_diff, actual_mel_len, max_mel_len):
        """Build three per-block float additive attention masks with padding.

        DiT blocks have three distinct attention patterns based on their
        look_backward_block/look_ahead_block attributes:
          - mask_local (0,0): most blocks — attend only within same block
          - mask_backward (1,0): blocks 0,20 — attend current + previous block
          - mask_ahead (0,1): block 10 — attend current + next block

        Args:
            block_diff: (batch, heads, actual_mel_len, actual_mel_len)
                from _create_block_diff. Values are block index differences.
            actual_mel_len: Actual sequence length before padding
            max_mel_len: Padded sequence length (compiled size)

        Returns:
            Tuple of 3 masks, each (batch, 1, max_mel_len, max_mel_len) float32.
            0.0 for attend, -1e4 for don't attend.
        """
        mask_batch = self._dit_batch_size or 1
        bd = block_diff[0, 0]  # (actual_mel_len, actual_mel_len)

        # Three patterns: (look_backward, look_ahead)
        patterns = [(0, 0), (1, 0), (0, 1)]
        masks = []
        for lb, la in patterns:
            bool_mask = (bd >= -float(lb)) & (bd <= float(la))
            valid = torch.where(
                bool_mask,
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(-1e4, dtype=torch.float32),
            )
            mask = torch.full(
                (mask_batch, 1, max_mel_len, max_mel_len),
                -1e4, dtype=torch.float32,
            )
            for b in range(mask_batch):
                mask[b, 0, :actual_mel_len, :actual_mel_len] = valid
            masks.append(mask)

        return masks[0], masks[1], masks[2]

    @torch.no_grad()
    def __call__(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """Generate waveform. DiT core runs on Neuron if compiled, else CPU.

        Monkeypatches dit.forward to split execution:
        - CPU: preprocessing (time/text/input embed, rotary, block_diff)
        - Neuron: 22 transformer blocks + norm + proj
        - CPU: ODE solver, BigVGAN vocoder
        """
        if self._neuron_dit_core is not None:
            dit = self._get_dit_module()
            original_forward = dit.forward
            neuron_core = self._neuron_dit_core
            max_mel_len = self._dit_max_mel_len
            expected_batch = self._dit_batch_size
            build_masks = self._build_attention_masks

            def neuron_dit_forward(
                hidden_states,
                condition_vector,
                speaker_embedding,
                quantized_code,
                time_step,
                drop_audio_conditioning=False,
                drop_code=False,
                apply_cfg=True,
            ):
                """DiT forward with Neuron-accelerated transformer core."""
                batch_size = hidden_states.shape[0]
                if time_step.ndim == 0:
                    time_step = time_step.repeat(batch_size)

                # Estimate mel_len early. input_embed doubles batch for
                # CFG, so we must fall back BEFORE calling it — otherwise
                # original_forward would receive already-modified tensors.
                est_mel_len = hidden_states.shape[1]
                if est_mel_len > max_mel_len:
                    logger.warning(
                        "mel_len %d > max %d, falling back to CPU",
                        est_mel_len,
                        max_mel_len,
                    )
                    return original_forward(
                        hidden_states,
                        condition_vector,
                        speaker_embedding,
                        quantized_code,
                        time_step,
                        drop_audio_conditioning=drop_audio_conditioning,
                        drop_code=drop_code,
                        apply_cfg=apply_cfg,
                    )

                # CPU: compute embeddings (same as HF original)
                time_embedding = dit.time_embed(time_step)
                text_embedding = dit.text_embed(
                    quantized_code,
                    drop_code=False if apply_cfg else drop_code,
                )
                text_embedding_uncond = (
                    dit.text_embed(quantized_code, drop_code=True)
                    if apply_cfg
                    else None
                )

                # CPU: input embedding (ECAPA-TDNN, CFG batch doubling)
                hidden_states = dit.input_embed(
                    hidden_states,
                    speaker_embedding,
                    condition_vector,
                    text_embedding,
                    drop_audio_cond=drop_audio_conditioning,
                    code_embed_uncond=text_embedding_uncond,
                    apply_cfg=apply_cfg,
                )

                # CPU: positional encodings
                cos, sin = dit.rotary_embed(hidden_states)
                block_diff = dit._create_block_diff(hidden_states)

                actual_mel_len = hidden_states.shape[1]
                actual_batch = hidden_states.shape[0]

                # Build three per-block attention masks
                mask_local, mask_backward, mask_ahead = build_masks(
                    block_diff, actual_mel_len, max_mel_len
                )

                # Pad to compiled shapes
                pad_mel = max_mel_len - actual_mel_len
                if pad_mel > 0:
                    hidden_states = torch.nn.functional.pad(
                        hidden_states, (0, 0, 0, pad_mel)
                    )
                    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad_mel))
                    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad_mel))

                pad_batch = expected_batch - actual_batch
                if pad_batch > 0:
                    hidden_states = torch.nn.functional.pad(
                        hidden_states, (0, 0, 0, 0, 0, pad_batch)
                    )
                    cos = torch.nn.functional.pad(
                        cos, (0, 0, 0, 0, 0, pad_batch)
                    )
                    sin = torch.nn.functional.pad(
                        sin, (0, 0, 0, 0, 0, pad_batch)
                    )

                # Run transformer core on Neuron (3 per-block masks)
                output = neuron_core(
                    hidden_states.float(),
                    time_embedding.float(),
                    cos.float(),
                    sin.float(),
                    mask_local.float(),
                    mask_backward.float(),
                    mask_ahead.float(),
                )

                # Unpad to actual sizes
                output = output[:actual_batch, :actual_mel_len]
                return output

            dit.forward = neuron_dit_forward
            try:
                result = self.model(
                    code=code,
                    conditioning=conditioning,
                    reference_mel=reference_mel,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    sway_coefficient=sway_coefficient,
                    **kwargs,
                )
            finally:
                dit.forward = original_forward
            return result
        else:
            return self.model(
                code=code,
                conditioning=conditioning,
                reference_mel=reference_mel,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                sway_coefficient=sway_coefficient,
                **kwargs,
            )

    @classmethod
    def from_pretrained_state_dict(cls, token2wav_config, state_dict):
        """Create Token2Wav with Neuron DiT support."""
        token2wav = cls(token2wav_config)

        t2w_keys = {}
        for key, value in state_dict.items():
            if any(
                key.startswith(p)
                for p in [
                    "lm_head.", "visual.", "audio_tower.",
                    "thinker.", "talker.", "token2wav.",
                ]
            ):
                continue
            t2w_keys[key] = value

        missing, unexpected = token2wav.load_state_dict(t2w_keys, strict=False)
        if missing:
            logger.warning("Token2Wav missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Token2Wav unexpected keys: %s", unexpected[:10])
        logger.info(
            "Loaded %d weights into Token2Wav (Neuron DiT capable)",
            len(t2w_keys),
        )

        return token2wav
