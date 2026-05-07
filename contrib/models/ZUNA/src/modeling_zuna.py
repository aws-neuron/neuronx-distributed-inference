"""
ZUNA Neuron Modeling Module
============================
Onboards the ZUNA EEG foundation model (Zyphra/ZUNA, 382M-param masked diffusion
autoencoder) to AWS Neuron using torch_neuronx.trace().

Strategy:
  - Patch flex_attention module with dummy symbols BEFORE importing zuna
  - Load EncoderDecoder model from HuggingFace
  - Monkey-patch Attention and CrossAttention to use F.scaled_dot_product_attention
  - Monkey-patch encoder/decoder forward methods to skip create_document_mask
  - Run the rectified flow diffusion loop (Euler ODE, 50 steps default) in Python,
    calling the Neuron-traced decoder at each step

Architecture notes:
  - 16-layer encoder (self-attn + SwiGLU + register interleaving + MMD bottleneck)
  - 16-layer decoder (cross-attn + self-attn + AdaRMSNorm timestep conditioning)
  - 4D axial RoPE over (x, y, z, t_coarse)
  - Rectified flow / flow matching with Euler ODE solver
  - sliding_window=65536 >> seqlen=100, so full attention is correct for inference
"""

import sys
import time
import types
import dataclasses
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# flex_attention patching (must run before any zuna imports)
# ============================================================================


def _comprehensive_flex_attention_patch():
    """Provide all symbols that ZUNA's transformer.py expects from flex_attention.

    ZUNA's transformer.py imports: create_block_mask, BlockMask,
    _mask_mod_signature, noop_mask from torch.nn.attention.flex_attention.
    These symbols exist in PyTorch 2.9 but flex_attention itself is not
    supported on Neuron (XLA device). We provide dummy implementations so
    the import succeeds, then monkey-patch the model to use SDPA instead.
    """
    _mask_mod_signature = Callable

    def noop_mask(b, h, q_idx, kv_idx):
        return True

    class BlockMask:
        def __init__(self, *args, **kwargs):
            pass

    def create_block_mask(*args, **kwargs):
        return BlockMask()

    def flex_attention(*args, **kwargs):
        raise RuntimeError(
            "flex_attention should not be called -- model should use SDPA"
        )

    if "torch.nn.attention" not in sys.modules:
        attn_mod = types.ModuleType("torch.nn.attention")
        sys.modules["torch.nn.attention"] = attn_mod
        torch.nn.attention = attn_mod

    flex_mod = types.ModuleType("torch.nn.attention.flex_attention")
    flex_mod.flex_attention = flex_attention
    flex_mod.create_block_mask = create_block_mask
    flex_mod.BlockMask = BlockMask
    flex_mod._mask_mod_signature = _mask_mod_signature
    flex_mod.noop_mask = noop_mask
    sys.modules["torch.nn.attention.flex_attention"] = flex_mod
    torch.nn.attention.flex_attention = flex_mod

    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.capture_scalar_outputs = True


# Apply patch immediately on import
_comprehensive_flex_attention_patch()


# ============================================================================
# ZUNA imports (safe now that flex_attention is patched)
# ============================================================================
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

import zuna

_zuna_pkg_dir = Path(zuna.__file__).parent
_model_dir = _zuna_pkg_dir / "inference" / "AY2l" / "lingua"
sys.path.insert(0, str(_model_dir))
sys.path.insert(0, str(_model_dir / "apps" / "AY2latent_bci"))

from apps.AY2latent_bci.transformer import (
    EncoderDecoder,
    DecoderTransformerArgs,
)
from apps.AY2latent_bci.xattn import CrossAttention
from lingua.transformer import Attention, apply_rotary_emb


# ============================================================================
# Model loading
# ============================================================================


def dataclass_from_dict(klass, d):
    """Recursively create a dataclass from a dict."""
    fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
    filtered = {}
    for k, v in d.items():
        if k in fieldtypes:
            ft = fieldtypes[k]
            if dataclasses.is_dataclass(ft):
                filtered[k] = dataclass_from_dict(ft, v)
            else:
                filtered[k] = v
    return klass(**filtered)


def load_model(device="cpu"):
    """Load ZUNA model from HuggingFace.

    Returns:
        model: EncoderDecoder model on the specified device
        model_args: DecoderTransformerArgs with model configuration
    """
    import json

    repo_id = "Zyphra/ZUNA"

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Config is nested under "model" key
    model_config = config.get("model", config)
    model_args = dataclass_from_dict(DecoderTransformerArgs, model_config)

    model = EncoderDecoder(model_args)

    weights_path = hf_hub_download(
        repo_id=repo_id, filename="model-00001-of-00001.safetensors"
    )
    state_dict = load_safetensors(weights_path)
    # Strip "model." prefix from state dict keys
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    return model, model_args


# ============================================================================
# Attention patching (flex_attention -> SDPA)
# ============================================================================


def _sdpa_attention_forward(
    self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"
):
    """Replacement Attention.forward using F.scaled_dot_product_attention.

    Drops mask entirely since sliding_window=65536 >> seqlen, making full
    attention correct for single-sample inference.
    """
    bsz, seq_len, _ = x.shape
    xq = self.wq(x)
    xk = self.wk(x)
    xv = self.wv(x)
    output_shape = xq.shape

    xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

    # Apply RoPE (1D or 4D axial)
    if self.rope_dim == 1:
        fc = freq_cis[tok_idx] if tok_idx is not None else freq_cis[0:seq_len]
        xq, xk = apply_rotary_emb(xq, xk, 1, fc)
    elif self.rope_dim == 4:
        if tok_idx is not None:
            fc_parts = [freq_cis[tok_idx[:, i]] for i in range(4)]
        else:
            idx_1d = torch.arange(seq_len, device=x.device)
            fc_parts = [freq_cis[idx_1d] for _ in range(4)]
        freqcis_4 = torch.cat(fc_parts, dim=1)
        xq, xk = apply_rotary_emb(xq, xk, 1, freqcis_4)

    # GQA expansion
    if self.n_kv_heads < self.n_heads:
        n_rep = self.n_heads // self.n_kv_heads
        xk = (
            xk.unsqueeze(3)
            .expand(-1, -1, -1, n_rep, -1)
            .reshape(bsz, seq_len, self.n_heads, self.head_dim)
        )
        xv = (
            xv.unsqueeze(3)
            .expand(-1, -1, -1, n_rep, -1)
            .reshape(bsz, seq_len, self.n_heads, self.head_dim)
        )

    # SDPA: [B, H, S, D]
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    output = F.scaled_dot_product_attention(xq, xk, xv)
    output = output.transpose(1, 2).contiguous().reshape(output_shape)
    return self.wo(output)


def _sdpa_cross_attention_forward(
    self,
    xq_in,
    xkv,
    freq_cis,
    tok_idx=None,
    cross_tok_idx=None,
    mask=None,
    attn_impl="sdpa",
):
    """Replacement CrossAttention.forward using SDPA."""
    bsz, seq_len_q, _ = xq_in.shape
    _, seq_len_kv, _ = xkv.shape

    xq = self.wq(xq_in)
    xk = self.wk(xkv)
    xv = self.wv(xkv)
    output_shape = xq.shape

    xq = xq.view(bsz, seq_len_q, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)

    # Cross-attention RoPE with separate freq_cis for Q and K
    if self.rope_dim == 1:
        from apps.AY2latent_bci.xattn import apply_rotary_emb_xattn

        fc_q = freq_cis[tok_idx] if tok_idx is not None else freq_cis[0:seq_len_q]
        fc_k = (
            freq_cis[cross_tok_idx]
            if cross_tok_idx is not None
            else freq_cis[0:seq_len_kv]
        )
        xq, xk = apply_rotary_emb_xattn(xq, xk, 1, fc_q, fc_k)
    elif self.rope_dim == 4:
        from apps.AY2latent_bci.xattn import apply_rotary_emb_xattn

        if tok_idx is not None:
            fc_q_parts = [freq_cis[tok_idx[:, i]] for i in range(4)]
        else:
            idx_q = torch.arange(seq_len_q, device=xq_in.device)
            fc_q_parts = [freq_cis[idx_q] for _ in range(4)]
        if cross_tok_idx is not None:
            fc_k_parts = [freq_cis[cross_tok_idx[:, i]] for i in range(4)]
        else:
            idx_k = torch.arange(seq_len_kv, device=xq_in.device)
            fc_k_parts = [freq_cis[idx_k] for _ in range(4)]
        freqcis_q = torch.cat(fc_q_parts, dim=1)
        freqcis_k = torch.cat(fc_k_parts, dim=1)
        xq, xk = apply_rotary_emb_xattn(xq, xk, 1, freqcis_q, freqcis_k)

    # GQA expansion
    if self.n_kv_heads < self.n_heads:
        n_rep = self.n_heads // self.n_kv_heads
        xk = (
            xk.unsqueeze(3)
            .expand(-1, -1, -1, n_rep, -1)
            .reshape(bsz, seq_len_kv, self.n_heads, self.head_dim)
        )
        xv = (
            xv.unsqueeze(3)
            .expand(-1, -1, -1, n_rep, -1)
            .reshape(bsz, seq_len_kv, self.n_heads, self.head_dim)
        )

    # SDPA: [B, H, S, D]
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    output = F.scaled_dot_product_attention(xq, xk, xv)
    output = output.transpose(1, 2).contiguous().reshape(output_shape)
    return self.wo(output)


def patch_model_for_neuron(model):
    """Patch ZUNA model for Neuron compatibility.

    1. Replace all Attention.forward with SDPA version
    2. Replace all CrossAttention.forward with SDPA version
    3. Patch encoder.forward to skip create_document_mask
    4. Patch decoder.forward to skip create_document_mask and @torch.compile

    Args:
        model: EncoderDecoder model instance

    Returns:
        The same model, patched in-place
    """
    patched_attn = 0
    patched_xattn = 0

    for name, module in model.named_modules():
        if isinstance(module, CrossAttention):
            module.forward = types.MethodType(_sdpa_cross_attention_forward, module)
            patched_xattn += 1
        elif isinstance(module, Attention):
            module.forward = types.MethodType(_sdpa_attention_forward, module)
            patched_attn += 1

    # Patch encoder forward to skip mask creation
    def _patched_encoder_forward(
        self,
        token_values,
        seq_lens,
        distill_target=None,
        tok_idx=None,
        mask=None,
        attn_impl="sdpa",
        repa_target=None,
        do_idx=None,
        print_layerwise_activation_stats=False,
    ):
        _, orig_seqlen, _ = token_values.shape
        if self.use_compression_free_conv_stem:
            token_values = self.compression_free_conv_stem_input(token_values)
        token_values, num_groups = self._interleave_registers(token_values)
        bsz, seqlen, _ = token_values.shape
        if do_idx is not None:
            do_idx = (token_values.sum(axis=2) == 0).squeeze(0)
        if self.dropout_vec is not None:
            token_values[:, do_idx, :] = self.dropout_vec
        h = self.tok_embeddings(token_values)

        # Skip create_document_mask -- SDPA uses full attention
        mask = None

        if tok_idx is not None:
            # tok_idx must be [1, seqlen, 4] -- repeat_interleave on dim=1
            tok_idx = tok_idx.repeat_interleave(repeats=2, dim=1)
            tok_idx = tok_idx.squeeze().squeeze()

        from apps.AY2latent_bci.transformer import BaseTransformer

        h, repa_loss = BaseTransformer.forward(
            self,
            h,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl="sdpa",
            repa_target=repa_target,
            num_groups=num_groups,
            original_seqlen=orig_seqlen,
            downsample_factor=self.downsample_factor,
            do_idx=do_idx,
        )

        h, non_regs = self._extract_registers_and_non_registers(
            h,
            num_groups,
            original_seqlen=orig_seqlen,
            return_non_registers=distill_target is not None,
        )

        logits = self.output(self.norm(h))
        logits, losses = self.bottleneck(logits)

        if distill_target is not None:
            losses["encoder_distill"] = (
                1
                - F.cosine_similarity(
                    self.distill_output(self.distill_norm(non_regs)),
                    distill_target,
                    dim=-1,
                ).mean()
            ) * 0.1

        if repa_target is not None:
            losses["encoder_repa_loss"] = repa_loss

        return logits, losses

    model.encoder.forward = types.MethodType(_patched_encoder_forward, model.encoder)

    # Patch decoder forward to skip mask creation and @torch.compile
    def _patched_decoder_forward(
        self,
        tokens,
        cross_attended,
        timeD,
        seq_lens,
        cross_seq_lens,
        target=None,
        tok_idx=None,
        cross_tok_idx=None,
        mask=None,
        cross_attn_mask=None,
        attn_impl="sdpa",
        time_masks=None,
        channel_loss_weighting=None,
        repa_target=None,
        freq_masks=None,
        do_idx=None,
        print_layerwise_activation_stats=False,
    ):
        tokens = tokens.squeeze(1)
        bsz, seqlen, dim = tokens.shape
        if self.training and freq_masks is not None:
            with torch.no_grad():
                tokens *= freq_masks
        if self.use_compression_free_conv_stem:
            tokens = self.compression_free_conv_stem_input(tokens)
        h = self.tok_embeddings(tokens)
        t = self.t_embedder(timeD)
        cross_attended = self.encoder_proj(cross_attended)

        # Skip create_document_mask
        mask = None
        cross_attn_mask = None

        if tok_idx is not None:
            if tok_idx.ndim == 3 and tok_idx.shape[0] == 1:
                tok_idx = tok_idx.squeeze().squeeze()
        if cross_tok_idx is not None:
            if cross_tok_idx.ndim == 3 and cross_tok_idx.shape[0] == 1:
                cross_tok_idx = cross_tok_idx.squeeze().squeeze()

        from apps.AY2latent_bci.transformer import BaseTransformerDecoder

        h, repa_loss = BaseTransformerDecoder.forward(
            self,
            h,
            cross_attended,
            t=t,
            tok_idx=tok_idx,
            cross_tok_idx=cross_tok_idx,
            mask=mask,
            cross_attn_mask=cross_attn_mask,
            attn_impl="sdpa",
            repa_target=repa_target,
            do_idx=do_idx,
        )

        h_normed = self.norm(h, t)
        logits = self.output(h_normed)
        if self.use_compression_free_conv_stem:
            logits = self.compression_free_conv_stem_output(logits)

        losses = {}
        if target is not None:
            if self.huber_c is None:
                batchwise_loss = F.mse_loss(
                    target.float(), logits.float(), reduction="none"
                )
            else:
                from apps.AY2latent_bci.transformer import huber_loss

                batchwise_loss = huber_loss(
                    target.float(), logits.float(), self.huber_c
                )
            losses["decoder_rf_loss"] = batchwise_loss.mean()

        if repa_target is not None:
            losses["decoder_repa_loss"] = repa_loss

        return logits, losses

    model.decoder.forward = types.MethodType(_patched_decoder_forward, model.decoder)

    return model


# ============================================================================
# Wrapper modules for torch_neuronx.trace()
# ============================================================================


class EncoderWrapper(nn.Module):
    """Wraps encoder for torch_neuronx.trace() with fixed input shapes.

    Args:
        encoder: The patched EncoderTransformer module
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, token_values, tok_idx):
        """
        Args:
            token_values: [1, seqlen, input_dim]
            tok_idx: [1, seqlen, 4] -- 4D spatial coordinates for axial RoPE
        """
        seqlen = token_values.shape[1]
        seq_lens = torch.tensor([seqlen], device=token_values.device)
        if tok_idx.ndim == 2:
            tok_idx = tok_idx.unsqueeze(0)
        enc_out, _ = self.encoder(
            token_values=token_values,
            seq_lens=seq_lens,
            tok_idx=tok_idx,
        )
        return enc_out


class DecoderWrapper(nn.Module):
    """Wraps decoder for torch_neuronx.trace() with fixed input shapes.

    Args:
        decoder: The patched DecoderTransformer module
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, enc_out, t, tok_idx):
        """
        Args:
            z: [1, seqlen, input_dim] -- noisy latent at current timestep
            enc_out: [1, enc_seqlen, enc_dim] -- encoder output (bottleneck)
            t: [1, 1, 1] -- timestep scalar
            tok_idx: [1, seqlen, 4] -- 4D spatial coordinates
        """
        seqlen = z.shape[1]
        cross_seqlen = enc_out.shape[1]
        seq_lens = torch.tensor([seqlen], device=z.device)
        cross_seq_lens = torch.tensor([cross_seqlen], device=z.device)
        if tok_idx.ndim == 2:
            tok_idx = tok_idx.unsqueeze(0)
        logits, _ = self.decoder(
            tokens=z.unsqueeze(1),
            cross_attended=enc_out,
            timeD=t,
            seq_lens=seq_lens,
            cross_seq_lens=cross_seq_lens,
            tok_idx=tok_idx,
            cross_tok_idx=tok_idx,
        )
        return logits


# ============================================================================
# Synthetic data generation
# ============================================================================


def make_synthetic_input(seqlen=50, input_dim=32, seed=42):
    """Create synthetic EEG-like input with 4D spatial coordinates.

    Args:
        seqlen: Number of time points (electrodes/channels)
        input_dim: Feature dimension per time point
        seed: Random seed for reproducibility

    Returns:
        encoder_input: [1, seqlen, input_dim]
        tok_idx: [1, seqlen, 4] (x, y, z, t_coarse coordinates)
    """
    torch.manual_seed(seed)
    encoder_input = torch.randn(1, seqlen, input_dim) * 0.1
    tok_idx = torch.zeros(1, seqlen, 4, dtype=torch.long)
    for i in range(seqlen):
        tok_idx[0, i, 0] = i % 10  # x position
        tok_idx[0, i, 1] = i % 10  # y position
        tok_idx[0, i, 2] = 0  # z position
        tok_idx[0, i, 3] = i  # t_coarse
    return encoder_input, tok_idx


# ============================================================================
# Diffusion loop
# ============================================================================


def run_diffusion(
    encoder_fn,
    decoder_fn,
    encoder_input,
    tok_idx,
    sample_steps=50,
    cfg=1.0,
    sigma=0.1,
    noise_seed=None,
):
    """Run the rectified flow diffusion loop (Euler ODE solver).

    Args:
        encoder_fn: callable (token_values, tok_idx) -> enc_out
        decoder_fn: callable (z, enc_out, t, tok_idx) -> velocity
        encoder_input: [1, seqlen, input_dim]
        tok_idx: [1, seqlen, 4]
        sample_steps: Number of Euler steps (default 50)
        cfg: Classifier-free guidance scale (1.0 = no guidance)
        sigma: Noise scale for initialization
        noise_seed: Seed for noise initialization

    Returns:
        z: Final denoised output [1, seqlen, input_dim]
        timing: dict with encoder_ms, step_times_ms, total_ms, etc.
    """
    if noise_seed is not None:
        torch.manual_seed(noise_seed)

    with torch.no_grad():
        # Encoder pass (once)
        t0 = time.perf_counter()
        enc_out = encoder_fn(encoder_input, tok_idx)
        enc_ms = (time.perf_counter() - t0) * 1000

        # Initialize noise
        z = sigma * torch.randn_like(encoder_input)
        dt_val = 1.0 / sample_steps

        step_times = []
        for step_num in range(sample_steps, 0, -1):
            t_val = dt_val * step_num
            t_tensor = torch.tensor([[[t_val]]])

            t_step = time.perf_counter()
            vc = decoder_fn(z, enc_out, t_tensor, tok_idx)
            step_ms = (time.perf_counter() - t_step) * 1000
            step_times.append(step_ms)

            if cfg != 1.0:
                vc_uncond = decoder_fn(z, torch.zeros_like(enc_out), t_tensor, tok_idx)
                vc = vc_uncond + cfg * (vc - vc_uncond)

            z = z - dt_val * vc

        total_ms = enc_ms + sum(step_times)

    timing = {
        "encoder_ms": enc_ms,
        "step_times_ms": step_times,
        "total_ms": total_ms,
        "avg_step_ms": sum(step_times) / len(step_times),
        "min_step_ms": min(step_times),
        "max_step_ms": max(step_times),
    }

    return z, timing


# ============================================================================
# High-level API
# ============================================================================


class ZUNANeuronModel:
    """High-level wrapper for ZUNA on Neuron.

    Handles model loading, patching, compilation, and inference.

    Usage::

        model = ZUNANeuronModel()
        model.load_and_patch()
        model.compile(seqlen=50)
        z, timing = model.infer(encoder_input, tok_idx, sample_steps=50)
    """

    def __init__(self):
        self.model = None
        self.model_args = None
        self.encoder_neuron = None
        self.decoder_neuron = None
        self._seqlen = None

    def load_and_patch(self, device="cpu"):
        """Load ZUNA from HuggingFace and apply Neuron patches."""
        self.model, self.model_args = load_model(device=device)
        self.model = patch_model_for_neuron(self.model)
        return self

    def compile(self, seqlen=50, compiler_args=None, save_dir=None):
        """Compile encoder and decoder for Neuron.

        Args:
            seqlen: Sequence length to compile for
            compiler_args: List of compiler args (default: --auto-cast matmult -O2)
            save_dir: Directory to save/load compiled models (optional)
        """
        import torch_neuronx

        if compiler_args is None:
            compiler_args = ["--auto-cast", "matmult", "-O2"]

        input_dim = self.model_args.input_dim
        self._seqlen = seqlen

        # Check for cached models
        if save_dir is not None:
            save_path = Path(save_dir)
            enc_path = save_path / f"encoder_seqlen{seqlen}.pt"
            dec_path = save_path / f"decoder_seqlen{seqlen}.pt"
            if enc_path.exists() and dec_path.exists():
                self.encoder_neuron = torch.jit.load(str(enc_path))
                self.decoder_neuron = torch.jit.load(str(dec_path))
                return self

        # Create example inputs
        example_input, example_tok_idx = make_synthetic_input(
            seqlen=seqlen,
            input_dim=input_dim,
        )

        # Compile encoder
        encoder_wrapper = EncoderWrapper(self.model.encoder)
        encoder_wrapper.eval()
        self.encoder_neuron = torch_neuronx.trace(
            encoder_wrapper,
            (example_input, example_tok_idx),
            compiler_args=compiler_args,
            inline_weights_to_neff=True,
        )

        # Get encoder output shape for decoder example inputs
        with torch.no_grad():
            enc_out_example = self.encoder_neuron(example_input, example_tok_idx)

        # Compile decoder
        decoder_wrapper = DecoderWrapper(self.model.decoder)
        decoder_wrapper.eval()
        example_z = torch.randn(1, seqlen, input_dim)
        example_enc = torch.randn(1, enc_out_example.shape[1], enc_out_example.shape[2])
        example_t = torch.tensor([[[0.5]]])

        self.decoder_neuron = torch_neuronx.trace(
            decoder_wrapper,
            (example_z, example_enc, example_t, example_tok_idx),
            compiler_args=compiler_args,
            inline_weights_to_neff=True,
        )

        # Save if requested
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self.encoder_neuron.save(str(save_path / f"encoder_seqlen{seqlen}.pt"))
            self.decoder_neuron.save(str(save_path / f"decoder_seqlen{seqlen}.pt"))

        return self

    def infer(self, encoder_input, tok_idx, sample_steps=50, cfg=1.0, noise_seed=None):
        """Run full diffusion inference pipeline.

        Args:
            encoder_input: [1, seqlen, input_dim]
            tok_idx: [1, seqlen, 4]
            sample_steps: Number of diffusion steps
            cfg: Classifier-free guidance scale
            noise_seed: Seed for noise initialization

        Returns:
            z: Final denoised output [1, seqlen, input_dim]
            timing: dict with timing breakdown
        """
        sigma = float(getattr(self.model, "global_sigma", 0.1))

        return run_diffusion(
            encoder_fn=lambda tv, ti: self.encoder_neuron(tv, ti),
            decoder_fn=lambda z, e, t, ti: self.decoder_neuron(z, e, t, ti),
            encoder_input=encoder_input,
            tok_idx=tok_idx,
            sample_steps=sample_steps,
            cfg=cfg,
            sigma=sigma,
            noise_seed=noise_seed,
        )
