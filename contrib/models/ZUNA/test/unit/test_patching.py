"""
Unit tests for ZUNA Neuron patching, wrappers, and diffusion logic.

These tests run on CPU only -- no Neuron hardware or torch_neuronx required.
They validate that the flex_attention patching, model loading, SDPA monkey-
patching, encoder/decoder wrappers, and rectified flow diffusion loop all
work correctly before compilation.

Usage:
    pytest test_patching.py -v

Prerequisites:
    pip install zuna pytest
"""

import sys
import time
import types
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQLEN = 50
INPUT_DIM = 32
SAMPLE_STEPS = 10  # fewer steps for fast CPU tests


# ---------------------------------------------------------------------------
# flex_attention patching (must run before zuna imports)
# ---------------------------------------------------------------------------


def _comprehensive_flex_attention_patch():
    """Provide dummy flex_attention symbols for ZUNA's transformer.py."""
    _mask_mod_signature = Callable

    def noop_mask(b, h, q_idx, kv_idx):
        return True

    class BlockMask:
        def __init__(self, *args, **kwargs):
            pass

    def create_block_mask(*args, **kwargs):
        return BlockMask()

    def flex_attention(*args, **kwargs):
        raise RuntimeError("flex_attention should not be called")

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


_comprehensive_flex_attention_patch()


# ---------------------------------------------------------------------------
# ZUNA imports
# ---------------------------------------------------------------------------
import dataclasses
import json
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


# ---------------------------------------------------------------------------
# Helpers (duplicated from integration test to keep unit tests standalone)
# ---------------------------------------------------------------------------


def dataclass_from_dict(klass, d):
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


def _sdpa_attention_forward(
    self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"
):
    bsz, seq_len, _ = x.shape
    xq = self.wq(x)
    xk = self.wk(x)
    xv = self.wv(x)
    output_shape = xq.shape
    xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

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
    bsz, seq_len_q, _ = xq_in.shape
    _, seq_len_kv, _ = xkv.shape
    xq = self.wq(xq_in)
    xk = self.wk(xkv)
    xv = self.wv(xkv)
    output_shape = xq.shape
    xq = xq.view(bsz, seq_len_q, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len_kv, self.n_kv_heads, self.head_dim)

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

    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    output = F.scaled_dot_product_attention(xq, xk, xv)
    output = output.transpose(1, 2).contiguous().reshape(output_shape)
    return self.wo(output)


def _patch_model(model):
    """Apply all Neuron-compatibility patches to ZUNA model."""
    for name, module in model.named_modules():
        if isinstance(module, CrossAttention):
            module.forward = types.MethodType(_sdpa_cross_attention_forward, module)
        elif isinstance(module, Attention):
            module.forward = types.MethodType(_sdpa_attention_forward, module)

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
        mask = None
        if tok_idx is not None:
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


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, token_values, tok_idx):
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
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, enc_out, t, tok_idx):
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


def make_synthetic_input(seqlen=SEQLEN, input_dim=INPUT_DIM, seed=42):
    torch.manual_seed(seed)
    encoder_input = torch.randn(1, seqlen, input_dim) * 0.1
    tok_idx = torch.zeros(1, seqlen, 4, dtype=torch.long)
    for i in range(seqlen):
        tok_idx[0, i, 0] = i % 10
        tok_idx[0, i, 1] = i % 10
        tok_idx[0, i, 2] = 0
        tok_idx[0, i, 3] = i
    return encoder_input, tok_idx


def run_diffusion(
    encoder_fn,
    decoder_fn,
    encoder_input,
    tok_idx,
    sample_steps=SAMPLE_STEPS,
    sigma=0.1,
    noise_seed=None,
):
    """Run rectified flow diffusion loop on CPU."""
    if noise_seed is not None:
        torch.manual_seed(noise_seed)

    with torch.no_grad():
        enc_out = encoder_fn(encoder_input, tok_idx)
        z = sigma * torch.randn_like(encoder_input)
        dt_val = 1.0 / sample_steps

        for step_num in range(sample_steps, 0, -1):
            t_val = dt_val * step_num
            t_tensor = torch.tensor([[[t_val]]])
            vc = decoder_fn(z, enc_out, t_tensor, tok_idx)
            z = z - dt_val * vc

    return z


def load_cpu_model():
    """Load ZUNA model on CPU with Neuron patches applied."""
    config_path = hf_hub_download(repo_id="Zyphra/ZUNA", filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    model_config = config.get("model", config)
    model_args = dataclass_from_dict(DecoderTransformerArgs, model_config)
    model = EncoderDecoder(model_args)

    weights_path = hf_hub_download(
        repo_id="Zyphra/ZUNA", filename="model-00001-of-00001.safetensors"
    )
    state_dict = load_safetensors(weights_path)
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _patch_model(model)

    return model, model_args


# ---------------------------------------------------------------------------
# Fixtures (module-scoped -- model loads once for all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_model_and_args():
    return load_cpu_model()


@pytest.fixture(scope="module")
def cpu_model(cpu_model_and_args):
    return cpu_model_and_args[0]


@pytest.fixture(scope="module")
def model_args(cpu_model_and_args):
    return cpu_model_and_args[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlexAttentionPatch:
    """Verify that the flex_attention patch installs all required symbols."""

    def test_module_exists(self):
        """torch.nn.attention.flex_attention module is present."""
        assert "torch.nn.attention.flex_attention" in sys.modules

    def test_create_block_mask_callable(self):
        """create_block_mask is callable and returns a BlockMask."""
        from torch.nn.attention.flex_attention import create_block_mask, BlockMask

        result = create_block_mask()
        assert isinstance(result, BlockMask)

    def test_noop_mask_callable(self):
        """noop_mask returns True."""
        from torch.nn.attention.flex_attention import noop_mask

        assert noop_mask(0, 0, 0, 0) is True

    def test_mask_mod_signature_exists(self):
        """_mask_mod_signature is available."""
        from torch.nn.attention.flex_attention import _mask_mod_signature

        assert _mask_mod_signature is not None

    def test_flex_attention_raises(self):
        """flex_attention itself raises RuntimeError (should never be called)."""
        from torch.nn.attention.flex_attention import flex_attention

        with pytest.raises(RuntimeError, match="should not be called"):
            flex_attention()


class TestModelLoading:
    """Verify model loads from HuggingFace and patches apply correctly."""

    def test_model_loads(self, cpu_model):
        """Model loads without error."""
        assert cpu_model is not None

    def test_parameter_count(self, cpu_model):
        """Model has expected ~382M parameters."""
        param_count = sum(p.numel() for p in cpu_model.parameters())
        assert 350e6 < param_count < 420e6, f"Unexpected param count: {param_count}"

    def test_model_in_eval_mode(self, cpu_model):
        """Model is in eval mode after loading."""
        assert not cpu_model.training

    def test_model_args_input_dim(self, model_args):
        """Model config has expected input_dim=32."""
        assert model_args.input_dim == INPUT_DIM


class TestPatching:
    """Verify that SDPA monkey-patching was applied to all attention modules."""

    def test_self_attention_patched(self, cpu_model):
        """All Attention modules use the SDPA forward method."""
        attn_count = 0
        for name, module in cpu_model.named_modules():
            if isinstance(module, Attention):
                attn_count += 1
                # The patched forward is a bound method of _sdpa_attention_forward
                assert (
                    "sdpa" in module.forward.__func__.__name__
                    or "sdpa" in module.forward.__func__.__qualname__
                ), f"Attention {name} not patched"
        assert attn_count == 32, f"Expected 32 Attention modules, found {attn_count}"

    def test_cross_attention_patched(self, cpu_model):
        """All CrossAttention modules use the SDPA forward method."""
        xattn_count = 0
        for name, module in cpu_model.named_modules():
            if isinstance(module, CrossAttention):
                xattn_count += 1
                assert (
                    "sdpa" in module.forward.__func__.__name__
                    or "sdpa" in module.forward.__func__.__qualname__
                ), f"CrossAttention {name} not patched"
        assert xattn_count == 16, (
            f"Expected 16 CrossAttention modules, found {xattn_count}"
        )


class TestSyntheticInput:
    """Verify synthetic input generation."""

    def test_shapes(self):
        """make_synthetic_input returns tensors with correct shapes."""
        ei, ti = make_synthetic_input(seqlen=50, input_dim=32)
        assert ei.shape == (1, 50, 32)
        assert ti.shape == (1, 50, 4)

    def test_tok_idx_dtype(self):
        """tok_idx is long dtype for indexing."""
        _, ti = make_synthetic_input()
        assert ti.dtype == torch.long

    def test_deterministic(self):
        """Same seed produces identical inputs."""
        ei1, ti1 = make_synthetic_input(seed=42)
        ei2, ti2 = make_synthetic_input(seed=42)
        assert torch.equal(ei1, ei2)
        assert torch.equal(ti1, ti2)

    def test_different_seeds(self):
        """Different seeds produce different inputs."""
        ei1, _ = make_synthetic_input(seed=42)
        ei2, _ = make_synthetic_input(seed=99)
        assert not torch.equal(ei1, ei2)


class TestEncoderWrapper:
    """Verify the encoder wrapper produces valid output on CPU."""

    def test_output_shape(self, cpu_model):
        """Encoder wrapper produces 3D output."""
        enc = EncoderWrapper(cpu_model.encoder)
        enc.eval()
        ei, ti = make_synthetic_input()
        with torch.no_grad():
            out = enc(ei, ti)
        assert out.ndim == 3
        assert out.shape[0] == 1  # batch

    def test_output_finite(self, cpu_model):
        """Encoder output contains no NaN or Inf."""
        enc = EncoderWrapper(cpu_model.encoder)
        enc.eval()
        ei, ti = make_synthetic_input()
        with torch.no_grad():
            out = enc(ei, ti)
        assert torch.isfinite(out).all(), "Encoder output contains NaN or Inf"

    def test_tok_idx_2d_auto_unsqueeze(self, cpu_model):
        """Encoder handles 2D tok_idx by auto-adding batch dim."""
        enc = EncoderWrapper(cpu_model.encoder)
        enc.eval()
        ei, ti = make_synthetic_input()
        ti_2d = ti.squeeze(0)  # [seqlen, 4]
        with torch.no_grad():
            out = enc(ei, ti_2d)
        assert out.ndim == 3


class TestDecoderWrapper:
    """Verify the decoder wrapper produces valid output on CPU."""

    def test_output_shape(self, cpu_model):
        """Decoder wrapper output matches input shape [1, seqlen, input_dim]."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input()
        with torch.no_grad():
            enc_out = enc(ei, ti)
            z = torch.randn(1, SEQLEN, INPUT_DIM)
            t = torch.tensor([[[0.5]]])
            out = dec(z, enc_out, t, ti)
        assert out.shape == (1, SEQLEN, INPUT_DIM)

    def test_output_finite(self, cpu_model):
        """Decoder output contains no NaN or Inf."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input()
        with torch.no_grad():
            enc_out = enc(ei, ti)
            z = torch.randn(1, SEQLEN, INPUT_DIM)
            t = torch.tensor([[[0.5]]])
            out = dec(z, enc_out, t, ti)
        assert torch.isfinite(out).all(), "Decoder output contains NaN or Inf"

    def test_different_timesteps_different_output(self, cpu_model):
        """Decoder produces different output for different timesteps."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input()
        with torch.no_grad():
            enc_out = enc(ei, ti)
            z = torch.randn(1, SEQLEN, INPUT_DIM)
            out_t1 = dec(z, enc_out, torch.tensor([[[0.1]]]), ti)
            out_t9 = dec(z, enc_out, torch.tensor([[[0.9]]]), ti)
        assert not torch.equal(out_t1, out_t9), (
            "Decoder should produce different output at different timesteps"
        )


class TestDiffusionLoop:
    """Verify the rectified flow diffusion loop math on CPU."""

    def test_output_shape(self, cpu_model):
        """Diffusion output shape matches encoder input shape."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input()
        z = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=5,
            noise_seed=42,
        )
        assert z.shape == (1, SEQLEN, INPUT_DIM)

    def test_output_finite(self, cpu_model):
        """Diffusion output is finite (no divergence)."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input()
        z = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=10,
            noise_seed=42,
        )
        assert torch.isfinite(z).all(), "Diffusion output diverged (NaN/Inf)"

    def test_deterministic_with_seed(self, cpu_model):
        """Same noise_seed produces identical diffusion output."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input(seed=42)
        z1 = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=5,
            noise_seed=123,
        )
        z2 = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=5,
            noise_seed=123,
        )
        assert torch.equal(z1, z2), "Diffusion should be deterministic with same seed"

    def test_different_seeds_different_output(self, cpu_model):
        """Different noise_seeds produce different diffusion output."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input(seed=42)
        z1 = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=5,
            noise_seed=100,
        )
        z2 = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=5,
            noise_seed=200,
        )
        assert not torch.equal(z1, z2)

    def test_more_steps_changes_output(self, cpu_model):
        """Different step counts produce different output."""
        enc = EncoderWrapper(cpu_model.encoder)
        dec = DecoderWrapper(cpu_model.decoder)
        enc.eval()
        dec.eval()
        ei, ti = make_synthetic_input(seed=42)
        z5 = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=5,
            noise_seed=42,
        )
        z10 = run_diffusion(
            lambda tv, tidx: enc(tv, tidx),
            lambda zz, e, t, tidx: dec(zz, e, t, tidx),
            ei,
            ti,
            sample_steps=10,
            noise_seed=42,
        )
        assert not torch.equal(z5, z10)
