"""
Integration tests for ZUNA EEG foundation model on Neuron.

Tests compile and run the ZUNA masked diffusion autoencoder (Zyphra/ZUNA, 382M
params) using torch_neuronx.trace() on Inferentia2. Validates accuracy by
comparing Neuron output against CPU reference using cosine similarity across
the full 50-step rectified flow diffusion pipeline.

Usage:
    # Run with pytest
    pytest test_model.py --capture=tee-sys -v

    # Run standalone
    python test_model.py

Prerequisites:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    pip install zuna
"""

import json
import os
import subprocess
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

MODEL_ID = "Zyphra/ZUNA"
COMPILED_DIR = "/home/ubuntu/neuron_models/ZUNA"

SEQLEN = 50
INPUT_DIM = 32
SAMPLE_STEPS = 50
BATCH_SIZE = 1

COMPILER_ARGS = ["--auto-cast", "matmult", "-O2"]

# Accuracy thresholds (with --auto-cast matmult)
COSINE_SIM_THRESHOLD = 0.930  # measured: 0.990 mean, 0.937 min over 50 seeds
COSINE_SIM_THRESHOLD_NOAUTOCAST = 0.999  # measured: 1.000000 with auto-cast=none

# Performance thresholds
PIPELINE_LATENCY_THRESHOLD_MS = 200.0  # 50-step pipeline, measured: ~102ms
THROUGHPUT_THRESHOLD = 5.0  # samples/sec minimum, measured: ~9.8


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
# Helpers (self-contained -- does not import from src/ to keep test standalone)
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


def load_cpu_model():
    """Load ZUNA model on CPU with Neuron patches applied."""
    import json as json_mod

    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
    with open(config_path) as f:
        config = json_mod.load(f)

    model_config = config.get("model", config)
    model_args = dataclass_from_dict(DecoderTransformerArgs, model_config)
    model = EncoderDecoder(model_args)

    weights_path = hf_hub_download(
        repo_id=MODEL_ID, filename="model-00001-of-00001.safetensors"
    )
    state_dict = load_safetensors(weights_path)
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Apply attention patches
    _patch_model(model)

    return model, model_args


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
    """Run rectified flow diffusion loop. Returns (z, timing_dict)."""
    if noise_seed is not None:
        torch.manual_seed(noise_seed)

    with torch.no_grad():
        t0 = time.perf_counter()
        enc_out = encoder_fn(encoder_input, tok_idx)
        enc_ms = (time.perf_counter() - t0) * 1000

        z = sigma * torch.randn_like(encoder_input)
        dt_val = 1.0 / sample_steps
        step_times = []

        for step_num in range(sample_steps, 0, -1):
            t_val = dt_val * step_num
            t_tensor = torch.tensor([[[t_val]]])
            t_step = time.perf_counter()
            vc = decoder_fn(z, enc_out, t_tensor, tok_idx)
            step_times.append((time.perf_counter() - t_step) * 1000)
            z = z - dt_val * vc

        total_ms = enc_ms + sum(step_times)

    timing = {
        "encoder_ms": enc_ms,
        "total_ms": total_ms,
        "avg_step_ms": sum(step_times) / len(step_times),
    }
    return z, timing


def compile_neuron_models(cpu_model, model_args, compiler_args=COMPILER_ARGS):
    """Compile encoder and decoder for Neuron, caching results."""
    import torch_neuronx

    enc_path = os.path.join(COMPILED_DIR, f"encoder_seqlen{SEQLEN}.pt")
    dec_path = os.path.join(COMPILED_DIR, f"decoder_seqlen{SEQLEN}.pt")

    if os.path.exists(enc_path) and os.path.exists(dec_path):
        return torch.jit.load(enc_path), torch.jit.load(dec_path)

    os.makedirs(COMPILED_DIR, exist_ok=True)
    input_dim = model_args.input_dim

    example_input, example_tok_idx = make_synthetic_input(
        seqlen=SEQLEN,
        input_dim=input_dim,
    )

    encoder_wrapper = EncoderWrapper(cpu_model.encoder)
    encoder_wrapper.eval()
    encoder_neuron = torch_neuronx.trace(
        encoder_wrapper,
        (example_input, example_tok_idx),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )

    with torch.no_grad():
        enc_out_example = encoder_neuron(example_input, example_tok_idx)

    decoder_wrapper = DecoderWrapper(cpu_model.decoder)
    decoder_wrapper.eval()
    example_z = torch.randn(1, SEQLEN, input_dim)
    example_enc = torch.randn(1, enc_out_example.shape[1], enc_out_example.shape[2])
    example_t = torch.tensor([[[0.5]]])

    decoder_neuron = torch_neuronx.trace(
        decoder_wrapper,
        (example_z, example_enc, example_t, example_tok_idx),
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )

    encoder_neuron.save(enc_path)
    decoder_neuron.save(dec_path)

    return encoder_neuron, decoder_neuron


def get_neuron_core_count():
    """Detect available NeuronCores via neuron-ls."""
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            devices = json.loads(result.stdout)
            return sum(d["nc_count"] for d in devices)
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------
# Fixtures (module-scoped so compile happens once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_model_and_args():
    """Load the CPU reference model."""
    return load_cpu_model()


@pytest.fixture(scope="module")
def cpu_model(cpu_model_and_args):
    return cpu_model_and_args[0]


@pytest.fixture(scope="module")
def model_args(cpu_model_and_args):
    return cpu_model_and_args[1]


@pytest.fixture(scope="module")
def neuron_models(cpu_model, model_args):
    """Compile and load the Neuron encoder+decoder."""
    return compile_neuron_models(cpu_model, model_args)


@pytest.fixture(scope="module")
def encoder_neuron(neuron_models):
    return neuron_models[0]


@pytest.fixture(scope="module")
def decoder_neuron(neuron_models):
    return neuron_models[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelLoads:
    """Smoke tests that the model compiles and loads on Neuron."""

    def test_encoder_loads(self, encoder_neuron):
        """Encoder compiles and loads successfully."""
        assert encoder_neuron is not None

    def test_decoder_loads(self, decoder_neuron):
        """Decoder compiles and loads successfully."""
        assert decoder_neuron is not None

    def test_encoder_runs(self, encoder_neuron):
        """Encoder produces output of expected shape."""
        encoder_input, tok_idx = make_synthetic_input()
        out = encoder_neuron(encoder_input, tok_idx)
        assert out.shape[0] == 1
        # Encoder output has bottleneck dim, seqlen depends on register interleaving
        assert out.ndim == 3

    def test_decoder_runs(self, encoder_neuron, decoder_neuron):
        """Decoder produces output of expected shape."""
        encoder_input, tok_idx = make_synthetic_input()
        enc_out = encoder_neuron(encoder_input, tok_idx)
        z = torch.randn(1, SEQLEN, INPUT_DIM)
        t = torch.tensor([[[0.5]]])
        out = decoder_neuron(z, enc_out, t, tok_idx)
        assert out.shape == (1, SEQLEN, INPUT_DIM)

    def test_full_pipeline(self, encoder_neuron, decoder_neuron):
        """Full 50-step diffusion pipeline runs to completion."""
        encoder_input, tok_idx = make_synthetic_input(seed=99)
        z, timing = run_diffusion(
            lambda tv, ti: encoder_neuron(tv, ti),
            lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            noise_seed=99,
        )
        assert z.shape == (1, SEQLEN, INPUT_DIM)
        assert timing["total_ms"] > 0


class TestAccuracy:
    """Validate Neuron output against CPU reference."""

    @pytest.mark.parametrize("seed", [0, 7, 13, 42, 99])
    def test_cosine_similarity(self, cpu_model, encoder_neuron, decoder_neuron, seed):
        """Cosine similarity of full diffusion output exceeds threshold."""
        encoder_input, tok_idx = make_synthetic_input(seed=seed)
        sigma = float(getattr(cpu_model, "global_sigma", 0.1))

        # CPU reference
        enc_cpu = EncoderWrapper(cpu_model.encoder)
        enc_cpu.eval()
        dec_cpu = DecoderWrapper(cpu_model.decoder)
        dec_cpu.eval()

        z_cpu, _ = run_diffusion(
            lambda tv, ti: enc_cpu(tv, ti),
            lambda z, e, t, ti: dec_cpu(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=seed + 10000,
        )

        # Neuron
        z_neuron, _ = run_diffusion(
            lambda tv, ti: encoder_neuron(tv, ti),
            lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=seed + 10000,
        )

        cosine = F.cosine_similarity(
            z_cpu.flatten().float(), z_neuron.flatten().float(), dim=0
        ).item()

        print(f"  seed={seed}: cosine_sim={cosine:.6f}")
        assert cosine >= COSINE_SIM_THRESHOLD, (
            f"Cosine similarity {cosine:.6f} below threshold {COSINE_SIM_THRESHOLD}"
        )

    def test_mse_bounded(self, cpu_model, encoder_neuron, decoder_neuron):
        """MSE between CPU and Neuron is within expected bounds."""
        encoder_input, tok_idx = make_synthetic_input(seed=42)
        sigma = float(getattr(cpu_model, "global_sigma", 0.1))

        enc_cpu = EncoderWrapper(cpu_model.encoder)
        enc_cpu.eval()
        dec_cpu = DecoderWrapper(cpu_model.decoder)
        dec_cpu.eval()

        z_cpu, _ = run_diffusion(
            lambda tv, ti: enc_cpu(tv, ti),
            lambda z, e, t, ti: dec_cpu(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=10042,
        )
        z_neuron, _ = run_diffusion(
            lambda tv, ti: encoder_neuron(tv, ti),
            lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=10042,
        )

        mse = F.mse_loss(z_cpu.float(), z_neuron.float()).item()
        print(f"  MSE: {mse:.8f}")
        # Measured: ~3.3e-05 with auto-cast matmult
        assert mse < 0.01, f"MSE {mse:.8f} exceeds threshold 0.01"


class TestDataParallel:
    """Test DataParallel for full-instance throughput."""

    def test_data_parallel_runs(self, cpu_model, model_args):
        """DataParallel encoder+decoder runs correctly on 2 NeuronCores."""
        import torch_neuronx

        num_cores = get_neuron_core_count()
        if num_cores < 2:
            pytest.skip("Only 1 NeuronCore available")

        input_dim = model_args.input_dim
        encoder_wrapper = EncoderWrapper(cpu_model.encoder)
        encoder_wrapper.eval()
        decoder_wrapper = DecoderWrapper(cpu_model.decoder)
        decoder_wrapper.eval()

        example_input, example_tok_idx = make_synthetic_input()

        encoder_neuron = torch_neuronx.trace(
            encoder_wrapper,
            (example_input, example_tok_idx),
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        encoder_dp = torch_neuronx.DataParallel(
            encoder_neuron,
            device_ids=[0, 1],
            dim=0,
        )

        # Run encoder DP
        ei_batch = example_input.repeat(2, 1, 1)
        ti_batch = example_tok_idx.repeat(2, 1, 1)
        with torch.no_grad():
            enc_out = encoder_dp(ei_batch, ti_batch)

        assert enc_out.shape[0] == 2

        # Compile and run decoder DP
        example_z = torch.randn(1, SEQLEN, input_dim)
        example_enc = torch.randn(1, enc_out.shape[1] // 1, enc_out.shape[2])
        example_t = torch.tensor([[[0.5]]])

        decoder_neuron = torch_neuronx.trace(
            decoder_wrapper,
            (example_z, example_enc, example_t, example_tok_idx),
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        decoder_dp = torch_neuronx.DataParallel(
            decoder_neuron,
            device_ids=[0, 1],
            dim=0,
        )

        z_batch = torch.randn(2, SEQLEN, input_dim)
        t_batch = torch.tensor([[[0.5]]]).repeat(2, 1, 1)
        ti_batch = example_tok_idx.repeat(2, 1, 1)

        with torch.no_grad():
            out = decoder_dp(z_batch, enc_out, t_batch, ti_batch)

        assert out.shape == (2, SEQLEN, input_dim)
        print(f"  DataParallel OK: {num_cores} cores, output shape {out.shape}")

    def test_data_parallel_speedup(
        self, cpu_model, model_args, encoder_neuron, decoder_neuron
    ):
        """DataParallel achieves meaningful speedup over single core."""
        import torch_neuronx
        import numpy as np

        num_cores = get_neuron_core_count()
        if num_cores < 2:
            pytest.skip("Only 1 NeuronCore available")

        sigma = float(getattr(cpu_model, "global_sigma", 0.1))

        # Single-core baseline
        single_times = []
        for trial in range(2 + 5):
            encoder_input, tok_idx = make_synthetic_input(seed=trial)
            _, timing = run_diffusion(
                lambda tv, ti: encoder_neuron(tv, ti),
                lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
                encoder_input,
                tok_idx,
                sample_steps=SAMPLE_STEPS,
                sigma=sigma,
                noise_seed=trial + 10000,
            )
            if trial >= 2:
                single_times.append(timing["total_ms"])
        single_throughput = 1000.0 / np.mean(single_times)

        # DataParallel
        input_dim = model_args.input_dim
        enc_wrapper = EncoderWrapper(cpu_model.encoder)
        enc_wrapper.eval()
        dec_wrapper = DecoderWrapper(cpu_model.decoder)
        dec_wrapper.eval()

        example_input, example_tok_idx = make_synthetic_input()
        enc_traced = torch_neuronx.trace(
            enc_wrapper,
            (example_input, example_tok_idx),
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        encoder_dp = torch_neuronx.DataParallel(
            enc_traced,
            device_ids=[0, 1],
            dim=0,
        )

        with torch.no_grad():
            enc_out_ex = encoder_dp(
                example_input.repeat(2, 1, 1), example_tok_idx.repeat(2, 1, 1)
            )

        example_z = torch.randn(1, SEQLEN, input_dim)
        example_enc = torch.randn(1, enc_out_ex.shape[1] // 1, enc_out_ex.shape[2])
        example_t = torch.tensor([[[0.5]]])
        dec_traced = torch_neuronx.trace(
            dec_wrapper,
            (example_z, example_enc, example_t, example_tok_idx),
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        decoder_dp = torch_neuronx.DataParallel(
            dec_traced,
            device_ids=[0, 1],
            dim=0,
        )

        dp_times = []
        for trial in range(2 + 5):
            encoder_input, tok_idx = make_synthetic_input(seed=trial)
            ei_batch = encoder_input.repeat(2, 1, 1)
            ti_batch = tok_idx.repeat(2, 1, 1)

            with torch.no_grad():
                t0 = time.perf_counter()
                enc_out = encoder_dp(ei_batch, ti_batch)
                z = sigma * torch.randn(2, SEQLEN, input_dim)
                dt_val = 1.0 / SAMPLE_STEPS
                for step_num in range(SAMPLE_STEPS, 0, -1):
                    t_val = dt_val * step_num
                    t_tensor = torch.tensor([[[t_val]]]).repeat(2, 1, 1)
                    vc = decoder_dp(z, enc_out, t_tensor, ti_batch)
                    z = z - dt_val * vc
                total_ms = (time.perf_counter() - t0) * 1000

            if trial >= 2:
                dp_times.append(total_ms)

        dp_throughput = 2000.0 / np.mean(dp_times)
        speedup = dp_throughput / single_throughput

        print(
            f"  Single: {single_throughput:.2f} S/s, DP: {dp_throughput:.2f} S/s, "
            f"Speedup: {speedup:.2f}x"
        )
        assert speedup > 1.3, (
            f"DataParallel speedup {speedup:.2f}x too low (expected >1.3x)"
        )


class TestPerformance:
    """Benchmark throughput and latency."""

    def test_pipeline_throughput(self, cpu_model, encoder_neuron, decoder_neuron):
        """Single-core pipeline throughput exceeds minimum threshold."""
        import numpy as np

        sigma = float(getattr(cpu_model, "global_sigma", 0.1))

        # Warmup
        for i in range(3):
            encoder_input, tok_idx = make_synthetic_input(seed=i)
            run_diffusion(
                lambda tv, ti: encoder_neuron(tv, ti),
                lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
                encoder_input,
                tok_idx,
                sample_steps=SAMPLE_STEPS,
                sigma=sigma,
                noise_seed=i,
            )

        # Benchmark
        latencies = []
        for i in range(10):
            encoder_input, tok_idx = make_synthetic_input(seed=i + 100)
            _, timing = run_diffusion(
                lambda tv, ti: encoder_neuron(tv, ti),
                lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
                encoder_input,
                tok_idx,
                sample_steps=SAMPLE_STEPS,
                sigma=sigma,
                noise_seed=i + 10100,
            )
            latencies.append(timing["total_ms"])

        avg_ms = np.mean(latencies)
        throughput = 1000.0 / avg_ms

        print(f"  50-step pipeline: {avg_ms:.1f} ms, {throughput:.2f} samples/sec")
        assert throughput >= THROUGHPUT_THRESHOLD, (
            f"Throughput {throughput:.2f} below threshold {THROUGHPUT_THRESHOLD}"
        )

    def test_pipeline_latency(self, cpu_model, encoder_neuron, decoder_neuron):
        """50-step pipeline latency is below threshold."""
        import numpy as np

        sigma = float(getattr(cpu_model, "global_sigma", 0.1))

        # Warmup
        for i in range(3):
            encoder_input, tok_idx = make_synthetic_input(seed=i)
            run_diffusion(
                lambda tv, ti: encoder_neuron(tv, ti),
                lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
                encoder_input,
                tok_idx,
                sample_steps=SAMPLE_STEPS,
                sigma=sigma,
                noise_seed=i,
            )

        latencies = []
        for i in range(10):
            encoder_input, tok_idx = make_synthetic_input(seed=i + 200)
            _, timing = run_diffusion(
                lambda tv, ti: encoder_neuron(tv, ti),
                lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
                encoder_input,
                tok_idx,
                sample_steps=SAMPLE_STEPS,
                sigma=sigma,
                noise_seed=i + 10200,
            )
            latencies.append(timing["total_ms"])

        p50 = np.percentile(latencies, 50)
        print(f"  p50 pipeline latency: {p50:.1f} ms")
        assert p50 <= PIPELINE_LATENCY_THRESHOLD_MS, (
            f"p50 latency {p50:.1f} ms exceeds threshold "
            f"{PIPELINE_LATENCY_THRESHOLD_MS} ms"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("ZUNA Integration Tests")
    print("=" * 60)

    print("\n[1/7] Loading CPU model...")
    cpu_model, model_args = load_cpu_model()
    param_count = sum(p.numel() for p in cpu_model.parameters())
    print(f"  Model loaded: {param_count / 1e6:.1f}M params")
    sigma = float(getattr(cpu_model, "global_sigma", 0.1))

    print("\n[2/7] Compiling for Neuron...")
    encoder_neuron, decoder_neuron = compile_neuron_models(cpu_model, model_args)
    print("  Compile OK")

    print("\n[3/7] Testing encoder + decoder run...")
    encoder_input, tok_idx = make_synthetic_input()
    enc_out = encoder_neuron(encoder_input, tok_idx)
    print(f"  Encoder output: {enc_out.shape}")
    z_test = torch.randn(1, SEQLEN, INPUT_DIM)
    t_test = torch.tensor([[[0.5]]])
    dec_out = decoder_neuron(z_test, enc_out, t_test, tok_idx)
    print(f"  Decoder output: {dec_out.shape}")

    print("\n[4/7] Accuracy validation (5 seeds)...")
    enc_cpu = EncoderWrapper(cpu_model.encoder)
    enc_cpu.eval()
    dec_cpu = DecoderWrapper(cpu_model.decoder)
    dec_cpu.eval()

    all_pass = True
    for seed in [0, 7, 13, 42, 99]:
        encoder_input, tok_idx = make_synthetic_input(seed=seed)
        z_cpu, _ = run_diffusion(
            lambda tv, ti: enc_cpu(tv, ti),
            lambda z, e, t, ti: dec_cpu(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=seed + 10000,
        )
        z_neuron, _ = run_diffusion(
            lambda tv, ti: encoder_neuron(tv, ti),
            lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
            encoder_input,
            tok_idx,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=seed + 10000,
        )
        cosine = F.cosine_similarity(
            z_cpu.flatten().float(), z_neuron.flatten().float(), dim=0
        ).item()
        status = "PASS" if cosine >= COSINE_SIM_THRESHOLD else "FAIL"
        if cosine < COSINE_SIM_THRESHOLD:
            all_pass = False
        print(f"  seed={seed}: cosine={cosine:.6f} [{status}]")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    print("\n[5/7] Performance benchmark (50 steps)...")
    # Warmup
    for i in range(3):
        ei, ti = make_synthetic_input(seed=i)
        run_diffusion(
            lambda tv, ti: encoder_neuron(tv, ti),
            lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
            ei,
            ti,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=i,
        )
    latencies = []
    for i in range(10):
        ei, ti = make_synthetic_input(seed=i + 100)
        _, timing = run_diffusion(
            lambda tv, ti: encoder_neuron(tv, ti),
            lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
            ei,
            ti,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=i + 10100,
        )
        latencies.append(timing["total_ms"])
    avg_ms = np.mean(latencies)
    print(f"  Throughput: {1000.0 / avg_ms:.2f} samples/sec")
    print(f"  p50: {np.percentile(latencies, 50):.1f} ms")

    print("\n[6/7] DataParallel test...")
    num_cores = get_neuron_core_count()
    if num_cores >= 2:
        import torch_neuronx

        enc_wrapper = EncoderWrapper(cpu_model.encoder)
        enc_wrapper.eval()
        dec_wrapper = DecoderWrapper(cpu_model.decoder)
        dec_wrapper.eval()

        example_input, example_tok_idx = make_synthetic_input()
        enc_traced = torch_neuronx.trace(
            enc_wrapper,
            (example_input, example_tok_idx),
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        encoder_dp = torch_neuronx.DataParallel(
            enc_traced,
            device_ids=[0, 1],
            dim=0,
        )
        with torch.no_grad():
            enc_out_dp = encoder_dp(
                example_input.repeat(2, 1, 1), example_tok_idx.repeat(2, 1, 1)
            )

        example_z = torch.randn(1, SEQLEN, model_args.input_dim)
        example_enc = torch.randn(1, enc_out_dp.shape[1] // 1, enc_out_dp.shape[2])
        example_t = torch.tensor([[[0.5]]])
        dec_traced = torch_neuronx.trace(
            dec_wrapper,
            (example_z, example_enc, example_t, example_tok_idx),
            compiler_args=COMPILER_ARGS,
            inline_weights_to_neff=True,
        )
        decoder_dp = torch_neuronx.DataParallel(
            dec_traced,
            device_ids=[0, 1],
            dim=0,
        )

        dp_times = []
        for trial in range(2 + 5):
            ei, ti = make_synthetic_input(seed=trial)
            ei_b = ei.repeat(2, 1, 1)
            ti_b = ti.repeat(2, 1, 1)
            with torch.no_grad():
                t0 = time.perf_counter()
                eo = encoder_dp(ei_b, ti_b)
                z = sigma * torch.randn(2, SEQLEN, model_args.input_dim)
                dt_val = 1.0 / SAMPLE_STEPS
                for step_num in range(SAMPLE_STEPS, 0, -1):
                    t_val = dt_val * step_num
                    tt = torch.tensor([[[t_val]]]).repeat(2, 1, 1)
                    vc = decoder_dp(z, eo, tt, ti_b)
                    z = z - dt_val * vc
                total_ms = (time.perf_counter() - t0) * 1000
            if trial >= 2:
                dp_times.append(total_ms)

        dp_throughput = 2000.0 / np.mean(dp_times)
        single_throughput = 1000.0 / avg_ms
        speedup = dp_throughput / single_throughput
        print(
            f"  DP ({num_cores} cores): {dp_throughput:.2f} S/s, {speedup:.2f}x speedup"
        )
    else:
        print("  Skipped: only 1 NeuronCore")

    print("\n[7/7] CPU baseline comparison...")
    cpu_latencies = []
    for i in range(3):
        ei, ti = make_synthetic_input(seed=i + 300)
        _, timing = run_diffusion(
            lambda tv, ti: enc_cpu(tv, ti),
            lambda z, e, t, ti: dec_cpu(z, e, t, ti),
            ei,
            ti,
            sample_steps=SAMPLE_STEPS,
            sigma=sigma,
            noise_seed=i + 10300,
        )
        cpu_latencies.append(timing["total_ms"])
    cpu_avg = np.mean(cpu_latencies)
    neuron_avg = avg_ms
    print(
        f"  CPU: {cpu_avg:.0f} ms, Neuron: {neuron_avg:.1f} ms, "
        f"Speedup: {cpu_avg / neuron_avg:.1f}x"
    )

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
