# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SongPrep-7B on AWS Neuron (Trainium2).

Two-stage pipeline for song structure parsing and lyrics transcription:
  Stage 1: MuCodec audio encoder (329.5M params, FP32)
           CPU MelSTFT preprocessing + Neuron Conformer+RVQ
  Stage 2: Qwen2 7B decoder (BF16) via NxD Inference
           Generates structured lyrics with timestamps

Architecture:
  Audio -> MuCodec(MelSTFT -> Conformer -> RVQ) -> codec tokens
  -> token offset + framing -> Qwen2 -> [structure][start:end]lyrics

Reference: https://github.com/tencent-ailab/SongPrep
Weights: https://huggingface.co/tencent/SongPrep-7B
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Token constants from SongPrep tokenizer
SEP_TOKEN_ID = 151655  # <|extra_1|>
PAD_TOKEN_ID = 151654  # <|extra_0|>
EOS_TOKEN_ID = 151643  # <|endoftext|>
TEXT_OFFSET = 151656  # codec tokens shifted by this

SAMPLE_RATE = 48000
CHUNK_SAMPLES_48K = 1_920_000  # 40s at 48kHz
CHUNK_SAMPLES_24K = 960_000  # 40s at 24kHz
TOKENS_PER_SECOND = 25


@dataclass
class SongPrepNeuronConfig:
    """Configuration for SongPrep on Neuron."""

    # Paths
    model_path: str = ""  # HuggingFace model directory (SongPrep-7B)
    mucodec_neff_path: str = ""  # Pre-traced MuCodec NEFF path (optional)
    qwen2_compiled_path: str = ""  # Pre-compiled Qwen2 NEFFs path (optional)

    # Qwen2 NxDI config
    tp_degree: int = 2
    batch_size: int = 1
    seq_len: int = 4096
    max_context_length: int = 2048
    max_new_tokens: int = 2048
    max_length: int = 4096

    # MuCodec tracing config
    mucodec_compiler_args: list = field(
        default_factory=lambda: ["--auto-cast", "matmult"]
    )

    # Generation config
    do_sample: bool = True
    top_p: float = 0.1
    temperature: float = 0.1


# ============================================================
# Stage 1: MuCodec Audio Encoder
# ============================================================


class MuCodecConformerRVQ(nn.Module):
    """
    Neuron-traceable module: Conformer encoder + RVQ quantizer.

    Extracts hidden states from layer 6 of the Conformer, then quantizes
    through the RVQ codebook to produce discrete codec tokens.
    """

    def __init__(self, musicfm, rvq, layer=6):
        super().__init__()
        self.conv = musicfm.model.conv
        self.conformer = musicfm.model.conformer
        self.rvq = rvq
        self.layer = layer

    def forward(self, mel_features):
        x = self.conv(mel_features)
        out = self.conformer(x, output_hidden_states=True)
        hidden_states = out["hidden_states"]
        bestrq_emb = hidden_states[self.layer]
        bestrq_emb = bestrq_emb.permute(0, 2, 1).contiguous()
        bestrq_emb = bestrq_emb.float()
        quantized, codes, latents, commitment_loss, codebook_loss, n_q = self.rvq(
            bestrq_emb
        )
        return codes


def _remove_weight_norm(model):
    """Remove weight_norm from all modules (required before tracing)."""
    for name, module in model.named_modules():
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:
                pass
        elif hasattr(module, "parametrizations") and hasattr(
            module.parametrizations, "weight"
        ):
            try:
                nn.utils.parametrize.remove_parametrizations(module, "weight")
            except Exception:
                pass
    return model


def trace_mucodec_encoder(
    model_path: str,
    output_path: str,
    compiler_args: Optional[list] = None,
):
    """
    Trace the MuCodec Conformer+RVQ encoder to a Neuron NEFF.

    The MelSTFT preprocessing stage runs on CPU (uses torch.stft which is
    not traceable on Neuron due to overlapping window strides). Only the
    Conformer backbone and RVQ quantizer are traced to Neuron.

    Args:
        model_path: Path to SongPrep-7B model directory containing mucodec.safetensors
        output_path: Path to save the traced NEFF (.pt file)
        compiler_args: Neuron compiler args (default: ['--auto-cast', 'matmult'])

    Returns:
        Path to the saved NEFF file
    """
    import torch_neuronx

    if compiler_args is None:
        compiler_args = ["--auto-cast", "matmult"]

    # Import SongPrep's MuCodec
    sys.path.insert(0, os.path.dirname(model_path))
    from mucodec.generate_1rvq import Tango

    # Load model
    mucodec_safetensors = os.path.join(model_path, "mucodec.safetensors")
    tango = Tango(model_path=mucodec_safetensors, device="cpu")
    model = tango.model
    model.eval()
    _remove_weight_norm(model)

    # Build traceable module
    traceable = MuCodecConformerRVQ(model.bestrq, model.quantizer)
    traceable.eval()

    # Generate dummy mel input for 40s chunk
    # MelSTFT output shape: [1, 128, T] where T depends on audio length
    # For 40s at 24kHz -> 960,000 samples -> MelSTFT -> [1, 128, 4000]
    dummy_audio = torch.randn(1, CHUNK_SAMPLES_24K)
    musicfm_model = model.bestrq.model
    with torch.no_grad():
        x = musicfm_model.preprocessing(dummy_audio, features=["melspec_2048"])
        x = musicfm_model.normalize(x)
        dummy_mel = x["melspec_2048"]

    print(f"Tracing MuCodec Conformer+RVQ (mel input shape: {dummy_mel.shape})...")
    traced = torch_neuronx.trace(
        traceable,
        dummy_mel,
        compiler_args=compiler_args,
    )

    torch.jit.save(traced, output_path)
    print(f"Saved MuCodec NEFF to: {output_path}")
    return output_path


def _load_mucodec(model_path: str, neff_path: str):
    """
    Load MuCodec model components.

    Returns:
        mucodec_model: Full MuCodec model (for CPU MelSTFT preprocessing)
        neuron_encoder: Traced Conformer+RVQ NEFF on Neuron
    """
    import torch_neuronx  # Must import before torch.jit.load

    sys.path.insert(0, os.path.dirname(model_path))
    from mucodec.generate_1rvq import Tango

    mucodec_safetensors = os.path.join(model_path, "mucodec.safetensors")
    tango = Tango(model_path=mucodec_safetensors, device="cpu")
    model = tango.model
    model.eval()
    _remove_weight_norm(model)

    neuron_encoder = torch.jit.load(neff_path)
    return model, neuron_encoder


def _cpu_preprocess(musicfm, audio_24k):
    """Run MelSTFT preprocessing on CPU."""
    model = musicfm.model
    x = model.preprocessing(audio_24k, features=["melspec_2048"])
    x = model.normalize(x)
    return x["melspec_2048"]


def encode_audio(mucodec_model, neuron_encoder, audio_48k):
    """
    Encode audio waveform to codec tokens.

    Pipeline: resample 48k->24k -> MelSTFT (CPU) -> Conformer+RVQ (Neuron)

    Args:
        mucodec_model: Full MuCodec model (for CPU preprocessing)
        neuron_encoder: Traced Conformer+RVQ on Neuron
        audio_48k: Tensor of shape [channels, samples] at 48kHz

    Returns:
        Tensor of codec token IDs (0-indexed, before text_offset)
    """
    # Stereo handling and volume normalization
    if audio_48k.shape[0] > 1:
        ch0 = audio_48k[0:1]
        ch1 = audio_48k[1:2]
    else:
        ch0 = audio_48k
        ch1 = audio_48k

    threshold = 0.9
    for ch in [ch0, ch1]:
        max_vol = ch.abs().max()
        if max_vol > threshold:
            ch.div_(max_vol / threshold)

    # Resample 48k -> 24k
    rsq = mucodec_model.rsq48tobestrq
    ch0_24k = rsq(ch0)
    ch1_24k = rsq(ch1)
    mono_24k = (ch0_24k + ch1_24k) / 2.0

    # Pad to 40s chunk boundary
    total_samples = mono_24k.shape[1]
    n_chunks = (total_samples + CHUNK_SAMPLES_24K - 1) // CHUNK_SAMPLES_24K

    if total_samples < n_chunks * CHUNK_SAMPLES_24K:
        pad_len = n_chunks * CHUNK_SAMPLES_24K - total_samples
        mono_24k = torch.nn.functional.pad(mono_24k, (0, pad_len))

    all_codes = []
    for i in range(n_chunks):
        chunk = mono_24k[:, i * CHUNK_SAMPLES_24K : (i + 1) * CHUNK_SAMPLES_24K]

        # CPU: MelSTFT
        with torch.no_grad():
            mel = _cpu_preprocess(mucodec_model.bestrq, chunk)

        # Neuron: Conformer + RVQ
        with torch.no_grad():
            codes = neuron_encoder(mel)  # [1, 1, T_tokens]

        all_codes.append(codes[0, 0])  # [T_tokens]

    all_codes = torch.cat(all_codes, dim=0)

    # Trim to actual audio length
    audio_duration = audio_48k.shape[1] / SAMPLE_RATE
    expected_tokens = int(audio_duration * TOKENS_PER_SECOND)
    if len(all_codes) > expected_tokens:
        all_codes = all_codes[:expected_tokens]

    return all_codes


# ============================================================
# Stage 2: Qwen2 Decoder via NxD Inference
# ============================================================


def _load_qwen2(model_path: str, compiled_path: str, config: SongPrepNeuronConfig):
    """
    Load compiled Qwen2 model on Neuron via NxD Inference.

    Args:
        model_path: HuggingFace model directory
        compiled_path: Path to pre-compiled Qwen2 NEFFs
        config: SongPrepNeuronConfig

    Returns:
        Loaded NeuronQwen2ForCausalLM model
    """
    from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (
        NeuronQwen2ForCausalLM,
        Qwen2InferenceConfig,
        Qwen2NeuronConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    neuron_config = Qwen2NeuronConfig(
        tp_degree=config.tp_degree,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        max_context_length=config.max_context_length,
        max_new_tokens=config.max_new_tokens,
        max_length=config.max_length,
        n_positions=config.seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,  # CPU sampling (vocab too large for NKI kernel)
        padding_side="right",
        fused_qkv=False,
        output_logits=False,
    )

    inf_config = Qwen2InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    model = NeuronQwen2ForCausalLM(model_path, inf_config)
    model.load(compiled_path)

    return model


def compile_qwen2(model_path: str, output_path: str, config: SongPrepNeuronConfig):
    """
    Compile the Qwen2 decoder for Neuron.

    Args:
        model_path: HuggingFace model directory
        output_path: Directory to save compiled NEFFs
        config: SongPrepNeuronConfig
    """
    from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (
        NeuronQwen2ForCausalLM,
        Qwen2InferenceConfig,
        Qwen2NeuronConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    neuron_config = Qwen2NeuronConfig(
        tp_degree=config.tp_degree,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        max_context_length=config.max_context_length,
        max_new_tokens=config.max_new_tokens,
        max_length=config.max_length,
        n_positions=config.seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        padding_side="right",
        fused_qkv=False,
        output_logits=False,
    )

    inf_config = Qwen2InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    print("Compiling Qwen2 decoder for Neuron...")
    model = NeuronQwen2ForCausalLM(model_path, inf_config)
    model.compile(output_path)
    print(f"Saved compiled Qwen2 to: {output_path}")


def build_prompt_ids(codec_codes):
    """
    Build prompt token IDs from codec codes.

    Format: [sep] + (codec_codes + text_offset) + [sep]
    """
    offset_codes = codec_codes.numpy().astype(np.int32) + TEXT_OFFSET
    return [SEP_TOKEN_ID] + offset_codes.tolist() + [SEP_TOKEN_ID]


def generate_lyrics(qwen2_model, prompt_ids, config: SongPrepNeuronConfig):
    """
    Generate structured lyrics from prompt token IDs.

    Args:
        qwen2_model: Loaded NeuronQwen2ForCausalLM
        prompt_ids: List of token IDs (from build_prompt_ids)
        config: SongPrepNeuronConfig

    Returns:
        output_ids: Full output tensor including prompt
        elapsed: Generation time in seconds
    """
    from transformers import AutoTokenizer, GenerationConfig
    from neuronx_distributed_inference.utils.accuracy import (
        get_generate_outputs_from_token_ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generation_config = GenerationConfig(
        do_sample=config.do_sample,
        top_p=config.top_p,
        temperature=config.temperature,
        max_length=config.max_length,
        pad_token_id=EOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
    )

    input_ids = [prompt_ids]

    start = time.time()
    outputs, output_tokens = get_generate_outputs_from_token_ids(
        qwen2_model,
        input_ids,
        tokenizer,
        is_hf=False,
        generation_config=generation_config,
        max_length=config.max_length,
    )
    elapsed = time.time() - start

    if isinstance(outputs, torch.Tensor):
        output_ids = outputs
    else:
        output_ids = outputs.sequences

    return output_ids, elapsed


# ============================================================
# Full Pipeline
# ============================================================


class SongPrepPipeline:
    """
    End-to-end SongPrep pipeline on Neuron.

    Usage:
        config = SongPrepNeuronConfig(
            model_path="/path/to/SongPrep-7B",
            mucodec_neff_path="/path/to/mucodec_neuron.pt",
            qwen2_compiled_path="/path/to/qwen2-compiled/",
        )
        pipeline = SongPrepPipeline(config)
        result = pipeline.run("/path/to/audio.wav")
        print(result["lyrics"])
    """

    def __init__(self, config: SongPrepNeuronConfig):
        self.config = config
        self.mucodec_model = None
        self.neuron_encoder = None
        self.qwen2_model = None

    def load(self):
        """Load both MuCodec and Qwen2 models."""
        self.mucodec_model, self.neuron_encoder = _load_mucodec(
            self.config.model_path, self.config.mucodec_neff_path
        )
        self.qwen2_model = _load_qwen2(
            self.config.model_path,
            self.config.qwen2_compiled_path,
            self.config,
        )

    def load_mucodec_only(self):
        """Load only the MuCodec encoder."""
        self.mucodec_model, self.neuron_encoder = _load_mucodec(
            self.config.model_path, self.config.mucodec_neff_path
        )

    def load_qwen2_only(self):
        """Load only the Qwen2 decoder."""
        self.qwen2_model = _load_qwen2(
            self.config.model_path,
            self.config.qwen2_compiled_path,
            self.config,
        )

    def encode(self, audio_48k):
        """
        Encode audio to codec tokens.

        Args:
            audio_48k: Tensor [channels, samples] at 48kHz

        Returns:
            Tensor of codec token IDs (0-indexed)
        """
        assert self.mucodec_model is not None, (
            "Call load() or load_mucodec_only() first"
        )
        return encode_audio(self.mucodec_model, self.neuron_encoder, audio_48k)

    def decode(self, codec_codes):
        """
        Generate lyrics from codec tokens.

        Args:
            codec_codes: Tensor of codec token IDs (0-indexed)

        Returns:
            output_ids: Full output tensor
            elapsed: Generation time in seconds
        """
        assert self.qwen2_model is not None, "Call load() or load_qwen2_only() first"
        prompt_ids = build_prompt_ids(codec_codes)
        return generate_lyrics(self.qwen2_model, prompt_ids, self.config)

    def run(self, audio_path: str):
        """
        Run full pipeline: audio file -> structured lyrics.

        Args:
            audio_path: Path to WAV file

        Returns:
            dict with keys: lyrics, codec_tokens, n_generated, mucodec_time_s,
                           qwen2_time_s, total_time_s, tok_per_sec
        """
        import soundfile as sf

        assert self.mucodec_model is not None and self.qwen2_model is not None, (
            "Call load() first"
        )

        total_start = time.time()

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")
        audio = torch.tensor(audio).T
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if sr != SAMPLE_RATE:
            import torchaudio

            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

        audio_duration = audio.shape[1] / SAMPLE_RATE

        # Stage 1: MuCodec
        t0 = time.time()
        codec_codes = self.encode(audio)
        mucodec_time = time.time() - t0

        # Stage 2: Qwen2
        prompt_ids = build_prompt_ids(codec_codes)
        output_ids, gen_time = generate_lyrics(
            self.qwen2_model, prompt_ids, self.config
        )

        n_generated = output_ids.shape[1] - len(prompt_ids)
        tok_per_sec = n_generated / gen_time if gen_time > 0 else 0

        # Parse output
        lyrics = self._parse_output(output_ids, len(prompt_ids))

        total_time = time.time() - total_start

        return {
            "lyrics": lyrics,
            "audio_duration_s": audio_duration,
            "codec_tokens": len(codec_codes),
            "n_generated": n_generated,
            "mucodec_time_s": mucodec_time,
            "qwen2_time_s": gen_time,
            "total_time_s": total_time,
            "tok_per_sec": tok_per_sec,
        }

    def _parse_output(self, output_ids, prompt_len):
        """Parse generated output to extract structured lyrics text."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, use_fast=False, trust_remote_code=True
        )

        ids = output_ids[0].cpu().numpy()
        sep_positions = np.where(ids == SEP_TOKEN_ID)[0]

        if len(sep_positions) >= 2:
            start = sep_positions[1] + 1
            if len(sep_positions) >= 3:
                end = sep_positions[2]
            else:
                end = len(ids)
                while end > start and ids[end - 1] in (EOS_TOKEN_ID, PAD_TOKEN_ID, 0):
                    end -= 1
            generated_ids = ids[start:end]
        else:
            generated_ids = ids[prompt_len:]
            end_idx = len(generated_ids)
            while end_idx > 0 and generated_ids[end_idx - 1] in (
                EOS_TOKEN_ID,
                PAD_TOKEN_ID,
                0,
            ):
                end_idx -= 1
            generated_ids = generated_ids[:end_idx]

        return tokenizer.decode(generated_ids)
