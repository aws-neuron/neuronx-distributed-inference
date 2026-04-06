#!/usr/bin/env python3
"""
End-to-end audio verification for Gemma-4 on Neuron.

Pipeline:
1. Download sample audio (journal1.wav from HF reference example)
2. Extract mel spectrogram features (CPU, numpy)
3. Compile/load audio encoder on Neuron (Conformer layers + projections)
4. Run audio encoder: CPU subsampling + Neuron Conformer
5. Build vision_embeddings / vision_mask for Neuron text decoder
6. Compile E2B with multimodal support (ImageToTextModelWrapper)
7. Load and run inference on Neuron
8. Print transcription result

Usage:
    python verify_audio.py [--model_path PATH] [--compiled_path PATH]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_gemma4 import (
    NeuronGemma4ForConditionalGeneration,
    Gemma4InferenceConfig,
)
from gemma4_audio_encoder import (
    Gemma4AudioConfig,
    NeuronAudioEncoder,
    extract_mel_features,
    compute_audio_num_tokens,
)


AUDIO_URL = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/journal1.wav"
AUDIO_LOCAL = "/tmp/journal1.wav"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Gemma-4 audio on Neuron")
    parser.add_argument("--model_path", type=str,
                        default=os.path.expanduser("~/models/gemma-4-E2B-it"))
    parser.add_argument("--compiled_path", type=str,
                        default=os.path.expanduser("~/neuron_models/gemma-4-E2B-it-multimodal"))
    parser.add_argument("--audio_compiled_path", type=str,
                        default=os.path.expanduser("~/neuron_models/gemma4-audio-encoder"))
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    return parser.parse_args()


def download_audio(url: str, local_path: str) -> str:
    """Download audio file if not already cached."""
    if os.path.exists(local_path):
        print(f"  Using cached audio: {local_path}")
        return local_path
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, local_path)
    print(f"  Saved to: {local_path}")
    return local_path


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def build_audio_input_ids(
    num_audio_tokens: int,
    prompt_text: str,
    model_path: str,
) -> torch.Tensor:
    """
    Build input_ids with audio token placeholders.

    Format (matching HF Gemma4 chat template):
    <bos> <start_of_turn>user\n <boa> <audio>*N <eoa> <prompt_text> <end_of_turn>\n <start_of_turn>model\n

    Since transformers may not support Gemma4 tokenizer, we use the tokenizer.json directly.
    """
    # Load tokenizer
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    except ImportError:
        raise ImportError("Please install: pip install tokenizers")

    # Special token IDs from config.json and tokenizer
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    bos_id = 2
    audio_token_id = cfg["audio_token_id"]  # 258881 = <|audio|>
    boa_id = cfg["boa_token_id"]  # 256000 = <|audio>
    eoa_id = cfg["eoa_token_id"]  # 258883 = <audio|>
    start_turn_id = 105  # <|turn>
    end_turn_id = 106    # <turn|>

    # Encode text parts (only actual text, not special tokens)
    user_text = tokenizer.encode("user\n").ids
    prompt_ids = tokenizer.encode("\n" + prompt_text + "\n").ids
    model_text = tokenizer.encode("model\n").ids

    # Build: <bos><|turn>user\n<|audio><|audio|>*N<audio|>\nPrompt\n<turn|>\n<|turn>model\n
    ids = ([bos_id, start_turn_id] + user_text
           + [boa_id] + [audio_token_id] * num_audio_tokens + [eoa_id]
           + prompt_ids + [end_turn_id, 107, start_turn_id] + model_text)

    return torch.tensor([ids], dtype=torch.int32), audio_token_id


def build_vision_embeddings(
    input_ids: torch.Tensor,
    audio_embeddings: torch.Tensor,
    audio_token_id: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple:
    """
    Build vision_embeddings and vision_mask for the Neuron text decoder.

    vision_embeddings: [B, seq_len, hidden_size] - audio embeddings placed at correct positions
    vision_mask: [B, seq_len, 1] - int32, 1 at audio positions, 0 elsewhere
                                    (used as a blend mask, not position indices)
    """
    B, S = input_ids.shape

    # Find audio token positions
    audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
    num_audio_tokens = len(audio_positions)

    # Flatten audio embeddings to match num_audio_tokens
    flat_audio = audio_embeddings.reshape(-1, hidden_size)[:num_audio_tokens]

    # Pack audio embeddings sequentially at start; rest is zeros (for padding positions)
    vision_embeddings = torch.zeros(B, S, hidden_size, dtype=dtype)
    vision_embeddings[0, :num_audio_tokens] = flat_audio.to(dtype)

    # Position indices for scatter — pad with S-1 (last position) so that
    # zero-embedding padded entries overwrite the last (padding) position
    # instead of position 0 (BOS).  NxDI's own pad_inputs uses the same
    # convention (fill_value = padded_seq_len - 1).
    vision_mask = torch.full((B, S, 1), S - 1, dtype=torch.int32)
    vision_mask[0, :num_audio_tokens, 0] = audio_positions.to(torch.int32)

    return vision_embeddings, vision_mask


def generate_with_audio(
    model,
    input_ids: torch.Tensor,
    vision_embeddings: torch.Tensor,
    vision_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list,
    audio_token_id: int = 258881,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Generate tokens autoregressively using NxDI stateful KV cache.

    First pass (context encoding): full input_ids + vision_embeddings/vision_mask.
    Subsequent passes (token generation): single new token only.
    NxDI manages KV cache internally — we only feed new tokens after prefill.
    """
    from neuronx_distributed_inference.models.model_wrapper import prepare_sampling_params

    generated_ids = input_ids.clone()
    B = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    sampling_params = prepare_sampling_params(B)

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                # Replace audio tokens with PAD in input_ids before passing to model.
                # HF Gemma-4 does this so embed_tokens and PLE see PAD at audio positions,
                # then encode_vision_to_input replaces the PAD embeddings with audio embeddings.
                model_input_ids = input_ids.clone()
                model_input_ids[model_input_ids == audio_token_id] = pad_token_id

                # Context encoding: full sequence + audio embeddings
                position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(B, -1)
                outputs = model(
                    model_input_ids,
                    position_ids=position_ids,
                    sampling_params=sampling_params,
                    vision_embeddings=vision_embeddings,
                    vision_mask=vision_mask,
                )
            else:
                # Token generation: only the new token
                cur_pos = seq_len + step - 1
                position_ids = torch.tensor([[cur_pos]], dtype=torch.int32)
                # Must pass dummy vision tensors (empty = no audio injection)
                # The compiled model always expects these tensor args
                dummy_ve = torch.zeros(B, 1, vision_embeddings.shape[2],
                                       dtype=vision_embeddings.dtype)
                dummy_vm = torch.zeros(B, 1, 1, dtype=torch.int32)
                outputs = model(
                    next_token,
                    position_ids=position_ids,
                    sampling_params=sampling_params,
                    vision_embeddings=dummy_ve,
                    vision_mask=dummy_vm,
                )

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1).to(torch.int32)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Debug: print each token
        tok_id = next_token.item()
        top5 = torch.topk(next_token_logits[0], 5)
        print(f"  Step {step}: tok={tok_id} top5_ids={top5.indices.tolist()} top5_vals={[f'{v:.1f}' for v in top5.values.tolist()]}")

        # Check EOS
        if next_token.item() in eos_token_ids:
            break

    return generated_ids


def main():
    args = parse_args()

    print("=" * 70)
    print("Gemma-4 E2B Audio Verification on Neuron")
    print("=" * 70)

    # ---- Step 1: Load configs ----
    print("\n[1/7] Loading configurations...")
    with open(os.path.join(args.model_path, "config.json")) as f:
        raw_config = json.load(f)

    audio_cfg = Gemma4AudioConfig.from_dict(raw_config["audio_config"])
    text_hidden_size = raw_config["text_config"]["hidden_size"]
    eos_token_ids = raw_config.get("eos_token_id", [1, 106])
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    print(f"  Audio: {audio_cfg.num_hidden_layers} Conformer layers, hidden={audio_cfg.hidden_size}")
    print(f"  Text:  hidden={text_hidden_size}")

    # ---- Step 2: Download and process audio ----
    print("\n[2/7] Processing audio...")
    audio_path = download_audio(AUDIO_URL, AUDIO_LOCAL)
    raw_audio = load_audio(audio_path)
    print(f"  Audio: {len(raw_audio)} samples ({len(raw_audio)/16000:.1f}s at 16kHz)")

    mel_features, mel_mask = extract_mel_features(raw_audio)
    print(f"  Mel spectrogram: {mel_features.shape}")

    num_audio_tokens = compute_audio_num_tokens(len(raw_audio))
    print(f"  Audio tokens after subsampling: {num_audio_tokens}")

    # ---- Step 3: Compile/load audio encoder on Neuron ----
    print("\n[3/7] Audio encoder (Neuron)...")
    neuron_audio = NeuronAudioEncoder.from_pretrained(
        args.model_path, audio_cfg, text_hidden_size,
    )

    audio_model_file = os.path.join(args.audio_compiled_path, "model.pt")
    if not os.path.exists(audio_model_file):
        print(f"  Compiling audio encoder to {args.audio_compiled_path}...")
        t0 = time.time()
        neuron_audio.compile(args.audio_compiled_path)
        print(f"  Audio encoder compiled in {time.time()-t0:.1f}s")
    else:
        print(f"  Loading cached audio encoder: {args.audio_compiled_path}")
        neuron_audio.load(args.audio_compiled_path)

    # Run audio encoder (CPU subsampling + Neuron Conformer)
    t0 = time.time()
    audio_embeddings, valid_mask = neuron_audio(mel_features, mel_mask)
    t1 = time.time()

    if valid_mask is not None:
        audio_embeddings_valid = audio_embeddings[0][valid_mask]
    else:
        audio_embeddings_valid = audio_embeddings[0]

    print(f"  Audio embeddings (valid): {audio_embeddings_valid.shape}")
    print(f"  Encoder time: {t1-t0:.2f}s (CPU subsample + Neuron Conformer)")

    # ---- Step 4: Build input sequence ----
    print("\n[4/7] Building input sequence...")
    prompt = (
        "Transcribe the following speech segment in its original language. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
        "one point seven, and write 3 instead of three."
    )

    actual_audio_tokens = audio_embeddings_valid.shape[0]
    input_ids, audio_token_id = build_audio_input_ids(actual_audio_tokens, prompt, args.model_path)
    print(f"  Input sequence length: {input_ids.shape[1]}")
    print(f"  Audio token positions: {actual_audio_tokens}")

    # Check sequence fits in compiled model
    if input_ids.shape[1] > args.seq_len:
        print(f"  WARNING: Input ({input_ids.shape[1]}) > seq_len ({args.seq_len}), truncating")
        input_ids = input_ids[:, :args.seq_len]

    # Build vision_embeddings/vision_mask at bucket size (seq_len) so NxDI
    # pad_inputs won't replace them with dummy zeros
    padded_input_ids = torch.nn.functional.pad(
        input_ids, (0, args.seq_len - input_ids.shape[1]),
        value=0,
    )
    vision_embeddings, vision_mask = build_vision_embeddings(
        padded_input_ids, audio_embeddings_valid.unsqueeze(0),
        audio_token_id, text_hidden_size, torch.bfloat16,
    )
    print(f"  vision_embeddings: {vision_embeddings.shape}")
    print(f"  vision_mask: {vision_mask.shape}")

    # ---- Step 5: Compile text decoder with multimodal support ----
    print(f"\n[5/7] Compiling text decoder (multimodal)...")
    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        seq_len=args.seq_len,
        max_context_length=args.seq_len,
        torch_dtype=torch.bfloat16,
        save_sharded_checkpoint=True,
        attn_kernel_enabled=False,
    )

    config = Gemma4InferenceConfig.from_pretrained(args.model_path, neuron_config=neuron_config)

    compiled_path = args.compiled_path
    model = NeuronGemma4ForConditionalGeneration(args.model_path, config)

    if not os.path.exists(os.path.join(compiled_path, "text_model", "model.pt")):
        print(f"  Compiling to {compiled_path}...")
        t0 = time.time()
        model.compile(compiled_path)
        print(f"  Compilation done in {time.time()-t0:.1f}s")
    else:
        print(f"  Using cached compilation: {compiled_path}")

    # ---- Step 6: Load and run inference ----
    print(f"\n[6/7] Loading and running inference on Neuron...")
    model.load(compiled_path)

    # Limit generation to avoid overflowing the KV cache
    max_gen = min(args.max_new_tokens, args.seq_len - input_ids.shape[1])
    print(f"  Max generation tokens: {max_gen} (seq_len={args.seq_len}, input={input_ids.shape[1]})")

    print(f"  [DIAG] ve norm at audio pos: {vision_embeddings[0, :actual_audio_tokens].norm(dim=-1).mean():.2f}")
    print(f"  [DIAG] vm first 5: {vision_mask[0, :5, 0].tolist()}")
    print(f"  [DIAG] vm pad[0]: {vision_mask[0, actual_audio_tokens, 0].item()}")

    t0 = time.time()
    generated_ids = generate_with_audio(
        model, input_ids, vision_embeddings, vision_mask,
        max_new_tokens=max_gen,
        eos_token_ids=eos_token_ids,
        audio_token_id=audio_token_id,
    )
    t1 = time.time()

    # ---- Step 7: Decode and display result ----
    print(f"\n[7/7] Results:")

    # Decode using tokenizer
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
        # Get only the generated tokens (after input)
        new_ids = generated_ids[0, input_ids.shape[1]:].tolist()
        transcription = tokenizer.decode(new_ids)
    except Exception as e:
        transcription = f"[decode error: {e}]"
        new_ids = generated_ids[0, input_ids.shape[1]:].tolist()

    num_new_tokens = len(new_ids)
    total_time = t1 - t0

    print(f"  Generated {num_new_tokens} tokens in {total_time:.2f}s")
    if num_new_tokens > 0:
        print(f"  Throughput: {num_new_tokens/total_time:.1f} tok/s")
    print(f"\n  Transcription:")
    print(f"  {transcription}")
    print("\n" + "=" * 70)
    print("Audio verification complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
