# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for Voxtral Mini 3B on Neuron.

Two accuracy paths are covered:

1. **Text-only logit validation** via `check_accuracy_logits_v2` — the
   text decoder is a Llama backbone with scatter injection disabled at
   inference time when no audio is provided, so we can validate it the
   same way as any other Llama-family model.

2. **Audio transcription smoke** — compare the Neuron-generated
   transcription of a reference TED clip against a CPU BF16 reference
   generation, over the first NUM_AUDIO_TOKENS_TO_CHECK output tokens,
   using `neuron_allclose` on the argmax token IDs.

Prerequisites:
    pip install 'transformers>=4.54.0' 'mistral_common[audio]>=1.8.1' pytest

Environment variables:
    VOXTRAL_MODEL_PATH      Path to HF model weights (default: /home/ubuntu/models/voxtral-mini-3B)
    VOXTRAL_COMPILED_PATH   Path for compiled NEFFs (default: /home/ubuntu/compiled_models/voxtral-mini-3B)
    VOXTRAL_AUDIO_FILE      Path or URL to test audio file (default: TED-60 sample)
    VOXTRAL_TP_DEGREE       Tensor parallel degree (default: 1)
    VOXTRAL_SEQ_LEN         Context encoding bucket size (default: 2048)
    VOXTRAL_N_POSITIONS     KV cache length (default: 4096)
    VOXTRAL_ODS             Set to "1" to enable on-device sampling
    VOXTRAL_SKIP_CPU_REF    Set to "1" to skip the CPU logit-reference test
                            (useful for smoke-only runs on small memory hosts).
"""

import os
import sys
import time

import pytest
import torch

# Add the src directory to the path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from modeling_voxtral import NeuronApplicationVoxtral  # noqa: E402

# --- Configuration --------------------------------------------------------
MODEL_PATH = os.environ.get(
    "VOXTRAL_MODEL_PATH", "/home/ubuntu/models/voxtral-mini-3B"
)
COMPILED_PATH = os.environ.get(
    "VOXTRAL_COMPILED_PATH", "/home/ubuntu/compiled_models/voxtral-mini-3B"
)
AUDIO_FILE = os.environ.get(
    "VOXTRAL_AUDIO_FILE",
    "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav",
)
TP_DEGREE = int(os.environ.get("VOXTRAL_TP_DEGREE", "1"))
SEQ_LEN = int(os.environ.get("VOXTRAL_SEQ_LEN", "2048"))
N_POSITIONS = int(os.environ.get("VOXTRAL_N_POSITIONS", "4096"))
ODS = os.environ.get("VOXTRAL_ODS") == "1"
SKIP_CPU_REF = os.environ.get("VOXTRAL_SKIP_CPU_REF") == "1"
DTYPE = torch.bfloat16

# Accuracy thresholds
NUM_TEXT_TOKENS_TO_CHECK = 16
NUM_AUDIO_TOKENS_TO_CHECK = 16
LOGIT_DIVERGENCE_TOL = 0.02  # matches other multimodal contribs (gemma3-vision)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_model():
    """Compile and load the Voxtral model (module-scoped for reuse across tests)."""
    app = NeuronApplicationVoxtral(
        model_path=MODEL_PATH,
        tp_degree=TP_DEGREE,
        seq_len=SEQ_LEN,
        n_positions=N_POSITIONS,
        dtype=DTYPE,
        on_device_sampling=ODS,
    )

    # Compile if needed
    marker = os.path.join(COMPILED_PATH, "text_decoder", "text_model", "model.pt")
    if not os.path.exists(marker):
        print(f"\nCompiling Voxtral model to {COMPILED_PATH}...")
        app.compile(COMPILED_PATH)

    print(f"\nLoading compiled Voxtral model from {COMPILED_PATH}...")
    app.load(COMPILED_PATH)
    return app


# --- Tests ---------------------------------------------------------------


def test_model_loads(loaded_model):
    """Smoke test: all components load successfully."""
    assert loaded_model.audio_encoder is not None
    assert loaded_model.projector is not None
    assert loaded_model.vl_model is not None
    assert loaded_model.adapter is not None
    assert loaded_model.tokenizer is not None
    assert loaded_model.processor is not None


def test_text_generation_deterministic(loaded_model):
    """Greedy generation must be reproducible run-to-run."""
    result1 = loaded_model.generate("What is 2 + 2?", max_new_tokens=20)
    result2 = loaded_model.generate("What is 2 + 2?", max_new_tokens=20)
    assert result1 == result2, (
        f"Non-deterministic generation:\n  Run 1: {result1}\n  Run 2: {result2}"
    )


@pytest.mark.skipif(
    SKIP_CPU_REF,
    reason="VOXTRAL_SKIP_CPU_REF=1 set (CPU logit reference generation is slow "
    "on 4B params).",
)
def test_text_logit_validation(loaded_model):
    """Compare Neuron text-only logits against a CPU BF16 reference.

    Uses `check_accuracy_logits_v2` — the standard NxDI multi-tier tolerance
    check on the top-5, top-50, top-1000, and all-token logit
    distributions.  Runs on the Ministral-3B Llama backbone (text-only,
    no audio).
    """
    from neuronx_distributed_inference.utils.accuracy import (
        check_accuracy_logits_v2,
        generate_expected_logits,
    )
    from transformers import GenerationConfig

    prompt = "What is the capital of France?"
    input_ids = loaded_model.tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=NUM_TEXT_TOKENS_TO_CHECK,
    )

    # The neuron_model argument to generate_expected_logits is the NxDI
    # NeuronApplicationBase — for Voxtral, that's the wrapped VL text
    # decoder (vl_model).
    print(f"\nGenerating expected logits on CPU (this can take several "
          f"minutes for {NUM_TEXT_TOKENS_TO_CHECK} tokens on a 4B model)...")

    expected_logits = generate_expected_logits(
        neuron_model=loaded_model.vl_model,
        input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        generation_config=generation_config,
        num_tokens=NUM_TEXT_TOKENS_TO_CHECK,
        additional_input_args=None,
    )

    print("Running Neuron logit accuracy check...")
    check_accuracy_logits_v2(
        neuron_model=loaded_model.vl_model,
        expected_logits=expected_logits,
        inputs_input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        generation_config=generation_config,
        num_tokens_to_check=NUM_TEXT_TOKENS_TO_CHECK,
        additional_input_args=None,
        divergence_difference_tol=LOGIT_DIVERGENCE_TOL,
    )


def test_audio_transcription_matches_cpu_reference(loaded_model):
    """Audio transcription: compare the first N generated tokens vs CPU BF16 reference.

    We don't run `check_accuracy_logits_v2` on the audio path because it
    requires a matching CPU forward through the exact scatter-injection
    pipeline that only Neuron implements.  Instead: generate on both and
    require the first N argmax tokens to match exactly (BF16 greedy is
    deterministic once the input is fixed).
    """
    # 1. Neuron transcription
    neuron_text = loaded_model.transcribe(
        AUDIO_FILE, max_new_tokens=NUM_AUDIO_TOKENS_TO_CHECK * 2
    )
    assert len(neuron_text.strip()) > 0, "Neuron transcription is empty"

    # Take the first NUM_AUDIO_TOKENS_TO_CHECK tokens
    neuron_ids = loaded_model.tokenizer.encode(neuron_text, add_special_tokens=False)
    neuron_ids = torch.tensor(neuron_ids[:NUM_AUDIO_TOKENS_TO_CHECK])

    if SKIP_CPU_REF:
        pytest.skip(
            "VOXTRAL_SKIP_CPU_REF=1 set — skipping CPU reference comparison "
            "for audio path.  Neuron path produced non-empty output."
        )

    # 2. CPU BF16 reference — full HF model
    print("\nRunning CPU BF16 reference transcription (slow)...")
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    cpu_model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=DTYPE
    )
    cpu_model.eval()

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": AUDIO_FILE},
            {"type": "text", "text": "Transcribe this audio."},
        ],
    }]
    inputs = processor.apply_chat_template(conversation, return_tensors="pt")

    with torch.no_grad():
        cpu_ids = cpu_model.generate(
            **inputs,
            max_new_tokens=NUM_AUDIO_TOKENS_TO_CHECK * 2,
            do_sample=False,
        )
    # Strip prompt tokens
    cpu_generated = cpu_ids[0, inputs["input_ids"].shape[1]:]
    cpu_generated = cpu_generated[:NUM_AUDIO_TOKENS_TO_CHECK]

    # Compare
    n = min(len(neuron_ids), len(cpu_generated))
    match = int((neuron_ids[:n] == cpu_generated[:n]).sum().item())
    match_rate = match / n if n > 0 else 0.0
    print(f"Audio token match: {match}/{n} = {match_rate:.1%}")
    print(f"Neuron: {loaded_model.tokenizer.decode(neuron_ids[:n])!r}")
    print(f"CPU:    {loaded_model.tokenizer.decode(cpu_generated[:n])!r}")

    # Allow one drift token — greedy BF16 is deterministic in principle,
    # but tokenizer round-trip and CPU/Neuron kernel differences can
    # produce a single token difference near the tail.
    assert match_rate >= (n - 1) / n, (
        f"Audio transcription drift: {match}/{n} tokens match CPU reference"
    )


if __name__ == "__main__":
    # CLI mode for local sanity runs (bypasses pytest).
    print("=" * 60)
    print("Voxtral Mini 3B - Integration Test")
    print("=" * 60)
    print(f"Model path:    {MODEL_PATH}")
    print(f"Compiled path: {COMPILED_PATH}")
    print(f"Audio file:    {AUDIO_FILE}")
    print(f"TP degree:     {TP_DEGREE}")
    print(f"Seq len:       {SEQ_LEN}")
    print(f"N positions:   {N_POSITIONS}")
    print(f"ODS:           {ODS}")
    print(f"Dtype:         {DTYPE}")
    print()

    app = NeuronApplicationVoxtral(
        model_path=MODEL_PATH,
        tp_degree=TP_DEGREE,
        seq_len=SEQ_LEN,
        n_positions=N_POSITIONS,
        dtype=DTYPE,
        on_device_sampling=ODS,
    )

    marker = os.path.join(COMPILED_PATH, "text_decoder", "text_model", "model.pt")
    if not os.path.exists(marker):
        print("Compiling model...")
        app.compile(COMPILED_PATH)
    else:
        print("Using existing compiled model.\n")

    print("Loading compiled model...")
    app.load(COMPILED_PATH)
    print("Model loaded.\n")

    print("--- Text-only generation ---")
    start = time.perf_counter()
    result = app.generate("What is the capital of France?", max_new_tokens=50)
    print(f"Result:  {result}")
    print(f"Latency: {(time.perf_counter() - start) * 1000:.1f} ms\n")

    print("--- Audio transcription ---")
    start = time.perf_counter()
    result = app.transcribe(AUDIO_FILE, max_new_tokens=256)
    print(f"Transcription: {result}")
    print(f"Latency: {(time.perf_counter() - start) * 1000:.1f} ms\n")

    print("--- Determinism check ---")
    r1 = app.generate("What is 2 + 2?", max_new_tokens=20)
    r2 = app.generate("What is 2 + 2?", max_new_tokens=20)
    print("Determinism:", "PASS" if r1 == r2 else "FAIL")
    print("\nAll CLI checks done.  Run `pytest test/integration/test_model.py -v`")
    print("for the full accuracy suite (includes CPU logit reference).")
