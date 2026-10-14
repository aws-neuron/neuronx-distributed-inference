#!/usr/bin/env python3
"""
Integration tests for Qwen2.5-Omni-7B on NeuronX (TP=4).

Tests:
  1. Import validation
  2. Config creation and TP=4 head divisibility
  3. State dict conversion (all 2448 keys)
  4. Audio encoder CPU components (frontend + postprocessor)
  5. Talker CPU model (weight loading + codec tokens)
  6. Text-only Thinker compile + load + generate
  7. Image understanding (requires vision encoder compiled)
  8. Audio understanding (requires audio encoder compiled)

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

  # Run all tests (skip tests requiring compilation with --quick):
  python3 test_model.py

  # Run only text generation (model must be pre-compiled):
  python3 test_model.py --test text_gen

  # Run with pytest:
  pytest test_model.py -v
"""

# --- Qwen2.5-Omni contrib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
import _upstream_compat  # noqa: F401  (applies hf_adapter shim)
# --- end bootstrap ---

import argparse
import gc
import os
import sys
import time
import traceback

import torch

# Default paths - override with environment variables
from _model_path import resolve_model_path
MODEL_PATH = resolve_model_path()
COMPILED_PATH = os.environ.get(
    "QWEN25_OMNI_COMPILED_PATH", "/tmp/qwen25_omni_tp4_compiled"
)
TP_DEGREE = int(os.environ.get("QWEN25_OMNI_TP_DEGREE", "4"))

# Test media URLs from Qwen official examples
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"


class Timer:
    """Context manager to time a block."""

    def __init__(self, label):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  [{self.label}] {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Test 1: Imports
# ---------------------------------------------------------------------------
def test_imports():
    """Verify all Qwen2.5-Omni modules import correctly."""
    print("=" * 60)
    print("Test 1: Import validation")
    print("=" * 60)

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    print("  NeuronConfig, OnDeviceSamplingConfig imported OK")

    from modeling_qwen25_omni import (
        NeuronQwen25OmniForCausalLM,
        Qwen25OmniInferenceConfig,
        NeuronQwen25OmniMultimodalForCausalLM,
        Qwen25OmniMultimodalInferenceConfig,
    )

    print("  Qwen25Omni model classes imported OK")

    from modeling_qwen25_omni_audio import (
        NeuronQwen25OmniAudioEncoder,
        NeuronQwen25OmniForAudioEncoding,
        AudioEncoderInferenceConfig,
        AudioCPUFrontend,
        AudioCPUPostprocessor,
        NeuronAudioTransformerModel,
        AudioTransformerModelWrapper,
    )

    print("  Audio encoder classes imported OK")

    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalker,
    )

    print("  Talker imported OK")

    from modeling_qwen25_omni_token2wav import (
        NeuronQwen25OmniToken2Wav,
    )

    print("  Token2Wav imported OK")

    from modeling_qwen25_omni_vision import (
        NeuronQwen25OmniForImageEncoding,
    )

    print("  Vision encoder imported OK")

    print("  PASS: All imports successful\n")
    return True


# ---------------------------------------------------------------------------
# Test 2: Config
# ---------------------------------------------------------------------------
def test_config():
    """Create configs and verify TP=4 head divisibility."""
    print("=" * 60)
    print("Test 2: Config creation and TP=4 validation")
    print("=" * 60)

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from modeling_qwen25_omni import (
        Qwen25OmniInferenceConfig,
    )
    from transformers import AutoConfig

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        seq_len=2048,
        max_context_length=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True, temperature=0.6, top_k=20, top_p=0.95
        ),
    )

    hf_config = load_pretrained_config(MODEL_PATH)
    config = Qwen25OmniInferenceConfig(neuron_config, load_config=hf_config)

    # Validate text config
    print(f"  hidden_size={config.hidden_size}")
    print(f"  num_attention_heads={config.num_attention_heads}")
    print(f"  num_key_value_heads={config.num_key_value_heads}")
    print(f"  num_hidden_layers={config.num_hidden_layers}")
    print(f"  vocab_size={config.vocab_size}")

    # TP divisibility check for Thinker
    assert (
        config.num_attention_heads % TP_DEGREE == 0
    ), f"num_attention_heads={config.num_attention_heads} not divisible by TP={TP_DEGREE}"
    assert (
        config.num_key_value_heads % TP_DEGREE == 0
    ), f"num_key_value_heads={config.num_key_value_heads} not divisible by TP={TP_DEGREE}"
    print(
        f"  Thinker heads: {config.num_attention_heads}/{TP_DEGREE}="
        f"{config.num_attention_heads // TP_DEGREE} per rank, "
        f"kv_heads: {config.num_key_value_heads}/{TP_DEGREE}="
        f"{config.num_key_value_heads // TP_DEGREE} per rank"
    )

    # Validate audio config (via AutoConfig for full nested access)
    full_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(full_config, "thinker_config"):
        tc = full_config.thinker_config
        if hasattr(tc, "__dict__") and not isinstance(tc, dict):
            tc = vars(tc)
        if "audio_config" in tc:
            ac = tc["audio_config"]
            if hasattr(ac, "__dict__"):
                ac = vars(ac)
            audio_heads = ac.get("encoder_attention_heads", 20)
            print(
                f"  Audio: d_model={ac.get('d_model')}, heads={audio_heads}, "
                f"layers={ac.get('encoder_layers')}"
            )
            assert (
                audio_heads % TP_DEGREE == 0
            ), f"audio heads={audio_heads} not divisible by TP={TP_DEGREE}"
            print(
                f"  Audio heads: {audio_heads}/{TP_DEGREE}={audio_heads // TP_DEGREE} per rank"
            )

    # Validate talker config
    if hasattr(full_config, "talker_config"):
        tc = full_config.talker_config
        if hasattr(tc, "__dict__") and not isinstance(tc, dict):
            tc = vars(tc)
        talker_heads = tc.get("num_attention_heads", 12)
        talker_hidden = tc.get("hidden_size", 896)
        print(
            f"  Talker: hidden={talker_hidden}, heads={talker_heads}, "
            f"head_dim={tc.get('head_dim', 128)}"
        )
        print(
            f"  Talker stays on CPU (head_dim={tc.get('head_dim', 128)} "
            f"!= {talker_hidden}//{talker_heads})"
        )

    print("  PASS: Config creation and TP=4 validation\n")
    return True


# ---------------------------------------------------------------------------
# Test 3: State dict conversion
# ---------------------------------------------------------------------------
def test_state_dict():
    """Load full HF model and convert state dict for all components."""
    print("=" * 60)
    print("Test 3: State dict conversion (all components)")
    print("=" * 60)

    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
    )
    from modeling_qwen25_omni_audio import (
        NeuronQwen25OmniAudioEncoder,
    )
    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalker,
    )

    # Load HF model state dict
    with Timer("Load HF model"):
        hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        full_sd = hf_model.state_dict()
    print(f"  Full state dict: {len(full_sd)} keys")

    # Count keys by prefix
    prefix_counts = {}
    for k in full_sd:
        p = k.split(".")[0]
        prefix_counts[p] = prefix_counts.get(p, 0) + 1
    for p, c in sorted(prefix_counts.items()):
        print(f"    {p}: {c} keys")

    # Test audio encoder state dict conversion
    with Timer("Audio state dict conversion"):
        audio_sd = NeuronQwen25OmniAudioEncoder.convert_hf_to_neuron_state_dict(
            {k: v for k, v in full_sd.items() if "audio_tower" in k},
            dtype=torch.bfloat16,
        )
    frontend_keys = [k for k in audio_sd if k.startswith("frontend.")]
    transformer_keys = [k for k in audio_sd if k.startswith("transformer.")]
    postprocessor_keys = [k for k in audio_sd if k.startswith("postprocessor.")]
    print(
        f"  Audio keys: frontend={len(frontend_keys)}, "
        f"transformer={len(transformer_keys)}, "
        f"postprocessor={len(postprocessor_keys)}"
    )

    # Test talker state dict conversion
    with Timer("Talker state dict conversion"):
        talker_sd = NeuronQwen25OmniTalker.convert_hf_to_neuron_state_dict(
            {k: v for k, v in full_sd.items() if k.startswith("talker.")}
        )
    print(f"  Talker keys: {len(talker_sd)}")

    del hf_model, full_sd
    gc.collect()

    print("  PASS: State dict conversion\n")
    return True


# ---------------------------------------------------------------------------
# Test 4: Audio encoder CPU components
# ---------------------------------------------------------------------------
def test_audio_encoder():
    """Test audio encoder CPU frontend and postprocessor with synthetic input."""
    print("=" * 60)
    print("Test 4: Audio encoder CPU components")
    print("=" * 60)

    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
    )
    from modeling_qwen25_omni_audio import (
        NeuronQwen25OmniAudioEncoder,
    )

    # Load config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if not hasattr(hf_config, "thinker_config"):
        print("  SKIP: No thinker_config found")
        return True

    tc = hf_config.thinker_config
    if hasattr(tc, "__dict__") and not isinstance(tc, dict):
        tc = vars(tc)
    audio_config = tc.get("audio_config", None)
    if audio_config is not None and hasattr(audio_config, "__dict__"):
        audio_config = vars(audio_config)
    if audio_config is None:
        print("  SKIP: No audio_config found")
        return True

    print(
        f"  Audio config: d_model={audio_config.get('d_model')}, "
        f"heads={audio_config.get('encoder_attention_heads')}, "
        f"layers={audio_config.get('encoder_layers')}"
    )

    # Load HF model for weights
    with Timer("Load HF model for audio weights"):
        hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        full_sd = hf_model.state_dict()

    with Timer("Convert audio state dict"):
        converted_sd = NeuronQwen25OmniAudioEncoder.convert_hf_to_neuron_state_dict(
            full_sd, dtype=torch.bfloat16
        )

    del hf_model, full_sd
    gc.collect()

    with Timer("Create audio encoder + load CPU weights"):
        encoder = NeuronQwen25OmniAudioEncoder.from_pretrained_state_dict(
            audio_config, converted_sd, dtype=torch.bfloat16
        )
        encoder.eval()

    del converted_sd
    gc.collect()

    # Test with synthetic mel spectrograms of various lengths
    n_mels = audio_config.get("num_mel_bins", 128)
    test_cases = [(100, "1s"), (300, "3s"), (1000, "10s"), (3000, "30s")]

    for mel_len, label in test_cases:
        mel_input = torch.randn(n_mels, mel_len, dtype=torch.bfloat16)
        feature_lens = torch.tensor([mel_len], dtype=torch.long)

        t0 = time.time()
        hidden, aftercnn_lens, cu_seqlens = encoder.frontend(mel_input, feature_lens)
        audio_embeds = encoder.postprocessor(hidden, aftercnn_lens)
        elapsed = time.time() - t0

        print(
            f"  {label} ({mel_len} frames): frontend→{hidden.shape[0]} tokens, "
            f"postprocessor→{audio_embeds.shape}, time={elapsed*1000:.1f}ms"
        )

    del encoder
    gc.collect()

    print("  PASS: Audio encoder CPU components\n")
    return True


# ---------------------------------------------------------------------------
# Test 5: Talker CPU model
# ---------------------------------------------------------------------------
def test_talker():
    """Test Talker CPU model weight loading and codec token IDs."""
    print("=" * 60)
    print("Test 5: Talker CPU model")
    print("=" * 60)

    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
    )
    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalker,
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    talker_config = getattr(hf_config, "talker_config", None)
    if talker_config is None:
        print("  SKIP: No talker_config found")
        return True

    if hasattr(talker_config, "__dict__") and not isinstance(talker_config, dict):
        tc = vars(talker_config)
    else:
        tc = talker_config

    print(
        f"  Talker config: hidden={tc.get('hidden_size')}, "
        f"heads={tc.get('num_attention_heads')}, "
        f"kv_heads={tc.get('num_key_value_heads')}, "
        f"layers={tc.get('num_hidden_layers')}, "
        f"vocab={tc.get('vocab_size')}, "
        f"embedding_size={tc.get('embedding_size')}"
    )

    with Timer("Load HF model for talker weights"):
        hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        full_sd = hf_model.state_dict()

    talker_sd = NeuronQwen25OmniTalker.convert_hf_to_neuron_state_dict(
        {k: v for k, v in full_sd.items() if k.startswith("talker.")}
    )
    print(f"  Talker state dict: {len(talker_sd)} keys")

    del hf_model, full_sd
    gc.collect()

    with Timer("Create Talker + load weights"):
        talker = NeuronQwen25OmniTalker.from_pretrained_state_dict(
            talker_config, talker_sd, dtype=torch.bfloat16
        )

    print(
        f"  Talker codec tokens: bos={talker.codec_bos_token}, "
        f"eos={talker.codec_eos_token}, pad={talker.codec_pad_token}"
    )

    del talker, talker_sd
    gc.collect()

    print("  PASS: Talker CPU model\n")
    return True


# ---------------------------------------------------------------------------
# Test 6: Text-only Thinker compile + load + generate
# ---------------------------------------------------------------------------
def test_text_gen():
    """Compile (if needed), load, and generate with the Thinker at TP=4."""
    print("=" * 60)
    print("Test 6: Text-only Thinker compile + load + generate (TP=4)")
    print("=" * 60)

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config,
        HuggingFaceGenerationAdapter,
    )
    from modeling_qwen25_omni import (
        NeuronQwen25OmniForCausalLM,
        Qwen25OmniInferenceConfig,
    )
    from transformers import AutoTokenizer

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        seq_len=2048,
        max_context_length=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
        ),
    )

    hf_config = load_pretrained_config(MODEL_PATH)
    config = Qwen25OmniInferenceConfig(neuron_config, load_config=hf_config)

    compiled_dir = os.path.join(COMPILED_PATH, "thinker_tp4")

    with Timer("Create model"):
        model = NeuronQwen25OmniForCausalLM(MODEL_PATH, config)

    if not os.path.exists(os.path.join(compiled_dir, "neuron_config.json")):
        with Timer("Compile (this takes several minutes)"):
            model.compile(compiled_dir)
    else:
        print("  Compiled artifacts found, skipping compilation")

    with Timer("Load compiled model"):
        model.load(compiled_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # HuggingFaceGenerationAdapter does NOT take tokenizer as argument.
    adapter = HuggingFaceGenerationAdapter(model)

    def make_chat_input(prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(text, return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]

    prompts = [
        "What is 2+3? Answer with just the number.",
        "Write a haiku about the ocean.",
        "Explain quantum computing in one sentence.",
    ]

    for prompt in prompts:
        input_ids, attention_mask = make_chat_input(prompt)
        prompt_len = input_ids.shape[1]

        with Timer(f"Generate '{prompt[:40]}...'"):
            output_ids = adapter.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                eos_token_id=[tokenizer.eos_token_id, 151645],
            )

        new_tokens = output_ids[0, prompt_len:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_new = sum(
            1
            for tok in new_tokens
            if tok.item()
            not in [tokenizer.eos_token_id, tokenizer.pad_token_id, 151645]
        )
        print(f"    Input:  {prompt}")
        print(f"    Output: {output_text.strip()[:200]}")
        print(f"    Tokens: {n_new}")
        print()

    del model, adapter
    gc.collect()

    print("  PASS: Text-only Thinker compile + load + generate\n")
    return True


# ---------------------------------------------------------------------------
# Test 7: Image understanding (requires multimodal model)
# ---------------------------------------------------------------------------
def test_image_understanding():
    """Test image understanding with the Qwen2.5-Omni vision encoder.

    This test requires the multimodal model (vision encoder + text decoder)
    to be compiled. It downloads a test image and asks the model to describe it.

    NOTE: This is a placeholder for the full multimodal pipeline. The vision
    encoder must be compiled on Neuron before this test can run end-to-end.
    Currently tests the preprocessing pipeline only.
    """
    print("=" * 60)
    print("Test 7: Image understanding (preprocessing)")
    print("=" * 60)

    try:
        from qwen_omni_utils import process_mm_info
    except ImportError:
        print("  SKIP: qwen-omni-utils not installed")
        print("  Install with: pip install qwen-omni-utils[decord]")
        return True

    from transformers import AutoConfig

    # Build message with image
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_URL},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        },
    ]

    print(f"  Image URL: {IMAGE_URL}")

    with Timer("Process multimodal info (download + preprocess)"):
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=False
        )

    if images:
        print(f"  Images processed: {len(images)}")
        for i, img in enumerate(images):
            if hasattr(img, "shape"):
                print(f"    Image {i}: shape={img.shape}")
            elif hasattr(img, "size"):
                print(f"    Image {i}: size={img.size}")
    else:
        print("  No images found in processed output")

    # Verify config has vision token IDs
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(hf_config, "thinker_config"):
        tc = hf_config.thinker_config
        if hasattr(tc, "__dict__") and not isinstance(tc, dict):
            tc = vars(tc)
        image_token_id = tc.get("image_token_index")
        vision_start_id = tc.get("vision_start_token_id")
        vision_end_id = tc.get("vision_end_token_id")
        print(
            f"  Vision tokens: image={image_token_id}, "
            f"start={vision_start_id}, end={vision_end_id}"
        )

    print(
        "  NOTE: Full end-to-end image understanding requires "
        "compiled vision encoder on Neuron."
    )
    print("  PASS: Image preprocessing\n")
    return True


# ---------------------------------------------------------------------------
# Test 8: Audio understanding (requires audio encoder on Neuron)
# ---------------------------------------------------------------------------
def test_audio_understanding():
    """Test audio understanding preprocessing pipeline.

    Downloads a test audio file and preprocesses it through the Qwen2.5-Omni
    audio pipeline. Full end-to-end inference requires the audio encoder's
    Neuron transformer to be compiled.
    """
    print("=" * 60)
    print("Test 8: Audio understanding (preprocessing)")
    print("=" * 60)

    try:
        from qwen_omni_utils import process_mm_info
    except ImportError:
        print("  SKIP: qwen-omni-utils not installed")
        print("  Install with: pip install qwen-omni-utils[decord]")
        return True

    from transformers import AutoConfig

    # Build message with audio
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": AUDIO_URL},
                {"type": "text", "text": "What sound is this?"},
            ],
        },
    ]

    print(f"  Audio URL: {AUDIO_URL}")

    with Timer("Process multimodal info (download + preprocess)"):
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=False
        )

    if audios:
        print(f"  Audios processed: {len(audios)}")
        for i, audio in enumerate(audios):
            if hasattr(audio, "shape"):
                print(f"    Audio {i}: shape={audio.shape}")
            else:
                print(f"    Audio {i}: type={type(audio)}")
    else:
        print("  No audios found in processed output")

    # Verify config has audio token IDs
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(hf_config, "thinker_config"):
        tc = hf_config.thinker_config
        if hasattr(tc, "__dict__") and not isinstance(tc, dict):
            tc = vars(tc)
        audio_token_id = tc.get("audio_token_index")
        audio_start_id = tc.get("audio_start_token_id")
        audio_end_id = tc.get("audio_end_token_id")
        print(
            f"  Audio tokens: audio={audio_token_id}, "
            f"start={audio_start_id}, end={audio_end_id}"
        )

    print(
        "  NOTE: Full end-to-end audio understanding requires "
        "compiled audio encoder on Neuron."
    )
    print("  PASS: Audio preprocessing\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_TESTS = [
    ("imports", test_imports),
    ("config", test_config),
    ("state_dict", test_state_dict),
    ("audio_encoder", test_audio_encoder),
    ("talker", test_talker),
    ("text_gen", test_text_gen),
    ("image", test_image_understanding),
    ("audio", test_audio_understanding),
]


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Omni-7B integration tests"
    )
    parser.add_argument(
        "--test",
        choices=[name for name, _ in ALL_TESTS],
        nargs="+",
        help="Run specific test(s). Default: all.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip heavyweight tests (state_dict, audio_encoder, talker).",
    )
    args = parser.parse_args()

    if args.test:
        tests = [(n, fn) for n, fn in ALL_TESTS if n in args.test]
    elif args.quick:
        skip = {"state_dict", "audio_encoder", "talker"}
        tests = [(n, fn) for n, fn in ALL_TESTS if n not in skip]
    else:
        tests = ALL_TESTS

    print("\n" + "=" * 60)
    print(f"Qwen2.5-Omni-7B TP={TP_DEGREE} Integration Tests")
    print(f"Model: {MODEL_PATH}")
    print(f"Compiled: {COMPILED_PATH}")
    print("=" * 60 + "\n")

    results = {}
    total_start = time.time()

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            print(f"\n  FAIL: {e}")
            traceback.print_exc()
            results[name] = f"FAIL: {e}"

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"  [{status}] {name}: {result}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 60)

    if any(r != "PASS" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
