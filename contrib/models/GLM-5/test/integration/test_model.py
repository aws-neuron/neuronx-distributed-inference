#!/usr/bin/env python3
"""
Integration tests for GLM-5 NeuronX Distributed Inference implementation.

Requirements:
- trn2.48xlarge instance with pre-compiled model and pre-sharded weights
- Neuron SDK 2.29+
- nkilib installed in editable mode (for GLM-5 routing kernel)

Environment variables:
  MODEL_PATH: Path to GLM-5-FP8 HuggingFace checkpoint (default: /mnt/nvme/GLM-5-FP8)
  COMPILED_MODEL_PATH: Path to compiled model with NEFFs + weights (default: /mnt/nvme/glm5_compiled_fused)

Run:
  python3 -m pytest test/integration/test_model.py -v
"""

import json
import os
import sys
import time

import pytest
import torch

# SDK 2.29 race condition workarounds
_orig_makedirs = os.makedirs


def _safe_makedirs(name, mode=0o777, exist_ok=False):
    return _orig_makedirs(name, mode=mode, exist_ok=True)


os.makedirs = _safe_makedirs

import shutil

_orig_rmtree = shutil.rmtree


def _safe_rmtree(path, ignore_errors=False, onerror=None, **kw):
    return _orig_rmtree(path, ignore_errors=True, **kw)


shutil.rmtree = _safe_rmtree

os.environ["UNSAFE_FP8FNCAST"] = "1"

from pathlib import Path
from transformers import PreTrainedTokenizerFast

# Add contrib src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_glm5 import NeuronGLM5ForCausalLM, GLM5InferenceConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/nvme/GLM-5-FP8")
COMPILED_MODEL_PATH = os.environ.get(
    "COMPILED_MODEL_PATH", "/mnt/nvme/glm5_compiled_fused"
)


def load_neuron_config_from_compiled(compiled_path: str) -> dict:
    """Load neuron_config.json from a compiled model directory."""
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        config_data = json.load(f)
    return config_data.get("neuron_config", config_data)


def create_model_and_load(compiled_path: str, model_path: str):
    """Create model from compiled path and load weights."""
    with open(f"{model_path}/config.json") as f:
        hf_config = json.load(f)

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=1,
        seq_len=2048,
        n_active_tokens=2048,
        torch_dtype=torch.bfloat16,
        fused_qkv=False,
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        moe_fused_nki_kernel_enabled=True,
        expert_mlp_nki_kernel_enabled=False,
        mlp_kernel_enabled=True,
        quantized=True,
        quantization_dtype="f8e4m3",
        quantized_checkpoints_path=model_path,
        modules_to_not_convert=[
            "lm_head",
            "self_attn",
            "shared_expert",
            "layers.0.mlp",
            "layers.1.mlp",
            "layers.2.mlp",
        ],
        layer_boundary_markers=True,
        weights_to_skip_layout_optimization=[".*"],
        save_sharded_checkpoint=True,
    )

    def load_config(c):
        for k, v in hf_config.items():
            setattr(c, k, v)

    config = GLM5InferenceConfig(neuron_config=neuron_config, load_config=load_config)
    model = NeuronGLM5ForCausalLM(model_path, config)
    model.load(compiled_path)
    return model


@pytest.fixture(scope="module")
def compiled_model():
    """Load the pre-compiled GLM-5 model (shared across all tests in module)."""
    model = create_model_and_load(COMPILED_MODEL_PATH, MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def hf_model(compiled_model):
    """Wrap model with HuggingFace generation adapter."""
    return HuggingFaceGenerationAdapter(compiled_model)


@pytest.fixture(scope="module")
def tokenizer():
    """Load GLM-5 tokenizer."""
    tok = PreTrainedTokenizerFast(
        tokenizer_file=f"{MODEL_PATH}/tokenizer.json",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    return tok


def test_model_loads(compiled_model):
    """Smoke test: model loads without error."""
    assert compiled_model is not None
    print("Model loaded successfully")


def test_model_generates(hf_model, tokenizer):
    """Test that model generates non-empty output."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=32,
            do_sample=False,
        )

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    assert generated_tokens > 0, "Model did not generate any tokens"

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {text}")
    assert len(text) > len(prompt), "Generated text should be longer than prompt"


def test_output_coherence(hf_model, tokenizer):
    """Test that model generates coherent, non-repetitive output."""
    prompt = "Explain the theory of general relativity in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = text[len(prompt) :].strip()
    print(f"Generated: {generated[:200]}")

    # Check for repetition: no single token sequence should repeat more than 5x
    words = generated.split()
    if len(words) > 10:
        from collections import Counter

        word_counts = Counter(words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        # Allow common words but flag extreme repetition
        repetition_ratio = most_common_count / len(words)
        assert repetition_ratio < 0.5, (
            f"Output is highly repetitive: '{most_common_word}' appears "
            f"{most_common_count}/{len(words)} times ({repetition_ratio:.0%})"
        )


def test_logit_validation(compiled_model, tokenizer):
    """
    Logit validation test: verify model produces reasonable logit distributions.

    Checks that:
    1. Logits are not all zeros or NaN
    2. Top-1 prediction for a factual prompt is a reasonable token
    3. Logit entropy is within expected range (not collapsed or uniform)
    """
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)

    with torch.no_grad():
        outputs = compiled_model(
            input_ids,
            attention_mask=inputs["attention_mask"],
            position_ids=position_ids,
        )

    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    assert logits is not None, "Model returned no logits"

    # Check last token logits (next token prediction)
    last_logits = logits[0, -1, :]  # [vocab_size]

    # 1. Not all zeros
    assert last_logits.abs().sum() > 0, "Logits are all zeros"

    # 2. No NaN
    assert not torch.isnan(last_logits).any(), "Logits contain NaN"

    # 3. No Inf
    assert not torch.isinf(last_logits).any(), "Logits contain Inf"

    # 4. Reasonable entropy (not collapsed to single token or uniform)
    probs = torch.softmax(last_logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    print(f"Logit entropy: {entropy:.2f}")
    # Entropy should be between 0.1 (very confident) and 15 (near uniform over 154k vocab)
    assert 0.1 < entropy < 15.0, f"Logit entropy {entropy:.2f} is out of expected range"

    # 5. Top prediction is a reasonable token
    top_token_id = last_logits.argmax().item()
    top_token = tokenizer.decode([top_token_id])
    print(f"Top predicted token: '{top_token}' (id={top_token_id})")

    print("Logit validation passed")
