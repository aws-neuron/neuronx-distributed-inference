#!/usr/bin/env python3
"""
Integration tests for blenderbot-3B NeuronX encoder-decoder implementation.

Tests weight splitting, compilation, loading, and inference accuracy
against the HuggingFace reference model.
"""

import pytest
import torch
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from configuration_blenderbot_neuron import BlenderbotInferenceConfig, BlenderbotNeuronConfig
from blenderbot_application import NeuronApplicationBlenderbot, split_hf_weights


# Test configuration - update paths for your environment
MODEL_PATH = os.environ.get("BLENDERBOT_MODEL_PATH", "/shared/dhwanw2/models/blenderbot-3B")
SPLIT_MODEL_PATH = os.environ.get("BLENDERBOT_SPLIT_PATH", "/tmp/blenderbot_split")
COMPILED_MODEL_PATH = os.environ.get("BLENDERBOT_COMPILED_PATH", "/tmp/blenderbot_compiled")

TP_DEGREE = 8
BATCH_SIZE = 1
SEQ_LEN = 128
DTYPE = "float32"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def neuron_model():
    """Split, compile, and load the Neuron model once for all tests."""
    # Split weights
    if not os.path.exists(os.path.join(SPLIT_MODEL_PATH, "encoder", "config.json")):
        split_hf_weights(MODEL_PATH, SPLIT_MODEL_PATH)

    neuron_config = BlenderbotNeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        torch_dtype=DTYPE,
        save_sharded_checkpoint=True,
    )

    config = BlenderbotInferenceConfig.from_pretrained(
        os.path.join(SPLIT_MODEL_PATH, "encoder"),
        neuron_config=neuron_config,
    )

    app = NeuronApplicationBlenderbot(model_path=SPLIT_MODEL_PATH, config=config)

    # Compile if needed
    if not os.path.exists(COMPILED_MODEL_PATH):
        app.compile(COMPILED_MODEL_PATH)

    app.load(COMPILED_MODEL_PATH)
    return app, config


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace reference model."""
    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.eval()
    return model


class TestBlenderbotInference:
    """Test blenderbot-3B inference accuracy on Neuron."""

    def test_smoke(self, neuron_model):
        """Test that model loads successfully."""
        app, config = neuron_model
        assert app is not None
        assert config.d_model == 2560
        assert config.encoder_layers == 2
        assert config.decoder_layers == 24

    @pytest.mark.parametrize("prompt", [
        "What is the capital of France?",
        "Tell me about machine learning.",
        "How are you doing today?",
        "What is the meaning of life?",
        "Can you help me with something?",
    ])
    def test_generation_matches_hf(self, neuron_model, hf_model, tokenizer, prompt):
        """Test that Neuron generation matches HF reference."""
        app, config = neuron_model
        max_new_tokens = 20

        input_ids = tokenizer([prompt], return_tensors="pt", padding=True).input_ids

        # HF generation
        with torch.no_grad():
            hf_output = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)

        # Neuron generation
        neuron_output = app.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=getattr(config, 'decoder_start_token_id', 1),
            eos_token_id=getattr(config, 'eos_token_id', 2),
            pad_token_id=getattr(config, 'pad_token_id', 0),
        )
        neuron_text = tokenizer.decode(neuron_output[0], skip_special_tokens=True)

        # Token-level comparison
        hf_tokens = hf_output[0].tolist()
        neuron_tokens = neuron_output[0].tolist()
        min_len = min(len(hf_tokens), len(neuron_tokens))
        matches = sum(1 for a, b in zip(hf_tokens[:min_len], neuron_tokens[:min_len]) if a == b)
        match_pct = matches / min_len * 100 if min_len > 0 else 0

        print(f"\nPrompt: {prompt}")
        print(f"  HF:     {hf_text}")
        print(f"  Neuron: {neuron_text}")
        print(f"  Match:  {matches}/{min_len} ({match_pct:.1f}%)")

        # Expect at least 70% match (Neuron may stop at EOS earlier)
        assert match_pct >= 70.0, f"Token match too low: {match_pct:.1f}%"

    def test_overall_accuracy(self, neuron_model, hf_model, tokenizer):
        """Test aggregate accuracy across multiple prompts."""
        app, config = neuron_model
        prompts = [
            "What is the capital of France?",
            "Tell me about machine learning.",
            "How are you doing today?",
        ]

        total_matches = 0
        total_tokens = 0

        for prompt in prompts:
            input_ids = tokenizer([prompt], return_tensors="pt", padding=True).input_ids

            with torch.no_grad():
                hf_output = hf_model.generate(
                    input_ids, max_new_tokens=20, do_sample=False, num_beams=1,
                )

            neuron_output = app.generate(
                input_ids, max_new_tokens=20,
                decoder_start_token_id=getattr(config, 'decoder_start_token_id', 1),
                eos_token_id=getattr(config, 'eos_token_id', 2),
                pad_token_id=getattr(config, 'pad_token_id', 0),
            )

            hf_tokens = hf_output[0].tolist()
            neuron_tokens = neuron_output[0].tolist()
            min_len = min(len(hf_tokens), len(neuron_tokens))
            matches = sum(1 for a, b in zip(hf_tokens[:min_len], neuron_tokens[:min_len]) if a == b)
            total_matches += matches
            total_tokens += min_len

        overall_pct = total_matches / total_tokens * 100 if total_tokens > 0 else 0
        print(f"\nOverall accuracy: {total_matches}/{total_tokens} ({overall_pct:.1f}%)")
        assert overall_pct >= 75.0, f"Overall accuracy too low: {overall_pct:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=tee-sys"])
