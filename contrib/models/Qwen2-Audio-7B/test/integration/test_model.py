#!/usr/bin/env python3
"""
Integration tests for Qwen2-Audio-7B NeuronX implementation.
"""

import pytest
import torch
import json
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from transformers import GenerationConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_qwen2_audio import NeuronQwen2AudioForConditionalGeneration
from configuration_qwen2_audio import Qwen2AudioMultimodalConfig, Qwen2AudioEncoderNeuronConfig


MODEL_PATH = "/home/ubuntu/models/Qwen2-Audio-7B/"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/Qwen2-Audio-7B/"


@pytest.fixture(scope="module")
def model_and_adapter():
    """Load pre-compiled model and wrap with generation adapter."""
    text_nc = NeuronConfig(
        tp_degree=2, batch_size=1, seq_len=1024,
        torch_dtype="bfloat16", on_cpu=False, save_sharded_checkpoint=False,
    )
    audio_nc = Qwen2AudioEncoderNeuronConfig(
        tp_degree=2, batch_size=1, seq_len=1500,
        torch_dtype="bfloat16", on_cpu=False, fused_qkv=False,
        buckets=[1], save_sharded_checkpoint=False,
    )
    config = Qwen2AudioMultimodalConfig.from_pretrained(
        MODEL_PATH, text_neuron_config=text_nc, audio_neuron_config=audio_nc,
    )
    model = NeuronQwen2AudioForConditionalGeneration(model_path=MODEL_PATH, config=config)
    model.load(COMPILED_MODEL_PATH)
    adapter = HuggingFaceGenerationAdapter(model)
    return model, adapter


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def gen_config():
    return GenerationConfig(
        do_sample=False, bos_token_id=151643,
        eos_token_id=[151645], pad_token_id=151643,
    )


@pytest.fixture(scope="module")
def sampling_params():
    return prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0])


def test_model_loads(model_and_adapter):
    """Test that model loads successfully."""
    model, _ = model_and_adapter
    assert model is not None
    assert hasattr(model, "config")


def test_text_only_generation(model_and_adapter, tokenizer, gen_config, sampling_params):
    """Test text-only inference (no audio)."""
    _, adapter = model_and_adapter
    input_ids = tokenizer(["What is the capital of France?"], return_tensors="pt").input_ids
    attention_mask = torch.ones_like(input_ids)
    outputs = adapter.generate(
        input_ids, attention_mask=attention_mask,
        sampling_params=sampling_params, generation_config=gen_config,
        max_new_tokens=10,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(text) > len("What is the capital of France?")


def test_audio_captioning(model_and_adapter, processor, tokenizer, gen_config, sampling_params):
    """Test audio-to-text captioning with a sample audio file."""
    _, adapter = model_and_adapter
    import librosa
    # Use a test audio file — adjust path as needed
    audio, sr = librosa.load("/tmp/test_audio.mp3", sr=processor.feature_extractor.sampling_rate)
    inputs = processor(
        text="<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:",
        audio=audio, return_tensors="pt", sampling_rate=sr,
    )
    outputs = adapter.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        audio_features=inputs["input_features"].to(torch.bfloat16),
        sampling_params=sampling_params, generation_config=gen_config,
        max_new_tokens=20,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert "English:" in text
    caption = text.split("English:")[1].strip()
    assert len(caption) > 0


if __name__ == "__main__":
    print("Run with: pytest <this_file> --capture=tee-sys")
