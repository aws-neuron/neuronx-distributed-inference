#!/usr/bin/env python3
"""Integration tests for Aria NeuronX implementation."""

import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_aria_text import NeuronAriaTextForCausalLM, AriaTextInferenceConfig

MODEL_PATH = "/shared/dhwanw2/models/Aria"
COMPILED_MODEL_PATH = "contrib/models/Aria/compiled_model"


def load_neuron_config_from_compiled(compiled_path: str):
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        config_data = json.load(f)
    return config_data.get("neuron_config", config_data)


def create_model_for_inference(compiled_path: str, model_path: str):
    neuron_config_dict = load_neuron_config_from_compiled(compiled_path)
    dtype_str = neuron_config_dict.get('torch_dtype', 'torch.bfloat16')
    if isinstance(dtype_str, str):
        dtype = getattr(torch, dtype_str.split('.')[1]) if dtype_str.startswith('torch.') else torch.bfloat16
    else:
        dtype = dtype_str
    neuron_config = NeuronConfig(
        tp_degree=neuron_config_dict.get('tp_degree', 2),
        batch_size=neuron_config_dict.get('batch_size', 1),
        seq_len=neuron_config_dict.get('seq_len', 512),
        torch_dtype=dtype,
        save_sharded_checkpoint=True,
        on_cpu=False,
    )
    try:
        model_config = AriaTextInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
    except (TypeError, AttributeError):
        model_config = AriaTextInferenceConfig(neuron_config, load_config=load_pretrained_config(model_path))
    try:
        model = NeuronAriaTextForCausalLM.from_pretrained(compiled_path, config=model_config)
    except Exception:
        model = NeuronAriaTextForCausalLM(model_path, model_config)
    return model, neuron_config


@pytest.fixture(scope="module")
def compiled_model():
    model, _ = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_model_loads(compiled_model):
    assert compiled_model is not None
    print("Smoke test passed")


def test_model_generates(compiled_model, tokenizer):
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    seq_len = inputs.input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    with torch.no_grad():
        outputs = compiled_model(inputs.input_ids, position_ids=position_ids)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    assert logits is not None
    print("Generation test passed")
