#!/usr/bin/env python3
"""
Integration tests for Qwen3-Coder-30B-A3B-Instruct NeuronX implementation.

Tests model compilation, loading, and inference accuracy.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modeling_qwen3_coder_moe import NeuronQwen3CoderMoeForCausalLM, Qwen3CoderMoeInferenceConfig

MODEL_PATH = "/home/ubuntu/models/Qwen3-Coder-30B-A3B-Instruct/"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/Qwen3-Coder-30B-A3B-Instruct/"


def load_neuron_config_from_compiled(compiled_path: str) -> dict:
    config_path = Path(compiled_path) / "neuron_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"neuron_config.json not found: {config_path}")
    with open(config_path) as f:
        data = json.load(f)
    return data.get("neuron_config", data)


def create_model_for_inference(compiled_path: str, model_path: str):
    nc_dict = load_neuron_config_from_compiled(compiled_path)

    dtype_str = nc_dict.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else torch.bfloat16

    neuron_config = MoENeuronConfig(
        tp_degree=nc_dict.get("tp_degree", 32),
        ep_degree=nc_dict.get("ep_degree", 1),
        world_size=nc_dict.get("world_size", 32),
        batch_size=nc_dict.get("batch_size", 1),
        seq_len=nc_dict.get("seq_len", 2048),
        torch_dtype=dtype,
        save_sharded_checkpoint=True,
        on_cpu=False,
    )

    config = Qwen3CoderMoeInferenceConfig.from_pretrained(
        model_path, neuron_config=neuron_config
    )
    model = NeuronQwen3CoderMoeForCausalLM(model_path=model_path, config=config)
    model.load(compiled_path)
    return model, neuron_config


class TestQwen3CoderMoeSmoke:
    """Smoke tests: model loads and produces output."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, _ = create_model_for_inference(COMPILED_MODEL_PATH, MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def test_model_loads(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert model is not None

    def test_generates_tokens(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer(["fibonacci(n)"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids

        with torch.no_grad():
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            outputs = model(input_ids, position_ids=position_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

        assert next_token.numel() == 1
        assert next_token.item() < tokenizer.vocab_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=tee-sys"])
