#!/usr/bin/env python3
"""Integration tests for MiniMax M2 NeuronX implementation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_config_import():
    """Test that config class can be imported."""
    from modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
    assert MiniMaxM2InferenceConfig is not None
    assert NeuronMiniMaxM2ForCausalLM is not None
    print("PASS: Config and model classes imported successfully")


def test_required_attributes():
    """Test that required attributes are defined."""
    from modeling_minimax_m2 import MiniMaxM2InferenceConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from transformers import AutoConfig
    import torch

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=1,
        seq_len=512,
        torch_dtype=torch.bfloat16,
        on_cpu=True,
    )
    # Use the bundled config.json to provide model-specific attributes
    config_path = Path(__file__).resolve().parents[2] / "src"
    hf_config = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)
    config = MiniMaxM2InferenceConfig(neuron_config, load_config=load_pretrained_config(hf_config=hf_config))
    required = config.get_required_attributes()
    assert "hidden_size" in required
    assert "num_local_experts" in required
    assert "num_experts_per_tok" in required
    print(f"PASS: {len(required)} required attributes defined")


def test_neuron_config_cls():
    """Test that MoENeuronConfig is returned."""
    from modeling_minimax_m2 import MiniMaxM2InferenceConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    assert MiniMaxM2InferenceConfig.get_neuron_config_cls() == MoENeuronConfig
    print("PASS: MoENeuronConfig returned")


if __name__ == "__main__":
    test_config_import()
    test_required_attributes()
    test_neuron_config_cls()
    print("\nAll tests passed!")
