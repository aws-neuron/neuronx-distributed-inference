#!/usr/bin/env python3
"""Integration tests for MiMo-V2-Flash NeuronX implementation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_config_import():
    """Test that config class can be imported."""
    from modeling_mimo_v2 import MiMoV2InferenceConfig, NeuronMiMoV2ForCausalLM
    assert MiMoV2InferenceConfig is not None
    assert NeuronMiMoV2ForCausalLM is not None
    print("PASS: Config and model classes imported successfully")


def test_required_attributes():
    """Test that required attributes are defined."""
    from modeling_mimo_v2 import MiMoV2InferenceConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    import torch

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=1,
        seq_len=512,
        torch_dtype=torch.bfloat16,
        on_cpu=True,
    )
    config = MiMoV2InferenceConfig(neuron_config)
    required = config.get_required_attributes()
    assert "hidden_size" in required
    assert "n_routed_experts" in required
    assert "num_experts_per_tok" in required
    assert "hybrid_layer_pattern" in required
    print(f"PASS: {len(required)} required attributes defined")


def test_neuron_config_cls():
    """Test that MoENeuronConfig is returned."""
    from modeling_mimo_v2 import MiMoV2InferenceConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    assert MiMoV2InferenceConfig.get_neuron_config_cls() == MoENeuronConfig
    print("PASS: MoENeuronConfig returned")


if __name__ == "__main__":
    test_config_import()
    test_required_attributes()
    test_neuron_config_cls()
    print("\nAll tests passed!")
