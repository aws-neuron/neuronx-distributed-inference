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
    # Check get_required_attributes without instantiation (requires many params)
    required = MiMoV2InferenceConfig.get_required_attributes(MiMoV2InferenceConfig)
    assert "hidden_size" in required
    assert "n_routed_experts" in required
    assert "num_experts_per_tok" in required
    assert "hybrid_layer_pattern" in required
    assert "v_head_dim" in required
    assert "swa_head_dim" in required
    print(f"PASS: {len(required)} required attributes defined")


def test_neuron_config_cls():
    """Test that MoENeuronConfig is returned."""
    from modeling_mimo_v2 import MiMoV2InferenceConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    assert MiMoV2InferenceConfig.get_neuron_config_cls() == MoENeuronConfig
    print("PASS: MoENeuronConfig returned")


def test_state_dict_converter():
    """Test that state dict converter function exists."""
    from modeling_mimo_v2 import NeuronMiMoV2ForCausalLM
    assert hasattr(NeuronMiMoV2ForCausalLM, "convert_hf_to_neuron_state_dict")
    print("PASS: State dict converter exists")


if __name__ == "__main__":
    test_config_import()
    test_required_attributes()
    test_neuron_config_cls()
    test_state_dict_converter()
    print("\nAll tests passed!")
