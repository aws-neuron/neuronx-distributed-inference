"""Shared pytest fixtures for Solar Open MoE tests.

Provides session-scoped fixtures for integration tests:
- model_dir: tiny random checkpoint in a temp directory
- traced_dir: temp directory for compiled Neuron model
- compiled_model: NeuronSolarOpenForCausalLM compiled once per test session
- neuron_config: MoENeuronConfig for the integration tests
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure contrib src is on path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


CONFIG_JSON = Path(__file__).parent / "integration" / "config_solar_open_2layers.json"


@pytest.fixture(scope="session")
def solar_open_config_dict():
    """Load Solar Open test config (2 layers, reduced dims)."""
    import json

    with open(CONFIG_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, solar_open_config_dict):
    """Create a temporary tiny random Solar Open model directory."""
    from test.integration.utils import create_tiny_solar_open_model

    tmpdir = tmp_path_factory.mktemp("solar_open_model")
    create_tiny_solar_open_model(str(tmpdir), str(CONFIG_JSON))
    return str(tmpdir)


@pytest.fixture(scope="session")
def traced_dir(tmp_path_factory):
    """Temporary directory for the compiled Neuron model."""
    return str(tmp_path_factory.mktemp("solar_open_traced"))


@pytest.fixture(scope="session")
def neuron_config():
    """MoENeuronConfig for integration tests."""
    from test.integration.utils import get_neuron_config

    return get_neuron_config()


@pytest.fixture(scope="session")
def compiled_model(model_dir, traced_dir, neuron_config):
    """Compile NeuronSolarOpenForCausalLM from tiny random checkpoint.

    Skips if Neuron hardware (NeuronCores) is not available.
    Compiles once per test session and returns the loaded model.
    """
    try:
        from solar_open.modeling_solar_open import (
            NeuronSolarOpenForCausalLM,
            SolarOpenInferenceConfig,
            load_solar_open_config,
        )
    except ImportError as e:
        pytest.skip(f"solar_open package not importable: {e}")

    try:
        import torch_neuronx  # noqa: F401
    except ImportError:
        pytest.skip("torch_neuronx not available — Neuron hardware required")

    # Compile
    config = SolarOpenInferenceConfig(
        neuron_config,
        load_config=load_solar_open_config(model_dir),
    )
    model = NeuronSolarOpenForCausalLM(model_dir, config)
    model.compile(traced_dir)

    # Copy weights and generation_config so load() and HuggingFaceGenerationAdapter can find them
    import shutil
    import os

    for fname in ("model.safetensors", "generation_config.json"):
        src = os.path.join(model_dir, fname)
        dst = os.path.join(traced_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Load compiled model
    model = NeuronSolarOpenForCausalLM(traced_dir)
    model.load(traced_dir)
    return model
