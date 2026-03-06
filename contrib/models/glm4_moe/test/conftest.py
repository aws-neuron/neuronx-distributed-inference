# coding=utf-8
"""Shared pytest fixtures for GLM-4.5 MoE tests."""

import random
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Add src to path so glm4_moe package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session", autouse=True)
def random_seed():
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(42)
    random.seed(42)
    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(42)
    except ImportError:
        pass
    yield


@pytest.fixture(scope="session")
def glm4_moe_config():
    """Load GLM-4.5 MoE test config (full architecture, few layers)."""
    import json

    config_path = Path(__file__).parent / "integration" / "config_glm4_moe_2layers.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def tmp_dir_path():
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
