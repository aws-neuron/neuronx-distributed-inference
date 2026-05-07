
import random
from pathlib import Path
import tempfile

import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers import Gemma3Config
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from neuronx_distributed_inference.utils.random import set_random_seed


@pytest.fixture
def base_compiler_flags():
    return [
        "--framework=XLA",
    ]


@pytest.fixture(scope="session")
def random_seed():
    seed = 42
    set_random_seed(seed)
    xm.set_rng_state(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@pytest.fixture(scope="session")
def hf_config():
    return Gemma3Config.from_pretrained((Path(__file__).parent / "assets" / "gemma3_27b_config.json"))


@pytest.fixture
def tmp_dir_path():
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)
    yield tmp_dir_path
    tmp_dir.cleanup()
