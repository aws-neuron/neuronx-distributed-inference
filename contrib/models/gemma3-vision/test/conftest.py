
import random
from pathlib import Path
import tempfile

from neuronx_distributed.parallel_layers import parallel_state
import pytest
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import init_cpu_env


@pytest.fixture
def neuron_env(monkeypatch, tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("neuron-compile-cache")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "1")
    monkeypatch.setenv("NEURON_COMPILE_CACHE_URL", str(temp_dir))

@pytest.fixture
def cpu_xla_env(monkeypatch):
    # monkeypatch.setenv("PJRT_DEVICE", "CPU")
    init_cpu_env()
    monkeypatch.setenv("NXD_CPU_MODE", "1")
    

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


@pytest.fixture(scope="module")
def tensor_parallelism_setup():
    dist.init_process_group(backend="xla")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    yield
    parallel_state.destroy_model_parallel()


@pytest.fixture(scope="session")
def hf_text_config():
    return Gemma3TextConfig.from_pretrained(Path(__file__).parent / "assets" / "gemma3_text_config.json")  # nosec B615


@pytest.fixture
def cpu_xla_env(monkeypatch):    
    monkeypatch.setenv("PJRT_DEVICE", "CPU")


@pytest.fixture
def tmp_dir_path():
    import tempfile
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)
    yield tmp_dir_path
    tmp_dir.cleanup()
