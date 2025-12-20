import os
import warnings
import torch
import torch_xla

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed_inference.utils.random import set_random_seed

def init_cpu_env(dist_framework="fairscale"):
    # destroy distributed process if already started
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # if need to run distributed framework on CPU
    print("Initializing cpu env")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8080"
    os.environ["RANK"] = "0"
    torch.distributed.init_process_group(backend="gloo")
    if dist_framework == "fairscale":
        # fairscale model parallel group init
        from fairscale.nn.model_parallel import initialize_model_parallel
        initialize_model_parallel(model_parallel_size_=1, model_parallel_backend="gloo")
    elif dist_framework == "nxd":
        # nxd model parallel group init 
        parallel_state.initialize_model_parallel()

def destroy_cpu_env():
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    from fairscale.nn.model_parallel import  destroy_model_parallel
    destroy_model_parallel()
    os.environ["NXD_CPU_MODE"] = "0"

def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    # for trn2
    os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    torch_xla._XLAC._set_ir_debug(True)
    set_random_seed(0)

def get_rtol(data_type, num_layers=1):
    if num_layers < 10:
        model_type = "tiny"
    else:
        model_type = "full"
    rtol_map = {
        # (data_type, model_type): rtol,
        (torch.float32, "tiny"): 1.3e-6,
        (torch.float32, "full"): 0.01,
        (torch.float16, "tiny"): 1.6e-3,
        (torch.float16, "full"): 0.05,
        (torch.bfloat16, "tiny"): 1.6e-2,
        (torch.bfloat16, "full"): 0.05,
    }
    if (data_type, model_type) in rtol_map:
        return rtol_map[(data_type, model_type)]
    else:
        warnings.warn(f"Does not support data_type {data_type} model_type {model_type} num_layers {num_layers}. Using rtol=0.0")
        return 0.0

def rand_interval(a, b, *size):
  return (b - a) * torch.rand(*size) + a