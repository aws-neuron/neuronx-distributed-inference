import os
import shutil
import uuid
import warnings
from pathlib import Path

import torch
import torch_xla
from neuronx_distributed.parallel_layers import parallel_state
from safetensors.torch import load_file, save_file

from neuronx_distributed_inference.models.llama4.modeling_llama4 import (
    Llama4InferenceConfig,
    Llama4NeuronConfig,
    NeuronLlama4ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
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
    from fairscale.nn.model_parallel import destroy_model_parallel

    destroy_model_parallel()
    os.environ["NXD_CPU_MODE"] = "0"


def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    # for trn2
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
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
        warnings.warn(
            f"Does not support data_type {data_type} model_type {model_type} num_layers {num_layers}. Using rtol=0.0"
        )
        return 0.0


def get_compiler_args():
    # Instantiate a dummy model to use the same compiler args defined there
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_16E_4layer.json")
    dummy_inference_config = Llama4InferenceConfig(
        text_neuron_config=Llama4NeuronConfig(),
        vision_neuron_config=Llama4NeuronConfig(),
        load_config=load_pretrained_config(config_path),
    )
    dummy_llama4_model = NeuronLlama4ForCausalLM(
        model_path=config_path, config=dummy_inference_config
    )
    compiler_args = dummy_llama4_model.get_compiler_args()

    # delete the model after we got the compiler args
    del dummy_llama4_model

    return compiler_args


def rand_interval(a, b, *size):
    return (b - a) * torch.rand(*size) + a


def get_rand_weights(model: torch.nn.Module, ckpt_path: str, dtype=torch.float32):
    randn_state_dict = {}
    for k, v in model.state_dict().items():
        # set different range for weight and bias
        if k.endswith("weight"):
            randn_state_dict[k] = torch.nn.Parameter(rand_interval(-0.05, 0.05, (v.shape))).to(
                dtype
            )
        elif k.endswith("bias"):
            randn_state_dict[k] = torch.nn.Parameter(rand_interval(-0.25, 0.25, (v.shape))).to(
                dtype
            )
        else:
            warnings.warn(f"Unsupported state dict key {k}, skip converting to random value")
            randn_state_dict[k] = v
    model.load_state_dict(randn_state_dict, strict=True)
    model.to(dtype)

    if ckpt_path.endswith(".pt"):
        torch.save(randn_state_dict, ckpt_path)
    elif ckpt_path.endswith(".safetensors"):
        save_file(randn_state_dict, ckpt_path)
    else:
        raise ValueError(f"Not support saving {ckpt_path}")
    return model


# Patch torch.Tensor.cuda() to bypass cuda() calls in the reference implementation
def patch_tensor_cuda():
    prev_cuda_fn = torch.Tensor.cuda

    def cuda_passthrough(self):
        if torch.cuda.is_available():
            return prev_cuda_fn(self)
        return self

    return cuda_passthrough


torch.Tensor.cuda = patch_tensor_cuda()


def get_tmp_workdir():
    # Get the current working directory
    cwd = os.getcwd()
    _id = uuid.uuid4()
    tmp_workdir = os.path.join(cwd, f"llama4_test_{_id}")
    os.makedirs(tmp_workdir)
    return tmp_workdir


def cleanup_tmp_workdir(tmp_workdir):
    if os.path.exists(tmp_workdir):
        shutil.rmtree(tmp_workdir)
    else:
        warnings.warn(f"Cannot find {tmp_workdir} to clean up. Skipping.")
    return
