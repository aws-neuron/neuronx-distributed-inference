import logging
import os
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch_xla
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.model_builder import ModelBuilder
from safetensors.torch import save_file

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance
from neuronx_distributed_inference.models.vit.modeling_vit import (
    NeuronViTForImageEncoding,
    ViTInferenceConfig,
)
from neuronx_distributed_inference.utils.benchmark import LatencyCollector
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

CKPT_DIR = "/tmp/test_vit/"
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

base_path = Path(__file__).parent.parent.parent / "resources_multi_modal/vit"
VIT_CONFIG_PATH = [str(p) for p in base_path.glob("*") if p.is_dir()]

BENCHMARK_NUM_ITERATIONS = 10


def init_cpu_env():
    # destroy distributed process if already started
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    # if need to run distributed framework on CPU
    logger.info("Initializing cpu env")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8080"
    os.environ["RANK"] = "0"
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel()


def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


def get_rtol(data_type, num_layers=1):
    if num_layers < 10:
        model_type = "tiny"
    else:
        model_type = "full"
    rtol_map = {
        # (data_type, model_type): rtol,
        (torch.float32, "tiny"): 2e-6,
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
            # dtype casting
            if torch.is_floating_point(v) and v.dtype not in [torch.float8_e4m3fn]:
                randn_state_dict[k] = v.to(dtype)
    model.load_state_dict(randn_state_dict, strict=True)
    model.to(dtype)
    # keep layernorm in FP32
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(torch.float32)

    if ckpt_path.endswith(".pt"):
        torch.save(randn_state_dict, ckpt_path)
    elif ckpt_path.endswith(".safetensors"):
        save_file(randn_state_dict, ckpt_path)
    else:
        raise ValueError(f"Not support saving {ckpt_path}")
    return model


def get_model_output(model, inputs, device):
    latency_collector = LatencyCollector()

    logger.info(f"Model type {type(model)}!")
    logger.info(f"Calling {device} model!")
    for i in range(BENCHMARK_NUM_ITERATIONS):
        logger.info(f"{device} Iteration # {i}")
        latency_collector.pre_hook()
        output = model(*inputs)
        latency_collector.hook()

    # print report
    for p in [25, 50, 90, 99]:
        latency = np.percentile(latency_collector.latency_list, p) * 1000
        logger.info(f"{device} inference latency_ms_p{p}: {latency}")

    return output


def get_checkpoint_loader_fn():
    state_dict = torch.load(os.path.join(CKPT_DIR, "checkpoint.pt"), map_location="cpu")
    # map state dicts names if needed, eg:
    # state_dict["gate_proj.weight"] = state_dict.pop("w1.weight")
    return state_dict


def get_compiler_args():
    dummy_inference_config = ViTInferenceConfig(
        neuron_config=NeuronConfig(), load_config=load_pretrained_config(VIT_CONFIG_PATH[0])
    )
    dummy_vit_model = NeuronViTForImageEncoding(
        model_path=VIT_CONFIG_PATH[0], config=dummy_inference_config
    )
    return dummy_vit_model.get_compiler_args()


def trace_nxd_model(example_inputs, model_cls, config, **kwargs):
    if config.neuron_config.tp_degree > 2:
        warnings.warn(
            f"Unit tests run on trn1.2xlarge instance, allowing tp_degree up to 2. Got {config.neuron_config.tp_degree}. Falling back to 2."
        )
        config.neuron_config.tp_degree = 2
    model_builder = ModelBuilder(
        router=None,
        tp_degree=config.neuron_config.tp_degree,
        checkpoint_loader=get_checkpoint_loader_fn,
    )
    logger.info("Initiated model builder!")

    model_builder.add(
        key=model_cls.__name__,
        model_instance=BaseModelInstance(
            module_cls=partial(model_cls, config, **kwargs), input_output_aliases={}
        ),
        example_inputs=[example_inputs],
        priority_model_idx=0,
        compiler_args=get_compiler_args(),
    )
    logger.info("Added model builder! Starting to trace!")
    start_time = time.time()

    traced_model = model_builder.trace()

    elapsed_time = time.time() - start_time
    logger.info(f"Traced time taken {elapsed_time} s")

    logger.info("Done tracing the model!")
    return traced_model


def run_on_cpu(test_inputs, model_cls, config, **kwargs):
    # If the original implementation uses distributed framework,
    # use init_cpu_env() to start a distributed process on cpu

    cpu_model = model_cls(config)
    # save random weights to be used to trace
    save_ckpt_path = os.path.join(CKPT_DIR, "checkpoint.pt")
    dtype = kwargs.pop("dtype", torch.float32)
    cpu_model = get_rand_weights(cpu_model, save_ckpt_path, dtype)

    logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

    # inference and benchmark
    cpu_output = get_model_output(cpu_model, test_inputs, device="cpu")

    # destroy distributed process to reinit for neuron
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    return cpu_output


def run_on_neuron(test_inputs, model_cls, config, **kwargs):
    # trace model
    example_inputs = tuple(torch.ones_like(input) for input in test_inputs)
    neuron_model = trace_nxd_model(example_inputs, model_cls, config, **kwargs)

    # inference and benchmark
    neuron_output = get_model_output(neuron_model, test_inputs, device="neuron")

    return neuron_output
