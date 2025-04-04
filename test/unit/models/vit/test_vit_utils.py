import logging
import os
import time
import warnings
from functools import partial

import numpy as np
import torch
import torch_xla
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.model_builder import ModelBuilder

from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance
from neuronx_distributed_inference.utils.benchmark import LatencyCollector

CKPT_DIR = "/tmp/test_vit/"
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    save_output_path = f"{device}_output.pt"
    torch.save(output, save_output_path)
    logger.info(f"Saved output to : {save_output_path}")

    return output


def get_checkpoint_loader_fn():
    state_dict = torch.load(os.path.join(CKPT_DIR, "checkpoint.pt"), map_location="cpu")
    # map state dicts names if needed, eg:
    # state_dict["gate_proj.weight"] = state_dict.pop("w1.weight")
    return state_dict


def get_compiler_args():
    # Flag for model type
    compiler_args = "-O1 --model-type=transformer"
    # Add flags for cc-overlap
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    # Prevent auto-down casting when running with fp32
    compiler_args += " --auto-cast=none"
    compiler_args += " --logfile='log-neuron-cc.txt '"
    logger.info(f"compiler_args: {compiler_args}")
    return compiler_args


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
        # compiler_args=None
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
    # we need to start a distributed process on cpu
    init_cpu_env()  # nn layers does not need this

    cpu_model = model_cls(config, **kwargs)
    # save state dict to be used to trace
    save_ckpt_path = os.path.join(CKPT_DIR, "checkpoint.pt")
    torch.save(cpu_model.state_dict(), save_ckpt_path)

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
