import logging
import uuid
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch_neuronx
from torch.distributions.uniform import Uniform
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance
import os

from neuronx_distributed_inference.utils.random import set_random_seed

logger = logging.getLogger("Neuron")


class FunctionModule(torch.nn.Module):
    """
    A module that wraps a function to run it on Neuron.
    """

    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, *args):
        return self.func(*args)


def destroy_mp():
    # destroy distributed process if already started
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def init_cpu_env():
    """
    If the CPU implementation uses a distributed framework,
    We will need to call this function first.
    """
    destroy_mp()
    print("Initializing cpu env")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8080"
    os.environ["RANK"] = "0"
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel()


def validate_accuracy(
    neuron_model,
    inputs: List[Tuple],
    expected_outputs: Optional[List] = None,
    cpu_callable: Optional[Callable] = None,
    assert_close_kwargs: Dict = {},
):
    """
    Validates the accuracy of a Neuron model. This function tests that the model produces expected
    outputs, which you can provide and/or produce on CPU. To compare outputs, this function uses
    torch_neuronx.testing.assert_close. If the output isn't similar, this function raises an
    AssertionError.

    Args:
        neuron_model: The Neuron model to validate.
        inputs: The list of inputs to use to run the model. Each input is passed to the model's
            forward function.
        expected_outputs: The list of expected outputs for each input. If not provided, this
            function compares against the CPU output for each input.
        cpu_callable: The callable to use to produce output on CPU.
        assert_close_kwargs: The kwargs to pass to torch_neuronx.testing.assert_close.
    """
    if expected_outputs is None and cpu_callable is None:
        raise ValueError("Provide expected_outputs or a cpu_callable to produce expected outputs")

    if not _is_tensor_tuple_list(inputs):
        raise ValueError("inputs must be a list of tensor tuples")
    if len(inputs) == 0:
        raise ValueError("inputs must not be empty")

    if expected_outputs is None:
        expected_outputs = [None] * len(inputs)
    if not isinstance(expected_outputs, list):
        raise ValueError("expected_outputs must be a list")
    if len(expected_outputs) != len(inputs):
        raise ValueError("len(expected_outputs) must match len(inputs)")

    for input, expected_output in zip(inputs, expected_outputs):
        logger.info(f"Validating model accuracy with input: {input}")
        if cpu_callable is not None:
            cpu_output = cpu_callable(*input)
            logger.info(f"CPU output: {cpu_output}")
            if expected_output is not None:
                torch_neuronx.testing.assert_close(
                    expected_output, cpu_output, **assert_close_kwargs
                )
            else:
                expected_output = cpu_output

        neuron_output = neuron_model(*input)
        logger.info(f"Expected output: {expected_output}")
        logger.info(f"Neuron output: {neuron_output}")
        torch_neuronx.testing.assert_close(expected_output, neuron_output, **assert_close_kwargs)
        logger.info(f"Model is accurate for input: {input}")


def build_function(
    func: Callable,
    example_inputs: List[Tuple[torch.Tensor]],
    tp_degree: int = 1,
    compiler_args: Optional[str] = None,
    compiler_workdir: Optional[str] = None,
    priority_model_idx: Optional[int] = 0,
    logical_nc_config: int = 1,
    dry_run: bool = False,
    checkpoint_loader_fn=lambda checkpoint_path: torch.load(checkpoint_path),
):
    """
    Compiles a function to Neuron.

    If the function has non-tensor inputs, you must convert it to a function that only takes
    tensor inputs. You can use `partial` to do this, where you provide the non-tensor inputs as
    constants in the partial function. This step is necessary because all inputs must be tensors
    in a Neuron model.

    Args:
        func: The function to compile.
        example_inputs: The list of example inputs to use to trace the function. This list must
            contain exactly one tuple of tensors.
        tp_degree: The TP degree to use. Defaults to 1.
        compiler_args: The compiler args to use.
        compiler_workdir: Where to save compiler artifacts. Defaults to a tmp folder with a UUID
            for uniqueness.
        priority_model_idx: default 0 indicating enable WLO (weight layout optimization)
        logical_nc_config: The number of logical neuron cores to use. Defaults to 1.
        dry_run: Whether to stop after trace (before compile). If priority_model_idx is set, then
            dry run mode compiles the priority model in order to produce the weight layout
            optimization model.
        checkpoint_loader_fn: Customized checkpoint loader function. Defaults to torch.load for checkpoint_path.

    Returns:
        The Neuron model, or None if dry run mode is enabled.
    """
    return build_module(
        module_cls=FunctionModule,
        example_inputs=example_inputs,
        tp_degree=tp_degree,
        compiler_args=compiler_args,
        compiler_workdir=compiler_workdir,
        module_init_kwargs={"func": func},
        priority_model_idx=priority_model_idx,
        logical_nc_config=logical_nc_config,
        dry_run=dry_run,
        checkpoint_loader_fn=checkpoint_loader_fn,
    )


def build_module(
    module_cls,
    example_inputs: List[Tuple[torch.Tensor]],
    module_init_kwargs: Dict = {},
    tp_degree: int = 1,
    world_size: Optional[int] = None,  # if you want to consider ep degree, you may need to set it
    local_ranks_size: Optional[int] = None,  # if you set world_size, you also need to set it
    compiler_args: Optional[str] = None,
    compiler_workdir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    priority_model_idx: Optional[int] = 0,
    logical_nc_config: int = 1,
    dry_run: bool = False,
    checkpoint_loader_fn=lambda checkpoint_path: torch.load(checkpoint_path),
):
    """
    Compiles a module to Neuron.

    Args:
        module_cls: The module class to compile.
        example_inputs: The list of example inputs to use to trace the module. This list must
            contain exactly one tuple of tensors.
        tp_degree: The TP degree to use. Defaults to 1.
        module_init_kwargs: The kwargs to pass when initializing the module.
        compiler_args: The compiler args to use.
        compiler_workdir: Where to save compiler artifacts. Defaults to a tmp folder with a UUID
            for uniqueness.
        checkpoint_path: The path to the checkpoint to load. By default, this function saves the
            module state dict to use as the checkpoint.
        priority_model_idx: default 0 indicating enable WLO (weight layout optimization)
        logical_nc_config: The number of logical neuron cores to use. Defaults to 1.
        dry_run: Whether to stop after trace (before compile). If priority_model_idx is set, then
            dry run mode compiles the priority model in order to produce the weight layout
            optimization model.
        checkpoint_loader_fn: Customized checkpoint loader function. Defaults to torch.load for checkpoint_path.


    Returns:
        The Neuron model, or None if dry run mode is enabled.
    """
    if not _is_tensor_tuple_list(example_inputs):
        raise ValueError("example_inputs must be a list of tensor tuples")
    if len(example_inputs) != 1:
        # Bucketing isn't currently supported for this utility.
        raise ValueError("example_inputs must contain exactly one input")

    _id = uuid.uuid4()
    test_workdir = Path(f"/tmp/nxdi_test_{_id}")
    compiler_workdir = (
        Path(compiler_workdir)
        if compiler_workdir is not None
        else test_workdir / "compiler_workdir"
    )
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None else test_workdir / "checkpoint.pt"
    )
    logger.info(f"Saving to compiler workdir: {compiler_workdir}")
    logger.info(f"Using checkpoint path: {checkpoint_path}")

    if not checkpoint_path.exists():
        _save_checkpoint(
            module_cls, module_init_kwargs, checkpoint_path, tp_degree, world_size if world_size else tp_degree
        )

    if not compiler_workdir.exists():
        compiler_workdir.parent.mkdir(parents=True, exist_ok=True)

    model_builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        world_size=world_size,
        local_ranks_size=local_ranks_size,
        checkpoint_loader=partial(checkpoint_loader_fn, checkpoint_path),
        compiler_workdir=compiler_workdir,
        logical_nc_config=logical_nc_config,
    )

    module_instance_cls = partial(module_cls, **module_init_kwargs)
    model_builder.add(
        key=_get_module_name(module_cls, module_init_kwargs),
        model_instance=BaseModelInstance(module_instance_cls, input_output_aliases={}),
        example_inputs=example_inputs,
        compiler_args=compiler_args,
        priority_model_idx=priority_model_idx,
    )

    neuron_model = model_builder.trace(initialize_model_weights=True, dry_run=dry_run)
    if not dry_run:
        neuron_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
    return neuron_model


def build_cpu_model(model_cls, config, dtype=torch.float32, checkpoint_dir="/tmp/nxd_inference"):
    """
    Run original Huggingface model inference on CPU with randomly initialized weights.

    Args:
        model_cls: CPU Model class to instantiate
        config: CPU Model configuration
        dtype: torch data type for model (defaults to torch.float32)
        checkpoint_dir: Directory path where checkpoint files will be stored (defaults to "/tmp/nxd_inference")

    Returns:
        cpu_model: Original Huggingface model
        ckpt_path: The path for a shared checkpoint file used by both CPU and Neuron inference.

    Notes:
        - Initializes model with random weights
        - Saves weights checkpoint to ckpt_path for later using the same weights for Neuron hardware inference

    Process:
        1. Instantiates model with provided configuration
        2. Initializes random weights and saves checkpoint
        3. Returns model and the checkpoint path
    """
    cpu_model = model_cls(config)
    ckpt_path = _get_shared_checkpoint_path(checkpoint_dir)
    cpu_model = _get_rand_weights(cpu_model, ckpt_path, dtype)
    logger.info(f"Got cpu_model, saved checkpoint to {ckpt_path}")
    return cpu_model, ckpt_path


def _get_module_name(module_cls, module_init_kwargs):
    if module_cls == FunctionModule:
        module_cls = module_init_kwargs["func"]
    if isinstance(module_cls, partial):
        module_cls = module_cls.func
    return module_cls.__name__


def _save_checkpoint(module_cls, module_init_kwargs, checkpoint_path, tp_degree, world_size):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Parallel state is required to init modules that have distributed layers like RPL/CPL.
    destroy_mp()

    torch.distributed.init_process_group(backend="xla", rank=0, world_size=world_size)
    parallel_state.initialize_model_parallel(tp_degree)

    # Set the parallel state random seed to ensure random weights match modules initialized on CPU.
    set_random_seed(0)
    module = module_cls(**module_init_kwargs)
    torch.save(module.state_dict(), checkpoint_path)

    destroy_mp()


def _is_tensor_tuple_list(tensor_tuple_list):
    return isinstance(tensor_tuple_list, list) and all(
        _is_tensor_tuple(item) for item in tensor_tuple_list
    )


def _is_tensor_tuple(tensor_tuple):
    return isinstance(tensor_tuple, tuple) and all(
        isinstance(tensor, torch.Tensor) for tensor in tensor_tuple
    )


def _rand_interval(a: float, b: float, dtype: torch.dtype, *size: int) -> torch.Tensor:
    """
    Generate random numbers uniformly distributed in the interval [a, b).

    Args:
        a (float): Lower bound of the interval (inclusive)
        b (float): Upper bound of the interval (exclusive)
        dtype (torch.dtype): Data type for the output tensor
        *size (int): The shape dimensions of the output tensor

    Returns:
        torch.Tensor: A tensor of random numbers uniformly distributed between [a, b)
                     with the specified size and dtype

    Example:
        >>> _rand_interval(0, 1, torch.float16, 2, 3)  # Returns a 2x3 tensor with values between [0,1) in float16
        >>> _rand_interval(-1, 1, torch.float32, 5)    # Returns a tensor of size (5,) with values between [-1,1) in float32
    """
    return Uniform(a, b).sample(torch.Size(size)).to(dtype)


def _get_rand_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    dtype: torch.dtype = torch.float32,
    weight_range: tuple[float, float] = (-0.05, 0.05),
    bias_range: tuple[float, float] = (-0.25, 0.25),
) -> torch.nn.Module:
    """
    Initialize model weights and biases with random values within specified ranges and save to checkpoint.

    Args:
        model (torch.nn.Module): The PyTorch model to initialize
        ckpt_path (str): Path to save the checkpoint file (.pt or .safetensors)
        dtype (torch.dtype, optional): Data type for the model parameters. Defaults to torch.float32
        weight_range (tuple, optional): A uniform distribution over interval [min, max) for random weight initialization.
        bias_range (tuple, optional): A uniform distribution over interval [min, max) for random bias initialization.

    Returns:
        torch.nn.Module: The model with randomly initialized weights

    Raises:
        ValueError: If the checkpoint path format is not supported (.pt or .safetensors)

    Notes:
        - LayerNorm layers are kept in FP32 precision regardless of the specified dtype
        - Parameters not ending with 'weight' or 'bias' maintain their original values
        - Supports saving in either .pt or .safetensors format
    """
    randn_state_dict = {}
    for k, v in model.state_dict().items():
        # set different range for weight and bias
        if k.endswith("weight"):
            randn_state_dict[k] = torch.nn.Parameter(
                _rand_interval(weight_range[0], weight_range[1], dtype, *v.shape)
            )
        elif k.endswith("bias"):
            randn_state_dict[k] = torch.nn.Parameter(
                _rand_interval(bias_range[0], bias_range[1], dtype, *v.shape)
            )
        else:
            logger.warning(f"Unsupported state dict key {k}, skip converting to random value")
            # dtype casting
            if torch.is_floating_point(v) and v.dtype not in [torch.float8_e4m3fn]:
                randn_state_dict[k] = v.to(dtype)

    model.load_state_dict(randn_state_dict, strict=True)
    model.to(dtype)
    # keep layernorm in FP32
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(torch.float32)

    torch.save(randn_state_dict, ckpt_path)
    return model


def _get_shared_checkpoint_path(checkpoint_dir: str) -> str:
    """
    Get the path for a shared checkpoint file used by both CPU and Neuron inference.

    This function creates a temporary directory to store model weights that can be
    accessed by both CPU and Neuron inference processes. The directory is created if
    it doesn't exist. The random name in the
    checkpoint file helps avoid potential conflicts when multiple processes are running.

    Args:
        checkpoint_dir (str): Directory path where checkpoint files will be stored

    Returns:
        str: The full path to the checkpoint file (e.g., '/tmp/nxd_inference/ckpt_a1b2c3a1.pt')
    """
    random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return f"{checkpoint_dir}/ckpt_{random_id}.pt"
