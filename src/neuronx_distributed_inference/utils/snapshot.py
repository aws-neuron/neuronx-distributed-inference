import logging
import os
import pickle
import numpy as np
import torch

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.hlo_utils import read_metaneff
from torch_neuronx.proto import metaneff_pb2


logger = logging.getLogger("Neuron")


class ScriptModuleWrapper(torch.nn.Module):
    """
    Wraps a torch.jit.ScriptModule to capture inputs/outputs.

    This class is useful for adding hooks to ScriptModules, which don't support hooks.
    """
    def __init__(self, module: torch.jit.ScriptModule):
        super().__init__()
        self.wrapped_module = module

    def forward(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)

    @property
    def __class__(self):
        # Enable the wrapper to appear to be a ScriptModule if checked with isinstance.
        return torch.jit.ScriptModule

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_module, name)

    def __setattr__(self, name, value):
        try:
            return super().__setattr__(name, value)
        except AttributeError:
            return setattr(self.wrapped_module, name, value)

    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.wrapped_module, name)

    def __repr__(self):
        return f'ScriptModuleWrapper({self.wrapped_module.__repr__()})'


class SnapshotOutputFormat(Enum):
    """
    Defines the output format for snapshots.
    """

    NUMPY_IMAGES = 0,
    """
    Saves the input tensors as numpy arrays, where each input tensor is a separate file, where the
    filename includes the input index. Each rank's input is stored in a separate folder, where the
    folder name includes the rank index.

    For example, a rank-0 snapshot with three input tensors will produce the following files:
    * rank0/input0.npy
    * rank0/input1.npy
    * rank0/input2.npy
    """

    NUMPY_PICKLE = 1,
    """
    Saves the input tensors as numpy arrays in a pickle object, where each input tensors is a value
    in a dict. The dict keys include each input's index. Each rank's input is saved to a separate
    pickle file, where the filename includes the rank index.

    For example, a rank-0 snapshot with three input tensors will produce a file named "inp-000.p",
    and the dict contains keys "input0", "input1", and "input2".
    """


def get_snapshot_hook(
    output_path: str,
    output_format: SnapshotOutputFormat,
    capture_at_requests: List[int],
    model_builder: ModelBuilder,
    ranks: Optional[List[int]] = None,
    is_input_ranked: bool = False,
):
    """
    Creates a forward hook that saves input snapshots.
    These input snapshots are used to provide repro artifacts for compiler/runtime.

    Input snapshots are saved to the output path in the following formats:
    1. NUMPY_IMAGES format
      `{output_path}/{submodel}/_tp0_bk{bucket_idx}/request{request_idx}/rank{rank}/input{idx}.npy`
    2. NUMPY_PICKLE format
      `{output_path}/{submodel}/_tp0_bk{bucket_idx}/request{request_idx}/{inp}-{rank}.pt`

    Args:
        output_path: The base path where input snapshots are saved.
        output_format: The output format to use.
            NUMPY_IMAGES: Save each tensor as a separate .npy file.
            NUMPY_PICKLE: Save tensors in .npy format in a pickle object file.
        capture_at_requests: The request numbers at which this hook captures input snapshots for
            each submodel bucket. For example, [0] means to capture the first request to each
            submodel bucket.
        model_builder: The ModelBuilder instance used to compile the model.
        ranks: The list of ranks to snapshot. Each rank is a separate NeuronCore device.
            Defauls to [0], which means to capture the snapshot for the rank0 device.
        is_input_ranked: Whether the first input arg is a list of ranked inputs. Set this to true
            when you create a snapshot hook for a model that uses async or pipeline execution.
            These execution modes use inputs that are on-device. To capture them, the hook moves the
            inputs to CPU.
    """
    if ranks is None:
        ranks = [0]

    submodel_bucket_request_counts: Dict[str, Dict[int, int]] = {}

    def snapshot_hook(traced_model, args, output):
        """
        Capture arguments, states, and weights.
        """
        if is_input_ranked:
            # When input is ranked, the first arg contains the ranked input, which is a input list
            # where each index is a rank. Therefore, args[0][0] retrieves the first rank's input.
            # TODO: Add support to capture all ranks.
            assert ranks == [0], "Ranked input snapshots only supports rank=0 currently"
            args = args[0][0]

        model_name, bucket_idx = traced_model.nxd_model.router(args)
        if model_name not in submodel_bucket_request_counts:
            submodel_bucket_request_counts[model_name] = defaultdict(int)
        request_idx = submodel_bucket_request_counts[model_name][bucket_idx]
        logger.debug(f"Called snapshot hook for {model_name=}, {bucket_idx=}, count={request_idx}")
        submodel_bucket_request_counts[model_name][bucket_idx] += 1
        if request_idx not in capture_at_requests:
            return

        all_rank_tensors = _get_all_input_tensors(
            model_builder,
            traced_model,
            model_name,
            bucket_idx,
            args,
            ranks,
        )
        for rank, rank_tensors in enumerate(all_rank_tensors):
            base_path = os.path.join(output_path, model_name, f"_tp0_bk{bucket_idx}", f"request{request_idx}")
            _save_tensors(rank_tensors, base_path, output_format, rank)
        logger.info(f"Saved input snapshot to {base_path}")

    return snapshot_hook


def _get_all_input_tensors(model_builder, traced_model, model_name, bucket_idx, input_args, ranks):
    all_rank_tensors = []
    flattener = getattr(traced_model.nxd_model.flattener_map, model_name)
    input_tensors = [input.to("cpu") for input in flattener(input_args)]
    for rank in ranks:
        state_tensors = [state.to("cpu") for state in traced_model.nxd_model.state[rank].values()]
        weights_dict = {key: weights.to("cpu") for key, weights in traced_model.nxd_model.weights[rank].items()}
        weights_tensors = _get_weights_tensors(model_builder, weights_dict, model_name, bucket_idx)
        rank_tensors = input_tensors + state_tensors + weights_tensors

        # Filter out empty tensors.
        rank_tensors = [tensor for tensor in rank_tensors if tensor.shape != ()]
        all_rank_tensors.append(rank_tensors)
    return all_rank_tensors


def _get_weights_tensors(model_builder, rank_weights, model_name, bucket_idx):
    # The model weights need to be filtered/reordered to match the compiled model inputs.
    # This process requires information from the artifacts in the compiler workdir,
    # which means that the compiler workdir must be present to capture input snapshots.
    # TODO: Update NxDModel to include info necessary to filter/reorder inputs on CPU
    #       so snapshot doesn't depend on compiler workdir being present.
    assert os.path.exists(model_builder.compiler_workdir), (
        "Unable to find compiler workdir. "
        "To create weights for a snapshot, the model's compiler workdir must be available."
    )
    layout_opt_path = os.path.join(model_builder.compiler_workdir, "layout_opt")
    assert os.path.exists(layout_opt_path), f"Unable to find layout_opt model: {layout_opt_path}"

    # Find weight tensor input order from model metaneff.
    submodel_compiler_workdir = os.path.join(model_builder.compiler_workdir, model_name, f"_tp0_bk{bucket_idx}")
    metaneff_path = os.path.join(submodel_compiler_workdir, "metaneff.pb")
    assert os.path.exists(metaneff_path), f"Unable to find metaneff: {metaneff_path}"
    metaneff = read_metaneff(metaneff_path)
    weight_input_keys = [
        input.checkpoint_key.decode() for input in metaneff.input_tensors
        if input.type == metaneff_pb2.MetaTensor.Type.INPUT_WEIGHT
    ]

    # Return weight tensors in the correct order.
    return [rank_weights[key] for key in weight_input_keys]


def _save_tensors(tensors, base_path, output_format, rank):
    os.makedirs(base_path, exist_ok=True)
    npy_tensors = [_to_numpy(tensor) for tensor in tensors]
    if output_format == SnapshotOutputFormat.NUMPY_IMAGES:
        for i, npy_tensor in enumerate(npy_tensors):
            rank_path = os.path.join(base_path, f"rank{rank}")
            os.makedirs(rank_path, exist_ok=True)
            output_path = os.path.join(rank_path, f"input{i}.npy")
            np.save(output_path, npy_tensor)
    elif output_format == SnapshotOutputFormat.NUMPY_PICKLE:
        npy_tensor_map = {f"input{i}": npy_tensor for i, npy_tensor in enumerate(npy_tensors)}
        output_path = os.path.join(base_path, f"inp-{rank:{0}{3}}.p")
        _dump_pickle(output_path, npy_tensor_map)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _to_numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
        np_tensor = tensor.numpy()
        np_tensor = np_tensor.view("|V2")
    elif tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.int8)
        np_tensor = tensor.numpy()
        np_tensor = np_tensor.view("|V1")
    else:
        np_tensor = tensor.numpy()
    return np_tensor


def _dump_pickle(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


_original_func_map: Dict[Any, Dict[str, Callable]] = defaultdict(dict)


def register_nxd_model_hook(traced_model, func_name, hook):
    """
    Registers a hook for a function on the given traced model's NxDModel.

    Args:
        traced_model: The traced model to update.
        func_name: The name of the function to hook into.
        hook: The hook function to add.
    """
    nxd_model = traced_model.nxd_model
    assert hasattr(nxd_model, func_name), f"nxd_model has no function named {func_name}"
    func = getattr(nxd_model, func_name)

    def wrapped_func(*args, **kwargs):
        output = func(*args, **kwargs)
        hook(traced_model, args, output)
        return output

    setattr(nxd_model, func_name, wrapped_func)
    _original_func_map[nxd_model][func_name] = func


def unregister_nxd_model_hooks(traced_model, func_name):
    """
    Unegisters hooks for a function on the given traced model's NxDModel.

    Args:
        traced_model: The traced model to update.
        func_name: The name of the function to restore.
    """
    nxd_model = traced_model.nxd_model
    assert hasattr(nxd_model, func_name), f"nxd_model has no function named {func_name}"
    if nxd_model in _original_func_map and func_name in _original_func_map[nxd_model]:
        setattr(nxd_model, func_name, _original_func_map[nxd_model][func_name])
        del _original_func_map[nxd_model][func_name]
