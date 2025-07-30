import logging
import os
import pickle
import numpy as np
import torch

from collections import defaultdict
from enum import Enum
from typing import Dict, List

from neuronx_distributed.trace.hlo_utils import get_input_order, get_wlt_map, read_hlo, read_metaneff
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
    app_model,
    ranks: List[int] = [0],
    save_transposed_priority_model_inputs: bool = False,
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
        app_model: The NeuronApplicationBase model.
        ranks: The list of ranks to snapshot. Each rank is a separate NeuronCore device.
            Defauls to [0], which means to capture the snapshot for the rank0 device.
        save_transposed_priority_model_inputs: Whether to save the transposed inputs for the
            priority model, which means to apply the priority model's transposed layout to its
            own inputs. When this is enabled, the snapshot includes two copies of the priority
            model inputs: one default (which matches the HLO), and one transposed (which
            matches the NEFF). The transposed inputs are saved in a subfolder named "transposed_inputs".
    """
    submodel_bucket_request_counts: Dict[str, Dict[int, int]] = {}

    def snapshot_hook(traced_model, args, output):
        """
        Capture arguments, states, and weights.
        """
        model_name, bucket_idx = traced_model.nxd_model.router(args)
        if model_name not in submodel_bucket_request_counts:
            submodel_bucket_request_counts[model_name] = defaultdict(int)
        request_idx = submodel_bucket_request_counts[model_name][bucket_idx]
        logger.debug(f"Called snapshot hook for {model_name=}, {bucket_idx=}, count={request_idx}")
        submodel_bucket_request_counts[model_name][bucket_idx] += 1
        if request_idx not in capture_at_requests:
            return

        is_priority_model = _is_priority_model(app_model, model_name, bucket_idx)
        all_rank_tensors = _get_all_input_tensors(
            app_model,
            traced_model,
            model_name,
            bucket_idx,
            args,
            ranks,
            apply_wlt=not is_priority_model,
        )
        for rank, rank_tensors in enumerate(all_rank_tensors):
            base_path = os.path.join(output_path, model_name, f"_tp0_bk{bucket_idx}", f"request{request_idx}")
            _save_tensors(rank_tensors, base_path, output_format, rank)
        logger.info(f"Saved input snapshot to {base_path}")

        if is_priority_model and save_transposed_priority_model_inputs:
            # Save an extra copy of the priority model inputs with layout optimization applied.
            all_rank_tensors = _get_all_input_tensors(
                app_model,
                traced_model,
                model_name,
                bucket_idx,
                args,
                ranks,
                apply_wlt=True,
            )
            for rank, rank_tensors in enumerate(all_rank_tensors):
                base_path = os.path.join(
                    output_path,
                    model_name,
                    f"_tp0_bk{bucket_idx}",
                    f"request{request_idx}",
                    "transposed_inputs",
                )
                _save_tensors(rank_tensors, base_path, output_format, rank)
            logger.info(f"Saved optimized priority model input snapshot to {base_path}")

    return snapshot_hook


def _get_all_input_tensors(app_model, traced_model, model_name, bucket_idx, input_args, ranks, apply_wlt):
    all_rank_tensors = []
    flattener = getattr(traced_model.nxd_model.flattener_map, model_name)
    input_tensors = flattener(input_args)
    for rank in ranks:
        state_tensors = [state.to("cpu") for state in traced_model.nxd_model.state[rank].values()]
        weights_dict = {key: weights.to("cpu") for key, weights in traced_model.nxd_model.weights[rank].items()}
        weights_tensors = _get_weights_tensors(app_model, weights_dict, apply_wlt, model_name, bucket_idx)
        rank_tensors = input_tensors + state_tensors + weights_tensors
        all_rank_tensors.append(rank_tensors)
    return all_rank_tensors


def _get_weights_tensors(app_model, rank_weights, apply_wlt, model_name, bucket_idx):
    # The model weights need to be transformed to match the compiled model inputs.
    # This process requires information from the artifacts in the compiler workdir,
    # which means that the compiler workdir must be present to capture input snapshots.
    # TODO: Update NxDModel to include info necessary to transform inputs on CPU
    #       so snapshot doesn't depend on compiler workdir being present.
    model_builder = app_model.get_builder()
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

    if apply_wlt:
        layout_opt_hlo_path = os.path.join(layout_opt_path, "model/graph.hlo")
        layout_opt_metaneff_path = os.path.join(layout_opt_path, "metaneff")
        wlt_metaneff = read_metaneff(layout_opt_metaneff_path)
        checkpoint_keys, _ = get_input_order(wlt_metaneff)
        _apply_weight_layout_transformation(rank_weights, layout_opt_hlo_path, checkpoint_keys)

    # Return weight tensors in the correct order.
    return [rank_weights[key] for key in weight_input_keys]


def _is_priority_model(app_model, model_name, bucket_idx):
    for model in app_model.models:
        if model.tag == model_name and bucket_idx == model.priority_model_idx:
            return True
    return False


def _apply_weight_layout_transformation(checkpoint, layout_opt_hlo_path, checkpoint_keys):
    wlt_hlo = read_hlo(layout_opt_hlo_path)
    wlt_map = get_wlt_map(wlt_hlo)

    for idx, key in enumerate(checkpoint_keys):
        if idx in wlt_map:
            transform = wlt_map[idx]
            prev_shape = checkpoint[key].shape
            checkpoint[key] = transform(checkpoint[key])
            logger.debug(f"Transformed {key} from {prev_shape} to {checkpoint[key].shape}")


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
