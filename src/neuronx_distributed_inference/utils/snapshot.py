import logging
import os
import pickle
import numpy as np
import torch
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set

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


class SnapshotCaptureConfig:
    """Configuration for model input snapshot capturing.


    This class configures when to capture input snapshots for LLM executions.
    Snapshots can be captured based on specific requests (executions of executables)
    or specific tokens being generated.

    Args:
        max_tokens_generated_per_request: The maximum number of tokens generated for a particular decode loop.
            This is usually 1 (and is the default), but can be higher if using speculative decoding.

    Examples:
        # Capture at the first execution of each executable
        config = SnapshotCaptureConfig().capture_at_request(0)

        # Capture inputs generating token 244
        config = SnapshotCaptureConfig().capture_for_token(token_indices=244)

        # Capture at every request
        config = SnapshotCaptureConfig().capture_at_request(-1)

        # Capture inputs generating multiple specific tokens
        config = SnapshotCaptureConfig().capture_for_token(token_indices=[244, 245, 246])
    """

    def __init__(self, max_tokens_generated_per_request: int = 1):
        """Initialize the configuration.
        """
        self.request_indices: Set[int] = set()
        self.token_indices: Set[Tuple[int, int]] = set()
        self.capture_all_requests: bool = False
        self.capture_all_tokens: bool = False
        self._capture_types: Set[str] = set()  # Track which types of captures are configured
        self.max_tokens_generated_per_request = max_tokens_generated_per_request

    def capture_at_request(self, request_indices: Union[int, List[int]]) -> 'SnapshotCaptureConfig':
        """Add request indices to capture.

        Args:
            request_indices: The request indices to capture. Can be a single index or a list of indices.
                           If -1 is provided, all requests will be captured.

        Returns:
            Self for method chaining.
        """
        if isinstance(request_indices, list) and len(request_indices) == 0:
            return self

        if isinstance(request_indices, int):
            assert request_indices >= 0 or request_indices == -1, f"Request indices must be >= 0 or -1 for capturing all requests, but was provided {request_indices=}"
            request_indices = [request_indices]

        self._capture_types.add('request')
        for idx in request_indices:
            if idx == -1:
                self.capture_all_requests = True
                return self
            else:
                self.request_indices.add(idx)

        return self

    def capture_for_token(self, token_indices: Union[int, List[int]], batch_line: int = 0) -> 'SnapshotCaptureConfig':
        """Add token indices to capture.

        Args:
            token_indices: The token indices to capture. Can be a single index or a list of indices.
                         If -1 is provided, all tokens will be captured. See class docstring for examples.
            batch_line: The specific batch line to capture snapshots for. Defaults to 0.

        Returns:
            Self for method chaining.
        """
        if isinstance(token_indices, list) and len(token_indices) == 0:
            return self

        if isinstance(token_indices, int):
            assert token_indices >= 0 or token_indices == -1, f"Only valid tokens (>=0) or -1 (all tokens) are supported, but was provided {token_indices=}"
            token_indices = [token_indices]

        self._capture_types.add('token')
        for idx in token_indices:
            if idx == -1:
                self.capture_all_tokens = True
                return self
            else:
                self.token_indices.add((batch_line, idx))

        return self

    def is_capturing_requests(self) -> bool:
        """Check if the config is capturing any requests."""
        return 'request' in self._capture_types

    def is_capturing_tokens(self) -> bool:
        """Check if the config is capturing any tokens."""
        return 'token' in self._capture_types

    def which_token(self, token_indices: List[int]) -> Union[Tuple[int, int], None]:
        """Determine which token and batch line should be captured, if any.

        Args:
            token_indices: List of current token indices (one per batch line).

        Returns:
            Tuple of (batch_line, token_idx) to capture, or None if no capture needed.
        """
        for batch_line, token in enumerate(token_indices):
            for offset in range(1, self.max_tokens_generated_per_request + 1):
                # Check if we should capture the token that will be generated
                target_token = token + offset
                if (batch_line, target_token) in self.token_indices:
                    return batch_line, target_token

        return None

    def should_capture(self, token_indices: List[int], request_idx: int) -> bool:
        """Determine if a snapshot should be captured.

        Args:
            token_indices: List of current token indices (one per batch line).
            request_idx: The index of the request (execution of an executable).

        Returns:
            True if a snapshot should be captured, False otherwise.
        """
        # If no capture conditions have been set, don't capture anything
        if not self._capture_types:
            return False

        # Check request condition
        if 'request' in self._capture_types:
            if self.capture_all_requests or request_idx in self.request_indices:
                return True

        # Check token condition
        if 'token' in self._capture_types:
            # Check each batch line against our token captures
            # or if we should capture all tokens
            if self.capture_all_tokens:
                return True

            return self.which_token(token_indices) is not None

        return False


def get_snapshot_hook(
    output_path: str,
    output_format: SnapshotOutputFormat,
    snapshot_config: SnapshotCaptureConfig,
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

        token_indices = []
        if snapshot_config.is_capturing_tokens():
            # move to cpu regardless of sync/async.
            # In async, input snaphsot capturing is a blocking operation so performance is lost,
            # but scheduling wrt bucketing is maintained, and still useful for debugging
            position_ids = args[2].cpu()  # input_ids, attn_mask, pos_ids, ...
            batch_size, dim1 = position_ids.shape
            if dim1 != 1:  # usually context encoding position ids is [batch_size, seq_len]
                position_ids = torch.max(position_ids, dim=1).values
            token_indices = position_ids.reshape((batch_size,)).tolist()  # convert to list

        model_name, bucket_idx = traced_model.nxd_model.router(args)
        if model_name not in submodel_bucket_request_counts:
            submodel_bucket_request_counts[model_name] = defaultdict(int)
        request_idx = submodel_bucket_request_counts[model_name][bucket_idx]
        logger.debug(f"Called snapshot hook for {model_name=}, {bucket_idx=}, count={request_idx}")
        submodel_bucket_request_counts[model_name][bucket_idx] += 1

        if not snapshot_config.should_capture(token_indices, request_idx):
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
            base_path = os.path.join(output_path, model_name, f"_tp0_bk{bucket_idx}")
            if snapshot_config.is_capturing_requests():
                base_path = os.path.join(base_path, f"request{request_idx}")
            if snapshot_config.is_capturing_tokens():
                token_res = snapshot_config.which_token(token_indices)
                assert token_res is not None
                batch_line, token_captured = token_res
                base_path = os.path.join(base_path, f"batch{batch_line}_token{token_captured}")
            _save_tensors(rank_tensors, base_path, output_format, rank)
        logger.info(f"Saved input snapshot to {base_path}")

    return snapshot_hook


def _get_all_input_tensors(model_builder, traced_model, model_name, bucket_idx, input_args, ranks):
    all_rank_tensors = []
    key = f"{model_name}_{bucket_idx}"
    if hasattr(traced_model.nxd_model.flattener_map, key):
        flattener = getattr(traced_model.nxd_model.flattener_map, key)
    else:
        # forwards compatibility for models traced before nxd commit c71d4f5a
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


def discover_bucket_request_mapping(snapshots_dir, model_name):
    """
    Find the bucket-request mapping from snapshot directories.

    Args:
        snapshots_dir: Path to the snapshots directory
        model_name: Name of the model subdirectory (e.g., "token_generation_model")

    Returns:
        List of (bucket_idx, request_idx) tuples representing the mapping
    """
    bucket_request_map = []
    model_snapshots_dir = snapshots_dir / model_name

    if not model_snapshots_dir.exists():
        raise FileNotFoundError(f"Model snapshots directory not found: {model_snapshots_dir}")

    request_paths = model_snapshots_dir.glob("_tp0_bk*/request*")

    for path_obj in request_paths:
        bucket_name = path_obj.parent.name
        request_name = path_obj.name

        bucket_idx = int(bucket_name.replace('_tp0_bk', ''))
        request_idx = int(request_name.replace('request', ''))

        bucket_request_map.append((bucket_idx, request_idx))

    bucket_request_map.sort()

    return bucket_request_map
