"""
Utilities for capturing and analyzing tensors during model execution.

This utility provides functions for capturing intermediate tensors from models
during inference, which is useful for debugging, and accuracy analysis
"""

import json
import logging
import os
import time
from functools import partial
from typing import List, Optional

import torch
from torch_neuronx.testing import neuron_allclose

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.image_to_text_model_base import NeuronBaseForImageToText


class TensorCaptureMetadata:
    """Manages metadata for captured tensors in a single JSON file."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.metadata_file = os.path.join(save_dir, "capture_metadata.json")
        self.metadata = {
            "capture_session": {
                "created_at": time.time(),
                "version": "1.0"
            },
            "tensors": {}
        }
        self._load_existing()

    def _load_existing(self):
        """Load existing metadata if file exists."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    existing_metadata = json.load(f)
                    if "tensors" in existing_metadata:
                        self.metadata["tensors"].update(existing_metadata["tensors"])
                    if "capture_session" in existing_metadata:
                        self.metadata["capture_session"] = existing_metadata["capture_session"]
            except Exception as e:
                logging.warning(f"Failed to load existing metadata: {e}")

    def add_tensor(self, filename: str, step: int, phase: str, tensor_type: str,
                   tensor_shape: List[int], tensor_dtype: str, module_name: Optional[str] = None,
                   tensor_index: Optional[int] = None):
        """Add tensor metadata."""
        tensor_metadata = {
            "step": step,
            "phase": phase,
            "tensor_type": tensor_type,
            "tensor_shape": tensor_shape,
            "tensor_dtype": tensor_dtype,
            "timestamp": time.time(),
        }

        if module_name is not None:
            tensor_metadata["module_name"] = module_name
        if tensor_index is not None:
            tensor_metadata["tensor_index"] = tensor_index

        self.metadata["tensors"][filename] = tensor_metadata
        self._save()

    def _save(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")


def _get_tensor_capture_config(application_model):
    """Get tensor capture config from application model."""
    if hasattr(application_model, 'text_config') and isinstance(application_model, NeuronBaseForImageToText):
        return application_model.text_config.neuron_config.tensor_capture_config
    return application_model.neuron_config.tensor_capture_config


def _save_tensor(tensor: torch.Tensor, file_path: str, metadata_manager: TensorCaptureMetadata,
                 step: int, phase: str, tensor_type: str, module_name: Optional[str] = None,
                 tensor_index: Optional[int] = None):
    """Save a single tensor with metadata."""
    # Save tensor
    cpu_tensor = tensor.detach().cpu()
    torch.save(cpu_tensor, file_path)

    # Add metadata
    filename = os.path.basename(file_path)
    metadata_manager.add_tensor(
        filename=filename,
        step=step,
        phase=phase,
        tensor_type=tensor_type,
        tensor_shape=list(tensor.shape),
        tensor_dtype=str(tensor.dtype),
        module_name=module_name,
        tensor_index=tensor_index,
    )

    # Log
    if module_name:
        logging.info(f"Saved {module_name} {tensor_type} tensor to {os.path.abspath(file_path)}")
    else:
        logging.info(f"Saved manual tensor {tensor_index} to {os.path.abspath(file_path)}")


def capture_model_tensors(
    application_model: NeuronApplicationBase,
    captured_tensors: List[torch.Tensor],
    capture_indices: Optional[List[int]] = None,
    tensor_capture_save_dir: Optional[str] = None,
):
    """
    Saves captured intermediate tensors at the specified capture indices.
    If capture_indices is None, saves tensors at all steps.

    Args:
        application_model: A NeuronApplicationBase model (e.g., NeuronLlamaForCausalLM)
        captured_tensors: List of captured tensors from tensor capture
        capture_indices: List of generation step indices to capture tensors at, or None to capture all steps
        tensor_capture_save_dir: Directory to save captured tensors
    """
    tensor_capture_config = _get_tensor_capture_config(application_model)

    # Check if tensor capture is enabled
    if not tensor_capture_config:
        logging.debug("Tensor capture is not enabled in the model configuration")
        return

    if not captured_tensors:
        logging.debug("No captured tensors available to save")
        return

    # Initialize or increment step counter
    if not hasattr(application_model, '_tensor_capture_step'):
        application_model._tensor_capture_step = 1
    else:
        application_model._tensor_capture_step += 1

    current_step = application_model._tensor_capture_step

    # Skip if this step is not in capture_indices (when specified)
    if capture_indices is not None and current_step not in capture_indices:
        return

    # Create tensor capture directory and initialize metadata manager
    if tensor_capture_save_dir is not None:
        os.makedirs(tensor_capture_save_dir, exist_ok=True)
    else:
        tensor_capture_save_dir = ""

    metadata_manager = TensorCaptureMetadata(tensor_capture_save_dir)
    model_phase = "cte" if (current_step == 1) else "tkg"

    # Process module-level tensors first
    capture_index = 0
    if tensor_capture_config.modules_to_capture:
        for module_name in tensor_capture_config.modules_to_capture:
            # Save module output tensor
            filename = f"captured_tensors_{model_phase}_step_{current_step}_module_{module_name}_output.pt"
            file_path = os.path.join(tensor_capture_save_dir, filename)
            _save_tensor(
                captured_tensors[capture_index], file_path, metadata_manager,
                current_step, model_phase, "output", module_name=module_name
            )
            capture_index += 1

            # Save module input tensor if capture_inputs is enabled
            if tensor_capture_config.capture_inputs and capture_index < len(captured_tensors):
                filename = f"captured_tensors_{model_phase}_step_{current_step}_module_{module_name}_input.pt"
                file_path = os.path.join(tensor_capture_save_dir, filename)
                _save_tensor(
                    captured_tensors[capture_index], file_path, metadata_manager,
                    current_step, model_phase, "input", module_name=module_name
                )
                capture_index += 1

    # Process manually registered tensors, filtering out padding tensors
    manual_capture_index = 0
    padding_count = 0
    while capture_index < len(captured_tensors):
        tensor = captured_tensors[capture_index]

        # Skip padding tensors (created with nan value)
        if tensor.numel() == 1 and torch.isnan(tensor).item():
            padding_count += 1
            capture_index += 1
            continue

        filename = f"captured_tensors_{model_phase}_step_{current_step}_manual_tensor_{manual_capture_index}.pt"
        file_path = os.path.join(tensor_capture_save_dir, filename)
        _save_tensor(
            tensor, file_path, metadata_manager, current_step, model_phase,
            "manual", tensor_index=manual_capture_index
        )
        capture_index += 1
        manual_capture_index += 1

    # Log the number of padding tensors found
    if padding_count > 0:
        logging.info(f"Skipped {padding_count} padding tensors (NaN values) in step {current_step}")


def get_tensor_capture_hook(capture_indices: Optional[List[int]] = None, tensor_capture_save_dir="captured_tensors"):
    """
    Creates a tensor capture hook that saves captured intermediate tensors to disk.

    Args:
        capture_indices: Optional list of generation step indices to capture tensors at.
                        If None, captures tensors at all steps.
        tensor_capture_save_dir: Directory to save captured tensors

    Returns:
        Partial function that can be used as a tensor capture hook
    """
    return partial(
        capture_model_tensors,
        capture_indices=capture_indices,
        tensor_capture_save_dir=tensor_capture_save_dir,
    )


def list_capturable_modules_in_application(application_model):
    """
    Lists all modules in a NeuronApplicationBase model that can be captured by tensor_capture_hook.

    This function examines each model wrapper in the application and returns a dictionary
    of all modules that can be captured. This is useful for discovering the correct module
    names to use with the tensor_capture feature.

    **IMPORTANT**: This function initializes and destroys parallel state during execution.
    It should be called BEFORE setting up your main parallel context for training/inference.
    If you have an existing parallel context, it will be destroyed and not restored.

    **Recommended usage**:
    ```python
    # Call this first, before initializing your parallel context
    modules = list_capturable_modules_in_application(model)
    print("Available modules:", modules)

    # Then set up your actual parallel context for training/inference
    torch.distributed.init_process_group(...)
    nxd.parallel_layers.initialize_model_parallel(...)
    ```

    Args:
        application_model: A NeuronApplicationBase model (e.g., NeuronLlamaForCausalLM)

    Returns:
        dict: A dictionary where keys are model names and values are module types

    Raises:
        ValueError: If the application model doesn't have models or has no model wrappers
        RuntimeError: If there's an error examining any model wrapper
    """
    from neuronx_distributed.utils.tensor_capture import get_available_modules
    from neuronx_distributed.parallel_layers import parallel_state
    import torch

    all_modules = {}

    if not hasattr(application_model, 'models'):
        raise ValueError("Application model does not have 'models' attribute. Cannot list capturable modules.")

    if len(application_model.models) == 0:
        raise ValueError("Application model has no model wrappers. Cannot list capturable modules.")

    # Warn if parallel state is already initialized
    if parallel_state.model_parallel_is_initialized() or torch.distributed.is_initialized():
        logging.warning(
            "Parallel state is already initialized. This function will destroy the existing "
            "parallel context and not restore it. Consider calling this function before "
            "setting up your main parallel context."
        )

    # Iterate through each model wrapper in the application
    for model_wrapper in application_model.models:
        model_name = model_wrapper.tag
        logging.info(f"Examining {model_name}:")

        try:
            # Save current parallel state
            was_aot_mode = parallel_state.get_aot_mode() if parallel_state.model_parallel_is_initialized() else False

            # Set up parallel state for this model
            if parallel_state.model_parallel_is_initialized():
                parallel_state.set_aot_mode(False)
                parallel_state.destroy_model_parallel()

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            # Initialize process group
            torch.distributed.init_process_group('xla', rank=0, world_size=model_wrapper.config.neuron_config.world_size)

            # Initialize model parallel
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=model_wrapper.config.neuron_config.tp_degree,
                pipeline_model_parallel_size=model_wrapper.config.neuron_config.pp_degree,
                expert_model_parallel_size=model_wrapper.config.neuron_config.ep_degree,
                skip_collective_init=True,
                lnc_size=model_wrapper.config.neuron_config.logical_nc_config
            )
            parallel_state.set_aot_mode(True)

            # Get model instance
            model_instance = model_wrapper.get_model_instance()
            model = model_instance.model_cls(model_instance.config, **model_instance.kwargs)

            # Get available modules
            all_modules[model_name] = get_available_modules(model)

            # Clean up
            parallel_state.set_aot_mode(False)
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

            # Restore original parallel state if needed
            if was_aot_mode:
                parallel_state.set_aot_mode(True)

        except Exception as e:
            raise RuntimeError(f"Error examining {model_name}: {e}")

    return all_modules


def analyze_captured_tensors(tensor_dir: str, reference_dir: Optional[str] = None):
    """
    Analyzes captured tensors, comparing them with reference tensors if provided.

    Args:
        tensor_dir: Directory containing captured tensors
        reference_dir: Optional directory containing reference tensors for comparison

    Returns:
        Dict containing analysis results
    """
    results = {
        "tensors": [],
        "summary": {
            "total_tensors": 0,
            "modules_captured": set(),
            "steps_captured": set(),
            "phases": set()
        }
    }

    if not os.path.exists(tensor_dir):
        logging.error(f"Tensor directory {tensor_dir} does not exist")
        return results

    # Load metadata from JSON file
    metadata_file = os.path.join(tensor_dir, "capture_metadata.json")
    if not os.path.exists(metadata_file):
        logging.error(f"Metadata file {metadata_file} not found")
        return results

    try:
        with open(metadata_file, "r") as f:
            metadata_json = json.load(f)
            metadata_dict = metadata_json.get("tensors", {})
    except Exception as e:
        logging.error(f"Failed to load metadata file: {e}")
        return results

    tensor_files = [f for f in os.listdir(tensor_dir) if f.startswith("captured_tensors_") and f.endswith(".pt")]
    results["summary"]["total_tensors"] = len(tensor_files)

    for tensor_file in tensor_files:
        if tensor_file not in metadata_dict:
            logging.warning(f"No metadata found for {tensor_file}, skipping")
            continue

        metadata = metadata_dict[tensor_file]
        phase = metadata["phase"]
        step = metadata["step"]
        module_name = metadata.get("module_name")

        results["summary"]["steps_captured"].add(step)
        results["summary"]["phases"].add(phase)
        if module_name:
            results["summary"]["modules_captured"].add(module_name)

        # Load tensor and compute statistics
        tensor_path = os.path.join(tensor_dir, tensor_file)
        try:
            tensor = torch.load(tensor_path)
            tensor_info = {
                "name": tensor_file,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "phase": phase,
                "step": step
            }

            # Compare with reference if available
            if reference_dir and os.path.exists(os.path.join(reference_dir, tensor_file)):
                ref_tensor = torch.load(os.path.join(reference_dir, tensor_file))
                if tensor.shape == ref_tensor.shape:
                    # Use neuron_allclose for comparison
                    tensor_info["ref_comparison"] = neuron_allclose(tensor, ref_tensor)

            results["tensors"].append(tensor_info)

        except Exception as e:
            logging.error(f"Error analyzing tensor {tensor_file}: {e}")

    # Convert sets to lists for JSON serialization
    results["summary"]["modules_captured"] = list(results["summary"]["modules_captured"])
    results["summary"]["steps_captured"] = list(results["summary"]["steps_captured"])
    results["summary"]["phases"] = list(results["summary"]["phases"])

    return results
