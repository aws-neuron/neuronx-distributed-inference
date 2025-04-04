import logging
import os
from functools import partial
from typing import List, Optional

import torch

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM


def capture_model_inputs(
    self: NeuronBaseForCausalLM,
    inputs: List,
    capture_indices: List[int],
    input_capture_save_dir: Optional[str] = None,
):
    """
    Saves model inputs at the specified capture indices. If 0 is in capture indices, CTE input is captured, index 1 would
    indicate saving the 1st TKG pass inputs. One CTE input tuple will be saved, and two TKG input tuples are saved, with
    KV cache and without. If specified, inputs for each index are saved in input_capture_save_dir, otherwise they are
    saved to the current directory.
    """

    if capture_indices is None:
        raise ValueError("capture_model_inputs was called, but no capture_indices were specified")

    capture_indices = [self.initial_input_size + i - 1 for i in capture_indices]

    position_ids = inputs[2]
    current_index = torch.max(position_ids)

    if current_index not in capture_indices:
        return

    if input_capture_save_dir is not None and not os.path.exists(input_capture_save_dir):
        logging.debug(f"{input_capture_save_dir} directory not found, making directory")
        os.makedirs(input_capture_save_dir)

    input_capture_save_dir = input_capture_save_dir if input_capture_save_dir is not None else ""

    # If on_device_sampling is not enabled, we can ignore the sampling parameters input
    if not self.on_device_sampling:
        inputs = inputs[:-1]

    inputs = tuple(inputs)

    pass_index = current_index - (self.initial_input_size - 1)

    if not self.kv_cache_populated:
        logging.debug(f"capturing CTE inputs during pass {pass_index}")
        inputs = self.context_encoding_model.convert_int64_to_int32(*inputs)
        inputs = self.context_encoding_model.pad_inputs(*inputs, pad_type="first_fit")

        cte_save_path = f"saved_inputs_cte_output_{pass_index}.pt"
        torch.save(inputs, os.path.join(input_capture_save_dir, cte_save_path))
        logging.info(f"saved CTE inputs to {cte_save_path}")

        return

    logging.debug(f"capturing TKG inputs during pass {pass_index}")

    generation_model = self.get_generation_model()
    inputs = generation_model.convert_int64_to_int32(*inputs)
    inputs = generation_model.pad_inputs(*inputs, pad_type="first_fit")

    inputs = list(inputs)

    neuron_cache = self.context_encoding_model.model.nxd_model.state
    cpu_cache = []

    for i in range(0, len(neuron_cache)):
        cpu_cache += [neuron_cache[i][key].to("cpu") for key in neuron_cache[i].keys()]

    tkg_save_path_no_kv_cache = f"saved_inputs_tkg_without_kv_cache_output_{pass_index}.pt"
    tkg_save_path_kv_cache = f"saved_inputs_tkg_with_kv_cache_output_{pass_index}.pt"

    torch.save(
        tuple(inputs),
        os.path.join(input_capture_save_dir, tkg_save_path_no_kv_cache),
    )
    logging.info(
        f"saved TKG inputs without kv cache to {os.path.abspath(tkg_save_path_no_kv_cache)}"
    )

    torch.save(
        tuple(inputs + cpu_cache),
        os.path.join(input_capture_save_dir, tkg_save_path_kv_cache),
    )
    logging.info(f"saved TKG inputs with kv cache to {os.path.abspath(tkg_save_path_kv_cache)}")


def get_input_capture_hook(capture_indices: List[int], input_capture_save_dir="saved_inputs"):
    return partial(
        capture_model_inputs,
        capture_indices=capture_indices,
        input_capture_save_dir=input_capture_save_dir,
    )
