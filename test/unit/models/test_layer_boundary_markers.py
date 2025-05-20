import unittest
from pathlib import Path

import pytest
import torch
import torch_neuronx
from torch_neuronx.xla_impl.trace import (
    generate_hlo,
    get_hlo_computation_by_id,
    hlo_entry_computation,
)

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


def get_opcodes_with_dereferenced_calls(hlo, computation):
    """
    Flatten computation into a single list of opcodes.
    If the operation is a custom-call, insert the custom-call-target instead of the opcode.
    """
    ops = []
    for instruction in computation.instructions:
        if len(instruction.called_computation_ids) == 0:
            if instruction.opcode == "custom-call":
                ops.append(instruction.custom_call_target)
            else:
                ops.append(instruction.opcode)
        else:
            for called_computation_id in instruction.called_computation_ids:
                ops.extend(
                    get_opcodes_with_dereferenced_calls(
                        hlo, get_hlo_computation_by_id(hlo, called_computation_id)
                    )
                )
    return ops


def assert_relative_ordering(actual: list, ordering: list):
    """
    Check that:
     1. all elements of ordering appear in actual
     2. they appear in order
    Otherwise, raise ValueError
    """
    prev_idx = 0
    for operation in ordering:
        prev_idx = actual.index(operation, prev_idx)


def assert_hlo_has_relative_ordering(f, example_inputs, expected_ordering):
    hlo = generate_hlo(f, example_inputs).hlo_module
    opcodes = get_opcodes_with_dereferenced_calls(hlo, hlo_entry_computation(hlo))
    assert_relative_ordering(opcodes, expected_ordering)


class TestLayerBoundaryMarkers(unittest.TestCase):
    def test_add_is_contained(self):
        def f(input: torch.Tensor):
            input = ModuleMarkerStartWrapper()(input)
            input = input + input
            input = ModuleMarkerEndWrapper()(input)
            return input

        expected_ordering = [
            "AwsNeuronModuleMarkerStart-Forward",
            "add",
            "AwsNeuronModuleMarkerEnd-Forward",
        ]
        example_input = torch.arange(16).reshape((4, 4))
        assert_hlo_has_relative_ordering(f, (example_input,), expected_ordering)

    def test_add_is_not_contained(self):
        def f(input: torch.Tensor):
            input = input + input
            input = ModuleMarkerStartWrapper()(input)
            input = input @ input
            input = ModuleMarkerEndWrapper()(input)
            input = input + input
            return input

        expected_ordering = [
            "add",
            "AwsNeuronModuleMarkerStart-Forward",
            "dot",
            "AwsNeuronModuleMarkerEnd-Forward",
            "add",
        ]
        example_input = torch.arange(16).reshape((4, 4))
        assert_hlo_has_relative_ordering(f, (example_input,), expected_ordering)

    def test_multiple_layers(self):
        def f(input: torch.Tensor):
            input = input + input
            input = ModuleMarkerStartWrapper()(input)
            input = input @ input
            input = ModuleMarkerEndWrapper()(input)
            input = input + input
            input = ModuleMarkerStartWrapper()(input)
            input = input @ input
            input = ModuleMarkerEndWrapper()(input)
            input = input + input
            return input

        expected_ordering = [
            "add",
            "AwsNeuronModuleMarkerStart-Forward",
            "dot",
            "AwsNeuronModuleMarkerEnd-Forward",
            "add",
            "AwsNeuronModuleMarkerStart-Forward",
            "dot",
            "AwsNeuronModuleMarkerEnd-Forward",
            "add",
        ]
        example_input = torch.arange(16).reshape((4, 4))
        assert_hlo_has_relative_ordering(f, (example_input,), expected_ordering)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
