import os
from functools import partial
from unittest.mock import Mock

import pytest
import torch
from neuronx_distributed.operators.topk import topk as nxd_topk
from neuronx_distributed.parallel_layers import ColumnParallelLinear

from neuronx_distributed_inference.utils.testing import (
    build_function,
    build_module,
    validate_accuracy,
)

torch.manual_seed(0)

SAMPLE_SIZE = 4


def example_sum(tensor):
    return torch.sum(tensor)


def example_topk(tensor, k, dim, on_cpu):
    if on_cpu:
        return torch.topk(tensor, k, dim)
    else:
        return nxd_topk(tensor, k, dim, gather_dim=dim)


class ExampleModule(torch.nn.Module):
    def __init__(self, distributed):
        super().__init__()
        if distributed:
            self.linear = ColumnParallelLinear(
                input_size=SAMPLE_SIZE,
                output_size=SAMPLE_SIZE,
                bias=False,
                dtype=torch.float32,
            )
        else:
            self.linear = torch.nn.Linear(
                in_features=SAMPLE_SIZE,
                out_features=SAMPLE_SIZE,
                bias=False,
                dtype=torch.float32,
            )

    def forward(self, x):
        return self.linear(x)


def test_validate_accuracy_basic_function():
    inputs = [(torch.tensor([1, 2, 3], dtype=torch.float32),)]
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]

    neuron_model = build_function(example_sum, example_inputs)
    validate_accuracy(neuron_model, inputs, cpu_callable=example_sum)


def test_validate_accuracy_function_with_expected_outputs():
    inputs = [
        (torch.tensor([1, 2, 3], dtype=torch.float32),),
        (torch.tensor([3, 4, 5], dtype=torch.float32),),
    ]
    expected_outputs = [
        torch.tensor(6, dtype=torch.float32),
        torch.tensor(12, dtype=torch.float32),
    ]
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]

    neuron_model = build_function(example_sum, example_inputs)
    validate_accuracy(
        neuron_model, inputs, expected_outputs=expected_outputs, cpu_callable=example_sum
    )


def test_validate_accuracy_function_with_custom_cpu_func():
    inputs = [
        (torch.tensor((0, 1, 2, 3), dtype=torch.float32),),
    ]
    example_inputs = [
        (torch.zeros((4), dtype=torch.float32),),
    ]

    func = partial(example_topk, k=1, dim=0, on_cpu=False)
    func_cpu = partial(example_topk, k=1, dim=0, on_cpu=True)
    neuron_model = build_function(func, example_inputs)
    validate_accuracy(
        neuron_model, inputs, cpu_callable=func_cpu, assert_close_kwargs={"check_dtype": False}
    )


def test_validate_accuracy_function_with_distributed_func_tp2():
    inputs = [
        (torch.tensor((0, 1, 2, 3), dtype=torch.float32),),
    ]
    expected_outputs = [
        (torch.tensor((3,), dtype=torch.float32), torch.tensor((3,), dtype=torch.int64)),
    ]
    example_inputs = [(torch.zeros((4), dtype=torch.float32),)]

    func = partial(example_topk, k=1, dim=0, on_cpu=False)
    func_cpu = partial(example_topk, k=1, dim=0, on_cpu=True)
    neuron_model = build_function(func, example_inputs, tp_degree=2)
    validate_accuracy(
        neuron_model,
        inputs,
        expected_outputs=expected_outputs,
        cpu_callable=func_cpu,
        assert_close_kwargs={"check_dtype": False},
    )


def test_validate_accuracy_basic_module():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    module_cls = partial(ExampleModule, distributed=False)
    neuron_model = build_module(module_cls, example_inputs)

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)


def test_validate_accuracy_module_with_expected_outputs():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    expected_outputs = [torch.tensor([-1.6587, 1.3036, -0.4648, -0.6878], dtype=torch.float32)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    module_cls = partial(ExampleModule, distributed=False)
    neuron_model = build_module(module_cls, example_inputs)

    validate_accuracy(
        neuron_model,
        inputs,
        expected_outputs=expected_outputs,
        cpu_callable=module_cpu,
        assert_close_kwargs={"rtol": 1e-3, "atol": 1e-3},
    )


def test_validate_accuracy_module_with_custom_cpu_module():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(
        ExampleModule, example_inputs, module_init_kwargs={"distributed": True}
    )

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)


def test_validate_accuracy_module_with_custom_cpu_module_tp2():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(
        ExampleModule, example_inputs, tp_degree=2, module_init_kwargs={"distributed": True}
    )

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)


def test_validate_accuracy_module_with_multiple_inputs():
    inputs = [
        (torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),),
        (torch.arange(SAMPLE_SIZE, SAMPLE_SIZE * 2, dtype=torch.float32),),
    ]
    expected_outputs = [
        torch.tensor([-1.6587, 1.3036, -0.4648, -0.6878], dtype=torch.float32),
        torch.tensor([-3.7188, 2.6158, -1.1106, -4.6734], dtype=torch.float32),
    ]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(
        ExampleModule, example_inputs, module_init_kwargs={"distributed": True}
    )

    validate_accuracy(
        neuron_model,
        inputs,
        expected_outputs,
        cpu_callable=module_cpu,
        assert_close_kwargs={"rtol": 1e-3, "atol": 1e-3},
    )


def test_validate_accuracy_module_with_custom_compiler_workdir_and_checkpoint_path():
    checkpoint_path = "/tmp/nxdi_checkpoint.pt"
    compiler_workdir = "/tmp/nxdi_compiler_workdir"

    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    torch.save(module_cpu.state_dict(), checkpoint_path)

    neuron_model = build_module(
        ExampleModule,
        example_inputs,
        compiler_workdir=compiler_workdir,
        checkpoint_path=checkpoint_path,
        module_init_kwargs={"distributed": True},
    )
    validate_accuracy(
        neuron_model,
        inputs,
        cpu_callable=module_cpu,
        assert_close_kwargs={"rtol": 1e-3, "atol": 1e-3},
    )

    # Verify the custom compiler workdir is used.
    assert len(os.listdir(compiler_workdir)) >= 1


def test_validate_accuracy_no_expected_outputs():
    neuron_model = Mock()
    inputs = [(torch.rand((1), dtype=torch.float32),)]
    with pytest.raises(
        ValueError, match="Provide expected_outputs or a cpu_callable to produce expected outputs"
    ):
        validate_accuracy(neuron_model, inputs)


def test_validate_accuracy_inputs_not_a_list():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = {}
    with pytest.raises(ValueError, match="inputs must be a list of tensor tuples"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_inputs_is_empty_list():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = []
    with pytest.raises(ValueError, match="inputs must not be empty"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_inputs_contains_non_tuple():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = [1]
    with pytest.raises(ValueError, match="inputs must be a list of tensor tuples"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_inputs_contains_tuple_with_non_tensor():
    neuron_model = Mock()
    cpu_callable = Mock()
    inputs = [(1,)]
    with pytest.raises(ValueError, match="inputs must be a list of tensor tuples"):
        validate_accuracy(neuron_model, inputs, cpu_callable=cpu_callable)


def test_validate_accuracy_expected_outputs_not_a_list():
    neuron_model = Mock()
    inputs = [(torch.rand((1), dtype=torch.float32),)]
    expected_outputs = {}
    with pytest.raises(ValueError, match="expected_outputs must be a list"):
        validate_accuracy(neuron_model, inputs, expected_outputs)


def test_validate_accuracy_expected_outputs_len_mismatch():
    neuron_model = Mock()
    inputs = [(torch.rand((1), dtype=torch.float32),)]
    expected_outputs = [
        (torch.rand((1), dtype=torch.float32),),
        (torch.rand((1), dtype=torch.float32),),
    ]
    with pytest.raises(ValueError, match=r"len\(expected_outputs\) must match len\(inputs\)"):
        validate_accuracy(neuron_model, inputs, expected_outputs)


def test_build_module_example_inputs_not_a_list():
    module_cls = Mock()
    example_inputs = {}
    with pytest.raises(ValueError, match="example_inputs must be a list of tensor tuples"):
        build_module(module_cls, example_inputs)


def test_build_module_example_inputs_empty_list():
    module_cls = Mock()
    example_inputs = []
    with pytest.raises(ValueError, match="example_inputs must contain exactly one input"):
        build_module(module_cls, example_inputs)


def test_build_module_example_inputs_contains_non_tuple():
    module_cls = Mock()
    example_inputs = [1]
    with pytest.raises(ValueError, match="example_inputs must be a list of tensor tuples"):
        build_module(module_cls, example_inputs)


def test_build_module_example_inputs_contains_tuple_with_non_tensor():
    module_cls = Mock()
    example_inputs = [(1,)]
    with pytest.raises(ValueError, match="example_inputs must be a list of tensor tuples"):
        build_module(module_cls, example_inputs)


def test_build_module_with_multiple_example_inputs():
    module_cls = Mock()
    example_inputs = [
        (torch.zeros((SAMPLE_SIZE), dtype=torch.float32),),
        (torch.zeros((SAMPLE_SIZE * 2), dtype=torch.float32),),
    ]
    with pytest.raises(ValueError, match="example_inputs must contain exactly one input"):
        build_module(module_cls, example_inputs)
