By default, this package is configured to run PyTest tests
(http://pytest.org/).

## Running tests

You must be on a Neuron device to run tests.

First install the package for testing
```
pip install -e .[test]
```

Run unit tests 
```
pytest test/unit/ --forked
```

Run integration tests
```
pytest test/integration/ --forked
```

## Writing tests

There are two test directories `unit` and `integration`. 

* `unit` - Unit tests that run on CPU and/or Neuron to validate functions and modules work as expected.
* `integration` - Integration tests that run on Neuron to validate modeling code works as expected. These tests run models end-to-end using generated weights, so these tests can be run using only resources available in this package (i.e. no network connections required).

Place test files in these directories, using file names that start with `test_`.

### Testing modules and functions on Neuron

NxD Inference provides common test utilities to help you validate code runs correctly on Neuron.

#### Running modules on Neuron

```
neuronx_distributed_inference.utils.testing.build_module(module_cls, example_inputs, module_init_kwargs={}, tp_degree=1, compiler_args=None, compiler_workdir=None, checkpoint_path=None)
```

Builds a module into a Neuron model. This function traces the module using the example
inputs, which is a list of tuples where each item is a tensor. Then, it compiles the
traced module to produce a Neuron model.


Arguments:
* `module_cls`: The module class to compile.
* `example_inputs`: The list of example inputs to use to trace the module. This list must
  contain exactly one tuple of tensors.
* `tp_degree`: The TP degree to use. Defaults to 1.
* `module_init_kwargs`: The kwargs to pass when initializing the module.
* `compiler_args`: The compiler args to use.
* `compiler_workdir`: Where to save compiler artifacts. Defaults to a tmp folder with a UUID
  for uniqueness.
* `checkpoint_path`: The path to the checkpoint to load. By default, this function saves the
  module state dict to use as the checkpoint.

#### Running functions on Neuron
                                                      
```
neuronx_distributed_inference.utils.testing.build_function(func, example_inputs, tp_degree=1, compiler_args=None, compiler_workdir=None)
```

Builds a function into a Neuron model. See `build_module` for more information about common arguments.

If the function has non-tensor inputs, you must convert it to a function that only takes
tensor inputs. You can use `partial` to do this, where you provide the non-tensor inputs as
constants in the partial function. This step is necessary because all inputs must be tensors
in a Neuron model.

```
def top_k(input: torch.Tensor, k: int, dim: int):
    return torch.topk(input, k, dim)


top_k_partial = partial(top_k, 1, 0)
model = build_fuction(top_k_partial, example_inputs=[(torch.rand(4)),])
output = model(torch.rand(4))
```


#### Validating accuracy

```
neuronx_distributed_inference.utils.testing.validate_accuracy(neuron_model, inputs, expected_outputs=None, cpu_callable=None, assert_close_kwargs={})`
```

Validates the accuracy of a Neuron model. This function tests that the model produces expected
outputs, which you can provide and/or produce on CPU. To compare outputs, this function uses
`torch_neuronx.testing.assert_close`. If the output isn't similar, this function raises an
AssertionError.

Arguments:
* `neuron_model`: The Neuron model to validate.
* `inputs`: The list of inputs to use to run the model. Each input is passed to the model's
  forward function.
* `expected_outputs`: The list of expected outputs for each input. If not provided, this
  function compares against the CPU output for each input.
* `cpu_callable`: The callable to use to produce output on CPU.
* `assert_close_kwargs`: The kwargs to pass to `torch_neuronx.testing.assert_close`.

#### Examples

##### Example: Basic module test
This example demonstrates how to validate the accuracy of a basic module with a single linear layer. In this example, we initialize the module separately on Neuron and CPU (using the `distributed` arg in `ExampleModule`). This flag enables us run a parallel linear layer on Neuron and compare it to a standard linear layer on CPU.

```
# Module to test.
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


def test_validate_accuracy_basic_module():
    inputs = [(torch.arange(0, SAMPLE_SIZE, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((SAMPLE_SIZE), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False)
    neuron_model = build_module(ExampleModule, example_inputs, module_init_kwargs={"distributed": True})

    validate_accuracy(neuron_model, inputs, cpu_callable=module_cpu)
```

##### Example: Basic function test
This example demonstrates how to validate the accuracy of a basic function with tensor args.

```
def example_sum(tensor):
    return torch.sum(tensor)

def test_validate_accuracy_basic_function():
    inputs = [(torch.tensor([1, 2, 3], dtype=torch.float32),)]
    example_inputs = [(torch.zeros((3), dtype=torch.float32),)]

    neuron_model = build_function(example_sum, example_inputs)
    validate_accuracy(neuron_model, inputs, cpu_callable=example_sum)
```

##### Additional examples
For additional examples of `build_module`, `build_function`, and `validate_accuracy`, see the [testing.py unit tests](unit/utils/test_testing.py).