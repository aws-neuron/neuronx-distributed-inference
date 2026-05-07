# Tensor Capture in NeuronxDistributedInference

This example demonstrates how to use the tensor capture functionality in NeuronxDistributedInference to capture and analyze intermediate tensors during model execution.

## Overview

NeuronxDistributedInference builds upon the tensor capture functionality provided by NeuronxDistributed, adding convenient utilities for:

1. Configuring tensor capture through the `TensorCaptureConfig` class
2. Capturing tensors during generation with a tensor capture hook
3. Saving captured tensors to disk for later analysis
4. Analyzing captured tensors and comparing them with reference tensors

## Current Limitations

- For multimodal models (like Llama4), tensor capture currently only works for the text model component, not for the vision model component.
- Tensor capture adds overhead to model execution, so it should be used primarily for debugging and not in production.
- The number of tensors captured can grow large, especially for long generation sequences.
- For very large models, capturing tensors from many modules may impact device HBM (High Bandwidth Memory) usage, as the model graph is updated to put the captured tensors in a registry and then added as extra outputs at the end.

## How It Works

The tensor capture system in NeuronxDistributedInference consists of three main components:

1. **Configuration**: Using `TensorCaptureConfig` in the model's `neuron_config`
2. **Tensor Registration**: Automatic registration of tensors from specified modules (handled by NeuronxDistributed)
3. **Tensor Capture Hook**: A hook that saves captured tensors during model execution

### Configuration

To enable tensor capture, configure it in your model's `neuron_config`:

```python
from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig

neuron_config = NeuronConfig(
    # ... other config options ...
    tensor_capture_config=TensorCaptureConfig(
        modules_to_capture=['layers.0.mlp', 'layers.1.self_attn'],  # List of module names to capture
        capture_inputs=True,  # Whether to capture input tensors for the specified modules
        max_intermediate_tensors=10,  # Maximum number of intermediate tensors to capture
    )
)
```

### Tensor Capture Hook

To save captured tensors during model execution, provide a tensor capture hook:

```python
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook

# Create a tensor capture hook that saves tensors at generation steps 1, 5, and 10
tensor_capture_hook = get_tensor_capture_hook(
    capture_indices=[1, 5, 10],
    tensor_capture_save_dir="captured_tensors"
)

# Use the hook during model generation
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tensor_capture_hook=tensor_capture_hook,
    # ... other generation options ...
)
```

## Relationship with NeuronxDistributed's Tensor Capture

NeuronxDistributedInference's tensor capture builds on top of NeuronxDistributed's tensor capture functionality:

1. **NeuronxDistributed's Tensor Capture**:
   - Provides the core functionality for capturing tensors from modules
   - Handles the wrapping of modules to capture inputs/outputs
   - Manages the tensor registry for storing captured tensors

2. **NeuronxDistributedInference's Tensor Capture**:
   - Provides a configuration interface through `TensorCaptureConfig`
   - Offers utilities for saving captured tensors during generation
   - Includes tools for analyzing captured tensors

## Example Usage

See the [tensor_capture_example.py](./tensor_capture_example.py) file for a complete example of using the tensor capture utility with a Llama3 model.

For multimodal models, see [multimodal_tensor_capture_example.py](./multimodal_tensor_capture_example.py), which demonstrates tensor capture with Llama4 multimodal models (note that currently only the text model component's tensors can be captured).

## Advanced Usage

### Analyzing Captured Tensors

After capturing tensors, you can analyze them using the provided utility:

```python
from neuronx_distributed_inference.utils.tensor_capture_utils import analyze_captured_tensors

# Analyze captured tensors
results = analyze_captured_tensors(
    tensor_dir="captured_tensors",
    reference_dir="reference_tensors"  # Optional
)

# Print summary
print(f"Total tensors captured: {results['summary']['total_tensors']}")
print(f"Modules captured: {results['summary']['modules_captured']}")
print(f"Steps captured: {results['summary']['steps_captured']}")

# Analyze individual tensors
for tensor_info in results['tensors']:
    print(f"Tensor: {tensor_info['name']}")
    print(f"  Shape: {tensor_info['shape']}")
    print(f"  Min/Max: {tensor_info['min']:.6f}/{tensor_info['max']:.6f}")
    print(f"  Mean/Std: {tensor_info['mean']:.6f}/{tensor_info['std']:.6f}")
    
    # If comparison with reference is available
    if 'ref_comparison' in tensor_info:
        print(f"  Max abs diff: {tensor_info['ref_comparison']['max_abs_diff']:.6f}")
        print(f"  Mean abs diff: {tensor_info['ref_comparison']['mean_abs_diff']:.6f}")
        print(f"  Relative diff: {tensor_info['ref_comparison']['rel_diff']:.6f}")
```

### Finding Capturable Modules

To discover which modules can be captured in your model, use the `list_capturable_modules_in_application` function:

```python
from neuronx_distributed_inference.utils.tensor_capture_utils import list_capturable_modules_in_application

# List all capturable modules in the model
modules = list_capturable_modules_in_application(model)

# Print available modules for each model component
for model_name, available_modules in modules.items():
    print(f"Available modules in {model_name}:")
    for module_name, module_type in available_modules.items():
        print(f"  {module_name}: {module_type}")
```

## Best Practices

1. **Be selective about which modules to capture**: Capturing too many modules can slow down execution and consume memory.
2. **Consider tensor size**: Capturing large tensors can consume significant memory.
3. **For accuracy verification**: Capture the same modules in both reference and test models for meaningful comparisons.
4. **For sharded tensors**: Be aware of the tensor parallelism degree and handle comparisons accordingly.

## Limitations

- Tensor capture adds overhead to model execution, so it should be used primarily for debugging and not in production.
- The number of tensors captured can grow large, especially for long generation sequences.
- For very large models, capturing tensors from many modules may impact memory usage.

## See Also

For more details on the underlying tensor capture mechanism, see the [NeuronxDistributed Tensor Capture README](https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/utils/tensor_capture/README.md).
