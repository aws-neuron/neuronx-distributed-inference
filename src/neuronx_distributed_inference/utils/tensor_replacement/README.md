# Tensor Replacement Example

This example demonstrates an end-to-end flow for:
1. **Capturing** intermediate tensors from both a CPU (eager Hugging Face) model and a Neuron-compiled model.
2. **Replacing** target tensors in the Neuron model at runtime using CPU captures (teacher forcing / tensor replacement).
3. **Visualizing** CPU vs. Neuron differences across steps.

It is built around the **Qwen3-MoE** architecture but can be generalized.

Current limitations:
- Tested for router logits in MoE models
- Tested with Vanilla config without SP/CP/EP and other strategies and kernels that require separate plumbing

---
## What’s Inside

- **`run_tensor_capture(...)`**  
  Captures tensors from CPU and Neuron runs for the given prompt(s), saving them into prompt-specific directories.

- **`run_tensor_replacement(...)`**  
  Runs the Neuron model with selective tensor replacement, using previously captured CPU tensors.

- **`compare_and_plot(...)`**  
  Loads saved tensors and generates scatter plots comparing CPU vs. Neuron outputs per step/module.

---
## Quick Start

1. **Provide a model config** at `./config.json` (small random HF model used for compilation).

2. **Run the example script:**

```bash
python tensor_replacement_example.py
```

3. **The script will:**

- Build and save a small random HF model.

- Compile and load a Neuron model with tensor capture enabled.

- Capture CPU and Neuron tensors under ~/tensor_replace_example/tensor_capture/.

- Run tensor replacement and save outputs under ~/tensor_replace_example/tensor_replace/.

- Generate scatter plots comparing CPU vs. Neuron results for selected steps.