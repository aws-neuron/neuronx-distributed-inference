# Solar Open MoE — Testing Guide

## Overview

This document describes how to test the Solar Open MoE NXD inference implementation for correctness.

---

## Test Strategy

Since `solar_open` is not registered in the `transformers` library, we cannot use `SolarOpenForCausalLM.from_pretrained(...)` as a CPU reference. Instead, `test_solar_open_accuracy.py` contains a pure PyTorch CPU reference implementation (`SolarOpenReferenceModel`) that:

1. Loads the same `model.safetensors` checkpoint
2. Runs a forward pass and greedy generation
3. Compares generated token IDs against the Neuron model

With random weights and greedy decoding (`top_k=1`), the outputs should be **exactly identical**.

---

## Prerequisites

1. **Neuron venv active**:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
   ```

2. **Tiny random model created**:
   ```bash
   python create_solar_open_tiny_random.py
   ```
   Output: `solar_open_tiny_random/` (config.json + model.safetensors)

3. **Model compiled** (or use existing traced model):
   ```bash
   python examples/generation_solar_open_demo.py
   ```
   Output: `solar_open_tiny_random_traced/` (model.pt + neuron_config.json + model.safetensors)

---

## Running the Accuracy Test

### Quick test (no Slack notifications):
```bash
python test_solar_open_accuracy.py --no-slack
```

### Full test (with Slack notifications):
```bash
python test_solar_open_accuracy.py
```

### Compile and test in one command:
```bash
python test_solar_open_accuracy.py --compile
```

### Custom paths:
```bash
python test_solar_open_accuracy.py \
    --model-path /path/to/solar_open_model \
    --traced-model-path /path/to/traced_model \
    --max-new-tokens 10
```

---

## Running the Demo Script

### Compile and generate (first time):
```bash
python examples/generation_solar_open_demo.py
```

### Skip compilation (load existing traced model):
```bash
python examples/generation_solar_open_demo.py --skip-compile
```

### Custom arguments:
```bash
python examples/generation_solar_open_demo.py \
    --model-path /path/to/solar_open_model \
    --traced-model-path /path/to/traced_model \
    --tp-degree 2 \
    --seq-len 64
```

---

## Expected Test Output

```
Starting Solar Open accuracy test...

============================================================
Loading CPU reference model...
============================================================
CPU reference model loaded successfully.

============================================================
Running CPU reference generation...
============================================================
Reference input_ids:  [[1, 100, 200, 300, 400]]
Reference new tokens: [[23045, 110508, 79732, 185678, 159306, 78468, 101317, 139425, 22825, 47784]]

============================================================
Running Neuron model generation...
============================================================
Neuron new tokens:   [[23045, 110508, 79732, 185678, 159306, 78468, 101317, 139425, 22825, 47784, ...]]

============================================================
Comparing outputs...
============================================================
✅ PASSED: Neuron output matches CPU reference!
  Generated 10 tokens, all match.
```

---

## Test Architecture

### CPU Reference Model (`SolarOpenReferenceModel`)

Pure PyTorch implementation in `test_solar_open_accuracy.py`:

- `SolarOpenAttention` — Full RoPE, GQA, no bias
- `SolarOpenMoE` — Sigmoid router + group routing + routed experts + shared experts
- `SolarOpenDecoderLayer` — Attention + MoE + RMSNorm
- `SolarOpenReferenceModel` — Complete forward pass + greedy generation

The reference model loads weights directly from safetensors with key mapping:
```
HF key                                  → Reference model key
mlp.experts.gate_up_proj               → mlp.experts_gate_up
mlp.experts.down_proj                  → mlp.experts_down
mlp.gate.weight                        → mlp.gate_weight
mlp.gate.e_score_correction_bias       → mlp.e_score_correction_bias
mlp.shared_experts.gate_proj.weight    → mlp.shared_gate_proj.weight
mlp.shared_experts.up_proj.weight      → mlp.shared_up_proj.weight
mlp.shared_experts.down_proj.weight    → mlp.shared_down_proj.weight
```

### Neuron Model

Loaded from compiled traced model path via `NeuronSolarOpenForCausalLM` + `HuggingFaceGenerationAdapter`.

---

## Verified Test Results

| Test Date | Input | Reference Output | Neuron Output | Match |
|-----------|-------|-----------------|---------------|-------|
| 2026-02-19 | `[1, 100, 200, 300, 400]` | `[23045, 110508, 79732, 185678, 159306, 78468, 101317, 139425, 22825, 47784]` | Same | ✅ PASS |

---

## Known Warnings (Expected)

These warnings appear during testing and are safe to ignore:

1. **`torch_neuronx.nki_jit is deprecated`** — Use `nki.jit` instead. Cosmetic only.
2. **`Found torch.float32 weights: e_score_correction_bias. Will convert to torch.bfloat16`** — The bias is stored as float32 and auto-converted on load.
3. **`Removing redundant keys from checkpoint: o_proj.weight, Wqkv.weight`** — NXD weight sharding removes unfused weights after fusion.
4. **`NET/OFI Failed to initialize rdma protocol`** — EFA not configured on this instance. Neuron collectives work without EFA.
5. **`NeuronConfig init: Unexpected keyword arguments`** — Fields from newer NXD versions not recognized. Safe to ignore.

---

## Troubleshooting

### `FileNotFoundError: Can not find model.safetensors in traced_model_path`
The demo script copies `model.safetensors` to the traced path automatically during compile. If missing, copy manually:
```bash
cp solar_open_tiny_random/model.safetensors solar_open_tiny_random_traced/
```

### `ValueError: model type solar_open not recognized`
This occurs if `load_pretrained_config` (which uses `AutoConfig`) is used instead of `load_solar_open_config`. Always use `load_solar_open_config(model_path)` for solar_open.

### `AttributeError: output_attentions not found`
If running with an old compiled model (before the `SolarOpenInferenceConfig` fix), recompile:
```bash
python examples/generation_solar_open_demo.py
```
