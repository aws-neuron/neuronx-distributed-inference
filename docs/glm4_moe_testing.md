# GLM-4.5 MoE Testing Results

## Test Environment

- Hardware: AWS Neuron (2 NeuronCores, TP=2)
- Framework: neuronx-distributed-inference
- Model: `glm4_moe_tiny_random` (random weights, 2 layers)
- Date: 2026-02-19

## Test 1: Compilation

**Script:** `examples/generation_glm4_moe_demo.py`

```
python examples/generation_glm4_moe_demo.py
```

**Result:** ✅ PASS

- Context encoding model: compiled successfully (HLO generation: 0.56s, compilation: ~6s)
- Token generation model: compiled successfully (HLO generation: 0.14s, compilation: ~6s)
- Warmup: completed in 0.18s
- No errors during compilation

**Compiled model saved to:** `glm4_moe_tiny_random_traced/`

## Test 2: Inference (Generation)

**Script:** `examples/generation_glm4_moe_demo.py --skip-compile`

```
python examples/generation_glm4_moe_demo.py --skip-compile
```

**Result:** ✅ PASS

Input (dummy tokens): `[1, 2, 3, 4, 5]`

Output (64 tokens generated successfully):
```
tensor([[1, 2, 3, 4, 5, 858, 90494, 22447, 93108, 94805, 5088, 85841, 22027, 113433, 67501, ...]])
```

## Test 3: Accuracy Comparison (Neuron vs HuggingFace)

**Script:** `test_glm4_moe_accuracy.py`

```
python test_glm4_moe_accuracy.py
```

**Result:** ✅ PASS — Exact token match

| Step | HF (CPU, float32) | Neuron (bfloat16) |
|------|-------------------|-------------------|
| 1 | 858 | 858 |
| 2 | 90494 | 90494 |
| 3 | 22447 | 22447 |
| 4 | 93108 | 93108 |
| 5 | 94805 | 94805 |
| 6 | 5088 | 5088 |
| 7 | 85841 | 85841 |
| 8 | 22027 | 22027 |
| 9 | 113433 | 113433 |
| 10 | 67501 | 67501 |

All 10 generated tokens match exactly between HuggingFace (CPU float32) and Neuron (bfloat16).

## Bug Fixed During Testing

### `hf_adapter.py`: `tensor_capture_hook` NameError and TypeError

**Root cause:** `hf_adapter.py` referenced `tensor_capture_hook` without first defining it,
then unconditionally passed it to `model.forward()` which doesn't accept it for text models.

**Fix applied in:** `src/neuronx_distributed_inference/utils/hf_adapter.py`

1. Added `tensor_capture_hook = kwargs.get("tensor_capture_hook", None)` (line ~296)
2. Removed unconditional `"tensor_capture_hook": tensor_capture_hook` from `model_inputs`
3. Added `import inspect`
4. Added conditional inclusion only when the model's `forward()` accepts it

This fix is backward-compatible: multimodal models (Pixtral, etc.) that declare
`tensor_capture_hook` in their `forward()` continue to work unchanged.

## Files Created

| File | Description |
|------|-------------|
| `src/neuronx_distributed_inference/models/glm4_moe/__init__.py` | Module init |
| `src/neuronx_distributed_inference/models/glm4_moe/modeling_glm4_moe.py` | Main implementation |
| `examples/generation_glm4_moe_demo.py` | Generation demo script |
| `test_glm4_moe_accuracy.py` | Accuracy comparison test |
| `create_glm4_tiny_random.py` | Script to create tiny random model |
| `glm4_moe_tiny_random/` | Tiny random model checkpoint |
| `glm4_moe_tiny_random_traced/` | Compiled Neuron model |
| `docs/glm4_moe_implementation.md` | Architecture documentation |
| `docs/glm4_moe_testing.md` | This file |
