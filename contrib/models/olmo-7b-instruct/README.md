# OLMo-7B-Instruct NeuronX Port

NeuronX Distributed Inference port of [allenai/OLMo-7B-Instruct](https://huggingface.co/allenai/OLMo-7B-Instruct).

## Architecture

| Property | Value |
|----------|-------|
| Parameters | 6.9B |
| Hidden size | 4096 |
| Attention | MHA, 32 heads |
| Layers | 32 |
| Intermediate | 11008 (fused: 22016 in checkpoint) |
| Vocab | 50304 (padded from 50280) |
| Activation | SwiGLU |
| Position encoding | RoPE |
| Normalization | LayerNorm, non-affine (no learnable params) |
| Weight tying | No |

## Key Implementation Details

- **Fused QKV splitting**: HF checkpoint has `att_proj [12288, 4096]` which is split into Q, K, V (each `[4096, 4096]`)
- **Fused MLP splitting**: `ff_proj [22016, 4096]` split into up (first half) and gate (second half), each `[11008, 4096]`. OLMo SwiGLU convention: `x, gate = chunk(2); silu(gate) * x`
- **Non-affine LayerNorm**: No norm weights in state dict; uses manual mean/var ops for Neuron traceability
- **Config format**: `model_type: "hf_olmo"` uses non-standard field names (d_model, n_heads, n_layers, etc.)

## Compile & Validate

```bash
# Submit as SLURM job
sbatch run_validation.sh

# Or run directly on a Neuron instance
python compile_and_validate.py

# Compile only
python compile_and_validate.py --compile-only

# Validate only (requires existing compiled model)
python compile_and_validate.py --validate-only
```

## Configuration

- **TP degree**: 2
- **Batch size**: 1
- **Sequence length**: 128
- **Dtype**: bfloat16

## Validation Results

| Metric | Value |
|--------|-------|
| Greedy token match | 93.75% (300/320) |
| Teacher-forced match | **99.38%** |
| Prompts tested | 10 |
| Tokens per prompt | 32 |
| Perfect prompts | 9/10 (100% match) |

Validated on NXDI 0.6.0, Neuron SDK 2.x.
