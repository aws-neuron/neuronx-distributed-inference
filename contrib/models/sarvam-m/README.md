# Contrib Model: sarvam-m

NeuronX Distributed Inference support for [sarvam-m](https://huggingface.co/sarvamai/sarvam-m), a 24B Mistral-architecture decoder-only LLM optimized for Indian languages (Hindi, Tamil, Telugu, etc.) and English.

## Model Information

- **HuggingFace ID:** `sarvamai/sarvam-m`
- **Model Type:** Decoder-only transformer (MistralForCausalLM)
- **Parameters:** ~24B (BF16)
- **Architecture:** GQA (32Q/8KV heads), head_dim=128, hidden_size=5120, 40 layers, RoPE (theta=1M), SiLU, vocab 131,072, 32K context
- **License:** Check HuggingFace model card

## Issue: Mistral head_dim Mismatch

sarvam-m uses `hidden_size=5120` with `num_attention_heads=32`, giving a computed `head_dim` of 160. However, the model config explicitly sets `head_dim=128` -- the actual dimension used for attention projections.

NxDI's `NeuronMixtralAttention` (which handles Mistral-family models) hardcodes `head_dim = config.hidden_size // config.num_attention_heads`, ignoring the explicit `head_dim` in the config. This causes a shape mismatch: K/V projections output `num_kv_heads * 128 = 1024` elements, but the KV cache is allocated for `num_kv_heads * 160 = 1280`.

Other model implementations in NxDI (Llama, Qwen, Gemma) correctly use `getattr(config, "head_dim", ...)` to read the explicit config value. The Mistral/Mixtral code path is the only one missing this pattern.

### Patches

The `src/setup_patches.py` script applies three patches:

1. **Mixtral head_dim** (`modeling_mixtral.py`): Use `getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)` in `NeuronMixtralAttention.__init__()` for both `RotaryEmbedding` and the `head_dim` passed to the parent class. This matches the pattern used by `NeuronLlamaAttention` and other implementations.

2. **nkilib QKV CTE eps guard** (`nkilib/core/qkv/qkv_cte.py`): Guard `nisa.memset(value=norm_eps)` against `None` when `norm_eps` is not set.

3. **neuronxcc QKV CTE eps guard** (`neuronxcc/nki/_pre_prod_kernels/qkv_cte_impl.py`): Guard `bias_eps[...] = eps` against `None`.

## Validation Results

**Validated:** 2026-04-22
**Instance:** trn2.3xlarge (LNC=1, 8 NeuronCores, TP=8)
**SDK:** Neuron SDK 2.29, PyTorch 2.9, NxDI 0.9, vLLM 0.16.0 + vllm-neuron 0.5.0

### Benchmark Results

| Workload (in/out) | Concurrency=1 tok/s | Concurrency=1 TPOT | Concurrency=4 tok/s | Concurrency=4 TPOT |
|--------------------|---------------------|---------------------|---------------------|---------------------|
| 128/128            | 42.3                | 23.6 ms             | 157.0               | 25.5 ms             |
| 128/512            | 42.4                | 23.6 ms             | 160.1               | 25.0 ms             |
| 2048/128           | 38.4                | 26.1 ms             | 120.9               | 33.1 ms             |
| 2048/512           | 39.8                | 25.1 ms             | 144.4               | 27.7 ms             |

**Peak throughput:** 160.1 tok/s (128 in / 512 out, concurrency=4)

### TP=4 (LNC=2) Comparison

| Config | Single-stream tok/s | Peak tok/s (conc=4) |
|--------|---------------------|---------------------|
| TP=8, LNC=1 | 42.3 | 160.1 |
| TP=4, LNC=2 | 36.1 | 132.8 |

TP=8 with LNC=1 is 17% faster single-stream. TP=4 is suboptimal for this model.

### Accuracy Validation

| Test | Status | Result |
|------|--------|--------|
| Greedy determinism | PASS | Identical output across multiple runs |
| English accuracy | PASS | Correct factual responses |
| Hindi accuracy | PASS | Coherent Devanagari script output |
| Content coherence | PASS | No repetition, proper punctuation |

## Usage

### 1. Apply patches

```bash
# On the Neuron instance, activate the venv first
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Apply patches (requires sudo for site-packages)
sudo python contrib/models/sarvam-m/src/setup_patches.py
```

### 2. Set LNC=1 for TP=8

```bash
# Set LNC=1 (required for TP=8 on trn2.3xlarge)
echo 'NEURON_LOGICAL_NC_CONFIG=1' | sudo tee /etc/environment
sudo reboot
# After reboot, verify: neuron-ls should show 8 cores
```

### 3. Start vLLM

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model sarvamai/sarvam-m \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --no-enable-prefix-caching \
  --additional-config '{"override_neuron_config": {
    "qkv_nki_kernel_enabled": true,
    "qkv_kernel_enabled": true
  }}'
```

### 4. Query

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sarvamai/sarvam-m",
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "temperature": 0
  }' | python3 -m json.tool
```

## Compatibility Matrix

| Instance/Version | SDK 2.29 | SDK 2.28 |
|------------------|----------|----------|
| trn2.3xlarge (TP=8, LNC=1) | VALIDATED | Not tested |
| trn2.3xlarge (TP=4, LNC=2) | VALIDATED (suboptimal) | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| Inf2 | Not tested | Not tested |

## Example Checkpoints

* [sarvamai/sarvam-m](https://huggingface.co/sarvamai/sarvam-m)

## Testing Instructions

```bash
# Apply patches first
python contrib/models/sarvam-m/src/setup_patches.py

# Run integration tests
pytest contrib/models/sarvam-m/test/integration/test_model.py -v --capture=tee-sys

# Or run directly
python contrib/models/sarvam-m/test/integration/test_model.py
```

Environment variables for custom paths:
```bash
export SARVAM_M_MODEL_PATH=/path/to/local/model
export SARVAM_M_TP_DEGREE=8
export SARVAM_M_MAX_MODEL_LEN=8192
```

## Known Issues

1. **head_dim patch required**: Without the head_dim patch, the model fails at compilation with an XLA broadcast shape mismatch. This is a general issue affecting any Mistral-family model where `head_dim != hidden_size // num_attention_heads`.

2. **NKI eps patches required**: The NKI QKV CTE kernel crashes when `norm_eps=None` is passed. These patches are model-agnostic guards.

3. **fused_qkv not supported**: The Mistral code path does not support `fused_qkv` (only the Llama code path does). Enabling `fused_qkv` causes a `KeyError` on `Wqkv.weight`.

4. **TP=4 suboptimal**: Despite having fewer inter-core communication overhead, TP=4 (LNC=2) is 17% slower than TP=8 (LNC=1) for this 24B model due to memory capacity constraints.

## Maintainer

Jim Burtoft

**Last Updated:** 2026-04-23
