# Contrib Model: Mistral-7B-Instruct-v0.3

NKI-optimized NeuronX Distributed Inference configuration for Mistral-7B-Instruct-v0.3 with TKG fused attention kernel.

## Model Information

- **HuggingFace ID:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Model Type:** Decoder-only dense transformer (7.2B params)
- **Architecture:** MistralForCausalLM (standard, no extraction needed)
- **GQA:** 32 query heads / 8 KV heads, head_dim=128
- **NxDI Path:** NeuronMistralModel (stock `modeling_mistral.py`)
- **License:** Apache 2.0

## Performance Results

**Instance:** trn2.3xlarge | **SDK:** 2.29 | **TP:** 4 | **LNC:** 2 | **Precision:** BF16

### TKG Optimized (max_num_seqs=1)

| Workload | tok/s P50 | TPOT P50 | vs Baseline |
|----------|:---------:|:--------:|:-----------:|
| short-short (128/128) | **115.2** | 8.68ms | +18.6% |
| short-long (128/512) | **108.9** | 9.19ms | +18.6% |
| long-short (2048/128) | **114.9** | 8.71ms | +18.9% |
| long-long (2048/512) | **~109** | ~9.2ms | +24.9% |

### Baseline (max_num_seqs=4, no TKG)

| Workload | tok/s P50 | TPOT P50 |
|----------|:---------:|:--------:|
| short-short (128/128) | 97.1 | 10.3ms |
| short-long (128/512) | 91.8 | 10.9ms |
| long-short (2048/128) | 96.6 | 10.4ms |
| long-long (2048/512) | 87.3 | 11.5ms |

### GPU Comparison (1x H100, vLLM 0.19.0)

| Metric | Neuron (TKG) | GPU | Ratio |
|--------|:-----------:|:---:|:-----:|
| tok/s (short-short) | **115.2** | 169.1 | 1.47x GPU |
| $/M tokens (spot) | **$2.17** | $2.55 | **Neuron 15% cheaper** |

## Quick Start

### 1. Launch Instance

trn2.3xlarge with Deep Learning AMI Neuron (Ubuntu 24.04) 20260410 (SDK 2.29).

### 2. Apply Patches

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
python src/setup_patches.py
```

This applies 6 patches to the NxDI Mistral code path:
1. `rms_norm_eps` pass-through in `modeling_mistral.py`
2. nkilib QKV CTE eps guard
3. neuronxcc QKV CTE eps guard
4. QKV weight fusion in `modeling_mistral.py`
5. Multi-KV TKG kernel installation
6. Multi-KV adapter monkeypatch

### 3. Start vLLM Server

**TKG optimized (recommended, max_num_seqs=1):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --port 8000 \
    --additional-config '{"override_neuron_config": {
        "fused_qkv": true,
        "qkv_nki_kernel_enabled": true,
        "qkv_kernel_enabled": true,
        "attn_block_tkg_nki_kernel_enabled": true,
        "attn_block_tkg_nki_kernel_cache_update": true
    }}'
```

**Baseline (no TKG, higher batch throughput):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --port 8000 \
    --additional-config '{"override_neuron_config": {
        "fused_qkv": true,
        "qkv_nki_kernel_enabled": true,
        "qkv_kernel_enabled": true
    }}'
```

## Technical Details

### Why Multi-KV TKG Kernel

At TP=4, this model has **2 KV heads per rank** (8 KV heads / 4 TP). The stock NxDI TKG kernel only supports 1 KV head per rank. The multi-KV kernel (derived from Leanstral) uses a virtual batch approach to handle multiple KV heads in a single TKG call.

### TKG Batch Size Limitation

The multi-KV TKG kernel triggers `NCC_INLA001` (BIR verification failure) at batch sizes > 1 on SDK 2.29 (neuronx-cc 2.24). This limits TKG to `max_num_seqs=1`. For throughput-optimized workloads (batch processing), use baseline config without TKG at `max_num_seqs=4`.

### Prefix Caching Workaround

`--no-enable-prefix-caching` is required. The block KV cache path in NxDI has a corruption issue that manifests after ~200 requests. This workaround gives stable, crash-free operation.

## Files

```
Mistral-7B-Instruct-v0.3/
  README.md                              -- This file
  src/
    __init__.py
    setup_patches.py                     -- Master patch script (6 patches)
    attention_block_tkg_multi_kv.py      -- Multi-KV TKG fused attention NKI kernel
    multi_kv_adapter.py                  -- Adapter monkeypatch for attention_base.py
    fix_nki030.py                        -- NKI 0.3.0 compatibility fix
    fix_nki030_v2.py                     -- NKI 0.3.0 compatibility fix (v2)
  test/
    __init__.py
    integration/
      __init__.py
      test_model.py                      -- Smoke test + token matching
```
