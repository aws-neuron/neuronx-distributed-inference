# Contrib Model: Ministral-8B-Instruct-2410

NKI-optimized NeuronX Distributed Inference configuration for Ministral-8B-Instruct-2410 with TKG fused attention kernel.

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-8B-Instruct-2410`
- **Model Type:** Decoder-only dense transformer (8B params)
- **Architecture:** MistralForCausalLM (standard, no extraction needed)
- **GQA:** 32 query heads / 8 KV heads, head_dim=128
- **NxDI Path:** NeuronMistralModel (stock `modeling_mistral.py`)
- **License:** Check HuggingFace model card

## Config Fix Required

The HuggingFace config has `sliding_window: 32768` and alternating `layer_types` (full/sliding attention). NxDI does not support per-layer sliding window. The `patch_config.py` script sets `sliding_window: null` and removes `layer_types`. This is equivalent for `max_model_len < 32768`.

## Performance Results

**Instance:** trn2.3xlarge | **SDK:** 2.29 | **TP:** 4 | **LNC:** 2 | **Precision:** BF16

### TKG Optimized (max_num_seqs=1)

| Workload | tok/s P50 | TPOT P50 | vs Baseline |
|----------|:---------:|:--------:|:-----------:|
| short-short (128/128) | **99.4** | 10.06ms | +21.1% |
| short-long (128/512) | **98.5** | 10.15ms | +21.9% |
| long-short (2048/128) | **97.2** | 10.30ms | +28.1% |
| long-long (2048/512) | **96.8** | 10.35ms | +29.4% |

### Baseline (max_num_seqs=4, no TKG)

| Workload | tok/s P50 | TPOT P50 |
|----------|:---------:|:--------:|
| short-short (128/128) | 82.1 | 12.2ms |
| short-long (128/512) | 80.8 | 12.4ms |
| long-short (2048/128) | 75.9 | 13.2ms |
| long-long (2048/512) | 74.8 | 13.4ms |

### GPU Comparison (1x H100, vLLM 0.8.5)

| Metric | Neuron (TKG) | GPU | Ratio |
|--------|:-----------:|:---:|:-----:|
| tok/s (short-short) | **99.4** | 131.0 | 1.32x GPU |
| $/M tokens (spot) | **$2.51** | $3.29 | **Neuron 24% cheaper** |

## Quick Start

### 1. Launch Instance

trn2.3xlarge with Deep Learning AMI Neuron (Ubuntu 24.04) 20260410 (SDK 2.29).

### 2. Download Model and Fix Config

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

huggingface-cli download mistralai/Ministral-8B-Instruct-2410 \
    --local-dir /home/ubuntu/models/Ministral-8B-Instruct-2410

# Fix sliding_window and layer_types in config
python src/patch_config.py /home/ubuntu/models/Ministral-8B-Instruct-2410
```

### 3. Apply Patches

```bash
python src/setup_patches.py
```

Same 6 patches as Mistral 7B (shared NeuronMistralModel code path).

### 4. Start vLLM Server

**TKG optimized (recommended, max_num_seqs=1):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Ministral-8B-Instruct-2410 \
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

## Technical Details

### Relationship to Mistral 7B

Same NxDI code path (NeuronMistralModel), same 6 patches, same multi-KV TKG kernel. Key differences:
- 36 layers (vs 32), intermediate_size=12288 (vs 14336)
- Requires config fix (sliding_window/layer_types)
- ~10% slower than Mistral 7B due to 4 extra layers

### TKG Batch Size Limitation

Same as Mistral 7B: BIR verification failure at batch sizes > 1. TKG requires `max_num_seqs=1`.

## Files

```
Ministral-8B-Instruct-2410/
  README.md                              -- This file
  src/
    __init__.py
    setup_patches.py                     -- Master patch script (6 patches)
    patch_config.py                      -- Fix sliding_window/layer_types in config.json
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
