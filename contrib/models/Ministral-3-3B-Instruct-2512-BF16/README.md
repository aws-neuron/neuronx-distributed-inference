# Contrib Model: Ministral-3-3B-Instruct-2512-BF16

NKI-optimized NeuronX Distributed Inference configuration for Ministral-3-3B with TKG fused attention kernel. Fastest Mistral-family model on Neuron (158 tok/s).

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-3-3B-Instruct-2512-BF16`
- **Model Type:** Decoder-only dense transformer (3.3B params, text-only extraction from multimodal)
- **Architecture:** Mistral3ForConditionalGeneration -> extracted as LlamaForCausalLM
- **GQA:** 32 query heads / 8 KV heads, head_dim=128
- **NxDI Path:** NeuronLlamaModel (stock `modeling_llama.py`)
- **License:** Check HuggingFace model card

## Text Extraction Required

The HuggingFace checkpoint is a multimodal model (`Mistral3ForConditionalGeneration`). The text decoder must be extracted before serving with vLLM. The `extract_text_model.py` script handles this, saving the result as `LlamaForCausalLM`.

### Why LlamaForCausalLM (Not MistralForCausalLM)

The NxDI Mistral code path (`modeling_mistral.py`) hardcodes `head_dim = hidden_size // num_attention_heads`. For this model, that gives `3072/32 = 96`, but the actual `head_dim` is 128. This causes RoPE shape mismatches. The Llama code path reads `head_dim` from config correctly.

## Performance Results

**Instance:** trn2.3xlarge | **SDK:** 2.29 | **TP:** 4 | **LNC:** 2 | **Precision:** BF16

### TKG Optimized (max_num_seqs=1)

| Workload | tok/s P50 | TPOT P50 | vs Baseline |
|----------|:---------:|:--------:|:-----------:|
| short-short (128/128) | **158.3** | 6.32ms | +18.0% |
| short-long (128/512) | **156.0** | 6.41ms | +16.9% |
| long-short (2048/128) | **155.3** | 6.44ms | +27.1% |
| long-long (2048/512) | **153.1** | 6.53ms | +26.1% |

Fastest model in the Mistral family. Lowest TPOT (6.3ms).

### Baseline (max_num_seqs=4, no TKG)

| Workload | tok/s P50 | TPOT P50 |
|----------|:---------:|:--------:|
| short-short (128/128) | 134.2 | 7.5ms |
| short-long (128/512) | 133.4 | 7.5ms |
| long-short (2048/128) | 122.2 | 8.2ms |
| long-long (2048/512) | 121.4 | 8.2ms |

### GPU Comparison

GPU baseline BLOCKED: `Mistral3ForConditionalGeneration` / `ministral3` config type requires transformers 5.x, incompatible with vLLM.

## Quick Start

### 1. Launch Instance

trn2.3xlarge with Deep Learning AMI Neuron (Ubuntu 24.04) 20260410 (SDK 2.29).

### 2. Download and Extract Text Model

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

huggingface-cli download mistralai/Ministral-3-3B-Instruct-2512-BF16 \
    --local-dir /home/ubuntu/models/Ministral-3-3B-Instruct-2512-BF16

python src/extract_text_model.py \
    --src /home/ubuntu/models/Ministral-3-3B-Instruct-2512-BF16 \
    --dst /home/ubuntu/models/Ministral-3B-text-only
```

### 3. Apply Patches

```bash
python src/setup_patches.py --tkg-mode multi-kv
```

Only 2 library-level patches needed (eps guards). The Llama code path already has rms_norm_eps, fused_qkv, and fused_rmsnorm built in. The multi-KV TKG kernel and adapter are installed for the 2 KV heads/rank at TP=4.

### 4. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Ministral-3B-text-only \
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

### TKG Eliminates Context-Length Degradation

Baseline drops from 134 -> 122 tok/s for long inputs (9%). TKG: 158 -> 153 tok/s (only 3%). TPOT is flat at 6.3-6.5ms regardless of context length.

### Voxtral Mini-3B Cross-Reference

Voxtral Mini-3B uses this exact same text backbone. Decoder optimizations from this contrib apply directly to Voxtral's text decoder path.

## Files

```
Ministral-3-3B-Instruct-2512-BF16/
  README.md                              -- This file
  src/
    __init__.py
    setup_patches.py                     -- Master patch script (--tkg-mode multi-kv|stock|none)
    extract_text_model.py                -- Extract text decoder from multimodal checkpoint
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
