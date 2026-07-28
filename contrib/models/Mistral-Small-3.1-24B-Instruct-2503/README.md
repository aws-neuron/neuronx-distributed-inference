# Contrib Model: Mistral-Small-3.1-24B-Instruct-2503

NKI-optimized NeuronX Distributed Inference configuration for Mistral-Small-3.1-24B with **stock TKG** fused attention kernel. Closest to GPU parity in the Mistral family (1.27x gap). No custom kernel needed.

## Model Information

- **HuggingFace ID:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- **Model Type:** Decoder-only dense transformer (24B params, text-only extraction from multimodal)
- **Architecture:** Mistral3ForConditionalGeneration -> extracted as LlamaForCausalLM
- **GQA:** 32 query heads / 8 KV heads, head_dim=128
- **NxDI Path:** NeuronLlamaModel (stock `modeling_llama.py`)
- **License:** Check HuggingFace model card

## Text Extraction Required

The HuggingFace checkpoint is a multimodal model (`Mistral3ForConditionalGeneration`). The text decoder must be extracted before serving with vLLM. The `extract_text_model.py` script handles this.

### Why LlamaForCausalLM (Not MistralForCausalLM)

Same as Ministral 3B: the NxDI Mistral path hardcodes `head_dim = hidden_size // num_attention_heads`, giving `5120/32 = 160` instead of the actual 128. The Llama path reads head_dim from config correctly.

## Performance Results

**Instance:** trn2.3xlarge | **SDK:** 2.29 | **TP:** 8 | **LNC:** 1 | **Precision:** BF16

### Stock TKG Optimized (max_num_seqs=1)

| Workload | tok/s P50 | TPOT P50 | vs Baseline |
|----------|:---------:|:--------:|:-----------:|
| short-short (128/128) | **47.73** | 20.95ms | +6.2% |
| short-long (128/512) | **47.56** | 21.03ms | +7.4% |
| long-short (2048/128) | **47.48** | 21.06ms | +13.7% |
| long-long (2048/512) | **47.21** | 21.18ms | +13.6% |

TPOT is flat at ~21ms regardless of context length. TKG eliminates context-length degradation entirely.

### Baseline (max_num_seqs=4, no TKG)

| Workload | tok/s P50 | TPOT P50 |
|----------|:---------:|:--------:|
| short-short (128/128) | 44.95 | 22.25ms |
| short-long (128/512) | 44.29 | 22.58ms |
| long-short (2048/128) | 41.75 | 23.95ms |
| long-long (2048/512) | 41.55 | 24.07ms |

### GPU Comparison (1x H100, vLLM 0.19.0)

| Metric | Neuron (TKG) | GPU | Ratio |
|--------|:-----------:|:---:|:-----:|
| tok/s (short-short) | **47.73** | 60.8 | 1.27x GPU |
| $/M tokens (spot) | **$5.24** | $7.08 | **Neuron 26% cheaper** |

## Quick Start

### 1. Launch Instance

trn2.3xlarge with Deep Learning AMI Neuron (Ubuntu 24.04) 20260410 (SDK 2.29).

**Set LNC=1 (required for TP=8):**
```bash
echo 'NEURON_LOGICAL_NC_CONFIG=1' | sudo tee /etc/environment
sudo reboot
# After reboot: neuron-ls should show 8 cores
```

### 2. Download and Extract Text Model

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

huggingface-cli download mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --local-dir /home/ubuntu/models/Mistral-Small-3.1-24B-Instruct-2503

python src/extract_text_model.py \
    --src /home/ubuntu/models/Mistral-Small-3.1-24B-Instruct-2503 \
    --dst /home/ubuntu/models/Mistral-Small-24B-text-only
```

### 3. Apply Patches

```bash
python src/setup_patches.py --tkg-mode stock
```

Only 2 library-level patches needed (eps guards). **No custom TKG kernel** -- at TP=8 there is 1 KV head per rank, so the stock NxDI TKG kernel works natively.

### 4. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Mistral-Small-24B-text-only \
    --tensor-parallel-size 8 \
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

### Why Stock TKG Works (No Custom Kernel)

At TP=8, the 8 KV heads are sharded to **1 KV head per rank**. The stock NxDI TKG kernel handles this natively. No multi-KV adapter or forked kernel is needed -- just enable the config flags. This makes this the cleanest TKG integration in the Mistral family.

### TKG Eliminates Long-Input Concurrency Degradation

Baseline long-short drops from 41.75 -> 33.54 tok/s at conc=4 (20% drop). TKG stays flat at 47.5 tok/s -- a +41.7% improvement for long-input concurrent workloads.

### Voxtral Small-24B Cross-Reference

Voxtral Small-24B uses this exact same text backbone (23.6B LlamaForCausalLM, hidden=5120, 40 layers). All decode-path optimizations from this contrib apply directly to Voxtral's text decoder. The stock TKG flag can be added to Voxtral's NeuronConfig without any custom kernel code.

## Files

```
Mistral-Small-3.1-24B-Instruct-2503/
  README.md                              -- This file
  src/
    __init__.py
    modeling_mistral3.py                 -- Original auto-generated model (kept for reference)
    setup_patches.py                     -- Master patch script (--tkg-mode stock)
    extract_text_model.py                -- Extract text decoder from multimodal checkpoint
    attention_block_tkg_multi_kv.py      -- Multi-KV TKG kernel (included for completeness)
    multi_kv_adapter.py                  -- Adapter monkeypatch (not needed for stock TKG)
    fix_nki030.py                        -- NKI 0.3.0 compatibility fix
    fix_nki030_v2.py                     -- NKI 0.3.0 compatibility fix (v2)
  test/
    __init__.py
    integration/
      __init__.py
      test_model.py                      -- Smoke test + token matching
```
