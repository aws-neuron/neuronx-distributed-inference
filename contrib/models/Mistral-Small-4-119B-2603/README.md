# Contrib Model: Mistral-Small-4-119B-2603

Custom NeuronX Distributed Inference implementation for Mistral-Small-4-119B on trn2.48xlarge.

## Model Information

- **HuggingFace ID:** `mistralai/Mistral-Small-4-119B-2603`
- **Model Type:** MoE + Multi-Latent Attention (MLA), text decoder extracted from multimodal
- **Parameters:** 119B total, ~4B active per token (4/128 experts)
- **License:** Check HuggingFace model card

## Architecture Details

This model uses DeepSeek-V3 architecture with:
- **Multi-Latent Attention (MLA)**: kv_lora_rank=256, q_lora_rank=1024, qk_rope_head_dim=64, v_head_dim=128
- **Mixture of Experts**: 128 routed experts + 1 shared expert, top-4 routing (sigmoid)
- 36 transformer layers, hidden_size=4096
- 32 attention heads, 32 KV heads
- Compressed KV cache: 320-dim per head (256 latent + 64 rope)
- Original: FP8 multimodal model, dequantized to BF16 text-only (238 GB)

## Performance (SDK 2.29, vLLM 0.16.0, trn2.48xlarge, TP=16)

| Workload | tok/s (conc=1) | TPOT | TTFT |
|----------|:--------------:|:----:|:----:|
| short-short (128/128) | **74.5** | 13.5ms | 307ms |
| short-long (128/512) | **68.5** | 14.6ms | 308ms |
| long-short (2048/128) | **67.9** | 14.7ms | 474ms |
| long-long (2048/512) | **63.0** | 15.9ms | 474ms |

**GPU baseline**: BLOCKED (transformers 5.x `mistral4` config type incompatibility)

## Bug Fix: MLA Attention

**CRITICAL**: NxDI's stock `DeepseekV3Attention` has a bug that affects this model.

The `out_absorb` slicing in `modeling_deepseek.py` line 230 uses `wkv_b[:, self.v_head_dim:, :]`.
This is only correct when `v_head_dim == qk_nope_head_dim` (as in stock DeepSeek V3 where both = 128).
For Mistral-Small-4 (`v_head_dim=128, qk_nope_head_dim=64`), it causes a shape mismatch crash.

Without this fix, the model either crashes or produces garbage output with ~10 tok/s.
With the fix, performance is **74.5 tok/s** (6.9x improvement).

## Setup Steps

### 1. Download FP8 Model

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Download (113 GB FP8, skip consolidated format)
huggingface-cli download mistralai/Mistral-Small-4-119B-2603 \
    --token YOUR_HF_TOKEN \
    --local-dir /mnt/nvme/models/Mistral-Small-4-119B-2603 \
    --include "model-*.safetensors" "*.json" "tokenizer*"
```

### 2. Extract Text Model + Dequantize FP8→BF16

```bash
# Edit src/extract_text_model.py to set correct SRC_DIR and DST_DIR
python src/extract_text_model.py
# Output: 238 GB BF16, 543 tensors, 35 shards (~5 minutes)
```

### 3. Fix Tokenizer

```bash
python src/fix_tokenizer.py /mnt/nvme/models/Mistral-Small-4-119B-text-only
```

### 4. Apply Patches

```bash
# Fix MLA attention bug
python src/fix_mla_attention.py

# Fix MoE torch_block_wise forwarding (SDK 2.29)
python src/patch_moe.py

# Register custom model class
python src/register_model.py

# Install custom model class
SITE_PKGS=$(python -c "import neuronx_distributed_inference; print(neuronx_distributed_inference.__path__[0])")
cp src/modeling_deepseekv3_full.py $SITE_PKGS/models/deepseek/modeling_deepseekv3_full.py
```

### 5. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/Mistral-Small-4-119B-text-only \
    --tensor-parallel-size 16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --additional-config '{"override_neuron_config": {"blockwise_matmul_config": {"use_torch_block_wise": true}}}' \
    --port 8000
```

Compilation takes ~28 minutes.

## Instance Requirements

| Resource | Minimum |
|----------|---------|
| Instance type | trn2.48xlarge |
| TP degree | 16 |
| LNC | 2 (default) |
| HBM | ~238 GB (model) + KV cache |
| Storage | 1.5 TB (NVMe required for model + extraction workspace) |
| EBS | 300 GB |
| Compile time | ~28 minutes |

## Files

| File | Purpose |
|------|---------|
| `src/modeling_deepseekv3_full.py` | Custom 429-line NeuronDeepseekV3ForCausalLM model class |
| `src/extract_text_model.py` | FP8→BF16 dequantization + text extraction from multimodal |
| `src/fix_mla_attention.py` | Patch for MLA out_absorb slicing bug |
| `src/fix_tokenizer.py` | Tokenizer compatibility fix |
| `src/patch_moe.py` | MoE torch_block_wise forwarding (SDK 2.29) |
| `src/register_model.py` | Register deepseek_v3 model type in NxDI constants |

## Compatibility

| Instance/SDK | SDK 2.29 | SDK 2.28 |
|--------------|----------|----------|
| trn2.48xlarge (TP=16) | ✅ Working (all patches) | ✅ Working (no moe patch needed) |
| trn2.48xlarge (TP=32) | ✅ Working but slower | ✅ Working but slower |

## Known Issues

1. **HuggingFace token required**: Model is gated, needs `--token` for download
2. **Model name changed**: Was `Mistral-Small-4-119B-Instruct-2507`, now `-2603`
3. **NVMe required**: 238 GB model + 113 GB source = 351 GB minimum working space
4. **`--trust-remote-code` needed**: For tokenizer loading compatibility

## Maintainer

Agent Andretti - Mistral Family Benchmark Project

**Last Updated:** 2026-04-20
