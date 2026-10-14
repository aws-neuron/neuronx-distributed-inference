# Contrib Model: Mixtral 8x22B Instruct v0.1

NeuronX Distributed Inference implementation of Mixtral 8x22B Instruct v0.1 on trn2.48xlarge.

## Model Information

- **HuggingFace ID:** `mistralai/Mixtral-8x22B-Instruct-v0.1`
- **Model Type:** Mixture-of-Experts (8 experts, top-2 routing)
- **Parameters:** 141B total, ~39B active per token
- **License:** Apache 2.0

## Architecture Details

- 56 transformer layers
- hidden_size: 6144, intermediate_size: 16384
- 48 attention heads, 8 KV heads (GQA)
- 8 experts per layer, top-2 routing (softmax)
- head_dim: 128, vocab_size: 32768

## Performance (SDK 2.29, vLLM 0.16.0, trn2.48xlarge)

| Workload | tok/s (conc=1) | TPOT | TTFT | GPU tok/s | GPU/Neuron |
|----------|:--------------:|:----:|:----:|:---------:|:----------:|
| short-short (128/128) | **25.8** | 38.8ms | 185ms | 97.3 | 3.8x |
| short-long (128/512) | **25.4** | 39.4ms | 185ms | 96.8 | 3.8x |
| long-short (2048/128) | **25.4** | 39.4ms | 555ms | 97.2 | 3.8x |
| long-long (2048/512) | **25.1** | 39.8ms | 555ms | 96.5 | 3.8x |

**GPU baseline**: 4x H100 (TP=4), vLLM 0.19.0

### SDK 2.29 vs 2.28

| Metric | SDK 2.28 | SDK 2.29 | Improvement |
|--------|:--------:|:--------:|:-----------:|
| tok/s (short-short) | 24.9 | **25.8** | +4% |
| tok/s (long-short) | 21.6 | **25.4** | +18% |
| TPOT (short) | 40.2ms | 38.8ms | -3% |

Notable improvement for long inputs: SDK 2.28 showed significant long-input degradation (24.9→21.6 tok/s, -13%), while SDK 2.29 is nearly flat (25.8→25.4, -2%).

## SDK 2.29 Required Workaround

Same as Mixtral 8x7B. See `src/patch_moe.py`.

**Two-part fix:**

1. **Patch moe.py** to forward `use_torch_block_wise` to ExpertMLPs:
```bash
python src/patch_moe.py
```

2. **Pass torch_block_wise config to vLLM**:
```bash
--additional-config '{"override_neuron_config": {"blockwise_matmul_config": {"use_torch_block_wise": true}}}'
```

## TKG Optimization: Not Applicable

TKG cannot be used on this model:
- kv_heads/TP = 8/16 = 0.5 → SHARD_OVER_HEADS mode
- Stock TKG kernel requires kv_heads/TP >= 1
- Even if applicable, MoE expert dispatch dominates TPOT (same finding as 8x7B)

## Quick Start

```bash
# Activate venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Apply moe.py patch (SDK 2.29 only)
python src/patch_moe.py

# Download model (262 GB -- use NVMe if available)
# Mount NVMe first on trn2.48xlarge:
#   sudo mkfs.ext4 /dev/nvme1n1
#   sudo mount /dev/nvme1n1 /mnt/nvme
huggingface-cli download mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --local-dir /mnt/nvme/models/Mixtral-8x22B-Instruct-v0.1

# Symlink HF cache to NVMe (vLLM saves a local copy during compilation)
ln -sf /mnt/nvme/local-models /home/ubuntu/local-models

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/nvme/models/Mixtral-8x22B-Instruct-v0.1 \
    --tensor-parallel-size 16 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --additional-config '{"override_neuron_config": {"blockwise_matmul_config": {"use_torch_block_wise": true}}}' \
    --disable-log-requests \
    --port 8000
```

## Instance Requirements

| Resource | Minimum |
|----------|---------|
| Instance type | trn2.48xlarge |
| TP degree | 16 |
| LNC | 2 (default) |
| HBM | ~262 GB (model) + KV cache |
| Storage | 1 TB+ (model = 262 GB, use NVMe) |
| EBS | 500 GB minimum |
| Compile time | ~40 minutes |

**Storage note**: The 262 GB model weights exceed typical EBS volumes. Use the NVMe instance store (4x 1.7 TB on trn2.48xlarge) for model storage.

## Compatibility

| Instance/SDK | SDK 2.29 | SDK 2.28 |
|--------------|----------|----------|
| trn2.48xlarge (TP=16) | ✅ Working (with patch) | ✅ Working (no patch needed) |
| trn2.48xlarge (TP=8) | ❌ OOM | ❌ OOM |
| trn2.3xlarge | ❌ OOM | ❌ OOM |

## Key Flags

- `--no-enable-prefix-caching`: Required to avoid OOB crash in block KV cache path
- `--additional-config '{"override_neuron_config": ...}'`: Required on SDK 2.29
- `--max-num-seqs 4`: Recommended for stable performance
- `--tensor-parallel-size 16`: Required (model too large for TP=8)

## Maintainer

Annapurna Labs / Agent Andretti (Mistral Family Benchmark Project)

**Last Updated:** 2026-04-20
