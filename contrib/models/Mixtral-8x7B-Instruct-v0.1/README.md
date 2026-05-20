# Contrib Model: Mixtral 8x7B Instruct v0.1

NeuronX Distributed Inference implementation of Mixtral 8x7B Instruct v0.1 on trn2.48xlarge.

## Model Information

- **HuggingFace ID:** `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Model Type:** Mixture-of-Experts (8 experts, top-2 routing)
- **Parameters:** 46.7B total, ~12.9B active per token
- **License:** Apache 2.0

## Architecture Details

- 32 transformer layers
- hidden_size: 4096, intermediate_size: 14336
- 32 attention heads, 8 KV heads (GQA)
- 8 experts per layer, top-2 routing (softmax)
- head_dim: 128, vocab_size: 32000

## Performance (SDK 2.29, vLLM 0.16.0, trn2.48xlarge)

| Workload | tok/s (conc=1) | TPOT | TTFT | GPU tok/s | GPU/Neuron |
|----------|:--------------:|:----:|:----:|:---------:|:----------:|
| short-short (128/128) | **40.4** | 24.9ms | 130ms | 123.8 | 3.1x |
| short-long (128/512) | **39.6** | 25.3ms | 130ms | 123.3 | 3.1x |
| long-short (2048/128) | **39.6** | 25.3ms | 370ms | 123.1 | 3.1x |
| long-long (2048/512) | **39.2** | 25.5ms | 370ms | 122.7 | 3.1x |

**GPU baseline**: 2x H100 (TP=2), vLLM 0.8.5

### SDK 2.29 vs 2.28

| Metric | SDK 2.28 | SDK 2.29 | Improvement |
|--------|:--------:|:--------:|:-----------:|
| tok/s (short-short) | 38.5 | **40.4** | +5% |
| TPOT | 26.0ms | 24.9ms | -4% |

The improvement comes from `torch_block_wise` MoE implementation replacing the broken NKI blockwise kernel.

## SDK 2.29 Required Workaround

**CRITICAL**: NKI 0.3.0 GA (SDK 2.29) removed `neuronxcc.nki._private.blockwise_mm`. Without the workaround, MoE models crash with `NotImplementedError: _call_shard_hidden_kernel is not available`.

**Two-part fix:**

1. **Patch moe.py** to forward `use_torch_block_wise` to ExpertMLPs:
```bash
python src/patch_moe.py
```

2. **Pass torch_block_wise config to vLLM**:
```bash
--additional-config '{"override_neuron_config": {"blockwise_matmul_config": {"use_torch_block_wise": true}}}'
```

Both steps are required. The patch only needs to be applied once per environment.

## TKG Optimization: Not Applicable

TKG was tested on this model (kv_heads/TP = 8/8 = 1, stock TKG eligible) but provides **no benefit**:
- Baseline: 40.4 tok/s, 24.9ms TPOT
- TKG: 40.3 tok/s, 25.0ms TPOT (+0%)

**Root cause**: MoE expert dispatch dominates TPOT (~60% of decode time). The attention kernel is not the bottleneck for MoE models.

## Quick Start

```bash
# Activate venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Apply moe.py patch (SDK 2.29 only)
python src/patch_moe.py

# Download model
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --local-dir /mnt/models/Mixtral-8x7B-Instruct-v0.1

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/models/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 8 \
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
| TP degree | 8 |
| LNC | 2 (default) |
| HBM | ~93 GB (model) + KV cache |
| EBS | 300 GB |
| Compile time | ~25 minutes |

**Note**: This model does NOT fit on trn2.3xlarge (93 GB > 96 GB total HBM at TP=4).

## Compatibility

| Instance/SDK | SDK 2.29 | SDK 2.28 |
|--------------|----------|----------|
| trn2.48xlarge | ✅ Working (with patch) | ✅ Working (no patch needed) |
| trn2.3xlarge | ❌ OOM | ❌ OOM |
| trn1.32xlarge | ✅ Working (TP=5-8) | ✅ Working |

## Validation Results (Legacy, Trn1)

**Validated:** 2026-01-29 on trn1.32xlarge
**Configuration:** TP=5, batch_size=None, seq_len=None

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ✅ PASS | **100.0% match** |
| Throughput | 5.28 tok/s (trn1, TP=5) |

## Key Flags

- `--no-enable-prefix-caching`: Required to avoid OOB crash in block KV cache path
- `--additional-config '{"override_neuron_config": ...}'`: Required on SDK 2.29
- `--max-num-seqs 4`: Recommended for stable performance

## Maintainer

Annapurna Labs / Agent Andretti (Mistral Family Benchmark Project)

**Last Updated:** 2026-04-20
