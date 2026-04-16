# Contrib Model: Kimi-K2-Instruct-0905

NeuronX Distributed Inference implementation of Moonshot AI's Kimi-K2-Instruct-0905.

## Model Information

- **HuggingFace ID:** `moonshotai/Kimi-K2-Instruct-0905`
- **Model Type:** Mixture of Experts (MoE) decoder-only transformer
- **Architecture:** DeepSeek-V3 variant with MLA attention
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Total parameters | ~1,000B |
| Active parameters per token | ~32B |
| Hidden size | 7168 |
| Attention heads | 128 |
| Layers | 61 |
| Vocabulary size | 163840 |
| Routed experts | 384 (8 active per token) |
| Shared experts | 1 per MoE layer |
| Dense layers | 1 (layer 0, `first_k_dense_replace=1`) |
| Expert intermediate size | 2048 |
| Dense intermediate size | 18432 |
| Attention type | Multi-Latent Attention (MLA) |
| KV LoRA rank | 512 |
| QK rope head dim | 64 |
| Q LoRA rank | 1536 |
| RoPE | YaRN (factor=64, max_position_embeddings=262144) |
| Quantization | Blockwise FP8 (e4m3, 128x128 blocks) |
| Router activation | Sigmoid with `e_score_correction_bias` |
| Top-K normalization | Enabled (`norm_topk_prob=True`) |
| Routed scaling factor | 2.827 |

### Key Implementation Details

- **Multi-Latent Attention (MLA):** Compressed KV cache with only 576 bytes/token/layer
  (qk_rope_head_dim + kv_lora_rank = 64 + 512). Weight absorption is used to avoid
  decompressing KV during decode.

- **Blockwise FP8 Quantization:** Routed expert weights are kept in FP8 (e4m3) with
  128x128 block scales. Non-expert weights (attention, embeddings, shared experts, norms)
  are dequantized to BF16 during loading. Requires the
  `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` compiler flag.

- **Streaming Checkpoint Loader:** Custom `checkpoint_loader_fn` that processes the 62
  safetensor shards one at a time to avoid OOM on 2TB host RAM. Each shard is loaded,
  processed (FP8 handling, expert packing, router renaming), and accumulated.

- **Monkey Patches (applied during `load()`):**
  - `_apply_ep_scale_fix`: Prevents EP-sharding of per-channel FP8 scale tensors (shape [1,1,W]).
  - `_apply_blockwise_scale_stride_fix`: Forces stride=1 for blockwise scale partitioning.

- **Selective Loading Threshold:** Must be patched to 0.0 in
  `neuronx_distributed/modules/moe/model_utils.py` on the target instance to ensure all
  384 expert weights load correctly.

## Validation Results

**Validated:** 2026-04-16
**Configuration:** TP=64, EP=2, LNC=1, batch_size=1, seq_len=1024, blockwise FP8

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads on trn2.48xlarge |
| Generation | PASS | Correct answers for factual questions (10/13 prompts) |
| Throughput | PASS | 3.4 tok/s at BS=1 |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TPOT (per-token latency) | 297.5 ms |
| Throughput (BS=1) | 3.4 tok/s |
| TTFT (61 input tokens) | 1,788 ms |
| Compile time (total) | 73 min (TKG -O3: 49 min, CTE -O1: 24 min) |
| Model load time | 47 min |
| HBM utilization | ~78% (1,200 GB / 1,536 GB) |

### Token Generation Sweep (BS=1, seq_len=1024)

| Output Tokens | TTFT P50 (ms) | TPOT P50 (ms) | tok/s | E2E P50 (ms) |
|---------------|---------------|----------------|-------|---------------|
| 16 | 1,787.9 | 297.36 | 3.4 | 6,248.3 |
| 32 | 1,787.9 | 297.37 | 3.4 | 11,006.6 |
| 64 | 1,788.3 | 297.52 | 3.4 | 20,533.8 |
| 128 | 1,787.9 | 297.44 | 3.4 | 39,564.4 |
| 256 | 1,788.4 | 297.61 | 3.4 | 77,681.2 |
| 512 | 1,795.9 | 297.55 | 3.4 | 153,842.1 |

### Batching Results

Batching provides **zero throughput improvement** on this model. The MoE computation is
perfectly bandwidth-bound -- each TKG step must load all 192 local expert weight matrices
from HBM regardless of batch size. BS=4 TPOT scales linearly (1,191 ms), yielding the
same aggregate throughput as BS=1.

### Performance Bottleneck

TPOT breakdown (estimated per 297.5 ms token):

1. **MoE expert MLPs (~250 ms, ~84%):** 192 local experts x 2 matmuls per layer.
   FP8 weights are dequantized to BF16 before the NKI kernel.
2. **MLA attention (~25 ms, ~8%):** Weight absorption projections + KV cache.
3. **Router + all-to-all (~15 ms, ~5%):** Router TopK + expert dispatch across EP=2.
4. **Other (~7.5 ms, ~3%):** RMSNorm, residuals, lm_head.

Primary optimization opportunity: native blockwise FP8 kernel in the nki-lib MoE TKG
pipeline (currently blocked -- nki-lib requires per-channel FP8 scales).

## Usage

```python
import json
import os
import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig, RouterConfig

# Import model classes
from src.modeling_kimi_k2 import NeuronKimiK2ForCausalLM, KimiK2InferenceConfig

model_path = "/path/to/Kimi-K2-Instruct-0905"
compiled_path = "/path/to/compiled"

# Read HF config
with open(os.path.join(model_path, "config.json")) as f:
    hf_config = json.load(f)

# Configure for trn2.48xlarge
neuron_config = MoENeuronConfig(
    tp_degree=64,
    ep_degree=2,
    logical_nc_config=1,
    max_batch_size=1,
    seq_len=1024,
    n_active_tokens=128,
    torch_dtype="bfloat16",
    capacity_factor=1.0,
    glu_mlp=True,
    moe_ep_degree=2,
    moe_tp_degree=64,
    context_encoding_buckets=[128, 1024],
    router_config=RouterConfig(act_fn="sigmoid", dtype="float32"),
    # FP8 quantization
    quantized=True,
    quantized_checkpoints_path=model_path,
    quantization_dtype="f8e4m3",
    modules_to_not_convert=[
        "self_attn", "shared_experts", "embed_tokens",
        "lm_head", "norm", "router", "layers.0",
    ],
    quantization_type="blockwise_symmetric",
    quantization_block_axis=[1, 2],
    quantization_block_size=[128, 128],
)

# Build config from HF config fields
hf_kwargs = {k: v for k, v in hf_config.items()
             if k not in ("auto_map", "torch_dtype", "transformers_version", "architectures")}
config = KimiK2InferenceConfig(neuron_config=neuron_config, **hf_kwargs)

# Compile and load
model = NeuronKimiK2ForCausalLM(model_path, config)
model.compile(compiled_path)   # ~73 min
model.load(compiled_path)      # ~47 min

# Generate (CPU greedy sampling, no on-device sampling)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# See test/integration/test_model.py for the full generation loop
```

**Important:** Run with environment variables:
```bash
NEURON_LOGICAL_NC_CONFIG=1 LOCAL_WORLD_SIZE=128 python your_script.py
```

## Compatibility Matrix

| Instance / SDK Version | 2.28+ | 2.27 and earlier |
|------------------------|-------|------------------|
| trn2.48xlarge (LNC=1) | Working | Not tested |
| trn2.3xlarge | Not supported (needs TP=64, EP=2) | Not supported |
| trn1.32xlarge | Not supported (needs 128 cores) | Not supported |
| inf2 | Not supported | Not supported |

## Testing

Run integration tests on a trn2.48xlarge:

```bash
# Activate Neuron venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Run tests
NEURON_LOGICAL_NC_CONFIG=1 LOCAL_WORLD_SIZE=128 \
  pytest test/integration/test_model.py -v --capture=tee-sys
```

Or run standalone:

```bash
NEURON_LOGICAL_NC_CONFIG=1 LOCAL_WORLD_SIZE=128 \
  python test/integration/test_model.py
```

**Note:** Compilation takes ~73 min and loading takes ~47 min. The first run will compile
NEFFs to the compiled model path. Subsequent runs with existing NEFFs skip compilation.

## Prerequisites

1. **Selective loading threshold patch:** On the target instance, patch
   `neuronx_distributed/modules/moe/model_utils.py` to set the selective loading
   threshold to 0.0 (default is too high for 384 experts).

2. **Model weights:** Download from HuggingFace:
   ```bash
   huggingface-cli download moonshotai/Kimi-K2-Instruct-0905 \
     --local-dir /home/ubuntu/models/Kimi-K2-Instruct-0905
   ```

3. **Host RAM:** At least 2 TB (the streaming loader peaks at ~95 GB RSS, but
   safetensors mmap can use more virtual memory).

## Example Checkpoints

* [moonshotai/Kimi-K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905)

## Known Limitations

- **No on-device sampling:** The model uses CPU greedy sampling because the vocabulary
  size (163840) is not divisible by common TP degrees, causing shape mismatches in the
  on-device sampling kernel.

- **Elevated EOS logit:** The `<|im_end|>` token (ID 163586) has an elevated logit in
  early generation steps, likely due to the FP8->BF16 dequantization of shared expert
  weights or slight router bias approximation. Mitigated by masking EOS for the first
  few generation tokens (`min_tokens_before_eos=3`).

- **Batching does not improve throughput:** The MoE computation is bandwidth-bound
  (192 expert weight loads per step), so higher batch sizes increase latency linearly
  without improving aggregate throughput.

- **Compiler flags have no measurable impact:** -O3 with DGE vs -O1 showed 0% difference,
  confirming the bottleneck is weight bandwidth, not compute or scheduling.

## Maintainer

Annapurna Labs

**Last Updated:** 2026-04-16
