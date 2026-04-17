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

**Validated:** 2026-04-17
**Recommended Configuration:** TP=32, EP=2, LNC=2, batch_size=1, seq_len=1024, blockwise FP8

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads on trn2.48xlarge |
| Generation | PASS | Generates coherent text |
| Throughput | PASS | 5.2 tok/s at BS=1 (LNC=2) |

### Performance Metrics (Recommended: LNC=2, TP=32)

| Metric | Value |
|--------|-------|
| TPOT (per-token latency) | ~191 ms |
| Throughput (BS=1) | 5.2 tok/s |
| Compile time (total) | 67 min |
| Model load time | 30 min |
| HBM I/O utilization | 17.55 GB / 24 GB |

### LNC=2 vs LNC=1 Comparison

LNC=2 (TP=32, EP=2) is **53% faster** than LNC=1 (TP=64, EP=2) because each
logical core gets 2x HBM bandwidth, and MoE decode is purely bandwidth-bound.

| Config | TP | EP | Cores | TPOT | tok/s | Speedup |
|--------|----|----|-------|------|-------|---------|
| LNC=2 (recommended) | 32 | 2 | 64 | ~191 ms | 5.2 | **+53%** |
| LNC=1 | 64 | 2 | 128 | 297 ms | 3.4 | baseline |

### Batching Results

Batching provides **zero throughput improvement** on this model. The MoE computation is
perfectly bandwidth-bound -- each TKG step must load all 192 local expert weight matrices
from HBM regardless of batch size. BS=4 TPOT scales linearly (1,191 ms), yielding the
same aggregate throughput as BS=1.

### Performance Bottleneck

TPOT breakdown (estimated per ~191 ms token at LNC=2):

1. **MoE expert MLPs (~160 ms, ~84%):** 192 local experts x 2 matmuls per layer.
   FP8 weights are dequantized to BF16 before the NKI kernel.
2. **MLA attention (~16 ms, ~8%):** Weight absorption projections + KV cache.
3. **Router + all-to-all (~10 ms, ~5%):** Router TopK + expert dispatch across EP=2.
4. **Other (~5 ms, ~3%):** RMSNorm, residuals, lm_head.

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

# Configure for trn2.48xlarge (LNC=2, recommended)
neuron_config = MoENeuronConfig(
    tp_degree=32,
    ep_degree=2,
    logical_nc_config=2,
    max_batch_size=1,
    seq_len=1024,
    n_active_tokens=128,
    torch_dtype="bfloat16",
    capacity_factor=1.0,
    glu_mlp=True,
    moe_ep_degree=2,
    moe_tp_degree=32,
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
model.compile(compiled_path)   # ~67 min
model.load(compiled_path)      # ~30 min

# Generate (CPU greedy sampling, no on-device sampling)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# See test/integration/test_model.py for the full generation loop
```

**Important:** Run with environment variables:
```bash
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 python your_script.py
```

## Compatibility Matrix

| Instance / SDK Version | 2.28+ | 2.27 and earlier |
|------------------------|-------|------------------|
| trn2.48xlarge (LNC=2, recommended) | Working (5.2 tok/s) | Not tested |
| trn2.48xlarge (LNC=1) | Working (3.4 tok/s) | Not tested |
| trn2.3xlarge | Not supported (needs TP=32, EP=2 = 64 cores) | Not supported |
| trn1.32xlarge | Not supported (needs 64 cores at LNC=2) | Not supported |
| inf2 | Not supported | Not supported |

## Testing

Run integration tests on a trn2.48xlarge:

```bash
# Activate Neuron venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Run tests (LNC=2, recommended)
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
  pytest test/integration/test_model.py -v --capture=tee-sys
```

Or run standalone:

```bash
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
  python test/integration/test_model.py
```

**Note:** Compilation takes ~67 min and loading takes ~30 min. The first run will compile
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

**Last Updated:** 2026-04-17
