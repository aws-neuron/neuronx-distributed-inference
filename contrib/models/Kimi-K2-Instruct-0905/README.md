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

**Validated:** 2026-04-18 (SDK 2.28 and 2.29)
**Recommended Configuration:** TP=32, EP=2, LNC=2, batch_size=1, seq_len=1024, blockwise FP8

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads on trn2.48xlarge |
| Generation | PASS | Generates coherent text |
| Throughput | PASS | 6.0 tok/s at BS=1 (LNC=2) |

### Performance Metrics (Recommended: LNC=2, TP=32)

| Metric | Value |
|--------|-------|
| TPOT (per-token latency) | 165.5 ms |
| Throughput (BS=1) | 6.0 tok/s |
| TTFT (61 input tokens) | 1,420 ms |
| Compile time (total) | 67 min |
| Model load time | 30 min |
| HBM I/O utilization | 17.55 GB / 24 GB |

### LNC=2 vs LNC=1 Comparison

LNC=2 (TP=32, EP=2) is **76% faster** than LNC=1 (TP=64, EP=2) because each
logical core gets 2x HBM bandwidth, and MoE decode is purely bandwidth-bound.

| Config | TP | EP | Cores | TPOT | tok/s | Speedup |
|--------|----|----|-------|------|-------|---------|
| LNC=2 (recommended) | 32 | 2 | 64 | 165.5 ms | 6.0 | **+76%** |
| LNC=1 | 64 | 2 | 128 | 297 ms | 3.4 | baseline |

### Token Generation Sweep (LNC=2, BS=1, seq_len=1024)

| Output Tokens | TTFT P50 (ms) | TPOT P50 (ms) | tok/s | E2E P50 (ms) |
|---------------|---------------|----------------|-------|---------------|
| 16 | 1,420.4 | 166.38 | 6.0 | 3,916.1 |
| 32 | 1,419.8 | 165.58 | 6.0 | 6,553.0 |
| 64 | 1,419.7 | 165.56 | 6.0 | 11,849.8 |
| 128 | 1,419.8 | 165.48 | 6.0 | 22,435.8 |
| 256 | 1,419.9 | 165.42 | 6.0 | 43,604.1 |
| 512 | 1,420.0 | 165.47 | 6.0 | 85,974.4 |

### Batching Results

Batching provides **zero throughput improvement** on this model. The MoE computation is
perfectly bandwidth-bound -- each TKG step must load all 192 local expert weight matrices
from HBM regardless of batch size. BS=4 TPOT scales linearly (1,191 ms), yielding the
same aggregate throughput as BS=1.

### Performance Bottleneck

TPOT breakdown (estimated per ~165.5 ms token at LNC=2):

1. **MoE expert MLPs (~139 ms, ~84%):** 192 local experts x 2 matmuls per layer.
   FP8 weights are dequantized to BF16 before the NKI kernel.
2. **MLA attention (~13 ms, ~8%):** Weight absorption projections + KV cache.
3. **Router + all-to-all (~8 ms, ~5%):** Router TopK + expert dispatch across EP=2.
4. **Other (~5.5 ms, ~3%):** RMSNorm, residuals, lm_head.

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

| Instance / SDK Version | 2.29 | 2.28 | 2.27 and earlier |
|------------------------|------|------|------------------|
| trn2.48xlarge (LNC=2, recommended) | Working (6.0 tok/s)* | Working (6.0 tok/s) | Not tested |
| trn2.48xlarge (LNC=1) | Not tested | Working (3.4 tok/s) | Not tested |
| trn2.3xlarge | Not supported (needs TP=32, EP=2 = 64 cores) | Not supported | Not supported |
| trn1.32xlarge | Not supported (needs 64 cores at LNC=2) | Not supported | Not supported |
| inf2 | Not supported | Not supported | Not supported |

\*SDK 2.29 requires a workaround for context encoding (see SDK 2.29 Notes below).

## Testing

Run integration tests on a trn2.48xlarge:

```bash
# Activate Neuron venv (SDK 2.28)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Or for SDK 2.29 (apply forward_blockwise workaround first, install tiktoken)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

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

## SDK 2.29 Notes

SDK 2.29 (NxDI 0.9.17334) introduces a new `forward_blockwise` code path for MoE context
encoding. The default kernel dispatch (`_call_shard_hidden_kernel`) is a stub that raises
`NotImplementedError`. While nkilib IS installed in the DLAMI (bundled version matches the
standalone `nki-library` April 2026 release), the available alternative kernels
(`shard_on_intermediate`, `shard_on_block`) are incompatible with this model's dimensions:

- `use_shard_on_intermediate_dynamic_while`: MLIR verification failure due to small per-TP
  intermediate dimension (64) not matching kernel tile expectations.
- `use_shard_on_block_dynamic_while` + `PING_PONG`: Compiles but produces incorrect outputs
  (likely due to blockwise FP8 scale dequantization interaction with the kernel).

**Recommended workaround:** Patch `expert_mlps_v2.py` in the `neuronx_distributed` package
to use `forward_all_experts_EP` instead of `forward_blockwise` when expert parallelism is
enabled:

```python
# In neuronx_distributed/modules/moe/expert_mlps_v2.py, in the forward() method,
# find the context encoding dispatch (around line 1497):
#     return self.forward_blockwise(...)
# Replace with:
if self.moe_expert_model_parallel_group.size() > 1:
    return self.forward_all_experts_EP(hidden_states, expert_affinities, expert_index)
return self.forward_blockwise(hidden_states, expert_affinities, expert_index, ...)
```

**Impact:** Token generation (TPOT) is unaffected (166.1 ms, identical to SDK 2.28). Context
encoding (TTFT) is ~7x slower (10,185 ms vs 1,420 ms) because `forward_all_experts_EP` sends
every token through every local expert rather than using the optimized blockwise dispatch. For
long-output workloads this is negligible; for TTFT-sensitive workloads, use SDK 2.28.

### SDK 2.29 Benchmark (LNC=2, TP=32, EP=2, BS=1, seq_len=1024)

| Output Tokens | TTFT P50 (ms) | TPOT P50 (ms) | tok/s | E2E P50 (ms) |
|---------------|---------------|----------------|-------|---------------|
| 16 | 10,184.9 | 166.27 | 6.0 | 12,678.9 |
| 32 | 10,185.1 | 166.13 | 6.0 | 15,335.3 |
| 64 | 10,184.6 | 166.10 | 6.0 | 20,651.2 |
| 128 | 10,184.7 | 166.15 | 6.0 | 31,286.4 |
| 256 | 10,184.5 | 166.03 | 6.0 | 52,522.3 |
| 512 | 10,184.6 | 166.06 | 6.0 | 95,040.9 |

**Additional SDK 2.29 setup:** Install `tiktoken` (`pip install tiktoken`) in the venv.

## Known Limitations

- **No on-device sampling:** The model uses CPU greedy sampling because the vocabulary
  size (163840) is not divisible by common TP degrees, causing shape mismatches in the
  on-device sampling kernel.

- **Elevated EOS logit:** The `<|im_end|>` token (ID 163586) has an elevated logit in
  early generation steps, likely due to the FP8->BF16 dequantization of shared expert
  weights or slight router bias approximation. Mitigated by masking EOS for the first
  few generation tokens (`min_tokens_before_eos=3`).

- **Batching does not improve throughput:** NxDI compiles HLO with per-sequence shapes
  (`[1, seq_len]` for CTE, `[1, 1]` for TKG) regardless of `max_batch_size`. Multiple
  sequences in a batch are processed sequentially through the same NEFF. Combined with
  the bandwidth-bound nature of MoE (192 expert weight loads per decode step), BS>1
  provides no aggregate throughput benefit. Verified: BS=2 compile produces identical
  NEFF shapes to BS=1.

- **Compiler flags have no measurable impact:** -O3 with DGE vs -O1 showed 0% difference,
  confirming the bottleneck is weight bandwidth, not compute or scheduling.

## Maintainer

Annapurna Labs

**Last Updated:** 2026-04-20
