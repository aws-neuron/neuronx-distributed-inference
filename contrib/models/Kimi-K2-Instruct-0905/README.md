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

- **Selective Loading:** Uses the SDK default threshold (1.0). At EP=1, selective loading
  only loads the 8 active experts per token during TKG, producing a far simpler graph
  (6.2 min compile vs 3.5h) and 2.7x faster TPOT. Do NOT patch the threshold to 0.0.

## Validation Results

**Validated:** 2026-04-30 (SDK 2.29)
**Recommended Configuration:** TP=64, EP=1, LNC=2, batch_size=1, seq_len=512, blockwise FP8

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles and loads on trn2.48xlarge |
| Generation | PASS | Generates coherent text ("The capital of France is Paris.") |
| Coherence | PASS | Coherent quantum computing explanation |
| Throughput | PASS | 6.9 tok/s at BS=1 (LNC=2, EP=1) |

### Performance Metrics (Recommended: TP=64, EP=1, LNC=2)

| Metric | Value |
|--------|-------|
| TPOT (per-token latency) | 144.5 ms |
| Throughput (BS=1) | 6.9 tok/s |
| Compile time (CTE + TKG) | 16 min |
| Model load time | 17 min |

### Configuration Comparison

| Config | TP | EP | LNC | Cores | TPOT | tok/s | Compile | Speedup |
|--------|----|----|-----|-------|------|-------|---------|---------|
| **EP=1 selective (recommended)** | 64 | 1 | 2 | 64 | **144.5 ms** | **6.9** | **16 min** | **+105%** |
| EP=2 LNC=2 (previous) | 32 | 2 | 2 | 64 | 165.5 ms | 6.0 | 67 min | +76% |
| EP=2 LNC=1 (baseline) | 64 | 2 | 1 | 128 | 297.4 ms | 3.4 | ~60 min | baseline |

### Batching Results

Batching provides **zero throughput improvement** on this model. The MoE computation is
perfectly bandwidth-bound -- each TKG step must load 8 active experts' weight matrices
from HBM regardless of batch size. BS=4 TPOT scales linearly, yielding the
same aggregate throughput as BS=1.

### Performance Bottleneck

TPOT breakdown (estimated per ~144.5 ms token at TP=64, EP=1, LNC=2):

The decode step loads only 8 active experts per token (selective loading), but each expert's
weight matrices are TP-sharded across 64 cores. MoE decode remains bandwidth-bound — the
primary optimization opportunity is native per-channel FP8 support in the NKI kernel, which
could reduce weight transfer by ~50% and bring TPOT to ~22 ms (Task 017).

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

# Configure for trn2.48xlarge (TP=64, EP=1, LNC=2, recommended)
neuron_config = MoENeuronConfig(
    tp_degree=64,
    ep_degree=1,
    logical_nc_config=2,
    max_batch_size=1,
    seq_len=512,
    n_active_tokens=128,
    torch_dtype="bfloat16",
    capacity_factor=1.0,
    glu_mlp=True,
    moe_ep_degree=1,
    moe_tp_degree=64,
    context_encoding_buckets=[128, 512],
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
model.compile(compiled_path)   # ~16 min
model.load(compiled_path)      # ~17 min

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
| trn2.48xlarge (TP=64, EP=1, LNC=2, recommended) | **Working (6.9 tok/s)** | CTE compile fails (neuronx-cc 2.23 BIR error) | Not tested |
| trn2.48xlarge (TP=32, EP=2, LNC=2) | TKG working, CTE requires EP workaround* | Working (6.0 tok/s) | Not tested |
| trn2.48xlarge (TP=64, EP=2, LNC=1) | Not tested | Working (3.4 tok/s) | Not tested |
| trn2.3xlarge | Not supported (needs 64 cores) | Not supported | Not supported |
| trn1.32xlarge | Not supported (needs 64 cores at LNC=2) | Not supported | Not supported |
| inf2 | Not supported | Not supported | Not supported |

\*SDK 2.29 blockwise CTE kernels (shard_on_I, shard_on_block) produce wrong output with EP=2.
The `forward_all_experts_EP` workaround gives correct output but 7x slower TTFT. See SDK 2.29
Notes below for root cause analysis and workaround.

## Testing

Run integration tests on a trn2.48xlarge:

```bash
# Activate Neuron venv (SDK 2.29, recommended)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install tiktoken  # Required for tokenizer

# Run tests (TP=64, EP=1, LNC=2)
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
  pytest test/integration/test_model.py -v --capture=tee-sys
```

Or run standalone:

```bash
NEURON_LOGICAL_NC_CONFIG=2 LOCAL_WORLD_SIZE=64 \
  python test/integration/test_model.py
```

**Note:** Compilation takes ~16 min and loading takes ~17 min. The first run will compile
NEFFs to the compiled model path. Subsequent runs with existing NEFFs skip compilation.

## Prerequisites

1. **SDK 2.29:** Requires Neuron SDK 2.29 (neuronx-cc 2.24+). SDK 2.28 cannot compile
   CTE at EP=1 due to a BIR verification error in neuronx-cc 2.23.

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

SDK 2.29 (NxDI 0.9.17334) is the **recommended SDK** for this model. The EP=1 configuration
avoids the blockwise CTE kernel regressions that affected the EP=2 configuration on SDK 2.29.

### Historical: EP=2 Blockwise CTE Issues (Resolved by EP=1)

The previous EP=2 configuration was affected by SDK 2.29 removing the
`blockwise_mm_baseline_shard_hidden` NKI kernel used for MoE context encoding. The
replacement dynamic-while kernels (`shard_on_intermediate`, `shard_on_block`) produced
incorrect output for EP=2 / I_TP=64. A `forward_all_experts_EP` workaround provided
correct output but 7x slower TTFT (10,185 ms vs 1,420 ms).

**Resolution:** Switching to EP=1 (Task 016) eliminates expert parallelism entirely,
avoiding the broken blockwise CTE kernels. With `capacity_factor=1.0`, CTE uses the
`forward_capacity_factor` path which compiles correctly on neuronx-cc 2.24.

Detailed investigation of all blockwise kernel paths and the SDK 2.28 kernel porting
attempt is documented in the git history (commits prior to Task 016).

## Known Limitations

- **On-device sampling (ODS):** The model supports on-device sampling with `top_k=1`
  (greedy) for vocabulary size 163840. ODS avoids transferring full logits to CPU, but
  also means raw logits are not available for analysis during inference.

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

**Last Updated:** 2026-04-30
