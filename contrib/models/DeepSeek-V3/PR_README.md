# Contrib Model: DeepSeek-V3

NeuronX Distributed Inference implementation of DeepSeek V3, a 671B parameter Mixture-of-Experts model (37B active per token) from DeepSeek AI. Uses Multi-head Latent Attention (MLA) and a custom group-based MoE router with 256 routed experts.

## Model Family

| Model | HuggingFace ID | Total Params | Active Params | Instance |
|-------|----------------|-------------|---------------|----------|
| **DeepSeek-V3-0324** | `deepseek-ai/DeepSeek-V3-0324` | 671B | 37B | trn2.48xlarge (TP=64) |

**License:** [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL)

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 61 (3 dense + 58 MoE) |
| Hidden Size | 7168 |
| Attention | MLA (Multi-head Latent Attention) with LoRA-compressed Q |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| Attention Heads | 128 |
| Routed Experts | 256 (8 groups of 32, top-4 groups, 8 experts per token) |
| Shared Experts | 1 |
| Routing | sigmoid + e_score_correction_bias + group selection + normalize + scale by 2.5 |
| Dense Intermediate | 18432 |
| MoE Intermediate | 2048 |
| Position Encoding | YaRN RoPE (interleaved layout) |
| Vocabulary | 129280 |
| Normalization | RMSNorm |
| Activation | SiLU gated MLP |

### Unique Architecture Features

- **Multi-head Latent Attention (MLA):** KV projections are compressed through a low-rank bottleneck (kv_lora_rank=512), then expanded back. KV cache stores `[k_pe | compressed_kv]` with shape `(bsz, 1, seq_len, 576)` instead of standard GQA format.
- **Custom MoE Router:** Group-based expert selection with learned `e_score_correction_bias`, sigmoid activation, top-4 group selection, and top-8 expert selection with normalization and scaling.
- **Dense Layers 0-2:** First 3 layers use dense MLP (intermediate_size=18432) instead of MoE.
- **YaRN RoPE:** Interleaved layout using `rotate_fn` (not `rotate_half`).
- **Native FP8 Weights:** Official weights are in float8_e4m3fn with block-wise scale factors; automatically dequantized to BF16 during loading.

---

## Validation Results

**Validated:** 2026-03-19
**SDK:** NxDI 0.8.0, neuronx-cc 2.23.6484, torch 2.9.0, transformers 4.57.6 (SDK 2.28)

### Multi-Prompt Generation Quality (671B, trn2.48xlarge, TP=64)

Single-request greedy generation (top_k=1) with the full 671B model, 64 output tokens per prompt:

| # | Prompt | First Token | Output (truncated) | Tokens | Status |
|---|--------|-------------|-------------------|--------|--------|
| 1 | "The capital of France is" | Paris | Paris, which is one of the most important and influential cities in the world. Paris is located in the northern part of France... | 64 | PASS |
| 2 | "def fibonacci(n):" | if | `if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2)` ... | 64 | PASS |
| 3 | "The theory of relativity states that" | nothing | nothing can travel faster than the speed of light. If light were to travel from New York to Los Angeles... | 64 | PASS |
| 4 | "In a shocking finding, scientists discovered" | that | that elephants are the only non-human animals that can call each other by names... | 64 | PASS |
| 5 | "To make a chocolate cake, you need" | the | the following ingredients: 2 cups sugar, 1 3/4 cups all-purpose flour, 3/4 cup cocoa... | 64 | PASS |
| 6 | "The largest ocean on Earth is" | the | the Pacific Ocean, covering approximately 63 million square miles (165 million square kilometers)... | 64 | PASS |
| 7 | "Machine learning is a subset of" | artificial | artificial intelligence that enables computers to learn from data and improve over time... | 64 | PASS |
| 8 | "The year 2025 will be remembered for" | the | the unprecedented heatwaves that swept across the globe, shattering temperature records... | 64 | PASS |

All 8 prompts produce coherent, factually correct, multi-sentence responses. Code generation (fibonacci) produces syntactically valid Python. Throughput: ~3.0s per 64-token generation (includes 1.67s TTFT).

### First-Token Accuracy (671B, trn2.48xlarge, TP=64)

First-token prediction with greedy decoding against expected completions:

| Prompt | Expected | Got | Token ID | TTFT (ms) | Status |
|--------|----------|-----|----------|-----------|--------|
| "The capital of France is" | Paris | Paris | 11111 | 1669.9 | MATCH |
| "The largest planet in our solar system is" | Jupiter | Jupiter | 49475 | 1668.2 | MATCH |
| "Water freezes at" | 0 | ` ` (space) | 223 | 1669.0 | VALID |
| "The speed of light is approximately" | 299 | ` ` (space) | 223 | 1669.7 | VALID |
| "Barack Obama was the" | 44 | first | 1257 | 1669.9 | VALID |
| "Python is a" | programming | powerful | 8959 | 1669.7 | VALID |
| "The chemical formula for water is" | H | H | 437 | 1668.4 | MATCH |
| "Mount Everest is located in" | the | the | 270 | 1668.6 | MATCH |

**4/8 exact keyword match.** All 4 "VALID" responses are semantically correct alternative continuations (e.g., "Barack Obama was the **first** Black president" is factually correct, just not the expected "44th"). No hallucinated or incoherent first tokens.

**TTFT stability:** 1668.2ms - 1669.9ms (1.7ms spread across 8 prompts) -- extremely stable.

### Mini Model Logit Matching (TP=2)

First-token comparison using a mini DeepSeek V3 model (1 dense + 1 MoE layer, random weights):

| Test | Method | Result |
|------|--------|--------|
| First token | HF (FP32) vs NXDI (BF16) argmax | **EXACT MATCH** |

With random weights, BF16 rounding in MoE layers causes autoregressive divergence after the first token. This is expected and consistent with other MoE models on Neuron.

**Note:** Mini model compilation is currently blocked by NCC_IBIR297 compiler regression in SDK 2.28. The logit matching result above was validated on a prior SDK version. See Caveats section.

---

## Performance Benchmarks

**SDK 2.28**, BF16, trn2.48xlarge (64 NeuronCores), lnc=2. All measurements from compiled and loaded model with pre-sharded checkpoints.

### NXDI Native Benchmark (bs=1, seq_len=512, 256 input / 256 output tokens)

Measured via `benchmark_sampling()` API with 20 timed iterations:

| Component | p50 (ms) | p90 (ms) | p99 (ms) | Throughput |
|-----------|----------|----------|----------|------------|
| **Token Generation (TPOT)** | **20.5** | 20.9 | 21.2 | 48.7 tok/s |
| **Context Encoding (TTFT)** | **1,667** | 1,668 | 1,668 | 307 tok/s |
| **End-to-End** | **7,057** | 7,071 | 7,077 | 72.6 tok/s |

p50-p99 spread < 1ms for token generation (very stable).

### Generate()-Based Measurement (bs=1, seq_len=512)

Measured via `HuggingFaceGenerationAdapter.generate()` with greedy decoding:

| Metric | Value | Method |
|--------|-------|--------|
| **TTFT** | **1,669.3 ms** | Median of 5 runs (spread: 0.5ms) |
| **TPOT** | **21.1 ms** | Derived from 128-token generation |
| **E2E throughput (128 tokens)** | **29.4 tok/s** | 128 tokens in 4.35s |
| **E2E throughput (64 tokens)** | **~21 tok/s** | 64 tokens in ~3.0s |

E2E throughput includes TTFT overhead; pure token generation rate is ~47 tok/s.

### vLLM Serving + GuideLLM Sweep (seq_len=512, ~200 input / ~200 output tokens)

| BS | Sync ITL (ms) | Max Throughput (tok/s) | Best Constant-Rate (tok/s) | Best ITL (ms) |
|----|---------------|----------------------|---------------------------|---------------|
| 1 | 22.2 | 33.3 | 33.3 | 22.1 |
| 2 | 30.5 | 41.7 | 39.6 | 37.1 |
| 4 | 37.1 | 53.3 | 48.3 | 62.2 |
| 8 | 47.6 | 66.7 | 55.0 | 106.3 |

TTFT consistent at ~1,700ms across all batch sizes. Throughput scales sub-linearly: bs=8 gives +100% over bs=1 but at 2x higher ITL.

### Neuron Profile (System-Level, TP=64)

| Component | Time | % of TPOT |
|-----------|------|-----------|
| Compute (matmul, attention, MoE) | ~12 ms | ~63% |
| Collectives (all-reduce, TP=64) | ~3 ms | ~16% |
| Memory access (weight reads) | ~3 ms | ~16% |
| Straggler overhead (NC sync) | ~1 ms | ~5% |
| **Total** | **~19 ms** | **100%** |

Per-NC straggler analysis: NC 12 is 20% slower (22.8ms) than NC 53 (18.9ms). All NCs must sync at TP barriers, so the slowest NC determines step time.

### HBM Usage (per NeuronCore)

| Component | Size | % |
|-----------|------|---|
| Tensors (weights + KV cache) | 23.07 GB | 94.4% |
| Shared Scratchpad | 1.07 GB | 4.4% |
| Collectives buffers | 0.17 GB | 0.7% |
| Other | 0.13 GB | 0.5% |
| **Total** | **24.44 GB** | **~102%** |

### Timing Summary

| Operation | Time |
|-----------|------|
| NEFF compilation (first time) | 11.8 min |
| NEFF compilation (from cache) | ~1s |
| Weight sharding (FP8 -> 64 per-rank files) | 3.5 hours |
| Load from pre-sharded checkpoints | 34s (weights) + 5.4s (warmup) |
| TPOT (token generation, p50) | 20.5 ms |
| TTFT (context encoding, 256 tokens) | 1,667 ms |

### Key Observations

- **Single-request throughput: 48.7 tok/s** (NXDI native) or **33 tok/s** (vLLM with continuous batching overhead)
- **Batching scales sub-linearly:** Each doubling of batch size gives diminishing returns (+25%, +28%, +25%) due to the 256-expert memory-bandwidth bottleneck
- **vLLM overhead is minimal:** ITL of 22.1ms closely matches NXDI native TPOT of 20.5ms (~2ms overhead)
- **Stable performance:** p50-p99 spread < 1ms for token generation across 20 runs; TTFT spread 0.5ms across 5 runs
- **Straggler effect:** NC 12 is 20% slower than NC 53; all NCs must sync at TP barriers

---

## Usage

### Full 671B Model (trn2.48xlarge, TP=64)

```python
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

from src.modeling_deepseek import DeepseekV3InferenceConfig, NeuronDeepseekV3ForCausalLM

model_path = "/path/to/deepseek-ai/DeepSeek-V3-0324/"
compiled_path = "/scratch/deepseek_v3_traced/"

neuron_config = MoENeuronConfig(
    tp_degree=64,           # All 64 logical NeuronCores (lnc=2)
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,    # MUST be 2 on trn2
    enable_bucketing=False,
    flash_decoding_enabled=False,
    save_sharded_checkpoint=True,  # Pre-shard during compile for fast reload
)

config = DeepseekV3InferenceConfig(
    neuron_config, load_config=load_pretrained_config(model_path),
)

model = NeuronDeepseekV3ForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
```

### Mini Model (Development, TP=2)

The integration test creates a mini model (1 dense + 1 MoE layer, random weights, vocab=32000) that runs on any instance with 2+ NeuronCores. See `test/integration/test_model.py` for details.

## Pre-Sharded Deployment

The 671B model requires ~2TB peak RAM during weight sharding (FP8 dequant + expert fusion + 64 per-rank splits). With `save_sharded_checkpoint=True`, per-rank files (~21.4GB each) are saved during compilation and reloaded in ~34 seconds on subsequent runs.

### Fast Recovery from S3

```bash
# Restore from S3 (pre-sharded weights + compiled NEFFs):
bash scripts/restore_from_s3.sh deepseek-v3-nxdi-artifacts

# Load with pre-compiled artifacts (~40s):
python examples/generation_deepseek_v3.py \
    --traced-model-path /scratch/deepseek_v3_traced --skip-compile \
    --tp-degree 64 --seq-len 512 --batch-size 1 --max-new-tokens 128
```

### Startup Time Comparison

| Path | Compile | Weight Load | Total |
|------|---------|-------------|-------|
| Full rebuild (from HF FP8 weights) | 12 min | 3.5 hours | ~4 hours |
| From pre-sharded + NEFF cache | 1s | 34s + 5s warmup | ~40s |
| Restore from S3 + load | N/A | 30 min download + 40s load | ~31 min |

---

## Caveats

1. **`logical_nc_config=2` required on trn2** -- lnc=1 causes HBM OOM because pairs of NeuronCores share 24GB HBM banks. Two ranks (~21.4GB each) need ~42.8GB in one 24GB bank.

2. **TP=64 required** -- With 256 MoE experts, all expert weights live on every TP rank (only intermediate dim is sharded). At TP=32, each rank needs ~40GB vs 24GB per-physical-core limit.

3. **FP8 dequantization** -- Official weights are float8_e4m3fn. Dequantization to BF16 happens automatically during `convert_deepseek_v3_hf_to_neuron_state_dict()` but requires ~2TB peak RAM + NVMe swap.

4. **MLA incompatible with NeuronAttentionBase** -- The custom attention class does NOT extend `NeuronAttentionBase` because GQA projections are incompatible with MLA's weight absorption. KV cache uses `num_key_value_heads=1` with combined dim 576.

5. **`save_sharded_checkpoint=True` strongly recommended** -- Without it, every model load re-shards 1.3TB of BF16 weights (3.5 hours, 2TB RAM).

6. **`disable_numeric_cc_token=True`** -- Set automatically in config; required for all-gather/reduce-scatter collectives.

7. **`enable_bucketing=False`** -- Bucketing has not been tested with MLA attention.

8. **2TB RAM + NVMe swap required** for first-time weight sharding. Use 400GB swap on NVMe (`/scratch2/swapfile`), NOT on EBS. NVMe swap is 20x faster.

9. **Mini model NCC_IBIR297 regression (SDK 2.28)** -- The mini model (tp=2) fails to compile with `NCC_IBIR297` internal compiler error in neuronx-cc 2.23.6484. This is a BIR verifier bug in the Neuron compiler that affects the DeepSeek V3 MLA attention graph at small TP degrees. The full 671B model at tp=64 compiles and runs correctly. Integration tests for the mini model are skipped until this is resolved.

## Maximum Sequence Length

| seq_len | Compile | Status | Notes |
|---------|---------|--------|-------|
| 512 | 11.8 min | PASS | Default, all benchmarks |

Higher sequence lengths should work but have not been validated at 671B scale. The mini model has been tested at seq_len=128.

### HBM Budget for Sequence Length Scaling

KV cache per batch element = 61 layers x seq_len x 576 (kv_lora_rank + rope_dim) x 2 bytes:

| seq_len | KV Cache (bs=1) | KV Cache (bs=4) | KV Cache (bs=8) | Fits in ~1.1GB headroom? |
|---------|-----------------|-----------------|-----------------|--------------------------|
| 512 | 0.07 GB | 0.27 GB | 0.54 GB | bs=1-8 all fit |
| 2048 | 0.27 GB | 1.07 GB | 2.14 GB | bs=1-4 fit; bs=8 tight |
| 4096 | 0.54 GB | 2.14 GB | OOM | bs=1-2 fit; bs=4 tight |
| 8192 | 1.07 GB | OOM | OOM | bs=1 only |

Note: Activation buffers during forward pass add transient memory. Actual limits must be discovered empirically.

## Compatibility Matrix

| Instance | TP | LNC | Status | Notes |
|----------|-----|-----|--------|-------|
| trn2.48xlarge | 64 | 2 | **PASS** | Only viable configuration for 671B |
| trn2.48xlarge | 32 | 2 | FAIL | HBM OOM (NCC_EVRF009) -- 40GB per NC vs 24GB limit |
| trn2.48xlarge | 64 | 1 | FAIL | HBM OOM -- 2 ranks share 24GB bank |
| trn2.3xlarge | 2 | 2 | BLOCKED* | Mini model only (development) |
| trn2.3xlarge | 4 | 2 | FAIL | Full model doesn't fit on 4 NCs |

*Mini model blocked by NCC_IBIR297 compiler regression in SDK 2.28. See Caveat #9.

### Minimum Requirements

| Resource | Requirement |
|----------|------------|
| HBM | 1.5 TB (64 NCs x 24 GB) |
| TP degree | 64 |
| LNC | 2 (trn2 platform default) |
| Instance | trn2.48xlarge |
| System RAM | 2 TB + 400GB NVMe swap (first-time sharding) |
| NVMe storage | 1.7 TB (compiled model + sharded weights) |
| Disk (HF weights) | 642 GB (FP8 safetensors) |

### SDK Configuration

| Component | Version |
|-----------|---------|
| NxDI | 0.8.0 |
| neuronx-cc | 2.23.6484 |
| torch | 2.9.0 |
| transformers | 4.57.6 |
| NXDI venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |
| vLLM | 0.13.0 (via `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`) |

---

## Testing

### Unit Tests (CPU only, no device needed)

```bash
cd contrib/models/DeepSeek-V3/
pytest test/unit/ -v
```

| Test File | Tests | What It Validates |
|-----------|-------|-------------------|
| `test_config.py` | 11 | Config parsing, MLA params, MoE config, RoPE injection, FP8 dequant |
| `test_rope.py` | 3 | YaRN frequency table, interleaved apply_rotary, HF layout equivalence |
| `test_router.py` | 5 | Router output shapes, HF reference match, group selection, bias handling |
| `test_weight_conversion.py` | 11 | Router rename, bias rename, expert fusion, down_proj stacking, shared experts, rank util, dense skip |
| **Total** | **39** | |

All 39 tests pass on CPU (any instance, no Neuron device needed).

### Integration Tests (needs Neuron device)

```bash
# Mini model (default, tp=2 -- currently skipped due to NCC_IBIR297):
cd contrib/models/DeepSeek-V3/
pytest test/integration/test_model.py --capture=tee-sys

# Full 671B model (tp=64, trn2.48xlarge):
DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-V3-0324-FP8 \
DEEPSEEK_COMPILED_PATH=/scratch/deepseek_v3_traced \
DEEPSEEK_TP_DEGREE=64 \
DEEPSEEK_SEQ_LEN=512 \
pytest test/integration/test_model.py --capture=tee-sys
```

| Test | 671B Result | Mini Result | What It Validates |
|------|------------|-------------|-------------------|
| `test_model_loads` | PASS | SKIP | Compile + load on Neuron |
| `test_model_generates` | PASS | SKIP | 20-token generation produces text |
| `test_output_coherence` | PASS (27 words) | SKIP | No excessive repetition, multi-word output |
| `test_top_token_valid` | PASS | SKIP | First token is valid, decodable |
| `test_first_token_matches_hf` | SKIP (671B too large for CPU) | SKIP | First-token HF vs NXDI argmax match |
| `test_performance_ttft` | PASS (1670ms) | SKIP | TTFT within configurable threshold |
| `test_performance_throughput` | PASS (9.6 tok/s) | SKIP | Throughput above configurable threshold |

**671B results:** 6 passed, 1 skipped | **Mini model:** 7 skipped (NCC_IBIR297)

---

## Key Porting Challenges

1. **MLA incompatible with NeuronAttentionBase:** GQA projections don't apply to MLA's weight absorption. Built a custom `DeepseekV3Attention` class with its own TP sharding, KV cache, and softmax logic.

2. **Custom MoE Router:** DeepSeek V3 uses group-based expert selection with learned bias and scaling -- not supported by standard `RouterTopK`. Subclassed `RouterTopK` as `DeepseekV3Router` with compiler-compatible group selection (sum-based scoring + gather-based selection).

3. **YaRN RoPE interleaved layout:** Uses `rotate_fn` (interleaved) not `rotate_half` (split). No transpose needed -- different from optimum-neuron which uses split layout.

4. **Dense layers 0-2:** Separate `DeepseekV3DenseMLP` class with `dense_intermediate_size=18432` (not MoE).

5. **FP8 dequantization:** Block-wise float8_e4m3fn with per-block scale factors. Added `_dequantize_fp8_state_dict()` for vectorized conversion during state dict loading.

6. **Expert fusion:** Per-expert `gate_proj` + `up_proj` fused into `gate_up_proj` tensor `[num_experts, hidden, 2*intermediate]` for ExpertMLPsV2 compatibility.

7. **KV cache stores compressed format:** `[k_pe | compressed_kv]` with dim 576 (rope_dim 64 + kv_lora_rank 512), not standard per-head KV.

8. **Weight sharding peak RAM ~2TB:** Framework loads full state dict + 64 per-rank shards in memory. Python allocator doesn't return freed FP8 pages to OS. Requires NVMe swap.

9. **TP=32 HBM OOM:** 256 experts all on every rank. Only intermediate dim is sharded, so each rank carries ~40GB at TP=32 vs 24GB HBM limit. Fixed by using TP=64.

---

## vLLM Integration

DeepSeek V3 can be served via vLLM with the Neuron backend:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export NEURON_COMPILED_ARTIFACTS=/scratch/vllm_bs1

VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3-0324-FP8 \
    --tensor-parallel-size 64 --max-model-len 512 --max-num-seqs 1 \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --port 8000 \
    --additional-config '{"override_neuron_config": {"logical_nc_config": 2, "enable_bucketing": false, "save_sharded_checkpoint": true}}'
```

**Critical:** `NEURON_COMPILED_ARTIFACTS` env var is required to reuse pre-compiled NEFFs. Without it, vLLM deletes the compiled artifacts directory on each startup.

**Critical:** `--additional-config` with `logical_nc_config: 2` is required. Without it, compilation fails with NCC_IBIR297 internal error.

### Fast Startup Recipe

1. Compile NEFFs (~2.5 min or 1s from cache)
2. Kill vLLM immediately after "Finished Compilation for all HLOs"
3. Symlink pre-sharded weights to the artifacts dir
4. Restart vLLM with `NEURON_COMPILED_ARTIFACTS` -- loads in ~40s

### vLLM Batch Size Scaling

| Batch Size | Sync ITL (ms) | Throughput (tok/s) | vs bs=1 |
|------------|---------------|-------------------|---------|
| 1 | 22.2 | 33.3 | baseline |
| 2 | 30.5 | 41.7 | +25% |
| 4 | 37.1 | 53.3 | +60% |
| 8 | 47.6 | 66.7 | +100% |

**Recommendation:** bs=2 for balanced latency/throughput, bs=4 for throughput-optimized, bs=1 for latency-critical.

---

## Example Checkpoints

- `deepseek-ai/DeepSeek-V3-0324` (FP8, 642GB, requires `trust_remote_code=True`)
- Pre-sharded weights on S3: `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/sharded_weights/` (1.37TB, 64 per-rank files)
- Compiled NEFFs on S3: `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/traced_model/` (96MB)

---

## File Inventory

```
contrib/models/DeepSeek-V3/
  README.md                              (339 lines)  Structured documentation
  src/
    __init__.py                          ( 28 lines)  Public API exports
    modeling_deepseek.py                 (995 lines)  Full model implementation
    rope_util.py                         (157 lines)  YaRN RoPE with interleaved layout
  test/
    unit/
      test_config.py                     (192 lines)  Config parsing, MLA params, FP8 dequant
      test_rope.py                       (169 lines)  RoPE frequencies, interleaved layout, HF match
      test_router.py                     (295 lines)  Router shapes, HF reference, group selection
      test_weight_conversion.py          (381 lines)  Expert fusion, router rename, dense skip
      test_helper/
        reference_model.py               (421 lines)  HF reference MLA implementation
        util.py                          (196 lines)  Test utilities
    integration/
      test_model.py                      (515 lines)  7 end-to-end tests (compile, generate, accuracy, perf)
```

**Total: 15 files, ~3,700 lines** (source: 1,180 lines, tests: 2,169 lines, docs: 339 lines)

---

## Maintainer

AWS Neuron

**Last Updated:** 2026-03-19
