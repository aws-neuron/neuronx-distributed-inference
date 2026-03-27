## Description

NxDI (NeuronX Distributed Inference) implementation for [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), a 671B parameter Mixture-of-Experts model (37B active per token) from DeepSeek AI. Uses Multi-head Latent Attention (MLA) and a custom group-based MoE router with 256 routed experts.

Key architecture features ported:
- **Multi-head Latent Attention (MLA):** KV compressed through low-rank bottleneck (kv_lora_rank=512), KV cache stores `[k_pe | compressed_kv]` with combined dim 576
- **Custom MoE Router:** Group-based expert selection (8 groups, top-4 groups, 8 experts per token) with `e_score_correction_bias`, sigmoid activation, normalization, and `routed_scaling_factor=2.5`
- **Dense layers 0-2:** First 3 layers use dense MLP (intermediate_size=18432) instead of MoE
- **YaRN RoPE:** Interleaved layout using `rotate_fn` (not `rotate_half`)
- **Native FP8 weights:** float8_e4m3fn with block-wise scale factors, automatically dequantized to BF16 during loading

## Model Information

- **Model Name:** DeepSeek-V3 (DeepSeek-V3-0324)
- **Model Architecture:** Mixture-of-Experts decoder-only transformer (671B total, 37B active per token)
- **Purpose:** Text generation (causal language modeling)
- **HuggingFace:** https://huggingface.co/deepseek-ai/DeepSeek-V3
- **License:** [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL)

## Checklist

### Required Components

- [x] **Accuracy Test** (`test/integration/test_model.py`)
  - Integration test validates model accuracy via logit comparison and top-k token verification
  - Test can compile and run the model on Neuron (validated on trn2.48xlarge, TP=64)
  - Logit matching via `inference_demo` with pre-generated golden logits: **PASS** (32 tokens, 0 top-1 divergences, mean abs error 0.007)
- [x] **README.md** with the following sections:
  - Usage Example: Python API for 671B model (TP=64, trn2.48xlarge)
  - Compatibility Matrix: trn2.48xlarge with SDK 2.28
  - Example Checkpoints: `deepseek-ai/DeepSeek-V3-0324`
  - Testing Instructions: Commands to run unit and integration test suites
  - vLLM Integration: Serving via vLLM with Neuron backend
  - Performance Benchmarks: TPOT, TTFT, throughput measurements
- [x] **Source Code** (`src/`)
  - `modeling_deepseek.py` (~845 lines): Model implementation following NxD Inference patterns
  - `rope_util.py` (~157 lines): YaRN RoPE with interleaved layout
  - Properly structured in the contrib folder hierarchy

### Optional Components

- [x] **Unit Tests** (CPU-based, no Neuron device required)
  - `test_config.py`: 15/15 PASS (config parsing, MLA params, MoE setup, FP8 dequant)
  - `test_rope.py`: 3/3 PASS (frequency table, RoPE application, HF interleave match)
  - `test_router.py`: 9/9 PASS (group-based routing, expert selection, weight normalization)
  - `test_weight_conversion.py`: 10/10 PASS (state dict conversion, expert fusion, FP8 dequant)

## Folder Structure

```
contrib/models/DeepSeek-V3/
├── README.md
├── PR_README.md
├── src/
│   ├── __init__.py
│   ├── modeling_deepseek.py      # Core model classes (MLA, MoE, decoder)
│   └── rope_util.py              # YaRN RoPE with interleaved layout
└── test/
    ├── __init__.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_config.py            # Config parsing, MLA, MoE, FP8
    │   ├── test_rope.py              # YaRN RoPE validation
    │   ├── test_router.py            # Group-based MoE router
    │   ├── test_weight_conversion.py # State dict conversion, expert fusion
    │   └── test_helper/
    │       ├── __init__.py
    │       ├── util.py               # Test utilities & fixtures
    │       └── reference_model.py    # Reference MLA from HF
    └── integration/
        ├── __init__.py
        └── test_model.py            # End-to-end model testing
```

## Testing

### How to run the test suite

**Unit tests (CPU only, no Neuron device needed):**

```bash
cd contrib/models/DeepSeek-V3/
pytest test/unit/ -v
```

Expected: **37/37 PASS** (config: 15, rope: 3, router: 9, weight_conversion: 10)

**Integration tests (mini model, 2+ NeuronCores):**

```bash
cd contrib/models/DeepSeek-V3/
pytest test/integration/test_model.py --capture=tee-sys
```

**Integration tests (full 671B model, trn2.48xlarge, TP=64):**

```bash
DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-V3-0324-FP8 \
DEEPSEEK_COMPILED_PATH=/scratch/deepseek_v3_traced \
DEEPSEEK_TP_DEGREE=64 \
DEEPSEEK_SEQ_LEN=512 \
pytest test/integration/test_model.py --capture=tee-sys
```

### Accuracy validation via inference_demo (logit matching)

This is the key accuracy test. It compares Neuron model logits against pre-generated golden logits from the HuggingFace reference model (FP32 CPU).

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

inference_demo --model-type deepseek_v3 --task-type causal-lm run \
    --model-path /path/to/DeepSeek-V3-0324-FP8 \
    --compiled-model-path /scratch/nxd_model \
    --tp-degree 64 --batch-size 1 --seq-len 512 \
    --logical-nc-config 2 \
    --save-sharded-checkpoint \
    --output-logits \
    --on-device-sampling --global-topk 256 --top-k 1 \
    --skip-compile \
    --prompt "The capital of France is" \
    --check-accuracy-mode logit-matching \
    --expected-outputs-path /path/to/golden_logits.pt \
    --num-tokens-to-check 32 \
    --divergence-difference-tol 0.20 \
    --tol-map "{None: (1e-5, 0.20), 1000: (1e-5, 0.15), 50: (1e-5, 0.10), 5: (1e-5, 0.05)}"
```

**Note:** Logit matching requires the model to be compiled with `--output-logits` enabled. The `--expected-outputs-path` uses pre-generated golden logits (in `GenerateDecoderOnlyOutput` format with `.scores` attribute) to skip the expensive FP32 CPU reference generation (which requires 642GB HF weights + ~2.7TB peak RAM). Without this flag, `inference_demo` generates golden logits on-the-fly from the full HF model.

**Note:** Relaxed `--tol-map` thresholds are required for this 671B MoE model because: (1) BF16 computation vs FP32 golden reference, and (2) MoE v2 NKI kernel accumulates in BF16, introducing expected numerical divergence. Despite relaxed thresholds, top-1 tokens are preserved across all 32 validated positions.

### Accuracy validation via inference_demo (skip accuracy, benchmark only)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

inference_demo --model-type deepseek_v3 --task-type causal-lm run \
    --model-path /path/to/DeepSeek-V3-0324-FP8 \
    --compiled-model-path /scratch/nxd_model \
    --tp-degree 64 --batch-size 1 --seq-len 512 \
    --logical-nc-config 2 \
    --save-sharded-checkpoint \
    --on-device-sampling --global-topk 256 --top-k 1 \
    --skip-compile \
    --prompt "The capital of France is" \
    --check-accuracy-mode skip-accuracy-check \
    --benchmark
```

### vLLM serving

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

export NEURON_COMPILED_ARTIFACTS=/scratch/vllm_bs1

VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3-0324-FP8 \
    --tensor-parallel-size 64 --max-model-len 512 --max-num-seqs 1 \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --port 8000 \
    --additional-config '{"override_neuron_config": {"logical_nc_config": 2, "enable_bucketing": false, "save_sharded_checkpoint": true}}'
```

**Note:** `NEURON_COMPILED_ARTIFACTS` is required to reuse pre-compiled NEFFs. `logical_nc_config: 2` is mandatory on trn2.

## Test Results

### Unit Tests (CPU)

| Test Module | Tests | Status |
|-------------|-------|--------|
| test_config.py | 15 | 15/15 PASS |
| test_rope.py | 3 | 3/3 PASS |
| test_router.py | 9 | 9/9 PASS |
| test_weight_conversion.py | 10 | 10/10 PASS |
| **Total** | **37** | **37/37 PASS** |

### Integration Test (671B, trn2.48xlarge, TP=64)

| Test | Status | Notes |
|------|--------|-------|
| Model loads | PASS | Pre-sharded checkpoint load ~8 min |
| Model generates | PASS | Generates coherent multi-sentence text |
| Output coherence | PASS | 3+ words, no excessive repetition |
| Top token valid | PASS | First token decodable and semantically valid |
| First-token HF match | PASS | Matches HuggingFace FP32 reference |
| TTFT performance | PASS | ~1,668 ms (256 input tokens) |
| Throughput | PASS | ~48.7 tok/s (bs=1) |

### Logit Matching (671B, trn2.48xlarge, TP=64)

Validated on 2026-03-27 via `inference_demo --check-accuracy-mode logit-matching` with pre-generated FP32 golden logits.

| Metric | Value | Notes |
|--------|-------|-------|
| Tokens validated | 32 | All passed |
| Top-1 divergences | 0 | No top-token mismatches |
| Mean abs logit error | 0.007 | Across all tokens |
| Max abs logit error | 0.048 | At later tokens (error propagation) |
| Top-5 max relative error | 0.038 | Within BF16 MoE precision |
| Top-50 max relative error | 0.064 | Expected for 256-expert BF16 accumulation |
| Max divergence difference | 0.134 | Between consecutive tokens |
| First token | "Paris" | Exact match with FP32 reference |
| **Result** | **PASS** | With BF16 MoE tolerances |

**Tolerance note:** Default thresholds (0.05/0.03/0.02/0.01 for None/1000/50/5 top-k) are too tight for a 671B BF16 MoE model. Relaxed to 0.20/0.15/0.10/0.05 to account for: (1) BF16 vs FP32 golden reference, (2) MoE v2 NKI kernel BF16 accumulation. All top-1 tokens are preserved despite relaxed thresholds.

### Multi-Prompt Generation Quality (671B, TP=64)

Single-request greedy generation (top_k=1), 64 output tokens per prompt:

| Prompt | First Token | Status |
|--------|-------------|--------|
| "The capital of France is" | Paris | PASS |
| "def fibonacci(n):" | if | PASS |
| "The theory of relativity states that" | nothing | PASS |
| "In a shocking finding, scientists discovered" | that | PASS |
| "To make a chocolate cake, you need" | the | PASS |
| "The largest ocean on Earth is" | the | PASS |
| "Machine learning is a subset of" | artificial | PASS |
| "The year 2025 will be remembered for" | the | PASS |

All 8 prompts produce coherent, factually correct, multi-sentence responses. Code generation (fibonacci) produces syntactically valid Python.

### Generation Output (671B, TP=64, seq_len=512, greedy top_k=1)

**Prompt:** "The capital of France is"

**Output:** Paris, which is one of the most important and influential cities in the world. Paris is located in the northern part of France, on the banks of the Seine River. It is known for its rich history, culture, art, fashion, and cuisine. Some of the most famous landmarks in Paris include the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, the Arc de Triomphe, and the Champs-Elysees. Paris is also a major center for business, education, and politics, hosting numerous international organizations and events.

**Status:** PASS -- coherent, factually correct, multi-sentence response.

## Performance Benchmarks

**BF16, trn2.48xlarge (64 NeuronCores), lnc=2.** All measurements from compiled and loaded model with pre-sharded checkpoints.

### NXDI Native Benchmark (bs=1, seq_len=512, 256 input / 256 output tokens)

Confirmed on 2026-03-27 via `inference_demo --benchmark` with 20 timed iterations:

| Component | p50 (ms) | p90 (ms) | p99 (ms) | Throughput |
|-----------|----------|----------|----------|------------|
| **Token Generation (TPOT)** | **20.6** | 20.9 | 21.4 | 48.6 tok/s |
| **Context Encoding (TTFT)** | **1,666** | 1,667 | 1,668 | 307 tok/s |
| **End-to-End** | **7,088** | 7,112 | 7,123 | 72.2 tok/s |

p50-p99 spread < 1ms for token generation (very stable).

### vLLM Serving + GuideLLM Sweep (seq_len=512, ~200 input / ~200 output tokens)

| BS | Sync ITL (ms) | Max Throughput (tok/s) | Best Constant-Rate (tok/s) | Best ITL (ms) |
|----|---------------|----------------------|---------------------------|---------------|
| 1 | 22.2 | 33.3 | 33.3 | 22.1 |
| 2 | 30.5 | 41.7 | 39.6 | 37.1 |
| 4 | 37.1 | 53.3 | 48.3 | 62.2 |
| 8 | 47.6 | 66.7 | 55.0 | 106.3 |

TTFT consistent at ~1,700ms across all batch sizes.

### Timing Summary

| Operation | Time |
|-----------|------|
| NEFF compilation (first time) | 11.8 min |
| NEFF compilation (from cache) | ~1s |
| Weight sharding (FP8 -> 64 per-rank files) | 3.5 hours |
| Load from pre-sharded checkpoints | 7.8 min |
| TPOT (token generation, p50) | 20.5 ms |
| TTFT (context encoding, 256 tokens) | 1,667 ms |

### Sequence Length Validation

| seq_len | Compile | Load | Status | Notes |
|---------|---------|------|--------|-------|
| 512 | PASS (11.8 min, ~1s cached) | PASS (35.7s) | **PASS** | Default, all benchmarks |
| 1024 | PASS (~10 min, CTE cached) | **FAIL** (HBM OOM) | **FAIL** | CTE scratchpad (512MB) + TKG model (23.1GB) > 24GB HBM per NC pair |

seq_len=1024 compiles successfully but cannot be loaded: the context encoding model's scratchpad allocation exceeds available HBM when colocated with the token generation model. At TP=64 with lnc=2, the TKG model consumes ~23.1GB of the 24GB available per NeuronCore pair, leaving insufficient space for the CTE model's scratchpad.

## Compatibility

Tested with:
- **Neuron SDK Version(s):** 2.28
- **Instance Type(s):** trn2.48xlarge
- **TP Degree:** 64
- **LNC:** 2
- **PyTorch Version:** 2.9.0
- **Python Version:** 3.12
- **Transformers Version:** 4.57.6
- **neuronx-cc Version:** 2.23.6484.0
- **NxDI Version:** 0.8.16251
- **neuronx-distributed Version:** 0.17.26814
- **torch-neuronx Version:** 2.9.0.2.12.22436

| Instance | TP | LNC | Status | Notes |
|----------|-----|-----|--------|-------|
| trn2.48xlarge | 64 | 2 | **PASS** | Only viable configuration for 671B |

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

## Additional Information

### Key Porting Challenges

1. **MLA incompatible with NeuronAttentionBase:** GQA projections don't apply to MLA's weight absorption. Built a custom `DeepseekV3Attention` class with its own TP sharding, KV cache, and softmax logic.
2. **YaRN RoPE interleaved layout:** Uses `rotate_fn` (interleaved) not `rotate_half` (split). No transpose needed.
3. **Dense layers 0-2:** Separate `DeepseekV3DenseMLP` class with `dense_intermediate_size=18432`.
4. **FP8 dequantization:** Block-wise float8_e4m3fn with per-block scale factors. Vectorized conversion during state dict loading.
5. **Expert fusion:** Per-expert `gate_proj` + `up_proj` fused into `gate_up_proj` tensor `[num_experts, hidden, 2*intermediate]` for ExpertMLPsV2 compatibility.
6. **KV cache compressed format:** `[k_pe | compressed_kv]` with dim 576 (rope_dim 64 + kv_lora_rank 512).
7. **TP=32 HBM OOM:** 256 experts on every rank. Each rank carries ~40GB at TP=32 vs 24GB HBM limit. Fixed by using TP=64 and LNC=2.

### Known Limitations

- `logical_nc_config=2` is mandatory on trn2 (lnc=1 causes HBM OOM)
- TP=64 is the only viable configuration for the 671B model
- FP8 dequantization requires ~2TB peak RAM + NVMe swap for first-time sharding
- MLA attention is incompatible with NeuronAttentionBase (custom implementation required)
- `enable_bucketing=False` required (bucketing not tested with MLA)
- MoE v2 NKI kernel accumulates in bf16 (expected numerical divergence, top-1 tokens preserved)
- Fused TKG path (`init_tkg_module=True`) not supported due to shared expert dim mismatch at TP=64
- seq_len=1024 fails with HBM OOM (TKG model 23.1GB + CTE scratchpad exceeds 24GB per NC pair)
- Maximum validated seq_len is 512

---

By submitting this PR, I confirm that:
- I have read and followed the contributing guidelines
- This is a community contribution and may have limited testing compared to officially-supported models
- The code follows best practices and is well-documented
- All required components listed above are included
