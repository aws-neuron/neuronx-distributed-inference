# PR Plan: DeepSeek V3 Contrib Model Submission

## Overview

This plan brings our DeepSeek V3 NXDI implementation into alignment with the
"gold standard" established by the Trinity model PR (#55) for the
`contrib/models/` directory.

**Reference PR:** https://github.com/aws-neuron/neuronx-distributed-inference/pull/55

---

## Gap Analysis: Current State vs Trinity PR Standard

### What We Have (Phases 1-11 Complete)

| Component | Location | Status |
|-----------|----------|--------|
| Model code (992 lines) | `src/.../models/deepseek/modeling_deepseek.py` | Done |
| RoPE utility (154 lines) | `src/.../models/deepseek/rope_util.py` | Done |
| `__init__.py` | `src/.../models/deepseek/__init__.py` | Done |
| Unit tests: RoPE (3) | `test/unit/models/deepseek/test_rope.py` | 3/3 PASS |
| Unit tests: router (9) | `test/unit/models/deepseek/test_router.py` | 9/9 PASS |
| Unit tests: state dict (7) | `test/unit/models/deepseek/test_state_dict_conversion.py` | 7/7 PASS |
| Unit tests: router in state dict (4) | `test/unit/models/deepseek/test_state_dict_conversion.py` | 4/4 PASS |
| Reference MLA impl | `test/unit/models/deepseek/test_helper/reference_model.py` | Done |
| On-device attn tests (3) | `test/unit/models/deepseek/test_modeling_deepseek.py` | Need device |
| Logit matching (mini) | `examples/test_logit_matching.py` | Done (trn2) |
| Multi-step logit matching | `examples/test_logit_matching_full.py` | Done (trn2) |
| Generation script (671B) | `examples/generation_deepseek_v3.py` | Done (trn2.48xl) |
| Benchmark script | `examples/benchmark_deepseek_v3.py` | Done (trn2.48xl) |
| Profiling script | `examples/profile_deepseek_v3.py` | Done (trn2.48xl) |
| Install scripts | `install_deepseek.sh`, `install_deepseek_vllm.sh` | Done |
| PLAN doc | `PLAN_deepseek_v3.md` | Done (1361 lines) |
| S3 artifacts | `s3://deepseek-v3-nxdi-artifacts/` | Traced + sharded |

### What the Trinity PR Has That We Don't

| Gap | Trinity Has | We Have | Priority |
|-----|------------|---------|----------|
| **1. `contrib/` directory structure** | `contrib/models/Trinity/{README,src/,test/}` | Code in `src/` and `test/` at repo root | HIGH |
| **2. PR-ready README.md** | 795-line structured README with all required sections | `PLAN_deepseek_v3.md` (internal notes, not PR-format) | HIGH |
| **3. Integration test (`test_model.py`)** | Structured pytest with env-var config, accuracy + perf tests | Ad-hoc example scripts (not pytest-structured) | HIGH |
| **4. Config unit tests** | `test_config.py` (22 tests: params, layer types, fused eligibility) | Config tested implicitly via other tests | MEDIUM |
| **5. Token match rate analysis** | 8-prompt comparison table (Neuron vs CPU, 64 tokens each) | First-token exact match verified, no multi-prompt table | MEDIUM |
| **6. First-token accuracy table** | Top-1 match + top-20/full-vocab cosine similarity | Logit matching done but not formatted as PR table | MEDIUM |
| **7. Performance benchmarks table** | TTFT/TKG/throughput for every instance x TP x BS combo | Only trn2.48xlarge at tp=64 (bs=1,2,4,8 via GuideLLM) | MEDIUM |
| **8. Compatibility matrix** | Instance x TP x LNC x status table | Scattered in PLAN notes | LOW |
| **9. Usage code examples** | Copy-paste Python snippets per model size | In PLAN but not README format | LOW |
| **10. Caveats section** | Numbered list of gotchas | In PLAN "Issues Encountered" but not formatted | LOW |
| **11. Key porting challenges** | Numbered list of non-trivial solutions | In PLAN but not formatted | LOW |
| **12. License headers** | Apache 2.0 on all files | Partial (modeling file has HF license) | LOW |
| **13. GPU comparison** | g5.12xlarge A10G benchmarks | Not done (not applicable for 671B) | SKIP |

### What We Have That Trinity Doesn't

| Our Advantage | Detail |
|---------------|--------|
| FP8 dequantization | Automatic FP8->BF16 conversion for native FP8 weights |
| vLLM integration | Full vLLM serving tested with GuideLLM sweep |
| Spot recovery scripts | `save_to_s3.sh` / `restore_from_s3.sh` |
| Neuron profiling | System-level Perfetto trace, per-NC straggler analysis |
| Multi-batch-size GuideLLM sweep | bs=1,2,4,8 with detailed cross-comparison |
| Pre-sharded checkpoint workflow | Documented and tested fast-load path |

---

## Implementation Plan

### Phase A: Restructure into `contrib/` Layout (LOCAL, this instance)

**Effort:** ~2 hours | **Requires:** No device access

Create the PR-ready directory structure:

```
contrib/models/DeepSeek-V3/
  README.md                          # NEW: structured PR README
  src/
    __init__.py                      # COPY from src/.../deepseek/__init__.py
    modeling_deepseek.py             # COPY from src/.../deepseek/modeling_deepseek.py
    rope_util.py                     # COPY from src/.../deepseek/rope_util.py
  test/
    __init__.py                      # NEW
    unit/
      __init__.py                    # NEW
      test_config.py                 # NEW: config parsing + validation tests
      test_weight_conversion.py      # ADAPT from test/unit/.../test_state_dict_conversion.py
      test_rope.py                   # COPY from test/unit/.../test_rope.py
      test_router.py                 # COPY from test/unit/.../test_router.py
      test_helper/                   # COPY from test/unit/.../test_helper/
        __init__.py
        reference_model.py
        util.py
    integration/
      __init__.py                    # NEW
      test_model.py                  # NEW: structured integration test
```

**Tasks:**
- [x] A1: Create `contrib/models/DeepSeek-V3/` directory structure
- [x] A2: Copy source files (`src/`) with updated imports (relative)
- [x] A3: Copy and adapt unit tests (update import paths)
- [x] A4: Write `test_config.py` — config parsing, layer types, FP8 detection
- [x] A5: Write `test_model.py` integration test (env-var driven, pytest)
- [x] A6: Run all CPU unit tests in new structure to verify (39 pass, 7 skip)

### Phase B: Write PR-Ready README.md (LOCAL, this instance)

**Effort:** ~3 hours | **Requires:** No device access

Following the Trinity README template section by section:

#### Required Sections (per CONTRIBUTING.md)

- [x] B1: **Model Family** — table with HF ID, params, active params, instance
- [x] B2: **Architecture Details** — feature table (layers, dims, heads, experts, etc.)
- [x] B3: **Unique Architecture Features** — MLA, custom router, dense layers, YaRN RoPE
- [x] B4: **Validation Results** — token match rate, first-token accuracy (from existing data)
- [x] B5: **Performance Benchmarks** — TTFT/TKG/throughput tables (from Phase 8/11 data)
- [x] B6: **Usage** — code examples for mini model (tp=2) and full 671B (tp=64)
- [x] B7: **Caveats** — numbered list (lnc=2, FP8 dequant, MLA cache, etc.) + NCC_IBIR297
- [x] B8: **Compatibility Matrix** — instance x TP x LNC x status table
- [x] B9: **SDK Configuration** — versions table
- [x] B10: **Testing** — pytest commands and prerequisites
- [x] B11: **Key Porting Challenges** — numbered list of non-trivial solutions
- [x] B12: **vLLM Integration** — serving instructions, GuideLLM results
- [x] B13: **Example Checkpoints** — HuggingFace link, S3 artifacts

#### Data We Already Have (from PLAN)

| README Section | Source in PLAN |
|----------------|---------------|
| Benchmarks (trn2.48xl) | Phase 8 Run 2/3: 20.5ms TPOT, 1.67s TTFT, 72.6 tok/s |
| GuideLLM sweep | Phase 11: bs=1(22ms/33tok), bs=2(31ms/42tok), bs=4(37ms/53tok), bs=8(48ms/67tok) |
| Profiling | Phase 10: 18.9ms TPOT, 16% collectives, 20% straggler |
| Compatibility | Phase 8: tp=64/lnc=2 required, tp=32 OOM |
| First-token accuracy | Phase 7: exact match HF vs NXDI |
| Configuration | Phase 8: MoENeuronConfig with all params |

#### Data Gaps (need to generate/format)

| Gap | How to Fill |
|-----|-------------|
| Token match rate table (8 prompts) | Run on device or extract from existing logit matching |
| First-token cosine similarity | Compute from existing logit matching data |
| Maximum sequence length | Test at various seq_lens on device |

### Phase C: Write Integration Test (`test_model.py`) (LOCAL, this instance)

**Effort:** ~2 hours | **Requires:** No device access (test code only)

Modeled after Trinity's `test_model.py` structure:

```python
# Environment variables:
#   DEEPSEEK_MODEL_PATH  — HF weights directory
#   DEEPSEEK_COMPILED_PATH — pre-compiled traced model directory
#   DEEPSEEK_TP_DEGREE — tensor parallelism (default: 64)

class TestDeepSeekV3Model:
    # Smoke tests
    test_model_loads()           # compile + load succeeds
    test_model_generates()       # 20-token generation produces text

    # Accuracy tests
    test_output_coherence()      # 30 tokens, no excessive repetition
    test_top_token_valid()       # top-1 token is a valid token ID
    test_token_match_rate()      # 64-token Neuron vs mini-model comparison

    # Performance tests
    test_performance_ttft()      # TTFT within threshold (configurable)
    test_performance_throughput() # throughput within threshold (configurable)
```

**Tasks:**
- [x] C1: Write `test_model.py` skeleton with env-var config
- [x] C2: Implement smoke tests (load, generate)
- [x] C3: Implement accuracy tests (coherence, top-token, HF match)
- [x] C4: Implement performance tests (TTFT, throughput thresholds)
- [x] C5: Add `__main__` standalone runner

### Phase D: On-Device Validation (trn2.48xlarge, this instance)

**Effort:** ~4 hours | **Requires:** Neuron devices (currently occupied by vLLM)

**Prerequisite:** Stop vLLM server to free all 64 NeuronCores.

- [x] D1: Stop vLLM bs=8 server
- [ ] D2: Run 3 on-device attention unit tests (test_modeling_deepseek.py) — BLOCKED (not in contrib/)
- [x] D3: Mini model integration test — BLOCKED by NCC_IBIR297, tests skip cleanly
- [x] D4: Run integration test with full 671B model (tp=64, pre-sharded weights) — 6 PASS, 1 SKIP
- [x] D5: Generate multi-prompt generation quality table (8 prompts x 64 tokens) — all PASS
- [x] D6: Generate first-token accuracy table (8 prompts) — 4/8 exact match, all semantically valid
- [x] D7: HBM budget analysis for seq_len scaling (512/2048/4096/8192) — documented in PR_README.md
- [ ] D8: Restart vLLM server if needed

### Phase E: Benchmarking Data Collection Plan (MULTI-INSTANCE)

We currently have benchmarks ONLY for trn2.48xlarge (tp=64, lnc=2).
DeepSeek V3 671B requires 1.5TB HBM — only trn2.48xlarge can run it.
Unlike Trinity (3 model sizes, 4 instance types), DeepSeek V3 has a single
configuration: **tp=64, lnc=2, trn2.48xlarge**.

#### What We Already Have (trn2.48xlarge)

| Config | TTFT | TPOT/TKG | Throughput | Source |
|--------|------|----------|-----------|--------|
| tp=64, bs=1, seq=512 | 1,667ms | 20.5ms | 48.7 tok/s | NXDI benchmark |
| tp=64, bs=1, seq=512 (vLLM) | 1,710ms | 22.2ms ITL | 33 tok/s | GuideLLM |
| tp=64, bs=2, seq=512 (vLLM) | 1,670ms | 30.5ms ITL | 42 tok/s | GuideLLM |
| tp=64, bs=4, seq=512 (vLLM) | 1,690ms | 37.1ms ITL | 53 tok/s | GuideLLM |
| tp=64, bs=8, seq=512 (vLLM) | 1,699ms | 47.6ms ITL | 67 tok/s | GuideLLM |

#### What's Needed for the PR

The Trinity PR includes benchmarks across 4 instance types (inf2.xlarge,
inf2.8xlarge, trn2.3xlarge, trn2.48xlarge) because Trinity has 3 model sizes.
DeepSeek V3 has ONE model size (671B, 37B active) that only fits on
trn2.48xlarge. **No additional instance types are needed.**

However, the PR should include structured benchmark data in Trinity's format:

| Data Point | Status | Action |
|------------|--------|--------|
| NXDI native benchmark (bs=1) | HAVE | Format into table |
| NXDI native benchmark (bs=2,4,8) | MISSING | Run `benchmark_deepseek_v3.py` with bs=2,4,8 |
| GuideLLM sweep (bs=1,2,4,8) | HAVE | Format into table |
| vLLM serving latency | HAVE | Format into table |
| Compilation time | HAVE (11.8min) | Include in table |
| Weight loading time | HAVE (7.8min sharded) | Include in table |
| Max sequence length | PARTIAL | Test seq=1024, 2048, 4096 on device |

#### NXDI Native Benchmarks Still Needed (trn2.48xlarge)

Run `benchmark_deepseek_v3.py` with different batch sizes using the compiled
NEFFs from Phase 11:

```bash
# These require recompilation since the existing NEFFs are vLLM-format.
# Use the Phase 8 traced model (bs=1, already compiled).
# bs=2,4,8 would need recompilation (~12 min each).
python examples/benchmark_deepseek_v3.py \
    --traced-model-path /scratch/deepseek_v3_traced \
    --tp-degree 64 --batch-size 1 --num-runs 20
```

**Estimated time:** ~1 hour (load model + 20 runs)

#### Benchmarking Plan for Other Instance Types

**Not applicable.** DeepSeek V3 671B requires:
- 1.5TB HBM (256 experts x weight shards)
- tp=64 (all 64 logical NeuronCores on trn2.48xlarge)
- lnc=2 (trn2 platform default)

No smaller instance can run it:
- trn2.3xlarge: 4 NeuronCores, 96GB HBM — insufficient
- inf2.48xlarge: 24 NeuronCores, 384GB HBM — insufficient
- trn2.48xlarge is the ONLY option

A mini-model variant (small dims, random weights) runs on trn2.3xlarge (tp=2)
but that is only useful for development validation, not production benchmarking.

### Phase F: Finalize and Create PR

**Effort:** ~2 hours | **Requires:** No device access

- [x] F1: Verify all files have Apache 2.0 license headers (14/14 .py files)
- [x] F2: Run final CPU unit test pass in contrib structure (39 pass, 7 skip)
- [ ] F3: Create git branch from upstream main
- [ ] F4: Add all contrib files
- [ ] F5: Create PR with structured description

---

## Detailed README Outline

Following the Trinity template exactly:

```markdown
# Contrib Model: DeepSeek-V3

NeuronX Distributed Inference implementation of DeepSeek V3 (671B MoE, 37B active).

## Model Family
| Model | HuggingFace ID | Total Params | Active Params | Instance |
|-------|----------------|-------------|---------------|----------|
| DeepSeek-V3-0324 | deepseek-ai/DeepSeek-V3-0324 | 671B | 37B | trn2.48xlarge (TP=64) |

License: [DeepSeek License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL)

## Architecture Details
[Feature table: layers, hidden, MLA dims, experts, routing, RoPE, vocab]

### Unique Architecture Features
- Multi-head Latent Attention (MLA)
- Custom MoE Router (sigmoid + group selection + e_score_correction_bias)
- Dense layers 0-2 (no MoE)
- YaRN RoPE with interleaved layout
- Native FP8 weights (auto-dequantized to BF16)

## Validation Results
### Token Match Rate: Neuron vs CPU
### First-Token Accuracy

## Performance Benchmarks
### DeepSeek-V3 on trn2.48xlarge (TP=64, LNC=2)
[Table: bs x TTFT x TKG x throughput]
### vLLM Serving (GuideLLM)
[Table: bs x ITL x throughput x req/s]
### Key Observations

## Usage
[Code example for full 671B model]
[Code example for mini model (development)]

## Pre-Sharded Deployment
[Workflow: compile -> shard -> fast load]

## Caveats
1. lnc=2 required on trn2
2. FP8 dequantization adds ~3.5h to first-time weight sharding
3. MLA attention incompatible with NeuronAttentionBase GQA
4. 2TB RAM + NVMe swap needed for weight sharding
5. save_sharded_checkpoint=True strongly recommended
6. num_key_value_heads=1 in config (MLA, not GQA)
7. disable_numeric_cc_token=True required
8. enable_bucketing=False (not tested with MLA)

## Maximum Sequence Length
[Table: seq_len x compile x load x status]

## Compatibility Matrix
[Table: instance x TP x LNC x status]

## SDK Configuration
[Versions table]

## Testing
[pytest commands + prerequisites]

## Key Porting Challenges
1. MLA incompatible with NeuronAttentionBase (custom attention class)
2. Custom MoE router (group-based selection, bias, scaling)
3. YaRN RoPE interleaved layout (rotate_fn not rotate_half)
4. Dense layers 0-2 require separate MLP class
5. FP8 dequantization during state dict loading
6. Expert fusion (gate+up -> gate_up_proj)
7. KV cache stores [k_pe | compressed_kv] (576-dim)
8. Weight sharding peak RSS ~2TB
9. tp=32 HBM OOM (256 experts, all on every rank)

## vLLM Integration
[Serving commands, NEURON_COMPILED_ARTIFACTS, GuideLLM results]

## Example Checkpoints
- deepseek-ai/DeepSeek-V3-0324 (FP8, 642GB)
- S3 pre-sharded: s3://deepseek-v3-nxdi-artifacts/

## Maintainer
[Name + date]
```

---

## Test Plan Summary

### CPU Tests (run anywhere, no device needed)

| Test File | Tests | What It Validates |
|-----------|-------|-------------------|
| `test_rope.py` | 3 | YaRN RoPE frequencies, interleaved layout |
| `test_router.py` | 9 | Group selection, bias, scaling, output shapes |
| `test_state_dict_conversion.py` | 7+4 | Expert fusion, router rename, bias rename, dense passthrough |
| `test_config.py` (NEW) | ~8 | Config parsing, MoE params, dense layer detection, FP8 flag |
| **Total** | **~31** | |

### Integration Tests (need trn2.48xlarge or trn2.3xlarge)

| Test | Instance | TP | What It Validates |
|------|----------|-----|-------------------|
| `test_model_loads` | trn2.3xlarge | 2 | Mini model compile + load |
| `test_model_generates` | trn2.3xlarge | 2 | 20-token generation |
| `test_output_coherence` | trn2.3xlarge | 2 | No excessive repetition |
| `test_top_token_valid` | trn2.3xlarge | 2 | Valid token IDs |
| `test_token_match_rate` | trn2.3xlarge | 2 | Neuron vs CPU logit comparison |
| `test_performance_ttft` | trn2.48xlarge | 64 | TTFT < threshold |
| `test_performance_throughput` | trn2.48xlarge | 64 | Throughput > threshold |

### On-Device Tests (need 2+ NeuronCores)

| Test | Instance | What It Validates |
|------|----------|-------------------|
| `test_attn_neuron_e2e` (3 variants) | Any Neuron | MLA attention prefill+decode on device |

---

## Execution Order

```
Phase A (restructure)     ←── Can start NOW (no device needed)
  ↓
Phase B (README)          ←── Can start NOW (parallel with A)
  ↓
Phase C (integration test) ←── Can start NOW (parallel with A+B)
  ↓
Phase D (on-device)       ←── Needs device (stop vLLM first)
  ↓
Phase E (benchmarks)      ←── Needs device (runs on this trn2.48xlarge)
  ↓
Phase F (finalize PR)     ←── No device needed
```

**Phases A, B, C can run in parallel** — all are local, CPU-only work.
**Phase D requires stopping vLLM** to free NeuronCores.
**Phase E runs on this same trn2.48xlarge** — no other instances needed.

---

## Instance Requirements Summary

| Task | Instance | Why |
|------|----------|-----|
| CPU unit tests | Any (even laptop) | Pure CPU, no Neuron |
| On-device attention tests | trn2.3xlarge+ (2 NCs) | Needs NeuronCores for compile+run |
| Mini model integration | trn2.3xlarge (4 NCs) | tp=2 mini model |
| Full 671B integration | trn2.48xlarge (64 NCs) | tp=64, 1.5TB HBM |
| Full 671B benchmarks | trn2.48xlarge (64 NCs) | Same as above |
| vLLM serving tests | trn2.48xlarge (64 NCs) | Same as above |

**All testing can be done on this single trn2.48xlarge instance.**
No other instance types are needed because DeepSeek V3 671B only runs on
trn2.48xlarge.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NeuronCores occupied by vLLM | Can't run on-device tests | Stop vLLM server first |
| Weight sharding takes 3.5h | Delays first-time setup | Use pre-sharded from S3 |
| Spot reclamation mid-test | Lose NVMe artifacts | S3 backup already exists |
| Token match rate low (BF16 MoE) | PR review concern | Document as expected (same as Trinity) |
| No GPU comparison | PR may lack context | Not applicable for 671B model |

---

## Current Instance State (2026-03-19)

- **Instance:** trn2.48xlarge (i-011452ed53926c6ae)
- **NeuronCores:** All 64 occupied by vLLM bs=8 server
- **SDK:** NXDI 0.8.0, neuronx-cc 2.23.6484, PyTorch 2.9.0, transformers 4.57.6
- **vLLM:** 0.13.0 (serving on port 8000)
- **CPU tests:** 24/24 PASS (3 on-device tests SKIP — need free NeuronCores)
- **Pre-sharded weights:** Available at `/scratch/deepseek_v3_traced/weights/`
- **Compiled NEFFs:** Available for bs=1,2,4,8 at `/scratch/vllm_bs{1,2,4,8}/`
- **S3 backup:** All artifacts in `s3://deepseek-v3-nxdi-artifacts/`
