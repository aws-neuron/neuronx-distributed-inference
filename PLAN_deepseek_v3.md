# DeepSeek V3 on NXDI — Implementation Guide

## Quick Start

This document is a self-contained guide for working with the DeepSeek V3 NXDI implementation.
All model code lives in `~/environment/deepseek-nxdi/`.

### Setup on a Fresh Instance

```bash
cd ~/environment/deepseek-nxdi

# Install into NXDI venv (auto-detects /opt/aws_neuronx_venv_pytorch_*_nxd_inference/)
bash install_deepseek.sh

# Activate and run unit tests
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate  # or whichever version exists
python -m pytest test/unit/models/deepseek/ -v

# (Optional) Install into vLLM venv too
bash install_deepseek_vllm.sh
```

### Running on Device

```bash
# Mini model logit matching (needs 2+ NeuronCores, e.g. trn2.3xlarge):
python examples/test_logit_matching.py --tp-degree 2

# Full model (needs trn2.48xlarge + DeepSeek-V3 weights):
python examples/generation_deepseek_v3.py \
    --model-path /path/to/DeepSeek-V3 \
    --tp-degree 32 --seq-len 4096

# vLLM serving:
VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3 \
    --tensor-parallel-size 32 --max-model-len 4096 --max-num-seqs 1 \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill
```

---

## Repository Layout

```
deepseek-nxdi/
  install_deepseek.sh          # Install model code into NXDI venv
  install_deepseek_vllm.sh     # Install model code into vLLM venv
  PLAN_deepseek_v3.md          # This file
  scripts/
    save_to_s3.sh              # Save artifacts to S3 for spot recovery
    restore_from_s3.sh         # Restore from S3 on a new instance
  src/neuronx_distributed_inference/
    models/deepseek/
      __init__.py
      modeling_deepseek.py      # ~913 lines — full model implementation
      rope_util.py              # ~155 lines — YaRN RoPE
    utils/constants.py          # MODEL_TYPES registry (has "deepseek_v3" entry)
  test/unit/models/deepseek/
    test_modeling_deepseek.py   # MLA attention tests (prefill/decode)
    test_rope.py                # RoPE tests
    test_router.py              # Router correctness tests
    test_state_dict_conversion.py  # State dict conversion + router tests
    test_helper/
      reference_model.py        # Self-contained MLA reference impl
      util.py                   # Test utilities
  examples/
    generation_deepseek_v3.py   # Full model inference script (with memory monitor)
    test_logit_matching.py      # First-token accuracy test (needs device)
    test_logit_matching_full.py # Multi-step accuracy test (needs device)
```

---

## Architecture Overview

DeepSeek V3 is a 671B parameter MoE model (37B active per token).

| Feature | Value |
|---------|-------|
| Layers | 61 (3 dense + 58 MoE) |
| Hidden dim | 7168 |
| Attention | MLA (Multi-head Latent Attention) with LoRA-compressed Q |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128, qk_rope_head_dim: 64, v_head_dim: 128 |
| Heads | 128 |
| Routed experts | 256 (8 groups of 32, top-4 groups, 8 experts per token) |
| Shared experts | 1 |
| Routing | sigmoid + e_score_correction_bias + group selection + normalize + scale by 2.5 |
| RoPE | YaRN, interleaved layout |
| Dense intermediate | 18432 |
| MoE intermediate | 2048 |
| Vocab | 129280 |

### Key Design Decisions in the NXDI Implementation

1. **MLA Attention** (`DeepseekV3Attention`): Does NOT extend `NeuronAttentionBase` because GQA projections are incompatible with MLA's weight absorption. KV cache stores `(combined, combined)` where `combined = [k_pe | compressed_kv]` with shape `(bsz, 1, seq_len, 576)` (rope_dim 64 + kv_lora_rank 512).

2. **Custom Router** (`DeepseekV3Router`): Subclasses `RouterTopK` to implement group-based expert selection with `e_score_correction_bias`. Uses `_build_deepseek_moe()` instead of `initialize_moe_module`.

3. **Dense MLP** (`DeepseekV3DenseMLP`): Layers 0-2 use dense MLP with `intermediate_size=18432` (not MoE).

4. **RoPE**: Uses `rotate_fn` (interleaved layout) — NO transpose needed. This is different from optimum-neuron which uses `rotate_half` (split layout) and DOES need a transpose.

5. **State Dict Conversion**: `convert_deepseek_v3_hf_to_neuron_state_dict()` handles router rename, expert fusion (gate+up→gate_up_proj), e_score_correction_bias rename, and dense layer passthrough.

6. **Compiler Config**: `get_compiler_args()` returns `None` (uses framework defaults). `disable_numeric_cc_token = True` required for all-gather/reduce-scatter.

---

## Implementation Status

### Completed (Phases 1-9)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bug fixes + config completion | DONE |
| 2 | Dense MLP support (layers 0-2) | DONE |
| 3 | Custom MoE router (group selection, bias, scaling) | DONE |
| 4 | State dict conversion (HF → Neuron) | DONE |
| 5 | RoPE interleave verification | DONE |
| 6 | On-device compilation (mini model, tp=2) | DONE |
| 7 | Logit matching (HF vs NXDI, first token exact match) | DONE |
| 8 | Benchmarking (671B, tp=64, lnc=2) | DONE — 36.1ms TPOT, 27.7 tok/s |
| 9 | vLLM integration (mini model, serving works) | DONE |

All 17 unit tests pass. Logit matching: first token EXACT MATCH between HF (FP32) and NXDI (BF16).
Full 671B model: compilation, weight sharding, device loading, and generation all working.
NXDI native benchmark: 20.5ms TPOT (p50), 1.67s TTFT, 72.6 tok/s E2E (256-token generation).
GuideLLM sweep complete: bs=1 (22ms ITL, 33 tok/s) → bs=8 (67 tok/s, +100% throughput).

### Remaining

| Phase | Description | Status |
|-------|-------------|--------|
| 10 | Neuron profiling | DONE — system-level profile, 18.9ms TPOT, 16% collectives, 20% straggler |
| 11 | GuideLLM throughput optimization | DONE — bs=1,2,4,8 sweep complete |

---

## Phase 8: Benchmarking (COMPLETE)

### Goal
Measure compilation time, TTFT, TPOT, and throughput for DeepSeek V3 671B on trn2.48xlarge.

### Prerequisites
- trn2.48xlarge instance (64 NeuronCores, 1.5TB HBM, 2TB RAM)
- DeepSeek V3 weights downloaded
- NXDI installed via `install_deepseek.sh`

### Model Weights
Using the official `deepseek-ai/DeepSeek-V3-0324` from HuggingFace (native FP8 format):
- 163 safetensors files, 642GB on disk
- Native block-wise FP8 (float8_e4m3fn) with `weight_scale_inv` tensors
- ~46K `weight_scale_inv` scale keys for block-wise dequantization (block_size=128)
- Downloaded to `~/environment/models/DeepSeek-V3-0324-FP8/` (on EBS root volume, 1.5TB)
- Download completed 2026-03-17, ~41 minutes

```bash
huggingface-cli login --token <YOUR_TOKEN>
huggingface-cli download deepseek-ai/DeepSeek-V3-0324 \
    --local-dir ~/environment/models/DeepSeek-V3-0324-FP8
```

### Storage Layout
- **EBS root** (`/dev/root`, 1.5TB): OS + HF weights (642GB), ~300GB free
- **NVMe `/scratch`** (`/dev/nvme1n1`, 1.7TB): compiled model + sharded weights (1.4TB used)
  - Ephemeral — data lost on instance stop. Must restore from S3.
- **NVMe `/scratch2`** (`/dev/nvme2n1`, 1.7TB): 400GB swap file
  - 2 more NVMe drives available (nvme3n1, nvme4n1) if needed
- **Swap**: 100GB EBS (`/swapfile`) + 400GB NVMe (`/scratch2/swapfile`)
- Traced model at `/scratch/deepseek_v3_traced` (model.pt + weights/)

### FP8 Dequantization
Added `_dequantize_fp8_state_dict()` in `modeling_deepseek.py` to convert FP8 weights to BF16
during state dict loading. This is called automatically by `convert_deepseek_v3_hf_to_neuron_state_dict()`.
The dequantization uses vectorized block-wise multiplication for efficiency.

### Steps

1. **Compile and run** (see "Running the Benchmark" section below for full commands):
   ```bash
   python examples/generation_deepseek_v3.py \
       --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \
       --traced-model-path /scratch/deepseek_v3_traced \
       --tp-degree 64 --seq-len 512 --batch-size 1 --max-new-tokens 128
   ```

2. **Metrics** (built into the generation script):
   - Compilation time, Load time, TPOT, Throughput, Memory (RSS/Available/Swap logged every 30s)

### Memory Analysis (2026-03-17)

**Initial estimate vs actual (observed during first benchmark run):**

| Stage | Estimated | Actual RSS | Notes |
|-------|-----------|-----------|-------|
| Load FP8 safetensors | ~641GB | ~641GB | All 163 shards merged into one dict |
| Dequantize FP8→BF16 | grows to ~1,280GB | ~1,280GB | FP8 replaced by BF16 in-place |
| Expert fusion (per-layer) | +22GB temp | ~22GB | 58 MoE layers, sequential with gc.collect() |
| **After conversion** | **~1,300GB** | **~1,520GB** | Python allocator doesn't return freed FP8 pages to OS |
| Shard per TP rank (64x) | +21GB temp | **~1,980GB** | Shallow copies + contiguous slices + phantom RSS |
| **Actual peak RSS** | — | **~1,983GB** | 2TB RAM + 89GB swap used (nearly OOM) |

The ~680GB gap between estimated (1300GB) and actual (1980GB) is caused by:
1. Python's pymalloc arena allocator not returning freed FP8 memory to OS (~640GB phantom)
2. Per-rank sharding keeping `.contiguous()` copies alongside originals
3. Framework overhead (model parameters, buffers, graph structures)

**CRITICAL: 100GB EBS swap was nearly exhausted. Saved by adding 300GB NVMe swap mid-run.**
Swap thrashing on EBS (~250MB/s) made the loading phase extremely slow (~2 hours).
NVMe swap (~5GB/s) would be 20x faster.

**Framework loading flow** (from `application_base.py` and `model_builder.py`):
1. `checkpoint_loader_fn()` → `get_state_dict()` → `load_state_dict()` (loads ALL shards)
2. `convert_hf_to_neuron_state_dict()` (dequantize + expert fusion — in-place)
3. `shard_weights_with_cache()` per rank (shallow copy + TP slicing)
4. `nxd_model.initialize(weights)` — transfers sharded weights to HBM
5. The framework does NOT support streaming/shard-by-shard loading.

**Recommended setup for next time:**
- 300-400GB swap on NVMe (`/scratch/swapfile`) — NOT on EBS
- Use `save_sharded_checkpoint=True` in NeuronConfig to pre-shard during compilation
  - Compile: loads full state dict (1.98TB peak) → saves 64 per-rank files to `/scratch`
  - Load: reads only 21GB per rank → peak RAM drops to ~21GB (no swap needed!)
- Or: pre-convert weights offline to BF16 neuron format and upload to S3 for reuse

**Safety measures:**
- 100GB swap on EBS root (`/swapfile`) + 300GB swap on NVMe (`/scratch/swapfile`)
- Memory monitoring in generation script (logs RSS every 30s)
- Run under `screen` session (survives SSH disconnects)

### Issues Encountered

- **tp=32 OOM (NCC_EVRF009)**: With 256 MoE experts, all expert weights live on every TP rank
  (only intermediate dim is sharded). At tp=32 with lnc=2, the NEFF needs 40GB vs 24GB per-physical-core
  limit. Fixed by using **tp=64 with lnc=2** (64 logical cores, each with 24GB HBM).
- **lnc=1 HBM OOM**: Initial attempt used `lnc = max(1, 64 // tp_degree)` giving lnc=1 for tp=64.
  With lnc=1, trn2 exposes 128 visible NCs — but pairs share 24GB HBM banks. Two ranks (~21.4GB each)
  need ~42.8GB in one 24GB bank → `Failed to allocate 224.000MB on ND 7:NC 0`.
  **Fix**: Always use lnc=2 on trn2 (the platform default). Changed to `lnc = 2` in generation script.
- **Cached failed NEFF**: Previous compilation attempts that were killed (SIGHUP) leave cached failures.
  Clear with: `rm -rf /var/tmp/neuron-compile-cache/`
- **No generation_config.json**: The official DeepSeek-V3-0324 repo doesn't include one. The script
  creates a default GenerationConfig.
- **TP/KV heads warning**: "TP degree (64) and KV heads (128) are not divisible" warnings are harmless —
  MLA attention doesn't use the standard GQA sharding path (num_key_value_heads=1 in config).
- **Previous freeze (2026-03-17)**: Weights deleted from disk; no cached NEFFs survived. Likely caused
  by either (a) extremely slow NCC compilation for 671B MoE misinterpreted as a hang, or (b) OOM during
  weight sharding without swap. Now mitigated with swap + monitoring.
- **Near-OOM during load (2026-03-17 run 2)**: Compilation succeeded in 14.9m (only 6GB RSS).
  Weight loading hit 1983GB RSS + 89GB swap. Saved by adding 300GB NVMe swap mid-run.
  Swap thrashing on EBS made loading take ~2 hours instead of ~10 minutes.
  **Root cause**: framework loads full state dict + 64 per-rank shards all in memory;
  Python doesn't return freed allocations to OS → phantom RSS inflation.
  **Fix**: use `save_sharded_checkpoint=True` and NVMe swap from the start.

### Benchmark Results

**Run 1 (2026-03-17, lnc=1 — failed at device load):**

| Metric | Value | Notes |
|--------|-------|-------|
| **HLO generation** | 28s | Context encoding: 11.5s, Token generation: 4.9s |
| **NCC compilation** | 315s (5.2m) | Both NEFFs compiled, `Compiler status PASS` |
| **Model build** | 893s (14.9m) | Includes HLO + NCC + weight layout transform |
| **Compilation RSS** | 6GB peak | Weights not loaded during compilation |
| **Weight sharding** | 12,834s (3.56h) | 64 per-rank files × 21.4GB = 1.37TB |
| **Weight loading RSS** | 1,983GB peak | +89GB swap; NVMe swap would dramatically reduce time |
| **Device load** | FAILED | HBM OOM — lnc=1 causes two ranks to share 24GB bank |

**Run 2 (2026-03-18, lnc=2 — SUCCESS):**

| Metric | Value | Notes |
|--------|-------|-------|
| **HLO generation** | 22s | Context encoding + Token generation |
| **NCC compilation** | ~5m | `Compiler status PASS`, `--lnc=2` flag |
| **Model build** | 711s (11.8m) | Slightly faster than run 1 |
| **Compilation RSS** | 6GB peak | Same as run 1 |
| **Weight sharding** | 12,441s (3.46h) | 64 × 21.4GB files, peak RSS 1.98TB, 67GB swap |
| **Load time (sharded)** | 467s (7.8m) | Fast path from pre-sharded checkpoints |
| **Load RSS** | 1,344GB peak | Reading all 64 per-rank files, dropped to 34GB for inference |
| **TPOT** | **36.1ms** | Time per output token |
| **Throughput** | **27.7 tok/s** | 113 tokens in 4.08s |
| **Input tokens** | 6 | "The capital of France is" |
| **Output tokens** | 113 | Coherent, correct response about Paris |

**Output**: "The capital of France is Paris, which is one of the most important and influential
cities in the world. Paris is located in the northern part of France, on the banks of the
Seine River. It is known for its rich history, culture, art, fashion, and cuisine..."

**Run 3 (2026-03-18, NXDI native `benchmark_sampling` — 20 iterations):**

Same compiled model as Run 2. Uses NXDI's `benchmark_sampling()` API with random 256-token input,
generating up to max_length=512 (256 new tokens), 20 timed runs. Sub-model latencies measured via
forward hooks. Report saved to `/tmp/deepseek_v3_benchmark_report.json`.

| Component | p50 (ms) | p90 (ms) | p99 (ms) | p100 (ms) | avg (ms) | throughput |
|-----------|----------|----------|----------|-----------|----------|------------|
| **e2e_model** | 7,057 | 7,071 | 7,077 | 7,077 | 7,057 | 72.6 tok/s |
| **context_encoding_model** | 1,667 | 1,668 | 1,668 | 1,668 | 1,667 | 307.1 tok/s |
| **token_generation_model** | 20.5 | 20.9 | 21.2 | 37.5 | 20.6 | 48.7 tok/s |

Key observations:
- **TPOT = 20.5ms** (p50) — significantly better than Run 2's 36.1ms. Run 2 measured TPOT from
  E2E wall-clock time (includes Python overhead + sampling). The NXDI benchmark isolates the
  token_generation NEFF latency, which is the true per-token hardware cost.
- **Prefill (context encoding) = 1.67s** for 256 input tokens. This is the TTFT component.
  Throughput = 307 tok/s for encoding.
- **E2E = 7.06s** for ~256 new tokens ≈ prefill (1.67s) + 256 × 20.6ms (5.27s) + sampling/Python
  overhead (0.12s). Very little overhead.
- **p100 outlier** of 37.5ms on token_generation is likely a warmup artifact (the first measured
  run after the warmup iteration).
- **Stable performance**: p50 to p99 spread is <1ms for token generation, <1ms for context
  encoding, and <20ms for E2E across 20 runs.

Script: `examples/benchmark_deepseek_v3.py`

### Configuration Details
```python
MoENeuronConfig(
    tp_degree=64,        # Use all 64 logical NeuronCores (lnc=2)
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,  # MUST be 2 on trn2 — lnc=1 causes HBM OOM
    enable_bucketing=False,
    flash_decoding_enabled=False,
    save_sharded_checkpoint=True,  # Pre-shard during compile, fast load path
)
```

### save_sharded_checkpoint Flow
When `save_sharded_checkpoint=True`:
- **During compile**: After NCC compilation, loads full HF state dict (~641GB FP8), dequantizes to BF16
  (~1.3TB), then creates 64 per-rank checkpoint files at `{traced_model_path}/weights/tp{rank}_sharded_checkpoint.safetensors`
  (~21.4GB each, ~1.37TB total). This is the slow path (~3.5 hours).
- **During load** (with `--skip-compile`): Reads only the per-rank file for each rank (~21.4GB).
  Peak RAM drops to ~21GB per rank — no swap needed. This is the fast path (~minutes).
- The sharded checkpoints are **LNC-independent** — they can be reused across lnc=1 and lnc=2 configs.
  Only the NEFFs (model.pt) need to be recompiled when changing LNC.

### Running the Benchmark
```bash
# 1. Set up NVMe scratch + swap (one-time, requires sudo)
sudo mkfs.ext4 -L scratch /dev/nvme1n1
sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch && sudo chown ubuntu:ubuntu /scratch
sudo mkfs.ext4 -L scratch2 /dev/nvme2n1
sudo mkdir -p /scratch2 && sudo mount /dev/nvme2n1 /scratch2 && sudo chown ubuntu:ubuntu /scratch2
sudo fallocate -l 400G /scratch2/swapfile && sudo chmod 600 /scratch2/swapfile
sudo mkswap /scratch2/swapfile && sudo swapon /scratch2/swapfile

# 2. Clear any stale compile cache
rm -rf /var/tmp/neuron-compile-cache/

# 3. Full run (compile + shard weights + load + generate):
screen -S deepseek
cd ~/environment/deepseek-nxdi
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python examples/generation_deepseek_v3.py \
    --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \
    --traced-model-path /scratch/deepseek_v3_traced \
    --tp-degree 64 --seq-len 512 --batch-size 1 --max-new-tokens 128 \
    2>&1 | tee /tmp/deepseek_benchmark.log
# Detach: Ctrl-A D    Reattach: screen -r deepseek

# 4. Fast resume (skip compile, use pre-sharded weights):
python examples/generation_deepseek_v3.py \
    --traced-model-path /scratch/deepseek_v3_traced --skip-compile \
    --tp-degree 64 --seq-len 512 --batch-size 1 --max-new-tokens 128
```

### Fresh Instance: Time to First Token

Three paths from fastest to slowest, depending on which artifacts are available:

**Path A: Restore everything from S3 (~45 min)**
Requires: S3 bucket with all artifacts (NEFFs + sharded weights + code).

| Step | Time | What |
|------|------|------|
| NVMe setup + swap | 2 min | Format nvme1n1/nvme2n1, mount, create swap |
| Download NEFFs from S3 | 1 min | 96MB model.pt + tokenizer + config |
| Download sharded weights from S3 | ~30 min | 1.37TB at ~750 MB/s |
| Install code | 1 min | `bash install_deepseek.sh` |
| Load model + generate | ~8 min | Reads per-rank files, loads to HBM |
| **Total** | **~42 min** | |

**Path B: Recompile NEFFs, use sharded weights from S3 (~55 min)**
Requires: S3 sharded weights + HF config files (for compilation graph).

| Step | Time | What |
|------|------|------|
| NVMe setup + swap | 2 min | |
| Download sharded weights from S3 | ~30 min | 1.37TB |
| Install code | 1 min | |
| Compile NEFFs (lnc=2) | ~12 min | HLO gen + NCC, only 6GB RSS |
| Load model + generate | ~8 min | |
| **Total** | **~55 min** | |

**Path C: Full rebuild from HF weights (~4.5 hours)**
Requires: Only HF weights (from S3 or HuggingFace).

| Step | Time | What |
|------|------|------|
| NVMe setup + swap | 2 min | |
| Download HF weights | ~40 min | 642GB from S3 or HuggingFace |
| Install code | 1 min | |
| Compile NEFFs | ~12 min | HLO gen + NCC, 6GB RSS |
| Weight sharding | ~3.5 hours | FP8 load → BF16 dequant → 64 per-rank files (2TB peak RSS) |
| Load model + generate | ~8 min | |
| **Total** | **~4.5 hours** | |

**What to persist (priority order):**

| Artifact | Size | Saves | Priority |
|----------|------|-------|----------|
| Sharded checkpoints | 1.37TB | 3.5 hours (weight sharding) | Critical |
| Compiled NEFFs | 96MB | 12 min (compilation) | Nice to have |
| Source code + PLAN | 8MB | Rebuild from scratch | Essential |
| HF weights | 642GB | 40 min HF download | Optional (can re-download) |

The **sharded checkpoints are by far the most valuable** — they save 3.5 hours of CPU-intensive
work that requires 2TB RAM + swap. NEFFs are cheap to recompile (12 min, 6GB RSS). HF weights
can be re-downloaded from HuggingFace if needed.

---

## Spot Instance Recovery

### S3 Bucket: `deepseek-v3-nxdi-artifacts`

**Bucket contents** (prefix: `deepseek-v3-0324/`):

| S3 Path | Size | Description |
|---------|------|-------------|
| `traced_model/model.pt` | ~96MB | Compiled NEFFs (lnc=2, tp=64) — saves ~12 min compilation |
| `traced_model/neuron_config.json` | ~50KB | Model config (logical_nc_config=2) |
| `traced_model/tokenizer*` | ~10MB | Tokenizer files |
| `sharded_weights/tp{0..63}_sharded_checkpoint.safetensors` | ~1.37TB | 64 per-rank checkpoints — saves ~3.5h sharding |
| `hf_weights/` | ~642GB | Original HF FP8 weights (DeepSeek-V3-0324) |
| `neuron_compile_cache.tar.gz` | ~75MB | NCC compile cache |
| `deepseek_nxdi_code.tar.gz` | ~8MB | Full source code |
| `PLAN_deepseek_v3.md` | ~20KB | This file |

**Save/restore scripts:**
```bash
# Save all artifacts to S3:
bash scripts/save_to_s3.sh deepseek-v3-nxdi-artifacts

# Restore on a new instance (sets up NVMe, swap, restores everything):
bash scripts/restore_from_s3.sh deepseek-v3-nxdi-artifacts
```

### What survives spot reclamation
- **EBS root volume**: Survives IF `DeleteOnTermination: false` — contains HF weights (642GB)
- **NVMe instance store**: LOST — compiled model, swap, scratch all gone
- **Memory**: LOST — all in-flight state gone

### Resuming on a new instance
```bash
# Option A: Restore from S3 (if saved)
bash scripts/restore_from_s3.sh deepseek-v3-nxdi-artifacts

# Option B: Fresh start (if no S3 backup)
# 1. Set up NVMe scratch + swap
sudo mkfs.ext4 -L scratch /dev/nvme1n1
sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch && sudo chown ubuntu:ubuntu /scratch
sudo fallocate -l 400G /scratch/swapfile && sudo chmod 600 /scratch/swapfile
sudo mkswap /scratch/swapfile && sudo swapon /scratch/swapfile

# 2. Install code
cd ~/environment/deepseek-nxdi && bash install_deepseek.sh

# 3. Download HF weights (if EBS was lost, ~40 min)
huggingface-cli download deepseek-ai/DeepSeek-V3-0324 \
    --local-dir ~/environment/models/DeepSeek-V3-0324-FP8

# 4. Run benchmark under screen
screen -S deepseek
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python examples/generation_deepseek_v3.py \
    --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \
    --traced-model-path /scratch/deepseek_v3_traced \
    --tp-degree 64 --seq-len 512 --batch-size 1 --max-new-tokens 128 \
    2>&1 | tee /tmp/deepseek_benchmark.log
```

### Optimized resume (IMPLEMENTED)
`save_sharded_checkpoint=True` is now enabled in `generation_deepseek_v3.py`.
Sharded checkpoints are uploaded to S3 bucket `deepseek-v3-nxdi-artifacts`.

**Fast recovery on a new instance:**
1. Restore from S3: `bash scripts/restore_from_s3.sh deepseek-v3-nxdi-artifacts`
2. This downloads: NEFFs (~77MB), sharded weights (~1.37TB), source code
3. Run with `--skip-compile` — loads pre-sharded weights directly (~21GB per rank)
4. Total recovery time: ~30 min (download shards) + ~5 min (load to device)
5. Skips: 12 min compilation + 3.5 hour weight sharding

---

## Phase 10: Neuron Profiling (COMPLETE)

### Goal
Profile NEFFs to identify performance bottlenecks in token generation and context encoding.

### Method
Used `neuron-profile inspect` (v2.28.23.0) wrapping an NXDI benchmark script. Captures
system-level runtime trace (NEFF load, execute, collectives, DMA) across all 64 NCs.

**Device-level profiling (NTFF) was not possible** — HBM is 102% utilized (24.4/24 GB),
leaving no room for hardware profiling buffers. `neuron-profile capture` also failed with
TP=64 (collectives worker setup hung).

### Profiling Script
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
neuron-profile inspect -o /scratch/profiles/inspect_output -- \
    python examples/profile_deepseek_v3.py \
        --traced-model-path /scratch/deepseek_v3_traced \
        --num-iterations 3 --max-new-tokens 16
```

### Results

**Token Generation (TPOT)**:

| Metric | Value |
|--------|-------|
| Median TPOT | **18.9 ms** (52.9 tok/s) |
| p10-p90 range | 18.7 - 19.3 ms |
| Steady-state stdev | ~0.3 ms (very stable) |
| Warmup outlier | 163 ms (first execution) |

**Context Encoding (TTFT)**:

| Metric | Value |
|--------|-------|
| Median TTFT | **1,666 ms** |
| Range | 1,665 - 1,733 ms |

**Per-NC Straggler Analysis** (all 64 NCs profiled):

| NC | Mean Latency | Notes |
|----|-------------|-------|
| NC 53 (fastest) | 18.93 ms | Baseline |
| NC 12 (slowest) | 22.82 ms | +20.5% straggler |
| **Gap** | **3.9 ms** | TP barrier sync means slowest NC determines step time |

**Collectives (All-Reduce)**:

| Metric | Value |
|--------|-------|
| CC events per token gen step | ~12 |
| Median per-event duration | 29.4 us |
| p99 per-event duration | 238 us |
| Estimated CC per step | ~3 ms (~16% of TPOT) |

**HBM Usage (per NC)**:

| Component | Size | % |
|-----------|------|---|
| Tensors (weights + KV) | 23.07 GB | 94.4% |
| Shared Scratchpad | 1.07 GB | 4.4% |
| Collectives buffers | 0.17 GB | 0.7% |
| Other | 0.13 GB | 0.5% |
| **Total** | **24.44 GB** | **~102%** |

**Estimated TPOT Breakdown**:

| Component | Time | % | Notes |
|-----------|------|---|-------|
| Compute (matmul, attention, MoE) | ~12 ms | ~63% | 61 layers |
| Collectives (all-reduce, TP=64) | ~3 ms | ~16% | 12 CC events/step |
| Memory access (weight reads) | ~3 ms | ~16% | 21 GB per step |
| Straggler overhead | ~1 ms | ~5% | NC sync at barriers |
| **Total** | **~19 ms** | **100%** | |

### Optimization Opportunities

1. **Reduce straggler gap (potential -1-2ms)**: NC 12 is 20% slower than NC 53.
   Investigate NUMA affinity, collectives topology, HBM bank contention.
2. **Reduce collectives overhead (potential -1ms)**: 64-way all-reduce is expensive.
   Expert parallelism (EP) could replace some TP sharding for MoE layers.
3. **FP8 on-device inference (potential -3ms)**: Keep weights in FP8, halving memory
   bandwidth and freeing HBM for larger batch sizes.
4. **Eliminate layout_opt NEFF (-163ms per sequence)**: A 163ms weight layout
   transform runs between prefill and decode. Unifying layouts would remove this.

### Artifacts

| File | Size | Location |
|------|------|----------|
| System trace (Perfetto) | 452 MB | `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/profiles/deepseek_v3_system.pftrace` |
| Profile report | 5 KB | `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/profiles/PROFILE_REPORT.md` |
| Full JSON trace | 1.5 GB | `/scratch/profiles/system_profile.json` (local only, too large for S3) |
| Raw protobuf | 578 MB | `/scratch/profiles/inspect_output/` (local only) |

To visualize the Perfetto trace: download from S3 and open at https://ui.perfetto.dev/

---

## Phase 11: GuideLLM Throughput Optimization (COMPLETE)

### Goal
Find the optimal batch size for maximum throughput on trn2.48xlarge by serving DeepSeek V3 via
vLLM and benchmarking with [GuideLLM](https://github.com/vllm-project/guidellm).

Reference: [HuggingFace Benchmark Guide](https://huggingface.co/docs/optimum-neuron/en/guides/benchmark)

### Background

The NXDI native benchmark (Run 3) measured **20.5ms TPOT** at batch_size=1, seq_len=512.
With batch_size=1, compute is memory-bound — the NeuronCores spend most time reading weights,
not computing. Larger batch sizes amortize the weight-read cost across more sequences,
shifting toward compute-bound and increasing aggregate throughput (tokens/sec) at the cost of
higher per-request latency.

**Key constraint**: Each logical NeuronCore has 24GB HBM. Weights consume ~21.4GB, leaving
~1-2.6GB for the NEFF program, KV cache, and activations.

### HBM Budget Analysis

KV cache per batch element = 61 layers × seq_len × 576 (kv_lora_rank + rope_dim) × 2 (K+V) × 2 bytes.

| seq_len | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 |
|---------|------|------|------|------|-------|
| 512 | 0.07 GB | 0.13 GB | 0.27 GB | 0.54 GB | 1.07 GB |
| 2048 | 0.27 GB | 0.54 GB | 1.07 GB | 2.14 GB | OOM |
| 4096 | 0.54 GB | 1.07 GB | 2.14 GB | OOM | OOM |

With ~1.1GB headroom (conservative), safe configurations:
- **seq_len=512**: bs=1,2,4,8 all fit; bs=16 tight but possible
- **seq_len=2048**: bs=1,2,4 fit; bs=8 tight
- **seq_len=4096**: bs=1,2 fit; bs=4 tight

Note: Activation buffers during forward pass add transient memory. Actual limits must be
discovered empirically — the NEFF compiler will report OOM (NCC_EVRF009) if it can't fit.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  GuideLLM (client)                                  │
│  Sends OpenAI-compatible requests at various rates  │
│  Measures: TTFT, ITL, throughput, request latency   │
├─────────────────────────────────────────────────────┤
│            HTTP (localhost:8000/v1)                  │
├─────────────────────────────────────────────────────┤
│  vLLM server (continuous batching)                  │
│  --max-num-seqs = batch_size (sets NXDI batch_size) │
│  --max-model-len = seq_len                          │
├─────────────────────────────────────────────────────┤
│  NXDI DeepSeek V3 (tp=64, lnc=2)                   │
│  Compiled NEFF + sharded weights on NeuronCores     │
└─────────────────────────────────────────────────────┘
```

DeepSeek V3 uses all 64 NeuronCores (tp=64), so no data parallelism is possible.
The only throughput lever is **batch size** (= `--max-num-seqs` in vLLM).

### Critical vLLM + Neuron Configuration

vLLM's default Neuron config **does not set** `logical_nc_config` or disable bucketing.
This causes compilation failures on trn2 (NCC_IBIR297 internal compiler error without lnc=2,
or HBM OOM with bucketing enabled). Must always pass `--additional-config` with overrides:

```bash
--additional-config '{"override_neuron_config": {"logical_nc_config": 2, "enable_bucketing": false, "save_sharded_checkpoint": true}}'
```

Required overrides:
- **`logical_nc_config: 2`** — MANDATORY on trn2. Without this, compilation fails with NCC_IBIR297.
- **`enable_bucketing: false`** — Avoids compilation overhead and potential compiler bugs.
- **`save_sharded_checkpoint: true`** — Enables fast weight loading from pre-sharded checkpoints.

### Fast Weight Loading via Pre-Sharded Checkpoints

**Problem**: vLLM's default weight loading path dequantizes FP8→BF16 and shards weights
at runtime, taking **~2+ hours** for DeepSeek V3 671B (peak RSS ~2TB).

**Solution**: Use `NEURON_COMPILED_ARTIFACTS` env var + pre-sharded weights from Phase 8.

The sharded weights from Phase 8 (`/scratch/deepseek_v3_traced/weights/`, 64 files, 1.4TB)
are **batch-size-independent** — same tp=64 rank partitioning, same tensor shapes, same
dequantized BF16 format. Only NEFFs change with batch_size.

**Fast startup recipe (per batch_size)**:

```bash
BS=2  # or 4, 8, etc.
ARTIFACTS_DIR=/scratch/vllm_bs${BS}

# Step 1: Compile NEFFs only (~2.5 min, uses Neuron compile cache if available)
mkdir -p $ARTIFACTS_DIR
screen -dmS vllm-compile bash -c "
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export NEURON_COMPILED_ARTIFACTS=$ARTIFACTS_DIR
VLLM_PLUGINS=neuron vllm serve ~/environment/models/DeepSeek-V3-0324-FP8 \
    --tensor-parallel-size 64 --max-model-len 512 --max-num-seqs $BS \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --port 8000 \
    --additional-config '{\"override_neuron_config\": {\"logical_nc_config\": 2, \"enable_bucketing\": false, \"save_sharded_checkpoint\": true}}' \
    2>&1 | tee /tmp/vllm_compile_bs\${BS}.log
"

# Step 2: Watch for NEFF compilation completion
# Look for: "Finished Compilation for all HLOs"
# Then kill immediately (before slow weight loading starts):
pkill -9 -f 'vllm serve'

# Step 3: Symlink Phase 8 sharded weights
rmdir $ARTIFACTS_DIR/weights 2>/dev/null  # remove empty dir from compile
ln -s /scratch/deepseek_v3_traced/weights $ARTIFACTS_DIR/weights

# Step 4: Restart with pre-compiled artifacts (~8 min load)
screen -dmS vllm-serve bash -c "
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export NEURON_COMPILED_ARTIFACTS=$ARTIFACTS_DIR
VLLM_PLUGINS=neuron vllm serve ~/environment/models/DeepSeek-V3-0324-FP8 \
    --tensor-parallel-size 64 --max-model-len 512 --max-num-seqs $BS \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --port 8000 \
    --additional-config '{\"override_neuron_config\": {\"logical_nc_config\": 2, \"enable_bucketing\": false, \"save_sharded_checkpoint\": true}}' \
    2>&1 | tee /tmp/vllm_serve_bs\${BS}.log
"
# Wait for: "Application startup complete." (~8 min)
```

**Key insight**: `NEURON_COMPILED_ARTIFACTS` env var bypasses vLLM's default behavior of
wiping the compiled artifacts directory on each startup (`shutil.rmtree` in
`_get_compiled_model_path()`). Without this env var, pre-compiled artifacts are always deleted.

**Timing comparison**:
| Path | Compile | Weight Load | Total |
|------|---------|-------------|-------|
| Slow (default vLLM) | ~2.5 min | ~2+ hours (FP8 dequant + shard) | ~2.5 hours |
| Fast (pre-sharded) | ~2.5 min (or 1s from cache) | ~8 min (NVMe read) | ~10 min |

### GuideLLM CLI

GuideLLM v0.5.4 — the CLI differs from older docs. Correct invocation:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export GUIDELLM__REQUEST_TIMEOUT=120

guidellm benchmark run \
    --target "http://localhost:8000" \
    --model "/home/ubuntu/environment/models/DeepSeek-V3-0324-FP8" \
    --data '{"prompt_tokens": 200, "prompt_tokens_variance": 50, "output_tokens": 250, "output_tokens_variance": 50}' \
    --profile sweep \
    --max-seconds 120 \
    --output-path /tmp/guidellm_bs${BS}_seq512.json \
    --disable-console-interactive
```

The `sweep` profile runs 10 benchmarks: synchronous, throughput, and 8 constant rates.
Each runs for up to `--max-seconds` (120s). Total: ~20 min per batch_size.

Note: GuideLLM sends requests as `chat_completions` (not `text_completions`).

### Results

#### bs=1, seq_len=512 (COMPLETE)

GuideLLM sweep (10 benchmarks, 120s each) — key latency metrics (completed requests):

| Profile | TTFT (ms) p50 | ITL (ms) p50 | TPOT (ms) p50 | Request Latency (s) p50 |
|---------|---------------|--------------|---------------|------------------------|
| synchronous | 1,710 | 22.2 | 28.9 | 7.2 |
| throughput | 52,238 | 22.3 | 231.1 | 57.8 |
| constant (all 8) | ~1,681 | ~22.1 | ~28.8 | ~7.2 |

Server throughput (all requests):

| Profile | Requests/s (median) | Concurrency (median) | Output tok/s (median) | Total tok/s (median) |
|---------|---------------------|----------------------|-----------------------|----------------------|
| synchronous | 0.1 | 1.0 | 45.0 | 63.2 |
| throughput | 0.1 | 512.0 | 44.8 | 44.9 |
| constant (all 8) | 0.1 | 1.0 | ~45.1 | ~61.3 |

**Key findings for bs=1:**
- **ITL of 22.1ms** closely matches NXDI native benchmark TPOT of 20.5ms — minimal vLLM overhead (~2ms)
- TTFT ~1.7s for 200-token prompt (context encoding NEFF latency)
- With bs=1, throughput profile shows 512 concurrent queued requests but only 1 can run at a time,
  so TPOT balloons to 231ms (queueing delay). Output tok/s stays ~45 regardless of load.
- The constant rate profiles all converge to the same metrics as synchronous — the server is
  saturated at ~0.14 req/s (1 req every ~7.2s)
- **Single-request throughput: ~45 output tok/s, ~63 total tok/s**

Report saved: `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/benchmarks/guidellm_bs1_seq512.json`

#### bs=2, seq_len=512 (COMPLETE)

Used fast startup recipe (compile NEFFs in ~1s from cache + load sharded weights in ~8 min).
Compiled artifacts: `/scratch/vllm_bs2/` (model.pt=109MB, neuron_config.json, weights/ → symlink)

| Profile | TTFT (ms) p50 | ITL (ms) p50 | TPOT (ms) p50 | Request Latency (s) p50 |
|---------|---------------|--------------|---------------|------------------------|
| synchronous | 1,670 | 30.5 | 37.1 | 9.3 |
| throughput | 56,401 | 37.2 | 262.9 | 65.7 |
| constant (highest) | ~1,683 | ~37.1 | ~43.8 | ~11.0 |

Server throughput at highest constant rate: **75.5 total tok/s** (concurrency ~1.8), vs 61.3 for bs=1.

| Profile | Concurrency (median) | Output tok/s (median) | Total tok/s (mean) |
|---------|----------------------|-----------------------|--------------------|
| synchronous | 1.0 | 32.8 | 49.5 |
| throughput | 512.0 | 53.7 | 1,789.7 |
| constant (highest rate) | 2.0 | 52.9 | 75.5 |

**Key findings for bs=2:**
- **ITL increased from 22ms (bs=1) to 30-37ms (bs=2)** — each decode step processes 2 sequences
- **Per-request latency increased**: 9.3s vs 7.2s (sync), due to higher per-token cost
- **Aggregate throughput improved**: 75.5 total tok/s (highest constant rate) vs 61.3 (bs=1) — **+23%**
- **Throughput profile**: with max concurrency, output tok/s = 53.7 (median) vs 44.8 (bs=1) — **+20%**
- The server can now process ~2 concurrent requests, with the constant rate profiles showing
  increasing concurrency (1.0 → 2.0) as the request rate ramps up

Report saved: `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/benchmarks/guidellm_bs2_seq512.json`

#### bs=4, seq_len=512 (COMPLETE)

Used fast startup recipe (compile NEFFs in ~2.6 min + load sharded weights in ~37s).
Compiled artifacts: `/scratch/vllm_bs4/` (model.pt=120MB, neuron_config.json, weights/ → symlink)

| Profile | Completed | ITL mean (ms) | ITL p50 (ms) | TTFT mean (ms) | Total throughput (tok/s) |
|---------|-----------|---------------|--------------|----------------|------------------------|
| synchronous | 14 | 37.1 | 37.1 | 1,690.7 | 23.3 |
| throughput | 32 | 50.6 | 49.8 | 38,831.8 | 53.3 |
| constant 0.128 | 15 | 45.5 | 45.5 | 1,684.4 | 25.0 |
| constant 0.148 | 17 | 45.6 | 45.6 | 1,689.9 | 28.3 |
| constant 0.168 | 20 | 45.6 | 45.6 | 1,690.1 | 33.3 |
| constant 0.188 | 21 | 53.9 | 53.9 | 1,690.2 | 35.0 |
| constant 0.207 | 23 | 54.0 | 54.0 | 1,692.0 | 38.3 |
| constant 0.227 | 26 | 54.0 | 54.0 | 1,691.3 | 43.3 |
| constant 0.247 | 27 | 62.2 | 62.3 | 1,687.8 | 45.0 |
| constant 0.267 | 29 | 62.2 | 62.2 | 1,691.0 | 48.3 |

**Key findings for bs=4:**
- **ITL jumps in discrete steps**: ~37ms (1 concurrent) → ~45ms (2 concurrent) → ~54ms (3) → ~62ms (4)
  Each step corresponds to an additional active request in the batch
- **Synchronous ITL = 37ms** vs 22ms (bs=1) — the compiled NEFF always processes 4 slots even with 1 active request, so per-token cost is higher even for single requests
- **Max throughput: 53.3 tok/s** (throughput profile) vs 33 (bs=1), 42 (bs=2) — **+60% over bs=1**
- **Best constant rate: 48.3 tok/s** at 0.267 req/s with 62.2ms ITL
- At highest constant rate, all 4 batch slots are utilized, approaching the throughput profile limit

Report saved: `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/guidellm_bs4_seq512.json`

#### bs=8, seq_len=512 (COMPLETE)

Used fast startup recipe (compile NEFFs in ~2.7 min + load sharded weights in ~41s).
Compiled artifacts: `/scratch/vllm_bs8/` (model.pt=140MB, neuron_config.json, weights/ → symlink)

| Profile | Completed | ITL mean (ms) | ITL p50 (ms) | TTFT mean (ms) | Total throughput (tok/s) |
|---------|-----------|---------------|--------------|----------------|------------------------|
| synchronous | 11 | 47.6 | 47.6 | 1,698.9 | 18.3 |
| throughput | 40 | 77.0 | 77.0 | 39,426.3 | 66.7 |
| constant 0.115 | 13 | 56.0 | 56.0 | 1,692.8 | 21.7 |
| constant 0.146 | 17 | 56.0 | 56.0 | 1,696.7 | 28.3 |
| constant 0.177 | 20 | 64.4 | 64.4 | 1,697.1 | 33.3 |
| constant 0.208 | 22 | 72.3 | 72.7 | 1,692.0 | 36.7 |
| constant 0.240 | 26 | 72.8 | 72.8 | 1,695.7 | 43.3 |
| constant 0.271 | 29 | 81.1 | 81.1 | 1,697.7 | 48.3 |
| constant 0.302 | 32 | 89.6 | 89.5 | 1,693.1 | 53.3 |
| constant 0.333 | 33 | 106.3 | 106.3 | 1,693.2 | 55.0 |

**Key findings for bs=8:**
- **Synchronous ITL = 47.6ms** — highest of all batch sizes, as expected (NEFF processes 8 slots)
- **Max throughput: 66.7 tok/s** (throughput profile) — **+100% over bs=1** (33 tok/s)
- **ITL degrades quickly**: from 48ms (sync) to 106ms at highest load — 4.8x the bs=1 ITL
- **Best constant rate: 55.0 tok/s** at 0.333 req/s, but ITL = 106ms (unacceptable for interactive use)
- At 48.3 tok/s (0.271 req/s), ITL is 81ms — still significantly higher than bs=4 at same throughput
- Diminishing returns vs bs=4: only +13% more throughput (66.7 vs 53.3) but 2x higher ITL

Report saved: `s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/guidellm_bs8_seq512.json`

### Cross-Batch-Size Comparison

**Summary Table** (DeepSeek V3 671B, tp=64, lnc=2, seq_len=512, 200 input / 200 output tokens):

| Metric | bs=1 | bs=2 | bs=4 | bs=8 |
|--------|------|------|------|------|
| Synchronous ITL (ms) | 22.2 | 30.5 | 37.1 | 47.6 |
| Max throughput (tok/s) | 33.3 | 41.7 | 53.3 | 66.7 |
| Best constant-rate throughput (tok/s) | 33.3 | 39.6 | 48.3 | 55.0 |
| Best constant-rate ITL (ms) | 22.1 | 37.1 | 62.2 | 106.3 |
| TTFT (ms) | ~1,710 | ~1,670 | ~1,690 | ~1,699 |
| NEFF compilation time | 2.5 min | ~1s (cached) | 2.6 min | 2.7 min |
| Weight loading (sharded) | 37s | 37s | 37s | 41s |

**Throughput scaling** (max throughput vs bs=1):

| bs | Max throughput | Improvement | Throughput efficiency |
|----|---------------|-------------|---------------------|
| 1 | 33.3 tok/s | baseline | 100% |
| 2 | 41.7 tok/s | +25% | 62.5% per slot |
| 4 | 53.3 tok/s | +60% | 40% per slot |
| 8 | 66.7 tok/s | +100% | 25% per slot |

**Recommendations:**
- **Interactive / low-latency**: Use **bs=1** (22ms ITL, 33 tok/s)
- **Balanced throughput + latency**: Use **bs=2** (30-37ms ITL, 40-42 tok/s, +25% throughput)
- **Throughput-optimized**: Use **bs=4** (37-62ms ITL, 48-53 tok/s, +60% throughput)
  - Best ITL/throughput tradeoff: sub-50ms ITL at ~40 tok/s
- **Maximum throughput (batch workloads)**: Use **bs=8** (66.7 tok/s, +100%)
  - Only for offline/batch — ITL degrades to 77-106ms

**Key insight**: Throughput scales sub-linearly with batch size due to the memory-bandwidth bottleneck.
Each doubling of batch size gives diminishing returns: +25% (bs=2) → +28% more (bs=4) → +25% more (bs=8).
The 671B model's 256 experts mean significant per-token memory traffic regardless of batch size.

### Upload results to S3

All results uploaded:
```
s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/guidellm_bs1_seq512.json
s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/guidellm_bs2_seq512.json
s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/guidellm_bs4_seq512.json
s3://deepseek-v3-nxdi-artifacts/deepseek-v3-0324/guidellm_bs8_seq512.json
```

---

## NXDI Onboarding

Reference: [NxD Inference Model Onboarding Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html)

NxD Inference provides three evaluation tools for onboarded models: **Logit Matching**, **Token Matching**, and **Benchmarking**. These are the standard APIs used to validate that an onboarded model produces correct outputs on Neuron and to measure its performance.

### Logit Matching (`check_accuracy_logits_v2`)

Verifies that output logits from the Neuron model are within certain tolerances of expected "golden" logits (typically generated by the HF model on CPU).

**How it works:**
1. Runs greedy generation on Neuron device.
2. Compares output logits against expected logits at each token position.
3. If output tokens diverge from expected, re-runs generation from the point of divergence (teacher-forcing) and checks again. Repeats until all positions are covered.
4. Compares logit values for the top-k highest scoring tokens using absolute and relative tolerances.

**Default tolerances:**

| Parameter | Value |
|-----------|-------|
| Divergence difference tolerance | `0.001` |
| Absolute tolerance (all top-k) | `1e-5` |
| Relative tolerance (top-k=5) | `0.01` |
| Relative tolerance (top-k=50) | `0.02` |
| Relative tolerance (top-k=1000) | `0.03` |
| Relative tolerance (top-k=None, all tokens) | `0.05` |

**API signature:**
```python
from neuronx_distributed_inference.utils.accuracy import (
    check_accuracy_logits_v2,
    generate_expected_logits,
    prepare_inputs_from_prompt,
)

# Step 1: Prepare inputs
inputs = prepare_inputs_from_prompt(neuron_model, tokenizer, prompt="Hello, I am a language model")
# Returns tokenizer outputs with .input_ids and .attention_mask

# Step 2: Generate golden logits (HF model on CPU)
expected_logits = generate_expected_logits(
    neuron_model,          # NeuronApplicationBase — used to load HF model via .load_hf_model()
    inputs.input_ids,      # [batch_size, input_len]
    inputs.attention_mask,  # [batch_size, input_len]
    generation_config,     # HuggingFace GenerationConfig
    num_tokens=None,       # Optional: limit number of tokens (default: seq_len - input_len)
    tokenizer=tokenizer,   # Optional: for debug logging
)
# Returns: torch.Tensor of shape [num_tokens, batch_size, vocab_size]

# Step 3: Run logit matching
results = check_accuracy_logits_v2(
    neuron_model=neuron_model,           # The compiled+loaded Neuron model
    expected_logits=expected_logits,      # Golden logits from step 2
    inputs_input_ids=inputs.input_ids,
    inputs_attention_mask=inputs.attention_mask,
    generation_config=generation_config,
    divergence_difference_tol=0.001,     # Optional (default 0.001)
    tol_map=None,                         # Optional: position-specific tolerances
    num_tokens_to_check=None,            # Optional: limit tokens checked
)
# Raises LogitMatchingValidationError if logits don't match within tolerance
```

**Key parameters:**
- `expected_logits`: Shape `[num_tokens, batch_size, vocab_size]`. Can be pre-generated and saved to disk with `torch.save()` to avoid re-running HF on CPU each time.
- `tol_map`: Dict of `{position: {atol: float, rtol: float}}` for per-position tolerance overrides.
- `num_tokens_to_check`: Limits how many output tokens are validated. If `None`, checks all tokens up to `seq_len - input_len`.
- `additional_input_args`: Dict of extra kwargs passed to `model.generate()` (useful for multimodal models).

**Notes:**
- `generate_expected_logits()` calls `neuron_model.load_hf_model(neuron_model.model_path)` internally to instantiate the HF model on CPU. For 671B models this is impractical — generate goldens separately or with a smaller model.
- The older `check_accuracy_logits()` API is deprecated but still works. It combines input preparation, golden generation, and validation into one call.

### Token Matching (`check_accuracy`)

Verifies that output **tokens** (not logits) match between the Neuron model and expected output. Simpler but less informative than logit matching.

**How it works:**
1. Generates tokens with greedy sampling on Neuron.
2. Generates expected tokens with the HF model on CPU (or uses provided `expected_token_ids`).
3. Asserts `torch.equal(actual_token_ids, expected_token_ids)`.

**API signature:**
```python
from neuronx_distributed_inference.utils.accuracy import check_accuracy

check_accuracy(
    neuron_model,                    # The compiled+loaded Neuron model
    tokenizer,                       # HuggingFace tokenizer
    generation_config,               # HuggingFace GenerationConfig
    prompt="Hello, I am a language model",  # Optional (default: built-in TEST_PROMPT)
    expected_token_ids=None,         # Optional: pre-computed expected tokens [batch, seq_len]
    num_tokens_to_check=None,        # Optional: limit number of tokens compared
    do_sample=False,                 # Must be False for deterministic matching
)
# Raises AssertionError if tokens don't match
```

**Key parameters:**
- `expected_token_ids`: If `None`, the function loads the HF model on CPU and generates golden tokens automatically. For large models, pre-generate and pass them in.
- `num_tokens_to_check`: Truncates both actual and expected token sequences before comparison.
- `prompt`: The input prompt. Replicated across `batch_size` sequences.

**Warning from docs:** Token mismatches are acceptable in many scenarios, especially with large models or large sequence lengths. This tool should only be used for small models and small sequence lengths. Logit matching is preferred for larger models.

### Benchmarking (`benchmark_sampling`)

Measures latency and throughput of the Neuron model end-to-end and per sub-model (context encoding, token generation).

**How it works:**
1. Generates random input tensors matching the model's configured dimensions.
2. Runs a warmup iteration.
3. Runs `num_runs` timed iterations, collecting latency for each.
4. Reports p50/p90/p95/p99/p100 latencies and throughput.
5. If `target="all"`, also instruments sub-model latencies (context encoding, token generation) via forward hooks.

**API signature:**
```python
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

report = benchmark_sampling(
    model,                           # The compiled+loaded Neuron model
    generation_config=generation_config,  # HuggingFace GenerationConfig
    target=None,                     # Optional: "all" (default), "e2e", "context_encode", "token_gen"
    num_runs=20,                     # Number of timed iterations
    benchmark_report_path="./benchmark_report.json",  # Where to save JSON report
)
```

**Report structure:**
```json
{
    "e2e_model": {
        "latency_ms_p50": ...,
        "latency_ms_p90": ...,
        "latency_ms_p95": ...,
        "latency_ms_p99": ...,
        "latency_ms_p100": ...,
        "latency_ms_avg": ...,
        "throughput": ...
    },
    "context_encoding_model": { ... },
    "token_generation_model": { ... }
}
```

**Key parameters:**
- `target`: Controls which components are benchmarked. `"all"` runs E2E + instruments sub-models. `"e2e"` runs only E2E. `"context_encode"` and `"token_gen"` run sub-models in isolation.
- `num_runs`: More runs give more stable percentile estimates.
- The function uses `model.neuron_config` to determine input shapes (batch size, sequence length, etc.).
- Throughput = `(num_runs × max_length × batch_size) / total_time`.

### Using the `inference_demo` CLI

NxD Inference provides an `inference_demo` console script that wraps all three evaluation tools. It's available after installing `neuronx_distributed_inference`. Custom models must be added to the `MODEL_TYPES` dict in `inference_demo.py`.

```bash
# Token matching
inference_demo --model-type llama --task-type causal-lm run \
    --model-path /path/to/model --compiled-model-path /path/to/traced \
    --tp-degree 32 --batch-size 1 --seq-len 4096 \
    --prompt "Hello, I am a language model" \
    --check-accuracy-mode token-matching

# Logit matching
inference_demo ... --check-accuracy-mode logit-matching \
    --num-tokens-to-check 64 \
    --expected-outputs-path /path/to/golden_logits.pt

# Benchmarking
inference_demo ... --benchmark

# Skip accuracy check (just generate)
inference_demo ... --check-accuracy-mode skip-accuracy-check
```

**CLI arguments for evaluation:**
- `--check-accuracy-mode`: `token-matching`, `logit-matching`, `draft-logit-matching`, or `skip-accuracy-check`
- `--num-tokens-to-check`: Limit tokens validated
- `--expected-outputs-path`: Path to `torch.save()`'d golden tokens/logits
- `--divergence-difference-tol`: Float tolerance for logit divergence (default `0.001`)
- `--tol-map`: String repr of tolerance dict
- `--benchmark`: Enable benchmarking after generation
- `--benchmark-report-path`: Output path for benchmark JSON

### What the `inference_demo` Script Shows

The `inference_demo.py` script (see [source](https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/inference_demo.py)) orchestrates the full lifecycle: compile → load → accuracy check → generation → benchmark. Here's what each evaluation step outputs and how to interpret it.

#### Logit Matching Output

When `--check-accuracy-mode logit-matching` is used, the script calls `check_accuracy_logits()` → `logit_validation()`. The output has three phases:

**Phase 1: Golden logit generation (HF on CPU)**
```
Expected Output(tokens with the max logits): ['Paris is the capital of France...']
Expected Output (token ids with the max logits): tensor([[3681, 338, ...]])
Expected Logits Shape: torch.Size([128, 1, 129280])
```
This loads the HF model on CPU and runs greedy generation to produce reference logits. Shape is `[num_tokens, batch_size, vocab_size]`.

**Phase 2: Divergence detection and re-generation**
The core loop compares Neuron output tokens to expected tokens. If they match everywhere:
```
No divergence. Validating the remaining 128 tokens in each batch.
```
If the greedy argmax differs at some position (common with BF16 vs FP32):
```
Divergence at index 47. Validating 47 tokens in each batch.
```
When divergence occurs, the validator feeds the *expected* tokens up to the divergence point back into the model and re-generates from there. This repeats until all positions are covered. Multiple divergence messages may appear:
```
Divergence at index 47. Validating 47 tokens in each batch.
Divergence at index 93. Validating 46 tokens in each batch.
No divergence. Validating the remaining 35 tokens in each batch.
```

**Phase 3: Per-token logit validation**
At each token position, the validator checks:

1. **Divergence difference** — When Neuron picks a different top token than expected, how close were the two candidates' logit scores? Must be < `divergence_difference_tol` (default `0.001`). A small divergence difference means the model was "almost right" — the two top tokens had nearly identical scores.
   ```
   Test failed at batch 0 token 47. Divergence difference 0.0023 > 0.001.
   ```

2. **Top-k logit errors** — For each k in `{5, 50, 1000, None}`, compares the top-k logit values using `(|actual - expected| - atol) / max(|expected|) ≤ rtol`. A constant shift between actual and expected logits is removed via least-squares regression before comparison.
   ```
   Test failed at batch 0 token 47. Top k = 5 error 0.0142 > 0.01.
   ```

**Phase 4: Summary**
```
Summary: Max divergence difference = 0.00089 at index (batch 0 token 47),
  Top k = 5 max error = 0.0082 at index (batch 0 token 12),
  Top k = 50 max error = 0.0156 at index (batch 0 token 47),
  Top k = 1000 max error = 0.0234 at index (batch 0 token 47),
  Top k = None max error = 0.0412 at index (batch 0 token 47)

Complete Error Summary
These errors are normalized for each token by the largest expected logit for that token
{
    "max_top_k_errors": {
        "5":    {"error": 0.0082, "batch_index": 0, "token_index": 12},
        "50":   {"error": 0.0156, "batch_index": 0, "token_index": 47},
        "1000": {"error": 0.0234, "batch_index": 0, "token_index": 47},
        "null": {"error": 0.0412, "batch_index": 0, "token_index": 47}
    },
    "average_over_tokens": {
        "null": {"mean_abs_error": 0.00012, "mean_squared_error": 1.8e-08},
        "5":    {"max_abs_error": 0.0082, "max_squared_error": 6.7e-05},
        ...
    }
}

Test passes logit validation.
```

**How to interpret results:**
- **"Test passes logit validation"** — All tokens within tolerance. Model is numerically correct.
- **Divergences are normal** — BF16 rounding means tokens with close logit scores may swap order. What matters is that the divergence difference is small (< 0.001), meaning the "wrong" token had a nearly identical score.
- **Top-k errors increase with k** — Expected. Low-probability tokens have larger relative errors. The tolerances are set progressively looser (`0.01` for k=5, `0.05` for k=None).
- **Shift removal** — The validator removes a constant offset between actual and expected logits before comparison (via least-squares). This accounts for systematic BF16/FP32 differences.
- **Failure at a specific token** — Check `divergence_difference` (is the model picking a very different token?) and `error_map` (are logit values way off?). A failure at a single token with small error often means the tolerance just needs tuning via `--tol-map`.

#### Benchmark Output

When `--benchmark` is used, the script calls `benchmark_sampling()` which runs `num_runs` (default 20) timed E2E generations with random inputs, plus instruments sub-model latencies via forward hooks.

```
Starting end-to-end benchmark with 20
Benchmark completed and its result is as following
{
    "e2e_model": {
        "latency_ms_p50": 28890.24,
        "latency_ms_p90": 28977.73,
        "latency_ms_p95": 28983.17,
        "latency_ms_p99": 29032.21,
        "latency_ms_p100": 29044.47,
        "latency_ms_avg": 28879.50,
        "throughput": 283.66
    },
    "context_encoding_model": {
        "latency_ms_p50": 705.02,
        "latency_ms_p90": 705.37,
        "latency_ms_p95": 705.66,
        "latency_ms_p99": 705.84,
        "latency_ms_p100": 705.89,
        "latency_ms_avg": 705.04,
        "throughput": 5809.62
    },
    "token_generation_model": {
        "latency_ms_p50": 27.20,
        "latency_ms_p90": 27.30,
        "latency_ms_p95": 27.32,
        "latency_ms_p99": 27.66,
        "latency_ms_p100": 32.74,
        "latency_ms_avg": 27.20,
        "throughput": 147.22
    }
}
Completed saving result to ./benchmark_report.json
```

**How to interpret results:**
- **e2e_model** — Full generation latency (prefill + all decode steps + sampling). The throughput is `(num_runs × max_length × batch_size) / total_time` in tokens/sec.
- **context_encoding_model** — Prefill (context encoding) NEFF latency only. This is the TTFT component.
- **token_generation_model** — Single token generation NEFF latency. This is the per-step decode cost, which determines TPOT. In the example above, `27.2ms` per token ≈ `36.8 tok/s` per sequence.
- **p50/p90/p99** — Percentile latencies across the `num_runs` iterations. Small p99/p50 spread indicates stable performance. Large p100 outliers (like `32.7ms` vs `27.2ms` p50 above) are typically warmup artifacts.
- The benchmark uses **random input tokens** of length `max_context_length` (or `max_context_length // 2` if `max_context_length == max_length`), so results reflect worst-case prefill length.

#### Generation Output

Between accuracy check and benchmark, the script always runs a generation pass:
```
Generating outputs...
Prompts: ['Hello, I am a language model']
Generated outputs:
Output 0: Hello, I am a language model, and I am here to help you...
```

#### `inference_demo` End-to-End Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. Create NeuronConfig + InferenceConfig               │
│  2. Instantiate model: model_cls(model_path, config)    │
│  3. Compile: model.compile(compiled_model_path)         │  ← --skip-compile to skip
│  4. Load to Neuron: model.load(compiled_model_path)     │
│  5. Load tokenizer + GenerationConfig                   │
├─────────────────────────────────────────────────────────┤
│  6. Accuracy check (--check-accuracy-mode):             │
│     • token-matching  → check_accuracy()                │
│     • logit-matching  → check_accuracy_logits()         │
│     • skip (default)  → no-op                           │
├─────────────────────────────────────────────────────────┤
│  7. Generation: get_generate_outputs() → decode + print │
├─────────────────────────────────────────────────────────┤
│  8. Benchmark (--benchmark):                            │
│     • benchmark_sampling() → 20 runs → JSON report      │
└─────────────────────────────────────────────────────────┘
```

### DeepSeek V3 Considerations

For DeepSeek V3 (671B), the standard onboarding evaluation tools have specific constraints:

1. **Golden logit generation on CPU is impractical.** `generate_expected_logits()` loads the full HF model on CPU, which requires >1.3TB RAM for BF16 weights. Instead:
   - Generate goldens on a separate GPU instance and save with `torch.save(expected_logits, "golden_logits.pt")`.
   - Or use our existing `test_logit_matching.py` / `test_logit_matching_full.py` scripts, which use a mini model for validation.
   - Or compare against a known-good output from a previous run.

2. **Token matching is not recommended** for 671B. Small BF16 rounding differences in MoE routing can cause token divergence even when logits are close. Logit matching is the right approach.

3. **`benchmark_sampling` works as-is** once the model is compiled and loaded. It generates random inputs internally and doesn't need HF weights.

4. **`inference_demo` is now patched** by `install_deepseek.sh` to include `"deepseek_v3"` in its `MODEL_TYPES` dict. Usage:
   ```bash
   inference_demo --model-type deepseek_v3 --task-type causal-lm run \
       --model-path ~/environment/models/DeepSeek-V3-0324-FP8 \
       --compiled-model-path /scratch/deepseek_v3_traced \
       --tp-degree 64 --batch-size 1 --seq-len 512 \
       --prompt "The capital of France is" \
       --check-accuracy-mode logit-matching --benchmark
   ```
   Note: logit matching with `--check-accuracy-mode logit-matching` will attempt to load the full HF model on CPU for golden generation, which requires >1.3TB RAM for the 671B model (see point 1 above).

5. **DeepSeek V3 does NOT use `NeuronAttentionBase`** (MLA is incompatible with GQA), so module-level tests from `orchestrator_base.py` don't apply to the attention module.

---

## Technical Reference

### HF State Dict Key Names

```
# Attention (all 61 layers)
model.layers.{i}.self_attn.q_a_proj.weight           # [1536, 7168]
model.layers.{i}.self_attn.q_a_layernorm.weight       # [1536]
model.layers.{i}.self_attn.q_b_proj.weight            # [128*192, 1536]
model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight  # [576, 7168]
model.layers.{i}.self_attn.kv_a_layernorm.weight      # [512]
model.layers.{i}.self_attn.kv_b_proj.weight           # [128*256, 512]
model.layers.{i}.self_attn.o_proj.weight              # [7168, 128*128]

# Dense MLP (layers 0-2)
model.layers.{i}.mlp.gate_proj.weight                 # [18432, 7168]
model.layers.{i}.mlp.up_proj.weight                   # [18432, 7168]
model.layers.{i}.mlp.down_proj.weight                 # [7168, 18432]

# MoE (layers 3-60)
model.layers.{i}.mlp.gate.weight                      # [256, 7168] - router
model.layers.{i}.mlp.gate.e_score_correction_bias     # [256] - bias (buffer)
model.layers.{i}.mlp.experts.{e}.gate_proj.weight     # [2048, 7168]
model.layers.{i}.mlp.experts.{e}.up_proj.weight       # [2048, 7168]
model.layers.{i}.mlp.experts.{e}.down_proj.weight     # [7168, 2048]
model.layers.{i}.mlp.shared_experts.gate_proj.weight  # [2048, 7168]
model.layers.{i}.mlp.shared_experts.up_proj.weight    # [2048, 7168]
model.layers.{i}.mlp.shared_experts.down_proj.weight  # [7168, 2048]
```

### State Dict Conversion (HF → Neuron)

Key transformations in `convert_deepseek_v3_hf_to_neuron_state_dict()`:
1. `gate.weight` → `router.linear_router.weight`
2. `gate.e_score_correction_bias` → `router.e_score_correction_bias`
3. Per-expert `gate_proj` + `up_proj` → fused `gate_up_proj` [num_experts, 2*intermediate, hidden]
4. Per-expert `down_proj` → stacked `down_proj` [num_experts, hidden, intermediate]
5. Dense layers (0-2) pass through unchanged
6. `rank_util.rank` tensor added for TP

### vLLM Integration Notes

- vLLM parses `DeepseekV3ForCausalLM` → model type `"deepseekv3"` (no underscore, split on `For`, lowercase)
- NXDI registry uses `"deepseek_v3"` (with underscore)
- `install_deepseek_vllm.sh` registers BOTH keys
- Must use `VLLM_PLUGINS=neuron` if `optimum_neuron` is also installed in the venv
- `VLLM_MLA_DISABLE` is NOT needed (no MLA detection in vllm_neuron 0.13)

### Known Issues

- **DGE OOB with small models**: Mini model (vocab_size=32000, small dims) triggers DGE OOB on 2nd+ request. This is a Neuron compiler bug, not a model code issue. Expected to work with full 671B model.
- **vocab_size < ~20000**: Triggers DGE OOB. Use vocab_size >= 32000 for mini models.
- **BF16 vs FP32 divergence**: With random weights, autoregressive generation diverges after first token. This is expected due to small logit margins. Real trained weights have larger margins and should match better.

### Environment Details

- NXDI venv: `/opt/aws_neuronx_venv_pytorch_*_nxd_inference/` (comes with AMI)
- vLLM venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_*/` (comes with AMI)
- The install scripts copy model code INTO these venvs (they don't do editable installs)
- Unit tests run from the repo directory, not from the venv
