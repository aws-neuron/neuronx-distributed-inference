# Contrib Model: GLM-5.2

NeuronX Distributed Inference (NXDI) implementation of **GLM-5.2**
(`model_type="glm_moe_dsa"`), a ~753B-parameter Mixture-of-Experts LLM from
Z.ai / ZhipuAI. Runs the **FP8 checkpoint** (`zai-org/GLM-5.2-FP8`) on a single
trn2.48xlarge.

GLM-5.2 is architecturally DeepSeek-V3 (Multi-head Latent Attention + sigmoid-routed
MoE) plus a DeepSeek Sparse Attention (DSA) indexer.

## Status — runs end-to-end on trn2.48xlarge

**Validated:** 2026-06-17 · NXDI 0.9.0, neuronx-cc 2.24, NKI 0.3.0, torch 2.9, trn2.48xlarge

Measured, BS=1, TP=64, lnc=2, seq_len=2048, FP8 experts + BF16 attention/dense,
fused MoE TKG + MLP NKI kernels:

| Metric | Value |
|--------|-------|
| **TTFT** (2048-token prefill) | **6,377 ms** |
| **ITL / TPOT** (per-token) | **241.5 ms/tok** |
| **Throughput** (BS=1) | **2.96 tok/s** |
| Weight load (pre-sharded, 64 ranks) | 139 s |
| Warmup | 16 s |

Output is coherent and factually correct (e.g. *"The capital of France is Paris...
the Ile-de-France... where the French language was born"*). FP8 quantization
accuracy is fine.

> An MLA-attention NKI kernel variant (`mla_attention_nki_kernel_enabled=True`)
> is included for further decode speedup; see *Optimization*.

## Model Information

- **HuggingFace ID:** `zai-org/GLM-5.2-FP8` (FP8 checkpoint, 141 safetensors, ~705 GB)
- **Architecture:** `GlmMoeDsaForCausalLM` (`model_type="glm_moe_dsa"`)
- **Total params:** ~753B (~40B active/token)
- **License:** Check the HuggingFace model card

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 78 (3 dense + 75 MoE) + 1 MTP layer (dropped) |
| Hidden size | 6144 |
| Attention | MLA (LoRA-compressed Q + KV), 64 heads |
| q_lora_rank / kv_lora_rank | 2048 / 512 |
| qk_nope / qk_rope / v_head | 192 / 64 / 256 |
| Routed experts | 256 (n_group=1, top-8) + 1 shared |
| Routing | sigmoid + e_score_correction_bias (noaux_tc), L1 norm, ×2.5 |
| Dense / MoE intermediate | 12288 / 2048 |
| RoPE | interleaved, rope_theta=8e6 (no YaRN) |
| Vocab | 154880 |
| Quantization | FP8 (e4m3) experts; layernorms/router/attn/embed/lm_head in BF16 |

## Key Decisions for GLM-5.2

- **FP8 is required.** BF16 weights (~1506 GB) do not fit the trn2's 1536 GB HBM:
  the token-generation graph needs ~26 GB per core vs the 24 GB per-core bank
  (verified by a failed BF16 compile). The FP8 checkpoint (~705 GB) leaves room.
- **DSA indexer disabled (`dsa_enabled=False`).** GLM-5.2 has *heterogeneous*
  per-layer indexers (`indexer_types`: a per-layer mix of `full`/`shared`); some
  layers (e.g. layer 3) carry no indexer weights, which breaks a per-layer-uniform
  indexer assumption at load time (`Missing ...indexer.k_norm.weight`). For prompts
  of length ≤ `index_topk` (2048), the indexer is a mathematical no-op (top-2048
  over ≤2048 keys = full attention), so we run **standard full MLA** (KV cache dim
  576). Long-context (> 2048) sparse attention is future work.
- **MTP layer (index 78) dropped** — it is a speculative-decoding draft head, not
  needed for standard autoregressive generation.
- **Fused MoE TKG kernel** (`moe_fused_nki_kernel_enabled=True`) runs the FP8
  experts. This requires the nkilib routing fork — see *Prerequisites*.

## Prerequisites

### nkilib routing fork

GLM-5.2's sigmoid routing needs `selection_bias` (e_score_correction_bias) and
`routed_scaling_factor` support in the NKI fused-MoE router. Install the fork
(adds these as optional kernel params; stock nkilib auto-swaps to it via its
`sys.modules` mechanism — reversible with `pip uninstall nki_library`):

```bash
git clone https://github.com/jimburtoft/nki-library.git
cd nki-library
git checkout feature/selection-bias-routing
pip install -e . --no-deps
```

### Environment

```bash
export UNSAFE_FP8FNCAST=1   # Neuron FP8 cast (weights clamped to 240; Neuron treats exp-15 as NaN)
```

SDK 2.29-style installs may also need the `os.makedirs` / `shutil.rmtree`
monkey-patches shown in `examples/compile.py` (race-condition workarounds).

## Usage

Single-process SPMD (NOT torchrun) — one process drives all 64 NeuronCores.

```bash
# 1. Compile + pre-shard weights (~1.5h: HLO + NEFF + per-layer FP8 expert conversion + 64-rank shard)
MODEL_PATH=~/GLM-5.2-FP8 COMPILED_MODEL_PATH=~/glm52_compiled \
    python examples/compile.py

# 2. Load + benchmark (TTFT / ITL / throughput)
MODEL_PATH=~/GLM-5.2-FP8 COMPILED_MODEL_PATH=~/glm52_compiled \
    python examples/benchmark.py
```

The key `MoENeuronConfig` (see `examples/compile.py`): `tp_degree=64`,
`logical_nc_config=2`, `seq_len=2048`, `quantized=True`,
`quantization_dtype="f8e4m3"`, `moe_fused_nki_kernel_enabled=True`,
`mlp_kernel_enabled=True`, `save_sharded_checkpoint=True`,
`modules_to_not_convert=[lm_head, self_attn, shared_expert, layers.0/1/2.mlp]`,
and `config.dsa_enabled = False`.

## Optimization

`examples/compile.py` sets the proven config (fused MoE TKG + MLP NKI kernels).
The fused MoE TKG + MLP kernels are the dominant lever — reference data on the
sibling GLM-5:

| Config | tok/s | Latency |
|--------|-------|---------|
| Compiler only (no NKI) | ~1.6 | ~625 ms |
| + Fused MoE TKG kernel | ~2.1 | ~473 ms |
| + MLP kernel (this README's default) | 2.96 | 241 ms |

**MLA attention NKI kernel — measured, no benefit at BS=1.** Enabling
`mla_attention_nki_kernel_enabled=True` (the fused MLA decode kernel in
`src/mla_attention_nki*.py`; compatible with `dsa_enabled=False`) was measured at
**ITL 241.6 ms vs 241.5 ms** — i.e. no change. At BS=1 decode is **bound by MoE
expert memory bandwidth** (reading the active expert weights from HBM each token),
not by attention compute, so the MLA kernel does not move the needle. It may help
at larger batch sizes or longer context. Real latency wins at BS=1 would come from
reducing the MoE HBM traffic (e.g. expert quantization/packing, larger batch to
amortize weight reads), not from attention kernels.

## Compatibility Matrix

| Instance | TP | LNC | NXDI | neuronx-cc | NKI | Status |
|----------|----|-----|------|-----------|-----|--------|
| trn2.48xlarge | 64 | 2 | 0.9.0 | 2.24 | 0.3.0 | **PASS** (compile + load + generate) |

### Minimum resources

| Resource | Requirement |
|----------|------------|
| HBM | 1.5 TB (64 NeuronCores × 24 GB, lnc=2) |
| System RAM | ~1 TB peak (FP8 expert conversion + 64-rank shard) |
| NVMe | ~705 GB (FP8 checkpoint) + ~886 GB (compiled + sharded) |

## Testing

```bash
# Unit tests (CPU, no device)
cd contrib/models/GLM-5.2/
pytest test/unit/ -v

# Integration (trn2.48xlarge, requires nkilib fork + compiled model)
MODEL_PATH=~/GLM-5.2-FP8 COMPILED_MODEL_PATH=~/glm52_compiled \
    pytest test/integration/test_model.py -v

# MLA attention NKI kernel validation (standalone; --cpu-only for reference path)
python src/test_mla_attention_nki.py --cpu-only
```

## Provenance

The modeling code (`src/modeling_glm5.py`, `src/mla_attention_nki*.py`) is adapted
from the community GLM-5 contribution by **jimburtoft**
(`github.com/jimburtoft/neuronx-distributed-inference`, branch `contrib/GLM-5`,
for `zai-org/GLM-5-FP8`). GLM-5 and GLM-5.2 are the same `glm_moe_dsa`
architecture; the config is read directly from the checkpoint, so the only
GLM-5.2-specific change is `dsa_enabled=False` (see *Key Decisions*). The
rope_theta (8e6, nested in `rope_parameters`) is picked up automatically.

## Maintainer

AWS Neuron — contrib · **Last Updated:** 2026-06-17
