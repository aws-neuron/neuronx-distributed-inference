# Qwen-Image-Edit on Trainium2 — optimized (TP4×CP4 + WLO)

NeuronX adaptation of `Qwen/Qwen-Image-Edit-2509` for AWS Trainium2 inference, with the
latency optimizations applied. **Production config: TP=4 × CP=4 (world=16) + WLO**, which
runs the 896×1184 / 8-step / CFG=1 virtual try-on at **~4.5 s end-to-end** — faster than the
H100 vLLM-Omni reference (4.99 s), lossless.

> **Scope / applicability.** This optimization round targets a **few-step
> distillation–finetuned checkpoint run with classifier-free guidance disabled
> (CFG=1)** — i.e. ~8 denoising steps and a *single* (positive-only) transformer forward
> per step, no negative-prompt pass. Two consequences shape every number below:
> - With CFG=1 there is no negative branch to batch, so the **CFG-parallel (DP=2) path
>   buys nothing** — the right lever is **Context Parallel (CP)**, which shards the single
>   forward across more cores. (For CFG>1, V3 CFG's DP=2 batching of neg+pos is the better
>   layout; CP scaling here is specifically for the CFG=1 few-step regime.)
> - At only ~8 steps the fixed per-run cost (text encoder + VAE encode/decode, ~1.1 s) is a
>   large fraction of E2E, so VAE/text-encoder latency matters far more than it would for a
>   50-step run — which is why the VAE batched-tile win below is material here.
>
> The latency targets (matching/beating H100's 4.99 s) and the CP=4 sweet-spot choice are
> all stated **for this few-step / CFG=1 workload**; a many-step or CFG>1 run has a
> different optimum.

## What's here

```
src/                  full NeuronX model/compile/run source (the optimizations live here)
release_v3cp4_wlo/    the production config: compile.sh / run_tryon.sh / test / README + sample outputs
requirements.txt
```

Input try-on images (cloth/, input_img/) are NOT included — supply your own and pass via the
run script flags (see release_v3cp4_wlo/run_tryon.sh).

## Quick start (production config)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd <repo>/contrib/models/Qwen-Image-Edit

# 1) compile transformer + vision + LM at world=16 (+WLO); VAE symlinked. ~35 min.
bash release_v3cp4_wlo/compile.sh

# 2) run try-on (~4.5 s). Edit the cloth/model paths in the script or pass via QIE_CLOTH / QIE_MODEL_IMG.
bash release_v3cp4_wlo/run_tryon.sh out.png
```

See `release_v3cp4_wlo/README.md` for the full recipe, the CP scaling rationale (why CP=4),
and the quality test.

## The optimizations (vs the original ~7.9 s baseline)

| optimization | effect | how |
|---|---|---|
| **VAE batched-tile** | VAE encode 559 → 298 ms (−47%), decode 422 → 192 ms (−55%), **−6.1% E2E**; numerically identical | compile the tiled VAE encoder/decoder at `batch=N` (N = tiles/image — 6 for 1024², 12 for two-image edit) and run all tiles in **one** NEFF launch instead of N sequential launches. `_tiled_encode`/`_tiled_decode` enumerate tiles → run in chunks of the compiled batch → scatter+crop+blend back into the grid. Collapses ~37 ms/tile launch overhead; matters at few-step because VAE is a fixed cost. Set via `VAE_BATCH_SIZE` in `compile.sh`. **Both `_tiled_encode` and `_tiled_decode` must be updated** (half-applied regresses the decoder). |
| **WLO** (weight layout opt) | −3.2% step, **bit-exact** | pass `priority_model_key="inference"` to `ModelBuilder.compile()` — was never enabled. `QIE_WLO=1` (default). |
| **CP scaling** (CP=2 → CP=4) | **7.51 → 4.50 s** (−40%), step 793 → 411 ms, lossless | QIE transformer is compute-bound but V3 only used 8 of the chip's 32 logical cores; doubling cores (world=8 → 16) halves the step. Only a lever because CFG=1 runs a single forward (no DP=2 batching to exploit). Compile with `--tp_degree 4 --world_size 16`; run with `QIE_WORLD_SIZE=16 NEURON_RT_NUM_CORES=16`. |

Output is visually equivalent across CP degrees (CP=4 vs CP=2 mean |Δ| 0.78%); not bit-exact
because the CP partition changes bf16 accumulation order. VAE batched-tile and WLO are
numerically clean (identical / bit-exact); CP scaling is lossless in the visual sense.

### Per-optimization latency reduction

Each delta is measured against the *then-current* baseline (they are applied sequentially), so
this is a cumulative path, **not** a simple sum. Workload: 896×1184 / 8-step / **CFG=1**
two-image virtual try-on on `trn2.48xlarge`.

| step | optimization | E2E | what moved |
|---|---|---|---|
| 0 | baseline (V3 CP=2, fp32 reduce, per-tile VAE) | ~7.89 s | — |
| 1 | + bf16 TP all-reduce | ~7.5 s | TP all-reduce bytes halved (~−9% step) |
| 2 | + VAE batched-tile | 7.41 s | VAE enc 559→298 ms, dec 422→192 ms (−6.1% E2E) |
| 3 | + WLO | ~7.3 s | weight layout for inference (−3.2% step, bit-exact) |
| 4 | + CP scaling (CP=2 → CP=4) | **4.50 s** | transformer step 793 → 411 ms (2× cores) |

> bf16 TP all-reduce and VAE batched-tile predate this round (part of the base contribution);
> the **new** work here is **WLO** and **CP scaling**, which take the verified-correct config
> from 7.51 s to **4.50 s**. After CP scaling the transformer is ~2.3 s of the 4.5 s, so the
> fixed text-encoder + VAE (~1.1 s) is now the dominant remaining cost — which is exactly why
> the VAE batched-tile win is load-bearing in this few-step regime.

### Why TP=4 × CP=4 (not CP=8 or TP=8)
- **CP=8** (32 cores) reaches 4.10 s but the gain over CP=4 is only ~0.4 s — transformer
  marginal return drops to 0.70× (from 0.52×) and the larger world=32 adds ~200 ms of DP
  overhead to the TP=4 vision/LM. CP=4 is the sweet spot.
- **TP=8** triggers a flash-kernel seqlen-shard fallback and the 8-rank TP group has no
  replica-group mapping in NeuronX for world<64. TP=4 (6 heads/rank) maps cleanly to the torus.
- CP must be a power of 2 (world ∈ {8,16,32}); the runtime rejects other core counts (e.g.
  world=12 → "Unsupported topology").

## Key env vars (compile / run)

- `QIE_WLO=1` (default) — weight layout optimization (bit-exact speedup).
- `QIE_WORLD_SIZE=16`, `NEURON_RT_NUM_CORES=16` — required at run time for CP=4.
- `QIE_ALLREDUCE_BF16=1` (default), `QIE_OPT_LEVEL=2`, `QIE_CC_TILING=4` — existing tuned defaults.
- Use the SHORT prompt `让图2的模特换上图1的下装` for try-on (the long prompt mis-edits on this ckpt).

## Environment

- Instance `trn2.48xlarge` (64 NeuronCores = 32 logical at LNC2)
- Venv `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` (PyTorch 2.9, neuronx-cc 2.22, neuronx-distributed 0.16)
- `PYTHONPATH=src:$PYTHONPATH` for both compile and run
