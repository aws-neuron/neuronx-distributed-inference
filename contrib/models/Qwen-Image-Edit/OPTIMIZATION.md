# Qwen-Image-Edit on Trainium2 — optimized (TP4×CP4 + WLO)

NeuronX adaptation of `Qwen/Qwen-Image-Edit-2509` for AWS Trainium2 inference, with the
latency optimizations applied. **Production config: TP=4 × CP=4 (world=16) + WLO**, which
runs the 896×1184 / 8-step / CFG=1 virtual try-on at **~4.5 s end-to-end** — faster than the
H100 vLLM-Omni reference (4.99 s), lossless.

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

## The two optimizations (vs the original 7.6 s baseline)

| optimization | effect | how |
|---|---|---|
| **WLO** (weight layout opt) | −3.2% step, **bit-exact** | pass `priority_model_key="inference"` to `ModelBuilder.compile()` — was never enabled. `QIE_WLO=1` (default). |
| **CP scaling** (CP=2 → CP=4) | **7.51 → 4.50 s** (−40%), step 793 → 411 ms, lossless | QIE transformer is compute-bound but V3 only used 8 of the chip's 32 logical cores; doubling cores (world=8 → 16) halves the step. Compile with `--tp_degree 4 --world_size 16`; run with `QIE_WORLD_SIZE=16 NEURON_RT_NUM_CORES=16`. |

Output is visually equivalent across CP degrees (CP=4 vs CP=2 mean |Δ| 0.78%); not bit-exact
because the CP partition changes bf16 accumulation order.

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
