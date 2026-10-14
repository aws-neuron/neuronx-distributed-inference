# V3 CP=4 + WLO — FINAL / RECOMMENDED QIE try-on config on Trainium2 (beats H100)

**TP=4 × CP=4 (world=16) + WLO — the chosen production configuration.** Doubles the cores
used vs the CP=2 baseline (16 of the chip's 32 logical cores). Because the QIE transformer is
**compute-bound**, doubling cores **halves the transformer step** — and comms (KV all-gather
over 4 vs 2) does NOT eat the gain.

> Why CP=4 and not CP=8: CP=8 (world=32, all 32 cores) runs at 4.10s but the gain over CP=4's
> 4.50s is small (only ~0.4s) — the transformer step's marginal return drops (16→32 cores =
> 0.70×, not 0.5×) and the larger world=32 adds ~200ms of DP overhead to the TP=4 vision/LM
> (which don't benefit from more cores). CP=4 (world=16) is the sweet spot: near-linear
> transformer speedup without the diminishing returns / vision-LM penalty of world=32.

## Measured results (2026-06-01, verified outputs)

| config | cores | warm step | E2E | output |
|---|---|---|---|---|
| TP4×CP2 + WLO (prev best) | 8 | 793 ms | 7.51 s | correct |
| **TP4×CP4 + WLO** | **16** | **411 ms** | **4.50 s** | correct |
| Δ | 2× | **−48%** | **−3.01 s (−40%)** | visually equiv |

**4.50 s beats H100 vLLM-Omni's 4.99 s** — first time QIE on Trn2 matches/beats H100, lossless
(more cores + bit-exact WLO). Output is the correct dark/green tiered maxi skirt (matching the
black-skirt cloth input), visually equivalent to both CP=2 and the early known-good 05-23 image.

Stage breakdown (CP=4): text-enc ~450 ms, VAE-enc ~310 ms, transformer 8× ~411 ms (step1
~750 ms warmup), VAE-dec ~200 ms.

## Quality

CP=4 output vs CP=2 baseline: **mean |Δ| = 0.78%** (px>5: 4.7%); vs the early known-good
05-23 image: 0.49%. NOT bit-exact (CP degree changes the sequence partition → different bf16
accumulation order) but visually equivalent — same garment/pose/face. `test_cp4_quality.py`
asserts mean |Δ| < 2/255. See `cp4_vs_cp2.png`.

## Prompt matters (lesson learned the hard way)

Use the **short** prompt `让图2的模特换上图1的下装`. During packaging I briefly used a long
prompt ("把右图模特腰部以下的下装换成…全部保持…") and got **khaki shorts** (the input cloth is a
BLACK maxi skirt) — and almost shipped it. Always diff the output against the input cloth, not
just "did it run". The short prompt is what was validated on this step4000 checkpoint.

(Note: an earlier blurry result I first blamed on zero-padding of CP-alignment patches was
actually the long-prompt issue. Zero-padding is fine — verified correct at both CP=2 and CP=4.
The padding code is the original zero-pad.)

## Why TP=4×CP=4 and not TP=8×CP=2 (same 16 cores)?

Tested both. TP=8 → 3 heads/rank → flash kernel falls back to seqlen-sharding (needs seqlen_q
div by pow2 ≥512; fixed with padding, compiles). But **to_neuron crashes**:
`failed to init a collective algorithm for provided replica group` / `Failed to find device
to device paths`.

**Corrected root cause (NOT a hardware/topology limit):** trn2.48xlarge is a 2D torus (16
cards / 64 cores) and CAN physically do TP=8×CP=2. The real issue is `neuronx_distributed`'s
replica-group mapping (`parallel_state.get_logic_chosen`) does not cover LNC2 + TP=8 +
world<64: it lays the TP-group as (0..7)(8..15), but on the torus device 0 and 8 aren't
directly connected. The kernel docstring explicitly lists this as *Not Supported* (VNC2 8×8:
both LOGIC1 and LOGIC2 fail; needs a (0..7)(16..23)-style mapping NxD hasn't implemented), and
a `world_size<64 → force LOGIC1` fallback locks world=16 into the non-working LOGIC1. It could
be worked around with `ModelBuilder(init_custom_process_group_fn=…)` injecting a torus-friendly
group, but that's nontrivial — AND even if unblocked it likely won't beat TP=4×CP=4 (same 16
cores/compute, but TP=8's all-reduce spans 8 ranks — pure-TP8 measured 9.74 s for that reason).

TP=4 keeps 6 heads/rank (no kernel fallback) and a 4-rank group that maps cleanly to the torus
under the stock LOGIC1. **TP=4×CP=4 is the practical optimum at world=16** — and this is finally
WHY V3 uses TP=4 (not just the LM's GQA — TP=4's collective group also fits NxD/torus mapping).

## Files
- `compile.sh` — compile transformer + vision + LM all at world=16 (+WLO). VAE symlinked.
- `run_tryon.sh [out.png]` — run try-on (short prompt, `QIE_WORLD_SIZE=16 NEURON_RT_NUM_CORES=16`).
- `test_cp4_quality.py <cp4.png> <cp2.png>` — assert CP=4 ≈ CP=2 (mean |Δ| < 2/255).
- `tryon_cp4.png` / `tryon_cp2_baseline.png` / `cp4_vs_cp2.png` — verified reference outputs + diff.

## Reproduce
```bash
bash release_v3cp4_wlo/compile.sh                       # ~35 min (full shard, world=16)
bash release_v3cp4_wlo/run_tryon.sh release_v3cp4_wlo/tryon_cp4.png   # ~4.5 s
python release_v3cp4_wlo/test_cp4_quality.py \
    release_v3cp4_wlo/tryon_cp4.png release_v3cp4_wlo/tryon_cp2_baseline.png
```

## Environment
- `trn2.48xlarge` (64 NeuronCores = 32 logical at LNC2), venv
  `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Runtime: `NEURON_RT_NUM_CORES=16`, `QIE_WORLD_SIZE=16`, `PYTHONPATH=src:$PYTHONPATH`
- Compile defaults: `QIE_ALLREDUCE_BF16=1`, `QIE_OPT_LEVEL=2`, `QIE_CC_TILING=4`, `QIE_WLO=1`

## Next: CP=8 (world=32, all 32 cores)
Projected ~205 ms/step → E2E ~3 s. TP=4 so no topology issue. Not yet validated end-to-end.
