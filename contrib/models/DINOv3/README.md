# DINOv3 on AWS Neuron

Compile and run [Meta DINOv3](https://github.com/facebookresearch/dinov3) self-supervised vision foundation models on AWS Neuron (Trainium2 and Inferentia2).

## Model

| Property | Value |
|----------|-------|
| **Models** | DINOv3 ViT-S/B/L/H+ (21M-840M), ConvNeXt-T/B (28M-88M), ViT-7B (6.7B) |
| **Architecture** | Encoder-only vision transformer / ConvNeXt backbone |
| **Parameters** | 21M to 6.7B (FP32 by default) |
| **Input** | 224x224 RGB images |
| **Output** | Dense feature embeddings (CLS token for ViT, pooled features for ConvNeXt) |
| **Source** | https://github.com/facebookresearch/dinov3 |
| **License** | DINOv3 License (not Apache/MIT -- check distribution terms) |

## Compilation Approaches

DINOv3 models are compiled using two different approaches depending on model size:

| Model | Params | Approach | Instance |
|-------|--------|----------|----------|
| ViT-S/16 | 21.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-B/16 | 85.7M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-L/16 | 303.2M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-H+/16 | 840.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-7B/16 | 6,716M | `neuronx-distributed` ModelBuilder TP>=2 | inf2.24xlarge (TP>=2) / trn2.3xlarge (TP=4) |
| ConvNeXt-T | 27.8M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ConvNeXt-B | 87.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |

**Key insight**: ViT-7B is the first encoder-only vision model to use tensor parallelism on Neuron. The BF16 weights do not fit in a single NeuronCore's HBM, so TP via `neuronx-distributed` is required. TP=2 is the minimum on Inferentia2; TP=4 is used on trn2.3xlarge.

### Maximum model size on Inferentia2

**All 7 DINOv3 variants -- including the 6.7B ViT-7B -- run on Inferentia2, and all of them fit on a single inf2.xlarge** (1 device, 2 cores). Measured on inf2.xlarge and inf2.24xlarge, SDK 2.31.

Inferentia2 HBM: **32 GB per device, 16 GB per NeuronCore** (2 cores per device).

| Model | Params | Min config on inf2 | Peak core HBM | Status |
|-------|-------:|--------------------|--------------:|--------|
| ViT-S/16 | 21.6M | 1 core | < 0.1 GB | Fits easily |
| ViT-B/16 | 85.7M | 1 core | ~0.2 GB | Fits easily |
| ViT-L/16 | 303.2M | 1 core | ~0.5 GB | Fits easily |
| ViT-H+/16 | 840.6M | 1 core | ~1.4 GB | Fits easily |
| **ViT-7B/16** | **6,716M** | **TP=2 (1 device)** | **14.72 GB of 16 GB** | **Fits, TP>=2 required** |
| ViT-7B/16 | 6,716M | TP=1 | **15.95 GB -> OOM** | Does not fit |

Because TP=2 fits within a single Inferentia2 device, **inf2.xlarge is sufficient for every DINOv3 model** -- verified at 22.1 img/s for ViT-7B on inf2.xlarge (TP=2, BF16, BS=8; 15.1 img/s at the BS=1 default).


**The binding constraint is per-core HBM, and ViT-7B at TP=2 is right at the edge**: 14.72 GB of the 16 GB core (92%), of which 14.51 GB is tensors. At TP=1 the runtime reaches 15.95 GB and then fails to allocate the next 64 MB block (`TDRV:dmem_alloc_internal Failed to allocate DEVICE memory`).

**Practical ceiling**: ViT-7B at TP=2 consumes 3,360M params/rank. Since a BF16 param costs 2 bytes, one 16 GB core holds roughly **7.2B params/rank** before activations. This means:
- **~7B is the practical max at TP=2** (1 Inferentia2 device)
- **~14B at TP=4**, **~28B at TP=8** -- if a larger DINOv3-family backbone existed
- **TP must be a power of 2.** TP=12 fails with `4096 is not divisible by 12` (embed_dim 4096).

ViT-7B is the largest DINOv3 variant Meta publishes, so **inf2 covers the entire DINOv3 model family**. Headroom exists for larger models at higher TP.

## Results

### Accuracy

#### With real pretrained weights (LVD-1689M)

Validated against the **real `facebook/dinov3-*-pretrain-lvd1689m` checkpoints**, not random
init. 64 samples, embeddings compared per-sample against a CPU FP32 reference:

| Model | dtype | cos mean | cos min | rel L2 |
|-------|-------|---------:|--------:|-------:|
| ViT-S/16 | FP32 + matmult | 0.999955 | 0.999896 | 9.4e-03 |
| ViT-S/16 | BF16 | 0.999826 | 0.999699 | 1.9e-02 |
| ViT-B/16 | FP32 + matmult | 0.999945 | 0.999861 | 1.0e-02 |
| ViT-B/16 | BF16 | 0.999843 | 0.999724 | 1.8e-02 |
| ViT-L/16 | FP32 + matmult | 0.999974 | 0.999938 | 7.0e-03 |
| ViT-L/16 | BF16 | 0.999801 | 0.999591 | 2.0e-02 |

**Real weights behave the same as random weights.** BF16 roughly doubles relative L2 error
(1e-02 -> 2e-02) but cosine similarity stays above 0.9998 and the worst single sample is
0.99959. Retrieval geometry is preserved: pairwise-similarity correlation **0.9998**, and the
reference's true nearest neighbour is always within the compiled model's **top 3**.

> **⚠️ `torch_neuronx.trace()` MISCOMPILES the HuggingFace `transformers` DINOv3 port.**
> The checkpoints ship in `transformers` format, but tracing `AutoModel` directly gives
> **cos_sim 0.50** with the real ViT-S weights -- even with `--auto-cast=none`.
>
> **Root cause: `split()` with an explicit list of sizes, lowered with functionalization off.**
> HF's `apply_rotary_pos_emb` separates the prefix (CLS + register) tokens from the patch tokens
> with `q.split((num_prefix_tokens, num_patches), dim=-2)`. `torch_neuronx` sets
> `XLA_DISABLE_FUNCTIONALIZATION=1` by default, and with that setting torch_xla 2.9 lowers
> `split_with_sizes` to a reshape -> 1-D slice -> reshape with the wrong slice offsets. The
> wrong graph is handed to `neuronx-cc`, so the compiler is not the cause. The same thing
> happens on the XLA CPU backend with no Neuron device at all.
>
> | Test (`--auto-cast=none`, trn2 and inf2, neuronx-cc 2.26 and 2.27) | cos_sim vs CPU |
> |---|---:|
> | split + rotate + cat alone, **no model**, `split((5, 196), dim=-2)` | **0.167** |
> | same op, index slicing `q[:, :, :n, :]` | **1.000001** |
> | full HF model, real weights, stock | **0.502** |
> | full HF model, real weights, `apply_rotary_pos_emb` patched to index slicing | **1.000000** |
> | full HF model, real weights, stock, **`XLA_DISABLE_FUNCTIONALIZATION=0`** | **1.000000** |
>
> It is not specific to dim -2: `split((16, 48), dim=-1)` gives cos 0.009. Integer `split(n)`,
> `chunk`, `narrow`, `torch.tensor_split` and index slicing are all correct. The `dinov3`
> reference repo already uses index slicing in `Attention.apply_rope`, which is why the
> architecture this contrib traces is unaffected. **Any traced model that calls `split()` with
> an explicit size list on an inner dimension is likely affected**, silently.
>
> **Workarounds:**
>
> 1. Use the reference-repo architecture and remap the published checkpoints into it.
>    `benchmark/hf_to_repo_weights.py` does this and self-verifies on CPU (`remap_cos` must be
>    ~1.0 before any accuracy number is meaningful).
> 2. Keep the HF model and monkey-patch
>    `transformers.models.dinov3_vit.modeling_dinov3_vit.apply_rotary_pos_emb` with an
>    index-slicing version before tracing. Reproducer: `benchmark/repro_hf_rootcause.py`.
> 3. Set `XLA_DISABLE_FUNCTIONALIZATION=0` in the environment **before** `import torch_neuronx`
>    (torch_neuronx keeps a value you set). On the HF ViT-S model this gives exact output and
>    was also 5.2x faster at BS=1 on one core (355.5 vs 67.9 img/s). Only this model was
>    tested with that setting, so check your own model's output before relying on it.

#### Architecture check with random weights

Confirms the compiled graph computes the same function, independent of weight values:

| Model | Cosine Similarity | Max Abs Diff |
|-------|------------------:|-------------:|
| ViT-S/16 | 1.000000 | 3.6e-06 |
| ViT-B/16 | 1.000000 | < 1e-05 |
| ViT-L/16 | 1.000000 | < 1e-05 |
| ViT-H+/16 | 1.000000 | < 1e-05 |
| ViT-7B/16 | Deterministic (random weights) | -- |
| ConvNeXt-T | 0.999988 | < 0.001 |
| ConvNeXt-B | 0.999989 | < 0.001 |

#### Note on retrieval metrics

A naive `recall@1` on a synthetic gallery reads **0.89-0.95** and looks alarming. It is a
benchmark artifact, not an accuracy problem: disagreements occur only where the top-1/top-2
similarity margin is **0.00046**, versus **0.0038** when they agree -- an 8x difference, i.e.
only near-ties flip. With genuinely distinct images the margin rises to 0.011 and recall@1
reaches **0.9844**. Report margin-aware metrics or neighbour-set overlap instead.

## Reproducing these numbers

Every table below was produced by a script in [`benchmark/`](benchmark/).
**[benchmark/README.md](benchmark/README.md)** maps each table to its script, gives the exact
command, lists the setup steps (DLAMI, swap, model source), and documents the methodology
traps that will otherwise make your numbers disagree with these.

### Benchmark: inf2.xlarge -- best configuration per model (2 NeuronCores, SDK 2.31)

inf2.xlarge is the smallest and cheapest Inferentia2 instance: **1 Inferentia2 device, 2
NeuronCores, 4 vCPU, 16 GB host RAM**. Each number is the **best of a full sweep** --
dtype (FP32 + `--auto-cast=matmult` vs BF16 weights) x `--model-type=transformer` on/off x
batch size 1/2/4/8/16 -- with RoPE precompute on (the default). **96 configurations were
measured; every one passed an accuracy gate vs CPU FP32 (worst cos_sim 0.99996).**

| Model | Params | **Best img/s** | Config | p50 (ms) | Default config | Gain |
|-------|-------:|---------------:|--------|---------:|---------------:|-----:|
| ViT-S/16 | 21.6M | **2,661.0** | BF16, BS=4 | 3.00 | 1,984.6 | 1.34x |
| ViT-B/16 | 85.7M | **1,119.6** | BF16, BS=2 | 3.57 | 909.7 | 1.23x |
| ConvNeXt-T | 27.8M | **549.6** | BF16, BS=16 | 58.23 | 534.0 | 1.03x |
| ViT-L/16 | 303.2M | **381.8** | BF16, BS=2 | 10.48 | 328.9 | 1.16x |
| ConvNeXt-B | 87.6M | **291.9** | BF16, BS=8 | 54.80 | 286.6 | 1.02x |
| ViT-H+/16 | 840.6M | **162.2** | BF16, BS=1 | 12.48 | 158.1 | 1.03x |
| ViT-7B/16 | 6,716M | **22.1** | BF16, TP=2, BS=8 | -- | 15.1 | 1.46x |

"Default config" is what `trace_dinov3()` produces out of the box (FP32 + `--auto-cast=matmult`,
`--model-type=transformer` for ViT) at its own best batch size. Every ViT best config uses
`--model-type=transformer` (+1-6% over the same config without it). All ViT-7B rows are BF16 because
`compile_vit7b_tp()` is BF16-only; its "default" is BS=1 with in-graph RoPE.

Each cell is the mean of the last two of four converged timed repeats (last two within 3%),
taken with **2 independent worker processes pinned one per NeuronCore**, after 120 warmup
iterations. NEFFs were compiled ahead of time and only loaded on the inf2.xlarge, so no
compilation ever ran alongside a timed loop.

**The entire DINOv3 family -- including the 6.7B ViT-7B -- runs on a single inf2.xlarge.**
ViT-7B uses both cores as one TP=2 replica.

**Versus the previous version of this table** (FP32, RoPE computed in-graph), ViTs are
**3.2x (ViT-S), 2.7x (ViT-B), 2.9x (ViT-L), 2.3x (ViT-H+) and 1.9x (ViT-7B)** faster. Almost
all of that is RoPE precompute; BF16 and batch tuning add the rest. ConvNeXt has no RoPE and
moved only 1.06-1.10x.

#### Picking batch size -- the BF16 cliff

**With BF16 weights, ViT throughput falls off a cliff at BS=8.** FP32 degrades gently instead:

| ViT-L/16, 2 cores | BS=1 | BS=2 | BS=4 | BS=8 | BS=16 |
|-------------------|-----:|-----:|-----:|-----:|------:|
| **BF16** | 369.6 | **381.8** | 314.5 | **72.1** | 107.8 |
| FP32 + matmult | **328.9** | 321.9 | 258.9 | 252.5 | 239.1 |

| ViT-B/16, 2 cores | BS=1 | BS=2 | BS=4 | BS=8 | BS=16 |
|-------------------|-----:|-----:|-----:|-----:|------:|
| **BF16** | 1,041.1 | **1,119.6** | 1,071.0 | 590.0 | 318.1 |
| FP32 + matmult | **909.7** | 809.6 | 828.5 | 551.5 | 558.2 |

The cliff is **device-side, not host-side**: a single-core timing loop with no worker processes
reproduces it exactly (ViT-L BF16 170.9 img/s/core at BS=2, 161.0 at BS=4, 36.2 at BS=8), and
NEFF size is unchanged across batch sizes (~495-506 MB), so it is not a weight-loading effect.
It is reproducible run to run -- the four BF16 BS=8 repeats were 72.1 / 72.2 / 72.1 / 72.1.

**Rules:**
- **BF16 ViT: serve at BS=2-4 (BS=1-2 for ViT-L and ViT-H+). Never BS=8+** -- ViT-L loses 5.3x.
- **FP32 ViT: BS=1 is best**, and larger batches cost little. Use FP32 if you must batch to 8+.
- **ConvNeXt: batching helps slightly** in both dtypes (best at BS=8-16) and has no cliff.
- **ViT-7B (BF16, TP=2): BS=8 is best** (22.1 img/s vs 15.4 at BS=1); it declines at 16+.

#### Scaling to larger inf2 instances

Per-replica throughput is independent of instance size -- each worker is one process on one
NeuronCore -- so **estimate a larger inf2 as (2-core figure / 2) x (core count)**. That
prediction was verified to within 0.1-0.8% on inf2.24xlarge (12 cores) for ViT-S and ViT-B.
Choose instance size on cost and required aggregate throughput, not efficiency.

#### ViT-7B tensor parallelism on inf2

| TP | Params/rank | Compile | img/s | p50 (ms) | Peak core HBM |
|---:|------------:|--------:|------:|---------:|--------------:|
| 1 | 6,716M | -- | **OOM** | -- | 15.95 GB -> fail |
| 2 | 3,360M | 8.5s | **12.5** | 76.9 | 14.72 GB (92%) |
| 4 | 1,682M | 7.1s | 11.9 | 79.7 | -- |
| 8 | 843M | 6.5s | **15.9** | 70.5 | -- |

TP=2 and TP=4 are within noise of each other; TP=8 is ~27% faster than TP=2. Compile time is dominated by the TP sharding path and is fast (6-9s) because each rank compiles a smaller graph. (TP sweep measured BS=1, in-graph RoPE, on inf2.24xlarge.)

**Tuning ViT-7B at TP=2 on inf2.xlarge** (BF16, RoPE precompute, converged):

| Config | img/s |
|--------|------:|
| BS=1, in-graph RoPE, `-O1` (`compile_vit7b_tp()` default) | 15.1 |
| BS=1, RoPE precompute, `-O1` | 15.4 |
| BS=1, RoPE precompute, `-O2` | 16.1 |
| BS=2 | 17.6 |
| BS=4 (`-O1` or `-O2`, identical) | 18.9 |
| **BS=8** | **22.1** |
| BS=16 | 19.2 |
| BS=32 | 17.4 |

Unlike the smaller ViTs, **RoPE precompute is worth only +2% on ViT-7B** -- at 6.7B parameters
the 40 transformer layers dominate and the RoPE table is a rounding error. **Batch size is the
lever: BS=8 is 1.46x BS=1.** `-O2` helps only at BS=1 (+4%) and is identical from BS=4 on.

### Tested and rejected: NKI kernels on the traced path

A custom NKI kernel replacing `nn.GELU` with a single Scalar/activation-engine op was
built, validated, and **rejected on measurement**. Recording it here so a *single-op*
GELU kernel is not re-attempted on this path. This is a result about one kernel shape --
see the scope note at the end of this section before generalizing from it.

The rationale was sound on paper: `neuronx-cc` lowers exact erf-GELU as **1664 matmuls +
256 reciprocals + 2816 vector ops** where the kernel emits ~26 activation-table ops, and
the DINOv3 forward is **Vector-engine bound (86% active, Tensor only 57%)** with GELU at
**34% of forward Vector+Scalar time**. The kernel is also numerically exact
(**cos_sim 1.000000** on ViT, 0.99999 on ConvNeXt).

It is nonetheless slower almost everywhere. inf2.xlarge, 1 core, best-of-N per side:

| Model | BS=1 | BS=4 | BS=8 |
|-------|-----:|-----:|-----:|
| ViT-S/16 | -11.8% | -7.8% | -5.8% |
| ViT-B/16 | -10.7% | -8.1% | -7.9% |
| ViT-L/16 | +2.8% | -- | -5.5% |
| ConvNeXt-T | -12.7% | -8.4% | -9.7% |
| ConvNeXt-B | -14.5% | -7.3% | -10.8% |

**13 of 14 configurations regress.** The one positive (ViT-L BS=1, +2.8%) is within
run-to-run noise.

**Measurement caveat worth internalizing:** an earlier version of this table reported
**+32%** for ViT-L at BS=1. That was wrong -- the *baseline* had not converged. Measured
back-to-back in one process, the ViT-L BS=1 baseline climbs **52.0 -> 62.4 -> 68.4 img/s**
across three repeats, while the NKI path is flat at 70.2 from the first. Comparing a
warm kernel against a cold baseline manufactured the entire gain. **Always repeat the
baseline until it converges before claiming a kernel win.**

Root cause of the regression: the compiler already fuses GELU into the surrounding MLP
GEMMs, so a standalone kernel adds an HBM round-trip per call (load -> activate -> store)
that outweighs the cheaper transcendental. A separate fused-MLP NKI attempt (fc1 -> GELU
-> fc2 in one kernel) also regressed ~23%, so fusing more ops is not automatically a win.

The kernel remains in `src/nki_gelu_trace.py` as a **reference implementation** of the NKI
`nki_jit` calling convention for `torch_neuronx.trace()` -- it is not wired into
`trace_dinov3()` and should not be enabled without re-measuring.

#### Fused residual + LayerNorm kernel (ported from the whole-block fused path) -- also rejected

Separate work fusing whole transformer blocks into NKI -- attention + residual + LayerNorm, and
FFN + residual + next LayerNorm, keeping the fp32 residual stream on-chip -- measured **1.5-3.4x
over the compiler** on DINOv3 ViTs, but on a different compilation path and on Trainium2. Its
simplest kernel, the fused residual-add + LayerNorm, was ported to `torch_neuronx.trace()` and
tested on inf2.xlarge as a feasibility gate for the rest.

It fits comfortably in gen2 SBUF (13-43% of the 180 KB usable per partition across ViT-S..H+),
**compiles on Inferentia2, and is exact** (cos_sim 1.000001-1.000004, max abs err <= 1.7e-4) --
but it is **slower inside a real DINOv3 block**, replacing `x + ls1(attn(...))` and `norm2(...)`:

| Model, one block | Compiler | Fused (padded) | Fused (lean) |
|------------------|---------:|---------------:|-------------:|
| ViT-S/16 | 3,568.6 it/s | 0.807x | 0.870x |
| ViT-B/16 | 2,251.3 it/s | 0.927x | 0.917x |
| ViT-L/16 | 1,768.3 it/s | 0.911x | 0.929x |

"Lean" removes all wrapper overhead -- no host-side padding of the 201-token sequence to 256
(the last tile is masked instead) and the LayerNorm gamma/beta folded into the kernel. It
recovers ~6 points on ViT-S and nothing on ViT-B/L, so **the remaining loss is the kernel
boundary itself**: it forces the attention output through HBM and stops the compiler fusing
LN2 into the following fc1 GEMM. All six arms converged (last two repeats within 3%).

#### Fused attention + LayerNorm and fused MLP + LayerNorm kernels -- ported, also rejected

The two larger whole-block kernels were then ported as well, because their effect was hard to
predict from the small one:

- **attention + residual + LN2** (QKV projection with norm1 folded into the weights, in-kernel
  RoPE, softmax attention, out-projection with LayerScale folded in, residual add, LN2), and
- **MLP + residual + next LN1** (fc1, GELU, fc2 with LayerScale, residual add, next block's LN).

Porting them to Inferentia2 (NeuronCore-v2) under `trace()` took four changes, each forced by
the compiler. `tensor_copy` on the Scalar engine becomes `activation(op=copy)`. The SBUF->SBUF
`dma_transpose` of the softmax probabilities becomes a Tensor-engine `nc_transpose` through PSUM,
because gen2 only supports HBM->SBUF DMA transposes. The LayerNorm mean moves from an
`activation(copy, bias=tensor)` reduce to `tensor_reduce`, because gen2 accepts only an immediate
bias there. The LNC `program_id` sharding is removed. Wheel-style `@nki.jit` kernels can be
called directly inside `trace()`.

Both ported kernels **compile and are exact**: cos_sim >= 0.99999 against PyTorch inside a
block, and full-model cos_sim matching the compiler arm (0.99998 FP32, 0.9999 BF16). The fused
MLP does not fit gen2 SBUF at ViT-L width (106%) without weight streaming, so it was tested on
ViT-S/B only.

**Inside one block (1 core, ViT-S/B/L):** fused attention measured 1.074x / 0.970x / 1.042x.

**Whole model, every block fused, 1 core, inf2.xlarge:**

| Model | Dtype | BS | attn | mlp | attn+mlp |
|-------|-------|---:|-----:|----:|---------:|
| ViT-S/16 | FP32 | 1 | 0.789x | 0.831x | 0.748x |
| ViT-S/16 | FP32 | 4 | 0.895x | 1.021x | 0.886x |
| ViT-S/16 | BF16 | 1 | 0.690x | 0.812x | 0.655x |
| ViT-S/16 | BF16 | 4 | 0.670x | 0.924x | 0.658x |
| ViT-B/16 | FP32 | 1 | 0.828x | 0.847x | 0.799x |
| ViT-B/16 | BF16 | 1 | 0.733x | 0.857x | 0.708x |

On ViT-S and ViT-B the wins in the 2-block slices do not survive at full depth. The extra
kernel boundaries (two per block) cost more than the fusion inside them saves.

ViT-B at BS >= 2 cannot be read as a ratio. The same compiled graph on a single core ran
either at ~560 img/s or at ~230 img/s, sometimes switching between the two inside one run of
repeats, and the compiler arm hit the slow state more often. This is the reason the best-config
benchmark uses converged repeats on two independent worker processes. The decisive comparison
for ViT-B is therefore the best value ever observed: the compiler graph reaches 570 img/s per
core (BF16 BS2, which matches the 1,119.6 img/s 2-core best config). The best any fused arm
ever reached is 514.6 img/s (MLP), 426.0 (attention) and 397.0 (both).

**ViT-L, fused attention, standard 2-worker measurement** (img/s aggregate, last-2 mean):

| Dtype | BS | Compiler | Fused attention | Ratio |
|-------|---:|---------:|----------------:|------:|
| BF16 | 1 | 298.3 | 273.4 | 0.92x |
| BF16 | 2 | 317.4 | 298.1 | 0.94x |
| BF16 | 4 | 287.2 | 313.8 | 1.09x |
| FP32 + matmult | 4 | 257.2 | 314.7 | 1.22x |

Fused attention does help ViT-L at BS=4, where the compiled graph sits past its batch cliff.
It does not reach the ViT-L best config (381.8 img/s, BF16 BS2, in the best-config table),
so no shipped configuration improves.

**Verdict: none of the fused kernels are wired in.** On the traced path the compiler's
cross-block fusion beats hand fusion within a block for every DINOv3 width that fits. The only
win is a partial recovery of the BS=4 cliff, and the best configuration already avoids that
cliff.

> **Scope.** These results are specific to `torch_neuronx.trace()` on Inferentia2. They do not
> show that NKI cannot help DINOv3 in general -- the whole-block kernels win on Trainium2 on a
> different compilation path, where the whole block is the kernel. The rule that holds on
> both paths: kernel *boundary count* dominates kernel *content*. A kernel must remove
> boundaries, not add one.

### BF16 weights vs FP32 + `--auto-cast=matmult`

`trace_dinov3()` traces **FP32 weights + `--auto-cast=matmult`** by default. Tracing **BF16
weights** (`--auto-cast=none`) is accuracy-neutral (cos_sim >= 0.999996 on ViT, >= 0.99996 on
ConvNeXt vs CPU FP32) and, **at the right batch size**, faster. inf2.xlarge, 2 cores, each
dtype at its own best batch size:

| Model | FP32 + matmult | BF16 | Gain |
|-------|---------------:|-----:|-----:|
| ViT-S/16 | 1,984.6 (BS=2) | **2,661.0** (BS=4) | **1.34x** |
| ViT-B/16 | 909.7 (BS=1) | **1,119.6** (BS=2) | **1.23x** |
| ViT-L/16 | 328.9 (BS=1) | **381.8** (BS=2) | **1.16x** |
| ViT-H+/16 | 158.1 (BS=1) | **162.2** (BS=1) | 1.03x |
| ConvNeXt-T | 534.0 (BS=8) | **549.6** (BS=16) | 1.03x |
| ConvNeXt-B | 286.6 (BS=16) | **291.9** (BS=8) | 1.02x |

**BF16 is the faster choice for every model, but only at small batch** -- see "Picking batch
size -- the BF16 cliff" above. At BS=8 BF16 is *slower* than FP32 on ViT-L (72.1 vs 252.5).
The FP32 default is kept because it is safe at any batch size; switch to BF16 when you
control the batch size and will keep it at 1-4.

```python
model = load_dinov3_model("dinov3_vitl16", repo_dir=REPO)
precompute_rope(model, img_size=224)
model = model.to(torch.bfloat16)
example = torch.randn(2, 3, 224, 224, dtype=torch.bfloat16)        # BS=2 for ViT-L
model_neuron = torch_neuronx.trace(model, example,
                                   compiler_args=["--auto-cast", "none",
                                                  "--model-type", "transformer"],
                                   inline_weights_to_neff=True)
```

`--auto-cast=matmult` is a **no-op once the weights are BF16**: `bf16_matmult` and `bf16_none`
measured within noise of each other on all five models (e.g. ConvNeXt-B 1,740.7 vs 1,740.4
img/s), confirming the flag only casts FP32 matmuls.

> **Correction.** A previous version of this section said "ViT-H+ at batch 1 is 4.87x faster in
> BF16" because `neuronx-cc` lowers SwiGLU badly in FP32 at low batch (60.6 -> 294.9 img/s on 12
> cores). That was measured with RoPE computed in-graph. **With RoPE precompute (now the default),
> ViT-H+ FP32 BS=1 runs at 158.1 img/s on 2 cores and BF16 adds only 3%.** The "FP32 SwiGLU cliff"
> was really the in-graph RoPE; precompute removes it for both dtypes.

### Critical: `torch_neuronx.DataParallel` defaults to 2 worker threads

**`torch_neuronx.DataParallel` sets `num_workers = 2` regardless of how many NeuronCores you give it.** When device IDs are consecutive (the default), `_load_modules()` batch-loads into a single module object and sets `num_workers = 2 * 1 = 2`, so `parallel_apply()`'s `ThreadPoolExecutor` serializes all but two core dispatches.

Measured on inf2.24xlarge, ViT-L, 12 cores (single-core reference 65.1 img/s):

| `num_workers` | Throughput | Scaling | p50 (ms) |
|--------------:|-----------:|--------:|---------:|
| **2 (library default)** | 155.7 img/s | 2.39x | 77.05 |
| 4 | 307.9 img/s | 4.73x | 38.93 |
| 8 | 458.0 img/s | 7.04x | 26.19 |
| **12 (= core count)** | **806.2 img/s** | **12.38x** | **14.96** |
| 16 | 802.7 img/s | 12.33x | 14.97 |

**The one-line fix is worth 5.18x.** `num_workers` is not a constructor argument -- set it as an attribute after construction. Raising it above the core count gives nothing.

```python
model_dp = torch_neuronx.DataParallel(model_neuron, device_ids=list(range(num_cores)), dim=0)
model_dp.num_workers = num_cores   # REQUIRED -- default of 2 serializes across cores
```

`benchmark_dataparallel()` in `src/modeling_dinov3.py` now applies this automatically.

With the fix, `DataParallel` (806.2 img/s) comes within **14%** of independent worker processes (939.5 img/s). Processes still win -- they avoid the GIL entirely for input/output marshaling -- and remain the recommendation for production serving, but `DataParallel` is no longer catastrophically slow.

**Use exactly one worker per NeuronCore.** Oversubscribing fails: 4 processes on 2 cores produced `RuntimeError: The PyTorch Neuron Runtime could not be initialized` for the extras. A NeuronCore is single-tenant.


### Key Findings

1. **Precompute RoPE out of the traced graph -- the single largest lever, ~2.0x per core.** The reference-repo ViT computes 2D-axial RoPE (`torch.arange` / `torch.meshgrid` / `torch.cos` / `torch.sin` over `[H*W, D]`) *inside* `forward()` on every call, and Neuron traces these poorly. Baking the tables as constants (`trace_dinov3(..., precompute_rope_tables=True)`, default on) is accuracy-neutral (cos_sim 1.000000) and, measured single-core FP32 with converged baselines on both arms, worth **2.14x (ViT-S), 1.64x (ViT-B), 2.04x (ViT-L)**. It also removes what had looked like an FP32 SwiGLU problem on ViT-H+. It is worth only +2% on ViT-7B, where the 40 layers dominate. ⚠️ **It pins the compiled NEFF to one image size** -- compile one NEFF per resolution, or pass `precompute_rope_tables=False` for variable-resolution inference. ConvNeXt has no RoPE and is unaffected. *(An earlier figure of 2.76x was measured against an unconverged in-graph baseline: ViT-L in-graph climbs 48.6 -> 78.8 img/s over ~7 repeats, and the 2.76x used 58.9.)*
2. **BF16 weights add 1.16-1.34x on ViT -- but only at small batch.** BF16 has a sharp device-side cliff at BS=8 (ViT-L 381.8 at BS=2 -> 72.1 at BS=8); FP32 degrades gently. Serve BF16 ViT at **BS=1-4** and never BS=8+. FP32 + `--auto-cast=matmult` remains the default because it is safe at any batch size. See "Picking batch size -- the BF16 cliff".
3. **`--model-type=transformer` is a small net win on ViT: +1-6% at each model's best config.** It is not uniformly positive -- across 36 matched (dtype, BS) pairs it ranged 0.87x to 1.18x and lost in 5 -- but every model's best configuration uses it. Keep it (the default). Not applicable to ConvNeXt.
4. **`torch_neuronx.trace()` miscompiles the HuggingFace `transformers` DINOv3 port** (cos_sim 0.50 with real weights, even at `--auto-cast=none`). Root cause: `apply_rotary_pos_emb` uses `split()` with an explicit size list, which torch_xla mis-lowers when `XLA_DISABLE_FUNCTIONALIZATION=1` (the torch_neuronx default). This is not a compiler issue, and it reproduces on trn2 and inf2; a no-model micro-repro of just that op gives cos 0.17. **Replacing that one function with index slicing, or setting `XLA_DISABLE_FUNCTIONALIZATION=0` before importing torch_neuronx, restores cos 1.000000.** Use the `dinov3` reference-repo architecture (which already uses index slicing) and remap the published checkpoints into it -- see "With real pretrained weights".
5. **Inferentia2 runs the entire DINOv3 family**, including the 6.7B ViT-7B at TP=2 -- and all of it fits on a single **inf2.xlarge** (see "Maximum model size on Inferentia2").
6. **ViT-7B: batch size is the lever.** At TP=2, BS=8 gives 22.1 img/s vs 15.4 at BS=1 (1.46x). RoPE precompute (+2%) and `-O2` (+4% at BS=1 only) are marginal.
7. **Hand-written NKI kernels do not help on this path.** A single-op GELU kernel regressed 13 of 14 configs (-5 to -15%). A port of the fused residual+LayerNorm kernel from the whole-block fused path also regressed (-4 to -13% inside a real block) even after removing all wrapper overhead (no padding, LN affine folded in), because the kernel boundary itself blocks the compiler's fusion of LN2 into the following GEMM. The larger fused attention+LN and MLP+LN kernels were ported too. They are exact on Inferentia2 but slower at full depth on ViT-S/B (0.65-0.92x). On ViT-L they help only at BS=4 (up to 1.22x) and still fall short of the ViT-L best config. See "Tested and rejected: NKI kernels on the traced path".
8. **`DataParallel` needs `num_workers = num_cores`.** The library default of 2 costs **1.82-1.87x** on 4 trn2 cores and **5.18x** on 12 inf2 cores (the penalty scales with core count, since the default caps concurrency at 2). This is an API fix for anyone using `DataParallel` -- it does not affect the tables in this README, which all use independent worker processes.
9. **Inline weights into the NEFF.** `inline_weights=False` costs **6.8x** on inf2 (ViT-L 56.9 -> 8.4 img/s).
10. **Per-core throughput is independent of inf2 instance size.** One worker per NeuronCore scales linearly: the 12-core inf2.24xlarge reproduced 6x the 2-core inf2.xlarge figure to within 0.1-0.8% (ViT-S, ViT-B). Buy cores for throughput, not efficiency.
11. **ViT-7B requires TP>=2 on inf2**: at TP=1 the runtime reaches 15.95 GB of the 16 GB core and OOMs. TP must be a power of 2.
12. **With RoPE precompute, ViT is far ahead of ConvNeXt at equal size.** At the only near-matched pair (ViT-B 85.7M vs ConvNeXt-B 87.6M), ViT-B is **3.8x faster** on inf2.xlarge (1,119.6 vs 291.9 img/s). ConvNeXt does not benefit from RoPE precompute (it has no RoPE).

### Benchmark: Trainium2 (trn2.3xlarge, SDK 2.31)

Re-measured 2026-09-23 with independent worker processes and a batch-size sweep.
trn2.3xlarge is one Trainium2 device: LNC=2 (default) gives 4 logical cores at 24 GB each,
LNC=1 gives 8 at 12 GB.

| Model | LNC=2 (4 cores) | LNC=1 (8 cores) | Best | Winner |
|-------|----------------:|----------------:|-----:|--------|
| ViT-S/16 | 3,780.4 | **3,893.7** | 3,893.7 | LNC=1 |
| ConvNeXt-T | 2,310.9 | **2,401.8** | 2,401.8 | LNC=1 |
| ViT-B/16 | 1,930.5 | **1,969.5** | 1,969.5 | LNC=1 |
| ConvNeXt-B | 1,285.6 | 1,298.6 | 1,298.6 | tie |
| ViT-L/16 | 348.8 | **739.7** | 739.7 | **LNC=1 (2.12x)** |

**LNC=1 is the better default on trn2 for every model tested.** The margin is small
(1.0-1.04x) except for **ViT-L, where LNC=1 is 2.12x faster** -- consistent with the earlier
finding that ViT-L is memory-bandwidth bound and benefits from the smaller per-core NEFF.

#### These numbers supersede the previous trn2 table, which was understated 4.2-5.4x

The prior table reported DP=4 peaks of ViT-S 722.8 / ViT-B 422.9 / ViT-L 174.7 /
ConvNeXt-T 522.6 / ConvNeXt-B 257.8 img/s, measured with `torch_neuronx.DataParallel` at
its **broken `num_workers=2` default** and at **BS=1 only**. Re-measured properly they are
**4.23x to 5.39x higher**.

**Two independent causes were at work, and their relative weight differs sharply by model** --
worth knowing, because the fix differs. Decomposing ViT-L:

| Step | img/s | Factor |
|------|------:|-------:|
| old published (DP, `num_workers=2`, BS=1) | 174.7 | -- |
| re-measured, same config | 164.4 | reproduces the old value |
| + `num_workers=4` | 305.8 | **1.86x** |
| + independent processes and batch sweep | 348.8 | 1.14x |

For **ViT-L the `num_workers` defect is the dominant term** (1.86x of a 2.00x gap). For
**ViT-H+ it contributes nothing**: BS=1 re-measures at 11.2 against an old value of 10.5,
and the entire 10.7x comes from batching (see below). Do not read the 4.2-5.4x range as a
single effect.

The `num_workers` defect itself reproduces on trn2 as on inf2 (LNC=2, 4 cores, BS=1):

| Model | `num_workers=2` (default) | `num_workers=4` (fixed) | Gain |
|-------|--------------------------:|------------------------:|-----:|
| ViT-S/16 | 651.4 img/s | 1,216.0 | 1.87x |
| ViT-B/16 | 400.1 img/s | 726.8 | 1.82x |
| ViT-L/16 | 164.4 img/s | 305.8 | 1.86x |

Note ViT-L's broken-default figure (164.4) closely reproduces the old table's 174.7 --
confirming the historical numbers were measured with the defect.

#### Setting LNC on trn2: `NEURON_CC_FLAGS` is NOT enough

`trace_dinov3()` passes an explicit `compiler_args` list to `torch_neuronx.trace()`, which
**overrides the `NEURON_CC_FLAGS` environment variable**. Exporting
`NEURON_CC_FLAGS="--lnc=1"` therefore has no effect -- the NEFF compiles at the default
`--lnc=2` and the runtime rejects it at load:

```
NRT:nrt_load_util  Mismatch detected between Runtime configuration and NEFF.
    Runtime currently configured with `NEURON_LOGICAL_NC_CONFIG=1`
    but NEFF ... was compiled with `--lnc=2`.
```

Pass the target through `compiler_args` instead, alongside the runtime variable:

```python
compiler_args = COMPILER_ARGS_VIT + ["--logical-nc-config", "1"]   # compile-time
# and export NEURON_LOGICAL_NC_CONFIG=1                            # runtime
```

#### ViT-H+ and ViT-7B on trn2 (SDK 2.31)

| Model | Config | BS=1 | BS=4 | Compile |
|-------|--------|-----:|-----:|--------:|
| ViT-H+/16 | 4 independent procs, LNC=2 | 11.2 | **119.8** | 950s / 525s |
| ViT-7B/16 | TP=4, LNC=2 | **34.9** | -- | -- |

**ViT-H+ gains 10.7x from BS=1 -> BS=4** (11.2 -> 119.8 img/s). Its old SDK 2.28 figure of
10.5 img/s was understated **11.4x**, but note the cause: BS=1 re-measures at 11.2, almost
exactly the old value, so the understatement came **entirely from not batching**, not from
the `num_workers` defect. ViT-H+ must be batched.

ViT-7B TP=4 reproduces at 34.9 img/s vs the old 38.8 (-10%), i.e. no meaningful change on
SDK 2.31. TP models dispatch from a single process, so the `num_workers` defect never
applied to it.

## Compatibility

| Component | Version |
|-----------|---------|
| **Neuron SDK** | 2.31 |
| **torch-neuronx** | 2.9.0.2.15.32035 |
| **neuronx-cc** | 2.26.6360.0 |
| **neuronx-distributed** | 0.19.28492 (for ViT-7B TP only) |
| **torch** | 2.9.1 |
| **Instance (ViT-S/B/L/H+, ConvNeXt-T/B)** | inf2.xlarge and up, trn2.3xlarge |
| **Instance (ViT-7B)** | inf2.24xlarge (TP>=2), trn2.3xlarge (TP=4, LNC=2) |
| **DLAMI** | Deep Learning AMI Neuron (Ubuntu 24.04) 20260708 |

> **SDK 2.32 is not supported.** SDK 2.32 removed `neuronx-distributed` and
> `neuronx-distributed-inference` from the DLAMI entirely, and its vLLM venv ships only
> `libtorch-neuronx-lite` (no `torch_neuronx`). The venv path used below does not exist on
> 2.32. **Use SDK 2.31 (DLAMI 20260708 or 20260813).**


## Usage

### Setup

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate  # SDK 2.31

# Clone DINOv3 repository
git clone https://github.com/facebookresearch/dinov3.git /mnt/models/dinov3
```

### Trace a ViT Model

```python
import sys
sys.path.insert(0, "contrib/models/DINOv3/src")
from modeling_dinov3 import load_dinov3_model, trace_dinov3, validate_accuracy, benchmark_model

# Load model
model = load_dinov3_model("dinov3_vitb16", repo_dir="/mnt/models/dinov3")

# Compile for Neuron (RoPE precomputed by default -> ~2x per core; pins the NEFF to img_size=224)
model_neuron = trace_dinov3(model, is_convnext=False, save_path="/tmp/dinov3_vit_b.pt")

# For VARIABLE input resolution instead, keep RoPE in-graph (slower, but any H x W):
#   model_neuron = trace_dinov3(model, is_convnext=False, precompute_rope_tables=False)
# Or compile one NEFF per resolution you serve, passing img_size to trace_dinov3.

# Validate accuracy (uses the SAME img_size the model was traced at)
metrics = validate_accuracy(model, model_neuron)
print(f"Cosine similarity: {metrics['cosine_sim']:.6f}")

# Benchmark
perf = benchmark_model(model_neuron)
print(f"Throughput: {perf['throughput_img_s']:.1f} img/s")
```

### Trace a ConvNeXt Model

```python
model = load_dinov3_model("dinov3_convnext_tiny", repo_dir="/mnt/models/dinov3")
model_neuron = trace_dinov3(model, is_convnext=True, save_path="/tmp/dinov3_convnext_t.pt")
```

### Compile ViT-7B with Tensor Parallelism

```python
from modeling_dinov3 import compile_vit7b_tp, benchmark_model

# On inf2: TP=2 is the minimum (TP=1 OOMs at 16 GB/core). TP must be a power of 2.
# Requires NEURON_RT_NUM_CORES >= tp_degree.
nxd_model = compile_vit7b_tp(tp_degree=2)     # inf2.24xlarge
# nxd_model = compile_vit7b_tp(tp_degree=4)   # trn2.3xlarge
perf = benchmark_model(nxd_model)
print(f"throughput: {perf['throughput_img_s']:.1f} img/s")
```

### Multi-core throughput

Two options. `DataParallel` is fine **if you set `num_workers`** (see above); independent
processes are slightly faster and are the production recommendation:

```bash
# Launch one worker per core, each pinned to its own NeuronCore.
for core in $(seq 0 11); do
    NEURON_RT_VISIBLE_CORES=$core NEURON_RT_NUM_CORES=1 \
        python my_server.py --port $((8000 + core)) &
done
```

```python
from modeling_dinov3 import benchmark_dataparallel

# Convenience API -- benchmark_dataparallel() sets num_workers = num_cores for you
dp_results = benchmark_dataparallel(model_neuron, num_cores=12)
```

## Running Tests

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate  # SDK 2.31

# Run integration tests
python -m pytest contrib/models/DINOv3/test/integration/test_model.py -v

# Or standalone (with detailed output)
python contrib/models/DINOv3/test/integration/test_model.py

# Set custom paths
DINOV3_REPO_DIR=/path/to/dinov3 python -m pytest contrib/models/DINOv3/test/ -v
```

## Dependencies

Pre-installed in DLAMI PyTorch inference venv:
- torch-neuronx
- neuronx-distributed (for ViT-7B TP)
- numpy

Required (clone separately):
- DINOv3 repository: `git clone https://github.com/facebookresearch/dinov3.git`

## Notes

- All models use `pretrained=False` (random weights) for architecture validation. Replace with pretrained weights for production use.
- `--model-type=transformer` compiler flag is used for ViT models only (not ConvNeXt).
- ConvNeXt models exercise different Neuron ops (Conv2d, depthwise conv, GroupNorm) -- good diversity test for Neuron compiler.
- ViT-H+ runs at 158-162 img/s on an inf2.xlarge with RoPE precompute (both dtypes, BS=1). With RoPE computed in-graph it was 10.2 img/s at BS=1 in FP32 -- that slowdown, earlier attributed to FP32 SwiGLU lowering, was the in-graph RoPE.
- **Keep `inline_weights=True` (the default) on inf2.** Disabling it costs 6.8x throughput -- see "Critical: always inline weights into the NEFF on inf2". A prior recommendation to set `inline_weights=False` at BS>=4 came from a trn2/SDK 2.30 async XLA error and does not apply to inf2 on SDK 2.31.
- **Older SDKs miscompiled RoPE on Inferentia2.** `torch.Tensor.split()` on dim>=2 of a 4D tensor was compiled incorrectly by neuronx-cc 2.25.3371, collapsing ViT-L cosine similarity to 0.13 on inf2. Current upstream DINOv3 uses index slicing in `Attention.apply_rope`, and **cosine similarity is 1.000000 on inf2 with SDK 2.31** -- verified, no workaround needed. If you pin an older DINOv3 commit that still uses `split()`, replace it with index slicing or `.narrow()`.
- DINOv3 License is not Apache/MIT -- review before redistribution.


## Maintainer

Jim Burtoft (`jimburtoft`)
