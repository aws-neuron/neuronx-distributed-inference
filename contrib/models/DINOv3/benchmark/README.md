# Reproducing the DINOv3 benchmarks

Every number in the parent [README.md](../README.md) was produced by a script in this
directory. This file maps each table to its script and gives the exact command.

All results were measured on **Neuron SDK 2.31**:

| Component | Version |
|---|---|
| DLAMI | `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708` |
| neuronx-cc | 2.26.6360.0 |
| torch-neuronx | 2.9.0.2.15.32035 |
| neuronx-distributed | 0.19.28492 (ViT-7B TP only) |
| torch | 2.9.1 |
| venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` |

> **SDK 2.32 will not work.** It removed `torch_neuronx` and `neuronx_distributed` from the
> DLAMI entirely. Use a 2.31 DLAMI (`20260708` or `20260813`).

## Setup

```bash
# 1. Launch an inf2 or trn2 instance with the SDK 2.31 DLAMI above.
#    inf2.xlarge (2 cores) is enough for the single-instance tables.
#    inf2.24xlarge (12 cores) is needed for the 12-core scaling numbers.

# 2. Add swap. inf2.xlarge has only 15 GB RAM and neuronx-cc needs far more per
#    graph; without swap, large-model compiles are OOM-killed.
sudo fallocate -l 96G /swapfile && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile

# 3. Clone the DINOv3 reference repo (the model source).
sudo mkdir -p /mnt/models && sudo chown $USER /mnt/models
git clone --depth 1 https://github.com/facebookresearch/dinov3.git /mnt/models/dinov3

# 4. Activate the venv and point the scripts at the repo + model source.
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export DINOV3_REPO_DIR=/mnt/models/dinov3
export DINOV3_CONTRIB_SRC=$PWD/../src
```

## Which script produces which table

| README table | Script | Command |
|---|---|---|
| **inf2.xlarge best config per model** (headline table, BF16 cliff tables) | `sweep_best.py` | 3 phases -- see [Best config sweep](#best-config-sweep) below |
| **ViT-7B tuning (BS, RoPE, -O level)** | `bench_vit7b_tune.py` | `NEURON_RT_VISIBLE_CORES=0-1 python bench_vit7b_tune.py vit7b --tp 2 --bs 8 --rope precompute --opt=-O1` |
| **HF-port miscompile root cause** | `repro_hf_rootcause.py` | no args; needs an approved HF token |
| **Rejected fused residual+LN kernel** | `gate_fused_port.py`, `diag_fused_overhead.py` | no args; 1 core |
| inf2.xlarge quick throughput (older single-pass harness) | `bench_max_throughput.py` | `--models vit_s,vit_b,vit_l,convnext_tiny,convnext_base --workers 2 --batch-sizes 1,2,4,8,16 --n-cores 2` |
| **inf2 single-core + accuracy + artifact size** | `bench_inf2.py` | `--models vit_s,vit_b,vit_l,vit_h_plus,convnext_tiny,convnext_base --dp-cores 12` |
| **Maximum model size / ViT-7B TP** | `bench_vit7b_tp_inf2.py` | `NEURON_RT_NUM_CORES=8 python bench_vit7b_tp_inf2.py --tp 2,4,8` |
| **trn2 LNC=2 / LNC=1** | `bench_trn2.py` | see [trn2](#trn2) below |
| **trn2 ViT-H+ and ViT-7B** | `bench_trn2_big.py` | `--mode vith --cores 4 --batch-sizes 1,4` then `--mode vit7b --tp 4` |
| **`DataParallel` num_workers** | `bench_dp_numworkers.py` | `--model vit_l --cores 12 --workers 2,4,8,12,16` |
| **Multi-process vs DataParallel** | `bench_dp_mp.py` | `--model vit_l --cores 1,12 --batch-size 1` (one `--model` at a time) |
| **BF16 vs FP32+matmult** | `bench_bf16_default.py` | `--models vit_s,vit_b,vit_l,convnext_tiny,convnext_base --cores 12 --batch-sizes 1,4,8` |
| **Accuracy on real pretrained weights** | `validate_real_weights.py` | `--models vit_s,vit_b,vit_l --dtypes fp32,bf16 --n 64` |
| **recall@1 near-tie analysis** | `diag_recall.py` | `--model vit_l --dtype bf16 --n 64` |
| **Rejected NKI GELU kernel** | `ab_gelu_all.py` | `--models vit_s,vit_b,convnext_tiny,convnext_base --batch-sizes 1,4,8` |

Supporting modules (not run directly): `hf_to_repo_weights.py` remaps HuggingFace
checkpoints into the reference-repo architecture; `test_nki_gelu_inf2.py` is the 3-gate
harness for the NKI GELU kernel; `nki_res_ln_trace.py` is the ported fused residual+LayerNorm
kernel used by `gate_fused_port.py`.

## Headline numbers and how to get them

<a id="best-config-sweep"></a>
### inf2.xlarge best config per model -- `sweep_best.py`

This is the harness behind the headline table. It exists because simpler harnesses produced
wrong numbers in four distinct ways during this work (see Methodology notes). It runs in three
separate phases so compilation never overlaps timing:

```bash
# Phase 1 -- compile every config to a saved NEFF (torch.jit.save). Slow; RAM-hungry.
#   On an inf2.24xlarge (96 vCPU / 369 GB) run 10 compiles in parallel:
python sweep_best.py phase1 --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
    --batch-sizes 1,2,4,8,16 --out-dir neffs --jobs 10 --cores-avail 12
#   On an inf2.xlarge use --jobs 1 (one compile at a time, swap required).

# Phase 2 -- accuracy gate: every NEFF vs CPU FP32 on identical seeded weights.
python sweep_best.py phase2 --out-dir neffs

# Phase 3 -- timing on inf2.xlarge. 2 worker processes LOAD the saved NEFFs (no compile).
python sweep_best.py phase3 --out-dir neffs --cores 2 --iters 60 --warmup 120 --repeats 4
```

NEFFs are portable between inf2 sizes (same NeuronCore-v2, same SDK), so phase 1 can run on a
large instance and phase 3 on an inf2.xlarge. Phase 3 reports every repeat, a convergence flag
(last two within 3%), and `last2_mean`, which is what the README tables quote.

The sweep covers dtype (FP32 + `--auto-cast=matmult` | BF16) x `--model-type=transformer` on/off
x BS 1/2/4/8/16, RoPE precompute always on -- 80 configs for the five models above, plus
`--models vit_h_plus --batch-sizes 1,2,4,8` for 16 more. ViT-H+ at BS=1-2 needs
`--warmup 400 --repeats 5` to converge.

Expected best results (inf2.xlarge, 2 cores): ViT-S **~2,661**, ViT-B **~1,120**,
ConvNeXt-T **~550**, ViT-L **~382**, ConvNeXt-B **~292**, ViT-H+ **~162** img/s, all BF16.

**inf2.24xlarge (12 cores):** per-core throughput does not depend on instance size, so expect
**6x** the inf2.xlarge numbers (verified to within 0.8% for ViT-S and ViT-B with
`bench_max_throughput.py --workers 12 --n-cores 12`).

### ViT-7B with tensor parallelism

```bash
# TP must be a power of 2, and TP>=2 is required (TP=1 OOMs at 16 GB/core).
NEURON_RT_NUM_CORES=8 python bench_vit7b_tp_inf2.py --tp 2,4,8 --out results_7b.json
```

### trn2

LNC must be set in **two** places -- the runtime env var **and** the compile target -- or the
runtime rejects the NEFF at load with a `Mismatch detected between Runtime configuration and
NEFF` error.

`NEURON_CC_FLAGS="--lnc=N"` alone is **not** sufficient: `trace_dinov3()` passes an explicit
`compiler_args` list, which overrides the env var, so the NEFF silently compiles at the
default `--lnc=2`. `bench_trn2.py` reads `NEURON_LOGICAL_NC_CONFIG` and appends
`--logical-nc-config` to `compiler_args` for you, so setting the one env var is enough here:

```bash
export NEURON_LOGICAL_NC_CONFIG=2          # runtime: 4 logical cores at 24 GB
python bench_trn2.py --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
    --cores 4 --batch-sizes 1,4,8 --mode mp --out results_trn2_lnc2.json

export NEURON_LOGICAL_NC_CONFIG=1          # runtime: 8 logical cores at 12 GB
python bench_trn2.py --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
    --cores 8 --batch-sizes 1,4,8 --mode mp --out results_trn2_lnc1.json
```

### trn2 ViT-H+ and ViT-7B

`bench_trn2_big.py` selects the model with `--mode`, not `--models`:

```bash
export NEURON_LOGICAL_NC_CONFIG=2
python bench_trn2_big.py --mode vith  --cores 4 --batch-sizes 1,4   # ViT-H+ (trace)
NEURON_RT_NUM_CORES=4 python bench_trn2_big.py --mode vit7b --tp 4  # ViT-7B (TP)
```

ViT-H+ compiles take ~950 s at BS=1 on trn2, so allow time.

### Accuracy on real pretrained weights

Needs an **approved HuggingFace token** -- the `facebook/dinov3-*` checkpoints are
manually gated. Request access on the model page first, then:

```bash
mkdir -p ~/.cache/huggingface && echo "$HF_TOKEN" > ~/.cache/huggingface/token
pip install 'transformers>=4.57'
python validate_real_weights.py --models vit_s,vit_b,vit_l --dtypes fp32,bf16 --n 64
```

The checkpoints ship in `transformers` format, which **`torch_neuronx.trace()`
miscompiles** (cos 0.47-0.62). `hf_to_repo_weights.py` remaps them into the reference-repo
architecture and self-verifies on CPU -- it aborts unless HF-vs-repo agreement is ~1.0.

## Methodology notes -- read these before comparing your numbers

These are not incidental. Each one produced a materially wrong result during this work.

1. **Use independent worker processes, not `torch_neuronx.DataParallel`.** `DataParallel`
   pins `num_workers = 2` regardless of core count, costing up to 5.18x on 12 cores. All
   tables here use one process per core, pinned with `NEURON_RT_VISIBLE_CORES`. If you do
   use `DataParallel`, set `model_dp.num_workers = num_cores` after construction.

2. **One worker per NeuronCore. No oversubscription.** A NeuronCore is single-tenant; extra
   processes fail with `The PyTorch Neuron Runtime could not be initialized`.

3. **Never compile and measure at the same time on a small host.** On a 15 GB inf2.xlarge
   this drove 10.7 GB into swap and corrupted the numbers -- one config read 147.6 img/s
   against ~310 for its neighbours. `bench_bf16_default.py` compiles everything in a first
   phase, then measures. Prefer that pattern.

4. **A loaded traced model pins its core for the process lifetime.** `gc.collect()` does not
   release it; only process exit does. That is why the scripts fan out to subprocesses and
   sleep between phases.

5. **Repeat the baseline until it converges.** A cold baseline against a warm variant
   manufactures gains. One ViT-L BS=1 baseline climbed 52.0 -> 62.4 -> 68.4 img/s across
   three back-to-back runs; comparing against the first produced a fabricated +32%. Always
   take best-of-N, and treat a variant with much lower variance than its baseline as a
   measurement smell.

6. **Keep `inline_weights=True`** (the default). Disabling it costs 6.8x on inf2 -- weights
   are re-fed from the host on every call.

7. **Optimal batch size depends on dtype as well as model.** BF16 ViT has a sharp device-side
   cliff at BS=8 (ViT-L 381.8 -> 72.1 img/s); FP32 does not. Sweep batch size **per dtype**.
   A recommendation measured in one dtype does not transfer to the other.

8. **`torch_neuronx` must be imported before `torch.jit.load` of a saved traced model.** It
   registers the `__torch__.torch.classes.neuron.Model` custom class the saved NEFF
   references; without it every load fails with a TorchScript class-resolution error.

9. **Accuracy-gate every config, not just one.** Phase 2 of `sweep_best.py` checks all of
   them. A batch-dependent miscompile would otherwise slip through a spot check.

## Expected run times

| Scope | Approx wall clock |
|---|---|
| `bench_max_throughput.py`, 5 models x 4 batches, 2 cores | ~2-3 h |
| Same on 12 cores | ~3-4 h |
| `bench_inf2.py` full sweep (includes ViT-H+, ~330 s compile) | ~1.5 h |
| `bench_trn2.py` one LNC mode | ~2 h |
| `validate_real_weights.py` 3 models x 2 dtypes | ~40 min |

ViT-H+ dominates compile time (330 s on inf2.24xlarge, ~950 s on trn2). ViT-7B TP compiles
are fast (6-9 s per rank) because each rank compiles a smaller graph.
