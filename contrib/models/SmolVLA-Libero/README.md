# Contrib Model: SmolVLA-Libero

NeuronX Distributed Inference port of `HuggingFaceVLA/smolvla_libero` — a
SmolVLM2-backed flow-matching vision-language-action (VLA) policy fine-tuned
on the LIBERO benchmark. Three compiled subgraphs, maximally on-Neuron,
written in the per-model NxDI structure.

## Model Information

- **HuggingFace ID:** `HuggingFaceVLA/smolvla_libero`
- **Backbone:** `HuggingFaceTB/SmolVLM2-500M-Instruct` (full 32-layer text decoder)
- **Model Type:** Flow-matching VLA (SigLIP vision + SmolLM-style VLM + action expert)
- **Action head:** 32-layer expert with self/cross-attn alternation, 50-step action chunk, 10-step Euler denoising
- **License:** Check HuggingFace model card

## Architecture Details

| Component                                              | Where      | Subgraph |
|--------------------------------------------------------|------------|----------|
| SigLIP vision encoder (12 layers, hidden=768)          | **Neuron** | #1       |
| Pixel-shuffle 4× + connector + scale by sqrt(960)      | **Neuron** | #1       |
| Lang token embed + scale by sqrt(960)                  | **Neuron** | #2       |
| State projection (32 → 960)                            | **Neuron** | #2       |
| VLM 32-layer text decoder (eager GQA, RoPE, RMSNorm)   | **Neuron** | #2       |
| Pad-aware position_ids + 2D attention mask             | **Neuron** | #2 / #3  |
| Action expert: 16× self-attn (concat past KV) layers   | **Neuron** | #3       |
| Action expert: 16× cross-attn (Q from suffix) layers   | **Neuron** | #3       |
| Sinusoidal timestep embedding                          | **Neuron** | #3       |
| Action in/out projections + time MLP                   | **Neuron** | #3       |
| Image preprocessing (resize-with-pad, normalize)       | CPU        | —        |
| Tokenization                                           | CPU        | —        |
| 10-step Euler denoising loop                           | CPU        | —        |

**Deviations from "everything on Neuron":**

1. The 10-step Euler loop runs on CPU. Static-shape compilation cannot host
   a Python `for step in range(N)` as a single graph; the loop body is the
   compiled subgraph. Each step calls NEFF #3 with the updated `noisy_actions`.
2. Tokenization, image flip / resize-with-pad, and state-vector composition
   run on CPU because they are data-loading, not model compute.

**Hardware constraint flagged:** `tp_degree = 1` because
`num_attention_heads = 15` and `num_kv_heads = 5` — neither divides cleanly
into the 4 Neuron cores on `trn3pd98.3xlarge`. The NxDI parallel primitives
(`ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`) are still
used throughout, so this code is portable to instances with divisor-friendly
head counts. On this instance, 3 of 4 cores idle; the model fits comfortably
in one core's HBM with vast headroom.

## Validation Results

**Validated:** 2026-05-06
**Configuration:** TP=1, batch_size=1, bfloat16
**Instance:** trn3pd98.3xlarge
**NxDI:** 2.29 (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`)

### Test Results

| Check                                                        | Result                                       |
|--------------------------------------------------------------|----------------------------------------------|
| Vision NEFF vs HF SmolVLM2 vision (single image)             | cos_sim = 0.99990                            |
| Prefix KV layer 0..31 vs lerobot CPU                         | max abs diff ≤ 0.4 (BF16)                    |
| Full action chunk vs lerobot CPU (matched noise)             | cos_sim = 0.9999, mean abs diff = 0.007      |
| Full action chunk: Neuron vs lerobot CPU (this `test_model.py`) | cos_sim = 0.999921, mean abs diff = 0.0015 |
| Closed-loop LIBERO `libero_object` task 0                    | success                                      |
| End-to-end inference latency (one chunk)                     | warm p50 = 62.5 ms (10 iters)                |

The numerical match against the lerobot CPU reference replicates four
lerobot-specific quirks that aren't in the SmolVLM2 HF config:

1. **`resize_with_pad` pads top+left only** (image lands in the bottom-right
   corner of the 512×512 frame), not centered.
2. **Pad-aware attention**: dynamic 2D mask + cumsum-based position_ids that
   skip padding tokens. A static prefix-LM mask leaks attention into pad-token
   positions.
3. **RoPE max_wavelength = 10000** (lerobot hardcodes this in `apply_rope`);
   the SmolVLM2 HF config says 100000, but lerobot trained the model with 10000.
4. **Image flip** in the LIBERO env (180° rotate, both H and W) per the
   `libero_processor` step in `lerobot.processor.env_processor`.

## Inference Flow

```
images [2 cams × [B, 3, 512, 512]]   lang_token_ids [B, 48]   lang_mask [B, 48]   state [B, 32]
                |                              |                    |                 |
        [Neuron NEFF #1]                       |                    |                 |
        Vision (per camera)                    |                    |                 |
                |                              |                    |                 |
        [B, 128, 960] vision_features          |                    |                 |
                |______________________________|____________________|_________________|
                                                |
                                       [Neuron NEFF #2]
                                       VLM Prefix (32 layers, pad-aware)
                                                |
                                       prefix_keys, prefix_values
                                       [32, B, 177, 5, 64] each
                                                |
                          ┌─────────────────────┴─────────────────────┐
                          │  CPU Euler loop (10 steps)                │
                          │     for t in [1.0, 0.9, ..., 0.1]:        │
                          │         v_t = NEFF#3(x_t, t, K, V, pad)   │
                          │         x_t += dt * v_t                   │
                          └─────────────────────┬─────────────────────┘
                                                |
                                       action_chunk [B, 50, 32]
                                       (first 7 dims used by env)
```

## Source Layout

```
SmolVLA-Libero/
├── README.md
├── src/
│   ├── __init__.py
│   ├── config_constants.py          # All architecture constants from the checkpoint
│   ├── modeling_smolvla.py          # SmolVLAPolicy: orchestrator (compile / load / generate)
│   ├── modeling_smolvla_vision.py   # SigLIP-12L + connector  (NEFF #1)
│   ├── modeling_smolvla_text.py     # VLM 32L prefix + Action expert 32L denoiser  (NEFF #2 + #3)
│   ├── neuron_action_head_base.py   # NeuronDenoisingConfig — ModelWrapper-compatible config shim
│   ├── weight_mapping.py            # HF safetensors -> 3 per-subgraph state dicts
│   └── run_inference.py             # CLI: compile / run / benchmark (synthetic inputs)
└── test/
    ├── __init__.py
    ├── integration/
    │   ├── __init__.py
    │   └── test_model.py            # Smoke + numerical tests against lerobot CPU
    └── unit/
        └── __init__.py
```

We add `_text` because SmolVLA has separate text-prefix and action-expert
subgraphs that the existing per-model layout (e.g. `pixtral/` with
`modeling_pixtral.py` + `modeling_pixtral_vision.py`) does not need.

## Usage

### Setup

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Download checkpoint (one-time)
python -c "from huggingface_hub import snapshot_download; \
    print(snapshot_download(repo_id='HuggingFaceVLA/smolvla_libero'))"
```

### Compile + run via CLI

```bash
cd contrib/models/SmolVLA-Libero/src

# Compile (one-time, ~90s wall clock for 3 NEFFs)
python run_inference.py --action compile \
    --hf-checkpoint /home/ubuntu/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/<hash>/ \
    --neff-dir      /home/ubuntu/smol_vla_neff_libero

# Run inference (synthetic inputs, p50 / p99 latency, NaN check)
python run_inference.py --action run \
    --hf-checkpoint /home/ubuntu/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/<hash>/ \
    --neff-dir      /home/ubuntu/smol_vla_neff_libero
```

### Programmatic

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("contrib/models/SmolVLA-Libero/src")))

from modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy(hf_checkpoint_dir="<path>", tp_degree=1)
policy.load("/home/ubuntu/smol_vla_neff_libero")

# images: list of NUM_CAMERAS tensors, each [B, 3, 512, 512] BF16
# lang_token_ids: [B, 48] INT32
# lang_mask: [B, 48] BOOL  (True = real token, False = pad)
# state: [B, 32] FP32  (already normalized, zero-padded)
action_chunk = policy.generate(images, lang_token_ids, state, lang_mask=lang_mask)
# action_chunk: [B, 50, 32] FP32  (first 7 dims used by LIBERO)
```

## Compatibility Matrix

| Instance / NxDI | 2.29 |
|-----------------|------|
| Trn3            | ✅ Working |
| Trn2            | Not tested |
| Trn1 / Inf2     | Not tested |

## Testing

The integration test compiles the three NEFFs (or reuses an already-compiled
directory), loads them, and runs three checks:

1. **Smoke** — full pipeline returns a finite `[B, 50, 32]` action chunk with
   non-zero variance.
2. **Warm latency** — p50 latency over 5 iterations is under a generous bound
   (1 s; expected ~65 ms on `trn3pd98.3xlarge`).
3. **Neuron vs lerobot CPU parity** (NxDI accuracy check) — loads the
   upstream `lerobot.SmolVLAPolicy` from the same HF checkpoint, runs a
   CPU forward with identical inputs and identical seeded initial noise,
   and asserts cosine similarity ≥ 0.99 and mean abs diff < 0.05 against
   the Neuron action chunk. This is the SmolVLA equivalent of the logit
   validation NxDI uses for CausalLM contrib models — it validates that
   the Neuron port reproduces the reference implementation, not just
   self-consistency. Skipped automatically if `lerobot` is not installed.

```bash
# One-time, point the test at a checkpoint and a NEFF output directory:
export SMOLVLA_CKPT=/home/ubuntu/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/<hash>/
export SMOLVLA_NEFF=/home/ubuntu/smol_vla_neff_libero

# Run
pytest contrib/models/SmolVLA-Libero/test/integration/test_model.py --capture=tee-sys

# Or directly
cd contrib/models/SmolVLA-Libero
python test/integration/test_model.py
```

The first invocation compiles the three NEFFs into `$SMOLVLA_NEFF` (~90 s).
Subsequent runs reuse the compiled artifacts and only re-load + execute.

## Example Checkpoints

- [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
  — used for the validation results above

## Maintainer

Community contribution.

**Last Updated:** 2026-05-06
