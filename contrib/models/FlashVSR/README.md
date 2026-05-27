# Contrib Model: FlashVSR

Video super-resolution (4x upscaling) on AWS Trainium using a streaming DiT architecture with NKI tiled flash attention.

## Model Information

- **HuggingFace ID:** `JunhaoZhuang/FlashVSR-v1.1`
- **Model Type:** Video super-resolution DiT (Denoising Diffusion Transformer)
- **Parameters:** ~1.3B (BF16) DiT + 288M LQ Projection + 45M TCDecoder
- **Architecture:** 30-layer DiT with factored 3D RoPE, LCSA self-attention, text cross-attention, AdaLN modulation, QK-norm with DistributedRMSNorm
- **Base Model:** Wan 2.1 1.3B (dim=1536, 12 heads, head_dim=128)
- **License:** Check HuggingFace model card

## Validation Results

**Validated:** 2026-05-26
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores)
**SDK:** Neuron SDK 2.29.1, PyTorch 2.9, NKI 0.3.0

### Benchmark Results

| Metric | Value |
|--------|-------|
| End-to-end throughput | **10.3 FPS** (768x1280 output, 85 frames) |
| Total DiT time | 5.0s (1 first chunk + 8 stream chunks) |
| Total TCDecoder time (NxDI, co-resident) | 2.4s (22 calls × 89ms, HBM state persistence) |
| LQ Projection | 0.86s (single pass, all frames) |
| Model loading | DiT 40s + TCDecoder 1.8s (one-time startup) |

### Accuracy Validation

| Metric | Value |
|--------|-------|
| DiT neuron_allclose vs CPU (rtol=0.05, atol=0.1) | PASS |
| DiT max_rel_error | 0.025 |
| DiT cosine similarity | 0.9997 |
| DiT per-chunk latency (first chunk, f=6) | ~1720 ms |
| DiT per-chunk latency (stream, f=2) | ~410 ms |
| Full pipeline visual quality | Matches reference implementation (DMD single-step) |

## Usage

### Recommended: Use the Notebook

The easiest way to reproduce results is the end-to-end notebook at
`notebooks/tcdecoder_benchmark.ipynb`. It includes all compilation, loading,
inference, and color correction steps with expected outputs saved inline.

A sample output video is provided at `notebooks/output_sample.mp4` for visual
comparison. If your output looks washed out or blurry, see **Troubleshooting** below.

### Quick Start (trn2.3xlarge, SDK 2.29.1+)

```bash
# 1. Activate the NxDI venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# 2. Download weights
python -m src.download_weights --output-dir ~/FlashVSR-v1.1

# 3. Compile all models (one-time, ~30 min)
python -m src.pipeline compile \
    --weights-dir ~/FlashVSR-v1.1 \
    --output-dir ~/compiled \
    --height 768 --width 1280 --tp-degree 4

# 4. Run the notebook
jupyter nbconvert --to notebook --execute \
    notebooks/tcdecoder_benchmark.ipynb \
    --output tcdecoder_benchmark_executed.ipynb
```

### Programmatic API

```python
from src.pipeline import compile_pipeline, load_pipeline, run_inference

# Step 1: Compile models (one-time per resolution)
compile_pipeline(
    weights_dir="/path/to/FlashVSR-v1.1",
    output_dir="/path/to/compiled",
    height=768,
    width=1280,
    tp_degree=4,
)

# Step 2: Load compiled pipeline
pipeline = load_pipeline(
    compiled_dir="/path/to/compiled",
    weights_dir="/path/to/FlashVSR-v1.1",
    prompt_path="/path/to/FlashVSR-v1.1/posi_prompt.pth",
    tp_degree=4,
)

# Step 3: Run inference
output_path = run_inference(
    pipeline,
    input_video="/path/to/input.mp4",
    output_dir="/path/to/output",
    scale=4,
)
```

## Pipeline Architecture

FlashVSR has three separately compiled Neuron components, all co-resident in HBM:

| Component | Compilation Method | TP Degree | Role |
|-----------|-------------------|-----------|------|
| DiT (first chunk) | NxDI ModelBuilder | TP=4 | Denoising, f=6 latent frames |
| DiT (stream chunk) | NxDI ModelBuilder | TP=4 | Denoising, f=2 latent frames |
| LQ Projection | torch_neuronx.trace | TP=1 | Generates conditioning tokens |
| TCDecoder | NxDI ModelBuilder | TP=4 | Latent-to-RGB (HBM state persistence) |

All models are loaded at startup and remain co-resident in HBM (total ~15 GB out of 96 GB available on trn2.3xlarge). This eliminates model transition overhead between pipeline stages.

The streaming architecture processes video in chunks: first chunk (6 latent frames = 24 output frames) followed by overlapping stream chunks (2 latent frames = 8 output frames each).

## Key Technical Details

- **NKI Flash Attention:** Uses `attention_cte` from nkilib -- tiles attention computation in SRAM, never materializes the full S*S attention matrix in HBM. Enables 23040-token sequences on trn2.3xlarge.
- **DistributedRMSNorm:** QK-norm with all-reduce across TP ranks for global variance computation. Essential for accuracy at TP>1.
- **Co-resident HBM models:** DiT (7.5 GB × 2) + TCDecoder (378 MB) all loaded simultaneously in 96 GB HBM. Eliminates model swap overhead between pipeline stages.
- **TCDecoder HBM State Persistence:** Uses `input_output_aliases` to keep 9 MemBlock states in device HBM between sequential calls. No PCIe state transfer per frame. Output reshaped from `(4, 3, H, W)` to `(1, 12, H, W)` inside the NEFF to prevent TP from sharding the temporal dimension across ranks.
- **Phase 2 LCSA (optional):** Block-sparse Locality-Constrained Sparse Attention behind `USE_BLOCK_SPARSE_LCSA` toggle. Generates per-layer sparse masks inside the traced graph via topk + index_select. Requires trn2.48xlarge with TP=16.
- **Single-step DMD:** FlashVSR-v1.1 uses Distribution Matching Distillation for single-step denoising (timestep=1000).

## Compatibility Matrix

| Instance/Config | SDK 2.30 | SDK 2.29.1 | SDK 2.29 | SDK 2.28 |
|-----------------|----------|------------|----------|----------|
| trn2.3xlarge, TP=4, LNC=2 | **VALIDATED (12.6 FPS)** | VALIDATED (10.3 FPS) | VALIDATED (8.3 FPS) | Not tested |

**SDK 2.30 gives a 22% DiT speedup** (f=2 stream: 416ms → 325ms) with zero code changes — just recompile on `Deep Learning AMI Neuron (Ubuntu 24.04) 20260522`.

## Multi-Bucket Streaming (Tested, Not Recommended)

The codebase includes support for multi-bucket DiT streaming via `FLASHVSR_STREAM_BUCKETS=8,4,2`.
This compiles f=8, f=4, and f=2 NEFFs co-resident in HBM and uses a greedy scheduler to
minimize the number of DiT calls for long videos.

**Benchmark result: larger buckets are SLOWER for this model.**

| Bucket | Per-Frame Latency | vs f=2 |
|--------|------------------|--------|
| f=2    | 162.5 ms         | baseline |
| f=4    | 197.2 ms         | 21% slower |
| f=8    | 287.7 ms         | 77% slower |

FlashVSR uses full temporal self-attention (not tiled/windowed), so attention cost scales
super-linearly with frame count. The per-call overhead savings are overwhelmed by quadratic
attention growth. **Use the default `FLASHVSR_STREAM_BUCKETS=2` for optimal throughput.**

The multi-bucket code is retained as a reference implementation for models with windowed/tiled
attention where larger batches DO reduce per-frame cost.

## Example Checkpoints

* [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)

## Testing Instructions

```bash
# Run DiT accuracy test (neuron_allclose vs CPU reference)
pytest test/integration/test_dit_accuracy.py -v

# Run full pipeline E2E test (PSNR validation)
pytest test/integration/test_pipeline_e2e.py -v
```

## Known Issues

- **Resolution constraint:** Input video must produce output dimensions divisible by 128 (e.g., 768x1280). Other resolutions require recompilation.
- **Phase 2 LCSA:** Block-sparse attention requires trn2.48xlarge with TP=16 (not available on trn2.3xlarge). Production uses Phase 1 dense attention.
- **TCDecoder temporal recurrence:** Each frame must be processed serially due to MemBlock temporal dependencies. The NxDI HBM state persistence approach minimizes this cost (89ms/call vs 237ms with PCIe state transfer).
- **Text embedding:** Uses a pre-computed positive prompt embedding (`posi_prompt.pth`). Custom prompts require running the T5 text encoder separately.

## Troubleshooting: Poor Output Quality

If the upscaled video appears blurry, has color drift, or looks worse than expected:

1. **Missing LQ projection conditioning.** The LQ projection provides per-token guidance
   from the low-quality input. Without it, the DiT produces generic denoised output without
   content fidelity. Verify that `lq_proj_model` is loaded and `all_lq_tokens` is passed
   to each DiT chunk via the `lq_residual_0` parameter.

2. **Missing color correction.** The DiT + TCDecoder output has correct structure but may
   have color drift from BF16 quantization. The AdaIN color correction step (Stage 4 in the
   notebook) aligns color distribution with the LQ reference. Without it, output may appear
   washed out or have shifted hues.

3. **TCDecoder state not reset.** The TCDecoder uses 9 stateful MemBlocks that persist across
   calls. You **must** call `tcd_app.reset_states()` before processing a new video. Leftover
   state from a previous video or warmup will corrupt output.

4. **Frame count mismatch.** The input must be in `8n+1` format (e.g., 89 frames). The number
   of LQ conditioning frames for TCDecoder must equal `latents_T * 4` (the pixel shuffle
   temporal factor). Using the wrong frame count leads to misaligned conditioning.

5. **Wrong prompt embedding.** The model expects the pre-computed `posi_prompt.pth` from the
   FlashVSR-v1.1 weights. Using a different or missing prompt drastically reduces quality.

**Reference output:** Compare against `notebooks/output_sample.mp4` (85 frames, 768x1280,
generated from the included test video `example0_cropped_192x320.mp4`).

## Maintainer

Jim Burtoft
