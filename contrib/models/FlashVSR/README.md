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

**Validated:** 2026-05-18
**Instance:** trn2.3xlarge (LNC=2, 4 logical NeuronCores)
**SDK:** Neuron SDK 2.29, PyTorch 2.9, NKI 0.3.0

### Benchmark Results

| Metric | Value |
|--------|-------|
| End-to-end throughput | 8.27 FPS (768x1280 output) |
| DiT latency (first chunk, f=6) | ~95 ms |
| DiT latency (stream chunk, f=2) | ~35 ms |
| TCDecoder latency (sequential, per frame) | ~45 ms |
| Pipeline overhead (LQ proj + color correction) | ~200 ms total |

### Accuracy Validation

| Metric | Value |
|--------|-------|
| DiT neuron_allclose vs CPU (rtol=0.01) | PASS |
| TCDecoder PSNR vs CPU reference | >45 dB |
| Full pipeline visual quality | Matches GPU reference (DMD single-step) |

## Usage

```python
from src.pipeline import compile_pipeline, load_pipeline, run_inference

# Step 1: Download weights (one-time)
# python -m src.download_weights --output-dir /path/to/FlashVSR-v1.1

# Step 2: Compile models (one-time per resolution)
compile_pipeline(
    weights_dir="/path/to/FlashVSR-v1.1",
    output_dir="/path/to/compiled",
    height=768,
    width=1280,
    tp_degree=4,
)

# Step 3: Load compiled pipeline
pipeline = load_pipeline(
    compiled_dir="/path/to/compiled",
    weights_dir="/path/to/FlashVSR-v1.1",
    prompt_path="/path/to/FlashVSR-v1.1/posi_prompt.pth",
    tp_degree=4,
    tcdecoder_path="/path/to/compiled/tcdecoder_seq.pt",
    lq_proj_path="/path/to/compiled/lq_proj.pt",
)

# Step 4: Run inference
output_path = run_inference(
    pipeline,
    input_video="/path/to/input.mp4",
    output_dir="/path/to/output",
    scale=4,
)
```

## Pipeline Architecture

FlashVSR has three separately compiled Neuron components:

| Component | Compilation Method | TP Degree | Role |
|-----------|-------------------|-----------|------|
| DiT (first chunk) | NxDI ModelBuilder | TP=4 | Denoising, f=6 latent frames |
| DiT (stream chunk) | NxDI ModelBuilder | TP=4 | Denoising, f=2 latent frames |
| LQ Projection | torch_neuronx.trace | TP=1 | Generates conditioning tokens |
| TCDecoder | torch_neuronx.trace | TP=1 | Latent-to-RGB (sequential mode) |

The streaming architecture processes video in chunks: first chunk (6 latent frames = 24 output frames) followed by overlapping stream chunks (2 latent frames = 8 output frames each).

## Key Technical Details

- **NKI Flash Attention:** Uses `attention_cte` from nkilib -- tiles attention computation in SRAM, never materializes the full S*S attention matrix in HBM. Enables 23040-token sequences on trn2.3xlarge.
- **DistributedRMSNorm:** QK-norm with all-reduce across TP ranks for global variance computation. Essential for accuracy at TP>1.
- **Phase 2 LCSA (optional):** Block-sparse Locality-Constrained Sparse Attention behind `USE_BLOCK_SPARSE_LCSA` toggle. Generates per-layer sparse masks inside the traced graph via topk + index_select. Requires trn2.48xlarge with TP=16.
- **Single-step DMD:** FlashVSR-v1.1 uses Distribution Matching Distillation for single-step denoising (timestep=1000).

## Compatibility Matrix

| Instance/Config | SDK 2.29 | SDK 2.28 |
|-----------------|----------|----------|
| trn2.3xlarge, TP=4, LNC=2 | VALIDATED | Not tested |

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
- **TCDecoder sequential mode:** Each frame must be processed serially due to MemBlock temporal recurrence. Cannot be parallelized without quality loss.
- **Text embedding:** Uses a pre-computed positive prompt embedding (`posi_prompt.pth`). Custom prompts require running the T5 text encoder separately.

## Maintainer

Jim Burtoft
