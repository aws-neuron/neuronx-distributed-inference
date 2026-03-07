# Contrib Model: ZUNA

EEG foundation model inference on AWS Inferentia2 using `torch_neuronx.trace()`.

## Model Information

- **HuggingFace ID:** `Zyphra/ZUNA`
- **Model Type:** Masked diffusion autoencoder (encoder-decoder)
- **Parameters:** ~382M (FP32)
- **Architecture:** 16-layer encoder (self-attention + SwiGLU + register interleaving + MMD bottleneck), 16-layer decoder (cross-attention + self-attention + AdaRMSNorm timestep conditioning), 4D axial RoPE, rectified flow / flow matching with Euler ODE solver
- **License:** Check HuggingFace model card

## Architecture Details

ZUNA is a masked diffusion autoencoder for EEG signals. The encoder compresses multi-channel EEG input into a bottleneck latent space, and the decoder reconstructs it via a 50-step rectified flow diffusion process conditioned on the encoder output and a timestep embedding.

Key architectural features:
- **4D axial RoPE** over (x, y, z, t_coarse) spatial coordinates
- **Register interleaving** in the encoder (doubles sequence length internally)
- **AdaRMSNorm** for timestep conditioning in the decoder
- **MMD bottleneck** between encoder and decoder
- **Rectified flow** with Euler ODE solver (50 steps default)

This model uses `torch_neuronx.trace()` (not NxD Inference) since it is an encoder-decoder diffusion model rather than an autoregressive LLM. The encoder is traced once per sample, and the decoder is traced and called 50 times (once per diffusion step).

### Neuron Porting Challenges

**flex_attention incompatibility**: ZUNA uses PyTorch's `flex_attention` API (`create_block_mask`, `BlockMask`, `_mask_mod_signature`, `noop_mask`) which is not supported on Neuron's XLA device. The solution is to:
1. Patch the `torch.nn.attention.flex_attention` module with dummy symbols before importing ZUNA
2. Monkey-patch all 32 `Attention` and 16 `CrossAttention` modules to use `F.scaled_dot_product_attention`
3. Replace encoder and decoder forward methods to skip `create_document_mask` calls

Since `sliding_window=65536` far exceeds the inference sequence length (~100), full attention without masking is mathematically correct for single-sample inference.

**NKI kernel research**: The Neuron NKI library provides a `flash_fwd` kernel that the compiler automatically deploys for SDPA patterns, but it does not support the `mask_mod`/`score_mod` callbacks that flex_attention uses. This confirms the SDPA replacement strategy is the correct approach.

## Validation Results

**Validated:** 2026-03-05
**Instance:** inf2.xlarge (1 Inferentia2 chip, 2 NeuronCores)
**SDK:** Neuron SDK 2.27, PyTorch 2.9

### Benchmark Results

| Configuration | Pipeline Latency | Throughput | Speedup |
|--------------|-----------------|-----------|---------|
| Neuron single core (auto-cast=matmult) | 102 ms | 9.80 samples/sec | 106x vs CPU |
| Neuron DataParallel (2 cores) | 123 ms | 16.29 samples/sec | 1.66x vs single |
| GPU A10G (g5.xlarge) | 2,854 ms | 0.35 samples/sec | baseline |
| CPU (inf2 host) | 10,856 ms | 0.09 samples/sec | baseline |

**Neuron is 28x faster than GPU (A10G) and 106x faster than CPU.**

| Diffusion Steps | Neuron Latency | Throughput |
|----------------|---------------|-----------|
| 10 steps | 21.7 ms | 46.0 samples/sec |
| 25 steps | 51.9 ms | 19.3 samples/sec |
| 50 steps | 102.1 ms | 9.80 samples/sec |

### Accuracy Validation

Accuracy measured over 50 random seeds, comparing full 50-step diffusion output (CPU FP32 vs Neuron):

| Metric | auto-cast=matmult | auto-cast=none (pure FP32) |
|--------|------------------|---------------------------|
| Cosine similarity (mean) | 0.990 | 1.000000 |
| Cosine similarity (min) | 0.937 | 1.000000 |
| MSE (mean) | 2.37e-04 | ~0 |
| Channel correlation (mean) | 0.990 | 1.000 |
| PSD correlation (mean) | 0.988 | 1.000 |

#### auto-cast Impact Analysis

All accuracy loss comes exclusively from `--auto-cast=matmult` BF16 conversion, **not** from the Neuron compiler or SDPA replacement. This was confirmed by running the same 5 test seeds with both compiler settings:

| Seed | auto-cast=matmult | auto-cast=none |
|------|------------------|----------------|
| 0 | 0.9817 | 1.000000 |
| 7 | 0.9655 | 1.000000 |
| 13 | 0.9999 | 1.000000 |
| 42 | 0.9987 | 1.000000 |
| 99 | 0.9993 | 1.000000 |

Seed 7 is a worst-case outlier (cosine=0.966 with matmult) that becomes **perfect** (1.000000) without auto-cast. This demonstrates that the Neuron compiler and the SDPA attention replacement introduce zero numerical error. The BF16 matmul conversion interacts with certain random initialization patterns, and the small per-step error compounds across 50 diffusion steps -- but remains bounded (peaks at step 30, then self-corrects by step 50).

**Recommendation**: Use `--auto-cast=matmult` (default) for production. The 0.990 mean cosine similarity is excellent for EEG applications, and it delivers 2x throughput (102ms vs 212ms). Use `--auto-cast=none` only when bit-exact CPU reproduction is required.

## Usage

```bash
# On an inf2 or trn2 instance with DLAMI
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
pip install zuna
```

### Quick Start (Python API)

```python
from src.modeling_zuna import ZUNANeuronModel, make_synthetic_input

# Load, patch, and compile
model = ZUNANeuronModel()
model.load_and_patch()
model.compile(seqlen=50, save_dir="/home/ubuntu/neuron_models/ZUNA")

# Run inference
encoder_input, tok_idx = make_synthetic_input(seqlen=50, seed=42)
z, timing = model.infer(encoder_input, tok_idx, sample_steps=50)

print(f"Output shape: {z.shape}")           # [1, 50, 32]
print(f"Pipeline latency: {timing['total_ms']:.1f} ms")
print(f"Throughput: {1000.0 / timing['total_ms']:.2f} samples/sec")
```

### Detailed Usage

```python
import torch
from src.modeling_zuna import (
    load_model, patch_model_for_neuron,
    EncoderWrapper, DecoderWrapper,
    make_synthetic_input, run_diffusion,
)
import torch_neuronx

# Step 1: Load and patch
model, model_args = load_model(device="cpu")
model = patch_model_for_neuron(model)

# Step 2: Compile encoder
encoder_wrapper = EncoderWrapper(model.encoder)
encoder_wrapper.eval()
example_input, example_tok_idx = make_synthetic_input(seqlen=50)

encoder_neuron = torch_neuronx.trace(
    encoder_wrapper,
    (example_input, example_tok_idx),
    compiler_args=["--auto-cast", "matmult", "-O2"],
    inline_weights_to_neff=True,
)

# Step 3: Compile decoder
with torch.no_grad():
    enc_out_example = encoder_neuron(example_input, example_tok_idx)

decoder_wrapper = DecoderWrapper(model.decoder)
decoder_wrapper.eval()
example_z = torch.randn(1, 50, 32)
example_enc = torch.randn(1, enc_out_example.shape[1], enc_out_example.shape[2])
example_t = torch.tensor([[[0.5]]])

decoder_neuron = torch_neuronx.trace(
    decoder_wrapper,
    (example_z, example_enc, example_t, example_tok_idx),
    compiler_args=["--auto-cast", "matmult", "-O2"],
    inline_weights_to_neff=True,
)

# Step 4: Run diffusion
encoder_input, tok_idx = make_synthetic_input(seqlen=50, seed=42)
sigma = float(getattr(model, "global_sigma", 0.1))

z, timing = run_diffusion(
    encoder_fn=lambda tv, ti: encoder_neuron(tv, ti),
    decoder_fn=lambda z, e, t, ti: decoder_neuron(z, e, t, ti),
    encoder_input=encoder_input,
    tok_idx=tok_idx,
    sample_steps=50,
    sigma=sigma,
    noise_seed=42,
)
```

### DataParallel (2 NeuronCores)

```python
# After compiling single-core models:
encoder_dp = torch_neuronx.DataParallel(encoder_neuron, device_ids=[0, 1], dim=0)
decoder_dp = torch_neuronx.DataParallel(decoder_neuron, device_ids=[0, 1], dim=0)

# Batch=2 for DP (one sample per core)
ei_batch = encoder_input.repeat(2, 1, 1)
ti_batch = tok_idx.repeat(2, 1, 1)
# ... run diffusion loop with DP models
```

## Compatibility Matrix

| Instance/Version | SDK 2.27 | SDK 2.28 |
|------------------|----------|----------|
| inf2.xlarge | VALIDATED | Not tested |
| trn2.3xlarge | Not tested | Not tested |

## Example Checkpoints

* [Zyphra/ZUNA](https://huggingface.co/Zyphra/ZUNA)

## Testing Instructions

```bash
# On a Neuron instance (inf2.xlarge or larger)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
pip install zuna pytest

# Run with pytest
pytest contrib/models/ZUNA/test/integration/test_model.py --capture=tee-sys -v

# Or run standalone
python contrib/models/ZUNA/test/integration/test_model.py
```

The test suite includes 46 tests across 7 classes:

**Unit tests** (`test/unit/test_patching.py`) -- CPU-only, no Neuron hardware required:
- **TestFlexAttentionPatch** (5 tests): Verifies all dummy flex_attention symbols are installed
- **TestModelLoading** (4 tests): Model loads from HuggingFace, ~382M params, eval mode, correct config
- **TestPatching** (2 tests): All 32 Attention + 16 CrossAttention modules patched to SDPA
- **TestSyntheticInput** (4 tests): Input shapes, dtype, determinism, seed independence
- **TestEncoderWrapper** (3 tests): Output shape, finiteness, 2D tok_idx auto-unsqueeze
- **TestDecoderWrapper** (3 tests): Output shape, finiteness, timestep sensitivity
- **TestDiffusionLoop** (5 tests): Output shape, finiteness, determinism, seed/step sensitivity

**Integration tests** (`test/integration/test_model.py`) -- requires Neuron hardware:
- **TestModelLoads** (5 tests): Encoder/decoder compile, run, full 50-step pipeline
- **TestAccuracy** (6 tests): Cosine similarity across 5 seeds with `--auto-cast=matmult`, MSE bounds
- **TestNoAutocast** (5 tests): Compiles with `--auto-cast=none` and verifies perfect cosine similarity (>=0.999), confirming all accuracy loss comes from BF16 matmul conversion
- **TestDataParallel** (2 tests): Multi-core execution and speedup validation
- **TestPerformance** (2 tests): Throughput and latency threshold assertions

## Known Issues

1. **flex_attention not supported on Neuron**: All flex_attention symbols must be patched before importing ZUNA. The patching code is included in both `src/modeling_zuna.py` and `test/integration/test_model.py`.

2. **tok_idx must be 3D**: The `tok_idx` tensor must be `[1, seqlen, 4]` (with batch dimension), not `[seqlen, 4]`. The encoder's `repeat_interleave(repeats=2, dim=1)` operates on dim=1 and requires the batch dimension.

3. **auto-cast=matmult accuracy tradeoff**: With `--auto-cast=matmult`, cosine similarity is ~0.990 mean (0.937 min over 50 seeds). Outlier seeds (e.g., seed 7 at 0.966) become perfect (1.000000) with `--auto-cast=none`, confirming all error originates from BF16 matmul conversion. Use `--auto-cast=none` for bit-exact results at ~2x latency cost (212ms vs 102ms).

4. **Fixed sequence length**: Models are traced for a specific sequence length (default 50). Different sequence lengths require recompilation.

## Maintainer
Jim Burtoft
Community contribution

**Last Updated:** 2026-03-05
