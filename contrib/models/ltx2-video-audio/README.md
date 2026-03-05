# LTX-2 on AWS Trainium (NxDI)

Port of the [Lightricks LTX-2](https://huggingface.co/Lightricks/LTX-2) 19B-parameter audio-video diffusion model to AWS Neuron (Trainium/Inferentia) using the NxDI (NeuronX Distributed Inference) framework.

## Architecture

LTX-2 generates synchronized video + audio from text prompts. The model has three main components:

| Component | Parameters | Runs on | Notes |
|-----------|-----------|---------|-------|
| Text encoder (Gemma 3-12B) | 12B | **Neuron TP=4** | Compiled with custom encoder |
| DiT transformer (48 blocks) | ~6B | **Neuron TP=4** | The denoising bottleneck |
| Video + Audio VAE decoders | ~1B | CPU | Run once per generation |

Both the text encoder and DiT transformer backbone are compiled for Neuron. They coexist on the same 4 NeuronCores (TP=4) and execute sequentially: text encoding then denoising.

## Performance

Tested on trn2.3xlarge (1 NeuronDevice, 4 logical NeuronCores with LNC=2):

| Metric | Neuron (trn2, spot) | GPU (g5.12xlarge) |
|--------|--------------------|--------------------|
| Generation (CFG, 8 steps) | ~66s | ~48s |
| Hardware cost/hr | ~$0.90 | ~$5.67 |
| Cost per generation | ~$0.016 | ~$0.076 |

The Neuron pipeline produces nearly identical output to the GPU at **~5x lower cost per generation**.

## Requirements

- **Instance**: trn2.3xlarge (sa-east-1 or ap-southeast-4)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 24.04) 20260227
- **Python env**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`
- **Diffusers**: 0.37.0.dev0 (install from git main)
- **Disk**: ~100GB for model weights + compilation cache

## Quick Start

### 1. Instance Setup

```bash
# SSH into your trn2.3xlarge instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Install diffusers from git (LTX-2 requires dev version)
pip install git+https://github.com/huggingface/diffusers.git
pip install imageio imageio-ffmpeg

# Upload this package
# scp -r ltx2-neuron/ ubuntu@<instance-ip>:/home/ubuntu/
```

### 2. Compile Models (First Run Only)

```bash
cd /home/ubuntu/ltx2-neuron/src

# Compile the DiT transformer backbone (~2 minutes)
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  python compile_gemma3.py

# Pre-shard Gemma3 weights for fast loading (~2 minutes)
python shard_gemma3_weights.py
```

The DiT backbone is compiled automatically on first use by the E2E script.

### 3. Generate Video + Audio

```bash
cd /home/ubuntu/ltx2-neuron/examples

NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  python neuron_e2e.py
```

### 4. Notebook

For an interactive walkthrough, use the Jupyter notebook:

```bash
cd /home/ubuntu/ltx2-neuron/notebooks
jupyter notebook ltx2_neuron_inference.ipynb
```

## File Structure

```
ltx2-neuron/
├── README.md
├── src/                              # Core NxDI package
│   ├── __init__.py
│   ├── modeling_ltx2.py              # DiT backbone: TP sharding, SPMD, config
│   ├── modeling_gemma3_encoder.py    # Gemma3 text encoder for Neuron
│   ├── application.py                # NeuronLTX2Application orchestrator
│   ├── pipeline.py                   # NeuronTransformerWrapper (CFG batch-split)
│   ├── compile_gemma3.py             # Gemma3 encoder compilation script
│   ├── shard_gemma3_weights.py       # Pre-shard Gemma3 weights to disk
│   └── generate_ltx2.py             # CLI entry point with argument parsing
├── notebooks/
│   └── ltx2_neuron_inference.ipynb   # Step-by-step compile + generate notebook
├── examples/
│   ├── neuron_e2e.py                 # Full E2E: load + generate (both on Neuron)
│   └── gpu_generate.py              # GPU reference generation script
└── samples/
    ├── neuron/                       # Output from Neuron (trn2.3xlarge)
    │   ├── frame_0000.png
    │   ├── frame_0012.png
    │   └── frame_0024.png
    └── gpu/                          # Output from GPU (g5.12xlarge)
        ├── frame_0000.png
        ├── frame_0012.png
        └── frame_0024.png
```

## Generation Settings

Both Neuron and GPU samples were generated with identical settings:

```python
prompt = ("A golden retriever puppy runs across a sunny green meadow, "
          "its ears flapping in the wind. The camera follows from a low angle. "
          "Birds chirp in the background.")
height = 384
width = 512
num_frames = 25
num_inference_steps = 8
guidance_scale = 4.0       # CFG (pipeline default)
max_sequence_length = 1024  # Pipeline default
seed = 42
model = "Lightricks/LTX-2"
```

## Technical Details

### CFG (Classifier-Free Guidance) on Neuron

With `guidance_scale=4.0`, the pipeline runs the text encoder twice (positive + negative prompt) and the DiT backbone twice per denoising step (uncond + cond). Since the Neuron backbone is compiled for `batch_size=1`, the `NeuronTransformerWrapper` handles CFG by splitting the batch and calling the backbone twice per step, then concatenating results.

### 22-Input Transformer Signature

The compiled Neuron DiT model takes 22 positional tensor arguments, all preprocessed on CPU:

| # | Input | Shape | Description |
|---|-------|-------|-------------|
| 1 | hidden_states | (1, 768, 4096) | Video latents after proj_in |
| 2 | audio_hidden_states | (1, 26, 2048) | Audio latents after audio_proj_in |
| 3 | encoder_hidden_states | (1, 1024, 4096) | Text embeddings for video |
| 4 | audio_encoder_hidden_states | (1, 1024, 2048) | Text embeddings for audio |
| 5-8 | temb, temb_audio, embedded_ts, audio_embedded_ts | various | Time embeddings |
| 9-12 | cross-attn scale/shift/gate | various | Cross-attention conditioning |
| 13-16 | video/audio rotary cos/sin | various | Self-attention RoPE |
| 17-20 | cross-attn video/audio rotary cos/sin | various | Cross-attention RoPE |
| 21-22 | encoder/audio_attention_mask | (1, 1, 1024) | Additive attention bias |

### Critical Implementation Details

- **SPMDRank RoPE**: Uses NxD `SPMDRank` module (not a Python int) for per-rank RoPE slicing. A Python int gets baked as constant 0 during SPMD XLA tracing, causing all ranks to apply the same RoPE shard.

- **DistributedRMSNorm**: QK-norm uses all-reduce across TP ranks to compute global variance.

- **BMM SDPA**: Replaces `torch.nn.functional.scaled_dot_product_attention` with explicit BMM operations for Neuron XLA compatibility.

- **RoPE BF16 cast**: RoPE modules return float32 for numerical precision. Tensors must be cast to bfloat16 before passing to the compiled Neuron model.

- **Additive attention mask**: Binary masks are converted to additive bias format (-10000 for masked positions) before compilation.

- **Pre-sharded weights**: Gemma3 weights are pre-sharded to disk (~5.5 GB per rank) using `.contiguous().clone()` to avoid serializing the full unsharded storage backing sliced tensors.

### Compiler Flags

```
--model-type=transformer -O1 --auto-cast=none --enable-saturate-infinity
--enable-mixed-precision-accumulation --lnc=2
--tensorizer-options='--enable-ccop-compute-overlap'
```

Environment variables:
```bash
NEURON_FUSE_SOFTMAX=1
NEURON_CUSTOM_SILU=1
NEURON_RT_STOCHASTIC_ROUNDING_EN=0
```
