# USAGE — HunyuanVideo-1.5 Neuron Port

## What This Is

This directory contains a complete port of HunyuanVideo-1.5 (8.3B parameter text-to-video model) for AWS Neuron hardware. All model components run on NeuronCores — there is no CPU fallback.

Given a text prompt, the pipeline generates 33 video frames at 480×640 resolution.

## Directory Structure

```
├── run_inference.py              # Main entry point — runs end-to-end inference
├── compile_all.py                           # Compiles all models into Neuron NEFFs
├── modeling_hunyuan_video15_transformer.py   # Transformer backbone (8.3B, TP=2)
├── modeling_hunyuan_video15_vae.py           # VAE decoder (871M, 24 shards)
├── modeling_hunyuan_video15_text.py          # ByT5 encoder, token refiner, token reorder
├── modeling_qwen2vl_encoder.py              # Qwen2.5-VL text encoder (7B, TP=2)
├── neuron_full_backbone.py                  # Transformer block implementation (imported by transformer modeling)
├── compiled_transformer/                    # Pre-compiled transformer NEFF (TP=2)
│   ├── model.pt
│   ├── neuron_config.json
│   └── weights/
├── compiled_qwen2vl/                        # Pre-compiled Qwen2.5-VL NEFF (TP=2)
│   ├── model.pt
│   ├── neuron_config.json
│   └── weights/
├── vae_shards/                              # Pre-compiled VAE decoder (24 shard NEFFs)
│   ├── vae_conv_in.pt
│   ├── vae_mid_resnet0.pt
│   ├── vae_mid_attn0.pt
│   ├── ... (24 files total)
│   └── vae_norm_conv_out.pt
├── byt5_traced.pt                           # Pre-compiled ByT5 encoder NEFF
├── refiner_traced.pt                        # Pre-compiled token refiner NEFF
├── reorder_traced.pt                        # Pre-compiled token reorder NEFF
├── cond_type_embed_weight.pt                # Extracted embedding weights
├── TRN2_REWORK.md                           # Documents trn1→trn2 changes
├── VAE_COMPILATION_WORKAROUNDS.md           # Documents VAE model rewrites for Neuron
├── WHY_COMPILATION_IS_SLOW.md               # Explains VAE compilation time
└── output_cfg3/                             # Sample output (33 frames)
```

## Required Instance

- **trn2.48xlarge** (16 NeuronDevices, 64 NeuronCores, 96 GB HBM per device)
- trn1.32xlarge also works but NEFFs must be recompiled (they are architecture-specific)

## Step 1: Install Neuron SDK

If `neuron-ls` is not available or shows no devices, install the SDK:

```bash
# Add Neuron apt repository
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com jammy main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Install driver and runtime
sudo apt-get update -y
sudo apt-get install -y aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools

# Verify devices are visible
neuron-ls
# Expected: 16 NeuronDevices with 4 cores each
```

## Step 2: Install Python Dependencies

```bash
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    torch-neuronx neuronx-cc neuronx-distributed neuronx-distributed-inference \
    transformers diffusers accelerate huggingface_hub sentencepiece protobuf
```

Verify:
```bash
python3 -c "import torch_neuronx; print(torch_neuronx.__version__)"
```

## Step 3: Download HuggingFace Model Weights

The pipeline downloads weights automatically on first run. To pre-download (~32 GB total):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v')
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')
snapshot_download('google/byt5-small')
"
```

Weights are cached in `~/.cache/huggingface/hub/`.

If HuggingFace requires authentication:
```bash
pip install huggingface_hub
huggingface-cli login
```

## Step 4: Compile NEFFs (only if pre-compiled artifacts are missing)

The pre-compiled NEFFs in this directory were built for trn2.48xlarge. If they are present, skip this step.

If NEFFs are missing or you are on a different instance type, compile from scratch:

```bash
cd /path/to/this/directory

# Compile everything (takes ~4.5 hours, dominated by VAE)
python3 compile_all.py --output_dir .

# Or compile individual components:
python3 compile_all.py --output_dir . --only transformer   # ~10 min
python3 compile_all.py --output_dir . --only qwen          # ~2 min
python3 compile_all.py --output_dir . --only traced        # ~2 min
python3 compile_all.py --output_dir . --only weights       # ~30s
python3 compile_all.py --output_dir . --only vae           # ~4.5 hrs

# Skip components that are already compiled:
python3 compile_all.py --output_dir . --skip_transformer --skip_vae
```

After compilation, verify all artifacts exist:
```bash
ls compiled_transformer/model.pt compiled_qwen2vl/model.pt \
   byt5_traced.pt refiner_traced.pt reorder_traced.pt \
   cond_type_embed_weight.pt vae_shards/vae_conv_in.pt
```

## Step 5: Run Inference

```bash
cd /path/to/this/directory

python3 run_inference.py \
    --prompt "A cat walking on a beach at sunset, realistic, 4K" \
    --output_dir my_video/ \
    --steps 50 \
    --cfg 3.0 \
    --seed 42
```

This produces 33 PNG frames at 480×640 in `my_video/`. Takes ~8 minutes.

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | (see source) | Text description of the video to generate |
| `--output_dir` | `pipeline_output` | Directory for output PNG frames |
| `--steps` | 50 | Number of denoising steps (more = better quality, slower) |
| `--cfg` | 3.0 | Classifier-free guidance scale (higher = stronger prompt following, risk of oversaturation) |
| `--seed` | 42 | Random seed for reproducibility |
| `--compiled_dir` | (script directory) | Path to directory containing compiled NEFFs |
| `--vae_on_cpu` | off | Run VAE decoder on CPU (skips need for VAE NEFF compilation) |

### Fast start (skip VAE compilation)

To get running in ~15 minutes instead of ~4.5 hours, skip VAE compilation and use CPU decode:

```bash
# Compile everything except VAE (~15 min)
python3 compile_all.py --output_dir . --skip_vae

# Run with CPU VAE (~5s slower decode, but no 4.5hr compile)
python3 run_inference.py \
    --prompt "A cat walking on a beach at sunset" \
    --vae_on_cpu
```

## Troubleshooting

**`neuron-ls` shows no devices**: Run `sudo modprobe neuron` and check `dmesg` for driver errors.

**`RuntimeError: The PyTorch Neuron Runtime could not be initialized`**: Another process is holding the NeuronCores. Kill it with `sudo kill $(pgrep -f neuron)` or wait for it to finish.

**`NCC_EVRF029: Operation sort is not supported on trn2`**: The `reorder_traced.pt` was compiled for trn1. Recompile with `python3 compile_all.py --output_dir . --only traced`.

**`neuronx-cc failed with 70` during VAE compilation**: Expected for the full decoder (33M instructions). The `compile_all.py` script compiles it as 24 individual shards which each stay under the limit.

**Compilation takes much longer than expected**: The VAE shards at full resolution (33×480×640) take ~44 minutes each. Total VAE compilation is ~4.5 hours. This is a one-time cost.

**Output looks oversaturated / "deep fried"**: Lower the `--cfg` value. Default is 3.0. Values above 6.0 tend to oversaturate.
