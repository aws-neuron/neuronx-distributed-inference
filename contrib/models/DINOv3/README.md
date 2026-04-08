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
| ViT-S/16 | 21.6M | `torch_neuronx.trace()` | inf2.xlarge |
| ViT-B/16 | 85.7M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-L/16 | 303.2M | `torch_neuronx.trace()` | trn2.3xlarge |
| ViT-H+/16 | 840.6M | `torch_neuronx.trace()` | trn2.3xlarge |
| ViT-7B/16 | 6,716M | `neuronx-distributed` ModelBuilder TP=4 | trn2.3xlarge |
| ConvNeXt-T | 27.8M | `torch_neuronx.trace()` | inf2.xlarge |
| ConvNeXt-B | 87.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |

**Key insight**: ViT-7B is the first encoder-only vision model to use tensor parallelism on Neuron. The 20.1 GB NEFF does not fit in single-core HBM, so TP=4 via `neuronx-distributed` is required.

## Results

### Accuracy (matmult bf16 vs CPU FP32)

| Model | Cosine Similarity | Max Abs Diff |
|-------|------------------:|-------------:|
| ViT-S/16 | 1.000000 | < 0.001 |
| ViT-B/16 | 1.000000 | < 0.001 |
| ViT-L/16 | 1.000000 | < 0.001 |
| ViT-H+/16 | 1.000000 | < 0.001 |
| ViT-7B/16 | Deterministic (random weights) | -- |
| ConvNeXt-T | 0.999989 | < 0.001 |
| ConvNeXt-B | 0.999989 | < 0.001 |

### Benchmark (trn2.3xlarge, LNC=2, DP=4)

| Model | NEFF Size | Compile Time | 1-Core (img/s) | DP=4 Peak (img/s) |
|-------|----------:|-------------:|----------------:|-------------------:|
| ViT-S/16 | 68 MB | 84s | 367 | **722.8** |
| ViT-B/16 | 264 MB | 45s | 222 | **438.7** |
| ViT-L/16 | 931 MB | 123s | 87.6 | **174.7** |
| ViT-H+/16 | 2,595 MB | 688s | 5.2 | 10.5 |
| ViT-7B/16 | TP=4 NEFF | 5.9s | OOM | **38.8** (TP=4) |
| ConvNeXt-T | 90 MB | 44s | 183 | **363.3** |
| ConvNeXt-B | 275 MB | 63s | 130 | **257.8** |

### Key Findings

1. **`--auto-cast=matmult` is critical**: FP32 models get 50-60% speedup with matmult bf16 autocast, consistent with SigLIP and MoLFormer results
2. **ViT 1.7x faster than ConvNeXt**: At comparable parameter counts, ViT models are significantly faster on Neuron (transformer ops are heavily optimized)
3. **DataParallel scales near-perfectly**: DP=4 achieves ~1.95-2.0x over single-core across all models
4. **ViT-H+ is HBM-bandwidth limited**: 2.5 GB NEFF saturates single-core HBM bandwidth, resulting in only 10.5 img/s DP=4 (16.6x slower than ViT-L)
5. **ViT-7B requires TP=4**: 20.1 GB NEFF exceeds single-core HBM. Tensor parallelism via `neuronx-distributed` ModelBuilder achieves 38.8 img/s at 25.77ms latency

### GPU Comparison (A10G g5.xlarge)

| Model | Neuron Best (trn2 DP=4) | GPU Best (A10G torch.compile BS=16) | Winner |
|-------|------------------------:|------------------------------------:|--------|
| ViT-B/16 | **440.6 img/s** | 380.0 img/s | Neuron 1.16x |
| ConvNeXt-Tiny | 364.5 img/s | **1,156.2 img/s** | GPU 3.2x |

Neuron excels on ViT (transformer ops), GPU excels on ConvNeXt (conv ops). The hardware advantage depends on model architecture.

<details>
<summary>Full GPU results (A10G, PyTorch 2.6)</summary>

**ViT-B/16:**

| Batch Size | Eager (img/s) | torch.compile (img/s) |
|-----------:|--------------:|----------------------:|
| 1 | 92.1 | 213.2 |
| 4 | 240.2 | 302.5 |
| 8 | 290.1 | 341.0 |
| 16 | 330.5 | 380.0 |

**ConvNeXt-Tiny:**

| Batch Size | Eager (img/s) | torch.compile (img/s) |
|-----------:|--------------:|----------------------:|
| 1 | 213.0 | 571.4 |
| 4 | 551.7 | 893.5 |
| 8 | 665.2 | 1,020.3 |
| 16 | 800.1 | 1,156.2 |

</details>

## Compatibility

| Component | Version |
|-----------|---------|
| **Neuron SDK** | 2.28 |
| **torch-neuronx** | 2.9.0.2.11 |
| **neuronx-cc** | 2.22.12471 |
| **neuronx-distributed** | 0.16.25997 (ViT-7B TP only) |
| **Instance (small/medium)** | inf2.xlarge, trn2.3xlarge |
| **Instance (ViT-7B)** | trn2.3xlarge (TP=4, LNC=2) |
| **DLAMI** | Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 |

## Usage

### Setup

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

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

# Compile for Neuron
model_neuron = trace_dinov3(model, is_convnext=False, save_path="/tmp/dinov3_vit_b.pt")

# Validate accuracy
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

# Requires NEURON_RT_NUM_CORES=4 and trn2.3xlarge
nxd_model = compile_vit7b_tp(tp_degree=4)
perf = benchmark_model(nxd_model)
print(f"TP=4 throughput: {perf['throughput_img_s']:.1f} img/s")
```

### DataParallel Benchmark

```python
from modeling_dinov3 import benchmark_dataparallel

# DP=4 across all NeuronCores on trn2.3xlarge (LNC=2)
dp_results = benchmark_dataparallel(model_neuron, num_cores=4)
for bs, r in dp_results.items():
    print(f"BS={bs}: {r['throughput_img_s']:.1f} img/s")
```

## Running Tests

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

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
- ViT-H+ traces successfully but is HBM-bandwidth-limited (2.5 GB NEFF). Consider TP for production use of models > 500M params.
- DINOv3 License is not Apache/MIT -- review before redistribution.

## Maintainer

Jim Burtoft (`jimburtoft`)
