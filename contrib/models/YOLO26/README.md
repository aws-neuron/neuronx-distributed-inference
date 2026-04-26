# Contrib Model: YOLO26

Ultralytics YOLO26 object detection models on AWS Trainium2 using `torch_neuronx.trace()`.

## Model Information

- **Source:** [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics)
- **Model Type:** Object detection (also supports segmentation, pose estimation, oriented bounding boxes)
- **Variants:** 5 detection sizes — n (2.4M), s (10.0M), m (21.9M), l (26.3M), x (58.9M)
- **Architecture:** CNN backbone (Conv2d + BatchNorm + SiLU), FPN/PAN neck, Detect head with C2PSA attention
- **Input:** `[B, 3, 640, 640]` (fixed resolution)
- **Output:** `[B, 84, 8400]` (4 bbox + 80 COCO class scores per anchor)
- **License:** AGPL-3.0 or Ultralytics Enterprise License

## Architecture Details

YOLO26 is a 24-layer convolutional neural network optimized for real-time object detection. Unlike transformer-based vision models, it is dominated by Conv2d operations with a small C2PSA self-attention block on the P5 feature map.

This contrib uses `torch_neuronx.trace()` rather than NxDI model classes because: (1) all variants fit trivially on a single NeuronCore (<180 MB NEFF), (2) there is no KV cache or token generation, and (3) the Conv2d-dominant architecture does not benefit from NxDI's attention infrastructure. Data Parallelism across NeuronCores provides throughput scaling.

Key Neuron porting challenges:
- **`topk`/`sort` unsupported:** End-to-end postprocessing requires `torch.topk` which fails with `NCC_EVRF029`. Solution: trace with `end2end=False` for raw output, run postprocessing on CPU.
- **FP32 SB overflow for m/l/x:** Larger variants exceed Neuron's SB allocation in FP32. Solution: BF16 compilation (halves tensor sizes).
- **`--auto-cast=matmult` produces NaN:** Conv2d-dominant models get NaN with matmult autocast. Solution: no autocast flags.

## Validation Results

**Validated:** 2026-04-25
**Instance:** trn2.3xlarge (1 Trainium2 chip)
**SDK:** Neuron SDK 2.28, PyTorch 2.9

### Peak Throughput (LNC=1, DP=8)

| Variant | Params | Dtype | NEFF (MB) | BS/core | img/s | A10G Compiled | Speedup |
|---------|--------|-------|-----------|---------|-------|---------------|---------|
| YOLO26n | 2.4M | FP32 | 19.5 | 1 | 272 | 2,166 | 0.13x |
| YOLO26s | 10.0M | FP32 | 69.6 | 32 | 1,523 | 1,065 | **1.43x** |
| YOLO26m | 21.9M | BF16 | 66.4 | 32 | 1,267 | 474 | **2.67x** |
| YOLO26l | 26.3M | BF16 | 80.8 | 32 | 1,093 | 371 | **2.95x** |
| YOLO26x | 58.9M | BF16 | 177.7 | 16 | 876 | 195 | **4.49x** |

### Accuracy Validation

| Variant | Dtype | Cosine Similarity | Max Error | Has NaN |
|---------|-------|-------------------|-----------|---------|
| YOLO26n | FP32 | 0.9943 | 373.3 | No |
| YOLO26s | FP32 | 0.9932 | 439.8 | No |
| YOLO26m | BF16 | 0.9879 | 488.0 | No |
| YOLO26l | BF16 | 0.9967 | 242.0 | No |
| YOLO26x | BF16 | 0.9950 | 378.0 | No |

### Additional Task Heads

| Task | Head | CosSim | img/s (single core) | Status |
|------|------|--------|---------------------|--------|
| Pose | Pose26 | 0.9996 | 81.6 | Production ready |
| OBB | OBB26 | 0.9999 | 85.3 | Production ready |
| Segmentation | Segment26 | 0.995/0.858 | 63.9 | Proto mask needs validation |
| Classification | Classify | 0.257 | 671.0 | Precision issue (softmax sensitivity) |

**Status:** VALIDATED

## Usage

### Quick Start

```python
from src import YOLO26NeuronModel

# Single core
model = YOLO26NeuronModel("s", batch_size=1)
output = model(torch.randn(1, 3, 640, 640))

# Data parallel (4 cores on LNC=2)
model = YOLO26NeuronModel("s", batch_size=8, num_cores=4)
output = model(torch.randn(32, 3, 640, 640))

# Benchmark
results = model.benchmark(warmup=10, iterations=50)
print(f"Throughput: {results['throughput_img_s']} img/s")
```

### Low-Level API

```python
from src import prepare_yolo26, compile_yolo26, validate_accuracy

# Prepare and compile
model = compile_yolo26("yolo26s.pt", batch_size=1, save_path="compiled/yolo26s.pt")

# Validate accuracy
metrics = validate_accuracy("yolo26s.pt", model)
print(f"CosSim: {metrics['cosine_similarity']}")
```

### Known Issues

1. **`topk` not supported on Neuron.** Models must be traced with `end2end=False`. Postprocessing (topk, NMS) runs on CPU with ~0.1ms overhead.
2. **FP32 fails for m/l/x variants.** Use BF16 (`torch.bfloat16`) for these variants. FP32 for n/s only.
3. **`--auto-cast=matmult` produces NaN.** Do not use autocast flags with YOLO26.
4. **LNC=1 requires `--lnc 1` compiler flag.** NEFFs compiled without this flag cannot run on LNC=1 runtime.
5. **Classification variant has precision issue.** Narrow logit range + softmax amplification causes CosSim 0.257. Detection, pose, and OBB are unaffected.

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.29 |
|----------|----------|----------|
| trn2.3xlarge | Validated | Expected compatible |
| trn2.48xlarge | Expected compatible | Expected compatible |
| inf2.xlarge | Not tested (n/s only) | Not tested |

## Testing Instructions

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install ultralytics

# Run integration tests
cd contrib/models/YOLO26
pytest test/integration/test_model.py -v

# Or standalone
python test/integration/test_model.py
```

## Maintainer

Jim Burtoft
Community contribution

**Last Updated:** 2026-04-25
