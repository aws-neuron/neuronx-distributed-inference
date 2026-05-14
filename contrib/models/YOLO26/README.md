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

**Validated:** 2026-04-29
**Instance:** trn2.3xlarge (1 Trainium2 chip), inf2.xlarge (1 Inferentia2 chip)
**SDK:** Neuron SDK 2.28 and 2.29, PyTorch 2.9

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

1. **C2PSA attention: `neuronx-cc` bug with `torch.Tensor.split()` (unequal sizes).** The Neuron compiler produces incorrect output when `.split([32, 32, 64], dim=2)` is applied to a 4D tensor after a `.view()` reshape. This caused the C2PSA attention module to produce CosSim ~0.46 vs CPU. **Fixed in `prepare_yolo26()`** by patching `Attention.forward` to use tensor slicing instead of `.split()`. The fix produces CosSim 0.9999 at batch_size=1. See [aws-neuron-sdk#1323](https://github.com/aws-neuron/aws-neuron-sdk/issues/1323) for the upstream compiler issue.
2. **C2PSA `.split()` at batch_size >= 2: corrupts non-first batch elements.** The same `.split()` compiler bug from #7 also affects C2PSA's `cv1(x).split((c, c), dim=1)` — when combined with downstream attention, all batch elements except element 0 produce garbled output (CosSim ~0.08-0.23). **Fixed in `prepare_yolo26()`** by patching `C2PSA.forward` to use `.chunk(2, 1)` instead of `.split((c, c), dim=1)`. Batch sizes > 1 now work correctly. See [aws-neuron-sdk#1323](https://github.com/aws-neuron/aws-neuron-sdk/issues/1323).
3. **`topk` not supported on Neuron.** Models must be traced with `end2end=False`. Use `postprocess_detections()` for CPU NMS postprocessing (~0.1ms overhead).
4. **FP32 fails for m/l/x variants.** Use BF16 (`torch.bfloat16`) for these variants. FP32 for n/s only.
5. **`--auto-cast=matmult` produces NaN.** Do not use autocast flags with YOLO26.
6. **LNC=1 requires `--lnc 1` compiler flag.** NEFFs compiled without this flag cannot run on LNC=1 runtime.
7. **`torch.Tensor.split()` compiler bug (two manifestations):**
   - *Numerical corruption:* `.split()` with unequal sizes on dim=2 of a 4D tensor (in C2PSA Attention) produces CosSim ~0.45. Fixed by patching `Attention.forward` to use tensor slicing.
   - *Compilation failure:* `.split((c, c), dim=1)` in C2f blocks causes exit code 70 at batch_size=4 with small spatial dimensions (H×W < ~264). Fixed by using `.chunk(2, 1)` instead. See [aws-neuron-sdk#1323](https://github.com/aws-neuron/aws-neuron-sdk/issues/1323).
8. **Classification variant has precision issue.** Narrow logit range + softmax amplification causes CosSim 0.257. Detection, pose, and OBB are unaffected.

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.29 |
|----------|----------|----------|
| trn2.3xlarge | Validated (13/13 tests) | Validated (13/13 tests) |
| trn2.48xlarge | Expected compatible | Expected compatible |
| inf2.xlarge | Validated (n/s) | Validated (all 5 variants) |

### inf2 Single-Core Throughput (SDK 2.29)

| Variant | Dtype | CosSim | img/s | trn2 single-core |
|---------|-------|--------|-------|------------------|
| YOLO26n | FP32 | 0.9966 | 55.2 | 32.3 |
| YOLO26s | FP32 | 0.9934 | 82.8 | 66.0 |
| YOLO26m | BF16 | 0.9877 | 104.0 | 75.5 |
| YOLO26l | BF16 | 0.9966 | 91.6 | 66.3 |
| YOLO26x | BF16 | 0.9950 | 69.9 | 57.3 |

*Note: inf2 single-core outperforms trn2 single-core for all variants. trn2 advantage comes from DP=8 (LNC=1) scaling.*

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
