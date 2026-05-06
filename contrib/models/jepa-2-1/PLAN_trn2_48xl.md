# V-JEPA 2.1 — ViT-g / ViT-G on trn2.48xlarge

## Objective

Compile, test, and benchmark the two larger V-JEPA 2.1 model sizes on a trn2.48xlarge instance (2 TB RAM, 64 NeuronCores). These models OOM'd during compilation on trn2.3xlarge (124 GB RAM).

| Model | Params | embed_dim | depth | num_heads | Checkpoint |
|-------|--------|-----------|-------|-----------|------------|
| ViT-g | 1.01B | 1408 | 40 | 22 | `vjepa2_1_vitg_384.pt` (~4 GB) |
| ViT-G | 1.8B | 1664 | 48 | 26 | `vjepa2_1_vitG_384.pt` (~7 GB) |

## Instance

- **Instance ID:** `i-09812af3093beb594`
- **Type:** trn2.48xlarge (96 vCPUs, 2 TB RAM, 64 NeuronCores)
- **Region:** us-east-2
- **AMI:** ami-0a81a0376c52f4d22
- **Access:** SSM (no SSH)
- **State:** stopped → starting

## Execution Plan

### Phase 1: Environment Setup

1. Start instance, wait for SSM connectivity
2. Copy `contrib/models/jepa-2-1/` to instance via SSM + tar
3. Activate Neuron venv: `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
4. Verify SDK: `neuron-ls`, `pip show torch-neuronx neuronx-cc`

### Phase 2: ViT-g (1B params)

5. Compile ViT-g with `pretrained=False` (random weights) — image input (1,3,1,384,384)
6. Compile ViT-g with `pretrained=False` — video input (1,3,16,384,384)
7. Validate accuracy: Neuron BF16 vs CPU FP32 (cosine similarity, `neuron_allclose`)
8. Compile ViT-g with `pretrained=True` — validate pretrained weights
9. Benchmark: single NC latency (100 iterations), DataParallel throughput

### Phase 3: ViT-G (1.8B params)

10. Repeat steps 5–9 for ViT-G

### Phase 4: Results

11. Collect all metrics, update this file with results
12. Stop instance

## Expected Outputs

- Compilation time for each model × input type
- Single NeuronCore latency (median, p5, p95)
- DataParallel throughput (clips/sec)
- Cosine similarity (BF16 Neuron vs FP32 CPU)
- Real-time video processing factor (vs 30fps, 16 frames = 0.53s)

## Results

### ViT-g (1B)

| Metric | Image (1 frame) | Video (16 frames) |
|--------|------------------|--------------------|
| Compilation time | — | — |
| Output shape | — | — |
| Cosine similarity | — | — |
| Median latency (ms) | — | — |
| p5 / p95 (ms) | — / — | — / — |
| DP throughput (clips/s) | — | — |

### ViT-G (1.8B)

| Metric | Image (1 frame) | Video (16 frames) |
|--------|------------------|--------------------|
| Compilation time | — | — |
| Output shape | — | — |
| Cosine similarity | — | — |
| Median latency (ms) | — | — |
| p5 / p95 (ms) | — / — | — / — |
| DP throughput (clips/s) | — | — |

### Comparison (all models, video 16 frames, single NC)

| Model | Params | Median (ms) | Real-time factor | Instance |
|-------|--------|-------------|------------------|----------|
| ViT-B | 86M | 247.4 | 2.1x | trn2.3xlarge |
| ViT-L | 300M | 741.8 | 0.7x | trn2.3xlarge |
| ViT-g | 1.01B | — | — | trn2.48xlarge |
| ViT-G | 1.8B | — | — | trn2.48xlarge |

## Notes

- `use_sdpa=False` required (SDPA not supported by `torch_neuronx.trace()`)
- `--auto-cast none` compiler flag required for BF16
- ViT-g/G use `target_encoder` key in checkpoint (not `ema_encoder`)
- NKI flash attention disabled for 16-frame inference (slower at 4,608 tokens)
