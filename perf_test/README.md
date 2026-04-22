# Performance Test Plan for PR #119 (MiMo-V2-Flash & MiniMax-M2)

## Overview

Use vllm-neuron to benchmark both models on trn2.48xlarge with various batch sizes and parallelism configs.

## Prerequisites

- **Instance**: trn2.48xlarge (32 NeuronCores, 2TB RAM, 1.7TB NVMe)
- **vllm-neuron**: Fork with MiMo support (https://github.com/whn09/vllm-neuron/tree/feature/mimo-support)
- **NxDI**: PR #119 branch installed at `/tmp/nxdi-fork-main/`
- **Model weights** (BF16, from S3):
  - MiMo-V2-Flash: `s3://datalab/xiaomi/models/MiMo-V2-Flash-BF16/`
  - MiniMax-M2: `s3://datalab/minimax/model_hf/MiniMax-M2-BF16/` (already downloaded)

## Test Configurations

### MiMo-V2-Flash

| Config | BS | TP | EP | CB | Optimizations | Benchmark Concurrency |
|--------|----|----|----|----|---------------|----------------------|
| 1 | 1 | 64 | 1 | No | baseline | 1 |
| 2 | 32 | 1 | 64 | Yes | index_calc, blockwise, scratchpad | 1, 16, 32 |
| 3 | 128 | 1 | 64 | Yes | index_calc, blockwise, scratchpad | 1, 16, 32, 128 |

### MiniMax-M2

| Config | BS | TP | EP | CB | Optimizations | Benchmark Concurrency |
|--------|----|----|----|----|---------------|----------------------|
| 1 | 1 | 64 | 1 | No | baseline, fused_qkv=true | 1 |
| 2 | 256 | 1 | 64 | Yes | index_calc, blockwise, scratchpad | 1, 16, 32, 128, 256 |

### Benchmark Parameters
- Dataset: random
- Input length: 900 tokens
- Output length: 90 tokens
- Range ratio: 0.03
- Prompts per run: 16 (concurrency=1), 128 (concurrency 16/32), 512 (concurrency 128/256)

## Scripts

1. `0_setup.sh` - Install vllm-neuron, download model weights
2. `1_bench_mimo_v2_flash.sh` - MiMo-V2-Flash benchmark (all configs)
3. `2_bench_minimax_m2.sh` - MiniMax-M2 benchmark (all configs)

## Execution

```bash
# Step 1: Setup (one-time)
bash 0_setup.sh

# Step 2: Run MiMo benchmarks
bash 1_bench_mimo_v2_flash.sh 2>&1 | tee /tmp/mimo_bench_results.log

# Step 3: Run MiniMax benchmarks
bash 2_bench_minimax_m2.sh 2>&1 | tee /tmp/minimax_bench_results.log
```
