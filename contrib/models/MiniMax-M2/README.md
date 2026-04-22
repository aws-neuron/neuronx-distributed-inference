# Contrib Model: MiniMax M2

NeuronX Distributed Inference implementation of [MiniMax/MiniMax-M2](https://huggingface.co/MiniMax/MiniMax-M2).

## Model Information

- **HuggingFace ID:** `MiniMax/MiniMax-M2`
- **Model Type:** Decoder-only MoE transformer
- **Architecture:** Custom MoE with sigmoid routing and e_score_correction_bias
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Hidden Size | 3072 |
| Layers | 62 |
| Attention Heads | 48 Q / 8 KV (GQA) |
| Head Dim | 128 |
| Experts | 256 (top-8 routing) |
| Expert Intermediate | 1536 |
| MLP Intermediate | 8192 |
| Vocab Size | 200,064 |
| RoPE | Partial (50% of head_dim), theta=5M |
| Max Position | 196,608 |

Key features:
- **QK Norm**: Applied before reshape on full projection output
- **Partial RoPE**: Only first 64 of 128 dims use rotary encoding
- **Sigmoid Router**: With learnable e_score_correction_bias for expert selection
- **fused_qkv**: Supported for efficient Q/K/V projection

## Prerequisites

- **Instance**: trn2.48xlarge (32 NeuronCores, logical_nc_config=2 -> 64 logical cores)
- **Weights**: BF16 format (convert from FP8 original if needed)

## Usage

```python
import sys
from pathlib import Path

# Make this contrib package's src/ importable (flat, per upstream contrib convention).
sys.path.insert(0, str(Path("contrib/models/MiniMax-M2/src").resolve()))

import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

from modeling_minimax_m2 import NeuronMiniMaxM2ForCausalLM, MiniMaxM2InferenceConfig

model_path = "/path/to/MiniMax-M2-BF16/"
compiled_path = "/path/to/compiled/"

neuron_config = MoENeuronConfig(
    tp_degree=64,
    moe_tp_degree=64,
    moe_ep_degree=1,
    batch_size=1,
    seq_len=512,
    max_context_length=256,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    sequence_parallel_enabled=True,
    fused_qkv=True,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95
    ),
    router_config={act_fn: sigmoid},
)

config = MiniMaxM2InferenceConfig(
    neuron_config, load_config=load_pretrained_config(model_path)
)

model = NeuronMiniMaxM2ForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
adapter = HuggingFaceGenerationAdapter(model, tokenizer)
output = adapter.generate("Hello, how are you?", max_new_tokens=128)
```

## vLLM Integration

MiniMax-M2 can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A patch is required to add MiniMax architecture support.

### Setup

```bash
# 1. Install vllm-neuron
pip install vllm-neuron

# 2. Apply the MiMo/MiniMax patch
cd /path/to/vllm-neuron
git apply /path/to/neuronx-distributed-inference/contrib/models/MiniMax-M2/perf_test/vllm-neuron-patch.patch
pip install -e .
```

### Serving

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/MiniMax-M2-BF16 \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 256 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 64,
            "logical_nc_config": 2,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": true,
            "glu_mlp": true,
            "moe_mask_padded_tokens": true,
            "disable_numeric_cc_token": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 256,
            "ctx_batch_size": 1,
            "tkg_batch_size": 256,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "fused_qkv": false,
            "enable_bucketing": true,
            "normalize_top_k_affinities": true,
            "use_index_calc_kernel": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": true,
                "skip_dma_token": true
            },
            "scratchpad_page_size": 1024
        }
    }'
```

### Key vLLM Patch Changes

The patch (`contrib/models/MiniMax-M2/perf_test/vllm-neuron-patch.patch`) modifies vllm-neuron to:
- Pass `hf_config` from vLLM to NxDI (required for `trust_remote_code` models)
- Replace `AutoModelForCausalLM.from_pretrained` with `snapshot_download` for model loading

See `contrib/models/MiniMax-M2/perf_test/bench_minimax_m2.sh` for full benchmark configurations with BS=1/256. Run `contrib/models/MiniMax-M2/perf_test/0_setup.sh` first to install vllm-neuron and fetch weights.

## Performance

### vLLM Serving — Config 1 (trn2.48xlarge, BF16, BS=1, TP=64/EP=1, non-CB)

Input/output: 900/90 tokens (random dataset)

| Concurrency | Throughput (tok/s) | TPOT (ms) | TTFT (ms) |
|-------------|-------------------|-----------|-----------|
| 1 | 39.28 | 13.56 | 1088 |

### vLLM Serving — Config 2 (trn2.48xlarge, BF16, BS=256, TP=64/EP=64, CB)

Input/output: 900/90 tokens (random dataset)

| Concurrency | Throughput (tok/s) | TPOT (ms) | TTFT (ms) |
|-------------|-------------------|-----------|-----------|
| 1 | 5.76 | 173.83 | 165 |
| 16 | 54.69 | 287.09 | 513 |
| 32 | 75.85 | 408.66 | 1066 |
| 128 | 106.72 | 1158.08 | 3950 |
| 256 | 128.94 | 1860.69 | 11263 |

> **Note:** Large MoE models like MiniMax-M2 require extended engine startup time. Set `VLLM_ENGINE_READY_TIMEOUT_S=3600` before launching the vLLM server.

## Compatibility Matrix

| Instance/Version | 2.22+ (PyTorch 2.9) | 2.21 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not supported (requires 64 cores) | Not supported |
| Inf2 | Not supported | Not supported |

## Testing

```bash
pytest contrib/models/MiniMax-M2/test/integration/test_model.py -v
```

## Example Checkpoints

* [MiniMax/MiniMax-M2](https://huggingface.co/MiniMax/MiniMax-M2)
* [MiniMax/MiniMax-M2-unquantized](https://huggingface.co/MiniMax/MiniMax-M2-unquantized) (BF16)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-13
