# Contrib Model: MiMo-V2-Flash

NeuronX Distributed Inference implementation of [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash).

## Model Information

- **HuggingFace ID:** `XiaomiMiMo/MiMo-V2-Flash`
- **Model Type:** Decoder-only MoE transformer with hybrid attention
- **Architecture:** Custom MoE with full + sliding window attention
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Hidden Size | 4096 |
| Layers | 48 |
| Attention Heads | 64 Q |
| KV Heads (full attn) | 4 |
| KV Heads (sliding window) | 8 |
| Q/K Head Dim | 192 |
| V Head Dim | 128 |
| Experts | 256 (top-8 routing) |
| Expert Intermediate | 1536 |
| Vocab Size | 151,936 |
| RoPE | Partial (34% of dims), theta=5M |
| Sliding Window | 32,768 |
| Max Position | 262,144 |
| Total Params | ~143B (FP8) / ~286B (BF16) |

Key features:
- **Hybrid Attention**: 9 full attention layers (0, 5, 11, 17, 23, 29, 35, 41, 47) + 39 sliding window layers
- **Asymmetric Head Dims**: Q/K use 192, V uses 128 (fused_qkv not supported)
- **Attention Sink Bias**: Learnable per-head bias in sliding window layers
- **Sigmoid Router**: For MoE expert selection
- **Expert Parallelism**: Supports EP=64 for prefill with hybrid sharding (EP=1 for token generation)

## Prerequisites

- **Instance**: trn2.48xlarge (32 NeuronCores, logical_nc_config=2 -> 64 logical cores)
- **Weights**: BF16 format (convert from FP8 using `conversion_script/preprocess_mimo_v2_fp8.py`)

## FP8 to BF16 Conversion

The original model uses block-wise FP8 quantization incompatible with Neuron FP8. Convert to BF16:

```bash
python src/neuronx_distributed_inference/models/mimo_v2/conversion_script/preprocess_mimo_v2_fp8.py \
    --input-dir /path/to/MiMo-V2-Flash \
    --output-dir /path/to/MiMo-V2-Flash-BF16
```

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

from src.modeling_mimo_v2 import NeuronMiMoV2ForCausalLM, MiMoV2InferenceConfig

model_path = "/path/to/MiMo-V2-Flash-BF16/"
compiled_path = "/path/to/compiled/"

neuron_config = MoENeuronConfig(
    tp_degree=64,
    moe_tp_degree=1,
    moe_ep_degree=64,
    batch_size=1,
    seq_len=512,
    max_context_length=128,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    sequence_parallel_enabled=True,
    fused_qkv=False,  # Required: asymmetric Q/K vs V dims
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95
    ),
    router_config={act_fn: sigmoid},
)

config = MiMoV2InferenceConfig(
    neuron_config, load_config=load_pretrained_config(model_path)
)

model = NeuronMiMoV2ForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
adapter = HuggingFaceGenerationAdapter(model, tokenizer)
output = adapter.generate("Hello, how are you?", max_new_tokens=128)
```

## vLLM Integration

MiMo-V2-Flash can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A patch is required to add MiMo architecture support.

### Setup

```bash
# 1. Install vllm-neuron
pip install vllm-neuron

# 2. Apply the MiMo/MiniMax patch
cd /path/to/vllm-neuron
git apply /path/to/neuronx-distributed-inference/perf_test/vllm-neuron-mimo-minimax.patch
pip install -e .
```

### Serving

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/MiMo-V2-Flash-BF16 \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 64,
            "logical_nc_config": 2,
            "fused_qkv": false,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": true,
            "glu_mlp": true,
            "normalize_top_k_affinities": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "async_mode": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            }
        }
    }'
```

### Key vLLM Patch Changes

The patch (`perf_test/vllm-neuron-mimo-minimax.patch`) modifies vllm-neuron to:
- Map MiMo architecture to Qwen2 model loader (MiMo is Qwen2-based)
- Pass `hf_config` from vLLM to NxDI (required for `trust_remote_code` models)
- Replace `AutoModelForCausalLM.from_pretrained` with `snapshot_download` for model loading

See `perf_test/1_bench_mimo_v2_flash.sh` for full benchmark configurations with BS=1/32/128.

## Performance

Standalone NxDI (trn2.48xlarge, BF16, TP=64, EP=64):

| Batch Size | Throughput (tok/s) |
|------------|-------------------|
| 1 | 29.92 |
| 8 | 215.94 |
| 32 | 649.14 |

## Compatibility Matrix

| Instance/Version | 2.22+ (PyTorch 2.9) | 2.21 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested | Not tested |
| Trn1 | Not supported (requires 64 cores) | Not supported |
| Inf2 | Not supported | Not supported |

## Testing

```bash
pytest contrib/models/MiMo-V2-Flash/test/integration/test_model.py -v
```

## Key Implementation Notes

1. **Hybrid Attention**: `hybrid_layer_pattern` list determines full vs sliding window per layer.
2. **CONVERT_TO_MHA**: When TP > num_kv_heads (4), K/V are replicated to match Q heads (64).
3. **Attention Sink Bias**: Adds learnable sink column to attention weights in sliding window layers.
4. **EP Hybrid Sharding**: EP is used during prefill only; token generation uses EP=1 unless batch_size >= 32.
5. **FP8 Conversion**: Original uses OCP block-wise FP8, requires conversion to BF16 or Neuron-compatible FP8 format.

## Example Checkpoints

* [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-09
