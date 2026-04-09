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
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

from src.modeling_minimax_m2 import NeuronMiniMaxM2ForCausalLM, MiniMaxM2InferenceConfig

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

**Last Updated:** 2026-04-09
