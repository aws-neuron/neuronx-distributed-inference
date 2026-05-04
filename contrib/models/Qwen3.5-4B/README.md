# Contrib Model: Qwen3.5-4B

NeuronX Distributed Inference implementation of Qwen3.5-4B, a dense hybrid DeltaNet + GQA decoder from Alibaba Cloud.

This variant reuses the proven PR 140/141 architecture:

- Standard GQA layers use NxDI `KVCacheManager`.
- DeltaNet layers return dummy KV tensors to satisfy NxDI cache plumbing.
- Real DeltaNet state is carried through layer-local `recurrent_state_buffer` and `conv_state_buffer` side-channel aliases.

## Model Information

| Feature | Value |
| --- | --- |
| HuggingFace ID | `Qwen/Qwen3.5-4B` |
| Model type | `qwen3_5_text` under top-level `qwen3_5` |
| Layers | 32: 24 DeltaNet + 8 GQA |
| Layer pattern | `[3 DeltaNet + 1 GQA] x 8` |
| Hidden size | 2560 |
| MLP | Dense SwiGLU, intermediate size 9216 |
| GQA attention | 16 Q heads, 4 KV heads, head_dim 256 |
| DeltaNet | 32 value heads, 16 key heads, k_dim=v_dim=128 |
| Conv kernel | 4, state stores last 3 pre-conv QKV tokens |
| RoPE | Partial RoPE, 25% of head_dim = 64 dims |
| Vocabulary | 248,320 |
| Tied embeddings | Yes |

Derived DeltaNet shapes:

| Tensor | Shape |
| --- | --- |
| `in_proj_qkv.weight` | `[8192, 2560]` |
| `in_proj_z.weight` | `[4096, 2560]` |
| `in_proj_a.weight` | `[32, 2560]` |
| `in_proj_b.weight` | `[32, 2560]` |
| `conv1d.weight` | `[8192, 1, 4]` |
| `recurrent_state_buffer` | `[max_batch, 32, 128, 128]` |
| `conv_state_buffer` | `[max_batch, 8192, 3]` |

## Compatibility

| Instance | Neuron SDK / environment | TP | dtype | seq_len | Status |
| --- | --- | --- | --- | --- | --- |
| `trn2.48xlarge` | PyTorch 2.9 NxDI inference env | 4 | BF16 | 160 | Unit and integration tests pass |

## Status

Validated on Trn2 with TP=4, batch=1, and seq_len=160. TP=2, Trn1, long-context HBM limits, and quantized inference are not validated for this contrib model.

## Compatible Checkpoints

- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

## Usage

```python
import json
import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
)

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/mnt/models/Qwen3.5-4B"
compiled_path = "/mnt/models/qwen35_4b_traced"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=160,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    save_sharded_checkpoint=True,
)

with open(os.path.join(model_path, "config.json")) as f:
    hf_config = json.load(f)
text_config = hf_config.get("text_config", hf_config)
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
config_dict["tie_word_embeddings"] = hf_config.get(
    "tie_word_embeddings",
    text_config.get("tie_word_embeddings", True),
)
if "rope_parameters" in text_config:
    config_dict["rope_theta"] = text_config["rope_parameters"].get(
        "rope_theta", 10000000
    )

config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)

model = NeuronQwen35ForCausalLM(model_path, config)
model.compile(compiled_path)

model = NeuronQwen35ForCausalLM(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gen_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=32,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Testing

CPU unit tests:

```bash
cd contrib/models/Qwen3.5-4B
pytest test/unit -v
```

Trainium integration:

```bash
cd contrib/models/Qwen3.5-4B
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
QWEN35_MODEL_PATH=/home/ubuntu/models/Qwen3.5-4B \
QWEN35_COMPILED_PATH=/home/ubuntu/models/qwen35_4b_traced_trn2 \
QWEN35_TP_DEGREE=4 \
QWEN35_SEQ_LEN=160 \
pytest test/integration/test_model.py --capture=tee-sys -v
```

Validated results on `trn2.48xlarge`:

- Unit tests: `45 passed`
- Integration tests: `9 passed`
- TTFT: `83.2 ms`
- Throughput: `68.1 tok/s`

## Known Limitations

- DeltaNet weights are replicated across TP ranks in v1.
- DeltaNet layers still allocate dummy KV cache through NxDI's normal cache manager.
- MoE, VL, quantization, speculation, and custom hybrid cache cleanup are out of scope.
