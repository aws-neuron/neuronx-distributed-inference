# Contrib Model: Qwen3.5-9B

NeuronX Distributed Inference implementation of Qwen3.5-9B, a dense hybrid DeltaNet + GQA decoder from Alibaba Cloud.

This variant is forked from the proven Qwen3.5-2B contrib implementation in PR 141. It keeps the same working cache architecture:

- Standard GQA layers use NxDI `KVCacheManager`.
- DeltaNet layers return dummy KV tensors to satisfy NxDI cache plumbing.
- Real DeltaNet state is carried through layer-local `recurrent_state_buffer` and `conv_state_buffer` side-channel aliases.

## Model Information

| Feature | Value |
| --- | --- |
| HuggingFace ID | `Qwen/Qwen3.5-9B` |
| Model type | `qwen3_5_text` under top-level `qwen3_5` |
| Layers | 32: 24 DeltaNet + 8 GQA |
| Layer pattern | `[3 DeltaNet + 1 GQA] x 8` |
| Hidden size | 4096 |
| MLP | Dense SwiGLU, intermediate size 12288 |
| GQA attention | 16 Q heads, 4 KV heads, head_dim 256 |
| DeltaNet | 32 value heads, 16 key heads, k_dim=v_dim=128 |
| Conv kernel | 4, state stores last 3 pre-conv QKV tokens |
| RoPE | Partial RoPE, 25% of head_dim = 64 dims |
| Vocabulary | 248,320 |
| Tied embeddings | No |

Derived DeltaNet shapes:

| Tensor | Shape |
| --- | --- |
| `in_proj_qkv.weight` | `[8192, 4096]` |
| `in_proj_z.weight` | `[4096, 4096]` |
| `in_proj_a.weight` | `[32, 4096]` |
| `in_proj_b.weight` | `[32, 4096]` |
| `conv1d.weight` | `[8192, 1, 4]` |
| `recurrent_state_buffer` | `[max_batch, 32, 128, 128]` |
| `conv_state_buffer` | `[max_batch, 8192, 3]` |

## Status

This 9B contrib is prepared for bring-up. The implementation should be validated on Trn2 before TP=2 or Trn1 experiments.

Validated baseline:

- Qwen3.5-2B PR 141: trn2.3xlarge, TP=4, LNC=2, SDK 2.29, NKI 0.3.

Unvalidated for this folder until run:

- Qwen3.5-9B compile and generation
- TP=2
- Trn1
- long-context HBM limits

## Usage

```python
import json
import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/mnt/models/Qwen3.5-9B"
compiled_path = "/mnt/models/qwen35_9b_traced"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=128,
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
    text_config.get("tie_word_embeddings", False),
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
cd contrib/models/Qwen3.5-9B
pytest test/unit -v
```

Trainium integration:

```bash
cd contrib/models/Qwen3.5-9B
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-9B \
QWEN35_COMPILED_PATH=/mnt/models/qwen35_9b_traced \
QWEN35_TP_DEGREE=4 \
QWEN35_SEQ_LEN=128 \
pytest test/integration/test_model.py --capture=tee-sys -v
```

## Known Limitations

1. SDK 2.29+ and NKI 0.3 are expected.
2. DeltaNet weights are replicated across TP ranks in v1.
3. Dummy KV wastes HBM for DeltaNet layers.
4. First-token and multi-token logit parity are expected to show the same BF16 recurrent divergence reported by PR 141 until the DeltaNet precision work is done.
5. Hybrid cache, DeltaNet TP sharding, quantization, speculative decoding, and MoE are out of scope for first bring-up.
