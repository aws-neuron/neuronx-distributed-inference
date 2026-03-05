# Contrib Model: Qwen3.5-35B-A3B

NeuronX Distributed Inference implementation of Qwen3.5-35B-A3B, a hybrid DeltaNet + Standard Attention + Mixture-of-Experts architecture.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- **Model Type:** Hybrid decoder-only transformer (DeltaNet + GQA + MoE)
- **Parameters:** 35B total, 3B active per token (sparse MoE)
- **License:** Check HuggingFace model card

## Architecture Details

Qwen3.5-35B-A3B is a novel hybrid architecture combining three key components:

- **30 DeltaNet layers** (linear recurrent attention): Gated delta rule recurrence with causal conv1d, using custom NKI kernels for context encoding and PyTorch recurrence for token generation. State is carried between context encoding and token generation via `input_output_aliases` on `nn.Parameter` buffers.
- **10 standard GQA attention layers**: With output gate (sigmoid-gated, head-interleaved in q_proj), partial RoPE (25% of head_dim = 64 of 256), and per-head QK norm.
- **All 40 layers use sparse MoE**: 256 routed experts (top-8) + 1 sigmoid-gated shared expert. Expert weights are pre-fused (gate_up_proj, down_proj).

Key implementation details:

- **NKI DeltaNet kernel** (`nki_deltanet.py`): Custom NKI v2 kernel implementing the gated delta rule recurrence. Processes one (batch, head) pair per kernel call. Returns both output and final recurrent state for CTE-to-TKG state carry-over.
- **Padding-aware recurrence**: NxDI right-pads inputs to bucket size. Padding tokens' decay factor `g` is zeroed so `exp(0) = 1`, preserving recurrent state. Conv state is saved from the last 3 *valid* positions using `torch.gather`.
- **Sigmoid-gated shared expert**: Wraps NxDI's `SharedExperts` with a sigmoid gate (`SigmoidGatedSharedExperts`), since Qwen3.5's shared expert gating differs from NxDI's default additive behavior.
- **RMSNorm weight conversion**: Qwen3.5 uses `(1 + weight)` RMSNorm (weights initialized to 0). Converted to standard `weight` RMSNorm (weights initialized to 1) by adding 1.0 during state dict conversion.

## Validation Results

**Validated:** 2026-03-04
**Configuration:** TP=4, batch_size=1, max_context_length=128, max_new_tokens=32, BF16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Generation ("Paris") | PASS | CTE top-1 = "Paris" (score 17.88) |
| Coherence | PASS | Multi-prompt coherent generation |

### Generation Examples

| Prompt | CTE Top-1 Token | TKG Output (10 tokens) |
|--------|-----------------|------------------------|
| "The capital of France is" | " Paris" (17.88) | "a city called Paris.\nA. correct" |
| "1 + 1 =" | " " (18.13) | " 2\n1 + 1 = 2" |
| "The color of the sky is" | " blue" (18.75) | "a result of the scattering of light by the atmosphere" |

## Usage

```python
import json
import os
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

# Import model classes from src
import sys
sys.path.insert(0, "/path/to/Qwen3.5-35B-A3B/src")
from modeling_qwen35_moe import NeuronQwen35MoeForCausalLM, Qwen35MoeInferenceConfig

model_path = "/path/to/Qwen3.5-35B-A3B/"
compiled_model_path = "/path/to/compiled/"

# Load HF config
with open(os.path.join(model_path, "config.json")) as f:
    full_config = json.load(f)
text_config = full_config.get("text_config", full_config)

# Configure
# IMPORTANT: block_size=2048 works around a blockwise MoE bug in SDK 2.28.
# It forces the forward_all_experts path for small context lengths.
neuron_config = MoENeuronConfig(
    tp_degree=4,
    max_batch_size=1,
    max_context_length=128,
    max_new_tokens=32,
    on_device_sampling_config=None,
    torch_dtype=torch.bfloat16,
    fused_qkv=True,
    moe_tp_degree=4,
    moe_ep_degree=1,
    blockwise_matmul_config={"block_size": 2048},
)

# Merge text_config with NeuronConfig
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
if "rope_parameters" in text_config:
    config_dict["rope_theta"] = text_config["rope_parameters"].get("rope_theta", 10000000)
if config_dict.get("tie_word_embeddings") is None:
    config_dict["tie_word_embeddings"] = False

config = Qwen35MoeInferenceConfig(neuron_config=neuron_config, **config_dict)

# Compile, load, and run
model = NeuronQwen35MoeForCausalLM(model_path=model_path, config=config)

# Remove XLA_DISABLE_FUNCTIONALIZATION (set by torch_neuronx import)
os.environ.pop("XLA_DISABLE_FUNCTIONALIZATION", None)

model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer("The capital of France is", return_tensors="pt")

# Context encoding (prefill)
with torch.no_grad():
    output = model(
        input_ids=inputs["input_ids"],
        position_ids=torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )

logits = output[0] if isinstance(output, tuple) else output.logits
next_token = torch.argmax(logits[:, -1, :], dim=-1)
print(f"Next token: {tokenizer.decode(next_token)}")  # " Paris"
```

## Compatibility Matrix

| Instance/Version | SDK 2.28 | SDK 2.27 and earlier |
|------------------|----------|----------------------|
| trn2 (trn2.3xlarge) | **Validated** | Not tested |
| trn1 | Not tested | Not tested |
| inf2 | Not tested | Not tested |

### Known Issues

- **MoE blockwise bug (SDK 2.28)**: The `forward_blockwise` code path in NxDI's `expert_mlps_v2.py` produces incorrect output on trn2. Workaround: set `blockwise_matmul_config={"block_size": 2048}` to force the `forward_all_experts` path for small context lengths (total_tokens * top_k < block_size).
- **NEURON_PLATFORM_TARGET_OVERRIDE**: Must set `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` when running with NKI v2 `@nki.jit` kernels on trn2.
- **Memory**: The full model in BF16 is ~67GB. trn2.3xlarge (124GB system RAM) can load it but requires careful memory management during compilation.

## Testing

### Environment Setup

```bash
# On trn2.3xlarge with Neuron SDK 2.28
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
```

### Run integration tests

```bash
pytest contrib/models/Qwen3.5-35B-A3B/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/Qwen3.5-35B-A3B
python3 test/integration/test_model.py
```

## Example Checkpoints

- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (67 GB, 14 safetensors shards)

## Files

| File | Description |
|------|-------------|
| `src/modeling_qwen35_moe.py` | Main NxDI model: DeltaNet, attention, MoE, config, state dict converter |
| `src/nki_deltanet.py` | NKI v2 kernels for DeltaNet gated delta rule recurrence |
| `src/__init__.py` | Re-exports public classes |
| `test/integration/test_model.py` | Integration tests (accuracy + performance) |

## Maintainer

Jim Burtoft - AWS
