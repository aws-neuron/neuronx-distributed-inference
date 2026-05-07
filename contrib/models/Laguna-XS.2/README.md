# Contrib Model: Laguna-XS.2

NeuronX Distributed Inference implementation of Laguna-XS.2, a 33B-parameter Mixture-of-Experts model with 3B active parameters per token, designed for agentic coding tasks.

## Model Information

- **HuggingFace ID:** `poolside/Laguna-XS.2`
- **Model Type:** Decoder-only transformer (MoE)
- **Parameters:** 33B total / 3B active (256 routed experts + 1 shared expert, top-8 routing)
- **Architecture:** Mixed SWA/Full attention, GQA, RoPE (YaRN + default), Softplus attention gating, Sigmoid MoE routing
- **License:** Apache 2.0
- **Maintainer:** Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))

## Architecture Highlights

Laguna-XS.2 has several novel features not found in standard NxDI models:

| Feature | Description |
|---------|-------------|
| **Softplus Attention Gating** | Per-head gating via `F.softplus(g_proj(hidden_states))` — gates attention output before residual |
| **Variable GQA Heads** | 48 Q-heads (full-attention layers) vs 64 Q-heads (SWA layers), KV=8 constant |
| **Mixed Attention** | 10 full-attention layers + 30 sliding-window layers (window_size=4096) |
| **Dual RoPE** | YaRN (factor=32, max_position=131072) for full-attn, default for SWA |
| **Sigmoid MoE Routing** | Sigmoid activation + L1 normalization + `e_score_correction_bias` for expert selection |
| **MoE Scaling** | `routed_output *= 2.5` then `result = routed_output + shared_expert_output` |

## Validation Results

**Validated:** 2026-05-05
**Instance:** trn2.3xlarge (LNC=2, 4 NeuronCores)
**SDK:** Neuron SDK 2.29 (torch-neuronx 2.9.0, neuronx-cc 2.24, NxDI 0.9.17334)

### Benchmark Results

| Batch Size | Sequence Length | Throughput (tok/s) | TPOT (ms) |
|:----------:|:--------------:|:------------------:|:----------:|
| 1 | 8192 | 91 | 11.0 |
| 4 | 4096 | 223 | 4.5 |
| 8 | 2048 | 310 | 3.2 |

**Notes:**
- TP=4, BF16 precision
- Max single-bucket CTE: 8192 tokens (instruction limit at 16K+)
- Recommended production config: BS=4, seq_len=4096

### Accuracy Validation

Logit validation using the NxDI `logit_validation()` framework against pre-computed CPU reference logits:

| Mode | Tokens Validated | Top-5 Tolerance | Result |
|------|:----------------:|:---------------:|:------:|
| CTE (context encoding) | 1 | (1e-5, 0.01) | PASS |
| TKG (token generation) | 32 | (1e-5, 0.01) | PASS |

## Usage

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

# Add contrib model to path
import sys
sys.path.insert(0, "contrib/models/Laguna-XS.2")
from src.modeling_laguna import NeuronLagunaForCausalLM, LagunaInferenceConfig

# Configuration
model_path = "/path/to/Laguna-XS.2"
compiled_path = "/path/to/laguna-compiled"

neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=4096,
    n_positions=[4096],
    on_device_embedding=True,
    on_device_generation=True,
    fused_rmsnorm=True,
    use_torch_block_wise=True,
)

config = LagunaInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)

# Build and compile
model = NeuronLagunaForCausalLM(compiled_path, config)
model.compile(serialize=True)
model.load(compiled_path)
model.to_neuron()

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer.encode("def fibonacci(n):", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0]))
```

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 |
|----------|:--------:|:--------:|
| trn2.3xlarge (TP=4, LNC=2) | **VALIDATED** | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| inf2 / trn1 | Not supported (NxDI 0.9.x dropped Trn1/Inf2) | Not tested |

**Memory:** ~66.6 GB BF16 model weights. Fits trn2.3xlarge TP=4 (96 GB HBM) with headroom for KV cache up to 131K context.

## Example Checkpoints

* [poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2)

## Testing Instructions

### Prerequisites

1. trn2.3xlarge instance with Neuron SDK 2.29
2. Model weights downloaded to `/mnt/models/Laguna-XS.2/`
3. Pre-computed reference logits at `/mnt/models/laguna_reference_logits.pt`

### Generate Reference Logits

Reference logits must be generated using `transformers >= 5.7.0` (the model requires `trust_remote_code=True` which needs the latest transformers):

```bash
# In a separate venv with transformers >= 5.7.0
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('poolside/Laguna-XS.2', torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('poolside/Laguna-XS.2')

prompt = 'def fibonacci(n):'
input_ids = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

torch.save({'input_ids': input_ids, 'logits': logits}, '/mnt/models/laguna_reference_logits.pt')
"
```

### Run Tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd /mnt/models/neuronx-distributed-inference
export PYTHONPATH=src:contrib/models/Laguna-XS.2
export LAGUNA_MODEL_PATH=/mnt/models/Laguna-XS.2
export LAGUNA_COMPILED_PATH=/mnt/models/laguna-compiled
export LAGUNA_TP_DEGREE=4

# Basic integration test (compile + generate)
python contrib/models/Laguna-XS.2/test/integration/test_laguna.py

# Logit validation (CTE only)
python contrib/models/Laguna-XS.2/test/integration/test_logit_validation.py --cte-only

# Full logit validation (CTE + TKG)
python contrib/models/Laguna-XS.2/test/integration/test_logit_validation.py
```

## Known Issues

1. **Max CTE bucket size: 8192 tokens.** Context lengths above 8192 hit Neuron compiler instruction limits. Use chunked prefill or shorter prompts for production.

2. **Requires `transformers >= 5.7.0` for reference generation.** The HuggingFace model uses `trust_remote_code=True` with custom modeling code that requires the latest transformers. The NxDI implementation itself has no such dependency.

3. **TKG mega-kernel not fused with softplus gating.** The standard NxDI attention TKG mega-kernel does not support softplus gating natively. Gating is applied separately after the attention kernel, adding one extra operation per layer during token generation.

4. **Sigmoid routing NKI kernel.** The MoE fused TKG NKI kernel natively supports sigmoid routing (SDK 2.29). No workaround needed.
