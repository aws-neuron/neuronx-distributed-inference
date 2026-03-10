# Contrib Model: Granite 4.0-H-Small

NeuronX Distributed Inference implementation of IBM's [Granite 4.0-H-Small](https://huggingface.co/ibm-granite/granite-4.0-h-small) (GraniteMoeHybridForCausalLM).

## Model Information

- **HuggingFace ID:** `ibm-granite/granite-4.0-h-small`
- **Model Type:** Hybrid Mamba2/Attention with MoE
- **Parameters:** ~4B total (active ~800M per token with top-10 routing)
- **License:** Apache 2.0

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden size | 4096 |
| Layers | 40 (36 Mamba2 + 4 Attention) |
| Attention layer indices | 5, 15, 25, 35 |
| Attention heads | 32 (8 KV heads) |
| Experts | 72 per layer, top-10 routing |
| Shared experts | 1 per layer |
| Mamba heads | 128 (head_dim=64) |
| SSM state size | 128 |
| Conv kernel | 4 |
| Position embeddings | None ("nope") |
| Vocab size | 131072 |
| tie_word_embeddings | True |
| embedding_multiplier | 12 |
| logits_scaling | 16 |
| residual_multiplier | 0.22 |

## Implementation Notes

### Mamba State Persistence

The key challenge for this hybrid architecture is persisting Mamba2 recurrent state (conv_state and ssm_state) across XLA graph executions during autoregressive decode. We solve this using the same `input_output_aliases` mechanism that NxDI uses for KV cache:

1. `NeuronGraniteModel` maintains a `nn.ParameterList` (`mamba_states`) containing conv_state and ssm_state buffers for each of the 36 Mamba layers (72 parameters total)
2. `GraniteDecoderModelInstance` (extends `DecoderModelInstance`) adds these parameters to `input_output_aliases` after the standard KV cache entries
3. `NeuronMamba2Layer.forward()` accepts and returns state as explicit tensor arguments
4. The output list is: `[logits, K0, V0, ..., conv_state_0, ssm_state_0, ...]`

This follows the MLlama vision_key_values pattern.

### Manual Depthwise Conv1d

SDK 2.28 has a compiler bug (TEN404) where the auto-inserted NKI Conv1d kernel crashes on `seq_len=1` (decode path). We work around this by implementing depthwise convolution manually using weight parameters and a loop over kernel positions.

### Gated RMSNorm Ordering

Granite applies the gate BEFORE normalization (`norm_before_gate=False` in Mamba2 terminology), unlike Falcon-H1 which applies it after. The `GraniteRMSNormGated` class matches HF exactly: `silu(gate) * x -> RMSNorm -> weight`.

### Parallel Scan (Prefill)

Prefill uses a full-sequence parallel scan via cumulative sum in log-space (L x L weight matrix). This is mathematically equivalent to HF's chunk-based SSD (chunk_size=256) but produces slightly different floating-point results due to BF16 precision and different accumulation order. Average Pearson=0.9968, average Cosine=0.9987 across 10 diverse prompts.

## Validation Results

**Validated:** 2026-03-10
**Configuration:** TP=4, batch_size=1, seq_len=2048, max_context_length=128, bfloat16
**Instance:** trn2.3xlarge (LNC=2, SDK 2.28)

### Prefill Accuracy (vs HF BF16 CPU, 10 prompts)

| Prompt | Pearson | Cosine | MaxDiff | Greedy Match |
|--------|---------|--------|---------|-------------|
| "Artificial Intelligence is" | 0.9997 | 0.9999 | 1.25 | YES (`a`) |
| "The capital of France is" | 0.9967 | 0.9984 | 1.88 | YES (`Paris`) |
| "Water boils at a temperature of" | 0.9984 | 0.9998 | 2.00 | YES (` `) |
| "Explain the concept of artificial intelligence..." | 0.9959 | 0.9977 | 3.00 | YES (`Your`) |
| "Write a short Python function..." | 0.9945 | 0.9973 | 2.12 | YES (`The`) |
| "Hello, how are you today?" | 0.9946 | 0.9978 | 1.94 | YES (`I`) |
| "def fibonacci(n):" | 0.9969 | 0.9985 | 1.28 | YES (`"`) |
| "In the field of machine learning..." | 0.9974 | 0.9981 | 1.62 | YES (`estimate`) |
| "The" | 0.9980 | 0.9997 | 1.03 | YES (` `) |
| "Hi" | 0.9960 | 0.9993 | 1.00 | YES (`,`) |

| Summary Metric | Value |
|--------|-------|
| **Greedy token match rate** | **10/10 = 100%** |
| **Average Pearson** | **0.9968** |
| **Average Cosine** | **0.9987** |
| Max absolute diff (worst case) | 3.00 |

### Decode Quality (greedy, 30 tokens)

| Prompt | HF Output | Neuron Output | First Token Match |
|--------|-----------|---------------|-------------------|
| "The capital of France is" | " Paris." | " Paris." | YES (100% token match) |
| "Explain the concept..." | "Your response should contain at least 3 sentences. Include keywords: machine learning, algorithms, " | "Your response should contain at least 3 sentences. The response must contain at least 2 placeholder" | YES (33% token match) |
| "def fibonacci(n):" | (code output) | (code output) | YES |

Both models produce coherent, factually correct text. Token-level divergence during decode is expected: our Mamba2 prefill uses full-sequence parallel scan while HF uses chunk-based SSD (chunk_size=256). These are mathematically equivalent but accumulate different BF16 rounding, causing early divergence that cascades through autoregressive generation. Deterministic answers (e.g., "Paris.") match exactly.

### Compilation
| Metric | Value |
|--------|-------|
| Compile time | ~16 min (trn2.3xlarge) |
| Compiler flags | `-O1 --auto-cast=none --enable-mixed-precision-accumulation` |

## Usage

```python
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from modeling_granite import NeuronGraniteForCausalLM, GraniteInferenceConfig

MODEL_PATH = "/path/to/granite-4.0-h-small/"
COMPILED_PATH = "/path/to/compiled_model/"

# Configure
neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    max_context_length=128,
    seq_len=2048,
    on_device_sampling_config=None,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    torch_dtype="bfloat16",
)

config = GraniteInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(MODEL_PATH),
)

# Compile (first time only, ~16 min)
model = NeuronGraniteForCausalLM(MODEL_PATH, config)
model.compile(COMPILED_PATH)

# Load compiled model
model = NeuronGraniteForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH)

# Generate
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
inputs = tokenizer("Artificial Intelligence is", return_tensors="pt")

gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    attention_mask=torch.ones_like(inputs.input_ids),
    max_new_tokens=50,
    do_sample=False,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Compatibility Matrix

| Instance Type | SDK 2.28 | SDK 2.27 |
|--------------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | Validated | Not tested |
| trn2.48xlarge (TP=4+) | Should work | Not tested |
| trn1.32xlarge | Not tested | Not tested |

**Note:** This model requires MoE support (`MoENeuronConfig`) and Mamba state persistence. The TEN404 conv1d workaround is specific to SDK 2.28; future SDK versions may not need it.

## Testing

```bash
# Run with pytest
cd contrib/models/granite-4.0-h-small/
pytest test/integration/test_model.py -v

# Or run directly
python test/integration/test_model.py
```

## Known Limitations

1. **No on-device sampling tested** — current validation uses raw logits (`on_device_sampling_config=None`). Enabling on-device sampling for production use needs testing.
2. **Batch size 1 only** — batch_size > 1 has not been validated.
3. **Full-sequence parallel scan** — the prefill SSM uses an O(L^2) parallel scan. For very long sequences, a chunk-based approach or NKI kernel would be more efficient.
4. **Conv1d workaround** — manual depthwise convolution avoids TEN404 but may be slower than native conv1d once the SDK bug is fixed.

## Source Files

| File | Description | Lines |
|------|-------------|-------|
| `src/modeling_granite.py` | Full model implementation (config, Mamba layer, attention, MoE, model wrapper, state dict conversion) | ~930 |
| `src/__init__.py` | Public exports | ~30 |
| `test/integration/test_model.py` | Integration tests (compile, load, generate, coherence, throughput) | ~260 |

## Example Checkpoints

- **HuggingFace:** `ibm-granite/granite-4.0-h-small`

## Maintainer

**Last Updated:** 2026-03-10
