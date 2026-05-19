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

The CTE (context encoding) path uses a manual depthwise conv1d loop over K=4 taps. This produces a compact HLO graph that compiles quickly at all batch sizes. The decode path uses `nn.Conv1d` weights directly (dot product at seq_len=1). The TEN404 compiler bug that required this workaround in SDK 2.28 is fixed in SDK 2.29.1, but the manual loop remains for CTE to avoid NCC_IBCG901 (f_packing assertion at output width > 128 tiles) and for faster compilation at higher batch sizes.

### Gated RMSNorm Ordering

Granite applies the gate BEFORE normalization (`norm_before_gate=False` in Mamba2 terminology), unlike Falcon-H1 which applies it after. The `GraniteRMSNormGated` class matches HF exactly: `silu(gate) * x -> RMSNorm -> weight`.

### Parallel Scan (Prefill)

Prefill uses a full-sequence parallel scan via cumulative sum in log-space (L x L weight matrix). This is mathematically equivalent to HF's chunk-based SSD (chunk_size=256) but produces slightly different floating-point results due to BF16 precision and different accumulation order. Average Pearson=0.9968, average Cosine=0.9987 across 10 diverse prompts.

## Validation Results

**Validated:** 2026-05-19
**Configuration:** TP=4, batch_size=7, seq_len=2048, max_context_length=128, bfloat16
**Instance:** trn2.3xlarge (LNC=2, SDK 2.29.1)

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
| Compile time | ~13 min (trn2.48xlarge), ~16 min (trn2.3xlarge) |
| Compiler flags | `-O1 --auto-cast=none --enable-mixed-precision-accumulation` |

### Inference Performance

| Metric | BS=1 | BS=4 | BS=7 (optimal) |
|--------|------|------|----------------|
| Decode throughput | 24.7 tok/s | 44.3 tok/s | **62.8 tok/s** |
| Decode per-token | 40.5 ms | 90.2 ms | 111.6 ms |
| Prefill latency | 307 ms | 588 ms | ~2.0 s |
| Instance | trn2.3xlarge | trn2.3xlarge | trn2.3xlarge |
| Config | TP=4, LNC=2, BF16 | TP=4, LNC=2, BF16 | TP=4, LNC=2, BF16 |

**BS=7 is the optimal batch size** for maximum throughput on trn2.3xlarge. At BS=8 and above, the MoE layer switches from selective expert loading (10/72 experts) to all-expert mode (72/72 experts), causing a 17% throughput regression (53.6 tok/s at BS=8 vs 62.8 tok/s at BS=7). The threshold is `batch_size * top_k / num_experts >= 1.0`, i.e. `BS * 10 / 72 >= 1.0` → BS >= 8 triggers all-expert mode.

**TP=2 is not possible** — at TP=2, the compiler reports 30 GB needed per core vs 24 GB available (NCC_EVRF009). Weights at TP=2 are ~14 GB, Mamba state ~8 GB, plus activation/IO tensors push it over the limit. TP=4 is the minimum for this model on trn2.

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
    batch_size=7,  # BS=7 is optimal (selective expert loading)
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

| Instance Type | SDK 2.29.1 | SDK 2.29 | SDK 2.28 |
|--------------|----------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | **Validated** (62.8 tok/s @ BS=7) | Validated | Validated |
| trn2.48xlarge (TP=4, LNC=2) | Expected | Expected | Validated (identical to 3xlarge) |
| trn1.32xlarge | Not supported | Not supported | Not tested |

> **NxDI 2.29+ requires Trn2 or newer hardware.** For Trn1 support, pin to SDK 2.28.

**Note:** This model requires MoE support (`MoENeuronConfig`) and Mamba state persistence. The NKI selective scan kernel uses chunked SSD by default (`USE_CHUNKED_SSD = True`) for 2.5x faster prefill over the quadratic O(L^2) scan.

## Testing

```bash
# Run with pytest
cd contrib/models/granite-4.0-h-small/
pytest test/integration/test_model.py -v

# Or run directly
python test/integration/test_model.py
```

## Known Limitations

1. **Maximum context length is 128.** This is a hard limit imposed by per-core HBM on trn2. See [Context Length Limits](#context-length-limits) below for details.
2. **Minimum TP=4.** TP=2 exceeds per-core HBM (30 GB needed vs 24 GB limit). TP=4 is the minimum parallelism degree.
3. **No on-device sampling tested** — current validation uses raw logits (`on_device_sampling_config=None`). Enabling on-device sampling for production use needs testing.
4. **MoE expert threshold** — batch sizes >= 8 trigger all-expert mode, reducing throughput. Use BS=7 for maximum throughput.
5. **Full-sequence parallel scan** — the prefill SSM uses chunked SSD (O(L) matmul-based). For very long sequences, the HBM state buffers are the bottleneck, not compute.

## Context Length Limits

**`max_context_length=128` is the maximum on all current trn2 instance types.**

Extensive testing on trn2.48xlarge (32 Neuron devices, 3 TB total HBM) confirmed that L=256 cannot be achieved with any combination of tensor parallelism, LNC configuration, or scan implementation:

| Config | Context | TP | LNC | Result |
|--------|---------|-----|-----|--------|
| Quadratic scan | 128 | 4 | 2 | **Works** — 17.6 tok/s |
| Quadratic scan | 256 | 4 | 2 | Compiler HBM OOM (24.46 GB > 24 GB per-core limit) |
| NKI scan | 256 | 4 | 2 | Compiles but HBM OOM at load (22.9 GB + 1 GB scratchpad needed) |
| NKI scan | 256 | 8 | 1 | Compiles but HBM OOM at load (23.5 GB + 8 GB scratchpad) |

### Root Cause

The 36 Mamba2 layers each require persistent state buffers (`conv_state` and `ssm_state`) that scale with context length. At L=256, the combined per-core I/O tensors (15.4 GB) plus scratchpad (8 GB) exceed the 24 GB per-HBM-bank limit. These state buffers are **per-layer, per-core** — they do not shard with tensor parallelism. Increasing TP from 4 to 8 shards the MoE weights but not the SSM state, so per-core memory stays above the limit.

### Why Other Parallelism Strategies Don't Help

- **Pipeline parallelism (PP):** Would split layers across devices, but `pp_degree > 1` is untested for NxDI inference. Mamba state persistence via `input_output_aliases` has no support for PP boundaries, and decode latency would degrade from inter-device communication at every pipeline stage.
- **Context parallelism (CP):** Splits the sequence dimension across cores. Well-supported in NxDI for attention models (Llama4, Qwen3-MoE). However, CP only helps the 4 attention layers (at indices 5, 15, 25, 35). The 36 Mamba layers use sequential SSM recurrence (`h[t] = A * h[t-1] + B * x[t]`) which cannot be split across the sequence axis without a custom parallel scan implementation — a research-level effort.

### What Would Enable Longer Contexts

1. **Custom parallel scan partitioning** for Mamba SSM state across multiple cores (research effort)
2. **Future Neuron hardware** with larger per-core HBM (>24 GB)
3. **Model architecture changes** — fewer Mamba layers or smaller SSM state dimension would reduce per-core memory

## Source Files

| File | Description | Lines |
|------|-------------|-------|
| `src/modeling_granite.py` | Full model implementation (config, Mamba layer, attention, MoE, model wrapper, state dict conversion) | ~930 |
| `src/__init__.py` | Public exports | ~30 |
| `test/integration/test_model.py` | Integration tests (compile, load, generate, coherence, throughput) | ~260 |

## Example Checkpoints

- **HuggingFace:** `ibm-granite/granite-4.0-h-small`

## Maintainer

- **GitHub:** [@jimburtoft](https://github.com/jimburtoft)

**Last Updated:** 2026-05-19
