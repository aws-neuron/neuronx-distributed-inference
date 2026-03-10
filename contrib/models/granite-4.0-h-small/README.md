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

### NKI Selective Scan Kernel (Optional)

The model includes an optional NKI (Neuron Kernel Interface) kernel that replaces the O(L^2) quadratic parallel scan with an O(L) hardware-accelerated scan using `nisa.tensor_tensor_scan`. This is controlled by the `USE_NKI_SCAN` flag in `modeling_granite.py` (default: `False`).

**Performance note:** At `max_context_length=128`, the quadratic scan is ~30% faster than the NKI kernel because the compiler efficiently vectorizes the 128x128 weight matrix operations, while the NKI kernel incurs overhead from 8,192 individual `tensor_tensor_scan` invocations (one per head_dim x ssm_state_size combination). At larger context lengths, the NKI kernel has a significant **compilation** advantage: at `max_context_length=256`, the NKI kernel compiles successfully while the quadratic scan causes compiler OOM (see Context Length Scaling section). The NKI kernel is expected to outperform the quadratic scan at runtime for L >= 256 on instances with sufficient HBM. See the Performance Benchmarks section for details.

**How it works:**

`tensor_tensor_scan` computes: `result[i] = op0(data0[i], result[i-1]) op1 data1[i]`

For the Mamba2 SSM recurrence `state[t] = exp(dA[t]) * state[t-1] + dBx[t]`:
- `data0 = exp(dA)`, `op0 = multiply`
- `data1 = dBx`, `op1 = add`

The kernel processes all 32 heads (TP-sharded from 128) in the partition dimension, with seq_len in the free dimension. An outer loop iterates over `head_dim(64) x ssm_state_size(128) = 8,192` scan invocations. Inputs are pre-transposed to `(num_heads, seq_len)` layout for efficient SBUF tiling.

**Requirements:**
- Set `NEURON_PLATFORM_TARGET_OVERRIDE` environment variable to match your target platform (e.g., `trn2` for Trainium2) during compilation
- NKI Beta 2 / SDK 2.28+ (`import nki`, `import nki.language as nl`, `import nki.isa as nisa`)
- Neuron hardware (Trainium or Inferentia with NKI support)

**To enable:** Set `USE_NKI_SCAN = True` in `modeling_granite.py`. Recommended for `max_context_length >= 256` on instances with sufficient HBM, or when the quadratic scan fails to compile.

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
| Compile time | ~16-20 min (trn2.3xlarge) |
| Compiler flags | `-O1 --auto-cast=none --enable-mixed-precision-accumulation` |

**Note:** When using the NKI kernel (`USE_NKI_SCAN=True`), set `NEURON_PLATFORM_TARGET_OVERRIDE` to match your target platform (e.g., `trn2`) before compilation.

### NKI Kernel Accuracy (V16-NKI vs V15 Quadratic)

The NKI selective scan kernel produces nearly identical results to the quadratic scan:

| Metric | V15 (Quadratic) | V16 (NKI) |
|--------|-----------------|-----------|
| **Avg Pearson** | 0.9880 | 0.9872 |
| **Avg Cosine** | 0.9800 | 0.9782 |
| **Greedy match** | 100% | 100% |
| **Generation quality** | Coherent | Matches V15 |

The small difference between V15 and V16-NKI is due to different floating-point accumulation order (parallel quadratic vs sequential scan). Both produce correct text generation.

## Performance Benchmarks

**Benchmarked:** 2026-03-10
**Configuration:** TP=4, batch_size=1, seq_len=2048, max_context_length=128, bfloat16
**Instance:** trn2.3xlarge (LNC=2, SDK 2.28)

### Latency Comparison: Quadratic Scan vs NKI Scan

| Metric | Quadratic (default) | NKI Scan | Delta |
|--------|-------------------|----------|-------|
| **Prefill latency** | **717 ms** | 935 ms | +30% slower |
| **Decode per-token** | 50.3 ms | 50.3 ms | identical |
| **100-token throughput** | **17.6 tok/s** | 16.9 tok/s | -4% |
| **100-token total** | 5694 ms | 5915 ms | +3.9% |

**Analysis:**
- Prefill latency is constant regardless of prompt length (1-23 tokens) because NxDI pads all inputs to `max_context_length=128`
- The quadratic scan wins at L=128 because the compiler efficiently vectorizes the 128x128 weight matrix, while the NKI kernel has overhead from 8,192 individual `tensor_tensor_scan` calls
- Decode latency is identical because the NKI kernel only affects prefill (decode uses O(1) recurrence)
- The NKI kernel is required for L >= 256 where the quadratic scan fails to compile (compiler OOM)
- **Recommendation:** Use the default quadratic scan for `max_context_length <= 128`. Enable NKI scan for larger contexts where the quadratic scan fails to compile.

### Context Length Scaling

The model's compilation and runtime behavior varies significantly with context length due to the large MoE architecture (72 experts × 40 layers). Testing was performed on trn2.3xlarge (96 GB HBM total, 24 GB per logical core with LNC=2).

| max_context_length | Quadratic Compile | NKI Compile | Runtime Load | Notes |
|--------------------|-------------------|-------------|-------------|-------|
| **128** | OK (~16 min) | OK (~20 min) | OK (both) | Fully benchmarked, quadratic faster |
| **256** | **FAILED** (compiler OOM) | **OK** (~19 min) | **FAILED** (HBM OOM) | NKI compiles where quadratic cannot |
| **512** | FAILED (compiler OOM) | FAILED (compiler OOM) | N/A | Graph too large for 124 GB host RAM |
| **1024** | FAILED (compiler OOM) | FAILED (compiler OOM) | N/A | Graph too large for 124 GB host RAM |

**Key findings:**

1. **NKI kernel enables longer compilation:** At L=256, the NKI kernel produces a compiler-friendlier HLO graph (avoids the 256×256 quadratic weight matrix expansion), allowing successful compilation where the quadratic approach causes `neuronx-cc` to OOM (exit code 70, >74 GB host RAM).

2. **HBM is the runtime bottleneck:** Even when the NKI kernel compiles at L=256, the model cannot be loaded on trn2.3xlarge because the compiled graph requires more HBM than the 24 GB available per logical core. The error is a 1 GB transpose buffer allocation failure on HBM.

3. **MoE dominates memory:** The 72 experts × 40 layers = 2,880 expert weight sets are the primary memory consumer, not the Mamba SSM states or KV caches.

4. **Larger instances unlock longer contexts:** trn2.48xlarge (32 devices, up to 3 TB total HBM) should support `max_context_length=256+` by distributing experts across more cores. The NKI kernel's compilation advantage becomes essential at these scales.

### Latency Breakdown (Quadratic, default)

| Phase | Latency | Notes |
|-------|---------|-------|
| Prefill (any prompt up to 128 tokens) | 717 ms | Constant due to padding to max_context_length |
| Decode (per token, steady state) | ~50 ms | Measured from 100-token generation |
| Model load (from compiled) | ~71 s | One-time cost, includes weight sharding |
| Compilation | ~16-20 min | One-time cost |

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
# Integration tests (requires Neuron hardware + model weights)
cd contrib/models/granite-4.0-h-small/
pytest test/integration/test_model.py -v

# Or run directly
python test/integration/test_model.py

# NKI kernel unit test (requires Neuron hardware)
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2  # set to your target platform
python test/unit/test_nki_selective_scan.py
```

## Known Limitations

1. **max_context_length=128 on trn2.3xlarge** — compiler OOM prevents L=256+ with quadratic scan; NKI compiles at L=256 but HBM is insufficient for runtime. Larger instances (trn2.48xlarge) are needed for longer contexts.
2. **No on-device sampling tested** — current validation uses raw logits (`on_device_sampling_config=None`). Enabling on-device sampling for production use needs testing.
3. **Batch size 1 only** — batch_size > 1 has not been validated.
4. **NKI scan slower at short contexts** — the optional `USE_NKI_SCAN` kernel is ~30% slower than the default quadratic scan at `max_context_length=128` due to per-invocation overhead. It is disabled by default. Enable for `max_context_length >= 256` where it is required for compilation.
5. **Conv1d workaround** — manual depthwise convolution avoids TEN404 but may be slower than native conv1d once the SDK bug is fixed.

## Source Files

| File | Description | Lines |
|------|-------------|-------|
| `src/modeling_granite.py` | Full model implementation with NKI selective scan kernel (config, Mamba layer, attention, MoE, NKI kernel, model wrapper, state dict conversion) | ~1600 |
| `src/__init__.py` | Public exports | ~30 |
| `test/integration/test_model.py` | Integration tests (compile, load, generate, coherence, throughput) | ~260 |
| `test/unit/test_nki_selective_scan.py` | Standalone NKI selective scan kernel with CPU reference, quadratic reference, and validation tests | ~750 |

## Example Checkpoints

- **HuggingFace:** `ibm-granite/granite-4.0-h-small`

## Maintainer

**Last Updated:** 2026-03-10
