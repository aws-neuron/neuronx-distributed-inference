# Contrib Model: Aria (Text Decoder)

NeuronX Distributed Inference implementation of the Aria text decoder model.

## Model Information

- **HuggingFace ID:** `rhymes-ai/Aria`
- **Architecture:** MoE Decoder-only transformer (LLaMA-based)
- **Parameters:** ~3.9B total (text decoder)
- **MoE Configuration:** 64 routed experts, top-6 routing, 2 shared experts
- **Attention:** 20 Q heads, 20 KV heads (MHA), RoPE
- **Layers:** 28 decoder layers
- **License:** Check HuggingFace model card

## Implementation Notes

This port implements only the text decoder component of Aria.
The vision tower and multi-modal projector are not included.

The implementation reuses NXDI's standard MoE infrastructure (`initialize_moe_module`
from `moe_v2`) which handles both routed and shared experts, following the same
pattern as built-in Mixtral and Qwen3-MoE models.

## Validation Results

**Validated:** 2026-03-16

| Metric | Value |
|--------|-------|
| Inference test | PASSED (3/3 prompts coherent) |
| Throughput (TKG) | 4.2 tokens/sec |
| Config | tp=8, batch=1, seq_len=512, bf16 |
| Instance | trn1.32xlarge |

### Inference-Only Test Output

| Prompt | Generated |
|--------|-----------|
| "The capital of France is" | " Paris. The capital of Germany is Berlin..." |
| "In machine learning, neural networks" | " are a set of algorithms, modeled after the human brain..." |
| "def fibonacci(n):" | " if n == 0: return 0 elif..." |

Note: Full HF golden comparison skipped because the reference HF model
(`AriaForConditionalGeneration`) is multimodal with 64 MoE experts,
making CPU-based token generation prohibitively slow.

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-16
