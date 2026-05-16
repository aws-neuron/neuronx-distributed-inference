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

**Validated:** 2026-03-17

| Metric | Value |
|--------|-------|
| Teacher-forced match | **98.12%** (PASSED) |
| Greedy match | 72.81% |
| Throughput (TKG) | 4.2 tokens/sec |
| Config | tp=8, batch=1, seq_len=512, bf16 |
| Instance | trn1.32xlarge |
| HF golden reference | `AriaForConditionalGeneration` (text-only input) |

### Token Match Details (10 prompts, 32 tokens each)

| Prompt | Greedy | Teacher-Forced |
|--------|--------|----------------|
| "The theory of general relativity..." | 90.6% | 96.9% |
| "The French Revolution began in..." | 100.0% | 100.0% |
| "To solve a quadratic equation..." | 100.0% | 100.0% |
| "Once upon a time in a distant galaxy..." | 12.5% | 96.9% |
| "def fibonacci(n):..." | 100.0% | 100.0% |
| "The Amazon River flows through..." | 96.9% | 93.8% |
| "The concept of free will..." | 28.1% | 96.9% |
| "To make a cup of coffee, first..." | 0.0% | 96.9% |
| "List three benefits of regular exercise..." | 100.0% | 100.0% |
| "If all roses are flowers..." | 100.0% | 100.0% |

Note: HF golden uses full `AriaForConditionalGeneration` with text-only input
(no pixel_values). `AriaTextForCausalLM` cannot be loaded directly due to
weight prefix mismatch (`language_model.*` in checkpoint). Greedy divergence
is expected for MoE models in bf16 due to cascading expert routing differences.

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Performance

Profiled on trn1.32xlarge (single NeuronCore utilization):

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| Throughput | - | 4.2 tok/s |
| MBU (Memory) | 1.4% | 2.8% |
| MFU (Compute) | 0.9% | 0.0% |

*Batch size 1, sequence length 512, BF16 precision, TP=8*

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-17
