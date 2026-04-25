# Contrib Model: Sarvam-30B

NeuronX Distributed Inference implementation of the Sarvam-30B model (`sarvamai/sarvam-30b`), a 32B total / 2.4B active parameter Mixture-of-Experts language model with sigmoid routing, shared experts, and GQA.

## Model Information

- **HuggingFace ID:** `sarvamai/sarvam-30b`
- **Model Type:** Decoder-only transformer (Mixture-of-Experts)
- **Parameters:** ~32B total, ~2.4B active per token (BF16)
- **Architecture:** 128 routed experts + 1 shared expert, top-6 sigmoid routing, GQA (64Q/4KV heads, head_dim=64), RoPE, RMSNorm, QK normalization
- **License:** Apache 2.0
- **Maintainer:** Jim Burtoft (@jimburtoft)

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 19 (1 dense + 18 MoE) |
| Hidden Size | 4096 |
| Attention Heads | 64 |
| KV Heads (GQA) | 4 |
| Head Dim | 64 |
| Routed Experts | 128 |
| Shared Experts | 1 |
| Active Experts (TopK) | 6 |
| Routing | Sigmoid with expert bias and scaling |
| Dense Intermediate | 8192 (layer 0 only) |
| MoE Intermediate | 1024 |
| Max Position Embeddings | 131,072 |
| Vocabulary | 262,144 |
| RoPE θ | 8,000,000 |
| Activation | SiLU gated MLP |

### Key Architecture Features

- **Sigmoid MoE routing** with learned expert bias and routed scaling factor (2.5×)
- **Shared expert** on every MoE layer (extracted as standalone module for NKI fused TKG compatibility)
- **First layer dense**: Layer 0 uses a standard MLP instead of MoE
- **QK normalization**: Per-head RMSNorm on Q and K projections
- **Fused QKV**: Single weight matrix for query, key, value projections

## Validation Results

**Validated:** 2026-04-24
**Instance:** trn2.3xlarge (TP=4, LNC=2)
**SDK:** NxDI 0.9.17334, neuronx-cc 2.24.5133, torch-neuronx 2.9.0.2.13, SDK 2.29

### Benchmark Results

All results: BF16, TP=4, trn2.3xlarge, LNC=2, NKI fused TKG enabled.

#### Sequence Length Scaling (batch_size=1)

| seq_len | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Compile Time |
|---------|-----------|-----------|-------------------|-------------|
| 256 | 152.8 | 7.52 | 102.2 | 71.6s |
| 1024 | 192.4 | 7.68 | 94.6 | 82.6s |
| 4096 | 341.4 | 7.71 | 77.4 | 128.7s |
| 8192 | 559.0 | 8.00 | 60.2 | 227.5s |
| 16384 | 1127.7 | 8.43 | 38.6 | 555.8s |

Maximum compilable sequence length: 16384. At 32768, CTE compilation fails.

#### Batch Size Scaling (seq_len=4096)

| Batch Size | TTFT (ms) | TPOT (ms) | Per-request tok/s | Aggregate tok/s |
|-----------|-----------|-----------|-------------------|-----------------|
| 1 | 341.0 | 7.80 | 76.9 | 76.9 |
| 2 | 528.1 | 11.98 | 49.9 | 99.8 |
| 4 | 921.3 | 15.49 | 33.7 | 134.9 |
| 8 | 1804.3 | 23.68 | 19.4 | 155.3 |

#### NKI Fused TKG Optimization

| Config | Throughput | TPOT | Compile Time |
|--------|-----------|------|-------------|
| Baseline (no NKI) | 82.3 tok/s | 12.15ms | 103s |
| **NKI fused TKG** | **100.0 tok/s** | **~10ms** | **70.1s** |
| Improvement | **+21.5%** | **-18%** | **-32%** |

NKI fused TKG requires extracting shared experts as a standalone module on the decoder layer (the SDK 2.29 `moe_block_tkg` kernel does not support shared experts internally).

#### GPU Comparison (1x NVIDIA H100 80GB, vLLM 0.19.1, BF16)

| Metric | GPU (H100) | Neuron (trn2.3xlarge) |
|--------|-----------|----------------------|
| Decode tok/s (BS=1, short input) | 277.6 | 102.2 |
| TPOT (BS=1) | 3.60ms | 7.52ms |
| Aggregate tok/s (BS=4) | 835.2 | 134.9 |

### Accuracy Validation

**Method:** `check_accuracy_logits_v2` with teacher forcing, comparing full logit distributions against BF16 CPU reference outputs.

| Parameter | Value |
|-----------|-------|
| Tokens validated per prompt | 20 |
| Number of prompts | 5 |
| Tolerance map (all tiers) | atol=1e-5, rtol=1.2 |
| Divergence difference tolerance | 0.30 |
| **Result** | **5/5 PASS** |

Tolerances are wider than typical dense models due to BF16 accumulation across 128-expert × top-6 sigmoid routing dispatch. Token-level agreement (top-1 match) is 100% across all prompts.

## Usage

```python
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

# Import from contrib src
from modeling_sarvam_moe import (
    NeuronSarvamMoEForCausalLM,
    SarvamMoEInferenceConfig,
    load_sarvam_moe_config,
)

MODEL_PATH = "/path/to/sarvamai/sarvam-30b"
COMPILED_PATH = "/path/to/compiled/artifacts"

# Configure for NKI fused TKG (recommended)
neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    max_context_length=4096,
    seq_len=4096,
    on_device_sampling_config={"top_k": 1},
    torch_dtype="bfloat16",
    fused_qkv=True,
    glu_mlp=True,
    blockwise_matmul_config={
        "use_shard_on_intermediate_dynamic_while": True,
        "skip_dma_token": True,
    },
    moe_fused_nki_kernel_enabled=True,
    expert_mlp_nki_kernel_enabled=True,
    router_topk_nki_kernel_enabled=False,  # ISA fallback for sigmoid routing
)

config = SarvamMoEInferenceConfig(
    neuron_config=neuron_config,
    load_config=load_sarvam_moe_config(MODEL_PATH),
)

model = NeuronSarvamMoEForCausalLM(MODEL_PATH, config)
model.compile(compiled_model_path=COMPILED_PATH)
model.load(COMPILED_PATH)

# Generate
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
generation_model = HuggingFaceGenerationAdapter(model)
gen_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=128,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
)

prompt = "Explain quantum entanglement in simple terms."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
tokens = tokenizer(text, return_tensors="pt")

output = generation_model.generate(
    tokens.input_ids,
    generation_config=gen_config,
    attention_mask=tokens.attention_mask,
)
print(tokenizer.decode(output[0][tokens.input_ids.shape[1]:], skip_special_tokens=True))
```

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 |
|----------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | **VALIDATED** | Not tested |

## Example Checkpoints

* [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) — requires `trust_remote_code=True`

## Testing Instructions

```bash
# Set paths
export SARVAM_MODEL_PATH="/path/to/sarvamai/sarvam-30b"
export SARVAM_COMPILED_PATH="/path/to/compiled/artifacts"
export SARVAM_GOLDEN_PATH="/path/to/golden_references.pt"

# Run integration tests
pytest test/integration/test_sarvam_moe.py -v --capture=tee-sys
```

### Generating golden references

Golden CPU reference logits must be generated before running accuracy tests.
This requires a machine with ~64 GB RAM for BF16 model loading:

```bash
python src/generate_golden_logits.py \
    --model-path /path/to/sarvamai/sarvam-30b \
    --output /path/to/golden_references.pt
```

### Compiling for validation

The compiled artifacts for accuracy testing must use `on_device_sampling_config=None`
so that `check_accuracy_logits_v2` can access raw logits via `output_scores=True`.

## Implementation Notes

### Shared Expert Extraction (Trinity Pattern)

The SDK 2.29 `moe_block_tkg` NKI kernel does not support shared experts. This implementation extracts shared experts from the MoE module to a standalone `NeuronSarvamSharedExpert` on the decoder layer, following the same pattern used in the Trinity contrib model.

During TKG forward (seq_len==1) with the fused kernel, the decoder layer:
1. Applies `post_attention_layernorm` to get normed hidden states for the shared expert
2. Runs the MoE module (routed experts only, via fused NKI kernel)
3. Runs the shared expert separately
4. Adds shared expert output to MoE output

The state dict remapping handles:
- `mlp.shared_experts.*` → `shared_expert.*` on the decoder layer
- Cloned `post_attention_layernorm.weight` → `mlp.moe_fused_tkg.post_attention_layernorm.weight`
- Transposed router weight → `mlp.router.weight_T`

### Sigmoid Routing Patch

The sigmoid routing with expert bias uses an ISA fallback (`router_topk_nki_kernel_enabled=False`) because the NKI router kernel expects softmax routing. A runtime patch (`_patch_fused_tkg_for_sigmoid`) applies the sigmoid + bias + scaling logic at model initialization.

### Dense First Layer

Layer 0 uses a standard dense MLP (intermediate_size=8192) instead of MoE. The `NeuronSarvamDenseMLP` module handles this transparently through the standard `glu_mlp` path.

## Known Issues

- **32K sequence length fails**: CTE compilation crashes at seq_len=32768 (neuronx-cc limitation, not HBM). Maximum compilable sequence length is 16384.
- **BF16 logit divergence**: MoE routing with 128 experts × top-6 in BF16 accumulates numerical drift faster than dense models. Logit validation requires wider tolerances (rtol=1.2) but token-level accuracy is 100%.
- **Compilation time scales with batch size**: BS=8 at seq_len=4096 takes ~10 minutes to compile vs ~2 minutes for BS=1.
