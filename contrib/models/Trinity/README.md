# Contrib Model: Trinity

NeuronX Distributed Inference implementation of the Trinity model family (AfmoeForCausalLM) from Arcee AI. A single unified implementation supports all three model sizes.

## Model Family

| Model | HuggingFace ID | Total Params | Active Params | Instance |
|-------|----------------|-------------|---------------|----------|
| **Nano** | `arcee-ai/Trinity-Nano-Preview` | ~6B | ~1B | inf2.8xlarge / trn2.3xlarge |
| **Mini** | `arcee-ai/Trinity-Mini` | ~26B | ~4.5B | trn2.3xlarge (TP=4) |
| **Large** | `arcee-ai/Trinity-Large-Preview` | ~250B | ~15B | trn2.48xlarge (TP=64) |

**License:** Apache 2.0

## Architecture Details

| Feature | Nano | Mini | Large |
|---------|------|------|-------|
| Layers | 56 (2 dense + 54 MoE) | 32 (2 dense + 30 MoE) | 60 (6 dense + 54 MoE) |
| Hidden Size | 1024 | 2048 | 3072 |
| Attention Heads | 8 | 32 | 48 |
| KV Heads (GQA) | 2 | 4 | 8 |
| Head Dim | 128 | 128 | 128 |
| Experts per MoE layer | 128 | 128 | 256 |
| Active Experts (TopK) | 8 | 8 | 4 |
| Shared Experts | 1 | 1 | 1 |
| Dense Intermediate | 3072 | 6144 | 12288 |
| MoE Intermediate | 256 | 1024 | 3072 |
| Sliding Window | 2048 | 2048 | 4096 |
| Max Position Embeddings | 131,072 | 131,072 | 262,144 |
| Vocabulary | 200,192 | 200,192 | 200,192 |
| Routing | Sigmoid + normalize (scale baked into weights) |
| Activation | SiLU gated MLP (`glu_type="glu"`) |
| Position Encoding | RoPE (sliding attention layers only) |
| Normalization | RMSNorm (4 per layer) |

### Unique Architecture Features

- **Mixed Attention:** Alternating sliding window and full attention (every 4th layer)
- **Gated Attention:** Sigmoid gate applied to attention output before o_proj
- **QK Normalization:** Per-head RMSNorm on Q and K
- **muP Scaling:** Embedding output scaled by hidden_size^0.5
- **Expert Bias:** Learned bias added to routing scores for expert selection
- **Conditional RoPE:** Rotary embeddings applied only to sliding attention layers

## Validation Results

**Validated:** 2026-02-26
**SDK:** NxDI 0.7.15063, neuronx-cc 2.22.12471, torch-neuronx 2.9.0.2.11, transformers 4.56.2

All results below are from the **unified `modeling_trinity.py`** (this code).

### Trinity-Nano on trn2.3xlarge (TP=2, LNC=2)

| Metric | Result |
|--------|--------|
| Compilation Time | 5.1 min |
| Load Time | 2.2 min |
| Forward Pass Latency | ~0.50s |

**First-token predictions:**

| Prompt | Top-1 Token | Logit | Top-5 |
|--------|-------------|-------|-------|
| "Hello, how are you?" | I | 17.75 | I, Hello, How |
| "Explain quantum computing in simple terms." | Answer | 21.00 | Answer, Quantum, What |
| "Write a Python function that calculates the Fibonacci sequence." | The | 24.75 | The, Your, Additionally |

**Generation (5 tokens):**
- "Hello, how are you?" -> "I am fine, thank"
- "Explain quantum computing in simple terms." -> "Answer: Quantum computing uses"

### Trinity-Mini on trn2.3xlarge (TP=4, LNC=2)

| Metric | Result |
|--------|--------|
| Compilation Time | 4.9 min |
| Load Time | 4.1 min (from pre-compiled) |
| Forward Pass Latency | ~0.37s |

**First-token predictions:**

| Prompt | Top-1 Token | Logit | Top-5 |
|--------|-------------|-------|-------|
| "Hello, how are you?" | I | 20.12 | I, This, My |
| "Explain quantum computing in simple terms." | What | 20.75 | What, How, Quantum |
| "Write a Python function that calculates the Fibonacci sequence." | The | 28.00 | The, Your, It |

**Generation (5 tokens):**
- "Hello, how are you?" -> "I'm fine, thank"
- "Explain quantum computing in simple terms." -> "What are the key differences"

### Trinity-Nano on inf2.8xlarge (TP=1, no LNC)

| Metric | Result |
|--------|--------|
| Compilation Time | Reused from trn2.3xlarge |
| Load Time | 47.7s |
| Forward Pass Latency | ~0.73s |

**Note:** inf2.xlarge (16GB system RAM) cannot run Nano -- OOM killed at 15.3GB RSS during weight loading. inf2.8xlarge (123GB system RAM) works with TP=1. NxDI auto-converts GQA to MHA when `TP=1` and `num_kv_heads=2`.

### Trinity-Large on trn2.48xlarge (TP=64, LNC=2)

| Metric | Result |
|--------|--------|
| Compilation Time | 8.6 min |
| Load Time | 15.6 min |
| Forward Pass Latency | ~1.15s |

**First-token predictions:**

| Prompt | Top-1 Token |
|--------|-------------|
| "Hello, how are you?" | I |
| "Explain quantum computing in simple terms." | Quantum |
| "Write a Python function that calculates the Fibonacci sequence." | The |

**Notes:**
- TP=32 is insufficient -- sharded weights consume ~23.5GB per logical NeuronCore, exceeding the ~24GB HBM per physical NC and leaving no room for scratchpad/KV cache. TP=64 (all 64 logical cores on trn2.48xlarge) is required.
- Model is ~516GB on disk (31 safetensors in bf16). Root EBS volume (600GB) is insufficient -- NVMe instance store is required for model storage (`/mnt/nvme/`).
- Set `TMPDIR`, `BASE_COMPILE_WORK_DIR`, and `NEURON_COMPILE_CACHE_URL` to NVMe paths to avoid filling root disk during compilation.

## Usage

### Trinity-Nano-Preview (~6B total, ~1B active)

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_trinity import NeuronTrinityForCausalLM, TrinityInferenceConfig

model_path = "/path/to/arcee-ai/Trinity-Nano-Preview/"
compiled_path = "/path/to/compiled-nano/"

neuron_config = MoENeuronConfig(
    tp_degree=2,       # Nano is small enough for TP=2
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
)

config = TrinityInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config
)

model = NeuronTrinityForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, padding_side="right", trust_remote_code=True
)
```

**Instance:** inf2.8xlarge (TP=1) or trn2.3xlarge (TP=2). Does NOT fit inf2.xlarge (16GB system RAM causes OOM).

### Trinity-Mini (~26B total, ~4.5B active)

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_trinity import NeuronTrinityForCausalLM, TrinityInferenceConfig

model_path = "/path/to/arcee-ai/Trinity-Mini/"
compiled_path = "/path/to/compiled-mini/"

neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
)

config = TrinityInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config
)

model = NeuronTrinityForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, padding_side="right", trust_remote_code=True
)
```

**Instance:** trn2.3xlarge (TP=4). Does NOT fit inf2.8xlarge (~48GB bf16).

### Trinity-Large-Preview (~250B total, ~15B active)

```python
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_trinity import NeuronTrinityForCausalLM, TrinityInferenceConfig

model_path = "/path/to/arcee-ai/Trinity-Large-Preview/"
compiled_path = "/path/to/compiled-large/"

neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=4096,
    torch_dtype=torch.bfloat16,
)

config = TrinityInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config
)

model = NeuronTrinityForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, padding_side="right", trust_remote_code=True
)
```

**Instance:** trn2.48xlarge only (TP=64, capacity block required, NVMe instance store for model storage).

## Caveats

1. **`padding_side="right"` required** -- NKI flash attention kernel does not support left-padding. Always set `padding_side="right"` on the tokenizer.

2. **MoE v2 bf16 accumulation** -- The NxDI MoE v2 NKI kernel accumulates in bf16, causing ~23x more divergence per MoE layer compared to dense layers. Full-vocab cosine similarity is ~0.936, but top-1 token accuracy is preserved. A fix ticket has been filed.

3. **`trust_remote_code=True` required** -- Trinity uses a custom `AfmoeForCausalLM` architecture not in standard transformers. The HuggingFace download requires `trust_remote_code=True`.

4. **transformers version sensitivity** -- Use transformers 4.56.2 with SDK 2.27. Reference outputs may vary across transformers versions.

5. **GLU type** -- Trinity uses `SiLU(gate) * up` which maps to NxDI's `glu_type="glu"`, NOT `"swiglu"`. This is handled automatically by the config class.

6. **route_scale baked into weights** -- NxDI MoE v2 does not support `route_scale` natively. The scale is baked into routed expert `down_proj` weights during weight conversion. Shared expert weights are NOT scaled.

7. **Gate padding at high TP** -- When `num_attention_heads` is not evenly divisible by `tp_degree` (e.g., Large at TP=64: 48/64), gate weights are padded with interleaved layout matching the Q projection. This is handled automatically during weight conversion.

## Compatibility Matrix

| Model | Instance | TP | LNC | Status |
|-------|----------|-----|-----|--------|
| Nano | inf2.xlarge | 1 | N/A | FAIL (16GB system RAM OOM) |
| Nano | inf2.8xlarge | 1 | N/A | Validated |
| Nano | trn2.3xlarge | 2 | 2 | Validated |
| Mini | inf2.8xlarge | -- | -- | Does NOT fit |
| Mini | trn2.3xlarge | 4 | 2 | Validated |
| Large | trn2.48xlarge | 32 | 2 | FAIL (HBM OOM per NC) |
| Large | trn2.48xlarge | 64 | 2 | Validated |

### Minimum Requirements by Model Size

| Model | Min HBM | Min TP | Min Instance |
|-------|---------|--------|-------------|
| Nano | ~12GB bf16 | 1 | inf2.8xlarge (123GB system RAM required) |
| Mini | ~48GB bf16 | 4 | trn2.3xlarge |
| Large | ~500GB bf16 | 64 | trn2.48xlarge (capacity block, NVMe storage) |

### SDK Configuration

| Component | Version |
|-----------|---------|
| NxDI | 0.7.15063 |
| neuronx-cc | 2.22.12471 |
| torch-neuronx | 2.9.0.2.11 |
| torch | 2.9.0 |
| transformers | 4.56.2 |
| Venv | `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/` |

## Testing

```bash
# Set paths for your model
export TRINITY_MODEL_PATH="/path/to/model"
export TRINITY_COMPILED_PATH="/path/to/compiled"

# Run integration tests
pytest test/integration/test_trinity.py --capture=tee-sys

# Or run directly
python test/integration/test_trinity.py
```

**Prerequisites:**
- Pre-compiled model at `TRINITY_COMPILED_PATH`
- HuggingFace model weights downloaded with `trust_remote_code=True`
- Appropriate instance for model size (see Compatibility Matrix)

## Key Porting Challenges

This model required solving several non-trivial porting challenges:

1. **GLU type mismatch:** Trinity uses `SiLU(gate)*up` which maps to NxDI's `"glu"` type, NOT `"swiglu"` (`gate*SiLU(gate)*up`).
2. **Gated attention:** Trinity applies `sigmoid(gate(input))` to attention output before o_proj. Solved via inline override of attention forward methods (required for Neuron tracer compatibility).
3. **Dual intermediate sizes:** Dense layers use `intermediate_size`, MoE experts use `moe_intermediate_size`. Config swaps values for MoE module compatibility.
4. **route_scale not supported by NxDI MoE v2:** Baked into expert `down_proj` weights during conversion.
5. **expert_bias not supported by NxDI:** Created custom `RouterTopKWithBias` subclass.
6. **Conditional RoPE:** Only sliding attention layers get rotary embeddings.
7. **Mixed attention masks:** Framework provides both global and local masks; decoder layer selects based on layer type.
8. **Gate weight padding at high TP:** Interleaved padding matching Q projection layout (prevents wrong-head gating on 54/64 cores).
9. **Shared expert weight loading:** Standalone module for reliable weight mapping vs NxDI built-in shared expert handling.

## NKI Kernels

The NxDI framework uses several NKI (Neuron Kernel Interface) kernels during Trinity compilation and inference. These are hardware-accelerated kernels that execute directly on Neuron cores.

| Kernel | Source | Purpose |
|--------|--------|---------|
| **Flash Attention (Context Encoding)** | `neuronxcc.nki._pre_prod_kernels.attn_fwd` | Full-sequence attention during context encoding (prompt processing). Fused QKV attention with causal masking and sliding window support. |
| **Flash Attention ISA** | `neuronxcc.nki.kernels.attention.attention_isa_kernel` | ISA-level flash attention implementation used as BIR (Built-in Runtime) fallback for context encoding. |
| **Token Gen Attention** | `neuronxcc.nki._private_kernels.attention.attention_tkg_fwd_isa_kernel` | Single-token attention with KV cache lookup during autoregressive token generation. |
| **Token Gen Attention Block (Fused)** | `neuronxcc.nki._pre_prod_kernels.attention_token_gen.llama3_nki_attention_block_token_gen_kernel` | Fused kernel combining attention + RMSNorm + residual connection for token generation. Used when `attn_block_tkg_nki_kernel_enabled` is true. |
| **Blockwise Matmul (MoE Experts)** | `neuronx_distributed.modules.moe.blockwise.BlockwiseMatmulNKIFunc` | Expert MLP computation in MoE layers (gate, up, down projections). Handles sparse expert dispatch with token routing. **Note:** Accumulates in bf16, causing slightly higher numerical divergence vs CPU reference. |
| **Custom RMSNorm** | `neuronx_distributed_inference.modules.custom_calls.CustomRMSNorm` | Hardware-accelerated RMSNorm via `AwsNeuronRmsNorm` custom call. Used 4 times per decoder layer (input_norm, post_attn_norm, pre_ff_norm, post_ff_norm). |
| **Cumsum** | `neuronxcc.nki.kernels.cumsum` | Attention mask computation for causal mask prefix sums. Used in both context encoding and token generation paths. |
| **Router TopK** | `neuronx_distributed.kernels.router_topk_kernel` | Expert selection in MoE routing -- selects top-k experts from sigmoid routing scores. Used once per MoE layer. |

### NKI Kernel Interaction with Trinity-Specific Features

- **Gated attention bypass:** When NKI fused attention block kernels are enabled (`attn_block_tkg_nki_kernel_enabled` or `attn_block_cte_nki_kernel_enabled`), Trinity's custom gated attention is bypassed and the base class fused kernel is used instead. The gated attention path is used when fused kernels are disabled.
- **MoE bf16 accumulation:** The blockwise matmul NKI kernel accumulates expert outputs in bf16 rather than fp32, which is the primary source of numerical divergence between Neuron and CPU reference outputs. Top-1 token accuracy is preserved.
- **Left-padding unsupported:** The NKI flash attention kernels require right-padding (`padding_side="right"`). Left-padding produces incorrect results.

## Example Checkpoints

- `arcee-ai/Trinity-Nano-Preview` (requires `trust_remote_code=True`)
- `arcee-ai/Trinity-Mini` (requires `trust_remote_code=True`)
- `arcee-ai/Trinity-Large-Preview` (requires `trust_remote_code=True`)

## Maintainer

Jim Burtoft

**Last Updated:** 2026-02-27
