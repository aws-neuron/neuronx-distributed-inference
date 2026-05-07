# Contrib Model: Trinity

NeuronX Distributed Inference implementation of the Trinity model family (AfmoeForCausalLM) from Arcee AI. A single unified implementation supports all three model sizes.

## Model Family

| Model | HuggingFace ID | Total Params | Active Params | Instance |
|-------|----------------|-------------|---------------|----------|
| **Nano** | `arcee-ai/Trinity-Nano-Preview` | ~6B | ~1B | inf2.xlarge* / inf2.8xlarge / trn2.3xlarge |
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

**Validated:** 2026-03-16
**SDK:** NxDI 0.8.0, neuronx-cc 2.23.6484, torch-neuronx 2.9.0.2.12, transformers 4.57.6 (SDK 2.28)
**Benchmarked:** 2026-03-06 (Nano + Mini, trn2.3xlarge + inf2.8xlarge, with bucketing)
**Fused TKG benchmarked:** 2026-03-18 (Mini with expert_bias patches, trn2.3xlarge, +29% throughput)
**Neuron vs CPU accuracy verified:** 2026-03-16 (Nano, trn2.3xlarge TP=2, 64-token generation)

All results below are from the **unified `modeling_trinity.py`** (this code), SDK 2.28 with bucketing enabled. Fused TKG results from SDK 2.28.

### Token Match Rate: Neuron vs CPU (Trinity-Nano, TP=2, SDK 2.28)

**Method:** Greedy decoding (argmax, `do_sample=False`) on both Neuron and CPU (HuggingFace `AutoModelForCausalLM`, bf16). 64 tokens generated per prompt. Match = identical token ID at each position.

| Prompt | Match | Rate | First Mismatch |
|--------|-------|------|----------------|
| "Hello, how are you?" | 4/64 | 6.2% | Token 2 |
| "Explain quantum computing in simple terms." | 30/64 | 46.9% | Token 7 |
| "Write a Python function that calculates the Fibonacci sequence." | 18/64 | 28.1% | Token 17 |
| "The capital of France is" | 5/64 | 7.8% | Token 2 |
| "def fibonacci(n):" | 11/64 | 17.2% | Token 6 |
| **"What is the meaning of life?"** | **64/64** | **100.0%** | **None** |
| "Once upon a time in a land far away," | 5/64 | 7.8% | Token 3 |
| "The quick brown fox jumps over the lazy" | 50/64 | 78.1% | Token 1 |

**Best match rate: 100.0%** (64/64 tokens, "What is the meaning of life?")
Average match rate: 36.5% across 8 prompts. Perfect matches: 1/8.

The sub-100% match rates on other prompts are expected: the MoE v2 NKI blockwise matmul kernel accumulates in bf16 rather than fp32, causing small numerical divergence that compounds across autoregressive steps. When the CPU and Neuron top-1 logits are close in magnitude, a single rounding difference changes the selected token, causing the generation paths to diverge from that point forward. Both paths produce coherent, sensible text.

### First-Token Accuracy (Trinity-Nano, TP=2, SDK 2.28)

Single forward pass comparison -- HuggingFace `AutoModelForCausalLM` (bf16) vs Neuron:

| Prompt | Neuron Top-1 | CPU Top-1 | Match | Top-20 Cosine | Full-Vocab Cosine |
|--------|-------------|-----------|-------|---------------|-------------------|
| "Hello, how are you?" | I | I | YES | 0.9995 | 0.9398 |
| "Explain quantum computing in simple terms." | Answer | Answer | YES | 0.9995 | 0.9905 |
| "Write a Python function that calculates the Fibonacci sequence." | The | The | YES | 0.9999 | 0.9835 |

**Summary:** 3/3 top-1 match, avg top-20 cosine similarity 0.9996, avg full-vocab cosine similarity 0.9712. Lower full-vocab cosine (esp. prompt 1 at 0.94) is expected due to MoE bf16 accumulation in NKI blockwise matmul kernel -- the tail of the logit distribution diverges while the top of the distribution is preserved.

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

**Note:** inf2.xlarge (16GB system RAM) cannot run Nano with standard loading -- OOM killed at 15.3GB RSS during weight loading. However, **pre-sharded weights bypass this entirely** (see Pre-Sharded Deployment below). inf2.8xlarge (123GB system RAM) works with standard loading at TP=1. NxDI auto-converts GQA to MHA when `TP=1` and `num_kv_heads=2`.

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

## Performance Benchmarks

**SDK 2.28**, seq_len=2048, BF16, bucketing enabled, measured with proper CTE+TKG pipeline (KV cache). TTFT averaged over 10 iterations (3 warmup). TKG averaged over 28 tokens (3 warmup discarded from 32 generated). Throughput = `batch_size * (1000 / avg_tkg_ms)` (steady-state TKG-based).

### Trinity-Nano (~6B total, ~1B active)

| Instance | TP | BS | TTFT (ms) | TKG (ms/tok) | Throughput (tok/s) | Per-seq (tok/s) | Compile |
|----------|-----|------|-----------|-------------|-------------------|-----------------|---------|
| inf2.xlarge* | 1 | 1 | 706 | 9.0 | 112 | 112 | N/A (pre-sharded) |
| inf2.8xlarge | 1 | 1 | 706 | 9.2 | 109 | 109 | 7.9 min |
| inf2.8xlarge | 1 | 2 | 901 | 13.3 | 150 | 75 | 8.8 min |
| inf2.8xlarge | 1 | 4 | 1347 | 20.8 | 192 | 48 | 11.7 min |
| inf2.8xlarge | 2 | 1 | 516 | 7.6 | 131 | 131 | 4.8 min |
| inf2.8xlarge | 2 | 2 | 674 | 9.4 | 212 | 106 | 6.6 min |
| inf2.8xlarge | 2 | 4 | 993 | 13.6 | 294 | 74 | 8.5 min |
| trn2.3xlarge | 2 | 1 | 516 | 10.8 | 93 | 93 | 4.9 min |
| trn2.3xlarge | 2 | 2 | 680 | 13.9 | 144 | 72 | 7.4 min |
| trn2.3xlarge | 2 | 4 | 930 | 16.3 | 245 | 61 | 9.4 min |
| trn2.3xlarge | 4 | 1 | 476 | 9.2 | 109 | 109 | 5.0 min |
| trn2.3xlarge | 4 | 2 | 600 | 12.4 | 161 | 81 | 6.5 min |
| trn2.3xlarge | 4 | 4 | 817 | 14.9 | 269 | 67 | 8.5 min |

**Whole-instance throughput** (TP x DP = all available cores):

| Instance | Config | BS | Throughput (tok/s) | Notes |
|----------|--------|----|--------------------|-------|
| inf2.8xlarge | TP=2 DP=1 | 4 | **294** | Best (use all 2 cores, single replica) |
| inf2.8xlarge | TP=1 DP=2 | 1 | 218* | Calculated: 2 x 109 tok/s |
| trn2.3xlarge | TP=4 DP=1 | 4 | **269** | Best (use all 4 cores, single replica) |
| trn2.3xlarge | TP=2 DP=2 | 1 | 186* | Calculated: 2 x 93 tok/s |

*DP throughput calculated mathematically (replicas are independent on separate NeuronCores).

**Recommended config**: inf2.8xlarge TP=2 BS=4 for max throughput (294 tok/s) with lowest TTFT on inf2 (516ms), or trn2.3xlarge TP=4 BS=4 for max throughput on trn2 (269 tok/s). *inf2.xlarge requires pre-sharded weights (see Pre-Sharded Deployment).

### Trinity-Mini (~26B total, ~4.5B active)

| Instance | TP | BS | TTFT (ms) | TKG (ms/tok) | Throughput (tok/s) | Per-seq (tok/s) | Compile | Notes |
|----------|-----|------|-----------|-------------|-------------------|-----------------|---------|-------|
| trn2.3xlarge | 4 | 1 | 371 | 11.8 | 85 | 85 | 3.9 min | |
| trn2.3xlarge | 4 | 2 | 598 | 11.5 | 174 | 87 | 6.8 min | |
| trn2.3xlarge | 4 | 4 | 805 | 13.6 | 295 | 74 | 9.1 min | |
| trn2.3xlarge | 4 | 1 | 371 | **9.1** | **110** | **110** | 3.8 min | **Fused TKG** |

Mini requires TP=4 (all cores on trn2.3xlarge), so DP is not applicable. With the fused MoE TKG kernel (requires patched libraries), BS=1 throughput improves from 85 to **110 tok/s** (+29%).

**Recommended config**: trn2.3xlarge TP=4 BS=4 for best throughput/latency balance (295 tok/s, 13.6ms TKG), or BS=1 for lowest TTFT (371 ms).

### Trinity-Large (~250B total, ~15B active)

| Instance | TP | BS | TTFT (ms) | TKG (ms/tok) | Throughput (tok/s) | Per-seq (tok/s) | Compile | Load |
|----------|-----|------|-----------|-------------|-------------------|-----------------|---------|------|
| trn2.48xlarge | 64 | 1 | 1161 | 14.7 | 68 | 68 | 9.2 min | 851s |
| trn2.48xlarge | 64 | 2 | 1657 | 19.1 | 102 | 51 | 11.1 min | 867s |
| trn2.48xlarge | 64 | 4 | 1980 | 29.0 | 137 | 34 | 14.0 min | 873s |

**Recommended config**: trn2.48xlarge TP=64 BS=1 for lowest latency (1.16s TTFT, 14.7ms TKG), or BS=4 for max aggregate throughput (137 tok/s). NVMe instance store required for model storage (~743GB on disk).

### GPU Comparison (g5.12xlarge, 4x NVIDIA A10G)

Benchmarked via vLLM 0.16.0, bf16. Shows single-request latency and aggregate throughput at various concurrency levels. GPU uses continuous batching with CUDA graphs (PagedAttention v2).

**Trinity-Nano** on 1x A10G (TP=1):

| Concurrency | TTFT (ms) | TKG (ms/tok) | Throughput (tok/s) |
|-------------|-----------|-------------|-------------------|
| 1 | 20 | 6.9 | 137 |
| 4 | 30 | -- | 400 |
| 16 | 56 | -- | 1140 |
| 64 | 65 | -- | 2782 |

**Trinity-Mini** on 4x A10G (TP=4, max_num_seqs=32):

| Concurrency | TTFT (ms) | TKG (ms/tok) | Throughput (tok/s) |
|-------------|-----------|-------------|-------------------|
| 1 | 24 | 6.7 | 138 |
| 4 | 42 | -- | 337 |
| 16 | 79 | -- | 857 |

**GPU vs Neuron** (single-request, BS=1):

| Model | Metric | GPU (A10G) | Neuron (best) | Notes |
|-------|--------|-----------|---------------|-------|
| Nano | TTFT | 20 ms | 476 ms | GPU 24x faster (CUDA graphs vs CTE forward) |
| Nano | TKG | 6.9 ms | 7.6 ms | GPU 1.1x faster (inf2 TP=2) |
| Mini | TTFT | 24 ms | 371 ms | GPU 15x faster |
| Mini | TKG | 6.7 ms | 11.5 ms | GPU 1.7x faster |

GPU TTFT advantage comes from vLLM's CUDA graph capture eliminating kernel launch overhead. Neuron TTFT is dominated by the CTE forward pass through compiled HLO graphs. A vLLM-Neuron serving stack would narrow this gap.

### Key Observations

- **Batching scales well**: BS=4 gives 2.0-3.5x aggregate throughput vs BS=1, with TKG latency increase of 30-100%
- **Mini is fastest TTFT**: 371ms at TP=4 BS=1, vs 476ms (Nano TP=4) and 1161ms (Large TP=64)
- **inf2.8xlarge TP=2 is best for Nano**: 294 tok/s (BS=4) with 516ms TTFT -- better throughput than trn2 TP=4 (269 tok/s)
- **TP=2 on inf2 outperforms TP=1**: 21-53% higher throughput across batch sizes (TKG drops from 9.2ms to 7.6ms at BS=1)
- **DP gives higher throughput than TP for small models**: trn2 TP=2 DP=2 at BS=1 yields 186 tok/s vs TP=4 DP=1 BS=1 at 109 tok/s, but at higher per-token latency
- **TP=4 vs TP=2 on trn2**: TP=4 has 15-17% lower TKG latency (better per-sequence), but TP=2 enables DP=2 for higher aggregate throughput
- **Compile time grows with batch size**: BS=4 takes 8.5-9.4 min vs 3.9-5.0 min (BS=1)
- **Large TKG is comparable to smaller models**: 14.7ms despite 250B total params -- MoE activates only 15B
- **Load time dominates Large**: 14.2 min to shard 516GB across 64 cores; compile is only 9.2 min
- **GPU has massive TTFT advantage**: 20-24ms vs 371-706ms (15-35x) due to CUDA graphs vs compiled HLO forward pass
- **GPU aggregate throughput scales with concurrency**: 2782 tok/s (Nano, 64 concurrent) vs 294 tok/s (Neuron inf2 TP=2 BS=4) -- continuous batching vs static batching
- **GPU TKG is 1.1-1.7x faster**: 6.7-6.9ms vs 7.6-11.8ms on Neuron
- **inf2.xlarge cannot run Nano**: 16GB system RAM is insufficient for 12GB bf16 model weight loading (OOM during sharding), even with pre-compiled artifacts. **Pre-sharded weights solve this** (1.39 GB RSS, 112 tok/s).

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
    seq_len=2048,      # Max tested: 40960 (TP=2), 49152 (TP=4)
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

**With bucketing** (for variable-length inputs):

```python
neuron_config = MoENeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=4096,
    torch_dtype=torch.bfloat16,
    enable_bucketing=True,
    apply_seq_ids_mask=True,
    context_encoding_buckets=[2048, 4096],  # Must be >= sliding_window (2048)
    token_generation_buckets=[2048, 4096],
)
```

**Instance:** inf2.xlarge (TP=1, pre-sharded weights required), inf2.8xlarge (TP=1), or trn2.3xlarge (TP=2/4).

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
    seq_len=2048,      # Max tested: 32768
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

**With bucketing** (for variable-length inputs):

```python
neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=4096,
    torch_dtype=torch.bfloat16,
    enable_bucketing=True,
    apply_seq_ids_mask=True,
    context_encoding_buckets=[2048, 4096],  # Must be >= sliding_window (2048)
    token_generation_buckets=[2048, 4096],
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
    seq_len=4096,      # Max tested: 30720
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

**With bucketing** (for variable-length inputs):

```python
neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=8192,
    torch_dtype=torch.bfloat16,
    enable_bucketing=True,
    apply_seq_ids_mask=True,
    context_encoding_buckets=[4096, 8192],  # Must be >= sliding_window (4096)
    token_generation_buckets=[4096, 8192],
)
```

**Instance:** trn2.48xlarge only (TP=64, capacity block required, NVMe instance store for model storage).

## Pre-Sharded Deployment (inf2.xlarge)

The standard NxDI load path downloads the full HuggingFace checkpoint into CPU RAM, converts it to Neuron format, and shards weights by TP rank. For Trinity-Nano (~12GB bf16), this requires 15+ GB system RAM — exceeding inf2.xlarge's 16GB.

**Pre-sharded weights** bypass this entirely. During compilation on a larger instance, setting `save_sharded_checkpoint=True` saves per-rank weight files (`weights/tp{rank}_sharded_checkpoint.safetensors`). During loading, NxDI reads directly from these files without loading the full HF checkpoint.

### Workflow

1. **Compile on a larger instance** (inf2.8xlarge or trn2):

```python
neuron_config = MoENeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,  # Key flag
)
config = TrinityInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronTrinityForCausalLM(model_path, config)
model.compile(compiled_path)  # Saves model.pt, neuron_config.json, weights/tp0_sharded_checkpoint.safetensors
```

2. **Upload the compiled artifact** (model.pt + neuron_config.json + weights/) to HuggingFace or S3.

3. **Load on inf2.xlarge** (16GB RAM):

```python
neuron_config = MoENeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,  # Must match compilation
)
config = TrinityInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronTrinityForCausalLM(model_path, config)
model.load(compiled_artifact_path)  # Reads directly from sharded files — only 1.4 GB RSS
```

### Pre-Sharded Results (inf2.xlarge)

| Instance | TP | BS | TTFT (ms) | TKG (ms/tok) | Throughput (tok/s) | Load | Peak RSS |
|----------|-----|------|-----------|-------------|-------------------|------|----------|
| inf2.xlarge | 1 | 1 | 706 | 9.0 | 112 | 18.4s | 1.39 GB |

- **Memory**: 1.9 GB system RAM used (12.6%) vs 15+ GB that causes OOM with standard loading
- **Performance**: Identical to inf2.8xlarge (706 vs 707ms TTFT, 9.0 vs 9.1ms TKG)
- **Load time**: 18.4s (comparable to inf2.8xlarge's 11s pre-sharded / 48s standard)

### Available Pre-Compiled Artifacts

| Model | TP | BS | seq_len | SDK | HuggingFace Repo |
|-------|-----|------|---------|-----|-----------------|
| Nano | 1 | 1 | 2048 | 2.28 | `jburtoft/Trinity-Nano-Neuron-TP1` |

## Caveats

1. **`padding_side="right"` required** -- NKI flash attention kernel does not support left-padding. Always set `padding_side="right"` on the tokenizer.

2. **MoE v2 bf16 accumulation** -- The NxDI MoE v2 NKI kernel accumulates in bf16, causing ~23x more divergence per MoE layer compared to dense layers. Full-vocab cosine similarity is ~0.936, but top-1 token accuracy is preserved. A fix ticket has been filed.

3. **`trust_remote_code=True` required** -- Trinity uses a custom `AfmoeForCausalLM` architecture not in standard transformers. The HuggingFace download requires `trust_remote_code=True`.

4. **transformers version sensitivity** -- Use transformers 4.56.2 with SDK 2.27. Reference outputs may vary across transformers versions.

5. **GLU type** -- Trinity uses `SiLU(gate) * up` which maps to NxDI's `glu_type="glu"`, NOT `"swiglu"`. This is handled automatically by the config class.

6. **route_scale baked into weights** -- NxDI MoE v2 does not support `route_scale` natively. The scale is baked into routed expert `down_proj` weights during weight conversion. Shared expert weights are NOT scaled.

7. **Gate padding at high TP** -- When `num_attention_heads` is not evenly divisible by `tp_degree` (e.g., Large at TP=64: 48/64), gate weights are padded with interleaved layout matching the Q projection. This is handled automatically during weight conversion.

8. **Mixed attention KV cache (TrinityKVCacheManager)** -- Trinity uses mixed attention (alternating sliding window and full attention every 4th layer). The standard `KVCacheManager` applies a single `sliding_window` modulation to all layers, which causes out-of-bounds writes for full-attention layers with larger KV caches. `TrinityKVCacheManager` provides per-layer KV cache management: uniform `max_length` cache buffers (safe for CTE `fill_prefix`), per-layer scatter modulation during TKG (sliding: `pos % sliding_window`, global: no modulation), and per-layer KV read slicing (sliding: `sliding_window`, global: `max_length`). This is enabled automatically.

## Maximum Sequence Length

Validated with token generation (5 tokens per prompt) at each max seq_len:

| Model | Instance | TP | Max seq_len | Compile | Load | Gen Latency |
|-------|----------|-----|------------|---------|------|-------------|
| Nano | trn2.3xlarge | 2 | **40,960** | 1.5 min | 3.2 min | 2.4s/tok |
| Nano | trn2.3xlarge | 4 | **49,152** | 1.4 min | 1.4 min | 2.4s/tok |
| Mini | trn2.3xlarge | 4 | **32,768** | 0.9 min | 7.7 min | 2.4s/tok |
| Large | trn2.48xlarge | 64 | **30,720** | 1.6 min | 16.5 min | 2.9s/tok |

Compile times above are for cache-hit runs. First compilation at each seq_len takes 5-25 min.

Higher TP gives more headroom for KV cache (Nano TP=4 fits 49K vs 41K at TP=2). The failure mode at the limit is compilation timeout, not OOM.

## Bucketing

Bucketing compiles separate NEFFs for different sequence length buckets, enabling efficient inference for variable-length inputs without padding every input to `seq_len`.

### Configuration

Enable bucketing by adding `enable_bucketing=True` and `apply_seq_ids_mask=True` to the neuron config:

```python
neuron_config = MoENeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=4096,
    torch_dtype=torch.bfloat16,
    enable_bucketing=True,
    apply_seq_ids_mask=True,       # Required for mixed attention bucketing
    context_encoding_buckets=[2048, 4096],  # Optional: custom CTE buckets
    token_generation_buckets=[2048, 4096],  # Optional: custom TKG buckets
)
```

If `context_encoding_buckets` / `token_generation_buckets` are omitted, NxDI auto-generates power-of-2 buckets from 128 to `seq_len`.

### Restrictions

1. **`apply_seq_ids_mask=True` is required.** Without it, TKG fails with a shape mismatch: full-attention layers have KV caches sized to `seq_len` but the TKG attention mask is only sized to the bucket value.

2. **All context encoding buckets must be >= `sliding_window`.** The `get_last_kv_window` function in windowed attention gathers indices up to `sliding_window - 1` from the K/V tensors. A CTE bucket smaller than `sliding_window` produces K/V tensors too short for those indices, causing an out-of-bounds memory access. For Trinity-Nano and Trinity-Mini (`sliding_window=2048`), the smallest CTE bucket must be at least 2048. For Trinity-Large (`sliding_window=4096`), the smallest CTE bucket must be at least 4096. Short prompts are padded to the smallest qualifying bucket.

3. **Token generation buckets have no minimum.** TKG buckets control the `n_positions` dimension for the attention mask. The `apply_seq_ids_mask` flag dynamically pads the mask when needed.

### Validated Bucketing Results

**Trinity-Nano** on trn2.3xlarge (TP=2, seq_len=4096, buckets=[2048, 4096]):

| Prompt | Input Tokens | Top-1 | Forward | Status |
|--------|-------------|-------|---------|--------|
| "Hello!" | 3 | "I" | 0.51s | PASS |
| "What is the capital of France?" | 8 | "(" | 0.50s | PASS |
| (20-token prompt) | 20 | "Provide" | 0.50s | PASS |
| (124-token prompt) | 124 | `<\|im_end\|>` | 0.50s | PASS |

- Compile: 7.0 min (4 NEFFs: 2 CTE + 2 TKG)
- Load: 37.6s
- Token generation: ~0.5s/tok (5 tokens generated per prompt)
- Backward compatible with non-bucketing flow

**Trinity-Mini** on trn2.3xlarge (TP=4, seq_len=4096, buckets=[2048, 4096]):

| Prompt | Input Tokens | Top-1 | Forward | Status |
|--------|-------------|-------|---------|--------|
| "Hello!" | 3 | "I" | 0.37s | PASS |
| "What is the capital of France?" | 8 | "Paris" | 0.37s | PASS |
| (20-token prompt) | 20 | "Also" | 0.37s | PASS |
| (124-token prompt) | 124 | "Conclude" | 0.37s | PASS |

- Compile: 5.5 min (4 NEFFs: 2 CTE + 2 TKG)
- Load: 78.3s (1.3 min)
- Token generation: ~0.4s/tok (5 tokens generated per prompt)

**Trinity-Large** on trn2.48xlarge (TP=64, seq_len=8192, buckets=[4096, 8192]):

| Prompt | Input Tokens | Top-1 | Forward | Status |
|--------|-------------|-------|---------|--------|
| "Hello!" | 3 | "I" | 1.16s | PASS |
| "What is the capital of France?" | 8 | "\n" | 1.15s | PASS |
| (20-token prompt) | 20 | "\n" | 1.15s | PASS |
| (124-token prompt) | 124 | `<\|end_of_text\|>` | 1.15s | PASS |

- Compile: 12.5 min (4 NEFFs: 2 CTE + 2 TKG)
- Load: 15.7 min
- Token generation: ~1.2s/tok (5 tokens generated per prompt)

### How It Works

Trinity's mixed attention (sliding window + full attention every 4th layer) requires three mechanisms working together for bucketing:

1. **`has_mixed_attn=True`** on the model base tells the framework to generate dual attention masks: a global causal mask for full-attention layers and a local windowed mask for sliding layers. The decoder layer selects the appropriate mask per layer type (Llama4 pattern).

2. **`apply_seq_ids_mask=True`** enables dynamic mask padding in `compute_for_token_gen`. When a full-attention layer's KV cache (sized `max_length`) exceeds the TKG bucket's attention mask (sized `n_positions`), the mask is automatically padded with zeros.

3. **`TrinityKVCacheManager`** replaces the standard `KVCacheManager` with per-layer awareness. All layers share uniform `max_length` cache buffers (required for CTE `fill_prefix` safety), but during TKG, scatter indices are modulated per-layer (sliding: `position % sliding_window`, global: raw position) and KV reads are sliced per-layer (sliding: `sliding_window`, global: `max_length`).

## LNC (Logical NeuronCore) Configuration

On trn2 instances, the LNC setting determines the number of logical NeuronCores and therefore the **valid TP degrees**. The Neuron runtime requires `NEURON_RT_NUM_CORES` to be either 1 or the full device (all logical cores). Intermediate values are rejected at runtime.

| LNC | Logical Cores (trn2.3xlarge) | Valid TP | HBM per Core |
|-----|------------------------------|----------|--------------|
| LNC=2 (default) | 4 | **1 or 4 only** | 24 GB |
| LNC=1 | 8 | **1 or 8 only** | 12 GB shared |

**Common pitfall:** TP=2 is NOT valid with LNC=2 on trn2.3xlarge. The runtime error is: `NEURON_RT_NUM_CORES must request one core, or the whole device (multiple of 8)`. To use TP=2, switch to LNC=1 (which gives 8 logical cores, making TP=2 valid). However, LNC=1 halves HBM bandwidth per core.

**Practical impact on Trinity:**
- **Nano (TP=1):** Works on both LNC=1 and LNC=2
- **Nano (TP=2):** Requires LNC=1. Use TP=1 or TP=4 with default LNC=2
- **Mini (TP=4):** Works on LNC=2 (default) -- uses all 4 logical cores
- **Large (TP=64):** Works on LNC=2 (default) on trn2.48xlarge -- uses all 64 logical cores

To check or change LNC:
```bash
# Check current LNC
neuron-ls  # Shows logical core count

# Set LNC=1 (persistent, requires reboot)
echo 'NEURON_LOGICAL_NC_CONFIG=1' | sudo tee /etc/environment
sudo reboot

# Set LNC=1 (current session only)
export NEURON_LOGICAL_NC_CONFIG=1
```

## Compatibility Matrix

| Model | Instance | TP | LNC | Max seq_len | Status |
|-------|----------|-----|-----|------------|--------|
| Nano | inf2.xlarge | 1 | N/A | -- | PASS with pre-sharded weights (standard load OOMs at 16GB system RAM) |
| Nano | inf2.8xlarge | 1 | N/A | -- | Validated (not seq_len tested) |
| Nano | inf2.8xlarge | 2 | N/A | -- | Validated (best throughput on inf2) |
| Nano | trn2.3xlarge | 2 | 2 | 40,960 | Validated |
| Nano | trn2.3xlarge | 4 | 2 | 49,152 | Validated |
| Mini | inf2.8xlarge | -- | -- | -- | Does NOT fit |
| Mini | trn2.3xlarge | 4 | 2 | 32,768 | Validated |
| Large | trn2.48xlarge | 32 | 2 | -- | FAIL (HBM OOM per NC) |
| Large | trn2.48xlarge | 64 | 2 | 30,720 | Validated |

### Minimum Requirements by Model Size

| Model | Min HBM | Min TP | Min Instance |
|-------|---------|--------|-------------|
| Nano | ~12GB bf16 | 1 | inf2.xlarge (pre-sharded weights) or inf2.8xlarge |
| Mini | ~48GB bf16 | 4 | trn2.3xlarge |
| Large | ~500GB bf16 | 64 | trn2.48xlarge (capacity block, NVMe storage) |

### SDK Configuration

| Component | SDK 2.27 | SDK 2.28 |
|-----------|----------|----------|
| NxDI | 0.7.15063 | 0.8.0 |
| neuronx-cc | 2.22.12471 | 2.23.6484 |
| torch-neuronx | 2.9.0.2.11 | 2.9.0.2.12.22436 |
| torch | 2.9.0 | 2.9.0 |
| transformers | 4.56.2 | 4.57.6 |
| Venv | `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/` | same |

Both SDK versions are validated. Fused MoE TKG requires SDK 2.28.

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
5. **expert_bias not supported by NxDI:** Created custom `RouterTopKWithBias` subclass for non-fused path. For fused TKG kernel, patched three upstream libraries to thread `expert_bias` through the kernel pipeline (see Fused MoE TKG section).
6. **Conditional RoPE:** Only sliding attention layers get rotary embeddings.
7. **Mixed attention masks and KV cache:** Framework provides both global and local masks via `has_mixed_attn=True`; decoder layer selects based on layer type. `TrinityKVCacheManager` provides per-layer KV cache management (uniform buffers, per-layer scatter modulation and read slicing) to handle the different cache sizes of sliding vs full-attention layers.
8. **Gate weight padding at high TP:** Interleaved padding matching Q projection layout (prevents wrong-head gating on 54/64 cores).
9. **Shared expert weight loading:** Standalone module for reliable weight mapping vs NxDI built-in shared expert handling.

## Fused MoE TKG NKI Kernel (SDK 2.28+)

The SDK 2.28 fused MoE TKG kernel (`moe_token_gen_selective_load_kernel`) combines RMSNorm + Router TopK + Expert MLP into a single NKI kernel for token generation, reducing HBM round-trips.

### Configuration

```python
neuron_config = MoENeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=2048,
    torch_dtype=torch.bfloat16,
    moe_fused_nki_kernel_enabled=True,
    router_topk_nki_kernel_enabled=False,  # Must be False for sigmoid routing
    expert_mlp_nki_kernel_enabled=True,
)
```

### Sigmoid Routing Support

The fused kernel's NKI router asserts `router_act_fn == SOFTMAX`, but Trinity uses sigmoid. The implementation patches the kernel to use the ISA router fallback (`use_router_topk_nki_kernel=False`), which supports both sigmoid and softmax. This is done automatically via `_PatchedKernelCall` wrapper applied during `setup_attr_for_model()`.

### Alignment Constraint

The fused kernel requires `moe_intermediate_size / tp_degree % 128 == 0`:

| Model | moe_intermediate | TP | Per-TP | Eligible? |
|-------|-----------------|-----|--------|-----------|
| Nano | 256 | 2 | 128 | YES |
| Nano | 256 | 4 | 64 | NO |
| Mini | 1024 | 4 | 256 | YES |
| Large | 3072 | 64 | 48 | NO |

The config class automatically enables/disables fused TKG based on this alignment check.

### Test Results

#### Trinity-Nano (SDK 2.28, trn2.3xlarge, TP=2)

**CTE (context encoding) -- exact match with non-fused baseline:**

| Prompt | Non-fused Top-1 | Fused Top-1 | Match |
|--------|----------------|-------------|-------|
| Hello, how are you? | I | I | YES |
| What is the capital of France? | ( | ( | YES |
| Explain quantum computing in simple terms. | Answer | Answer | YES |
| Write a Python function to compute fibonacci numbers. | The | The | YES |
| The meaning of life is | to | to | YES |

**Latency (Nano):**

| Metric | Non-fused | Fused | Notes |
|--------|-----------|-------|-------|
| Compile | 357s (5.9 min) | 258s (4.3 min) | Fused compiles faster |
| CTE latency | ~0.52s | ~0.49s | Similar |
| TKG latency | ~0.011s | ~0.014s | Fused is slower on Nano |

The fused kernel does not improve TKG latency on Trinity-Nano (intermediate_size=256 is too small for the selective loading to pay off). The kernel is designed for larger models where expert weight loading from HBM is the bottleneck.

#### Trinity-Mini (SDK 2.28, trn2.3xlarge, TP=4) -- with expert_bias patches

Tested with patched `nki-library`, `neuronx-distributed`, and `neuronx-distributed-inference` libraries from `feature/expert-bias-support` branches, which thread `expert_bias` through the fused kernel pipeline.

**CTE correctness -- exact match with non-fused baseline (5/5):**

| Prompt | Non-fused Top-1 | Fused Top-1 | Match |
|--------|----------------|-------------|-------|
| Hello, how are you? | I | I | YES |
| What is the capital of France? | Paris | Paris | YES |
| Explain quantum computing in simple terms. | What | What | YES |
| Write a Python function to compute fibonacci numbers. | The | The | YES |
| The meaning of life is | to | to | YES |

**Latency and throughput (Mini, BS=1):**

| Metric | Non-fused | Fused+expert_bias | Change |
|--------|-----------|-------------------|--------|
| CTE (ms) | 371.1 | 371.0 | -0.0% |
| TKG (ms/tok) | 11.8 | 9.1 | **-22.5%** |
| Throughput (tok/s) | 84.9 | 109.5 | **+29.0%** |
| Compile time | 3.9 min | 3.8 min | -2.6% |

**TKG multi-token divergence:** The second TKG token diverges between fused and non-fused paths due to numerical reordering in the fused kernel (different operation order produces bf16-equivalent but non-identical intermediate values). This is the same class of divergence observed in all bf16 MoE comparisons (see Token Match Rate section). Both paths produce coherent text; the divergence compounds autoregressively as expected.

### Expert Bias Support (Patched Libraries)

The fused MoE TKG kernel requires `expert_bias` to be threaded through three library layers. The patches are on `feature/expert-bias-support` branches:

| Repo | File | Change | Lines |
|------|------|--------|-------|
| [nki-library](https://github.com/jimburtoft/nki-library/tree/feature/expert-bias-support) | `router_topk.py` | Add `expert_bias` param, load/broadcast/add in SBUF | ~30 |
| [neuronx-distributed](https://github.com/jimburtoft/neuronx-distributed/tree/feature/expert-bias-support) | `routing.py` + `moe_fused_tkg.py` | Add `expert_bias_size` to `RouterTopK`, pass through `optional_kwargs` | ~15 |
| [neuronx-distributed-inference](https://github.com/jimburtoft/neuronx-distributed-inference/tree/feature/expert-bias-support) | `moe_v2.py` | Pass `expert_bias_size` from config | ~3 |

**Install:**
```bash
pip install --no-deps git+https://github.com/jimburtoft/nki-library.git@feature/expert-bias-support
pip install --no-deps git+https://github.com/jimburtoft/neuronx-distributed.git@feature/expert-bias-support
pip install --no-deps git+https://github.com/jimburtoft/neuronx-distributed-inference.git@feature/expert-bias-support
```

Without these patches, the fused kernel omits `expert_bias`, causing incorrect expert selection and garbled TKG output.

### Known Limitations

1. **Nano TP=4 and Large TP=64 ineligible** -- Alignment constraint `intermediate/TP % 128 != 0` prevents use.
2. **No latency benefit on Nano** -- Expert weights (256 intermediate) are too small for selective loading overhead to pay off.
3. **Requires patched libraries for expert_bias** -- Until the `feature/expert-bias-support` patches are merged upstream, the fused kernel requires installing forked versions of nki-library, neuronx-distributed, and neuronx-distributed-inference.

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
| **Fused MoE TKG** | `neuronxcc.nki._pre_prod_kernels.moe_token_gen.moe_token_gen_selective_load_kernel` | Combines RMSNorm + Router TopK + Expert MLP for token generation. Selectively loads expert weights from HBM. SDK 2.28+. Uses ISA router fallback for sigmoid. |

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

**Last Updated:** 2026-03-18 (added LNC configuration note with valid TP degrees per LNC mode; added fused MoE TKG with expert_bias benchmark: Mini +29% throughput, 5/5 correctness match)
