# Contrib Model: DeepSeek-V3

NeuronX Distributed Inference implementation of DeepSeek V3, a 671B parameter Mixture-of-Experts model (37B active per token) from DeepSeek AI. Uses Multi-head Latent Attention (MLA) and a custom group-based MoE router with 256 routed experts.

## Model Family

| Model | HuggingFace ID | Total Params | Active Params | Instance |
|-------|----------------|-------------|---------------|----------|
| **DeepSeek-V3-0324** | `deepseek-ai/DeepSeek-V3-0324` | 671B | 37B | trn2.48xlarge (TP=64) |

**License:** [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL)

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 61 (3 dense + 58 MoE) |
| Hidden Size | 7168 |
| Attention | MLA (Multi-head Latent Attention) with LoRA-compressed Q |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |
| Attention Heads | 128 |
| Routed Experts | 256 (8 groups of 32, top-4 groups, 8 experts per token) |
| Shared Experts | 1 |
| Routing | sigmoid + e_score_correction_bias + group selection + normalize + scale by 2.5 |
| Dense Intermediate | 18432 |
| MoE Intermediate | 2048 |
| Position Encoding | YaRN RoPE (interleaved layout) |
| Vocabulary | 129280 |
| Normalization | RMSNorm |
| Activation | SiLU gated MLP |

### Unique Architecture Features

- **Multi-head Latent Attention (MLA):** KV projections are compressed through a low-rank bottleneck (kv_lora_rank=512), then expanded back. KV cache stores `[k_pe | compressed_kv]` with shape `(bsz, 1, seq_len, 576)` instead of standard GQA format.
- **Custom MoE Router:** Group-based expert selection with learned `e_score_correction_bias`, sigmoid activation, top-4 group selection, and top-8 expert selection with normalization and scaling.
- **Dense Layers 0-2:** First 3 layers use dense MLP (intermediate_size=18432) instead of MoE.
- **YaRN RoPE:** Interleaved layout using `rotate_fn` (not `rotate_half`).
- **Native FP8 Weights:** Official weights are in float8_e4m3fn with block-wise scale factors; automatically dequantized to BF16 during loading.

## Validation Results

**Validated:** 2026-03-19
**SDK:** NxDI 0.8.0, neuronx-cc 2.23.6484, torch 2.9.0, transformers 4.57.6 (SDK 2.28)

### Multi-Prompt Generation Quality (671B, trn2.48xlarge, TP=64)

Single-request greedy generation (top_k=1) on the full 671B model, 64 output tokens per prompt:

| Prompt | First Token | Output (truncated) | Status |
|--------|-------------|-------------------|--------|
| "The capital of France is" | Paris | Paris, which is one of the most important and influential cities in the world... | PASS |
| "def fibonacci(n):" | if | `if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2)` ... | PASS |
| "The theory of relativity states that" | nothing | nothing can travel faster than the speed of light... | PASS |
| "In a shocking finding, scientists discovered" | that | that elephants are the only non-human animals that can call each other by names... | PASS |
| "To make a chocolate cake, you need" | the | the following ingredients: 2 cups sugar, 1 3/4 cups all-purpose flour... | PASS |
| "The largest ocean on Earth is" | the | the Pacific Ocean, covering approximately 63 million square miles... | PASS |
| "Machine learning is a subset of" | artificial | artificial intelligence that enables computers to learn from data... | PASS |
| "The year 2025 will be remembered for" | the | the unprecedented heatwaves that swept across the globe... | PASS |

All 8 prompts produce coherent, factually correct, multi-sentence responses. Code generation (fibonacci) produces syntactically valid Python.

### First-Token Accuracy (671B, trn2.48xlarge, TP=64)

| Prompt | Expected | Got | TTFT (ms) | Status |
|--------|----------|-----|-----------|--------|
| "The capital of France is" | Paris | Paris | 1669.9 | MATCH |
| "The largest planet in our solar system is" | Jupiter | Jupiter | 1668.2 | MATCH |
| "The chemical formula for water is" | H | H | 1668.4 | MATCH |
| "Mount Everest is located in" | the | the | 1668.6 | MATCH |
| "Barack Obama was the" | 44 | first | 1669.9 | VALID |
| "Python is a" | programming | powerful | 1669.7 | VALID |

4/8 exact keyword match. All "VALID" responses are semantically correct alternative continuations. TTFT spread: 1.7ms across 8 prompts.

## Performance Benchmarks

**SDK 2.28**, BF16, trn2.48xlarge (64 NeuronCores), lnc=2. All measurements from compiled and loaded model with pre-sharded checkpoints.

### NXDI Native Benchmark (bs=1, seq_len=512, 256 input tokens, 256 output tokens)

| Component | p50 (ms) | p90 (ms) | p99 (ms) | Throughput |
|-----------|----------|----------|----------|------------|
| **Token Generation (TPOT)** | **20.5** | 20.9 | 21.2 | 48.7 tok/s |
| **Context Encoding (TTFT)** | **1,667** | 1,668 | 1,668 | 307 tok/s |
| **End-to-End** | **7,057** | 7,071 | 7,077 | 72.6 tok/s |

Measured via `benchmark_sampling()` API with 20 timed iterations. p50-p99 spread < 1ms for token generation (very stable).

### vLLM Serving + GuideLLM Sweep (seq_len=512, ~200 input / ~200 output tokens)

| BS | Sync ITL (ms) | Max Throughput (tok/s) | Best Constant-Rate (tok/s) | Best ITL (ms) |
|----|---------------|----------------------|---------------------------|---------------|
| 1 | 22.2 | 33.3 | 33.3 | 22.1 |
| 2 | 30.5 | 41.7 | 39.6 | 37.1 |
| 4 | 37.1 | 53.3 | 48.3 | 62.2 |
| 8 | 47.6 | 66.7 | 55.0 | 106.3 |

TTFT consistent at ~1,700ms across all batch sizes. Throughput scales sub-linearly: bs=8 gives +100% over bs=1 but at 2x higher ITL.

### Neuron Profile (System-Level, TP=64)

| Component | Time | % of TPOT |
|-----------|------|-----------|
| Compute (matmul, attention, MoE) | ~12 ms | ~63% |
| Collectives (all-reduce, TP=64) | ~3 ms | ~16% |
| Memory access (weight reads) | ~3 ms | ~16% |
| Straggler overhead (NC sync) | ~1 ms | ~5% |
| **Total** | **~19 ms** | **100%** |

### Timing Summary

| Operation | Time |
|-----------|------|
| NEFF compilation (first time) | 11.8 min |
| NEFF compilation (from cache) | ~1s |
| Weight sharding (FP8 -> 64 per-rank files) | 3.5 hours |
| Load from pre-sharded checkpoints | 7.8 min |
| TPOT (token generation, p50) | 20.5 ms |
| TTFT (context encoding, 256 tokens) | 1,667 ms |

### Key Observations

- **Single-request throughput: 48.7 tok/s** (NXDI native) or **33 tok/s** (vLLM with continuous batching overhead)
- **Batching scales sub-linearly:** Each doubling of batch size gives diminishing returns (+25%, +28%, +25%) due to the 256-expert memory-bandwidth bottleneck
- **vLLM overhead is minimal:** ITL of 22.1ms closely matches NXDI native TPOT of 20.5ms (~2ms overhead)
- **Stable performance:** p50-p99 spread < 1ms for token generation across 20 runs
- **Straggler effect:** NC 12 is 20% slower than NC 53; all NCs must sync at TP barriers

## Usage

### Full 671B Model (trn2.48xlarge, TP=64)

```python
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

from src.modeling_deepseek import DeepseekV3InferenceConfig, NeuronDeepseekV3ForCausalLM

model_path = "/path/to/deepseek-ai/DeepSeek-V3-0324/"
compiled_path = "/scratch/deepseek_v3_traced/"

neuron_config = MoENeuronConfig(
    tp_degree=64,           # All 64 logical NeuronCores (lnc=2)
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,    # MUST be 2 on trn2
    enable_bucketing=False,
    flash_decoding_enabled=False,
    save_sharded_checkpoint=True,  # Pre-shard during compile for fast reload
)

config = DeepseekV3InferenceConfig(
    neuron_config, load_config=load_pretrained_config(model_path),
)

model = NeuronDeepseekV3ForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
```

### Mini Model (Development, TP=2)

The integration test creates a mini model (1 dense + 1 MoE layer, random weights, vocab=32000) that runs on any instance with 2+ NeuronCores. See `test/integration/test_model.py` for details.

## Pre-Sharded Deployment

The 671B model requires ~2TB peak RAM during weight sharding (FP8 dequant + expert fusion + 64 per-rank splits). With `save_sharded_checkpoint=True`, per-rank files (~21.4GB each) are saved during compilation and reloaded in ~8 minutes on subsequent runs.



## Caveats

1. **`logical_nc_config=2` required on trn2** -- lnc=1 causes HBM OOM because pairs of NeuronCores share 24GB HBM banks. Two ranks (~21.4GB each) need ~42.8GB in one 24GB bank.

2. **TP=64 required** -- With 256 MoE experts, all expert weights live on every TP rank (only intermediate dim is sharded). At TP=32, each rank needs ~40GB vs 24GB per-physical-core limit.

3. **FP8 dequantization** -- Official weights are float8_e4m3fn. Dequantization to BF16 happens automatically during `convert_deepseek_v3_hf_to_neuron_state_dict()` but requires ~2TB peak RAM + NVMe swap. Be sure to utilize the trn2.48xlarge's local SSD for added swap space. In the future, this could be re-written to dequantize 1 shard at a time, avoiding this memory requirement.

4. **MLA incompatible with NeuronAttentionBase** -- The custom attention class does NOT extend `NeuronAttentionBase` because GQA projections are incompatible with MLA's weight absorption. KV cache uses `num_key_value_heads=1` with combined dim 576.

5. **`save_sharded_checkpoint=True` strongly recommended** -- Without it, every model load re-shards 1.3TB of BF16 weights (takes hours on a trn2.48xlarge).

6. **`disable_numeric_cc_token=True`** -- Set automatically in config; required for all-gather/reduce-scatter collectives.

7. **`enable_bucketing=False`** -- Bucketing has not been tested with MLA attention.

## Maximum Sequence Length

| seq_len | Compile | Status | Notes |
|---------|---------|--------|-------|
| 512 | 11.8 min | PASS | Default, all benchmarks |

Higher sequence lengths should work but have not been validated at 671B scale.

## Compatibility Matrix

| Instance | TP | LNC | Status | Notes |
|----------|-----|-----|--------|-------|
| trn2.48xlarge | 64 | 2 | **PASS** | Only viable configuration for 671B |


### Minimum Requirements

| Resource | Requirement |
|----------|------------|
| HBM | 1.5 TB (64 NCs x 24 GB) |
| TP degree | 64 |
| LNC | 2 (trn2 platform default) |
| Instance | trn2.48xlarge |
| System RAM | 2 TB + 400GB NVMe swap (first-time sharding) |
| NVMe storage | 1.7 TB (compiled model + sharded weights) |
| Disk (HF weights) | 642 GB (FP8 safetensors) |

### SDK Configuration

| Component | Version |
|-----------|---------|
| NxDI | 0.8.0 |
| neuronx-cc | 2.23.6484 |
| torch | 2.9.0 |
| transformers | 4.57.6 |
| NXDI venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |
| vLLM | 0.13.0 (via `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`) |

## Testing

### Unit Tests (CPU only, no device needed)

```bash
cd contrib/models/DeepSeek-V3/
pytest test/unit/ -v
```

Tests: config parsing (11), RoPE (3), router (5), weight conversion (11), router in weight_conversion (9) = **39 tests**.

### Integration Tests (needs 2+ NeuronCores)

```bash
# Mini model (default, tp=2):
cd contrib/models/DeepSeek-V3/
pytest test/integration/test_model.py --capture=tee-sys

# Full 671B model (tp=64, trn2.48xlarge):
DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-V3-0324-FP8 \
DEEPSEEK_COMPILED_PATH=/scratch/deepseek_v3_traced \
DEEPSEEK_TP_DEGREE=64 \
DEEPSEEK_SEQ_LEN=512 \
pytest test/integration/test_model.py --capture=tee-sys
```

Tests: model loads, generates, coherence, top-token valid, first-token HF match, TTFT, throughput = **7 tests**.

## Key Porting Challenges

1. **MLA incompatible with NeuronAttentionBase:** GQA projections don't apply to MLA's weight absorption. Built a custom `DeepseekV3Attention` class with its own TP sharding, KV cache, and softmax logic.

2. **YaRN RoPE interleaved layout:** Uses `rotate_fn` (interleaved) not `rotate_half` (split). No transpose needed — different from optimum-neuron which uses split layout.

3. **Dense layers 0-2:** Separate `DeepseekV3DenseMLP` class with `dense_intermediate_size=18432` (not MoE).

4. **FP8 dequantization:** Block-wise float8_e4m3fn with per-block scale factors. Added `_dequantize_fp8_state_dict()` for vectorized conversion during state dict loading.

5. **Expert fusion:** Per-expert `gate_proj` + `up_proj` fused into `gate_up_proj` tensor `[num_experts, hidden, 2*intermediate]` for ExpertMLPsV2 compatibility.

6. **KV cache stores compressed format:** `[k_pe | compressed_kv]` with dim 576 (rope_dim 64 + kv_lora_rank 512), not standard per-head KV.

7. **TP=32 HBM OOM:** 256 experts on every rank. Each rank carries ~40GB at TP=32 vs 24GB HBM limit. Fixed by using TP=64 and LNC=2.

## vLLM Integration

DeepSeek V3 can be served via vLLM with the Neuron backend:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export NEURON_COMPILED_ARTIFACTS=/scratch/vllm_bs1

VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3-0324-FP8 \
    --tensor-parallel-size 64 --max-model-len 512 --max-num-seqs 1 \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --port 8000 \
    --additional-config '{"override_neuron_config": {"logical_nc_config": 2, "enable_bucketing": false, "save_sharded_checkpoint": true}}'
```

**Note:** `NEURON_COMPILED_ARTIFACTS` env var is required to reuse pre-compiled NEFFs. Without it, vLLM deletes the compiled artifacts directory on each startup.

**Note:** `--additional-config` with `logical_nc_config: 2` is required. Without it, compilation fails with NCC_IBIR297 internal error.

### Fast Startup Recipe

1. Compile NEFFs (~2.5 min or 1s from cache)
2. Kill vLLM immediately after "Finished Compilation for all HLOs"
3. Symlink pre-sharded weights from to the artifacts dir
4. Restart vLLM with `NEURON_COMPILED_ARTIFACTS` — loads in ~8 min

## Example Checkpoints

- `deepseek-ai/DeepSeek-V3-0324` (FP8, 642GB, requires `trust_remote_code=True`)

## Maintainer

AWS Neuron

**Last Updated:** 2026-03-19
