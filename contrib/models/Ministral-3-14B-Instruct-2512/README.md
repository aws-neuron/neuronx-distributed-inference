# Contrib Model: Ministral-3-14B-Instruct-2512 (Leanstral)

NeuronX Distributed Inference contrib for Ministral-3-14B-Instruct-2512 on AWS Trainium 2.
This model uses Mistral's 14B dense GQA text decoder with 8 KV heads, served via the
LlamaForCausalLM code path in NxDI with custom NKI kernels for multi-KV-head attention.

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-3-14B-Instruct-2512`
- **Architecture:** Dense GQA (runs as LlamaForCausalLM via hf-overrides)
- **Parameters:** 14B (40 layers, hidden=5120, 32 Q / 8 KV heads, d_head=128)
- **Vocab:** 32768 (text-only extraction from VL checkpoint)
- **License:** Check HuggingFace model card (gated access)

## Architecture Details

- 40 layers, hidden\_size=5120 (mapped to 3584 for text extraction), intermediate\_size=16384
- num\_attention\_heads=32, num\_kv\_heads=8, head\_dim=128, rope\_theta=1e9
- At TP=4: q\_heads\_per\_rank=8, kv\_heads\_per\_rank=2
- Original checkpoint is FP8 E4M3 — dequantized to BF16 via `extract_text_model.py`

### Key Adaptations for SDK 2.29

1. **LlamaForCausalLM code path**: vLLM 0.16 auto-promotes MistralForCausalLM to Pixtral.
   We use `--hf-overrides '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}'`
   to force the Llama code path, which handles the GQA sharding natively.

2. **QKV NKI kernel (recommended)**: The standard NxDI QKV NKI kernel provides the best
   per-request and aggregate throughput. No additional kernel patches are needed beyond
   the QKV fixes in `setup_patches.py`.

3. **FP8→BF16 text extraction**: The HuggingFace checkpoint is a VL model with FP8 weights.
   `extract_text_model.py` strips vision keys, dequantizes FP8→BF16, fixes tokenizer issues,
   and writes a clean text-only checkpoint.

4. **Multi-KV-head TKG kernel**: Modified `attention_block_tkg` kernel supporting
   kv\_heads\_per\_rank > 1 via virtual-batch approach. With LNC=2 (default on trn2.3xlarge),
   the TKG kernel **matches baseline TPOT at BS=4** and adds only 5-9% overhead at BS=8.
   See [TKG Kernel Results](#tkg-kernel-results) for details.

## Prerequisites

- **SDK 2.29** (neuronx-cc >= 2.24, neuronx-distributed-inference >= 0.9, vLLM 0.16 + vllm-neuron 0.5)
- **trn2.3xlarge** (TP=4, LNC=2, 96 GB HBM)
- **Model checkpoint**: `mistralai/Ministral-3-14B-Instruct-2512` from HuggingFace (gated)
- **Disk**: ~300 GB EBS for checkpoint + compiled model artifacts

### Environment Setup

```bash
# Activate pre-installed vLLM 0.16 environment (SDK 2.29)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Install aiohttp for benchmark script
pip install aiohttp
```

## Quick Start

### Step 1: Download Model

```bash
huggingface-cli download mistralai/Ministral-3-14B-Instruct-2512 \
  --local-dir /home/ubuntu/models/Ministral-3-14B-Instruct-2512
```

### Step 2: Extract Text-Only BF16 Checkpoint

```bash
python src/extract_text_model.py \
  --input-dir /home/ubuntu/models/Ministral-3-14B-Instruct-2512 \
  --output-dir /home/ubuntu/models/Ministral-3-14B-text-bf16
```

This produces a ~27 GB checkpoint with:
- 6 safetensors shards (BF16, vision keys removed, `language_model.` prefix stripped)
- Fixed `tokenizer_config.json` (removes Pixtral processor references)
- Proper `config.json` for LlamaForCausalLM

### Step 3: Apply Runtime Patches

```bash
python src/setup_patches.py
```

Applies 6 patches to the installed NxDI/nkilib packages:
1. `rms_norm_eps` pass-through in model base
2. nkilib QKV kernel epsilon guard
3. neuronxcc QKV kernel epsilon guard
4. `convert_state_dict_to_fused_qkv` fix for non-standard head counts
5. Fused RMSNorm config support
6. Multi-KV TKG kernel + NKI 0.3.0 V cache fix + attention adapter

### Step 4: Launch vLLM Server

```bash
export NEURON_CC_FLAGS="--auto-cast=matmult"

python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/Ministral-3-14B-text-bf16 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --block-size 8 \
  --no-enable-prefix-caching \
  --port 8000 \
  --hf-overrides '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}' \
  --additional-config '{"override_neuron_config": {"fused_qkv": true, "qkv_nki_kernel_enabled": true, "qkv_kernel_enabled": true}}'
```

First launch compiles the model (~5 minutes). Subsequent launches use the NCC cache.

For higher batch sizes (BS=8), change `--max-num-seqs 8`.

### Step 5: Query

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "leanstral",
       "messages": [{"role": "user", "content": "What is the capital of France?"}],
       "max_tokens": 256}'
```

## Performance Results

Measured on trn2.3xlarge (TP=4, LNC=2, SDK 2.29) via vLLM 0.16:

### vLLM Serving — Baseline (QKV NKI Kernel, Recommended)

**BS=4 (`--max-num-seqs 4`):**

| Workload | Conc | TTFT P50 (ms) | tok/s P50 | TPOT P50 (ms) | E2E P50 (ms) |
|----------|------|---------------|-----------|---------------|--------------|
| short-short (128/128) | 1 | 100.6 | 63.3 | 15.8 | 2106.9 |
| short-short (128/128) | 4 | 200.0 | 58.5 | 15.9 | 2465.7 |
| short-long (128/512) | 1 | 101.4 | 62.8 | 15.9 | 8234.2 |
| short-long (128/512) | 4 | 200.0 | 62.3 | 15.9 | 8498.9 |
| long-short (2048/128) | 1 | 303.6 | 57.4 | 17.4 | 2514.7 |
| long-short (2048/128) | 4 | 609.2 | 50.9 | 17.3 | 3400.3 |
| long-long (2048/512) | 1 | 304.0 | 57.7 | 17.3 | 9156.5 |
| long-long (2048/512) | 4 | 608.9 | 55.9 | 17.3 | 10053.9 |

**BS=8 (`--max-num-seqs 8`):**

| Workload | Conc | TTFT P50 (ms) | tok/s P50 | TPOT P50 (ms) | E2E P50 (ms) |
|----------|------|---------------|-----------|---------------|--------------|
| short-short (128/128) | 1 | 102.7 | 59.7 | 16.7 | 2229.5 |
| short-short (128/128) | 4 | 201.4 | 57.0 | 16.8 | 2519.0 |
| short-short (128/128) | 8 | 348.6 | 53.6 | 16.8 | 2907.6 |
| long-long (2048/512) | 1 | 306.8 | 53.5 | 18.7 | 9864.4 |
| long-long (2048/512) | 4 | 613.1 | 51.5 | 18.8 | 10813.4 |
| long-long (2048/512) | 8 | 1069.3 | 49.5 | 18.7 | 11982.9 |

**Notes:**
- The tok/s column reports **per-request** throughput. At conc=N, the **aggregate system
  throughput** is ~Nx higher (e.g., TPOT=15.9ms → 4/0.0159 = **252 tok/s aggregate** at BS=4).
- TPOT scales gracefully with batch size: 15.8ms (BS=4) → 16.7ms (BS=8) = only 6% increase.

### Aggregate Throughput Comparison

| SDK | Config | Aggregate tok/s (BS=4) | Per-request tok/s (BS=1) |
|-----|--------|----------------------|------------------------|
| **2.29** | QKV NKI kernel (baseline) | **~252** (4/TPOT) | 63.3 |
| **2.29** | TKG kernel (LNC=2) | **~255** (4/TPOT) | 63.5 |
| 2.28 | Fused QKV+TKG | 213.7 | 71.0 |
| GPU | H100 FP8 | 140.3 | — |

At BS=4, both baseline and TKG configs **exceed H100 by 1.8x**. The TKG kernel with LNC=2
matches baseline throughput while fusing the KV cache update into the attention kernel.

## TKG Kernel Results

The multi-KV-head TKG (Token-Key-Generation) kernel fuses attention computation and KV cache
update into a single NKI kernel. It uses a virtual-batch approach
(`B_virt = batch_size * kv_heads_per_rank`) to handle GQA models with kv\_heads\_per\_rank > 1.

With **LNC=2** (default on trn2.3xlarge, grid=2), the TKG kernel shards the virtual-batch
computation across 2 NeuronCores per physical core, effectively halving the HBM load overhead.
This eliminates the performance regression seen with LNC=1 (grid=1).

### TKG Launch Command

To enable TKG, add two flags to `--additional-config`:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/Ministral-3-14B-text-bf16 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --block-size 8 \
  --no-enable-prefix-caching \
  --port 8000 \
  --hf-overrides '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}' \
  --additional-config '{"override_neuron_config": {"fused_qkv": true, "qkv_nki_kernel_enabled": true, "qkv_kernel_enabled": true, "attn_block_tkg_nki_kernel_enabled": true, "attn_block_tkg_nki_kernel_cache_update": true}}'
```

### TKG Performance (LNC=2, SDK 2.29)

**BS=4 TKG (`--max-num-seqs 4`):**

| Workload | Conc | TTFT P50 (ms) | tok/s P50 | TPOT P50 (ms) | E2E P50 (ms) |
|----------|------|---------------|-----------|---------------|--------------|
| short-short (128/128) | 1 | 101.9 | 63.5 | 15.7 | 2101.5 |
| short-short (128/128) | 4 | 247.1 | 59.4 | 15.7 | 2380.8 |
| short-long (128/512) | 1 | 102.2 | 63.4 | 15.8 | 8166.4 |
| short-long (128/512) | 4 | 249.7 | 61.0 | 16.1 | 8661.1 |
| long-short (2048/128) | 1 | 304.5 | 56.6 | 17.7 | 2550.1 |
| long-short (2048/128) | 4 | 754.9 | 48.6 | 17.1 | 3362.8 |
| long-long (2048/512) | 1 | 303.3 | 58.8 | 17.0 | 8992.8 |
| long-long (2048/512) | 4 | 753.1 | 56.3 | 16.9 | 9832.8 |

**BS=8 TKG (`--max-num-seqs 8`):**

| Workload | Conc | TTFT P50 (ms) | tok/s P50 | TPOT P50 (ms) | E2E P50 (ms) |
|----------|------|---------------|-----------|---------------|--------------|
| short-short (128/128) | 1 | 101.6 | 57.3 | 17.5 | 2317.7 |
| short-short (128/128) | 4 | 246.6 | 53.7 | 17.5 | 2613.6 |
| short-short (128/128) | 8 | 390.3 | 50.8 | 17.5 | 2981.1 |
| long-long (2048/512) | 1 | 301.9 | 49.6 | 20.1 | 10612.0 |
| long-long (2048/512) | 4 | 750.2 | 47.3 | 20.3 | 11560.2 |
| long-long (2048/512) | 8 | 1197.7 | 45.5 | 20.3 | 12705.5 |

### TKG vs Baseline TPOT Comparison

With LNC=2, the TKG overhead is minimal at BS=4 and modest at BS=8:

| Batch Size | Workload | Baseline TPOT | TKG LNC=2 TPOT | Overhead |
|-----------|----------|---------------|-----------------|----------|
| BS=4 | short-short | 15.8ms | 15.7ms | **-0.6%** |
| BS=4 | long-long | 17.3ms | 17.0ms | **-1.7%** |
| BS=8 | short-short | 16.7ms | 17.5ms | **+4.8%** |
| BS=8 | long-long | 18.7ms | 20.1ms | **+7.5%** |

For comparison, with LNC=1 (grid=1) the overhead was 11% at BS=4 and 27% at BS=8.
LNC=2 sharding distributes the virtual-batch HBM loads across 2 NeuronCores, halving
the per-core overhead.

### Design Notes

The TKG kernel uses a "virtual batch" expansion where each KV head group becomes a separate
batch entry (`B_virt = BS * kv_heads_per_rank`). The inner attention kernel's batch loops
(`_compute_qk_matmul`, `_compute_pv_matmul_and_store`) iterate over `B_virt` entries, each
loading a KV cache slice from HBM. With LNC=2, the `grid=2` NKI sharding distributes these
iterations across 2 physical NeuronCores, effectively halving the HBM bandwidth pressure.

The kernel and adapter are included as a reference implementation for upstream multi-KV-head
TKG support in nkilib.

## Known Limitations

1. **TKG scaling at BS=8**: The multi-KV-head TKG kernel matches baseline at BS=4 but adds
   5-9% TPOT overhead at BS=8 due to the virtual-batch approach. For latency-critical BS=8
   workloads, the baseline QKV NKI kernel config may be preferred.

2. **KVDP not supported**: KV data parallelism is not compatible with the multi-KV-head
   kernel path.

3. **FP8 checkpoint**: The original checkpoint uses FP8 E4M3 weights. These are dequantized
   to BF16 during extraction. Runtime FP8 inference is not currently supported.

4. **Pixtral auto-promotion**: vLLM 0.16 auto-promotes Mistral models to Pixtral even with
   tokenizer fixes. The `--hf-overrides` flag is mandatory to force LlamaForCausalLM.

5. **Text-only**: This contrib extracts and serves only the text decoder. Vision-language
   inference requires the full VL model and additional patches not included here.

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 | Earlier |
|----------|----------|----------|---------|
| trn2.3xlarge (TP=4) | **Tested** | Tested (prior version) | Not supported |
| trn2.48xlarge | Not tested | Not tested | Not tested |
| trn1 / inf2 | Not supported | Not supported | Not supported |

## Source Files

| File | Description |
|------|-------------|
| `src/setup_patches.py` | SDK 2.29 runtime patch installer (6 patches) |
| `src/extract_text_model.py` | FP8→BF16 text-only checkpoint extraction |
| `src/attention_block_tkg_multi_kv.py` | Multi-KV-head TKG kernel (NKI 0.3.0 compatible) |
| `src/multi_kv_adapter.py` | TKG kernel adapter for attention_base.py |
| `src/fix_nki030.py` | NKI 0.3.0 compatibility fixes |
| `src/modeling_leanstral.py` | Legacy model class (SDK 2.28, reference only) |
| `src/patch_native_multi_kv.py` | Legacy adapter (SDK 2.28, reference only) |
| `bench.py` | Async streaming benchmark script |
| `test/integration/test_model.py` | Integration test |

## Upstream NxDI Gaps

This contrib identifies NxDI gaps that would benefit from upstream support:

1. **Multi-KV-head TKG kernel** — the bundled kernel hardcodes kv\_heads=1. The nki-library
   kernel fork adds `n_kv_heads` parameter with virtual-batch dispatch. With LNC=2 sharding
   (grid=2), performance matches baseline at BS=4 and has only 5-9% overhead at BS=8.
2. **Fused QKV conversion** — `convert_state_dict_to_fused_qkv` assumes standard Llama head
   ratios; non-standard ratios (32Q/8KV at TP=4) need a fix to compute interleave groups.
3. **RMS norm epsilon** — NxDI model base doesn't pass `rms_norm_eps` from config, defaulting
   to 1e-5 which differs from Mistral's 1e-5 (same in this case, but other models differ).

## Maintainer

Leanstral Project

**Last Updated:** 2026-04-26
