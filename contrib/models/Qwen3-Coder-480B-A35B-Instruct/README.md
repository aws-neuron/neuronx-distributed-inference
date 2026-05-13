# Contrib Model: Qwen3-Coder-480B-A35B-Instruct on AWS Trainium2

Optimized NxD Inference configuration for serving **Qwen3-Coder-480B-A35B-Instruct** on `trn2.48xlarge` via vLLM.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3-Coder-480B-A35B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)
- **Model Type:** Mixture-of-Experts (MoE) decoder-only transformer
- **Total Parameters:** 480B (35B active per token)
- **License:** Check HuggingFace model card

## Architecture Details

Qwen3-Coder-480B shares the `qwen3_moe` architecture with Qwen3-235B-A22B but differs in several dimensions that affect Neuron compilation and optimization:

| Parameter | Qwen3-Coder-480B | Qwen3-235B-A22B |
|-----------|-------------------|------------------|
| num_key_value_heads | **8** | 4 |
| num_attention_heads | **96** | 64 |
| head_dim | **192** | 128 |
| hidden_size | **6144** | 5120 |
| num_hidden_layers | **62** | 94 |
| num_experts | **160** | 128 |
| num_experts_per_tok | 8 | 8 |
| moe_intermediate_size | 2560 | 2560 |
| max_position_embeddings | 262144 | 131072 |

These differences -- particularly **8 KV heads** (vs 4) and **head_dim=192** (vs 128) -- required specific optimization work on SDK 2.28 to achieve full NKI kernel compatibility and determine HBM-safe operating points.

## Hardware Requirements

- **Instance:** `trn2.48xlarge` (64 NeuronCores, 32 Neuron devices)
- **LNC Mode:** LNC=2 (default, 24 GB HBM per logical core)
- **Disk:** 900+ GB for model weights (241 safetensor shards)
- **RAM:** ~1.5 TB system RAM during weight loading
- **Neuron SDK:** 2.28+
- **DLAMI:** `Deep Learning AMI Neuron (Ubuntu 24.04) 20260227`

## Quick Start (vLLM)

### 1. Download Model

```bash
pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --local-dir /mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/
```

### 2. Launch Server (Recommended Config)

```bash
# Activate the pre-installed Neuron venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Set compilation cache directory
export NEURON_COMPILED_ARTIFACTS=/mnt/nvme/neuron_cache
export NEURON_CC_FLAGS="--retry_failed_compilation"
export VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference'
export MALLOC_ARENA_MAX=64

mkdir -p $NEURON_COMPILED_ARTIFACTS

python -m vllm.entrypoints.openai.api_server \
  --model=/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/ \
  --tensor-parallel-size=64 \
  --max-num-seqs=16 \
  --max-model-len=8192 \
  --additional-config='{"override_neuron_config": {
    "async_mode": true,
    "attn_kernel_enabled": true,
    "batch_size": 16,
    "cc_pipeline_tiling_factor": 2,
    "cp_degree": 1,
    "ctx_batch_size": 1,
    "enable_bucketing": true,
    "flash_decoding_enabled": false,
    "fused_qkv": true,
    "is_continuous_batching": true,
    "logical_nc_config": 2,
    "max_context_length": 8192,
    "mode_mask_padded_tokens": true,
    "moe_ep_degree": 1,
    "moe_tp_degree": 64,
    "qkv_cte_nki_kernel_fuse_rope": true,
    "qkv_kernel_enabled": true,
    "qkv_nki_kernel_enabled": true,
    "scratch_pad_size": 1024,
    "seq_len": 8192,
    "sequence_parallel_enabled": true,
    "torch_dtype": "bfloat16",
    "tp_degree": 64
  }}' \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port=8000
```

First launch compiles NEFFs (~22 min for 7 CTE + 7 TKG buckets). Subsequent launches load from cache (~10 min).

### 3. Test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/nvme/Qwen3-Coder-480B-A35B-Instruct/",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 128,
    "temperature": 0.6
  }' | python3 -m json.tool
```

## Configurations

Two pre-validated configs are provided in `configs/`:

### Config A: `throughput_optimized.json` (Recommended)

- **seq_len=8192, batch_size=16, pure TP=64**
- Auto-bucketing (128, 256, 512, 1024, 2048, 4096, 8192)
- All NKI QKV kernels + fused RoPE enabled
- Best for: Production serving, high concurrent throughput

### Config B: `long_context.json`

- **seq_len=16384, batch_size=8, pure TP=64**
- Single bucket (16384)
- Best for: Applications requiring 16K context, at the cost of throughput

### Configuration Comparison

| Metric | Config A (8192/BS=16) | Config B (16384/BS=8) |
|--------|----------------------|----------------------|
| TTFT (short prompt) | **0.85s** | 18.47s |
| Decode throughput | **15.3 tok/s** | ~8.5 tok/s |
| Max concurrent requests | **16** | 8 |
| Aggregate TPS @ 1 concurrent | **14.73** | 8.37 |
| Aggregate TPS @ 4 concurrent | **43.42** | 9.74 |
| Aggregate TPS @ 8 concurrent | **73.23** | 10.41 |
| Max context length | 8192 | **16384** |
| Compile time | ~22 min | ~15 min |
| Weight load time | ~10 min | ~10 min |
| Generation quality | Correct | Correct |

## Benchmark Results

Validated on `trn2.48xlarge`, SDK 2.28, `Deep Learning AMI Neuron (Ubuntu 24.04) 20260227`.

### Config A: Throughput Optimized (Recommended)

| Concurrency | Aggregate TPS | Avg per-request TPS | Avg Latency |
|-------------|--------------|---------------------|-------------|
| 1 | 14.73 | 14.73 | 17.4s |
| 2 | 28.14 | 14.07 | 18.2s |
| 4 | 43.42 | 10.86 | 23.6s |
| 8 | 73.23 | 9.15 | 28.0s |

- **TTFT (short prompt):** 0.85s (auto-bucketing selects optimal bucket)
- **TTFT (8K prompt):** ~7.1s
- **Peak single-request decode:** 15.3 tok/s

### Config B: Long Context (16384)

| Concurrency | Aggregate TPS | Avg per-request TPS | Avg Latency |
|-------------|--------------|---------------------|-------------|
| 1 | 8.37 | 8.37 | 30.6s |
| 4 | 9.74 | 2.44 | 105.1s |
| 8 | 10.41 | 1.30 | 196.3s |

## Known Issues and Limitations

### 1. `flash_decoding_enabled` Must Be `false`

Enabling `flash_decoding_enabled=true` causes:
```
AssertionError: kv_shared parallel group is not initialized
```
during `attention_tokengen`. The 8 KV heads / 96 Q heads ratio in pure TP=64 config is incompatible with the flash decoding path.

### 2. NKI QKV Kernel Incompatible with `attention_dp_degree > 1`

`qkv_nki_kernel_enabled=true` and `qkv_cte_nki_kernel_fuse_rope=true` fail with `attention_dp_degree > 1`:
```
AssertionError: Output bfloat16 %out_tensor(1, 1024, 3584) has no store def
```
Workaround: Use only `qkv_kernel_enabled=true` (legacy kernel) when `attention_dp_degree > 1`.

### 3. Context Parallelism (`cp_degree=16`) Causes ICE with `head_dim=192`

The Coder's `head_dim=192` creates attention tensors (64x192=12288) that exceed SBUF partition limits during CTE linking:
```
Allocated memory out of bound {I-xxx}@SB<0,229376>(128x12288)
```
`cp_degree=8` compiles successfully, but produces garbage output for `seq_len > 8192`. Only `cp_degree=1` (pure TP) produces correct results at all sequence lengths.

### 4. Expert Parallelism Requires `batch_size >= 20`

With `moe_ep_degree=32`, selective loading triggers when `batch_size * top_k / num_experts < 1.0`. For the Coder (160 experts, top_k=8): `batch_size >= 20` is required to bypass this.

### 5. Maximum Context Length

| seq_len | Max batch_size | Status |
|---------|---------------|--------|
| 8192 | 16 | Working (recommended) |
| 12288 | - | NEFF load OOM |
| 16384 | 8 | Working (reduced throughput) |

HBM is the limiting factor: 24 GB per core at LNC=2, with `hidden_size=6144` producing larger I/O tensors than the 235B model.

### 6. `on_device_sampling_config` Adds Per-Token Overhead

Removing `on_device_sampling_config` from the NeuronConfig (letting vLLM handle sampling on CPU) improves decode throughput by ~11% (13.8 -> 15.3 tok/s) and aggregate throughput by 3.3x when combined with auto-bucketing.

### 7. Auto-Bucketing is Critical for TTFT

Without explicit bucket lists, NxDI auto-generates buckets (128, 256, 512, 1024, 2048, 4096, 8192). This reduces TTFT from 7.14s to 0.85s for short prompts (8.4x improvement), since short prompts no longer pad to the maximum sequence length.

## Compilation Strategy

First compilation takes ~22 minutes (Config A) or ~15 minutes (Config B). Use `NEURON_COMPILED_ARTIFACTS` to cache NEFFs:

```bash
export NEURON_COMPILED_ARTIFACTS=/mnt/nvme/neuron_cache
mkdir -p $NEURON_COMPILED_ARTIFACTS
```

Subsequent server startups skip compilation and load from cache (~10 min for weight loading).

If compilation fails with OOM, reduce `batch_size` first. The model's `hidden_size=6144` uses more HBM than the 235B model per sequence.

## NxDI Direct Usage

See `src/generation_qwen3_coder_demo.py` for a standalone NxDI example (without vLLM).

## Compatibility Matrix

| Instance/SDK | 2.28+ | 2.27 and earlier |
|--------------|-------|------------------|
| trn2.48xlarge | Working (recommended) | Limited (no NKI QKV kernels) |
| trn2.3xlarge | Not enough NeuronCores (needs TP=64) | N/A |
| trn1 / Inf2 | Not tested | Not tested |

## Testing

Run the benchmark script against a running vLLM server:

```bash
# Start vLLM server (see Quick Start above), then in another terminal:
python3 contrib/models/Qwen3-Coder-480B-A35B-Instruct/test/integration/test_model.py
```

Or with pytest:

```bash
pytest contrib/models/Qwen3-Coder-480B-A35B-Instruct/test/integration/test_model.py -v --capture=tee-sys
```

## Example Checkpoints

- [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)

## Maintainer

Jim Burtoft - Annapurna Labs

**Last Updated:** 2026-03-10
