# YuE Music Generation on AWS Neuron (NxDI)

Full-song music generation from lyrics using [YuE](https://github.com/multimodal-art-projection/YuE) (M-A-P/HKUST) on AWS Trainium via neuronx-distributed-inference (NxDI).

YuE is a two-stage autoregressive pipeline: S1 (7B LLaMA) generates coarse audio codec tokens from lyrics, S2 (1B LLaMA) refines them via teacher-forcing, then xcodec_mini decodes tokens to audio waveforms. Both S1 and S2 are standard `LlamaForCausalLM` architectures with expanded vocabularies (~84K tokens).

## Requirements

- **Instance**: trn2.3xlarge (or larger)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 24.04) 20260227 or later (SDK 2.28)
- **Storage**: 300GB+ EBS (gp3)
- **Neuron venv**: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/`

## Quick Start

```bash
# SSH to your trn2 instance
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>

# Copy this directory to the instance
scp -i ~/.ssh/your-key.pem -r YuE-Music-Generation/ ubuntu@<instance-ip>:~/

# Run setup (downloads models, installs deps, copies scripts)
cd ~/YuE-Music-Generation
chmod +x setup.sh
./setup.sh /mnt/models

# Activate venv and run
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
cd /mnt/models
python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt --seed 123
```

First run compiles models to Neuron IR (~7 min). Subsequent runs use `--skip-compile`.

### With NKI Kernel Optimization (Recommended)

Enable fused NKI MLP kernels for a **20% speedup** on token generation:

```bash
python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt --seed 123 --nki-kernels
```

First run with `--nki-kernels` requires a separate compilation (~7 min). Subsequent runs use `--skip-compile`.

### With S2 Batching

Process multiple S2 chunks simultaneously for better throughput:

```bash
python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt --seed 123 \
    --nki-kernels --s2-batch-size 2
```

## Architecture

```
                    +-----------------+
  genre.txt ------->|                 |
  lyrics.txt ------>|  Orchestrator   |----> vocals.wav
                    | (yue_e2e_neuron)|----> instrumentals.wav
                    |                 |----> mix.wav
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v-------+ +---v----+ +-------v--------+
     | S1 Worker (7B) | | S2 (1B)| | xcodec_mini    |
     | TP=2, NxDI     | | TP=1   | | CPU decode     |
     | CFG bs=2       | | NxDI   | |                |
     +----------------+ +--------+ +----------------+
       subprocess         subprocess   main process
```

**Why subprocesses?** NxDI models with different TP degrees cannot coexist in the same process -- the Neuron runtime segfaults during warmup. Each stage runs in its own subprocess with args passed via the `YUE_STAGE_ARGS` environment variable.

## Files

| File | Description |
|------|-------------|
| `setup.sh` | Downloads models, installs deps, copies scripts to MODEL_DIR |
| `src/yue_e2e_neuron.py` | E2E orchestrator (entry point) |
| `src/yue_stage1_worker.py` | S1 (7B) subprocess worker with CFG |
| `src/yue_stage2_worker.py` | S2 (1B) subprocess worker (teacher-forcing + batching) |
| `src/nki_mlp_patch.py` | NKI MLP kernel monkeypatch (enables fused MLP for token-gen) |
| `genre.txt` | Sample genre tags |
| `lyrics.txt` | Sample lyrics (verse + chorus) |

## Options

```
python yue_e2e_neuron.py --genre_txt GENRE --lyrics_txt LYRICS [OPTIONS]

Required:
  --genre_txt FILE          Genre description file
  --lyrics_txt FILE         Lyrics file with [verse]/[chorus] sections

Optional:
  --output_dir DIR          Output directory (default: ./output_neuron)
  --seed INT                Random seed (default: 42)
  --max_new_tokens INT      Max tokens per segment (default: 3000)
  --run_n_segments INT      Number of lyrics segments (default: 2)
  --skip-compile            Skip Neuron compilation (use cached)
  --nki-kernels             Enable NKI MLP TKG fused kernels (20% speedup)
  --s2-batch-size N         S2 batch size (default: 1, try 2 for best pipeline time)
  --s1-tp-degree N          S1 TP degree (default: 2, use 1 for LNC=1 single-core)
  --no-cfg                  Disable CFG (faster, lower vocal quality)
  --no-kv-cache             Disable KV-cache S2 optimization (use legacy loop)
  --guidance-scale-first F  CFG scale, first segment (default: 1.5)
  --guidance-scale-rest F   CFG scale, subsequent segments (default: 1.2)
  --rescale                 Rescale audio to max amplitude
```

## Classifier-Free Guidance (CFG)

CFG is critical for vocal quality. Without it, vocals collapse to near-silence.

The implementation uses a **custom generation loop** (not HF `generate()`) because:
1. HF `generate()` with batch_size=2 samples independently per row, causing KV cache divergence
2. CFG requires blending logits from conditional (row 0) and unconditional (row 1) before sampling
3. The **same** sampled token must be fed to **both** rows to keep KV caches synchronized

Key details:
- S1 compiled with `batch_size=2`, `padding_side=right` (NxDI requires right-padding for bs>1)
- Row 0: full conditional prompt (lyrics + genre + context)
- Row 1: unconditional (last token only, right-padded)
- CFG formula in log-probability space: `uncond_lp + scale * (cond_lp - uncond_lp)`

## NKI Kernel Optimization

The `--nki-kernels` flag enables NxDI's built-in NKI MLP TKG (token generation) kernel, which fuses RMSNorm + Gate/Up projection + SiLU activation + Down projection into a single kernel that keeps all intermediates in SBUF (on-chip SRAM), eliminating 5-6 HBM round-trips per transformer layer during token generation.

### How It Works

The `nki_mlp_patch.py` monkeypatch routes MLP calls based on batch*seqlen:
- **Token generation** (B*S <= 128): Uses NxDI's `nki_mlp_tkg_isa_kernel` -- no dimension limits
- **Context encoding** (B*S > 128): Falls back to manual matmul with transposed weights, bypassing the compiler's 4096 intermediate dimension limit

This is necessary because:
1. The **TKG NKI kernel** has no dimension limit and works for both S1 (intermediate=11008) and S2 (intermediate=5504)
2. The **CTE MLP kernel** has a hard `intermediate_size <= 4096` limit in the closed-source compiler backend (`walrus/inline_bir_kernel/src/kernels_impl/mlp.cpp:196`)

### Why Only MLP?

Exhaustive analysis of all available NKI TKG kernels found that only the MLP kernel is compatible with YuE's model configurations:

| NKI Kernel | Status | Blocking Constraint |
|-----------|--------|-------------------|
| `mlp_tkg_nki_kernel` | **Working** | None |
| `mlp_kernel` (CTE) | Blocked | Compiler limit: `intermediate_size <= 4096` |
| `attn_block_tkg_nki_kernel` | Blocked | Two independent blockers: (1) `kv_heads == 1` hardcoded at 16 sites in kernel, (2) PSUM capacity `I <= 4096` for SBUF output |
| `attn_tkg_nki_kernel` | Blocked | `kv_heads == 1` assumed in Python calling code |
| `qkv_kernel` / `fused_qkv` | Blocked | Shape mismatch with GQA-to-MHA conversion |

The fused attention block kernel is designed for large models at high TP degrees (e.g., Llama3-70B at TP=32) where per-rank head counts are small enough to fit in PSUM. For 7B at TP=1-2, the QKV output is too large.

### Profiling Results

Neuron hardware profiling confirms the MLP kernel eliminates the GPSIMD bottleneck:

| Metric | Baseline S1 TKG | NKI S1 TKG | Change |
|--------|-----------------|------------|--------|
| Per-step time | 52.3ms | 43.2ms | -17% |
| DMA active | 91.7% | 77.3% | -14pt |
| GPSIMD active | 51.9% | **1.6%** | -50pt |
| Tensor Engine | 40.0% | 33.6% | -6pt |

The GPSIMD collapse (51.9% -> 1.6%) confirms elimination of unfused RMSNorm/SiLU operations. After NKI optimization, token generation is purely DMA/bandwidth-bound (weight streaming from HBM).

## Performance

Benchmarked on trn2.3xlarge with SDK 2.28 (seed=123, 2 segments, CFG enabled):

### GPU Comparison (CFG enabled, ~29s audio)

| Metric | L4 GPU (24GB) | trn2 Baseline | trn2 + NKI + bs=2 | Neuron vs GPU |
|--------|--------------|---------------|-------------------|---------------|
| S1 gen | 300.3s (9.7 tok/s) | 85s (34.5 tok/s) | 85s (34.5 tok/s) | **3.5x faster** |
| S2 gen | 1004.1s (105s/chunk) | 317s (40s/chunk) | 235s (bs=2) | **4.3x faster** |
| Pipeline total | 1461.5s | 514s | ~440s | **3.3x faster** |
| RTF | 50.3x | 17.6x | ~15x | -- |

Benchmarked on g6.xlarge (NVIDIA L4, 24GB, PyTorch 2.9.1, BF16, SDPA).

### Default Configuration (LNC=2, TP=2/1)

| Stage | Metric | Baseline | KV-cache | KV-cache + NKI |
|-------|--------|----------|----------|---------------|
| **S1 (7B, CFG, TP=2)** | Throughput | 34.5 tok/s | 34.5 tok/s | 34.5 tok/s |
| | Generation time | 85s | 85s | 85s |
| **S2 (1B, TP=1)** | Per-chunk time | ~130s | ~40s | ~32s |
| | Total S2 time | 1271s | 317s | 252s |
| **xcodec_mini (CPU)** | Decode time | 3.1s | 3.2s | 3.2s |
| **Full Pipeline** | Total wall time | 1399s | 514s | **~440s** |
| | Audio duration | 29.3s | 29.3s | 29.3s |
| | **Real-time factor** | 47.8x | 17.6x | **~15x** |

### LNC=1 Configuration (TP=1, Single-Core, NKI Enabled)

Running both models at TP=1 on LNC=1 cores (8 logical cores, 12GB HBM each):

| Metric | Baseline | NKI MLP TKG | Change |
|--------|----------|-------------|--------|
| S1 gen time | 185.8s (16.1 tok/s) | 155.7s (19.3 tok/s) | **-16.2%, +19.9% throughput** |
| S2 gen time | 504.4s (100.9s/batch) | 393.5s (78.7s/batch) | **-22.0%** |
| Total compute | 690.2s | 549.2s | **-20.4%** |
| Compute RTF | 23.0x | 18.3x | **-20.4%** |

### S2 Batch Size Comparison (29.3s audio, 8 chunks)

| S2 Batch Size | S2 Compute | Pipeline Total | RTF | Per-Core Throughput |
|--------------|-----------|---------------|-----|-------------------|
| 1 (default) | 320s | 515s | 17.6x | 0.025 chunks/s |
| **2 (best)** | **235s** | **467s** | **16.0x** | 0.034 chunks/s |
| 4 | 215s | 539s | 18.4x | 0.037 chunks/s |
| 8 | 198s | 678s | 23.2x | 0.040 chunks/s |

**Key insight**: Per-core throughput improves with batch size, but model load overhead grows faster. **Batch_size=2 gives the best end-to-end pipeline time.**

## Customization

### Model Directory

Set `MODEL_DIR` environment variable to change the model directory:

```bash
export MODEL_DIR=/path/to/models
python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt
```

### LNC=1 Mode

To use LNC=1 (8 logical cores with 12GB HBM each instead of 4 with 24GB):

```bash
# Set LNC=1 (requires reboot)
echo 'NEURON_LOGICAL_NC_CONFIG=1' | sudo tee /etc/environment
sudo reboot

# After reboot, verify 8 cores
neuron-ls

# Run with TP=1 for both S1 and S2
python yue_e2e_neuron.py --genre_txt genre.txt --lyrics_txt lyrics.txt \
    --seed 123 --nki-kernels --s1-tp-degree 1
```

### Lyrics Format

Lyrics must use section tags:

```
[verse]
First verse lyrics here
Multiple lines supported

[chorus]
Chorus lyrics here
```

Supported tags: `[verse]`, `[chorus]`, `[bridge]`, `[outro]`, `[intro]`, etc.

### Genre Tags

Free-form genre description, space-separated:

```
pop electronic upbeat female vocal
```

```
rock guitar male vocal energetic drums
```

## Known Limitations

1. **S2 still the bottleneck**: Even with KV-cache optimization (3.3x speedup) and NKI kernels (20% speedup), S2 teacher-forcing accounts for ~60-70% of pipeline time.
2. **DMA-bound token generation**: After NKI MLP optimization, token generation is 77% DMA active -- the remaining time is dominated by weight streaming from HBM, which is a fundamental bandwidth characteristic.
3. **Single-song pipeline**: No batching across songs. One song at a time per instance.
4. **Unlimited segments, truncated context**: More lyrics segments produce longer songs (tested up to 6 segments = 132s). Context is truncated to ~1095 tokens for segments 2+.
5. **NKI attention kernel blocked**: The fused attention block TKG kernel requires `kv_heads == 1` and PSUM capacity `I <= 4096`, which excludes all YuE configurations at TP <= 2.

## License

YuE model weights and inference code: Apache 2.0 (M-A-P/HKUST)
These Neuron adaptation scripts: Apache 2.0
