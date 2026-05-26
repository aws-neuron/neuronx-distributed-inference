# Contrib Model: Qwen2.5-Omni-7B

NeuronX Distributed Inference implementation of [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) with full multimodal support: text generation, image understanding, audio understanding, and text-to-speech.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-Omni-7B`
- **Model Type:** Multimodal encoder-decoder (Thinker + Vision + Audio + Talker + Token2Wav)
- **Architecture:** Qwen2-based text backbone with vision/audio encoders and speech synthesis
- **License:** Check HuggingFace model card

## Architecture Details

| Component | Runtime | TP | Parameters |
|-----------|---------|-----|------------|
| Thinker (text) | Neuron | 4 | hidden=3584, heads=28, kv_heads=4, layers=28 |
| Vision encoder | Neuron | 4 | embed=1280, heads=16, depth=32, SwiGLU MLP |
| Audio encoder | Neuron | 4 | d_model=1280, heads=20, layers=32, chunked attention |
| Talker | Neuron | 4 | hidden=896, heads=12, kv_heads=4, head_dim=128, layers=24, vocab=8448, fused embed (8448→896) |
| Token2Wav | Neuron | 1 | DiT: dim=1024, 22 blocks; BigVGAN: 6 upsample stages (T=256 NEFF + chunked overlap-add); fp32, single-core `torch_neuronx.trace`, fp32 ODE on CPU |

**Total state dict keys:** 2448 (Text: 339, Vision: 518, Audio: 489, Talker: 293, Token2Wav: 809)

Key features:
- **Thinker**: Architecturally identical to Qwen2.5-7B; reuses `NeuronQwen2ForCausalLM` with state-dict prefix remapping (28 heads / 4 TP = 7 per rank, 4 kv_heads / 4 TP = 1 per rank)
- **Vision encoder**: SwiGLU MLP, RMSNorm, separate QKV projections, PatchMerger (16 heads / 4 TP = 4 per rank)
- **Audio encoder**: Whisper-style with chunked attention. Hybrid CPU+Neuron: Conv1d frontend + chunking on CPU, 32 transformer layers on Neuron (20 heads / 4 TP = 5 per rank), AvgPool + LayerNorm + projection on CPU
- **Talker**: Neuron-compiled with fused embedding (embed_tokens 8448→3584 + thinker_to_talker_proj 3584→896 collapsed into 8448→896), explicit head_dim=128, 3D mRoPE, per-step thinker state injection via vision_embeddings (12 heads / 4 TP = 3 per rank, 4 kv_heads / 4 TP = 1 per rank). Auto-pads vision_embeddings to max_context_length for compiled bucket compatibility.
- **Token2Wav**: DiT transformer core (22 blocks) and BigVGAN vocoder both on Neuron, ODE sampling (Runge-Kutta 4, 10 steps), float32. Split architecture: CPU preprocessing (ECAPA-TDNN, codec embed, input embed, rotary) + Neuron DiT core + CPU ODE solver + Neuron BigVGAN (T=256 NEFF + chunked overlap-add at runtime; see "BigVGAN compile cap" under Performance). Automatic CPU fallback when mel_len exceeds compiled max or when `QWEN25_OMNI_BIGVGAN_NEURON=0`.

## Prerequisites

- **Instance**: trn2.48xlarge or trn2.3xlarge (4+ NeuronCores sufficient)
- **Weights**: Download from [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — the example scripts auto-download via `huggingface_hub.snapshot_download` on first run. Set `QWEN25_OMNI_MODEL_PATH=/path/to/local/snapshot` to skip the download and use a local copy (the directory must contain `config.json`).
- **Python dependencies** (on top of the NxDI venv):
  ```bash
  pip install soundfile          # writes WAV output in generate_qwen25_omni_speech.py
  pip install qwen-omni-utils[decord]   # process_mm_info() in generate_qwen25_omni.py for image/audio/video inputs
  ```
- **Pin all three Neuron models to the same core group**. Set `NEURON_RT_VISIBLE_CORES=0-3` before launching the speech pipeline so the Thinker (TP=4), Talker (TP=4), and the single-device Token2Wav DiT all live on the same four NeuronCores. Without this the Neuron runtime places the DiT NEFF on a different core group and every DiT forward pays a cross-group scheduling penalty (~30% slower). `examples/generate_qwen25_omni_speech.py` already sets this via `os.environ.setdefault` before any Neuron module is imported; if you embed the pipeline in your own entrypoint, do the same.
  ```bash
  export NEURON_RT_VISIBLE_CORES=0-3
  python examples/generate_qwen25_omni_speech.py
  ```

## Usage

### Text-only (Thinker)

```python
import sys
from pathlib import Path

# Make this contrib package's src/ importable (flat, per upstream contrib convention).
sys.path.insert(0, str(Path("contrib/models/Qwen2.5-Omni-7B/src").resolve()))
import _upstream_compat  # noqa: F401  (applies hf_adapter bug fix)

import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from modeling_qwen25_omni import (
    NeuronQwen25OmniForCausalLM,
    Qwen25OmniInferenceConfig,
)

model_path = "/path/to/Qwen2.5-Omni-7B/"
compiled_path = "/path/to/compiled/"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    seq_len=4096,
    max_context_length=4096,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95
    ),
)

config = Qwen25OmniInferenceConfig(
    neuron_config, load_config=load_pretrained_config(model_path)
)

model = NeuronQwen25OmniForCausalLM(model_path, config)
model.compile(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
adapter = HuggingFaceGenerationAdapter(model, tokenizer)
output = adapter.generate("What is quantum computing?", max_new_tokens=256)
```

### Multimodal (Vision + Audio + Speech)

The full multimodal pipeline (image / audio / text → text → speech) is wired up
in the runnable example entrypoints:

- `examples/generate_qwen25_omni.py` — text and image/audio understanding via
  `NeuronQwen25OmniMultimodalForCausalLM` + `Qwen25OmniMultimodalInferenceConfig`
  (see `run_text_only` / `run_multimodal`).
- `examples/generate_qwen25_omni_speech.py` — full Thinker → Talker → Token2Wav
  speech synthesis pipeline (compiles each component, sets
  `NEURON_RT_VISIBLE_CORES=0-3`, writes a WAV).

The pipeline calls these methods on the multimodal model in order:
`enable_audio_encoder` → `compile_audio_encoder` → `load_audio_encoder` →
`enable_talker` → `enable_token2wav(state_dict, speaker_dict_path="spk_dict.pt")`.
Refer to the example scripts above for the exact wiring.

## vLLM Integration

Qwen2.5-Omni can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron) for text-only inference. A patch is required for the nested config structure.

### Setup

```bash
# 1. Install vllm-neuron
pip install vllm-neuron

# 2. Apply the Qwen2.5-Omni patch
python perf_test/apply_vllm_neuron_patch_qwen25omni.py
```

### Serving

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen2.5-Omni-7B \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 4,
            "fused_qkv": false,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": false,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 4096,
            "seq_len": 4096,
            "is_continuous_batching": true,
            "enable_bucketing": true,
            "async_mode": true,
            "on_device_sampling_config": {
                "do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95
            }
        }
    }'
```

### Key vLLM Patch Changes

The patch (`perf_test/apply_vllm_neuron_patch_qwen25omni.py`) modifies vllm-neuron to:
- Extract text config from nested `thinker_config.text_config`
- Map `Qwen2_5OmniModel` architecture to `qwen2_5_omni` model type
- Handle layer count extraction for nested config

See `perf_test/3_bench_qwen25_omni_7b.sh` for full benchmark configurations.

## Performance

### Full Neuron Speech Pipeline (trn2.48xlarge, TP=4, BF16)

End-to-end full-utterance speech synthesis from
`examples/generate_qwen25_omni_speech.py`, prompt `"Say hello and briefly
introduce yourself in two sentences."`, speaker Ethan, **4-core shared
layout** (`NEURON_RT_VISIBLE_CORES=0-3`, DiT and BigVGAN collapsed onto
cores 0-3), `--greedy --seed 1234`, median of 3 runs:

| Stage | Time | Notes |
|-------|------|-------|
| Thinker (7B, Neuron TP=4, greedy) | 0.26s | 25 text tokens, ~10ms TPOT |
| Hidden state extraction (HF CPU) | 0.45s | one forward pass to harvest thinker states |
| Talker prep (projection, conditioning) | 0.18s | CPU |
| Talker (690M, Neuron TP=4, sampled, seeded) | 1.76s | 448 codec tokens, ~4ms TPOT, per-step thinker injection |
| Token2Wav (Neuron DiT + Neuron BigVGAN chunked) | 6.10s | mel_len=896 → 4 BigVGAN chunks × T=256 NEFF + cos² crossfade |
| **Pipeline total** | **8.76s** | **9.0s audio, RTF 0.97x** |

Model load (one-time cost, excluded from pipeline) takes ~5min on a cold
NVMe cache (DiT trace dominates); subsequent loads from the on-disk NEFF
cache are well under a minute.

#### Sampling configuration

Talker compiles **without** `OnDeviceSamplingConfig`: the on-device
sampling NEFF returns a 1-D token tensor and `outputs.logits` is `None`,
which causes the HF generation adapter to skip its `LogitsProcessorList`
— silently dropping `repetition_penalty=1.05`. Without that penalty
the talker mode-collapses into a single codec token (audible "ooooo")
after ~75 tokens. Falling back to CPU sampling (NEFF returns gathered
bf16 logits, HF samples on CPU) lets `repetition_penalty` take effect
and adds <2ms per step. `run_talker` calls `torch.manual_seed(seed)`
right before `talker_adapter.generate` so codec sequences are
reproducible across runs on a single host (cross-hardware byte
equality is unreachable: CUDA RNG ≠ CPU RNG, and bf16 matmul numerics
diverge between H100 and Trn2).

The 8-core split layout (`NEURON_RT_VISIBLE_CORES=0-7`, DiT on core 4,
BigVGAN on core 5) primarily benefits streaming / TTFB by overlapping
DiT with talker decode; for full-utterance synthesis the two layouts
land within noise of each other.

### Trn2 vs H100 (full-utterance, BF16, aligned sampling)

Same prompt, speaker, sampling kwargs (`do_sample=False` for thinker;
talker `do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
repetition_penalty=1.05`), and seed (`1234`) on both platforms. H100 row
is the median of runs 2-4 from `examples/test_gpu_baseline_bench.py`
(run 1 is warmup-only and excluded); Trn2 row is the median of 3
steady-state runs from `examples/generate_qwen25_omni_speech.py`:

| Platform | Audio | Pipeline (steady-state) | RTF |
|----------|-------|-------------------------|-----|
| Trn2 (trn2.48xlarge, TP=4, all-Neuron, 4-core shared) | 9.0s | **8.76s** | **0.97x** |
| H100 80GB (single GPU, SDPA) | 7.18s | 9.55s | 1.33x |

Trn2 finishes ~9% sooner in wall time and clears real-time (RTF < 1.0).
The audio-length gap (9.0s vs 7.18s) is unavoidable: CUDA RNG ≠ CPU RNG
and bf16 matmul numerics diverge across hardware, so the Trn2 talker
draws a different — and in this run slightly longer — codec sequence
even from identical prompts and parameters. The longer codec adds
Token2Wav work, yet Trn2's faster Token2Wav still finishes first.

Reproduce with:

```bash
# H100
python examples/test_gpu_baseline_bench.py --greedy --seed 1234 --num-runs 4

# Trn2 (after `--compile --greedy`)
NEURON_RT_VISIBLE_CORES=0-3 \
    python examples/generate_qwen25_omni_speech.py \
    --greedy --seed 1234 --num-runs 3
```

Streaming / TTFB numbers (where the 8-core split layout matters)
are tracked separately in `examples/test_ttfb_streaming_bench.py`.

### Per-Module Micro-Bench (Trn2 4-core vs H100 80GB, BF16)

To remove pipeline coupling and run-to-run sampling noise, each top-level
module is timed in isolation with fixed input shapes and fixed
``max_new_tokens``. Median of 5 runs:

| Module | Trn2 (4-core, BF16) | H100 (BF16, SDPA) | Neuron / GPU |
|--------|---------------------|--------------------|--------------|
| Thinker (TPOT, 32 tok) | 10.3 ms | 24.1 ms | **2.3x faster** |
| Talker (TPOT, 200 tok) | 4.1 ms | 21.1 ms | **5.1x faster** |
| DiT (per step, mel=1024, batch=2 CFG, fp32) | 62.4 ms | 29.8 ms | 2.1x slower |
| BigVGAN (mel=1024, chunked T=256 on Neuron, single fwd on H100) | 280 ms | 90 ms | 3.1x slower |

Reproduce with:

```bash
# Trn2 (4-core layout). Compile once; the 1024 bucket avoids the 4x O(n^2)
# tax of falling through to the next-larger 2048 bucket.
NEURON_RT_VISIBLE_CORES=0-3 \
    python examples/bench_modules_neuron.py --num-runs 5 \
    --dit-mel-len 1024 --json bench_neuron.json

# H100 (single GPU)
python examples/bench_modules_gpu.py --num-runs 5 --json bench_gpu.json

# Side-by-side markdown table
python examples/compare_bench.py --neuron bench_neuron.json \
    --gpu bench_gpu.json --md bench_compare.md
```

DiT runs in fp32 on both platforms (Token2Wav requires fp32 ODE
precision); H100's fp32 matmul throughput plus exclusive-device
scheduling beats the 4-core shared NeuronCore layout on this fixed-shape
kernel. BigVGAN compiles only up to T=256 on neuronxcc <= 2.25 (see the
"BigVGAN compile cap" subsection below), so the runtime does chunked
overlap-add at T=256 to handle full utterances. The 280ms cell at
mel=1024 is the cost of 5 NEFF calls + 4 crossfades + Python dispatch;
H100 runs the same workload as a single fp32 forward on one device,
which is the bulk of its 3.1x lead. Once the SDK lifts the T=256 cap,
a single mel=1024 NEFF should close most of this gap.
Thinker and Talker are autoregressive and dominated by per-step Python /
sampling overhead in HF ``generate``, where Neuron's fused-embed talker
pulls ahead. Talker's 5.1x per-step margin × the ~450 codec tokens of a
typical reply is what keeps the full-utterance pipeline ahead of the
H100 baseline.

#### BigVGAN compile cap (neuronxcc <= 2.25) and chunked workaround

`compile_bigvgan` traces one NEFF per mel_len bucket. On neuronxcc
2.25.3371 we observe two related compiler bugs:

1. **`T >= 256` crashes with `[NCC_ITIN902] TensorInitialization`** on
   `--auto-cast=none`. Sweeping `--optlevel`, `--target=trn2`,
   `--logical-nc-config=2`, `--disable-internal-io-dge`, and
   `--model-type=transformer` does not help — the failing pass is the
   same. **`--auto-cast=all` bypasses it** (the compiler is allowed to
   cast internal matmuls to bf16, which sidesteps a fragile fp32 layout
   path). Numerics stay tight: at T=256, output cosine vs CPU fp32 =
   0.9999, max_abs = 5.8e-4 (~ −70 dB), well below audible threshold.
   Verified with `examples/probe_bigvgan_buckets.py`.
2. **`T >= 512` still crashes even with `--auto-cast=all`** — the same
   internal pass overflows on a larger tile budget. T=256 is the hard
   cap on this SDK.

`compile_bigvgan` therefore picks `--auto-cast=all` automatically when
`bucket > 128` and `--auto-cast=none` otherwise. To handle full
utterances (mel_len ≫ 256) the runtime BigVGAN forward shim does
**chunked overlap-add**: split the mel into 256-frame chunks with a
32-frame overlap (≈ 7680 wav samples / ~480ms after the 240× upsample),
run each chunk through the T=256 NEFF, and crossfade the wav-domain
overlap with an equal-amplitude `cos²` window (`fade_in + fade_out = 1`
sample-wise). Because BigVGAN is fully convolutional with deterministic
240× upsample, no state has to be carried between chunks. Verified end
to end: an 11.9s utterance synthesized from the speech pipeline (5
chunks at mel_len=1146) is artifact-free at chunk boundaries.

#### NKI flash-attention kernel evaluation (negative result)

The DiT attention currently uses an explicit-matmul implementation
(`_monkeypatch_dit_attention_for_neuron` in
`src/modeling_qwen25_omni_token2wav.py`). We evaluated whether replacing
it with one of NxDI's NKI flash-attention kernels would help. Numbers
below are from `examples/bench_nki_flash_attn.py` on the DiT shape
(B=2 CFG, H=16, head_dim=64, S=1024, single core):

| Kernel | dtype | median | speedup | cosine vs CPU fp32 |
|--------|-------|--------|---------|--------------------|
| explicit_matmul (current) | fp32 | 4.30 ms | 1.00x | 1.0 |
| `attention_cte` | fp32 | 3.65 ms | 1.18x | 1.0 |
| `attention_isa_kernel` | fp32 | 3.55 ms | 1.21x | 1.0 |
| `attention_cte` | bf16 | 2.25 ms | 1.91x | 1.0 |
| `attention_isa_kernel` | bf16 | 2.23 ms | 1.93x | 1.0 |
| `flash_fwd` | — | — | — | unsupported (`seqlen_k % 2048 == 0`, S=1024) |

**We did not integrate any of these.** Reasons:

1. **NKI kernels do not accept arbitrary additive masks.** The DiT uses
   block-diagonal sparse masks (`block_size=24`) for 19 of 22 blocks
   (`look_backward=0, look_ahead=0`), and block-tridiagonal masks for
   blocks 0/10/20. `attention_cte` only exposes
   `causal_mask`/`sliding_window`/`k_prior`/`sink`; `attention_isa_kernel`
   only exposes `sliding_window`. Block-diagonal-with-block_size=24
   cannot be expressed by sliding_window. Substituting any of these
   kernels would silently turn DiT attention into global full attention,
   leaking across chunk boundaries and corrupting the audio output.
   The 1.18-1.21x numbers above were measured against a zero mask, which
   is **not what the real DiT runs**.
2. Even ignoring (1), the attention-only speedup (~20% in fp32) translates
   to ~5-8% on the DiT pipeline (attention is one part of each block ×
   22 blocks × 10 ODE × 2 CFG), which does not justify forking the
   attention path or maintaining a kernel-vs-mask compatibility shim.
3. Token2Wav requires fp32 for ODE accumulation, so the bf16 row of the
   table is informational only.

The bench script remains in the repo (`examples/bench_nki_flash_attn.py`)
as a reference for future SDK upgrades — if a future kernel exposes a
generic `attention_bias` argument, this is the test harness to validate
it on.

## Compatibility Matrix

| Instance/Version | 2.23+ (PyTorch 2.9) | 2.22 and earlier |
|------------------|---------------------|------------------|
| Trn2 (trn2.48xlarge) | Tested (TP=4) | Not tested |
| Trn2 (trn2.3xlarge) | Supported (TP=4) | Not tested |
| Trn1 (trn1.32xlarge) | Should work (TP=4, 4 NeuronCores) | Not tested |
| Inf2 (inf2.48xlarge) | Should work (TP=4) | Not tested |

## Testing

Verified on trn2.48xlarge with real Qwen2.5-Omni-7B weights:

- **Imports**: All model classes import successfully
- **Config**: TP=4 head divisibility verified (Thinker 7/1, Audio 5, Vision 4 per rank)
- **State dict**: All 2448 keys converted correctly (text=339, audio=489, vision=518, talker=293, token2wav=809)
- **Text generation (TP=4)**: Compile + load + generate working, TPOT ~10ms, correct outputs verified
- **Speech pipeline**: Full Thinker → Talker → Token2Wav verified end-to-end on Neuron, RTF 0.97x (see Performance)

```bash
# End-to-end multimodal test (text / image+text / audio+text / text→speech)
python test/integration/test_e2e_qwen25_omni.py

# Or run the example entrypoints directly
python examples/generate_qwen25_omni.py --mode text
python examples/generate_qwen25_omni_speech.py
```

## Key Implementation Notes

1. **TP=4 for all Neuron components**: Thinker (28 heads/4=7 per rank), Vision (16 heads/4=4), Audio (20 heads/4=5). All heads divisible by 4.
2. **Audio encoder hybrid architecture**: Conv1d frontend + chunking on CPU, 32 transformer layers on Neuron with TP=4, AvgPool + LayerNorm + projection on CPU. Asymmetric attention bias (q/v have bias, k has none) handled via ColumnParallelLinear.
3. **Talker on Neuron**: Non-standard head_dim (128 != 896/12), 3D mRoPE with per-step thinker-state injection, ~690M params. Uses ImageToTextModelWrapper with 24 positional args. Fused embedding (embed_tokens 8448→3584 + proj 3584→896 collapsed into 8448→896). Per-step thinker reply states injected via vision_embeddings during token generation. Vision embeddings auto-padded to max_context_length for compiled bucket compatibility. TPOT 4.0ms.
4. **Token2Wav split architecture**: DiT transformer core (22 blocks) on Neuron via torch_neuronx.trace(). CPU preprocessing: ECAPA-TDNN speaker encoder, codec embedding (repeat_interleave), input embedding, rotary embedding, block_diff mask. CPU postprocessing: ODE solver (RK4, 10 steps), BigVGAN vocoder. Float32 for ODE precision. XLA fixes: DiTAttention in-place slice assignment → torch.cat, SDPA dispatch → explicit matmul attention, float additive attention mask. Automatic CPU fallback when mel_len exceeds compiled max.
5. **Speaker support**: `spk_dict.pt` contains per-speaker conditioning (Ethan, Chelsie)
6. **State dict prefix remapping**: `thinker.model.*` -> `model.*`, `thinker.lm_head.*` -> `lm_head.*`, `thinker.visual.*` -> `visual.*`, `thinker.audio_tower.*` -> `frontend.*`/`transformer.*`/`postprocessor.*`

## Example Checkpoints

* [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-15
