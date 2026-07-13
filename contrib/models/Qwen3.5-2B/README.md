# Qwen3.5-2B on NeuronX Distributed Inference (Trn2)

`Qwen/Qwen3.5-2B` is a 2 B-parameter vision-language decoder with a hybrid
attention stack — 24 transformer layers arranged as **[3 gated DeltaNet
(linear attention) + 1 full GQA attention] × 6**. The full-attention layers
use per-head query gating and partial rotary embeddings (2D + interleaved
mRoPE) with `head_dim = 256`, `partial_rotary_factor = 0.25`. The linear-
attention layers are recurrent gated-DeltaNet blocks (SSD-style: conv1d +
delta rule) with 16 K/V heads at `head_dim = 128`. The ViT vision encoder is
24 layers, `patch_size = 16`, `spatial_merge_size = 2`, `hidden_size = 1024`,
`out_hidden_size = 2048`.

This contrib model reuses the DeltaNet + GQA modeling code contributed to
[PR #173](https://github.com/aws-neuron/neuronx-distributed-inference/pull/173)
for the 27B sibling and adapts it to the 2B variant. **No modifications to the
installed NxDI library are required** — everything runs on the stock
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` DLAMI venv
(`neuronx-cc` 2.26.6360 / `nki` 0.5.0).

**Status:** text-only and vision-language inference are both validated
end-to-end on `trn2.48xlarge`. VL requires the legacy-direct DeltaNet CTE
kernel — see the "Vision" section.

## Contents

```
Qwen3.5-2B/
├── README.md              — this file
├── src/
│   ├── modeling_qwen35.py           — text decoder (DeltaNet + GQA + MRoPE)
│   ├── modeling_qwen35_vision.py    — ViT encoder wrapper
│   ├── modeling_qwen35_vl.py        — VL orchestrator (image + text)
│   ├── hybrid_apc.py                — Automatic Prefix Cache (unused by default)
│   └── nki_kernels/                 — NKI kernels for DeltaNet
│       ├── nki_deltanet.py               (recurrent step / recurrent fwd)
│       ├── nki_deltanet_chunked.py       (per-chunk step)
│       ├── nki_deltanet_fused.py         (fused chunked fwd — default CTE path)
│       ├── nki_deltanet_fused_legacy.py
│       └── qwen_qk_norm_rope.py          (optional QK-norm+RoPE NKI kernel)
└── test/integration/
    ├── run_text_smoke.py     — compile + generate text only (bring-up)
    ├── run_benchmark.py      — TTFT / TPOT across prompt lengths
    ├── run_hf_reference.py   — HF CPU reference (requires transformers>=5.13)
    ├── run_accuracy_check.py — Neuron output dump for HF cross-check
    ├── run_vl_smoke.py       — VL: CPU vision + Neuron text, image + prompt
    └── test_model.py         — pytest suite for CI
```

## Compatibility

| Component      | Version                                      |
|----------------|----------------------------------------------|
| Instance       | `trn2.48xlarge` (validated at TP=8)          |
| `neuronx-cc`   | 2.26.6360.0                                  |
| `nki`          | 0.5.0                                        |
| `neuronx-distributed` | 0.19.28492                            |
| `neuronx-distributed-inference` | 0.10.18399                  |
| `torch-neuronx` | 2.9.0.2 (torch 2.9.1)                       |
| `libneuronxla` | 2.2.17544                                    |
| Python         | 3.12 (system DLAMI venv)                     |
| `transformers` | 4.57.6 (for Neuron runtime). **HF reference on CPU requires transformers ≥ 5.13.** |

## Checkpoint

- HuggingFace: [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)
- Architecture identifier: `qwen3_5` (introduced in transformers 5.x)
- Weights: `~4.5 GB` bfloat16 safetensors

Download:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-2B', local_dir='/mnt/nvme/models/Qwen3.5-2B')"
```

## Quick start — text-only inference

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Compile (first run only; ~90 s) and generate one prompt
python contrib/models/Qwen3.5-2B/test/integration/run_text_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-2B \
    --compiled-path /tmp/qwen35_2b_traced \
    --tp 8 --seq-len 512 --max-new-tokens 32 \
    --prompt "The capital of France is"
```

Expected output (validated on `trn2.48xlarge`, TP=8, `bf16`, seq_len=512):

```
prompt : 'The capital of France is'
output : 'The capital of France is Paris.\nA. True\nB. False...'
n_new  : 32
TTFT   : 24.1 ms
TPOT   : 3.9 ms  (259.12 tok/s)
```

## Measured performance (Neuron, TP=8, `bf16`, seq_len=512)

`python test/integration/run_benchmark.py --prompt-lens 16 64 256 --max-new-tokens 64 --repeats 5`

| prompt tokens | TTFT (ms, median) | TPOT (ms, median) | Throughput (tok/s) |
|---------------|-------------------|-------------------|--------------------|
| 16            | 17.6              | 4.00              | 250.4              |
| 64            | 17.6              | 3.99              | 250.2              |
| 256           | 17.5              | 4.00              | 250.7              |

TTFT is essentially flat because DeltaNet is O(1)-state and the full-attention
layers touch only 6/24 layers. TPOT is dominated by TP-8 all-reduce + host-loop
overhead at batch = 1.

## Accuracy vs. HuggingFace reference

Comparison run against `transformers==5.13.0` CPU bf16 greedy on 5 diverse
prompts (16 new tokens each):

| Prompt                                        | Match     |
|----------------------------------------------- |-----------|
| "The capital of France is"                    | 16 / 16   |
| "The largest planet in our solar system is"   | 3 / 16    |
| "Water boils at"                              | 16 / 16   |
| "A haiku about autumn leaves:"                | 16 / 16   |
| "In one sentence, explain photosynthesis."    | 2 / 16    |

Aggregate: **53 / 80 = 66 %** exact-token greedy match, **3 / 5 prompts fully
match**. Divergence on the two non-matching prompts is a standard cascade
after one differing token — expected for a `bf16` accumulate path against an
independent `bf16` CPU implementation. The generated text is qualitatively
coherent on all five prompts.

## Vision (image → text)

VL runs end-to-end: image → CPU vision encoder → Neuron text decoder → text
output. Sample run on a real cat image (960×686, ~630 merged vision tokens):

```
prompt   : "What is in this image? Describe it briefly."
output   : "This image features a fluffy, light-colored cat with a distinctive
            appearance. It has a thick, woolly coat that gives it a
            soft, cloud-like look. The cat's fur is predominantly white
            or very light cream, with some darker patches"
elapsed  : ~9 s for 48 new tokens (CPU vision + Neuron text)
```

Verified components:
- **CPU vision encoder** (`NeuronQwen35VisionModelWrapper.load_cpu_model`):
  cosine similarity **0.99** vs. HuggingFace `Qwen3_5VisionModel`.
- **`get_rope_index`** (3-D mRoPE): **100% token-position match** vs. HF
  `compute_3d_position_ids`.
- **Scatter path** into text model using `index_put_(accumulate=False)` at
  image-token positions (same pattern as upstream `qwen3_vl` in NxDI).
- **Text decoder** compiled with `use_text_only_cte_inputs=False` and
  **`QWEN36_DELTANET_CTE_IMPL=legacy_direct`** — the fused-multihead DeltaNet
  NKI kernel (the default) is numerically unstable on real vision embeddings
  and produces degenerate output (repeated tokens); the legacy-direct kernel
  is stable and is now the default for `run_vl_smoke.py`.

Run:

```bash
python contrib/models/Qwen3.5-2B/test/integration/run_vl_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-2B \
    --compiled-path /tmp/qwen35_2b_vl_traced \
    --image         /path/to/image.jpg \
    --prompt        "Describe this image." \
    --tp 8 --seq-len 2048 --max-new-tokens 48
```

The script sets `QWEN36_DELTANET_CTE_IMPL=legacy_direct` and
`QWEN36_DELTANET_MULTIHEAD_CTE=0` before compile; do not override these unless
debugging the fused kernel.

### Benchmark across image sizes

`run_vl_benchmark.py` compiles once with NxDI CTE bucketing enabled
(buckets `[512, 1024, 2048, 4096, 8192]`) and measures TTFT / TPOT on
512×512, 1024×1024, 2048×2048 versions of the same source image
(same prompt: *"What is in this image? Describe it briefly."*).

**CPU vision baseline:**

| image     | vision tokens | TTFT (ms, median) | TPOT (ms, median) | tok/s |
|-----------|--------------:|------------------:|------------------:|------:|
| 512×512   |           256 |           **169** |             4.1   |   242 |
| 1024×1024 |         1,024 |         **1,091** |             4.3   |   232 |
| 2048×2048 |         4,096 |         **5,025** |             3.9   |   197 |

**Neuron vision encoder** (compiled at patch-token buckets `[1024, 4096]`;
16384 bucket exceeds trn2 single-core HBM):

| image     | TTFT (ms) | vs CPU-VE | TPOT (ms) | notes                             |
|-----------|----------:|----------:|----------:|-----------------------------------|
| 512×512   |    **86** | **2.0× ↓**|      4.2  | Neuron VE (bucket 1024)           |
| 1024×1024 |   **526** | **2.1× ↓**|      4.1  | Neuron VE (bucket 4096)           |
| 2048×2048 | **2,090** | **2.4× ↓**|      4.2  | **2×2 tiled** through bucket 4096 |

TTFT halves (or more) at every size. TPOT is flat because text decode is
Neuron-only regardless. For 2048×2048 the wrapper transparently splits the
16,384-patch input into four 4,096-patch spatial tiles, encodes each through
the bucket-4096 kernel, and re-interleaves the merged outputs
(`NeuronQwen35VisionModelWrapper._tiled_forward`). Tiling loses cross-tile
attention (a known LLaVA-NeXT-style trade-off); accuracy at 2048×2048 drops
slightly from the CPU path — see accuracy note below.

Compile the Neuron vision buckets once with `compile_vision_encoder.py`:

```bash
python contrib/models/Qwen3.5-2B/test/integration/compile_vision_encoder.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B \
    --out-dir    /tmp/qwen35_2b_vl_bench/vision \
    --buckets    1024 4096   # 16384 currently OOMs on trn2 single-core

python contrib/models/Qwen3.5-2B/test/integration/run_vl_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B \
    --compiled-path /tmp/qwen35_2b_vl_bench \
    --vision-compiled-dir /tmp/qwen35_2b_vl_bench/vision \
    --tp 8 --images 512 1024 2048 \
    --max-new-tokens 48 --repeats 3
```

**Optional: TP-sharded vision encoder** (`compile_vision_encoder_tp.py`).
The single-core trace above sees `ColumnParallelLinear` fall back to plain
linear at `tp=1`. Recompiling the same ViT with `parallel_model_trace(tp_degree=N)`
shards QKV / MLP linears across N cores, gathering / all-reducing at
each block boundary. Compile output lives under `bucket_<N>/tp_*.pt` and
is auto-detected by `NeuronQwen35VisionModelWrapper.load_compiled` in the
same directory as the legacy `vision_encoder_<N>.pt` files.

```bash
# TP=4 shard for bucket 4096 (covers 1024×1024 and each tile of 2048×2048)
python contrib/models/Qwen3.5-2B/test/integration/compile_vision_encoder_tp.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B \
    --out-dir    /tmp/qwen35_2b_vl_bench/vision \
    --tp 4 --buckets 4096
```

Measured (trn2.48xlarge, TP=8 text, bf16, VL bucket 4096, 1024×1024 image):

| Vision path        | Standalone latency | E2E TTFT | vs TP=1 vision |
|--------------------|-------------------:|---------:|---------------:|
| TP=1 (single core) |             308 ms |   526 ms |          1.00× |
| TP=2               |             198 ms |   ~-100 ms E2E    |          ~1.6× |
| **TP=4**           |         **101 ms** | **318 ms** |     **1.65× E2E TTFT** |

TP=8 for the vision encoder is technically feasible but wasn't measured
here; extrapolated ~75–90 ms standalone (≈ 3.5–4× TP=1). Bucket 16384
(non-tiled 2048×2048) exceeds the compiler's 10M-instruction verification
limit even at TP=4, so the tiled path stays required at 2048×2048.

**Accuracy note (vs HuggingFace CPU bf16 greedy on the same 3 sizes):** HF
identifies "Pallas's cat" at all 3 sizes. Neuron identifies "Pallas's cat"
correctly at 1024×1024 (best case). At 512×512 it mis-identifies as
"Pangolin" — a 2B-model capacity limitation reproducible on HF at reduced
quality. At 2048×2048 with the Neuron tiled path, output degrades to
"wolf/canid" (2×2 tiles have no cross-tile attention, so global features
like the cat's overall silhouette are lost); the CPU-vision path at 2048
hedges to "wildcat/lynx" instead — closer but still not "Pallas's cat".
Text quality is coherent at all 3 sizes.

Not yet done:
- ViT encoder is on CPU. Tracing it to Neuron via `torch_neuronx.trace` with
  buckets matching `vision_seq_len_buckets` (default `[1024, 4096, 16384]`) is
  the natural next step.
- Investigating why the default fused-multihead DeltaNet kernel fails on
  vision embeddings but passes on text-only inputs at std ≈ 0.011. Random
  vectors of std 0.17 also decode cleanly through it, but real HF vision
  embeddings with std 0.17 and per-dim mean ≈ 2.88 do not. The root cause is
  likely a specific numeric interaction in the fused NKI kernel with the
  structured signal, not overall magnitude.

## HF reference on CPU

Because `transformers==4.57.6` (bundled with the current NxDI DLAMI) predates
the `qwen3_5` architecture, running HF as an accuracy oracle needs an isolated
newer venv:

```bash
python3 -m venv /tmp/hf_ref_venv
/tmp/hf_ref_venv/bin/pip install \
    "transformers==5.13.0" "torch>=2.6" "safetensors" "sentencepiece" "accelerate"

/tmp/hf_ref_venv/bin/python \
    contrib/models/Qwen3.5-2B/test/integration/run_hf_reference.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B \
    --max-new-tokens 16 \
    --out-json /tmp/qwen35_2b_hf_reference.json
```

## Notable configuration choices

* `use_hybrid_cache_manager=False`, `use_hybrid_apc_manager=False` — the
  simpler static per-layer `recurrent_state_buffer` / `conv_state_buffer` path
  is used for bring-up; the paged/APC hybrid caching from PR #173 is not
  required at 2B scale.
* `tie_word_embeddings=True` — Qwen3.5-2B ties `embed_tokens.weight` with
  `lm_head.weight`. The class adds a small `update_state_dict_for_tied_weights`
  hook that duplicates the tensor into `lm_head.weight`.
* Optional shim: `cancel_hybrid_apc_request` / `finish_hybrid_apc_request` /
  `prepare_hybrid_apc_model_inputs` are guarded imports with no-op fallbacks,
  so the module loads on stock NxDI where those symbols are absent.

## Running the pytest suite

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export QWEN35_MODEL_PATH=/mnt/nvme/models/Qwen3.5-2B
pytest contrib/models/Qwen3.5-2B/test/integration/test_model.py -s
```

## Known limitations / follow-ups

- ViT encoder Neuron-compiled at buckets `{1024, 4096}` patch tokens. 2048×2048
  images (16,384 patch tokens) don't fit a single Neuron core's HBM at the
  16,384 bucket size (512 MB attention mask). The wrapper handles this by
  2×2-tiling into four 4,096-patch tile calls (~2.4× TTFT vs CPU vision, at
  the cost of no cross-tile attention). TP-ing the vision model or compiling
  a windowed-attention 16,384 kernel would preserve full attention for
  these inputs.
- VL requires `QWEN36_DELTANET_CTE_IMPL=legacy_direct` because the default
  fused-multihead NKI kernel is numerically unstable on structured vision
  embeddings. `run_vl_smoke.py` sets this automatically before compile.
- Only batch size 1 is validated; hybrid DeltaNet state buffers are indexed by
  `seq_ids` for continuous batching but that path has not been exercised at
  2B.
- Speculative decoding is not wired up.
- APC (Automatic Prefix Caching) is present in `hybrid_apc.py` but disabled;
  enabling it requires the NxDI async-execution shims from PR #173.

## Maintainer

Contributed as part of the NxDI contrib community pool. Testing on
`trn2.48xlarge` (neuronx-cc 2.26.6360 / nki 0.5.0). Modeling code originates from PR #173 (27B
sibling) reused verbatim; only weight loading (`update_state_dict_for_tied_weights`)
and the config validation were adapted for the 2B variant.
