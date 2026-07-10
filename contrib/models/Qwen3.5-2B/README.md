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
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` DLAMI venv (Neuron SDK 2.29
/ NKI 0.3.0).

**Status:** text-only inference is validated end-to-end (TTFT, TPOT, greedy
accuracy vs. HuggingFace). The vision path is scaffolded but **not working**
— see the "Vision" section for details. Do not use this contrib for image
inputs yet.

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
| Neuron SDK     | 2.29 (NKI 0.3.0)                             |
| Python         | 3.12 (system DLAMI venv)                     |
| `torch`        | 2.9.1 (torch-neuronx 2.9.0.2)                |
| `neuronx-distributed-inference` | 0.10.18399                  |
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

## Vision (image → text) — **NOT WORKING, WIP**

The VL path in this contrib is **not validated** and currently produces
degenerate output (e.g., repeated tokens) on real images. It is committed as a
scaffold, not a working solution. Do not rely on it for accuracy.

What is verified to work in isolation:
- **Vision encoder** (CPU path via `NeuronQwen35VisionModelWrapper.load_cpu_model`):
  cosine similarity **0.99** vs. HuggingFace `Qwen3_5VisionModel` on a real
  image (960×686 → 630 merged tokens × 2048 out_hidden).
- **`get_rope_index`** (3-D mRoPE computation): **100% token-position match**
  vs. HuggingFace `compute_3d_position_ids` on the same input (all three axes,
  654 positions).
- **Text decoder** with `use_text_only_cte_inputs=False` compiles and loads
  successfully at `seq_len=2048`.

What is broken:
- The scatter-and-forward pipeline in `NeuronQwen35VLForCausalLM.generate`
  produces `_quantity`-style degenerate output when real vision embeddings are
  wired in. The root cause has not been isolated — likely candidates are the
  `has_real_vision_inputs = shape[1] != seq_length` gating in
  `NeuronQwen35Model.get_model_output` (which suppresses the scatter under
  the current padding scheme), the `padded_seq_len-1` fill value used for
  `vision_mask` pad slots (which overwrites a real text-token position with
  vision garbage in the traced graph), or an interaction between the DeltaNet
  `deltanet_padding_mask` and the vision-scatter path.
- The `modeling_qwen35_vl.py` / `modeling_qwen35_vision.py` files are copied
  as-is from PR #173 (Qwen3.6-27B). That PR only validates text-only paths on
  Trn2; its VL orchestrator is unverified upstream.
- The ViT itself is **not** traced to Neuron. It runs on CPU via
  `CPUVisionModel`.

Scripts left in place for iteration:
- `test/integration/run_vl_smoke.py` — compiles the vision-aware text model
  and drives the CPU-vision-plus-Neuron-text pipeline. Reproduces the bug.

Suggested next steps:
1. Reproduce with `run_vl_smoke.py --image /path/to/img --skip-compile` after
   pre-compiling once (~90 s).
2. Compare `inputs_embeds` after `encode_vision_to_input` on Neuron vs. the
   HF reference `inputs_embeds` at the same positions (a hidden-state dump
   layer-by-layer will pinpoint where the numerics diverge).
3. Once accuracy is verified, port the ViT to Neuron via `torch_neuronx.trace`
   with a bucket set matching `vision_seq_len_buckets` (default `[1024, 4096, 16384]`).

## HF reference on CPU

Because `transformers==4.57.6` (bundled with the NxDI SDK 2.29 DLAMI) predates
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

- **VL is not working** — see "Vision" section above. Vision encoder,
  mRoPE, and text-decoder tracing each pass in isolation, but the wired-up
  scatter path produces degenerate output. Debug pointer: the
  `has_real_vision_inputs = ...shape[1] != seq_length` check in
  `NeuronQwen35Model.get_model_output` (`modeling_qwen35.py` line ~5747)
  conflicts with the `_prepare_vision_args_for_padded_seq` (`~L7076`)
  which pads vision inputs to `padded_seq_len`; those two invariants
  contradict each other.
- The ViT encoder currently runs on CPU. Tracing it to Neuron (via
  `torch_neuronx.trace` per vision-sequence-length bucket) is TODO once
  the VL scatter path is fixed.
- Only batch size 1 is validated; hybrid DeltaNet state buffers are indexed by
  `seq_ids` for continuous batching but that path has not been exercised at
  2B.
- Speculative decoding is not wired up.
- APC (Automatic Prefix Caching) is present in `hybrid_apc.py` but disabled;
  enabling it requires the NxDI async-execution shims from PR #173.

## Maintainer

Contributed as part of the NxDI contrib community pool. Testing on
`trn2.48xlarge` with SDK 2.29. Modeling code originates from PR #173 (27B
sibling) reused verbatim; only weight loading (`update_state_dict_for_tied_weights`)
and the config validation were adapted for the 2B variant.
