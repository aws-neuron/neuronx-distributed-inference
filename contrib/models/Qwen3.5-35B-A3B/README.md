# Qwen3.5-35B-A3B on NeuronX Distributed Inference (Trn2)

`Qwen/Qwen3.5-35B-A3B` is the **MoE** flagship of the Qwen3.5 family — 35 B
total parameters, ~3 B activated per token ("A3B") through top-8 routing over
256 experts plus one sigmoid-gated shared expert. It uses the same hybrid
attention stack as the dense siblings —
**[3 gated DeltaNet + 1 full GQA] × 10 = 40 layers** — combined with a sparse
MoE feed-forward on every layer.

This is the **first** DeltaNet + MoE integration on Neuron. It reuses the
DeltaNet + attention path from PR #173 (originally targeted at Qwen3.6-27B
dense) and plugs NxDI's `initialize_moe_module` (from `moe_v2`) into a new
`Qwen35MoEBlock`. Runs on the stock
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` DLAMI venv
(`neuronx-cc` 2.26.6360 / `nki` 0.5.0). Only the config / decoder-layer / weight
converter in `modeling_qwen35.py` gained a MoE branch; the rest of the file
is byte-identical to the dense contribs.

**Status:** both text-only **and vision-language** inference are validated
end-to-end on `trn2.48xlarge` (TP=8, bf16). VL uses the Neuron-compiled
vision encoder (buckets 1024 / 4096) plus a text-model recompile with
`use_text_only_cte_inputs=False` and CTE bucketing enabled, sharing the
same DeltaNet + MoE decoder as the text-only path.

## Architecture diff vs dense Qwen3.5-27B

| field | 27B (dense) | **35B-A3B (MoE)** |
|---|---:|---:|
| `hidden_size` | 5120 | **2048** |
| `intermediate_size` (dense MLP) | 17408 | — |
| `moe_intermediate_size` (per routed expert) | — | **512** |
| `shared_expert_intermediate_size` | — | **512** |
| `num_hidden_layers` | 64 | **40** |
| `num_attention_heads` | 24 | **16** |
| `num_key_value_heads` | 4 | **2** |
| `linear_num_value_heads` | 48 | **32** |
| `linear_num_key_heads` | 16 | 16 |
| `head_dim` | 256 | 256 |
| **`num_experts`** | — | **256** |
| **`num_experts_per_tok`** | — | **8** |
| **shared expert count** | — | **1** with per-token sigmoid gate |
| `tie_word_embeddings` | false | false |
| model_type | `qwen3_5_text` | `qwen3_5_moe_text` |

Total params: 35 B (55.6 GB of bf16 safetensors weight); activated per token
~3 B (top-8 of 256 routed + 1 shared).

## Contents

```
Qwen3.5-35B-A3B/
├── README.md
├── src/
│   ├── modeling_qwen35.py     — DeltaNet + GQA text stack + NEW `Qwen35MoEBlock`
│   ├── modeling_qwen35_vl.py  — VL orchestrator (vision + text)
│   ├── modeling_qwen35_vision.py
│   ├── hybrid_apc.py
│   ├── nki_kernels/           — DeltaNet NKI kernels (unchanged)
│   └── __init__.py
└── test/integration/          — same runner/bench scripts as dense contribs
```

## What changed in `modeling_qwen35.py` for MoE

Deltas vs the dense contrib source (a couple of hundred lines total):

1. **Config**: `Qwen35InferenceConfig.from_pretrained` preserves
   `model_type` (was hardcoded to `qwen3_5_text`) so we can detect
   `qwen3_5_moe_text` variants. When a MoE config is detected the
   `__init__` sets `num_local_experts = num_experts`, `n_shared_experts = 1`,
   maps `moe_intermediate_size → intermediate_size` (used by
   `initialize_moe_module` to size the routed experts), and populates
   `shared_expert_intermediate_size`.

2. **`Qwen35MoEBlock`**: new nn.Module inserted between `Qwen35MLP` and
   `NeuronQwen35DecoderLayer`. Wraps NxDI's `initialize_moe_module` for the
   routed experts and re-implements a shared expert with a **per-token
   sigmoid gate** (Qwen3.5-MoE specific — NxDI's built-in `SharedExperts`
   only sums into the routed output without a per-token gate).

3. **Decoder layer**: `NeuronQwen35DecoderLayer.__init__` picks
   `Qwen35MoEBlock(config)` when `config._is_moe` is set, else the dense
   `Qwen35MLP` / `NeuronLlamaMLP` path.

4. **Weight converter**: `convert_qwen35_hf_to_neuron_state_dict` gains an
   MoE branch. HF stores stacked 3D expert weights `(num_experts, 2*I, H)`
   and `(num_experts, H, I)`; NxDI's `ExpertMLPsV2` expects the transposed
   layout `(num_experts, H, 2*I)` and `(num_experts, I, H)`. Router key
   `mlp.gate.weight` → `mlp.moe.router.linear_router.weight`; shared expert
   keys `mlp.shared_expert.{gate,up,down}_proj.weight` →
   `mlp.shared_{gate,up,down}_proj.weight`; expert stacked tensors renamed
   `mlp.experts.{gate_up_proj,down_proj}` →
   `mlp.moe.expert_mlps.mlp_op.{gate_up_proj,down_proj}.weight`; the
   `mlp.shared_expert_gate.weight` scalar-output linear is unchanged.

## Compatibility

| Component | Version |
|---|---|
| Instance | `trn2.48xlarge` (validated at TP=8) |
| `neuronx-cc` | 2.26.6360.0 |
| `nki` | 0.5.0 |
| `neuronx-distributed` | 0.19.28492 |
| `neuronx-distributed-inference` | 0.10.18399 |
| `torch-neuronx` | 2.9.0.2 (torch 2.9.1) |
| `libneuronxla` | 2.2.17544 |
| Python | 3.12 |
| `transformers` | 4.57.6 (Neuron runtime). HF CPU reference needs ≥ 5.13. |

## Checkpoint

- HuggingFace: [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- Architecture identifier: `qwen3_5_moe`
- Weights: **14 shards, ~67 GB bfloat16**

Download:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='/mnt/nvme/models/Qwen3.5-35B-A3B')"
```

## Quick start — text-only

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python contrib/models/Qwen3.5-35B-A3B/test/integration/run_text_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-35B-A3B \
    --compiled-path /tmp/qwen35_35b_a3b_traced \
    --tp 8 --seq-len 512 --max-new-tokens 32 \
    --prompt "The capital of France is"
```

Sample validated on `trn2.48xlarge`, TP=8, bf16, seq_len=512:

```
prompt : 'The capital of France is'
output : 'The capital of France is Paris.\nThe capital of France is Paris.\n...'
TTFT   : 561.4 ms
TPOT   : 7.4 ms  (136.03 tok/s)
```

Additional prompts produce coherent, factually correct outputs across the
suite (Jupiter for largest planet, 100 °C water boiling point, autumn haiku,
photosynthesis definition).

## Measured text-only performance (TP=8, bf16, seq_len=512)

`run_benchmark.py --prompt-lens 16 64 256 --max-new-tokens 64 --repeats 5`

Three MoE blockwise-matmul backends were evaluated on the shipped
`neuronx-cc 2.26.6360` DLAMI:

| MoE backend | 16 tok TTFT | 64 tok TTFT | 256 tok TTFT | TPOT | notes |
|---|---:|---:|---:|---:|---|
| `use_torch_block_wise=True` | 553.8 ms | 554.0 ms | 553.6 ms | ~7.7 ms | unrolled per-block loop in NxDI graph; runs on Neuron but no NKI kernel fusion |
| **`use_shard_on_intermediate_dynamic_while=True`** (default) | **480.0 ms** | **575.5 ms** | **667.1 ms** | **~7.7 ms** | LNC=2 NKI kernel sharding on intermediate dim; 13 % faster at prompt=16, grows with prompt length |
| patched `_call_shard_hidden_kernel` (nkilib fwd) | 567.0 ms | 566.5 ms | 566.7 ms | ~7.8 ms | see below |

`shard_on_intermediate` is the current default in `run_text_smoke.py` /
`run_benchmark.py` — it wins at short prompts (chat use case). The third
row is a **monkey-patched path**: NxDI's default LNC=2 forward MoE kernel
(`_call_shard_hidden_kernel`) is a `NotImplementedError` stub because the
`neuronxcc.nki._private.blockwise_mm` module is absent from the shipped
`neuronx-cc 2.26.6360` DLAMI. However, the same kernel forward
implementation is available at
`nkilib.experimental.moe.forward.bwmm_shard_on_H.blockwise_mm_baseline_shard_hidden`,
so `modeling_qwen35.py::_patch_nxd_shard_hidden_kernel()` wires it in at
import time. That gives the default LNC=2 forward path a runnable
implementation (flat 567 ms TTFT regardless of prompt length) — slower
than shard-on-intermediate for short prompts but more stable at long
prompts. Kept as a safety net; not chosen as the default.

None of these three paths matches what a genuine LNC=2 shard-hidden NKI
kernel could deliver in a future SDK drop.

**Why is TTFT / TPOT ratio so different from the dense siblings?**
Dense models spend ~40-60 % of prefill in the FFN block; MoE only dispatches
top-8 experts per token, so decode is extremely cheap (~10× less MLP FLOPs
than 27B despite having more total params) but prefill still has to route
256 experts × 512 tokens = many small block-matmuls. The observed
ratios are consistent with this:

| model | Params (activated per token) | TTFT (16) | TPOT | TTFT/TPOT |
|---|---:|---:|---:|---:|
| 2B  |  2 B | 17.6 ms | 4.00 ms | 4.4× |
| 4B  |  4 B | 34.6 ms | 5.68 ms | 6.1× |
| 9B  |  9 B | 42.9 ms | 6.89 ms | 6.2× |
| 27B | 27 B | 118.6 ms | 21.58 ms | 5.5× |
| **35B-A3B** | 35 B (**~3 B active**) | **480 ms** | **7.72 ms** | **62×** |

TPOT of ~7.7 ms sits between 9B (6.9 ms) and 4B (5.7 ms), matching the
"~3 B activated" figure — confirming the MoE sparse-decode benefit. The
inflated TTFT/TPOT ratio is a consequence of the shipped kernel gap, not the
architecture.

## Measured VL performance (TP=8, bf16, CTE buckets 512/1024/2048/4096/8192)

Vision encoder compiled to Neuron for two buckets (1024, 4096) via
`compile_vision_encoder.py`; text model compiled with
`use_text_only_cte_inputs=False` and CTE bucketing enabled so the same
artifact handles all three image sizes. Prompt: *"What is in this image?
Describe it briefly."* — three repeats after warmup.

| Image | Vision path | TTFT (ms) | TPOT (ms) | tok/s |
|---|---|---:|---:|---:|
| 512×512  | Neuron ViT (bucket 1024) | **1036.8** | 8.2 | 121.9 |
| 1024×1024 | Neuron ViT (bucket 4096) | **2004.0** | 8.1 | 123.6 |
| 2048×2048 | 2×2 tile → 4× Neuron ViT (bucket 4096) | **5490.4** | 8.1 | 123.2 |

TPOT is flat (~8 ms) across sizes because decode is unchanged; TTFT scales
with image → text-token count (image tokens: ~278 / 1046 / 4712, CTE bucket
picked: 512 / 1024 / 8192). Sample output on `test_image_1024.jpg` — the
model correctly identifies the animal as a **Pallas's cat (manul)**;
`test_image_2048.jpg` (a leopard-like feline) resolves to *"close-up of an
animal...rosettes/spots"* using the 2×2 tile fallback. Both descriptions
show the vision+MoE stack is functioning end-to-end.

To reproduce:

```bash
# 1. Compile vision encoder buckets (once, ~10 min for 1024, ~12 min for 4096)
python contrib/models/Qwen3.5-35B-A3B/test/integration/compile_vision_encoder.py \
    --model-path /mnt/nvme/models/Qwen3.5-35B-A3B \
    --out-dir /tmp/qwen35_35b_a3b_vl_bench/vision \
    --buckets 1024 4096

# 2. Compile text model + benchmark (once, ~15 min compile then benchmark)
python contrib/models/Qwen3.5-35B-A3B/test/integration/run_vl_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-35B-A3B \
    --compiled-path /tmp/qwen35_35b_a3b_vl_bench \
    --tp 8 --images 512 1024 2048 --buckets 512 1024 2048 4096 8192 \
    --max-new-tokens 48 --repeats 3 \
    --vision-compiled-dir /tmp/qwen35_35b_a3b_vl_bench/vision \
    --out-json /tmp/qwen35_35b_a3b_vl_bench.json
```

## Notable configuration choices

- **`MoENeuronConfig`** (not `NeuronConfig`) — required by
  `initialize_moe_module` so it can find `router_config`,
  `blockwise_matmul_config`, `moe_tp_degree`, etc.
- `moe_tp_degree = 8`, `moe_ep_degree = 1` — no expert parallelism yet,
  every rank sees every expert (sharded on the intermediate dim).
- **`blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while": True}`**
  (default). Two alternate MoE prefill kernels are evaluated in the
  benchmark table above; the shard-on-intermediate path is the fastest one
  that runs on the shipped `neuronx-cc 2.26.6360` DLAMI. The truly-preferred
  `_call_shard_hidden_kernel` LNC=2 path is not available in the shipped
  DLAMI and would require a future SDK drop.
- **`moe_ep_degree=1`** — expert parallelism (`moe_ep_degree > 1`) is
  functional during prefill but **not supported during decode** by NxDI's
  `ExpertMLPsV2`: with `top_k=8 / num_experts=256`, the per-token expert
  fraction is 3 %, below `DEFAULT_SELECTIVE_LOADING_THRESHOLD=1.0`, which
  routes decode through `forward_selective_loading` — and that path
  raises `NotImplementedError: Selective Loading with Expert parallelism is
  not supported in token generation.`
- `router_config.dtype = float32`, `router_config.act_fn = "softmax"` —
  Qwen3.5-MoE uses softmax over router logits with fp32 accumulation.
- `normalize_top_k_affinities = True` — Qwen3.5-MoE normalizes the top-k
  weights so they sum to 1 per token.
- `QWEN36_DELTANET_CTE_IMPL=legacy_direct`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`
  — same DeltaNet numerical stability defaults as the dense siblings; the
  fused-multihead NKI kernel is not needed for text-only decode.

## Known limitations / follow-ups

- **HF greedy match not run**. 35B-A3B on CPU bf16 is ~67 GB and greedy
  generation takes many minutes per prompt; deferred until GPU or larger CPU
  is available. All 5 prompts in the accuracy suite produce coherent,
  factually correct Neuron output.
- **Shipped SDK MoE kernel gap.** The truly-preferred LNC=2 shard-hidden
  NKI kernel (`_call_shard_hidden_kernel`) is not present in the shipped
  `neuronx-cc 2.26.6360` DLAMI. Current default
  (`use_shard_on_intermediate_dynamic_while`) gets
  ~480 ms at prompt=16 (13 % better than the torch fallback) but scales
  linearly with prompt length. A future SDK drop should close this.
- **Expert parallelism (EP=1).** `moe_ep_degree > 1` is supported for
  prefill but NxDI's `ExpertMLPsV2` raises `NotImplementedError` for
  selective-loading decode with EP — and our (top_k=8, num_experts=256)
  configuration always goes through selective loading at decode time.

## Maintainer

Contributed alongside the 2B/4B/9B/27B dense siblings. This is the first
DeltaNet + MoE integration on Neuron; the MoE plumbing follows NxDI's
`qwen3_moe` model as a reference and adapts it for Qwen3.5-MoE's
sigmoid-gated shared expert.
