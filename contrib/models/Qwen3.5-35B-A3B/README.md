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
(Neuron SDK 2.29 / NKI 0.3.0). Only the config / decoder-layer / weight
converter in `modeling_qwen35.py` gained a MoE branch; the rest of the file
is byte-identical to the dense contribs.

**Status:** text-only inference is validated end-to-end on `trn2.48xlarge`
(TP=8, bf16, seq_len=512). Vision-language is not attempted in this contrib
yet — see "Known limitations". VL requires the vision encoder to be
compiled separately (same as the 2B/4B/9B/27B recipe) and a text model
recompile with `use_text_only_cte_inputs=False`.

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
│   ├── modeling_qwen35_vl.py  — (unused for text-only)
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
| Neuron SDK | 2.29 (NKI 0.3.0) |
| Python | 3.12 |
| `torch` | 2.9.1 (torch-neuronx 2.9.0.2) |
| `neuronx-distributed-inference` | 0.10.18399 |
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

| prompt tokens | TTFT (ms, median) | TPOT (ms, median) | Throughput (tok/s) |
|---:|---:|---:|---:|
| 16  | **553.8** | 7.67 | 129.9 |
| 64  | 554.0 | 7.79 | 128.2 |
| 256 | 553.6 | 7.74 | 129.2 |

TTFT is dominated by the MoE prefill (256 experts × top-8 routing per token,
running through PyTorch's fallback blockwise-matmul path because the stock
SDK 2.29 DLAMI does not ship the LNC=2 shard-hidden NKI kernel needed by
NxDI's default `ExpertMLPsV2`). TPOT of ~7.7 ms is comparable to dense 9B
(6.89 ms) — MoE decode benefits from only 8 experts active per token.

## Notable configuration choices

- **`MoENeuronConfig`** (not `NeuronConfig`) — required by
  `initialize_moe_module` so it can find `router_config`,
  `blockwise_matmul_config`, `moe_tp_degree`, etc.
- `moe_tp_degree = 8`, `moe_ep_degree = 1` — no expert parallelism yet,
  every rank sees every expert (sharded on the intermediate dim).
- **`blockwise_matmul_config={"use_torch_block_wise": True}`** — required
  because the DLAMI-shipped NKI kernel path
  (`_call_shard_hidden_kernel` for LNC=2) is not available. Torch fallback
  is functionally correct but slower — a genuine NKI blockwise-matmul kernel
  would drop TTFT substantially.
- `router_config.dtype = float32`, `router_config.act_fn = "softmax"` —
  Qwen3.5-MoE uses softmax over router logits with fp32 accumulation.
- `normalize_top_k_affinities = True` — Qwen3.5-MoE normalizes the top-k
  weights so they sum to 1 per token.
- `QWEN36_DELTANET_CTE_IMPL=legacy_direct`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`
  — same DeltaNet numerical stability defaults as the dense siblings; the
  fused-multihead NKI kernel is not needed for text-only decode.

## Known limitations / follow-ups

- **VL not attempted.** Vision encoder compile + text recompile with
  `use_text_only_cte_inputs=False` + the tiled path all work in the dense
  siblings, so extension should be mechanical, but combined MoE + vision
  scatter has not been exercised.
- **HF greedy match not run**. 35B-A3B on CPU bf16 is ~67 GB and greedy
  generation takes many minutes per prompt; deferred until GPU or larger CPU
  is available. All 5 prompts in the accuracy suite produce coherent,
  factually correct Neuron output.
- **Torch fallback for blockwise MoE.** ~550 ms TTFT is dominated by the
  Python-level blockwise matmul. A native NKI shard-hidden kernel from a
  future SDK drop would substantially speed up prefill.
- **Expert parallelism (EP=1).** With EP > 1 the model would shard experts
  across cores instead of intermediate dim, likely giving better peak
  utilization at large batch sizes.

## Maintainer

Contributed alongside the 2B/4B/9B/27B dense siblings. This is the first
DeltaNet + MoE integration on Neuron; the MoE plumbing follows NxDI's
`qwen3_moe` model as a reference and adapts it for Qwen3.5-MoE's
sigmoid-gated shared expert.
