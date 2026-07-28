# Contrib Model: MiniMax-M3 (text backbone)

NeuronX Distributed Inference port of the [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) **text backbone**.

The release is a vision-language MoE with ~428B total / ~23B active parameters. This contrib port targets the **text-only causal LM** portion of the model — the vision tower, the multimodal projector, and the Multi-Token Prediction (MTP) modules are not included.

## Status (2026-07-03)

**End-to-end usable at ≤ 2K context on Trn2.48xlarge.** Prefill + KV-cache decode both produce coherent, correct output.

Sanity checks (short prompts, batch=32, seq_len=512, greedy):

| Prompt | Neuron top-1 | Neuron continuation |
|---|---|---|
| `"1+1="` | `2` ✅ | `2\) and \(2+1=3\). So the answer is 3.` |
| `"The capital of France is"` | ` Paris` ✅ | ` Paris.\nThe capital of France is Paris...` |
| `"Berlin is the capital of"` | ` Germany` ✅ | ` Germany. Madrid is the capital of Spain. Rome...` |
| `"The largest planet in our solar system is"` | ` Jupiter` ✅ | ` Jupiter. It is the fifth planet from the Sun and is a gas giant.` |

LongBench (real long-context QA):

| Seq | Batch | TTFT | ITL | Answer 1 | Answer 2 |
|---|---|---|---|---|---|
| 2K | 8 | 7.3 s | 54 ms/tok | ✅ `South West Ultras` | ✅ `15–3` |
| 4K | 4 | 17–19 s | 43–49 ms/tok | ❌ garbage (edge case) | ✅ `pooling approach for face recognition …` |
| 8K, 16K | — | — | — | ❌ per-rank 24 GB HBM insufficient — needs KV-cache DP/CP |

## Architecture Details (text backbone)

| Field | Value |
|---|---|
| Hidden size | 6144 |
| Layers | 60 |
| Attention heads (Q / KV) | 64 / 4 (GQA) |
| Head dim | 128 |
| Rotary dim | 64 (partial RoPE, first half of each head) |
| RoPE theta | 5,000,000 |
| Max position embeddings | 1,048,576 |
| Vocab size | 200,064 |
| Routed experts | 128, top-4 (sigmoid + correction bias) |
| Shared experts | 1 (intermediate = 3072) |
| Dense MLP intermediate | 12,288 (used by first 3 layers) |
| MoE expert intermediate | 3,072 |
| Routed scaling factor | 2.0 |
| Activation | SwiGLU-OAI (`gate * sigmoid(alpha*gate) * (up + 1.0)`, `alpha=1.702`, clamp ± 7.0) |
| Norm | Gemma-style RMSNorm (scale = `1 + weight`), `eps = 1e-6` |
| MSA | Lightning Indexer, `block_size = 128`, `topk_blocks = 16`, `local_blocks = 1` |

## What this port supports

- ✅ **GQA attention** with per-head Gemma RMSNorm on Q/K and partial RoPE.
- ✅ **SwiGLU-OAI everywhere**, including routed MoE experts — via
  `RoutedExpertsMLPOpsConfig(bias=True, hidden_act_bias=1.0, ...)` so
  NxDI's block-sparse NKI kernel receives an explicit `up_bias = 1.0`.
  Dense/shared layers apply the `(up + 1.0)` term directly in
  `MiniMaxM3DenseMLP`.
- ✅ **Sigmoid routing with `e_score_correction_bias`** as an
  `nn.Parameter` in fp32 (bias values dominate the 0..1 sigmoid range,
  so bf16 precision loss on the add would shift top-K choices).
- ✅ **Shared expert** as a sibling `MiniMaxM3DenseMLP` added after
  scaling the routed branch by `routed_scaling_factor = 2.0`.
- ✅ **Gemma `(1 + weight)` pre-shift** baked into every RMSNorm weight
  (`input_layernorm`, `post_attention_layernorm`, `q_layernorm`,
  `k_layernorm`, `indexer.q_norm`, `indexer.k_norm`, final `norm`) so
  the fused TKG kernel's plain `x_norm * w` produces Gemma semantics.
- ✅ **Fused `gate_up_proj`** with `stride=2` ColumnParallel sharding so
  each TP rank holds interleaved (gate, up) chunks — critical for TP=64
  where a contiguous split would give some ranks "all gate" or "all up".
- ✅ **MSA prefill (Lightning Indexer + block-sparse causal mask)** on
  the 57 sparse layers (indices 3..59). Adds `MiniMaxM3Indexer` submodule
  running q/k projections, per-head RMSNorm, partial RoPE, block-max
  pooling, and top-K block selection to produce the additive attention
  mask consumed by NxDI's attention path.
- ✅ **HF state-dict converter**: strip `language_model.model.` prefix,
  fuse per-expert `w1/w3` into `gate_up_proj`, stack `w2` into
  `down_proj`, inject zero biases for MoE experts (NxDI's preshard hook
  adds `+1.0` to the up-half), rename `index_{q,k}_{proj,norm}` under
  `indexer.*`, dequantize MXFP8 to bf16 on host.
- ✅ **Right-padded batching**. NxDI's compiled NEFF gathers the last
  real token via `argmax(position_ids)` — using left-padded input silently
  reads a pad position and produces identical garbage across prompts.
- ✅ Compile + 64-rank shard + load + warmup + TTFT + ITL on the full
  60-layer / 128-expert / 854 GB model with TP=64, `moe_ep=64`, `moe_tp=1`,
  batch=32, seq_len=512; and batch=8, seq_len=2048 (via monkey-patching
  `DEFAULT_SELECTIVE_LOADING_THRESHOLD=0` to route TKG through
  `forward_all_experts_EP` instead of the unsupported selective-loading
  branch when `batch × top_k / num_experts < 1`).

## What this port does NOT do

- ❌ **Vision tower, multi-modal projector, MTP modules** — filtered by
  the state-dict converter. This is a text-only port.
- ❌ **Long-context (≥ 8K) inference on batch ≥ 4**. Prefill KV cache at
  8 K × batch 8 needs ~30 GB and per-rank HBM is 24 GB. Enabling
  `attention_dp_degree > 1` or `cp_degree > 1` (currently `1` in this
  port) would shard KV cache and unblock long context.
- ❌ **Decode-side MSA**. Prefill runs the Lightning Indexer, but
  decode falls back to dense causal attention over the full KV cache
  — MSA over decode requires a persistent `idx_k` cache with NxDI
  output aliasing (attempted in v13 but hit an XLA async-trace race).
  Not a correctness issue at moderate context lengths; a performance
  optimization for the 1 M-token max_position range.
- ❌ **Batch = 8, seq_len = 4096** — the compile succeeds but the
  produced NEFF hangs at collectives `OP:0` (Neuron compiler bug at
  that specific shape). Workaround: `batch=4` for 4K compiles.
- ❌ **Left-padded input**. Use `tokenizer.padding_side="right"`; the
  compiled model reads the last token via `max(position_ids).indices`
  which requires right padding to correctly locate the final real token.

## Usage

```python
import torch
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Make `src/` importable
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path("contrib/models/MiniMax-M3/src")))
from modeling_minimax_m3 import (
    NeuronMiniMaxM3ForCausalLM, MiniMaxM3InferenceConfig, MiniMaxM3NeuronConfig,
)

MODEL_PATH = "/home/ubuntu/models/MiniMax-M3/"
COMPILED_PATH = "/home/ubuntu/neuron_models/MiniMax-M3/"

neuron_config = MiniMaxM3NeuronConfig(
    tp_degree=64, ep_degree=1, logical_nc_config=2,
    batch_size=32, max_batch_size=32, ctx_batch_size=32, tkg_batch_size=32,
    seq_len=512, n_active_tokens=512,
    torch_dtype=torch.bfloat16, capacity_factor=2.0, glu_mlp=True,
    moe_ep_degree=64, moe_tp_degree=1,
    context_encoding_buckets=[512],
    fused_qkv=True, save_sharded_checkpoint=True,
    router_config={"act_fn": "sigmoid", "dtype": "float32"},
    blockwise_matmul_config={
        "use_shard_on_block_dynamic_while": True,
        "block_sharding_strategy": "PING_PONG",
    },
)

# Promote text_config → top-level for the sparse causal LM head.
with open(f"{MODEL_PATH}/config.json") as f:
    raw = json.load(f)
text_cfg = dict(raw["text_config"])
text_cfg["model_type"] = "minimax_m3"
text_cfg["architectures"] = ["MiniMaxM3SparseForCausalLM"]
hf_cfg = PretrainedConfig(**text_cfg)
hf_cfg._name_or_path = MODEL_PATH

config = MiniMaxM3InferenceConfig(
    neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg),
)
config._name_or_path = MODEL_PATH

model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
model.compile(COMPILED_PATH)
model.load(COMPILED_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, padding_side="right",
)
# ...generate — see test/integration/test_model.py for a full loop.
```

## Compatibility Matrix

| Instance / SDK | Status |
|---|---|
| Trn2.48xlarge | Supported. TP=64, LNC=2, batch=32, seq_len=512 works end-to-end. Larger seq_len is limited by per-rank 24 GB HBM (see "What this port does NOT do"). |
| Trn1 | Not tested; the model is unlikely to fit. |
| Inf2 | Not tested. |

NeuronX SDK: tested against the `aws_neuronx_venv_pytorch_2_9_nxd_inference` venv at `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` (transformers 4.57+).

## Testing

### Synthetic smoke test (no checkpoint needed)

Validates the modeling code end-to-end on Neuron hardware against a tiny
M3-shaped config built from random weights. Useful when iterating on the
modeling code or before paying the 854 GB download cost.

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
  contrib/models/MiniMax-M3/test/integration/smoke_test_synthetic.py
```

### Full integration test (real checkpoint)

```bash
# Download the checkpoint (854 GB). With hf_transfer this takes ~2-3 hours.
HF_HUB_ENABLE_HF_TRANSFER=1 hf download MiniMaxAI/MiniMax-M3 \
  --local-dir /mnt/data/models/MiniMax-M3/

# Run the integration test (compiles on first run, then exercises TTFT / ITL).
M3_MODEL_PATH=/mnt/data/models/MiniMax-M3 \
M3_COMPILED_PATH=/mnt/data/neuron_models/MiniMax-M3 \
M3_TP_DEGREE=64 \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
  contrib/models/MiniMax-M3/test/integration/test_model.py
```

Useful environment variables (all optional):

| Variable | Default | Notes |
|---|---|---|
| `M3_MODEL_PATH` | `/home/ubuntu/models/MiniMax-M3/` | HF checkpoint path |
| `M3_COMPILED_PATH` | `/home/ubuntu/neuron_models/MiniMax-M3/` | NEFF cache path |
| `M3_TP_DEGREE` | `64` | Tensor parallel degree |
| `M3_BATCH_SIZE` | `32` | |
| `M3_SEQ_LEN` | `512` | Compile context length |
| `M3_NUM_LAYERS` | `0` | If > 0, override `num_hidden_layers` (smoke testing) |
| `M3_NUM_EXPERTS` | `0` | If > 0, override `num_local_experts` (smoke testing) |

## Development History

The port went through 15 iterations (v3 – v15) to reach correctness. Each
one fixed a specific issue — padding side, RMSNorm Gemma pre-shift, MoE
router bias, MSA Lightning Indexer, and finally the missing SwiGLU-OAI
`(up + 1.0)` bias in the block-sparse NKI kernel path. Full narrative
with symptoms, hypotheses ruled out, and evidence chains is in
[`HISTORY.md`](./HISTORY.md).

## Maintainer

Contributed by community via the NxDI contrib folder. See `CONTRIBUTING.md`.
