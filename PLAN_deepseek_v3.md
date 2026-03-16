# DeepSeek V3 on NXDI — Implementation Guide

## Quick Start

This document is a self-contained guide for working with the DeepSeek V3 NXDI implementation.
All model code lives in `~/environment/deepseek-nxdi/`.

### Setup on a Fresh Instance

```bash
cd ~/environment/deepseek-nxdi

# Install into NXDI venv (auto-detects /opt/aws_neuronx_venv_pytorch_*_nxd_inference/)
bash install_deepseek.sh

# Activate and run unit tests
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate  # or whichever version exists
python -m pytest test/unit/models/deepseek/ -v

# (Optional) Install into vLLM venv too
bash install_deepseek_vllm.sh
```

### Running on Device

```bash
# Mini model logit matching (needs 2+ NeuronCores, e.g. trn2.3xlarge):
python examples/test_logit_matching.py --tp-degree 2

# Full model (needs trn2.48xlarge + DeepSeek-V3 weights):
python examples/generation_deepseek_v3.py \
    --model-path /path/to/DeepSeek-V3 \
    --tp-degree 32 --seq-len 4096

# vLLM serving:
VLLM_PLUGINS=neuron vllm serve /path/to/DeepSeek-V3 \
    --tensor-parallel-size 32 --max-model-len 4096 --max-num-seqs 1 \
    --trust-remote-code --dtype bfloat16 \
    --no-enable-prefix-caching --no-enable-chunked-prefill
```

---

## Repository Layout

```
deepseek-nxdi/
  install_deepseek.sh          # Install model code into NXDI venv
  install_deepseek_vllm.sh     # Install model code into vLLM venv
  PLAN_deepseek_v3.md          # This file
  src/neuronx_distributed_inference/
    models/deepseek/
      __init__.py
      modeling_deepseek.py      # ~913 lines — full model implementation
      rope_util.py              # ~155 lines — YaRN RoPE
    utils/constants.py          # MODEL_TYPES registry (has "deepseek_v3" entry)
  test/unit/models/deepseek/
    test_modeling_deepseek.py   # MLA attention tests (prefill/decode)
    test_rope.py                # RoPE tests
    test_router.py              # Router correctness tests
    test_state_dict_conversion.py  # State dict conversion + router tests
    test_helper/
      reference_model.py        # Self-contained MLA reference impl
      util.py                   # Test utilities
  examples/
    generation_deepseek_v3.py   # Full model inference script
    test_logit_matching.py      # First-token accuracy test (needs device)
    test_logit_matching_full.py # Multi-step accuracy test (needs device)
```

---

## Architecture Overview

DeepSeek V3 is a 671B parameter MoE model (37B active per token).

| Feature | Value |
|---------|-------|
| Layers | 61 (3 dense + 58 MoE) |
| Hidden dim | 7168 |
| Attention | MLA (Multi-head Latent Attention) with LoRA-compressed Q |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128, qk_rope_head_dim: 64, v_head_dim: 128 |
| Heads | 128 |
| Routed experts | 256 (8 groups of 32, top-4 groups, 8 experts per token) |
| Shared experts | 1 |
| Routing | sigmoid + e_score_correction_bias + group selection + normalize + scale by 2.5 |
| RoPE | YaRN, interleaved layout |
| Dense intermediate | 18432 |
| MoE intermediate | 2048 |
| Vocab | 129280 |

### Key Design Decisions in the NXDI Implementation

1. **MLA Attention** (`DeepseekV3Attention`): Does NOT extend `NeuronAttentionBase` because GQA projections are incompatible with MLA's weight absorption. KV cache stores `(combined, combined)` where `combined = [k_pe | compressed_kv]` with shape `(bsz, 1, seq_len, 576)` (rope_dim 64 + kv_lora_rank 512).

2. **Custom Router** (`DeepseekV3Router`): Subclasses `RouterTopK` to implement group-based expert selection with `e_score_correction_bias`. Uses `_build_deepseek_moe()` instead of `initialize_moe_module`.

3. **Dense MLP** (`DeepseekV3DenseMLP`): Layers 0-2 use dense MLP with `intermediate_size=18432` (not MoE).

4. **RoPE**: Uses `rotate_fn` (interleaved layout) — NO transpose needed. This is different from optimum-neuron which uses `rotate_half` (split layout) and DOES need a transpose.

5. **State Dict Conversion**: `convert_deepseek_v3_hf_to_neuron_state_dict()` handles router rename, expert fusion (gate+up→gate_up_proj), e_score_correction_bias rename, and dense layer passthrough.

6. **Compiler Config**: `get_compiler_args()` returns `None` (uses framework defaults). `disable_numeric_cc_token = True` required for all-gather/reduce-scatter.

---

## Implementation Status

### Completed (Phases 1-7 + 9)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Bug fixes + config completion | DONE |
| 2 | Dense MLP support (layers 0-2) | DONE |
| 3 | Custom MoE router (group selection, bias, scaling) | DONE |
| 4 | State dict conversion (HF → Neuron) | DONE |
| 5 | RoPE interleave verification | DONE |
| 6 | On-device compilation (mini model, tp=2) | DONE |
| 7 | Logit matching (HF vs NXDI, first token exact match) | DONE |
| 9 | vLLM integration (mini model, serving works) | DONE |

All 17 unit tests pass. Logit matching: first token EXACT MATCH between HF (FP32) and NXDI (BF16).

### Remaining (Phases 8, 10)

| Phase | Description | Requirements |
|-------|-------------|-------------|
| 8 | Benchmarking (TTFT, TPOT, throughput) | Full 671B model + trn2.48xlarge |
| 10 | Neuron Explorer profiling | Full 671B model + trn2.48xlarge |

---

## Phase 8: Benchmarking (TODO)

### Goal
Measure compilation time, TTFT, TPOT, and throughput for DeepSeek V3 671B on trn2.48xlarge.

### Prerequisites
- trn2.48xlarge instance (64 NeuronCores)
- DeepSeek V3 weights downloaded (671B, ~1.3TB in BF16 safetensors)
- NXDI installed via `install_deepseek.sh`

### Steps

1. **Download model weights**:
   ```bash
   # From HuggingFace (requires access token):
   huggingface-cli download deepseek-ai/DeepSeek-V3 --local-dir /path/to/DeepSeek-V3
   ```

2. **Compile and run**:
   ```bash
   python examples/generation_deepseek_v3.py \
       --model-path /path/to/DeepSeek-V3 \
       --traced-model-path /tmp/deepseek_v3_traced \
       --tp-degree 32 --seq-len 4096
   ```

3. **Measure metrics**:
   - Compilation time: time the `model.compile()` call
   - TTFT: time from `generate()` call to first token output
   - TPOT: average time per subsequent token
   - Throughput: tokens/sec at various batch sizes

### Anticipated Issues

- **XLA weight layout optimizer crash**: Heterogeneous layers (3 dense + 58 MoE) may crash the XLA weight layout optimizer. If so, apply Moonlight's `_patch_neuron_config_skip_weights()` workaround — add a method that returns the dense layer weight names as skip list.
- **NCC_ITEN404 at large seq_len**: Start with `seq_len=512`, increase gradually.
- **OOM**: Use `priority_model_idx=0` on token_generation builder for weight sharing between context_encoding and token_generation NEFFs.
- **TP degree**: 671B with BF16 needs tp=32 minimum. Try tp=64 if tp=32 OOMs.

---

## Phase 10: Profiling (TODO)

### Goal
Profile NEFFs with Neuron Explorer to identify bottlenecks.

### Steps
1. Enable profiling during compilation
2. Use Neuron Explorer (not legacy NTFF profiler) to analyze NEFFs
3. Focus on: attention compute, MoE routing overhead, expert computation, all-reduce latency

---

## Technical Reference

### HF State Dict Key Names

```
# Attention (all 61 layers)
model.layers.{i}.self_attn.q_a_proj.weight           # [1536, 7168]
model.layers.{i}.self_attn.q_a_layernorm.weight       # [1536]
model.layers.{i}.self_attn.q_b_proj.weight            # [128*192, 1536]
model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight  # [576, 7168]
model.layers.{i}.self_attn.kv_a_layernorm.weight      # [512]
model.layers.{i}.self_attn.kv_b_proj.weight           # [128*256, 512]
model.layers.{i}.self_attn.o_proj.weight              # [7168, 128*128]

# Dense MLP (layers 0-2)
model.layers.{i}.mlp.gate_proj.weight                 # [18432, 7168]
model.layers.{i}.mlp.up_proj.weight                   # [18432, 7168]
model.layers.{i}.mlp.down_proj.weight                 # [7168, 18432]

# MoE (layers 3-60)
model.layers.{i}.mlp.gate.weight                      # [256, 7168] - router
model.layers.{i}.mlp.gate.e_score_correction_bias     # [256] - bias (buffer)
model.layers.{i}.mlp.experts.{e}.gate_proj.weight     # [2048, 7168]
model.layers.{i}.mlp.experts.{e}.up_proj.weight       # [2048, 7168]
model.layers.{i}.mlp.experts.{e}.down_proj.weight     # [7168, 2048]
model.layers.{i}.mlp.shared_experts.gate_proj.weight  # [2048, 7168]
model.layers.{i}.mlp.shared_experts.up_proj.weight    # [2048, 7168]
model.layers.{i}.mlp.shared_experts.down_proj.weight  # [7168, 2048]
```

### State Dict Conversion (HF → Neuron)

Key transformations in `convert_deepseek_v3_hf_to_neuron_state_dict()`:
1. `gate.weight` → `router.linear_router.weight`
2. `gate.e_score_correction_bias` → `router.e_score_correction_bias`
3. Per-expert `gate_proj` + `up_proj` → fused `gate_up_proj` [num_experts, 2*intermediate, hidden]
4. Per-expert `down_proj` → stacked `down_proj` [num_experts, hidden, intermediate]
5. Dense layers (0-2) pass through unchanged
6. `rank_util.rank` tensor added for TP

### vLLM Integration Notes

- vLLM parses `DeepseekV3ForCausalLM` → model type `"deepseekv3"` (no underscore, split on `For`, lowercase)
- NXDI registry uses `"deepseek_v3"` (with underscore)
- `install_deepseek_vllm.sh` registers BOTH keys
- Must use `VLLM_PLUGINS=neuron` if `optimum_neuron` is also installed in the venv
- `VLLM_MLA_DISABLE` is NOT needed (no MLA detection in vllm_neuron 0.13)

### Known Issues

- **DGE OOB with small models**: Mini model (vocab_size=32000, small dims) triggers DGE OOB on 2nd+ request. This is a Neuron compiler bug, not a model code issue. Expected to work with full 671B model.
- **vocab_size < ~20000**: Triggers DGE OOB. Use vocab_size >= 32000 for mini models.
- **BF16 vs FP32 divergence**: With random weights, autoregressive generation diverges after first token. This is expected due to small logit margins. Real trained weights have larger margins and should match better.

### Environment Details

- NXDI venv: `/opt/aws_neuronx_venv_pytorch_*_nxd_inference/` (comes with AMI)
- vLLM venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_*/` (comes with AMI)
- The install scripts copy model code INTO these venvs (they don't do editable installs)
- Unit tests run from the repo directory, not from the venv
