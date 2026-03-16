# DeepSeek V3 NXDI Project

## What This Is

NXDI (NeuronX Distributed Inference) implementation of DeepSeek V3 (671B MoE) for AWS Trainium.
Based on the upstream NXDI framework with custom model code for DeepSeek V3's MLA attention and MoE routing.

## Project Structure

- Model code: `src/neuronx_distributed_inference/models/deepseek/`
- Tests: `test/unit/models/deepseek/`
- Examples: `examples/` (generation scripts, logit matching tests)
- Install scripts: `install_deepseek.sh` (NXDI venv), `install_deepseek_vllm.sh` (vLLM venv)
- Detailed plan: `PLAN_deepseek_v3.md` — READ THIS FIRST for architecture details and remaining work

## Setup

```bash
bash install_deepseek.sh    # Install into NXDI venv
source /opt/aws_neuronx_venv_pytorch_*_nxd_inference/bin/activate
python -m pytest test/unit/models/deepseek/ -v
```

## Current Status

Phases 1-7 + 9 complete. Phases 8 (benchmarking) and 10 (profiling) need 671B model on trn2.48xlarge.
See `PLAN_deepseek_v3.md` for full details.

## Key Technical Notes

- MLA attention does NOT use NeuronAttentionBase (GQA incompatible with MLA weight absorption)
- KV cache format: `(combined, combined)` where combined = `[k_pe | compressed_kv]`, shape `(bsz, 1, seq_len, 576)`
- RoPE uses interleaved layout with `rotate_fn` — NO transpose needed (unlike optimum-neuron which uses `rotate_half`)
- Custom router `DeepseekV3Router` handles group-based selection + `e_score_correction_bias`
- `disable_numeric_cc_token = True` required for all-gather/reduce-scatter
- vLLM model type is `"deepseekv3"` (no underscore) due to architecture name parsing
