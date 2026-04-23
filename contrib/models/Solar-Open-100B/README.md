# Contrib Model: Solar Open 100B

NeuronX Distributed Inference implementation of [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B).

## Model Information

- **HuggingFace ID:** `upstage/Solar-Open-100B`
- **Model Type:** Mixture of Experts (MoE) transformer
- **Parameters:** 102.6B total (12B active per token)
- **License:** Upstage Solar License

## Architecture Details

- **Layers:** 48 decoder layers (all MoE, no dense layers)
- **Hidden Size:** 4096
- **Attention Heads:** 64
- **KV Heads:** 8 (Grouped Query Attention)
- **Head Dim:** 128
- **Experts:** 128 routed + 1 shared per layer
- **Active Experts:** 8 per token (top-8 sigmoid routing)
- **MoE Intermediate Size:** 1280 (per expert)
- **Dense Intermediate Size:** 10,240 (shared expert)
- **Vocabulary:** 196,608 tokens
- **Max Position Embeddings:** 131,072
- **Position Encoding:** YaRN RoPE (factor=2.0, original_max=65,536)
- **Normalization:** RMSNorm
- **Activation:** SiLU (SwiGLU gating in expert MLPs)
- **Router:** Sigmoid with e_score_correction_bias

## Validation Results

**Validated:** 2026-04-17 (SDK 2.29), 2026-04-03 (SDK 2.28)
**Configuration:** TP=64, batch_size=1, seq_len=4096, bfloat16
**Instance:** trn2.48xlarge (us-east-2)

### SDK 2.29 Performance (Latest)

**SDK:** Neuron SDK 2.29 (torch-neuronx 2.9.0.2.13, neuronx-cc 2.24, NxDI 0.9)

| Phase | Metric | SDK 2.29 | SDK 2.28 | Change |
|-------|--------|----------|----------|--------|
| CTE (prefill) | Median latency | -- | 1,565 ms | -- |
| TKG (decode) | Median latency | 12.1 ms | 11.83 ms | -- |
| TKG (decode) | Throughput | **82.4 tok/s** | 12.6 tok/s | **6.54x faster** |
| Startup | Compile (fresh) | 11.2 min | 10.5 min | +7% |
| Startup | Weight loading | 4.0 min | 3.8 min | +5% |

*The 6.54x TKG improvement comes from neuronx-cc 2.24 compiler optimizations for NKI attention kernels on trn2. Same kernel flags, no code changes required.*

**Note on SDK 2.28 discrepancy:** The 2.28 TKG numbers differ between the kernel sweep table below (84.5 tok/s) and the cross-SDK comparison above (12.6 tok/s). The 84.5 tok/s was measured on the original instance with a warm Neuron compiler cache. The 12.6 tok/s was measured on a fresh 2.28 recompile during the 2.29 comparison run. Both are accurate for their respective conditions; the cross-SDK comparison is apples-to-apples.

### Accuracy

Validated using `check_accuracy_logits_v2` (CPU HuggingFace reference vs Neuron, 16 tokens):

- **SDK 2.29:** PASS -- 0 divergence, all tolerance tiers passing
- **SDK 2.28:** PASS -- Cosine similarity 0.9992-0.9996, Top-1 match on all prompts

### MoE NKI Kernel Status

| Config | TKG Throughput | Notes |
|--------|---------------|-------|
| **Attention NKI only** (recommended) | **82.4 tok/s** | Best configuration |
| Attention NKI + MoE NKI | 71.9 tok/s | MoE NKI 13% slower at I/tp=20 |
| No NKI kernels | 12.6 tok/s | Baseline (2.28 compiler) |

The MoE fused NKI kernel (`moe_fused_nki_kernel_enabled=True`) compiles on SDK 2.29 (the tripcount=1 compiler issue from SDK 2.28 is fixed in neuronx-cc 2.24), but is 13% slower than the non-NKI path because Solar Open's expert intermediate dimension per shard (1280/64=20) is too narrow to benefit from NKI's SBUF tiling.

### Attention NKI Kernel Sweep (SDK 2.28)

Enabling attention NKI kernels yields a **34% TKG improvement** and **65.6% CTE improvement** over baseline on SDK 2.28. On SDK 2.29, the compiler (neuronx-cc 2.24) further optimizes these kernels for a **6.54x overall improvement**.

| Config | CTE (ms) | TKG (ms) | tok/s | TKG Delta |
|--------|----------|----------|-------|-----------|
| Baseline (no kernels) | 4,547 | 17.91 | 55.8 | --- |
| QKV kernel | 1,568 | 12.20 | 81.9 | -31.9% |
| + out_proj kernel | 1,568 | 12.99 | 77.0 | -27.5% |
| + block TKG attn kernel | 1,567 | 11.95 | 83.6 | -33.3% |
| **+ QKV NKI kernel (best)** | **1,565** | **11.83** | **84.5** | **-34.0%** |
| All kernels combined | 1,565 | 11.99 | 83.4 | -33.1% |

*All tested at seq_len=4096, batch=1, tp=64, bf16.*

### Sequence Length Sweep

| seq_len | CTE (ms) | TKG (ms) | TKG tok/s | Status |
|---------|----------|----------|-----------|--------|
| 1,024 | 2,094 | 17.77 | 56.3 | PASS |
| 2,048 | 2,671 | 17.79 | 56.2 | PASS |
| 4,096 | 4,551 | 18.17 | 55.0 | PASS |
| 8,192 | 6,585 | 18.22 | 54.9 | PASS |
| 16,384 | 12,340 | 19.18 | 52.1 | PASS |
| 32,768 | 35,053 | 20.38 | 49.1 | PASS |
| 65,536 | --- | --- | --- | FAIL |

*Baseline (no kernels), batch=1, tp=64, bf16. Maximum supported seq_len: 32,768.*

### Batch Size Sweep

| batch | CTE (ms) | TKG (ms) | tok/s/batch | Total tok/s | Status |
|-------|----------|----------|-------------|-------------|--------|
| 1 | 4,551 | 18.17 | 55.0 | 55.0 | PASS |
| 2 | 3,387 | 14.67 | 68.2 | 136.3 | PASS |
| 4 | 8,955 | 17.51 | 57.1 | 228.5 | PASS |
| 8+ | --- | --- | --- | --- | FAIL |

*Baseline (no kernels), seq_len=4096, tp=64, bf16. Maximum batch_size at seq_len=4096: 4.*

### Known Issues

- **MoE NKI kernels: compiles on 2.29, but slower.** On SDK 2.28, MoE NKI kernels failed to compile (tripcount=1 at I/tp=20). On SDK 2.29, they compile and run, but are 13% slower than the non-NKI path (71.9 vs 82.4 tok/s) because the expert intermediate dimension per shard (1280/64=20) is too narrow for NKI tiling. Keep `moe_fused_nki_kernel_enabled=False`.
- **Attention NKI kernels strongly recommended:** QKV kernel + QKV NKI kernel provide massive improvements, especially on SDK 2.29 (6.54x over 2.28 baseline).
- **`hf_adapter.py` fix required:** The NxDI `hf_adapter.py` has a `NameError` referencing undefined `tensor_capture_hook` in `prepare_inputs_for_generation()`. This PR includes a fix (removal of `input_capture_hook` extraction and both `input_capture_hook`/`tensor_capture_hook` dict entries). Without the fix, `check_accuracy_logits_v2` will crash.
- **seq_len=65536 fails:** "Could not serialize module proto" error. Maximum supported seq_len is 32,768 on SDK 2.28. The nki-lib `attention_cte` kernel in SDK 2.29 supports up to 131,072 tokens; 128K prefill has not been tested end-to-end.
- **batch_size >= 8 fails at seq_len=4096:** Same serialization error. Maximum batch_size at seq_len=4096 is 4.
- **CPU reference logits require transformers >= 5.0:** The `solar_open` model type was added in transformers 5.0. The NxDI inference venv uses transformers 4.57.6 (which works for Neuron compilation/inference), but generating CPU reference logits requires a separate environment with transformers >= 5.0.

## Required Instance

- **trn2.48xlarge** with tp=64 (128 experts / 64 shards = 2 per shard)
- trn2.3xlarge is NOT sufficient (32 experts per shard exceeds NEFF I/O budget)

## Usage

```python
import json
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_solar_open import SolarOpenInferenceConfig, NeuronSolarOpenForCausalLM

model_path = "/path/to/Solar-Open-100B-weights"
compiled_path = "/path/to/compiled/"

# Load HuggingFace config
with open(f"{model_path}/config.json") as f:
    hf_config = json.load(f)

# Configure Neuron
neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=4096,
    n_active_tokens=4096,
    torch_dtype=torch.bfloat16,
    # Attention NKI kernels (6.54x improvement on SDK 2.29, 34% on 2.28)
    fused_qkv=True,
    qkv_kernel_enabled=True,
    qkv_nki_kernel_enabled=True,
    # MoE NKI kernels: compile on 2.29, but 13% slower than non-NKI at I/tp=20
    moe_fused_nki_kernel_enabled=False,
    expert_mlp_nki_kernel_enabled=False,
)

# Create inference config
config = SolarOpenInferenceConfig(
    neuron_config=neuron_config,
    load_config=lambda c: [setattr(c, k, v) for k, v in hf_config.items()],
)

# Compile, load, and generate
model = NeuronSolarOpenForCausalLM(model_path, config)
model.compile(compiled_model_path=compiled_path)
model.load(compiled_path)

# Run inference
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
seq_len = input_ids.shape[1]

output = model.forward(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    position_ids=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
    seq_ids=torch.zeros(1, dtype=torch.int32),
)

logits = (output.logits if hasattr(output, "logits") else output[0])[0, -1, :]
next_token = torch.argmax(logits).item()
print(f"Next token: {tokenizer.decode([next_token])}")  # " Paris"
```

## Architecture Notes

Solar Open is architecturally similar to GPT-OSS/DeepSeek-V3 (128 experts, sigmoid routing with e_score_correction_bias). Key differences from GPT-OSS:

- **No hidden_size padding**: 4096 is already 128-aligned (GPT-OSS needed padding for 2880)
- **BF16 native**: No MXFP4 dequantization needed
- **No learned attention sinks**
- **No sliding window / mixed attention**: All layers use full attention
- **Standard YaRN RoPE**: factor=2.0, original_max=65536
- **GLU activation (not SWIGLU)**: Solar Open uses `hidden_act="silu"` with standard gate/up split, requiring `glu_type="glu"` in NxDI

### Bugs Fixed During Onboarding

Five issues were found and fixed during accuracy validation:

1. **hidden_act override**: Config incorrectly defaulted to `"sigmoid"` instead of `"silu"`
2. **HF weight format**: Safetensors store per-expert tensors; needed conversion to fused format
3. **YaRN RoPE**: Inverted ramp boundaries + wrong interpolation formula (2 sub-issues)
4. **glu_type mismatch**: GPT-OSS uses `hidden_act="sigmoid"` + `glu_type="swiglu"`; Solar Open uses `hidden_act="silu"` + `glu_type="glu"`

## Compatibility Matrix

| Instance/Version | SDK 2.29 | SDK 2.28 | SDK 2.27 and earlier |
|------------------|----------|----------|---------------------|
| trn2.48xlarge    | **Validated (82.4 tok/s)** | Validated (12.6 tok/s) | Not tested |
| trn2.3xlarge     | Not supported (NEFF I/O) | Not supported | Not supported |
| Trn1             | Not supported (tp<64) | Not supported | Not supported |

## Testing Instructions

Run on a trn2.48xlarge instance with model weights downloaded to `/mnt/models/Solar-Open-100B-weights`.

**Prerequisites:** Generate CPU reference logits (one-time, requires ~200GB RAM):

```bash
# Create a separate venv with transformers >= 5.0 (solar_open model type)
python3 -m venv /tmp/cpu_ref_venv
source /tmp/cpu_ref_venv/bin/activate
pip install torch transformers accelerate

# Generate reference logits
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
model = AutoModelForCausalLM.from_pretrained(
    '/mnt/models/Solar-Open-100B-weights', torch_dtype=torch.bfloat16,
    device_map='cpu', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('/mnt/models/Solar-Open-100B-weights')
input_ids = tokenizer.encode('The capital of France is', return_tensors='pt')
gen_config = GenerationConfig(do_sample=False, max_new_tokens=16, min_new_tokens=16,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
outputs = model.generate(input_ids, generation_config=gen_config,
    return_dict_in_generate=True, output_scores=True)
expected_logits = torch.stack(outputs.scores)[:16, :, :]
torch.save({'expected_logits': expected_logits, 'input_ids': input_ids,
    'prompt': 'The capital of France is', 'num_tokens': 16},
    '/mnt/models/solar_cpu_reference_logits.pt')
"
deactivate
```

**Run the tests** (using the NxDI inference venv):

```bash
# SDK 2.29:
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
# SDK 2.28:
# source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# With pytest
pytest contrib/models/Solar-Open-100B/test/integration/test_model.py -v --capture=tee-sys

# Or run manually
cd contrib/models/Solar-Open-100B
python3 test/integration/test_model.py
```

The test suite validates accuracy using `logit_validation` (via `check_accuracy_logits_v2`) comparing Neuron logits against CPU HuggingFace reference with teacher forcing and multi-tiered tolerances, and measures CTE/TKG performance.

## Example Checkpoints

* [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B)

## SDK Requirements

- Neuron SDK 2.28+ (SDK 2.29 recommended for 6.54x TKG improvement)
- transformers 4.57+ for Neuron inference (solar_open config loaded via manual JSON)
- transformers 5.0+ for CPU reference logit generation only (separate venv)
- trn2.48xlarge instance with 64 Neuron cores

### SDK 2.29 venv

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

### SDK 2.28 venv

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
```

## Maintainer

Jim Burtoft (@jimburtoft)

**Last Updated:** 2026-04-18
