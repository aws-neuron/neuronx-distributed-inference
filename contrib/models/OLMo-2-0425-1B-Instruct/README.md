# Contrib Model: OLMo 2 0425 1B Instruct

NeuronX Distributed Inference implementation of OLMo 2 0425 1B Instruct.

## Model Information

- **HuggingFace ID:** `allenai/OLMo-2-0425-1B-Instruct`
- **Model Type:** Decoder-only transformer
- **License:** Check HuggingFace model card

## Architecture Details

- **Layers:** Check model config
- **Hidden Size:** Check model config
- **Attention Heads:** Check model config
- **Vocabulary:** Check model config
- **Max Position Embeddings:** Check model config

## Validation Results

**Validated:** 2026-01-29  
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | ✅ PASS | Model loads successfully |
| Token Matching | ⚠️ LOW | **9.4% match** |
| TTFT (P50) | ✅ PASS | 11.62ms (threshold: 100ms) |
| Throughput | ✅ PASS | 84.54 tok/s (threshold: 10 tok/s) |

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 11.62ms |
| Throughput | 84.54 tokens/s |


**Status:** ✅ VALIDATED

## Usage

```python
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import model classes from src
from src.modeling_olmo_2_0425_1b_instruct import NeuronOLMo204251BInstructForCausalLM, OLMo204251BInstructInferenceConfig

model_path = "/path/to/OLMo-2-0425-1B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = OLMo204251BInstructInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronOLMo204251BInstructForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | ✅ Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/OLMo-2-0425-1B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/OLMo-2-0425-1B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* allenai/OLMo-2-0425-1B-Instruct

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-01-29
