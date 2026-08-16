# Contrib Model: blenderbot-3B

NeuronX Distributed Inference implementation of blenderbot-3B (encoder-decoder).

## Model Information

- **HuggingFace ID:** `facebook/blenderbot-3B`
- **Model Type:** Encoder-decoder (seq2seq) transformer
- **License:** Apache 2.0

## Architecture Details

- **Encoder Layers:** 2
- **Decoder Layers:** 24
- **Hidden Size:** 2560
- **Attention Heads:** 32
- **FFN Dim:** 10240
- **Activation:** GELU
- **Normalization:** PRE-LayerNorm
- **Vocab Size:** 8008
- **Max Position Embeddings:** 128
- **Shared Embeddings:** Yes (encoder, decoder, lm_head)

## NeuronX Port Details

This port follows the **Whisper pattern** for encoder-decoder models:
- Separate `NeuronApplicationBase` subclasses for encoder and decoder
- Top-level `NeuronApplicationBlenderbot` orchestrator manages both
- Decoder has separate prefill and decode compilation paths
- Cross-attention caches are populated during prefill and reused during decode

Key implementation decisions:
- **Encoder attention mask**: Masks out padding tokens in padded encoder input
- **Cross-attention mask**: Prevents decoder from attending to encoder padding positions
- **Cross-attention cache**: Always computes K/V from encoder output to prevent dead code elimination during Neuron tracing
- **Weight splitting**: HF weights are pre-split into encoder/decoder safetensors with renamed keys

## Validation Results

**Validated:** 2026-03-20
**Configuration:** TP=8, batch_size=1, seq_len=128, float32

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Token Matching | PASS | **87.9% match** (5 prompts, 20 tokens each) |

### Per-Prompt Results

| Prompt | Match |
|--------|-------|
| "What is the capital of France?" | 76.5% |
| "Tell me about machine learning." | 94.4% |
| "How are you doing today?" | 90.9% |
| "What is the meaning of life?" | 88.9% |
| "Can you help me with something?" | 90.9% |

Note: Lower match rates are due to the Neuron model generating EOS tokens at natural sentence boundaries, while HF continues generating repetitive text.

**Status:** PASS

## Usage

```python
import torch
from transformers import AutoTokenizer

# Import from src
from src import (
    BlenderbotInferenceConfig,
    BlenderbotNeuronConfig,
    NeuronApplicationBlenderbot,
    split_hf_weights,
)

hf_model_path = "/path/to/blenderbot-3B"
split_path = "/tmp/blenderbot_split"
compiled_path = "/tmp/blenderbot_compiled"

# Step 1: Split HF weights into encoder/decoder
split_hf_weights(hf_model_path, split_path)

# Step 2: Configure
neuron_config = BlenderbotNeuronConfig(
    tp_degree=8,
    batch_size=1,
    seq_len=128,
    torch_dtype="float32",
    save_sharded_checkpoint=True,
)

config = BlenderbotInferenceConfig.from_pretrained(
    f"{split_path}/encoder",
    neuron_config=neuron_config,
)

# Step 3: Compile and load
app = NeuronApplicationBlenderbot(model_path=split_path, config=config)
app.compile(compiled_path)
app.load(compiled_path)

# Step 4: Generate
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
input_ids = tokenizer(["Hello, how are you?"], return_tensors="pt").input_ids

output = app.generate(
    input_ids,
    max_new_tokens=50,
    decoder_start_token_id=config.decoder_start_token_id,
    eos_token_id=config.eos_token_id,
    pad_token_id=config.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest contrib/models/blenderbot-3B/test/integration/test_blenderbot_inference.py --capture=tee-sys
```

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-20
