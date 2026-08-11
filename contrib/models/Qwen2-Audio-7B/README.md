# Contrib Model: Qwen2-Audio-7B

NeuronX Distributed Inference implementation of [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B).

Both the audio encoder and the 7B language model run entirely on AWS Neuron hardware (Trainium/Inferentia2).

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2-Audio-7B`
- **Model Type:** Multimodal encoder-decoder (audio-to-text)
- **Parameters:** ~8.2B total (audio encoder ~600M + language model ~7.6B)
- **License:** Apache 2.0
- **Modalities:** Audio input → Text output

## Architecture Details

### Audio Encoder (Whisper-like)

| Property | Value |
|----------|-------|
| Type | Whisper-style transformer encoder |
| Hidden Size (d_model) | 1280 |
| Attention Heads | 20 |
| Encoder Layers | 32 |
| FFN Dim | 5120 |
| Max Source Positions | 1500 |
| Mel Bins | 128 |
| Activation | GELU |
| Output | 750 tokens × 1280 dim → projected to 4096 |

### Language Model (Qwen2)

| Property | Value |
|----------|-------|
| Type | Decoder-only transformer |
| Hidden Size | 4096 |
| Attention Heads | 32 |
| KV Heads | 32 |
| Hidden Layers | 32 |
| Intermediate Size | 11008 |
| Vocab Size | 156032 |
| Max Position Embeddings | 8192 |
| RoPE Theta | 10000 |
| Activation | SiLU |
| Normalization | RMSNorm (eps=1e-5) |

### Multi-Modal Projector

Linear projection from encoder output (1280) to LM hidden size (4096) with bias.

## Performance

Measured on trn1.32xlarge, TP=2, batch_size=1, BF16 precision:

| Metric | Value |
|--------|-------|
| Audio Encoding (Neuron) | ~60ms for 3-4s audio |
| Token Generation | 15-16 tok/s |
| End-to-End (4s audio → 30 tokens) | ~2s |

## Validation Results

**Configuration:** TP=2, batch_size=1, seq_len=1024, BF16

| Test | Audio | Prompt | Output | Status |
|------|-------|--------|--------|--------|
| Speech Transcription | "The quick brown fox jumps over the lazy dog" | Transcribe the speech word for word: | The quick brown fox jumps over the lazy dog. | ✅ PASS |
| Audio Captioning | Glass breaking sound | Generate the caption in English: | Glass is breaking. | ✅ PASS |
| Text-Only | N/A | What is the capital of France? | (correct response) | ✅ PASS |

## Usage

```python
import torch
import librosa
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

from src import NeuronQwen2AudioForConditionalGeneration, Qwen2AudioMultimodalConfig
from src.configuration_qwen2_audio import Qwen2AudioEncoderNeuronConfig

MODEL_PATH = "/path/to/Qwen2-Audio-7B/"
COMPILED_PATH = "/path/to/compiled/"

# 1. Configure
text_nc = NeuronConfig(
    tp_degree=2, batch_size=1, seq_len=1024,
    torch_dtype="bfloat16", save_sharded_checkpoint=False,
)
audio_nc = Qwen2AudioEncoderNeuronConfig(
    tp_degree=2, batch_size=1, seq_len=1500,
    torch_dtype="bfloat16", fused_qkv=False,
    buckets=[1], save_sharded_checkpoint=False,
)
config = Qwen2AudioMultimodalConfig.from_pretrained(
    MODEL_PATH, text_neuron_config=text_nc, audio_neuron_config=audio_nc,
)

# 2. Compile (first time only)
model = NeuronQwen2AudioForConditionalGeneration(model_path=MODEL_PATH, config=config)
model.compile(COMPILED_PATH)

# 3. Load compiled model
model = NeuronQwen2AudioForConditionalGeneration(model_path=MODEL_PATH, config=config)
model.load(COMPILED_PATH)

# 4. Generate
gen_model = HuggingFaceGenerationAdapter(model)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

audio, sr = librosa.load("audio.wav", sr=processor.feature_extractor.sampling_rate)
inputs = processor(
    text="<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:",
    audio=audio, return_tensors="pt", sampling_rate=sr,
)

outputs = gen_model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    audio_features=inputs["input_features"].to(torch.bfloat16),
    sampling_params=prepare_sampling_params(batch_size=1, top_k=[1]),
    generation_config=GenerationConfig(do_sample=False, eos_token_id=[151645]),
    max_new_tokens=50,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Implementation Notes

### Why `vision_*` References Appear in the Code

The NXDI framework's only multimodal base class is `NeuronBaseForImageToText`, which was
originally built for image-understanding models like Qwen2-VL and Llama-4. It hardcodes
field names like `vision_config`, `vision_embeddings`, `vision_mask`, `vision_encoder_model`,
`enable_vision_encoder()`, `encode_vision_to_input()`, etc. throughout its implementation
(`image_to_text_model_base.py`, `image_to_text_model_wrapper.py`, `model_base.py`).

There is no generic `NeuronBaseForEncoderDecoder` or `NeuronBaseForAudioToText` in the
framework. The underlying pattern is identical regardless of modality — an encoder produces
embeddings, they get scattered into the decoder's input sequence, and the decoder generates
tokens. The framework just named everything after its first use case (vision).

Since the porting guidelines prohibit modifying framework code, this implementation inherits
from `NeuronBaseForImageToText` and must use its method signatures and attribute names.
Specifically:

- **`encode_vision_to_input()`** — Framework calls this by name in `NeuronBaseModel.forward()`.
  It merges audio embeddings into text embeddings on Neuron. Cannot be renamed.
- **`vision_config` / `vision_encoder_model` / `vision_models`** — Framework attributes set
  and read by `NeuronBaseForImageToText.__init__()`, `compile()`, `load()`, and builders.
- **`vision_embeddings` / `vision_mask`** — Parameter names in `super().forward()` and
  `_get_model_outputs()` signatures. The compiled NEFF expects these at specific argument positions.
- **`enable_vision_encoder()` / `get_vision_compiler_args()`** — Called by framework during init
  and compilation.

Our public API uses `audio_*` names (`audio_config`, `audio_encoder`, `audio_features`,
`audio_neuron_config`). The `vision_*` names only appear in framework method overrides and
internal plumbing that users don't interact with.

### Encoder Layer Naming: `blocks.*` vs `layers.*`

The audio encoder's transformer layers are named `blocks.*` in the Neuron model (not `layers.*`
as in the HF checkpoint). This avoids a state dict key collision with the text model's
`layers.*` keys, since both the encoder and decoder share a single state dict during weight
loading. The `convert_hf_to_neuron_state_dict` function handles this remapping:
`audio_tower.layers.N.*` → `blocks.N.*`.

### Audio Encoder Compilation Strategy

The audio encoder uses `NeuronAttentionBase` (the same attention primitive used by all NXDI
models) rather than `torch_neuronx.trace`. Direct tracing of the full 32-layer encoder into
a single HLO graph causes the Neuron compiler to apply cross-layer optimizations that
accumulate numerical divergence, destroying the semantic content of the embeddings.
`NeuronAttentionBase` compiles each attention operation individually, preserving numerical
accuracy — the same approach used by Qwen2-VL's vision encoder.

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

```bash
pytest contrib/models/qwen2-audio-7b/test/integration/test_model.py --capture=tee-sys
```

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-21
