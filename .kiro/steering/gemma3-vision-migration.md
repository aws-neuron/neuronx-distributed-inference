---
inclusion: fileMatch
fileMatchPattern: ['**/contrib/models/gemma3-vision/**/*', '**/tmp/external-code/**/*']
---

# Gemma3 Vision Model Migration

## Context

Migrate Gemma3 VLM from `tmp/external-code/` to `contrib/models/gemma3-vision/`.

**Version Compatibility:**
- Source: NxDI v0.6.10598 (Neuron 2.26.1)
- Target: NxDI v0.7.14366 (Neuron 2.27.1)
- **Expect API breaking changes requiring fixes**

**Architecture:**
- Gemma3: VLM with dual configs (text + vision), SigLIP encoder, no custom KV cache
- Reference (Cohere2): Text-only, custom KV cache manager
- Use Cohere2 for structure, not implementation details

## Migration Milestones

### Milestone 1: Core Migration & Integration Test

1. **Move VLM files** to `contrib/models/gemma3-vision/src/gemma3_vision/`:
   - `tmp/external-code/models/gemma3/modeling_gemma3.py` → `modeling_gemma3.py`
   - `tmp/external-code/models/gemma3/modeling_gemma3_vision.py` → `modeling_gemma3_vision.py`
   - `tmp/external-code/models/gemma3/modeling_gemma3_text.py` → `modeling_gemma3_text.py` (optional)
   - `tmp/external-code/models/siglip/` → `siglip/`
   - Create `__init__.py` exporting: `NeuronGemma3ForCausalLM`, `Gemma3InferenceConfig`, `NeuronGemma3VisionModel`

2. **Fix imports** - Update all import paths to work in new location

3. **Write integration test** at `contrib/models/gemma3-vision/test/integration/test_model.py`:
   - Use `tmp/external-code/scripts/generation_gemma3.py` as template
   - Follow `contrib/models/cohere2/test/integration/test_model.py` structure
   - Test text+image generation (primary) and text-only (secondary)
   - Validate accuracy via logit matching against HuggingFace
   - Use config from `v14_bs1.py` (non-quantized, TP=8, BS=1, SEQ=512)

4. **Fix API breaks** - Run tests, fix v0.6→v0.7 API changes until tests pass

5. **Create README.md** following `contrib/models/cohere2/README.md`:
   - Usage example with image input
   - Compatibility matrix
   - Checkpoint: `google/gemma-3-27b-it`
   - Test command

### Milestone 2: Unit Tests (Optional)

Migrate from `tmp/external-code/test/unit/models/gemma3/`:
- `test_rope.py` - Dual RoPE (global/local)
- `test_vision_model.py` - Vision encoder accuracy
- Refactor per NxDI conventions

### Milestone 3: vLLM Integration Assessment

Check if `tmp/external-code/vllm_neuron_modified/` patches still needed in Neuron 2.27.1:
- `worker/neuronx_distributed_model_loader.py`
- `worker/neuronx_distributed_model_runner.py`

### Milestone 4: Code Simplification

Review workarounds for v0.6 bugs - many may be fixed in v0.7.

## Source File Priorities

**HIGH (Milestone 1):**
- `tmp/external-code/models/gemma3/*.py` - Core model
- `tmp/external-code/models/siglip/` - Vision encoder
- `tmp/external-code/scripts/generation_gemma3.py` - Test template

**MEDIUM (Milestone 2):**
- `tmp/external-code/test/unit/models/gemma3/` - Unit tests

**LOW (Reference only):**
- `tmp/external-code/e2e_pipeline/configs/` - Config examples (v14_bs1, v16_bs4, v18_bs1, v19_bs1)

**DEFERRED (Milestone 3):**
- `tmp/external-code/vllm_neuron_modified/` - vLLM patches

## Target Structure

```
contrib/models/gemma3-vision/
├── README.md
├── src/gemma3_vision/
│   ├── __init__.py
│   ├── modeling_gemma3.py
│   ├── modeling_gemma3_vision.py
│   ├── modeling_gemma3_text.py
│   └── siglip/
│       ├── __init__.py
│       ├── modeling_siglip.py
│       └── layers.py
└── test/
    ├── integration/
    │   └── test_model.py
    └── unit/ (optional)
```

## Key Patterns

### Configuration (Dual Config for VLM)

```python
# Text config for context/token generation
text_config = NeuronConfig(
    tp_degree=8, batch_size=1, seq_len=512,
    fused_qkv=True, attn_kernel_enabled=True,
    enable_bucketing=True, context_encoding_buckets=[512],
    token_generation_buckets=[512]
)

# Vision config for encoder
vision_config = NeuronConfig(
    tp_degree=8, batch_size=1, seq_len=512,
    fused_qkv=False, attn_kernel_enabled=True,
    enable_bucketing=True, buckets=[1]  # Auto-bucket 1024→seq_len
)

# Combined config
config = Gemma3InferenceConfig(
    text_neuron_config=text_config,
    vision_neuron_config=vision_config,
    load_config=load_pretrained_config(model_path)
)
```

### Model Class Hierarchy

```python
class Gemma3InferenceConfig(ImageToTextInferenceConfig):
    # Extends ImageToTextInferenceConfig (not InferenceConfig)
    # Has text_neuron_config and vision_neuron_config

class NeuronGemma3ForCausalLM(NeuronBaseForImageToText):
    # Extends NeuronBaseForImageToText (not NeuronBaseForCausalLM)
    text_model_cls = NeuronGemma3TextModel
    vision_model_cls = NeuronGemma3VisionModel
    
    def enable_vision_encoder(self, enable_wlt_optimization=True):
        # Auto-bucketing for vision: 1024→seq_len
        
    def get_compiler_args(self):
        # O1 for vision, O2 for token gen
        
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        # Handle text + vision state dicts
```

### Quantization Exclusions

```python
modules_to_not_convert = [
    "multi_modal_projector",
    "vision_tower",
    *[f"language_model.model.layers.{l}.self_attn" for l in range(num_layers)],
    "language_model.lm_head",
]
```

### Integration Test Structure

```python
def test_model_accuracy_and_performance(batch_size, seq_len):
    # 1. Setup configs
    text_config = NeuronConfig(...)
    vision_config = NeuronConfig(...)
    config = Gemma3InferenceConfig(text_config, vision_config, ...)
    
    # 2. Compile/load model
    model = NeuronGemma3ForCausalLM(model_path, config)
    model.compile(compiled_path)
    model.load(compiled_path)
    
    # 3. Test text+image
    input_ids, attention_mask, pixel_values, vision_mask = prepare_generation_inputs_hf(
        text_prompt, image_path, processor, 'user', config
    )
    outputs = model.generate(...)
    
    # 4. Validate accuracy
    check_accuracy_logits(model, tokenizer, generation_config)
    
    # 5. Test text-only
    input_ids, attention_mask, _, _ = prepare_generation_inputs_hf(
        text_prompt, None, processor, 'user'
    )
    outputs = model.generate(...)
```

## Common API Changes (v0.6→v0.7)

When fixing imports and API breaks, check:
- Base class method signatures
- Config parameter names
- Import paths for utilities
- Generation/sampling APIs
- KV cache interfaces (if used)

## Validation Checklist

**Milestone 1:**
- [ ] All imports resolve
- [ ] Model compiles (context, token gen, vision)
- [ ] Integration test passes
- [ ] Text+image generation works
- [ ] Text-only generation works
- [ ] Logits match HF reference
- [ ] README complete

**Milestone 2:**
- [ ] Unit tests migrated and passing

**Milestone 3:**
- [ ] vLLM integration assessed

**Milestone 4:**
- [ ] Code simplified (v0.6 workarounds removed)
