# Requirements: Gemma3-Vision Model Migration

## 1. Overview

Migrate the Gemma3-Vision VLM (Vision-Language Model) from `tmp/external-code/` to the proper contrib models structure at `contrib/models/gemma3-vision/`. This migration involves upgrading from NxDI v0.6.10598 (Neuron 2.26.1) to v0.7.14366 (Neuron 2.27.1), which includes API breaking changes that must be addressed.

**Key Context:**
- **Architecture**: Gemma3 is a VLM with dual configs (text + vision), SigLIP vision encoder, and no custom KV cache
- **Version Upgrade**: Source code is on v0.6, target is v0.7 - expect and fix API breaking changes
- **Reference Model**: Use Cohere2 contrib model for structure patterns, not implementation details
- **Milestone Focus**: This spec covers Milestone 1 (Core Migration & Integration Test)

## 2. User Stories

### 2.1 As a developer, I want the Gemma3-Vision VLM files migrated to the contrib structure
So that the model follows NxDI organizational standards and works with the v0.7 API.

**Acceptance Criteria:**
- Core VLM files migrated to `contrib/models/gemma3-vision/src/gemma3_vision/`:
  - `modeling_gemma3.py` (main model with dual config support)
  - `modeling_gemma3_vision.py` (vision encoder integration)
  - `modeling_gemma3_text.py` (text model, optional but recommended)
- SigLIP vision encoder files migrated to `contrib/models/gemma3-vision/src/gemma3_vision/siglip/`:
  - `modeling_siglip.py`
  - `layers.py`
- `__init__.py` created at `src/gemma3_vision/` exporting:
  - `NeuronGemma3ForCausalLM` (main model class)
  - `Gemma3InferenceConfig` (dual config class)
  - `NeuronGemma3VisionModel` (vision model class)
- `__init__.py` created at `src/gemma3_vision/siglip/` exporting SigLIP components
- All imports updated from `models.gemma3.*` and `models.siglip.*` to new package paths
- All v0.6 → v0.7 API breaking changes identified and fixed

### 2.2 As a developer, I want integration tests that validate Gemma3-Vision on Neuron hardware
So that I can verify the model compiles correctly and produces accurate outputs for both text+image and text-only inputs.

**Acceptance Criteria:**
- Integration test created at `contrib/models/gemma3-vision/test/integration/test_model.py`
- Test structure follows `contrib/models/cohere2/test/integration/test_model.py` pattern
- Test uses `tmp/external-code/scripts/generation_gemma3.py` as functional template
- Test configuration based on `v14_bs1.py` (non-quantized, TP=8, BS=1, SEQ=512)
- Test validates **dual config setup**:
  - Text config: `fused_qkv=True`, `attn_kernel_enabled=True`, bucketing enabled
  - Vision config: `fused_qkv=False`, `attn_kernel_enabled=True`, auto-bucketing (1024→seq_len)
- Test validates **text+image generation** (primary use case):
  - Loads image and text prompt
  - Calls `prepare_generation_inputs_hf()` with image
  - Generates output tokens
  - Validates logits match HuggingFace reference using `check_accuracy_logits()`
- Test validates **text-only generation** (secondary use case):
  - Calls `prepare_generation_inputs_hf()` without image
  - Generates output tokens
  - Validates accuracy
- Test includes parametrized cases for different batch sizes and sequence lengths
- Model compilation succeeds for context encoding, token generation, and vision encoder
- All test cases pass with correct logit matching

### 2.3 As a user, I want comprehensive documentation for Gemma3-Vision
So that I understand how to use this VLM with dual configs and multimodal inputs.

**Acceptance Criteria:**
- `contrib/models/gemma3-vision/README.md` created following `contrib/models/cohere2/README.md` structure
- Documentation includes:
  - **Model Description**: Gemma3-Vision VLM with SigLIP encoder and dual config architecture
  - **Usage Example**: Complete code showing:
    - Dual config setup (text + vision NeuronConfig)
    - Model initialization with `NeuronGemma3ForCausalLM`
    - Image + text input preparation
    - Generation with multimodal inputs
    - Text-only generation example
  - **Compatibility Matrix**: Tested Neuron SDK versions (2.27.1+) and instance types (Trn1/Trn2/Inf2)
  - **Example Checkpoint**: `google/gemma-3-27b-it` from HuggingFace
  - **Testing Instructions**: Command to run integration tests
  - **Key Architecture Notes**: Dual config requirement, SigLIP encoder, quantization exclusions

## 3. Technical Requirements

### 3.1 File Structure
```
contrib/models/gemma3-vision/
├── README.md
├── src/
│   └── gemma3_vision/
│       ├── __init__.py
│       ├── modeling_gemma3.py
│       ├── modeling_gemma3_vision.py
│       ├── modeling_gemma3_text.py
│       └── siglip/
│           ├── __init__.py
│           ├── modeling_siglip.py
│           └── layers.py
└── test/
    ├── integration/
    │   └── test_model.py
    └── unit/
        └── .gitkeep
```

**Note**: `utils.py` and `ndxi_patch.py` from source may not be needed - evaluate during migration.

### 3.2 Model Class Hierarchy Requirements
- `Gemma3InferenceConfig` must extend `ImageToTextInferenceConfig` (not `InferenceConfig`)
- `NeuronGemma3ForCausalLM` must extend `NeuronBaseForImageToText` (not `NeuronBaseForCausalLM`)
- Must implement `text_model_cls` and `vision_model_cls` attributes
- Must implement `enable_vision_encoder()` method with auto-bucketing
- Must implement `get_compiler_args()` returning O1 for vision, O2 for token gen
- Must implement `convert_hf_to_neuron_state_dict()` handling text + vision state dicts

### 3.3 Dual Configuration Requirements
The model requires separate NeuronConfig instances for text and vision:

**Text Config** (context/token generation):
- `fused_qkv=True`
- `attn_kernel_enabled=True`
- `enable_bucketing=True`
- `context_encoding_buckets` and `token_generation_buckets` specified

**Vision Config** (encoder):
- `fused_qkv=False`
- `attn_kernel_enabled=True`
- `enable_bucketing=True`
- `buckets=[1]` for auto-bucketing (1024→seq_len)

### 3.4 Quantization Exclusions
Must exclude from quantization:
- `multi_modal_projector`
- `vision_tower`
- All `self_attn` layers in language model
- `lm_head`

### 3.5 Integration Test Requirements
- Use `google/gemma-3-27b-it` checkpoint path
- Test config: TP=8, BS=1, SEQ=512 (from v14_bs1.py)
- Test both text+image and text-only generation
- Use `prepare_generation_inputs_hf()` for input preparation
- Validate with `check_accuracy_logits()` against HuggingFace reference
- Include parametrized test cases for different configurations

### 3.6 API Migration Requirements
Must identify and fix v0.6 → v0.7 breaking changes in:
- Base class method signatures
- Config parameter names
- Import paths for utilities
- Generation/sampling APIs
- KV cache interfaces (if used)

## 4. Dependencies

- NeuronX Distributed Inference v0.7.14366 (Neuron SDK 2.27.1)
- HuggingFace transformers library
- PyTorch with Neuron support
- Access to `google/gemma-3-27b-it` checkpoint
- Test image file (can use `tmp/external-code/scripts/dog.jpg`)

## 5. Constraints

- **Version Compatibility**: Must work with NxDI v0.7 API (breaking changes from v0.6)
- **Hardware Requirements**: Must run on Neuron hardware (Trn1/Trn2/Inf2 instances)
- **Architecture Constraints**: Must use dual config pattern for VLM
- **No Custom KV Cache**: Unlike Cohere2, Gemma3 uses standard KV cache
- **Preserve Source**: Do not modify `tmp/external-code/` until migration is verified

## 6. Success Criteria

**Milestone 1 Complete When:**
- [ ] All imports resolve in new location
- [ ] Model compiles successfully (context, token gen, vision)
- [ ] Integration test passes
- [ ] Text+image generation works and produces correct outputs
- [ ] Text-only generation works and produces correct outputs
- [ ] Logits match HuggingFace reference (accuracy validation passes)
- [ ] README.md is complete with usage examples and compatibility info

## 7. Out of Scope (Future Milestones)

**Milestone 2** (Optional):
- Unit test migration from `tmp/external-code/test/unit/models/gemma3/`
- `test_rope.py` (dual RoPE validation)
- `test_vision_model.py` (vision encoder accuracy)

**Milestone 3** (Deferred):
- vLLM integration assessment
- Evaluation of `tmp/external-code/vllm_neuron_modified/` patches

**Milestone 4** (Future):
- Code simplification (removing v0.6 workarounds)
- Performance optimization
- Additional feature development

**Not Included:**
- Migrating e2e_pipeline scripts
- Migrating benchmark configurations
- Adding new model features
