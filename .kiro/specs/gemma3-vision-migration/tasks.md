# Tasks: Gemma3-Vision Model Migration

## Overview

This task list implements the migration of Gemma3-Vision VLM from `tmp/external-code/` to `contrib/models/gemma3-vision/` with API compatibility fixes for NxDI v0.7.14366.

## Task List

### Phase 1: File Migration and Structure Setup

- [x] 1. Create directory structure for gemma3-vision contrib model
  - [x] 1.1 Create `contrib/models/gemma3-vision/src/gemma3_vision/` directory
  - [x] 1.2 Create `contrib/models/gemma3-vision/src/gemma3_vision/siglip/` directory
  - [x] 1.3 Create `contrib/models/gemma3-vision/test/integration/` directory
  - [x] 1.4 Create `contrib/models/gemma3-vision/test/unit/` directory with `.gitkeep`

- [ ] 2. Migrate core Gemma3 model files
  - [ ] 2.1 Copy `tmp/external-code/models/gemma3/modeling_gemma3.py` to `contrib/models/gemma3-vision/src/gemma3_vision/modeling_gemma3.py`
  - [ ] 2.2 Copy `tmp/external-code/models/gemma3/modeling_gemma3_vision.py` to `contrib/models/gemma3-vision/src/gemma3_vision/modeling_gemma3_vision.py`
  - [ ] 2.3 Copy `tmp/external-code/models/gemma3/modeling_gemma3_text.py` to `contrib/models/gemma3-vision/src/gemma3_vision/modeling_gemma3_text.py`

- [ ] 3. Migrate SigLIP vision encoder files
  - [ ] 3.1 Copy `tmp/external-code/models/siglip/modeling_siglip.py` to `contrib/models/gemma3-vision/src/gemma3_vision/siglip/modeling_siglip.py`
  - [ ] 3.2 Copy `tmp/external-code/models/siglip/layers.py` to `contrib/models/gemma3-vision/src/gemma3_vision/siglip/layers.py`

- [ ] 4. Create package initialization files
  - [ ] 4.1 Update `contrib/models/gemma3-vision/src/gemma3_vision/__init__.py` with exports for NeuronGemma3ForCausalLM, Gemma3InferenceConfig, NeuronGemma3VisionModel, NeuronGemma3TextModel
  - [ ] 4.2 Create `contrib/models/gemma3-vision/src/gemma3_vision/siglip/__init__.py` with exports for NeuronSiglipVisionModel, NeuronSiglipAttention, OutputChannelParallelConv2d
  - [ ] 4.3 Copy `tmp/external-code/models/utils.py` to `contrib/models/gemma3-vision/src/gemma3_vision/utils.py` (contains convert_state_dict_to_fused_qkv utility)

### Phase 2: Import Path Updates

- [ ] 5. Update imports in modeling_gemma3.py
  - [ ] 5.1 Replace `from models.gemma3.modeling_gemma3_text import` with `from gemma3_vision.modeling_gemma3_text import`
  - [ ] 5.2 Replace `from models.gemma3.modeling_gemma3_vision import` with `from gemma3_vision.modeling_gemma3_vision import`
  - [ ] 5.3 Replace `from models.utils import` with `from gemma3_vision.utils import`

- [ ] 6. Update imports in modeling_gemma3_vision.py
  - [ ] 6.1 Replace `from models.siglip.modeling_siglip import` with `from gemma3_vision.siglip.modeling_siglip import`
  - [ ] 6.2 Update any other relative imports to use new package structure

- [ ] 7. Update imports in modeling_gemma3_text.py
  - [ ] 7.1 Update any relative imports to use new package structure

- [ ] 8. Update imports in SigLIP files
  - [ ] 8.1 Update imports in `siglip/modeling_siglip.py` to use new package structure
  - [ ] 8.2 Update imports in `siglip/layers.py` to use new package structure

- [ ] 9. Verify all imports resolve without errors
  - [ ] 9.1 Run Python import test: `python -c "from gemma3_vision import NeuronGemma3ForCausalLM, Gemma3InferenceConfig"`
  - [ ] 9.2 Run Python import test: `python -c "from gemma3_vision.siglip import NeuronSiglipVisionModel"`

### Phase 3: API Compatibility Fixes (v0.6 → v0.7)

- [ ] 10. Review and document API compatibility status
  - [ ] 10.1 Verify Gemma3InferenceConfig already extends ImageToTextInferenceConfig (confirmed in source)
  - [ ] 10.2 Verify NeuronGemma3ForCausalLM extends NeuronBaseForImageToText (need to check)
  - [ ] 10.3 Document any v0.7 API changes discovered during initial compilation attempt
  - [ ] 10.4 Create list of required fixes based on compilation errors

- [ ] 11. Fix any identified API compatibility issues
  - [ ] 11.1 Update method signatures to match v0.7 base classes
  - [ ] 11.2 Update config initialization to match v0.7 requirements
  - [ ] 11.3 Fix any deprecated API usage
  - [ ] 11.4 Verify all base class method overrides match v0.7 signatures

- [ ] 12. Verify API compatibility fixes
  - [ ] 12.1 Attempt model initialization to catch runtime API errors
  - [ ] 12.2 Fix any remaining API compatibility issues discovered
  - [ ] 12.3 Document all API changes made for future reference

### Phase 4: Integration Test Implementation

- [ ] 13. Update integration test file
  - [ ] 13.1 Replace template imports with Gemma3-Vision imports (NeuronGemma3ForCausalLM, Gemma3InferenceConfig)
  - [ ] 13.2 Update model_path to `/home/ubuntu/models/google/gemma-3-27b-it/`
  - [ ] 13.3 Update compiled_model_path to `/home/ubuntu/neuron-models/gemma-3-27b-it/`
  - [ ] 13.4 Add AutoProcessor import for image processing

- [ ] 14. Implement dual config setup in test
  - [ ] 14.1 Create text_config with NeuronConfig(tp_degree=8, batch_size=1, seq_len=512, fused_qkv=True, attn_kernel_enabled=True, enable_bucketing=True)
  - [ ] 14.2 Create vision_config with NeuronConfig(tp_degree=8, batch_size=1, seq_len=512, fused_qkv=False, attn_kernel_enabled=True, enable_bucketing=True, buckets=[1])
  - [ ] 14.3 Initialize Gemma3InferenceConfig with both text_neuron_config and vision_neuron_config

- [ ] 15. Implement text+image generation test
  - [ ] 15.1 Add test image path (use `tmp/external-code/scripts/dog.jpg` or similar)
  - [ ] 15.2 Initialize AutoProcessor for image processing
  - [ ] 15.3 Call check_accuracy_logits with image_path parameter for multimodal validation
  - [ ] 15.4 Add property annotation: "Feature: gemma3-vision-migration, Property 3: Text+Image Generation Correctness"

- [ ] 16. Implement text-only generation test
  - [ ] 16.1 Call check_accuracy_logits without image_path parameter for text-only validation
  - [ ] 16.2 Add property annotation: "Feature: gemma3-vision-migration, Property 4: Text-Only Generation Correctness"

- [ ] 17. Add parametrized test cases
  - [ ] 17.1 Update pytest.mark.parametrize with configurations: (batch_size=1, seq_len=512), (batch_size=1, seq_len=2048)
  - [ ] 17.2 Add property annotation: "Feature: gemma3-vision-migration, Property 5: Model Compilation Success"

### Phase 5: Documentation

- [ ] 18. Update README.md for gemma3-vision
  - [ ] 18.1 Replace template content with Gemma3-Vision model description
  - [ ] 18.2 Add dual config architecture explanation (text + vision NeuronConfig requirement)
  - [ ] 18.3 Create usage example showing dual config setup with fused_qkv=True for text, fused_qkv=False for vision
  - [ ] 18.4 Add usage example for text+image generation with AutoProcessor and image input
  - [ ] 18.5 Add usage example for text-only generation
  - [ ] 18.6 Update compatibility matrix with Neuron SDK 2.27.1+ and instance types (Trn1/Trn2/Inf2)
  - [ ] 18.7 Update example checkpoint to `google/gemma-3-27b-it`
  - [ ] 18.8 Update testing instructions to reference gemma3-vision test path
  - [ ] 18.9 Add key architecture notes: SigLIP encoder, dual config requirement, quantization exclusions
  - [ ] 18.10 Add supported features table with status for TP, bucketing, quantization, etc.

### Phase 6: Validation and Testing

- [ ] 19. Run import validation tests
  - [ ] 19.1 Test main package imports: `python -c "from gemma3_vision import NeuronGemma3ForCausalLM, Gemma3InferenceConfig"`
  - [ ] 19.2 Test SigLIP imports: `python -c "from gemma3_vision.siglip import NeuronSiglipVisionModel"`
  - [ ] 19.3 Verify no ImportError or ModuleNotFoundError

- [ ] 20. Run model compilation test
  - [ ] 20.1 Initialize model with test configuration (TP=8, BS=1, SEQ=512)
  - [ ] 20.2 Compile model for context encoding, token generation, and vision encoder
  - [ ] 20.3 Verify compilation succeeds without errors
  - [ ] 20.4 Check compiled artifacts are created in expected locations

- [ ] 21. Run integration tests
  - [ ] 21.1 Execute `pytest contrib/models/gemma3-vision/test/integration/test_model.py -v`
  - [ ] 21.2 Verify text+image generation test passes
  - [ ] 21.3 Verify text-only generation test passes
  - [ ] 21.4 Verify logit matching validation succeeds

- [ ] 22. Run accuracy validation
  - [ ] 22.1 Compare Neuron model outputs with HuggingFace reference for text+image inputs
  - [ ] 22.2 Compare Neuron model outputs with HuggingFace reference for text-only inputs
  - [ ] 22.3 Verify logits match within tolerance (typically 1e-2 for bfloat16)
  - [ ] 22.4 Document any accuracy differences and investigate if needed

- [ ] 23. Final validation checklist
  - [ ] 23.1 All files migrated to new location with correct structure
  - [ ] 23.2 All imports resolve in new location
  - [ ] 23.3 All v0.6 → v0.7 API changes fixed
  - [ ] 23.4 Model compiles successfully (context, token gen, vision)
  - [ ] 23.5 Integration test passes
  - [ ] 23.6 Text+image generation works and produces correct outputs
  - [ ] 23.7 Text-only generation works and produces correct outputs
  - [ ] 23.8 Logits match HuggingFace reference (accuracy validation passes)
  - [ ] 23.9 README.md is complete with usage examples and compatibility info

## Notes

- **Checkpoint Path**: Use `/home/ubuntu/models/google/gemma-3-27b-it/` (with trailing slash for consistency)
- **Test Configuration**: TP=8, BS=1, SEQ=512 (baseline configuration)
- **Dual Config Requirement**: Text config must have `fused_qkv=True`, vision config must have `fused_qkv=False`
- **Auto-Bucketing**: Vision encoder uses `buckets=[1]` for auto-bucketing from 1024→seq_len
- **Compiler Args**: Vision encoder uses `-O1`, token generation uses `-O2`
- **Quantization**: Exclude `multi_modal_projector`, `vision_tower`, all `self_attn` layers, and `lm_head`
- **Utils File**: The `utils.py` file contains `convert_state_dict_to_fused_qkv` utility needed by the model
- **API Compatibility**: Source code already uses ImageToTextInferenceConfig and NeuronBaseForImageToText, so major API structure is v0.7 compatible

## Dependencies

- NeuronX Distributed Inference v0.7.14366 (Neuron SDK 2.27.1)
- HuggingFace transformers library
- PyTorch with Neuron support
- Access to `google/gemma-3-27b-it` checkpoint
- Test image file (can use `tmp/external-code/scripts/dog.jpg`)

## Success Criteria

All tasks in Phase 6 (Validation and Testing) must pass, including:
- Import tests succeed
- Model compilation succeeds
- Integration tests pass
- Accuracy validation passes (logits match HuggingFace reference)
- Documentation is complete

## Current Status

**Completed:**
- Directory structure created (Phase 1, Task 1)
- Template README.md and test file exist (need updating)

**Next Steps:**
- Begin Phase 1, Task 2: Migrate core Gemma3 model files
- Continue with import path updates and API compatibility verification
