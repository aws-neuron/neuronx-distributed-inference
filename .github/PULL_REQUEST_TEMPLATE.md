## Description

*Provide a brief description of your contribution*

## Model Information

**Model Name:** *e.g., Llama 3.3*

**Model Architecture:** *e.g., Decoder-only transformer*

**Purpose:** *e.g., Text generation, image-to-text, etc.*

## Checklist

Please ensure your PR includes the following items. Refer to the [contrib/CONTRIBUTING.md](../contrib/CONTRIBUTING.md) for detailed guidelines.

### Required Components

- [ ] **Accuracy Test** (ex. `test/integration/test_model.py`)
  - At least one integration test that validates model accuracy
  - Uses logit validation or equivalent accuracy verification
  - Test can compile and run the model on Neuron

- [ ] **README.md** with the following sections:
  - [ ] **Usage Example**: Clear code example showing how to use the model
  - [ ] **Compatibility Matrix**: Table showing tested Neuron SDK versions and instance types (Trn1/Trn2/Inf2)
  - [ ] **Example Checkpoints**: Links to compatible model checkpoints (e.g., HuggingFace Hub)
  - [ ] **Testing Instructions**: Command to run the test suite for the model

- [ ] **Source Code** (`src/`)
  - Modeling code following NxD Inference patterns
  - Properly structured in the contrib folder hierarchy

### Optional Components

- [ ] **Unit Tests** (CPU or Neuron-based)
  - Tests for individual modeling components
  - Located in `test/unit/` directory

## Folder Structure

Confirm your contribution follows this structure:

```
/contrib/models/<model_name>/
  README.md
  /src
    <modeling code>
  /test
    /unit (optional)
    /integration
      test_model.py
```

## Testing

**How did you test this model?**

*Describe your testing approach, instance types used, and any specific configurations*

**Test Results:**

*Paste relevant test output or link to test runs*

## Compatibility

**Tested with:**
- **Neuron SDK Version(s):** *e.g., 2.24*
- **Instance Type(s):** *e.g., Trn1, Inf2*
- **PyTorch Version:** *if applicable*

## Additional Information

*Any other relevant information about the model, known limitations, performance characteristics, or special requirements*

## Related Issues

*Link any related issues or discussions*

## vLLM Integration

- [ ] This model is intended for use with vLLM
- [ ] Documentation includes vLLM registration instructions

*For vLLM integration details, see: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html#nxdi-onboarding-models-vllm*

---

**By submitting this PR, I confirm that:**
- [ ] I have read and followed the [contributing guidelines](../contrib/CONTRIBUTING.md)
- [ ] This is a community contribution and may have limited testing compared to officially-supported models
- [ ] The code follows best practices and is well-documented
- [ ] All required components listed above are included
