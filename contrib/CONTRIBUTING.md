# GitHub Contrib Folder Guidelines

## Contributing

NxD Inference maintains a `contrib` folder for community contributions. This folder contains models that community members have created and tested with the Neuron SDK. You can run or adapt these models in your inference workflows.

**Note: These models are community contributions and may have limited testing or performance optimizations, compared to officially-supported model architectures in the NxD Inference library. See the README in each model folder for details about how the model is tested.**

### Folder structure

Each model has a README, a source folder, and a test folder. The README has information about the model and what NxDI versions that this model is tested with.

```
/contrib
  CONTRIBUTING.md (this doc)
  /models
    /<model_name>
      /README.md
      /src
        <modeling code>
      /test
        /unit
        /integration
          /test_model.py
```

### How do I contribute?

We encourage you to create a PR to contribute modeling code. This PR should include the following:

1. At least one test that evaluates the accuracy and performance of the model. We provide a sample test script (`test_model.py`) that you can use as a starting point. This sample test script uses logit validation to validate accuracy. This test script also benchmarks the model and compares the time-to-first-token (TTFT) and throughput against given thresholds.
2. A README that includes information about the model. This README must include the following:
    1. A usage example.
    2. A compatibility matrix that shows what Neuron instance types and Neuron SDK versions that this model is tested with.
    3. Links to compatible checkpoints (such as on HuggingFace Hub).
    4. Information about how to run the tests for the model.
3. (Optional) Unit tests for the modeling code. To improve code quality, you can write unit tests for the modeling code. You can provide classic unit tests that run on CPU, and/or unit tests that run on Neuron instances. For more information about utilities to help you write Neuron unit tests, see [Testing modules and functions on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/writing-tests.html#testing-modules-and-functions-on-neuron).

After you create a PR, the Neuron team will review it and request any changes as needed.

### How do I use models in the `contrib` folder?

Follow these instructions to use a model from the `contrib` folder.

1. Clone the repository.
2. Refer to the model’s README and test script to understand how to run the model.
3. Copy the modeling code into your application.
4. Download a model checkpoint from HuggingFace or another source.
5. Use the checkpoint and modeling code to run inference.

### How do I use custom models with vLLM?

NxD Inference integrates with vLLM so that you can easily serve models for inference. You can register a custom model to serve it through vLLM. For more information, see the following:

* [vLLM User Guide for NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html)
* [Integrating custom models with vLLM](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html#nxdi-onboarding-models-vllm)
