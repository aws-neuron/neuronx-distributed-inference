# KVCacheReconstruct in NeuronxDistributed Inference
___
## Overview
Upon discovering an accuracy deviation, we want to (1) provide an easily interpretable accuracy profile/visualization/analysis to (2) quickly identify the location of the accuracy deviation.

By extracting the KV Cache of a Neuron implementation and comparing to a Golden (HuggingFace or Neuron CPU) implementation, we can analyze the autoregressive step (token index), layer, and KV head where the deviation may occur, reducing the time and search space of accuracy debugging.

Limitation: While KVCacheReconstruct can help locate the location of an accuracy issue, more fine-grained (i.e. module) level comparison is needed to root cause the issue.

### Goal
Currently, KV Cache comparison to Neuron Device is difficult because sharding leads to the KV Cache to be distributed across different ranks on device. Furthermore, many different sharding strategies lead to different KV Cache layouts.

Thus, the goal of this tool is to allow developers to reconstruct the KV Cache across different models and attention flavors for accuracy debugging. 

Through an extensible "plugin" interface, model builders need only to add a reconstruction function for a single layer (`reconstruct_cache_layer_<SHARDING_STRATEGY>(...)`) which can be used by `reconstruct_neuron_device_kv_cache_base(...)` for full KV Cache reconstruction.

### Expected Use Cases
Accuracy Mismatch (Top Down/Bottom Up):
- Top Down: Developer identifies an accuracy issue through task-level/logit-level deviation
    - They can use KVCacheReconstruct to find the layer and token index the deviation occurs
- Bottom Up: Developer wants to verify the correctness of a new feature
    - They can use KVCacheReconstruct to verify the correctness of their implementation.

Precision `dtype` Data Collection:
- Developers can compare KV Cache tensors in different precisions (i.e. FP32, FP16, BF16) to collect data on the impact of `dtype` on precision errors.
___
## Concepts
- Baseline or Golden: The reference implementation (either HuggingFace CPU or Neuron CPU) to compare against
	- Usually we run all CPU Baselines using `Tensor Parallelism Degree (TP) = 1`
- Test: The implementation to test (either Neuron CPU or Neuron Device)

When given an Accuracy Mismatch on Neuron Device from Goldens, the followup diagnostics steps below can help us pinpoint the accuracy issue:

| Golden Device      | Test Device   | Has Deviation? | Interpretation                                                                                                 |
| ------------------ | ------------- | -------------- | -------------------------------------------------------------------------------------------------------------- |
| 1. HuggingFace CPU    | Neuron CPU    | Yes            | NxDI implementation is incorrect                                                                               |
| 2. HuggingFace CPU    | Neuron CPU    | No             | Accuracy Issue likely comes from lowering/compiler/runtime/collectives; run CPU vs. Neuron Device as next step |
| 3. CPU (HF or Neuron) | Neuron Device | Yes            | Accuracy Issue comes from lowering/compiler/runtime/collectives                                                |
___
## How it Works
To extract the KV Cache from a Neuron Device Model:
1. Initialize model with Configurations
2. Run generation with a given prompt
3. Extract Sharded KV Cache
4. Reconstruct KV Cache \[Feature added through [`kv_cache_reconstruct_utils.py`](./kv_cache_reconstruct_utils.py)]

### End-To-End Comparison Flow
For Accuracy Debugging, we repeat the flow above for Golden and Reference implementation and compare the output tensors

```
+-----------------------------------------------------------------+
|                   Inference Config + Prompt                     |
+-----------------------------------------------------------------+
                     |                     |
                     |                     |
                     V                     V
+-----------------------------+     +-----------------------------+
| GOLDEN                      |     | TEST                        |
| Initialize CPU HF/Neuron    |     | Initialize Neuron           |
| Model                       |     | Device Model                |
+-----------------------------+     +-----------------------------+
              |                                    |
              |                                    |
              v                                    v
+-----------------------------+     +-----------------------------+
| Run Generation              |     | Run Generation              |
+-----------------------------+     +-----------------------------+
              |                                    |
              |                                    |
              v                                    v
+-----------------------------+     +-----------------------------+
| Extract Unsharded KV Cache  |     | Extract Sharded KV Cache    |
+-----------------------------+     +-----------------------------+
              |                                    |
              |                                    |
              |                                    v
              |                     +-----------------------------+
              |                     | Reconstruct KV Cache        |
              |                     +-----------------------------+
              |                                    |
              |                                    |
              |                                    |
              |                                    |
              v                                    v
+-----------------------------------------------------------------+
|                  Analysis: Compare KV Caches                    |
+-----------------------------------------------------------------+

```
___
## Example Usage
To run an example integration test, see [`test_kv_cache_reconstruct_utils.py`](../../../test/unit/utils/test_kv_cache_reconstruct_utils.py)
- Note: `--forked` is needed in `pytest` to ensure environment variables are isolated (i.e. distributed backends `gloo`, `fairscale`) 

```zsh
# Run as script
python ./test/integration/tp32/kv_cache_reconstruct/test_llama_3_2_1b_kv_cache_reconstruct.py

# Run through Pytest
pytest test/integration/tp32/kv_cache_reconstruct/test_llama_3_2_1b_kv_cache_reconstruct.py --forked
```
___
## How to Add Support for Additional Sharding Strategies
### Add Reconstruction Utility 
In [`kv_cache_reconstruct_utils.py`](./kv_cache_reconstruct_utils.py) add a sharding function for a specific layer using the `extract_all_k_or_v_heads_from_rank(...)` function 

```python
# Use function to move all ranks from Neuron Device to CPU
def extract_all_k_or_v_heads_from_rank(
    neuron_cache: NeuronDeviceCacheType, rank_idx: int, layer_idx: int, is_v_head: bool
) -> torch.Tensor:
	pass

def reconstruct_cache_layer_<SHARDING_STRATEGY>(
    neuron_cache: NeuronDeviceCacheType,
    inference_config: InferenceConfig,
    layer_idx: int,
    is_v_cache: bool,
) -> torch.Tensor:
	# Use extract_all_k_or_v_heads_from_rank(...) here
	pass

def reconstruct_neuron_device_kv_cache_<SHARDING_STRATEGY>(
    neuron_cache: NeuronDeviceCacheType,
    inference_config: InferenceConfig,
    layer_idx: int,
    is_v_cache: bool,
) -> torch.Tensor:
	k_cache, v_cache = reconstruct_neuron_device_kv_cache_base(
        neuron_cache, inference_config, reconstruct_cache_layer_<SHARDING_STRATEGY>
    )
    return k_cache, v_cache
```

### Add Unit Test
Add unit test for `reconstruct_cache_layer_<SHARDING_STRATEGY>(...)` and  `reconstruct_neuron_device_kv_cache_<SHARDING_STRATEGY>(...)`. 

See example: [`test_kv_cache_reconstruct_utils.py`](../../../test/unit/utils/test_kv_cache_reconstruct_utils.py)

### Add Integration Test
Add integration test for `reconstruct_neuron_device_kv_cache_<SHARDING_STRATEGY>(...)` comparing against HF CPU or Neuron CPU.

See example: [`test_llama_3_2_1b_kv_cache_reconstruct.py`](../../../test/integration/tp32/kv_cache_reconstruct/test_llama_3_2_1b_kv_cache_reconstruct.py)
___