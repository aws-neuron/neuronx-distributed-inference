import pytest
import torch

from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel
from neuronx_distributed.trace.mock_torchdist import mock_distributed

import neuronx_distributed_inference.modules.attention.attention_process_groups as attention_process_groups

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

@pytest.mark.parametrize(
    "cp_degree, tp_degree, expected_tp_mesh",
    # fmt: off
    [
        (8, 16, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]),
        (8, 32, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]),
    ],
    # fmt: on
)
def test_context_parallel_attention_process_groups(cp_degree, tp_degree, expected_tp_mesh):
    with mock_distributed(world_size=tp_degree):
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp_degree)
        initialize_model_parallel(tensor_model_parallel_size=tp_degree, skip_collective_init=True)

        neuron_config = NeuronConfig(tp_degree=tp_degree, cp_degree=cp_degree)
        attention_process_groups.init_context_parallel_attention_process_groups(InferenceConfig(neuron_config))

    tp_group = attention_process_groups.get_context_parallel_attention_tp_group()
    assert tp_group._mesh == expected_tp_mesh

    transposed_mesh = [[row[i] for row in tp_group._mesh] for i in range(len(tp_group._mesh[0]))]
    cp_group = attention_process_groups.get_context_parallel_attention_cp_group()

    assert transposed_mesh == cp_group._mesh

def test_uninitialized_process_groups():
    with pytest.raises(AssertionError):
        attention_process_groups.get_context_parallel_attention_cp_group()
        attention_process_groups.get_context_parallel_attention_tp_group()

