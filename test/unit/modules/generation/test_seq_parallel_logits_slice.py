import pytest

import torch
import torch.distributed
from torch_neuronx.utils import get_platform_target
import torch_xla.core.xla_model as xm

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.generation.seq_parallel_logits_slice import seq_parallel_slice_last_token
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import _reduce_scatter_along_dim


"""
The follwoing is the testing methodology for a sequence parallel function that processes hidden states (B,S,H) and position_ids (B,S), slicing the last token along the sequence dimension.

Test Configurations
    1. Without tile_cc
        In this configuration, hidden states are randomly generated with shape (B,S,H) and then split along the sequence dimension during the forward pass.

        Example:

            Original hidden states: [0,1,2,3,4,5,6,7,8,9,10,11]
            Tensor Parallelism (TP) = 2
            After sharding:
                Core 0: [0,1,2,3,4,5]
                Core 1: [6,7,8,9,10,11]

    2. With tile_cc
        In this configuration, hidden states are first tiled and then sharded based on the tensor parallelism degree.

        Example:

            Original hidden states: [0,1,2,3,4,5,6,7,8,9,10,11]
            Tensor Parallelism (TP) = 2
            Tiling factor = 3
            After tiling:
                Tile 0: [0,1,2,3]
                Tile 1: [4,5,6,7]
                Tile 2: [8,9,10,11]
            After sharding:
                Core 0: [0,1,4,5,8,9]
                Core 1: [2,3,6,7,10,11]

    3. With chunked prefill
        In this configuration, the hidden states remain same as above both cases but we pass num_queries arg to the sequence
        parallel optimization function which handles the chunked prefill case.

Position IDs
For position IDs, a random_position_id function generates position IDs of shape (B,S).
"""

torch.manual_seed(42)

class SequenceParallelTestModule(torch.nn.Module):
    def __init__(self, 
                 sequence_length=None, 
                 hidden_size=None, 
                 batch_size=1,
                 tp_degree=1,
                 num_queries=None,
                 neuron_config=None,
                 config=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.sequence_dimension = 1
        self.batch_size = batch_size
        self.tp_degree = tp_degree
        self.num_queries = num_queries
        self.neuron_config = neuron_config
        self.config = config

    def forward(self, hidden_states, position_ids):
        if self.neuron_config.tile_cc:
            input_ids_tiled = hidden_states.reshape(self.batch_size, 
                                                    self.neuron_config.cc_pipeline_tiling_factor, 
                                                    self.sequence_length//self.neuron_config.cc_pipeline_tiling_factor, 
                                                    self.hidden_size)
            input_ids_tiled = input_ids_tiled.reshape(self.batch_size, 
                                                      self.neuron_config.cc_pipeline_tiling_factor, 
                                                      self.tp_degree, 
                                                      self.sequence_length//self.neuron_config.cc_pipeline_tiling_factor//self.tp_degree, 
                                                      self.hidden_size)
            input_ids_tiled = input_ids_tiled.permute(0,2,1,3,4)
            input_ids_distributed = input_ids_tiled.reshape(self.batch_size, 
                                                           self.tp_degree, 
                                                           self.sequence_length//self.tp_degree,
                                                           self.hidden_size)
            hidden_states = input_ids_distributed.reshape(self.batch_size, self.sequence_length, self.hidden_size)
        
        if self.neuron_config.is_chunked_prefill:
            self.num_queries = self.num_queries.to("xla")

        hidden_states = _reduce_scatter_along_dim(
                hidden_states,
                self.sequence_dimension,
                xm.REDUCE_MAX,
                process_group=parallel_state.get_tensor_model_parallel_group(as_list=False),
            )
        out = seq_parallel_slice_last_token(hidden_states, 
                                            position_ids, 
                                            self.sequence_dimension,
                                            self.batch_size,
                                            self.hidden_size,
                                            self.num_queries,
                                            self.neuron_config,
                                            self.config
                                            )
        return out
    
    @torch.inference_mode()
    def cpu_forward(self, hidden_states, position_ids):
        if self.neuron_config.is_chunked_prefill:
            index = torch.cumsum(self.num_queries.reshape(1, -1).float(), dim=1).int() - 1
            index = index[:,:, None].reshape(1, -1, 1)
            index_expanded = index.expand(self.batch_size, -1, self.hidden_size).to(torch.int64)
        else:
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index_expanded = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
        print(index_expanded)
        out = torch.gather(hidden_states, dim=1, index=index_expanded)
        return out

def random_position_ids(batch_size, seq_length):
    """
    Generates a random position ID tensor of shape (batch_size, seq_length) with the following properties:
    
    - Each sequence starts from position ID `1` and increases sequentially.
    - A random stopping point is chosen for each sequence independently.
    - Positions beyond the stopping point are filled with `-1`.
    """
    torch.manual_seed(torch.randint(0, 10000, (1,)))  # Random seed for variability
    stop_positions = torch.randint(1, seq_length + 1, (batch_size,))
    pos_ids = torch.full((batch_size, seq_length), -1, dtype=torch.int32)
    
    for i in range(batch_size):
        pos_ids[i, :stop_positions[i]] = torch.arange(1, stop_positions[i] + 1)
    return pos_ids


@pytest.mark.parametrize(
    "batch_size, sequence_length, hidden_size, tp_degree, tile_cc, cc_pipeline_tiling_factor, chunked_prefill, num_queries",
    [
        (1, 1024, 256, 2, False, -1, False, -1),
        (4, 1024, 128, 2, False, -1, False, -1),
        (1, 1024, 256, 2, True, 2, False, -1),
        (4, 1024, 256, 2, True, 4, False, -1),
        (1, 1024, 256, 2, False, -1, True, 1),
        (4, 1024, 256, 2, False, -1, True, 4),
    ],
)
def test_seq_parallel_slice(batch_size, sequence_length, hidden_size, tp_degree, tile_cc, cc_pipeline_tiling_factor, chunked_prefill, num_queries):
    hardware = get_platform_target()
    if hardware == "trn1" and tp_degree == 64:
        pytest.skip("Not supported in trn1")
    if hardware == "trn2" and tp_degree == 32:
        pytest.skip("Not supported in trn2")

    # dummy config dict
    config_dict = {
        "tp_degree": tp_degree,
        "torch_dtype": torch.float32,
        "batch_size": batch_size,
        "sequence_parallel_enabled": True,
        "tile_cc": False
    }
    neuron_config = NeuronConfig(**config_dict)
    neuron_config.tile_cc = tile_cc
    neuron_config.is_chunked_prefill = chunked_prefill
    if tile_cc:
        neuron_config.cc_pipeline_tiling_factor = cc_pipeline_tiling_factor
    config = InferenceConfig(neuron_config=neuron_config, **config_dict)

    input_shape = (batch_size, sequence_length, hidden_size)
    hidden_states = torch.randn(input_shape, dtype=torch.float32)

    position_ids = random_position_ids(batch_size, sequence_length)

    if chunked_prefill:
        num_queries = torch.randint(3,10,(num_queries,), dtype=torch.int32)

    module_cpu = SequenceParallelTestModule(batch_size=batch_size,
                                           sequence_length=sequence_length,
                                           hidden_size=hidden_size,
                                           tp_degree=tp_degree,
                                           num_queries=num_queries,
                                           neuron_config=neuron_config,
                                           config=config)

    example_inputs =[(hidden_states, position_ids,)]
    module_neuron = build_module(SequenceParallelTestModule,
                                example_inputs,
                                tp_degree = tp_degree,
                                module_init_kwargs={
                                    "batch_size":batch_size,
                                    "sequence_length":sequence_length,
                                    "hidden_size":hidden_size,
                                    "tp_degree":tp_degree,
                                    "num_queries":num_queries,
                                    "neuron_config":neuron_config,
                                    "config":config
                                    }
                                )

    validate_accuracy(module_neuron, example_inputs, cpu_callable=module_cpu.cpu_forward)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
