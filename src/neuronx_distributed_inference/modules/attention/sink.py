from neuronx_distributed.parallel_layers.layers import BaseParallelLinear
import torch
from typing import Optional, List
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes, divide


class LearnedSink(BaseParallelLinear):

    def __init__(
        self,
        learned_sinks_size,
        num_attention_heads: int,
        torch_dtype,
        tensor_model_parallel_size: Optional[int] = None,
        rank_ordering: List[int] = None,
    ):
        super().__init__()
        assert (
            learned_sinks_size == 1
        ), f"Learned sinks only supports learned_sinks_size == 1 ({learned_sinks_size})"
        self.tensor_model_parallel_size = (
            tensor_model_parallel_size
            if tensor_model_parallel_size is not None
            else get_tensor_model_parallel_size()
        )
        sink_size_per_partition = divide(num_attention_heads, tensor_model_parallel_size)
        self.sink = torch.nn.Parameter(
            torch.zeros(sink_size_per_partition, dtype=torch_dtype), requires_grad=False
        )
        print(f"learned_sink size = {self.sink.shape}")
        set_tensor_model_parallel_attributes(
            self.sink,
            is_parallel=True,
            dim=0,
            stride=1,
            num_partitions=self.tensor_model_parallel_size,
            rank_ordering=rank_ordering
        )

    def get_sink(self) -> torch.Tensor:
        return self.sink
