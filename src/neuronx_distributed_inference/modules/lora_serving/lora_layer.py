from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, mappings
from neuronx_distributed_inference.modules.kvcache.utils import fill_prefix, dynamic_update_slice


class BaseMultiLora(nn.Module):
    def get_checkpoint_shape(self):
        return self.weight_shape

    def get_checkpoint_shape_active(self):
        return self.weight_shape_active

    def get_weight_dtype(self):
        return self.dtype

    def _init_weight_parameter(self, shape):
        return nn.Parameter(torch.empty(*shape, dtype=self.get_weight_dtype()), requires_grad=False)

    def init_weights_parameter(self):
        self.weight = self._init_weight_parameter(self.weight_shape)
        self.weight_active = self._init_weight_parameter(self.weight_shape_active)
        self.updated_weight = None
        self.is_continuous_batching = False

    def set_weight_shapes(self, single_lora_shape, single_lora_shape_active=None):
        if single_lora_shape_active is None:
            single_lora_shape_active = single_lora_shape

        self.weight_shape = (self.max_loras, *single_lora_shape)
        self.weight_shape_active = (self.max_loras_active, *single_lora_shape_active)

    def _get_lora_weights(self, adapter_ids, gather_dim=-1):
        tensors = torch.index_select(self.weight, 0, adapter_ids)
        if gather_dim >= 0:
            tensors = mappings._gather_along_dim(tensors, gather_dim)
        return tensors

    def update_lora_tensor(self, adapter_ids, gather_dim=-1) -> torch.Tensor:
        tensors = self._get_lora_weights(adapter_ids, gather_dim)
        self.updated_weight = fill_prefix(self.weight_active, tensors)
        return self.updated_weight

    def update_lora_tensor_for_continuous_batching(self, adapter_ids, seq_ids, gather_dim=-1) -> torch.Tensor:
        assert adapter_ids.shape == seq_ids.shape
        self.is_continuous_batching = True
        tensors = self._get_lora_weights(adapter_ids, gather_dim)
        # following update_kv_by_layer_id() in KVCacheManager to update the LoRA weights in continuous batching
        indices = [seq_ids] + [torch.zeros(1, device=seq_ids.device) for _ in range(self.weight_active.dim() - 1)]
        indices = [t.squeeze().to(torch.int32) for t in indices]
        self.updated_weight = dynamic_update_slice(self.weight_active, tensors, indices)
        return self.updated_weight

    def get_weight(self, adapter_ids: torch.Tensor, is_context_encoding: bool = False) -> torch.Tensor:
        if is_context_encoding and self.is_continuous_batching:
            return self.updated_weight[adapter_ids]
        elif is_context_encoding:
            return self.updated_weight
        else:
            return self.weight_active

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        raise NotImplementedError

    def _einsum_forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.memory_transpose:
            return torch.einsum("bij,bjk->bik", x, weights)
        return torch.einsum("bij,bkj->bik", x, weights)


class MultiLoraLinear(BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        memory_transpose: bool = True,
    ) -> None:
        self.max_loras = max_loras
        self.max_loras_active = max_loras_active
        self.dtype = dtype
        self.memory_transpose = memory_transpose
        super().__init__()
        single_lora_shape = (input_size, output_size) if self.memory_transpose else (output_size, input_size)
        self.set_weight_shapes(single_lora_shape)
        self.init_weights_parameter()

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        weights = self.get_weight(adapter_ids, is_context_encoding)
        return self._einsum_forward(x, weights)


class MultiLoraConv2d(BaseMultiLora, nn.Conv2d):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        kernel_size,
        stride,
        padding,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_loras = max_loras
        self.max_loras_active = max_loras_active
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        super().__init__(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding=padding,
            bias=False,
            dtype=dtype,
        )
        single_lora_shape = (self.input_size, self.output_size, *self.kernel_size)
        self.set_weight_shapes(single_lora_shape)
        self.init_weights_parameter()

    def _forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, weight, None)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        ret = []
        weights = self.get_weight(adapter_ids, is_context_encoding)

        for i in range(adapter_ids.numel()):
            output = self._forward(x[i], weights[i])
            ret.append(output)
        return torch.stack(ret)


class MultiLoraEmbedding(BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        padding_idx: Optional[int],
        max_norm: Optional[float],
        norm_type: float,
        scale_grad_by_freq: bool,
        sparse: bool,
        dtype: torch.dtype = torch.float32,
        memory_transpose: bool = True,
    ) -> None:
        self.max_loras = max_loras
        self.max_loras_active = max_loras_active
        self.dtype = dtype
        self.memory_transpose = memory_transpose
        self.input_size = input_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        super().__init__()
        single_lora_shape = (input_size, output_size) if self.memory_transpose else (output_size, input_size)
        self.set_weight_shapes(single_lora_shape)
        self.init_weights_parameter()

    def _forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not self.memory_transpose:
            weight = weight.T
        return F.embedding(
            x,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        ret = []
        weights = self.get_weight(adapter_ids, is_context_encoding)

        for i in range(adapter_ids.numel()):
            output = self._forward(x[i], weights[i])
            ret.append(output)
        return torch.stack(ret)


class MultiLoraColumnParallelLinear(BaseMultiLora, ColumnParallelLinear):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        memory_transpose: bool = True,
        kv_replicate=None,
        **kwargs,
    ) -> None:
        self.max_loras = max_loras
        self.max_loras_active = max_loras_active
        self.dtype = dtype
        self.memory_transpose = memory_transpose
        self.kv_replicate = kv_replicate
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
            bias=False,
            gather_output=False,
            **kwargs,
        )

    def set_weight_and_bias_config(self) -> None:
        single_lora_shape = (self.input_size, self.output_size_per_partition) if self.memory_transpose else (self.output_size_per_partition, self.input_size)
        self.set_weight_shapes(single_lora_shape)
        self.init_weights_parameter()

        self.weight_partition_dim = 2 if self.memory_transpose else 1
        self.bias_shape = None

    def get_checkpoint_shape(self):
        if not self.memory_transpose:
            return (self.max_loras, self.output_size, self.input_size)
        else:
            return (self.max_loras, self.input_size, self.output_size)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        weights = self.get_weight(adapter_ids, is_context_encoding)
        return self._einsum_forward(x, weights)


class MultiLoraRowParallelLinear(BaseMultiLora, RowParallelLinear):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        memory_transpose: bool = True,
        **kwargs,
    ) -> None:
        self.max_loras = max_loras
        self.max_loras_active = max_loras_active
        self.dtype = dtype
        self.memory_transpose = memory_transpose
        super().__init__(
            input_size=input_size, output_size=output_size, dtype=dtype, bias=False, **kwargs
        )

    def set_weight_and_bias_config(self) -> None:
        single_lora_shape = (self.input_size_per_partition, self.output_size) if self.memory_transpose else (self.output_size, self.input_size_per_partition)
        self.set_weight_shapes(single_lora_shape)
        self.init_weights_parameter()

        self.weight_partition_dim = 1 if self.memory_transpose else 2
        self.bias_shape = None

    def get_checkpoint_shape(self):
        if not self.memory_transpose:
            return (self.max_loras, self.output_size, self.input_size)
        else:
            return (self.max_loras, self.input_size, self.output_size)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        weights = self.get_weight(adapter_ids, is_context_encoding)
        output_parallel = self._einsum_forward(x, weights)
        return mappings.reduce_from_tensor_model_parallel_region(output_parallel)


class MultiLoraColumnShardedLinear(MultiLoraColumnParallelLinear):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        memory_transpose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            max_loras=max_loras,
            max_loras_active=max_loras_active,
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
            memory_transpose=memory_transpose,
            **kwargs,
        )

    def set_weight_and_bias_config(self) -> None:
        single_lora_shape = (self.input_size, self.output_size_per_partition) if self.memory_transpose else (self.output_size_per_partition, self.input_size)
        single_lora_shape_active = (self.input_size, self.output_size) if self.memory_transpose else (self.output_size, self.input_size)
        self.set_weight_shapes(single_lora_shape, single_lora_shape_active)
        self.init_weights_parameter()

        self.weight_partition_dim = 2 if self.memory_transpose else 1
        self.weight_active_partition_dim = self.weight_partition_dim
        self.bias_shape = None

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        weights = self.get_weight(adapter_ids, is_context_encoding)
        return self._einsum_forward(x, weights)


class MultiLoraRowShardedLinear(MultiLoraRowParallelLinear):
    def __init__(
        self,
        max_loras: int,
        max_loras_active: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        memory_transpose: bool = True,
        lora_memory_reorder: bool = False,
        **kwargs,
    ) -> None:
        self.max_loras = max_loras
        self.max_loras_active = max_loras_active
        self.dtype = dtype
        self.memory_transpose = memory_transpose
        self.lora_memory_reorder = lora_memory_reorder
        super().__init__(
            max_loras=max_loras,
            max_loras_active=max_loras_active,
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
            memory_transpose=memory_transpose,
            **kwargs
        )

    def set_weight_and_bias_config(self) -> None:
        single_lora_shape = (self.input_size_per_partition, self.output_size) if self.memory_transpose else (self.output_size, self.input_size_per_partition)
        single_lora_shape_active = (self.input_size, self.output_size) if self.memory_transpose else (self.output_size, self.input_size)
        self.set_weight_shapes(single_lora_shape, single_lora_shape_active)
        self.init_weights_parameter()

        self.weight_partition_dim = 1 if self.memory_transpose else 2
        self.weight_active_partition_dim = self.weight_partition_dim
        self.bias_shape = None

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, is_context_encoding: bool) -> torch.Tensor:
        weights = self.get_weight(adapter_ids, is_context_encoding)
        return self._einsum_forward(x, weights)
