import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    mappings,
)


class BaseMultiLora(nn.Module):
    def __init__(self):
        super().__init__()

    def get_checkpoint_shape(self):
        raise NotImplementedError

    def get_weight_dtype(self):
        return self.dtype

    def get_weight(self, adapter_ids) -> torch.Tensor:
        if adapter_ids is None:
            adapter_ids = 0
        return self.weight[adapter_ids].squeeze()


class MultiLoraLinear(BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_loras = max_loras
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.weight_shape = (self.max_loras, self.output_size, self.input_size)
        self.weight = nn.Parameter(torch.empty(*self.weight_shape, dtype=self.dtype), requires_grad=False)

    def get_checkpoint_shape(self):
        return self.weight_shape

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return F.linear(x, weight)


class MultiLoraConv2d(nn.Conv2d, BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        kernel_size,
        stride,
        padding,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_loras = max_loras
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
        self.weight = nn.Parameter(
            torch.empty(
                self.max_loras,
                self.input_size,
                self.output_size,
                *self.kernel_size,
                dtype=self.dtype),
            requires_grad=False
        )

    def get_checkpoint_shape(self):
        return self.weight.size()

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return self._conv_forward(x, weight, None)


class MultiLoraEmbedding(BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        padding_idx: Optional[int],
        max_norm: Optional[float],
        norm_type: float,
        scale_grad_by_freq: bool,
        sparse: bool,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_loras = max_loras
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        super().__init__()
        self.weight_shape = (self.max_loras, self.output_size, self.input_size)
        self.weight = nn.Parameter(torch.empty(*self.weight_shape, dtype=self.dtype), requires_grad=False)

    def get_checkpoint_shape(self):
        return self.weight_shape

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return F.embedding(
            x,
            weight.T,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )



class MultiLoraColumnParallelLinear(ColumnParallelLinear, BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        self.max_loras = max_loras
        self.dtype = dtype
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
            bias=False,
            gather_output=False,
            **kwargs
        )

    def set_weight_and_bias_config(self) -> None:
        self.weight_shape = (
            self.max_loras,
            self.output_size_per_partition,
            self.input_size,
        )
        self.weight_partition_dim = 1
        self.bias_shape = None

    def get_checkpoint_shape(self):
        return (self.max_loras, self.output_size, self.input_size)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return self._forward_impl(
            input=x,
            weight=weight,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            autograd_func_class=self.autograd_func_class,
        )


class MultiLoraRowParallelLinear(RowParallelLinear, BaseMultiLora):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        self.max_loras = max_loras
        self.dtype = dtype
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            dtype=dtype,
            bias=False,
            **kwargs
        )

    def set_weight_and_bias_config(self) -> None:
        self.weight_shape = (
            self.max_loras,
            self.output_size,
            self.input_size_per_partition,
        )
        self.weight_partition_dim = 2
        self.bias_shape = None

    def get_checkpoint_shape(self):
        return (self.max_loras, self.output_size, self.input_size)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        output_parallel = self._forward_impl(
            input=x,
            weight=weight,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            autograd_func_class=self.autograd_func_class,
        )
        return mappings.reduce_from_tensor_model_parallel_region(output_parallel)
