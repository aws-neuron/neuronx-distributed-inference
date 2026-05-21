import torch
import torch.nn as nn

from .config import LoraServingConfig
from .lora_layer import (
    MultiLoraColumnParallelLinear,
    MultiLoraColumnShardedLinear,
    MultiLoraConv2d,
    MultiLoraEmbedding,
    MultiLoraLinear,
    MultiLoraRowParallelLinear,
    MultiLoraRowShardedLinear,
)


def is_lora_module(module):
    return isinstance(module, MultiLoraModule)


class MultiLoraModule(nn.Module):
    def __init__(self, base_layer: nn.Module, lora_config: LoraServingConfig) -> None:
        if lora_config.max_lora_rank <= 0:
            raise ValueError(
                f"`lora_rank` should be a positive integer value but the value passed is {lora_config.lora_rank}"
            )

        super().__init__()
        self.lora_max_rank = lora_config.max_lora_rank
        self.max_loras = lora_config.max_loras
        self.max_loras_active = lora_config.batch_size
        self.base_layer = base_layer
        self.lora_dtype = base_layer.weight.dtype
        self.lora_config = lora_config
        self.skip_dtype_convert = False
        self.lora_memory_transpose = lora_config.lora_memory_transpose
        self.shard_linear_layer = lora_config.lora_shard_linear_layer
        self.is_context_encoding = lora_config.is_context_encoding

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # ColumnParallelLinear, RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "num_embeddings") and hasattr(base_layer, "embedding_dim"):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.lora_A, self.lora_B = None, None
        self.create_lora()

    def create_lora(self):
        r"""
        Create the corresponding LoraAdapter according to its module type, such as nn.Linear and nn.Embedding.
        """
        raise NotImplementedError

    def get_base_layer(self) -> nn.Module:
        return self.base_layer

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        base_layer = self.get_base_layer()
        result = base_layer(x, *args, **kwargs)
        result = result + self.lora_B(self.lora_A(x, adapter_ids, self.is_context_encoding), adapter_ids, self.is_context_encoding)
        if not self.skip_dtype_convert:
            result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class MultiLoraModuleLinear(MultiLoraModule):
    def create_lora(self):
        self.lora_A = MultiLoraLinear(
            self.max_loras,
            self.max_loras_active,
            self.in_features,
            self.lora_max_rank,
            self.lora_dtype,
            self.lora_memory_transpose,
        )
        self.lora_B = MultiLoraLinear(
            self.max_loras,
            self.max_loras_active,
            self.lora_max_rank,
            self.out_features,
            self.lora_dtype,
            self.lora_memory_transpose,
        )


class MultiLoraModuleConv2d(MultiLoraModule):
    def create_lora(self):
        base_layer = self.get_base_layer()

        self.lora_A = MultiLoraConv2d(
            self.max_loras,
            self.max_loras_active,
            self.in_features,
            self.lora_max_rank,
            base_layer.kernel_size,
            base_layer.stride,
            base_layer.padding,
            self.lora_dtype,
        )
        self.lora_B = MultiLoraConv2d(
            self.max_loras,
            self.max_loras_active,
            self.lora_max_rank,
            self.out_features,
            (1, 1),
            (1, 1),
            0,
            self.lora_dtype,
        )


class MultiLoraModuleEmbedding(MultiLoraModule):
    def create_lora(self):
        base_layer = self.get_base_layer()
        self.lora_A = MultiLoraEmbedding(
            self.max_loras,
            self.max_loras_active,
            self.in_features,
            self.lora_max_rank,
            base_layer.padding_idx,
            base_layer.max_norm,
            base_layer.norm_type,
            base_layer.scale_grad_by_freq,
            base_layer.sparse,
            self.lora_dtype,
            self.lora_memory_transpose,
        )
        self.lora_B = MultiLoraLinear(
            self.max_loras,
            self.max_loras_active,
            self.lora_max_rank,
            self.out_features,
            self.lora_dtype,
            self.lora_memory_transpose,
        )
        self.skip_dtype_convert = True


class MultiLoraModuleColumnParallelLinear(MultiLoraModule):
    def __init__(
        self, base_layer: nn.Module, lora_config: LoraServingConfig, kv_replicate=None
    ) -> None:
        self.kv_replicate = kv_replicate
        super().__init__(base_layer, lora_config)

    def create_lora(self):
        multi_lora_linear = (
            MultiLoraRowShardedLinear if self.shard_linear_layer else MultiLoraLinear
        )
        self.lora_A = multi_lora_linear(
            self.max_loras,
            self.max_loras_active,
            self.in_features,
            self.lora_max_rank,
            self.lora_dtype,
            self.lora_memory_transpose,
        )
        self.lora_B = MultiLoraColumnParallelLinear(
            self.max_loras,
            self.max_loras_active,
            self.lora_max_rank,
            self.out_features,
            self.lora_dtype,
            memory_transpose=self.lora_memory_transpose,
            kv_replicate=self.kv_replicate,
        )


class MultiLoraModuleRowParallelLinear(MultiLoraModule):
    def create_lora(self):
        self.lora_A = MultiLoraRowParallelLinear(
            self.max_loras,
            self.max_loras_active,
            self.in_features,
            self.lora_max_rank,
            self.lora_dtype,
            self.lora_memory_transpose,
        )
        multi_lora_linear = (
            MultiLoraColumnShardedLinear if self.shard_linear_layer else MultiLoraLinear
        )
        self.lora_B = multi_lora_linear(
            self.max_loras,
            self.max_loras_active,
            self.lora_max_rank,
            self.out_features,
            self.lora_dtype,
            self.lora_memory_transpose,
        )
