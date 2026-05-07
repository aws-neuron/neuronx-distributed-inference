import math
from typing import Optional, Tuple, Union, Any, Callable

from neuronx_distributed.parallel_layers.layers import (
    _as_tuple2,
    _initialize_affine_weight_neuron,
    _initialize_parameter_cpu,

    CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION,
    CONV_KERNEL_INPUT_CHANNEL_DIMENSION,
    conv2d_with_weight_grad_allreduce
    )
from neuronx_distributed.parallel_layers.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
)
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
from neuronx_distributed.parallel_layers.utils import (
    divide,
    get_padding_length,
    set_tensor_model_parallel_attributes,
)
import neuronx_distributed.trace.trace as nxd_tracing_utils
import torch
from torch.nn.parameter import Parameter


class BaseParallelConv(torch.nn.Module):


    def set_weight_shape(self) -> None:
        if self.partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION:
            if self.partition_pad:
                self.partition_pad_size = get_padding_length(self.out_channels, self.world_size)
                self.out_channels = self.out_channels + self.partition_pad_size

            self.channels_per_partition = divide(self.out_channels, self.world_size)
            self.weight_shape = [self.channels_per_partition, self.in_channels, *_as_tuple2(self.kernel_size)]
        elif self.partition_dim == CONV_KERNEL_INPUT_CHANNEL_DIMENSION:
            if self.partition_pad:
                self.partition_pad_size = get_padding_length(self.in_channels, self.world_size)
                self.in_channels = self.in_channels + self.partition_pad_size

            self.channels_per_partition = divide(self.in_channels, self.world_size)
            self.weight_shape = [self.out_channels, self.channels_per_partition, *_as_tuple2(self.kernel_size)]
        else:
            assert False, f"Unsupported partition dim: {self.partition_dim}"

    def set_bias_shape(self) -> None:
        if self.add_bias:
            self.bias_shape = (
                self.channels_per_partition
                if self.partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION
                else self.out_channels
            )
        else:
            self.bias_shape = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        groups: int,
        bias: bool,
        padding_mode: str,
        partition_dim: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        keep_master_params: bool = False,
        partition_pad: bool = False,
    ):
        if not all(d == 1 for d in _as_tuple2(dilation)):
            raise NotImplementedError(f"Non-1 dilation is not yet supported. Received: {dilation}")
        if groups != 1:
            raise NotImplementedError(f"Non-1 groups is not yet supported. Received: {groups}")
        if padding_mode != "zeros":
            raise NotImplementedError(f"Non-zeros padding is not yet supported. Received: {padding_mode}")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.partition_dim = partition_dim
        self.arg_init_method = init_method
        self.dtype = dtype
        self.device = device
        self.keep_master_params = keep_master_params
        self.partition_pad = partition_pad
        self.add_bias = bias
        self.world_size = get_tensor_model_parallel_size()

        self.set_weight_shape()
        self.set_bias_shape()

        # Get torch init device if device is not explicitly mentioned
        init_device = self.device
        self.weight = Parameter(torch.empty(*self.weight_shape, device=init_device, dtype=self.dtype))
        self.device = self.weight.device

        if self.device.type == "cpu":
            self.master_weight = _initialize_parameter_cpu(
                    self.weight,
                    partition_dim=partition_dim,
                    num_partitions=self.world_size,
                    init_method=self._init_weight,
                    return_master_param=self.keep_master_params,
                    param_dtype=self.dtype,
                    stride=1,
                )
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight,
                is_parallel=True,
                dim=partition_dim,
                stride=1,
                num_partitions=self.world_size,
            )
        else:
            assert device and device.type == "xla", "Currently only xla device type is supported"
            _initialize_affine_weight_neuron(
                self.weight,
                self._init_weight,
                partition_dim=partition_dim,
                num_partitions=self.world_size,
                stride=1,
            )

        if self.add_bias:
            # Bias is added before running the all-gather collective
            # If conv layer is sharded across output channels (partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION),
            # then the bias must be sharded
            # 1. We initialize the bias to an empty parameter tensor of shape (C_out,) or (C_out/TP,)
            self.bias = Parameter(torch.empty(self.bias_shape, dtype=dtype, device=device))

            # 2. Parameter initialization
            # These parallel layers are used for both training and inference. When training from scratch, weight
            # initialization must be carefully done, especially when distributed (e.g. ensure the same seed is used on every rank)
            # Such careful initialization is not needed when tracing (device.type == meta) or at inference
            if self.device.type == "cpu":
                if partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION:
                    self.master_bias = _initialize_parameter_cpu(
                        self.bias,
                        CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION,
                        num_partitions=self.world_size,
                        init_method=self._init_bias,
                        return_master_param=self.keep_master_params,
                        param_dtype=self.dtype,
                        stride=1,
                        )
                else:
                    self._init_bias(self.bias)
                    self.master_bias = self.bias if self.keep_master_params else None
            elif self.device.type == "meta":
                if partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION:
                    set_tensor_model_parallel_attributes(
                            self.bias,
                            is_parallel=True,
                            dim=self.partition_dim,
                            stride=1,
                            num_partitions=self.world_size,
                            )
                self.master_bias = self.bias if self.keep_master_params else None
            else:
                assert device and device.type == "xla", "Currently only xla device type is supported"
                if partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION:
                        set_tensor_model_parallel_attributes(
                                self.bias,
                                is_parallel=True,
                                dim=self.partition_dim,
                                stride=1,
                                num_partitions=self.world_size,
                                )
                self._init_bias(self.bias)
                self.master_bias = self.bias if self.keep_master_params else None
        else:
            self.register_parameter("bias", None)

        self._forward_impl = conv2d_with_weight_grad_allreduce

    def _init_weight(self, weight):
        if self.arg_init_method is None:
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            self.arg_init_method(weight)

    def _init_bias(self, bias):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias, -bound, bound)


class OutputChannelParallelConv2d(BaseParallelConv):
    """Conv2d layer with parallelism on its output channels

    The definition of a Conv2d layer can be found at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    This layer parallelizes the Conv2d along the output channel dimension

    .. note::
        Input is expected to be four dimensional, in order [N, C, H, W]

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels in the original Conv that is being parallelized. Parallelization is handled internally by this class
        kernel_size: Size of the kernel. Can be a single number for a square kernel or a tuple of two numbers
        stride: Stride of the convolution. Can be a single number for uniform H/W stride or a tuple of two numbers
        padding: Padding of the convolution. Can be a single number for uniform H/W padding or a tuple of two numbers
        bias: If true, add bias
        gather_output: If true, call all-gather on the output to assemble the partial outputs produced by each Neuron device into the full output, and make the full output available on all Neuron devices
        dtype: Datatype of the weights
        device: Device on which the weights should be initialized
        init_method: Method for initializing the weight
        keep_master_weight: If device="cpu", whether to keep the original ("master") weight the per-worker weights are split from
        partition_pad: Pad the output channel dimension if needed to make the output channel count divisible by the tensor model parallel size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        keep_master_weight: bool = False,
        partition_pad: bool = False,
    ):
        # Base class expects these all to be tuples so it can support N-dimensional convs
        kernel_size = _as_tuple2(kernel_size)
        stride = _as_tuple2(stride)
        padding = _as_tuple2(padding)
        dilation = _as_tuple2(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION,
            dtype,
            device,
            init_method,
            keep_master_weight,
            partition_pad,
        )
        self.kernel_size: Tuple[int, int]
        self.stride: Tuple[int, int]
        self.padding: Tuple[int, int]
        self.dilation: Tuple[int, int]

        self.allreduce_weight_grad = get_tensor_model_parallel_size() > 1
        self.gather_output = gather_output

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward of OutputChannelParallelConv2d

        Args:
            in_tensor: 4D tensor in order [N, C, H ,W]

        Returns:
            - output
        """

        if self.allreduce_weight_grad:
            input_parallel = in_tensor
        else:
            input_parallel = copy_to_tensor_model_parallel_region(in_tensor)

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            allreduce_weight_grad=self.allreduce_weight_grad,
        )

        # We intentionally did the bias add in _forward_impl to do less work overall
        # This way, each worker only has to do 1/world_size of the bias add
        if self.gather_output:
            # All-gather across the partitions
            output = gather_from_tensor_model_parallel_region_with_dim(output_parallel, gather_dim=1)
            if self.partition_pad and self.partition_pad_size > 0:
                output = torch.narrow(output, 1, 0, self.out_channels - self.partition_pad_size)
        else:
            output = output_parallel

        return output

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> None:
        if not self.partition_pad or self.partition_pad_size == 0:
            return
        if self.out_channels != model_state_dict[prefix].shape[0] + self.partition_pad_size:
            size = model_state_dict[prefix].shape[0]
            raise RuntimeError(
                f"State dict {prefix} is of an unexpected size {size} expected {size - self.partition_pad_size}"
            )
        model_state_dict[prefix] = torch.nn.functional.pad(
            model_state_dict[prefix], (0, 0, 0, 0, 0, 0, 0, self.partition_pad_size)
        )

nxd_tracing_utils.__SUPPORTED_SHARDED_MODULES = nxd_tracing_utils.__SUPPORTED_SHARDED_MODULES + (OutputChannelParallelConv2d, )
