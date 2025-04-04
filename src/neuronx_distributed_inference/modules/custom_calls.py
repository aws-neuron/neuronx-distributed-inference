import torch
from torch import nn, ones
from torch_neuronx.xla_impl.ops import RmsNorm, nki_jit

try:
    from neuronxcc.nki._private_kernels.cumsum import cumsum as nki_cumsum
except ImportError:
    from neuronxcc.nki.kernels.cumsum import cumsum as nki_cumsum


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(torch.float32)
        result = RmsNorm.apply(
            hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1
        )

        return result.to(original_dtype)


# Cumsum
custom_cumsum = nki_jit()(nki_cumsum)


def neuron_cumsum(input):
    """
    NKI implementation of cumsum

    Currently it
    1. only accumulates on the last dim
    2. only works with floating dtype
    """
    output = torch.zeros_like(input)
    return custom_cumsum(input, output, axis=1)
