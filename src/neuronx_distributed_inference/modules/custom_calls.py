import math
import torch
from torch import nn, ones
from torch_neuronx.xla_impl.ops import RmsNorm, nki_jit

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

try:
    from neuronxcc.nki._private_kernels.cumsum import cumsum as nki_cumsum
except ImportError:
    from neuronxcc.nki.kernels.cumsum import cumsum as nki_cumsum


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-6):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = None
        if hidden_size is not None:
            self.weight = nn.Parameter(ones(hidden_size))  # specify hidden size
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if self.hidden_size is None and self.weight is None:
            self.weight = nn.Parameter(
                ones(
                    hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
                )
            )
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


@nki.jit
def rmsnorm_kernel(x, w, axis, n, eps):
    bs, seq_len, D = x.shape
    assert D == w.shape[0]

    out_tensor = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(D)[None, :]
    w_tile = nl.load(w.reshape((1, D))[iw, iy])

    for b in nl.static_range(bs):
        for i in nl.affine_range(math.ceil(seq_len / 128)):
            x_tile = nl.load(x[b][i * 128 + ix, iy], mask=(i * 128 + ix < seq_len))
            rmsnorm_result = nl.rms_norm(
                x_tile,
                w_tile,
                axis,
                n,
                eps,
                mask=(i * 128 + ix < seq_len),
            )
            nl.store(
                out_tensor[b][i * 128 + ix, iy], value=rmsnorm_result, mask=(i * 128 + ix < seq_len)
            )

    return out_tensor
