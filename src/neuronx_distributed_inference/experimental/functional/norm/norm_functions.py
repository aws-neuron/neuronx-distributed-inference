import logging

import torch

logger = logging.getLogger("Neuron")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Functional implementation of RMSNorm.
    """
    # Convert to float for better numerical precision
    x_float = x.float()
    # Calculate the root mean square along the last dimension
    variance = x_float.pow(2).mean(-1, keepdim=True)
    # Normalize
    x_norm = x_float * torch.rsqrt(variance + eps)
    # Convert back to original dtype and apply weight
    return (x_norm.type_as(x)) * weight
