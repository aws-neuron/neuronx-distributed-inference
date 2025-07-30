import pytest
import numpy as np
import torch
from torch_xla.core import xla_model as xm
from neuronx_distributed.kernels.topk import topk_rotated
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from torch_neuronx.utils import get_platform_target


@pytest.mark.parametrize('batch_size, seq_len, vocab_size, topk, sorted', [
    (1, 1, 2004, 8, True), # base case for extra configs
    (1, 1, 2004, 48, True),
    (1, 3, 3157, 256, True), # padding case of vocab size 
    # llama 3 configs
    (1, 1, 2004, 256, True), 
    (2, 1, 2004, 256, True), 
    (4, 1, 2004, 256, True), 
    (8, 1, 2004, 256, True), 
    (1, 5, 2004, 256, True),  # spec length 5

    # llama 4 configs
    (1, 1, 3168, 256, True), 
    (2, 1, 3168, 256, True), 
    (4, 1, 3168, 256, True), 
    (8, 1, 3168, 256, True), 

    # coverage for sorted=False
    (1, 5, 2004, 256, False),  # spec length 5
    (1, 1, 3168, 128, False),
    # 2D case , pass seq_len = None
    (4, None, 3168, 256, True),
])
def test_topk_kernel(batch_size, seq_len, vocab_size, topk, sorted):
    np.random.seed(0)
    if seq_len:
        n_programs =  1 if batch_size * seq_len == 1 else 2
    else:
        n_programs =  1 if batch_size  == 1 else 2

    def nc(x): return x if get_platform_target() == "trn1" else nl.nc(x)
    if seq_len:
        logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
    else:
        logits = np.random.randn(batch_size, vocab_size).astype(np.float32)

    out = topk_rotated[(nc(n_programs),)](torch.tensor(logits).to(device=xm.xla_device()), k=topk, sorted=sorted)
    torch_golden = torch.topk(torch.tensor(logits, dtype=torch.float32), k=topk)

    def maybe_cpu(x): return x.cpu() if hasattr(x, 'device') else x
    actual, idx = [maybe_cpu(x) for x in out]
    dim = len(logits.shape)-1
    if sorted:
        np.testing.assert_allclose(idx,torch_golden[1])
        np.testing.assert_allclose(actual=actual, desired=torch_golden[0])
    else:
        np.testing.assert_allclose(np.sort(idx, axis=dim), torch.sort(torch_golden[1], dim=dim).values)
        sorted_actual = torch.sort(actual, dim=dim).values
        sorted_expected = torch.sort(torch_golden[0], dim=dim).values
        np.testing.assert_allclose(actual=sorted_actual, desired=sorted_expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])