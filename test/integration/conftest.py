import pytest
import torch

@pytest.fixture(autouse=True)
def set_constant_seed():
    """
    Sets a constant seed before each test runs to ensure deterministic behavior between runs.

    Without a constant seed, the random weights differ between runs, which can result in logit
    validation failing due to varying precision loss accumulation from one run to the next.
    Therefore, a constant seed improves test stability.
    """
    torch.manual_seed(0)
