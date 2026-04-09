import pytest
import torch
from unittest.mock import patch, MagicMock

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
    NeuronQwen3VLTextModelWrapper,
)

# Index of rotary_position_ids in the positional args tuple
RPI_INDEX = NeuronQwen3VLTextModelWrapper._ROTARY_POSITION_IDS_INDEX  # 21

# Total number of args for Qwen3 VL (indices 0-24)
NUM_ARGS = 25


def _build_args(rotary_position_ids, batch_size=1):
    """Build a minimal args tuple mimicking the TKG decode path."""
    args = []
    for i in range(NUM_ARGS):
        if i == RPI_INDEX:
            args.append(rotary_position_ids)
        elif i == 0:  # input_ids: [batch, 1]
            args.append(torch.zeros(batch_size, 1, dtype=torch.int32))
        elif i == 1:  # attention_mask: [batch, seq]
            args.append(torch.zeros(batch_size, 128, dtype=torch.int32))
        elif i == 2:  # position_ids: [batch, 1]
            args.append(torch.zeros(batch_size, 1, dtype=torch.int32))
        elif i == 3:  # seq_ids: [batch]
            args.append(torch.arange(batch_size, dtype=torch.int32))
        elif i == 4:  # sampling_params: [batch, 3]
            args.append(torch.zeros(batch_size, 3, dtype=torch.float32))
        else:
            args.append(torch.empty(0))
    return tuple(args)


@pytest.fixture
def make_wrapper():
    """Factory fixture: returns a callable that creates a mocked wrapper."""

    def _make(compiled_batch_size):
        wrapper = object.__new__(NeuronQwen3VLTextModelWrapper)
        wrapper.neuron_config = MagicMock()
        wrapper.neuron_config.batch_size = compiled_batch_size
        wrapper.neuron_config.max_batch_size = compiled_batch_size
        wrapper.is_prefix_caching = False
        wrapper.async_mode = False
        return wrapper

    return _make


@pytest.fixture
def run_forward_with_pad():
    """Fixture: runs _forward_with_pad with mocked super() and returns the captured rpi."""

    def _run(wrapper, rotary_position_ids, batch_size=1):
        args = _build_args(rotary_position_ids, batch_size=batch_size)
        captured = {}

        def mock_super_forward(self_arg, *fwd_args):
            captured["rpi"] = fwd_args[RPI_INDEX]
            return torch.zeros(batch_size, 1)

        with patch.object(
            NeuronQwen3VLTextModelWrapper.__bases__[0],
            "_forward_with_pad",
            mock_super_forward,
        ):
            wrapper._forward_with_pad(*args)

        return captured["rpi"]

    return _run


class TestForwardWithPad:
    """Tests for NeuronQwen3VLTextModelWrapper._forward_with_pad."""

    def test_dim0_reconstruction(self, make_wrapper, run_forward_with_pad):
        """[3,1,1] sliced to [1,1,1] should be reconstructed back to [3,1,1]."""
        wrapper = make_wrapper(compiled_batch_size=1)
        sliced_rpi = torch.tensor([[[42]], [[42]], [[42]]])[0:1]  # [1,1,1]

        result = run_forward_with_pad(wrapper, sliced_rpi, batch_size=1)

        assert result.shape == (3, 1, 1), f"Expected [3,1,1], got {result.shape}"
        assert (result == 42).all(), f"Expected all 42, got {result}"

    def test_batch_padding(self, make_wrapper, run_forward_with_pad):
        """[3,1,1] should be padded to [3,4,1] when compiled batch=4."""
        wrapper = make_wrapper(compiled_batch_size=4)
        rpi = torch.tensor([[[10]], [[20]], [[30]]])  # [3,1,1]

        result = run_forward_with_pad(wrapper, rpi, batch_size=1)

        assert result.shape == (3, 4, 1), f"Expected [3,4,1], got {result.shape}"
        for b in range(4):
            assert result[0, b, 0] == 10
            assert result[1, b, 0] == 20
            assert result[2, b, 0] == 30

    def test_combined_reconstruct_and_pad(self, make_wrapper, run_forward_with_pad):
        """[3,1,1] sliced to [1,1,1] with compiled batch=2 -> [3,2,1]."""
        wrapper = make_wrapper(compiled_batch_size=2)
        sliced_rpi = torch.tensor([[[7]], [[7]], [[7]]])[0:1]  # [1,1,1]

        result = run_forward_with_pad(wrapper, sliced_rpi, batch_size=1)

        assert result.shape == (3, 2, 1), f"Expected [3,2,1], got {result.shape}"
        assert (result == 7).all(), f"Expected all 7, got {result}"

    def test_rpi_tkg_compiled_bs4_runtime_bs2(self, make_wrapper, run_forward_with_pad):
        """TKG decode: compiled_bs=4, runtime_bs=2, seq_len=1.

        During TKG, each batch line generates 1 token at a time (seq_len=1).
        Different batch lines can be at different decode positions (e.g.,
        batch_line_0 at step 10, batch_line_1 at step 25).
        Parent slices rpi[0:2] → [2,2,1], corrupting MRoPE dim from 3→2.
        Should reconstruct [3,2,1] then pad to [3,4,1].
        """
        wrapper = make_wrapper(compiled_batch_size=4)
        # TKG decode: seq_len=1, all 3 MRoPE components identical,
        # but different positions per batch line (different context lengths)
        original_rpi = torch.tensor([
            [[10], [25]],
            [[10], [25]],
            [[10], [25]],
        ])  # [3, 2, 1]
        sliced_rpi = original_rpi[0:2]  # Simulates parent's dim-0 slice → [2, 2, 1]

        result = run_forward_with_pad(wrapper, sliced_rpi, batch_size=2)

        assert result.shape == (3, 4, 1)
        for mrope_dim in range(3):
            assert result[mrope_dim, 0, 0] == 10
            assert result[mrope_dim, 1, 0] == 25

    def test_empty_passthrough(self, make_wrapper, run_forward_with_pad):
        """torch.empty(0) should pass through unchanged."""
        wrapper = make_wrapper(compiled_batch_size=2)

        result = run_forward_with_pad(wrapper, torch.empty(0), batch_size=1)

        assert result.dim() == 1, f"Expected 1D empty tensor, got dim={result.dim()}"
        assert result.numel() == 0, f"Expected empty tensor, got numel={result.numel()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
