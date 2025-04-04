import unittest
from typing import List
from unittest.mock import patch

import torch

from neuronx_distributed_inference.modules.async_execution import AsyncTensorWrapper


class TestAsyncTensorWrapper(unittest.TestCase):
    # NOTE: we mock is_ranked_io because it checks if the device is on Neuron.
    # The test cases mock the expected return value of this particular check.

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_initialization(self, mock_is_ranked_io):
        TP_DEGREE = 32
        mock_ranked_tensor = [
            [torch.rand(2, 2)] for _ in range(TP_DEGREE)
        ]  # usually this is on Neuron Device, but we mock this

        mock_is_ranked_io.return_value = True
        _ = AsyncTensorWrapper(mock_ranked_tensor, batch_padded=False, on_cpu=False)

        mock_is_ranked_io.return_value = False
        _ = AsyncTensorWrapper(
            mock_ranked_tensor[0],  # should be initialized with list of tensors if on_cpu=True
            batch_padded=False,
            on_cpu=True,
        )

        did_fail_assertion = False
        try:
            mock_is_ranked_io.return_value = False
            _ = AsyncTensorWrapper(mock_ranked_tensor[0], batch_padded=False, on_cpu=False)
        except AssertionError:
            did_fail_assertion = True
        finally:
            assert (
                did_fail_assertion
            ), "It should not be possible to initialize an AsyncTensorWrapper object with a CPU tensor with on_cpu=False"

        did_fail_assertion = False
        try:
            mock_is_ranked_io.return_value = True
            _ = AsyncTensorWrapper(mock_ranked_tensor, batch_padded=False, on_cpu=True)
        except AssertionError:
            did_fail_assertion = True
        finally:
            assert (
                did_fail_assertion
            ), "It should not be possible to initialize an AsyncTensorWrapper object with a ranked tensor on Neuron with on_cpu=True"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_get_ranked_tensor(self, mock_is_ranked_io):
        TP_DEGREE = 32
        mock_ranked_tensor = [
            [torch.rand(2, 2)] for _ in range(TP_DEGREE)
        ]  # usually this is on Neuron Device, but we mock this

        mock_is_ranked_io.return_value = True
        async_tensor_wrapper_not_on_cpu = AsyncTensorWrapper(
            mock_ranked_tensor, batch_padded=False, on_cpu=False
        )

        mock_is_ranked_io.return_value = False
        async_tensor_wrapper_on_cpu = AsyncTensorWrapper(
            mock_ranked_tensor[0],  # should be initialized with list of tensors if on_cpu=True
            batch_padded=False,
            on_cpu=True,
        )

        # check if tensor returned is equal to what we passed in
        retrieved_rank_tensor = async_tensor_wrapper_not_on_cpu.get_ranked_tensor()
        assert isinstance(retrieved_rank_tensor, list) and (
            len(retrieved_rank_tensor) == len(mock_ranked_tensor) == TP_DEGREE
        )

        did_fail_assertion = False
        try:
            retrieved_rank_tensor = async_tensor_wrapper_on_cpu.get_ranked_tensor()
        except AssertionError:
            did_fail_assertion = True
        finally:
            assert (
                did_fail_assertion
            ), "It shouldn't be possible to get a ranked tensor when AsyncTensorWrapper was initialized with on_cpu=True"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_sync_async_result_to_cpu_with_ranked_tensor_simple(self, mock_is_ranked_io):
        TP_DEGREE = 32
        BATCH_SIZE = 2
        mock_ranked_tensor = [
            [torch.rand(2, 2)] for _ in range(TP_DEGREE)
        ]  # usually this is on Neuron Device, but we mock this

        mock_is_ranked_io.return_value = True
        async_tensor_wrapper = AsyncTensorWrapper(
            mock_ranked_tensor, batch_padded=False, on_cpu=False
        )

        mock_seq_ids = torch.arange(0, BATCH_SIZE).reshape(BATCH_SIZE, 1)

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids
        )

        assert torch.equal(
            synced_tensor[0], mock_ranked_tensor[0][0]
        ), "synced tensor does not equal the 0th rank tensor from the ranked tensor"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_sync_async_result_to_cpu_with_ranked_tensor_batch_padded(self, mock_is_ranked_io):
        TP_DEGREE = 32
        BATCH_SIZE = 2
        mock_ranked_tensor = [
            [torch.rand(BATCH_SIZE, 2)] for _ in range(TP_DEGREE)
        ]  # usually this is on Neuron Device, but we mock this

        mock_is_ranked_io.return_value = True
        async_tensor_wrapper = AsyncTensorWrapper(
            mock_ranked_tensor, batch_padded=True, on_cpu=False
        )

        mock_seq_ids_1 = torch.tensor([[0]], dtype=torch.int32)
        mock_seq_ids_2 = torch.tensor([[1]], dtype=torch.int32)

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids_1
        )
        assert torch.equal(
            synced_tensor[0], mock_ranked_tensor[0][0][mock_seq_ids_1.squeeze(0)]
        ), "synced tensor does not equal the 0th seq_id from the 0th rank tensor from the ranked tensor"

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids_2
        )
        assert torch.equal(
            synced_tensor[0], mock_ranked_tensor[0][0][mock_seq_ids_2.squeeze(0)]
        ), "synced tensor does not equal the 1st seq_id from the 0th rank tensor from the ranked tensor"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_sync_async_result_to_cpu_with_on_cpu_simple(self, mock_is_ranked_io):
        BATCH_SIZE = 2
        REQUEST_BATCH_SIZE = 1
        output_logits = [[torch.rand(REQUEST_BATCH_SIZE, 2)], [torch.rand(REQUEST_BATCH_SIZE, 2)]]

        mock_is_ranked_io.return_value = False
        async_tensor_wrapper = AsyncTensorWrapper(output_logits, batch_padded=False, on_cpu=True)

        mock_seq_ids = torch.arange(0, BATCH_SIZE).reshape(BATCH_SIZE, 1)
        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids
        )

        assert synced_tensor.shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            2,
        ), f"Tensor shape does not match expected concatenated shape of (2, 2), got {synced_tensor.shape}"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_sync_async_result_to_cpu_with_on_cpu_batch_padded(self, mock_is_ranked_io):
        BATCH_SIZE = 2
        REQUEST_BATCH_SIZE = 1
        output_logits = [
            [torch.rand(BATCH_SIZE, 2)],
        ]

        mock_is_ranked_io.return_value = False
        async_tensor_wrapper = AsyncTensorWrapper(output_logits, batch_padded=True, on_cpu=True)

        mock_seq_ids_1 = torch.tensor([[0]], dtype=torch.int32)
        mock_seq_ids_2 = torch.tensor([[1]], dtype=torch.int32)

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids_1
        )
        assert synced_tensor.shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            2,
        ), f"Tensor shape does not match expected shape of (1, 2), got {synced_tensor.shape}"

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids_2
        )
        assert synced_tensor.shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            2,
        ), f"Tensor shape does not match expected shape of (1, 2), got {synced_tensor.shape}"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_sync_async_result_to_cpu_with_on_cpu_and_fusedspec_simple(self, mock_is_ranked_io):
        BATCH_SIZE = 2
        REQUEST_BATCH_SIZE = 1
        output_logits = [
            [torch.rand(REQUEST_BATCH_SIZE, 2), torch.rand(REQUEST_BATCH_SIZE, 4)],
            [torch.rand(REQUEST_BATCH_SIZE, 2), torch.rand(REQUEST_BATCH_SIZE, 4)],
        ]

        mock_is_ranked_io.return_value = False
        async_tensor_wrapper = AsyncTensorWrapper(output_logits, batch_padded=False, on_cpu=True)

        mock_seq_ids = torch.arange(0, BATCH_SIZE).reshape(BATCH_SIZE, 1)
        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids, is_fused_speculation=True
        )

        assert synced_tensor[0].shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            2,
        ), f"Tensor shape does not match expected concatenated shape of ({REQUEST_BATCH_SIZE * len(output_logits)}, 2), got {synced_tensor[0].shape}"
        assert synced_tensor[1].shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            4,
        ), f"Tensor shape does not match expected concatenated shape of ({REQUEST_BATCH_SIZE * len(output_logits)}, 4), got {synced_tensor[1].shape}"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_sync_async_result_to_cpu_with_on_cpu_and_fusedspec_batch_padded(
        self, mock_is_ranked_io
    ):
        BATCH_SIZE = 2
        REQUEST_BATCH_SIZE = 1
        output_logits = [
            [torch.rand(BATCH_SIZE, 2), torch.rand(BATCH_SIZE, 4)],
        ]

        mock_is_ranked_io.return_value = False
        async_tensor_wrapper = AsyncTensorWrapper(output_logits, batch_padded=True, on_cpu=True)

        mock_seq_ids_1 = torch.tensor([[0]], dtype=torch.int32)
        mock_seq_ids_2 = torch.tensor([[1]], dtype=torch.int32)

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids_1, is_fused_speculation=True
        )
        assert synced_tensor[0].shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            2,
        ), f"Tensor shape does not match expected concatenated shape of ({REQUEST_BATCH_SIZE * len(output_logits)}, 2), got {synced_tensor[0].shape}"
        assert synced_tensor[1].shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            4,
        ), f"Tensor shape does not match expected concatenated shape of ({REQUEST_BATCH_SIZE * len(output_logits)}, 4), got {synced_tensor[1].shape}"

        # returns a list of tensor(s)
        synced_tensor: List[torch.Tensor] = async_tensor_wrapper.sync_async_result_to_cpu(
            mock_seq_ids_2, is_fused_speculation=True
        )
        assert synced_tensor[0].shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            2,
        ), f"Tensor shape does not match expected concatenated shape of ({REQUEST_BATCH_SIZE * len(output_logits)}, 2), got {synced_tensor[0].shape}"
        assert synced_tensor[1].shape == (
            REQUEST_BATCH_SIZE * len(output_logits),
            4,
        ), f"Tensor shape does not match expected concatenated shape of ({REQUEST_BATCH_SIZE * len(output_logits)}, 4), got {synced_tensor[1].shape}"

    @patch("neuronx_distributed_inference.modules.async_execution.is_ranked_io")
    def test_early_exit(self, mock_is_ranked_io):
        TP_DEGREE = 32
        BATCH_SIZE = 2
        mock_ranked_tensor = [[torch.rand(BATCH_SIZE, 2)] for _ in range(TP_DEGREE)]

        mock_is_ranked_io.return_value = True
        async_tensor_wrapper_not_on_cpu = AsyncTensorWrapper(
            mock_ranked_tensor, batch_padded=False, on_cpu=False
        )

        mock_seq_ids = torch.arange(0, BATCH_SIZE, dtype=torch.int32).reshape(BATCH_SIZE, 1)
        res = async_tensor_wrapper_not_on_cpu.sync_async_result_to_cpu(
            mock_seq_ids, early_exit=True
        )

        assert res is None, f"Early Exit should return None, but found {res}"
