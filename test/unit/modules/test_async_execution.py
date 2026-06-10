import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

from neuronx_distributed_inference.modules.async_execution import (
    AsyncTensorWrapper,
    _async_request_ids_signature,
    _combine_vectorized_hybrid_apc_inputs,
    _is_chunked_prefill_execution,
    _is_context_encoding_execution,
    _with_hybrid_apc_candidate_owner_metadata,
    _with_hybrid_apc_owner_metadata,
    cancel_hybrid_apc_request,
    execute_model_prefix_caching,
    finish_hybrid_apc_request,
    prepare_disabled_hybrid_apc_model_inputs,
    prepare_hybrid_apc_model_inputs,
    prepare_hybrid_apc_request_for_execution,
)


class TestAsyncRequestIdsSignature(unittest.TestCase):
    def test_signature_preserves_request_order(self):
        model = SimpleNamespace(_qwen36_vllm_request_ids=["req-b", "req-a"])

        self.assertEqual(
            _async_request_ids_signature(model),
            ("req-b", "req-a"),
        )

    def test_signature_accepts_single_request_id(self):
        model = SimpleNamespace(_qwen36_vllm_request_ids="req-a")

        self.assertEqual(_async_request_ids_signature(model), ("req-a",))

    def test_signature_accepts_tensor_request_ids(self):
        model = SimpleNamespace(
            _qwen36_vllm_request_ids=torch.tensor([1, 0], dtype=torch.int32)
        )

        self.assertEqual(_async_request_ids_signature(model), (1, 0))


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


class TestHybridAPCAsyncBridge(unittest.TestCase):
    def test_bridge_is_empty_when_hybrid_apc_disabled(self):
        base = SimpleNamespace(config=SimpleNamespace(use_hybrid_apc_manager=False))

        args = prepare_hybrid_apc_model_inputs(base, {"seq_ids": torch.tensor([0])})

        self.assertEqual(args, [])

    def test_bridge_builds_restore_and_commit_tensors(self):
        base = SimpleNamespace(config=SimpleNamespace(use_hybrid_apc_manager=True))
        input_dict = {
            "seq_ids": torch.tensor([3, 4], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[256], [0]], dtype=torch.int32),
            "hybrid_restore_slot_ids": torch.tensor([7, 0], dtype=torch.int32),
            "hybrid_restore_mask": torch.tensor([1, 0], dtype=torch.int32),
            "hybrid_commit_slot_ids": torch.tensor([8, 9], dtype=torch.int32),
            "hybrid_commit_mask": torch.tensor([1, 0], dtype=torch.int32),
        }

        args = prepare_hybrid_apc_model_inputs(base, input_dict)

        self.assertEqual(len(args), 14)
        self.assertTrue(torch.equal(args[9], torch.tensor([7, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[10], torch.tensor([1, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[11], torch.tensor([256, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[12], torch.tensor([8, 9], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[13], torch.tensor([1, 0], dtype=torch.int32)))

    def test_bridge_does_not_infer_restore_mask_from_slot_presence(self):
        base = SimpleNamespace(config=SimpleNamespace(use_hybrid_apc_manager=True))
        input_dict = {
            "seq_ids": torch.tensor([3], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[256]], dtype=torch.int32),
            "hybrid_restore_slot_ids": torch.tensor([7], dtype=torch.int32),
        }

        args = prepare_hybrid_apc_model_inputs(base, input_dict)

        self.assertTrue(torch.equal(args[9], torch.tensor([7], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[10], torch.tensor([0], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[11], torch.tensor([256], dtype=torch.int32)))

    def test_bridge_debug_switches_zero_restore_and_commit_masks(self):
        base = SimpleNamespace(config=SimpleNamespace(use_hybrid_apc_manager=True))
        input_dict = {
            "seq_ids": torch.tensor([3, 4], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[256], [0]], dtype=torch.int32),
            "hybrid_restore_slot_ids": torch.tensor([7, 0], dtype=torch.int32),
            "hybrid_restore_mask": torch.tensor([1, 0], dtype=torch.int32),
            "hybrid_commit_slot_ids": torch.tensor([8, 9], dtype=torch.int32),
            "hybrid_commit_mask": torch.tensor([1, 1], dtype=torch.int32),
        }

        with patch.dict(
            "os.environ",
            {"QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT": "1"},
        ):
            args = prepare_hybrid_apc_model_inputs(base, input_dict)

        self.assertTrue(torch.equal(args[9], torch.tensor([7, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[10], torch.tensor([0, 0], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[12], torch.tensor([8, 9], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[13], torch.tensor([0, 0], dtype=torch.int32)))

    def test_bridge_rejects_active_slot_out_of_range(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                max_gdn_checkpoint_slots=2,
            )
        )
        input_dict = {
            "seq_ids": torch.tensor([0], dtype=torch.int32),
            "hybrid_restore_slot_ids": torch.tensor([2], dtype=torch.int32),
            "hybrid_restore_mask": torch.tensor([1], dtype=torch.int32),
        }

        with self.assertRaisesRegex(ValueError, "outside \\[0, 2\\)"):
            prepare_hybrid_apc_model_inputs(base, input_dict)

    def test_bridge_validates_active_slots_against_allocator(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                max_gdn_checkpoint_slots=10,
            ),
            hybrid_apc_slot_allocator=SimpleNamespace(
                committed_slots=(5,),
                reserved_slots=(7,),
            ),
        )
        input_dict = {
            "seq_ids": torch.tensor([0], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[128]], dtype=torch.int32),
            "hybrid_restore_slot_ids": torch.tensor([5], dtype=torch.int32),
            "hybrid_restore_mask": torch.tensor([1], dtype=torch.int32),
            "hybrid_commit_slot_ids": torch.tensor([7], dtype=torch.int32),
            "hybrid_commit_mask": torch.tensor([1], dtype=torch.int32),
        }

        args = prepare_hybrid_apc_model_inputs(base, input_dict)
        self.assertTrue(torch.equal(args[9], torch.tensor([5], dtype=torch.int32)))
        self.assertTrue(torch.equal(args[12], torch.tensor([7], dtype=torch.int32)))

        input_dict["hybrid_commit_slot_ids"] = torch.tensor([6], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "not a reserved checkpoint slot"):
            prepare_hybrid_apc_model_inputs(base, input_dict)

    def test_disabled_bridge_builds_inert_decode_args_without_validation(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                max_gdn_checkpoint_slots=1,
            ),
            hybrid_apc_slot_allocator=SimpleNamespace(
                committed_slots=(),
                reserved_slots=(),
            ),
        )
        input_dict = {
            "seq_ids": torch.tensor([3, 4], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[128], [256]], dtype=torch.int32),
            "hybrid_restore_slot_ids": torch.tensor([99, 100], dtype=torch.int32),
            "hybrid_restore_mask": torch.tensor([1, 1], dtype=torch.int32),
            "hybrid_commit_slot_ids": torch.tensor([101, 102], dtype=torch.int32),
            "hybrid_commit_mask": torch.tensor([1, 1], dtype=torch.int32),
        }

        args = prepare_disabled_hybrid_apc_model_inputs(base, input_dict)

        self.assertEqual(len(args), 14)
        for index in (9, 10, 11, 12, 13):
            self.assertTrue(
                torch.equal(args[index], torch.zeros((2,), dtype=torch.int32))
            )

    def test_prefix_caching_execution_prepares_and_finishes_hybrid_apc(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel()
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_apc_bridge": bridge,
                "request_id": "req-1",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
                "cumulative_hashes_by_prefix_len": {2: "h2", 4: "h4"},
                "attention_block_refs": {4: (11, 12)},
                "actual_refs": (21, 22),
            }
        )

        result, is_neuron = execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(result, "model-output")
        self.assertFalse(is_neuron)
        self.assertEqual(bridge.prepare_kwargs["request_id"], "req-1")
        self.assertEqual(bridge.prepare_kwargs["attention_hit_len"], 2)
        self.assertEqual(
            bridge.prepare_kwargs["cumulative_hashes_by_prefix_len"],
            {2: "h2", 4: "h4"},
        )
        self.assertEqual(
            bridge.prepare_kwargs["attention_block_refs_by_prefix_len"],
            {4: (11, 12)},
        )
        self.assertTrue(
            torch.equal(
                model.calls[0][0],
                torch.tensor([[12, 13]], dtype=torch.int32),
            )
        )
        self.assertIn("_hybrid_apc_prepared", input_dict)

        finish_hybrid_apc_request(input_dict)

        self.assertEqual(bridge.committed[0][0].request_id, "req-1")
        self.assertEqual(bridge.committed[0][1], (21, 22))
        self.assertEqual(bridge.finished, ["req-1"])
        self.assertNotIn("_hybrid_apc_prepared", input_dict)

    def test_prefix_caching_execution_uses_wrapper_hybrid_apc_owner(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel()
        model.config = SimpleNamespace(use_hybrid_apc_manager=True)
        model.hybrid_apc_bridge = bridge
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "request_id": "req-wrapper-owner",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
                "actual_refs": (31, 32),
            }
        )

        result, is_neuron = execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(result, "model-output")
        self.assertFalse(is_neuron)
        self.assertEqual(bridge.prepare_kwargs["request_id"], "req-wrapper-owner")
        self.assertIn("_hybrid_apc_prepared", input_dict)
        self.assertTrue(
            torch.equal(model.calls[0][-1], torch.tensor([1], dtype=torch.int32))
        )

        finish_hybrid_apc_request(input_dict)

        self.assertEqual(bridge.committed[0][0].request_id, "req-wrapper-owner")
        self.assertEqual(bridge.committed[0][1], (31, 32))
        self.assertEqual(bridge.finished, ["req-wrapper-owner"])

    def test_prefix_caching_execution_uses_wrapper_direct_runtime_flag(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=False),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel()
        model.use_hybrid_apc_manager = True
        model.hybrid_apc_bridge = bridge
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "request_id": "req-wrapper-direct",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        result, is_neuron = execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(result, "model-output")
        self.assertFalse(is_neuron)
        self.assertEqual(bridge.prepare_kwargs["request_id"], "req-wrapper-direct")
        self.assertIn("_hybrid_apc_prepared", input_dict)

    def test_prefix_caching_execution_finds_context_wrapper_bridge(self):
        bridge = _FakeHybridBridge()
        context_owner = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            context_encoding_model=context_owner,
        )
        model = _FakePrefixModel()
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "request_id": "req-context-owner",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(bridge.prepare_kwargs["request_id"], "req-context-owner")
        self.assertIn("_hybrid_apc_prepared", input_dict)

    def test_prefix_caching_execution_reuses_last_bridge_for_continuation(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            hybrid_apc_bridge=bridge,
        )
        model = _FakePrefixModel()
        first_input = _prefix_input_dict()
        first_input.update(
            {
                "request_id": "req-first",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        execute_model_prefix_caching(base, model, first_input)
        finish_hybrid_apc_request(first_input)
        base.hybrid_apc_bridge = None

        second_input = _prefix_input_dict()
        second_input.update(
            {
                "request_id": "req-second",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        execute_model_prefix_caching(base, model, second_input)

        self.assertEqual(bridge.prepare_kwargs["request_id"], "req-second")
        self.assertIn("_hybrid_apc_prepared", second_input)

    def test_prefix_caching_execution_uses_wrapper_scheduler_records(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel()
        model.config = SimpleNamespace(use_hybrid_apc_manager=True)
        model.hybrid_apc_bridge = bridge
        model._qwen36_vllm_request_ids = ("req-from-record",)
        model._qwen36_vllm_hybrid_apc_request_records = (
            {
                "request_id": "req-from-record",
                "vllm_attention_hit_len": 2,
                "request_prefix_len": 4,
                "cumulative_hashes_by_prefix_len": {2: "h2", 4: "h4"},
            },
        )
        input_dict = _prefix_input_dict()

        result, is_neuron = execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(result, "model-output")
        self.assertFalse(is_neuron)
        self.assertEqual(bridge.prepare_kwargs["request_id"], "req-from-record")
        self.assertEqual(bridge.prepare_kwargs["attention_hit_len"], 2)
        self.assertEqual(
            bridge.prepare_kwargs["cumulative_hashes_by_prefix_len"],
            {2: "h2", 4: "h4"},
        )
        self.assertIn("_hybrid_apc_prepared", input_dict)

    def test_prefix_caching_execution_does_not_prepare_hybrid_apc_for_generation(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel(tag="token_generation_model")
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.tensor([[13]], dtype=torch.int32),
                "attention_mask": torch.ones((1, 5), dtype=torch.int32),
                "position_ids": torch.tensor([[4]], dtype=torch.int32),
                "slot_mapping": torch.tensor([[4]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[5]], dtype=torch.int32),
                "computed_context_lens": torch.tensor([[4]], dtype=torch.int32),
                "hybrid_apc_bridge": bridge,
                "request_id": "req-1",
                "vllm_attention_hit_len": torch.tensor([4], dtype=torch.int32),
                "hybrid_restore_slot_ids": torch.tensor([5], dtype=torch.int32),
                "hybrid_restore_mask": torch.tensor([1], dtype=torch.int32),
                "hybrid_commit_slot_ids": torch.tensor([7], dtype=torch.int32),
                "hybrid_commit_mask": torch.tensor([1], dtype=torch.int32),
            }
        )

        with patch(
            "neuronx_distributed_inference.modules.async_execution.prepare_hybrid_apc_model_inputs",
            side_effect=AssertionError("decode should use inert Hybrid APC args"),
        ):
            result, is_neuron = execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(result, "model-output")
        self.assertFalse(is_neuron)
        self.assertEqual(bridge.prepare_calls, [])
        self.assertTrue(
            torch.equal(model.calls[0][-4], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(model.calls[0][-1], torch.tensor([0], dtype=torch.int32))
        )

    def test_prefix_caching_generation_rejects_invalid_token_before_neuron(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                vocab_size=248320,
            ),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            _qwen36_vllm_request_ids=("req-invalid",),
        )
        model = _FakePrefixModel(tag="token_generation_model")
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.tensor([[1065353216]], dtype=torch.int32),
                "attention_mask": torch.ones((1, 1280), dtype=torch.int32),
                "position_ids": torch.tensor([[1024]], dtype=torch.int32),
                "slot_mapping": torch.tensor([[2560]], dtype=torch.int32),
                "block_table": torch.tensor([[6, 7, 8, 9, 10]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[1025]], dtype=torch.int32),
                "computed_context_lens": torch.tensor([[1024]], dtype=torch.int32),
                "num_queries": torch.tensor([[1]], dtype=torch.int32),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Token generation input_ids contract violated before Neuron execution",
        ) as cm:
            execute_model_prefix_caching(base, model, input_dict)

        self.assertIn("0x3f800000", str(cm.exception))
        self.assertEqual(model.calls, [])

    def test_chunked_prefill_with_nonzero_positions_still_uses_context_execution(self):
        base = SimpleNamespace(
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            _is_prefill=lambda position_ids: not bool(position_ids.min().item()),
        )
        model = _FakePrefixModel(tag="token_generation_model")
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.arange(256, dtype=torch.int32).reshape(1, 256),
                "position_ids": torch.arange(256, 512, dtype=torch.int32).reshape(1, 256),
            }
        )

        self.assertTrue(_is_context_encoding_execution(base, model, input_dict))
        self.assertTrue(
            _is_chunked_prefill_execution(
                base,
                input_dict,
                is_fused_speculation=False,
            )
        )

    def test_single_token_decode_remains_generation_execution(self):
        base = SimpleNamespace(
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            _is_prefill=lambda position_ids: not bool(position_ids.min().item()),
        )
        model = _FakePrefixModel(tag="token_generation_model")
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.tensor([[13]], dtype=torch.int32),
                "position_ids": torch.tensor([[512]], dtype=torch.int32),
            }
        )

        self.assertFalse(_is_context_encoding_execution(base, model, input_dict))
        self.assertFalse(
            _is_chunked_prefill_execution(
                base,
                input_dict,
                is_fused_speculation=False,
            )
        )

    def test_single_token_cached_prefill_continuation_uses_context_execution(self):
        base = SimpleNamespace(
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            _is_prefill=lambda position_ids: not bool(position_ids.min().item()),
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.empty((1, 0), dtype=torch.int32),
                "position_ids": torch.empty((1, 0), dtype=torch.int32),
                "hybrid_prefill_completion_state": torch.tensor([0], dtype=torch.int32),
                "vllm_attention_hit_len": torch.tensor([2048], dtype=torch.int32),
                "request_prefix_len": 2049,
                "active_suffix_len": 1,
            }
        )

        self.assertTrue(
            _is_chunked_prefill_execution(
                base,
                input_dict,
                is_fused_speculation=False,
            )
        )

    def test_owner_metadata_single_token_continuation_uses_context_execution(self):
        base = SimpleNamespace(
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            _qwen36_vllm_request_ids=("req-2049",),
            _qwen36_vllm_prefill_completion_state=torch.tensor(
                [0],
                dtype=torch.int32,
            ),
            _qwen36_vllm_hybrid_apc_metadata_by_request_id={
                "req-2049": {
                    "vllm_attention_hit_len": 2048,
                    "request_prefix_len": 2049,
                    "active_suffix_len": 1,
                    "full_input_ids": tuple(range(2049)),
                },
            },
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.empty((1, 0), dtype=torch.int32),
                "position_ids": torch.empty((1, 0), dtype=torch.int32),
            }
        )

        enriched = _with_hybrid_apc_owner_metadata(input_dict, base)

        self.assertIn("hybrid_request_records", enriched)
        self.assertTrue(
            _is_chunked_prefill_execution(
                base,
                enriched,
                is_fused_speculation=False,
            )
        )

    def test_candidate_owner_metadata_uses_wrapper_records_for_prefill_probe(self):
        base = SimpleNamespace()
        wrapper = SimpleNamespace(
            _qwen36_vllm_request_ids=("req-wrapper-2049",),
            _qwen36_vllm_prefill_completion_state=torch.tensor(
                [0],
                dtype=torch.int32,
            ),
            _qwen36_vllm_hybrid_apc_metadata_by_request_id={
                "req-wrapper-2049": {
                    "vllm_attention_hit_len": 2048,
                    "request_prefix_len": 2049,
                    "active_suffix_len": 1,
                },
            },
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.empty((1, 0), dtype=torch.int32),
                "position_ids": torch.empty((1, 0), dtype=torch.int32),
            }
        )

        enriched = _with_hybrid_apc_candidate_owner_metadata(input_dict, base, wrapper)

        self.assertIn("hybrid_request_records", enriched)
        self.assertTrue(
            _is_chunked_prefill_execution(
                SimpleNamespace(
                    neuron_config=SimpleNamespace(
                        enable_fused_speculation=False,
                        enable_eagle_speculation=False,
                    )
                ),
                enriched,
                is_fused_speculation=False,
            )
        )

    def test_completed_single_token_decode_stays_generation_execution(self):
        base = SimpleNamespace(
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            _is_prefill=lambda position_ids: not bool(position_ids.min().item()),
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.tensor([[13]], dtype=torch.int32),
                "position_ids": torch.tensor([[2048]], dtype=torch.int32),
                "hybrid_prefill_completion_state": torch.tensor([1], dtype=torch.int32),
                "vllm_attention_hit_len": torch.tensor([2048], dtype=torch.int32),
                "request_prefix_len": 2048,
                "active_suffix_len": 1,
            }
        )

        self.assertFalse(
            _is_chunked_prefill_execution(
                base,
                input_dict,
                is_fused_speculation=False,
            )
        )

    def test_commit_debug_switch_cancels_instead_of_committing_metadata(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel()
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_apc_bridge": bridge,
                "request_id": "req-no-commit",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        with patch.dict("os.environ", {"QWEN36_DISABLE_HYBRID_GDN_COMMIT": "1"}):
            execute_model_prefix_caching(base, model, input_dict)
            finish_hybrid_apc_request(input_dict)

        self.assertEqual(bridge.committed, [])
        self.assertEqual(bridge.cancelled[0].request_id, "req-no-commit")
        self.assertNotIn("_hybrid_apc_prepared", input_dict)

    def test_hybrid_apc_debug_trace_includes_restore_commit_evidence(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "request_id": "req-debug",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        stdout = StringIO()
        with patch.dict("os.environ", {"QWEN36_HYBRID_APC_DEBUG": "1"}):
            with redirect_stdout(stdout):
                prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        log = stdout.getvalue()
        self.assertIn("[hybrid_apc_debug] prepare", log)
        self.assertIn("request_id='req-debug'", log)
        self.assertIn("attention_hit_len=2", log)
        self.assertIn("restore_len=2", log)
        self.assertIn("commit_prefix_len=4", log)
        self.assertIn("restore_slot=5", log)
        self.assertIn("commit_slot=7", log)
        self.assertIn("input_shape=(1, 4)", log)
        self.assertIn("prepared_shape=(1, 2)", log)
        self.assertIn("computed=tensor([[2]], dtype=torch.int32)", log)
        self.assertIn("restore_mask=tensor([1], dtype=torch.int32)", log)
        self.assertIn("commit_mask=tensor([1], dtype=torch.int32)", log)
        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor([[12, 13]], dtype=torch.int32),
            )
        )

    def test_prepare_debug_switches_zero_prepared_restore_commit_masks(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "request_id": "req-debug-switches",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        with patch.dict(
            "os.environ",
            {"QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT": "1"},
        ):
            prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_slot_ids"],
                torch.tensor([5], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_slot_ids"],
                torch.tensor([7], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_prepare_debug_switch_zeroes_only_restore_mask(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "request_id": "req-zero-restore-only",
                "vllm_attention_hit_len": torch.tensor([2], dtype=torch.int32),
            }
        )

        with patch.dict(
            "os.environ",
            {"QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK": "1"},
        ):
            prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([1], dtype=torch.int32),
            )
        )

    def test_prefix_caching_execution_cancels_hybrid_apc_on_model_failure(self):
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
        )
        bridge = _FakeHybridBridge()
        model = _FakePrefixModel(should_fail=True)
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_apc_bridge": bridge,
                "request_id": "req-1",
                "vllm_attention_hit_len": 2,
            }
        )

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(bridge.cancelled[0].request_id, "req-1")
        self.assertNotIn("_hybrid_apc_prepared", input_dict)

    def test_prefix_caching_execution_uses_model_registered_bridge_and_derived_hit(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            hybrid_apc_bridge=bridge,
        )
        model = _FakePrefixModel()
        input_dict = _prefix_input_dict()

        result, is_neuron = execute_model_prefix_caching(base, model, input_dict)

        self.assertEqual(result, "model-output")
        self.assertFalse(is_neuron)
        self.assertEqual(bridge.prepare_kwargs["request_id"], ("seq_id", 0))
        self.assertEqual(bridge.prepare_kwargs["attention_hit_len"], 0)

    def test_prefix_caching_execution_uses_full_prompt_tokens_for_suffix_request(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            hybrid_apc_bridge=bridge,
        )
        model = _FakePrefixModel()
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["hybrid_full_input_ids"] = torch.tensor(
            [[10, 11, 12, 13]],
            dtype=torch.int32,
        )

        execute_model_prefix_caching(base, model, input_dict)

        self.assertTrue(
            torch.equal(
                bridge.prepare_kwargs["input_dict"]["input_ids"],
                torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
            )
        )
        self.assertEqual(bridge.prepare_kwargs["attention_hit_len"], 2)

    def test_prefix_caching_execution_skips_hybrid_apc_for_suffix_without_full_prompt(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            hybrid_apc_bridge=bridge,
        )
        model = _FakePrefixModel()
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)

        execute_model_prefix_caching(base, model, input_dict)

        self.assertIsNone(bridge.prepare_kwargs)
        self.assertTrue(
            torch.equal(
                model.calls[0][0],
                torch.tensor([[12, 13]], dtype=torch.int32),
            )
        )

    def test_single_token_same_request_suffix_prepares_hybrid_apc(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            neuron_config=SimpleNamespace(
                enable_fused_speculation=False,
                enable_eagle_speculation=False,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "input_ids": torch.tensor([[99]], dtype=torch.int32),
                "attention_mask": torch.ones((1, 1), dtype=torch.int32),
                "position_ids": torch.tensor([[2048]], dtype=torch.int32),
                "slot_mapping": torch.tensor([[2304]], dtype=torch.int32),
                "block_table": torch.arange(10, dtype=torch.int32).reshape(1, 10),
                "computed_context_lens": torch.tensor([[2048]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[2049]], dtype=torch.int32),
                "request_id": "req-2049",
                "hybrid_cached_request_ids": ("req-2049",),
                "hybrid_prefill_completion_state": torch.tensor([0], dtype=torch.int32),
                "vllm_attention_hit_len": torch.tensor([2048], dtype=torch.int32),
                "request_prefix_len": 2049,
                "active_suffix_len": 1,
            }
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(bridge.suffix_prepare_calls[0]["request_id"], "req-2049")
        self.assertEqual(bridge.suffix_prepare_calls[0]["attention_hit_len"], 2048)
        self.assertEqual(bridge.suffix_prepare_calls[0]["request_prefix_len"], 2049)
        self.assertTrue(
            torch.equal(prepared["input_ids"], torch.tensor([[99]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                prepared["computed_context_lens"],
                torch.tensor([[2048]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(prepared["num_queries"], torch.tensor([[1]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(prepared["hybrid_restore_mask"], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_prefix_lens"],
                torch.tensor([2048], dtype=torch.int32),
            )
        )

    def test_vectorized_no_hit_batch_skips_hybrid_apc_request_prep(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-a", "req-b"),
                "full_context_lens": torch.tensor([4, 4], dtype=torch.int32),
                "computed_context_lens": torch.tensor([0, 0], dtype=torch.int32),
            }
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertIs(prepared, input_dict)
        self.assertIsNone(bridge.prepare_kwargs)
        self.assertNotIn("_hybrid_apc_prepared", input_dict)

    def test_vectorized_attention_hit_batch_prepares_each_row(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-a", "req-b"),
                "vllm_attention_hit_len": (2, 2),
                "request_prefix_len": (4, 4),
                "full_context_lens": torch.tensor([4, 4], dtype=torch.int32),
                "computed_context_lens": torch.tensor([2, 2], dtype=torch.int32),
            }
        )
        input_dict["input_ids"] = input_dict["input_ids"].repeat(2, 1)
        input_dict["attention_mask"] = input_dict["attention_mask"].repeat(2, 1)
        input_dict["position_ids"] = input_dict["position_ids"].repeat(2, 1)
        input_dict["seq_ids"] = torch.tensor([0, 1], dtype=torch.int32)
        input_dict["sampling_params"] = input_dict["sampling_params"].repeat(2, 1)
        input_dict["adapter_ids"] = torch.tensor([0, 0], dtype=torch.int32)
        input_dict["slot_mapping"] = input_dict["slot_mapping"].repeat(2, 1)
        input_dict["block_table"] = input_dict["block_table"].repeat(2, 1)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(
            [call["request_id"] for call in bridge.prepare_calls],
            ["req-a", "req-b"],
        )
        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor([[12, 13], [12, 13]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["computed_context_lens"],
                torch.tensor([[2], [2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2], [2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([1, 1], dtype=torch.int32),
            )
        )
        self.assertEqual(len(input_dict["_hybrid_apc_prepared"]), 2)

        finish_hybrid_apc_request(input_dict)

        self.assertEqual([item[0].request_id for item in bridge.committed], ["req-a", "req-b"])
        self.assertEqual(bridge.finished, ["req-a", "req-b"])

    def test_vectorized_strict_metadata_is_selected_per_row(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-a", "req-b"),
                "full_context_lens": torch.tensor([4, 4], dtype=torch.int32),
                "computed_context_lens": torch.tensor([2, 2], dtype=torch.int32),
                "cumulative_hashes_by_prefix_len": (
                    {2: b"a2", 4: b"a4"},
                    {2: b"b2", 4: b"b4"},
                ),
                "attention_block_refs_by_prefix_len": (
                    {4: (11, 12)},
                    {4: (21, 22)},
                ),
            }
        )
        input_dict["input_ids"] = input_dict["input_ids"].repeat(2, 1)
        input_dict["attention_mask"] = input_dict["attention_mask"].repeat(2, 1)
        input_dict["position_ids"] = input_dict["position_ids"].repeat(2, 1)
        input_dict["seq_ids"] = torch.tensor([0, 1], dtype=torch.int32)
        input_dict["sampling_params"] = input_dict["sampling_params"].repeat(2, 1)
        input_dict["adapter_ids"] = torch.tensor([0, 0], dtype=torch.int32)
        input_dict["slot_mapping"] = input_dict["slot_mapping"].repeat(2, 1)
        input_dict["block_table"] = input_dict["block_table"].repeat(2, 1)

        prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(
            bridge.prepare_calls[0]["cumulative_hashes_by_prefix_len"],
            {2: b"a2", 4: b"a4"},
        )
        self.assertEqual(
            bridge.prepare_calls[1]["cumulative_hashes_by_prefix_len"],
            {2: b"b2", 4: b"b4"},
        )
        self.assertEqual(
            bridge.prepare_calls[0]["attention_block_refs_by_prefix_len"],
            {4: (11, 12)},
        )
        self.assertEqual(
            bridge.prepare_calls[1]["attention_block_refs_by_prefix_len"],
            {4: (21, 22)},
        )

    def test_vectorized_strict_metadata_keeps_missing_rows_aligned(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("cached-a", "new-a"),
                "vllm_attention_hit_len": (None, 2),
                "request_prefix_len": (None, 4),
                "full_context_lens": torch.tensor([4, 4], dtype=torch.int32),
                "computed_context_lens": torch.tensor([0, 2], dtype=torch.int32),
                "cumulative_hashes_by_prefix_len": (None, {2: b"new-a-2"}),
                "attention_block_refs_by_prefix_len": (None, {2: (21,)}),
            }
        )
        input_dict["input_ids"] = input_dict["input_ids"].repeat(2, 1)
        input_dict["attention_mask"] = input_dict["attention_mask"].repeat(2, 1)
        input_dict["position_ids"] = input_dict["position_ids"].repeat(2, 1)
        input_dict["seq_ids"] = torch.tensor([0, 1], dtype=torch.int32)
        input_dict["sampling_params"] = input_dict["sampling_params"].repeat(2, 1)
        input_dict["adapter_ids"] = torch.tensor([0, 0], dtype=torch.int32)
        input_dict["slot_mapping"] = input_dict["slot_mapping"].repeat(2, 1)
        input_dict["block_table"] = input_dict["block_table"].repeat(2, 1)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(
            [call["request_id"] for call in bridge.prepare_calls],
            ["new-a"],
        )
        self.assertEqual(
            bridge.prepare_calls[0]["cumulative_hashes_by_prefix_len"],
            {2: b"new-a-2"},
        )
        self.assertEqual(
            bridge.prepare_calls[0]["attention_block_refs_by_prefix_len"],
            {2: (21,)},
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[4], [2]], dtype=torch.int32),
            )
        )

    def test_vectorized_mixed_hit_batch_pads_prepared_rows(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-a", "req-b"),
                "full_context_lens": torch.tensor([[4], [4]], dtype=torch.int32),
                "computed_context_lens": torch.tensor([[0], [2]], dtype=torch.int32),
            }
        )
        input_dict["input_ids"] = input_dict["input_ids"].repeat(2, 1)
        input_dict["attention_mask"] = input_dict["attention_mask"].repeat(2, 1)
        input_dict["position_ids"] = input_dict["position_ids"].repeat(2, 1)
        input_dict["seq_ids"] = torch.tensor([0, 1], dtype=torch.int32)
        input_dict["sampling_params"] = input_dict["sampling_params"].repeat(2, 1)
        input_dict["adapter_ids"] = torch.tensor([0, 0], dtype=torch.int32)
        input_dict["slot_mapping"] = input_dict["slot_mapping"].repeat(2, 1)
        input_dict["block_table"] = input_dict["block_table"].repeat(2, 1)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor([[10, 11, 12, 13], [12, 13, 0, 0]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["attention_mask"],
                torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["slot_mapping"],
                torch.tensor([[0, 1, 2, 3], [2, 3, -1, -1]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[4], [2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0, 1], dtype=torch.int32),
            )
        )

    def test_vectorized_packed_suffix_batch_splits_by_query_lengths(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-a", "req-b"),
                "input_ids": torch.tensor([[12, 13, 22, 23]], dtype=torch.int32),
                "attention_mask": torch.ones((1, 4), dtype=torch.int32),
                "position_ids": torch.tensor([[2, 3, 2, 3]], dtype=torch.int32),
                "seq_ids": torch.tensor([0], dtype=torch.int32),
                "adapter_ids": torch.tensor([0], dtype=torch.int32),
                "slot_mapping": torch.tensor([[2, 3, 6, 7]], dtype=torch.int32),
                "block_table": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[4], [4]], dtype=torch.int32),
                "computed_context_lens": torch.tensor([[2], [2]], dtype=torch.int32),
                "num_queries": torch.tensor([[2], [2]], dtype=torch.int32),
                "cumulative_hashes_by_prefix_len": (
                    {2: "hash-a-2", 4: "hash-a-4"},
                    {2: "hash-b-2", 4: "hash-b-4"},
                ),
                "attention_block_refs_by_prefix_len": (
                    {2: (1,), 4: (1, 2)},
                    {2: (3,), 4: (3, 4)},
                ),
            }
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(
            [call["request_id"] for call in bridge.suffix_prepare_calls],
            ["req-a", "req-b"],
        )
        self.assertEqual(
            bridge.suffix_prepare_calls[0]["cumulative_hashes_by_prefix_len"],
            {2: "hash-a-2", 4: "hash-a-4"},
        )
        self.assertEqual(
            bridge.suffix_prepare_calls[1]["attention_block_refs_by_prefix_len"],
            {2: (3,), 4: (3, 4)},
        )
        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor([[12, 13], [22, 23]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["position_ids"],
                torch.tensor([[2, 3], [2, 3]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["slot_mapping"],
                torch.tensor([[2, 3], [6, 7]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["seq_ids"],
                torch.tensor([0, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([1, 1], dtype=torch.int32),
            )
        )

    def test_vectorized_request_records_override_collapsed_metadata(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": "cold-only",
                "vllm_attention_hit_len": 0,
                "input_ids": torch.tensor(
                    [[12, 13, 20, 21, 22, 23]],
                    dtype=torch.int32,
                ),
                "attention_mask": torch.ones((1, 6), dtype=torch.int32),
                "position_ids": torch.tensor(
                    [[2, 3, 0, 1, 2, 3]],
                    dtype=torch.int32,
                ),
                "seq_ids": torch.tensor([0], dtype=torch.int32),
                "adapter_ids": torch.tensor([0], dtype=torch.int32),
                "slot_mapping": torch.tensor(
                    [[2, 3, 10, 11, 12, 13]],
                    dtype=torch.int32,
                ),
                "block_table": torch.tensor(
                    [[1, 2], [5, 6]],
                    dtype=torch.int32,
                ),
                "hybrid_request_records": (
                    {
                        "request_id": "warm",
                        "vllm_attention_hit_len": 2,
                        "request_prefix_len": 4,
                        "active_suffix_len": 2,
                        "cumulative_hashes_by_prefix_len": {
                            2: "warm-h2",
                            4: "warm-h4",
                        },
                        "attention_block_refs_by_prefix_len": {
                            2: (1,),
                            4: (1, 2),
                        },
                    },
                    {
                        "request_id": "cold",
                        "vllm_attention_hit_len": 0,
                        "request_prefix_len": 4,
                        "active_suffix_len": 4,
                    },
                ),
            }
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(
            [call["request_id"] for call in bridge.suffix_prepare_calls],
            ["warm"],
        )
        self.assertEqual(bridge.prepare_calls, [])
        self.assertEqual(
            bridge.suffix_prepare_calls[0]["cumulative_hashes_by_prefix_len"],
            {2: "warm-h2", 4: "warm-h4"},
        )
        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor([[12, 13, 0, 0], [20, 21, 22, 23]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([1, 0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2], [4]], dtype=torch.int32),
            )
        )

    def test_vectorized_cached_decode_row_does_not_require_prefix_restore(self):
        bridge = _FakeHybridBridge()
        original_prepare_request = bridge.prepare_request

        def prepare_request_with_vector_full_context_lens(**kwargs):
            prepared = original_prepare_request(**kwargs)
            prepared.input_dict["full_context_lens"] = prepared.input_dict[
                "full_context_lens"
            ].reshape(-1)
            return prepared

        bridge.prepare_request = prepare_request_with_vector_full_context_lens
        base = SimpleNamespace(
            config=SimpleNamespace(use_hybrid_apc_manager=True),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-cached", "req-new"),
                "hybrid_cached_request_ids": ("req-cached",),
                "hybrid_prefill_completion_state": torch.tensor(
                    [True, False],
                    dtype=torch.bool,
                ),
                "input_ids": torch.tensor([[99, 20, 21, 22]], dtype=torch.int32),
                "attention_mask": torch.ones((1, 4), dtype=torch.int32),
                "position_ids": torch.tensor([[4, 0, 1, 2]], dtype=torch.int32),
                "seq_ids": torch.tensor([0], dtype=torch.int32),
                "adapter_ids": torch.tensor([0], dtype=torch.int32),
                "slot_mapping": torch.tensor([[4, 5, 6, 7]], dtype=torch.int32),
                "block_table": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[5], [3]], dtype=torch.int32),
                "computed_context_lens": torch.tensor([[4], [0]], dtype=torch.int32),
                "num_queries": torch.tensor([[1], [3]], dtype=torch.int32),
            }
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(
            [call["request_id"] for call in bridge.prepare_calls],
            ["req-new"],
        )
        self.assertEqual(bridge.suffix_prepare_calls, [])
        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor([[99, 0, 0], [20, 21, 22]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0, 0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0, 1], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["full_context_lens"],
                torch.tensor([[5], [3]], dtype=torch.int32),
            )
        )

    def test_vectorized_cached_decode_row_pads_to_cte_bucket(self):
        bridge = _FakeHybridBridge()
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                pad_token_id=0,
            ),
            neuron_config=SimpleNamespace(
                context_encoding_buckets=[2, 4],
                pa_block_size=2,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict.update(
            {
                "hybrid_request_id": ("req-cached", "req-new"),
                "hybrid_cached_request_ids": ("req-cached",),
                "hybrid_prefill_completion_state": torch.tensor(
                    [True, False],
                    dtype=torch.bool,
                ),
                "input_ids": torch.tensor([[99, 20, 21, 22]], dtype=torch.int32),
                "attention_mask": torch.ones((1, 4), dtype=torch.int32),
                "position_ids": torch.tensor([[4, 0, 1, 2]], dtype=torch.int32),
                "seq_ids": torch.tensor([0], dtype=torch.int32),
                "adapter_ids": torch.tensor([0], dtype=torch.int32),
                "slot_mapping": torch.tensor([[8, 10, 11, 12]], dtype=torch.int32),
                "block_table": torch.tensor(
                    [[1, 2, 3], [4, 5, 6]],
                    dtype=torch.int32,
                ),
                "full_context_lens": torch.tensor([[5], [3]], dtype=torch.int32),
                "computed_context_lens": torch.tensor([[4], [0]], dtype=torch.int32),
                "num_queries": torch.tensor([[1], [3]], dtype=torch.int32),
            }
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertTrue(
            torch.equal(
                prepared["input_ids"],
                torch.tensor(
                    [[99, 0, 0, 0], [20, 21, 22, 0]],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["attention_mask"],
                torch.tensor(
                    [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["slot_mapping"],
                torch.tensor([[8, -1, -1, -1], [10, 11, 12, -1]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["block_table"],
                torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            )
        )

    def test_vectorized_combiner_repairs_short_active_slot_mapping(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                pad_token_id=0,
            ),
            neuron_config=SimpleNamespace(
                context_encoding_buckets=[4],
                pa_block_size=2,
            ),
        )
        row_cached_decode = {
            "input_ids": torch.tensor([[99]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 1), dtype=torch.int32),
            "position_ids": torch.tensor([[4]], dtype=torch.int32),
            "seq_ids": torch.tensor([0], dtype=torch.int32),
            "slot_mapping": torch.tensor([8], dtype=torch.int32),
            "block_table": torch.tensor([[2, 3]], dtype=torch.int32),
            "full_context_lens": torch.tensor([[5]], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[4]], dtype=torch.int32),
            "num_queries": torch.tensor([[1]], dtype=torch.int32),
        }
        row_prefill = {
            "input_ids": torch.tensor([[20, 21, 22]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 3), dtype=torch.int32),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.int32),
            "slot_mapping": torch.tensor([10], dtype=torch.int32),
            "block_table": torch.tensor([[4, 5]], dtype=torch.int32),
            "full_context_lens": torch.tensor([[3]], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[0]], dtype=torch.int32),
            "num_queries": torch.tensor([[3]], dtype=torch.int32),
        }

        combined = _combine_vectorized_hybrid_apc_inputs(
            base,
            dict(row_cached_decode),
            [row_cached_decode, row_prefill],
        )

        self.assertTrue(
            torch.equal(
                combined["slot_mapping"],
                torch.tensor(
                    [[8, -1, -1, -1], [8, 9, 10, -1]],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["seq_ids"],
                torch.tensor([0, 1], dtype=torch.int32),
            )
        )

    def test_vectorized_combiner_preserves_restore_prefix_block_table(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                pad_token_id=0,
            ),
            neuron_config=SimpleNamespace(
                context_encoding_buckets=[4],
                pa_block_size=2,
            ),
        )
        row_a = {
            "input_ids": torch.tensor([[30, 31]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 2), dtype=torch.int32),
            "position_ids": torch.tensor([[4, 5]], dtype=torch.int32),
            "slot_mapping": torch.tensor([[20, 21]], dtype=torch.int32),
            "block_table": torch.tensor([[7, 8]], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[4]], dtype=torch.int32),
            "num_queries": torch.tensor([[2]], dtype=torch.int32),
            "hybrid_restore_mask": torch.tensor([1], dtype=torch.int32),
            "hybrid_restore_prefix_lens": torch.tensor([4], dtype=torch.int32),
            "rotary_position_ids": torch.tensor(
                [[[4, 5]], [[4, 5]], [[4, 5]]],
                dtype=torch.int32,
            ),
        }
        row_b = {
            **row_a,
            "input_ids": torch.tensor([[40, 41]], dtype=torch.int32),
            "block_table": torch.tensor([[9, 10]], dtype=torch.int32),
            "rotary_position_ids": torch.tensor(
                [[[6, 7]], [[6, 7]], [[6, 7]]],
                dtype=torch.int32,
            ),
        }

        combined = _combine_vectorized_hybrid_apc_inputs(
            base,
            dict(row_a),
            [row_a, row_b],
        )

        self.assertTrue(
            torch.equal(
                combined["block_table"],
                torch.tensor([[7, 8], [9, 10]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                combined["rotary_position_ids"],
                torch.tensor(
                    [
                        [[4, 5], [6, 7]],
                        [[4, 5], [6, 7]],
                        [[4, 5], [6, 7]],
                    ],
                    dtype=torch.int32,
                ),
            )
        )

    def test_vectorized_combiner_repairs_active_window_slot_mapping(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                pad_token_id=0,
            ),
            neuron_config=SimpleNamespace(
                context_encoding_buckets=[4],
                pa_block_size=2,
            ),
        )
        row_suffix = {
            "input_ids": torch.tensor([[30, 31]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 2), dtype=torch.int32),
            "position_ids": torch.tensor([[2, 3]], dtype=torch.int32),
            "seq_ids": torch.tensor([0], dtype=torch.int32),
            "slot_mapping": torch.full((1, 2), -1, dtype=torch.int32),
            "block_table": torch.tensor([[4]], dtype=torch.int32),
            "full_context_lens": torch.tensor([[4]], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[2]], dtype=torch.int32),
            "num_queries": torch.tensor([[2]], dtype=torch.int32),
        }
        row_decode = {
            "input_ids": torch.tensor([[99]], dtype=torch.int32),
            "attention_mask": torch.ones((1, 1), dtype=torch.int32),
            "position_ids": torch.tensor([[4]], dtype=torch.int32),
            "slot_mapping": torch.full((1, 1), -1, dtype=torch.int32),
            "block_table": torch.tensor([[5]], dtype=torch.int32),
            "full_context_lens": torch.tensor([[5]], dtype=torch.int32),
            "computed_context_lens": torch.tensor([[4]], dtype=torch.int32),
            "num_queries": torch.tensor([[1]], dtype=torch.int32),
        }

        combined = _combine_vectorized_hybrid_apc_inputs(
            base,
            dict(row_suffix),
            [row_suffix, row_decode],
        )

        self.assertTrue(
            torch.equal(
                combined["slot_mapping"],
                torch.tensor(
                    [[8, 9, -1, -1], [10, -1, -1, -1]],
                    dtype=torch.int32,
                ),
            )
        )

    def test_cancel_hybrid_apc_request_is_noop_without_prepared_request(self):
        input_dict = {}

        cancel_hybrid_apc_request(input_dict)

        self.assertEqual(input_dict, {})

    def test_strict_hybrid_apc_requires_attached_bridge(self):
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "requires a scheduler bridge"):
            prepare_hybrid_apc_request_for_execution(base, _prefix_input_dict())

    def test_strict_hybrid_apc_rejects_suffix_without_full_prompt(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        bridge.prepare_suffix_only_request = None
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "suffix-only input"):
            prepare_hybrid_apc_request_for_execution(base, input_dict)

    def test_strict_hybrid_apc_suffix_chunk_uses_active_prefix_boundary(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["request_prefix_len"] = 6
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_active_suffix_len"] = torch.tensor([2], dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertEqual(len(bridge.suffix_prepare_calls), 1)
        self.assertEqual(bridge.suffix_prepare_calls[0]["request_prefix_len"], 4)
        self.assertTrue(
            torch.equal(
                prepared["full_context_lens"],
                torch.tensor([[4]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_prefix_lens"],
                torch.tensor([2], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_suffix_chunk_without_checkpoint_uses_inert_controls(
        self,
    ):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True

        def raise_unbacked_suffix_error(**kwargs):
            bridge.suffix_prepare_calls.append(kwargs)
            raise ValueError(
                "suffix-only hybrid APC received an attention prefix hit "
                "without scheduler-authorized GDN checkpoint metadata"
            )

        bridge.prepare_suffix_only_request = raise_unbacked_suffix_error
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["full_context_lens"] = torch.tensor([[6]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["request_prefix_len"] = 6
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_active_suffix_len"] = torch.tensor([2], dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertEqual(len(bridge.suffix_prepare_calls), 1)
        self.assertEqual(bridge.suffix_prepare_calls[0]["request_prefix_len"], 4)
        self.assertNotIn("_hybrid_apc_prepared", input_dict)
        self.assertTrue(torch.equal(prepared["input_ids"], input_dict["input_ids"]))
        self.assertTrue(
            torch.equal(
                prepared["computed_context_lens"],
                torch.tensor([[2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["full_context_lens"],
                torch.tensor([[4]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_suffix_chunk_uses_scheduled_active_len(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor(
            [[12, 13, 14, 15]],
            dtype=torch.int32,
        )
        input_dict["attention_mask"] = torch.ones((1, 4), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["full_context_lens"] = torch.tensor([[6]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["request_prefix_len"] = 6
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_active_suffix_len"] = torch.tensor([2], dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertFalse(bridge.suffix_prepare_calls)
        self.assertTrue(torch.equal(prepared["input_ids"], input_dict["input_ids"]))
        self.assertTrue(
            torch.equal(
                prepared["attention_mask"],
                torch.tensor([[1, 1, 0, 0]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["full_context_lens"],
                torch.tensor([[4]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_full_suffix_without_checkpoint_still_raises(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True

        def raise_unbacked_suffix_error(**kwargs):
            bridge.suffix_prepare_calls.append(kwargs)
            raise ValueError(
                "suffix-only hybrid APC received an attention prefix hit "
                "without scheduler-authorized GDN checkpoint metadata"
            )

        bridge.prepare_suffix_only_request = raise_unbacked_suffix_error
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["request_prefix_len"] = 4
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_active_suffix_len"] = torch.tensor([2], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "scheduler-authorized GDN"):
            prepare_hybrid_apc_request_for_execution(base, input_dict)

    def test_strict_hybrid_apc_cached_prefill_suffix_without_checkpoint_is_inert(
        self,
    ):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True

        def raise_unbacked_suffix_error(**kwargs):
            bridge.suffix_prepare_calls.append(kwargs)
            raise ValueError(
                "suffix-only hybrid APC received an attention prefix hit "
                "without scheduler-authorized GDN checkpoint metadata"
            )

        bridge.prepare_suffix_only_request = raise_unbacked_suffix_error
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["request_id"] = "req-cached-prefill"
        input_dict["request_prefix_len"] = 4
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_active_suffix_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_cached_request_ids"] = ("req-cached-prefill",)
        input_dict["hybrid_prefill_completion_state"] = torch.tensor(
            [False],
            dtype=torch.bool,
        )

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertEqual(len(bridge.suffix_prepare_calls), 1)
        self.assertTrue(
            torch.equal(
                prepared["full_context_lens"],
                torch.tensor([[4]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_seq_id_prefill_suffix_without_checkpoint_is_inert(
        self,
    ):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True

        def raise_unbacked_suffix_error(**kwargs):
            bridge.suffix_prepare_calls.append(kwargs)
            raise ValueError(
                "suffix-only hybrid APC received an attention prefix hit "
                "without scheduler-authorized GDN checkpoint metadata"
            )

        bridge.prepare_suffix_only_request = raise_unbacked_suffix_error
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["full_context_lens"] = torch.tensor([[4]], dtype=torch.int32)
        input_dict["seq_ids"] = torch.tensor([0], dtype=torch.int32)
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertFalse(bridge.suffix_prepare_calls)
        self.assertTrue(
            torch.equal(
                prepared["full_context_lens"],
                torch.tensor([[4]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[2]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_suffix_chunk_other_bridge_error_still_raises(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True

        def raise_other_suffix_error(**kwargs):
            bridge.suffix_prepare_calls.append(kwargs)
            raise ValueError("unrelated suffix bridge error")

        bridge.prepare_suffix_only_request = raise_other_suffix_error
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[2, 3]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[2]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["request_prefix_len"] = 6
        input_dict["vllm_attention_hit_len"] = torch.tensor([2], dtype=torch.int32)
        input_dict["hybrid_active_suffix_len"] = torch.tensor([2], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "unrelated suffix bridge error"):
            prepare_hybrid_apc_request_for_execution(base, input_dict)

    def test_strict_hybrid_apc_allows_zero_hit_partial_chunk(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[0, 1]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[0, 1]], dtype=torch.int32)
        input_dict["full_context_lens"] = torch.tensor([[4]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[0]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["vllm_attention_hit_len"] = torch.tensor([0], dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertFalse(bridge.suffix_prepare_calls)
        self.assertTrue(torch.equal(prepared["input_ids"], input_dict["input_ids"]))
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_commits_zero_hit_chunk_boundary_with_hash(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["input_ids"] = torch.tensor([[12, 13]], dtype=torch.int32)
        input_dict["attention_mask"] = torch.ones((1, 2), dtype=torch.int32)
        input_dict["position_ids"] = torch.tensor([[0, 1]], dtype=torch.int32)
        input_dict["slot_mapping"] = torch.tensor([[0, 1]], dtype=torch.int32)
        input_dict["full_context_lens"] = torch.tensor([[4]], dtype=torch.int32)
        input_dict["computed_context_lens"] = torch.tensor([[0]], dtype=torch.int32)
        input_dict["request_id"] = "req-strict"
        input_dict["vllm_attention_hit_len"] = torch.tensor([0], dtype=torch.int32)
        input_dict["cumulative_hashes_by_prefix_len"] = {2: b"hash-2"}
        input_dict["attention_block_refs_by_prefix_len"] = {2: [9]}

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertEqual(len(bridge.prepare_calls), 1)
        self.assertFalse(bridge.suffix_prepare_calls)
        self.assertEqual(bridge.prepare_kwargs["request_prefix_len"], 2)
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([1], dtype=torch.int32),
            )
        )

    def test_strict_hybrid_apc_allows_zero_hit_without_hash_metadata(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["request_id"] = "req-short-cold"
        input_dict["request_prefix_len"] = 255
        input_dict["vllm_attention_hit_len"] = torch.tensor([0], dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertTrue(torch.equal(prepared["input_ids"], input_dict["input_ids"]))
        self.assertTrue(
            torch.equal(
                prepared["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["hybrid_commit_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_zero_hit_strict_hybrid_apc_rebuilds_active_attention_mask(self):
        bridge = _FakeHybridBridge()
        bridge.requires_external_metadata = True
        base = SimpleNamespace(
            config=SimpleNamespace(
                use_hybrid_apc_manager=True,
                hybrid_apc_require_vllm_metadata=True,
            ),
            hybrid_apc_bridge=bridge,
        )
        input_dict = _prefix_input_dict()
        input_dict["request_id"] = "req-cold"
        input_dict["request_prefix_len"] = 4
        input_dict["vllm_attention_hit_len"] = torch.tensor([0], dtype=torch.int32)
        input_dict["attention_mask"] = torch.zeros((1, 4), dtype=torch.int32)

        prepared = prepare_hybrid_apc_request_for_execution(base, input_dict)

        self.assertFalse(bridge.prepare_calls)
        self.assertTrue(
            torch.equal(
                prepared["num_queries"],
                torch.tensor([[4]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared["attention_mask"],
                torch.ones((1, 4), dtype=torch.int32),
            )
        )


def _prefix_input_dict():
    return {
        "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        "attention_mask": torch.ones((1, 4), dtype=torch.int32),
        "position_ids": torch.arange(4, dtype=torch.int32).unsqueeze(0),
        "seq_ids": torch.tensor([0], dtype=torch.int32),
        "sampling_params": torch.zeros((1, 1), dtype=torch.int32),
        "adapter_ids": torch.zeros((1,), dtype=torch.int32),
        "slot_mapping": torch.arange(4, dtype=torch.int32).unsqueeze(0),
        "block_table": torch.tensor([[1, 2]], dtype=torch.int32),
        "full_context_lens": torch.tensor([[4]], dtype=torch.int32),
        "computed_context_lens": torch.tensor([[0]], dtype=torch.int32),
    }


class _FakeHybridBridge:
    def __init__(self):
        self.prepare_kwargs = None
        self.prepare_calls = []
        self.suffix_prepare_calls = []
        self.committed = []
        self.finished = []
        self.cancelled = []

    def prepare_request(self, **kwargs):
        self.prepare_kwargs = kwargs
        self.prepare_calls.append(kwargs)
        input_dict = dict(kwargs["input_dict"])
        restore_len = int(kwargs["attention_hit_len"])
        prompt_len = int(input_dict["input_ids"].shape[1])
        suffix_len = prompt_len - restore_len
        restore_slot = 5 if restore_len > 0 else 0
        input_dict.update(
            {
                "input_ids": input_dict["input_ids"][:, restore_len:prompt_len],
                "attention_mask": input_dict["attention_mask"][:, restore_len:prompt_len],
                "position_ids": input_dict["position_ids"][:, restore_len:prompt_len],
                "slot_mapping": input_dict["slot_mapping"][:, restore_len:prompt_len],
                "computed_context_lens": torch.tensor([[restore_len]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[prompt_len]], dtype=torch.int32),
                "num_queries": torch.tensor([[suffix_len]], dtype=torch.int32),
                "hybrid_restore_slot_ids": torch.tensor([restore_slot], dtype=torch.int32),
                "hybrid_restore_mask": torch.tensor([1 if restore_len > 0 else 0], dtype=torch.int32),
                "hybrid_restore_prefix_lens": torch.tensor([restore_len], dtype=torch.int32),
                "hybrid_commit_slot_ids": torch.tensor([7], dtype=torch.int32),
                "hybrid_commit_mask": torch.tensor([1], dtype=torch.int32),
            }
        )
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            input_dict=input_dict,
            plan=SimpleNamespace(
                restore_checkpoint_prefix_len=restore_len,
                checkpoint_slot=restore_slot,
            ),
            commit_prefix_len=prompt_len,
            commit_slot=7,
            attention_block_refs=(11, 12),
        )

    def prepare_suffix_only_request(self, **kwargs):
        self.suffix_prepare_calls.append(kwargs)
        input_dict = dict(kwargs["input_dict"])
        restore_len = int(kwargs["attention_hit_len"])
        prompt_len = int(kwargs["request_prefix_len"])
        suffix_len = int(input_dict["input_ids"].shape[1])
        input_dict.update(
            {
                "computed_context_lens": torch.tensor([[restore_len]], dtype=torch.int32),
                "full_context_lens": torch.tensor([[prompt_len]], dtype=torch.int32),
                "num_queries": torch.tensor([[suffix_len]], dtype=torch.int32),
                "hybrid_restore_slot_ids": torch.tensor([5], dtype=torch.int32),
                "hybrid_restore_mask": torch.tensor([1], dtype=torch.int32),
                "hybrid_restore_prefix_lens": torch.tensor([restore_len], dtype=torch.int32),
                "hybrid_commit_slot_ids": torch.tensor([0], dtype=torch.int32),
                "hybrid_commit_mask": torch.tensor([0], dtype=torch.int32),
            }
        )
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            input_dict=input_dict,
            plan=SimpleNamespace(
                restore_checkpoint_prefix_len=restore_len,
                checkpoint_slot=5,
            ),
            commit_prefix_len=prompt_len,
            commit_slot=None,
            attention_block_refs=(11, 12),
        )

    def commit_prefill(self, prepared, *, attention_block_refs=None):
        self.committed.append((prepared, tuple(attention_block_refs)))

    def finish_request(self, request_id):
        self.finished.append(request_id)

    def cancel_request(self, prepared):
        self.cancelled.append(prepared)


class _FakePrefixModel:
    def __init__(self, should_fail=False, tag="context_encoding_model"):
        self.should_fail = should_fail
        self.tag = tag
        self.calls = []

    def __call__(self, *args, **kwargs):
        if self.should_fail:
            raise RuntimeError("model failed")
        self.calls.append(args)
        return "model-output"

    def is_neuron(self):
        return False
