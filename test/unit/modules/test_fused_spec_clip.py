import unittest
from typing import List

import torch

from neuronx_distributed_inference.models.model_base import NeuronFusedSpecModel
from neuronx_distributed_inference.modules.eagle.hidden_state import HiddenStateRollingBuffer
from neuronx_distributed_inference.utils.testing import build_module

BATCH_SIZE = 4
SPECULATION_LEN = 5
VOCAB_SIZE = 12345
BUCKET_SIZE = 128
TP_DEGREE = 2


class MockOnDeviceSamplingConfig:
    pass


class MockNeuronConfig:
    def __init__(self, enable_eagle_speculation=True):
        self.batch_size = BATCH_SIZE
        self.speculation_length = SPECULATION_LEN
        self.medusa_speculation_length = 0
        self.enable_eagle_speculation = enable_eagle_speculation
        self.enable_token_tree = False
        self.token_tree_config = None
        self.on_device_sampling_config = MockOnDeviceSamplingConfig()
        self.output_logits = False


class MockInferenceConfig:
    def __init__(self, neuron_config):
        self.neuron_config = neuron_config


class MockHiddenStateRollingBuffer(HiddenStateRollingBuffer):
    def set_state(self, seq_ids, position_ids, hidden_state):
        return hidden_state


class TestNeuronFusedSpecModel(NeuronFusedSpecModel):
    def __init__(self, config, neuron_config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.neuron_config = neuron_config
        self.config.neuron_config = self.neuron_config

        self.batch_size = self.neuron_config.batch_size
        self.acceptance_padding_token = 0
        self.n_positions = BUCKET_SIZE
        self.hidden_state_rolling_buffer = MockHiddenStateRollingBuffer(
            self.batch_size, self.n_positions, self.n_positions
        )  # Included for compatibility

    def _eagle_token_gen_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        acceptance,  # To mock expected acceptance rate - Determines number of mock tokens generated per sequence in the forward pass.
    ):
        return self._tkg_impl(input_ids, attention_mask, position_ids, seq_ids, acceptance)

    def _token_gen_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        acceptance,  # To mock expected acceptance rate - Determines number of mock tokens generated per sequence in the forward pass.
    ):
        return self._tkg_impl(input_ids, attention_mask, position_ids, seq_ids, acceptance)

    def _tkg_impl(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        acceptance,  # To mock expected acceptance rate - Determines number of mock tokens generated per sequence in the forward pass.
    ):
        batch_size = attention_mask.shape[0]
        next_input_ids = input_ids.expand((self.batch_size, self.neuron_config.speculation_length))

        indices = (
            torch.arange(self.neuron_config.speculation_length, dtype=torch.int32)
            .expand(self.batch_size, self.neuron_config.speculation_length)
            .to(input_ids.device)
        )
        acceptance = acceptance.reshape(batch_size, 1)
        next_inputs_mask = indices < acceptance
        pad_tokens = torch.full_like(indices, fill_value=0, dtype=torch.int32)
        accepted_tokens = torch.where(next_inputs_mask, next_input_ids, pad_tokens).to(torch.int32)

        # Candidate and target are the same, so we accept to the maximum length prior to the first 0 token.
        return [
            accepted_tokens,
            accepted_tokens,
        ]


def generate_mock_tkg_batch(
    batch_size: int,
    vocab_size: int,
    position_ids: List[int],
    bucket_size: int,
    acceptance: List[int],
):
    assert batch_size > 0
    assert len(position_ids) == len(acceptance) == batch_size

    # Acceptance must be at least 1 for each batch line - The tokengen postprocessor cannot accept zero tokens.
    assert min(acceptance) > 0

    input_ids = torch.randint(1, vocab_size, (batch_size, 1)).to(torch.int32)
    attention_mask = torch.tensor(
        [[1 if i < position_ids[b] else 0 for i in range(bucket_size)] for b in range(batch_size)],
        dtype=torch.int32,
    )
    position_ids = torch.tensor([[position_ids[i]] for i in range(batch_size)], dtype=torch.int32)
    sequence_ids = torch.arange(batch_size, dtype=torch.int32).unsqueeze(1)
    acceptance = torch.tensor(acceptance, dtype=torch.int32).unsqueeze(1)

    return input_ids, attention_mask, position_ids, sequence_ids, acceptance


def output_position_ids_to_list(outputs):
    next_positions = outputs[3]
    return next_positions.detach().flatten().to("cpu").tolist()


class TestFusedSpecTokenGenPositionClip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define common constants
        cls.batch_size = BATCH_SIZE
        cls.vocab_size = VOCAB_SIZE
        cls.bucket_size = BUCKET_SIZE
        cls.speculation_length = SPECULATION_LEN

        # Initialize mock EAGLE Speculative Decoding model
        cls.eagle_neuron_config = MockNeuronConfig()
        cls.eagle_config = MockInferenceConfig(cls.eagle_neuron_config)

        position_ids = [cls.bucket_size // 2] * cls.batch_size
        acceptance = [1] * cls.batch_size
        warmup_inputs = generate_mock_tkg_batch(
            batch_size=BATCH_SIZE,
            vocab_size=VOCAB_SIZE,
            position_ids=position_ids,
            bucket_size=BUCKET_SIZE,
            acceptance=acceptance,
        )

        cls.model = build_module(
            module_cls=TestNeuronFusedSpecModel,
            example_inputs=[tuple(warmup_inputs)],
            module_init_kwargs={
                "config": cls.eagle_config,
                "neuron_config": cls.eagle_neuron_config,
            },
            tp_degree=TP_DEGREE,
        )

        # Initialize mock Fused Speculative Decoding model
        cls.fs_neuron_config = MockNeuronConfig(enable_eagle_speculation=False)
        cls.fs_config = MockInferenceConfig(cls.fs_neuron_config)
        cls.fs_model = build_module(
            module_cls=TestNeuronFusedSpecModel,
            example_inputs=[tuple(warmup_inputs)],
            module_init_kwargs={
                "config": cls.fs_config,
                "neuron_config": cls.fs_neuron_config,
            },
            tp_degree=TP_DEGREE,
        )

    def _test_single_iteration(self, position_ids, acceptance, assertion_fn):
        # TODO: Extend test to fused speculation
        for model_type in ["eagle"]:
            with self.subTest("_test_single_iteration", model_type=model_type):
                model = self.model if model_type == "eagle" else self.fs_model
                example_input = generate_mock_tkg_batch(
                    batch_size=self.batch_size,
                    vocab_size=self.vocab_size,
                    position_ids=position_ids,
                    bucket_size=self.bucket_size,
                    acceptance=acceptance,
                )
                outputs = model(*example_input)
                next_positions = output_position_ids_to_list(outputs)

                assertion_fn(next_positions)

    def test_upper_bound_min_speculation(self):
        """Check positions clipped at the upper bound (bucket_size - generation_length) in a
        single forward pass. Simulate minimal movement between position_ids and
        next_position_ids.
        """
        # Exceed valid positions [0, bucket_size)
        position_ids = [self.bucket_size + 20] * self.batch_size
        acceptance = [1] * self.batch_size

        def assertion_fn(next_positions):
            assert all(
                (x == self.bucket_size - self.speculation_length + 1 for x in next_positions)
            )

        # assertion_fn = lambda positions: assert all((x == self.bucket_size - self.neuron_config.speculation_length + 1 for x in positions))
        self._test_single_iteration(position_ids, acceptance, assertion_fn)

    def test_upper_bound_max_speculation(self):
        """Check positions clipped at the upper bound (bucket_size - generation_length) in a
        single forward pass. Start in bounds and simulate movement that would produce out-of-bounds
        without clipping."""
        # Within valid positions but can exceed with speculative decoding.
        position_ids = [self.bucket_size - self.speculation_length // 2] * self.batch_size
        acceptance = [self.speculation_length] * self.batch_size

        def assertion_fn(next_positions):
            assert all((x == self.bucket_size for x in next_positions))

        self._test_single_iteration(position_ids, acceptance, assertion_fn)

    def test_lower_bound_min_speculation(self):
        """Check positions clipped at the lower bound (0) in a single
        forward pass. Simulate minimal movement between position_ids and
        next_position_ids.
        """
        # Fall below valid positions [0, bucket_size)
        position_ids = [-1000] * self.batch_size
        acceptance = [1] * self.batch_size

        def assertion_fn(next_positions):
            assert all((x == 1 for x in next_positions))

        self._test_single_iteration(position_ids, acceptance, assertion_fn)

    def test_in_bounds_variable_speculation(self):
        """Check positions remain the same when in-bounds in a single
        forward pass. Simulate minimal movement between position_ids and
        next_position_ids."""
        # Set position IDs near center of range
        position_ids = [self.bucket_size // 2] * self.batch_size
        acceptance = torch.arange(1, self.batch_size + 1).tolist()

        def assertion_fn(next_positions):
            assert all((x == i + 1 + (self.bucket_size // 2) for i, x in enumerate(next_positions)))

        self._test_single_iteration(position_ids, acceptance, assertion_fn)

    def test_in_bounds_max_speculation(self):
        """Check positions remain the same when in-bounds in a single
        forward pass. Start in bounds and simulate movement that would remain in-bounds.
        """
        # Set position IDs near center of range
        position_ids = [self.bucket_size // 2] * self.batch_size
        acceptance = [
            max(self.speculation_length, 1),
            max(self.speculation_length - 2, 1),
            max(self.speculation_length - 3, 1),
            max(self.speculation_length, 1),
        ]

        def assertion_fn(next_positions):
            assert all(
                (x == acceptance[i] + (self.bucket_size // 2) for i, x in enumerate(next_positions))
            )

        self._test_single_iteration(position_ids, acceptance, assertion_fn)

    def test_upper_bound_multiple_max_speculation(self):
        """Check positions over multiple simulated token generation passes. In this test,
        run past the point that generation will reach the upper bound to verify correct
        clipping.
        """
        # Approximate maximum iterations needed to advance to the end of the sequence, starting at position 0
        num_iterations = int(round(self.bucket_size // self.speculation_length))

        # Set up acceptance and starting position IDs - Use slightly offset position IDs.
        curr_position_ids = torch.arange(self.batch_size).tolist()
        acceptance = [self.eagle_neuron_config.speculation_length] * self.batch_size

        for i in range(num_iterations * 2):
            curr_input = generate_mock_tkg_batch(
                batch_size=self.batch_size,
                vocab_size=self.vocab_size,
                position_ids=curr_position_ids,
                bucket_size=self.bucket_size,
                acceptance=acceptance,
            )
            outputs = self.model(*curr_input)
            curr_position_ids = output_position_ids_to_list(outputs)

            assert all((x >= 0 for x in curr_position_ids))
            assert all((x <= self.bucket_size for x in curr_position_ids))
