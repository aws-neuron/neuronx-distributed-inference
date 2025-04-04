import unittest
from copy import deepcopy

import pytest
import torch

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    ModelWrapper,
)


class TestInputTruncation(unittest.TestCase):
    def setUp(self):
        # Replicate truncation behavior by passing input longer than max_context length or max_length
        # Input length=10, max_context_length=5, max_length=8

        self.pad_token_id = 1
        self.model_cls = NeuronBaseModel
        self.max_context_length = 5
        self.max_length = 7
        self.config = InferenceConfig(
            max_context_length=10,
            pad_token_id=self.pad_token_id,
            neuron_config=NeuronConfig(
                max_context_length=self.max_context_length, max_length=self.max_length
            ),
        )

        self.test_inputs = [
            torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),  # input_ids
            torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),  # attention_mask
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),  # position_ids
            torch.tensor([0]),  # seq_ids
        ]

    def test_pad_inputs_context_encoding_no_prompt_truncation(self):
        # Test ValueError thrown for context encoding model with default settings

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.tag = CONTEXT_ENCODING_MODEL_TAG

        with self.assertRaises(ValueError) as context:
            model_wrapper.pad_inputs(*self.test_inputs, pad_type="max")

        assert "Inputs supplied" in str(context.exception)

    def test_pad_inputs_no_prompt_truncation(self):
        # Test ValueError thrown with default settings

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)

        with self.assertRaises(ValueError) as context:
            model_wrapper.pad_inputs(*self.test_inputs, pad_type="max")

        assert "Attention mask supplied" in str(context.exception)

    def test_pad_inputs_context_encoding_allow_prompt_truncation(self):
        # Test warning thrown and input_ids, attention_mask, position_ids truncated for context encoding model with allow_input_truncation=True

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.tag = CONTEXT_ENCODING_MODEL_TAG
        model_wrapper.neuron_config.allow_input_truncation = True

        with self.assertWarns(UserWarning) as context:
            padded_inputs = model_wrapper.pad_inputs(*self.test_inputs, pad_type="max")

        assert (
            padded_inputs[0].shape[1] == self.max_context_length
        ), f"Found {padded_inputs[0].shape[1]} != {self.max_context_length}"
        assert (
            padded_inputs[1].shape[1] == self.max_context_length
        ), f"Found {padded_inputs[1].shape[1]} != {self.max_context_length}"
        assert (
            padded_inputs[2].shape[1] == self.max_context_length
        ), f"Found {padded_inputs[2].shape[1]} != {self.max_context_length}"
        assert torch.equal(
            padded_inputs[3], self.test_inputs[3]
        ), f"Found {padded_inputs[3]} != {self.test_inputs[3]}"
        assert "Truncating" in str(context.warning)

    def test_pad_inputs_allow_prompt_truncation(self):
        # Test warning thrown and attention_mask truncated with allow_input_truncation=True

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.neuron_config.allow_input_truncation = True

        with self.assertWarns(UserWarning) as context:
            padded_inputs = model_wrapper.pad_inputs(*self.test_inputs, pad_type="max")

        assert (
            padded_inputs[1].shape[1] == self.max_length
        ), f"Found {padded_inputs[1].shape[1]} != {self.max_length}"
        assert torch.equal(
            padded_inputs[3], self.test_inputs[3]
        ), f"Found {padded_inputs[3]} != {self.test_inputs[3]}"
        assert "Truncating" in str(context.warning)

    def test_pad_inputs_with_first_fit_strategy_allow_prompt_truncation(self):
        # Test warning thrown and attention_mask truncated with allow_input_truncation=True

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.neuron_config.allow_input_truncation = True
        model_wrapper.neuron_config.buckets = [2, 5, 7]

        with self.assertWarns(UserWarning) as context:
            padded_inputs = model_wrapper.pad_inputs(*self.test_inputs)

        assert (
            padded_inputs[1].shape[1] == self.max_length
        ), f"Found {padded_inputs[1].shape[1]} != {self.max_length}"
        assert torch.equal(
            padded_inputs[3], self.test_inputs[3]
        ), f"Found {padded_inputs[3]} != {self.test_inputs[3]}"
        assert "Truncating" in str(context.warning)

    def test_pad_inputs_with_first_fit_strategy_disallow_prompt_truncation(self):
        # Test warning thrown and attention_mask truncated with allow_input_truncation=False

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.neuron_config.allow_input_truncation = False
        model_wrapper.neuron_config.buckets = [2, 5, 7]

        with self.assertRaises(ValueError) as context:
            model_wrapper.pad_inputs(*self.test_inputs)

        assert "exceeds largest bucket" in str(context.exception)

    def test_pad_inputs_with_second_fit_strategy_allow_prompt_truncation(self):
        # Test warning thrown and attention_mask truncated with allow_input_truncation=True

        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.neuron_config.allow_input_truncation = True
        model_wrapper.neuron_config.buckets = [2, 5, 7]

        with self.assertWarns(UserWarning) as context:
            padded_inputs = model_wrapper.pad_inputs(*self.test_inputs, pad_type="second_fit")

        assert (
            padded_inputs[1].shape[1] == self.max_length
        ), f"Found {padded_inputs[1].shape[1]} != {self.max_length}"
        assert torch.equal(
            padded_inputs[3], self.test_inputs[3]
        ), f"Found {padded_inputs[3]} != {self.test_inputs[3]}"
        assert "Truncating" in str(context.warning)

    def test_pad_inputs_with_second_fit_strategy_disallow_prompt_truncation(self):
        # Test warning thrown and attention_mask truncated with allow_input_truncation=False
        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.neuron_config.allow_input_truncation = False
        model_wrapper.neuron_config.buckets = [2, 5, 7]

        with self.assertRaises(ValueError) as context:
            model_wrapper.pad_inputs(*self.test_inputs, pad_type="second_fit")

        assert "exceeds largest bucket" in str(context.exception)
