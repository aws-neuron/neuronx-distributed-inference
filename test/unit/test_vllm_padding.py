import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn.functional as F
import sys
from io import StringIO
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import ( 
    ModelWrapper,
)
CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
class TestVllmCteRepadding(unittest.TestCase):
    def setUp(self):
        self.held, sys.stdout = sys.stdout, StringIO()
        self.model_cls = NeuronBaseModel
        self.max_context_length = 256
        self.max_length = 256
        self.pad_token_id = 0
        self.buckets=[32,64,128,256]
        self.tensor_pad_vals = [self.pad_token_id, 0, 1]
        self.config = InferenceConfig(
            max_context_length=self.max_context_length,
            pad_token_id=self.pad_token_id,
            neuron_config=NeuronConfig(
                max_context_length=self.max_context_length,
                max_length=self.max_length,
                buckets=self.buckets,
            ),
        )
    def tearDown(self):
        self.result = sys.stdout.getvalue()
        sys.stdout = self.held
        print(self.result)
    
    def test_vllm_cte_repadding(self):
        # model wrapper initialization
        model_wrapper = ModelWrapper(config=self.config, model_cls=self.model_cls)
        model_wrapper.tag = CONTEXT_ENCODING_MODEL_TAG

        # Prepare test inputs
        batch_size = 4
        seq_length_list = [10, 20, 40, 100]
        expected_result_len = [32, 32, 64,128]
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []

        for i in range(batch_size):
            seq_length = seq_length_list[i]
            input_ids = torch.randint(0, 1000, (1, seq_length))
            attention_mask = torch.ones((1, seq_length))
            position_ids = torch.arange(seq_length).unsqueeze(0)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
        
        max_seq_length = max(seq_length_list)

        padded_input_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        padded_attention_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        padded_position_ids = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

        for i in range(batch_size):
            seq_length = seq_length_list[i]
            padded_input_ids[i, :seq_length] = input_ids_list[i]
            padded_attention_mask[i, :seq_length] = attention_mask_list[i]
            padded_position_ids[i, :seq_length] = position_ids_list[i]
        
        args = (padded_input_ids, padded_attention_mask, padded_position_ids)

        # Apply pad_inputs method
        model_wrapper.pad_inputs(*args)

        # Test the batch processing
        for i in range(batch_size):
            single_batch_args = [arg[i:i+1] for arg in args]
            
            # Call vllm_cte_repadding for test
            result = model_wrapper.vllm_cte_repadding(single_batch_args)

            # Check num of args passed out
            self.assertEqual(len(result), 3)

            # Check result shape, value
            # In addition, the input_ids is padded with pad_token_id
            # attention_mask is padded with 0
            # position id is padded with 1

            for j in range(3):
                self.assertEqual(result[j].shape, (1, expected_result_len[i]))
                self.assertTrue(torch.equal(result[j][:, :seq_length_list[i]], single_batch_args[j][:, :seq_length_list[i]]))
                self.assertEqual(torch.sum(result[j]), torch.sum(single_batch_args[j]) + (self.tensor_pad_vals[j] * (result[j].shape[1] - seq_length_list[i])))


if __name__ == '__main__':
    unittest.main()