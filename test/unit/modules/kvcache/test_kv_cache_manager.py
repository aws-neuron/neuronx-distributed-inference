import unittest

import torch
import os
from parameterized import parameterized

from unittest.mock import patch
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager


class TestKVCacheManager(unittest.TestCase):
    def setUp(self):

        class MockConfig:
            def __init__(self):
                self.neuron_config = type(
                    "NeuronConfig",
                    (),
                    {
                        "is_medusa": False,
                        "num_medusa_heads": 0,
                        "padding_side": "right",
                        "is_continuous_batching": True,
                        "flash_decoding_enabled": False,
                        "kv_cache_batch_size": 6,
                        "kv_cache_padding_size": 1,
                        "kv_cache_tiling": False,
                        "torch_dtype": torch.float32,
                        "attention_dtype": torch.float32,
                        "kv_cache_quant": False,
                        "tp_degree": 1,
                        "cp_degree": 1,
                        "attention_dp_degree": 1,
                        "max_length": 10,
                        "batch_size": 2,
                        "k_cache_transposed": False,
                        "apply_seq_ids_mask": False,
                        "is_prefill_stage": True,
                    },
                )
                self.num_cores_per_group = 1
                self.num_hidden_layers = 1
                self.hidden_size = 32
                self.num_attention_heads = 4  # head_dim = 32/4=>8

        self.config = MockConfig()
        self.kv_cache_manager = KVCacheManager(config=self.config, num_kv_head=4)

    def test_update_cache_smaller_batch_size(self):
        # Test case where batch_size (2) < kv_cache_batch_size (4)
        batch_size = 2
        seq_len = 10
        active_seq_len = 3
        head_dim = 8
        num_kv_heads = 4

        # Create sample inputs
        seq_ids = torch.tensor([0, 2], dtype=torch.int32)  # Update sequences 0 and 2
        position_ids = torch.tensor([[5, 6, 7], [2, 3, 4]], dtype=torch.int32)

        # Create new key values to be updated
        new_k = torch.ones(batch_size, num_kv_heads, active_seq_len, head_dim)
        new_v = torch.ones(batch_size, num_kv_heads, active_seq_len, head_dim) * 2
        new_key_values = [[new_k, new_v]]

        # Update cache
        updated_cache = self.kv_cache_manager.update_cache(
            is_for_context_encoding=False,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=new_key_values,
            seq_len=seq_len,
        )

        # Verify results
        updated_k = updated_cache[0]
        updated_v = updated_cache[1]

        # Check shape
        expected_shape = (7, num_kv_heads, seq_len, head_dim)  # 6 + 1 padding
        self.assertEqual(updated_k.shape, expected_shape)
        self.assertEqual(updated_v.shape, expected_shape)
        # Check values for updated sequences

        for kv_head in range(num_kv_heads):
            # Seq 0 should be updated for all the kv heads with matching position ids
            self.assertTrue(
                torch.all(updated_k[0][kv_head][5] == 1)
                and torch.all(updated_k[0][kv_head][6] == 1)
                and torch.all(updated_k[0][kv_head][7] == 1)
            )
            # Seq 0 should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_k[0][kv_head][0] == 0)
                and torch.all(updated_k[0][kv_head][1] == 0)
                and torch.all(updated_k[0][kv_head][2] == 0)
            )
            self.assertTrue(
                torch.all(updated_k[0][kv_head][3] == 0)
                and torch.all(updated_k[0][kv_head][4] == 0)
                and torch.all(updated_k[0][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_k[0][kv_head][9] == 0))

            # Seq 2 should be updated at the correct posids
            self.assertTrue(
                torch.all(updated_k[2][kv_head][2] == 1)
                and torch.all(updated_k[2][kv_head][3] == 1)
                and torch.all(updated_k[2][kv_head][4] == 1)
            )
            # Seq 2 should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_k[2][kv_head][0] == 0)
                and torch.all(updated_k[2][kv_head][1] == 0)
                and torch.all(updated_k[2][kv_head][5] == 0)
            )
            self.assertTrue(
                torch.all(updated_k[2][kv_head][6] == 0)
                and torch.all(updated_k[2][kv_head][7] == 0)
                and torch.all(updated_k[2][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_k[2][kv_head][9] == 0))

            self.assertTrue(torch.all(updated_k[1][kv_head] == 0))  # Seq 1 should remain zero
            self.assertTrue(torch.all(updated_k[3][kv_head] == 0))  # Seq 3 should remain zero
            self.assertTrue(torch.all(updated_k[4][kv_head] == 0))  # Seq 4 should remain zero
            self.assertTrue(torch.all(updated_k[5][kv_head] == 0))  # Seq 5 should remain zero
            self.assertTrue(torch.all(updated_k[6][kv_head] == 0))  # padding should remain zero

            # Similar checks for values
            self.assertTrue(
                torch.all(updated_v[0][kv_head][5] == 2)
                and torch.all(updated_v[0][kv_head][6] == 2)
                and torch.all(updated_v[0][kv_head][7] == 2)
            )
            # Seq 0 should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_v[0][kv_head][0] == 0)
                and torch.all(updated_v[0][kv_head][1] == 0)
                and torch.all(updated_v[0][kv_head][2] == 0)
            )
            self.assertTrue(
                torch.all(updated_v[0][kv_head][3] == 0)
                and torch.all(updated_v[0][kv_head][4] == 0)
                and torch.all(updated_v[0][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_v[0][kv_head][9] == 0))

            self.assertTrue(
                torch.all(updated_v[2][kv_head][2] == 2)
                and torch.all(updated_v[2][kv_head][3] == 2)
                and torch.all(updated_v[2][kv_head][4] == 2)
            )
            # Seq 2 should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_v[2][kv_head][0] == 0)
                and torch.all(updated_v[2][kv_head][1] == 0)
                and torch.all(updated_v[2][kv_head][5] == 0)
            )
            self.assertTrue(
                torch.all(updated_v[2][kv_head][6] == 0)
                and torch.all(updated_v[2][kv_head][7] == 0)
                and torch.all(updated_v[2][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_v[2][kv_head][9] == 0))

            self.assertTrue(torch.all(updated_v[1][kv_head] == 0))  # Seq 1 should remain zero
            self.assertTrue(torch.all(updated_v[3][kv_head] == 0))  # Seq 3 should remain zero
            self.assertTrue(torch.all(updated_v[4][kv_head] == 0))  # Seq 4 should remain zero
            self.assertTrue(torch.all(updated_v[5][kv_head] == 0))  # Seq 5 should remain zero
            self.assertTrue(torch.all(updated_v[6][kv_head] == 0))  # padding should remain zero

    def test_update_cache_invalid_seq_ids(self):
        # Test with invalid sequence IDs
        batch_size = 4
        seq_len = 10
        active_seq_len = 3
        head_dim = 8
        num_kv_heads = 4

        # Create sample inputs
        seq_ids = torch.tensor([1, 10, 16, 150], dtype=torch.int32)  # Update sequences 0 and 2
        position_ids = torch.tensor([[5, 6, 7], [2, 3, 4], [2, 3, 4], [3, 4, 5]], dtype=torch.int32)

        new_k = torch.ones(batch_size, num_kv_heads, active_seq_len, head_dim)
        new_v = torch.ones(batch_size, num_kv_heads, active_seq_len, head_dim) * 2
        new_key_values = [[new_k, new_v]]

        # Update should handle invalid seq_id gracefully
        updated_cache = self.kv_cache_manager.update_cache(
            is_for_context_encoding=False,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=new_key_values,
            seq_len=seq_len,
        )
        # Verify results
        updated_k = updated_cache[0]
        updated_v = updated_cache[1]

        # Check shape
        expected_shape = (7, num_kv_heads, seq_len, head_dim)  # 6 + 1 padding
        self.assertEqual(updated_k.shape, expected_shape)
        self.assertEqual(updated_v.shape, expected_shape)
        # Check values for updated sequences

        for kv_head in range(num_kv_heads):
            # Seq 1 should be updated for all the kv heads with matching position ids
            self.assertTrue(
                torch.all(updated_k[1][kv_head][5] == 1)
                and torch.all(updated_k[1][kv_head][6] == 1)
                and torch.all(updated_k[1][kv_head][7] == 1)
            )
            # Seq 1 should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_k[1][kv_head][0] == 0)
                and torch.all(updated_k[1][kv_head][1] == 0)
                and torch.all(updated_k[1][kv_head][2] == 0)
            )
            self.assertTrue(
                torch.all(updated_k[1][kv_head][3] == 0)
                and torch.all(updated_k[1][kv_head][4] == 0)
                and torch.all(updated_k[1][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_k[1][kv_head][9] == 0))

            # Seq 1 should be updated for all the kv heads with matching position ids
            self.assertTrue(
                torch.all(updated_v[1][kv_head][5] == 2)
                and torch.all(updated_v[1][kv_head][6] == 2)
                and torch.all(updated_v[1][kv_head][7] == 2)
            )
            # Seq 1 should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_v[1][kv_head][0] == 0)
                and torch.all(updated_v[1][kv_head][1] == 0)
                and torch.all(updated_v[1][kv_head][2] == 0)
            )
            self.assertTrue(
                torch.all(updated_v[1][kv_head][3] == 0)
                and torch.all(updated_v[1][kv_head][4] == 0)
                and torch.all(updated_v[1][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_v[1][kv_head][9] == 0))

            self.assertTrue(
                torch.all(updated_k[6][kv_head][3] == 1)
                and torch.all(updated_k[6][kv_head][4] == 1)
                and torch.all(updated_k[6][kv_head][5] == 1)
            )
            # Seq 6 Padded seq should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_k[6][kv_head][0] == 0)
                and torch.all(updated_k[6][kv_head][1] == 0)
                and torch.all(updated_k[6][kv_head][2] == 0)
            )
            self.assertTrue(
                torch.all(updated_k[6][kv_head][6] == 0)
                and torch.all(updated_k[6][kv_head][7] == 0)
                and torch.all(updated_k[6][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_k[6][kv_head][9] == 0))

            self.assertTrue(
                torch.all(updated_v[6][kv_head][3] == 2)
                and torch.all(updated_v[6][kv_head][4] == 2)
                and torch.all(updated_v[6][kv_head][5] == 2)
            )
            # Seq 6 Padded seq should not be updated for all the kv heads with not matching position ids
            self.assertTrue(
                torch.all(updated_v[6][kv_head][0] == 0)
                and torch.all(updated_v[6][kv_head][1] == 0)
                and torch.all(updated_v[6][kv_head][2] == 0)
            )
            self.assertTrue(
                torch.all(updated_v[6][kv_head][6] == 0)
                and torch.all(updated_v[6][kv_head][7] == 0)
                and torch.all(updated_v[6][kv_head][8] == 0)
            )
            self.assertTrue(torch.all(updated_v[6][kv_head][9] == 0))

            self.assertTrue(torch.all(updated_k[0][kv_head] == 0))  # Seq 1 should remain zero
            self.assertTrue(torch.all(updated_k[3][kv_head] == 0))  # Seq 3 should remain zero
            self.assertTrue(torch.all(updated_k[4][kv_head] == 0))  # Seq 4 should remain zero
            self.assertTrue(torch.all(updated_k[5][kv_head] == 0))  # Seq 5 should remain zero

            self.assertTrue(torch.all(updated_v[0][kv_head] == 0))  # Seq 1 should remain zero
            self.assertTrue(torch.all(updated_v[3][kv_head] == 0))  # Seq 3 should remain zero
            self.assertTrue(torch.all(updated_v[4][kv_head] == 0))  # Seq 4 should remain zero
            self.assertTrue(torch.all(updated_v[5][kv_head] == 0))  # Seq 5 should remain zero
    
    def test_update_cache_context_parallel_full_tp_decode(self):
        tp_degree = 32
        cp_degree = 4
        batch_size = 1
        seq_len = 10
        active_seq_len = 1
        head_dim = 8
        num_kv_heads = 16
        
        self.config.neuron_config.cp_degree = cp_degree
        self.config.neuron_config.tp_degree = tp_degree
        self.config.neuron_config.is_prefill_stage = True
        self.config.neuron_config.kv_cache_batch_size = batch_size
        self.config.neuron_config.batch_size = batch_size
        self.config.neuron_config.is_continuous_batching = False

        # set cpu-mode to true to use the CPU version of fill_prefix
        os.environ["NXD_CPU_MODE"] = "1"
        
        class MockRank:
            def get_rank(self):
                return torch.tensor(0, dtype=torch.int32)
        
        kv_cache_manager = KVCacheManager(config=self.config, num_kv_head=num_kv_heads, global_rank=MockRank())
        
        k_cache = torch.zeros((batch_size, 1, seq_len, head_dim), dtype=torch.float32)
        v_cache = torch.zeros((batch_size, 1, seq_len, head_dim), dtype=torch.float32)
        
        # Create new key values to be updated, each head will have distinct values so we can tell which head we updated
        latest_k = torch.ones(batch_size, 2, active_seq_len, head_dim, dtype=torch.float32)
        latest_k[:, 0:1, :, :] = latest_k[:, 0:1, :, :] * 5
        latest_k[:, 1:2, :, :] = latest_k[:, 1:2, :, :] * 10

        latest_v = torch.ones(batch_size, 2, active_seq_len, head_dim, dtype=torch.float32)
        latest_v[:, 0:1, :, :] = latest_v[:, 0:1, :, :] * 5
        latest_v[:, 1:2, :, :] = latest_v[:, 1:2, :, :] * 10

        
        seq_ids = torch.tensor([0], dtype=torch.int32)
        position_ids = torch.tensor([[0]], dtype=torch.long)
        
        def mock_get_indices(*args, **kwargs):
            return torch.cat([torch.zeros((16), dtype=torch.long), torch.ones((16), dtype=torch.long)])
        
        with patch('neuronx_distributed_inference.modules.kvcache.kv_cache_manager.get_kv_head_indices_context_parallel_full_tp_decode', 
                side_effect=mock_get_indices):
            
            updated_k, updated_v = kv_cache_manager.update_kv_by_layer_id(
                idx=0,
                is_for_context_encoding=True,
                seq_ids=seq_ids,
                position_ids=position_ids,
                kv_per_layer=(latest_k, latest_v),
                seq_len=seq_len,
                kvcache_buffer=[k_cache, v_cache]
            )
        
        # verify we only used head 0 for the update
        self.assertTrue(torch.all(updated_k[0, 0, 0:1] == 5))
        self.assertTrue(torch.all(updated_v[0, 0, 0:1] == 5))

            
    @parameterized.expand(
        [
            # Test case 1: seq 0 in chunk 0, seq 1 in chunk 1
            ('test case 1', torch.tensor([[2], [4]], dtype=torch.int32), [4]),
            # Test case 2: both seqs in chunk 1
            ('test case 2', torch.tensor([[2], [2]], dtype=torch.int32), [4]),
            # Test case 3: seq 0 in chunk 0
            ('test case 3', torch.tensor([[0]], dtype=torch.int32), [4]),
            # Test case 4: NOPE layer. Position ids stay the same
            ('test case 4', torch.tensor([[2], [4]], dtype=torch.int32), [10]),
            # Test case 5: seq 0 in chunk 0, seq 1 in chunk 1, no mixed cache sizes
            ('test case 5', torch.tensor([[2], [4]], dtype=torch.int32), None),
        ]
    )    
    def test_chunked_attention_get_scatter_indices_for_cache_update_during_tkg(self, test_name, position_ids, cache_sizes):
        """Test get scatter indices for kv cache update during tkg for chunked attention"""
        # Setup
        os.environ["NXD_CPU_MODE"] = "1"

        # Test parameters
        batch_size = position_ids.shape[0]
        head_dim = 8
        num_kv_heads = 4
        attention_chunk_size = 4

        self.config.neuron_config.tp_degree = 1
        self.config.neuron_config.cp_degree = 1
        self.config.neuron_config.is_prefill_stage = False
        self.config.neuron_config.kv_cache_batch_size = batch_size
        self.config.neuron_config.batch_size = batch_size
        self.config.neuron_config.is_continuous_batching = False
        self.config.neuron_config.kv_cache_padding_size  = 0
        self.config.neuron_config.max_length = 10

        # Initialize kv cache manager
        kv_cache_manager = KVCacheManager(config=self.config, num_kv_head=num_kv_heads, attention_chunk_size=attention_chunk_size)
        if cache_sizes is not None:
            kv_cache_manager.v_shapes = [(batch_size, num_kv_heads, cache_sizes[0], head_dim)]

        latest_k = torch.zeros((batch_size, num_kv_heads, 1, head_dim))

        # Test the _get_index_to_update_new_position method 
        scatter_indices = kv_cache_manager._get_index_to_update_new_position(None, position_ids, latest_k, False, 0)

        # Check that scatter_indices have the correct shape
        self.assertEqual(scatter_indices.shape, (batch_size, num_kv_heads, 1, head_dim))

        if cache_sizes is not None:
            expected_indices = (position_ids % cache_sizes[0])[:, None, :, None].expand(batch_size, num_kv_heads, 1, head_dim)
        else:
            expected_indices = (position_ids % attention_chunk_size)[:, None, :, None].expand(batch_size, num_kv_heads, 1, head_dim)

        # Check that scatter_indices have the correct values
        self.assertTrue(torch.all(scatter_indices == expected_indices),
                       f"Mismatch for test {test_name}")
           
    

if __name__ == "__main__":
    unittest.main()
