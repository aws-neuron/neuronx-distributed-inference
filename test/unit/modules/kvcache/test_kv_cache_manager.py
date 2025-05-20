import unittest

import torch

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
                        "kv_cache_quant": False,
                        "tp_degree": 1,
                        "max_length": 10,
                        "batch_size": 2,
                        "k_cache_transposed": False,
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


if __name__ == "__main__":
    unittest.main()
