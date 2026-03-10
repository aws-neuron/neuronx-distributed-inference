import pytest
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group


class TestCalculateNumCoresPerGroup:
    """Test suite for calculate_num_cores_per_group function"""

    def test_grouped_query_attention(self):
        """Test with GQA where num_kv_heads < num_attn_heads"""
        result = calculate_num_cores_per_group(num_attn_heads=32, num_kv_heads=4, tp_degree=8)
        assert result == 2  # min(8, 32) / 4 = 2

    def test_multiquery_attention(self):
        """Test with MQA where num_kv_heads = 1"""
        result = calculate_num_cores_per_group(num_attn_heads=32, num_kv_heads=1, tp_degree=8)
        assert result == 8  # min(8, 32) / 1 = 8

    def test_tp_larger_than_attn_heads(self):
        """Test when tp_degree > num_attn_heads"""
        result = calculate_num_cores_per_group(num_attn_heads=16, num_kv_heads=8, tp_degree=32)
        assert result == 2  # min(32, 16) / 8 = 2
