import unittest
import torch
from parameterized import parameterized
from unittest.mock import patch

from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed.quantization.quantization_config import KVQuantizationConfig
from neuronx_distributed.quantization.quantization_config import QuantizationType

class TestKVCacheQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        
        class MockConfig:
            def __init__(self):
                self.neuron_config = type(
                    "NeuronConfig",
                    (),
                    {
                        "is_medusa": False,
                        "num_medusa_heads": 0,
                        "padding_side": "right",
                        "is_continuous_batching": False,
                        "flash_decoding_enabled": False,
                        "kv_cache_batch_size": 2,
                        "kv_cache_padding_size": 0,
                        "kv_cache_tiling": False,
                        "torch_dtype": torch.float32,
                        "attention_dtype": torch.float32,
                        "kv_cache_quant": False,
                        "kv_quant_config": None,
                        "tp_degree": 1,
                        "cp_degree": 1,
                        "attention_dp_degree": 1,
                        "max_length": 128,
                        "batch_size": 2,
                        "max_batch_size": 2,
                        "k_cache_transposed": False,
                        "apply_seq_ids_mask": False,
                        "is_prefill_stage": True,
                        "attn_tkg_builtin_kernel_enabled": False,
                        "attn_tkg_nki_kernel_enabled": False,
                        "attn_block_tkg_nki_kernel_enabled": False,
                        "kv_cache_update_with_kernel": False,
                        "logical_nc_config": 1,
                        "switch_cc": False,
                        "token_generation_batches": None,
                    },
                )
                self.num_cores_per_group = 1
                self.num_hidden_layers = 2
                self.hidden_size = 64
                self.num_attention_heads = 8
                
        self.config = MockConfig()

    def test_quantization_config_initialization(self):
        # Test default configuration
        config = KVQuantizationConfig()
        self.assertEqual(config.k_quant_method, QuantizationType.PER_TENSOR_SYMMETRIC)
        self.assertEqual(config.v_quant_method, QuantizationType.PER_TENSOR_SYMMETRIC)
        self.assertEqual(config.quant_dtype, torch.float8_e4m3fn)
        self.assertTrue(config.direct_cast)
        
        # Test custom configuration
        config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            v_quant_method=QuantizationType.PER_CHANNEL_SYMMETRIC,
            quant_dtype=torch.bfloat16,
            direct_cast=False
        )
        self.assertEqual(config.k_quant_method, QuantizationType.PER_TENSOR_SYMMETRIC)
        self.assertEqual(config.v_quant_method, QuantizationType.PER_CHANNEL_SYMMETRIC)
        self.assertEqual(config.quant_dtype, torch.bfloat16)
        self.assertFalse(config.direct_cast)

    def test_kv_cache_manager_without_quantization(self):
        manager = KVCacheManager(config=self.config, num_kv_head=8)

        self.assertEqual(manager.cache_dtype, torch.float32)
        self.assertEqual(len(manager.past_key_values), 4)  # 2 layers * 2 (K and V)
        
        for cache in manager.past_key_values:
            self.assertEqual(cache.dtype, torch.float32)

    @parameterized.expand([
        (torch.float8_e4m3fn,),
        (torch.bfloat16,),
        (torch.float16,),
    ])
    def test_direct_cast_quantization(self, quant_dtype):
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            v_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            quant_dtype=quant_dtype,
            direct_cast=True
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config

        manager = KVCacheManager(config=self.config, num_kv_head=8)

        self.assertEqual(manager.cache_dtype, quant_dtype)
        for cache in manager.past_key_values:
            self.assertEqual(cache.dtype, quant_dtype)

        test_tensor = torch.randn(2, 8, 10, 8, dtype=torch.float32)

        quantized = manager._quantize_cache(test_tensor, layer_idx=0, is_key=True)
        self.assertEqual(quantized.dtype, quant_dtype)

        dequantized = manager._dequantize_cache(quantized, layer_idx=0, is_key=True)
        self.assertEqual(dequantized.dtype, torch.float32)
        
        # For direct cast, values should be relatively close (depending on dtype precision)
        if quant_dtype != torch.float8_e4m3fn:
            # Skip float8 since it'll have much larger errors than fp16 / bf16
            torch.testing.assert_close(dequantized, test_tensor, rtol=1e-2, atol=1e-2)

    def test_per_tensor_quantization(self):
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            v_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            quant_dtype=torch.bfloat16,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        manager = KVCacheManager(config=self.config, num_kv_head=8)

        scale_value = 2.0
        manager.k_scales[0].data.fill_(scale_value)
    
        batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 8
        test_tensor = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32) * 10.0
    
        quantized = manager._quantize_cache(test_tensor, layer_idx=0, is_key=True)
        
        # Verify that all values are scaled by the same factor
        expected_quantized = (test_tensor / scale_value).to(torch.bfloat16)
        torch.testing.assert_close(quantized, expected_quantized, rtol=1e-2, atol=1e-2)

        dequantized = manager._dequantize_cache(quantized, layer_idx=0, is_key=True)

        torch.testing.assert_close(dequantized, test_tensor, rtol=1e-2, atol=1e-2)

    def test_per_key_quantization(self):
        
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_KEY_SYMMETRIC,
            v_quant_method=QuantizationType.PER_KEY_SYMMETRIC,
            quant_dtype=torch.bfloat16,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        num_heads_per_rank = 4
        with patch.object(KVCacheManager, '_get_num_kv_heads_per_rank', return_value=num_heads_per_rank):
            manager = KVCacheManager(config=self.config, num_kv_head=8)
        
        # Set different scale values for each head
        scale_values = torch.tensor([1.0, 2.0, 0.5, 4.0]).reshape(num_heads_per_rank, 1, 1)
        manager.k_scales[0].data = scale_values

        batch_size, seq_len, head_dim = 2, 10, 8
        test_tensor = torch.ones(batch_size, num_heads_per_rank, seq_len, head_dim, dtype=torch.float32) * 8.0

        quantized = manager._quantize_cache(test_tensor, layer_idx=0, is_key=True)

        for head_idx in range(num_heads_per_rank):
            head_scale = scale_values[head_idx, 0, 0].item()
            expected_value = (8.0 / head_scale)
            # Check that all values for this head are scaled correctly
            actual_values = quantized[:, head_idx, :, :].to(torch.float32)
            expected_tensor = torch.full_like(actual_values, expected_value)
            torch.testing.assert_close(actual_values, expected_tensor, rtol=1e-2, atol=1e-2)

        dequantized = manager._dequantize_cache(quantized, layer_idx=0, is_key=True)
        torch.testing.assert_close(dequantized, test_tensor, rtol=1e-2, atol=1e-2)

    def test_per_channel_quantization(self):
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_CHANNEL_SYMMETRIC,
            v_quant_method=QuantizationType.PER_CHANNEL_SYMMETRIC,
            quant_dtype=torch.bfloat16,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        head_dim = 8
        with patch.object(KVCacheManager, '_get_hidden_dim_per_head', return_value=head_dim):
            manager = KVCacheManager(config=self.config, num_kv_head=8)

        scale_values = torch.tensor([1.0, 2.0, 0.5, 4.0, 1.5, 3.0, 0.75, 2.5]).reshape(1, 1, head_dim)
        manager.k_scales[0].data = scale_values

        batch_size, num_heads, seq_len = 2, 4, 10
        test_tensor = torch.ones(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32) * 12.0
    
        quantized = manager._quantize_cache(test_tensor, layer_idx=0, is_key=True)
        
        # Verify that each channel is scaled by its corresponding scale
        for channel_idx in range(head_dim):
            channel_scale = scale_values[0, 0, channel_idx].item()
            expected_value = (12.0 / channel_scale)

            actual_values = quantized[:, :, :, channel_idx].to(torch.float32)
            expected_tensor = torch.full_like(actual_values, expected_value)
            torch.testing.assert_close(actual_values, expected_tensor, rtol=1e-2, atol=1e-2)

        dequantized = manager._dequantize_cache(quantized, layer_idx=0, is_key=True)
        torch.testing.assert_close(dequantized, test_tensor, rtol=1e-2, atol=1e-2)

    @parameterized.expand([
        (QuantizationType.PER_TENSOR_SYMMETRIC),
        (QuantizationType.PER_KEY_SYMMETRIC),
        (QuantizationType.PER_CHANNEL_SYMMETRIC),
    ])
    def test_quantization_method_with_random_data(self, quant_method):
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=quant_method,
            v_quant_method=quant_method,
            quant_dtype=torch.bfloat16,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        num_heads_per_rank = 4
        head_dim = 8
        with patch.object(KVCacheManager, '_get_num_kv_heads_per_rank', return_value=num_heads_per_rank):
            with patch.object(KVCacheManager, '_get_hidden_dim_per_head', return_value=head_dim):
                manager = KVCacheManager(config=self.config, num_kv_head=8)
        
        # Set random scale values based on the method
        if quant_method == QuantizationType.PER_TENSOR_SYMMETRIC:
            manager.k_scales[0].data = torch.tensor([2.5])
            manager.v_scales[0].data = torch.tensor([1.5])
        elif quant_method == QuantizationType.PER_KEY_SYMMETRIC:
            manager.k_scales[0].data = torch.rand(num_heads_per_rank, 1, 1) * 3 + 0.5
            manager.v_scales[0].data = torch.rand(num_heads_per_rank, 1, 1) * 3 + 0.5
        else:  # PER_CHANNEL
            manager.k_scales[0].data = torch.rand(1, 1, head_dim) * 3 + 0.5
            manager.v_scales[0].data = torch.rand(1, 1, head_dim) * 3 + 0.5

        batch_size, seq_len = 2, 10
        test_k = torch.randn(batch_size, num_heads_per_rank, seq_len, head_dim, dtype=torch.float32) * 5.0
        test_v = torch.randn(batch_size, num_heads_per_rank, seq_len, head_dim, dtype=torch.float32) * 5.0

        quantized_k = manager._quantize_cache(test_k, layer_idx=0, is_key=True)
        self.assertEqual(quantized_k.dtype, torch.bfloat16)
        dequantized_k = manager._dequantize_cache(quantized_k, layer_idx=0, is_key=True)

        quantized_v = manager._quantize_cache(test_v, layer_idx=0, is_key=False)
        self.assertEqual(quantized_v.dtype, torch.bfloat16)
        dequantized_v = manager._dequantize_cache(quantized_v, layer_idx=0, is_key=False)
        
        # Verify that values are recovered within acceptable error
        torch.testing.assert_close(dequantized_k, test_k, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(dequantized_v, test_v, rtol=5e-2, atol=5e-2)

        # Verify that quantization actually reduces precision (values should be different)
        self.assertFalse(torch.allclose(quantized_k.to(torch.float32) * manager.k_scales[0], test_k, rtol=1e-5))
        self.assertFalse(torch.allclose(quantized_v.to(torch.float32) * manager.v_scales[0], test_v, rtol=1e-5))

    def test_scale_based_quantization(self):
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            v_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            quant_dtype=torch.float8_e4m3fn,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        manager = KVCacheManager(config=self.config, num_kv_head=8)

        manager.k_scales[0].data.fill_(2.0)
        manager.v_scales[0].data.fill_(0.5)

        test_k = torch.tensor([[[[4.0, 8.0]]]], dtype=torch.float32)
        test_v = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float32)
        
        quantized_k = manager._quantize_cache(test_k, layer_idx=0, is_key=True)
        self.assertEqual(quantized_k.dtype, torch.float8_e4m3fn)
        
        dequantized_k = manager._dequantize_cache(quantized_k, layer_idx=0, is_key=True)
        self.assertEqual(dequantized_k.dtype, torch.float32)
        torch.testing.assert_close(dequantized_k, test_k, rtol=1e-1, atol=1e-1)
        
        quantized_v = manager._quantize_cache(test_v, layer_idx=0, is_key=False)
        self.assertEqual(quantized_v.dtype, torch.float8_e4m3fn)
        
        dequantized_v = manager._dequantize_cache(quantized_v, layer_idx=0, is_key=False)
        self.assertEqual(dequantized_v.dtype, torch.float32)
        torch.testing.assert_close(dequantized_v, test_v, rtol=1e-1, atol=1e-1)

    def test_quantization_in_update_kv(self):
        kv_quant_config = KVQuantizationConfig(
            quant_dtype=torch.bfloat16,
            direct_cast=True
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        manager = KVCacheManager(config=self.config, num_kv_head=8)
    
        batch_size = 2
        num_heads = 8
        seq_len = 10
        head_dim = 8
        
        seq_ids = torch.tensor([0, 1], dtype=torch.int32)
        position_ids = torch.tensor([[0], [0]], dtype=torch.int32)

        latest_k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
        latest_v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
        kv_per_layer = (latest_k, latest_v)
    
        def mock_fill_prefix(cache, prefix):
            cache[:, :, :prefix.shape[2], :] = prefix
            return cache

        with patch('neuronx_distributed_inference.modules.kvcache.kv_cache_manager.fill_prefix', side_effect=mock_fill_prefix):
            k_cache, v_cache = manager.update_kv_by_layer_id(
                idx=0,
                is_for_context_encoding=True,
                seq_ids=seq_ids,
                position_ids=position_ids,
                kv_per_layer=kv_per_layer,
                seq_len=seq_len,
            )
        
        # Verify that the cache contains quantized values
        self.assertEqual(k_cache.dtype, torch.bfloat16)
        self.assertEqual(v_cache.dtype, torch.bfloat16)

    def test_dequantization_in_get_kv(self):
        kv_quant_config = KVQuantizationConfig(
            quant_dtype=torch.float16,
            direct_cast=True
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        manager = KVCacheManager(config=self.config, num_kv_head=8)

        manager.past_key_values[0].data.fill_(1.0)  # K cache for layer 0
        manager.past_key_values[1].data.fill_(2.0)  # V cache for layer 0
    
        k_cache, v_cache = manager.get_kv_by_layer_id(
            idx=0,
            seq_len=10,
            skip_slice=True
        )
        
        # Verify dequantization happened (dtype should be float32)
        self.assertEqual(k_cache.dtype, torch.float32)
        self.assertEqual(v_cache.dtype, torch.float32)

        self.assertTrue(torch.all(k_cache == 1.0))
        self.assertTrue(torch.all(v_cache == 2.0))

    def test_mixed_quantization_methods(self):
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
            v_quant_method=QuantizationType.PER_CHANNEL_SYMMETRIC,
            quant_dtype=torch.float8_e4m3fn,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        with patch.object(KVCacheManager, '_get_num_kv_heads_per_rank', return_value=2):
            with patch.object(KVCacheManager, '_get_hidden_dim_per_head', return_value=128):
                manager = KVCacheManager(config=self.config, num_kv_head=8)
        
        # Check K scales (PER_TENSOR: single scale)
        for k_scale in manager.k_scales:
            self.assertEqual(k_scale.shape, (1,))
        
        # Check V scales (PER_CHANNEL: scale per head dimension)
        for v_scale in manager.v_scales:
            self.assertEqual(v_scale.shape, (1, 1, 128))

    def test_quantization_with_multiple_layers(self):
        self.config.num_hidden_layers = 4
        
        kv_quant_config = KVQuantizationConfig(
            k_quant_method=QuantizationType.PER_KEY_SYMMETRIC,
            v_quant_method=QuantizationType.PER_KEY_SYMMETRIC,
            quant_dtype=torch.bfloat16,
            direct_cast=False
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        with patch.object(KVCacheManager, '_get_num_kv_heads_per_rank', return_value=2):
            manager = KVCacheManager(config=self.config, num_kv_head=8)
        
        # Check that scales are created for all layers
        self.assertEqual(len(manager.k_scales), 4)
        self.assertEqual(len(manager.v_scales), 4)
        
        # Set different scales for each layer
        for i in range(4):
            manager.k_scales[i].data.fill_(float(i + 1))
            manager.v_scales[i].data.fill_(float(i + 0.5))
        
        # Test quantization for each layer
        test_tensor = torch.ones(2, 2, 10, 8, dtype=torch.float32) * 4.0
        
        for layer_idx in range(4):
            quantized_k = manager._quantize_cache(test_tensor, layer_idx, is_key=True)
            dequantized_k = manager._dequantize_cache(quantized_k, layer_idx, is_key=True)

            torch.testing.assert_close(dequantized_k, test_tensor, rtol=1e-2, atol=1e-2)

            quantized_v = manager._quantize_cache(test_tensor, layer_idx, is_key=False)
            dequantized_v = manager._dequantize_cache(quantized_v, layer_idx, is_key=False)
            
            torch.testing.assert_close(dequantized_v, test_tensor, rtol=1e-2, atol=1e-2)

    def test_quantization_with_continuous_batching(self):
        self.config.neuron_config.is_continuous_batching = True
        self.config.neuron_config.kv_cache_update_with_kernel = False
        
        kv_quant_config = KVQuantizationConfig(
            quant_dtype=torch.float16,
            direct_cast=True
        )
        self.config.neuron_config.kv_quant_config = kv_quant_config

        manager = KVCacheManager(config=self.config, num_kv_head=8)

        seq_ids = torch.tensor([0], dtype=torch.int32)
        position_ids = torch.tensor([[5]], dtype=torch.int32)
        
        latest_k = torch.randn(1, 8, 1, 8, dtype=torch.float32)
        latest_v = torch.randn(1, 8, 1, 8, dtype=torch.float32)
        kv_per_layer = (latest_k, latest_v)

        with patch('neuronx_distributed_inference.modules.kvcache.kv_cache_manager.update_cache_const_indices') as mock_update:
            mock_update.return_value = torch.zeros(2, 8, 128, 8, dtype=torch.float16)
            
            k_cache, v_cache = manager.update_kv_by_layer_id(
                idx=0,
                is_for_context_encoding=True,
                seq_ids=seq_ids,
                position_ids=position_ids,
                kv_per_layer=kv_per_layer,
                seq_len=10,
            )
            
            # Verify quantization was applied
            self.assertEqual(k_cache.dtype, torch.float16)
            self.assertEqual(v_cache.dtype, torch.float16)

    @parameterized.expand([
        (True, torch.float8_e4m3fn),
        (True, torch.bfloat16),
        (False, torch.float8_e4m3fn),
        (False, torch.float16),
    ])
    def test_end_to_end_quantization_flow(self, direct_cast, quant_dtype):
        # When direct_cast is True, we must use PER_TENSOR_SYMMETRIC
        if direct_cast:
            kv_quant_config = KVQuantizationConfig(
                k_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
                v_quant_method=QuantizationType.PER_TENSOR_SYMMETRIC,
                quant_dtype=quant_dtype,
                direct_cast=direct_cast
            )
        else:
            kv_quant_config = KVQuantizationConfig(
                k_quant_method=QuantizationType.PER_KEY_SYMMETRIC,
                v_quant_method=QuantizationType.PER_KEY_SYMMETRIC,
                quant_dtype=quant_dtype,
                direct_cast=direct_cast
            )
        self.config.neuron_config.kv_quant_config = kv_quant_config
        
        with patch.object(KVCacheManager, '_get_num_kv_heads_per_rank', return_value=2):
            manager = KVCacheManager(config=self.config, num_kv_head=8)
        
        if not direct_cast:
            for i in range(len(manager.k_scales)):
                manager.k_scales[i].data.fill_(1.0)
                manager.v_scales[i].data.fill_(1.0)
        
        batch_size = 2
        num_heads = 2
        seq_len = 10
        head_dim = 8
        
        seq_ids = torch.tensor([0, 1], dtype=torch.int32)
        position_ids = torch.tensor([[0], [0]], dtype=torch.int32)
        
        latest_k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32) * 2.0
        latest_v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32) * 2.0
        
        new_key_values = [
            [latest_k, latest_v],
            [latest_k.clone(), latest_v.clone()]
        ]

        def mock_fill_prefix(cache, prefix):
            cache[:, :, :prefix.shape[2], :] = prefix
            return cache

        with patch('neuronx_distributed_inference.modules.kvcache.kv_cache_manager.fill_prefix', side_effect=mock_fill_prefix):
            updated_cache = manager.update_cache(
                is_for_context_encoding=True,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=new_key_values,
                seq_len=seq_len,
            )

        for cache in updated_cache:
            self.assertEqual(cache.dtype, quant_dtype)

        past_key_values = manager.get_cache(seq_len=seq_len, skip_slice=True)
        
        for layer_kv in past_key_values:
            k_cache, v_cache = layer_kv
            # Should be dequantized back to float32
            self.assertEqual(k_cache.dtype, torch.float32)
            self.assertEqual(v_cache.dtype, torch.float32)

if __name__ == "__main__":
    unittest.main()
