import pytest
import torch
from typing import Tuple, Dict, Any

from torch_xla.core import xla_model as xm

from neuronx_distributed_inference.experimental.functional import tokengen_attention_megakernel_standard_kv

torch.manual_seed(0)

class TestTokengenAttentionMegakernelStandardKV:
    @pytest.fixture
    def base_config(self):
        """Base configuration for tests"""
        return {
            'batch_size': 1,
            'seq_len': 1,
            'hidden_size': 4096,
            'head_dim': 128,
            'num_heads': 1,
            'num_kv_heads': 1,
            'max_seq_len': 4096,
            'dtype': torch.float16,
            'rmsnorm_eps': 1e-6,
            'fused_rmsnorm': True,
            'skip_rope': False,
            'use_qk_norm': False,
        }

    def create_standard_kv_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """Create test input tensors for standard KV implementation"""
        device = xm.xla_device()

        hidden_states = torch.randn(
            config['batch_size'], config['seq_len'], config['hidden_size'],
            dtype=config['dtype'], device=device
        )

        W_qkv = torch.randn(
            config['hidden_size'], 3 * config['num_heads'] * config['head_dim'],
            dtype=config['dtype'], device=device
        )

        W_out = torch.randn(
            config['num_heads'] * config['head_dim'], config['hidden_size'],
            dtype=config['dtype'], device=device
        )

        W_gamma = torch.ones(
            1, config['hidden_size'],
            dtype=config['dtype'], device=device
        )

        # Create K_cache with appropriate shape based on k_cache_transposed
        if config.get('k_cache_transposed', False):
            K_cache = torch.randn(
                config['batch_size'], config['num_kv_heads'], 
                config['head_dim'], config['max_seq_len'],
                dtype=config['dtype'], device=device
            )
        else:
            K_cache = torch.randn(
                config['batch_size'], config['num_kv_heads'], 
                config['max_seq_len'], config['head_dim'],
                dtype=config['dtype'], device=device
            )

        V_cache = torch.randn(
            config['batch_size'], config['num_kv_heads'], 
            config['max_seq_len'], config['head_dim'],
            dtype=config['dtype'], device=device
        )

        attention_mask = torch.ones(
            config['batch_size'], 1, config['seq_len'], config['max_seq_len'],
            dtype=torch.int32, device=device
        )

        active_mask = torch.ones(
            config['batch_size'], 1, config['seq_len'], config['seq_len'],
            dtype=torch.int32, device=device
        )

        position_ids = torch.zeros(
            config['batch_size'], config['seq_len'],
            dtype=torch.int32, device=device
        )

        return (
            hidden_states, W_qkv, W_out, W_gamma, K_cache, V_cache,
            attention_mask, active_mask, position_ids
        )

    @pytest.mark.parametrize(
        "config_updates",
        [
            {},  # Default configuration
            {'batch_size': 2, 'seq_len': 2},  # Multi-batch, multi-sequence
            {'k_cache_transposed': True},  # Transposed K cache
            {'use_qk_norm': True},  # With QK normalization
            {'update_cache_in_kernel': True}, # Update cache in kernel
        ]
    )
    def test_standard_kv_configurations(self, base_config, config_updates):
        """Test different configurations for standard KV implementation"""
        config = {**base_config, **config_updates}
        
        try:
            inputs = self.create_standard_kv_inputs(config)
            
            output = tokengen_attention_megakernel_standard_kv(
                *inputs,
                rmsnorm_eps=config['rmsnorm_eps'],
                head_dim=config['head_dim'],
                num_heads=config['num_heads'],
                num_kv_heads=config['num_kv_heads'],
                k_cache_transposed=config.get('k_cache_transposed', False),
                update_cache_in_kernel=config.get('update_cache_in_kernel', True),
                fused_rmsnorm=config['fused_rmsnorm'],
                skip_rope=config['skip_rope'],
                use_qk_norm=config['use_qk_norm'],
            )
            
            self._validate_standard_kv_output(output, config)
            
        except Exception as e:
            pytest.fail(f"Standard KV configuration failed with error: {str(e)}")


    def _validate_standard_kv_output(self, output, config):
        """Validate output from standard KV implementation"""
        attn_output, (K, V), cos_cache, sin_cache = output
        
        # Check shapes
        expected_output_shape = (
            config['batch_size'], 
            config['seq_len'], 
            config['hidden_size']
        )
        assert attn_output.shape == expected_output_shape
        
        # Check K cache shape
        if config.get('k_cache_transposed', False):
            expected_k_shape = (
                config['batch_size'], 
                config['num_kv_heads'], 
                config['head_dim'], 
                config['max_seq_len'] if config.get('update_cache_in_kernel', True) else 1
            )
        else:
            expected_k_shape = (
                config['batch_size'], 
                config['num_kv_heads'], 
                config['max_seq_len'] if config.get('update_cache_in_kernel', True) else 1,
                config['head_dim']
            )
        
        assert K.shape == expected_k_shape


    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16]
    )
    def test_dtypes(self, base_config, dtype):
        """Test different data types for both implementations"""
        config = {**base_config, 'dtype': dtype}
        
        # Test standard KV implementation
        try:
            inputs = self.create_standard_kv_inputs(config)
            output = tokengen_attention_megakernel_standard_kv(
                *inputs,
                rmsnorm_eps=config['rmsnorm_eps'],
                head_dim=config['head_dim'],
                num_heads=config['num_heads'],
                num_kv_heads=config['num_kv_heads']
            )
            assert output[0].dtype == dtype
        except Exception as e:
            pytest.fail(f"Standard KV dtype {dtype} failed with error: {str(e)}")
