import pytest
import torch
from typing import Tuple, Dict, Any

from torch_xla.core import xla_model as xm

from neuronx_distributed_inference.experimental.functional import tokengen_attention_megakernel_block_kv

torch.manual_seed(0)

class TestTokengenAttentionMegakernelBlockKV:
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


    def create_block_kv_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """Create test input tensors for blocked KV implementation"""
        device = xm.xla_device()
        
        # Most inputs are the same as standard KV
        base_inputs = list(self.create_standard_kv_inputs(config))
        
        # Replace K_cache and V_cache with blocked versions
        block_size = config.get('paged_attention_block_size', 16)
        num_blocks = (config['max_seq_len'] + block_size - 1) // block_size
        
        K_cache = torch.randn(
            num_blocks, block_size, config['num_kv_heads'], config['head_dim'],
            dtype=config['dtype'], device=device
        )
        
        V_cache = torch.randn(
            num_blocks, block_size, config['num_kv_heads'], config['head_dim'],
            dtype=config['dtype'], device=device
        )
        
        # Replace in base inputs
        base_inputs[4] = K_cache
        base_inputs[5] = V_cache
        
        return tuple(base_inputs)


    @pytest.mark.parametrize(
        "config_updates",
        [
            {},  # Default configuration
            {'batch_size': 2, 'seq_len': 2},  # Multi-batch, multi-sequence
            {'paged_attention_block_size': 32},  # Different block size
        ]
    )
    def test_block_kv_configurations(self, base_config, config_updates):
        """Test different configurations for blocked KV implementation"""
        config = {**base_config, **config_updates, 'num_kv_heads': 1}  # Force num_kv_heads=1 for blocked implementation
        
        try:
            inputs = self.create_block_kv_inputs(config)
            
            # Create active block table
            device = xm.xla_device()
            active_block_table = torch.zeros(
                config['batch_size'], config['max_seq_len']//config.get('paged_attention_block_size', 16),
                dtype=torch.int32, device=device
            )
            
            output = tokengen_attention_megakernel_block_kv(
                *inputs,
                rmsnorm_eps=config['rmsnorm_eps'],
                head_dim=config['head_dim'],
                num_heads=config['num_heads'],
                num_kv_heads=config['num_kv_heads'],
                active_block_table=active_block_table,
                paged_attention_block_size=config.get('paged_attention_block_size', 16),
                fused_rmsnorm=config['fused_rmsnorm'],
                skip_rope=config['skip_rope'],
                use_qk_norm=config['use_qk_norm'],
            )
            
            self._validate_block_kv_output(output, config)
            
        except Exception as e:
            pytest.fail(f"Blocked KV configuration failed with error: {str(e)}")


    def _validate_block_kv_output(self, output, config):
        """Validate output from blocked KV implementation"""
        attn_output, (K, V), cos_cache, sin_cache = output
        
        # Check shapes
        expected_output_shape = (
            config['batch_size'], 
            config['seq_len'], 
            config['hidden_size']
        )
        assert attn_output.shape == expected_output_shape
        
        block_size = config.get('paged_attention_block_size', 16)
        num_blocks = (config['max_seq_len'] + block_size - 1) // block_size
        
        expected_cache_shape = (
            num_blocks,
            block_size,
            config['num_kv_heads'],
            config['head_dim']
        )
        assert K.shape == expected_cache_shape
        assert V.shape == expected_cache_shape


    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16]
    )
    def test_dtypes(self, base_config, dtype):
        """Test different data types for both implementations"""
        config = {**base_config, 'dtype': dtype}
        
        # Test block KV implementation
        try:
            config['num_kv_heads'] = 1  # Force num_kv_heads=1 for block implementation
            inputs = self.create_block_kv_inputs(config)
            active_block_table = torch.zeros(
                config['batch_size'], config['max_seq_len']//config.get('paged_attention_block_size', 16),
                dtype=torch.int32, device=xm.xla_device()
            )
            output = tokengen_attention_megakernel_block_kv(
                *inputs,
                rmsnorm_eps=config['rmsnorm_eps'],
                head_dim=config['head_dim'],
                num_heads=config['num_heads'],
                num_kv_heads=config['num_kv_heads'],
                active_block_table=active_block_table,
                paged_attention_block_size=16
            )
            assert output[0].dtype == dtype
        except Exception as e:
            pytest.fail(f"Block KV dtype {dtype} failed with error: {str(e)}")
