"""
Minimal unit tests for NeuronFluxTransformer2DModel and related classes
Tests basic functionality using mocks to ensure they work on CPU without Neuron dependencies.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


@pytest.fixture
def mock_config():
    """Create a mock inference config"""
    neuron_config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        world_size=32,
        tp_degree=32,
    )
    
    config = SimpleNamespace(
        in_channels=16,
        num_attention_heads=18,
        attention_head_dim=64,
        num_layers=2,
        num_single_layers=2,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        patch_size=2,
        height=1024,
        width=1024,
        vae_scale_factor=8,
        neuron_config=neuron_config,
    )
    return config


class TestFluxBackboneInferenceConfig:
    """Test suite for FluxBackboneInferenceConfig"""
    
    def test_config_required_attributes(self):
        """Test that config has required attributes"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import FluxBackboneInferenceConfig
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.InferenceConfig.__init__'):
            config = FluxBackboneInferenceConfig()
            required_attrs = config.get_required_attributes()
            
            expected_attrs = [
                "attention_head_dim", "guidance_embeds", "in_channels",
                "joint_attention_dim", "num_attention_heads", "num_layers",
                "num_single_layers", "patch_size", "pooled_projection_dim",
                "height", "width",
            ]
            
            assert set(required_attrs) == set(expected_attrs)


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_attention_wrapper_sharded(self):
        """Test attention wrapper function"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import attention_wrapper_sharded_without_swap
        
        query = torch.randn(1, 18, 256, 64)
        key = torch.randn(1, 18, 256, 64)
        value = torch.randn(1, 18, 256, 64)
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._flash_fwd_call') as mock_flash:
            mock_flash.return_value = None
            output = attention_wrapper_sharded_without_swap(query, key, value)
            assert output is not None
            assert output.shape == (1, 18, 256, 64)

    def test_attention_wrapper_sharded_with_vc_size_2(self):
        """Test attention wrapper with VC size 2"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import attention_wrapper_sharded_without_swap
        
        query = torch.randn(1, 18, 256, 64)
        key = torch.randn(1, 18, 256, 64)
        value = torch.randn(1, 18, 256, 64)
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._flash_fwd_call') as mock_flash, \
             patch.dict('os.environ', {'NEURON_RT_VIRTUAL_CORE_SIZE': '2'}):
            mock_flash.return_value = None
            output = attention_wrapper_sharded_without_swap(query, key, value)
            assert output is not None


class TestModelWrapperFluxBackbone:
    """Test suite for ModelWrapperFluxBackbone"""
    
    def test_wrapper_input_generator(self, mock_config):
        """Test input generator"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import ModelWrapperFluxBackbone, NeuronFluxTransformer2DModel
        from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
        
        def mock_wrapper_init(self, *args, **kwargs):
            nn.Module.__init__(self)
        
        with patch.object(ModelWrapper, '__init__', mock_wrapper_init):
            wrapper = ModelWrapperFluxBackbone(config=mock_config, model_cls=NeuronFluxTransformer2DModel)
            wrapper.config = mock_config
        
            inputs = wrapper.input_generator()
            assert inputs is not None
            assert len(inputs) == 1
            assert len(inputs[0]) == 6

    def test_wrapper_get_model_instance(self, mock_config):
        """Test get model instance"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import ModelWrapperFluxBackbone, NeuronFluxTransformer2DModel
        from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
        
        def mock_wrapper_init(self, *args, **kwargs):
            nn.Module.__init__(self)
        
        with patch.object(ModelWrapper, '__init__', mock_wrapper_init), \
             patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.BaseModelInstance') as mock_instance:
            
            wrapper = ModelWrapperFluxBackbone(config=mock_config, model_cls=NeuronFluxTransformer2DModel)
            wrapper.config = mock_config
            wrapper.model_cls = NeuronFluxTransformer2DModel
            
            instance = wrapper.get_model_instance()
            assert instance is not None
            mock_instance.assert_called_once()

    def test_wrapper_forward_with_config(self, mock_config):
        """Test wrapper forward method with proper config setup"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import ModelWrapperFluxBackbone, NeuronFluxTransformer2DModel
        from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
        
        def mock_wrapper_init(self, *args, **kwargs):
            nn.Module.__init__(self)
        
        with patch.object(ModelWrapper, '__init__', mock_wrapper_init):
            wrapper = ModelWrapperFluxBackbone(config=mock_config, model_cls=NeuronFluxTransformer2DModel)
            wrapper.config = mock_config
            wrapper.model = MagicMock()
            wrapper._forward = MagicMock(return_value=torch.randn(1, 256, 16))
            
            output = wrapper.forward(
                hidden_states=torch.randn(1, 256, 16),
                encoder_hidden_states=torch.randn(1, 512, 4096),
                pooled_projections=torch.randn(1, 768),
                timestep=torch.tensor([500.0]),
                img_ids=torch.randn(4096, 3),
                txt_ids=torch.randn(512, 3),
                guidance=torch.tensor([3.5]),
            )
            assert output is not None

    def test_wrapper_forward_no_guidance(self, mock_config):
        """Test wrapper forward without guidance"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import ModelWrapperFluxBackbone, NeuronFluxTransformer2DModel
        from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
        
        def mock_wrapper_init(self, *args, **kwargs):
            nn.Module.__init__(self)
        
        mock_config.guidance_embeds = False
        with patch.object(ModelWrapper, '__init__', mock_wrapper_init):
            wrapper = ModelWrapperFluxBackbone(config=mock_config, model_cls=NeuronFluxTransformer2DModel)
            wrapper.config = mock_config
            wrapper.model = MagicMock()
            wrapper._forward = MagicMock(return_value=torch.randn(1, 256, 16))
            
            output = wrapper.forward(
                hidden_states=torch.randn(1, 256, 16),
                encoder_hidden_states=torch.randn(1, 512, 4096),
                pooled_projections=torch.randn(1, 768),
                timestep=torch.tensor([500.0]),
                img_ids=torch.randn(4096, 3),
                txt_ids=torch.randn(512, 3),
                guidance=None,
            )
            assert output is not None


class TestNeuronFluxBackboneApplication:
    """Test suite for NeuronFluxBackboneApplication"""
    
    def test_convert_hf_to_neuron_state_dict(self, mock_config):
        """Test state dict conversion"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxBackboneApplication
        
        state_dict = {
            'single_transformer_blocks.0.proj_out.weight': torch.randn(2304, 1152),
            'single_transformer_blocks.0.proj_out.bias': torch.randn(1152),
            'single_transformer_blocks.1.proj_out.weight': torch.randn(2304, 1152),
            'single_transformer_blocks.1.proj_out.bias': torch.randn(1152),
        }
        
        converted = NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(state_dict, mock_config)
        
        assert 'global_rank.rank' in converted
        assert 'single_transformer_blocks.0.proj_out_attn.weight' in converted
        assert 'single_transformer_blocks.0.proj_out_mlp.weight' in converted

    def test_update_state_dict_for_tied_weights(self):
        """Test update state dict for tied weights"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxBackboneApplication
        
        state_dict = {'test': torch.randn(10, 10)}
        NeuronFluxBackboneApplication.update_state_dict_for_tied_weights(state_dict)

    def test_get_compiler_args_trn1(self, mock_config):
        """Test compiler args for TRN1"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxBackboneApplication
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.NeuronApplicationBase.__init__'), \
             patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._HARDWARE', 'trn1'):
            
            app = NeuronFluxBackboneApplication.__new__(NeuronFluxBackboneApplication)
            app.config = mock_config
            app.context_parallel_enabled = True
            
            compiler_args = app.get_compiler_args()
            # The actual implementation uses -O1, not -O2 for TRN1
            assert '-O1' in compiler_args

    def test_get_compiler_args_trn2_context_parallel(self, mock_config):
        """Test compiler args for TRN2 with context parallel"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxBackboneApplication
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.NeuronApplicationBase.__init__'), \
             patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._HARDWARE', 'trn2'):
            
            app = NeuronFluxBackboneApplication.__new__(NeuronFluxBackboneApplication)
            app.config = mock_config
            app.context_parallel_enabled = True
            
            compiler_args = app.get_compiler_args()
            assert '--enable-ccop-compute-overlap' in compiler_args

    def test_get_compiler_args_trn2_no_context_parallel(self, mock_config):
        """Test compiler args for TRN2 without context parallel"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import NeuronFluxBackboneApplication
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.NeuronApplicationBase.__init__'), \
             patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._HARDWARE', 'trn2'):
            
            app = NeuronFluxBackboneApplication.__new__(NeuronFluxBackboneApplication)
            app.config = mock_config
            app.context_parallel_enabled = False
            
            compiler_args = app.get_compiler_args()
            assert '--cc-pipeline-tiling-factor=4' in compiler_args


class TestBasicFunctionality:
    """Test basic functionality without complex initialization"""
    
    def test_split_along_dim_function(self):
        """Test split_along_dim function"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import split_along_dim
        
        tensor = torch.randn(2, 4, 6)
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.scatter_to_process_group_spmd') as mock_scatter:
            mock_scatter.return_value = tensor
            
            result = split_along_dim(tensor=tensor, dim=1, rank=0, data_parallel_group=MagicMock())
            assert result is not None
            mock_scatter.assert_called_once()

    def test_attention_wrapper_context_parallel(self):
        """Test context parallel attention wrapper"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import attention_wrapper_context_parallel_single_transformer
        
        query = torch.randn(1, 18, 256, 64)
        key = torch.randn(1, 18, 256, 64)
        value = torch.randn(18, 256, 64)
        process_group = MagicMock()
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.gather_from_tensor_model_parallel_region_with_dim') as mock_gather, \
             patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._flash_fwd_call') as mock_flash:
            
            mock_gather.return_value = torch.randn(256, 18, 64)
            mock_flash.return_value = None
            
            output = attention_wrapper_context_parallel_single_transformer(query, key, value, process_group)
            assert output is not None
            assert output.shape == (1, 18, 256, 64)

    def test_attention_wrapper_context_parallel_vc_size_2(self):
        """Test context parallel attention with VC size 2"""
        from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import attention_wrapper_context_parallel_single_transformer
        
        query = torch.randn(1, 18, 256, 64)
        key = torch.randn(1, 18, 256, 64)
        value = torch.randn(18, 256, 64)
        process_group = MagicMock()
        
        with patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux.gather_from_tensor_model_parallel_region_with_dim') as mock_gather, \
             patch('neuronx_distributed_inference.models.diffusers.flux.modeling_flux._flash_fwd_call') as mock_flash, \
             patch.dict('os.environ', {'NEURON_RT_VIRTUAL_CORE_SIZE': '2'}):
            
            mock_gather.return_value = torch.randn(256, 18, 64)
            mock_flash.return_value = None
            
            output = attention_wrapper_context_parallel_single_transformer(query, key, value, process_group)
            assert output is not None
            assert output.shape == (1, 18, 256, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
