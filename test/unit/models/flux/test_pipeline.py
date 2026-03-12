"""
Unit tests for NeuronFluxPipeline
Tests the pipeline using mocks to ensure it works on CPU without Neuron dependencies.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from types import SimpleNamespace


class ConfigNamespace(SimpleNamespace):
    """A SimpleNamespace subclass that also supports dict-like .get() method"""

    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler"""
    scheduler = MagicMock()
    scheduler.config = ConfigNamespace(
        base_image_seq_len=256,
        max_image_seq_len=4096,
        base_shift=0.5,
        max_shift=1.16,
    )
    scheduler.order = 1
    
    # Create a proper mock for set_timesteps that accepts sigmas parameter
    def mock_set_timesteps(num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
        if sigmas is not None:
            scheduler.timesteps = torch.tensor(sigmas)
        elif timesteps is not None:
            scheduler.timesteps = torch.tensor(timesteps)
        else:
            scheduler.timesteps = torch.linspace(1.0, 0.0, num_inference_steps if num_inference_steps else 10)
    
    # Mock step to return latents with same shape as input
    def mock_step(noise_pred, t, latents, return_dict=False):
        # Return latents with same shape as input
        return [latents]
    
    scheduler.set_timesteps = mock_set_timesteps
    scheduler.timesteps = torch.tensor([1.0, 0.5, 0.1])
    scheduler.step = mock_step
    return scheduler


@pytest.fixture
def mock_vae():
    """Create a mock VAE"""
    vae = MagicMock()
    vae.config = ConfigNamespace(
        block_out_channels=[128, 256, 512, 512],
        scaling_factor=0.13025,
        shift_factor=0.0,
    )
    vae.decode = MagicMock(return_value=[torch.randn(1, 3, 1024, 1024)])
    vae.enable_slicing = MagicMock()
    vae.disable_slicing = MagicMock()
    vae.enable_tiling = MagicMock()
    vae.disable_tiling = MagicMock()
    return vae


@pytest.fixture
def mock_text_encoder():
    """Create a mock CLIP text encoder"""
    text_encoder = MagicMock()
    text_encoder.dtype = torch.float32
    
    # Mock the output with pooler_output attribute
    mock_output = MagicMock()
    mock_output.pooler_output = torch.randn(1, 768)
    text_encoder.return_value = mock_output
    
    return text_encoder


@pytest.fixture
def mock_text_encoder_2():
    """Create a mock T5 text encoder"""
    text_encoder_2 = MagicMock()
    text_encoder_2.dtype = torch.float32
    text_encoder_2.return_value = [torch.randn(1, 512, 4096)]
    return text_encoder_2


@pytest.fixture
def mock_tokenizer():
    """Create a mock CLIP tokenizer"""
    tokenizer = MagicMock()
    tokenizer.model_max_length = 77
    
    # Mock tokenizer output as an object with attributes
    mock_output = ConfigNamespace(
        input_ids=torch.randint(0, 1000, (1, 77)),
        attention_mask=torch.ones(1, 77),
    )
    tokenizer.return_value = mock_output
    tokenizer.batch_decode = MagicMock(return_value=["truncated text"])
    
    return tokenizer


@pytest.fixture
def mock_tokenizer_2():
    """Create a mock T5 tokenizer"""
    tokenizer_2 = MagicMock()
    
    # Mock tokenizer output as a ConfigNamespace with input_ids attribute
    mock_output = ConfigNamespace(
        input_ids=torch.randint(0, 1000, (1, 512)),
        attention_mask=torch.ones(1, 512),
    )
    tokenizer_2.return_value = mock_output
    tokenizer_2.batch_decode = MagicMock(return_value=["truncated text"])
    
    return tokenizer_2


@pytest.fixture
def mock_transformer():
    """Create a mock Flux transformer"""
    transformer = MagicMock()
    transformer.config = ConfigNamespace(
        in_channels=64,
        guidance_embeds=True,
    )
    transformer.dtype = torch.float32
    transformer.return_value = [torch.randn(1, 256, 64)]
    return transformer


@pytest.fixture
def flux_pipeline(mock_scheduler, mock_vae, mock_text_encoder, mock_tokenizer,
                  mock_text_encoder_2, mock_tokenizer_2, mock_transformer):
    """Create a NeuronFluxPipeline instance with mocked components"""
    from neuronx_distributed_inference.models.diffusers.flux.pipeline import NeuronFluxPipeline
    
    pipeline = NeuronFluxPipeline(
        scheduler=mock_scheduler,
        vae=mock_vae,
        text_encoder=mock_text_encoder,
        tokenizer=mock_tokenizer,
        text_encoder_2=mock_text_encoder_2,
        tokenizer_2=mock_tokenizer_2,
        transformer=mock_transformer,
    )
    return pipeline


class TestNeuronFluxPipeline:
    """Test suite for NeuronFluxPipeline"""
    
    def test_pipeline_initialization(self, flux_pipeline):
        """Test that pipeline initializes correctly"""
        assert flux_pipeline is not None
        assert flux_pipeline.vae_scale_factor == 8
        assert flux_pipeline.tokenizer_max_length == 77
        assert flux_pipeline.default_sample_size == 128
    
    def test_get_clip_prompt_embeds(self, flux_pipeline):
        """Test _get_clip_prompt_embeds method"""
        prompt = "a beautiful landscape"
        
        embeds = flux_pipeline._get_clip_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=1,
            device=torch.device('cpu'),
        )
        
        assert embeds is not None
        assert embeds.shape[0] == 1  # batch size
        assert embeds.shape[1] == 768  # pooled embedding dim
    
    def test_get_t5_prompt_embeds(self, flux_pipeline):
        """Test _get_t5_prompt_embeds method"""
        prompt = "a beautiful landscape"
        
        embeds = flux_pipeline._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=1,
            max_sequence_length=512,
            device=torch.device('cpu'),
        )
        
        assert embeds is not None
        assert embeds.shape[0] == 1  # batch size
        assert embeds.shape[1] == 512  # sequence length
        assert embeds.shape[2] == 4096  # hidden dim
    
    def test_encode_prompt(self, flux_pipeline):
        """Test encode_prompt method"""
        prompt = "a beautiful landscape"
        prompt_2 = "a scenic view"
        
        prompt_embeds, pooled_embeds, text_ids = flux_pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=torch.device('cpu'),
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        
        assert prompt_embeds is not None
        assert pooled_embeds is not None
        assert text_ids is not None
        assert text_ids.shape[-1] == 3  # Should have 3 dimensions
    
    def test_check_inputs_valid(self, flux_pipeline):
        """Test check_inputs with valid inputs"""
        # Should not raise any exceptions
        flux_pipeline.check_inputs(
            prompt="test prompt",
            prompt_2="test prompt 2",
            height=1024,
            width=1024,
        )
    
    def test_check_inputs_missing_prompt(self, flux_pipeline):
        """Test check_inputs raises error when prompt is missing"""
        with pytest.raises(ValueError, match="Cannot leave both `prompt` and `prompt_embeds` undefined"):
            flux_pipeline.check_inputs(
                prompt=None,
                prompt_2=None,
                height=1024,
                width=1024,
            )
    
    def test_check_inputs_conflicting_prompt(self, flux_pipeline):
        """Test check_inputs raises error with conflicting inputs"""
        with pytest.raises(ValueError, match="Cannot forward both `prompt`"):
            flux_pipeline.check_inputs(
                prompt="test",
                prompt_2=None,
                height=1024,
                width=1024,
                prompt_embeds=torch.randn(1, 512, 4096),
            )
    
    def test_check_inputs_max_sequence_length(self, flux_pipeline):
        """Test check_inputs validates max_sequence_length"""
        with pytest.raises(ValueError, match="cannot be greater than 512"):
            flux_pipeline.check_inputs(
                prompt="test",
                prompt_2=None,
                height=1024,
                width=1024,
                max_sequence_length=600,
            )
    
    def test_prepare_latent_image_ids(self, flux_pipeline):
        """Test _prepare_latent_image_ids static method"""
        latent_ids = flux_pipeline._prepare_latent_image_ids(
            batch_size=1,
            height=32,
            width=32,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        
        assert latent_ids.shape == (32 * 32, 3)
        assert latent_ids.dtype == torch.float32
    
    def test_pack_latents(self, flux_pipeline):
        """Test _pack_latents static method"""
        latents = torch.randn(1, 16, 64, 64)
        
        packed = flux_pipeline._pack_latents(
            latents=latents,
            batch_size=1,
            num_channels_latents=16,
            height=64,
            width=64,
        )
        
        assert packed.shape == (1, 32 * 32, 16 * 4)
    
    def test_unpack_latents(self, flux_pipeline):
        """Test _unpack_latents static method"""
        # For height=1024, width=1024, vae_scale_factor=8:
        # Latent dimensions: height=128, width=128 (after //8)
        # After packing: (128//2) * (128//2) = 64 * 64 = 4096 patches
        # Channels after packing: 16 * 4 = 64
        packed_latents = torch.randn(1, 4096, 64)
        
        unpacked = flux_pipeline._unpack_latents(
            latents=packed_latents,
            height=1024,
            width=1024,
            vae_scale_factor=8,
        )
        
        assert unpacked.shape[0] == 1  # batch
        assert unpacked.shape[1] == 16  # channels (64 // 4)
        assert unpacked.shape[2] == 128  # height (after unpacking)
        assert unpacked.shape[3] == 128  # width (after unpacking)
    
    def test_prepare_latents(self, flux_pipeline):
        """Test prepare_latents method"""
        latents, latent_ids = flux_pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=1024,
            width=1024,
            dtype=torch.float32,
            device=torch.device('cpu'),
            generator=None,
        )
        
        assert latents is not None
        assert latent_ids is not None
        assert latents.shape[0] == 1  # batch
    
    def test_vae_slicing_methods(self, flux_pipeline):
        """Test VAE slicing enable/disable methods"""
        flux_pipeline.enable_vae_slicing()
        flux_pipeline.vae.enable_slicing.assert_called_once()
        
        flux_pipeline.disable_vae_slicing()
        flux_pipeline.vae.disable_slicing.assert_called_once()
    
    def test_vae_tiling_methods(self, flux_pipeline):
        """Test VAE tiling enable/disable methods"""
        flux_pipeline.enable_vae_tiling()
        flux_pipeline.vae.enable_tiling.assert_called_once()
        
        flux_pipeline.disable_vae_tiling()
        flux_pipeline.vae.disable_tiling.assert_called_once()
    
    def test_pipeline_properties(self, flux_pipeline):
        """Test pipeline properties"""
        flux_pipeline._guidance_scale = 3.5
        flux_pipeline._joint_attention_kwargs = {"test": "value"}
        flux_pipeline._num_timesteps = 28
        flux_pipeline._interrupt = False
        
        assert flux_pipeline.guidance_scale == 3.5
        assert flux_pipeline.joint_attention_kwargs == {"test": "value"}
        assert flux_pipeline.num_timesteps == 28
        assert flux_pipeline.interrupt is False
    
    def test_pipeline_call_basic(self, flux_pipeline):
        """Test basic pipeline call"""
        # Mock progress_bar
        with patch.object(flux_pipeline, 'progress_bar') as mock_progress:
            mock_progress.return_value.__enter__ = Mock(return_value=Mock())
            mock_progress.return_value.__exit__ = Mock(return_value=None)
            
            # Mock maybe_free_model_hooks
            flux_pipeline.maybe_free_model_hooks = MagicMock()
            
            result = flux_pipeline(
                prompt="a beautiful landscape",
                height=1024,
                width=1024,
                num_inference_steps=3,
                guidance_scale=3.5,
                output_type="latent",
            )
            
            assert result is not None
    
    def test_pipeline_call_with_negative_prompt(self, flux_pipeline):
        """Test pipeline call with negative prompt"""
        with patch.object(flux_pipeline, 'progress_bar') as mock_progress:
            mock_progress.return_value.__enter__ = Mock(return_value=Mock())
            mock_progress.return_value.__exit__ = Mock(return_value=None)
            
            flux_pipeline.maybe_free_model_hooks = MagicMock()
            
            result = flux_pipeline(
                prompt="a beautiful landscape",
                negative_prompt="ugly, distorted",
                height=1024,
                width=1024,
                num_inference_steps=3,
                true_cfg_scale=1.5,
                output_type="latent",
            )
            
            assert result is not None
    
    def test_pipeline_unsupported_features(self, flux_pipeline):
        """Test that unsupported features raise appropriate errors"""
        
        with pytest.raises(AssertionError, match="does not support ip_adapter_image"):
            with patch.object(flux_pipeline, 'progress_bar'):
                flux_pipeline.maybe_free_model_hooks = MagicMock()
                flux_pipeline(
                    prompt="test",
                    ip_adapter_image="fake_image",
                    num_inference_steps=1,
                    output_type="latent",
                )
    
    def test_encode_prompt_with_list_prompts(self, flux_pipeline):
        """Test encode_prompt with list of prompts"""
        prompts = ["prompt 1", "prompt 2"]
        prompts_2 = ["prompt 1 detailed", "prompt 2 detailed"]
        
        prompt_embeds, pooled_embeds, text_ids = flux_pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompts_2,
            device=torch.device('cpu'),
            num_images_per_prompt=1,
        )
        
        assert prompt_embeds.shape[0] == 2  # batch size
        assert pooled_embeds.shape[0] == 2
    
    def test_prepare_latents_with_generator(self, flux_pipeline):
        """Test prepare_latents with random generator"""
        generator = torch.Generator().manual_seed(42)
        
        latents1, _ = flux_pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=512,
            width=512,
            dtype=torch.float32,
            device=torch.device('cpu'),
            generator=generator,
        )
        
        # Reset generator and create again - should be same
        generator = torch.Generator().manual_seed(42)
        latents2, _ = flux_pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=512,
            width=512,
            dtype=torch.float32,
            device=torch.device('cpu'),
            generator=generator,
        )
        
        assert torch.allclose(latents1, latents2)
    
    def test_prepare_latents_with_provided_latents(self, flux_pipeline):
        """Test prepare_latents when latents are provided"""
        provided_latents = torch.randn(1, 16, 64, 64)
        
        latents, latent_ids = flux_pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=1024,
            width=1024,
            dtype=torch.float32,
            device=torch.device('cpu'),
            generator=None,
            latents=provided_latents,
        )
        
        assert torch.equal(latents, provided_latents)
        assert latent_ids is not None
    
    def test_check_inputs_missing_pooled_embeds(self, flux_pipeline):
        """Test check_inputs raises error when pooled_embeds missing with prompt_embeds"""
        with pytest.raises(ValueError, match="pooled_prompt_embeds"):
            flux_pipeline.check_inputs(
                prompt=None,
                prompt_2=None,
                height=1024,
                width=1024,
                prompt_embeds=torch.randn(1, 512, 4096),
            )
    
    def test_check_inputs_negative_prompt_conflicts(self, flux_pipeline):
        """Test check_inputs with negative prompt conflicts"""
        with pytest.raises(ValueError, match="Cannot forward both `negative_prompt`"):
            flux_pipeline.check_inputs(
                prompt="test",
                prompt_2=None,
                negative_prompt="bad",
                height=1024,
                width=1024,
                negative_prompt_embeds=torch.randn(1, 512, 4096),
            )
    
    def test_check_inputs_prompt_type_validation(self, flux_pipeline):
        """Test check_inputs validates prompt types"""
        with pytest.raises(ValueError, match="has to be of type"):
            flux_pipeline.check_inputs(
                prompt=12345,  # Invalid type
                prompt_2=None,
                height=1024,
                width=1024,
            )
    
    def test_prepare_latents_generator_list_mismatch(self, flux_pipeline):
        """Test prepare_latents with mismatched generator list length"""
        generators = [torch.Generator(), torch.Generator()]
        
        with pytest.raises(ValueError, match="batch size matches the length"):
            flux_pipeline.prepare_latents(
                batch_size=1,  # Mismatch with 2 generators
                num_channels_latents=16,
                height=512,
                width=512,
                dtype=torch.float32,
                device=torch.device('cpu'),
                generator=generators,
            )
    
    def test_encode_prompt_with_pregenerated_embeds(self, flux_pipeline):
        """Test encode_prompt with pre-generated embeddings"""
        prompt_embeds = torch.randn(1, 512, 4096)
        pooled_embeds = torch.randn(1, 768)
        
        result_embeds, result_pooled, text_ids = flux_pipeline.encode_prompt(
            prompt=None,
            prompt_2=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            device=torch.device('cpu'),
        )
        
        assert torch.equal(result_embeds, prompt_embeds)
        assert torch.equal(result_pooled, pooled_embeds)
    
    def test_pipeline_call_with_pil_output(self, flux_pipeline):
        """Test pipeline call with PIL output type"""
        
        with patch.object(flux_pipeline, 'progress_bar') as mock_progress:
            mock_progress.return_value.__enter__ = Mock(return_value=Mock())
            mock_progress.return_value.__exit__ = Mock(return_value=None)
            
            flux_pipeline.maybe_free_model_hooks = MagicMock()
            
            result = flux_pipeline(
                prompt="test",
                height=1024,
                width=1024,
                num_inference_steps=2,
                output_type="pil",
            )
            
            assert result is not None
    
    def test_pipeline_call_return_tuple(self, flux_pipeline):
        """Test pipeline call with return_dict=False"""
        
        with patch.object(flux_pipeline, 'progress_bar') as mock_progress:
            mock_progress.return_value.__enter__ = Mock(return_value=Mock())
            mock_progress.return_value.__exit__ = Mock(return_value=None)
            
            flux_pipeline.maybe_free_model_hooks = MagicMock()
            
            result = flux_pipeline(
                prompt="test",
                height=1024,
                width=1024,
                num_inference_steps=2,
                output_type="latent",
                return_dict=False,
            )
            
            assert isinstance(result, tuple)
    
    def test_pipeline_call_with_callback(self, flux_pipeline):
        """Test pipeline call with callback function"""
        callback_mock = MagicMock(return_value={})
        
        with patch.object(flux_pipeline, 'progress_bar') as mock_progress:
            mock_progress.return_value.__enter__ = Mock(return_value=Mock())
            mock_progress.return_value.__exit__ = Mock(return_value=None)
            
            flux_pipeline.maybe_free_model_hooks = MagicMock()
            
            result = flux_pipeline(
                prompt="test",
                height=1024,
                width=1024,
                num_inference_steps=2,
                output_type="latent",
                callback_on_step_end=callback_mock,
            )
            
            assert result is not None
            # Callback should have been called
            assert callback_mock.called
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
