import pytest
from unittest.mock import Mock, patch, call

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.utils.snapshot import (
    ScriptModuleWrapper,
    SnapshotOutputFormat,
)


class TestRegisterSnapshotHooksFromEnv:
    """Test suite for _register_snapshot_hooks_from_env method."""
    
    @pytest.fixture
    def neuron_app(self):
        """Create a NeuronApplicationBase instance for testing."""
        with patch.object(NeuronApplicationBase, '__init__', lambda x, *args, **kwargs: None):
            app = NeuronApplicationBase.__new__(NeuronApplicationBase)
            return app
    
    def test_register_snapshot_hooks_from_env_success(self, neuron_app, monkeypatch):
        """Test successful registration of snapshot hooks from environment variables."""
        # Arrange
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_PATH', '/output/path')
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT', 'NUMPY_IMAGES')
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_AT_REQUESTS', '0,1,2')

        mock_register = Mock()
        neuron_app.register_snapshot_hooks = mock_register
        
        # Act
        neuron_app._register_snapshot_hooks_from_env()
        
        # Assert
        mock_register.assert_called_once_with(
            output_path='/output/path',
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            capture_at_requests=[0, 1, 2],
        )
    
    def test_register_snapshot_hooks_from_env_default_transposed_inputs(self, neuron_app, monkeypatch):
        """Test registration with default transposed inputs setting."""
        # Arrange
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_PATH', '/output/path')
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT', 'NUMPY_PICKLE')
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_AT_REQUESTS', '0,1,2')

        mock_register = Mock()
        neuron_app.register_snapshot_hooks = mock_register
        
        # Act
        neuron_app._register_snapshot_hooks_from_env()
        
        # Assert
        mock_register.assert_called_once_with(
            output_path='/output/path',
            output_format=SnapshotOutputFormat.NUMPY_PICKLE,
            capture_at_requests=[0, 1, 2],
        )
    
    def test_register_snapshot_hooks_from_env_missing_output_path(self, neuron_app, monkeypatch):
        """Test assertion when output path is missing."""
        # Arrange
        monkeypatch.delenv('NXD_INFERENCE_CAPTURE_OUTPUT_PATH', raising=False)
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT', 'NUMPY_IMAGES')
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_AT_REQUESTS', '0')
        
        # Act & Assert
        with pytest.raises(AssertionError, match="Must set NXD_INFERENCE_CAPTURE_OUTPUT_PATH"):
            neuron_app._register_snapshot_hooks_from_env()
    
    def test_register_snapshot_hooks_from_env_missing_output_format(self, neuron_app, monkeypatch):
        """Test assertion when output format is missing."""
        # Arrange
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_PATH', '/output/path')
        monkeypatch.delenv('NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT', raising=False)
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_AT_REQUESTS', '0')
        
        # Act & Assert
        with pytest.raises(AssertionError, match="Must set NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT"):
            neuron_app._register_snapshot_hooks_from_env()
    
    def test_register_snapshot_hooks_from_env_missing_capture_requests(self, neuron_app, monkeypatch):
        """Test assertion when capture requests is missing."""
        # Arrange
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_PATH', '/output/path')
        monkeypatch.setenv('NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT', 'NUMPY_IMAGES')
        monkeypatch.delenv('NXD_INFERENCE_CAPTURE_AT_REQUESTS', raising=False)
        
        # Act & Assert
        with pytest.raises(AssertionError, match="Must set NXD_INFERENCE_CAPTURE_AT_REQUESTS"):
            neuron_app._register_snapshot_hooks_from_env()


class TestRegisterSnapshotHooks:
    """Test suite for register_snapshot_hooks method."""
    
    @pytest.fixture
    def neuron_app(self):
        """Create a NeuronApplicationBase instance for testing."""
        with patch.object(NeuronApplicationBase, '__init__', lambda x, *args, **kwargs: None):
            app = NeuronApplicationBase.__new__(NeuronApplicationBase)
            app.is_loaded_to_neuron = True
            
            # Create mock models
            mock_model1 = Mock()
            mock_model1.tag = 'model1'
            mock_model1.model = Mock()
            
            mock_model2 = Mock()
            mock_model2.tag = 'model2'
            mock_model2.model = Mock()
            
            app.models = [mock_model1, mock_model2]
            return app
    

    @patch("neuronx_distributed_inference.models.application_base.ScriptModuleWrapper")
    @patch("neuronx_distributed_inference.models.application_base.get_snapshot_hook")
    @patch("neuronx_distributed_inference.models.application_base.register_nxd_model_hook")
    def test_register_snapshot_hooks_success(self, mock_register_hook, mock_get_snapshot_hook, mock_wrapper, neuron_app):
        """Test successful registration of snapshot hooks."""
        # Arrange
        mock_wrapper_instance1 = Mock()
        mock_wrapper_instance2 = Mock()
        mock_wrapper.side_effect = [mock_wrapper_instance1, mock_wrapper_instance2]

        mock_hook1 = Mock()
        mock_hook2 = Mock()
        mock_get_snapshot_hook.side_effect = [mock_hook1, mock_hook2]

        mock_builder = Mock()
        neuron_app._builder = mock_builder
        neuron_app.models[0].async_mode = False
        neuron_app.models[0].pipeline_execution = False
        neuron_app.models[1].async_mode = False
        neuron_app.models[1].pipeline_execution = False

        model1 = neuron_app.models[0].model
        model2 = neuron_app.models[1].model
        
        # Act
        neuron_app.register_snapshot_hooks(
            output_path='/test/path',
            output_format=SnapshotOutputFormat.NUMPY_IMAGES,
            capture_at_requests=[0, 1, 2],
            ranks=[0, 1],
        )
        
        # Assert
        assert mock_wrapper.call_count == 2
        mock_wrapper.assert_has_calls([
            call(model1),
            call(model2)
        ])
        
        assert mock_get_snapshot_hook.call_count == 2
        mock_get_snapshot_hook.assert_has_calls([
            call('/test/path', SnapshotOutputFormat.NUMPY_IMAGES, [0, 1, 2], mock_builder, [0, 1], is_input_ranked=False),
            call('/test/path', SnapshotOutputFormat.NUMPY_IMAGES, [0, 1, 2], mock_builder, [0, 1], is_input_ranked=False)
        ])

        assert mock_register_hook.call_count == 4
        mock_register_hook.assert_has_calls([
            call(mock_wrapper_instance1, "forward_async", mock_hook1),
            call(mock_wrapper_instance1, "forward_ranked", mock_hook1),
            call(mock_wrapper_instance2, "forward_async", mock_hook2),
            call(mock_wrapper_instance2, "forward_ranked", mock_hook2),
        ])
        
        # Check that models are wrapped and hooks are registered
        assert neuron_app.models[0].model == mock_wrapper_instance1
        assert neuron_app.models[1].model == mock_wrapper_instance2
        
        mock_wrapper_instance1.register_forward_hook.assert_called_once_with(mock_hook1)
        mock_wrapper_instance2.register_forward_hook.assert_called_once_with(mock_hook2)
    
    def test_register_snapshot_hooks_model_not_loaded(self, neuron_app):
        """Test assertion when model is not loaded to neuron."""
        neuron_app.is_loaded_to_neuron = False
        
        with pytest.raises(AssertionError, match="Must load model before you register snapshot hooks"):
            neuron_app.register_snapshot_hooks(
                output_path='/test/path',
                output_format=SnapshotOutputFormat.NUMPY_IMAGES,
                capture_at_requests=[0]
            )


class TestUnregisterSnapshotHooks:
    """Test suite for unregister_snapshot_hooks method."""
    
    @pytest.fixture
    def neuron_app(self):
        """Create a NeuronApplicationBase instance for testing."""
        with patch.object(NeuronApplicationBase, '__init__', lambda x, *args, **kwargs: None):
            app = NeuronApplicationBase.__new__(NeuronApplicationBase)
            return app
    
    @patch("neuronx_distributed_inference.models.application_base.unregister_nxd_model_hooks")
    def test_unregister_snapshot_hooks_with_wrapped_models(self, mock_unregister_hooks, neuron_app):
        """Test unregistering hooks when all models are wrapped."""
        # Arrange
        mock_traced_model1 = Mock()
        mock_traced_model2 = Mock()
        
        mock_wrapper1 = Mock(spec=ScriptModuleWrapper)
        mock_wrapper1.wrapped_module = mock_traced_model1
        
        mock_wrapper2 = Mock(spec=ScriptModuleWrapper)
        mock_wrapper2.wrapped_module = mock_traced_model2
        
        mock_model1 = Mock()
        mock_model1.tag = 'model1'
        mock_model1.model = mock_wrapper1
        
        mock_model2 = Mock()
        mock_model2.tag = 'model2'
        mock_model2.model = mock_wrapper2
        
        neuron_app.models = [mock_model1, mock_model2]
        
        # Act
        neuron_app.unregister_snapshot_hooks()
        
        # Assert
        assert mock_model1.model == mock_traced_model1
        assert mock_model2.model == mock_traced_model2

        # Verify unregister_nxd_model_hooks was called correctly for each model
        expected_calls = [
            call(mock_traced_model1, "forward_async"),
            call(mock_traced_model1, "forward_ranked"),
            call(mock_traced_model2, "forward_async"),
            call(mock_traced_model2, "forward_ranked")
        ]
        mock_unregister_hooks.assert_has_calls(expected_calls, any_order=False)
        assert mock_unregister_hooks.call_count == 4
    
    @patch("neuronx_distributed_inference.models.application_base.unregister_nxd_model_hooks")
    def test_unregister_snapshot_hooks_with_non_wrapped_models(self, mock_unregister_hooks, neuron_app):
        """Test unregistering hooks when models are not wrapped (no-op)."""
        # Arrange
        mock_traced_model1 = Mock()
        mock_traced_model2 = Mock()
        
        mock_model1 = Mock()
        mock_model1.tag = 'model1'
        mock_model1.model = mock_traced_model1
        
        mock_model2 = Mock()
        mock_model2.tag = 'model2'
        mock_model2.model = mock_traced_model2
        
        neuron_app.models = [mock_model1, mock_model2]
        
        # Act
        neuron_app.unregister_snapshot_hooks()
        
        # Models should remain unchanged since they're not wrapped
        assert mock_model1.model == mock_traced_model1
        assert mock_model2.model == mock_traced_model2

        # Verify unregister_nxd_model_hooks was NOT called since models aren't wrapped
        mock_unregister_hooks.assert_not_called()
    
    @patch("neuronx_distributed_inference.models.application_base.unregister_nxd_model_hooks")
    def test_unregister_snapshot_hooks_mixed_models(self, mock_unregister_hooks, neuron_app):
        """Test unregistering hooks with mix of wrapped and non-wrapped models."""
        # Arrange
        mock_traced_model = Mock()
        mock_wrapper = Mock(spec=ScriptModuleWrapper)
        mock_wrapper.wrapped_module = mock_traced_model
        
        mock_model1 = Mock()
        mock_model1.tag = 'wrapped_model'
        mock_model1.model = mock_wrapper
        
        mock_model2 = Mock()
        mock_model2.tag = 'regular_model'
        mock_model2.model = mock_traced_model
        
        neuron_app.models = [mock_model1, mock_model2]
        
        # Act
        neuron_app.unregister_snapshot_hooks()

        # Models should remain unchanged since they're not wrapped
        assert mock_model1.model == mock_traced_model
        assert mock_model2.model == mock_traced_model

        # Verify unregister_nxd_model_hooks was called only for the wrapped model
        expected_calls = [
            call(mock_traced_model, "forward_async"),
            call(mock_traced_model, "forward_ranked")
        ]
        mock_unregister_hooks.assert_has_calls(expected_calls, any_order=False)
        assert mock_unregister_hooks.call_count == 2
