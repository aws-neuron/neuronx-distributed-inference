import pytest
import types

from unittest.mock import Mock, patch

from transformers import Llama4TextModel

import neuronx_distributed_inference.models.llama4.utils.patch_llama4 as patch_llama4


class TestPatchLlama4TextMoeForward:
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock Llama4TextModel with MoE layers."""
        model = Mock(spec=Llama4TextModel)
        
        # Create mock layers - some MoE, some not
        moe_layer1 = Mock()
        moe_layer1.is_moe_layer = True
        moe_layer1.feed_forward = Mock()
        moe_layer1.feed_forward.forward = Mock()
        
        moe_layer2 = Mock()
        moe_layer2.is_moe_layer = True
        moe_layer2.feed_forward = Mock()
        moe_layer2.feed_forward.forward = Mock()
        
        non_moe_layer = Mock()
        non_moe_layer.is_moe_layer = False
        non_moe_layer.feed_forward = Mock()
        non_moe_layer.feed_forward.forward = Mock()
        
        model.layers = [moe_layer1, non_moe_layer, moe_layer2]
        return model
    
    @pytest.mark.parametrize("transformers_version", [
        "4.54.0", "4.56.0",
    ])
    def test_patches_moe_layers_when_version_above_threshold(self, mock_model, transformers_version):
        """Test that MoE layers are patched when transformers version >= 4.54.0."""
        with patch.object(patch_llama4.transformers, "__version__", transformers_version):
            original_forwards = [layer.feed_forward.forward for layer in mock_model.layers]
            
            patch_llama4.patch_llama4_text_moe_forward(mock_model)
            
            for i, layer in enumerate(mock_model.layers):
                if layer.is_moe_layer:
                    # Forward method should be different, and method is bound correctly
                    assert layer.feed_forward.forward != original_forwards[i]
                    assert isinstance(layer.feed_forward.forward, types.MethodType)
                    assert layer.feed_forward.forward.__self__ == layer.feed_forward
                else:
                    assert layer.feed_forward.forward == original_forwards[i]
    
    @pytest.mark.parametrize("transformers_version", [
        "4.53.0", "4.6.0",
    ])
    def test_skips_patching_when_version_below_threshold(self, mock_model, transformers_version):
        """Test that patching is skipped when transformers version < 4.54.0."""
        with patch.object(patch_llama4.transformers, "__version__", transformers_version):
            original_forwards = [layer.feed_forward.forward for layer in mock_model.layers]
            
            patch_llama4.patch_llama4_text_moe_forward(mock_model)
            
            # All forward methods should remain unchanged
            for i, layer in enumerate(mock_model.layers):
                assert layer.feed_forward.forward == original_forwards[i]