import pytest
import json
from pathlib import Path
from unittest.mock import patch
import tempfile
from neuronx_distributed_inference.modules.checkpoint import (
    create_n_layer_checkpoint,
    find_layer_idx,
    find_and_update_key,
    load_config_and_update_layer_num,
)


# Test find_layer_idx
class TestFindLayerIdx:
    
    def test_finds_layer_index_basic(self):
        """Test finding layer index in a typical key"""
        key = "model.layers.5.attention.weight"
        result = find_layer_idx(key, layer_prefix=["layers"])
        assert result == 5
    
    def test_finds_layer_index_first_layer(self):
        """Test finding layer index 0"""
        key = "model.layers.0.weight"
        result = find_layer_idx(key, layer_prefix=["layers"])
        assert result == 0
    
    def test_finds_layer_index_double_digits(self):
        """Test finding layer index with multiple digits"""
        key = "transformer.layers.42.mlp.weight"
        result = find_layer_idx(key, layer_prefix=["layers"])
        assert result == 42
    
    def test_when_no_layer_prefix(self):
        """Test returns -1 when layer_prefix not found"""
        key = "model.embedding.weight"
        result = find_layer_idx(key, layer_prefix=["layers"])
        assert result == -1
    
    def test_custom_layer_prefix(self):
        """Test with custom layer prefix"""
        key = "model.blocks.10.weight"
        result = find_layer_idx(key, layer_prefix=["blocks"])
        assert result == 10
    
    def test_multiple_layer_prefixes(self):
        """Test with multiple layer prefixes in list"""
        key1 = "model.layers.5.weight"
        key2 = "model.blocks.3.weight"
        
        prefixes = ["layers", "blocks"]
        
        assert find_layer_idx(key1, layer_prefix=prefixes) == 5
        assert find_layer_idx(key2, layer_prefix=prefixes) == 3
    
    def test_raises_error_when_no_index_after_prefix(self):
        """Test raises error when layer_prefix is at end of key"""
        key = "model.layers"
        with pytest.raises(ValueError, match="Unable to fetch layer index"):
            find_layer_idx(key, layer_prefix=["layers"])
    
    def test_raises_error_when_index_not_numeric(self):
        """Test raises error when layer index is not a number"""
        key = "model.layers.abc.weight"
        with pytest.raises(ValueError, match="Unable to fetch layer index"):
            find_layer_idx(key, layer_prefix=["layers"])
    
    def test_default_layer_prefix(self):
        """Test with default layer_prefix parameter"""
        key = "model.layers.7.weight"
        result = find_layer_idx(key)
        assert result == 7


# Test find_and_update_key
class TestFindAndUpdateKey:
    
    def test_updates_top_level_key(self):
        """Test updating a key at the top level"""
        d = {"num_hidden_layers": 32, "hidden_size": 768}
        result = find_and_update_key(d, "num_hidden_layers", 12)
        
        assert result is True
        assert d["num_hidden_layers"] == 12
        assert d["hidden_size"] == 768
    
    def test_updates_nested_key(self):
        """Test updating a key in a nested dictionary"""
        d = {
            "model_type": "llama",
            "text_config": {
                "num_hidden_layers": 32,
                "hidden_size": 4096
            }
        }
        result = find_and_update_key(d, "num_hidden_layers", 6)
        
        assert result is True
        assert d["text_config"]["num_hidden_layers"] == 6
    
    def test_updates_deeply_nested_key(self):
        """Test updating a key in a deeply nested dictionary"""
        d = {
            "level1": {
                "level2": {
                    "level3": {
                        "target_key": "old_value"
                    }
                }
            }
        }
        result = find_and_update_key(d, "target_key", "new_value")
        
        assert result is True
        assert d["level1"]["level2"]["level3"]["target_key"] == "new_value"
    
    def test_updates_multiple_occurrences(self):
        """Test updating multiple occurrences of the same key"""
        d = {
            "text_config": {"num_hidden_layers": 32},
            "vision_config": {"num_hidden_layers": 24}
        }
        result = find_and_update_key(d, "num_hidden_layers", 6)
        
        assert result is True
        assert d["text_config"]["num_hidden_layers"] == 6
        assert d["vision_config"]["num_hidden_layers"] == 6
    
    def test_returns_false_when_key_not_found(self):
        """Test returns False when key doesn't exist"""
        d = {"other_key": "value"}
        result = find_and_update_key(d, "num_hidden_layers", 12)
        
        assert result is False
        assert d == {"other_key": "value"}
    
    def test_handles_list_of_dicts(self):
        """Test updating keys inside a list of dictionaries"""
        d = {
            "layers": [
                {"num_hidden_layers": 10},
                {"num_hidden_layers": 20}
            ]
        }
        result = find_and_update_key(d, "num_hidden_layers", 5)
        
        assert result is True
        assert d["layers"][0]["num_hidden_layers"] == 5
        assert d["layers"][1]["num_hidden_layers"] == 5
    
    def test_empty_dict(self):
        """Test with empty dictionary"""
        d = {}
        result = find_and_update_key(d, "key", "value")
        
        assert result is False


# Test load_config_and_update_layer_num
class TestLoadConfigAndUpdateLayerNum:
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"num_hidden_layers": 32, "hidden_size": 768}, f)
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_updates_num_hidden_layers(self, temp_config):
        """Test updating num_hidden_layers"""
        result_path = load_config_and_update_layer_num(temp_config, 6)
        
        with open(temp_config, 'r') as f:
            config = json.load(f)
        
        assert config["num_hidden_layers"] == 6
        assert config["hidden_size"] == 768
        assert result_path == temp_config
    
    def test_updates_nested_config(self):
        """Test updating nested config"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "model_type": "llama",
                "text_config": {"num_hidden_layers": 32}
            }, f)
            temp_path = f.name
        
        try:
            load_config_and_update_layer_num(temp_path, 12)
            
            with open(temp_path, 'r') as f:
                config = json.load(f)
            
            assert config["text_config"]["num_hidden_layers"] == 12
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_updates_multiple_keys(self):
        """Test updating multiple layer config keys"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "num_hidden_layers": 32,
                "depth": 24
            }, f)
            temp_path = f.name
        
        try:
            load_config_and_update_layer_num(
                temp_path, 
                6, 
                layer_config_keys=["num_hidden_layers", "depth"]
            )
            
            with open(temp_path, 'r') as f:
                config = json.load(f)
            
            assert config["num_hidden_layers"] == 6
            assert config["depth"] == 6
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_default_layer_config_keys(self, temp_config):
        """Test default layer_config_keys parameter"""
        load_config_and_update_layer_num(temp_config, 8)
        
        with open(temp_config, 'r') as f:
            config = json.load(f)
        
        assert config["num_hidden_layers"] == 8
    
    def test_returns_config_path(self, temp_config):
        """Test that function returns the config path as string"""
        result = load_config_and_update_layer_num(temp_config, 6)
        
        assert result == str(temp_config)
        assert isinstance(result, str)


# Test create_n_layer_checkpoint
class TestCreateNLayerCheckpoint:
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock the dependencies"""
        with patch('neuronx_distributed_inference.modules.checkpoint.copytree') as mock_copytree, \
             patch('neuronx_distributed_inference.modules.checkpoint.ignore_patterns') as mock_ignore, \
             patch('neuronx_distributed_inference.modules.checkpoint.load_state_dict') as mock_load, \
             patch('neuronx_distributed_inference.modules.checkpoint.save_state_dict_safetensors') as mock_save, \
             patch('neuronx_distributed_inference.modules.checkpoint.load_config_and_update_layer_num') as mock_config, \
             patch('builtins.print'):
            mock_config.return_value = "/tgt/config.json"
            yield mock_copytree, mock_ignore, mock_load, mock_save, mock_config
    
    def test_copies_non_model_files(self, mock_dependencies):
        """Test that non-model files are copied using copytree"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        mock_load.return_value = {}
        
        create_n_layer_checkpoint(5, "/src", "/tgt")
        
        mock_ignore.assert_called_once_with('*.pt', '.safetensors')
        mock_copytree.assert_called_once()
        call_args = mock_copytree.call_args
        assert call_args[0][0] == "/src"
        assert call_args[0][1] == "/tgt"
        assert call_args[1]["dirs_exist_ok"] is True
    
    def test_filters_layers_correctly(self, mock_dependencies):
        """Test that layers are filtered based on n"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        
        full_state_dict = {
            "model.embedding.weight": "emb_tensor",
            "model.layers.0.weight": "layer0_tensor",
            "model.layers.1.weight": "layer1_tensor",
            "model.layers.2.weight": "layer2_tensor",
            "model.layers.3.weight": "layer3_tensor",
            "model.layers.4.weight": "layer4_tensor",
            "model.output.weight": "output_tensor",
        }
        mock_load.return_value = full_state_dict
        
        result = create_n_layer_checkpoint(3, "/src", "/tgt")
        
        # Should include layers 0, 1, 2 and non-layer weights
        assert "model.embedding.weight" in result
        assert "model.layers.0.weight" in result
        assert "model.layers.1.weight" in result
        assert "model.layers.2.weight" in result
        assert "model.output.weight" in result
        
        # Should not include layers >= 3
        assert "model.layers.3.weight" not in result
        assert "model.layers.4.weight" not in result
    
    def test_filters_with_multiple_layer_prefixes(self, mock_dependencies):
        """Test filtering with multiple layer prefixes"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        
        full_state_dict = {
            "model.layers.0.weight": "layer0_tensor",
            "model.layers.1.weight": "layer1_tensor",
            "vision.blocks.0.weight": "block0_tensor",
            "vision.blocks.1.weight": "block1_tensor",
        }
        mock_load.return_value = full_state_dict
        
        result = create_n_layer_checkpoint(
            1, "/src", "/tgt", 
            layer_prefix=["layers", "blocks"]
        )
        
        assert "model.layers.0.weight" in result
        assert "model.layers.1.weight" not in result
        assert "vision.blocks.0.weight" in result
        assert "vision.blocks.1.weight" not in result
    
    def test_saves_filtered_state_dict(self, mock_dependencies):
        """Test that the filtered state dict is saved"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        
        full_state_dict = {
            "model.layers.0.weight": "layer0_tensor",
            "model.layers.1.weight": "layer1_tensor",
        }
        mock_load.return_value = full_state_dict
        
        create_n_layer_checkpoint(1, "/src", "/tgt")
        
        mock_save.assert_called_once()
        saved_dict = mock_save.call_args[0][0]
        assert "model.layers.0.weight" in saved_dict
        assert "model.layers.1.weight" not in saved_dict
        assert mock_save.call_args[0][1] == "/tgt"
    
    def test_updates_config_json(self, mock_dependencies):
        """Test that config.json is updated with new layer count"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        mock_load.return_value = {}
        
        create_n_layer_checkpoint(
            5, "/src", "/tgt",
            layer_config_keys=["num_hidden_layers", "depth"]
        )
        
        mock_config.assert_called_once()
        call_args = mock_config.call_args[0]
        assert "config.json" in call_args[0]
        assert call_args[1] == 5
        assert call_args[2] == ["num_hidden_layers", "depth"]
    
    def test_returns_filtered_state_dict(self, mock_dependencies):
        """Test that the function returns the filtered state dict"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        
        full_state_dict = {
            "model.layers.0.weight": "layer0_tensor",
            "model.layers.5.weight": "layer5_tensor",
        }
        mock_load.return_value = full_state_dict
        
        result = create_n_layer_checkpoint(3, "/src", "/tgt")
        
        assert isinstance(result, dict)
        assert len(result) == 1
        assert "model.layers.0.weight" in result
    
    def test_handles_n_equals_zero(self, mock_dependencies):
        """Test with n=0 (should only include non-layer weights)"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        
        full_state_dict = {
            "model.embedding.weight": "emb_tensor",
            "model.layers.0.weight": "layer0_tensor",
            "model.layers.1.weight": "layer1_tensor",
        }
        mock_load.return_value = full_state_dict
        
        result = create_n_layer_checkpoint(0, "/src", "/tgt")
        
        assert "model.embedding.weight" in result
        assert "model.layers.0.weight" not in result
        assert "model.layers.1.weight" not in result
    
    def test_handles_n_larger_than_layers(self, mock_dependencies):
        """Test with n larger than number of layers"""
        mock_copytree, mock_ignore, mock_load, mock_save, mock_config = mock_dependencies
        
        full_state_dict = {
            "model.layers.0.weight": "layer0_tensor",
            "model.layers.1.weight": "layer1_tensor",
        }
        mock_load.return_value = full_state_dict
        
        result = create_n_layer_checkpoint(100, "/src", "/tgt")
        
        assert len(result) == 2


# Integration test
class TestIntegration:
    
    def test_end_to_end_checkpoint_creation(self):
        """Test the complete workflow with real file operations"""
        
        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as tgt_dir:
            
            src_path = Path(src_dir)
            tgt_path = Path(tgt_dir)
            
            # Create config files
            config_data = {"num_hidden_layers": 4, "hidden_size": 768}
            (src_path / "config.json").write_text(json.dumps(config_data))
            (src_path / "tokenizer.json").write_text('{}')
            
            # Mock load_state_dict and save_state_dict_safetensors
            with patch('neuronx_distributed_inference.modules.checkpoint.load_state_dict') as mock_load, \
                 patch('neuronx_distributed_inference.modules.checkpoint.save_state_dict_safetensors') as mock_save, \
                 patch('builtins.print'):
                
                mock_load.return_value = {
                    "embed.weight": "emb",
                    "model.layers.0.attn.weight": "l0",
                    "model.layers.1.attn.weight": "l1",
                    "model.layers.2.attn.weight": "l2",
                }
                
                result = create_n_layer_checkpoint(2, src_dir, tgt_dir)

                # Verify save was called
                mock_save.assert_called_once()
                
                # Verify config files were copied
                assert (tgt_path / "config.json").exists()
                assert (tgt_path / "tokenizer.json").exists()
                
                # Verify config.json was updated
                with open(tgt_path / "config.json", 'r') as f:
                    updated_config = json.load(f)
                assert updated_config["num_hidden_layers"] == 2
                
                # Verify correct layers in result
                assert len(result) == 3  # embed + layer 0 + layer 1
                assert "model.layers.2.attn.weight" not in result
    
    def test_end_to_end_with_nested_config(self):
        """Test with nested config structure"""
        
        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as tgt_dir:
            
            src_path = Path(src_dir)
            tgt_path = Path(tgt_dir)
            
            # Create nested config
            config_data = {
                "model_type": "vl",
                "text_config": {"num_hidden_layers": 32},
                "vision_config": {"num_hidden_layers": 24}
            }
            (src_path / "config.json").write_text(json.dumps(config_data))
            
            with patch('neuronx_distributed_inference.modules.checkpoint.load_state_dict') as mock_load, \
                 patch('neuronx_distributed_inference.modules.checkpoint.save_state_dict_safetensors'), \
                 patch('builtins.print'):
                
                mock_load.return_value = {
                    "text.layers.0.weight": "t0",
                    "vision.blocks.0.weight": "v0",
                }
                
                create_n_layer_checkpoint(
                    1, src_dir, tgt_dir,
                    layer_prefix=["layers", "blocks"],
                    layer_config_keys=["num_hidden_layers"]
                )
                
                # Verify nested config was updated
                with open(tgt_path / "config.json", 'r') as f:
                    updated_config = json.load(f)
                
                assert updated_config["text_config"]["num_hidden_layers"] == 1
                assert updated_config["vision_config"]["num_hidden_layers"] == 1
