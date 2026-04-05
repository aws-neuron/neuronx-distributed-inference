import torch
from unittest.mock import Mock

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import (
    ModelWrapper,
    get_modules_to_not_convert,
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    VISION_ENCODER_MODEL_TAG,
    FUSED_SPECULATION_MODEL_TAG,
)


class MockModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


def create_test_config():
    neuron_config = NeuronConfig(
        batch_size=2,
        torch_dtype=torch.float32,
        buckets=[128, 256, 512],
    )
    config = InferenceConfig(neuron_config=neuron_config)
    config.pad_token_id = 0
    return config


def test_get_modules_to_not_convert():
    neuron_config = NeuronConfig()
    assert get_modules_to_not_convert(neuron_config) is None

    neuron_config.modules_to_not_convert = ["layer1"]
    assert get_modules_to_not_convert(neuron_config) == ["layer1"]


def test_model_wrapper_init():
    config = create_test_config()
    wrapper = ModelWrapper(config, MockModel)

    assert wrapper.config == config
    assert wrapper.model_cls == MockModel
    assert wrapper.model is None


def test_compiler_args_by_tag():
    config = create_test_config()

    wrapper = ModelWrapper(config, MockModel, tag=CONTEXT_ENCODING_MODEL_TAG)
    assert "-O1" in wrapper.compiler_args

    wrapper = ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)
    assert "-O2" in wrapper.compiler_args


def test_is_neuron():
    config = create_test_config()
    wrapper = ModelWrapper(config, MockModel)

    assert not wrapper.is_neuron()
    wrapper.model = Mock(spec=torch.jit.ScriptModule)
    assert wrapper.is_neuron()


def test_convert_int64_to_int32():
    config = create_test_config()
    wrapper = ModelWrapper(config, MockModel)

    int64_tensor = torch.tensor([1], dtype=torch.int64)
    result = wrapper.convert_int64_to_int32(int64_tensor)
    assert result[0].dtype == torch.int32


def test_get_target_bucket():
    config = create_test_config()
    wrapper = ModelWrapper(config, MockModel)

    # Round up -> should select 128
    input_shape = (1, 100)
    input_ids = torch.ones((1, 1), dtype=torch.int32)
    attention_mask = torch.ones(input_shape, dtype=torch.int32)
    position_ids = torch.ones((1, 1), dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)
    bucket = wrapper.get_target_bucket(input_ids, attention_mask, position_ids, seq_ids)
    assert bucket == 128

    # Exact match -> should select 256
    input_shape = (1, 255)
    input_ids = torch.ones((1, 1), dtype=torch.int32)
    attention_mask = torch.ones(input_shape, dtype=torch.int32)
    position_ids = torch.ones((1, 1), dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)
    bucket = wrapper.get_target_bucket(input_ids, attention_mask, position_ids, seq_ids)
    assert bucket == 256


def test_quantization_flags():
    config = create_test_config()
    config.neuron_config.quantized = True
    config.neuron_config.quantization_dtype = "f8e4m3"

    wrapper = ModelWrapper(config, MockModel)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in wrapper.compiler_args


def create_base_config():
    neuron_config = NeuronConfig(torch_dtype=torch.float32)
    config = InferenceConfig(neuron_config=neuron_config)
    config.pad_token_id = 0
    return config


def test_custom_compiler_args():
    """Test when compiler_args is provided."""
    config = create_base_config()
    custom_args = "--custom-flag=value"
    wrapper = ModelWrapper(config, MockModel, compiler_args=custom_args)
    assert wrapper.compiler_args == "--custom-flag=value --internal-hlo2tensorizer-options='--verify-hlo=true' "


def test_token_generation_tiling_factor():
    """Test cc_pipeline_tiling_factor set to 1 for token generation models."""
    config = create_base_config()
    ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)
    assert config.neuron_config.cc_pipeline_tiling_factor == 1


def test_fused_speculation_tiling_factor():
    """Test cc_pipeline_tiling_factor set to 1 for fused speculation models."""
    config = create_base_config()
    ModelWrapper(config, MockModel, tag=FUSED_SPECULATION_MODEL_TAG)
    assert config.neuron_config.cc_pipeline_tiling_factor == 1


def test_vectorize_strided_dma_exclusion():
    """Test vectorize-strided-dma excluded for specific config."""
    config = create_base_config()
    config.neuron_config.logical_nc_config = 2
    config.neuron_config.seq_len = 128 * 1024
    wrapper = ModelWrapper(config, MockModel, tag=CONTEXT_ENCODING_MODEL_TAG)
    assert "--vectorize-strided-dma" not in wrapper.compiler_args


def test_vectorize_strided_dma_included():
    """Test vectorize-strided-dma included by default."""
    config = create_base_config()
    wrapper = ModelWrapper(config, MockModel)
    assert "--vectorize-strided-dma" in wrapper.compiler_args


def test_long_context_mode():
    """Test long context mode flag."""
    config = create_base_config()
    config.neuron_config.enable_long_context_mode = True
    wrapper = ModelWrapper(config, MockModel)
    assert "--internal-disable-fma-on-ios" in wrapper.compiler_args
    assert "--disable-mixed-precision-accumulation" in wrapper.compiler_args


def test_scratchpad_page_size():
    """Test scratchpad page size flag."""
    config = create_base_config()
    config.neuron_config.scratchpad_page_size = 1024
    wrapper = ModelWrapper(config, MockModel)
    assert "--hbm-scratchpad-page-size=1024" in wrapper.compiler_args


def test_block_kv_layout_flags():
    """Test block KV layout flags."""
    config = create_base_config()
    config.neuron_config.is_block_kv_layout = True
    config.neuron_config.attn_block_tkg_nki_kernel_enabled = True
    wrapper = ModelWrapper(config, MockModel)
    assert "--enable-verifier=false" in wrapper.compiler_args


def test_context_encoding_optimization():
    """Test context encoding model gets -O1."""
    config = create_base_config()
    wrapper = ModelWrapper(config, MockModel, tag=CONTEXT_ENCODING_MODEL_TAG)
    assert "-O1" in wrapper.compiler_args
    assert "--modular-flow-mac-threshold=10" in wrapper.compiler_args


def test_vision_encoder_optimization():
    """Test vision encoder model gets -O1."""
    config = create_base_config()
    wrapper = ModelWrapper(config, MockModel, tag=VISION_ENCODER_MODEL_TAG)
    assert "-O1" in wrapper.compiler_args
    assert "--modular-flow-mac-threshold=10" in wrapper.compiler_args


def test_other_models_get_o2():
    """Test non-CTE/vision models get -O2."""
    config = create_base_config()
    wrapper = ModelWrapper(config, MockModel, tag="other_model")
    assert "-O2" in wrapper.compiler_args


def test_spill_reload_dge():
    """Test spill reload DGE flag."""
    config = create_base_config()
    config.neuron_config.enable_spill_reload_dge = True
    wrapper = ModelWrapper(config, MockModel)
    assert "--internal-enable-dge-levels spill_reload" in wrapper.compiler_args


def test_target_flag():
    """Test target flag."""
    config = create_base_config()
    config.neuron_config.target = "trn1"
    wrapper = ModelWrapper(config, MockModel)
    assert "--target trn1" in wrapper.compiler_args


def test_layer_boundary_markers():
    """Test layer boundary markers flag."""
    config = create_base_config()
    config.neuron_config.layer_boundary_markers = True
    wrapper = ModelWrapper(config, MockModel)
    assert "--recursive-layer-det=false" in wrapper.compiler_args


def test_quantization_f8e4m3():
    """Test f8e4m3 quantization flag."""
    config = create_base_config()
    config.neuron_config.quantized = True
    config.neuron_config.quantization_dtype = "f8e4m3"
    wrapper = ModelWrapper(config, MockModel)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in wrapper.compiler_args


def test_kv_cache_quant():
    """Test KV cache quantization flag."""
    config = create_base_config()
    config.neuron_config.kv_cache_quant = True
    wrapper = ModelWrapper(config, MockModel)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in wrapper.compiler_args


def test_output_completion_notifications():
    """Test output completion notifications flag."""
    config = create_base_config()
    config.neuron_config.enable_output_completion_notifications = True
    wrapper = ModelWrapper(config, MockModel)
    assert "--enable-output-completion-notifications" in wrapper.compiler_args


def test_hlo2tensorizer_always_present():
    """Test hlo2tensorizer options always added."""
    config = create_base_config()
    wrapper = ModelWrapper(config, MockModel)
    assert "--verify-hlo=true" in wrapper.compiler_args


def test_tensor_replacement_config_padding():
    """Test tensor replacement configuration padding method."""
    from unittest.mock import Mock, patch
    
    config = create_base_config()
    config.neuron_config.batch_size = 4
    config.neuron_config.tensor_replacement_config = True
    wrapper = ModelWrapper(config, MockModel)
    
    # Mock TensorReplacementRegister with 2 modules
    mock_register = Mock()
    mock_register.module_superset = ["module1", "module2"]
    
    with patch('neuronx_distributed_inference.models.model_wrapper.TensorReplacementRegister') as mock_tr_class:
        mock_tr_class.get_instance.return_value = mock_register
        
        # Create test args structure
        padded_args = [torch.tensor([1, 2, 3, 4]) for _ in range(7)]  # 7 basic args
        
        regular_args = [torch.tensor([[5, 6], [7, 8]]), torch.tensor([[9, 10], [11, 12]])]
        empty_tensors = [torch.empty(0) for _ in range(18)]
        tf_tensors = [torch.tensor([[10, 20], [30, 40]]), torch.tensor([[50, 60], [70, 80]])]
        tf_masks = [torch.tensor([True]), torch.tensor([False])]
        
        args = tuple(padded_args + regular_args + empty_tensors + tf_tensors + tf_masks)
        
        # Test the method using the actual _pad_helper
        result = wrapper._pad_tensor_replacement_args(args, padded_args.copy(), wrapper._pad_helper, None)
        
        # Verify results
        assert len(result) == 31  # 7 + 2 + 18 + 2 + 2 (both masks)
        
        # Check tf_tensors were padded with repeat_first_batchline
        # _pad_helper copies original data first, then fills remaining with first row
        expected_tf1 = torch.tensor([[10, 20], [30, 40], [10, 20], [10, 20]])
        expected_tf2 = torch.tensor([[50, 60], [70, 80], [50, 60], [50, 60]])
        assert torch.equal(result[27], expected_tf1)
        assert torch.equal(result[28], expected_tf2)
        
        # Check both masks were added as-is
        assert torch.equal(result[29], tf_masks[0])
        assert torch.equal(result[30], tf_masks[1])


def test_batch_bucketing_flag():
    """Test batch bucketing flag is set when token_generation_batches is provided."""
    config = create_base_config()
    config.neuron_config.tkg_batch_size = 4
    config.neuron_config.token_generation_batches = [2]
    config.neuron_config.buckets = [[4, 128], [4, 256], [2, 128], [2, 256]]
    wrapper = ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)
    assert wrapper.is_batch_bucketing is True


def test_batch_bucketing_disabled_by_default():
    """Test batch bucketing is disabled when token_generation_batches is None."""
    config = create_base_config()
    config.neuron_config.buckets = [128, 256]
    wrapper = ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)
    assert wrapper.is_batch_bucketing is False


def test_batch_bucketing_input_generation():
    """Test input generation creates correct batch sizes for 2D buckets."""
    config = create_base_config()
    config.neuron_config.batch_size = 4
    config.neuron_config.tkg_batch_size = 4
    config.neuron_config.token_generation_batches = [2]
    config.neuron_config.buckets = [[4, 128], [2, 128]]
    config.neuron_config.n_active_tokens = 1
    config.hidden_size = 128

    wrapper = ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)
    inputs = wrapper.input_generator()

    # Should have 2 buckets
    assert len(inputs) == 2
    # First bucket: batch=4, seq=128
    assert inputs[0][0].shape[0] == 4  # input_ids batch size
    assert inputs[0][1].shape == (4, 128)  # attention_mask shape
    # Second bucket: batch=2, seq=128
    assert inputs[1][0].shape[0] == 2  # input_ids batch size
    assert inputs[1][1].shape == (2, 128)  # attention_mask shape


def test_batch_bucketing_target_bucket_selection():
    """Test get_target_bucket selects smallest bucket that fits."""
    config = create_base_config()
    config.neuron_config.batch_size = 4
    config.neuron_config.tkg_batch_size = 4
    config.neuron_config.token_generation_batches = [2]
    config.neuron_config.buckets = [
        [4, 32], [4, 64], [4, 128],
        [2, 32], [2, 64], [2, 128],
        [1, 32], [1, 64], [1, 128],
    ]

    wrapper = ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)

    # Test bucket selection: (batch_size, seq_len, expected_bucket)
    test_cases = [
        (1, 20, [1, 32]),  # batch=1, seq=20 -> [1, 32]
        (2, 20, [2, 32]),  # batch=2, seq=20 -> [2, 32]
        (3, 20, [4, 32]),  # batch=3, seq=20 -> [4, 32]
        (4, 20, [4, 32]),  # batch=4, seq=20 -> [4, 32]
        (1, 50, [1, 64]),  # batch=1, seq=50 -> [1, 64]
        (2, 50, [2, 64]),  # batch=2, seq=50 -> [2, 64]
        (3, 100, [4, 128]),  # batch=3, seq=100 -> [4, 128]
        (4, 100, [4, 128]),  # batch=4, seq=100 -> [4, 128]
    ]

    for batch_size, seq_len, expected_bucket in test_cases:
        input_ids = torch.ones((batch_size, 1), dtype=torch.int32)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int32)
        position_ids = torch.ones((batch_size, 1), dtype=torch.int32)
        seq_ids = torch.arange(batch_size, dtype=torch.int32)

        bucket = wrapper.get_target_bucket(input_ids, attention_mask, position_ids, seq_ids)
        assert bucket == expected_bucket


def test_batch_bucketing_pad_inputs():
    """Test pad_inputs extracts seq_len from 2D buckets."""
    config = create_base_config()
    config.neuron_config.batch_size = 4
    config.neuron_config.tkg_batch_size = 4
    config.neuron_config.token_generation_batches = [2]
    config.neuron_config.buckets = [[4, 128], [2, 128]]
    config.neuron_config.max_length = 128

    wrapper = ModelWrapper(config, MockModel, tag=TOKEN_GENERATION_MODEL_TAG)

    # Create inputs that need padding: [1, 100] -> [2, 129]
    input_shape = (1, 100)
    input_ids = torch.ones((1, 1), dtype=torch.int32)
    attention_mask = torch.ones(input_shape, dtype=torch.int32)
    position_ids = torch.ones((1, 1), dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)
    padded_args = wrapper.pad_inputs(input_ids, attention_mask, position_ids, seq_ids)
    assert padded_args[1].shape[1] == 128
