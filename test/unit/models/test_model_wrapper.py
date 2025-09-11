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
    
    attention_mask = torch.ones((1, 100))
    bucket = wrapper.get_target_bucket(None, attention_mask)
    assert bucket == 128


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


def test_chunked_prefill_layer_unroll():
    """Test chunked prefill adds layer unroll factor."""
    config = create_base_config()
    config.neuron_config.is_chunked_prefill = True
    wrapper = ModelWrapper(config, MockModel, tag=CONTEXT_ENCODING_MODEL_TAG)
    assert "--layer-unroll-factor=4" in wrapper.compiler_args


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
