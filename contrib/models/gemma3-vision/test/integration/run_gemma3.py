# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

from gemma3_vision.ndxi_patch import apply_patch
apply_patch()

import logging # noqa: E402
import os # noqa: E402
from pathlib import Path # noqa: E402

import torch
from transformers import AutoTokenizer, AutoProcessor, GenerationConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama4.utils.input_processor import (
    prepare_generation_inputs_hf
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter
)

from gemma3_vision.modeling_gemma3 import NeuronGemma3ForConditionalGeneration, Gemma3InferenceConfig


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Setting paths
BASE_PATH = os.getenv('PROJECT_HOME', '/home/ubuntu/nxdi-gemma3-contribution')
DATA_PATH = os.getenv('DATA_HOME', '/home/ubuntu')

# Model configuration constants
CONFIG = {
    'TEXT_TP_DEGREE': 8,
    'VISION_TP_DEGREE': 8,
    'WORLD_SIZE': 8,
    'BATCH_SIZE': 1,
    'SEQ_LENGTH': 1024,
    'CTX_BUCKETS': [1024], # Set to a single bucket or powers of two between 128 and the SEQ_LENGTH.
    'TKG_BUCKETS': [1024], # Set to a single bucket or powers of two between 128 and the SEQ_LENGTH.
    'DTYPE': torch.bfloat16,
    'MODEL_PATH': f"{DATA_PATH}/models/gemma-3-27b-it",
    'TRACED_MODEL_PATH': f"{DATA_PATH}/traced_model/gemma-3-27b-it-small",
    'IMAGE_PATH': f"{BASE_PATH}/dog.jpg",
    'MAX_NEW_TOKENS': 100,
    # Optimizations
    'QUANTIZED': False,
    'QUANTIZED_CHECKPOINTS_PATH': None, # path to pre-quantized model state dict OR path to save quantized model state_dict
    'ATTN_KERNEL_ENABLED': True,
    'VISION_ATTN_KERNEL_ENABLED': True,
    'ATTN_TKG_NKI_KERNEL_ENABLED': False,
    'FUSED_QKV': True,
    'VISION_FUSED_QKV': False,
    'ASYNC_MODE': True,
    'OUTPUT_LOGITS': True,
    'ON_DEVICE_SAMPLING': OnDeviceSamplingConfig(
        dynamic=True, # Allow per-request sampling config
        do_sample=True,
        deterministic=True,
        temperature=1.0,
        top_p=1.0,
        top_k=32,
        global_topk=256,
        top_k_kernel_enabled=True,
        ),
    }

# attn_tkg_nki_kernel_enabled fails if TP != 16
if CONFIG['TEXT_TP_DEGREE'] != 16:
    CONFIG['ATTN_TKG_NKI_KERNEL_ENABLED'] = False
# validate and configure settings for quantized models
if CONFIG['QUANTIZED']:
    os.environ['XLA_HANDLE_SPECIAL_SCALAR'] = "1"
    os.environ['UNSAFE_FP8FNCAST'] = "1"
    assert CONFIG['QUANTIZED_CHECKPOINTS_PATH'] is not None, (
        "Quantized checkpoints path must be provided for quantized model"
    )
# validate bucket lengths
assert CONFIG['SEQ_LENGTH'] == max(CONFIG['CTX_BUCKETS']), (
    f"Context bucket {max(CONFIG['CTX_BUCKETS'])} should be <= {CONFIG['SEQ_LENGTH']}"
)
assert CONFIG['SEQ_LENGTH'] == max(CONFIG['TKG_BUCKETS']), (
    f"Token generation bucket {max(CONFIG['TKG_BUCKETS'])} should be <= {CONFIG['SEQ_LENGTH']}"
)

# Environment setup
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn1'
os.environ['NEURON_RT_STOCHASTIC_ROUNDING_EN'] = '0'

torch.manual_seed(0)

def create_neuron_configs():
    """Create text and vision neuron configurations."""
    hf_config = Gemma3TextConfig.from_pretrained(CONFIG['MODEL_PATH'])  # nosec B615

    text_config = NeuronConfig(

        ## Basic configs ##
        batch_size=CONFIG['BATCH_SIZE'],
        seq_len=CONFIG['SEQ_LENGTH'], # max input+output length
        torch_dtype=CONFIG['DTYPE'],
        # cast_type="as-declared", # comment out if optimizing for latency. uncomment if optimizing for accuracy

        ## Compiler configs ##
        cc_pipeline_tiling_factor=1,
        logical_nc_config=1,

        ## Distributed configs ##
        tp_degree=CONFIG['TEXT_TP_DEGREE'],
        cp_degree=1,
        # rpl_reduce_dtype=torch.float32, # comment out if optimizing for latency. uncomment if optimizing for accuracy
        save_sharded_checkpoint=True,
        skip_sharding=False,

        ## Continuous batching ##
        is_continuous_batching=True, # set to true for vLLM integration
        ctx_batch_size=1, # set to 1 for vLLM integration

        ## Bucketing ##
        enable_bucketing=True,
        context_encoding_buckets=CONFIG['CTX_BUCKETS'],
        token_generation_buckets=CONFIG['TKG_BUCKETS'],

        ## Optimizations ##
        async_mode=CONFIG['ASYNC_MODE'],
        on_device_sampling_config=CONFIG['ON_DEVICE_SAMPLING'],
        output_logits=CONFIG['OUTPUT_LOGITS'], # When on-device sampling, logits are not returned by default, set to true to return logits when on-device sampling is enabled
        fused_qkv=CONFIG['FUSED_QKV'],
        sequence_parallel_enabled=False, # always set to false. has meaningful impacts for long-context use cases only

        ## Kernels for Optimization ##
        attn_kernel_enabled=CONFIG['ATTN_KERNEL_ENABLED'], # attn kernels for context_encoding
        attn_tkg_nki_kernel_enabled=CONFIG['ATTN_TKG_NKI_KERNEL_ENABLED'], # attn kernels for token generation
        attn_tkg_builtin_kernel_enabled=False, # always set to false. incompatible with gemma3.
        qkv_kernel_enabled=False, # QKV kernels. always set to false. incompatible with gemma3.
        mlp_kernel_enabled=False, # MLP kernels. always set to false. incompatible with gemma3.

        ## Quantization ##
        quantized=CONFIG['QUANTIZED'],
        quantized_checkpoints_path=CONFIG['QUANTIZED_CHECKPOINTS_PATH'],
        quantization_type="per_channel_symmetric",
        quantization_dtype="f8e4m3",
        modules_to_not_convert=[
            # Targeted at NeuronApplicationBase.generate_quantized_state_dict which works on the HF state dict
            # The following patterns must match keys in the HF state dict.
            "multi_modal_projector",
            "vision_tower",
            *[f"language_model.model.layers.{layer_idx}.self_attn" for layer_idx in range(hf_config.num_hidden_layers)],
            "language_model.lm_head",
            # Targeted at DecoderModelInstance.load_module which dynamically replaces [Row|Column]ParallelLinear
            # layers with Quantized[Row|Column]Parallel layers.
            # The following patterns must match keys in the Neuron state dict of NeuronGemma3[Text|Vision]Model
            *[f"layers.{layer_idx}.self_attn" for layer_idx in range(hf_config.num_hidden_layers)],
            "lm_head",
            ],
        kv_cache_quant=False,
        quantized_mlp_kernel_enabled=False,
    )

    vision_config = NeuronConfig(

        ## Basic configs ##
        batch_size=CONFIG['BATCH_SIZE'] * 2,
        seq_len=CONFIG['SEQ_LENGTH'],
        torch_dtype=CONFIG['DTYPE'],
        # cast_type="as-declared", # comment out if optimizing for latency. uncomment if optimizing for accuracy

        ## Compiler configs ##
        cc_pipeline_tiling_factor=1,
        logical_nc_config=1,

        ## Distributed configs ##
        tp_degree=CONFIG['VISION_TP_DEGREE'],
        world_size=CONFIG['WORLD_SIZE'],
        # rpl_reduce_dtype=torch.float32, # comment out if optimizing for latency. uncomment if optimizing for accuracy
        save_sharded_checkpoint=True,

        ## Continuous batching ##
        is_continuous_batching=True, # set to true for vLLM integration
        ctx_batch_size=1, # set to 1 for vLLM integration

        ## Bucketing ##
        enable_bucketing=True,
        buckets=[1],

        ## Optimizations ##
        fused_qkv=CONFIG['VISION_FUSED_QKV'],

        ## Kernels for Optimization ##
        attn_kernel_enabled=CONFIG['VISION_ATTN_KERNEL_ENABLED'], # attn kernels for context_encoding
        qkv_kernel_enabled=False, # QKV kernels. always set to false. incompatible with gemma3.
        mlp_kernel_enabled=False, # MLP kernels. always set to false. incompatible with gemma3.
    )

    return text_config, vision_config


def setup_model_and_tokenizer():
    """Initialize model configuration, tokenizer, and processor."""
    text_config, vision_config = create_neuron_configs()

    config = Gemma3InferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(CONFIG['MODEL_PATH']),
    )
    config.vision_config.num_hidden_layers = 1
    config.text_config.num_hidden_layers = 1
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_PATH'], padding_side="right")  # nosec B615
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(CONFIG['MODEL_PATH'])  # nosec B615

    return config, tokenizer, processor


def compile_or_load_model(config, tokenizer):
    """Compile model if needed, otherwise load from checkpoint."""
    if not os.path.exists(CONFIG['TRACED_MODEL_PATH']):
        if config.neuron_config.quantized and config.neuron_config.save_sharded_checkpoint:
            quantized_state_dict_path = Path(config.neuron_config.quantized_checkpoints_path)
            quantized_sd_available = quantized_state_dict_path.exists()
            if not quantized_sd_available:
            # Weights quantized at compile-time. Directory must already exist.
                print("\nQuantizing and saving model weights...")
                quantized_state_dict_path.mkdir(parents=True, exist_ok=True)
                NeuronGemma3ForConditionalGeneration.save_quantized_state_dict(CONFIG['MODEL_PATH'], config)
        print("\nCompiling and saving model...")
        model = NeuronGemma3ForConditionalGeneration(CONFIG['MODEL_PATH'], config)
        model.compile(CONFIG['TRACED_MODEL_PATH'], debug=True)
        tokenizer.save_pretrained(CONFIG['TRACED_MODEL_PATH'])

    print("\nLoading model from compiled checkpoint...")
    model = NeuronGemma3ForConditionalGeneration(CONFIG['TRACED_MODEL_PATH'])
    model.load(CONFIG['TRACED_MODEL_PATH'], skip_warmup=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['TRACED_MODEL_PATH'])  # nosec B615

    return model, tokenizer


def generate_outputs(model, tokenizer, input_ids, attention_mask, pixel_values=None, vision_mask=None, max_new_tokens=50):
    """Generate text using the model."""
    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig.from_pretrained(CONFIG['MODEL_PATH'])  # nosec B615
    sampling_params = prepare_sampling_params(batch_size=CONFIG['BATCH_SIZE'], top_k=[1], top_p=[1.0], temperature=[0.0])

    return_dict_in_generate = False

    generation_config.update(**{
        "do_sample": True,
        "output_scores": False, # Post-processed logits
        "output_logits": False, # Raw logits
        "return_dict_in_generate": return_dict_in_generate,
        })

    outputs = generation_model.generate(
        input_ids,
        generation_config=generation_config,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        pixel_values=pixel_values,
        vision_mask=vision_mask.to(torch.bool) if vision_mask is not None else None,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=False,
    )

    output_sequences = outputs.sequences if return_dict_in_generate else outputs

    output_tokens = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs, output_tokens


def run_gemma3_generate_image_to_text(run_test_inference=False, run_benchmark=False):
    """Main function to run Gemma3 text and image generation."""
    # Setup
    config, tokenizer, processor = setup_model_and_tokenizer()
    model, tokenizer = compile_or_load_model(config, tokenizer)

    if run_test_inference:
        print("Running output check...")

        # Test 1: Text + Image generation
        print("\n=== Text + Image Generation ===")
        text_prompt = "Describe what you see in the following image(s)"

        input_ids, attention_mask, pixel_values, vision_mask = prepare_generation_inputs_hf(
            text_prompt, [CONFIG['IMAGE_PATH'], CONFIG['IMAGE_PATH']], processor, 'user', config
        )

        if CONFIG['BATCH_SIZE'] > 1:
            input_ids = input_ids.repeat(CONFIG['BATCH_SIZE'], 1)
            attention_mask = attention_mask.repeat(CONFIG['BATCH_SIZE'], 1)
            pixel_values = pixel_values.repeat(CONFIG['BATCH_SIZE'], 1, 1, 1)
            vision_mask = vision_mask.repeat(CONFIG['BATCH_SIZE'], 1, 1)

        outputs, output_tokens = generate_outputs(
            model, tokenizer, input_ids, attention_mask, pixel_values, vision_mask, max_new_tokens=CONFIG['MAX_NEW_TOKENS']
        )

        for i, output_token in enumerate(output_tokens):
            print(f"Output {i}: {output_token}")


        print("\n=== Text-Only Generation ===")
        text_prompt = "What is the recipe of mayonnaise in two sentences?"

        input_ids, attention_mask, _, _ = prepare_generation_inputs_hf(
            text_prompt, None, processor, 'user'
        )

        if CONFIG['BATCH_SIZE'] > 1:
            input_ids = input_ids.repeat(CONFIG['BATCH_SIZE'], 1)
            attention_mask = attention_mask.repeat(CONFIG['BATCH_SIZE'], 1)

        outputs, output_tokens = generate_outputs(
            model, tokenizer, input_ids, attention_mask, max_new_tokens=CONFIG['MAX_NEW_TOKENS']
        )

        for i, output_token in enumerate(output_tokens):
            print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    run_gemma3_generate_image_to_text(run_test_inference=True, run_benchmark=False)
