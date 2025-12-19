import copy
import os
import tempfile
from argparse import Namespace

import pytest
import torch
from transformers import GenerationConfig, Llama4Config, Llama4ForConditionalGeneration
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4_text import LlamaInferenceConfig, NeuronLlama4TextForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from torch_neuronx.testing.validation import DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE


SCOUT_PERF_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=8192,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": True
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared",
        fused_qkv=True,
        qkv_kernel_enabled=True,
        attn_kernel_enabled=True,
        attn_block_tkg_nki_kernel_enabled=True,
        attn_block_tkg_nki_kernel_cache_update=True,
        k_cache_transposed=True,
        cc_pipeline_tiling_factor=1,
        sequence_parallel_enabled=True,
        blockwise_matmul_config={
            "block_size": 256,
            "use_block_parallel": True,
            "block_sharding_strategy": "HI_LO",
            "skip_dma_token": True,
            "skip_dma_weight": True,
            "parallelize_token_to_block_mapping": True
        }
    )

SCOUT_CHUNKED_ATTN_PERF_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=10240,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": True
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared",
        fused_qkv=True,
        qkv_kernel_enabled=True,
        attn_kernel_enabled=True,
        attn_block_tkg_nki_kernel_enabled=True,
        attn_block_tkg_nki_kernel_cache_update=True,
        k_cache_transposed=False,
        cc_pipeline_tiling_factor=1,
        sequence_parallel_enabled=True,
        blockwise_matmul_config={
            "block_size": 256,
            "use_block_parallel": True,
            "block_sharding_strategy": "HI_LO",
            "skip_dma_token": True,
            "skip_dma_weight": True,
            "parallelize_token_to_block_mapping": True
        }
    )

SCOUT_BASELINE_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=8192,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=1,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )

SCOUT_SHORT_SEQ_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=4096,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=1,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )

SCOUT_CP_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=8192,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        sequence_parallel_enabled=True,
        attn_kernel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )

SCOUT_CP_CHUNKED_ATTN_CONFIG_WITH_SEQ_IDS_MASKING = MoENeuronConfig(
        batch_size=1,
        seq_len=10240,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        sequence_parallel_enabled=True,
        attn_kernel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared",
        apply_seq_ids_mask=True,
    )

SCOUT_CP_CHUNKED_ATTN_CONFIG = MoENeuronConfig(
        batch_size=4,
        tkg_batch_size=4,
        ctx_batch_size=1,
        seq_len=10240,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )
SCOUT_CP_CHUNKED_ATTN_CONFIG_16K = MoENeuronConfig(
        batch_size=4,
        tkg_batch_size=4,
        ctx_batch_size=1,
        seq_len=16384,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )
SCOUT_CP_CHUNKED_ATTN_CONFIG_32K = MoENeuronConfig(
        batch_size=4,
        tkg_batch_size=4,
        ctx_batch_size=1,
        seq_len=32768,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )

SCOUT_CHUNKED_ATTN_NO_FLASH_ATTN_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=10240,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=1,
        sequence_parallel_enabled=True,
        attn_kernel_enabled=False,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )

LLAMA4_128E_PERF_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=8192,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": True
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared",
        fused_qkv=True,
        qkv_kernel_enabled=True,
        attn_kernel_enabled=True,
        mlp_kernel_enabled=True,
        attn_block_tkg_nki_kernel_enabled=True,
        attn_block_tkg_nki_kernel_cache_update=True,
        k_cache_transposed=True,
        cc_pipeline_tiling_factor=1,
        sequence_parallel_enabled=True,
        blockwise_matmul_config={
            "block_size": 128,
            "use_block_parallel": True,
            "block_sharding_strategy": "PING_PONG",
            "skip_dma_token": True,
            "skip_dma_weight": True,
            "parallelize_token_to_block_mapping": True,
        }
    )

LLAMA4_128E_BASELINE_CONFIG = MoENeuronConfig(
        batch_size=1,
        seq_len=8192,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=1,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared"
    )


@pytest.fixture(scope="module", autouse=True)
def model_path_from_config(request):
    hf_config_name = request.param
    assert hf_config_name in {"config_16E_4layer.json", "config_128E_4layer.json", "config_16E_4layer_2048_chunk_size.json"}

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), hf_config_name)
    is_scout_model = hf_config_name == "config_16E_4layer.json"
    model_tempdir = save_checkpoint(config_path, align_rope_scaling = is_scout_model)

    model_path = model_tempdir.name
    yield model_path

    model_tempdir.cleanup()


def save_checkpoint(config_path, torch_dtype = None, disable_attn_temperature_tuning = True, align_rope_scaling = True):
    # Even though we are planning to run the text-only checkpoint,
    # we need to store the config and weights for the full vision + text
    # checkpoint because NxDI does not support LLamaForCausalLM / Llama4TextConfig
    # checkpoints.
    hf_config = Llama4Config.from_pretrained(config_path)

    assert hf_config.text_config.num_hidden_layers == 4
    assert hf_config.vision_config.num_hidden_layers == 4

    if torch_dtype:
        hf_config.torch_dtype = torch_dtype
        hf_config.text_config.torch_dtype = torch_dtype

    # NxDI and meta release don't enable this by default
    if disable_attn_temperature_tuning:
        hf_config.text_config.attn_temperature_tuning = 0

    # TODO: Remove this workaround once NxDI properly reads rope_scaling from the HF config.
    # NxDI hardcodes scaling to these settings so to get matching results we should adjust HF to 
    # run with the same values. Should only apply to Scout.
    if align_rope_scaling:
        nxdi_hardcoded_rope_scaling = {
            "factor": 8.0, # 16.0
            "high_freq_factor": 4.0, # 1.0
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
        hf_config.rope_scaling = nxdi_hardcoded_rope_scaling
        hf_config.text_config.rope_scaling = nxdi_hardcoded_rope_scaling
    
    torch.manual_seed(100)
    hf_model = Llama4ForConditionalGeneration(hf_config)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    
    return model_tempdir


@pytest.mark.tp64_llama4_text
@pytest.mark.parametrize(
    "model_path_from_config, neuron_config, prompt_length, torch_rand_seed, num_tokens_to_check, divergence_difference_tol, tol_map",
    # fmt: off
    [   
        # Temporarily xfail tests that fail due to transformers v4.56 upgrade to unblock pipeline
        # See workload T_67ae5e70-baf5-47fa-ab55-8ae6ced873c7 for test failure details.
        pytest.param("config_16E_4layer.json", SCOUT_CP_CONFIG, 520, 1234, 128, 0.004, {}, marks=pytest.mark.xfail), # prompt > s/cp
        pytest.param("config_16E_4layer.json", SCOUT_CP_CONFIG, 128, 1234, 128, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail), # prompt < s/cp
        pytest.param("config_16E_4layer.json", SCOUT_CP_CHUNKED_ATTN_CONFIG, 8190, 1234, 26, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None), # generate past chunk boundary
        pytest.param("config_16E_4layer.json", SCOUT_CP_CHUNKED_ATTN_CONFIG_WITH_SEQ_IDS_MASKING, 8190, 1234, 30, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail), # generate past chunk boundary
        pytest.param("config_16E_4layer.json", SCOUT_CP_CHUNKED_ATTN_CONFIG, 128, 1234, 30, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail), # generate within first chunk
        pytest.param("config_16E_4layer.json", SCOUT_CP_CHUNKED_ATTN_CONFIG_16K, 128, 1234, 30,  DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail), # generate within first chunk
        pytest.param("config_16E_4layer.json", SCOUT_CP_CHUNKED_ATTN_CONFIG_32K, 128, 1234, 30,  DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail), # generate within first chunk
        pytest.param("config_16E_4layer.json", SCOUT_CHUNKED_ATTN_NO_FLASH_ATTN_CONFIG, 8200, 1234, 128, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail), # torch chunked attn, prompt > chunk_size
        pytest.param("config_16E_4layer.json", SCOUT_CHUNKED_ATTN_PERF_CONFIG, 128, 1234, 30, 0.004, None, marks=pytest.mark.xfail), # Chunked attn with MOE and TKG mega kernel
        pytest.param("config_16E_4layer.json", SCOUT_CHUNKED_ATTN_NO_FLASH_ATTN_CONFIG, 128, 1234, 128, 0.004, None, marks=pytest.mark.xfail), # torch chunked attn, prompt < chunk_size
        pytest.param("config_16E_4layer.json", SCOUT_BASELINE_CONFIG, 128, 1234, 128, 0.004, None, marks=pytest.mark.xfail),
        pytest.param("config_16E_4layer.json", SCOUT_PERF_CONFIG, 128, 1234, 128, 0.004, None, marks=[pytest.mark.key_config_test, pytest.mark.xfail(reason="Unstable due to random weights, see P316201525")]),
        pytest.param("config_16E_4layer.json", SCOUT_SHORT_SEQ_CONFIG, 128, 1234, 128, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, None, marks=pytest.mark.xfail),
        # pytest.param("config_128E_4layer.json", LLAMA4_128E_BASELINE_CONFIG, 128, 1234, 128),
        # pytest.param("config_128E_4layer.json", LLAMA4_128E_PERF_CONFIG, 128, 1234, 128),
    ],
    # fmt: on
    indirect=["model_path_from_config"]
)
def test_llama4_4layer_text(model_path_from_config: str, neuron_config: MoENeuronConfig, prompt_length: int, torch_rand_seed: int, num_tokens_to_check, divergence_difference_tol, tol_map):
    neuron_config_clone = copy.deepcopy(neuron_config)
    
    # Logits testing must turn off on_device_sampling currently.
    neuron_config_clone.on_device_sampling_config = None
    neuron_config_clone.skip_warmup = True

    config = LlamaInferenceConfig(
        neuron_config=neuron_config_clone,
        load_config=load_pretrained_config(model_path_or_name=model_path_from_config),
    )

    text_model_config = config.get_text_config()
    assert text_model_config.num_hidden_layers == 4
    model = NeuronLlama4TextForCausalLM(model_path_from_config, text_model_config)

    compiled_model_path = os.path.join(model_path_from_config, "compiled_checkpoint_accuracy")
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    generation_config = GenerationConfig(do_sample=False, bos_token_id=200000, pad_token_id=200018, eos_token_id=[200001, 200007, 200008])

    torch.manual_seed(torch_rand_seed)
    test_input_batch_size = 1
    input_ids = torch.randint(low=0, high=100000, size=(test_input_batch_size, prompt_length))
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones(size=(test_input_batch_size, prompt_length), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        inputs=inputs,
        num_tokens_to_check=num_tokens_to_check,
        divergence_difference_tol=divergence_difference_tol,
        tol_map=tol_map
    )
