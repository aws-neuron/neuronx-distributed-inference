import logging
import os

import torch

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_qwen2_vl_config(dtype=torch.float32,
                        tkg_batch_size=1,
                        text_tp_degree=4,
                        vision_tp_degree=4,
                        world_size=4,
                        text_seq_length=2048,
                        vision_seq_len=2048,
                        max_new_tokens=128,
                        text_buckets=None,
                        vision_buckets=None,
                        flash_decoding_enabled=False,
                        sequence_parallel_enabled=False,
                        use_text_kernels=False,
                        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")):

    text_neuron_config = NeuronConfig(
        batch_size=tkg_batch_size,
        ctx_batch_size=1,   # CTE and VE alway BS1
        tkg_batch_size=tkg_batch_size,
        seq_len=text_seq_length,
        max_new_tokens=max_new_tokens,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=True,
        tp_degree=text_tp_degree,
        cp_degree=1,
        world_size=world_size,
        context_encoding_buckets=text_buckets,
        token_generation_buckets=text_buckets,
        capacity_factor=None,
        flash_decoding_enabled=flash_decoding_enabled,
        sequence_parallel_enabled=sequence_parallel_enabled,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        mlp_kernel_enabled=False,
        enable_bucketing=True,
        attn_block_tkg_nki_kernel_enabled=False,
        attn_block_tkg_nki_kernel_cache_update=False,
        cc_pipeline_tiling_factor=2,
        attention_dtype=dtype,
        rpl_reduce_dtype=dtype,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )

    vision_neuron_config = NeuronConfig(
        batch_size=1,  # CTE and VE alway BS1
        seq_len=vision_seq_len,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=True,
        tp_degree=vision_tp_degree,
        cp_degree=1,
        world_size=world_size,
        on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
        buckets=vision_buckets,
        fused_qkv=True,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=True,
        mlp_kernel_enabled=True,
        enable_bucketing=True,
        cc_pipeline_tiling_factor=2,
        rpl_reduce_dtype=dtype,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )

    config = Qwen2VLInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    return config
