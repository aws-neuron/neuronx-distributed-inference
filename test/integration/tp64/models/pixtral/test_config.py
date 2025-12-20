import logging
import os

import torch

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.pixtral.modeling_pixtral import PixtralInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_pixtral_config(dtype=torch.float32,
                       tkg_batch_size=1,
                       text_tp_degree=64,
                       vision_tp_degree=16,
                       world_size=64,
                       text_seq_length=2048,
                       vision_seq_len=2048,
                       max_new_tokens=128,
                       text_buckets=None,
                       vision_buckets=None,
                       flash_decoding_enabled=False,
                       sequence_parallel_enabled=True,
                       use_text_kernels=True,
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
        fused_qkv=use_text_kernels,
        qkv_kernel_enabled=use_text_kernels,
        mlp_kernel_enabled=use_text_kernels,
        enable_bucketing=True,
        attn_block_tkg_nki_kernel_enabled=use_text_kernels,
        attn_block_tkg_nki_kernel_cache_update=use_text_kernels,
        cc_pipeline_tiling_factor=1,
        attention_dtype=dtype,
        rpl_reduce_dtype=dtype,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )

    # TODO: integrate NeuronAttentionBase with non-causal block attention mask for image attention
    # and enable kernels for perf
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
        fused_qkv=False,
        qkv_kernel_enabled=False, # Vision model has not been tested with kernels yet
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        enable_bucketing=True,
        cc_pipeline_tiling_factor=1,
        rpl_reduce_dtype=dtype,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )

    config = PixtralInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    return config
