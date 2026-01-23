# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import logging
import os

import torch

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM, Gemma3InferenceConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CONFIG = {
    'TEXT_TP_DEGREE': 16,
    'VISION_TP_DEGREE': 16, 
    'WORLD_SIZE': 16,
    'BATCH_SIZE': 1,
    'SEQ_LENGTH': 4096,
    'DTYPE': torch.bfloat16,
    }


def get_gemma3_config(dtype=torch.bfloat16, 
                      model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")):


    text_config = NeuronConfig(
        batch_size=CONFIG['BATCH_SIZE'],
        seq_len=CONFIG['SEQ_LENGTH'],
        torch_dtype=CONFIG['DTYPE'],
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=CONFIG['TEXT_TP_DEGREE'],
        cp_degree=1,
        on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
        world_size=CONFIG['WORLD_SIZE'],
        capacity_factor=None,
        fused_qkv=False,
        attention_dtype=dtype,
        rpl_reduce_dtype=torch.float32,
        cast_type="as-declared",
        enable_bucketing=True,
        context_encoding_buckets=[CONFIG['SEQ_LENGTH']],
        token_generation_buckets=[CONFIG['SEQ_LENGTH']],
        qkv_kernel_enabled=False, 
        mlp_kernel_enabled=False, 
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False, 
        logical_nc_config=1
    )

    vision_config = NeuronConfig(
        batch_size=CONFIG['BATCH_SIZE'],
        seq_len=CONFIG['SEQ_LENGTH'],
        torch_dtype=CONFIG['DTYPE'],
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=CONFIG['VISION_TP_DEGREE'],
        cp_degree=1,
        on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
        world_size=CONFIG['WORLD_SIZE'],
        fused_qkv=False,
        rpl_reduce_dtype=torch.float32,
        cast_type="as-declared",
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        enable_bucketing=True,
        buckets=[1],
        logical_nc_config=1
    )
    
    config = Gemma3InferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(model_path),
    )

    return config