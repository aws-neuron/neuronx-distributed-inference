# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

import torch
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig

CONFIG = {
    'TEXT_TP_DEGREE': 16,
    'VISION_TP_DEGREE': 16, 
    'WORLD_SIZE': 16,
    'BATCH_SIZE': 1,
    'SEQ_LENGTH': 2048,
    'DTYPE': torch.bfloat16,
    'MODEL_PATH': "/home/ubuntu/model_hf/gemma-3-27b-it",
    'TRACED_MODEL_PATH': "/home/ubuntu/traced_model/gemma-3-27b-it-v3",
    'IMAGE_PATH': "/home/ubuntu/daanggn-neuron-inference-migration/scripts/dog.jpg",
    'MAX_NEW_TOKENS': 100,
    # Optimizations
    'QUANTIZED': False,
    'QUANTIZED_CHECKPOINTS_PATH': None, # path to pre-quantized model state dict OR path to save quantized model state_dict
    'ATTN_KERNEL_ENABLED': False,
    'ATTN_TKG_NKI_KERNEL_ENABLED': False, 
    'FUSED_QKV': False,
    'ASYNC_MODE': False,
    'ON_DEVICE_SAMPLING': None
    # OnDeviceSamplingConfig(
    #     dynamic=True, # Allow per-request sampling config
    #     do_sample=True, 
    #     deterministic=True,
    #     temperature=1.0,
    #     top_p=1.0,
    #     top_k=32,
    #     global_topk=256, 
    #     top_k_kernel_enabled=True,
    #     ), 
    }