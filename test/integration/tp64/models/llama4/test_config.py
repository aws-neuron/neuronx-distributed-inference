import logging
import os

import torch

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4 import (
    Llama4InferenceConfig,
    Llama4NeuronConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 1
TEXT_TP_DEGREE = 64
VISION_TP_DEGERE = 16
WORLD_SIZE = 64
SEQ_LENGTH = 8192


def get_llama4_config(dtype=torch.float32, 
                      model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_16E_4layer.json")):

    router_config = {"dtype": torch.float32, "act_fn": "sigmoid"}

    text_neuron_config = Llama4NeuronConfig(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=TEXT_TP_DEGREE,
        cp_degree=1,
        on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
        world_size=WORLD_SIZE,
        capacity_factor=None,
        fused_qkv=False,
        attention_dtype=dtype,
        rpl_reduce_dtype=torch.float32,
        early_expert_affinity_modulation=True,
        disable_normalize_top_k_affinities=True,
        cast_type="as-declared",
        router_config=router_config,
        logical_neuron_cores=2,
        output_logits=True,
    )

    # Vision kernels with FP32 are known to not pass accuracy threshold. Turning off kernels in FP32.
    if dtype == torch.float32:
        use_vision_kernel = False
    else:
        use_vision_kernel = True
    vision_neuron_config = Llama4NeuronConfig(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=VISION_TP_DEGERE,
        cp_degree=1,
        on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
        dp_degree=WORLD_SIZE//VISION_TP_DEGERE,
        world_size=WORLD_SIZE,
        fused_qkv=True,
        qkv_kernel_enabled=use_vision_kernel,
        attn_kernel_enabled=use_vision_kernel,
        mlp_kernel_enabled=use_vision_kernel,
        enable_bucketing=use_vision_kernel,
        buckets=[8, 16, 88],
        logical_neuron_cores=2,
    )

    config = Llama4InferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    return config
