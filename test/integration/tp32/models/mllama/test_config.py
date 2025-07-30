import logging
import os

import torch

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.mllama.modeling_mllama import MllamaInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.models.config import MultimodalVisionNeuronConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 1
TP_DEGREE = 32
SEQ_LENGTH = 4096

def get_mllama_config(dtype=torch.float32,
                       model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")):

    neuron_config = MultimodalVisionNeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        enable_bucketing=False,
        sequence_parallel_enabled=True,
        fused_qkv=True,
        async_mode=False,
        torch_dtype=dtype,
    )

    config = MllamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    return config
