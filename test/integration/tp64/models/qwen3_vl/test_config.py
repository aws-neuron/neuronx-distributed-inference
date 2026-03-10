import logging
import os

import torch

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLInferenceConfig, Qwen3VLNeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_qwen3_vl_text_config(dtype=torch.float32,
                        tkg_batch_size=1,
                        text_tp_degree=4,
                        world_size=4,
                        text_seq_length=2048,
                        max_new_tokens=128,
                        text_buckets=None,
                        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_4layer.json")):

    TEXT_SEQ_LENGTH = text_seq_length
    TEXT_BUCKETS = text_buckets if text_buckets is not None else [TEXT_SEQ_LENGTH]
    text_neuron_config = Qwen3VLNeuronConfig(batch_size=1,
                                seq_len=TEXT_SEQ_LENGTH,
                                ctx_batch_size=1,
                                tkg_batch_size=tkg_batch_size,
                                tp_degree=text_tp_degree,
                                world_size=world_size,
                                torch_dtype=dtype,
                                attention_dtype=dtype,
                                rpl_reduce_dtype=dtype,
                                cp_degree=1,
                                save_sharded_checkpoint=True,
                                sequence_parallel_enabled=False,
                                fused_qkv=False,
                                qkv_kernel_enabled=False,
                                mlp_kernel_enabled=False,
                                enable_bucketing = False,
                                context_encoding_buckets=TEXT_BUCKETS,
                                token_generation_buckets=TEXT_BUCKETS,
                                attn_block_tkg_nki_kernel_enabled=False,
                                attn_block_tkg_nki_kernel_cache_update=False,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1), 
                                output_logits=True,
                                cc_pipeline_tiling_factor=2,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                max_new_tokens=max_new_tokens,
                                )
    VISION_SEQ_LENGTH = TEXT_SEQ_LENGTH * 4  # Pre-merger length; 2x2 spatial merger reduces 4 vision tokens → 1 text tokens
    VISION_BUCKETS = [VISION_SEQ_LENGTH]
    dummy_vision_neuron_config = Qwen3VLNeuronConfig(batch_size=1,
                                seq_len=VISION_SEQ_LENGTH,
                                ctx_batch_size=1,
                                tp_degree=text_tp_degree,
                                world_size=text_tp_degree,
                                torch_dtype=dtype,
                                attention_dtype=dtype,
                                rpl_reduce_dtype=dtype,
                                cp_degree=1,
                                save_sharded_checkpoint=True,
                                sequence_parallel_enabled=False,
                                fused_qkv=True,
                                qkv_kernel_enabled=False,
                                # TODO: Currently, to support dynamic image resolution, attention mask will be non-causal block mask which is not supported by kernel yet.
                                attn_kernel_enabled=False,
                                mlp_kernel_enabled=False,
                                enable_bucketing=True,
                                buckets=VISION_BUCKETS,
                                cc_pipeline_tiling_factor=2,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                )

    print("Loading Qwen3 VL Inference Config... Qwen3VLInferenceConfig")
    config = Qwen3VLInferenceConfig(
        text_neuron_config,
        dummy_vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    config.text_config.transformers_version=config.transformers_version # Qwen3 VL requires transformers >=4.57.3
    return config
