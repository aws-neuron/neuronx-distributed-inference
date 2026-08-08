# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ASR contrib model for NeuronX Distributed Inference."""

from .modeling_qwen3_asr import (
    NeuronQwen3ASRForCausalLM,
    create_inference_config,
    get_encoder_output_length,
    AUDIO_PAD_ID,
    AUDIO_START_ID,
    AUDIO_END_ID,
    IM_START_ID,
    IM_END_ID,
    EOS_ID,
    ASR_TEXT_TOKEN_ID,
)
from .audio_encoder import (
    StaticQwen3ASREncoder,
    trace_encoder,
    load_encoders,
    select_bucket,
)
