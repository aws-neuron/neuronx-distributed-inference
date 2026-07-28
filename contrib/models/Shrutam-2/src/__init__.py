# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shrutam-2: Multilingual Indic ASR on AWS Neuron."""

from .modeling_shrutam2 import (
    Shrutam2Pipeline,
    build_llm_model,
    trace_encoder,
    trace_smear,
    MelSpectrogramPreprocessor,
    ConformerEncoder,
    ConformerEncoderForTrace,
    EncoderDownsamplerConv1d,
    MoELayer_SMEAR,
    SMEARForTrace,
)

__all__ = [
    "Shrutam2Pipeline",
    "build_llm_model",
    "trace_encoder",
    "trace_smear",
    "MelSpectrogramPreprocessor",
    "ConformerEncoder",
    "ConformerEncoderForTrace",
    "EncoderDownsamplerConv1d",
    "MoELayer_SMEAR",
    "SMEARForTrace",
]
