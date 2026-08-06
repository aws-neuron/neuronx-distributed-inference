# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom NKI kernels for Qwen3.5-27B DeltaNet layers.

Contains three kernel implementations:
- nki_deltanet: Per-token recurrent kernel (used for token generation)
- nki_deltanet_chunked: Per-chunk kernel (legacy, superseded by fused)
- nki_deltanet_fused: Fused single-kernel chunked forward (used for context encoding)
"""
