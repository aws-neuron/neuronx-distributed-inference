"""OpenFold3 on AWS Trainium 2.

Provides compilation and inference utilities for accelerating OpenFold3
biomolecular structure prediction on Neuron hardware using vanilla
torch_neuronx.trace() compilation with weight replacement.

Two PairFormer compilation strategies:
  - Monolithic (N <= 256): Single PairFormerBlock trace per layer
  - Decomposed (N > 256, up to 2048): N-range-aware sub-op traces with
    fused TriMul and merged segments at small N, chunked MHA at large N
"""

from .modeling_openfold3 import (
    OpenFold3NeuronPipeline,
    DecomposedPairFormerCompiler,
    PairFormerBlockWrapper,
    MSABlockWrapper,
    TemplatePairBlockWrapper,
    DiffCondForwardWrapper,
    TriMulProjectionWrapper,
    TriMulBmmWrapper,
    TriMulOutputWrapper,
    TriMulFullWrapper,
    TriMulBmmOutputWrapper,
    FusedTriMulOutInWrapper,
    TriAttnBiasWrapper,
    TriAttnMHAWrapper,
    TriAttnMHAChunkedWrapper,
    TriAttnFullWrapper,
    AttnPairBiasBiasWrapper,
    AttnPairBiasMHAWrapper,
    AttnPairBiasFullWrapper,
    PairTransitionWrapper,
    SingleTransitionWrapper,
    create_dummy_batch,
    patch_openfold3_source,
    C_S,
    C_Z,
    C_M,
    C_TOKEN,
    CHUNKED_ATTN_THRESHOLD,
    CHUNKED_ATTN_CHUNK_SIZE,
    FUSED_TRIMUL_OUT_IN_MAX_N,
    MERGED_TRIMUL_MAX_N,
    MERGED_TRIMUL_BMM_OUTPUT_MAX_N,
    MERGED_ATTN_MAX_N,
)
