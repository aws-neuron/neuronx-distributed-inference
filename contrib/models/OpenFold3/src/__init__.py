"""OpenFold3 on AWS Trainium 2.

Provides compilation and inference utilities for accelerating OpenFold3
biomolecular structure prediction on Neuron hardware using vanilla
torch_neuronx.trace() compilation with weight replacement.
"""

from .modeling_openfold3 import (
    OpenFold3NeuronPipeline,
    PairFormerBlockWrapper,
    MSABlockWrapper,
    TemplatePairBlockWrapper,
    DiffCondForwardWrapper,
    create_dummy_batch,
    patch_openfold3_source,
    C_S,
    C_Z,
    C_M,
    C_TOKEN,
)
