"""RosettaFold3 on AWS Trainium 2.

Provides compilation and inference utilities for accelerating RF3 structure
prediction on Neuron hardware using vanilla torch_neuronx.trace() compilation.
"""

from .modeling_rf3 import (
    RF3NeuronPipeline,
    FullPairformerBlock,
    MSAPairUpdateBlock,
    TemplPairformerBlock,
    DiffTransformerBlock,
    StaticWindowedAttnBlock,
    precompute_window_indices,
    patch_rf3_for_neuron,
    C_S,
    C_Z,
    C_TOKEN,
    C_Z_TEMPL,
    C_ATOM,
    C_ATOMPAIR,
    ATOM_N_HEAD,
    ATOM_C_HEAD,
    ATOM_QBATCH,
    ATOM_KBATCH,
)
from .rf3_bucketing import (
    BucketConfig,
    pad_tensor,
    unpad_tensor,
    pad_pair,
    unpad_pair,
    pad_single,
    unpad_single,
    pad_atom_pair,
    unpad_atom_pair,
    pad_atom_single,
    unpad_atom_single,
    make_pair_mask,
    make_single_mask,
    make_atom_mask,
    CompiledModelCache,
)
