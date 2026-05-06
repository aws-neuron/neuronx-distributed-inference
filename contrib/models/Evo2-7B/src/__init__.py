"""Evo2-7B on AWS Trainium 2.

Provides compilation and inference utilities for running Arc Institute's
Evo 2 (7B, StripedHyena 2 architecture) DNA language model on Neuron
hardware using vanilla torch_neuronx.trace() compilation.

Two inference modes:
  - Prefill: Block-by-block compilation (32 NEFFs, seq_len up to 2048)
  - Decode: Batched decode with HBM-resident KV-cache and SSM state
    (single mega-NEFF, DP=4 via process-per-core)

FFT replacement:
  Evo2's SSM layers use torch.fft.* which is unsupported on Neuron XLA.
  NeuronFFT provides a pure-PyTorch Stockham FFT that compiles on Neuron,
  supporting seq_len up to 2048.
"""

from .modeling_evo2 import (
    # FFT
    NeuronFFT,
    neuron_rfft,
    neuron_irfft,
    neuron_fft_real,
    neuron_ifft,
    # Patching and model loading
    patch_vortex_for_neuron,
    install_neuron_fft_patch,
    load_config,
    build_model,
    # Prefill
    BlockWrapper,
    Evo2PrefillPipeline,
    # Decode blocks (single-sequence)
    ATTDecodeBlock,
    HCLDecodeBlock,
    FIRDecodeBlock,
    Evo2DecodeBlock,
    # Batched decode blocks
    BatchATTDecode,
    BatchHCLDecode,
    BatchFIRDecode,
    BatchMegaDecode,
    # High-level pipeline
    Evo2NeuronPipeline,
    # Constants
    BLOCK_TYPE_MAP,
    REPRESENTATIVE_BLOCKS,
)
