"""LaughterSegmentation on AWS Neuron (Inferentia2 / Trainium2).

Provides model loading, compilation, and inference utilities for running
omine-me/LaughterSegmentation (Wav2Vec2-based audio frame classifier)
on Neuron hardware using vanilla torch_neuronx.trace().

Key features:
  - Automatic removal of weight_norm parametrizations (required for XLA)
  - Compiler args tuned for FP32 encoder model (--auto-cast matmult)
  - Single-core and DataParallel (multi-core) inference support
  - Cosine similarity > 0.999999 vs CPU reference
"""

from .modeling_laughter import (
    LaughterNeuronPipeline,
    load_cpu_model,
    remove_parametrizations,
    compile_neuron_model,
)

__all__ = [
    "LaughterNeuronPipeline",
    "load_cpu_model",
    "remove_parametrizations",
    "compile_neuron_model",
]
