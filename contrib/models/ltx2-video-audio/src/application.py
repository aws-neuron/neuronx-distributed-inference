"""
NxDI LTX-2 Application
======================
Top-level orchestrator for the LTX-2 video+audio diffusion model on Neuron.

Architecture:
  - Transformer backbone → Neuron (TP=4, compiled with NxDI)
  - Text encoder (Gemma 3-12B) → CPU (too large for Neuron NC)
  - Video VAE decoder → CPU
  - Audio VAE decoder + vocoder → CPU

Unlike Flux which compiles ALL components (CLIP, T5, transformer, VAE) onto Neuron,
LTX-2 only compiles the DiT transformer. The text encoder (Gemma 3-12B) and VAEs
remain on CPU because:
  1. Gemma 3-12B exceeds single NC memory even with TP
  2. VAE decoding is sequential and runs once per generation
  3. The transformer is the bottleneck (48 blocks × 8 denoising steps)

Usage:
    from nxdi.application import NeuronLTX2Application

    app = NeuronLTX2Application(
        model_path="Lightricks/LTX-2",
        backbone_config=backbone_config,
        height=384, width=512, num_frames=25,
    )
    app.compile(compiled_model_path)
    app.load(compiled_model_path)
    output = app(prompt="A golden retriever runs across a meadow")
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

try:
    from .modeling_ltx2 import (
        NeuronLTX2BackboneApplication,
        LTX2BackboneInferenceConfig,
    )
    from .pipeline import NeuronLTX2Pipeline
except ImportError:
    from modeling_ltx2 import NeuronLTX2BackboneApplication, LTX2BackboneInferenceConfig
    from pipeline import NeuronLTX2Pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronLTX2Application(nn.Module):
    """Top-level application class for LTX-2 on Neuron.

    Orchestrates:
    1. Loading the Diffusers LTX2Pipeline from HuggingFace
    2. Compiling/loading the DiT transformer backbone on Neuron
    3. Swapping the pipeline's transformer with the Neuron version
    4. Running end-to-end inference (text → video+audio)

    The compile/load/call pattern follows NeuronFluxApplication:
    - compile(path) → traces and saves the Neuron model
    - load(path) → loads compiled model onto Neuron devices
    - __call__(...) → runs the full pipeline
    """

    def __init__(
        self,
        model_path: str,
        backbone_config: LTX2BackboneInferenceConfig,
        transformer_path: Optional[str] = None,
        height: int = 384,
        width: int = 512,
        num_frames: int = 25,
        num_inference_steps: int = 8,
        instance_type: str = "trn2",
    ):
        super().__init__()
        self.model_path = model_path
        self.transformer_path = transformer_path or os.path.join(
            model_path, "transformer"
        )
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.instance_type = instance_type

        self.backbone_config = backbone_config

        # Load the stock Diffusers pipeline (CPU)
        # This gives us text encoder, VAEs, vocoder, scheduler, etc.
        logger.info("Loading Diffusers LTX2Pipeline from %s", model_path)
        self.pipe = NeuronLTX2Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        # Wrap the transformer with NxDI
        self.pipe.neuron_backbone = NeuronLTX2BackboneApplication(
            model_path=self.transformer_path,
            config=self.backbone_config,
        )

    def compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None):
        """Compile the transformer backbone for Neuron.

        Only compiles the transformer — text encoder and VAEs stay on CPU.
        """
        logger.info("Compiling LTX-2 transformer backbone to %s", compiled_model_path)
        self.pipe.neuron_backbone.compile(
            os.path.join(compiled_model_path, "transformer/"),
            debug,
            pre_shard_weights_hook,
        )

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        """Load the compiled transformer backbone onto Neuron devices.

        After loading, swaps the pipeline's CPU transformer with the Neuron version.
        """
        logger.info("Loading compiled LTX-2 transformer from %s", compiled_model_path)
        self.pipe.neuron_backbone.load(
            os.path.join(compiled_model_path, "transformer/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )

        # Swap the pipeline's transformer with the Neuron wrapper
        self.pipe._swap_transformer_to_neuron()

    def __call__(self, *args, **kwargs):
        """Run the full LTX-2 pipeline.

        Supports all arguments that LTX2Pipeline.__call__ accepts:
          - prompt, width, height, num_frames, num_inference_steps
          - guidance_scale, generator, max_sequence_length, etc.
        """
        # Set defaults for common parameters
        kwargs.setdefault("height", self.height)
        kwargs.setdefault("width", self.width)
        kwargs.setdefault("num_frames", self.num_frames)
        kwargs.setdefault("num_inference_steps", self.num_inference_steps)
        kwargs.setdefault("guidance_scale", 4.0)  # CFG (pipeline default)

        return self.pipe(*args, **kwargs)
