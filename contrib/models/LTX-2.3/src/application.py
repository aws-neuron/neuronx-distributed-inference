"""
NeuronLTX23Application — Top-level compositor for LTX-2.3 on Neuron
===================================================================

Manages the Gemma3 text encoder and DiT backbone as NeuronApplicationBase
subclasses, following the Flux pattern from NxDI core
(src/neuronx_distributed_inference/models/diffusers/flux/application.py).

Key difference from Flux: LTX-2.3 uses **sequential NeuronCore sharing**
on trn2.3xlarge. Both the Gemma3 12B encoder and 22B DiT backbone need
all 4 TP cores, so they cannot be loaded simultaneously. The compositor
manages load/unload cycling between sub-models.

Usage:
    from application import NeuronLTX23Application

    app = NeuronLTX23Application(
        backbone_config=backbone_config,
        encoder_config=encoder_config,
        model_path="/path/to/ltx23-safetensors",
        encoder_path="/path/to/gemma3-weights",
    )

    # Compile both sub-models
    app.compile("/path/to/compiled/")

    # Load and run
    app.load_text_encoder("/path/to/compiled/")
    hidden_states = app.encode_text(input_ids, attention_mask)
    app.unload_text_encoder()

    app.load_backbone("/path/to/compiled/")
    video_out, audio_out = app.backbone_app(*backbone_inputs)
    app.unload_backbone()
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from neuronx_distributed_inference.models.config import InferenceConfig

    _NXDI_AVAILABLE = True
except ImportError:
    _NXDI_AVAILABLE = False


class NeuronLTX23Application(nn.Module):
    """Top-level compositor for LTX-2.3 on Neuron.

    NOT a NeuronApplicationBase subclass (same pattern as NeuronFluxApplication).
    Holds two sub-applications that share NeuronCores sequentially.

    Sub-applications:
        backbone_app: NeuronLTX23BackboneApplication (22B DiT transformer)
        encoder_app: NeuronGemma3EncoderApplication (12B Gemma3 text encoder)

    CPU components (VAE decoders, vocoder, embeddings processor) are managed
    separately by the caller or pipeline.
    """

    def __init__(
        self,
        backbone_config: "InferenceConfig",
        encoder_config: "InferenceConfig",
        model_path: str,
        encoder_path: Optional[str] = None,
    ):
        """
        Args:
            backbone_config: InferenceConfig for the DiT backbone
            encoder_config: InferenceConfig for the Gemma3 encoder
            model_path: Path to the LTX-2.3 model weights (safetensors dir or file)
            encoder_path: Path to Gemma3 weights (HuggingFace dir). If None,
                defaults to model_path + "/text_encoder"
        """
        super().__init__()
        try:
            from .modeling_ltx23 import NeuronLTX23BackboneApplication
            from .modeling_gemma3_encoder import NeuronGemma3EncoderApplication
        except ImportError:
            from modeling_ltx23 import NeuronLTX23BackboneApplication
            from modeling_gemma3_encoder import NeuronGemma3EncoderApplication

        self.model_path = model_path
        self.encoder_path = encoder_path or os.path.join(model_path, "text_encoder")

        self.backbone_config = backbone_config
        self.encoder_config = encoder_config

        self.backbone_app = NeuronLTX23BackboneApplication(
            model_path=model_path, config=backbone_config
        )
        self.encoder_app = NeuronGemma3EncoderApplication(
            model_path=self.encoder_path, config=encoder_config
        )

        self._backbone_loaded = False
        self._encoder_loaded = False

    def compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None):
        """Compile both sub-models to separate subdirectories.

        Creates:
            compiled_model_path/backbone/model.pt + weights/
            compiled_model_path/text_encoder/model.pt + weights/
        """
        backbone_path = os.path.join(compiled_model_path, "backbone/")
        encoder_path = os.path.join(compiled_model_path, "text_encoder/")

        logger.info("Compiling DiT backbone to %s", backbone_path)
        self.backbone_app.compile(backbone_path, debug, pre_shard_weights_hook)

        logger.info("Compiling Gemma3 encoder to %s", encoder_path)
        self.encoder_app.compile(encoder_path, debug, pre_shard_weights_hook)

        logger.info("Compilation complete for both sub-models")

    def compile_backbone(
        self, compiled_model_path, debug=False, pre_shard_weights_hook=None
    ):
        """Compile only the DiT backbone."""
        backbone_path = os.path.join(compiled_model_path, "backbone/")
        logger.info("Compiling DiT backbone to %s", backbone_path)
        self.backbone_app.compile(backbone_path, debug, pre_shard_weights_hook)

    def compile_encoder(
        self, compiled_model_path, debug=False, pre_shard_weights_hook=None
    ):
        """Compile only the Gemma3 encoder."""
        encoder_path = os.path.join(compiled_model_path, "text_encoder/")
        logger.info("Compiling Gemma3 encoder to %s", encoder_path)
        self.encoder_app.compile(encoder_path, debug, pre_shard_weights_hook)

    def load_text_encoder(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        """Load the compiled Gemma3 encoder to NeuronCores.

        Must be called before encode_text(). Should be unloaded before
        loading the backbone (sequential NeuronCore sharing).
        """
        if self._backbone_loaded:
            raise RuntimeError(
                "Cannot load text encoder while backbone is loaded. "
                "Call unload_backbone() first (sequential NeuronCore sharing)."
            )
        encoder_path = os.path.join(compiled_model_path, "text_encoder/")
        logger.info("Loading Gemma3 encoder from %s", encoder_path)
        self.encoder_app.load(
            encoder_path, start_rank_id, local_ranks_size, skip_warmup
        )
        self._encoder_loaded = True
        logger.info("Gemma3 encoder loaded and ready")

    def unload_text_encoder(self):
        """Unload the Gemma3 encoder from NeuronCores.

        Frees NeuronCore resources so the backbone can be loaded.
        """
        if not self._encoder_loaded:
            logger.warning("Text encoder not loaded, nothing to unload")
            return
        # Destroy the traced model to release NRT resources
        if (
            hasattr(self.encoder_app, "traced_model")
            and self.encoder_app.traced_model is not None
        ):
            del self.encoder_app.traced_model
            self.encoder_app.traced_model = None
        for model_wrapper in self.encoder_app.models:
            model_wrapper.model = None
        self.encoder_app.is_loaded_to_neuron = False
        self._encoder_loaded = False

        import gc

        gc.collect()
        logger.info("Gemma3 encoder unloaded from NeuronCores")

    def load_backbone(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        """Load the compiled DiT backbone to NeuronCores.

        Must be called before running the denoising loop. Should be unloaded
        before loading the encoder (sequential NeuronCore sharing).
        """
        if self._encoder_loaded:
            raise RuntimeError(
                "Cannot load backbone while text encoder is loaded. "
                "Call unload_text_encoder() first (sequential NeuronCore sharing)."
            )
        backbone_path = os.path.join(compiled_model_path, "backbone/")
        logger.info("Loading DiT backbone from %s", backbone_path)
        self.backbone_app.load(
            backbone_path, start_rank_id, local_ranks_size, skip_warmup
        )
        self._backbone_loaded = True
        logger.info("DiT backbone loaded and ready")

    def unload_backbone(self):
        """Unload the DiT backbone from NeuronCores.

        Frees NeuronCore resources so the encoder can be loaded.
        """
        if not self._backbone_loaded:
            logger.warning("Backbone not loaded, nothing to unload")
            return
        if (
            hasattr(self.backbone_app, "traced_model")
            and self.backbone_app.traced_model is not None
        ):
            del self.backbone_app.traced_model
            self.backbone_app.traced_model = None
        for model_wrapper in self.backbone_app.models:
            model_wrapper.model = None
        self.backbone_app.is_loaded_to_neuron = False
        self._backbone_loaded = False

        import gc

        gc.collect()
        logger.info("DiT backbone unloaded from NeuronCores")

    def encode_text(self, input_ids, attention_mask):
        """Run the Gemma3 encoder. Must have called load_text_encoder() first.

        Args:
            input_ids: (B, seq_len) int64
            attention_mask: (B, seq_len) int64

        Returns:
            stacked_hidden_states: (B, seq_len, hidden_size, 49)
        """
        if not self._encoder_loaded:
            raise RuntimeError(
                "Text encoder not loaded. Call load_text_encoder() first."
            )
        return self.encoder_app(input_ids, attention_mask)

    @property
    def is_backbone_loaded(self):
        return self._backbone_loaded

    @property
    def is_encoder_loaded(self):
        return self._encoder_loaded

    def __call__(self, *args, **kwargs):
        """Forward pass through the backbone. Alias for backbone_app.forward()."""
        if not self._backbone_loaded:
            raise RuntimeError("Backbone not loaded. Call load_backbone() first.")
        return self.backbone_app(*args, **kwargs)
