import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import NeuronClipApplication
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
    NeuronFluxBackboneApplication,
)
from neuronx_distributed_inference.models.diffusers.flux.pipeline import NeuronFluxPipeline
from neuronx_distributed_inference.models.diffusers.flux.vae.modeling_vae import NeuronVAEDecoderApplication

from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import NeuronT5Application

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: see if we need to merge with NeuronApplicationBase at the end
class NeuronFluxApplication(nn.Module):
    def __init__(
        self,
        model_path: str,
        text_encoder_config: InferenceConfig,
        text_encoder2_config: InferenceConfig,
        backbone_config: InferenceConfig,
        decoder_config: InferenceConfig,
        text_encoder_path: Optional[str] = None,
        text_encoder_2_path: Optional[str] = None,
        vae_decoder_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        instance_type: str = "trn1",
    ):
        super().__init__()
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path or os.path.join(model_path, "text_encoder")
        self.text_encoder_2_path = text_encoder_2_path or os.path.join(model_path, "text_encoder_2")
        self.transformer_path = transformer_path or os.path.join(model_path, "transformer")
        self.vae_decoder_path = vae_decoder_path or os.path.join(model_path, "vae")

        self.height = height
        self.width = width
        self.instance_type = instance_type
        self.max_sequence_length = 512

        self.pipe = NeuronFluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        self.text_encoder_config = text_encoder_config
        self.text_encoder2_config = text_encoder2_config
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config

        self.pipe.text_encoder = NeuronClipApplication(
            model_path=self.text_encoder_path, config=self.text_encoder_config
        )
        self.pipe.text_encoder_2 = NeuronT5Application(
            model_path=self.text_encoder_2_path, config=self.text_encoder2_config
        )
        self.pipe.transformer = NeuronFluxBackboneApplication(
            model_path=self.transformer_path,
            config=self.backbone_config,
        )
        self.pipe.vae.decoder = NeuronVAEDecoderApplication(
            model_path=self.vae_decoder_path, config=self.decoder_config
        )

    def compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None):
        self.pipe.text_encoder.compile(
            os.path.join(compiled_model_path, "text_encoder/"), debug, pre_shard_weights_hook
        )
        self.pipe.text_encoder_2.compile(
            os.path.join(compiled_model_path, "text_encoder_2/"), debug, pre_shard_weights_hook
        )
        self.pipe.transformer.compile(
            os.path.join(compiled_model_path, "transformer/"), debug, pre_shard_weights_hook
        )
        self.pipe.vae.decoder.compile(
            os.path.join(compiled_model_path, "decoder/"), debug, pre_shard_weights_hook
        )

    def load(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False
    ):
        self.pipe.text_encoder.load(
            os.path.join(compiled_model_path, "text_encoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self.pipe.text_encoder_2.load(
            os.path.join(compiled_model_path, "text_encoder_2/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self.pipe.transformer.load(
            os.path.join(compiled_model_path, "transformer/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self.pipe.vae.decoder.load(
            os.path.join(compiled_model_path, "decoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)
