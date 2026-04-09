import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import (
    CLIPInferenceConfig,
    NeuronClipApplication,
)
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
    FluxBackboneInferenceConfig,
    NeuronFluxBackboneApplication,
)
from neuronx_distributed_inference.models.diffusers.flux.pipeline import NeuronFluxPipeline
from neuronx_distributed_inference.models.diffusers.flux.vae.modeling_vae import (
    NeuronVAEDecoderApplication,
    VAEDecoderInferenceConfig,
)
from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import (
    NeuronT5Application,
    T5InferenceConfig,
)
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_flux_parallelism_config(
    backbone_tp_degree: int,
    context_parallel_enabled: bool = False,
    cfg_parallel_enabled: bool = False
) -> int:
    """
    Get the world_size based on backbone_tp_degree and parallelism settings.

    Args:
        backbone_tp_degree: The tensor parallelism degree for the backbone model
        context_parallel_enabled: Whether context parallelism is enabled (default: False)
        cfg_parallel_enabled: Whether CFG parallelism is enabled (default: False)

    Returns:
        int: world_size (equals backbone_tp_degree, or 2x if context/CFG parallel enabled)

    Note:
        context_parallel_enabled and cfg_parallel_enabled are mutually exclusive.
        Both require world_size = 2 × backbone_tp_degree (dp_degree=2).
    """
    # Validate mutual exclusivity
    if context_parallel_enabled and cfg_parallel_enabled:
        raise ValueError(
            "context_parallel_enabled and cfg_parallel_enabled are mutually exclusive. "
            "Only one can be True at a time."
        )

    # Determine if we need 2x world_size (either for context parallel or CFG parallel)
    use_2x_world_size = context_parallel_enabled or cfg_parallel_enabled

    world_size = backbone_tp_degree * 2 if use_2x_world_size else backbone_tp_degree

    return world_size


def create_flux_config(model_path, world_size, backbone_tp_degree, dtype, height, width, inpaint=False,
                       cfg_parallel_enabled=False, context_parallel_enabled=False):
    text_encoder_path = os.path.join(model_path, "text_encoder")
    text_encoder_2_path = os.path.join(model_path, "text_encoder_2")
    backbone_path = os.path.join(model_path, "transformer")
    vae_decoder_path = os.path.join(model_path, "vae")

    clip_neuron_config = NeuronConfig(
        tp_degree=1,
        world_size=world_size,
        torch_dtype=dtype,
    )
    clip_config = CLIPInferenceConfig(
        neuron_config=clip_neuron_config,
        load_config=load_pretrained_config(text_encoder_path),
    )

    t5_neuron_config = NeuronConfig(
        tp_degree=world_size,  # T5: TP degree = world_size
        world_size=world_size,
        torch_dtype=dtype,
    )
    t5_config = T5InferenceConfig(
        neuron_config=t5_neuron_config,
        load_config=load_pretrained_config(text_encoder_2_path),
    )

    backbone_neuron_config = NeuronConfig(
        tp_degree=backbone_tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    backbone_config = FluxBackboneInferenceConfig(
        cfg_parallel_enabled=cfg_parallel_enabled,
        context_parallel_enabled=context_parallel_enabled,
        neuron_config=backbone_neuron_config,
        load_config=load_diffusers_config(backbone_path),
        height=height,
        width=width,
    )

    decoder_neuron_config = NeuronConfig(
        tp_degree=1,
        world_size=world_size,
        torch_dtype=dtype,
    )
    if inpaint:
        decoder_config = VAEDecoderInferenceConfig(
            neuron_config=decoder_neuron_config,
            load_config=load_diffusers_config(vae_decoder_path),
            height=height,
            width=width,
        )
    else:
        decoder_config = VAEDecoderInferenceConfig(
            neuron_config=decoder_neuron_config,
            load_config=load_diffusers_config(vae_decoder_path),
            height=height,
            width=width,
            transformer_in_channels=backbone_config.in_channels,
        )

    setattr(backbone_config, "vae_scale_factor", decoder_config.vae_scale_factor)

    return (clip_config, t5_config, backbone_config, decoder_config)


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
        pipeline_class=NeuronFluxPipeline,
    ):
        super().__init__()
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path or os.path.join(model_path, "text_encoder")
        self.text_encoder_2_path = text_encoder_2_path or os.path.join(model_path, "text_encoder_2")
        self.transformer_path = transformer_path or os.path.join(model_path, "transformer")
        self.vae_decoder_path = vae_decoder_path or os.path.join(model_path, "vae")

        self.height = height
        self.width = width
        self.max_sequence_length = 512

        self.pipe = pipeline_class.from_pretrained(
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

    def _compile_component(self, component, component_name, compiled_model_path, compiler_workdir, debug):
        component_path = os.path.join(compiled_model_path, f"{component_name}/")
        if os.path.exists(component_path):
            logger.info(f"{component_name} already compiled at {component_path}, skipping compilation.")
        else:
            os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(compiler_workdir, component_name)
            component.compile(component_path, debug)

    def compile(self, compiled_model_path, debug=False):
        compiler_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self._compile_component(self.pipe.text_encoder, "text_encoder", compiled_model_path, compiler_workdir, debug)
        self._compile_component(self.pipe.text_encoder_2, "text_encoder_2", compiled_model_path, compiler_workdir, debug)
        self._compile_component(self.pipe.transformer, "transformer", compiled_model_path, compiler_workdir, debug)
        self._compile_component(self.pipe.vae.decoder, "decoder", compiled_model_path, compiler_workdir, debug)
        os.environ["BASE_COMPILE_WORK_DIR"] = compiler_workdir

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
