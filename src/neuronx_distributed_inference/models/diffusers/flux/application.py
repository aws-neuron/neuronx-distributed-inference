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
from neuronx_distributed_inference.models.diffusers.flux.pipeline import (
    NeuronFluxPipeline,
    NeuronFluxFillPipeline,
    NeuronFluxControlPipeline,
)
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


def get_flux_parallelism_config(instance_type: str, context_parallel_enabled: bool = True):
    """
    Get the world_size and backbone_tp_degree based on instance type and context parallel settings.

    Args:
        instance_type: The instance type (e.g., "trn1", "trn2")
        context_parallel_enabled: Whether context parallelism is enabled (default: True)

    Returns:
        tuple: (world_size, backbone_tp_degree)
    """
    world_size = 8
    backbone_tp_degree = 8

    if instance_type == "trn1":
        if context_parallel_enabled:
            world_size = 16
            backbone_tp_degree = 8
        else:
            world_size = 8
            backbone_tp_degree = 8
    elif instance_type == "trn2":
        if context_parallel_enabled:
            world_size = 8
            backbone_tp_degree = 4
        else:
            world_size = 4
            backbone_tp_degree = 4

    return world_size, backbone_tp_degree


def create_flux_config(model_path, world_size, backbone_tp_degree, dtype, height, width, inpaint=False):
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
        compiler_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "text_encoder"
        )
        self.pipe.text_encoder.compile(
            os.path.join(compiled_model_path, "text_encoder/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "text_encoder_2"
        )
        self.pipe.text_encoder_2.compile(
            os.path.join(compiled_model_path, "text_encoder_2/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "transformers"
        )
        self.pipe.transformer.compile(
            os.path.join(compiled_model_path, "transformer/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(compiler_workdir, "decoder")
        self.pipe.vae.decoder.compile(
            os.path.join(compiled_model_path, "decoder/"), debug, pre_shard_weights_hook
        )
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


class NeuronFluxFillApplication(nn.Module):
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
    ):
        super().__init__()
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path or os.path.join(
            model_path, "text_encoder"
        )
        self.text_encoder_2_path = text_encoder_2_path or os.path.join(
            model_path, "text_encoder_2"
        )
        self.transformer_path = transformer_path or os.path.join(
            model_path, "transformer"
        )
        self.vae_decoder_path = vae_decoder_path or os.path.join(model_path, "vae")

        self.height = height
        self.width = width
        self.max_sequence_length = 512

        self.pipe = NeuronFluxFillPipeline.from_pretrained(
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
        compiler_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "text_encoder"
        )
        self.pipe.text_encoder.compile(
            os.path.join(compiled_model_path, "text_encoder/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "text_encoder_2"
        )
        self.pipe.text_encoder_2.compile(
            os.path.join(compiled_model_path, "text_encoder_2/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "transformers"
        )
        self.pipe.transformer.compile(
            os.path.join(compiled_model_path, "transformer/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(compiler_workdir, "decoder")
        self.pipe.vae.decoder.compile(
            os.path.join(compiled_model_path, "decoder/"), debug, pre_shard_weights_hook
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = compiler_workdir

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
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


class NeuronFluxControlApplication(nn.Module):
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
    ):
        super().__init__()
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path or os.path.join(
            model_path, "text_encoder"
        )
        self.text_encoder_2_path = text_encoder_2_path or os.path.join(
            model_path, "text_encoder_2"
        )
        self.transformer_path = transformer_path or os.path.join(
            model_path, "transformer"
        )
        self.vae_decoder_path = vae_decoder_path or os.path.join(model_path, "vae")

        self.height = height
        self.width = width
        self.max_sequence_length = 512

        self.pipe = NeuronFluxControlPipeline.from_pretrained(
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
        compiler_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "text_encoder"
        )
        self.pipe.text_encoder.compile(
            os.path.join(compiled_model_path, "text_encoder/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "text_encoder_2"
        )
        self.pipe.text_encoder_2.compile(
            os.path.join(compiled_model_path, "text_encoder_2/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(
            compiler_workdir, "transformers"
        )
        self.pipe.transformer.compile(
            os.path.join(compiled_model_path, "transformer/"),
            debug,
            pre_shard_weights_hook,
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = os.path.join(compiler_workdir, "decoder")
        self.pipe.vae.decoder.compile(
            os.path.join(compiled_model_path, "decoder/"), debug, pre_shard_weights_hook
        )
        os.environ["BASE_COMPILE_WORK_DIR"] = compiler_workdir

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
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
