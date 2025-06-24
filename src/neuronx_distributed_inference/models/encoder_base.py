import copy
import logging
import os
from typing import List, Union

from torch import nn

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronEncoderBase(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()

    def forward():
        raise NotImplementedError("forward() is not implemented")


class NeuronEncoderApplication(NeuronApplicationBase):
    """
    Encoder applications/piplines should inherent/copy this appliation class.
    Name the child class similar to HF convention:
    Just output embedding: NeuronXXXModel;
    or based on the application: NeuronXXXForXXXX or NeuronXXXPipeline.
    eg. NeuronViTModel, NeuronViTForImageClassification, NeuronDiTPipeline.
    """

    def __init__(
        self,
        model_path: str,
        config: InferenceConfig = None,
        neuron_config: NeuronConfig = None,
    ):
        super().__init__(model_path, config, neuron_config)

        # Copied from NeuronBaseForCausalLM.__init__()
        # async related
        self.async_mode = self.neuron_config.async_mode
        self.next_cpu_inputs = None
        self.prior_outputs = None
        self.unequal_batching = (
            self.neuron_config.ctx_batch_size != self.neuron_config.tkg_batch_size
        )
        if self.async_mode:
            os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "2"

        self.model_wrappers = self.get_model_wrapper_cls()
        self.enable_models()

    def get_model_wrapper_cls(self) -> List[List[Union[nn.Module, ModelWrapper]]]:
        """
        Returns:
            List[List[Union[nn.Module, ModelWrapper]]]: A list of all the sub-models in this encoder application
            [[model_cls, model_wrapper_cls]]

        Example:
        return [
            [DiT, ModelWrapperDiT],
            [VAE, ModelWrapperVAE],
            [CLIP, ModelWrapperClipTextEncoder],
        ]
        """
        raise NotImplementedError("get_model_wrapper_cls() is not implemented")

    def enable_models(self, **model_init_kwargs):
        for model_cls, model_wrapper_cls in self.model_wrappers:
            new_config = copy.deepcopy(self.config)

            new_model = model_wrapper_cls(
                config=new_config,
                model_cls=model_cls,
                tag=model_cls.__name__,
                compiler_args=self.get_compiler_args(),  # FIXME: should different sub-model / different hlo use different compiler args?
                model_init_kwargs=model_init_kwargs,
            )
            setattr(
                self, model_cls.__name__, new_model
            )  # FIXME: shoule we use tag? eg. DIFFUSION_BACKBONE_TAG = "diffusion_backbone"
            self.models.append(new_model)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1 "

        # Flag for model type
        compiler_args += "--model-type=transformer "
        compiler_args += " --auto-cast=none"

        return compiler_args

    def forward(self, *inputs):
        """
        Args: inputs: List[torch.Tensor]
        """
        raise NotImplementedError("forward() is not implemented")
