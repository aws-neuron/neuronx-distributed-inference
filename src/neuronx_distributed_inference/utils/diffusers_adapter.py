from diffusers.configuration_utils import ConfigMixin
from typing import Optional, Union
import os
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
)


def load_diffusers_config(
    model_path_or_name: Optional[Union[str, os.PathLike]] = None,
    hf_config: Optional[ConfigMixin] = None,
):
    """Return a load_config hook for InferenceConfig that loads the config from a config.json for diffuser models."""
    class DiffusersConfig(ConfigMixin):
        config_name = "config.json"

        def __init__(self):
            super().__init__()

    def load_config(self: InferenceConfig):
        if (model_path_or_name is None and hf_config is None) or (
            model_path_or_name is not None and hf_config is not None
        ):
            raise ValueError('Please provide only one of "model_path_or_name" or "hf_config"')

        if model_path_or_name is not None:
            config = DiffusersConfig()
            config = config.load_config(model_path_or_name)
        else:
            config = hf_config
        config["_name_or_path"] = model_path_or_name  # we need this attribute to load weight.
        self.__dict__.update(config)
    return load_config
