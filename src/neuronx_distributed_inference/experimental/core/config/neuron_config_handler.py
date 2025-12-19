import torch
import logging
from omegaconf import OmegaConf
from copy import deepcopy


logger = logging.getLogger("NeuronConfig")
logger.setLevel(logging.INFO)


"""
dtype attribute from YAML config needs to be resolved to a torch datatype object explicity.
This is because OmegaConf does not automatically know how to convert the string torch.bfloat16
into the actual torch.bfloat16 object. It tries to find a key named torch.bfloat16 within the
configuration or a pre-registered resolver and fails. Hence we must register a resolver
that can map a string to an attribute of the torch module.
"""
OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))


# TODO: Remove this function and replace it with new ones in examples
def load_yaml_config(cfg_path):
    return OmegaConf.load(cfg_path)


def load_neuron_config(config_file_path):
    """
    Load neuron configuration and apply model tag overrides if present.

    Loads a YAML configuration file and checks for config_override section.
    If overrides exist, processes them using parse_config_with_model_tags_overrides.

    Args:
        config_file_path (str): Path to the YAML configuration file.

    Returns:
        OmegaConf: Configuration object with overrides applied if present,
            otherwise returns the original loaded configuration.
    """
    logger.info(f"Loading config from {config_file_path}")
    config = OmegaConf.load(config_file_path)

    if "config_override" not in config:
        return config

    return parse_config_with_model_tags_overrides(config)


def parse_config_with_model_tags_overrides(config):
    """
    Parse config object and apply model-specific overrides based on tags.

    Processes a configuration object containing default configs and model-specific
    overrides. Creates separate configurations for different model tags while
    maintaining a default configuration.

    Args:
        config (OmegaConf): Configuration object containing base config and
            config_override section with model tag overrides.

    Returns:
        OmegaConf: Parsed configuration object containing:
            - 'model': Original model configuration
            - 'default': Base configuration with overrides removed
            - '{model_tag}': Model-specific configurations with overrides applied
    """
    # Create a default config copy which will be applied to all model tags
    default_config = deepcopy(config)
    del default_config.config_override
    del default_config.model

    parsed_config = OmegaConf.create()
    parsed_config['model'] = config.model
    parsed_config['default'] = default_config

    default_compiler_args = default_config.get("build", {}).get("compiler_args", [])

    # Loop over each override for model_tags
    config_override = deepcopy(config.config_override)
    for override in config_override:
        model_tags = override.model_tags
        del override.model_tags
        for model_tag in model_tags:
            # Initialize the config for model_tag with default values
            parsed_config[model_tag] = default_config

            # Loop over the override values and update model_tag config modules
            for section_name, updates in override.items():
                # If a module config is only provided for model_tag and is not a part of default config
                if section_name not in parsed_config[model_tag]:
                    parsed_config[model_tag][section_name] = updates
                else:
                    parsed_config[model_tag][section_name] = OmegaConf.merge(
                        parsed_config[model_tag][section_name], updates
                    )
                # TODO: By default, merge() is replacing the target list with the source list
                #  and does not extend it. Need to upgrade to OmegaConf 2.4.0 to be able to
                #  use ListMergeMode ENUM which extends list during merge().
                #  Hence, special handling is required for compiler args which is a list.
                if section_name == "build":
                    parsed_config[model_tag][section_name]["compiler_args"] = default_compiler_args + updates.get("compiler_args", [])

    return parsed_config


def get_config_for_model_tag(config, model_tag):
    """
    Returns the configuration for the specified model tag if it exists,
    otherwise returns the default configuration.

    Args:
        config (OmegaConf): Parsed configuration object containing model tags.
        model_tag (str): Name of the model tag to retrieve configuration for.

    Returns:
        OmegaConf: Configuration object for the specified model tag or
            default configuration if tag not found.
    """
    if model_tag in config:
        return config[model_tag]
    else:
        return config.default
