import json
import logging
import os

import torch

from neuronx_distributed.trace.trace import shard_children
from neuronx_distributed_inference.modules.checkpoint import _torch_load, load_file
from neuronx_distributed_inference.modules.attention.gqa import replicate_kv
from safetensors.torch import save_file
from typing import Optional

from .config import LoraServingConfig
from .lora_layer import MultiLoraColumnParallelLinear

logger = logging.getLogger("Neuron")


class LoraCheckpoint:
    def __init__(self, config: LoraServingConfig):
        self.lora_config = config
        self.lora_ckpts = None
        self.lora_ckpts_cpu = None
        if config is not None:
            self.ckpt_paths = config.lora_ckpt_paths
            self.ckpt_paths_cpu = config.lora_ckpt_paths_cpu

        self.lora_weights = {}
        self.lora_weights_active = {}
        self.lora_weights_cpu = {}

    def is_lora_module(self, name):
        keywords = ["lora_A", "lora_B"]
        return any(keyword in name for keyword in keywords)

    def load_lora_state_dicts(self, ckpt_paths):
        r"""
        We support two checkpoint formats for LoRA adapters:

        1. PEFT checkpoint format from Huggingface PEFT (https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint).
        Each checkpoint path is a folder that contains a checkpoint file (.safetensors, .bin, or .pt) and a configuration json file (.json).

        2. LoRA checkpoint format from NxD. Each checkpoint path is a checkpoint file (.pt) that includes both LoRA adapter weights and the configuration.
        """
        if ckpt_paths is None:
            return

        lora_ckpts = {}
        for key, path in ckpt_paths.items():
            assert os.path.exists(path)
            if os.path.isdir(path):
                lora_scaling, state_dict = self._load_lora_state_dict_from_folder(path)
            else:
                lora_scaling, state_dict = self._load_lora_state_dict_from_file(path)
            lora_ckpts[key] = {"lora_scaling": lora_scaling, "state_dict": state_dict}

        return lora_ckpts

    def _extract_lora_scaling(self, lora_adapter_config):
        if lora_adapter_config is not None:
            lora_alpha = lora_adapter_config.get("lora_alpha", 0)
            use_rslora = lora_adapter_config.get("use_rslora", False)
            return (lora_alpha, use_rslora)
        else:
            return None

    def _load_lora_state_dict_from_file(self, filename):
        if filename.endswith(".safetensors"):
            state_dict = load_file(filename)
        elif filename.endswith(".bin") or filename.endswith(".pt"):
            state_dict = _torch_load(filename)
        else:
            raise FileNotFoundError(f"Invalid checkpoint filename {filename} for LoRA adapter.")

        lora_adapter_config = state_dict.get("lora_config")
        lora_scaling = self._extract_lora_scaling(lora_adapter_config)
        state_dict.pop("lora_config", None)
        return lora_scaling, state_dict

    def _load_lora_state_dict_from_folder(self, path):
        lora_scaling, state_dict = None, None
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if filename.strip() == "adapter_config.json":
                with open(file_path) as f:
                    lora_adapter_config = json.load(f)
                    lora_scaling = self._extract_lora_scaling(lora_adapter_config)
            elif filename.endswith(".safetensors"):
                state_dict = load_file(file_path)
            elif filename.endswith(".bin") or filename.endswith(".pt"):
                state_dict = _torch_load(file_path)

        if state_dict is None:
            raise ValueError(
                f"No valid LoRA adapter checkpoint in {path}."
                f"Supported checkpoint formats include '*.safetensors', '*.bin', and '.pt'."
            )
        return lora_scaling, state_dict

    def _get_module_checkpoint(self, name, lora_ckpt):
        r"""
        We must match the NxDI LoRA module with its weights in lora_ckpt because the module name cannot exactly match its weight name in LoRA adapter checkpoints.
        For example, given a Llama2 model and one of its NxDI LoRA module is named as `layers.0.self_attn.o_proj.o_proj.lora_A`. However, its weight name in PEFT LoRA checkpoints is
        `base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight` and in NxD LoRA checkpoints is `model.layers.0.self_attn.o_proj.lora_A.weight`.

        We need to extract layer_module_name and lora_module_name from both NxDI LoRA module name and weight name for matching. In the above example, layer_module_name=`layers.0.self_attn`,
        which includes the layer number and the layer type; lora_module_name=`o_proj.lora_A`, which includes the base module type and LoRA type, i.e., LoRA A or LoRA B.

        We first obtain layer_module_name and lora_module_name from the NxDI LoRA module name;
        we then for loop all weights in lora_ckpt for matching by extracting layer_module_name and lora_module_name from their names.
        """
        state_dict = lora_ckpt["state_dict"]
        layer_module_name = self._get_layer_module_name(name)
        lora_module_name = self._get_lora_module_name_in_layer(name)

        for key, weight in state_dict.items():
            ckpt_layer_module_name = self._get_layer_module_name(key)
            ckpt_lora_module_name = self._get_lora_module_name_in_layer(key)
            if (
                layer_module_name == ckpt_layer_module_name
                and lora_module_name == ckpt_lora_module_name
            ):
                if "lora_A" in key:
                    lora_scaling = lora_ckpt.get("lora_scaling")
                    lora_rank = weight.shape[0]
                    scaling = 1.0  # default scaling

                    if lora_scaling is not None:
                        try:
                            lora_alpha, use_rslora = lora_scaling

                            if lora_alpha != 0:
                                scaling = (
                                    lora_alpha / lora_rank
                                    if not use_rslora
                                    else lora_alpha / lora_rank**0.5
                                )
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing lora_scaling: {e}")
                            # Keep default scaling of 1.0

                    # Apply scaling to the weights of LoRA A
                    weight.mul_(scaling)
                return weight
        return None

    def _get_base_module_type(self, name):
        return name.split(".lora")[0].split(".")[-1]

    def _get_layer_module_name(self, name):
        r"""
        There are two cases when extracting layer_module_name:
            1. keyword `layers` in name for recurrent layers, such as attentions layers. It returns the layer number and the layer name, like `0.self_attn`.
            2. keyword `layers` not in name, such as embedding layer. It returns the module type only, such as `embed_tokens`, which is right before the keyword `.lora`.
        """
        name = name.replace(".weight", "")
        if "layers." in name:
            # return the layer number and the layer name
            return ".".join(name.split("layers.")[1].split(".")[:2])
        else:
            # handle other layers, such as embedding
            return self._get_base_module_type(name)

    def _get_lora_module_name_in_layer(self, name):
        # Return lora_module_name
        module_type = self._get_base_module_type(name)
        # check if the lora module is LoRA A or LoRA B
        if "lora_A" in name:
            return f"{module_type}.lora_A"
        elif "lora_B" in name:
            return f"{module_type}.lora_B"
        else:
            return None

    def _get_module_checkpoints(self, name, ckpt_paths, lora_ckpts):
        ret = []
        for key in ckpt_paths:
            matched_ckpt = self._get_module_checkpoint(name, lora_ckpts[key])
            ret.append(matched_ckpt)
        return ret

    def _get_lora_weight(self, weights, id):
        return weights[id].squeeze()

    def _convert_lora_weight(self, ckpt, module, weight_dtype):
        tensor = ckpt.squeeze()
        # handle LoRA weights replications here
        if isinstance(module, MultiLoraColumnParallelLinear):
            kv_replicate = module.kv_replicate
            if kv_replicate is not None:
                source_heads, repeats = kv_replicate
                # we refer to gqa.py for the following
                if repeats > 1:
                    tensor, _ = replicate_kv(tensor, source_heads, repeats)
        if self.lora_config.lora_memory_transpose:
            tensor = tensor.t()
        return tensor.to(weight_dtype)

    def _load_lora_weights(self, model, ckpt_paths, lora_ckpts, lora_weights):
        if lora_ckpts is None:
            return

        for name, module in model.named_modules():
            if self.is_lora_module(name):
                checkpoints = self._get_module_checkpoints(name, ckpt_paths, lora_ckpts)
                weight_name = f"{name}.weight"
                weights = lora_weights[weight_name]
                weight_dtype = module.get_weight_dtype()

                for i, ckpt in enumerate(checkpoints):
                    if ckpt is not None:
                        lora_weight = self._get_lora_weight(weights, i)
                        ckpt = self._convert_lora_weight(ckpt, module, weight_dtype)
                        shape = ckpt.size()
                        # pad LoRA checkpoint into LoRA weights
                        lora_tensor = lora_weight[: shape[0], : shape[1]]
                        lora_tensor.copy_(ckpt)

    def _update_scale_for_quantized_model(self, model_sd):
        names = list(model_sd.keys())
        for name in names:
            r"""
            For quantized models, the scale parameter name includes ".scale", e.g., "model.layers.0.self_attn.qkv_proj.q_proj.scale".
            After adding LoRA module, the scale parameter name should be "model.layers.0.self_attn.qkv_proj.q_proj.base_layer.scale".
            If the weight of the module with the scale is still in the model state dict, it implies this module has no LoRA module and there is no need to update its name.
            """
            if ".scale" in name and name.replace(".scale", ".weight") not in model_sd:
                new_name = name.replace(".scale", ".base_layer.scale")
                model_sd[new_name] = model_sd.pop(name)
        return model_sd

    def update_weights_for_lora(self, model, model_sd):
        # step 1: load state dicts from ckpt paths
        if not self.lora_ckpts:
            self.lora_ckpts = self.load_lora_state_dicts(self.ckpt_paths)

        # step 2: update the weight name for base modules because the module name are modified by LoRA
        for name, _ in model.named_modules():
            if ".base_layer" in name:
                name = name.replace(".base_layer", "")
                weight_name = f"{name}.weight"
                lora_weight_name = f"{name}.base_layer.weight"

                if lora_weight_name not in model_sd:
                    model_sd[lora_weight_name] = model_sd.pop(weight_name)

        # update the scale parameter names for quantized base modules
        model_sd = self._update_scale_for_quantized_model(model_sd)

        # step 3: initialize LoRA adapter weights
        for name, module in model.named_modules():
            if self.is_lora_module(name):
                weight_shape = module.get_checkpoint_shape()
                weight_shape_active = module.get_checkpoint_shape_active()
                weight_dtype = module.get_weight_dtype()
                weight_name = f"{name}.weight"
                if weight_name not in model_sd:
                    self.lora_weights[weight_name] = torch.zeros(
                        *weight_shape, dtype=weight_dtype, device="cpu"
                    )
                    self.lora_weights_active[weight_name + "_active"] = torch.zeros(
                        *weight_shape_active, dtype=weight_dtype, device="cpu"
                    )

        # step 4: load LoRA checkpoints into the LoRA weights
        self._load_lora_weights(model, self.ckpt_paths, self.lora_ckpts, self.lora_weights)

        # step 5: add LoRA adapter weights into model_sd
        model_sd.update(self.lora_weights)
        model_sd.update(self.lora_weights_active)

        return model_sd

    def update_weights_for_lora_cpu(self, model):
        # step 1: load state dicts from ckpt paths
        if not self.lora_ckpts_cpu:
            self.lora_ckpts_cpu = self.load_lora_state_dicts(self.ckpt_paths_cpu)

        # step 2: initialize LoRA CPU adapter weights
        for name, module in model.named_modules():
            if self.is_lora_module(name):
                weight_shape = module.get_checkpoint_shape()
                weight_shape_cpu = (self.lora_config.max_cpu_loras, ) + weight_shape[1:]
                weight_dtype = module.get_weight_dtype()
                weight_name = f"{name}.weight"
                self.lora_weights_cpu[weight_name] = torch.zeros(
                    *weight_shape_cpu, dtype=weight_dtype, device="cpu"
                )

        # step 3: load CPU LoRA checkpoints into the CPU LoRA weights
        self._load_lora_weights(model, self.ckpt_paths_cpu, self.lora_ckpts_cpu, self.lora_weights_cpu)

    def shard_cpu_checkpoints(self, start_rank_id, local_ranks_size, tp_deg, model, serialize_path=None):
        sharded_lora_cpu_checkpoints = []

        for rank in range(start_rank_id, start_rank_id + local_ranks_size):
            sharded_lora_cpu_checkpoints.append(self._shard_cpu_weights(rank, tp_deg, model, self.lora_weights_cpu, self.ckpt_paths_cpu.keys(), serialize_path))

        return sharded_lora_cpu_checkpoints

    def _shard_cpu_weights(self, rank, tp_deg, model, checkpoint, adapter_names, serialize_path: Optional[str] = None) -> None:
        sharded_checkpoint = checkpoint.copy()

        # Shards the checkpoint to the right weight for the rank
        shard_children(model, sharded_checkpoint, "", None, rank, tp_deg, True)

        if serialize_path is not None:
            assert len(adapter_names) > 0
            for i, adapter_name in enumerate(adapter_names):
                filename = f"{adapter_name}_tp{rank}_sharded_lora_cpu_checkpoint.safetensors"
                save_dict = {}
                for k, v in sharded_checkpoint.items():
                    if "lora_A" in k or "lora_B" in k:
                        save_dict[k] = v.contiguous()
                save_file(save_dict, os.path.join(serialize_path, filename))

        return sharded_checkpoint

    def load_sharded_cpu_checkpoints(self, compiled_model_path, start_rank_id, local_ranks_size):
        logger.info(
            f"Loading presharded CPU LoRA adapter checkpoints for ranks: {start_rank_id}...{start_rank_id + local_ranks_size - 1}"
        )

        sharded_lora_cpu_weights = []

        for rank in range(start_rank_id, start_rank_id + local_ranks_size):
            sharded_lora_cpu_weights.append(self._load_rank_checkpoint(compiled_model_path, rank))

        return sharded_lora_cpu_weights

    def _load_rank_checkpoint(self, compiled_model_path, rank):
        load_dict = {}
        for _, adapter_name in enumerate(self.ckpt_paths_cpu.keys()):
            lora_cpu_ckpt = load_file(
                os.path.join(
                    compiled_model_path,
                    f"weights/{adapter_name}_tp{rank}_sharded_lora_cpu_checkpoint.safetensors",
                )
            )
            for k, v in lora_cpu_ckpt.items():
                if "lora_A" in k or "lora_B" in k:
                    if k not in load_dict:
                        load_dict[k] = []
                    load_dict[k].append(v)
                else:
                    load_dict[k] = v

        for k, v in load_dict.items():
            if "lora_A" in k or "lora_B" in k:
                for _ in range(
                    len(load_dict[k]),
                    self.lora_config.max_cpu_loras,
                ):
                    load_dict[k].append(
                        torch.zeros(load_dict[k][0].shape, dtype=load_dict[k][0].dtype)
                    )
                load_dict[k] = torch.stack(load_dict[k])

        return load_dict
