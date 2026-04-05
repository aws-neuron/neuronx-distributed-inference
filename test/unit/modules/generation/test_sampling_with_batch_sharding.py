import unittest
import os
from functools import partial
from copy import deepcopy
import pytest
import torch
import torch_neuronx

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, SPMDRank
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch import nn
from torch_neuronx.utils import get_platform_target


from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.modules.generation.sampling import (
    create_sampler,
    prepare_sampling_params
)
torch.manual_seed(42)

def get_lm_head_pad_config(vocab_size: int, tp_degree: int, lm_head_pad_alignment_size: int = 1, skip_lm_head_pad: bool = False):
    """
    Check if lm_head padding is necessary to achieve good performance.

    Args:
        vocab_size (int): vocabulary size of the model
        tp_degree (int): tp_degree used for lm_head
        lm_head_pad_alignment_size (int): usually you want to set this to LNC degree
        skip_lm_head_pad (bool): always disable padding (for debug purpose)

    Returns:
        (bool, int): Tuple indiciating if we should pad and what the pad_alignment_size should be.
    """
    if vocab_size % (tp_degree * lm_head_pad_alignment_size) == 0 or skip_lm_head_pad:
        return False, 1

    return True, lm_head_pad_alignment_size

class SamplingModel(nn.Module):
    def __init__(self, is_distributed=False, batch_sharding=True, config=None):
        super().__init__()
        self.config = config
        self.batch_sharding = batch_sharding
        if is_distributed:
            should_pad_lm_head, lm_head_pad_alignment_size = get_lm_head_pad_config(
                vocab_size=config.vocab_size,
                tp_degree=config.neuron_config.tp_degree,
                lm_head_pad_alignment_size=config.neuron_config.lm_head_pad_alignment_size * config.neuron_config.logical_nc_config,
                skip_lm_head_pad=not config.neuron_config.lm_head_pad)
            
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=False,
                bias=should_pad_lm_head,
                pad=True,
                pad_alignment_size_per_rank=lm_head_pad_alignment_size,
                keep_padded_output=should_pad_lm_head,
                dtype=config.neuron_config.torch_dtype,
                tensor_model_parallel_group=get_tp_group(config),
            )

            self.lm_head.rank_util = SPMDRank(world_size=config.neuron_config.tp_degree)

        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        self.lm_head.training = False

    def forward(self, input_ids, sampling_params):
        logits = self.lm_head(input_ids)
        lm_head_tp_degree = None
        if hasattr(self, "lm_head") and hasattr(self.lm_head, "tensor_parallel_group"):
            lm_head_tp_degree = self.lm_head.tensor_parallel_group.size.return_value
        sampler = create_sampler(self.config.neuron_config, lm_head_tp_degree)
        output = sampler(logits, sampling_params, rank_id=self.lm_head.rank_util.get_rank(), return_values=False)
        return output

    @torch.inference_mode()
    def cpu_forward(self, input_ids, sampling_params):
        logits = self.lm_head(input_ids)
        sampler = create_sampler(self.config.neuron_config, None)
        output = sampler(logits, sampling_params)
        return output

def rand_interval(a, b, *size):
  return (b - a) * torch.rand(*size) + a

@pytest.mark.parametrize(
    "batch_size, tp_degree, sampling_dp_degree, num_dims",
    [   
        (1, 64, 1, 2),
	    (4, 64, 1, 2),
        (8, 64, 8, 2),
        (16, 64, 8, 2),
        (32, 64, 8, 2),
        (64, 64, 4, 2),
        (64, 64, 8, 2),
        (64, 64, 16, 2),
        (1, 64, 1, 3),
	    (4, 64, 1, 3),
        (8, 64, 8, 3),
        (16, 64, 8, 3),
        (32, 64, 8, 3),
        (64, 64, 4, 3),
        (64, 64, 8, 3),
        (64, 64, 16, 3),
    ],
)
def test_sampling_batch_sharding(batch_size, tp_degree, sampling_dp_degree, num_dims):
    hardware = get_platform_target()
    if hardware == "trn1" and tp_degree == 64:
        pytest.skip("Not supported in trn1")
    if hardware == "trn2" and tp_degree == 32:
        pytest.skip("Not supported in trn2")

    def get_ckpt():
        model_sd = torch.load("/tmp/model.pt")
        model_sd["lm_head.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return model_sd

    hidden_size = 3072
    if num_dims == 3:
        input_shape = (batch_size, 1, hidden_size)
    else:
        input_shape = (batch_size, hidden_size)

    config_dict = {
        "world_size": 64,
        "hidden_size": hidden_size,
        "num_attention_heads": 32,
        "num_hidden_layers": 1,
        "num_key_value_heads": 8,
        "pad_token_id": 0,
        "vocab_size": 201088,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-05,
        "hidden_act": "silu",
        "tp_degree": tp_degree,
        "torch_dtype": torch.float32,
        "batch_size": batch_size      
    }
    config_dict["on_device_sampling_config"] = {
        "top_k_kernel_enabled": False,
        "global_topk": 256,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.7,
        "sampling_dp_degree": sampling_dp_degree
    }

    neuron_config = NeuronConfig(**config_dict)
    config = InferenceConfig(neuron_config=neuron_config, **config_dict)
    sampling_params = prepare_sampling_params(batch_size=batch_size)

    cpu_config = deepcopy(config)
    cpu_config.neuron_config.on_cpu = True
    model = partial(SamplingModel, is_distributed=False, config=cpu_config)()
    model.lm_head.weight = torch.nn.Parameter(rand_interval(-0.05, 0.05, (model.lm_head.weight.shape)))
    torch.save(model.state_dict(), "/tmp/model.pt")
    model.load_state_dict(model.state_dict())

    input_ids = torch.randn(input_shape)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=get_ckpt,
        num_cores_per_group=1,
        logical_nc_config=2,
    )

    
    model_cls = partial(SamplingModel, is_distributed=True, config=config)

    builder.add(
        key="main",
        model_instance=BaseModelInstance(module_cls=model_cls, input_output_aliases={}),
        example_inputs=[(input_ids,sampling_params,)],
        priority_model_idx=None,
    )

    traced_model = builder.trace(initialize_model_weights=True)
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
    traced_output = traced_model(input_ids, sampling_params)
    traced_output = traced_output.to(torch.int64)
    model.eval()
    cpu_output = model.cpu_forward(input_ids, sampling_params)
    torch_neuronx.testing.assert_close(
        cpu_output,
        traced_output,
        atol=1e-3,
        rtol=1e-3,
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])