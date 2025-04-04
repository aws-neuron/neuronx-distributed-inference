import unittest
from functools import partial

import pytest
import torch
import torch.distributed
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers import ColumnParallelLinear
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder

from neuronx_distributed_inference.modules.attention.utils import distributed_softmax
from neuronx_distributed_inference.modules.generation.sampling import mask_padded_logits


class Model(torch.nn.Module):
    def __init__(self, pad_size=0, tp_degree=64):
        super().__init__()
        self.tp_degree = tp_degree
        self.pad_size = pad_size

        self.rank_util = SPMDRank(self.tp_degree)

    def cpu_forward(self, logits):
        return mask_padded_logits(logits, torch.tensor(0), 1, pad_size=self.pad_size)

    def forward(self, logits):
        logits = _reduce_scatter_along_dim(
            logits,
            1,
            computation=xm.REDUCE_MAX,
            process_group=parallel_state.get_tensor_model_parallel_group(as_list=False),
        )
        rank_id = self.rank_util.get_rank()
        world_size = self.tp_degree

        logits = mask_padded_logits(logits, rank_id, world_size, pad_size=self.pad_size)
        logits = _gather_along_dim(
            logits, 1, process_group=parallel_state.get_tensor_model_parallel_group(as_list=False)
        )
        return logits


def get_ckpt_func(tp_degree):
    def get_ckpt():
        return {"rank_util.rank": torch.arange(0, tp_degree, dtype=torch.int32)}

    return get_ckpt


@pytest.mark.xfail(reason="not valid test for trn1.2xlarge", strict=False)
@pytest.mark.parametrize(
    "batch_size, vocab_size, pad_size, tp_degree, logical_nc_config",
    [
        (1, 129832, 24, 1, 1),
        (1, 129832, 24, 2, 1),
        (1, 129832, 24, 32, 1),
        (1, 129832, 24, 64, 2),
        (1, 129832, 88, 64, 2),
        (2, 129832, 88, 64, 2),
    ],
)
def test_logits_masking(batch_size, vocab_size, pad_size, tp_degree, logical_nc_config):
    model = Model(pad_size=pad_size, tp_degree=tp_degree)

    torch.save(model.state_dict(), "/tmp/model.pt")
    model.load_state_dict(torch.load("/tmp/model.pt"))

    logits = torch.cat(
        [torch.rand((batch_size, vocab_size)), torch.zeros((batch_size, pad_size))], dim=1
    )

    # run cpu golden
    model.eval()
    cpu_output = model.cpu_forward(logits)

    # run Neuron
    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=get_ckpt_func(tp_degree),
        logical_nc_config=logical_nc_config,
    )
    model_cls = partial(Model, pad_size=pad_size, tp_degree=tp_degree)

    active_shape = (batch_size, vocab_size + pad_size)
    builder.add(
        key="main",
        model_instance=BaseModelInstance(module_cls=model_cls, input_output_aliases={}),
        example_inputs=[(torch.ones(active_shape, dtype=torch.float32),)],
        priority_model_idx=0,
    )

    traced_model = builder.trace(initialize_model_weights=True)
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))

    traced_output = traced_model(logits)

    torch.testing.assert_close(
        cpu_output,
        traced_output.to("cpu"),
        atol=1e-3,
        rtol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
