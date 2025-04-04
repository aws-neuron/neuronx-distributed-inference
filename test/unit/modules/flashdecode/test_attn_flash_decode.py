import unittest
from functools import partial

import pytest
import torch
import torch.distributed
from neuronx_distributed.parallel_layers import ColumnParallelLinear
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder

from neuronx_distributed_inference.modules.attention.utils import distributed_softmax


class Model(torch.nn.Module):
    def __init__(self, is_distributed=False, num_heads=None, past_sequence_length=None, dtype=None):
        super().__init__()
        if is_distributed:
            self.prior_scores = ColumnParallelLinear(
                input_size=num_heads,
                output_size=past_sequence_length,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
        else:
            self.prior_scores = torch.nn.Linear(
                in_features=num_heads, out_features=past_sequence_length, bias=False, dtype=dtype
            )
        self.num_heads = num_heads

    def forward(self, active_scores=None):
        out = distributed_softmax(
            self.prior_scores.weight.t(), active_scores.expand(self.num_heads, 1)
        )
        return out

    def cpu_forward(self, active_scores):
        score = torch.concat(
            [self.prior_scores.weight.t(), active_scores.expand(self.num_heads, 1)], dim=-1
        )

        softmax_score = torch.softmax(score, dim=-1, dtype=torch.float32).to(score.dtype)
        return (softmax_score[..., :-1], softmax_score[..., -1:])


def get_ckpt():
    model_sd = torch.load("/tmp/model.pt")
    return model_sd


@pytest.mark.parametrize(
    "num_heads, past_sequence_length, tp_degree",
    [
        (2, 128, 2),
        (2, 256, 2),
        (4, 128, 2),
        (4, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_shard_over_sequence(num_heads, past_sequence_length, tp_degree, dtype):
    num_kv_heads = 1
    active_seq_len = 1
    active_shape = (num_kv_heads, active_seq_len)

    model = partial(
        Model,
        is_distributed=False,
        num_heads=num_heads,
        past_sequence_length=past_sequence_length,
        dtype=dtype,
    )()
    torch.save(model.state_dict(), "/tmp/model.pt")
    model.load_state_dict(get_ckpt())

    builder = ModelBuilder(
        router=None, tp_degree=tp_degree, checkpoint_loader=get_ckpt, num_cores_per_group=2
    )
    model_cls = partial(
        Model,
        is_distributed=True,
        num_heads=num_heads,
        past_sequence_length=past_sequence_length,
        dtype=dtype,
    )
    builder.add(
        key="main",
        model_instance=BaseModelInstance(module_cls=model_cls, input_output_aliases={}),
        example_inputs=[(torch.ones(active_shape, dtype=dtype),)],
        priority_model_idx=0,
    )

    traced_model = builder.trace(initialize_model_weights=True)
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))

    active_scores = torch.randn(active_shape, dtype=dtype)
    traced_output = traced_model(active_scores)

    print(f" traced_output shape {traced_output[0].shape} {traced_output[1].shape}")

    model.eval()
    cpu_output = model.cpu_forward(active_scores)
    print(f" cpu_output shape {cpu_output[0].shape} {cpu_output[1].shape}")
    torch.testing.assert_close(
        cpu_output[0][..., : (past_sequence_length // tp_degree)],
        traced_output[0].to("cpu"),
        atol=1e-3,
        rtol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
