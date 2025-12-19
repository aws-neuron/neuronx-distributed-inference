import os
import logging
import pytest
import torch
import tempfile

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import (
    NeuronGptOssMoE,
    GptOssRMSNormV2,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.testing import build_module, destroy_mp, init_cpu_env
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig as HFGptOssConfig
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BS = 1
SEQ_LEN = 128
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 3072
NUM_EXPERTS = 128
EXPERTS_PER_TOKEN = 4
DTYPE = torch.float32


@pytest.fixture
def test_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(autouse=True)
def seed_every_test():
    torch.manual_seed(0)


def _rand_interval(a, b, *size):
    return (b - a) * torch.rand(*size) + a


def generate_ckpt(layer_idx, test_dir):
    gpt_oss_hf_config = HFGptOssConfig()
    gpt_oss_hf_config.hidden_size = HIDDEN_SIZE
    gpt_oss_hf_config.intermediate_size = INTERMEDIATE_SIZE
    gpt_oss_hf_config.num_local_experts = NUM_EXPERTS
    gpt_oss_hf_config.num_experts_per_tok = EXPERTS_PER_TOKEN

    decoder_layer_hf = GptOssDecoderLayer(gpt_oss_hf_config, layer_idx).to(DTYPE)
    mlp = decoder_layer_hf.mlp

    logger.info(f"MLP type: {type(mlp)}")
    logger.info(f"MLP attributes: {[attr for attr in dir(mlp) if not attr.startswith('_')]}")

    mlp.router.weight = torch.nn.Parameter(
        _rand_interval(-0.05, 0.05, mlp.router.weight.shape).to(DTYPE)
    )
    if mlp.router.bias is not None:
        mlp.router.bias = torch.nn.Parameter(
            _rand_interval(-0.05, 0.05, mlp.router.bias.shape).to(DTYPE)
        )

    experts = mlp.experts
    experts.gate_up_proj = torch.nn.Parameter(
        _rand_interval(-0.05, 0.05, experts.gate_up_proj.shape).to(DTYPE)
    )
    if experts.gate_up_proj_bias is not None:
        experts.gate_up_proj_bias = torch.nn.Parameter(
            _rand_interval(-0.05, 0.05, experts.gate_up_proj_bias.shape).to(DTYPE)
        )

    experts.down_proj = torch.nn.Parameter(
        _rand_interval(-0.05, 0.05, experts.down_proj.shape).to(DTYPE)
    )
    if experts.down_proj_bias is not None:
        experts.down_proj_bias = torch.nn.Parameter(
            _rand_interval(-0.05, 0.05, experts.down_proj_bias.shape).to(DTYPE)
        )

    mlp_state_dict = mlp.state_dict()

    checkpoint = {"rmsnorm.weight": torch.nn.Parameter(torch.ones(HIDDEN_SIZE).to(torch.float32))}

    for key, value in mlp_state_dict.items():
        if "router" in key:
            checkpoint[key] = value.clone()
        elif "expert" in key:
            checkpoint[key] = value.clone()

    torch.save(checkpoint, os.path.join(test_dir, "checkpoint.pt"))

    return decoder_layer_hf


def check_accuracy(target_model, cpu_model, batch_size, seq_len):

    # Collect CPU goldens
    test_inputs_and_goldens = []
    for _ in range(1):
        hidden = (torch.randn((batch_size, seq_len, HIDDEN_SIZE)) * 0.05).to(DTYPE)
        expected_outputs = cpu_model.forward(hidden)
        print(f"output[1] hf: {expected_outputs[1]}")
        test_inputs_and_goldens.append((hidden, expected_outputs))

    for hidden, ref_outputs in test_inputs_and_goldens:
        neuron_outputs = target_model(hidden)
        ref_hidden, ref_expert_affinities = ref_outputs
        neuron_hidden, neuron_expert_affinities = neuron_outputs

        passed, max_error = check_accuracy_embeddings(
            ref_expert_affinities, neuron_expert_affinities
        )
        assert (
            passed
        ), f"ref expert_affinities doesn't match neuron_expert_affinities! Max error: {max_error}"

        assert (
            neuron_hidden.shape == ref_hidden.shape
        ), f"Hidden shape mismatch: Neuron {neuron_hidden.shape}, Reference {ref_hidden.shape}"
        assert neuron_hidden.dtype == ref_hidden.dtype

        passed, max_error = check_accuracy_embeddings(ref_hidden, neuron_hidden, plot_outputs=True)
        assert (
            passed
        ), f"ref MoE module output doesn't match neuron MoE module output! Max error: {max_error}"


class HFMoE(torch.nn.Module):
    """HF MoE wrapper using GptOssDecoderLayer"""

    def __init__(self, layer_idx, decoder_layer):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = decoder_layer.mlp
        self.rmsnorm = torch.nn.RMSNorm(HIDDEN_SIZE, eps=1e-5).to(DTYPE)

    def forward(self, hidden_states):
        normed_hidden_states = self.rmsnorm(hidden_states)
        output = self.mlp(normed_hidden_states)

        hidden_states = output[0]
        expert_affinities = output[1]
        return hidden_states, expert_affinities


class NxDMoE(torch.nn.Module):
    def __init__(self, layer_idx, neuron_config, test_dir, neuron_moe=None, cpu_mode=False):
        super().__init__()
        self.layer_idx = layer_idx

        config = InferenceConfig(neuron_config)
        config.hidden_size = HIDDEN_SIZE
        config.intermediate_size = INTERMEDIATE_SIZE
        config.num_local_experts = NUM_EXPERTS
        config.num_experts_per_tok = EXPERTS_PER_TOKEN
        config.hidden_act = "sigmoid"
        config.n_shared_experts = 0

        self.cpu_mode = cpu_mode

        # TODO: add padded test
        self.neuron_moe = neuron_moe

        if neuron_moe is None:
            norm = GptOssRMSNormV2(HIDDEN_SIZE, 1e-05)
            self.neuron_moe = NeuronGptOssMoE(config=config, rmsnorm=norm)

        state_dict = torch.load(os.path.join(test_dir, "checkpoint.pt"), map_location="cpu")
        self.model_state_dict = dict(state_dict)

    def preshard_hook_cpu(self):
        def reorder_interleaved(weight):
            """
            Convert interleaved layout [g0,u0,g1,u1,...]
            → contiguous halves [g0,g1,...,u0,u1,...].
            Works with arbitrary leading dims.
            """
            gate = weight[..., ::2]  # even positions
            up = weight[..., 1::2]  # odd positions
            return torch.cat([gate, up], dim=-1).contiguous()

        # Standard passthrough weights that don't need reordering
        weight_mappings = [
            ("rmsnorm.weight", self.neuron_moe.moe.rmsnorm, "weight"),
            ("router.weight", self.neuron_moe.moe.router.linear_router, "weight"),
            ("router.bias", self.neuron_moe.moe.router.linear_router, "bias"),
            ("experts.down_proj", self.neuron_moe.moe.expert_mlps.mlp_op.down_proj, "weight"),
            ("experts.down_proj_bias", self.neuron_moe.moe.expert_mlps.mlp_op.down_proj, "bias"),
        ]

        # Handle everything except gate_up_proj first
        for checkpoint_key, target_module, attr_name in weight_mappings:
            assert (
                checkpoint_key in self.model_state_dict
            ), f"{checkpoint_key} missing in model_state_dict"
            weight = self.model_state_dict[checkpoint_key].contiguous()
            setattr(target_module, attr_name, torch.nn.Parameter(weight))

        # Special-case gate_up_proj projection weight
        gu_key = "experts.gate_up_proj"
        assert gu_key in self.model_state_dict, f"{gu_key} missing"

        W = self.model_state_dict[gu_key]
        W_reordered = reorder_interleaved(W)

        self.neuron_moe.moe.expert_mlps.mlp_op.gate_up_proj.weight = torch.nn.Parameter(W_reordered)

        # Special-case bias (same layout interleaving)
        gu_b_key = "experts.gate_up_proj_bias"
        if gu_b_key in self.model_state_dict:
            b = self.model_state_dict[gu_b_key]
            b_reordered = reorder_interleaved(b)
            # add the +1 part for up here in weight loading
            # in cpu mode, the preshard hook of the ExpertMLPsV2 won't be called, thus we need to bake the +1 here
            b_reordered[..., HIDDEN_SIZE:] += 1.0

        self.neuron_moe.moe.expert_mlps.mlp_op.gate_up_proj.bias = torch.nn.Parameter(b_reordered)

    def forward(self, hidden_states):
        if self.cpu_mode:
            self.preshard_hook_cpu()
        # router_logits: the dense router logits (before topk and softmax.expert_affinities returned by router is what you want.
        hidden_states, router_logits, expert_index = self.neuron_moe(hidden_states)
        expert_affinities = torch.zeros_like(router_logits, dtype=torch.float32)

        topk_weights, expert_index = torch.topk(
            router_logits.to(torch.float32), EXPERTS_PER_TOKEN, dim=1
        )
        topk_affinities = F.softmax(topk_weights, dim=1, dtype=torch.float32)
        expert_affinities = expert_affinities.scatter_(1, expert_index, topk_affinities)
        return hidden_states, expert_affinities


@pytest.mark.parametrize(
    "layer_idx,batch_size,seq_len", [(1, 1, 128), (1, 32, 1)]  # prefill scenario  # decode scenario
)
def test_moe_neuron_cpu_vs_ref_cpu(layer_idx, batch_size, seq_len, test_dir):
    try:
        init_cpu_env()
        decoder_layer_hf = generate_ckpt(layer_idx, test_dir)

        hf_moe = HFMoE(layer_idx, decoder_layer_hf).to(DTYPE)

        neuron_config = MoENeuronConfig(
            moe_ep_degree=1,
            torch_dtype=DTYPE,
            world_size=1,
            glu_type="swiglu",
            hidden_act_scaling_factor=1.702,
            return_router_logits=True,
            return_expert_index=True,
        )

        neuron_moe_cpu = NxDMoE(layer_idx, neuron_config, test_dir, cpu_mode=True)
        check_accuracy(neuron_moe_cpu, hf_moe, batch_size, seq_len)
    finally:
        destroy_mp()


def customize_loader(checkpoint_path):
    hf_state_dict = torch.load(checkpoint_path, map_location="cpu")
    neuron_state_dict = {}

    def reorder_interleaved(weight):
        """
        Convert interleaved [g0,u0,g1,u1,...] layout
        into contiguous [g0,g1,...,u0,u1,...].
        Works for both [out,in] and [*, 2*d] shapes.
        """
        # last dimension holds interleaving
        gate = weight[..., ::2]  # even indices
        up = weight[..., 1::2]  # odd indices
        return torch.cat([gate, up], dim=-1)

    # main mappings
    weight_mappings = [
        ("rmsnorm.weight", "moe.rmsnorm.weight"),
        ("router.weight", "moe.router.linear_router.weight"),
        ("router.bias", "moe.router.linear_router.bias"),
        ("experts.down_proj", "moe.expert_mlps.mlp_op.down_proj.weight"),
        ("experts.down_proj_bias", "moe.expert_mlps.mlp_op.down_proj.bias"),
    ]

    # regular 1-to-1 copies
    for ckpt_key, target_key in weight_mappings:
        assert ckpt_key in hf_state_dict, f"{ckpt_key} missing"
        neuron_state_dict[target_key] = hf_state_dict[ckpt_key]

    gu_ckpt = "experts.gate_up_proj"
    gu_bias = "experts.gate_up_proj_bias"

    assert gu_ckpt in hf_state_dict, f"{gu_ckpt} missing"
    W = hf_state_dict[gu_ckpt]

    # reorder interleaved → contiguous halves
    W_reordered = reorder_interleaved(W)

    neuron_state_dict["moe.expert_mlps.mlp_op.gate_up_proj.weight"] = W_reordered

    # bias (also interleaved the same way)
    if gu_bias in hf_state_dict:
        b = hf_state_dict[gu_bias]
        b_reordered = reorder_interleaved(b)
        # For Neuron, the preshard hook of the ExpertMLPsV2 will be called to bake the +1 to the bias,
        # thus we don't need +1 here
        # b_reordered[..., HIDDEN_SIZE:] += 1.0
        neuron_state_dict["moe.expert_mlps.mlp_op.gate_up_proj.bias"] = b_reordered

    return neuron_state_dict


@pytest.mark.parametrize(
    "layer_idx,batch_size,seq_len, tp_degree, ep_degree, moe_fused_nki_kernel_enabled",
    [
        (1, 1, 128, 16, 1, False),  # tp16.json, prefill
        (1, 1, 1, 16, 1, False),  # tp16.json, decode
        (1, 64, 1, 4, 16, False),  # high_throughput.json, decode
        (1, 64, 1, 64, 1, False),  # baseline_neuron.json, decode
    ],
)
def test_moe_neuron_selective_loading_tp16_flat_compiler(
    layer_idx, batch_size, seq_len, tp_degree, ep_degree, moe_fused_nki_kernel_enabled, test_dir
):
    try:
        decoder_layer_hf = generate_ckpt(layer_idx, test_dir)

        if moe_fused_nki_kernel_enabled:
            pytest.xfail("Skipping because alpha compiler is blocked")

        hf_moe = HFMoE(layer_idx, decoder_layer_hf).to(DTYPE)

        neuron_config = MoENeuronConfig(
            moe_tp_degree=tp_degree,
            moe_ep_degree=ep_degree,
            torch_dtype=DTYPE,
            glu_type="swiglu",
            hidden_act_scaling_factor=1.702,
            return_router_logits=True,
            return_expert_index=True,
            moe_fused_nki_kernel_enabled=moe_fused_nki_kernel_enabled,
        )

        config = InferenceConfig(neuron_config)
        config.hidden_size = HIDDEN_SIZE
        config.intermediate_size = INTERMEDIATE_SIZE
        config.num_local_experts = NUM_EXPERTS
        config.num_experts_per_tok = EXPERTS_PER_TOKEN
        config.hidden_act = "sigmoid"
        config.n_shared_experts = 0

        hidden = (torch.randn((batch_size, seq_len, HIDDEN_SIZE)) * 0.05).to(DTYPE)
        example_inputs = [(hidden,)]

        world_size = neuron_config.moe_ep_degree * neuron_config.moe_tp_degree
        neuron_moe = build_module(
            module_cls=NeuronGptOssMoE,
            example_inputs=example_inputs,
            tp_degree=neuron_config.moe_tp_degree,
            world_size=world_size,
            local_ranks_size=world_size,
            module_init_kwargs={
                "config": config,
                "rmsnorm": GptOssRMSNormV2(HIDDEN_SIZE, 1e-05),
            },
            compiler_args=None,
            compiler_workdir=os.path.join(test_dir, "compiler_workdir_moe"),
            checkpoint_path=os.path.join(test_dir, "checkpoint.pt"),
            logical_nc_config=2,
            checkpoint_loader_fn=customize_loader,
        )
        target_model = NxDMoE(layer_idx, neuron_config, test_dir, neuron_moe=neuron_moe)

        check_accuracy(target_model, hf_moe, batch_size, seq_len)

    finally:
        destroy_mp()


if __name__ == "__main__":
    # you'll need to install pytest-forked, and pytest-xdist
    pytest.main([__file__, "-v", "--forked", "--no-cov"])
