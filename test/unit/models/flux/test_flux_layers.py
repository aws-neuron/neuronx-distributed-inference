import logging
import math
import os
import time
from types import SimpleNamespace
from unittest import TestCase, main, skip

import torch
import torch.nn as nn
import torch_xla
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_tensor_model_parallel_size,
    get_world_group,
)
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.models.diffusers.embeddings import (
    FluxPosEmbed,
    NeuronCombinedTimestepGuidanceTextProjEmbeddings,
    NeuronCombinedTimestepTextProjEmbeddings,
)
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
    NeuronFluxAttention,
    NeuronFeedForward,
    NeuronFluxSingleTransformerBlock,
    NeuronFluxTransformer2DModel,
    NeuronFluxTransformerBlock,
    split_along_dim,
    FluxBackboneInferenceConfig,
)
from neuronx_distributed_inference.models.diffusers.normalization import (
    NeuronAdaLayerNormContinuous,
)
from neuronx_distributed_inference.models.diffusers.padder import MaybePadder
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.distributed import get_dp_rank_spmd
from neuronx_distributed_inference.models.config import NeuronConfig

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("matplotlib not found. Install via `pip install matplotlib`.")
    matplotlib = None
    plt = None

logger = logging.getLogger("Test")
logger.setLevel(logging.INFO)
torch.manual_seed(0)
CKPT_DIR = "/tmp/"
if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)

dtype = torch.float32
# dtype = torch.bfloat16
_HARDWARE = hardware(get_platform_target())


def get_checkpoint_loader_fn():
    state_dict = torch.load(
        os.path.join(CKPT_DIR, "checkpoint.pt"), map_location="cpu", weights_only=True
    )
    return state_dict


def trace_nxd_model(
    example_inputs,
    model_cls,
    constructor_kwargs,
    tp_degree=2,  # auto tests are running on trn1.2xlarge, which only supports tp_degree=2
    world_size=None,
    checkpoint_loader=get_checkpoint_loader_fn,
):
    # Shouldn't use the same tensor tuple for tracing and inference, it could hide some tracing issue
    # This won't work if we have 0/1 flag tensors in the input, since it may influence the traced computation graph
    trace_inputs = tuple(
        torch.rand(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        for tensor in example_inputs
    )

    if world_size is None:
        world_size = tp_degree

    model_builder = ModelBuilder(
        router=None,
        debug=False,
        tp_degree=tp_degree,
        world_size=world_size,
        local_ranks_size=world_size,
        checkpoint_loader=checkpoint_loader,
    )
    logger.info("Initiated model builder!")

    def create_model():
        model = model_cls(**constructor_kwargs)
        model.eval()
        if dtype == torch.bfloat16:
            model.bfloat16()
        return model

    model_builder.add(
        key="test_flux_layers",
        model_instance=BaseModelInstance(module_cls=create_model, input_output_aliases={}),
        example_inputs=[trace_inputs],
        priority_model_idx=0,
        compiler_args=get_compiler_args(world_size=world_size),
    )
    logger.info("Added model builder! Starting to trace!")
    start_time = time.time()

    traced_model = model_builder.trace(initialize_model_weights=True)
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))

    elapsed_time = time.time() - start_time
    logger.info(f"Traced time taken {elapsed_time} s")

    logger.info("Done tracing the model!")
    return traced_model


def init_cpu_env():
    # destroy distributed process if already started
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    # if need to run distributed framework on CPU
    logger.info("Initializing cpu env")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8080"
    os.environ["RANK"] = "0"
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel()


def get_model_output(model, inputs, device):
    logger.info(f"Model type {type(model)}!")
    logger.info(f"Calling {device} model!")
    output = model(*inputs)
    return output


def run_on_cpu(test_inputs, model_cls, constructor_kwargs, load_existing_checkpoint=False):
    # If the original implementation uses distributed framework,
    # we need to start a distributed process on cpu
    # nn layers does not need this
    init_cpu_env()

    cpu_model = model_cls(**constructor_kwargs)
    cpu_model.eval()
    if dtype == torch.bfloat16:
        cpu_model.bfloat16()

    if load_existing_checkpoint:
        state_dict = get_checkpoint_loader_fn()
        missing_keys, unexpected_keys = cpu_model.load_state_dict(state_dict, strict=False)
        assert len(missing_keys) == 0, f"Missing required keys from the checkpoint: {missing_keys}"
        print(
            f"Ignored {len(unexpected_keys)} parameters from the checkpoint, they are unrelated to this model."
        )
    else:
        # save state dict to be used to trace
        save_ckpt_path = os.path.join(CKPT_DIR, "checkpoint.pt")
        model_state_dict = cpu_model.state_dict()
        torch.save(model_state_dict, save_ckpt_path)
        logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

    # inference and benchmark
    cpu_output = get_model_output(cpu_model, test_inputs, device="cpu")

    # destroy distributed process to reinit for neuron
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    return cpu_output


def run_on_neuron(test_inputs, model_cls, constructor_kwargs):
    neuron_model = trace_nxd_model(test_inputs, model_cls, constructor_kwargs)
    neuron_output = get_model_output(neuron_model, test_inputs, device="neuron")
    return neuron_output


def setup_debug_env():
    # os.environ["NEURON_RT_LOG_LEVEL"] = "DEBUG"
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


def get_compiler_args(world_size=4):
    compiler_args = "--model-type=transformer -O1"
    # Add flags for cc-overlap
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"

    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "1"
    if _HARDWARE == hardware.TRN2:
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        compiler_args += " --target=trn2 --lnc=2"

    print(f"compiler_args: {compiler_args}")
    return compiler_args


class TestLayers(TestCase):
    def setUp(self):
        setup_debug_env()
        # same as the default values of the FluxTransformer2DModel constructor
        self.config = SimpleNamespace(
            patch_size=1,
            in_channels=64,
            out_channels=None,
            num_layers=19,
            num_single_layers=38,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=False,
            axes_dims_rope=(16, 56, 56),
        )
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

    def check_results(self, test_name, actual_output, expected_output):
        print("-" * 20)
        print(f"Test result of {test_name}:")
        self.assertTrue(
            check_accuracy_embeddings(
                actual_output, expected_output, plot_outputs=False, atol=0.000000001
            )
        )
        print("-" * 20)

    def test_CombinedTimestepTextProjEmbeddings(self):
        constructor_kwargs = dict(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        test_inputs = (torch.randn([1]), torch.randn([1, 768]))
        expected_output = run_on_cpu(
            test_inputs=test_inputs,
            model_cls=CombinedTimestepTextProjEmbeddings,
            constructor_kwargs=constructor_kwargs,
        )
        actual_output = run_on_neuron(
            test_inputs=test_inputs,
            model_cls=NeuronCombinedTimestepTextProjEmbeddings,
            constructor_kwargs=constructor_kwargs,
        )
        self.check_results(
            "test_CombinedTimestepTextProjEmbeddings", actual_output, expected_output
        )

    def test_CombinedTimestepGuidanceTextProjEmbeddings(self):
        constructor_kwargs = dict(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        test_inputs = (torch.randn([1]), torch.tensor([3.5]), torch.randn([1, 768]))
        expected_output = run_on_cpu(
            test_inputs=test_inputs,
            model_cls=CombinedTimestepGuidanceTextProjEmbeddings,
            constructor_kwargs=constructor_kwargs,
        )
        actual_output = run_on_neuron(
            test_inputs=test_inputs,
            model_cls=NeuronCombinedTimestepGuidanceTextProjEmbeddings,
            constructor_kwargs=constructor_kwargs,
        )
        self.check_results(
            "test_CombinedTimestepGuidanceTextProjEmbeddings", actual_output, expected_output
        )

    def test_FeedForward(self):
        constructor_kwargs = dict(
            dim=self.inner_dim, dim_out=self.inner_dim, activation_fn="gelu-approximate"
        )
        test_inputs = (torch.randn([1, 4096, 3072]),)
        expected_output = run_on_cpu(
            test_inputs=test_inputs, model_cls=FeedForward, constructor_kwargs=constructor_kwargs
        )
        actual_output = run_on_neuron(
            test_inputs=test_inputs,
            model_cls=NeuronFeedForward,
            constructor_kwargs=constructor_kwargs,
        )
        self.check_results("test_FeedForward", actual_output, expected_output)

    def test_AdaLayerNormContinuous(self):
        constructor_kwargs = dict(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        test_inputs = (
            torch.randn([1, 4096, 3072], dtype=dtype),
            torch.randn([1, 3072], dtype=dtype),
        )
        expected_output = run_on_cpu(
            test_inputs=test_inputs,
            model_cls=AdaLayerNormContinuous,
            constructor_kwargs=constructor_kwargs,
        )
        actual_output = run_on_neuron(
            test_inputs=test_inputs,
            model_cls=NeuronAdaLayerNormContinuous,
            constructor_kwargs=constructor_kwargs,
        )
        self.check_results("test_AdaLayerNormContinuous", actual_output, expected_output)

    @skip("Auto test uses trn1.2xlarge, it can't handle tp16")
    def test_Attention(self):
        dim = self.inner_dim
        num_attention_heads = self.config.num_attention_heads
        attention_head_dim = self.config.attention_head_dim
        neuron_constructor_kwargs = dict(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
        )
        hf_constructor_kwargs = dict(
            **neuron_constructor_kwargs,
            processor=FluxAttnProcessor2_0(),
        )
        neuron_test_inputs = (
            torch.randn([1, 4096, 3072]),
            torch.randn([4096 + 512, 128, 2]),
            torch.zeros([1, 1, 4096 + 512, 4096 + 512]),
            torch.randn([1, 512, 3072]),
        )

        # run on CPU
        cpu_model = Attention(**hf_constructor_kwargs)
        save_ckpt_path = os.path.join(CKPT_DIR, "checkpoint.pt")
        model_state_dict = cpu_model.state_dict()
        torch.save(model_state_dict, save_ckpt_path)
        e1, e2 = torch.unbind(neuron_test_inputs[1], dim=-1)
        expected_output1, expected_output2 = cpu_model(
            hidden_states=neuron_test_inputs[0],
            encoder_hidden_states=neuron_test_inputs[3],
            attention_mask=neuron_test_inputs[2],
            image_rotary_emb=(e1, e2),
        )
        # run on Neuron
        neuron_constructor_kwargs["context_parallel_enabled"] = False
        neuron_constructor_kwargs["reduce_dtype"] = dtype

        # test head padding, 16 is not dividable by the number of heads 24, it will be padded to 32
        tp_degree = 16

        def checkpoint_loader():
            checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            padded_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
            padded_inner_dim = attention_head_dim * padded_heads
            padder = MaybePadder(padded_inner_dim)
            state_dict["to_q.weight"] = padder(state_dict["to_q.weight"], dim=0)
            state_dict["to_q.bias"] = padder(state_dict["to_q.bias"], dim=0)
            state_dict["to_k.weight"] = padder(state_dict["to_k.weight"], dim=0)
            state_dict["to_k.bias"] = padder(state_dict["to_k.bias"], dim=0)
            state_dict["to_v.weight"] = padder(state_dict["to_v.weight"], dim=0)
            state_dict["to_v.bias"] = padder(state_dict["to_v.bias"], dim=0)
            state_dict["add_q_proj.weight"] = padder(state_dict["add_q_proj.weight"], dim=0)
            state_dict["add_q_proj.bias"] = padder(state_dict["add_q_proj.bias"], dim=0)
            state_dict["add_k_proj.weight"] = padder(state_dict["add_k_proj.weight"], dim=0)
            state_dict["add_k_proj.bias"] = padder(state_dict["add_k_proj.bias"], dim=0)
            state_dict["add_v_proj.weight"] = padder(state_dict["add_v_proj.weight"], dim=0)
            state_dict["add_v_proj.bias"] = padder(state_dict["add_v_proj.bias"], dim=0)
            # for RowParallelLinear we don't pad bias
            state_dict["to_out.0.weight"] = padder(state_dict["to_out.0.weight"], dim=1)
            state_dict["to_add_out.weight"] = padder(state_dict["to_add_out.weight"], dim=1)
            return state_dict

        neuron_model = trace_nxd_model(
            neuron_test_inputs,
            NeuronFluxAttention,
            neuron_constructor_kwargs,
            tp_degree=tp_degree,
            checkpoint_loader=checkpoint_loader,
        )
        actual_output1, actual_output2 = get_model_output(
            neuron_model, neuron_test_inputs, device="neuron"
        )

        self.check_results("test_Attention1", actual_output1, expected_output1)
        self.check_results("test_Attention2", actual_output2, expected_output2)

    @skip("Auto test uses trn1.2xlarge, it can't handle tp16")
    def test_context_parallel_Attention(self):
        dim = self.inner_dim
        num_attention_heads = self.config.num_attention_heads
        attention_head_dim = self.config.attention_head_dim
        neuron_constructor_kwargs = dict(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
        )
        hf_constructor_kwargs = dict(
            **neuron_constructor_kwargs,
            processor=FluxAttnProcessor2_0(),
        )

        neuron_test_inputs = (
            torch.randn([1, 4096, 3072], dtype=dtype),
            torch.randn([4096 + 512, 128, 2], dtype=dtype),
            torch.zeros([1, 1, 4096 + 512, 4096 + 512], dtype=dtype),
            torch.randn([1, 512, 3072], dtype=dtype),
        )

        # run on CPU
        cpu_model = Attention(**hf_constructor_kwargs)
        if dtype == torch.bfloat16:
            cpu_model = cpu_model.bfloat16()
        save_ckpt_path = os.path.join(CKPT_DIR, "checkpoint.pt")
        model_state_dict = cpu_model.state_dict()
        torch.save(model_state_dict, save_ckpt_path)
        e1, e2 = torch.unbind(neuron_test_inputs[1], dim=-1)
        expected_output1, expected_output2 = cpu_model(
            hidden_states=neuron_test_inputs[0],
            encoder_hidden_states=neuron_test_inputs[3],
            attention_mask=neuron_test_inputs[2],
            image_rotary_emb=(e1, e2),
        )
        # run on Neuron
        neuron_constructor_kwargs["context_parallel_enabled"] = True
        neuron_constructor_kwargs["reduce_dtype"] = dtype

        # test head padding, 16 is not dividable by the number of heads 24, it will be padded to 32
        tp_degree = 4
        cp_degree = 2
        world_size = tp_degree * cp_degree

        def checkpoint_loader():
            checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            padded_heads = math.ceil(num_attention_heads / tp_degree) * tp_degree
            padded_inner_dim = attention_head_dim * padded_heads
            padder = MaybePadder(padded_inner_dim)
            state_dict["to_q.weight"] = padder(state_dict["to_q.weight"], dim=0)
            state_dict["to_q.bias"] = padder(state_dict["to_q.bias"], dim=0)
            state_dict["to_k.weight"] = padder(state_dict["to_k.weight"], dim=0)
            state_dict["to_k.bias"] = padder(state_dict["to_k.bias"], dim=0)
            state_dict["to_v.weight"] = padder(state_dict["to_v.weight"], dim=0)
            state_dict["to_v.bias"] = padder(state_dict["to_v.bias"], dim=0)
            state_dict["add_q_proj.weight"] = padder(state_dict["add_q_proj.weight"], dim=0)
            state_dict["add_q_proj.bias"] = padder(state_dict["add_q_proj.bias"], dim=0)
            state_dict["add_k_proj.weight"] = padder(state_dict["add_k_proj.weight"], dim=0)
            state_dict["add_k_proj.bias"] = padder(state_dict["add_k_proj.bias"], dim=0)
            state_dict["add_v_proj.weight"] = padder(state_dict["add_v_proj.weight"], dim=0)
            state_dict["add_v_proj.bias"] = padder(state_dict["add_v_proj.bias"], dim=0)
            # for RowParallelLinear we don't pad bias
            state_dict["to_out.0.weight"] = padder(state_dict["to_out.0.weight"], dim=1)
            state_dict["to_add_out.weight"] = padder(state_dict["to_add_out.weight"], dim=1)

            state_dict["scatter_inputs.global_rank.rank"] = torch.arange(
                0, world_size, dtype=torch.int32
            )
            return state_dict

        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        neuron_model = trace_nxd_model(
            neuron_test_inputs,
            NeuronContextParallelAttentionForTest,
            neuron_constructor_kwargs,
            tp_degree=tp_degree,
            world_size=world_size,
            checkpoint_loader=checkpoint_loader,
        )
        actual_output1, actual_output2 = get_model_output(
            neuron_model, neuron_test_inputs, device="neuron"
        )

        self.check_results("test_Attention1", actual_output1, expected_output1)
        self.check_results("test_Attention2", actual_output2, expected_output2)

    def test_FluxSingleTransformerBlock(self):
        constructor_kwargs = dict(
            dim=self.inner_dim,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
        )
        neuron_test_inputs = (
            torch.randn([1, 4608, 3072], dtype=dtype),
            torch.randn([1, 3072], dtype=dtype),
            torch.randn([4608, 128, 2], dtype=dtype),
        )
        e1, e2 = torch.unbind(neuron_test_inputs[-1], dim=-1)
        hf_test_inputs = (neuron_test_inputs[0], neuron_test_inputs[1], (e1, e2))
        expected_output = run_on_cpu(
            test_inputs=hf_test_inputs,
            model_cls=FluxSingleTransformerBlock,
            constructor_kwargs=constructor_kwargs,
        )

        def checkpoint_loader():
            checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # the shape of weights is [out_size, in_size], we want to split the in_size dimension
            # there is no need to split the bias, because is shape is [out_size]
            state_dict["proj_out_attn.weight"] = state_dict["proj_out.weight"][
                :, : self.inner_dim
            ].contiguous()
            state_dict["proj_out_attn.bias"] = state_dict["proj_out.bias"]
            state_dict["proj_out_mlp.weight"] = state_dict["proj_out.weight"][
                :, self.inner_dim :
            ].contiguous()
            del state_dict["proj_out.weight"]
            del state_dict["proj_out.bias"]
            return state_dict

        neuron_model = trace_nxd_model(
            neuron_test_inputs,
            NeuronFluxSingleTransformerBlock,
            constructor_kwargs,
            checkpoint_loader=checkpoint_loader,
        )
        actual_output = get_model_output(neuron_model, neuron_test_inputs, device="neuron")
        self.check_results("test_FluxSingleTransformerBlock", actual_output, expected_output)

    def test_FluxTransformerBlock(self):
        constructor_kwargs = dict(
            dim=self.inner_dim,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
        )
        neuron_test_inputs = (
            torch.randn([1, 4096, 3072], dtype=dtype),
            torch.randn([1, 512, 3072], dtype=dtype),
            torch.randn([1, 3072], dtype=dtype),
            torch.randn([4096 + 512, 128, 2], dtype=dtype),
        )
        e1, e2 = torch.unbind(neuron_test_inputs[-1], dim=-1)
        hf_test_inputs = (
            neuron_test_inputs[0],
            neuron_test_inputs[1],
            neuron_test_inputs[2],
            (e1, e2),
        )
        expected_output1, expected_output2 = run_on_cpu(
            test_inputs=hf_test_inputs,
            model_cls=FluxTransformerBlock,
            constructor_kwargs=constructor_kwargs,
        )
        actual_output1, actual_output2 = run_on_neuron(
            test_inputs=neuron_test_inputs,
            model_cls=NeuronFluxTransformerBlock,
            constructor_kwargs=constructor_kwargs,
        )
        self.check_results("test_FluxTransformerBlock1", actual_output1, expected_output1)
        self.check_results("test_FluxTransformerBlock2", actual_output2, expected_output2)

    # TODO: move this test to the integration test folder after it works
    def test_E2EWithTinyModel(self):
        hf_config = dict(
            patch_size=1,
            in_channels=64,
            out_channels=None,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            axes_dims_rope=(16, 56, 56),
        )
        backbone_neuron_config = NeuronConfig(
            tp_degree = 2,
            world_size = 2,
            torch_type = dtype
        )
        backbone_config = FluxBackboneInferenceConfig(
            neuron_config = backbone_neuron_config,
            **hf_config,
            height=1024,
            width=1024,
        )
        constructor_kwargs = dict(config=backbone_config)

        # Pre-calculate these rotary embeddings to
        # 1. avoid NaN issues 2. latency gain
        text_ids = torch.rand([4096, 3])
        latent_image_ids = torch.rand([512, 3])
        ids = torch.cat((text_ids, latent_image_ids), dim=0)
        pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))
        image_rotary_emb = torch.stack(pos_embed(ids), dim=2)

        hidden_states = torch.randn([1, 4096, 64])
        encoder_hidden_states = torch.randn([1, 512, 4096])
        pooled_projections = torch.randn([1, 768])
        timestep = torch.randn([1])
        guidance = torch.tensor([3.5], dtype=dtype)

        test_cpu_inputs = (
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            text_ids,
            latent_image_ids,
            guidance,
        )
        test_neuron_inputs = (
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            guidance,
            image_rotary_emb,
        )
        expected_output = run_on_cpu(
            test_inputs=test_cpu_inputs,
            model_cls=FluxTransformer2DModel,
            constructor_kwargs=hf_config,
        )[0]

        def checkpoint_loader():
            checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            for i in range(hf_config["num_single_layers"]):
                # the shape of weights is [out_size, in_size], we want to split the in_size dimension
                # there is no need to split the bias, because is shape is [out_size]
                state_dict[f"single_transformer_blocks.{i}.proj_out_attn.weight"] = state_dict[
                    f"single_transformer_blocks.{i}.proj_out.weight"
                ][:, : self.inner_dim].contiguous()
                state_dict[f"single_transformer_blocks.{i}.proj_out_attn.bias"] = state_dict[
                    f"single_transformer_blocks.{i}.proj_out.bias"
                ]
                state_dict[f"single_transformer_blocks.{i}.proj_out_mlp.weight"] = state_dict[
                    f"single_transformer_blocks.{i}.proj_out.weight"
                ][:, self.inner_dim :].contiguous()
                del state_dict[f"single_transformer_blocks.{i}.proj_out.weight"]
                del state_dict[f"single_transformer_blocks.{i}.proj_out.bias"]
            return state_dict

        neuron_model = trace_nxd_model(
            test_neuron_inputs,
            NeuronFluxTransformer2DModel,
            constructor_kwargs,
            checkpoint_loader=checkpoint_loader,
        )
        actual_output = get_model_output(neuron_model, test_neuron_inputs, device="neuron")
        self.check_results("test_E2EWithTinyModel", actual_output, expected_output)


class ScatterContextParallelInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_parallel_group = get_data_parallel_group()
        self.global_rank = SPMDRank(world_size=get_world_group().size())

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb, attention_mask):

        dp_rank = get_dp_rank_spmd(
            global_rank=self.global_rank.get_rank(),
            tp_degree=get_tensor_model_parallel_size(),
        )

        rotary_emb_text = image_rotary_emb[: encoder_hidden_states.shape[1]]
        rotary_emb_image = image_rotary_emb[encoder_hidden_states.shape[1] :]

        hidden_states = split_along_dim(
            hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group
        )
        encoder_hidden_states = split_along_dim(
            encoder_hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group
        )
        rotary_emb_text = split_along_dim(
            rotary_emb_text, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group
        )
        rotary_emb_image = split_along_dim(
            rotary_emb_image, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group
        )
        attention_mask = None

        return (
            hidden_states,
            encoder_hidden_states,
            rotary_emb_text,
            rotary_emb_image,
            attention_mask,
        )


class NeuronContextParallelAttentionForTest(NeuronFluxAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scatter_inputs = ScatterContextParallelInputs()

    def forward(
        self,
        hidden_states,
        image_rotary_emb,
        attention_mask=None,
        encoder_hidden_states=None,
        rotary_emb_text=None,
        rotary_emb_image=None,
    ):
        hidden_states, encoder_hidden_states, rotary_emb_text, rotary_emb_image, attention_mask = (
            self.scatter_inputs(
                hidden_states,
                encoder_hidden_states,
                image_rotary_emb,
                attention_mask,
            )
        )

        output = super().forward(
            hidden_states,
            image_rotary_emb,
            attention_mask,
            encoder_hidden_states,
            rotary_emb_text,
            rotary_emb_image,
        )

        if isinstance(output, tuple):
            hidden_states = gather_from_tensor_model_parallel_region_with_dim(
                output[0], gather_dim=1, process_group=self.data_parallel_group
            )
            encoder_hidden_states = gather_from_tensor_model_parallel_region_with_dim(
                output[1], gather_dim=1, process_group=self.data_parallel_group
            )
            return hidden_states, encoder_hidden_states

        output = gather_from_tensor_model_parallel_region_with_dim(
            output, gather_dim=1, process_group=self.data_parallel_group
        )
        return output


if __name__ == "__main__":
    main()
