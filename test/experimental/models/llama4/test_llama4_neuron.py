import os
import re
import copy
import pytest
import torch
import torch_neuronx

from neuronx_distributed import NxDParallelState, ModelBuilder, shard_checkpoint
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import initialize_fallback_parallel_state
from neuronx_distributed.utils.model_utils import init_on_device

from neuronx_distributed_inference.experimental.models.config import Llama4_Scout
from neuronx_distributed_inference.experimental.models.llama4.examples.llama4_neuron import main
from neuronx_distributed_inference.experimental.models.llama4.model import (
    MOE,
    SharedExperts,
    Experts,
    Router,
    Attention,
    precompute_rope,
    load_llama_checkpoint,
)

torch.manual_seed(0)


def test_main():
    # Validate execution is complete
    assert main() is None


@pytest.mark.parametrize("model_path,tp_degree,batch_size,seq_len", [
    ("/home/ubuntu/model_hf/Llama-4-Scout-17B-16E-Instruct/model_merged.safetensors", 32, 2, 128),
])
def test_moe(
    model_path,
    tp_degree,
    batch_size,
    seq_len,
):
    # Test with float32
    cfg = copy.deepcopy(Llama4_Scout)
    cfg.dtype = torch.float32

    hidden_size = cfg.hidden_size
    hidden = torch.randn((batch_size, seq_len, hidden_size), dtype=cfg.dtype)

    # Save weights for testing
    neuron_moe_state_dict = _load_moe_state_dict(cfg, model_path, tp_degree)
    cpu_moe_state_dict = _load_moe_state_dict(cfg, model_path, 1)

    # Test CPU version
    initialize_fallback_parallel_state()
    moe = MOE(cfg)
    moe.load_state_dict(cpu_moe_state_dict, strict=False)
    cpu_o = moe(hidden)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

    # Test Neuron version
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        moe = MOE(cfg)
        neuron_moe = ModelBuilder(model=moe) \
            .trace(args=(hidden,), tag="moe") \
            .compile()


    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), init_on_device(torch.device("meta")):
        sharded_ckpts = shard_checkpoint(
            checkpoint=neuron_moe_state_dict,
            model=MOE(cfg),
            start_rank=0,
            end_rank=tp_degree-1,
        )

    neuron_moe.set_weights(sharded_ckpts)
    neuron_moe.to_neuron()

    neuron_o = neuron_moe(hidden)
    torch_neuronx.testing.assert_close(cpu_o, neuron_o)

    print("MOE test passed!")


def _load_moe_state_dict(cfg, model_path, tp_degree):
    model_path = os.path.expanduser(model_path)

    state_dict = load_llama_checkpoint(
        cfg=cfg,
        model_path=model_path,
        tp_degree=tp_degree
    )

    # Use weights from first MOE layer
    moe_layer_id = cfg.interleave_moe_layer_step - 1  # First MOE layer
    moe_dict = {
        re.sub(f"layers\.{moe_layer_id}\.feed_forward\.", "", k): v.to(cfg.dtype)
        for (k, v) in state_dict.items()
        if re.search(f"layers\.{moe_layer_id}\.feed_forward\.", k)
    }

    return moe_dict


@pytest.mark.parametrize("model_path,tp_degree,batch_size,seq_len", [
    ("/home/ubuntu/model_hf/Llama-4-Scout-17B-16E-Instruct/model_merged.safetensors", 32, 2, 128),
])
def test_shared_experts(
    model_path,
    tp_degree,
    batch_size,
    seq_len,
):
    torch.manual_seed(0)

    # Test with float32
    cfg = copy.deepcopy(Llama4_Scout)
    cfg.dtype = torch.float32

    hidden_size = cfg.hidden_size
    hidden = torch.randn((batch_size, seq_len, hidden_size), dtype=cfg.dtype)

    # Save weights for testing
    neuron_shared_expert_state_dict = _load_shared_expert_state_dict(cfg, model_path, tp_degree)
    cpu_shared_expert_state_dict = _load_shared_expert_state_dict(cfg, model_path, 1)

    # Test CPU version
    initialize_fallback_parallel_state()
    shared_expert = SharedExperts(cfg)
    shared_expert.load_state_dict(cpu_shared_expert_state_dict, strict=False)
    cpu_o = shared_expert(hidden)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

    # Test Neuron version
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        shared_expert = SharedExperts(cfg)
        neuron_shared_expert = ModelBuilder(model=shared_expert) \
            .trace(args=(hidden,), tag="shared_expert") \
            .compile()


    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), init_on_device(torch.device("meta")):
        sharded_ckpts = shard_checkpoint(
            checkpoint=neuron_shared_expert_state_dict,
            model=SharedExperts(cfg),
            start_rank=0,
            end_rank=tp_degree-1,
        )

    neuron_shared_expert.set_weights(sharded_ckpts)
    neuron_shared_expert.to_neuron()
    
    neuron_o = neuron_shared_expert(hidden)
    torch_neuronx.testing.assert_close(cpu_o, neuron_o)

    print("SharedExperts test passed!")


def _load_shared_expert_state_dict(cfg, model_path, tp_degree):
    model_path = os.path.expanduser(model_path)

    state_dict = load_llama_checkpoint(
        cfg=cfg,
        model_path=model_path,
        tp_degree=tp_degree
    )

    # Use weights from first MOE layer's shared expert
    moe_layer_id = cfg.interleave_moe_layer_step - 1  # First MOE layer
    shared_expert_dict = {
        re.sub(f"layers\.{moe_layer_id}\.feed_forward\.shared_expert\.", "", k): v.to(cfg.dtype)
        for (k, v) in state_dict.items()
        if re.search(f"layers\.{moe_layer_id}\.feed_forward\.shared_expert\.", k)
    }

    return shared_expert_dict


@pytest.mark.parametrize("model_path,tp_degree,batch_size,seq_len", [
    ("/home/ubuntu/model_hf/Llama-4-Scout-17B-16E-Instruct/model_merged.safetensors", 32, 2, 128),
])
def test_experts(
    model_path,
    tp_degree,
    batch_size,
    seq_len,
):
    torch.manual_seed(0)

    # Test with float32
    cfg = copy.deepcopy(Llama4_Scout)
    cfg.dtype = torch.float32

    hidden_size = cfg.hidden_size
    num_experts = cfg.num_local_experts
    # Create input that mimics what would come from router
    tokens_per_expert = batch_size * seq_len
    routed_tokens = torch.randn((num_experts * tokens_per_expert, hidden_size), dtype=cfg.dtype)

    # Save weights for testing
    neuron_experts_state_dict = _load_experts_state_dict(cfg, model_path, tp_degree)
    cpu_experts_state_dict = _load_experts_state_dict(cfg, model_path, 1)

    # Test CPU version
    initialize_fallback_parallel_state()
    experts = Experts(cfg)
    experts.load_state_dict(cpu_experts_state_dict, strict=False)
    cpu_o = experts(routed_tokens)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

    # Test Neuron version
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        experts = Experts(cfg)
        neuron_experts = ModelBuilder(model=experts) \
            .trace(args=(routed_tokens,), tag="experts") \
            .compile()

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), init_on_device(torch.device("meta")):
        sharded_ckpts = shard_checkpoint(
            checkpoint=neuron_experts_state_dict,
            model=Experts(cfg),
            start_rank=0,
            end_rank=tp_degree-1,
        )

    neuron_experts.set_weights(sharded_ckpts)
    neuron_experts.to_neuron()
    
    neuron_o = neuron_experts(routed_tokens)
    torch_neuronx.testing.assert_close(cpu_o, neuron_o)

    print("Experts test passed!")


def _load_experts_state_dict(cfg, model_path, tp_degree):
    model_path = os.path.expanduser(model_path)

    state_dict = load_llama_checkpoint(
        cfg=cfg,
        model_path=model_path,
        tp_degree=tp_degree
    )

    # Use weights from first MOE layer's experts
    moe_layer_id = cfg.interleave_moe_layer_step - 1  # First MOE layer
    experts_dict = {
        re.sub(f"layers\.{moe_layer_id}\.feed_forward\.experts\.", "", k): v.to(cfg.dtype)
        for (k, v) in state_dict.items()
        if re.search(f"layers\.{moe_layer_id}\.feed_forward\.experts\.", k)
    }

    return experts_dict


@pytest.mark.parametrize("model_path,tp_degree,batch_size,seq_len", [
    ("/home/ubuntu/model_hf/Llama-4-Scout-17B-16E-Instruct/model_merged.safetensors", 32, 2, 128),
])
def test_router(
    model_path,
    tp_degree,
    batch_size,
    seq_len,
):
    torch.manual_seed(0)

    # Test with float32
    cfg = copy.deepcopy(Llama4_Scout)
    cfg.dtype = torch.float32

    hidden_size = cfg.hidden_size
    hidden = torch.randn((batch_size * seq_len, hidden_size), dtype=cfg.dtype)

    # Save weights for testing
    neuron_router_state_dict = _load_router_state_dict(cfg, model_path, tp_degree)
    cpu_router_state_dict = _load_router_state_dict(cfg, model_path, 1)

    # Test CPU version
    initialize_fallback_parallel_state()
    router = Router(
        cfg=cfg,
        dtype=cfg.dtype
    )
    router.load_state_dict(cpu_router_state_dict, strict=False)
    cpu_scores, cpu_indices, cpu_probs = router(hidden)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

    # Test Neuron version
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        router = Router(
            cfg=cfg,
            dtype=cfg.dtype
        )
        neuron_router = ModelBuilder(model=router) \
            .trace(args=(hidden,), tag="router") \
            .compile()

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), init_on_device(torch.device("meta")):
        sharded_ckpts = shard_checkpoint(
            checkpoint=neuron_router_state_dict,
            model=Router(
                cfg=cfg,
                dtype=cfg.dtype
            ),
            start_rank=0,
            end_rank=tp_degree - 1,
        )

    neuron_router.set_weights(sharded_ckpts)
    neuron_router.to_neuron()

    neuron_scores, neuron_indices, neuron_probs = neuron_router(hidden)

    # Compare outputs
    torch_neuronx.testing.assert_close(cpu_scores, neuron_scores)
    torch_neuronx.testing.assert_close(cpu_indices, neuron_indices)
    torch_neuronx.testing.assert_close(cpu_probs, neuron_probs)

    print("Router test passed!")


def _load_router_state_dict(cfg, model_path, tp_degree):
    model_path = os.path.expanduser(model_path)

    state_dict = load_llama_checkpoint(
        cfg=cfg,
        model_path=model_path,
        tp_degree=tp_degree
    )

    # Use weights from first MOE layer's router
    moe_layer_id = cfg.interleave_moe_layer_step - 1  # First MOE layer
    router_dict = {
        re.sub(f"layers\.{moe_layer_id}\.feed_forward\.router\.", "", k): v.to(cfg.dtype)
        for (k, v) in state_dict.items()
        if re.search(f"layers\.{moe_layer_id}\.feed_forward\.router\.", k)
    }

    return router_dict


@pytest.mark.parametrize("model_path,tp_degree,batch_size,seq_len", [
    ("/home/ubuntu/model_hf/Llama-4-Scout-17B-16E-Instruct/model_merged.safetensors", 32, 2, 128),
])
def test_attention(
    model_path,
    tp_degree,
    batch_size,
    seq_len,
):
    torch.manual_seed(0)

    # Test with float32
    cfg = copy.deepcopy(Llama4_Scout)
    cfg.dtype = torch.float32

    # Create inputs
    hidden_size = cfg.hidden_size
    head_dim = cfg.head_dim
    hidden = torch.randn((batch_size, seq_len, hidden_size), dtype=cfg.dtype)
    start_pos = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    rope_cache = precompute_rope("cpu", cfg.rope_theta, head_dim, seq_len)

    # Save weights for testing
    neuron_attn_state_dict = _load_attn_state_dict(cfg, model_path, tp_degree)
    cpu_attn_state_dict = _load_attn_state_dict(cfg, model_path, 1)

    # Test CPU version
    initialize_fallback_parallel_state()
    attn = Attention(
        cfg=cfg,
        batch_size=batch_size,
        seq_len=seq_len,
        use_rope=True,
        use_qk_norm=True
    )
    attn.load_state_dict(cpu_attn_state_dict, strict=False)
    cpu_o = attn(hidden, start_pos, rope_cache, mask)
    # Save CPU KV cache for comparison
    cpu_cache_k = attn.cache_k.clone()
    cpu_cache_v = attn.cache_v.clone()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

    # Test Neuron version
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        attn = Attention(
            cfg=cfg,
            batch_size=batch_size,
            seq_len=seq_len,
            use_rope=True,
            use_qk_norm=True
        )
        neuron_attn = ModelBuilder(model=attn) \
            .trace(args=(hidden, start_pos, rope_cache, mask), tag="attention") \
            .compile()


    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), init_on_device(torch.device("meta")):
        sharded_ckpts = shard_checkpoint(
            checkpoint=neuron_attn_state_dict,
            model=Attention(
                cfg=cfg,
                batch_size=batch_size,
                seq_len=seq_len,
                use_rope=True,
                use_qk_norm=True
            ),
            start_rank=0,
            end_rank=tp_degree - 1,
        )

    neuron_attn.set_weights(sharded_ckpts)
    neuron_attn.to_neuron()

    neuron_o = neuron_attn(hidden, start_pos, rope_cache, mask)
    torch_neuronx.testing.assert_close(cpu_o, neuron_o)

    # Compare KV cache if tp_degree=1
    if tp_degree == 1:
        rank = 0
        neuron_cache_k = neuron_attn.states[rank]["cache_k"].to("cpu")
        neuron_cache_v = neuron_attn.states[rank]["cache_v"].to("cpu")
        torch_neuronx.testing.assert_close(cpu_cache_k, neuron_cache_k, rtol=1e-5, atol=1e-5)
        torch_neuronx.testing.assert_close(cpu_cache_v, neuron_cache_v, rtol=1e-5, atol=1e-5)

    print("Attention test passed!")


def _load_attn_state_dict(cfg, model_path, tp_degree):
    model_path = os.path.expanduser(model_path)

    state_dict = load_llama_checkpoint(
        cfg=cfg,
        model_path=model_path,
        tp_degree=tp_degree
    )

    # Use weights from first layer's attention
    attn_dict = {
        re.sub(r"layers\.0\.self_attn\.", "", k): v.to(cfg.dtype)
        for (k, v) in state_dict.items()
        if re.search(r"layers\.0\.self_attn\.", k)
    }

    return attn_dict