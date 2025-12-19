import copy
import pytest
import os
import torch

from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig
from transformers import AutoConfig, AutoModel
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from test.integration.utils.test_utils import save_checkpoint


def setup_model(config_name, attn_block_tkg_nki_kernel_enabled, attn_block_tkg_nki_kernel_cache_update):
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/" + config_name
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    traced_model_path = os.path.join(model_path, "compiled_checkpoint")

    neuron_config = NeuronConfig(
        torch_dtype="bfloat16",
        batch_size=2,
        max_context_length=2048,  # increase length so we have coverage on chunked attention
        max_length=2048,
        attn_kernel_enabled=False,
        seq_len=2048,
        fused_qkv=True,
        sequence_parallel_enabled=True,
        local_ranks_size=32,
        tp_degree=32,
        start_rank_id=0,
        pad_token_id=0,
        ctx_batch_size=1,
        tkg_batch_size=2,
        max_batch_size=2,
        is_continuous_batching=True,
        apply_seq_ids_mask=True,
        qkv_kernel_enabled=attn_block_tkg_nki_kernel_enabled,
        attn_block_tkg_nki_kernel_enabled=attn_block_tkg_nki_kernel_enabled,
        attn_block_tkg_nki_kernel_cache_update=attn_block_tkg_nki_kernel_cache_update,
        save_sharded_checkpoint=True,
        disable_kv_cache_tiling=True,
        skip_warmup=True,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
    )
    config_cls, model_cls = LlamaInferenceConfig, NeuronLlamaForCausalLM

    config = config_cls(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = model_cls(model_path, config)
    model.compile(traced_model_path)

    # Load from compiled checkpoint.
    model.load(traced_model_path)

    return model

def get_kv_cache(model):
    state =  model.context_encoding_model.model.nxd_model.state
    for _, per_tp_state in enumerate(state):
        for _, val in per_tp_state.items():
            return val.to("cpu")

def check_apply_seq_ids_mask_test_accuracy(neuron_model):
    # prefill stage
    kv_cache = copy.deepcopy(get_kv_cache(neuron_model))
    input_len = 1024
    input_ids = torch.rand((neuron_model.config.neuron_config.batch_size, input_len)) * 100
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((neuron_model.config.neuron_config.batch_size, input_len), dtype=torch.int32)

    seq_len = neuron_model.config.neuron_config.seq_len
    batch_size = neuron_model.config.neuron_config.batch_size
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    print("Position IDs shape: ", position_ids.shape)

    with torch.no_grad():
        neuron_model.forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=kv_cache,
                        use_cache=True)
    # fist_layer_k_cache_tp0
    prefill_kv_cache = copy.deepcopy(get_kv_cache(neuron_model))

    # decode stage
    position_ids_decode = torch.tensor([[len(input_ids)-1]], dtype=torch.long) # start the position_id_decode from length of the prompt previously

    new_input_ids = input_ids[:, -1:]
    new_attention_mask = torch.ones((neuron_model.config.neuron_config.batch_size, 1), dtype=torch.int32)

    with torch.no_grad():
        neuron_model.forward(input_ids=new_input_ids,
                        attention_mask=new_attention_mask,
                        seq_ids=torch.tensor([0]),
                        past_key_values=prefill_kv_cache,
                        position_ids=position_ids_decode,
                        use_cache=True)

    # fist_layer_k_cache_tp0
    cache = copy.deepcopy(get_kv_cache(neuron_model))
    # no-active seq suppose to only write to non-valid position
    assert torch.equal(prefill_kv_cache[1, :, :neuron_model.config.neuron_config.max_length, :],
        cache[1, :, :neuron_model.config.neuron_config.max_length, :])

    print("All outputs accurate!")

TEST_LIST = [
    ("models/llama/llama3.1/8b/config.json", False, False),
    ("models/llama/llama3.1/8b/config.json", True, False),
    ("models/llama/llama3.1/8b/config.json", True, True),
]

@pytest.mark.tp32
@pytest.mark.parametrize("config_path,attn_block_tkg_nki_kernel_enabled,attn_block_tkg_nki_kernel_cache_update", TEST_LIST)
def test_apply_seq_ids_mask(config_path,
    attn_block_tkg_nki_kernel_enabled, attn_block_tkg_nki_kernel_cache_update):

    if attn_block_tkg_nki_kernel_enabled:
        pytest.xfail("Test fail on compiler error, need to create ticket")

    model = setup_model(config_path,
        attn_block_tkg_nki_kernel_enabled, attn_block_tkg_nki_kernel_cache_update)
    check_apply_seq_ids_mask_test_accuracy(model)

if __name__ == "__main__":
    pytest.main([ __file__, "-v", "-s"])
