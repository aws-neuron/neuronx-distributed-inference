import torch
from neuronx_distributed_inference.models.llama4.modeling_llama4_text  import NeuronLlama4TextForCausalLM, LlamaInferenceConfig as Llama4InferenceConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from transformers import AutoConfig, AutoModel
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
import copy
import pytest
import os
from test.integration.utils.test_utils import save_checkpoint

def setup_model(config_name, attn_block_tkg_nki_kernel_enabled, attn_block_tkg_nki_kernel_cache_update):
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/" + config_name
    temp_dir = save_checkpoint(config_path)
    model_path = temp_dir.name
    traced_model_path = os.path.join(model_path, "compiled_checkpoint")

    neuron_config = MoENeuronConfig(
        batch_size=2,
        ctx_batch_size=1,
        tkg_batch_size=2,
        max_batch_size=2,
        seq_len=10240,
        torch_dtype=torch.float16,
        rpl_reduce_dtype=torch.float32,
        tp_degree=64,
        cp_degree=16,
        sequence_parallel_enabled=True,
        attn_kernel_enabled=True,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        world_size=64,
        on_cpu=False,
        on_device_sampling_config={
            "top_k": 1,
            "dynamic": True,
            "top_k_kernel_enabled": False
        },
        is_continuous_batching=True,
        logical_neuron_cores=2,
        cast_type="as-declared",
        apply_seq_ids_mask=True,
        skip_warmup=True,
        attn_block_tkg_nki_kernel_enabled=attn_block_tkg_nki_kernel_enabled,
        attn_block_tkg_nki_kernel_cache_update=attn_block_tkg_nki_kernel_cache_update,
    )
    config_cls, model_cls = Llama4InferenceConfig, NeuronLlama4TextForCausalLM

    config = config_cls(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    config = config.get_text_config()

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

    assert torch.equal(prefill_kv_cache[1, :, :neuron_model.config.attention_chunk_size, :],
        cache[1, :, :neuron_model.config.attention_chunk_size, :])

    print("All outputs accurate!")

TEST_LIST = [
    ("config_16E_4layer.json", False, False),
    ("config_16E_4layer.json", True, False),
    ("config_16E_4layer.json", True, True),
]

@pytest.mark.tp64_llama4_text
@pytest.mark.key_config_test # enable auto test coverage
@pytest.mark.parametrize("config_path,attn_block_tkg_nki_kernel_enabled,attn_block_tkg_nki_kernel_cache_update", TEST_LIST)
def test_apply_seq_ids_mask(config_path,
    attn_block_tkg_nki_kernel_enabled, attn_block_tkg_nki_kernel_cache_update):

    model = setup_model(config_path,
        attn_block_tkg_nki_kernel_enabled, attn_block_tkg_nki_kernel_cache_update)
    check_apply_seq_ids_mask_test_accuracy(model)

if __name__ == "__main__":
    pytest.main([ __file__, "-v", "-s"])
