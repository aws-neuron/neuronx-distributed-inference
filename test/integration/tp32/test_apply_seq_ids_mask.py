import torch
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from transformers import AutoTokenizer, AutoConfig, AutoModel
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
import copy
import pytest
import os
import tempfile


def save_checkpoint(config_path):
    
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)
 
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

def setup_model(config_name):
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/" + config_name
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    traced_model_path = model_path + "/compiled_checkpoint"

    kwargs = {
        "torch_dtype": "bfloat16",
        "batch_size": 2,
        "max_context_length": 128,
        "seq_len": 128,
        "local_ranks_size": 32,
        "tp_degree": 32,
        "start_rank_id": 0,
        "pad_token_id": 0,
        "ctx_batch_size": 1,
        "tkg_batch_size": 2,
        "max_batch_size": 2,
        "is_continuous_batching": True,
        "apply_seq_ids_mask": True,
        "save_sharded_checkpoint": True,
        "disable_kv_cache_tiling": True,
    }

    kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig()
    neuron_config = NeuronConfig(**kwargs)

    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
        
    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path, config)
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
    input_len = 5
    input_ids = torch.rand((neuron_model.config.neuron_config.batch_size, input_len)) * neuron_model.config.vocab_size
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

    final_kv_cache = copy.deepcopy(get_kv_cache(neuron_model))

    # Check that it did not write to the 1st position
    assert torch.equal(prefill_kv_cache[1, :, :-1, :], final_kv_cache[1, :, :-1, :])

    print("All outputs accurate!")

TEST_LIST = [ 
    ("llama_3_8b", "models/llama/llama_3_8b_4l_config.json")
]
@pytest.mark.tp32
@pytest.mark.parametrize("model_name,config_path", TEST_LIST)
def test_apply_seq_ids_mask(model_name, config_path):
    # Save model onto machine
    model = setup_model(config_path)
    check_apply_seq_ids_mask_test_accuracy(model)

if __name__ == "__main__":
    pytest.main([ __file__, "-v", "-s"])