import torch
import time
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from transformers import AutoTokenizer, AutoConfig, AutoModel
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
import copy
import pytest
import os
import tempfile
import threading
import collections

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_checkpoint(config_path):

    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=torch.bfloat16)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    logger.info(f"Saving model with random weights to {model_path}")
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
        "save_sharded_checkpoint": True,
        "skip_warmup": True,
        "enable_output_completion_notifications": True,
    }

    kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig()
    neuron_config = NeuronConfig(**kwargs)

    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Compile and save model.
    logger.info("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(traced_model_path)

    # Load from compiled checkpoint.
    model.load(traced_model_path)

    return model

def get_kv_cache_all(model):

    kv_caches = []
    tp_tensors_map = collections.defaultdict(list)
    state = model.context_encoding_model.model.nxd_model.state

    # rearrange tensors with tp
    for tp_idx, per_tp_state in enumerate(state):
        for key, val in per_tp_state.items():
            tp_tensors_map[tp_idx].append(val)

    for i in range(len(tp_tensors_map[0])):
        for tp, tensors in tp_tensors_map.items():
            kv_caches.append(tensors[i])

    logger.info("num of kv_caches %s", len(kv_caches))
    return kv_caches


def wait_completion_handler(kv_caches, batch_size, num_prefill_steps, num_decode_steps, completion_events):
    timeout = -1

    cnt = 0
    try:
        # Wait for prefill completion
        for step in range(num_prefill_steps):
            # given we are running with CB, it will iterate through the input sequences and run BS times
            cnt += batch_size
            logger.info(f"[thread] wait for prefill {step} tensor with cnt {cnt}")
            for i, tensor in enumerate(kv_caches):
                torch.ops.neuron._nrt_wait_output_completion_with_timeout(tensor, timeout, cnt)
            logger.info(f"[thread] wait for prefill {step} is done and set completion event")
            completion_events[f'prefill_{step}'].set()

        # Wait for decode completion
        for step in range(num_decode_steps):
            cnt += 1
            logger.info(f"[thread] wait for decode {step} tensor with cnt {cnt}")
            for i, tensor in enumerate(kv_caches):
                torch.ops.neuron._nrt_wait_output_completion_with_timeout(tensor, timeout, cnt)
            logger.info(f"[thread] wait for decode {step} is done and set completion event")
            completion_events[f'decode_{step}'].set()
    except Exception as e:
        raise e

def check_tensor_completion_test_accuracy(neuron_model):

    # Create events for synchronization
    completion_events = {}
    prefill_steps = 10
    decode_steps = 10
    for i in range(prefill_steps):
        completion_events[f'prefill_{i}'] = threading.Event()

    for i in range(decode_steps):
        completion_events[f'decode_{i}'] = threading.Event()

    # Start the waiting thread
    thread = threading.Thread(
        target=wait_completion_handler,
        args=(get_kv_cache_all(neuron_model), neuron_model.config.neuron_config.batch_size,
            prefill_steps, decode_steps, completion_events)
    )
    logger.info("launching wait completion thread")
    thread.start()

    # prefill stage
    input_len = 5

    seq_len = neuron_model.config.neuron_config.seq_len

    time.sleep(1) # wait for sometime for thread to start

    input_ids = torch.rand((neuron_model.config.neuron_config.batch_size, input_len)) * neuron_model.config.vocab_size
    input_ids = input_ids.to(dtype=torch.int32)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(neuron_model.config.neuron_config.batch_size, -1)
    attention_mask = torch.ones((neuron_model.config.neuron_config.batch_size, input_len), dtype=torch.int32)

    for i in range(prefill_steps):

        assert not completion_events[f'prefill_{i}'].is_set(), f"Prefill completion event {i} was set before prefill forward call"

        logger.info(f"running {i} prefill step...")
        with torch.no_grad():
            neuron_model.forward(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=True)

        if not completion_events[f'prefill_{i}'].wait(timeout=5):
            raise TimeoutError(f"Prefill completion {i} timed out")

    # decode stage

    input_ids = input_ids.to(dtype=torch.int32)
    position_ids_decode = torch.tensor([[len(input_ids)-1]], dtype=torch.long) # start the position_id_decode from length of the prompt previously

    new_input_ids = input_ids[:, -1:]
    new_attention_mask = torch.ones((neuron_model.config.neuron_config.batch_size, 1), dtype=torch.int32)

    for i in range(decode_steps):

        assert not completion_events[f'decode_{i}'].is_set(), f"Decode completion event {i} was set before decode forward call"

        logger.info(f"running {i} decode step...")

        with torch.no_grad():
            neuron_model.forward(input_ids=new_input_ids,
                            attention_mask=new_attention_mask,
                            seq_ids=torch.tensor([0]),
                            position_ids=position_ids_decode,
                            use_cache=True)

        if not completion_events[f'decode_{i}'].wait(timeout=5):
            raise TimeoutError(f"Decode completion {i} timed out")

    try:
        thread.join()
    except Exception as e:
        raise e

    logger.info("Done!")



TEST_LIST = [
    ("llama_3_8b", "models/llama/llama_3_8b_4l_config.json")
]
@pytest.mark.tp32
@pytest.mark.parametrize("model_name,config_path", TEST_LIST)
@pytest.mark.xfail(reason="pending runtime change to be merged")
#pytest.mark
def test_tensor_completion(model_name, config_path):
    # Save model onto machine
    model = setup_model(config_path)
    check_tensor_completion_test_accuracy(model)

if __name__ == "__main__":
    pytest.main([ __file__, "-v", "-s"])
