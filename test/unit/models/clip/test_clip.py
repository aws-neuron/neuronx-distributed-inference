import logging
import os
import sys
import time
import unittest
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch_xla
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch_neuronx.testing.validation import custom_allclose
from transformers import CLIPTextConfig, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPTextEmbeddings,
    CLIPTextTransformer,
    CLIPVisionConfig,
)

from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import (
    NeuronCLIPAttention,
    NeuronCLIPEncoder,
    NeuronCLIPEncoderLayer,
    NeuronCLIPMLP,
    NeuronCLIPTextEmbeddings,
    NeuronCLIPTextTransformer,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("matplotlib not found. Install via `pip install matplotlib`.")
    matplotlib = None
    plt = None


logger = logging.getLogger(__name__)
torch.manual_seed(0)
CKPT_DIR = "/tmp/"
if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)


# Get the customer clip config
def get_custom_clip_text_config():
    text_config = CLIPTextConfig(
        **{
            "vocab_size": 49408,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "max_position_embeddings": 77,
        }
    )
    return text_config


def get_custom_clip_vision_config():
    vision_config = CLIPVisionConfig(
        **{
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "image_size": 224,
            "patch_size": 14,
        }
    )
    return vision_config


def check_accuracy_embeddings(
    actual_output: torch.Tensor,
    expected_output: torch.Tensor,
    plot_outputs: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
):
    assert (
        expected_output.dtype == actual_output.dtype
    ), f"dtypes {expected_output.dtype} and {actual_output.dtype} does not match!"
    dtype = expected_output.dtype

    # Set default rtol, atol based on dtype if not provided
    if not rtol:
        if dtype == torch.bfloat16:
            rtol = 0.05
        elif dtype == torch.float32:
            rtol = 0.01
        else:
            NotImplementedError(f"Specify rtol for dtype {dtype}")
    print(f"Using rtol = {rtol} for dtype {dtype}")
    print(f"Using atol = {atol}")

    if plot_outputs and matplotlib and plt:
        # Save plot, expecting a y=x straight line
        matplotlib.rcParams["agg.path.chunksize"] = 10000
        matplotlib.rcParams["path.simplify_threshold"] = 1.0
        plt.plot(
            actual_output.float().detach().numpy().reshape(-1),
            expected_output.float().detach().numpy().reshape(-1),
        )
        plt.xlabel("Actual Output")
        plt.ylabel("Expected Output")
        plot_path = "plot.png"
        plt.savefig(plot_path, format="png")
        print(f"Saved outputs plot to {plot_path}.")

    # NxD logit validation tests uses this method
    # equivalent to torch.allclose except rtol is multiplied by absolute max, not abs
    # this matches the behavior of the compiler's birsim-to-xla_infergoldens verification
    passed, max_err = custom_allclose(expected_output, actual_output, atol=atol, rtol=rtol)
    print(f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}")
    return passed


def get_checkpoint_loader_fn():
    state_dict = torch.load(os.path.join(CKPT_DIR, "checkpoint.pt"), map_location="cpu")
    return state_dict


def trace_nxd_model(example_inputs, model_cls, constructor_kwargs):
    logger.info("Starting to trace the model!")
    model_builder = ModelBuilder(
        router=None,
        debug=False,
        tp_degree=2,
        checkpoint_loader=get_checkpoint_loader_fn,
    )
    logger.info("Initiated model builder!")
    print(constructor_kwargs)

    model_builder.add(
        key="test_parallel_full_text_transformer",
        model_instance=BaseModelInstance(
            module_cls=partial(model_cls, **constructor_kwargs), input_output_aliases={}
        ),
        example_inputs=[example_inputs],
        priority_model_idx=0,
        compiler_args=get_compiler_args(),
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
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    logger.info("Initializing cpu env")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8080"
    os.environ["RANK"] = "0"
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel()


def initialize_distributed():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def init_distributed():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return world_size, rank


def get_model_output(model, inputs, device):
    logger.info(f"Model type {type(model)}!")
    logger.info(f"Calling {device} model!")
    if isinstance(inputs, dict):
        with torch.no_grad():
            output = model(**inputs)
    else:
        output = model(*inputs)
    return output


def run_on_cpu(test_inputs, model_cls, constructor_kwargs):
    init_cpu_env()

    cpu_model = model_cls(**constructor_kwargs)

    save_ckpt_path = os.path.join(CKPT_DIR, "checkpoint.pt")
    model_state_dict = cpu_model.state_dict()
    torch.save(model_state_dict, save_ckpt_path)
    logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

    cpu_output = get_model_output(cpu_model, test_inputs, device="cpu")

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    return cpu_output


def run_on_neuron(test_inputs, model_cls, constructor_kwargs):
    neuron_model = trace_nxd_model(test_inputs, model_cls, constructor_kwargs)
    neuron_output = get_model_output(neuron_model, test_inputs, device="neuron")
    return neuron_output


def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


def get_compiler_args():
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1 "
    compiler_args += " --model-type=transformer"
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
    print(f"compiler_args: {compiler_args}")
    return compiler_args


# Write a custom CLIP tokenizer class
class CustomCLIPTokenizer(PreTrainedTokenizer):
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.vocab.update({f"TOKEN_{i}": i + 4 for i in range(49404)})  # CLIP vocab size is 49408
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__()

    def _tokenize(self, text: str) -> List[str]:
        # This is a very basic tokenization. In practice, you'd want something more sophisticated.
        return text.lower().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, "[UNK]")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._convert_id_to_token(id) for id in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        return self.convert_tokens_to_ids(tokens)

    def encode_plus(
        self, text: str, max_length: Optional[int] = None, padding: bool = False
    ) -> Dict[str, List[int]]:
        input_ids = self.encode(text)
        if max_length:
            input_ids = input_ids[:max_length]
            if padding:
                input_ids = input_ids + [self.vocab["[PAD]"]] * (max_length - len(input_ids))
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __call__(
        self, texts: List[str], padding: bool = True, return_tensors: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        batch = [self.encode_plus(text) for text in texts]
        max_length = max(len(item["input_ids"]) for item in batch)

        padded_batch = {
            "input_ids": [
                item["input_ids"] + [self.vocab["[PAD]"]] * (max_length - len(item["input_ids"]))
                for item in batch
            ],
            "attention_mask": [
                item["attention_mask"] + [0] * (max_length - len(item["attention_mask"]))
                for item in batch
            ],
        }

        if return_tensors == "pt":
            return {k: torch.tensor(v) for k, v in padded_batch.items()}
        return padded_batch


class TestCLIPComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            initialize_distributed()

        setup_debug_env()
        # cls.config = get_custom_clip_config()
        cls.text_config = get_custom_clip_text_config()
        cls.vision_config = get_custom_clip_vision_config()

        # Initialize tokenizer (you might want to replace this with a custom tokenizer)
        # Here this tokenizer is downloading from outside resources.
        cls.tokenizer = CustomCLIPTokenizer()
        print(cls.tokenizer)

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)

    @classmethod
    def tearDownClass(cls):
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_parallel_clip_attention(self):
        constructor_kwargs = {"config": self.text_config}

        batch_size, seq_len = 2, 77
        hidden_states = torch.randn(batch_size, seq_len, self.text_config.hidden_size)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        causal_attention_mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len))

        test_inputs = (hidden_states, attention_mask, causal_attention_mask)
        expected_output = run_on_cpu(test_inputs, CLIPAttention, constructor_kwargs)
        actual_output = run_on_neuron(test_inputs, NeuronCLIPAttention, constructor_kwargs)

        self.assertTrue(
            check_accuracy_embeddings(
                actual_output[0], expected_output[0], plot_outputs=True, rtol=0.005, atol=0
            )
        )

    def test_parallel_clip_mlp(self):
        constructor_kwargs = {"config": self.text_config}

        batch_size, seq_len = 2, 77
        hidden_states = torch.randn(batch_size, seq_len, self.text_config.hidden_size)

        test_inputs = (hidden_states,)
        expected_output = run_on_cpu(test_inputs, CLIPMLP, constructor_kwargs)
        actual_output = run_on_neuron(test_inputs, NeuronCLIPMLP, constructor_kwargs)

        self.assertTrue(
            check_accuracy_embeddings(
                actual_output, expected_output, plot_outputs=False, rtol=0.005, atol=0
            )
        )

    def test_parallel_clip_encoder_layer(self):
        constructor_kwargs = {"config": self.text_config}

        batch_size, seq_len = 2, 77
        hidden_states = torch.randn(batch_size, seq_len, self.text_config.hidden_size)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        causal_attention_mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len))

        test_inputs = (hidden_states, attention_mask, causal_attention_mask)
        expected_output = run_on_cpu(test_inputs, CLIPEncoderLayer, constructor_kwargs)
        actual_output = run_on_neuron(test_inputs, NeuronCLIPEncoderLayer, constructor_kwargs)

        self.assertTrue(
            check_accuracy_embeddings(
                actual_output[0], expected_output[0], plot_outputs=False, rtol=0.005, atol=0
            )
        )

    def test_parallel_clip_encoder(self):
        constructor_kwargs = {"config": self.text_config}

        batch_size, seq_len = 2, 77
        inputs_embeds = torch.randn(batch_size, seq_len, self.text_config.hidden_size)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        causal_attention_mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len))

        test_inputs = (inputs_embeds, attention_mask, causal_attention_mask)
        expected_output = run_on_cpu(test_inputs, CLIPEncoder, constructor_kwargs)
        actual_output = run_on_neuron(test_inputs, NeuronCLIPEncoder, constructor_kwargs)

        actual_last_hidden_state = (
            actual_output.last_hidden_state
            if isinstance(actual_output, BaseModelOutput)
            else actual_output["last_hidden_state"]
        )
        self.assertTrue(
            check_accuracy_embeddings(
                actual_last_hidden_state,
                expected_output.last_hidden_state,
                plot_outputs=False,
                rtol=0.005,
                atol=0,
            )
        )

    def test_neuron_clip_text_embeddings(self):
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.text_config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        test_inputs = (input_ids, position_ids)
        constructor_kwargs = {"config": self.text_config}

        expected_output = run_on_cpu(test_inputs, CLIPTextEmbeddings, constructor_kwargs)
        actual_output = run_on_neuron(test_inputs, NeuronCLIPTextEmbeddings, constructor_kwargs)

        self.assertTrue(
            check_accuracy_embeddings(
                actual_output, expected_output, plot_outputs=False, rtol=0.005, atol=0
            )
        )

    def test_neuron_clip_text_transformer(self):
        text = ["A photo of a cat", "An image of a dog"]
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")

        constructor_kwargs = {"config": self.text_config}
        model_inputs = (inputs["input_ids"], inputs["attention_mask"])

        expected_outputs = run_on_cpu(model_inputs, CLIPTextTransformer, constructor_kwargs)
        actual_outputs = run_on_neuron(model_inputs, NeuronCLIPTextTransformer, constructor_kwargs)

        self.assertTrue(
            check_accuracy_embeddings(
                actual_outputs["last_hidden_state"],
                expected_outputs.last_hidden_state,
                plot_outputs=False,
                rtol=0.005,
                atol=0,
            )
        )
        self.assertTrue(
            check_accuracy_embeddings(
                actual_outputs["pooler_output"],
                expected_outputs.pooler_output,
                plot_outputs=False,
                rtol=0.005,
                atol=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
