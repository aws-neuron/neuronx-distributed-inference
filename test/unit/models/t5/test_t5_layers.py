import logging
import os
import time
import unittest
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch_xla
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch_neuronx.testing.validation import custom_allclose
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Config,
    T5DenseGatedActDense,
    T5EncoderModel,
    T5LayerFF,
)

from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import (
    NeuronT5Attention,
    NeuronT5DenseGatedActDense,
    NeuronT5EncoderModel,
    NeuronT5LayerFF,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("matplotlib not found. Install via `pip install matplotlib`.")
    matplotlib = None
    plt = None

ARTIFACTS_FOLDER = "/tmp/bfl_flux_t5/artifacts/"
CKPT_DIR = ARTIFACTS_FOLDER
GOLDEN_FOLDER = "/tmp/bfl_flux_t5/golden/"
DTYPE = torch.float32
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
os.makedirs(GOLDEN_FOLDER, exist_ok=True)

torch.manual_seed(0)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseInferenceModelInstance(BaseModelInstance):
    def load_module(self):
        self.module = self.module_cls().eval()


def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


def get_compiler_args():
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1"
    # Flag for model type
    compiler_args += " --model-type=transformer"
    # Add flags for cc-overlap
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
    print(f"compiler_args: {compiler_args}")
    return compiler_args


def get_checkpoint_loader_fn():
    os.makedirs(CKPT_DIR, exist_ok=True)
    state_dict = torch.load(os.path.join(CKPT_DIR, "checkpoint.pt"), map_location="cpu")
    return state_dict


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

    cpu_model = model_cls(**constructor_kwargs).eval()

    if load_existing_checkpoint:
        state_dict = get_checkpoint_loader_fn()
        missing_keys, unexpected_keys = cpu_model.load_state_dict(state_dict, strict=False)
        assert len(missing_keys) == 0, f"Missing required keys from the checkpoint: {missing_keys}"
        print(
            f"Ignored {len(unexpected_keys)} parameters from the checkpoint, they are unrelated to this model."
        )
    else:
        # save state dict to be used to trace
        os.makedirs(CKPT_DIR, exist_ok=True)
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


def run_on_neuron(test_inputs, model_cls, constructor_kwargs, config):
    neuron_model = trace_nxd_model(test_inputs, model_cls, constructor_kwargs, config).eval()
    neuron_output = get_model_output(neuron_model, test_inputs, device="neuron")
    return neuron_output


def trace_nxd_model(example_inputs, model_cls, constructor_kwargs, config):
    model_builder = ModelBuilder(
        router=None,
        debug=config["debug"],
        tp_degree=config["tp_degree"],
        checkpoint_loader=get_checkpoint_loader_fn,
    )
    logger.info("Initiated model builder!")

    model_builder.add(
        key="test_flux_layers",
        model_instance=BaseInferenceModelInstance(
            module_cls=partial(model_cls, **constructor_kwargs),
            input_output_aliases={},
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


class T5EncoderModelWrapper(nn.Module):
    def __init__(self, model_cls, **constructor_kwargs):
        super().__init__()
        self.model = model_cls(**constructor_kwargs)

    def forward(self, inputs_embeds):
        return self.model(inputs_embeds=inputs_embeds).last_hidden_state


class TestT5EncoderModel(unittest.TestCase):
    def setUp(self):
        setup_debug_env()
        self.model_config = T5Config(
            _name_or_path="google/t5-v1_1-xxl",
            architectures=["T5EncoderModel"],
            classifier_dropout=0.0,
            d_ff=10240,
            d_kv=64,
            d_model=4096,
            decoder_start_token_id=0,
            dense_act_fn="gelu_new",
            dropout_rate=0.1,
            eos_token_id=1,
            feed_forward_proj="gated-gelu",
            initializer_factor=1.0,
            is_decoder=False,
            is_encoder_decoder=False,
            is_gated_act=True,
            layer_norm_epsilon=1e-06,
            model_type="t5",
            num_decoder_layers=24,
            num_heads=64,
            num_layers=24,
            output_past=True,
            pad_token_id=0,
            relative_attention_max_distance=128,
            relative_attention_num_buckets=32,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            transformers_version="4.43.3",
            use_cache=False,
            vocab_size=32128,
        )
        logger.debug(f"model_config={self.model_config}")

        self.neuron_config = {"debug": False, "tp_degree": 2}
        logger.debug(f"neuron_config={self.neuron_config}")

    def check_results(self, test_name, actual_output, expected_output):
        print("-" * 20)
        print(f"Test result of {test_name}:")
        self.assertTrue(
            check_accuracy_embeddings(
                actual_output, expected_output, plot_outputs=False, rtol=0.005, atol=0
            )
        )
        print("-" * 20)

    def test_gated_dense(self):
        logger.info("Running T5DenseGatedActDense ...")
        test_inputs = (torch.randn((1, 4, 4096), dtype=DTYPE),)
        constructor_kwargs = dict(config=self.model_config)
        cpu_output = run_on_cpu(test_inputs, T5DenseGatedActDense, constructor_kwargs)
        neuron_output = run_on_neuron(
            test_inputs,
            NeuronT5DenseGatedActDense,
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_gated_dense", cpu_output, neuron_output)

    def test_ff(self):
        logger.info("Running T5LayerFF test ...")
        test_inputs = (torch.randn((1, 4, 4096), dtype=DTYPE),)
        constructor_kwargs = dict(config=self.model_config)
        cpu_output = run_on_cpu(test_inputs, T5LayerFF, constructor_kwargs)
        neuron_output = run_on_neuron(
            test_inputs,
            NeuronT5LayerFF,
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_ff", cpu_output, neuron_output)

    def test_attention(self):
        class T5AttentionWrapper(nn.Module):
            def __init__(self, model_cls, **constructor_kwargs):
                super().__init__()
                self.model = model_cls(**constructor_kwargs)

            def forward(self, hidden_states, mask):
                return self.model(
                    hidden_states,
                    mask=mask,
                    key_value_states=None,
                    position_bias=None,
                    past_key_value=None,
                    layer_head_mask=None,
                    query_length=hidden_states.shape[1],
                    use_cache=False,
                    output_attentions=False,
                )[0]

        logger.info("Running T5Attention test ...")
        test_inputs = (
            torch.randn((1, 4, 4096), dtype=DTYPE),  # hidden_states
            torch.zeros((1, 1, 1, 4), dtype=DTYPE),  # mask
        )

        # T5Block 0: Compute `position_bias` with a Embedding layer called `relative_attention_bias`.
        constructor_kwargs = dict(
            config=self.model_config,
            has_relative_attention_bias=True,
        )
        cpu_output = run_on_cpu(
            test_inputs,
            partial(T5AttentionWrapper, T5Attention),
            constructor_kwargs,
        )
        neuron_output = run_on_neuron(
            test_inputs,
            partial(T5AttentionWrapper, NeuronT5Attention),
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_attention_with_bias", cpu_output, neuron_output)

        # T5Block 1+: Directly use the `position_bias` computed from T5Block 0.
        constructor_kwargs = dict(
            config=self.model_config,
            has_relative_attention_bias=False,
        )
        cpu_output = run_on_cpu(
            test_inputs,
            partial(T5AttentionWrapper, T5Attention),
            constructor_kwargs,
        )
        neuron_output = run_on_neuron(
            test_inputs,
            partial(T5AttentionWrapper, NeuronT5Attention),
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_attention_without_bias", cpu_output, neuron_output)

    def test_1_layer(self):
        logger.info("Running T5EncoderModel 1-layer test ...")
        test_inputs = (torch.randn((1, 4, 4096), dtype=DTYPE),)
        model_config = deepcopy(self.model_config)
        model_config.num_layers = 1
        logger.debug(f"Model config: {model_config}")

        constructor_kwargs = dict(config=model_config)
        cpu_output = run_on_cpu(
            test_inputs,
            partial(T5EncoderModelWrapper, T5EncoderModel),
            constructor_kwargs,
        )
        neuron_output = run_on_neuron(
            test_inputs,
            partial(T5EncoderModelWrapper, NeuronT5EncoderModel),
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_1_layer", cpu_output, neuron_output)

    def test_2_layers(self):
        logger.info("Running T5EncoderModel 2-layer test ...")
        test_inputs = (torch.randn((1, 4, 4096), dtype=DTYPE),)
        model_config = deepcopy(self.model_config)
        model_config.num_layers = 2
        logger.debug(f"Model config: {model_config}")

        constructor_kwargs = dict(config=model_config)
        cpu_output = run_on_cpu(
            test_inputs,
            partial(T5EncoderModelWrapper, T5EncoderModel),
            constructor_kwargs,
        )
        neuron_output = run_on_neuron(
            test_inputs,
            partial(T5EncoderModelWrapper, NeuronT5EncoderModel),
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_2_layers", cpu_output, neuron_output)

    def _test_e2e(self):
        test_inputs = (torch.randn((1, 4, 4096), dtype=DTYPE),)
        logger.debug(f"Model config: {self.model_config}")
        constructor_kwargs = dict(config=self.model_config)
        logger.info("Running T5EncoderModel e2e test w/ dummy weights ...")
        cpu_output = run_on_cpu(
            test_inputs,
            partial(T5EncoderModelWrapper, T5EncoderModel),
            constructor_kwargs,
        )
        neuron_output = run_on_neuron(
            test_inputs,
            partial(T5EncoderModelWrapper, NeuronT5EncoderModel),
            constructor_kwargs,
            self.neuron_config,
        )
        self.check_results("test_e2e", cpu_output, neuron_output)


if __name__ == "__main__":
    unittest.main()
