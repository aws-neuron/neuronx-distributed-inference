import logging
import os
import unittest
from typing import List, Tuple

import torch
import torch_xla
import yaml
from safetensors.torch import save_file
from torch import nn
from torch_neuronx import BucketModelConfig

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.encoder_base import (
    NeuronEncoderApplication,
    NeuronEncoderBase,
)
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.checkpoint import _SAFETENSORS_MODEL_FILENAME
from neuronx_distributed_inference.modules.padding import pad_tensor, unpad_tensor
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(0)

MODEL_PATH = "/tmp/encoder_base/"
TRACED_MODEL_PATH = "/tmp/encoder_base/traced_model/"


if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)


def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


# Original implementation
class OriginalSimpleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        x = self.layer(x) + x
        return x


# Neuron implementation
class NeuronSimpleEncoder(NeuronEncoderBase):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.layer = nn.Identity()

    def forward(self, x):
        x = self.layer(x) + x
        return x


@torch.jit.script
def simple_encoder_bk(tensors: List[torch.Tensor], buckets, padding_side: str, pad_token: int):
    """
    Custom Bucket Router for this encoder model.
    Different model may have different routing logic.

    Reference: neuronx_distributed_inference.modules.autobucketing
    """
    image_size = tensors[0].shape[-1]
    for bucket_idx, bucket in enumerate(buckets):
        if image_size <= bucket:
            # route to closest bucket
            return tensors, torch.tensor(bucket_idx)
    # route to largest bucket
    return tensors, torch.tensor(-1)


def get_simple_encoder_bk():
    return simple_encoder_bk


class ModelWrapperSimpleEncoder(ModelWrapper):
    """
    Each encoder model should have their own ModelWrapper.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = self.get_bucket_model_config()

    def get_bucket_model_config(self):
        """Reference: neuronx_distributed_inference.modules.model_wrapper.get_bucket_model_config_from_tag"""
        bucket_degree = len(self.config.neuron_config.buckets)
        if bucket_degree == 1:
            return None

        logger.info("Using custom bucket kernel get_simple_encoder_bk")

        return BucketModelConfig(
            bucket_kernel=get_simple_encoder_bk,
            bucket_kernel_constant_args=(
                torch.tensor(self.config.neuron_config.buckets),
                self.config.neuron_config.padding_side,
                0,  # pad_token
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override ModelWrapper.input_generator().
        Generate a list of valid sample inputs containing one input list for each bucket.
        Different model may have a different set of input args.

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
        """
        inputs = []
        if not self.neuron_config.buckets:
            self.neuron_config.buckets = [self.config.image_size]
        for bucket in self.neuron_config.buckets:
            img_latent = torch.ones(
                [self.neuron_config.batch_size, self.config.in_channels, bucket, bucket]
            )
            inputs.append((img_latent,))
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def pad_inputs(
        self,
        *args: Tuple[torch.Tensor],
    ) -> List[List[torch.Tensor]]:
        """
        Override ModelWrapper.pad_inputs().
        Pad input to the closest bucket shape.
        Different model may have different input(s) that needs padding.

        Returns:
            padded_inputs (List[torch.Tensor]): All padded input args to the target bucket size.
            padding_masks (List[torch.Tensor]): All padding masks for each input arg.
        """
        padded_inputs = []
        padding_masks = []

        target_bucket_idx = self.get_target_bucket(*args)
        logger.info(f"target_bucket_idx {target_bucket_idx}")
        target_inputs = self.input_generator()[target_bucket_idx]

        for i, target_input in enumerate(target_inputs):
            padded_input, padding_mask = pad_tensor(args[i], target_input.shape)
            padded_inputs.append(padded_input)
            padding_masks.append(padding_mask.float())

        return padded_inputs, padding_masks

    def get_target_bucket(
        self,
        *args: Tuple[torch.Tensor],
    ) -> int:
        """
        Override ModelWrapper.get_target_bucket().
        Get the index of closest bucket.
        Different model may have different logic to identify closest bucket.

        Returns:
            int: target bucket index
        """
        if not self.neuron_config.buckets:
            return 0

        image_size = args[0].shape[-1]
        for i, bucket in enumerate(self.neuron_config.buckets):
            if image_size <= bucket:
                return i

        largest_bucket = self.neuron_config.buckets[-1]
        if image_size == largest_bucket:
            return -1

        raise ValueError(
            f"Image size {image_size} exceeds largest bucket ({largest_bucket}) for {self.tag}"
        )

    def unpad_outputs(
        self,
        outputs: List[torch.Tensor],
        padding_masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Unpad every output given their corresponding padding mask.
        Different model may have different unpadding logic.

        Returns:
            unpadded_outputs (List[torch.Tensor]): All unpadded outputs.
        """
        unpadded_outputs = []

        for output, mask in zip(outputs, padding_masks):
            unpadded_output = unpad_tensor(output, mask)
            unpadded_outputs.append(unpadded_output)

        return unpadded_outputs

    def forward(self, *args):
        """
        Override ModelWrapper.forward().
        """
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        logger.info(f"ModelWrapperSimpleEncoder.forward() args {type(args)} {args[0].shape}")
        padded_inputs, padding_masks = self.pad_inputs(*args)
        logger.info(
            f"ModelWrapperSimpleEncoder.forward() padded_inputs {type(padded_inputs)} {padded_inputs[0].shape}"
        )
        logger.info(
            f"ModelWrapperSimpleEncoder.forward() padding_masks {type(padding_masks)} {padding_masks[0].shape}"
        )
        logger.info(f"non zeros {torch.count_nonzero(padding_masks[0])}")

        output = self._forward(*padded_inputs)
        logger.info(f"ModelWrapperSimpleEncoder.forward() output {type(output)} {output.shape}")

        # unpad_outputs handles a list of outputs, but this simple encoder only have one single output
        unpadded_output = self.unpad_outputs([output], padding_masks)[0]
        logger.info(f"in Modelwrapper.unpad_outputs, unpadded_output.shape {unpadded_output.shape}")

        return unpadded_output


class NeuronSimpleEncoderModel(NeuronEncoderApplication):
    """
    Application class of the encoder model(s).
    """

    def get_model_wrapper_cls(self):
        # One encoder application could have multiple sub-models
        return [[NeuronSimpleEncoder, ModelWrapperSimpleEncoder]]

    def forward(self, *inputs) -> torch.Tensor:
        return self.NeuronSimpleEncoder(*inputs)


class TestEncoderBase(unittest.TestCase):
    def test_padding_no_bucketing(self):
        # Get config
        config = {  # any custom config
            "in_channels": 4,
            "out_channels": 2048,
            "kernel_size": 2,
            "stride": 2,
            "image_size": 128,
        }
        neuron_config = NeuronConfig(
            batch_size=1,
            torch_dtype=torch.float32,
        )
        inference_config = InferenceConfig(neuron_config, **config)

        # Get cpu model
        cpu_model = OriginalSimpleEncoder(inference_config)
        logger.info(f"cpu_model:\n{cpu_model}")
        # save state dict to be used to by neuron model
        save_ckpt_path = os.path.join(MODEL_PATH, _SAFETENSORS_MODEL_FILENAME)
        save_file(cpu_model.state_dict(), save_ckpt_path)
        logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

        # Get neuron model
        neuron_model = NeuronSimpleEncoderModel(model_path=MODEL_PATH, config=inference_config)

        # Compile and load model on Neuron
        neuron_model.compile(TRACED_MODEL_PATH)
        neuron_model.load(TRACED_MODEL_PATH)

        # Construct input tuple or dict, your model can have >=1 inputs
        test_inputs_0 = (
            torch.randn(
                [
                    inference_config.neuron_config.batch_size,
                    inference_config.in_channels,
                    32,  # should be padded to image_size 128
                    64,  # should be padded to image_size 128
                ]
            ),
        )

        # Run original implementation on CPU - get golden
        expected_output_0 = cpu_model(*test_inputs_0)

        # Run original implementation on Neuron
        actual_output_0 = neuron_model(*test_inputs_0)

        # Compare output logits
        passed, max_err = check_accuracy_embeddings(
            actual_output_0, expected_output_0, plot_outputs=False, rtol=1.3e-6, atol=1e-5
        )
        assert passed, f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}"

    def test_bucketing(self):
        # Get config
        config = {  # any custom config
            "in_channels": 4,
            "out_channels": 2048,
            "kernel_size": 2,
            "stride": 2,
            "image_size": 128,
        }
        neuron_config = NeuronConfig(
            batch_size=1,
            torch_dtype=torch.float32,
            buckets=[64, 128],
        )
        inference_config = InferenceConfig(neuron_config, **config)

        # Get cpu model
        cpu_model = OriginalSimpleEncoder(inference_config)
        logger.info(f"cpu_model:\n{cpu_model}")
        # save state dict to be used to by neuron model
        save_ckpt_path = os.path.join(MODEL_PATH, _SAFETENSORS_MODEL_FILENAME)
        save_file(cpu_model.state_dict(), save_ckpt_path)
        logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

        # Get neuron model
        neuron_model = NeuronSimpleEncoderModel(model_path=MODEL_PATH, config=inference_config)

        # Compile and load model on Neuron
        neuron_model.compile(TRACED_MODEL_PATH)
        neuron_model.load(TRACED_MODEL_PATH)

        """Test bucket 0"""
        # Construct input tuple or dict, your model can have >=1 inputs
        test_inputs_0 = (
            torch.randn(
                [
                    inference_config.neuron_config.batch_size,
                    inference_config.in_channels,
                    56,  # should be padded to bucket #0 size 64
                    56,  # should be padded to bucket #0 size 64
                ]
            ),
        )

        # Run original implementation on CPU - get golden
        expected_output_0 = cpu_model(*test_inputs_0)

        # Run original implementation on Neuron
        actual_output_0 = neuron_model(*test_inputs_0)

        # Compare output logits
        passed, max_err = check_accuracy_embeddings(
            actual_output_0, expected_output_0, plot_outputs=False, rtol=1.3e-6, atol=1e-5
        )
        assert passed, f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}"

        """Test bucket 1"""
        # Construct input tuple or dict, your model can have >=1 inputs
        test_inputs_1 = (
            torch.randn(
                [
                    inference_config.neuron_config.batch_size,
                    inference_config.in_channels,
                    116,  # should be padded to bucket #1 size 128
                    116,  # should be padded to bucket #1 size 128
                ]
            ),
        )

        # Run original implementation on CPU - get golden
        expected_output_1 = cpu_model(*test_inputs_1)

        # Run original implementation on Neuron
        actual_output_1 = neuron_model(*test_inputs_1)

        # Compare output logits
        passed, max_err = check_accuracy_embeddings(
            actual_output_1, expected_output_1, plot_outputs=False, rtol=1.3e-6, atol=1e-5
        )
        assert passed, f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}"


if __name__ == "__main__":
    # Set flags for debugging
    setup_debug_env()

    unittest.main()
