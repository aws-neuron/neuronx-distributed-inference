import logging
import os
from typing import List, Tuple

import torch
from safetensors.torch import save_file
from torch import nn

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.encoder_base import NeuronEncoderBase
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.checkpoint import _SAFETENSORS_MODEL_FILENAME
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from torch.profiler import profile, ProfilerActivity
import torch_xla

def is_neuron_input(x):
    """
    Check if input is a Neuron ranked input (list of lists with non-CPU tensors)
    
    Args:
        x: Input to check - either CPU tensor or list of ranked tensors
        
    Returns:
        bool: True if input is Neuron ranked format, False if CPU tensor
    """
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list) and len(x[0]) > 0:
        # Check device of first tensor in ranked format
        if hasattr(x[0][0], 'device'):
            return x[0][0].device.type != 'cpu'
    return False

def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)
    
def profile_function(func, *args, name):
    """
    Profile a function using torch.profiler.
    
    Args:
    func (callable): The function to profile
    *args: Arguments to pass to the function
    name (str): A name for the profiling session

    Returns:
    None
    """
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for _ in range(10):
            func(*args)
    
    prof.export_chrome_trace(f"torch_profile_{name}.json")
    print(f"Profiling results for {name}:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_DIR = "/tmp/encoder_pipeline/"

if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)


class SimpleVisionEncoder(NeuronEncoderBase):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.layer = nn.Linear(config.in_channels, config.hidden_size, bias=False)

    def forward(self, image):
        image = image.flatten(start_dim=2).transpose(-1, -2)
        return self.layer(image)


class SimpleTextEncoder(NeuronEncoderBase):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.layer = nn.Linear(config.vocab_size, config.hidden_size, bias=False)

    def forward(self, text_seq):
        return self.layer(text_seq)


class SimpleFusionEncoder(NeuronEncoderBase):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, image_emb, text_emb):
        emb = torch.cat((image_emb, text_emb), dim=1)
        return self.layer(emb)


class ModelWrapperSimpleVisionEncoder(ModelWrapper):
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
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        image = torch.ones(
            [
                self.neuron_config.batch_size,
                self.config.in_channels,
                self.config.image_size,
                self.config.image_size,
            ]
        )
        inputs = [(image,)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        output = self._forward(*args)
        return output


class ModelWrapperSimpleTextEncoder(ModelWrapper):
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
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        text_seq = torch.ones(
            [self.neuron_config.batch_size, self.config.seq_len, self.config.vocab_size]
        )
        inputs = [(text_seq,)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        output = self._forward(*args)
        return output


class ModelWrapperSimpleFusionEncoder(ModelWrapper):
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
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        image_emb = torch.ones(
            [
                self.neuron_config.batch_size,
                self.config.image_size * self.config.image_size,
                self.config.hidden_size,
            ]
        )
        text_emb = torch.ones(
            [self.neuron_config.batch_size, self.config.seq_len, self.config.hidden_size]
        )
        inputs = [(image_emb, text_emb)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        output = self._forward(*args)
        return output
    

class NeuronSimpleVisionEncoderApplication(NeuronApplicationBase):
    _model_cls = SimpleVisionEncoder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return ModelWrapperSimpleVisionEncoder

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1 "
        compiler_args += "--model-type=transformer"
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        print(f"compiler_args: {compiler_args}")
        return compiler_args

    def forward(self, image):
        rank_inputs = [
            [image] for _ in range(self.neuron_config.tp_degree)
        ]
        outputs = self.models[0].model.nxd_model.forward_ranked(rank_inputs)
        return outputs


class NeuronSimpleTextEncoderApplication(NeuronApplicationBase):
    _model_cls = SimpleTextEncoder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return ModelWrapperSimpleTextEncoder

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1 "
        compiler_args += "--model-type=transformer"
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        print(f"compiler_args: {compiler_args}")
        return compiler_args
    

    def forward(self, text_seq):
        rank_inputs = [
            [text_seq] for _ in range(self.neuron_config.tp_degree)
        ]
        outputs = self.models[0].model.nxd_model.forward_ranked(rank_inputs)
        return outputs


class NeuronSimpleFusionEncoderApplication(NeuronApplicationBase):
    _model_cls = SimpleFusionEncoder

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return ModelWrapperSimpleFusionEncoder

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation -O1 "
        compiler_args += "--model-type=transformer"
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        print(f"compiler_args: {compiler_args}")
        return compiler_args

    
    def forward(self, image_emb, text_emb):
        """Forward pass with ranked inputs/outputs"""
        
        fusion_ranked_inputs = [
            [image_emb, text_emb] 
            for _ in range(self.neuron_config.tp_degree)
        ]
        
        fusion_outputs = self.models[0].model.nxd_model.forward_ranked(fusion_ranked_inputs)
        return fusion_outputs
    


class NeuronSimpleEncoderPipeline(nn.Module):
    def __init__(
        self,
        model_path: str,
        vision_config: InferenceConfig,
        text_config: InferenceConfig,
        fusion_config: InferenceConfig,
    ):
        super().__init__()

        self.model_path = model_path
        self.set_up_model_path()

        self.vision_config = vision_config
        self.text_config = text_config
        self.fusion_config = fusion_config

        self.vision_encoder = NeuronSimpleVisionEncoderApplication(
            model_path=self.vision_encoder_path, config=self.vision_config
        )
        self.text_encoder = NeuronSimpleTextEncoderApplication(
            model_path=self.text_encoder_path, config=self.text_config
        )
        self.fusion_encoder = NeuronSimpleFusionEncoderApplication(
            model_path=self.fusion_encoder_path, config=self.fusion_config
        )

    def set_up_model_path(self):
        self.vision_encoder_path = os.path.join(self.model_path, "SimpleVisionEncoder")
        self.text_encoder_path = os.path.join(self.model_path, "SimpleTextEncoder")
        self.fusion_encoder_path = os.path.join(self.model_path, "SimpleFusionEncoder")

    def compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None):
        self.vision_encoder.compile(
            os.path.join(compiled_model_path, "SimpleVisionEncoder/"), debug, pre_shard_weights_hook
        )
        self.text_encoder.compile(
            os.path.join(compiled_model_path, "SimpleTextEncoder/"), debug, pre_shard_weights_hook
        )
        self.fusion_encoder.compile(
            os.path.join(compiled_model_path, "SimpleFusionEncoder/"), debug, pre_shard_weights_hook
        )
        self.compiled_model_path = compiled_model_path

    def load(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False
    ):
        self.vision_encoder.load(
            os.path.join(compiled_model_path, "SimpleVisionEncoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self.text_encoder.load(
            os.path.join(compiled_model_path, "SimpleTextEncoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup,
        )
        self.fusion_encoder.load(
            os.path.join(compiled_model_path, "SimpleFusionEncoder/"),
            start_rank_id,
            local_ranks_size,
            skip_warmup=skip_warmup,
        )

    def forward(self, image, text_seq) -> torch.Tensor:
        """Regular forward pass without ranking"""
        image_emb = self.vision_encoder.models[0](image)
        text_emb = self.text_encoder.models[0](text_seq)
        out = self.fusion_encoder.models[0](image_emb, text_emb)
        return out

    def forward_ranked(self, image, text_seq) -> torch.Tensor:
        """Forward pass with ranked inputs/outputs"""
        vision_ranked_outputs = self.vision_encoder(image)  # [[v0], [v1], ..., [vn]]
        text_ranked_outputs = self.text_encoder(text_seq)   # [[t0], [t1], ..., [tn]]
        
        fusion_ranked_inputs = [
            [vision_ranked_outputs[i][0], text_ranked_outputs[i][0]] 
            for i in range(self.vision_config.neuron_config.tp_degree) 
        ] # [ [v0, t0 ], [v1, t1], [v2, t2] ... [vn, tn]]
        
        fusion_outputs = self.fusion_encoder.models[0].model.nxd_model.forward_ranked(fusion_ranked_inputs)
        return fusion_outputs
    
    # Usage example:
    def forward_mixed_ranked(self, image_emb, text_emb) -> torch.Tensor:
        """
        Forward pass handling both CPU and ranked Neuron inputs
        
        Args:
            image_emb: Either CPU tensor or list of ranked tensors for vision embeddings
            text_emb: Either CPU tensor or list of ranked tensors for text embeddings
            
        Returns:
            Ranked outputs from fusion encoder
        """
        # Check input types
        vision_on_cpu = not is_neuron_input(image_emb)
        text_on_cpu = not is_neuron_input(text_emb)
        
        # Handle vision embeddings
        if vision_on_cpu:
            # Replicate CPU tensor for each rank
            vision_ranked = [[image_emb] for _ in range(self.vision_config.neuron_config.tp_degree)]
        else:
            vision_ranked = image_emb
            
        # Handle text embeddings 
        if text_on_cpu:
            # Replicate CPU tensor for each rank
            text_ranked = [[text_emb] for _ in range(self.text_config.neuron_config.tp_degree)]
        else:
            text_ranked = text_emb

        # Combine ranked inputs for fusion
        fusion_ranked_inputs = [
            [vision_ranked[i][0], text_ranked[i][0]]
            for i in range(self.vision_config.neuron_config.tp_degree)
        ]
        
        # Forward through fusion encoder
        fusion_outputs = self.fusion_encoder.models[0].model.nxd_model.forward_ranked(fusion_ranked_inputs)
        return fusion_outputs


def get_configs():
    vision_config_data = {
        "in_channels": 4,
        "hidden_size": 256,
        "image_size": 32,
    }
    vision_neuron_config = NeuronConfig(
        batch_size=1,
        torch_dtype=torch.float32,
        tp_degree=2,
    )
    vision_config = InferenceConfig(vision_neuron_config, **vision_config_data)

    text_config_data = {
        "seq_len": 128,
        "vocab_size": 4096,
        "hidden_size": 256,
    }
    text_neuron_config = NeuronConfig(
        batch_size=1,
        torch_dtype=torch.float32,
        tp_degree=2,
    )
    text_config = InferenceConfig(text_neuron_config, **text_config_data)

    fusion_config_data = {
        "seq_len": 128,
        "image_size": 32,
        "hidden_size": 256,
    }
    fusion_neuron_config = NeuronConfig(
        batch_size=1,
        torch_dtype=torch.float32,
        tp_degree=2,
    )
    fusion_config = InferenceConfig(fusion_neuron_config, **fusion_config_data)

    return vision_config, text_config, fusion_config


def test_vision_encoder():
    model_path = os.path.join(BASE_DIR, "SimpleVisionEncoder")
    traced_model_path = os.path.join(BASE_DIR, "traced_model", "SimpleVisionEncoder")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    vision_config, _, _ = get_configs()

    cpu_model = SimpleVisionEncoder(vision_config)
    logger.info(f"cpu_model:\n{cpu_model}")
    save_ckpt_path = os.path.join(model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_model.state_dict(), save_ckpt_path)
    logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

    neuron_model = NeuronSimpleVisionEncoderApplication(
        model_path=model_path, config=vision_config
    )

    neuron_model.compile(traced_model_path)
    neuron_model.load(traced_model_path)

    test_inputs_vision = (
        torch.randn(
            [
                vision_config.neuron_config.batch_size,
                vision_config.in_channels,
                vision_config.image_size,
                vision_config.image_size,
            ]
        ),
    )
    torch.save(test_inputs_vision[0], os.path.join(BASE_DIR, "input_image.pt"))

    expected_output_vision = cpu_model(*test_inputs_vision)
    torch.save(expected_output_vision, os.path.join(BASE_DIR, "output_image_emb.pt"))

    actual_output_vision = neuron_model(*test_inputs_vision)
    cpu = torch.device("cpu")
    actual_output_vision = actual_output_vision[0][0].to(cpu)

    passed, max_err = check_accuracy_embeddings(
        actual_output_vision, expected_output_vision, plot_outputs=False, rtol=1.3e-6, atol=1e-5
    )
    assert passed, f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}"


def test_text_encoder():
    model_path = os.path.join(BASE_DIR, "SimpleTextEncoder")
    traced_model_path = os.path.join(BASE_DIR, "traced_model", "SimpleTextEncoder")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    _, text_config, _ = get_configs()

    cpu_model = SimpleTextEncoder(text_config)
    logger.info(f"cpu_model:\n{cpu_model}")
    save_ckpt_path = os.path.join(model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_model.state_dict(), save_ckpt_path)
    logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

    neuron_model = NeuronSimpleTextEncoderApplication(model_path=model_path, config=text_config)

    neuron_model.compile(traced_model_path)
    neuron_model.load(traced_model_path)

    test_inputs_text = (
        torch.randn(
            [
                text_config.neuron_config.batch_size,
                text_config.seq_len,
                text_config.vocab_size,
            ]
        ),
    )
    torch.save(test_inputs_text[0], os.path.join(BASE_DIR, "input_text.pt"))

    expected_output_text = cpu_model(*test_inputs_text)
    torch.save(expected_output_text, os.path.join(BASE_DIR, "output_text_emb.pt"))

    actual_output_text = neuron_model(*test_inputs_text)
    cpu = torch.device("cpu")
    actual_output_text = actual_output_text[0][0].to(cpu)

    passed, max_err = check_accuracy_embeddings(
        actual_output_text, expected_output_text, plot_outputs=False, rtol=1.3e-6, atol=1e-5
    )
    assert passed, f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}"


def test_e2e_pipeline():
    """
    End-to-end pipeline test with proper state dict management
    """
    # Get configs
    vision_config, text_config, fusion_config = get_configs()

    # Set up model paths
    vision_model_path = os.path.join(BASE_DIR, "SimpleVisionEncoder")
    text_model_path = os.path.join(BASE_DIR, "SimpleTextEncoder")
    fusion_model_path = os.path.join(BASE_DIR, "SimpleFusionEncoder")
    
    # Create directories if they don't exist
    for path in [vision_model_path, text_model_path, fusion_model_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Initialize CPU models and save state dicts
    # Vision encoder
    cpu_vision_model = SimpleVisionEncoder(vision_config)
    vision_ckpt_path = os.path.join(vision_model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_vision_model.state_dict(), vision_ckpt_path)
    logger.info(f"Saved vision encoder checkpoint to {vision_ckpt_path}")

    # Text encoder
    cpu_text_model = SimpleTextEncoder(text_config)
    text_ckpt_path = os.path.join(text_model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_text_model.state_dict(), text_ckpt_path)
    logger.info(f"Saved text encoder checkpoint to {text_ckpt_path}")

    # Fusion encoder
    cpu_fusion_model = SimpleFusionEncoder(fusion_config)
    fusion_ckpt_path = os.path.join(fusion_model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_fusion_model.state_dict(), fusion_ckpt_path)
    logger.info(f"Saved fusion encoder checkpoint to {fusion_ckpt_path}")

    # Initialize Neuron pipeline
    neuron_model = NeuronSimpleEncoderPipeline(
        model_path=BASE_DIR,
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=fusion_config,
    )

    # Compile and load models
    traced_model_path = os.path.join(BASE_DIR, "traced_model/")
    neuron_model.compile(traced_model_path)
    neuron_model.load(traced_model_path)

    # Load test inputs
    test_inputs = (
        torch.load(os.path.join(BASE_DIR, "input_image.pt")),
        torch.load(os.path.join(BASE_DIR, "input_text.pt")),
    )

    # Generate golden output using CPU models
    cpu_vision_emb = cpu_vision_model(test_inputs[0])
    cpu_text_emb = cpu_text_model(test_inputs[1])
    golden_output = cpu_fusion_model(cpu_vision_emb, cpu_text_emb)
    
    # Save golden output
    torch.save(golden_output, os.path.join(BASE_DIR, "output_pipeline_golden.pt"))

    # Test ranked forward
    print("Testing ranked forward in Pipeline...")
    actual_output_ranked = neuron_model.forward_ranked(*test_inputs)
    actual_output_ranked = actual_output_ranked[0][0].to('cpu')
    
    # Compare with golden output
    passed, max_err = check_accuracy_embeddings(
        actual_output_ranked,
        golden_output,
        plot_outputs=False,
        rtol=1.3e-6,
        atol=1e-5,
    )
    
    # Profile the ranked forward
    profile_function(
        neuron_model.forward_ranked,
        *test_inputs,
        name="e2e_ranked_forward"
    )
    
    assert passed, f"Ranked forward passed accuracy validation: {passed}, max_err: {max_err}"
    print("End-to-end pipeline test passed!")


def test_fusion_encoder():
    model_path = os.path.join(BASE_DIR, "SimpleFusionEncoder")
    traced_model_path = os.path.join(BASE_DIR, "traced_model", "SimpleFusionEncoder")
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Get config
    _, _, fusion_config = get_configs()

    # Get cpu model
    cpu_model = SimpleFusionEncoder(fusion_config)
    logger.info(f"cpu_model:\n{cpu_model}")
    save_ckpt_path = os.path.join(model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_model.state_dict(), save_ckpt_path)
    logger.info(f"Got cpu_model, saved checkpoint to {save_ckpt_path}")

    # Get neuron model
    neuron_model = NeuronSimpleFusionEncoderApplication(
        model_path=model_path, config=fusion_config
    )

    # Compile and load model on Neuron
    neuron_model.compile(traced_model_path)
    neuron_model.load(traced_model_path)

    # Construct input tuple or dict, your model can have >=1 inputs
    test_inputs_fusion = (
        torch.load(os.path.join(BASE_DIR, "output_image_emb.pt")),
        torch.load(os.path.join(BASE_DIR, "output_text_emb.pt")),
    )

    # Run on CPU - get golden
    expected_output_fusion = cpu_model(*test_inputs_fusion)
    # Save output to be used by other test
    torch.save(expected_output_fusion, os.path.join(BASE_DIR, "output_pipeline_golden.pt"))

    # Run on Neuron
    actual_output_fusion = neuron_model(*test_inputs_fusion)
    cpu = torch.device("cpu")
    actual_output_fusion = actual_output_fusion[0][0].to(cpu)

    # Compare output logits
    passed, max_err = check_accuracy_embeddings(
        actual_output_fusion, 
        expected_output_fusion, 
        plot_outputs=False, 
        rtol=1.3e-6, 
        atol=1e-5
    )
    
    profile_function(neuron_model, *test_inputs_fusion, name="e2e_forward_on_rank")
    assert passed, f"Embeddings passed accuracy validation: {passed}, max_err: {max_err}"


def test_mixed_cpu_neuron_pipeline():
    """
    Tests pipeline with CPU vision encoder + Neuron text/fusion encoders
    to demonstrate mixed execution capability.
    """
    # Get configs
    vision_config, text_config, fusion_config = get_configs()

    # Set up model paths
    vision_model_path = os.path.join(BASE_DIR, "SimpleVisionEncoder")
    text_model_path = os.path.join(BASE_DIR, "SimpleTextEncoder")
    fusion_model_path = os.path.join(BASE_DIR, "SimpleFusionEncoder")
    
    # Create directories if they don't exist
    for path in [vision_model_path, text_model_path, fusion_model_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Initialize and save CPU models
    # Vision encoder
    cpu_vision_model = SimpleVisionEncoder(vision_config)
    vision_ckpt_path = os.path.join(vision_model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_vision_model.state_dict(), vision_ckpt_path)
    logger.info(f"Saved vision encoder checkpoint to {vision_ckpt_path}")

    # Text encoder
    cpu_text_model = SimpleTextEncoder(text_config)
    text_ckpt_path = os.path.join(text_model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_text_model.state_dict(), text_ckpt_path)
    logger.info(f"Saved text encoder checkpoint to {text_ckpt_path}")

    # Fusion encoder
    cpu_fusion_model = SimpleFusionEncoder(fusion_config)
    fusion_ckpt_path = os.path.join(fusion_model_path, _SAFETENSORS_MODEL_FILENAME)
    save_file(cpu_fusion_model.state_dict(), fusion_ckpt_path)
    logger.info(f"Saved fusion encoder checkpoint to {fusion_ckpt_path}")

    # Initialize pipeline
    neuron_model = NeuronSimpleEncoderPipeline(
        model_path=BASE_DIR,
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=fusion_config,
    )

    # Compile and load models
    traced_model_path = os.path.join(BASE_DIR, "traced_model/")
    neuron_model.compile(traced_model_path)
    neuron_model.load(traced_model_path)

    # Load test inputs
    test_image = torch.load(os.path.join(BASE_DIR, "input_image.pt"))
    test_text = torch.load(os.path.join(BASE_DIR, "input_text.pt"))

    # Generate CPU embeddings and golden output
    cpu_vision_emb = cpu_vision_model(test_image)
    cpu_text_emb = cpu_text_model(test_text)
    golden_output = cpu_fusion_model(cpu_vision_emb, cpu_text_emb)
    
    # Save golden output
    torch.save(golden_output, os.path.join(BASE_DIR, "output_pipeline_mixed_golden.pt"))
    
    # Get Neuron text embeddings through ranked forward
    text_ranked_outputs = neuron_model.text_encoder(test_text)

    print("Testing mixed CPU + Neuron inputs...")
    print(f"CPU Vision embedding shape: {cpu_vision_emb.shape}")
    print(f"Neuron Text embedding shape: {text_ranked_outputs[0][0].shape}")
    
    # Run mixed forward through fusion encoder
    actual_output = neuron_model.forward_mixed_ranked(
        image_emb=cpu_vision_emb,  # CPU tensor
        text_emb=text_ranked_outputs  # Ranked Neuron output
    )

    # Move output to CPU for comparison
    actual_output = actual_output[0][0].to('cpu')

    # Compare with golden output
    passed, max_err = check_accuracy_embeddings(
        actual_output,
        golden_output,
        plot_outputs=False,
        rtol=1.3e-6,
        atol=1e-5,
    )
    
    # Profile the mixed execution
    profile_function(
        neuron_model.forward_mixed_ranked,
        cpu_vision_emb,
        text_ranked_outputs,
        name="mixed_cpu_neuron_forward"
    )
    
    assert passed, f"Mixed CPU-Neuron pipeline validation: {passed}, max_err: {max_err}"
    print("Mixed CPU-Neuron pipeline test passed!")
    
    

if __name__ == "__main__":
    setup_debug_env()
    
    # Run individual encoder tests first
    test_vision_encoder()
    test_text_encoder()
    test_fusion_encoder()
    # # Run mixed CPU + Neuron Input test:
    test_mixed_cpu_neuron_pipeline()
    # Run end-to-end pipeline test
    test_e2e_pipeline()
