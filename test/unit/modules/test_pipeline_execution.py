# test_simple_pipeline.py

import logging
import os
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from typing import List, Tuple

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.encoder_base import NeuronEncoderBase
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.checkpoint import _SAFETENSORS_MODEL_FILENAME
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_DIR = "/tmp/simple_pipeline_test/"

def create_directories():
    directories = [
        BASE_DIR,
        os.path.join(BASE_DIR, "model1"),
        os.path.join(BASE_DIR, "model2"),
        os.path.join(BASE_DIR, "model3"),
        os.path.join(BASE_DIR, "model4"),
        os.path.join(BASE_DIR, "model5"),
        os.path.join(BASE_DIR, "compiled"),
        os.path.join(BASE_DIR, "compiled/model1"),
        os.path.join(BASE_DIR, "compiled/model2"),
        os.path.join(BASE_DIR, "compiled/model3"),
        os.path.join(BASE_DIR, "compiled/model4"),
        os.path.join(BASE_DIR, "compiled/model5")
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
# Simple Models
class Model1(NeuronEncoderBase):
    """
    First stage of pipeline: Input Transformer
    - Takes raw input tensor of size [batch_size, input_size=64]
    - Transforms it to hidden representation [batch_size, hidden_size=128]
    - Output is distributed across ranks (tp_degree=2) for parallel processing
    """
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.linear(x)

class Model2(NeuronEncoderBase):
    """
    Second stage of pipeline: Hidden State Processor
    - Takes ranked hidden state from Model1 [batch_size, hidden_size=128]
    - Transforms it to final output [batch_size, output_size=32]
    - Demonstrates simple pipeline connection: Model1 -> Model2
    """
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, config.output_size, bias=False)

    def forward(self, x):
        return self.linear(x)

class Model3(NeuronEncoderBase):
    """
    Third stage of pipeline: Mixed Input Processor
    - Takes two inputs:
        1. Ranked hidden state from Model1 [batch_size, hidden_size=128]
        2. Non-ranked auxiliary input [batch_size, aux_size=16]
    - Processes both inputs separately and combines results
    - Demonstrates handling mixed ranked/non-ranked inputs in pipeline
    """
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.linear1 = nn.Linear(config.hidden_size, config.output_size, bias=False)
        self.linear2 = nn.Linear(config.aux_size, config.output_size, bias=False)
        
    def forward(self, x1, x2):
        return self.linear1(x1) + self.linear2(x2)

class Model4(NeuronEncoderBase):
    """
    Fourth stage of pipeline: Multiple Mixed Input Processor
    Takes four inputs in CPU-Neuron-CPU-Neuron pattern:
    1. CPU input1 [batch_size, aux_size1=16]
    2. Ranked input1 from Model1 [batch_size, hidden_size=128]
    3. CPU input2 [batch_size, aux_size2=24]
    4. Ranked input2 from Model2 [batch_size, output_size=32]
    """
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.linear1 = nn.Linear(config.aux_size1, config.final_size, bias=False)
        self.linear2 = nn.Linear(config.hidden_size, config.final_size, bias=False)
        self.linear3 = nn.Linear(config.aux_size2, config.final_size, bias=False)
        self.linear4 = nn.Linear(config.output_size, config.final_size, bias=False)
        
    def forward(self, cpu1, ranked1, cpu2, ranked2):
        return (self.linear1(cpu1) + 
                self.linear2(ranked1) + 
                self.linear3(cpu2) + 
                self.linear4(ranked2))

class Model5(NeuronEncoderBase):
    """
    Fifth stage of pipeline: Post-processing on Model4 output
    - Takes Model4 output [batch_size, final_size=48]
    - Applies non-linear transformation
    - Demonstrates non-pipelined execution
    """
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.final_size, config.post_size, bias=False)
        
    def forward(self, x):
        # Add non-linear transformation
        x = F.relu(x)
        return self.linear(x)
        

class Model1Wrapper(ModelWrapper):
    """ModelWrapper for Model1 which transforms input through a linear layer."""
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = False,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, 
            model_cls, 
            tag, 
            compiler_args, 
            priority_model_idx, 
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        x = torch.ones([
            self.neuron_config.batch_size,
            self.config.input_size
        ])
        inputs = [(x,)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() before calling forward"
            )
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)
        output = self._forward(*args)
        return output

class Model2Wrapper(ModelWrapper):
    """ModelWrapper for Model2 which transforms hidden states."""
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = False,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, 
            model_cls, 
            tag, 
            compiler_args, 
            priority_model_idx, 
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        x = torch.ones([
            self.neuron_config.batch_size,
            self.config.hidden_size
        ])
        inputs = [(x,)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() before calling forward"
            )
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)
        output = self._forward(*args)
        return output

class Model3Wrapper(ModelWrapper):
    """ModelWrapper for Model3 which handles mixed ranked/non-ranked inputs."""
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = False,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, 
            model_cls, 
            tag, 
            compiler_args, 
            priority_model_idx, 
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        x1 = torch.ones([
            self.neuron_config.batch_size,
            self.config.hidden_size
        ])
        x2 = torch.ones([
            self.neuron_config.batch_size,
            self.config.aux_size
        ])
        inputs = [(x1, x2)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() before calling forward"
            )
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)
        output = self._forward(*args)
        return output

class Model4Wrapper(ModelWrapper):
    """ModelWrapper for Model4 which handles multiple alternating CPU and ranked inputs."""
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = True,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, 
            model_cls, 
            tag, 
            compiler_args, 
            priority_model_idx, 
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        cpu1 = torch.ones([
            self.neuron_config.batch_size,
            self.config.aux_size1
        ])
        ranked1 = torch.ones([
            self.neuron_config.batch_size,
            self.config.hidden_size
        ])
        cpu2 = torch.ones([
            self.neuron_config.batch_size,
            self.config.aux_size2
        ])
        ranked2 = torch.ones([
            self.neuron_config.batch_size,
            self.config.output_size
        ])
        inputs = [(cpu1, ranked1, cpu2, ranked2)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() before calling forward"
            )
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)
        output = self._forward(*args)
        return output

class Model5Wrapper(ModelWrapper):
    """ModelWrapper for Model5 with pipeline_execution=False"""
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = False,  # Set to False
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, 
            model_cls, 
            tag, 
            compiler_args, 
            priority_model_idx, 
            pipeline_execution,
            model_init_kwargs,
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        x = torch.ones([
            self.neuron_config.batch_size,
            self.config.final_size
        ])
        inputs = [(x,)]
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(self.model_cls, self.config)

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() before calling forward"
            )
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)
        output = self._forward(*args)
        return output

    
class Model1App(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = Model1Wrapper(
            config=config,
            model_cls=Model1,
            pipeline_execution=True,
            tag="model1"
        )
        self.models.append(self.model)
    
    def forward(self, x):
        return self.models[0].forward(x)

class Model2App(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = Model2Wrapper(
            config=config,
            model_cls=Model2,
            pipeline_execution=True,
            tag="model2"
        )
        self.models.append(self.model)
    
    def forward(self, x):
        return self.models[0].forward(x)

class Model3App(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = Model3Wrapper(
            config=config,
            model_cls=Model3,
            pipeline_execution=True,
            tag="model3"
        )
        self.models.append(self.model)
    
    def forward(self, x1, x2):
        return self.models[0].forward(x1, x2)

class Model4App(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = Model4Wrapper(
            config=config,
            model_cls=Model4,
            pipeline_execution=True,
            return_ranked_to_cpu=True,
            tag="model4"
        )
        self.models.append(self.model)
    
    def forward(self, cpu1, ranked1, cpu2, ranked2):
        return self.models[0].forward(cpu1, ranked1, cpu2, ranked2)
    

class Model5App(NeuronApplicationBase):
    def __init__(self, model_path: str, config: InferenceConfig):
        super().__init__(model_path=model_path, config=config)
        self.model = Model5Wrapper(
            config=config,
            model_cls=Model5,
            pipeline_execution=False,  # Set pipeline execution to False
            tag="model5"
        )
        self.models.append(self.model)
    
    def forward(self, x):
        return self.models[0].forward(x)
    
class SimplePipeline(nn.Module):
    """
        Complete Pipeline Architecture:

                                        [Neuron]                                      [Neuron]
                            --> Model2 --------------------------> output2 [batch, output_size]
                            /                                    \
        [CPU]              /                                     \
        Input --> [Neuron] /                                     \
                Model1    \                                     --> [Neuron] --> output4 [batch, final_size]
                            \          [CPU]                     /    Model4        ^          ^
                            --> [Neuron] <-- auxiliary_input1  /                   |          |
                                Model3        |                                 [CPU]       [CPU]
                                |                                          aux_input2  aux_input1
                                v
                                [Neuron]
                                output3 [batch, output_size]

        Data Flow Types:
        [CPU]   : Tensor on CPU
        [Neuron]: Tensor on Neuron device
    """
    def __init__(self, config1: InferenceConfig, config2: InferenceConfig, 
                 config3: InferenceConfig, config4: InferenceConfig,
                 config5: InferenceConfig):  # Add config5
        super().__init__()
        self.model1_path = os.path.join(BASE_DIR, "model1")
        self.model2_path = os.path.join(BASE_DIR, "model2")
        self.model3_path = os.path.join(BASE_DIR, "model3")
        self.model4_path = os.path.join(BASE_DIR, "model4")
        self.model5_path = os.path.join(BASE_DIR, "model5")

        self.model1_app = Model1App(self.model1_path, config1)
        self.model2_app = Model2App(self.model2_path, config2)
        self.model3_app = Model3App(self.model3_path, config3)
        self.model4_app = Model4App(self.model4_path, config4)
        self.model5_app = Model5App(self.model5_path, config5)
        
        self.config1 = config1
        self.config2 = config2
        self.config3 = config3
        self.config4 = config4
        self.config5 = config5


    def compile_and_load(self, compiled_path):
        for app, name in [(self.model1_app, "model1"), 
                         (self.model2_app, "model2"),
                         (self.model3_app, "model3"),
                         (self.model4_app, "model4"),
                         (self.model5_app, "model5")]:
            app.compile(os.path.join(compiled_path, name))
            app.load(os.path.join(compiled_path, name))
        
        
    def forward(self, x, aux_input1, aux_input2):
        # First stage: transform input
        model1_output = self.model1_app.forward(x)

        # Second stage: parallel branches
        model2_output = self.model2_app.forward(model1_output)
        model3_output = self.model3_app.forward(model1_output, aux_input1)
        
        # Third stage: combine multiple inputs
        model4_output = self.model4_app.forward(
            aux_input1,
            model1_output,
            aux_input2,
            model2_output
        )
        
        model4_output_cpu = model4_output
        model5_output = self.model5_app.forward(model4_output_cpu)
        
        return model2_output, model3_output, model4_output, model5_output
    

def test_pipeline():
    """Test the complete pipeline with all models."""
    # Create all necessary directories
    create_directories()
    
    # Configs for all models
    config1 = InferenceConfig(
        NeuronConfig(batch_size=1, torch_dtype=torch.float32, tp_degree=2),
        input_size=64,
        hidden_size=128
    )
    
    config2 = InferenceConfig(
        NeuronConfig(batch_size=1, torch_dtype=torch.float32, tp_degree=2),
        hidden_size=128,
        output_size=32
    )
    
    config3 = InferenceConfig(
        NeuronConfig(batch_size=1, torch_dtype=torch.float32, tp_degree=2),
        hidden_size=128,
        aux_size=16,
        output_size=32
    )

    config4 = InferenceConfig(
        NeuronConfig(batch_size=1, torch_dtype=torch.float32, tp_degree=2),
        hidden_size=128,
        output_size=32,
        aux_size1=16,
        aux_size2=24,
        final_size=48
    )

    config5 = InferenceConfig(
        NeuronConfig(batch_size=1, torch_dtype=torch.float32, tp_degree=2),
        final_size=48,
        post_size=24  # New output size after post-processing
    )

    # Create CPU models
    cpu_model1 = Model1(config1)
    cpu_model2 = Model2(config2)
    cpu_model3 = Model3(config3)
    cpu_model4 = Model4(config4)
    cpu_model5 = Model5(config5)
    
    # Save model states
    save_file(cpu_model1.state_dict(), 
              os.path.join(BASE_DIR, "model1", _SAFETENSORS_MODEL_FILENAME))
    save_file(cpu_model2.state_dict(), 
              os.path.join(BASE_DIR, "model2", _SAFETENSORS_MODEL_FILENAME))
    save_file(cpu_model3.state_dict(), 
              os.path.join(BASE_DIR, "model3", _SAFETENSORS_MODEL_FILENAME))
    save_file(cpu_model4.state_dict(), 
              os.path.join(BASE_DIR, "model4", _SAFETENSORS_MODEL_FILENAME))
    save_file(cpu_model5.state_dict(), 
              os.path.join(BASE_DIR, "model5", _SAFETENSORS_MODEL_FILENAME))

    # Create pipeline
    pipeline = SimplePipeline(config1, config2, config3, config4, config5)
    
    # Compile and load
    compiled_path = os.path.join(BASE_DIR, "compiled")
    pipeline.compile_and_load(compiled_path)

    # Create test inputs
    test_input = torch.randn(1, 64)     # [batch_size, input_size]
    aux_input1 = torch.randn(1, 16)     # [batch_size, aux_size1]
    aux_input2 = torch.randn(1, 24)     # [batch_size, aux_size2]

    # Generate CPU reference outputs
    with torch.no_grad():
        # Model1 intermediate output used by other models
        cpu_intermediate1 = cpu_model1(test_input)
        
        # Model2 output
        cpu_output2 = cpu_model2(cpu_intermediate1)
        
        # Model3 output (uses Model1 output and aux_input1)
        cpu_output3 = cpu_model3(cpu_intermediate1, aux_input1)
        
        # Model4 output (uses aux_input1, Model1 output, aux_input2, Model2 output)
        cpu_output4 = cpu_model4(
            aux_input1,         # CPU input1
            cpu_intermediate1,  # Ranked input1 from Model1
            aux_input2,        # CPU input2
            cpu_output2        # Ranked input2 from Model2
        )
        
        cpu_output5 = cpu_model5(cpu_output4)

    # Get pipeline outputs including Model5
    pipeline_output2, pipeline_output3, pipeline_output4, pipeline_output5 = pipeline(
        test_input, aux_input1, aux_input2
    )
    
    # Convert pipeline outputs to CPU
    pipeline_output2 = pipeline_output2[0][0].to('cpu')
    pipeline_output3 = pipeline_output3[0][0].to('cpu')

    # Test if Model4 output is already on CPU
    assert pipeline_output4.device.type == 'cpu', "Model4 output should be on CPU due to return_ranked_to_cpu=True"
    assert pipeline_output5.device.type == 'cpu', "Model5 output should be on CPU due to pipeline_execution=False"
    


    # Check accuracy for all models
    results = []
    for name, p_out, c_out in [
        ("Model2", pipeline_output2, cpu_output2),
        ("Model3", pipeline_output3, cpu_output3),
        ("Model4", pipeline_output4, cpu_output4),
        ("Model5", pipeline_output5, cpu_output5)
    ]:
        passed, max_err = check_accuracy_embeddings(
            p_out, 
            c_out, 
            plot_outputs=False, 
            rtol=1.3e-6, 
            atol=1e-5
        )
        print(f"Pipeline test for {name} {'passed' if passed else 'failed'} "
              f"with max error: {max_err}")
        results.append(passed)
    
    
    # Test passes only if all models pass
    final_result = all(results)
    print(f"\nOverall pipeline test: {'PASSED' if final_result else 'FAILED'}")


if __name__ == "__main__":
    # Set environment variables
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    
    # Run test
    test_pipeline()
