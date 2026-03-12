import torch
import os
import numpy as np

from transformers import AutoProcessor, GenerationConfig
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig as SmplConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl import NeuronQwen2VLForCausalLM, Qwen2VLInferenceConfig
from neuronx_distributed_inference.models.qwen2_vl.utils.input_processor import prepare_generation_inputs_hf
from neuronx_distributed_inference.utils.benchmark import LatencyCollector

model_path = "/shared/cache/checkpoints/qwen/Qwen2-VL-7B-Instruct/"
traced_model_path = "/shared/cache/checkpoints/qwen/Qwen2-VL-7B-Instruct/traced_model/"
torch.manual_seed(0)


NUM_OF_IMAGES = 128
TEXT_SEQ_LENGTH = 32768
VISION_SEQ_LENGTH = 1012*NUM_OF_IMAGES
TEXT_BUCKETS = [2048, 15360, 32768]
VISION_BUCKETS = [1, 50, 128]


def generate_image_to_text():
    # Initialize configs and tokenizer.
    text_neuron_config = NeuronConfig(batch_size=1,
                                seq_len=TEXT_SEQ_LENGTH,
                                ctx_batch_size=1,
                                tp_degree=4,
                                world_size=4,
                                torch_dtype=torch.float16,
                                attention_dtype=torch.float16,
                                rpl_reduce_dtype=torch.float16,
                                cp_degree=1,
                                save_sharded_checkpoint=True,
                                sequence_parallel_enabled=True, 
                                fused_qkv=True,
                                qkv_kernel_enabled=True,
                                mlp_kernel_enabled=False,
                                enable_bucketing = True,
                                context_encoding_buckets=TEXT_BUCKETS,
                                token_generation_buckets=TEXT_BUCKETS,
                                attn_kernel_enabled=True,
                                attn_block_tkg_nki_kernel_enabled=False,
                                attn_block_tkg_nki_kernel_cache_update=False,
                                on_device_sampling_config=SmplConfig(dynamic=False, top_k=1),
                                cc_pipeline_tiling_factor=2,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                )

    vision_neuron_config = NeuronConfig(batch_size=1,
                                seq_len=VISION_SEQ_LENGTH,
                                tp_degree=4,
                                world_size=4,
                                enable_bucketing=True,
                                save_sharded_checkpoint=True,
                                torch_dtype=torch.bfloat16,
                                buckets=VISION_BUCKETS,
                                cc_pipeline_tiling_factor=2,
                                fused_qkv=True,
                                qkv_kernel_enabled=False,  # disable qvk kernel due to compile error
                                attn_kernel_enabled=True,
                                mlp_kernel_enabled=True,
                                cast_type="as-declared",
                                logical_neuron_cores=2,
                                )

    config = Qwen2VLInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    hf_qwen2_vl_processor = AutoProcessor.from_pretrained(model_path)
    model = NeuronQwen2VLForCausalLM(model_path=model_path, config=config)
    
    if not os.path.exists(traced_model_path):
        # Compile and save model.
        print("\nCompiling and saving model...")
        model.compile(traced_model_path)
        hf_qwen2_vl_processor.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model.load(traced_model_path)
    hf_qwen2_vl_processor = AutoProcessor.from_pretrained(traced_model_path)

    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig(do_sample=False,
                                         bos_token_id = 151643,
                                         eos_token_id = [151645],
                                         pad_token_id=151645, 
                                         output_logits=True)
    
    # Test Text-Only inputs
    text_prompt="what is the recipe of mayonnaise in two sentences?"
    image_paths=None
    role='user'

    inputs = prepare_generation_inputs_hf(text_prompt, image_paths, hf_qwen2_vl_processor, role)
    sampling_params = prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0],  temperature=[1.0])
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        sampling_params=sampling_params,
        generation_config=generation_config,
        max_new_tokens=128,
    )
    output_tokens = hf_qwen2_vl_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Generated outputs shape: {outputs.shape}")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    # Test Text + Image inputs
    text_prompt="What do you see in these images?"
    role='user'
    image_paths = ["./car.jpg"] * NUM_OF_IMAGES
    inputs = prepare_generation_inputs_hf(text_prompt, image_paths, hf_qwen2_vl_processor, role)

    sampling_params = prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0],  temperature=[1.0])

    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values = inputs.pixel_values,
        image_grid_thw = inputs.image_grid_thw,
        sampling_params=sampling_params,
        generation_config=generation_config,
        max_new_tokens=128,
    )
    output_tokens = hf_qwen2_vl_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Generated outputs shape: {outputs.shape}")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    # Benchmark lantecy
    # TODO: add support in benchmark_sampling to support Qwen2-VL, 
    # Currently Qwen2-VL only supports fixed image input and it is not in the config.
    # Need to fix after dynamic image input padding
    neuron_latency_collector = LatencyCollector()
    for i in range(20):
        neuron_latency_collector.pre_hook()
        outputs = generation_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values = inputs.pixel_values,
            image_grid_thw = inputs.image_grid_thw,
            sampling_params=sampling_params,
            generation_config=generation_config,
            max_new_tokens=1,
        )
        output_tokens = hf_qwen2_vl_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        neuron_latency_collector.hook()
    # Benchmark report
    for p in [25, 50, 90, 99]:
        latency = np.percentile(neuron_latency_collector.latency_list, p) * 1000
        print(f"Neuron inference latency_ms_p{p}: {latency}")

if __name__ == "__main__":
    generate_image_to_text()