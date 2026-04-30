#!/usr/bin/env python3
"""
Example script demonstrating tensor capture functionality with Llama4 multimodal models.

This script shows how to:
1. Configure tensor capture in a Llama4 model's neuron_config
2. Create a tensor capture hook to save tensors during generation
3. Analyze captured tensors from the text model component

NOTE: Currently, tensor capture for multimodal models only captures intermediates 
from the text model component, not from the vision model component.

Usage:
  # Compile a model with tensor capture enabled
  python multimodal_tensor_capture_example.py compile --model-path /path/to/model --output-dir ./compiled_model

  # Run inference with tensor capture
  python multimodal_tensor_capture_example.py run --compiled-model-dir ./compiled_model --output-dir ./captured_tensors --image-path image.jpg

  # Analyze captured tensors
  python multimodal_tensor_capture_example.py analyze --tensor-dir ./captured_tensors
"""

import argparse
import json
import os
import time
import copy
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4 import Llama4InferenceConfig, NeuronLlama4ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.tensor_capture_utils import (
    get_tensor_capture_hook,
    analyze_captured_tensors,
    list_capturable_modules_in_application,
)


def create_neuron_config(
    tp_degree: int = 64,
    batch_size: int = 1,
    modules_to_capture: Optional[List[str]] = None,
    capture_inputs: bool = False,
    max_intermediate_tensors: Optional[int] = None,
) -> NeuronConfig:
    """Create a NeuronConfig with tensor capture enabled."""
    # Create tensor capture config if any tensor capture options are specified
    tensor_capture_config = None
    if modules_to_capture or max_intermediate_tensors is not None:
        tensor_capture_config = TensorCaptureConfig(
            modules_to_capture=modules_to_capture or [],
            capture_inputs=capture_inputs,
            max_intermediate_tensors=max_intermediate_tensors,
        )

    # Create neuron config with tensor capture
    return NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_context_length=128,
        seq_len=256,
        output_logits=True,
        on_device_sampling_config={"do_sample": False},
        tensor_capture_config=tensor_capture_config,
    )


def compile_model(args):
    """Compile a model with tensor capture enabled."""
    print(f"Compiling model from {args.model_path} with tensor capture enabled...")
    
    # Create neuron config with tensor capture
    neuron_config = create_neuron_config(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        modules_to_capture=args.modules_to_capture,
        capture_inputs=args.capture_inputs,
        max_intermediate_tensors=args.max_intermediate_tensors,
    )
    
    # Create model config
    config = Llama4InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(args.model_path),
    )
    
    # Initialize model
    model = NeuronLlama4ForCausalLM(args.model_path, config)
    
    # Compile model
    os.makedirs(args.output_dir, exist_ok=True)
    model.compile(args.output_dir)
    
    print(f"Model compiled successfully to {args.output_dir}")
    
    # Save the tensor capture configuration for reference
    config_file = os.path.join(args.output_dir, "tensor_capture_config.json")
    with open(config_file, "w") as f:
        json.dump({
            "modules_to_capture": args.modules_to_capture,
            "capture_inputs": args.capture_inputs,
            "max_intermediate_tensors": args.max_intermediate_tensors,
        }, f, indent=2)
    
    print(f"Tensor capture configuration saved to {config_file}")


def run_inference(args):
    """Run inference with tensor capture enabled."""
    print(f"Running inference with tensor capture on model in {args.compiled_model_dir}...")
    
    # Load the tensor capture configuration
    config_file = os.path.join(args.compiled_model_dir, "tensor_capture_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            tc_config = json.load(f)
        print(f"Loaded tensor capture configuration: {tc_config}")
    else:
        print("No tensor capture configuration found, using default values")
        tc_config = {
            "modules_to_capture": [],
            "capture_inputs": False,
            "max_intermediate_tensors": None,
        }
    
    # Create neuron config with tensor capture
    neuron_config = create_neuron_config(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        modules_to_capture=tc_config.get("modules_to_capture", []),
        capture_inputs=tc_config.get("capture_inputs", False),
        max_intermediate_tensors=tc_config.get("max_intermediate_tensors"),
    )
    
    # Create model config
    config = Llama4InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(args.model_path),
    )
    
    # Initialize model
    model = NeuronLlama4ForCausalLM(args.model_path, config)
    
    # Load compiled model
    model.load(args.compiled_model_dir)
    
    # Create processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Load and process image if provided
    inputs = {}
    if args.image_path and os.path.exists(args.image_path):
        print(f"Processing image: {args.image_path}")
        image = Image.open(args.image_path)
        inputs = processor(text=args.prompt, images=image, return_tensors="pt")
    else:
        print("No image provided or image not found. Running text-only inference.")
        inputs = processor(text=args.prompt, return_tensors="pt")
    
    # Convert inputs to int32
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].to(torch.int32)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.int32)
    
    # Create tensor capture hook
    os.makedirs(args.output_dir, exist_ok=True)
    tensor_capture_hook = get_tensor_capture_hook(
        capture_indices=args.capture_indices,
        tensor_capture_save_dir=args.output_dir,
    )
    
    # Create generation adapter
    generation_model = HuggingFaceGenerationAdapter(model)
    
    # Run generation with tensor capture
    print("Generating text with tensor capture...")
    start_time = time.time()
    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        tensor_capture_hook=tensor_capture_hook,
    )
    generation_time = time.time() - start_time
    
    # Decode output
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated text ({generation_time:.2f}s):")
    print(f"{output_text}")
    
    print(f"\nTensor capture complete. Tensors saved to {args.output_dir}")


def analyze_tensors(args):
    """Analyze captured tensors."""
    print(f"Analyzing tensors in {args.tensor_dir}...")
    
    # Analyze tensors
    results = analyze_captured_tensors(
        tensor_dir=args.tensor_dir,
        reference_dir=args.reference_dir,
    )
    
    # Print summary
    print("\nSummary:")
    print(f"Total tensors captured: {results['summary']['total_tensors']}")
    print(f"Modules captured: {results['summary']['modules_captured']}")
    print(f"Steps captured: {results['summary']['steps_captured']}")
    print(f"Phases: {results['summary']['phases']}")
    
    # Print tensor details
    if args.verbose:
        print("\nTensor details:")
        for tensor_info in results['tensors']:
            print(f"\nTensor: {tensor_info['name']}")
            print(f"  Shape: {tensor_info['shape']}")
            print(f"  Min/Max: {tensor_info['min']:.6f}/{tensor_info['max']:.6f}")
            print(f"  Mean/Std: {tensor_info['mean']:.6f}/{tensor_info['std']:.6f}")
            
            # If comparison with reference is available
            if 'ref_comparison' in tensor_info:
                print(f"  Max abs diff: {tensor_info['ref_comparison']['max_abs_diff']:.6f}")
                print(f"  Mean abs diff: {tensor_info['ref_comparison']['mean_abs_diff']:.6f}")
                print(f"  Relative diff: {tensor_info['ref_comparison']['rel_diff']:.6f}")
    
    # Save results to file
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis results saved to {args.output_file}")


def list_modules(args):
    """List capturable modules in a model."""
    print(f"Listing capturable modules in {args.model_path}...")
    
    # Create neuron config
    neuron_config = create_neuron_config(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
    )
    
    # Create model config
    config = Llama4InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(args.model_path),
    )
    
    # Initialize model
    model = NeuronLlama4ForCausalLM(args.model_path, config)
    
    # List capturable modules
    modules = list_capturable_modules_in_application(model)
    
    # Print available modules for each model component
    for model_name, available_modules in modules.items():
        print(f"\nAvailable modules in {model_name}:")
        for module_name, module_type in available_modules.items():
            print(f"  {module_name}: {module_type}")
    
    # Save results to file
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({k: list(v.keys()) for k, v in modules.items()}, f, indent=2)
        print(f"\nModule list saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Tensor capture example for Llama4 multimodal models in NeuronxDistributedInference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile a model with tensor capture enabled")
    compile_parser.add_argument("--model-path", type=str, default="/path/to/model", help="Path to the model")
    compile_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for compiled model")
    compile_parser.add_argument("--tp-degree", type=int, default=64, help="Tensor parallelism degree")
    compile_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    compile_parser.add_argument("--modules-to-capture", type=str, nargs="+", default=["layers.0", "layers.1.feed_forward.moe.router", "layers.3"], 
                               help="List of text module names to capture")
    compile_parser.add_argument("--capture-inputs", action="store_true", help="Whether to capture input tensors")
    compile_parser.add_argument("--max-intermediate-tensors", type=int, default=None, 
                               help="Maximum number of intermediate tensors to capture")
    
    # Run inference command
    run_parser = subparsers.add_parser("run", help="Run inference with tensor capture")
    run_parser.add_argument("--compiled-model-dir", type=str, required=True, help="Directory with compiled model")
    run_parser.add_argument("--model-path", type=str, default="/path/to/model", help="Path to model for processor")
    run_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for captured tensors")
    run_parser.add_argument("--tp-degree", type=int, default=64, help="Tensor parallelism degree")
    run_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    run_parser.add_argument("--image-path", type=str, default=None, help="Path to input image (optional)")
    run_parser.add_argument("--prompt", type=str, default="Describe this image:", help="Prompt for generation")
    run_parser.add_argument("--max-new-tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    run_parser.add_argument("--capture-indices", type=int, nargs="+", default=[1, 5, 10], 
                           help="Generation step indices to capture tensors at")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze captured tensors")
    analyze_parser.add_argument("--tensor-dir", type=str, required=True, help="Directory with captured tensors")
    analyze_parser.add_argument("--reference-dir", type=str, default=None, help="Directory with reference tensors")
    analyze_parser.add_argument("--output-file", type=str, default=None, help="Output file for analysis results")
    analyze_parser.add_argument("--verbose", action="store_true", help="Print detailed tensor information")
    
    # List modules command
    list_parser = subparsers.add_parser("list", help="List capturable modules in a model")
    list_parser.add_argument("--model-path", type=str, default="/path/to/model", help="Path to the model")
    list_parser.add_argument("--tp-degree", type=int, default=64, help="Tensor parallelism degree")
    list_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    list_parser.add_argument("--output-file", type=str, default=None, help="Output file for module list")
    
    args = parser.parse_args()
    
    if args.command == "compile":
        compile_model(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "analyze":
        analyze_tensors(args)
    elif args.command == "list":
        list_modules(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    main()
