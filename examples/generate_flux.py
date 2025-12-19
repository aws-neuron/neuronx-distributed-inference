import os
import argparse
import time
import torch
from neuronx_distributed_inference.models.diffusers.flux.application import NeuronFluxApplication
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.diffusers.flux.clip.modeling_clip import CLIPInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import T5InferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import FluxBackboneInferenceConfig
from neuronx_distributed_inference.models.diffusers.flux.vae.modeling_vae import VAEDecoderInferenceConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)

# Existing Compiled working directory for the compiler
BASE_COMPILE_WORK_DIR = "/tmp/flux/compiler_workdir/"


def create_flux_config(model_path, world_size, backbone_tp_degree, dtype, height, width):
    text_encoder_path = os.path.join(model_path, "text_encoder")
    text_encoder_2_path = os.path.join(model_path, "text_encoder_2")
    backbone_path = os.path.join(model_path, "transformer")
    vae_decoder_path = os.path.join(model_path, "vae")

    clip_neuron_config = NeuronConfig(
        tp_degree=1,
        world_size=world_size,
        torch_dtype=dtype,
    )
    clip_config = CLIPInferenceConfig(
        neuron_config=clip_neuron_config,
        load_config=load_pretrained_config(text_encoder_path),
    )

    t5_neuron_config = NeuronConfig(
        tp_degree = world_size,     # T5: TP degree = world_size
        world_size = world_size,
        torch_dtype=dtype
    )
    t5_config = T5InferenceConfig(
        neuron_config=t5_neuron_config,
        load_config=load_pretrained_config(text_encoder_2_path),
    )

    backbone_neuron_config = NeuronConfig(
        tp_degree = backbone_tp_degree,
        world_size = world_size,
        torch_type = dtype
    )
    backbone_config = FluxBackboneInferenceConfig(
        neuron_config = backbone_neuron_config,
        load_config = load_diffusers_config(backbone_path),
        height = height,
        width = width,
    )

    decoder_neuron_config = NeuronConfig(
        tp_degree = 1,
        world_size = world_size,
        torch_type = dtype
    )
    decoder_config = VAEDecoderInferenceConfig(
        neuron_config = decoder_neuron_config,
        load_config = load_diffusers_config(vae_decoder_path),
        height = height,
        width = width,
        transformer_in_channels = backbone_config.in_channels,
    )

    setattr(backbone_config, "vae_scale_factor", decoder_config.vae_scale_factor)

    return (clip_config, t5_config, backbone_config, decoder_config)

def run_flux_generate(args):
    print(f"run_flux_generate with args: {args}")
    world_size = 8
    backbone_tp_degree = 8
    if args.instance_type == "trn1":
        if args.context_parallel_enabled:
            world_size = 16
            backbone_tp_degree = 8
        else:
            world_size = 8
            backbone_tp_degree = 8
    elif args.instance_type == "trn2":
        if args.context_parallel_enabled:
            world_size = 8
            backbone_tp_degree = 4
        else:
            world_size = 4
            backbone_tp_degree = 4

    dtype = torch.bfloat16

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(args.checkpoint_dir, world_size, backbone_tp_degree, dtype, args.height, args.width)

    flux_app = NeuronFluxApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config = clip_config,
        text_encoder2_config = t5_config,
        backbone_config = backbone_config,
        decoder_config = decoder_config,
        instance_type = args.instance_type,
        height = args.height,
        width = args.width,
    )
    flux_app.compile(BASE_COMPILE_WORK_DIR)
    flux_app.load(BASE_COMPILE_WORK_DIR)

    warmup_rounds = 5
    print("Warming up the model for better latency testing")
    for i in range(warmup_rounds):
        flux_app(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]


    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
            _run_flux_helper(flux_app, args)

        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace(f"{args.profile_name}")
    else:
        _run_flux_helper(flux_app, args)


def _run_flux_helper(flux_app, args):
    total_time = 0
    for i in range(args.num_images):
        start_time = time.time()

        image = flux_app(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]

        end_time = time.time()
        generation_time = end_time - start_time
        total_time += generation_time

        if args.save_image:
            filename = f"output_{i+1}.png"
            image.save(filename)

        print(f"Image {i+1} generated in {generation_time:.2f} seconds")

    average_time = total_time / args.num_images
    print(f"\nAverage generation time: {average_time:.2f} seconds")


if __name__ == "__main__":
    # The Ckpt directory root under huggingface
    CKPT_DIR = "/shared/flux/FLUX.1-dev/"

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=25)
    parser.add_argument("-i", "--instance_type", type=str, default="trn2")
    parser.add_argument("-g", "--guidance_scale", type=float, default=3.5)
    parser.add_argument("-c", "--checkpoint_dir", type=str, default=CKPT_DIR)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_name", type=str, default="flux_torch_profile.json")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--context_parallel_enabled", action="store_true", default=True)

    args = parser.parse_args()
    run_flux_generate(args)
