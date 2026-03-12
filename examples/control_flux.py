import argparse
import time

import torch
from diffusers.utils import load_image
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxControlApplication,
    create_flux_config,
    get_flux_parallelism_config,
)
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)

DEFAULT_COMPILE_WORK_DIR = "/tmp/flux_control/compiler_workdir/"
DEFAULT_CKPT_DIR = "/shared/flux/FLUX.1-Depth-dev/"


def run_flux_control(args):
    print(f"run_flux_control with args: {args}")

    control_image = load_image(args.control_image).resize(
        (args.width, args.height)
    )

    world_size, backbone_tp_degree = get_flux_parallelism_config(
        args.instance_type, args.context_parallel_enabled
    )

    dtype = torch.bfloat16

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        args.checkpoint_dir,
        world_size,
        backbone_tp_degree,
        dtype,
        args.height,
        args.width,
        inpaint=True,  # Control uses same VAE config as inpaint
    )

    flux_app = NeuronFluxControlApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=args.height,
        width=args.width,
    )
    flux_app.compile(args.compile_workdir)
    flux_app.load(args.compile_workdir)

    if args.profile:
        from torch.profiler import profile, ProfilerActivity

        warmup_rounds = 5
        print("Warming up the model for better latency testing")
        for _ in range(warmup_rounds):
            flux_app(
                prompt=args.prompt,
                control_image=control_image,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            ).images[0]

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            _run_flux_control_helper(flux_app, args, control_image)

        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
        )
        prof.export_chrome_trace(f"{args.profile_name}")
    else:
        _run_flux_control_helper(flux_app, args, control_image)


def _run_flux_control_helper(flux_app, args, control_image):
    total_time = 0
    for i in range(args.num_images):
        start_time = time.time()

        result_image = flux_app(
            prompt=args.prompt,
            control_image=control_image,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        ).images[0]

        end_time = time.time()
        generation_time = end_time - start_time
        total_time += generation_time

        if args.save_image:
            filename = f"output_{i+1}.png"
            result_image.save(filename)

        print(f"Image {i+1} generated in {generation_time:.2f} seconds")

    average_time = total_time / args.num_images
    print(f"\nAverage generation time: {average_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--prompt", type=str, default="A robot made of exotic candies and chocolates"
    )
    parser.add_argument(
        "--control_image",
        type=str,
        required=True,
        help="Path to the control image (depth map or canny edge)",
    )
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "-i", "--instance_type", type=str, default="trn2", choices=["trn1", "trn2"]
    )
    parser.add_argument("-g", "--guidance_scale", type=float, default=30.0)
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--compile_workdir",
        type=str,
        default=DEFAULT_COMPILE_WORK_DIR,
        help="Path to the compile working directory for compiler artifacts",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_name", type=str, default="flux_control_torch_profile.json")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--context_parallel_enabled", action="store_true")

    args = parser.parse_args()
    run_flux_control(args)
