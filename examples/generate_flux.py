import argparse
import time
import torch
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)

# Default values for compile working directory and checkpoint directory
DEFAULT_COMPILE_WORK_DIR = "/tmp/flux/compiler_workdir/"
DEFAULT_CKPT_DIR = "/shared/flux/FLUX.1-dev/"


def run_flux_generate(args):
    print(f"run_flux_generate with args: {args}")
    
    # Determine backbone_tp_degree: use user-specified value or default based on hardware
    if args.backbone_tp_degree is not None:
        backbone_tp_degree = args.backbone_tp_degree
    else:
        # Default based on instance type
        if args.instance_type == "trn1":
            backbone_tp_degree = 8
        else:
            backbone_tp_degree = 4
    
    world_size = get_flux_parallelism_config(
        backbone_tp_degree,
        context_parallel_enabled=args.context_parallel_enabled,
        cfg_parallel_enabled=args.cfg_parallel_enabled
    )

    dtype = torch.bfloat16

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        args.checkpoint_dir,
        world_size,
        backbone_tp_degree,
        dtype,
        args.height,
        args.width,
        cfg_parallel_enabled=args.cfg_parallel_enabled,
        context_parallel_enabled=args.context_parallel_enabled,
    )
    # backbone_config.num_layers=4
    # backbone_config.num_single_layers=4

    flux_app = NeuronFluxApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config = clip_config,
        text_encoder2_config = t5_config,
        backbone_config = backbone_config,
        decoder_config = decoder_config,
        height = args.height,
        width = args.width,
    )
    # flux_app.compile(args.compile_workdir, debug=True)
    flux_app.compile(args.compile_workdir)
    flux_app.load(args.compile_workdir)

    warmup_rounds = 5
    print("Warming up the model for better latency testing")
    
    # Configure CFG parameters for warmup
    if args.use_cfg:
        warmup_negative_prompt = args.negative_prompt
        warmup_true_cfg_scale = 2.0
    else:
        warmup_negative_prompt = None
        warmup_true_cfg_scale = 1.0
    
    for _ in range(warmup_rounds):
        flux_app(
            args.prompt,
            negative_prompt=warmup_negative_prompt,
            true_cfg_scale=warmup_true_cfg_scale,
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
    
    # Configure CFG parameters if use_cfg is enabled
    if args.use_cfg:
        negative_prompt = args.negative_prompt
        true_cfg_scale = 2.0
    else:
        negative_prompt = None
        true_cfg_scale = 1.0
    
    for i in range(args.num_images):
        start_time = time.time()

        image = flux_app(
            args.prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=25)
    parser.add_argument("-i", "--instance_type", type=str, default="trn2", choices=["trn1", "trn2"])
    parser.add_argument("-g", "--guidance_scale", type=float, default=3.5)
    parser.add_argument("-c", "--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR,
                        help="Path to the model checkpoint directory")
    parser.add_argument("--compile_workdir", type=str, default=DEFAULT_COMPILE_WORK_DIR,
                        help="Path to the compile working directory for compiler artifacts")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_name", type=str, default="flux_torch_profile.json")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--context_parallel_enabled", action="store_true")
    parser.add_argument("--use_cfg", action="store_true",
                        help="Enable CFG inference with negative_prompt and true_cfg_scale=2.0")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt for CFG inference (only used when --use_cfg is enabled)")
    parser.add_argument("--cfg_parallel_enabled", action="store_true",
                        help="Enable CFG parallel processing (can only be true when --use_cfg is enabled)")
    parser.add_argument("--backbone_tp_degree", type=int, default=None,
                        help="Tensor parallelism degree for the backbone model. If not specified, defaults to 8 for trn1 and 4 for others.")

    args = parser.parse_args()
    
    # Validate that cfg_parallel_enabled can only be true when use_cfg is enabled
    if args.cfg_parallel_enabled and not args.use_cfg:
        parser.error("--cfg_parallel_enabled can only be enabled when --use_cfg is enabled")
    run_flux_generate(args)
