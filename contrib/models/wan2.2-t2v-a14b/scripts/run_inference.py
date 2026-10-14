"""
All-CP pipeline: both T1 and T2 on CP=4 (split) for maximum precision.

Architecture:
  - T1 first-half: CP=4 on cores 0-3
  - T1 second-half: CP=4 on cores 4-7
  - T2 first-half: CP=4 on cores 8-11
  - T2 second-half: CP=4 on cores 12-15
  - Main process: orchestrates text encoding, scheduler, VAE, IPC

Run: python 26_generate_allcp.py
"""
import os, sys, time, torch, subprocess, shutil, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CACHE_DIR = os.environ.get("CACHE_DIR", "/mnt/work/.cache")
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/work/.cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/work/wan2.2-lint-fix/neuron_output_allcp")
WORK_DIR = os.environ.get("WORK_DIR", "/dev/shm/wan_workers")
os.makedirs(f"{OUTPUT_DIR}/frames", exist_ok=True)

SEED = 42
PROMPT = "A cat walking on a beach at sunset"
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
BOUNDARY_TIMESTEP = 875.0


# ---- IPC helpers ----
def call_worker(worker_type, request_data, timeout=60):
    """Send a request to a worker and wait for its response synchronously."""
    wdir = os.path.join(WORK_DIR, worker_type)
    torch.save(request_data, os.path.join(wdir, "_request.pt"))
    open(os.path.join(wdir, "_request_ready"), "w").close()
    t0 = time.time()
    while not os.path.exists(os.path.join(wdir, "_response_ready")):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Worker {worker_type} timed out after {timeout}s")
        time.sleep(0.002)
    response = torch.load(os.path.join(wdir, "_response.pt"), weights_only=False)
    os.remove(os.path.join(wdir, "_response_ready"))
    return response


def send_request(worker_type, request_data):
    """Send a request to a worker without waiting for a response."""
    wdir = os.path.join(WORK_DIR, worker_type)
    torch.save(request_data, os.path.join(wdir, "_request.pt"))
    open(os.path.join(wdir, "_request_ready"), "w").close()


def wait_response(worker_type, timeout=60):
    """Wait for and return a worker's response."""
    wdir = os.path.join(WORK_DIR, worker_type)
    t0 = time.time()
    while not os.path.exists(os.path.join(wdir, "_response_ready")):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Worker {worker_type} timed out after {timeout}s")
        time.sleep(0.002)
    response = torch.load(os.path.join(wdir, "_response.pt"), weights_only=False)
    os.remove(os.path.join(wdir, "_response_ready"))
    return response


def wait_for_worker(worker_type, proc, timeout=180):
    """Block until a worker signals readiness or raise on failure."""
    wdir = os.path.join(WORK_DIR, worker_type)
    t0 = time.time()
    while not os.path.exists(os.path.join(wdir, "_worker_ready")):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Worker {worker_type} not ready after {timeout}s")
        if proc.poll() is not None:
            log_path = f"{OUTPUT_DIR}/{worker_type}_worker.log"
            out = open(log_path).read()[-3000:] if os.path.exists(log_path) else ""
            raise RuntimeError(f"Worker {worker_type} died (exit={proc.returncode}).\n{out}")
        time.sleep(1)
    print(f"  {worker_type} ready ({time.time()-t0:.0f}s)")


def shutdown_workers(workers):
    """Signal all workers to shut down and terminate their processes."""
    for name, _ in workers:
        wdir = os.path.join(WORK_DIR, name)
        os.makedirs(wdir, exist_ok=True)
        open(os.path.join(wdir, "_shutdown"), "w").close()
    time.sleep(2)
    for _, p in workers:
        p.terminate()
    time.sleep(1)
    for _, p in workers:
        p.kill()


def launch_worker(name, worker_type, cores, subfolder, compiled_path, env_base, workers):
    """Launch a subprocess worker with the given configuration."""
    env = env_base.copy()
    env["WORKER_TYPE"] = worker_type
    env["WORKER_NAME"] = name
    env["NEURON_RT_VISIBLE_CORES"] = cores
    env["SUBFOLDER"] = subfolder
    env["COMPILED_PATH"] = compiled_path
    log = open(f"{OUTPUT_DIR}/{name}_worker.log", "w")
    p = subprocess.Popen(["python", "worker.py"], env=env,
                         cwd=os.environ.get("PROJECT_DIR", "/mnt/work/wan2.2-lint-fix"),
                         stdout=log, stderr=subprocess.STDOUT)
    workers.append((name, p))
    return p


def compute_wan_rope(num_frames=13, height=480, width=832,
                     patch_size=(1,2,2), attention_head_dim=128,
                     max_seq_len=1024, theta=10000.0):
    """Compute rotary position embeddings for the Wan model."""
    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    p_t, p_h, p_w = patch_size
    vae_t, vae_s = 4, 8
    ppf = ((num_frames - 1) // vae_t + 1) // p_t
    pph = (height // vae_s) // p_h
    ppw = (width // vae_s) // p_w
    h_dim = w_dim = 2 * (attention_head_dim // 6)
    t_dim = attention_head_dim - h_dim - w_dim
    fc_list, fs_list = [], []
    for dim in [t_dim, h_dim, w_dim]:
        fc, fs = get_1d_rotary_pos_embed(dim, max_seq_len, theta,
            use_real=True, repeat_interleave_real=True, freqs_dtype=torch.float64)
        fc_list.append(fc); fs_list.append(fs)
    fc_full = torch.cat(fc_list, dim=1)
    fs_full = torch.cat(fs_list, dim=1)
    splits = [t_dim, h_dim, w_dim]
    fc = fc_full.split(splits, dim=1)
    fs = fs_full.split(splits, dim=1)
    cos_out = torch.cat([fc[0][:ppf].view(ppf,1,1,-1).expand(ppf,pph,ppw,-1),
                         fc[1][:pph].view(1,pph,1,-1).expand(ppf,pph,ppw,-1),
                         fc[2][:ppw].view(1,1,ppw,-1).expand(ppf,pph,ppw,-1)],
                        dim=-1).reshape(1, ppf*pph*ppw, 1, -1)
    sin_out = torch.cat([fs[0][:ppf].view(ppf,1,1,-1).expand(ppf,pph,ppw,-1),
                         fs[1][:pph].view(1,pph,1,-1).expand(ppf,pph,ppw,-1),
                         fs[2][:ppw].view(1,1,ppw,-1).expand(ppf,pph,ppw,-1)],
                        dim=-1).reshape(1, ppf*pph*ppw, 1, -1)
    return cos_out.bfloat16(), sin_out.bfloat16()


def run_cfg_pipelined(first_worker, second_worker, latent_input, timestep_val,
                      prompt_emb, negative_emb, rope_cos, rope_sin):
    """Run pipelined CFG: overlap second(cond) with first(uncond)."""
    ts = torch.tensor([timestep_val], dtype=torch.bfloat16)
    req = lambda enc: {"hs": latent_input, "ts": ts, "enc": enc,
                       "rc": rope_cos, "rs": rope_sin}

    # 1. first_half(cond)
    cond_first = call_worker(first_worker, req(prompt_emb))

    # 2. second_half(cond) || first_half(uncond)
    send_request(second_worker, {
        "hs": cond_first["out_0"], "temb": cond_first["out_1"],
        "ts_proj": cond_first["out_2"], "enc_proj": cond_first["out_3"],
        "rc": rope_cos, "rs": rope_sin,
    })
    send_request(first_worker, req(negative_emb))
    cond_second = wait_response(second_worker)
    uncond_first = wait_response(first_worker)

    # 3. second_half(uncond)
    uncond_second = call_worker(second_worker, {
        "hs": uncond_first["out_0"], "temb": uncond_first["out_1"],
        "ts_proj": uncond_first["out_2"], "enc_proj": uncond_first["out_3"],
        "rc": rope_cos, "rs": rope_sin,
    })

    return cond_second["output"], uncond_second["output"]


def main():
    """Main entry point for the All-CP inference pipeline."""
    global workers

    print("=" * 80)
    print("All-CP Pipeline: T1 CP=4 + T2 CP=4 (persistent workers)")
    print("=" * 80)

    # ---- Launch 4 CP workers ----
    print("\n[1/7] Launching workers...")
    sys.stdout.flush()

    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    env_base = os.environ.copy()
    env_base["WORK_DIR"] = WORK_DIR
    project_dir = os.environ.get("PROJECT_DIR", "/mnt/work/wan2.2-lint-fix")
    env_base["PYTHONPATH"] = project_dir + ":" + env_base.get("PYTHONPATH", "")

    workers = []

    # T1 CP workers (transformer)
    p1 = launch_worker("t1_first", "cp_first", "0-3", "transformer",
                        os.environ.get("COMPILED_CP_T1_FIRST", "/mnt/work/wan2.2-lint-fix/compiled_cp_transformer_first"),
                        env_base, workers)
    p2 = launch_worker("t1_second", "cp_second", "4-7", "transformer",
                        os.environ.get("COMPILED_CP_T1_SECOND", "/mnt/work/wan2.2-lint-fix/compiled_cp_transformer_second"),
                        env_base, workers)
    # T2 CP workers (transformer_2)
    p3 = launch_worker("t2_first", "cp_first", "8-11", "transformer_2",
                        os.environ.get("COMPILED_CP_T2_FIRST", "/mnt/work/wan2.2-lint-fix/compiled_cp_transformer_2_first"),
                        env_base, workers)
    p4 = launch_worker("t2_second", "cp_second", "12-15", "transformer_2",
                        os.environ.get("COMPILED_CP_T2_SECOND", "/mnt/work/wan2.2-lint-fix/compiled_cp_transformer_2_second"),
                        env_base, workers)

    print("  Waiting for workers to load NEFFs...")
    sys.stdout.flush()
    try:
        wait_for_worker("t1_first", p1)
        wait_for_worker("t1_second", p2)
        wait_for_worker("t2_first", p3)
        wait_for_worker("t2_second", p4)
    except (TimeoutError, RuntimeError) as e:
        print(f"\n  FATAL: {e}")
        shutdown_workers(workers)
        sys.exit(1)

    print("  All 4 workers ready!")
    sys.stdout.flush()

    # ---- Text encoding (Neuron, subprocess) ----
    print("\n[2/7] Text encoding (Neuron TP=2)...")
    sys.stdout.flush()

    te_script = f'''
import torch, os, sys
sys.path.insert(0, '{project_dir}')
os.environ["NEURON_RT_VISIBLE_CORES"] = "16-17"
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.diffusers.flux.t5.modeling_t5 import T5InferenceConfig
from neuronx_distributed_inference.utils.diffusers_adapter import load_diffusers_config
from nxdi_wan.application_umt5 import NeuronUMT5Application
from transformers import AutoTokenizer
TE_PATH = "{MODEL_PATH}/text_encoder"
nc = NeuronConfig(tp_degree=2, world_size=2, torch_dtype=torch.bfloat16, batch_size=1)
config = T5InferenceConfig(neuron_config=nc, load_config=load_diffusers_config(TE_PATH), max_length=512)
config.is_decoder = False; config.use_cache = False; config.is_encoder_decoder = False
config.output_attentions = False; config.output_hidden_states = False
app = NeuronUMT5Application(model_path=TE_PATH, config=config)
app.load("{project_dir}/compiled_umt5_final")
tok = AutoTokenizer.from_pretrained("{MODEL_ID}", subfolder="tokenizer", cache_dir="{CACHE_DIR}")
def encode(text):
    tokens = tok(text, padding="max_length", max_length=512, truncation=True,
                 add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    seq_len = int(tokens.attention_mask.sum().item())
    embeds = app(tokens.input_ids, tokens.attention_mask).to(torch.bfloat16)
    embeds[0, seq_len:] = 0
    return embeds
torch.save({{"prompt": encode("{PROMPT}"), "negative": encode("")}}, "/dev/shm/wan_te_embeds.pt")
print("DONE")
'''
    te_script_path = os.path.join(os.environ.get('TMPDIR', '/tmp'), '_run_te.py')
    with open(te_script_path, 'w') as f:
        f.write(te_script)

    t0 = time.time()
    te_log = open(f"{OUTPUT_DIR}/te_worker.log", "w")
    te_proc = subprocess.run(["python", te_script_path], cwd=project_dir,
                             stdout=te_log, stderr=subprocess.STDOUT, timeout=120)
    te_log.close()
    if te_proc.returncode != 0:
        print(f"  FATAL: Text encoder failed. See {OUTPUT_DIR}/te_worker.log")
        sys.exit(1)
    embeds = torch.load("/dev/shm/wan_te_embeds.pt", weights_only=False)
    prompt_embeds = embeds["prompt"]
    negative_embeds = embeds["negative"]
    print(f"  Done in {time.time()-t0:.0f}s (prompt std={prompt_embeds.float().std():.4f})")

    # ---- RoPE ----
    print("\n[3/7] RoPE...")
    rope_cos, rope_sin = compute_wan_rope()
    print(f"  RoPE computed: cos={rope_cos.shape}, sin={rope_sin.shape}")

    # ---- Latents ----
    print("\n[4/7] Latents...")
    from diffusers import UniPCMultistepScheduler, AutoencoderKLWan
    gen = torch.Generator("cpu").manual_seed(SEED)
    latents = torch.randn(1, 16, 4, 60, 104, generator=gen, dtype=torch.float32)
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", cache_dir=CACHE_DIR)
    scheduler.set_timesteps(NUM_STEPS)

    # ---- Warmup ----
    print("\n[5/7] Warmup...")
    sys.stdout.flush()
    dummy = torch.randn(1, 16, 4, 60, 104, dtype=torch.bfloat16)
    dummy_enc = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    try:
        t0 = time.time()
        run_cfg_pipelined("t1_first", "t1_second", dummy, 999.0, dummy_enc, dummy_enc, rope_cos, rope_sin)
        print(f"  T1 warmup: {time.time()-t0:.1f}s")
        t0 = time.time()
        run_cfg_pipelined("t2_first", "t2_second", dummy, 500.0, dummy_enc, dummy_enc, rope_cos, rope_sin)
        print(f"  T2 warmup: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  Warmup failed: {e}")
        import traceback; traceback.print_exc()
        shutdown_workers(workers)
        sys.exit(1)
    del dummy, dummy_enc
    sys.stdout.flush()

    # ---- Denoising ----
    print(f"\n[6/7] Denoising ({NUM_STEPS} steps)...")
    sys.stdout.flush()
    t_total = time.time()
    t1_time = 0
    t2_time = 0

    try:
        for step_idx, t in enumerate(scheduler.timesteps):
            t0 = time.time()
            latent_input = latents.to(torch.bfloat16)

            with torch.no_grad():
                if t >= BOUNDARY_TIMESTEP:
                    cond, uncond = run_cfg_pipelined("t1_first", "t1_second",
                        latent_input, float(t), prompt_embeds, negative_embeds, rope_cos, rope_sin)
                    t1_time += time.time() - t0
                else:
                    cond, uncond = run_cfg_pipelined("t2_first", "t2_second",
                        latent_input, float(t), prompt_embeds, negative_embeds, rope_cos, rope_sin)
                    t2_time += time.time() - t0

            noise_pred = uncond.float() + GUIDANCE_SCALE * (cond.float() - uncond.float())
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            elapsed = time.time() - t0
            stage = "T1-CP4" if t >= BOUNDARY_TIMESTEP else "T2-CP4"
            if (step_idx + 1) % 5 == 0 or step_idx == 0 or step_idx == len(scheduler.timesteps) - 1:
                print(f"  Step {step_idx+1}/{NUM_STEPS} ({stage}, t={t:.0f}): "
                      f"std={latents.std():.4f}, time={elapsed:.1f}s")
                sys.stdout.flush()
    except Exception as e:
        print(f"\n  Denoising failed at step {step_idx+1}: {e}")
        import traceback; traceback.print_exc()
        shutdown_workers(workers)
        sys.exit(1)

    total_time = time.time() - t_total
    t1_steps = sum(1 for t in scheduler.timesteps if t >= BOUNDARY_TIMESTEP)
    t2_steps = NUM_STEPS - t1_steps
    print(f"\n  T1 time: {t1_time:.0f}s ({t1_time/max(t1_steps,1):.1f}s/step, {t1_steps} steps)")
    print(f"  T2 time: {t2_time:.0f}s ({t2_time/max(t2_steps,1):.1f}s/step, {t2_steps} steps)")
    print(f"  Total denoising: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ---- Shutdown workers ----
    print("\n  Shutting down workers...")
    shutdown_workers(workers)

    # ---- VAE (hybrid: frames 0+1 CPU, frames 2+3 Neuron) ----
    print("\n[7/7] VAE decode (hybrid Neuron)...")
    sys.stdout.flush()
    t_vae = time.time()
    torch.save(latents, "/dev/shm/wan_vae_latents.pt")
    vae_log = open(f"{OUTPUT_DIR}/vae_worker.log", "w")
    vae_env = os.environ.copy()
    vae_env["NEURON_RT_VISIBLE_CORES"] = "20-21"
    vae_proc = subprocess.run(
        ["python", "vae_decode_hybrid.py", "/dev/shm/wan_vae_latents.pt", "/dev/shm/wan_vae_out.pt"],
        env=vae_env, cwd=project_dir,
        stdout=vae_log, stderr=subprocess.STDOUT, timeout=120)
    vae_log.close()
    if vae_proc.returncode != 0:
        print(f"  VAE failed! See {OUTPUT_DIR}/vae_worker.log")
        # Fallback to CPU
        from diffusers import AutoencoderKLWan as VAECLS
        vae = VAECLS.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR)
        vae.eval()
        lv = latents.to(vae.dtype)
        lm = torch.tensor(vae.config.latents_mean).view(1,16,1,1,1).to(lv.dtype)
        ls = 1.0/torch.tensor(vae.config.latents_std).view(1,16,1,1,1).to(lv.dtype)
        with torch.no_grad():
            video = vae.decode(lv/ls+lm, return_dict=False)[0]
    else:
        video = torch.load("/dev/shm/wan_vae_out.pt", weights_only=True)
    t_vae = time.time() - t_vae
    print(f"  VAE decode: {t_vae:.0f}s")

    from diffusers.video_processor import VideoProcessor
    video_pt = VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="pt")
    torch.save(video_pt, f"{OUTPUT_DIR}/video_tensor.pt")

    from PIL import Image
    ref_path = os.environ.get("REFERENCE_PATH", "/mnt/work/wan2.2-lint-fix/reference/reference_frames.pt")
    ref = torch.load(ref_path, weights_only=True)
    for fi in range(min(video_pt.shape[1], 13)):
        nf = (video_pt[0, fi].float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        rf = (ref[0, fi].float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(nf).save(f"{OUTPUT_DIR}/frames/neuron_{fi:02d}.png")
        Image.fromarray(rf).save(f"{OUTPUT_DIR}/frames/ref_{fi:02d}.png")
        Image.fromarray(np.concatenate([rf, nf], axis=1)).save(f"{OUTPUT_DIR}/frames/compare_{fi:02d}.png")

    cos = torch.nn.functional.cosine_similarity(
        video_pt.flatten().float().unsqueeze(0), ref.flatten().float().unsqueeze(0)).item()

    print(f"\n{'='*80}")
    print(f"RESULTS (All-CP: T1 CP=4 + T2 CP=4)")
    print(f"  Cosine vs reference: {cos:.6f}  {'PASS' if cos > 0.98 else 'FAIL'} (target > 0.98)")
    print(f"  Std: {video_pt.float().std():.4f} (ref: {ref.float().std():.4f})")
    print(f"  T1 time: {t1_time:.0f}s, T2 time: {t2_time:.0f}s")
    print(f"  Total denoising: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
