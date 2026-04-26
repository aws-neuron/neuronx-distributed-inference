"""
SDK E2E T2V Inference Pipeline with Real Text Encoding.

Full text-to-video pipeline using SDK-traced components:
  1. Qwen2.5-VL LLM text encode (CPU fp16, ~2-5s)
  2. byT5 glyph encode (CPU, ~160ms)
  3. Preprocessor (CPU) -- patch embed, timestep embed, token refiner, RoPE
  4. DiT denoise (Neuron TP=4, 50 Euler steps, ~287ms/step)
  5. Unpatchify + Euler step (CPU per step)
  6. VAE decode (Neuron tiled, ~8.5s)
  7. Save video (MP4 + PNG frames)

Flow matching scheduler: Euler with SD3-style time shift (shift=5.0).

Launch:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    NEURON_RT_NUM_CORES=4 \
    XLA_DISABLE_FUNCTIONALIZATION=1 \
    NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    NEURON_FUSE_SOFTMAX=1 \
    python ./e2e_pipeline.py --steps 50

    # Custom prompt:
    python ./e2e_pipeline.py --steps 50 \
        --prompt "A golden retriever running on a beach at sunset"
"""

import sys
import os
import time
import json
import argparse

# Set env BEFORE imports
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

sys.path.insert(0, os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HUNYUAN_ATTN_MODE"] = "torch"

import torch
import torch.nn.functional as F

MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")


def sd3_time_shift(t, shift=5.0):
    return (shift * t) / (1 + (shift - 1) * t)


def create_timestep_schedule(num_steps, shift=5.0):
    sigmas = torch.linspace(1, 0, num_steps + 1)
    sigmas = sd3_time_shift(sigmas, shift=shift)
    timesteps = (sigmas[:-1] * 1000.0).to(torch.float32)
    return timesteps, sigmas


def euler_step(sample, velocity, sigma_cur, sigma_next):
    dt = sigma_next - sigma_cur
    return sample.float() + velocity.float() * dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean, golden light reflecting on gentle waves, "
        "cinematic quality, 4K, smooth camera pan",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="CFG guidance scale (default 6.0, use 1.0 to disable)",
    )
    parser.add_argument(
        "--batched-cfg",
        action="store_true",
        help="Use B=2 batched DiT for CFG (uncond+cond in single pass, requires dit_tp4_480p_b2)",
    )
    args = parser.parse_args()

    do_cfg = args.guidance_scale > 1.0
    use_batched_cfg = args.batched_cfg and do_cfg

    dtype = torch.bfloat16
    B = 1

    # 480p_5f: VAE ffactor_spatial=16, ffactor_temporal=4
    T_lat, H_lat, W_lat = 2, 30, 53
    IMG_SEQ_LEN = T_lat * H_lat * W_lat  # 3180
    TXT_SEQ_LEN = 320
    PATCH_SIZE = [1, 1, 1]
    OUT_CHANNELS = 32

    print("=" * 60)
    print("HunyuanVideo-1.5 E2E T2V Inference Pipeline")
    print("=" * 60)
    print(f"Video:   {(T_lat - 1) * 4 + 1} frames at {H_lat * 16}x{W_lat * 16}")
    print(f"Steps:   {args.steps}, shift={args.shift}, seed={args.seed}")
    print(
        f"CFG:     {'enabled' if do_cfg else 'disabled'}, guidance_scale={args.guidance_scale}"
        f"{', BATCHED B=2' if use_batched_cfg else ''}"
    )
    print(f"DiT:     {IMG_SEQ_LEN} img + {TXT_SEQ_LEN} txt tokens, TP=4")
    print()

    timings = {}
    t_pipeline_start = time.time()

    # ===== Stage 1: Load all models =====
    print("=" * 60)
    print("Stage 1: Load Models")
    print("=" * 60)

    # --- byT5 encoder (CPU) ---
    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2, ByT5Mapper
    import safetensors.torch as st_module
    import glob

    t0 = time.time()
    byt5_args = dict(
        byT5_google_path=f"{MODELS_DIR}/byt5-small",
        byT5_ckpt_path=f"{MODELS_DIR}/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
        multilingual_prompt_format_color_path=f"{MODELS_DIR}/Glyph-SDXL-v2/assets/color_idx.json",
        multilingual_prompt_format_font_path=f"{MODELS_DIR}/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
        byt5_max_length=256,
    )
    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=torch.device("cpu"))
    byt5_model = byt5_kwargs["byt5_model"].eval()
    byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]

    # Load ByT5Mapper
    dit_ckpt_path = f"{MODELS_DIR}/HunyuanVideo-1.5/transformer/480p_t2v"
    st_files = sorted(glob.glob(f"{dit_ckpt_path}/*.safetensors"))
    byt5_in_state = {}
    for sf in st_files:
        with st_module.safe_open(sf, framework="pt") as f:
            for key in f.keys():
                if key.startswith("byt5_in."):
                    byt5_in_state[key.replace("byt5_in.", "")] = f.get_tensor(key)
    fc1_w = byt5_in_state["fc1.weight"]
    byt5_mapper = ByT5Mapper(
        fc1_w.shape[1],
        byt5_in_state["fc2.weight"].shape[0],
        fc1_w.shape[0],
        byt5_in_state["fc3.weight"].shape[0],
        use_residual=(fc1_w.shape[1] == byt5_in_state["fc2.weight"].shape[0]),
    )
    byt5_mapper.load_state_dict(byt5_in_state)
    byt5_mapper = byt5_mapper.to(dtype).eval()
    byt5_load_time = time.time() - t0
    print(f"byT5 + mapper loaded: {byt5_load_time:.1f}s")
    timings["byt5_load_s"] = round(byt5_load_time, 1)

    # --- Preprocessor (CPU) ---
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
        HunyuanVideo_1_5_DiffusionTransformer,
    )
    from dit_wrapper import HunyuanDiTPreprocessor

    t0 = time.time()
    dit_full = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        f"{MODELS_DIR}/HunyuanVideo-1.5/transformer/480p_t2v",
        torch_dtype=dtype,
    )
    dit_full.set_attn_mode("torch")
    preprocessor = HunyuanDiTPreprocessor(dit_full, device="cpu", dtype=dtype)
    preprocess_load_time = time.time() - t0

    # Free the double_blocks and final_layer (they'll run on Neuron)
    del dit_full.double_blocks
    del dit_full.final_layer
    del dit_full
    import gc

    gc.collect()
    print(f"Preprocessor loaded: {preprocess_load_time:.1f}s")
    timings["preprocess_load_s"] = round(preprocess_load_time, 1)

    # --- DiT TP=4 (Neuron) ---
    import neuronx_distributed
    from safetensors.torch import load_file
    from dit_tp_wrapper import HunyuanDiTTPWrapper, build_weight_map

    t0 = time.time()
    if use_batched_cfg:
        SAVE_DIR = f"{COMPILED_DIR}/dit_tp4_480p_b2"
        print(f"Loading B=2 DiT from {SAVE_DIR}")
    else:
        SAVE_DIR = f"{COMPILED_DIR}/dit_tp4_480p"
    nxd_model = torch.jit.load(os.path.join(SAVE_DIR, "nxd_model.pt"))

    weights_path = os.path.join(SAVE_DIR, "weights")
    sharded_weights = []
    for rank in range(4):
        sharded_weights.append(
            load_file(f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors")
        )

    nxd_model.set_weights(sharded_weights)
    nxd_model.to_neuron()
    del sharded_weights
    dit_load_time = time.time() - t0
    print(f"DiT TP=4 loaded: {dit_load_time:.1f}s")
    timings["dit_load_s"] = round(dit_load_time, 1)

    # ===== Stage 2: Text Encoding (CPU) =====
    print(f"\n{'=' * 60}")
    print("Stage 2: Text Encoding (CPU)")
    print("=" * 60)

    prompt = args.prompt
    print(f"Prompt: {prompt}")

    # --- 2a: LLM text encoding (Qwen2.5-VL language model on CPU, fp16) ---
    from transformers import AutoModel, AutoTokenizer
    from copy import deepcopy

    LLM_PATH = f"{MODELS_DIR}/Qwen2.5-VL-7B-Instruct"
    LLM_MAX_LENGTH = 1000  # HunyuanVideo default
    HIDDEN_STATE_SKIP_LAYER = 2  # Use 3rd-from-last hidden layer

    # Video prompt template (from HunyuanVideo)
    PROMPT_TEMPLATE_VIDEO = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video.",
        },
        {"role": "user", "content": "{}"},
    ]

    # Try to load cached negative embeddings
    cached_neg_path = f"{COMPILED_DIR}/cached_neg_embeddings.pt"
    use_cached_neg = os.path.exists(cached_neg_path) and do_cfg
    if use_cached_neg:
        print(f"\nLoading cached negative embeddings from {cached_neg_path}")
        t0 = time.time()
        neg_cache = torch.load(cached_neg_path, map_location="cpu", weights_only=True)
        neg_text_states = neg_cache["neg_text_states"].to(dtype)
        neg_text_mask = neg_cache["neg_text_mask"].to(dtype)
        neg_byt5_text_states = neg_cache["neg_byt5_states"].to(dtype)
        neg_byt5_mask = neg_cache["neg_byt5_mask"]
        crop_start = neg_cache["crop_start"]
        cache_load_time = time.time() - t0
        print(f"  Loaded in {cache_load_time * 1000:.0f}ms (crop_start={crop_start})")
        print(
            f"  neg_text_states: {list(neg_text_states.shape)}, valid={neg_text_mask.sum().int().item()}"
        )
        timings["neg_cache_ms"] = round(cache_load_time * 1000, 0)
    else:
        if do_cfg:
            print(f"\nNo cached negative embeddings found at {cached_neg_path}")
            print("  Will encode negative prompt through LLM (slower)")

    print(f"\nLoading Qwen2.5-VL-7B language model (fp16, CPU)...")
    t0 = time.time()
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, padding_side="right")
    llm_full = AutoModel.from_pretrained(LLM_PATH, low_cpu_mem_usage=True)
    # Extract just the language model backbone (discard vision encoder)
    llm_model = (
        llm_full.language_model if hasattr(llm_full, "language_model") else llm_full
    )
    llm_model.final_layer_norm = llm_model.norm
    llm_model = llm_model.to(dtype=torch.float16).eval()
    llm_model.requires_grad_(False)
    del llm_full
    gc.collect()
    llm_load_time = time.time() - t0
    print(f"LLM loaded: {llm_load_time:.1f}s")
    timings["llm_load_s"] = round(llm_load_time, 1)

    # Find crop_start if not from cache
    if not use_cached_neg:
        template = deepcopy(PROMPT_TEMPLATE_VIDEO)
        for item in template:
            if isinstance(item, dict) and "content" in item:
                item["content"] = item["content"].format(prompt if prompt else " ")
        temp_tokens = llm_tokenizer.apply_chat_template(
            template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        marker = "<|im_start|>user\n"
        marker_ids = llm_tokenizer(marker, add_special_tokens=False)["input_ids"]
        input_ids_list = temp_tokens["input_ids"][0].tolist()
        crop_start = 0
        for i in range(len(input_ids_list) - len(marker_ids) + 1):
            if input_ids_list[i : i + len(marker_ids)] == marker_ids:
                crop_start = i + len(marker_ids)
                break
    print(f"  crop_start={crop_start} (system/instruction prefix tokens)")

    def encode_llm_prompt(prompt_text, llm_model, llm_tokenizer, crop_start_val):
        """Encode a single prompt through the LLM."""
        tmpl = deepcopy(PROMPT_TEMPLATE_VIDEO)
        for item in tmpl:
            if isinstance(item, dict) and "content" in item:
                item["content"] = item["content"].format(
                    prompt_text if prompt_text else " "
                )
        tokens = llm_tokenizer.apply_chat_template(
            tmpl,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=LLM_MAX_LENGTH + crop_start_val,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            out = llm_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
            hidden = hidden[:, crop_start_val:]
            mask = tokens["attention_mask"][:, crop_start_val:].to(dtype)
        return hidden.to(dtype), mask

    def encode_llm_prompt_batched(
        pos_text, neg_text, llm_model, llm_tokenizer, crop_start_val
    ):
        """Encode positive + negative prompts in a single B=2 forward pass."""
        # Tokenize both prompts
        templates = []
        for p in [pos_text, neg_text]:
            tmpl = deepcopy(PROMPT_TEMPLATE_VIDEO)
            for item in tmpl:
                if isinstance(item, dict) and "content" in item:
                    item["content"] = item["content"].format(p if p else " ")
            templates.append(tmpl)

        # Tokenize each and stack (apply_chat_template doesn't natively support batched lists of messages)
        all_input_ids = []
        all_attn_masks = []
        for tmpl in templates:
            tokens = llm_tokenizer.apply_chat_template(
                tmpl,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                truncation=True,
                max_length=LLM_MAX_LENGTH + crop_start_val,
                padding="max_length",
                return_tensors="pt",
            )
            all_input_ids.append(tokens["input_ids"])  # [1, seq_len]
            all_attn_masks.append(tokens["attention_mask"])  # [1, seq_len]

        # Stack to B=2
        input_ids_b2 = torch.cat(all_input_ids, dim=0)  # [2, seq_len]
        attn_mask_b2 = torch.cat(all_attn_masks, dim=0)  # [2, seq_len]

        with torch.no_grad():
            out = llm_model(
                input_ids=input_ids_b2,
                attention_mask=attn_mask_b2,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[
                -(HIDDEN_STATE_SKIP_LAYER + 1)
            ]  # [2, seq_len, 3584]
            hidden = hidden[:, crop_start_val:]  # [2, 1000, 3584]
            mask = attn_mask_b2[:, crop_start_val:].to(dtype)  # [2, 1000]

        # Split back to B=1 each
        pos_hidden = hidden[0:1].to(dtype)  # [1, 1000, 3584]
        neg_hidden = hidden[1:2].to(dtype)  # [1, 1000, 3584]
        pos_mask = mask[0:1]  # [1, 1000]
        neg_mask = mask[1:2]  # [1, 1000]
        return pos_hidden, pos_mask, neg_hidden, neg_mask

    t_llm = time.time()
    if use_cached_neg or not do_cfg:
        # Single-prompt path: positive only (negative is cached or not needed)
        text_states, text_mask = encode_llm_prompt(
            prompt, llm_model, llm_tokenizer, crop_start
        )
    else:
        # B=2 batched path: encode positive + negative in one forward pass
        negative_prompt = ""
        print("  Encoding positive + negative prompts in single B=2 forward pass...")
        text_states, text_mask, neg_text_states, neg_text_mask = (
            encode_llm_prompt_batched(
                prompt, negative_prompt, llm_model, llm_tokenizer, crop_start
            )
        )
    llm_time = time.time() - t_llm
    batch_label = " (B=2 batched)" if (do_cfg and not use_cached_neg) else ""
    cache_label = " (pos only, neg cached)" if use_cached_neg else ""
    print(f"LLM encode: {llm_time:.1f}s{cache_label}{batch_label}")
    print(
        f"  Positive: {list(text_states.shape)}, valid={text_mask.sum().int().item()}"
    )
    if do_cfg:
        print(
            f"  Negative: {list(neg_text_states.shape)}, valid={neg_text_mask.sum().int().item()}"
        )
    timings["llm_encode_s"] = round(llm_time, 1)
    timings["llm_batched"] = do_cfg and not use_cached_neg

    # Free LLM to reclaim ~14GB memory
    del llm_model, llm_tokenizer
    gc.collect()
    print("  LLM freed from memory")

    # --- 2b: byT5 glyph encoding ---
    def encode_byt5(prompt_text):
        tokens = byt5_tokenizer(
            prompt_text,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = byt5_model(
                tokens.input_ids, attention_mask=tokens.attention_mask.float()
            )[0]
        return out.to(dtype), tokens.attention_mask.to(torch.long)

    t_byt5 = time.time()
    byt5_text_states, byt5_mask = encode_byt5(prompt)
    # Negative byT5: use cache if available
    if not use_cached_neg and do_cfg:
        neg_byt5_text_states, neg_byt5_mask = encode_byt5("")
    elif not do_cfg:
        neg_byt5_text_states = None
        neg_byt5_mask = None
    byt5_time = time.time() - t_byt5
    print(
        f"byT5 encode: {byt5_time * 1000:.1f}ms{' (positive only)' if use_cached_neg else ''}, shape={byt5_text_states.shape}"
    )
    timings["byt5_encode_ms"] = round(byt5_time * 1000, 1)

    # --- 2c: text_states_2 and vision_states ---
    # text_states_2 is always None in HunyuanVideo 1.5 (no second text encoder)
    text_states_2 = None
    # vision_states: zeros for t2v (no reference image)
    vision_states = torch.zeros(B, 16, 1152, dtype=dtype)

    print(f"\nText encoding summary:")
    print(
        f"  LLM text_states: {list(text_states.shape)} (Qwen2.5-VL hidden_states[-3])"
    )
    print(f"  text_states_2:   None (not used in HunyuanVideo 1.5)")
    print(f"  byT5:            {list(byt5_text_states.shape)}")
    print(f"  Vision:          zeros {list(vision_states.shape)} [t2v mode]")

    # ===== Stage 3: DiT Denoising Loop =====
    print(f"\n{'=' * 60}")
    print(f"Stage 3: DiT Denoising ({args.steps} steps)")
    print("=" * 60)

    timesteps, sigmas = create_timestep_schedule(args.steps, shift=args.shift)

    # Initial noise
    generator = torch.Generator().manual_seed(args.seed)
    latents = torch.randn(B, 32, T_lat, H_lat, W_lat, generator=generator, dtype=dtype)
    cond_latents = torch.zeros(
        B, 33, T_lat, H_lat, W_lat, dtype=dtype
    )  # t2v: no image cond

    # Warmup DiT
    dit_B = 2 if use_batched_cfg else B  # B=2 for batched CFG
    print(f"Warming up DiT (B={dit_B})...")
    dummy_img = torch.randn(dit_B, IMG_SEQ_LEN, 2048, dtype=dtype)
    dummy_txt = torch.randn(dit_B, TXT_SEQ_LEN, 2048, dtype=dtype)
    dummy_vec = torch.randn(dit_B, 2048, dtype=dtype)
    dummy_mask = torch.ones(dit_B, TXT_SEQ_LEN, dtype=torch.long)
    dummy_cos = torch.randn(IMG_SEQ_LEN, 128, dtype=dtype)
    dummy_sin = torch.randn(IMG_SEQ_LEN, 128, dtype=dtype)
    for i in range(3):
        t0 = time.time()
        _ = nxd_model(
            [dummy_img, dummy_txt, dummy_vec, dummy_mask, dummy_cos, dummy_sin], {}
        )
        print(f"  Warmup {i}: {(time.time() - t0) * 1000:.0f}ms")
    del dummy_img, dummy_txt, dummy_vec, dummy_mask, dummy_cos, dummy_sin

    # Run preprocessing once for text (tokens don't change per step)
    # Preprocess at step 0 timestep to get the text tokens and RoPE
    print("\nPreprocessing (CPU)...")
    hidden_states = torch.cat([latents, cond_latents], dim=1)  # [B, 65, T, H, W]

    def preprocess_and_prepare(text_st, text_msk, byt5_st, byt5_msk, label=""):
        """Preprocess text through token refiner and reorder, return fixed txt tokens and mask."""
        out = preprocessor.preprocess(
            hidden_states=hidden_states,
            timestep=timesteps[0:1],
            text_states=text_st,
            text_states_2=text_states_2,
            encoder_attention_mask=text_msk,
            byt5_text_states=byt5_st,
            byt5_text_mask=byt5_msk.float(),
            vision_states=vision_states,
            guidance=None,
            mask_type="t2v",
        )
        txt = out["txt"]
        txt_m = out["txt_mask"]  # [B, L_txt] with 1=valid, 0=padding
        L_txt = txt.shape[1]
        if L_txt < TXT_SEQ_LEN:
            txt = F.pad(txt, (0, 0, 0, TXT_SEQ_LEN - L_txt))
            txt_m = F.pad(
                txt_m, (0, TXT_SEQ_LEN - L_txt), value=0
            )  # pad mask with 0 (invalid)
        elif L_txt > TXT_SEQ_LEN:
            txt = txt[:, :TXT_SEQ_LEN]
            txt_m = txt_m[:, :TXT_SEQ_LEN]
        if label:
            valid_count = txt_m.sum().int().item()
            print(
                f"  {label}: txt {out['txt'].shape} -> {TXT_SEQ_LEN}, valid={valid_count}/{TXT_SEQ_LEN}"
            )
        return txt, txt_m, out

    t_preprocess = time.time()
    with torch.no_grad():
        # Positive prompt
        txt_tokens_cond, txt_mask_cond, preprocess_out = preprocess_and_prepare(
            text_states, text_mask, byt5_text_states, byt5_mask, "Positive"
        )
        # Negative prompt (for CFG)
        if do_cfg:
            txt_tokens_uncond, txt_mask_uncond, _ = preprocess_and_prepare(
                neg_text_states,
                neg_text_mask,
                neg_byt5_text_states,
                neg_byt5_mask,
                "Negative",
            )
    preprocess_time = time.time() - t_preprocess
    print(f"Preprocess: {preprocess_time * 1000:.0f}ms")
    print(f"  img: {preprocess_out['img'].shape}, vec: {preprocess_out['vec'].shape}")

    # Fixed across timesteps
    freqs_cos = preprocess_out["freqs_cos"]
    freqs_sin = preprocess_out["freqs_sin"]
    tt, th, tw = preprocess_out["tt"], preprocess_out["th"], preprocess_out["tw"]

    L_img_actual = preprocess_out["img"].shape[1]
    print(f"  L_img: {L_img_actual} (compiled: {IMG_SEQ_LEN})")

    if L_img_actual != IMG_SEQ_LEN:
        print(
            f"  WARNING: img seq len mismatch! Padding {L_img_actual} -> {IMG_SEQ_LEN}"
        )
        if L_img_actual < IMG_SEQ_LEN:
            pad = IMG_SEQ_LEN - L_img_actual
            freqs_cos = F.pad(freqs_cos, (0, 0, 0, pad))
            freqs_sin = F.pad(freqs_sin, (0, 0, 0, pad))

    txt_mask_tensor_cond = txt_mask_cond
    if do_cfg:
        txt_mask_tensor_uncond = txt_mask_uncond
    timings["preprocess_ms"] = round(preprocess_time * 1000, 0)

    # Denoising loop
    cfg_mode = ""
    if do_cfg:
        cfg_mode = f" (CFG={args.guidance_scale}, {'B=2 batched' if use_batched_cfg else '2x sequential'})"
    print(f"\nStarting {args.steps}-step denoising{cfg_mode}...")
    step_times = []

    for i in range(args.steps):
        t_step = time.time()

        # 1. On CPU: create hidden_states and preprocess (per-step for timestep embed)
        hidden_states = torch.cat([latents, cond_latents], dim=1)

        with torch.no_grad():
            pp = preprocessor.preprocess(
                hidden_states=hidden_states,
                timestep=timesteps[i : i + 1],
                text_states=text_states,
                text_states_2=text_states_2,
                encoder_attention_mask=text_mask,
                byt5_text_states=byt5_text_states,
                byt5_text_mask=byt5_mask.float(),
                vision_states=vision_states,
                guidance=None,
                mask_type="t2v",
            )

        img_tokens = pp["img"]
        vec_cond = pp["vec"]

        # Pad img if needed
        if img_tokens.shape[1] < IMG_SEQ_LEN:
            pad = IMG_SEQ_LEN - img_tokens.shape[1]
            img_tokens = F.pad(img_tokens, (0, 0, 0, pad))

        if do_cfg:
            # Also get uncond vec (same timestep, different text)
            with torch.no_grad():
                pp_uncond = preprocessor.preprocess(
                    hidden_states=hidden_states,
                    timestep=timesteps[i : i + 1],
                    text_states=neg_text_states,
                    text_states_2=text_states_2,
                    encoder_attention_mask=neg_text_mask,
                    byt5_text_states=neg_byt5_text_states,
                    byt5_text_mask=neg_byt5_mask.float(),
                    vision_states=vision_states,
                    guidance=None,
                    mask_type="t2v",
                )
            vec_uncond = pp_uncond["vec"]

            if use_batched_cfg:
                # B=2 batched path: stack uncond+cond, single Neuron call
                img_b2 = torch.cat([img_tokens, img_tokens], dim=0)  # [2, L_img, 2048]
                txt_b2 = torch.cat(
                    [txt_tokens_uncond, txt_tokens_cond], dim=0
                )  # [2, L_txt, 2048]
                vec_b2 = torch.cat([vec_uncond, vec_cond], dim=0)  # [2, 2048]
                mask_b2 = torch.cat(
                    [txt_mask_tensor_uncond, txt_mask_tensor_cond], dim=0
                )  # [2, L_txt]

                inputs_b2 = [img_b2, txt_b2, vec_b2, mask_b2, freqs_cos, freqs_sin]
                out_b2 = nxd_model(inputs_b2, {})
                if isinstance(out_b2, tuple):
                    vel_b2 = out_b2[0]
                else:
                    vel_b2 = out_b2
                vel_uncond = vel_b2[0:1]  # [1, L_img, 32]
                vel_cond = vel_b2[1:2]  # [1, L_img, 32]
            else:
                # Sequential path: two separate Neuron calls
                # 2a. Run DiT with unconditional text
                inputs_uncond = [
                    img_tokens,
                    txt_tokens_uncond,
                    vec_uncond,
                    txt_mask_tensor_uncond,
                    freqs_cos,
                    freqs_sin,
                ]
                out_uncond = nxd_model(inputs_uncond, {})
                if isinstance(out_uncond, tuple):
                    vel_uncond = out_uncond[0]
                else:
                    vel_uncond = out_uncond

                # 2b. Run DiT with conditional text
                inputs_cond = [
                    img_tokens,
                    txt_tokens_cond,
                    vec_cond,
                    txt_mask_tensor_cond,
                    freqs_cos,
                    freqs_sin,
                ]
                out_cond = nxd_model(inputs_cond, {})
                if isinstance(out_cond, tuple):
                    vel_cond = out_cond[0]
                else:
                    vel_cond = out_cond

            # CFG: velocity = uncond + scale * (cond - uncond)
            velocity_tokens = vel_uncond + args.guidance_scale * (vel_cond - vel_uncond)
        else:
            # No CFG: single forward pass
            inputs = [
                img_tokens,
                txt_tokens_cond,
                vec_cond,
                txt_mask_tensor_cond,
                freqs_cos,
                freqs_sin,
            ]
            out = nxd_model(inputs, {})
            if isinstance(out, tuple):
                velocity_tokens = out[0]
            else:
                velocity_tokens = out

        # 3. Unpatchify velocity -> [B, 32, T, H, W]
        velocity_tokens_trimmed = velocity_tokens[:, :L_img_actual]  # Remove padding
        velocity_latent = preprocessor.unpatchify(
            velocity_tokens_trimmed, tt, th, tw, out_channels=OUT_CHANNELS
        )

        # 4. Euler step
        latents = euler_step(latents, velocity_latent, sigmas[i], sigmas[i + 1]).to(
            dtype
        )

        elapsed = time.time() - t_step
        step_times.append(elapsed)

        if i % 10 == 0 or i == args.steps - 1:
            lat_mean = latents.float().mean().item()
            lat_std = latents.float().std().item()
            print(
                f"  Step {i:3d}/{args.steps}: {elapsed * 1000:.0f}ms "
                f"(lat mean={lat_mean:.4f}, std={lat_std:.4f})"
            )

    total_dit = sum(step_times)
    avg_step = (
        sum(step_times[1:]) / max(1, len(step_times) - 1)
        if len(step_times) > 1
        else step_times[0]
    )
    print(f"\nDiT complete: {total_dit:.1f}s total, {avg_step * 1000:.0f}ms/step avg")
    timings["dit_total_s"] = round(total_dit, 1)
    timings["dit_avg_step_ms"] = round(avg_step * 1000, 0)

    # Save latents for diagnostic use
    latents_save_path = f"{COMPILED_DIR}/debug_latents.pt"
    torch.save(latents, latents_save_path)
    print(f"Saved denoised latents to {latents_save_path} ({list(latents.shape)})")

    # ===== Stage 4: VAE Decode (Neuron tiled) =====
    if not args.skip_vae:
        print(f"\n{'=' * 60}")
        print("Stage 4: VAE Decode (Neuron tiled)")
        print("=" * 60)

        from tiled_vae_decode import TiledVAEDecoderNeuron

        VAE_DIR = f"{COMPILED_DIR}/vae_decoder_neuron"
        t0 = time.time()
        vae_decoder = TiledVAEDecoderNeuron(
            VAE_DIR,
            tile_latent_h=8,
            tile_latent_w=8,
            overlap_h=2,
            overlap_w=2,
        )
        vae_load_time = time.time() - t0
        print(f"VAE loaded: {vae_load_time:.1f}s")
        timings["vae_load_s"] = round(vae_load_time, 1)

        scaling_factor = 1.03682  # From HunyuanVideo VAE config
        latents_for_vae = latents / scaling_factor
        print(
            f"Latents: {list(latents_for_vae.shape)}, scaled by 1/{scaling_factor:.5f}"
        )

        t_vae = time.time()
        video_frames, vae_stats = vae_decoder.decode(latents_for_vae, verbose=False)
        vae_time = time.time() - t_vae

        video_frames = (video_frames.float() / 2 + 0.5).clamp(0, 1)
        T_out = (T_lat - 1) * 4 + 1
        print(
            f"VAE decode: {vae_time:.1f}s ({vae_stats['n_tiles']} tiles, {vae_stats['avg_tile_ms']:.0f}ms/tile)"
        )
        print(
            f"Output: {list(video_frames.shape)} ({T_out} frames at {H_lat * 16}x{W_lat * 16})"
        )
        print(
            f"Range: [{video_frames.min().item():.3f}, {video_frames.max().item():.3f}]"
        )
        timings["vae_decode_s"] = round(vae_time, 1)
        timings["vae_tiles"] = vae_stats["n_tiles"]
        timings["vae_tile_ms"] = vae_stats["avg_tile_ms"]
    else:
        print("\n(VAE decode skipped)")
        timings["vae_decode_s"] = 0.0

    # ===== Summary =====
    total = (
        timings.get("llm_encode_s", 0)
        + timings["byt5_encode_ms"] / 1000
        + timings["dit_total_s"]
        + timings.get("vae_decode_s", 0)
    )
    timings["total_pipeline_s"] = round(total, 1)
    total_load = (
        timings.get("llm_load_s", 0)
        + timings["byt5_load_s"]
        + timings["preprocess_load_s"]
        + timings["dit_load_s"]
    )

    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print("=" * 60)
    llm_method = (
        " (pos only, neg cached)"
        if use_cached_neg
        else (" (B=2 batched)" if timings.get("llm_batched") else "")
    )
    print(
        f"  LLM encode (CPU fp16):   {timings.get('llm_encode_s', 0):.1f}s{llm_method}"
    )
    print(f"  byT5 encode (CPU):       {timings['byt5_encode_ms']:.0f}ms")
    print(f"  Preprocess (CPU):        {timings['preprocess_ms']:.0f}ms/step")
    dit_cfg_label = ""
    if do_cfg:
        dit_cfg_label = (
            " (B=2 batched CFG)" if use_batched_cfg else " (2x sequential CFG)"
        )
    print(
        f"  DiT denoise (Neuron):    {timings['dit_total_s']:.1f}s "
        f"({args.steps} steps x {timings['dit_avg_step_ms']:.0f}ms)"
        f"{dit_cfg_label}"
    )
    print(f"  VAE decode (Neuron):     {timings.get('vae_decode_s', 'skipped')}s")
    print(f"  ---")
    print(f"  Total pipeline:          {total:.1f}s")
    print(f"  CFG guidance scale:      {args.guidance_scale}")
    print(f"  Model load time:         {total_load:.1f}s (not counted)")
    print(f"  Prompt: {prompt}")

    # ===== Stage 5: Save Video =====
    if not args.skip_vae:
        print(f"\n{'=' * 60}")
        print("Stage 5: Save Video Output")
        print("=" * 60)

        output_dir = f"{COMPILED_DIR}/output_video"
        os.makedirs(output_dir, exist_ok=True)

        # video_frames: [1, 3, T, H, W] float in [0, 1]
        frames = video_frames[0]  # [3, T, H, W]
        frames_uint8 = (frames * 255).clamp(0, 255).to(torch.uint8)
        T_out_actual = frames_uint8.shape[1]

        # Save individual frames as PNG
        from PIL import Image

        for fi in range(T_out_actual):
            frame_np = frames_uint8[:, fi].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img = Image.fromarray(frame_np)
            img.save(f"{output_dir}/frame_{fi:03d}.png")
            print(
                f"  Saved frame_{fi:03d}.png ({frame_np.shape[1]}x{frame_np.shape[0]})"
            )

        # Save as MP4 using imageio-ffmpeg
        try:
            import imageio

            mp4_path = f"{output_dir}/output.mp4"
            writer = imageio.get_writer(
                mp4_path, fps=8, codec="libx264", quality=8, pixelformat="yuv420p"
            )
            for fi in range(T_out_actual):
                frame_np = frames_uint8[:, fi].permute(1, 2, 0).cpu().numpy()
                writer.append_data(frame_np)
            writer.close()
            print(f"  Saved {mp4_path} ({T_out_actual} frames, 8 fps)")
        except Exception as e:
            print(f"  MP4 save failed: {e}")
            print(f"  Frames saved as PNGs in {output_dir}/")

        print(f"\nOutput directory: {output_dir}/")

    results = {
        "task": "e2e-pipeline-full-text-encoding",
        "prompt": prompt,
        "steps": args.steps,
        "shift": args.shift,
        "seed": args.seed,
        "video": f"{(T_lat - 1) * 4 + 1} frames at {H_lat * 16}x{W_lat * 16}",
        "dit_img_seq_len": IMG_SEQ_LEN,
        "dit_txt_seq_len": TXT_SEQ_LEN,
        "tp_degree": 4,
        "text_encoder": "Qwen2.5-VL-7B-Instruct (language_model, fp16, CPU)",
        "llm_method": "cached_neg"
        if use_cached_neg
        else ("batched_b2" if timings.get("llm_batched") else "sequential"),
        "guidance_scale": args.guidance_scale,
        "cfg_enabled": do_cfg,
        "cfg_method": "batched_b2"
        if use_batched_cfg
        else ("sequential" if do_cfg else "none"),
        "vae_method": "neuron_tiled",
        "timings": timings,
        "step_times_ms": [round(t * 1000, 1) for t in step_times],
    }
    results_path = f"{COMPILED_DIR}/e2e_full_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
