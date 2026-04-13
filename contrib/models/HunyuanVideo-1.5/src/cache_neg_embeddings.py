"""
Pre-cache the negative (empty string) prompt embeddings for HunyuanVideo CFG.

The negative prompt is always "" (empty string), so we can pre-compute its
LLM + byT5 embeddings once and save them. This saves ~14s per pipeline run
(one full LLM forward pass on CPU).

Output:
    ./compiled/cached_neg_embeddings.pt
    Contains: neg_text_states, neg_text_mask, neg_byt5_states, neg_byt5_mask

Launch:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python ./cache_neg_embeddings.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5"))
os.environ["HUNYUAN_ATTN_MODE"] = "torch"

MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")

LLM_PATH = f"{MODELS_DIR}/Qwen2.5-VL-7B-Instruct"
LLM_MAX_LENGTH = 1000
HIDDEN_STATE_SKIP_LAYER = 2
dtype = torch.bfloat16

# Video prompt template (same as pipeline)
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


def main():
    print("=" * 60)
    print("Pre-cache Negative Prompt Embeddings")
    print("=" * 60)

    import gc
    from copy import deepcopy
    from transformers import AutoModel, AutoTokenizer

    negative_prompt = ""

    # --- LLM encoding ---
    print("\nLoading Qwen2.5-VL-7B...")
    t0 = time.time()
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, padding_side="right")
    llm_full = AutoModel.from_pretrained(
        LLM_PATH, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    llm_model = (
        llm_full.language_model if hasattr(llm_full, "language_model") else llm_full
    )
    llm_model.final_layer_norm = llm_model.norm
    llm_model = llm_model.to(dtype=torch.float16).eval()
    llm_model.requires_grad_(False)
    del llm_full
    gc.collect()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Find crop_start
    template = deepcopy(PROMPT_TEMPLATE_VIDEO)
    for item in template:
        if isinstance(item, dict) and "content" in item:
            item["content"] = item["content"].format(" ")
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
    print(f"  crop_start={crop_start}")

    # Encode negative prompt
    print("\nEncoding negative prompt through LLM...")
    template = deepcopy(PROMPT_TEMPLATE_VIDEO)
    for item in template:
        if isinstance(item, dict) and "content" in item:
            item["content"] = item["content"].format(
                negative_prompt if negative_prompt else " "
            )
    tokens = llm_tokenizer.apply_chat_template(
        template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        truncation=True,
        max_length=LLM_MAX_LENGTH + crop_start,
        padding="max_length",
        return_tensors="pt",
    )
    t0 = time.time()
    with torch.no_grad():
        out = llm_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )
        neg_text_states = out.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
        neg_text_states = neg_text_states[:, crop_start:]
        neg_text_mask = tokens["attention_mask"][:, crop_start:].to(dtype)
    llm_time = time.time() - t0
    print(f"  LLM forward: {llm_time:.1f}s")
    print(f"  neg_text_states: {list(neg_text_states.shape)}")
    print(f"  neg_text_mask valid: {neg_text_mask.sum().int().item()}")

    neg_text_states = neg_text_states.to(dtype)

    del llm_model, llm_tokenizer
    gc.collect()

    # --- byT5 encoding ---
    print("\nEncoding negative prompt through byT5...")
    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2

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

    tokens = byt5_tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        neg_byt5_states = byt5_model(
            tokens.input_ids, attention_mask=tokens.attention_mask.float()
        )[0].to(dtype)
    neg_byt5_mask = tokens.attention_mask.to(torch.long)
    print(f"  neg_byt5_states: {list(neg_byt5_states.shape)}")
    print(f"  neg_byt5_mask valid: {neg_byt5_mask.sum().int().item()}")

    # --- Save ---
    save_path = f"{COMPILED_DIR}/cached_neg_embeddings.pt"
    cache = {
        "neg_text_states": neg_text_states,
        "neg_text_mask": neg_text_mask,
        "neg_byt5_states": neg_byt5_states,
        "neg_byt5_mask": neg_byt5_mask,
        "crop_start": crop_start,
        "negative_prompt": negative_prompt,
    }
    torch.save(cache, save_path)
    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"\nSaved to {save_path} ({file_size_mb:.1f} MB)")
    print(f"  Keys: {list(cache.keys())}")

    print(
        "\nDone! Use this cache to skip negative prompt LLM encoding in the pipeline."
    )


if __name__ == "__main__":
    main()
