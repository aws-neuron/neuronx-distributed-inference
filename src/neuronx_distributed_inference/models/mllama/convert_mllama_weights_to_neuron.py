# Adopted from Meta-> HF conversion script here: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/convert_mllama_weights_to_hf.py

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import math
import os
from typing import List, Optional

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

CONTEXT_LENGTH = 131072
# Constants from Meta's original PyTorch implementation
VISION_NUM_LAYERS = 32
VISION_NUM_LAYERS_GLOBAL = 8
VISION_PATCH_SIZE = 14
VISION_NUM_CHANNELS = 3
VISION_DIM = 1280
VISION_NUM_HEADS = 16
VISION_INTERMEDIATE_LAYERS_INDICES = [3, 7, 15, 23, 30]
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
EOS_TOKEN_ID = [128001, 128008, 128009]

logger = logging.getLogger(__name__)


def compute_intermediate_size(hidden_dim, multiple_of=1024, ffn_dim_multiplier=1.3):
    hidden_dim = 4 * int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def write_model_config(
    model_path,
    input_base_path,
    instruct=False,
):
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(input_base_path, "params.json"), "r") as f:
        params = json.load(f)

    params = params.get("model", params)

    model_config = dict(
        architectures=["MllamaForConditionalGeneration"],
        model_type="mllama",
        torch_dtype="bfloat16",
        checkpoint="META",
    )

    cross_attention_num_layers = params["vision_num_cross_attention_layers"]
    text_num_layers = params["n_layers"]
    cross_attention_frequency = math.ceil(text_num_layers / cross_attention_num_layers)
    text_num_total_layers = text_num_layers + cross_attention_num_layers
    cross_attention_layers_shift = list(
        range(cross_attention_frequency - 1, text_num_total_layers, cross_attention_frequency + 1)
    )

    text_config = dict(
        num_attention_heads=params["n_heads"],
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        num_hidden_layers=text_num_layers,
        cross_attention_layers=cross_attention_layers_shift,
        vision_num_cross_attention_layers=cross_attention_num_layers,
        intermediate_size=compute_intermediate_size(
            params["dim"], multiple_of=params["multiple_of"]
        ),
        max_position_embeddings=CONTEXT_LENGTH,
        bos_token_id=BOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID if instruct else EOS_TOKEN_ID[0],
        torch_dtype="bfloat16",
        num_key_value_heads=params["n_kv_heads"],
        hidden_act="silu",
    )

    vision_config = dict(
        torch_dtype="bfloat16",
        patch_size=VISION_PATCH_SIZE,
        max_num_tiles=params["vision_max_num_chunks"],
        image_size=params["vision_chunk_size"],
        num_hidden_layers=VISION_NUM_LAYERS,
        num_global_layers=VISION_NUM_LAYERS_GLOBAL,
        num_channels=VISION_NUM_CHANNELS,
        hidden_size=VISION_DIM,
        attention_heads=VISION_NUM_HEADS,
        intermediate_layers_indices=VISION_INTERMEDIATE_LAYERS_INDICES,
    )

    model_config["text_config"] = text_config
    model_config["vision_config"] = vision_config

    config_file_path = os.path.join(model_path, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(model_config, f, indent=4, sort_keys=True)

    # generation config
    if instruct:
        logger.info("Saving generation config...")
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=model_config["text_config"]["bos_token_id"],
            eos_token_id=model_config["text_config"]["eos_token_id"],
            pad_token_id=model_config["text_config"]["pad_token_id"],
        )
        generation_config.save_pretrained(model_path)

    return model_config


def prepare_dim_map(config: dict):
    dim_map = {
        "text_model.tok_embeddings.weight": 0,
        "text_model.learnable_embedding.weight": 0,
        "vision_model.vision_encoder.conv1._linear.weight": 0,
        "vision_model.vision_projection.weight": 0,
        "vision_model.vision_projection.bias": 0,
        "text_model.output.weight": 0,
    }

    # vision_encoder 32 layers (constant)
    for i in range(VISION_NUM_LAYERS):
        prefix = f"vision_model.vision_encoder.transformer.resblocks.{i}"
        # ImageFeedForward
        dim_map[f"{prefix}.mlp.c_fc.weight"] = 0
        dim_map[f"{prefix}.mlp.c_fc.bias"] = 0
        dim_map[f"{prefix}.mlp.c_proj.weight"] = 1

        # ImageAttention
        dim_map[f"{prefix}.attn.wq.weight"] = 0
        dim_map[f"{prefix}.attn.wk.weight"] = 0
        dim_map[f"{prefix}.attn.wv.weight"] = 0
        dim_map[f"{prefix}.attn.wo.weight"] = 1

    # text self attention layers
    for i in range(config["text_config"]["num_hidden_layers"]):
        prefix = f"text_model.layers.{i}"
        # Attention
        dim_map[f"{prefix}.attention.wq.weight"] = 0
        dim_map[f"{prefix}.attention.wk.weight"] = 0
        dim_map[f"{prefix}.attention.wv.weight"] = 0
        dim_map[f"{prefix}.attention.wo.weight"] = 1

        # FeedForward
        dim_map[f"{prefix}.feed_forward.w1.weight"] = 0
        dim_map[f"{prefix}.feed_forward.w3.weight"] = 0
        dim_map[f"{prefix}.feed_forward.w2.weight"] = 1

    # text CrossAttention layers
    for i in range(config["text_config"]["vision_num_cross_attention_layers"]):
        prefix = f"text_model.cross_attention_layers.{i}"
        dim_map[f"{prefix}.attention.wq.weight"] = 0
        dim_map[f"{prefix}.attention.wk.weight"] = 0
        dim_map[f"{prefix}.attention.wv.weight"] = 0
        dim_map[f"{prefix}.attention.wo.weight"] = 1
        dim_map[f"{prefix}.feed_forward.w1.weight"] = 0
        dim_map[f"{prefix}.feed_forward.w3.weight"] = 0
        dim_map[f"{prefix}.feed_forward.w2.weight"] = 1

    # global transformer 8 layers (constant)
    for i in range(VISION_NUM_LAYERS_GLOBAL):
        prefix = f"vision_model.vision_encoder.global_transformer.resblocks.{i}"
        # ImageFeedForward
        dim_map[f"{prefix}.mlp.c_fc.weight"] = 0
        dim_map[f"{prefix}.mlp.c_fc.bias"] = 0
        dim_map[f"{prefix}.mlp.c_proj.weight"] = 1

        # ImageAttention
        dim_map[f"{prefix}.attn.wq.weight"] = 0
        dim_map[f"{prefix}.attn.wk.weight"] = 0
        dim_map[f"{prefix}.attn.wv.weight"] = 0
        dim_map[f"{prefix}.attn.wo.weight"] = 1

    return dim_map


def save_sharded_checkpoints(model_path: str, loaded: List[torch.tensor], config: dict):
    dim_map = prepare_dim_map(config)

    model_state = loaded[0]
    # validate
    for k in dim_map:
        assert k in model_state
    logger.info("Passed validation")

    for i in range(1, len(loaded)):
        logger.info(f"Loading {i}")
        state_dict = loaded[i]
        for k in state_dict:
            if k not in dim_map:
                continue
            state = torch.cat((model_state[k], state_dict[k]), dim_map[k])
            model_state[k] = state

    save_file(model_state, os.path.join(model_path, "model.safetensors"))


def write_neuron_model(
    model_path,
    input_base_path,
    num_shards,
    model_config,
):
    os.makedirs(model_path, exist_ok=True)

    logger.info(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    if num_shards == 1:
        loaded = [
            torch.load(
                os.path.join(input_base_path, "consolidated.pth"), map_location="cpu", mmap=True
            )
        ]
    else:
        loaded = [
            torch.load(
                os.path.join(input_base_path, f"consolidated.{i:02d}.pth"),
                map_location="cpu",
                mmap=True,
            )
            for i in range(num_shards)
        ]

    save_sharded_checkpoints(model_path, loaded, model_config)


class MllamaConverter(TikTokenConverter):
    def __init__(
        self,
        vocab_file,
        special_tokens: List[str],
        pattern: str,
        model_max_length: int,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(vocab_file, pattern=pattern)
        self.additional_special_tokens = special_tokens
        tokenizer = self.converted()
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
            **kwargs,
        )


def write_tokenizer(tokenizer_path: str, save_dir: str, instruct: bool = False):
    model_max_length = CONTEXT_LENGTH
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: W605

    # Special tokens
    num_reserved_special_tokens = 256
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    special_tokens += [
        f"<|reserved_special_token_{i + 2}|>"
        for i in range(num_reserved_special_tokens - len(special_tokens))
    ]
    # original tokenizer has <|image|> with 128011 token_id,
    # however, later in the code it is replaced with 128256 token_id
    special_tokens.append("<|image|>")

    # Chat template
    chat_template = (
        "{% for message in messages %}"
        "{% if loop.index0 == 0 %}"
        "{{ bos_token }}"
        "{% endif %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|image|>' }}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    converter = MllamaConverter(
        vocab_file=tokenizer_path,
        pattern=pattern,
        special_tokens=special_tokens,
        model_max_length=model_max_length,
        chat_template=chat_template if instruct else None,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>" if not instruct else "<|eot_id|>",
        pad_token="<|finetune_right_pad_id|>",
    )
    tokenizer = converter.tokenizer
    tokenizer.save_pretrained(save_dir)

    if instruct:
        logger.info("Saving chat template...")
        chat_template_path = os.path.join(save_dir, "chat_template.json")
        with open(chat_template_path, "w") as f:
            json.dump({"chat_template": chat_template}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="Llama-3.2-11B-Vision/original",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output-dir",
        default="Llama-3.2-11B-Vision",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe-serialization",
        default=True,
        type=bool,
        help="Whether or not to save using `safetensors`.",
    )
    parser.add_argument(
        "--special-tokens",
        default=None,
        type=List[str],
        help="The list of special tokens that should be added to the model.",
    )
    parser.add_argument(
        "--num-shards",
        default=1,
        type=int,
        help="The number of consolidated_xx.pth in input-dir.",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )
    args = parser.parse_args()

    model_config = write_model_config(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        instruct=args.instruct,
    )

    write_neuron_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        num_shards=args.num_shards,
        model_config=model_config,
    )

    write_tokenizer(
        tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
        save_dir=args.output_dir,
        instruct=args.instruct,
    )


if __name__ == "__main__":
    main()
