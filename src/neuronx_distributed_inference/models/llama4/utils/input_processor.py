# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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

import base64
from io import BytesIO
from PIL import Image


def prepare_generation_inputs_hf(text_prompt, image_data, hf_llama4_processor, role="user", config=None):
    if image_data is not None:
        if not isinstance(image_data, list):
            image_data = [image_data]
        content = []
        for image_or_image_path in image_data:
            if isinstance(image_or_image_path, str):
                with open(image_or_image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    content.append(
                        {"type": "image", "url": f"data:image/jpeg;base64,{base64_image}"}
                    )
            elif isinstance(image_or_image_path, Image.Image):
                # Convert PIL Image to bytes using an in-memory buffer
                buffer = BytesIO()
                image_or_image_path.save(buffer, format="JPEG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                content.append(
                    {"type": "image", "url": f"data:image/jpeg;base64,{base64_image}"}
                )
            else:
                raise TypeError(f"Invalid image_data, it should be one or a list of str or PIL.Image, but got {image_or_image_path}")
        content.append({"type": "text", "text": text_prompt})
        messages = [
            {
                "role": role,
                "content": content
            },
        ]
    else:
        messages = [
            {
                "role": role,
                "content": [
                    {"type": "text", "text": text_prompt},
                ]
            },
        ]

    inputs = hf_llama4_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # prepare image mask
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None:
        image_mask = (inputs["input_ids"] == config.image_token_index).unsqueeze(-1)
    else:
        pixel_values = image_mask = None
    return inputs["input_ids"], inputs["attention_mask"], pixel_values, image_mask
