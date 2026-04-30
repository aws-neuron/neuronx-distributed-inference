import base64
from io import BytesIO
from PIL import Image
import torch


def prepare_generation_inputs_hf(text_prompt, image_data, hf_pixtral_processor, role="user", config=None):
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

    hf_pixtral_processor.tokenizer.pad_token = hf_pixtral_processor.tokenizer.eos_token
    inputs = hf_pixtral_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    # prepare image mask
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None:
        image_mask = (inputs["input_ids"] == config.image_token_index).unsqueeze(-1)
        image_sizes = inputs["image_sizes"]
    else:
        pixel_values = image_mask = image_sizes = None
    return inputs["input_ids"], inputs["attention_mask"], pixel_values, image_mask, image_sizes


def pad_vision_embeddings(vision_embeddings, pad_limit):
    padding_size = pad_limit - vision_embeddings.shape[1]
    if padding_size > 0:
        padding = torch.full(
            (vision_embeddings.shape[0], padding_size, vision_embeddings.shape[2]), 0, dtype=vision_embeddings.dtype, device=vision_embeddings.device
        )
        vision_embeddings = torch.cat([vision_embeddings, padding], dim=1)
    return vision_embeddings
