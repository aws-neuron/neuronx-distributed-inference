from pathlib import Path

from openai import OpenAI

MODEL_ID = "/home/ubuntu/models/gemma-3-27b-it" # HF model ID or path to HF model artifacts

input_image_path = Path(__file__).resolve().parent / "data" / "dog.jpg"
IMAGE_URL = f"file://{input_image_path.as_posix()}"


client = OpenAI(
    api_key = "EMPTY",  # pragma: allowlist secret
    base_url = "http://localhost:8080/v1"
)

print("== Test text input ==")
completion = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is the recipe of mayonnaise in two sentences?"},
        ]
    }]
)
print(completion.choices[0].message.content)


print("== Test image+text input ==")
completion = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
        ]
    }]
)
print(completion.choices[0].message.content)
