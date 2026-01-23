# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

from openai import OpenAI

MODEL = "/home/ubuntu/model_hf/gemma-3-27b-it/"

client = OpenAI(
    api_key = "EMPTY",  # pragma: allowlist secret
    base_url = "http://localhost:8080/v1"
)

print("== Test text input ==")
completion = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is the recipe of mayonnaise in two sentences?"},
        ]
    }]
)
print(completion.choices[0].message.content)


print("== Test image input ==")
completion = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///home/ubuntu/daanggn-neuron-inference-migration/scripts/dog.jpg"}},
            {"type": "text", "text": "Describe this image"},
        ]
    }]
)
print(completion.choices[0].message.content)