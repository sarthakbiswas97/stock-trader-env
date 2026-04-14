"""Debug HF Inference API response format."""

import os
from huggingface_hub import InferenceClient

token = os.environ.get("HF_TOKEN")
client = InferenceClient(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", token=token)

messages = [
    {"role": "user", "content": "What is 2+2? Answer in one word."},
]

response = client.chat_completion(messages=messages, max_tokens=50, temperature=0.7)

print("Type:", type(response))
print("Response:", response)
print()
print("Choices:", response.choices)
print("Choice 0:", response.choices[0])
print("Message:", response.choices[0].message)
print("Content:", repr(response.choices[0].message.content))
