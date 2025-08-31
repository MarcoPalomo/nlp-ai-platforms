import requests
import os

MISTRAL_HOST = os.getenv("MISTRAL_HOST", "http://localhost:8000")

def chat_with_mistral(prompt: str, history: list[str]):
    messages = []
    if history:
        for idx, past in enumerate(history):
            messages.append({"role": "user" if idx % 2 == 0 else "assistant", "content": past})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "mistral",  # utile si multi-modèle, sinon ignoré
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False
    }

    response = requests.post(f"{MISTRAL_HOST}/v1/chat/completions", json=payload)

    if response.status_code != 200:
        raise Exception(f"Mistral error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]