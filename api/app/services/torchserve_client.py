import requests
import os

TORCHSERVE_HOST = os.getenv("TORCHSERVE_HOST", "http://localhost:8080")
MODEL_NAME = os.getenv("TORCHSERVE_MODEL", "ner")

def get_ner(text: str):
    endpoint = f"{TORCHSERVE_HOST}/predictions/{MODEL_NAME}"
    response = requests.post(endpoint, data=text.encode("utf-8"))

    if response.status_code != 200:
        raise Exception(f"TorchServe error: {response.text}")
    
    return response.json()  # ou response.text selon le handler