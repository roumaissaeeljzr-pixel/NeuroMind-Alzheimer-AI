import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "gemma3:1b"


def generate_with_llm(prompt):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]