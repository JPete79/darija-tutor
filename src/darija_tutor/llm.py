import httpx

from . import config


def chat(messages: list[dict], model: str | None = None) -> str:
    """Send a chat history to the local Ollama server and return the reply text."""
    resp = httpx.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": model or config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            # Gemma 4 otherwise spends the reply budget on hidden reasoning and returns nothing;
            # models without thinking (Atlas-Chat) ignore it
            "think": False,
            "options": config.LLM_OPTIONS,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
