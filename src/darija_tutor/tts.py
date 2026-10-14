import httpx

from . import config


def synthesize(text: str, speaker: str) -> bytes:
    """Ask the local HADRA server (scripts/hadra_server.py) for a WAV of `text` in `speaker`'s voice."""
    resp = httpx.post(f"{config.TTS_URL}/speak", json={"text": text, "speaker": speaker}, timeout=120)
    resp.raise_for_status()
    return resp.content


def play(wav: bytes) -> None:
    """Play WAV bytes on the default output device, blocking until finished (Windows)."""
    import winsound

    winsound.PlaySound(wav, winsound.SND_MEMORY)


def available() -> bool:
    try:
        return httpx.get(f"{config.TTS_URL}/health", timeout=3).json().get("ready", False)
    except httpx.HTTPError:
        return False
