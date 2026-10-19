import re
import time

import httpx

from . import config

_SENTENCE_END = re.compile(r"(?<=[.!?؟…])\s+")
_PUNCT = ".,!?؟،؛:;()\"'«»"
_PAUSE = {"normal": 0.0, "sentences": 1.0, "words": 0.6}   # seconds between parts


def synthesize(text: str, speaker: str, speed: float | None = None) -> bytes:
    """Ask the local HADRA server (scripts/hadra_server.py) for a WAV of `text` in `speaker`'s voice.

    `speed` below 1.0 slows the voice; omit it to use the server default.
    """
    body = {"text": text, "speaker": speaker}
    if speed is not None:
        body["speed"] = speed
    resp = httpx.post(f"{config.TTS_URL}/speak", json=body, timeout=120)
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


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def split_words(text: str) -> list[str]:
    return [w for w in (t.strip(_PUNCT) for t in text.split()) if w]


class Voice:
    """A speaker plus the learner-adjustable settings: speed and how the text is broken up.

    mode "normal" speaks the reply as one piece, "sentences" one sentence at a time with a pause,
    "words" one word at a time (a single word can sound a little odd on its own).
    """

    MIN_SPEED, MAX_SPEED = 0.5, 1.1
    MODES = ("normal", "sentences", "words")

    def __init__(self, speaker: str, speed: float = 0.95, mode: str = "normal"):
        self.speaker = speaker
        self.speed = speed
        self.mode = mode

    def set_speed(self, value: float) -> float:
        self.speed = round(min(max(value, self.MIN_SPEED), self.MAX_SPEED), 2)
        return self.speed

    def say(self, text: str, mode: str | None = None) -> None:
        mode = mode or self.mode
        parts = {"normal": [text], "sentences": split_sentences(text), "words": split_words(text)}[mode]
        for i, part in enumerate(parts):
            play(synthesize(part, self.speaker, self.speed))
            if i < len(parts) - 1 and _PAUSE[mode]:
                time.sleep(_PAUSE[mode])
