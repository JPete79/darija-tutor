from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
LESSONS_DIR = ROOT / "lessons"

# Persona sets the tutor's name and grammatical gender; "voice" picks the TTS voice later.
PERSONAS = {
    "noual": {"name": "Noual", "voice": "female", "gender": "female"},
    "abdul-ali": {"name": "Abdul-Ali", "voice": "male", "gender": "male"},
}
DEFAULT_PERSONA = "noual"

OLLAMA_URL = "http://localhost:11434"
# Set once a Darija GGUF is imported into Ollama (see README / scripts/find_gguf.py).
LLM_MODEL = "atlas-chat-2b"

# Low temperature keeps a small model coherent; raise for more varied chat.
LLM_OPTIONS = {"temperature": 0.6, "num_ctx": 4096}
