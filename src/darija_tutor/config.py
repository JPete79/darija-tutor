from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
LESSONS_DIR = ROOT / "lessons"

# Persona sets the tutor's name and grammatical gender; "voice" picks the TTS voice later.
PERSONAS = {
    # "speaker" is a HADRA reference voice: F1-F4 female, M1-M3 male (models/hadra-voices/).
    "noual": {"name": "Noual", "voice": "female", "gender": "female", "speaker": "F1",
              "expertise": "Moroccan food and cooking, religion, and spirituality (including Sufism, "
                           "religious traditions and holidays)"},
    "abdul-ali": {"name": "Abdul-Ali", "voice": "male", "gender": "male", "speaker": "M1",
                  "expertise": "history, archaeology, and the city of Fes"},
}
DEFAULT_PERSONA = "noual"
# Telling the model it is an "expert" did not improve accuracy and cut its "I'm not sure" warnings
# (see PLAN.md), so the persona expertise line is off. Set True to re-enable.
EXPERTISE_PROMPT = False

# Discussion prompts are versioned (prompts/discussion_system_<v>.md) so evals can compare them;
# v1 is the prompt of the run 2 scores and the 2026-10-24 ear test.
PROMPT_VERSION = "v2"

# 127.0.0.1, not localhost: on Windows localhost tries IPv6 first and adds ~2 s per request.
OLLAMA_URL = "http://127.0.0.1:11434"
# Set once a Darija GGUF is imported into Ollama (see README / scripts/find_gguf.py).
LLM_MODEL = "atlas-chat-2b"

# Low temperature keeps a small model coherent; raise for more varied chat.
# num_ctx is set per model in modelfiles/ (9B: 2048 so it fits next to the TTS on an 8 GB GPU).
LLM_OPTIONS = {"temperature": 0.6}

# HADRA TTS runs as a separate local server (own Python env); see scripts/hadra_server.py.
TTS_URL = "http://127.0.0.1:8021"
