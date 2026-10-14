# Darija Tutor

Voice-first conversational tutor for Moroccan Darija, built on open-source models (no paid APIs).
Target hardware: RTX 4070 Laptop (8 GB VRAM), 31 GB RAM, Windows 11 — so everything is quantized.

Pipeline (built LLM-first): `mic -> STT -> LLM -> TTS -> speaker`. Two modes: **lesson** and **discussion**.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
winget install Ollama.Ollama
```

## Phase 2 — LLM (text-only)

1. `python scripts/find_gguf.py` to find an Atlas-Chat GGUF (2B first, 9B Q4_K_M later).
2. Download the `.gguf` into `models/`, then write a `Modelfile` containing `FROM ./models/<file>.gguf` (and Gemma chat template) and run `ollama create atlas-chat-2b -f Modelfile`.
3. Chat: `python -m darija_tutor.cli discussion` or `python -m darija_tutor.cli lesson greetings`.

Personas: `--persona noual` (default, female voice later) or `--persona abdul-ali` (male voice later); changes the name only.
Examples:
- `python -m darija_tutor.cli lesson verbs_past_kteb --persona abdul-ali` (structured verb drill)
- `python -m darija_tutor.cli discussion --topic "spirituality in Morocco"` (free conversation)
Type to the tutor in English, Arabic script or Arabizi. Replies show Arabic, Arabizi and English; pick views with `--show arabic arabizi english`, and the model with `--model atlas-chat-9b`.
Voice (discussion mode): start the speech server in one terminal, then chat with `--voice` in another:
```
.venv-hadra\python.exe scripts\hadra_server.py
.venv\Scripts\python.exe -m darija_tutor.cli discussion --model atlas-chat-9b --voice
```
Noual speaks with HADRA voice F1 and Abdul-Ali with M1; override with `--speaker F1..F4|M1..M3`. See PLAN.md for how the TTS environment and weights were set up.
Lesson files in `lessons/` may have `phrases`, `verbs` (conjugation tables) and free-text `instructions`.

Plan: `C:\Users\jimpe\.claude\plans\i-am-looking-to-mossy-lamport.md`
