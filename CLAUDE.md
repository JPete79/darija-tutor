# Darija Tutor

Voice-first conversational tutor for Moroccan Darija, open-source models only (no paid APIs). Learning project / MVP. The user studied Darija in Fes and is the quality judge for Darija output.

Full plan: see PLAN.md (read it first). Hardware: RTX 4070 Laptop 8 GB VRAM, 31 GB RAM, Windows 11, Python 3.11 — everything must be quantized.

## Status
The checklist in PLAN.md ("Progress" and "Findings") is the source of truth; update it as tasks complete and as new information arises.
- Done: scaffold, text-only chat loop, personas (Noual/Abdul-Ali), lesson format (phrases/verbs), public GitHub repo (JPete79/darija-tutor), Ollama + gh installed, .venv created.
- Decision: build LLM-first, then TTS, then STT, then Gradio UI. Compare candidate Darija GGUFs by ear before choosing a size.
- GGUFs exist for Atlas-Chat 2B/9B and GemMaroc (see PLAN.md Findings); no self-quantizing needed.
- Not done: no model downloaded, nothing run against Ollama yet.
- Next: pick/download candidate GGUFs, import into Ollama, try `python -m darija_tutor.cli discussion`.
