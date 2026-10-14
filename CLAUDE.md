# Darija Tutor

Voice-first conversational tutor for Moroccan Darija, open-source models only (no paid APIs). Learning project / MVP. The user studied Darija in Fes and is the quality judge for Darija output.

Full plan: see PLAN.md (read it first). Hardware: RTX 4070 Laptop 8 GB VRAM, 31 GB RAM, Windows 11, Python 3.11 — everything must be quantized.

## Status
- Phase 1 done: scaffold, git init (no commits yet), text-only chat loop (src/darija_tutor), prompts, greetings lesson, scripts/find_gguf.py.
- Decision: build LLM-first (Atlas-Chat 2B GGUF via Ollama), then TTS, then STT, then Gradio UI.
- Not done: Ollama not installed, venv not created, no model downloaded, nothing run/tested yet. No Atlas-Chat 2B/9B GGUF confirmed on HF (only 27B by DevQuasar); may need to quantize with llama.cpp.
- Next: install Ollama, create venv + `pip install -e .`, run scripts/find_gguf.py, get Atlas-Chat 2B running, try `python -m darija_tutor.cli discussion`.
