# Darija Tutor

Voice-first conversational tutor for Moroccan Darija, open-source models only (no paid APIs). Learning project / MVP. The user studied Darija in Fes and is the quality judge for Darija output.

Full plan: see PLAN.md (read it first). Hardware: RTX 4070 Laptop 8 GB VRAM, 31 GB RAM, Windows 11, Python 3.11 — everything must be quantized.

## Status
The checklist in PLAN.md ("Progress" and "Findings") is the source of truth; update it as tasks complete and as new information arises.
- Done: text chat loop, personas (Noual/Abdul-Ali, name + gender), lessons (phrases/verbs), Atlas-Chat 2B/9B in Ollama (9B = working model, num_ctx 2048), Arabic/Arabizi/English display with terminal RTL fix, HADRA TTS (own env `.venv-hadra`, server `scripts/hadra_server.py`, voices F1 Noual / M1 Abdul-Ali) wired in via `--voice`. Public repo JPete79/darija-tutor.
- Decisions: keep the 9B for now (GemMaroc / 27B maybe later); facts get an "I'm not sure" disclaimer by question type; user's Fes ear is the quality judge.
- Not done: STT (typed input only), Gradio UI, lesson catalog/topic list, Fes corrections list, better Arabizi vowels. Latest work may be uncommitted: check `git status`.
- Next: STT (voice in), then UI. See PLAN.md for details.
