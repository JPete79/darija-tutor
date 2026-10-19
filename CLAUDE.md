# Darija Tutor

Voice-first conversational tutor for Moroccan Darija, open-source models only (no paid APIs). Learning project / MVP. The user studied Darija in Fes and is the quality judge for Darija output.

Full plan: see PLAN.md (read it first). Hardware: RTX 4070 Laptop 8 GB VRAM, 31 GB RAM, Windows 11, Python 3.11 — everything must be quantized.

## Status
The checklist in PLAN.md ("Progress" and "Findings") is the source of truth; update it as tasks complete and as new information arises.
- Done: text chat loop, personas (Noual/Abdul-Ali, name + gender), lessons (phrases/verbs), Atlas-Chat 2B/9B in Ollama (9B = working model, num_ctx 2048), Arabic/Arabizi/English display with terminal RTL fix, HADRA TTS (own env `.venv-hadra`, server `scripts/hadra_server.py`, voices F1 Noual / M1 Abdul-Ali) wired in via `--voice`. Public repo JPete79/darija-tutor.
- Decisions: keep the 9B for now (GemMaroc / 27B maybe later); facts get an "I'm not sure" disclaimer by question type; user's Fes ear is the quality judge.
- Not done: STT (typed input only), Gradio UI, lesson catalog/topic list, Fes corrections list, better Arabizi vowels. Latest work may be uncommitted: check `git status`.
- Now: Phase 2b in PLAN.md (Track A has the full status + next steps). A 2x3 grid: Qwen3.5-9B / Gemma-4-12B-it x (1 as released, 2 GemMaroc recipe = TULU 50K LoRA, 3 TULU 50K + AtlasChat 100K, each from the released model). Parallel tracks: workbooks (user awaiting PDFs from the Fes school) and verb fact sheets. Then STT, then UI.
- Eval: tooling in `eval/` (see `eval/README.md`). Every run logs itself to `eval/experiments.csv`; per-model summary in `eval/scoreboard.csv`. Rules: full benchmark sets only, same scorer for all models, one model on the GPU at a time, stop the HADRA server during evals. The user is learning evaluation: explain choices.
- Baselines done (local, Q4): Atlas-Chat 2B / 9B DarijaMMLU 44.50 / 58.42 (match published); DarijaHellaSwag uses llama.cpp per-token scoring, which runs higher than published (2B 42.66 vs 35.08), fine for internal comparison (9B 58.59 vs 43.65).
- Next (user-confirmed order, PLAN.md Track A steps A2-A8): (1) download a small model per family (Qwen3.5-0.8B, Gemma-4-E2B; keep them for training tests) and check the scoring methods work on them (caveats in `eval/README.md`: hybrid Qwen3.5 in llama.cpp HellaSwag, tokenizer-dependent per-token scoring, letter tokens); (2) download Qwen3.5-9B + Gemma-4-12B-it Q4_K_M (~13 GB) and run phase 1 (as released) tests; (3) train the small models locally to find bugs; (4) rent a GPU and train run 2 (GemMaroc recipe) and run 3 (GemMaroc + AtlasChat hybrid).
