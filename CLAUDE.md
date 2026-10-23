# Darija Tutor

Voice-first conversational tutor for Moroccan Darija, open-source models only (no paid APIs). Learning project / MVP. The user studied Darija in Fes and is the quality judge for Darija output.

Full plan: see PLAN.md (read it first). Hardware: RTX 4070 Laptop 8 GB VRAM, 31 GB RAM, Windows 11, Python 3.11 — everything must be quantized.

## Status
The checklist in PLAN.md ("Progress" and "Findings") is the source of truth; update it as tasks complete and as new information arises.
- Done: text chat loop, personas (Noual/Abdul-Ali, name + gender), lessons (phrases/verbs), Atlas-Chat 2B/9B in Ollama (9B = working model, num_ctx 2048), Arabic/Arabizi/English display with terminal RTL fix, HADRA TTS (own env `.venv-hadra`, server `scripts/hadra_server.py`, voices F1 Noual / M1 Abdul-Ali) wired in via `--voice`. Public repo JPete79/darija-tutor.
- Decisions: keep the 9B for now (GemMaroc / 27B maybe later); facts get an "I'm not sure" disclaimer by question type; user's Fes ear is the quality judge.
- Not done: STT (typed input only), Gradio UI, lesson catalog/topic list, Fes corrections list, better Arabizi vowels. Latest work may be uncommitted: check `git status`.
- Now: Phase 2b, Track A (PLAN.md: read the "Start here" block under Track A first). Decision 2026-10-23: train **Gemma-4-E4B-it** (not the 9B/12B grid for now) with run 2 (GemMaroc TULU 50K) and run 3 (TULU 50K + AtlasChat 100K) on a rented GPU, plus a fluency baseline (Darija vs MSA). Parallel tracks: workbooks (user awaiting PDFs from the Fes school) and verb fact sheets. Then STT, then UI.
- Eval: tooling in `eval/` (see `eval/README.md`). Every run logs itself to `eval/experiments.csv`; per-model summary in `eval/scoreboard.csv`; `eval/report.py` writes a readable HTML summary to `data/report.html` (user prefers this over CSVs; keep its STATUS/KEY_POINTS text current). Rules: full benchmark sets only, same scorer for all models, one model on the GPU at a time, stop the HADRA server during evals. The user is learning evaluation: explain choices.
- Results (local Q4_K_M, full DarijaMMLU / DarijaHellaSwag per-token): Atlas-Chat-9B 58.42 / 58.59 (current tutor), Atlas-Chat-2B 44.50 / 42.66, Qwen3.5-9B 61.05 / 46.10, Gemma-4-E4B-it 60.09 / 49.97, Gemma-4-12B-it 54.12 / not run (answer-format issue: line break before the letter, not a knowledge gap). HellaSwag per-token is not fair across tokenizers. DarijaMMLU is not fluency.
- Training (A5 done 2026-10-23): env `.venv-train` (conda env; call `.venv-train\python.exe`), `train/train_lora.py` (LoRA, GemMaroc settings) and `train/export_gguf.py` (merge -> llama.cpp b11565 converter in `models/llama.cpp-src/` -> Q4_K_M -> Ollama). Smoke-tested on E2B and E4B (QLoRA, local) plus the full export path. The laptop trains at only ~70-80 tokens/s: full runs need a rented GPU.
- Hard-won rules: never force-kill a process stuck in the GPU driver (it was followed by a Windows bugcheck/reboot); ask the user to reboot instead. Do not use paged optimizers. Use 127.0.0.1, not localhost, for Ollama on Windows. Gemma 4 needs exactly one `<bos>` (its chat template adds it). The user does not want overnight runs.
