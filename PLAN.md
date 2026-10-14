# Darija Tutor MVP — Setup & Starting Plan

## Context
Build a voice-first conversational tutor for Moroccan Darija (user studied in Fes), learning-project scale, using only open-source models first (no paid APIs). Flow: choose **Lesson** or **Discussion** -> user speaks -> STT -> LLM -> TTS -> spoken reply, looping conversationally.
The current working dir (CoatingSimulation) is unrelated. Create a fresh project at `C:\Users\jimpe\dev\darija-tutor` (outside OneDrive: model weights are many GB and OneDrive will try to sync them).

## Open-source building blocks (verify license + quality on your own audio before committing)
| Stage | Candidates | Notes |
|---|---|---|
| STT | [MoulSot v0.3 (Atlasia)](https://huggingface.co/01Yassine/moulsot.v0.3) — built for conversational/code-switched Darija; [whisper-medium-darija-combined](https://huggingface.co/amineouaki/whisper-medium-darija-combined); [whisper-small-darija](https://huggingface.co/Amassu/whisper-small-darija) (small, fast baseline) | Whisper has no Darija token; use `arabic` + transcribe. WER numbers are self-reported — benchmark yourself. |
| LLM | [Atlas-Chat 2B/9B (MBZUAI-Paris, Gemma 2)](https://arxiv.org/html/2409.17912v2) with the Darija-SFT-Mixture dataset (458K); [GemMaroc (Gemma 3 + LoRA)](https://www.arxiv.org/pdf/2505.17082) | Atlas-Chat is already Darija-tuned, so **start with it as-is + prompting**; LoRA comes later for tutor behavior (corrections, level-adaptive replies). |
| TTS | [XTTS-v2 Darija fine-tune](https://huggingface.co/medmac01/xtt2_darija_v0.1); [SpeechT5 Darija (DODa)](https://huggingface.co/spaces/HAMMALE/speecht5-darija/blob/main/app.py); [DODa CosyVoice2 dataset](https://huggingface.co/datasets/Jip7e/doda-darija-cosyvoice2) for fine-tuning | Weakest link — no mature option. Fallback: a generic Arabic voice (e.g. edge-tts is free but not open-source; Piper/MMS-TTS-ara) to unblock the pipeline. |
| Data | DODa (Darija Open Dataset), Darija-SFT-Mixture, DarijaBERT | For eval and fine-tuning. |

## Architecture
```
mic -> VAD/record -> STT -> (transcript) -> Tutor LLM (mode prompt + history) -> reply text -> TTS -> speaker
                                   \-> transcript + reply shown on screen (Arabic script + Latin transliteration + English gloss)
```
- Python 3.11, `uv` or venv, PyTorch (CUDA if GPU available), `transformers`, `llama-cpp-python` or Ollama (GGUF quantized Atlas-Chat for CPU/low-VRAM), `faster-whisper`/transformers for STT, `sounddevice` + `silero-vad` for mic.
- UI: start with **Gradio** (mic input, audio output, mode selector) — fastest path; revisit later.
- Each stage behind a small interface (`stt.transcribe(audio)->str`, `llm.reply(history, mode)->str`, `tts.speak(text)->wav`) so models can be swapped/benchmarked.

## Repo layout
```
darija-tutor/
  README.md  pyproject.toml  .gitignore (models/, data/, *.wav)
  src/darija_tutor/{stt.py, llm.py, tts.py, tutor.py (mode logic), app.py (Gradio), config.py}
  prompts/{lesson_system.md, discussion_system.md}
  lessons/            # structured lesson JSON/YAML (greetings, market, taxi, family...)
  eval/{stt_samples/, run_stt_bench.py, llm_prompts.md}
  notebooks/          # exploration, LoRA fine-tune
  models/ data/       # gitignored
```

## Phases
1. **Setup (day 1):** create dir, `git init`, venv, install deps, check GPU/VRAM (`nvidia-smi`) — this decides 2B vs 9B and quantization. Download one model per stage.
2. **LLM first (text-only, terminal):** run quantized Atlas-Chat 2B (then 9B Q4) via Ollama/llama.cpp. Build the tutor brain here: `discussion` mode (asks your topic, replies in simple Darija at your level, gently corrects) and `lesson` mode (driven by `lessons/*.yaml`: target phrases, vocab, drills). Typed input only. Judge quality with your Fes ear; record tokens/sec and VRAM. Decide model size here — it sets the budget for everything else.
3. **Add TTS (output):** speak the LLM replies; compare XTTS-v2 Darija vs SpeechT5 vs fallback voice using whatever VRAM the LLM leaves. Typed in, spoken out.
4. **Add STT (input):** record ~20 of your own phrases, compare whisper-small-darija / MoulSot / whisper-medium by eye/WER; add mic + VAD. Now the full loop is voice in, voice out.
5. **Gradio UI** with Lesson/Discussion choice, transcript view, Arabic-script + transliteration + English toggle; then latency tuning of the full loop (model load/unload sequencing).
6. **Improve (later):** LoRA (PEFT/QLoRA via `trl`/`unsloth`) on tutor-style dialogues (generate with Atlas-Chat, hand-correct) to get correction behavior; fine-tune TTS on DODa; latency work (streaming, smaller models).

## Hardware budget (laptop: Ryzen 7, NVIDIA RTX 8 GB VRAM) — quantization-first
Confirm exact GPU model + system RAM with `nvidia-smi` / `systeminfo` in step 1. Rough VRAM budget (approximate, to be measured):
| Stage | Config | ~VRAM |
|---|---|---|
| STT | `faster-whisper` (CTranslate2) int8/fp16 of whisper-small-darija (convert HF checkpoint with `ct2-transformers-converter`); medium only if it fits | 0.5–1.5 GB (or run on Ryzen CPU int8 to save GPU) |
| LLM | Atlas-Chat **2B** GGUF Q8/Q4_K_M via `llama.cpp`/Ollama (start here) -> **9B Q4_K_M** (~5.5–6 GB) as the upgrade | 2–3 GB / ~6 GB |
| TTS | XTTS-v2 fine-tune (fp16 ~2–3 GB, slowish) or SpeechT5 (small, light) | 1–3 GB |
- 8 GB cannot comfortably hold 9B + XTTS + STT at once. Strategy: **2B on GPU for MVP**; 9B variant requires STT on CPU and TTS loaded/unloaded per turn, or LLM partial offload (`n_gpu_layers`) — test and measure latency.
- Check whether GGUF quantizations of Atlas-Chat already exist on Hugging Face before quantizing yourself (`llama.cpp convert + llama-quantize` as fallback).
- LoRA later: QLoRA (4-bit) fits 2B on 8 GB; 9B QLoRA is borderline — consider free Colab/Kaggle GPU for training only, then pull the adapter locally.
- Windows note: install CUDA-enabled builds (`llama-cpp-python` CUDA wheel or Ollama, which is simplest on Windows).

## Key risks / decisions to flag
- **TTS quality** is the biggest unknown; **latency** on consumer hardware (STT+9B LLM+TTS) may be several seconds — use 2B or 4-bit quantization first.
- Darija has no standard orthography; chat input/output may be Arabic script or Latin ("arabizi"). Pick Arabic script for model I/O, show transliteration in UI.
- Check each model license (Gemma terms, XTTS/Coqui non-commercial) before any non-personal use.
- 8 GB VRAM: all three models must be quantized/sequenced; measure per-stage latency and VRAM in the Phase 3 loop before choosing model sizes.

## Verification
- Phase 2: STT transcribes your recorded phrases with readable Arabic-script output; LLM replies coherently in Darija; TTS audio intelligible to you.
- Phase 3: spoken question in -> spoken Darija answer out, 3+ turns with coherent context, end-to-end latency measured.
- Phase 4/5: both modes launchable from the UI; lesson mode walks through a lesson file; discussion mode asks for a topic first.

## First implementation actions (after approval)
Create `C:\Users\jimpe\dev\darija-tutor`, scaffold the layout above, write README + pyproject, `git init`, hardware check script, and stub `stt.py/llm.py/tts.py` with the first Atlas-Chat/Whisper-darija smoke tests.
