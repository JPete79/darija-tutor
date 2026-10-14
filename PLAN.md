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

## Progress (keep this updated as tasks complete)
- [x] Create project dir, scaffold layout, README, pyproject, .gitignore
- [x] `git init`, first commit, public repo pushed: https://github.com/JPete79/darija-tutor (commit author uses GitHub no-reply email)
- [x] Text-only chat loop (`tutor.py`, `llm.py`, `cli.py`), mode prompts, `lessons/greetings.yaml`
- [x] Personas: Noual (female) / Abdul-Ali (male): name + grammatical gender (see below); `voice` tag reserved for TTS; `--persona` flag
- [x] Lesson format supports `phrases`, `verbs` (conjugation tables), free-text `instructions`; sample `verbs_past_kteb.yaml` (DRAFT, forms need Fes-ear verification)
- [x] Discussion accepts `--topic`
- [x] Hardware check: RTX 4070 Laptop, 8188 MiB VRAM, Python 3.11.4
- [x] Install Ollama (0.40.4, local only, no account needed), GitHub CLI (gh, logged in)
- [x] Create `.venv` and `pip install -e .`
- [x] `scripts/find_gguf.py` updated: lists candidate repos by downloads; `<repo_id>` lists .gguf files + sizes
- [x] Researched available Darija LLMs / GGUFs (see Findings)
- [x] Downloaded to `models/` (gitignored): `Atlas-Chat-2B.Q4_K_M.gguf` (1.71 GB), `Atlas-Chat-9B.Q4_K_M.gguf` (5.76 GB), from mradermacher
- [x] Imported both into Ollama via `modelfiles/*.Modelfile` (Gemma chat template): `atlas-chat-2b`, `atlas-chat-9b`
- [x] Smoke test through `cli.py` works for both (added UTF-8 console fix and `--model` flag). First impressions, not yet judged by Fes ear:
  - 2B: ~5 s for 2 turns, ~2.3 GB VRAM; Darija plausible but ignores the prompt's transliteration/English-gloss lines.
  - 9B (before the fix below): ~16 s for 2 turns; 7.4 GB footprint at 4096 ctx = 24% CPU / 76% GPU split; looked like repeated replies (resolved, see next items).
- [x] 9B context/GPU fix: Ollama's default split put 2 layers on CPU even at 2048 ctx; `PARAMETER num_gpu 99` (now in both Modelfiles) puts it 100% on GPU. Measured: 9B ~40 tok/s at num_ctx 4096, 6.6 GB reported / ~7.1 GB used in nvidia-smi (leaves ~1 GB, so no room for TTS/STT on GPU alongside it; plan sequencing/CPU for those). 2048 ctx = 6.2 GB.
- [x] 9B "repetition" re-checked: not a template/stop-token bug (4 samples all stop cleanly, 36-89 tokens). It was the model inconsistently following the 3-line format. The "Arabic printed twice instead of transliteration" problem is resolved by moving transliteration into code (see multi-script item).
- [ ] (Later) download GemMaroc 4B and Qwen2.5-7B-darija candidates
- [x] First user verdict (2026-10-14): 9B latency acceptable (~40 tok/s), conversation quality good, EN glosses read well. Arabic quality not fully judged until TTS lets the user hear it. Tentatively use 9B as the working LLM; 2B stays as fast fallback. GemMaroc 4B / Qwen 7B not yet compared.
- [ ] Compare candidates by ear + tokens/sec + VRAM; pick LLM size
- [x] Multi-script display: user may write English / Arabic script / Arabizi (9B understands all three). Model answers in Arabic script + an `EN:` gloss line; code adds Arabizi via `src/darija_tutor/translit.py` (rule-based, ~0.88 similarity on lesson gold; misses short vowels, e.g. ktb vs kteb). CLI `--show arabic arabizi english`. Asking the LLM itself for Arabizi output failed (it echoed English), so transliteration lives in code. Lessons can use the exact `translit` stored in YAML instead.
- [x] Grammatical gender: personas now carry `gender`; system prompt tells the model its own gender and (optionally, `--learner male|female`) the learner's. Test (feel/happy prompt, 2 samples each): 9B used feminine فرحانة for Noual and masculine فرحان for Abdul-Ali (4/4); 2B used masculine فرحان for Noual (0/2 correct for female). Another reason to prefer 9B. Still to verify with Fes ear: other gendered forms (adjectives, participles, "you" forms).
- [x] Terminal Arabic display: the Windows/VS Code terminal has no right-to-left support, so Arabic printed reversed and unjoined (text itself was fine). CLI now reshapes + reorders by default via `arabic-reshaper` + `python-bidi` (`translit.for_terminal`); `--logical` prints raw text. Gradio/browser UI will not need this. Verified in the user's VS Code terminal with `scripts/arabic_test.py`: mode C (shaped + bidi, the CLI default) displays correctly; earlier wrong screenshots came from a chat session started before the fixes (restart the CLI after code edits). Also fixed Arabizi bug: ليوم now `lyum` (was `liwm`). Follow-up fix: long Arabic replies wrapped bottom-to-top in the terminal (the whole line was reordered, then the terminal wrapped it); `for_terminal` now word-wraps to the terminal width first, reorders each row, right-aligns, and the tutor name prints on its own line (visual mode only). Verified row order logically; user to confirm on screen.
- [x] Prompt guard for facts/opinions + literal EN (2026-10-14): user wants facts allowed but flagged. A plain "say if you're unsure" rule did nothing (9B can't tell when it's wrong: invented founders for Al-Qarawiyyin, stated the 2022 World Cup as current). What works: **hedge by question type** (always disclaim for history/people/dates/places/news/sports; never for everyday chat) plus few-shot examples in `prompts/discussion_system.md`; measured ~8/9 factual answers carry "I am not sure, according to my old knowledge...", 0/6 chat replies do, opinions flagged. Example content leaked once (copied a date) so the example now uses a placeholder. Facts after the disclaimer are still often wrong (that is the model's limit). EN line instructed to be a literal translation with `(lit. "...")` for idioms.
- User decision: keep Atlas-Chat 9B for now; may try GemMaroc later.
- [ ] Corrections list for Fes-preferred forms (feeds the prompt + Arabizi); user to supply examples (e.g. how a Fes speaker says/spells "you talk")
- [x] Golha TTS set up (2026-10-14): repo in `models/golha-darija-tts/` (gitignored), base weights fetched (~3.2 GB), separate conda env `.venv-tts` (Python 3.12, torch 2.6.0+cu126, transformers 4.46.3 pinned; needed `setuptools<81` because Perth imports `pkg_resources`). Run: `cd models\golha-darija-tts; ..\..\.venv-tts\python.exe tts_server.py` -> http://127.0.0.1:8020 (~1 min startup).
  - First run on 6 real 9B tutor replies (`tts_samples/golha/*.wav`, gitignored): RTF 0.43-0.76 on the 4070 Laptop (faster than real time, ~1.6-2.4 s per short reply); GPU memory ~7.9 GB while resident = **cannot coexist with the 9B** (unload one, or run TTS on CPU/other model).
  - Failure seen: greeting "مرحبا! شنو بغيتي تهضر عليه اليوم؟" produced a 39.7 s runaway clip for a ~3 s sentence (63 s to generate). Need a guard (max length by text length / retry) if we keep Golha.
  - Audio quality NOT yet judged: user to listen. Voice is default Chatterbox/Golha voice; no gender control yet (server passes `audio_prompt_path=None`; cloning would need code changes).
- [x] HADRA / Habibi-TTS Darija set up (2026-10-14), Golha server stopped (its files/env kept). Env `.venv-hadra` (Python 3.12, torch 2.6.0+cu126, f5-tts; `setuptools<81`), weights in `models/habibi-tts-doda-darija/` (EMA 674 MB + vocab), DODa reference clips for the 7 speakers (found via the model card's golden transcripts in `Jip7e/doda-darija-cosyvoice2`) in `tts_samples/hadra_refs/` (F1-F4 female, M1-M3 male; the full 1.2 GB `wavs.tar.gz` sits in `models/doda-cosyvoice2/`, can be deleted). Voice = which reference clip is passed; Noual -> F1 (Salma), Abdul-Ali -> M1 (Amine) for now.
  - Samples in `tts_samples/hadra/` (gitignored): 6 tutor replies x 2 voices + the taxi sentence in all 7 voices. Measured on the 4070 Laptop: RTF ~0.11-0.13 (nfe 16, sway -1.0, cfg 2.0, speed 0.95), peak 1.4 GB VRAM, all clip lengths sane (greeting 2.4 s; Golha ran away to 40 s on the same text). Needs reference clip + exact transcript per speaker.
  - 9B (7.1 GB at 4096 ctx) + HADRA (1.4 GB) = too tight; at num_ctx 2048 (6.2 GB) it should fit - to test.
  - Audio quality NOT yet judged by the user's ear.
- [x] **LLM + TTS integrated** (2026-10-14): `scripts/hadra_server.py` (run with `.venv-hadra`, port 8021, voices from `models/hadra-voices/`), `src/darija_tutor/tts.py` client (HTTP + `winsound` playback), CLI `--voice` / `--speaker F1..M3`; discussion mode only (lesson replies mix English and Darija). Persona -> voice: Noual = F1 (Salma), Abdul-Ali = M1 (Amine); `config.PERSONAS[...]["speaker"]`. The EN gloss is never spoken; text (Arabic/Arabizi/English) still prints.
  - 9B now `num_ctx 2048` (set only in the Modelfile; `config.LLM_OPTIONS` no longer overrides it). Measured with both loaded: 7.38 GB of 8.19 GB used, 9B 100% on GPU; per turn ~3-4 s LLM + ~0.5-1.3 s TTS (playback muted in the test). Tradeoff: 2048-token context limits long conversations and big lesson tables.
  - Not yet tested by ear in a live voice chat (user to try).
- [ ] TTS: check which Darija voices exist for male AND female (many are single-speaker; may need XTTS-style reference-audio cloning)
- [ ] Improve translit (schwa heuristics, or an Arabic->Arabizi model: candidates `akachkach/byt5-darija-transliteration-v1`, `akhanafer/arabic-to-arabizi`, unevaluated; `atlasia/Transliteration-Moroccan-Darija` is Arabizi->Arabic and gated)
- [ ] Lesson catalog: `type` (phrases/verbs/vocab/pronouns/grammar), tags, level; code-driven lesson picker
- [ ] Conversation topic list (greetings, casual chat, going out, sports, religion, ...)
- [ ] **Next:** TTS comparison (Phase 3), then STT (Phase 4), then Gradio UI (Phase 5)
- [x] Committed and pushed to GitHub (see git log for the latest commit)

## Findings
- **GGUFs already exist** (checked on Hugging Face index): Atlas-Chat 2B and 9B via mradermacher, QuantFactory, RichardErkhov, tensorblock. No need to self-quantize. `mradermacher/Atlas-Chat-9B-GGUF` Q4_K_M = 5.76 GB (Q5_K_M 6.65 GB is tight, Q8 9.8 GB won't fit).
- **Candidates to compare by ear:** Atlas-Chat 2B Q4_K_M (baseline), GemMaroc 4B tulu Q4_K_M (Gemma 3), Atlas-Chat 9B Q4_K_M (likely quality leader), GemMaroc Qwen2.5-7B-Instruct-darija (`GemMaroc/...-gguf`, Qwen base).
- Other small community Darija fine-tunes exist (Qwen2.5-Darija-7B-Full, llama3.1-8b-moroccan, Gemma-3-1B-Moroccan, silma-darija-2b); quality unknown, download counts are low (hundreds) for all.
- Terjman (AtlasIA) = translation only; Al-Atlas-0.5B = tiny base model; not tutor candidates.
- **Possible STT news:** Morocco + Mistral announced open-source Darija ASR (Voxtral-based) and a dialect classifier (late Sep/Oct 2026); release status, license and accuracy unconfirmed. Re-check at the STT phase alongside MoulSot (now `atlasia/moulsot.v0.3`).
- **New TTS candidates (found 2026-10-14, unevaluated):** `ai-ssam/darija-omnivoice-kore-v1` (updated Sep 2026), `Oussamamaat/golha-darija-tts` (Oct 2026). Add to the existing list: XTTS-v2 Darija fine-tune (`medmac01/xtt2_darija_v0.1`), SpeechT5 Darija, DarijaTTS-v0.1-500M (mradermacher GGUF), spark-tts-darija, neutts-air-darija (GGUF repos seen in HF search), generic Arabic fallback voice. Need: male AND female voice for Noual/Abdul-Ali.
- **New STT candidates (found 2026-10-14, unevaluated):** `digitalplacema/qwen3-asr-1.7b-darija-moulsot-replica` and `digitalplacema/qwenasr-1.7b-darija-v1` (Oct 2026), `hamzadr2/qwen3-asr-1.7b-darija-moulsot-replica`; existing: `atlasia/moulsot.v0.3`, whisper-small/medium darija; Mistral/Voxtral Darija ASR still unconfirmed.
- **TTS card review (2026-10-14):**
  - `Oussamamaat/golha-darija-tts`: LoRA (383 MB) on Chatterbox Multilingual (~3 GB base, downloaded by `fetch_base_model.py`); takes Arabic script, handles Darija+French code-switching; ships `tts_server.py` (localhost:8020) + UI; needs CUDA; 1.2 s/sentence (0.33x real-time) on RTX 3090 with its CUDA-graph speedups (our 4070 Laptop will be slower); known issues with numbers like "6 bar"/"10000"; MIT + Apache-2.0. Voice options unstated (Chatterbox base supports reference-audio cloning). **Recommended first TTS to try.**
  - `Tilas/neutts-air-darija-v1` (mradermacher GGUF quants, Apache-2.0): small, GGUF so likely CPU-friendly; good fit for the 8 GB budget; second to try.
  - `ai-ssam/darija-omnivoice-kore-v1`: one fixed voice ("Kore"), 2.5 GB, CC-BY-NC-4.0 (fine for personal use).
  - `HAMMALE/speecht5-darija`: takes Latin-script Darija with multiple male/female speaker embeddings; older/likely more robotic; our Arabizi lacks vowels so it's a poor input match.
  - GPU sharing: the 9B uses ~7 GB, so test TTS with the LLM unloaded (`ollama stop atlas-chat-9b`) and design sequencing later.
- **More TTS candidates (checked 2026-10-14 after the user shared a comparison table):**
  - `Jip7e/habibi-tts-doda-darija` ("HADRA", Apache-2.0, 336M-param F5-TTS flow-matching DiT on SWivid/Habibi-TTS; EMA weights 1.3 GB): Darija-specific, trained on DODa; **7 speakers (4 female, 3 male)** + zero-shot cloning; non-autoregressive so should not suffer Golha's runaway-length failure. Base `SWivid/Habibi-TTS` is CC-BY-NC-SA-4.0 (fine for personal use). **Now the top candidate to try next** (fits the Noual/Abdul-Ali voice need).
  - `sherif1313/3arab-TTS-500M-v2` / `-VoiceDesign` (Apache-2.0, ~553M, 48 kHz): general Arabic (not Darija-specific), 70 designed voices from text descriptions, no reference audio needed.
  - `oddadmix/lahgtna-chatterbox-v1` (MIT, Chatterbox-based, 7.5 GB repo; GGUF by `cstr/lahgtna-chatterbox-v1-GGUF`) and `oddadmix/lahgtna-omnivoice-v2`: multi-dialect Arabic incl. Moroccan; wants **diacritized** Arabic, which our LLM does not produce.
- No Atlas-Chat successor found (9B last updated Oct 2024). Darija LoRAs on Atlas-Chat are niche (Algerian, empathy, summarization); a LoRA changes style/domain, not world knowledge, so stale knowledge (user observed a 2022-era answer on soccer champions) needs a newer base model or a prompt caveat. Possible newer-base Darija candidates, unevaluated: `ElmehdiSMILI/darija-qwen35-2b-lora`, `abdnaouri/silma-darija-9b-lora`, `Mansourizona/mira-darija-v1`.
- **Hardware for "full scale" (rough estimates, not measured):** current laptop = 8 GB (9B Q4 at 2048 ctx + HADRA 1.4 GB = 7.4 GB used). 9B Q8 file is 9.8 GB -> ~12 GB GPU; Atlas-Chat 27B Q4 ~16-17 GB weights -> 24 GB GPU (RTX 3090/4090) with modest context; comfortable "everything resident" (9B Q8 or 27B Q4 + HADRA + Whisper STT + longer context) = 24 GB; several simultaneous users = server GPU(s) with a batching server (vLLM or llama.cpp server), KV cache per session. Apple Silicon with 32-64 GB unified memory is the other route for the 27B.
- Licenses not yet checked (Gemma, Qwen terms); fine for personal use.
- Lesson choice by the LLM itself is unreliable at 2B-9B; let code shortlist lessons and have the model only phrase the offer.

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
