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
- [x] Paired translations (2026-10-14): the user noticed a 5-line Arabic reply with one unrelated EN line. Cause: single trailing `EN:` line can't cover several sentences and the 9B drifted into a different answer. Fix: prompt now requires each Arabic sentence followed by its own `EN:` line (max 5 sentences, "slow down" = still answer, in simpler words); display is per sentence block (Arabic / Arabizi / EN); `tutor.pair_lines` parses; `Tutor.complete_translations` sends any Arabic line still lacking English to the model in one short numbered translation request and rewrites the history in clean paired form. Live test (6 runs, gates of Fes): 19 Arabic lines, 0 without English, 3-8 s per turn. Known gaps: the fact disclaimer still sometimes missing on long answers (0/5 on the "slow down" gates request, fine on World Cup 4/4 and Qarawiyyin 3/4), facts still often wrong, and the fallback translations are the same model's work (verify with your ear).
- [x] Persona expertise test (2026-10-14): user asked whether "Noual is an expert in food/religion/spirituality, Abdul-Ali in history/archaeology/Fes" would make answers more accurate. Prompts do not re-weight the model; they only change the context it conditions on. A/B on the 9B (5 domain questions x 3 samples per persona, keyword-scored, one run, small n): correct Abdul-Ali 8/15 baseline vs 8/15 expert, Noual 8/15 vs 6/15; "I'm not sure" disclaimers dropped 12/15 -> 7/15 and 13/15 -> 3/15 with the expert line. Several facts failed in both (Bab Boujloud 0/3, pastilla 0/3). Result: expertise stays in `config.PERSONAS[...]["expertise"]` but `config.EXPERTISE_PROMPT = False`. Ideas: softer "enjoys talking about" personality wording (untested), or verified-fact lesson files per topic.
- [x] Disk cleanup (2026-10-14, user-approved list only): today's installs were ~45 GB. Deleted pip cache, `models/golha-darija-tts`, `models/doda-cosyvoice2` (1.2 GB voice-clip archive), and the duplicate `models/Atlas-Chat-*.gguf` (Ollama keeps its own copy; `modelfiles/*.Modelfile` point at `../models/*.gguf`, so re-creating those Ollama models needs the GGUFs re-downloaded from `mradermacher/Atlas-Chat-{2B,9B}-GGUF`). Kept: `.venv-tts` (5.3 GB, only needed for Golha, user not yet asked), `.venv-hadra`, HADRA weights and voices, Ollama models, Hugging Face cache.
- [x] Verb-conjugation failure (2026-10-14, user transcript "conjugate TKLM"): the 9B swapped the asked verb (تكلم) for its default Darija verb (هضر) in the first reply, then copied that swap and the "here is the conjugation" promise (with no list) for the rest of the chat; the 5-sentence cap also blocked lists. Changes: prompt now says use EXACTLY the word given and give every item of a list, one per line with its own EN line (list exemption from the cap); new `/clear` chat command resets the history. Measured effect of the prompt change is within noise (asked verb used 2/5 -> 2/5, full 6-form list 1/5 -> 2/5), and when it did conjugate it mixed verbs (كتبتي "you wrote" given as "you speak"). Conclusion: do not trust free-chat conjugation; plan a code-generated conjugation helper for regular (sound) verbs (prefix/suffix rules applied to the Arabic-script stem, Arabizi via translit, spoken by HADRA) and user-verified lesson tables for irregular verbs. Needs the user's verified forms (e.g. tkellem, hder).
- **Scorecard for Atlas-Chat 9B (user's summary, 2026-10-14):** conversation = plus; facts/history = neutral (often wrong but now disclaimed); verbs/lessons = minus. Design consequence: let the model do conversation; drive structured lessons (verbs, vocab, pronouns, parts of speech) from verified data + code, using the model only to chat around them. Candidate follow-ups: build a small eval set from the user's verified forms and compare 9B vs GemMaroc vs Atlas-Chat 27B on the same tasks.
- **Source material for lessons (user, 2026-10-14; nothing started):** user's workbooks from living in Morocco, Peace Corps Moroccan Arabic textbooks, YouTube lessons/discussions. Ideas: convert to the `lessons/*.yaml` format (phrases/verbs/vocab/dialogues, tagged by topic + level) and verify each against the source page; use dialogues as practice material and later LoRA data; use YouTube audio as an STT test set. Watch-outs: Peace Corps books mostly use Latin transliteration with special symbols, while the LLM/TTS need Arabic script (needs a careful Latin->Arabic step; `atlasia/Transliteration-Moroccan-Darija` goes Arabizi->Arabic but is gated); regional dialect differences (user is the Fes judge); copyright: keep source files in a git-ignored `sources/` folder and commit only content known to be shareable (Peace Corps/US-government works are often public domain, check per document).
- **Model-family check (2026-10-14, from a table the user found):** `GemMaroc/GemMaroc-27b-it` is real (Gemma 3 27B fine-tune on TULU-3 50k Darija-English; 55 GB full weights, GGUFs exist; the table's "advanced historical reasoning" is the card's claim that base reasoning is preserved, not a Darija-facts result). "SILMA-9B-Darija" is not one model: closest is `abdnaouri/silma-darija-9b-lora` (LoRA by an individual, ~6 downloads, on `silma-ai/SILMA-9B-Instruct-v1.0`, a Gemma-license Arabic model; trained on Moroccan civic/legal/monetary/cultural data; its self-reported "0% hallucination" scores belong to a compound system with deterministic engines, and the card shows the raw neural model scoring 0/6 on its out-of-distribution set). Nothing run yet. Plan: build `scripts/compare_models.py` to run the same prompt set (conversation / facts / verbs) through several Ollama models and write a side-by-side file for blind judging; candidates sized for 8 GB first (GemMaroc 4B, GemMaroc-Qwen2.5-7B, SILMA 9B), 27B via CPU offload as a one-off. Downloads need user approval.
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
- [x] Learner-controlled voice (2026-10-14): in-chat commands `/slow` `/fast` `/speed X` (0.5-1.1, server default 0.95; model card recommends 0.90-0.95, lower is untested for quality), `/sentences` (one sentence at a time, 1 s pause), `/words` (one word at a time, 0.6 s pause; single words may sound odd), `/normal`, `/repeat [mode]`, `/help`; implemented as `tts.Voice` + `command()` in `cli.py`; `scripts/hadra_server.py` accepts a per-request `speed` (server must be restarted to pick it up). Tested offline with a stubbed model/voice; not yet heard live. Idea for later: also accept natural phrases ("slow down", "break it up") and per-word English glosses.
- [x] Clipped endings fixed (2026-10-14): the user heard replies occasionally cut off at the end. Cause: F5 sizes each clip from text length (ref seconds per UTF-8 byte), so the final syllable is sometimes squeezed off; measured speech still sounding in the last 50 ms in 5/30 clips (F1+M1, 5 texts x 3 runs). Tested: +0.35 s fixed duration 1/30 (slows the whole sentence, +8% length); appending " ..." to the text 0/30 (+4% length). Server now appends " ..." and a 0.25 s silent tail (guards against `winsound` dropping the buffer end). `scripts/talk.ps1` / `talk.cmd` added: one command starts/reuses the server and opens the voice chat; `COMMANDS.md` lists run commands.
- [ ] TTS: check which Darija voices exist for male AND female (many are single-speaker; may need XTTS-style reference-audio cloning)
- [ ] Improve translit (schwa heuristics, or an Arabic->Arabizi model: candidates `akachkach/byt5-darija-transliteration-v1`, `akhanafer/arabic-to-arabizi`, unevaluated; `atlasia/Transliteration-Moroccan-Darija` is Arabizi->Arabic and gated)
- [ ] Lesson catalog: `type` (phrases/verbs/vocab/pronouns/grammar), tags, level; code-driven lesson picker
- [ ] Conversation topic list (greetings, casual chat, going out, sports, religion, ...)
- [ ] **Next:** Phase 2b (retrain the GemMaroc recipe on Qwen3.5-9B / Gemma-4-12B-it; status and next steps under "Phase 2b > Track A"), then STT (Phase 4), then Gradio UI (Phase 5)
- [x] Eval tooling + Atlas-Chat baselines (2026-10-18): see Phase 2b > Track A and `eval/README.md`.
- [x] Committed and pushed to GitHub (see git log for the latest commit)

## Phase 2b: Retrain the GemMaroc recipe on a newer model (planned 2026-10-18)
The goal is a newer Darija model than Atlas-Chat (Gemma 2, 2024; it has stale knowledge and is weak on verbs). The work runs in three parallel tracks (A model, B workbooks, C fact sheets) that merge at the end. Status 2026-10-19: eval tooling, Atlas-Chat baselines, small-model check (A2), downloads (A3) done; A4 done for Qwen3.5-9B and Gemma-4-E4B, Gemma-4-12B next (Track A); Tracks B and C not started.

**Facts checked on Hugging Face (2026-10-18):**
- GemMaroc runs LoRA on **instruction-tuned** checkpoints, not raw base models (`GemMaroc-27b-it` has base_model `google/gemma-3-27b-it`; the Qwen variants start from `Qwen2.5-*-Instruct`). Their paper: LoRA r32/a64 (4B) or r16/a32 (27B), lr 1e-4, 3 epochs on TULU, max length 2048, bf16; 27B took 48 H100 GPU-hours.
- Published datasets: `GemMaroc/TULU-3-50k-darija-english` (≈20% kept in English), `Deita-5k-...`, `Lima-1k-...`, `AtlasChat-data-{50k,70k,100k}-randomlysampled-darija` (plus `-qwen-template` copies), and `TULU-3-50k-darija-english-plus-AtlasChat-data-50k-randomlysampled-darija`. GemMaroc only published a 27B model for the TULU + AtlasChat-50k mix (no scores on its card), so TULU + 100k has not been published.
- GemMaroc paper scores (self-reported, zero-shot): Atlas-Chat-9B DarijaMMLU 58.3 / DarijaHellaSwag 43.7; GemMaroc-27B 61.6 / 60.5; GemMaroc-4B-TULU 47.5 / 47.1 (the 4B base model scored 32.8 / 36.3).
- Candidate models: `Qwen/Qwen3.5-9B` (+ `-Base`; Feb 2026, Apache-2.0, hybrid Gated DeltaNet + attention, 201 languages, vision built in) and `google/gemma-4-12B-it` (+ base `gemma-4-12B`; May 2026, Apache-2.0, ~12B, text/image/**audio** input, 140+ languages pre-trained). Neither card mentions Darija or names Arabic.
- Benchmarks (all `MBZUAI-Paris/*`): DarijaMMLU (22K multiple-choice questions), DarijaHellaSwag, DarijaBench (sentiment, translation, summarization, etc.), DarijaAlpacaEval (needs an LLM judge). The first two were machine-translated with Claude 3.5 Sonnet, so they test general Darija, not Fes speech. No maintained text leaderboard was found.

**Design decisions / cautions:**
- Start from the **-it** checkpoints, as GemMaroc does. 50-150K examples is too little to turn a raw base model into a good chat model; a base-model run would be a separate, third experiment.
- **Screen before training:** score the untrained candidates plus Atlas-Chat 9B first (an eval-only GPU rental). These scores are the baseline that shows whether training helped, and they decide which model is worth training.
- **8 GB fit:** Qwen3.5-9B at Q4 should fit next to HADRA (similar to today's 9B). Gemma 4 12B at Q4 is about 7+ GB, so it likely **won't fit with HADRA on 8 GB** (TTS would need the CPU, or partial offload). Measure this before choosing.
- **Contamination:** run B trains on Atlas-Chat SFT data, and DarijaBench comes from the same source pool. Expect B to look better on DarijaBench partly for that reason; compare A and B mainly on DarijaMMLU and DarijaHellaSwag.
- Train text-only (skip the vision/audio towers). Check that the trainer (unsloth or TRL+PEFT) and llama.cpp/Ollama support the exact architecture, especially Qwen3.5 Gated DeltaNet, before paying for GPU time.
- Workbook material (copyrighted school books) and any LoRA trained on it stay out of the public repo; only code and schemas are committed.

### Track A: model
**Start here (hand-off 2026-10-23, for a new chat):**
- Done: eval tooling + baselines; as-released scores for Qwen3.5-9B, Gemma-4-E4B, Gemma-4-12B (A4 results below); training env + `train/train_lora.py` + `train/export_gguf.py`, smoke-tested on E2B and E4B (A5 progress below). E4B smoke: 69.8M LoRA params, eval loss 1.634 -> 1.470 in 10 steps, checkpoints OK. Everything committed.
- Decided by the user: train **Gemma-4-E4B-it** with run 2 (`--data tulu`) and run 3 (`--data tulu+atlas`) on a rented GPU, each from the released model, bf16 LoRA (no `--qlora`), GemMaroc settings (script defaults). Measure a **fluency baseline** before comparing trained vs released.
- Next, in order:
  1. Pinned requirements for the rented machine (`train/requirements.txt`: torch 2.14.1, transformers 5.20.2, peft 0.21.2, trl 1.16.0, accelerate, pyarrow, huggingface_hub; no bitsandbytes needed in bf16).
  2. Step-by-step GPU-rental guide (RunPod suggested; the user creates the account and pays): pick GPU (H100 80GB or A100 80GB), PyTorch image, copy `train/`, install, **15-minute check first** (20-step run: loss, tokens/s, checkpoint save) to measure speed and cost, then launch run 2 (and run 3 on a second pod if the budget allows), `--save-steps ~500`, download the `adapter/` and selected checkpoints.
  3. While it trains (local GPU free): fluency baseline on released E4B (+ Atlas-9B, Qwen-9B for reference): a fixed tutor prompt set (~30 prompts, reused for the A8 ear test), Darija-vs-MSA share via an Arabic dialect classifier (research which one handles Moroccan), DarijaBench translation/summarization (chrF).
  4. After: `train/export_gguf.py` per adapter/checkpoint, then DarijaMMLU (full), HellaSwag, fluency; compare against our own Q4_K_M of the released E4B (`train/export_gguf.py --no-adapter --name gemma-4-e4b-ours`; E2B check showed our pipeline matches unsloth's file). Score a few checkpoints, not only the last: training can hurt (E2B smoke dipped 50.2 -> 46.4).
- Estimates (H100, not measured yet): run 2 = 137M tokens (3 epochs) ~3-4 h ~$10; run 3 ~2x run 2. Open question for the user: both runs at once (two pods) or run 2 first.
- Later: Gemma-4-12B 3-shot / chat-template check; tokenizer-fair (per-byte) HellaSwag in `.venv-train`.

User goal (2026-10-18): this track is also for **learning how to evaluate and benchmark models**. No candidate is picked yet; let the scores decide. Build the eval step by step and explain each choice (scoring method, prompt format, sample size).

**Experiment design (user-confirmed 2026-10-18):** a 2 x 3 grid. Models: Qwen3.5-9B (Alibaba) and Gemma-4-12B-it (Google), both the released instruction-tuned versions.
| Run | Starting model | Training data | Method |
|---|---|---|---|
| 1. As released | released model | none | score only |
| 2. GemMaroc as-is (GemMaroc's recipe, not their model files) | released model | `GemMaroc/TULU-3-50k-darija-english` (50K, ~20% English) | LoRA, GemMaroc settings: lr 1e-4, 3 epochs, max length 2048, bf16, r16-32 |
| 3. GemMaroc + AtlasChat hybrid | released model (**not** run 2's output) | TULU 50K + `GemMaroc/AtlasChat-data-100k-randomlysampled-darija` (150K) | identical settings to run 2 (so ~3x the steps and cost) |
Run 3 with 100K AtlasChat rows has not been published (GemMaroc only released a 50K-AtlasChat 27B, without scores), so it is compared against our own run 2.

**Evaluation rules (learned 2026-10-18):**
- Compare models on **full benchmark sets only**. A 500-question sample of the same model ranged 50.0-66.6% on DarijaMMLU (10,000 resamples); the first one drew 64.0 vs 58.4 on the full set. 500-question runs are for quick checks only.
- Same scorer, prompt and settings for every model; never compare numbers from different scorers (local llama.cpp/Ollama vs the harness on a rented GPU go in separate columns).
- One model on the GPU at a time (two Ollama models swap on every request and both stall). Stop the HADRA server for big models.
- Every run is logged automatically in `eval/experiments.csv` (pass `--note "why"`); per-model summary in `eval/scoreboard.csv` (filled by `eval/scoreboard.py` from full-set standard runs). Details in `eval/README.md`.

**Done:**
- [x] A0a. Local eval tooling (2026-10-18), all in `eval/`:
  - `darija_mmlu.py`: DarijaMMLU via Ollama `top_logprobs` (letter scoring as in Atlas-Chat's harness fork, same Darija prompt); options `--scoring logprob|generate`, `--template raw|chat`, `--shots 0-3`, `--n/--seed`, `--note`. Timestamped result files, never overwritten.
  - `darija_hellaswag.py`: DarijaHellaSwag (validation, 10,042) via `llama-perplexity --hellaswag` on the same GGUF Ollama uses (llama.cpp b11565 CUDA 12.4 in git-ignored `models/llama.cpp/`, ~1.5 GB unpacked). Scores **acc_norm per token**; the published numbers are acc_norm per byte. No per-question output (llama.cpp shuffles internally).
  - `compare_runs.py` (paired McNemar test), `registry.py` (experiment log), `scoreboard.py`.
  - Data facts: DarijaMMLU test = 21,792 questions (card says 22,027), 2-5 choices (chance ~28%), gold letters skewed to A (28.5%), smallest dev split 3 (max 3 shots). Dataset re-uploaded 2024-09-27 (95 answers / 208 choice lists changed vs Aug 2024).
- [x] Baselines, local, Q4_K_M (2026-10-18):
  | Model | DarijaMMLU (logprob, raw, 0-shot) | Published | DarijaHellaSwag (llama.cpp per-token) | Published (per-byte) |
  |---|---|---|---|---|
  | Atlas-Chat-2B | 44.50 | 44.97 | 42.66 (41.70-43.63) | 35.08 |
  | Atlas-Chat-9B | 58.42 (57.76-59.07) | 58.23 | 58.59 (57.63-59.55) | 43.65 |
  - DarijaMMLU reproduces the published numbers. DarijaHellaSwag runs higher than published on the full set (2B +7.6, 9B +14.9 points), so the difference is method, not sampling (likely per-token vs per-byte normalization, or the card reporting plain acc). User decision: fine to use our own consistent scores; the harness run on the rented GPU (incl. Atlas-Chat-9B) will calibrate against published numbers later.
  - Other findings: 2B chat template vs raw 46.4 vs 44.4 (p = 0.19, no real difference); with the chat template both models answer "A" without the leading space. 9B under-predicts D (3,173 vs 4,963 gold), over-predicts B. 9B: Arab-world exam questions 60.1 vs translated US MMLU 55.2. Speeds: DarijaMMLU 2B ~0.1 s/question (~25 min full), 9B ~0.18 s (~65 min); HellaSwag 2B ~13 min.

**Next steps:**
- [x] A1. 9B DarijaHellaSwag baseline done (2026-10-18, 58.59; needed `--ctx 1024`: at 2048 the 9B spilled 1.5 GB into system RAM and ran 5x slower). Scoreboard refreshed.
**Next steps (user-confirmed order, 2026-10-18; continue in a new chat):**
- [x] A2. **Small-model check of the scoring methods** (done 2026-10-19, results below). Download one small model per family and keep them (needed later for local training tests): Qwen3.5-0.8B (~0.5 GB Q4) and Gemma-4-E2B (~2-3 GB Q4), e.g. from `unsloth/*-GGUF`; import into Ollama. Run a few hundred questions of DarijaMMLU and DarijaHellaSwag on each. Check: (a) both architectures load in Ollama 0.40.4 and llama.cpp b11565; (b) DarijaMMLU `no_answer` stays ~0 (letter tokens read correctly); (c) llama-perplexity HellaSwag works on the hybrid Qwen3.5 (recurrent layers + shared-prefix branching): compare its answers with scoring each ending separately. See the caveats in `eval/README.md`.
  - Models: `unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M (0.53 GB) and `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M (3.11 GB) in `models/`, Ollama `qwen3.5-0.8b` / `gemma-4-e2b` (`modelfiles/`; Qwen needs `RENDERER qwen3.5` + `PARSER qwen3.5`, Gemma gets its renderer automatically). Original bf16 weights (`Qwen/Qwen3.5-0.8B` 1.75 GB, `google/gemma-4-E2B-it` 10.25 GB, ungated) in the HF cache for A5. Skipped the vision `mmproj` files (text-only).
  - (a) Both load and run in Ollama 0.40.4 and llama.cpp b11565, no errors.
  - (b) **Scorer bug found and fixed:** `letter_logprobs` treated a whitespace token (stripped to `""`) as a letter, because `"" in "ABCDE"` is True. Qwen often ranks `"

"` first, so 161/500 answers were blank and scored wrong, while `no_answer` (counts only `None`) still showed 0. Fixed; after the fix `no_answer` = 0 for both. Earlier baselines unaffected: Atlas-9B full set 0 questions, Atlas-2B full set 7 of 21,792 (<0.03 points). Letters arrive as single `" A"` tokens in both families.
  - (c) llama.cpp HellaSwag is correct on the hybrid: `eval/hellaswag_crosscheck.py` scored the same 300 questions with llama.cpp (Q4, one question per call) and with transformers (bf16, each ending scored separately); Gemma as the plain-Transformer control. Same verdict: Qwen 96.3% (31.7 vs 32.0), Gemma 93.3% (41.7 vs 43.0, McNemar p = 0.50). Both match best when the first differing token is scored, so llama.cpp scores it.
  - **Gemma 4 needs BOS:** its HF tokenizer does not add `<bos>` (llama.cpp does); without it Gemma degenerates ("The capital of France is France is France...", HellaSwag 29% vs 43%). The A5 training pipeline must add `<bos>` for Gemma 4 (check the chat template output).
  - First scores (500 questions, ±4): DarijaMMLU Qwen3.5-0.8B 30.0 (chance ~28), Gemma-4-E2B-it 50.6 (Atlas-Chat-2B 44.4 on the same 500); DarijaHellaSwag Qwen 28.6, Gemma 39.4 (Atlas-2B 42.66 full set).
  - Speeds: DarijaMMLU 500 in ~34 s for both; transformers bf16 Qwen 0.2 s/question, Gemma E2B 4 s (10 GB bf16 partly offloaded to RAM); without `flash-linear-attention` transformers uses a slow reference kernel for Qwen's DeltaNet layers (fine at 0.8B, install it for training).
- [x] A3. **Download the large models** (done 2026-10-19; plus `unsloth/gemma-4-E4B-it-GGUF` Q4_K_M 4.98 GB as an extra candidate: it likely fits next to HADRA. Ollama: `qwen3.5-9b`, `gemma-4-12b`, `gemma-4-e4b`; Qwen3.5-9B loads 100% on GPU at 5.2 GB) (~13 GB): `unsloth/Qwen3.5-9B-GGUF` `Qwen3.5-9B-Q4_K_M.gguf` (5.68 GB) and `unsloth/gemma-4-12B-it-GGUF` `gemma-4-12b-it-Q4_K_M.gguf` (7.12 GB; may spill on 8 GB: stop HADRA, use `--ctx 1024`, watch GPU power/shared memory). Alternative for Gemma: Google's QAT `google/gemma-4-12B-it-qat-q4_0-gguf` (6.98 GB). Create Modelfiles; set `ollama_model` in the scoreboard rows.
- [ ] A4 results so far (2026-10-19, full sets, local Q4_K_M, standard setting):
  | Model | DarijaMMLU (95% CI) | exam / translated MMLU | DarijaHellaSwag (per-token) |
  |---|---|---|---|
  | Atlas-Chat-9B (current tutor) | 58.42 (57.8-59.1) | 60.1 / 55.2 | 58.59 |
  | Qwen3.5-9B, as released | **61.05** (60.4-61.7) | 62.2 / 58.7 | 46.10 (45.1-47.1) |
  | Gemma-4-E4B-it, as released | **60.09** (59.4-60.7) | 61.9 / 56.5 | 49.97 (49.0-51.0) |
  - Paired McNemar on DarijaMMLU: Qwen vs Atlas p < 0.0001 (only Qwen right 3,381, only Atlas right 2,808: they know different things); Qwen vs E4B p = 0.006 (1 point). Untrained, both beat the current tutor on DarijaMMLU and lose clearly on HellaSwag.
  - **Caution (discussed with user):** DarijaMMLU = understanding + knowledge (input), not fluency. HellaSwag is closer to fluency but our per-token score is not fair across tokenizers (Qwen vs Gemma vs Atlas). Needed before choosing: (1) a generation check: tutor prompts -> share of Darija vs MSA replies via a dialect classifier, plus DarijaBench translation/summarization (chrF); (2) per-byte HellaSwag (transformers 4-bit, in the A5 training env, extend `hellaswag_crosscheck.py`); the A6 harness run calibrates later. To be added to the plan after the 12B tests.
  - E4B's first full run crashed once at 6,037 (not reproducible, likely a dropped connection); `darija_mmlu.py` now retries Ollama requests.
  - Remaining (next session, user's choice: no overnight runs): Gemma-4-12B fit check (7.1 GB; stop HADRA, watch for spill), full DarijaMMLU + HellaSwag, 3-shot format checks.
  - **Gemma-4-12B-it (2026-10-23): DarijaMMLU 54.12** (53.5-54.8), below E4B. Answer-format issue, not knowledge: its top token is a line break on ~31% of questions (E4B ~0%), median best-letter probability ~55% vs ~93%; where it answers with a letter first it scores 58.7%. Needs a 3-shot / chat-template check before judging. HellaSwag not run (user moved on to training). Fit: 7.7 GB, 100% GPU, 0.12 s/question; a ~0.7-1.9 GB shared-memory spill does not slow the short eval prompts. (False alarm while measuring: `localhost` in a test script added ~2 s per request on Windows (IPv6 first); the eval scripts use 127.0.0.1.)
  - **User decision (2026-10-23):** step back from scoring; train Gemma-4-E4B: local slices to find bugs, then rented GPU; fluency baseline (Darija vs MSA) to be measured before comparing trained vs released. `eval/report.py` writes a one-page HTML summary (`data/report.html`).
- [ ] A4. **Phase 1 tests (run 1, as released):** full DarijaMMLU (0-shot standard) + full DarijaHellaSwag, plus DarijaMMLU 3-shot as a format check (published Gemma-2-9B-it: 36 zero-shot vs 56 3-shot). Run `eval/scoreboard.py`. Cross-family comparison: rely on DarijaMMLU (HellaSwag per-token scoring depends on the tokenizer).
- [ ] A5 progress (2026-10-23):
  - Env `.venv-train` (conda env like `.venv-hadra`; a venv on Anaconda's base Python failed to load torch: WinError 1114 in c10.dll): torch 2.14.1+cu126, transformers 5.20.2, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2.
  - `train/train_lora.py`: rebuilds each conversation with the model's own chat template (the datasets' `text` column is Gemma 2/3 format), one `<bos>` (Gemma 4's template adds it), loss on assistant replies only, LoRA r32/a64 on attention + MLP of the language layers (not the vision/audio towers or per-layer-embedding projections), lr 1e-4, warmup 3% + cosine, effective batch 16, AdamW. `--qlora` = 4-bit + per-layer embedding tables on CPU for the 8 GB laptop.
  - Data facts (Gemma 4 tokenizer, chat template): TULU 50K = 44,725 conversations, 45.5M tokens/epoch (951 over 2048); AtlasChat 100K = 100,000, median 156 tokens, 44.0M tokens/epoch. So run 3 is ~2x run 2, not 3x.
  - transformers 5.x changes: `warmup_ratio` -> `warmup_steps` (float < 1 = ratio), `group_by_length` -> `train_sampling_strategy="group_by_length"`.
  - **Bug: `paged_adamw_8bit` hung writing optimizer.pt at the first checkpoint; force-killing the process (stuck in the NVIDIA driver) was followed by a Windows bugcheck 0x1E and reboot.** Fixed: plain AdamW (LoRA state ~400 MB). Rule: never force-kill a process stuck in the GPU driver; reboot normally.
  - E2B smoke test (300 conversations, 20 steps, len 1024, QLoRA): works end to end; eval loss 1.774 -> 1.671; checkpoints 387 MB optimizer state; only ~79 tokens/s locally (4-bit + memory spill), so full runs must be rented.
  - `train/export_gguf.py` (merge on CPU -> llama.cpp b11565 converter from `models/llama.cpp-src/` -> llama-quantize Q4_K_M -> Ollama): works. Converter warns "Unknown RoPE type: proportional" and duplicated keys: harmless (released E2B through our pipeline scores 50.2 vs unsloth's 50.6 on the same 500 DarijaMMLU questions, p = 0.90).
  - The 20-step smoke model dropped to 46.4 (p = 0.015): early-training dip plus QLoRA-trained-on-4-bit merged into bf16 weights. Not predictive of the bf16 rented run, but training can hurt: keep checkpoints every ~500 steps and score them locally to pick the best. Smoke model replies to Darija greetings in MSA ("أنا بخير، شكراً لك"), as expected before real training.
  - Remaining before renting: a few local steps on E4B (architecture check), cloud runbook (RunPod), pinned requirements file.
- [ ] A5. **Local training on the small models to find bugs:** training script (LoRA, GemMaroc settings: lr 1e-4, 3 epochs, len 2048), run on Qwen3.5-0.8B and Gemma-4-E2B with a small TULU subset; check loss goes down, merge LoRA, convert to GGUF, load in Ollama, score with the same eval scripts. Fix bugs here, not on rented time.
- [ ] A6. **Rented GPU (RunPod suggested):** train run 2 (GemMaroc recipe: TULU 50K) and run 3 (GemMaroc + AtlasChat hybrid: TULU 50K + AtlasChat 100K), each from the released model, for Qwen3.5-9B and/or Gemma-4-12B-it. Merge, quantize with our own pipeline (also re-quantize the released models the same way so run 1 is strictly comparable), and run Atlas-Chat's harness fork (bf16, per-byte) on all incl. Atlas-Chat-9B for calibration, plus English MMLU/GSM8K and DarijaBench.
- [ ] A7. Score runs 2 and 3 locally exactly like run 1; fill the scoreboard.
- [ ] A8. User's blind ear test on a fixed tutor prompt set (conversation / facts / verbs) via `scripts/compare_models.py`. Choose the model; check it fits next to HADRA on 8 GB.
- GPU provider (none yet; user is open): suggested **RunPod** (per-second billing, PyTorch templates, persistent volume; A100 80GB ~$0.9-1.2/h, H100 ~$2-2.7/h in early-2026 roundups, check live prices). Vast.ai cheaper but less reliable; Lambda stable but pricier. An A100 80GB suffices for 9-12B LoRA in bf16.
- Rough cost (estimate, not measured): 9-12B LoRA on 50K examples x 3 epochs is a few hours on one H100; run 3 about 3x that. Estimated tens of dollars total, not hundreds.

### Track B: workbooks (parallel)
- [ ] B1. Get the Fes workbooks (printed) into git-ignored `sources/workbooks/`. The user has asked the school for PDF copies of updated materials (2026-10-18); scan at 300 dpi only if PDFs are not available.
- [ ] B2. Extract text: if the PDFs have a text layer, extract it directly (check whether the Arabic comes out in the right order and joined); otherwise use OCR or a vision model (printed Arabic is feasible). The user verifies every page. If the books use Latin transliteration, they need a Latin-to-Arabic step (see the source-material note in Progress).
- [ ] B3. Structure the content into the lesson schema (phrases / vocab / dialogues / verbs, with topic, level, and source page).
- [ ] B4. **Hold out ~20% as a Fes gold eval set before any training.** The rescore in step 6 is only honest if this set was never trained on.
- [ ] B5. Build LoRA training pairs from the rest: translation both ways, dialogue continuation, error correction. Expect hundreds to low thousands of pairs; this mainly teaches Fes forms and style, not facts.
- [ ] B6. Semantic layer: embed the entries with a multilingual embedding model (test Darija retrieval quality), store a small local index, and return the top-k entries as prompt context. The 2048-token context limits how much can be added.

### Track C: fact sheets (parallel)
- [ ] C1. Verb schema `facts/verbs/*.yaml`: stem, class, past / present / imperative / future for all persons (ana, nta, nti, huwa, hiya, hna, ntuma, huma), Arabic script + Arabizi + EN, and a `verified` flag.
- [ ] C2. Code generator for regular classes (sound verbs like kteb; derived t- forms like tkellem; quadriliteral verbs), checked by tests against user-verified tables.
- [ ] C3. Irregular verbs listed explicitly, never generated: hollow (kan, gal), final-weak (mša, bda, šra), doubled (ḥebb), initial-weak/hamza (kla, xda, ja). User verifies every form. Present prefix (user, 2026-10-18): Fes uses k- for the continuous present, but it is usually dropped, e.g. knt3lm -> nt3lm. The workbooks will give the exact rules, so the schema should allow both forms.

### Merge
- [ ] 5. Train a small second LoRA on the merged winning model using the Track B data (cheaper than redoing the big run; one more short rental).
- [ ] 6. Rescore: same benchmarks + the workbook holdout + an ear test. Check that general Darija did not regress.
- [ ] 7. Add the semantic layer (retrieval) to `tutor.py`.
- [ ] 8. Agent rules in code: detect intent (conjugation question -> fact sheet; lesson -> YAML; chat -> model). The model only phrases replies around verified data. Model tool-calling at this size is unreliable, so code does the routing.
- [ ] 9. STT + TTS pipeline (Phase 4 STT candidates). Gemma 4 12B accepts audio input, so it could be tested as STT-free voice input if it is the chosen model; Darija audio quality and VRAM are unknown.
- [ ] 10. End-to-end test (voice in, voice out, lessons + discussion).

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
