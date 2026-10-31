# Next steps: after run 3 (hand-off 2026-10-31)

For a new chat. Read CLAUDE.md and PLAN.md (Roadmap, and Track A "Run 3 results and conclusion") first; this file is the short version of where things stand and what is open.

## Where we are

**Run 3 is scored and ear-tested; it is not going forward as the tutor.** Released E4B (`gemma-4-e4b-ours`) is the working base for now, with prompt v3.

| Model | DarijaMMLU | HellaSwag | Tutor Darija v1 / v2 (MSA) | Bare | DODa chrF | Arabizi chrF |
|---|---|---|---|---|---|---|
| e4b-atlas-final (run 3) | 60.10 | 51.33 | 66% / 68% (19% / 22%) | 83% | 42.5 | 50.9 |
| gemma-4-e4b-ours (released) | 59.55 | 49.86 | 76% / 74% (6% / 21%) | 16% | 28.9 | 41.1 |
| e4b-tulu-1000 (run 2 best) | 58.31 | 50.79 | 73% | 86% | 30.8 | 33.1 |
| atlas-chat-9b | 58.42 | 58.59 | 80% / 79% (6% / 5%) | 80% | 43.5 | 67.7 |

- **Scores:** no knowledge lost, translation into Darija at Atlas level, Arabizi back and better. Tutor share slightly lower (within noise on 30 prompts), more MSA, the "not sure" remark on non-fact questions.
- **User's ear (free chats, 2026-10-29):** run 3 breaks the conversation. It analyses the learner's words instead of answering ("khdema" -> "did you mean khdama?"), turns "how do you say AI in Darija" into a Fusha question, breaks persona ("I'm your assistant", masculine forms for Noual), and the format collapses after a misunderstanding. E4B-ours: smoother conversation, but more MSA (أنني، في، دائما), long replies, broke persona once ("I'm just a program") and invented a wrong correction (صعيبة -> صعيب). **User verdict: E4B-ours is smoother; none of our trained models holds a natural conversation.**
- **Cause (checked, not a pipeline bug):** same chat template for both models in Ollama; training rendered rows with Gemma 4's own template, one `<bos>`, no thinking block, loss on replies only. The `atlas-tutor` mix was ~93% one-turn "convert this text" rows (median answer ~9 tokens) and 7% single-turn chat, no system prompts. With r256 (558M params) on that narrow mix, the model learned "a user message is text to process; answer briefly".
- **The rule both runs taught us:** fine-tuning keeps only the skills that are in the mix. Run 2 (no Arabizi in TULU) lost Arabizi; run 3 (almost no conversation) lost conversation. Atlas-Chat-9B kept both because its full mix is ~35% chat (161K TULU-Darija rows, 19K multi-turn ShareGPT) plus 17K transliteration rows.

## Why the evals missed it (fix before any new run)

- Every tutor prompt is **turn 1**; the failures came at turns 2-4 (follow-up, misunderstanding, correction).
- The scores measure **which language** (Darija share, chrF), not **whether the reply answers the learner**. Rule counts (EN lines, remark) only check format.
- Warning signs were there and under-weighted: tutor 76 -> 66%, MSA 6 -> 19%, en02 answered with a counter-question.
- **Bare-prompt Darija is a drift gauge, not a quality score:** the app always sends the tutor prompt. 16% -> 83% meant training pushed the model's defaults hard; that same push changed its conversation style. For a gentle run, expect a smaller jump (~40-60%); 85% again means the LoRA is still too strong.

**To build: multi-turn conversation check** (part of a "skill panel" every candidate must pass with nothing dropping vs released E4B): scripted 3-5-turn conversations from the user's transcripts, each turn with a written expectation (asked her job -> names a job, stays in persona; "s3iba bzef" about lkhedma -> no correction; "how do you say X in Darija" -> gives the word, no Fusha detour). Automatic checks for persona breaks, self-gender, remark on non-fact turns, length; a local judge model (e.g. Qwen3.5-9B) for "answered what was asked?" only if it agrees with the user's blind ratings on 20-30 replies; the user's blind A/B as the final word.

## Prompt findings (2026-10-29)

- **v3** (`prompts/discussion_system_v3.md`): v2 + "repetition drill" allowed in long replies + a rule to say/repeat something exactly N times. Measured (temperature 0.6, 10 seeds, run 3 model): v2 17/20 vs v3 17/20, so not better; the model fails ~1 in 6 either way and then copies its own one-line pattern. Reliable repetition = the CLI's `/repeat [words|sentences]` and `/slow`, or code that plays a line N times. v3 is **not** the app default yet (still v2); run the 30-prompt tutor check on it before switching.
- **Disposable test, tutor prompt in Darija** (`discussion_system_xdar.md` = all rules in Darija, `_xhyb.md` = Darija persona paragraph only; logged as "DISPOSABLE" in `eval/experiments.csv`): hybrid did nothing; all-Darija raised the Darija share on all three models (run 3 68 -> 79%, E4B 74 -> 76%, Atlas 79 -> 90%, wide intervals) but rules got worse: the "not sure" example line was copied verbatim into everyday questions (E4B 1 -> 6 of 28, Atlas 1 -> 4) and Atlas answered "where is the station?" with the prompt's Eiffel example. **Keep rules in English; put Darija into the content** (topic packs, scenario cards).
- App context is still `num_ctx 2048` for every model (set by `export_gguf.py`); E4B at 16K costs ~240 MB more (measured 2026-10-28). An 8-10-turn chat reaches the limit. Make a 16K Ollama entry for chatting; evals keep 2048.

## Game plan (agreed in discussion 2026-10-30/31; user reviewing with a fresh agent)

Layers: **model** = how she speaks (flow, Darija, format, corrections); **lookup / semantic layer** = what she knows (verbs, facts, vocab, dialogues, stories); **code** = modes and which source a turn uses.

1. **Base now:** `gemma-4-e4b-ours` + prompt v3 (after its tutor check), 16K context. Run: `.\talk.cmd --model gemma-4-e4b-ours --prompt v3`.
2. **Topic packs from the Peace Corps book** (`data/sources/peace-corps-2011/`, git-ignored). Every entry is printed in three columns: **English | Peace Corps transcription | Arabic script**. The Arabic is in the PDF but not extractable as text (pypdf gets only English + transcription). Plan: build a machine-readable chart from the book's own transcription table (pp. 1-3 + "New Sounds" p. 3: š ش, x خ, ġ غ, ḥ (extracts as "н") ح, ع, q ق, ḍ ṣ ṭ, g گ, doubled letter = shedda) with a chat-Arabizi column (ch, kh, gh, 7, 3, 9); get the Arabic column by OCR of rendered pages (Tesseract-ara or a vision model in Ollama; our E4B builds have no vision) **and** by converting the transcription with the chart; auto-accept where both agree, the user checks only disagreements. Pages render with `pypdfium2` + `pillow` (added to `.venv`; PyMuPDF crashed with an access violation and was removed).
   - Content: ~15 scenes (greetings, nationalities, hanoot, directions, bargaining, clothing, produce, food, health, travel, hotel, post office, taxi, renting a house), past/present/future/imperative + a 30-page verb glossary (pp. 163-192), grammar, topic vocabulary, holidays. Skip the Peace Corps mission and security chapters.
   - **Pilot: "At the Hanoot" (book p. 34 = PDF page 39)**, then "Medical & Body" (pp. 97-100).
   - Watch: general Moroccan, not Fes (the Fes corrections list overrides); the 10 tutor test sentences from the book (`eval/fluency_prompts.jsonl`, field `pc`) stay out of every pack and training set; confirm public-domain status before committing extracted content (public repo).
3. **Pack schema** (shared by the Peace Corps chapters and the Fes school workbooks, which have the same shape: 1-2 dialogues of 15-20 turns + vocabulary + stories per unit, e.g. body = doctor dialogue, body parts, sicknesses, stories on Moroccan hospitals and herbal medicine): `topic, level, sources, dialogues[turns{speaker, ar, translit, en}], vocab[{ar, translit, en, group}], expressions, stories[{text, questions}], notes, fes_overrides`. The Fes book wins where sources differ. Code injects only the part the current mode needs. School workbooks are private: keep them and anything extracted out of the repo; a model trained on them is not published.
4. **Pack test:** E4B-ours + v3 **with vs without** the hanoot pack, same scripted multi-turn hanoot conversations, several seeds (Atlas-9B optional third). Measure: uses the pack's words, stays in scene, Darija share/MSA, no forms contradicting the pack, multi-turn checks; user's blind A/B decides.
5. **Build out from the result:** remaining chapters, verb tables generated by code from the glossary (never trust free-chat conjugation), Fes fact sheet, semantic search for facts (with a ~50-question retrieval test set in English/Darija/Arabizi incl. questions where nothing should match; measure hit rate and false hits). Routing: explicit modes first, then rules/exact matches (verbs, scenes), embeddings only for facts.
6. **Teaching modes** (PLAN.md "Target modes"): (4) framed dialogues with word swaps (the Peace Corps scenes / scenario cards), (2) quiz on verbs and facts, (3) story read aloud + questions (listening), (5) topic teacher, (1) free conversation. Most structured first: they need no training.
7. Alongside: **chat logging** (the CLI saves nothing today; every chat is future training data), git-ignored under `data/`.

## Training: open decisions for the user

- **Run 4 (candidate, user leaning towards it):** restore conversation by matching **Atlas-Chat's full-mix proportions** at a smaller size, not the `atlas-tutor` slice. An exact copy (458K rows x 3 epochs, ~600M tokens) is several days on an H100 (~$150-400): not practical, and no overnight runs. Instead ~100K rows in Atlas proportions (GemMaroc's `AtlasChat-data-100k-randomlysampled-darija` is exactly a random 100K of the mixture, ~1/3 chat; ~44M tokens) minus `hard_coded` (MBZUAI identity) and `*_few_shot` (literal tags); optionally more short multi-turn ShareGPT (19,193 multi-turn conversations in the mixture, only 1,100 with every answer <= 60 words; e.g. a 20-questions game in natural Darija; many are long or MSA-ish translations). Plus **E4B's own tutor replies, filtered for Darija by `eval/dialect_scores.py`** (self-distillation: keeps E4B's tone). Gentler: LoRA r16-64, one pass, checkpoints scored on the skill panel.
- **Do first (free, local):** export checkpoint 1000 and **run 3 at half strength** (`lora_alpha` 128 -> 64 in a copy of `adapter_config.json`). If either chats more smoothly while keeping most of the Darija gain, the drift explanation holds and a gentle run 4 is well aimed.
- **Teacher flows in training or only in the prompt + semantic layer? Not decided.** Current position: content (verbs, facts, vocabulary, dialogues, stories) always lives in packs / lookup, never trained as memory (trained facts blur and cannot be corrected; run 3 still conjugated wrong). Teaching method comes from the prompt + code modes first. Train on teacher flows only if the pack test shows gaps the prompt cannot fix, and then train the *skill of using a pack*: rows = tutor prompt + a pack in the system prompt + multi-turn dialogue that uses it (from Peace Corps / Fes dialogues, drafted, user checks a sample); hold out whole units (~4 chapters / 20% of workbooks) to test transfer to unseen packs.
- RunPod: credit is left on the account (check the balance); runbook `train/RUNPOD.md`.

## Uncommitted state at hand-off

`NEXT-STEPS.md`, `PLAN.md`, `CLAUDE.md`, `eval/report.py` (STATUS / KEY_POINTS), `eval/experiments.csv` (run 3 + disposable rows), `prompts/discussion_system_v3.md`, `prompts/discussion_system_x{dar,hyb}.md` (disposable, kept so the logged rows can be reproduced). `.venv` gained `pypdfium2` and `pillow`. Exported: `e4b-atlas-final` in Ollama; checkpoints 1000-4000 not exported.
