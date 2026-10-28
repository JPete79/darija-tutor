# Evaluation

Learning-first benchmark scripts (PLAN.md Phase 2b, track A). Results go to `data/eval_runs/` (git-ignored).

## Scoreboard: `eval/scoreboard.csv` (start here)

One row per model variant: vendor, base model, size, the Darija training data added, and for each benchmark the published score next to ours. It holds the reference models (Atlas-Chat, GemMaroc, untrained Gemma), and the planned grid (Qwen3.5-9B and Gemma-4-12B-it x released / + TULU 50K / + TULU 50K + AtlasChat 100K).

- Add or edit rows by hand: one row per new model or training variant; fill `ollama_model` once it runs locally.
- `.venv\Scripts\python eval\scoreboard.py` fills the `*_ours` columns from the experiment log: the latest complete **full-set** run in the standard setting (logprob, raw, 0-shot). 500-question checks never reach the scoreboard.

## Experiment log: `eval/experiments.csv`

Every run appends one row automatically (finished, stopped with Ctrl+C, or crashed): when, model + Ollama digest + quantization, every setting, accuracy with 95% CI, the published score, the code commit, the per-question results file, and a `note`. Open it in Excel, or print it with `.venv\Scripts\python eval\registry.py`.

- Pass `--note "..."` on every run: the question it answers or the hypothesis it tests. Add the conclusion to the note afterwards by hand.
- Result files are timestamped, so a rerun never overwrites an earlier one.
- The CSV is committed to git; the per-question files in `data/eval_runs/` are not (they are large and can be regenerated).
- Runs from before the log existed were added with `registry.py --backfill`, so their `started` time is the file's time.

## DarijaMMLU on local Ollama models

```
.venv\Scripts\python eval\darija_mmlu.py --model atlas-chat-9b --n 500
.venv\Scripts\python eval\compare_runs.py data\eval_runs\<run A>.jsonl data\eval_runs\<run B>.jsonl
```

`darija_mmlu.py` uses the same prompt as Atlas-Chat's lm-evaluation-harness fork: question, lettered options, then `الجواب:` ("The answer:"). Each run writes one row per question (`.jsonl`) and a summary (`.json`):

| Field | Meaning |
|---|---|
| `acc`, `ci95` | Accuracy and its 95% confidence interval. On 500 questions the interval is about ±4 points. |
| `chance` | Expected score from random guessing. Questions have 2-5 options, so this is ~28%, not 25%. |
| `pred_letters` vs `gold_letters` | Position bias: does the model favour some letters? |
| `by_source` | `arabic_mmlu` (Arab-world school exams) vs `mmlu` (translated US MMLU) |
| `no_answer` | Questions where no valid letter was produced or found in the top 20 tokens |

Change one setting at a time, keeping `--n` and `--seed` fixed so every run answers the same questions:

| Setting | What it tests |
|---|---|
| `--scoring logprob` vs `generate` | Reading the model's letter probabilities vs letting it write and parsing the answer |
| `--template raw` vs `chat` | Plain prompt (what the harness does) vs the model's chat format (how the tutor uses it) |
| `--shots 0..3` | Solved examples before the question. Long prompts can exceed `num_ctx` (2048 for the 9B); add `--num-ctx 4096` with the TTS server stopped. |
| `--seed` | A different random 500: how much does the score move from the sample alone? |

`compare_runs.py` counts the questions where only one run was right and gives McNemar's p-value. With p < 0.05 the gap is unlikely to be sampling noise.

## Results so far

Setting: Q4_K_M in Ollama, `--scoring logprob --template raw --shots 0 --n 500 --seed 0` (2026-10-18).

| Model | Questions | Ours (95% CI) | Published (full set, zero-shot) |
|---|---|---|---|
| Atlas-Chat-2B | 500 | 44.4 (40.1-48.8) | 44.97 |
| Atlas-Chat-9B | 500 | 64.0 (59.7-68.1) | 58.23 |
| Atlas-Chat-9B | all 21,792 | **58.4 (57.8-59.1)** | 58.23 |

The 500-question 9B score was a lucky sample: drawing 10,000 random 500-question samples from the full run gives scores from 50.0 to 66.6 (middle 95%: 54.0-62.6), and only 0.5% reach 64. On the full set the 9B matches the published number, so our method reproduces both models. Lesson: use the full set (about 65 min for the 9B) for any number you compare; use 500-question runs only for quick checks, and compare those with `compare_runs.py` on the same questions.

Other things ruled out while chasing the gap: the dataset re-upload of 2024-09-27 (changed 95 answers and 208 choice lists, ≤1.4 points), BOS token handling (both add it), prompt changes (none after Aug 2024), and running two jobs at once (same answers on all 500 questions). With the chat template the 9B answers "A" without the leading space, so the harness's strict " A" scoring would behave differently under a chat template; our loose scoring gave 64.8 vs 64.0 raw on the 500 sample. The 2B with the chat template: 46.4 vs 44.4 raw (McNemar p = 0.19, so no real difference shown); with the chat template it also answers "A" without the space (strict score 0).

Full-set details (9B): `arabic_mmlu` 60.1, `mmlu` 55.2; under-predicts D (3,173 predicted vs 4,963 gold) and over-predicts B (6,826 vs 5,643).

### DarijaHellaSwag (`darija_hellaswag.py`, llama.cpp per-token acc_norm, full validation set)

| Model | Ours (95% CI) | Published (harness, per-byte acc_norm) |
|---|---|---|
| Atlas-Chat-2B | 42.66 (41.70-43.63) | 35.08 |
| Atlas-Chat-9B | 58.59 (57.63-59.55) | 43.65 |

Our numbers are well above published on the full set, so this is a method difference, not sampling: likely llama.cpp's per-token normalization vs the harness's per-byte one, or the card reporting plain `acc`. Fine for comparing our own runs with each other (same scorer for every model); the harness run on a rented GPU will calibrate against published numbers later.

### Candidates as released (full sets, 2026-10-19)

| Model | DarijaMMLU | DarijaHellaSwag (per-token) |
|---|---|---|
| Atlas-Chat-9B | 58.42 | 58.59 |
| Qwen3.5-9B | 61.05 (60.4-61.7) | 46.10 (45.1-47.1) |
| Gemma-4-E4B-it | 60.09 (59.4-60.7) | 49.97 (49.0-51.0) |

DarijaMMLU compares fairly across families (one letter, no length normalization); the HellaSwag column does not (see caveat 2), and neither measures whether a model *writes* Darija rather than MSA.

### Caveats before scoring new model families (Qwen3.5, Gemma 4)

Checked 2026-10-18; to verify with small sibling models (Qwen3.5-0.8B, Gemma-4-E2B) before the big downloads:

1. **Hybrid architectures and llama.cpp HellaSwag (tested OK 2026-10-19, see below).** llama.cpp b11565 supports the `qwen35` and `gemma4` architectures. Its HellaSwag scorer evaluates the shared context once and branches into the 4 endings (one token belonging to several sequences). That is routine for Transformers (Atlas-Chat, Gemma), but Qwen3.5 is a hybrid with recurrent Gated DeltaNet layers, where this may be unsupported, slow, or silently wrong. Test: compare its llama-perplexity answers with scoring each ending separately.
2. **Per-token normalization depends on the tokenizer.** Our HellaSwag score averages log-probability per token; Gemma and Qwen split Darija into tokens differently, so a Qwen-vs-Gemma comparison carries a tokenizer effect (the published per-byte normalization does not). Within one family (runs 1 -> 2 -> 3, same tokenizer) it is fair. For cross-family comparisons use DarijaMMLU (single letter, no length normalization) and the per-byte harness numbers from the rented GPU.
3. **Quantization pipelines differ.** unsloth's GGUFs use calibrated (imatrix) quantization; Atlas-Chat's mradermacher GGUF is static. Small effect; removed later by quantizing every model through our own pipeline.
4. **DarijaMMLU letter reading.** If a tokenizer emits the space and the letter as separate tokens, letters will not appear as the first token: watch `no_answer` (it must stay ~0). Checked: both families emit `" A"` as one token.

### Small-model check (A2, 2026-10-19)

- **Scorer bug (fixed):** a whitespace top token (`"

"`, stripped to `""`) counted as a letter because `"" in "ABCDE"` is True in Python. Qwen3.5 ranks `"

"` first on about a third of questions, so 161/500 predictions were blank, and `no_answer` hid it (it counted only `None`). Atlas-Chat almost never does this (9B full set: 0; 2B: 7 of 21,792), so the baselines stand.
- **`hellaswag_crosscheck.py`:** per-question cross-check of llama.cpp's HellaSwag. Engine `llamacpp` runs one question per llama-perplexity call (multi-question files are shuffled, so per-question answers are lost otherwise); engine `hf` scores every ending as a separate sequence with the original weights in transformers (run it with `.venv-hadra\python.exe`, the only env with torch); `compare` gives agreement and McNemar. A plain Transformer (Gemma) is the control for the Q4-vs-bf16 difference.

| Model (300 questions) | llama.cpp Q4 | transformers bf16 | Same verdict |
|---|---|---|---|
| Gemma-4-E2B-it (control) | 41.7 | 43.0 | 93.3% (p = 0.50) |
| Qwen3.5-0.8B (hybrid) | 31.7 | 32.0 | 96.3% (p = 1.00) |

  The hybrid agrees at least as well as the control, so llama-perplexity's shared-prefix branching is fine for Qwen3.5.
- **Gemma 4 needs `<bos>`:** its HF tokenizer does not add it, llama.cpp does. Without it the first control run scored 29% and greedy text degenerated ("France is France is..."). Lesson: when two scorers disagree, check token ids first (`llama-tokenize.exe --ids` vs the HF tokenizer).
- First small-model scores (500 questions, ±4): DarijaMMLU Qwen3.5-0.8B 30.0, Gemma-4-E2B-it 50.6 (Atlas-Chat-2B 44.4 on the same questions); DarijaHellaSwag 28.6 / 39.4.

**One model at a time on 8 GB:** running jobs on two different models makes Ollama swap them on every request and both runs stall. Stop the HADRA voice server during evals: llama-perplexity with the 9B fills the GPU by itself (7.9 GB), and with HADRA loaded it spilled into system memory and ran at half speed (~80 vs ~170 questions/min). Even alone, the 9B at `--ctx 2048` spilled 1.5 GB into system RAM and slowed to ~65/min (signs: GPU "100%" busy but drawing ~40 W; Windows counter `GPU Adapter Memory\Shared Usage` > 0 on the NVIDIA adapter). `--ctx 1024` fits (7.5 GB) and runs ~300/min; HellaSwag questions are short, so scores are unaffected. Use `--ctx 1024` for 9B+ models.

## Fluency check (`fluency.py`, 2026-10-24)

DarijaMMLU and HellaSwag only read probabilities; this check makes the model *write*, then asks whether what it wrote is Moroccan Darija, MSA or English.

```
.venv\Scripts\python eval\fluency.py --model gemma-4-e4b-ours --note "why"
```

- **tutor**: the 30 prompts in `fluency_prompts.jsonl` after the tutor's real system prompt. v2 (current): 10 English, plus 10 sentences from the Peace Corps Moroccan Arabic textbook (2011), each in Arabic script and in Arabizi (same sentence, so the script is the only difference); the `pc` field gives the book page and its original transcription. Keep these sentences out of any later Peace Corps translation test. v1 (`fluency_prompts_v1.jsonl`, self-written) was used for one run only.
- **bare**: the 20 Darija/Arabizi prompts with no system prompt (the model's own habit).
- **translate**: 200 DODa (everyday) + 200 FLORES+ (formal) English->Darija sentences from DarijaBench's test splits, with DarijaBench's Darija instruction plus "write only the translation"; chrF against the reference.
- Every Arabic-script sentence goes through `dialect_scores.py` (run in `.venv-train`, CPU): ALDi (0 = MSA ... 1 = dialect) and a dialect router (`ma` = Moroccan, `ar` = MSA, ...). Checked on 300 DarijaBench MSA/Darija pairs: MSA side ALDi 0.01-0.04, router `ar` 74-80%; Darija side ALDi 0.46-0.74, router `ma` 62-75%. About 1 sentence in 4 is mislabelled, so read the shares over many sentences, never one reply.
- Logged per part as `fluency_tutor`, `fluency_bare`, `translate_doda`, `translate_flores` (`acc` = Moroccan share; chrF, MSA share, ALDi and Latin-script share in the note). Replies to read by ear: `data/eval_runs/<run>.md`.
- Greedy decoding with `think: false`: without it Gemma 4 spends the whole token budget on hidden reasoning and returns an empty reply.
- With 30 prompts the 95% intervals are about ±15 points: only large differences count. The prompt set is v1; changing it makes later runs incomparable with earlier ones.
- **Two different "v1/v2"s:** `--prompts` picks the *learner* prompt set (above); `--prompt` picks the *tutor system prompt* (`prompts/discussion_system_<v>.md`, default v1 = the prompt of all run 2 scores). Tutor rows are logged with template `tutor_system` (v1) or `tutor_system_<v>`; compare tutor scores only within one template.
- Shorter replies shift the sentence mix: under tutor prompt v2 chat replies have 1-2 sentences, so the list prompts (numbers, conjugations: single words and MSA pronouns the router often labels MSA) make up a bigger share of all sentences. Read the per-prompt `.md` before trusting a change in the MSA share.

## Ear replay (`ear_replay.py`, 2026-10-28)

Sends the scripted ear-test conversations (`ear_test.md`: cafe, medina, fes) to the real tutor without typing, to compare prompt versions or models on identical learner lines. Text only, temperature 0.6 as in the app, a seed per replay (Ollama on the GPU is not fully deterministic, so a rerun is close but not identical; use several seeds).

```
.venv\Scripts\python eval\ear_replay.py --model gemma-4-e4b-ours --prompt v1 v2 --seeds 0 1 2
```

Writes `data/ear_runs/<time>_<model>.md` (every reply in Arabic, Arabizi and EN, flagged) and `.json`. The counts per reply flag things to read, not a score: sentences, replies with more than one question, the "not sure" remark (should appear only on the factual question, fes line 3), Arabic words in brackets (template placeholders), and context tokens used (num_ctx 2048). Not logged in `experiments.csv`; the user's ear stays the judge.

