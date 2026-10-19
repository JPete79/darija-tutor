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

### Caveats before scoring new model families (Qwen3.5, Gemma 4)

Checked 2026-10-18; to verify with small sibling models (Qwen3.5-0.8B, Gemma-4-E2B) before the big downloads:

1. **Hybrid architectures and llama.cpp HellaSwag (needs a test).** llama.cpp b11565 supports the `qwen35` and `gemma4` architectures. Its HellaSwag scorer evaluates the shared context once and branches into the 4 endings (one token belonging to several sequences). That is routine for Transformers (Atlas-Chat, Gemma), but Qwen3.5 is a hybrid with recurrent Gated DeltaNet layers, where this may be unsupported, slow, or silently wrong. Test: compare its llama-perplexity answers with scoring each ending separately.
2. **Per-token normalization depends on the tokenizer.** Our HellaSwag score averages log-probability per token; Gemma and Qwen split Darija into tokens differently, so a Qwen-vs-Gemma comparison carries a tokenizer effect (the published per-byte normalization does not). Within one family (runs 1 -> 2 -> 3, same tokenizer) it is fair. For cross-family comparisons use DarijaMMLU (single letter, no length normalization) and the per-byte harness numbers from the rented GPU.
3. **Quantization pipelines differ.** unsloth's GGUFs use calibrated (imatrix) quantization; Atlas-Chat's mradermacher GGUF is static. Small effect; removed later by quantizing every model through our own pipeline.
4. **DarijaMMLU letter reading.** If a tokenizer emits the space and the letter as separate tokens, letters will not appear as the first token: watch `no_answer` (it must stay ~0).

**One model at a time on 8 GB:** running jobs on two different models makes Ollama swap them on every request and both runs stall. Stop the HADRA voice server during evals: llama-perplexity with the 9B fills the GPU by itself (7.9 GB), and with HADRA loaded it spilled into system memory and ran at half speed (~80 vs ~170 questions/min). Even alone, the 9B at `--ctx 2048` spilled 1.5 GB into system RAM and slowed to ~65/min (signs: GPU "100%" busy but drawing ~40 W; Windows counter `GPU Adapter Memory\Shared Usage` > 0 on the NVIDIA adapter). `--ctx 1024` fits (7.5 GB) and runs ~300/min; HellaSwag questions are short, so scores are unaffected. Use `--ctx 1024` for 9B+ models.
