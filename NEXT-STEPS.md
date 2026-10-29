# Next steps: score run 3 (hand-off 2026-10-28)

For a new chat. Read CLAUDE.md and PLAN.md (Track A, "Run 3 prep" and "Conclusion run 2") first; this file is the short version of where things stand.

## Where we are

Run 3 trained and downloaded: **Gemma-4-E4B-it + LoRA with the Atlas-Chat method** on our own `atlas-tutor` mix (69,957 rows of native Darija: DODa incl. Arabizi reading/writing, MADAR, NLLB-seed, MSMMG, 5K short chat rows; 500 held-out val rows).

- Settings: bf16 LoRA r256 / alpha 128 (558,104,576 trainable params), lr 5e-5, 1 epoch, effective batch 16 (`--batch 8 --accum 2`, gradient checkpointing on), 4,373 steps, 4.53M tokens (1.76M in the loss). H100 80GB on RunPod, ~1 h 05 of training. Batch 16 and batch 8 without checkpointing ran out of memory; 8/2 with checkpointing peaked at 59.5 GB.
- Paused once at step 3033 and resumed from checkpoint 3000 on a migrated pod (the original machine's GPU was taken while stopped); the learning-rate schedule resumed correctly.
- **Eval loss (500 val rows):** 1.332 (250), 1.235, 1.141, 1.116 (1000), 1.073, 1.019, 1.010, 0.978 (2000), 0.967, 0.951, 0.935, 0.922 (3000), 0.913, 0.903, 0.898, 0.896 (4000), 0.895 (4250), **0.895 (final)**. Steady fall, flat over the last ~750 steps, no overfitting. Not comparable with run 2's losses (different eval set).
- Pod terminated after the download was verified.

## Files on the laptop (git-ignored)

`data/train_runs/e4b-atlas/`
- `adapter/` = final (step 4373), `checkpoint-1000`, `-2000`, `-3000`, `-4000`: each `adapter_model.safetensors` is 2,232,496,800 bytes, 516 tensors, header size checked (complete).
- `checkpoint-4250/trainer_state.json` (full eval history up to 4250), `summary.json` (final eval loss 0.8953).
- `data/train_runs/e4b-atlas.log`: training log (both parts; the `done:` line's tokens/s is overstated after the resume, known bug).
- Not downloaded (gone with the pod): checkpoints 250-750, 1250-1750, 2250-2750, 3250-3750, 4250 adapters, all optimizer states.

## What to do next

### 1. Export each adapter to GGUF (Q4_K_M) and load it in Ollama
Before **every** export: no model loaded in Ollama (`ollama ps` empty; `ollama stop <model>`), HADRA server stopped. Two bugchecks so far came from GPU memory pressure (see CLAUDE.md). Never force-kill a process stuck in the GPU driver.

```powershell
.venv-train\python.exe train\export_gguf.py --base google/gemma-4-E4B-it --adapter data\train_runs\e4b-atlas\adapter --name e4b-atlas-final
.venv-train\python.exe train\export_gguf.py --base google/gemma-4-E4B-it --adapter data\train_runs\e4b-atlas\checkpoint-1000 --name e4b-atlas-1000
```
(`-2000`, `-3000`, `-4000` the same way.) The base model is in the local HF cache; disk has ~314 GB free.

### 2. Score, one model on the GPU at a time
Same scorers and full sets as every earlier model. Pass `--note` on each run (the question it answers).

```powershell
.venv\Scripts\python eval\darija_mmlu.py --model e4b-atlas-final --n 0 --note "run 3 (Atlas method, atlas-tutor mix, E4B) final; compare with gemma-4-e4b-ours"
.venv\Scripts\python eval\darija_hellaswag.py --model e4b-atlas-final --note "..."
.venv\Scripts\python eval\fluency.py --model e4b-atlas-final --prompt v1 --note "..."
.venv\Scripts\python eval\fluency.py --model e4b-atlas-final --prompt v2 --parts tutor --note "..."
```
Run 2 timings per model: DarijaMMLU ~33 min, HellaSwag ~18 min, fluency a few minutes. Suggested order: **final, then 1000, then 3000**; add 2000 / 4000 only if the trend between them is unclear.

Then `.venv\Scripts\python eval\scoreboard.py` and `eval\report.py` (update STATUS / KEY_POINTS; the user prefers `data/report.html` over the CSVs).

### 3. Compare against the bars

| Model | DarijaMMLU | HellaSwag | Fluency tutor (v1) | Arabizi chrF |
|---|---|---|---|---|
| gemma-4-e4b-ours (released E4B, our pipeline) | 59.55 | 49.86 | 76% (MSA 6%) | 41.1 |
| e4b-tulu-1000 (run 2 best) | 58.31 | 50.79 | 73% | 33.1 |
| atlas-chat-9b (current tutor) | 58.42 | 58.59 | 80% (MSA 6%) | 67.7 |

What counts (from run 2's conclusion): **fluency and the user's ear first; DarijaMMLU is a guardrail (lose less than ~1 point vs 59.55)**. Arabizi: expected to recover or improve (the mix has DODa ar_dr / dr_ar), but note it is partly in-distribution (DODa-style sentences; exact test sentences were removed from the mix). Check the bare-prompt Darija share too (run 2 raised it 16% -> 86%).

### 4. Ear test
`eval/ear_replay.py --model e4b-atlas-<best> --prompt v2 --seeds 0 1 2`, then the user's own cafe / medina / fes chat and a free conversation, against `gemma-4-e4b-ours` and `atlas-chat-9b`. This feeds roadmap step 9 (base = Atlas-9B or E4B), which stays open until this run is ear-tested.

### 5. Decide on a second epoch (only after scoring)
Eval loss was flat at the end, so a second epoch is not obviously worth it. Only consider it if run 3 clearly beats the released E4B on fluency/ear and the checkpoints show later is better. A 2-epoch run is a fresh run (`--epochs 2`, ~8,750 steps, ~2 h 45 on an H100, ~$10), not a continuation (the cosine schedule already ended). Settings and the pause/migrate lessons are in `train/RUNPOD.md`.

### 6. Record
Update PLAN.md (Track A, step 10 and the roadmap) and CLAUDE.md "Now" / "Results" with the scores; commit.
