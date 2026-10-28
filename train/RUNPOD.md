# Training on a rented GPU (RunPod)

Runbook for PLAN.md Track A, step A6: LoRA-train Gemma-4-E4B-it on the GemMaroc data.
- Run 2: `--data tulu` (TULU 50K, ~137M tokens over 3 epochs)
- Run 3: `--data-file data/train_mix/atlas-tutor.jsonl` (Atlas-Chat method on a 70K-row native Darija mix; Part 6)

Both runs start from the released model. They use bf16 LoRA (no `--qlora`) with the script's GemMaroc defaults.

The money rule: **a pod bills every second it exists, even when idle.** Start the pod, then do the 15-minute check before anything else. Download your results and **terminate** the pod when you're done.

---

## Part 1. Account (one time, ~10 min)

1. Sign up at https://www.runpod.io.
2. **Billing**: add credit. RunPod is prepaid, so the credit you load is your spending cap. **$25** covers the check plus run 2 with room to spare. Add more for run 3 after you see the measured speed.
3. Optional: under **Settings**, turn on low-balance email alerts.

## Part 2. Start a pod (~5 min)

1. Go to **Pods** and click **Deploy**.
2. **GPU**: pick **H100 80GB** (PCIe, SXM or NVL are all fine) or **A100 80GB**, whichever is available. Note the $/hour shown. An H100 costs about 2x per hour but runs about 2x faster, so the total cost is similar; the H100 just finishes sooner. Choose **Secure Cloud** (more reliable than Community Cloud for a first run).
3. **Template**: a "RunPod PyTorch" template (any recent 2.x). The script installs its own pinned versions anyway.
4. Click **Edit Template** and set:
   - **Container Disk**: 30 GB (temporary, wiped when the pod stops)
   - **Volume Disk**: 100 GB, mounted at `/workspace` (kept while the pod exists; holds the model, data and checkpoints)
   - Leave **Jupyter** enabled (port 8888)
5. **Deploy On-Demand** (not Spot: a Spot pod can be taken away mid-run).
6. Wait until the pod shows **Running**, then click **Connect** and **Jupyter Lab**. In Jupyter: **File, New, Terminal**.

Everything below is typed into that terminal.

## Part 3. Install (~5-10 min)

```bash
cd /workspace
git clone https://github.com/JPete79/darija-tutor.git
cd darija-tutor
export HF_HOME=/workspace/hf          # model + dataset cache on the kept volume
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
pip install -r train/requirements.txt
pip uninstall -y torchvision torchaudio  # the image's copies are built for its older torch and break transformers; we train text only
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `2.14.1... True NVIDIA H100...` (or A100). If it prints `False`, stop and send me the output.

## Part 4. The 15-minute check (do not skip)

This is a 20-step run on 2,000 conversations. It also saves a checkpoint at step 10 and step 20.

```bash
mkdir -p /workspace/runs
python train/train_lora.py --model google/gemma-4-E4B-it --data tulu --limit 2000 \
  --max-steps 20 --save-steps 10 --batch 4 --accum 4 --out /workspace/runs/check
```

The first time, it downloads the model (~16 GB) and the data (a few minutes). Check these things:
- `sample start:` begins with exactly **one** `<bos>`.
- `loss starts at:` shows the start of an assistant reply. It should not show the user's question.
- `trainable params: ~69.8M` (same as the laptop smoke test).
- The loss lines (`{'loss': ...}`) should not show `nan`. On 20 steps the loss should go down a little.
- `checkpoint-10` and `checkpoint-20` appear in `/workspace/runs/check`.
- The last line should look like: `done: 20 steps in X min, ~N tokens/s; eval loss ...; peak GPU memory M GB`

**Send me that last line plus the pod's $/hour.** From that I will work out the time and cost of each run, and whether `--batch 8 --accum 2` fits (which is faster). The cost math is simple: hours = 137M tokens / (tokens/s x 3600), and cost = hours x $/hour.

Why `--batch 4 --accum 4` instead of the defaults (`--batch 1 --accum 16`)? Both settings give an effective batch of 16, so the optimizer sees the same thing as GemMaroc's recipe. Batch 1 was for the 8 GB laptop. On an 80 GB card it would leave the GPU mostly idle, which wastes paid time.

If the check fails with "CUDA out of memory", rerun it with `--batch 2 --accum 8`.

## Part 5. Run 2

```bash
cd /workspace/darija-tutor
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train/train_lora.py --model google/gemma-4-E4B-it --data tulu \
  --batch 2 --accum 8 --no-checkpointing --save-steps 500 --keep 0 --out /workspace/runs/e4b-tulu \
  > /workspace/runs/e4b-tulu.log 2>&1 &
tail -f /workspace/runs/e4b-tulu.log      # Ctrl+C stops watching, NOT the training
```

- **Auto-stop (recommended for overnight runs):** a pod has no idle shutoff, so run the training through `sh -c '...; runpodctl stop pod $RUNPOD_POD_ID'`. When training ends (finished or crashed), the pod stops itself and the volume is kept. Test first with `echo $RUNPOD_POD_ID; runpodctl get pod`. Example:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup sh -c 'python train/train_lora.py --model google/gemma-4-E4B-it --data tulu --batch 2 --accum 8 --no-checkpointing --save-steps 500 --keep 0 --out /workspace/runs/e4b-tulu > /workspace/runs/e4b-tulu.log 2>&1; runpodctl stop pod $RUNPOD_POD_ID' > /dev/null 2>&1 &`
- Settings measured on the H100 (2026-10-23): `--batch 2 --accum 8 --no-checkpointing` ran at ~3,100-3,600 tokens/s with 65 GB peak memory, vs 2,655 for the check's settings. `--batch 4` without checkpointing ran out of memory. Expect run 2 to take ~11 h (~$44 at $4.01/h). Run 3 (Part 6) is ~4.5M tokens, ~1/30 of run 2: under an hour of training.
- `nohup ... &` keeps the run going if you close the browser or Jupyter.
- `--keep 0` keeps every checkpoint (~every 500 steps, about 17 in total). We score several of them, because training can make the model worse after a point.
- To check on the run later, open a terminal and run `tail -n 20 /workspace/runs/e4b-tulu.log`. For GPU use, run `nvidia-smi`.
- Each 500 steps the log prints an `eval_loss`. It should go down, then flatten out. If it climbs steadily, tell me.

**Pausing (to avoid running overnight):** stop the process, then the pod.
1. Run `pkill -f train_lora.py` to stop the process.
2. Click **Stop** on the pod. The volume is kept and costs a small storage fee.

Later, **Start** the pod. The installed packages lived on the container disk and are gone, so first redo Part 3: `cd /workspace/darija-tutor && git pull`, then the `export`, `pip install` and `pip uninstall` lines. The model cache on the volume is kept. Then run the same command again. The run resumes from the last checkpoint and loses at most 500 steps. Caveat: a stopped pod is tied to its machine, so if that GPU is busy when you come back, you may have to wait.

## Part 6. Run 3 (Atlas-Chat method, chosen 2026-10-28)

Run 3 is no longer TULU + AtlasChat: it is E4B on our own `atlas-tutor` mix (70K rows of native Darija from Atlas-Chat's training data, built by `train/atlas_mix.py`, see PLAN.md step 10) with Atlas-Chat's LoRA settings: r256 / alpha 128, lr 5e-5, one pass. Effective batch 16 as in run 2; a checkpoint every 250 steps (~4,000 rows), ~17 in all.

**Upload the data** (it is git-ignored): in Jupyter's file browser go to `/workspace/darija-tutor/data/train_mix/` (create the folders), and upload `atlas-tutor.jsonl` (28 MB) and `atlas-tutor-val.jsonl` from the laptop's `data	rain_mix\`. Check the hash matches `atlas-tutor.json` on the laptop:

```bash
sha256sum /workspace/darija-tutor/data/train_mix/atlas-tutor*.jsonl
```

**Check first (~10 min):** r256 is 8x the LoRA of run 2 (~560M trainable parameters, ~9 GB more for gradients and optimizer), but the rows are short. Try the fastest setting first:

```bash
python train/train_lora.py --model google/gemma-4-E4B-it --data-file data/train_mix/atlas-tutor.jsonl   --r 256 --alpha 128 --lr 5e-5 --epochs 1 --eval-n 500   --batch 16 --accum 1 --no-checkpointing --max-steps 30 --save-steps 15 --out /workspace/runs/check3
```

Same checks as Part 4 (one `<bos>`, loss on the reply, no `nan`, checkpoints), plus `trainable params: ~560M`. If it runs out of memory, use `--batch 8 --accum 2`. Send me the `done:` line and the peak memory.

**The run:**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train/train_lora.py --model google/gemma-4-E4B-it   --data-file data/train_mix/atlas-tutor.jsonl --r 256 --alpha 128 --lr 5e-5 --epochs 1 --eval-n 500   --batch 16 --accum 1 --no-checkpointing --save-steps 250 --keep 0 --out /workspace/runs/e4b-atlas   > /workspace/runs/e4b-atlas.log 2>&1 &
```

(With `--batch 8 --accum 2` if the check needed it: same effective batch, so the same training.) Download as in Part 7 with `e4b-atlas` in place of `e4b-tulu`. Each r256 checkpoint adapter is ~2.2 GB (run 2's r32 adapters were 279 MB), so all 17 would be ~38 GB: download only `adapter`, `checkpoint-250` ... `checkpoint-2000` and every 1000 after (~11, ~24 GB, one archive per few checkpoints), unless the eval loss says otherwise. Check the volume has room (100 GB: model cache ~16 GB + checkpoints with optimizer state ~6-7 GB each = too much at `--keep 0` for 17): use a **200 GB** volume for this run.

## Part 7. Download the results

Pack the adapters only. The optimizer files (`optimizer.pt`, ~2/3 of each checkpoint) are only needed to resume training.

```bash
cd /workspace/runs
tar czf e4b-tulu.tgz --exclude=optimizer.pt --exclude=scheduler.pt --exclude='rng_state*' \
  e4b-tulu/adapter e4b-tulu/checkpoint-* e4b-tulu/summary.json e4b-tulu.log
ls -lh e4b-tulu.tgz
```

1. In Jupyter's file browser (left panel), open `runs`, right-click `e4b-tulu.tgz`, and click **Download**.
2. On the laptop, unpack it into `data\train_runs\` (git-ignored). PowerShell: `tar -xzf e4b-tulu.tgz -C data\train_runs`

Before you terminate, check that the download unpacks and that `data\train_runs\e4b-tulu\adapter\adapter_model.safetensors` exists.

## Part 8. Terminate

Once **all** downloads are verified, click **Terminate** on the pod (not just Stop). Terminating deletes the volume and ends all billing.

## Part 9. Back on the laptop (with Claude)

Each checkpoint folder is a complete adapter, so it can be exported and scored like the final one:

```powershell
.venv-train\python.exe train\export_gguf.py --base google/gemma-4-E4B-it --adapter data\train_runs\e4b-tulu\checkpoint-500 --name e4b-tulu-500
```

Then score each export with DarijaMMLU and HellaSwag (full sets), plus the fluency check. Compare against `gemma-4-e4b-ours` (the released model through the same pipeline).
