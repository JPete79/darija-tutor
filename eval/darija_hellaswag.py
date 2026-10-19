"""DarijaHellaSwag on a local GGUF model via llama.cpp (PLAN.md Phase 2b, track A).

Each question is a context ("activity: situation") plus 4 possible endings;
the model "answers" by finding one ending more likely than the others. Unlike
DarijaMMLU this needs the probability of every token of a given ending, which
Ollama cannot report, so this uses llama.cpp's own HellaSwag scorer
(`llama-perplexity --hellaswag`) on the same GGUF file Ollama uses.

Matches Atlas-Chat's harness task: validation split (10,042 questions), query
= activity_label + ": " + ctx, each ending appended after a space, zero-shot.
One difference: llama.cpp averages the log-probability per token of the
ending; the harness's published acc_norm divides by the ending's length in
bytes. Both remove the bias toward short endings, but can differ slightly.

llama-perplexity shuffles questions internally (fixed seed), so the result is
the overall accuracy and its 95% interval, without per-question answers.

  .venv\\Scripts\\python eval\\darija_hellaswag.py --model atlas-chat-9b --note "why"
  (--n 1000 for a quick check on 1000 random questions)
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

import registry

ROOT = registry.ROOT
OUT_DIR = registry.RUNS_DIR
LLAMA_PERPLEXITY = ROOT / "models" / "llama.cpp" / "llama-perplexity.exe"
FULL = 10042


def write_task_file() -> Path:
    """llama.cpp's HellaSwag format: 6 lines per question (query, gold index, 4 endings)."""
    path = snapshot_download("MBZUAI-Paris/DarijaHellaSwag", repo_type="dataset",
                             allow_patterns="data/validation-*.parquet")
    rows = pq.read_table(os.path.join(path, "data", "validation-00000-of-00001.parquet")).to_pylist()
    lines = []
    for r in rows:
        lines += [f"{r['activity_label']}: {r['ctx']}", str(int(r["label"])), *r["endings"]]
    out = OUT_DIR / "darijahellaswag_validation.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def gguf_path(model: str) -> Path:
    """The weights file behind an Ollama model (the FROM line of its Modelfile)."""
    text = subprocess.run(["ollama", "show", "--modelfile", model], capture_output=True,
                          text=True, encoding="utf-8").stdout
    return Path(re.search(r"^FROM (.+)$", text, re.M).group(1).strip())


def unload_ollama() -> None:
    """Free the GPU: 8 GB holds one 9B model, so Ollama must not keep one loaded."""
    for m in httpx.get(f"{registry.OLLAMA_URL}/api/ps", timeout=10).json().get("models", []):
        httpx.post(f"{registry.OLLAMA_URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="atlas-chat-2b", help="Ollama model name (its GGUF file is used)")
    ap.add_argument("--n", type=int, default=0, help=f"questions (0 = all {FULL:,}); fewer = random subset")
    ap.add_argument("--ctx", type=int, default=1024,
                    help="llama.cpp context size; 2048 made the 9B spill into system RAM on 8 GB (5x slower)")
    ap.add_argument("--note", default="", help="why this run: the question or hypothesis it tests")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    n = args.n or FULL
    started = datetime.now()
    name = f"{started:%Y%m%d-%H%M%S}_{args.model.replace(':', '-')}_hellaswag_n{n}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / f"{name}.log"
    tasks = write_task_file()
    unload_ollama()
    digest, quant = registry.model_info(args.model)

    cmd = [str(LLAMA_PERPLEXITY), "-m", str(gguf_path(args.model)), "-f", str(tasks),
           "--hellaswag", "--hellaswag-tasks", str(n), "-ngl", "99", "-c", str(args.ctx)]
    t0 = time.perf_counter()
    last = None
    status = "crashed"
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                log.write(line)
                m = re.match(r"^(\d+)\t([\d.]+)%\t\[([\d.]+)%, ([\d.]+)%\]", line)
                if m:
                    last = [int(m.group(1))] + [float(x) / 100 for x in m.group(2, 3, 4)]
                    if last[0] % 500 == 0:
                        print(f"{last[0]}/{n}  acc_norm {last[1]:.3f}", flush=True)
            proc.wait()
        status = "complete" if proc.returncode == 0 and last and last[0] == n else f"failed (exit {proc.returncode})"
    except KeyboardInterrupt:
        proc.kill()
        status = "stopped"
    finally:
        if last is None:
            print(f"no results; see {log_file}")
            return
        done, acc, lo, hi = last
        summary = {"run_id": name, "started": started.isoformat(timespec="seconds"),
                   "benchmark": "darijahellaswag", "model": args.model,
                   "scoring": "acc_norm_per_token", "template": "raw", "shots": 0, "n": done, "seed": 1,
                   "acc": acc, "ci95": [lo, hi], "chance": 0.25, "no_answer": 0,
                   "seconds": round(time.perf_counter() - t0, 1), "note": args.note,
                   "code_commit": registry.code_commit()}
        label = status if status == "complete" else f"{status} ({done}/{n})"
        registry.record(summary, label, log_file, digest, quant)
        print(f"{args.model}: acc_norm {acc:.4f} (95% CI {lo:.4f}-{hi:.4f}) on {done} questions; "
              f"logged as {label}")


if __name__ == "__main__":
    main()
