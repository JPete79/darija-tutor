"""Experiment log: one row per benchmark run in eval/experiments.csv.

Every eval script calls `record()` when a run ends (finished, stopped with
Ctrl+C, or crashed), so the log never depends on remembering to write it.
The CSV is committed to git; the per-question files it points to stay in the
git-ignored data/eval_runs/. Open the CSV in Excel, or:

  .venv\\Scripts\\python eval\\registry.py           # print the log
  .venv\\Scripts\\python eval\\registry.py --backfill # add runs found on disk but missing from the log

The `note` column is for the hypothesis or reason behind a run (`--note` on
the eval scripts); edit it by hand afterwards if needed. Rows are never
rewritten by code once written.
"""
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "eval" / "experiments.csv"
RUNS_DIR = ROOT / "data" / "eval_runs"
OLLAMA_URL = "http://127.0.0.1:11434"

FIELDS = [
    "run_id", "started", "status", "benchmark", "model", "model_digest", "quant",
    "scoring", "template", "shots", "n", "seed",
    "acc", "ci_low", "ci_high", "published", "chance", "acc_strict", "no_answer",
    "acc_arabic_mmlu", "acc_mmlu", "seconds", "code_commit", "results_file", "note",
]

# Published zero-shot scores (model cards / papers) to compare against.
PUBLISHED = {
    ("darijammlu", "atlas-chat-2b"): 44.97,
    ("darijammlu", "atlas-chat-9b"): 58.23,
    ("darijahellaswag", "atlas-chat-2b"): 35.08,
    ("darijahellaswag", "atlas-chat-9b"): 43.65,
}


def model_info(model: str) -> tuple[str, str]:
    """Ollama digest (identifies the exact weights) and quantization level."""
    try:
        for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()["models"]:
            if m["name"] in (model, f"{model}:latest"):
                return m["digest"][:12], m["details"].get("quantization_level", "")
    except httpx.HTTPError:
        pass
    return "", ""


def code_commit() -> str:
    """Git commit of the eval code, marked -dirty if there were uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD")
    return sha + ("-dirty" if git("status", "--porcelain", "eval") else "")


def read() -> list[dict]:
    if not LOG.exists():
        return []
    with open(LOG, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append(row: dict) -> None:
    new = not LOG.exists()
    # utf-8-sig so Excel opens it with the right encoding
    with open(LOG, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def pct(x) -> str:
    return "" if x is None or x == "" else f"{100 * float(x):.2f}"


def record(summary: dict, status: str, results_file: Path, digest: str = "", quant: str = "") -> None:
    """Append one run. `summary` is the dict an eval script writes next to its results."""
    if not digest:
        digest, quant = model_info(summary["model"])
    src = summary.get("by_source", {})
    ci = summary.get("ci95") or ["", ""]
    append({
        **{k: summary.get(k, "") for k in ("run_id", "started", "benchmark", "model", "scoring",
                                           "template", "shots", "n", "seed", "no_answer",
                                           "seconds", "note")},
        "status": status,
        "model_digest": digest,
        "quant": quant,
        "acc": pct(summary.get("acc")),
        "ci_low": pct(ci[0]),
        "ci_high": pct(ci[1]),
        "published": PUBLISHED.get((summary.get("benchmark"), summary["model"]), ""),
        "chance": pct(summary.get("chance")),
        "acc_strict": pct(summary.get("acc_strict")),
        "acc_arabic_mmlu": pct(src.get("arabic_mmlu", {}).get("acc")),
        "acc_mmlu": pct(src.get("mmlu", {}).get("acc")),
        "code_commit": summary.get("code_commit", ""),
        "results_file": str(results_file.relative_to(ROOT)).replace("\\", "/"),
    })


def backfill() -> None:
    """Log result files on disk that predate the registry (or were never logged)."""
    known = {r["results_file"] for r in read()}
    for jsonl in sorted(RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        rel = str(jsonl.relative_to(ROOT)).replace("\\", "/")
        if rel in known:
            continue
        summary_file = jsonl.with_suffix(".json")
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            status = "complete" if lines else "complete (per-question file lost)"
        else:
            # Name format: <model>_<scoring>_<template>_<K>shot_n<N>_s<seed>
            model, scoring, template, shots, n, seed = jsonl.stem.rsplit("_", 5)
            rows = [json.loads(l) for l in lines]
            summary = {"model": model, "scoring": scoring, "template": template,
                       "shots": shots.removesuffix("shot"), "n": len(rows), "seed": seed.lstrip("s"),
                       "acc": sum(r["correct"] for r in rows) / len(rows) if rows else None}
            status = f"incomplete ({len(rows)}/{n.lstrip('n')})"
        started = datetime.fromtimestamp(jsonl.stat().st_mtime).isoformat(timespec="minutes")
        summary = {"benchmark": "darijammlu", "run_id": f"backfill-{jsonl.stem}",
                   "started": started + " (file time)"} | summary
        record(summary, status, jsonl)
        print("logged", rel, status)


def show() -> None:
    cols = ["started", "status", "model", "scoring", "template", "shots", "n", "acc", "ci_low", "ci_high", "published", "note"]
    rows = read()
    widths = {c: max([len(c)] + [len(r.get(c, "")[:40]) for r in rows]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(r.get(c, "")[:40].ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--backfill" in sys.argv:
        backfill()
    show()
