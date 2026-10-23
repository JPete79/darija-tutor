"""DarijaMMLU on a local Ollama model (PLAN.md Phase 2b, step A0a).

The prompt copies Atlas-Chat's lm-evaluation-harness task
(MBZUAI-Paris/lm-evaluation-harness-Atlas-Chat, tasks/darijammlu), so
`--scoring logprob --template raw --shots 0` is the setting comparable to
published numbers (up to quantization and the subset drawn here).

Knobs, one experiment each:
  --scoring logprob   read the next-token probabilities of " A", " B", ... and
                      pick the highest (what the harness does)
  --scoring generate  let the model write a short answer and parse the letter
  --template raw      plain text prompt, no chat turns (what the harness does)
  --template chat     prompt wrapped in the model's chat template
  --shots K           K solved examples from the subject's dev split first (0-3)
  --n / --seed        size of the random subset; same n+seed = same questions,
                      and a smaller n is a prefix of a larger one

Example:
  .venv\\Scripts\\python eval\\darija_mmlu.py --model atlas-chat-2b --n 500
Writes data/eval_runs/<timestamp>_<settings>.jsonl (one row per question) +
.json (summary), and appends a row to eval/experiments.csv (also when stopped
with Ctrl+C or crashed). Add --note "why" to record the purpose of the run.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

import registry

OLLAMA_URL = "http://127.0.0.1:11434"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval_runs"

# Same text as the harness utils.py: "This is a multiple-choice question
# (with its answer) about {subject}" ... "The answer:"
PROMPT = "هادا سؤال متعدد الخيارات (مع الجواب ديالو) على {}\n\n{}\n{}\nالجواب:"
LETTERS = "ABCDE"


def load(split: str) -> dict[str, list[dict]]:
    """All rows of one split, grouped by subject, each with a stable id."""
    path = snapshot_download("MBZUAI-Paris/DarijaMMLU", repo_type="dataset",
                             allow_patterns="*/*.parquet")
    by_subject = {}
    for f in sorted(glob.glob(os.path.join(path, "*", f"{split}-*.parquet"))):
        subject = os.path.basename(os.path.dirname(f))
        rows = pq.read_table(f).to_pylist()
        for i, r in enumerate(rows):
            r["id"] = f"{subject}/{i}"
            r["config"] = subject
        by_subject[subject] = rows
    return by_subject


def doc_to_text(doc: dict) -> str:
    question = doc["question"] if not doc["context"] else f"{doc['context']}\n\n{doc['question']}"
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(doc["choices"]))
    return PROMPT.format(doc["subject_darija"], question, options)


def build_prompt(doc: dict, shots: list[dict]) -> str:
    # Harness format: each example is text + " " + letter, examples joined by a blank line.
    parts = [f"{doc_to_text(s)} {LETTERS[s['answer']]}" for s in shots]
    return "\n\n".join(parts + [doc_to_text(doc)])


def ask(model: str, prompt: str, template: str, scoring: str, num_ctx: int | None) -> dict:
    options = {"temperature": 0, "num_predict": 1 if scoring == "logprob" else 24}
    if num_ctx:
        options["num_ctx"] = num_ctx
    body = {"model": model, "prompt": prompt, "raw": template == "raw",
            "stream": False, "options": options}
    if scoring == "logprob":
        body |= {"logprobs": True, "top_logprobs": 20}
    # retry: a full run is ~22K requests, and one dropped connection (Gemma-4-E4B,
    # 2026-10-19, not reproducible) used to end the whole run
    for attempt in range(3):
        try:
            resp = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=300)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            if attempt == 2:
                raise
            print(f"Ollama request failed ({type(e).__name__}: {e}); retrying", flush=True)
            time.sleep(5)


def letter_logprobs(result: dict, k: int, strict: bool = False) -> dict[str, float]:
    """Best log-probability per valid letter among the top-20 first tokens.
    A letter not in the top 20 is simply absent (treated as very unlikely).
    strict: only the exact token " A" counts, as in the harness, which scores
    the continuation " A" (space + letter) and nothing else."""
    found = {}
    for t in (result.get("logprobs") or [{}])[0].get("top_logprobs", []):
        if strict:
            tok = t["token"][1:] if t["token"][:1] == " " else ""
        else:
            tok = t["token"].strip().rstrip(".")
        # a list, not the string: "" in "ABCDE" is True, so a whitespace token
        # stripped to "" would count as a letter (Qwen3.5 often ranks "\n\n" first)
        if tok in list(LETTERS[:k]) and t["logprob"] > found.get(tok, -math.inf):
            found[tok] = t["logprob"]
    return found


def parse_generated(text: str, choices: list[str]) -> tuple[str | None, str]:
    """Letter from a free-text answer: a standalone A-E first, else the text of
    exactly one option appearing in the answer."""
    valid = LETTERS[:len(choices)]
    m = re.search(rf"(?<![A-Za-z])([{valid}])(?![A-Za-z])", text)
    if m:
        return m.group(1), "letter"
    hits = [LETTERS[i] for i, c in enumerate(choices) if c.strip() and c.strip() in text]
    if len(hits) == 1:
        return hits[0], "option_text"
    return None, "unparsed"


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy measured on n questions."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(rows: list[dict], meta: dict) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    lo, hi = wilson(correct, n)
    by_source = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        by_source[src] = {"n": len(sub), "acc": round(sum(r["correct"] for r in sub) / len(sub), 4)}
    extra = {}
    if "pred_strict" in rows[0]:
        extra["acc_strict"] = round(sum(r["pred_strict"] == r["gold"] for r in rows) / n, 4)
        extra["top_tokens"] = dict(Counter(r["top_token"] for r in rows).most_common(8))
    return meta | {
        "n": n,
        "acc": round(correct / n, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        **extra,
        "chance": round(sum(1 / r["n_choices"] for r in rows) / n, 4),
        "no_answer": sum(r["pred"] is None for r in rows),
        "by_source": by_source,
        "pred_letters": dict(sorted(Counter(r["pred"] or "-" for r in rows).items())),
        "gold_letters": dict(sorted(Counter(r["gold"] for r in rows).items())),
        "seconds": round(sum(r["seconds"] for r in rows), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="atlas-chat-2b", help="Ollama model name")
    ap.add_argument("--n", type=int, default=500, help="questions to sample (0 = all 21,792)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scoring", choices=["logprob", "generate"], default="logprob")
    ap.add_argument("--template", choices=["raw", "chat"], default="raw")
    ap.add_argument("--shots", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--num-ctx", type=int, help="override the Modelfile context size (few-shot prompts get long)")
    ap.add_argument("--note", default="", help="why this run: the question or hypothesis it tests (goes in the log)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    test = [r for rows in load("test").values() for r in rows]
    dev = load("dev")
    random.Random(args.seed).shuffle(test)
    sample = test[:args.n] if args.n else test

    # Timestamped so a rerun of the same settings never overwrites an earlier run.
    started = datetime.now()
    name = (f"{started:%Y%m%d-%H%M%S}_{args.model.replace(':', '-')}_{args.scoring}_{args.template}"
            f"_{args.shots}shot_n{len(sample)}_s{args.seed}")
    meta = {"run_id": name, "started": started.isoformat(timespec="seconds"), "benchmark": "darijammlu",
            "code_commit": registry.code_commit()} | vars(args)
    digest, quant = registry.model_info(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUT_DIR / f"{name}.jsonl"
    rows = []
    status = "crashed"
    try:
        with open(results_file, "w", encoding="utf-8") as out:
            run_questions(args, sample, dev, rows, out)
        status = "complete"
    except KeyboardInterrupt:
        status = "stopped"
    finally:
        if rows:
            summary = summarize(rows, meta)
            if status != "complete":
                summary["acc_partial_of"] = len(sample)
            (OUT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            label = status if status == "complete" else f"{status} ({len(rows)}/{len(sample)})"
            registry.record(summary, label, results_file, digest, quant)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"logged to {registry.LOG.relative_to(ROOT)} as {label}")


def run_questions(args, sample: list[dict], dev: dict, rows: list[dict], out) -> None:
    for i, doc in enumerate(sample, 1):
        prompt = build_prompt(doc, dev[doc["config"]][:args.shots])
        t0 = time.perf_counter()
        result = ask(args.model, prompt, args.template, args.scoring, args.num_ctx)
        seconds = time.perf_counter() - t0
        k = len(doc["choices"])
        row = {"id": doc["id"], "source": doc["source"], "n_choices": k,
               "gold": LETTERS[doc["answer"]], "response": result.get("response", ""),
               "prompt_tokens": result.get("prompt_eval_count"), "seconds": round(seconds, 3)}
        if args.scoring == "logprob":
            lp = letter_logprobs(result, k)
            row["letter_logprobs"] = {l: round(v, 3) for l, v in lp.items()}
            row["pred"] = max(lp, key=lp.get) if lp else None
            strict = letter_logprobs(result, k, strict=True)
            row["pred_strict"] = max(strict, key=strict.get) if strict else None
            row["top_token"] = (result.get("logprobs") or [{}])[0].get("token")
        else:
            row["pred"], row["parse"] = parse_generated(row["response"], doc["choices"])
        row["correct"] = row["pred"] == row["gold"]
        rows.append(row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        if i % 50 == 0 or i == len(sample):
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"{i}/{len(sample)}  acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
