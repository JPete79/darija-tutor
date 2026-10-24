"""Fluency check: does the model *write* Moroccan Darija, or drift into MSA / English?
(PLAN.md Phase 2b, Track A, step 3.) DarijaMMLU and HellaSwag only read the model's
probabilities; this one makes it generate.

Three parts, one model per run, greedy decoding (temperature 0) so a rerun gives the same text:
  tutor      the 30 prompts of eval/fluency_prompts.jsonl (v2: 10 English; 10 sentences from the
             Peace Corps Moroccan Arabic textbook, each in Arabic script and in Arabizi, with page
             and original transcription in `pc`), each as the first turn after the tutor's real
             system prompt (Noual, discussion mode)
  bare       the 20 Darija/Arabizi prompts with no system prompt: what the model does on its own
  translate  English -> Darija, with DarijaBench's own Darija instruction: 200 everyday
             sentences (DODa) + 200 formal ones (FLORES+), scored with chrF against the
             reference (character n-grams, forgiving of Darija's loose spelling)

Every Arabic-script sentence the model writes is classified by eval/dialect_scores.py
(run in .venv-train). Main numbers per part:
  moroccan  share of sentences the router labels Moroccan (logged as `acc`)
  msa       share labelled MSA;  aldi = mean dialectness 0 (MSA) .. 1 (dialect)
  latin     share of replies written mostly in Latin letters (the tutor must use Arabic script)
95% intervals come from resampling prompts/sentences (bootstrap): with 30 prompts they are wide,
so small differences mean nothing. Caveat: Atlas-Chat trained on DarijaBench's training splits;
these are the test splits, but the style is familiar to it.

  .venv\\Scripts\\python eval\\fluency.py --model gemma-4-e4b-ours --note "baseline before training"
Writes data/eval_runs/<timestamp>_<model>_fluency.jsonl (one row per prompt) + .json (summary)
+ .md (all replies, for reading by ear), and logs one row per part in eval/experiments.csv
(benchmarks fluency_tutor, fluency_bare, translate_doda, translate_flores).
"""
import argparse
import glob
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import sacrebleu
from huggingface_hub import snapshot_download

import registry

ROOT = registry.ROOT
OUT_DIR = registry.RUNS_DIR
PROMPTS = Path(__file__).with_name("fluency_prompts.jsonl")
TRAIN_PY = ROOT / ".venv-train" / "python.exe"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 2048}
ARABIC = re.compile(r"[؀-ۿ]")
# "Write only the translation, without explanation": chat models otherwise list several options
# with explanations in MSA, which says more about their chattiness than their Darija
ONLY_TRANSLATION = "\n\nكتب غير الترجمة، بلا شرح."


def tutor_system() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from darija_tutor.tutor import Tutor
    return Tutor("discussion", persona="noual").history[0]["content"]


def translation_items(n: int, seed: int) -> list[dict]:
    path = snapshot_download("MBZUAI-Paris/DarijaBench", repo_type="dataset", allow_patterns="data/*.parquet")
    items = []
    for name in ("doda", "flores_plus"):
        rows = [r for r in pq.read_table(glob.glob(f"{path}/data/{name}-*.parquet")[0]).to_pylist()
                if r["direction"] == "en_dr"]
        random.Random(seed).shuffle(rows)
        items += [{"part": f"translate_{name.split('_')[0]}", "id": r["id"],
                   "prompt": r["messages"][0]["content"] + ONLY_TRANSLATION, "ref": r["messages"][1]["content"]}
                  for r in rows[:n]]
    return items


def first_translation(reply: str) -> str:
    """First Arabic-script line without markdown (skips a preamble like "Here is the translation:")."""
    line = next((l for l in reply.splitlines() if ARABIC.search(l)), "")
    return re.sub(r"\([^)]*[A-Za-z][^)]*\)", "", re.sub(r"[*>\"#]|^\s*[-•]\s*", "", line)).strip()


class Degenerate(Exception):
    """Ollama aborted the reply because the model kept repeating the same token."""


def chat(model: str, messages: list[dict], max_tokens: int) -> str:
    # think=False: Gemma 4 otherwise spends the whole token budget on hidden reasoning and
    # returns an empty reply (models without thinking ignore the flag)
    r = httpx.post(f"{registry.OLLAMA_URL}/api/chat", timeout=600, json={
        "model": model, "messages": messages, "stream": False, "think": False,
        "options": OPTIONS | {"num_predict": max_tokens}})
    if r.status_code == 500 and "repeat limit" in r.text:
        raise Degenerate(r.json()["error"])
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def bootstrap(values: list, stat, n: int = 1000) -> list[float]:
    rng = random.Random(0)
    draws = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n))
    return [draws[int(0.025 * n)], draws[int(0.975 * n)]]


def label_share(rows: list[dict], label: str) -> float | None:
    labels = [l for r in rows for l in r["reply_dialect"]["labels"]]
    return sum(l == label for l in labels) / len(labels) if labels else None


def summarize(rows: list[dict]) -> dict:
    d = [r["reply_dialect"] for r in rows]
    aldi = [x["aldi"] for x in d if x["aldi"] is not None]
    s = {"n": len(rows), "segments": sum(x["segments"] for x in d),
         "acc": label_share(rows, "ma"), "msa": label_share(rows, "ar"),
         "aldi": sum(aldi) / len(aldi) if aldi else None,
         "latin": sum(x["latin_share"] > 0.5 for x in d) / len(d),
         "loops": sum("error" in r for r in rows)}
    if s["acc"] is not None:
        s["ci95"] = bootstrap(rows, lambda rs: label_share(rs, "ma") or 0.0)
    if "ref" in rows[0]:
        hyps, refs = [r["reply"] for r in rows], [r["ref"] for r in rows]
        s["chrf"] = sacrebleu.corpus_chrf(hyps, [refs]).score
        pairs = list(zip(hyps, refs))
        s["chrf_ci95"] = bootstrap(pairs, lambda ps: sacrebleu.corpus_chrf([h for h, _ in ps], [[r for _, r in ps]]).score, 300)
    return s


def write_md(path: Path, model: str, rows: list[dict], summary: dict) -> None:
    lines = [f"# Fluency replies: {model}", ""]
    for part, s in summary["parts"].items():
        extra = f", chrF {s['chrf']:.1f}" if "chrf" in s else ""
        lines.append(f"- **{part}**: Moroccan {pct(s['acc'])}, MSA {pct(s['msa'])}, ALDi {s['aldi'] or 0:.2f}, "
                     f"Latin replies {pct(s['latin'])}{extra}")
    for r in rows:
        if r["part"].startswith("translate"):
            continue
        d = r["reply_dialect"]
        lines += ["", f"## {r['part']} {r['id']} ({r['script']}, {r['kind']})", "", f"> {r['prompt']}", "",
                  r["reply"], "", f"*ALDi {d['aldi']}, labels {' '.join(d['labels'])}*"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--n-translate", type=int, default=200, help="sentences per translation set (0 = skip)")
    ap.add_argument("--prompts", default=str(PROMPTS),
                    help="prompt set (v2 = Peace Corps sentences; fluency_prompts_v1.jsonl = the first, self-written set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="why this run: the question or hypothesis it tests")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    started = datetime.now()
    run_id = f"{started:%Y%m%d-%H%M%S}_{args.model}_fluency"
    out = OUT_DIR / f"{run_id}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest, quant = registry.model_info(args.model)
    system = tutor_system()
    prompt_set = "v1" if Path(args.prompts).stem.endswith("_v1") else "v2"
    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = [{"part": "tutor", **p, "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": p["text"]}], "max": 400}
            for p in prompts]
    jobs += [{"part": "bare", **p, "messages": [{"role": "user", "content": p["text"]}], "max": 400}
             for p in prompts if p["script"] != "english"]
    if args.n_translate:
        jobs += [{**t, "script": "english", "kind": "translate", "text": t["prompt"],
                  "messages": [{"role": "user", "content": t["prompt"]}], "max": 200}
                 for t in translation_items(args.n_translate, args.seed)]

    t0 = time.perf_counter()
    rows = []
    with open(out, "w", encoding="utf-8") as f:
        for i, j in enumerate(jobs, 1):
            row = {k: j[k] for k in ("part", "id", "script", "kind")} | {"prompt": j["text"]}
            if "ref" in j:
                row["ref"] = j["ref"]
            try:
                row["reply"] = chat(args.model, j["messages"], j["max"])
            except Degenerate as e:  # a looping model is a failed answer, not a crashed run
                row["reply"], row["error"] = "", str(e)
            if "ref" in j:  # score only the translation itself; keep the whole answer
                row["raw"], row["reply"] = row["reply"], first_translation(row["reply"])
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} generated, {time.perf_counter() - t0:.0f} s", flush=True)

    subprocess.run([str(TRAIN_PY), str(Path(__file__).with_name("dialect_scores.py")), str(out),
                    "--field", "reply"], check=True)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    secs = round(time.perf_counter() - t0)
    parts = {}
    for part in dict.fromkeys(r["part"] for r in rows):
        parts[part] = summarize([r for r in rows if r["part"] == part])
    summary = {"run_id": run_id, "started": started.isoformat(timespec="seconds"), "model": args.model,
               "seed": args.seed, "seconds": secs, "prompts": prompt_set, "note": args.note,
               "code_commit": registry.code_commit(),
               "parts": parts}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(out.with_suffix(".md"), args.model, rows, summary)

    for part, s in parts.items():
        chrf = f"chrF {s['chrf']:.1f}; " if "chrf" in s else ""
        registry.record({"run_id": f"{run_id}_{part}", "started": summary["started"], "benchmark": f"fluency_{part}"
                         if not part.startswith("translate") else part, "model": args.model,
                         "scoring": "moroccan_share", "template": "tutor_system" if part == "tutor" else "chat",
                         "shots": 0, "n": s["n"], "seed": args.seed, "acc": s["acc"], "ci95": s.get("ci95"),
                         "seconds": secs, "code_commit": summary["code_commit"],
                         "note": f"prompts {prompt_set}; {chrf}MSA {pct(s['msa'])}, ALDi {s['aldi'] or 0:.2f}, Latin {pct(s['latin'])}, loops {s['loops']}. {args.note}".strip()},
                        "complete", out, digest, quant)
        print(f"{part:17s} Moroccan {pct(s['acc'])} {[round(100 * x) for x in s.get('ci95', [])]}  MSA {pct(s['msa'])}  "
              f"ALDi {s['aldi'] or 0:.2f}  Latin {pct(s['latin'])}  {chrf}loops {s['loops']} ({s['segments']} sentences)")
    print("replies to read:", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
