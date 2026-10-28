"""Replay the ear-test conversations (eval/ear_test.md) without typing, for comparing prompt
versions and models on exactly the same learner lines.

Runs the real tutor (same code, prompt, persona and temperature 0.6 as the app; text only) and
writes one transcript per model to data/ear_runs/<timestamp>_<model>.md, with every prompt
version and seed side by side, plus a .json with the raw replies. Each reply gets simple counts
that flag things for the ear to check, not a score:
  sentences  Arabic lines in the reply (v2 asks for one or two)
  questions  Arabic lines ending in a question mark (v2 asks for at most one)
  not sure   replies with the "I'm not sure" remark (wanted only on factual questions)
  brackets   Arabic words in brackets, i.e. a template placeholder like "(hna s3r lqhua dyalk)"
  context    prompt + reply tokens of the reply (num_ctx is 2048 for the 9B)

  .venv\\Scripts\\python eval\\ear_replay.py --model atlas-chat-9b --prompt v1 v2
Stop HADRA first; one model at a time.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import registry

sys.path.insert(0, str(registry.ROOT / "src"))
from darija_tutor import config  # noqa: E402
from darija_tutor.translit import to_arabizi  # noqa: E402
from darija_tutor.tutor import Tutor, pair_lines  # noqa: E402

SCRIPTS = Path(__file__).with_name("ear_test.md")
OUT_DIR = registry.ROOT / "data" / "ear_runs"
HEDGE = re.compile(r"متأكد|not sure", re.I)
# Arabic words in () or [], e.g. "(hna s3r lqhua dyalk)", "[الاسم]"; not gender endings like "داير(ة)"
PLACEHOLDER = re.compile(r"[(\[][^)\]]*[؀-ۿ]{2,}[^)\]]*[)\]]")


def load_scripts() -> dict[str, dict]:
    """Conversations from ear_test.md: '## Conversation N: name', optional 'Topic: `...`', and
    table rows '| n | `learner line` | tests |'."""
    scripts, cur = {}, None
    for line in SCRIPTS.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"## Conversation \d+: (\w+)", line):
            cur = scripts.setdefault(m[1].lower(), {"topic": None, "lines": []})
        elif not line.startswith("## ") and cur is not None:
            if m := re.match(r"Topic: `(.+)`", line):
                cur["topic"] = m[1]
            elif m := re.match(r"\|\s*\d+\s*\|\s*`(.+?)`\s*\|\s*(.*?)\s*\|", line):
                cur["lines"].append({"text": m[1], "tests": m[2]})
        elif line.startswith("## "):
            cur = None
    return scripts


def counts(reply: str) -> dict:
    arabic = [a for a, _ in pair_lines(reply) if a]
    return {"sentences": len(arabic),
            "questions": sum(a.rstrip().endswith(("؟", "?")) for a in arabic),
            "hedge": bool(HEDGE.search(reply)),
            "placeholder": bool(PLACEHOLDER.search("\n".join(arabic)))}


def run(model: str, prompt: str, script: dict, persona: str, learner: str | None, seed: int) -> list[dict]:
    config.LLM_OPTIONS = {**config.LLM_OPTIONS, "seed": seed}
    tutor = Tutor("discussion", model=model, persona=persona, topic=script["topic"],
                  learner_gender=learner, prompt=prompt)
    turns = []
    for line in [None] + script["lines"]:
        reply = tutor.say(line["text"] if line else None)
        turns.append({"learner": line["text"] if line else None, "tests": line["tests"] if line else "opener",
                      "reply": reply, "context": tutor.last_usage.get("prompt", 0) + tutor.last_usage.get("reply", 0),
                      **counts(reply)})
    return turns


def render(reply: str) -> list[str]:
    out = []
    for arabic, english in pair_lines(reply):
        if arabic:
            out += [f"> {arabic}  ", f"> *{to_arabizi(arabic)}*  "]
        if english:
            out.append(f"> EN: {english}  ")
    return out


def summary_row(label: str, turns: list[dict]) -> str:
    sent = [t["sentences"] for t in turns]
    hedges = [str(i) for i, t in enumerate(turns) if t["hedge"]]
    return (f"| {label} | {sum(sent) / len(sent):.1f} (max {max(sent)}) | "
            f"{sum(t['questions'] > 1 for t in turns)} | {', '.join(hedges) or '-'} | "
            f"{sum(t['placeholder'] for t in turns)} | {max(t['context'] for t in turns)} |")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--prompt", nargs="+", default=[config.PROMPT_VERSION], help="prompt versions, e.g. v1 v2")
    ap.add_argument("--scripts", nargs="+", help="conversations to run (default: all in ear_test.md)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="one replay per seed")
    ap.add_argument("--persona", default="noual", choices=list(config.PERSONAS))
    ap.add_argument("--learner", choices=["male", "female"],
                    help="learner gender (default: none, as in the 2026-10-24 ear test)")
    ap.add_argument("--note", default="", help="why this run")
    args = ap.parse_args()

    scripts = load_scripts()
    names = args.scripts or list(scripts)
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{started:%Y%m%d-%H%M%S}_{args.model}"
    runs = []
    t0 = time.perf_counter()
    for name in names:
        for prompt in args.prompt:
            for seed in args.seeds:
                turns = run(args.model, prompt, scripts[name], args.persona, args.learner, seed)
                runs.append({"script": name, "prompt": prompt, "seed": seed, "turns": turns})
                print(f"{name} {prompt} seed {seed}: {time.perf_counter() - t0:.0f} s", flush=True)

    meta = {"model": args.model, "started": started.isoformat(timespec="seconds"), "persona": args.persona,
            "learner": args.learner, "options": config.LLM_OPTIONS, "code_commit": registry.code_commit(),
            "note": args.note}
    out.with_suffix(".json").write_text(json.dumps({**meta, "runs": runs}, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    md = [f"# Ear replay: {args.model}", "",
          f"{meta['started']}, persona {args.persona}, learner {args.learner or 'not given'}, "
          f"temperature {config.LLM_OPTIONS.get('temperature')}, commit {meta['code_commit']}. {args.note}".strip(), "",
          "Counts per reply (opener = reply 0): average sentences, replies with more than one question, "
          "replies with the \"not sure\" remark, bracket placeholders, most context tokens used.", "",
          "| Run | Sentences | >1 question | \"Not sure\" in reply | Placeholders | Context |",
          "|---|---|---|---|---|---|"]
    md += [summary_row(f"{r['script']} {r['prompt']} s{r['seed']}", r["turns"]) for r in runs]
    for r in runs:
        md += ["", f"## {r['script']}, prompt {r['prompt']}, seed {r['seed']}"]
        for i, t in enumerate(r["turns"]):
            flags = [f for f, on in (("not sure", t["hedge"]), ("placeholder", t["placeholder"]),
                                     (f"{t['questions']} questions", t["questions"] > 1)) if on]
            md += ["", f"**{i}. {t['learner'] or '(opener)'}** — {t['tests']}"
                   + (f" — *{', '.join(flags)}*" if flags else ""), ""]
            md += render(t["reply"])
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[6:6 + 2 + len(runs)]))
    print("transcript:", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
