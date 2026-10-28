"""Choose a subset of Atlas-Chat's training data (MBZUAI-Paris/Darija-SFT-Mixture, 458K rows,
ODC-By overall; DODa, GPT4-Alpaca and LIMA parts are non-commercial) for a small run in the
Atlas-Chat style (PLAN.md Track A, step 10).

  stats   per subset (and per translation direction): rows, turns, answer tokens with the
          training model's tokenizer (loss is on answers only, so answer tokens, not rows,
          measure how much a subset teaches), prompt tokens, rows with chat-template tags inside
          the text, and contamination: rows sharing a sentence with a test set (DarijaBench test
          splits, our fluency prompts, the ear-test lines). Writes data/train_mix/stats.md
          (with 2 example rows per subset) and stats.json.
  build   writes one mix (MIXES below) to data/train_mix/<name>.jsonl plus <name>-val.jsonl
          (held-out rows for eval loss, never trained on) and <name>.json (manifest: counts per
          subset, rows dropped as contaminated, file hashes). train_lora.py --data-file reads it.

  .venv-train\\python.exe train\\atlas_mix.py stats
  .venv-train\\python.exe train\\atlas_mix.py build --mix atlas-tutor
"""
import argparse
import glob
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "train_mix"
MIXTURE = "MBZUAI-Paris/Darija-SFT-Mixture"
TAGS = re.compile(r"<\|?(assistant|user|system)\|?>|<start_of_turn>|<end_of_turn>", re.I)
# translation subsets are split by direction (e.g. doda en_dr); dr = Darija in Arabic script,
# ar = Darija in Latin letters (Arabizi), msa = Modern Standard Arabic
TEMPLATE_MIN = 20  # a line in this many mixture rows counts as template text
BY_DIRECTION = ("doda", "flores+", "madar", "nllb-seed")

# Chosen by the user 2026-10-28 (PLAN.md step 10): native Darija in, MSA out; few-shot rows (literal
# <|assistant|> tags), sentiment/QA labels, 9esa stories, long translated chat and hard_coded
# (says it was made by MBZUAI) left out. Groups are subset names or "subset direction"; "chat" is
# a random sample from a pool of short-answer rows.
TUTOR_DIRECTIONS = ("en_dr", "dr_en", "ar_dr", "dr_ar", "msa_dr")
MIXES = {
    "atlas-tutor": {
        "groups": [f"{s} {d}" for s in ("doda", "doda_multi_turns") for d in TUTOR_DIRECTIONS]
                  + ["madar msa_dr", "madar_multi_turns msa_dr"]
                  + [f"{s} {d}" for s in ("nllb-seed", "nllb-seed_multi_turns") for d in ("en_dr", "dr_en")]
                  + ["MSMMG", "MSMMG_multi_turns"],
        "chat": {"from": ["gpt4_alpaca", "oasst1"], "n": 5000, "max_answer_tokens": 120},
        "val": 500,
    },
}


def load_mixture() -> list[dict]:
    path = snapshot_download(MIXTURE, repo_type="dataset")
    rows = []
    for f in sorted(glob.glob(f"{path}/data/*.parquet")):
        rows += pq.read_table(f, columns=["dataset", "id", "messages", "direction"]).to_pylist()
    return rows


def group(row: dict) -> str:
    base = row["dataset"]
    return f"{base} {row['direction']}" if row["direction"] and base.split("_")[0] in BY_DIRECTION else base


def norm(text: str) -> str:
    """Compare sentences without diacritics, punctuation, case or extra spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if not unicodedata.combining(c) and c != "ـ")
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def sentences(messages: list[dict]) -> set[str]:
    """Normalized sentences of a conversation: user turns lose their instruction line
    ("ترجم من الفرنساوية للدارجة:\\n<sentence>"); one-word lines are ignored."""
    out = set()
    for m in messages:
        text = m["content"]
        if m["role"] == "user" and "\n" in text:
            text = text.split("\n", 1)[1]
        for line in text.split("\n"):
            n = norm(line)
            if len(n.split()) >= 2:
                out.add(n)
    return out


def test_sentences() -> dict[str, set[str]]:
    sets = {}
    path = snapshot_download("MBZUAI-Paris/DarijaBench", repo_type="dataset", allow_patterns="data/*.parquet")
    for f in sorted(glob.glob(f"{path}/data/*.parquet")):
        name = "DarijaBench " + Path(f).name.split("-")[0]
        sets[name] = set().union(*(sentences(r["messages"]) for r in pq.read_table(f, columns=["messages"]).to_pylist()))
    ours = set()
    for line in (ROOT / "eval" / "fluency_prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            ours |= {n for n in [norm(json.loads(line)["text"])] if len(n.split()) >= 2}
    for m in re.finditer(r"\|\s*\d+\s*\|\s*`(.+?)`", (ROOT / "eval" / "ear_test.md").read_text(encoding="utf-8")):
        ours.add(norm(m[1]))
    sets["our fluency + ear-test lines"] = ours
    return sets


def test_index(rows: list[dict]) -> dict[str, str]:
    """Test sentence -> test set name. Lines repeated across many mixture rows are instruction
    templates ("-سلبي -ايجابي ..."), not sentences: matching on them flagged nearly every
    sentiment row as contaminated."""
    seen = Counter(x for r in rows for x in sentences(r["messages"]))
    by_test = {}
    for name, s in test_sentences().items():
        for sent in s:
            if seen[sent] < TEMPLATE_MIN:
                by_test.setdefault(sent, name)
    return by_test


def stats(args) -> None:
    import transformers as tf
    tok = tf.AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_mixture()
    by_test = test_index(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[group(r)].append(r)
    report = {}
    for g, rs in sorted(groups.items()):
        answers = [m["content"] for r in rs for m in r["messages"] if m["role"] == "assistant"]
        prompts = [m["content"] for r in rs for m in r["messages"] if m["role"] != "assistant"]
        a_len = [len(x) for x in tok(answers, add_special_tokens=False)["input_ids"]]
        p_tok = sum(len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"])
        hits = defaultdict(int)
        for r in rs:
            found = {by_test[s] for s in sentences(r["messages"]) if s in by_test}
            for name in found:
                hits[name] += 1
        report[g] = {"rows": len(rs), "turns": statistics.median(sum(m["role"] == "assistant" for m in r["messages"]) for r in rs),
                     "answer_tokens": sum(a_len), "answer_median": statistics.median(a_len), "prompt_tokens": p_tok,
                     "tag_rows": sum(bool(TAGS.search(m["content"])) for r in rs for m in r["messages"][:1]),
                     "contaminated": dict(hits),
                     "examples": [r["messages"] for r in random.Random(0).sample(rs, min(2, len(rs)))]}
        print(f"{g:32s} {len(rs):7d} rows  {sum(a_len) / 1e6:6.2f}M answer tokens  median {statistics.median(a_len):5.0f}"
              f"  contaminated {sum(hits.values())}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Darija-SFT-Mixture by subset", "",
          f"Tokens: {args.tokenizer} tokenizer, text only (no chat template). Contaminated = rows sharing a "
          f"sentence (2+ words, normalized) with a test set; lines found in {TEMPLATE_MIN}+ mixture rows are "
          "template text and ignored.", "",
          "| Subset | Rows | Turns | Answer tokens | Median answer | Prompt tokens | Template tags | Contaminated |",
          "|---|---|---|---|---|---|---|---|"]
    for g, s in report.items():
        cont = ", ".join(f"{k.replace('DarijaBench ', 'DB ')} {v}" for k, v in s["contaminated"].items()) or "-"
        md.append(f"| {g} | {s['rows']:,} | {s['turns']:g} | {s['answer_tokens'] / 1e6:.2f}M | {s['answer_median']:g} | "
                  f"{s['prompt_tokens'] / 1e6:.2f}M | {s['tag_rows'] or '-'} | {cont} |")
    for g, s in report.items():
        md += ["", f"## {g}"]
        for ex in s["examples"]:
            md.append("")
            for m in ex:
                text = m["content"] if len(m["content"]) < 600 else m["content"][:600] + " [...]"
                md.append(f"> **{m['role']}:** " + text.replace("\n", "  \n> "))
    (OUT / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", OUT / "stats.md")


def build(args) -> None:
    import hashlib
    import transformers as tf
    spec = MIXES[args.mix]
    rows = load_mixture()
    by_test = test_index(rows)
    clean = lambda r: (all(m["content"].strip() for m in r["messages"]) and r["messages"][-1]["role"] == "assistant"
                       and not TAGS.search(" ".join(m["content"] for m in r["messages"])))
    hit = lambda r: any(s in by_test for s in sentences(r["messages"]))
    picked, counts, dropped = [], Counter(), Counter()
    for r in rows:
        g = group(r)
        if g in spec["groups"]:
            if not clean(r) or hit(r):
                dropped[g] += 1
            else:
                picked.append(r)
                counts[g] += 1
    if chat := spec.get("chat"):
        tok = tf.AutoTokenizer.from_pretrained(args.tokenizer)
        pool = [r for r in rows if r["dataset"] in chat["from"] and clean(r) and not hit(r)]
        longest = [max(len(x) for x in tok([m["content"] for m in r["messages"] if m["role"] == "assistant"],
                                           add_special_tokens=False)["input_ids"]) for r in pool]
        pool = [r for r, n in zip(pool, longest) if n <= chat["max_answer_tokens"]]
        sample = random.Random(args.seed).sample(pool, min(chat["n"], len(pool)))
        picked += sample
        counts.update(f"{r['dataset']} (chat, answers <= {chat['max_answer_tokens']} tokens)" for r in sample)
        print(f"chat pool {len(pool)} rows with answers <= {chat['max_answer_tokens']} tokens, sampled {len(sample)}")
    random.Random(args.seed).shuffle(picked)
    val, train = picked[:spec["val"]], picked[spec["val"]:]
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for suffix, part in (("", train), ("-val", val)):
        path = OUT / f"{args.mix}{suffix}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in part:
                f.write(json.dumps({k: r[k] for k in ("dataset", "id", "direction", "messages")}, ensure_ascii=False) + "\n")
        files[path.name] = {"rows": len(part), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"mix": args.mix, "source": MIXTURE, "seed": args.seed, "spec": spec, "files": files,
                "rows_per_group": dict(sorted(counts.items())),
                "dropped_contaminated_or_malformed": dict(sorted(dropped.items()))}
    (OUT / f"{args.mix}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for g, n in sorted(counts.items()):
        print(f"{n:7d}  {g}" + (f"   (dropped {dropped[g]})" if dropped[g] else ""))
    print(f"train {len(train)}, val {len(val)} -> {OUT / args.mix}.jsonl")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    s.set_defaults(func=stats)
    b = sub.add_parser("build")
    b.add_argument("--mix", required=True, choices=MIXES)
    b.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=build)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
