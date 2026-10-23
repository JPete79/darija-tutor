"""Cross-check llama.cpp's DarijaHellaSwag scorer, question by question (PLAN.md Phase 2b, A2c).

Why: llama-perplexity evaluates a question's shared context once and branches
into the 4 endings (one token belonging to several sequences). Routine for
Transformers, but Qwen3.5 is a hybrid with recurrent Gated DeltaNet layers,
where this could be silently wrong. So score the same questions twice and
compare per question:

  llamacpp  the Q4 GGUF in llama-perplexity, ONE question per call (it shuffles
            multi-question files, so this is the only way to get per-question
            answers); slow-ish (one model load per question), fine for small models
  hf        the original bf16 weights in transformers, every ending scored as its
            own sequence (no branching); needs torch + transformers, which only
            .venv-hadra has, so run that engine with .venv-hadra\\python.exe

Quantization (Q4 vs bf16) alone changes some answers, so a plain Transformer
(Gemma) run through both engines is the control: if Qwen's agreement is about
as high as Gemma's, the branching works on the hybrid.

Both engines use llama.cpp's scoring: tokenize context + " " + ending, find the
prefix shared by all 4, average the log-probability of the ending tokens after
it (acc_norm per token), pick the highest.

  .venv\\Scripts\\python eval\\hellaswag_crosscheck.py llamacpp --model qwen3.5-0.8b
  .venv-hadra\\python.exe eval\\hellaswag_crosscheck.py hf --model qwen3.5-0.8b --hf Qwen/Qwen3.5-0.8B
  .venv\\Scripts\\python eval\\hellaswag_crosscheck.py compare data\\eval_runs\\<a>.jsonl data\\eval_runs\\<b>.jsonl
"""
import argparse
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "eval_runs"
TASKS = RUNS / "darijahellaswag_validation.txt"  # written by darija_hellaswag.py


def load_sample(n: int, seed: int) -> list[dict]:
    lines = TASKS.read_text(encoding="utf-8").split("\n")
    total = len(lines) // 6
    idx = sorted(random.Random(seed).sample(range(total), n))
    return [{"i": i, "context": lines[6 * i], "gold": int(lines[6 * i + 1]),
             "endings": lines[6 * i + 2:6 * i + 6]} for i in idx]


def out_path(args, engine: str) -> Path:
    return RUNS / f"crosscheck_{args.model.replace(':', '-')}_{engine}_n{args.n}_s{args.seed}.jsonl"


def run_llamacpp(args) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import darija_hellaswag as dh
    if not TASKS.exists():
        dh.write_task_file()
    dh.unload_ollama()
    gguf = dh.gguf_path(args.model)
    out = out_path(args, "llamacpp")
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8") as f, tempfile.TemporaryDirectory() as tmp:
        task = Path(tmp) / "one.txt"
        for k, q in enumerate(load_sample(args.n, args.seed), 1):
            task.write_text("\n".join([q["context"], str(q["gold"]), *q["endings"]]) + "\n", encoding="utf-8")
            res = subprocess.run([str(dh.LLAMA_PERPLEXITY), "-m", str(gguf), "-f", str(task), "--hellaswag",
                                  "--hellaswag-tasks", "1", "-ngl", "99", "-c", str(args.ctx)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            m = re.search(r"^1\t([\d.]+)%", res.stdout + res.stderr, re.M)
            if not m:
                sys.exit(f"question {q['i']}: no result from llama-perplexity\n{res.stderr[-2000:]}")
            f.write(json.dumps({"i": q["i"], "correct": float(m.group(1)) > 50}) + "\n")
            if k % 50 == 0:
                print(f"{k}/{args.n}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {out}")


def run_hf(args) -> None:
    import torch
    import transformers as tf
    tok = tf.AutoTokenizer.from_pretrained(args.hf)
    kw = {"dtype": torch.bfloat16, "device_map": "auto",
          "max_memory": {0: args.gpu_mem, "cpu": "24GiB"}}
    try:
        model = tf.AutoModelForCausalLM.from_pretrained(args.hf, **kw)
    except (ValueError, KeyError):  # checkpoints saved as vision-language models
        model = tf.AutoModelForImageTextToText.from_pretrained(args.hf, **kw)
    model.eval()
    dev = next(model.parameters()).device
    out = out_path(args, "hf")
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8") as f:
        for k, q in enumerate(load_sample(args.n, args.seed), 1):
            seqs = [tok(q["context"] + " " + e, add_special_tokens=True)["input_ids"] for e in q["endings"]]
            # llama.cpp adds BOS when the model has one; Gemma 4's HF tokenizer does not,
            # and without it Gemma degenerates (scored 29% vs 42% with BOS)
            if tok.bos_token_id is not None and seqs[0][0] != tok.bos_token_id:
                seqs = [[tok.bos_token_id] + s for s in seqs]
            prefix = 0
            while all(len(s) > prefix + 1 for s in seqs) and len({s[prefix] for s in seqs}) == 1:
                prefix += 1
            scores = []
            for s in seqs:  # one sequence at a time: no shared state between endings
                ids = torch.tensor([s], device=dev)
                with torch.no_grad():
                    logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
                # llama.cpp scores tokens prefix+1 .. end (the first differing token is not
                # scored); score_from0 also includes it, to see which one llama.cpp matches
                lp = [logp[j - 1, s[j]].item() for j in range(prefix, len(s))]
                scores.append((sum(lp[1:]) / max(len(lp) - 1, 1), sum(lp) / len(lp)))
            pred = max(range(4), key=lambda j: scores[j][0])
            pred0 = max(range(4), key=lambda j: scores[j][1])
            f.write(json.dumps({"i": q["i"], "gold": q["gold"], "pred": pred, "correct": pred == q["gold"],
                                "correct_from0": pred0 == q["gold"], "prefix": prefix,
                                "ending_tokens": [len(s) - prefix for s in seqs]}) + "\n")
            if k % 50 == 0:
                print(f"{k}/{args.n}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {out}")


def compare(args) -> None:
    a, b = ({r["i"]: r for r in map(json.loads, open(p, encoding="utf-8"))} for p in (args.a, args.b))
    common = sorted(a.keys() & b.keys())
    # A's verdict vs each of B's (an hf file also has the "first differing token scored" variant)
    for key in [k for k in ("correct", "correct_from0") if k in b[common[0]]]:
        pairs = [(a[i]["correct"], b[i][key]) for i in common]
        n = len(pairs)
        same = sum(x == y for x, y in pairs)
        only_a = sum(x and not y for x, y in pairs)
        only_b = sum(y and not x for x, y in pairs)
        d = only_a + only_b
        p = min(1.0, 2 * sum(math.comb(d, k) for k in range(min(only_a, only_b) + 1)) / 2 ** d) if d else 1.0
        print(f"[{key}] {n} questions: A {sum(x for x, _ in pairs) / n:.3f}, B {sum(y for _, y in pairs) / n:.3f}; "
              f"same verdict {same / n:.1%}; only A right {only_a}, only B right {only_b}, McNemar p = {p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="engine", required=True)
    for name in ("llamacpp", "hf"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="Ollama model name (file label for hf)")
        p.add_argument("--n", type=int, default=300)
        p.add_argument("--seed", type=int, default=0)
        if name == "llamacpp":
            p.add_argument("--ctx", type=int, default=1024)
        else:
            p.add_argument("--hf", required=True, help="Hugging Face repo of the original weights")
            p.add_argument("--gpu-mem", default="6GiB", help="GPU share; the rest goes to system RAM")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"llamacpp": run_llamacpp, "hf": run_hf, "compare": compare}[args.engine](args)


if __name__ == "__main__":
    main()
