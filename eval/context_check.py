"""How far can the tutor's context window (num_ctx) grow on the 8 GB laptop, next to HADRA?
(PLAN.md roadmap: E4B is built for 131K tokens, Atlas-Chat-9B for 8K; both run at 2048 today.)

For each num_ctx the model is reloaded and gets the real tutor system prompt plus a chat history
of real tutor turns (from the latest eval/ear_replay.py run of that model), filled to ~85% of the
window. The learner's first line plants a detail (the school cat's name); the last line asks for it.
A control run asks right after the planted line (no filler). Per size it records:
  memory    GPU memory used (nvidia-smi, includes HADRA) and the model's share on the GPU (ollama ps)
  speed     prompt processing (tokens/s), reply speed (tokens/s), seconds for the first (cold) reply, without model loading; later trials reuse Ollama's prompt cache
  recall    trials (temperature 0.6, different seeds) whose reply names the cat
  format    replies with Arabic lines each followed by an EN: line
  voice     HADRA still synthesizes a sentence with the model loaded

  .venv\\Scripts\\python eval\\context_check.py --model gemma-4-e4b-ours --ctx 2048 4096 8192 16384
Start HADRA first (.\\talk.cmd starts it; or .venv-hadra\\python.exe scripts\\hadra_server.py).
Writes data/eval_runs/<timestamp>_<model>_context.json.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import httpx

import registry

sys.path.insert(0, str(registry.ROOT / "src"))
from darija_tutor import config, tts  # noqa: E402
from darija_tutor.tutor import Tutor, pair_lines  # noqa: E402

OLLAMA = config.OLLAMA_URL
# Planted in Arabic script and asked in plain English, so the test is memory, not reading Arabizi
# (a first version asked in Arabizi: most replies took "smit lmech" for the school's name).
PLANT = "سلام! على فكرة، فالمدرسة ديالي ففاس كان واحد القط، سميتو زعفران. كلشي كان كيبغيه."
ASK = "Do you remember the name of the cat at my school in Fes? I told you at the start."
CAT = re.compile(r"زعفران|zaafran|za3fran", re.I)


def gpu_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def unload(model: str) -> None:
    httpx.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(3)


def chat(model: str, messages: list[dict], ctx: int, seed: int, num_predict: int = 300) -> dict:
    resp = httpx.post(f"{OLLAMA}/api/chat", timeout=600, json={
        "model": model, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": 0.6, "seed": seed, "num_ctx": ctx, "num_predict": num_predict}})
    resp.raise_for_status()
    return resp.json()


def filler_turns(model: str) -> list[dict]:
    """Learner/tutor pairs from the model's latest ear replay (real replies in the tutor format)."""
    runs = sorted(glob.glob(str(registry.ROOT / "data" / "ear_runs" / f"*_{model}.json")), key=os.path.getmtime)
    if not runs:
        sys.exit(f"no ear replay for {model}: run eval/ear_replay.py --model {model} first")
    turns = []
    for run in json.load(open(runs[-1], encoding="utf-8"))["runs"]:
        for t in run["turns"]:
            if t["learner"]:
                turns += [{"role": "user", "content": t["learner"]}, {"role": "assistant", "content": t["reply"]}]
    return turns


def build(system: str, filler: list[dict], n_pairs: int) -> list[dict]:
    hist = [{"role": "system", "content": system}, {"role": "user", "content": PLANT},
            {"role": "assistant", "content": "زعفران؟ سمية زوينة!\nEN: Zaafrane? A nice name!"}]
    for i in range(n_pairs):
        hist += filler[(2 * i) % len(filler):(2 * i) % len(filler) + 2]
    return hist + [{"role": "user", "content": ASK}]


def good_format(reply: str) -> bool:
    pairs = pair_lines(reply)
    return bool(pairs) and all(a and e for a, e in pairs)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--fill", type=float, default=0.85, help="share of num_ctx to fill before the question")
    args = ap.parse_args()
    if not tts.available():
        sys.exit("HADRA is not running: start it first, the point is to measure next to it")

    system = Tutor("discussion", model=args.model, persona="noual").history[0]["content"]
    filler = filler_turns(args.model)
    results = []
    # control first: the question right after the planted line (no filler), at the smallest window
    for ctx, control in [(min(args.ctx), True)] + [(c, False) for c in args.ctx]:
        unload(args.model)
        # grow the history until the prompt fills the window (count with 1-token replies)
        n, prompt_tokens = 4, 0
        while not control:
            used = chat(args.model, build(system, filler, n), ctx, 0, num_predict=1)["prompt_eval_count"]
            if used >= args.fill * ctx - 150 or used < prompt_tokens:  # full, or Ollama started truncating
                break
            prompt_tokens = used
            n = max(n + 1, int(n * (args.fill * ctx) / used))
        n = 0 if control else n
        messages = build(system, filler, n)
        unload(args.model)  # the counting calls left this prompt in Ollama's cache: time it cold
        row = {"ctx": ctx, "pairs": n, "trials": []}
        for seed in range(args.trials):
            r = chat(args.model, messages, ctx, seed)
            reply = r["message"]["content"]
            row["trials"].append({
                "seed": seed, "prompt_tokens": r["prompt_eval_count"], "reply_tokens": r["eval_count"],
                "prompt_tps": round(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)),
                "reply_tps": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1),
                "seconds": round((r["total_duration"] - r.get("load_duration", 0)) / 1e9, 2), "recall": bool(CAT.search(reply)),
                "format": good_format(reply), "reply": reply})
        ps = httpx.get(f"{OLLAMA}/api/ps", timeout=30).json()["models"][0]
        row |= {"gpu_share": round(ps["size_vram"] / ps["size"], 3), "model_gb": round(ps["size"] / 1e9, 2)}
        row["gpu_mb_loaded"] = gpu_mb()
        t0 = time.perf_counter()
        try:
            tts.synthesize("لاباس، الحمد لله. شنو بغيتي نهضرو عليه اليوم؟", "F1")
            row["voice_s"] = round(time.perf_counter() - t0, 1)
        except Exception as e:  # noqa: BLE001 - a failed voice is a result, not a crash
            row["voice_s"] = f"failed: {e}"
        row["gpu_mb_after_voice"] = gpu_mb()
        results.append(row)
        tr = row["trials"]
        row["control"] = control
        print(f"{'control' if control else 'ctx'} {ctx:6d}: prompt {tr[0]['prompt_tokens']:6d} tok, GPU {row['gpu_mb_loaded']} MB "
              f"(after voice {row['gpu_mb_after_voice']}), on GPU {row['gpu_share']:.0%}, "
              f"prompt {tr[0]['prompt_tps']} tok/s, reply {tr[0]['reply_tps']} tok/s, first answer "
              f"{tr[0]['seconds']} s, recall {sum(t['recall'] for t in tr)}/{len(tr)}, "
              f"format {sum(t['format'] for t in tr)}/{len(tr)}, voice {row['voice_s']} s", flush=True)
    unload(args.model)
    out = registry.RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.model}_context.json"
    out.write_text(json.dumps({"model": args.model, "code_commit": registry.code_commit(), "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("details:", out)


if __name__ == "__main__":
    main()
