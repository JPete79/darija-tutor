"""Darija-vs-MSA scores for generated text (used by fluency.py; needs torch, so it runs in .venv-train).

Two small classifiers, run on the CPU so they never compete with Ollama for the GPU:
  ALDi    AMR-KELEG/Sentence-ALDi: level of dialectness, 0 = MSA ... 1 = strong dialect
          (any dialect, not only Moroccan)
  router  yrrhall/dialect-router-v0.1: which variety (ma = Moroccan, ar = MSA, eg, tn, ...)

Checked on 300 DarijaBench MSA->Darija pairs (MADAR, DODa, FLORES+, 100 each):
MSA sides ALDi 0.01-0.04, router 'ar' 74-80%; Darija sides ALDi 0.68/0.74 (FLORES 0.46:
formal translated text), router 'ma' 62-75%. Per sentence it errs about 1 in 4 times,
so read it as a share over many sentences, not a verdict on one.

Each text is cut into Arabic-script segments (lines, then sentences); `EN:` lines and
Latin-only lines are skipped. Rewrites the .jsonl in place, adding per text:
  segments, aldi (mean), labels (router label per segment), latin_share (share of
  letters that are Latin: an English or Arabizi answer instead of Arabic script)

  .venv-train\\python.exe eval\\dialect_scores.py data\\eval_runs\\<run>.jsonl --field reply
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def segments(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("EN:") or not ARABIC.search(line):
            continue
        out += [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", line) if ARABIC.search(s)]
    return out


class Scorer:
    def __init__(self):
        self.aldi = [f("AMR-KELEG/Sentence-ALDi") for f in (AutoTokenizer.from_pretrained,
                                                             AutoModelForSequenceClassification.from_pretrained)]
        self.router = [f("yrrhall/dialect-router-v0.1") for f in (AutoTokenizer.from_pretrained,
                                                                   AutoModelForSequenceClassification.from_pretrained)]

    @torch.no_grad()
    def _logits(self, pair, texts):
        tok, model = pair
        return model.eval()(**tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)).logits

    def score(self, text: str) -> dict:
        segs = segments(text)
        letters = len(ARABIC.findall(text)) + len(LATIN.findall(re.sub(r"(?im)^\s*EN:.*$", "", text)))
        latin = len(LATIN.findall(re.sub(r"(?im)^\s*EN:.*$", "", text))) / letters if letters else 0.0
        if not segs:
            return {"segments": 0, "aldi": None, "labels": [], "latin_share": round(latin, 3)}
        aldi = self._logits(self.aldi, segs)[:, 0].clamp(0, 1)
        id2label = self.router[1].config.id2label
        labels = [id2label[int(i)] for i in self._logits(self.router, segs).argmax(-1)]
        return {"segments": len(segs), "aldi": round(float(aldi.mean()), 3), "labels": labels,
                "latin_share": round(latin, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--field", default="reply", help="text field to score; scores go to <field>_dialect")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path(args.jsonl)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    scorer = Scorer()
    for r in rows:
        r[f"{args.field}_dialect"] = scorer.score(r.get(args.field) or "")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"scored {len(rows)} rows in {path.name}")


if __name__ == "__main__":
    main()
