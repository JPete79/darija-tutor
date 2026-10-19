"""Compare two DarijaMMLU runs on the same questions (paired comparison).

Two accuracies 2 points apart on 500 questions can still be noise. Because
both runs answered the same questions, the useful evidence is the questions
where they disagree: if A is right and B wrong about as often as the reverse,
the difference is not real. McNemar's exact test turns that into a p-value.

  .venv\\Scripts\\python eval\\compare_runs.py data\\eval_runs\\A.jsonl data\\eval_runs\\B.jsonl
"""
import json
import math
import sys
from pathlib import Path


def load(path: str) -> dict[str, bool]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["correct"] for r in map(json.loads, f)}


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided p-value: chance of a split at least this lopsided if A and B were equally good."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    a_path, b_path = sys.argv[1:3]
    a, b = load(a_path), load(b_path)
    ids = a.keys() & b.keys()
    if not ids:
        sys.exit("No shared question ids: use the same --n and --seed for both runs.")
    both = sum(a[i] and b[i] for i in ids)
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(b[i] and not a[i] for i in ids)
    n = len(ids)
    print(f"A = {Path(a_path).stem}\nB = {Path(b_path).stem}\n")
    print(f"shared questions   {n}")
    print(f"accuracy A / B     {(both + only_a) / n:.3f} / {(both + only_b) / n:.3f}")
    print(f"both right         {both}")
    print(f"only A right       {only_a}")
    print(f"only B right       {only_b}")
    print(f"both wrong         {n - both - only_a - only_b}")
    p = mcnemar_exact(only_a, only_b)
    verdict = "difference is likely real" if p < 0.05 else "could be noise"
    print(f"McNemar p          {p:.4f}  ({verdict} at the usual 0.05 cut-off)")


if __name__ == "__main__":
    main()
