"""High-level scoreboard: one row per model variant, published vs our scores.

eval/scoreboard.csv is edited by hand (add a row per new model or training
variant, set `ollama_model` once it can run locally). This script fills the
`*_ours` columns from eval/experiments.csv: the latest complete full-set run
in the standard setting (see STANDARD; closest to how the published numbers
were measured). 500-question checks never go on the scoreboard.

  .venv\\Scripts\\python eval\\scoreboard.py
"""
import csv
import sys

import registry

BOARD = registry.ROOT / "eval" / "scoreboard.csv"
FULL_SIZE = {"darijammlu": 21792, "darijahellaswag": 10042}  # add each benchmark when its runner exists
STANDARD = {
    "darijammlu": {"scoring": "logprob", "template": "raw", "shots": "0"},
    "darijahellaswag": {"scoring": "acc_norm_per_token", "template": "raw", "shots": "0"},
}


def best_runs() -> dict[tuple[str, str], dict]:
    runs = {}
    for r in registry.read():
        full = FULL_SIZE.get(r["benchmark"])
        if (r["status"] == "complete" and full and r["n"] == str(full)
                and all(r[k] == v for k, v in STANDARD[r["benchmark"]].items())):
            runs[(r["model"], r["benchmark"])] = r  # log is in time order, so the latest wins
    return runs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(BOARD, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    runs = best_runs()
    for row in rows:
        for bench in FULL_SIZE:
            run = runs.get((row["ollama_model"], bench))
            if run:
                row[f"{bench}_ours"] = run["acc"]
    with open(BOARD, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    cols = ["row_id", "base_model", "added_darija_data", "darijammlu_pub", "darijammlu_ours",
            "darijahellaswag_pub", "darijahellaswag_ours"]
    widths = {c: max(len(c), *(len(r[c][:38]) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(r[c][:38].ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
