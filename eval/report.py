"""One-page HTML summary of the model comparison (scoreboard + experiment log).

  .venv\\Scripts\\python eval\\scoreboard.py     (refresh the scoreboard first)
  .venv\\Scripts\\python eval\\report.py         -> data\\report.html (open in a browser)

Numbers come from eval/scoreboard.csv and eval/experiments.csv; the short
"status" and "key points" text below is written by hand: update it when the
picture changes.
"""
import csv
import html
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "report.html"

STATUS = [  # (step, state, text); state: done | now | next
    ("A0-A1", "done", "Local eval tooling; Atlas-Chat baselines reproduce the published DarijaMMLU scores"),
    ("A2", "done", "Small-model checks of the scoring (Qwen3.5-0.8B, Gemma-4-E2B); fixed a scorer bug"),
    ("A3", "done", "Downloaded Qwen3.5-9B, Gemma-4-12B-it and Gemma-4-E4B-it (Q4_K_M)"),
    ("A4", "done", "Scored all three as released (Gemma-4-12B: DarijaMMLU only, answer-format issue)"),
    ("A5", "done", "Training pipeline tested on small local slices (E2B, E4B): train, merge, GGUF, Ollama, score"),
    ("A6", "done", "Run 2 on a rented H100 (E4B + TULU, $32.58); all checkpoints scored except 500 (cancelled)"),
    ("Fluency", "done", "Darija-vs-MSA share of replies + DarijaBench translations, before and after training"),
    ("Ear test", "done", "Cafe + medina scripts and free chat: Atlas best at scripts, released E4B best listener"),
    ("Next", "next", "Prompt fixes, then run 3: E4B with the Atlas-Chat method on a smaller Darija mix"),
]

KEY_POINTS = [
    "Untrained, all three new models beat the current tutor (Atlas-Chat-9B) on DarijaMMLU except Gemma-4-12B, "
    "which is held back by answer format, not knowledge (see note).",
    "Qwen3.5-9B and Gemma-4-E4B are 1 point apart on DarijaMMLU (a real but small gap); E4B is half the "
    "compute and likely fits next to the HADRA voice on 8 GB.",
    "Atlas-Chat-9B still leads DarijaHellaSwag clearly; part of that gap may be our per-token scoring, "
    "which is not fair across tokenizers.",
    "Run 2 (TULU) made E4B answer in Darija without a system prompt (16% -> 86% by step 1000), but gave no "
    "gain inside the tutor prompt, cost 1-3 DarijaMMLU points, and removed its ability to read Arabizi.",
    "Training past step 1000 added nothing: 8000 and final score lower on DarijaMMLU and fluency.",
    "Ear test: Atlas-Chat-9B follows role-play scripts best; in free conversation the released E4B listens "
    "better. No model corrects mistakes reliably, and none knows Fes facts.",
    "Decided next: E4B with the Atlas-Chat method (Darija and Arabizi data, one pass); Fes facts by lookup.",
]

# hand-kept facts the CSVs do not hold
FITS_WITH_HADRA = {"atlas-chat-2b": "yes", "atlas-chat-9b": "yes (7.4 of 8.2 GB with HADRA, measured)",
                   "qwen35-9b": "likely (5.2 GB)", "gemma4-e4b": "yes (4.2 GB with HADRA, measured)",
                   "gemma4-12b": "no (7.7 GB)"}
ROW_NOTES = {
    "atlas-chat-9b": "Current tutor model.",
    "atlas-chat-2b": "Small Atlas-Chat.",
    "qwen35-9b": "Best DarijaMMLU so far; HellaSwag low (tokenizer effect unknown).",
    "gemma4-e4b": "Chosen for training (runs 2 and 3).",
    "gemma4-12b": "Puts a line break before the letter on 31% of questions; 58.7% when it answers with "
                  "a letter first. Needs a 3-shot or chat-format check.",
}
LABELS = {"atlas-chat-2b": "Atlas-Chat-2B", "atlas-chat-9b": "Atlas-Chat-9B", "atlas-chat-27b": "Atlas-Chat-27B",
          "gemma-2-9b-it": "Gemma-2-9B-it", "gemma-3-4b": "Gemma-3-4B-it", "gemmaroc-4b-tulu": "GemMaroc-4B",
          "gemmaroc-27b-tulu": "GemMaroc-27B", "qwen35-9b": "Qwen3.5-9B", "gemma4-e4b": "Gemma-4-E4B-it",
          "gemma4-12b": "Gemma-4-12B-it"}


def read(name: str) -> list[dict]:
    with open(ROOT / "eval" / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def e(x) -> str:
    return html.escape(str(x))


def fmt(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def training(b: dict) -> str:
    """Short label for the Darija data a model was trained on."""
    d = b["added_darija_data"]
    if d == "none":
        return "none (as released)"
    short = {"Darija-SFT-Mixture (Atlas-Chat)": "Atlas-Chat SFT", "TULU-3 50K Darija-English (GemMaroc)": "GemMaroc TULU"}
    return f"{short.get(d, d)}, {b['darija_examples']}"


def best_run(runs: list[dict], model: str, bench: str) -> dict | None:
    """Latest complete full-set run in the standard setting (what the scoreboard uses)."""
    full = {"darijammlu": "21792", "darijahellaswag": "10042"}[bench]
    c = [r for r in runs if r["model"] == model and r["benchmark"] == bench and r["status"] == "complete"
         and r["n"] == full and r["shots"] == "0" and r["template"] == "raw"]
    return c[-1] if c else None


def chart(rows: list[dict]) -> str:
    """Horizontal bars: our full-set DarijaMMLU with 95% CI; tick = published score; dashed = chance."""
    lo_axis, hi_axis = 20, 70
    x = lambda v: (v - lo_axis) / (hi_axis - lo_axis) * 100
    out = []
    for r in rows:
        acc, lo, hi, pub = r["acc"], r["lo"], r["hi"], r["pub"]
        tip = f"{r['label']}: {acc:.2f} (95% CI {lo:.2f}-{hi:.2f})" + (f"; published {pub:.2f}" if pub else "")
        pub_mark = f'<span class="pub" style="left:{x(pub):.2f}%"></span>' if pub else ""
        out.append(f"""
      <div class="bar-row" tabindex="0" data-tip="{e(tip)}">
        <div class="bar-label">{e(r['label'])}</div>
        <div class="bar-track">
          <span class="bar" style="width:{x(acc):.2f}%"></span>
          <span class="ci" style="left:{x(lo):.2f}%;width:{x(hi) - x(lo):.2f}%"></span>
          {pub_mark}
          <span class="chance" style="left:{x(27.9):.2f}%"></span>
        </div>
        <div class="bar-value">{acc:.1f}</div>
      </div>""")
    ticks = "".join(f'<span style="left:{x(t):.2f}%">{t}</span>' for t in range(lo_axis, hi_axis + 1, 10))
    return f"""
    <div class="chart" role="img" aria-label="DarijaMMLU scores, full set, our local runs">
      {''.join(out)}
      <div class="bar-row axis"><div class="bar-label"></div><div class="bar-track ticks">{ticks}</div><div class="bar-value"></div></div>
      <div class="legend">
        <span><i class="lg-bar"></i>Our score (full set, 0-shot)</span>
        <span><i class="lg-ci"></i>95% confidence interval</span>
        <span><i class="lg-pub"></i>Published score</span>
        <span><i class="lg-chance"></i>Random guessing (27.9)</span>
      </div>
      <div class="tip" hidden></div>
    </div>"""


def main() -> None:
    board = read("scoreboard.csv")
    runs = read("experiments.csv")

    tested, published_only = [], []
    for b in board:
        rid = b["row_id"]
        m = best_run(runs, b["ollama_model"], "darijammlu") if b["ollama_model"] else None
        h = best_run(runs, b["ollama_model"], "darijahellaswag") if b["ollama_model"] else None
        if m or h:
            tested.append({"id": rid, "b": b, "m": m, "h": h})
        elif b["status"].startswith("published"):
            published_only.append(b)

    bars = sorted(({"label": LABELS.get(t["id"], t["id"]), "acc": num(t["m"]["acc"]), "lo": num(t["m"]["ci_low"]),
                    "hi": num(t["m"]["ci_high"]), "pub": num(t["b"]["darijammlu_pub"])}
                   for t in tested if t["m"]), key=lambda r: -r["acc"])

    main_rows = []
    for t in sorted(tested, key=lambda t: -(num(t["m"]["acc"]) if t["m"] else 0)):
        b, m, h = t["b"], t["m"], t["h"]
        mm = f'<b>{fmt(num(m["acc"]), 2)}</b><small>{fmt(num(m["ci_low"]))}-{fmt(num(m["ci_high"]))}</small>' if m else "–"
        split = f'{fmt(num(m["acc_arabic_mmlu"]))} / {fmt(num(m["acc_mmlu"]))}' if m else "–"
        hs = f'<b>{fmt(num(h["acc"]), 2)}</b><small>{fmt(num(h["ci_low"]))}-{fmt(num(h["ci_high"]))}</small>' if h else "not run"
        main_rows.append(f"""
        <tr>
          <th scope="row">{e(LABELS.get(t['id'], t['id']))}<small>{e(b['vendor'])}</small></th>
          <td>{e(b['size_b'])}B</td>
          <td>{e(training(b))}</td>
          <td class="n">{mm}</td>
          <td class="n">{e(b['darijammlu_pub'] or '–')}</td>
          <td class="n">{split}</td>
          <td class="n">{hs}</td>
          <td>{e(FITS_WITH_HADRA.get(t['id'], '?'))}</td>
          <td class="note">{e(ROW_NOTES.get(t['id'], ''))}</td>
        </tr>""")

    pub_rows = "".join(f"""
        <tr><th scope="row">{e(LABELS.get(b['row_id'], b['row_id']))}<small>{e(b['vendor'])}</small></th>
          <td>{e(b['size_b'])}B</td><td>{e(training(b))}</td>
          <td class="n">{e(b['darijammlu_pub'] or '–')}</td><td class="n">{e(b['darijahellaswag_pub'] or '–')}</td>
          <td class="note">{e(b['notes'])}</td></tr>""" for b in published_only)

    # quick checks + full log, newest first
    def run_row(r: dict) -> str:
        setting = "per-token" if r["benchmark"] == "darijahellaswag" else f"{r['scoring']}, {r['template']}, {r['shots']}-shot"
        bench = {"darijammlu": "DarijaMMLU", "darijahellaswag": "HellaSwag"}.get(r["benchmark"], r["benchmark"])
        ok = r["status"] == "complete"
        return (f"<tr class=\"{'' if ok else 'dim'}\"><td>{e(r['started'][:16].replace('T', ' '))}</td><td>{e(r['model'])}</td>"
                f"<td>{e(bench)}</td><td>{e(setting)}</td><td class=\"n\">{int(r['n']):,}</td>"
                f"<td class=\"n\">{e(r['acc'])}</td><td>{e(r['status'])}</td><td class=\"note\">{e(r['note'])}</td></tr>")
    log_rows = "".join(run_row(r) for r in reversed(runs))

    status_html = "".join(f'<li class="{s}"><span class="chip">{ {"done": "Done", "now": "In progress", "next": "Next"}[s] }</span>'
                          f'<b>{e(step)}</b> {e(text)}</li>' for step, s, text in STATUS)
    points_html = "".join(f"<li>{e(p)}</li>" for p in KEY_POINTS)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Darija Model Scoreboard</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f6f4; --surface: #fcfcfb; --border: #e3e2de; --grid: #ecebe7;
  --text: #0b0b0b; --text-2: #52514e; --muted: #7b7a75;
  --series: #2a78d6; --ci: #0b0b0b; --pub: #eb6834;
  --done: #008300; --now: #c98500; --next: #7b7a75;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #33332f; --grid: #2a2a27;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #9a998f;
    --series: #3987e5; --ci: #f0efec; --pub: #d95926;
    --done: #3fae3f; --now: #e0a000; --next: #9a998f;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #33332f; --grid: #2a2a27;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #9a998f;
  --series: #3987e5; --ci: #f0efec; --pub: #d95926;
  --done: #3fae3f; --now: #e0a000; --next: #9a998f;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 0 0 12px; }}
.sub {{ color: var(--text-2); margin: 0 0 24px; }}
section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px; margin-bottom: 20px; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.grid2 section {{ margin: 0; }}
@media (max-width: 760px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
ul.status {{ list-style: none; margin: 0; padding: 0; }}
ul.status li {{ padding: 6px 0; border-bottom: 1px solid var(--grid); }}
ul.status li:last-child {{ border: 0; }}
.chip {{ display: inline-block; min-width: 86px; font-size: 12px; font-weight: 600; margin-right: 8px; }}
li.done .chip {{ color: var(--done); }} li.done .chip::before {{ content: "✓ "; }}
li.now .chip {{ color: var(--now); }} li.now .chip::before {{ content: "▶ "; }}
li.next .chip {{ color: var(--next); }} li.next .chip::before {{ content: "○ "; }}
ul.points {{ margin: 0; padding-left: 20px; }} ul.points li {{ margin-bottom: 8px; }}
.chart {{ position: relative; }}
.bar-row {{ display: grid; grid-template-columns: 150px 1fr 48px; align-items: center; gap: 12px; height: 34px; }}
.bar-row:focus {{ outline: 2px solid var(--series); outline-offset: 2px; border-radius: 4px; }}
.bar-label {{ text-align: right; color: var(--text-2); font-size: 14px; }}
.bar-value {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
.bar-track {{ position: relative; height: 100%;
  background: repeating-linear-gradient(to right, var(--grid) 0 1px, transparent 1px 20%); }}
.bar {{ position: absolute; left: 0; top: 9px; height: 16px; background: var(--series); border-radius: 0 4px 4px 0; }}
.ci {{ position: absolute; top: 16px; height: 2px; background: var(--ci); }}
.pub {{ position: absolute; top: 5px; width: 3px; height: 24px; background: var(--pub); border-radius: 2px; }}
.chance {{ position: absolute; top: 0; bottom: 0; border-left: 2px dashed var(--muted); }}
.axis {{ height: 22px; }}
.ticks {{ background: none; }}
.ticks span {{ position: absolute; transform: translateX(-50%); font-size: 12px; color: var(--muted); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0 0 162px; font-size: 13px; color: var(--text-2); }}
@media (max-width: 760px) {{ .legend {{ margin-left: 0; }} .bar-row {{ grid-template-columns: 110px 1fr 40px; }} }}
.legend i {{ display: inline-block; vertical-align: middle; margin-right: 6px; }}
.lg-bar {{ width: 14px; height: 10px; background: var(--series); border-radius: 2px; }}
.lg-ci {{ width: 16px; height: 2px; background: var(--ci); }}
.lg-pub {{ width: 3px; height: 14px; background: var(--pub); }}
.lg-chance {{ width: 0; height: 14px; border-left: 2px dashed var(--muted); }}
.tip {{ position: absolute; pointer-events: none; background: var(--text); color: var(--surface);
  font-size: 13px; padding: 6px 10px; border-radius: 6px; white-space: nowrap; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }}
thead th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .03em; color: var(--muted); font-weight: 600; }}
tbody th {{ font-weight: 600; white-space: nowrap; }}
th small, td small {{ display: block; color: var(--muted); font-weight: 400; font-size: 12px; }}
td.n {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.note {{ color: var(--text-2); min-width: 220px; }}
tr.dim td {{ color: var(--muted); }}
details summary {{ cursor: pointer; font-weight: 600; }}
.caveats dt {{ font-weight: 600; margin-top: 10px; }}
.caveats dd {{ margin: 2px 0 0; color: var(--text-2); }}
footer {{ color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>Darija Model Scoreboard</h1>
  <p class="sub">Which model should power the Darija tutor? Local benchmark results so far.
    Generated {datetime.now():%Y-%m-%d %H:%M} from <code>eval/scoreboard.csv</code> and <code>eval/experiments.csv</code>.</p>

  <div class="grid2">
    <section><h2>Where we are</h2><ul class="status">{status_html}</ul></section>
    <section><h2>Key points</h2><ul class="points">{points_html}</ul></section>
  </div>

  <section>
    <h2>DarijaMMLU: understanding + knowledge in Darija</h2>
    <p class="sub" style="margin-bottom:12px">21,792 multiple-choice questions, full set, same prompt and scorer for every model (higher is better). Hover a bar for exact numbers.</p>
    {chart(bars)}
  </section>

  <section>
    <h2>Models we have tested locally</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>Size</th><th>Darija training</th><th>DarijaMMLU<small>ours, 95% CI</small></th>
        <th>Published</th><th>Exam / translated<small>DarijaMMLU parts</small></th>
        <th>HellaSwag<small>ours, per-token</small></th><th>Fits next to voice</th><th>Note</th></tr></thead>
      <tbody>{''.join(main_rows)}</tbody>
    </table></div>
  </section>

  <section>
    <h2>How to read these numbers</h2>
    <dl class="caveats">
      <dt>DarijaMMLU is the fair cross-model comparison</dt>
      <dd>One answer letter per question, no length effects. Our setup reproduces Atlas-Chat's published scores (58.42 vs 58.23).</dd>
      <dt>HellaSwag (ours) is not fair across model families</dt>
      <dd>It averages per token, and Qwen, Gemma and Atlas split Darija into tokens differently. Fine within one family
        (before vs after training); a tokenizer-fair per-byte version is planned. It also runs higher than the published numbers.</dd>
      <dt>Neither one measures fluency</dt>
      <dd>Both are multiple choice: the model never writes. Whether a model answers in natural Darija or in Modern Standard
        Arabic needs the fluency step and your ear test.</dd>
      <dt>Small gaps</dt>
      <dd>On the full set the 95% interval is about ±0.65 points. Models are compared on the same questions with a paired test
        (Qwen3.5-9B vs Atlas-Chat-9B: real; Qwen3.5-9B vs Gemma-4-E4B: real but only 1 point).</dd>
    </dl>
  </section>

  <section>
    <h2>Published scores of other Darija models (not run locally)</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>Size</th><th>Darija training</th><th>DarijaMMLU</th><th>HellaSwag<small>per-byte</small></th><th>Note</th></tr></thead>
      <tbody>{pub_rows}</tbody>
    </table></div>
  </section>

  <section>
    <details>
      <summary>All runs ({len(runs)}), newest first</summary>
      <div class="table-wrap" style="margin-top:12px"><table>
        <thead><tr><th>Started</th><th>Model</th><th>Benchmark</th><th>Setting</th><th>Questions</th><th>Score</th><th>Status</th><th>Note</th></tr></thead>
        <tbody>{log_rows}</tbody>
      </table></div>
    </details>
  </section>

  <footer>All local scores: Q4_K_M models in Ollama / llama.cpp on an RTX 4070 Laptop (8 GB).</footer>
</main>
<script>
document.querySelectorAll('.chart').forEach(chart => {{
  const tip = chart.querySelector('.tip');
  const show = (row, x, y) => {{
    tip.textContent = row.dataset.tip; tip.hidden = false;
    const r = chart.getBoundingClientRect();
    tip.style.left = Math.min(x - r.left + 12, r.width - tip.offsetWidth) + 'px';
    tip.style.top = (y - r.top - 36) + 'px';
  }};
  chart.querySelectorAll('.bar-row[data-tip]').forEach(row => {{
    row.addEventListener('mousemove', ev => show(row, ev.clientX, ev.clientY));
    row.addEventListener('mouseleave', () => tip.hidden = true);
    row.addEventListener('focus', () => {{ const b = row.getBoundingClientRect(); show(row, b.left + b.width / 2, b.top); }});
    row.addEventListener('blur', () => tip.hidden = true);
  }});
}});
</script>
</body>
</html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
