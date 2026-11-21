#!/usr/bin/env python3
"""Paper appendix E: third order (A3) split by context. CPU only; reads the same files as paper/claims.py.

For every model, source block, direction type and context: the median over directions of the large-step errors
at the smallest push (J, J+Q, J+Q+T, J+Q+T with single-piece third-order terms only), the median between-block
share of T, and the share of the T fix that single-piece terms give, (JQ - JQT_self) / (JQ - JQT), where T helps.
Writes results/a3_by_context.csv and prints a compact table.
Run:  QLENS_OUT=<outputs dir> python paper/a3_by_context.py
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

W = Path(os.environ.get("QLENS_OUT", "outputs"))
OUT = Path(__file__).resolve().parents[1] / "results" / "a3_by_context.csv"
FILES = {"Qwen3.8-27B": ("qanat_q38-27b", [f"A3_s{s}_32.json" for s in (8, 24, 40)]),
         "Qwen3-8B": ("qanat_qwen3-8b", ["A3.json"])}
rows_out = []
for model, (d, files) in FILES.items():
    for f in files:
        js = json.loads((W / d / f).read_text())
        C, D, a = js["n_ctx"], js["n_dir"], min(js["alphas"])
        for s, v in js["sources"].items():
            for tn in ("iso", "data"):
                rows = v[tn]["rows"]
                for c in range(C):
                    rr = rows[c * D:(c + 1) * D]          # context-major, as in claims.py grid()
                    e = {p: float(np.median([r["large_step"][str(a)][p][0] for r in rr]))
                         for p in ("J", "JQ", "JQT", "JQT_self")}
                    btw = float(np.median([r["pair"]["cross_jk_share"] for r in rr]))
                    fix = e["JQ"] - e["JQT"]
                    single = (e["JQ"] - e["JQT_self"]) / fix if fix > 0 else float("nan")
                    rows_out.append(dict(model=model, src=s, type=tn, ctx=c, alpha=a, err_J=e["J"], err_JQ=e["JQ"],
                                         err_JQT=e["JQT"], err_JQT_self=e["JQT_self"], between_share=btw,
                                         single_piece_share_of_fix=single, file=f"{d}/{f}"))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    for r in rows_out:
        w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
for r in rows_out:
    print(f"{r['model']:12s} s{r['src']:>2s} {r['type']:4s} c{r['ctx']} a={r['alpha']}  J {100*r['err_J']:7.2f}%  "
          f"JQ {100*r['err_JQ']:7.2f}%  JQT {100*r['err_JQT']:7.2f}%  self {100*r['err_JQT_self']:7.2f}%  "
          f"between {r['between_share']:.2f}  single/fix {r['single_piece_share_of_fix']:.2f}")
