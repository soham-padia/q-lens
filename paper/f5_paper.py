#!/usr/bin/env python3
"""Paper figure F5 (third order).
Panel (b) shows the large-step errors only where each added Taylor order lowers the error at a 5% push: Qwen3-8B (medians
over its four contexts, all of which improve) and the five Qwen3.8-27B (push block, context) cases that improve, one bar
group each (block 8 context 0, block 24 contexts 0-1, block 40 contexts 0-1; results/a3_by_context.csv). The pooled 27B medians mix
contexts where added orders help with contexts where none does (appendix F), so they are not plotted. Reads results/claims.csv and results/a3_by_context.csv.
Writes figures/F5_third_order_paper.pdf/.png.
Run:  python paper/f5_paper.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

A = Path(__file__).resolve().parents[1]
P = A / "results"
ORANGE, ORANGE_L, BLUE, BLUE_L, NAVY = "#C9551F", "#E8A77F", "#2A6FC9", "#8DB4E6", "#163E73"
INK, MUTED, GRID = "#16202E", "#6B6A66", "#E4E2DC"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "lines.linewidth": 1.6, "savefig.dpi": 200, "pdf.fonttype": 42})
R = list(csv.DictReader(open(P / "claims.csv")))
for r in R:
    for k in ("value", "lo", "hi"):
        r[k] = float(r[k])


def get(model, quantity, pred):
    return [r for r in R if r["claim"] == "C5" and r["model"] == model and r["quantity"] == quantity
            and pred(r["setting"])]


def kv(setting, key):
    for tok in setting.split():
        if tok.startswith(key + "="):
            return float(tok.split("=")[1])


fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.1, 1.15]})
xt, xl, x = [], [], 0
pair_c = []
for m in ("Qwen3.8-27B", "Qwen3-8B"):
    for s in sorted({kv(r["setting"], "src") for r in get(m, "self_share_mean", lambda st: True)}):
        for tn in ("iso", "data"):
            pr = lambda st, s=s, tn=tn: kv(st, "src") == s and f" {tn} " in st + " " and "alpha" not in st
            v = [get(m, f"{q}_share_mean", pr)[0]["value"] for q in ("self", "within_block", "between_block")]
            bot = 0
            for vv, col in zip(v, (ORANGE, ORANGE_L, BLUE)):
                a.bar(x, vv, bottom=bot, color=col, width=0.8, lw=0)
                bot += vv
            xt.append(x)
            xl.append("")
            x += 1
        pair_c.append((x - 1.5, int(s)))
        x += 0.4
    x += 0.8
a.set_xticks([c for c, _ in pair_c], [str(n) for _, n in pair_c], fontsize=6.8)
a.set_ylim(0, 1.0)
a.set_ylabel("share of the third-order term T")
a.text(0, 1.02, "Qwen3.8-27B", fontsize=7.5, color=INK, transform=a.get_xaxis_transform())
a.text(xt[len(xt) // 2], 1.02, "Qwen3-8B", fontsize=7.5, color=INK, transform=a.get_xaxis_transform())
a.legend(handles=[Patch(color=BLUE, label="between blocks: a block bending another block's bend"),
                  Patch(color=ORANGE_L, label="within a block: a site bending a bend made earlier in its block"),
                  Patch(color=ORANGE, label="self: each site on its own")],
         frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.06), ncol=1)
a.set_xlabel("push block (each pair: random, then data directions)")
a.grid(axis="x", visible=False)

# Qwen3-8B: medians over its four contexts (every context improves with each order). Qwen3.8-27B: the three
# (push block, context) cases in which each added order lowers the error, one bar group each (results/a3_by_context.csv).
A3 = {(r["model"], r["src"], r["type"], r["ctx"]): r for r in csv.DictReader(open(P / "a3_by_context.csv"))}
groups = [("Qwen3-8B", 4, None, "4"), ("Qwen3-8B", 14, None, "14"), ("Qwen3-8B", 22, None, "22"),
          ("Qwen3.8-27B", 8, "0", "8\nc0"), ("Qwen3.8-27B", 24, "0", "24\nc0"), ("Qwen3.8-27B", 24, "1", "24\nc1"),
          ("Qwen3.8-27B", 40, "0", "40\nc0"), ("Qwen3.8-27B", 40, "1", "40\nc1")]
preds = [("J", "J", ORANGE), ("JQ", "J + Q", BLUE), ("JQT_self", "J + Q + self part of T", BLUE_L),
         ("JQT", "J + Q + all of T", NAVY)]
w = 0.2
for j, (pk, lab, col) in enumerate(preds):
    vals = []
    for m, s, ctx, _ in groups:
        if ctx is None:
            vals.append(get(m, f"large_step_err_{pk}",
                            lambda st, s=s: kv(st, "src") == s and " iso " in st and "alpha=0.05" in st)[0]["value"])
        else:
            vals.append(float(A3[(m, str(s), "iso", ctx)][f"err_{pk}"]))
    b.bar(np.arange(len(groups)) + (j - 1.5) * w, vals, width=w, color=col, label=lab, lw=0)
b.set_yscale("log")
b.set_ylim(3e-4, 6)
b.set_xticks(range(len(groups)), [g[3] for g in groups], fontsize=6.3)
b.text(1, -0.2, "Qwen3-8B", ha="center", va="top", fontsize=6.8, color=INK, transform=b.get_xaxis_transform())
b.text(5, -0.2, "Qwen3.8-27B", ha="center", va="top", fontsize=6.8, color=INK, transform=b.get_xaxis_transform())
b.set_xlabel("push block (random directions)", labelpad=24)
b.set_ylabel("relative error, push = 5%\nof a typical state's norm")
b.set_title("Where each added order helps", loc="left")
b.legend(frameon=False, fontsize=6.2, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.2)
b.grid(axis="x", visible=False)
(A / "figures").mkdir(exist_ok=True)
fig.subplots_adjust(wspace=0.38)
for ext in ("pdf", "png"):
    fig.savefig(A / "figures" / f"F5_third_order_paper.{ext}", bbox_inches="tight")
print("wrote", A / "figures" / "F5_third_order_paper.pdf")
