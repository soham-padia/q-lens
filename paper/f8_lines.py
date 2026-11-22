#!/usr/bin/env python3
"""Paper figure F8 (Step 5, which lines act together).

(a) One Mathlib context at block 29, target t*: the exact pair map D2 next to the path-averaged Q map. The context is
    chosen by a fixed rule: the Mathlib context whose path-averaged-Q Spearman at block 29 is the lower median over the
    16 contexts. Mathlib is Apache-2.0, so its lines may be shown (results/fig_inputs/F8_example.json, written by
    paper/step5_inputs.py). No research-library text is read here.
(b) Median Spearman correlation with exact removal by block, for Q at the state, path-averaged Q and Q3 (claims.csv).
(c) Detection AUC by block, real against null contexts (claims.csv).
Reads results/claims.csv, results/step5/resp_map_v2/resp_v2_ML.json and results/fig_inputs/F8_example.json.
Writes figures/F8_lines.pdf/.png.
Run: python paper/f8_lines.py
"""
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = Path(__file__).resolve().parents[1]
P = A / "results"
FIG = A / "figures"
W = P / "step5" / "resp_map_v2"
BLUE, NAVY, INK, MUTED, GRID, PURPLE = "#2A6FC9", "#163E73", "#16202E", "#6B6A66", "#E4E2DC", "#B4589A"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.5, "savefig.dpi": 200, "pdf.fonttype": 42})
MCOL = {"D2": INK, "Q": BLUE, "pQ": NAVY, "Q3": PURPLE}
MLAB = {"D2": "exact removal $D_2$", "Q": "$Q$ at the state", "pQ": r"path-averaged $\bar Q$", "Q3": "third order $Q_3$"}
CORP = {"ML": ("Mathlib", "o", "-"), "FW": ("prose", "s", "--"), "RH": ("research library", "^", ":")}
BLOCKS = [11, 20, 29]

R = {}
for r in csv.DictReader(open(P / "claims.csv")):
    R[(r["model"], r["setting"], r["quantity"])] = r


def g(m, s, q, f="value"):
    return float(R[(m, s, q)][f])


def rank(xs):
    o = sorted(range(len(xs)), key=lambda k: xs[k])
    r = [0.0] * len(xs)
    for n, k in enumerate(o):
        r[k] = n
    return r


def spear(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def pq(p):
    return sum(p["pQ"]) / len(p["pQ"])


# ---------------------------------------------------------------- (a) the example context (Mathlib only)
ML = json.load(open(W / "resp_v2_ML.json"))
sc = []
for k, c in enumerate(ML["contexts"]):
    P29 = c["L"]["29"]["tstar"]["pairs"]
    if any(p["Q3"] is None for p in P29):
        continue
    sc.append((spear([pq(p) for p in P29], [p["D2"] for p in P29]), k))
sc.sort()
rho_ex, kx = sc[(len(sc) - 1) // 2]
ctx = ML["contexts"][kx]
EX = json.loads((P / "fig_inputs" / "F8_example.json").read_text())  # its lines, t* and file name (Mathlib)
assert EX["context"] == kx and len(EX["lines"]) == len(ctx["spans"]), (EX["context"], kx)
S = len(ctx["spans"])
pairs = ctx["L"]["29"]["tstar"]["pairs"]
MD, MP = np.full((S, S), np.nan), np.full((S, S), np.nan)
CUT = np.full((S, S), np.nan)  # two pieces of one line cut at 16 tokens: left out of every score, shown grey
for p in pairs:
    if p["j"] == p["i"] + 1 and not ctx["spans"][p["i"]]["nl"]:
        CUT[p["j"], p["i"]] = 1.0
        continue
    MD[p["j"], p["i"]] = p["D2"]
    MP[p["j"], p["i"]] = pq(p)


def short(t, n=46):
    t = " ".join(t.replace("\n", " ").split())
    return t if len(t) <= n else t[: n - 1] + "~"


fig = plt.figure(figsize=(7.2, 5.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1, 1, 1.25], hspace=0.30, wspace=0.32)
vmax = np.nanmax(np.abs(np.concatenate([MD.ravel(), MP.ravel()])))
for col, (M, title) in enumerate([(MD, "exact removal $D_2$"), (MP, r"path-averaged $\bar Q$")]):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.imshow(CUT, cmap=matplotlib.colors.ListedColormap(["#BDBDBD"]), vmin=0, vmax=2)
    ax.set_xticks(range(S))
    ax.set_yticks(range(S))
    ax.set_xticklabels([str(i) for i in range(S)], fontsize=6)
    ax.set_yticklabels([str(i) for i in range(S)], fontsize=6)
    ax.set_xlabel("line $i$")
    ax.set_ylabel("line $j$")
    ax.grid(False)
    ax.set_title(("(a) " if col == 0 else "") + title, loc="left")
    for s_ in ax.spines.values():
        s_.set_visible(False)
cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.025, pad=0.02)
cb.set_label(r"pair effect on the logit of $t^*$", fontsize=7)
cb.ax.tick_params(labelsize=6)
axt = fig.add_subplot(gs[0, 2])
axt.axis("off")
lines = [f"{i:>2}  {short(s_)}" for i, s_ in enumerate(EX["lines"])]
axt.text(0.16, 1.0, "Mathlib lines (next token $t^*$ = '" + EX["tstar"].strip() + "')\n" + "\n".join(lines),
         va="top", ha="left", family="monospace", fontsize=5.2, transform=axt.transAxes)

# ---------------------------------------------------------------- (b) Spearman by block
ax = fig.add_subplot(gs[1, 0:2])
off = {"ML": -0.9, "FW": 0.0, "RH": 0.9}
for m in ("Q", "pQ", "Q3"):
    for c, (lab, mk, ls) in CORP.items():
        ys = [g(m, f"map {c} tstar L{b}", "median Spearman") for b in BLOCKS]
        lo = [g(m, f"map {c} tstar L{b}", "median Spearman", "lo") for b in BLOCKS]
        hi = [g(m, f"map {c} tstar L{b}", "median Spearman", "hi") for b in BLOCKS]
        xs = [b + off[c] for b in BLOCKS]
        ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=MCOL[m], marker=mk, ls=ls,
                    ms=4, capsize=1.5, lw=1.3 if m != "pQ" else 1.9)
ax.set_xticks(BLOCKS)
ax.set_xticklabels(["block 11", "block 20", "block 29"])
ax.set_ylabel("rank correlation with $D_2$\n(median over 16 contexts)")
ax.set_ylim(-0.3, 1.12)
ax.axhline(0, color=GRID, lw=0.8)
ax.grid(True, color=GRID, lw=0.6)
ax.set_title("(b) agreement with exact removal", loc="left")
hm = [plt.Line2D([], [], color=MCOL[m], lw=1.9 if m == "pQ" else 1.3, label=MLAB[m]) for m in ("Q", "pQ", "Q3")]
hc = [plt.Line2D([], [], color=MUTED, marker=mk, ls=ls, label=lab) for c, (lab, mk, ls) in CORP.items()]
l1 = ax.legend(handles=hm, loc="upper left", frameon=False)
ax.add_artist(l1)
ax.legend(handles=hc, loc="lower right", frameon=False)

# ---------------------------------------------------------------- (c) detection AUC by block
ax = fig.add_subplot(gs[1, 2])
offc = {"ML": -0.7, "FW": 0.7}
for m in ("D2", "Q", "pQ"):
    for c in ("ML", "FW"):
        lab, mk, ls = CORP[c]
        st_ = [f"detect {c} vs {c}null tstar L{b}" for b in BLOCKS]
        ys = [g(m, s, "AUC") for s in st_]
        lo = [g(m, s, "AUC", "lo") for s in st_]
        hi = [g(m, s, "AUC", "hi") for s in st_]
        ax.errorbar([b + offc[c] for b in BLOCKS], ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)],
                    color=MCOL[m], marker=mk, ls=ls, ms=3.5, capsize=1.5, lw=1.9 if m == "pQ" else 1.2)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(BLOCKS)
ax.set_xticklabels(["11", "20", "29"])
ax.set_xlabel("block")
ax.set_ylabel("AUC, real vs null contexts")
ax.set_ylim(0.2, 1.02)
ax.grid(True, color=GRID, lw=0.6)
ax.set_title("(c) detection", loc="left")
ax.legend(handles=[plt.Line2D([], [], color=INK, lw=1.2, label=MLAB["D2"])], loc="lower left", frameon=False)

FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "F8_lines.pdf", bbox_inches="tight")
fig.savefig(FIG / "F8_lines.png", bbox_inches="tight")
json.dump({"context": kx, "file": EX["file"], "spearman_pQ_L29": rho_ex, "S": S, "tstar": EX["tstar"]},
          open(FIG / "F8_lines_example.json", "w"), indent=1)
print("F8 example: ML context", kx, EX["file"], f"Spearman {rho_ex:.3f}", "S", S)
