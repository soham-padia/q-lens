#!/usr/bin/env python3
"""Paper figures F0-F4, F6 and F7 (labels spelled out for a reader with no context; F5 is paper/f5_paper.py).
Reads results/claims.csv, results/qj_curves.csv, results/explorable_data/, results/tables/ and results/fig_inputs/
(written from the raw outputs by paper/fig_inputs.py). Writes figures/.
Run: python paper/figs_paper.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = Path(__file__).resolve().parents[1]
P = A / "results"
FIG = A / "figures"
FIG.mkdir(exist_ok=True)
ORANGE, ORANGE_L, BLUE, BLUE_L, NAVY = "#C9551F", "#E8A77F", "#2A6FC9", "#8DB4E6", "#163E73"
INK, MUTED, GRID = "#16202E", "#6B6A66", "#E4E2DC"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "lines.linewidth": 1.6, "savefig.dpi": 200, "pdf.fonttype": 42})
QWEN = ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "Qwen3.8-27B"]
QMARK = {"Qwen3-8B": "o", "Qwen3-14B": "s", "Qwen3-32B": "^", "Qwen3.8-27B": "D"}
QCOL = {"Qwen3-8B": "#2A9D8F", "Qwen3-14B": "#7B5EA7", "Qwen3-32B": NAVY, "Qwen3.8-27B": BLUE}
NL = {"Qwen3-8B": 36, "Qwen3-14B": 40, "Qwen3-32B": 64, "Qwen3.8-27B": 64}
# one colour per method in every figure (Okabe-Ito-like, colour-blind safe): J / AtP orange, J+Q / Q blue,
# calibrated J green, stochastic flag purple, random grey; swap pushes ink, generic directions grey
C_J, C_JQ, C_CAL, C_FLAG, C_RAND = ORANGE, BLUE, "#009E73", "#B4589A", "#8C8C8C"

R = list(csv.DictReader(open(P / "claims.csv")))
for r in R:
    for k in ("value", "lo", "hi"):
        r[k] = float(r[k])


def get(claim, model, quantity, pred=lambda s: True):
    return [r for r in R if r["claim"] == claim and r["model"] == model and r["quantity"] == quantity
            and pred(r["setting"])]


def kv(setting, key):
    for tok in setting.split():
        if tok.startswith(key + "="):
            return float(tok.split("=")[1])


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- F0: the 1-D picture and the setup (schematic)
from matplotlib.patches import FancyBboxPatch
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw={"width_ratios": [1, 1.45]})
x = np.linspace(-0.15, 1.25, 200)
f = lambda t: 0.9 * t + 0.55 * t ** 2 - 0.15 * t ** 3 + 0.35
f1, f2 = 0.9, 1.1          # slope and second derivative at t = 0
a.plot(x, f(x), color=INK, lw=1.8, label="true response")
a.plot(x, f(0) + f1 * x, color=ORANGE, ls="--", lw=1.4, label="J: slope (first order)")
a.plot(x, f(0) + f1 * x + 0.5 * f2 * x ** 2, color=BLUE, lw=1.4, label="J + Q: slope and curvature")
a.scatter([0], [f(0)], color=INK, s=14, zorder=4)
xa = 0.9
a.annotate("", xy=(xa, f(0) + f1 * xa), xytext=(xa, f(xa)), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
a.text(xa + 0.04, f(0) + f1 * xa - 0.02, "missed by a\nlinear reading", fontsize=6.6, color=MUTED, va="top")
a.set_xticks([0, xa], ["state h", "h + push"])
a.set_yticks([])
a.set_xlabel("hidden state, along the push direction")
a.set_ylabel("output")
a.set_title("(a) one dimension", loc="left")
a.legend(frameon=False, fontsize=6.5, loc="upper left")
a.grid(False)
b.set_xlim(0, 10); b.set_ylim(0, 4); b.axis("off")
b.set_title("(b) what is pushed and where it is read", loc="left")
b.annotate("", xy=(9.9, 2), xytext=(0.1, 2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
b.text(9.9, 2.2, "residual stream", fontsize=6.6, color=MUTED, ha="right")
for i, xb in enumerate((2.3, 3.6, 4.9, 6.2)):
    b.add_patch(FancyBboxPatch((xb, 1.55), 0.9, 0.9, boxstyle="round,pad=0.02", fc="white", ec=INK, lw=0.8))
    b.text(xb + 0.45, 2.0, ["l+1", "l+2", "...", "m"][i], ha="center", va="center", fontsize=7)
b.annotate("", xy=(1.5, 2.0), xytext=(1.5, 3.35), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
b.text(1.5, 3.45, "push after block l:\nadd \u03b1U at every position", ha="center", fontsize=6.6, color=ORANGE)
b.annotate("", xy=(7.6, 0.75), xytext=(7.6, 2.0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
b.text(7.6, 0.25, "read at block m: F = sum over\npositions; as logits via W_U(\u03b3\u2299\u00b7)", ha="center", fontsize=6.6, color=BLUE)
b.annotate("", xy=(2.25, 1.2), xytext=(7.15, 1.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
b.text(4.7, 0.85, "k = m \u2212 l blocks traversed", ha="center", fontsize=6.8, color=INK)
save(fig, "F0_schematic")

# ---------------------------------------------------------------- F1: when the straight line holds (C1)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
sel = lambda s: "alpha=0.1 " in s and s.endswith("iso")
for q, col, lab in (("rel_err_J", ORANGE, "J (first order: straight line)"), ("rel_err_JQ", BLUE, "J + Q (second order: with the bend)")):
    rows = sorted(get("C1", "Qwen3.8-27B", q, sel), key=lambda r: kv(r["setting"], "k"))
    k = [kv(r["setting"], "k") for r in rows]
    a.plot(k, [r["value"] for r in rows], color=col, marker="o", ms=4, label=lab)
    a.fill_between(k, [r["lo"] for r in rows], [r["hi"] for r in rows], color=col, alpha=0.15, lw=0)
a.set_yscale("log")
a.set_xlabel("blocks between push and final block")
a.set_ylabel("relative error of the prediction")
a.set_title("Qwen3.8-27B, 10% push", loc="left")
a.legend(frameon=False, loc="upper left", fontsize=7)
for m in QWEN:
    rows = sorted(get("C1", m, "gain_J_over_JQ", sel), key=lambda r: kv(r["setting"], "k"))
    k = [kv(r["setting"], "k") for r in rows]
    if m == "Qwen3.8-27B":   # range over the 8 contexts (per-context gains), drawn behind the bootstrap bars
        cr = sorted(get("C1s", m, "gain ctx range", sel), key=lambda r: kv(r["setting"], "k"))
        b.fill_between([kv(r["setting"], "k") for r in cr], [r["lo"] for r in cr], [r["hi"] for r in cr],
                       color=BLUE_L, alpha=0.35, lw=0, label="Qwen3.8-27B, range over 8 contexts")
    b.errorbar(k, [r["value"] for r in rows], yerr=[[r["value"] - r["lo"] for r in rows],
                                                    [r["hi"] - r["value"] for r in rows]],
               color=QCOL[m], marker=QMARK[m], ms=4, lw=1.2, elinewidth=0.6, capsize=0, alpha=0.95, label=m,
               mfc="white" if m != "Qwen3.8-27B" else QCOL[m])
b.axhline(1, color=MUTED, lw=0.8, ls="--")
b.text(0.5, 0.93, "no gain", color=MUTED, ha="left", va="top", fontsize=7)
b.set_yscale("log")
b.set_xlabel("blocks between push and final block")
b.set_ylabel("gain = error of J / error of J + Q")
b.set_title("Gain from adding the bend, four models", loc="left")
b.set_ylim(0.004, 40)
b.legend(frameon=False, ncol=2, loc="lower left", fontsize=6.0, handlelength=1.2, columnspacing=0.7)
fig.subplots_adjust(wspace=0.32)
save(fig, "F1_taylor_error")

# ---------------------------------------------------------------- F2: bend grows with blocks traversed (C2, C3)
cur = defaultdict(list)
for r in csv.DictReader(open(P / "qj_curves.csv")):
    if r["type"] == "iso":
        cur[r["model"]].append((int(r["k"]), float(r["qj_median"])))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.75, 1]})
for m in QWEN:
    pts = np.array(cur[m])
    kk = np.linspace(1, pts[:, 0].max(), 100)   # fitted line only over the distances this model has
    ax.scatter(pts[:, 0], pts[:, 1], s=10, marker=QMARK[m], facecolor="white" if m != "Qwen3.8-27B" else QCOL[m],
               edgecolor=QCOL[m], lw=0.7, alpha=0.8, label=m, zorder=3)
    b_ = get("C2", m, "dlog(Q/J)/dk", lambda s: s.startswith("iso all"))[0]["value"]
    a0 = get("C2", m, "intercept_log(Q/J)_and_rms_resid", lambda s: s.startswith("iso"))[0]["value"]
    ax.plot(kk, np.exp(a0 + b_ * kk), color=QCOL[m], lw=0.9, alpha=0.8)
LOOPC = {"Ouro-1.4B": ORANGE_L, "Ouro-2.6B": ORANGE, "Huginn-3.5B": "#8A3A12"}
LOOPM = {"Ouro-1.4B": "v", "Ouro-2.6B": "P", "Huginn-3.5B": "X"}
for m, col in LOOPC.items():
    byk = defaultdict(list)
    for k, v in cur[m]:
        byk[k].append(v)
    ks = sorted(byk)
    ax.plot(ks, [np.median(byk[k]) for k in ks], color=col, marker=LOOPM[m], ms=3.5, lw=1.2, label=f"{m} (looped)")
ax.set_yscale("log")
ax.set_xlim(0, 182)
ax.set_xlabel("blocks traversed by the push, k")
ax.set_ylabel("bend / straight line,  ‖Q‖ / ‖J‖")
slopes = [get("C2", m, "dlog(Q/J)/dk", lambda s: s.startswith("iso all"))[0]["value"] for m in QWEN]
ax.set_title("(a) r = ‖Q‖/‖J‖ grows with the blocks traversed", loc="left", fontsize=8.3)
ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=6.2, handlelength=1.2, columnspacing=0.8)
# (b) numerator and denominator of r separately (D1_raw.pt of the runs of results/jq_split_depth.jsonl): pushed at each
# model's earliest push block, the median over cases at each read-out block, divided by its value one block after the
# push. One thin line per model. The medians are extracted from D1_raw.pt by paper/fig_inputs.py.
F2B = json.loads((P / "fig_inputs" / "F2b_norms.json").read_text())
for d in ("so_qwen3-8b", "so_qwen3-14b", "so_qwen3-32b", "so_llama3_1-8b"):
    for key, col in (("norm_J", C_J), ("norm_Q", C_JQ)):
        med = np.array(F2B[d][key])
        ax2.plot(F2B[d]["k"], med / med[0], color=col, lw=0.9, alpha=0.85, marker="o", ms=2)
ax2.set_yscale("log")
ax2.axhline(1, color=MUTED, lw=0.6, ls=":")
ax2.text(0.96, 0.9, "‖Q(U,U)‖, the bend", color=C_JQ, transform=ax2.transAxes, ha="right", fontsize=7)
ax2.text(0.96, 0.08, "‖JU‖, the straight line", color=C_J, transform=ax2.transAxes, ha="right", fontsize=7)
ax2.set_xlabel("blocks traversed")
ax2.set_ylabel("growth since one block after the push")
ax2.set_title("(b) ‖Q‖ grows; ‖JU‖ dips, then recovers", loc="left", fontsize=8.3)
fig.subplots_adjust(wspace=0.3)
save(fig, "F2_bend_vs_depth")

# ---------------------------------------------------------------- F3: bends that cancel (C4)
E4 = json.loads((P / "explorable_data" / "E4_block_lengths.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
e = E4["Qwen3.8-27B"]["8/iso"]
# median per-block profile, rescaled so the chain's total is the median cancellation (sum of medians != median sum)
L = np.array(e["block_norm_over_Q_median"])
L = L * e["ci_median"] / L.sum()
x0 = 0
for i, l in enumerate(L):
    a.barh(1, l, left=x0, height=0.5, color=ORANGE if i % 2 == 0 else ORANGE_L, lw=0)
    x0 += l
a.barh(0, 1.0, height=0.5, color=BLUE, lw=0)
a.set_yticks([0, 1], ["their sum, Q", "55 block terms\nof Q, end to end"])
a.grid(axis="y", visible=False)
a.text(x0 + 0.08, 1, f"{x0:.1f}×", va="center", color=INK, fontsize=8)
a.text(1.08, 0, "1×", va="center", color=INK, fontsize=8)
a.set_xlim(0, x0 + 0.7)
a.set_xlabel("length, in units of ‖Q‖")
a.set_title("Qwen3.8-27B, push at block 8, random directions", loc="left", fontsize=8.3)
# (b) per case, sqrt(sum |c_b|^2) / |Q| = C / sqrt(N_eff): 1 when the block terms are exactly orthogonal
for m, mk, dx in (("Qwen3.8-27B", "D", -0.6), ("Qwen3-8B", "o", 0.6)):
    for tn, fc, dd in (("iso", QCOL[m], -0.3), ("data", "white", 0.3)):
        rows = sorted(get("C4", m, "block_ci_over_sqrt_pr case IQR", lambda s: s.endswith(tn)),
                      key=lambda r: kv(r["setting"], "blocks"))
        k = np.array([kv(r["setting"], "blocks") for r in rows]) + dx + dd
        b.errorbar(k, [r["value"] for r in rows], yerr=[[r["value"] - r["lo"] for r in rows],
                                                        [r["hi"] - r["value"] for r in rows]],
                   fmt=mk, ms=4, mfc=fc, mec=QCOL[m], ecolor=QCOL[m], elinewidth=0.7, capsize=0, lw=0,
                   label=f"{m}, {'random' if tn == 'iso' else 'data'}")
b.axhline(1, color=MUTED, lw=0.8, ls="--")
b.text(57, 1.42, "dashed line at 1: the block terms\nadd in quadrature (orthogonal)", color=MUTED, ha="right", va="top", fontsize=6.4)
b.set_ylim(0.6, 1.45)
b.set_xlim(0, 58)
b.set_xlabel("blocks between push and final block")
b.set_ylabel("√(Σ‖block term‖²) / ‖Q‖, per case")
b.set_title("Per case: near 1 at every push block", loc="left", fontsize=8.3)
b.legend(frameon=False, loc="lower left", fontsize=5.9, ncol=2, handlelength=1.0, borderaxespad=0.2, columnspacing=0.8)
fig.subplots_adjust(wspace=0.45)
save(fig, "F3_cancellation")

# ---------------------------------------------------------------- F4: no piece is the answer (C4)
E5 = json.loads((P / "explorable_data" / "E5_reconstruction.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6))
LV = [("block", "largest block", ORANGE, "o"), ("site", "largest site (a norm, mixer or MLP)", ORANGE_L, "s"),
      ("heads", "largest attention head", NAVY, "^"), ("neurons", "largest MLP neuron", BLUE, "v")]
for lvl, lab, col, mk in LV:
    rows = sorted(get("C4", "Qwen3.8-27B", f"top_{lvl}_along_Q", lambda s: s.endswith("iso")),
                  key=lambda r: kv(r["setting"], "blocks"))
    k = [kv(r["setting"], "blocks") for r in rows]
    a.errorbar(k, [100 * r["value"] for r in rows], yerr=[[100 * (r["value"] - r["lo"]) for r in rows],
                                                         [100 * (r["hi"] - r["value"]) for r in rows]],
               color=col, marker=mk, ms=4, lw=1.2, elinewidth=0.6, label=lab)
a.set_yscale("log")
a.set_xlabel("blocks between push and final block")
a.set_ylabel("largest piece's share of Q (%)")
a.set_title("Qwen3.8-27B, random directions", loc="left")
a.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
cur5 = E5["Qwen3.8-27B"]["8/iso"]
for lvl, lab, col, mk in [("scal_block", "blocks (55)", ORANGE, "o"), ("scal_site", "sites (220)", ORANGE_L, "s"),
                          ("scal_heads", "heads (2,304)", NAVY, "^"), ("scal_neurons", "neurons (957,440)", BLUE, "v")]:
    c = cur5[lvl]
    N = c["N"]
    pts = sorted([(n / N, e_) for n, e_ in zip(c["n"], c["err"])] + [(c["n50"] / N, 0.5), (c["n90"] / N, 0.1)])
    b.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=3.5, lw=1.2, label=lab)
b.axhline(0.1, color=MUTED, lw=0.8, ls="--")
b.text(1.2e-6, 0.115, "within 10% of Q", color=MUTED, fontsize=7)
b.set_xscale("log")
b.set_xlabel("fraction of pieces kept, largest first")
b.set_ylabel("error of the partial sum\nalong Q's direction")
b.set_title("Push at block 8: no small set of pieces rebuilds Q", loc="left")
b.legend(frameon=False, fontsize=6.5, loc="center left")
fig.subplots_adjust(wspace=0.42)
save(fig, "F4_no_single_piece")

print("wrote", sorted(p.name for p in FIG.glob("F*.pdf")))

# ---------------------------------------------------------------- F6: where the bend matters (dose, attribution patching)
import re

NOTES = P / "tables"
fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.2))
# (a) dose miss by swap layer: raw J, raw J+Q, J rescaled per layer (5-fold CV), from results/tables/layer_pick_tables_dense.md
LP = (NOTES / "layer_pick_tables_dense.md").read_text()
LP = LP[LP.index("Fitted factors"):]
dl, mJ, mQ, mC = [], [], [], []
for line in LP.splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 7 and re.fullmatch(r"L\d+", c[0]):
        jr, jd = [float(x) for x in re.findall(r"[\d.]+", c[4])]
        qr = float(re.findall(r"[\d.]+", c[5])[0])
        dl.append(int(c[0][1:])); mJ.append(jr); mQ.append(qr); mC.append(jd)
a = axs[0, 0]
for y, col, lab, yo in ((mJ, C_J, "J", 0), (mQ, C_JQ, "J + Q", 0), (mC, C_CAL, "J, rescaled per layer\non swaps already run", 0)):
    a.plot(dl, y, color=col, marker="o", ms=3, lw=1.1)
a.text(dl[1], mJ[1] * 1.12, "J", color=C_J, fontsize=7, va="bottom")
a.text(dl[1], mQ[1] * 1.12, "J + Q", color=C_JQ, fontsize=7, va="bottom")
a.text(dl[0], 0.93, "J rescaled per layer (on swaps already run)", color=C_CAL, fontsize=6.5, ha="left", va="top")
a.axhline(1, color=MUTED, lw=0.6, ls=":")
a.set_yscale("log")
a.set_ylim(0.66, 14)
a.set_yticks([1, 2, 4, 8], ["×1", "×2", "×4", "×8"])
a.set_xlabel("swap layer (64 blocks; later = fewer blocks left)")
a.set_ylabel("median miss of the flip dose")
a.set_title("(a) the bend sets a swap's dose", loc="left", fontsize=8.3)
# (b)-(d) attribution patching on Qwen3-8B, from the per-pair outputs of src/atp_q.py (as src/flag_then_patch.py).
# paper/fig_inputs.py reads those outputs (atp_load there) into results/fig_inputs/F6_atp_q.npz.
K_ = 32
EPS = 1e-12
F6IN = np.load(P / "fig_inputs" / "F6_atp_q.npz")


def atp_load(task):
    n = len({k.split(".")[1] for k in F6IN.files if k.startswith(f"{task}.pair")})
    out = [{v: F6IN[f"{task}.pair{i:03d}.{v}"] for v in ("a", "atp", "J", "Q", "qs")} for i in range(n)]
    spl = {kd: [F6IN[k] for k in sorted(F6IN.files) if k.startswith(f"{task}.split{kd}.")] for kd in (0, 1)}
    return out, spl


def patch_curve(Pp, score_fn, over, xs, rand=False):
    F, E = [], []
    for x in xs:
        err = mass = 0.0
        cost = []
        for i, p in enumerate(Pp):
            a_, n = p["a"], len(p["a"])
            kk = min(int(round(x / 100 * n)), n)
            if rand:
                es = []
                rng = np.random.default_rng(1000 * i)
                for _ in range(20):
                    est = p["atp"].copy(); idx = rng.permutation(n)[:kk]; est[idx] = a_[idx]
                    es.append(np.abs(est - a_).sum())
                err += float(np.mean(es))
            else:
                est = p["atp"].copy(); idx = np.argsort(-score_fn(p), kind="stable")[:kk]; est[idx] = a_[idx]
                err += np.abs(est - a_).sum()
            mass += np.abs(a_).sum()
            cost.append(over + kk)
        F.append(np.median(cost)); E.append(err / mass)
    return F, E


XS = [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100]
TN = {"ioi": "indirect object", "fact": "factual recall"}
b = axs[0, 1]
LS = {("ioi", 0): "-", ("fact", 0): "-", ("ioi", 1): "--", ("fact", 1): "--"}
NU = 1
for ti, (task, ax_) in enumerate((("ioi", axs[1, 0]), ("fact", axs[1, 1]))):
    Pp, spl = atp_load(task)
    for kd, nm in ((0, "heads"), (1, "MLPs")):
        A_ = spl[kd]
        n_ = max(len(s) for s in A_)
        M = np.array([np.concatenate([s, np.ones(n_ - len(s))]) for s in A_])
        y = 100 * np.median(M, 0)
        col = C_JQ if ti == 0 else NAVY
        b.plot(np.arange(1, n_ + 1), y, color=col, lw=1.1, ls=LS[(task, kd)], marker="o", ms=2, label=f"{TN[task]}, {nm}")
        NU = max(NU, int(np.argmax(y >= 99.5)) + 1)
    n = int(np.median([len(p["a"]) for p in Pp]))
    fa, ea = patch_curve(Pp, lambda p: np.abs(p["atp"]), 4, XS)
    fw, ew = patch_curve(Pp, lambda p: np.abs(p["qs"] / 2) / (np.abs(p["atp"]) + EPS), 4 + 6 * K_, XS)
    fr, er = patch_curve(Pp, None, 4, XS, rand=True)
    jq = sum(np.abs(p["J"] + p["Q"] / 2 - p["a"]).sum() for p in Pp) / sum(np.abs(p["a"]).sum() for p in Pp)
    ax_.plot(fr, er, color=C_RAND, lw=1.0, ls="--")
    ax_.plot(fw, ew, color=C_FLAG, lw=1.2, marker="o", ms=2.2)
    ax_.plot(fa, ea, color=C_J, lw=1.2, marker="o", ms=2.2)
    ax_.plot([2 + 4 * n], [jq], marker="s", ms=5, color=C_JQ, ls="none")
    ax_.plot([1 + n], [0], marker="*", ms=8, color=INK, ls="none", clip_on=False)
    ax_.set_xscale("log")
    ax_.set_xlim(3, 1.6e4)
    ax_.set_ylim(0, 1.08 * max(max(er), max(ew), max(ea), jq))
    ax_.set_xlabel("cost (forward passes per prompt pair)")
    ax_.set_title(f"({'c' if ti == 0 else 'd'}) {TN[task]}: error against cost", loc="left", fontsize=8.3)
    ym = ax_.get_ylim()[1]
    ax_.text(4.5, 0.12 * ym, "AtP, then patch\nits largest", color=C_J, fontsize=6.3, va="bottom")
    ax_.text(fw[5], ew[5] + 0.04 * ym, "stochastic flag,\nthen patch flagged", color=C_FLAG, fontsize=6.3, va="bottom")
    ax_.text(fr[8] / 1.3, er[8], "patch random", color=C_RAND, fontsize=6.3, va="center", ha="right")
    ax_.text(2 + 4 * n, jq + 0.05 * ym, "exact J + Q", color=C_JQ, fontsize=6.3, ha="center", va="bottom")
    ax_.text(1 + n, 0.04 * ym, "patch all", color=INK, fontsize=6.3, ha="center", va="bottom")
axs[1, 0].set_ylabel("relative error against patching")
b.axhline(90, color=MUTED, lw=0.6, ls=":")
b.set_xlabel("nonlinear units from the patch, in order")
b.set_ylabel("cumulative share of ½Q (%)")
b.set_ylim(0, 104)
b.set_xlim(0.7, NU + 0.5)
b.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.8)
b.set_title("(b) AtP's second-order error lies downstream", loc="left", fontsize=8.3)
fig.subplots_adjust(wspace=0.32, hspace=0.62)
save(fig, "F6_matters")

# ---------------------------------------------------------------- F7: swap pushes depart from the generic baseline
SWT = (NOTES / "swap_split_tables.md").read_text()


def swt(layer):
    sec = SWT[SWT.index(f"## L{layer} (K"):]
    sec = sec[:sec.index("\n## ", 5)]
    d = {}
    for line in sec.splitlines():
        if line.startswith("| ") and not line.startswith("| quantity"):
            c = [x.strip() for x in line.strip("|").split("|")]
            d[c[0]] = [[float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", x)] for x in c[1:]]
    return d


SWL = [36, 40, 48]
SWK = {36: 27, 40: 23, 48: 15}
S = {L: swt(L) for L in SWL}
QN = [("(Q/J) per unit push norm", 1, "(a) curvature per unit push", "‖Q‖ / ‖J‖ per unit push norm"),
      ("block ci/√pr (vector)", 1, "(b) block terms reinforce", "√(Σ‖block term‖²) / ‖Q‖"),
      ("top neuron along q̂", 100, "(c) single neurons carry more", "largest neuron's share of Q (%)")]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35))
x = np.arange(len(SWL))
for ax_, (q, sc, ttl, yl) in zip(axs, QN):
    gen = np.array([[S[L][q][c][0] * sc for c in (2, 3, 4)] for L in SWL])  # random at swap norms, random, data
    ax_.fill_between(x, gen.min(1), gen.max(1), color=C_RAND, alpha=0.3, lw=0, step=None)
    for col_i, mk, fc, dx in ((0, "o", INK, -0.06), (1, "o", "white", 0.06)):
        v = np.array([S[L][q][col_i] for L in SWL]) * sc
        ax_.errorbar(x + dx, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=3.8, mfc=fc, mec=INK,
                     ecolor=INK, elinewidth=0.7, capsize=0, lw=0)
    ax_.set_xticks(x, [f"L{L}\n({SWK[L]} left)" for L in SWL], fontsize=6.6)
    ax_.set_xlim(-0.4, len(SWL) - 0.6)
    ax_.set_title(ttl, loc="left", fontsize=8.3)
    ax_.set_ylabel(yl, fontsize=7.3)
    ax_.grid(axis="x", visible=False)
a, b, c = axs
a.set_ylim(0, 0.075)
a.text(1, S[40][QN[0][0]][0][0] + 0.009, "● concept swap", color=INK, fontsize=6.4, ha="center")
a.text(1, S[40][QN[0][0]][1][0] + 0.008, "○ swap of an\nunrelated pair", color=INK, fontsize=6.0, ha="center")
a.text(1, 0.022, "random and data", color="#5E5E5E", fontsize=6.4, ha="center")
b.axhline(1, color=MUTED, lw=0.7, ls="--")
b.set_ylim(0.45, 1.25)
b.text(2.35, 1.03, "add in quadrature", color=MUTED, fontsize=6.2, ha="right", va="bottom")
b.text(2.35, 0.47, "below 1: reinforce", color=MUTED, fontsize=6.2, ha="right", va="bottom")
b.text(2.35, 1.23, "above 1: cancel", color=MUTED, fontsize=6.2, ha="right", va="top")
c.set_ylim(0, 17)
neur = 957440 // 55  # MLP neurons per block of Qwen3.8-27B
c.text(-0.3, 0.6, f"even split: under {100 / (min(SWK.values()) * neur):.4f}%", color=MUTED, fontsize=6.0, va="bottom")
fig.subplots_adjust(wspace=0.55)
save(fig, "F7_swap_departure")
print("wrote F6_matters, F7_swap_departure")
