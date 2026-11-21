"""Flag-then-patch: is "AtP + stochastic |Q/2|/|J| flag, then actually patch only the flagged sites" a cheaper
route to accurate attribution than patching everything? CPU only, from the per-pair outputs of src/atp_q.py.

    python src/flag_then_patch.py [OUT_DIR] [--dry]   (--dry: print the section, write nothing but the figure)

Writes the "## Flag-then-patch" section of results/tables/atp_q_results.md (replaced in place between markers; rerun after
src/summarize_atp_q.py, which rewrites that file) and results/tables/figs_atp_q/flag_then_patch.png.

Sites: comp = every attention head (o_proj input slice) and MLP output at the last position with delta != 0,
n ~ 1170 per pair; ground truth A = actual activation patching; effects normalised per pair by D.
Workflow W(x): AtP (4 F) + stochastic K=32 diagonal (196 F) -> patch the top x% of sites by |q/2|/|J| and use the
patched value there; AtP elsewhere (W) or the stochastic J+Q estimate elsewhere (W-JQ). Cost 200 + x n.
"""
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY = "--dry" in sys.argv
OUT = ARGS[0] if ARGS else os.path.join(os.environ.get("QLENS_OUT", "outputs"), "atp_q_qwen3-8b")
FIG = os.path.join(ROOT, "results", "tables", "figs_atp_q")
DST = os.path.join(ROOT, "results", "tables", "atp_q_results.md")
K, IG_M = 32, 16
C_ATP, C_STOCH, C_IG = 4, 4 + 6 * K, 2 + 3 * IG_M  # nominal forward-pass equivalents (summarize_atp_q.py)
XS_TAB = [0, 2, 5, 10, 20, 30, 50]
XS_FIG = [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100]
N_RAND, D_MIN, EPS = 20, 0.5, 1e-12
TASKS = ("ioi", "fact")
START, END = "<!-- flag-then-patch:start -->", "<!-- flag-then-patch:end -->"


def load(t):
    P = []
    for f in sorted(glob.glob(f"{OUT}/{t}_*.pt")):
        r = torch.load(f, weights_only=False)
        D = float(r["m_clean"] - r["m_corrupt"])
        if abs(D) <= D_MIN:
            continue
        kind, null, fam = np.asarray(r["kind"]), np.asarray(r["null"], bool), np.asarray(r["fam"])
        q = np.full(len(kind), np.nan)
        q[np.nonzero(fam == 0)[0]] = np.asarray(r["stoch"][0])[:K].mean(0)
        sel = np.isin(kind, (0, 1)) & ~null
        g = lambda v: np.asarray(v, np.float64)[sel] / D  # noqa: E731
        p = dict(a=g(r["actual"]), atp=g(r["atp"]), J=g(r["J"]), Q=g(r["Q"]), qs=g(q), ig=g(r["ig"][IG_M]))
        assert all(np.isfinite(v).all() for v in p.values()), f
        t, c = r["times"], r["counts"]
        p["sec"] = (t["stoch0"] * K / c["K"], t["patch"] * c["fam_live"][0] / max(c["live"], 1), int(sel.sum()))
        P.append(p)
    return P


# method -> (label, ranking score, estimate kept elsewhere, overhead F, extra sites patched at equal x)
def spec(p, n):
    flag_s = np.abs(p["qs"] / 2) / (np.abs(p["atp"]) + EPS)
    flag_x = np.abs(p["Q"] / 2) / (np.abs(p["J"]) + EPS)
    jq_s, jq_x = p["atp"] + p["qs"] / 2, p["J"] + p["Q"] / 2
    c_exact = 2 + 4 * n
    return {
        "W": ("**W**: stoch flag, AtP elsewhere", flag_s, p["atp"], C_ATP + C_STOCH, 0),
        "W-JQ": ("**W-JQ**: stoch flag, stoch J+Q elsewhere", flag_s, jq_s, C_ATP + C_STOCH, 0),
        "a": ("(a) top \\|AtP\\|, same x", np.abs(p["atp"]), p["atp"], C_ATP, 0),
        "a=": ("(a=) top \\|AtP\\|, W's cost (x n + 196 sites)", np.abs(p["atp"]), p["atp"], C_ATP, C_STOCH),
        "f": ("(f) top \\|J+½q̂\\|, stoch J+Q elsewhere (W-JQ's cost)", np.abs(jq_s), jq_s, C_ATP + C_STOCH, 0),
        "b": ("(b) random x, AtP elsewhere", None, p["atp"], C_ATP, 0),
        "c": ("(c) exact flag, AtP elsewhere", flag_x, p["atp"], c_exact, 0),
        "c-JQ": ("(c-JQ) exact flag, exact J+Q elsewhere", flag_x, jq_x, c_exact, 0),
        "d": ("(d) EAP-IG16 + top \\|IG\\| at W's cost, IG elsewhere", np.abs(p["ig"]), p["ig"], C_IG,
              C_ATP + C_STOCH - C_IG),
    }


METHODS = ["W", "W-JQ", "a", "a=", "f", "b", "c", "c-JQ", "d"]


def topk_recall(a, y, k):
    return len(set(np.argsort(-np.abs(a))[:k]) & set(np.argsort(-np.abs(y))[:k])) / k


def pair_metrics(a, est):
    return dict(err=np.abs(est - a).sum(), mass=np.abs(a).sum(), t10=topk_recall(a, est, 10),
                t50=topk_recall(a, est, 50), sp=spearmanr(a, est)[0])


def evaluate(P, meth, x, seed=0):
    """Per-pair metrics and nominal cost of one method at x% (random: mean over N_RAND draws)."""
    out = []
    for i, p in enumerate(P):
        a, n = p["a"], len(p["a"])
        _, score, base, over, extra = spec(p, n)[meth]
        kk = min(int(round(x / 100 * n)) + extra, n)
        if score is None:
            rng = np.random.default_rng(seed + 1000 * i)
            ms = []
            for _ in range(N_RAND):
                est = base.copy(); idx = rng.permutation(n)[:kk]; est[idx] = a[idx]
                ms.append(pair_metrics(a, est))
            m = {k: float(np.mean([d[k] for d in ms])) for k in ms[0]}
        else:
            est = base.copy(); idx = np.argsort(-score, kind="stable")[:kk]; est[idx] = a[idx]
            m = pair_metrics(a, est)
        m["F"] = over + kk
        out.append(m)
    return out


def agg(ms):
    return dict(rel=sum(m["err"] for m in ms) / sum(m["mass"] for m in ms), t10=np.mean([m["t10"] for m in ms]),
                t50=np.mean([m["t50"] for m in ms]), sp=np.median([m["sp"] for m in ms]),
                F=np.median([m["F"] for m in ms]), per=np.array([m["err"] / m["mass"] for m in ms]),
                sp_per=np.array([m["sp"] for m in ms]))


R = {t: load(t) for t in TASKS}
NS = {t: int(np.median([len(p["a"]) for p in R[t]])) for t in TASKS}
RES = {(t, m, x): agg(evaluate(R[t], m, x)) for t in TASKS for m in METHODS for x in sorted(set(XS_TAB + XS_FIG))}
LABEL = {m: spec(R["ioi"][0], NS["ioi"])[m][0] for m in METHODS}


def f3(v):
    return f"{v:.3f}"


def rel(t, m, x):
    return f3(RES[(t, m, x)]["rel"])


def share(t, m, ref, x, key="per", lower=True):
    """Share of pairs where method m is better than ref (lower rel L1, or higher Spearman)."""
    u, v = RES[(t, m, x)][key], RES[(t, ref, x)][key]
    return float(np.mean(u < v if lower else u > v))


def first_x(t, m, ref, key):
    """Smallest tabled x > 0 where m beats ref on the pooled metric (rel: lower; sp: higher)."""
    for x in XS_TAB[1:]:
        u, v = RES[(t, m, x)][key], RES[(t, ref, x)][key]
        if (u < v) if key == "rel" else (u > v):
            return x
    return None


for t in TASKS:  # console summary
    print(f"== {t}: {len(R[t])} pairs, n = {NS[t]}")
    for m in METHODS:
        print(f"{m:5s}", " ".join(f"x{x:>2}:{RES[(t, m, x)]['rel']:.3f}/{RES[(t, m, x)]['sp']:.3f}" for x in XS_TAB))

SEC = {t: np.median(np.array([p["sec"] for p in R[t]]), 0) for t in TASKS}
SEC_ALL = np.median(np.array([p["sec"] for t in TASKS for p in R[t]]), 0)
max_w_rel = max(share(t, "W", "a=", x) for t in TASKS for x in XS_TAB[1:])
max_wa_rel = max(share(t, "W", "a", x) for t in TASKS for x in XS_TAB[1:])
max_wjq_f = max(share(t, "W-JQ", "f", x) for t in TASKS for x in XS_TAB[1:])
sp_x = {t: first_x(t, "W", "a=", "sp") for t in TASKS}
rel_x = {t: first_x(t, "W", "a", "rel") for t in TASKS}
x0 = 10
lines = [START, "## Flag-then-patch", ""]
verdict = (
    "**Verdict.** "
    + ("No. " if all(v is None for v in rel_x.values()) else "Partly. ")
    + "AtP + the stochastic K=32 flag + patching the flagged x% (W) does not beat patching the top x% by |AtP| "
    "(a) on error or on top-k recall, at any x in either task. At W's own cost ((a=): AtP + the top x n + "
    f"{C_STOCH} sites by |AtP|), the |AtP| route has the lower rel L1 at every x (W is lower in at most "
    f"{round(max_w_rel * len(R['ioi']))} of {len(R['ioi'])} pairs at any x): at x = {x0}% ({RES[('ioi', 'W', x0)]['F']:.0f} F, "
    f"{RES[('ioi', 'W', x0)]['F'] / (1 + NS['ioi']):.0%} of patching everything) rel L1 is "
    f"{rel('ioi', 'W', x0)} (W) vs {rel('ioi', 'a=', x0)} (a=) on IOI and {rel('fact', 'W', x0)} vs "
    f"{rel('fact', 'a=', x0)} on fact, top-10 recall {RES[('ioi', 'W', x0)]['t10']:.2f} / "
    f"{RES[('fact', 'W', x0)]['t10']:.2f} vs {RES[('ioi', 'a=', x0)]['t10']:.2f} / "
    f"{RES[('fact', 'a=', x0)]['t10']:.2f}. W loses even to (a) at the same x without the extra "
    f"{C_STOCH} F ({rel('ioi', 'a', x0)} / {rel('fact', 'a', x0)}) and is only a little better than patching a "
    f"random x% ({rel('ioi', 'b', x0)} / {rel('fact', 'b', x0)}). The reason: the flag ranks sites by AtP's "
    "*relative* error, which is largest where effects are small, while the absolute error sits at the "
    "large-|AtP| sites. The one metric W wins is the whole-list per-pair Spearman over all ~1170 sites, which "
    "small sites dominate: W beats (a=) from x = "
    f"{sp_x['ioi']}% (IOI) / {sp_x['fact']}% (fact), e.g. {RES[('ioi', 'W', x0)]['sp']:.3f} vs "
    f"{RES[('ioi', 'a=', x0)]['sp']:.3f} / {RES[('fact', 'W', x0)]['sp']:.3f} vs "
    f"{RES[('fact', 'a=', x0)]['sp']:.3f} at x = {x0}%. "
    "Keeping the stochastic J+Q elsewhere (W-JQ) does not change the verdict. With q̂ in hand, ranking by "
    f"|J + ½q̂| (f) beats ranking by the flag (W-JQ lower in at most {round(max_wjq_f * len(R['ioi']))} of "
    f"{len(R['ioi'])} pairs at any x > 0), but still "
    f"loses to (a=) ({rel('ioi', 'f', x0)} / {rel('fact', 'f', x0)} at x = {x0}%). The exact flag (c), at "
    "≥ 4n cost, is worse than |AtP| ranking at the same x up to 20% and level at 30%. EAP-IG16 at matched "
    f"cost (d) is also worse than (a=) ({rel('ioi', 'd', x0)} / {rel('fact', 'd', x0)}; ρ "
    f"{RES[('ioi', 'd', x0)]['sp']:.2f} / {RES[('fact', 'd', x0)]['sp']:.2f}). Practical reading: at ~1170 "
    "sites the cheap accurate route is AtP, then patch the top few % by |AtP|. The top 5% "
    f"({RES[('ioi', 'a', 5)]['F']:.0f} F, {RES[('ioi', 'a', 5)]['F'] / (1 + NS['ioi']):.0%} of patching "
    f"everything) gives rel L1 {rel('ioi', 'a', 5)} / {rel('fact', 'a', 5)} and top-10 recall "
    f"{RES[('ioi', 'a', 5)]['t10']:.2f} / {RES[('fact', 'a', 5)]['t10']:.2f}, against exact J+Q at every "
    f"site at {rel('ioi', 'c-JQ', 0)} / {rel('fact', 'c-JQ', 0)} for {RES[('ioi', 'c-JQ', 0)]['F']:.0f} F. "
    "The flag is a good diagnostic (AUC tables above) but not a patching priority."
)
lines += ["VERDICT_PLACEHOLDER", "",
          f"Script `src/flag_then_patch.py` (CPU, reads `{OUT}`). Sites: heads + MLPs at the last position with "
          f"δ ≠ 0, n = {NS['ioi']} (IOI) / {NS['fact']} (fact) per pair (median); {len(R['ioi'])} + "
          f"{len(R['fact'])} pairs. Patched sites take their actual patching value; all other sites keep the "
          "named estimate. Metrics against actual patching, effects normalised by D: rel L1 = Σ|est − A| / Σ|A| "
          "pooled over pairs; top-k = mean over pairs of |top-k by |est| ∩ top-k by |A|| / k; ρ = median "
          "per-pair Spearman. Flag = |½q̂|/|J| with q̂ the mean of the first 32 Rademacher draws (stoch) or "
          f"the exact Q (exact). Random (b) is the mean over {N_RAND} draws per pair.", "",
          "Nominal cost (forward-pass equivalents, median over pairs; x n sites patched at 1 F each): "
          f"W, W-JQ, (a=), (f), (d) = 200 + x n (AtP 4 + stochastic K=32 196, as specified; (a=) spends the 196 on "
          f"extra |AtP|-ranked patches, (d) = EAP-IG16 {C_IG} + top-|IG| patches to the same total); (a), (b) = 4 + x n; "
          "(c), (c-JQ) = 2 + 4n + x n (exact J+Q for every site, more than patching everything); (e) patch "
          f"everything = 1 + n ≈ {1 + NS['ioi']}. Measured seconds (median per pair, both tasks): the K=32 "
          f"stochastic pass {SEC_ALL[0]:.2f} s against {SEC_ALL[1]:.2f} s for patching all ~{SEC_ALL[2]:.0f} sites "
          f"in batches of 127, i.e. ≈{SEC_ALL[0] / SEC_ALL[1] * SEC_ALL[2]:.0f} patched sites (nominal: "
          f"{C_STOCH}).", ""]
hdr = "| method | " + " | ".join(f"x = {x}%" for x in XS_TAB) + " |"
sep = "|---|" + "---|" * len(XS_TAB)
lines += ["**Nominal cost (F, IOI; fact within ±1%)**", "", hdr, sep]
for m in METHODS:
    lines.append(f"| {LABEL[m]} | " + " | ".join(f"{RES[('ioi', m, x)]['F']:.0f}" for x in XS_TAB) + " |")
lines.append(f"| (e) patch everything | {1 + NS['ioi']} (all x) |" + " |" * (len(XS_TAB) - 1))
lines.append("")
for t in TASKS:
    lines += [f"**{t}: rel L1 against actual patching** (lower is better; (e) = 0)", "", hdr, sep]
    for m in METHODS:
        row = [RES[(t, m, x)]["rel"] for x in XS_TAB]
        lines.append(f"| {LABEL[m]} | " + " | ".join(f3(v) for v in row) + " |")
    lines += ["", f"**{t}: top-10 / top-50 recall / per-pair Spearman ρ** ((e) = 1 / 1 / 1)", "", hdr, sep]
    for m in METHODS:
        lines.append(f"| {LABEL[m]} | " + " | ".join(
            f"{RES[(t, m, x)]['t10']:.2f} / {RES[(t, m, x)]['t50']:.2f} / {RES[(t, m, x)]['sp']:.3f}"
            for x in XS_TAB) + " |")
    lines += ["", f"**{t}: head to head, share of pairs where the first method is better**", "",
              "| x | W vs (a=), rel L1 | W vs (a=), ρ | W vs (a) same x, rel L1 | W-JQ vs (f), rel L1 | "
              "W vs (b), rel L1 |", "|---|---|---|---|---|---|"]
    for x in XS_TAB[1:]:
        lines.append(f"| {x}% | {share(t, 'W', 'a=', x):.2f} | {share(t, 'W', 'a=', x, 'sp_per', False):.2f} | "
                     f"{share(t, 'W', 'a', x):.2f} | {share(t, 'W-JQ', 'f', x):.2f} | {share(t, 'W', 'b', x):.2f} |")
    lines.append("")
lines += ["![](figs_atp_q/flag_then_patch.png)", "", END]
SECTION = "\n".join(lines)

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"W": "#2a78d6", "W-JQ": "#1baf7a", "a": "#eb6834", "b": "#8c8c8c", "c": "#6250d6", "c-JQ": "#e87ba4",
       "d": "#e34948", "f": "#008300"}
SHORT = {"W": "W: AtP + stoch flag, patch flagged", "W-JQ": "W-JQ: same, stoch J+Q elsewhere",
         "a": "(a) patch top |AtP|", "b": "(b) patch random", "c": "(c) exact flag (cost ≥ 4n)",
         "c-JQ": "(c-JQ) exact flag, exact J+Q elsewhere", "d": "(d) EAP-IG16 + patch top |IG|",
         "f": "(f) stoch J+Q + patch top |J+½q̂|"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": "#333333",
                     "xtick.color": "#555555", "ytick.color": "#555555"})
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
YMAX = 1.05 * max(RES[(t, m, x)]["rel"] for t in TASKS for m in METHODS for x in XS_FIG)
for ax, t in zip(axs, TASKS):
    for m in ["a", "W", "W-JQ", "f", "b", "d", "c", "c-JQ"]:
        X = [RES[(t, m, x)]["F"] for x in XS_FIG]
        Y = [RES[(t, m, x)]["rel"] for x in XS_FIG]
        ax.plot(X, Y, color=COL[m], lw=2, marker="o", ms=3.5, label=SHORT[m],
                ls="--" if m in ("b", "c", "c-JQ") else "-")
    ax.plot([1 + NS[t]], [0], marker="*", ms=12, color="#222222", ls="none", label="(e) patch everything")
    ax.axvline(1 + NS[t], color="#cccccc", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(3, 2e4)
    ax.set_ylim(-0.01, YMAX)
    ax.grid(True, which="major", color="#eeeeee", lw=0.6)
    ax.set_title(f"{t}: heads + MLPs, {len(R[t])} pairs, n ≈ {NS[t]} sites", fontsize=9)
    ax.set_xlabel("nominal cost per pair (forward-pass equivalents)")
axs[0].set_ylabel("rel L1 error vs actual patching")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(os.path.join(FIG, "flag_then_patch.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- write section
SECTION = SECTION.replace("VERDICT_PLACEHOLDER", VERDICT if "VERDICT" in globals() else verdict)
if DRY:
    print(SECTION); sys.exit(0)
doc = open(DST).read()
if START in doc:
    doc = doc[:doc.index(START)] + SECTION + doc[doc.index(END) + len(END):]
elif "\n## Figures" in doc:
    i = doc.index("\n## Figures") + 1
    doc = doc[:i] + SECTION + "\n\n" + doc[i:]
else:
    doc = doc.rstrip("\n") + "\n\n" + SECTION + "\n"
open(DST, "w").write(doc)
print("wrote", DST, "and", os.path.join(FIG, "flag_then_patch.png"))
