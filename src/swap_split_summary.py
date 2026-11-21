#!/usr/bin/env python3
"""Summarise src/swap_split.py: is Q along the actual swap pushes as collective as along iso/data directions (C4)?

  python src/swap_split_summary.py $QLENS_OUT/swap_split_v1/swap_split*.json [--md results/tables/swap_split_tables.md]

Per layer and direction type, median over trials (types with 2 directions per trial are averaged within the trial
first), 95% bootstrap CI over trials (2000 resamples, seed 0). Paired: swap minus baseline per trial. C4 quantities
along q_hat = Q/|Q| (as paper/claims.py), and the same along u = dr/dh_L (the decision readout).
Outcome split (swap rows): flip at the full swap (S2 r(lam = 1) > 0), the S2 flip dose (interpolated first crossing),
and the curvature ratio uQ/(2 uJ) = (1/2) g2_Q / g1; Spearman over trials with bootstrap CIs.
Reference: A2's own fineweb runs (F = sum over valid positions, T = 128) at the same layers, from A2_b32.json.
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARGV = sys.argv[1:]
MD = ARGV[ARGV.index("--md") + 1] if "--md" in ARGV else None
paths = [Path(x) for x in ARGV if not x.startswith("--") and x != MD]
TR, TYPES, NSEL = [], None, 0
for fp in paths:
    d = json.loads(fp.read_text())
    TR += d["trials"]; TYPES = d["types"]; NSEL = d["n_selected"]
LAYERS = sorted({int(l) for t in TR for l in t["layers"]})
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def boot_ci(vals, stat=np.median, B=2000, seed=0):
    vals = np.asarray([v for v in vals if v == v], float)
    if len(vals) < 2:
        return float(stat(vals)) if len(vals) else float("nan"), float("nan"), float("nan")
    rnd = np.random.default_rng(seed)
    bs = [stat(vals[rnd.integers(0, len(vals), len(vals))]) for _ in range(B)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(stat(vals)), float(lo), float(hi)


def fmt(v):
    m, lo, hi = v
    return f"{m:.3g} [{lo:.3g}, {hi:.3g}]"


# ---- quantities per row (C4 definitions; u-versions use the decision readout)
def q_top(r, lvl, sfx=""):
    s = r[lvl + sfx]
    return s["top_share"]["1"] * s["ci"] * abs(s["total"]) / abs(r["scal_block" + sfx]["total"])


QTY = {
    "block ci/√pr (vector)": lambda r, K: r["block"]["ci"] / math.sqrt(r["block"]["pr"]),
    "best single block leaves (vector)": lambda r, K: r["block"]["best_single_err"],
    "block n90 / K (vector)": lambda r, K: r["block"]["n90"] / K,
    "top block along q̂": lambda r, K: q_top(r, "scal_block"),
    "top head along q̂": lambda r, K: q_top(r, "scal_heads"),
    "top neuron along q̂": lambda r, K: q_top(r, "scal_neurons"),
    "block n90 along q̂ / K": lambda r, K: r["scal_block"]["n90"] / K,
    "top block along u": lambda r, K: q_top(r, "scal_block", "_u"),
    "top head along u": lambda r, K: q_top(r, "scal_heads", "_u"),
    "top neuron along u": lambda r, K: q_top(r, "scal_neurons", "_u"),
    "block n90 along u / K": lambda r, K: r["scal_block_u"]["n90"] / K,
    "block ci along u": lambda r, K: r["scal_block_u"]["ci"],
    "MLP type share along u": lambda r, K: r["type_share_u"]["mlp"],
    "(Q/J) per unit push norm": lambda r, K: r["Q_over_J"] / r["fro"],
}


def per_trial(l, typ, f):
    """value per trial (mean over the type's directions in that trial)."""
    out = []
    for t in TR:
        e = t["layers"].get(str(l))
        if e is None:
            out.append(float("nan")); continue
        K = len(e["blocks"])
        v = [f(r, K) for r in e["rows"] if r["type"] == typ]
        out.append(float(np.mean(v)) if v else float("nan"))
    return np.array(out)


say(f"# Q split along the actual swap pushes (S2 T, Qwen3.8-27B fp32)")
say()
say(f"Files: {', '.join(p.name for p in paths)} ({len(TR)} of {NSEL} selected trials); layers {LAYERS}; directions per "
    f"trial and layer: {TYPES}.")
say()
# ---- gates
say("## Exactness gates (max over trials)")
gk = sorted({k for t in TR for e in t["layers"].values() for k in e["gates"]})
say("| layer | " + " | ".join(gk) + " |")
say("|---|" + "---|" * len(gk))
for l in LAYERS:
    es = [t["layers"][str(l)] for t in TR if str(l) in t["layers"]]
    say(f"| L{l} (n={len(es)}) | " + " | ".join(f"{max(abs(e['gates'].get(k, float('nan'))) for e in es):.1e}" for k in gk) + " |")
say()
say(f"Seconds per (trial, layer): " + ", ".join(
    f"L{l} {np.median([t['layers'][str(l)]['secs'] for t in TR if str(l) in t['layers']]):.0f}" for l in LAYERS))
say()

# ---- main table: medians per type
BASE = [t for t in dict.fromkeys(TYPES) if t != "swap"]
for l in LAYERS:
    K = len(next(t["layers"][str(l)]["blocks"] for t in TR if str(l) in t["layers"]))
    say(f"## L{l} (K = {K} downstream blocks): median over trials [95% CI]")
    say("| quantity | swap | " + " | ".join(BASE) + " | swap − iso (paired) | swap − rand (paired) |")
    say("|---|---|" + "---|" * len(BASE) + "---|---|")
    for q, f in QTY.items():
        v = {typ: per_trial(l, typ, f) for typ in dict.fromkeys(TYPES)}
        cells = [fmt(boot_ci(v["swap"]))] + [fmt(boot_ci(v[b])) for b in BASE]
        cells += [fmt(boot_ci(v["swap"] - v["iso"])), fmt(boot_ci(v["swap"] - v["rand"]))]
        say(f"| {q} | " + " | ".join(cells) + " |")
    say()

# ---- block profile along u (where along depth the decision's curvature sits)
say("## Where the decision curvature sits: median share of u·Q per downstream block (swap rows)")
for l in LAYERS:
    es = [t["layers"][str(l)] for t in TR if str(l) in t["layers"]]
    bl = es[0]["blocks"]
    sh = np.array([[x / r["uQ"] for x in r["block_along_u"]] for e in es for r in e["rows"] if r["type"] == "swap"])
    med = np.median(sh, 0)
    top = np.argsort(-np.abs(med))[:5]
    say(f"- L{l}: largest median block shares " + ", ".join(f"L{bl[i]} {med[i]:+.2f}" for i in top)
        + f"; first half of the span {med[: len(bl) // 2].sum():+.2f}, second half {med[len(bl) // 2:].sum():+.2f}")
say()


# ---- outcome split (swap rows)
def flip_dose(s2, r0):
    pl, pr = 0.0, r0
    for lam, rv in zip(s2["lams"], s2["r"]):
        if rv > 0:
            return pl + (lam - pl) * (-pr) / (rv - pr)
        pl, pr = lam, rv
    return float("inf")


say("## Outcome split (swap rows): does a flipping swap, or a more curved one, have a more concentrated split?")
say("flip@1 = S2 r(λ=1) > 0. dose = S2 flip dose (inf if none within λ ≤ 2; ranks treat it as largest). "
    "κ = uQ/(2uJ) = the Q share of the decision's second-order term relative to its first-order term.")
say()
say("| layer | quantity | n flip / no | median flip | median no flip | flip − no [95% CI] | Spearman with κ [95% CI] | "
    "Spearman with dose [95% CI] |")
say("|---|---|---|---|---|---|---|---|")
OQ = ["best single block leaves (vector)", "block ci/√pr (vector)", "top block along u", "block n90 along u / K",
      "top head along u", "top neuron along u", "block ci along u"]
pool = {q: [] for q in OQ}
for l in [*LAYERS, "all"]:
    items = []
    for t in TR:
        for ll in (LAYERS if l == "all" else [l]):
            e = t["layers"].get(str(ll))
            if not e or not e.get("s2"):
                continue
            sw = next(r for r in e["rows"] if r["type"] == "swap")
            s2 = e["s2"]
            items.append((t, e, sw, s2["r"][s2["lams"].index(1.0)] > 0, flip_dose(s2, t["clean_r"]), sw["uQ"] / (2 * sw["uJ"])))
    if not items:
        continue
    rng = np.random.default_rng(0)
    for q in OQ:
        x = np.array([QTY[q](sw, len(e["blocks"])) for (t, e, sw, fl, dz, kp) in items])
        fl = np.array([it[3] for it in items]); dz = np.array([min(it[4], 1e9) for it in items]); kp = np.array([it[5] for it in items])
        a, b_ = x[fl], x[~fl]
        diff = float(np.median(a) - np.median(b_)) if len(a) and len(b_) else float("nan")
        if len(a) > 1 and len(b_) > 1:
            bs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b_, len(b_))) for _ in range(2000)]
            dlo, dhi = np.percentile(bs, [2.5, 97.5])
        else:
            dlo = dhi = float("nan")

        def sp(y):
            rho = spearmanr(x, y).statistic
            n = len(x)
            bs = [spearmanr(x[ix], y[ix]).statistic for ix in (rng.integers(0, n, n) for _ in range(2000))]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            return f"{rho:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        say(f"| {l if l == 'all' else 'L' + str(l)} | {q} | {len(a)} / {len(b_)} | {np.median(a) if len(a) else float('nan'):.3g} | "
            f"{np.median(b_) if len(b_) else float('nan'):.3g} | {diff:+.3g} [{dlo:+.3g}, {dhi:+.3g}] | {sp(kp)} | {sp(dz)} |")
say()
say("Pooled rows ('all') repeat each trial once per layer; their CIs treat (trial, layer) as independent.")
say()

# ---- reference: A2 on fineweb at the same layers
W = Path(os.environ.get("QLENS_OUT", "outputs")) / "qanat_q38-27b"
ref = {}
for fn in ("A2_b32.json", "A2_a32.json"):
    if (W / fn).exists():
        js = json.loads((W / fn).read_text())
        for s, v in js["sources"].items():
            ref[int(s)] = v
say("## Reference: A2 on fineweb contexts (F = sum over valid positions, T = 128; 8 contexts × 8 directions)")
say("| layer | type | best single block leaves | block ci/√pr | block n90 / K | top block along q̂ | top head | top neuron |")
say("|---|---|---|---|---|---|---|---|")
for l in LAYERS:
    if l not in ref:
        say(f"| L{l} | (not in A2) | | | | | | |"); continue
    K = len(ref[l]["blocks"])
    for tn in ("iso", "data"):
        rows = ref[l][tn]["rows"]
        g = lambda f: float(np.median([f(r, K) for r in rows]))  # noqa: E731
        say(f"| L{l} | {tn} | {g(QTY['best single block leaves (vector)']):.3g} | {g(QTY['block ci/√pr (vector)']):.3g} | "
            f"{g(QTY['block n90 / K (vector)']):.3g} | {g(QTY['top block along q̂']):.3g} | {g(QTY['top head along q̂']):.3g} | "
            f"{g(QTY['top neuron along q̂']):.3g} |")
say()
if MD:
    Path(MD).write_text("\n".join(OUT) + "\n")
    print("wrote", MD)
