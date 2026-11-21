#!/usr/bin/env python3
"""Picking the layer, not just the dose: do the J / J+Q flip-dose predictions of S2 choose WHERE to swap?

Input: S2.json shards (second_order.py --stages S2, part T: two-hop "croissant" swaps on Qwen3.8-27B). Per trial and
method m (single layer l, or band clamp a-b): the clean gap r0 = logit(new) - logit(old) < 0, exact jets g1 = r'(0),
g2 = r''(0), and the actual gap r(lam) on a lam grid (+ beta_l = |Delta_l|/|h_l| for single layers, the L1 push budget
sum_l |d_l|/|h_l| per lam for bands).

  python src/layer_pick.py [S2.json ...] [--md out.md] [--tag name]
    default inputs: $QLENS_OUT/second_order_v1/s2/t*/S2.json (the original 6-point grid)

Doses. Actual flip dose lam*_act = first crossing of r(lam) = 0 on the grid, linear interpolation between the
bracketing grid points (as summarize_s2.py); inf if no flip within the grid. Predicted: lam*_J = -r0/g1 (g1 > 0),
lam*_JQ = smallest positive root of r0 + g1 lam + g2 lam^2 / 2. Push = the relative push size at the flip: lam * beta_l
for a single layer, L1(lam) for a band (budget curve interpolated through (0, 0) and the grid; linear extrapolation
past the grid). r0 is common to every method of a trial, so J's dose ranking is the ranking of 1/g1.

Questions (single layers = the 11-layer S2 grid; all = singles + the 5 bands):
  (i)   pick: argmin over layers of the predicted dose (or push) vs the layer that actually needs the smallest dose
        (push). Accuracy (on all trials, and on trials whose actual best is resolved by the grid brackets), regret =
        actual cost of the pick / actual best cost, fraction of picks that do not flip within the grid. Baselines:
        the fixed layer chosen by 2-fold CV (most often best on the other half), and beta alone for push.
  (ii)  band vs single: does the best band need less than every single layer? predicted vs actual, dose and push.
  (iii) Spearman per trial between predicted and actual cost over methods (censored = tied at the top).
CIs: paired bootstrap over trials (2000 resamples, seed 0), as summarize_s2.py.
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ARGV = sys.argv[1:]
MD = ARGV[ARGV.index("--md") + 1] if "--md" in ARGV else None
TAG = ARGV[ARGV.index("--tag") + 1] if "--tag" in ARGV else ""
OLD = ARGV[ARGV.index("--old") + 1] if "--old" in ARGV else None
skip = {MD, TAG, OLD}
paths = [Path(x) for x in ARGV if not x.startswith("--") and x not in skip] or \
    sorted((Path(os.environ.get("QLENS_OUT", "outputs")) / "second_order_v1" / "s2").glob("t*/S2.json"))
D = {}
for fp in paths:
    d = json.loads(fp.read_text())
    for k, v in d.items():
        if k == "twohop" and k in D:
            D[k] += v
        elif k == "secs":
            D["secs"] = round(D.get("secs", 0) + v, 1)
        else:
            D.setdefault(k, v)
OUT = []
INF = math.inf


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def boot(n, stat, B=2000, seed=0):
    rnd = random.Random(seed)
    vals = sorted(stat([rnd.randrange(n) for _ in range(n)]) for _ in range(B))
    vals = [v for v in vals if v == v]
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


# ---------------------------------------------------------------- doses
def flip_actual(r0, lams, rs):
    """(interpolated first crossing, bracket lo, bracket hi); (inf, last lam, inf) if no flip within the grid."""
    pl, pr = 0.0, r0
    for lam, rv in zip(lams, rs):
        if rv > 0:
            x = pl + (lam - pl) * (-pr) / (rv - pr) if rv != pr else lam
            return x, pl, lam
        pl, pr = lam, rv
    return INF, lams[-1], INF


def flip_J(r0, g1):
    return -r0 / g1 if g1 > 0 else INF


def flip_JQ(r0, g1, g2):
    if abs(g2) < 1e-12:
        return flip_J(r0, g1)
    disc = g1 * g1 - 2 * g2 * r0
    if disc < 0:
        return INF
    pos = [x for x in ((-g1 + s * math.sqrt(disc)) / g2 for s in (1, -1)) if x > 0]
    return min(pos) if pos else INF


def interp_budget(lam, lams, L1):
    if not math.isfinite(lam):
        return INF
    xs, ys = [0.0] + list(lams), [0.0] + list(L1)
    for i in range(1, len(xs)):
        if lam <= xs[i]:
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (lam - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[-1] + (ys[-1] - ys[-2]) * (lam - xs[-1]) / (xs[-1] - xs[-2])


# ---------------------------------------------------------------- per trial, per method
T = [r for r in D["twohop"] if "single" in r]
LAMS, BL, LS = D["lams"], D["band_lams"], D["layers"]
BK = list(T[0]["band"])
METH = [f"L{l}" for l in LS] + [f"B{k}" for k in BK]
NS = len(LS)
SING, ALL = METH[:NS], METH
PUSH_UNITS = ("L1", "L2", "max")


def table(r):
    """method -> {beta, dose: {act, lo, hi, J, JQ}, conv: {unit: dose -> push}}"""
    r0 = r["clean"]["r"]
    out = {}
    for l in LS:
        e = r["single"][str(l)]
        a, lo, hi = flip_actual(r0, LAMS, [o["r"] for o in e["out"]])
        b = e["beta"]
        f = (lambda x, b=b: x * b)  # one layer: L1 = L2 = max = lam * beta
        out[f"L{l}"] = dict(beta=b, dose=dict(act=a, lo=lo, hi=hi, J=flip_J(r0, e["g1"]), JQ=flip_JQ(r0, e["g1"], e["g2"])),
                            conv={u: f for u in PUSH_UNITS})
    for k in BK:
        e = r["band"][k]
        a, lo, hi = flip_actual(r0, BL, [o["r"] for o in e["out"]])
        conv = {u: (lambda x, ys=[bb[u] for bb in e["budget"]]: interp_budget(x, BL, ys)) for u in PUSH_UNITS}
        out[f"B{k}"] = dict(beta=None, dose=dict(act=a, lo=lo, hi=hi, J=flip_J(r0, e["g1"]), JQ=flip_JQ(r0, e["g1"], e["g2"])),
                            conv=conv)
    return out


TB = [table(r) for r in T]
n = len(T)
fin = math.isfinite

# per-method calibration (2-fold CV by trial parity): divide the predicted dose by the method's median predicted/actual
# ratio on the other half, over trials that flip within the grid. Removes the layer-dependent overshoot (s2_results).
# Censoring: where fewer than half of the training trials flip (the earliest layers), the factor is not identifiable
# (the flipped trials are a selected easy subset; bounding the rest at the grid top makes the layer look cheap), so that
# method keeps its raw prediction (factor 1). Which methods are affected is printed.
CAL, RAW = {}, set()
for f in (0, 1):
    train = [i for i in range(n) if i % 2 != f]
    for s_ in ("J", "JQ"):
        for c in METH:
            rat = [TB[i][c]["dose"][s_] / TB[i][c]["dose"]["act"] for i in train
                   if fin(TB[i][c]["dose"][s_]) and fin(TB[i][c]["dose"]["act"])]
            ok = sum(fin(TB[i][c]["dose"]["act"]) for i in train) >= 0.5 * len(train) and rat
            m = median(rat) if ok else 1.0
            if not ok:
                RAW.add(c)
            CAL[(f, c, s_)] = (m, len(rat))
            for i in range(n):
                if i % 2 == f:
                    TB[i][c]["dose"][s_ + "c"] = TB[i][c]["dose"][s_] / m


def cost(i, c, unit, src):
    d = TB[i][c]["dose"][src]
    return d if unit == "dose" else TB[i][c]["conv"][unit](d)


def argmin(vals):
    best, bi = INF, None
    for i, v in enumerate(vals):
        if v < best:
            best, bi = v, i
    return bi


def pick(i, cands, unit, src):
    b = argmin([cost(i, c, unit, src) for c in cands])
    return cands[b] if b is not None else None


def cv_fixed(cands, unit):
    """2-fold CV by parity: the candidate most often actually best on the other half, used for every trial."""
    out = [None] * n
    for f in (0, 1):
        cnt = {c: 0 for c in cands}
        for i in range(n):
            if i % 2 != f:
                b = pick(i, cands, unit, "act")
                if b:
                    cnt[b] += 1
        best = max(cands, key=lambda c: cnt[c])
        for i in range(n):
            if i % 2 == f:
                out[i] = best
    return out


def resolved(i, cands, unit):
    """The actual best method's bracket lies entirely below every other method's bracket."""
    b = pick(i, cands, unit, "act")
    return b is not None and all(cost(i, b, unit, "hi") <= cost(i, c, unit, "lo") for c in cands if c != b)


def bci(xs):
    lo, hi = boot(len(xs), lambda ix: mean(xs[i] for i in ix))
    return f"{mean(xs):+.2f} [{lo:+.2f}, {hi:+.2f}]"


def caplog(x):
    return min(math.log(x), 5.0) if fin(x) else 5.0


# ---------------------------------------------------------------- report
say(f"# Layer picking from the S2 jets ({TAG or 'S2 two-hop'})")
say()
say(f"Source: {len(paths)} shard(s): {', '.join(str(p_.parent.name) for p_ in paths)} under {paths[0].parent.parent}")
say(f"Trials: {n} two-hop swaps the model answers correctly. Single-layer grid λ ∈ {LAMS}; band grid λ ∈ {BL}.")
say("Push units: β_l = |Δ_l|/|h_l| (relative to the clean residual norm; S2.json stores no absolute |Δ|). One layer: "
    "push = λβ_l in every unit. Band: the realised clamp pushes d_l summed as L1 = Σ|d_l|/|h_l|, L2 = root-sum-square, "
    "max = largest single-layer push.")
say()
say("## Resolution of the actual flip dose")
say("Per method: trials that flip within the grid, median bracket width in λ, median width / interpolated dose.")
say()
say("| " + " | ".join(METH) + " |")
say("|" + "---|" * len(METH))
row = []
for c in METH:
    ok = [i for i in range(n) if fin(TB[i][c]["dose"]["act"])]
    w = [TB[i][c]["dose"]["hi"] - TB[i][c]["dose"]["lo"] for i in ok]
    rel = [w_ / TB[i][c]["dose"]["act"] for w_, i in zip(w, ok)]
    row.append(f"{len(ok)}, {median(w):.2f}, {median(rel):.2f}")
say("| " + " | ".join(row) + " |")
say()
for unit in ("dose", "L1"):
    for nm, cands in (("11 single layers", SING), ("16 methods", ALL)):
        say(f"- {unit}, {nm}: the actual best is resolved by the grid brackets in "
            f"{mean(resolved(i, cands, unit) for i in range(n)):.2f} of trials.")
say()
say("Not calibrated (fewer than half of the training trials flip within the grid): " + (", ".join(sorted(RAW, key=METH.index)) or "none"))
say("Calibration factors (median predicted/actual dose per method, fold 0 | fold 1, J+Q): " +
    ", ".join(f"{c} {CAL[(0, c, 'JQ')][0]:.2f}|{CAL[(1, c, 'JQ')][0]:.2f}" for c in METH))
say()

# --- (i)
say("## (i) Does the predicted cost pick the layer that actually needs the least?")
say("Pick = argmin over the 11 single layers of the predicted dose λ* (unit dose) or push λ*·β (unit L1). Hit = the "
    "pick is the actual best layer. Regret = actual cost at the pick / actual best cost (inf if the pick does not flip "
    "within the grid). 'cal' = the per-method calibrated prediction (2-fold CV). CV-fixed = the one layer most often "
    "best on the other half, for every trial. β-only = argmin β (an equal-dose guess). Paired bootstrap over trials.")
say()
say("| unit | picker | hit rate | hit, resolved trials | median regret | share regret ≤ 1.1 | pick does not flip | "
    "mean log regret (inf → 5) |")
say("|---|---|---|---|---|---|---|---|")
PK = {}
for unit in ("dose", "L1"):
    best = [pick(i, SING, unit, "act") for i in range(n)]
    valid = [i for i in range(n) if best[i]]
    res = set(i for i in valid if resolved(i, SING, unit))
    pk = {"J": [pick(i, SING, unit, "J") or SING[-1] for i in range(n)],
          "J+Q": [pick(i, SING, unit, "JQ") or SING[-1] for i in range(n)],
          "J cal": [pick(i, SING, unit, "Jc") or SING[-1] for i in range(n)],
          "J+Q cal": [pick(i, SING, unit, "JQc") or SING[-1] for i in range(n)],
          "CV-fixed": cv_fixed(SING, unit)}
    if unit != "dose":
        pk["β-only"] = [SING[argmin([TB[i][c]["beta"] for c in SING])] for i in range(n)]
    for name, p_ in pk.items():
        hit = [float(p_[i] == best[i]) for i in valid]
        reg = [cost(i, p_[i], unit, "act") / cost(i, best[i], unit, "act") for i in valid]
        PK[(unit, name)] = (hit, [caplog(x) for x in reg])
        hres = [h for h, i in zip(hit, valid) if i in res]
        say(f"| {unit} | {name} | {mean(hit):.2f} | {mean(hres):.2f} (n={len(hres)}) | {median(reg):.2f} | "
            f"{mean(x <= 1.1 for x in reg):.2f} | {mean(not fin(x) for x in reg):.2f} | {mean(PK[(unit, name)][1]):.3f} |")
    for a_, b_ in (("J+Q", "J"), ("J+Q cal", "J cal"), ("J+Q", "CV-fixed"), ("J+Q cal", "CV-fixed")):
        ha, la = PK[(unit, a_)]; hb, lb = PK[(unit, b_)]
        say(f"| {unit} | {a_} − {b_} | {bci([x - y for x, y in zip(ha, hb)])} | | | | | "
            f"{bci([x - y for x, y in zip(la, lb)])} |")
say()
for unit in ("dose", "L1"):
    cnt = lambda lst: ", ".join(f"{c} {lst.count(c)}" for c in SING if lst.count(c))  # noqa: E731
    say(f"- {unit}: actual best: {cnt([pick(i, SING, unit, 'act') for i in range(n)])} | J: "
        f"{cnt([pick(i, SING, unit, 'J') for i in range(n)])} | J+Q: {cnt([pick(i, SING, unit, 'JQ') for i in range(n)])}"
        f" | J+Q cal: {cnt([pick(i, SING, unit, 'JQc') for i in range(n)])}")
say()

# --- (ii)
say("## (ii) Does the prediction say when a band beats every single layer?")
say("Band wins = min over the 5 bands of the actual cost < min over the 11 single layers. Predicted band wins = the "
    "same with predicted costs. Units: dose λ (λ = 1 = the full clean swap, at one layer or clamped at every band "
    "layer), and the push at the flip in L1, L2 and max. Balanced accuracy = mean hit rate on band-wins and "
    "single-wins trials. Last two columns: the pick over all 16 methods.")
say()
say("| unit | actual band-wins | predictor | predicted band-wins | accuracy | balanced acc. | 16-way hit rate | "
    "16-way median regret |")
say("|---|---|---|---|---|---|---|---|")
for unit in ("dose",) + PUSH_UNITS:
    valid = [i for i in range(n) if pick(i, ALL, unit, "act")]
    aw = [min(cost(i, c, unit, "act") for c in ALL[NS:]) < min(cost(i, c, unit, "act") for c in SING) for i in valid]
    acc_ = {}
    for pname, src in (("J", "J"), ("J+Q", "JQ"), ("J+Q cal", "JQc")):
        pw = [min(cost(i, c, unit, src) for c in ALL[NS:]) < min(cost(i, c, unit, src) for c in SING) for i in valid]
        acc = [float(a == p_) for a, p_ in zip(aw, pw)]
        pos = [p_ for a, p_ in zip(aw, pw) if a]; neg = [not p_ for a, p_ in zip(aw, pw) if not a]
        bal = (mean(pos) + mean(neg)) / 2 if pos and neg else float("nan")
        pk_ = [pick(i, ALL, unit, src) or ALL[-1] for i in valid]
        bst = [pick(i, ALL, unit, "act") for i in valid]
        hit = [float(a == b) for a, b in zip(pk_, bst)]
        reg = [cost(i, a, unit, "act") / cost(i, b, unit, "act") for i, a, b in zip(valid, pk_, bst)]
        acc_[pname] = (acc, hit)
        say(f"| {unit} | {mean(aw):.2f} | {pname} | {mean(pw):.2f} | {mean(acc):.2f} | {bal:.2f} | {mean(hit):.2f} | "
            f"{median(reg):.2f} |")
    say(f"| {unit} | | J+Q − J | | {bci([x - y for x, y in zip(acc_['J+Q'][0], acc_['J'][0])])} | | "
        f"{bci([x - y for x, y in zip(acc_['J+Q'][1], acc_['J'][1])])} | |")
say()

# --- (iii)
say("## (iii) Does the predicted ranking follow the actual ranking? (Spearman per trial)")
say("Censored costs (no flip within the grid, or no predicted flip) are tied at the top. 'population' = each "
    "method's mean actual rank on the other half of the trials (2-fold CV); β-only = rank by β (L1, singles).")
say()
say("| unit | methods | J | J+Q | J+Q cal | population | J+Q − J | J+Q cal − population | β-only |")
say("|---|---|---|---|---|---|---|---|---|")


def ranks_of(xs):
    return rankdata([x if fin(x) else 1e300 for x in xs])


def spear(a, b):
    ra, rb = ranks_of(a), ranks_of(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


for unit in ("dose", "L1"):
    for mname, cands in (("11 singles", SING), ("16 methods", ALL)):
        act = [[cost(i, c, unit, "act") for c in cands] for i in range(n)]
        S = {s_: [spear([cost(i, c, unit, s_) for c in cands], act[i]) for i in range(n)] for s_ in ("J", "JQ", "JQc")}
        pop = [None] * n
        for f in (0, 1):
            rk = list(np.mean([ranks_of(act[i]) for i in range(n) if i % 2 != f], 0))
            for i in range(n):
                if i % 2 == f:
                    pop[i] = spear(rk, act[i])
        S["pop"] = pop
        if unit != "dose" and cands is SING:
            S["beta"] = [spear([TB[i][c]["beta"] for c in cands], act[i]) for i in range(n)]
        ok = [i for i in range(n) if all(v[i] == v[i] for v in S.values())]
        m_ = lambda k: f"{mean(S[k][i] for i in ok):.2f} ({median([S[k][i] for i in ok]):.2f})"  # noqa: E731
        say(f"| {unit} | {mname} (n={len(ok)}) | {m_('J')} | {m_('JQ')} | {m_('JQc')} | {m_('pop')} | "
            f"{bci([S['JQ'][i] - S['J'][i] for i in ok])} | {bci([S['JQc'][i] - S['pop'][i] for i in ok])} | "
            f"{m_('beta') if 'beta' in S else ''} |")
say()
say("Entries: mean (median) over trials.")
say()

# --- dose error by method
say("## Dose error by method (median |log(predicted / actual)| over trials that flip within the grid)")
say("| method | n flip | J | J+Q | J+Q cal | J+Q / actual, median |")
say("|---|---|---|---|---|---|")
for c in METH:
    ok = [i for i in range(n) if fin(TB[i][c]["dose"]["act"])]
    e = {s_: median([abs(math.log(TB[i][c]["dose"][s_] / TB[i][c]["dose"]["act"])) if fin(TB[i][c]["dose"][s_]) else INF
                     for i in ok]) for s_ in ("J", "JQ", "JQc")}
    rq = median([TB[i][c]["dose"]["JQ"] / TB[i][c]["dose"]["act"] for i in ok if fin(TB[i][c]["dose"]["JQ"])])
    say(f"| {c} | {len(ok)} | {e['J']:.2f} | {e['JQ']:.2f} | {e['JQc']:.2f} | {rq:.2f} |")
say()
# --- the S2 dose claim (summarize_s2.py: median |log(predicted/actual)| over every (trial, method) that flips), with a
# cluster bootstrap over trials (all methods of a resampled trial together)
say("## The S2 dose claim on this grid: median dose miss over all (trial, method) cases that flip")
say("Miss = exp(median |log(predicted / actual)|) (no predicted flip = infinite miss), as summarize_s2.py. 95% CI: "
    "bootstrap over trials, 2000 resamples (each trial carries all its methods). Bias = exp(median log(pred/actual)).")
say()
say("| case set | n cases | J miss | J+Q miss | J+Q closer | J+Q bias | J+Q predicts no flip |")
say("|---|---|---|---|---|---|---|")
CASES = []  # (trial, method, eJ, eQ, signed log JQ)
for i in range(n):
    for c in METH:
        d = TB[i][c]["dose"]
        if not fin(d["act"]):
            continue
        eJ = abs(math.log(d["J"] / d["act"])) if fin(d["J"]) else INF
        eQ = abs(math.log(d["JQ"] / d["act"])) if fin(d["JQ"]) else INF
        CASES.append((i, c, eJ, eQ, math.log(d["JQ"] / d["act"]) if fin(d["JQ"]) else INF, d["act"]))


def claim_row(name, cs):
    byt = {}
    for x in cs:
        byt.setdefault(x[0], []).append(x)
    tl = list(byt)
    rnd = np.random.default_rng(0)

    def stat(sub):
        eJ = np.array([x[2] for x in sub]); eQ = np.array([x[3] for x in sub]); sg = np.array([x[4] for x in sub])
        return (math.exp(np.median(eJ)), math.exp(np.median(eQ)), float(np.mean(eQ < eJ)),
                math.exp(np.median(sg)), float(np.mean(~np.isfinite(eQ))))
    v = stat(cs)
    bs = np.array([stat([x for t in rnd.choice(tl, len(tl)) for x in byt[t]]) for _ in range(2000)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    f = lambda k, p: f"{v[k]:.{p}f} [{lo[k]:.{p}f}, {hi[k]:.{p}f}]"  # noqa: E731
    say(f"| {name} | {len(cs)} | ×{f(0, 2)} | ×{f(1, 2)} | {f(2, 2)} | ×{f(3, 2)} | {f(4, 3)} |")


claim_row("all methods", CASES)
claim_row("all methods, actual dose ≤ 2 (the original grid's range)", [x for x in CASES if x[5] <= 2.0])
claim_row("single layers", [x for x in CASES if x[1][0] == "L"])
claim_row("bands", [x for x in CASES if x[1][0] == "B"])
for c in ("L40", "L44", "L48", "B40-55"):
    claim_row(c, [x for x in CASES if x[1] == c])
say()
if OLD:
    od = {}
    for fp in sorted(Path(OLD).parent.parent.glob(Path(OLD).parent.name + "/" + Path(OLD).name)) if "*" in OLD else [Path(OLD)]:
        for r in json.loads(fp.read_text())["twohop"]:
            if "single" in r:
                od[(r["func"], r["src"], r["tgt"])] = r
    sh = {"single": [], "band": []}
    for i, r in enumerate(T):
        o = od.get((r["func"], r["src"], r["tgt"]))
        if o is None:
            continue
        r0 = o["clean"]["r"]
        for l in LS:
            a_old = flip_actual(r0, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], [x["r"] for x in o["single"][str(l)]["out"]])[0]
            a_new = TB[i][f"L{l}"]["dose"]["act"]
            if fin(a_old) and fin(a_new):
                sh["single"].append(math.log(a_old / a_new))
        for k in BK:
            a_old = flip_actual(r0, [0.25, 0.5, 1.0, 2.0], [x["r"] for x in o["band"][k]["out"]])[0]
            a_new = TB[i][f"B{k}"]["dose"]["act"]
            if fin(a_old) and fin(a_new):
                sh["band"].append(math.log(a_old / a_new))
    say("Original-grid interpolated dose against this grid's, same (trial, method), both finite: " + "; ".join(
        f"{k}: n={len(v)}, median ratio ×{math.exp(np.median(v)):.3f}, median |log| {np.median(np.abs(v)):.3f}, "
        f"90th pct |log| {np.percentile(np.abs(v), 90):.3f}" for k, v in sh.items() if v))
    say()
# ---------------------------------------------------------------- calibrated-J baseline (is J+Q more than a scale fix?)
# Each prediction is multiplied by a constant exp(-m), m = median log(predicted / actual) over the TRAINING trials'
# flipping cases (5-fold CV by trial, folds from a seed-0 permutation). Global: one m for all methods; per depth: one m
# per method (11 layers + 5 bands; fewer than MINC training cases -> the global m). A predicted no-flip (J: g1 <= 0;
# J+Q: no positive root) stays an infinite miss. Evaluated on every flipping (trial, method); the case filter
# "actual <= 2" is applied after fitting. CIs: cluster bootstrap over trials of the CV errors (fits held fixed).
KF, MINC = 5, 20
_perm = np.random.default_rng(0).permutation(n)
FOLD = np.empty(n, int)
FOLD[_perm] = np.arange(n) % KF
CC = [(i, c, TB[i][c]["dose"]["J"], TB[i][c]["dose"]["JQ"], TB[i][c]["dose"]["act"])
      for i in range(n) for c in METH if fin(TB[i][c]["dose"]["act"])]
CT = np.array([x[0] for x in CC]); CM = np.array([x[1] for x in CC]); CA = np.array([x[4] for x in CC])
SG = {s_: np.array([math.log(x[j] / x[4]) if fin(x[j]) else INF for x in CC]) for s_, j in (("J", 2), ("JQ", 3))}
FITS = {}


def cv_shift(s_, per):
    sh = np.zeros(len(CC))
    if per == "raw":
        return sh
    for k in range(KF):
        tr, te = FOLD[CT] != k, FOLD[CT] == k
        ok = tr & np.isfinite(SG[s_])
        g = float(np.median(SG[s_][ok]))
        FITS[(s_, "global", k)] = g
        if per == "global":
            sh[te] = g
            continue
        for c in METH:
            sel = ok & (CM == c)
            m = float(np.median(SG[s_][sel])) if sel.sum() >= MINC else g
            FITS[(s_, c, k)] = (m, int(sel.sum()))
            sh[te & (CM == c)] = m
    return sh


ERR = {(s_, per): SG[s_] - cv_shift(s_, per) for s_ in ("J", "JQ") for per in ("raw", "global", "depth")}
BYT = {}
for k_, t in enumerate(CT):
    BYT.setdefault(int(t), []).append(k_)
BYT = {t: np.array(v) for t, v in BYT.items()}


def cal_stats(eJ, eQ):
    aJ, aQ = np.abs(eJ), np.abs(eQ)
    mJ, mQ = np.median(aJ), np.median(aQ)
    return (math.exp(mJ), math.exp(mQ), math.exp(mJ - mQ) if np.isfinite(mJ - mQ) else INF, float(np.mean(aQ < aJ)),
            math.exp(np.median(eJ)), math.exp(np.median(eQ)), float(np.mean(eJ > 0)))


def cal_row(name, per, mask):
    idx = np.nonzero(mask)[0]
    tl = sorted({int(CT[k_]) for k_ in idx})
    keep = np.zeros(len(CC), bool); keep[idx] = True
    eJ, eQ = ERR[("J", per)], ERR[("JQ", per)]
    v = cal_stats(eJ[idx], eQ[idx])
    rnd = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        ii = np.concatenate([BYT[t][keep[BYT[t]]] for t in rnd.choice(tl, len(tl))])
        bs.append(cal_stats(eJ[ii], eQ[ii]))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    f = lambda k, p: f"{v[k]:.{p}f} [{lo[k]:.{p}f}, {hi[k]:.{p}f}]"  # noqa: E731
    say(f"| {name} | {len(idx)} | ×{f(0, 2)} | ×{f(1, 2)} | ×{f(2, 2)} | {f(3, 3)} | ×{f(4, 2)} | ×{f(5, 2)} | "
        f"{f(6, 3)} |")


say("## Calibrated-J baseline: is J+Q's advantage over J more than one scale factor?")
say(f"Prediction × exp(-m), m = median log(pred/actual) on the training trials ({KF}-fold CV by trial); global = one m "
    f"for all methods, per depth = one m per method (< {MINC} training cases: global). Miss = exp(median |log|); "
    "J/J+Q = ratio of the two misses; closer = share of cases with |log err| of J+Q < that of J (strict); "
    "bias = exp(median signed log); J over = share of cases where the J prediction exceeds the actual dose (no "
    "predicted flip counts as over). 95% CI: cluster bootstrap over trials, 2000 resamples, CV fits held fixed.")
say()
say("| case set, calibration | n | J miss | J+Q miss | J / J+Q | J+Q closer | J bias | J+Q bias | J over |")
say("|---|---|---|---|---|---|---|---|---|")
for cname, cmask in (("actual ≤ 2", CA <= 2.0), ("all flips", CA < INF), ("single layers", np.char.startswith(CM.astype(str), "L")),
                     ("bands", np.char.startswith(CM.astype(str), "B"))):
    for per in ("raw", "global", "depth"):
        cal_row(f"{cname}, {per}", per, cmask)
say()
say("Fitted factors (geometric mean over folds of exp(m), i.e. median pred/actual; per-depth n = training cases in fold 0):")
say("| method | n flip | J factor | J+Q factor | J miss raw → depth | J+Q miss raw → depth | J+Q closer (depth) |")
say("|---|---|---|---|---|---|---|")
for s_ in ("J", "JQ"):
    say_g = math.exp(np.mean([FITS[(s_, "global", k)] for k in range(KF)]))
    FITS[(s_, "g")] = say_g
say(f"| global | {len(CC)} | ×{FITS[('J', 'g')]:.2f} | ×{FITS[('JQ', 'g')]:.2f} | | | |")
for c in METH:
    sel = CM == c
    fJ = math.exp(np.mean([FITS[("J", c, k)][0] for k in range(KF)]))
    fQ = math.exp(np.mean([FITS[("JQ", c, k)][0] for k in range(KF)]))
    nfb = sum(FITS[("J", c, k)][1] < MINC for k in range(KF))
    mi = lambda s_, p: math.exp(np.median(np.abs(ERR[(s_, p)][sel])))  # noqa: E731
    cl = float(np.mean(np.abs(ERR[("JQ", "depth")][sel]) < np.abs(ERR[("J", "depth")][sel])))
    say(f"| {c} | {int(sel.sum())} | ×{fJ:.2f}{' (global in %d folds)' % nfb if nfb else ''} | ×{fQ:.2f} | "
        f"×{mi('J', 'raw'):.2f} → ×{mi('J', 'depth'):.2f} | ×{mi('JQ', 'raw'):.2f} → ×{mi('JQ', 'depth'):.2f} | {cl:.2f} |")
say()
if MD:
    Path(MD).write_text("\n".join(OUT) + "\n")
    print("wrote", MD)
