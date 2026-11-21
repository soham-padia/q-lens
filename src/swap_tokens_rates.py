#!/usr/bin/env python3
"""Rates over swaps from src/swap_tokens.py's per-(swap, layer) JSONL (CPU, no model), with cluster-bootstrap CIs.

Metrics, per layer, over all swaps and per relation:
  target (new) rank 1 / <=4 / <=10 among the raised tokens of J, 1/2 Q [all swaps], and of the J+Q prediction and the
  actual change at lambda* [swaps that flip on the grid at that layer]; the neighbour pattern (J top-1 not the target,
  1/2 Q top-1 is) and its reverse; paired 1/2 Q better than J and the median rank ratio; the source (old) in the 4 (600)
  most-lowered of J and 1/2 Q; cos(J, 1/2 Q) median and IQR; J+Q dose overshoot, prediction cosines, actual rank <= 2.
Matching: "any" = best rank over every first-token variant of new / old (S2's rule; primary), "strict" = the single id
tid(new[0]) (v1 / the paper's table). CIs: 95% percentile, cluster bootstrap over the unordered country pair, B=1000,
seed 0; a rate over a subset is the pooled ratio inside each resample.

  python src/swap_tokens_rates.py $QLENS_OUT/swap_tokens_v2/swap_tokens.jsonl [--md results/tables/swap_token_rates_tables.md]
         [--v1 $QLENS_OUT/swap_tokens_v1/swap_tokens.json] [--json-out results/swap_tokens_rates.json] [--B 1000] [--seed 0]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("--md", default=None)
ap.add_argument("--json-out", default=None, help="default: <jsonl dir>/swap_tokens_rates.json")
ap.add_argument("--v1", default=None, help="v1 swap_tokens.json: check the overlapping cases reproduce")
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--primary", default="24,40,56")
A = ap.parse_args()
PRIMARY = [int(x) for x in A.primary.split(",")]
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- load (last line wins per (swap, layer))
rows = {}
for ln in Path(A.jsonl).read_text().splitlines():
    try:
        r = json.loads(ln)
    except json.JSONDecodeError:
        continue
    rows[(r["func"], r["src"], r["tgt"], int(r["layer"]))] = r
rows = list(rows.values())
LAYERS = sorted({int(r["layer"]) for r in rows})
FUNCS = ["capital", "language", "continent"]
CL = sorted({"|".join(sorted((r["src"], r["tgt"]))) for r in rows})
CI_ = {c: i for i, c in enumerate(CL)}
rng = np.random.default_rng(A.seed)
DRAW = rng.integers(0, len(CL), size=(A.B, len(CL)))          # one set of cluster resamples, shared by every metric
CNT = np.stack([np.bincount(d, minlength=len(CL)) for d in DRAW])  # [B, n_clusters] multiplicity of each cluster
INF = float("inf")


def fnum(x):
    return INF if x in ("inf", "Infinity", None) else float(x)


def feats(r, match):
    e, nk, ok = r["e"], ("new_any" if match == "any" else "new"), ("old_any" if match == "any" else "old")
    rk = e["ranks"]
    la, lq = fnum(e["lam_act"]), fnum(e["lam_JQ"])
    fl = bool(e.get("flipped", math.isfinite(la)))
    return {"cl": CI_["|".join(sorted((r["src"], r["tgt"])))], "func": r["func"], "flip": fl,
            "J": rk["J"][nk]["up"], "Q": rk["halfQ"][nk]["up"], "P": rk["pred"][nk]["up"], "Act": rk["actual"][nk]["up"],
            "oJ": rk["J"][ok]["down"], "oQ": rk["halfQ"][ok]["down"],
            "cosJQ": e["cos"]["J_halfQ"], "cosPA": e["cos"]["pred_actual"], "cosJA": e["cos"]["Jlam_actual"],
            "relP": e["pred_vs_actual_rel"], "relJ": e["J_vs_actual_rel"], "lam": la, "lamJQ": lq}


def boot_rate(f, num, den):
    """pooled rate sum(num)/sum(den), with cluster-bootstrap CI. num, den: per-row 0/1 arrays."""
    cl = np.array([x["cl"] for x in f])
    num, den = np.asarray(num, float), np.asarray(den, float)
    sn = np.bincount(cl, weights=num, minlength=len(CL)); sd = np.bincount(cl, weights=den, minlength=len(CL))
    D = int(den.sum())
    if D == 0:
        return {"rate": None, "k": 0, "n": 0, "lo": None, "hi": None}
    bn, bd = CNT @ sn, CNT @ sd
    with np.errstate(invalid="ignore", divide="ignore"):
        br = bn / bd
    lo, hi = np.nanpercentile(br, [2.5, 97.5])
    return {"rate": float(num.sum() / D), "k": int(num.sum()), "n": D, "lo": float(lo), "hi": float(hi)}


def boot_median(f, vals, mask=None):
    cl = np.array([x["cl"] for x in f]); v = np.asarray(vals, float)
    if mask is not None:
        cl, v = cl[np.asarray(mask, bool)], v[np.asarray(mask, bool)]
    if len(v) == 0:
        return {"med": None, "q25": None, "q75": None, "lo": None, "hi": None, "n": 0}
    meds = []
    for c in CNT:
        w = c[cl]
        meds.append(np.median(np.repeat(v, w)) if w.sum() else np.nan)
    lo, hi = np.nanpercentile(np.array(meds, float), [2.5, 97.5])
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"med": float(med), "q25": float(q25), "q75": float(q75), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


def metrics(f):
    one = [1] * len(f)
    fl = [int(x["flip"]) for x in f]
    M = {"n": len(f), "n_flip": sum(fl)}
    for nm in ("J", "Q"):
        for k in (1, 4, 10):
            M[f"{nm}_le{k}"] = boot_rate(f, [x[nm] <= k for x in f], one)
            M[f"{nm}_le{k}_flip"] = boot_rate(f, [x[nm] <= k and x["flip"] for x in f], fl)
    for nm in ("P", "Act"):
        for k in (1, 2, 4, 10):
            M[f"{nm}_le{k}"] = boot_rate(f, [x[nm] <= k and x["flip"] for x in f], fl)
    M["neigh"] = boot_rate(f, [x["J"] > 1 and x["Q"] == 1 for x in f], one)
    M["neigh_rev"] = boot_rate(f, [x["J"] == 1 and x["Q"] > 1 for x in f], one)
    M["neigh4"] = boot_rate(f, [x["J"] > 4 and x["Q"] <= 4 for x in f], one)
    M["neigh4_rev"] = boot_rate(f, [x["J"] <= 4 and x["Q"] > 4 for x in f], one)
    M["Q_better"] = boot_rate(f, [x["Q"] < x["J"] for x in f], one)
    M["tie"] = boot_rate(f, [x["Q"] == x["J"] for x in f], one)
    M["ratio"] = boot_median(f, [x["Q"] / x["J"] for x in f])
    M["J_rank"] = boot_median(f, [x["J"] for x in f])
    M["Q_rank"] = boot_median(f, [x["Q"] for x in f])
    for nm, key in (("J", "oJ"), ("Q", "oQ")):
        M[f"old_{nm}_le4"] = boot_rate(f, [x[key] <= 4 for x in f], one)
        M[f"old_{nm}_le600"] = boot_rate(f, [x[key] <= 600 for x in f], one)
    M["cosJQ"] = boot_median(f, [x["cosJQ"] for x in f])
    M["overshoot"] = boot_rate(f, [x["flip"] and x["lamJQ"] > x["lam"] for x in f], fl)
    M["dose_ratio"] = boot_median(f, [(min(x["lamJQ"], 1e9) / x["lam"]) if x["flip"] else np.nan for x in f], fl)  # no J+Q root -> 1e9
    M["cosPA"] = boot_median(f, [x["cosPA"] for x in f], fl)
    M["cosJA"] = boot_median(f, [x["cosJA"] for x in f], fl)
    M["relP"] = boot_median(f, [x["relP"] for x in f], fl)
    M["relJ"] = boot_median(f, [x["relJ"] for x in f], fl)
    M["P_beats_J_cos"] = boot_rate(f, [x["flip"] and x["cosPA"] > x["cosJA"] for x in f], fl)
    return M


def R(m, pct=True):  # rate with CI
    if m.get("rate") is None:
        return "–"
    return f"{100 * m['rate']:.0f}% ({100 * m['lo']:.0f}–{100 * m['hi']:.0f}) [{m['k']}/{m['n']}]"


def Rs(m):  # short
    return "–" if m.get("rate") is None else f"{100 * m['rate']:.0f} ({100 * m['lo']:.0f}–{100 * m['hi']:.0f})"


def Md(m, d=2):
    if m.get("med") is None:
        return "–"
    return f"{m['med']:.{d}f} ({m['lo']:.{d}f}–{m['hi']:.{d}f}) IQR {m['q25']:.{d}f}–{m['q75']:.{d}f}"


# ---------------------------------------------------------------- compute
RES = {"jsonl": A.jsonl, "B": A.B, "seed": A.seed, "clusters": CL, "layers": LAYERS, "by": {}}
for match in ("any", "strict"):
    for l in LAYERS:
        rl = [r for r in rows if int(r["layer"]) == l]
        for sub in ["all"] + FUNCS:
            rs = rl if sub == "all" else [r for r in rl if r["func"] == sub]
            if rs:
                RES["by"][f"{match}|{l}|{sub}"] = metrics([feats(r, match) for r in rs])
G = lambda match, l, sub="all": RES["by"].get(f"{match}|{l}|{sub}")  # noqa: E731

# ---------------------------------------------------------------- gates
LIM = {"J_vs_fd": 2e-2, "Q_vs_fd": 1e-1, "g1_vs_s2": 2e-2, "g2_vs_s2": 1e-1, "z0_vs_hf": 1e-3, "r0_vs_s2": 1e-3}
gmax, gfail, nfd = {}, [], 0
for r in rows:
    g = r["e"]["gates"]
    nfd += "J_vs_fd" in g
    for k, v in g.items():
        gmax[k] = max(gmax.get(k, 0.0), abs(v))
        if k in LIM and not abs(v) < LIM[k] and (k != "r0_vs_s2" or "g1_vs_s2" in g):  # r0 gated only with S2 gate on
            gfail.append((r["func"], r["src"], r["tgt"], r["layer"], k, v))
ovl = [min(r["e"]["fd_topk_overlap_halfQ"].values()) for r in rows if r["e"].get("fd_topk_overlap_halfQ")]
RES["gates"] = {"max": gmax, "fail": gfail, "n_cases": len(rows), "n_fd": nfd,
                "fd_trials": sorted({(r["idx"], r["func"], r["src"], r["tgt"]) for r in rows if r.get("fd")}),
                "fd_top10_overlap_min": min(ovl) if ovl else None}

say("# Swap token readout as rates (tables generated by src/swap_tokens_rates.py)")
say()
say(f"Source: `{A.jsonl}`. {len(rows)} (swap, layer) cases, {len({(r['func'], r['src'], r['tgt']) for r in rows})} swaps, "
    f"layers {LAYERS}, {len(CL)} clusters (unordered country pairs). CIs: 95% cluster bootstrap, B={A.B}, seed {A.seed}.")
say()
say(f"**Gates.** {len(rows)} cases; FD gate on {nfd} cases ({len(RES['gates']['fd_trials'])} trials: canonical indices "
    f"{[t[0] for t in RES['gates']['fd_trials']]}). Max over cases: " + ", ".join(f"{k} {v:.1e}" for k, v in sorted(gmax.items()))
    + f". FD top-10 ½Q overlap min {RES['gates']['fd_top10_overlap_min']}. Failures: {len(gfail)} {gfail[:5]}")
say()

# v1 reproduction check
if A.v1:
    v1 = json.loads(Path(A.v1).read_text())
    byk = {(r["func"], r["src"], r["tgt"], int(r["layer"])): r for r in rows}
    chk = []
    for t in v1["trials"]:
        for l, e in t["layers"].items():
            r = byk.get((t["func"], t["src"], t["tgt"], int(l)))
            if r is None:
                continue
            a = [e["ranks"][x]["new"]["up"] for x in ("J", "halfQ", "pred", "actual")]
            b = [r["e"]["ranks"][x]["new"]["up"] for x in ("J", "halfQ", "pred", "actual")]
            chk.append((t["func"], l, a, b, a == b))
    RES["v1_check"] = chk
    say(f"**v1 reproduction** (strict target ranks J/½Q/pred/actual, v1 vs v2): "
        f"{sum(c[-1] for c in chk)}/{len(chk)} identical. " + "; ".join(f"{c[0]} L{c[1]} {c[2]} vs {c[3]}" for c in chk if not c[-1]))
    say()

for match in ("any", "strict"):
    say(f"## Matching: {match}" + (" (primary: any first-token variant of new / old, S2's rule)" if match == "any"
                                   else " (single id tid(new[0]), as v1 and the paper's table)"))
    say()
    say("### Target rank among raised tokens, all relations (rate, 95% CI, [k/n])")
    say("| L | n | n_flip | ½Q ≤4 | J ≤4 | ½Q =1 | J =1 | ½Q ≤10 | J ≤10 | pred ≤4 (flip) | actual ≤4 (flip) | actual ≤2 (flip) |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in LAYERS:
        M = G(match, l)
        say(f"| {l}{'*' if l in PRIMARY else ''} | {M['n']} | {M['n_flip']} | {R(M['Q_le4'])} | {R(M['J_le4'])} | "
            f"{Rs(M['Q_le1'])} | {Rs(M['J_le1'])} | {Rs(M['Q_le10'])} | {Rs(M['J_le10'])} | {Rs(M['P_le4'])} | "
            f"{Rs(M['Act_le4'])} | {Rs(M['Act_le2'])} |")
    say()
    say("### Neighbour pattern and paired comparison, all relations")
    say("| L | J top-1 not target, ½Q top-1 is | reverse | J >4, ½Q ≤4 | reverse | ½Q better than J | tie | "
        "median rank ½Q/J | median rank J | median rank ½Q |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for l in LAYERS:
        M = G(match, l)
        say(f"| {l}{'*' if l in PRIMARY else ''} | {R(M['neigh'])} | {Rs(M['neigh_rev'])} | {Rs(M['neigh4'])} | "
            f"{Rs(M['neigh4_rev'])} | {Rs(M['Q_better'])} | {Rs(M['tie'])} | {Md(M['ratio'], 3)} | "
            f"{M['J_rank']['med']:.0f} | {M['Q_rank']['med']:.0f} |")
    say()
    say("### Source answer among the most lowered; cos(J, ½Q); doses and prediction quality, all relations")
    say("| L | old in J lowered top-4 | old in ½Q lowered top-4 | old in J top-600 | old in ½Q top-600 | cos(J,½Q) median (CI) IQR |"
        " J+Q dose overshoots λ* (flip) | median λ_JQ/λ* | cos(pred,act) | cos(λ*J,act) | pred cos beats J | rel err pred |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in LAYERS:
        M = G(match, l)
        say(f"| {l}{'*' if l in PRIMARY else ''} | {Rs(M['old_J_le4'])} | {Rs(M['old_Q_le4'])} | {Rs(M['old_J_le600'])} | "
            f"{Rs(M['old_Q_le600'])} | {Md(M['cosJQ'])} | {Rs(M['overshoot'])} | {Md(M['dose_ratio'])} | "
            f"{Md(M['cosPA'])} | {Md(M['cosJA'])} | {Rs(M['P_beats_J_cos'])} | {Md(M['relP'])} |")
    say()
    say("### Per relation, primary layers")
    say("| relation | L | n | n_flip | ½Q ≤4 | J ≤4 | ½Q =1 | J =1 | neighbour (J≠, ½Q=) | reverse | ½Q better | "
        "old in ½Q low top-4 | old in J low top-4 | cos(J,½Q) med | pred ≤4 | actual ≤4 |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sub in FUNCS:
        for l in [x for x in PRIMARY if x in LAYERS]:
            M = G(match, l, sub)
            if not M:
                continue
            say(f"| {sub} | {l} | {M['n']} | {M['n_flip']} | {Rs(M['Q_le4'])} | {Rs(M['J_le4'])} | {Rs(M['Q_le1'])} | "
                f"{Rs(M['J_le1'])} | {Rs(M['neigh'])} | {Rs(M['neigh_rev'])} | {Rs(M['Q_better'])} | {Rs(M['old_Q_le4'])} | "
                f"{Rs(M['old_J_le4'])} | {M['cosJQ']['med']:.2f} | {Rs(M['P_le4'])} | {Rs(M['Act_le4'])} |")
    say()
    if len(LAYERS) > len(PRIMARY):
        say("### Rate vs layer per relation: ½Q ≤4 / J ≤4 (%)")
        say("| relation | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
        say("|---|" + "---|" * len(LAYERS))
        for sub in FUNCS:
            say(f"| {sub} | " + " | ".join(
                (f"{100 * G(match, l, sub)['Q_le4']['rate']:.0f} / {100 * G(match, l, sub)['J_le4']['rate']:.0f}"
                 if G(match, l, sub) else "–") for l in LAYERS) + " |")
        say()

jo = Path(A.json_out) if A.json_out else Path(A.jsonl).with_name("swap_tokens_rates.json")
jo.write_text(json.dumps(RES, default=str, indent=0))
say(f"Numbers: `{jo}`.")
if A.md:
    Path(A.md).write_text("\n".join(OUT) + "\n")
