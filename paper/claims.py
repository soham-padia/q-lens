#!/usr/bin/env python3
"""Phase 1 (W3 + W2): read every headline number C1-C5 from the raw outputs, with bootstrap CIs.

The CIs are two-way bootstrap intervals that resample contexts AND directions with replacement (B = 2000, fixed
seed); the statistic is the median over rows unless stated otherwise. CPU only, no model is loaded.

Writes:
  results/claims.csv           one row per number: claim, model, setting, quantity, value, lo, hi, n, file
  results/qj_curves.csv        |Q|/|J| against blocks traversed k, every model and source (figure F2, explorable E3)
  results/explorable_data/*.json data for the explorables E2-E5
Added 2026-11-21 (cold-read round 1): per-context spreads (C1s), within-push-block depth fits and r at one block
(C2), per-case IQR of the cancellation ratio (C4), and the swap numbers from S1/S2 (S: AUCs, baselines, dose errors
in grid terms). The new bootstraps use their own generator (RNG2), so every earlier row is unchanged.
Added 2026-11-22 (v8.4): the Lean pilot, the responsibility maps, the FD precision check, the cost curve and the
numeric-intermediate control, in claims_v84.py with their own generator (RNG3), read from results/step5/.
Run from anywhere:  QLENS_OUT=<outputs dir> python paper/claims.py
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
P = Path(__file__).resolve().parents[1] / "results"
B = 2000
RNG = np.random.default_rng(20261120)
RNG2 = np.random.default_rng(20261121)   # new rows only; keeps RNG's sequence (and every older CI) unchanged
QWEN = [("Qwen3-8B", "so_qwen3-8b"), ("Qwen3-14B", "so_qwen3-14b"), ("Qwen3-32B", "so_qwen3-32b"),
        ("Qwen3.8-27B", "second_order_v1")]
LOOPS = [("Ouro-1.4B", "loop_ouro14"), ("Ouro-2.6B", "loop_ouro26"), ("Huginn-3.5B", "loop_huginn")]
ROWS, CHECKS = [], []


def idx(C, D):
    return [(RNG.integers(0, C, C), RNG.integers(0, D, D)) for _ in range(B)]


def idx2(C, D):
    return [(RNG2.integers(0, C, C), RNG2.integers(0, D, D)) for _ in range(B)]


def boot(stat, arrays, ix):
    """stat(*arrays), each array shaped (C, D, ...). Returns value, 2.5% and 97.5% bootstrap percentiles."""
    v = stat(*arrays)
    bs = np.array([stat(*[a[ci][:, di] for a in arrays]) for ci, di in ix])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return v, lo, hi


def med(x):
    return float(np.median(x))


def add(claim, model, setting, quantity, vlh, n, f):
    v, lo, hi = vlh
    ROWS.append(dict(claim=claim, model=model, setting=setting, quantity=quantity, value=float(v), lo=float(lo),
                     hi=float(hi), n=n, file=f))
    return float(v)


def grid(rows, C, D, get):
    """A2/A3 rows are context-major (q_anatomy.py loops contexts outside directions)."""
    return np.array([get(r) for r in rows], dtype=float).reshape(C, D)


# ---------------------------------------------------------------- C1 + C2: D1 (Taylor error, |Q|/|J| against k)
curves = []          # (model, family, src, tn, k, med, lo, hi)
e2 = {}
fits = {}
for name, d in QWEN:
    js = json.loads((W / d / "D1.json").read_text())
    raw = torch.load(W / d / "D1_raw.pt", map_location="cpu", weights_only=False)
    al, nl, f = js["alphas"], js["n_layers"], f"{d}/D1_raw.pt"
    srcs = sorted(raw.keys())
    C, D = js["n_ctx"], js["n_dir"]
    ix = idx(C, D)
    e2[name] = {"alphas": al, "n_layers": nl, "sources": {}}
    for tn in ("iso", "data"):
        pairs = []      # (k, qj grid) for the slope fit
        psrc = []       # push block of each pair (for the within-push-block fits)
        for s in srcs:
            R = raw[s][tn]
            tg = raw[s]["targets"]
            fi = tg.index(nl - 1)
            rl, rq = R["rel_lin"].numpy(), R["rel_quad"].numpy()
            qj = (R["norm_Q"] / R["norm_J"]).numpy()
            # consistency with the JSON medians (torch takes the lower middle value, numpy the mean of the two)
            jsm = js["sources"][str(s)]["summary"][tn][str(nl - 1)]
            for a_i in range(len(al)):
                x = np.sort(rl[:, :, a_i, fi].ravel())
                CHECKS.append(abs(x[(len(x) - 1) // 2] - jsm["rel_lin"][a_i]) / max(1e-9, abs(jsm["rel_lin"][a_i])))
            ent = e2[name]["sources"].setdefault(str(s), {"k_final": nl - 1 - s})
            for a_i, a in enumerate(al):
                lin = boot(lambda x: med(x), [rl[:, :, a_i, fi]], ix)
                quad = boot(lambda x: med(x), [rq[:, :, a_i, fi]], ix)
                gain = boot(lambda x, y: med(x) / med(y), [rl[:, :, a_i, fi], rq[:, :, a_i, fi]], ix)
                ent.setdefault(tn, {}).setdefault("rel_lin", []).append([round(z, 4) for z in lin])
                ent[tn].setdefault("rel_quad", []).append([round(z, 4) for z in quad])
                if name == "Qwen3.8-27B" or a == 0.1:
                    st = f"src={s} k={nl - 1 - s} alpha={a} {tn}"
                    add("C1", name, st, "rel_err_J", lin, C * D, f)
                    add("C1", name, st, "rel_err_JQ", quad, C * D, f)
                    add("C1", name, st, "gain_J_over_JQ", gain, C * D, f)
                # C1s: spread over contexts (per-context medians over the 8 directions) and over cases
                x_l, x_q = rl[:, :, a_i, fi], rq[:, :, a_i, fi]
                pcl, pcg = np.median(x_l, 1), np.median(x_l, 1) / np.median(x_q, 1)
                st = f"src={s} k={nl - 1 - s} alpha={a} {tn}"
                ROWS.append(dict(claim="C1s", model=name, setting=st, quantity="rel_err_J ctx range", value=med(x_l),
                                 lo=float(pcl.min()), hi=float(pcl.max()), n=C, file=f))
                ROWS.append(dict(claim="C1s", model=name, setting=st, quantity="gain ctx range",
                                 value=med(x_l) / med(x_q), lo=float(pcg.min()), hi=float(pcg.max()), n=C, file=f))
                ROWS.append(dict(claim="C1s", model=name, setting=st, quantity="n ctx gain<1",
                                 value=float((pcg < 1).sum()), lo=float(np.argmin(pcg)), hi=float("nan"), n=C, file=f))
                ROWS.append(dict(claim="C1s", model=name, setting=st, quantity="rel_err_J case IQR", value=med(x_l),
                                 lo=float(np.percentile(x_l, 25)), hi=float(np.percentile(x_l, 75)), n=C * D, file=f))
            for t_i, t in enumerate(tg):
                k = t - s
                g = qj[:, :, t_i]
                v = boot(lambda x: med(x), [g], ix)
                curves.append((name, "qwen", s, tn, k, *v))
                pairs.append((k, g))
                psrc.append(s)
                if k == 1:
                    ROWS.append(dict(claim="C2", model=name, setting=f"src={s} {tn}", quantity="Q/J at k=1",
                                     value=v[0], lo=v[1], hi=v[2], n=C * D, file=f"{d}/D1_raw.pt"))
        # one per-block rate: OLS slope of log(median Q/J) on k over every (source, target) pair, paired bootstrap
        ks = np.array([p[0] for p in pairs], float)
        G = np.stack([p[1] for p in pairs], -1)          # (C, D, npairs)

        def slope(g, ks=ks):
            y = np.log(np.median(g.reshape(-1, g.shape[-1]), 0))
            return np.polyfit(ks, y, 1)[0]

        def fitall(g, ks=ks):
            y = np.log(np.median(g.reshape(-1, g.shape[-1]), 0))
            return np.polyfit(ks, y, 1)

        v = boot(slope, [G], ix)
        add("C2", name, f"{tn} all (src,target) pairs", "dlog(Q/J)/dk", v, C * D, f"{d}/D1_raw.pt")
        b, a0 = fitall(G)
        resid = np.log(np.median(G.reshape(-1, G.shape[-1]), 0)) - (a0 + b * ks)
        fits[(name, tn)] = (b, a0, float(np.sqrt(np.mean(resid ** 2))), len(ks))
        ROWS.append(dict(claim="C2", model=name, setting=f"{tn} fit", quantity="intercept_log(Q/J)_and_rms_resid",
                         value=float(a0), lo=float(np.sqrt(np.mean(resid ** 2))), hi=float("nan"), n=len(ks),
                         file=f"{d}/D1_raw.pt"))
        # Within one push block: a separate intercept per push block (k and the push block move together in the
        # pooled fit). Exponential (log r linear in k) and power law (log r linear in log k), and the exponential
        # fitted only on k >= 8, each with the same paired bootstrap; rms log residuals for the first two.
        sv = np.array(psrc)
        Dm = np.stack([(sv == u).astype(float) for u in sorted(set(psrc))], 1)

        def fe(g, x, keep=None, Dm=Dm):
            y = np.log(np.median(g.reshape(-1, g.shape[-1]), 0))
            X = np.c_[Dm, x]
            if keep is not None:
                X, y = X[keep], y[keep]
                X = X[:, np.r_[X[:, :-1].sum(0) > 1, True]]   # drop push blocks with < 2 points left
            b_ = np.linalg.lstsq(X, y, rcond=None)[0]
            return b_[-1], float(np.sqrt(np.mean((y - X @ b_) ** 2)))

        k8 = ks >= 8
        ix2 = idx2(C, D)
        for qn, x, keep in (("dlog(Q/J)/dk within push block", ks, None), ("dlog(Q/J)/dlogk within push block", np.log(ks), None),
                            ("dlog(Q/J)/dk within push block, k>=8", ks, k8)):
            v = boot(lambda g, x=x, keep=keep: fe(g, x, keep)[0], [G], ix2)
            ROWS.append(dict(claim="C2", model=name, setting=f"{tn} all (src,target) pairs", quantity=qn, value=v[0],
                             lo=v[1], hi=v[2], n=C * D, file=f"{d}/D1_raw.pt"))
            ROWS.append(dict(claim="C2", model=name, setting=f"{tn} all (src,target) pairs", quantity=qn + " rms resid",
                             value=fe(G, x, keep)[1], lo=float("nan"), hi=float("nan"), n=int(len(ks) if keep is None else keep.sum()),
                             file=f"{d}/D1_raw.pt"))

# looped models (L1a): the same metrics on the explicit loop forward
for name, d in LOOPS:
    raw = torch.load(W / d / "L1a_raw.pt", map_location="cpu", weights_only=False)
    for s, v in raw.items():
        for tn in ("iso", "data"):
            qj = (v[tn]["norm_Q"] / v[tn]["norm_J"]).numpy()
            C, D = qj.shape[:2]
            ix = idx(C, D)
            for t_i, (k, t) in enumerate(zip(v["k"], v["targets"])):
                if str(t).startswith(("out", "coda")):
                    continue
                curves.append((name, "looped", s, tn, int(k), *boot(lambda x: med(x), [qj[:, :, t_i]], ix)))

with open(P / "qj_curves.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "family", "src", "type", "k", "qj_median", "lo", "hi"])
    for c in curves:
        w.writerow([c[0], c[1], c[2], c[3], c[4]] + [f"{z:.5g}" for z in c[5:]])


# C3: looped vs Qwen at matched k (Qwen reference = per model, median of pair medians with |k - k0| <= 3)
def qwen_ref(k0, tn):
    out = []
    for name, _ in QWEN:
        vals = [c[5] for c in curves if c[0] == name and c[3] == tn and abs(c[4] - k0) <= 3]
        if vals:
            out.append(float(np.median(vals)))
    return out


for name, _ in LOOPS:
    for tn in ("iso", "data"):
        for k0 in (1, 4, 8, 24, 48, 96):
            vals = [c for c in curves if c[0] == name and c[3] == tn and c[4] == k0]
            ref = qwen_ref(k0, tn)
            if not vals or not ref:
                continue
            m = float(np.median([c[5] for c in vals]))
            ROWS.append(dict(claim="C3", model=name, setting=f"{tn} k={k0}", quantity="Q/J looped (median over srcs)",
                             value=m, lo=min(c[6] for c in vals), hi=max(c[7] for c in vals), n=len(vals),
                             file="loop_*/L1a_raw.pt"))
            ROWS.append(dict(claim="C3", model=name, setting=f"{tn} k={k0}", quantity="Qwen band / looped",
                             value=float(np.median(ref)) / m, lo=min(ref) / m, hi=max(ref) / m, n=len(ref),
                             file="so_*/D1_raw.pt"))

# ---------------------------------------------------------------- C4: A2 anatomy of Q (exact split)
e4, e5 = {}, {}
# fp32 files win when present (the TF32 originals fail the 1e-2 gate on some rows)
A2 = [("Qwen3.8-27B", "qanat_q38-27b", ["A2_a32.json", "A2_b32.json"]),
      ("Qwen3-8B", "qanat_qwen3-8b", ["A2_32.json"] if (W / "qanat_qwen3-8b" / "A2_32.json").exists() else ["A2.json"])]
A3_FILES = {"Qwen3.8-27B": ("qanat_q38-27b", [f"A3_s{s}_32.json" for s in (8, 24, 40)]
                            if all((W / "qanat_q38-27b" / f"A3_s{s}_32.json").exists() for s in (8, 24, 40))
                            else ["A3.json"]),
            "Qwen3-8B": ("qanat_qwen3-8b", ["A3.json"])}
USED = {k: v[1] for k, v in A3_FILES.items()}
for name, d, files in A2:
    for fn in files:
        js = json.loads((W / d / fn).read_text())
        C, D, f = js["n_ctx"], js["n_dir"], f"{d}/{fn}"
        for s, v in js["sources"].items():
            for tn in ("iso", "data"):
                rows = v[tn]["rows"]
                if len(rows) != C * D:
                    continue
                ix = idx(C, D)
                st = f"src={s} blocks={v['blocks'] if not isinstance(v['blocks'], list) else len(v['blocks'])} {tn}"
                g = lambda fn_: grid(rows, C, D, fn_)
                gate = max(max(r["site"]["gate"], r["block"]["gate"]) for r in rows)
                ROWS.append(dict(claim="C4", model=name, setting=st, quantity="max_gate", value=gate, lo=gate, hi=gate,
                                 n=C * D, file=f))
                ci_b, pr_b = g(lambda r: r["block"]["ci"]), g(lambda r: r["block"]["pr"])
                ci_s, pr_s = g(lambda r: r["site"]["ci"]), g(lambda r: r["site"]["pr"])
                add("C4", name, st, "block_cancellation_ci", boot(med, [ci_b], ix), C * D, f)
                add("C4", name, st, "block_sqrt_pr", boot(lambda x: med(np.sqrt(x)), [pr_b], ix), C * D, f)
                add("C4", name, st, "block_ci_over_sqrt_pr", boot(lambda x, y: med(x / np.sqrt(y)), [ci_b, pr_b], ix),
                    C * D, f)
                rat = (ci_b / np.sqrt(pr_b)).ravel()      # = sqrt(sum_b |c_b|^2) / |Q| per case
                ROWS.append(dict(claim="C4", model=name, setting=st, quantity="block_ci_over_sqrt_pr case IQR",
                                 value=med(rat), lo=float(np.percentile(rat, 25)), hi=float(np.percentile(rat, 75)),
                                 n=C * D, file=f))
                add("C4", name, st, "site_cancellation_ci", boot(med, [ci_s], ix), C * D, f)
                add("C4", name, st, "site_sqrt_pr", boot(lambda x: med(np.sqrt(x)), [pr_s], ix), C * D, f)
                add("C4", name, st, "block_n90", boot(med, [g(lambda r: r["block"]["n90"])], ix), C * D, f)
                add("C4", name, st, "site_n90", boot(med, [g(lambda r: r["site"]["n90"])], ix), C * D, f)
                add("C4", name, st, "best_single_block_err", boot(med, [g(lambda r: r["block"]["best_single_err"])], ix),
                    C * D, f)
                add("C4", name, st, "best_single_site_err", boot(med, [g(lambda r: r["site"]["best_single_err"])], ix),
                    C * D, f)
                # largest single piece. Vector: |c_top|/|Q| = top1_norm_share * ci (block level).
                add("C4", name, st, "top_block_norm_over_Q",
                    boot(med, [g(lambda r: r["block"]["top1_norm_share"] * r["block"]["ci"])], ix), C * D, f)
                # Scalar (along q_hat = Q/|Q|): |s_top|/|Q| = top_share * ci_scal * total_level / |Q|
                for lvl in ("scal_block", "scal_site", "scal_heads", "scal_neurons", "scal_fine"):
                    sh = g(lambda r: r[lvl]["top_share"]["1"] * r[lvl]["ci"] * abs(r[lvl]["total"])
                           / abs(r["scal_block"]["total"]))
                    add("C4", name, st, f"top_{lvl[5:]}_along_Q", boot(med, [sh], ix), C * D, f)
                    add("C4", name, st, f"{lvl}_n90", boot(med, [g(lambda r: r[lvl]["n90"])], ix), C * D, f)
                for ty in ("n1", "mix", "n2", "mlp"):
                    add("C4", name, st, f"type_share_{ty}", boot(med, [g(lambda r: r["type_share"][ty])], ix), C * D, f)
                # explorables E4 (block lengths, push at 8) and E5 (reconstruction curves)
                bn = np.array([r["block_norm"] for r in rows], float)
                bs = np.array([r["block_share"] for r in rows], float)
                CHECKS.append(float(np.max(np.abs(bs.sum(1) - 1))))
                CHECKS.append(float(np.max(np.abs(bn.sum(1) / ci_b.ravel() - 1))))
                e4.setdefault(name, {})[f"{s}/{tn}"] = {
                    "block_norm_over_Q_median": np.round(np.median(bn, 0), 5).tolist(),
                    "block_share_median": np.round(np.median(bs, 0), 5).tolist(),
                    "ci_median": round(med(ci_b), 3), "sqrt_pr_median": round(med(np.sqrt(pr_b)), 3)}
                cur = {}
                for lvl in ("block", "site"):
                    cur[lvl] = {"n": list(range(1, 11)),
                                "err": np.round(np.median([r[lvl]["err_top"] for r in rows], 0), 4).tolist(),
                                "n50": med(g(lambda r: r[lvl]["n50"])), "n90": med(g(lambda r: r[lvl]["n90"]))}
                for lvl in ("scal_block", "scal_site", "scal_heads", "scal_neurons", "scal_fine"):
                    ns = sorted(rows[0][lvl]["err_top"].keys(), key=int)
                    cur[lvl] = {"n": [int(n) for n in ns], "N": rows[0][lvl]["N"],
                                "err": [round(med(g(lambda r: r[lvl]["err_top"][n])), 4) for n in ns],
                                "n50": med(g(lambda r: r[lvl]["n50"])), "n90": med(g(lambda r: r[lvl]["n90"]))}
                e5.setdefault(name, {})[f"{s}/{tn}"] = cur

# ---------------------------------------------------------------- C5: A3 third order (self / within / between)
pooled = {}
for name, (d, a3files) in A3_FILES.items():
  allc = []
  for a3f in a3files:
    js = json.loads((W / d / a3f).read_text())
    C, D, f = js["n_ctx"], js["n_dir"], f"{d}/{a3f}"
    for s, v in js["sources"].items():
        for tn in ("iso", "data"):
            rows = v[tn]["rows"]
            ix = idx(C, D)
            st = f"src={s} {tn} (n_ctx={C})"
            g = lambda fn_: grid(rows, C, D, fn_)
            gate = max(max(r["gate"], r["pair"]["gate"]) for r in rows)
            ROWS.append(dict(claim="C5", model=name, setting=st, quantity="max_gate", value=gate, lo=gate, hi=gate,
                             n=C * D, file=f))
            for q, fn_ in (("self", lambda r: r["pair"]["self_share_t"]),
                           ("within_block", lambda r: r["pair"]["within_block_share"]),
                           ("between_block", lambda r: r["pair"]["cross_jk_share"])):
                x = g(fn_)
                add("C5", name, st, f"{q}_share_median", boot(med, [x], ix), C * D, f)
                add("C5", name, st, f"{q}_share_mean", boot(lambda y: float(np.mean(y)), [x], ix), C * D, f)
            allc += [r["pair"]["cross_jk_share"] for r in rows]
            add("C5", name, st, "T_over_Q", boot(med, [g(lambda r: r["T_over_Q"])], ix), C * D, f)
            for a in js["alphas"]:
                for pred in ("J", "JQ", "JQT", "JQT_self"):
                    add("C5", name, f"{st} alpha={a}", f"large_step_err_{pred}",
                        boot(med, [g(lambda r: r["large_step"][str(a)][pred][0])], ix), C * D, f)
  allc = np.array(allc)
  pooled[name] = (float(np.median(allc)), float(np.mean(allc > 0.5)), len(allc))
  ROWS.append(dict(claim="C5", model=name, setting="pooled all sources x types", quantity="between_block_median",
                   value=pooled[name][0], lo=float("nan"), hi=float("nan"), n=len(allc), file=f))
  ROWS.append(dict(claim="C5", model=name, setting="pooled all sources x types", quantity="frac_rows_between>0.5",
                   value=pooled[name][1], lo=float("nan"), hi=float("nan"), n=len(allc), file=f))

# ---------------------------------------------------------------- S: concept swaps (S1 one-hop, S2 two-hop)
# S1 (second_order_v1/S1.json): one-hop trials, single-layer swaps; g1, g2 are derivatives of the PRE-final-norm gap
# (gamma * (u_new - u_old)) . h_L,last, so they cannot be added to the clean gap (true logits); flip label = top-1 at
# lambda = 1. S2 (second_order_v1/s2/*/S2.json): two-hop trials; r(lambda) and its jets g1, g2 are the TRUE logit gap
# at the last position (final norm included), so g0 + g1 + g2/2 is the predicted gap at lambda = 1.
# Dose errors follow src/summarize_s2.py exactly (upper median x[n // 2], smallest positive root of the J+Q quadratic).
S_DIR = W / "second_order_v1"


def auc(sc, y):
    sc, y = np.asarray(sc, float), np.asarray(y, bool)
    p_, n_ = sc[y], sc[~y]
    if len(p_) == 0 or len(n_) == 0:
        return float("nan")
    return float(((p_[:, None] > n_[None, :]).sum() + 0.5 * (p_[:, None] == n_[None, :]).sum()) / (len(p_) * len(n_)))


def auc_rows(tag, units, scores, y, f, groups):
    """scores: {name: array}; units: trial index per row (bootstrap resamples trials; groups: cluster id per trial)."""
    units, y = np.asarray(units), np.asarray(y, bool)
    for nm, sc in scores.items():
        ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=tag, quantity=f"AUC {nm}", value=auc(sc, y),
                         lo=float("nan"), hi=float("nan"), n=int(len(y)), file=f))
    ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=tag, quantity="flips / cases", value=float(y.sum()),
                     lo=float(len(y)), hi=float("nan"), n=int(len(y)), file=f))
    ut = np.unique(units)
    rows_of = {u: np.flatnonzero(units == u) for u in ut}
    gr = np.asarray([groups[u] for u in ut])
    for a_, b_ in (("g2", "g1"), ("gap JQ at 1", "gap J at 1")):
        if a_ not in scores or b_ not in scores:
            continue
        diff = auc(scores[a_], y) - auc(scores[b_], y)
        for how in ("trials", "clusters"):
            bs = []
            for _ in range(B):
                if how == "trials":
                    pick = RNG2.choice(ut, len(ut))
                else:
                    cl = np.unique(gr)
                    pick = np.concatenate([ut[gr == c] for c in RNG2.choice(cl, len(cl))])
                ii = np.concatenate([rows_of[u] for u in pick])
                bs.append(auc(scores[a_][ii], y[ii]) - auc(scores[b_][ii], y[ii]))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=tag, quantity=f"AUC {a_} - {b_} (boot {how})",
                             value=diff, lo=float(lo), hi=float(hi), n=int(len(ut)), file=f))


s1 = json.loads((S_DIR / "S1.json").read_text())
i1 = s1["lams"].index(1.0)
for tag, cats, layers in (("onehop all categories L36-48", None, (36, 40, 48)),
                          ("onehop countries all layers", ("countries",), (24, 36, 40, 48, 56)),
                          ("onehop countries L36-48", ("countries",), (36, 40, 48)),
                          ("onehop animals L36-48", ("animals",), (36, 40, 48)),
                          ("onehop months L36-48", ("months",), (36, 40, 48)),
                          ("onehop numbers L36-48", ("numbers",), (36, 40, 48))):
    U_, G1, G2, G0, Y, ACT, grp = [], [], [], [], [], [], {}
    for ti, t in enumerate(s1["trials"]):
        if cats and t["cat"] not in cats:
            continue
        grp[ti] = f"{t['cat']}:{t['src']}"
        for l in layers:
            e = t["layers"][str(l)]
            U_.append(ti); G1.append(e["g1"][0]); G2.append(e["g2"][0]); G0.append(t["clean_gap"])
            Y.append(e["success"][i1]); ACT.append(e["act"][i1][0])
    G1, G2 = np.array(G1), np.array(G2)
    auc_rows(tag, U_, {"g1": G1, "g2": G2, "g1+g2/2": G1 + 0.5 * G2, "clean gap g0": np.array(G0),
                       "actual change": np.array(ACT)}, Y, "second_order_v1/S1.json", grp)

S2 = {}
for fp in sorted((S_DIR / "s2").glob("*/S2.json")):
    d_ = json.loads(fp.read_text())
    for k_, v_ in d_.items():
        if k_ in ("onehop", "twohop") and k_ in S2:
            S2[k_] += v_
        elif k_ != "secs":
            S2.setdefault(k_, v_)
ok2 = [r for r in S2["twohop"] if "single" in r]
LAM2, BLAM2, L2 = S2["lams"], S2["band_lams"], S2["layers"]
li2 = LAM2.index(1.0)
f2 = "second_order_v1/s2/*/S2.json"
ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting="twohop", quantity="trials answered correctly", value=len(ok2),
                 lo=len(S2["twohop"]), hi=float("nan"), n=len(S2["twohop"]), file=f2))
for tag, layers in (("twohop single layers L36-48", (36, 40, 44, 48)), ("twohop single layers all", tuple(L2)),
                    ("twohop single layer L40", (40,))):
    U_, sc, Y, grp = [], {k_: [] for k_ in ("clean gap g0", "g1", "g2", "gap J at 1", "gap JQ at 1", "actual gap")}, [], {}
    for i, r in enumerate(ok2):
        grp[i] = r["src"]
        for l in layers:
            e = r["single"][str(l)]
            g0_, g1_, g2_ = r["clean"]["r"], e["g1"], e["g2"]
            U_.append(i); Y.append(e["out"][li2]["succ"])
            for k_, v_ in (("clean gap g0", g0_), ("g1", g1_), ("g2", g2_), ("gap J at 1", g0_ + g1_),
                           ("gap JQ at 1", g0_ + g1_ + 0.5 * g2_), ("actual gap", e["out"][li2]["r"])):
                sc[k_].append(v_)
    auc_rows(tag, U_, {k_: np.array(v_) for k_, v_ in sc.items()}, Y, f2, grp)


def flip_actual(r0, lams, rs_):
    prev_l, prev_r = 0.0, r0
    for lam, rv in zip(lams, rs_):
        if rv > 0:
            return (prev_l + (lam - prev_l) * (-prev_r) / (rv - prev_r) if rv != prev_r else lam), prev_l, lam
        prev_l, prev_r = lam, rv
    return math.inf, None, None


def flip_J(r0, g1):
    return -r0 / g1 if g1 > 0 else math.inf


def flip_JQ(r0, g1, g2):
    if abs(g2) < 1e-9:
        return flip_J(r0, g1)
    disc = g1 * g1 - 2 * g2 * r0
    if disc < 0:
        return math.inf
    pos = [x for x in ((-g1 + sg * math.sqrt(disc)) / g2 for sg in (1, -1)) if x > 0]
    return min(pos) if pos else math.inf


def umed(x):
    x = sorted(x)
    return x[len(x) // 2]


dose = {"all": [], "single layers": [], "bands": []}
for r in ok2:
    r0 = r["clean"]["r"]
    cs = [("single layers", f"layer {l}", r["single"][str(l)], LAM2) for l in L2] + \
         [("bands", f"band {k_}", e, BLAM2) for k_, e in r["band"].items()]
    for grp_, nm, e, lams in cs:
        fa, a_, b_ = flip_actual(r0, lams, [o["r"] for o in e["out"]])
        if not math.isfinite(fa):
            continue
        fj, fq = flip_J(r0, e["g1"]), flip_JQ(r0, e["g1"], e["g2"])

        def lb(pv, a_=a_, b_=b_):       # distance (log) from the prediction to the grid interval holding the flip
            if not math.isfinite(pv):
                return math.inf
            return max(0.0, math.log(pv / b_)) if a_ == 0 else max(0.0, math.log(a_ / pv), math.log(pv / b_))
        c_ = dict(ej=abs(math.log(fj / fa)) if math.isfinite(fj) else math.inf,
                  eq=abs(math.log(fq / fa)) if math.isfinite(fq) else math.inf, bj=lb(fj), bq=lb(fq),
                  width=math.log(b_ / a_) if a_ > 0 else math.inf, j_over=fj > fa, q_over=fq > fa,
                  j_none=not math.isfinite(fj), q_none=not math.isfinite(fq))
        dose["all"].append(c_); dose[grp_].append(c_); dose.setdefault(nm, []).append(c_)
for grp_, cs in dose.items():
    st = f"twohop dose, {grp_}"
    fin = lambda key, cs=cs: [c_[key] for c_ in cs if math.isfinite(c_[key])]  # noqa: E731
    for q, key in (("median factor J", "ej"), ("median factor J+Q", "eq"), ("median factor to grid interval J", "bj"),
                   ("median factor to grid interval J+Q", "bq"), ("median grid interval width factor", "width")):
        xs = [c_[key] for c_ in cs]
        qs = np.percentile(fin(key), [25, 75])
        ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=st, quantity=q, value=math.exp(umed(xs)),
                         lo=math.exp(qs[0]), hi=math.exp(qs[1]), n=len(cs), file=f2))
    for q, v_ in (("frac J+Q closer", np.mean([c_["eq"] < c_["ej"] for c_ in cs])),
                  ("frac J inside grid interval", np.mean([c_["bj"] == 0 for c_ in cs])),
                  ("frac J+Q inside grid interval", np.mean([c_["bq"] == 0 for c_ in cs])),
                  ("frac J predicts larger dose", np.mean([c_["j_over"] for c_ in cs])),
                  ("frac J+Q predicts larger dose", np.mean([c_["q_over"] for c_ in cs])),
                  ("frac J predicts no flip", np.mean([c_["j_none"] for c_ in cs])),
                  ("frac J+Q predicts no flip", np.mean([c_["q_none"] for c_ in cs]))):
        ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=st, quantity=q, value=float(v_), lo=float("nan"),
                         hi=float("nan"), n=len(cs), file=f2))

# push size of a full swap: beta_l = |Delta_l| / |h_l| over positions 1..T-1 (two-hop, single layers, lambda = 1)
for l in L2:
    bt = [r["single"][str(l)]["beta"] for r in ok2]
    ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=f"twohop layer {l}", quantity="swap push |Delta|/|h| at lambda=1",
                     value=med(bt), lo=float(np.percentile(bt, 25)), hi=float(np.percentile(bt, 75)), n=len(bt), file=f2))
# B3x: the Q-lens as a readout lens (J-lens readout J d of a state's deviation d, with and without Q(d, d)/2)
for fn in ("wsfix/B3x_a_wsfix", "B3x_b", "B3x_c"):  # B3x_a (multihop, order-ops) rerun 2026-11-22 after the space-token fix
    for setn, v_ in json.loads((S_DIR / f"{fn}.json").read_text())["sets"].items():
        for meth in ("J_d", "J_d+Q/2", "logit"):
            ROWS.append(dict(claim="S", model="Qwen3.8-27B", setting=f"lens eval {setn}", quantity=f"AUC {meth}",
                             value=v_["summary"][meth]["auc"], lo=float("nan"), hi=float("nan"), n=v_["n_items"],
                             file=f"second_order_v1/{fn}.json"))

# ---------------------------------------------------------------- C2r: per-case r against the error of J (alpha = 0.05)
from scipy.stats import spearmanr  # noqa: E402
for name, d in QWEN:
    js = json.loads((W / d / "D1.json").read_text())
    raw = torch.load(W / d / "D1_raw.pt", map_location="cpu", weights_only=False)
    al, nl = js["alphas"], js["n_layers"]
    a5 = al.index(0.05)
    for s in sorted(raw.keys()):
        R = raw[s]["iso"]
        fi = raw[s]["targets"].index(nl - 1)
        rr = (R["norm_Q"] / R["norm_J"]).numpy()[:, :, fi]
        ee = R["rel_lin"].numpy()[:, :, a5, fi]
        st = f"src={s} k={nl - 1 - s} iso"
        ROWS.append(dict(claim="C2r", model=name, setting=st, quantity="spearman case r vs rel_err_J alpha=0.05",
                         value=float(spearmanr(rr.ravel(), ee.ravel())[0]), lo=float("nan"), hi=float("nan"),
                         n=rr.size, file=f"{d}/D1_raw.pt"))
        pc = np.median(rr, 1)
        for ci in range(len(pc)):
            ROWS.append(dict(claim="C2r", model=name, setting=st, quantity=f"ctx{ci} median r", value=float(pc[ci]),
                             lo=float(np.median(ee[ci])), hi=float("nan"), n=rr.shape[1], file=f"{d}/D1_raw.pt"))

# ---------------------------------------------------------------- C2f: other families (results/d1_family_fits.jsonl,
# written by src/fit_d1_family.py with the C2 method above; it reproduces Qwen3-8B's slope). Models that fail the
# finite-difference gate (Gemma-4-31B) are not used.
FAM = P / "d1_family_fits.jsonl"
if FAM.exists():
    for line in FAM.read_text().splitlines():
        fam = json.loads(line)
        if "PASS" not in fam["gate"] or fam["dir"] == "so_qwen3-8b":
            continue
        nm = {"so_llama3_1-8b": "Llama-3.1-8B"}.get(fam["dir"], fam["dir"].replace("so_olmo2-1b_ckpt/", "OLMo-2-1B "))
        for tn in ("iso", "data"):
            fv = fam[tn]
            for q_, key in (("slope", "slope"), ("within push block", "within_push_block")):
                ROWS.append(dict(claim="C2f", model=nm, setting=tn, quantity=q_, value=fv[key][0], lo=fv[key][1],
                                 hi=fv[key][2], n=fam["n_layers"], file="results/d1_family_fits.jsonl"))
            ROWS.append(dict(claim="C2f", model=nm, setting=tn, quantity="fitted Q/J at k=1", value=fv["qj_k1"],
                             lo=float("nan"), hi=float("nan"), n=fam["n_layers"], file="results/d1_family_fits.jsonl"))
            for gk in fv["gain_final_alpha0p1"]:
                ROWS.append(dict(claim="C2f", model=nm, setting=f"{tn} k={gk['k']} alpha=0.1", quantity="gain_J_over_JQ",
                                 value=gk["gain"], lo=gk["relerr_J"], hi=float("nan"), n=fam["n_layers"],
                                 file="results/d1_family_fits.jsonl"))

# ---------------------------------------------------------------- v8.4: Lean pilot, responsibility maps, FD precision,
# cost curve, numeric-intermediate control (paper/claims_v84.py; own generator RNG3, after every older row)
import claims_v84  # noqa: E402
claims_v84.run(ROWS)

# ---------------------------------------------------------------- write
with open(P / "claims.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
    w.writeheader()
    for r in ROWS:
        w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
ex = P / "explorable_data"
(ex / "E2_taylor_error.json").write_text(json.dumps(e2))
(ex / "E3_qj_curves.json").write_text(json.dumps(
    [dict(model=c[0], family=c[1], src=str(c[2]), type=c[3], k=c[4], qj=round(c[5], 4)) for c in curves]))
(ex / "E4_block_lengths.json").write_text(json.dumps(e4))
(ex / "E5_reconstruction.json").write_text(json.dumps(e5))

# ---------------------------------------------------------------- compact console summary
print(f"rows={len(ROWS)} curves={len(curves)} max consistency err={max(CHECKS):.2e}")
for (name, tn), (b, a0, rms, n) in fits.items():
    r = [x for x in ROWS if x["claim"] == "C2" and x["model"] == name and x["setting"].startswith(tn + " all")][0]
    print(f"C2 {name:12s} {tn:4s} slope={b:.4f} [{r['lo']:.4f},{r['hi']:.4f}] /block  Q/J(k=1)~{np.exp(a0 + b):.2f}"
          f"  rms_log_resid={rms:.2f}  pairs={n}")
print("A2/A3 files:", [a[2] for a in A2], USED)
print("C5 pooled between-block:", {k: (round(v[0], 3), round(v[1], 2), v[2]) for k, v in pooled.items()})
