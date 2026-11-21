#!/usr/bin/env python3
"""Curvature circuits: is the between-block third order a sparse, reusable set of (j, k) block pairs, or collective?

Input: A3 raw files of src/q_anatomy.py (--a3-pair 1). raw[l][type] = list over contexts of dicts with
  M [8, K, K]  M[i, j, k] = 3 sum_{X in block k} <J_{X->L}^T t_hat, D^2 f_X[tau_X, q_X^(j)]>, direction i,
               j = block that made the curvature (row), k >= j = block whose second derivative bends it (column);
               signed, along t_hat = T/|T|, units of |T| (lower triangle is zero; diagonal = within-block)
  T, C3_self, C3_cross [8, K, 4, d]  (self = D^3 terms, cross = 3 D^2 terms, per block and site)
CPU only, no model is loaded.

  python src/curv_circuits.py [--model 27b|8b] [--md out.md] [--figs results/figs_curv]

Statistics per case (one context x one direction) on the between-block pairs j < k (N = K(K-1)/2), x = M/|T|:
  pr = (sum|x|)^2 / sum x^2 (effective number of pairs); ci = sum|x| / |sum x| (cancellation);
  yardstick: random signs on the same magnitudes give ci ~ sqrt(pr) (rms), so ci/sqrt(pr) ~ 1 means "as cancelling
  as unrelated contributions"; z = sum x / ||x||_2 (~N(0,1) under random signs; = sqrt(pr) if all signs agree).
  top-n mass share = sum of the n largest |x| / sum|x|; err_top n = |S - sum of the n largest x| / |S|;
  n90 = pairs (largest first) needed to stay within 10% of S = sum x.
CIs: two-way bootstrap over contexts AND directions (B = 2000, seed 20261121), statistic = median over cases.
Consistency: Spearman of |M| (and of signed M) between cases, by relation (contexts / directions / iso-data /
sources), against (a) a full permutation null and (b) a distance-stratified permutation null (entries shuffled
within each diagonal k - j = d), plus the same on residuals of log|M| after removing row and column effects.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
ARGV = sys.argv[1:]
MODEL = ARGV[ARGV.index("--model") + 1] if "--model" in ARGV else "27b"
MD = ARGV[ARGV.index("--md") + 1] if "--md" in ARGV else None
FIGS = Path(ARGV[ARGV.index("--figs") + 1]) if "--figs" in ARGV else Path("results/figs_curv")
FILES = {"27b": [W / f"qanat_q38-27b/A3_s{s}_32_raw.pt" for s in (8, 24, 40)],
         "8b": [W / "qanat_qwen3-8b/A3_raw.pt"]}[MODEL]
JSONS = {"27b": [W / f"qanat_q38-27b/A3_s{s}_32.json" for s in (8, 24, 40)],
         "8b": [W / "qanat_qwen3-8b/A3.json"]}[MODEL]
A2F = {"27b": [W / "qanat_q38-27b/A2_a32.json", W / "qanat_q38-27b/A2_b32.json"],
       "8b": [W / "qanat_qwen3-8b/A2_32.json"]}[MODEL]
NAME = {"27b": "Qwen3.8-27B (fp32)", "8b": "Qwen3-8B (TF32)"}[MODEL]
CACHE = W / "curv_circuits" / f"{MODEL}_M.npz"
B = 2000
SEED = 20261121
TYPES = ("iso", "data")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


# ---------------------------------------------------------------- 1. extract (once) to a compact cache
def extract():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["data"].item()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    js = {}
    for f in JSONS:
        for l, v in json.loads(f.read_text())["sources"].items():
            js[int(l)] = v
    for f in FILES:
        t0 = time.time()
        raw = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        for l, v in raw.items():
            l = int(l)
            ent = {}
            for tn in TYPES:
                Ms, Tn, st, ct, blocks = [], [], [], [], None
                for w in v[tn]:
                    blocks = list(w["blocks"])
                    T = w["T"].double()
                    tn_ = T.norm(dim=-1)
                    th = T / tn_[:, None]
                    S = w["C3_self"].double().sum((1, 2))
                    X = w["C3_cross"].double().sum((1, 2))
                    Ms.append(w["M"].double().numpy()); Tn.append(tn_.numpy())
                    st.append((S * th).sum(-1).numpy()); ct.append((X * th).sum(-1).numpy())
                rows = js[l][tn]["rows"]
                ent[tn] = {"M": np.stack(Ms), "Tn": np.stack(Tn), "self_t": np.stack(st), "cross_t": np.stack(ct),
                           "gate_json": np.array([r["pair"]["gate"] for r in rows]).reshape(len(Ms), -1),
                           "cross_share_json": np.array([r["cross_share"] for r in rows]).reshape(len(Ms), -1),
                           "cross_jk_json": np.array([r["pair"]["cross_jk_share"] for r in rows]).reshape(len(Ms), -1)}
            ent["blocks"] = blocks
            data[l] = ent
        print(f"extracted {f.name} in {time.time() - t0:.1f}s", flush=True)
        del raw
    np.savez_compressed(CACHE, data=np.array(data, dtype=object))
    return data


# ---------------------------------------------------------------- helpers
def boot2(x, stat=np.median, rng=None):
    """x [C, D] (per context, direction). Two-way bootstrap percentile CI of stat over all cells."""
    C, D = x.shape
    v = stat(x.ravel())
    bs = np.empty(B)
    for b in range(B):
        ci, di = rng.integers(0, C, C), rng.integers(0, D, D)
        bs[b] = stat(x[ci][:, di].ravel())
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return v, lo, hi


def fmt(vlh, p=2):
    v, lo, hi = vlh
    return f"{v:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def n_stay(err, eps):
    bad = np.nonzero(err > eps)[0]
    return int(bad.max()) + 2 if len(bad) else 1


def ranks(a):
    """Ranks along the last axis (ties are measure-zero for float M on j < k)."""
    return np.argsort(np.argsort(a, axis=-1, kind="stable"), axis=-1, kind="stable").astype(float)


def zrows(r):
    r = r - r.mean(-1, keepdims=True)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def resid_logabs(Mu, K, iu):
    """log|M| on j<k minus a two-way additive fit a_j + b_k (least squares), per case. Mu [n, N]."""
    jj, kk = iu
    A = np.zeros((len(jj), 2 * K))
    A[np.arange(len(jj)), jj] = 1
    A[np.arange(len(jj)), K + kk] = 1
    P = A @ np.linalg.pinv(A)
    y = np.log(np.abs(Mu) + 1e-300)
    return y - y @ P.T


# ---------------------------------------------------------------- 2. per-case statistics
def case_stats(x, K, iu, rng, n_sign=1000):
    """x [N] between-block pair shares (M/|T|, j<k) of one case."""
    a = np.abs(x)
    S, A, Q = x.sum(), a.sum(), (x * x).sum()
    N = len(x)
    order = np.argsort(-a)
    cs = np.cumsum(x[order])
    err = np.abs(S - cs) / abs(S)
    n1 = max(1, round(0.01 * N))
    sg = rng.choice([-1.0, 1.0], size=(n_sign, N))
    ci_null = np.median(A / np.abs(sg @ a))
    jj, kk = iu
    d = kk - jj
    rowm = np.bincount(jj, a, minlength=K); colm = np.bincount(kk, a, minlength=K)
    rows_ = np.bincount(jj, x, minlength=K); cols_ = np.bincount(kk, x, minlength=K)
    dm = np.bincount(d, a, minlength=K) / A
    dnet = np.bincount(d, x, minlength=K)
    return {"S": S, "A": A, "pr": A * A / Q, "N": N, "ci": A / abs(S), "sqrt_pr": math.sqrt(A * A / Q),
            "ci_null": ci_null, "z": S / math.sqrt(Q), "pos_frac_mass": a[x > 0].sum() / A,
            "top1_T": a[order[0]], "top1_mass": a[order[0]] / A, "top10_mass": a[order[:10]].sum() / A,
            "top1pct_mass": a[order[:n1]].sum() / A, "n1pct": n1,
            "err_top1": err[0], "err_top10": err[9], "err_top1pct": err[n1 - 1],
            "n50": n_stay(err, 0.5), "n90": n_stay(err, 0.1),
            "rowm": rowm / A, "colm": colm / A, "rows": rows_, "cols": cols_, "dmass": dm, "dnet": dnet,
            "row_pr": rowm.sum() ** 2 / (rowm ** 2).sum(), "col_pr": colm.sum() ** 2 / (colm ** 2).sum(),
            "row_ci": np.abs(rows_).sum() / abs(rows_.sum()), "col_ci": np.abs(cols_).sum() / abs(cols_.sum()),
            "row_sqrt_pr": math.sqrt(np.abs(rows_).sum() ** 2 / (rows_ ** 2).sum()),
            "col_sqrt_pr": math.sqrt(np.abs(cols_).sum() ** 2 / (cols_ ** 2).sum())}


def gauss_ref(N, rng, n=2000):
    """Top-n mass shares and pr/N for i.i.d. Gaussian contributions of the same count (no structure)."""
    g = np.abs(rng.standard_normal((n, N)))
    g = -np.sort(-g, axis=1)
    A = g.sum(1)
    n1 = max(1, round(0.01 * N))
    return {"top1": np.median(g[:, 0] / A), "top10": np.median(g[:, :10].sum(1) / A),
            "top1pct": np.median(g[:, :n1].sum(1) / A), "pr_frac": np.median(A ** 2 / (g ** 2).sum(1) / N)}


# ---------------------------------------------------------------- 3. consistency
def corr_matrix(V):
    Z = zrows(ranks(V))
    return Z @ Z.T


def strat_perm(V, groups, rng):
    """Shuffle each row's entries within groups (index arrays)."""
    P = V.copy()
    for g in groups:
        sub = P[:, g]
        idx = np.argsort(rng.random(sub.shape), axis=1)
        P[:, g] = np.take_along_axis(sub, idx, axis=1)
    return P


def topk_overlap(V, k):
    top = np.argsort(-V, axis=1)[:, :k]
    n = len(V)
    Mk = np.zeros((n, V.shape[1]), bool)
    np.put_along_axis(Mk, top, True, axis=1)
    Mk = Mk.astype(float)
    return Mk @ Mk.T / k


def relation_masks(meta):
    """meta: list of (type, ctx, dir). Returns relation -> boolean [n, n] (upper triangle only)."""
    t = np.array([m[0] for m in meta]); c = np.array([m[1] for m in meta]); d = np.array([m[2] for m in meta])
    st, sc, sd = t[:, None] == t[None], c[:, None] == c[None], d[:, None] == d[None]
    up = np.triu(np.ones((len(meta),) * 2, bool), 1)
    return {"contexts (same type & dir)": up & st & ~sc & sd,
            "directions (same type & ctx)": up & st & sc & ~sd,
            "iso vs data (same ctx)": up & ~st & sc,
            "neither (same type)": up & st & ~sc & ~sd}


def boot_rel(R, meta, mask, rng, nb=B):
    """Two-way bootstrap (contexts, directions) of the mean of R over pairs in mask."""
    c = np.array([m[1] for m in meta]); d = np.array([m[2] for m in meta])
    C, D = c.max() + 1, d.max() + 1
    v = R[mask].mean()
    bs = []
    for _ in range(nb):
        wc = np.bincount(rng.integers(0, C, C), minlength=C); wd = np.bincount(rng.integers(0, D, D), minlength=D)
        w = wc[c] * wd[d]
        Wm = (w[:, None] * w[None]) * mask
        s = Wm.sum()
        if s > 0:
            bs.append((Wm * R).sum() / s)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return v, lo, hi


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    data = extract()
    srcs = sorted(data)
    say(f"# {NAME}: curvature circuits from A3 M[j,k] (src/curv_circuits.py)")
    say()
    # --- gates
    say("## Gates (recomputed from the raw tensors)")
    say("| src | type | max pair gate (recomputed) | max pair gate (JSON) | max \\|ΣM − ⟨cross, t̂⟩\\| / \\|T\\| | "
        "max \\|ΣM/\\|T\\| − cross_share (JSON)\\| | lower triangle max \\|M\\| |")
    say("|---|---|---|---|---|---|---|")
    stats, allM = {}, {}
    for l in srcs:
        for tn in TYPES:
            e = data[l][tn]
            M, Tn = e["M"], e["Tn"]
            sm = M.sum((-1, -2))
            g_re = np.abs(sm + e["self_t"] - Tn) / Tn
            g_cx = np.abs(sm - e["cross_t"]) / Tn
            g_js = np.abs(sm / Tn - e["cross_share_json"])
            low = np.abs(np.tril(M, -1)).max()
            say(f"| {l} | {tn} | {g_re.max():.1e} | {e['gate_json'].max():.1e} | {g_cx.max():.1e} | {g_js.max():.1e} | {low:.1e} |")
    say()
    # --- per-case stats
    for l in srcs:
        K = len(data[l]["blocks"])
        iu = np.triu_indices(K, 1)
        for tn in TYPES:
            e = data[l][tn]
            C, D = e["M"].shape[:2]
            X = e["M"] / e["Tn"][..., None, None]
            stats[(l, tn)] = [[case_stats(X[c, d][iu], K, iu, rng) for d in range(D)] for c in range(C)]
            allM[(l, tn)] = X
    # --- concentration table
    say("## Concentration and cancellation of the between-block pairs (j < k), medians over 32 cases [two-way bootstrap CI]")
    say("x = M/|T|. Uniform baseline for the top-n mass share is n/N; 'Gauss' = i.i.d. Gaussian contributions of the "
        "same count (no structure).")
    say()
    say("| src | type | N pairs | between share Σx | top-1 pair \\|x\\| (share of \\|T\\|) | top-1 / top-10 / top-1% mass share | "
        "Gauss ref | pr (pr/N) | ci | √pr | ci/√pr | sign-null ci | z = Σx/‖x‖ | n90 (of N) | err after top-10 |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    G = lambda l, tn, k: np.array([[s[k] for s in row] for row in stats[(l, tn)]])  # noqa: E731
    summ = {}
    for l in srcs:
        K = len(data[l]["blocks"]); N = K * (K - 1) // 2
        gr = gauss_ref(N, rng)
        for tn in TYPES:
            r = {k: boot2(G(l, tn, k), rng=rng) for k in ("S", "top1_T", "top1_mass", "top10_mass", "top1pct_mass", "pr",
                                                       "ci", "sqrt_pr", "ci_null", "z", "n90", "err_top10", "n50")}
            r["ratio"] = boot2(G(l, tn, "ci") / G(l, tn, "sqrt_pr"), rng=rng)
            r["pr_frac"] = boot2(G(l, tn, "pr") / N, rng=rng)
            summ[(l, tn)] = r
            n1 = max(1, round(0.01 * N))
            say(f"| {l} | {tn} | {N} | {fmt(r['S'])} | {fmt(r['top1_T'], 3)} | {r['top1_mass'][0]:.3f} / {r['top10_mass'][0]:.3f} / "
                f"{r['top1pct_mass'][0]:.3f} (top {n1}) | {gr['top1']:.3f} / {gr['top10']:.3f} / {gr['top1pct']:.3f} | "
                f"{r['pr'][0]:.0f} ({r['pr_frac'][0]:.2f}; Gauss {gr['pr_frac']:.2f}) | {fmt(r['ci'], 1)} | {fmt(r['sqrt_pr'], 1)} | "
                f"{fmt(r['ratio'])} | {r['ci_null'][0]:.1f} | {fmt(r['z'], 1)} | {r['n90'][0]:.0f} | {r['err_top10'][0]:.2f} |")
    say()
    # pooled
    allr = np.concatenate([(G(l, tn, "ci") / G(l, tn, "sqrt_pr")).ravel() for l in srcs for tn in TYPES])
    allz = np.concatenate([G(l, tn, "z").ravel() for l in srcs for tn in TYPES])
    allpos = np.concatenate([G(l, tn, "pos_frac_mass").ravel() for l in srcs for tn in TYPES])
    for l in srcs:
        for tn in TYPES:
            say(f"- src {l} {tn}: best single pair leaves {np.median(G(l, tn, 'err_top1')):.2f} of the between-block net "
                f"(top 1% of pairs: {np.median(G(l, tn, 'err_top1pct')):.2f}); n90 = {np.median(G(l, tn, 'n90') / G(l, tn, 'N')):.2f} of pairs; "
                f"negative share of |x| mass {1 - np.median(G(l, tn, 'pos_frac_mass')):.2f}.")
    say(f"- Pooled over {len(allr)} cases: median ci/√pr = {np.median(allr):.2f} (IQR {np.percentile(allr, 25):.2f}–"
        f"{np.percentile(allr, 75):.2f}); median z = {np.median(allz):.2f}; fraction of |x| mass with x > 0: "
        f"median {np.median(allpos):.3f}; cases with z > 3: {np.mean(allz > 3):.2f}, z < 0: {np.mean(allz < 0):.2f}.")
    say()
    # --- distance profile
    say("## Distance profile (k − j)")
    say("mass(d) = Σ_{k−j=d} |x| / Σ|x|; pairs(d) = fraction of pairs at distance d; intensity = mass/pairs (1 = flat); "
        "net(d) = Σ_{k−j=d} x (share of |T|). Medians over cases (iso and data pooled, 64 cases).")
    say()
    say("| src | d=1 mass / pairs / net | d≤3 mass / pairs | d≤5 mass / pairs | d≤K/4 mass / pairs | far half (d>K/2) mass / pairs | "
        "net d=1 | net d≥2 |")
    say("|---|---|---|---|---|---|---|---|")
    prof = {}
    for l in srcs:
        K = len(data[l]["blocks"]); N = K * (K - 1) // 2
        pairs = np.array([(K - dd) / N if dd > 0 else 0 for dd in range(K)])
        dm = np.array([s["dmass"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        dn = np.array([s["dnet"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        prof[l] = (np.median(dm, 0), pairs, np.percentile(dm, 25, 0), np.percentile(dm, 75, 0))
        cm = lambda m: np.median(m.sum(1))  # noqa: E731
        q = K // 4; h = K // 2
        say(f"| {l} (K={K}) | {np.median(dm[:, 1]):.3f} / {pairs[1]:.3f} / {np.median(dn[:, 1]):+.3f} | "
            f"{cm(dm[:, 1:4]):.3f} / {pairs[1:4].sum():.3f} | {cm(dm[:, 1:6]):.3f} / {pairs[1:6].sum():.3f} | "
            f"{cm(dm[:, 1:q + 1]):.3f} / {pairs[1:q + 1].sum():.3f} | {cm(dm[:, h + 1:]):.3f} / {pairs[h + 1:].sum():.3f} | "
            f"{np.median(dn[:, 1]):+.3f} | {np.median(dn[:, 2:].sum(1)):+.3f} |")
    say()
    # --- sources and benders (per-pair intensity: row j has K-1-j pairs, column k has k pairs)
    say("## Row sums (curvature sources j) and column sums (benders k)")
    say("Row j has K−1−j between-block pairs and column k has k, so raw mass shares favour early rows and late columns "
        "by counting alone. Intensity = (Σ_k |x_jk| / Σ|x|) / ((K−1−j)/N) for rows, (Σ_j |x_jk| / Σ|x|) / (k/N) for "
        "columns; 1 = flat. pr = effective number of rows / columns carrying the mass (flat reference in brackets). "
        "Top blocks by median intensity over 64 cases; 'top-3 freq' = fraction of cases in which the block is among "
        "the 3 most intense (chance 3/(K−1)).")
    say()
    say("| src | row pr (flat) | col pr (flat) | top source blocks j: intensity, top-3 freq | "
        "top bender blocks k: intensity, top-3 freq | chance |")
    say("|---|---|---|---|---|---|")
    blockprof = {}
    for l in srcs:
        blocks = np.array(data[l]["blocks"]); K = len(blocks); N = K * (K - 1) // 2
        rp = np.arange(K - 1, -1, -1) / N; cp = np.arange(K) / N
        flat_r = rp.sum() ** 2 / (rp ** 2).sum()
        rm = np.array([s["rowm"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        cmm = np.array([s["colm"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        ri = rm[:, :-1] / rp[None, :-1]; ci_ = cmm[:, 1:] / cp[None, 1:]
        rpr = np.median([s["row_pr"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        cpr = np.median([s["col_pr"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        f3r = (np.argsort(-ri, 1)[:, :3, None] == np.arange(K - 1)[None, None]).any(1).mean(0)
        f3c = (np.argsort(-ci_, 1)[:, :3, None] == np.arange(K - 1)[None, None]).any(1).mean(0)
        mr, mc = np.median(ri, 0), np.median(ci_, 0)
        blockprof[l] = (blocks[:-1], mr, blocks[1:], mc, f3r, f3c)
        tr = np.argsort(-mr)[:3]; tc = np.argsort(-mc)[:3]
        say(f"| {l} (K={K}) | {rpr:.1f} ({flat_r:.1f}) | {cpr:.1f} ({flat_r:.1f}) | "
            + ", ".join(f"L{blocks[i]}: {mr[i]:.2f}, {f3r[i]:.2f}" for i in tr) + " | "
            + ", ".join(f"L{blocks[1 + i]}: {mc[i]:.2f}, {f3c[i]:.2f}" for i in tc) + f" | {3 / (K - 1):.2f} |")
    say()
    # --- per context: the hub pair and the hub bender
    say("## Per context (16 cases each: 8 iso + 8 data directions)")
    say("Top pair = the pair with the largest median |x| over the context's 16 cases, with that median (share of |T|) "
        "and the fraction of the 16 cases in which it is the single largest pair. Bender = column with the largest "
        "median mass share Σ_j|x_jk|/Σ|x|. Adjacent = mass share at k − j = 1.")
    say()
    say("| src | ctx | top pair j→k | its median \\|x\\| | is top-1 in | top bender k: mass share | median top-1 \\|x\\| | adjacent mass | between share Σx |")
    say("|---|---|---|---|---|---|---|---|---|")
    for l in srcs:
        bl = data[l]["blocks"]; K = len(bl); iu = np.triu_indices(K, 1)
        X = np.concatenate([allM[(l, tn)] for tn in TYPES], 1)  # [C, 16, K, K]
        for c in range(X.shape[0]):
            V = X[c][:, iu[0], iu[1]]; Va = np.abs(V)
            med = np.median(Va, 0); t = int(np.argmax(med))
            frac = np.mean(Va.argmax(1) == t)
            colm = np.array([np.bincount(iu[1], v, minlength=K) / v.sum() for v in Va])
            kc = int(np.argmax(np.median(colm, 0)))
            adj = np.median(Va[:, iu[1] - iu[0] == 1].sum(1) / Va.sum(1))
            say(f"| {l} | {c} | L{bl[iu[0][t]]}→L{bl[iu[1][t]]} | {med[t]:.3f} | {frac:.2f} | L{bl[kc]}: {np.median(colm, 0)[kc]:.2f} | "
                f"{np.median(Va.max(1)):.3f} | {adj:.2f} | {np.median(V.sum(1)):.2f} |")
    say()
    # --- consistency
    say("## Consistency: do the same pairs recur? (Spearman of |M| over the N between-block pairs)")
    say("Mean over case pairs in each relation [two-way bootstrap CI]. Nulls: 'perm' = entries of one case shuffled "
        "(mean, 99th percentile over pairs); 'strat' = shuffled within each distance d (keeps the distance profile). "
        "'resid' = Spearman of log|M| residuals after removing row (j) and column (k) effects: pair-specific recurrence "
        "beyond which blocks are big. 'signed' = Spearman of signed M. top-10 = overlap fraction of the 10 largest-|M| "
        "pairs (perm null 10/N).")
    say()
    say("| src | relation | n pairs | Spearman \\|M\\| | perm null mean (p99) | strat null mean | resid Spearman | "
        "signed Spearman | top-10 overlap (strat null) | top-1% overlap |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    cons = {}
    for l in srcs:
        K = len(data[l]["blocks"]); iu = np.triu_indices(K, 1); N = len(iu[0])
        meta, V, Vs = [], [], []
        for tn in TYPES:
            X = allM[(l, tn)]
            for c in range(X.shape[0]):
                for d in range(X.shape[1]):
                    meta.append((tn, c, d)); Vs.append(X[c, d][iu])
        Vs = np.array(Vs); V = np.abs(Vs)
        R = corr_matrix(V)
        Rs = corr_matrix(Vs)
        Rr = corr_matrix(resid_logabs(Vs, K, iu))
        groups = [np.nonzero(iu[1] - iu[0] == dd)[0] for dd in range(1, K)]
        Rp = corr_matrix(np.array([rng.permutation(v) for v in V]))
        Vst = strat_perm(V, groups, rng)
        # strat null: correlate the observed cases with independently stratified-shuffled copies
        Zo, Zs = zrows(ranks(V)), zrows(ranks(Vst))
        Rst = Zo @ Zs.T
        n1 = max(1, round(0.01 * N))
        O10, O1 = topk_overlap(V, 10), topk_overlap(V, n1)
        O10s = np.zeros_like(O10)
        top_o = np.argsort(-V, 1)[:, :10]; top_s = np.argsort(-Vst, 1)[:, :10]
        for a in range(len(V)):
            O10s[a] = [len(set(top_o[a]) & set(top_s[b])) / 10 for b in range(len(V))]
        masks = relation_masks(meta)
        cons[l] = {}
        for name, mk in masks.items():
            v = boot_rel(R, meta, mk, rng)
            pn = Rp[mk]
            cons[l][name] = (v, R[mk])
            say(f"| {l} | {name} | {mk.sum()} | {fmt(v)} | {pn.mean():+.3f} ({np.percentile(pn, 99):.3f}) | {Rst[mk].mean():.2f} | "
                f"{fmt(boot_rel(Rr, meta, mk, rng))} | {fmt(boot_rel(Rs, meta, mk, rng))} | {O10[mk].mean():.2f} ({O10s[mk].mean():.2f}) | "
                f"{O1[mk].mean():.2f} (top {n1}; null {n1 / N:.3f}) |")
    say()
    # across source layers: common blocks, same type & context
    say("### Across source layers (blocks common to both spans; same type and context)")
    say("| sources | common blocks | relation | n pairs | Spearman \\|M\\| | strat null | resid Spearman | signed Spearman | top-10 overlap |")
    say("|---|---|---|---|---|---|---|---|---|")
    for a_ in range(len(srcs)):
        for b_ in range(a_ + 1, len(srcs)):
            la, lb = srcs[a_], srcs[b_]
            ba, bb = data[la]["blocks"], data[lb]["blocks"]
            com = [x for x in ba if x in bb]
            ia = [ba.index(x) for x in com]; ib = [bb.index(x) for x in com]
            Kc = len(com); iu = np.triu_indices(Kc, 1)
            groups = [np.nonzero(iu[1] - iu[0] == dd)[0] for dd in range(1, Kc)]
            for tn in TYPES:
                Xa = allM[(la, tn)][:, :, ia][:, :, :, ia]; Xb = allM[(lb, tn)][:, :, ib][:, :, :, ib]
                C, D = Xa.shape[:2]
                Va = np.array([Xa[c, d][iu] for c in range(C) for d in range(D)])
                Vb = np.array([Xb[c, d][iu] for c in range(C) for d in range(D)])
                ca = np.repeat(np.arange(C), D); da = np.tile(np.arange(D), C)
                Za, Zb = zrows(ranks(np.abs(Va))), zrows(ranks(np.abs(Vb)))
                Rab = Za @ Zb.T
                Rsab = zrows(ranks(Va)) @ zrows(ranks(Vb)).T
                Rrab = zrows(ranks(resid_logabs(Va, Kc, iu))) @ zrows(ranks(resid_logabs(Vb, Kc, iu))).T
                Rst = Za @ zrows(ranks(strat_perm(np.abs(Vb), groups, rng))).T
                ta = np.argsort(-np.abs(Va), 1)[:, :10]; tb = np.argsort(-np.abs(Vb), 1)[:, :10]
                O = np.array([[len(set(ta[i]) & set(tb[j])) / 10 for j in range(len(Vb))] for i in range(len(Va))])
                for name, mk in (("same ctx, same dir index", (ca[:, None] == ca[None]) & (da[:, None] == da[None])),
                                 ("same ctx, other dir", (ca[:, None] == ca[None]) & (da[:, None] != da[None]))):
                    say(f"| {la} vs {lb} {tn} | {Kc} (L{com[0]}–{com[-1]}) | {name} | {mk.sum()} | {Rab[mk].mean():.2f} | "
                        f"{Rst[mk].mean():.2f} | {Rrab[mk].mean():.2f} | {Rsab[mk].mean():.2f} | {O[mk].mean():.2f} |")
    say()
    q_yardstick(srcs, stats, rng)
    # --- figures
    try:
        figures(data, allM, stats, prof, blockprof, cons, srcs)
    except Exception as ex:  # figures are a convenience; the tables are the record
        say(f"(figures failed: {ex!r})")
    if MD:
        Path(MD).write_text("\n".join(OUT) + "\n")
        print("wrote", MD)


def q_yardstick(srcs, stats, rng):
    """Q's per-block pieces (A2 fp32 rows; same iso/data directions, contexts 0-7) in the same measures as T's pairs."""
    js = {}
    for f in A2F:
        d = json.loads(f.read_text())
        for l, v in d["sources"].items():
            js[int(l)] = (v, d["n_ctx"], d["n_dir"])
    say("## Yardstick: Q's per-block pieces in the same measures (A2 fp32; same directions; A2 contexts 0–3 = A3's)")
    say("Q scalar = block contributions projected on q̂ (like M on t̂); Q vector = C4's block-vector version. "
        "|top-1|/|net| = largest single piece over the net it belongs to (T: pair over the between-block net).")
    say()
    say("| src | type | ci/√pr: Q scalar blocks | Q vector blocks (C4) | T between-block pairs | "
        "\\|top-1\\|/\\|net\\|: Q block | T pair | pr/N: Q blocks | T pairs |")
    say("|---|---|---|---|---|---|---|---|---|")
    for l in srcs:
        if l not in js:
            continue
        v, C, D = js[l]
        for tn in TYPES:
            rows = v[tn]["rows"]
            K = len(rows[0]["block_share"])
            qs = np.median([r["scal_block"]["ci"] / math.sqrt(r["scal_block"]["pr"]) for r in rows])
            qv = np.median([r["block"]["ci"] / math.sqrt(r["block"]["pr"]) for r in rows])
            qt = np.median([r["scal_block"]["top_share"]["1"] * r["scal_block"]["ci"] for r in rows])
            qp = np.median([r["scal_block"]["pr"] / K for r in rows])
            ts = [s for row in stats[(l, tn)] for s in row]
            tr = np.median([s["ci"] / s["sqrt_pr"] for s in ts])
            tt = np.median([s["top1_mass"] * s["ci"] for s in ts])
            tp = np.median([s["pr"] / s["N"] for s in ts])
            say(f"| {l} | {tn} | {qs:.2f} | {qv:.2f} | {tr:.2f} | {qt:.3f} | {tt:.3f} | {qp:.2f} | {tp:.2f} |")
    say()
    say("Q hubs per A2 context (median over the context's 16 cases of the largest |block share of Q| along q̂, and the "
        "block that is most often the largest):")
    say()
    say("| src | " + " | ".join(f"ctx {c}" for c in range(8)) + " |")
    say("|---|" + "---|" * 8)
    for l in srcs:
        if l not in js:
            continue
        v, C, D = js[l]
        bs = np.array([[r["block_share"] for r in v[tn]["rows"]] for tn in TYPES])  # [2, C*D, K]
        bs = bs.reshape(2, C, D, -1).transpose(1, 0, 2, 3).reshape(C, 2 * D, -1)
        mx = np.median(np.abs(bs).max(-1), 1); am = [int(np.bincount(np.abs(b).argmax(-1)).argmax()) + l + 1 for b in bs]
        say(f"| {l} | " + " | ".join(f"{mx[c]:.2f} (L{am[c]})" for c in range(C)) + " |")
    say()
    say("Block-level consistency, like for like: Spearman of the K-vector |Q block share| (A2) and of T's bender "
        "profile (column mass Σ_j|M_jk|) and source profile (row mass), by relation; mean over case pairs "
        "(perm null p99 in brackets). Q uses contexts 0–7 (and 0–3 in the last column).")
    say()
    say("| src | relation | Q blocks, ctx 0–7 | T benders (columns) | T sources (rows) | Q blocks, ctx 0–3 |")
    say("|---|---|---|---|---|---|")
    for l in srcs:
        if l not in js:
            continue
        v, C, D = js[l]
        P, meta = [], []
        for tn in TYPES:
            for i, r in enumerate(v[tn]["rows"]):
                P.append(np.abs(r["block_share"])); meta.append((tn, i // D, i % D))
        P = np.array(P); RQ = corr_matrix(P); mq = relation_masks(meta)
        pq = corr_matrix(np.array([rng.permutation(x) for x in P]))
        keep = np.array([m[1] < 4 for m in meta])
        RQ4 = corr_matrix(P[keep]); mq4 = relation_masks([m for m, k in zip(meta, keep) if k])
        tmeta = [(tn, c, d) for tn in TYPES for c in range(len(stats[(l, tn)])) for d in range(len(stats[(l, tn)][0]))]
        colm = np.array([stats[(l, tn)][c][d]["colm"][1:] for tn, c, d in tmeta])
        rowm = np.array([stats[(l, tn)][c][d]["rowm"][:-1] for tn, c, d in tmeta])
        RC, RR = corr_matrix(colm), corr_matrix(rowm); mt = relation_masks(tmeta)
        pc = corr_matrix(np.array([rng.permutation(x) for x in colm]))
        for name in mq:
            say(f"| {l} | {name} | {RQ[mq[name]].mean():.2f} ({np.percentile(pq[mq[name]], 99):.2f}) | "
                f"{RC[mt[name]].mean():.2f} ({np.percentile(pc[mt[name]], 99):.2f}) | {RR[mt[name]].mean():.2f} | "
                f"{RQ4[mq4[name]].mean():.2f} |")
    say()


def figures(data, allM, stats, prof, blockprof, cons, srcs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    FIGS.mkdir(parents=True, exist_ok=True)
    ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    cat = ["#2a78d6", "#eb6834", "#1baf7a"]
    seq = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "text.color": ink, "figure.facecolor": surf, "axes.facecolor": surf, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    tag = MODEL
    # 1. median |M|/|T| heatmaps + one signed example per source
    fig, ax = plt.subplots(2, len(srcs), figsize=(4.2 * len(srcs), 7.6))
    ax = np.atleast_2d(ax).reshape(2, len(srcs))
    for i, l in enumerate(srcs):
        bl = data[l]["blocks"]; K = len(bl)
        A = np.median(np.abs(np.concatenate([allM[(l, tn)].reshape(-1, K, K) for tn in TYPES])), 0)
        A = np.where(np.triu(np.ones((K, K)), 0) > 0, A, np.nan)
        im = ax[0, i].imshow(A, cmap=seq, origin="upper", extent=(bl[0] - .5, bl[-1] + .5, bl[-1] + .5, bl[0] - .5))
        ax[0, i].set_title(f"push at block {l}: median |M[j,k]| / |T|", fontsize=9)
        ax[0, i].set_xlabel("k (bender)"); ax[0, i].set_ylabel("j (curvature source)")
        fig.colorbar(im, ax=ax[0, i], shrink=0.8)
        ex = allM[(l, "iso")][0, 0]
        lim = np.abs(ex).max()
        exm = np.where(np.triu(np.ones((K, K)), 0) > 0, ex, np.nan)
        im = ax[1, i].imshow(exm, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), origin="upper",
                             extent=(bl[0] - .5, bl[-1] + .5, bl[-1] + .5, bl[0] - .5))
        ax[1, i].set_title(f"one case (iso, ctx 0, dir 0): signed M / |T|", fontsize=9)
        ax[1, i].set_xlabel("k (bender)"); ax[1, i].set_ylabel("j (curvature source)")
        fig.colorbar(im, ax=ax[1, i], shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_M_heatmaps.png", dpi=130); plt.close(fig)
    # 1b. per-context median |M|/|T| (log colour): the hubs are context-specific
    from matplotlib.colors import LogNorm
    C = allM[(srcs[0], "iso")].shape[0]
    fig, ax = plt.subplots(len(srcs), C, figsize=(3.3 * C, 3.0 * len(srcs)))
    ax = np.array(ax).reshape(len(srcs), C)
    for i, l in enumerate(srcs):
        bl = data[l]["blocks"]; K = len(bl)
        for c in range(C):
            A = np.median(np.abs(np.concatenate([allM[(l, tn)][c] for tn in TYPES])), 0)
            A = np.where(np.triu(np.ones((K, K)), 1) > 0, A, np.nan)
            im = ax[i, c].imshow(A, cmap=seq, norm=LogNorm(1e-4, 0.5), origin="upper",
                                 extent=(bl[0] - .5, bl[-1] + .5, bl[-1] + .5, bl[0] - .5))
            ax[i, c].set_title(f"push at {l}, context {c}", fontsize=9)
            if i == len(srcs) - 1:
                ax[i, c].set_xlabel("k (bender)")
            if c == 0:
                ax[i, c].set_ylabel("j (source)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="median |M[j,k]| / |T| over 16 directions (j < k)")
    fig.savefig(FIGS / f"{tag}_M_by_context.png", dpi=110); plt.close(fig)
    # 2. distance profile: intensity = mass share / pair share
    fig, axx = plt.subplots(figsize=(5.6, 3.6))
    for i, l in enumerate(srcs):
        med, pairs, lo, hi = prof[l]
        d = np.arange(1, len(pairs))
        axx.plot(d, med[1:] / pairs[1:], color=cat[i], lw=2, label=f"push at block {l}")
        axx.fill_between(d, lo[1:] / pairs[1:], hi[1:] / pairs[1:], color=cat[i], alpha=0.15, lw=0)
    axx.axhline(1, color=ink2, lw=1, ls="--"); axx.text(1, 1.05, "flat (no distance preference)", color=ink2, fontsize=8)
    axx.set_xscale("log"); axx.set_yscale("log")
    axx.set_xlabel("distance k − j (blocks)"); axx.set_ylabel("mass per pair, relative to flat")
    axx.set_title("Between-block |M| by distance (median, IQR band)", fontsize=10); axx.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_distance_profile.png", dpi=130); plt.close(fig)
    # 3. row (source) and column (bender) intensity by absolute block
    fig, axx = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for i, l in enumerate(srcs):
        br, mr, bc, mc, f3r, f3c = blockprof[l]
        axx[0].plot(br, mr, color=cat[i], lw=2, label=f"push at block {l}")
        axx[1].plot(bc, mc, color=cat[i], lw=2, label=f"push at block {l}")
    for a, t in zip(axx, ("curvature sources j (row intensity)", "benders k (column intensity)")):
        a.axhline(1, color=ink2, lw=1, ls="--"); a.set_title(t + ", 1 = flat per pair", fontsize=10)
        a.set_xlabel("block"); a.set_yscale("log")
    axx[0].set_ylabel("median mass per pair, relative to flat"); axx[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_sources_benders.png", dpi=130); plt.close(fig)
    # 4. ci vs sqrt(pr) scatter
    fig, axx = plt.subplots(figsize=(4.4, 4.0))
    mx = 0
    for i, l in enumerate(srcs):
        sp = np.array([s["sqrt_pr"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        ci = np.array([s["ci"] for tn in TYPES for row in stats[(l, tn)] for s in row])
        axx.scatter(sp, ci, s=14, color=cat[i], alpha=0.8, label=f"push at block {l}", edgecolors=surf, linewidths=0.5)
        mx = max(mx, sp.max())
    axx.plot([1, mx], [1, mx], color=ink2, lw=1, ls="--"); axx.text(mx * 0.5, mx * 0.62, "random signs (ci = √pr)", color=ink2, fontsize=8, rotation=35)
    axx.set_xscale("log"); axx.set_yscale("log")
    axx.set_xlabel("√pr (yardstick)"); axx.set_ylabel("cancellation ci = Σ|M| / |ΣM|")
    axx.set_title("Between-block pairs, one dot per case", fontsize=10); axx.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_cancellation.png", dpi=130); plt.close(fig)
    # 5. consistency by relation
    fig, axx = plt.subplots(figsize=(6.4, 3.4))
    rels = list(next(iter(cons.values())).keys())
    w = 0.25
    for i, l in enumerate(srcs):
        for j, r in enumerate(rels):
            (v, lo, hi), vals = cons[l][r]
            axx.bar(j + (i - 1) * w, v, w * 0.9, color=cat[i], label=f"push at block {l}" if j == 0 else None)
            axx.errorbar(j + (i - 1) * w, v, [[v - lo], [hi - v]], color=ink, lw=1, capsize=2)
    axx.set_xticks(range(len(rels))); axx.set_xticklabels([r.split(" (")[0] for r in rels])
    axx.set_ylabel("mean Spearman of |M|"); axx.axhline(0, color=ink2, lw=1)
    axx.set_title("Do the same block pairs recur? (|M| rank correlation between cases)", fontsize=10)
    axx.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / f"{tag}_consistency.png", dpi=130); plt.close(fig)
    say(f"Figures: {', '.join(str(FIGS / f'{tag}_{n}.png') for n in ('M_heatmaps', 'M_by_context', 'distance_profile', 'sources_benders', 'cancellation', 'consistency'))}")


if __name__ == "__main__":
    main()
