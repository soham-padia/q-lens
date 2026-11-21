"""Summarise AtP-Q (src/atp_q.py): accuracy of AtP, EAP-IG, exact and stochastic J+Q against actual activation
patching, AtP false negatives, |Q/2|/|J| as a detector of AtP error, partial patches, and everything against cost.

    python src/summarize_atp_q.py [OUT_DIR] [--smoke]
writes results/tables/atp_q_results.md (atp_q_smoke.md with --smoke) and figures in results/tables/figs_atp_q/.
Effects are normalised per pair by D = m_clean - m_corrupt before pooling over pairs.
"""
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
SMOKE = "--smoke" in sys.argv
OUT = argv[0] if argv else os.path.join(os.environ.get("QLENS_OUT", "outputs"), "atp_q_qwen3-8b") + ("_smoke" if SMOKE else "")
FIG = os.path.join(ROOT, "results", "tables", "figs_atp_q")
os.makedirs(FIG, exist_ok=True)
SUF = "_smoke" if SMOKE else ""
TASKS = [t for t in ("ioi", "fact") if glob.glob(f"{OUT}/{t}_*.pt")]
R = {t: [torch.load(f, weights_only=False) for f in sorted(glob.glob(f"{OUT}/{t}_*.pt"))] for t in TASKS}
assert R, f"no pair files in {OUT}"
D_MIN = 0.5  # pairs whose corruption barely moves m are dropped (normalising by D would blow them up)
DROPPED = {t: sum(abs(r["m_clean"] - r["m_corrupt"]) <= D_MIN for r in R[t]) for t in TASKS}
R = {t: [r for r in R[t] if abs(r["m_clean"] - r["m_corrupt"]) > D_MIN] for t in TASKS}
TAU_LARGE, REL_BAD, REL_FLOOR = 0.05, 0.5, 0.01  # large effect (x D); "wrong" = rel error > 0.5; relevance floor
GROUPS = {"heads": (0,), "mlp": (1,), "comp": (0, 1), "res": (2,)}
KS = [k for k in (8, 32, 128)]


def estimates(r):
    """method -> [S] estimate (nan where unavailable)."""
    S = len(r["actual"])
    e = {"AtP": r["atp"]}
    for m in sorted(r["ig"]):
        e[f"EAP-IG{m}"] = r["ig"][m]
    fam_sites = [np.nonzero(r["fam"] == f)[0] for f in (0, 1)]
    Kmax = min(v.shape[0] for v in r["stoch"].values())
    for K in KS:
        if K > Kmax:
            continue
        q = np.full(S, np.nan)
        for f, qh in r["stoch"].items():
            q[fam_sites[f]] = qh[:K].mean(0)
        q[r["null"]] = 0.0
        e[f"J+Q stoch K={K}"] = r["atp"] + q / 2
    if r.get("ones"):
        q = np.full(S, np.nan)
        for f, v in r["ones"].items():
            q[fam_sites[f]] = v
        q[r["null"]] = 0.0
        e["J+Q all-ones"] = r["atp"] + q / 2
    e["J+Q exact"] = r["J"] + r["Q"] / 2
    return e


def cost_nominal(method, n, r):
    """Forward-pass equivalents per pair for n live sites of one family (forward 1, backward 2, jvp 2,
    nested jvp 4, reverse-over-forward HVP 6; clean + corrupt forwards 2 where delta is needed)."""
    if method == "actual":
        return 1 + n
    if method == "AtP":
        return 4
    if method.startswith("EAP-IG"):
        return 2 + 3 * int(method[6:])
    if method.startswith("J+Q stoch"):
        return 4 + 6 * int(method.split("=")[1])
    if method == "J+Q all-ones":
        return 10
    if method == "J+Q exact":
        return 2 + 4 * n
    raise ValueError(method)


def cost_measured(method, fam, r):
    """Measured seconds per pair attributable to one family (proportional split of shared stages)."""
    t, c = r["times"], r["counts"]
    nf = c["fam_live"][fam]
    if method == "actual":
        return t["patch"] * nf / max(c["live"], 1)
    if method == "AtP":
        return t["atp"]
    if method.startswith("EAP-IG"):
        return t["ig"] * int(method[6:]) / c["ig_rows"]
    if method.startswith("J+Q stoch"):
        return t["atp"] + t.get(f"stoch{fam}", np.nan) * int(method.split("=")[1]) / c["K"]
    if method == "J+Q all-ones":
        return t["atp"] + t.get(f"ones{fam}", np.nan)
    if method == "J+Q exact":
        return t["exact"] * nf / max(c["exact"], 1)
    raise ValueError(method)


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = rankdata(score)
    return float((rk[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


lines = [f"# AtP vs second order (J+Q) against activation patching{' (SMOKE)' if SMOKE else ''}", "",
         f"Source `{OUT}`, written by `src/summarize_atp_q.py`. Model Qwen3-8B, fp32 with TF32 off, sdpa math kernel.",
         "Noising: clean base run, each site patched with its corrupt activation; m = logit(clean answer) − "
         "logit(corrupt answer) at the last position; effects normalised per pair by D = m_clean − m_corrupt.",
         "Sites: **heads** = per-head slice of o_proj's input at the last position (36 × 32), **mlp** = MLP output "
         "at the last position (36), **comp** = heads + mlp, **res** = block output × position (layers "
         f"{R[TASKS[0]][0]['res_layers']}; positions before the first differing token have δ = 0 and are dropped).",
         "",
         "**Prior work (read before citing anything here as new).** Zhang & Wang 2026, *When Attribution Patching "
         "Lies* (arXiv 2606.09899) already correct AtP with the exact per-component Hessian term ½δᵀHδ (one HVP per "
         "component, up to 9B: GPT-2, Pythia, Qwen2.5, Gemma-2, Llama-3.1-8B; IOI, greater-than, factual recall), "
         "use the reliability score R̃ = |δᵀHδ|/(2|Δ̂|), i.e. our |½Q|/|J| (AUC 0.70 on Pythia-410M), compare "
         "multi-step HVP with per-head IG at matched cost, and find the network-level curvature 22–66× the "
         "activation's own. So exact per-site J+Q and the |½Q|/|J| flag are **replications** here (on Qwen3-8B, "
         "with exact Q at every site and FD gates). Complementary parts: (a) the stochastic all-site diagonal "
         "(Rademacher probes; cost K HVPs for every site at once, compared with EAP-IG and the deterministic "
         "all-ones HVP at equal cost), which that paper does not have (its all-at-once HVP keeps the cross terms); "
         "(c) the exact Q-lens split of each site's Q over the downstream blocks, which it does not do (its "
         "layerwise appendix is a heuristic sum that ignores mixed terms); plus partial patches λδ and "
         "residual-stream × position sites.",
         ""]

# ---------------------------------------------------------------- data overview and gates
lines += ["## Pairs and gates", "",
          f"Pairs dropped for |D| ≤ {D_MIN}: {DROPPED}.", "",
          "| task | pairs | T | m_clean median | m_corrupt median | max atp_vs_J | max one-hot HVP rel | "
          "max full-res abs | FD-of-jvp Q_rel (worst hard site) | value-FD Q_rel (worst) | FD J_rel (worst) | stoch frac abs z<3 (comp/res) | s/pair |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in TASKS:
    rs = R[t]
    g = [r["gates"] for r in rs]
    fdq = max(x["Q_rel"] for gg in g for x in gg["fd"] if x.get("hard", True))
    fdj = max(x["J_rel"] for gg in g for x in gg["fd"] if x.get("hard", True))
    fdqj = max(x.get("Qj_rel", np.nan) for gg in g for x in gg["fd"] if x.get("hard", True))
    zc = np.mean([gg["stoch_z"][0]["frac_abs_z_lt3"] for gg in g if 0 in gg["stoch_z"]])
    zr = np.mean([gg["stoch_z"][1]["frac_abs_z_lt3"] for gg in g if 1 in gg["stoch_z"]])
    lines.append(f"| {t} | {len(rs)} | {int(np.median([r['T'] for r in rs]))} | "
                 f"{np.median([r['m_clean'] for r in rs]):.2f} | {np.median([r['m_corrupt'] for r in rs]):.2f} | "
                 f"{max(x['atp_vs_J'] for x in g):.1e} | {max(x['hvp_onehot_rel'] for x in g):.1e} | "
                 f"{max(max(x['full_res_abs'].values()) for x in g):.1e} | {fdqj:.1e} | {fdq:.1e} | {fdj:.1e} | "
                 f"{zc:.2f} / {zr:.2f} | {np.median([r['secs'] for r in rs]):.0f} |")
lines += ["", "FD-of-jvp = Q as a central difference of the first-order jvp tangent (Richardson; the hard gate, since "
          "fp32 second differences of values drown the small per-head deltas). Value FD = central differences of the final hidden state (Richardson over h, h/2; best of h = 0.2, 0.05) "
          "against the nested jvp (hard gate sites: the comp sites at the 25th and 50th percentile of AtP relative error among the top 60 by |A|, and the deepest-layer "
          "residual site at the last position). The stochastic z-score is (mean of 128 draws − exact Q)/SE; ≈0.99 "
          "expected for an unbiased estimator with a light-tailed spread.", ""]

# ---------------------------------------------------------------- main accuracy table
cost_rows = {}
for t in TASKS:
    rs = R[t]
    E = [estimates(r) for r in rs]
    methods = list(E[0].keys())
    lines += [f"## {t}: accuracy against actual patching", "",
              "Pooled over pairs and sites (normalised effects). rel L1 = Σ|est − A| / Σ|A|. top-k = mean over "
              "pairs of |top-k by |est| ∩ top-k by |A|| / k. 'beats AtP' = share of relevant sites "
              f"(max(|A|, |AtP|) ≥ {REL_FLOOR} D) where |est − A| < |AtP − A|.", ""]
    for gname, kinds in GROUPS.items():
        lines += [f"**{gname}**", "",
                  "| method | n | Pearson | Spearman | per-pair Spearman (median) | rel L1 | top-10 | top-50 | "
                  "beats AtP | cost F (nominal) | cost s (measured) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for meth in methods:
            xs, ys, sp, t10, t50, rel_n, rel_d, beat = [], [], [], [], [], 0.0, 0.0, []
            n_live, cmeas = [], []
            for r, e in zip(rs, E):
                sel = np.isin(r["kind"], kinds) & ~r["null"] & np.isfinite(e[meth]) & np.isfinite(e["J+Q exact"])
                if sel.sum() < 3:
                    continue
                D = r["m_clean"] - r["m_corrupt"]
                a, y, atp = r["actual"][sel] / D, e[meth][sel] / D, r["atp"][sel] / D
                xs.append(a); ys.append(y)
                sp.append(spearmanr(a, y)[0])
                for k, acc in ((10, t10), (50, t50)):
                    if len(a) >= 2 * k:
                        acc.append(len(set(np.argsort(-np.abs(a))[:k]) & set(np.argsort(-np.abs(y))[:k])) / k)
                rel_n += np.abs(y - a).sum(); rel_d += np.abs(a).sum()
                rel = np.maximum(np.abs(a), np.abs(atp)) >= REL_FLOOR
                beat += list((np.abs(y - a) < np.abs(atp - a))[rel])
                fam = 1 if kinds == (2,) else 0
                n_live.append(int(sel.sum()))
                cmeas.append(cost_measured(meth, fam, r) if gname in ("comp", "res") else np.nan)
            if not xs:
                continue
            X, Y = np.concatenate(xs), np.concatenate(ys)
            nF = cost_nominal(meth, int(np.median(n_live)), rs[0])
            row = dict(n=len(X), pear=pearsonr(X, Y)[0], spear=spearmanr(X, Y)[0], pp=np.median(sp),
                       rel=rel_n / rel_d, t10=np.mean(t10) if t10 else np.nan, t50=np.mean(t50) if t50 else np.nan,
                       beat=np.mean(beat) if beat else np.nan, F=nF, s=np.nanmedian(cmeas))
            cost_rows[(t, gname, meth)] = row
            lines.append(f"| {meth} | {row['n']} | {fmt(row['pear'])} | {fmt(row['spear'])} | {fmt(row['pp'])} | "
                         f"{fmt(row['rel'])} | {fmt(row['t10'], 2)} | {fmt(row['t50'], 2)} | "
                         f"{fmt(row['beat'] if meth != 'AtP' else np.nan, 2)} | "
                         f"{row['F']:.0f} | {fmt(row['s'], 2)} |")
        nact = int(np.median([(np.isin(r['kind'], kinds) & ~r['null']).sum() for r in rs]))
        if gname in ("comp", "res"):
            fam = 1 if gname == "res" else 0
            sa = np.median([cost_measured("actual", fam, r) for r in rs])
            cost_rows[(t, gname, "actual")] = dict(F=cost_nominal("actual", nact, rs[0]), s=sa)
            lines.append(f"| *actual patching* | | 1 | 1 | 1 | 0 | 1 | 1 | | {cost_nominal('actual', nact, rs[0]):.0f} "
                         f"| {sa:.2f} |")
        lines.append("")

    # ---------------------------------------------------------------- false negatives
    lines += [f"### {t}: AtP false negatives", "",
              f"Large effect: |A| ≥ {TAU_LARGE} D. AtP false negative: large and |AtP − A| > {REL_BAD}|A| (missed or "
              f"wrong sign). Recovered by a method: |est − A| ≤ {REL_BAD}|A|.", "",
              "| group | large sites | AtP FN | " + " | ".join(f"recovered by {m}" for m in methods if m != "AtP") + " |",
              "|---|---|---|" + "---|" * (len(methods) - 1)]
    for gname, kinds in GROUPS.items():
        nl, fn, rec = 0, 0, {m: 0 for m in methods}
        for r, e in zip(rs, E):
            D = r["m_clean"] - r["m_corrupt"]
            sel = np.isin(r["kind"], kinds) & ~r["null"] & np.isfinite(e["J+Q exact"])
            a = r["actual"] / D
            large = sel & (np.abs(a) >= TAU_LARGE)
            bad = large & (np.abs(r["atp"] / D - a) > REL_BAD * np.abs(a))
            nl += large.sum(); fn += bad.sum()
            for m in methods:
                rec[m] += (bad & (np.abs(e[m] / D - a) <= REL_BAD * np.abs(a))).sum()
        lines.append(f"| {gname} | {nl} | {fn} | " + " | ".join(f"{rec[m]}" for m in methods if m != "AtP") + " |")
    lines.append("")

    # ---------------------------------------------------------------- detector
    lines += [f"### {t}: does |½Q|/|J| flag the sites where AtP is wrong?", "",
              f"Sites with max(|A|, |AtP|) ≥ {REL_FLOOR} D; label = |AtP − A| > {REL_BAD}·max(|A|, |AtP|). "
              "AUC of each score as a detector (0.5 = chance).", "",
              "| group | sites | AtP wrong | AUC |½Q|/|J| exact | AUC stoch K=32 | AUC stoch K=128 | AUC |½Q|/D exact | "
              "AUC ‖δ‖ |", "|---|---|---|---|---|---|---|---|"]
    for gname, kinds in GROUPS.items():
        sc = {k: [] for k in ("rx", "r32", "r128", "qa", "dn")}
        lab = []
        for r, e in zip(rs, E):
            D = r["m_clean"] - r["m_corrupt"]
            a, j = r["actual"] / D, r["atp"] / D
            sel = np.isin(r["kind"], kinds) & ~r["null"] & np.isfinite(r["Q"]) & \
                (np.maximum(np.abs(a), np.abs(j)) >= REL_FLOOR)
            lab.append((np.abs(j - a) > REL_BAD * np.maximum(np.abs(a), np.abs(j)))[sel])
            sc["rx"].append((np.abs(r["Q"] / 2) / np.maximum(np.abs(r["J"]), 1e-30))[sel])
            for K, key in ((32, "r32"), (128, "r128")):
                k = f"J+Q stoch K={K}"
                sc[key].append((np.abs(e[k] - r["atp"]) / np.maximum(np.abs(r["atp"]), 1e-30))[sel]
                               if k in e else np.full(sel.sum(), np.nan))
            sc["qa"].append(np.abs(r["Q"] / 2 / D)[sel])
            sc["dn"].append(r["dnorm"][sel])
        L = np.concatenate(lab)
        A = {k: auc(np.concatenate(v), L) if np.isfinite(np.concatenate(v)).all() else np.nan for k, v in sc.items()}
        lines.append(f"| {gname} | {len(L)} | {int(L.sum())} | {fmt(A['rx'])} | {fmt(A['r32'])} | {fmt(A['r128'])} | "
                     f"{fmt(A['qa'])} | {fmt(A['dn'])} |")
    lines.append("")

    # ---------------------------------------------------------------- partial patches
    lams = sorted(set(rs[0]["lams"]) | {1.0})
    lines += [f"### {t}: partial patches λδ (where the Taylor series stops working)", "",
              "Subset per pair: the top-20 and 20 random live sites of each family by |A|. rel L1 of λJ and "
              "λJ + λ²Q/2 against the actual λ-patch; 'J+Q wins' = share of sites (|A_λ| ≥ 0.01 D) where J+Q is "
              "closer; 'A_λ/(λA_1)' median = 1 for a linear site.", "",
              "| family | λ | rel L1 J | rel L1 J+Q | J+Q wins | median A_λ/(λ A_1) |", "|---|---|---|---|---|---|"]
    for fam, fname in ((0, "comp"), (1, "res")):
        for lam in lams:
            nj = njq = den = 0.0
            wins, lin = [], []
            for r in rs:
                s = np.asarray(r["partial_sites"])
                s = s[r["fam"][s] == fam]
                if not len(s):
                    continue
                D = r["m_clean"] - r["m_corrupt"]
                pos = {int(x): i for i, x in enumerate(r["partial_sites"])}
                a = (r["actual"][s] if lam == 1.0 else np.array([r["partial"][lam][pos[int(x)]] for x in s])) / D
                pj, pjq = lam * r["J"][s] / D, (lam * r["J"][s] + lam ** 2 * r["Q"][s] / 2) / D
                nj += np.abs(pj - a).sum(); njq += np.abs(pjq - a).sum(); den += np.abs(a).sum()
                ok = np.abs(a) >= REL_FLOOR
                wins += list((np.abs(pjq - a) < np.abs(pj - a))[ok])
                a1 = r["actual"][s] / D
                okl = np.abs(a1) >= REL_FLOOR
                lin += list((a / (lam * a1))[okl])
            if den:
                lines.append(f"| {fname} | {lam} | {nj / den:.3f} | {njq / den:.3f} | {fmt(np.mean(wins), 2)} | "
                             f"{fmt(np.median(lin), 2)} |")
    lines.append("")

# ---------------------------------------------------------------- Q-lens split of the AtP error's second-order part
for t in TASKS:
    rs = [r for r in R[t] if len(r.get("split_sites", []))]
    if not rs:
        continue
    lines += [f"## {t}: where the second-order part of the AtP error lives (exact Q-lens split)", "",
              "Top-10 comp and top-5 res sites by |A| per pair. Q_s = Σ_k ⟨g_k, D²f_k[u_k, u_k]⟩ + final-norm term "
              "+ own, exact by the chain rule (own = the MLP half of the head's layer; 0 for mlp/res sites, "
              "checked). 'first' = the first nonlinearity the patch meets (heads: own MLP half; mlp/res: the next "
              "block). Medians over sites.", "",
              "| group | sites | |½Q|/|A−J| | |A−J−½Q|/|A−J| | first share | others / first | top block share | "
              "blocks for 90% | final-norm share | |Σ|/Σ|·| |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for gname, kinds in (("heads", (0,)), ("mlp", (1,)), ("res", (2,))):
        rows = []
        for r in rs:
            L = r["split"].shape[1] - 1
            for b, s in enumerate(r["split_sites"]):
                if r["kind"][s] not in kinds:
                    continue
                tm = np.nan_to_num(r["split"][b].copy())
                own = r["split_own"][b] if r["kind"][s] == 0 else 0.0
                parts = np.concatenate([[own], tm])  # [own, block 0..L-1, norm]
                l = int(r["layer"][s])
                first_i = 0 if r["kind"][s] == 0 else (l + 2 if l + 1 < L else L + 1)
                ab = np.abs(parts)
                tot = ab.sum()
                if tot <= 0:
                    continue
                srt = np.sort(ab)[::-1]
                err = r["actual"][s] - r["J"][s]
                rows.append((abs(r["Q"][s] / 2) / max(abs(err), 1e-30), abs(err - r["Q"][s] / 2) / max(abs(err), 1e-30),
                             ab[first_i] / tot, (tot - ab[first_i]) / max(ab[first_i], 1e-30), srt[0] / tot,
                             int(np.searchsorted(np.cumsum(srt) / tot, 0.9) + 1), ab[-1] / tot,
                             abs(parts.sum()) / tot))
        if rows:
            M = np.median(np.array(rows), 0)
            lines.append(f"| {gname} | {len(rows)} | {M[0]:.2f} | {M[1]:.2f} | {M[2]:.2f} | {M[3]:.1f} | {M[4]:.2f} | "
                         f"{M[5]:.0f} | {M[6]:.2f} | {M[7]:.2f} |")
    lines.append("")

# ---------------------------------------------------------------- figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["AtP", "EAP-IG16", "J+Q all-ones", "J+Q stoch K=32", "J+Q exact"]
    for t in TASKS:
        rs = R[t]
        E = [estimates(r) for r in rs]
        fig, ax = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.4))
        for gi, (gname, kinds) in enumerate((("comp", (0, 1)), ("res", (2,)))):
            for mi, meth in enumerate(show):
                X, Y = [], []
                for r, e in zip(rs, E):
                    if meth not in e:
                        continue
                    D = r["m_clean"] - r["m_corrupt"]
                    sel = np.isin(r["kind"], kinds) & ~r["null"] & np.isfinite(e[meth])
                    X.append(r["actual"][sel] / D); Y.append(e[meth][sel] / D)
                a = ax[gi, mi]
                if X:
                    X, Y = np.concatenate(X), np.concatenate(Y)
                    a.scatter(X, Y, s=4, alpha=0.4, color="#3b6ea5", linewidths=0)
                    lim = np.nanpercentile(np.abs(np.concatenate([X, Y])), 99.8) * 1.1
                    a.plot([-lim, lim], [-lim, lim], color="#999", lw=0.8)
                    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
                a.set_title(f"{gname}: {meth}", fontsize=9)
                a.set_xlabel("actual / D", fontsize=8)
                a.set_ylabel("estimate / D", fontsize=8)
                a.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"scatter_{t}{SUF}.png"), dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for gi, gname in enumerate(("comp", "res")):
        a = ax[gi]
        for t, mk in zip(TASKS, ("o", "s")):
            pts = [(v["F"], v["spear"], m) for (tt, g, m), v in cost_rows.items() if tt == t and g == gname
                   and "spear" in v]
            for F, s, m in pts:
                col = {"AtP": "#444"}.get(m, "#c0504d" if m.startswith("EAP") else
                                          ("#3b6ea5" if "stoch" in m else "#2e8b57"))
                a.scatter(F, s, marker=mk, color=col)
                a.annotate(m.replace("J+Q ", ""), (F, s), fontsize=6, xytext=(3, 2), textcoords="offset points")
            if (t, gname, "actual") in cost_rows:
                a.axvline(cost_rows[(t, gname, "actual")]["F"], color="#999", ls=":", lw=0.8)
        a.set_xscale("log")
        a.set_xlabel("cost per pair (forward-pass equivalents, nominal)")
        a.set_ylabel("Spearman with actual patching")
        a.set_title(f"{gname} sites (o = ioi, s = fact; dotted = actual patching)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"accuracy_vs_cost{SUF}.png"), dpi=130)
    plt.close(fig)
    lines += ["## Figures", ""] + [f"![]({'figs_atp_q'}/scatter_{t}{SUF}.png)" for t in TASKS] + \
        [f"![](figs_atp_q/accuracy_vs_cost{SUF}.png)", ""]
except Exception as ex:  # figures are optional
    lines += [f"(figures skipped: {ex})", ""]

lines += ["## Cost model", "",
          "Nominal forward-pass equivalents per pair for one family of n live sites: actual patching 1 + n; AtP 4 "
          "(clean forward + backward, corrupt forward; all families at once); EAP-IG m: 2 + 3m (all families); "
          "J+Q stochastic K: 4 + 6K per family (one reverse-over-forward HVP ≈ jvp 2 + backward 4); J+Q exact "
          "2 + 4n (one nested jvp per site, an upper bound: tangents start at the site's layer). Measured seconds "
          "are per pair on the job's GPU with the batching used (patching 127 sites per forward), split "
          "proportionally between families; batched and unbatched passes are not comparable one to one.", ""]
dst = os.path.join(ROOT, "results", "tables", f"atp_q_{'smoke' if SMOKE else 'results'}.md")
open(dst, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", dst)
