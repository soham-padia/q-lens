"""Summarise the Q-robust pilot (src/q_robust.py) into results/tables/q_robust_results.md (+ summary.json in the run dir).

Every hypothesis test here has a behavioural outcome (KL / top-1 flips on held-out contexts, capital-fact accuracy,
continuation edit distance and fluency); |Q|/|J| is only ever the predictor. Fragility = the push dose alpha* at
which a measure crosses a threshold (log-interpolated on the alpha grid; censored at > 0.4), sign-averaged unless
stated. Abruptness = curvature of log KL against log alpha, and the gap between the actual KL and the KL of the
first-order (J-only) logits on the same held-out contexts.

Usage: python src/summarize_q_robust.py [--out $QLENS_OUT/q_robust_qwen3-8b] [--notes results/tables/q_robust_results.md]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q_robust as QR  # noqa: E402

A = np.array(QR.ALPHAS)
TAU = {"kl": (0.01, 0.1, 1.0), "flip": (0.05, 0.2, 0.5), "facts_err": (0.1, 0.5), "edit": (0.25, 0.5)}
PRIMARY = {"kl": 0.1, "flip": 0.2, "facts_err": 0.1, "edit": 0.5}
CLASSES = ("iso", "data", "svd", "sharp", "maxr", "ctrl")


def load_layer(d: Path):
    need = ("dirs", "fit", "ctrl", "held", "kl")
    if not all((d / f"{k}.pt").exists() for k in need):
        return None
    r = {k: torch.load(d / f"{k}.pt", weights_only=False) for k in need}
    for k in ("facts", "cont"):
        r[k] = torch.load(d / f"{k}.pt", weights_only=False) if (d / f"{k}.pt").exists() else None
    for k in ("gates", "quick"):
        r[k] = json.loads((d / f"{k}.json").read_text()) if (d / f"{k}.json").exists() else {}
    return r


def dose(y, tau, log_y):
    a, f = QR.cross_dose(A, y, tau, log_y)
    return a, f


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (tuple, list)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    ax = abs(x)
    if ax != 0 and (ax < 1e-2 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def ci_str(est_ci):
    est, (lo, hi) = est_ci
    f2 = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"  # noqa: E731
    return f"{f2(est)} [{f2(lo)}, {f2(hi)}]"


def analyse(l, R, n_boot):
    dirs, ctrl = R["dirs"], R["ctrl"]
    cls = list(dirs["cls"]) + ["ctrl"] * len(ctrl["U"])
    cls = np.array(cls)
    N = len(cls)
    fit = {k: torch.cat([R["fit"][k], ctrl["fit"][k]]).double().numpy() for k in ("J2", "Q2", "JQ")}
    held = {k: v.double().numpy() for k, v in R["held"].items() if torch.is_tensor(v)}
    Ch = held["J2"].shape[1]
    r_fit = np.sqrt(fit["Q2"].sum(1) / fit["J2"].sum(1))
    r_held = np.sqrt(held["Q2"].sum(1) / held["J2"].sum(1))
    r_held_ctx = np.sqrt(held["Q2"] / held["J2"])  # [N, Ch] per-context ratio
    j_fit = np.sqrt(fit["J2"].mean(1))
    q_fit = np.sqrt(fit["Q2"].mean(1))
    fisher = held["fisher"].mean(1)  # first-order KL coefficient on the held-jet contexts

    kl = R["kl"]["kl"].double().numpy()  # [N, nA, nS, C]
    fl = R["kl"]["flip"].double().numpy()
    kl_sm, fl_sm = kl.mean(-1).mean(-1), fl.mean(-1).mean(-1)  # [N, nA]
    kl_ws, fl_ws = kl.mean(-1).max(-1), fl.mean(-1).max(-1)
    # Taylor comparison on the held-jet contexts (same contexts for actual and predicted)
    kl_h = kl[..., :Ch].mean(-1)  # [N, nA, nS]
    kl_lin, kl_quad = held["kl_lin"].mean(1), held["kl_quad"].mean(1)
    fl_h, fl_lin = fl[..., :Ch].mean(-1), held["fl_lin"].mean(1)
    gap_lin = np.log(np.maximum(kl_h, 1e-30) / np.maximum(kl_lin, 1e-30)).mean(-1)  # [N, nA] sign-mean
    gap_quad = np.log(np.maximum(kl_h, 1e-30) / np.maximum(kl_quad, 1e-30)).mean(-1)
    flgap = (fl_h - fl_lin).mean(-1)
    curv = np.array([QR.loglog_curvature(A, kl_sm[i]) for i in range(N)])
    slopes = np.array([QR.loglog_slopes(A, kl_sm[i]) for i in range(N)])

    out = {"layer": l, "rho": float(dirs["rho"]), "N": N, "classes": {c: int((cls == c).sum()) for c in CLASSES},
           "gates": R["gates"]}
    meas = {"kl": kl_sm, "flip": fl_sm}
    meas_ws = {"kl": kl_ws, "flip": fl_ws}
    fct = R["facts"]
    if fct is not None and fct["cor"].numel():
        cor = fct["cor"].double().numpy()  # [N, nA, nS, F]
        meas["facts_err"] = 1 - cor.mean(-1).mean(-1)
        meas_ws["facts_err"] = 1 - cor.mean(-1).min(-1)
        kinds = np.array([it.get("kind", "") for it in fct["items"]])
        out["facts"] = {"n": int(cor.shape[-1]), "n_country": int((kinds == "country").sum()),
                        "acc_zero": float(fct["cor_zero"].float().mean())}
        if (kinds == "country").any():
            out["facts"]["country_err_a0.4_by_class"] = {
                c: float(1 - cor[cls == c][:, -1][..., kinds == "country"].mean()) for c in CLASSES if (cls == c).any()}
    cont = R["cont"]

    # ---------------- fragility doses
    def doses(M, key):
        res = {}
        for tau in TAU[key]:
            ds = np.array([dose(M[i], tau, key == "kl")[0] for i in range(len(M))])
            res[tau] = ds
        return res
    D = {k: doses(v, k) for k, v in meas.items()}
    Dws = {k: doses(v, k) for k, v in meas_ws.items()}
    frag = {k: -np.log(np.where(np.isinf(D[k][PRIMARY[k]]), 1.0, D[k][PRIMARY[k]])) for k in D}  # censored -> 0

    # ---------------- per-class table
    tab = {}
    for c in CLASSES:
        i = cls == c
        if not i.any():
            continue
        row = {"n": int(i.sum()), "ratio_fit": float(np.median(r_fit[i])), "ratio_held": float(np.median(r_held[i])),
               "J_fit": float(np.median(j_fit[i])), "Q_fit": float(np.median(q_fit[i])),
               "kl": [float(np.median(kl_sm[i, a])) for a in range(len(A))],
               "flip_a0.4": float(np.median(fl_sm[i, -1])), "gap_lin_a0.4": float(np.median(gap_lin[i, -1])),
               "gap_quad_a0.02": float(np.median(gap_quad[i, 0])), "curv": float(np.median(curv[i]))}
        for k in D:
            ds = D[k][PRIMARY[k]][i]
            row[f"dose_{k}"] = float(np.median(ds))
            row[f"cens_{k}"] = float(np.isinf(ds).mean())
        if "facts_err" in meas:
            row["facts_err_a0.4"] = float(np.median(meas["facts_err"][i, -1]))
        tab[c] = row
    out["by_class"] = tab

    # ---------------- transfer of the ratio: fit -> held-out contexts
    sp = lambda x, y: QR.spearman(x, y)  # noqa: E731
    out["transfer"] = {"pooled": QR.boot_ci(sp, [r_fit, r_held], n_boot),
                       "log_ratio_fit_minus_held_med": float(np.median(np.log(r_fit / r_held)))}
    for c in CLASSES:
        i = cls == c
        if i.sum() >= 6:
            out["transfer"][c] = QR.boot_ci(sp, [r_fit[i], r_held[i]], n_boot)
            out["transfer"][c + "_overfit"] = float(np.median(np.log(r_fit[i] / r_held[i])))
    # per-context spread of the held ratio (is |Q|/|J| a property of the direction or of the context?)
    lr = np.log(r_held_ctx)
    out["ratio_ctx_sd_over_dir_sd"] = float(lr.std(1).mean() / lr.mean(1).std())

    # ---------------- H1: ratio ranks directions by fragility (Spearman + bootstrap), vs |J| and partial on |J|
    h1 = {}
    sets = {"pooled": np.ones(N, bool), "iso": cls == "iso", "data": cls == "data", "iso+data": np.isin(cls, ["iso", "data"]),
            "no_maxr": cls != "maxr"}
    cont_metrics = {"logKL_a0.1": np.log(np.maximum(kl_sm[:, 2], 1e-30)), "logKL_a0.4": np.log(np.maximum(kl_sm[:, -1], 1e-30)),
                    "flip_a0.4": fl_sm[:, -1]}
    if "facts_err" in meas:
        cont_metrics["facts_err_a0.4"] = meas["facts_err"][:, -1]
    for k, v in frag.items():
        cont_metrics[f"-log dose_{k}"] = v
    for sname, i in sets.items():
        if i.sum() < 8:
            continue
        h1[sname] = {}
        for mname, y in cont_metrics.items():
            h1[sname][mname] = {
                "ratio_fit": QR.boot_ci(sp, [r_fit[i], y[i]], n_boot),
                "ratio_held": QR.boot_ci(sp, [r_held[i], y[i]], n_boot),
                "J_fit": QR.boot_ci(sp, [j_fit[i], y[i]], n_boot),
                "ratio_fit|J_fit": QR.boot_ci(lambda a, b, z: QR.partial_spearman(a, b, z), [r_fit[i], y[i], j_fit[i]], n_boot),
                "ratio_fit|fisher": QR.boot_ci(lambda a, b, z: QR.partial_spearman(a, b, z), [r_fit[i], y[i], fisher[i]], n_boot),
            }
    out["H1"] = h1

    # ---------------- H2: abruptness, top vs bottom quartile of fit ratio (pooled and iso+data)
    h2 = {}
    abr = {"curv_logKL": curv, "gap_lin_a0.4": gap_lin[:, -1], "gap_lin_a0.2": gap_lin[:, -2], "flipgap_a0.4": flgap[:, -1],
           "max_slope": slopes.max(1)}
    for sname in ("pooled", "iso+data", "no_maxr"):
        i = np.where(sets[sname])[0]
        if len(i) < 16:
            continue
        q1, q3 = np.quantile(r_fit[i], [0.25, 0.75])
        top, bot = i[r_fit[i] >= q3], i[r_fit[i] <= q1]
        h2[sname] = {"n_top": len(top), "n_bot": len(bot), "ratio_top_med": float(np.median(r_fit[top])),
                     "ratio_bot_med": float(np.median(r_fit[bot]))}
        for name, x in abr.items():
            def diff(xt, xb):
                return float(np.median(xt) - np.median(xb))
            rng = np.random.default_rng(1)
            est = diff(x[top], x[bot])
            bs = [diff(x[rng.choice(top, len(top))], x[rng.choice(bot, len(bot))]) for _ in range(n_boot)]
            h2[sname][name] = {"top_med": float(np.median(x[top])), "bot_med": float(np.median(x[bot])),
                               "diff": (est, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))),
                               "spearman_ratio": QR.boot_ci(sp, [r_fit[i], x[i]], n_boot)}
    out["H2"] = h2

    # ---------------- H3: matched controls (same fit sum_c |J_c u|^2, low ratio)
    nb = len(dirs["cls"])
    hi = ctrl["match_of"].numpy()
    ci = nb + np.arange(len(hi))
    dmatch = fit["J2"].sum(1)[ci] / fit["J2"].sum(1)[hi]
    rcon = r_fit[ci] / r_fit[hi]
    good = (np.abs(np.log(dmatch)) < math.log(1.5)) & (rcon < 0.5)
    h3 = {"n_pairs": len(hi), "n_good": int(good.sum()), "D_ctrl_over_D_high": [float(x) for x in dmatch],
          "ratio_ctrl_over_high": [float(x) for x in rcon], "high_classes": [str(cls[h]) for h in hi],
          "held_J_ctrl_over_high_med": float(np.median(np.sqrt(held["J2"].mean(1)[ci] / held["J2"].mean(1)[hi])))}
    pm = {"logKL_a0.1": cont_metrics["logKL_a0.1"], "logKL_a0.4": cont_metrics["logKL_a0.4"],
          "flip_a0.4": fl_sm[:, -1], "gap_lin_a0.4": gap_lin[:, -1], "curv_logKL": curv, "-log dose_kl": frag["kl"]}
    if "facts_err" in meas:
        pm["facts_err_a0.4"] = meas["facts_err"][:, -1]
    for subset, mask in (("all", np.ones(len(hi), bool)), ("good", good)):
        if mask.sum() < 3:
            continue
        h3[subset] = {}
        for name, x in pm.items():
            d = x[hi[mask]] - x[ci[mask]]
            h3[subset][name] = {"mean_diff_high_minus_ctrl": QR.boot_ci(np.mean, [d], n_boot),
                                "n_high_gt_ctrl": int((d > 0).sum()), "n": int(mask.sum())}
    out["H3"] = h3

    # ---------------- continuations (subset)
    if cont is not None:
        sub = np.array(cont["sub"])
        ed = cont["edit"].double().numpy().mean(-1).mean(-1)  # [n_sub, nA] sign- and ctx-mean
        nll = cont["nll"].double().numpy()  # [n_sub, nA, nS, C]
        base_nll = cont["nll_clean"].double().numpy()  # [C]
        disfl = (nll > base_nll[None, None, None, :] + 2.0).mean(-1).mean(-1)  # share of continuations > clean + 2 nats/token
        changed_fluent = ((cont["edit"].double().numpy() >= 0.5) & (nll <= base_nll + 2.0)).mean(-1).mean(-1)
        cc = {"n_sub": len(sub), "edit_zero": float(cont["edit_zero"].mean()), "nll_clean_mean": float(base_nll.mean()),
              "by_class": {}}
        for c in CLASSES:
            j = cls[sub] == c
            if j.any():
                cc["by_class"][c] = {"n": int(j.sum()), "edit": [float(np.median(ed[j, a])) for a in range(len(A))],
                                     "nll_excess_a0.4": float(np.median(nll[j, -1].mean((-1, -2)) - base_nll.mean())),
                                     "disfluent_a0.4": float(np.median(disfl[j, -1])),
                                     "changed_fluent_a0.4": float(np.median(changed_fluent[j, -1]))}
        if len(sub) >= 8:
            ed_dose = np.array([dose(ed[k], PRIMARY["edit"], False)[0] for k in range(len(sub))])
            fe = -np.log(np.where(np.isinf(ed_dose), 1.0, ed_dose))
            cc["spearman_ratio_fit_vs_edit_a0.4"] = QR.boot_ci(sp, [r_fit[sub], ed[:, -1]], n_boot)
            cc["spearman_J_fit_vs_edit_a0.4"] = QR.boot_ci(sp, [j_fit[sub], ed[:, -1]], n_boot)
            cc["spearman_ratio_fit_vs_-logdose_edit"] = QR.boot_ci(sp, [r_fit[sub], fe], n_boot)
            cc["partial_ratio_fit_vs_edit_a0.4|J"] = QR.boot_ci(lambda a, b, z: QR.partial_spearman(a, b, z),
                                                               [r_fit[sub], ed[:, -1], j_fit[sub]], n_boot)
        out["cont"] = cc

    # ---------------- sharp vs maxr vs svd: overlap of the families
    U = torch.cat([dirs["U"], ctrl["U"]]).double()
    fam = {c: U[torch.from_numpy(cls == c)] for c in ("svd", "sharp", "maxr") if (cls == c).any()}
    ov = {}
    for a_ in fam:
        for b_ in fam:
            if a_ < b_:
                ov[f"{a_}~{b_}"] = float((fam[a_] @ fam[b_].T).abs().max(1).values.median())
    out["family_overlap_median_max_cos"] = ov
    out["svd_ev_top"] = [float(x) for x in dirs["svd_ev"][:5]]
    out["sharp_ev_top"] = [float(x) for x in dirs["sharp_ev"][:5]]
    mh = dirs["maxr_hist"]
    r0 = torch.cat([h[0]["ratio"] for h in mh]).numpy()
    rb = torch.cat([torch.stack([x["ratio"] for x in h]).amax(0) for h in mh]).numpy()
    out["maxr_ascent"] = {"ratio_start_med": float(np.median(r0)), "ratio_best_med": float(np.median(rb)),
                          "gain_min": float((rb / r0).min()), "gain_med": float(np.median(rb / r0))}
    # ---------------- beyond first order: actual KL vs the Fisher prediction 0.5 alpha^2 F on the same contexts
    act = np.maximum(kl_h.mean(-1), 1e-30)  # [N, nA] sign-mean, held-jet contexts
    fisKL = 0.5 * A[None] ** 2 * fisher[:, None]
    g_f = np.log(act / fisKL)
    t_q = np.log(np.maximum(kl_quad.mean(-1), 1e-30) / np.maximum(kl_lin.mean(-1), 1e-30))
    fo = {"by_class": {}, "spearman": {}}
    for c in CLASSES:
        i = cls == c
        if i.any():
            fo["by_class"][c] = {"act_over_fisher": [float(np.median(g_f[i, a])) for a in (0, 2, 4)],
                                 "taylor_quad_over_lin": [float(np.median(t_q[i, a])) for a in (2, 4)],
                                 "act_over_quad": [float(np.median(gap_quad[i, a])) for a in (2, 4)]}
    for sname in ("pooled", "no_maxr", "iso+data"):
        i = sets[sname]
        fo["spearman"][sname] = {f"{pn} a={A[a]}": QR.boot_ci(sp, [x[i], g_f[i, a]], n_boot)
                                 for a in (2, 4) for pn, x in (("ratio_held", r_held), ("ratio_fit", r_fit))}
    dF = np.log(fisher[hi] / fisher[ci])
    fo["pairs"] = {"log_fisher_high_minus_ctrl": QR.boot_ci(np.mean, [dF], n_boot),
                   "log_KL_a0.1": QR.boot_ci(np.mean, [np.log(act[hi, 2] / act[ci, 2])], n_boot),
                   "act_over_fisher_a0.1": QR.boot_ci(np.mean, [g_f[hi, 2] - g_f[ci, 2]], n_boot),
                   "act_over_fisher_a0.4": QR.boot_ci(np.mean, [g_f[hi, 4] - g_f[ci, 4]], n_boot),
                   "n_pos_a0.1": int((g_f[hi, 2] > g_f[ci, 2]).sum()), "n_pos_a0.4": int((g_f[hi, 4] > g_f[ci, 4]).sum())}
    out["first_order"] = fo
    # small-alpha consistency: the jets describe the actual behaviour near 0 (a check, not a result)
    out["taylor_a0.02"] = {"median_abs_log_actual_over_quad": float(np.median(np.abs(gap_quad[:, 0]))),
                           "median_abs_log_actual_over_lin": float(np.median(np.abs(gap_lin[:, 0])))}
    out["noise"] = {"kl_zero_max": float(R["kl"]["kl_zero"].abs().max()), "flip_zero_mean": float(R["kl"]["flip_zero"].mean())}
    out["dose_worst_sign_vs_mean_med"] = {k: float(np.median(np.log(np.where(np.isinf(Dws[k][PRIMARY[k]]), 1, Dws[k][PRIMARY[k]])
                                                                    / np.where(np.isinf(D[k][PRIMARY[k]]), 1, D[k][PRIMARY[k]]))))
                                          for k in D}
    return out


def _kept_verdict(path):
    """A hand-written '## Verdict' section in an existing notes file survives re-generation."""
    p = Path(path)
    if not p.exists():
        return None
    txt = p.read_text()
    if "\n## Verdict" not in txt:
        return None
    v = txt[txt.index("\n## Verdict") + 1:]
    nxt = v.find("\n## ", 3)
    return v if nxt < 0 else v[:nxt + 1]


def write_md(res, path, outdir):
    verdict = _kept_verdict(path)
    L = []
    w = L.append
    w("# Q-robust pilot: does |Q|/|J| along a push direction predict behavioural fragility? (Qwen3-8B, fp32)")
    w("")
    w(f"Generated by `src/summarize_q_robust.py` from `{outdir}` (code `src/q_robust.py`).")
    w("")
    if verdict:  # hand-written, kept across reruns; it runs up to the next '## ' heading, which is Setup
        w(verdict.rstrip("\n"))
        w("")
    w("## Setup, checks and related work")
    w("")
    w("**Setup.** Push `s*alpha*rho_l*u` (unit u, s = +-1, alpha in {0.02, 0.05, 0.1, 0.2, 0.4}) at every valid "
      "position (16..T-2) of source block l (D1 convention). Predictor: `ratio = sqrt(sum_c |Q_c|^2 / sum_c |J_c u|^2)` "
      "with J, Q the exact first and second derivatives (nested jvp) of the centred logits at every valid position, "
      "on 8 fit contexts (fineweb docs 1500+). Q is the full second derivative, so along u the second-order logit term "
      "equals the first-order one at alpha = 2/ratio. All outcomes are behavioural and on 32 held-out contexts: next-token "
      "KL(clean||pushed) and top-1 flips at every valid position; ~100 few-shot capital facts (push at 16..T-1, "
      "accuracy at the last token); greedy 20-token continuations on 16 contexts for a direction subset (token edit "
      "distance from the clean continuation, NLL under the unpushed model). Classes: iso (random), data (centred "
      "residuals), svd (top right-singular vectors of J), sharp (top Hessian eigenvectors of the next-token loss "
      "w.r.t. the push), maxr (gradient ascent of the ratio), ctrl (same fit |J u|, low ratio, one per top-ratio "
      "direction). Fragility = dose alpha* at which a sign-averaged measure crosses a threshold "
      f"(primary: KL {PRIMARY['kl']}, flip {PRIMARY['flip']}, facts error {PRIMARY['facts_err']}, edit "
      f"{PRIMARY['edit']}; censored at > 0.4, ranked last). `-log dose` is larger for more fragile directions.")
    w("")
    w("**Score-hacking guard.** A Q-derived direction once moved a readout without changing the model's state. Here "
      "the ratio is only a predictor; every claim below is checked against KL, flips, fact accuracy or continuations.")
    w("")
    w("**Related work.** Azizi et al. 2026 (arXiv 2610.16084, NeurIPS 2026) compute a steering Hessian of the "
      "next-token loss with respect to a steering vector at one layer (Lanczos on finite-difference HVPs, 7-8B models) "
      "and find that sharp directions anti-steer. Their object is the curvature of one scalar (the loss) along the push; "
      "ours is the exact second-order term Q of the whole logit map (every position, every token), normalised by the "
      "first-order term, so |Q|/|J| measures nonlinearity rather than loss sharpness: a direction can be sharp for the "
      "loss because J u is large (Gauss-Newton part), with a small ratio. The `sharp` class here reproduces their "
      "directions with exact HVPs (forward-over-reverse) so the two can be compared on the same behavioural tests.")
    w("")
    for l, r in sorted(res.items()):
        w(f"## Source layer {l} (rho = {fmt(r['rho'], 1)}, N = {r['N']} directions)")
        w("")
        g = r.get("gates", {})
        w(f"Gates: tail vs full {fmt(g.get('tail_vs_full_rel'))}; residual jets vs so_core J {fmt(g.get('resid_J_rel'))}, "
          f"Q {fmt(g.get('resid_Q_rel'))}; best FD rel. error J {fmt(g.get('fd_J_rel_min'))}, Q {fmt(g.get('fd_Q_rel_min'))}; "
          f"Taylor KL at alpha 0.02 median |log actual/(J+Q/2)| {fmt(r['taylor_a0.02']['median_abs_log_actual_over_quad'])}; "
          f"zero-push KL {fmt(r['noise']['kl_zero_max'])}, flip floor {fmt(r['noise']['flip_zero_mean'])}.")
        w("")
        w("| class | n | ratio fit | ratio held | abs J fit | abs Q fit | KL a=.02/.05/.1/.2/.4 | flip a=.4 | facts err a=.4 | "
          "dose KL (cens) | dose flip (cens) | dose facts (cens) | log KL/KL_lin a=.4 | curv log KL |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c, t in r["by_class"].items():
            w(f"| {c} | {t['n']} | {fmt(t['ratio_fit'])} | {fmt(t['ratio_held'])} | {fmt(t['J_fit'])} | {fmt(t['Q_fit'])} | "
              f"{' / '.join(fmt(x, 3) for x in t['kl'])} | {fmt(t['flip_a0.4'])} | {fmt(t.get('facts_err_a0.4'))} | "
              f"{fmt(t.get('dose_kl'))} ({fmt(t.get('cens_kl'), 2)}) | {fmt(t.get('dose_flip'))} ({fmt(t.get('cens_flip'), 2)}) | "
              f"{fmt(t.get('dose_facts_err'))} ({fmt(t.get('cens_facts_err'), 2)}) | {fmt(t['gap_lin_a0.4'])} | {fmt(t['curv'])} |")
        w("")
        w("Medians per class. abs J, abs Q: per-context RMS of |J u| and |Q(u,u)| in centred-logit units at alpha = 1. "
          "KL in nats per position.")
        w("")
        tr = r["transfer"]
        w(f"**Transfer (fit -> held-out ratio).** Spearman pooled {ci_str(tr['pooled'])}"
          + "".join(f"; {c} {ci_str(tr[c])} (median log fit/held {fmt(tr[c + '_overfit'], 2)})" for c in CLASSES if c in tr)
          + f". Per-context SD of log ratio / across-direction SD: {fmt(r['ratio_ctx_sd_over_dir_sd'], 2)}.")
        w("")
        w("**H1: does the ratio rank directions by fragility?** Spearman [95% bootstrap CI]; `|J` and `|fisher` are partial "
          "Spearman with fit |J u| or the held-out first-order KL coefficient regressed out.")
        w("")
        for sname, d in r["H1"].items():
            w(f"*{sname}*")
            w("")
            w("| outcome | ratio fit | ratio held | abs J fit | ratio fit, J out | ratio fit, fisher out |")
            w("|---|---|---|---|---|---|")
            for mname, v in d.items():
                w(f"| {mname} | {ci_str(v['ratio_fit'])} | {ci_str(v['ratio_held'])} | {ci_str(v['J_fit'])} | "
                  f"{ci_str(v['ratio_fit|J_fit'])} | {ci_str(v['ratio_fit|fisher'])} |")
            w("")
        w("**H2: are top-ratio directions more abrupt?** Top vs bottom quartile of fit ratio; diff of medians [CI].")
        w("")
        for sname, d in r["H2"].items():
            w(f"*{sname}* (n top {d['n_top']}, bottom {d['n_bot']}; ratio {fmt(d['ratio_top_med'])} vs {fmt(d['ratio_bot_med'])})")
            w("")
            w("| abruptness | top | bottom | diff [CI] | Spearman with ratio |")
            w("|---|---|---|---|---|")
            for name, v in d.items():
                if isinstance(v, dict):
                    w(f"| {name} | {fmt(v['top_med'])} | {fmt(v['bot_med'])} | {ci_str(v['diff'])} | {ci_str(v['spearman_ratio'])} |")
            w("")
        h3 = r["H3"]
        w(f"**H3: matched controls.** {h3['n_pairs']} pairs (high-ratio classes: "
          f"{', '.join(sorted(set(h3['high_classes'])))}); {h3['n_good']} good (fit D within x1.5 and ratio < half). "
          f"Median held-out |J| ctrl/high {fmt(h3['held_J_ctrl_over_high_med'], 2)}; D ctrl/high "
          f"{fmt(float(np.median(h3['D_ctrl_over_D_high'])), 2)}, ratio ctrl/high {fmt(float(np.median(h3['ratio_ctrl_over_high'])), 2)}.")
        w("")
        for subset in ("all", "good"):
            if subset in h3:
                w(f"*{subset} pairs*: " + "; ".join(
                    f"{name} {ci_str(v['mean_diff_high_minus_ctrl'])} ({v['n_high_gt_ctrl']}/{v['n']} high > ctrl)"
                    for name, v in h3[subset].items()))
                w("")
        fo = r["first_order"]
        w("**Beyond first order.** `log act/F`: log of actual KL over the first-order (Fisher) prediction 0.5 a^2 F on the "
          "same held-out contexts (0 = behaviour is fully first order). `log quad/lin`: what the J+Q/2 Taylor logits predict "
          "Q adds to KL. `log act/quad`: actual vs that prediction.")
        w("")
        w("| class | log act/F a=.02 / .1 / .4 | log quad/lin a=.1 / .4 | log act/quad a=.1 / .4 |")
        w("|---|---|---|---|")
        for c, t in fo["by_class"].items():
            w(f"| {c} | {' / '.join(f'{x:+.3f}' for x in t['act_over_fisher'])} | "
              f"{' / '.join(f'{x:+.2f}' for x in t['taylor_quad_over_lin'])} | {' / '.join(f'{x:+.2f}' for x in t['act_over_quad'])} |")
        w("")
        for sname, d in fo["spearman"].items():
            w(f"- Spearman(ratio, log act/F), {sname}: " + "; ".join(f"{k} {ci_str(v)}" for k, v in d.items()))
        pp = fo["pairs"]
        w(f"- Matched pairs (high - ctrl): log held-out Fisher {ci_str(pp['log_fisher_high_minus_ctrl'])}, log KL a=.1 "
          f"{ci_str(pp['log_KL_a0.1'])}; Fisher-normalised log act/F a=.1 {ci_str(pp['act_over_fisher_a0.1'])} "
          f"({pp['n_pos_a0.1']}/{h3['n_pairs']} > 0), a=.4 {ci_str(pp['act_over_fisher_a0.4'])} ({pp['n_pos_a0.4']}/{h3['n_pairs']} > 0).")
        w("")
        if "cont" in r:
            cc = r["cont"]
            w(f"**Continuations** ({cc['n_sub']} directions x 16 contexts; zero-push edit {fmt(cc['edit_zero'])}, "
              f"clean NLL {fmt(cc['nll_clean_mean'], 2)} nats/token).")
            w("")
            w("| class | n | edit a=.02/.05/.1/.2/.4 | NLL excess a=.4 | disfluent a=.4 | changed & fluent a=.4 |")
            w("|---|---|---|---|---|---|")
            for c, t in cc["by_class"].items():
                w(f"| {c} | {t['n']} | {' / '.join(fmt(x, 2) for x in t['edit'])} | {fmt(t['nll_excess_a0.4'], 2)} | "
                  f"{fmt(t['disfluent_a0.4'], 2)} | {fmt(t['changed_fluent_a0.4'], 2)} |")
            w("")
            for k in ("spearman_ratio_fit_vs_edit_a0.4", "spearman_J_fit_vs_edit_a0.4", "spearman_ratio_fit_vs_-logdose_edit",
                      "partial_ratio_fit_vs_edit_a0.4|J"):
                if k in cc:
                    w(f"- {k}: {ci_str(cc[k])}")
            w("")
        if "facts" in r:
            f = r["facts"]
            w(f"**Facts**: {f['n']} items ({f['n_country']} countries), zero-push accuracy {fmt(f['acc_zero'], 3)}. "
              "Country-only error at alpha 0.4 by class: "
              + ", ".join(f"{c} {fmt(v, 2)}" for c, v in f.get("country_err_a0.4_by_class", {}).items()) + ".")
            w("")
        w(f"**Families.** maxr ascent ratio {fmt(r['maxr_ascent']['ratio_start_med'])} -> {fmt(r['maxr_ascent']['ratio_best_med'])} "
          f"(min gain x{fmt(r['maxr_ascent']['gain_min'], 2)}). Median max |cos| between families: "
          + ", ".join(f"{k} {fmt(v, 2)}" for k, v in r["family_overlap_median_max_cos"].items())
          + f". Top J singular values^2 {fmt(r['svd_ev_top'][:3])}; top loss-Hessian eigenvalues {fmt(r['sharp_ev_top'][:3])}. "
          f"Worst-sign vs sign-mean dose (median log ratio): "
          + ", ".join(f"{k} {fmt(v, 2)}" for k, v in r["dose_worst_sign_vs_mean_med"].items()) + ".")
        w("")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "q_robust_qwen3-8b"))
    ap.add_argument("--notes", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables", "q_robust_results.md"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    res = {}
    for d in sorted(Path(a.out).glob("L*")):
        R = load_layer(d)
        if R is None:
            print(f"skip {d}: incomplete")
            continue
        res[int(d.name[1:])] = analyse(int(d.name[1:]), R, a.n_boot)
        print(f"analysed {d}")
    if not res:
        sys.exit(f"no complete layers under {a.out}")
    (Path(a.out) / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res, a.notes, a.out)
    print(f"wrote {a.notes} and {Path(a.out) / 'summary.json'}")


if __name__ == "__main__":
    main()
