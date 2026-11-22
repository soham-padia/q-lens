"""v8.4 rows for paper/claims.csv: the Lean pilot (E1-E4), the responsibility maps (v2), the finite-difference precision
check, the cost curve and the numeric-intermediate control. Called from claims.py (after every older row, so RNG and
RNG2 sequences, and every older CI, are unchanged). New bootstraps use their own generator, RNG3.

Called from claims.py, or alone (python paper/claims_v84.py OUT.csv writes the Step 5 rows only).

Sources: results/step5/, numbers-only copies of the raw outputs written by paper/step5_inputs.py (no text, no
token ids, no file names from the research library; the file column keeps the raw output's name):
  lean_v1/summary.json             E1-E4 CIs as written by src/summarize_lean_qlens.py (pre-registered bootstraps:
                                   cluster over contexts, seed 0; E2/E3 B = 2000, E1/E4 B = 1000). Transcribed as is.
  lean_v1/<model>/E23_<cor>.json   E2/E3 point statistics (per-layer medians, fractions), recomputed here with the
                                   summarizer's own functions and gates (Amendment 2 included).
  resp_map_v2/resp_v2_*.json       maps: Spearman, top-pair and AND counts, detection AUC, cost (RNG3 bootstraps over
                                   contexts, B = 2000). Each span carries one flag, nl: the span ends a line.
  resp_map_v2/fd_precision/        finite-difference precision check (fp32 vs fp64; Q_fp32_vs_fp64.json holds the
                                   relative difference of the two Q tensors, Q_float{32,64}.pt, per pair-block).
  resp_map_v2/cost_curve/          cost curve at S = 12 / 50 / 200 lines.
  orderops_layers/ctrl_order-ops.json   numeric-intermediate control (both controls; RNG3 bootstrap over intermediates).
"""
import ast
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1] / "results" / "step5"
SRC = Path(__file__).resolve().parents[1] / "src"
B = 2000
RNG3 = np.random.default_rng(20261122)
NAN = float("nan")
MOD = {"P": "Goedel-Prover-V2-8B", "B": "Qwen3-8B-Base", "C": "Qwen3-8B"}


def run(ROWS):
    def add(claim, model, setting, quantity, v, lo=NAN, hi=NAN, n="", f=""):
        ROWS.append(dict(claim=claim, model=model, setting=setting, quantity=quantity, value=float(v), lo=float(lo),
                         hi=float(hi), n=n, file=f))

    # ------------------------------------------------------------ Lean E1-E4 (summary.json, pre-registered CIs)
    S = json.loads((W / "lean_v1" / "summary.json").read_text())
    fS = "lean_v1/summary.json"
    for key, d in S["E1"]["own"]["iso"].items():
        m, cor = key.split("_")
        for q in ("level", "slope", "within", "power"):
            add("L1", MOD[m], f"E1 {cor} iso", q, *d[q], n=8, f=fS)
    for key, d in S["E1_diff"]["own"]["iso"].items():
        for q in ("level", "slope", "power"):
            add("L1d", "lean", f"E1 diff {key}", q, *d[q], n=8, f=fS)
    for key, d in S["E2"].items():
        m, cor = key.split("_")
        for q in ("D", "D_cal"):
            add("L2", MOD[m], f"E2 {cor}", q, *d[q], n=d["n"], f=fS)
        for q in ("med_eJ", "med_eJQ", "med_eJcal", "med_eJQcal"):
            add("L2", MOD[m], f"E2 {cor}", q, d[q], n=d["n"], f=fS)
    for cor, v in S["E2_PB"].items():
        add("L2", "P - B", f"E2 {cor}", "D(P) - D(B)", *v, f=fS)
    for key, d in S["E3"].items():
        m, cor = key.split("_")
        for q in ("a_dcos", "b_rel", "share"):
            add("L3", MOD[m], f"E3 {cor}", q, *d[q], n=d["n_pairs"], f=fS)
    for cor, d in S["E3_PB"].items():
        for q, v in d.items():
            add("L3", "P - B", f"E3 {cor}", q, *v, f=fS)
    for key, d in S["E4"].items():
        m, cor = key.split("_")
        for q in ("J1", "J4", "hQ1", "hQ4", "T24", "d4", "d1"):
            add("L4", MOD[m], f"E4 {cor}", q, *d["rates"][q], n=d["n_rows"], f=fS)

    # ------------------------------------------------------------ Lean E2/E3 point statistics from the raw outputs
    sys.path.insert(0, str(SRC))
    argv, sys.argv = sys.argv, sys.argv[:1]
    import summarize_lean_qlens as SL  # noqa: E402  (its W defaults to lean_v1)
    sys.argv = argv
    for m, tag in (("P", "goedel-prover-v2-8b"), ("B", "qwen3-8b-base")):
        for cor in ("RH", "ML"):
            p = W / "lean_v1" / tag / f"E23_{cor}.json"
            d = json.loads(p.read_text())
            g2p = W / "lean_v1" / tag / f"gate2_{cor}.json"
            G2 = json.loads(g2p.read_text()) if g2p.exists() else None
            f = f"lean_v1/{tag}/E23_{cor}.json"
            layers = [str(x) for x in d["layers"]]
            g = SL.gates_e2(d["cases"], layers, G2)
            ok = [x for x in layers if g[x]["pass"]]
            rows = SL.e2_errors(d["cases"], ok)
            cJ = SL.calibrated(rows, ok, 5)
            add("L2p", MOD[m], f"E2 {cor}", "frac J+Q closer", np.mean([r[3] < r[2] for r in rows]), n=len(rows), f=f)
            add("L2p", MOD[m], f"E2 {cor}", "frac J+Q no root", np.mean([not math.isfinite(r[6]) for r in rows]),
                n=len(rows), f=f)
            add("L2p", MOD[m], f"E2 {cor}", "med r_edit", SL.med([r[8] for r in rows]), n=len(rows), f=f)
            add("L2p", MOD[m], f"E2 {cor}", "layers passing", len(ok), n=len(layers), f=f)
            for x in ok:
                ix = [i for i, r in enumerate(rows) if r[1] == x]
                st_ = f"E2 {cor} L{x}"
                add("L2p", MOD[m], st_, "med err J", SL.med([rows[i][2] for i in ix]), n=len(ix), f=f)
                add("L2p", MOD[m], st_, "med err J+Q", SL.med([rows[i][3] for i in ix]), n=len(ix), f=f)
                add("L2p", MOD[m], st_, "med err J cal", SL.med([cJ[i] for i in ix]), n=len(ix), f=f)
                add("L2p", MOD[m], st_, "med r_edit", SL.med([rows[i][8] for i in ix]), n=len(ix), f=f)
            g3 = SL.gates_e3(d["pairs"], layers, G2)
            ok3 = [x for x in layers if g3[x]["pass"]]
            r3 = [(i, pp["ci"], x, pp["L"][x]) for i, pp in enumerate(d["pairs"]) for x in ok3 if x in pp["L"]]
            for q in ("cos", "cos_mis", "cos_arb", "rand_cos", "rel", "rand_rel"):
                add("L3p", MOD[m], f"E3 {cor}", f"med {q}", SL.med([r[3][q] for r in r3]), n=len(r3), f=f)
            for s_ in ("0.5", "0.1"):
                for q in ("cos_s", "rel_s"):
                    add("L3p", MOD[m], f"E3 {cor}", f"med {q} s={s_}",
                        SL.med([r[3][q][s_] for r in r3 if s_ in r[3][q]]), n=len(r3), f=f)
            for x in ok3:
                sel = [r for r in r3 if r[2] == x]
                add("L3p", MOD[m], f"E3 {cor} L{x}", "med cos", SL.med([r[3]["cos"] for r in sel]), n=len(sel), f=f)
                add("L3p", MOD[m], f"E3 {cor} L{x}", "med rel", SL.med([r[3]["rel"] for r in sel]), n=len(sel), f=f)

    # ------------------------------------------------------------ responsibility maps v2
    files = {c: W / "resp_map_v2" / f"resp_v2_{c}.json" for c in ("ML", "FW", "MLnull", "FWnull")}
    files["RH"] = W / "resp_map_v2" / "resp_v2_RH.json"
    R = {c: json.loads(p.read_text()) for c, p in files.items() if p.exists()}

    def rank(xs):  # as src/summarize_resp_v2.py
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

    def val(p, m):
        return sum(p["pQ"]) / len(p["pQ"]) if m == "pQ" else p[m]

    def cross(c, P):  # drop the two pieces of one line cut at 16 tokens (span i does not end a line)
        sp = c["spans"]
        return [p for p in P if not (p["j"] == p["i"] + 1 and not sp[p["i"]]["nl"])]

    def auc(pos, neg):
        if not pos or not neg:
            return NAN
        return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))

    def bmed(xs):
        xs = np.asarray(xs, float)
        bs = [np.median(xs[RNG3.integers(0, len(xs), len(xs))]) for _ in range(B)]
        return float(np.median(xs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def bauc(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        bs = [auc(list(pos[RNG3.integers(0, len(pos), len(pos))]), list(neg[RNG3.integers(0, len(neg), len(neg))]))
              for _ in range(B)]
        return auc(list(pos), list(neg)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    METH = ["Q", "pQ", "Q3"]
    for c, r in R.items():
        f = f"resp_v2_{c}.json"
        for tg in ("tstar", "line"):
            for x in [str(y) for y in r["layers"]]:
                sp = {m: [] for m in METH}
                sign = {m: 0 for m in METH}
                top3 = {m: 0 for m in METH}
                fnd = {m: 0 for m in METH}
                n = 0
                for ctx in r["contexts"]:
                    P = ctx["L"][x][tg]["pairs"]
                    if any(p["Q3"] is None for p in P):
                        continue
                    n += 1
                    D = [p["D2"] for p in P]
                    X = cross(ctx, P)
                    byD = sorted(X, key=lambda p: -abs(p["D2"]))[:3]
                    ta = max(X, key=lambda p: p["D2"])
                    for m in METH:
                        sp[m].append(spear([val(p, m) for p in P], D))
                        t = max(X, key=lambda p: abs(val(p, m)))
                        sign[m] += (val(t, m) > 0) == (t["D2"] > 0)
                        top3[m] += t in byD
                        fnd[m] += ta in sorted(X, key=lambda p: -val(p, m))[:3]
                for m in METH:
                    stn = f"map {c} {tg} L{x}"
                    add("R", m, stn, "median Spearman", *bmed(sp[m]), n=n, f=f)
                    add("R", m, stn, "top pair sign ok", sign[m], n=n, f=f)
                    add("R", m, stn, "top pair in exact top 3", top3[m], n=n, f=f)
                    add("R", m, stn, "AND pair found", fnd[m], n=n, f=f)
    for a, b in (("ML", "MLnull"), ("FW", "FWnull")):
        if a not in R or b not in R:
            continue
        for tg in ("tstar", "line"):
            for x in [str(y) for y in R[a]["layers"]]:
                def scores(cr, m):
                    out = []
                    for ctx in R[cr]["contexts"]:
                        X = cross(ctx, ctx["L"][x][tg]["pairs"])
                        if m != "D2" and any(p["Q3"] is None for p in X):
                            continue
                        out.append(max(abs(p["D2"] if m == "D2" else val(p, m)) for p in X))
                    return out
                for m in ["D2"] + METH:
                    sa, sb = scores(a, m), scores(b, m)
                    add("Rd", m, f"detect {a} vs {b} {tg} L{x}", "AUC", *bauc(sa, sb), n=f"{len(sa)}/{len(sb)}",
                        f=f"resp_v2_{a}.json, resp_v2_{b}.json")
    for c, r in R.items():
        for tg in ("tstar", "line"):
            T = {k: [] for k in ("exact", "Q_jvp", "pathQ", "Q_hvp", "Q3_hvp")}
            Ss, chk = [], []
            for ctx in r["contexts"]:
                Ss.append(len(ctx["spans"]))
                for x in ctx["L"]:
                    row = ctx["L"][x][tg]
                    for k in T:
                        if k in row["times"]:
                            T[k].append(row["times"][k])
                    chk.append(row["hvp_vs_jvp"])
            for k, v in T.items():
                add("Rc", k, f"cost {c} {tg}", "median s per context-block", float(np.median(v)), n=len(v),
                    f=f"resp_v2_{c}.json")
            add("Rc", "lines", f"cost {c} {tg}", "median S", float(np.median(Ss)), n=len(Ss), f=f"resp_v2_{c}.json")
            add("Rc", "Q_hvp vs Q_jvp", f"cost {c} {tg}", "max rel diff", max(chk), n=len(chk), f=f"resp_v2_{c}.json")

    # named v1 failures (numbers only; Mathlib and FineWeb-Edu contexts only, never the RH file):
    # signed rank among all pairs, 1 = most positive, as src/summarize_resp_v2.py
    NAMEDP = [("ML", 14, "20", (8, 10)), ("ML", 14, "20", (4, 10)), ("FW", 4, "20", (2, 12))]
    for c, ci, x, (i, j) in NAMEDP:
        if c not in R or ci >= len(R[c]["contexts"]):
            continue
        P = R[c]["contexts"][ci]["L"][x]["tstar"]["pairs"]
        p = next(q for q in P if (q["i"], q["j"]) == (i, j))
        stn = f"named {c} c{ci} tstar L{x} pair [{i},{j}]"
        add("Rn", "pairs", stn, "count", len(P), n=len(P), f=f"resp_v2_{c}.json")
        add("Rn", "D2", stn, "value", p["D2"], n=len(P), f=f"resp_v2_{c}.json")
        add("Rn", "D2 rank by |D2|", stn, "rank", 1 + sorted(P, key=lambda q: -abs(q["D2"])).index(p), n=len(P),
            f=f"resp_v2_{c}.json")
        for m in METH:
            add("Rn", m, stn, "value", val(p, m), n=len(P), f=f"resp_v2_{c}.json")
            add("Rn", m, stn, "signed rank", 1 + sorted(P, key=lambda q: -val(q, m)).index(p), n=len(P),
                f=f"resp_v2_{c}.json")
        for lam, y in zip(R[c]["lams"], p["pQ"]):
            add("Rn", "pQ point", stn, f"Q at eta={lam}", y, n=len(P), f=f"resp_v2_{c}.json")

    # ------------------------------------------------------------ finite-difference precision
    fd = {dt: json.loads((W / "resp_map_v2" / "fd_precision" / f"fd_{dt}.json").read_text())
          for dt in ("float32", "float64")}
    ff = "resp_map_v2/fd_precision/fd_{float32,float64}.json"
    for dt, d in fd.items():
        grid = d["grid"]
        mins, argm, last, bestcos = [], [], [], []
        for x, prs in d["layers"].items():
            for p in prs:
                e = p["rel_err"]
                k = int(np.argmin(e))
                mins.append(e[k]); argm.append(grid[k]); last.append(e[-1]); bestcos.append(max(p["cos"]))
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "min rel err (min over pairs)", min(mins), n=len(mins), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "min rel err (max over pairs)", max(mins), n=len(mins), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "argmin step (min)", min(argm), n=len(mins), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "argmin step (max)", max(argm), n=len(mins), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "rel err at s=1e-4 (min)", min(last), n=len(last), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "rel err at s=1e-4 (max)", max(last), n=len(last), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "best cos (min over pairs)", min(bestcos), n=len(mins), f=ff)
        add("FD", dt, "fd check ML c0 4 pairs x 3 blocks", "best cos (max over pairs)", max(bestcos), n=len(mins), f=ff)
    # ||Q32 - Q64|| / ||Q64|| per pair-block, computed from Q_float{32,64}.pt by paper/step5_inputs.py
    rd = json.loads((W / "resp_map_v2" / "fd_precision" / "Q_fp32_vs_fp64.json").read_text())["rel_diff"]
    add("FD", "fp32 vs fp64", "Q itself, 12 pair-blocks", "rel diff (min)", min(rd), n=len(rd), f="Q_float{32,64}.pt")
    add("FD", "fp32 vs fp64", "Q itself, 12 pair-blocks", "rel diff (max)", max(rd), n=len(rd), f="Q_float{32,64}.pt")

    # ------------------------------------------------------------ cost curve
    cc = json.loads((W / "resp_map_v2" / "cost_curve" / "cost_curve.json").read_text())
    fc = "resp_map_v2/cost_curve/cost_curve.json"
    for row in cc["rows"]:
        stn = f"cost curve S={row['S']} L{row['layer']}"
        add("K", "T tokens", stn, "count", row["T"], n=row["P"], f=fc)
        for k in ("exact", "Q_jvp", "pQ", "Q_hvp", "Q3"):
            v = row.get(f"{k}_s", NAN)
            how = row.get(f"{k}_how", "OOM" if row.get(k) == "OOM" else "")
            add("K", k, stn, f"seconds ({how})", v, n=row["P"], f=fc)
        if isinstance(row.get("Q_hvp_s"), float):
            add("K", "exact / Q_hvp", stn, "ratio", row["exact_s"] / row["Q_hvp_s"], n=row["P"], f=fc)
        add("K", "pQ / exact", stn, "ratio", row["pQ_s"] / row["exact_s"], n=row["P"], f=fc)
        add("K", "Q_hvp", stn, "peak GB", row.get("Q_hvp_peak_gb", NAN), n=row["P"], f=fc)

    # ------------------------------------------------------------ numeric-intermediate control (order-ops, both controls)
    D = json.loads((W / "orderops_layers" / "ctrl_order-ops.json").read_text())
    LS = sorted(x for x in D["layers_done"] if 24 <= x <= 56)
    Rr, items, dig = D["res"], D["items"], D["digit_tokens"]
    fn = "orderops_layers/ctrl_order-ops.json"

    def ints(it):
        v = it["intermediates"]
        return ast.literal_eval(v) if isinstance(v, str) else v

    def best(arm, i):  # best rank over layers 24-56 of each of the ten digit tokens
        return [min(Rr[str(x)][arm]["digits"][i][q] for x in LS) for q in range(10)]

    rows = []
    for i, it in enumerate(items):
        for w in ints(it):
            if not w[0].isdigit():
                continue
            c = dig.index(w[0])
            a, b = best("J d", i), best("J d + ½Q", i)
            gain = [math.log(a[q]) - math.log(b[q]) for q in range(10)]
            pd = {ch for ch in it["prompt"] if ch.isdigit()}
            used = set(it["target"]) | {h for x in ints(it) for h in x if h.isdigit()}
            prom = [dig.index(ch) for ch in pd if ch not in used]
            unused = [q for q in range(10) if dig[q] not in (pd | used)]
            place = {arm: (lambda bb: 1 + sum(bb[q] < bb[c] for q in range(10)))(best(arm, i))
                     for arm in ("logit lens", "J d", "J d + ½Q", "½Q(d,d)")}
            rows.append(dict(g=gain[c], prom=np.mean([gain[q] for q in prom]) if prom else None,
                             unused=np.mean([gain[q] for q in unused]) if unused else None, inp=w[0] in pd, place=place))

    def bmean(xs):
        xs = np.asarray(xs, float)
        bs = [xs[RNG3.integers(0, len(xs), len(xs))].mean() for _ in range(B)]
        return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for name, key, flt in (("prompt digits", "prom", lambda r: True), ("unused digits", "unused", lambda r: True),
                           ("prompt digits, intermediate not in prompt", "prom", lambda r: not r["inp"])):
        sel = [r for r in rows if r[key] is not None and flt(r)]
        dd = [r["g"] - r[key] for r in sel]
        stn = f"order-ops vs {name}"
        add("N", D["model"].split("/")[-1], stn, "gain intermediate", np.mean([r["g"] for r in sel]), n=len(sel), f=fn)
        add("N", D["model"].split("/")[-1], stn, "gain comparison", np.mean([r[key] for r in sel]), n=len(sel), f=fn)
        add("N", D["model"].split("/")[-1], stn, "difference", *bmean(dd), n=len(sel), f=fn)
        add("N", D["model"].split("/")[-1], stn, "intermediate gains more", sum(x > 0 for x in dd), n=len(sel), f=fn)
        add("N", D["model"].split("/")[-1], stn, "intermediate gains less", sum(x < 0 for x in dd), n=len(sel), f=fn)
    for arm in ("logit lens", "J d", "J d + ½Q", "½Q(d,d)"):
        add("N", D["model"].split("/")[-1], "order-ops place among ten digits", arm, np.mean([r["place"][arm] for r in rows]),
            n=len(rows), f=fn)


if __name__ == "__main__":  # the Step 5 rows alone, in the format of results/claims.csv
    import csv
    ROWS = []
    run(ROWS)
    out = sys.argv[1] if len(sys.argv) > 1 else "claims_step5.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        for r in ROWS:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(ROWS)} rows to {out}")
