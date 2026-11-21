#!/usr/bin/env python3
"""Summarize S2.json (second_order.py --stages S2): single layer vs many layers, for reading and steering.

  python src/summarize_s2.py [S2.json ...] [--md out.md]   (default: every second_order_v1/s2/*/S2.json)

Every selection (best layer, best window, best band) is made on one half of the items/trials and scored on the
other half, then the halves swap (2-fold cross-validation, split by index parity). CIs: paired bootstrap over items
or trials (2000 resamples, seed 0).
"""
import json
import math
import os
import random
import sys
from pathlib import Path

md_out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
paths = [Path(x) for x in sys.argv[1:] if not x.startswith("--") and x != md_out] or \
    sorted((Path(os.environ.get("QLENS_OUT", "outputs")) / "second_order_v1" / "s2").glob("*/S2.json"))
D = {}
for fp in paths:  # shards: lists are concatenated, everything else comes from the first file that has it
    d = json.loads(fp.read_text())
    for k, v in d.items():
        if k in ("onehop", "twohop") and k in D:
            D[k] += v
        elif k == "secs":
            D["secs"] = round(D.get("secs", 0) + v, 1)
        else:
            D.setdefault(k, v)
path = ", ".join(map(str, paths))
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def boot(n, stat, B=2000, seed=0):
    """Percentile CI of stat(index list) over resampled indices."""
    rnd = random.Random(seed)
    vals = sorted(stat([rnd.randrange(n) for _ in range(n)]) for _ in range(B))
    vals = [v for v in vals if v == v]
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def cv_pick(n, score_of, cands):
    """2-fold CV: for each fold pick the candidate maximizing mean score on the other fold.
    Returns per-item test-fold choice (list of candidate per item) and the two picks."""
    folds = [[i for i in range(n) if i % 2 == f] for f in (0, 1)]
    choice, picks = [None] * n, []
    for f in (0, 1):
        train, test = folds[1 - f], folds[f]
        best = max(cands, key=lambda c: mean(score_of(i, c) for i in train))
        picks.append(best)
        for i in test:
            choice[i] = best
    return choice, picks


# ---------------------------------------------------------------- R: reading
if "read" in D:
    R = D["read"]; wins = [tuple(w) for w in R["windows"]]; LL = R["lens_layers"]
    single = [k for k, (a, b) in enumerate(wins) if a == b]
    multi = [k for k, (a, b) in enumerate(wins) if b > a]
    allk = len(wins) - 1
    say("## Reading: J-lens rank of the latent intermediate (last position)")
    say("Window = mean of per-layer z-scored J-lens logits over layers [a, b]. Best layer / window chosen on one half,")
    say("scored on the other (2-fold CV). min-over-layers = the J-lens paper's per-item best layer (no selection penalty);")
    say("its control is the same statistic for another item's intermediate.")
    say()
    say("| set | n | method | picked | top-1 | top-10 | median rank |")
    say("|---|---|---|---|---|---|---|")
    for s_ in ("twohop", "multihop"):
        it = [x for x in R["items"] if x["set"] == s_]
        n = len(it)
        if not n:
            continue
        sc = lambda i, k: (it[i]["rank"][k] < 10) - 1e-3 * math.log1p(it[i]["rank"][k])  # noqa: E731
        cs, ps = cv_pick(n, sc, single)
        cm, pm = cv_pick(n, sc, multi)
        rows = {"best single layer": ([it[i]["rank"][cs[i]] for i in range(n)], ", ".join(f"L{LL[wins[k][0]]}" for k in ps)),
                "  control intermediate, same layer": ([it[i]["rank_ctrl"][cs[i]] for i in range(n)], "as above"),
                "best multi-layer window": ([it[i]["rank"][cm[i]] for i in range(n)],
                                            ", ".join(f"L{LL[wins[k][0]]}-{LL[wins[k][1]]}" for k in pm)),
                "  control intermediate, same window": ([it[i]["rank_ctrl"][cm[i]] for i in range(n)], "as above"),
                "all-layer mean": ([x["rank"][allk] for x in it], "all"),
                "min over layers (oracle)": ([min(x["rank"][k] for k in single) for x in it], "per item"),
                "  control intermediate, min over layers": ([min(x["rank_ctrl"][k] for k in single) for x in it], "per item")}
        for name, (rk, pick) in rows.items():
            srt = sorted(rk)
            say(f"| {s_} | {n} | {name} | {pick} | {mean(r < 1 for r in rk):.2f} | {mean(r < 10 for r in rk):.2f} | {srt[n // 2]} |")
        a = [it[i]["rank"][cs[i]] < 10 for i in range(n)]; b = [it[i]["rank"][cm[i]] < 10 for i in range(n)]
        lo, hi = boot(n, lambda ix: mean(b[i] - a[i] for i in ix))
        say(f"| {s_} | | multi − single (top-10) | | | {mean(b) - mean(a):+.2f} [{lo:+.2f}, {hi:+.2f}] | |")
    say()

# ---------------------------------------------------------------- O: one-hop band clamp
if D.get("onehop"):
    O = D["onehop"]; Ls = D["layers"]
    bk = list(O[0]["band"]); rk = list(O[0]["reswap"])
    say("## One-hop swaps (J-lens paper's flexible-generalization trials), top-1 success")
    say("Clamp = coordinates set to the swapped clean-pass values at every band layer. Reswap = S1's version (swaps the")
    say("current state again at every layer).")
    say()
    say("| category | n | clean top-1 = old | best single layer @1 (CV) | best band clamp @1 (CV) | paper band clamp @1 / @2 | reswap @1 / @2 |")
    say("|---|---|---|---|---|---|---|")
    for cat in sorted({r["cat"] for r in O}) + ["all"]:
        rs = [r for r in O if cat in ("all", r["cat"])]
        n = len(rs)
        cs, ps = cv_pick(n, lambda i, l: rs[i]["single"][str(l)]["out"][0]["succ"], Ls)
        cb, pb = cv_pick(n, lambda i, k: rs[i]["band"][k]["out"][0]["succ"], bk)
        pk = bk[-1]
        say(f"| {cat} | {n} | {mean(r['clean']['keep'] for r in rs):.2f} | "
            f"{mean(rs[i]['single'][str(cs[i])]['out'][0]['succ'] for i in range(n)):.2f} (L{'/'.join(map(str, ps))}) | "
            f"{mean(rs[i]['band'][cb[i]]['out'][0]['succ'] for i in range(n)):.2f} ({'/'.join(pb)}) | "
            f"{mean(r['band'][pk]['out'][0]['succ'] for r in rs):.2f} / {mean(r['band'][pk]['out'][1]['succ'] for r in rs):.2f} | "
            f"{mean(r['reswap'][rk[0]][0]['succ'] for r in rs):.2f} / {mean(r['reswap'][rk[0]][1]['succ'] for r in rs):.2f} |")
    say()
    say("Per layer / band, all categories, success @1 / @2:")
    say("single: " + "  ".join(f"L{l} {mean(r['single'][str(l)]['out'][0]['succ'] for r in O):.2f}/"
                              f"{mean(r['single'][str(l)]['out'][1]['succ'] for r in O):.2f}" for l in Ls))
    say("band:   " + "  ".join(f"{k} {mean(r['band'][k]['out'][0]['succ'] for r in O):.2f}/"
                              f"{mean(r['band'][k]['out'][1]['succ'] for r in O):.2f}" for k in bk))
    say()

# ---------------------------------------------------------------- T: two-hop swaps
if D.get("twohop"):
    T = D["twohop"]; ok = [r for r in T if "single" in r]; Ls = D["layers"]
    LAMS, BL, SL = D["lams"], D["band_lams"], D["spread_lams"]
    li, bi = LAMS.index(1.0), BL.index(1.0)
    bk = list(ok[0]["band"]) if ok else []
    say("## Two-hop swaps (latent intermediate, e.g. 'the capital of the country famous for croissants')")
    for fn in sorted({r["func"] for r in T}):
        rs = [r for r in T if r["func"] == fn]
        say(f"- {fn}: model answers the two-hop prompt correctly on {sum('single' in r for r in rs)}/{len(rs)} trials")
    n = len(ok)
    if n:
        say()
        say(f"Top-1 success over the {n} trials the model answers correctly:")
        say()
        say("| method | " + " | ".join(f"λ={x}" for x in LAMS) + " |")
        say("|---|" + "---|" * len(LAMS))
        for l in Ls:
            say(f"| single L{l} | " + " | ".join(f"{mean(r['single'][str(l)]['out'][k]['succ'] for r in ok):.2f}" for k in range(len(LAMS))) + " |")
        say(f"| random dir (all layers) | " + " | ".join(
            f"{mean(r['single'][str(l)]['rand'][k]['succ'] for r in ok for l in Ls):.2f}" for k in range(len(LAMS))) + " |")
        say()
        say("| band | " + " | ".join(f"clamp λ={x}" for x in BL) + " | " + " | ".join(f"spread λ={x}" for x in SL)
            + " | ctrl pair λ=1 succ / keep | ctrl λ=2 succ / keep |")
        say("|---|" + "---|" * (len(BL) + len(SL) + 2))
        for k in bk:
            say(f"| {k} | " + " | ".join(f"{mean(r['band'][k]['out'][j]['succ'] for r in ok):.2f}" for j in range(len(BL)))
                + " | " + " | ".join(f"{mean(r['band'][k]['spread'][j]['succ'] for r in ok):.2f}" for j in range(len(SL)))
                + f" | {mean(r['band'][k]['ctrl'][0]['succ'] for r in ok):.2f} / {mean(r['band'][k]['ctrl'][0]['keep'] for r in ok):.2f}"
                + f" | {mean(r['band'][k]['ctrl'][1]['succ'] for r in ok):.2f} / {mean(r['band'][k]['ctrl'][1]['keep'] for r in ok):.2f} |")
        say()
        # CV: best single layer vs best band at λ = 1 (and the max over λ)
        cs, ps = cv_pick(n, lambda i, l: ok[i]["single"][str(l)]["out"][li]["succ"], Ls)
        cb, pb = cv_pick(n, lambda i, k: ok[i]["band"][k]["out"][bi]["succ"], bk)
        a = [ok[i]["single"][str(cs[i])]["out"][li]["succ"] for i in range(n)]
        b = [ok[i]["band"][cb[i]]["out"][bi]["succ"] for i in range(n)]
        lo, hi = boot(n, lambda ix: mean(b[i] - a[i] for i in ix))
        say(f"- **λ = 1, CV-picked:** best single layer {mean(a):.2f} (L{'/'.join(map(str, ps))}), best band clamp "
            f"{mean(b):.2f} ({'/'.join(pb)}); band − single {mean(b) - mean(a):+.2f} [{lo:+.2f}, {hi:+.2f}].")
        # success within a push budget: per trial, smallest L1 budget at which a method succeeds
        def min_budget_single(r, l):
            e = r["single"][str(l)]
            return min([lam * e["beta"] for lam, o in zip(LAMS, e["out"]) if o["succ"]], default=math.inf)

        def min_budget_band(r, k, kind):
            e = r["band"][k]
            outs, bud = (e["out"], e["budget"]) if kind == "clamp" else (e["spread"], e["spread_budget"])
            return min([bb["L1"] for o, bb in zip(outs, bud) if o["succ"]], default=math.inf)
        say("- **Success within a total push budget** (L1 = sum over layers of |Δ_l|/|h_l|; per trial the smallest budget at")
        say("  which the method flips the answer; method picked by CV on the fraction flipped within budget 0.2):")
        Bs = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
        fams = {"single layer": (Ls, lambda r, c: min_budget_single(r, c)),
                "band clamp": (bk, lambda r, c: min_budget_band(r, c, "clamp")),
                "band spread": (bk, lambda r, c: min_budget_band(r, c, "spread"))}
        say()
        say("| method | picked | " + " | ".join(f"≤{x}" for x in Bs) + " |")
        say("|---|---|" + "---|" * len(Bs))
        for name, (cands, f) in fams.items():
            ch, pk = cv_pick(n, lambda i, c: f(ok[i], c) <= 0.2, cands)
            mb = [f(ok[i], ch[i]) for i in range(n)]
            say(f"| {name} | {'/'.join(map(str, pk))} | " + " | ".join(f"{mean(x <= B for x in mb):.2f}" for B in Bs) + " |")
        say()
        # depth: intermediate (bridge) swap vs answer swap, per layer at λ = 1 and 2
        say("- **Bridge swap vs answer swap by layer** (success @1 / @2; the J-lens paper's control: an intermediate swap")
        say("  should work at earlier layers than swapping the answer itself):")
        say("  bridge: " + "  ".join(f"L{l} {mean(r['single'][str(l)]['out'][LAMS.index(1.0)]['succ'] for r in ok):.2f}/"
                                     f"{mean(r['single'][str(l)]['out'][LAMS.index(2.0)]['succ'] for r in ok):.2f}" for l in Ls))
        say("  answer: " + "  ".join(f"L{l} {mean(r['single'][str(l)]['answer_swap'][0]['succ'] for r in ok):.2f}/"
                                     f"{mean(r['single'][str(l)]['answer_swap'][1]['succ'] for r in ok):.2f}" for l in Ls))
        say()

        # Q: predicted flip strength from the jets vs the actual flip strength
        def flip_actual(r0, lams, rs_):
            prev_l, prev_r = 0.0, r0
            for lam, rv in zip(lams, rs_):
                if rv > 0:
                    return prev_l + (lam - prev_l) * (-prev_r) / (rv - prev_r) if rv != prev_r else lam
                prev_l, prev_r = lam, rv
            return math.inf

        def flip_J(r0, g1):
            return -r0 / g1 if g1 > 0 else math.inf

        def flip_JQ(r0, g1, g2):
            if abs(g2) < 1e-9:
                return flip_J(r0, g1)
            disc = g1 * g1 - 2 * g2 * r0
            if disc < 0:
                return math.inf
            roots = [(-g1 + s * math.sqrt(disc)) / g2 for s in (1, -1)]
            pos = [x for x in roots if x > 0]
            return min(pos) if pos else math.inf

        say("- **Does Q predict how hard to push?** Gap r(λ) = logit(new) − logit(old). J predicts the flip at r0 + g1 λ = 0,")
        say("  J+Q at r0 + g1 λ + ½ g2 λ² = 0; actual = interpolated on the λ grid. Error = |log(predicted / actual)|, over")
        say("  cases that flip within the grid (median; fraction where J+Q is closer than J):")
        say()
        say("| method | cases flipped | median err J | median err J+Q | J+Q closer | J+Q predicts no flip |")
        say("|---|---|---|---|---|---|")
        groups = [(f"single L{l}", [(r["clean"]["r"], r["single"][str(l)]["g1"], r["single"][str(l)]["g2"], LAMS,
                                     [o["r"] for o in r["single"][str(l)]["out"]]) for r in ok]) for l in Ls]
        groups += [(f"band {k}", [(r["clean"]["r"], r["band"][k]["g1"], r["band"][k]["g2"], BL,
                                   [o["r"] for o in r["band"][k]["out"]]) for r in ok]) for k in bk]
        allc = []
        for name, cases in groups:
            ev = []
            for r0, g1, g2, lams, rs_ in cases:
                fa = flip_actual(r0, lams, rs_)
                if not math.isfinite(fa):
                    continue
                fj, fq = flip_J(r0, g1), flip_JQ(r0, g1, g2)
                ej = abs(math.log(fj / fa)) if math.isfinite(fj) else math.inf
                eq = abs(math.log(fq / fa)) if math.isfinite(fq) else math.inf
                ev.append((ej, eq))
            allc += ev
            if ev:
                ej = sorted(e[0] for e in ev); eq = sorted(e[1] for e in ev)
                say(f"| {name} | {len(ev)} | {ej[len(ej) // 2]:.2f} | {eq[len(eq) // 2]:.2f} | {mean(e[1] < e[0] for e in ev):.2f} | "
                    f"{mean(not math.isfinite(e[1]) for e in ev):.2f} |")
        if allc:
            ej = sorted(e[0] for e in allc); eq = sorted(e[1] for e in allc)
            say(f"| all | {len(allc)} | {ej[len(ej) // 2]:.2f} | {eq[len(eq) // 2]:.2f} | {mean(e[1] < e[0] for e in allc):.2f} | "
                f"{mean(not math.isfinite(e[1]) for e in allc):.2f} |")
        say()
        # the croissant example
        ex = [r for r in ok if r["src"] == "France" and r["tgt"] == "China" and r["func"] == "capital"]
        if ex:
            r = ex[0]
            say(f"- **Example:** “{r['prompt']}” (clean top-5 {r['clean']['top5']}), France→China:")
            bl = max(Ls, key=lambda l: sum(o["succ"] for o in r["single"][str(l)]["out"]))
            for lam, o in zip(LAMS, r["single"][str(bl)]["out"]):
                say(f"  - single L{bl}, λ={lam}: top-5 {o['top5']}, gap {o['r']:+.2f}")
            for k in bk:
                o = r["band"][k]["out"][bi]
                say(f"  - band clamp {k}, λ=1: top-5 {o['top5']}, gap {o['r']:+.2f}, budget L1 {r['band'][k]['budget'][bi]['L1']:.3f}")
say(f"\n(source: {path}; secs {D.get('secs')})")
if md_out:
    Path(md_out).write_text("\n".join(OUT) + "\n")
