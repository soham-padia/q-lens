#!/usr/bin/env python3
"""Cost curve: exact removal of every pair of lines vs the Q estimators, at S = 12 / 50 / 200 lines (timing only).

Context of S lines = ML contexts joined with a newline token, cut after the S-th span (responsibility_map.spans_of).
Push, mu (from real[:16]) and the t* target as responsibility_map_v2. Per (S, block):
  exact  = (1 + S + P) forwards              : unit time per forward measured; full run measured when S <= --exact-max
  Q_jvp  = P nested jvps (one per pair)       : unit time per pair measured on --jvp-pairs pairs, x P
  pQ     = 3P nested jvps at per-pair bases   : unit time measured at lam = .5 on the same pairs, x 3P
  Q_hvp  = 1 gradient + S HVPs                : measured for min(S, --hvp-max) HVPs, the rest extrapolated
  Q3     = 1 gradient + S HVPs + S third-order : as Q_hvp
Every total is flagged 'measured' or 'extrapolated'; peak GPU memory per method is recorded; an OOM is a result.
No pair values are reported (cost only). Output: <out>/cost_curve.json; a markdown table in the log.
usage: python src/cost_curve.py [--S 12,50,200 --layers 20,29] [--tiny]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_qlens as LQ  # noqa: E402
import responsibility_map as RM  # noqa: E402
import responsibility_map_v2 as R2  # noqa: E402

W = os.environ.get("QLENS_OUT", "outputs")


def build(tok, real, S, nl):
    ids, k = [], 0
    while True:
        ids = ids + list(real[k]["ids"]) + [nl]
        k += 1
        sp = RM.spans_of(tok, ids)
        if len(sp) >= S + 1 or k == len(real):
            break
    cut = sp[min(S, len(sp)) - 1][-1]
    ids = ids[: cut + 2]  # last position (cut + 1) is never removed
    return ids, RM.spans_of(tok, ids), k


def clock(fn, reps=1):
    R2.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    R2.sync()
    return (time.perf_counter() - t0) / reps


def peak_reset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0


def oom_safe(box, key, f):
    peak_reset()
    try:
        f()
        box[key + "_peak_gb"] = round(peak(), 2)
    except torch.cuda.OutOfMemoryError as e:
        box[key] = "OOM"
        box[key + "_peak_gb"] = round(peak(), 2)
        LQ.log(f"{key}: OOM ({str(e)[:120]})")
        torch.cuda.empty_cache()


def hvps(sf, h0, D, third, nmax):
    """Time the gradient (create_graph) and the first nmax HVPs (+ third-order backward); returns (t_grad, [t_k])."""
    h = h0.detach().clone().requires_grad_(True)
    ts = []
    with LQ.math_sdpa(), torch.enable_grad():
        R2.sync(); t0 = time.perf_counter()
        y = sf(h)
        g, = torch.autograd.grad(y, h, create_graph=True)
        R2.sync(); tg = time.perf_counter() - t0
        for k in range(min(nmax, D.shape[0])):
            R2.sync(); t0 = time.perf_counter()
            hd, = torch.autograd.grad((g * D[k]).sum(), h, create_graph=third, retain_graph=True)
            if third:
                torch.autograd.grad((hd * D[k]).sum(), h, retain_graph=True)
            R2.sync(); ts.append(time.perf_counter() - t0)
        del y, g
    return tg, ts


def one(args, m, tok, real, mu, S, l, dev, dt):
    nl = tok("\n", add_special_tokens=False).input_ids[0]
    ids, sp, nctx = build(tok, real, S, nl)
    T, S_ = len(ids), len(sp)
    P = S_ * (S_ - 1) // 2
    pairs = [(i, j) for i in range(S_) for j in range(i + 1, S_)]
    box, z = LQ.capture_last(m, torch.tensor([ids], device=dev), [l])
    h0 = box[l]
    ts_ = int(z[0].argmax())
    D = torch.zeros(S_, T, m.d_model, device=dev, dtype=dt)
    for k, s in enumerate(sp):
        D[k, s] = mu[l] - h0[0, s]
    tail = LQ.Tail(m, l, T)
    fn = lambda H: LQ.centre(tail.last(H))  # noqa: E731
    sf = lambda H: fn(H)[..., ts_].sum()  # noqa: E731
    bs = args.batch if T <= 800 else (2 if T <= 1600 else 1)
    r = {"S": S_, "S_req": S, "T": T, "P": P, "layer": l, "n_src_ctx": nctx, "batch": bs,
         "passes": {"exact": 1 + S_ + P, "Q_jvp": P, "pQ": 3 * P, "Q_hvp": 1 + S_, "Q3": 1 + 2 * S_}}
    LQ.log(f"S={S_} T={T} P={P} L{l} batch={bs}")

    def exact_unit():
        R2.fwd(fn, h0 + D[:bs], bs)  # warm-up
        r["exact_unit_s"] = clock(lambda: R2.fwd(fn, h0 + D[:bs], bs), args.reps) / bs
        est = r["exact_unit_s"] * r["passes"]["exact"]
        if S_ <= args.exact_max:
            def full():
                f0 = R2.fwd(fn, h0, 1)[0]
                fA = R2.fwd(fn, h0 + D, bs)
                for b in range(0, P, bs):
                    a_, b_ = zip(*pairs[b: b + bs])
                    R2.fwd(fn, h0 + D[list(a_)] + D[list(b_)], bs)
                return f0, fA
            r["exact_s"], r["exact_how"] = clock(full), "measured"
            r["exact_unit_x_passes_s"] = est
        else:
            r["exact_s"], r["exact_how"] = est, "extrapolated"
    oom_safe(r, "exact", exact_unit)

    npj = min(args.jvp_pairs, P)
    sel = pairs[:npj]
    a_, b_ = [list(x) for x in zip(*sel)]

    def jvp_unit():
        def run(lam):
            for b in range(0, npj, bs):
                ia, ib = a_[b: b + bs], b_[b: b + bs]
                R2.jets_fn(fn, h0 + lam * (D[ia] + D[ib]), D[ia], D[ib])
        run(0.0)  # warm-up
        r["Q_jvp_unit_s"] = clock(lambda: run(0.0)) / npj
        r["pQ_unit_s"] = clock(lambda: run(0.5)) / npj
        r["Q_jvp_s"] = r["Q_jvp_unit_s"] * P
        r["pQ_s"] = r["pQ_unit_s"] * 3 * P
        r["Q_jvp_how"] = r["pQ_how"] = "measured" if npj == P else f"extrapolated from {npj} pairs"
    oom_safe(r, "Q_jvp", jvp_unit)

    for key, third in (("Q_hvp", False), ("Q3", True)):
        def go(key=key, third=third):
            hvps(sf, h0, D[:2], third, 1)  # warm-up
            tg, tk = hvps(sf, h0, D, third, args.hvp_max)
            per = sum(tk) / len(tk)
            r[f"{key}_grad_s"], r[f"{key}_per_hvp_s"] = tg, per
            r[f"{key}_s"] = tg + per * S_
            r[f"{key}_how"] = "measured" if len(tk) == S_ else f"extrapolated from {len(tk)} HVPs"
        oom_safe(r, key, go)
    LQ.log(json.dumps(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--S", default="12,50,200")
    ap.add_argument("--layers", default="20,29")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--exact-max", type=int, default=50, help="run the full exact sweep when S <= this")
    ap.add_argument("--jvp-pairs", type=int, default=16)
    ap.add_argument("--hvp-max", type=int, default=50)
    ap.add_argument("--out", default=f"{W}/resp_map_v2/cost_curve")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.S, args.layers, args.out, args.reps = "4,8", "1,2", f"{W}/resp_v2_tiny/cost_curve", 1
    os.makedirs(args.out, exist_ok=True)
    hf, m, tok = LQ.load(args)
    for prm in hf.parameters():
        prm.requires_grad_(False)
    dev, dt = LQ.dev_of(m)
    layers = [int(x) for x in args.layers.split(",")]
    C = torch.load(f"{W}/lean_v1/data/contexts.pt", weights_only=False)
    real = C["corpora"]["ML"]["contexts"]
    H = []
    for x in real[: R2.NREAL]:
        box, _ = LQ.capture_last(m, torch.tensor([x["ids"]], device=dev), layers)
        H.append({l: box[l][0, 1:-1].cpu() for l in layers})
    mu = {l: torch.cat([h[l] for h in H]).mean(0).to(dev) for l in layers}
    del H
    rows = []
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    for S in [int(x) for x in args.S.split(",")]:
        for l in layers:
            rows.append(one(args, m, tok, real, mu, S, l, dev, dt))
            with open(f"{args.out}/cost_curve.json", "w") as fh:
                json.dump({"model": args.model, "gpu": gpu, "doc": __doc__, "rows": rows}, fh, indent=1)
    f = lambda r, k: (f"{r[k]:.1f}" if isinstance(r.get(k), float) else str(r.get(k)))  # noqa: E731
    print("| S | T | P | block | exact s | Q_jvp s | pQ s | Q_hvp s | Q3 s | peak GB exact / jvp / hvp / Q3 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['S']} | {r['T']} | {r['P']} | {r['layer']} | {f(r, 'exact_s')} ({r.get('exact_how', '-')}) | "
              f"{f(r, 'Q_jvp_s')} | {f(r, 'pQ_s')} | {f(r, 'Q_hvp_s')} | {f(r, 'Q3_s')} | "
              + " / ".join(str(r.get(k + "_peak_gb")) for k in ("exact", "Q_jvp", "Q_hvp", "Q3")) + " |")
    LQ.log("COST DONE")


if __name__ == "__main__":
    main()
