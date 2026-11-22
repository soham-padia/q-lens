#!/usr/bin/env python3
"""Finite-difference check of Q against step size, in fp32 and fp64 (resolves the v1 FD gate at best cos 0.90-0.99).

ML c0, first 4 pairs of lines, blocks 11/20/29 (as the v1 gate):
  rel err(s) = || (f(h+s d_i+s d_j) - f(h+s d_i) - f(h+s d_j) + f(h)) / s^2 - Q(d_i, d_j) || / ||Q||
for s = 1 ... 1e-4. In fp64 the RMSNorms are patched to stay in fp64 (the HF norm computes in fp32 and casts back).
Reading (fixed before the run): a V in fp32 with a continued fall in fp64 means precision;
a floor in fp64 that does not fall means the derivative is wrong.
usage: python src/fd_precision.py --dtype float64 [--tiny --layers 1,2 --out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_qlens as LQ  # noqa: E402
import responsibility_map as RM  # noqa: E402

W = os.environ.get("QLENS_OUT", "outputs")
GRID = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 5e-4, 3e-4, 1e-4]


def rms_keep(self, x):
    return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--layers", default="11,20,29")
    ap.add_argument("--out", default=f"{W}/resp_map_v2/fd_precision")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    args.load_dtype = getattr(torch, args.dtype)
    os.makedirs(args.out, exist_ok=True)
    hf, m, tok = LQ.load(args)
    if args.dtype == "float64" and not args.tiny:
        n = 0
        for mod in hf.modules():
            if type(mod).__name__.endswith("RMSNorm"):
                mod.forward = types.MethodType(rms_keep, mod)
                n += 1
        LQ.log(f"patched {n} RMSNorm modules to keep fp64")
    dev, dt = LQ.dev_of(m)
    LQ.log(f"model dtype {dt}")
    layers = [int(x) for x in args.layers.split(",")]
    C = torch.load(f"{W}/lean_v1/data/contexts.pt", weights_only=False)
    real = C["corpora"]["ML"]["contexts"][:16]
    H = []
    for x in real:
        box, _ = LQ.capture_last(m, torch.tensor([x["ids"]], device=dev), layers)
        H.append({l: box[l].cpu() for l in layers})
    mu = {l: torch.cat([h[l][0, 1:-1] for h in H]).mean(0).to(dev) for l in layers}
    ids = real[0]["ids"]
    T = len(ids)
    sp = RM.spans_of(tok, ids)
    P = [(i, j) for i in range(len(sp)) for j in range(i + 1, len(sp))][:4]
    res = {"dtype": args.dtype, "grid": GRID, "pairs": P, "layers": {}}
    Qsave = {}
    for l in layers:
        tail = LQ.Tail(m, l, T)
        h0 = H[0][l].to(dev)
        rows = []
        for (i, j) in P:
            di = torch.zeros(1, T, m.d_model, device=dev, dtype=dt)
            dj = torch.zeros_like(di)
            di[0, sp[i]] = mu[l] - h0[0, sp[i]]
            dj[0, sp[j]] = mu[l] - h0[0, sp[j]]
            _, _, Q = LQ.jets(tail, h0, di, dj)
            Q = LQ.centre(Q[0].double())
            errs, coss = [], []
            for s in GRID:
                X = torch.cat([h0, h0 + s * di, h0 + s * dj, h0 + s * (di + dj)])
                fz = LQ.centre(LQ.tail_batched(tail, X).double())
                fd = (fz[3] - fz[1] - fz[2] + fz[0]) / s ** 2
                errs.append(float((fd - Q).norm() / Q.norm()))
                coss.append(LQ.cos(fd, Q))
            rows.append({"i": i, "j": j, "nQ": float(Q.norm()), "rel_err": errs, "cos": coss})
            Qsave[f"{l}_{i}_{j}"] = Q.cpu()
            LQ.log(f"{args.dtype} L{l} pair ({i},{j}) best rel err {min(errs):.2e} at s={GRID[errs.index(min(errs))]} "
                   f"best cos {max(coss):.6f}")
        res["layers"][str(l)] = rows
        with open(f"{args.out}/fd_{args.dtype}.json", "w") as fh:
            json.dump(res, fh)
    torch.save(Qsave, f"{args.out}/Q_{args.dtype}.pt")
    LQ.log("FD DONE")


if __name__ == "__main__":
    main()
