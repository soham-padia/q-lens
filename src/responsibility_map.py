#!/usr/bin/env python3
"""Responsibility map: which lines of the context are responsible for the model's next token, alone and together.

Reuses src/lean_qlens.py (model, Tail, jets).
  spans  lines of the context (> 16 tokens cut into 16-token pieces; whitespace-only lines merged into the previous
         span); position 0 (attention sink) and the last position (the readout's own state) are never removed.
  push   d_i = mu_l - h_l on span i's positions (mean ablation at block l; mu_l = mean block-l state of the corpus's
         pilot contexts, positions 1..T-2).
  output f = centred last-position logits; t* = the model's own top-1 next token.
  alone  A_i = f(h+d_i) - f(h) exact; J d_i; 1/2 Q(d_i, d_i).
  pair   D2_ij = f(h+d_i+d_j) - f(h+d_i) - f(h+d_j) + f(h) exact, at s = 1 and s = .5 (divided by s^2);
         Q(d_i, d_j) by nested jvp (mixed second derivative). J has no pair term.
Output: <out>/resp_<corpus>.json (rewritten after every context).
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

W = os.environ.get("QLENS_OUT", "outputs")
MAXSPAN = 16
TOPK = 6


def spans_of(tok, ids):
    """Lines of positions 1..T-2 as lists of positions (long lines cut, whitespace-only lines merged)."""
    T = len(ids)
    pieces = [tok.decode([i]) for i in ids]
    lines, cur = [], []
    for p in range(1, T - 1):
        cur.append(p)
        if "\n" in pieces[p]:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        if out and not tok.decode([ids[p] for p in ln]).strip():
            out[-1] = out[-1] + ln
            continue
        for k in range(0, len(ln), MAXSPAN):
            out.append(ln[k: k + MAXSPAN])
    return out


def tops(tok, v, nv):
    v = v[:nv]
    up = v.topk(TOPK).indices.tolist()
    dn = (-v).topk(TOPK).indices.tolist()
    return [tok.decode([i]) for i in up], [tok.decode([i]) for i in dn]


def cosv(a, b):
    return LQ.cos(a, b)


def run_corpus(args, m, tok, C, corpus, n):
    dev, dt = LQ.dev_of(m)
    layers = [int(x) for x in args.layers.split(",")]
    nv = len(tok)
    ctxs = C["corpora"][corpus]["contexts"][:n]
    T = len(ctxs[0]["ids"])
    # states and mean per layer
    H, Z = [], []
    for x in ctxs:
        assert len(x["ids"]) == T
        box, z = LQ.capture_last(m, torch.tensor([x["ids"]], device=dev), layers)
        H.append({l: box[l].cpu() for l in layers})
        Z.append(z[0].double().cpu())
    mu = {l: torch.cat([h[l][0, 1:T - 1] for h in H]).mean(0) for l in layers}
    res = {"model": args.model, "corpus": corpus, "layers": layers, "method": __doc__, "contexts": []}
    out_path = f"{args.out}/resp_{corpus}.json"
    t0 = time.perf_counter()
    for ci, x in enumerate(ctxs):
        ids = x["ids"]
        sp = spans_of(tok, ids)
        S = len(sp)
        z = Z[ci]
        p = torch.softmax(z, -1)
        ts = int(z.argmax())
        rec = {"ci": ci, "file": x.get("file"), "target": tok.decode([x["target"]]) if "target" in x else None,
               "tail": tok.decode(ids[sp[-1][-1] + 1:]) if sp else "",
               "head": tok.decode(ids[:1]), "spans": [{"pos": [s[0], s[-1]], "text": tok.decode([ids[q] for q in s])}
                                                    for s in sp],
               "tstar": tok.decode([ts]), "next_top": [[tok.decode([i]), round(float(p[i]), 4)]
                                                       for i in p.topk(5).indices.tolist()], "L": {}}
        for l in layers:
            tail = LQ.Tail(m, l, T)
            h0 = H[ci][l].to(dev)
            D = torch.zeros(S, T, m.d_model, device=dev, dtype=dt)
            for k, s in enumerate(sp):
                D[k, s] = mu[l].to(dev) - h0[0, s]
            f0 = LQ.centre(LQ.tail_batched(tail, h0)[0].double())
            # singles: exact (s = 1 and .5) and jets
            fA = LQ.centre(LQ.tail_batched(tail, h0 + D).double())
            fAh = LQ.centre(LQ.tail_batched(tail, h0 + 0.5 * D).double())
            Js, Qs = [], []
            for b in range(0, S, args.batch):
                _, J, Q = LQ.jets(tail, h0, D[b: b + args.batch])
                Js.append(LQ.centre(J.double())); Qs.append(LQ.centre(Q.double()))
            Js, Qs = torch.cat(Js), torch.cat(Qs)
            singles = []
            for k in range(S):
                A = fA[k] - f0
                singles.append({"A": float(A[ts]), "J": float(Js[k, ts]), "Qh": float(0.5 * Qs[k, ts]),
                                "Ah": float((fAh[k] - f0)[ts]), "nA": float(A.norm()),
                                "cos_J": cosv(Js[k], A), "cos_JQ": cosv(Js[k] + 0.5 * Qs[k], A),
                                "rel_J": LQ.rel(Js[k], A), "rel_JQ": LQ.rel(Js[k] + 0.5 * Qs[k], A),
                                "A_tok": tops(tok, A, nv), "J_tok": tops(tok, Js[k], nv)})
            # pairs
            P = [(i, j) for i in range(S) for j in range(i + 1, S)]
            pairs = []
            for b in range(0, len(P), args.batch):
                blk = P[b: b + args.batch]
                I = [i for i, _ in blk]; Jx = [j for _, j in blk]
                _, _, Qp = LQ.jets(tail, h0, D[I], D[Jx])
                Qp = LQ.centre(Qp.double())
                fP = LQ.centre(LQ.tail_batched(tail, h0 + D[I] + D[Jx]).double())
                fPh = LQ.centre(LQ.tail_batched(tail, h0 + 0.5 * (D[I] + D[Jx])).double())
                for k, (i, j) in enumerate(blk):
                    D2 = fP[k] - fA[i] - fA[j] + f0
                    D2h = (fPh[k] - fAh[i] - fAh[j] + f0) / 0.25
                    pairs.append({"i": i, "j": j, "D2": float(D2[ts]), "D2h": float(D2h[ts]), "Q": float(Qp[k, ts]),
                                  "nD2": float(D2.norm()), "nQ": float(Qp[k].norm()),
                                  "cos": cosv(Qp[k], D2), "cos_h": cosv(Qp[k], D2h),
                                  "D2_tok": tops(tok, D2, nv), "Q_tok": tops(tok, Qp[k], nv)})
            row = {"singles": singles, "pairs": pairs, "base_tstar": float(f0[ts])}
            if ci == 0:  # FD gate: D2(s)/s^2 -> Q at small s, first 4 pairs
                gate = []
                for (i, j) in P[:4]:
                    _, _, Qij = LQ.jets(tail, h0, D[i:i + 1], D[j:j + 1])
                    Qij = LQ.centre(Qij[0].double())
                    best = -2.0
                    for s in (0.3, 0.1, 0.03):
                        X = torch.stack([h0[0], h0[0] + s * D[i], h0[0] + s * D[j], h0[0] + s * (D[i] + D[j])])
                        fz = LQ.centre(LQ.tail_batched(tail, X).double())
                        best = max(best, cosv((fz[3] - fz[1] - fz[2] + fz[0]) / s ** 2, Qij))
                    gate.append(best)
                row["fd_gate_best_cos"] = gate
                LQ.log(f"{corpus} L{l} FD gate best cos {['%.4f' % g for g in gate]}")
            rec["L"][str(l)] = row
            torch.cuda.empty_cache()
        res["contexts"].append(rec)
        with open(out_path, "w") as fh:
            json.dump(res, fh, ensure_ascii=False)
        LQ.log(f"{corpus} c{ci}: S={S} pairs={S * (S - 1) // 2} t*={rec['tstar']!r} ({time.perf_counter() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--corpora", default="ML:16,FW:6")
    ap.add_argument("--layers", default="11,20,29")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=f"{W}/resp_map_v1")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _, m, tok = LQ.load(args)
    C = torch.load(f"{W}/lean_v1/data/contexts.pt", weights_only=False)
    for item in args.corpora.split(","):
        corpus, n = item.split(":")
        run_corpus(args, m, tok, C, corpus, int(n))
    LQ.log("ALL DONE")


if __name__ == "__main__":
    main()
