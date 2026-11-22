#!/usr/bin/env python3
"""Responsibility map v2 (2026-11-22): path-integrated Q, a whole-next-line target, O(S) Q by Hessian-vector
products with a third-order correction, timing, and matched null contexts.

Spans, removal and t* as in src/responsibility_map.py.
  targets  tstar: centred last-position logits (vector; values along t*, the model's own top-1).
           line : log p(c | context) of the model's own greedy next line c (up to and including the first newline
                  after c_0, at most 16 tokens), teacher-forced; a scalar. The last context position and the
                  continuation positions are never removed.
  exact    D2_ij = f(h+d_i+d_j) - f(h+d_i) - f(h+d_j) + f(h) at s = 1, and at s = .5 divided by .25.
  Q        Q(d_i, d_j) at h, nested jvp per pair.
  pQ       mean of Q at h + lam (d_i + d_j), lam = .25 .5 .75, nested jvp per pair (O(S^2) bases).
  Q_hvp    Q_ij = d_i . H d_j from S Hessian-vector products of the scalar along t* (or the line): all pairs at O(S).
  Q3       Q_ij + (v_i . d_j + v_j . d_i) / 2, v_i = grad_h (d_i' H d_i): third-order Taylor term of D2, O(S).
  null     <corpus>null: context k keeps its last line(s) and replaces everything before with lines (Lean) or
           8-14-token chunks + newline (prose) from other documents (contexts 16 and up); mu from the real corpus.
Output: <out>/resp_v2_<corpus>.json, rewritten after every context.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager

import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_qlens as LQ  # noqa: E402
import responsibility_map as RM  # noqa: E402

W = os.environ.get("QLENS_OUT", "outputs")
LAMS = (0.25, 0.5, 0.75)
KMAX = 16
NREAL = 16


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(box, key):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    box[key] = box.get(key, 0.0) + time.perf_counter() - t0


def blocks(tail, h):
    for blk, mask in zip(tail.blocks, tail.masks):
        o = blk(h, attention_mask=mask, position_embeddings=tail.pe, position_ids=tail.pos)
        h = o[0] if isinstance(o, tuple) else o
    return h


def line_lp(tail, h, start, cont):
    """Teacher-forced log p of cont [K] read at positions start..start+K-1; returns [B]."""
    z = tail.head(tail.norm(blocks(tail, h)[:, start:start + cont.numel()]))
    lp = torch.log_softmax(z, -1)
    return lp.gather(-1, cont.view(1, -1, 1).expand(lp.shape[0], -1, 1)).squeeze(-1).sum(-1)


@torch.no_grad()
def fwd(fn, H, bs):
    with LQ.math_sdpa():
        return torch.cat([fn(H[i: i + bs]) for i in range(0, len(H), bs)]).double()


def jets_fn(fn, base, U, V):
    G = lambda v: fn(base + v)  # noqa: E731
    zero = torch.zeros_like(U)
    with LQ.math_sdpa():
        (_, J), (_, Q) = jvp(lambda v: jvp(G, (v,), (U,)), (zero,), (V,))
    return J.double(), Q.double()


def hvp_q(sf, h0, D, third):
    """Q_ij = d_i . H d_j for all pairs from S HVPs of the scalar sf; with third, also Q3 (third-order Taylor of D2)."""
    S = D.shape[0]
    h = h0.detach().clone().requires_grad_(True)
    HD, V3 = [], []
    with LQ.math_sdpa(), torch.enable_grad():
        y = sf(h)
        g, = torch.autograd.grad(y, h, create_graph=True)
        for k in range(S):
            hd, = torch.autograd.grad((g * D[k]).sum(), h, create_graph=third, retain_graph=True)
            if third:
                v, = torch.autograd.grad((hd * D[k]).sum(), h, retain_graph=True)
                V3.append(v.detach().double())
            HD.append(hd.detach().double())
        del y, g
    Dd = D.double()
    Q = torch.einsum("itd,jtd->ij", Dd, torch.cat(HD))
    asym = float((Q - Q.T).abs().max() / Q.abs().max().clamp_min(1e-30))
    Q = 0.5 * (Q + Q.T)
    Q3 = None
    if third:
        Tm = torch.einsum("itd,jtd->ij", torch.cat(V3), Dd)  # Tm[i, j] = T(d_i, d_i, d_j)
        Q3 = Q + 0.5 * (Tm + Tm.T)
    return Q, Q3, asym


def analyse(fn, h0, D, P, bs, tok, ts, vec):
    S = D.shape[0]
    nv = len(tok)
    tm = {}
    pk = (lambda v: v[..., ts]) if vec else (lambda v: v)  # noqa: E731
    I = [i for i, _ in P]
    Jx = [j for _, j in P]
    blk = [(I[b: b + bs], Jx[b: b + bs]) for b in range(0, len(P), bs)]

    with timed(tm, "exact"):  # 1 + S + P forwards
        f0 = fwd(fn, h0, 1)[0]
        fA = fwd(fn, h0 + D, bs)
        fP = torch.cat([fwd(fn, h0 + D[a] + D[b], bs) for a, b in blk])
    with timed(tm, "exact_half"):
        fAh = fwd(fn, h0 + 0.5 * D, bs)
        fPh = torch.cat([fwd(fn, h0 + 0.5 * (D[a] + D[b]), bs) for a, b in blk])
    with timed(tm, "jet_singles"):
        JQ = [jets_fn(fn, h0, D[b: b + bs], D[b: b + bs]) for b in range(0, S, bs)]
        Js, Qs = torch.cat([x[0] for x in JQ]), torch.cat([x[1] for x in JQ])
    with timed(tm, "Q_jvp"):
        Qp = torch.cat([jets_fn(fn, h0, D[a], D[b])[1] for a, b in blk])
    pQ = []
    with timed(tm, "pathQ"):
        for lam in LAMS:
            pQ.append(torch.cat([jets_fn(fn, h0 + lam * (D[a] + D[b]), D[a], D[b])[1] for a, b in blk]))
    pQm = sum(pQ) / len(pQ)
    sf = lambda h: pk(fn(h)).sum()  # noqa: E731
    with timed(tm, "Q_hvp"):
        Qh, _, asym = hvp_q(sf, h0, D, third=False)
    Q3 = None
    try:
        with timed(tm, "Q3_hvp"):
            _, Q3, _ = hvp_q(sf, h0, D, third=True)
    except RuntimeError as e:  # e.g. out of memory: keep the rest
        LQ.log(f"Q3 failed: {str(e)[:200]}")
        torch.cuda.empty_cache()

    singles = []
    for k in range(S):
        A = fA[k] - f0
        r = {"A": float(pk(A)), "Ah": float(pk(fAh[k] - f0)), "J": float(pk(Js[k])), "Qh": float(0.5 * pk(Qs[k]))}
        if vec:
            r.update(cos_J=LQ.cos(Js[k], A), cos_JQ=LQ.cos(Js[k] + 0.5 * Qs[k], A), A_tok=RM.tops(tok, A, nv))
        singles.append(r)
    pairs = []
    for n, (i, j) in enumerate(P):
        D2 = fP[n] - fA[i] - fA[j] + f0
        D2h = (fPh[n] - fAh[i] - fAh[j] + f0) / 0.25
        r = {"i": i, "j": j, "D2": float(pk(D2)), "D2h": float(pk(D2h)), "Q": float(pk(Qp[n])),
             "pQ": [float(pk(x[n])) for x in pQ], "Qhvp": float(Qh[i, j]),
             "Q3": None if Q3 is None else float(Q3[i, j])}
        if vec:
            r.update(nD2=float(D2.norm()), cos=LQ.cos(Qp[n], D2), cos_h=LQ.cos(Qp[n], D2h), cos_p=LQ.cos(pQm[n], D2),
                     cos_ph=LQ.cos(pQm[n], D2h), D2_tok=RM.tops(tok, D2, nv), Q_tok=RM.tops(tok, Qp[n], nv),
                     pQ_tok=RM.tops(tok, pQm[n], nv))
        pairs.append(r)
    qmax = max(max(abs(p["Q"]) for p in pairs), 1e-30)
    return {"singles": singles, "pairs": pairs, "base": float(pk(f0)), "times": tm,
            "hvp_vs_jvp": max(abs(p["Qhvp"] - p["Q"]) for p in pairs) / qmax, "hvp_asym": asym,
            "passes": {"exact": 1 + S + len(P), "Q_jvp": len(P), "pathQ": len(LAMS) * len(P), "Q_hvp": 1 + S,
                       "Q3_hvp": 1 + 2 * S}}


def selftest(fn, h0, D, pk, name):
    """Tiny model (fp64): Q_hvp = Q_jvp; |D2 - Q| ~ s^3 while |D2 - Q3| and |D2 - pQ| ~ s^4."""
    sf = lambda h: pk(fn(h)).sum()  # noqa: E731
    out = []
    for s in (0.2, 0.1, 0.05):
        Ds = s * D[:2]
        a, b = Ds[0:1], Ds[1:2]
        f = fwd(fn, torch.cat([h0, h0 + a, h0 + b, h0 + a + b]), 4)
        d2 = float(pk(f[3] - f[1] - f[2] + f[0]))
        Qh, Q3, _ = hvp_q(sf, h0, Ds, third=True)
        qj = float(pk(jets_fn(fn, h0, a, b)[1][0]))
        pq = sum(float(pk(jets_fn(fn, h0 + lam * (a + b), a, b)[1][0])) for lam in LAMS) / len(LAMS)
        q, q3 = float(Qh[0, 1]), float(Q3[0, 1])
        out.append((s, d2, abs(d2 - q), abs(d2 - q3), abs(d2 - pq), abs(qj - q) / max(abs(q), 1e-30)))
        LQ.log(f"selftest {name} s={s:.2f} D2={d2:+.3e} |D2-Q|={out[-1][2]:.2e} |D2-Q3|={out[-1][3]:.2e} "
               f"|D2-pQ|={out[-1][4]:.2e} rel|Qjvp-Qhvp|={out[-1][5]:.1e}")
    LQ.log(f"selftest {name} error ratio s=.1/.05: Q {out[1][2] / out[2][2]:.1f} (s^3 -> 8), "
           f"Q3 {out[1][3] / out[2][3]:.1f}, pQ {out[1][4] / out[2][4]:.1f} (s^4 -> 16)")
    assert out[2][5] < 1e-6, "HVP Q differs from nested-jvp Q"
    assert out[2][3] < out[2][2] and out[2][4] < out[2][2], "third-order terms do not improve on Q"


def nl_positions(tok, ids):
    return [p for p in range(len(ids)) if "\n" in tok.decode([ids[p]])]


def whole_lines(tok, ids, lo=3, hi=16):
    nl = nl_positions(tok, ids)
    return [ids[a + 1: b + 1] for a, b in zip(nl, nl[1:])
            if lo <= b - a <= hi and tok.decode(ids[a + 1: b + 1]).strip()]


def build_null(tok, real, src, k, rng, nl_id):
    ids = real[k]["ids"]
    T = len(ids)
    if src == "FW":
        keep = ids[T - 16:]
    else:  # Lean: last full line + final fragment, at most 32 tokens
        nl = [p for p in nl_positions(tok, ids) if p < T - 1]
        start = nl[-2] + 1 if len(nl) >= 2 else T - 32
        keep = ids[max(start, T - 32):]
    need = T - 1 - len(keep)
    pool = list(range(NREAL, len(real)))
    rng.shuffle(pool)
    fill, used = [], []
    for c in pool:
        if sum(map(len, fill)) >= need:
            break
        x = real[c]
        if src != "FW" and x.get("file") == real[k].get("file"):
            continue
        if src == "FW":
            L = rng.randint(8, 14)
            a = rng.randint(1, len(x["ids"]) - L - 1)
            fill.append(x["ids"][a: a + L] + [nl_id])
        else:
            cand = whole_lines(tok, x["ids"])
            if not cand:
                continue
            fill.append(rng.choice(cand))
        used.append(c)
    flat = [t for ln in fill for t in ln]
    flat = flat[len(flat) - need:]
    out = [ids[0]] + flat + keep
    assert len(out) == T, (len(out), T)
    return {"ids": out, "src_k": k, "kept": len(keep), "fill_from": used}


@torch.no_grad()
def greedy_line(hf, tok, ids, ts, dev):
    x = torch.tensor([ids + [ts]], device=dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    out = hf.generate(x, max_new_tokens=KMAX - 1, do_sample=False, pad_token_id=pad)
    c = [ts] + out[0, len(ids) + 1:].tolist()
    for k in range(1, len(c)):
        if "\n" in tok.decode([c[k]]):
            return c[:k + 1]
    return c[:KMAX]


def run_corpus(args, hf, m, tok, C, corpus, n):
    dev, dt = LQ.dev_of(m)
    layers = [int(x) for x in args.layers.split(",")]
    src = corpus[:-4] if corpus.endswith("null") else corpus
    real = C["corpora"][src]["contexts"]
    Hr = []
    for x in real[:NREAL]:
        box, _ = LQ.capture_last(m, torch.tensor([x["ids"]], device=dev), layers)
        Hr.append({l: box[l][0, 1:-1].cpu() for l in layers})
    mu = {l: torch.cat([h[l] for h in Hr]).mean(0).to(dev) for l in layers}
    del Hr
    nl_id = tok("\n", add_special_tokens=False).input_ids
    assert len(nl_id) == 1
    if corpus.endswith("null"):
        rng = random.Random(1234)
        ctxs = [build_null(tok, real, src, k, rng, nl_id[0]) for k in range(n)]
    else:
        ctxs = [{"ids": x["ids"], "file": x.get("file")} for x in real[:n]]
    quiet = src == "RH"  # unlicensed text: no tokens of it in the log
    res = {"model": args.model, "corpus": corpus, "layers": layers, "method": __doc__, "lams": LAMS, "contexts": []}
    out_path = f"{args.out}/resp_v2_{corpus}.json"
    t0 = time.perf_counter()
    for ci, x in enumerate(ctxs):
        ids = x["ids"]
        T = len(ids)
        sp = RM.spans_of(tok, ids)
        S = len(sp)
        P = [(i, j) for i in range(S) for j in range(i + 1, S)]
        box, z = LQ.capture_last(m, torch.tensor([ids], device=dev), layers)
        p = torch.softmax(z[0].double(), -1)
        ts = int(z[0].argmax())
        cont = greedy_line(hf, tok, ids, ts, dev)
        ext = ids + cont[:-1]
        TT = len(ext)
        boxx, _ = LQ.capture_last(m, torch.tensor([ext], device=dev), layers)
        contT = torch.tensor(cont, device=dev)
        rec = {k: v for k, v in x.items() if k != "ids"}
        rec.update(ci=ci, ids=ids, spans=[{"pos": [s[0], s[-1]], "text": tok.decode([ids[q] for q in s])} for s in sp],
                   tstar=tok.decode([ts]), next_top=[[tok.decode([i]), round(float(p[i]), 4)]
                                                    for i in p.topk(5).indices.tolist()],
                   cont=[tok.decode([c]) for c in cont], L={})
        tsum = {}
        for l in layers:
            D = torch.zeros(S, T, m.d_model, device=dev, dtype=dt)
            for k, s in enumerate(sp):
                D[k, s] = mu[l] - box[l][0, s]
            Dx = torch.zeros(S, TT, m.d_model, device=dev, dtype=dt)
            Dx[:, :T] = D
            tail, tailx = LQ.Tail(m, l, T), LQ.Tail(m, l, TT)
            fn_t = lambda H, tail=tail: LQ.centre(tail.last(H))  # noqa: E731
            fn_l = lambda H, tailx=tailx: line_lp(tailx, H, T - 1, contT)  # noqa: E731
            if args.tiny and ci == 0 and l == layers[0]:
                selftest(fn_t, box[l], D, lambda v: v[..., ts], "tstar")
                selftest(fn_l, boxx[l], Dx, lambda v: v, "line")
            row = {"tstar": analyse(fn_t, box[l], D, P, args.batch, tok, ts, True),
                   "line": analyse(fn_l, boxx[l], Dx, P, args.batch, tok, ts, False)}
            for tg in row:
                for kk, v in row[tg]["times"].items():
                    tsum[f"{tg[0]}:{kk}"] = tsum.get(f"{tg[0]}:{kk}", 0.0) + v
            rec["L"][str(l)] = row
            torch.cuda.empty_cache()
        res["contexts"].append(rec)
        with open(out_path, "w") as fh:
            json.dump(res, fh, ensure_ascii=False)
        what = "" if quiet else f" t*={rec['tstar']!r} line={''.join(rec['cont'])!r}"
        LQ.log(f"{corpus} c{ci}: S={S} pairs={len(P)} K={len(cont)}{what} "
               + " ".join(f"{k}={v:.1f}" for k, v in tsum.items()) + f" ({time.perf_counter() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--corpora", default="ML:16")
    ap.add_argument("--layers", default="11,20,29")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=f"{W}/resp_map_v2")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hf, m, tok = LQ.load(args)
    for prm in hf.parameters():
        prm.requires_grad_(False)
    C = torch.load(f"{W}/lean_v1/data/contexts.pt", weights_only=False)
    for item in args.corpora.split(","):
        corpus, n = item.split(":")
        run_corpus(args, hf, m, tok, C, corpus, int(n))
    LQ.log("ALL DONE")


if __name__ == "__main__":
    main()
