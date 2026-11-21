#!/usr/bin/env python3
"""Exact second-order split (q_anatomy A2) along the ACTUAL swap pushes of S2 T, with same-prompt baselines.

Question: "what decides a swap lives across layers" was only tested on iso/data directions (C4). Here the tangent is
the per-position swap push Delta_l = (c_flip - c) V^T of S2 T (two-hop "croissant" swaps, src -> tgt country) at a
single layer l, exactly as second_order.py builds it (lens pair V = [v_src, v_tgt] from the fitted mean J-lens,
c = clean coordinates, all positions). Q = D^2 F[Delta, Delta] is split exactly over the sites (n1, mix, n2, mlp) of
every downstream block, and inside mix/mlp over heads and neurons, with q_anatomy's own functions (imported, not
edited; this file adds only the per-position tangent and the readouts below).

F = block L's output (pre-norm) at the LAST position (the answer position; A2 sums over valid positions, which for a
swap would mix in positions that never decide anything). Two scalar readouts of each contribution:
  q_hat = Q/|Q|            (A2's readout: same statistics as C4)
  u     = d r / d h_L      r = logit(new) - logit(old) through the final norm, at the clean state: the decision.
          u.J = S2's g1, u.Q + D^2 r[J, J] = S2's g2 (gate against the stored S2 jets).
Directions per (trial, layer), one batch of 8: swap, ctrl (an irrelevant-pair swap: another country <-> tgt, as S2's
band control), rand x2 (S2's random control: same per-position norms, random unit direction), iso x2 and data x2
(A2's own directions for that layer, spread over every position but the first: the prompts are
shorter than A2's valid-position minimum), all at the Frobenius norm of the swap push
except ctrl (its own natural norm). Split statistics are scale-free; Q/J is compared at equal norm.

  python src/swap_split.py --model Qwen/Qwen3.8-27B --out DIR [--layers 48,40,36] [--n-trials 48] [--shard 0/3]
Tiny CPU test: QANAT_DEVICE=cpu python src/swap_split.py --model <tiny dir> --out DIR --layers 5,3 --n-trials 2
               --lens random --s2-gate 0
Summary: python src/swap_split_summary.py DIR/swap_split*.json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="")
ap.add_argument("--layers", default="48,40,36", help="single-layer swap sites (S2 layers), run in this order")
ap.add_argument("--n-trials", type=int, default=48, help="evenly spaced over the valid S2 T trials (canonical order)")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--s2-glob", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "second_order_v1/s2/t*/S2.json"))
ap.add_argument("--lens", default="fit", choices=["fit", "random"], help="random = tiny-model test only")
ap.add_argument("--lens-dir", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "fit_v1"))
ap.add_argument("--s2-gate", type=int, default=1, help="compare u.J, u.Q + D2r with the stored S2 jets (real model)")
ap.add_argument("--tf32", type=int, default=0)
ap.add_argument("--max-min", type=float, default=170.0, help="stop starting new trials after this many minutes")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)

# q_anatomy is a script: import it with its own argv and no stages (it loads the model, tokenizer and corpus contexts)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["q_anatomy.py", "--model", A.model, "--out", str(OUT / "qa_import"), "--stages", "none",
            "--tf32", str(A.tf32), "--sources", "48", "--a3-sources", "48", "--check-source", "48"]
import torch  # noqa: E402
import q_anatomy as qa  # noqa: E402

m, L, D, DEV, dtype, tok = qa.m, qa.L, qa.D, qa.DEV, qa.dtype, qa.tok
LAYERS = [int(x) for x in A.layers.split(",")]
T00 = time.perf_counter()
log = qa.log


def tid(word):  # = second_order.tid
    for w in (" " + word, word):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], True
    return tok.encode(" " + word, add_special_tokens=False)[0], False


# ---------------------------------------------------------------- trials (from the S2 T run)
s2 = [json.load(open(fp)) for fp in sorted(glob.glob(A.s2_glob))]
CN, FN = list(s2[0]["countries"]), list(s2[0]["funcs"])
allrows = [r for d in s2 for r in d["twohop"]]
allrows.sort(key=lambda r: (FN.index(r["func"]), CN.index(r["src"]), CN.index(r["tgt"])))  # second_order's `two` order
valid = [r for r in allrows if "single" in r]
N = min(A.n_trials, len(valid))
sel = [valid[round(i * len(valid) / N)] for i in range(N)]
si, sn = (int(x) for x in A.shard.split("/"))
mine = sel[si::sn]
log(f"swap_split: {len(valid)} valid S2 T trials, {N} selected, shard {A.shard}: {len(mine)} trials; layers {LAYERS}")

# ---------------------------------------------------------------- lens vectors (= second_order S2: v_w = (gamma*u_w)^T E[J_l])
words = sorted(set(CN))
WI = {w: i for i, w in enumerate(words)}
if A.lens == "random":
    g_ = torch.Generator().manual_seed(5)
    JV = {l: torch.randn(len(words), D, generator=g_).to(DEV) for l in LAYERS}
else:
    U_w = qa.W_U[[tid(w)[0] for w in words]].float() * qa.gamma
    JV, ntot = {}, 0
    for fp in sorted(glob.glob(f"{A.lens_dir}/shard*of*.pt")):
        ck = torch.load(fp, map_location="cpu", weights_only=True, mmap=True)
        n = ck["n_prompts"]; ntot += n
        for l in LAYERS:
            JV[l] = JV.get(l, 0) + (U_w @ ck["J"][l].to(DEV).float()) * n
        del ck
    JV = {l: v / ntot for l, v in JV.items()}
log(f"lens vectors ({A.lens}) for {len(words)} words at layers {LAYERS}")


def swap_push(Hc, a, b, l):
    """second_order Pair(a, b).D[l]: (c_flip - c) V^T, all positions; Hc [T, d] clean block-l output."""
    V = torch.stack([JV[l][WI[a]], JV[l][WI[b]]], 1)
    P = torch.linalg.pinv(V)
    c = Hc @ P.T
    return (c.flip(-1) - c) @ V.T


# A2's own iso/data directions at these layers (same seeds and contexts as q_anatomy A2)
st, dirs = qa.directions(LAYERS, 8)
TYPES = ["swap", "ctrl", "rand", "rand", "iso", "iso", "data", "data"]


# ---------------------------------------------------------------- the split (q_anatomy.anatomy2 + a second readout u)
def anatomy_swap(R, l, A1, u):
    b = A1.shape[0]
    T1, T2 = qa.all_sites(R, l, A1)
    Jf, Qf = T1[("L", None)], T2[("L", None)]
    tin = qa.site_inputs(l, A1, T1)
    rc = qa.resid_check(l, A1, T1)
    del T2
    qhat = Qf / Qf.norm(dim=-1, keepdim=True)
    sites = [(X, k) for k in range(l + 1, L + 1) for X in ("mix", "mlp")]
    g = qa.cotangents(R, b, l + 1, sites, qhat)
    gu = qa.cotangents(R, b, l + 1, sites, u[None].expand(b, -1).contiguous())
    blocks = list(range(l + 1, L + 1))
    C, heads, neur, heads_u, neur_u = {}, {}, {}, {}, {}
    chk = {"head_sum_vs_jvp": 0.0, "neuron_sum_vs_jvp": 0.0, "head_sum_vs_jvp_u": 0.0, "neuron_sum_vs_jvp_u": 0.0}
    uQ = (u[None] * Qf).sum(-1)
    for k in blocks:
        s = qa.block_d2(R, b, k, tin, extra=True)
        c = qa.push_down(R, k, s)
        for X in qa.SITES:
            C[(X, k)] = c[X]
        Wo, Wd = qa.module("oin", k).weight.float(), qa.module("din", k).weight.float()
        for (gg, ref, hd, nd, sfx, den) in ((g, qhat, heads, neur, "", Qf.norm(dim=-1)), (gu, u[None], heads_u, neur_u, "_u", uQ.abs())):
            ph = ((gg[("mix", k)] @ Wo) * s["oin"]).view(b, R.T, -1, qa.head_dim(k)).sum((1, 3))  # [b, H]
            pn = ((gg[("mlp", k)] @ Wd) * s["din"]).sum(1)  # [b, I]
            pm, pl = (ref * c["mix"]).sum(-1), (ref * c["mlp"]).sum(-1)
            chk["head_sum_vs_jvp" + sfx] = max(chk["head_sum_vs_jvp" + sfx], float(((ph.sum(1) - pm).abs() / den).max()))
            chk["neuron_sum_vs_jvp" + sfx] = max(chk["neuron_sum_vs_jvp" + sfx], float(((pn.sum(1) - pl).abs() / den).max()))
            hd[k], nd[k] = ph, pn
        del s, c
    rows_out = []
    for i in range(b):
        Cs = torch.stack([C[(X, k)][i] for k in blocks for X in qa.SITES])  # [4K, d]
        Cb = Cs.view(len(blocks), 4, D).sum(1)  # [K, d]
        Q = Qf[i]
        Ql = qa.lg(Q[None])[0]
        Cl = qa.lg(Cs)
        qh = qhat[i]
        r = {}
        for sfx, ref, hd, nd in (("", qh, heads, neur), ("_u", u, heads_u, neur_u)):
            p_site = Cs @ ref
            p_n1, p_n2 = p_site.view(-1, 4)[:, 0], p_site.view(-1, 4)[:, 2]
            p_heads = torch.cat([hd[k][i] for k in blocks])
            p_neur = torch.cat([nd[k][i] for k in blocks])
            r.update({"scal_block" + sfx: qa.scal_stats(Cb @ ref), "scal_site" + sfx: qa.scal_stats(p_site),
                      "scal_fine" + sfx: qa.scal_stats(torch.cat([p_n1, p_n2, p_heads, p_neur])),
                      "scal_heads" + sfx: qa.scal_stats(p_heads), "scal_neurons" + sfx: qa.scal_stats(p_neur),
                      "block_along" + sfx: [float(x) for x in Cb @ ref]})
        r.update({"Q_over_J": float(Q.norm() / Jf[i].norm()), "Jn": float(Jf[i].norm()), "Qn": float(Q.norm()),
                  "site": qa.vec_stats(Cs, Q), "block": qa.vec_stats(Cb, Q),
                  "site_logit": qa.vec_stats(Cl, Ql), "block_logit": qa.vec_stats(Cl.view(len(blocks), 4, -1).sum(1), Ql),
                  "type_share": {X: float(Cs.view(-1, 4, D)[:, j].sum(0) @ Q / Q.norm() ** 2) for j, X in enumerate(qa.SITES)},
                  "type_share_u": {X: float(Cs.view(-1, 4, D)[:, j].sum(0) @ u / uQ[i]) for j, X in enumerate(qa.SITES)},
                  "block_share": [float(x) for x in Cb @ Q / Q.norm() ** 2],
                  "block_norm": [float(x) for x in Cb.norm(dim=-1) / Q.norm()]})
        rows_out.append(r)
        del Cl
    chk["resid_structure"] = rc
    return rows_out, Jf, Qf, chk, blocks


class LastRunner(qa.Runner):
    """q_anatomy.Runner with F read at the last position only. The swap prompts (~15 tokens) are shorter than
    jlens's valid-position minimum (> 17), so the mask is replaced, not computed."""

    def __init__(self, ids1):
        orig = qa.valid_position_mask
        qa.valid_position_mask = lambda T_: torch.arange(T_) == T_ - 1
        try:
            super().__init__(ids1)
        finally:
            qa.valid_position_mask = orig


# ---------------------------------------------------------------- per trial
res = {"args": vars(A), "types": TYPES, "layers": LAYERS, "n_valid": len(valid), "n_selected": N,
       "defs": __doc__, "trials": []}
fname = OUT / f"swap_split{A.name}.json"
for ti, tr in enumerate(mine):
    if (time.perf_counter() - T00) / 60 > A.max_min:
        log("time budget reached"); break
    t0 = time.perf_counter()
    ids = m.encode(tr["prompt"], max_length=64)
    nid, oid = tid(tr["new"][0])[0], tid(tr["old"][0])[0]
    wdiff = (qa.W_U[nid].float() - qa.W_U[oid].float())
    R = LastRunner(ids)  # F = block L output at the answer position
    v_pos = torch.arange(1, R.T, device=DEV)  # iso/data spread: every position but the first (attention sink)
    with torch.no_grad(), qa.ActivationRecorder(m.layers, at=LAYERS) as rec:
        m.forward(ids)
        Hc = {l: rec.activations[l][0].float() for l in LAYERS}
    rfn = lambda h: m._final_norm(h).float() @ wdiff  # noqa: E731  r = logit(new) - logit(old), h [B, d]
    ent = {k: tr[k] for k in ("func", "src", "tgt", "ctrl", "prompt", "old", "new")}
    ent.update({"T": R.T, "clean_r": tr["clean"]["r"], "layers": {}})
    for l in LAYERS:
        t1 = time.perf_counter()
        Dsw = swap_push(Hc[l], tr["src"], tr["tgt"], l)
        Dct = swap_push(Hc[l], tr["ctrl"], tr["tgt"], l)
        fro = Dsw.norm()
        gen = torch.Generator().manual_seed(A.seed + 1000 * l + ti + 100 * si)
        rnd = [torch.randn(D, generator=gen).to(DEV) for _ in range(2)]
        rnd = [x / x.norm() for x in rnd]
        A1 = [Dsw, Dct] + [Dsw.norm(dim=-1, keepdim=True) * x for x in rnd]
        for tn in ("iso", "data"):
            for j in range(2):
                Z = torch.zeros(R.T, D, device=DEV, dtype=dtype)
                Z[v_pos] = dirs[l][tn][j]
                A1.append(Z * (fro / Z.norm()))
        A1 = torch.stack(A1).to(dtype)  # [8, T, d]
        F0 = qa.forward_F(R, l, torch.zeros_like(A1[:1]))  # clean block-L output at the answer position [1, d]
        with torch.enable_grad():
            h = F0[0].detach().clone().requires_grad_(True)
            u = torch.autograd.grad(rfn(h[None])[0], h)[0].detach()
        rows, Jf, Qf, chk, blocks = anatomy_swap(R, l, A1, u)
        r1, r2 = qa.d2(rfn, (F0.expand(len(A1), -1).contiguous(),), (Jf,), (Jf,))
        uJ, uQ = (u[None] * Jf).sum(-1), (u[None] * Qf).sum(-1)
        g1, g2 = uJ, uQ + r2
        for i, row in enumerate(rows):
            row.update({"type": TYPES[i], "uJ": float(uJ[i]), "uQ": float(uQ[i]), "readout_curv": float(r2[i]),
                        "g1": float(g1[i]), "g2": float(g2[i]), "fro": float(A1[i].norm())})
        # gates: split sums to Q; u.J matches the readout jvp; swap jets vs finite differences and vs stored S2 jets
        hh = 0.05
        with torch.no_grad():
            rp, r0_, rm = (float(rfn(qa.forward_F(R, l, x * A1[:1]))[0]) for x in (hh, 0.0, -hh))
        gates = {"site_gate": max(r_["site"]["gate"] for r_ in rows), "block_gate": max(r_["block"]["gate"] for r_ in rows),
                 **chk, "uJ_vs_readout_jvp": float(((r1 - uJ).abs() / uJ.abs().clamp_min(1e-30)).max()),
                 "r0_vs_s2": r0_ - tr["clean"]["r"],
                 "g1_vs_fd": abs(float(g1[0]) - (rp - rm) / (2 * hh)) / max(abs(float(g1[0])), 1e-30),
                 "g2_vs_fd": abs(float(g2[0]) - (rp - 2 * r0_ + rm) / hh ** 2) / max(abs(float(g2[0])), 1e-30)}
        if A.s2_gate:
            e = tr["single"][str(l)]
            gates.update({"g1_vs_s2": abs(float(g1[0]) - e["g1"]) / abs(e["g1"]), "g2_vs_s2": abs(float(g2[0]) - e["g2"]) / abs(e["g2"])})
        e2 = tr["single"].get(str(l))  # None for the tiny test (its layers are not S2 layers)
        ent["layers"][str(l)] = {"blocks": blocks, "rows": rows, "gates": gates, "secs": round(time.perf_counter() - t1, 1),
                                 "s2": e2 and {"beta": e2["beta"], "g1": e2["g1"], "g2": e2["g2"], "lams": s2[0]["lams"],
                                               "r": [o["r"] for o in e2["out"]], "succ": [o["succ"] for o in e2["out"]]}}
        sw = rows[0]
        log(f"trial {ti} {tr['func']} {tr['src']}->{tr['tgt']} L{l} T={R.T} {time.perf_counter() - t1:.0f}s | gates "
            + " ".join(f"{k}={v:.1e}" for k, v in gates.items())
            + f" | swap Q/J={sw['Q_over_J']:.3f} uQ/2uJ={sw['uQ'] / 2 / sw['uJ']:+.3f} block best_err="
            f"{sw['block']['best_single_err']:.2f} n90={sw['block']['n90']} ci/sqrt(pr)={sw['block']['ci'] / sw['block']['pr'] ** 0.5:.2f}"
            f" | u: top block {sw['scal_block_u']['top_share'].get(1, float('nan')):.2f} n90={sw['scal_block_u']['n90']}"
            f" | iso best_err={rows[4]['block']['best_single_err']:.2f}")
        del A1, Jf, Qf, rows
    ent["secs"] = round(time.perf_counter() - t0, 1)
    res["trials"].append(ent)
    del R
    if DEV == "cuda":
        torch.cuda.empty_cache()
    res["secs"] = round(time.perf_counter() - T00, 1)
    res["peak_mem_gb"] = round(qa.mem_gb(peak=True), 1)
    fname.write_text(json.dumps(res, default=str))
    log(f"wrote {fname} ({len(res['trials'])} trials, {ent['secs']:.0f}s this trial)")
# gate: a dependent job (afterok) runs only if the split is exact (u-readout head/neuron sums and FD are reported only:
# u.Q can be ~0 for the baseline directions, and FD carries O(h^2) error)
LIM = {"site_gate": 1e-2, "block_gate": 1e-2, "head_sum_vs_jvp": 1e-2, "neuron_sum_vs_jvp": 1e-2, "resid_structure": 1e-4,
       "uJ_vs_readout_jvp": 1e-3, "g1_vs_s2": 2e-2, "g2_vs_s2": 1e-1}
bad = [(t["func"], t["src"], t["tgt"], l, k, v) for t in res["trials"] for l, e in t["layers"].items()
       for k, v in e["gates"].items() if k in LIM and not abs(v) < LIM[k]]
log("SWAP_SPLIT_DONE", f"peak_mem={qa.mem_gb(peak=True):.1f} GB", f"gate failures: {len(bad)}", bad[:10])
if bad or not res["trials"]:
    log("GATES FAILED (or no trial finished): dependent jobs must not run")
    sys.exit(3)
