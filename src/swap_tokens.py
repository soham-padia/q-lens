#!/usr/bin/env python3
"""Token readout of the swap jets: which vocabulary tokens the first-order term J and the second-order term Q of the
logits raise and lower along an actual S2 T swap push.

Swap push Delta_l = (c_flip - c) V^T at one layer l, all positions, exactly as second_order.py / swap_split.py build it
(lens pair V = [v_src, v_tgt] from the fitted mean J-lens fit_v1, c = clean coordinates). F(lam) = block-63 output at
the LAST position with lam * Delta_l added to block l's output; logits z(lam) = W_U final_norm(F(lam)) (through the norm,
as S2's r = z[new] - z[old]). Jets by nested forward-mode jvp (math SDPA, TF32 off, fp32):
  z1 = dz/dlam = Dz[J_F]           z2 = d2z/dlam2 = Dz[Q_F] + D2z[J_F, J_F]
Centred: subtract the mean over the vocabulary (ids < len(tokenizer); the embedding-padding rows are left out).
Reported per (swap, layer): top-k tokens raised / lowered by J (= z1) and by Q/2 (= z2 / 2); ranks of the target answer
(new) and source answer (old) under J, Q/2, the second-order prediction J lam* + Q lam*^2 / 2 at the actual flip dose
lam* (S2 grid, linear interpolation), and the actual logit change at lam*. Nothing is filtered: every token is classed
as word / punct (no letter or digit) / space (whitespace only) / byte (undecodable piece) and the classes are counted.

Gates: z1[new]-z1[old], z2[new]-z2[old] vs the stored S2 g1, g2 (2e-2, 1e-1 as swap_split); r0 vs S2; z(0) vs the HF
logits; logit-space J and Q vs central finite differences over the whole centred vocabulary (h = 0.05, relative L2;
2e-2 and 1e-1), plus Richardson (h, 2h) and the top-k overlap between the jvp and FD lists (reported).

  python src/swap_tokens.py --model Qwen/Qwen3.8-27B --out DIR [--layers 24,40,56]
       [--trials capital:France:China,language:France:China,continent:France:China]
Tiny CPU test: QANAT_DEVICE=cpu python src/swap_tokens.py --model <tiny dir> --out DIR --layers 5,3 --lens random
               --s2-gate 0
Table: python src/swap_tokens_table.py DIR/swap_tokens.json --md results/tables/swap_token_tables_full.md
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="24,40,56")
ap.add_argument("--trials", default="capital:France:China,language:France:China,continent:France:China",
                help="func:src:tgt,... (S2 T trials)")
ap.add_argument("--s2-glob", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "layer_pick_v1", "t*", "S2.json"),
                help="dense-grid S2 run (same trials and jets as the original; finer flip doses)")
ap.add_argument("--lens", default="fit", choices=["fit", "random"], help="random = tiny-model test only")
ap.add_argument("--lens-dir", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "fit_v1"))
ap.add_argument("--s2-gate", type=int, default=1)
ap.add_argument("--tf32", type=int, default=0)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--h", type=float, default=0.05)
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["q_anatomy.py", "--model", A.model, "--out", str(OUT / "qa_import"), "--stages", "none",
            "--tf32", str(A.tf32), "--sources", "48", "--a3-sources", "48", "--check-source", "48"]
import torch  # noqa: E402
import q_anatomy as qa  # noqa: E402

m, L, D, DEV, dtype, tok, log = qa.m, qa.L, qa.D, qa.DEV, qa.dtype, qa.tok, qa.log
LAYERS = [int(x) for x in A.layers.split(",")]
T00 = time.perf_counter()
NV = min(qa.W_U.shape[0], len(tok))
WU = qa.W_U[:NV].float()


def tid(word):  # = second_order.tid
    for w in (" " + word, word):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], True
    return tok.encode(" " + word, add_special_tokens=False)[0], False


def flip_actual(r0, lams, rs):  # = layer_pick.flip_actual (interpolated first crossing of r = 0)
    pl, pr = 0.0, r0
    for lam, rv in zip(lams, rs):
        if rv > 0:
            return pl + (lam - pl) * (-pr) / (rv - pr) if rv != pr else lam
        pl, pr = lam, rv
    return math.inf


def flip_JQ(r0, g1, g2):
    disc = g1 * g1 - 2 * g2 * r0
    if abs(g2) < 1e-12:
        return -r0 / g1 if g1 > 0 else math.inf
    if disc < 0:
        return math.inf
    pos = [x for x in ((-g1 + s * math.sqrt(disc)) / g2 for s in (1, -1)) if x > 0]
    return min(pos) if pos else math.inf


# ---------------------------------------------------------------- trials and lens vectors (as swap_split.py)
s2 = [json.load(open(fp)) for fp in sorted(glob.glob(A.s2_glob))]
LAMS = s2[0]["lams"]
CN = list(s2[0]["countries"])
byk = {(r["func"], r["src"], r["tgt"]): r for d in s2 for r in d["twohop"] if "single" in r}
want = [tuple(x.split(":")) for x in A.trials.split(",")]
mine = [byk[k] for k in want]
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
log(f"swap_tokens: {len(mine)} trials {want}, layers {LAYERS}, vocab {NV} of {qa.W_U.shape[0]} rows, lens {A.lens}")


def swap_push(Hc, a, b, l):
    V = torch.stack([JV[l][WI[a]], JV[l][WI[b]]], 1)
    c = Hc @ torch.linalg.pinv(V).T
    return (c.flip(-1) - c) @ V.T


class LastRunner(qa.Runner):  # = swap_split.LastRunner: F at the last position only
    def __init__(self, ids1):
        orig = qa.valid_position_mask
        qa.valid_position_mask = lambda T_: torch.arange(T_) == T_ - 1
        try:
            super().__init__(ids1)
        finally:
            qa.valid_position_mask = orig


def zfn(h):  # logits through the final norm, [B, d] -> [B, NV]
    return m._final_norm(h).float() @ WU.T


def cen(z):
    return z - z.mean(-1, keepdim=True)


def kind(s):
    if "�" in s:
        return "byte"
    if s.strip() == "":
        return "space"
    if not any(ch.isalnum() for ch in s):
        return "punct"
    return "word"


def top(v, k, sign):
    vals, ids = (sign * v).topk(k)
    return [{"id": int(i), "tok": tok.decode([int(i)]), "val": float(sign * x), "kind": kind(tok.decode([int(i)]))}
            for x, i in zip(vals, ids)]


def ranks(v, i):  # 1 = most raised / most lowered
    return {"up": int((v > v[i]).sum()) + 1, "down": int((v < v[i]).sum()) + 1}


def rel(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


res = {"args": vars(A), "layers": LAYERS, "vocab": NV, "defs": __doc__, "trials": []}
fname = OUT / "swap_tokens.json"
K = A.k
for tr in mine:
    t0 = time.perf_counter()
    ids = m.encode(tr["prompt"], max_length=64)
    (nid, n1), (oid, o1) = tid(tr["new"][0]), tid(tr["old"][0])
    R = LastRunner(ids)
    with torch.no_grad(), qa.ActivationRecorder(m.layers, at=LAYERS) as rec:
        m.forward(ids)
        Hc = {l: rec.activations[l][0].float() for l in LAYERS}
    try:
        with torch.no_grad(), qa.math_sdpa():
            zh = qa.hf(input_ids=ids).logits[0, -1, :NV].float()
    except Exception as e:  # noqa: BLE001
        log(f"HF logits check skipped: {e!r}"); zh = None
    ent = {k: tr[k] for k in ("func", "src", "tgt", "prompt", "old", "new")}
    ent.update({"T": R.T, "new_id": nid, "old_id": oid, "new_single_token": n1, "old_single_token": o1,
                "new_tok": tok.decode([nid]), "old_tok": tok.decode([oid]), "clean_r": tr["clean"]["r"], "layers": {}})
    for l in LAYERS:
        t1 = time.perf_counter()
        A1 = swap_push(Hc[l], tr["src"], tr["tgt"], l)[None].to(dtype)  # [1, T, d]
        f = lambda d: R.run(1, l + 1, {("in", l + 1): d}, [("L", None)], stop=False)  # noqa: E731
        (Jf,), (Qf,) = qa.d2(f, (torch.zeros_like(A1),), (A1,), (A1,))
        F0 = qa.forward_F(R, l, torch.zeros_like(A1))
        z0 = zfn(F0)
        z1, zjj = qa.d2(zfn, (F0,), (Jf,), (Jf,))
        zq = qa.jvp(zfn, (F0,), (Qf,))[1]
        z0, z1, z2 = z0[0], z1[0], (zq + zjj)[0]
        r0, g1, g2 = (float(z[nid] - z[oid]) for z in (z0, z1, z2))
        # finite differences in logit space
        with torch.no_grad():
            zx = {x: zfn(qa.forward_F(R, l, x * A1))[0] for x in (2 * A.h, A.h, -A.h, -2 * A.h)}
        h = A.h
        d1 = (zx[h] - zx[-h]) / (2 * h); d1b = (zx[2 * h] - zx[-2 * h]) / (4 * h)
        d2_ = (zx[h] - 2 * z0 + zx[-h]) / h ** 2; d2b = (zx[2 * h] - 2 * z0 + zx[-2 * h]) / (4 * h ** 2)
        c1, c2, cd1, cd2 = cen(z1), cen(z2), cen(d1), cen(d2_)
        gates = {"r0_vs_s2": r0 - tr["clean"]["r"],
                 "J_vs_fd": rel(c1, cd1), "Q_vs_fd": rel(c2, cd2),
                 "J_vs_fd_richardson": rel(c1, cen((4 * d1 - d1b) / 3)),
                 "Q_vs_fd_richardson": rel(c2, cen((4 * d2_ - d2b) / 3))}
        if zh is not None:
            gates["z0_vs_hf"] = float((z0 - zh).abs().max() / zh.abs().max())
        e2 = tr["single"].get(str(l))
        if A.s2_gate:
            gates.update({"g1_vs_s2": abs(g1 - e2["g1"]) / abs(e2["g1"]), "g2_vs_s2": abs(g2 - e2["g2"]) / abs(e2["g2"])})
        lam = flip_actual(tr["clean"]["r"], LAMS, [o["r"] for o in e2["out"]]) if e2 else 1.0
        lamq, lamj = flip_JQ(r0, g1, g2), (-r0 / g1 if g1 > 0 else math.inf)
        lam_use = lam if math.isfinite(lam) else 1.0
        P = lam_use * c1 + 0.5 * lam_use ** 2 * c2
        with torch.no_grad():
            Act = cen(zfn(qa.forward_F(R, l, lam_use * A1))[0] - z0)
        hq = 0.5 * c2
        ov = {s: len({t["id"] for t in top(hq, K, sg)} & {t["id"] for t in top(0.5 * cd2, K, sg)}) / K
              for s, sg in (("up", 1), ("down", -1))}
        ent["layers"][str(l)] = {
            "lam_act": lam, "lam_used": lam_use, "lam_JQ": lamq, "lam_J": lamj, "r0": r0, "g1": g1, "g2": g2,
            "beta": e2 and e2["beta"], "gates": gates, "fd_topk_overlap_halfQ": ov,
            "norms": {"J": float(c1.norm()), "halfQ": float(hq.norm()), "pred_at_lam": float(P.norm()),
                      "actual_at_lam": float(Act.norm())},
            "cos": {"J_halfQ": cos(c1, hq), "pred_actual": cos(P, Act), "Jlam_actual": cos(lam_use * c1, Act)},
            "pred_vs_actual_rel": rel(Act, P), "J_vs_actual_rel": rel(Act, lam_use * c1),
            "top": {"J_up": top(c1, K, 1), "J_down": top(c1, K, -1), "halfQ_up": top(hq, K, 1), "halfQ_down": top(hq, K, -1),
                    "pred_up": top(P, K, 1), "pred_down": top(P, K, -1), "act_up": top(Act, K, 1), "act_down": top(Act, K, -1)},
            "ranks": {nm: {"new": ranks(v, nid), "old": ranks(v, oid)}
                      for nm, v in (("J", c1), ("halfQ", hq), ("pred", P), ("actual", Act))},
            "new_vals": {"J": float(c1[nid]), "halfQ": float(hq[nid])}, "old_vals": {"J": float(c1[oid]), "halfQ": float(hq[oid])},
            "secs": round(time.perf_counter() - t1, 1)}
        E = ent["layers"][str(l)]
        log(f"{tr['func']} {tr['src']}->{tr['tgt']} L{l} lam*={lam:.2f} (JQ {lamq:.2f}) {E['secs']:.0f}s | gates "
            + " ".join(f"{k}={v:.1e}" for k, v in gates.items())
            + f" | topk overlap FD {ov} | ranks new J/Q/pred/act "
            + "/".join(str(E["ranks"][x]["new"]["up"]) for x in ("J", "halfQ", "pred", "actual"))
            + " old(down) " + "/".join(str(E["ranks"][x]["old"]["down"]) for x in ("J", "halfQ", "pred", "actual"))
            + f" | Q/2 up: {[t['tok'] for t in E['top']['halfQ_up']]} down: {[t['tok'] for t in E['top']['halfQ_down']]}")
        del Jf, Qf, zx, A1
    ent["secs"] = round(time.perf_counter() - t0, 1)
    res["trials"].append(ent)
    del R
    if DEV == "cuda":
        torch.cuda.empty_cache()
    res["secs"] = round(time.perf_counter() - T00, 1)
    res["peak_mem_gb"] = round(qa.mem_gb(peak=True), 1)
    fname.write_text(json.dumps(res, default=str))
    log(f"wrote {fname}")
LIM = {"J_vs_fd": 2e-2, "Q_vs_fd": 1e-1, "g1_vs_s2": 2e-2, "g2_vs_s2": 1e-1, "z0_vs_hf": 1e-3}
bad = [(t["func"], l, k, v) for t in res["trials"] for l, e in t["layers"].items()
       for k, v in e["gates"].items() if k in LIM and not abs(v) < LIM[k]]
bad += [(t["func"], l, "r0_vs_s2", e["gates"]["r0_vs_s2"]) for t in res["trials"] for l, e in t["layers"].items()
        if A.s2_gate and not abs(e["gates"]["r0_vs_s2"]) < 1e-3]
log("SWAP_TOKENS_DONE", f"peak_mem={qa.mem_gb(peak=True):.1f} GB", f"gate failures: {len(bad)}", bad[:10])
if bad or not res["trials"]:
    sys.exit(3)
