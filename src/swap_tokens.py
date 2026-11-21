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

Target / source matching: the strict single id tid(new[0]) (as v1 and the paper's table) and, in addition, every
first-token variant of every word in new / old (S2's own success rule, second_order.variants): ranks "new_any" /
"old_any" are the best rank over that id set.

Scale-out (v2): --trials all = every S2 two-hop record that has the "single" jets (147 on layer_pick_v1), in the
canonical order (sorted glob, record order within). --layers "a,b;c,d" runs passes (all trials at a,b first, then
c,d). Results are appended one line per (swap, layer) to OUT/swap_tokens.jsonl and a rerun skips the (swap, layer)
pairs already there (resumable); OUT/swap_tokens.json is rebuilt from the JSONL at the end in v1's layout.
--fd-every N: the full-vocabulary FD gate runs only on canonical indices 0, N, 2N, ... plus the three France->China
trials (N = 1, the default, = every trial, as v1). The S2 / r0 / z(0) gates run on every case.

  python src/swap_tokens.py --model Qwen/Qwen3.8-27B --out DIR [--layers 24,40,56]
       [--trials capital:France:China,language:France:China,continent:France:China | all] [--fd-every 15]
Tiny CPU test: QANAT_DEVICE=cpu python src/swap_tokens.py --model <tiny dir> --out DIR --layers 5,3 --lens random
               --s2-gate 0
Table: python src/swap_tokens_table.py DIR/swap_tokens.json --md results/tables/swap_token_tables_full.md
Rates: python src/swap_tokens_rates.py DIR/swap_tokens.jsonl --md results/tables/swap_token_rates_tables.md
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
                help="func:src:tgt,... (S2 T trials), or 'all'")
ap.add_argument("--fd-every", type=int, default=1,
                help="FD gate on canonical indices 0, N, 2N, ... plus the France->China trials (1 = every trial)")
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
PASSES = [[int(x) for x in p.split(",") if x.strip()] for p in A.layers.split(";") if p.strip()]
LAYERS = sorted({l for p in PASSES for l in p})
T00 = time.perf_counter()
NV = min(qa.W_U.shape[0], len(tok))
WU = qa.W_U[:NV].float()


def tid(word):  # = second_order.tid
    for w in (" " + word, word):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], True
    return tok.encode(" " + word, add_special_tokens=False)[0], False


def variants(word):  # = second_order.variants (order_ops off): first-token ids of w / ' w', lower, Capitalised
    ids = set()
    for v in {word, word.lower(), word.capitalize()}:
        for p in ("", " "):
            e = tok.encode(p + v, add_special_tokens=False)
            if e:
                ids.add(e[0])
    return sorted(ids)


def vset(ws):  # = second_order's S2 vset
    return sorted({x for w in ws for x in variants(w)})


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
allrec = [r for d in s2 for r in d["twohop"] if "single" in r]  # canonical order
IDX = {(r["func"], r["src"], r["tgt"]): i for i, r in enumerate(allrec)}
byk = {(r["func"], r["src"], r["tgt"]): r for r in allrec}
want = list(IDX) if A.trials.strip() == "all" else [tuple(x.split(":")) for x in A.trials.split(",")]
mine = [byk[k] for k in want]
FD_SET = {k for k, i in IDX.items() if i % max(A.fd_every, 1) == 0} | {k for k in IDX if k[1:] == ("France", "China")}
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
log(f"swap_tokens: {len(mine)} trials {want if len(want) <= 6 else 'all'}, passes {PASSES}, vocab {NV} of "
    f"{qa.W_U.shape[0]} rows, lens {A.lens}, FD gate on {len(FD_SET & set(want))} of {len(want)} trials "
    f"(fd_every {A.fd_every}: {sorted(IDX[k] for k in FD_SET & set(want))})")


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


def ranks_any(v, ids):  # best rank over an id set (S2 variants), with the id that attains it
    t = torch.tensor(ids, device=v.device)
    up = (v[None, :] > v[t][:, None]).sum(1) + 1
    dn = (v[None, :] < v[t][:, None]).sum(1) + 1
    iu, idn = int(up.argmin()), int(dn.argmin())
    return {"up": int(up[iu]), "up_id": ids[iu], "down": int(dn[idn]), "down_id": ids[idn]}


def rel(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


K = A.k
JL = OUT / "swap_tokens.jsonl"
fname = OUT / "swap_tokens.json"
META = ("func", "src", "tgt", "prompt", "old", "new")


def load_lines():  # valid lines only (a job killed mid-write leaves at most one broken last line)
    rows = []
    if JL.exists():
        for ln in JL.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                log(f"dropping a broken JSONL line ({len(ln)} chars)")
    return rows


old_rows = load_lines()
if JL.exists():  # rewrite clean (drops a broken tail) so appends start on a fresh line
    tmp = JL.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in old_rows))
    os.replace(tmp, JL)
done = {(r["func"], r["src"], r["tgt"], int(r["layer"])) for r in old_rows}
log(f"resume: {len(done)} (swap, layer) pairs already in {JL}")
fo = open(JL, "a")
n_new = 0
for P_ in PASSES:
    for tr in mine:
        key = (tr["func"], tr["src"], tr["tgt"])
        todo = [l for l in P_ if key + (l,) not in done]
        if not todo:
            continue
        t0 = time.perf_counter()
        do_fd = key in FD_SET
        ids = m.encode(tr["prompt"], max_length=64)
        (nid, n1), (oid, o1) = tid(tr["new"][0]), tid(tr["old"][0])
        nv, ov = vset(tr["new"]), vset(tr["old"])
        R = LastRunner(ids)
        with torch.no_grad(), qa.ActivationRecorder(m.layers, at=todo) as rec:
            m.forward(ids)
            Hc = {l: rec.activations[l][0].float() for l in todo}
        try:
            with torch.no_grad(), qa.math_sdpa():
                zh = qa.hf(input_ids=ids).logits[0, -1, :NV].float()
        except Exception as e:  # noqa: BLE001
            log(f"HF logits check skipped: {e!r}"); zh = None
        ent = {k: tr[k] for k in META}
        ent.update({"idx": IDX[key], "fd": do_fd, "T": R.T, "new_id": nid, "old_id": oid, "new_single_token": n1,
                    "old_single_token": o1, "new_tok": tok.decode([nid]), "old_tok": tok.decode([oid]),
                    "new_ids": nv, "old_ids": ov, "new_toks": [tok.decode([i]) for i in nv],
                    "old_toks": [tok.decode([i]) for i in ov], "new_old_overlap": sorted(set(nv) & set(ov)),
                    "clean_r": tr["clean"]["r"]})
        for l in todo:
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
            c1, c2 = cen(z1), cen(z2)
            hq = 0.5 * c2
            gates = {"r0_vs_s2": r0 - tr["clean"]["r"]}
            ov_fd = None
            if do_fd:  # finite differences in logit space, over the whole centred vocabulary
                with torch.no_grad():
                    zx = {x: zfn(qa.forward_F(R, l, x * A1))[0] for x in (2 * A.h, A.h, -A.h, -2 * A.h)}
                h = A.h
                d1 = (zx[h] - zx[-h]) / (2 * h); d1b = (zx[2 * h] - zx[-2 * h]) / (4 * h)
                d2_ = (zx[h] - 2 * z0 + zx[-h]) / h ** 2; d2b = (zx[2 * h] - 2 * z0 + zx[-2 * h]) / (4 * h ** 2)
                cd1, cd2 = cen(d1), cen(d2_)
                gates.update({"J_vs_fd": rel(c1, cd1), "Q_vs_fd": rel(c2, cd2),
                              "J_vs_fd_richardson": rel(c1, cen((4 * d1 - d1b) / 3)),
                              "Q_vs_fd_richardson": rel(c2, cen((4 * d2_ - d2b) / 3))})
                ov_fd = {s: len({t["id"] for t in top(hq, K, sg)} & {t["id"] for t in top(0.5 * cd2, K, sg)}) / K
                         for s, sg in (("up", 1), ("down", -1))}
                del zx
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
            vecs = (("J", c1), ("halfQ", hq), ("pred", P), ("actual", Act))
            E = {"lam_act": lam, "flipped": bool(math.isfinite(lam)), "lam_used": lam_use, "lam_JQ": lamq, "lam_J": lamj,
                 "r0": r0, "g1": g1, "g2": g2, "beta": e2 and e2["beta"], "gates": gates, "fd_topk_overlap_halfQ": ov_fd,
                 "norms": {"J": float(c1.norm()), "halfQ": float(hq.norm()), "pred_at_lam": float(P.norm()),
                           "actual_at_lam": float(Act.norm())},
                 "cos": {"J_halfQ": cos(c1, hq), "pred_actual": cos(P, Act), "Jlam_actual": cos(lam_use * c1, Act)},
                 "pred_vs_actual_rel": rel(Act, P), "J_vs_actual_rel": rel(Act, lam_use * c1),
                 "top": {"J_up": top(c1, K, 1), "J_down": top(c1, K, -1), "halfQ_up": top(hq, K, 1),
                         "halfQ_down": top(hq, K, -1), "pred_up": top(P, K, 1), "pred_down": top(P, K, -1),
                         "act_up": top(Act, K, 1), "act_down": top(Act, K, -1)},
                 "ranks": {nm: {"new": ranks(v, nid), "old": ranks(v, oid), "new_any": ranks_any(v, nv),
                                "old_any": ranks_any(v, ov)} for nm, v in vecs},
                 "new_vals": {"J": float(c1[nid]), "halfQ": float(hq[nid])},
                 "old_vals": {"J": float(c1[oid]), "halfQ": float(hq[oid])},
                 "secs": round(time.perf_counter() - t1, 1)}
            fo.write(json.dumps({**ent, "layer": l, "e": E}, default=str) + "\n"); fo.flush()
            done.add(key + (l,)); n_new += 1
            log(f"#{IDX[key]} {tr['func']} {tr['src']}->{tr['tgt']} L{l} lam*={lam:.2f} (JQ {lamq:.2f}) {E['secs']:.0f}s"
                f"{' FD' if do_fd else ''} | gates " + " ".join(f"{k}={v:.1e}" for k, v in gates.items())
                + (f" | topk overlap FD {ov_fd}" if do_fd else "")
                + " | ranks new J/Q/pred/act " + "/".join(str(E["ranks"][x]["new"]["up"]) for x, _ in vecs)
                + " any " + "/".join(str(E["ranks"][x]["new_any"]["up"]) for x, _ in vecs)
                + " old(down) " + "/".join(str(E["ranks"][x]["old"]["down"]) for x, _ in vecs)
                + f" | Q/2 up: {[t['tok'] for t in E['top']['halfQ_up'][:5]]}")
            del Jf, Qf, A1
        del R
        if DEV == "cuda":
            torch.cuda.empty_cache()
        log(f"#{IDX[key]} done {len(todo)} layers in {time.perf_counter() - t0:.1f}s; total {n_new} new, "
            f"{len(done)} in file, {(time.perf_counter() - T00) / 60:.1f} min")
fo.close()

# consolidated JSON in v1's layout (trials in canonical order, layers in run order)
rows = load_lines()
trials = {}
for r in rows:
    k = (r["func"], r["src"], r["tgt"])
    t = trials.setdefault(k, {**{x: r[x] for x in r if x not in ("layer", "e")}, "layers": {}})
    t["layers"][str(r["layer"])] = r["e"]
res = {"args": vars(A), "layers": sorted({int(r["layer"]) for r in rows}), "vocab": NV, "defs": __doc__,
       "trials": sorted(trials.values(), key=lambda t: t.get("idx", 0)),
       "secs_this_run": round(time.perf_counter() - T00, 1), "peak_mem_gb": round(qa.mem_gb(peak=True), 1)}
fname.write_text(json.dumps(res, default=str))
log(f"wrote {fname} ({len(trials)} trials, {len(rows)} swap-layer cases; {n_new} new this run)")
LIM = {"J_vs_fd": 2e-2, "Q_vs_fd": 1e-1, "g1_vs_s2": 2e-2, "g2_vs_s2": 1e-1, "z0_vs_hf": 1e-3}
bad = [(t["func"], t["src"], t["tgt"], l, k, v) for t in res["trials"] for l, e in t["layers"].items()
       for k, v in e["gates"].items() if k in LIM and not abs(v) < LIM[k]]
bad += [(t["func"], t["src"], t["tgt"], l, "r0_vs_s2", e["gates"]["r0_vs_s2"]) for t in res["trials"]
        for l, e in t["layers"].items() if A.s2_gate and not abs(e["gates"]["r0_vs_s2"]) < 1e-3]
nfd = sum(1 for t in res["trials"] for e in t["layers"].values() if "J_vs_fd" in e["gates"])
log("SWAP_TOKENS_DONE", f"peak_mem={qa.mem_gb(peak=True):.1f} GB", f"cases {len(rows)} (FD-gated {nfd})",
    f"gate failures: {len(bad)}", bad[:10])
if bad or not res["trials"]:
    sys.exit(3)
