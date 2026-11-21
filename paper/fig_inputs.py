#!/usr/bin/env python3
"""Figure inputs that come from raw outputs: panel (b) of F2 and panels (b)-(d) of F6. CPU only.
Reads $QLENS_OUT/so_{qwen3-8b,qwen3-14b,qwen3-32b,llama3_1-8b}/D1_raw.pt and $QLENS_OUT/atp_q_qwen3-8b/{ioi,fact}_*.pt.
Writes results/fig_inputs/F2b_norms.json and results/fig_inputs/F6_atp_q.npz, which paper/figs_paper.py reads, so the
figures rebuild from results/ alone. The loading code is that of the paper's figure script; only where its result is
kept differs (floats are stored exactly).
Run:  QLENS_OUT=<outputs dir> python paper/fig_inputs.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
OUT = Path(__file__).resolve().parents[1] / "results" / "fig_inputs"
OUT.mkdir(parents=True, exist_ok=True)

# F2 (b): pushed at each model's earliest push block, the median over cases of ||JU|| and ||Q(U,U)|| at each read-out
# block (figs_paper.py divides by the value one block after the push)
f2b = {}
for d in ("so_qwen3-8b", "so_qwen3-14b", "so_qwen3-32b", "so_llama3_1-8b"):
    raw = torch.load(W / d / "D1_raw.pt", map_location="cpu", weights_only=False)
    rec = {}
    for key in ("norm_J", "norm_Q"):
        s = min(raw)
        tg = raw[s]["targets"]
        med = np.array([float(np.median(raw[s]["iso"][key].numpy()[:, :, i])) for i in range(len(tg))])
        assert tg[0] - s == 1
        rec["k"] = [int(t_ - s) for t_ in tg]
        rec[key] = [float(v) for v in med]
    f2b[d] = rec
(OUT / "F2b_norms.json").write_text(json.dumps(f2b, indent=1) + "\n")

# F6 (b)-(d): attribution patching on Qwen3-8B, per prompt pair (as src/flag_then_patch.py)
ATP = str(W / "atp_q_qwen3-8b")
K_ = 32


def atp_load(task):
    out, spl = [], {0: [], 1: []}
    for f in sorted(glob.glob(f"{ATP}/{task}_*.pt")):
        r = torch.load(f, weights_only=False)
        D = float(r["m_clean"] - r["m_corrupt"])
        if abs(D) <= 0.5:
            continue
        kind, null, fam = np.asarray(r["kind"]), np.asarray(r["null"], bool), np.asarray(r["fam"])
        q = np.full(len(kind), np.nan)
        q[np.nonzero(fam == 0)[0]] = np.asarray(r["stoch"][0])[:K_].mean(0)
        sel = np.isin(kind, (0, 1)) & ~null
        g = lambda v: np.asarray(v, np.float64)[sel] / D  # noqa: E731
        out.append(dict(a=g(r["actual"]), atp=g(r["atp"]), J=g(r["J"]), Q=g(r["Q"]), qs=g(q)))
        # exact Q-lens split of each analysed site's 1/2 Q, in order of distance from the patch
        L = r["split"].shape[1] - 1
        for b_, s in enumerate(r.get("split_sites", [])):
            kd, l = int(r["kind"][s]), int(r["layer"][s])
            if kd not in (0, 1):
                continue
            tm = np.nan_to_num(np.asarray(r["split"][b_], np.float64))
            seq = ([r["split_own"][b_]] if kd == 0 else []) + list(tm[l + 1:L]) + [tm[L]]
            ab = np.abs(seq)
            if ab.sum() > 0:
                spl[kd].append(np.cumsum(ab) / ab.sum())
    return out, spl


arrs = {}
for task in ("ioi", "fact"):
    Pp, spl = atp_load(task)
    for i, p in enumerate(Pp):
        for v, x in p.items():
            arrs[f"{task}.pair{i:03d}.{v}"] = x
    for kd in (0, 1):
        for j, x in enumerate(spl[kd]):
            arrs[f"{task}.split{kd}.{j:05d}"] = x
np.savez_compressed(OUT / "F6_atp_q.npz", **arrs)
print("wrote", OUT / "F2b_norms.json", OUT / "F6_atp_q.npz")
