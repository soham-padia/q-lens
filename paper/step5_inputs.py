#!/usr/bin/env python3
"""Step 5: numbers-only copies of the raw outputs that paper/claims_v84.py and paper/f8_lines.py read.

Reads $QLENS_OUT (default ./outputs) and writes results/step5/ (same layout as the outputs) and
results/fig_inputs/F8_example.json. CPU only, no model is loaded.

  lean_v1/summary.json                          copied (src/summarize_lean_qlens.py; its E4 examples are Mathlib)
  lean_v1/<model>/E23_<RH,ML>.json, gate2_*     the fields the E2/E3 point statistics read: indices, edit-class labels
                                                and numbers (no token ids)
  resp_map_v2/resp_v2_<corpus>.json             per context: span count with one flag per span (nl: the span ends a
                                                line), and per block and target the pair values (i, j, D2, Q, pQ, Q3),
                                                timings and the HVP check. No text, token ids, targets or file names.
                                                The RH input is $QLENS_OUT/lean_v1/data/resp_map_v2/resp_v2_RH.json.
  resp_map_v2/fd_precision/fd_float*.json       copied (one Mathlib context; numbers and span-index pairs)
  resp_map_v2/fd_precision/Q_fp32_vs_fp64.json  ||Q32 - Q64|| / ||Q64|| per pair-block, from Q_float{32,64}.pt
  resp_map_v2/cost_curve/cost_curve.json        copied (timings only)
  orderops_layers/ctrl_order-ops.json           copied (prompts are the J-lens order-ops evaluation set)
  fig_inputs/F8_example.json                    the Mathlib context of figure F8 (a): its lines, t* and file name

RH is the research library (no licence): from its outputs only numbers and indices are written, and the script stops
if a string other than a fixed label, or an integer list longer than two, would be written for it.
Run: QLENS_OUT=<outputs dir> python paper/step5_inputs.py
"""
import json
import os
import shutil
import statistics as st
from pathlib import Path

import torch

W = Path(os.environ.get("QLENS_OUT", "outputs"))
R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "step5"
T = True
MODELS = ("goedel-prover-v2-8b", "qwen3-8b-base")
E23 = {"corpus": T, "layers": T,
       "cases": [{"ci": T, "cand": T, "cls": T, "g_full": T,
                  "L": {"*": {"lam_star": T, "lam_J": T, "lam_JQ": T, "r_edit": T, "g_tail1": T,
                              "fd": {"rel_g1": T, "rel_g2": T}}}}],
       "pairs": [{"ci": T, "L": {"*": {"cos": T, "cos_mis": T, "cos_arb": T, "rand_cos": T, "rel": T, "rand_rel": T,
                                       "cos_s": T, "rel_s": T}}}]}
GATE2 = {"corpus": T, "layers": T,
         "E2": {"*": {"n": T, "med_rel_g1": T, "med_rel_g2": T, "pass": T, "max_g0_diff": T}},
         "E3": {"*": {"n": T, "med_best_cos": T, "pass": T, "max_cos_diff": T}}}
RESP = {"model": T, "corpus": T, "layers": T, "lams": T,
        "contexts": [{"ci": T, "spans": "NL",
                      "L": {"*": {"*": {"pairs": [{"i": T, "j": T, "D2": T, "Q": T, "pQ": T, "Q3": T}],
                                        "times": T, "hvp_vs_jvp": T}}}}]}
LABELS = {"RH", "rel", "op", "type", "const", "quant", "minmax", "fn", "swap", "Goedel-LM/Goedel-Prover-V2-8B"}


def keep(x, spec):
    """The parts of x named by spec (dict: key -> spec, '*' = every key; [spec] = every list item; T = all)."""
    if spec is T:
        return x
    if spec == "NL":  # spans -> one flag each: the span ends a line (used to drop pieces of one cut line)
        return [{"nl": s["text"].endswith("\n")} for s in x]
    if isinstance(spec, list):
        return [keep(v, spec[0]) for v in x]
    out = {}
    for k, v in x.items():
        s = spec.get(k, spec.get("*"))
        if s is not None:
            out[k] = keep(v, s)
    return out


def guard(x, where):
    """Numbers only: strings must be fixed labels, integer lists at most two long (span-index pairs), except the
    list of blocks."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k != "layers":  # block indices
                guard(v, f"{where}.{k}")
    elif isinstance(x, list):
        assert not (len(x) > 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in x)), where
        for v in x:
            guard(v, where + "[]")
    elif isinstance(x, str):
        assert x in LABELS, f"string in research-library output at {where}"


def dump(obj, rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, separators=(",", ":")))
    return p


def copy(rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(W / rel, p)
    return p


def spear(a, b):  # as paper/f8_lines.py
    def rank(xs):
        o = sorted(range(len(xs)), key=lambda k: xs[k])
        r = [0.0] * len(xs)
        for n, k in enumerate(o):
            r[k] = n
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def main():
    written = [copy("lean_v1/summary.json")]
    for tag in MODELS:
        for cor in ("RH", "ML"):
            e = keep(json.loads((W / "lean_v1" / tag / f"E23_{cor}.json").read_text()), E23)
            g = keep(json.loads((W / "lean_v1" / tag / f"gate2_{cor}.json").read_text()), GATE2)
            if cor == "RH":
                guard(e, f"E23_RH {tag}")
                guard(g, f"gate2_RH {tag}")
            written += [dump(e, f"lean_v1/{tag}/E23_{cor}.json"), dump(g, f"lean_v1/{tag}/gate2_{cor}.json")]
    raw = {c: W / "resp_map_v2" / f"resp_v2_{c}.json" for c in ("ML", "FW", "MLnull", "FWnull")}
    raw["RH"] = W / "lean_v1" / "data" / "resp_map_v2" / "resp_v2_RH.json"
    for c, p in raw.items():
        r = keep(json.loads(p.read_text()), RESP)
        if c == "RH":
            guard(r, "resp_v2_RH")
        written.append(dump(r, f"resp_map_v2/resp_v2_{c}.json"))
    for dt in ("float32", "float64"):
        written.append(copy(f"resp_map_v2/fd_precision/fd_{dt}.json"))
    q32 = torch.load(W / "resp_map_v2" / "fd_precision" / "Q_float32.pt", weights_only=False)
    q64 = torch.load(W / "resp_map_v2" / "fd_precision" / "Q_float64.pt", weights_only=False)
    rd = [float((q32[k].double() - q64[k].double()).norm() / q64[k].double().norm()) for k in q64]
    written.append(dump({"pair_blocks": [str(k) for k in q64], "rel_diff": rd},
                        "resp_map_v2/fd_precision/Q_fp32_vs_fp64.json"))
    written.append(copy("resp_map_v2/cost_curve/cost_curve.json"))
    written.append(copy("orderops_layers/ctrl_order-ops.json"))

    # figure F8 (a): the Mathlib context whose path-averaged-Q Spearman at block 29 is the lower median (f8_lines.py)
    ML = json.loads(raw["ML"].read_text())
    sc = []
    for k, c in enumerate(ML["contexts"]):
        P29 = c["L"]["29"]["tstar"]["pairs"]
        if any(p["Q3"] is None for p in P29):
            continue
        sc.append((spear([sum(p["pQ"]) / len(p["pQ"]) for p in P29], [p["D2"] for p in P29]), k))
    sc.sort()
    kx = sc[(len(sc) - 1) // 2][1]
    ctx = ML["contexts"][kx]
    ex = {"corpus": "Mathlib (Apache-2.0)", "context": kx, "file": ctx["file"], "tstar": ctx["tstar"],
          "lines": [s["text"] for s in ctx["spans"]]}
    p = R / "fig_inputs" / "F8_example.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n")
    written.append(p)
    for q in written:
        print(f"{q.stat().st_size:>9}  {q.relative_to(R.parent)}")


if __name__ == "__main__":
    main()
