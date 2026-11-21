"""Scale ladder comparison of the exact second-order stages (check, B1, B2x, B4f, B4c) across Qwen models.

Layers in each run were chosen as round(l64 * n_layers / 64) from 64-layer nominal lists, so every result is
reported at its nominal 64-scale depth (= relative depth x 64). Missing stages/models are skipped.
Writes results/tables/scale_ladder.md and $QLENS_OUT/scale_ladder.json.

Usage (CPU only: JSON only, configs from the HF cache):
  HF_HUB_OFFLINE=1 python src/compare_scale.py
  HF_HUB_OFFLINE=1 python src/compare_scale.py _ctx8   # the 8-context B2x/B4c reruns in <dir>_ctx8 (cited in the
                                                        # paper); writes scale_ladder_ctx8.{md,json}
"""
import json
import math
import os
import sys
from pathlib import Path

import transformers

W = Path(os.environ.get("QLENS_OUT", "outputs"))
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""  # run-dir suffix, e.g. _ctx8
RUNS = [  # (label, hf id, nominal params in B, run dir)
    ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", 0.8, "so_qwen3_5-0_8b"),
    ("Qwen3.5-2B", "Qwen/Qwen3.5-2B", 2, "so_qwen3_5-2b"),
    ("Qwen3.5-4B", "Qwen/Qwen3.5-4B", 4, "so_qwen3_5-4b"),
    ("Qwen3.5-9B", "Qwen/Qwen3.5-9B", 9, "so_qwen3_5-9b"),
    ("Qwen3.5-27B", "Qwen/Qwen3.5-27B", 27, "so_qwen3_5-27b"),
    ("Qwen3.8-27B", "Qwen/Qwen3.8-27B", 27, "second_order_v1"),
    ("Qwen3-8B*", "Qwen/Qwen3-8B", 8, "so_qwen3-8b"),
    ("Qwen3-14B*", "Qwen/Qwen3-14B", 14, "so_qwen3-14b"),
    ("Qwen3-32B*", "Qwen/Qwen3-32B", 32, "so_qwen3-32b"),
    ("Qwen3.5-122B-A10B†", "Qwen/Qwen3.5-122B-A10B", 122, "so_qwen3_5-122b-a10b"),
]


def load(d, stage):
    for p in (W / d / f"{stage}.json", W / f"{d}_{stage.lower()}" / f"{stage}.json"):  # 122B ran B4c separately
        if p.exists():
            return json.load(open(p))
    return None


def nominal(l, n):
    return int(round(int(l) * 64 / n))


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "·"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


out = {}
for label, hf_id, size, d in RUNS:
    d += SUFFIX
    if not (W / d).exists():
        continue
    try:
        cfg = transformers.AutoConfig.from_pretrained(hf_id)
        n = getattr(cfg, "text_config", cfg).num_hidden_layers
    except Exception:  # config not cached yet
        continue
    r = {"n_layers": n, "size_B": size, "dir": d}
    c = load(d, "check")
    if c:
        fz = c.get("fd_frozen_routing_vs_jvp") or {}
        r["check"] = {"layer_nom": nominal(c["layer"], n),
                      "fd_cos_eps0.01": min((fz or c["fd_vs_jvp"])["0.01"]["cos"]), "frozen": bool(fz),
                      "fd_cos_eps0.01_unfrozen": min(c["fd_vs_jvp"]["0.01"]["cos"]),
                      "tf32_cos": min(c["cos_tf32_vs_fp32"])}
    b1 = load(d, "B1")
    if b1:
        r["B1"] = {}
        alphas = [float(a) for a in b1["alphas"]]
        for l, v in b1["layers"].items():
            s = {float(a): x for a, x in v["summary"].items()}
            lin = {a: s[a]["median_actual_over_lin"] for a in alphas}
            quad = {a: s[a]["median_actual_over_quad"] for a in alphas}
            r["B1"][nominal(l, n)] = {
                "act/lin@.05": lin[0.05], "act/lin@.1": lin[0.1], "act/quad@.05": quad[0.05], "act/quad@.1": quad[0.1],
                "gain@.05": abs(math.log(abs(lin[0.05]) + 1e-9)) / max(abs(math.log(abs(quad[0.05]) + 1e-9)), 1e-3),
                "cos_lin@.05": s[0.05]["mean_cos_lin"],
                **{f"act/frozen@{a}": s[a]["median_actual_over_frozen"] for a in alphas if "median_actual_over_frozen" in s[a]},
                **{f"frozen/lin@{a}": s[a]["median_frozen_over_lin"] for a in alphas if "median_frozen_over_lin" in s[a]}}
    b2 = load(d, "B2x")
    if b2:
        r["B2x"] = {}
        for l, v in b2["layers"].items():
            s = v["summary"]
            r["B2x"][nominal(l, n)] = {
                "|Q|/|J(a+b)|": s["norm_ratio_Q_over_Jab"], "|I1|/|J(a+b)|": s["norm_ratio_I1.0_over_alpha_Jab"],
                "cos(I.25,Q)": s["cos_I0.25_vs_alpha2_Q"], "cos(I1,Q)": s["cos_I1.0_vs_alpha2_Q"],
                "Q_top10": s["Q"]["top10"], "Q_beats_singles": s["Q"]["frac_beats_both_singles"],
                "Qmis_top10": s["Q_mismatch"]["top10"], "Jab_top100": s["J_a+J_b"]["top100"]}
    b4f = load(d, "B4f")
    if b4f:
        r["B4f"] = {}
        rows = b4f["rows"]
        for task, v in b4f["summary"].items():
            for l, x in v["layers"].items():
                rr = [row["layers"][l] for row in rows if row["task"] == task and l in row["layers"]]
                s_by_lam = {lam: sum(y["lam"][lam]["S_patch"] for y in rr) / len(rr) for lam in rr[0]["lam"]} if rr else {}
                r["B4f"].setdefault(task, {})[nominal(l, n)] = {
                    "top1_patch": x["top1_actual_patch"], "top1_2nd": x["top1_second_order"],
                    "S_act": x["S_actual_mean"], "S_Q": x["S_Qst_mean"],
                    "capture": x["S_Qst_mean"] / x["S_actual_mean"] if x["top1_actual_patch"] >= 0.5 and abs(x["S_actual_mean"]) > 1 else float("nan"),
                    "cos.25": x["cos_I_vs_Qst_by_lam"]["0.25"], "corr.25": x["corr_S_lam0.25_Qst"],
                    "S/lam2": {k: round(vv, 3) for k, vv in s_by_lam.items()}}
    b4c = load(d, "B4c")
    if b4c:
        r["B4c"] = {nominal(l, n): {k: (v["summary"][k]["kappa_median"], v["summary"][k]["finite_median"])
                                   for k in ("unrelated", "related", "self", "random")}
                    for l, v in b4c["layers"].items()}
    out[label] = r

(W / f"scale_ladder{SUFFIX}.json").write_text(json.dumps(out, indent=1, default=str))

md = ["# Scale ladder — exact second-order stages across Qwen models", "",
      "Auto-generated by `src/compare_scale.py`; columns are nominal depth (layer × 64 / n_layers). "
      "`*` = Qwen3 dense, standard attention (architecture control; 8B/14B/32B = 36/40/64 layers at d 4096/5120/5120, so 14B vs 32B isolates depth at fixed width); `†` = MoE (256 experts, top-8; bf16 weights, "
      "fp32 compute, 2 GPUs); the rest are the dense qwen3_5 hybrid. "
      "Steering magnitudes are fractions α of the layer's typical residual norm ρ.", ""]


def table(title, depths, getter, nd=2, note=""):
    md.append(f"## {title}")
    if note:
        md.append(note)
    md.append("")
    md.append("| model | L | " + " | ".join(str(x) for x in depths) + " |")
    md.append("|---|---|" + "---|" * len(depths))
    for label, r in out.items():
        cells = [fmt(getter(r, dd), nd) for dd in depths]
        if all(c == "·" for c in cells):
            continue
        md.append(f"| {label} | {r['n_layers']} | " + " | ".join(cells) + " |")
    md.append("")


def g(stage, key, task=None):
    def f(r, dd):
        s = r.get(stage)
        if s and task:
            s = s.get(task)
        if not s:
            return None
        # nominal depths can be off by one after rounding: take the nearest available within 2
        near = [k for k in s if abs(int(k) - dd) <= 2]
        if not near:
            return None
        k = min(near, key=lambda k: abs(int(k) - dd))
        v = s[k]
        return v[key] if isinstance(key, str) else key(v)
    return f


md.append("Exactness check (nested jvp vs finite differences at eps 0.01ρ, min cos over 2 dirs): " + ", ".join(
    f"{k} {fmt(r['check']['fd_cos_eps0.01'], 3)}" + (f" (frozen routing; unfrozen {fmt(r['check']['fd_cos_eps0.01_unfrozen'], 2)})"
    if r['check']['frozen'] else "") for k, r in out.items() if "check" in r))
md.append("")
B1D, B2D, B4D, BCD = [16, 24, 32, 40, 48, 56], [32, 40, 48, 56], [4, 8, 16, 24, 32, 40], [24, 40, 56]
table("B1 — first-order miss: actual/linear at α=0.05 (1 = J exact)", B1D, g("B1", "act/lin@.05"))
table("B1 — second-order: actual/quadratic at α=0.05", B1D, g("B1", "act/quad@.05"))
table("B1 — error-reduction factor from adding Q at α=0.05 (|log act/lin| ÷ |log act/quad|; >1 = Q helps)",
      B1D, g("B1", "gain@.05"), 1)
for a in (0.05, 0.2, 0.8):
    table(f"B1 — MoE only: actual / frozen-routing response at α={a} (≠1 = expert switching)", B1D,
          g("B1", lambda v, a=a: v.get(f"act/frozen@{a}")))
table("B2x — curvature size ‖Q(a,b)‖/‖J(a+b)‖ at α=1", B2D, g("B2x", "|Q|/|J(a+b)|"))
table("B2x — actual interaction size ‖I_1‖/‖J(a+b)‖ at α=1", B2D, g("B2x", "|I1|/|J(a+b)|"))
table("B2x — cos(actual interaction at α=0.25, Q)", B2D, g("B2x", "cos(I.25,Q)"))
table("B2x — binding: conjunction c in top-10 of the Q readout (mismatched-pair control in B2x json)", B2D,
      g("B2x", "Q_top10"))
table("B2x — Q readout of c beats both single concepts", B2D, g("B2x", "Q_beats_singles"))
for task in ("arith_xy", "relation", "negation_xor"):
    table(f"B4f {task} — actual double patch gives c_st top-1", B4D, g("B4f", "top1_patch", task))
    table(f"B4f {task} — share of the full-swap interaction captured by Q (S_Q/S_act; only where the swap flips the answer ≥50%)", B4D, g("B4f", "capture", task), 3)
    table(f"B4f {task} — cos(I/λ² at λ=0.25, Q_st)", B4D, g("B4f", "cos.25", task))
for cat in ("unrelated", "related", "self", "random"):
    table(f"B4c κ median — {cat}", BCD, g("B4c", lambda v, c=cat: v[c][0]))
    table(f"B4c finite α=0.5 own-effect ratio − 1, median — {cat}", BCD, g("B4c", lambda v, c=cat: v[c][1]))
md.append("B4f S/λ² by λ (normalized interaction; constant = pure second order) and B4c finite-α ratios are in "
          f"`$QLENS_OUT/scale_ladder{SUFFIX}.json`.")
(Path(__file__).resolve().parents[1] / "results" / "tables" / f"scale_ladder{SUFFIX}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
