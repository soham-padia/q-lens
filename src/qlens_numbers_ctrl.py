#!/usr/bin/env python3
"""Specificity control for "adding Q(d,d)/2 helps numeric intermediates" (found 2026-11-22 after the space-token fix).

After the fix, J d + Q/2 beats J d on numeric intermediates:
  - order-ops: AUC 0.663 -> 0.770; 35 better, 8 worse;
  - multihop: 0.162 -> 0.325; 20 of 21 better.
It hurts operations (0.754 -> 0.563). At '=' the model is about to write digits, so Q might raise every digit, not
the specific intermediate. This job records, per layer and readout, the rank of:
  - each intermediate (same variant rule as B3x);
  - a matched control per intermediate:
      - a number of the same length whose first digit appears nowhere in the prompt, answer or intermediates;
      - an operation absent from the prompt;
      - for a multihop word, the next item's intermediate sharing no word;
  - all ten digit tokens and the answer.
It also saves the per-layer tensors (h, J d, Q, J mu) so any later token query is CPU work.

The machinery is imported from src/qlens_app.py: the model, mu, contexts, sweep_layer = B3x per layer.
The sets are order-ops and multihop, at SW_LAYERS (15 layers, 4..60).
Output: $QLENS_OUT/orderops_layers/ctrl_<set>.json and tensors_<set>.pt, written
after every layer.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import torch

sys.argv = sys.argv[:1]
import qlens_app as A  # noqa: E402  (loads the model, the fitted J-lens, mu and the contexts)

OUT = f"{A.W}/orderops_layers"
E = os.path.join(A.JLENS_DIR, "data", "evaluations")
OPS = A.OPSW


def first_ids(words):
    ids = set()
    for w_ in words:
        for v in {w_, w_.lower(), w_.capitalize()}:
            for p in ("", " "):
                e = [i for i in A.tok.encode(p + v, add_special_tokens=False) if A.tok.decode([i]).strip()]
                if e:
                    ids.add(e[0])
    return sorted(ids)


def variants(word, order_ops):  # = second_order.variants (fixed)
    if order_ops:
        return A.variants_oo(word)
    return first_ids({word})


def op_of(word):
    for k, syn in OPS.items():
        if word.lower() in syn or word.lower() == k:
            return k
    return None


def controls(items, order_ops, seed=0):
    """One control word per intermediate (None if no clean control exists)."""
    rng = random.Random(seed)
    out = []
    for i, it in enumerate(items):
        text = " ".join([it["prompt"], str(it["target"])] + list(it["intermediates"]))
        used_digits = {c for c in text if c.isdigit()}
        used_ops = {k for k, syn in OPS.items() if any(s in it["prompt"] for s in syn if not s.isalpha())}
        row = []
        for w in it["intermediates"]:
            if w[0].isdigit():
                free = [c for c in "0123456789" if c not in used_digits and not (len(w) > 1 and c == "0")]
                row.append(rng.choice(free) + w[1:] if free else None)
            elif order_ops and op_of(w):
                free = [k for k in OPS if k not in used_ops and k != op_of(w)]
                row.append(rng.choice(free) if free else None)
            else:  # multihop word: the next item's intermediate that shares no word with this item's
                mine = {x.lower() for x in it["intermediates"]}
                ctrl = None
                for off in range(1, len(items)):
                    cand = items[(i + off) % len(items)]["intermediates"]
                    if not mine & {x.lower() for x in cand}:
                        ctrl = cand[0]
                        break
                row.append(ctrl)
        out.append(row)
    return out


DIGIT_IDS = [A.tok.encode(str(c), add_special_tokens=False)[0] for c in range(10)]


def run(setname):
    order_ops = setname == "order-ops"
    items = json.load(open(f"{E}/lens-eval-{setname}.json"))["items"]
    ctrl = controls(items, order_ops)
    var_int = [[variants(w, order_ops) for w in it["intermediates"]] for it in items]
    var_ctl = [[variants(c, order_ops) if c else [] for c in row] for row in ctrl]
    var_tgt = [variants(str(it["target"]), order_ops) for it in items]
    layers = A.SW_LAYERS
    hs = []
    for it in items:
        ids = A.m.encode(it["prompt"], max_length=512)
        with torch.no_grad(), A.ActivationRecorder(A.m.layers, at=layers) as rec:
            A.m.forward(ids)
            hs.append(torch.stack([rec.activations[l][0, -1].float() for l in layers]))
    H = torch.stack(hs)  # [N, n_layers, d]
    T = {"layers": layers, "H": H.cpu(), "Jd": {}, "Q": {}, "Jmu": {}}
    res = {"model": A.MODEL, "set": setname, "layers_done": [], "arms": A.SW_ARMS,
           "items": [{"prompt": it["prompt"], "target": it["target"], "intermediates": it["intermediates"],
                      "controls": c} for it, c in zip(items, ctrl)],
           "digit_tokens": [A.tok.decode([i]) for i in DIGIT_IDS], "res": {},
           "method": "B3x per layer (src/qlens_app.sweep_layer): last position; d = h - mu_l; J d = mean over all "
                     "valid source positions; Q exact at 3 positions x 2 ctx (seed 1234+l); ranks 1-based, best "
                     "first-token variant (bare-space tokens skipped)"}
    t0 = time.time()
    for li, l in enumerate(layers):
        Hl = H[:, li].to(A.DEV)
        Jd, Q, Jmu = A.sweep_layer(l, Hl)
        outs = [Hl, Hl @ A.Jm(l).T, Jd + Jmu, Jd, Q, Jd + 0.5 * Q]
        R = {}
        with torch.no_grad():
            for k, y in zip(A.SW_ARMS, outs):
                lg = A.logits_of(y)
                R[k] = {"int": [[A.rank(lg[i], v) for v in var_int[i]] for i in range(len(items))],
                        "ctrl": [[A.rank(lg[i], v) if v else None for v in var_ctl[i]] for i in range(len(items))],
                        "target": [A.rank(lg[i], var_tgt[i]) for i in range(len(items))],
                        "digits": [[A.rank(lg[i], [d_]) for d_ in DIGIT_IDS] for i in range(len(items))]}
        res["res"][str(l)] = R
        res["layers_done"].append(l)
        T["Jd"][l], T["Q"][l], T["Jmu"][l] = Jd.cpu(), Q.cpu(), Jmu.cpu()
        with open(f"{OUT}/ctrl_{setname}.json", "w") as fh:
            json.dump(res, fh, ensure_ascii=False)
        torch.save(T, f"{OUT}/tensors_{setname}.pt")
        A.log(f"{setname} L{l} done ({li + 1}/{len(layers)}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in os.environ.get("CTRL_SETS", "order-ops,multihop").split(","):
        run(s)
    A.log("ALL DONE")
