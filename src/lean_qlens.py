"""Lean Q-lens pilot (Step 5 of the paper). Every rule below was fixed before the runs (the paper's Lean protocol).

Stages (--stage, one model per call except `data`):
  data   CPU, tokenizer only, model-independent: Lean contexts (RH = the research library, ML = Mathlib 0df444a) by the
         tactic-step rule, FineWeb contexts (the paper's), edit candidates and E3 edit pairs
         -> <data>/contexts.pt (it holds RH token ids: the library has no licence, do not publish this file).
  check  the paper's finite-difference check of D1's Q at block 18 on one RH and one FW context -> <out>/check.json
  E1     the paper's D1 depth curve on RH / ML / FW (random + data directions) -> <out>/E1_<corpus>.{json,pt}
  E23    E2 (edit dose: J vs J+Q) and E3 (two-edit interaction vs Q(da,db)) on RH and ML -> <out>/E23_<corpus>.json

Push / readout conventions as so_core / second_order.py D1 (fp32, math SDPA, TF32 off). E2/E3 run the blocks after
the push block with q_robust.Tail (tested equal to the full model) and read the last position through the full final
norm. --tiny runs everything on a random 4-block Qwen3 (float64, CPU) with the real tokenizer, for the CPU test.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.func import jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
from so_core import Perturbed, first_and_second, math_sdpa, rademacher  # noqa: E402

W = Path(os.environ.get("QLENS_OUT", "outputs"))
SEED = 20261121
T = 128
BASE_TOK = "Qwen/Qwen3-8B-Base"
SUB_CLASSES = {  # string pairs; kept where both spellings have the same token count (changed tokens only differ)
    "rel": [("<", "≤"), (">", "≥"), ("=", "≠"), ("∈", "∉"), ("⊆", "⊂"), ("→", "↔")],
    "op": [("+", "*"), ("+", "-"), ("∧", "∨"), ("∪", "∩")],
    "type": [("ℕ", "ℝ"), ("ℤ", "ℚ"), ("Nat", "Int")],
    "const": [("0", "1"), ("1", "2"), ("True", "False")],
    "quant": [("∀", "∃")],
    "minmax": [("min", "max"), ("⊤", "⊥")],
    "fn": [("sin", "cos"), ("exp", "log")],
}
SWAP_OPS = ["<", "≤", ">", "≥", "=", "≠", "+", "*", "-", "∧", "∨"]
KEYWORDS = {"by", "at", "fun", "in", "do", "if", "then", "else", "with", "as", "let", "have", "show", "from", "use",
            "rw", "simp", "end", "def", "for", "to", "is", "of", "on", "or", "and", "not", "Type", "Prop", "Sort"}
IDENT = re.compile(r"^ [^\W\d][\w']{0,2}$")
BY_END = re.compile(r"(^|[^\w.'])by$")
BY_INLINE = re.compile(r"(?<![\w.'])by ([A-Za-z])")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ===================================================================== data: tactic steps and contexts
def tactic_starts(text: str) -> list[int]:
    """Char offsets of tactic steps (the Lean protocol's tactic-step rule)."""
    out, stack, pending, in_block, pos = [], [], None, 0, 0
    for ln in text.split("\n"):
        start, pos = pos, pos + len(ln) + 1
        # strip block comments (nesting-aware) and line comments -> code
        code, i = [], 0
        while i < len(ln):
            if ln.startswith("/-", i):
                in_block += 1; i += 2; continue
            if in_block and ln.startswith("-/", i):
                in_block -= 1; i += 2; continue
            if not in_block and ln.startswith("--", i):
                break
            code.append(ln[i] if not in_block else " "); i += 1
        code = "".join(code)
        s = code.strip()
        if not s:
            continue
        ind = len(code) - len(code.lstrip(" "))
        while stack and stack[-1] > ind:
            stack.pop()
        if pending is not None:  # the line after a trailing `by` opens a block at its column (Mathlib style puts the
            if ind > 0:          # proof at column 2 under a statement continued at column 4, so no "deeper" test)
                stack.append(ind)
            pending = None
        if stack and stack[-1] == ind:
            col = ind
            while code.startswith("· ", col):  # focus dot(s): tactic after the dot, nested block at col + 2
                col += 2
                stack.append(col)
            if col < len(code) and code[col].isascii() and code[col].isalpha():
                out.append(start + col)
        for mt in BY_INLINE.finditer(code):
            out.append(start + mt.start(1))
        if BY_END.search(code.rstrip()):
            pending = ind
    return sorted(set(out))


def build_subs(tok):
    """[(ids_a, ids_b, cls, str_a, str_b)] for both directions and spacing forms with equal token counts; the single-token
    swap operators; the dropped spellings."""
    enc = lambda x: tok(x, add_special_tokens=False)["input_ids"]  # noqa: E731
    subs, dropped = [], []
    for cls, pairs in SUB_CLASSES.items():
        for a, b in pairs:
            for pre in ("", " "):
                ia, ib = enc(pre + a), enc(pre + b)
                if len(ia) != len(ib):
                    dropped.append(f"{pre + a!r}/{pre + b!r}")
                    continue
                subs += [(ia, ib, cls, pre + a, pre + b), (ib, ia, cls, pre + b, pre + a)]
    ops = {x[0] for x in (enc(" " + o) for o in SWAP_OPS) if len(x) == 1}
    return subs, ops, dropped


def _check_edit(tok, text, offs, k, ctx, reps, changes):
    """Re-tokenize the file text up to the cut with the replacements reps [(lo, hi, old, new)] (window token ranges and
    their strings) applied; True if the last 128 ids equal ctx with `changes` [(p, new id)] (natural tokenization)."""
    base = k - T
    cut = offs[k][0]
    lead = max(0, offs[base][0] - 2000)
    pieces, last = [], lead
    for lo, hi, old, new in sorted(reps):
        s_, e_ = offs[base + lo][0], max(o[1] for o in offs[base + lo: base + hi])
        if text[s_:e_] != old or s_ < last:
            return False
        pieces += [text[last:s_], new]; last = e_
    pieces.append(text[last:cut])
    want = list(ctx)
    for p, nid in changes:
        want[p] = nid
    got = tok("".join(pieces), add_special_tokens=False)["input_ids"]
    return got[-T:] == want


def lean_contexts(corpus, repo, tok, n_need, subs, ops, rng):
    files = open(f"{repo}.sample_files.txt").read().split()
    rng.shuffle(files)
    out, stats = [], {"files_seen": 0, "no_tactic": 0, "too_short": 0, "retok_fail": 0}
    for path in files:
        if len(out) >= n_need:
            break
        stats["files_seen"] += 1
        text = open(Path(repo) / path, encoding="utf-8").read()
        starts = tactic_starts(text)
        if not starts:
            stats["no_tactic"] += 1; continue
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        tstart = [o[0] for o in offs]
        elig = []
        for c in starts:
            k = bisect.bisect_right(tstart, c) - 1
            if k >= T and offs[k][0] <= c < offs[k][1]:
                elig.append((c, k))
        if not elig:
            stats["too_short"] += 1; continue
        c, k = elig[rng.randrange(len(elig))]
        ctx = ids[k - T: k]
        if not _check_edit(tok, text, offs, k, ctx, [], []):
            stats["retok_fail"] += 1; continue
        cands = edit_candidates(tok, text, offs, k, ctx, subs, ops)
        rng.shuffle(cands)  # the seeded visiting order (E2 keeps the first 4 flipping edits per model)
        pairs = edit_pairs(tok, text, offs, k, ctx, cands, rng)
        out.append({"corpus": corpus, "file": path, "char": c, "tok_index": k, "ids": ctx, "target": ids[k],
                    "n_tactic_steps": len(starts), "n_eligible": len(elig), "cands": cands, "pairs": pairs,
                    "chars_per_token": (offs[k][0] - offs[k - T][0]) / T})
    stats["n"] = len(out)
    return out, stats


def edit_candidates(tok, text, offs, k, ctx, subs, ops):
    cands = []
    for ia, ib, cls, sa, sb in subs:
        n = len(ia)
        for p in range(16, T - 1 - n + 1):
            if ctx[p: p + n] == ia:
                if sa.strip().isdigit():  # numerals only: not a digit of an identifier (h1) or of a longer number
                    pv, nx = tok.decode([ctx[p - 1]]), tok.decode([ctx[p + n]])
                    if (pv and (pv[-1].isalnum() or pv[-1] in "_.'")) or (nx and (nx[0].isalnum() or nx[0] in "_'")):
                        continue
                ch = [(p + i, ib[i]) for i in range(n) if ia[i] != ib[i]]
                rp = [(p, p + n, sa, sb)]
                if _check_edit(tok, text, offs, k, ctx, rp, ch):
                    cands.append({"cls": cls, "changes": ch, "reps": rp, "what": f"{sa}->{sb}"})
    for p in range(17, T - 2):
        if ctx[p] in ops:
            x, y = ctx[p - 1], ctx[p + 1]
            sx, sy = tok.decode([x]), tok.decode([y])
            if x != y and IDENT.match(sx) and IDENT.match(sy) and sx[1:] not in KEYWORDS and sy[1:] not in KEYWORDS:
                ch = [(p - 1, y), (p + 1, x)]
                rp = [(p - 1, p, sx, sy), (p + 1, p + 2, sy, sx)]
                if _check_edit(tok, text, offs, k, ctx, rp, ch):
                    cands.append({"cls": "swap", "changes": ch, "reps": rp, "what": f"{sx}{tok.decode([ctx[p]])}{sy} swap"})
    return cands


def edit_pairs(tok, text, offs, k, ctx, cands, rng, n_pairs=3):
    allp = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            pi = [p for p, _ in cands[i]["changes"]]; pj = [p for p, _ in cands[j]["changes"]]
            si = {q for lo, hi, _, _ in cands[i]["reps"] for q in range(lo, hi)}
            sj = {q for lo, hi, _, _ in cands[j]["reps"] for q in range(lo, hi)}
            if min(abs(a - b) for a in pi for b in pj) >= 2 and not si & sj:
                allp.append((i, j))
    rng.shuffle(allp)
    keep = []
    for i, j in allp:
        if len(keep) >= n_pairs:
            break
        if _check_edit(tok, text, offs, k, ctx, cands[i]["reps"] + cands[j]["reps"],
                       cands[i]["changes"] + cands[j]["changes"]):
            keep.append((i, j))
    return keep


def stage_data(args):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(BASE_TOK)
    subs, ops, dropped = build_subs(tok)
    D = Path(args.data)
    res = {"seed": SEED, "tokenizer": BASE_TOK, "dropped_pairs": dropped, "n_subs": len(subs), "corpora": {}}
    for corpus, name in (("RH", "riemann-lean"), ("ML", "mathlib")):
        rng = random.Random(SEED)
        ctxs, st = lean_contexts(corpus, str(D / name), tok, args.n_lean, subs, ops, rng)
        res["corpora"][corpus] = {"commit": open(D / f"{name}.commit").read().strip(), "stats": st, "contexts": ctxs}
        log(corpus, st, "cands/ctx", sum(len(c["cands"]) for c in ctxs) / max(1, len(ctxs)),
            "pairs/ctx", sum(len(c["pairs"]) for c in ctxs) / max(1, len(ctxs)))
    docs = [json.loads(l)["text"] for l in open(W / "corpus/fineweb_edu_2000.jsonl")][1000:1064]
    fw = [tok(t, truncation=True, max_length=T)["input_ids"] for t in docs]
    fw = [x for x in fw if len(x) == T]
    res["corpora"]["FW"] = {"contexts": [{"corpus": "FW", "ids": x} for x in fw],
                            "chars_per_token": sum(len(tok.decode(x)) for x in fw) / (T * len(fw))}
    torch.save(res, D / "contexts.pt")
    log("dropped pairs (not single tokens):", dropped)
    log("wrote", D / "contexts.pt", {c: len(v["contexts"]) for c, v in res["corpora"].items()})


def split(corpus_entry, corpus, n_e23=96):
    """(E1 pushed, E1 data-direction contexts, E2/E3 contexts, extra E2-only contexts) as (index, context) lists.
    Lean: c0..c15 / c96..c103 / c0..c95 / c104.. (amendment 1); FW: the paper's first 16 / last 8."""
    cs = list(enumerate(corpus_entry["contexts"]))
    if corpus == "FW":
        return cs[:16], cs[-8:], [], []
    return cs[:16], cs[n_e23: n_e23 + 8], cs[:n_e23], cs[n_e23 + 8:]


# ===================================================================== model
def tiny_model(vocab=151936):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if p.dim() == 2 and "embed" not in n and "lm_head" not in n:
                p.mul_(8.0)  # properly nonlinear blocks
    import types

    def _rms64(self, x):  # keep float64 so finite differences are exact references
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    for mod in hf.modules():
        if type(mod).__name__.endswith("RMSNorm"):
            mod.forward = types.MethodType(_rms64, mod)
    return hf


def load(args):
    import transformers
    import jlens
    if args.tiny:
        hf = tiny_model()
        tok = transformers.AutoTokenizer.from_pretrained(BASE_TOK)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        tok = transformers.AutoTokenizer.from_pretrained(args.model)
        base = transformers.AutoTokenizer.from_pretrained(BASE_TOK)
        assert tok.get_vocab() == base.get_vocab(), f"{args.model} tokenizer differs from {BASE_TOK}: contexts invalid"
        hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(args, "load_dtype", torch.float32),
                                                               attn_implementation="sdpa", device_map="cuda")
    hf.eval()
    m = jlens.from_hf(hf, tok)
    return hf, m, tok


def dev_of(m):
    return m._lm_head.weight.device, m._lm_head.weight.dtype


def ids_t(m, x):
    dev, _ = dev_of(m)
    return torch.tensor([x["ids"] if isinstance(x, dict) else x], device=dev)


def capture_last(m, ids, layers):
    """Block outputs {l: [B, T, d]} and last-position logits [B, V] of one batch of ids (no grad)."""
    box = {}

    def mk(l_):
        def hook(_m, _i, o):
            box[l_] = (o if torch.is_tensor(o) else o[0]).detach()
        return hook
    hs = [m.layers[l_].register_forward_hook(mk(l_)) for l_ in layers]
    try:
        with torch.no_grad(), math_sdpa():
            out = m.forward(ids)
            z = m._lm_head(out.last_hidden_state[:, -1])
    finally:
        for h in hs:
            h.remove()
    return box, z


def gamma_of(m):
    dev, dt = dev_of(m)
    with torch.no_grad():
        return m._final_norm(torch.ones(1, m.d_model, device=dev, dtype=dt))[0]


# ===================================================================== check (the paper's FD gate for D1's Q)
def stage_check(args, m, C):
    from jlens.hooks import ActivationRecorder  # noqa: F401
    dev, dt = dev_of(m)
    L = m.n_layers - 1
    layer = max(1, min(L - 1, round(32 * m.n_layers / 64)))
    res = {"layer": layer}
    for corpus in ("RH", "FW"):
        cs = C["corpora"][corpus]["contexts"]
        rho = residual_stats(m, cs[:8], [layer])[layer]["rho"]
        ids = ids_t(m, cs[0])
        g = torch.Generator().manual_seed(7)
        a = torch.randn(2, m.d_model, generator=g, dtype=torch.float64).to(dev, dt); a = a / a.norm(dim=-1, keepdim=True) * rho
        b = torch.randn(2, m.d_model, generator=g, dtype=torch.float64).to(dev, dt); b = b / b.norm(dim=-1, keepdim=True) * rho
        f = Perturbed(m, ids.expand(2, -1), layer, L)
        e = rademacher(2, len(f.valid), dev, torch.Generator().manual_seed(0)).to(dt)
        A, Bv = f.spread(a, e), f.spread(b, e)
        t0 = time.perf_counter()
        with math_sdpa():
            Q = jvp(lambda d: jvp(f, (d,), (A,))[1], (torch.zeros_like(A),), (Bv,))[1]
        tq = time.perf_counter() - t0
        with torch.no_grad(), math_sdpa():
            F0 = f(torch.zeros_like(A))
            fd = {}
            for eps in (0.3, 0.1, 0.03, 0.01):
                Dd = (f(eps * (A + Bv)) - f(eps * A) - f(eps * Bv) + F0) / eps ** 2
                fd[eps] = {"cos": torch.nn.functional.cosine_similarity(Dd, Q, dim=-1).tolist(),
                           "rel_err": ((Dd - Q).norm(dim=-1) / Q.norm(dim=-1)).tolist()}
        best = max(min(v["cos"]) for v in fd.values())
        res[corpus] = {"rho": rho, "fd_vs_jvp": fd, "best_min_cos": best, "pass": best > 0.99, "secs_Q_batch2": tq}
        log(f"check {corpus}: best min-cos {best:.5f} {'PASS' if best > 0.99 else 'FAIL'}  Q batch2 {tq:.2f}s")
    (Path(args.out) / "check.json").write_text(json.dumps(res, indent=1))


# ===================================================================== E1 (the paper's D1)
class MultiTarget(Perturbed):
    """Perturbed with several target blocks in one forward: delta -> [B, n_targets, d] (sum over valid positions)."""

    def __init__(self, m_, ids, layer, targets):
        super().__init__(m_, ids, layer, max(targets))
        self.targets = list(targets)

    def __call__(self, delta):
        box = {}

        def add_hook(_mod, _inp, o):
            return o + delta if torch.is_tensor(o) else (o[0] + delta,) + tuple(o[1:])

        def rec(t):
            def hook(_mod, _inp, o):
                box[t] = o if torch.is_tensor(o) else o[0]
            return hook
        hs = [self.m.layers[self.layer].register_forward_hook(add_hook)]
        hs += [self.m.layers[t].register_forward_hook(rec(t)) for t in self.targets]
        try:
            self.m.forward(self.ids)
        finally:
            for h in hs:
                h.remove()
        return torch.stack([box[t][:, self.valid].sum(1) for t in self.targets], 1)


@torch.no_grad()
def residual_stats(m, ctxs, layers):
    acc = {l: [] for l in layers}
    L = m.n_layers - 1
    for x in ctxs:
        ids = ids_t(m, x)
        box, _ = capture_last(m, ids, layers)
        v = Perturbed(m, ids, 0, L).valid
        for l in layers:
            acc[l].append(box[l][0, v].double())
    out = {}
    for l in layers:
        H = torch.cat(acc[l])
        out[l] = {"mu": H.mean(0), "rho": float(H.norm(dim=-1).mean())}
    return out


def stage_E1(args, m, C, corpus):
    dev, dt = dev_of(m)
    L = m.n_layers - 1
    sc = lambda x: min(L - 1, max(1, round(x * m.n_layers / 64)))  # noqa: E731  (the paper's rescaling)
    srcs = sorted({sc(x) for x in (8, 16, 24, 32, 40, 48, 56)})
    ks, alphas = [1, 2, 4, 8, 16, 24, 32, 48], [0.05, 0.1, 0.2, 0.4]
    push, held, _, _ = split(C["corpora"][corpus], corpus, args.n_e23)
    push, held = [x for _, x in push[:args.n_e1]], [x for _, x in held[:args.n_dir]]
    nC, nD = len(push), args.n_dir
    DST = residual_stats(m, push[:8], srcs)
    dgen = torch.Generator().manual_seed(4321)
    data_dirs = {l: [] for l in srcs}
    for x in held:
        ids = ids_t(m, x)
        box, _ = capture_last(m, ids, srcs)
        v = Perturbed(m, ids, 0, L).valid
        tp = int(v[torch.randint(len(v), (1,), generator=dgen)])
        for l in srcs:
            data_dirs[l].append(box[l][0, tp].double() - DST[l]["mu"])
    raw, summ = {}, {}
    t0 = time.perf_counter()
    for l in srcs:
        rho = DST[l]["rho"]
        targets = sorted({min(l + k, L) for k in ks} | {L})
        types = {"iso": torch.randn(nD, m.d_model, generator=torch.Generator().manual_seed(77 + l)).double(),
                 "data": torch.stack(data_dirs[l]).cpu()}
        R = {"targets": targets, "rho": rho}
        for tn, U in types.items():
            U = (U / U.norm(dim=-1, keepdim=True) * rho).to(dev, dt)
            met = {k_: torch.zeros(nC, nD, len(alphas), len(targets)) for k_ in ("rel_lin", "rel_quad")}
            nrm = {k_: torch.zeros(nC, nD, len(targets)) for k_ in ("J", "Q")}
            for ci, x in enumerate(push):
                ids = ids_t(m, x)
                for s0 in range(0, nD, args.batch):
                    aa = U[s0: s0 + args.batch]
                    B_ = len(aa); sl = slice(s0, s0 + B_)
                    f = MultiTarget(m, ids.expand(B_, -1), l, targets)
                    ja, _, q = first_and_second(f, aa, None, signs=False)
                    nV = len(f.valid)
                    ja, q = ja.double() * nV, q.double() * nV
                    nrm["J"][ci, sl] = ja.norm(dim=-1).float().cpu(); nrm["Q"][ci, sl] = q.norm(dim=-1).float().cpu()
                    A1 = f.spread(aa, None)
                    with torch.no_grad(), math_sdpa():
                        F0 = f(torch.zeros_like(A1)).double()
                        for ai, al in enumerate(alphas):
                            act = f(al * A1).double() - F0
                            lin, quad = al * ja, al * ja + 0.5 * al ** 2 * q
                            an = act.norm(dim=-1).clamp_min(1e-30)
                            met["rel_lin"][ci, sl, ai] = ((act - lin).norm(dim=-1) / an).float().cpu()
                            met["rel_quad"][ci, sl, ai] = ((act - quad).norm(dim=-1) / an).float().cpu()
            R[tn] = {**met, "norm_J": nrm["J"], "norm_Q": nrm["Q"]}
            summ.setdefault(tn, {})[l] = {t - l: round(float((nrm["Q"][:, :, ti] / nrm["J"][:, :, ti]).median()), 4)
                                          for ti, t in enumerate(targets)}
        raw[l] = R
        log(f"E1 {corpus} L{l} rho={rho:.1f} r(k):", json.dumps({tn: summ[tn][l] for tn in summ}),
            f"{time.perf_counter() - t0:.0f}s")
    meta = {"corpus": corpus, "n_layers": m.n_layers, "alphas": alphas, "ks": ks, "srcs": srcs, "n_ctx": nC,
            "n_dir": nD, "rho": {l: DST[l]["rho"] for l in srcs}, "r_median": summ,
            "secs": round(time.perf_counter() - t0, 1)}
    torch.save({"meta": meta, "raw": raw}, Path(args.out) / f"E1_{corpus}.pt")
    (Path(args.out) / f"E1_{corpus}.json").write_text(json.dumps(meta, indent=1))


# ===================================================================== E2 / E3
class Tail:
    """Blocks l+1..L, final norm and lm_head applied to a block-l output h [B, T, d] (copy of q_robust.Tail)."""

    def __init__(self, m, layer: int, T_: int):
        from transformers.masking_utils import create_causal_mask
        text = m._text_module
        w = m._lm_head.weight
        self.blocks = [m.layers[i] for i in range(layer + 1, m.n_layers)]
        types = list(getattr(text.config, "layer_types", None) or ["full_attention"] * m.n_layers)[layer + 1:]
        self.pos = torch.arange(T_, device=w.device)[None]
        dummy = torch.zeros(1, T_, m.d_model, device=w.device, dtype=w.dtype)
        self.pe = text.rotary_emb(dummy, self.pos)
        kw = dict(config=text.config, inputs_embeds=dummy, attention_mask=None, past_key_values=None,
                  position_ids=self.pos)
        masks = {"full_attention": create_causal_mask(**kw)}
        if "sliding_attention" in types:
            from transformers.masking_utils import create_sliding_window_causal_mask
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kw)
        self.masks = [masks[t] for t in types]
        self.norm, self.head = m._final_norm, m._lm_head

    def last(self, h):
        for blk, mask in zip(self.blocks, self.masks):
            o = blk(h, attention_mask=mask, position_embeddings=self.pe, position_ids=self.pos)
            h = o[0] if isinstance(o, tuple) else o
        return self.head(self.norm(h[:, -1]))  # [B, V]


def centre(z):
    return z - z.mean(-1, keepdim=True)


def apply_changes(ids, changes):
    x = list(ids)
    for p, nid in changes:
        x[p] = nid
    return x


def flip_J(g0, g1):
    return -g0 / g1 if g1 > 0 else math.inf


def flip_JQ(g0, g1, g2):
    """Smallest positive root of g0 + g1 x + g2 x^2 / 2 (summarize_s2.flip_JQ)."""
    if abs(g2) < 1e-9:
        return flip_J(g0, g1)
    disc = g1 * g1 - 2 * g2 * g0
    if disc < 0:
        return math.inf
    pos = [x for x in ((-g1 + s * math.sqrt(disc)) / g2 for s in (1, -1)) if x > 0]
    return min(pos) if pos else math.inf


@torch.no_grad()
def tail_batched(tail, H, bs=16):
    with math_sdpa():
        return torch.cat([tail.last(H[i: i + bs]) for i in range(0, len(H), bs)])


def dose(tail, h0, d, to, te, n_grid=41, n_bis=10):
    """Actual flip dose lam* of g(lam) = z[te] - z[to] under h0 + lam d: first crossing on the grid, bisected."""
    lam = torch.linspace(0, 1, n_grid, device=h0.device, dtype=h0.dtype)
    z = tail_batched(tail, h0 + lam[:, None, None] * d)
    g = (z[:, te] - z[:, to]).double().cpu()
    i = next((j for j in range(1, n_grid) if g[j] > 0), None)
    if i is None or g[0] >= 0:
        return math.inf, g.tolist()
    lo, hi = float(lam[i - 1]), float(lam[i])
    for _ in range(n_bis):
        mid = 0.5 * (lo + hi)
        zm = tail_batched(tail, h0 + mid * d)[0]
        if float(zm[te] - zm[to]) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), g.tolist()


def jets(tail, h0, U, Vt=None):
    """(z0, J U, Q(U, V)) of the last-position logits at h0 along U (and V; V = U by default), nested forward mode."""
    Vt = U if Vt is None else Vt
    G = lambda v: tail.last(h0 + v)  # noqa: E731
    zero = torch.zeros_like(U)
    with math_sdpa():
        (z0, J), (_, Q) = jvp(lambda v: jvp(G, (v,), (U,)), (zero,), (Vt,))
    return z0, J, Q


def matched_random(d, gen):
    """Random push with the per-position norms of d [1, T, D] (random direction at each position)."""
    r = torch.randn(d.shape, generator=gen, dtype=torch.float64).to(d.device, d.dtype)
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30) * d.norm(dim=-1, keepdim=True)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))


def rel(a, ref):
    return float((a.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30))


def stage_E23(args, m, C, corpus):
    dev, dt = dev_of(m)
    layers = [int(x) for x in args.layers.split(",")]
    _, _, ctxs, extra = split(C["corpora"][corpus], corpus, args.n_e23)
    ctxs = ctxs + extra[:args.n_e2_extra]  # E3 pairs only from the first n_e23 contexts; E2 uses all
    t0 = time.perf_counter()
    # ---------------- pass 1: original + edited runs -> E2 cases, E3 pairs, pushes per layer (CPU)
    H0, cases, pairs, PUSH = {}, [], [], {}
    acc = {"n_ctx": len(ctxs), "top1_is_target": 0, "n_cands": 0, "n_flip_cands": 0}
    for ci, x in ctxs:
        need = sorted(set(range(len(x["cands"]))))
        idl = [x["ids"]] + [apply_changes(x["ids"], x["cands"][j]["changes"]) for j in need]
        top, zl, states = [], [], {l: [] for l in layers}
        for s in range(0, len(idl), args.fwd_batch):
            box, z = capture_last(m, torch.tensor(idl[s: s + args.fwd_batch], device=dev), layers)
            zl.append(z.double().cpu())
            for l in layers:
                states[l].append(box[l].cpu())
        Z = torch.cat(zl); top = Z.argmax(-1).tolist()
        S = {l: torch.cat(states[l]) for l in layers}
        H0[ci] = {l: S[l][0:1] for l in layers}
        to = top[0]
        acc["top1_is_target"] += int(to == x["target"]); acc["n_cands"] += len(need)
        flips = [j for j in need if top[1 + j] != to]
        acc["n_flip_cands"] += len(flips)
        for j in flips[:args.max_e2]:
            te = top[1 + j]
            key = ("e2", ci, j)
            PUSH[key] = {l: S[l][1 + j: 2 + j] - S[l][0:1] for l in layers}
            cases.append({"ci": ci, "cand": j, "cls": x["cands"][j]["cls"], "to": to, "te": te,
                          "g_full": float(Z[1 + j, te] - Z[1 + j, to]), "g_clean": float(Z[0, te] - Z[0, to])})
        for i, j in (x["pairs"] if ci < args.n_e23 else []):
            for jj in (i, j):
                PUSH.setdefault(("e3", ci, jj), {l: S[l][1 + jj: 2 + jj] - S[l][0:1] for l in layers})
            pairs.append({"ci": ci, "a": i, "b": j, "cls": [x["cands"][i]["cls"], x["cands"][j]["cls"]]})
    log(f"E23 {corpus} pass1: {acc} cases={len(cases)} pairs={len(pairs)} {time.perf_counter() - t0:.0f}s")
    npairs = len(pairs)
    for k, pr in enumerate(pairs):  # fixed derangement for the mismatched control: pair k takes pair k + n/2
        k2 = (k + npairs // 2) % npairs
        for _ in range(npairs):
            if pairs[k2]["ci"] != pr["ci"]:
                break
            k2 = (k2 + 1) % npairs
        pr["mis"] = k2
    # ---------------- pass 2: per layer
    for l in layers:
        tail = Tail(m, l, T)
        tl = time.perf_counter()
        for n, c in enumerate(cases):
            h0 = H0[c["ci"]][l].to(dev); d = PUSH[("e2", c["ci"], c["cand"])][l].to(dev)
            to, te = c["to"], c["te"]
            ls, grid = dose(tail, h0, d, to, te)
            z0, J, Q = jets(tail, h0, d)
            g0, g1, g2 = (float(v[0, te] - v[0, to]) for v in (z0, J, Q))
            r = c.setdefault("L", {})
            r[l] = {"lam_star": ls, "g0": g0, "g1": g1, "g2": g2, "lam_J": flip_J(g0, g1), "lam_JQ": flip_JQ(g0, g1, g2),
                    "g_tail1": grid[-1], "r_edit": float(centre(Q).norm() / centre(J).norm().clamp_min(1e-30)),
                    "grid": [round(v, 5) for v in grid[::4]]}
            if n < args.n_fd:  # FD gate: central differences of g
                eps = [1e-2, 3e-3, 1e-3]
                lamv = torch.tensor([0.0] + [s * e for e in eps for s in (1, -1)], device=dev, dtype=dt)
                gz = tail_batched(tail, h0 + lamv[:, None, None] * d)
                gg = (gz[:, te] - gz[:, to]).double().cpu()
                fd1 = [float((gg[1 + 2 * i] - gg[2 + 2 * i]) / (2 * e)) for i, e in enumerate(eps)]
                fd2 = [float((gg[1 + 2 * i] - 2 * gg[0] + gg[2 + 2 * i]) / e ** 2) for i, e in enumerate(eps)]
                r[l]["fd"] = {"g1": fd1, "g2": fd2,
                              "rel_g1": min(abs(v - g1) / max(abs(g1), 1e-30) for v in fd1),
                              "rel_g2": min(abs(v - g2) / max(abs(g2), 1e-30) for v in fd2)}
        log(f"E2 {corpus} L{l}: {len(cases)} cases {time.perf_counter() - tl:.0f}s")
        tl = time.perf_counter()
        for n, pr in enumerate(pairs):
            ci = pr["ci"]
            h0 = H0[ci][l].to(dev)
            da = PUSH[("e3", ci, pr["a"])][l].to(dev); db = PUSH[("e3", ci, pr["b"])][l].to(dev)
            mp = pairs[pr["mis"]]
            dbm = PUSH[("e3", mp["ci"], mp["b"])][l].to(dev)
            g = torch.Generator().manual_seed(SEED + 1000 * n + l)
            ra, rb = matched_random(da, g), matched_random(db, g)
            scales = [1.0, 0.5, 0.1] + ([0.03, 0.01, 0.003] if n < args.n_fd else [])
            Hs = [h0] + [h0 + s * v for s in scales for v in (da, db, da + db)] + [h0 + ra, h0 + rb, h0 + ra + rb]
            F = centre(tail_batched(tail, torch.cat(Hs))).double()
            f0 = F[0]
            D2 = {s: F[1 + 3 * i + 2] - F[1 + 3 * i] - F[1 + 3 * i + 1] + f0 for i, s in enumerate(scales)}
            jr = 1 + 3 * len(scales)
            D2r = F[jr + 2] - F[jr] - F[jr + 1] + f0
            A = torch.cat([da, da, da, ra]); Bt = torch.cat([db, dbm, rb, rb])
            _, _, Q4 = jets(tail, h0.expand(4, -1, -1), A, Bt)
            Q4 = centre(Q4).double()
            Qab, Qmis, Qarb, Qrr = Q4[0], Q4[1], Q4[2], Q4[3]
            joint = F[1 + 2] - f0
            r = pr.setdefault("L", {})
            r[l] = {"cos": cos(D2[1.0], Qab), "rel": rel(Qab, D2[1.0]), "cos_mis": cos(D2[1.0], Qmis),
                    "cos_arb": cos(D2[1.0], Qarb), "share": float(D2[1.0].norm() / joint.norm().clamp_min(1e-30)),
                    "cos_s": {s: cos(D2[s], Qab) for s in scales}, "rel_s": {s: rel(s * s * Qab, D2[s]) for s in scales},
                    "rand_cos": cos(D2r, Qrr), "rand_rel": rel(Qrr, D2r),
                    "rand_share": float(D2r.norm() / (F[jr + 2] - f0).norm().clamp_min(1e-30)),
                    "norm_D2": float(D2[1.0].norm()), "norm_Q": float(Qab.norm()), "norm_joint": float(joint.norm()),
                    "norm_a": float((F[1] - f0).norm()), "norm_b": float((F[2] - f0).norm()),
                    "norm_push_a": float(da.norm()), "norm_push_b": float(db.norm())}
        log(f"E3 {corpus} L{l}: {len(pairs)} pairs {time.perf_counter() - tl:.0f}s")
        (Path(args.out) / f"E23_{corpus}.json").write_text(json.dumps(
            {"corpus": corpus, "layers": layers, "acc": acc, "cases": cases, "pairs": pairs,
             "secs": round(time.perf_counter() - t0, 1)}, default=str))
    log(f"E23 {corpus} done {time.perf_counter() - t0:.0f}s")


def stage_gate2(args, m, C, corpus):
    """Amendment 2: E2/E3 FD gates re-checked at larger steps (fp32 round-off swamps the pre-registered small steps).
    Reads E23_<corpus>.json, rebuilds the same pushes, writes gate2_<corpus>.json. Prints gate values only."""
    dev, dt = dev_of(m)
    E = json.loads((Path(args.out) / f"E23_{corpus}.json").read_text())
    layers = [int(l) for l in E["layers"]]
    _, _, ctxs, extra = split(C["corpora"][corpus], corpus, args.n_e23)
    X = dict(ctxs + extra[:args.n_e2_extra])
    need = {}
    for c in E["cases"]:
        need.setdefault(c["ci"], set()).add(c["cand"])
    for pr in E["pairs"]:
        need.setdefault(pr["ci"], set()).update((pr["a"], pr["b"]))
    t0 = time.perf_counter()
    H0, PUSH = {}, {}
    for ci, js in need.items():
        js = sorted(js)
        idl = [X[ci]["ids"]] + [apply_changes(X[ci]["ids"], X[ci]["cands"][j]["changes"]) for j in js]
        states = {l: [] for l in layers}
        for s0 in range(0, len(idl), args.fwd_batch):
            box, _ = capture_last(m, torch.tensor(idl[s0: s0 + args.fwd_batch], device=dev), layers)
            for l in layers:
                states[l].append(box[l].cpu())
        S = {l: torch.cat(states[l]) for l in layers}
        H0[ci] = {l: S[l][0:1] for l in layers}
        for k, j in enumerate(js):
            PUSH[(ci, j)] = {l: S[l][1 + k: 2 + k] - S[l][0:1] for l in layers}
    log(f"gate2 {corpus}: rebuilt {len(PUSH)} pushes for {len(E['cases'])} cases, {len(E['pairs'])} pairs "
        f"{time.perf_counter() - t0:.0f}s")
    eps = [0.2, 0.1, 0.05]
    sc = [0.3, 0.2, 0.1, 0.05]
    out = {"corpus": corpus, "layers": layers, "rule": "Lean protocol, amendment 2", "E2": {}, "E3": {}}
    for l in layers:
        tail = Tail(m, l, T)
        tl = time.perf_counter()
        e2 = []
        lamv = torch.tensor([0.0] + [s_ * e for e in eps for s_ in (1, -1)], device=dev, dtype=dt)
        for c in E["cases"]:
            st = c["L"][str(l)]
            h0 = H0[c["ci"]][l].to(dev); d = PUSH[(c["ci"], c["cand"])][l].to(dev)
            gz = tail_batched(tail, h0 + lamv[:, None, None] * d)
            gg = (gz[:, c["te"]] - gz[:, c["to"]]).double().cpu().tolist()
            D1 = {e: (gg[1 + 2 * i] - gg[2 + 2 * i]) / (2 * e) for i, e in enumerate(eps)}
            D2 = {e: (gg[1 + 2 * i] - 2 * gg[0] + gg[2 + 2 * i]) / e ** 2 for i, e in enumerate(eps)}
            est1 = list(D1.values()) + [(4 * D1[.1] - D1[.2]) / 3, (4 * D1[.05] - D1[.1]) / 3]
            est2 = list(D2.values()) + [(4 * D2[.1] - D2[.2]) / 3, (4 * D2[.05] - D2[.1]) / 3]
            e2.append({"rel_g1": min(abs(v - st["g1"]) / max(abs(st["g1"]), 1e-30) for v in est1),
                       "rel_g2": min(abs(v - st["g2"]) / max(abs(st["g2"]), 1e-30) for v in est2),
                       "g0_diff": abs(gg[0] - st["g0"]) / max(abs(st["g0"]), 1e-12)})
        e3 = []
        for pr in E["pairs"]:
            st = pr["L"][str(l)]
            h0 = H0[pr["ci"]][l].to(dev)
            da = PUSH[(pr["ci"], pr["a"])][l].to(dev); db = PUSH[(pr["ci"], pr["b"])][l].to(dev)
            Hs = [h0] + [h0 + sg * s_ * v for s_ in sc for sg in (1, -1) for v in (da, db, da + db)] + [h0 + da, h0 + db,
                                                                                                         h0 + da + db]
            F = centre(tail_batched(tail, torch.cat(Hs))).double()
            dd = lambda k: F[1 + 3 * k + 2] - F[1 + 3 * k] - F[1 + 3 * k + 1] + F[0]  # noqa: E731
            M = {s_: (dd(2 * i) + dd(2 * i + 1)) / (2 * s_ * s_) for i, s_ in enumerate(sc)}
            ests = [M[s_] for s_ in sc] + [(4 * M[.1] - M[.2]) / 3, (4 * M[.05] - M[.1]) / 3]
            _, _, Qab = jets(tail, h0, da, db)
            Qab = centre(Qab).double()[0]
            cs = [cos(v, Qab) for v in ests]
            jr = 1 + 3 * 2 * len(sc)
            D21 = F[jr + 2] - F[jr] - F[jr + 1] + F[0]
            e3.append({"best_cos": max(cs), "cos": cs, "cos_diff": abs(cos(D21, Qab) - st["cos"])})
        med = lambda v: float(sorted(v)[len(v) // 2]) if v else float("nan")  # noqa: E731
        r1, r2 = med([x["rel_g1"] for x in e2]), med([x["rel_g2"] for x in e2])
        bc = med([x["best_cos"] for x in e3])
        out["E2"][str(l)] = {"n": len(e2), "med_rel_g1": r1, "med_rel_g2": r2, "pass": bool(e2) and r1 < .05 and r2 < .05,
                             "max_g0_diff": max([x["g0_diff"] for x in e2], default=None), "per_case": e2}
        out["E3"][str(l)] = {"n": len(e3), "med_best_cos": bc, "pass": bool(e3) and bc > .99,
                             "max_cos_diff": max([x["cos_diff"] for x in e3], default=None), "per_pair": e3}
        log(f"gate2 {corpus} L{l}: E2 n={len(e2)} rel_g1 {r1:.4f} rel_g2 {r2:.4f} -> {out['E2'][str(l)]['pass']} "
            f"(max g0 diff {out['E2'][str(l)]['max_g0_diff']}); E3 n={len(e3)} best cos {bc:.4f} -> "
            f"{out['E3'][str(l)]['pass']} (max cos diff {out['E3'][str(l)]['max_cos_diff']}) "
            f"{time.perf_counter() - tl:.0f}s")
    (Path(args.out) / f"gate2_{corpus}.json").write_text(json.dumps(out))


def rebuild_pushes(args, m, X, need, layers):
    """Original and edited runs for the given {context index: candidate ids}; block-l pushes on CPU."""
    dev, _ = dev_of(m)
    H0, PUSH = {}, {}
    for ci, js in need.items():
        js = sorted(js)
        idl = [X[ci]["ids"]] + [apply_changes(X[ci]["ids"], X[ci]["cands"][j]["changes"]) for j in js]
        states = {l: [] for l in layers}
        for s0 in range(0, len(idl), args.fwd_batch):
            box, _ = capture_last(m, torch.tensor(idl[s0: s0 + args.fwd_batch], device=dev), layers)
            for l in layers:
                states[l].append(box[l].cpu())
        S = {l: torch.cat(states[l]) for l in layers}
        H0[ci] = {l: S[l][0:1] for l in layers}
        for k, j in enumerate(js):
            PUSH[(ci, j)] = {l: S[l][1 + k: 2 + k] - S[l][0:1] for l in layers}
    return H0, PUSH


def stage_E4(args, m, C, corpus, tok):
    """E4 (Lean protocol): token readout of J d, 1/2 Q(d,d) and J + 1/2 Q for every E2 case and layer."""
    dev, dt = dev_of(m)
    E = json.loads((Path(args.out) / f"E23_{corpus}.json").read_text())
    layers = [int(l) for l in E["layers"]]
    _, _, ctxs, extra = split(C["corpora"][corpus], corpus, args.n_e23)
    X = dict(ctxs + extra[:args.n_e2_extra])
    need = {}
    for c in E["cases"]:
        need.setdefault(c["ci"], set()).add(c["cand"])
    t0 = time.perf_counter()
    H0, PUSH = rebuild_pushes(args, m, X, need, layers)
    nv = len(tok)
    dec = [tok.decode([i]) for i in range(nv)]
    loose = {}
    for i, sx in enumerate(dec):
        if sx.strip():
            loose.setdefault(sx.strip(), []).append(i)
    log(f"E4 {corpus}: rebuilt {len(PUSH)} pushes, vocab {nv} {time.perf_counter() - t0:.0f}s")

    def up(v, i):
        return int((v > v[i]).sum()) + 1

    def down(v, i):
        return int((v < v[i]).sum()) + 1

    def ids_of(i):
        return loose.get(dec[i].strip(), [i]) if dec[i].strip() else [i]

    def topl(v, k, sg):
        _, ix = (sg * v).topk(k)
        return [[int(i), dec[int(i)]] for i in ix]

    xs = torch.tensor([0.0, .1, -.1, .2, -.2], device=dev, dtype=dt)
    cases, fd = [], {}
    for c in E["cases"]:
        x = X[c["ci"]]; cand = x["cands"][c["cand"]]
        rec = {"ci": c["ci"], "cand": c["cand"], "cls": c["cls"], "what": cand["what"], "to": c["to"], "te": c["te"],
               "to_tok": dec[c["to"]], "te_tok": dec[c["te"]], "L": {}}
        if corpus == "ML":  # Mathlib is Apache-2.0 and quotable; RH (no licence): no file name or context text is recorded
            rec["file"] = x["file"]
            rec["ctx_tail"] = tok.decode(x["ids"])[-120:]
        cases.append(rec)
    for l in layers:
        tail = Tail(m, l, T)
        tl = time.perf_counter()
        rels = []
        for n, (c, rec) in enumerate(zip(E["cases"], cases)):
            h0 = H0[c["ci"]][l].to(dev); d = PUSH[(c["ci"], c["cand"])][l].to(dev)
            _, Jv, Qv = jets(tail, h0, d)
            Jv, Qv = Jv[0, :nv].double(), Qv[0, :nv].double()
            te, to = c["te"], c["to"]
            st = c["L"][str(l)]
            g_rel = max(abs(float(Jv[te] - Jv[to]) - st["g1"]) / max(abs(st["g1"]), 1e-12),
                        abs(float(Qv[te] - Qv[to]) - st["g2"]) / max(abs(st["g2"]), 1e-12))
            c1 = Jv - Jv.mean(); hq = 0.5 * (Qv - Qv.mean()); t2 = c1 + hq
            r = {"g_rel": g_rel, "cos_J_hQ": cos(c1, hq)}
            for nm, v in (("J", c1), ("hQ", hq), ("T2", t2)):
                r[nm] = {"new_up": up(v, te), "old_down": down(v, to),
                         "new_up_loose": min(up(v, i) for i in ids_of(te)),
                         "old_down_loose": min(down(v, i) for i in ids_of(to))}
            r["top"] = {"J_up": topl(c1, 10, 1), "J_down": topl(c1, 10, -1), "hQ_up": topl(hq, 10, 1),
                        "hQ_down": topl(hq, 10, -1)}
            if n < args.n_fd:  # full-vocabulary FD gate on 1/2 Q: central differences, Richardson (h = .2, .1)
                z = tail_batched(tail, h0 + xs[:, None, None] * d)[:, :nv].double()
                d2 = lambda h, i: (z[i] - 2 * z[0] + z[i + 1]) / h ** 2  # noqa: E731
                R = 0.5 * centre((4 * d2(.1, 1) - d2(.2, 3)) / 3)
                ov = len({int(i) for i in hq.topk(10).indices} & {int(i) for i in R.topk(10).indices}) / 10
                r["fd"] = {"rel_hQ": rel(hq, R), "top10_overlap": ov}
                rels.append((rel(hq, R), ov))
            rec["L"][str(l)] = r
        fr = sorted(x_[0] for x_ in rels)
        fd[str(l)] = {"n": len(rels), "med_rel": fr[len(fr) // 2] if fr else None,
                      "min_overlap": min((x_[1] for x_ in rels), default=None),
                      "max_g_rel": max(rec["L"][str(l)]["g_rel"] for rec in cases),
                      "pass": bool(fr) and fr[len(fr) // 2] < 0.1}
        log(f"E4 {corpus} L{l}: {len(cases)} cases, FD gate {fd[str(l)]} {time.perf_counter() - tl:.0f}s")
    (Path(args.out) / f"E4_{corpus}.json").write_text(json.dumps(
        {"corpus": corpus, "layers": layers, "nv": nv, "fd_gate": fd, "cases": cases}))


# ===================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="data | check | E1 | E23 | gate2 | E4 (comma list)")
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--out", default="")
    ap.add_argument("--data", default=str(W / "lean_v1/data"))
    ap.add_argument("--corpora-e1", default="RH,ML,FW")
    ap.add_argument("--corpora-e23", default="RH,ML")
    ap.add_argument("--n-lean", type=int, default=504, help="contexts built per Lean corpus (amendment 1: 104 -> 504)")
    ap.add_argument("--n-e1", type=int, default=16)
    ap.add_argument("--n-dir", type=int, default=8)
    ap.add_argument("--n-e23", type=int, default=96, help="contexts c0.. for E2 and E3 (c_n..c_n+7: E1 data dirs)")
    ap.add_argument("--n-e2-extra", type=int, default=400, help="E2-only contexts after the E1 data-dir ones")
    ap.add_argument("--max-e2", type=int, default=4)
    ap.add_argument("--n-fd", type=int, default=8)
    ap.add_argument("--layers", default="11,20,29")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fwd-batch", type=int, default=16)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)
    stages = args.stage.split(",")
    if "data" in stages:
        stage_data(args)
        stages.remove("data")
    if not stages:
        return
    Path(args.out).mkdir(parents=True, exist_ok=True)
    C = torch.load(Path(args.data) / "contexts.pt", weights_only=False)
    hf, m, tok = load(args)
    log(f"loaded {args.model if not args.tiny else 'tiny'} n_layers={m.n_layers} d={m.d_model} "
        f"dtype={dev_of(m)[1]} mem={torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0:.1f}GB")
    if "check" in stages:
        stage_check(args, m, C)
    if "E1" in stages:
        for corpus in args.corpora_e1.split(","):
            stage_E1(args, m, C, corpus)
    if "E23" in stages:
        for corpus in args.corpora_e23.split(","):
            stage_E23(args, m, C, corpus)
    if "gate2" in stages:
        for corpus in args.corpora_e23.split(","):
            stage_gate2(args, m, C, corpus)
    if "E4" in stages:
        for corpus in args.corpora_e23.split(","):
            stage_E4(args, m, C, corpus, tok)


if __name__ == "__main__":
    main()
