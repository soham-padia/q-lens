"""Q-robust pilot: are directions with large |Q|/|J| the directions along which behaviour is fragile?

Setting (D1 convention, src/second_order.py): source block l, push delta = s * alpha * rho_l * u (u unit, s = +-1)
added to the output of block l at every valid position (16..T-2); rho_l = mean residual norm over the valid
positions of the fit contexts. Readout map, per context c:

    G_c(u) = centred logits at every valid position of tail(h_l + push)    (tail = blocks l+1..L, final norm, lm_head)

centred = minus the mean over the vocabulary (a uniform shift changes no probability). With U = rho u:
    J_c u = d/dalpha G_c(alpha U)|_0,   Q_c(u,u) = d^2/dalpha^2 G_c(alpha U)|_0       (nested jvp, exact)
    ratio(u) = sqrt(sum_c |Q_c|^2 / sum_c |J_c u|^2)     (Frobenius over positions x vocab, alpha = 1 units)
so the second-order share of the logit change at push alpha is ~ alpha * ratio / 2. One map is used for the SVD,
the ascent, the scores and the behavioural KL (all valid positions, not only the last one).

Candidates per layer (classes):
  iso    random directions
  data   centred residuals h_l,t - mu_l at random valid positions of separate "pool" docs
  svd    top right-singular vectors of the stacked J_c over the fit contexts (subspace iteration, jvp + vjp,
         Rayleigh-Ritz)
  sharp  top eigenvectors of the Hessian of the next-token loss w.r.t. the push (exact HVP, forward-over-reverse;
         the "sharp" steering directions of Azizi et al. 2026, arXiv 2610.16084, for comparison)
  maxr   projected gradient ascent of log ratio on the sphere from random starts (gradient by reverse-over-forward)
  ctrl   for each of the n_high highest-ratio candidates: a direction with the same fit sum_c |J_c u|^2 and a low
         ratio (descent of log ratio + lam (log D - log D*)^2 from the closest low-ratio candidate)

Behaviour, on held-out contexts never used for scoring or fitting directions:
  (a) next-token KL(clean || pushed) and top-1 flip rate, mean over valid positions;
  (b) country/region -> capital facts (few-shot prompt, single-token answers the clean model gets right; push at
      positions 16..T-1, read at T-1): accuracy, KL and answer log-prob change;
  (c) greedy 20-token continuations (push at every position >= 16, prompt and generated): token edit distance from
      the clean continuation and NLL of the continuation under the unpushed model (fluency).
Held-out jets give the held-out ratio and, per context, the J-only and J+Q/2 Taylor predictions of KL and flips,
so the nonlinearity of the behaviour itself is measured against its first-order prediction.

Usage:  python src/q_robust.py --smoke                 (gates + timing projection; exit 1 if a gate fails)
        python src/q_robust.py --depths 0.3            (one source layer per job; resumable per stage)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp, vjp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
from so_core import Perturbed, first_and_second, math_sdpa, no_tf32  # noqa: E402

ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.4)
SIGNS = (1.0, -1.0)
BASE = ("iso", "data", "svd", "sharp", "maxr")

# ---------------------------------------------------------------- facts (neutral factual recall)
SHOTS = "The capital of Germany is Berlin.\nThe capital of Peru is Lima.\nThe capital of Kenya is Nairobi.\n"
COUNTRIES = [  # single-token capitals in the Qwen3 tokenizer (checked); variants separated by |
    ("Afghanistan", "Kabul"), ("Australia", "Canberra"), ("Austria", "Vienna"), ("Belgium", "Brussels"),
    ("Bulgaria", "Sofia"), ("Canada", "Ottawa"), ("Chile", "Santiago"), ("China", "Beijing"), ("Cuba", "Havana"),
    ("the Czech Republic", "Prague"), ("Denmark", "Copenhagen"), ("Egypt", "Cairo"), ("Finland", "Helsinki"),
    ("France", "Paris"), ("Greece", "Athens"), ("Guyana", "Georgetown"), ("Hungary", "Budapest"),
    ("Indonesia", "Jakarta"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Ireland", "Dublin"), ("Italy", "Rome"),
    ("Jamaica", "Kingston"), ("Japan", "Tokyo"), ("Lebanon", "Beirut"), ("Luxembourg", "Luxembourg"),
    ("Monaco", "Monaco"), ("the Netherlands", "Amsterdam"), ("New Zealand", "Wellington"),
    ("North Korea", "Pyongyang"), ("Norway", "Oslo"), ("Pakistan", "Islamabad"), ("the Philippines", "Manila"),
    ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Russia", "Moscow"), ("Saudi Arabia", "Riyadh"),
    ("Singapore", "Singapore"), ("South Korea", "Seoul"), ("Spain", "Madrid"), ("Sweden", "Stockholm"),
    ("Switzerland", "Bern"), ("Syria", "Damascus"), ("Taiwan", "Taipei"), ("Thailand", "Bangkok"),
    ("Tunisia", "Tunis"), ("Turkey", "Ankara"), ("Ukraine", "Kyiv|Kiev"), ("the United Kingdom", "London"),
    ("the United States", "Washington"),
]
REGIONS = [  # states / provinces (same relation, same template); ambiguous names (Georgia, Quebec) left out
    ("Alabama", "Montgomery"), ("Arizona", "Phoenix"), ("California", "Sacramento"), ("Colorado", "Denver"),
    ("Connecticut", "Hartford"), ("Delaware", "Dover"), ("Hawaii", "Honolulu"), ("Idaho", "Boise"),
    ("Illinois", "Springfield"), ("Indiana", "Indianapolis"), ("Maine", "Augusta"), ("Massachusetts", "Boston"),
    ("Michigan", "Lansing"), ("Mississippi", "Jackson"), ("Montana", "Helena"), ("Nebraska", "Lincoln"),
    ("New Hampshire", "Concord"), ("New York", "Albany"), ("North Carolina", "Raleigh"), ("Ohio", "Columbus"),
    ("Oregon", "Salem"), ("Rhode Island", "Providence"), ("South Carolina", "Columbia"), ("South Dakota", "Pierre"),
    ("Tennessee", "Nashville"), ("Texas", "Austin"), ("Virginia", "Richmond"), ("Washington State", "Olympia"),
    ("West Virginia", "Charleston"), ("Wisconsin", "Madison"), ("Ontario", "Toronto"),
    ("British Columbia", "Victoria"), ("Alberta", "Edmonton"), ("Manitoba", "Winnipeg"), ("Nova Scotia", "Halifax"),
    ("Saskatchewan", "Regina"), ("Queensland", "Brisbane"), ("the Australian state of Victoria", "Melbourne"),
    ("New South Wales", "Sydney"), ("Western Australia", "Perth"), ("South Australia", "Adelaide"),
    ("Bavaria", "Munich"), ("Catalonia", "Barcelona"), ("Scotland", "Edinburgh"), ("Wales", "Cardiff"),
    ("Northern Ireland", "Belfast"),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================== maps
class Tail:
    """Blocks l+1..L, final norm and lm_head of the HF model, applied to a block-l output h [B, T, d]."""

    def __init__(self, m, layer: int, T: int):
        from transformers.masking_utils import create_causal_mask
        text = m._text_module
        w = m._lm_head.weight
        self.blocks = [m.layers[i] for i in range(layer + 1, m.n_layers)]
        types = list(getattr(text.config, "layer_types", None) or ["full_attention"] * m.n_layers)[layer + 1:]
        self.pos = torch.arange(T, device=w.device)[None]
        dummy = torch.zeros(1, T, m.d_model, device=w.device, dtype=w.dtype)
        self.pe = text.rotary_emb(dummy, self.pos)
        kw = dict(config=text.config, inputs_embeds=dummy, attention_mask=None, past_key_values=None,
                  position_ids=self.pos)
        masks = {"full_attention": create_causal_mask(**kw)}
        if "sliding_attention" in types:
            from transformers.masking_utils import create_sliding_window_causal_mask
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kw)
        self.masks = [masks[t] for t in types]
        self.norm, self.head = m._final_norm, m._lm_head

    def states(self, h):
        for blk, mask in zip(self.blocks, self.masks):
            o = blk(h, attention_mask=mask, position_embeddings=self.pe, position_ids=self.pos)
            h = o[0] if isinstance(o, tuple) else o
        return h

    def logits(self, h, out_pos):
        return self.head(self.norm(self.states(h)[:, out_pos]))


class Map:
    """u [B, d] -> centred logits [B, P, V] of tail(h0 + push_mask * u) at out_pos (one context h0 [1, T, d])."""

    def __init__(self, tail: Tail, h0, push_pos, out_pos, labels=None):
        T = h0.shape[1]
        pm = torch.zeros(1, T, 1, device=h0.device, dtype=h0.dtype)
        pm[0, push_pos] = 1
        self.tail, self.h0, self.pm, self.out_pos, self.labels = tail, h0, pm, out_pos, labels

    def __call__(self, u):
        z = self.tail.logits(self.h0 + self.pm * u[:, None, :], self.out_pos)
        return z - z.mean(-1, keepdim=True)

    def resid(self, u):
        """D1's F: sum over out_pos of the final block output (cross-check against so_core.Perturbed)."""
        return self.tail.states(self.h0 + self.pm * u[:, None, :])[:, self.out_pos].sum(1)

    def loss(self, u):
        """Sum over the batch of the mean next-token NLL at out_pos (labels [P] = the context's next tokens)."""
        lp = torch.log_softmax(self(u), -1)
        return -lp.gather(-1, self.labels[None, :, None].expand(len(u), -1, 1)).mean((1, 2)).sum()


def jets(G, U):
    """(z0, J, Q), each [B, P, V], along U [B, d]: G(0), dG(tU)/dt, d^2G(tU)/dt^2 at t = 0 (nested forward mode)."""
    zero = torch.zeros_like(U)
    with math_sdpa():
        (z0, J), (_, Q) = jvp(lambda v: jvp(G, (v,), (U,)), (zero,), (U,))
    return z0, J, Q


def taylor_stats(z0, J, Q, alphas=ALPHAS, signs=SIGNS):
    """Per element: |J|^2, |Q|^2, <J,Q>, Fisher coefficient (KL ~ 0.5 alpha^2 fisher), and for each (alpha, sign)
    the KL (mean over positions) and top-1 flip rate of the J-only and the J+Q/2 Taylor logits."""
    B = J.shape[0]
    lp0 = torch.log_softmax(z0[:1], -1)
    p0 = lp0.exp()
    top0 = z0[:1].argmax(-1)
    mJ = (p0 * J).sum(-1)
    out = {"J2": J.pow(2).sum((1, 2)), "Q2": Q.pow(2).sum((1, 2)), "JQ": (J * Q).sum((1, 2)),
           "fisher": ((p0 * J * J).sum(-1) - mJ ** 2).mean(-1)}
    for k in ("kl_lin", "kl_quad", "fl_lin", "fl_quad"):
        out[k] = torch.zeros(B, len(alphas), len(signs), device=J.device)
    for ai, a in enumerate(alphas):
        for si, s in enumerate(signs):
            for name, z in (("lin", z0 + s * a * J), ("quad", z0 + s * a * J + 0.5 * a * a * Q)):
                lp = torch.log_softmax(z, -1)
                out["kl_" + name][:, ai, si] = (p0 * (lp0 - lp)).sum(-1).mean(-1)
                out["fl_" + name][:, ai, si] = (z.argmax(-1) != top0).float().mean(-1)
                del lp, z
    return {k: v.float().cpu() for k, v in out.items()}


def score(maps, U, rho, bs, taylor=False):
    """Jets of every direction (rows of unit U [N, d], push rho*u) on every map: dict of [N, C(, nA, nS)]."""
    N, C = len(U), len(maps)
    res = {}
    for ci, G in enumerate(maps):
        for s in range(0, N, bs):
            z0, J, Q = jets(G, rho * U[s: s + bs])
            if taylor:
                st = taylor_stats(z0, J, Q)
            else:
                st = {"J2": J.pow(2).sum((1, 2)).float().cpu(), "Q2": Q.pow(2).sum((1, 2)).float().cpu(),
                      "JQ": (J * Q).sum((1, 2)).float().cpu()}
            for k, v in st.items():
                if k not in res:
                    res[k] = torch.zeros((N, C) + tuple(v.shape[1:]))
                res[k][s: s + len(v), ci] = v
            del z0, J, Q
    return res


def ratio_of(sc, ctx=None):
    J2, Q2 = sc["J2"], sc["Q2"]
    if ctx is not None:
        J2, Q2 = J2[:, ctx], Q2[:, ctx]
    return (Q2.sum(1) / J2.sum(1).clamp_min(1e-30)).sqrt()


# ================================================================== direction finders
def subspace_top(apply, d, k, p, n_iter, gen, device, dtype):
    """Top-k eigenpairs of a symmetric operator apply(X [n, d]) -> [n, d] (rows) by subspace iteration with k + p
    vectors and Rayleigh-Ritz on the last iterate. Returns (vectors [k, d], values [k], history of values)."""
    X = torch.linalg.qr(torch.randn(d, k + p, generator=gen, dtype=torch.float64)).Q.T.to(device, dtype)
    hist = []
    for it in range(n_iter):
        Y = apply(X)
        H = X.double() @ Y.double().T
        ev, V = torch.linalg.eigh((H + H.T) / 2)
        ev, V = ev.flip(0), V.flip(1)
        hist.append([float(x) for x in ev])
        ritz = (V.T @ X.double()).to(dtype)
        if it < n_iter - 1:
            X = torch.linalg.qr(Y.double().T).Q.T.to(dtype)
    return ritz[:k], ev[:k].float().cpu(), hist


def JtJ_apply(maps, bs):
    def apply(X):
        Y = torch.zeros_like(X)
        for G in maps:
            for s in range(0, len(X), bs):
                xb = X[s: s + bs]
                zero = torch.zeros_like(xb)
                with math_sdpa():
                    _, Jx = jvp(G, (zero,), (xb,))
                    _, f_vjp = vjp(G, zero)
                    Y[s: s + bs] += f_vjp(Jx)[0]
                del Jx, f_vjp
        return Y
    return apply


def hvp_loss(G, X):
    """Hessian of the next-token loss w.r.t. the push (at 0) times X [B, d] (forward-over-reverse, exact)."""
    zero = torch.zeros_like(X)
    with math_sdpa():
        _, HX = jvp(grad(G.loss), (zero,), (X,))
    return HX


def H_apply(maps, bs):
    def apply(X):
        Y = torch.zeros_like(X)
        for G in maps:
            for s in range(0, len(X), bs):
                Y[s: s + bs] += hvp_loss(G, X[s: s + bs]) / len(maps)
        return Y
    return apply


def opt_eval(maps, u, rho, mode, tgt=None, lam=2.0):
    """Pass 1: psi(u) [B] and what the gradient pass needs. N = sum_c |Q_c(U,U)|^2, D = sum_c |J_c U|^2, U = rho u.
      mode 'maxr':    psi = -log N + log D                           (maximise the ratio^2 N / D)
      mode 'matched': psi = log N - log D + lam (log D - log D*)^2    (low ratio at a target first-order size)"""
    B = len(u)
    N = torch.zeros(B, dtype=torch.float64, device=u.device)
    D = torch.zeros_like(N)
    Qs = []
    for G in maps:
        _, J, Q = jets(G, rho * u)
        N += Q.pow(2).sum((1, 2)).double()
        D += J.pow(2).sum((1, 2)).double()
        Qs.append(Q)
        del J
    psi = -N.log() + D.log() if mode == "maxr" else N.log() - D.log() + lam * (D.log() - tgt) ** 2
    return psi, N, D, Qs


def opt_grad(maps, u, rho, mode, N, D, Qs, tgt=None, lam=2.0):
    """Pass 2: d psi / d u [B, d]. The vjp of v -> (G(v), J(v) U) with cotangents (c1, c2) is
    J^T c1 + (D^2G[U, .])^T c2, and grad_U D = 2 J^T J U, grad_U N = 4 (D^2G[U, .])^T Q."""
    if mode == "maxr":
        a1, a2 = 2 / D, -4 / N
    else:
        a1, a2 = 2 * (-1 + 2 * lam * (D.log() - tgt)) / D, 4 / N
    a1, a2 = a1.to(u.dtype)[:, None, None], a2.to(u.dtype)[:, None, None]
    U = rho * u
    g = torch.zeros_like(u)
    for G, Q in zip(maps, Qs):
        zero = torch.zeros_like(U)
        with math_sdpa():
            (_, J), f_vjp = vjp(lambda v: jvp(G, (v,), (U,)), zero)
            g += f_vjp((a1 * J, a2 * Q))[0]
        del J, f_vjp
    return rho * g


def sphere_opt(maps, u0, rho, steps, mode, logD_target=None, lam=2.0, eta0=0.5):
    """Minimise psi(u) (see opt_eval) over unit u by projected gradient steps with per-direction backtracking: keep
    the best iterate; grow the step 1.25x on success, halve it on failure. Returns (best u, history)."""
    B = len(u0)
    dev = u0.device
    u_best, u_try = u0.clone(), u0.clone()
    psi_best = torch.full((B,), float("inf"), dtype=torch.float64, device=dev)
    g_best = torch.zeros_like(u0)
    eta = torch.full((B,), eta0, dtype=u0.dtype, device=dev)
    tgt = None if logD_target is None else logD_target.to(dev, torch.float64)
    hist = []
    for step in range(steps):
        psi, N, D, Qs = opt_eval(maps, u_try, rho, mode, tgt, lam)
        acc = psi < psi_best
        u_best[acc], psi_best[acc] = u_try[acc], psi[acc]
        eta = torch.where(acc, eta * 1.25, eta * 0.5)
        hist.append({"ratio": (N / D).sqrt().float().cpu(), "D": D.float().cpu(), "psi": psi.float().cpu(),
                     "acc": acc.cpu()})
        if step == steps - 1:
            break
        if acc.any():
            g = opt_grad(maps, u_try, rho, mode, N, D, Qs, tgt, lam)
            g_best[acc] = g[acc]
        del Qs
        gt = g_best - (g_best * u_best).sum(-1, keepdim=True) * u_best
        u_try = u_best - eta[:, None] * gt / gt.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        u_try = u_try / u_try.norm(dim=-1, keepdim=True)
    return u_best, hist


# ================================================================== behaviour
@torch.no_grad()
def capture(m, ids, layers):
    """Clean block outputs {layer: [B, T, d]} for one batch of ids."""
    box = {}

    def mk(l_):
        def hook(_m, _i, o):
            box[l_] = (o if torch.is_tensor(o) else o[0]).detach()
        return hook
    hs = [m.layers[l_].register_forward_hook(mk(l_)) for l_ in layers]
    try:
        with math_sdpa():
            m.forward(ids)
    finally:
        for h in hs:
            h.remove()
    return box


@torch.no_grad()
def kl_flip(tail, H0, push_pos, out_pos, deltas, max_seqs):
    """For every (config k, context c): mean over out_pos of KL(clean || pushed) and of the top-1 flip, pushing
    deltas[k] at push_pos of clean block-l states H0 [C, T, d]. Returns (KL [K, C], flip [K, C]) on CPU."""
    C, T, _ = H0.shape
    pm = torch.zeros(1, 1, T, 1, device=H0.device, dtype=H0.dtype)
    pm[0, 0, push_pos] = 1
    with math_sdpa():
        z0 = tail.logits(H0, out_pos)
    lp0 = torch.log_softmax(z0, -1)
    p0, top0 = lp0.exp(), z0.argmax(-1)
    del z0
    K = len(deltas)
    KL, FL = torch.zeros(K, C), torch.zeros(K, C)
    kb = max(1, max_seqs // C)
    for s in range(0, K, kb):
        dk = deltas[s: s + kb]
        h = (H0[None] + pm * dk[:, None, None, :]).reshape(-1, T, H0.shape[-1])
        with math_sdpa():
            z = tail.logits(h, out_pos).view(len(dk), C, len(out_pos), -1)
        top = z.argmax(-1)
        lp = torch.log_softmax(z, -1)
        del z
        KL[s: s + len(dk)] = (p0[None] * (lp0[None] - lp)).sum(-1).mean(-1).cpu()
        FL[s: s + len(dk)] = (top != top0[None]).float().mean(-1).cpu()
        del lp, h
    return KL, FL


@torch.no_grad()
def facts_eval(tail, H0, ans, deltas, max_tokens):
    """Facts of one prompt length: H0 [F, T, d] clean block-l states, ans [F, A] accepted answer ids (-1 pad).
    Push deltas[k] at positions 16..T-1, read the last position. Returns correct [K, F] bool, KL [K, F],
    dlogp [K, F] (best accepted answer, pushed - clean)."""
    F_, T, _ = H0.shape
    pm = torch.zeros(1, 1, T, 1, device=H0.device, dtype=H0.dtype)
    pm[0, 0, 16:] = 1
    last = torch.tensor([T - 1], device=H0.device)
    okm = ans >= 0
    ansc = ans.clamp_min(0)

    def best_lp(lp):  # lp [..., F, V] -> [..., F]
        g = lp.gather(-1, ansc.expand(lp.shape[:-2] + ansc.shape))
        return g.masked_fill(~okm, -float("inf")).amax(-1)
    with math_sdpa():
        lp0 = torch.log_softmax(tail.logits(H0, last)[:, 0], -1)
    a0 = best_lp(lp0)
    K = len(deltas)
    COR, KL, DLP = torch.zeros(K, F_, dtype=torch.bool), torch.zeros(K, F_), torch.zeros(K, F_)
    kb = max(1, max_tokens // (F_ * T))
    for s in range(0, K, kb):
        dk = deltas[s: s + kb]
        h = (H0[None] + pm * dk[:, None, None, :]).reshape(-1, T, H0.shape[-1])
        with math_sdpa():
            lp = torch.log_softmax(tail.logits(h, last)[:, 0], -1).view(len(dk), F_, -1)
        top = lp.argmax(-1)
        COR[s: s + len(dk)] = ((top[..., None] == ans[None]) & okm[None]).any(-1).cpu()
        KL[s: s + len(dk)] = (lp0.exp()[None] * (lp0[None] - lp)).sum(-1).cpu()
        DLP[s: s + len(dk)] = (best_lp(lp) - a0[None]).cpu()
        del lp, h
    return COR, KL, DLP


class PushHook:
    """Adds deltas [B, d] to the output of one block at absolute positions >= push_from (KV-cache aware: set .off
    to the absolute position of the chunk's first token before each call)."""

    def __init__(self, block, deltas, push_from):
        self.d, self.p0, self.off = deltas, push_from, 0
        self.h = block.register_forward_hook(self)

    def __call__(self, _m, _i, o):
        h = o if torch.is_tensor(o) else o[0]
        pos = torch.arange(self.off, self.off + h.shape[1], device=h.device)
        h = h + (pos >= self.p0).to(h.dtype)[None, :, None] * self.d[:, None, :]
        return h if torch.is_tensor(o) else (h,) + tuple(o[1:])

    def remove(self):
        self.h.remove()


@torch.no_grad()
def greedy(hf, m, layer, ids, deltas, n_new, push_from=16):
    """Greedy continuation [B, n_new] of ids [B, T] with deltas [B, d] pushed at every position >= push_from."""
    hk = PushHook(m.layers[layer], deltas, push_from)
    try:
        with math_sdpa():
            out = hf(input_ids=ids, use_cache=True, logits_to_keep=1)
            hk.off = ids.shape[1]
            nxt = out.logits[:, -1].argmax(-1)
            toks, past = [nxt], out.past_key_values
            for _ in range(n_new - 1):
                out = hf(input_ids=nxt[:, None], past_key_values=past, use_cache=True, logits_to_keep=1)
                hk.off += 1
                nxt = out.logits[:, -1].argmax(-1)
                toks.append(nxt)
                past = out.past_key_values
    finally:
        hk.remove()
    return torch.stack(toks, 1)


@torch.no_grad()
def cont_nll(hf, ids, cont):
    """Mean NLL per token of cont [B, n] given ids [B, T] under the unpushed model."""
    with math_sdpa():
        lg = hf(input_ids=torch.cat([ids, cont], 1), use_cache=False, logits_to_keep=cont.shape[1] + 1).logits
    lp = torch.log_softmax(lg[:, :-1].float(), -1).gather(-1, cont[..., None])[..., 0]
    return -lp.mean(-1)


def edit_distance(a, b) -> int:
    """Levenshtein distance between two token sequences."""
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


# ================================================================== dose-response statistics (pure numpy)
def cross_dose(alphas, y, tau, log_y=False):
    """Smallest dose at which y reaches tau, interpolated linearly in log alpha (and in log y if log_y).
    Returns (dose, flag): flag 0 inside the grid, -1 already reached at the first dose, +1 never (dose = inf)."""
    a = np.log(np.asarray(alphas, dtype=float))
    y = np.asarray(y, dtype=float)
    for k in range(len(y)):
        if y[k] >= tau:
            if k == 0:
                return float(alphas[0]), -1
            y0, y1, t = y[k - 1], y[k], tau
            if log_y:
                y0, y1, t = math.log(max(y0, 1e-300)), math.log(y1), math.log(tau)
            f = (t - y0) / (y1 - y0) if y1 > y0 else 1.0
            return float(math.exp(a[k - 1] + f * (a[k] - a[k - 1]))), 0
    return float("inf"), 1


def loglog_slopes(alphas, y, floor=1e-12):
    la = np.log(np.asarray(alphas, dtype=float))
    ly = np.log(np.maximum(np.asarray(y, dtype=float), floor))
    return np.diff(ly) / np.diff(la)


def loglog_curvature(alphas, y, floor=1e-12):
    """Second derivative of log y in log alpha from a quadratic fit (0 for a power law, > 0 for an abrupt onset)."""
    la = np.log(np.asarray(alphas, dtype=float))
    ly = np.log(np.maximum(np.asarray(y, dtype=float), floor))
    return float(2 * np.polyfit(la, ly, 2)[0])


def steepness(alphas, y):
    """Largest increase of a bounded measure per e-fold of dose, and that increase as a share of the total."""
    d = np.diff(np.asarray(y, dtype=float)) / np.diff(np.log(np.asarray(alphas, dtype=float)))
    tot = float(y[-1] - y[0])
    return float(d.max()), float(d.max() / tot) if tot > 1e-12 else float("nan")


def _rank(x):
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    return rankdata(np.where(np.isinf(x), np.sign(x) * 1e300, x))


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    """Spearman of x and y with the ranks of z ([n] or [n, k] covariates) regressed out of both."""
    rx, ry = _rank(x), _rank(y)
    z = np.asarray(z, dtype=float)
    z = z[:, None] if z.ndim == 1 else z
    Z = np.column_stack([np.ones(len(rx))] + [_rank(z[:, j]) for j in range(z.shape[1])])
    res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]  # noqa: E731
    ex, ey = res(rx), res(ry)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(fn, arrays, n_boot=2000, seed=0):
    """Statistic fn(*arrays) and its 95% percentile bootstrap CI, resampling the first axis of every array."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    est = fn(*arrays)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = fn(*[a[i] for a in arrays])
        if np.isfinite(v):
            bs.append(v)
    if len(bs) < n_boot // 2:
        return est, (float("nan"), float("nan"))
    return est, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


# ================================================================== driver
def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "q_robust_qwen3-8b"))
    ap.add_argument("--corpus", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "corpus/fineweb_edu_2000.jsonl"))
    ap.add_argument("--ctx-start", type=int, default=1500, help="docs 1500+: disjoint from the J-lens fit docs "
                    "(first 96) and from the analysis contexts 1000-1063 used elsewhere")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--depths", default="0.3,0.55,0.8", help="source layers as fractions of n_layers")
    ap.add_argument("--layers", default="", help="explicit source layers (overrides --depths)")
    ap.add_argument("--n-fit", type=int, default=8)
    ap.add_argument("--n-held", type=int, default=32)
    ap.add_argument("--n-held-jets", type=int, default=32, help="held-out contexts that also get jets")
    ap.add_argument("--n-pool", type=int, default=16, help="docs the data directions come from")
    ap.add_argument("--n-iso", type=int, default=64)
    ap.add_argument("--n-data", type=int, default=64)
    ap.add_argument("--n-svd", type=int, default=32)
    ap.add_argument("--svd-over", type=int, default=8)
    ap.add_argument("--svd-iter", type=int, default=4)
    ap.add_argument("--n-sharp", type=int, default=16)
    ap.add_argument("--sharp-over", type=int, default=8)
    ap.add_argument("--sharp-iter", type=int, default=3)
    ap.add_argument("--n-maxr", type=int, default=32)
    ap.add_argument("--opt-steps", type=int, default=8)
    ap.add_argument("--n-high", type=int, default=16, help="highest-ratio candidates that get matched controls")
    ap.add_argument("--ctrl-steps", type=int, default=6)
    ap.add_argument("--jb", type=int, default=8, help="directions per jet batch")
    ap.add_argument("--beh-seqs", type=int, default=32, help="sequences per behavioural forward")
    ap.add_argument("--fact-tokens", type=int, default=8192)
    ap.add_argument("--n-cont", type=int, default=16)
    ap.add_argument("--cont-new", type=int, default=20)
    ap.add_argument("--cont-per-class", type=int, default=2, help="top and bottom fit-ratio dirs per class")
    ap.add_argument("--cont-high", type=int, default=8, help="highest-ratio dirs (and their controls) continued")
    ap.add_argument("--gen-seqs", type=int, default=64)
    ap.add_argument("--max-facts", type=int, default=100)
    ap.add_argument("--fresh", action="store_true", help="ignore saved stages")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.out = a.out.rstrip("/") + "_smoke"
        a.fresh = True
        a.n_fit, a.n_held, a.n_held_jets, a.n_pool = 2, 4, 4, 2
        a.n_iso, a.n_data, a.n_svd, a.svd_over, a.svd_iter = 4, 4, 4, 4, 2
        a.n_sharp, a.sharp_over, a.sharp_iter = 2, 2, 2
        a.n_maxr, a.opt_steps, a.n_high, a.ctrl_steps = 4, 3, 2, 2
        a.n_cont, a.cont_per_class, a.cont_high, a.max_facts = 2, 1, 1, 16
    return a


def full_counts(a_full):
    """Unit counts of the default (full) configuration, per layer, for the smoke timing projection."""
    base = a_full.n_iso + a_full.n_data + a_full.n_svd + a_full.n_sharp + a_full.n_maxr
    N = base + a_full.n_high
    n_cont_dirs = 2 * a_full.cont_per_class * len(BASE) + 2 * a_full.cont_high
    return {"score_fit": base * a_full.n_fit,
            "svd": (a_full.n_svd + a_full.svd_over) * a_full.n_fit * a_full.svd_iter,
            "sharp": (a_full.n_sharp + a_full.sharp_over) * a_full.n_fit * a_full.sharp_iter,
            "maxr": a_full.n_maxr * a_full.n_fit * a_full.opt_steps,
            "ctrl": a_full.n_high * a_full.n_fit * (a_full.ctrl_steps + 1),
            "held": N * a_full.n_held_jets,
            "kl": (N * len(ALPHAS) * len(SIGNS) + 1) * a_full.n_held,
            "facts": (N * len(ALPHAS) * len(SIGNS) + 1) * a_full.max_facts,
            "cont": (n_cont_dirs * len(ALPHAS) * len(SIGNS) + 1) * a_full.n_cont}


def run_layer(l, ctx, args, hf, m, facts, units, gates):
    od = Path(args.out) / f"L{l:02d}"
    od.mkdir(parents=True, exist_ok=True)
    dev, dt = m._lm_head.weight.device, m._lm_head.weight.dtype
    T = args.max_len
    from jlens.fitting import valid_position_mask
    valid = valid_position_mask(T).nonzero(as_tuple=True)[0].to(dev)

    def states(idl):
        return torch.cat([capture(m, torch.cat(idl[i: i + 8]), [l])[l] for i in range(0, len(idl), 8)])
    Hfit, Hheld, Hpool = states(ctx["fit"]), states(ctx["held"]), states(ctx["pool"])
    R = Hfit[:, valid].reshape(-1, m.d_model)
    mu, rho = R.mean(0), float(R.norm(dim=-1).mean())
    tail = Tail(m, l, T)
    lab = lambda ids: ids[0, valid + 1]  # noqa: E731  next tokens of the valid positions
    maps_fit = [Map(tail, Hfit[i: i + 1], valid, valid, lab(ctx["fit"][i])) for i in range(len(Hfit))]
    maps_held = [Map(tail, Hheld[i: i + 1], valid, valid) for i in range(args.n_held_jets)]
    log(f"L{l}: rho={rho:.2f}  fit={len(Hfit)} held={len(Hheld)} pool={len(Hpool)}")
    cuda = torch.cuda.is_available() and dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def stage(name, fn, unit_key=None, n_units=None):
        p = od / f"{name}.pt"
        if p.exists() and not args.fresh:
            log(f"L{l} {name}: resume {p}")
            return torch.load(p, weights_only=False)
        sync(); t0 = time.perf_counter()
        r = fn()
        sync(); r["secs"] = time.perf_counter() - t0
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        torch.save(r, p)
        if unit_key:
            units.setdefault(l, {})[unit_key] = r["secs"] / max(1, n_units)
        log(f"L{l} {name}: {r['secs']:.1f}s peak {r['peak_gb']:.1f} GB")
        return r

    # ---------------- gates (always on the first fit context; cheap)
    if args.smoke or not (od / "gates.json").exists() or args.fresh:
        gates[l] = layer_gates(m, l, tail, Hfit, ctx["fit"][0], valid, rho, mu, Hpool)
        (od / "gates.json").write_text(json.dumps(gates[l], indent=1))
        log(f"L{l} gates", json.dumps(gates[l]))

    # ---------------- A: directions
    def build_dirs():
        d = m.d_model
        g = torch.Generator().manual_seed(1000 + l)
        iso = torch.randn(args.n_iso, d, generator=g).to(dev, dt)
        g2 = torch.Generator().manual_seed(4000 + l)
        nV = len(valid)
        rows = []
        for j in range(args.n_data):
            c = j % len(Hpool)
            t = int(valid[torch.randint(nV, (1,), generator=g2)])
            rows.append(Hpool[c, t] - mu)
        data = torch.stack(rows)
        t0 = time.perf_counter()
        svd, svd_ev, svd_hist = subspace_top(JtJ_apply(maps_fit, args.jb), d, args.n_svd, args.svd_over,
                                             args.svd_iter, torch.Generator().manual_seed(3000 + l), dev, dt)
        t_svd = time.perf_counter() - t0
        t0 = time.perf_counter()
        sharp, sharp_ev, sharp_hist = subspace_top(H_apply(maps_fit, args.jb), d, args.n_sharp, args.sharp_over,
                                                   args.sharp_iter, torch.Generator().manual_seed(5000 + l), dev, dt)
        t_sharp = time.perf_counter() - t0
        t0 = time.perf_counter()
        st = torch.randn(args.n_maxr, d, generator=torch.Generator().manual_seed(2000 + l)).to(dev, dt)
        st = st / st.norm(dim=-1, keepdim=True)
        outs, hist = [], []
        for s in range(0, args.n_maxr, args.jb):
            u, h = sphere_opt(maps_fit, st[s: s + args.jb], rho, args.opt_steps, "maxr")
            outs.append(u); hist.append(h)
        t_maxr = time.perf_counter() - t0
        U = torch.cat([iso, data, svd, sharp, torch.cat(outs)])
        U = U / U.norm(dim=-1, keepdim=True)
        cls = ["iso"] * args.n_iso + ["data"] * args.n_data + ["svd"] * args.n_svd + ["sharp"] * args.n_sharp + \
            ["maxr"] * args.n_maxr
        units.setdefault(l, {}).update(svd=t_svd / ((args.n_svd + args.svd_over) * len(maps_fit) * args.svd_iter),
                                       sharp=t_sharp / ((args.n_sharp + args.sharp_over) * len(maps_fit) * args.sharp_iter),
                                       maxr=t_maxr / (args.n_maxr * len(maps_fit) * args.opt_steps))
        return {"U": U.cpu(), "cls": cls, "svd_ev": svd_ev, "svd_hist": svd_hist, "sharp_ev": sharp_ev,
                "sharp_hist": sharp_hist, "maxr_hist": hist, "maxr_start": st.cpu(), "rho": rho, "mu": mu.cpu(),
                "t_svd": t_svd, "t_sharp": t_sharp, "t_maxr": t_maxr}
    dirs = stage("dirs", build_dirs)
    U = dirs["U"].to(dev, dt)
    cls = list(dirs["cls"])
    fit = stage("fit", lambda: score(maps_fit, U, rho, args.jb), "score_fit", len(U) * len(maps_fit))

    # ---------------- C: matched controls (same fit D = sum_c |J_c u|^2, low ratio)
    def build_ctrl():
        r = ratio_of(fit)
        D = fit["J2"].sum(1)
        base = torch.arange(len(U))
        high = base[r.argsort()[-args.n_high:]]
        pool = [int(i) for i in base if int(i) not in set(high.tolist()) and r[i] <= r.median()]
        pool = torch.tensor(pool)
        starts = torch.stack([pool[(D[pool].log() - D[h].log()).abs().argmin()] for h in high])
        outs, hist = [], []
        for s in range(0, len(high), args.jb):
            hh, ss = high[s: s + args.jb], starts[s: s + args.jb]
            u, hst = sphere_opt(maps_fit, U[ss], rho, args.ctrl_steps, "matched", logD_target=D[hh].log())
            outs.append(u); hist.append(hst)
        Uc = torch.cat(outs)
        sc = score(maps_fit, Uc, rho, args.jb)
        return {"U": Uc.cpu(), "match_of": high, "start": starts, "hist": hist, "fit": sc}
    ctrl = stage("ctrl", build_ctrl, "ctrl", args.n_high * len(maps_fit) * (args.ctrl_steps + 1))
    U_all = torch.cat([U, ctrl["U"].to(dev, dt)])
    cls_all = cls + ["ctrl"] * len(ctrl["U"])
    fit_all = {k: torch.cat([fit[k], ctrl["fit"][k]]) for k in ("J2", "Q2", "JQ")}
    r_fit = ratio_of(fit_all)

    # ---------------- D: held-out jets (ratio transfer, Taylor predictions)
    held = stage("held", lambda: score(maps_held, U_all, rho, args.jb, taylor=True), "held",
                 len(U_all) * len(maps_held))

    # ---------------- E: KL and flips on held-out contexts
    sg = torch.tensor(SIGNS, device=dev, dtype=dt)
    al = torch.tensor(ALPHAS, device=dev, dtype=dt)

    def deltas_for(Ud):  # [n, nA, nS, d] -> flat configs, plus one zero config at the end
        Dl = rho * al[None, :, None, None] * sg[None, None, :, None] * Ud[:, None, None, :]
        return torch.cat([Dl.reshape(-1, Ud.shape[-1]), torch.zeros(1, Ud.shape[-1], device=dev, dtype=dt)])

    nA, nS = len(ALPHAS), len(SIGNS)

    def run_kl():
        dl = deltas_for(U_all)
        KL, FL = kl_flip(tail, Hheld, valid, valid, dl, args.beh_seqs)
        n = len(U_all)
        return {"kl": KL[:-1].view(n, nA, nS, -1), "flip": FL[:-1].view(n, nA, nS, -1), "kl_zero": KL[-1],
                "flip_zero": FL[-1]}
    beh = stage("kl", run_kl, "kl", (len(U_all) * nA * nS + 1) * len(Hheld))

    # ---------------- F: facts
    def run_facts():
        dl = deltas_for(U_all)
        n = len(U_all)
        out = {"items": [], "cor": [], "kl": [], "dlogp": []}
        for grp in facts:
            Hf = torch.cat([capture(m, grp["ids"][i: i + 32], [l])[l] for i in range(0, len(grp["ids"]), 32)])
            ft = Tail(m, l, grp["ids"].shape[1])
            C_, K_, D_ = facts_eval(ft, Hf, grp["ans"], dl, args.fact_tokens)
            out["items"] += grp["items"]
            out["cor"].append(C_); out["kl"].append(K_); out["dlogp"].append(D_)
        res = {"items": out["items"]}
        for k in ("cor", "kl", "dlogp"):
            X = torch.cat(out[k], 1)
            res[k], res[k + "_zero"] = X[:-1].view(n, nA, nS, -1), X[-1]
        return res
    nfacts = sum(len(g["items"]) for g in facts)
    fct = stage("facts", run_facts, "facts", (len(U_all) * nA * nS + 1) * max(1, nfacts))

    # ---------------- G: continuations on a subset of directions
    def pick_subset():
        sel = []
        for c in BASE:
            idx = [i for i, x in enumerate(cls_all) if x == c]
            o = sorted(idx, key=lambda i: float(r_fit[i]))
            k = min(args.cont_per_class, len(o) // 2)
            sel += o[:k] + o[-k:]
        high = ctrl["match_of"].tolist()[::-1][: args.cont_high]  # highest ratio first
        ci = [len(U) + int(j) for j, h in enumerate(ctrl["match_of"].tolist()) if h in high]
        sel += [int(h) for h in high] + ci
        return sorted(set(sel))

    def run_cont():
        sub = pick_subset()
        ids = torch.cat(ctx["held"][: args.n_cont])
        C = len(ids)
        clean = greedy(hf, m, l, ids, torch.zeros(C, m.d_model, device=dev, dtype=dt), args.cont_new)
        nll_clean = cont_nll(hf, ids, clean).cpu()
        dl = deltas_for(U_all[sub])
        dl = torch.cat([dl[-1:], dl[:-1]])  # zero config first: noise floor inside a mixed batch
        K = len(dl)
        toks = torch.zeros(K, C, args.cont_new, dtype=torch.long)
        nll = torch.zeros(K, C)
        kb = max(1, args.gen_seqs // C)
        for s in range(0, K, kb):
            dk = dl[s: s + kb]
            ii = ids.repeat(len(dk), 1)
            dd = dk.repeat_interleave(C, 0)
            tk = greedy(hf, m, l, ii, dd, args.cont_new)
            toks[s: s + len(dk)] = tk.view(len(dk), C, -1).cpu()
            nll[s: s + len(dk)] = cont_nll(hf, ii, tk).view(len(dk), C).cpu()
        ed = torch.tensor([[edit_distance(toks[k, c].tolist(), clean[c].tolist()) for c in range(C)]
                           for k in range(K)], dtype=torch.float32) / args.cont_new
        n = len(sub)
        return {"sub": sub, "clean": clean.cpu(), "nll_clean": nll_clean, "edit_zero": ed[0], "nll_zero": nll[0],
                "toks": toks[1:].view(n, nA, nS, C, -1), "edit": ed[1:].view(n, nA, nS, C),
                "nll": nll[1:].view(n, nA, nS, C)}
    n_sub = len(pick_subset())
    cont = stage("cont", run_cont, "cont", (n_sub * nA * nS + 1) * args.n_cont)

    # ---------------- behaviour-vs-Taylor gate (small alpha: the jets must describe the actual behaviour)
    kh = beh["kl"][:, 0, :, : args.n_held_jets].mean(-1)  # [N, nS] at the smallest alpha
    kq = held["kl_quad"][:, :, 0, :].mean(1)
    rel = ((kh - kq).abs() / kq.clamp_min(1e-30)).flatten()
    gates.setdefault(l, {})["taylor_kl_rel_median_a0"] = float(rel.median())
    gates[l]["kl_zero_max"] = float(beh["kl_zero"].abs().max())
    gates[l]["flip_zero_max"] = float(beh["flip_zero"].max())
    gates[l]["edit_zero_mean"] = float(cont["edit_zero"].mean())
    gates[l]["facts_zero_acc"] = float(fct["cor_zero"].float().mean()) if fct["cor_zero"].numel() else float("nan")
    gates[l]["maxr_ratio_gain_min"] = _maxr_gain(dirs["maxr_hist"])
    gates[l]["svd_top_vs_best_other"] = float(fit["J2"].sum(1)[[i for i, c in enumerate(cls) if c == "svd"]].max()
                                              / fit["J2"].sum(1)[[i for i, c in enumerate(cls) if c in ("iso", "data")]].max())
    (od / "gates.json").write_text(json.dumps(gates[l], indent=1))
    summ = quick_summary(l, rho, cls_all, fit_all, held, beh, fct, cont, ctrl)
    (od / "quick.json").write_text(json.dumps(summ, indent=1))
    log(f"L{l} quick", json.dumps(summ))


def _maxr_gain(hist_batches):
    """Smallest final/initial ratio over the max-ratio ascents (best iterate vs start)."""
    g = []
    for h in hist_batches:
        r0 = h[0]["ratio"]
        best = torch.stack([x["ratio"] for x in h]).amax(0)
        g.append(best / r0)
    return float(torch.cat(g).min())


def quick_summary(l, rho, cls, fit, held, beh, fct, cont, ctrl):
    r = ratio_of(fit).numpy()
    rh = ratio_of(held).numpy()
    kl = beh["kl"].mean(-1).mean(-1).numpy()  # [N, nA] sign- and context-mean
    out = {"layer": l, "rho": rho, "spearman_ratio_fit_vs_held": spearman(r, rh), "by_class": {}}
    for c in sorted(set(cls)):
        i = [k for k, x in enumerate(cls) if x == c]
        out["by_class"][c] = {"n": len(i), "ratio_fit_med": float(np.median(r[i])),
                              "ratio_held_med": float(np.median(rh[i])),
                              "J_fit_med": float(np.median(np.sqrt(fit["J2"][i].mean(1).numpy()))),
                              "kl_med_per_alpha": [float(x) for x in np.median(kl[i], 0)],
                              "flip_med_a0.4": float(beh["flip"][i, -1].mean((-1, -2)).median()),
                              "facts_acc_a0.4": float(fct["cor"][i, -1].float().mean()) if fct["cor"].numel() else None}
    out["spearman_ratio_fit_vs_kl_a0.1"] = spearman(r, kl[:, 2])
    out["spearman_J_fit_vs_kl_a0.1"] = spearman(fit["J2"].sum(1).numpy(), kl[:, 2])
    out["partial_ratio_vs_kl_a0.1_given_J"] = partial_spearman(r, kl[:, 2], fit["J2"].sum(1).numpy())
    return out


def layer_gates(m, l, tail, Hfit, ids0, valid, rho, mu, Hpool):
    """(1) tail == hooked full model; (2) tail jets == so_core.first_and_second on Perturbed (residual space);
    (3) logit jets vs central finite differences (+ Richardson), for one iso, one data direction."""
    g = {}
    dev, dt = Hfit.device, Hfit.dtype
    L = m.n_layers - 1
    box = capture(m, ids0, [L])
    with torch.no_grad(), math_sdpa():
        full = m._lm_head(m._final_norm(box[L][:, valid]))
        tl = tail.logits(Hfit[:1], valid)
    g["tail_vs_full_rel"] = float((tl - full).norm() / full.norm())
    gen = torch.Generator().manual_seed(99 + l)
    u = torch.stack([torch.randn(m.d_model, generator=gen).to(dev, dt), Hpool[0, int(valid[5])] - mu])
    u = u / u.norm(dim=-1, keepdim=True)
    U = rho * u
    f = Perturbed(m, ids0.expand(2, -1), l, L)
    ja, _, q = first_and_second(f, U, None, signs=False)
    nV = len(f.valid)
    G = Map(tail, Hfit[:1], valid, valid)
    zero = torch.zeros_like(U)
    with math_sdpa():
        (_, rJ), (_, rQ) = jvp(lambda v: jvp(G.resid, (v,), (U,)), (zero,), (U,))
    g["resid_J_rel"] = float(((rJ - ja * nV).norm(dim=-1) / rJ.norm(dim=-1)).max())
    g["resid_Q_rel"] = float(((rQ - q * nV).norm(dim=-1) / rQ.norm(dim=-1)).max())
    z0, J, Q = jets(G, U)
    fdJ, fdQ = {}, {}
    with torch.no_grad(), math_sdpa():
        G0 = G(zero)
        prevQ = prevJ = None
        for eps in (0.2, 0.1, 0.05, 0.025, 0.0125):
            Gp, Gm = G(eps * U), G(-eps * U)
            Jf, Qf = (Gp - Gm) / (2 * eps), (Gp - 2 * G0 + Gm) / eps ** 2
            rj = lambda A: [float(x) for x in ((A - J).flatten(1).norm(dim=-1) / J.flatten(1).norm(dim=-1))]  # noqa: E731
            rq = lambda A: [float(x) for x in ((A - Q).flatten(1).norm(dim=-1) / Q.flatten(1).norm(dim=-1))]  # noqa: E731
            fdJ[eps], fdQ[eps] = rj(Jf), rq(Qf)
            if prevQ is not None:  # Richardson: (4 f(eps) - f(2 eps)) / 3 kills the eps^2 term
                fdJ[f"R{eps}"], fdQ[f"R{eps}"] = rj((4 * Jf - prevJ) / 3), rq((4 * Qf - prevQ) / 3)
            prevQ, prevJ = Qf, Jf
            del Gp, Gm
    g["fd_J_rel"], g["fd_Q_rel"] = {str(k): v for k, v in fdJ.items()}, {str(k): v for k, v in fdQ.items()}
    g["fd_J_rel_min"] = max(min(v[i] for v in fdJ.values()) for i in range(2))
    g["fd_Q_rel_min"] = max(min(v[i] for v in fdQ.values()) for i in range(2))
    g["ratio_check_dirs"] = [float(x) for x in (Q.flatten(1).norm(dim=-1) / J.flatten(1).norm(dim=-1))]
    return g


GATE_LIMITS = {"tail_vs_full_rel": 1e-4, "resid_J_rel": 1e-3, "resid_Q_rel": 1e-2, "fd_J_rel_min": 1e-2,
               "fd_Q_rel_min": 5e-2, "kl_zero_max": 1e-6, "edit_zero_mean": 0.05, "taylor_kl_rel_median_a0": 0.1}


def check_gates(gates):
    fails = []
    for l, g in gates.items():
        for k, lim in GATE_LIMITS.items():
            v = g.get(k)
            if v is None or not (v <= lim):
                fails.append(f"L{l} {k}={v} > {lim}")
        if not g.get("maxr_ratio_gain_min", 0) > 1.0:
            fails.append(f"L{l} maxr ascent did not raise the ratio ({g.get('maxr_ratio_gain_min')})")
        if not g.get("svd_top_vs_best_other", 0) >= 1.0:
            fails.append(f"L{l} top svd direction has smaller |Ju| than an iso/data one")
        if not g.get("facts_zero_acc", 0) >= 0.98:
            fails.append(f"L{l} facts at alpha=0 not all correct ({g.get('facts_zero_acc')})")
    return fails


def build_facts(m, tok, max_facts, dev):
    """Few-shot capital prompts; keep single-token answers the clean model gets right; group by prompt length."""
    items = []
    for kind, lst in (("country", COUNTRIES), ("region", REGIONS)):
        for name, caps in lst:
            ids = [tok.encode(" " + c, add_special_tokens=False) for c in caps.split("|")]
            ids = [i[0] for i in ids if len(i) == 1]
            if ids:
                items.append({"kind": kind, "name": name, "answer": caps, "ans": ids,
                              "prompt": SHOTS + f"The capital of {name} is"})
    groups = {}
    for it in items:
        ids = m.encode(it["prompt"], max_length=256)
        groups.setdefault(ids.shape[1], []).append((it, ids))
    kept = []
    for T, lst in groups.items():
        ids = torch.cat([x[1] for x in lst])
        with torch.no_grad(), math_sdpa():
            top = m.unembed(capture(m, ids, [m.n_layers - 1])[m.n_layers - 1][:, -1]).argmax(-1).tolist()
        for (it, i1), t in zip(lst, top):
            it["clean_top"] = int(t)
            if t in it["ans"] and T >= 18:
                kept.append((it, i1))
    n_all, n_right = len(items), len(kept)
    kept = kept[:max_facts]
    out = {}
    for it, i1 in kept:
        out.setdefault(i1.shape[1], []).append((it, i1))
    facts = []
    for T, lst in sorted(out.items()):
        A = max(len(x[0]["ans"]) for x in lst)
        ans = torch.full((len(lst), A), -1, dtype=torch.long)
        for j, (it, _) in enumerate(lst):
            ans[j, : len(it["ans"])] = torch.tensor(it["ans"])
        facts.append({"ids": torch.cat([x[1] for x in lst]), "ans": ans.to(dev),
                      "items": [{k: v for k, v in x[0].items() if k != "prompt"} for x in lst]})
    return facts, n_all, n_right


def main(argv=None):
    args = parse(argv)
    import transformers
    import jlens
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                           attn_implementation="sdpa").cuda()
    m = jlens.from_hf(hf, tok)
    dev = m._lm_head.weight.device
    L = m.n_layers - 1
    layers = [int(x) for x in args.layers.split(",")] if args.layers else \
        sorted({min(L - 1, max(1, round(float(f) * m.n_layers))) for f in args.depths.split(",")})
    log(f"loaded {args.model}: {m.n_layers} blocks d={m.d_model} layers={layers} "
        f"({time.perf_counter() - t_start:.0f}s) gpu={torch.cuda.get_device_name()}")
    docs = [json.loads(line)["text"] for line in open(args.corpus)]
    pool = []
    for t in docs[args.ctx_start:]:
        ids = m.encode(t, max_length=args.max_len)
        if ids.shape[1] == args.max_len:
            pool.append(ids)
        if len(pool) == 8 + 32 + 16:
            break
    ctx = {"fit": pool[:args.n_fit], "held": pool[8: 8 + args.n_held], "pool": pool[40: 40 + args.n_pool]}
    facts, n_items, n_kept = build_facts(m, tok, args.max_facts, dev)
    log(f"contexts: fit {len(ctx['fit'])} held {len(ctx['held'])} pool {len(ctx['pool'])} (docs {args.ctx_start}+); "
        f"facts: clean model right on {n_kept} of {n_items} single-token items; using "
        f"{sum(len(g['items']) for g in facts)}, prompt lengths {[g['ids'].shape[1] for g in facts]}")
    meta = {"args": vars(args), "layers": layers, "alphas": ALPHAS, "signs": SIGNS, "n_fact_items": n_items,
            "n_facts_kept": n_kept, "gpu": torch.cuda.get_device_name()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    units, gates = {}, {}
    skipped = []
    with no_tf32():
        for l in layers:
            if args.smoke and time.perf_counter() - t_start > 400:  # stay inside the 10-min smoke allocation
                log(f"smoke: skipping layer {l} (time guard)")
                skipped.append(l)
                continue
            run_layer(l, ctx, args, hf, m, facts, units, gates)
    secs = time.perf_counter() - t_start
    log(f"done in {secs:.0f}s")
    if args.smoke:
        full = parse([])
        cnt = full_counts(full)
        proj = {l: {k: round(u.get(k, 0) * cnt[k] / 60, 1) for k in cnt} for l, u in units.items()}
        for l in skipped:  # scale the nearest measured layer by the number of blocks the tail runs
            near = min(units, key=lambda x: abs(x - l))
            f = (m.n_layers - l + 3) / (m.n_layers - near + 3)  # + ~3 block-equivalents for the lm_head
            proj[l] = {k: round(v * f, 1) for k, v in proj[near].items()}
        for l in proj:
            proj[l]["total_min"] = round(sum(v for k, v in proj[l].items() if k != "total_min"), 1)
        fails = check_gates(gates)
        if len(gates) < 2:
            fails.append(f"only {len(gates)} layer(s) ran in the smoke")
        rep = {"secs": secs, "projection_min": proj, "units_secs": units, "gates": gates, "fails": fails,
               "skipped": skipped,
               "n_facts_kept_full": n_kept}
        (out / "smoke_report.json").write_text(json.dumps(rep, indent=1, default=str))
        log("PROJECTION (min per layer, full run):", json.dumps(proj))
        if fails:
            log("SMOKE GATES FAILED:", *fails)
            sys.exit(1)
        log("SMOKE GATES PASSED")


if __name__ == "__main__":
    main()
