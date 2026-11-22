"""Second-order ("double Jacobian") J-lens experiments on Qwen3.8-27B. See src/so_core.py for
definitions (F, J_l a, Q_l(a,b)). Stages (pick with --stages, results in <out>/<stage>.json):

  check  exactness: nested-jvp Q vs central finite differences in fp32; TF32 on/off agreement;
         timing per batch.
  vecs   concept vectors v_k = J_l^T (gamma*u_k) by reverse-mode VJP (sum over target positions,
         mean over valid sources: the repo's estimator for one token row), averaged over contexts.
  B1     curvature along lens directions ("steer at every position", unsigned): actual
         readout change for finite steps alpha*rho*v_k vs first-order and second-order Taylor.
         Answers the LRE-beta question: does first order underestimate, and is it fixed by Q?
  B2     binding / interaction lens (same-position, signed): does Q(v_a, v_b) read out the
         conjunction c of (a, b) (France x capital -> Paris)? vs first-order J(v_a+v_b), and vs
         mismatched-pair controls.
  D1     depth: nonlinearity of the map h_l -> h_{l+k} as the number of layers traversed k grows (one
         forward records every target). Steer at every position along lens / isotropic / data directions;
         actual change vs J and J+Q/2 in residual space and through W_U gamma. k is absolute, so models of
         different depth can be compared per layer traversed.
  B3     quadratic lens on the repo's eval prompts: readouts of J h (paper), J d, Q(d,d) and
         J d + Q(d,d)/2 with d = h - mean residual, scored by min-over-layers rank of the known
         intermediate (pass@k, normalized log-k AUC over k=1..1000).
  S1     one-hop broadcast lens-coordinate swap (France->China), single layers, Taylor-decomposed.
  S2     single layer vs many layers, for reading and steering: J-lens readout of latent two-hop intermediates
         per layer vs multi-layer windows (R); corrected band clamp on the S1 trials (O); two-hop "croissant" swaps
         at single layers vs band clamps vs a static spread, with controls and exact jets (T).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from so_core import Perturbed, first_and_second, math_sdpa, rademacher, rank_of, readout, no_tf32  # noqa: E402
import so_core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--revision", default=None, help="HF revision/branch (training checkpoints); default = main")
ap.add_argument("--corpus", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "corpus/fineweb_edu_2000.jsonl"))
ap.add_argument("--out", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "second_order_v1"))
ap.add_argument("--stages", default="check,vecs,B1,B2")
ap.add_argument("--layers", default="16,24,32,40,48,56")
ap.add_argument("--dtype", default="float32", help="compute dtype")
ap.add_argument("--weights-dtype", default="", help="storage dtype of the weights (default = --dtype); bfloat16 + --dtype float32 = fp32 compute on bf16-stored weights (so_core.enable_upcast)")
ap.add_argument("--tf32", type=int, default=1, help="allow TF32 matmuls in bulk stages (check measures the effect)")
ap.add_argument("--ctx-start", type=int, default=1000, help="context docs come from corpus[ctx-start:], disjoint from fit prompts")
ap.add_argument("--n-ctx", type=int, default=4, help="contexts averaged per Q estimate (B1/B2)")
ap.add_argument("--n-vec-ctx", type=int, default=8, help="contexts averaged for concept vectors")
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--b3-sets", default="multihop,multilingual,poetry")
ap.add_argument("--b3-layers", default="24,32,40,48,56")
ap.add_argument("--b3-n-ctx", type=int, default=2)
ap.add_argument("--b3-name", default="B3")
ap.add_argument("--b3-exact-pos", type=int, default=0, help=">0: exact same-position Q with this many sampled positions")
ap.add_argument("--b3-limit", type=int, default=0, help="max items per eval set (0 = all)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--b2x-layers", default="32,40,48,56")
ap.add_argument("--b2x-n-pos", type=int, default=4, help="source positions sampled per context (exact, no sign trick)")
ap.add_argument("--b2x-n-ctx", type=int, default=2)
ap.add_argument("--b4f-layers", default="4,8,16,24,32,40")
ap.add_argument("--b4c-layers", default="24,40,56")
ap.add_argument("--b4c-n-ctx", type=int, default=2)
ap.add_argument("--b4c-n-pos", type=int, default=3)
ap.add_argument("--n1-layers", default="24,16,40,32,48,56")
ap.add_argument("--n1-lens-dir", default=os.path.join(os.environ.get("QLENS_OUT", "outputs"), "fit_v1"), help="fitted mean J-lens shards (N1)")
ap.add_argument("--n1-n-ctx", type=int, default=4)
ap.add_argument("--n1-name", default="N1", help="output json name for N1")
ap.add_argument("--p1-layers", default="40,48,56")
ap.add_argument("--p1-batch", type=int, default=4, help="HVP rows per reverse-over-forward call (memory)")
ap.add_argument("--s1-layers", default="24,36,40,48,56")
ap.add_argument("--s1-band", default="24,59", help="first,last layer of the paper-style band-clamped swap")
ap.add_argument("--s1-limit", type=int, default=0, help="max trials (0 = all 192)")
ap.add_argument("--s2-layers", default="16,20,24,28,32,36,40,44,48,52,56", help="S2 single-layer swap grid")
ap.add_argument("--s2-bands", default="16-31,24-39,32-47,40-55,24-59", help="S2 multi-layer bands, first-last inclusive")
ap.add_argument("--s2-parts", default="R,O,T", help="S2 parts: R reading, O one-hop band clamp, T two-hop swaps")
ap.add_argument("--s2-limit", type=int, default=0, help="max two-hop trials / one-hop trials (0 = all)")
ap.add_argument("--s2-max-min", type=float, default=200, help="stop S2 loops after this many minutes (partial dump)")
ap.add_argument("--s2-shard", default="0/1", help="i/n: run every n-th two-hop and one-hop trial starting at i")
ap.add_argument("--s2-lams", default="", help="S2 T single-layer swap strengths, comma list; must contain 1 and 2 "
                "(empty = the default 0.25,0.5,0.75,1,1.5,2)")
ap.add_argument("--s2-band-lams", default="", help="S2 T band clamp strengths, comma list; must contain 1 "
                "(empty = the default 0.25,0.5,1,2)")
ap.add_argument("--d1-sources", default="8,16,24,32,40,48,56", help="D1 source layers (nominal, rescaled)")
ap.add_argument("--d1-ks", default="1,2,4,8,16,24,32,48", help="D1 layers traversed, ABSOLUTE (not rescaled); final block always added")
ap.add_argument("--d1-n-ctx", type=int, default=8)
ap.add_argument("--d1-n-dir", type=int, default=8, help="D1 directions per type (lens, iso, data)")
ap.add_argument("--d1-alphas", default="0.05,0.1,0.2,0.4")
args = ap.parse_args()
stages = args.stages.split(",")
LAYERS = [int(x) for x in args.layers.split(",")]
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed); random.seed(args.seed)
gen = torch.Generator().manual_seed(args.seed)
EVAL_DIR = Path(JLENS_DIR) / "data" / "evaluations"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dump(name, obj):
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    log(f"wrote {out / (name + '.json')}")


# ---------------------------------------------------------------- model
dtype = getattr(torch, args.dtype)
_rev = dict(revision=args.revision) if args.revision else {}
tok = transformers.AutoTokenizer.from_pretrained(args.model, **_rev)
_arch = (transformers.AutoConfig.from_pretrained(args.model, **_rev).architectures or [""])[0]
_cls = transformers.AutoModelForImageTextToText if "ConditionalGeneration" in _arch else transformers.AutoModelForCausalLM
_cfg = transformers.AutoConfig.from_pretrained(args.model, **_rev)
wdtype = getattr(torch, args.weights_dtype) if args.weights_dtype else dtype
N_GPU = torch.cuda.device_count()
_kw = dict(dtype=wdtype, attn_implementation="sdpa")
if getattr(getattr(_cfg, "text_config", _cfg), "num_experts", 0):
    _kw["experts_implementation"] = "eager"  # per-expert loop: forward-mode AD works (grouped_mm has no jvp)
_kw["device_map"] = so_core.layer_device_map(_cls, _cfg, N_GPU) if N_GPU > 1 else "cuda"
hf = _cls.from_pretrained(args.model, **_rev, **_kw)
if wdtype != dtype:
    so_core.enable_upcast(hf, dtype)
m = jlens.from_hf(hf, tok)
if N_GPU > 1:
    so_core.gather_layer_outputs(m.layers, "cuda:0")
N_ROUTERS = so_core.patch_routers(hf)  # MoE only: enables frozen-routing finite differences
L = m.n_layers - 1
W_U = hf.get_output_embeddings().weight.detach()
with torch.no_grad():  # RMSNorm gain, architecture-agnostic: norm(ones) = gain (Qwen3.5: 1+w, Llama: w)
    gamma = m._final_norm(torch.ones(1, m.d_model, device="cuda", dtype=dtype)).float()[0]
# Layer indices in the args are written for the 64-layer Qwen3.8-27B; rescale by relative depth.
_sc = lambda xs: ",".join(str(min(L - 1, max(1, round(int(x) * m.n_layers / 64)))) for x in xs.split(","))  # noqa: E731
LAYERS = [int(x) for x in _sc(args.layers).split(",")]
for _k in ("b2x_layers", "b4f_layers", "b4c_layers", "b3_layers", "n1_layers", "p1_layers", "s1_layers", "s1_band",
           "d1_sources", "s2_layers"):
    setattr(args, _k, _sc(getattr(args, _k)))
args.s2_bands = ",".join(f"{_sc(b.split('-')[0])}-{_sc(b.split('-')[1])}" for b in args.s2_bands.split(","))
CHECK_LAYER = int(_sc("32"))
# B2x/B4c/N1/P1 read residual stats and concept vectors, which exist only for LAYERS: fail now, not 40 min in
for _k in ("b2x_layers", "b4c_layers", "n1_layers", "p1_layers"):
    _bad = sorted(set(int(x) for x in getattr(args, _k).split(",")) - set(LAYERS))
    assert not _bad, f"--{_k.replace('_', '-')} {_bad} not in --layers {LAYERS}"
log(f"arch={_arch} n_layers={m.n_layers} d={m.d_model} LAYERS={LAYERS} b2x={args.b2x_layers} b4f={args.b4f_layers} "
    f"b4c={args.b4c_layers} gamma_mean={float(gamma.mean()):.3f}")
torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32); torch.backends.cudnn.allow_tf32 = bool(args.tf32)
log(f"loaded {args.model} compute={args.dtype} weights={wdtype} n_gpu={N_GPU} experts={_kw.get('experts_implementation')} "
    f"mem={[round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(N_GPU)]} GB  L={L}")

docs = [json.loads(l)["text"] for l in open(args.corpus)]
ctx_ids = [m.encode(t, max_length=args.max_len) for t in docs[args.ctx_start: args.ctx_start + 64]]
ctx_ids = [c for c in ctx_ids if c.shape[1] == args.max_len]


def tid(word: str) -> tuple[int, bool]:
    """Token id for a word: ' word' if single-token, else 'word', else first token of ' word'."""
    for w in (" " + word, word):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], True
    return tok.encode(" " + word, add_special_tokens=False)[0], False


NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
       "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
OPS = {"addition": ["+", "plus", "add", "addition", "sum"], "subtraction": ["-", "minus", "subtract", "subtraction", "difference"],
       "multiplication": ["*", "×", "times", "multiply", "multiplication", "product"],
       "division": ["/", "÷", "divided", "divide", "division", "quotient"]}


def variants(word: str, order_ops: bool = False) -> list[int]:
    """First-token ids of ' w', 'w', ' W', 'W' (lenient, applied identically to every lens);
    order-ops numbers/operations expanded to synonyms (same rule as src/eval_lens.py)."""
    words = {word}
    if order_ops:
        if word.isdigit() and int(word) < len(NUM):
            words.add(NUM[int(word)])
        for k, syn in OPS.items():
            if word.lower() in syn or word.lower() == k:
                words |= set(syn)
    ids = set()
    for w in words:
        for v in {w, w.lower(), w.capitalize()}:
            for p in ("", " "):
                e = [i for i in tok.encode(p + v, add_special_tokens=False) if tok.decode([i]).strip()]  # skip bare-space first token: ' 9' -> ' ', '9'
                if e:
                    ids.add(e[0])
    return sorted(ids)


# ---------------------------------------------------------------- residual stats per layer
@torch.no_grad()
def residual_stats(layers, n=8):
    acc = {l: [] for l in layers}
    with ActivationRecorder(m.layers, at=layers) as rec:
        for ids in ctx_ids[:n]:
            m.forward(ids)
            v = Perturbed(m, ids, 0, L).valid
            for l in layers:
                acc[l].append(rec.activations[l][0, v].float())
    stats = {}
    for l in layers:
        H = torch.cat(acc[l])
        stats[l] = {"mu": H.mean(0), "rho": float(H.norm(dim=-1).mean()),
                    "rho_centered": float((H - H.mean(0)).norm(dim=-1).mean())}
    return stats


all_layers = sorted(set(LAYERS) | set(int(x) for x in args.b3_layers.split(",")))
STATS = residual_stats(all_layers)
log("rho per layer:", {l: round(STATS[l]["rho"], 1) for l in all_layers},
    "rho_centered:", {l: round(STATS[l]["rho_centered"], 1) for l in all_layers})


def batched_Q(layer, a, b=None, *, signs, n_ctx, ctx_offset=0, second=True, exact_pos=0):
    """Average (Ja, Jb, Q) over n_ctx contexts for direction batches a, b [N, d] (chunked).
    exact_pos > 0: Q is the exact same-position block, averaged over exact_pos sampled source
    positions per context (no Rademacher sign trick; B2x showed the sign trick is noise-dominated)."""
    from torch.func import jvp
    N = a.shape[0]
    Ja = torch.zeros(N, m.d_model, device="cuda"); Jb = torch.zeros_like(Ja); Q = torch.zeros_like(Ja)
    pg = torch.Generator().manual_seed(1234 + layer)
    for ci in range(n_ctx):
        ids = ctx_ids[(ctx_offset + ci) % len(ctx_ids)]
        if exact_pos:
            valid = Perturbed(m, ids, layer, L).valid
            tps = valid[torch.randperm(len(valid), generator=pg)[:exact_pos].to(valid.device)].tolist()
        for s in range(0, N, args.batch):
            aa = a[s: s + args.batch].to(dtype)
            bb = None if b is None else b[s: s + args.batch].to(dtype)
            f = Perturbed(m, ids.expand(aa.shape[0], -1), layer, L)
            ja, jb, q = first_and_second(f, aa, bb, signs=signs, gen=gen, second=second and not exact_pos)
            Ja[s: s + len(aa)] += ja.float(); Jb[s: s + len(aa)] += jb.float(); Q[s: s + len(aa)] += q.float()
            if exact_pos and second:
                bq = aa if bb is None else bb
                for tp in tps:
                    tpos = torch.full((len(aa),), tp, device="cuda")
                    A, Bm = at_pos(f, aa, tpos), at_pos(f, bq, tpos)
                    with math_sdpa():
                        qx = jvp(lambda d: jvp(f, (d,), (A,))[1], (torch.zeros_like(A),), (Bm,))[1]
                    Q[s: s + len(aa)] += qx.float() / len(tps)
    return Ja / n_ctx, Jb / n_ctx, Q / n_ctx


def F_of(f, delta):
    with math_sdpa(), torch.no_grad():
        return f(delta).float()


# ---------------------------------------------------------------- stage: check
if "check" in stages:
    res = {}
    layer = CHECK_LAYER
    ids = ctx_ids[0]
    rho = STATS[layer]["rho"]
    _g = torch.Generator().manual_seed(7)  # fixed directions, independent of how loading used the RNG
    a = torch.randn(2, m.d_model, generator=_g).cuda(); a = a / a.norm(dim=-1, keepdim=True) * rho
    b = torch.randn(2, m.d_model, generator=_g).cuda(); b = b / b.norm(dim=-1, keepdim=True) * rho
    f = Perturbed(m, ids.expand(2, -1), layer, L)
    e = rademacher(2, len(f.valid), "cuda", gen)
    A, Bv = f.spread(a.to(dtype), e), f.spread(b.to(dtype), e)
    for tf32 in (0, 1):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        from torch.func import jvp
        with math_sdpa():
            _, Q = jvp(lambda d: jvp(f, (d,), (A,))[1], (torch.zeros_like(A),), (Bv,))
        torch.cuda.synchronize(); tq = time.perf_counter() - t0
        res[f"Q_tf32={tf32}"] = Q.float()
        res[f"secs_Q_batch2_tf32={tf32}"] = round(tq, 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    Q = res["Q_tf32=0"]
    cos_tf = torch.nn.functional.cosine_similarity(Q, res["Q_tf32=1"], dim=-1)
    fd = {}
    F0 = F_of(f, torch.zeros_like(A))
    for eps in (0.3, 0.1, 0.03, 0.01):
        D = (F_of(f, eps * (A + Bv)) - F_of(f, eps * A) - F_of(f, eps * Bv) + F0) / eps**2
        fd[eps] = {"cos": torch.nn.functional.cosine_similarity(D, Q, dim=-1).tolist(),
                   "rel_err": ((D - Q).norm(dim=-1) / Q.norm(dim=-1)).tolist()}
    fd_frozen = {}
    if N_ROUTERS:
        with so_core.routing("record"):
            F0r = F_of(f, torch.zeros_like(A))
        with so_core.routing("replay"):
            for eps in (0.3, 0.1, 0.03, 0.01):
                D = (F_of(f, eps * (A + Bv)) - F_of(f, eps * A) - F_of(f, eps * Bv) + F0r) / eps**2
                fd_frozen[eps] = {"cos": torch.nn.functional.cosine_similarity(D, Q, dim=-1).tolist(),
                                  "rel_err": ((D - Q).norm(dim=-1) / Q.norm(dim=-1)).tolist()}
    t0 = time.perf_counter(); F_of(f, torch.zeros_like(A)); torch.cuda.synchronize()
    check = {"layer": layer, "rho": rho, "fd_vs_jvp": fd, "fd_frozen_routing_vs_jvp": fd_frozen, "n_routers": N_ROUTERS, "cos_tf32_vs_fp32": cos_tf.tolist(),
             "Q_norm": Q.norm(dim=-1).tolist(), "secs_forward_batch2": round(time.perf_counter() - t0, 2),
             **{k: v for k, v in res.items() if k.startswith("secs")}}
    log("CHECK", json.dumps(check))
    dump("check", check)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

# ---------------------------------------------------------------- concept words
TRIPLES = [  # (a, b, conjunction c)
    ("France", "capital", "Paris"), ("Japan", "capital", "Tokyo"), ("Italy", "capital", "Rome"),
    ("Germany", "capital", "Berlin"), ("Spain", "capital", "Madrid"), ("Egypt", "capital", "Cairo"),
    ("Russia", "capital", "Moscow"), ("China", "capital", "Beijing"),
    ("Japan", "currency", "yen"), ("Britain", "currency", "pound"), ("Europe", "currency", "euro"),
    ("America", "currency", "dollar"), ("Brazil", "language", "Portuguese"), ("Mexico", "language", "Spanish"),
    ("spider", "legs", "eight"), ("insect", "legs", "six"), ("opposite", "hot", "cold"),
    ("opposite", "big", "small"), ("opposite", "up", "down"), ("opposite", "black", "white"),
    ("sky", "color", "blue"), ("grass", "color", "green"), ("banana", "color", "yellow"),
    ("blood", "color", "red"), ("snow", "color", "white"), ("dog", "baby", "puppy"),
    ("cat", "baby", "kitten"), ("king", "female", "queen"), ("actor", "female", "actress"),
    ("man", "plural", "men"), ("go", "past", "went"), ("eat", "past", "ate"),
    ("French", "hello", "bonjour"), ("Spanish", "thanks", "gracias"), ("Hamlet", "author", "Shakespeare"),
    ("planet", "red", "Mars"), ("moon", "Earth", "orbit"), ("bee", "make", "honey"),
    ("cow", "milk", "dairy"), ("water", "frozen", "ice"),
]
B1_WORDS = ["Paris", "Italy", "euro", "spider", "eight", "blue", "dog", "music", "anger", "Shakespeare",
            "France", "capital", "money", "red", "water", "love", "war", "seven", "Python", "doctor"]
WORDS = sorted({w for t in TRIPLES for w in t} | set(B1_WORDS))
TID = {w: tid(w) for w in WORDS}
VEC_PATH = out / "concept_vecs.pt"

# ---------------------------------------------------------------- stage: vecs
if "vecs" in stages:
    t0 = time.perf_counter()
    ids_k = torch.tensor([TID[w][0] for w in WORDS], device="cuda")
    wk = (W_U[ids_k].float() * gamma).to(dtype)  # [K, d]
    vecs = {l: torch.zeros(len(WORDS), m.d_model, device="cuda") for l in LAYERS}
    chunk = 4
    for ci in range(args.n_vec_ctx):
        ids = ctx_ids[ci]
        valid = Perturbed(m, ids, 0, L).valid
        s = 0
        while s < len(WORDS):
            k = min(chunk, len(WORDS) - s)
            try:
                with ActivationRecorder(m.layers, at=[*LAYERS, L], start_graph_at=min(LAYERS)) as rec, \
                        torch.enable_grad(), so_core.bf16_backward():
                    m.forward(ids.expand(k, -1))
                    tgt = rec.activations[L]
                    cot = torch.zeros_like(tgt)
                    cot[:, valid] = wk[s: s + k, None, :]
                    grads = torch.autograd.grad(tgt, [rec.activations[l] for l in LAYERS], grad_outputs=cot)
                for l, g in zip(LAYERS, grads):
                    vecs[l][s: s + k] += g[:, valid].float().mean(1)
                s += k
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); chunk = max(1, chunk // 2); log("OOM in vecs -> chunk", chunk)
        log(f"vecs ctx {ci + 1}/{args.n_vec_ctx}")
    vecs = {l: (v / args.n_vec_ctx).cpu() for l, v in vecs.items()}
    torch.save({"words": WORDS, "tid": TID, "vecs": vecs, "n_ctx": args.n_vec_ctx}, VEC_PATH)
    # sanity: does each v_k read out its own token (first-order J applied to v_k)?
    dump("vecs", {"secs": round(time.perf_counter() - t0, 1), "n_words": len(WORDS),
                  "multi_token_words": [w for w in WORDS if not TID[w][1]],
                  "norms": {l: {w: round(float(vecs[l][i].norm()), 3) for i, w in enumerate(WORDS)} for l in LAYERS}})

if any(s in stages for s in ("B1", "B2", "B2x", "B4c", "N1", "P1", "D1")):
    VV = torch.load(VEC_PATH)
    assert VV["words"] == WORDS, "concept_vecs.pt was built for a different word list"
    VHAT = {l: (v / v.norm(dim=-1, keepdim=True)).cuda() for l, v in VV["vecs"].items()}
    WIDX = {w: i for i, w in enumerate(WORDS)}

# ---------------------------------------------------------------- stage: B1 (curvature, steering)
if "B1" in stages:
    t0 = time.perf_counter()
    alphas = [0.05, 0.1, 0.2, 0.4, 0.8]
    words = [w for w in B1_WORDS if w in WIDX]
    kid = torch.tensor([TID[w][0] for w in words], device="cuda")
    res = {"alphas": alphas, "words": words, "layers": {}}
    for l in LAYERS:
        rho = STATS[l]["rho"]
        a = VHAT[l][[WIDX[w] for w in words]] * rho  # [N, d], unit lens dirs scaled to typical norm
        rows = {w: {"actual": [0.0] * len(alphas), "lin": [0.0] * len(alphas), "quad": [0.0] * len(alphas),
                    "frozen": [0.0] * len(alphas),
                    "cos_lin": [0.0] * len(alphas), "cos_quad": [0.0] * len(alphas)} for w in words}
        for ci in range(args.n_ctx):
            ids = ctx_ids[ci]
            for s in range(0, len(words), args.batch):
                aa = a[s: s + args.batch].to(dtype)
                f = Perturbed(m, ids.expand(len(aa), -1), l, L)
                ja, _, q = first_and_second(f, aa, None, signs=False)
                nV = len(f.valid)
                ja, q = ja.float() * nV, q.float() * nV  # back to sum-over-sources (all positions steered)
                F0 = F_of(f, torch.zeros(len(aa), f.T, m.d_model, device="cuda", dtype=dtype))
                A1 = f.spread(aa, None)
                if N_ROUTERS:
                    with so_core.routing("record"):
                        F0r = F_of(f, torch.zeros_like(A1))
                for ai, al in enumerate(alphas):
                    dF = F_of(f, al * A1) - F0
                    if N_ROUTERS:
                        with so_core.routing("replay"):
                            lgf = readout(F_of(f, al * A1) - F0r, W_U, gamma)
                        for j in range(len(aa)):
                            rows[words[s + j]]["frozen"][ai] += float(lgf[j, kid[s + j]]) / args.n_ctx
                    lin, quad = al * ja, al * ja + 0.5 * al**2 * q
                    lg = {k: readout(v, W_U, gamma) for k, v in (("act", dF), ("lin", lin), ("quad", quad))}
                    for j in range(len(aa)):
                        w = words[s + j]; k = kid[s + j]
                        rows[w]["actual"][ai] += float(lg["act"][j, k]) / args.n_ctx
                        rows[w]["lin"][ai] += float(lg["lin"][j, k]) / args.n_ctx
                        rows[w]["quad"][ai] += float(lg["quad"][j, k]) / args.n_ctx
                        cs = torch.nn.functional.cosine_similarity
                        rows[w]["cos_lin"][ai] += float(cs(lg["act"][j], lg["lin"][j], dim=0)) / args.n_ctx
                        rows[w]["cos_quad"][ai] += float(cs(lg["act"][j], lg["quad"][j], dim=0)) / args.n_ctx
        summ = {}
        for ai, al in enumerate(alphas):
            act = torch.tensor([rows[w]["actual"][ai] for w in words])
            lin = torch.tensor([rows[w]["lin"][ai] for w in words])
            quad = torch.tensor([rows[w]["quad"][ai] for w in words])
            frz = torch.tensor([rows[w]["frozen"][ai] for w in words])
            summ[al] = {"median_actual_over_lin": float((act / lin).median()),
                        **({"median_frozen_over_lin": float((frz / lin).median()),
                            "median_actual_over_frozen": float((act / frz).median())} if N_ROUTERS else {}),
                        "median_actual_over_quad": float((act / quad).median()),
                        "mean_cos_lin": sum(rows[w]["cos_lin"][ai] for w in words) / len(words),
                        "mean_cos_quad": sum(rows[w]["cos_quad"][ai] for w in words) / len(words)}
        res["layers"][l] = {"rho": rho, "summary": summ, "per_word": rows}
        log(f"B1 L{l}", json.dumps(summ))
        dump("B1", res)
    res["secs"] = round(time.perf_counter() - t0, 1); dump("B1", res)

# ---------------------------------------------------------------- stage: D1 (depth: nonlinearity vs layers traversed)
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


if "D1" in stages:
    t0 = time.perf_counter()
    alphas = [float(x) for x in args.d1_alphas.split(",")]
    ks = [int(x) for x in args.d1_ks.split(",")]
    srcs = sorted(set(int(x) for x in args.d1_sources.split(",")))
    nD, nC = args.d1_n_dir, args.d1_n_ctx
    assert len(ctx_ids) >= nC + nD, f"need {nC + nD} full-length contexts, have {len(ctx_ids)}"
    DST = residual_stats(srcs)
    # data directions: centred residuals at one random valid position of each of nD contexts disjoint from the steered ones
    dgen = torch.Generator().manual_seed(4321)
    data_dirs = {l: [] for l in srcs}
    with ActivationRecorder(m.layers, at=srcs) as rec, torch.no_grad():
        for ids in ctx_ids[-nD:]:
            m.forward(ids)
            v = Perturbed(m, ids, 0, L).valid
            tp = int(v[torch.randint(len(v), (1,), generator=dgen)])
            for l in srcs:
                data_dirs[l].append(rec.activations[l][0, tp].float() - DST[l]["mu"])
    words = [w for w in B1_WORDS if w in WIDX][:nD]
    res = {"alphas": alphas, "ks": ks, "n_ctx": nC, "n_dir": nD, "words": words, "n_layers": m.n_layers,
           "sources": {}}
    raw = {}
    cs = torch.nn.functional.cosine_similarity
    for l in srcs:
        rho = DST[l]["rho"]
        targets = sorted({min(l + k, L) for k in ks} | {L})
        types = {"iso": torch.randn(nD, m.d_model, generator=torch.Generator().manual_seed(77 + l)).cuda(),
                 "data": torch.stack(data_dirs[l])}
        if l in VHAT:
            types["lens"] = VHAT[l][[WIDX[w] for w in words]]
        kid = torch.tensor([TID[w][0] for w in words], device="cuda")
        M_ = {}  # type -> metric -> [nC, nD, nA, nT]
        for tn, U in types.items():
            U = (U / U.norm(dim=-1, keepdim=True)) * rho
            met = {k_: torch.zeros(nC, nD, len(alphas), len(targets)) for k_ in
                   ("rel_lin", "rel_quad", "proj", "lg_rel_lin", "lg_rel_quad", "lg_cos_lin", "lg_cos_quad", "own")}
            nrm = {k_: torch.zeros(nC, nD, len(targets)) for k_ in ("J", "Q")}
            for ci in range(nC):
                ids = ctx_ids[ci]
                for s0 in range(0, nD, args.batch):
                    aa = U[s0: s0 + args.batch].to(dtype)
                    B_ = len(aa)
                    f = MultiTarget(m, ids.expand(B_, -1), l, targets)
                    ja, _, q = first_and_second(f, aa, None, signs=False)
                    nV = len(f.valid)
                    ja, q = ja.float() * nV, q.float() * nV  # [B, nT, d], sum over steered sources
                    nrm["J"][ci, s0: s0 + B_] = ja.norm(dim=-1).cpu(); nrm["Q"][ci, s0: s0 + B_] = q.norm(dim=-1).cpu()
                    A1 = f.spread(aa, None)
                    with no_tf32():  # finite differences: no TF32 rounding in the cancellation F(a) - F(0)
                        F0 = F_of(f, torch.zeros_like(A1))
                        for ai, al in enumerate(alphas):
                            act = F_of(f, al * A1) - F0
                            lin, quad = al * ja, al * ja + 0.5 * al ** 2 * q
                            an = act.norm(dim=-1).clamp_min(1e-30)
                            sl = slice(s0, s0 + B_)
                            met["rel_lin"][ci, sl, ai] = ((act - lin).norm(dim=-1) / an).cpu()
                            met["rel_quad"][ci, sl, ai] = ((act - quad).norm(dim=-1) / an).cpu()
                            met["proj"][ci, sl, ai] = ((act * lin).sum(-1) / lin.pow(2).sum(-1).clamp_min(1e-30)).cpu()
                            la, ll, lq = (readout(v_, W_U, gamma) for v_ in (act, lin, quad))
                            lan = la.norm(dim=-1).clamp_min(1e-30)
                            met["lg_rel_lin"][ci, sl, ai] = ((la - ll).norm(dim=-1) / lan).cpu()
                            met["lg_rel_quad"][ci, sl, ai] = ((la - lq).norm(dim=-1) / lan).cpu()
                            met["lg_cos_lin"][ci, sl, ai] = cs(la, ll, dim=-1).cpu()
                            met["lg_cos_quad"][ci, sl, ai] = cs(la, lq, dim=-1).cpu()
                            if tn == "lens":  # B1's quantity: own-token logit, actual / linear
                                kk = kid[sl]
                                own_a = la[torch.arange(B_), :, kk]; own_l = ll[torch.arange(B_), :, kk]
                                met["own"][ci, sl, ai] = (own_a / own_l).cpu()
                            del la, ll, lq
            M_[tn] = {**met, **{f"norm_{k_}": v_ for k_, v_ in nrm.items()}}
        raw[l] = {"targets": targets, **M_}
        summ = {}
        for tn, met in M_.items():
            summ[tn] = {}
            for ti, t in enumerate(targets):
                row = {"k": t - l}
                for k_ in ("rel_lin", "rel_quad", "proj", "lg_rel_lin", "lg_rel_quad", "lg_cos_lin", "lg_cos_quad"):
                    row[k_] = [round(float(met[k_][:, :, ai, ti].median()), 5) for ai in range(len(alphas))]
                if tn == "lens":
                    row["own_act_over_lin"] = [round(float(met["own"][:, :, ai, ti].median()), 5) for ai in range(len(alphas))]
                row["Q_over_J"] = round(float((met["norm_Q"][:, :, ti] / met["norm_J"][:, :, ti]).median()), 5)
                summ[tn][t] = row
        res["sources"][l] = {"rho": rho, "nominal": round(l * 64 / m.n_layers), "targets": targets, "summary": summ}
        log(f"D1 L{l} rho={rho:.1f}", json.dumps({tn: {t: (r["k"], r["rel_lin"][2], r["rel_quad"][2]) for t, r in sm.items()}
                                                  for tn, sm in summ.items()}))
        dump("D1", res)
        torch.save(raw, out / "D1_raw.pt")
    res["secs"] = round(time.perf_counter() - t0, 1)
    res["defs"] = ("Steer alpha*rho_l*unit(dir) at every valid position of block l; response at block t = l+k summed "
                   "over valid positions. rel_lin = |act - alpha J d| / |act|, rel_quad with + alpha^2 Q(d,d)/2; proj = "
                   "<act, lin>/|lin|^2 (actual/linear along the linear response); lg_* the same through W_U gamma "
                   "(logit-lens space at intermediate t); own = own-token logit act/lin (lens dirs, B1's metric). "
                   "Medians over contexts x directions; lists are per alpha. Q_over_J = |Q(d,d)|/|J d| at alpha=1.")
    dump("D1", res)

# ---------------------------------------------------------------- stage: B2 (binding / interaction)
if "B2" in stages:
    t0 = time.perf_counter()
    trip = [t for t in TRIPLES if all(w in WIDX for w in t)]
    n = len(trip)
    shift = [(i + 7) % n for i in range(n)]  # mismatched b for controls (fixed derangement)
    res = {"triples": trip, "multi_token": [w for t in trip for w in t if not TID[w][1]], "layers": {}}
    for l in LAYERS:
        rho = STATS[l]["rho"]
        a = VHAT[l][[WIDX[t[0]] for t in trip]] * rho
        b = VHAT[l][[WIDX[t[1]] for t in trip]] * rho
        bx = b[shift]
        Ja, Jb, Q = batched_Q(l, a, b, signs=True, n_ctx=args.n_ctx)
        _, _, Qx = batched_Q(l, a, bx, signs=True, n_ctx=args.n_ctx, ctx_offset=0)
        Qc = Q - Q.mean(0, keepdim=True)
        Qxc = Qx - Q.mean(0, keepdim=True)
        lg = {"Q": readout(Q, W_U, gamma), "Qc": readout(Qc, W_U, gamma), "J_ab": readout(Ja + Jb, W_U, gamma),
              "Q_mismatch": readout(Qx, W_U, gamma), "Qc_mismatch": readout(Qxc, W_U, gamma)}
        per = []
        for i, (wa, wb, wc) in enumerate(trip):
            cids = variants(wc)
            r = {k: rank_of(v[i], cids) for k, v in lg.items()}
            r["top10_Qc"] = [tok.decode([int(t)]) for t in lg["Qc"][i].topk(10).indices]
            r["top10_J_ab"] = [tok.decode([int(t)]) for t in lg["J_ab"][i].topk(10).indices]
            per.append({"a": wa, "b": wb, "c": wc, **r})
        summ = {k: {"median_rank": float(torch.tensor([p[k] for p in per]).float().median()),
                    "top10": sum(p[k] < 10 for p in per) / n, "top100": sum(p[k] < 100 for p in per) / n}
                for k in lg}
        summ["Q_norm_over_J_norm"] = float((Q.norm(dim=-1) / (Ja + Jb).norm(dim=-1)).median())
        summ["cos_Q_across_pairs_mean"] = float(torch.nn.functional.cosine_similarity(
            Q[:, None], Q[None], dim=-1).fill_diagonal_(0).sum() / (n * (n - 1)))
        res["layers"][l] = {"summary": summ, "per_triple": per}
        log(f"B2 L{l}", json.dumps(summ))
        torch.save({"Q": Q.cpu(), "Qx": Qx.cpu(), "Ja": Ja.cpu(), "Jb": Jb.cpu()}, out / f"B2_L{l}.pt")
        dump("B2", res)
    res["secs"] = round(time.perf_counter() - t0, 1); dump("B2", res)

# ---------------------------------------------------------------- stage: B2x (exact same-position binding)
def at_pos(f, vecs, tpos):
    out_ = torch.zeros(vecs.shape[0], f.T, vecs.shape[1], device=vecs.device, dtype=vecs.dtype)
    out_[torch.arange(len(tpos)), tpos] = vecs
    return out_


if "B2x" in stages:
    from torch.func import jvp
    t0 = time.perf_counter()
    trip = [t for t in TRIPLES if all(w in WIDX for w in t)]
    n = len(trip)
    shift = [(i + 7) % n for i in range(n)]
    alphas = [0.25, 0.5, 1.0]
    res = {"triples": trip, "alphas": alphas, "n_pos": args.b2x_n_pos, "n_ctx": args.b2x_n_ctx, "layers": {}}
    pgen = torch.Generator().manual_seed(args.seed + 1)
    for l in [int(x) for x in args.b2x_layers.split(",")]:
        rho = STATS[l]["rho"]
        a = VHAT[l][[WIDX[t[0]] for t in trip]] * rho
        b = VHAT[l][[WIDX[t[1]] for t in trip]] * rho
        bx = b[shift]
        keys = ["Ja", "Jb", "Q", "Qx"] + [f"I{al}" for al in alphas] + [f"Ix{al}" for al in alphas]
        acc = {k: torch.zeros(n, m.d_model, device="cuda") for k in keys}
        cnt = 0
        for ci in range(args.b2x_n_ctx):
            ids = ctx_ids[ci]
            valid = Perturbed(m, ids, l, L).valid
            tps = valid[torch.randperm(len(valid), generator=pgen)[: args.b2x_n_pos].to(valid.device)]
            for tp in tps.tolist():
                cnt += 1
                for s in range(0, n, args.batch):
                    aa, bb, bxx = (v[s: s + args.batch].to(dtype) for v in (a, b, bx))
                    k_ = len(aa)
                    f = Perturbed(m, ids.expand(k_, -1), l, L)
                    tpos = torch.full((k_,), tp, device="cuda")
                    A, Bm, Bx = at_pos(f, aa, tpos), at_pos(f, bb, tpos), at_pos(f, bxx, tpos)
                    zero = torch.zeros_like(A)
                    with math_sdpa():
                        ja = jvp(f, (zero,), (A,))[1]; jb = jvp(f, (zero,), (Bm,))[1]
                        inner = lambda d: jvp(f, (d,), (A,))[1]  # noqa: E731
                        q = jvp(inner, (zero,), (Bm,))[1]; qx = jvp(inner, (zero,), (Bx,))[1]
                    sl = slice(s, s + k_)
                    acc["Ja"][sl] += ja.float(); acc["Jb"][sl] += jb.float()
                    acc["Q"][sl] += q.float(); acc["Qx"][sl] += qx.float()
                    F0 = F_of(f, zero)
                    for al in alphas:
                        Fa = F_of(f, al * A)
                        acc[f"I{al}"][sl] += F_of(f, al * (A + Bm)) - Fa - F_of(f, al * Bm) + F0
                        acc[f"Ix{al}"][sl] += F_of(f, al * (A + Bx)) - Fa - F_of(f, al * Bx) + F0
            log(f"B2x L{l} ctx {ci + 1}/{args.b2x_n_ctx}")
        acc = {k: v / cnt for k, v in acc.items()}
        torch.save({k: v.cpu() for k, v in acc.items()}, out / f"B2x_L{l}.pt")
        R = {"J_a": acc["Ja"], "J_b": acc["Jb"], "J_a+J_b": acc["Ja"] + acc["Jb"], "Q": acc["Q"], "Q_mismatch": acc["Qx"],
             **{f"I{al}": acc[f"I{al}"] for al in alphas}, **{f"I{al}_mismatch": acc[f"Ix{al}"] for al in alphas}}
        lg = {k: readout(v, W_U, gamma) for k, v in R.items()}
        per = []
        for i, (wa, wb, wc) in enumerate(trip):
            cids = variants(wc)
            r = {k: rank_of(v[i], cids) for k, v in lg.items()}
            r["top8_Q"] = [tok.decode([int(t)]) for t in lg["Q"][i].topk(8).indices]
            r["top8_I1.0"] = [tok.decode([int(t)]) for t in lg["I1.0"][i].topk(8).indices]
            per.append({"a": wa, "b": wb, "c": wc, **r})
        med = lambda xs: float(torch.tensor(xs).float().median())  # noqa: E731
        summ = {k: {"median_rank": med([p[k] for p in per]), "top10": sum(p[k] < 10 for p in per) / n,
                    "top100": sum(p[k] < 100 for p in per) / n} for k in lg}
        for k in ["Q"] + [f"I{al}" for al in alphas]:
            summ[k]["frac_beats_both_singles"] = sum(p[k] < min(p["J_a"], p["J_b"]) for p in per) / n
            summ[k]["frac_beats_mismatch"] = sum(p[k] < p[k + "_mismatch"] for p in per) / n
        jn = (acc["Ja"] + acc["Jb"]).norm(dim=-1)
        summ["norm_ratio_Q_over_Jab"] = float((acc["Q"].norm(dim=-1) / jn).median())
        for al in alphas:
            summ[f"norm_ratio_I{al}_over_alpha_Jab"] = float((acc[f"I{al}"].norm(dim=-1) / (al * jn)).median())
            summ[f"cos_I{al}_vs_alpha2_Q"] = float(torch.nn.functional.cosine_similarity(acc[f"I{al}"], acc["Q"], dim=-1).mean())
        p_signed = out / f"B2_L{l}.pt"
        if p_signed.exists():
            Qs = torch.load(p_signed)["Q"].cuda()
            summ["cos_signedQ_vs_exactQ"] = float(torch.nn.functional.cosine_similarity(Qs, acc["Q"], dim=-1).mean())
            summ["norm_signedQ_over_exactQ"] = float((Qs.norm(dim=-1) / acc["Q"].norm(dim=-1)).median())
        res["layers"][l] = {"rho": rho, "summary": summ, "per_triple": per}
        log(f"B2x L{l}", json.dumps(summ))
        dump("B2x", res)
    res["secs"] = round(time.perf_counter() - t0, 1); dump("B2x", res)


# ---------------------------------------------------------------- stage: B4f (2x2 factorial, cross-position)
def b4f_items():
    """(task, P00, Ps, Pt, Pst, answer_spec). answer_spec None -> model argmax on clean cells;
    else (A, B) for the XOR negation task: cells (very,A)->A, (not,A)->B, (very,B)->B, (not,B)->A."""
    items = []
    shot = "Germany, capital: Berlin\nGermany, currency: euro\nGermany, language: German\n"
    subj = [("France", "Japan"), ("Italy", "China"), ("Spain", "Russia"), ("Brazil", "India"), ("Egypt", "Mexico")]
    rels = [("capital", "currency"), ("capital", "language"), ("currency", "language")]
    for s0, s1 in subj:
        for r0, r1 in rels:
            P = lambda s_, r_: f"{shot}{s_}, {r_}:"  # noqa: E731
            items.append(("relation", P(s0, r0), P(s0, r1), P(s1, r0), P(s1, r1), None))
    shot = "2+5=7\n1+3=4\n3+3=6\n"
    seen = 0
    for x in range(1, 5):
        for xp in range(1, 5):
            for y in range(1, 6):
                for yp in range(1, 6):
                    sums = {x + y, xp + y, x + yp, xp + yp}
                    if x < xp and y < yp and len(sums) == 4 and max(sums) < 10 and seen < 24:
                        seen += 1
                        P = lambda a_, b_: f"{shot}{a_}+{b_}="  # noqa: E731
                        items.append(("arith_xy", P(x, y), P(xp, y), P(x, yp), P(xp, yp), None))
    neg = [("hot", "cold", ["tea", "room", "soup", "water"]), ("big", "small", ["box", "house", "dog"]),
           ("happy", "sad", ["man", "girl", "child"]), ("fast", "slow", ["car", "train", "horse"]),
           ("old", "new", ["car", "book"])]
    for A_, B_, nouns in neg:
        for nn in nouns:
            for mod in ("very", "really"):
                P = lambda mo, ad: f"Q: The {nn} is {mo} {ad}. Is it {A_} or {B_}?\nA: It is"  # noqa: E731
                items.append(("negation_xor", P(mod, A_), P("not", A_), P(mod, B_), P("not", B_), (A_, B_)))
    return items


if "B4f" in stages:
    from torch.func import jvp
    t0 = time.perf_counter()
    b4_layers = [int(x) for x in args.b4f_layers.split(",")]
    lams = [0.25, 0.5, 1.0]
    rows, skipped = [], []
    for task, P00, Ps, Pt, Pst, spec in b4f_items():
        ids = {k: m.encode(p_, max_length=256) for k, p_ in (("00", P00), ("s", Ps), ("t", Pt), ("st", Pst))}
        if len({v.shape[1] for v in ids.values()}) != 1:
            skipped.append((task, P00[-40:], "length")); continue
        ds_pos = (ids["00"] != ids["s"])[0].nonzero().flatten().tolist()
        dt_pos = (ids["00"] != ids["t"])[0].nonzero().flatten().tolist()
        if len(ds_pos) != 1 or len(dt_pos) != 1:
            skipped.append((task, P00[-40:], f"slots {ds_pos} {dt_pos}")); continue
        sp, tp = ds_pos[0], dt_pos[0]
        H, clean = {}, {}
        with torch.no_grad(), ActivationRecorder(m.layers, at=sorted(set(b4_layers) | {L})) as rec:
            for k, v in ids.items():
                m.forward(v)
                H[k] = {l: rec.activations[l][0].clone() for l in b4_layers}
                clean[k] = m.unembed(rec.activations[L][0, -1]).float()
        if spec is None:
            c = {k: int(clean[k].argmax()) for k in clean}
            if c["st"] in (c["s"], c["t"]) or c["s"] == c["t"]:
                skipped.append((task, P00[-40:], "answers not distinct")); continue
        else:
            A_id, B_id = tid(spec[0])[0], tid(spec[1])[0]
            c = {"00": A_id, "s": B_id, "t": B_id, "st": A_id}
        S = lambda X: float(X[c["st"]] - 0.5 * (X[c["s"]] + X[c["t"]]))  # noqa: E731  interaction contrast
        row = {"task": task, "prompt_00": P00.split("\n")[-1], "answers": {k: tok.decode([v]) for k, v in c.items()},
               "clean_ok": {k: int(clean[k].argmax()) == c[k] for k in c} if spec else None,
               "S_clean_interaction": S(clean["st"]) - S(clean["s"]) - S(clean["t"]) + S(clean["00"]), "layers": {}}
        for l in b4_layers:
            T = ids["00"].shape[1]
            Ds = torch.zeros(1, T, m.d_model, device="cuda", dtype=dtype); Dt = torch.zeros_like(Ds)
            Ds[0, sp] = (H["s"][l][sp] - H["00"][l][sp]).to(dtype)
            Dt[0, tp] = (H["t"][l][tp] - H["00"][l][tp]).to(dtype)
            f2 = Perturbed(m, ids["00"].expand(2, -1), l, L, reduce="last")
            f3 = Perturbed(m, ids["00"].expand(3, -1), l, L, reduce="last")
            with math_sdpa():
                J2 = jvp(f2, (torch.zeros(2, T, m.d_model, device="cuda", dtype=dtype),), (torch.cat([Ds, Dt]),))[1].float()
                inner_t = torch.cat([Ds, Ds, Dt]); outer_t = torch.cat([Dt, Ds, Dt])
                Q3 = jvp(lambda d: jvp(f3, (d,), (inner_t,))[1], (torch.zeros_like(inner_t),), (outer_t,))[1].float()
            Jds, Jdt = J2[0], J2[1]
            Qst, Qss, Qtt = Q3[0], Q3[1], Q3[2]
            pert = []
            for lam in lams:
                pert += [torch.zeros_like(Ds), lam * Ds, lam * Dt, lam * (Ds + Dt)]
            fN = Perturbed(m, ids["00"].expand(len(pert), -1), l, L, reduce="last")
            G = F_of(fN, torch.cat(pert))  # [4*len(lams), d]
            G0 = G[0]
            lin = lambda y: readout(y, W_U, gamma)  # noqa: E731
            st = {}
            for li, lam in enumerate(lams):
                g0, gs, gt, gst = G[4 * li: 4 * li + 4]
                I = gst - gs - gt + g0
                st[lam] = {"S_patch": S(lin(I)) / lam**2,
                           "cos_I_over_lam2_vs_Qst": float(torch.nn.functional.cosine_similarity(lin(I), lin(Qst), dim=0)),
                           "normI_over_lam2_normQst": float(I.norm() / lam**2 / Qst.norm())}
            g_both = G[-1]
            pred1 = G0 + Jds + Jdt
            pred2 = pred1 + 0.5 * Qss + 0.5 * Qtt + Qst
            rk = lambda y: rank_of(m.unembed(y.to(W_U.dtype)).float(), [c["st"]])  # noqa: E731
            row["layers"][l] = {"S_Qst": S(lin(Qst)), "lam": st,
                                "rank_cst_actual_patch": rk(g_both), "rank_cst_first": rk(pred1), "rank_cst_second": rk(pred2),
                                "rank_cst_first_plus_Qst_only": rk(pred1 + Qst),
                                "total_effect_norm_ratio_actual_vs_first": float((g_both - G0).norm() / (Jds + Jdt).norm())}
        rows.append(row)
    summ = {}
    for task in sorted({r["task"] for r in rows}):
        rr = [r for r in rows if r["task"] == task]
        summ[task] = {"n": len(rr), "S_clean_mean": sum(r["S_clean_interaction"] for r in rr) / len(rr), "layers": {}}
        for l in b4_layers:
            xs = [r["layers"][l] for r in rr]
            Sa = torch.tensor([x["lam"][1.0]["S_patch"] for x in xs]); Sq = torch.tensor([x["S_Qst"] for x in xs])
            Ssmall = torch.tensor([x["lam"][0.25]["S_patch"] for x in xs])
            corr = lambda u, v: float(torch.corrcoef(torch.stack([u, v]))[0, 1]) if len(u) > 2 else float("nan")  # noqa: E731
            summ[task]["layers"][l] = {
                "top1_actual_patch": sum(x["rank_cst_actual_patch"] == 0 for x in xs) / len(xs),
                "top1_first_order": sum(x["rank_cst_first"] == 0 for x in xs) / len(xs),
                "top1_second_order": sum(x["rank_cst_second"] == 0 for x in xs) / len(xs),
                "top1_first_plus_Qst": sum(x["rank_cst_first_plus_Qst_only"] == 0 for x in xs) / len(xs),
                "median_rank_first": float(torch.tensor([x["rank_cst_first"] for x in xs]).float().median()),
                "median_rank_second": float(torch.tensor([x["rank_cst_second"] for x in xs]).float().median()),
                "S_actual_mean": float(Sa.mean()), "S_Qst_mean": float(Sq.mean()), "S_actual_lam0.25_mean": float(Ssmall.mean()),
                "frac_S_actual_pos": float((Sa > 0).float().mean()), "frac_S_Qst_pos": float((Sq > 0).float().mean()),
                "sign_agree": float(((Sa > 0) == (Sq > 0)).float().mean()), "corr_S_actual_Qst": corr(Sa, Sq),
                "corr_S_lam0.25_Qst": corr(Ssmall, Sq),
                "cos_I_vs_Qst_by_lam": {lam: sum(x["lam"][lam]["cos_I_over_lam2_vs_Qst"] for x in xs) / len(xs) for lam in lams},
            }
        log(f"B4f {task}", json.dumps(summ[task]))
    dump("B4f", {"summary": summ, "rows": rows, "skipped": skipped, "secs": round(time.perf_counter() - t0, 1)})

# ---------------------------------------------------------------- stage: B4c (competition / self-curvature)
if "B4c" in stages:
    from torch.func import jvp
    t0 = time.perf_counter()
    W12 = ["France", "Paris", "Italy", "euro", "dog", "cat", "seven", "eight", "music", "anger", "doctor", "water"]
    W12 = [w for w in W12 if w in WIDX]
    related = {frozenset(p) for p in [("France", "Paris"), ("Italy", "euro"), ("dog", "cat"), ("seven", "eight"),
                                      ("France", "Italy"), ("Paris", "euro")]}
    nW = len(W12)
    pairs = [(i, j) for i in range(nW) for j in range(nW) if i != j] + [(i, "rand") for i in range(nW)] + [(i, i) for i in range(nW)]
    cat_of = lambda i, j: "self" if j == i else "random" if j == "rand" else ("related" if frozenset((W12[i], W12[j])) in related else "unrelated")  # noqa: E731
    pgen = torch.Generator().manual_seed(args.seed + 2)
    res = {"words": W12, "layers": {}}
    alpha = 0.5
    for l in [int(x) for x in args.b4c_layers.split(",")]:
        rho = STATS[l]["rho"]
        Vd = VHAT[l][[WIDX[w] for w in W12]] * rho
        R = torch.randn(nW, m.d_model, generator=torch.Generator().manual_seed(l)).cuda()
        R = R / R.norm(dim=-1, keepdim=True) * rho
        wv = (W_U[[TID[w][0] for w in W12]].float() * gamma)  # [nW, d] readout rows
        acc = {"q": torch.zeros(len(pairs)), "with": torch.zeros(len(pairs)), "without": torch.zeros(len(pairs)),
               "own": torch.zeros(nW)}
        cnt = 0
        for ci in range(args.b4c_n_ctx):
            ids = ctx_ids[ci]
            valid = Perturbed(m, ids, l, L).valid
            for tpv in valid[torch.randperm(len(valid), generator=pgen)[: args.b4c_n_pos].to(valid.device)].tolist():
                cnt += 1
                f = Perturbed(m, ids.expand(nW, -1), l, L)
                Aall = at_pos(f, Vd.to(dtype), torch.full((nW,), tpv, device="cuda"))
                with math_sdpa():
                    ja = jvp(f, (torch.zeros_like(Aall),), (Aall,))[1].float()
                acc["own"] += (ja * wv).sum(-1).cpu()
                for s in range(0, len(pairs), args.batch):
                    pb = pairs[s: s + args.batch]; k_ = len(pb)
                    a1 = torch.stack([Vd[i] for i, _ in pb]).to(dtype)
                    b1 = torch.stack([R[i] if j == "rand" else Vd[j] for i, j in pb]).to(dtype)
                    fk = Perturbed(m, ids.expand(k_, -1), l, L)
                    tpos = torch.full((k_,), tpv, device="cuda")
                    A, Bm = at_pos(fk, a1, tpos), at_pos(fk, b1, tpos)
                    zero = torch.zeros_like(A)
                    with math_sdpa():
                        q = jvp(lambda d: jvp(fk, (d,), (A,))[1], (zero,), (Bm,))[1].float()
                    w_ = wv[[i for i, _ in pb]]
                    acc["q"][s: s + k_] += (q * w_).sum(-1).cpu()
                    G0 = F_of(fk, zero); Gb = F_of(fk, alpha * Bm)
                    acc["with"][s: s + k_] += ((F_of(fk, alpha * (A + Bm)) - Gb) * w_).sum(-1).cpu()
                    acc["without"][s: s + k_] += ((F_of(fk, alpha * A) - G0) * w_).sum(-1).cpu()
            log(f"B4c L{l} ctx {ci + 1}/{args.b4c_n_ctx}")
        acc = {k: v / cnt for k, v in acc.items()}
        per = []
        for pi, (i, j) in enumerate(pairs):
            own = float(acc["own"][i])
            per.append({"a": W12[i], "b": "rand" if j == "rand" else W12[j], "cat": cat_of(i, j),
                        "kappa": float(acc["q"][pi]) / own,  # frac. change of a's own effect per unit b (2nd order)
                        "finite_ratio_minus1": float(acc["with"][pi] / acc["without"][pi]) - 1.0, "own_first": own})
        summ = {}
        for cat in ("self", "random", "related", "unrelated"):
            pc = [p for p in per if p["cat"] == cat]
            kap = torch.tensor([p["kappa"] for p in pc]); fin = torch.tensor([p["finite_ratio_minus1"] for p in pc])
            summ[cat] = {"n": len(pc), "kappa_median": float(kap.median()), "kappa_frac_neg": float((kap < 0).float().mean()),
                         "finite_median": float(fin.median()), "finite_frac_neg": float((fin < 0).float().mean())}
        res["layers"][l] = {"rho": rho, "alpha_finite": alpha, "summary": summ, "pairs": per}
        log(f"B4c L{l}", json.dumps(summ))
        dump("B4c", res)
    res["secs"] = round(time.perf_counter() - t0, 1); dump("B4c", res)

# ---------------------------------------------------------------- stage: N1 (null space of the mean lens)
# The fitted mean lens E[J_l] is low-rank early (eff. rank 265/5120 at L24 of Qwen3.8-27B). Directions it maps to
# ~0 can be (a) dead, (b) first-order but context-dependent (per-context J_c n large, cancelling in the mean), or
# (c) purely nonlinear (only Q acts). Bands of E[J_l]'s right singular vectors by cumulative energy s^2:
# top (90%), mid (90-99%), tail (last 1%), plus iso (isotropic random). Per band, random unit directions n, steered
# at every valid position with alpha*rho*n (B1 convention); everything read out in centred-logit space R(.):
#   lin/lens/q       |R(J_c n)|, |R(E[J] n)|, |R(Q_c(n,n))| at alpha=1 (unit-rho steer)
#   act{a}           |R(F(a n) - F(0))|;  err_{lin,quad,lens}{a}: relative error of each prediction
#   p_scale          log-log slope of act between alpha .05 and .2 (1 = first order, 2 = pure second order)
#   cross            alpha where |first-order term| = |second-order term|
#   coherence        |mean_c R(J_c n)| / mean_c |R(J_c n)| across contexts ((b) -> small)
#   gate_*           does n modulate a concept a? |u_a.Q(a,n)| / u_a.J a (own-token gain change per rho of n),
#                    |Q(a,n)|/|J a|, and the finite interaction I = F(b(a+n))-F(ba)-F(bn)+F(0) vs Q (b=.25)
#   var_frac         share of the model's own centred residual variance that lies in the band
if "N1" in stages:
    import glob
    t0 = time.perf_counter()
    n1_layers = [int(x) for x in args.n1_layers.split(",")]
    alphas, beta = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6], 0.25
    files = sorted(glob.glob(f"{args.n1_lens_dir}/shard*of*.pt"))
    assert files, f"no lens shards in {args.n1_lens_dir}"
    JL, ntot = {}, 0
    for fp in files:  # n_prompts-weighted mean (= JacobianLens.merge), only the layers needed
        ck = torch.load(fp, map_location="cpu", weights_only=True, mmap=True)
        n = ck["n_prompts"]; ntot += n
        for l in n1_layers:
            x = ck["J"][l].float() * n
            JL[l] = x if l not in JL else JL[l] + x
        del ck
    JL = {l: (v / ntot).cuda() for l, v in JL.items()}
    log(f"N1 lens: {len(files)} shards, {ntot} prompts, layers {n1_layers}")
    cwords = [w for w in B1_WORDS if w in WIDX][: args.batch]
    ckid = torch.tensor([TID[w][0] for w in cwords], device="cuda")
    res = {"alphas": alphas, "beta": beta, "concepts": cwords, "lens_dir": args.n1_lens_dir, "n_prompts": ntot,
           "tf32": args.tf32, "layers": {}}

    def R(y):  # centred logits (a uniform logit shift is irrelevant)
        lg = readout(y, W_U, gamma)
        return lg - lg.mean(-1, keepdim=True)

    def nrm(x):
        return x.norm(dim=-1)

    keys = ["lin", "lens", "q", "cross", "p_scale", "gate_own", "gate_norm", "I_cos_Q", "I_over_Q", "I_over_Ja"] + \
        [f"{k}{a}" for k in ("act", "err_lin", "err_quad", "err_lens") for a in alphas]
    for l in n1_layers:
        rho = STATS[l]["rho"]
        J = JL[l]
        _, S, Vh = torch.linalg.svd(J, full_matrices=False)  # J [d_out, d_in]: input directions = rows of Vh
        p = S ** 2 / (S ** 2).sum(); e = p.cumsum(0)
        k90, k99 = int((e < 0.90).sum()) + 1, int((e < 0.99).sum()) + 1
        bands = {"top": Vh[:k90], "mid": Vh[k90:k99], "tail": Vh[k99:], "iso": None}
        with torch.no_grad(), ActivationRecorder(m.layers, at=[l]) as rec:
            Hs = []
            for ids in ctx_ids[:args.n1_n_ctx]:
                m.forward(ids)
                Hs.append(rec.activations[l][0, Perturbed(m, ids, 0, L).valid].float().cuda())
        H = torch.cat(Hs); Hc = H - STATS[l]["mu"].cuda()
        var = {b: (None if V is None else float((Hc @ V.T).pow(2).sum() / Hc.pow(2).sum())) for b, V in bands.items()}
        raw = {b: (None if V is None else float((H @ V.T).pow(2).sum() / H.pow(2).sum())) for b, V in bands.items()}
        g = torch.Generator().manual_seed(100 + l)
        dirs = {}
        for b, V in bands.items():
            X = torch.randn(args.batch, m.d_model if V is None else V.shape[0], generator=g).cuda()
            X = X if V is None else X @ V
            dirs[b] = X / X.norm(dim=-1, keepdim=True)
        conc = VHAT[l][[WIDX[w] for w in cwords]]
        acc = {b: {k: [] for k in keys} for b in bands}
        jsum = {b: 0 for b in bands}; jnorm = {b: 0 for b in bands}
        own = lambda X: X.gather(1, ckid[:, None])[:, 0]  # noqa: E731
        for ci in range(args.n1_n_ctx):
            ids = ctx_ids[ci]
            for b, D in dirs.items():
                nn_ = (D * rho).to(dtype)
                f = Perturbed(m, ids.expand(len(nn_), -1), l, L)
                nV = len(f.valid)
                jn, _, qn = first_and_second(f, nn_, None, signs=False)
                jn, qn = jn.float() * nV, qn.float() * nV  # every valid position steered (B1 convention)
                lensn = (D * rho) @ J.T * nV  # mean-lens prediction of the same steer
                F0 = F_of(f, torch.zeros(len(nn_), f.T, m.d_model, device="cuda", dtype=dtype))
                A1 = f.spread(nn_, None)
                Rj, Rq, Rl = R(jn), R(qn), R(lensn)
                a_ = acc[b]
                a_["lin"] += nrm(Rj).tolist(); a_["lens"] += nrm(Rl).tolist(); a_["q"] += nrm(Rq).tolist()
                a_["cross"] += (2 * nrm(Rj) / nrm(Rq)).tolist()
                jsum[b] = jsum[b] + Rj; jnorm[b] = jnorm[b] + nrm(Rj)
                acts = {}
                for al in alphas:
                    Ra = R(F_of(f, al * A1) - F0); acts[al] = nrm(Ra)
                    a_[f"act{al}"] += acts[al].tolist()
                    a_[f"err_lin{al}"] += (nrm(Ra - al * Rj) / acts[al]).tolist()
                    a_[f"err_quad{al}"] += (nrm(Ra - al * Rj - 0.5 * al ** 2 * Rq) / acts[al]).tolist()
                    a_[f"err_lens{al}"] += (nrm(Ra - al * Rl) / acts[al]).tolist()
                a_["p_scale"] += (torch.log(acts[0.2] / acts[0.05]) / math.log(4)).tolist()
                aa = (conc * rho).to(dtype)  # gating: does n modulate concept a's effect?
                ja, _, qan = first_and_second(f, aa, nn_, signs=False)
                ja, qan = ja.float() * nV, qan.float() * nV
                B_ = f.spread(aa, None)
                I = F_of(f, beta * (B_ + A1)) - F_of(f, beta * B_) - F_of(f, beta * A1) + F0
                Rja, Rqan, RI = R(ja), R(qan), R(I)
                a_["gate_own"] += (own(Rqan).abs() / own(Rja)).tolist()
                a_["gate_norm"] += (nrm(Rqan) / nrm(Rja)).tolist()
                a_["I_cos_Q"] += torch.nn.functional.cosine_similarity(RI, Rqan, dim=-1).tolist()
                a_["I_over_Q"] += (nrm(RI) / (beta ** 2 * nrm(Rqan))).tolist()
                a_["I_over_Ja"] += (nrm(RI) / (beta * nrm(Rja))).tolist()
            log(f"N1 L{l} ctx {ci + 1}/{args.n1_n_ctx}")
        summ = {}
        for b in bands:
            s_ = {k: float(torch.tensor(v).median()) for k, v in acc[b].items() if v}
            s_["coherence"] = float((nrm(jsum[b]) / jnorm[b]).median())
            s_["dim"] = m.d_model if bands[b] is None else bands[b].shape[0]
            s_["var_frac"], s_["raw_frac"] = var[b], raw[b]
            summ[b] = s_
        res["layers"][l] = {"rho": rho, "k90": k90, "k99": k99, "eff_rank": float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum())),
                            "top_sv": float(S[0]), "summary": summ, "per_dir": acc}
        log(f"N1 L{l} k90={k90} k99={k99}", json.dumps({b: {k: (round(v, 4) if isinstance(v, float) else v)
                                                              for k, v in s_.items()} for b, s_ in summ.items()}))
        dump(args.n1_name, res)
    res["secs"] = round(time.perf_counter() - t0, 1); dump(args.n1_name, res)


# ---------------------------------------------------------------- lens loader (fitted mean J, selected layers)
def load_mean_lens(lens_dir, layers):
    import glob
    files = sorted(glob.glob(f"{lens_dir}/shard*of*.pt"))
    assert files, f"no lens shards in {lens_dir}"
    acc, ntot = {}, 0
    for fp in files:  # n_prompts-weighted mean (= JacobianLens.merge)
        ck = torch.load(fp, map_location="cpu", weights_only=True, mmap=True)
        n = ck["n_prompts"]; ntot += n
        for l in layers:
            x = ck["J"][l].float() * n
            acc[l] = x if l not in acc else acc[l] + x
        del ck
    return {l: (v / ntot).cuda() for l, v in acc.items()}, ntot


# ---------------------------------------------------------------- stage: P1 (partner lens, inverse binding)
# B2x asked: does the readout of Q(a,b) rank the conjunction c? P1 asks the inverse: given (a, c), which token b's
# J-vector maximises u_c.Q(v_a, v_b)? One reverse-over-forward HVP per (a, c, sample) returns a whole Hessian row
#   g = d/dh_tp [ (gamma*u_c) . J(h) (rho v_a @ tp) ]  (exact same-position block),
# scored against every token's normalised J-vector v_k = E[J_l]^T (gamma*u_k) from the fitted lens (one matmul).
# Controls: mismatched c' (another triple's conjunction), random a (matched norm), hub z-score across queries, and
# first-order baselines: b alone predicts c (v_c . v_b) and association (cos(v_a, v_b)).
if "P1" in stages:
    from torch.func import grad, jvp
    t0 = time.perf_counter()
    p1_layers = [int(x) for x in args.p1_layers.split(",")]
    JLp, ntot = load_mean_lens(args.n1_lens_dir, p1_layers)
    trip = [t for t in TRIPLES if all(w in WIDX for w in t)]
    n = len(trip)
    shift = [(i + 7) % n for i in range(n)]
    cid = torch.tensor([TID[t[2]][0] for t in trip], device="cuda")
    res = {"triples": trip, "n_ctx": args.b2x_n_ctx, "n_pos": args.b2x_n_pos, "lens_prompts": ntot, "layers": {}}
    pgen = torch.Generator().manual_seed(args.seed + 11)
    rgen = torch.Generator().manual_seed(args.seed + 12)

    def hvp_rows(f, A, Wc):
        """A [k,T,d] direction at one position, Wc [k,d] readout weights -> grad [k,T,d] of Wc . J A."""
        zero = torch.zeros_like(A)
        s = lambda d: (jvp(f, (d,), (A,))[1].float() * Wc).sum()  # noqa: E731
        with math_sdpa():
            return grad(s)(zero)

    for l in p1_layers:
        rho = STATS[l]["rho"]
        J = JLp[l]
        Vd = torch.empty(W_U.shape[0], m.d_model, device="cuda")  # token J-vectors (rows) v_k = (gamma*u_k)^T J
        for s0 in range(0, W_U.shape[0], 32768):
            Vd[s0: s0 + 32768] = (W_U[s0: s0 + 32768].float() * gamma) @ J
        vc_raw = Vd[cid].clone()  # unnormalised v_c (first-order baseline: v_c . v_b = u_c.gamma.J v_b)
        Vd /= Vd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a_dirs = VHAT[l][[WIDX[t[0]] for t in trip]]  # [n, d] unit
        r_dirs = torch.randn(n, m.d_model, generator=rgen).cuda(); r_dirs /= r_dirs.norm(dim=-1, keepdim=True)
        conds = {"Q": (a_dirs, cid), "Q_mismatch": (a_dirs, cid[shift]), "Q_rand_a": (r_dirs, cid)}
        G = {k: torch.zeros(n, m.d_model, device="cuda") for k in conds}
        per_sample_rank = {k: [[] for _ in range(n)] for k in conds}
        bvar = [variants(t[1]) for t in trip]
        cnt = 0
        for ci in range(args.b2x_n_ctx):
            ids = ctx_ids[ci]
            valid = Perturbed(m, ids, l, L).valid
            tps = valid[torch.randperm(len(valid), generator=pgen)[: args.b2x_n_pos].to(valid.device)]
            for tp in tps.tolist():
                cnt += 1
                for k, (dirs, cc) in conds.items():
                    s0, bs = 0, args.p1_batch
                    while s0 < n:
                        try:
                            dd = (dirs[s0: s0 + bs] * rho).to(dtype)
                            f = Perturbed(m, ids.expand(len(dd), -1), l, L)
                            A = at_pos(f, dd, torch.full((len(dd),), tp, device="cuda"))
                            g = hvp_rows(f, A, (W_U[cc[s0: s0 + bs]].float() * gamma).to(dtype))[:, tp].float()
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache(); bs = max(1, bs // 2); log("P1 OOM -> batch", bs); continue
                        G[k][s0: s0 + len(dd)] += g
                        sc = g @ Vd.T  # [k, V] per-sample partner scores
                        for j in range(len(dd)):
                            per_sample_rank[k][s0 + j].append(rank_of(sc[j], bvar[s0 + j]))
                        s0 += len(dd)
            log(f"P1 L{l} ctx {ci + 1}/{args.b2x_n_ctx}")
        G = {k: v / cnt for k, v in G.items()}
        P = {k: v @ Vd.T for k, v in G.items()}  # [n, V]
        Z = (P["Q"] - P["Q"].mean(0, keepdim=True)) / P["Q"].std(0, keepdim=True).clamp_min(1e-12)  # hub-corrected
        P_first = vc_raw @ Vd.T  # b alone predicts c
        P_assoc = a_dirs @ Vd.T  # cos(v_a, v_b)
        cons = torch.nn.functional.cosine_similarity(a_dirs, Vd[[TID[t[0]][0] for t in trip]], dim=-1)
        per = []
        for i, (wa, wb, wc) in enumerate(trip):
            r = {"a": wa, "b": wb, "c": wc}
            for k in conds:
                r["rank_" + k] = rank_of(P[k][i], bvar[i])
                r["median_sample_rank_" + k] = float(torch.tensor(per_sample_rank[k][i]).float().median())
            r["rank_Q_hubz"] = rank_of(Z[i], bvar[i])
            r["rank_first_order"] = rank_of(P_first[i], bvar[i])
            r["rank_assoc"] = rank_of(P_assoc[i], bvar[i])
            r["top20_Q"] = [tok.decode([int(t)]) for t in P["Q"][i].topk(20).indices]
            r["top20_Q_hubz"] = [tok.decode([int(t)]) for t in Z[i].topk(20).indices]
            r["vhat_vs_lens_cos_a"] = float(cons[i])
            per.append(r)
        top20 = torch.stack([P["Q"][i].topk(20).indices for i in range(n)])
        uniq, counts = top20.flatten().unique(return_counts=True)
        rk = [k for k in per[0] if k.startswith("rank_")]
        med = lambda xs: float(torch.tensor(xs).float().median())  # noqa: E731
        summ = {k: {"median": med([p[k] for p in per]), "top10": sum(p[k] < 10 for p in per) / n,
                    "top100": sum(p[k] < 100 for p in per) / n} for k in rk}
        summ["frac_Q_beats_first_order"] = sum(p["rank_Q"] < p["rank_first_order"] for p in per) / n
        summ["frac_Q_beats_assoc"] = sum(p["rank_Q"] < p["rank_assoc"] for p in per) / n
        summ["frac_Q_beats_mismatch"] = sum(p["rank_Q"] < p["rank_Q_mismatch"] for p in per) / n
        summ["hub_tokens_in_half_of_top20"] = [tok.decode([int(t)]) for t in uniq[counts >= n / 2]]
        summ["median_vhat_vs_lens_cos_a"] = med([p["vhat_vs_lens_cos_a"] for p in per])
        res["layers"][l] = {"rho": rho, "summary": summ, "per_triple": per}
        log(f"P1 L{l}", json.dumps(summ, ensure_ascii=False))
        dump("P1", res)
        del Vd, P, Z
        torch.cuda.empty_cache()
    res["secs"] = round(time.perf_counter() - t0, 1); dump("P1", res)


# ---------------------------------------------------------------- stage: S1 (broadcast swap, Taylor-decomposed)
# J-lens paper sec. 3.5 (flexible generalization): prompt about src (France), lens-coordinate swap src<->tgt (China)
# h <- h + lam * V(sigma(c) - c), V = [v_src v_tgt] (fitted-lens J-vectors), c = V^+ h, at every position. Per trial
# and single layer l: the swap direction D (fixed from the clean h_l) and the scalar readout
#   g(lam) = (gamma*(u_new - u_old)) . h_L,last(h_l + lam D)   (sign = sign of the true logit gap; RMSNorm > 0)
# g'(0) one jvp, g''(0) one nested jvp (all positions, cross-position terms included = exact 2nd derivative of the
# edit), actual g(lam) and true top-1 success on a lam grid; random control: same per-position norms, random fixed
# direction. Band version (paper replication): clamped swap at every layer of --s1-band, lam 1 and 2, top-1 only.
if "S1" in stages:
    from torch.func import jvp
    t0 = time.perf_counter()
    trials_src = json.load(open(os.path.join(JLENS_DIR, "data", "experiments", "flexible-generalization.json")))
    trials = []
    for cat in trials_src["categories"]:
        for fn in cat["funcs"]:
            for s_ in cat["args"]:
                for t_ in cat["args"]:
                    if s_ != t_ and fn["answers"][s_] != fn["answers"][t_]:
                        trials.append({"cat": cat["name"], "func": fn["name"], "src": s_, "tgt": t_,
                                       "prompt": fn["template"].format(arg=s_), "old": fn["answers"][s_],
                                       "new": fn["answers"][t_]})
    if args.s1_limit:
        trials = trials[:: max(1, len(trials) // args.s1_limit)][: args.s1_limit]
    s1_layers = [int(x) for x in args.s1_layers.split(",")]
    b0, b1 = (int(x) for x in args.s1_band.split(","))
    need = sorted(set(s1_layers) | set(range(b0, b1 + 1)))
    words = sorted({t["src"] for t in trials} | {t["tgt"] for t in trials})
    wtid = {w: tid(w)[0] for w in words}
    import glob
    U_args = (W_U[[wtid[w] for w in words]].float() * gamma).cpu()  # [nw, d]
    JV, ntot = {}, 0  # J-vectors of the arg tokens per layer: v_k = (gamma*u_k)^T E[J_l]
    for fp in sorted(glob.glob(f"{args.n1_lens_dir}/shard*of*.pt")):
        ck = torch.load(fp, map_location="cpu", weights_only=True, mmap=True)
        n = ck["n_prompts"]; ntot += n
        for l in need:
            x = (U_args @ ck["J"][l].float()) * n
            JV[l] = x if l not in JV else JV[l] + x
        del ck
    JV = {l: (v / ntot).cuda() for l, v in JV.items()}
    WI = {w: i for i, w in enumerate(words)}
    lams = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rgen = torch.Generator().manual_seed(args.seed + 21)
    log(f"S1: {len(trials)} trials, layers {s1_layers}, band {b0}-{b1}, lens prompts {ntot}")

    def swap_delta(h, Vl):  # h [..., d], Vl [d, 2] -> V(sigma(c) - c) = (c_t - c_s)(v_s - v_t)
        c = h.float() @ torch.linalg.pinv(Vl).T
        return (c[..., 1:2] - c[..., 0:1]) * (Vl[:, 0] - Vl[:, 1])

    def last_logits(ids, hooks=()):
        box = {}
        hs = [m.layers[l].register_forward_hook(fn) for l, fn in hooks]
        hs.append(m.layers[L].register_forward_hook(lambda _m, _i, o: box.__setitem__("h", o if torch.is_tensor(o) else o[0])))
        try:
            with torch.no_grad():
                m.forward(ids)
        finally:
            for h_ in hs:
                h_.remove()
        return (m._final_norm(box["h"][:, -1]).float() @ W_U.float().T)  # [B, V] true logits

    def band_hook(Vl, lam):
        def fn(_m, _i, out):
            h = out if torch.is_tensor(out) else out[0]
            h2 = h + (lam * swap_delta(h, Vl)).to(h.dtype)
            return h2 if torch.is_tensor(out) else (h2,) + tuple(out[1:])
        return fn

    rows = []
    for ti, tr in enumerate(trials):
        ids = m.encode(tr["prompt"], max_length=64)
        newv, oldv = variants(tr["new"]), variants(tr["old"])
        nid, oid = tid(tr["new"])[0], tid(tr["old"])[0]
        w = (gamma * (W_U[nid].float() - W_U[oid].float())).to(dtype)
        lg0 = last_logits(ids)[0]
        row = {**{k: tr[k] for k in ("cat", "func", "src", "tgt", "old", "new")}, "T": ids.shape[1],
               "clean_top1_is_old": int(lg0.argmax()) in oldv, "clean_gap": float(lg0[nid] - lg0[oid]), "layers": {}}
        Vpair = {l: torch.stack([JV[l][WI[tr["src"]]], JV[l][WI[tr["tgt"]]]], 1) for l in need}
        for lam in (1.0, 2.0):  # paper-style band-clamped swap
            lgb = last_logits(ids, [(l, band_hook(Vpair[l], lam)) for l in range(b0, b1 + 1)])[0]
            row[f"band_success{lam}"] = int(lgb.argmax()) in newv
        with ActivationRecorder(m.layers, at=s1_layers) as rec, torch.no_grad():
            m.forward(ids)
            Hl = {l: rec.activations[l][0].float() for l in s1_layers}
        for l in s1_layers:
            D = swap_delta(Hl[l], Vpair[l])  # [T, d]
            rnd = torch.randn(m.d_model, generator=rgen).cuda(); rnd /= rnd.norm()
            Dr = D.norm(dim=-1, keepdim=True) * rnd  # random control, same per-position norms
            DD = torch.stack([D, Dr]).to(dtype)  # [2, T, d]
            f = Perturbed(m, ids.expand(2, -1), l, L, reduce="last")
            zero = torch.zeros_like(DD)
            with math_sdpa():
                j1 = jvp(f, (zero,), (DD,))[1]
                inner = lambda d: jvp(f, (d,), (DD,))[1]  # noqa: E731
                q2 = jvp(inner, (zero,), (DD,))[1]
            g1, g2 = (j1.float() @ w.float()), (q2.float() @ w.float())  # [2]
            F0 = F_of(f, zero) @ w.float()
            ent = {"delta_over_rho": float(D.norm(dim=-1).median() / Hl[l].norm(dim=-1).median()),  # medians: pos-0 sink norm
                   "g1": g1.tolist(), "g2": g2.tolist(), "act": [], "success": [], "success_rand": []}
            for lam in lams:
                ent["act"].append(((F_of(f, lam * DD) @ w.float()) - F0).tolist())
                # true top-1 after the single-layer swap (real logits through the final norm)
                lgl = last_logits(ids.expand(2, -1), [(l, (lambda dd: (lambda _m, _i, o: (o + dd) if torch.is_tensor(o) else (o[0] + dd,) + tuple(o[1:])))(lam * DD))])
                ent["success"].append(int(lgl[0].argmax()) in newv)
                ent["success_rand"].append(int(lgl[1].argmax()) in newv)
            row["layers"][l] = ent
        rows.append(row)
        if (ti + 1) % 16 == 0:
            log(f"S1 {ti + 1}/{len(trials)}")
    # summaries per category x layer
    summ = {}
    cats = sorted({r["cat"] for r in rows})
    for cat in cats + ["all"]:
        rs = [r for r in rows if cat in ("all", r["cat"])]
        s_ = {"n": len(rs), "clean_top1_is_old": sum(r["clean_top1_is_old"] for r in rs) / len(rs),
              "band_success1": sum(r["band_success1.0"] for r in rs) / len(rs),
              "band_success2": sum(r["band_success2.0"] for r in rs) / len(rs), "layers": {}}
        for l in s1_layers:
            E = [r["layers"][l] for r in rs]
            med = lambda xs: float(torch.tensor(xs).float().median())  # noqa: E731
            li = {"delta_over_rho": med([e["delta_over_rho"] for e in E]),
                  "g1": med([e["g1"][0] for e in E]), "g2": med([e["g2"][0] for e in E]),
                  "g1_rand": med([e["g1"][1] for e in E]), "g2_rand": med([e["g2"][1] for e in E])}
            for k_, lam in enumerate(lams):
                act = torch.tensor([e["act"][k_][0] for e in E]); g1 = torch.tensor([e["g1"][0] for e in E])
                g2 = torch.tensor([e["g2"][0] for e in E])
                lin, quad = lam * g1, lam * g1 + 0.5 * lam ** 2 * g2
                li[f"success{lam}"] = sum(e["success"][k_] for e in E) / len(E)
                li[f"success_rand{lam}"] = sum(e["success_rand"][k_] for e in E) / len(E)
                li[f"act{lam}"] = float(act.median())
                li[f"err_lin{lam}"] = float(((act - lin).abs() / act.abs().clamp_min(1e-6)).median())
                li[f"err_quad{lam}"] = float(((act - quad).abs() / act.abs().clamp_min(1e-6)).median())
            li["frac_2nd_ge_1st_at_0.25"] = float(((0.5 * 0.25 ** 2 * torch.tensor([e["g2"][0] for e in E])).abs()
                                                  >= (0.25 * torch.tensor([e["g1"][0] for e in E])).abs()).float().mean())
            s_["layers"][l] = li
        summ[cat] = s_
        log(f"S1 {cat}", json.dumps({k: v for k, v in s_.items() if k != "layers"}),
            json.dumps({l: {k: round(v, 3) for k, v in li.items() if k in ("delta_over_rho", "g1", "g2", "success1.0", "success2.0", "err_lin0.25", "err_quad0.25")}
                        for l, li in s_["layers"].items()}))
    dump("S1", {"lams": lams, "layers": s1_layers, "band": [b0, b1], "lens_prompts": ntot, "summary": summ,
                "trials": rows, "secs": round(time.perf_counter() - t0, 1)})

# ---------------------------------------------------------------- stage: S2 (single layer vs many layers)
# R  reading. Last-position J-lens logits at every lens layer for two-hop prompts with a latent intermediate (our
#    croissant-style set + the repo's lens-eval-multihop set). Per item and window [a, b] of layers, the score is the
#    mean over the window of per-layer z-scored lens logits; window width 1 = single layer. Stored: min-over-variants
#    rank of the intermediate and of a control intermediate (another item's) for every window. Windows are chosen on
#    one half of the items and scored on the other offline (src/summarize_s2.py).
# O  one-hop S1 trials, band swaps done right. A clamp sets the pair's lens coordinates to the swapped CLEAN-pass
#    values at every band layer: h <- h + V(c*_l - V^+ h), c*_l = c_l + lam (sigma(c_l) - c_l), c_l from the clean
#    run. S1's band_hook swapped the CURRENT state, which is already swapped after the first layer, so it undid the
#    swap at every other layer; reproduced here as "reswap". Single-layer swaps on the S2 grid for comparison
#    (a one-layer clamp is exactly the additive swap lam * D).
# T  two-hop swaps: "The capital of the country famous for croissants is the city of" (latent France), France->China.
#    Single layers (lam grid; random control with the same per-position norms; exact r'(0), r''(0)), band clamps (lam
#    grid; exact jets), a static spread of the clean swap over the band (h_l += (lam/|band|) D_l), an irrelevant-pair
#    control (swap two countries that are not the latent one), and an answer swap (Paris<->Beijing) per layer.
#    Readout r = true logit gap new - old at the last position, through the final norm. Push size per layer
#    beta_l = |Delta_l| / |h_l| (Frobenius over positions 1..T-1, clean h); budgets L1 = sum beta_l, L2 = rms-sum.
S2_COUNTRIES = {  # clues avoid naming the country, its capital or its language
    "France": (["the country famous for croissants", "the country where the Eiffel Tower stands"],
               ["Paris"], ["French"], ["Europe"]),
    "China": (["the country famous for the Great Wall", "the country where giant pandas live in the wild"],
              ["Beijing"], ["Chinese", "Mandarin"], ["Asia"]),
    "Japan": (["the country famous for sushi", "the country where Mount Fuji stands"],
              ["Tokyo"], ["Japanese"], ["Asia"]),
    "Egypt": (["the country famous for the pyramids", "the country where the Sphinx stands"],
              ["Cairo"], ["Arabic"], ["Africa"]),
    "Italy": (["the country famous for pizza", "the country where the Leaning Tower of Pisa stands"],
              ["Rome"], ["Italian"], ["Europe"]),
    "Germany": (["the country famous for Oktoberfest", "the country where the Black Forest is"],
                ["Berlin"], ["German"], ["Europe"]),
    "Spain": (["the country famous for flamenco", "the country famous for paella"],
              ["Madrid"], ["Spanish"], ["Europe"]),
    "Canada": (["the country famous for maple syrup", "the country with a maple leaf on its flag"],
               ["Ottawa"], ["English"], ["North"]),
}
S2_FUNCS = {"capital": ("Fact: The capital of {clue} is the city of", 1),
            "language": ("Fact: Most people in {clue} speak", 2),
            "continent": ("Fact: {Clue} lies on the continent of", 3)}

if "S2" in stages:
    from torch.func import jvp
    import glob
    t0 = time.perf_counter()
    parts = args.s2_parts.split(",")
    s2_layers = [int(x) for x in args.s2_layers.split(",")]
    bands = [tuple(int(y) for y in b.split("-")) for b in args.s2_bands.split(",")]
    band_layers = sorted({l for a, b in bands for l in range(a, b + 1)})
    need = sorted(set(s2_layers) | set(band_layers))
    LAMS = [float(x) for x in args.s2_lams.split(",")] if args.s2_lams else [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    BAND_LAMS = [float(x) for x in args.s2_band_lams.split(",")] if args.s2_band_lams else [0.25, 0.5, 1.0, 2.0]
    assert 1.0 in LAMS and 2.0 in LAMS and 1.0 in BAND_LAMS, "--s2-lams needs 1 and 2, --s2-band-lams needs 1"
    SPREAD_LAMS = [1.0, 2.0, 4.0, 8.0, 16.0]
    CN = list(S2_COUNTRIES)
    th_prompt = lambda fn, clue: S2_FUNCS[fn][0].format(clue=clue, Clue=clue[0].upper() + clue[1:])  # noqa: E731
    ans = lambda c, fn: S2_COUNTRIES[c][S2_FUNCS[fn][1]]  # noqa: E731
    vset = lambda ws: sorted({x for w in ws for x in variants(w)})  # noqa: E731
    res = {"layers": s2_layers, "bands": bands, "lams": LAMS, "band_lams": BAND_LAMS, "spread_lams": SPREAD_LAMS,
           "parts": parts, "countries": {c: list(v) for c, v in S2_COUNTRIES.items()},
           "funcs": {k: v[0] for k, v in S2_FUNCS.items()}}
    over = lambda: (time.perf_counter() - t0) / 60 > args.s2_max_min  # noqa: E731

    # --- items: reading set, one-hop trials (S1's), two-hop trials
    read_items = [{"set": "twohop", "name": f"{c}-{fn}-{ci}", "prompt": th_prompt(fn, clue), "inter": [c]}
                  for c in CN for fn in S2_FUNCS for ci, clue in enumerate(S2_COUNTRIES[c][0])]
    read_items += [{"set": "multihop", "name": it["name"], "prompt": it["prompt"], "inter": it["intermediates"]}
                   for it in json.load(open(EVAL_DIR / "lens-eval-multihop.json"))["items"]]
    for s_ in ("twohop", "multihop"):  # control = the next item's intermediate that shares no word with this one
        idx = [i for i, it in enumerate(read_items) if it["set"] == s_]
        for k, i in enumerate(idx):
            mine = {w.lower() for w in read_items[i]["inter"]}
            for off in range(1, len(idx)):
                j = idx[(k + off) % len(idx)]
                if not mine & {w.lower() for w in read_items[j]["inter"]}:
                    read_items[i]["ctrl"] = read_items[j]["inter"]
                    break
    fg = json.load(open(os.path.join(JLENS_DIR, "data", "experiments", "flexible-generalization.json")))
    one = [{"cat": cat["name"], "func": fn["name"], "src": s_, "tgt": t_, "prompt": fn["template"].format(arg=s_),
            "old": [fn["answers"][s_]], "new": [fn["answers"][t_]]}
           for cat in fg["categories"] for fn in cat["funcs"] for s_ in cat["args"] for t_ in cat["args"]
           if s_ != t_ and fn["answers"][s_] != fn["answers"][t_]]
    two = []
    for fn in S2_FUNCS:
        for s_ in CN:
            for t_ in CN:
                if s_ == t_ or set(ans(s_, fn)) & set(ans(t_, fn)):
                    continue
                z = next(CN[(CN.index(t_) + k) % len(CN)] for k in range(1, len(CN))
                         if CN[(CN.index(t_) + k) % len(CN)] not in (s_, t_))
                two.append({"func": fn, "src": s_, "tgt": t_, "ctrl": z, "prompt": th_prompt(fn, S2_COUNTRIES[s_][0][0]),
                            "old": ans(s_, fn), "new": ans(t_, fn)})
    si, sn = (int(x) for x in args.s2_shard.split("/"))
    one, two = one[si::sn], two[si::sn]
    res["shard"] = args.s2_shard
    if args.s2_limit:
        one = one[:: max(1, len(one) // args.s2_limit)][: args.s2_limit]
        two = two[:: max(1, len(two) // args.s2_limit)][: args.s2_limit]
    words = sorted(set(CN) | {t["src"] for t in one} | {t["tgt"] for t in one}
                   | {t["old"][0] for t in two} | {t["new"][0] for t in two})
    wtid = {w: tid(w)[0] for w in words}
    WI = {w: i for i, w in enumerate(words)}
    files = sorted(glob.glob(f"{args.n1_lens_dir}/shard*of*.pt"))
    lens_layers = sorted(torch.load(files[0], map_location="cpu", weights_only=True, mmap=True)["J"].keys())
    log(f"S2: parts {parts}; read {len(read_items)} items; one-hop {len(one)}; two-hop {len(two)}; layers {s2_layers}; "
        f"bands {bands}; lens layers {lens_layers[0]}..{lens_layers[-1]}")

    # --- last-position residuals of the reading items at every lens layer
    if "R" in parts:
        Hread = {l: [] for l in lens_layers}
        with torch.no_grad(), ActivationRecorder(m.layers, at=lens_layers) as rec:
            for it in read_items:
                m.forward(m.encode(it["prompt"], max_length=256))
                for l in lens_layers:
                    Hread[l].append(rec.activations[l][0, -1].float())
        Hread = {l: torch.stack(v) for l, v in Hread.items()}
    # --- one pass over the lens shards: J-vectors of all words (every layer) and J_l h_l of the reading items
    W_Uf = W_U.float()  # one fp32 copy (mixed mode would otherwise upcast 248k x d on every readout)
    U_w = W_Uf[[wtid[w] for w in words]] * gamma
    JV, JH, ntot = {}, {}, 0
    for fp in files:
        ck = torch.load(fp, map_location="cpu", weights_only=True, mmap=True)
        n = ck["n_prompts"]; ntot += n
        for l in lens_layers:
            Jl = ck["J"][l].cuda().float()
            JV[l] = JV.get(l, 0) + (U_w @ Jl) * n
            if "R" in parts:
                JH[l] = JH.get(l, 0) + (Hread[l] @ Jl.T) * n
            del Jl
        del ck
        log(f"S2 lens shard {Path(fp).name} done")
    JV = {l: v / ntot for l, v in JV.items()}
    JH = {l: v / ntot for l, v in JH.items()}
    res["lens_prompts"] = ntot

    # --- R: ranks for every window
    if "R" in parts:
        nl = len(lens_layers)
        wins = [(i, i) for i in range(nl)]
        for wd in (2, 4, 8, 16, 32):
            wins += [(s, s + wd - 1) for s in range(0, nl - wd + 1, 2)]
        wins.append((0, nl - 1))
        Si = torch.tensor([a for a, _ in wins], device="cuda"); Ei = torch.tensor([b for _, b in wins], device="cuda")
        Wn = (Ei - Si + 1).float()[:, None]

        def win_ranks(Sc, ids_):
            best = None
            for j in ids_:
                r_ = (Sc > Sc[:, j: j + 1]).sum(1)
                best = r_ if best is None else torch.minimum(best, r_)
            return best.tolist()

        rr = []
        for i, it in enumerate(read_items):
            Z = readout(torch.stack([JH[l][i] for l in lens_layers]), W_Uf, gamma)  # [nl, V]
            Z = (Z - Z.mean(1, keepdim=True)) / Z.std(1, keepdim=True)
            P = torch.cat([torch.zeros_like(Z[:1]), Z.cumsum(0)])
            Sc = (P[Ei + 1] - P[Si]) / Wn
            rr.append({"set": it["set"], "name": it["name"], "inter": it["inter"], "ctrl": it["ctrl"],
                       "rank": win_ranks(Sc, vset(it["inter"])), "rank_ctrl": win_ranks(Sc, vset(it["ctrl"]))})
            del Z, P, Sc
        res["read"] = {"lens_layers": lens_layers, "windows": wins, "items": rr}
        for s_ in ("twohop", "multihop"):
            R_ = [r for r in rr if r["set"] == s_]
            top10 = lambda k: sum(r["rank"][k] < 10 for r in R_) / len(R_)  # noqa: E731
            bl = max(range(nl), key=top10)
            log(f"S2 R {s_}: n={len(R_)} best single layer {lens_layers[bl]} top10 {top10(bl):.2f}; all-layer mean "
                f"top10 {top10(len(wins) - 1):.2f} (in-sample, for the log only)")
        dump("S2", res)
        del Hread, JH
        torch.cuda.empty_cache()

    # --- helpers for the swaps
    def run_last(ids, hooks):
        """Run with (layer, fn(h) -> h) hooks; returns final-normed last-position residual [B, d]."""
        box = {}

        def mk(fn):
            def hook(_m, _i, out):
                h = out if torch.is_tensor(out) else out[0]
                h2 = fn(h)
                return h2 if torch.is_tensor(out) else (h2,) + tuple(out[1:])
            return hook
        hs = [m.layers[l].register_forward_hook(mk(fn)) for l, fn in hooks]
        hs.append(m.layers[L].register_forward_hook(lambda _m, _i, o: box.__setitem__("h", o if torch.is_tensor(o) else o[0])))
        try:
            m.forward(ids)
        finally:
            for h_ in hs:
                h_.remove()
        return m._final_norm(box["h"][:, -1])

    def logits_of(ids, hooks):
        with torch.no_grad():
            return run_last(ids, hooks).float() @ W_Uf.T

    def jets(f):
        z = torch.zeros((), device="cuda", dtype=dtype); o = torch.ones_like(z)
        with math_sdpa():
            g1 = jvp(f, (z,), (o,))[1]
            g2 = jvp(lambda x: jvp(f, (x,), (o,))[1], (z,), (o,))[1]
        return float(g1), float(g2)

    def summ(lg, newv, oldv, nid, oid, top=True):
        lp = torch.log_softmax(lg, -1)
        t5 = lg.topk(5).indices.tolist()
        e = {"succ": t5[0] in newv, "keep": t5[0] in oldv, "r": round(float(lg[nid] - lg[oid]), 4),
             "p_new": round(float(lp[newv].exp().sum()), 5), "p_old": round(float(lp[oldv].exp().sum()), 5),
             "rank_new": rank_of(lg, newv)}
        if top:
            e["top5"] = [tok.decode([t]) for t in t5]
        return e

    def Vpair(a, b, l):
        return torch.stack([JV[l][WI[a]], JV[l][WI[b]]], 1)  # [d, 2]

    def clean_states(ids, layers):
        with torch.no_grad(), ActivationRecorder(m.layers, at=layers) as rec:
            m.forward(ids)
            return {l: rec.activations[l][0].float() for l in layers}

    class Pair:
        """Lens pair (a, b) at a set of layers, with clean coordinates of one prompt."""
        def __init__(self, a, b, layers, Hc):
            self.V = {l: Vpair(a, b, l) for l in layers}
            self.P = {l: torch.linalg.pinv(self.V[l]) for l in layers}
            self.c = {l: Hc[l] @ self.P[l].T for l in layers}  # [T, 2]
            self.D = {l: (self.c[l].flip(-1) - self.c[l]) @ self.V[l].T for l in layers}  # clean swap [T, d]
            self.hn = {l: float(Hc[l][1:].norm()) for l in layers}
            self.beta = {l: float(self.D[l][1:].norm()) / self.hn[l] for l in layers}

        def clamp(self, layers, lam, acc=None):
            def mk(l):
                def fn(h):
                    tgt = self.c[l] + lam * (self.c[l].flip(-1) - self.c[l])
                    d = (tgt - h.float() @ self.P[l].T) @ self.V[l].T
                    if acc is not None:
                        acc.append(float(d[0, 1:].norm()) / self.hn[l])
                    return h + d.to(h.dtype)
                return fn
            return [(l, mk(l)) for l in layers]

        def reswap(self, layers, lam):  # S1's band_hook: swaps the current (already swapped) state
            def mk(l):
                def fn(h):
                    c = h.float() @ self.P[l].T
                    return h + (lam * (c.flip(-1) - c) @ self.V[l].T).to(h.dtype)
                return fn
            return [(l, mk(l)) for l in layers]

        def add(self, layers, mu):
            return [(l, (lambda dd: (lambda h: h + dd))((mu * self.D[l]).to(dtype))) for l in layers]

    def budget(acc):
        return {"L1": round(sum(acc), 5), "L2": round(math.sqrt(sum(a * a for a in acc)), 5), "max": round(max(acc), 5)}

    rgen2 = torch.Generator().manual_seed(args.seed + 31)

    # --- O: one-hop band clamp vs reswap vs single layers
    if "O" in parts:
        rows = []
        for ti, tr in enumerate(one):
            if over():
                log("S2 O: time budget reached"); break
            ids = m.encode(tr["prompt"], max_length=64)
            newv, oldv = vset(tr["new"]), vset(tr["old"])
            nid, oid = tid(tr["new"][0])[0], tid(tr["old"][0])[0]
            lg0 = logits_of(ids, [])[0]
            Hc = clean_states(ids, need)
            pr = Pair(tr["src"], tr["tgt"], need, Hc)
            row = {**{k: tr[k] for k in ("cat", "func", "src", "tgt", "old", "new")}, "clean": summ(lg0, newv, oldv, nid, oid),
                   "single": {}, "band": {}, "reswap": {}}
            for l in s2_layers:
                row["single"][l] = {"beta": round(pr.beta[l], 5),
                                    "out": [summ(logits_of(ids, pr.add([l], lam))[0], newv, oldv, nid, oid, top=False) for lam in (1.0, 2.0)]}
            for (a, b) in bands:
                ent = {"out": [], "budget": []}
                for lam in (1.0, 2.0):
                    acc = []
                    ent["out"].append(summ(logits_of(ids, pr.clamp(range(a, b + 1), lam, acc))[0], newv, oldv, nid, oid, top=False))
                    ent["budget"].append(budget(acc))
                row["band"][f"{a}-{b}"] = ent
            a, b = bands[-1]
            row["reswap"][f"{a}-{b}"] = [summ(logits_of(ids, pr.reswap(range(a, b + 1), lam))[0], newv, oldv, nid, oid, top=False)
                                        for lam in (1.0, 2.0)]
            rows.append(row)
            if (ti + 1) % 32 == 0:
                log(f"S2 O {ti + 1}/{len(one)}")
                res["onehop"] = rows; dump("S2", res)
        res["onehop"] = rows
        for cat in sorted({r["cat"] for r in rows}) + ["all"]:
            rs = [r for r in rows if cat in ("all", r["cat"])]
            fr = lambda xs: round(sum(xs) / max(1, len(xs)), 3)  # noqa: E731
            log(f"S2 O {cat} n={len(rs)} clean_ok {fr([r['clean']['keep'] for r in rs])} | single@1 "
                + " ".join(f"L{l}:{fr([r['single'][l]['out'][0]['succ'] for r in rs])}" for l in s2_layers)
                + " | band clamp@1 " + " ".join(f"{k}:{fr([r['band'][k]['out'][0]['succ'] for r in rs])}" for k in rs[0]["band"])
                + " | reswap@1 " + " ".join(f"{k}:{fr([r['reswap'][k][0]['succ'] for r in rs])}" for k in rs[0]["reswap"]))
        dump("S2", res)

    # --- T: two-hop swaps
    if "T" in parts:
        rows = []
        checked = False
        for ti, tr in enumerate(two):
            if over():
                log("S2 T: time budget reached"); break
            ids = m.encode(tr["prompt"], max_length=64)
            newv, oldv = vset(tr["new"]), vset(tr["old"])
            nid, oid = tid(tr["new"][0])[0], tid(tr["old"][0])[0]
            wdiff = (W_Uf[nid] - W_Uf[oid])
            lg0 = logits_of(ids, [])[0]
            row = {**{k: tr[k] for k in ("func", "src", "tgt", "ctrl", "prompt", "old", "new")}, "T": ids.shape[1],
                   "clean": summ(lg0, newv, oldv, nid, oid)}
            if not row["clean"]["keep"]:  # the model must answer the two-hop prompt before a swap can redirect it
                rows.append(row)
                continue
            Hc = clean_states(ids, need)
            pr = Pair(tr["src"], tr["tgt"], need, Hc)
            pz = Pair(tr["ctrl"], tr["tgt"], band_layers, Hc)
            pa = Pair(tr["old"][0], tr["new"][0], s2_layers, Hc)
            rfun = lambda hooks: (run_last(ids, hooks).float() @ wdiff)[0]  # noqa: E731
            row["single"], row["band"] = {}, {}
            for l in s2_layers:
                rnd = torch.randn(m.d_model, generator=rgen2).cuda(); rnd /= rnd.norm()
                DD = torch.stack([pr.D[l], pr.D[l].norm(dim=-1, keepdim=True) * rnd]).to(dtype)  # swap, random control
                ent = {"beta": round(pr.beta[l], 5), "out": [], "rand": []}
                for lam in LAMS:
                    lg = logits_of(ids.expand(2, -1), [(l, (lambda dd: (lambda h: h + dd))(lam * DD))])
                    ent["out"].append(summ(lg[0], newv, oldv, nid, oid))
                    ent["rand"].append(summ(lg[1], newv, oldv, nid, oid, top=False))
                ent["g1"], ent["g2"] = jets(lambda lam: rfun([(l, (lambda h: h + lam * pr.D[l].to(dtype)))]))
                ent["answer_swap"] = [summ(logits_of(ids, pa.add([l], lam))[0], newv, oldv, nid, oid, top=False) for lam in (1.0, 2.0)]
                row["single"][l] = ent
            for (a, b) in bands:
                Ls = range(a, b + 1)
                ent = {"out": [], "budget": [], "spread": [], "spread_budget": [], "ctrl": []}
                for lam in BAND_LAMS:
                    acc = []
                    ent["out"].append(summ(logits_of(ids, pr.clamp(Ls, lam, acc))[0], newv, oldv, nid, oid))
                    ent["budget"].append(budget(acc))
                ent["g1"], ent["g2"] = jets(lambda lam: rfun(pr.clamp(Ls, lam)))
                for lam in SPREAD_LAMS:
                    mu = lam / len(Ls)
                    ent["spread"].append(summ(logits_of(ids, pr.add(Ls, mu))[0], newv, oldv, nid, oid))
                    ent["spread_budget"].append(budget([mu * pr.beta[l] for l in Ls]))
                for lam in (1.0, 2.0):
                    ent["ctrl"].append(summ(logits_of(ids, pz.clamp(Ls, lam))[0], newv, oldv, nid, oid, top=False))
                row["band"][f"{a}-{b}"] = ent
            if not checked:  # exactness gate on the first valid trial (the smoke job fails here if the jets are wrong)
                checked = True
                lc = s2_layers[len(s2_layers) // 2]
                (a, b) = bands[len(bands) // 2]
                for name, mkh, g in (("single", lambda lam: [(lc, (lambda h: h + lam * pr.D[lc].to(dtype)))],
                                      (row["single"][lc]["g1"], row["single"][lc]["g2"])),
                                     ("band", lambda lam: pr.clamp(range(a, b + 1), lam), (row["band"][f"{a}-{b}"]["g1"], row["band"][f"{a}-{b}"]["g2"]))):
                    hh = 0.02
                    with torch.no_grad():
                        rp, r0, rm = (float(rfun(mkh(torch.tensor(x, device="cuda", dtype=dtype)))) for x in (hh, 0.0, -hh))
                    fd1, fd2 = (rp - rm) / (2 * hh), (rp - 2 * r0 + rm) / hh ** 2
                    log(f"S2 T gate {name}: g1 {g[0]:.4f} fd {fd1:.4f} | g2 {g[1]:.4f} fd {fd2:.4f} | r0 {r0:.4f} clean {row['clean']['r']}")
                    assert abs(g[0] - fd1) <= 0.05 * abs(fd1) + 0.05, f"S2 gate {name}: first derivative mismatch"
                    assert abs(g[1] - fd2) <= 0.25 * abs(fd2) + 0.5, f"S2 gate {name}: second derivative mismatch"
                    assert abs(r0 - row["clean"]["r"]) <= 1e-2 * (1 + abs(r0)), f"S2 gate {name}: lam=0 is not the clean run"
                # a one-layer clamp must equal the additive swap
                r_add = float(rfun(pr.add([lc], 1.0))); r_cl = float(rfun(pr.clamp([lc], 1.0)))
                log(f"S2 T gate one-layer clamp {r_cl:.4f} vs additive {r_add:.4f}")
                assert abs(r_add - r_cl) <= 1e-2 * (1 + abs(r_add)), "S2 gate: one-layer clamp != additive swap"
            rows.append(row)
            if (ti + 1) % 8 == 0:
                log(f"S2 T {ti + 1}/{len(two)} ({(time.perf_counter() - t0) / 60:.1f} min)")
                res["twohop"] = rows; dump("S2", res)
        res["twohop"] = rows
        ok = [r for r in rows if "single" in r]
        fr = lambda xs: round(sum(xs) / max(1, len(xs)), 3)  # noqa: E731
        log(f"S2 T n={len(rows)} clean_ok {len(ok)}")
        if ok:
            li = LAMS.index(1.0); bi = BAND_LAMS.index(1.0)
            log("S2 T single@1 " + " ".join(f"L{l}:{fr([r['single'][l]['out'][li]['succ'] for r in ok])}" for l in s2_layers)
                + " | rand@1 " + str(fr([r["single"][l]["rand"][li]["succ"] for r in ok for l in s2_layers]))
                + " | band@1 " + " ".join(f"{k}:{fr([r['band'][k]['out'][bi]['succ'] for r in ok])}" for k in ok[0]["band"])
                + " | ctrl@1 " + " ".join(f"{k}:{fr([r['band'][k]['ctrl'][0]['succ'] for r in ok])}" for k in ok[0]["band"]))
    res["secs"] = round(time.perf_counter() - t0, 1)
    dump("S2", res)

# ---------------------------------------------------------------- stage: B3 (quadratic lens on eval prompts)
if "B3" in stages:
    t0 = time.perf_counter()
    b3_layers = [int(x) for x in args.b3_layers.split(",")]
    res = {"layers": b3_layers, "sets": {}}
    ks = [int(round(10 ** (i / 20))) for i in range(61)]  # 1..1000 log-spaced

    def auc(ranks):  # ranks: min-over-layers 0-based ranks, one per intermediate
        pk = [sum(r < k for r in ranks) / len(ranks) for k in ks]
        return sum((pk[i] + pk[i + 1]) / 2 * (math.log(ks[i + 1]) - math.log(ks[i])) for i in range(len(ks) - 1)) / math.log(1000), pk

    for name in args.b3_sets.split(","):
        items = json.load(open(EVAL_DIR / f"lens-eval-{name}.json"))["items"]
        if args.b3_limit:
            items = items[: args.b3_limit]
        lens_names = ["logit", "J_h", "J_d", "Q_dd", "J_d+Q/2"]
        best = {k: [] for k in lens_names}  # one min-over-layers rank per (item, intermediate)
        per_item = []
        # residuals at the readout position for every item and layer
        H = {l: [] for l in b3_layers}
        for it in items:
            ids = m.encode(it["prompt"], max_length=512)
            pos = ids.shape[1] - 1
            if name == "poetry":
                toks = [tok.decode([int(t)]) for t in ids[0]]
                nl = [i for i, s in enumerate(toks) if "\n" in s]
                pos = nl[-1] if nl else pos
            with torch.no_grad(), ActivationRecorder(m.layers, at=b3_layers) as rec:
                m.forward(ids)
            for l in b3_layers:
                H[l].append(rec.activations[l][0, pos].float())
        ranks = {k: [[math.inf] * len(it["intermediates"]) for it in items] for k in lens_names}
        for l in b3_layers:
            h = torch.stack(H[l])
            d = h - STATS[l]["mu"]
            Jd, _, Qdd = batched_Q(l, d, None, signs=True, n_ctx=args.b3_n_ctx, exact_pos=args.b3_exact_pos)
            Jmu = batched_Q(l, STATS[l]["mu"][None], None, signs=True, n_ctx=args.b3_n_ctx, second=False)[0]
            outs = {"logit": readout(h, W_U, gamma), "J_h": readout(Jd + Jmu, W_U, gamma),
                    "J_d": readout(Jd, W_U, gamma), "Q_dd": readout(Qdd, W_U, gamma),
                    "J_d+Q/2": readout(Jd + 0.5 * Qdd, W_U, gamma)}
            for i, it in enumerate(items):
                for j, wd in enumerate(it["intermediates"]):
                    cids = variants(wd, name == "order-ops")
                    for k in lens_names:
                        ranks[k][i][j] = min(ranks[k][i][j], rank_of(outs[k][i], cids))
            log(f"B3 {name} L{l} done")
        flat = {k: [r for row in ranks[k] for r in row] for k in lens_names}
        summ = {}
        for k in lens_names:
            a_, pk = auc(flat[k])
            summ[k] = {"auc": round(a_, 3), "pass@1": pk[0], "pass@10": pk[20], "pass@100": pk[40]}
        res["sets"][name] = {"n_items": len(items), "summary": summ,
                             "per_item": [{"name": it["name"], **{k: ranks[k][i] for k in lens_names}}
                                          for i, it in enumerate(items)]}
        log(f"B3 {name}", json.dumps(summ))
        dump(args.b3_name, res)
    res["secs"] = round(time.perf_counter() - t0, 1); dump(args.b3_name, res)

log("ALL DONE", stages)
