"""CPU tests for src/q_robust.py on a tiny random Qwen3 (float64): the tail map equals the full model, the jets
equal finite differences and so_core's, the ratio, the SVD / sharp / ascent finders, and the behavioural metrics
and dose-response statistics on toy data."""
import math
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
import q_robust as QR  # noqa: E402
from so_core import Perturbed, first_and_second, math_sdpa  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
T = 24


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import jlens
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for p in hf.parameters():
            if p.dim() == 2:
                p.mul_(8.0)  # properly nonlinear blocks

    def _rms64(self, x):  # HF's RMSNorm casts to fp32; keep float64 so references are exact
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    for mod in hf.modules():
        if type(mod).__name__.endswith("RMSNorm"):
            mod.forward = types.MethodType(_rms64, mod)
    m = jlens.from_hf(hf, types.SimpleNamespace(bos_token_id=None))
    g = torch.Generator().manual_seed(1)
    ids = [torch.randint(0, 97, (1, T), generator=g) for _ in range(3)]
    return m, hf, ids


def _setup(l=1):
    m, hf, ids = _tiny()
    valid = torch.arange(16, T - 1)
    H = torch.cat([QR.capture(m, i, [l])[l] for i in ids])
    rho = float(H[:, valid].norm(dim=-1).mean())
    tail = QR.Tail(m, l, T)
    maps = [QR.Map(tail, H[i: i + 1], valid, valid, ids[i][0, valid + 1]) for i in range(len(ids))]
    return m, hf, ids, valid, H, rho, tail, maps


def test_tail_jets_ratio():
    m, hf, ids, valid, H, rho, tail, maps = _setup()
    L = m.n_layers - 1
    # tail == hooked full model (logits) and == so_core.Perturbed (residual jets)
    full = m._lm_head(m._final_norm(QR.capture(m, ids[0], [L])[L][:, valid]))
    assert float((tail.logits(H[:1], valid) - full).norm() / full.norm()) < 1e-12
    u = torch.randn(2, 32, generator=torch.Generator().manual_seed(2))
    U = rho * u / u.norm(dim=-1, keepdim=True)
    f = Perturbed(m, ids[0].expand(2, -1), 1, L)
    ja, _, q = first_and_second(f, U, None, signs=False)
    G = maps[0]
    zero = torch.zeros_like(U)
    with math_sdpa():
        (_, rJ), (_, rQ) = torch.func.jvp(lambda v: torch.func.jvp(G.resid, (v,), (U,)), (zero,), (U,))
    nV = len(valid)
    assert float((rJ - ja * nV).norm() / rJ.norm()) < 1e-10
    assert float((rQ - q * nV).norm() / rQ.norm()) < 1e-10
    # logit jets vs central finite differences
    z0, J, Q = QR.jets(G, U)
    eps = 1e-4
    Gp, Gm, G0 = G(eps * U), G(-eps * U), G(zero)
    assert float(((Gp - Gm) / (2 * eps) - J).norm() / J.norm()) < 1e-6
    assert float(((Gp - 2 * G0 + Gm) / eps ** 2 - Q).norm() / Q.norm()) < 1e-4
    assert float(z0.mean(-1).abs().max()) < 1e-12  # centred
    # the ratio is sqrt(sum |Q|^2 / sum |J|^2) over contexts
    sc = QR.score(maps, U / rho, rho, bs=2)
    r = QR.ratio_of(sc)
    man = torch.stack([torch.stack([QR.jets(Gc, U[k: k + 1])[2].pow(2).sum() for Gc in maps]).sum()
                       / torch.stack([QR.jets(Gc, U[k: k + 1])[1].pow(2).sum() for Gc in maps]).sum()
                       for k in range(2)]).sqrt()
    assert torch.allclose(r.double(), man, rtol=1e-5)
    # Taylor stats: at a tiny push the J-only KL equals the Fisher quadratic form
    st = QR.taylor_stats(z0, J, Q, alphas=(1e-3,), signs=(1.0,))
    assert torch.allclose(st["kl_lin"][:, 0, 0].double(), 0.5 * 1e-6 * st["fisher"].double(), rtol=1e-2)


def test_svd_sharp_and_ascent():
    m, hf, ids, valid, H, rho, tail, maps = _setup()
    d = 32
    # SVD: matches the exact top singular vector of the stacked Jacobian
    with math_sdpa():
        Jm = torch.cat([torch.func.jacfwd(lambda x, G=G: G(x[None])[0].flatten())(torch.zeros(d)) for G in maps])
    S = torch.linalg.svdvals(Jm)
    V, ev, hist = QR.subspace_top(QR.JtJ_apply(maps, 4), d, 2, 4, 6, torch.Generator().manual_seed(3), "cpu",
                                  torch.float64)
    assert abs(float(ev[0]) / float(S[0] ** 2) - 1) < 1e-3
    assert float((Jm @ V[0]).norm() / S[0]) > 0.999
    # sharp: Hessian-vector product of the loss matches finite differences of its gradient
    G = maps[0]
    x = torch.randn(1, d, generator=torch.Generator().manual_seed(4))
    with math_sdpa():
        gp = torch.func.grad(G.loss)(1e-5 * x)
        gm = torch.func.grad(G.loss)(-1e-5 * x)
    hv = QR.hvp_loss(G, x)
    assert float(((gp - gm) / 2e-5 - hv).norm() / hv.norm()) < 1e-5
    # ascent: analytic gradient == finite differences of psi (both modes), and the ratio goes up
    u0 = torch.randn(3, d, generator=torch.Generator().manual_seed(5))
    u0 = u0 / u0.norm(dim=-1, keepdim=True)
    e = torch.randn(3, d, generator=torch.Generator().manual_seed(6))
    tgt = torch.full((3,), 1.0)
    for mode in ("maxr", "matched"):
        psi, N, D, Qs = QR.opt_eval(maps, u0, rho, mode, tgt, 2.0)
        g = QR.opt_grad(maps, u0, rho, mode, N, D, Qs, tgt, 2.0)
        h = 1e-5
        fd = (QR.opt_eval(maps, u0 + h * e, rho, mode, tgt, 2.0)[0]
              - QR.opt_eval(maps, u0 - h * e, rho, mode, tgt, 2.0)[0]) / (2 * h)
        assert torch.allclose((g * e).sum(-1), fd, rtol=1e-5, atol=1e-8), (mode, (g * e).sum(-1), fd)
    u, hist = QR.sphere_opt(maps, u0, rho, 6, "maxr")
    r0, rbest = hist[0]["ratio"], torch.stack([x["ratio"] for x in hist]).amax(0)
    assert bool((rbest > r0 * 1.05).all()), (r0, rbest)
    assert torch.allclose(QR.ratio_of(QR.score(maps, u, rho, 3)).double(), rbest.double(), rtol=1e-5)
    # matched mode moves the first-order size toward the target
    D0 = hist[0]["D"].double()
    tgt = D0.log() + math.log(2.0)
    uc, hc = QR.sphere_opt(maps, u0, rho, 6, "matched", logD_target=tgt, lam=5.0)
    Dc = QR.score(maps, uc, rho, 3)["J2"].sum(1).double()
    assert float((Dc.log() - tgt).abs().max()) < 0.9 * math.log(2.0), (Dc.log(), tgt)


def test_behaviour_metrics():
    m, hf, ids, valid, H, rho, tail, maps = _setup()
    d = 32
    u = torch.randn(2, d, generator=torch.Generator().manual_seed(7))
    u = u / u.norm(dim=-1, keepdim=True)
    deltas = torch.cat([torch.zeros(1, d), rho * 0.01 * u, rho * 2.0 * u])
    KL, FL = QR.kl_flip(tail, H, valid, valid, deltas, max_seqs=4)
    assert KL.shape == (5, 3) and float(KL[0].abs().max()) < 1e-12 and float(FL[0].max()) == 0
    assert bool((KL[3:] > KL[1:3] * 10).all())
    # KL at a small push agrees with the Taylor prediction from the jets
    z0, J, Q = QR.jets(maps[0], rho * u)
    st = QR.taylor_stats(z0, J, Q, alphas=(0.01,), signs=(1.0,))
    assert torch.allclose(KL[1:3, 0], st["kl_quad"][:, 0, 0].double(), rtol=2e-2)
    # facts: pushing at 16..T-1 of a 24-token prompt; zero push keeps the clean answers
    top = m.unembed(QR.capture(m, ids[0], [m.n_layers - 1])[m.n_layers - 1][:, -1]).argmax(-1)
    ans = torch.tensor([[int(top), -1]])
    C_, K_, D_ = QR.facts_eval(tail, H[:1], ans, deltas, max_tokens=48)
    assert bool(C_[0, 0]) and float(K_[0].abs().max()) < 1e-12 and float(D_[0].abs().max()) < 1e-12
    # greedy with a zero push equals HF generate; continuation NLL is finite
    hf.generation_config.eos_token_id = None
    hf.generation_config.pad_token_id = 0
    ref = hf.generate(ids[0], max_new_tokens=5, do_sample=False)[:, T:]
    gz = QR.greedy(hf, m, 1, ids[0], torch.zeros(1, d), 5)
    assert torch.equal(gz, ref)
    gp = QR.greedy(hf, m, 1, ids[0], rho * 3.0 * u[:1], 5)
    assert gp.shape == (1, 5) and torch.isfinite(QR.cont_nll(hf, ids[0], gp)).all()
    # pushing only from position >= T (generated tokens) leaves the first generated token unchanged
    late = QR.greedy(hf, m, 1, ids[0], rho * 3.0 * u[:1], 5, push_from=T)
    assert int(late[0, 0]) == int(ref[0, 0])
    # edit distance
    assert QR.edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert QR.edit_distance([1, 2, 3], [1, 3]) == 1
    assert QR.edit_distance([1, 2, 3, 4], [4, 3, 2, 1]) == 4
    assert QR.edit_distance([], [5, 6]) == 2


def test_dose_response_stats():
    al = np.array(QR.ALPHAS)
    y = 2.0 * al ** 2  # power law, slope 2, zero curvature
    s = QR.loglog_slopes(al, y)
    assert np.allclose(s, 2.0) and abs(QR.loglog_curvature(al, y)) < 1e-9
    abrupt = al ** 2 * np.exp(10 * al)
    assert QR.loglog_curvature(al, abrupt) > 0.5
    dose, flag = QR.cross_dose(al, y, 2.0 * 0.1 ** 2, log_y=True)
    assert flag == 0 and abs(dose - 0.1) < 1e-9
    assert QR.cross_dose(al, y, 1e3)[1] == 1 and QR.cross_dose(al, y, 1e-9)[1] == -1
    dose, flag = QR.cross_dose(al, [0, 0, 0.1, 0.3, 0.9], 0.2)
    assert flag == 0 and 0.1 < dose < 0.2
    mx, share = QR.steepness(al, [0, 0, 0, 0, 1.0])
    assert abs(share * math.log(2) - 1) < 1e-9
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    z = x + 0.1 * rng.normal(size=200)
    yv = z + 0.1 * rng.normal(size=200)
    assert QR.spearman(x, yv) > 0.9
    assert abs(QR.partial_spearman(x, yv, z)) < 0.3  # x adds little once z is known
    est, (lo, hi) = QR.boot_ci(QR.spearman, [x, yv], n_boot=200)
    assert lo < est < hi and lo > 0.8
    assert QR.spearman([1, 2, np.inf], [1, 2, 3]) == 1.0


def test_run_layer_end_to_end(tmp_path):
    """The per-layer driver on the tiny model with smoke-sized counts: every stage runs and has the right shapes."""
    m, hf, _ = _tiny()
    g = torch.Generator().manual_seed(11)
    pool = [torch.randint(0, 97, (1, T), generator=g) for _ in range(56)]
    ctx = {"fit": pool[:2], "held": pool[8:12], "pool": pool[40:42]}
    fids = torch.randint(0, 97, (3, 20), generator=g)
    L = m.n_layers - 1
    top = m.unembed(QR.capture(m, fids, [L])[L][:, -1]).argmax(-1)
    facts = [{"ids": fids, "ans": torch.stack([top, torch.full_like(top, -1)], 1),
              "items": [{"name": str(i)} for i in range(3)]}]
    args = QR.parse(["--smoke", "--out", str(tmp_path / "o"), "--max-len", str(T), "--layers", "1"])
    args.n_held_jets, args.jb = 4, 4
    units, gates = {}, {}
    QR.run_layer(1, ctx, args, hf, m, facts, units, gates)
    od = tmp_path / "o_smoke" / "L01"
    dirs = torch.load(od / "dirs.pt", weights_only=False)
    n = len(dirs["U"]) + args.n_high
    beh = torch.load(od / "kl.pt", weights_only=False)
    assert beh["kl"].shape == (n, 5, 2, 4) and float(beh["kl_zero"].abs().max()) < 1e-12
    fct = torch.load(od / "facts.pt", weights_only=False)
    assert fct["cor"].shape == (n, 5, 2, 3) and bool(fct["cor_zero"].all())
    cont = torch.load(od / "cont.pt", weights_only=False)
    assert cont["edit"].shape[1:] == (5, 2, 2) and float(cont["edit_zero"].max()) == 0
    held = torch.load(od / "held.pt", weights_only=False)
    assert held["kl_quad"].shape == (n, 4, 5, 2)
    g1 = gates[1]
    assert g1["tail_vs_full_rel"] < 1e-12 and g1["resid_Q_rel"] < 1e-8 and g1["maxr_ratio_gain_min"] > 1
    assert g1["taylor_kl_rel_median_a0"] < 0.05, g1
    assert (od / "quick.json").exists() and set(units[1]) >= {"svd", "sharp", "maxr", "kl", "facts", "cont"}
    # the summariser runs on these stage files and writes the notes table
    import summarize_q_robust as SQ
    res = {1: SQ.analyse(1, SQ.load_layer(od), n_boot=20)}
    SQ.write_md(res, tmp_path / "r.md", str(tmp_path))
    txt = (tmp_path / "r.md").read_text()
    assert "Source layer 1" in txt and "| maxr |" in txt and res[1]["N"] == n
    i = txt.index("## Setup")
    (tmp_path / "r.md").write_text(txt[:i] + "## Verdict\n\nkeep me\n\n" + txt[i:])
    SQ.write_md(res, tmp_path / "r.md", str(tmp_path))  # a re-run keeps the hand-written verdict, once
    txt2 = (tmp_path / "r.md").read_text()
    assert txt2.count("keep me") == 1 and txt2.count("## Setup") == 1 and txt2.index("keep me") < txt2.index("## Setup")
