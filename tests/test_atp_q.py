"""CPU tests for src/atp_q.py on a tiny random Qwen3 in float64: patch mechanics, AtP == jvp, J + Q/2 == the quadratic
Taylor term of an actual patch, the reverse-over-forward HVP against the nested jvp, the stochastic diagonal
estimator's unbiasedness (algebraic identity and Monte Carlo), EAP-IG completeness, and the per-pair driver."""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import atp_q as AQ  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)


def _tiny(res_layers=(0, 1)):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(cfg).double().eval()
    with torch.no_grad():
        for p in hf.parameters():
            if p.dim() == 2:
                p.mul_(6.0)  # make the blocks properly nonlinear
    for p in hf.parameters():
        p.requires_grad_(False)

    def _rms64(self, x):  # HF's Qwen3RMSNorm casts to fp32; keep float64 so the references are exact
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    for mod in hf.modules():
        if type(mod).__name__.endswith("RMSNorm"):
            mod.forward = types.MethodType(_rms64, mod)
    T = 12
    clean = torch.randint(0, 97, (1, T), generator=torch.Generator().manual_seed(1))
    corrupt = clean.clone()
    corrupt[0, 5], corrupt[0, 9] = (clean[0, 5] + 7) % 97, (clean[0, 9] + 13) % 97
    return AQ.Lab(hf, clean, corrupt, 3, 7, list(res_layers))


def test_patch_mechanics():
    lab = _tiny(res_layers=(-1, 0, 1))
    for l, v in lab.full_res_gate().items():
        assert v < 1e-12, (l, v)  # whole-layer residual patch == corrupt run (embedding and block outputs)
    assert lab.null.sum() == 10 + 2 * 5  # embedding: all but the 2 changed tokens; blocks: the 5 positions before
    assert not lab.null[lab.kind != 2].any()
    a = lab.patch(np.arange(lab.S), bs=7)  # small batches: every batch carries its own baseline row
    b = lab.patch(np.arange(lab.S), bs=64)
    assert np.abs(a - b).max() < 1e-12
    assert np.abs(a[lab.null]).max() < 1e-12


def test_atp_equals_jvp_and_taylor():
    lab = _tiny()
    S = np.arange(lab.S)
    atp = lab.atp()
    J, Q = lab.jets_all(S, bs=16)
    assert np.abs(atp - J).max() < 1e-10 * np.abs(J).max()
    # J + Q/2 is the quadratic Taylor term of the actual patch: central differences in lambda
    lam = 1e-3
    live = S[~lab.null]
    fp, fm = lab.patch(live, [lam] * len(live)), lab.patch(live, [-lam] * len(live))
    assert np.abs((fp - fm) / (2 * lam) - J[live]).max() < 1e-6 * np.abs(J).max()
    assert np.abs((fp + fm) / lam ** 2 - Q[live]).max() < 1e-4 * np.abs(Q).max()
    # and the full patch differs from J + Q/2 by a cubic remainder: smaller patches agree better
    for lam_big in (0.05,):
        a = lab.patch(live, [lam_big] * len(live))
        r2 = np.abs(a - lam_big * J[live] - lam_big ** 2 * Q[live] / 2)
        r1 = np.abs(a - lam_big * J[live])
        assert np.median(r2) < np.median(r1) / 5
    # vector jets agree with the scalar ones through the metric readout
    Jv, Qv = lab.jets(list(live[:6]), vec=True)
    assert torch.allclose(Jv @ lab.net.wd, torch.tensor(J[live[:6]]), rtol=1e-10, atol=1e-12)
    assert torch.allclose(Qv @ lab.net.wd, torch.tensor(Q[live[:6]]), rtol=1e-8, atol=1e-12)


def test_stochastic_estimator_unbiased():
    lab = _tiny()
    _, Q = lab.jets_all(np.arange(lab.S), bs=16)
    for f in (0, 1):
        sites = lab.fam_sites[f]
        n = len(sites)
        # one-hot probes give the full cross matrix C[s, s'] = delta_s^T H_ss' delta_s'; its diagonal is exact Q
        C = lab.hvp(f, torch.eye(n)).numpy().T  # row b = probe e_b -> column b of C
        assert np.abs(np.diag(C) - Q[sites]).max() < 1e-9 * max(np.abs(Q[sites]).max(), 1e-30)
        assert np.abs(C - C.T).max() < 1e-9 * np.abs(C).max()  # Hessian symmetry
        # each draw is exactly qhat = z o (C z), so E[qhat] = diag(C) because E[z_s z_s'] = [s == s']
        z, qh = lab.stoch(f, 8, torch.Generator().manual_seed(5), bs=8)
        assert np.abs(qh - z * (z @ C.T)).max() < 1e-9 * np.abs(C).max()
        # Monte Carlo: the mean over many draws converges to exact Q at the 1/sqrt(K) rate
        z, qh = lab.stoch(f, 4000, torch.Generator().manual_seed(6), bs=1000)
        se = qh.std(0, ddof=1) / np.sqrt(len(qh))
        live = ~lab.null[sites]
        zs = np.abs(qh.mean(0) - Q[sites])[live] / np.maximum(se[live], 1e-30)
        assert zs.max() < 4.5, zs.max()
        off = C - np.diag(np.diag(C))
        assert np.allclose(qh.var(0, ddof=1)[live], (off ** 2).sum(1)[live], rtol=0.15)  # Var = sum_{s'!=s} C^2


def test_eap_ig_completeness():
    lab = _tiny(res_layers=(-1,))
    ig = lab.eap_ig((1, 64), bs=32)
    tot = lab.m_corrupt - lab.m_clean
    emb = lab.fam_sites[1]
    assert abs(ig[64][emb].sum() - tot) < 2e-3 * abs(tot)  # midpoint rule, O(1/m^2)
    # one step = the gradient at the midpoint input, times delta
    a = torch.tensor([0.5])
    lab.net.interp = (a, lab.ediff)
    E0 = {k: torch.zeros((1,) + lab.shape[k], requires_grad=True) for k in lab.keys}
    m = lab.net(E0)
    G = torch.autograd.grad(m.sum(), [E0[k] for k in lab.keys])
    lab.net.interp = None
    ref = lab.readoff(dict(zip(lab.keys, G)), lab.keys)[0].numpy()
    assert np.abs(ig[1] - ref).max() < 1e-12


def test_run_pair_end_to_end():
    lab = _tiny()
    r = AQ.run_pair(lab, K=16, ig_steps=(2, 4), lams=(0.25, 0.5), n_partial=4, bs_patch=16, bs_jet=8, bs_hvp=8,
                    bs_ig=6, seed=3)
    assert not AQ.hard_gates(r["gates"], fd_Q=1e-5, fd_J=1e-4), r["gates"]
    assert all(x["Qj_rel"] < 1e-6 and x["Q_rel"] < 1e-3 for x in r["gates"]["fd"]), r["gates"]["fd"]
    assert r["gates"]["atp_vs_J"] < 1e-10 and r["gates"]["hvp_onehot_rel"] < 1e-9
    assert r["stoch"][0].shape == (16, len(lab.fam_sites[0])) and set(r["ig"]) == {2, 4}
    assert np.isfinite(r["Q"]).all() and len(r["partial"][0.25]) == len(r["partial_sites"])
    # partial patches: J + Q/2 tracks the actual effect better at small lambda than at large
    a = {lam: r["partial"][lam] for lam in (0.25, 0.5)}
    s = r["partial_sites"]
    e = {lam: np.abs(a[lam] - lam * r["J"][s] - lam ** 2 * r["Q"][s] / 2).sum() / np.abs(a[lam]).sum()
         for lam in a}
    assert e[0.25] < e[0.5]


def test_block_split_sums_to_Q():
    lab = _tiny(res_layers=(-1, 0, 1))
    _, Q = lab.jets_all(np.arange(lab.S), bs=16)
    live = np.nonzero(~lab.null)[0]
    sites = [s for s in live if lab.kind[s] == 0][:3] + [s for s in live if lab.kind[s] == 1] + \
        [s for s in live if lab.kind[s] == 2][::3]
    terms = lab.split_terms = lab.block_split(sites)
    own = Q[sites] - np.nansum(terms, 1)
    for b, s in enumerate(sites):
        assert np.isnan(terms[b, : lab.site_block(s) + 1]).all() and np.isfinite(terms[b, lab.site_block(s) + 1:]).all()
        if lab.kind[s] != 0:  # mlp and res sites: the split over downstream blocks + final norm is exact
            assert abs(own[b]) < 1e-10 * max(np.nansum(np.abs(terms[b])), 1e-30), (s, own[b], terms[b])
    assert np.abs(own[[b for b, s in enumerate(sites) if lab.kind[s] == 0]]).max() > 1e-8  # heads: MLP half bends
