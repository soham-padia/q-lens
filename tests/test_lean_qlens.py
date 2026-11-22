"""CPU tests for src/lean_qlens.py: the tactic-step rule, edit enumeration with the re-tokenization check, and, on a
random 4-block Qwen3 in float64 with the real Qwen3 tokenizer: Tail == full model, an edit push reproduces the edited
run exactly, the jets equal finite differences, the double difference tends to Q(da, db), the dose finder, and an
end-to-end run of every stage on synthetic Lean files (`python tests/test_lean_qlens.py e2e <dir>`)."""
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
JLENS_DIR = os.environ.get("JLENS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens"))
sys.path.insert(0, JLENS_DIR)
import lean_qlens as LQ  # noqa: E402

torch.set_num_threads(4)

SNIP = """import Mathlib

/-- A doc comment with -- dashes and a by word. -/
theorem t0 (x y : ℕ) (h : x < y) : x ≤ y + 0 := by
  have h1 : x ≤ y := le_of_lt h
  · exact Nat.le_add_right_of_le h1

lemma t1 (a b : ℤ) (h : a ≤ b) : a + 1 ≤ b + 1 := by
  linarith -- a comment by the way
  calc a = a := by rfl
    _ = a := by simp

example (p q : Prop) (hp : p) (hq : q) : p ∧ q := ⟨hp, by exact hq⟩
"""


def test_tactic_starts():
    got = [SNIP[o: o + 8].split()[0] for o in LQ.tactic_starts(SNIP)]
    assert got == ["have", "exact", "linarith", "calc", "rfl", "simp", "exact"], got


def _tok():
    import transformers
    return transformers.AutoTokenizer.from_pretrained(LQ.BASE_TOK)


def _lean_file(seed, n=7):
    r = random.Random(seed)
    out = ["import Mathlib\n"]
    for i in range(n):
        x, y = r.sample(["x", "y", "n", "m", "a", "b"], 2)
        T_ = r.choice(["ℕ", "ℤ", "ℝ", "Nat", "Int"])
        out.append(f"theorem t{seed}_{i} ({x} {y} : {T_}) (h : {x} < {y}) (h2 : 0 ≤ {x} + {y}) :\n"
                   f"    {x} * 1 ≤ {y} + {r.randint(1, 9)} ∧ True := by\n"
                   f"  have h1 : {x} ≤ {y} := le_of_lt h\n"
                   f"  constructor\n  · nlinarith [h1, h2, sq_nonneg ({x} - {y})]\n  · trivial\n")
    return "\n".join(out)


def test_edits():
    tok = _tok()
    subs, ops, dropped = LQ.build_subs(tok)
    text = _lean_file(0)
    starts = LQ.tactic_starts(text)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    tstart = [o[0] for o in offs]
    import bisect
    ks = [bisect.bisect_right(tstart, c) - 1 for c in starts]
    k = [k for k in ks if k >= LQ.T][0]
    ctx = ids[k - LQ.T: k]
    assert tok.decode([ids[k]]).strip().isalpha()
    assert LQ._check_edit(tok, text, offs, k, ctx, [], [])
    cands = LQ.edit_candidates(tok, text, offs, k, ctx, subs, ops)
    cls = {c["cls"] for c in cands}
    assert {"rel", "op", "const", "swap"} <= cls, cls
    for c in cands:
        assert all(16 <= p <= LQ.T - 2 for p, _ in c["changes"])
        ed = LQ.apply_changes(ctx, c["changes"])
        assert ed != ctx
    pairs = LQ.edit_pairs(tok, text, offs, k, ctx, cands, random.Random(0))
    assert len(pairs) == 3


def _tiny():
    import jlens
    import types
    hf = LQ.tiny_model()
    m = jlens.from_hf(hf, types.SimpleNamespace(bos_token_id=None))
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 5000, (1, LQ.T), generator=g)
    return hf, m, ids


def test_tail_edit_jets_dose_mixed():
    hf, m, ids = _tiny()
    l = 1
    box, z = LQ.capture_last(m, ids, [l])
    with torch.no_grad():
        ref = hf(ids).logits[:, -1]
    assert torch.allclose(z, ref, atol=1e-9, rtol=1e-9)
    tail = LQ.Tail(m, l, LQ.T)
    h0 = box[l]
    with torch.no_grad():
        assert torch.allclose(tail.last(h0), ref, atol=1e-9, rtol=1e-9)
    ed = ids.clone(); ed[0, 40] = 777
    ed2 = ids.clone(); ed2[0, 90] = 999
    be, ze = LQ.capture_last(m, ed, [l])
    b2, z2 = LQ.capture_last(m, ed2, [l])
    d = be[l] - h0; d2 = b2[l] - h0
    assert float(d[0, :40].abs().max()) == 0.0
    with torch.no_grad():
        assert torch.allclose(tail.last(h0 + d), ze, atol=1e-8, rtol=1e-8)  # the edit push reproduces the edited run
    to, te = int(z.argmax()), int(ze.argmax())
    if to == te:
        te = int(z[0].topk(2).indices[1])
    # jets vs central differences
    z0, J, Q = LQ.jets(tail, h0, d)
    g = lambda lam: float((lambda zz: zz[0, te] - zz[0, to])(tail.last(h0 + lam * d).detach()))  # noqa: E731
    e = 1e-4
    g1, g2 = float(J[0, te] - J[0, to]), float(Q[0, te] - Q[0, to])
    assert abs((g(e) - g(-e)) / (2 * e) - g1) < 1e-5 * max(1, abs(g1))
    assert abs((g(e) - 2 * g(0) + g(-e)) / e ** 2 - g2) < 1e-3 * max(1, abs(g2))
    # dose: g(lam*) ~ 0 when a crossing exists
    ls, grid = LQ.dose(tail, h0, d, to, te)
    if math.isfinite(ls):
        assert 0 < ls <= 1 and abs(g(ls)) < 1e-2 * max(abs(v) for v in grid)
    # mixed second derivative: double difference / s^2 -> Q(d, d2)
    _, _, Qm = LQ.jets(tail, h0, d, d2)
    f = lambda v: LQ.centre(tail.last(h0 + v)).detach()[0]  # noqa: E731
    s = 1e-3
    D2 = (f(s * (d + d2)) - f(s * d) - f(s * d2) + f(0 * d)) / s ** 2
    assert LQ.rel(LQ.centre(Qm)[0], D2) < 1e-3
    # symmetric
    _, _, Qm2 = LQ.jets(tail, h0, d2, d)
    assert torch.allclose(Qm, Qm2, rtol=1e-6, atol=1e-8)


def test_flip_formulas():
    assert abs(LQ.flip_JQ(-1, 1, 2) - (math.sqrt(5) - 1) / 2) < 1e-12
    assert LQ.flip_J(-1, 2) == 0.5 and LQ.flip_J(-1, -1) == math.inf
    assert LQ.flip_JQ(-1, 0, -2) == math.inf


def e2e(out):
    """Every stage on synthetic Lean files and the tiny model (CPU)."""
    out = Path(out); data = out / "data"
    for name in ("riemann-lean", "mathlib"):
        (data / name).mkdir(parents=True, exist_ok=True)
        files = []
        for i in range(14):
            fn = f"F{i}.lean"
            (data / name / fn).write_text(_lean_file(i + (100 if name == "mathlib" else 0)))
            files.append(fn)
        (data / f"{name}.sample_files.txt").write_text("\n".join(files) + "\n")
        (data / f"{name}.commit").write_text("synthetic\n")
    LQ.main(["--stage", "data", "--data", str(data), "--n-lean", "12"])
    C = torch.load(data / "contexts.pt", weights_only=False)
    assert len(C["corpora"]["RH"]["contexts"]) == 12, C["corpora"]["RH"]["stats"]
    LQ.main(["--stage", "check,E1,E23", "--tiny", "--data", str(data), "--out", str(out / "tiny"), "--n-lean", "12",
             "--n-e1", "2", "--n-dir", "2", "--n-e23", "4", "--layers", "1,2", "--n-fd", "2"])
    chk = json.loads((out / "tiny/check.json").read_text())
    assert chk["RH"]["pass"] and chk["FW"]["pass"], chk
    for c in ("RH", "ML", "FW"):
        E1 = torch.load(out / f"tiny/E1_{c}.pt", weights_only=False)
        assert set(E1["raw"]) == {1, 2}
    LQ.main(["--stage", "gate2", "--tiny", "--data", str(data), "--out", str(out / "tiny"), "--n-lean", "12",
             "--n-e23", "4"])
    for c in ("RH", "ML"):
        G = json.loads((out / f"tiny/gate2_{c}.json").read_text())
        for l in ("1", "2"):
            assert G["E2"][l]["max_g0_diff"] < 1e-9 and G["E3"][l]["max_cos_diff"] < 1e-9, (c, l)
            assert G["E2"][l]["n"] == len(json.loads((out / f"tiny/E23_{c}.json").read_text())["cases"])
        print(c, "gate2", {l: (G["E2"][l]["med_rel_g2"], G["E3"][l]["med_best_cos"]) for l in ("1", "2")})
    LQ.main(["--stage", "E4", "--tiny", "--data", str(data), "--out", str(out / "tiny"), "--n-lean", "12",
             "--n-e23", "4", "--n-fd", "2"])
    for c in ("RH", "ML"):
        G = json.loads((out / f"tiny/E4_{c}.json").read_text())
        for l, g in G["fd_gate"].items():
            assert g["pass"] and g["max_g_rel"] < 1e-8, (c, l, g)
        x = G["cases"][0]["L"]["1"]
        assert x["J"]["new_up_loose"] <= x["J"]["new_up"] and len(x["top"]["hQ_up"]) == 10
        assert ("ctx_tail" in G["cases"][0]) == (c == "ML")
        print(c, "E4", G["fd_gate"])
    for c in ("RH", "ML"):
        E = json.loads((out / f"tiny/E23_{c}.json").read_text())
        assert E["pairs"], "no E3 pairs"
        for pr in E["pairs"]:
            for l, r in pr["L"].items():
                assert -1 <= r["cos"] <= 1 and r["share"] >= 0
        fd = [r["cos_s"]["0.003"] for pr in E["pairs"][:2] for r in pr["L"].values()]
        assert min(fd) > 0.99, fd
        for c_ in E["cases"][:2]:
            for r in c_["L"].values():
                assert r["fd"]["rel_g1"] < 1e-3 and r["fd"]["rel_g2"] < 1e-2, r["fd"]
                assert abs(r["g_tail1"] - c_["g_full"]) < 1e-6 * max(1, abs(c_["g_full"]))
        print(c, "cases", len(E["cases"]), "pairs", len(E["pairs"]), E["acc"])
    print("E2E_OK")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "e2e":
    e2e(sys.argv[2])
