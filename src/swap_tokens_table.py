#!/usr/bin/env python3
"""Tables from src/swap_tokens.py output (CPU, no model): gates, token lists (Markdown + LaTeX), ranks, junk counts.

  python src/swap_tokens_table.py $QLENS_OUT/swap_tokens_v1/swap_tokens.json [--md results/tables/swap_token_tables_full.md] [--k 10]
"""
import json
import sys
from pathlib import Path

ARGV = sys.argv[1:]
MD = ARGV[ARGV.index("--md") + 1] if "--md" in ARGV else None
K = int(ARGV[ARGV.index("--k") + 1]) if "--k" in ARGV else 10
src = [x for x in ARGV if x.endswith(".json")][0]
R = json.loads(Path(src).read_text())
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def vis(s):  # Markdown: visible spaces / newlines
    return s.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␣").replace("|", "\\|")


def tex(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", " ": r"\textvisiblespace{}",
           "\n": r"\textbackslash{}n", "\t": r"\textbackslash{}t"}
    return "".join(rep.get(c, c) for c in s)


def name(t):
    return f"{t['func']}: {t['src']}→{t['tgt']} ({t['old'][0]}→{t['new'][0]})"


def fmt(v):
    return "∞" if v in (None, float("inf")) or (isinstance(v, float) and v != v) else f"{v:.2f}"


say(f"# Token readout of the swap jets ({Path(src).name})")
say()
say(f"Vocabulary: {R['vocab']} tokens (ids < len(tokenizer)); centred logits. Layers {R['layers']}. Lists unfiltered.")
say()
say("## Gates (relative L2 over the centred vocabulary unless noted)")
say("| swap | L | J vs FD | Q vs FD | J vs FD (Richardson) | Q vs FD (Richardson) | g1 vs S2 | g2 vs S2 | r0 vs S2 (abs) | "
    "z(0) vs HF logits | FD top-k overlap Q/2 up/down | secs |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in R["trials"]:
    for l, e in t["layers"].items():
        g = e["gates"]
        say(f"| {t['func']} | {l} | {g['J_vs_fd']:.1e} | {g['Q_vs_fd']:.1e} | {g['J_vs_fd_richardson']:.1e} | "
            f"{g['Q_vs_fd_richardson']:.1e} | {g.get('g1_vs_s2', float('nan')):.1e} | {g.get('g2_vs_s2', float('nan')):.1e} | "
            f"{abs(g['r0_vs_s2']):.1e} | {g.get('z0_vs_hf', float('nan')):.1e} | "
            f"{e['fd_topk_overlap_halfQ']['up']:.1f}/{e['fd_topk_overlap_halfQ']['down']:.1f} | {e['secs']} |")
say()
say("## Doses, sizes and agreement")
say("| swap | L | λ* actual | λ* J+Q | λ* J | |J| | |Q/2| | cos(J, Q/2) | cos(pred, actual) at λ* | cos(λ*J, actual) | "
    "rel err pred | rel err J only |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in R["trials"]:
    for l, e in t["layers"].items():
        say(f"| {t['func']} | {l} | {fmt(e['lam_act'])} | {fmt(e['lam_JQ'])} | {fmt(e['lam_J'])} | {e['norms']['J']:.1f} | "
            f"{e['norms']['halfQ']:.1f} | {e['cos']['J_halfQ']:+.2f} | {e['cos']['pred_actual']:.2f} | "
            f"{e['cos']['Jlam_actual']:.2f} | {e['pred_vs_actual_rel']:.2f} | {e['J_vs_actual_rel']:.2f} |")
say()
say("## Ranks of the answers (target = new: rank among the most raised; source = old: rank among the most lowered)")
say("| swap | L | target ↑ J | ↑ Q/2 | ↑ J+Q/2 at λ* | ↑ actual at λ* | source ↓ J | ↓ Q/2 | ↓ J+Q/2 at λ* | ↓ actual |")
say("|---|---|---|---|---|---|---|---|---|---|")
for t in R["trials"]:
    for l, e in t["layers"].items():
        rk = e["ranks"]
        say(f"| {name(t)} | {l} | " + " | ".join(str(rk[x]["new"]["up"]) for x in ("J", "halfQ", "pred", "actual")) + " | "
            + " | ".join(str(rk[x]["old"]["down"]) for x in ("J", "halfQ", "pred", "actual")) + " |")
say()
say("## Token lists (top %d; value = centred logit change per unit λ for J, per unit λ² for Q/2)" % K)
junk = {}
for t in R["trials"]:
    for l, e in t["layers"].items():
        say(f"### {name(t)}, L{l}  (λ* = {fmt(e['lam_act'])})")
        for key, lab in (("halfQ_up", "Q/2 raises"), ("halfQ_down", "Q/2 lowers"), ("J_up", "J raises"), ("J_down", "J lowers"),
                         ("act_up", "actual change at λ* raises"), ("act_down", "actual change at λ* lowers")):
            lst = e["top"][key][:K]
            for x in lst:
                junk.setdefault(key, {}).setdefault(x["kind"], 0)
                junk[key][x["kind"]] += 1
            say(f"- {lab}: " + ", ".join(f"`{vis(x['tok'])}` {x['val']:+.2f}" + ("" if x["kind"] == "word" else f" [{x['kind']}]")
                                         for x in lst))
        say()
say("## Junk count over all lists (classes: word / punct = no letter or digit / space = whitespace only / byte = undecodable)")
for key, d in junk.items():
    say(f"- {key}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())))
say()
nonascii = any(ord(c) > 127 for t in R["trials"] for e in t["layers"].values() for key in ("halfQ_up", "halfQ_down", "J_up", "J_down")
               for x in e["top"][key][:K] for c in x["tok"])
say("## LaTeX (booktabs + tabularx; ␣ = \\textvisiblespace" + ("; contains non-ASCII tokens: XeLaTeX/xeCJK or replace" if nonascii else "") + ")")
say("```latex")
say(r"\begin{table*}[t]\centering\scriptsize")
say(r"\caption{Tokens moved by the swap jets in logit space (centred), at the last position of the two-hop prompt, for "
    r"the actual swap push at one layer. $\tfrac12 Q$: second-order term per unit $\lambda^2$; $J$: first-order term per "
    r"unit $\lambda$. Ranks: target answer among the most raised, source answer among the most lowered, under $J$, "
    r"$\tfrac12 Q$ and $\lambda^* J+\tfrac12\lambda^{*2}Q$ at the actual flip dose $\lambda^*$.}")
say(r"\begin{tabularx}{\textwidth}{@{}l r r X X c c@{}}\toprule")
say(r"Swap & $\ell$ & $\lambda^*$ & $\tfrac12 Q$ raises (top %d) & $\tfrac12 Q$ lowers (top %d) & target rank $J$/$\tfrac12Q$/$J{+}\tfrac12Q$ & source rank \\ \midrule" % (K, K))
for t in R["trials"]:
    first = True
    for l, e in t["layers"].items():
        rk = e["ranks"]
        sw = tex(f"{t['old'][0]}→{t['new'][0]}").replace("→", r"$\to$") if first else ""
        first = False
        up = ", ".join(r"\texttt{%s}" % tex(x["tok"]) for x in e["top"]["halfQ_up"][:K])
        dn = ", ".join(r"\texttt{%s}" % tex(x["tok"]) for x in e["top"]["halfQ_down"][:K])
        say(f"{sw} & {l} & {fmt(e['lam_act'])} & {up} & {dn} & "
            + "/".join(str(rk[x]["new"]["up"]) for x in ("J", "halfQ", "pred")) + " & "
            + "/".join(str(rk[x]["old"]["down"]) for x in ("J", "halfQ", "pred")) + r" \\")
    say(r"\midrule" if t is not R["trials"][-1] else r"\bottomrule")
say(r"\end{tabularx}\end{table*}")
say()
say(r"\begin{table*}[t]\centering\scriptsize")
say(r"\caption{Contrast: tokens moved by the first-order term $J$ (same swaps and layers).}")
say(r"\begin{tabularx}{\textwidth}{@{}l r X X@{}}\toprule")
say(r"Swap & $\ell$ & $J$ raises (top %d) & $J$ lowers (top %d) \\ \midrule" % (K, K))
for t in R["trials"]:
    first = True
    for l, e in t["layers"].items():
        sw = tex(f"{t['old'][0]}→{t['new'][0]}").replace("→", r"$\to$") if first else ""
        first = False
        up = ", ".join(r"\texttt{%s}" % tex(x["tok"]) for x in e["top"]["J_up"][:K])
        dn = ", ".join(r"\texttt{%s}" % tex(x["tok"]) for x in e["top"]["J_down"][:K])
        say(f"{sw} & {l} & {up} & {dn} \\\\")
    say(r"\midrule" if t is not R["trials"][-1] else r"\bottomrule")
say(r"\end{tabularx}\end{table*}")
say("```")
if MD:
    Path(MD).write_text("\n".join(OUT) + "\n")
    print("wrote", MD)
