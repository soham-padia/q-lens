#!/usr/bin/env python3
"""LaTeX token table for the paper (Table tab:tokens) from src/swap_tokens.py output. CPU only, ASCII output.

Display rules (stated in the table caption):
  * tokens in a non-Latin script are shown as a language tag plus an English gloss (GLOSS below), e.g. ru:`China';
  * tokens that are the same string up to case, a leading space or a leading punctuation mark are merged into one
    entry, followed by the number of variants when there is more than one; a trailing hyphen marks a word piece;
  * the entries are taken in order from the 10 most raised tokens, at most 5 per cell.
The raw token strings are in swap_tokens.json and in results/tables/swap_token_tables_full.md.

  python paper/token_table.py $QLENS_OUT/swap_tokens_v1/swap_tokens.json > results/tables/token_table.tex
"""
import json
import sys
import unicodedata

GLOSS = {  # lower-cased token, stripped of leading space/punctuation -> (language tag, gloss)
    "维也纳": ("zh", "Vienna"), "ви": ("ru", "vi-"), "китай": ("ru", "China"), "الصينية": ("ar", "Chinese"),
    "奥地利": ("zh", "Austria"), "中央": ("zh", "central"), "亚洲": ("zh", "Asia"), "越南": ("zh", "Vietnam"),
    "중국": ("ko", "China"), "泰": ("zh", "Thai"),
}
PIECES = {"vi", "be", "viet", "vien", "asi", "hung", "dan", "bav", "chines", "asiat", "ar", "alp", "mand"}
REL = {"capital": r"Paris$\to$Beijing", "language": r"French$\to$Chinese", "continent": r"Europe$\to$Asia"}
LEAD = " _-./("
ACC = {"́": "'", "̀": "`", "̂": "^", "̈": '"', "̃": "~", "̧": "c"}


def key(t):
    return t.lstrip(LEAD).lower()


def latin(s):
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}.get(ch, ch))
            continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and ord(d[0]) < 128 and d[1] in ACC:
            out.append("\\%s{%s}" % (ACC[d[1]], d[0]))
            continue
        return None
    return "".join(out)


def entries(lst, n=5):
    groups = []  # [key, display token, count]: the display token is the first capitalised variant, else the first
    for t in lst:
        k = key(t["tok"])
        for g in groups:
            if g[0] == k:
                g[2] += 1
                if not g[1].lstrip(LEAD)[:1].isupper() and t["tok"].lstrip(LEAD)[:1].isupper():
                    g[1] = t["tok"]
                break
        else:
            groups.append([k, t["tok"], 1])
    cells = []
    for k, first, c in groups[:n]:
        if k in GLOSS:
            lang, gl = GLOSS[k]
            s = "%s:`%s'" % (lang, gl)
        else:
            s = latin(first.lstrip(" ") if c == 1 else first.lstrip(LEAD))
            assert s is not None, ("needs a gloss", first)
            if k in PIECES:
                s += "-"
        if c > 1:
            s += r"\,(%d)" % c
        cells.append(s)
    return ", ".join(cells)


def rank_fmt(x):
    return "{:,}".format(x).replace(",", "{,}")


def main(path):
    R = json.load(open(path))
    rows = []
    for tr in R["trials"]:
        first = True
        for L in sorted(tr["layers"], key=int):
            c = tr["layers"][L]
            top = c["top"]
            rk = c["ranks"]
            if first:
                rows.append("\\multicolumn{5}{@{}l}{\\emph{%s}} \\\\" % REL[tr["func"]])
            rows.append("%s & %s & %s & %s / %s & %.2f / %.2f \\\\" % (
                L, entries(top["halfQ_up"]), entries(top["J_up"]),
                rank_fmt(rk["J"]["new"]["up"]), rank_fmt(rk["halfQ"]["new"]["up"]), c["lam_JQ"], c["lam_act"]))
            first = False
        rows.append(r"\midrule")
    rows.pop()
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1])
