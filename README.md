# Q-lens: exact second-order readouts across layers

**Code for the paper "The Q-Lens: Exact Second-Order Readouts Show That Language Models Bend Across Layers"
(Soham Padia, 2026).** This is not QLens (arXiv:2510.11963) or Query Lens (arXiv:2606.07617), which are different
works.

The Q-lens is the exact second-order term Q(u, v) of a downstream readout of a language model: push the residual
stream of block l by a direction u, read the response F at a later block (here the sum over positions of the last
block's output), and Q is the second derivative of F along the push, next to the first-order term J u. Q is computed
by nesting forward-mode automatic differentiation (`torch.func.jvp` of a `jvp`), so it costs a few forward passes
and no Hessian is ever formed. Because Q is a sum over the blocks the push passes through, it splits exactly over
blocks, sublayers, attention heads and MLP neurons, and one more level of nesting gives the third-order term.

## Install

Python 3.11. The versions used for the paper are pinned in `requirements.txt`; install the `torch` build that
matches your CUDA.

```bash
pip install -r requirements.txt
git clone https://github.com/anthropics/jacobian-lens.git
git -C jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
```

The J-lens dependency is [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0). It is
installed from its URL: `requirements.txt` pins `jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398`
(full commit hash in the file). The code wraps models with `jlens.from_hf`, uses its valid-position convention
(positions 16..T-2), fits the first-order lens with `jlens.fit` (`src/fit_shard.py`), averages the lens shards as `JacobianLens.merge` does, and
reads its evaluation and experiment data files (`data/evaluations/`, `data/experiments/`). Those data files are not
part of the pip package, hence the clone. No J-lens code is copied into this repository.

The looped models (Huginn-0125, Ouro) need transformers 4.55 for their remote code. Run `src/loop_lens.py` in a
separate environment built from `requirements-loop.txt` (transformers 4.55.4, huggingface_hub 0.36.2; the other
packages as above) and point `LOOP_PY` at its python.

### Paths

| variable | default | meaning |
|---|---|---|
| `QLENS_OUT` | `./outputs` | root of all experiment outputs (corpus, lens shards, per-run JSON and tensors) |
| `JLENS_DIR` | `./jacobian-lens` | the jacobian-lens clone (put on `sys.path` before the installed `jlens`) |
| `HF_HOME` | Hugging Face default | model weights; the runners set `HF_HUB_OFFLINE=1` after fetching |

## Quickstart

J and Q for one push on a small model (Qwen3.5-0.8B), compared with the actual change of the readout:

```bash
python scripts/quickstart.py                # ~15 s on 4 CPU cores; --device, --model, --layer, --alpha
```

Output on CPU (fp32):

```
Qwen/Qwen3.5-0.8B: push at block 12 of 24, read at block 23, T = 45, 28 valid positions, rho = 2.4, alpha = 0.1
direction 0: |J u| = 125  |Q(u,u)| = 171.3  r = 1.370  rel. error J = 0.0699  J+Q/2 = 0.0056
direction 1: |J u| = 118.1  |Q(u,u)| = 182.6  r = 1.547  rel. error J = 0.0756  J+Q/2 = 0.0062
```

The core is four lines of `src/so_core.py`:

```python
f = Perturbed(m, ids.expand(2, -1), layer, m.n_layers - 1)   # delta -> F = sum over valid positions of h_L
Ju, _, Quu = first_and_second(f, U, None, signs=False)       # nested jvp: J u and Q(u, u), per pushed position
act = f(alpha * f.spread(U, None)) - f(0 * f.spread(U, None))
# act ~ alpha J u + alpha^2 Q(u, u) / 2   (after scaling Ju, Quu by the number of pushed positions)
```

Tests (tiny random Qwen3 models in float64 on CPU, about 25 s): `python -m pytest tests/`.

## Reproducing the paper

`results/` holds the files the paper's numbers and figures are read from: `claims.csv` (one row per number, with
bootstrap interval and source file), `qj_curves.csv`, `explorable_data/`, `a3_by_context.csv`,
`d1_family_fits*.jsonl`, `jq_split_{depth,paths}.jsonl`, `fig_inputs/` (the arrays two figure panels read, extracted
from the raw outputs by `paper/fig_inputs.py`), and the generated tables in `results/tables/`. The figures need only
these:

```bash
bash scripts/21_figures.sh                  # figures/F0-F7 (CPU, about a minute)
```

To regenerate everything from scratch, run the GPU runners in `scripts/` (each is a plain bash script; every
command in it is one job of the paper's runs, so they can be submitted separately), then the CPU summaries:

```bash
bash scripts/00_data.sh                     # corpus (FineWeb-Edu, 2000 docs, seed 0) + weights
bash scripts/01_fit_jlens.sh                # ... through 11_swap_tokens.sh
bash scripts/20_summaries.sh                # results/*.csv, results/tables/*.md, results/claims.csv
bash scripts/21_figures.sh
```

Section numbers are those of the arXiv version; Appendix A ("Full results") gives the full numbers for Sections 3-9,
one subsection per section (A.1 for Section 3 through A.6 for Sections 8-9).

| paper section | script | runner | output (`$QLENS_OUT/...` unless noted) |
|---|---|---|---|
| 3 The Q-lens: exactness, timing; A.1; App. D exactness checks | `src/second_order.py --stages check`, `src/q_anatomy.py --stages check`, `src/loop_lens.py --stages L0` | `02`-`07` | `<run>/check.json`, `loop_*/L0.json` |
| 4 Near the output (Fig. 2: Taylor error of J and J+Q); A.2 (Table 3); App. E per-depth tables (Tables 7, 8) | `src/second_order.py --stages D1`, `paper/claims.py` (C1) | `02_qwen38_27b.sh`, `03_qwen3_depth.sh`, `20` | `second_order_v1/D1.json`, `so_qwen3-{8b,14b,32b}/D1.json`; `results/claims.csv` (C1), `results/tables/depth_test.md` |
| 5 The bend grows with the blocks traversed (Fig. 3a, four Qwen models); A.3 | `paper/claims.py` (C2 fits) | `20_summaries.sh` | `results/claims.csv` (C2), `results/qj_curves.csv` |
| 5 Other families (Llama-3.1-8B, OLMo-2-1B checkpoints); A.3; App. E two forms of fit (Table 9) | `src/second_order.py --stages check,vecs,D1`, `src/fit_d1_family.py` | `05_families.sh`, `20` | `so_llama3_1-8b/`, `so_olmo2-1b_ckpt/<rev>/`; `results/d1_family_fits.jsonl` (C2f), `results/d1_family_fits_v2.jsonl` (Table 9) |
| 5 The bend grows, \|JU\| dips and recovers (Fig. 3b); A.3 path of the two parts of r (Table 4); App. E log slopes of \|JU\| and \|Q\| (Table 10) | `src/jq_split_paths.py`, `src/jq_split_depth.py`, `paper/fig_inputs.py` | `20` | `results/jq_split_paths.jsonl`, `results/jq_split_depth.jsonl`, `results/fig_inputs/F2b_norms.json` |
| 5 Looped models bend less (Huginn, Ouro; Fig. 3a); A.3 | `src/loop_lens.py --stages L0,L1a,L1b` | `06_looped.sh` | `loop_{huginn,ouro14,ouro26}/L1a_raw.pt`; `results/claims.csv` |
| 6 The bend sets the dose (Fig. 4a; one- and two-hop swaps, fine grid of swap layers); A.4 (Table 5); App. I swap protocol; App. K reading and steering across layers | `src/second_order.py --stages S1,S2`, `src/summarize_s2.py`, `src/layer_pick.py` | `02`, `20` | `second_order_v1/S1.json`, `second_order_v1/s2/*/S2.json`, `layer_pick_v1/t*/S2.json`; `results/claims.csv` (S), `results/tables/s2_results.md`, `s2_reading.md` (App. K reading), `layer_pick_tables_{grid6,dense}.md` |
| 6 The bend carries interactions (conjunction test at 8 contexts); A.4 | `src/second_order.py --stages B2x,B4c --b2x-n-ctx 8`, `src/compare_scale.py _ctx8` | `02`, `03`, `20` | `<run>_ctx8/B2x.json`; `results/tables/scale_ladder_ctx8.md` |
| 6 Q is not a better lens; A.4 | `src/second_order.py --stages B3 --b3-name B3x_*` | `02` | `second_order_v1/B3x_{a,b,c}.json`; `results/claims.csv` |
| 6 Attribution patching (Fig. 4b-d); A.4; App. H (Table 14) | `src/atp_q.py`, `src/summarize_atp_q.py`, `paper/fig_inputs.py` | `09_atp_q.sh`, `20` | `atp_q_qwen3-8b/`; `results/tables/atp_q_results.md`, `results/fig_inputs/F6_atp_q.npz` |
| 6 Flag, then patch the flagged sites (Fig. 4c-d); App. H | `src/flag_then_patch.py` | `20` | `results/tables/atp_q_results.md` (section Flag-then-patch), `results/tables/figs_atp_q/flag_then_patch.png` |
| 7.1 Along swaps the bend departs from the baseline (Fig. 5); A.5; App. J split along swap pushes (Table 15) | `src/swap_split.py`, `src/swap_split_summary.py` | `08_swap_split.sh`, `20` | `swap_split_v1/swap_split_s*.json`; `results/tables/swap_split_tables.md` |
| 7.1 Token readout of a swap (Table 2); A.5 | `src/swap_tokens.py`, `src/swap_tokens_table.py`, `paper/token_table.py` | `11_swap_tokens.sh`, `20` | `swap_tokens_v1/swap_tokens.json`; `results/tables/swap_token_tables_full.md` (raw tokens), `results/tables/token_table.tex` (Table 2) |
| 7.2 On random and data directions the block terms add in quadrature (Fig. 6); A.5; App. F anatomy by push block (Fig. 8, Tables 11, 12) | `src/q_anatomy.py --stages A2` | `07_anatomy.sh` | `qanat_q38-27b/A2_{a32,b32}.json`, `qanat_qwen3-8b/A2_32.json`; `results/claims.csv` (C4), `results/explorable_data/` |
| 7.3 Layers bend each other's bends (Fig. 7, third order); A.5; App. G third order by context (Table 13) | `src/q_anatomy.py --stages A3`, `src/curv_circuits.py`, `paper/a3_by_context.py` | `07`, `20` | `qanat_*/A3*.json`, `A3*_raw.pt`; `results/claims.csv` (C5), `results/a3_by_context.csv`, `results/tables/curvature_circuits_tables_{27b,8b}.md` |
| 9 Limitations: does r mark fragile directions?; A.6 | `src/q_robust.py`, `src/summarize_q_robust.py` | `10_q_robust.sh`, `20` | `q_robust_qwen3-8b/`; `results/tables/q_robust_results.md` |
| A.1 timings; App. C setup (Table 6); App. D checks on the 122B mixture of experts and the Qwen3.5 ladder | `src/second_order.py`, `src/tiny_moe.py`, `src/compare_scale.py` | `04_scale_ladder.sh`, `20` | `so_qwen3_5-*/`; `results/tables/scale_ladder.md` |
| Figures 1-8 | `paper/figs_paper.py`, `paper/f5_paper.py` | `21_figures.sh` | `figures/F*.pdf` (below) |

Figure files: Fig. 1 `F0_schematic`, Fig. 2 `F1_taylor_error`, Fig. 3 `F2_bend_vs_depth`, Fig. 4 `F6_matters`,
Fig. 5 `F7_swap_departure`, Fig. 6 `F3_cancellation`, Fig. 7 `F5_third_order_paper`, Fig. 8 `F4_no_single_piece`.

Supporting scripts: `src/build_corpus.py` (prompt corpus), `src/fetch.py` (weights, verified by bytes on disk),
`src/fit_shard.py` (first-order J-lens on Qwen3.8-27B, used for the swap lens directions).

Raw outputs are not included in this release. The figures rebuild from `results/` alone. `paper/claims.py` and the
summaries need the raw outputs (corpus, per-run JSON and tensors), which are tens of GB; they are available on
request: open an issue. Notes on regenerating:

- The output names and layer list of the three `B3x` runs in `02_qwen38_27b.sh` are read off the outputs; the rest of
  that command is from the run log.
- `src/fit_d1_family.py` re-seeds its bootstrap for each (model, direction type), so a model's intervals do not
  depend on which other models one call fits.
- A few numbers in the text are arithmetic on these files rather than rows of `claims.csv` (for example the
  even-split null shares, and the count of exponential vs power-law fits in `d1_family_fits_v2.jsonl`).

## Hardware

Most model runs used one NVIDIA B200 GPU (180 GB); Qwen3.8-27B in fp32 fits on one. The swap runs keep its weights in
bf16 and some of their shards, like the looped models, ran on a 96 GB NVIDIA RTX PRO 6000. Qwen3.5-122B-A10B runs on
two B200s with bf16-stored weights. One nested jvp (two direction pairs) takes about 0.9 s on the
27B model and 4.0 s on the 122B model. Summaries, figures and tests are CPU only.

## Precision

The second-order term is small next to bf16 rounding, so the forward-mode path is fixed as follows (all in
`src/so_core.py`, unchanged from the paper's runs):

- **fp32 arithmetic.** Weights are loaded in fp32, or kept in their bf16 checkpoint dtype and upcast to fp32 at each
  use (`enable_upcast`, `--weights-dtype bfloat16 --dtype float32`), which computes the same function.
- **Math attention.** Scaled dot-product attention runs on the reference kernel (`math_sdpa`) inside every jvp.
- **TF32.** Finite-difference checks always run with TF32 off (`no_tf32`); bulk stages take `--tf32 0|1` as listed
  in the paper's setup table (the runners pass the paper's settings).
- **Mixture of experts.** Experts run eagerly (`experts_implementation="eager"`, which forward-mode AD supports).
  jvp differentiates with each token's expert set held fixed; the finite-difference check records and replays the
  routing (`patch_routers`, `routing("record"/"replay")`), and the difference from free routing is reported.
- **Multi-GPU.** Decoder layers are split over GPUs by `layer_device_map`, and every block output is gathered on
  GPU 0 (`gather_layer_outputs`) so hooks see one device.

## Citation

arXiv id TBD.

```bibtex
@misc{padia2026qlens,
  title         = {The Q-Lens: Exact Second-Order Readouts Show That Language Models Bend Across Layers},
  author        = {Soham Padia},
  year          = {2026},
  eprint        = {TBD},
  archivePrefix = {arXiv}
}
```

## License

MIT (see `LICENSE`). The J-lens dependency is Apache-2.0 and is installed separately, not redistributed here.
