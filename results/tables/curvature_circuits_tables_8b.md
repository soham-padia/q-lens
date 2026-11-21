# Qwen3-8B (TF32): curvature circuits from A3 M[j,k] (src/curv_circuits.py)

## Gates (recomputed from the raw tensors)
| src | type | max pair gate (recomputed) | max pair gate (JSON) | max \|ΣM − ⟨cross, t̂⟩\| / \|T\| | max \|ΣM/\|T\| − cross_share (JSON)\| | lower triangle max \|M\| |
|---|---|---|---|---|---|---|
| 4 | iso | 6.3e-04 | 6.3e-04 | 5.4e-04 | 5.4e-04 | 0.0e+00 |
| 4 | data | 7.4e-04 | 7.4e-04 | 4.3e-04 | 4.3e-04 | 0.0e+00 |
| 14 | iso | 5.1e-04 | 5.1e-04 | 6.2e-04 | 6.2e-04 | 0.0e+00 |
| 14 | data | 6.5e-04 | 6.5e-04 | 4.8e-04 | 4.8e-04 | 0.0e+00 |
| 22 | iso | 5.0e-04 | 5.0e-04 | 1.6e-04 | 1.6e-04 | 0.0e+00 |
| 22 | data | 2.9e-04 | 2.9e-04 | 2.5e-04 | 2.5e-04 | 0.0e+00 |

## Concentration and cancellation of the between-block pairs (j < k), medians over 32 cases [two-way bootstrap CI]
x = M/|T|. Uniform baseline for the top-n mass share is n/N; 'Gauss' = i.i.d. Gaussian contributions of the same count (no structure).

| src | type | N pairs | between share Σx | top-1 pair \|x\| (share of \|T\|) | top-1 / top-10 / top-1% mass share | Gauss ref | pr (pr/N) | ci | √pr | ci/√pr | sign-null ci | z = Σx/‖x‖ | n90 (of N) | err after top-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | iso | 465 | 0.55 [0.36, 0.73] | 0.074 [0.053, 0.129] | 0.047 / 0.227 / 0.140 (top 5) | 0.009 / 0.070 / 0.038 | 100 (0.22; Gauss 0.64) | 3.6 [2.8, 4.4] | 10.0 [8.0, 11.4] | 0.37 [0.28, 0.48] | 14.3 | 2.7 [2.1, 3.6] | 133 | 0.58 |
| 4 | data | 465 | 0.63 [0.43, 0.79] | 0.061 [0.046, 0.083] | 0.037 / 0.198 / 0.123 (top 5) | 0.009 / 0.070 / 0.038 | 128 (0.28; Gauss 0.64) | 2.7 [2.1, 5.2] | 11.3 [8.4, 13.6] | 0.30 [0.21, 0.39] | 16.1 | 3.4 [2.7, 4.7] | 144 | 0.64 |
| 14 | iso | 210 | 0.53 [0.43, 0.71] | 0.104 [0.062, 0.159] | 0.073 / 0.333 / 0.119 (top 2) | 0.018 / 0.140 / 0.034 | 57 (0.27; Gauss 0.64) | 2.1 [1.8, 3.2] | 7.5 [4.4, 9.2] | 0.36 [0.27, 0.48] | 10.5 | 2.8 [2.1, 3.7] | 60 | 0.35 |
| 14 | data | 210 | 0.54 [0.43, 0.68] | 0.063 [0.047, 0.088] | 0.056 / 0.288 / 0.089 (top 2) | 0.018 / 0.140 / 0.034 | 69 (0.33; Gauss 0.64) | 2.1 [1.4, 2.7] | 8.3 [6.3, 9.1] | 0.27 [0.20, 0.32] | 12.2 | 3.7 [3.2, 5.1] | 74 | 0.57 |
| 22 | iso | 78 | 0.22 [0.17, 0.45] | 0.042 [0.031, 0.109] | 0.082 / 0.428 / 0.082 (top 1) | 0.042 / 0.310 / 0.042 | 35 (0.45; Gauss 0.64) | 2.6 [1.5, 5.3] | 5.9 [5.4, 6.1] | 0.41 [0.26, 0.86] | 8.2 | 2.4 [1.1, 4.0] | 38 | 0.39 |
| 22 | data | 78 | 0.49 [0.37, 0.55] | 0.063 [0.039, 0.094] | 0.087 / 0.439 / 0.087 (top 1) | 0.042 / 0.310 / 0.042 | 32 (0.40; Gauss 0.64) | 1.4 [1.2, 2.3] | 5.6 [5.2, 6.3] | 0.25 [0.20, 0.32] | 8.1 | 3.9 [3.2, 5.0] | 40 | 0.48 |

- src 4 iso: best single pair leaves 0.90 of the between-block net (top 1% of pairs: 0.69); n90 = 0.29 of pairs; negative share of |x| mass 0.36.
- src 4 data: best single pair leaves 0.91 of the between-block net (top 1% of pairs: 0.74); n90 = 0.31 of pairs; negative share of |x| mass 0.31.
- src 14 iso: best single pair leaves 0.79 of the between-block net (top 1% of pairs: 0.72); n90 = 0.29 of pairs; negative share of |x| mass 0.26.
- src 14 data: best single pair leaves 0.87 of the between-block net (top 1% of pairs: 0.78); n90 = 0.35 of pairs; negative share of |x| mass 0.26.
- src 22 iso: best single pair leaves 0.84 of the between-block net (top 1% of pairs: 0.84); n90 = 0.49 of pairs; negative share of |x| mass 0.30.
- src 22 data: best single pair leaves 0.87 of the between-block net (top 1% of pairs: 0.87); n90 = 0.51 of pairs; negative share of |x| mass 0.14.
- Pooled over 192 cases: median ci/√pr = 0.31 (IQR 0.23–0.44); median z = 3.25; fraction of |x| mass with x > 0: median 0.715; cases with z > 3: 0.56, z < 0: 0.01.

## Distance profile (k − j)
mass(d) = Σ_{k−j=d} |x| / Σ|x|; pairs(d) = fraction of pairs at distance d; intensity = mass/pairs (1 = flat); net(d) = Σ_{k−j=d} x (share of |T|). Medians over cases (iso and data pooled, 64 cases).

| src | d=1 mass / pairs / net | d≤3 mass / pairs | d≤5 mass / pairs | d≤K/4 mass / pairs | far half (d>K/2) mass / pairs | net d=1 | net d≥2 |
|---|---|---|---|---|---|---|---|
| 4 (K=31) | 0.133 / 0.065 / +0.107 | 0.300 / 0.187 | 0.434 / 0.301 | 0.532 / 0.406 | 0.139 / 0.258 | +0.107 | +0.455 |
| 14 (K=21) | 0.191 / 0.095 / +0.137 | 0.408 / 0.271 | 0.551 / 0.429 | 0.551 / 0.429 | 0.181 / 0.262 | +0.137 | +0.350 |
| 22 (K=13) | 0.199 / 0.154 / +0.037 | 0.450 / 0.423 | 0.634 / 0.641 | 0.450 / 0.423 | 0.303 / 0.269 | +0.037 | +0.323 |

## Row sums (curvature sources j) and column sums (benders k)
Row j has K−1−j between-block pairs and column k has k, so raw mass shares favour early rows and late columns by counting alone. Intensity = (Σ_k |x_jk| / Σ|x|) / ((K−1−j)/N) for rows, (Σ_j |x_jk| / Σ|x|) / (k/N) for columns; 1 = flat. pr = effective number of rows / columns carrying the mass (flat reference in brackets). Top blocks by median intensity over 64 cases; 'top-3 freq' = fraction of cases in which the block is among the 3 most intense (chance 3/(K−1)).

| src | row pr (flat) | col pr (flat) | top source blocks j: intensity, top-3 freq | top bender blocks k: intensity, top-3 freq | chance |
|---|---|---|---|---|---|
| 4 (K=31) | 13.5 (22.9) | 16.5 (22.9) | L5: 1.59, 0.47, L6: 1.31, 0.30, L7: 1.04, 0.14 | L6: 2.25, 0.45, L9: 1.65, 0.19, L16: 1.49, 0.16 | 0.10 |
| 14 (K=21) | 8.7 (15.4) | 13.3 (15.4) | L15: 2.06, 0.61, L16: 1.35, 0.44, L19: 1.08, 0.34 | L16: 9.92, 0.70, L17: 2.99, 0.52, L18: 2.73, 0.47 | 0.15 |
| 22 (K=13) | 5.8 (9.4) | 9.1 (9.4) | L23: 1.98, 0.70, L34: 1.61, 0.64, L24: 1.08, 0.38 | L24: 2.70, 0.77, L25: 2.16, 0.75, L26: 1.53, 0.48 | 0.25 |

## Per context (16 cases each: 8 iso + 8 data directions)
Top pair = the pair with the largest median |x| over the context's 16 cases, with that median (share of |T|) and the fraction of the 16 cases in which it is the single largest pair. Bender = column with the largest median mass share Σ_j|x_jk|/Σ|x|. Adjacent = mass share at k − j = 1.

| src | ctx | top pair j→k | its median \|x\| | is top-1 in | top bender k: mass share | median top-1 \|x\| | adjacent mass | between share Σx |
|---|---|---|---|---|---|---|---|---|
| 4 | 0 | L5→L17 | 0.020 | 0.12 | L17: 0.10 | 0.053 | 0.13 | 0.56 |
| 4 | 1 | L21→L22 | 0.026 | 0.31 | L33: 0.08 | 0.070 | 0.13 | 0.74 |
| 4 | 2 | L5→L6 | 0.049 | 0.31 | L9: 0.07 | 0.084 | 0.18 | 0.62 |
| 4 | 3 | L19→L20 | 0.024 | 0.12 | L24: 0.09 | 0.076 | 0.12 | 0.50 |
| 14 | 0 | L15→L16 | 0.057 | 0.50 | L21: 0.08 | 0.102 | 0.26 | 0.50 |
| 14 | 1 | L22→L33 | 0.029 | 0.12 | L33: 0.12 | 0.069 | 0.15 | 0.65 |
| 14 | 2 | L15→L16 | 0.118 | 0.75 | L16: 0.16 | 0.122 | 0.32 | 0.46 |
| 14 | 3 | L19→L34 | 0.032 | 0.12 | L34: 0.13 | 0.059 | 0.15 | 0.56 |
| 22 | 0 | L23→L24 | 0.030 | 0.25 | L34: 0.13 | 0.041 | 0.21 | 0.35 |
| 22 | 1 | L31→L35 | 0.081 | 0.25 | L35: 0.26 | 0.109 | 0.15 | 0.50 |
| 22 | 2 | L23→L24 | 0.028 | 0.44 | L34: 0.11 | 0.038 | 0.21 | 0.24 |
| 22 | 3 | L23→L24 | 0.026 | 0.44 | L35: 0.18 | 0.047 | 0.21 | 0.36 |

## Consistency: do the same pairs recur? (Spearman of |M| over the N between-block pairs)
Mean over case pairs in each relation [two-way bootstrap CI]. Nulls: 'perm' = entries of one case shuffled (mean, 99th percentile over pairs); 'strat' = shuffled within each distance d (keeps the distance profile). 'resid' = Spearman of log|M| residuals after removing row (j) and column (k) effects: pair-specific recurrence beyond which blocks are big. 'signed' = Spearman of signed M. top-10 = overlap fraction of the 10 largest-|M| pairs (perm null 10/N).

| src | relation | n pairs | Spearman \|M\| | perm null mean (p99) | strat null mean | resid Spearman | signed Spearman | top-10 overlap (strat null) | top-1% overlap |
|---|---|---|---|---|---|---|---|---|---|
| 4 | contexts (same type & dir) | 96 | 0.10 [-0.11, 0.41] | +0.001 (0.099) | 0.03 | 0.02 [-0.01, 0.05] | 0.02 [-0.02, 0.06] | 0.05 (0.03) | 0.02 (top 5; null 0.011) |
| 4 | directions (same type & ctx) | 224 | 0.34 [0.17, 0.50] | -0.001 (0.112) | 0.03 | 0.06 [0.02, 0.10] | 0.10 [0.04, 0.20] | 0.23 (0.05) | 0.19 (top 5; null 0.011) |
| 4 | iso vs data (same ctx) | 256 | 0.32 [0.16, 0.48] | +0.005 (0.110) | 0.03 | 0.05 [0.01, 0.09] | 0.10 [0.03, 0.21] | 0.23 (0.03) | 0.17 (top 5; null 0.011) |
| 4 | neither (same type) | 672 | 0.09 [-0.11, 0.42] | +0.002 (0.103) | 0.03 | 0.02 [0.00, 0.04] | 0.01 [-0.01, 0.03] | 0.05 (0.04) | 0.03 (top 5; null 0.011) |
| 14 | contexts (same type & dir) | 96 | 0.16 [0.02, 0.40] | -0.002 (0.129) | 0.01 | 0.03 [-0.02, 0.09] | 0.04 [-0.01, 0.12] | 0.19 (0.07) | 0.17 (top 2; null 0.010) |
| 14 | directions (same type & ctx) | 224 | 0.28 [0.14, 0.39] | -0.002 (0.152) | 0.02 | 0.06 [0.01, 0.12] | 0.15 [0.08, 0.24] | 0.29 (0.08) | 0.23 (top 2; null 0.010) |
| 14 | iso vs data (same ctx) | 256 | 0.26 [0.14, 0.38] | +0.003 (0.143) | 0.02 | 0.04 [0.00, 0.11] | 0.13 [0.06, 0.23] | 0.28 (0.07) | 0.24 (top 2; null 0.010) |
| 14 | neither (same type) | 672 | 0.13 [0.03, 0.36] | -0.001 (0.144) | 0.02 | 0.02 [-0.00, 0.04] | 0.03 [-0.00, 0.08] | 0.16 (0.07) | 0.13 (top 2; null 0.010) |
| 22 | contexts (same type & dir) | 96 | 0.23 [0.09, 0.39] | +0.014 (0.312) | 0.04 | 0.03 [-0.06, 0.13] | 0.21 [0.05, 0.41] | 0.34 (0.15) | 0.18 (top 1; null 0.013) |
| 22 | directions (same type & ctx) | 224 | 0.22 [0.14, 0.34] | -0.009 (0.223) | 0.03 | 0.02 [-0.04, 0.09] | 0.17 [0.10, 0.27] | 0.33 (0.15) | 0.17 (top 1; null 0.013) |
| 22 | iso vs data (same ctx) | 256 | 0.22 [0.14, 0.32] | -0.011 (0.222) | 0.02 | 0.02 [-0.01, 0.07] | 0.15 [0.06, 0.25] | 0.33 (0.16) | 0.17 (top 1; null 0.013) |
| 22 | neither (same type) | 672 | 0.14 [0.00, 0.27] | +0.001 (0.278) | 0.01 | 0.00 [-0.04, 0.05] | 0.10 [0.01, 0.21] | 0.26 (0.13) | 0.10 (top 1; null 0.013) |

### Across source layers (blocks common to both spans; same type and context)
| sources | common blocks | relation | n pairs | Spearman \|M\| | strat null | resid Spearman | signed Spearman | top-10 overlap |
|---|---|---|---|---|---|---|---|---|
| 4 vs 14 iso | 21 (L15–35) | same ctx, same dir index | 32 | 0.29 | 0.02 | 0.04 | 0.10 | 0.28 |
| 4 vs 14 iso | 21 (L15–35) | same ctx, other dir | 224 | 0.29 | 0.02 | 0.05 | 0.11 | 0.29 |
| 4 vs 14 data | 21 (L15–35) | same ctx, same dir index | 32 | 0.29 | 0.07 | 0.09 | 0.18 | 0.23 |
| 4 vs 14 data | 21 (L15–35) | same ctx, other dir | 224 | 0.21 | 0.04 | 0.05 | 0.10 | 0.21 |
| 4 vs 22 iso | 13 (L23–35) | same ctx, same dir index | 32 | 0.12 | -0.01 | -0.01 | 0.05 | 0.25 |
| 4 vs 22 iso | 13 (L23–35) | same ctx, other dir | 224 | 0.12 | 0.02 | 0.02 | 0.05 | 0.23 |
| 4 vs 22 data | 13 (L23–35) | same ctx, same dir index | 32 | 0.09 | 0.01 | -0.04 | 0.08 | 0.18 |
| 4 vs 22 data | 13 (L23–35) | same ctx, other dir | 224 | 0.10 | 0.00 | -0.01 | 0.05 | 0.21 |
| 14 vs 22 iso | 13 (L23–35) | same ctx, same dir index | 32 | 0.10 | 0.02 | 0.01 | 0.03 | 0.24 |
| 14 vs 22 iso | 13 (L23–35) | same ctx, other dir | 224 | 0.12 | 0.01 | 0.02 | 0.03 | 0.23 |
| 14 vs 22 data | 13 (L23–35) | same ctx, same dir index | 32 | 0.14 | -0.01 | 0.04 | 0.19 | 0.26 |
| 14 vs 22 data | 13 (L23–35) | same ctx, other dir | 224 | 0.10 | 0.01 | 0.01 | 0.05 | 0.19 |

## Yardstick: Q's per-block pieces in the same measures (A2 fp32; same directions; A2 contexts 0–3 = A3's)
Q scalar = block contributions projected on q̂ (like M on t̂); Q vector = C4's block-vector version. |top-1|/|net| = largest single piece over the net it belongs to (T: pair over the between-block net).

| src | type | ci/√pr: Q scalar blocks | Q vector blocks (C4) | T between-block pairs | \|top-1\|/\|net\|: Q block | T pair | pr/N: Q blocks | T pairs |
|---|---|---|---|---|---|---|---|---|
| 4 | iso | 0.38 | 1.02 | 0.37 | 0.213 | 0.154 | 0.41 | 0.22 |
| 4 | data | 0.36 | 0.99 | 0.30 | 0.220 | 0.104 | 0.44 | 0.28 |
| 14 | iso | 0.42 | 0.93 | 0.36 | 0.244 | 0.207 | 0.47 | 0.27 |
| 14 | data | 0.36 | 0.92 | 0.27 | 0.216 | 0.136 | 0.56 | 0.33 |
| 22 | iso | 0.44 | 0.91 | 0.41 | 0.319 | 0.197 | 0.56 | 0.45 |
| 22 | data | 0.41 | 0.87 | 0.25 | 0.294 | 0.127 | 0.59 | 0.40 |

Q hubs per A2 context (median over the context's 16 cases of the largest |block share of Q| along q̂, and the block that is most often the largest):

| src | ctx 0 | ctx 1 | ctx 2 | ctx 3 | ctx 4 | ctx 5 | ctx 6 | ctx 7 |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.23 (L5) | 0.22 (L22) | 0.20 (L5) | 0.18 (L23) | 0.28 (L5) | 0.31 (L5) | 0.21 (L15) | 0.16 (L6) |
| 14 | 0.27 (L15) | 0.20 (L15) | 0.25 (L15) | 0.17 (L19) | 0.25 (L21) | 0.23 (L15) | 0.23 (L15) | 0.25 (L15) |
| 22 | 0.35 (L23) | 0.35 (L23) | 0.32 (L23) | 0.32 (L23) | 0.28 (L23) | 0.27 (L23) | 0.35 (L23) | 0.23 (L23) |

Block-level consistency, like for like: Spearman of the K-vector |Q block share| (A2) and of T's bender profile (column mass Σ_j|M_jk|) and source profile (row mass), by relation; mean over case pairs (perm null p99 in brackets). Q uses contexts 0–7 (and 0–3 in the last column).

| src | relation | Q blocks, ctx 0–7 | T benders (columns) | T sources (rows) | Q blocks, ctx 0–3 |
|---|---|---|---|---|---|
| 4 | contexts (same type & dir) | 0.13 (0.44) | 0.08 (0.38) | 0.63 | 0.07 |
| 4 | directions (same type & ctx) | 0.41 (0.41) | 0.67 (0.37) | 0.74 | 0.44 |
| 4 | iso vs data (same ctx) | 0.39 (0.41) | 0.66 (0.37) | 0.73 | 0.42 |
| 4 | neither (same type) | 0.11 (0.41) | 0.08 (0.38) | 0.61 | 0.05 |
| 14 | contexts (same type & dir) | 0.34 (0.52) | 0.00 (0.47) | 0.73 | 0.39 |
| 14 | directions (same type & ctx) | 0.47 (0.49) | 0.47 (0.50) | 0.77 | 0.51 |
| 14 | iso vs data (same ctx) | 0.44 (0.52) | 0.43 (0.48) | 0.77 | 0.46 |
| 14 | neither (same type) | 0.31 (0.49) | -0.01 (0.52) | 0.71 | 0.34 |
| 22 | contexts (same type & dir) | 0.35 (0.63) | 0.45 (0.52) | 0.63 | 0.38 |
| 22 | directions (same type & ctx) | 0.62 (0.65) | 0.43 (0.67) | 0.66 | 0.72 |
| 22 | iso vs data (same ctx) | 0.59 (0.63) | 0.39 (0.71) | 0.65 | 0.68 |
| 22 | neither (same type) | 0.32 (0.63) | 0.32 (0.62) | 0.58 | 0.36 |

Figures: results/figs_curv/8b_M_heatmaps.png, results/figs_curv/8b_M_by_context.png, results/figs_curv/8b_distance_profile.png, results/figs_curv/8b_sources_benders.png, results/figs_curv/8b_cancellation.png, results/figs_curv/8b_consistency.png
