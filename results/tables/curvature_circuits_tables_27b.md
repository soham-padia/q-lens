# Qwen3.8-27B (fp32): curvature circuits from A3 M[j,k] (src/curv_circuits.py)

## Gates (recomputed from the raw tensors)
| src | type | max pair gate (recomputed) | max pair gate (JSON) | max \|ΣM − ⟨cross, t̂⟩\| / \|T\| | max \|ΣM/\|T\| − cross_share (JSON)\| | lower triangle max \|M\| |
|---|---|---|---|---|---|---|
| 8 | iso | 1.2e-03 | 1.2e-03 | 4.6e-04 | 4.6e-04 | 0.0e+00 |
| 8 | data | 1.6e-03 | 1.6e-03 | 4.5e-04 | 4.5e-04 | 0.0e+00 |
| 24 | iso | 3.8e-04 | 3.8e-04 | 1.3e-04 | 1.3e-04 | 0.0e+00 |
| 24 | data | 3.5e-04 | 3.5e-04 | 2.7e-04 | 2.7e-04 | 0.0e+00 |
| 40 | iso | 3.5e-04 | 3.5e-04 | 2.4e-04 | 2.4e-04 | 0.0e+00 |
| 40 | data | 3.4e-04 | 3.4e-04 | 2.0e-04 | 2.0e-04 | 0.0e+00 |

## Concentration and cancellation of the between-block pairs (j < k), medians over 32 cases [two-way bootstrap CI]
x = M/|T|. Uniform baseline for the top-n mass share is n/N; 'Gauss' = i.i.d. Gaussian contributions of the same count (no structure).

| src | type | N pairs | between share Σx | top-1 pair \|x\| (share of \|T\|) | top-1 / top-10 / top-1% mass share | Gauss ref | pr (pr/N) | ci | √pr | ci/√pr | sign-null ci | z = Σx/‖x‖ | n90 (of N) | err after top-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | iso | 1485 | 0.52 [0.27, 0.59] | 0.069 [0.028, 0.567] | 0.033 / 0.174 / 0.217 (top 15) | 0.003 / 0.025 / 0.036 | 167 (0.11; Gauss 0.64) | 3.5 [1.5, 7.4] | 12.9 [2.0, 18.7] | 0.72 [0.17, 1.03] | 18.6 | 1.4 [1.0, 6.0] | 244 | 0.81 |
| 8 | data | 1485 | 0.43 [0.17, 0.60] | 0.062 [0.030, 0.493] | 0.066 / 0.255 / 0.313 (top 15) | 0.003 / 0.025 / 0.036 | 64 (0.04; Gauss 0.64) | 4.1 [1.5, 6.4] | 8.0 [2.1, 17.4] | 0.58 [0.22, 1.01] | 11.6 | 1.7 [1.0, 4.9] | 90 | 0.75 |
| 24 | iso | 741 | 0.56 [0.22, 0.62] | 0.062 [0.024, 0.582] | 0.059 / 0.281 / 0.227 (top 7) | 0.006 / 0.047 / 0.034 | 70 (0.09; Gauss 0.64) | 2.3 [1.8, 7.2] | 8.3 [2.8, 14.7] | 0.43 [0.14, 1.08] | 11.5 | 2.3 [0.8, 7.9] | 194 | 0.75 |
| 24 | data | 741 | 0.56 [0.31, 0.63] | 0.054 [0.025, 0.558] | 0.043 / 0.228 / 0.186 (top 7) | 0.006 / 0.047 / 0.034 | 105 (0.14; Gauss 0.64) | 2.4 [1.5, 3.4] | 10.2 [1.8, 15.0] | 0.33 [0.15, 0.99] | 15.1 | 3.0 [1.0, 6.3] | 135 | 0.77 |
| 40 | iso | 253 | 0.58 [0.32, 0.86] | 0.096 [0.048, 0.433] | 0.094 / 0.348 / 0.178 (top 3) | 0.015 / 0.118 / 0.041 | 44 (0.17; Gauss 0.64) | 1.6 [1.3, 2.5] | 6.6 [2.5, 8.2] | 0.35 [0.23, 0.70] | 9.4 | 2.9 [1.4, 4.4] | 58 | 0.50 |
| 40 | data | 253 | 0.59 [0.45, 0.76] | 0.063 [0.037, 0.331] | 0.066 / 0.312 / 0.145 (top 3) | 0.015 / 0.118 / 0.041 | 58 (0.23; Gauss 0.64) | 1.4 [1.2, 1.8] | 7.6 [2.9, 9.6] | 0.22 [0.13, 0.77] | 11.0 | 4.6 [1.3, 7.7] | 78 | 0.63 |

- src 8 iso: best single pair leaves 0.95 of the between-block net (top 1% of pairs: 0.73); n90 = 0.16 of pairs; negative share of |x| mass 0.36.
- src 8 data: best single pair leaves 0.96 of the between-block net (top 1% of pairs: 0.64); n90 = 0.06 of pairs; negative share of |x| mass 0.38.
- src 24 iso: best single pair leaves 0.94 of the between-block net (top 1% of pairs: 0.78); n90 = 0.26 of pairs; negative share of |x| mass 0.29.
- src 24 data: best single pair leaves 0.95 of the between-block net (top 1% of pairs: 0.82); n90 = 0.18 of pairs; negative share of |x| mass 0.29.
- src 40 iso: best single pair leaves 0.85 of the between-block net (top 1% of pairs: 0.74); n90 = 0.23 of pairs; negative share of |x| mass 0.19.
- src 40 data: best single pair leaves 0.93 of the between-block net (top 1% of pairs: 0.84); n90 = 0.31 of pairs; negative share of |x| mass 0.14.
- Pooled over 192 cases: median ci/√pr = 0.38 (IQR 0.18–0.93); median z = 2.66; fraction of |x| mass with x > 0: median 0.744; cases with z > 3: 0.45, z < 0: 0.01.

## Distance profile (k − j)
mass(d) = Σ_{k−j=d} |x| / Σ|x|; pairs(d) = fraction of pairs at distance d; intensity = mass/pairs (1 = flat); net(d) = Σ_{k−j=d} x (share of |T|). Medians over cases (iso and data pooled, 64 cases).

| src | d=1 mass / pairs / net | d≤3 mass / pairs | d≤5 mass / pairs | d≤K/4 mass / pairs | far half (d>K/2) mass / pairs | net d=1 | net d≥2 |
|---|---|---|---|---|---|---|---|
| 8 (K=55) | 0.139 / 0.036 / +0.149 | 0.296 / 0.107 | 0.402 / 0.175 | 0.680 / 0.420 | 0.122 / 0.255 | +0.149 | +0.256 |
| 24 (K=39) | 0.157 / 0.051 / +0.145 | 0.296 / 0.150 | 0.411 / 0.243 | 0.612 / 0.413 | 0.132 / 0.256 | +0.145 | +0.321 |
| 40 (K=23) | 0.193 / 0.087 / +0.097 | 0.366 / 0.249 | 0.509 / 0.395 | 0.509 / 0.395 | 0.212 / 0.261 | +0.097 | +0.442 |

## Row sums (curvature sources j) and column sums (benders k)
Row j has K−1−j between-block pairs and column k has k, so raw mass shares favour early rows and late columns by counting alone. Intensity = (Σ_k |x_jk| / Σ|x|) / ((K−1−j)/N) for rows, (Σ_j |x_jk| / Σ|x|) / (k/N) for columns; 1 = flat. pr = effective number of rows / columns carrying the mass (flat reference in brackets). Top blocks by median intensity over 64 cases; 'top-3 freq' = fraction of cases in which the block is among the 3 most intense (chance 3/(K−1)).

| src | row pr (flat) | col pr (flat) | top source blocks j: intensity, top-3 freq | top bender blocks k: intensity, top-3 freq | chance |
|---|---|---|---|---|---|
| 8 (K=55) | 19.3 (40.9) | 12.0 (40.9) | L61: 6.77, 0.44, L62: 6.29, 0.58, L43: 1.11, 0.06 | L62: 0.94, 0.45, L43: 0.79, 0.09, L63: 0.78, 0.45 | 0.06 |
| 24 (K=39) | 16.3 (28.9) | 8.5 (28.9) | L61: 2.82, 0.39, L62: 2.26, 0.52, L46: 0.88, 0.06 | L63: 0.90, 0.45, L47: 0.78, 0.08, L50: 0.64, 0.17 | 0.08 |
| 40 (K=23) | 9.5 (16.9) | 6.9 (16.9) | L62: 1.77, 0.45, L59: 0.96, 0.31, L61: 0.92, 0.20 | L63: 0.98, 0.28, L59: 0.91, 0.31, L60: 0.74, 0.38 | 0.14 |

## Per context (16 cases each: 8 iso + 8 data directions)
Top pair = the pair with the largest median |x| over the context's 16 cases, with that median (share of |T|) and the fraction of the 16 cases in which it is the single largest pair. Bender = column with the largest median mass share Σ_j|x_jk|/Σ|x|. Adjacent = mass share at k − j = 1.

| src | ctx | top pair j→k | its median \|x\| | is top-1 in | top bender k: mass share | median top-1 \|x\| | adjacent mass | between share Σx |
|---|---|---|---|---|---|---|---|---|
| 8 | 0 | L31→L32 | 0.025 | 0.56 | L48: 0.04 | 0.029 | 0.12 | 0.49 |
| 8 | 1 | L40→L42 | 0.070 | 0.44 | L42: 0.12 | 0.078 | 0.11 | 0.42 |
| 8 | 2 | L62→L63 | 0.572 | 1.00 | L63: 0.95 | 0.572 | 0.70 | 0.57 |
| 8 | 3 | L58→L59 | 0.058 | 0.62 | L63: 0.24 | 0.058 | 0.16 | 0.15 |
| 24 | 0 | L25→L26 | 0.015 | 0.12 | L43: 0.07 | 0.022 | 0.15 | 0.59 |
| 24 | 1 | L49→L50 | 0.039 | 0.38 | L50: 0.11 | 0.053 | 0.15 | 0.58 |
| 24 | 2 | L62→L63 | 0.578 | 0.94 | L63: 0.96 | 0.578 | 0.68 | 0.56 |
| 24 | 3 | L58→L59 | 0.054 | 0.44 | L63: 0.17 | 0.059 | 0.14 | 0.21 |
| 40 | 0 | L41→L42 | 0.019 | 0.31 | L63: 0.08 | 0.052 | 0.19 | 0.38 |
| 40 | 1 | L49→L50 | 0.095 | 0.62 | L50: 0.27 | 0.095 | 0.22 | 0.75 |
| 40 | 2 | L62→L63 | 0.479 | 0.62 | L63: 0.95 | 0.537 | 0.48 | 0.62 |
| 40 | 3 | L58→L59 | 0.045 | 0.62 | L59: 0.27 | 0.053 | 0.12 | 0.50 |

## Consistency: do the same pairs recur? (Spearman of |M| over the N between-block pairs)
Mean over case pairs in each relation [two-way bootstrap CI]. Nulls: 'perm' = entries of one case shuffled (mean, 99th percentile over pairs); 'strat' = shuffled within each distance d (keeps the distance profile). 'resid' = Spearman of log|M| residuals after removing row (j) and column (k) effects: pair-specific recurrence beyond which blocks are big. 'signed' = Spearman of signed M. top-10 = overlap fraction of the 10 largest-|M| pairs (perm null 10/N).

| src | relation | n pairs | Spearman \|M\| | perm null mean (p99) | strat null mean | resid Spearman | signed Spearman | top-10 overlap (strat null) | top-1% overlap |
|---|---|---|---|---|---|---|---|---|---|
| 8 | contexts (same type & dir) | 96 | 0.01 [-0.35, 0.20] | +0.002 (0.065) | -0.00 | 0.03 [0.01, 0.05] | 0.00 [-0.02, 0.03] | 0.04 (0.02) | 0.03 (top 15; null 0.010) |
| 8 | directions (same type & ctx) | 224 | 0.52 [0.37, 0.72] | -0.002 (0.054) | 0.07 | 0.14 [0.07, 0.20] | 0.21 [0.05, 0.40] | 0.39 (0.05) | 0.40 (top 15; null 0.010) |
| 8 | iso vs data (same ctx) | 256 | 0.48 [0.32, 0.70] | -0.001 (0.049) | 0.06 | 0.12 [0.07, 0.17] | 0.20 [0.08, 0.37] | 0.38 (0.06) | 0.38 (top 15; null 0.010) |
| 8 | neither (same type) | 672 | 0.01 [-0.36, 0.18] | -0.002 (0.058) | 0.00 | 0.03 [0.01, 0.05] | 0.01 [-0.01, 0.03] | 0.04 (0.02) | 0.03 (top 15; null 0.010) |
| 24 | contexts (same type & dir) | 96 | -0.01 [-0.33, 0.26] | +0.001 (0.090) | -0.00 | 0.00 [-0.02, 0.03] | 0.01 [-0.02, 0.04] | 0.04 (0.03) | 0.04 (top 7; null 0.009) |
| 24 | directions (same type & ctx) | 224 | 0.46 [0.34, 0.59] | +0.002 (0.072) | 0.06 | 0.07 [0.03, 0.12] | 0.18 [0.09, 0.30] | 0.32 (0.06) | 0.30 (top 7; null 0.009) |
| 24 | iso vs data (same ctx) | 256 | 0.46 [0.35, 0.60] | -0.001 (0.087) | 0.06 | 0.07 [0.03, 0.12] | 0.19 [0.12, 0.31] | 0.32 (0.06) | 0.30 (top 7; null 0.009) |
| 24 | neither (same type) | 672 | -0.02 [-0.32, 0.24] | +0.001 (0.083) | -0.00 | 0.02 [-0.00, 0.04] | 0.01 [-0.01, 0.03] | 0.04 (0.03) | 0.04 (top 7; null 0.009) |
| 40 | contexts (same type & dir) | 96 | 0.01 [-0.26, 0.22] | -0.002 (0.144) | 0.01 | 0.00 [-0.05, 0.06] | -0.02 [-0.14, 0.12] | 0.06 (0.05) | 0.04 (top 3; null 0.012) |
| 40 | directions (same type & ctx) | 224 | 0.41 [0.23, 0.60] | -0.005 (0.139) | 0.05 | 0.10 [0.04, 0.20] | 0.27 [0.15, 0.48] | 0.35 (0.09) | 0.26 (top 3; null 0.012) |
| 40 | iso vs data (same ctx) | 256 | 0.41 [0.21, 0.62] | +0.003 (0.137) | 0.07 | 0.10 [0.03, 0.18] | 0.26 [0.13, 0.50] | 0.36 (0.08) | 0.30 (top 3; null 0.012) |
| 40 | neither (same type) | 672 | 0.01 [-0.23, 0.18] | +0.001 (0.171) | 0.01 | 0.01 [-0.02, 0.04] | -0.02 [-0.11, 0.11] | 0.05 (0.05) | 0.03 (top 3; null 0.012) |

### Across source layers (blocks common to both spans; same type and context)
| sources | common blocks | relation | n pairs | Spearman \|M\| | strat null | resid Spearman | signed Spearman | top-10 overlap |
|---|---|---|---|---|---|---|---|---|
| 8 vs 24 iso | 39 (L25–63) | same ctx, same dir index | 32 | 0.45 | 0.04 | 0.08 | 0.16 | 0.28 |
| 8 vs 24 iso | 39 (L25–63) | same ctx, other dir | 224 | 0.46 | 0.05 | 0.08 | 0.15 | 0.29 |
| 8 vs 24 data | 39 (L25–63) | same ctx, same dir index | 32 | 0.44 | 0.05 | 0.10 | 0.20 | 0.34 |
| 8 vs 24 data | 39 (L25–63) | same ctx, other dir | 224 | 0.43 | 0.05 | 0.08 | 0.16 | 0.31 |
| 8 vs 40 iso | 23 (L41–63) | same ctx, same dir index | 32 | 0.38 | 0.05 | 0.07 | 0.13 | 0.30 |
| 8 vs 40 iso | 23 (L41–63) | same ctx, other dir | 224 | 0.33 | 0.05 | 0.04 | 0.11 | 0.27 |
| 8 vs 40 data | 23 (L41–63) | same ctx, same dir index | 32 | 0.36 | 0.04 | 0.06 | 0.15 | 0.29 |
| 8 vs 40 data | 23 (L41–63) | same ctx, other dir | 224 | 0.37 | 0.05 | 0.05 | 0.16 | 0.31 |
| 24 vs 40 iso | 23 (L41–63) | same ctx, same dir index | 32 | 0.36 | 0.04 | 0.05 | 0.13 | 0.28 |
| 24 vs 40 iso | 23 (L41–63) | same ctx, other dir | 224 | 0.39 | 0.04 | 0.06 | 0.16 | 0.31 |
| 24 vs 40 data | 23 (L41–63) | same ctx, same dir index | 32 | 0.39 | 0.04 | 0.09 | 0.20 | 0.32 |
| 24 vs 40 data | 23 (L41–63) | same ctx, other dir | 224 | 0.39 | 0.05 | 0.07 | 0.21 | 0.32 |

## Yardstick: Q's per-block pieces in the same measures (A2 fp32; same directions; A2 contexts 0–3 = A3's)
Q scalar = block contributions projected on q̂ (like M on t̂); Q vector = C4's block-vector version. |top-1|/|net| = largest single piece over the net it belongs to (T: pair over the between-block net).

| src | type | ci/√pr: Q scalar blocks | Q vector blocks (C4) | T between-block pairs | \|top-1\|/\|net\|: Q block | T pair | pr/N: Q blocks | T pairs |
|---|---|---|---|---|---|---|---|---|
| 8 | iso | 0.28 | 0.93 | 0.72 | 0.150 | 0.321 | 0.40 | 0.11 |
| 8 | data | 0.30 | 0.92 | 0.58 | 0.203 | 0.314 | 0.31 | 0.04 |
| 24 | iso | 0.28 | 1.01 | 0.43 | 0.165 | 0.283 | 0.43 | 0.09 |
| 24 | data | 0.28 | 0.96 | 0.33 | 0.171 | 0.195 | 0.44 | 0.14 |
| 40 | iso | 0.32 | 0.98 | 0.35 | 0.194 | 0.230 | 0.49 | 0.17 |
| 40 | data | 0.28 | 0.90 | 0.22 | 0.162 | 0.113 | 0.60 | 0.23 |

Q hubs per A2 context (median over the context's 16 cases of the largest |block share of Q| along q̂, and the block that is most often the largest):

| src | ctx 0 | ctx 1 | ctx 2 | ctx 3 | ctx 4 | ctx 5 | ctx 6 | ctx 7 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.09 (L9) | 0.12 (L42) | 0.64 (L63) | 0.34 (L59) | 0.47 (L62) | 0.12 (L9) | 0.13 (L9) | 0.10 (L60) |
| 24 | 0.16 (L25) | 0.15 (L49) | 0.66 (L63) | 0.34 (L59) | 0.52 (L62) | 0.12 (L25) | 0.13 (L25) | 0.10 (L25) |
| 40 | 0.18 (L41) | 0.17 (L42) | 0.60 (L63) | 0.34 (L59) | 0.24 (L62) | 0.12 (L42) | 0.19 (L41) | 0.14 (L41) |

Block-level consistency, like for like: Spearman of the K-vector |Q block share| (A2) and of T's bender profile (column mass Σ_j|M_jk|) and source profile (row mass), by relation; mean over case pairs (perm null p99 in brackets). Q uses contexts 0–7 (and 0–3 in the last column).

| src | relation | Q blocks, ctx 0–7 | T benders (columns) | T sources (rows) | Q blocks, ctx 0–3 |
|---|---|---|---|---|---|
| 8 | contexts (same type & dir) | 0.04 (0.31) | 0.37 (0.36) | -0.06 | -0.02 |
| 8 | directions (same type & ctx) | 0.41 (0.29) | 0.75 (0.29) | 0.60 | 0.45 |
| 8 | iso vs data (same ctx) | 0.40 (0.34) | 0.70 (0.28) | 0.59 | 0.46 |
| 8 | neither (same type) | 0.05 (0.33) | 0.36 (0.32) | -0.05 | -0.01 |
| 24 | contexts (same type & dir) | 0.07 (0.35) | 0.25 (0.42) | 0.05 | 0.03 |
| 24 | directions (same type & ctx) | 0.44 (0.35) | 0.69 (0.33) | 0.60 | 0.49 |
| 24 | iso vs data (same ctx) | 0.45 (0.41) | 0.67 (0.34) | 0.61 | 0.49 |
| 24 | neither (same type) | 0.05 (0.38) | 0.24 (0.39) | 0.03 | 0.02 |
| 40 | contexts (same type & dir) | 0.14 (0.45) | 0.33 (0.50) | 0.14 | -0.01 |
| 40 | directions (same type & ctx) | 0.53 (0.52) | 0.58 (0.48) | 0.56 | 0.52 |
| 40 | iso vs data (same ctx) | 0.46 (0.47) | 0.57 (0.49) | 0.57 | 0.48 |
| 40 | neither (same type) | 0.12 (0.49) | 0.33 (0.49) | 0.12 | 0.02 |

Figures: results/figs_curv/27b_M_heatmaps.png, results/figs_curv/27b_M_by_context.png, results/figs_curv/27b_distance_profile.png, results/figs_curv/27b_sources_benders.png, results/figs_curv/27b_cancellation.png, results/figs_curv/27b_consistency.png
