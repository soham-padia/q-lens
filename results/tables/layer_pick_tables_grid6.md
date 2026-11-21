# Layer picking from the S2 jets (original 6-point grid)

Source: 6 shard(s): t0, t1, t2, t3, t4, t5 under outputs/second_order_v1/s2
Trials: 147 two-hop swaps the model answers correctly. Single-layer grid λ ∈ [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]; band grid λ ∈ [0.25, 0.5, 1.0, 2.0].
Push units: β_l = |Δ_l|/|h_l| (relative to the clean residual norm; S2.json stores no absolute |Δ|). One layer: push = λβ_l in every unit. Band: the realised clamp pushes d_l summed as L1 = Σ|d_l|/|h_l|, L2 = root-sum-square, max = largest single-layer push.

## Resolution of the actual flip dose
Per method: trials that flip within the grid, median bracket width in λ, median width / interpolated dose.

| L16 | L20 | L24 | L28 | L32 | L36 | L40 | L44 | L48 | L52 | L56 | B16-31 | B24-39 | B32-47 | B40-55 | B24-59 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12, 0.50, 0.27 | 36, 0.50, 0.30 | 134, 0.50, 0.35 | 139, 0.25, 0.35 | 141, 0.25, 0.36 | 144, 0.25, 0.34 | 146, 0.25, 0.36 | 147, 0.25, 0.36 | 146, 0.25, 0.33 | 145, 0.50, 0.36 | 118, 0.50, 0.35 | 135, 1.00, 0.67 | 146, 0.50, 0.71 | 147, 0.50, 0.73 | 147, 0.50, 0.77 | 147, 0.50, 0.78 |

- dose, 11 single layers: the actual best is resolved by the grid brackets in 0.22 of trials.
- dose, 16 methods: the actual best is resolved by the grid brackets in 0.11 of trials.
- L1, 11 single layers: the actual best is resolved by the grid brackets in 0.01 of trials.
- L1, 16 methods: the actual best is resolved by the grid brackets in 0.01 of trials.

Not calibrated (fewer than half of the training trials flip within the grid): L16, L20
Calibration factors (median predicted/actual dose per method, fold 0 | fold 1, J+Q): L16 1.00|1.00, L20 1.00|1.00, L24 2.57|2.51, L28 2.61|2.56, L32 2.62|2.63, L36 2.19|2.20, L40 1.56|1.50, L44 1.42|1.36, L48 1.42|1.36, L52 1.22|1.16, L56 1.02|0.99, B16-31 2.31|2.10, B24-39 1.87|1.96, B32-47 1.56|1.58, B40-55 1.25|1.20, B24-59 1.10|1.11

## (i) Does the predicted cost pick the layer that actually needs the least?
Pick = argmin over the 11 single layers of the predicted dose λ* (unit dose) or push λ*·β (unit L1). Hit = the pick is the actual best layer. Regret = actual cost at the pick / actual best cost (inf if the pick does not flip within the grid). 'cal' = the per-method calibrated prediction (2-fold CV). CV-fixed = the one layer most often best on the other half, for every trial. β-only = argmin β (an equal-dose guess). Paired bootstrap over trials.

| unit | picker | hit rate | hit, resolved trials | median regret | share regret ≤ 1.1 | pick does not flip | mean log regret (inf → 5) |
|---|---|---|---|---|---|---|---|
| dose | J | 0.24 | 0.22 (n=32) | 1.25 | 0.39 | 0.01 | 0.349 |
| dose | J+Q | 0.37 | 0.34 (n=32) | 1.05 | 0.59 | 0.01 | 0.192 |
| dose | J cal | 0.39 | 0.44 (n=32) | 1.04 | 0.66 | 0.01 | 0.123 |
| dose | J+Q cal | 0.51 | 0.62 (n=32) | 1.00 | 0.74 | 0.01 | 0.096 |
| dose | CV-fixed | 0.50 | 0.59 (n=32) | 1.00 | 0.80 | 0.01 | 0.087 |
| dose | J+Q − J | +0.13 [+0.07, +0.19] | | | | | -0.16 [-0.28, -0.04] |
| dose | J+Q cal − J cal | +0.12 [+0.05, +0.18] | | | | | -0.03 [-0.04, -0.01] |
| dose | J+Q − CV-fixed | -0.13 [-0.25, +0.00] | | | | | +0.11 [+0.07, +0.15] |
| dose | J+Q cal − CV-fixed | +0.01 [-0.11, +0.12] | | | | | +0.01 [-0.01, +0.03] |
| L1 | J | 0.11 | 0.00 (n=1) | 1.29 | 0.23 | 0.01 | 0.459 |
| L1 | J+Q | 0.27 | 0.00 (n=1) | 1.10 | 0.50 | 0.02 | 0.248 |
| L1 | J cal | 0.38 | 1.00 (n=1) | 1.04 | 0.67 | 0.00 | 0.082 |
| L1 | J+Q cal | 0.46 | 1.00 (n=1) | 1.01 | 0.76 | 0.01 | 0.124 |
| L1 | CV-fixed | 0.45 | 1.00 (n=1) | 1.02 | 0.67 | 0.04 | 0.295 |
| L1 | β-only | 0.03 | 0.00 (n=1) | inf | 0.03 | 0.92 | 4.618 |
| L1 | J+Q − J | +0.16 [+0.10, +0.21] | | | | | -0.21 [-0.35, -0.06] |
| L1 | J+Q cal − J cal | +0.08 [+0.01, +0.15] | | | | | +0.04 [-0.03, +0.15] |
| L1 | J+Q − CV-fixed | -0.18 [-0.31, -0.05] | | | | | -0.05 [-0.25, +0.15] |
| L1 | J+Q cal − CV-fixed | +0.01 [-0.08, +0.10] | | | | | -0.17 [-0.31, -0.06] |

- dose: actual best: L28 2, L32 12, L36 3, L40 74, L44 41, L48 1, L56 14 | J: L28 1, L40 7, L44 33, L48 3, L52 5, L56 98 | J+Q: L28 1, L32 1, L40 27, L44 45, L48 3, L52 2, L56 68 | J+Q cal: L24 1, L28 9, L32 24, L36 24, L40 52, L44 22, L48 3, L56 12
- L1: actual best: L16 4, L20 1, L28 4, L32 66, L40 3, L44 52, L48 11, L56 6 | J: L16 1, L32 1, L44 43, L48 13, L52 4, L56 85 | J+Q: L16 5, L32 8, L40 1, L44 76, L48 15, L52 4, L56 38 | J+Q cal: L24 2, L28 8, L32 81, L40 2, L44 35, L48 13, L56 6

## (ii) Does the prediction say when a band beats every single layer?
Band wins = min over the 5 bands of the actual cost < min over the 11 single layers. Predicted band wins = the same with predicted costs. Units: dose λ (λ = 1 = the full clean swap, at one layer or clamped at every band layer), and the push at the flip in L1, L2 and max. Balanced accuracy = mean hit rate on band-wins and single-wins trials. Last two columns: the pick over all 16 methods.

| unit | actual band-wins | predictor | predicted band-wins | accuracy | balanced acc. | 16-way hit rate | 16-way median regret |
|---|---|---|---|---|---|---|---|
| dose | 0.97 | J | 0.99 | 0.97 | 0.50 | 0.71 | 1.00 |
| dose | 0.97 | J+Q | 0.99 | 0.97 | 0.50 | 0.72 | 1.00 |
| dose | 0.97 | J+Q cal | 0.91 | 0.88 | 0.45 | 0.61 | 1.00 |
| dose | | J+Q − J | | +0.00 [-0.02, +0.02] | | +0.01 [-0.01, +0.04] | |
| L1 | 0.00 | J | 0.00 | 1.00 | nan | 0.11 | 1.29 |
| L1 | 0.00 | J+Q | 0.00 | 1.00 | nan | 0.27 | 1.10 |
| L1 | 0.00 | J+Q cal | 0.00 | 1.00 | nan | 0.46 | 1.01 |
| L1 | | J+Q − J | | +0.00 [+0.00, +0.00] | | +0.16 [+0.10, +0.21] | |
| L2 | 0.95 | J | 0.96 | 0.95 | 0.68 | 0.16 | 1.18 |
| L2 | 0.95 | J+Q | 0.96 | 0.96 | 0.74 | 0.25 | 1.16 |
| L2 | 0.95 | J+Q cal | 0.93 | 0.92 | 0.66 | 0.54 | 1.00 |
| L2 | | J+Q − J | | +0.01 [+0.00, +0.03] | | +0.10 [+0.05, +0.14] | |
| max | 0.99 | J | 0.99 | 0.98 | 0.49 | 0.29 | 1.22 |
| max | 0.99 | J+Q | 0.99 | 0.98 | 0.49 | 0.42 | 1.04 |
| max | 0.99 | J+Q cal | 0.99 | 0.98 | 0.49 | 0.82 | 1.00 |
| max | | J+Q − J | | +0.00 [+0.00, +0.00] | | +0.13 [+0.08, +0.19] | |

## (iii) Does the predicted ranking follow the actual ranking? (Spearman per trial)
Censored costs (no flip within the grid, or no predicted flip) are tied at the top. 'population' = each method's mean actual rank on the other half of the trials (2-fold CV); β-only = rank by β (L1, singles).

| unit | methods | J | J+Q | J+Q cal | population | J+Q − J | J+Q cal − population | β-only |
|---|---|---|---|---|---|---|---|---|
| dose | 11 singles (n=147) | 0.59 (0.57) | 0.74 (0.74) | 0.81 (0.87) | 0.83 (0.89) | +0.15 [+0.13, +0.17] | -0.02 [-0.06, +0.01] |  |
| dose | 16 methods (n=147) | 0.72 (0.74) | 0.84 (0.86) | 0.87 (0.90) | 0.90 (0.94) | +0.12 [+0.10, +0.13] | -0.04 [-0.06, -0.02] |  |
| L1 | 11 singles (n=147) | 0.42 (0.41) | 0.61 (0.67) | 0.81 (0.86) | 0.79 (0.86) | +0.19 [+0.16, +0.22] | +0.02 [-0.02, +0.06] | -0.13 (-0.10) |
| L1 | 16 methods (n=147) | 0.48 (0.50) | 0.70 (0.78) | 0.86 (0.91) | 0.85 (0.90) | +0.22 [+0.19, +0.24] | +0.01 [-0.02, +0.03] |  |

Entries: mean (median) over trials.

## Dose error by method (median |log(predicted / actual)| over trials that flip within the grid)
| method | n flip | J | J+Q | J+Q cal | J+Q / actual, median |
|---|---|---|---|---|---|
| L16 | 12 | 2.12 | 0.83 | 0.83 | 2.30 |
| L20 | 36 | 2.02 | 0.98 | 0.98 | 2.43 |
| L24 | 134 | 2.02 | 0.95 | 0.26 | 2.51 |
| L28 | 139 | 2.12 | 0.94 | 0.27 | 2.56 |
| L32 | 141 | 2.13 | 0.97 | 0.27 | 2.62 |
| L36 | 144 | 1.83 | 0.79 | 0.26 | 2.20 |
| L40 | 146 | 1.29 | 0.41 | 0.21 | 1.51 |
| L44 | 147 | 1.03 | 0.33 | 0.23 | 1.40 |
| L48 | 146 | 1.03 | 0.32 | 0.23 | 1.38 |
| L52 | 145 | 0.68 | 0.19 | 0.18 | 1.20 |
| L56 | 118 | 0.37 | 0.12 | 0.12 | 1.01 |
| B16-31 | 135 | 1.93 | 0.83 | 0.31 | 2.21 |
| B24-39 | 146 | 1.63 | 0.67 | 0.24 | 1.95 |
| B32-47 | 147 | 1.29 | 0.45 | 0.17 | 1.56 |
| B40-55 | 147 | 0.76 | 0.21 | 0.19 | 1.21 |
| B24-59 | 147 | 0.56 | 0.21 | 0.19 | 1.10 |

## The S2 dose claim on this grid: median dose miss over all (trial, method) cases that flip
Miss = exp(median |log(predicted / actual)|) (no predicted flip = infinite miss), as summarize_s2.py. 95% CI: bootstrap over trials, 2000 resamples (each trial carries all its methods). Bias = exp(median log(pred/actual)).

| case set | n cases | J miss | J+Q miss | J+Q closer | J+Q bias | J+Q predicts no flip |
|---|---|---|---|---|---|---|
| all methods | 2030 | ×3.85 [3.48, 4.23] | ×1.64 [1.54, 1.75] | 0.95 [0.94, 0.97] | ×1.64 [1.53, 1.75] | 0.009 [0.003, 0.015] |
| all methods, actual dose ≤ 2 (the original grid's range) | 2030 | ×3.85 [3.48, 4.23] | ×1.64 [1.54, 1.75] | 0.95 [0.94, 0.97] | ×1.64 [1.53, 1.75] | 0.009 [0.003, 0.015] |
| single layers | 1308 | ×4.25 [3.78, 4.85] | ×1.72 [1.59, 1.88] | 0.97 [0.95, 0.98] | ×1.72 [1.59, 1.88] | 0.008 [0.002, 0.015] |
| bands | 722 | ×3.38 [3.04, 3.66] | ×1.54 [1.45, 1.63] | 0.93 [0.90, 0.95] | ×1.53 [1.44, 1.61] | 0.011 [0.004, 0.019] |
| L40 | 146 | ×3.64 [3.37, 4.04] | ×1.51 [1.47, 1.61] | 0.99 [0.98, 1.00] | ×1.51 [1.47, 1.61] | 0.000 [0.000, 0.000] |
| L44 | 147 | ×2.81 [2.58, 3.12] | ×1.40 [1.32, 1.48] | 0.99 [0.97, 1.00] | ×1.40 [1.31, 1.48] | 0.000 [0.000, 0.000] |
| L48 | 146 | ×2.79 [2.57, 3.14] | ×1.38 [1.30, 1.48] | 0.99 [0.97, 1.00] | ×1.38 [1.29, 1.48] | 0.000 [0.000, 0.000] |
| B40-55 | 147 | ×2.13 [1.98, 2.33] | ×1.24 [1.19, 1.31] | 0.98 [0.95, 1.00] | ×1.21 [1.18, 1.31] | 0.000 [0.000, 0.000] |

## Calibrated-J baseline: is J+Q's advantage over J more than one scale factor?
Prediction × exp(-m), m = median log(pred/actual) on the training trials (5-fold CV by trial); global = one m for all methods, per depth = one m per method (< 20 training cases: global). Miss = exp(median |log|); J/J+Q = ratio of the two misses; closer = share of cases with |log err| of J+Q < that of J (strict); bias = exp(median signed log); J over = share of cases where the J prediction exceeds the actual dose (no predicted flip counts as over). 95% CI: cluster bootstrap over trials, 2000 resamples, CV fits held fixed.

| case set, calibration | n | J miss | J+Q miss | J / J+Q | J+Q closer | J bias | J+Q bias | J over |
|---|---|---|---|---|---|---|---|---|
| actual ≤ 2, raw | 2030 | ×3.85 [3.48, 4.23] | ×1.64 [1.54, 1.75] | ×2.34 [2.24, 2.46] | 0.955 [0.938, 0.969] | ×3.85 [3.48, 4.23] | ×1.64 [1.53, 1.75] | 0.980 [0.968, 0.991] |
| actual ≤ 2, global | 2030 | ×1.81 [1.74, 1.88] | ×1.39 [1.36, 1.42] | ×1.30 [1.27, 1.34] | 0.866 [0.844, 0.885] | ×1.02 [0.92, 1.12] | ×1.01 [0.94, 1.07] | 0.505 [0.460, 0.548] |
| actual ≤ 2, depth | 2030 | ×1.45 [1.39, 1.52] | ×1.26 [1.23, 1.30] | ×1.15 [1.12, 1.18] | 0.802 [0.769, 0.835] | ×1.01 [0.93, 1.12] | ×1.01 [0.96, 1.06] | 0.508 [0.443, 0.573] |
| all flips, raw | 2030 | ×3.85 [3.48, 4.23] | ×1.64 [1.54, 1.75] | ×2.34 [2.24, 2.46] | 0.955 [0.938, 0.969] | ×3.85 [3.48, 4.23] | ×1.64 [1.53, 1.75] | 0.980 [0.968, 0.991] |
| all flips, global | 2030 | ×1.81 [1.74, 1.88] | ×1.39 [1.36, 1.42] | ×1.30 [1.27, 1.34] | 0.866 [0.844, 0.885] | ×1.02 [0.92, 1.12] | ×1.01 [0.94, 1.07] | 0.505 [0.460, 0.548] |
| all flips, depth | 2030 | ×1.45 [1.39, 1.52] | ×1.26 [1.23, 1.30] | ×1.15 [1.12, 1.18] | 0.802 [0.769, 0.835] | ×1.01 [0.93, 1.12] | ×1.01 [0.96, 1.06] | 0.508 [0.443, 0.573] |
| single layers, raw | 1308 | ×4.25 [3.78, 4.85] | ×1.72 [1.59, 1.88] | ×2.47 [2.34, 2.63] | 0.969 [0.955, 0.982] | ×4.25 [3.78, 4.85] | ×1.72 [1.59, 1.88] | 0.985 [0.976, 0.994] |
| single layers, global | 1308 | ×1.85 [1.78, 1.94] | ×1.41 [1.38, 1.45] | ×1.31 [1.27, 1.35] | 0.871 [0.847, 0.892] | ×1.12 [1.00, 1.28] | ×1.06 [0.97, 1.16] | 0.545 [0.500, 0.587] |
| single layers, depth | 1308 | ×1.45 [1.38, 1.53] | ×1.27 [1.23, 1.31] | ×1.14 [1.11, 1.17] | 0.811 [0.777, 0.845] | ×1.02 [0.93, 1.13] | ×1.01 [0.96, 1.06] | 0.511 [0.447, 0.576] |
| bands, raw | 722 | ×3.38 [3.04, 3.66] | ×1.54 [1.45, 1.63] | ×2.19 [2.05, 2.29] | 0.928 [0.904, 0.949] | ×3.38 [3.04, 3.66] | ×1.53 [1.44, 1.61] | 0.971 [0.953, 0.985] |
| bands, global | 722 | ×1.72 [1.66, 1.81] | ×1.35 [1.32, 1.38] | ×1.27 [1.24, 1.33] | 0.856 [0.829, 0.883] | ×0.89 [0.79, 0.97] | ×0.94 [0.89, 1.00] | 0.434 [0.390, 0.477] |
| bands, depth | 722 | ×1.44 [1.38, 1.52] | ×1.25 [1.21, 1.29] | ×1.16 [1.13, 1.20] | 0.787 [0.745, 0.827] | ×1.00 [0.94, 1.10] | ×1.01 [0.96, 1.07] | 0.501 [0.434, 0.568] |

Fitted factors (geometric mean over folds of exp(m), i.e. median pred/actual; per-depth n = training cases in fold 0):
| method | n flip | J factor | J+Q factor | J miss raw → depth | J+Q miss raw → depth | J+Q closer (depth) |
|---|---|---|---|---|---|---|
| global | 2030 | ×3.79 | ×1.62 | | | |
| L16 | 12 | ×3.79 (global in 5 folds) | ×1.62 | ×6.46 → ×2.30 | ×2.26 → ×1.71 | 0.75 |
| L20 | 36 | ×6.51 | ×2.45 | ×7.29 → ×2.10 | ×2.55 → ×1.69 | 0.83 |
| L24 | 134 | ×7.40 | ×2.49 | ×7.52 → ×1.54 | ×2.57 → ×1.30 | 0.74 |
| L28 | 139 | ×8.15 | ×2.57 | ×8.30 → ×1.53 | ×2.57 → ×1.31 | 0.78 |
| L32 | 141 | ×8.43 | ×2.59 | ×8.45 → ×1.63 | ×2.63 → ×1.33 | 0.79 |
| L36 | 144 | ×6.15 | ×2.20 | ×6.18 → ×1.51 | ×2.20 → ×1.30 | 0.83 |
| L40 | 146 | ×3.63 | ×1.52 | ×3.64 → ×1.40 | ×1.51 → ×1.24 | 0.86 |
| L44 | 147 | ×2.80 | ×1.39 | ×2.81 → ×1.43 | ×1.40 → ×1.26 | 0.82 |
| L48 | 146 | ×2.78 | ×1.38 | ×2.79 → ×1.40 | ×1.38 → ×1.27 | 0.82 |
| L52 | 145 | ×1.97 | ×1.20 | ×1.98 → ×1.42 | ×1.21 → ×1.22 | 0.88 |
| L56 | 118 | ×1.44 | ×1.01 | ×1.44 → ×1.27 | ×1.13 → ×1.13 | 0.77 |
| B16-31 | 135 | ×6.58 | ×2.20 | ×6.86 → ×1.61 | ×2.30 → ×1.36 | 0.72 |
| B24-39 | 146 | ×5.00 | ×1.94 | ×5.07 → ×1.47 | ×1.95 → ×1.28 | 0.76 |
| B32-47 | 147 | ×3.62 | ×1.55 | ×3.62 → ×1.43 | ×1.56 → ×1.20 | 0.82 |
| B40-55 | 147 | ×2.15 | ×1.21 | ×2.13 → ×1.37 | ×1.24 → ×1.20 | 0.82 |
| B24-59 | 147 | ×1.76 | ×1.11 | ×1.75 → ×1.42 | ×1.24 → ×1.22 | 0.82 |

