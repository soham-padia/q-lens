# Layer picking from the S2 jets (dense grid)

Source: 6 shard(s): t0, t1, t2, t3, t4, t5 under outputs/layer_pick_v1
Trials: 147 two-hop swaps the model answers correctly. Single-layer grid λ ∈ [0.1, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]; band grid λ ∈ [0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0].
Push units: β_l = |Δ_l|/|h_l| (relative to the clean residual norm; S2.json stores no absolute |Δ|). One layer: push = λβ_l in every unit. Band: the realised clamp pushes d_l summed as L1 = Σ|d_l|/|h_l|, L2 = root-sum-square, max = largest single-layer push.

## Resolution of the actual flip dose
Per method: trials that flip within the grid, median bracket width in λ, median width / interpolated dose.

| L16 | L20 | L24 | L28 | L32 | L36 | L40 | L44 | L48 | L52 | L56 | B16-31 | B24-39 | B32-47 | B40-55 | B24-59 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 44, 0.50, 0.19 | 91, 0.50, 0.18 | 142, 0.10, 0.08 | 145, 0.05, 0.07 | 145, 0.05, 0.07 | 146, 0.05, 0.07 | 146, 0.05, 0.07 | 147, 0.05, 0.07 | 147, 0.05, 0.07 | 147, 0.10, 0.07 | 141, 0.10, 0.08 | 135, 0.10, 0.11 | 146, 0.05, 0.07 | 147, 0.05, 0.08 | 147, 0.05, 0.08 | 147, 0.05, 0.09 |

- dose, 11 single layers: the actual best is resolved by the grid brackets in 0.61 of trials.
- dose, 16 methods: the actual best is resolved by the grid brackets in 0.61 of trials.
- L1, 11 single layers: the actual best is resolved by the grid brackets in 0.31 of trials.
- L1, 16 methods: the actual best is resolved by the grid brackets in 0.31 of trials.

Not calibrated (fewer than half of the training trials flip within the grid): L16
Calibration factors (median predicted/actual dose per method, fold 0 | fold 1, J+Q): L16 1.00|1.00, L20 2.87|2.39, L24 2.56|2.51, L28 2.58|2.63, L32 2.57|2.67, L36 2.12|2.23, L40 1.59|1.51, L44 1.43|1.36, L48 1.41|1.35, L52 1.22|1.17, L56 1.01|0.98, B16-31 2.36|2.10, B24-39 1.97|2.05, B32-47 1.56|1.53, B40-55 1.25|1.21, B24-59 1.11|1.11

## (i) Does the predicted cost pick the layer that actually needs the least?
Pick = argmin over the 11 single layers of the predicted dose λ* (unit dose) or push λ*·β (unit L1). Hit = the pick is the actual best layer. Regret = actual cost at the pick / actual best cost (inf if the pick does not flip within the grid). 'cal' = the per-method calibrated prediction (2-fold CV). CV-fixed = the one layer most often best on the other half, for every trial. β-only = argmin β (an equal-dose guess). Paired bootstrap over trials.

| unit | picker | hit rate | hit, resolved trials | median regret | share regret ≤ 1.1 | pick does not flip | mean log regret (inf → 5) |
|---|---|---|---|---|---|---|---|
| dose | J | 0.23 | 0.24 (n=89) | 1.24 | 0.40 | 0.00 | 0.294 |
| dose | J+Q | 0.37 | 0.39 (n=89) | 1.06 | 0.59 | 0.01 | 0.193 |
| dose | J cal | 0.41 | 0.42 (n=89) | 1.04 | 0.67 | 0.01 | 0.123 |
| dose | J+Q cal | 0.48 | 0.53 (n=89) | 1.01 | 0.73 | 0.01 | 0.098 |
| dose | CV-fixed | 0.51 | 0.60 (n=89) | 1.00 | 0.81 | 0.01 | 0.087 |
| dose | J+Q − J | +0.14 [+0.08, +0.20] | | | | | -0.10 [-0.17, -0.02] |
| dose | J+Q cal − J cal | +0.07 [+0.01, +0.14] | | | | | -0.02 [-0.04, -0.01] |
| dose | J+Q − CV-fixed | -0.14 [-0.27, -0.02] | | | | | +0.11 [+0.07, +0.15] |
| dose | J+Q cal − CV-fixed | -0.03 [-0.14, +0.07] | | | | | +0.01 [-0.01, +0.03] |
| L1 | J | 0.12 | 0.15 (n=46) | 1.29 | 0.22 | 0.00 | 0.408 |
| L1 | J+Q | 0.26 | 0.22 (n=46) | 1.11 | 0.48 | 0.01 | 0.216 |
| L1 | J cal | 0.39 | 0.46 (n=46) | 1.03 | 0.64 | 0.00 | 0.084 |
| L1 | J+Q cal | 0.50 | 0.59 (n=46) | 1.00 | 0.74 | 0.01 | 0.092 |
| L1 | CV-fixed | 0.47 | 0.59 (n=46) | 1.01 | 0.65 | 0.01 | 0.173 |
| L1 | β-only | 0.03 | 0.04 (n=46) | inf | 0.06 | 0.70 | 3.596 |
| L1 | J+Q − J | +0.14 [+0.09, +0.20] | | | | | -0.19 [-0.28, -0.08] |
| L1 | J+Q cal − J cal | +0.11 [+0.04, +0.18] | | | | | +0.01 [-0.04, +0.08] |
| L1 | J+Q − CV-fixed | -0.21 [-0.33, -0.08] | | | | | +0.04 [-0.10, +0.18] |
| L1 | J+Q cal − CV-fixed | +0.03 [-0.06, +0.13] | | | | | -0.08 [-0.17, -0.02] |

- dose: actual best: L28 2, L32 11, L36 4, L40 75, L44 41, L56 14 | J: L28 1, L40 7, L44 33, L48 3, L52 5, L56 98 | J+Q: L28 1, L32 1, L40 27, L44 45, L48 3, L52 2, L56 68 | J+Q cal: L20 2, L24 1, L28 8, L32 24, L36 22, L40 53, L44 24, L48 3, L56 10
- L1: actual best: L16 4, L20 1, L28 3, L32 69, L40 3, L44 49, L48 11, L56 7 | J: L16 1, L32 1, L44 43, L48 13, L52 4, L56 85 | J+Q: L16 5, L32 8, L40 1, L44 76, L48 15, L52 4, L56 38 | J+Q cal: L20 2, L24 2, L28 10, L32 75, L40 2, L44 39, L48 11, L56 6

## (ii) Does the prediction say when a band beats every single layer?
Band wins = min over the 5 bands of the actual cost < min over the 11 single layers. Predicted band wins = the same with predicted costs. Units: dose λ (λ = 1 = the full clean swap, at one layer or clamped at every band layer), and the push at the flip in L1, L2 and max. Balanced accuracy = mean hit rate on band-wins and single-wins trials. Last two columns: the pick over all 16 methods.

| unit | actual band-wins | predictor | predicted band-wins | accuracy | balanced acc. | 16-way hit rate | 16-way median regret |
|---|---|---|---|---|---|---|---|
| dose | 0.99 | J | 0.99 | 0.98 | 0.50 | 0.63 | 1.00 |
| dose | 0.99 | J+Q | 0.99 | 0.98 | 0.50 | 0.65 | 1.00 |
| dose | 0.99 | J+Q cal | 0.95 | 0.93 | 0.47 | 0.56 | 1.00 |
| dose | | J+Q − J | | +0.00 [-0.02, +0.02] | | +0.02 [-0.01, +0.05] | |
| L1 | 0.00 | J | 0.00 | 1.00 | nan | 0.12 | 1.29 |
| L1 | 0.00 | J+Q | 0.00 | 1.00 | nan | 0.26 | 1.11 |
| L1 | 0.00 | J+Q cal | 0.00 | 1.00 | nan | 0.50 | 1.00 |
| L1 | | J+Q − J | | +0.00 [+0.00, +0.00] | | +0.14 [+0.09, +0.20] | |
| L2 | 0.96 | J | 0.96 | 0.95 | 0.65 | 0.14 | 1.19 |
| L2 | 0.96 | J+Q | 0.96 | 0.96 | 0.74 | 0.26 | 1.14 |
| L2 | 0.96 | J+Q cal | 0.93 | 0.92 | 0.64 | 0.53 | 1.00 |
| L2 | | J+Q − J | | +0.01 [+0.00, +0.03] | | +0.12 [+0.07, +0.17] | |
| max | 0.99 | J | 0.99 | 0.98 | 0.49 | 0.29 | 1.25 |
| max | 0.99 | J+Q | 0.99 | 0.98 | 0.49 | 0.42 | 1.11 |
| max | 0.99 | J+Q cal | 0.99 | 0.98 | 0.49 | 0.82 | 1.00 |
| max | | J+Q − J | | +0.00 [+0.00, +0.00] | | +0.14 [+0.08, +0.19] | |

## (iii) Does the predicted ranking follow the actual ranking? (Spearman per trial)
Censored costs (no flip within the grid, or no predicted flip) are tied at the top. 'population' = each method's mean actual rank on the other half of the trials (2-fold CV); β-only = rank by β (L1, singles).

| unit | methods | J | J+Q | J+Q cal | population | J+Q − J | J+Q cal − population | β-only |
|---|---|---|---|---|---|---|---|---|
| dose | 11 singles (n=147) | 0.60 (0.57) | 0.74 (0.75) | 0.81 (0.88) | 0.84 (0.89) | +0.15 [+0.13, +0.17] | -0.03 [-0.07, +0.01] |  |
| dose | 16 methods (n=147) | 0.72 (0.73) | 0.84 (0.85) | 0.87 (0.91) | 0.90 (0.93) | +0.12 [+0.11, +0.13] | -0.04 [-0.06, -0.01] |  |
| L1 | 11 singles (n=147) | 0.36 (0.39) | 0.57 (0.62) | 0.76 (0.83) | 0.76 (0.84) | +0.21 [+0.18, +0.25] | +0.00 [-0.04, +0.04] | -0.06 (-0.05) |
| L1 | 16 methods (n=147) | 0.42 (0.45) | 0.68 (0.74) | 0.84 (0.88) | 0.84 (0.88) | +0.25 [+0.23, +0.28] | -0.00 [-0.02, +0.02] |  |

Entries: mean (median) over trials.

## Dose error by method (median |log(predicted / actual)| over trials that flip within the grid)
| method | n flip | J | J+Q | J+Q cal | J+Q / actual, median |
|---|---|---|---|---|---|
| L16 | 44 | 2.36 | 1.36 | 1.36 | 3.70 |
| L20 | 91 | 2.19 | 1.01 | 0.37 | 2.73 |
| L24 | 142 | 2.02 | 0.94 | 0.27 | 2.51 |
| L28 | 145 | 2.13 | 0.96 | 0.27 | 2.60 |
| L32 | 145 | 2.14 | 0.97 | 0.28 | 2.61 |
| L36 | 146 | 1.82 | 0.79 | 0.28 | 2.20 |
| L40 | 146 | 1.31 | 0.42 | 0.22 | 1.52 |
| L44 | 147 | 1.03 | 0.33 | 0.23 | 1.40 |
| L48 | 147 | 1.02 | 0.32 | 0.22 | 1.38 |
| L52 | 147 | 0.68 | 0.19 | 0.19 | 1.20 |
| L56 | 141 | 0.31 | 0.13 | 0.13 | 0.99 |
| B16-31 | 135 | 1.98 | 0.83 | 0.30 | 2.23 |
| B24-39 | 146 | 1.65 | 0.70 | 0.25 | 2.02 |
| B32-47 | 147 | 1.30 | 0.44 | 0.21 | 1.53 |
| B40-55 | 147 | 0.76 | 0.21 | 0.20 | 1.23 |
| B24-59 | 147 | 0.56 | 0.21 | 0.21 | 1.11 |

## The S2 dose claim on this grid: median dose miss over all (trial, method) cases that flip
Miss = exp(median |log(predicted / actual)|) (no predicted flip = infinite miss), as summarize_s2.py. 95% CI: bootstrap over trials, 2000 resamples (each trial carries all its methods). Bias = exp(median log(pred/actual)).

| case set | n cases | J miss | J+Q miss | J+Q closer | J+Q bias | J+Q predicts no flip |
|---|---|---|---|---|---|---|
| all methods | 2163 | ×4.05 [3.63, 4.50] | ×1.69 [1.59, 1.81] | 0.95 [0.93, 0.96] | ×1.69 [1.59, 1.80] | 0.011 [0.005, 0.018] |
| all methods, actual dose ≤ 2 (the original grid's range) | 2030 | ×3.88 [3.48, 4.29] | ×1.66 [1.55, 1.75] | 0.96 [0.94, 0.97] | ×1.65 [1.54, 1.75] | 0.009 [0.003, 0.015] |
| single layers | 1441 | ×4.56 [4.07, 5.11] | ×1.80 [1.65, 1.95] | 0.96 [0.94, 0.97] | ×1.80 [1.65, 1.95] | 0.011 [0.005, 0.019] |
| bands | 722 | ×3.42 [3.06, 3.66] | ×1.57 [1.49, 1.66] | 0.93 [0.91, 0.95] | ×1.55 [1.47, 1.65] | 0.011 [0.004, 0.019] |
| L40 | 146 | ×3.67 [3.39, 4.02] | ×1.52 [1.47, 1.62] | 0.99 [0.98, 1.00] | ×1.52 [1.47, 1.62] | 0.000 [0.000, 0.000] |
| L44 | 147 | ×2.81 [2.57, 3.12] | ×1.40 [1.32, 1.49] | 0.99 [0.97, 1.00] | ×1.40 [1.32, 1.49] | 0.000 [0.000, 0.000] |
| L48 | 147 | ×2.78 [2.57, 3.04] | ×1.38 [1.30, 1.47] | 0.99 [0.97, 1.00] | ×1.38 [1.29, 1.47] | 0.000 [0.000, 0.000] |
| B40-55 | 147 | ×2.15 [1.98, 2.39] | ×1.23 [1.18, 1.31] | 0.98 [0.95, 1.00] | ×1.23 [1.17, 1.30] | 0.000 [0.000, 0.000] |

Original-grid interpolated dose against this grid's, same (trial, method), both finite: single: n=1308, median ratio ×1.001, median |log| 0.008, 90th pct |log| 0.033; band: n=722, median ratio ×1.005, median |log| 0.027, 90th pct |log| 0.094

## Calibrated-J baseline: is J+Q's advantage over J more than one scale factor?
Prediction × exp(-m), m = median log(pred/actual) on the training trials (5-fold CV by trial); global = one m for all methods, per depth = one m per method (< 20 training cases: global). Miss = exp(median |log|); J/J+Q = ratio of the two misses; closer = share of cases with |log err| of J+Q < that of J (strict); bias = exp(median signed log); J over = share of cases where the J prediction exceeds the actual dose (no predicted flip counts as over). 95% CI: cluster bootstrap over trials, 2000 resamples, CV fits held fixed.

| case set, calibration | n | J miss | J+Q miss | J / J+Q | J+Q closer | J bias | J+Q bias | J over |
|---|---|---|---|---|---|---|---|---|
| actual ≤ 2, raw | 2030 | ×3.88 [3.48, 4.29] | ×1.66 [1.55, 1.75] | ×2.34 [2.23, 2.46] | 0.957 [0.941, 0.970] | ×3.88 [3.48, 4.29] | ×1.65 [1.54, 1.75] | 0.979 [0.968, 0.989] |
| actual ≤ 2, global | 2030 | ×1.81 [1.75, 1.90] | ×1.40 [1.37, 1.44] | ×1.29 [1.27, 1.33] | 0.864 [0.842, 0.883] | ×0.97 [0.88, 1.08] | ×0.99 [0.92, 1.05] | 0.489 [0.444, 0.532] |
| actual ≤ 2, depth | 2030 | ×1.45 [1.39, 1.54] | ×1.26 [1.23, 1.31] | ×1.15 [1.12, 1.19] | 0.800 [0.765, 0.834] | ×1.00 [0.93, 1.11] | ×1.00 [0.95, 1.06] | 0.503 [0.438, 0.567] |
| all flips, raw | 2163 | ×4.05 [3.63, 4.50] | ×1.69 [1.59, 1.81] | ×2.39 [2.27, 2.52] | 0.950 [0.935, 0.963] | ×4.05 [3.63, 4.50] | ×1.69 [1.59, 1.80] | 0.977 [0.966, 0.987] |
| all flips, global | 2163 | ×1.84 [1.77, 1.94] | ×1.41 [1.38, 1.45] | ×1.30 [1.27, 1.35] | 0.863 [0.842, 0.883] | ×1.02 [0.91, 1.13] | ×1.01 [0.95, 1.08] | 0.506 [0.464, 0.546] |
| all flips, depth | 2163 | ×1.46 [1.40, 1.55] | ×1.27 [1.24, 1.31] | ×1.15 [1.12, 1.19] | 0.793 [0.757, 0.827] | ×1.00 [0.94, 1.11] | ×1.00 [0.95, 1.06] | 0.503 [0.441, 0.566] |
| single layers, raw | 1441 | ×4.56 [4.07, 5.11] | ×1.80 [1.65, 1.95] | ×2.53 [2.42, 2.68] | 0.958 [0.944, 0.971] | ×4.56 [4.07, 5.11] | ×1.80 [1.65, 1.95] | 0.982 [0.973, 0.990] |
| single layers, global | 1441 | ×1.89 [1.80, 2.00] | ×1.44 [1.40, 1.49] | ×1.31 [1.27, 1.37] | 0.867 [0.842, 0.889] | ×1.16 [1.02, 1.29] | ×1.08 [0.98, 1.17] | 0.552 [0.508, 0.591] |
| single layers, depth | 1441 | ×1.46 [1.41, 1.56] | ×1.28 [1.24, 1.33] | ×1.14 [1.11, 1.18] | 0.800 [0.766, 0.833] | ×1.01 [0.94, 1.12] | ×1.00 [0.96, 1.06] | 0.505 [0.442, 0.567] |
| bands, raw | 722 | ×3.42 [3.06, 3.66] | ×1.57 [1.49, 1.66] | ×2.18 [2.05, 2.28] | 0.932 [0.910, 0.951] | ×3.42 [3.06, 3.66] | ×1.55 [1.47, 1.65] | 0.968 [0.952, 0.982] |
| bands, global | 722 | ×1.76 [1.68, 1.85] | ×1.37 [1.33, 1.40] | ×1.29 [1.25, 1.33] | 0.855 [0.826, 0.880] | ×0.86 [0.77, 0.93] | ×0.93 [0.88, 0.99] | 0.416 [0.371, 0.461] |
| bands, depth | 722 | ×1.45 [1.38, 1.53] | ×1.25 [1.23, 1.30] | ×1.15 [1.11, 1.20] | 0.778 [0.735, 0.820] | ×1.00 [0.92, 1.09] | ×1.00 [0.95, 1.07] | 0.500 [0.434, 0.567] |

Fitted factors (geometric mean over folds of exp(m), i.e. median pred/actual; per-depth n = training cases in fold 0):
| method | n flip | J factor | J+Q factor | J miss raw → depth | J+Q miss raw → depth | J+Q closer (depth) |
|---|---|---|---|---|---|---|
| global | 2163 | ×3.98 | ×1.67 | | | |
| L16 | 44 | ×9.23 | ×3.44 | ×10.09 → ×2.17 | ×3.84 → ×1.71 | 0.68 |
| L20 | 91 | ×8.59 | ×2.66 | ×8.97 → ×1.68 | ×2.75 → ×1.40 | 0.79 |
| L24 | 142 | ×7.43 | ×2.50 | ×7.49 → ×1.53 | ×2.57 → ×1.31 | 0.74 |
| L28 | 145 | ×8.30 | ×2.60 | ×8.40 → ×1.52 | ×2.61 → ×1.33 | 0.79 |
| L32 | 145 | ×8.53 | ×2.60 | ×8.48 → ×1.64 | ×2.62 → ×1.34 | 0.81 |
| L36 | 146 | ×6.18 | ×2.20 | ×6.16 → ×1.51 | ×2.20 → ×1.32 | 0.81 |
| L40 | 146 | ×3.65 | ×1.53 | ×3.67 → ×1.41 | ×1.52 → ×1.24 | 0.86 |
| L44 | 147 | ×2.82 | ×1.40 | ×2.81 → ×1.43 | ×1.40 → ×1.26 | 0.83 |
| L48 | 147 | ×2.78 | ×1.38 | ×2.78 → ×1.43 | ×1.38 → ×1.26 | 0.82 |
| L52 | 147 | ×1.97 | ×1.19 | ×1.98 → ×1.42 | ×1.21 → ×1.21 | 0.88 |
| L56 | 141 | ×1.37 | ×0.99 | ×1.36 → ×1.26 | ×1.14 → ×1.15 | 0.70 |
| B16-31 | 135 | ×7.13 | ×2.21 | ×7.24 → ×1.63 | ×2.29 → ×1.32 | 0.69 |
| B24-39 | 146 | ×5.12 | ×2.01 | ×5.18 → ×1.49 | ×2.02 → ×1.29 | 0.73 |
| B32-47 | 147 | ×3.64 | ×1.55 | ×3.65 → ×1.40 | ×1.56 → ×1.24 | 0.81 |
| B40-55 | 147 | ×2.16 | ×1.22 | ×2.15 → ×1.38 | ×1.23 → ×1.21 | 0.84 |
| B24-59 | 147 | ×1.74 | ×1.10 | ×1.74 → ×1.43 | ×1.24 → ×1.23 | 0.82 |

