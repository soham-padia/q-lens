## Reading: J-lens rank of the latent intermediate (last position)
Window = mean of per-layer z-scored J-lens logits over layers [a, b]. Best layer / window chosen on one half,
scored on the other (2-fold CV). min-over-layers = the J-lens paper's per-item best layer (no selection penalty);
its control is the same statistic for another item's intermediate.

| set | n | method | picked | top-1 | top-10 | median rank |
|---|---|---|---|---|---|---|
| twohop | 48 | best single layer | L40, L49 | 0.25 | 0.46 | 16 |
| twohop | 48 |   control intermediate, same layer | as above | 0.00 | 0.00 | 579 |
| twohop | 48 | best multi-layer window | L38-41, L46-53 | 0.29 | 0.42 | 18 |
| twohop | 48 |   control intermediate, same window | as above | 0.00 | 0.00 | 488 |
| twohop | 48 | all-layer mean | all | 0.12 | 0.29 | 35 |
| twohop | 48 | min over layers (oracle) | per item | 0.40 | 0.83 | 1 |
| twohop | 48 |   control intermediate, min over layers | per item | 0.00 | 0.04 | 133 |
| twohop | | multi − single (top-10) | | | -0.04 [-0.10, +0.00] | |
| multihop | 93 | best single layer | L62, L61 | 0.09 | 0.34 | 19 |
| multihop | 93 |   control intermediate, same layer | as above | 0.04 | 0.08 | 1292 |
| multihop | 93 | best multi-layer window | L54-61, L60-61 | 0.13 | 0.32 | 25 |
| multihop | 93 |   control intermediate, same window | as above | 0.01 | 0.04 | 2481 |
| multihop | 93 | all-layer mean | all | 0.05 | 0.18 | 90 |
| multihop | 93 | min over layers (oracle) | per item | 0.32 | 0.76 | 2 |
| multihop | 93 |   control intermediate, min over layers | per item | 0.06 | 0.30 | 187 |
| multihop | | multi − single (top-10) | | | -0.02 [-0.14, +0.09] | |

## One-hop swaps (J-lens paper's flexible-generalization trials), top-1 success
Clamp = coordinates set to the swapped clean-pass values at every band layer. Reswap = S1's version (swaps the
current state again at every layer).

| category | n | clean top-1 = old | best single layer @1 (CV) | best band clamp @1 (CV) | paper band clamp @1 / @2 | reswap @1 / @2 |
|---|---|---|---|---|---|---|
| animals | 1 | 1.00 | 0.00 (L16/16) | 0.00 (16-31/24-39) | 1.00 / 0.00 | 0.00 / 0.00 |
| countries | 1 | 1.00 | 0.00 (L16/24) | 1.00 (16-31/16-31) | 1.00 / 1.00 | 0.00 / 0.00 |
| months | 1 | 0.00 | 0.00 (L16/16) | 0.00 (16-31/16-31) | 0.00 / 0.00 | 0.00 / 0.00 |
| all | 3 | 0.67 | 0.00 (L16/24) | 0.33 (16-31/24-39) | 0.67 / 0.33 | 0.00 / 0.00 |

Per layer / band, all categories, success @1 / @2:
single: L16 0.00/0.33  L20 0.00/0.33  L24 0.33/0.67  L28 0.33/0.67  L32 0.33/0.67  L36 0.33/0.67  L40 0.33/0.33  L44 0.33/0.33  L48 0.33/0.33  L52 0.00/0.33  L56 0.00/0.00
band:   16-31 0.33/0.67  24-39 0.67/0.67  32-47 0.67/0.67  40-55 0.33/0.33  24-59 0.67/0.33

## Two-hop swaps (latent intermediate, e.g. 'the capital of the country famous for croissants')
- capital: model answers the two-hop prompt correctly on 2/2 trials
- language: model answers the two-hop prompt correctly on 1/1 trials

Top-1 success over the 3 trials the model answers correctly:

| method | λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 | λ=1.5 | λ=2.0 |
|---|---|---|---|---|---|---|
| single L16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| single L20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| single L24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 |
| single L28 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| single L32 | 0.00 | 0.00 | 0.00 | 0.33 | 0.67 | 1.00 |
| single L36 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| single L40 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| single L44 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| single L48 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| single L52 | 0.00 | 0.00 | 0.00 | 0.33 | 1.00 | 1.00 |
| single L56 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.67 |
| random dir (all layers) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

| band | clamp λ=0.25 | clamp λ=0.5 | clamp λ=1.0 | clamp λ=2.0 | spread λ=1.0 | spread λ=2.0 | spread λ=4.0 | spread λ=8.0 | spread λ=16.0 | ctrl pair λ=1 succ / keep | ctrl λ=2 succ / keep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16-31 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| 24-39 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.67 | 0.33 | 0.00 / 1.00 | 0.00 / 1.00 |
| 32-47 | 0.00 | 0.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 / 1.00 | 0.00 / 0.67 |
| 40-55 | 0.00 | 0.33 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 0.67 | 0.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| 24-59 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 / 1.00 | 0.00 / 1.00 |

- **λ = 1, CV-picked:** best single layer 0.67 (L40/36), best band clamp 1.00 (24-39/24-39); band − single +0.33 [+0.00, +1.00].
- **Success within a total push budget** (L1 = sum over layers of |Δ_l|/|h_l|; per trial the smallest budget at
  which the method flips the answer; method picked by CV on the fraction flipped within budget 0.2):

| method | picked | ≤0.05 | ≤0.1 | ≤0.2 | ≤0.4 | ≤0.8 | ≤1.6 |
|---|---|---|---|---|---|---|---|
| single layer | 24/28 | 0.00 | 0.00 | 0.67 | 0.67 | 0.67 | 0.67 |
| band clamp | 16-31/24-39 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| band spread | 16-31/24-39 | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 | 1.00 |

- **Bridge swap vs answer swap by layer** (success @1 / @2; the J-lens paper's control: an intermediate swap
  should work at earlier layers than swapping the answer itself):
  bridge: L16 0.00/0.00  L20 0.00/0.00  L24 0.00/0.67  L28 0.00/1.00  L32 0.33/1.00  L36 0.67/1.00  L40 1.00/1.00  L44 1.00/1.00  L48 1.00/1.00  L52 0.33/1.00  L56 0.67/0.67
  answer: L16 0.00/0.00  L20 0.00/0.00  L24 0.00/0.67  L28 0.00/1.00  L32 0.00/0.67  L36 0.00/1.00  L40 0.67/1.00  L44 0.67/1.00  L48 0.67/1.00  L52 0.33/0.67  L56 0.67/1.00

- **Does Q predict how hard to push?** Gap r(λ) = logit(new) − logit(old). J predicts the flip at r0 + g1 λ = 0,
  J+Q at r0 + g1 λ + ½ g2 λ² = 0; actual = interpolated on the λ grid. Error = |log(predicted / actual)|, over
  cases that flip within the grid (median; fraction where J+Q is closer than J):

| method | cases flipped | median err J | median err J+Q | J+Q closer | J+Q predicts no flip |
|---|---|---|---|---|---|
| single L24 | 3 | 2.98 | 1.13 | 1.00 | 0.00 |
| single L28 | 3 | 3.03 | 1.19 | 1.00 | 0.00 |
| single L32 | 3 | 3.01 | 1.21 | 1.00 | 0.00 |
| single L36 | 3 | 2.60 | 1.13 | 1.00 | 0.00 |
| single L40 | 3 | 1.92 | 0.81 | 1.00 | 0.00 |
| single L44 | 3 | 1.70 | 0.83 | 1.00 | 0.00 |
| single L48 | 3 | 1.50 | 0.73 | 1.00 | 0.00 |
| single L52 | 3 | 1.18 | 0.56 | 1.00 | 0.00 |
| single L56 | 3 | 0.55 | 0.12 | 1.00 | 0.00 |
| band 16-31 | 3 | 2.83 | 0.54 | 1.00 | 0.00 |
| band 24-39 | 3 | 2.46 | 0.96 | 1.00 | 0.00 |
| band 32-47 | 3 | 1.83 | 0.78 | 1.00 | 0.00 |
| band 40-55 | 3 | 1.22 | 0.58 | 1.00 | 0.00 |
| band 24-59 | 3 | 0.92 | 0.26 | 0.67 | 0.00 |
| all | 42 | 1.45 | 0.62 | 0.98 | 0.00 |

- **Example:** “Fact: The capital of the country famous for croissants is the city of” (clean top-5 [' Paris', '\n', ' ______', ' what', ' ____']), France→China:
  - single L40, λ=0.25: top-5 [' Paris', '\n', ' ______', ' what', ' ____'], gap -9.77
  - single L40, λ=0.5: top-5 [' Paris', '\n', ' ______', ' what', ' ____'], gap -6.53
  - single L40, λ=0.75: top-5 [' Beijing', ' Bangkok', '\n', ' ______', ' what'], gap +5.17
  - single L40, λ=1.0: top-5 [' Beijing', '\n', ' what', ' Bangkok', ' ______'], gap +9.85
  - single L40, λ=1.5: top-5 [' Beijing', '\n', ' what', ' __', ' ______'], gap +11.48
  - single L40, λ=2.0: top-5 [' Beijing', '\n', ' what', ' __', ' ______'], gap +12.10
  - band clamp 16-31, λ=1: top-5 [' Vienna', '\n', ' what', ' ______', '...'], gap -3.00, budget L1 0.139
  - band clamp 24-39, λ=1: top-5 [' Beijing', ' Bangkok', '\n', ' what', ' ______'], gap +9.58, budget L1 0.175
  - band clamp 32-47, λ=1: top-5 [' Beijing', ' Bangkok', '\n', ' what', ' __'], gap +11.07, budget L1 0.176
  - band clamp 40-55, λ=1: top-5 [' Beijing', '\n', ' what', ' __', ' ______'], gap +11.22, budget L1 0.211
  - band clamp 24-59, λ=1: top-5 [' Beijing', ' Bangkok', '\n', ' what', ' __'], gap +11.37, budget L1 0.349

(source: outputs/second_order_v1/s2_smoke_rtx/S2.json; secs 332.2)
