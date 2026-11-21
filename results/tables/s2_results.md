## One-hop swaps (J-lens paper's flexible-generalization trials), top-1 success
Clamp = coordinates set to the swapped clean-pass values at every band layer. Reswap = S1's version (swaps the
current state again at every layer).

| category | n | clean top-1 = old | best single layer @1 (CV) | best band clamp @1 (CV) | paper band clamp @1 / @2 | reswap @1 / @2 |
|---|---|---|---|---|---|---|
| animals | 48 | 0.44 | 0.02 (L40/20) | 0.23 (24-59/24-59) | 0.23 / 0.17 | 0.04 / 0.00 |
| countries | 48 | 0.75 | 0.60 (L44/40) | 0.81 (24-59/24-59) | 0.81 / 0.67 | 0.17 / 0.00 |
| months | 48 | 0.69 | 0.00 (L16/16) | 0.25 (24-59/24-59) | 0.25 / 0.42 | 0.00 / 0.00 |
| numbers | 48 | 0.44 | 0.00 (L16/16) | 0.00 (24-59/16-31) | 0.06 / 0.02 | 0.00 / 0.02 |
| all | 192 | 0.58 | 0.17 (L44/40) | 0.34 (24-59/24-59) | 0.34 / 0.32 | 0.05 / 0.01 |

Per layer / band, all categories, success @1 / @2:
single: L16 0.00/0.05  L20 0.01/0.12  L24 0.09/0.24  L28 0.14/0.32  L32 0.15/0.32  L36 0.15/0.29  L40 0.18/0.28  L44 0.17/0.28  L48 0.11/0.26  L52 0.03/0.20  L56 0.03/0.10
band:   16-31 0.11/0.32  24-39 0.22/0.42  32-47 0.26/0.43  40-55 0.24/0.31  24-59 0.34/0.32

## Two-hop swaps (latent intermediate, e.g. 'the capital of the country famous for croissants')
- capital: model answers the two-hop prompt correctly on 49/56 trials
- continent: model answers the two-hop prompt correctly on 42/42 trials
- language: model answers the two-hop prompt correctly on 56/56 trials

Top-1 success over the 147 trials the model answers correctly:

| method | λ=0.25 | λ=0.5 | λ=0.75 | λ=1.0 | λ=1.5 | λ=2.0 |
|---|---|---|---|---|---|---|
| single L16 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.06 |
| single L20 | 0.00 | 0.01 | 0.01 | 0.01 | 0.07 | 0.19 |
| single L24 | 0.01 | 0.01 | 0.08 | 0.33 | 0.68 | 0.81 |
| single L28 | 0.01 | 0.03 | 0.18 | 0.46 | 0.79 | 0.86 |
| single L32 | 0.01 | 0.03 | 0.20 | 0.50 | 0.82 | 0.87 |
| single L36 | 0.01 | 0.02 | 0.22 | 0.59 | 0.86 | 0.88 |
| single L40 | 0.01 | 0.03 | 0.53 | 0.82 | 0.93 | 0.93 |
| single L44 | 0.01 | 0.03 | 0.49 | 0.79 | 0.94 | 0.94 |
| single L48 | 0.01 | 0.01 | 0.25 | 0.71 | 0.92 | 0.95 |
| single L52 | 0.01 | 0.01 | 0.03 | 0.29 | 0.87 | 0.96 |
| single L56 | 0.01 | 0.01 | 0.10 | 0.27 | 0.48 | 0.50 |
| random dir (all layers) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

| band | clamp λ=0.25 | clamp λ=0.5 | clamp λ=1.0 | clamp λ=2.0 | spread λ=1.0 | spread λ=2.0 | spread λ=4.0 | spread λ=8.0 | spread λ=16.0 | ctrl pair λ=1 succ / keep | ctrl λ=2 succ / keep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16-31 | 0.01 | 0.03 | 0.27 | 0.86 | 0.07 | 0.62 | 0.90 | 0.94 | 0.89 | 0.01 / 0.97 | 0.03 / 0.93 |
| 24-39 | 0.01 | 0.07 | 0.85 | 0.94 | 0.40 | 0.88 | 0.93 | 0.90 | 0.76 | 0.01 / 0.97 | 0.10 / 0.83 |
| 32-47 | 0.01 | 0.18 | 0.88 | 0.93 | 0.50 | 0.92 | 0.91 | 0.86 | 0.65 | 0.03 / 0.94 | 0.13 / 0.78 |
| 40-55 | 0.01 | 0.11 | 0.93 | 0.94 | 0.63 | 0.96 | 0.90 | 0.54 | 0.05 | 0.03 / 0.95 | 0.08 / 0.85 |
| 24-59 | 0.01 | 0.24 | 0.92 | 0.93 | 0.19 | 0.91 | 0.97 | 0.67 | 0.18 | 0.03 / 0.94 | 0.10 / 0.82 |

- **λ = 1, CV-picked:** best single layer 0.82 (L40/40), best band clamp 0.93 (40-55/40-55); band − single +0.11 [+0.06, +0.16].
- **Success within a total push budget** (L1 = sum over layers of |Δ_l|/|h_l|; per trial the smallest budget at
  which the method flips the answer; method picked by CV on the fraction flipped within budget 0.2):

| method | picked | ≤0.05 | ≤0.1 | ≤0.2 | ≤0.4 | ≤0.8 | ≤1.6 |
|---|---|---|---|---|---|---|---|
| single layer | 44/48 | 0.04 | 0.69 | 0.94 | 0.94 | 0.94 | 0.94 |
| band clamp | 32-47/32-47 | 0.01 | 0.05 | 0.59 | 0.93 | 0.95 | 0.95 |
| band spread | 24-39/24-39 | 0.00 | 0.34 | 0.88 | 0.94 | 0.95 | 0.95 |

- **Bridge swap vs answer swap by layer** (success @1 / @2; the J-lens paper's control: an intermediate swap
  should work at earlier layers than swapping the answer itself):
  bridge: L16 0.01/0.06  L20 0.01/0.19  L24 0.33/0.81  L28 0.46/0.86  L32 0.50/0.87  L36 0.59/0.88  L40 0.82/0.93  L44 0.79/0.94  L48 0.71/0.95  L52 0.29/0.96  L56 0.27/0.50
  answer: L16 0.01/0.07  L20 0.02/0.10  L24 0.14/0.61  L28 0.22/0.67  L32 0.22/0.69  L36 0.29/0.73  L40 0.56/0.93  L44 0.54/0.93  L48 0.38/0.88  L52 0.30/0.88  L56 0.76/0.95

- **Does Q predict how hard to push?** Gap r(λ) = logit(new) − logit(old). J predicts the flip at r0 + g1 λ = 0,
  J+Q at r0 + g1 λ + ½ g2 λ² = 0; actual = interpolated on the λ grid. Error = |log(predicted / actual)|, over
  cases that flip within the grid (median; fraction where J+Q is closer than J):

| method | cases flipped | median err J | median err J+Q | J+Q closer | J+Q predicts no flip |
|---|---|---|---|---|---|
| single L16 | 12 | 2.12 | 0.83 | 1.00 | 0.00 |
| single L20 | 36 | 2.02 | 0.98 | 0.97 | 0.03 |
| single L24 | 134 | 2.02 | 0.95 | 0.96 | 0.04 |
| single L28 | 139 | 2.12 | 0.94 | 0.98 | 0.01 |
| single L32 | 141 | 2.13 | 0.97 | 0.99 | 0.01 |
| single L36 | 144 | 1.83 | 0.79 | 0.99 | 0.00 |
| single L40 | 146 | 1.29 | 0.41 | 0.99 | 0.00 |
| single L44 | 147 | 1.03 | 0.33 | 0.99 | 0.00 |
| single L48 | 146 | 1.03 | 0.32 | 0.99 | 0.00 |
| single L52 | 145 | 0.68 | 0.19 | 0.99 | 0.00 |
| single L56 | 118 | 0.37 | 0.12 | 0.82 | 0.00 |
| band 16-31 | 135 | 1.93 | 0.83 | 0.93 | 0.06 |
| band 24-39 | 146 | 1.63 | 0.67 | 0.99 | 0.00 |
| band 32-47 | 147 | 1.29 | 0.45 | 0.99 | 0.00 |
| band 40-55 | 147 | 0.76 | 0.21 | 0.98 | 0.00 |
| band 24-59 | 147 | 0.56 | 0.21 | 0.76 | 0.00 |
| all | 2030 | 1.35 | 0.50 | 0.95 | 0.01 |

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

(source: outputs/second_order_v1/s2/o0/S2.json, outputs/second_order_v1/s2/o1/S2.json, outputs/second_order_v1/s2/t0/S2.json, outputs/second_order_v1/s2/t1/S2.json, outputs/second_order_v1/s2/t2/S2.json, outputs/second_order_v1/s2/t3/S2.json, outputs/second_order_v1/s2/t4/S2.json, outputs/second_order_v1/s2/t5/S2.json; secs 9930.3)
