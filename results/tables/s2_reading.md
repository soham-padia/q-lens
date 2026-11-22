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
| multihop | 93 | best single layer | L62, L61 | 0.09 | 0.33 | 21 |
| multihop | 93 |   control intermediate, same layer | as above | 0.04 | 0.08 | 1516 |
| multihop | 93 | best multi-layer window | L54-61, L60-61 | 0.13 | 0.32 | 36 |
| multihop | 93 |   control intermediate, same window | as above | 0.01 | 0.04 | 3085 |
| multihop | 93 | all-layer mean | all | 0.05 | 0.16 | 336 |
| multihop | 93 | min over layers (oracle) | per item | 0.32 | 0.70 | 3 |
| multihop | 93 |   control intermediate, min over layers | per item | 0.06 | 0.15 | 249 |
| multihop | | multi − single (top-10) | | | -0.01 [-0.13, +0.10] | |


(source: outputs/second_order_v1/wsfix/S2.json; secs 129.9)
