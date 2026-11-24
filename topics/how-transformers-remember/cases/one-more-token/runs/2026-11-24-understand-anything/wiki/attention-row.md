# Read the same cache two ways

A new query changes the mixture of values, even when the cache stays fixed. In a two-position toy row, changing the query from 1 to 0 changes the attention output from 8 to 6.

The scaled score is q·kᵢ/√d. Softmax turns these scores into weights that sum to one. Attention returns the sum of each weight times its value.

Take d = 1, keys 0 and ln(3), and values 2 and 10. For q = 1, exponentiating the scores gives 1 and 3. The weights are ¼ and ¾, and the weighted sum is ¼ × 2 + ¾ × 10 = 8. For q = 0, each exponent is 1 and the answer is (2 + 10)/2 = 6. A real decoder includes the current position among those it can attend to.

This is how different queries read the [[causal-cache]]. It also explains how [[shared-heads]] can share stored values while producing different answers. [Transformer, §3.2](https://arxiv.org/abs/1706.03762)
