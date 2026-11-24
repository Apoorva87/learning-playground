# What can the next token reuse?

A causal decoder saves a key and a value for each past position at each attention layer. Appending a token cannot change what an earlier position could see, so its old projections remain valid.

Suppose the prompt says “The spare key is in the blue drawer. Open the …”. Prefill processes the prompt and stores its projections. During decode the new position computes its own query, key, and value. It appends its key and value and uses its query to score the eligible keys. The key determines how relevant a position is to this query. The value supplies what it contributes. Each new query needs fresh scores.

Follow [[attention-row]] to see the same cache return two different answers, then [[dense-budget]] to count what gets stored. [Transformer attention](https://arxiv.org/abs/1706.03762)
