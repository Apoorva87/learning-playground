# Predict what changes

Start with 4,096 tokens and 8 KV heads in the toy model. Double the sequence length and halve the number of KV heads. Predict the memory cost before opening the linked explanation.

Check: in [[dense-budget]], the factors of 2 and ½ cancel. The result stays at 512 MiB. All 32 query heads still have a KV group to read. Now compare two designs that each read 512 global entries per query. If one stores a same-sized entry per token and the other stores one per four tokens, which cache grows faster?

Check: the first stores about four times as many entries over a long prefix. A read limit is insufficient to determine storage. Revisit [[compressed-history]] for the mechanism, then try explaining [[global-budget]] without looking up its two factors.
