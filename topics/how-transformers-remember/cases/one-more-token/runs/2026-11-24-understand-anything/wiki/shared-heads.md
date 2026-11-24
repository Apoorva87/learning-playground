# Share keys and values across heads

Keep 32 query heads. With 8 KV heads, each group of four queries reads one KV bank. With one KV head, every query reads the same bank. Each query still computes its own attention weights.

In the [[dense-budget]] example, MHA (32 KV heads), GQA (8), and MQA (1) use 2 GiB, 512 MiB, and 64 MiB respectively at 4,096 tokens. At 8,192 tokens they use 4 GiB, 1 GiB, and 128 MiB. These are hypothetical architectures with equal layer count, head size, and precision. An existing MHA model would need training or adaptation to share heads this way.

Historically, MQA appeared in 2019 and GQA in 2023. A different way to share information across heads is [[latent-attention]]. [MQA paper](https://arxiv.org/abs/1911.02150) · [GQA paper](https://arxiv.org/abs/2305.13245)
