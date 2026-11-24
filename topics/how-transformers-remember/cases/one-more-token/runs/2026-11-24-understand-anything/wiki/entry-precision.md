# Price one compressed entry

A V4.1‑Flash global entry has a 288-byte main vector and a 68-byte index key. Count the scale bytes as well as the packed four-bit values.

For the 512-channel main vector, two four-bit values occupy one byte and each group of 16 channels shares one scale byte. The total is 512/2 + 512/16 = 288 bytes. For the 128-channel index key, groups of 32 share a scale byte, giving 128/2 + 128/32 = 68 bytes. The combined price is 356 bytes. The local sliding-window cache uses FP8.

Multiply this price by the global writer rate from [[shared-layers]] to get the [[global-budget]]. [DeepSeek‑V4.1‑Flash precision](https://arxiv.org/html/2609.19969v1) · [Reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
