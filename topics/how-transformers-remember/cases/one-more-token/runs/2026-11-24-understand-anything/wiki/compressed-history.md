# Compress older positions

DeepSeek‑V4‑Flash combines a detailed recent window with compressed global history. Forming summaries reduces the inventory of stored entries. Sparse selection reduces how many entries a query reads.

Compressed Sparse Attention (CSA) produces finer summaries and selects relevant ones for each query. Heavily Compressed Attention (HCA) uses coarser summaries and can read all of them. A model can therefore store many entries yet consult only a few on a particular step. Grouping positions also changes the detail represented by each entry.

V4.1‑Flash adds sharing across layers through [[shared-layers]]. Check the distinction between stored and selected entries in [[transfer-check]]. [DeepSeek‑V4‑Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
