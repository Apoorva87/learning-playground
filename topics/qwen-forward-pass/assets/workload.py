"""Shared inputs and the FP32 CPU replay for the five Qwen experiments."""
import json
from pathlib import Path

import numpy as np
import torch

ASSETS = Path(__file__).resolve().parent
SPEC = json.loads((ASSETS / "workload.json").read_text())


class AttentionCore(torch.nn.Module):
    """Same six query heads per KV head as layer 3 of the pinned checkpoint."""

    def forward(self, q, k, v):
        keys = k.repeat_interleave(6, dim=1)
        values = v.repeat_interleave(6, dim=1)
        scores = torch.matmul(q, keys.transpose(-2, -1)) * SPEC["scale"]
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, values)


def load_replay():
    with np.load(ASSETS / "layer3-decode.npz", allow_pickle=False) as fixture:
        return tuple(torch.from_numpy(fixture[name].copy()) for name in ("q", "k", "v", "expected"))
