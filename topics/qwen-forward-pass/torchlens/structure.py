"""Trace both text passes with no checkpoint tensors and save their geometry."""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torchlens as tl
import transformers
from torchlens.options import CaptureOptions
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
from workload import SPEC

CONFIG = json.loads((HERE.parent / "assets/config.json").read_text())


class TwoPassText(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm = Qwen3_5ForConditionalGeneration(config)

    def forward(self, prompt, decode):
        first = self.lm(input_ids=prompt, use_cache=True, logits_to_keep=1)
        second = self.lm(input_ids=decode, use_cache=True, cache=first.cache, logits_to_keep=1)
        return first.logits, second.logits


def write_outputs(trace):
    operations = list(trace)
    records = []
    phase_counts = Counter()
    rows = defaultdict(list)
    for op in operations:
        path = list(op.modules)
        phase = {"lm:1": "prefill", "lm:2": "decode"}.get(next(iter(path), ""), "boundary")
        match = re.search(r"\blayers\.(\d+):", " ".join(path))
        block = int(match.group(1)) if match else None
        shape = list(op.shape)
        records.append([op.label, list(op.parents), op.type, op.module, shape, str(op.dtype), phase, block])
        phase_counts[phase] += 1
        if block is not None:
            rows[(phase, block)].append(op)

    # Preserve all raw parent labels. TorchLens omits ':1' on single-use labels.
    graph = {
        "model": SPEC["model"], "revision": SPEC["revision"], "mode": "structure_only",
        "columns": ["label", "parent_labels", "type", "module", "hypothesis_shape", "dtype", "phase", "block"],
        "rows": records,
    }
    packed = json.dumps(graph, separators=(",", ":"), ensure_ascii=True).encode()
    (HERE / "text-topology.json.gz").write_bytes(gzip.compress(packed, mtime=0))

    def block_row(ops, i):
        counts = Counter(op.type for op in ops)
        last = ops[-1]
        return {"block": i, "type": CONFIG["text_config"]["layer_types"][i], "records": len(ops),
                "frequent": counts.most_common(5), "output": list(last.shape), "last": last.label}

    summary = {
        "model": SPEC["model"], "revision": SPEC["revision"],
        "tool": tl.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
        "records": len(operations), "ops": len(trace.ops), "edges": trace.num_edges,
        "parameters": trace.num_params, "counts": phase_counts,
        "blocks": {phase: [block_row(rows[(phase, i)], i) for i in range(64)]
                   for phase in ("prefill", "decode")},
        "output_shapes": [list(trace[label].shape) for label in trace.output_layers],
    }
    (HERE / "structure.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(1)
    config = Qwen3_5Config(**CONFIG)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = TwoPassText(config).eval()
        inputs = (torch.tensor([SPEC["prompt_ids"]]), torch.tensor([[SPEC["decode_id"]]]))
    assert all(p.device.type == "meta" for p in model.parameters())
    with torch.no_grad():
        trace = tl.trace(model, inputs, save="none",
                         capture=CaptureOptions(structure_only=True, save_source_context=False))
    summary = write_outputs(trace)
    assert all(b["records"] > 0 for blocks in summary["blocks"].values() for b in blocks)
    assert summary["output_shapes"] == [[1, 1, config.text_config.vocab_size]] * 2
    print(summary["records"], "records,", summary["ops"], "tensor operations,", summary["edges"], "edges")


if __name__ == "__main__":
    main()
