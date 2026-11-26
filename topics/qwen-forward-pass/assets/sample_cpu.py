"""Sample aggregate CPU occupancy through a separate, warmed run of the pinned checkpoint."""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from workload import ASSETS, SPEC


def measure(model, inputs, cache=None):
    samples = []
    blocks = []
    ready, stop = threading.Event(), threading.Event()

    def sample():
        while True:
            samples.append((time.perf_counter_ns(), time.process_time_ns(), time.thread_time_ns()))
            ready.set()
            if stop.is_set():
                return
            stop.wait(0.05)

    def boundary(index):
        def enter(module, args):
            blocks.append([index, time.perf_counter_ns()])
        return enter

    hooks = [layer.register_forward_pre_hook(boundary(i))
             for i, layer in enumerate(model.model.language_model.layers)]
    thread = threading.Thread(target=sample)
    thread.start()
    ready.wait()
    try:
        output = model(input_ids=inputs, cache=cache, use_cache=True, logits_to_keep=1)
    finally:
        stop.set()
        thread.join()
        for hook in hooks:
            hook.remove()
    start = samples[0][0]
    windows = []
    for first, last in zip(samples, samples[1:]):
        cpu_ns = (last[1]-first[1]) - (last[2]-first[2])
        windows.append([round((last[0]-start)/1000), round(cpu_ns/1000)])
    assert len(blocks) == 64 and all(end > begin for (_,begin),(_,end) in zip(blocks, blocks[1:]))
    return output, dict(windows=windows, blocks=[[i, round((t-start)/1000)] for i,t in blocks],
                        top_id=int(output.logits[0,-1].argmax()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(SPEC["torch_threads_full"])
    torch.set_num_interop_threads(1)
    model = qwen.Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, attn_implementation="eager", dtype=torch.bfloat16).eval()
    prompt, token = torch.tensor([SPEC["prompt_ids"]]), torch.tensor([[SPEC["decode_id"]]])
    with torch.inference_mode():
        warm = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        model(input_ids=token, cache=warm.cache, use_cache=True, logits_to_keep=1)
        del warm
        print("Warmup complete", flush=True)
        first, prefill = measure(model, prompt)
        _, decode = measure(model, token, first.cache)
    expected = json.loads((ASSETS/"whole-pass.json").read_text())
    result = {"revision":SPEC["revision"], "logical_cpus":os.cpu_count(),
              "target_interval_ms":50, "prefill":prefill, "decode":decode}
    for phase in ("prefill","decode"):
        assert result[phase]["top_id"] == expected[phase]["top_id"]
        print(phase, len(result[phase]["windows"]), "windows", flush=True)
    (ASSETS/"cpu-windows.json").write_text(json.dumps(result,separators=(",",":"))+"\n")
