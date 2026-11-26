"""Capture the whole checkpoint on CPU and export identical inputs for five tools."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.profiler import ExecutionTraceObserver, ProfilerActivity, profile, record_function
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from workload import ASSETS, SPEC, AttentionCore


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(path)


def verify(checkpoint):
    for shard in SPEC["shards"]:
        path = checkpoint / shard["name"]
        if path.stat().st_size != shard["size"]:
            raise ValueError(f"Incomplete download: {path.name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != shard["sha256"]:
            raise ValueError(f"Incorrect checksum: {path.name}")
        print("Verified", path.name, flush=True)


def cache_rows(cache):
    rows = []
    for i, layer in enumerate(cache.layers):
        for key in ("keys", "values", "conv_states", "recurrent_states"):
            tensors = getattr(layer, key, {})
            if not isinstance(tensors, dict):
                tensors = {0: tensors}
            for slot, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    rows.append(dict(layer=i, name=key, slot=slot, shape=list(tensor.shape),
                                     dtype=str(tensor.dtype), bytes=tensor.numel()*tensor.element_size()))
    return rows


class Clock:
    """Inclusive PyTorch module wall and process CPU intervals, in microseconds."""
    def __init__(self, model):
        self.handles = []
        self.modules = {}
        self.records = []
        language = model.model.language_model
        self.add("embedding", language.embed_tokens)
        for i, layer in enumerate(language.layers):
            for name, module in layer.named_modules():
                if not name or "." not in name or isinstance(module, torch.nn.Linear):
                    self.add(f"block.{i}" + ("." + name if name else ""), module)
        self.add("final_norm", language.norm)
        self.add("lm_head", model.lm_head)
        self.active = False

    def add(self, name, module):
        self.modules[name] = module
        self.handles.extend([module.register_forward_pre_hook(self.start_node(name)),
                             module.register_forward_hook(self.end_node(name))])

    def start_node(self, name):
        def start(module, args):
            if self.active:
                self.starts[name] = (time.perf_counter_ns(), time.process_time_ns())
        return start

    def end_node(self, name):
        def end(module, args, output):
            if not self.active:
                return
            stop, cpu = time.perf_counter_ns(), time.process_time_ns()
            start, start_cpu = self.starts.pop(name)
            row = dict(name=name, start_us=round((start-self.origin)/1e3, 2),
                       wall_us=round((stop-start)/1e3, 2), cpu_us=round((cpu-start_cpu)/1e3, 2))
            if isinstance(module, torch.nn.Linear):
                x = args[0]
                row.update(weight_bytes=module.weight.numel()*module.weight.element_size(),
                           input_bytes=x.numel()*x.element_size(),
                           output_bytes=output.numel()*output.element_size(),
                           flops=2*(x.numel()//module.in_features)*module.weight.numel())
            self.records.append(row)
        return end

    def measure(self, function):
        self.records, self.starts = [], {}
        self.origin = time.perf_counter_ns()
        cpu = time.process_time_ns()
        self.active = True
        try:
            output = function()
        finally:
            wall, elapsed = time.perf_counter_ns()-self.origin, time.process_time_ns()-cpu
            self.active = False
        return output, dict(wall_us=round(wall/1e3,2), cpu_us=round(elapsed/1e3,2),
                            events=sorted(self.records, key=lambda r:r["start_us"]),
                            top_id=int(output.logits[0,-1].argmax()), cache=cache_rows(output.cache))


def export_fixture(captured):
    q, k, v, native = [tensor.float().contiguous() for tensor in captured]
    assert [list(t.shape) for t in (q,k,v)] == [SPEC[n+"_shape"] for n in ("q","k","v")]
    expected = AttentionCore()(q,k,v)
    tmp = ASSETS / "layer3-pending.npz"
    np.savez_compressed(tmp, **{n:t.numpy() for n,t in
                          zip(("q","k","v","expected","native"),(q,k,v,expected,native))})
    tmp.replace(ASSETS / "layer3-decode.npz")
    print("Real comparison fixture ready", flush=True)
    return dict(sha256=hashlib.sha256((ASSETS/"layer3-decode.npz").read_bytes()).hexdigest(),
                max_abs_fp32_vs_native=float((native-expected).abs().max()))


def native_trace(fn, label, local):
    prefix = "full" if label == "decode" else "prefill"
    et = local / (prefix+"-et.json")
    observer = ExecutionTraceObserver().register_callback(str(et))
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True,
                 execution_trace_observer=observer) as tracer:
        with record_function(f"Qwen3.8-27B/{label}"):
            out = fn()
    observer.unregister_callback()
    tracer.export_chrome_trace(str(local / (prefix+"-kineto.json")))
    print(label, "native traces ready", flush=True)
    return out


def run(checkpoint, local):
    torch.set_num_threads(SPEC["torch_threads_full"])
    torch.set_num_interop_threads(1)
    model = qwen.Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, attn_implementation="eager", dtype=torch.bfloat16).eval()
    prompt = torch.tensor([SPEC["prompt_ids"]])
    token = torch.tensor([[SPEC["decode_id"]]])
    def prefill():
        return model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    def decode(prefilled):
        return model(input_ids=token, cache=prefilled.cache, use_cache=True, logits_to_keep=1)
    captured = []
    mlp_io = {}
    def save_mlp(module, args, output):
        label = "decode" if args[0].shape[1] == 1 else "prefill"
        mlp_io[label+"_input"] = args[0].detach().float().contiguous().numpy()
        mlp_io[label+"_output"] = output.detach().float().contiguous().numpy()
    mlp_handle = model.model.language_model.layers[SPEC["layer_index"]].mlp.register_forward_hook(save_mlp)
    original = qwen.eager_attention_forward
    def capture(module, query, key, value, attention_mask, **kwargs):
        output = original(module, query, key, value, attention_mask, **kwargs)
        if getattr(module, "layer_idx", -1) == SPEC["layer_index"] and query.shape[-2] == 1:
            assert attention_mask is None or torch.all(attention_mask == 0)
            captured.extend(t.detach().clone() for t in (query, key, value, output[0].transpose(1,2)))
        return output
    with torch.inference_mode():
        qwen.eager_attention_forward = capture
        try:
            warm = prefill()
            warm_prefill_top = int(warm.logits[0,-1].argmax())
            warm_decode_top = int(decode(warm).logits[0,-1].argmax())
        finally:
            qwen.eager_attention_forward = original
            mlp_handle.remove()
        np.savez_compressed(ASSETS/"layer3-mlp.npz", **mlp_io)
        fixture = export_fixture(captured)
        del warm
        clock = Clock(model)
        prof_prefill, first = clock.measure(prefill)
        prof_decode, second = clock.measure(lambda:decode(prof_prefill))
        assert (first["top_id"],second["top_id"]) == (warm_prefill_top,warm_decode_top)
        assert sum(row["name"].count(".")==1 and row["name"].startswith("block.") for row in second["events"])==len(model.model.language_model.layers)
        for handle in clock.handles:
            handle.remove()
        del prof_prefill, prof_decode
        result = dict(model=SPEC["model"], revision=SPEC["revision"], torch=torch.__version__,
                      transformers=transformers.__version__, device="CPU", machine=platform.machine(),
                      threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
                      warmup_pairs=1, timed_pairs=1, fixture=fixture, prefill=first, decode=second)
        dump(ASSETS/"whole-pass.json", result)
        print("Full model timings ready", flush=True)
        # Label decoder blocks in the profiler's timeline. These markers run in a separate pass.
        labels = {}
        handles = []
        def enter(label):
            def pre(module, args):
                labels[label] = record_function(label)
                labels[label].__enter__()
            return pre
        def exit_(label):
            def post(module, args, output):
                labels.pop(label).__exit__(None,None,None)
            return post
        for i, block in enumerate(model.model.language_model.layers):
            name = f"Qwen/block.{i}"
            handles += [block.register_forward_pre_hook(enter(name)), block.register_forward_hook(exit_(name))]
        traced = native_trace(prefill, "prefill", local)
        predicted = native_trace(lambda:decode(traced),"decode",local)
        assert int(predicted.logits[0,-1].argmax()) == warm_decode_top
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--skip-sha", action="store_true", help="Use only after verifying all shard hashes.")
    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_sha:
        verify(args.checkpoint)
    run(args.checkpoint, args.trace_dir)
