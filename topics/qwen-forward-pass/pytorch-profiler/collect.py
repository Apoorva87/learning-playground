"""Reduce PyTorch 2.11 Kineto captures using the emitted, start-ordered host spans.

Memory usage is proportional to the retained rows plus call-stack depth. Each
header is parsed once. The record-ID count exported by select_traces.py checks
that the version-specific reader accounted for all CPU operations/annotations.
"""
import argparse
from collections import defaultdict
import gzip
import json
import math
import mmap
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RAW_X = re.compile(
    rb'"ph": "X", "cat": "(cpu_op|user_annotation)", "name": "([^"\n]+)", '
    rb'"pid": (\d+), "tid": (\d+),\s*"ts": ([\d.]+), "dur": ([\d.]+)'
)


def save_trace(path, trace):
    with gzip.GzipFile(str(path), "wb", mtime=0) as output:
        output.write(json.dumps(trace, separators=(",", ":")).encode())


def ns(raw):
    """Kineto uses microseconds with up to three decimal places."""
    whole, _, fraction = raw.partition(b".")
    return int(whole) * 1000 + int(fraction.ljust(3, b"0"))


def tensors(record):
    result = []
    for value, shape, dtype in zip(record["values"], record["shapes"], record["types"]):
        if dtype.startswith("Tensor("):
            # ET stores tensor ID, storage ID, offset, numel, bytes per element, device.
            result.append([shape, dtype[7:-1].replace("c10::", ""), math.prod(shape) * value[4]])
    return result


def native_args(args):
    sizes = {"c10::BFloat16": 2, "float": 4, "long long": 8, "bool": 1,
             "double": 8, "int": 4}
    return [[shape, dtype.replace("c10::", ""), math.prod(shape) * sizes[dtype]]
            for dtype, shape in zip(args.get("Input type", []), args.get("Input Dims", []))
            if dtype in sizes]


def read_phase(source, phase):
    compact = source / "compact"
    tag = "prefill" if phase == "prefill" else "full"
    kin = tag + "-kineto.json"
    selected = json.loads((compact / kin).read_text())["traceEvents"]
    selection = json.loads((compact / (tag + "-et-selection.json")).read_text())
    nodes = json.loads((compact / (tag + "-et.json")).read_text())["nodes"]
    by_rf = {attr["value"]: node for node in nodes for attr in node["attrs"] if attr["name"] == "rf_id"}
    retained = {(int(round(e["ts"] * 1000)), e["name"]): e for e in selected}
    # Integer ns for exact alignment even at large epoch offsets. A float may
    # round one ns either way when reading an absolute JSON timestamp.
    stacks = defaultdict(list)
    threads = defaultdict(int)
    totals = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    measured = []
    origins = {}
    serial = 0
    origin = None
    conv_index = None
    cond = source / kin

    def finish(frame):
        name, start, end, direct, row, in_conv = frame
        duration = end - start
        own = duration - direct
        assert own >= 0, (name, own)
        stats = totals[name]
        stats[0] += 1
        stats[1] += duration
        stats[2] += own
        if in_conv:
            stats[3] += 1
            stats[4] += duration
            stats[5] += own
        if row is not None:
            row[4] = round(own / 1000, 3)

    parsed_names = {}
    with cond.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for match in RAW_X.finditer(data):
            serial += 1
            key = match[2]
            name = parsed_names.get(key)
            if name is None:
                name = parsed_names[key] = key.decode()
            thread = match[4]
            threads[thread] += 1
            start, duration = ns(match[5]), ns(match[6])
            end = start + duration
            if origin is None:
                origin = start
            stack = stacks[thread]
            while stack and start >= stack[-1][2]:
                finish(stack.pop())
            if stack:
                assert start >= stack[-1][1] and end <= stack[-1][2], (name, stack[-1][:3], start, end)
                stack[-1][3] += duration
            in_conv = name == "aten::conv1d" or bool(stack and stack[-1][-1])
            if name == "aten::conv1d":
                conv_index = 0
            if in_conv:
                conv_index += 1
            native = retained.get((start, name))
            if native is None and (start % 2):
                native = retained.get((start - 1, name)) or retained.get((start + 1, name))
            if native is None and in_conv and conv_index <= 21:
                b = data.rfind(b'\n  {\n', max(0, match.start() - 18), match.start())
                e = data.find(b'\n  }', match.end())
                native = json.loads(data[b:e + 4])
                selected.append(native)
            row = None
            if native:
                rf = native["args"]["Record function id"]
                node = by_rf.get(rf)
                if node is not None:
                    ins, outs = tensors(node["inputs"]), tensors(node["outputs"])
                else:
                    ins, outs = native_args(native["args"]), None
                row = [name, round((start - origin) / 1000, 3), round(duration / 1000, 3),
                       len(stack), 0, ins, outs, rf]
                measured.append(row)
                origins[rf] = row
            stack.append([name, start, end, 0, row, in_conv])
    for stack in stacks.values():
        while stack:
            finish(stack.pop())
    assert serial == selection["kineto_events_with_record_id"]
    assert len(selected) == len(origins) == len(measured)
    assert len(threads) == 1
    phase_span = next(row for row in measured if row[0].startswith("Qwen3.8-27B/"))
    assert abs(phase_span[2] * 1000 - sum(values[2] for values in totals.values())) < 2
    whole = json.loads((HERE.parent / "assets/whole-pass.json").read_text())[phase]
    metrics = {k: whole[k] for k in ("wall_us", "cpu_us")}
    metrics["blocks"] = {e["name"]: [e["wall_us"], e["cpu_us"]] for e in whole["events"]
                         if re.fullmatch(r"block\.\d+", e["name"])}
    def us_stats(values):
        return [round(v / 1000, 3) if i % 3 else v for i, v in enumerate(values)]
    save_trace(HERE / (phase + "-operators.json.gz"),
               {"traceEvents": sorted(selected, key=lambda e: e["ts"]), "displayTimeUnit": "ms"})
    return dict(rows=measured, aggregate={k: us_stats(v) for k, v in totals.items()},
                metrics=metrics, host_tid=int(next(iter(threads))), count=serial,
                trace_us=phase_span[2], backend_groups=10240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("phase", choices=["prefill", "decode"])
    args = parser.parse_args()
    phase_data = read_phase(args.source, args.phase)
    dest = HERE / (args.phase + "-data.js")
    dest.write_text("window.profileData=window.profileData||{};window.profileData." + args.phase + "="
                    + json.dumps(phase_data, separators=(",", ":")) + ";\n")
    print(args.phase, phase_data["count"], len(phase_data["rows"]), flush=True)
