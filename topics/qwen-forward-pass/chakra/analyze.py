"""Build the study from paired ET/Kineto traces with the installed Chakra reader and schema."""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from chakra.schema.protobuf import et_def_pb2 as pb
from chakra.src.host_trace.trace import HostTrace
from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage

HERE = Path(__file__).resolve().parent
BIN = Path(sys.executable).parent
LABELS = (
    ("Expand keys", "aten::repeat_interleave"),
    ("Expand values", "aten::repeat_interleave"),
    ("Transpose keys", "aten::transpose"),
    ("Score queries", "aten::matmul"),
    ("Scale scores", "aten::mul"),
    ("Normalize scores", "aten::softmax"),
    ("Mix values", "aten::matmul"),
)


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def content_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compress(source, dest):
    with source.open("rb") as src, dest.open("wb") as dst, gzip.GzipFile(fileobj=dst, mode="wb", mtime=0) as gz:
        shutil.copyfileobj(src, gz)


def read_et(path):
    meta, rows = pb.GlobalMetadata(), []
    with path.open("rb") as file:
        if not decodeMessage(file, meta):
            raise ValueError("Missing ET metadata")
        while True:
            node = pb.Node()
            if not decodeMessage(file, node):
                break
            rows.append(node)
    return meta, rows


def tensors(io):
    result = []
    for v, t, shape, stride in zip(io["values"], io["types"], io["shapes"], io["strides"]):
        if isinstance(t, str) and t.startswith("Tensor(") and isinstance(v, list) and len(v) == 6:
            tensor, storage, offset, size, width, device = v
            result.append(dict(id=tensor, storage=storage, offset=offset, elements=size, bytes=size*width,
                               dtype=t[7:-1], device=device, shape=shape, strides=stride))
    return result


def union_width(intervals):
    total, end = 0.0, float("-inf")
    for lo, hi in sorted(intervals):
        total += max(0, hi-max(lo, end))
        end = max(hi, end)
    return total


class Pair:
    """Match by RecordFunction id and exact name inside one paired capture."""
    def __init__(self, host, kineto):
        self.trace = HostTrace.from_file(str(host))
        events = read_json(kineto)["traceEvents"]
        by_rf = defaultdict(list)
        for event in events:
            if event.get("ph") == "X" and event.get("cat") in ("cpu_op", "user_annotation"):
                by_rf[event.get("args", {}).get("Record function id")].append(event)
        self.events = {}
        for node in self.trace.nodes.values():
            matches = [e for e in by_rf.get(node.rf_id, []) if e["name"] == node.name]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous Kineto match at ET node {node.id}")
            if matches:
                self.events[node.id] = matches[0]
        self.paths = {"host_sha256": content_hash(host), "kineto_sha256": content_hash(kineto)}

    def children(self, node):
        return [child for child in self.trace.reachable_from(node) if child.id != node.id]

    def exclusive(self, node):
        event = self.events[node.id]
        lo, hi = event["ts"], event["ts"] + event["dur"]
        children = [self.events[c.id] for c in node.children if c.id in self.events]
        covered = [(max(lo, c["ts"]), min(hi, c["ts"]+c["dur"])) for c in children if c["tid"] == event["tid"]]
        return max(0.0, event["dur"]-union_width(covered))

    def counts(self):
        tids = Counter(e["tid"] for e in self.events.values())
        return dict(host_nodes=len(self.trace.nodes), matched=len(self.events), cpu_tracks=len(tids),
                    thread_events=dict(tids), schema=self.trace.schema.text, **self.paths)


def export_projection(pair, root, rows, out):
    """Write the region and its direct operations; tensor edges join exact IDs between siblings."""
    meta = pb.GlobalMetadata(version=pair.trace.schema.text)
    meta.attr.add(name="projection", string_val="Replay region + seven top-level CPU operations. ctrl_deps = scope parent; data_deps = same tensor ID between siblings. Times relative to region start; inclusive, rounded to us.")
    with out.open("wb") as file:
        encodeMessage(file, meta)
        region = pb.Node(id=root.id, name=root.name, type=pb.METADATA_NODE, duration_micros=round(pair.events[root.id]["dur"]))
        encodeMessage(file, region)
        for r in rows:
            node = pair.trace.nodes[r["id"]]
            p = pb.Node(id=node.id, name=node.name, type=pb.COMP_NODE, ctrl_deps=[root.id],
                        data_deps=r["deps"], start_time_micros=round(r["start"]), duration_micros=round(r["dur"]))
            for side in ("inputs", "outputs"):
                io, dest = getattr(node, side), getattr(p, side)
                for key in ("values", "shapes", "types"):
                    setattr(dest, key, json.dumps(io[key]))
            p.attr.add(name="rf_id", int64_val=node.rf_id)
            p.attr.add(name="kineto_tid", int64_val=pair.events[node.id]["tid"])
            encodeMessage(file, p)
    _, decoded = read_et(out)
    assert len(decoded) == 1 + len(rows)
    for encoded, row in zip(decoded[1:], rows, strict=True):
        assert encoded.id == row["id"] and list(encoded.ctrl_deps) == [root.id] and list(encoded.data_deps) == row["deps"]
        assert encoded.start_time_micros == round(row["start"])
    subprocess.run([BIN / "chakra_jsonizer", "--input_filename", out, "--output_filename", out.with_suffix(".stream")], check=True)
    # The official jsonizer emits concatenated JSON objects. Parse each before publishing one JSON array.
    stream = out.with_suffix(".stream")
    data, decoder, messages = stream.read_text().lstrip(), json.JSONDecoder(), []
    while data:
        value, pos = decoder.raw_decode(data)
        messages.append(value)
        data = data[pos:].lstrip()
    assert len(messages) == len(decoded) + 1
    write_json(out.with_suffix(".json"), messages)
    stream.unlink()


def native_conversion(raw, out):
    linked = raw / "native-linked.json"
    native = out / "native-replay.et"
    cmd = [BIN / "chakra_trace_link", "--rank", "0", "--chakra-host-trace", (raw/"host.json").resolve(),
           "--chakra-device-trace", (raw/"kineto.json").resolve(), "--output-file", linked.resolve()]
    result = subprocess.run(cmd, cwd=raw, capture_output=True, text=True)
    (raw/"link.log").write_text(result.stdout + result.stderr)
    # A host-only conversion is directly supported by the official converter.
    source = linked if result.returncode == 0 else raw / "host.json"
    subprocess.run([BIN/"chakra_converter", "--log-filename", raw/"converter.log", "PyTorch",
                    "--input", source, "--output", native], check=True)
    _, rows = read_et(native)
    return dict(link_exit=result.returncode, source=source.name, nodes=len(rows),
                order_edges=sum(len(n.data_deps) for n in rows), timed_nodes=sum(n.duration_micros > 0 for n in rows),
                link_tail=[s for s in (result.stdout+result.stderr).splitlines() if s][-1][:300] if result.returncode else None)


def summarize_replay(raw, out):
    host, kineto = raw/"host.json", raw/"kineto.json"
    pair = Pair(host, kineto)
    scope = next(n for n in pair.trace.nodes.values() if n.name == "layer3.decode.attention_core")
    zero = pair.events[scope.id]["ts"]
    ops = sorted(scope.children, key=lambda n: pair.events[n.id]["ts"])
    writes, rows = {}, []
    for (label, opname), node in zip(LABELS, ops, strict=True):
        assert node.name == opname
        event = pair.events[node.id]
        inputs, outputs = tensors(node.inputs), tensors(node.outputs)
        deps = sorted({writes[t["id"]] for t in inputs if t["id"] in writes})
        descendants = pair.children(node)
        nested = [dict(id=child.id, parent=child.parent, name=child.name,
                       start=round(pair.events[child.id]["ts"]-zero,3), dur=round(pair.events[child.id]["dur"],3))
                  for child in descendants if child.id in pair.events]
        rows.append(dict(label=label, name=node.name, id=node.id, rf=node.rf_id, parent=node.parent, deps=deps,
                         inputs=inputs, outputs=outputs, nested=nested, tid=event["tid"],
                         start=round(event["ts"]-zero,3), dur=round(event["dur"],3),
                         self_us=round(pair.exclusive(node),3)))
        for t in outputs:
            writes[t["id"]] = node.id
    export_projection(pair, scope, rows, out/"attention.et")
    compress(host, out/"replay-host.json.gz")
    compress(kineto, out/"replay-kineto.json.gz")
    return dict(**pair.counts(), root=scope.id, duration=pair.events[scope.id]["dur"], rows=rows,
                native=native_conversion(raw, out), check=read_json(raw/"replay-check.json"))


def summarize_phase(path, prefix, label):
    pair = Pair(path/(prefix+"-et.json"), path/(prefix+"-kineto.json"))
    region = next(n for n in pair.trace.nodes.values() if n.name == "Qwen3.8-27B/"+label)
    root_event = pair.events[region.id]
    blocks = []
    for node in pair.trace.nodes.values():
        if not (match := re.fullmatch(r"Qwen/block\.(\d+)", node.name)):
            continue
        children = pair.children(node)
        event = pair.events[node.id]
        totals, counts, examples, tensors_by_op = Counter(), Counter(), {}, {}
        for child in children:
            if child.id not in pair.events:
                continue
            name = child.name
            counts[name] += 1
            totals[name] += pair.events[child.id]["dur"]
            if name not in examples:
                examples[name] = child.id
                tensors_by_op[name] = [t["shape"] for t in tensors(child.inputs)]
        hot = [dict(name=name, count=counts[name], inclusive_us=round(total,2), sample=examples[name],
                    shapes=tensors_by_op[name]) for name, total in totals.most_common(8)]
        blocks.append(dict(index=int(match[1]), id=node.id, rf=node.rf_id, parent=node.parent,
                           start=round(event["ts"]-root_event["ts"],2), dur=round(event["dur"],2),
                           ops=len(children), matched=sum(counts.values()), hot=hot))
    blocks.sort(key=lambda b:b["index"])
    selection_path = path/(prefix+"-et-selection.json")
    selection = read_json(selection_path) if selection_path.exists() else {}
    return dict(**pair.counts(), source_host_nodes=selection.get("et_nodes", len(pair.trace.nodes)),
                duration=root_event["dur"], region=region.id, blocks=blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", required=True, type=Path)
    ap.add_argument("--full-traces", required=True, type=Path)
    ap.add_argument("--whole-pass", required=True, type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    replay = summarize_replay(args.replay.resolve(), args.output)
    observed = read_json(args.whole_pass)
    full = {}
    for label, prefix in (("prefill", "prefill"), ("decode", "full")):
        phase = summarize_phase(args.full_traces, prefix, label)
        wall, cpu = [observed[label][k] for k in ("wall_us", "cpu_us")]
        phase.update(wall_us=wall, cpu_us=cpu, threads=observed["threads"])
        full[label] = phase
    result = dict(replay=replay, full=full)
    (args.output/"study-data.js").write_text("window.chakraStudy = " + json.dumps(result, separators=(",", ":"), allow_nan=False) + ";\n")
    print("Replay", {k:v for k,v in replay.items() if k not in ("rows","paths")})
    for phase, s in full.items():
        print(phase, s["host_nodes"], "host nodes,", s["matched"], "matched,", len(s["blocks"]), "blocks")


if __name__ == "__main__":
    main()
