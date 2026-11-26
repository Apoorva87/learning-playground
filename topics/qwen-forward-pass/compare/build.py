"""Refresh the inline chart data using only the saved study outputs and Python 3."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent


def read(path):
    return json.loads((TOPIC / path).read_text())


def js_data(path, prefix):
    return json.loads((TOPIC / path).read_text().split(prefix, 1)[1].strip().removesuffix(";"))


def fixture_vector(archive, name):
    with archive.open(name + ".npy") as source:
        assert source.read(8) == b"\x93NUMPY\x01\x00"
        header_size, = struct.unpack("<H", source.read(2))
        header = ast.literal_eval(source.read(header_size).decode("ascii").strip())
        assert header["descr"] == "<f4" and not header["fortran_order"]
        return struct.unpack(f"<{math.prod(header['shape'])}f", source.read())


def model_data():
    whole = read("assets/whole-pass.json")
    config = read("assets/config.json")["text_config"]
    record = {}
    for phase in ("prefill", "decode"):
        captured = whole[phase]
        kv_bytes = sum(c["bytes"] for c in captured["cache"] if c["name"] in ("keys", "values"))
        total_bytes = sum(c["bytes"] for c in captured["cache"])
        record[phase] = dict(wall=captured["wall_us"], cpu=captured["cpu_us"],
                             kv=kv_bytes, state=total_bytes - kv_bytes)
    record["types"] = config["layer_types"]
    return record


def torchlens_data():
    trace = read("torchlens/trace.json")
    nodes = [{k: row[k] for k in ("id", "parents", "macs", "tensor_bytes", "read_bytes", "write_bytes")}
             for row in trace["nodes"]]
    fixture = TOPIC / "assets/layer3-decode.npz"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == trace["fixture_sha256"]
    with ZipFile(fixture) as archive:
        values = fixture_vector(archive, "v")
        expected = fixture_vector(archive, "expected")
    rows = trace["probabilities"]
    assert len(rows) == 24 and all(len(row) == 17 for row in rows)
    outputs = []
    for head, weights in enumerate(rows):
        feature = sum(w * values[((head // 6) * 17 + t) * 256] for t, w in enumerate(weights))
        assert math.isclose(feature, expected[head * 256], rel_tol=1e-4, abs_tol=1e-5)
        outputs.append(expected[head * 256])
    return dict(nodes=nodes, weights=rows, ids=trace["token_ids"], outputs=outputs,
                verified=trace["verified_elements"], aten=trace["operator_attributions"])


def profiler_data():
    trace = js_data("pytorch-profiler/replay-data.js", "window.profileData.replay=")
    outer = [r for r in trace["rows"] if r[3] == 1]
    names = ["Expand K", "Expand V", "Transpose", "Score", "Scale", "Softmax", "Mix"]
    assert [r[0].split("::")[1] for r in outer] == [
        "repeat_interleave", "repeat_interleave", "transpose", "matmul", "mul", "softmax", "matmul"]
    rows = []
    for r, name in zip(outer, names):
        positive = sum(b for time, b in trace["allocations"] if r[1] <= time < r[1] + r[2] and b > 0)
        rows.append(dict(label=name, start=r[1], duration=r[2], self=r[4], allocated=positive))
    first = outer[0]
    children = [r for r in trace["rows"] if r[3] == 2 and first[1] <= r[1] < first[1] + first[2]]
    return dict(rows=rows, total=trace["trace_us"], count=trace["count"],
                children=[dict(label=r[0].split("::")[1], start=r[1], duration=r[2]) for r in children])


def dace_data():
    data = read("dace/results.json")
    stages = []
    for s in data["stages"]:
        stages.append(dict(name=s["name"], iterations=s["iterations"], time=s["trace_us"],
                           reads=sum(x["bytes"] for x in s["reads"]),
                           writes=sum(x["bytes"] for x in s["writes"]),
                           input=[x["array"] for x in s["reads"]],
                           output=[x["array"] for x in s["writes"]]))
    return dict(stages=stages, unique=data["unique_bytes"], elapsed=data["wall_us"])


def chakra_data():
    trace = js_data("chakra/study-data.js", "window.chakraStudy = ")["replay"]
    rows = []
    for r in trace["rows"]:
        rows.append(dict(label=r["label"], id=r["id"], deps=r["deps"], start=r["start"],
                         duration=r["dur"], input=[t["storage"] for t in r["inputs"]],
                         output=r["outputs"][0]["storage"], size=r["outputs"][0]["bytes"]))
    return dict(rows=rows, total=trace["duration"], tracks=trace["cpu_tracks"])


def build():
    data = dict(model=model_data(), lens=torchlens_data(), profiler=profiler_data(),
                dace=dace_data(), chakra=chakra_data())
    data["compiler"] = compiler_data()
    page = HERE / "index.html"
    start = '<script type="application/json" id="study-data">'
    before, content = page.read_text().split(start, 1)
    _, after = content.split("</script>", 1)
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    page.write_text(before + start + encoded + "</script>" + after)


def compiler_data():
    base = "torchinductor/evidence/attention/"
    run = read(base + "run.json")
    code = (TOPIC / base / "output_code.py").read_text()
    tree = ast.parse(code)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call")
    buffers = []
    kernels = []
    for node in method.body:
        call = getattr(node, "value", None)
        if not isinstance(call, ast.Call):
            continue
        name = ast.unparse(call.func)
        if name == "empty_strided_cpu":
            shape = ast.literal_eval(call.args[0])
            assert ast.unparse(call.args[2]) == "torch.float32"
            buffers.append({"name": ast.unparse(node.targets[0]), "bytes": math.prod(shape) * 4})
        elif name.startswith(("cpp_fused_", "torch.ops.aten.", "extern_kernels.")):
            kernels.append(dict(name=name, args=[ast.unparse(a) for a in call.args]))
    assert len(buffers) == 2 and len(kernels) == 2
    assert kernels[0]["name"].startswith("cpp_fused_") and "flash_attention" in kernels[1]["name"]
    assert [b["name"] for b in buffers] == kernels[0]["args"][2:] == kernels[1]["args"][1:]
    return dict(kernels=kernels, buffers=buffers, inputs=run["inputs"], output=run["output"],
                times=run["timing"]["wall_us_per_call"], batches=run["timing"]["batches"],
                calls=run["timing"]["calls_per_batch"], median=run["timing"]["median_wall_us"],
                error=run["compiled_vs_reference"]["max_abs"])


if __name__ == "__main__":
    build()
