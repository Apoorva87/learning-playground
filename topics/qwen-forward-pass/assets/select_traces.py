"""Produce bounded JSON views of CPU native traces without loading millions of nested events.

The eager CPU depthwise convolution dispatches one convolution per channel in the
observed backend. Keep major calculation families plus every enclosing host
scope. Original event and record-function IDs still join the two files.
"""
import argparse
from array import array
from collections import Counter
import json
import mmap
from pathlib import Path
import re

ET = re.compile(rb'"id": (\d+), "name": "([^"\n]+)", "ctrl_deps": (\d+)')
RF = re.compile(rb'"Record function id":\s*(\d+)')
MAJOR = {b"aten::embedding", b"aten::linear", b"aten::mm", b"aten::bmm", b"aten::matmul",
         b"aten::conv1d", b"aten::convolution", b"aten::softmax", b"aten::_softmax",
         b"aten::sigmoid", b"aten::silu", b"aten::linalg_vector_norm", b"aten::layer_norm"}


def select(et_path, kineto_path, destination):
    parent = array("I", [0])*65536
    wanted = set()
    with et_path.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        total = 0
        for m in ET.finditer(data):
            i, name, up = int(m[1]), m[2], int(m[3])
            if i >= len(parent):
                parent.extend(array("I",[0])*max(len(parent),i+1-len(parent)))
            parent[i] = up
            if name in MAJOR or name.startswith((b"Qwen",b"[pytorch|")):
                wanted.add(i)
            total += 1
        for i in tuple(wanted):
            up = parent[i]
            while up and up not in wanted:
                wanted.add(up)
                up = parent[up]
        selected, rf = [], set()
        for m in ET.finditer(data):
            if int(m[1]) not in wanted:
                continue
            start = data.rfind(b"{", max(0,m.start()-35),m.start())
            stop = data.find(b'\n      }',m.end())
            if start == -1 or stop == -1:
                raise ValueError("Unexpected PyTorch ExecutionTrace JSON format")
            node = json.loads(data[start:stop+8])
            selected.append(node)
            rf.update(int(a["value"]) for a in node["attrs"] if a["name"]=="rf_id" and a["value"])
        a = data.find(b'"nodes":')
        header=json.loads(data[:a]+b'"nodes":[],'+data[data.rfind(b'"finish_ts"'):])
        assert len(selected)==len(wanted)
        header["nodes"]=selected
    destination.mkdir(exist_ok=True,parents=True)
    (destination/et_path.name).write_text(json.dumps(header,separators=(",",":")))
    native, all_rf = [], 0
    with kineto_path.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        for m in RF.finditer(data):
            all_rf += 1
            if int(m[1]) not in rf:
                continue
            start=data.rfind(b'\n  {\n',0,m.start())
            stop=data.find(b'\n  }',m.end())
            if start==-1 or stop==-1:
                raise ValueError("Unexpected Kineto event format")
            native.append(json.loads(data[start:stop+4]))
    (destination/kineto_path.name).write_text(json.dumps({"traceEvents":native,"displayTimeUnit":"ms"},separators=(",",":")))
    info=dict(et_nodes=total, retained_et_nodes=len(selected), kineto_events_with_record_id=all_rf,
              retained_kineto_events=len(native), major=sorted(s.decode() for s in MAJOR),
              names=Counter(e["name"] for e in native))
    (destination/(et_path.stem+"-selection.json")).write_text(json.dumps(info,indent=2)+"\n")
    print(et_path.name,total,len(selected),all_rf,len(native),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("phase",choices=["full","prefill"])
    args=parser.parse_args()
    select(args.directory/(args.phase+"-et.json"),args.directory/(args.phase+"-kineto.json"),args.directory/"compact")
