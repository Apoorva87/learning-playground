"""Turn the saved TorchLens op topology into a costed, layered dependency graph.

Inputs (already in the repo, nothing is re-run):
  ../torchlens/text-topology.json.gz   17,469 op records, parents, shapes, dtypes
  ../assets/config.json                model dimensions (sanity checks only)
Output: data.js  (window.QWEN_GRAPH)

Cost model = SHAPES ONLY. FLOPs and bytes are logical counts for an unfused eager
execution: every real op reads all its input tensors and writes its output once.
Metadata-only ops (views, slices, no-op casts) cost nothing and are drawn as edges.
"""
import gzip, json, math, re, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent / "torchlens" / "text-topology.json.gz"

DT = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.float64": 8,
      "torch.int64": 8, "torch.int32": 4, "torch.bool": 1, "torch.uint8": 1}
VIEW = {"getitem", "reshape", "view", "transpose", "unsqueeze", "squeeze", "expand",
        "chunk", "permute", "splitwithsizes", "narrow", "flatten", "unflatten", "t", "detach"}
CAST = {"to", "float", "typeas", "bfloat16", "half"}
TRANS = {"exp", "rsqrt", "silu", "sigmoid", "softplus", "cos", "sin", "tanh", "log", "sqrt", "pow"}
REDUCE = {"mean", "sum", "amax", "max", "cumsum", "norm"}
COPYLIKE = {"cat", "pad", "copy", "contiguous", "zeros", "zeroslike", "ones", "arange",
            "repeatinterleave", "clone", "stack", "embedding", "setitem", "index", "full", "oneslike"}

STAGES = [  # fixed order = fixed palette slot
    ("proj", "Projections (QKV, out)"), ("mlp", "MLP"), ("norm", "Norms"),
    ("rec", "Recurrent core"), ("attn", "Full-attention core"), ("resid", "Residual add"),
    ("vocab", "Embedding / LM head"), ("setup", "Mask & RoPE tables"),
]
SID = {k: i for i, (k, _) in enumerate(STAGES)}


DROPPED = {}


def numel(s):
    return math.prod(s) if s else 1


def stage_of(module):
    m = re.sub(r":\d+$", "", module or "")
    m = re.sub(r"^lm\.(model\.language_model\.)?", "", m)
    if m.startswith("embed_tokens") or m.startswith("lm_head") or m == "":
        return "vocab"
    if m == "norm" or m.endswith(("layernorm", "q_norm", "k_norm", "linear_attn.norm")):
        return "norm"
    if re.search(r"(in_proj_\w+|q_proj|k_proj|v_proj|o_proj|out_proj)$", m):
        return "proj"
    if ".mlp" in m:
        return "mlp"
    if m.endswith("linear_attn"):
        return "rec"
    if m.endswith("self_attn"):
        return "attn"
    if re.fullmatch(r"layers\.\d+", m):
        return "resid"
    if m.startswith(("rotary_emb", "language_model")):
        return "setup"
    return "setup"


def build(phase, rows):
    by = {r[0]: r for r in rows}
    # TorchLens writes a parent label without its ":1" pass suffix; restore it
    canon = lambda p: p if p in by else (p + ":1" if p + ":1" in by else p)
    rows = [[r[0], [canon(p) for p in r[1]]] + r[2:] for r in rows]
    by = {r[0]: r for r in rows}
    mine = [r for r in rows if r[6] == phase]
    shape = lambda lab: by[lab][4] if lab in by else []
    nbytes = lambda lab: numel(shape(lab)) * DT.get(by[lab][5], 4) if lab in by else 0

    def is_view(r):
        t = r[2]
        if t in VIEW:
            return True
        if t in CAST:
            ps = [p for p in r[1] if p in by]
            return bool(ps) and by[ps[0]][5] == r[5]          # same dtype: alias
        return False

    in_phase = {r[0] for r in mine}
    # "live" = depends on the token input or on state from the other pass. Ops that
    # depend only on weights and constants (1+w, arange, zeros) are not activations:
    # they are treated as external inputs so they cannot masquerade as parallel work.
    live = {}
    for r in mine:
        live[r[0]] = any(
            (p in by and (by[p][2] == "input" or by[p][6] != phase)) or live.get(p, False)
            for p in r[1])
    real_label = {}                                              # label -> drawn?
    for r in mine:
        real_label[r[0]] = (not is_view(r)) and live[r[0]]
    dropped = [r for r in mine if not live[r[0]] and not is_view(r)]
    DROPPED[phase] = (len(dropped), sum(numel(r[4]) * DT.get(r[5], 4) for r in dropped))

    memo = {}
    def resolve(lab):
        """nearest drawn ancestors reached through free (view) ops"""
        if lab in memo:
            return memo[lab]
        if lab not in in_phase:
            out = set()
        elif real_label[lab]:
            out = {lab}
        else:
            out = set()
            for p in by[lab][1]:
                out |= resolve(p)
        memo[lab] = out
        return out

    nodes, idx = [], {}
    for r in mine:                                               # TorchLens order is topological
        if not real_label[r[0]]:
            continue
        lab, parents, typ, mod, shp, dt = r[0], r[1], r[2], r[3], r[4], r[5]
        out_b = numel(shp) * DT.get(dt, 4)
        in_b = sum(nbytes(p) for p in parents)
        w_b = 0
        flops = 0
        if typ == "linear":
            K = shape(parents[0])[-1]; N = shp[-1]; tok = numel(shp[:-1])
            flops = 2 * tok * K * N
            w_b = K * N * DT.get(dt, 2)
        elif typ in ("matmul", "bmm") and len(parents) >= 2:
            flops = 2 * numel(shp) * shape(parents[0])[-1]
        elif typ == "conv1d":
            flops = 2 * numel(shp) * 4
            w_b = shp[1] * 4 * DT.get(dt, 2)
        elif typ == "solvetriangular":
            flops = numel(shp) * (shp[-2] if len(shp) > 1 else 1)
        elif typ in ("softmax",):
            flops = 5 * numel(shp)
        elif typ in REDUCE:
            flops = numel(shape(parents[0])) if parents else numel(shp)
        elif typ in COPYLIKE:
            flops = 0
            if typ == "setitem":
                vb = min((nbytes(p) for p in parents if nbytes(p)), default=out_b)
                in_b, out_b = vb, vb
            if typ == "embedding":
                in_b = 0
        elif typ in TRANS:
            flops = 4 * numel(shp)
        else:
            flops = numel(shp)                                   # generic elementwise
        idx[lab] = len(nodes)
        nodes.append(dict(label=lab, type=typ, module=re.sub(r"^lm\.(model\.language_model\.)?", "", re.sub(r":\d+$", "", mod or "")),
                          stage=SID[stage_of(mod)], block=-1 if r[7] is None else r[7],
                          shape=shp, dtype=dt.replace("torch.", ""), flops=flops,
                          in_b=in_b, out_b=out_b, w_b=w_b, parents=sorted(resolve_all(parents, resolve))))
    # levels (ASAP), edges
    edges = set()
    for i, n in enumerate(nodes):
        ps = [idx[p] for p in n["parents"] if p in idx]
        n["pi"] = ps
        n["level"] = 1 + max((nodes[p]["level"] for p in ps), default=0)
        for p in ps:
            edges.add((p, i))
    # lanes: barycentre-seeking free-slot assignment, level by level
    byl = collections.defaultdict(list)
    for i, n in enumerate(nodes):
        byl[n["level"]].append(i)
    for lv in sorted(byl):
        taken = set()
        order = sorted(byl[lv], key=lambda i: (sum(nodes[p]["lane"] for p in nodes[i]["pi"]) / len(nodes[i]["pi"])) if nodes[i]["pi"] else 0)
        for i in order:
            ps = nodes[i]["pi"]
            want = round(0.7 * sum(nodes[p]["lane"] for p in ps) / len(ps)) if ps else 0
            cand = min((c for c in range(-40, 41) if c not in taken),
                       key=lambda c: (abs(c - want), abs(c)))
            nodes[i]["lane"] = cand
            taken.add(cand)
    return nodes, sorted(edges)


def resolve_all(parents, resolve):
    out = set()
    for p in parents:
        out |= resolve(p)
    return out


def pack(nodes, edges):
    mods, shapes = {}, {}
    def tab(d, v):
        return d.setdefault(v, len(d))
    types = {}
    cols = collections.defaultdict(list)
    for n in nodes:
        cols["type"].append(tab(types, n["type"]))
        cols["mod"].append(tab(mods, n["module"]))
        cols["shape"].append(tab(shapes, "x".join(map(str, n["shape"])) + " " + n["dtype"]))
        for k in ("stage", "block", "level", "lane", "flops", "in_b", "out_b", "w_b"):
            cols[k].append(n[k])
    return dict(n=len(nodes), cols=cols, types=list(types), mods=list(mods), shapes=list(shapes),
                edges=[x for e in edges for x in e], labels=[n["label"] for n in nodes])


def main():
    topo = json.load(gzip.open(TOPO))
    rows = topo["rows"]
    out = dict(model="Qwen/Qwen3.8-27B", revision=topo["revision"],
               stages=STAGES, phases={})
    layer_types = json.load(open(HERE.parent / "assets/config.json"))["text_config"]["layer_types"]
    out["layer_types"] = ["F" if t == "full_attention" else "R" for t in layer_types]
    for ph in ("prefill", "decode"):
        nodes, edges = build(ph, rows)
        out["phases"][ph] = pack(nodes, edges)
        out["phases"][ph]["constant_ops"] = DROPPED[ph][0]
        out["phases"][ph]["constant_bytes"] = DROPPED[ph][1]
        print(ph, "constant-only ops left out:", DROPPED[ph])
        L = max(n["level"] for n in nodes)
        print(ph, "drawn ops", len(nodes), "edges", len(edges), "levels", L,
              "GFLOP %.1f" % (sum(n["flops"] for n in nodes) / 1e9),
              "GB %.2f" % (sum(n["in_b"] + n["out_b"] + n["w_b"] for n in nodes) / 1e9),
              "max lane width", max(collections.Counter(n["level"] for n in nodes).values()))
    js = "window.QWEN_GRAPH=" + json.dumps(out, separators=(",", ":")) + ";\n"
    (HERE / "data.js").write_text(js)
    print("data.js", len(js) // 1024, "KiB")


if __name__ == "__main__":
    main()
