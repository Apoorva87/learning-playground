"""Summarize an Inkling 2 Mini capture into the figures the articles use.

Usage: python summarize_capture.py <capture.json> [summary.json]
Prints a readable report and optionally writes a compact JSON for the pages.
"""
import json
import sys
from collections import defaultdict
from statistics import mean, median


def overlap(routes):
    """Expert overlap between passes of the same core layer, averaged over tokens."""
    by = defaultdict(dict)  # layer -> pass -> experts[token]
    for rt in routes:
        if rt["pass"]:
            by[rt["layer"]][rt["pass"]] = [set(e) for e in rt["experts"]]
    rows = []
    for layer in sorted(by):
        p = by[layer]
        if not all(k in p for k in (1, 2, 3)):
            continue
        n = len(p[1])
        f = lambda a, b: mean(len(p[a][t] & p[b][t]) for t in range(n))
        last = lambda a, b: len(p[a][-1] & p[b][-1])
        rows.append({"layer": layer, "p1p2": round(f(1, 2), 2), "p2p3": round(f(2, 3), 2), "p1p3": round(f(1, 3), 2),
                     "all3": round(mean(len(p[1][t] & p[2][t] & p[3][t]) for t in range(n)), 2),
                     "last_p1p2": last(1, 2), "last_p2p3": last(2, 3),
                     "union_last": len(p[1][-1] | p[2][-1] | p[3][-1])})
    return rows


def unique_experts(routes):
    """Distinct experts touched per stage and pass across all tokens (bandwidth proxy)."""
    out = defaultdict(set)
    for rt in routes:
        key = (rt["layer"], rt["pass"])
        for e in rt["experts"]:
            out[key].update(e)
    return out


def main(path, out=None):
    cap = json.load(open(path))
    meta, runs = cap["meta"], cap["runs"]
    S = {"meta": {k: meta[k] for k in ("model", "revision", "transformers_commit", "torch", "dtype", "device",
                                         "prompt", "tokens", "load_s", "expert_param_dtype", "loop") if k in meta}}
    print("prompt tokens:", len(meta["tokens"]), "| load", meta.get("load_s"), "s | experts", meta.get("expert_param_dtype"))

    S["by_passes"] = []
    for k in (1, 2, 3):
        run = runs[f"prefill_{k}pass"]
        S["by_passes"].append({"passes": k, "visits": len(run["visits"]), "wall_s": run["wall_s"], "top": run["top"][:5]})
        print(f"\n{k} pass(es): {len(run['visits'])} visits, {run['wall_s']} s, top:",
              ", ".join(f"{t['text']!r} {t['p']:.3f}" for t in run["top"][:5]))

    main_run = runs["prefill_3pass"]
    stage_ms = defaultdict(float)
    for v in main_run["visits"]:
        stage_ms[(v["stage"], v["pass"])] += v["ms"]
    S["stage_ms"] = [{"stage": s, "pass": p, "ms": round(ms, 1)} for (s, p), ms in stage_ms.items()]
    print("\nprefill time by stage/pass (ms):", {f"{s}{'' if p is None else p}": round(ms) for (s, p), ms in stage_ms.items()})
    S["visits"] = [{k: v[k] for k in ("visit", "layer", "pass", "stage", "kind", "ms", "rms_last")} for v in main_run["visits"]]
    print("rms_last per visit:", [v["rms_last"] for v in main_run["visits"]])

    ov = overlap(main_run["routes"])
    S["expert_overlap"] = ov
    print("\nexperts shared between passes (of 8), mean over tokens:")
    for row in ov:
        print(f"  layer {row['layer']:2d}: p1&p2 {row['p1p2']:.2f}  p2&p3 {row['p2p3']:.2f}  p1&p3 {row['p1p3']:.2f}  all three {row['all3']:.2f}  | last token union {row['union_last']}")
    if ov:
        print("  overall mean p1&p2 %.2f, p2&p3 %.2f, all three %.2f" % (mean(r["p1p2"] for r in ov), mean(r["p2p3"] for r in ov), mean(r["all3"] for r in ov)))
    ue = unique_experts(main_run["routes"])
    S["distinct_experts"] = [{"layer": l, "pass": p, "n": len(s)} for (l, p), s in sorted(ue.items(), key=lambda x: (x[0][0], x[0][1] or 0))]

    g = defaultdict(list)
    for x in main_run["gates"]:
        g[(x["pass"], x["role"])].append(x)
    S["gates"] = []
    print("\nmessage gates (prefill, 3 passes):")
    for (p, role), xs in sorted(g.items()):
        row = {"pass": p, "role": role, "forget_mean": round(mean(x["forget_mean"] for x in xs), 4),
               "msg_abs_mean": round(mean(x["msg_abs_mean"] for x in xs), 4),
               "add_over_keep_last": round(mean(x["add_norm_last"] / max(x["keep_norm_last"], 1e-9) for x in xs), 4)}
        S["gates"].append(row)
        print(f"  pass {p} {role:3s}: forget {row['forget_mean']:+.3f}  |msg| {row['msg_abs_mean']:.3f}  ‖g⊙m‖/‖(1-f)⊙h‖ {row['add_over_keep_last']:.3f}")
    S["gate_detail"] = [{k: x[k] for k in ("layer", "pass", "role", "forget_mean", "msg_mean", "msg_abs_mean", "h_norm_last",
                                             "m_norm_last", "keep_norm_last", "add_norm_last", "forget_hist", "msg_hist",
                                             "forget_last_first64", "msg_last_first64")} for x in main_run["gates"]]

    # Headline figures quoted in the articles; facts.json points at these.
    qkv = [r["add_over_keep_last"] for r in S["gates"] if r["role"] in ("q", "k", "v")]
    mlp = [r["add_over_keep_last"] for r in S["gates"] if r["role"] == "mlp"]
    S["headline"] = {
        "overlap_p1p2": round(mean(r["p1p2"] for r in ov), 2), "overlap_p2p3": round(mean(r["p2p3"] for r in ov), 2),
        "overlap_all3": round(mean(r["all3"] for r in ov), 2),
        "overlap_all3_layer17": next(round(r["all3"], 2) for r in ov if r["layer"] == 17),
        "overlap_p1p2_layer17": next(round(r["p1p2"], 2) for r in ov if r["layer"] == 17),
        "last_token_union_min": min(r["union_last"] for r in ov), "last_token_union_max": max(r["union_last"] for r in ov),
        "gate_ratio_qkv_min": min(qkv), "gate_ratio_qkv_max": max(qkv), "gate_ratio_mlp_mean": round(mean(mlp), 3),
        "forget_mean_min": min(r["forget_mean"] for r in S["gates"]),
        # layer 6 (the first loop layer) behaves very differently from the rest, so the means above need the median beside them
        "gate_ratio_qkv_median": round(median(d["add_norm_last"] / d["keep_norm_last"] for d in S["gate_detail"] if d["role"] in ("q", "k", "v")), 2),
        "gate_ratio_qkv_layer6_min": round(min(d["add_norm_last"] / d["keep_norm_last"] for d in S["gate_detail"] if d["layer"] == 6 and d["role"] in ("q", "k", "v")), 2),
        "gate_ratio_qkv_layer6_max": round(max(d["add_norm_last"] / d["keep_norm_last"] for d in S["gate_detail"] if d["layer"] == 6 and d["role"] in ("q", "k", "v")), 2),
        "forget_layer6_qkv_min": round(min(d["forget_mean"] for d in S["gate_detail"] if d["layer"] == 6 and d["role"] in ("q", "k", "v")), 2),
        "forget_layer6_qkv_max": round(max(d["forget_mean"] for d in S["gate_detail"] if d["layer"] == 6 and d["role"] in ("q", "k", "v")), 2),
        "forget_mean_mlp": round(mean(d["forget_mean"] for d in S["gate_detail"] if d["role"] == "mlp"), 2),
        "wall_s_by_passes": [r["wall_s"] for r in S["by_passes"]],
        "core_pass_s_mean": round(mean(ms for (st, p), ms in stage_ms.items() if st == "core") / 1000, 2),
        "prelude_s": round(stage_ms[("prelude", None)] / 1000, 2), "coda_s": round(stage_ms[("coda", None)] / 1000, 2),
    }

    kv = sum(p["bytes"] for row in main_run.get("cache", []) for k, p in row["parts"].items() if k in ("keys", "values"))
    S["headline"]["kv_bytes_per_token"] = kv / len(meta["tokens"])        # every visit keeps its own keys and values

    S["messages"] = main_run["messages"]
    if main_run["messages"]:
        m0 = main_run["messages"][0]
        print(f"\nmessage sources per read: {m0['n_sources']}; first read (layer {m0['layer']}) weights:", m0["w_last"])
    S["residual"] = main_run["residual"]
    fin = [x for x in main_run["residual"] if x["role"] == "final_input"]
    if fin:
        print("final_input weights over block states (last token):", fin[-1]["w_last"])

    for name in ("prefill_3pass", "decode_3pass"):
        c = runs[name].get("cache", [])
        tot = sum(p["bytes"] for row in c for p in row["parts"].values())
        kv = sum(p["bytes"] for row in c for k, p in row["parts"].items() if k in ("keys", "values"))
        print(f"\n{name}: {len(c)} cache layers, {tot/1e6:.2f} MB total, {kv/1e6:.2f} MB keys+values")
        S[f"cache_{name}"] = [{"cache_layer": row["cache_layer"], "type": row["type"], "sliding_window": row.get("sliding_window"),
                               "parts": {k: {"shape": p["shape"], "bytes": p["bytes"]} for k, p in row["parts"].items()}} for row in c]
    d = runs["decode_3pass"]
    S["decode"] = {"input": d["input"], "wall_s": d["wall_s"], "top": d["top"][:5],
                   "visits": [{k: v[k] for k in ("visit", "layer", "pass", "stage", "ms")} for v in d["visits"]],
                   "expert_overlap": overlap(d["routes"])}
    print(f"decode of {d['input']!r}: {d['wall_s']} s, top:", ", ".join(f"{t['text']!r} {t['p']:.3f}" for t in d["top"][:5]))
    if out:
        json.dump(S, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
        import os
        print("\nwrote", out, f"{os.path.getsize(out)/1e3:.0f} kB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
