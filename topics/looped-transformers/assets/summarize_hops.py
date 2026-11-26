"""Summarise hop-probe.json into the table the articles quote: for the two- and three-hop prompts,
how many were answered correctly and the right answer's average share among the candidates, at
each loop count.  python summarize_hops.py hop-probe.json hop-summary.json"""
import json
import sys
from statistics import mean

rows = json.load(open(sys.argv[1]))["rows"]
multi = [r for r in rows if r["hops"] >= 2]
out = {"by_passes": [], "rows": [{k: r[k] for k in ("story", "hops", "order", "passes", "correct")} | {"p_among": r["scores"][r["answer"]]["p_among"]} for r in rows]}
for k in (1, 2, 3):
    sel = [r for r in multi if r["passes"] == k]
    out["by_passes"].append({"passes": k, "n": len(sel), "correct": sum(r["correct"] for r in sel),
                             "mean_p_among": round(mean(r["scores"][r["answer"]]["p_among"] for r in sel), 2)})
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["by_passes"]))
