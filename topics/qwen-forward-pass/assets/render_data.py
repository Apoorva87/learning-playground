"""Build the file-local atlas data from the measured run and the shared fixture."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
raw = json.loads((HERE/"whole-pass.json").read_text())
config = json.loads((HERE/"config.json").read_text())["text_config"]
manifest = json.loads((HERE/"workload.json").read_text())
inventory = json.loads((HERE/"weights.json").read_text())
occupancy = json.loads((HERE/"cpu-windows.json").read_text())
atlas = {"types":config["layer_types"], "tokens":[t.strip() for t in manifest["token_text"]],
         "threads":raw["threads"], "weight_inventory": inventory["bytes_by_component"]}

for phase in ("prefill", "decode"):
    data = raw[phase]
    blocks = []
    for i in range(len(atlas["types"])):
        name = f"block.{i}"
        events = [e for e in data["events"] if e["name"]==name or e["name"].startswith(name+".")]
        block = dict(next(e for e in events if e["name"]==name))
        block["stages"] = [e for e in events if e["name"].count(".")==2]
        linears = [e for e in events if "flops" in e]
        block["linear_bytes"] = sum(e["weight_bytes"] for e in linears)
        block["linear_flops"] = sum(e["flops"] for e in linears)
        block["cache"] = [e for e in data["cache"] if e["layer"]==i]
        blocks.append(block)
    linears = [e for e in data["events"] if "flops" in e]
    atlas[phase] = {**{k:data[k] for k in ("wall_us","cpu_us","top_id")}, "blocks":blocks,
        "occupancy":occupancy[phase],
        "linear_bytes":sum(e["weight_bytes"] for e in linears),
        "linear_flops":sum(e["flops"] for e in linears),
        "cache_bytes":sum(e["bytes"] for e in data["cache"]),
        "linears":[[round(e["start_us"]),round(e["wall_us"]),e["weight_bytes"],e["name"],e["flops"]] for e in linears],
        "ends":[e for e in data["events"] if e["name"] in ("embedding","final_norm","lm_head")]}
with np.load(HERE/"layer3-decode.npz",allow_pickle=False) as d:
    q,k,v,out = (d[n][0] for n in ("q","k","v","expected"))
    columns = list(range(q.shape[-1]))
    heads=[]
    for h in range(24):
        scores = (q[h] @ k[h//6].T)[0] * manifest["scale"]
        exp = np.exp(scores-np.max(scores)); prob=exp/exp.sum()
        np.testing.assert_allclose(prob@v[h//6],out[h,0],rtol=1e-4,atol=1e-5)
        heads.append(dict(scores=scores.tolist(),prob=prob.tolist(),out=out[h,0,columns].tolist()))
    atlas["attention"]=dict(columns=columns,heads=heads,values=v.tolist(),
                            queries=q[:,0,:].tolist(), keys=k.tolist())
assert len(atlas["tokens"])==17
(HERE/"atlas-data.js").write_text("window.QWEN_ATLAS="+json.dumps(atlas,separators=(",",":"),allow_nan=False)+";\n")

# A static figure lets the topic directory show real measurements before opening the explorer.
decode = atlas["decode"]
pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="316" viewBox="0 0 960 316" role="img" aria-labelledby="title desc">',
          '<title id="title">A Qwen decode, from 64 decoder blocks to one attention head</title>',
          '<desc id="desc">Measured CPU time for 64 blocks and a heatmap of the 24 query heads attending to 17 real positions in block 3.</desc>',
          '<rect width="960" height="316" fill="#f2f3e9"/>',
          '<g font-family="system-ui,sans-serif" fill="#213931">',
          '<text x="26" y="36" font-size="23" font-weight="650">Where did the time go?</text>',
          '<text x="26" y="65" font-size="16">64 decoder blocks on the M4 Max CPU</text>',
          '<text x="26" y="99" font-size="14">Cached decode · 1 token</text>',
          '<rect x="26" y="110" width="428" height="42" fill="#d8ded5"/>']
for block, kind in zip(decode["blocks"], atlas["types"]):
    left = 26 + 428 * block["start_us"] / decode["wall_us"]
    width = 428 * block["wall_us"] / decode["wall_us"]
    color = "#a77539" if kind == "full_attention" else "#32756b"
    pieces.append(f'<rect x="{left:.3f}" y="110" width="{width:.3f}" height="42" fill="{color}" stroke="#edf0e7" stroke-width=".4"/>')
pieces.extend(['<text x="26" y="174" font-size="14">0 s</text>',
               f'<text x="454" y="174" text-anchor="end" font-size="14">{decode["wall_us"]/1e6:.2f} s</text>',
               '<rect x="26" y="197" width="12" height="12" fill="#32756b"/><text x="45" y="208" font-size="14">Recurrent block</text>',
               '<rect x="186" y="197" width="12" height="12" fill="#a77539"/><text x="205" y="208" font-size="14">Full-attention block</text>',
               '<path d="M477 21V295" stroke="#bdc7ba"/>',
               '<text x="502" y="36" font-size="23" font-weight="650">Which tokens did it read?</text>',
               '<text x="502" y="65" font-size="16">24 query heads × 17 cache positions in block 3</text>'])
for h, head in enumerate(atlas["attention"]["heads"]):
    for pos, probability in enumerate(head["prob"]):
        pieces.append(f'<rect x="{543+pos*22}" y="{83+h*7}" width="21" height="6.5" fill="#14604f" fill-opacity="{.06+.94*probability:.5f}"/>')
pieces.extend(['<text x="502" y="95" font-size="13">H0</text><text x="502" y="248" font-size="13">H23</text>',
               '<text x="543" y="272" font-size="13">First token</text><text x="916" y="272" text-anchor="end" font-size="13">Newest</text>',
               '<text x="502" y="298" font-size="13">Darker cells received more attention. One scale across all heads.</text>',
               '</g></svg>'])
(HERE/"forward-preview.svg").write_text("\n".join(pieces)+"\n")
