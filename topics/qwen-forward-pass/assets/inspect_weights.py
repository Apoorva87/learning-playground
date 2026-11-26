"""Index the checkpoint's on-disk tensors without loading their numeric values."""
import argparse
import json
from pathlib import Path

home = Path(__file__).resolve().parent
spec = json.loads((home/"workload.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
index = json.loads((args.checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
components = {}
layers = [{} for _ in range(64)]
seen = set()
for shard in spec["shards"]:
    with (args.checkpoint/shard["name"]).open("rb") as stream:
        length = int.from_bytes(stream.read(8),"little")
        header = json.loads(stream.read(length))
    for key,item in header.items():
        if key=="__metadata__":
            continue
        assert index[key] == shard["name"] and key not in seen and item["dtype"]=="BF16"
        seen.add(key)
        size = item["data_offsets"][1]-item["data_offsets"][0]
        if key.startswith("model.language_model.layers."):
            layer = int(key.split(".")[3])
            module = key.split(".")[4]
            layers[layer][module] = layers[layer].get(module,0)+size
            component = "decoder_blocks"
        elif key.startswith("model.visual."):
            component = "vision"
        elif key.startswith("mtp."):
            component = "multi_token_prediction"
        else:
            component = key
        components[component] = components.get(component,0)+size
assert seen==index.keys() and sum(components.values()) == 2*spec["checkpoint_parameters"]
(home/"weights.json").write_text(json.dumps(dict(tensors=len(seen),bytes_by_component=components,
                                                bytes_by_block=layers),indent=2)+"\n")
