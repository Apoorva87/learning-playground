"""Download the official pinned files. capture.py verifies the 18 shard hashes."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

manifest = json.loads(Path(__file__).with_name("workload.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
snapshot_download(
    repo_id=manifest["model"], revision=manifest["revision"], local_dir=args.destination,
    allow_patterns=["*.safetensors", "*.json", "*.jinja", "LICENSE"], max_workers=4,
)
