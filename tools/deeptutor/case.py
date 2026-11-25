"""Build the shared prompt or extract a completed public answer from a CLI turn."""
import argparse
import json
from pathlib import Path
import sys


def prompt() -> str:
    topic = Path(__file__).resolve().parents[2] / "topics/how-transformers-remember"
    brief = (topic / "cases/one-more-token/brief.md").read_text(encoding="utf-8")
    task = brief.split("## Shared teaching task\n\n", 1)[1].strip()
    guide = (topic / "assets/field-guide.md").read_text(encoding="utf-8")
    return f"{task}\n\n---\n\nSource notes supplied with the teaching task\n\n{guide}"


def answer(stream: Path) -> str:
    result, status = None, None
    with stream.open(encoding="utf-8") as lines:
        for line in lines:
            event = json.loads(line)
            if event["type"] == "result":
                result = event["metadata"]["response"]
            elif event["type"] == "done":
                status = event["metadata"]["status"]
    if status != "completed" or not isinstance(result, str) or not result.strip():
        raise SystemExit(f"DeepTutor did not return a complete answer (status {status!r}).")
    return result.rstrip("\n") + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("prompt")
    actions.add_parser("answer").add_argument("events", type=Path)
    args = parser.parse_args()
    sys.stdout.write(prompt() if args.action == "prompt" else answer(args.events))
