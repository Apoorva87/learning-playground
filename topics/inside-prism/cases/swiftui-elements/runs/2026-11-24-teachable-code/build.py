"""Rebuild the tool's walkthrough and our screen companion from the same checked source."""
import json
from pathlib import Path
import re
import subprocess
import sys

here = Path(__file__).resolve().parent
root = here.parents[5]
tool = root / ".local/tools/teachable-code"
prism = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else tool / "prism"
walkthrough = here / "walkthrough.html"
subprocess.run([sys.executable, str(tool / "source/scripts/build_lesson.py"), str(here),
                "--repo", str(prism), "--standalone", "--out", str(walkthrough)], check=True)
page = walkthrough.read_text()
data = json.loads(re.search(r'<script type="application/json" id="lesson">(.*?)</script>', page, re.S)[1])
companion = {"steps": data["steps"], "repo": data["repo"]}
payload = json.dumps(companion, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")
template = (here / "screen.template.html").read_text()
assert template.count("__SCREEN_DATA__") == 1
(here / "index.html").write_text(template.replace("__SCREEN_DATA__", payload))
