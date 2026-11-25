"""Run DeepTutor's CLI with an inspectable local Ollama request/reply exchange."""
import asyncio
import itertools
import json
import os
from pathlib import Path
import time

import httpx2 as httpx
from deeptutor_cli.main import main

folder = Path(os.environ["DEEPTUTOR_EXCHANGE_DIR"])
folder.mkdir(parents=True, exist_ok=True)
if any(folder.iterdir()):
    raise SystemExit("Use an empty exchange directory for each CLI run.")
counter = itertools.count(1)


async def exchange(self, request):
    if str(request.url) != "http://localhost:11434/v1/chat/completions" or request.method != "POST":
        raise ValueError(f"Unexpected model endpoint: {request.method} {request.url}")
    stem = f"{next(counter):03d}"
    tmp = folder / (stem + ".pending")
    tmp.write_bytes(await request.aread())
    tmp.replace(folder / (stem + ".request.json"))
    reply = folder / (stem + ".response")
    deadline = time.monotonic() + 1200
    while not reply.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Missing local model reply for {stem}")
        await asyncio.sleep(0.25)
    stream = json.loads(request.content).get("stream", False)
    content_type = "text/event-stream" if stream else "application/json"
    return httpx.Response(200, headers={"content-type": content_type}, content=reply.read_bytes())


httpx.AsyncHTTPTransport.handle_async_request = exchange
main()
