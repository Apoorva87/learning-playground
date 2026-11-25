"""Build the GitHub Pages artifact from the workspace's versioned public files."""
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
import os
import re
import shutil
import subprocess

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
ROOT_FILES = {"index.html", "guide.css", "feedback.js", "README.md", "preferences.md",
              "CODEX.md", "AGENTS.md", "install.sh"}
SITE_FOLDERS = {"tools", "topics", "skills"}
markdown = MarkdownIt("commonmark", {"html": True}).enable("table")


class Anchors(HTMLParser):
    """Find actual links without changing script contents in standalone lessons."""

    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.anchors = []
        self.lines = [0] + [m.end() for m in re.finditer("\n", source)]
        self.feed(source)
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            line, column = self.getpos()
            self.anchors.append((self.lines[line - 1] + column, self.get_starttag_text(), href))


def render_markdown(path, text):
    title = re.search(r"^# (.+)$", text, re.M)
    name = escape(title[1] if title else path.name)
    heading_ids = {}
    tokens = markdown.parse(text)
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            slug = re.sub(r"[^\w\s-]", "", tokens[index + 1].content.lower()).strip().replace(" ", "-")
            count = heading_ids.get(slug, 0)
            token.attrSet("id", f"{slug}-{count}" if count else slug)
            heading_ids[slug] = count + 1
    body = markdown.renderer.render(tokens, markdown.options, {})
    base = "../" * (len(path.parts) - 1)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{name} · Learning playground</title><link rel="stylesheet" href="{base}guide.css">'
            f'</head><body><main><nav><a href="{base}index.html">Learning playground</a></nav>'
            f'{body}</main></body></html>')


def link_reading_editions(path, source, files):
    for start, tag, href in reversed(Anchors(source).anchors):
        target = urlsplit(href)
        if target.scheme or target.netloc or not target.path.endswith(".md"):
            continue
        local = Path(os.path.normpath(path.parent / unquote(target.path)))
        if local not in files:
            continue
        new_href = urlunsplit(("", "", target.path + ".html", target.query, target.fragment))
        new_tag = re.sub(r"""(\bhref\s*=\s*)(['"])(.*?)\2""",
                         lambda m: m[1] + m[2] + escape(new_href, quote=True) + m[2],
                         tag, count=1, flags=re.I | re.S)
        source = source[:start] + new_tag + source[start + len(tag):]
    return source


def build():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {Path(p) for p in paths if p and (p in ROOT_FILES or Path(p).parts[0] in SITE_FOLDERS)}
    if OUT.exists():
        shutil.rmtree(OUT)
    for path in sorted(files):
        source = ROOT / path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular file for {path}")
        dest = OUT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".md":
            shutil.copy2(source, dest)
            html = render_markdown(path, source.read_text(encoding="utf-8"))
            dest = dest.with_suffix(".md.html")
        elif path.suffix == ".html" and not path.name.endswith((".src.html", ".template.html")):
            html = source.read_text(encoding="utf-8")
        else:
            shutil.copy2(source, dest)
            continue
        dest.write_text(link_reading_editions(path, html, files), encoding="utf-8")
    print(f"Built {len(files)} tracked source files into {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
