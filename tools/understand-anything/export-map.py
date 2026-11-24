"""Export an Understand-Anything knowledge graph as a browsable offline HTML file."""

import argparse
import json
import re
import textwrap
from html import escape
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("graph", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
graph = json.loads(args.graph.read_text())
nodes = {n["id"]: n for n in graph["nodes"]}


def link(node_id):
    return f'<a href="#{escape(node_id, quote=True)}">{escape(nodes[node_id]["name"])}</a>'


def markup(text):
    # The trial articles use paragraphs, internal wiki links, and HTTPS citations.
    pattern = re.compile(r"\[\[([^\]]+)\]\]|\[([^\]]+)\]\((https://[^\s)]+)\)")
    parts, start = [], 0
    for match in pattern.finditer(text):
        parts.append(escape(text[start:match.start()]))
        wiki, label, url = match.groups()
        if wiki:
            slug, _, caption = wiki.partition("|")
            node_id = "article:" + slug
            if node_id in nodes:
                parts.append(f'<a href="#{escape(node_id, quote=True)}">{escape(caption or nodes[node_id]["name"])}</a>')
            else:
                parts.append(escape(caption or slug))
        else:
            parts.append(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>')
        start = match.end()
    return "".join(parts) + escape(text[start:])


def connections(node_id, outgoing):
    own, other = ("source", "target") if outgoing else ("target", "source")
    items = []
    for edge in graph["edges"]:
        if edge[own] == node_id:
            label = escape(edge["type"].replace("_", " "))
            reason = escape(edge.get("description", ""))
            items.append(f'<li><small>{label}</small>{link(edge[other])}<span>{reason}</span></li>')
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>None</p>"


pages = []
for node_id, node in nodes.items():
    content = node.get("knowledgeMeta", {}).get("content", "")
    if content:
        paragraphs = [p for p in content.strip().split("\n\n") if not p.startswith("# ")]
        body = "".join(f"<p>{markup(p)}</p>" for p in paragraphs)
    else:
        body = f'<p>{escape(node["summary"])}</p>'
    name = escape(node["name"])
    pages.append(f'''<article id="{escape(node_id)}" hidden>
      <small class="kind">{escape(node['type'])}</small><h2 tabindex="-1">{name}</h2>{body}
      <section class="connections"><h3>Connected nodes</h3><div class="paths">
        <div><h4>Links to this node</h4>{connections(node_id, False)}</div>
        <div><h4>Links from this node</h4>{connections(node_id, True)}</div>
      </div></section></article>''')

tour = "".join(f'<details open><summary>{escape(s["title"])}</summary><ul>' +
               "".join(f"<li>{link(i)}</li>" for i in s["nodeIds"]) +
               "</ul></details>" for s in graph["tour"])
directory = "".join(f'<li>{link(i)} <small>{escape(n["type"])}</small></li>'
                    for i, n in nodes.items())
first_node = graph["tour"][0]["nodeIds"][0]
title = escape(graph["project"]["name"])


def article_map():
    columns = [step["nodeIds"] for step in graph["tour"]]
    width = 232 * len(columns)
    height = 70 * max(map(len, columns)) + 12
    positions = {node_id: (12 + col * 232, 8 + row * 70)
                 for col, group in enumerate(columns) for row, node_id in enumerate(group)}
    items = []
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["type"] != "related" or a not in positions or b not in positions:
            continue
        x, y = positions[a]
        u, v = positions[b]
        if x == u:
            x += 206
            u += 206
            path = f'M{x} {y+26} C{x+29} {y+26}, {u+29} {v+26}, {u} {v+26}'
        else:
            x += 206 if x < u else 0
            u += 0 if x < u else 206
            mid = (x + u) / 2
            path = f'M{x} {y+26} C{mid} {y+26}, {mid} {v+26}, {u} {v+26}'
        items.append(f'<path data-from="{escape(a)}" data-to="{escape(b)}" d="{path}"/>')
    for node_id, (x, y) in positions.items():
        name = nodes[node_id]["name"]
        lines = textwrap.wrap(name, 26)
        spans = "".join(f'<tspan x="{x+10}" y="{y + 30 + (j-(len(lines)-1)/2)*16}">{escape(line)}</tspan>'
                        for j, line in enumerate(lines))
        items.append(f'<a href="#{escape(node_id)}" aria-label="{escape(name, quote=True)}">'
                     f'<rect x="{x}" y="{y}" width="206" height="52" rx="4"/><text>{spans}</text></a>')
    return (f'<div class="map-scroll"><svg aria-label="Article map. Select an article to follow its links." '
            f'viewBox="0 0 {width} {height}">{"".join(items)}</svg></div>')


css = """
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#faf9f4;color:#253129;font:1rem/1.58 "Source Sans 3",system-ui,sans-serif}
main{max-width:77rem;margin:auto;padding:clamp(1rem,4vw,3rem)}header{border-top:4px solid #315842;padding:1.2rem 0 1.5rem;border-bottom:1px solid #b8c2b6}
h1,h2,h3{font-family:Spectral,Georgia,serif;font-weight:500;line-height:1.13}h1{font-size:clamp(2.5rem,5vw,4.4rem);margin:.3rem 0 .6rem}h2{font-size:clamp(1.9rem,4vw,3rem);margin:.35rem 0 1rem}h3{font-size:1.5rem;margin:1rem 0}
a,summary{color:#24543d;text-underline-offset:.19em}a:focus-visible,summary:focus-visible,input:focus-visible{outline:3px solid #945728;outline-offset:3px}
header p{margin:0}header small,.kind{color:#5c655a}.layout{display:grid;grid-template-columns:minmax(14rem,19rem) minmax(0,1fr);gap:clamp(1.4rem,4vw,3.5rem);margin-top:1.4rem}
aside{border-right:1px solid #c7cec2;padding-right:1.2rem}aside h2{font:600 1rem/1.5 system-ui;margin:0 0 .7rem}aside label{font-size:.95rem;font-weight:600}input{font:inherit;width:100%;padding:.45rem .55rem;border:1px solid #849582;background:white;border-radius:.2rem;margin-bottom:.85rem}
summary{cursor:pointer;font-weight:650}details{margin:.75rem 0}ul{list-style:none;padding:0;margin:.4rem 0}aside li{margin:.26rem 0;padding-left:.6rem}aside a[aria-current]{font-weight:750;color:#182c1e}
article{max-width:51rem}.paths{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}.connections{border-top:1px solid #b8c2b6;margin-top:1.4rem}
h4{margin:0 0 .55rem;font-size:.93rem}.paths li{margin:.45rem 0;padding:.45rem .65rem;border-left:2px solid #88a18d;background:#eff2ea}.paths small,.paths span{display:block;color:#4c584c;font-size:.82rem}.paths a{font-weight:600}footer{margin-top:2rem;border-top:1px solid #b8c2b6;padding-top:.7rem;font-size:.9rem}
[hidden]{display:none!important}@media(max-width:45rem){.layout{display:flex;flex-direction:column}aside{border:0;padding:0}aside details{display:inline-block;vertical-align:top;margin:.3rem .9rem .3rem 0}aside details[open]{display:block}.paths{grid-template-columns:1fr}}
.map-scroll{overflow-x:auto}.map-scroll svg{width:100%;min-width:46rem;display:block}.map-scroll path{stroke:#b7c6b6;fill:none;stroke-width:1.3;opacity:.35}.map-scroll path.active{stroke:#316e50;stroke-width:2.8;opacity:1}.map-scroll rect{fill:#fbfcf8;stroke:#91a492}.map-scroll text{fill:#253129;font:14px system-ui}.map-scroll a[aria-current] rect,.map-scroll a:hover rect,.map-scroll a:focus-visible rect{fill:#dcebdc;stroke:#24543d;stroke-width:2}
"""
script = """
const pages = [...document.querySelectorAll('article')];
const search = document.querySelector('#search');
const listing = document.querySelector('#results');
const tour = document.querySelector('#tour');
function selectNode() {
  const id = decodeURIComponent(location.hash.slice(1));
  const page = pages.find(p => p.id === id) || document.getElementById(document.body.dataset.first);
  pages.forEach(p => p.hidden = p !== page);
  document.querySelectorAll('aside a, svg a').forEach(a => {
    if (a.getAttribute('href') === '#' + page.id) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  document.querySelectorAll('svg path').forEach(p => p.classList.toggle('active', [p.dataset.from,p.dataset.to].includes(page.id)));
  if (id) page.querySelector('h2').focus({preventScroll:true});
}
function filterNodes() {
  const term = search.value.trim().toLowerCase();
  listing.hidden = !term;
  tour.hidden = Boolean(term);
  let matches = 0;
  for (const item of listing.children) {
    const node = document.getElementById(item.querySelector('a').hash.slice(1));
    const text = [item.textContent, ...Array.from(node.querySelectorAll(':scope > p'), p => p.textContent)].join(' ');
    item.hidden = !text.toLowerCase().includes(term);
    if (!item.hidden) matches++;
  }
  document.querySelector('#empty').hidden = !term || matches > 0;
}
search.addEventListener('input', filterNodes);
window.addEventListener('hashchange', selectNode);
selectNode();
"""
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Understand‑Anything</title><style>{css}</style></head>
<body data-first="{escape(first_node)}"><main><header><small>Understand‑Anything · Knowledge graph</small>
<h1>{title}</h1><p>Pick an article on the map to highlight its links, or use the reading tour below.</p></header>
{article_map()}
<div class="layout"><aside><label for="search">Search the graph</label>
<input id="search" type="search" placeholder="Search by concept or number"><div id="tour"><h2>Reading tour</h2>{tour}</div>
<ul id="results" hidden>{directory}</ul><p id="empty" hidden>No matching nodes.</p></aside>
<div>{''.join(pages)}</div></div><footer>{len(nodes)} nodes · {len(graph["edges"])} relationships ·
Offline reading view of the Understand‑Anything graph</footer></main><script>{script}</script></body></html>'''
args.output.write_text(page + "\n")
