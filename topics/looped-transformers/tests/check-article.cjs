#!/usr/bin/env node
/* Structure check for an article page: node check-article.cjs <article.html>...
   Every beat of kind "math" must give the story, the intuition and the math, and the math must
   show tensor shapes. Terms in the text must map to scene objects the layout can draw. */
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const Layout = require(path.join(__dirname, '../engine/layout.js'));
require(path.join(__dirname, '../engine/micro.js')); require(path.join(__dirname, '../engine/micro-walk.js'));            // registers the micro scenes so tensor ids can be checked

const SHAPE = /\[[^\]]*\d+[^\]]*\]|\[\s*T\s*,/;
function validIds(micro) {
  const ids = new Set(Layout.macroObjects(0).map(o => o.id));
  for (const name of Object.keys(Layout.MICRO)) Layout.microObjects(name).forEach(o => ids.add(o.id));
  const prefixes = new Set([...ids].flatMap(id => id.split('.').map((_, i, a) => a.slice(0, i + 1).join('.'))));   // 'tensor.h', 'bar.stable', 'pair' name everything beneath them
  return { has: id => ids.has(id) || prefixes.has(id) || /^hop\.\d+$/.test(id) || /^(L\d+|pass\.[123]|prelude|core|coda|full)$/.test(id) && Layout.idsFor(id).length > 0 };
}
/* The inner HTML of every <div class="... cls ..."> in html, counting nested divs. */
function blocks(html, cls) {
  const out = [], open = new RegExp(`<div[^>]*class="[^"]*\\b${cls}\\b[^"]*"[^>]*>`, 'g');
  let m;
  while ((m = open.exec(html))) {
    let depth = 1, i = open.lastIndex; const re = /<(\/?)div\b/g; re.lastIndex = i;
    for (let t; depth && (t = re.exec(html));) depth += t[1] ? -1 : 1, i = re.lastIndex;
    out.push(html.slice(open.lastIndex, i));
  }
  return out;
}
function loadBeats(file, html) {
  const m = /beats\/(beats-[abc])\.js/.exec(html); if (!m) throw new Error('no beats script in ' + file);
  const w = {}; vm.runInNewContext(fs.readFileSync(path.join(path.dirname(file), 'beats', m[1] + '.js'), 'utf8'), { window: w, LoopLayout: Layout });
  return Object.values(w)[0];
}

function check(file) {
  const errors = [], html = fs.readFileSync(file, 'utf8'), beats = loadBeats(file, html), valid = validIds();
  const sections = [...html.matchAll(/<section class="beat"[^>]*data-beat="([^"]+)"[^>]*>([\s\S]*?)<\/section>/g)].map(m => ({ id: m[1], html: m[2] }));
  if (sections.map(s => s.id).join() !== beats.map(b => b.id).join()) errors.push(`sections [${sections.map(s => s.id)}] and beats [${beats.map(b => b.id)}] differ`);
  for (const beat of beats) {
    const sec = sections.find(s => s.id === beat.id); if (!sec) continue;
    if (beat.kind === 'math') {
      for (const part of ['story', 'intuition', 'math']) if (!blocks(sec.html, part).length) errors.push(`${beat.id}: missing the ${part} block`);
      for (const b of blocks(sec.html, 'math')) if (!SHAPE.test(b.replace(/<[^>]+>/g, ''))) errors.push(`${beat.id}: a math block shows no tensor shape like [T, 2304]`);
    }
    for (const [, t] of sec.html.matchAll(/data-term="([^"]+)"/g)) if (!(beat.terms || {})[t]) errors.push(`${beat.id}: term "${t}" has no scene objects in the beat's terms`);
    for (const [t, ids] of Object.entries(beat.terms || {})) for (const id of ids) if (!valid.has(id)) errors.push(`${beat.id}: term "${t}" names unknown object "${id}"`);
    for (const id of beat.scene.lit) if (!valid.has(id)) errors.push(`${beat.id}: lit names unknown object "${id}"`);
  }
  return errors;
}
if (require.main === module) {
  let bad = 0;
  for (const f of process.argv.slice(2)) {
    const errs = check(path.resolve(f));
    for (const e of errs) console.log(`${path.basename(f)}: ${e}`);
    bad += errs.length;
  }
  if (bad) process.exit(1);
  console.log('article structure ok');
}
module.exports = { check };
