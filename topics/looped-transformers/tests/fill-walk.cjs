#!/usr/bin/env node
/* Fill (or check) the recorded numbers in Article C. Every <span data-walk="d:path" data-fmt="3">…</span>
   takes its text from the recording: d: is walkthrough-data.json, m: is walkthrough-mla.json.
   data-fmt: a number of significant digits (default 4), "int" (thousands separators), "pct" (percent, 3 digits), "sci" (2 digits). 
   node fill-walk.cjs page.html          rewrite the numbers in place
   node fill-walk.cjs --check page.html  exit 1 if any number differs from the recording */
const fs = require('node:fs'), path = require('node:path');
const A = path.join(__dirname, '../assets/walkthrough');
const ROOT = { d: JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-data.json'), 'utf8')), m: JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-mla.json'), 'utf8')) };
const dig = (o, p) => p.split('.').reduce((x, k) => (x === undefined ? x : x[Array.isArray(x) ? +k : k]), o);
function format(v, mode = '4') {
  if (typeof v !== 'number') return String(v);
  if (mode === 'int') return Math.round(v).toLocaleString('en-US');
  if (mode === 'pct') return Number((v * 100).toPrecision(3)) + '%';
  if (mode === 'sci') { const [m, e] = v.toExponential(2).split('e'); return `${m} × 10${e.replace(/^\+/, '').replace(/[-0-9]/g, c => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)])}`; }
  return String(Number(v.toPrecision(+mode)));
}
function resolve(spec, mode) {
  const [root, p] = spec.split(':'), v = dig(ROOT[root], p);
  if (v === undefined) throw new Error(`no recorded value at ${spec}`);
  return format(v, mode);
}
function fill(html) {
  return html.replace(/<span data-walk="([^"]+)"(?: data-fmt="([^"]+)")?>([^<]*)<\/span>/g, (m, spec, mode, _old) => `<span data-walk="${spec}"${mode ? ` data-fmt="${mode}"` : ''}>${resolve(spec, mode)}</span>`);
}
if (require.main === module) {
  const check = process.argv.includes('--check'), files = process.argv.slice(2).filter(a => a !== '--check');
  let bad = 0;
  for (const f of files) {
    const src = fs.readFileSync(f, 'utf8'), out = fill(src);
    if (check) { if (src !== out) { console.log(`${path.basename(f)}: recorded numbers are out of date (run fill-walk.cjs)`); bad++; } }
    else fs.writeFileSync(f, out);
  }
  if (bad) process.exit(1);
}
module.exports = { fill, format, resolve };
