#!/usr/bin/env node
/* Every number or quotation in the prose carries <span data-fact="id">. This script checks that
   each id exists in assets/facts.json and that each fact's source still says what we claim:
     { kind:'url',    url, section }                 a published page (not fetched here)
     { kind:'json',   file, path, expected }          a committed data file, path like "a.b.0"
     { kind:'config', file, path, expected }          the model's config.json (skipped when absent)
     { kind:'source', file, line, quote }             a line in the Transformers source at the pinned commit
   Usage: node check-facts.cjs [--facts f] [--transformers dir] page.html... */
const fs = require('node:fs'), path = require('node:path');
const TOPIC = path.resolve(__dirname, '..'), REPO = path.resolve(TOPIC, '../..');

const dig = (obj, p) => p.split('.').reduce((o, k) => (o === undefined || o === null ? undefined : o[k]), obj);
const same = (a, b) => (typeof a === 'number' && typeof b === 'number' ? Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)) : JSON.stringify(a) === JSON.stringify(b));

function checkFacts({ factsPath = path.join(TOPIC, 'assets/facts.json'), htmlFiles = [], root = TOPIC, transformers = null }) {
  const errors = [], warnings = [], facts = JSON.parse(fs.readFileSync(factsPath, 'utf8')), cited = new Set();
  for (const f of htmlFiles) {
    const html = fs.readFileSync(f, 'utf8');
    for (const [, list] of html.matchAll(/data-fact="([^"]+)"/g)) for (const id of list.split(/[\s,]+/).filter(Boolean)) {
      cited.add(id);
      if (!facts[id]) errors.push(`${path.basename(f)}: cites "${id}", which is not in facts.json`);
    }
  }
  for (const [id, fact] of Object.entries(facts)) {
    const s = fact.source;
    if (!s || !s.kind) { errors.push(`${id}: has no source`); continue; }
    if (s.kind === 'url') {
      if (!/^https?:\/\//.test(s.url || '')) errors.push(`${id}: url source needs a url`);
      if (!s.section) errors.push(`${id}: url source needs a section`);
    } else if (s.kind === 'json' || s.kind === 'config') {
      const file = path.resolve(s.kind === 'config' ? REPO : root, s.file);
      if (!fs.existsSync(file)) { (s.kind === 'config' ? warnings : errors).push(`${id}: ${s.kind} file ${s.file} ${s.kind === 'config' ? 'not found, skipped' : 'is missing'}`); continue; }
      const got = dig(JSON.parse(fs.readFileSync(file, 'utf8')), s.path);
      if (got === undefined) errors.push(`${id}: path ${s.path} not found in ${s.file}`);
      else if (!same(got, s.expected)) errors.push(`${id}: ${s.file} ${s.path} is ${JSON.stringify(got)}, the fact says ${JSON.stringify(s.expected)}`);
    } else if (s.kind === 'source') {
      if (!transformers) { warnings.push(`${id}: source line check skipped (pass --transformers <dir>)`); continue; }
      const file = path.join(transformers, s.file);
      const line = fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split('\n')[s.line - 1] : undefined;
      if (line === undefined) errors.push(`${id}: ${s.file} line ${s.line} not found`);
      else if (!line.includes(s.quote)) errors.push(`${id}: ${s.file} line ${s.line} does not contain "${s.quote}"`);
    } else errors.push(`${id}: unknown source kind ${s.kind}`);
  }
  return { errors, warnings, unused: Object.keys(facts).filter(id => !cited.has(id)) };
}

if (require.main === module) {
  const argv = process.argv.slice(2), opts = { htmlFiles: [] };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--facts') opts.factsPath = path.resolve(argv[++i]);
    else if (argv[i] === '--transformers') opts.transformers = path.resolve(argv[++i]);
    else opts.htmlFiles.push(path.resolve(argv[i]));
  }
  const r = checkFacts(opts);
  r.warnings.forEach(w => console.log('warning: ' + w));
  if (r.unused.length) console.log(`info: ${r.unused.length} facts are not cited by these pages`);
  r.errors.forEach(e => console.log('ERROR: ' + e));
  if (r.errors.length) process.exit(1);
  console.log('facts ok');
}
module.exports = { checkFacts };
