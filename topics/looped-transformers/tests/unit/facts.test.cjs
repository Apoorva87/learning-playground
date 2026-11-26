const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { checkFacts } = require('../check-facts.cjs');
const { checkLinks } = require('../check-links.cjs');

function sandbox(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'facts-'));
  for (const [name, body] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
    fs.writeFileSync(path.join(dir, name), typeof body === 'string' ? body : JSON.stringify(body));
  }
  return dir;
}
const page = ids => `<p>${ids.map(i => `<span data-fact="${i}">x</span>`).join(' ')}</p>`;
const base = { 'data.json': { a: { b: [10, 20.5] }, name: 'inkling' } };
const run = (dir, facts, html, extra = {}) => {
  fs.writeFileSync(path.join(dir, 'facts.json'), JSON.stringify(facts)); fs.writeFileSync(path.join(dir, 'page.html'), page(html));
  return checkFacts({ factsPath: path.join(dir, 'facts.json'), htmlFiles: [path.join(dir, 'page.html')], root: dir, ...extra });
};
const url = { claim: 'c', source: { kind: 'url', url: 'https://example.org/x', section: 'Intro' } };

test('a cited fact with a source passes', () => assert.deepEqual(run(sandbox(base), { f: url }, ['f']).errors, []));
test('citing an id that is not in facts.json fails', () => assert.match(run(sandbox(base), { f: url }, ['f', 'ghost']).errors.join('\n'), /ghost/));
test('a fact with no source fails, even if no page cites it', () => assert.match(run(sandbox(base), { f: url, g: { claim: 'no source' } }, ['f']).errors.join('\n'), /g.*source/));
test('a url source needs a url and a section', () => assert.match(run(sandbox(base), { f: { claim: 'c', source: { kind: 'url', url: 'https://e.org' } } }, ['f']).errors.join('\n'), /section/));
test('a json source compares the value at the path', () => {
  const ok = { claim: 'c', source: { kind: 'json', file: 'data.json', path: 'a.b.1', expected: 20.5 } };
  assert.deepEqual(run(sandbox(base), { f: ok }, ['f']).errors, []);
  const bad = { claim: 'c', source: { kind: 'json', file: 'data.json', path: 'a.b.1', expected: 21 } };
  assert.match(run(sandbox(base), { f: bad }, ['f']).errors.join('\n'), /f.*20.5.*21/);
});
test('a json source with a missing path fails', () => {
  const bad = { claim: 'c', source: { kind: 'json', file: 'data.json', path: 'a.z', expected: 1 } };
  assert.match(run(sandbox(base), { f: bad }, ['f']).errors.join('\n'), /a\.z/);
});
test('a source-line fact checks the quote on that line when a source tree is given', () => {
  const dir = sandbox({ ...base, 'tf/models/m.py': 'x = 1\ndef forward(self):\n    return gate\n' });
  const fact = q => ({ claim: 'c', source: { kind: 'source', file: 'models/m.py', line: 3, quote: q } });
  assert.deepEqual(run(dir, { f: fact('return gate') }, ['f'], { transformers: path.join(dir, 'tf') }).errors, []);
  assert.match(run(dir, { f: fact('return nothing') }, ['f'], { transformers: path.join(dir, 'tf') }).errors.join('\n'), /f.*line 3/);
  const none = run(dir, { f: fact('return gate') }, ['f']);
  assert.deepEqual(none.errors, []); assert.match(none.warnings.join('\n'), /skipped/);
});
test('facts nobody cites are listed as unused', () => assert.deepEqual(run(sandbox(base), { f: url, g: url }, ['f']).unused, ['g']));

test('links: relative links and anchors must resolve', () => {
  const dir = sandbox({ 'a.html': '<a href="b.html#top">b</a><a href="#here">h</a><a href="https://example.org">e</a><img src="img/x.png"><h2 id="here">h</h2>', 'b.html': '<h1 id="top">b</h1>', 'img/x.png': 'x' });
  assert.deepEqual(checkLinks([path.join(dir, 'a.html')]), []);
});
test('links: a missing file and a missing anchor are reported', () => {
  const dir = sandbox({ 'a.html': '<a href="nope.html">n</a><a href="b.html#gone">g</a><a href="#absent">a</a>', 'b.html': '<h1 id="top">b</h1>' });
  const errs = checkLinks([path.join(dir, 'a.html')]).join('\n');
  assert.match(errs, /nope\.html/); assert.match(errs, /gone/); assert.match(errs, /absent/);
});
