const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const { fill, format } = require('../fill-walk.cjs');

test('format: significant digits, integers with separators, percentages', () => {
  assert.equal(format(0.000549747, '3'), '0.00055'); assert.equal(format(42.6112, '4'), '42.61'); assert.equal(format(151936, 'int'), '151,936');
  assert.equal(format(0.34574588, 'pct'), '34.6%'); assert.equal(format(12095, 'int'), '12,095'); assert.equal(format(-0.0554199, '3'), '-0.0554');
});
test('format: scientific notation uses superscript exponents', () => assert.equal(format(2.12e-7, 'sci'), '2.12 × 10⁻⁷'));
test('fill rewrites a recorded number and leaves the rest of the page alone', () => {
  const out = fill('<p>x <span data-walk="d:head.chosen.id" data-fmt="int">…</span> y</p>');
  assert.equal(out, '<p>x <span data-walk="d:head.chosen.id" data-fmt="int">12,095</span> y</p>');
});
test('fill fails loudly on a path that is not in the recording', () => assert.throws(() => fill('<span data-walk="d:nope.nothing">x</span>'), /no recorded value/));
test('Article C quotes the recording exactly: every data-walk number is current', () => {
  const f = path.join(__dirname, '../../transformer-walkthrough.html');
  if (!fs.existsSync(f)) return;
  const src = fs.readFileSync(f, 'utf8');
  assert.equal(fill(src), src, 'run: node tests/fill-walk.cjs transformer-walkthrough.html');
  assert.ok((src.match(/data-walk=/g) || []).length > 20, 'the page should cite many recorded numbers');
});
