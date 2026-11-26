const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const R = require('../../engine/recorded.js');
const S = JSON.parse(fs.readFileSync(path.join(__dirname, '../../assets/inkling-capture-summary.json'), 'utf8'));

test('gate ratio table: 12 core layers, passes 2 and 3 for q, k, v, mlp, each = add_norm / keep_norm', () => {
  const t = R.gateRatio(S);
  assert.equal(t.rows.length, 12); assert.equal(t.head.length, 8);
  assert.deepEqual(t.rows.map(r => r.layer), [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
  const d = S.gate_detail.find(x => x.layer === 9 && x.pass === 3 && x.role === 'v');
  const cell = t.rows[3].cells[t.head.findIndex(h => h.pass === 3 && h.role === 'v')];
  assert.ok(Math.abs(cell - d.add_norm_last / d.keep_norm_last) < 1e-9);
  assert.ok(t.min > 0 && t.max > t.min);
});
test('expert overlap table copies the recorded numbers', () => {
  const t = R.overlap(S);
  assert.equal(t.rows.length, 12);
  const o = S.expert_overlap.find(x => x.layer === 17);
  assert.deepEqual(t.rows.find(r => r.layer === 17).cells, [o.p1p2, o.p2p3, o.all3, o.union_last]);
});
test('message table gives the weight each message puts on the previous pass final state', () => {
  const t = R.messages(S);
  assert.equal(t.rows.length, 12); assert.equal(t.head.length, 8);
  const m = S.messages.find(x => x.layer === 6 && x.before_visit === 18);
  assert.ok(Math.abs(t.rows[0].cells[0] - m.w_last.q[m.n_sources - 1]) < 1e-9);
  for (const r of t.rows) for (const c of r.cells) assert.ok(c >= 0 && c <= 1);
});
test('timing rows give the seconds of the prelude, each core pass and the coda', () => {
  const t = R.timing(S);
  assert.deepEqual(t.rows.map(r => r.label), ['prelude', 'core pass 1', 'core pass 2', 'core pass 3', 'coda']);
  assert.ok(Math.abs(t.rows.reduce((a, r) => a + r.seconds, 0) - 17.3) < 0.5);
});
test('html builder escapes labels and shows every number as text', () => {
  const html = R.tableHtml({ caption: 'a<b', head: [{ label: 'x&y' }], rows: [{ label: 'r', cells: [1.234] }], min: 0, max: 2 }, v => v.toFixed(2));
  assert.ok(html.includes('a&lt;b') && html.includes('x&amp;y') && html.includes('1.23'));
});
