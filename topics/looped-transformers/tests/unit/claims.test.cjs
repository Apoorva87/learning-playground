/* Claims in the prose that are derived from the recorded data, recomputed from the raw summaries. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const read = f => JSON.parse(fs.readFileSync(path.join(__dirname, '../../assets', f), 'utf8'));
const S = read('inkling-capture-summary.json'), H = read('hop-summary.json');
const median = a => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const ratio = d => d.add_norm_last / d.keep_norm_last;

test('hop caveat: without the ring three-hop shortcut prompts the counts are 0 of 6, 3 of 6 and 6 of 6', () => {
  const rows = H.rows.filter(r => r.hops >= 2 && !(r.story === 'ring' && r.hops === 3));
  assert.deepEqual([1, 2, 3].map(p => rows.filter(r => r.passes === p && r.correct).length), [0, 3, 6]);
  assert.deepEqual([1, 2, 3].map(p => rows.filter(r => r.passes === p).length), [6, 6, 6]);
  const withShortcut = H.rows.filter(r => r.hops >= 2 && r.passes === 1 && r.correct);
  assert.ok(withShortcut.every(r => r.story === 'ring' && r.hops === 3), 'both one-pass successes are shortcut prompts');
});
test('gate wording: median q/k/v ratio 0.73, layer 6 ratio 4.39-7.33 and forget 0.81-0.98, MLP forget mean -0.75', () => {
  const qkv = S.gate_detail.filter(d => ['q', 'k', 'v'].includes(d.role)), l6 = qkv.filter(d => d.layer === 6);
  assert.equal(+median(qkv.map(ratio)).toFixed(2), 0.73);
  assert.equal(+Math.min(...l6.map(ratio)).toFixed(2), 4.39); assert.equal(+Math.max(...l6.map(ratio)).toFixed(2), 7.33);
  assert.equal(+Math.min(...l6.map(d => d.forget_mean)).toFixed(2), 0.81); assert.equal(+Math.max(...l6.map(d => d.forget_mean)).toFixed(2), 0.98);
  const mlp = S.gate_detail.filter(d => d.role === 'mlp');
  assert.equal(+(mlp.reduce((a, d) => a + d.forget_mean, 0) / mlp.length).toFixed(2), -0.75);
  assert.ok(l6.filter(d => d.pass === 3).every(d => ratio(d) > 4.3 && ratio(d) < 5.6), 'pass 3 at layer 6 is 4.4 to 5.5');
  assert.ok(l6.filter(d => d.pass === 2).every(d => ratio(d) > 6.3 && ratio(d) < 7.4), 'pass 2 at layer 6 is 6.4 to 7.3');
});
test('outside layer 6 the message is smaller than what is kept for q/k/v in most layers', () => {
  const rest = S.gate_detail.filter(d => ['q', 'k', 'v'].includes(d.role) && d.layer !== 6);
  assert.ok(rest.filter(d => ratio(d) < 1).length / rest.length > 0.8);
});
