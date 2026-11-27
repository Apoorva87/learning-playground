/* The recorded Qwen3-0.6B-Base forward pass for "The capital of France is": schema and invariants. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const file = path.join(__dirname, '../../assets/walkthrough/walkthrough-data.json');
const D = JSON.parse(fs.readFileSync(file, 'utf8'));
const sum = a => a.reduce((x, y) => x + y, 0);
const shape = a => (Array.isArray(a) ? [a.length, ...shape(a[0])] : []);
const close = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

test('meta and config describe the pinned model and sentence', () => {
  assert.equal(D.meta.sentence, 'The capital of France is');
  assert.equal(D.meta.tokens.length, 5);
  assert.deepEqual(D.meta.tokens.map(t => t.text), ['The', ' capital', ' of', ' France', ' is']);
  assert.match(D.meta.revision, /^[0-9a-f]{40}$/); assert.equal(D.meta.dtype, 'float32');
  assert.deepEqual([D.config.hidden, D.config.layers, D.config.heads, D.config.kv_heads, D.config.head_dim, D.config.ffn, D.config.vocab], [1024, 28, 16, 8, 128, 3072, 151936]);
});
test('embedding rows and norms', () => {
  assert.deepEqual(shape(D.embed.h0), [5, 1024]);
  D.embed.h0.forEach((row, i) => assert.ok(close(Math.hypot(...row), D.embed.row_norms[i], 1e-4)));
});
test('layer 0 shapes', () => {
  const L = D.layer0;
  assert.deepEqual(shape(L.norm1.out), [5, 1024]); assert.equal(L.norm1.gain.length, 1024);
  assert.deepEqual(shape(L.qkv.q0), [5, 128]); assert.deepEqual(shape(L.qkv.k0), [5, 128]); assert.deepEqual(shape(L.qkv.v0), [5, 128]);
  assert.deepEqual(shape(L.qkv.q_last_heads), [16, 8]); assert.deepEqual(shape(L.qkv.k_last_heads), [8, 8]);
  assert.deepEqual(L.qkv.q_shape, [5, 16, 128]); assert.deepEqual(L.qkv.k_shape, [5, 8, 128]);
  assert.equal(L.qknorm.q_gain.length, 128); assert.deepEqual(shape(L.qknorm.q0_out), [5, 128]);
  assert.equal(L.rope.inv_freq.length, 64); assert.deepEqual(shape(L.rope.q0_out), [5, 128]);
  assert.equal(L.ffn.gate_last.length, 3072); assert.equal(L.ffn.prod_last.length, 3072); assert.equal(L.ffn.down_last.length, 1024);
  assert.deepEqual(shape(L.out.h1), [5, 1024]);
});
test('attention probabilities: 16 heads, causal, rows sum to one, and 2 query heads share each key/value head', () => {
  assert.equal(D.layer0.scores.heads.length, 16);
  for (const h of D.layer0.scores.heads) {
    assert.deepEqual(shape(h.probs), [5, 5]);
    h.probs.forEach((row, i) => { assert.ok(close(sum(row), 1, 1e-4), `row ${i}`); for (let j = i + 1; j < 5; j++) assert.equal(row[j], 0); });
  }
  assert.equal(D.config.heads / D.config.kv_heads, 2);
});
test('the stack and the head', () => {
  assert.equal(D.stack.layer_norms_last.length, 28); assert.equal(D.stack.logit_lens.length, 28);
  const top = D.head.logits.top20; assert.equal(top.length, 20);
  assert.ok(top.every((t, i) => i === 0 || top[i - 1].logit >= t.logit), 'sorted by logit');
  assert.equal(D.head.chosen.id, top[0].id); assert.ok(top[0].prob > 0 && top[0].prob < 1);
  assert.equal(D.head.logits.vocab, 151936); assert.equal(D.head.final_norm_out_last.length, 1024);
});
test('cache shapes and the decode step', () => {
  assert.deepEqual(D.cache.k_shape, [8, 5, 128]); assert.deepEqual(D.cache.v_shape, [8, 5, 128]); assert.equal(D.cache.layers, 28);
  assert.equal(D.decode.position, 5); assert.equal(D.decode.probs.length, 16);
  for (const row of D.decode.probs) { assert.equal(row.length, 6); assert.ok(close(sum(row), 1, 1e-4)); }
  assert.ok(D.decode.logits_top10.length === 10);
  assert.equal(D.decode.token.id, D.head.chosen.id, 'decode starts from the token prefill chose');
});
test('the file stays small enough to ship', () => assert.ok(fs.statSync(file).size < 2e6, fs.statSync(file).size + ' bytes'));
