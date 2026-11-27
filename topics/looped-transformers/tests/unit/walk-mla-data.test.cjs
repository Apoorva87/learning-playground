/* The recorded MLA layer (DeepSeek-V2-Lite, layer 0) on the same sentence. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const file = path.join(__dirname, '../../assets/walkthrough/walkthrough-mla.json');
const M = JSON.parse(fs.readFileSync(file, 'utf8'));
const shape = a => (Array.isArray(a) ? [a.length, ...shape(a[0])] : []);
const T = M.tokens.length;

test('meta, config and tokens', () => {
  assert.equal(M.meta.sentence, 'The capital of France is'); assert.match(M.meta.revision, /^[0-9a-f]{40}$/);
  assert.deepEqual([M.config.hidden, M.config.heads, M.config.kv_lora_rank, M.config.qk_nope, M.config.qk_rope, M.config.v_head], [2048, 16, 512, 128, 64, 128]);
  assert.equal(M.config.q_lora_rank, null);
  assert.deepEqual(M.tokens.map(t => t.id), [100000, 549, 6077, 280, 7239, 317], 'the start-of-sequence token, then the five word pieces');
  assert.equal(T, 6);
});
test('the latent cache has the right shapes', () => {
  assert.deepEqual(shape(M.latent.c_kv), [T, 512]); assert.deepEqual(shape(M.latent.k_rope), [T, 64]);
  assert.deepEqual(shape(M.query.q_absorbed_head0), [T, 512]); assert.deepEqual(shape(M.explicit.k_nope_head0), [T, 128]);
});
test('explicit and absorbed attention give the same scores, and both match the model', () => {
  const a = M.explicit.scores_head0, b = M.absorbed.scores_head0;
  for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) assert.ok(Math.abs(a[i][j] - b[i][j]) <= 1e-3 * Math.max(1, Math.abs(a[i][j])), `(${i},${j}): ${a[i][j]} vs ${b[i][j]}`);
  assert.ok(M.absorbed.max_abs_diff < 1e-3);
  assert.ok(M.hf.max_abs_prob_diff < 1e-4);
  M.explicit.probs_head0.forEach((row, i) => { assert.ok(Math.abs(row.reduce((x, y) => x + y, 0) - 1) < 1e-4); for (let j = i + 1; j < T; j++) assert.equal(row[j], 0); });
});
test('cache bytes per token per layer: latent plus rope key against full multi-head keys and values', () => {
  const c = M.config, mla = (c.kv_lora_rank + c.qk_rope) * 2, mha = c.heads * (c.qk_nope + c.qk_rope + c.v_head) * 2;
  assert.equal(M.cache_bytes.mla_per_token_per_layer, mla); assert.equal(M.cache_bytes.mha_per_token_per_layer, mha);
  assert.equal(mla, 1152); assert.equal(mha, 10240);
});
test('the file stays small', () => assert.ok(fs.statSync(file).size < 1e6));
