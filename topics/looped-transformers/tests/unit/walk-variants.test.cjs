/* What-if arithmetic for the design variants discussed in Article C, from Qwen3-0.6B's sizes (and DeepSeek-V2-Lite's for MLA). */
const test = require('node:test');
const assert = require('node:assert/strict');
const V = require('../../walkthrough/variants.js');
const cfg = V.QWEN3;

test('the Qwen3-0.6B config is the recorded one', () =>
  assert.deepEqual([cfg.layers, cfg.hidden, cfg.heads, cfg.kvHeads, cfg.headDim, cfg.ffn, cfg.vocab, cfg.maxPos, cfg.bytes], [28, 1024, 16, 8, 128, 3072, 151936, 32768, 2]));

test('attention: cache bytes per token per layer and at 32,768 tokens over 28 layers', () => {
  const a = k => V.attention(k, cfg);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.cachePerTokenPerLayer), [8192, 4096, 512]);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.cacheAtContext), [7516192768, 3758096384, 469762048]);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.cachePerToken), [229376, 114688, 14336]);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.kvHeads), [16, 8, 1]);
});
test('attention: key and value projection parameters per layer, and the whole attention block', () => {
  const a = k => V.attention(k, cfg);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.kvProjParams), [4194304, 2097152, 262144]);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.attnParams), [8388608, 6291456, 4456448]);
  assert.equal(28 * (a('mha').attnParams - a('gqa').attnParams), 58720256);
});
test('attention: decode reads the whole cache each step, and the arithmetic per byte of cache is the same for all three', () => {
  const g = V.attention('gqa', cfg), m = V.attention('mha', cfg);
  assert.equal(g.decodeReadAtContext, g.cacheAtContext);
  // per token and layer, scoring + mixing is 4 * heads * T * headDim operations on 2 * kvHeads * T * headDim * 2 bytes of cache
  assert.equal(g.decodeFlopsAtContext / g.decodeReadAtContext, m.decodeFlopsAtContext / m.decodeReadAtContext * 2);
  assert.ok(g.decodeFlopsAtContext === m.decodeFlopsAtContext, 'sharing keys does not reduce the arithmetic, only the bytes');
});
test('attention: MLA from DeepSeek-V2-Lite caches 1,152 bytes against 10,240 for full multi-head keys and values', () => {
  const m = V.attention('mla', cfg), d = V.DEEPSEEK_V2_LITE;
  assert.equal(m.cachePerTokenPerLayer, 1152); assert.equal(m.mhaEquivalentPerTokenPerLayer, 10240);
  assert.equal(Math.round(m.mhaEquivalentPerTokenPerLayer / m.cachePerTokenPerLayer * 100) / 100, 8.89);
  assert.equal(m.cachePerToken, 1152 * d.layers); assert.equal(m.kvProjParams, 2048 * 576 + 512 * 16 * 256);
});
test('position: learned absolute embeddings cost maxPos x hidden parameters, sinusoidal and RoPE none; RoPE rotates q and k in every layer', () => {
  assert.equal(V.position('learned', cfg).params, 33554432); assert.equal(V.position('sinusoidal', cfg).params, 0); assert.equal(V.position('rope', cfg).params, 0);
  assert.equal(V.position('rope', cfg).opsPerTokenPerLayer, 3 * (16 + 8) * 128);
  assert.equal(V.position('sinusoidal', cfg).opsPerToken, 1024);
  assert.match(V.position('rope', cfg).where, /queries and keys/);
});
test('norm: LayerNorm needs more operations and parameters than RMSNorm on the same vector', () => {
  const l = V.norm('layernorm', cfg), r = V.norm('rmsnorm', cfg);
  assert.equal(l.opsPerVector, 7 * 1024); assert.equal(r.opsPerVector, 4 * 1024); assert.equal(l.params, 2048); assert.equal(r.params, 1024);
  assert.equal(r.opsPerLayerForTokens, 2 * 5 * 4 * 1024); assert.equal(l.opsPerLayerForTokens, 2 * 5 * 7 * 1024);
});
test('feed-forward: ReLU at 4x width, SwiGLU at the 8/3 rule and at 3072, and a mixture of experts', () => {
  const f = k => V.ffn(k, cfg);
  assert.equal(f('relu').params, 8388608); assert.equal(f('swiglu').params, 9437184);
  assert.equal(f('swiglu-matched').width, 2730); assert.ok(Math.abs(f('swiglu-matched').params / f('relu').params - 1) < 0.001);
  assert.equal(f('relu').flopsPerToken, 2 * 8388608);
  const m = f('moe'); assert.equal(m.storedParams, 8 * 9437184); assert.equal(m.activeParams, 2 * 9437184); assert.equal(m.storedParams / m.activeParams, 4);
});
test('embeddings: untied costs a second vocab x hidden table', () => {
  assert.equal(V.embedding('tied', cfg).params, 155582464); assert.equal(V.embedding('untied', cfg).params, 2 * 155582464);
});
test('table(): each family returns rows whose cells equal the model functions, and says it is computed', () => {
  for (const fam of ['attention', 'position', 'norm', 'ffn', 'embedding']) {
    const t = V.table(fam, cfg);
    assert.ok(t.rows.length >= 2 && t.columns.length >= 2, fam); assert.match(t.note, /computed from the config/i);
  }
  const att = V.table('attention', cfg), gqa = att.rows.find(r => r.kind === 'gqa');
  assert.equal(gqa.cells.cachePerTokenPerLayer, 4096); assert.equal(gqa.cells.cacheAtContext, 3758096384);
});

test('attention: the five-token prompt of the running example, and the decode step', () => {
  const a = k => V.attention(k, cfg);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => x.cachePrompt), [1146880, 573440, 71680]);      // 5 tokens x bytes per token over 28 layers
  assert.equal(a('gqa').kvProjFlopsPrompt, 2 * 5 * 2097152 * 28);
  assert.equal(a('gqa').kvProjFlopsDecode, 2 * 2097152 * 28);
  assert.equal(a('gqa').cachePrompt, 5 * 114688);                                                           // the recorded 114,688 bytes per token
});
test('attention table: columns for the prompt cache, context cache and decode read, one row per variant', () => {
  const t = V.table('attention', cfg);
  assert.deepEqual(t.columns, ['kvHeads', 'kvProjParams', 'cachePerTokenPerLayer', 'cachePrompt', 'cacheAtContext', 'decodeReadAtContext']);
  assert.deepEqual(t.rows.map(r => r.kind), ['mha', 'mqa', 'gqa', 'mla']);
  assert.equal(t.rows.find(r => r.kind === 'mqa').cells.cachePrompt, 71680);
  const mla = t.rows.find(r => r.kind === 'mla').cells; assert.equal(mla.cachePerTokenPerLayer, 1152);
  for (const c of ['kvProjParams', 'cachePrompt', 'cacheAtContext', 'decodeReadAtContext']) assert.equal(mla[c], null, c + ' is not comparable across two models');
});

test('tables name each design in words and say what is and is not recorded', () => {
  for (const fam of ['position', 'embedding', 'norm', 'ffn', 'attention']) for (const r of V.table(fam, cfg).rows) assert.ok(r.label.length > 4 && r.label !== r.kind, `${fam}/${r.kind}: ${r.label}`);
  assert.match(V.table('attention', cfg).note, /nothing here is a new recorded run/i);
});
