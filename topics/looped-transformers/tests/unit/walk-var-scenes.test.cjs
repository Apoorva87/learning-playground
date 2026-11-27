/* The scenes behind the "how it changed" switches: one drawing per design, with numbers computed from the config (never recorded). */
const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../../engine/layout.js');
require('../../engine/micro.js'); require('../../engine/micro-walk.js'); require('../../engine/micro-var.js');
const V = require('../../walkthrough/variants.js');
const cfg = V.QWEN3;
const fam = { 'walk-var-attention': ['mha', 'mqa', 'gqa', 'mla'], 'walk-var-position': ['sinusoidal', 'learned', 'rope', 'alibi'], 'walk-var-norm': ['post-layernorm', 'pre-layernorm', 'pre-rmsnorm'], 'walk-var-ffn': ['relu', 'swiglu', 'moe'], 'walk-var-embed': ['tied', 'untied'] };
const key = n => n.replace('walk-var-', '').replace('embed', 'embedding');
const objs = (n, v) => L.microObjects(n, { variant: { [key(n)]: v } });
const labels = o => o.filter(x => x.label).map(x => x.label).join(' | ');
const n = x => x.toLocaleString('en-US');

for (const [name, variants] of Object.entries(fam)) for (const v of variants) test(`${name} / ${v}: unique ids, info on every shape, labels present`, () => {
  const o = objs(name, v), ids = o.map(x => x.id);
  assert.equal(new Set(ids).size, ids.length, 'ids are unique'); assert.ok(o.length >= 5);
  for (const x of o.filter(x => x.kind !== 'arrow')) assert.ok(x.info && x.info.title && x.info.lines.length, `${x.id} has info`);
  assert.match(labels(o), /\S/);
});
test('with no variant given each scene draws the design the model actually uses', () => {
  assert.match(labels(L.microObjects('walk-var-attention', {})), /8 key\/value heads/); assert.match(labels(L.microObjects('walk-var-position', {})), /rotat/i);
  assert.match(labels(L.microObjects('walk-var-norm', {})), /RMSNorm/); assert.match(labels(L.microObjects('walk-var-ffn', {})), /SwiGLU/); assert.match(labels(L.microObjects('walk-var-embed', {})), /same table/i);
});
test('attention: the number of key/value cells and the bytes in the labels follow variants.js', () => {
  const count = v => objs('walk-var-attention', v).filter(x => /^kvh\.\d+$/.test(x.id)).length;
  assert.deepEqual(['mha', 'mqa', 'gqa', 'mla'].map(count), [16, 1, 8, 1]);
  assert.equal(objs('walk-var-attention', 'gqa').filter(x => /^qh\.\d+$/.test(x.id)).length, 16);
  for (const [v, b] of [['mha', 8192], ['mqa', 512], ['gqa', 4096], ['mla', 1152]]) assert.ok(labels(objs('walk-var-attention', v)).includes(n(b) + ' B'), v);
  const g = labels(objs('walk-var-attention', 'gqa')); assert.ok(g.includes(n(V.attention('gqa', cfg).cacheAtContext)) || g.includes('3.5 GiB'), g);
  assert.match(labels(objs('walk-var-attention', 'mla')), /computed/i);
});
test('attention: each query head reads the key/value head the variant assigns it', () => {
  const read = (v, h) => objs('walk-var-attention', v).find(x => x.id === `qh.${h}`).info.lines.join(' ');
  assert.match(read('mha', 5), /key\/value head 5/); assert.match(read('mqa', 5), /key\/value head 0/); assert.match(read('gqa', 5), /key\/value head 2/); assert.match(read('mla', 5), /latent/);
});
test('position: where the position enters, parameters from variants.js', () => {
  const t = v => labels(objs('walk-var-position', v));
  assert.ok(t('learned').includes(n(33554432))); assert.match(t('sinusoidal'), /0 parameters|no parameters/i); assert.match(t('rope'), /every layer/i); assert.match(t('alibi'), /distance/i);
  assert.ok(objs('walk-var-position', 'learned').some(x => x.id === 'pos.add')); assert.ok(!objs('walk-var-position', 'rope').some(x => x.id === 'pos.add'));
});
test('norm: post-LN puts the norm after the residual add, pre-LN before the sublayer; operation counts from variants.js', () => {
  const y = (v, id) => objs('walk-var-norm', v).find(x => x.id === id).y;
  assert.ok(y('post-layernorm', 'nrm') < y('post-layernorm', 'add'), 'norm below the add'); assert.ok(y('pre-layernorm', 'nrm') > y('pre-layernorm', 'sub'), 'norm above the sublayer');
  assert.ok(labels(objs('walk-var-norm', 'pre-layernorm')).includes(n(7 * 1024))); assert.ok(labels(objs('walk-var-norm', 'pre-rmsnorm')).includes(n(4 * 1024)));
});
test('ffn: parameter counts in the labels match variants.js, and the mixture lights two of eight experts', () => {
  assert.ok(labels(objs('walk-var-ffn', 'relu')).includes(n(8388608))); assert.ok(labels(objs('walk-var-ffn', 'swiglu')).includes(n(9437184)));
  const m = objs('walk-var-ffn', 'moe'), ex = m.filter(x => /^expert\.\d$/.test(x.id)); assert.equal(ex.length, 8);
  assert.equal(ex.filter(x => x.info.lines.join(' ').includes('active')).length, 2); assert.ok(labels(m).includes(n(8 * 9437184)));
});
test('embedding: tied draws one table, untied two, with the counts from variants.js', () => {
  assert.equal(objs('walk-var-embed', 'tied').filter(x => /^table\./.test(x.id)).length, 1); assert.equal(objs('walk-var-embed', 'untied').filter(x => /^table\./.test(x.id)).length, 2);
  assert.ok(labels(objs('walk-var-embed', 'untied')).includes(n(2 * 155582464)));
});
test('every figure on the variant scenes says it is computed, not recorded', () => {
  for (const name of Object.keys(fam)) assert.match(labels(L.microObjects(name, {})), /computed from the config/i, name);
});

test('on-canvas labels stay short so they do not collide; the long wording lives in the hover detail', () => {
  for (const [name, variants] of Object.entries(fam)) for (const v of variants) for (const x of objs(name, v)) if (x.label && x.kind === 'cell') assert.ok(x.label.length <= 46, `${name}/${v}/${x.id}: ${x.label}`);
  assert.ok(labels(objs('walk-var-norm', 'pre-layernorm')).includes('per norm'));
});
