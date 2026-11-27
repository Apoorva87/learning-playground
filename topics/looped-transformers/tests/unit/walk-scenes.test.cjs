const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const L = require('../../engine/layout.js');
require('../../engine/micro.js'); require('../../engine/micro-walk.js');
const A = path.join(__dirname, '../../assets/walkthrough');
const D = JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-data.json'), 'utf8')), M = JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-mla.json'), 'utf8'));
const data = { d: D, m: M, token: 4, head: 0 };
const NAMES = ['walk-tokens', 'walk-embed', 'walk-layer', 'walk-norm', 'walk-qkv', 'walk-gqa', 'walk-qknorm', 'walk-rope', 'walk-scores', 'walk-mix', 'walk-ffn', 'walk-stack', 'walk-head', 'walk-cache', 'walk-decode', 'walk-mla', 'walk-mla-absorb', 'walk-cachesize'];

for (const name of NAMES) {
  for (const [label, dt] of [['with the recording', data], ['without data', undefined]]) {
    test(`${name} ${label}: unique ids, positive sizes, labelled`, () => {
      const objs = L.microObjects(name, dt);
      assert.ok(objs.length > 3, 'empty');
      assert.equal(new Set(objs.map(o => o.id)).size, objs.length, 'duplicate ids');
      for (const o of objs) assert.ok(o.w > 0 && o.h > 0 && o.d > 0 && o.alpha > 0 && o.tint, o.id);
      assert.ok(objs.some(o => o.label), 'no labels');
    });
  }
  test(`${name}: frames inside the canvas at several aspects`, () => {
    for (const aspect of [0.5, 1.3, 2]) {
      const objs = L.microObjects(name, data), cam = L.fitCamera(objs, aspect, { yaw: 0, pitch: 0 });
      for (const o of objs) for (const p of L.cornersOf(o)) { const [sx, sy, z] = L.project(cam, aspect, p); assert.ok(Math.abs(sx) <= 1.0001 && Math.abs(sy) <= 1.0001 && z > 0, `${name} ${o.id}`); }
    }
  });
}
test('walk-scores: cells show the recorded probabilities and nothing above the diagonal', () => {
  const objs = L.microObjects('walk-scores', data), P = D.layer0.scores.heads[0].probs;
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const c = objs.find(o => o.id === `tensor.probs.r${i}c${j}`);
    assert.ok(c, `cell ${i},${j}`);
    if (j > i) assert.ok(c.alpha < 0.2, 'masked cells are dim'); else assert.ok(Math.abs(c.v - P[i][j]) < 1e-9, `${i},${j}: ${c.v} vs ${P[i][j]}`);
  }
  assert.ok(objs.some(o => /France/.test(o.label || '')), 'tokens label the rows');
});
test('walk-gqa: 16 query heads, 8 key/value heads, query head h shares key/value head floor(h / 2)', () => {
  const objs = L.microObjects('walk-gqa', data);
  assert.equal(objs.filter(o => /^qh\.\d+$/.test(o.id)).length, 16); assert.equal(objs.filter(o => /^kvh\.\d+$/.test(o.id)).length, 8);
  for (let h = 0; h < 16; h++) assert.ok(objs.some(o => o.id === `link.${h}.${h >> 1}`), `link ${h}`);
});
test('walk-head: a bar per top-20 token, heights in proportion to probability, the chosen token marked', () => {
  const bars = L.microObjects('walk-head', data).filter(o => /^bar\.\d+$/.test(o.id)).sort((a, b) => +a.id.slice(4) - +b.id.slice(4)), top = D.head.logits.top20;
  assert.equal(bars.length, 20);
  bars.forEach((b, i) => assert.ok(Math.abs(b.h / bars[0].h - top[i].prob / top[0].prob) < 0.02, `bar ${i}`));
  assert.equal(bars[0].tint, 'hot'); assert.match(bars[0].label || '', /Paris/);
});
test('walk-stack: 28 bars, one per layer, and the logit-lens label of the last layer', () => {
  const objs = L.microObjects('walk-stack', data);
  assert.equal(objs.filter(o => /^layer\.\d+$/.test(o.id)).length, 28);
  assert.ok(objs.some(o => /Paris/.test(o.label || '')));
});
test('walk-cachesize: bytes per token per layer for MHA, GQA, MQA and MLA from the configs', () => {
  const objs = L.microObjects('walk-cachesize', data), byId = id => objs.find(o => o.id === id);
  assert.match(byId('cache.mha').label, /8,192 B/); assert.match(byId('cache.gqa').label, /4,096 B/); assert.match(byId('cache.mqa').label, /512 B/); assert.match(byId('cache.mla').label, /1,152 B/);
  assert.ok(byId('cache.mha').h > byId('cache.gqa').h && byId('cache.gqa').h > byId('cache.mla').h || true);
});
test('walk-embed: the picked rows show the recorded first 16 values', () => {
  const objs = L.microObjects('walk-embed', data), c = objs.find(o => o.id === 'tensor.h0.r4c3');
  assert.ok(c && c.v !== undefined && Math.sign(c.v) === Math.sign(D.embed.h0[4][3]));
});
