/* Hover and click detail: every data-driven cell carries info with the exact recorded value and the arithmetic behind it. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const L = require('../../engine/layout.js');
require('../../engine/micro.js'); require('../../engine/micro-walk.js');
const R = require('../../walkthrough/ref.js');
const A = path.join(__dirname, '../../assets/walkthrough');
const D = JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-data.json'), 'utf8')), M = JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-mla.json'), 'utf8'));
const data = { d: D, m: M, token: 4, head: 0 };
const objs = n => L.microObjects(n, data);
const byId = (n, id) => { const o = objs(n).find(x => x.id === id); assert.ok(o, `${n} ${id}`); return o; };
const text = o => o.info.title + ' | ' + o.info.lines.join(' | ');
const g = (v, d = 4) => String(Number(v.toPrecision(d)));

const SCENES = ['walk-tokens', 'walk-embed', 'walk-layer', 'walk-norm', 'walk-qkv', 'walk-gqa', 'walk-qknorm', 'walk-rope', 'walk-scores', 'walk-mix', 'walk-ffn', 'walk-stack', 'walk-head', 'walk-cache', 'walk-decode', 'walk-mla', 'walk-mla-absorb', 'walk-cachesize'];
for (const n of SCENES) test(`${n}: most objects carry info with a title and at least one line`, () => {
  const all = objs(n), withInfo = all.filter(o => o.info);
  assert.ok(withInfo.length >= Math.floor(all.filter(o => o.kind !== 'arrow').length * 0.6), `${withInfo.length} of ${all.length}`);
  for (const o of withInfo) { assert.ok(o.info.title && o.info.lines.length >= 1, o.id); assert.ok(o.info.lines.every(l => typeof l === 'string' && l.length), o.id); }
});
test('scores: a score cell shows the recorded score, the dot product it comes from, and the softmax weight', () => {
  const o = byId('walk-scores', 'tensor.raw.r4c2'), q = D.layer0.rope.q0_out[4], k = D.layer0.rope.k0_out[2], dot = R.dot(q, k);
  const t = text(o);
  assert.match(t, /is/); assert.match(t, /of/);
  assert.ok(t.includes(g(D.layer0.scores.heads[0].raw[4][2], 3)), t); assert.ok(t.includes(g(dot, 4)), 'the dot product ' + g(dot, 4) + ' in ' + t);
  assert.ok(t.includes('√128') || t.includes('11.31'));
  const p = byId('walk-scores', 'tensor.probs.r4c2'); assert.ok(text(p).includes(String(Number((D.layer0.scores.heads[0].probs[4][2] * 100).toPrecision(3))) + '%'), text(p));
  assert.match(text(byId('walk-scores', 'tensor.probs.r1c3')), /cannot look|masked|later/i);
});
test('embed: a cell shows its recorded value, token and the row norm', () => {
  const t = text(byId('walk-embed', 'tensor.h0.r4c3'));
  assert.ok(t.includes(g(D.embed.h0[4][3], 3))); assert.match(t, / is/); assert.ok(t.includes(g(D.embed.row_norms[4], 3)));
});
test('norm: the output cell shows x times the inverse root times the gain', () => {
  const t = text(byId('walk-norm', 'tensor.y.r0c0')), L0 = D.layer0.norm1;
  assert.ok(t.includes(g(D.embed.h0[4][0], 3)) && t.includes(g(L0.inv_rms[4], 4)) && t.includes(g(L0.gain[0], 3)) && t.includes(g(L0.out[4][0], 3)), t);
});
test('rope: a rotated cell shows the pair, the angle and both results', () => {
  const t = text(byId('walk-rope', 'tensor.after.r4c0')), Rp = D.layer0.rope, Q = D.layer0.qknorm;
  assert.ok(t.includes(g(Q.q0_out[4][0], 4)) && t.includes(g(Q.q0_out[4][64], 4)) && t.includes(g(Rp.q0_out[4][0], 4)), t); assert.ok(t.includes('4 rad') || t.includes('angle'));
});
test('head: a vocabulary bar shows the token, its ID, logit and probability', () => {
  const o = byId('walk-head', 'bar.0'), top = D.head.logits.top20[0], t = text(o);
  assert.ok(t.includes('Paris') && t.includes(String(top.id)) && t.includes(g(top.logit, 4)) && t.includes((top.prob * 100).toPrecision(3)), t);
});
test('gqa: query head 5 says it shares key/value head 2 with head 4', () => {
  const t = text(byId('walk-gqa', 'qh.5')); assert.match(t, /key\/value head 2/); assert.match(t, /head 4/);
  assert.match(text(byId('walk-gqa', 'kvh.2')), /query heads 4 and 5/);
});
test('qknorm: an output cell shows before, inverse rms, gain and after', () => {
  const t = text(byId('walk-qknorm', 'tensor.after.r4c2')), Q = D.layer0.qknorm;
  assert.ok(t.includes(g(D.layer0.qkv.q0[4][2], 3)) && t.includes(g(Q.q_gain[2], 3)) && t.includes(g(Q.q0_out[4][2], 3)), t);
});
test('mla-absorb: an explicit and an absorbed score cell both show their values and their difference', () => {
  const e = text(byId('walk-mla-absorb', 'tensor.explicit.r5c2')), a = text(byId('walk-mla-absorb', 'tensor.absorbed.r5c2'));
  assert.ok(e.includes(g(M.explicit.scores_head0[5][2], 4)) && a.includes(g(M.absorbed.scores_head0[5][2], 4)), e + ' / ' + a);
  assert.match(e, /difference/);
});
test('cachesize: a bar shows bytes per token per layer and the ratio to GQA', () => {
  const t = text(byId('walk-cachesize', 'cache.mha')); assert.ok(t.includes('8,192') && /2× GQA/.test(t), t);
  assert.match(text(byId('walk-cachesize', 'cache.mqa')), /8× smaller than GQA/); assert.match(text(byId('walk-cachesize', 'cache.gqa')), /Qwen3/);
});
test('stack: a layer bar shows the hidden-state norm and the logit-lens guess', () => {
  const t = text(byId('walk-stack', 'layer.24')); assert.ok(t.includes(g(D.stack.layer_norms_last[24], 4)) && /Paris/.test(t), t);
});

test('scores: with another head selected the tooltip does not claim head 0’s query-key product', () => {
  const o = L.microObjects('walk-scores', { d: D, m: M, token: 4, head: 1 }).find(x => x.id === 'tensor.raw.r4c2');
  assert.ok(!/query·key = /.test(text(o)), text(o)); assert.ok(text(o).includes(g(D.layer0.scores.heads[1].raw[4][2], 3)));
});
test('captions that are not tensors do not claim to be frames around a tensor', () => {
  const o = L.microObjects('walk-cachesize', { d: D, m: M }).find(x => x.id === 'cache.note');
  assert.ok(!/frame around one tensor/.test(text(o)), text(o));
});
