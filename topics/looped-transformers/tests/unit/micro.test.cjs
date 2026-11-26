const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../../engine/layout.js');
require('../../engine/micro.js');

const NAMES = ['toy-core', 'message-gate', 'depth-attn', 'sliding-layer', 'full-layer', 'moe', 'embed', 'message-input', 'head'];
for (const name of NAMES) {
  test(`micro scene ${name} has unique ids, positive sizes and a labelled tensor`, () => {
    const objs = L.microObjects(name);
    assert.ok(objs.length > 3, name + ' is empty');
    assert.equal(new Set(objs.map(o => o.id)).size, objs.length, 'duplicate ids');
    for (const o of objs) assert.ok(o.w > 0 && o.h > 0 && o.d > 0 && o.alpha > 0 && o.tint, o.id);
    assert.ok(objs.some(o => /^tensor\./.test(o.id) && o.label && /\[/.test(o.label)), 'no tensor carries a shape label');
  });
  test(`micro scene ${name} frames inside the canvas at several aspects`, () => {
    for (const aspect of [0.5, 1.3, 2]) {
      const objs = L.microObjects(name), cam = L.fitCamera(objs, aspect, { yaw: 0, pitch: 0 });
      for (const o of objs) for (const p of L.cornersOf(o)) {
        const [sx, sy, z] = L.project(cam, aspect, p); assert.ok(Math.abs(sx) <= 1.0001 && Math.abs(sy) <= 1.0001 && z > 0);
      }
    }
  });
}
test('a tensor name expands to its frame and all of its cells', () => {
  const ids = L.idsFor('tensor.h');
  assert.ok(ids.length > 1 && ids.includes('tensor.h'));
  assert.ok(ids.every(id => id === 'tensor.h' || id.startsWith('tensor.h.')));
  assert.ok(L.microObjects('message-gate').some(o => o.id === 'tensor.h'));
});
test('the toy scene uses the toy sizes, the Inkling scenes use the config sizes', () => {
  assert.ok(L.microObjects('toy-core').some(o => /\[T, 16\]/.test(o.label || '')));
  assert.ok(L.microObjects('message-gate').some(o => /2304/.test(o.label || '')));
  assert.ok(L.microObjects('sliding-layer').some(o => /\b24\b/.test(o.label || '') && /128/.test(o.label || '')));
  assert.ok(L.microObjects('moe').filter(o => /^expert\.\d+$/.test(o.id)).length === 128);
});

const toyData = {
  passes: 3, tokens: [0, 2, 1, 5, 2, 3, 3, 6, 4, 1, 5, 0, 6, 7, 7, 4, 3],             // pairs A->C B->F C->D D->G E->B F->A G->H H->E, query D
  attn: [[0, 0, 0, 0.9, 0.05, 0, 0.05, 0, 0], [0, 0, 0, 0, 0, 0, 0.95, 0.05, 0], [0, 0, 0, 0, 0, 0, 0, 0.9, 0.1]],
  argmax: [6, 7, 4],
};
test('toy-hops: one row of attention per pass, brightest on the pair holding the pointer, labelled with node names', () => {
  const objs = L.microObjects('toy-hops', toyData);
  const cell = (t, i) => objs.find(o => o.id === `tensor.attn.r${t}c${i}`);
  assert.ok(cell(0, 3).alpha > cell(0, 0).alpha * 2, 'pass 1 should light pair 3');
  assert.ok(cell(1, 6).alpha > cell(1, 3).alpha * 2, 'pass 2 should light pair 6');
  assert.ok(cell(2, 7).alpha > cell(2, 6).alpha);
  assert.ok(objs.some(o => o.label === 'D→G'), 'pair labels use node letters');
  assert.ok(objs.some(o => /pass 2/.test(o.label || '') && /H/.test(o.label || '')), 'row labels name the node read out');
  assert.equal(objs.filter(o => /^tensor\.attn\.r\d+c0$/.test(o.id)).length, 3);
});
test('toy-hops without data still draws a frame', () => assert.ok(L.microObjects('toy-hops').length > 3));
test('toy-gate: cells carry signed values, f and g are not rescaled, h m out are scaled to their own max', () => {
  const v = c => Array.from({ length: 16 }, (_, i) => c * Math.sin(i + 1));
  const data = { gate: { h: v(3), m: v(-5), f: v(0.4), g: v(0.2), out: v(2) } };
  const objs = L.microObjects('toy-gate', data);
  const vals = name => objs.filter(o => new RegExp(`^tensor\\.${name}\\.r0c\\d+$`).test(o.id)).map(o => o.v);
  assert.equal(vals('h').length, 16);
  assert.ok(Math.max(...vals('h').map(Math.abs)) <= 1.0000001 && Math.max(...vals('h').map(Math.abs)) > 0.99);
  assert.ok(Math.abs(vals('f')[1] - 0.4 * Math.sin(2)) < 1e-9);
  assert.ok(vals('m').some(x => x < 0) && vals('m').some(x => x > 0));
});
test('two-ways: the carry row has no restart bus; the Inkling row has a bus, message paths and gates', () => {
  const ids = L.microObjects('two-ways').map(o => o.id);
  assert.ok(ids.includes('inkling.bus') && ids.some(i => /^inkling\.gate\./.test(i)) && ids.some(i => /^inkling\.msg\./.test(i)));
  assert.ok(!ids.some(i => /^carry\.bus/.test(i)));
  assert.equal(ids.filter(i => /^(carry|inkling)\.core\.\d$/.test(i)).length, 6);
});
test('stability: the rho 1.2 bars grow, the rho 0.6 bars settle', () => {
  const o = L.microObjects('stability'), bars = k => o.filter(x => new RegExp(`^bar\\.${k}\\.\\d+$`).test(x.id)).sort((a, b) => a.x - b.x).map(x => x.h);
  const grow = bars('unstable'), settle = bars('stable');
  assert.equal(grow.length, 8);
  assert.ok(grow[7] > grow[0] * 8);
  assert.ok(settle[7] - settle[6] < settle[1] - settle[0]);
});

test('the attention scenes show the input-dependent sink and the relative bias, and the full layer shows what differs', () => {
  const labels = n => L.microObjects(n).map(o => o.label || '').join(' | ');
  for (const n of ['sliding-layer', 'full-layer']) { assert.match(labels(n), /sink logit/); assert.match(labels(n), /relative bias/); }
  assert.match(labels('full-layer'), /log-scale/); assert.match(labels('full-layer'), /dynamic/);
  assert.doesNotMatch(labels('sliding-layer'), /log-scale|dynamic/);
});
test('message-input forms four messages per loop layer; the head ends in the softcap', () => {
  const ids = L.microObjects('message-input').map(o => o.id);
  for (const r of ['q', 'k', 'v', 'mlp']) assert.ok(ids.includes('tensor.msg.' + r) && ids.includes('tensor.query.' + r), r);
  assert.match(L.microObjects('head').map(o => o.label || '').join(' | '), /tanh\(x \/ 30\)/);
});
