const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../../engine/layout.js');

test('physicalOf maps 48 visits onto 24 layers', () => {
  const got = [0, 5, 6, 17, 18, 29, 30, 41, 42, 47].map(L.physicalOf);
  assert.deepEqual(got, [0, 5, 6, 17, 6, 17, 6, 17, 18, 23]);
});
test('passOf is 0 outside the core and 1-3 inside it', () => {
  assert.deepEqual([0, 5, 6, 17, 18, 29, 30, 41, 42, 47].map(L.passOf), [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]);
});
test('every physical layer is visited once, except the core which is visited three times', () => {
  const counts = new Array(24).fill(0);
  for (let v = 0; v < 48; v++) counts[L.physicalOf(v)]++;
  assert.deepEqual(counts.map((c, i) => (i >= 6 && i < 18 ? 3 : 1)), counts);
});

test('folded stack has 48 visits on 24 distinct (x,y) positions', () => {
  const o = L.macroObjects(0).filter(o => /^V\d+$/.test(o.id));
  assert.equal(o.length, 48);
  assert.equal(new Set(o.map(o => o.x.toFixed(6) + ',' + o.y.toFixed(6))).size, 24);
});
test('a pass-2 visit sits behind its pass-1 visit, offset only along z, when folded', () => {
  const o = Object.fromEntries(L.macroObjects(0).map(o => [o.id, o]));
  assert.equal(o.V6.x, o.V18.x); assert.equal(o.V6.y, o.V18.y);
  assert.ok(o.V18.z !== o.V6.z && o.V30.z !== o.V18.z);
  assert.equal(o.V18.kind, 'ghost'); assert.equal(o.V6.kind, 'visit');
});
test('unrolled visits lie on a line with strictly increasing x', () => {
  const o = L.macroObjects(1).filter(o => /^V\d+$/.test(o.id));
  for (let i = 1; i < 48; i++) assert.ok(o[i].x > o[i - 1].x);
});
test('exactly 4 physical layers are full attention (5, 11, 17, 23)', () => {
  const full = new Set(L.macroObjects(0).filter(o => /^V\d+$/.test(o.id) && o.full).map(o => L.physicalOf(+o.id.slice(1))));
  assert.deepEqual([...full].sort((a, b) => a - b), [5, 11, 17, 23]);
});
test('the return arrow fades out as the stack unrolls', () => {
  const a = id => L.macroObjects(id).find(o => o.id === 'hop.return').alpha;
  assert.ok(a(0) > 0.9 && a(1) < 0.05);
});
test('idsFor expands layer, pass and region names', () => {
  assert.deepEqual(L.idsFor('L8'), ['V8', 'V20', 'V32']);
  assert.equal(L.idsFor('pass.2').length, 12);
  assert.equal(L.idsFor('core').length, 36);
  assert.equal(L.idsFor('prelude').length, 6);
  assert.equal(L.idsFor('coda').length, 6);
  assert.deepEqual(L.idsFor('msg.q.8'), ['msg.q.8']);
});

const corners = o => { const c = []; for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) c.push([o.x + sx * o.w / 2, o.y + sy * o.h / 2, o.z + sz * o.d / 2]); return c; };
for (const aspect of [0.5, 1, 2]) {
  for (const fold of [0, 0.5, 1]) {
    test(`fitCamera keeps every object inside the frame at aspect ${aspect}, fold ${fold}`, () => {
      const objs = L.macroObjects(fold);
      const cam = L.fitCamera(objs, aspect);
      for (const o of objs) for (const p of corners(o)) {
        const [sx, sy, depth] = L.project(cam, aspect, p);
        assert.ok(Math.abs(sx) <= 1.0001 && Math.abs(sy) <= 1.0001 && depth > 0, `${o.id} at ${sx},${sy}`);
      }
    });
  }
}
test('project puts the camera target at the screen centre', () => {
  const cam = { yaw: 0.4, pitch: 0.2, dist: 9, tx: 1, ty: 2, tz: 3 };
  const [sx, sy] = L.project(cam, 1.5, [1, 2, 3]);
  assert.ok(Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9);
});
test('microObjects returns nothing for an unknown scene', () => assert.deepEqual(L.microObjects('nope'), []));
