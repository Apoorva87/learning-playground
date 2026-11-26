const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../engine/state.js');

const spec = (lit, fold, extra = {}) => ({ id: extra.id || 'b', scene: { level: 'macro', fold, pass: 0, camera: { yaw: 0, pitch: 0, dist: 10, tx: 0, ty: 0, tz: 0 }, lit, flow: { kind: 'none' }, ...extra.scene } });

test('beatFromScroll finds the beat under the anchor and the progress through it', () => {
  const r = S.beatFromScroll([-100, 300, 900], 400, 1400);
  assert.equal(r.beat, 1);
  assert.ok(Math.abs(r.progress - 100 / 600) < 1e-12);
});
test('beatFromScroll uses docEnd as the bottom edge of the last beat', () => {
  const r = S.beatFromScroll([-100, 300, 900], 1150, 1400);
  assert.equal(r.beat, 2);
  assert.ok(Math.abs(r.progress - 250 / 500) < 1e-12);
});
test('beatFromScroll above the first section is beat 0, progress 0', () => {
  assert.deepEqual(S.beatFromScroll([200, 600], 100, 1000), { beat: 0, progress: 0 });
});
test('beatFromScroll never returns NaN for a zero-height section', () => {
  const r = S.beatFromScroll([0, 0, 500], 10, 900);
  assert.ok(Number.isFinite(r.progress));
  assert.ok(r.progress >= 0 && r.progress <= 1);
});
test('beatFromScroll clamps progress past the end of the document', () => {
  assert.equal(S.beatFromScroll([0, 500], 5000, 1000).progress, 1);
});

test('store clamps beat, rounds it, clamps progress', () => {
  const s = S.createStore(4);
  s.set({ beat: 9.4, progress: 3 }, 'test');
  assert.deepEqual(s.get(), { beat: 3, progress: 1 });
  s.set({ beat: -2, progress: -1 }, 'test');
  assert.deepEqual(s.get(), { beat: 0, progress: 0 });
  s.set({ beat: 1.6, progress: NaN }, 'test');
  assert.deepEqual(s.get(), { beat: 2, progress: 0 });
});
test('store notifies once per change, with the source, and not for identical state', () => {
  const s = S.createStore(4); const seen = [];
  const off = s.subscribe((st, src) => seen.push([st.beat, st.progress, src]));
  s.set({ beat: 1, progress: 0.5 }, 'scroll');
  s.set({ beat: 1, progress: 0.5 }, 'scroll');
  off();
  s.set({ beat: 2, progress: 0 }, 'step');
  assert.deepEqual(seen, [[1, 0.5, 'scroll']]);
});

test('sceneAt starts at the previous beat and ends at this beat', () => {
  const beats = [spec(['A'], 0, { id: 'one' }), spec(['B'], 1, { id: 'two' })];
  const a = S.sceneAt(beats, 1, 0, false);
  assert.equal(a.fold, 0); assert.equal(a.lit.A, 1); assert.ok(!a.lit.B);
  const b = S.sceneAt(beats, 1, 1, false);
  assert.equal(b.fold, 1); assert.equal(b.lit.B, 1); assert.ok(!b.lit.A);
  assert.equal(b.beatId, 'two');
  const m = S.sceneAt(beats, 1, S.ARRIVE / 2, false);
  assert.ok(m.fold > 0 && m.fold < 1);
  assert.ok(m.lit.A > 0 && m.lit.A < 1 && m.lit.B > 0 && m.lit.B < 1);
});
test('flow.t is 0 until ARRIVE and 1 at the end', () => {
  const beats = [spec([], 0), spec([], 0, { scene: { flow: { kind: 'hop' } } })];
  assert.equal(S.sceneAt(beats, 1, S.ARRIVE, false).flow.t, 0);
  assert.equal(S.sceneAt(beats, 1, 1, false).flow.t, 1);
  assert.equal(S.sceneAt(beats, 1, 1, false).flow.kind, 'hop');
});
test('reduced motion cuts to the target for any progress above 0', () => {
  const beats = [spec(['A'], 0), spec(['B'], 1)];
  assert.deepEqual(S.sceneAt(beats, 1, 0.01, true).lit, { B: 1 });
  assert.equal(S.sceneAt(beats, 1, 0.01, true).fold, 1);
  assert.equal(S.sceneAt(beats, 1, 0, true).fold, 0);
});
test('first beat has no previous beat: scene equals its own spec at any progress', () => {
  const beats = [spec(['A'], 0.3)];
  assert.equal(S.sceneAt(beats, 0, 0, false).fold, 0.3);
  assert.equal(S.sceneAt(beats, 0, 0, false).lit.A, 1);
});
test('layers crossfade between levels', () => {
  const beats = [spec([], 0), spec([], 0, { scene: { level: 'micro', micro: 'message-gate' } })];
  const mid = S.sceneAt(beats, 1, S.ARRIVE / 2, false).layers;
  assert.equal(mid.length, 2);
  assert.ok(Math.abs(mid[0].alpha + mid[1].alpha - 1) < 1e-12);
  assert.deepEqual(S.sceneAt(beats, 1, 1, false).layers, [{ level: 'micro', micro: 'message-gate', alpha: 1 }]);
});
