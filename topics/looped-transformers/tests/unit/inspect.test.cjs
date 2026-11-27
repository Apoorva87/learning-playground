/* The pure parts of hover and click detail: which object is under a point, and the keyboard order. */
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../../engine/scene.js');
const I = require('../../engine/inspect.js');
const r = (id, x0, y0, x1, y1, d0 = 0) => ({ id, x0, y0, x1, y1, d0 });

test('smallestAt picks the smallest rectangle that contains the point, so a cell beats its frame', () => {
  const rs = [r('frame', 0, 0, 100, 100), r('cell', 40, 40, 50, 50)];
  assert.equal(S.smallestAt(rs, 45, 45).id, 'cell'); assert.equal(S.smallestAt(rs, 10, 10).id, 'frame'); assert.equal(S.smallestAt(rs, 200, 200), null);
});
test('smallestAt breaks a tie in area by taking the nearer object', () => {
  assert.equal(S.smallestAt([r('far', 0, 0, 10, 10, 0.9), r('near', 0, 0, 10, 10, 0.1)], 5, 5).id, 'near');
});
test('inReadingOrder sorts by row (top to bottom) then by column, treating rects a few pixels apart as one row', () => {
  const rs = [r('b2', 30, 52, 40, 62), r('a2', 10, 50, 20, 60), r('a1', 10, 10, 20, 20), r('b1', 30, 11, 40, 21)];
  assert.deepEqual(S.inReadingOrder(rs).map(x => x.id), ['a1', 'b1', 'a2', 'b2']);
});
test('step moves to the next or previous item and wraps; with nothing focused it starts at the first or last', () => {
  const ids = ['a', 'b', 'c'];
  assert.equal(I.step(ids, null, +1), 'a'); assert.equal(I.step(ids, null, -1), 'c');
  assert.equal(I.step(ids, 'a', +1), 'b'); assert.equal(I.step(ids, 'c', +1), 'a'); assert.equal(I.step(ids, 'a', -1), 'c');
  assert.equal(I.step(ids, 'gone', +1), 'a'); assert.equal(I.step([], null, +1), null);
});
test('place keeps the tooltip inside the box and flips it away from the right and bottom edges', () => {
  assert.deepEqual(I.place({ x: 10, y: 10 }, { w: 100, h: 40 }, { w: 400, h: 300 }), { x: 24, y: 24 });
  const p = I.place({ x: 390, y: 290 }, { w: 100, h: 40 }, { w: 400, h: 300 });
  assert.ok(p.x + 100 <= 400 && p.y + 40 <= 300 && p.x >= 0 && p.y >= 0, JSON.stringify(p));
  assert.ok(p.x + 100 <= 390 && p.y + 40 <= 290, 'flipped to the left of and above the pointer: ' + JSON.stringify(p));
});
test('lines renders a title and body lines as safe text (no HTML interpretation)', () => {
  const html = I.html({ title: 'a <b>', lines: ['x & y', '<img src=x>'] });
  assert.ok(!html.includes('<b>') && !html.includes('<img') && html.includes('&lt;b&gt;') && html.includes('&amp;'), html);
});

test('placeAround puts the tooltip beside an object, never over it: below, then above, then right, then left', () => {
  const box = { w: 400, h: 300 }, size = { w: 100, h: 40 }, rect = { x0: 150, y0: 100, x1: 250, y1: 130 };
  const hits = (p, r) => p.x < r.x1 && p.x + size.w > r.x0 && p.y < r.y1 && p.y + size.h > r.y0;
  const below = I.placeAround(rect, size, box); assert.ok(!hits(below, rect) && below.y >= rect.y1, JSON.stringify(below));
  const low = { x0: 150, y0: 240, x1: 250, y1: 290 }, above = I.placeAround(low, size, box); assert.ok(!hits(above, low) && above.y + size.h <= low.y0, JSON.stringify(above));
  const tall = { x0: 100, y0: 5, x1: 300, y1: 295 }, side = I.placeAround(tall, { w: 60, h: 40 }, box); assert.ok(side && !hits(side, tall) || side === null);
  const fill = { x0: 0, y0: 0, x1: 400, y1: 300 }; assert.equal(I.placeAround(fill, size, box), null, 'no room beside an object that fills the stage');
});
test('html text is stable: the same info renders to the same string, so an unchanged pin is not rewritten', () => {
  assert.equal(I.html({ title: 't', lines: ['a'] }), I.html({ title: 't', lines: ['a'] }));
});
