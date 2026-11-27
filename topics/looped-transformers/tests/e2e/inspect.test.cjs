/* Hover and click detail on the real scenes: tooltip, pinned inspector, keyboard, touch, and cleanup on beat change. */
const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

async function go(page, name) {
  await page.waitForFunction(() => window.LOOP.walkReady);
  await page.evaluate(n => { const i = window.LOOP.beats.findIndex(b => b.id === n); window.LOOP.store.set({ beat: i, progress: 1 }, 'test'); }, name);
  await page.waitForTimeout(80);
}
/* The centre of an object in page coordinates. */
const centre = (page, id) => page.evaluate(i => { const p = window.LOOP.sceneApi.screenOf(i), r = window.LOOP.sceneApi.canvas().getBoundingClientRect(); return p && { x: r.left + p[0], y: r.top + p[1] }; }, id);

for (const renderer of ['', '&renderer=2d']) {
  test(`hover shows a tooltip, click pins it in the inspector, Escape clears (${renderer ? '2d' : 'webgl'})`, async () => {
    const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1' + renderer);
    await go(page, 'walk-scores');
    const c = await centre(page, 'tensor.raw.r4c2');
    await page.mouse.move(c.x, c.y);
    const tip = page.locator('#tip');
    await tip.waitFor({ state: 'visible' });
    const t = await tip.innerText();
    assert.match(t, /Score of “ is” looking at “ of”/); assert.match(t, /query·key = 91\.17/);
    const box = await tip.boundingBox(), stage = await page.locator('.stage-box').boundingBox();
    assert.ok(box.x >= stage.x && box.y >= stage.y && box.x + box.width <= stage.x + stage.width + 1 && box.y + box.height <= stage.y + stage.height + 1, 'tooltip stays inside the stage');
    await page.mouse.click(c.x, c.y);
    const ins = page.locator('#inspector');
    assert.match(await ins.innerText(), /Score of “ is” looking at “ of”/); assert.match(await ins.innerText(), /softmax weight 37%/);
    assert.equal(await ins.getAttribute('aria-live'), 'polite');
    await page.mouse.move(5, 5); await page.waitForTimeout(50);
    assert.ok(await tip.isHidden(), 'moving off hides the tooltip'); assert.match(await ins.innerText(), /Score of/, 'the pin stays');
    await page.keyboard.press('Escape');
    assert.doesNotMatch(await ins.innerText(), /Score of/);
    assert.deepEqual(page.errors, []);
    await page.context().close();
  });
}
test('a masked cell explains why it is empty', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-scores');
  const c = await centre(page, 'tensor.raw.r1c3'); await page.mouse.move(c.x, c.y);
  await page.locator('#tip').waitFor({ state: 'visible' });
  assert.match(await page.locator('#tip').innerText(), /cannot look at/);
  await page.context().close();
});
test('keyboard: Tab to the scene, arrows move through objects, Enter pins, and the focus ring and tooltip follow', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-head');
  await page.locator('#scene').focus();
  assert.equal(await page.evaluate(() => document.activeElement.id), 'scene');
  await page.keyboard.press('ArrowRight');
  const first = await page.locator('#tip').innerText();
  assert.ok(first.length > 5 && await page.locator('#tip').isVisible(), 'the tooltip shows for the focused object');
  const ring = await page.locator('#ring').boundingBox(); assert.ok(ring && ring.width > 2, 'a focus ring marks it');
  await page.keyboard.press('ArrowRight');
  const second = await page.locator('#tip').innerText(); assert.notEqual(second, first);
  await page.keyboard.press('ArrowLeft'); assert.equal(await page.locator('#tip').innerText(), first, 'ArrowLeft returns');
  await page.keyboard.press('Enter');
  assert.equal((await page.locator('#inspector').innerText()).includes(first.split('\n')[0]), true, 'Enter pins the same text');
  await page.keyboard.press('Escape');
  assert.ok(await page.locator('#tip').isHidden());
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
test('a vocabulary bar shows token, ID, logit and probability', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-head');
  const c = await centre(page, 'bar.0'); await page.mouse.click(c.x, c.y);
  const t = await page.locator('#inspector').innerText();
  assert.match(t, /Paris/); assert.match(t, /token ID \d+/); assert.match(t, /logit/); assert.match(t, /%/);
  await page.context().close();
});
test('changing the beat clears the tooltip, the ring and the pin', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-scores');
  const c = await centre(page, 'tensor.raw.r4c2'); await page.mouse.click(c.x, c.y);
  assert.match(await page.locator('#inspector').innerText(), /Score of/);
  await go(page, 'walk-embed');
  assert.ok(await page.locator('#tip').isHidden()); assert.ok(await page.locator('#ring').isHidden());
  assert.doesNotMatch(await page.locator('#inspector').innerText(), /Score of/);
  await page.context().close();
});
test('touch: a tap pins the object under the finger', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1', { width: 390, height: 844, hasTouch: true });
  await go(page, 'walk-head');
  const c = await centre(page, 'bar.0'); await page.touchscreen.tap(c.x, c.y);
  assert.match(await page.locator('#inspector').innerText(), /Paris/);
  await page.context().close();
});
test('a pin follows a redraw of the same beat, showing the new value', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-scores');
  const c = await centre(page, 'tensor.raw.r4c2'); await page.mouse.click(c.x, c.y);
  const before = await page.locator('#inspector').innerText();
  const want = await page.evaluate(() => { const w = window.__walk; window.LOOP.sceneApi.setData('walk', { ...w, head: 1 }); return +w.d.layer0.scores.heads[1].raw[4][2].toPrecision(3); });
  const after = await page.locator('#inspector').innerText();
  assert.notEqual(after, before); assert.ok(after.includes(String(want)), `${want} in ${after}`);
  await page.context().close();
});

test('on the real article, arrow keys on the focused scene do not move the article to another beat', async () => {
  const page = await open(env, 'transformer-walkthrough.html?test=1');
  await page.waitForFunction(() => window.LOOP.walkReady);
  await page.evaluate(() => { const i = window.LOOP.beats.findIndex(b => b.id === 'evo-attention'); window.LOOP.drivers.goTo('evo-attention'); });
  await page.waitForTimeout(200);
  const before = await page.evaluate(() => window.LOOP.state().beat);
  await page.locator('#scene').focus();
  for (const k of ['ArrowRight', 'ArrowDown', 'ArrowRight', 'Home', 'End', 'ArrowLeft']) await page.keyboard.press(k);
  await page.waitForTimeout(400);
  const after = await page.evaluate(() => ({ beat: window.LOOP.state().beat, mode: window.LOOP.state().mode }));
  assert.equal(after.beat, before); assert.equal(after.mode, 'scroll');
  assert.ok(await page.locator('#tip').isVisible(), 'the tooltip is still showing the focused object');
  await page.context().close();
});
test('keyboard focus is announced in a live region with the same words as the tooltip', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-head');
  await page.locator('#scene').focus(); await page.keyboard.press('ArrowRight');
  const tip = (await page.locator('#tip').innerText()).split('\n')[0];
  assert.equal(await page.locator('#inspect-live').getAttribute('aria-live'), 'polite');
  assert.ok((await page.locator('#inspect-live').textContent()).includes(tip), 'announced: ' + await page.locator('#inspect-live').textContent());
  await page.context().close();
});
test('a tooltip never covers the object it describes, on a phone', async () => {
  const page = await open(env, 'transformer-walkthrough.html?test=1', { width: 390, height: 844 });
  await page.waitForFunction(() => window.LOOP.walkReady);
  await page.evaluate(() => window.LOOP.drivers.goTo('evo-norm')); await page.waitForTimeout(200);
  await page.locator('#scene').focus();
  const bad = [];
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press('ArrowRight');
    const r = await page.evaluate(() => { const t = document.getElementById('tip'), g = document.getElementById('ring'); if (t.hidden) return { hidden: true }; const a = t.getBoundingClientRect(), b = g.getBoundingClientRect(); const w = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)), h = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top)); return { cover: w * h / Math.max(1, b.width * b.height) }; });
    if (r.cover > 0.05) bad.push(r.cover);
  }
  assert.deepEqual(bad, []);
  await page.context().close();
});
test('a tooltip or pin does not go stale when a design switch redraws the scene', async () => {
  const page = await open(env, 'transformer-walkthrough.html?test=1');
  await page.waitForFunction(() => window.LOOP.walkReady);
  await page.evaluate(() => window.LOOP.drivers.goTo('evo-attention')); await page.waitForTimeout(200);
  await page.evaluate(() => document.querySelector('[data-variants="attention"] [data-variant="gqa"]').click());
  await page.locator('#scene').focus();
  for (let i = 0; i < 40 && !/4,096/.test(await page.locator('#tip').innerText().catch(() => '')); i++) { await page.keyboard.press('ArrowRight'); }
  await page.evaluate(() => document.querySelector('[data-variants="attention"] [data-variant="mqa"]').click());
  const t = (await page.locator('#tip').isVisible()) ? await page.locator('#tip').innerText() : '';
  assert.doesNotMatch(t, /4,096/, 'the old text is not left on screen');
  await page.context().close();
});
test('Articles A and B, whose scenes carry no detail yet, show no hint and take no tab stop', async () => {
  for (const rel of ['why-loop.html', 'inside-inkling.html']) {
    const page = await open(env, rel + '?test=1');
    assert.ok(await page.locator('#inspector').isHidden(), rel + ': no inspector hint');
    assert.equal(await page.evaluate(() => document.getElementById('scene').hasAttribute('tabindex')), false, rel + ': canvas is not a tab stop');
    await page.context().close();
  }
});
test('the pinned text is not rewritten when a redraw leaves it unchanged', async () => {
  const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1');
  await go(page, 'walk-scores');
  const c = await centre(page, 'tensor.raw.r4c2'); await page.mouse.click(c.x, c.y);
  const n = await page.evaluate(() => new Promise(res => { let k = 0; const o = new MutationObserver(m => { k += m.length; }); o.observe(document.getElementById('inspector'), { childList: true, subtree: true, characterData: true }); for (let i = 0; i < 5; i++) window.LOOP.sceneApi.redraw(); setTimeout(() => res(k), 100); }));
  assert.equal(n, 0, 'mutations: ' + n);
  await page.context().close();
});
