/* Regressions found in the whole-branch review: touch, keyboard and click handling, a late load
   event, narrow desktop stages, and terms after a lost WebGL context. */
const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });
const mode = page => page.evaluate(() => window.LOOP.state().mode);

test('tapping Next on a touch screen keeps Step mode', async () => {
  const page = await open(env, 'why-loop.html?test=1', { width: 390, height: 844, hasTouch: true });
  await page.tap('#mode-step'); await page.tap('#next');
  await page.waitForFunction(() => window.LOOP.state().beat === 1);
  assert.equal(await mode(page), 'step');
  await page.context().close();
});
test('tapping Speed during Play on a touch screen keeps Play', async () => {
  const page = await open(env, 'why-loop.html?test=1', { width: 390, height: 844, hasTouch: true });
  await page.tap('#mode-play'); await page.tap('#speed');
  assert.equal(await mode(page), 'play');
  await page.context().close();
});
test('Space on the focused Next button keeps Step mode', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  await page.click('#mode-step'); await page.focus('#next'); await page.keyboard.press('Space');
  await page.waitForFunction(() => window.LOOP.state().beat === 1);
  assert.equal(await mode(page), 'step');
  await page.context().close();
});
test('a click in the article during Play stops it and hands control back', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  await page.click('#mode-play'); await page.waitForTimeout(100);
  await page.click('section.beat p');
  assert.equal(await mode(page), 'scroll');
  const a = await page.evaluate(() => window.LOOP.state().beat); await page.waitForTimeout(800);
  assert.equal(await page.evaluate(() => window.LOOP.state().beat), a);
  await page.context().close();
});
test('clicking the Play controls themselves does not stop Play', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  await page.click('#mode-play'); await page.selectOption('#speed', '2');
  assert.equal(await mode(page), 'play');
  await page.context().close();
});
test('a slow load event does not drag a reader who has scrolled on back to the linked beat', async () => {
  const page = await open(env, 'why-loop.html?test=1#b-costs-kv', {
    route: p => p.route('**/katex.min.js', async r => { await new Promise(res => setTimeout(res, 2500)); await r.continue(); }) });
  await page.waitForFunction(() => window.LOOP.scene().beatId === 'costs-kv');
  await page.evaluate(() => { const L = window.LOOP, i = L.beats.findIndex(b => b.id === 'benefits'), r = L.sections[i].getBoundingClientRect(); window.scrollTo(0, scrollY + r.top + 0.5 * r.height - L.drivers.anchorY()); });
  await page.waitForFunction(() => window.LOOP.state().beat === window.LOOP.beats.findIndex(b => b.id === 'benefits'));
  await page.waitForLoadState('load'); await page.waitForTimeout(300);
  assert.equal(await page.evaluate(() => window.LOOP.scene().beatId), 'benefits');
  await page.context().close();
});
test('a deep link to a middle beat opens there', async () => {
  const probe = await open(env, 'inside-inkling.html?test=1'); const ids = await probe.evaluate(() => window.LOOP.beats.map(b => b.id)); await probe.context().close();
  const id = ids[Math.floor(ids.length / 2)];
  const page = await open(env, `inside-inkling.html?test=1#b-${id}`);
  await page.waitForFunction(id => window.LOOP.scene().beatId === id, id);
  assert.equal(await mode(page), 'scroll');
  await page.context().close();
});
for (const rel of ['why-loop.html', 'inside-inkling.html']) {
  test(`${rel}: every beat fits the stage on narrow desktop windows and short phones`, async () => {
    for (const [w, h] of [[900, 1000], [1000, 900], [1100, 800], [844, 390]]) {
      const page = await open(env, rel + '?test=1', { width: w, height: h });
      const n = await page.evaluate(() => window.LOOP.beatCount);
      for (let i = 0; i < n; i++) {
        const b = await page.evaluate(i => { window.LOOP.store.set({ beat: i, progress: 1 }, 'test'); return window.LOOP.scene().bounds(); }, i);
        assert.ok(b.maxX <= 1.0 && b.maxY <= 1.0, `${rel} beat ${i} at ${w}x${h}: ${JSON.stringify(b)}`);
      }
      await page.context().close();
    }
  });
}
test('terms still light up scene objects after the WebGL context is lost', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  await page.evaluate(() => { document.getElementById('scene').dispatchEvent(new Event('webglcontextlost', { cancelable: true })); window.LOOP.store.set({ beat: window.LOOP.beats.findIndex(b => b.terms.loopback), progress: 1 }, 'test'); });
  assert.equal(await page.evaluate(() => window.LOOP.scene().renderer), '2d');
  const pt = await page.evaluate(() => { const c = window.LOOP.sceneApi.canvas().getBoundingClientRect(), p = window.LOOP.sceneApi.screenOf('V10'); return [c.left + p[0], c.top + p[1]]; });
  await page.mouse.move(pt[0], pt[1]);
  await page.waitForFunction(() => document.querySelector('.term-active'));
  await page.context().close();
});
