const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
const fake = require('./fake-speech.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });
const PAGE = 'transformer-walkthrough.html?test=1';

test('the walkthrough loads its recorded data, and every beat draws and fits at four window sizes', async () => {
  for (const [w, h] of [[1400, 900], [1000, 900], [390, 844], [844, 390]]) {
    const page = await open(env, PAGE, { width: w, height: h });
    await page.waitForFunction(() => window.LOOP.walkReady);
    const n = await page.evaluate(() => window.LOOP.beatCount);
    for (let i = 0; i < n; i++) {
      const r = await page.evaluate(i => { window.LOOP.store.set({ beat: i, progress: 1 }, 'test'); const s = window.LOOP.scene(); return { b: s.bounds(), cov: s.coverage(), id: s.beatId }; }, i);
      assert.ok(r.cov > 0.005, `${r.id} coverage ${r.cov} at ${w}x${h}`);
      assert.ok(r.b.maxX <= 1.0 && r.b.maxY <= 1.0, `${r.id} at ${w}x${h}: ${JSON.stringify(r.b)}`);
    }
    assert.deepEqual(page.errors, []);
    await page.context().close();
  }
});
test('recorded numbers are on the page, the equations are rendered, and lit labels show recorded values', async () => {
  const page = await open(env, PAGE);
  await page.waitForFunction(() => window.LOOP.walkReady);
  assert.ok((await page.$$eval('[data-walk]', e => e.length)) > 20);
  assert.match(await page.textContent('#b-head'), /34\.6%/);
  const i = await page.evaluate(() => window.LOOP.beats.findIndex(b => b.id === 'rmsnorm'));
  await page.evaluate(i => window.LOOP.store.set({ beat: i, progress: 1 }, 'test'), i); await page.waitForTimeout(150);
  const labels = await page.$$eval('#labels .lbl:not([hidden])', els => els.map(e => e.textContent));
  assert.ok(labels.some(t => /mean of squares = 0\.0005497/.test(t)), JSON.stringify(labels));
  await page.context().close();
});
test('Narrate reads the head beat with the equation in words and the probability, not TeX', async () => {
  const page = await open(env, PAGE, { init: fake });
  await page.waitForFunction(() => window.LOOP.walkReady);
  await page.evaluate(() => window.LOOP.drivers.goTo('stack')); await page.click('#mode-step'); await page.check('#narrate');
  await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length > 8, null, { timeout: 20000 });
  const said = await page.evaluate(() => window.__speech.spoken.map(s => s.text));
  assert.equal(said[0], 'The last vector becomes 151,936 scores');
  assert.ok(said.join(' ').includes('34.6%'));
  assert.ok(said.every(t => !/[\\${}]/.test(t)), 'no TeX reaches the voice');
  await page.context().close();
});
