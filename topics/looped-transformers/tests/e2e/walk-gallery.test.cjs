const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

for (const renderer of ['', '&renderer=2d']) {
  test(`every walkthrough scene draws its recorded data and fits (${renderer ? '2d' : 'webgl'})`, async () => {
    for (const [w, h] of [[1400, 900], [390, 844]]) {
      const page = await open(env, 'tests/fixtures/walk-gallery.html?test=1' + renderer, { width: w, height: h });
      await page.waitForFunction(() => window.LOOP.walkReady);
      const n = await page.evaluate(() => window.LOOP.beatCount);
      for (let i = 0; i < n; i++) {
        const r = await page.evaluate(i => { window.LOOP.store.set({ beat: i, progress: 1 }, 'test'); const s = window.LOOP.scene(); return { b: s.bounds(), cov: s.coverage(), id: s.beatId }; }, i);
        assert.ok(r.cov > 0.01, `${r.id} coverage ${r.cov} at ${w}x${h}`);
        assert.ok(r.b.maxX <= 1.0 && r.b.maxY <= 1.0, `${r.id} at ${w}x${h}: ${JSON.stringify(r.b)}`);
      }
      assert.deepEqual(page.errors, []);
      await page.context().close();
    }
  });
}
