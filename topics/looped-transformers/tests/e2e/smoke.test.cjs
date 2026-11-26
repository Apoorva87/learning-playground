const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');

let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

for (const scheme of ['light', 'dark']) {
  for (const renderer of ['', '?renderer=2d']) {
    test(`scene draws something (${scheme}${renderer ? ', 2d fallback' : ', webgl'})`, async () => {
      const page = await open(env, 'why-loop.html' + renderer, { scheme });
      const info = await page.evaluate(() => ({ kind: window.LOOP.scene().renderer, cov: window.LOOP.scene().coverage() }));
      assert.equal(info.kind, renderer ? '2d' : 'gl');
      assert.ok(info.cov > 0.02, 'coverage ' + info.cov);
      assert.deepEqual(page.errors, []);
      await page.context().close();
    });
  }
}
test('losing the WebGL context switches to the 2D renderer and keeps drawing', async () => {
  const page = await open(env, 'why-loop.html');
  const info = await page.evaluate(async () => {
    const c = document.getElementById('scene');
    c.dispatchEvent(new Event('webglcontextlost', { cancelable: true }));
    await new Promise(r => setTimeout(r, 50));
    return { kind: window.LOOP.scene().renderer, cov: window.LOOP.scene().coverage() };
  });
  assert.equal(info.kind, '2d'); assert.ok(info.cov > 0.02);
  await page.context().close();
});
test('phone width has no horizontal scroll', async () => {
  const page = await open(env, 'why-loop.html', { width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= 390));
  await page.context().close();
});
test('every beat keeps all of its objects inside the canvas at desktop and phone sizes', async () => {
  for (const [w, h] of [[1400, 900], [390, 844], [900, 600]]) {
    const page = await open(env, 'why-loop.html', { width: w, height: h });
    const n = await page.evaluate(() => window.LOOP.beatCount);
    for (let i = 0; i < n; i++) {
      const b = await page.evaluate(i => { window.LOOP.store.set({ beat: i, progress: 1 }, 'test'); return window.LOOP.scene().bounds(); }, i);
      assert.ok(b.maxX <= 1.0 && b.maxY <= 1.0, `beat ${i} at ${w}x${h}: ${JSON.stringify(b)}`);
    }
    await page.context().close();
  }
});
test('the flow marker moves when progress moves', async () => {
  const page = await open(env, 'why-loop.html');
  const xs = await page.evaluate(() => [0.5, 1].map(p => { window.LOOP.store.set({ beat: 0, progress: p }, 'test'); return window.LOOP.scene().marker(); }));
  assert.notDeepEqual(xs[0], xs[1]);
  await page.context().close();
});
