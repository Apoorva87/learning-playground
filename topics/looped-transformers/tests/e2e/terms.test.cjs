const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

test('hovering a term highlights its scene objects, and leaving clears them', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  const term = page.locator('.term[data-term="core"]').first();
  await term.scrollIntoViewIfNeeded(); await term.hover();
  const r = await page.evaluate(() => {
    const t = document.querySelector('.term[data-term="core"]'), sec = t.closest('section.beat');
    const b = window.LOOP.beats.find(b => b.id === sec.dataset.beat);
    const want = [...new Set(b.terms.core.flatMap(window.LoopLayout.idsFor))].sort();
    return { got: [...window.LOOP.scene().highlight].sort(), want };
  });
  assert.deepEqual(r.got, r.want); assert.ok(r.want.length > 0);
  await page.mouse.move(5, 5);
  assert.deepEqual(await page.evaluate(() => window.LOOP.scene().highlight), []);
  await page.context().close();
});
test('keyboard focus on a term highlights it too', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  await page.locator('.term[data-term="core"]').first().focus();
  assert.ok((await page.evaluate(() => window.LOOP.scene().highlight.length)) > 0);
  await page.context().close();
});
test('hovering an object in the scene marks the terms that name it', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  await page.evaluate(() => window.LOOP.store.set({ beat: window.LOOP.beats.findIndex(b => b.terms.loopback), progress: 1 }, 'test'));
  const pt = await page.evaluate(() => { const r = document.getElementById('scene').getBoundingClientRect(), p = window.LOOP.sceneApi.screenOf('V10'); return [r.left + p[0], r.top + p[1]]; });
  await page.mouse.move(pt[0], pt[1]);
  await page.waitForFunction(() => document.querySelector('.term-active'));
  const active = await page.evaluate(() => [...document.querySelectorAll('.term-active')].map(e => e.dataset.term));
  assert.ok(active.includes('core') && !active.includes('loopback'), JSON.stringify(active));   // V10 is a core layer, not the return arrow
  assert.ok(await page.evaluate(() => [...document.querySelectorAll('.term-active')].every(e => e.closest('section.beat').dataset.beat === window.LOOP.scene().beatId)));
  await page.mouse.move(5, 5);
  await page.waitForFunction(() => !document.querySelector('.term-active'));
  await page.context().close();
});
test('equations render with KaTeX when it loads', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  const has = await page.evaluate(() => typeof window.renderMathInElement === 'function');
  if (!has) return;                                      // offline: the TeX source stays readable
  await page.waitForSelector('.math .katex', { timeout: 10000 });
  await page.context().close();
});

for (const rel of ['why-loop.html', 'inside-inkling.html']) {
  test(`${rel}: no equation is left as raw TeX`, async () => {
    const page = await open(env, rel + '?test=1');
    const has = await page.evaluate(() => typeof window.renderMathInElement === 'function');
    if (!has) return;                                      // offline: nothing to render with
    const raw = await page.evaluate(() => [...document.querySelectorAll('#article .math, #article .story, #article .intuition, #article section > p')]
      .map(e => { const c = e.cloneNode(true); c.querySelectorAll('.katex-mathml, .katex-html').forEach(k => k.remove()); return c.textContent; })
      .filter(t => /\$\$|\\\(|\\\)|\\mathbb|\\frac|\\sum|\\operatorname/.test(t)));
    assert.deepEqual(raw.map(t => t.slice(0, 80)), []);
    await page.context().close();
  });
}
