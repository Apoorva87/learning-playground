const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

const page0 = () => open(env, 'why-loop.html?test=1');
const NODES = 'ABCDEFGH';

test('the toy loads and starts at the trained three passes', async () => {
  const page = await page0();
  await page.waitForFunction(() => window.LOOP.toy && window.LOOP.toy.ready, null, { timeout: 8000 });
  const t = await page.evaluate(() => ({ passes: window.LOOP.toy.passes, variant: window.LOOP.toy.variant }));
  assert.deepEqual(t, { passes: 3, variant: 'inkling' });
  await page.context().close();
});
test('changing the loop count changes the displayed hop to the matching one', async () => {
  const page = await page0();
  await page.waitForFunction(() => window.LOOP.toy && window.LOOP.toy.ready);
  for (const k of [1, 2, 4]) {
    await page.click(`[data-toy-passes="${k}"]`);
    const r = await page.evaluate(() => {
      const T = window.LOOP.toy, tk = T.tokens, f = {}; for (let i = 0; i < 8; i++) f[tk[2 * i]] = tk[2 * i + 1];
      return { passes: T.passes, text: document.getElementById('toy-readout').textContent, hops: (() => { let c = tk[16]; return Array.from({ length: 6 }, () => (c = f[c])); })(), argmax: T.argmax };
    });
    assert.equal(r.passes, k);
    assert.match(r.text, new RegExp(`After ${k} pass`));
    assert.ok(r.text.includes(`reads out ${NODES[r.argmax[k - 1]]}`), r.text);
    assert.equal(r.argmax[k - 1], r.hops[k - 1], 'the toy should read out the k-th hop after k passes');
  }
  await page.context().close();
});
test('the printed hop matrix equals toy-accuracy.json for each variant', async () => {
  const page = await page0();
  await page.waitForFunction(() => window.LOOP.toy && window.LOOP.toy.ready);
  for (const variant of ['inkling', 'carry']) {
    await page.selectOption('#toy-variant', variant);
    const r = await page.evaluate(async variant => {
      const acc = await (await fetch('assets/toy/toy-accuracy.json')).json();
      const cells = [...document.querySelectorAll('#toy-table tbody td')].map(td => td.textContent);
      return { want: acc[variant].hop_matrix.flat().map(v => Math.round(v * 100) + '%'), got: cells, extra: acc[variant].extrapolation_passes };
    }, variant);
    assert.deepEqual(r.got, r.want);
  }
  assert.match(await page.textContent('#toy-table'), /never trained/i);
  await page.context().close();
});
test('a new puzzle changes the readout, and the gate vectors exist from the second pass on', async () => {
  const page = await page0();
  await page.waitForFunction(() => window.LOOP.toy && window.LOOP.toy.ready);
  const before = await page.textContent('#toy-readout');
  await page.click('#toy-new');
  assert.notEqual(await page.textContent('#toy-readout'), before);
  await page.click('[data-toy-passes="3"]'); await page.selectOption('#toy-variant', 'inkling');
  const g = await page.evaluate(() => window.LOOP.toy.gate);
  assert.equal(g.f.length, 16); assert.equal(g.out.length, 16);
  await page.selectOption('#toy-variant', 'carry');
  assert.equal(await page.evaluate(() => window.LOOP.toy.gate), null);
  await page.context().close();
});
