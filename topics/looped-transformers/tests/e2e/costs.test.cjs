const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

test('scrolling through the cost beats unrolls the stack: fold rises from 0 to 1 and never falls', async () => {
  const page = await open(env, 'why-loop.html?test=1');
  const folds = await page.evaluate(async () => {
    const L = window.LOOP, first = L.beats.findIndex(b => b.id === 'costs-stored'), last = L.beats.findIndex(b => b.id === 'costs-latency');
    const y0 = L.sections[first].getBoundingClientRect().top + scrollY - L.drivers.anchorY();
    const y1 = L.sections[last].getBoundingClientRect().bottom + scrollY - L.drivers.anchorY();
    const out = [];
    for (let i = 0; i <= 120; i++) {
      window.scrollTo(0, y0 + (y1 - y0) * i / 120);
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      out.push(+L.scene().fold.toFixed(6));
    }
    return out;
  });
  assert.ok(folds[0] < 0.2, 'starts folded: ' + folds[0]);
  assert.equal(folds[folds.length - 1], 1);
  for (let i = 1; i < folds.length; i++) assert.ok(folds[i] >= folds[i - 1] - 1e-9, `fold fell at sample ${i}: ${folds[i - 1]} -> ${folds[i]}`);
  assert.ok(folds.some(f => f > 0.3 && f < 0.7), 'the morph is gradual, not a cut');
  await page.context().close();
});
