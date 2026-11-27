/* The "how it changed" switches on the real article: the scene redraws, the table follows, and every figure is the computed one. */
const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

const PAGE = 'transformer-walkthrough.html?test=1';
async function ready(page, beat) {
  await page.waitForFunction(() => window.LOOP.walkReady);
  await page.evaluate(b => { const i = window.LOOP.beats.findIndex(x => x.id === b); window.LOOP.store.set({ beat: i, progress: 1 }, 'test'); }, beat);
  await page.waitForTimeout(100);
}
const pick = (page, family, v) => page.evaluate(([f, x]) => document.querySelector(`[data-variants="${f}"] [data-variant="${x}"]`).click(), [family, v]);
const labels = page => page.evaluate(() => [...document.querySelectorAll('#labels .lbl')].filter(e => !e.hidden).map(e => e.textContent).join(' | '));
const FAMILY = { attention: 'evo-attention', position: 'evo-position', norm: 'evo-norm', ffn: 'evo-ffn', embedding: 'evo-embed' };

test('each table cell is the number the design model computes, and the table says it is computed', async () => {
  const page = await open(env, PAGE);
  await page.waitForFunction(() => window.LOOP.walkReady);
  const bad = await page.evaluate(fams => {
    const out = [];
    for (const fam of fams) {
      const t = LoopVariants.table(fam, LoopVariants.QWEN3), root = document.querySelector(`[data-variant-table="${fam}"]`);
      if (!/computed from the config/i.test(root.querySelector('caption').textContent + root.textContent)) out.push(fam + ': no computed label');
      for (const row of t.rows) {
        const tr = root.querySelector(`tr[data-kind="${row.kind}"]`); if (!tr) { out.push(`${fam}/${row.kind}: row missing`); continue; }
        t.columns.forEach((c, i) => { const got = tr.cells[i + 1].textContent, want = LoopVariantsUI.fmt(c, row.cells[c]); if (got !== want) out.push(`${fam}/${row.kind}/${c}: ${got} vs ${want}`); });
      }
    }
    return out;
  }, Object.keys(FAMILY));
  assert.deepEqual(bad, []); assert.deepEqual(page.errors, []);
  await page.context().close();
});
test('switching attention design redraws the scene and highlights the row', async () => {
  const page = await open(env, PAGE);
  await ready(page, 'evo-attention');
  assert.match(await labels(page), /4,096 B per token per layer/);
  for (const [v, b, n] of [['mha', '8,192', 16], ['mqa', '512', 1], ['mla', '1,152', 1], ['gqa', '4,096', 8]]) {
    await pick(page, 'attention', v);
    assert.match(await labels(page), new RegExp(`${b} B per token per layer`), v);
    assert.equal(await page.evaluate(() => LOOP.sceneApi.infoList().filter(r => /^kvh\./.test(r.id)).length), n, v + ' key/value cells');
    assert.equal(await page.getAttribute(`[data-variants="attention"] [data-variant="${v}"]`, 'aria-pressed'), 'true');
    assert.equal(await page.getAttribute(`[data-variant-table="attention"] tr[data-kind="${v}"]`, 'aria-current'), 'true');
    assert.equal(await page.locator('[data-variants="attention"] [aria-pressed="true"]').count(), 1);
  }
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
test('every family and every design draws without error, and a chosen design survives the recording loading', async () => {
  const page = await open(env, PAGE);
  for (const [fam, beat] of Object.entries(FAMILY)) {
    await ready(page, beat);
    const variants = await page.evaluate(f => [...document.querySelectorAll(`[data-variants="${f}"] [data-variant]`)].map(b => b.dataset.variant), fam);
    for (const v of variants) {
      await pick(page, fam, v);
      const cov = await page.evaluate(() => { const s = LOOP.scene(); return { c: s.coverage(), b: s.bounds() }; });
      assert.ok(cov.c > 0.01 && cov.b.maxX <= 1 && cov.b.maxY <= 1, `${fam}/${v}: ${JSON.stringify(cov)}`);
      assert.match(await labels(page), /computed from the config/i, `${fam}/${v}`);
    }
  }
  await ready(page, 'evo-ffn'); await pick(page, 'ffn', 'moe');
  await page.evaluate(() => { const w = LOOP.sceneApi.getData('walk'); LOOP.sceneApi.setData('walk', { ...w }); });
  assert.match(await labels(page), /Mixture of experts/);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
test('the switches work from the keyboard and never overflow a phone', async () => {
  const page = await open(env, PAGE, { width: 390, height: 844 });
  await ready(page, 'evo-position');
  await page.locator('[data-variants="position"] [data-variant="alibi"]').focus(); await page.keyboard.press('Enter');
  assert.match(await labels(page), /distance/i);
  const w = await page.evaluate(() => ({ s: document.documentElement.scrollWidth, i: window.innerWidth }));
  assert.ok(w.s <= w.i, JSON.stringify(w));
  await page.context().close();
});
test('hover and click work on the variant scenes too', async () => {
  const page = await open(env, PAGE);
  await ready(page, 'evo-attention'); await pick(page, 'attention', 'gqa');
  const c = await page.evaluate(() => { const p = LOOP.sceneApi.screenOf('qh.5'), r = LOOP.sceneApi.canvas().getBoundingClientRect(); return { x: r.left + p[0], y: r.top + p[1] }; });
  await page.mouse.click(c.x, c.y);
  const t = await page.locator('#inspector').innerText();
  assert.match(t, /Query head 5/); assert.match(t, /key\/value head 2/);
  await page.context().close();
});
