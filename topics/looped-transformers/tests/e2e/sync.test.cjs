/* Text and scene must always show the same beat, in every navigation mode. Runs on whichever
   article page is named in PAGES (default: every article that exists). */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { launch, open, ROOT } = require('./helpers.cjs');

const PAGES = (process.env.PAGES || 'why-loop.html,inside-inkling.html,transformer-walkthrough.html').split(',').filter(p => fs.existsSync(path.join(ROOT, p)));
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });

const q = '?test=1';
const settle = page => page.waitForTimeout(120);
/* Which section is under the anchor line right now, by looking at the page, not at the store. */
const domBeat = page => page.evaluate(() => {
  const a = window.LOOP.drivers.anchorY(); let b = 0;
  window.LOOP.sections.forEach((s, i) => { if (s.getBoundingClientRect().top <= a) b = i; });
  return b;
});
const scrollMid = (page, i, frac = 0.5) => page.evaluate(([i, frac]) => {
  const s = window.LOOP.sections[i], r = s.getBoundingClientRect();
  window.scrollTo(0, window.scrollY + r.top + frac * r.height - window.LOOP.drivers.anchorY());
}, [i, frac]);

for (const rel of PAGES) {
  test(`${rel}: scrolling to each beat puts the scene in that beat's state`, async () => {
    const page = await open(env, rel + q);
    const n = await page.evaluate(() => window.LOOP.beatCount);
    for (let i = 0; i < n; i++) {
      await scrollMid(page, i);
      await page.waitForFunction(i => window.LOOP.state().beat === i, i, { timeout: 3000 });
      const r = await page.evaluate(i => {
        const b = window.LOOP.beats[i], s = window.LOOP.scene();
        return { id: s.beatId, want: b.id, lit: s.lit, specLit: b.scene.lit };
      }, i);
      assert.equal(r.id, r.want);
      for (const id of r.specLit) assert.ok(r.lit.includes(id), `${r.want}: ${id} not lit (${r.lit})`);
    }
    assert.deepEqual(page.errors, []);
    await page.context().close();
  });

  test(`${rel}: scrolling back rewinds the scene exactly`, async () => {
    const page = await open(env, rel + q);
    await scrollMid(page, 1, 0.6); await settle(page);
    const y = await page.evaluate(() => window.scrollY), before = await page.evaluate(() => JSON.stringify(window.LOOP.scene()));
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight)); await settle(page);
    assert.notEqual(await page.evaluate(() => JSON.stringify(window.LOOP.scene())), before);
    await page.evaluate(y => window.scrollTo(0, y), y); await settle(page);
    assert.equal(await page.evaluate(() => JSON.stringify(window.LOOP.scene())), before);
    await page.context().close();
  });

  test(`${rel}: Step keeps text and scene on the same beat, and Left goes back`, async () => {
    const page = await open(env, rel + q);
    await page.click('#mode-step');
    const n = Math.min(6, (await page.evaluate(() => window.LOOP.beatCount)) - 1);
    for (let k = 1; k <= n; k++) {
      await page.click('#next');
      await page.waitForFunction(() => window.LOOP.state().progress === 1, null, { timeout: 4000 });
      assert.equal(await page.evaluate(() => window.LOOP.state().beat), k);
      assert.equal(await domBeat(page), k);
      assert.equal(await page.evaluate(() => window.LOOP.scene().beatId), await page.evaluate(k => window.LOOP.beats[k].id, k));
    }
    await page.keyboard.press('ArrowLeft');
    await page.waitForFunction(() => window.LOOP.state().progress === 1, null, { timeout: 4000 });
    assert.equal(await page.evaluate(() => window.LOOP.state().beat), n - 1);
    assert.equal(await domBeat(page), n - 1);
    assert.equal(await page.evaluate(() => window.LOOP.state().mode), 'step');
    await page.context().close();
  });

  test(`${rel}: Play advances on its own and the text never lags the scene`, async () => {
    const page = await open(env, rel + q);
    await page.selectOption('#speed', '4');
    await page.click('#mode-play');
    let bad = 0, maxBeat = 0;
    for (let k = 0; k < 40; k++) {
      await page.waitForTimeout(100);
      const [d, s] = [await domBeat(page), await page.evaluate(() => window.LOOP.state().beat)];
      maxBeat = Math.max(maxBeat, s);
      bad = d === s ? 0 : bad + 1;
      assert.ok(bad < 2, `text beat ${d} vs scene beat ${s}`);
    }
    assert.ok(maxBeat >= 1, 'Play never moved on');
    await page.context().close();
  });

  test(`${rel}: a manual scroll during Play stops it and hands control back`, async () => {
    const page = await open(env, rel + q);
    await page.click('#mode-play');
    await page.waitForTimeout(150);
    await page.mouse.move(300, 400); await page.mouse.wheel(0, 40);
    await page.waitForFunction(() => window.LOOP.state().mode === 'scroll', null, { timeout: 2000 });
    const a = await page.evaluate(() => window.LOOP.state().beat); await page.waitForTimeout(1000);
    assert.equal(await page.evaluate(() => window.LOOP.state().beat), a);
    assert.equal(await page.evaluate(() => document.getElementById('mode-scroll').checked), true);
    await page.context().close();
  });

  test(`${rel}: a deep link opens at that beat in Scroll mode, even with mode=play in the URL`, async () => {
    const probe = await open(env, rel + q);
    const ids = await probe.evaluate(() => window.LOOP.beats.map(b => b.id)); await probe.context().close();
    const id = ids[ids.length - 1];
    for (const extra of ['', '&mode=play']) {
      const page = await open(env, rel + q + extra + '#b-' + id);
      await page.waitForFunction(id => window.LOOP.scene().beatId === id, id, { timeout: 3000 });
      const st = await page.evaluate(() => window.LOOP.state());
      assert.equal(st.mode, 'scroll'); assert.equal(st.beat, ids.length - 1);
      assert.equal(await domBeat(page), ids.length - 1);
      assert.ok(st.progress > 0.35, 'pose has not arrived: ' + st.progress);
      await page.context().close();
    }
  });

  test(`${rel}: scrolling by hand while in Step mode returns to Scroll mode at the scroll position`, async () => {
    const page = await open(env, rel + q);
    await page.click('#mode-step'); await page.click('#next');
    await page.waitForFunction(() => window.LOOP.state().progress === 1, null, { timeout: 4000 });
    await page.evaluate(() => window.scrollTo(0, 0)); await settle(page);
    const st = await page.evaluate(() => window.LOOP.state());
    assert.equal(st.mode, 'scroll'); assert.equal(st.beat, 0);
    await page.context().close();
  });

  test(`${rel}: reduced motion cuts instead of tweening`, async () => {
    const page = await open(env, rel + q, { reducedMotion: 'reduce' });
    await page.click('#mode-step'); await page.click('#next');
    await page.waitForFunction(() => window.LOOP.state().progress === 1, null, { timeout: 60 });
    assert.equal(await page.evaluate(() => window.LOOP.state().beat), 1);
    await page.context().close();
  });

  test(`${rel}: progress stays inside 0-1 through resize and rotation`, async () => {
    const page = await open(env, rel + q);
    for (const [w, h] of [[390, 844], [844, 390], [1400, 900], [600, 300]]) {
      await page.setViewportSize({ width: w, height: h });
      for (const frac of [0, 0.5, 1]) {
        await scrollMid(page, 0, frac); await settle(page);
        const p = await page.evaluate(() => window.LOOP.state().progress);
        assert.ok(Number.isFinite(p) && p >= 0 && p <= 1);
      }
      assert.ok(await page.evaluate(() => window.LOOP.scene().bounds().maxX <= 1.0), `bounds at ${w}x${h}`);
    }
    await page.context().close();
  });
}
