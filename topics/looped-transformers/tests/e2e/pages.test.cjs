/* Run after `python3 scripts/build-site.py`: serves _site under a repository-style prefix and
   drives the learner's path through it: landing -> Article A -> Play, landing -> Article B -> a deep link. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const { chromium } = require('playwright-core');
const { serve } = require('../serve.cjs');
const SITE = path.resolve(__dirname, '../../../../_site'), PREFIX = '/learning_playground';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const skip = !fs.existsSync(SITE) && '_site not built (run scripts/build-site.py)';

let server, browser;
test.before(async () => { if (skip) return; server = await serve(SITE, PREFIX); browser = await chromium.launch({ executablePath: CHROME, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] }); });
test.after(async () => { if (browser) await browser.close(); if (server) await server.close(); });

const fresh = async () => { const page = await (await browser.newContext({ viewport: { width: 1400, height: 900 } })).newPage(); page.errors = []; page.on('pageerror', e => page.errors.push(String(e))); page.on('requestfailed', r => page.errors.push('failed ' + r.url())); page.on('response', r => { if (r.status() >= 400) page.errors.push(r.status() + ' ' + r.url()); }); return page; };

test('every local link on the three pages resolves under the repository prefix', { skip }, async () => {
  const page = await fresh(); const bad = [];
  for (const p of ['index.html', 'why-loop.html', 'inside-inkling.html']) {
    await page.goto(`${server.url}/topics/looped-transformers/${p}`);
    const hrefs = await page.$$eval('a[href]', as => as.map(a => a.href));
    for (const h of [...new Set(hrefs)].filter(h => h.startsWith(server.url))) {
      const res = await page.request.get(h.split('#')[0]);
      if (res.status() !== 200) bad.push(`${p}: ${h} -> ${res.status()}`);
    }
  }
  assert.deepEqual(bad, []);
  await page.context().close();
});
test('landing -> Article A -> Play moves the beat, with no failed requests', { skip }, async () => {
  const page = await fresh();
  await page.goto(`${server.url}/topics/looped-transformers/index.html`);
  await page.click('a.card[href="why-loop.html"]');
  await page.waitForFunction(() => window.LOOP && window.LOOP.ready && window.LOOP.toy && window.LOOP.toy.ready);
  await page.selectOption('#speed', '4');
  await page.click('#mode-play');
  await page.waitForFunction(() => window.LOOP.state().beat >= 1, null, { timeout: 30000 });
  assert.equal(await page.evaluate(() => window.LOOP.state().mode), 'play');
  assert.deepEqual(page.errors.filter(e => !/favicon/.test(e)), []);
  await page.context().close();
});
test('Article B deep link opens the recorded run with its tables', { skip }, async () => {
  const page = await fresh();
  await page.goto(`${server.url}/topics/looped-transformers/inside-inkling.html#b-loop-recorded`);
  await page.waitForFunction(() => window.LOOP && window.LOOP.scene().beatId === 'loop-recorded');
  await page.waitForSelector('[data-recorded="gate-ratio"] table.heat');
  assert.equal(await page.$$eval('[data-recorded] table.heat', t => t.length), 4);
  assert.deepEqual(page.errors.filter(e => !/favicon/.test(e)), []);
  await page.context().close();
});

test('landing -> Article C loads its recorded data under the repository prefix', { skip }, async () => {
  const page = await fresh();
  await page.goto(`${server.url}/topics/looped-transformers/index.html`);
  await page.click('a.card[href="transformer-walkthrough.html"]');
  await page.waitForFunction(() => window.LOOP && window.LOOP.walkReady, null, { timeout: 30000 });
  assert.ok((await page.$$eval('[data-walk]', e => e.length)) > 20);
  assert.deepEqual(page.errors.filter(e => !/favicon/.test(e)), []);
  await page.context().close();
});
