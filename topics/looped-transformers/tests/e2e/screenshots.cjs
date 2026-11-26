/* Screenshots of every beat of both articles, light and dark, desktop and phone, into .local/shots/.
   node e2e/screenshots.cjs [outDir]   (not part of the test suite; they are for looking at) */
const path = require('node:path'), fs = require('node:fs');
const { launch, open } = require('./helpers.cjs');
const OUT = path.resolve(process.argv[2] || path.join(__dirname, '../../../../.local/shots/all'));
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const env = await launch(); let n = 0;
  for (const [page_, label] of [['why-loop.html', 'a'], ['inside-inkling.html', 'b']]) {
    for (const [w, h, size] of [[1400, 900, 'desktop'], [390, 844, 'phone']]) for (const scheme of ['light', 'dark']) {
      const page = await open(env, page_ + '?test=1', { width: w, height: h, scheme });
      await page.waitForFunction(() => !document.getElementById('toy') || (window.LOOP.toy && window.LOOP.toy.ready));
      const ids = await page.evaluate(() => window.LOOP.beats.map(b => b.id));
      for (const [i, id] of ids.entries()) {
        await page.evaluate(id => window.LOOP.drivers.goTo(id), id); await page.waitForTimeout(150);
        await page.screenshot({ path: path.join(OUT, `${label}-${size}-${scheme}-${String(i).padStart(2, '0')}-${id}.png`) }); n++;
      }
      await page.context().close();
    }
  }
  console.log(n + ' screenshots in ' + OUT); await env.close();
})();
