const path = require('node:path');
const { chromium } = require('playwright-core');
const { serve } = require('../serve.cjs');
const ROOT = path.resolve(__dirname, '../..');       // topics/looped-transformers
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function launch() {
  const base = process.env.BASE ? { url: process.env.BASE, close: async () => {} } : await serve(ROOT);
  const browser = await chromium.launch({ executablePath: CHROME, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  return { base, browser, async close() { await browser.close(); await base.close(); } };
}
/* A fresh page that records console errors and failed requests. */
async function open(env, rel, { width = 1400, height = 900, scheme = 'light', reducedMotion = 'no-preference' } = {}) {
  const ctx = await env.browser.newContext({ viewport: { width, height }, colorScheme: scheme, reducedMotion });
  const page = await ctx.newPage();
  page.errors = [];
  page.on('console', m => { if (m.type() === 'error') page.errors.push(m.text()); });
  page.on('pageerror', e => page.errors.push(String(e)));
  page.on('requestfailed', r => page.errors.push('request failed: ' + r.url()));
  await page.goto(env.base.url + '/' + rel);
  await page.waitForFunction(() => window.LOOP && window.LOOP.ready, null, { timeout: 15000 });
  // Let fonts and KaTeX finish: they change section heights, and tests compare scroll positions.
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => typeof window.renderMathInElement !== 'function' || !document.querySelector('.math') || document.querySelector('.math .katex'), null, { timeout: 15000 });
  await page.waitForTimeout(60);
  return page;
}
module.exports = { launch, open, ROOT };
