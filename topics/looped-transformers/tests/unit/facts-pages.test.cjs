/* Every fact cited by the three articles must still match its source. Config and source-line facts need the
   local checkpoint and Transformers install, so those are skipped with a warning when they are absent. */
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path'), fs = require('node:fs');
const { checkFacts } = require('../check-facts.cjs');
const ROOT = path.join(__dirname, '../..');
const pages = ['why-loop.html', 'inside-inkling.html', 'transformer-walkthrough.html'].map(p => path.join(ROOT, p));
const SITE = path.join(ROOT, '../../.local/inkling2/venv/lib/python3.13/site-packages/transformers');

test('no cited or recorded fact is out of date', () => {
  const r = checkFacts({ htmlFiles: pages, transformers: fs.existsSync(SITE) ? SITE : null });
  assert.deepEqual(r.errors, []);
});
