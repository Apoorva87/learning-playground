const test = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const check = f => spawnSync('node', [path.join(__dirname, '../check-article.cjs'), path.join(__dirname, '../fixtures', f)], { encoding: 'utf8' });

test('a beat with story, intuition and a shaped equation passes', () => assert.equal(check('good.html').status, 0, check('good.html').stdout));
test('a math beat without an intuition block fails and names it', () => {
  const r = check('missing-intuition.html'); assert.equal(r.status, 1); assert.match(r.stdout, /one.*intuition/i);
});
test('a math block with no tensor shape fails', () => {
  const r = check('no-shape.html'); assert.equal(r.status, 1); assert.match(r.stdout, /one.*shape/i);
});
test('a term that the beat does not map to scene objects fails', () => {
  const r = check('unknown-term.html'); assert.equal(r.status, 1); assert.match(r.stdout, /mystery/);
});
test('interludes are exempt from the three-part structure', () => assert.doesNotMatch(check('good.html').stdout, /two/));
