const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const Sp = require('../../engine/speech.js');

const say = Sp.texToSpeech;
test('texToSpeech: symbols and shapes become words', () => {
  assert.equal(say('T'), 'T');
  assert.match(say('h \\in \\mathbb{R}^{[T,\\,2304]}'), /^h in R to the shape T by 2304$/);
  assert.equal(say('F_\\ell'), 'F sub ell');
  assert.equal(say('p_{t-1}'), 'p sub t minus 1');
  assert.equal(say('6 + 3 \\times 12 + 6 = 48'), '6 plus 3 times 12 plus 6 equals 48');
  assert.equal(say('\\frac{a}{b}'), 'a over b');
  assert.equal(say('\\sqrt{2304}'), 'root 2304');
  assert.equal(say('x^2'), 'x to the 2');
  assert.match(say('W_{\\text{down}}'), /^W sub down$/);
});
test('texToSpeech: thousands separators and the transpose sign', () => {
  assert.equal(say('155{,}582{,}464'), '155582464');
  assert.match(say('z E^{\\top}'), /^z E transpose$/);
  assert.equal(say('E^\\top'), 'E transpose');
});
test('texToSpeech: never leaves backslashes, braces or dollar signs', () => {
  for (const t of ['\\operatorname{softmax}_i\\!\\big(\\mathrm{src}_i \\cdot p_{t-1}\\big)', '\\Big(\\tfrac{1}{30}\\,W\\Big)', '\\text{for } t = 2, 3', '\\rho(A) &lt; 1', '\\mathcal{E} = \\operatorname{top\\text{-}8}(s)'])
    assert.doesNotMatch(say(t), /[\\{}$]/, say(t));
});
test('splitSentences: splits on sentence ends but not on decimals or abbreviations', () => {
  const text = 'It shares 4.78 of 8 experts, e.g. in layer 6. The next one is 0.7. Done!';
  const parts = Sp.splitSentences(text).map(r => text.slice(r.start, r.end).trim());
  assert.deepEqual(parts, ['It shares 4.78 of 8 experts, e.g. in layer 6.', 'The next one is 0.7.', 'Done!']);
});
test('splitSentences: long sentences are cut at commas, every piece under the limit, nothing lost', () => {
  const long = Array.from({ length: 12 }, (_, i) => `clause number ${i} goes on and on`).join(', ') + '.';
  const parts = Sp.splitSentences(long, 120);
  assert.ok(parts.length > 2 && parts.every(r => r.end - r.start <= 140));
  assert.equal(parts.map(r => long.slice(r.start, r.end)).join(''), long);
});
test('splitSentences: offsets cover the text contiguously', () => {
  const t = 'One. Two? Three!  Four';
  const p = Sp.splitSentences(t); assert.equal(p[0].start, 0); assert.equal(p[p.length - 1].end, t.length);
  for (let i = 1; i < p.length; i++) assert.equal(p[i].start, p[i - 1].end);
});

/* Authoring completeness: every display equation in both articles has a spoken version. */
for (const page of ['why-loop.html', 'inside-inkling.html', 'transformer-walkthrough.html']) {
  test(`${page}: every display equation is wrapped with a spoken version`, () => {
    const html = fs.readFileSync(path.join(__dirname, '../..', page), 'utf8');
    const display = [...html.matchAll(/\$\$[\s\S]*?\$\$/g)].length;
    const wrapped = [...html.matchAll(/<span class="eq" data-speak="([^"]+)">\$\$[\s\S]*?\$\$<\/span>/g)];
    assert.equal(wrapped.length, display);
    for (const [, spoken] of wrapped) { assert.ok(spoken.length > 25 && !/[\\{}$]/.test(spoken), spoken); }
  });
}
