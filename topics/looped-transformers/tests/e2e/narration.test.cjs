const test = require('node:test');
const assert = require('node:assert/strict');
const { launch, open } = require('./helpers.cjs');
const fake = require('./fake-speech.cjs');
let env;
test.before(async () => { env = await launch(); });
test.after(async () => { await env.close(); });
const spoken = page => page.evaluate(() => window.__speech.spoken.map(s => s.text));
const domBeat = page => page.evaluate(() => { const a = window.LOOP.drivers.anchorY(); let b = 0; window.LOOP.sections.forEach((s, i) => { if (s.getBoundingClientRect().top <= a) b = i; }); return b; });
const go = (rel = 'why-loop.html', opts = {}) => open(env, rel + '?test=1', { init: fake, ...opts });

test('the voice picker, narrate switch and rate are visible, with English voices first', async () => {
  const page = await go();
  for (const id of ['#narrate', '#voice', '#rate', '#narrate-pause']) assert.ok(await page.isVisible(id), id);
  const opts = await page.$$eval('#voice option', o => o.map(x => x.textContent));
  assert.match(opts[0], /System default/); assert.match(opts[1], /Samantha|Daniel|Zarvox/); assert.match(opts[opts.length - 1], /Thomas/);
  assert.deepEqual(page.errors, []);
  await page.context().close();
});
test('Step with Narrate reads the new beat aloud: title first, equations in words, no TeX', async () => {
  const page = await go();
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length > 6 && window.LOOP.state().progress === 1, null, { timeout: 15000 });
  const said = await spoken(page);
  assert.equal(said[0], 'Depth is a chain');
  assert.ok(said.some(t => /assembly line/i.test(t)));
  assert.ok(said.join(' ').includes('the next state equals the current state plus the layer'), 'the equation is read from its spoken text');
  assert.ok(said.every(t => !/[\\${}]/.test(t)), 'no TeX reaches the voice');
  await page.context().close();
});
test('Play with Narrate waits for the voice, progress follows it, text and scene stay together', async () => {
  const page = await go();
  await page.check('#narrate'); await page.click('#mode-play');
  const seen = []; let advancedWith = null, bad = 0;
  for (let k = 0; k < 120 && advancedWith === null; k++) {
    await page.waitForTimeout(40);
    const s = await page.evaluate(() => ({ b: window.LOOP.state().beat, p: window.LOOP.state().progress, ended: window.__speech.ended, started: window.__speech.spoken.length, active: !!window.__speech.active }));
    if (s.b === 0) { if (s.started >= 1) seen.push(s.p); } else advancedWith = s;   // from the moment the voice starts
    bad = (await domBeat(page)) === s.b ? 0 : bad + 1; assert.ok(bad < 3, 'text and scene disagree');
  }
  assert.ok(advancedWith, 'Play never moved on');
  assert.ok(advancedWith.started >= 3 && advancedWith.ended >= advancedWith.started - 1, `moved on while speaking: ${JSON.stringify(advancedWith)}`);
  for (let i = 1; i < seen.length; i++) assert.ok(seen[i] >= seen[i - 1] - 1e-9, 'progress fell');
  assert.ok(Math.max(...seen) > 0.5, 'progress follows the narration: ' + Math.max(...seen));
  await page.context().close();
});
test('a manual scroll during narration cancels the voice and hands control back', async () => {
  const page = await go();
  await page.check('#narrate'); await page.click('#mode-play'); await page.waitForTimeout(150);
  const before = await page.evaluate(() => window.__speech.cancels);
  await page.mouse.move(300, 400); await page.mouse.wheel(0, 40);
  await page.waitForFunction(() => window.LOOP.state().mode === 'scroll');
  assert.ok((await page.evaluate(() => window.__speech.cancels)) > before);
  assert.equal(await page.evaluate(() => window.speechSynthesis.speaking), false);
  await page.context().close();
});
test('switching Narrate off stops speaking at once', async () => {
  const page = await go();
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length >= 1);
  await page.uncheck('#narrate');
  assert.equal(await page.evaluate(() => window.speechSynthesis.speaking), false);
  const n = await page.evaluate(() => window.__speech.spoken.length); await page.waitForTimeout(400);
  assert.equal(await page.evaluate(() => window.__speech.spoken.length), n);
  await page.context().close();
});
test('Pause and resume use the browser voice', async () => {
  const page = await go();
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length >= 1);
  await page.click('#narrate-pause'); assert.equal(await page.evaluate(() => window.__speech.paused), true);
  await page.click('#narrate-pause'); assert.equal(await page.evaluate(() => window.__speech.paused), false);
  await page.context().close();
});
test('terms light up their scene objects while the sentence naming them is spoken', async () => {
  const page = await go();
  await page.check('#narrate'); await page.click('#mode-play');
  let lit = false;
  for (let k = 0; k < 100 && !lit; k++) { await page.waitForTimeout(30); lit = (await page.evaluate(() => window.LOOP.scene().highlight.length)) > 0; }
  assert.ok(lit, 'no term was highlighted during the narration of the first beat');
  await page.context().close();
});
test('the sentence being spoken is highlighted in the text', async () => {
  const page = await go();
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length >= 2);
  const r = await page.evaluate(() => window.CSS && CSS.highlights ? (CSS.highlights.has('speaking') ? CSS.highlights.get('speaking').size : 0) : -1);
  assert.ok(r === -1 || r >= 1, 'speaking highlight: ' + r);
  if (r === -1) assert.ok(await page.evaluate(() => !!document.querySelector('.speaking')));
  await page.context().close();
});
test('in Scroll mode the current beat is read after the reader rests, and scrolling on moves the voice along', async () => {
  const page = await go();
  await page.check('#narrate');
  await page.waitForFunction(() => window.__speech.spoken.length >= 1, null, { timeout: 8000 });
  assert.equal((await spoken(page))[0], 'Think before you write');
  const before = await page.evaluate(() => window.__speech.cancels);
  await page.evaluate(() => { const L = window.LOOP, i = L.beats.findIndex(b => b.id === 'lineage'), r = L.sections[i].getBoundingClientRect(); window.scrollTo(0, scrollY + r.top + 0.5 * r.height - L.drivers.anchorY()); });
  await page.waitForFunction(() => window.__speech.spoken.some(s => s.text === 'An old idea, rediscovered'), null, { timeout: 8000 });
  assert.ok((await page.evaluate(() => window.__speech.cancels)) > before, 'the first beat was cut off');
  await page.context().close();
});
test('the chosen voice and rate are used and remembered after a reload', async () => {
  const page = await go();
  await page.selectOption('#voice', { label: 'Daniel (en-GB)' }); await page.selectOption('#rate', '1.5');
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length >= 2);
  const first = await page.evaluate(() => window.__speech.spoken[0]);
  assert.equal(first.voice, 'Daniel'); assert.equal(first.rate, 1.5);
  await page.reload(); await page.waitForFunction(() => window.LOOP && window.LOOP.ready);
  assert.equal(await page.$eval('#voice', e => e.selectedOptions[0].textContent), 'Daniel (en-GB)');
  assert.equal(await page.$eval('#rate', e => e.value), '1.5');
  await page.context().close();
});
test('Article B narrates too, with its equations in words', async () => {
  const page = await go('inside-inkling.html');
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next'); // embed
  await page.waitForFunction(() => window.__speech.spoken.length > 4 && window.LOOP.state().progress === 1, null, { timeout: 15000 });
  const said = await spoken(page);
  assert.equal(said[0], 'Tokens to vectors');
  assert.ok(said.join(' ').includes('RMS norm divides each vector by its root mean square'));
  assert.ok(said.every(t => !/[\\${}]/.test(t)));
  await page.context().close();
});
test('without speech support the controls say so and nothing breaks', async () => {
  const page = await go('why-loop.html', { init: () => { Object.defineProperty(window, 'speechSynthesis', { value: undefined, configurable: true }); } });
  assert.equal(await page.isDisabled('#narrate'), true); assert.equal(await page.isDisabled('#voice'), true);
  assert.match(await page.textContent('#voice-note'), /not available/i);
  await page.click('#mode-step'); await page.click('#next');
  assert.deepEqual(page.errors, []);
  await page.context().close();
});

test('unchecking Narrate during Play lets Play carry on without the voice', async () => {
  const page = await go();
  await page.check('#narrate'); await page.click('#mode-play');
  await page.waitForFunction(() => window.__speech.spoken.length >= 1);
  await page.uncheck('#narrate');
  assert.equal(await page.evaluate(() => window.LOOP.state().mode), 'play');
  await page.waitForFunction(() => window.LOOP.state().beat >= 1, null, { timeout: 8000 });
  assert.equal(await page.evaluate(() => window.speechSynthesis.speaking), false);
  await page.context().close();
});
test('pausing and then pressing Next still reads the next beat', async () => {
  const page = await go();
  await page.click('#mode-step'); await page.check('#narrate'); await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.length >= 1);
  await page.click('#narrate-pause'); assert.equal(await page.evaluate(() => window.__speech.paused), true);
  await page.click('#next');
  await page.waitForFunction(() => window.__speech.spoken.some(s => s.text === 'Two ways to add depth'), null, { timeout: 20000 });
  assert.equal(await page.evaluate(() => window.__speech.paused), false);
  await page.context().close();
});
