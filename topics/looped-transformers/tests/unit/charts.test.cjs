/* The charts carry their numbers inline in the page. Check them against the committed sources. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const root = path.join(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'why-loop.html'), 'utf8');
const facts = JSON.parse(fs.readFileSync(path.join(root, 'assets/facts.json'), 'utf8'));
const hops = JSON.parse(fs.readFileSync(path.join(root, 'assets/hop-summary.json'), 'utf8'));
const charts = Object.fromEntries([...html.matchAll(/<figure class="chart[^"]*" data-chart='([^']+)'/g)].map(m => { const c = JSON.parse(m[1]); return [c.id, c]; }));
const numbers = claim => claim.match(/-?\d+(?:\.\d+)?(?:, |$)/g).join('').split(/,\s*/).filter(Boolean).map(Number);
const tail = c => c.slice(c.indexOf(':') + 1);

test('the research-model panels match the numbers in the model card facts', () => {
  assert.deepEqual(charts.deepswe.y, [0.7, 28.3, 47.4, 46.9, 48.2]);
  assert.deepEqual(charts.graphwalks.y, [21.1, 54.5, 63.2, 64.3, 64.9]);
  assert.deepEqual(charts.simpleqa.y, [6.3, 6.5, 7.3, 6.8, 6.7]);
  for (const [chart, fact] of [['deepswe', 'vl.deepswe'], ['graphwalks', 'vl.graphwalks'], ['simpleqa', 'vl.simpleqa']])
    assert.deepEqual(numbers(tail(facts[fact].claim).replace(/percent/, '').trim()), charts[chart].y, chart);
});
test('the Inkling 2 Mini panel matches hop-summary.json', () => {
  assert.deepEqual(charts.hops.y, hops.by_passes.map(b => b.correct));
  assert.deepEqual(charts.hops.note.map(s => +s.match(/[\d.]+$/)[0]), hops.by_passes.map(b => b.mean_p_among));
});
test('every chart has one series, a shared 0-100 axis for percentages, and matching x and y lengths', () => {
  for (const c of Object.values(charts)) { assert.equal(c.x.length, c.y.length, c.id); assert.ok(c.title && c.kind && c.ticks.length >= 2); }
  for (const id of ['deepswe', 'graphwalks', 'simpleqa']) assert.equal(charts[id].ymax, 100);
});
test('the table view repeats every plotted number', () => {
  const table = html.slice(html.indexOf('<details class="viewtable"'), html.indexOf('</details>', html.indexOf('<details class="viewtable"')));
  for (const v of [...charts.deepswe.y, ...charts.graphwalks.y, ...charts.simpleqa.y, ...charts.hops.y]) assert.ok(table.includes(String(v)), String(v));
});
