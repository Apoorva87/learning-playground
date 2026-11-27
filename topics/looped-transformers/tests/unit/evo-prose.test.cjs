/* The numbers typed into the "how it changed" beats must be the ones the design model computes. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const V = require('../../walkthrough/variants.js');
const html = fs.readFileSync(path.join(__dirname, '../../transformer-walkthrough.html'), 'utf8');
const facts = JSON.parse(fs.readFileSync(path.join(__dirname, '../../assets/facts.json'), 'utf8'));
const cfg = V.QWEN3, TOTAL = 596049920, GB = 596049920 * 2;
const section = id => html.match(new RegExp(`<section class="beat" id="b-${id}"[\\s\\S]*?</section>`))[0];
const text = id => section(id).replace(/<[^>]+>/g, '').replace(/\s+/g, ' ');
const r1 = x => Math.round(x * 10) / 10;

test('every number marked with a v.* fact reads exactly as the design model computes it', () => {
  let seen = 0;
  for (const [, id, shown] of html.matchAll(/<span data-fact="(v\.[^"]+)">([^<]*)<\/span>/g)) {
    assert.equal(Number(shown.replace(/,/g, '')), facts[id].value, `${id}: the page shows ${shown}`); seen++;
  }
  assert.ok(seen >= 25, 'checked ' + seen);
});
test('norm beat: operations, parameters and the share of the layer', () => {
  const t = text('evo-norm'), m = 5 * 2 * (V.attention('gqa', cfg).attnParams + V.ffn('swiglu', cfg).params);
  assert.equal(m, 157286400); assert.match(t, /about 157 million operations/);
  assert.ok(V.norm('layernorm', cfg).opsPerLayerForTokens / m < 0.0005, 'LayerNorm is under 0.05% of the layer');
  assert.equal(57 * 1024, 58368); assert.equal(57 * 2048, 116736); assert.match(t, /58,368/); assert.match(t, /116,736/);
});
test('position beat: the learned table is 5.6% of the model', () => {
  assert.equal(r1(100 * V.position('learned', cfg).params / TOTAL), 5.6); assert.match(text('evo-position'), /5\.6% of this model’s 596 million/);
  assert.equal(3 * (16 + 8) * 128, 9216); assert.equal(16 * 32768, 524288);
});
test('attention beat: ratios, GiB and the saving from sharing', () => {
  const t = text('evo-attention'), a = k => V.attention(k, cfg);
  assert.equal(r1(10240 / 1152), 8.9); assert.match(t, /about 8\.9 times smaller/);
  assert.equal(a('mha').cachePerTokenPerLayer / a('mqa').cachePerTokenPerLayer, 16); assert.match(t, /sixteen times smaller/);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => r1(x.cacheAtContext / 1e9)), [7.5, 3.8, 0.5]); assert.match(t, /7\.5, 3\.8 and 0\.5 GB\), against 1\.19 GB of weights/);
  const saved = 28 * (a('mha').attnParams - a('gqa').attnParams); assert.equal(saved, 58720256); assert.equal(r1(100 * saved / TOTAL), 9.9); assert.match(t, /58,720,256 weights/); assert.match(t, /about 9\.9% of the model/);
  assert.deepEqual([a('mha'), a('gqa'), a('mqa')].map(x => 2 * cfg.heads * 1 / (x.kvHeads * 2) * 2 / 2), [1, 2, 16]);   // n_h / n_kv
  assert.match(t, /1\.19 GB of weights/); assert.equal(Math.round(GB / 1e7) / 100, 1.19);
});
test('feed-forward beat: operations and bytes per token', () => {
  const t = text('evo-ffn'), f = k => V.ffn(k, cfg);
  assert.equal(r1(f('relu').flopsPerToken / 1e6), 16.8); assert.equal(r1(f('swiglu').flopsPerToken / 1e6), 18.9); assert.equal(r1(f('moe').flopsPerToken / 1e6), 37.8);
  assert.equal(r1(f('swiglu').params * 2 / 1e6), 18.9); assert.equal(r1(f('moe').activeParams * 2 / 1e6), 37.7); assert.equal(Math.round(f('moe').storedParams * 2 / 1e6), 151);
  for (const s of ['16.8 million', '18.9 million', '37.8 million', '18.9 MB', '37.7 MB', '151 MB']) assert.ok(t.includes(s), s);
  assert.equal(f('swiglu-matched').width, 2730); assert.match(t, /width 3,072, three times the model width/);
});
test('embedding beat: shares and sizes', () => {
  const t = text('evo-embed'), one = V.embedding('tied', cfg).params;
  assert.equal(r1(100 * one / TOTAL), 26.1); assert.equal(TOTAL + one, 751632384); assert.equal(one * 2, 311164928); assert.equal(V.embedding('untied', cfg).params * 2, 622329856);
  for (const s of ['26.1%', '751,632,384', '311,164,928 bytes tied', '622,329,856 untied', '596,049,920', 'about 311 MB']) assert.ok(t.includes(s), s);
});
test('every evolution beat names its sources and says what the comparison is computed from', () => {
  for (const id of ['evo-norm', 'evo-position', 'evo-attention', 'evo-ffn', 'evo-embed']) {
    const s = section(id); assert.ok((s.match(/data-fact="ev\./g) || []).length >= 2, id + ' cites its history'); assert.match(s, /data-variant-table=/); assert.match(s, /data-variants=/);
  }
});
test('no history claim is cited to a paper whose abstract does not make it', () => {
  assert.ok(!/2302\.05442/.test(JSON.stringify(facts)), 'the ViT-22B abstract does not mention QK normalisation');
  assert.ok(!/we hypothesize that re-scaling invariance/.test(html), 'RMSNorm’s abstract hypothesises re-centering invariance is dispensable');
});
test('prose a voice will read has no raw symbols (equations go through spoken text, inline math through TeX)', () => {
  for (const id of ['evo-norm', 'evo-position', 'evo-attention', 'evo-ffn', 'evo-embed']) {
    const t = section(id).replace(/<span class="eq"[^>]*>[\s\S]*?<\/span>/g, '').replace(/\\\([\s\S]*?\\\)/g, '').replace(/<[^>]+>/g, '');
    assert.ok(!/[_^{}\\]/.test(t), `${id}: ${(t.match(/.{20}[_^{}\\].{20}/) || [''])[0]}`);
  }
});

test('prose avoids symbols a voice would misread: operators, powers, fractions, abbreviations of units', () => {
  for (const id of ['evo-norm', 'evo-position', 'evo-attention', 'evo-ffn', 'evo-embed']) {
    const t = section(id).replace(/<span class="eq"[^>]*>[\s\S]*?<\/span>/g, '').replace(/\\\([\s\S]*?\\\)/g, '').replace(/<div class="cap sources">[\s\S]*?<\/div>/, '').replace(/<[^>]+>/g, '');
    const bad = t.match(/.{15}([×²]|\d\/\d|\d+x\b|\b\d+B parameters|GiB|\(x\+).{15}/g);
    assert.equal(bad, null, `${id}: ${bad}`);
  }
});
test('each evolution beat lists its sources as visible arXiv links', () => {
  for (const id of ['evo-norm', 'evo-position', 'evo-attention', 'evo-ffn', 'evo-embed']) {
    const m = section(id).match(/<div class="cap sources">([\s\S]*?)<\/div>/); assert.ok(m, id + ' has a sources line');
    const links = [...m[1].matchAll(/href="(https:\/\/arxiv\.org\/abs\/[\d.]+)"/g)].map(x => x[1]); assert.ok(links.length >= 1, id);
    for (const [, f] of section(id).matchAll(/data-fact="(ev\.[^"]+)"/g)) { const u = facts[f].source.url; if (/arxiv/.test(u)) assert.ok(links.includes(u), `${id}: ${f} cites ${u} but the sources line does not link it`); }
  }
});
test('the MLA sentence about a recorded run cites the recording, and the 93.3% figure names the baseline it compares against', () => {
  const t = text('evo-attention'); assert.match(t, /Compared with DeepSeek 67B/);
  assert.match(section('evo-attention'), /data-fact="mla\.bytes_mla"[^>]*>1,152/);
});
test('forward references do not claim the reader has already seen a later beat', () => {
  assert.doesNotMatch(text('evo-position'), /which you saw in the MLA beat/); assert.doesNotMatch(text('evo-norm'), /before the rotary beat/);
});
