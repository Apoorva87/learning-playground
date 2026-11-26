const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const Toy = require('../../toy/toy.js');
const dir = path.join(__dirname, '../../assets/toy');
const read = f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
const weights = read('toy-weights.json'), vectors = read('toy-vectors.json'), acc = read('toy-accuracy.json'), ev = read('toy-eval.json');
const model = Toy.load(weights);
const close = (a, b, what) => {
  const A = [a].flat(Infinity), B = [b].flat(Infinity);
  assert.equal(A.length, B.length, what + ' length');
  A.forEach((x, i) => assert.ok(Math.abs(x - B[i]) < 1e-5, `${what}[${i}]: ${x} vs ${B[i]}`));
};

for (const variant of ['carry', 'inkling']) {
  test(`${variant}: JavaScript matches the PyTorch reference on saved vectors within 1e-5`, () => {
    for (const row of vectors[variant]) {
      const r = Toy.run(model, variant, row.tokens, weights.meta.passes_max);
      row.passes.forEach((p, t) => {
        close(r.logits[t], p.logits, `pass ${t + 1} logits`);
        close(r.trace[t].hidden_last, p.hidden_last, `pass ${t + 1} hidden`);
        close(r.trace[t].attn, p.attn, `pass ${t + 1} attention`);
        if (p.f) { close(r.trace[t].gate.f, p.f, `pass ${t + 1} f`); close(r.trace[t].gate.g, p.g, `pass ${t + 1} g`); close(r.trace[t].gate.w, p.msg_w, `pass ${t + 1} message weights`); }
        else assert.equal(r.trace[t].gate, null);
      });
    }
  });
  test(`${variant}: the page's accuracy-versus-passes table reproduces on the saved held-out sequences`, () => {
    const hits = new Array(weights.meta.passes_max).fill(0);
    ev.tokens.forEach((tokens, i) => {
      const r = Toy.run(model, variant, tokens, weights.meta.passes_max);
      r.argmax.forEach((a, t) => { if (a === ev.targets[i][t]) hits[t]++; });
    });
    hits.forEach((h, t) => assert.ok(Math.abs(h / ev.tokens.length - ev[variant][t]) < 1e-9, `pass ${t + 1}: ${h / ev.tokens.length} vs ${ev[variant][t]}`));
  });
  test(`${variant}: accuracy file is consistent and honest about passes it never trained on`, () => {
    const a = acc[variant];
    assert.equal(a.passes.length, a.accuracy.length);
    assert.deepEqual(a.extrapolation_passes, [5, 6]);
    assert.ok(a.accuracy.every(x => x >= 0 && x <= 1));
  });
}
test('the inkling variant has gates, the carry variant has none', () => {
  assert.ok(weights.inkling.gates.length === 2 && !weights.carry.gates);
  const r = Toy.run(model, 'carry', vectors.carry[0].tokens, 2);
  assert.equal(r.trace[1].gate, null);
});
test('passes are independent of how many are requested', () => {
  const tk = vectors.inkling[0].tokens, a = Toy.run(model, 'inkling', tk, 3), b = Toy.run(model, 'inkling', tk, 6);
  close(a.logits, b.logits.slice(0, 3), 'prefix logits');
});
test('the gate trace is self-consistent: out = (1 - f) * h + g * m at the last position', () => {
  const r = Toy.run(model, 'inkling', vectors.inkling[1].tokens, 4);
  for (const pass of [1, 2, 3]) {
    const g = r.trace[pass].gate;
    g.f.forEach((f, j) => close(g.out[j], f.map((fv, c) => (1 - fv) * g.h[j][c] + g.g[j][c] * g.m[j][c]), `pass ${pass + 1} layer ${j}`));
  }
});
