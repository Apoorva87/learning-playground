/* The browser-side reference math must reproduce the recorded numbers. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const R = require('../../walkthrough/ref.js');
const A = path.join(__dirname, '../../assets/walkthrough');
const D = JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-data.json'), 'utf8')), M = JSON.parse(fs.readFileSync(path.join(A, 'walkthrough-mla.json'), 'utf8'));
const close = (a, b, tol, what) => { const A_ = [a].flat(Infinity), B_ = [b].flat(Infinity); assert.equal(A_.length, B_.length, what + ' length'); A_.forEach((x, i) => assert.ok(Math.abs(x - B_[i]) <= tol * Math.max(1, Math.abs(B_[i])), `${what}[${i}]: ${x} vs ${B_[i]}`)); };
const L0 = D.layer0, eps = D.config.eps, T = 5;

test('RMSNorm: mean square, inverse rms and output for every token, both norms', () => {
  D.embed.h0.forEach((row, i) => {
    const r = R.rmsnormParts(row, L0.norm1.gain, eps);
    close(r.meanSquare, L0.norm1.mean_square[i], 1e-5, 'ms'); close(r.invRms, L0.norm1.inv_rms[i], 1e-5, 'inv'); close(r.out, L0.norm1.out[i], 1e-4, 'out ' + i);
  });
  L0.resid1.out.forEach((row, i) => close(R.rmsnorm(row, L0.norm2.gain, eps), L0.norm2.out[i], 1e-4, 'norm2 ' + i));
});
test('QK-norm on head 0 (per-head RMSNorm over 128 features)', () => {
  L0.qkv.q0.forEach((row, i) => close(R.rmsnorm(row, L0.qknorm.q_gain, eps), L0.qknorm.q0_out[i], 1e-4, 'q ' + i));
  L0.qkv.k0.forEach((row, i) => close(R.rmsnorm(row, L0.qknorm.k_gain, eps), L0.qknorm.k0_out[i], 1e-4, 'k ' + i));
});
test('RoPE: half-split pairs rotated by position times the inverse frequencies', () => {
  close(R.ropeFreqs(128, D.config.rope_theta), L0.rope.inv_freq, 1e-5, 'inv_freq');
  L0.qknorm.q0_out.forEach((row, i) => close(R.rope(row, i, D.config.rope_theta), L0.rope.q0_out[i], 1e-4, 'q rope ' + i));
  L0.qknorm.k0_out.forEach((row, i) => close(R.rope(row, i, D.config.rope_theta), L0.rope.k0_out[i], 1e-4, 'k rope ' + i));
  assert.ok(Math.abs(R.norm(R.rope(L0.qknorm.q0_out[3], 3, D.config.rope_theta)) - R.norm(L0.qknorm.q0_out[3])) < 1e-3, 'rotation keeps the length');
});
test('scores, causal mask and softmax for head 0 match the recording', () => {
  const s = R.scores(L0.rope.q0_out, L0.rope.k0_out, D.config.head_dim);
  close(s.probs, L0.scores.heads[0].probs, 1e-4, 'probs'); s.probs.forEach((row, i) => row.forEach((p, j) => j > i && assert.equal(p, 0)));
  close(s.raw.map((row, i) => row.map((v, j) => (j <= i ? v : 0))), L0.scores.heads[0].raw, 1e-4, 'raw');
});
test('weighted values for head 0', () => close(R.matmul(L0.scores.heads[0].probs, L0.qkv.v0), L0.attn_out.head0, 1e-4, 'head 0 output'));
test('SwiGLU: SiLU of the gate times the up projection', () => {
  close(L0.ffn.gate_last.map(R.silu), L0.ffn.silu_last, 1e-4, 'silu'); close(R.swiglu(L0.ffn.gate_last, L0.ffn.up_last), L0.ffn.prod_last, 1e-4, 'prod');
});
test('softmax with temperature reproduces the recorded top probabilities', () => {
  const t = D.head.temperatures.find(x => x.t === 0.5).top;
  assert.ok(t[0].prob > D.head.chosen.prob, 'a lower temperature sharpens');
  const lows = R.softmax(D.head.logits.top20.map(x => x.logit / 0.5)), full = R.softmax(D.head.logits.top20.map(x => x.logit));
  assert.ok(lows[0] > full[0]);
  const p = R.softmax(D.head.logits.top20.map(x => x.logit / 0.5)); assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});
test('decode: the new query against the six cached keys of head 0', () => {
  const q = D.decode.q_head0, K = D.decode.k6_head0, s = K.map(k => R.dot(q, k) / Math.sqrt(D.config.head_dim));
  close(R.softmax(s), D.decode.probs[0], 1e-4, 'decode probs head 0');
});
test('MLA: explicit and absorbed scores both reproduce the recording', () => {
  const sc = M.config.softmax_scale, Tm = M.tokens.length;
  const explicit = M.query.q_nope_head0.map((qn, i) => M.explicit.k_nope_head0.map((kn, j) => (R.dot(qn, kn) + R.dot(M.query.q_pe_head0[i], M.latent.k_rope[j])) * sc));
  const absorbed = M.query.q_absorbed_head0.map((qa, i) => M.latent.c_kv.map((c, j) => (R.dot(qa, c) + R.dot(M.query.q_pe_head0[i], M.latent.k_rope[j])) * sc));
  const low = m => m.map((row, i) => row.map((v, j) => (j <= i ? v : 0)));
  close(low(explicit), M.explicit.scores_head0, 1e-4, 'explicit'); close(low(absorbed), M.absorbed.scores_head0, 1e-4, 'absorbed'); close(low(explicit), low(absorbed), 1e-4, 'explicit = absorbed');
  assert.equal(Tm, 5);
});
