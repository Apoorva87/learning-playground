/* Numbers typed into the article's prose and equations (not filled from the recording) must agree with it. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const R = require('../../walkthrough/ref.js');
const D = JSON.parse(fs.readFileSync(path.join(__dirname, '../../assets/walkthrough/walkthrough-data.json'), 'utf8'));
const html = fs.readFileSync(path.join(__dirname, '../../transformer-walkthrough.html'), 'utf8');
const r = (x, d) => Math.round(x * 10 ** d) / 10 ** d;
const L0 = D.layer0;

test('RMSNorm example: 0.0183 x 42.61 = 0.780, then x 0.1377 = 0.1074', () => {
  const x = D.embed.h0[4][0], inv = L0.norm1.inv_rms[4], g = L0.norm1.gain[0];
  assert.equal(r(x, 4), 0.0183); assert.equal(r(inv, 2), 42.61); assert.equal(r(x * inv, 3), 0.78); assert.equal(r(g, 4), 0.1377); assert.equal(r(x * inv * g, 4), 0.1074);
  for (const typed of ['0.780', '0.1074', '42.61', '0.1377']) assert.ok(html.includes(typed), typed);
});
test('after the learned gain the output no longer has unit root-mean-square, and the text does not say it does', () => {
  const out = L0.norm1.out[4], rms = Math.sqrt(out.reduce((a, v) => a + v * v, 0) / out.length);
  assert.ok(Math.abs(rms - 1) > 0.5, 'rms after gain: ' + rms);
  assert.doesNotMatch(html, /After the norm the vector has unit root-mean-square, whatever size/);
});
test('RoPE example: pair (-0.8644, -0.2421) at position 4 turns by 4 radians to (0.3818, 0.8125)', () => {
  const x0 = L0.qknorm.q0_out[4][0], x1 = L0.qknorm.q0_out[4][64], phi = 4 * L0.rope.inv_freq[0];
  assert.equal(r(phi, 6), 4); assert.equal(r(x0, 4), -0.8644); assert.equal(r(x1, 4), -0.2421);
  assert.equal(r(x0 * Math.cos(phi) - x1 * Math.sin(phi), 4), 0.3818); assert.equal(r(x1 * Math.cos(phi) + x0 * Math.sin(phi), 4), 0.8125);
  assert.ok(html.includes('0.3818'));
});
test('attention example: raw scores 8.17 and 8.06 are close, 4.57 is smaller by about 3.6', () => {
  const row = L0.scores.heads[0].raw[4];
  assert.equal(r(row[0], 2), 8.17); assert.equal(r(row[2], 2), 8.06); assert.equal(r(row[1], 2), 4.57); assert.equal(r(row[0] - row[1], 1), 3.6);
});
test('QK-norm: the head is multiplied by the inverse rms and then the gain, which flips the sign of the third feature', () => {
  const pre = L0.qkv.q0[4], inv = L0.qknorm.q0_inv_rms[4], g = L0.qknorm.q_gain;
  assert.equal(r(pre[2], 4), 0.0572); assert.equal(r(L0.qknorm.q0_out[4][2], 4), -0.1708);
  assert.ok(Math.abs(pre[2] * inv * g[2] - L0.qknorm.q0_out[4][2]) < 1e-4 && g[2] < 0);
});
test('decode cost figures: 500 GB/s over 1.19 GB is about 420 tokens/s; the cache is 40% of the weights at 4,096 tokens and 3x at 32,768', () => {
  const weights = 596049920 * 2, perToken = 114688;
  assert.equal(Math.round(500e9 / weights), 419);
  assert.equal(Math.round(4096 * perToken / weights * 100), 39); assert.equal(Math.round(32768 * perToken / weights * 10) / 10, 3.2);
  assert.equal(r(32768 * 229376 / 1e9, 1), 7.5); assert.equal(r(32768 * perToken / 1e9, 2), 3.76); assert.equal(r(32768 * 14336 / 1e9, 2), 0.47);
});
test('logit lens: Paris is the guess from layer 23 on, and the hidden norm falls at the last layer', () => {
  const S = D.stack;
  assert.ok(/Paris/.test(S.logit_lens[23].text) && /Paris/.test(S.logit_lens[27].text) && !/Paris/.test(S.logit_lens[22].text), 'first Paris layer');
  assert.ok(S.layer_norms_last[27] < S.layer_norms_last[26]);
});
