/* The small steps of a transformer layer, recomputed in the browser from recorded tensors, so the page
   shows arithmetic that agrees with the real model (tests compare it with the recording to 1e-4).
   Plain arrays in, plain arrays out. RoPE here is Qwen's: features i and i + d/2 form a pair. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopWalkRef = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = a => Math.sqrt(dot(a, a));
  const matmul = (A, B) => A.map(row => B[0].map((_, j) => { let s = 0; for (let k = 0; k < row.length; k++) s += row[k] * B[k][j]; return s; }));

  /* mean square, 1 / sqrt(mean square + eps), and x * inverse * gain */
  function rmsnormParts(x, gain, eps) {
    let ms = 0; for (const v of x) ms += v * v; ms /= x.length;
    const inv = 1 / Math.sqrt(ms + eps);
    return { meanSquare: ms, invRms: inv, out: x.map((v, i) => v * inv * gain[i]) };
  }
  const rmsnorm = (x, gain, eps) => rmsnormParts(x, gain, eps).out;

  function softmax(v) {
    const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  }
  const silu = x => x / (1 + Math.exp(-x));
  const swiglu = (gate, up) => gate.map((g, i) => silu(g) * up[i]);

  /* inverse frequencies theta^(-2i/d), i = 0 .. d/2 - 1 */
  const ropeFreqs = (dim, theta) => Array.from({ length: dim / 2 }, (_, i) => 1 / Math.pow(theta, (2 * i) / dim));
  /* rotate each pair (x_i, x_{i+d/2}) by pos * inv_freq_i */
  function rope(x, pos, theta) {
    const d = x.length, h = d / 2, f = ropeFreqs(d, theta), out = new Array(d);
    for (let i = 0; i < h; i++) {
      const a = pos * f[i], c = Math.cos(a), s = Math.sin(a);
      out[i] = x[i] * c - x[i + h] * s;
      out[i + h] = x[i + h] * c + x[i] * s;
    }
    return out;
  }

  /* q, k: [T][d]. raw = q.k / sqrt(d); probs = softmax of each row with the causal mask (zeros above the diagonal). */
  function scores(q, k, headDim) {
    const raw = q.map(qi => k.map(kj => dot(qi, kj) / Math.sqrt(headDim)));
    const probs = raw.map((row, i) => { const p = softmax(row.slice(0, i + 1)); return p.concat(new Array(row.length - i - 1).fill(0)); });
    return { raw, probs };
  }
  return { dot, norm, matmul, rmsnormParts, rmsnorm, softmax, silu, swiglu, ropeFreqs, rope, scores };
});
