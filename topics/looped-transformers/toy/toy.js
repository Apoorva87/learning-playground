/* The toy looped transformer, in plain JavaScript. It mirrors assets/toy/train_toy.py line for
   line (float64, same operation order), so the page shows the numbers the PyTorch model produces.
   Toy.run(model, variant, tokens, passes) -> { logits, argmax, trace }. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Toy = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const matvec = (W, x) => W.map(row => { let s = 0; for (let i = 0; i < x.length; i++) s += row[i] * x[i]; return s; });
  const add = (a, b) => a.map((v, i) => v + b[i]);
  function rms(x, eps, g) {
    let s = 0; for (const v of x) s += v * v;
    const r = 1 / Math.sqrt(s / x.length + eps);
    return g ? x.map((v, i) => v * r * g[i]) : x.map(v => v * r);
  }
  function softmax(v) {
    const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  }

  /* One pre-norm layer over all positions: bidirectional attention, then a ReLU MLP. */
  function layer(L, h, meta) {
    const { D, H, EPS } = meta, HD = D / H, T = h.length;
    const x = h.map(r => rms(r, EPS, L.g1));
    const q = x.map(r => matvec(L.wq, r)), k = x.map(r => matvec(L.wk, r)), v = x.map(r => matvec(L.wv, r));
    const att = [], mixed = Array.from({ length: T }, () => new Array(D).fill(0));
    for (let hh = 0; hh < H; hh++) {
      const lo = hh * HD, rows = [];
      for (let i = 0; i < T; i++) {
        const scores = [];
        for (let j = 0; j < T; j++) { let s = 0; for (let c = 0; c < HD; c++) s += q[i][lo + c] * k[j][lo + c]; scores.push(s / Math.sqrt(HD)); }
        const w = softmax(scores); rows.push(w);
        for (let j = 0; j < T; j++) for (let c = 0; c < HD; c++) mixed[i][lo + c] += w[j] * v[j][lo + c];
      }
      att.push(rows);
    }
    let out = h.map((r, i) => add(r, matvec(L.wo, mixed[i])));
    out = out.map(r => add(r, matvec(L.w2, matvec(L.w1, rms(r, EPS, L.g2)).map(z => Math.max(z, 0)))));
    return { h: out, att };
  }

  /* Message gate for one core layer: attention over the previous pass's layer outputs with a
     learned query, then (1 - f) * h + g * m. */
  function gate(G, h, sources, meta) {
    const { D, EPS } = meta;
    const out = [], f = [], g = [], w = [], ms = [];
    for (let i = 0; i < h.length; i++) {
      const S = sources.map(s => rms(s[i], EPS));
      const wi = softmax(S.map(r => { let s = 0; for (let c = 0; c < D; c++) s += r[c] * G.query[c]; return s / Math.sqrt(D); }));
      const m = new Array(D).fill(0);
      S.forEach((r, s) => { for (let c = 0; c < D; c++) m[c] += wi[s] * r[c]; });
      const z = matvec(G.down, rms(h[i], EPS).concat(rms(m, EPS)));
      const fi = matvec(G.wf, z).map(Math.tanh), gi = matvec(G.wm, z).map(Math.tanh);
      out.push(h[i].map((v, c) => (1 - fi[c]) * v + gi[c] * m[c]));
      f.push(fi); g.push(gi); w.push(wi); ms.push(m);
    }
    return { h: out, f, g, w, m: ms };
  }

  const load = weights => ({ ...weights });

  function run(model, variant, tokens, passes) {
    const W = model[variant], meta = model.meta, { D, EPS, N } = meta, T = N + 1;       // positions: N pairs and the query
    const emb = [];
    for (let i = 0; i < N; i++) emb.push(add(matvec(W.wsrc, W.tok[tokens[2 * i]]), matvec(W.wdst, W.tok[tokens[2 * i + 1]])));
    emb.push(matvec(W.wq, W.tok[tokens[2 * N]]));
    const e = layer(W.prelude, emb, meta).h;
    const logits = [], argmax = [], trace = [];
    let prev = null;
    for (let t = 1; t <= passes; t++) {
      let h = (variant === 'inkling' || t === 1) ? e : prev[prev.length - 1].map((r, i) => add(r, e[i]));
      const outs = [], attnRows = [], fs = [], gs = [], ws = [], hs = [], mm = [], os = [];
      W.core.forEach((L, j) => {
        if (variant === 'inkling' && t > 1) {
          hs.push(h[T - 1]);                                       // the state the gate saw, before mixing
          const r = gate(W.gates[j], h, prev, meta);
          h = r.h; fs.push(r.f[T - 1]); gs.push(r.g[T - 1]); ws.push(r.w[T - 1]); mm.push(r.m[T - 1]); os.push(h[T - 1]);
        }
        const r = layer(L, h, meta);
        h = r.h; outs.push(h);
        for (const head of r.att) attnRows.push(head[T - 1]);
      });
      prev = outs;
      const last = h[T - 1], lg = matvec(W.out, rms(last, EPS, W.gout));
      logits.push(lg); argmax.push(lg.indexOf(Math.max(...lg)));
      trace.push({ pass: t, hidden: h, hidden_last: last, attn: attnRows, gate: fs.length ? { f: fs, g: gs, w: ws, h: hs, m: mm, out: os } : null });
    }
    return { logits, argmax, trace };
  }

  return { load, run };
});
