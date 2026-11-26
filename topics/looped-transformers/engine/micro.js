/* Micro scenes: the tensors inside one layer, drawn flat as grids of cells. A tensor's grid is
   proportional to its dimensions but capped (8 rows, 16 columns), so its label carries the true
   shape. Each tensor is a frame object `tensor.<name>` plus cells `tensor.<name>.r<i>c<j>`, and
   idsFor('tensor.<name>') expands to all of them, so a beat can light a whole tensor by name. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) factory(require('./layout.js'));
  else factory(root.LoopLayout);
})(typeof self !== 'undefined' ? self : this, function (Layout) {
  'use strict';
  const PITCH = 0.22, CELL = 0.17;

  /* A tensor drawn as a grid. rows/cols are the (capped) cells shown; x, y is the centre. */
  /* opts: pitch (cell spacing), alpha(r, c), v(r, c) (a signed value in -1..1 that the renderer shows as brightness and sign). */
  function grid(name, label, x, y, rows, cols, tint, opts = {}) {
    rows = Math.min(rows, 8); cols = Math.min(cols, 16);
    const pitch = opts.pitch || PITCH, cell = pitch * (CELL / PITCH);
    const id = 'tensor.' + name, w = cols * pitch + 0.1, h = rows * pitch + 0.1;
    const out = [{ id, kind: 'tensor', x, y, z: -0.05, w, h, d: 0.04, tint, pass: 0, full: false, alpha: 0.3, label }];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const o = { id: `${id}.r${r}c${c}`, kind: 'cell', x: x + (c - (cols - 1) / 2) * pitch, y: y + ((rows - 1) / 2 - r) * pitch, z: 0.02,
        w: cell, h: cell, d: 0.1, tint: opts.tint ? opts.tint(r, c) : tint, pass: 0, full: false, alpha: opts.alpha ? opts.alpha(r, c) : 0.95 };
      if (opts.v) o.v = opts.v(r, c);
      out.push(o);
    }
    return out;
  }
  const arrow = (id, x0, x1, y, tint = 'neutral') => ({ id, kind: 'arrow', x: (x0 + x1) / 2, y, z: 0, w: Math.max(Math.abs(x1 - x0), 0.1), h: 0.07, d: 0.07, tint, pass: 0, full: false, alpha: 0.9 });
  const vline = (id, x, y0, y1, tint = 'neutral') => ({ id, kind: 'arrow', x, y: (y0 + y1) / 2, z: 0, w: 0.07, h: Math.abs(y1 - y0), d: 0.07, tint, pass: 0, full: false, alpha: 0.9 });

  const MICRO = Layout.MICRO;
  const NODES = 'ABCDEFGH';                                       // the toy's eight nodes, named by letter
  const box = (id, label, x, y, w, h, tint, alpha = 0.9) => ({ id, kind: 'tensor', x, y, z: 0, w, h, d: 0.3, tint, pass: 0, full: false, alpha, label });
  const maxAbs = v => Math.max(...v.map(Math.abs), 1e-12);

  MICRO['message-gate'] = () => [
    ...grid('h', 'h  kept state  [T, 2304]', 0, 1.7, 4, 16, 'tensor'),
    ...grid('m', 'm  message  [T, 2304]', 0, -1.7, 4, 16, 'msg'),
    arrow('op.rms_h', 2.2, 3.2, 1.7), arrow('op.rms_m', 2.2, 3.2, -1.7),
    ...grid('rms_h', 'rms(h)', 5.6, 1.7, 4, 16, 'tensor'), ...grid('rms_m', 'rms(m)', 5.6, -1.7, 4, 16, 'msg'),
    arrow('op.down', 8.2, 9.2, 0),
    ...grid('z', 'z = W_down [rms(h); rms(m)]  [T, 64]', 10.8, 0, 4, 8, 'gate'),
    arrow('op.wf', 12.6, 13.4, 0.9), arrow('op.wm', 12.6, 13.4, -0.9),
    ...grid('f', 'f = tanh(W_f z)  forget  [T, 2304]', 16, 1.7, 4, 16, 'gate'), ...grid('g', 'g = tanh(W_m z)  message gate  [T, 2304]', 16, -1.7, 4, 16, 'gate'),
    arrow('op.out', 18.4, 19.4, 0),
    ...grid('out', 'out = (1−f)⊙h + g⊙m  [T, 2304]', 22, 0, 4, 16, 'hot'),
  ];

  MICRO['depth-attn'] = () => [
    ...[0, 1, 2, 3].flatMap(i => grid('block.' + i, `block state ${i}  [T, 2304]`, 0, 3.2 - i * 1.5, 3, 16, 'tensor')),
    ...grid('partial', 'current partial sum  [T, 2304]', 0, -3.4, 3, 16, 'core'),
    ...grid('query', 'learned query w  [2304]', 7, 0.4, 1, 16, 'gate'),
    arrow('op.score', 3.2, 5.2, 0.4), arrow('op.mix', 8.9, 10.2, 0.4),
    ...grid('weights', 'softmax over sources  [T, n]', 12, 2, 3, 5, 'msg'),
    ...grid('input', 'sublayer input  [T, 2304]', 12, -1.4, 3, 16, 'hot'),
  ];

  /* One attention sublayer, from the source: q, k, v with their own clips and short convolutions, a
     relative-position bias from the layer input, softmax with one extra "sink" logit, a sigmoid output gate. */
  const attention = (full) => [
    ...grid('x', 'normed input  [T, 2304]', 0, 0, 3, 16, 'tensor'),
    arrow('op.proj', 2.2, 3.2, 0),
    ...grid('q', full ? 'q  [T, 24, 128]: scale, L2 clip, log-scale' : 'q  [T, 24, 128]: scale, L2 clip', 6.4, 3.6, 3, 16, 'core'),
    ...grid('k', full ? 'k  [T, 8, 128]: dynamic conv, L2 clip' : 'k  [T, 8, 128]: conv, L2 clip', 6.4, 0.2, 3, 16, 'prelude'),
    ...grid('v', full ? 'v  [T, 8, 128]: dynamic conv, RMS norm' : 'v  [T, 8, 128]: conv, RMS norm', 6.4, -3.2, 3, 16, 'coda'),
    arrow('op.score', 9.3, 10.3, 1.4),
    ...grid('rel', 'relative bias  [24, T, S]', 13.4, 3.6, 3, 8, 'gate'),
    ...grid('scores', 'scores + bias, softmax with a sink  [24, T, S]', 13.4, 0.4, 3, 8, 'gate'),
    ...grid('sink', 'sink logit  [T, 24]', 13.4, -2.4, 1, 8, 'msg'),
    arrow('op.mix', 15.6, 16.6, -1.0),
    ...grid('o', 'attention ⊙ sigmoid gate, then W_o  [T, 2304]', 20.2, -2.2, 3, 16, 'core'),
  ];
  MICRO['sliding-layer'] = () => attention(false);
  MICRO['full-layer'] = () => attention(true);

  MICRO['moe'] = () => {
    const experts = [];
    for (let i = 0; i < 128; i++) experts.push({ id: 'expert.' + i, kind: 'expert', x: 8 + (i % 16) * 0.26, y: 2.4 - Math.floor(i / 16) * 0.26, z: 0, w: 0.2, h: 0.2, d: 0.12, tint: 'tensor', pass: 0, full: false, alpha: 0.9 });
    return [
      ...grid('x', 'normed input  [T, 2304]', 0, 0, 3, 16, 'tensor'),
      arrow('op.router', 2.2, 3.2, 0),
      ...grid('router', 'router logits  [T, 128]', 5.4, 0, 3, 16, 'gate'),
      ...experts,
      { id: 'label.experts', kind: 'tensor', x: 10, y: 1.5, z: -0.05, w: 4.4, h: 2.2, d: 0.04, tint: 'neutral', pass: 0, full: false, alpha: 0.12, label: '128 experts, 8 chosen per token' },
      ...grid('topk', 'top-8 weights × 8  [T, 8]', 10, -2.2, 3, 8, 'msg'),
      arrow('op.sum', 12.4, 13.6, -2.2),
      ...grid('out', 'weighted sum of 8 expert outputs  [T, 2304]', 16.4, -2.2, 3, 16, 'hot'),
    ];
  };

  MICRO['toy-core'] = () => [
    ...grid('tokens', 'pairs and query  [T, 16]', 0, 0, 5, 8, 'tensor'),
    arrow('op.embed', 1.5, 2.5, 0),
    ...grid('e', 'e, prelude output  [T, 16]   (Inkling: [T, 2304])', 4.8, 2.2, 5, 8, 'prelude'),
    ...grid('m', 'm, message  [T, 16]', 4.8, -2.2, 5, 8, 'msg'),
    arrow('op.gate', 6.4, 7.6, 0),
    ...grid('gate', 'f and g  [T, 16]', 9.4, 0, 5, 8, 'gate'),
    arrow('op.core', 11, 12, 0),
    ...grid('h', 'h after the core  [T, 16]   (Inkling: [T, 2304])', 14.2, 0, 5, 8, 'core'),
    arrow('op.read', 15.8, 16.8, 0),
    ...grid('readout', 'readout, 8 nodes  [8]', 18.6, 0, 1, 8, 'hot'),
  ];

  /* Attention of the first core layer's first head at the last position, one row per pass.
     data = { passes, tokens (17), attn[pass][9], argmax[pass] }. */
  MICRO['toy-hops'] = (data) => {
    const passes = data ? data.passes : 3, P = 0.62, x0 = 0, y0 = 0;
    const pairs = Array.from({ length: 9 }, (_, i) => data && data.tokens ? (i < 8 ? `${NODES[data.tokens[2 * i]]}→${NODES[data.tokens[2 * i + 1]]}` : `query ${NODES[data.tokens[16]]}`) : '?');
    const out = pairs.map((p, i) => box(`pair.${i}`, p, x0 + (i - 4) * P, y0 + 0.95, P * 0.9, 0.5, i < 8 ? 'tensor' : 'msg'));
    out.push(...grid('attn', 'attention of the last position, by pass  [passes, T]', x0, y0 - ((passes - 1) / 2) * P + 0.0, passes, 9, 'core', {
      pitch: P,
      alpha: (r, c) => 0.12 + 0.88 * (data && data.attn ? data.attn[r][c] : 0),
      tint: (r, c) => (data && data.attn && data.attn[r].indexOf(Math.max(...data.attn[r])) === c ? 'hot' : 'core'),
    }));
    for (let t = 0; t < passes; t++)
      out.push(box(`ptr.${t + 1}`, data && data.argmax ? `pass ${t + 1} reads out ${NODES[data.argmax[t]]}` : `pass ${t + 1}`, x0 - 6.1, y0 - t * P, 2.6, 0.45, 'hot', 0.55));
    return out;
  };

  /* The gate's arithmetic on the toy's real vectors at the last position.
     data = { gate: { h, m, f, g, out } } (each 16 numbers). */
  MICRO['toy-gate'] = (data) => {
    const g = data && data.gate, P = 0.5;
    const row = (name, label, y, tint, vals, scale) => grid(name, label, 0, y, 1, 16, tint, { pitch: P, v: vals ? (r, c) => vals[c] / (scale ? maxAbs(vals) : 1) : undefined });
    return [
      ...row('h', 'h  restarted state  [16]   (Inkling: [T, 2304])', 3.6, 'tensor', g && g.h, true),
      ...row('m', 'm  message from the last pass  [16]', 2.0, 'msg', g && g.m, true),
      ...row('f', 'f = tanh(W_f z)  forget  [16]', 0.4, 'gate', g && g.f, false),
      ...row('g', 'g = tanh(W_m z)  message gate  [16]', -1.2, 'gate', g && g.g, false),
      ...row('out', 'out = (1−f)⊙h + g⊙m  [16]', -3.0, 'hot', g && g.out, true),
    ];
  };

  /* Two ways to feed pass t+1: carry the state forward, or restart from e and add a gated message. */
  MICRO['two-ways'] = () => {
    const X = [0, 4.4, 8.8, 13.2], out = [];
    out.push(box('carry.e', 'e (prelude output)', X[0], 2.6, 2.2, 1.0, 'prelude'));
    for (let t = 1; t <= 3; t++) out.push(box(`carry.core.${t}`, `pass ${t}`, X[t], 2.6, 2.4, 1.0, 'core'));
    for (let t = 0; t < 3; t++) out.push(arrow(`carry.arrow.${t}`, X[t] + 1.3, X[t + 1] - 1.3, 2.6));
    out.push(box('inkling.e', 'e (prelude output)', X[0], -2.0, 2.2, 1.0, 'prelude'));
    for (let t = 1; t <= 3; t++) out.push(box(`inkling.core.${t}`, `pass ${t}`, X[t], -2.0, 2.4, 1.0, 'core'));
    out.push(arrow('inkling.bus', X[0], X[3], -0.55, 'prelude'));
    out.push(arrow('inkling.feed.1', X[0] + 1.2, X[1] - 1.3, -2.0, 'prelude'));
    for (let t = 2; t <= 3; t++) out.push(vline(`inkling.drop.${t}`, X[t], -0.55, -1.45, 'prelude'));
    for (let t = 1; t <= 2; t++) {
      const gx = (X[t] + X[t + 1]) / 2 + 0.4;
      out.push(vline(`inkling.msg.${t}.down`, X[t], -2.5, -3.5, 'msg'), arrow(`inkling.msg.${t}.along`, X[t], gx - 0.7, -3.5, 'msg'));
      out.push(box(`inkling.gate.${t}`, 'gate', gx, -3.5, 1.4, 0.7, 'gate'));
      out.push(arrow(`inkling.msg.${t}.in`, gx + 0.7, X[t + 1], -3.5, 'msg'), vline(`inkling.msg.${t}.up`, X[t + 1], -3.5, -2.5, 'msg'));
    }
    out.push(box('label.carry', 'carry: pass t+1 starts from the state pass t ended in', 6.6, 4.3, 12, 0.5, 'neutral', 0.02));
    out.push(box('label.inkling', 'Inkling: every pass restarts from e and receives a gated message', 6.6, -5.1, 12, 0.5, 'neutral', 0.02));
    return out;
  };

  /* s_t = a * s_{t-1} + 1: the same loop with a spectral radius below and above 1. */
  MICRO['stability'] = () => {
    const out = [], series = (a) => { const s = []; let v = 0; for (let t = 0; t < 8; t++) { v = a * v + 1; s.push(v); } return s; };
    const stable = series(0.6), unstable = series(1.2), top = Math.max(...unstable);
    [['stable', stable, 'prelude', 0], ['unstable', unstable, 'hot', 10]].forEach(([k, s, tint, x0]) => {
      out.push(box(`bar.${k}.label`, k === 'stable' ? 'ρ = 0.6: the state settles' : 'ρ = 1.2: the state blows up', x0 + 3.5, 5.0, 8, 0.5, 'neutral', 0.02));
      s.forEach((v, t) => { const h = 4 * v / top + 0.05; out.push({ id: `bar.${k}.${t + 1}`, kind: 'cell', x: x0 + t, y: h / 2, z: 0, w: 0.8, h, d: 0.4, tint, pass: 0, full: false, alpha: 0.9 }); });
    });
    out.push({ id: 'bar.axis', kind: 'arrow', x: 8.5, y: -0.1, z: 0, w: 18.5, h: 0.06, d: 0.06, tint: 'neutral', pass: 0, full: false, alpha: 0.8, label: 'pass t = 1 … 8' });
    return out;
  };

  MICRO['embed'] = () => [
    ...grid('tokens', 'token ids  [T]', 0, 0, 1, 8, 'tensor'),
    arrow('op.lookup', 2.0, 3.2, 0),
    ...grid('table', 'embedding table  [201024, 2304]', 6.6, 0, 8, 16, 'prelude'),
    arrow('op.pick', 10.2, 11.4, 0),
    ...grid('rows', 'rows picked  [T, 2304]', 14.4, 0, 3, 16, 'core'),
    arrow('op.norm', 18.0, 19.2, 0),
    ...grid('h', 'h₀ = RMSNorm(E[ids])  [T, 2304]', 22.2, 0, 3, 16, 'hot'),
  ];

  /* message_input: for each loop layer, four learned queries read the previous pass's states. */
  MICRO['message-input'] = () => {
    const roles = ['q', 'k', 'v', 'mlp'];
    return [
      ...grid('src.0', 'previous pass: mid-loop block state  [T, 2304]', 0, 2.2, 3, 16, 'tensor'),
      ...grid('src.1', 'previous pass: final hidden state  [T, 2304]', 0, -2.2, 3, 16, 'core'),
      arrow('op.read', 2.2, 3.4, 0),
      ...roles.flatMap((r, i) => grid('query.' + r, `w_${r}  learned query  [2304]`, 6.2, 4.2 - i * 2.8, 1, 16, 'gate')),
      arrow('op.attend', 8.6, 9.8, 0),
      ...roles.flatMap((r, i) => grid('msg.' + r, `m_${r}  message  [T, 2304]`, 13.2, 4.2 - i * 2.8, 2, 16, 'msg')),
    ];
  };

  MICRO['head'] = () => [
    ...grid('final', 'final depth attention  [T, 2304]', 0, 0, 3, 16, 'tensor'),
    arrow('op.norm', 2.2, 3.2, 0),
    ...grid('norm', 'RMSNorm  [T, 2304]', 6.0, 0, 3, 16, 'core'),
    arrow('op.lm', 8.6, 9.6, 0),
    ...grid('raw', 'lm_head: raw logits  [T, 200058]', 13.2, 0, 3, 16, 'gate'),
    arrow('op.cap', 16.1, 17.1, 0),
    ...grid('logits', 'tanh(x / 30) · 30  softcapped logits  [T, 200058]', 22.4, 0, 3, 16, 'hot'),
  ];
});

