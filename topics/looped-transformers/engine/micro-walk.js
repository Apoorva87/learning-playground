/* Micro scenes for Article C, "One sentence through a modern transformer". Every scene draws the
   recorded numbers (first 16 features of each vector, real attention weights, real probabilities) with
   the same cell-grid style as the other micro scenes. data = { d: walkthrough-data, m: MLA data, token, head }.
   Without data each scene draws its structure with empty cells, so a beat can frame it before the data loads. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) factory(require('./layout.js'));
  else factory(root.LoopLayout);
})(typeof self !== 'undefined' ? self : this, function (Layout) {
  'use strict';
  const { grid, arrow, vline, box, maxAbs } = Layout.microKit, MICRO = Layout.MICRO;
  const fmt = (v, d = 3) => (v === undefined ? '…' : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +v.toPrecision(d) + '');
  const shown = lens => (lens ? lens.text.trim() || 'a space' : '…');
  const tokText = (d, i) => (d ? d.meta.tokens[i].text.trim() || '·' : 'tok ' + i);
  const T = d => (d ? d.meta.tokens.length : 5);
  const norm = rows => { const m = Math.max(1e-12, ...rows.flat().map(Math.abs)); return rows.map(r => r.map(v => v / m)); };   // signed values scaled to the largest
  const vfn = rows => (rows ? (r, c) => rows[r][c] : undefined);
  const first = (rows, n = 16) => rows.map(r => r.slice(0, n));
  const D = data => data && data.d, Mla = data => data && data.m;
  const cell = (id, x, y, w, h, tint, alpha = 0.9, label) => ({ id, kind: 'cell', x, y, z: 0, w, h, d: 0.3, tint, pass: 0, full: false, alpha, label });

  /* The sentence, cut into tokens with their IDs. */
  MICRO['walk-tokens'] = data => {
    const d = D(data), n = T(d), out = [box('sentence', d ? `“${d.meta.sentence}”` : 'sentence', 0, 2.2, n * 3.2, 0.9, 'neutral', 0.3)];
    for (let i = 0; i < n; i++) {
      const x = (i - (n - 1) / 2) * 3.2;
      out.push(box(`tok.${i}`, d ? `“${d.meta.tokens[i].text}”` : 'token', x, 0, 2.8, 1.0, 'prelude'), box(`id.${i}`, d ? `ID ${d.meta.tokens[i].id}` : 'ID', x, -1.7, 2.8, 0.8, 'core'));
      out.push(vline(`cut.${i}`, x, 1.6, 0.6));
    }
    return out;
  };

  /* IDs pick rows of the embedding table. */
  MICRO['walk-embed'] = data => {
    const d = D(data), n = T(d), out = [], h0 = d ? first(d.embed.h0) : null, sv = h0 ? norm(h0) : null;
    out.push(...grid('ids', 'token IDs  [5]', 0, 0, n, 1, 'core', { pitch: 0.5 }));
    out.push(arrow('op.lookup', 1.0, 2.6, 0));
    out.push(...grid('table', 'embedding table  [151936, 1024]', 6.0, 0, 8, 16, 'prelude', { pitch: 0.2, alpha: () => 0.35 }));
    out.push(arrow('op.pick', 9.0, 10.6, 0));
    out.push(...grid('h0', d ? 'h₀ = the rows picked  [5, 1024]  (first 16 of 1024 shown)' : 'h₀  [5, 1024]', 14.6, 0, n, 16, 'hot', { pitch: 0.38, v: sv ? vfn(sv) : undefined }));
    for (let i = 0; i < n; i++) out.push(cell(`rn.${i}`, 21.2, (n - 1) / 2 * 0.38 - i * 0.38, 2.6, 0.3, 'neutral', 0.25, d ? `${d.meta.tokens[i].text.trim()}  ‖h‖ = ${fmt(d.embed.row_norms[i])}` : 'row norm'));
    return out;
  };

  /* One decoder layer, zoomed out. */
  MICRO['walk-layer'] = data => {
    const d = D(data), n = d ? d.meta.tokens.length : 5, sh = `[${n}, 1024]`;
    const rows = [['norm1', 'RMSNorm', 'prelude'], ['attn', 'Attention: 16 query heads share 8 key/value heads', 'core'], ['add1', '+ residual', 'hot'], ['norm2', 'RMSNorm', 'prelude'], ['ffn', 'SwiGLU feed-forward: 1024 → 3072 → 1024', 'coda'], ['add2', '+ residual', 'hot']];
    const out = [box('blk.x', `x  ${sh}`, 0, 8.2, 8, 0.9, 'tensor')];
    rows.forEach(([id, label, tint], i) => { const y = 6.2 - i * 2.2; out.push(box('blk.' + id, label, 0, y, 8, 1.2, tint), vline(`blk.arrow.${i}`, 0, y + 1.7, y + 0.7)); });
    out.push(box('blk.out', `layer output  ${sh}`, 0, -7.2, 8, 0.9, 'tensor'), vline('blk.arrow.6', 0, -4.2, -6.7));
    out.push(vline('blk.skip1', 5.2, 8.2, 3.8, 'neutral'), vline('blk.skip2', 5.2, 3.0, -5.6, 'neutral'));
    return out;
  };

  /* RMSNorm on one token's real vector. */
  MICRO['walk-norm'] = data => {
    const d = D(data), t = data && data.token !== undefined ? data.token : 4, L = d && d.layer0.norm1;
    const x = d ? [d.embed.h0[t].slice(0, 16)] : null, g = d ? [L.gain.slice(0, 16)] : null, y = d ? [L.out[t].slice(0, 16)] : null;
    const name = d ? d.meta.tokens[t].text.trim() : 'token';
    return [
      ...grid('x', `x: the vector of “${name}”  [1024]  (first 16 shown)`, 0, 3.0, 1, 16, 'tensor', { pitch: 0.5, v: x ? vfn(norm(x)) : undefined }),
      box('stat.ms', d ? `mean of squares = ${fmt(L.mean_square[t], 4)}` : 'mean of squares', -2.2, 1.0, 5.2, 0.8, 'gate'),
      box('stat.inv', d ? `1 / √(mean + ε) = ${fmt(L.inv_rms[t], 4)}` : '1 / √(mean + ε)', 3.4, 1.0, 5.2, 0.8, 'gate'),
      ...grid('gain', 'learned gain  [1024]', 0, -0.6, 1, 16, 'msg', { pitch: 0.5, v: g ? vfn(norm(g)) : undefined }),
      ...grid('y', 'y = x · (1 / √(mean + ε)) · gain  [1024]', 0, -3.0, 1, 16, 'hot', { pitch: 0.5, v: y ? vfn(norm(y)) : undefined }),
    ];
  };

  /* The three projections, for the last token: 16 query heads, 8 key heads, 8 value heads. */
  MICRO['walk-qkv'] = data => {
    const d = D(data), Q = d && d.layer0.qkv;
    return [
      ...grid('xn', 'normed input  [5, 1024]', 0, 0, 5, 16, 'tensor', { pitch: 0.2, v: d ? vfn(norm(first(d.layer0.norm1.out))) : undefined }),
      arrow('op.proj', 2.2, 3.4, 0),
      ...grid('q', 'q = x W_q  [5, 2048] = 16 heads × 128  (last token, 8 of 128 dims per head)', 7.4, 3.6, 8, 8, 'core', { pitch: 0.3, v: Q ? vfn(norm(Q.q_last_heads.slice(0, 8))) : undefined }),
      ...grid('k', 'k = x W_k  [5, 1024] = 8 heads × 128', 7.4, -1.0, 8, 8, 'prelude', { pitch: 0.3, v: Q ? vfn(norm(Q.k_last_heads)) : undefined }),
      ...grid('v', 'v = x W_v  [5, 1024] = 8 heads × 128', 7.4, -4.6, 8, 8, 'coda', { pitch: 0.3, v: Q ? vfn(norm(Q.v_last_heads)) : undefined }),
    ];
  };

  /* Grouped-query attention: which query heads share which key/value head. */
  MICRO['walk-gqa'] = () => {
    const out = [], P = 1.0;
    for (let h = 0; h < 16; h++) out.push(cell(`qh.${h}`, (h - 7.5) * P, 3.0, 0.8, 0.9, 'core', 0.9, h % 2 === 0 ? `q${h}` : undefined));
    for (let k = 0; k < 8; k++) out.push(cell(`kvh.${k}`, (2 * k - 7) * P, -1.0, 1.8, 0.9, 'prelude', 0.9, `k/v ${k}`));
    for (let h = 0; h < 16; h++) out.push({ id: `link.${h}.${h >> 1}`, kind: 'arrow', x: ((h - 7.5) + (2 * (h >> 1) - 7)) / 2 * P, y: 1.0, z: 0, w: 0.04 + Math.abs((h - 7.5) - (2 * (h >> 1) - 7)) * P, h: 0.05, d: 0.05, tint: 'neutral', pass: 0, full: false, alpha: 0.6 });
    out.push(box('gqa.note', '2 query heads read each key/value head: the cache holds 8 heads, not 16', 0, -3.2, 14, 0.8, 'neutral', 0.1));
    return out;
  };

  /* QK-norm on query head 0: before, the per-feature gain, after. */
  MICRO['walk-qknorm'] = data => {
    const d = D(data), Q = d && d.layer0.qkv, N = d && d.layer0.qknorm;
    return [
      ...grid('before', 'q head 0 before the norm  [5, 128]  (first 16 dims)', 0, 3.4, 5, 16, 'core', { pitch: 0.3, v: Q ? vfn(norm(first(Q.q0))) : undefined }),
      ...grid('gain', 'learned gain  [128]', 0, 0.9, 1, 16, 'msg', { pitch: 0.3, v: N ? vfn(norm([N.q_gain.slice(0, 16)])) : undefined }),
      ...grid('after', 'after RMSNorm over the 128 features of the head  [5, 128]', 0, -1.6, 5, 16, 'hot', { pitch: 0.3, v: N ? vfn(norm(first(N.q0_out))) : undefined }),
      ...Array.from({ length: 5 }, (_, i) => cell(`inv.${i}`, 5.6, 4.1 - i * 0.3, 2.6, 0.26, 'neutral', 0.2, N ? `${d.meta.tokens[i].text.trim()}: ×${fmt(N.q0_inv_rms[i], 3)}` : 'scale')),
    ];
  };

  /* RoPE: the frequencies, the angle for each position, and a vector before and after the rotation. */
  MICRO['walk-rope'] = data => {
    const d = D(data), R = d && d.layer0.rope, Q = d && d.layer0.qknorm, out = [];
    const f = R ? R.inv_freq : Array.from({ length: 64 }, (_, i) => Math.pow(1e6, -i / 64));
    f.forEach((v, i) => { const h = 0.2 + 0.9 * (Math.log10(v) + 6) / 6.1; out.push(cell(`freq.${i}`, (i - 31.5) * 0.19, 4.2 + h / 2 - 0.5, 0.15, h, 'prelude', 0.9, i === 0 ? 'fast: 1 turn per 6 positions' : i === 63 ? 'slow' : undefined)); });
    out.push(box('rope.title', 'inverse frequencies θ^(−2i/128), i = 0 … 63, θ = 1,000,000', 0, 5.6, 12.6, 0.6, 'neutral', 0.1));
    out.push(...grid('angles', 'rotation angle = position × frequency  (first 12 pairs, in radians)', 0, 1.7, 5, 12, 'gate', { pitch: 0.3, v: R ? vfn(norm(R.angles.map(r => r.slice(0, 12)))) : undefined }));
    out.push(...grid('before', 'q head 0 before RoPE  [5, 128]', -5.6, -2.0, 5, 16, 'core', { pitch: 0.2, v: Q ? vfn(norm(first(Q.q0_out))) : undefined }));
    out.push(...grid('after', 'after RoPE  [5, 128]  (length kept, direction turned)', 5.6, -2.0, 5, 16, 'hot', { pitch: 0.2, v: R ? vfn(norm(first(R.q0_out))) : undefined }));
    out.push(arrow('op.rot', -3.6, 3.6, -2.0));
    return out;
  };

  /* Scores, causal mask, softmax: the real 5 × 5 for one head. */
  MICRO['walk-scores'] = data => {
    const d = D(data), n = T(d), h = (data && data.head) || 0, S = d && d.layer0.scores.heads[h], out = [];
    const mask = (r, c) => c > r;
    out.push(...grid('raw', `scores  q·k / √128, head ${h}  [5, 5]`, -4.0, 0, n, n, 'gate', { pitch: 0.55, alpha: (r, c) => (mask(r, c) ? 0.08 : 0.9), v: S ? (r, c) => (mask(r, c) ? 0 : S.raw[r][c] / 10) : undefined }));
    out.push(arrow('op.softmax', -1.6, -0.2, 0));
    out.push(...grid('probs', `attention weights after the mask and softmax  [5, 5]`, 3.6, 0, n, n, 'hot', { pitch: 0.55, alpha: (r, c) => (mask(r, c) ? 0.08 : 0.3 + 0.7 * (S ? S.probs[r][c] : 0.3)), v: S ? (r, c) => S.probs[r][c] : undefined }));
    for (let i = 0; i < n; i++) {
      out.push(cell(`rowname.${i}`, 7.1, (n - 1) / 2 * 0.55 - i * 0.55, 1.5, 0.3, 'neutral', 0.2, tokText(d, i)));
      out.push(cell(`colname.${i}`, 3.6 + (i - (n - 1) / 2) * 0.55, 2.0, 0.5, 0.3, 'neutral', 0.2, tokText(d, i).slice(0, 5)));
    }
    return out;
  };

  /* Weighted values, concatenated heads, output projection, residual. */
  MICRO['walk-mix'] = data => {
    const d = D(data), L = d && d.layer0;
    return [
      ...grid('probs', 'weights  [5, 5]', 0, 2.4, 5, 5, 'hot', { pitch: 0.4, v: L ? vfn(L.scores.heads[0].probs) : undefined }),
      ...grid('v', 'values, head 0  [5, 128]', 3.6, 2.4, 5, 16, 'coda', { pitch: 0.2, v: L ? vfn(norm(first(L.qkv.v0))) : undefined }),
      arrow('op.mix', 6.0, 7.0, 2.4),
      ...grid('head0', 'head 0 output = weights · values  [5, 128]', 10.6, 2.4, 5, 16, 'core', { pitch: 0.2, v: L ? vfn(norm(first(L.attn_out.head0))) : undefined }),
      ...grid('concat', 'all 16 heads side by side  [5, 2048]  (last token, first 16)', 0, -1.2, 1, 16, 'core', { pitch: 0.3, v: L ? vfn(norm([L.attn_out.concat_last.slice(0, 16)])) : undefined }),
      arrow('op.wo', 3.0, 4.2, -1.2),
      ...grid('o', 'after W_o  [5, 1024]', 8.4, -1.2, 5, 16, 'hot', { pitch: 0.2, v: L ? vfn(norm(first(L.attn_out.o_proj_out))) : undefined }),
      ...grid('resid1', 'x + attention  (the residual)  [5, 1024]', 8.4, -4.2, 5, 16, 'tensor', { pitch: 0.2, v: L ? vfn(norm(first(L.resid1.out))) : undefined }),
    ];
  };

  /* SwiGLU feed-forward for the last token. */
  MICRO['walk-ffn'] = data => {
    const d = D(data), F = d && d.layer0.ffn, row = a => (a ? [a.slice(0, 16)] : null), v = a => (a ? vfn(norm(row(a))) : undefined);
    return [
      ...grid('x', 'normed input  [1024]', 0, 6.0, 1, 16, 'tensor', { pitch: 0.4, v: d ? vfn(norm([d.layer0.norm2.out[4].slice(0, 16)])) : undefined }),
      ...grid('gate', 'gate = x W_gate  [3072]', -4.6, 3.2, 1, 16, 'gate', { pitch: 0.4, v: v(F && F.gate_last) }),
      ...grid('up', 'up = x W_up  [3072]', 4.6, 3.2, 1, 16, 'msg', { pitch: 0.4, v: v(F && F.up_last) }),
      ...grid('silu', 'SiLU(gate) = gate · σ(gate)', -4.6, 0.6, 1, 16, 'gate', { pitch: 0.4, v: v(F && F.silu_last) }),
      ...grid('prod', 'SiLU(gate) ⊙ up  [3072]', 0, -1.8, 1, 16, 'core', { pitch: 0.4, v: v(F && F.prod_last) }),
      ...grid('down', 'down = prod W_down  [1024]', 0, -4.2, 1, 16, 'hot', { pitch: 0.4, v: v(F && F.down_last) }),
    ];
  };

  /* The 28-layer stack: hidden-state norm and the logit-lens guess at each layer. */
  MICRO['walk-stack'] = data => {
    const d = D(data), S = d && d.stack, out = [], maxN = S ? Math.max(...S.layer_norms_last) : 1;
    for (let l = 0; l < 28; l++) {
      const hgt = 0.3 + 4.2 * (S ? S.layer_norms_last[l] / maxN : 0.3), lens = S && S.logit_lens[l], good = lens && /Paris/.test(lens.text);
      out.push(cell(`layer.${l}`, (l - 13.5) * 0.75, hgt / 2 - 2.5, 0.6, hgt, good ? 'hot' : 'core', 0.9, l === 0 ? `layer 0 guesses “${shown(lens)}”` : l === 14 ? `layer 14: “${shown(lens)}”` : l === 27 ? `layer 27: “${shown(lens)}” ${lens ? fmt(lens.prob * 100, 2) + '%' : ''}` : undefined));
    }
    out.push(box('stack.axis', 'height = norm of the last token’s hidden state after each layer; red = the layer’s own guess is “ Paris”', 0, -3.0, 21, 0.6, 'neutral', 0.1));
    return out;
  };

  /* Final norm, the vocabulary projection, softmax, and the choice. */
  MICRO['walk-head'] = data => {
    const d = D(data), Hd = d && d.head, out = [];
    out.push(...grid('final', 'final hidden state, normed  [1024]', -6.0, 3.6, 1, 16, 'tensor', { pitch: 0.4, v: Hd ? vfn(norm([Hd.final_norm_out_last.slice(0, 16)])) : undefined }));
    out.push(...grid('logits', Hd ? `logits = h · Eᵀ  [${Hd.logits.vocab.toLocaleString('en-US')}]  (the first 16)` : 'logits  [151936]', 4.0, 3.6, 1, 16, 'gate', { pitch: 0.4, v: Hd ? vfn(norm([Hd.logits.first16])) : undefined }));
    out.push(arrow('op.lm', -2.4, 0.6, 3.6));
    const top = Hd ? Hd.logits.top20 : Array.from({ length: 20 }, (_, i) => ({ text: '…', prob: 0.2 / (i + 1) }));
    top.forEach((t, i) => out.push(cell(`bar.${i}`, (i - 9.5) * 0.9, -1.6 + Math.max(0.05, 4.0 * t.prob / top[0].prob) / 2, 0.7, Math.max(0.05, 4.0 * t.prob / top[0].prob), i === 0 ? 'hot' : 'core', 0.9, i < 8 ? `${t.text.trim() || '·'} ${fmt(t.prob * 100, 2)}%` : undefined)));
    out.push(box('head.axis', 'softmax(logits): the 20 likeliest of 151,936 tokens; the tallest is picked', 0, -2.4, 18.4, 0.6, 'neutral', 0.1));
    return out;
  };

  /* The KV cache that prefill leaves behind: one K and one V per layer, [8, 5, 128]. */
  MICRO['walk-cache'] = data => {
    const d = D(data), out = [], n = T(d);
    out.push(...grid('k', 'K cache, layer 0  [8 heads, 5 positions, 128]', -4.0, 2.6, 8, n, 'prelude', { pitch: 0.42 }));
    out.push(...grid('v', 'V cache, layer 0  [8, 5, 128]', 2.4, 2.6, 8, n, 'coda', { pitch: 0.42 }));
    for (let l = 0; l < 28; l++) out.push(cell(`cache.layer.${l}`, (l - 13.5) * 0.7, -2.6, 0.55, 1.3, 'tensor', 0.7, l === 0 ? 'layer 0' : l === 27 ? 'layer 27' : undefined));
    out.push(box('cache.note', '28 layers × (K + V) × 8 heads × 128 × 2 bytes = 114,688 bytes per token', 0, -4.0, 19.6, 0.7, 'neutral', 0.12));
    return out;
  };

  /* Decode: one new row; its query reads the whole cache. */
  MICRO['walk-decode'] = data => {
    const d = D(data), De = d && d.decode, out = [];
    out.push(...grid('newq', 'new query, 16 heads × 8 of 128 dims  (position 5)', -6.2, 2.2, 8, 8, 'core', { pitch: 0.3, v: De ? vfn(norm(De.q_heads_first8.slice(0, 8))) : undefined }));
    out.push(...grid('cachek', 'K cache of layer 0, head 0, 6 positions  (5 old + 1 new)', 0, 2.8, 6, 4, 'prelude', { pitch: 0.4, v: De ? vfn(norm(De.cache_keys_head0_first4)) : undefined }));
    out.push(cell('newrow', 0, 2.8 + (6 - 1) / 2 * 0.4 - 5 * 0.4, 1.8, 0.34, 'hot', 0.25, 'the new key, appended'));
    out.push(...grid('dprobs', 'attention of each of the 16 heads over the 6 keys  [16, 6]', 6.4, 2.2, 8, 6, 'hot', { pitch: 0.4, v: De ? vfn(De.probs.slice(0, 8)) : undefined }));
    out.push(arrow('op.read', 2.4, 3.4, 2.8));
    if (De) De.logits_top10.slice(0, 5).forEach((t, i) => out.push(cell(`dnext.${i}`, -6.0 + i * 3.0, -2.8, 2.7, 0.8, i === 0 ? 'hot' : 'tensor', 0.9, `${t.text.trim() || '·'}  ${fmt(t.prob * 100, 2)}%`)));
    else for (let i = 0; i < 5; i++) out.push(cell(`dnext.${i}`, -6.0 + i * 3.0, -2.8, 2.7, 0.8, 'tensor', 0.5, 'next token'));
    out.push(box('decode.note', 'the next-token distribution after “Paris” (decode step)', 0, -4.0, 14, 0.6, 'neutral', 0.1));
    return out;
  };

  /* MLA: cache the small latent, rebuild keys and values from it. */
  MICRO['walk-mla'] = data => {
    const m = Mla(data), n = m ? m.tokens.length : 5, c = m ? m.cache_bytes : null;
    return [
      ...grid('x', `normed input  [${n}, 2048]`, -9.4, 0, n, 16, 'tensor', { pitch: 0.2 }),
      arrow('op.down', -7.4, -6.2, 0),
      ...grid('ckv', `latent c_KV = norm(x W_DKV)  [${n}, 512]  (this is cached)`, -2.8, 2.0, n, 16, 'hot', { pitch: 0.3, v: m ? vfn(norm(first(m.latent.c_kv))) : undefined }),
      ...grid('krope', `RoPE key  [${n}, 64]  (also cached, shared by all heads)`, -2.8, -1.6, n, 16, 'gate', { pitch: 0.3, v: m ? vfn(norm(first(m.latent.k_rope))) : undefined }),
      arrow('op.up', 1.2, 2.4, 2.0),
      ...grid('kv', `keys and values rebuilt per head  [${n}, 16, 128] each`, 6.4, 2.0, n, 16, 'prelude', { pitch: 0.3, alpha: () => 0.35 }),
      box('mla.cache', c ? `cached per token and layer: (512 + 64) × 2 B = ${c.mla_per_token_per_layer.toLocaleString('en-US')} B, against ${c.mha_per_token_per_layer.toLocaleString('en-US')} B for full multi-head keys and values` : 'cached per token', 0, -4.6, 18, 0.9, 'neutral', 0.12),
    ];
  };

  /* Absorbed attention: fold the key up-projection into the query and score against the latent. */
  MICRO['walk-mla-absorb'] = data => {
    const m = Mla(data), n = m ? m.tokens.length : 5, mask = (r, c) => c > r, h = 0.5;
    const sc = mat => (r, c) => (mask(r, c) ? 0 : mat[r][c] / 12);
    return [
      ...grid('qabs', `q_nope · W_UK  [${n}, 512]  (query, absorbed)`, -8.0, 3.2, n, 16, 'core', { pitch: 0.26, v: m ? vfn(norm(first(m.query.q_absorbed_head0))) : undefined }),
      ...grid('clat', `cached latent c_KV  [${n}, 512]`, 0.2, 3.2, n, 16, 'hot', { pitch: 0.26, v: m ? vfn(norm(first(m.latent.c_kv))) : undefined }),
      ...grid('explicit', m ? `explicit: rebuild each key, then q·k  [${n}, ${n}]` : 'explicit', -4.6, -1.8, n, n, 'prelude', { pitch: h, alpha: (r, c) => (mask(r, c) ? 0.08 : 0.9), v: m ? sc(m.explicit.scores_head0) : undefined }),
      ...grid('absorbed', m ? `absorbed: q·W_UK against the latent  [${n}, ${n}]   (largest difference ${m.absorbed.max_abs_diff.toExponential(1)})` : 'absorbed', 3.4, -1.8, n, n, 'gate', { pitch: h, alpha: (r, c) => (mask(r, c) ? 0.08 : 0.9), v: m ? sc(m.absorbed.scores_head0) : undefined }),
      arrow('op.same', -1.6, 0.6, -1.8),
    ];
  };

  /* Cache bytes per token per layer for four ways of storing keys and values. */
  MICRO['walk-cachesize'] = data => {
    const d = D(data), m = Mla(data), hd = d ? d.config.head_dim : 128, heads = d ? d.config.heads : 16, kv = d ? d.config.kv_heads : 8;
    const mha = heads * hd * 2 * 2, gqa = kv * hd * 2 * 2, mqa = 1 * hd * 2 * 2, mla = m ? m.cache_bytes.mla_per_token_per_layer : (512 + 64) * 2, top = mha;
    const bar = (id, x, b, tint, text) => cell(id, x, b / top * 4.2 / 2 - 2.0, 2.4, b / top * 4.2, tint, 0.9, `${text}: ${b.toLocaleString('en-US')} B`);
    return [
      bar('cache.mha', -7.5, mha, 'core', `MHA, ${heads} K/V heads`), bar('cache.gqa', -2.5, gqa, 'prelude', `GQA, ${kv} heads`), bar('cache.mqa', 2.5, mqa, 'coda', 'MQA, 1 head'), bar('cache.mla', 7.5, mla, 'hot', 'MLA latent'),
      box('cache.note', 'bytes of K and V per token per layer in BF16 (MHA, GQA, MQA use Qwen3-0.6B’s 128-wide heads; MLA is DeepSeek-V2-Lite’s)', 0, -3.0, 20, 0.7, 'neutral', 0.1),
    ];
  };
});
