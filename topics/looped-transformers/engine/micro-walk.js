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

  /* ---------- hover and click detail ----------
     Each data-driven object gets info {title, lines}: the exact recorded value and the arithmetic behind it. */
  const idx = id => { const m = /\.r(\d+)c(\d+)$/.exec(id); return m ? [+m[1], +m[2]] : null; };
  const I = (title, ...lines) => ({ title, lines: lines.flat().filter(Boolean) });
  const g = (v, n = 4) => String(Number(v.toPrecision(n)));
  const pct = v => Number((v * 100).toPrecision(3)) + '%';
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const withInfo = (name, fn) => {
    const base = MICRO[name];
    MICRO[name] = data => {
      const out = base(data), d = D(data);
      if (d) for (const o of out) { const info = fn(o, o.id, idx(o.id), data, d); if (info) o.info = info; else if (!o.info && o.kind !== 'arrow' && o.label) o.info = { ...I(o.label.replace(/\s+/g, ' '), /\[/.test(o.label) ? 'A frame around one tensor; hover over its cells for exact values.' : 'A caption for this part of the picture; hover over the shapes for exact values.'), generic: true }; }
      return out;
    };
  };
  const tk = (d, i) => d.meta.tokens[i].text;
  const q = (d, i) => `“${tk(d, i)}”`;

  withInfo('walk-tokens', (o, id, rc, data, d) => { const m = /^(tok|id)\.(\d)$/.exec(id); if (!m) return null; const t = d.meta.tokens[+m[2]];
    return I(`Token ${m[2]}: “${t.text}”`, `ID ${t.id}`, `position ${m[2]}`, t.text.startsWith(' ') ? 'the leading space is part of the token' : 'the first token has no leading space'); });
  withInfo('walk-embed', (o, id, rc, data, d) => {
    if (rc && id.startsWith('tensor.h0.')) return I(`h₀[${rc[0]}][${rc[1]}] = ${g(d.embed.h0[rc[0]][rc[1]], 3)}`, `token ${q(d, rc[0])} (ID ${d.meta.tokens[rc[0]].id}), feature ${rc[1]} of 1024 (first 16 shown)`, `this row's length ‖h₀[${rc[0]}]‖ = ${g(d.embed.row_norms[rc[0]], 3)}`);
    if (rc && id.startsWith('tensor.ids.')) return I(`Token ID ${d.meta.tokens[rc[0]].id}`, `selects row ${d.meta.tokens[rc[0]].id} of the embedding table`);
    if (id.startsWith('tensor.table.')) return I('Embedding table E', '151,936 rows × 1024 numbers = 155,582,464 parameters', 'only five rows are read for this sentence');
    const m = /^rn\.(\d)$/.exec(id); return m ? I(`Row norm of ${q(d, +m[1])}`, `‖h₀‖ = ${g(d.embed.row_norms[+m[1]], 4)}`) : null; });
  withInfo('walk-layer', (o, id) => ({ 'blk.x': I('Input x', 'shape [5, 1024]: five tokens, 1024 numbers each'), 'blk.norm1': I('RMSNorm before attention', '1024 learned gains', 'about 4 × 1024 operations per token'),
    'blk.attn': I('Attention sublayer', '6,291,456 weights: W_Q, W_K, W_V, W_O', '62.9 million operations for 5 tokens, plus 0.2 million for scores and mixing'), 'blk.add1': I('Residual add', 'x + attention output: nothing is overwritten'),
    'blk.norm2': I('RMSNorm before the feed-forward network', '1024 learned gains'), 'blk.ffn': I('SwiGLU feed-forward network', '9,437,184 weights: gate, up, down', '94.4 million operations for 5 tokens'),
    'blk.add2': I('Residual add', 'h′ + feed-forward output'), 'blk.out': I('Layer output', 'same shape as the input: [5, 1024]') }[id] || null));
  withInfo('walk-norm', (o, id, rc, data, d) => { const t = data.token === undefined ? 4 : data.token, L = d.layer0.norm1;
    if (rc && id.startsWith('tensor.x.')) return I(`x[${rc[1]}] = ${g(d.embed.h0[t][rc[1]], 3)}`, `feature ${rc[1]} of the vector of ${q(d, t)}`);
    if (rc && id.startsWith('tensor.gain.')) return I(`gain[${rc[1]}] = ${g(L.gain[rc[1]], 3)}`, 'a learned per-feature scale');
    if (rc && id.startsWith('tensor.y.')) return I(`y[${rc[1]}] = ${g(L.out[t][rc[1]], 3)}`, `= x × inverse × gain = ${g(d.embed.h0[t][rc[1]], 3)} × ${g(L.inv_rms[t], 4)} × ${g(L.gain[rc[1]], 3)}`);
    if (id === 'stat.ms') return I('Mean of squares', `(1/1024) Σ x² = ${g(L.mean_square[t], 4)}`); if (id === 'stat.inv') return I('Inverse root', `1 / √(${g(L.mean_square[t], 4)} + 10⁻⁶) = ${g(L.inv_rms[t], 4)}`); return null; });
  withInfo('walk-qkv', (o, id, rc, data, d) => { if (!rc) return null; const [r, c] = rc, Q = d.layer0.qkv;
    if (id.startsWith('tensor.q.')) return I(`query head ${r}, feature ${c} = ${g(Q.q_last_heads[r][c], 3)}`, `for the last token ${q(d, 4)}; 8 of 128 features shown`, `query head ${r} shares key/value head ${r >> 1}`);
    if (id.startsWith('tensor.k.')) return I(`key head ${r}, feature ${c} = ${g(Q.k_last_heads[r][c], 3)}`, `serves query heads ${2 * r} and ${2 * r + 1}`);
    if (id.startsWith('tensor.v.')) return I(`value head ${r}, feature ${c} = ${g(Q.v_last_heads[r][c], 3)}`, `serves query heads ${2 * r} and ${2 * r + 1}`);
    return I(`normed input[${r}][${c}] = ${g(d.layer0.norm1.out[r][c], 3)}`, `token ${q(d, r)}`); });
  withInfo('walk-gqa', (o, id, rc, data, d) => { let m;
    if ((m = /^qh\.(\d+)$/.exec(id))) { const h = +m[1]; return I(`Query head ${h}`, `shares key/value head ${h >> 1} with query head ${h ^ 1}`, 'its own queries, shared keys and values'); }
    if ((m = /^kvh\.(\d+)$/.exec(id))) { const k = +m[1]; return I(`Key/value head ${k}`, `serves query heads ${2 * k} and ${2 * k + 1}`, 'cached once: 2 × 128 × 2 bytes = 512 bytes per token per layer'); }
    if ((m = /^link\.(\d+)\.(\d+)$/.exec(id))) return I(`Query head ${m[1]} reads key/value head ${m[2]}`, `floor(${m[1]} / 2) = ${m[2]}`); return null; });
  withInfo('walk-qknorm', (o, id, rc, data, d) => { const Q = d.layer0.qkv, N = d.layer0.qknorm; let m;
    if (rc && id.startsWith('tensor.before.')) return I(`q₀[${rc[0]}][${rc[1]}] = ${g(Q.q0[rc[0]][rc[1]], 3)} before the norm`, `token ${q(d, rc[0])}, feature ${rc[1]} of 128`);
    if (rc && id.startsWith('tensor.gain.')) return I(`gain[${rc[1]}] = ${g(N.q_gain[rc[1]], 3)}`, 'learned per-feature scale of the query norm');
    if (rc && id.startsWith('tensor.after.')) return I(`after the norm = ${g(N.q0_out[rc[0]][rc[1]], 3)}`, `= before × inverse rms × gain = ${g(Q.q0[rc[0]][rc[1]], 3)} × ${g(N.q0_inv_rms[rc[0]], 4)} × ${g(N.q_gain[rc[1]], 3)}`);
    return (m = /^inv\.(\d)$/.exec(id)) ? I(`Inverse rms for ${q(d, +m[1])}`, `${g(N.q0_inv_rms[+m[1]], 4)}: the head is multiplied by this before the gain`) : null; });
  withInfo('walk-rope', (o, id, rc, data, d) => { const Rp = d.layer0.rope, Q = d.layer0.qknorm; let m;
    if ((m = /^freq\.(\d+)$/.exec(id))) { const f = Rp.inv_freq[+m[1]]; return I(`Pair ${m[1]}: ${g(f, 4)} rad per position`, `θ^(−2·${m[1]}/128) with θ = 10⁶`, `one full turn every ${g(2 * Math.PI / f, 3)} positions`); }
    if (rc && id.startsWith('tensor.angles.')) return I(`Angle for position ${rc[0]}, pair ${rc[1]} = ${g(Rp.angles[rc[0]][rc[1]], 4)} rad`, `= ${rc[0]} × ${g(Rp.inv_freq[rc[1]], 4)}`, `token ${q(d, rc[0])}`);
    if (rc && id.startsWith('tensor.before.')) return I(`q before RoPE[${rc[0]}][${rc[1]}] = ${g(Q.q0_out[rc[0]][rc[1]], 4)}`, `token ${q(d, rc[0])}; paired with feature ${rc[1] + 64} = ${g(Q.q0_out[rc[0]][rc[1] + 64], 4)}`);
    if (rc && id.startsWith('tensor.after.')) { const [r, c] = rc, a = Rp.angles[r][c]; return I(`q after RoPE[${r}][${c}] = ${g(Rp.q0_out[r][c], 4)}`, `pair (${g(Q.q0_out[r][c], 4)}, ${g(Q.q0_out[r][c + 64], 4)}) turned by ${g(a, 4)} rad (angle = ${r} × ${g(Rp.inv_freq[c], 4)})`, `x′ = x cos φ − y sin φ = ${g(Q.q0_out[r][c], 4)}·${g(Math.cos(a), 4)} − ${g(Q.q0_out[r][c + 64], 4)}·${g(Math.sin(a), 4)} = ${g(Rp.q0_out[r][c], 4)}`, `its partner becomes ${g(Rp.q0_out[r][c + 64], 4)}`); }
    return null; });
  withInfo('walk-scores', (o, id, rc, data, d) => { const h = (data && data.head) || 0, S = d.layer0.scores.heads[h]; let m;
    if (rc && id.startsWith('tensor.raw.')) { const [i, j] = rc; if (j > i) return I(`Masked: ${q(d, i)} cannot look at ${q(d, j)}`, 'a token may only attend to itself and earlier tokens: this score is set to −∞'); const qv = d.layer0.rope.q0_out[i], kv = d.layer0.rope.k0_out[j], dt = dot(qv, kv);
      return I(`Score of ${q(d, i)} looking at ${q(d, j)} = ${g(S.raw[i][j], 3)}`, h === 0 ? [`query·key = ${g(dt, 4)} (128 features, after QK-norm and RoPE)`, `÷ √128 (11.31) = ${g(S.raw[i][j], 3)}`] : `recorded score for head ${h}`, `softmax weight ${pct(S.probs[i][j])}`); }
    if (rc && id.startsWith('tensor.probs.')) { const [i, j] = rc; return j > i ? I(`Masked: ${q(d, i)} cannot look at ${q(d, j)}`, 'a later token: weight 0') : I(`Attention weight of ${q(d, i)} on ${q(d, j)}: ${pct(S.probs[i][j])}`, `softmax of the row of scores: e^${g(S.raw[i][j], 3)} / Σ e^score`, `row ${i} sums to 100% over ${i + 1} visible tokens`); }
    return (m = /^(rowname|colname)\.(\d)$/.exec(id)) ? I(`${m[1] === 'rowname' ? 'Query' : 'Key'}: ${q(d, +m[2])}`, `position ${m[2]}`) : null; });
  withInfo('walk-mix', (o, id, rc, data, d) => { if (!rc) return null; const [r, c] = rc, L0 = d.layer0;
    if (id.startsWith('tensor.probs.')) return I(`Weight ${q(d, r)} → ${q(d, c)} = ${pct(L0.scores.heads[0].probs[r][c])}`, 'head 0 attention weight');
    if (id.startsWith('tensor.v.')) return I(`value[${r}][${c}] = ${g(L0.qkv.v0[r][c], 3)}`, `token ${q(d, r)}, feature ${c} of 128 (key/value head 0)`);
    if (id.startsWith('tensor.head0.')) return I(`head 0 output[${r}][${c}] = ${g(L0.attn_out.head0[r][c], 3)}`, `= Σ over visible tokens of weight × value, for token ${q(d, r)}`);
    if (id.startsWith('tensor.concat.')) return I(`concatenated heads[${c}] = ${g(L0.attn_out.concat_last[c], 3)}`, 'last token, first 16 of 2048 numbers (16 heads × 128)');
    if (id.startsWith('tensor.o.')) return I(`after W_O[${r}][${c}] = ${g(L0.attn_out.o_proj_out[r][c], 3)}`, `token ${q(d, r)}: 2048 numbers mapped back to 1024`);
    return I(`x + attention[${r}][${c}] = ${g(L0.resid1.out[r][c], 3)}`, `= ${g(d.embed.h0[r][c], 3)} + ${g(L0.attn_out.o_proj_out[r][c], 3)}`, 'the residual keeps the original and adds the correction'); });
  withInfo('walk-ffn', (o, id, rc, data, d) => { if (!rc) return null; const c = rc[1], F = d.layer0.ffn;
    if (id.startsWith('tensor.x.')) return I(`normed input[${c}] = ${g(d.layer0.norm2.out[4][c], 3)}`, 'after the second RMSNorm, token “ is”');
    if (id.startsWith('tensor.gate.')) return I(`gate[${c}] = ${g(F.gate_last[c], 3)}`, 'x · W_gate, one of 3072 values');
    if (id.startsWith('tensor.up.')) return I(`up[${c}] = ${g(F.up_last[c], 3)}`, 'x · W_up, one of 3072 values');
    if (id.startsWith('tensor.silu.')) return I(`SiLU(gate)[${c}] = ${g(F.silu_last[c], 3)}`, `= gate / (1 + e^−gate) = ${g(F.gate_last[c], 3)} / (1 + e^${g(-F.gate_last[c], 3)})`);
    if (id.startsWith('tensor.prod.')) return I(`product[${c}] = ${g(F.prod_last[c], 3)}`, `= SiLU(gate) × up = ${g(F.silu_last[c], 3)} × ${g(F.up_last[c], 3)}`);
    return I(`down[${c}] = ${g(F.down_last[c], 3)}`, 'prod · W_down: 3072 numbers narrowed to 1024'); });
  withInfo('walk-stack', (o, id, rc, data, d) => { const m = /^layer\.(\d+)$/.exec(id); if (!m) return null; const l = +m[1], lens = d.stack.logit_lens[l];
    return I(`After layer ${l}`, `hidden-state norm ‖h‖ = ${g(d.stack.layer_norms_last[l], 4)}`, `logit lens: the guess here is “${lens.text.trim() || 'a space'}” with probability ${pct(lens.prob)}`); });
  withInfo('walk-head', (o, id, rc, data, d) => { const Hd = d.head; let m;
    if ((m = /^bar\.(\d+)$/.exec(id))) { const t = Hd.logits.top20[+m[1]]; return I(`${t.text.trim() || 'blank'}  ${pct(t.prob)}`, `token ID ${t.id}`, `logit ${g(t.logit, 4)}`, `probability = e^${g(t.logit, 4)} / Σ over 151,936 tokens = ${pct(t.prob)}`, +m[1] === 0 ? 'the tallest bar: greedy decoding picks it' : `rank ${+m[1] + 1} of 151,936`); }
    if (rc && id.startsWith('tensor.final.')) return I(`final normed vector[${rc[1]}] = ${g(Hd.final_norm_out_last[rc[1]], 3)}`, 'last token, first 16 of 1024');
    if (rc && id.startsWith('tensor.logits.')) return I(`logit of token ${rc[1]} = ${g(Hd.logits.first16[rc[1]], 3)}`, 'the dot product of the final vector with that row of E');
    return null; });
  withInfo('walk-cache', (o, id, rc, data, d) => { let m;
    if (rc && /^tensor\.[kv]\./.test(id)) { const kind = id.startsWith('tensor.k.') ? 'key' : 'value'; return I(`Layer 0 ${kind}, head ${rc[0]}, position ${rc[1]}`, `token ${q(d, rc[1])}`, '128 numbers × 2 bytes = 256 bytes', 'stored after QK-norm and RoPE for keys'); }
    return (m = /^cache\.layer\.(\d+)$/.exec(id)) ? I(`Layer ${m[1]} cache`, '2 × 8 heads × 128 × 2 bytes = 4,096 bytes per token', `${(5 * 4096).toLocaleString('en-US')} bytes for the 5 tokens`) : null; });
  withInfo('walk-decode', (o, id, rc, data, d) => { const De = d.decode; let m;
    if (rc && id.startsWith('tensor.newq.')) return I(`new query, head ${rc[0]}, feature ${rc[1]} = ${g(De.q_heads_first8[rc[0]][rc[1]], 3)}`, 'token “ Paris” at position 5, after QK-norm and RoPE');
    if (rc && id.startsWith('tensor.cachek.')) return I(`cached key, position ${rc[0]}, feature ${rc[1]} = ${g(De.cache_keys_head0_first4[rc[0]][rc[1]], 3)}`, 'head 0 of layer 0; positions 0–4 come from prefill, position 5 was just appended');
    if (rc && id.startsWith('tensor.dprobs.')) return I(`head ${rc[0]} puts ${pct(De.probs[rc[0]][rc[1]])} on ${rc[1] < 5 ? q(d, rc[1]) : '“ Paris” (itself)'}`, `softmax over the 6 cached keys; the row sums to 100%`);
    if ((m = /^dnext\.(\d)$/.exec(id))) { const t = De.logits_top10[+m[1]]; return I(`${t.text.trim() || 'blank'}  ${pct(t.prob)}`, `token ID ${t.id}`, `logit ${g(t.logit, 4)}`); }
    return id === 'newrow' ? I('The new key', 'computed for position 5 and appended to the cache') : null; });
  withInfo('walk-mla', (o, id, rc, data, d) => { const m = data.m; if (!m) return null; const tt = i => m.tokens[i].text.trim() || 'start';
    if (rc && id.startsWith('tensor.ckv.')) return I(`latent c_KV[${rc[0]}][${rc[1]}] = ${g(m.latent.c_kv[rc[0]][rc[1]], 3)}`, `token ${tt(rc[0])}; 512 numbers are cached for it`);
    if (rc && id.startsWith('tensor.krope.')) return I(`RoPE key[${rc[0]}][${rc[1]}] = ${g(m.latent.k_rope[rc[0]][rc[1]], 3)}`, `token ${tt(rc[0])}; 64 numbers, shared by all 16 heads`);
    if (rc && id.startsWith('tensor.kv.')) return I('Rebuilt on demand', 'keys and values are expanded from the latent by one up-projection', 'they are not stored');
    if (id === 'mla.cache') return I('Cache per token per layer', `(512 + 64) × 2 bytes = ${m.cache_bytes.mla_per_token_per_layer.toLocaleString('en-US')}`, `full multi-head keys and values: ${m.cache_bytes.mha_per_token_per_layer.toLocaleString('en-US')} (${g(m.cache_bytes.ratio, 3)}× larger)`); return null; });
  withInfo('walk-mla-absorb', (o, id, rc, data, d) => { const m = data.m; if (!m) return null; const tt = i => m.tokens[i].text.trim() || 'start';
    if (rc && id.startsWith('tensor.qabs.')) return I(`absorbed query[${rc[0]}][${rc[1]}] = ${g(m.query.q_absorbed_head0[rc[0]][rc[1]], 3)}`, 'q_nope · W_UK: the query already translated into latent space');
    if (rc && id.startsWith('tensor.clat.')) return I(`cached latent[${rc[0]}][${rc[1]}] = ${g(m.latent.c_kv[rc[0]][rc[1]], 3)}`, `token ${tt(rc[0])}`);
    if (rc && /^tensor\.(explicit|absorbed)\./.test(id)) { const [i, j] = rc; if (j > i) return I('Masked', 'a token cannot look at a later token'); const e = m.explicit.scores_head0[i][j], a = m.absorbed.scores_head0[i][j];
      return I(`${id.includes('explicit') ? 'Explicit' : 'Absorbed'} score ${tt(i)} → ${tt(j)} = ${g(id.includes('explicit') ? e : a, 4)}`, `explicit ${g(e, 5)}, absorbed ${g(a, 5)}, difference ${(Math.abs(e - a)).toExponential(1)}`); } return null; });
  withInfo('walk-cachesize', (o, id, rc, data, d) => { const ratio = { 'cache.mha': 1, 'cache.gqa': 2, 'cache.mqa': 16 }[id], bytes = { 'cache.mha': 8192, 'cache.gqa': 4096, 'cache.mqa': 512, 'cache.mla': 1152 }[id]; if (!bytes) return null;
    if (id === 'cache.mla') return I('MLA latent', '1,152 bytes per token per layer (DeepSeek-V2-Lite’s own sizes)', 'vs 10,240 for full multi-head keys and values at its sizes: 8.9× smaller');
    return I(id === 'cache.mha' ? 'Multi-head attention' : id === 'cache.gqa' ? 'Grouped-query attention' : 'Multi-query attention', `${bytes.toLocaleString('en-US')} B per token per layer`, id === 'cache.mha' ? '16 key/value heads: 2× GQA’s 4,096 bytes' : id === 'cache.gqa' ? 'Qwen3’s choice: 8 key/value heads, half of multi-head' : `${bytes === 512 ? 8 : ratio}× smaller than GQA`, `over 28 layers: ${(bytes * 28).toLocaleString('en-US')} B per token`); });
});
