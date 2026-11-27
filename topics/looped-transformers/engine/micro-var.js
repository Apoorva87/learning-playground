/* Scenes for the "how it changed" part of Article C: one drawing per design in a family (attention heads, position,
   normalisation, feed-forward, embeddings). The numbers come from walkthrough/variants.js, which computes them from the
   config. None is a recorded run of that design, and every scene says so. data = { variant: { attention, position, norm, ffn, embedding } }. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) factory(require('./layout.js'), require('../walkthrough/variants.js'));
  else factory(root.LoopLayout, root.LoopVariants);
})(typeof self !== 'undefined' ? self : this, function (Layout, V) {
  'use strict';
  const { arrow, vline, box } = Layout.microKit, MICRO = Layout.MICRO, cfg = V.QWEN3;
  const n = x => x.toLocaleString('en-US');
  const gib = b => (b / 2 ** 30).toFixed(1) + ' GiB';
  const I = (title, ...lines) => ({ title, lines: lines.flat().filter(Boolean) });
  const B = (id, label, x, y, w, h, tint, info, alpha = 0.9) => ({ ...box(id, label, x, y, w, h, tint, alpha), info });
  const cell = (id, x, y, w, h, tint, label, info, alpha = 0.9) => ({ id, kind: 'cell', x, y, z: 0, w, h, d: 0.3, tint, pass: 0, full: false, alpha, label, info });
  const chosen = (data, fam, dflt) => (data && data.variant && data.variant[fam]) || dflt;
  /* Which designs each scene can draw, so tests and checkers can visit all of them. */
  Layout.VARIANT_SCENES = { 'walk-var-attention': ['attention', ['mha', 'mqa', 'gqa', 'mla']], 'walk-var-position': ['position', ['sinusoidal', 'learned', 'rope', 'alibi']], 'walk-var-norm': ['norm', ['post-layernorm', 'pre-layernorm', 'pre-rmsnorm']], 'walk-var-ffn': ['ffn', ['relu', 'swiglu', 'moe']], 'walk-var-embed': ['embedding', ['tied', 'untied']] };
  const NOTE = 'Computed from the config, not recorded';
  const note = y => B('note', NOTE, 0, y, 22, 0.7, 'neutral', I('Computed, not recorded', 'All figures are computed from the config (Qwen3-0.6B-Base sizes); none is a recorded run of this design.', 'The recorded run is the Qwen3 design: GQA, RoPE, RMSNorm, SwiGLU, tied embeddings.'), 0.1);

  /* How many key/value heads, and what that does to the cache. */
  MICRO['walk-var-attention'] = data => {
    const kind = chosen(data, 'attention', 'gqa'), a = V.attention(kind, cfg), mhaB = V.attention('mha', cfg).cachePerTokenPerLayer, out = [];
    const xs = i => (i - 7.5) * 1.0, per = a.cachePerTokenPerLayer, W = 15.6;
    const kvOf = h => (kind === 'mha' ? h : kind === 'mqa' ? 0 : kind === 'gqa' ? h >> 1 : null);
    const kvText = { mha: '16 key/value heads: one per query head', mqa: '1 key/value head shared by all 16 query heads', gqa: '8 key/value heads: 2 query heads share each', mla: '1 latent per token (512 + 64 numbers): keys and values are rebuilt from it' }[kind];
    out.push(B('cap.q', '16 query heads', 0, 4.4, W, 0.6, 'neutral', I('Query heads', 'every design keeps 16 query heads; the designs differ in how many keys and values they store'), 0.1));
    for (let i = 0; i < 16; i++) {
      const j = kvOf(i);
      out.push(cell(`qh.${i}`, xs(i), 3, 0.8, 0.8, 'core', undefined, I(`Query head ${i}`, j === null ? 'reads the shared latent: keys and values are rebuilt from it' : `reads key/value head ${j}`, j !== null && kind !== 'mha' ? `shared with ${kind === 'mqa' ? 'all other query heads' : 'query head ' + (i ^ 1)}` : '', 'its own 128 query numbers per token')));
      out.push(vline(`link.${i}`, xs(i), 2.5, 0.5));
    }
    out.push(B('cap.kv', kvText, 0, 1.45, W, 0.6, 'neutral', I('Key/value heads', kvText, `cache per token per layer: ${n(per)} B`), 0.1));
    const kvInfo = k => I(kind === 'mla' ? 'The latent' : `Key/value head ${k}`, kind === 'mla' ? '(512 + 64) numbers × 2 bytes = 1,152 B per token per layer (DeepSeek-V2-Lite sizes)' : `2 × 128 numbers × 2 bytes = 512 B per token per layer`, kind === 'mqa' ? 'serves all 16 query heads' : kind === 'gqa' ? `serves query heads ${2 * k} and ${2 * k + 1}` : kind === 'mha' ? `serves query head ${k}` : 'one up-projection rebuilds all keys and values from it');
    if (kind === 'mha') for (let i = 0; i < 16; i++) out.push(cell(`kvh.${i}`, xs(i), 0, 0.8, 0.8, 'coda', undefined, kvInfo(i)));
    else if (kind === 'gqa') for (let k = 0; k < 8; k++) out.push(cell(`kvh.${k}`, (xs(2 * k) + xs(2 * k + 1)) / 2, 0, 1.8, 0.8, 'coda', undefined, kvInfo(k)));
    else out.push(cell('kvh.0', 0, 0, W, 0.8, kind === 'mla' ? 'prelude' : 'coda', kind === 'mla' ? 'latent: 512 + 64 numbers' : undefined, kvInfo(0)));
    const frac = kind === 'mla' ? per / a.mhaEquivalentPerTokenPerLayer : per / mhaB, ctxBytes = kind === 'mla' ? a.cachePerToken * cfg.context : a.cacheAtContext, wBar = Math.max(0.3, W * frac);
    const bar = (id, y, label, info) => cell(id, -W / 2 + wBar / 2, y, wBar, 0.7, kind === 'mla' ? 'prelude' : 'hot', label, info);
    out.push(bar('bar.token', -1.4, `${n(per)} B per token per layer`, I('Cache per token per layer', `${n(per)} B = ${kind === 'mla' ? '(512 + 64) × 2' : `2 × ${a.kvHeads} heads × 128 × 2`} bytes`, `${(frac * 100).toFixed(1)}% of its own multi-head size${kind === 'mla' ? ' (full multi-head at its sizes: 10,240 B)' : ''}`, `the 5-token prompt stores ${n(a.cachePrompt)} B over its layers`)));
    out.push(bar('bar.ctx', -2.9, `${gib(ctxBytes)} at 32,768 tokens`, I('Cache at a 32,768-token context', `${n(ctxBytes)} B = ${n(per)} × ${kind === 'mla' ? 27 : 28} layers × 32,768 tokens`, 'a decode step reads all of it once', kind === 'mla' ? 'DeepSeek-V2-Lite’s own layer count and sizes' : 'Qwen3-0.6B-Base sizes')));
    out.push(note(-4.9));
    return out;
  };

  /* Where the model learns the order of the tokens. */
  MICRO['walk-var-position'] = data => {
    const kind = chosen(data, 'position', 'rope'), p = V.position(kind, cfg), out = [];
    out.push(B('emb', 'token embeddings  [5, 1024]', -9, 0, 5.4, 1.2, 'prelude', I('Token embeddings', 'the same rows in every design; what differs is how position is added')));
    out.push(arrow('flow.in', -6.2, -2.4, 0));
    if (kind === 'sinusoidal' || kind === 'learned') {
      const lab = kind === 'learned' ? `learned table  [32,768, 1024] = ${n(p.params)} parameters` : 'fixed sin/cos vector for each position: 0 parameters';
      out.push(B('pos.vec', lab, -9, 3.2, 7.4, 1.2, 'hot', I(kind === 'learned' ? 'Learned position table' : 'Sinusoidal positions', p.where, `${n(p.params)} parameters`)));
      out.push(vline('flow.pos', -9, 2.5, 0.8));
      out.push(B('pos.add', '+', -2.4, 0, 1.0, 1.0, 'hot', I('Added once, at the input', 'x = embedding + position', 'after this the layers see only the sum')));
      out.push(arrow('flow.sum', -1.7, 0.8, 0));
      out.push(B('layers', '28 layers: position reaches them only through the sum', 5.8, 0, 9.6, 1.4, 'core', I('The layer stack', 'position was added once, before layer 1')));
    } else {
      out.push(B('layers', kind === 'rope' ? '28 layers' : '28 layers', 3, 0, 14, 1.4, 'core', I('The layer stack', kind === 'rope' ? 'nothing was added to the embeddings' : 'embeddings enter unchanged')));
      out.push(B(kind === 'rope' ? 'pos.rot' : 'pos.bias', kind === 'rope' ? 'in every layer: q and k are rotated by angle = position × frequency (0 parameters)' : 'in every layer: a penalty slope × distance is subtracted from each attention score (0 parameters)', 3, 3.2, 12, 1.2, 'hot',
        I(kind === 'rope' ? 'Rotary positions (RoPE)' : 'Linear biases (ALiBi)', p.where, kind === 'rope' ? `${n(p.opsPerTokenPerLayer)} operations per token per layer (our tally: 3 per number rotated)` : `${n(p.opsPerTokenPerLayer)} bias additions per layer at a 32,768-token context (our tally)`)));
      out.push(vline('flow.pos', 3, 2.5, 0.8));
    }
    out.push(B('where', { sinusoidal: 'enters once, at the input', learned: 'enters once, at the input', rope: 'enters inside attention, every layer', alibi: 'enters inside attention, every layer' }[kind], 0, -2.6, 20, 0.8, 'neutral', I('Where position enters', p.where), 0.12));
    out.push(note(-4.6));
    return out;
  };

  /* Where the norm sits, and which norm. */
  MICRO['walk-var-norm'] = data => {
    const v = chosen(data, 'norm', 'pre-rmsnorm'), post = v.startsWith('post'), kindN = v.endsWith('rmsnorm') ? 'rmsnorm' : 'layernorm', nr = V.norm(kindN, cfg), out = [];
    const nLabel = kindN === 'layernorm' ? `LayerNorm: (x − mean) / std × γ + β   ${n(nr.opsPerVector)} ops per vector, ${n(nr.params)} parameters per norm` : `RMSNorm: x / rms × g   ${n(nr.opsPerVector)} ops per vector, ${n(nr.params)} parameters per norm`;
    const nInfo = I(kindN === 'layernorm' ? 'LayerNorm' : 'RMSNorm', nLabel, post ? 'sits on the residual path, after the add' : 'works on a copy; the residual path is not touched', `5 tokens × 2 norms per layer: ${n(nr.opsPerLayerForTokens)} operations (our tally)`);
    const ys = post ? { x: 6, sub: 3.6, add: 1.2, nrm: -1.2, out: -3.6 } : { x: 6, nrm: 3.6, sub: 1.2, add: -1.2, out: -3.6 };
    out.push(B('x', 'x  [5, 1024]', 0, ys.x, 6, 0.9, 'tensor', I('Layer input', 'one row of 1024 numbers per token')));
    out.push(B('sub', 'attention or feed-forward sublayer', 0, ys.sub, 8, 1.1, 'core', I('The sublayer', 'attention, or the feed-forward network')));
    out.push(B('add', '+ residual', 0, ys.add, 6, 0.9, 'hot', I('Residual add', 'x + sublayer(·): the original is kept and a correction added')));
    out.push(B('nrm', nLabel, 0, ys.nrm, 12, 1.1, post ? 'hot' : 'prelude', nInfo));
    out.push(B('out', 'output  [5, 1024]', 0, ys.out, 6, 0.9, 'tensor', I('Layer output', 'same shape as the input: one row of 1024 numbers per token')));
    const order = post ? ['x', 'sub', 'add', 'nrm', 'out'] : ['x', 'nrm', 'sub', 'add', 'out'];
    for (let i = 0; i < 4; i++) out.push(vline(`flow.${i}`, 0, ys[order[i]] - 0.6, ys[order[i + 1]] + 0.6));
    out.push(vline('skip', 5.2, ys.x, ys.add, post ? 'hot' : 'neutral'));
    const why = post ? 'norm on the main path: large gradients near the output at initialisation, so training needs a learning-rate warm-up (Xiong et al. 2020)' : 'the main path is untouched: gradients are well-behaved at initialisation and the warm-up may be removable (Xiong et al. 2020)';
    out.push(B('why', post ? 'norm on the main path' : 'main path untouched', 0, -5.4, 20, 0.8, 'neutral', I(post ? 'Post-LN' : 'Pre-LN', why), 0.12));
    out.push(note(-7.0));
    return out;
  };

  /* The feed-forward block: two matrices, three, or many experts. */
  MICRO['walk-var-ffn'] = data => {
    const kind = chosen(data, 'ffn', 'swiglu'), f = V.ffn(kind, cfg), out = [];
    out.push(B('in', 'x  [5, 1024]', -9, 0, 3.6, 1, 'tensor', I('Input', 'one row of 1024 numbers per token')));
    out.push(arrow('flow.in', -7.0, -5.2, 0));
    if (kind === 'relu') {
      out.push(B('w1', 'W₁  [1024, 4096]', -3.4, 0, 3.6, 2.2, 'core', I('Up-projection', `1024 × 4096 = ${n(1024 * 4096)} weights`)), arrow('flow.a', -1.4, 0.2, 0));
      out.push(B('act', 'ReLU or GELU', 1.2, 0, 2.6, 1.0, 'hot', I('Activation', 'one number in, one out, no new weights')), arrow('flow.b', 2.6, 4.2, 0));
      out.push(B('w2', 'W₂  [4096, 1024]', 6.0, 0, 3.6, 2.2, 'core', I('Down-projection', `4096 × 1024 = ${n(1024 * 4096)} weights`)));
    } else if (kind === 'swiglu') {
      out.push(B('wg', 'W_gate  [1024, 3072]', -3.4, 1.6, 3.8, 1.5, 'core', I('Gate projection', `1024 × 3072 = ${n(1024 * 3072)} weights`)), B('wu', 'W_up  [1024, 3072]', -3.4, -1.6, 3.8, 1.5, 'core', I('Up projection', `1024 × 3072 = ${n(1024 * 3072)} weights`)));
      out.push(B('act', 'SiLU', 0.8, 1.6, 1.6, 0.9, 'hot', I('SiLU on the gate', 'gate / (1 + e^−gate)')), B('mul', '×', 3.0, 0, 1.0, 1.0, 'hot', I('Element-wise product', 'SiLU(gate) × up: the gate decides how much of each feature passes')));
      out.push(B('w2', 'W_down  [3072, 1024]', 6.6, 0, 3.8, 2.2, 'core', I('Down projection', `3072 × 1024 = ${n(1024 * 3072)} weights`)));
      out.push(arrow('flow.a', -1.4, -0.2, 0), arrow('flow.b', 4.2, 4.5, 0));
    } else {
      out.push(B('router', 'router', -4, 0, 1.8, 1.2, 'hot', I('Router', 'a small matrix scores the 8 experts for each token', 'the top 2 are used')));
      for (let e = 0; e < 8; e++) { const on = e === 1 || e === 5; out.push(B(`expert.${e}`, on ? '' : '', 0.2 + e * 1.6, 0, 1.3, 2.2, on ? 'core' : 'neutral', I(`Expert ${e}`, on ? 'active for this token (illustrative choice)' : 'not chosen for this token: its weights are stored but not used', `one SwiGLU block, ${n(3 * 1024 * 3072)} weights`), on ? 0.95 : 0.25)); }
    }
    const title = { relu: 'ReLU or GELU feed-forward: two matrices, width 4 × 1024', swiglu: 'SwiGLU feed-forward: three matrices, width 3072', moe: 'Mixture of experts: eight SwiGLU experts, two used per token' }[kind];
    out.push(B('title', title, 0, 4.0, 20, 0.8, 'neutral', I(title.split(':')[0], title), 0.12));
    const params = kind === 'moe' ? `stored ${n(f.storedParams)} parameters, active per token ${n(f.activeParams)} (8 experts, 2 used)` : `${n(f.params)} parameters per layer`;
    out.push(B('params', params, 0, -3.4, 20, 0.9, 'neutral', I('Parameters', params, `about ${(f.flopsPerToken / 1e6).toFixed(1)} million operations per token`), 0.12));
    out.push(note(-5.2));
    return out;
  };

  /* One embedding table, or two. */
  MICRO['walk-var-embed'] = data => {
    const kind = chosen(data, 'embedding', 'tied'), e = V.embedding(kind, cfg), one = V.embedding('tied', cfg).params, out = [];
    out.push(B('ids', 'token IDs in', -10, 1.5, 3.6, 1, 'tensor', I('Token IDs', 'rows of the input table are read by ID')));
    out.push(B('logits', 'scores for 151,936 tokens out', 10, 1.5, 4.4, 1, 'tensor', I('Scores out', 'one score per vocabulary entry')));
    if (kind === 'tied') {
      out.push(B('table.e', `the same table E  [151,936, 1024]: ${n(one)} parameters, counted once`, 0, 1.5, 10, 2.4, 'prelude', I('One shared table', 'reads token IDs in, and its transpose scores the output', `${n(one)} parameters, once`)));
    } else {
      out.push(B('table.in', `E_in  [151,936, 1024]: ${n(one)} parameters`, -2.6, 1.5, 6.2, 2.4, 'prelude', I('Input table', `${n(one)} parameters`)));
      out.push(B('table.out', `W_out  [1024, 151,936]: ${n(one)} parameters`, 4.0, 1.5, 6.2, 2.4, 'coda', I('Output matrix', `${n(one)} parameters`, 'trained separately from the input table')));
    }
    out.push(arrow('flow.in', -8.0, -5.6, 1.5), arrow('flow.out', 6.8, 7.6, 1.5));
    out.push(B('total', `${n(e.params)} parameters in ${e.tables === 1 ? 'one table' : 'two tables'}`, 0, -1.4, 20, 0.9, 'neutral', I('Total', `${n(e.params)} parameters`, kind === 'tied' ? 'Qwen3-0.6B-Base ties them' : `${n(one)} more than tied`), 0.12));
    out.push(note(-3.0));
    return out;
  };
});
