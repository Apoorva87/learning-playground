/* What-if arithmetic for the design variants Article C compares. Pure functions of a model's sizes.
   Qwen3-0.6B-Base is the running example; DeepSeek-V2-Lite supplies the MLA numbers. None of the alternatives
   is a recorded run: every figure is computed from the config, and the tables say so. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopVariants = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const QWEN3 = { name: 'Qwen3-0.6B-Base', layers: 28, hidden: 1024, heads: 16, kvHeads: 8, headDim: 128, ffn: 3072, vocab: 151936, maxPos: 32768, tokens: 5, context: 32768, bytes: 2 };
  const DEEPSEEK_V2_LITE = { name: 'DeepSeek-V2-Lite', layers: 27, hidden: 2048, heads: 16, kvRank: 512, rope: 64, nope: 128, vHead: 128, bytes: 2 };

  /* Keys and values per layer: how many heads there are decides the cache, the projection weights and the bytes read per step. */
  function attention(kind, cfg) {
    if (kind === 'mla') {
      const d = cfg.mla || DEEPSEEK_V2_LITE, per = (d.kvRank + d.rope) * d.bytes, mha = d.heads * (d.nope + d.rope + d.vHead) * d.bytes;
      return { kind, kvHeads: null, cachePerTokenPerLayer: per, mhaEquivalentPerTokenPerLayer: mha, cachePerToken: per * d.layers, mhaEquivalentPerToken: mha * d.layers, cachePrompt: per * d.layers * cfg.tokens,
        kvProjParams: d.hidden * (d.kvRank + d.rope) + d.kvRank * d.heads * (d.nope + d.vHead), model: d.name, layers: d.layers };
    }
    const kv = kind === 'mha' ? cfg.heads : kind === 'mqa' ? 1 : cfg.kvHeads;
    const per = 2 * kv * cfg.headDim * cfg.bytes, qParams = cfg.hidden * cfg.heads * cfg.headDim, oParams = cfg.heads * cfg.headDim * cfg.hidden, kvParams = 2 * cfg.hidden * kv * cfg.headDim;
    const cacheAtContext = per * cfg.layers * cfg.context;
    return { kind, kvHeads: kv, cachePerTokenPerLayer: per, cachePerToken: per * cfg.layers, cachePrompt: per * cfg.layers * cfg.tokens, cacheAtContext,
      kvProjFlopsPrompt: 2 * cfg.tokens * kvParams * cfg.layers, kvProjFlopsDecode: 2 * kvParams * cfg.layers, kvProjParams: kvParams, attnParams: qParams + kvParams + oParams,
      decodeReadAtContext: cacheAtContext,                                                         // every cached key and value is read once per generated token
      decodeFlopsAtContext: 4 * cfg.heads * cfg.context * cfg.headDim * cfg.layers };               // scores plus weighted values, for all query heads
  }

  /* How the model learns where a token is. */
  function position(kind, cfg) {
    const base = { kind, params: 0 };
    if (kind === 'sinusoidal') return { ...base, opsPerToken: cfg.hidden, where: 'added to the embedding, once, at the input' };
    if (kind === 'learned') return { ...base, params: cfg.maxPos * cfg.hidden, opsPerToken: cfg.hidden, where: 'a learned vector per position, added to the embedding at the input' };
    if (kind === 'rope') return { ...base, opsPerToken: 0, opsPerTokenPerLayer: 3 * (cfg.heads + cfg.kvHeads) * cfg.headDim, where: 'queries and keys are rotated in every layer; nothing is added to the embedding' };
    if (kind === 'alibi') return { ...base, opsPerToken: 0, opsPerTokenPerLayer: cfg.heads * cfg.context, where: 'a distance-proportional penalty is added to every attention score, per head' };
    throw new Error('unknown position scheme ' + kind);
  }

  /* One normalisation of a vector of the model's width. Operation counts are our tally of the formulas in the beat. */
  function norm(kind, cfg) {
    const d = cfg.hidden, ops = kind === 'layernorm' ? 7 * d : 4 * d, params = kind === 'layernorm' ? 2 * d : d;
    return { kind, opsPerVector: ops, params, opsPerLayerForTokens: 2 * cfg.tokens * ops, paramsPerLayer: 2 * params,
      placement: kind === 'postnorm' ? 'after the residual add' : 'before each sublayer, on a copy' };
  }

  /* The feed-forward block. */
  function ffn(kind, cfg) {
    const d = cfg.hidden;
    if (kind === 'relu' || kind === 'gelu') { const w = 4 * d, p = 2 * d * w; return { kind, width: w, matrices: 2, params: p, flopsPerToken: 2 * p, activeParams: p }; }
    if (kind === 'swiglu' || kind === 'swiglu-matched') {
      const w = kind === 'swiglu' ? cfg.ffn : Math.floor(8 * d / 3), p = 3 * d * w;
      return { kind, width: w, matrices: 3, params: p, flopsPerToken: 2 * p, activeParams: p };
    }
    if (kind === 'moe') {
      const E = 8, k = 2, one = 3 * d * cfg.ffn;
      return { kind, experts: E, topk: k, width: cfg.ffn, matrices: 3, params: E * one, storedParams: E * one, activeParams: k * one, flopsPerToken: 2 * k * one + 2 * d * E };
    }
    throw new Error('unknown feed-forward ' + kind);
  }

  function embedding(kind, cfg) { const one = cfg.vocab * cfg.hidden; return { kind, params: kind === 'tied' ? one : 2 * one, tables: kind === 'tied' ? 1 : 2 }; }

  const NOTE = 'All figures computed from the config, including the design Qwen3 uses; nothing here is a new recorded run.';
  function table(family, cfg) {
    if (family === 'attention') {
      const kinds = ['mha', 'mqa', 'gqa'];
      const rows = kinds.map(k => { const a = attention(k, cfg); return { kind: k, label: { mha: 'MHA, 16 K/V heads', mqa: 'MQA, 1 K/V head', gqa: 'GQA, 8 K/V heads (Qwen3)' }[k],
        cells: { kvHeads: a.kvHeads, kvProjParams: a.kvProjParams, cachePerTokenPerLayer: a.cachePerTokenPerLayer, cachePrompt: a.cachePrompt, cacheAtContext: a.cacheAtContext, decodeReadAtContext: a.decodeReadAtContext } }; });
      const m = attention('mla', cfg);
      rows.push({ kind: 'mla', label: 'MLA latent (DeepSeek-V2-Lite)', cells: { kvHeads: null, kvProjParams: null, cachePerTokenPerLayer: m.cachePerTokenPerLayer, cachePrompt: null, cacheAtContext: null, decodeReadAtContext: null } });
      return { family, columns: ['kvHeads', 'kvProjParams', 'cachePerTokenPerLayer', 'cachePrompt', 'cacheAtContext', 'decodeReadAtContext'], rows, note: NOTE + ' MLA uses DeepSeek-V2-Lite’s own sizes, so only its bytes per token per layer are shown.' };
    }
    if (family === 'position') {
      const rows = ['sinusoidal', 'learned', 'rope', 'alibi'].map(k => { const p = position(k, cfg); return { kind: k, label: { sinusoidal: 'Sinusoidal, added at the input', learned: 'Learned table, added at the input', rope: 'RoPE, rotation in every layer (Qwen3)', alibi: 'ALiBi, distance penalty' }[k], cells: { params: p.params, opsPerToken: p.opsPerToken, opsPerTokenPerLayer: p.opsPerTokenPerLayer || 0 } }; });
      return { family, columns: ['params', 'opsPerToken', 'opsPerTokenPerLayer'], rows, note: NOTE };
    }
    if (family === 'norm') {
      const rows = ['layernorm', 'rmsnorm'].map(k => { const n = norm(k, cfg); return { kind: k, label: k === 'layernorm' ? 'LayerNorm' : 'RMSNorm', cells: { opsPerVector: n.opsPerVector, params: n.params, opsPerLayerForTokens: n.opsPerLayerForTokens } }; });
      return { family, columns: ['opsPerVector', 'params', 'opsPerLayerForTokens'], rows, note: NOTE };
    }
    if (family === 'ffn') {
      const rows = ['relu', 'swiglu-matched', 'swiglu', 'moe'].map(k => { const f = ffn(k, cfg); return { kind: k, label: { relu: 'ReLU or GELU, width 4d', 'swiglu-matched': 'SwiGLU, width 8d/3', swiglu: 'SwiGLU, width 3072 (Qwen3)', moe: 'MoE, 8 experts, 2 active' }[k],
        cells: { width: f.width, params: f.params, activeParams: f.activeParams, flopsPerToken: f.flopsPerToken } }; });
      return { family, columns: ['width', 'params', 'activeParams', 'flopsPerToken'], rows, note: NOTE };
    }
    if (family === 'embedding') {
      const rows = ['tied', 'untied'].map(k => ({ kind: k, label: { tied: 'Tied, one table (Qwen3)', untied: 'Untied, two tables' }[k], cells: { tables: embedding(k, cfg).tables, params: embedding(k, cfg).params } }));
      return { family, columns: ['tables', 'params'], rows, note: NOTE };
    }
    throw new Error('unknown family ' + family);
  }
  return { QWEN3, DEEPSEEK_V2_LITE, attention, position, norm, ffn, embedding, table };
});
