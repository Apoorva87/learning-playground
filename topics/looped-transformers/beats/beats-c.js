/* Beat data for Article C, "One sentence through a modern transformer": one beat per stage. */
window.LOOP_BEATS_C = (() => {
  const micro = (name, lit, extra) => ({ level: 'micro', micro: name, fold: 0, pass: 0, camera: LoopLayout.microCamera(name), lit, flow: { kind: 'none' }, ...extra });
  const beat = (id, title, scene, terms = {}, kind = 'math') => ({ id, title, kind, scene, terms });
  return [
    beat('map', 'The journey', micro('walk-layer', ['blk'])),
    beat('tokens', 'Tokens', micro('walk-tokens', ['tok', 'id', 'sentence'])),
    beat('embed', 'From IDs to vectors', micro('walk-embed', ['tensor', 'rn'])),
    beat('layer-overview', 'One layer, from above', micro('walk-layer', ['blk']), { residual: ['blk.skip1', 'blk.skip2'] }),
    beat('rmsnorm', 'RMSNorm, with real numbers', micro('walk-norm', ['tensor', 'stat'])),
    beat('qkv', 'Queries, keys and values', micro('walk-qkv', ['tensor']), { q: ['tensor.q'], k: ['tensor.k'], v: ['tensor.v'] }),
    beat('gqa', 'Grouped-query attention', micro('walk-gqa', ['qh', 'kvh', 'link', 'gqa']), { kvh: ['kvh'] }),
    beat('qknorm', 'Normalising queries and keys', micro('walk-qknorm', ['tensor', 'inv'])),
    beat('rope', 'Rotary positions', micro('walk-rope', ['tensor', 'freq', 'rope'])),
    beat('scores', 'Scores, the mask and softmax', micro('walk-scores', ['tensor', 'rowname', 'colname']), { mask: ['tensor.raw'] }),
    beat('mix', 'Mixing values, merging heads', micro('walk-mix', ['tensor'])),
    beat('ffn', 'The feed-forward network', micro('walk-ffn', ['tensor']), { gate: ['tensor.gate'], up: ['tensor.up'], down: ['tensor.down'] }),
    beat('stack', 'Twenty-eight layers', micro('walk-stack', ['layer', 'stack']), { lens: ['layer'] }),
    beat('head', 'The last vector becomes 151,936 scores', micro('walk-head', ['tensor', 'bar', 'head'])),
    beat('token-pick', 'Picking a token', micro('walk-head', ['bar.0', 'bar.1', 'bar.2', 'head'])),
    beat('prefill-cost', 'Prefill by the numbers', micro('walk-layer', ['blk'])),
    beat('kv-cache', 'The KV cache', micro('walk-cache', ['tensor', 'cache'])),
    beat('decode', 'Decode: one new token', micro('walk-decode', ['tensor', 'newrow', 'dnext', 'decode'])),
    beat('decode-cost', 'Why decode is slow', micro('walk-decode', ['tensor', 'dnext'])),
    beat('gqa-vs-mha', 'What GQA buys', micro('walk-cachesize', ['cache'])),
    beat('mla', 'Multi-head latent attention', micro('walk-mla', ['tensor', 'mla'])),
    beat('mla-absorb', 'Never rebuilding the keys', micro('walk-mla-absorb', ['tensor'])),
    beat('other-features', 'Other new features', micro('walk-layer', ['blk']), {}, 'interlude'),
    beat('recap', 'The whole picture', micro('walk-layer', ['blk'])),
  ];
})();
