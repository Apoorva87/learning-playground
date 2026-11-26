/* Beat data for Article B, "Inside Inkling 2 Mini": one beat per component. */
window.LOOP_BEATS_B = (() => {
  const cam = (yaw, pitch, dist, tx = 0, ty = 0, tz = 0) => ({ yaw, pitch, dist, tx, ty, tz });
  const mcam = (fold, yaw, pitch) => { const c = LoopLayout.fitCamera(LoopLayout.macroObjects(fold), 1.3, { yaw, pitch }); return cam(yaw, pitch, +(c.dist * 1.08).toFixed(2), +c.tx.toFixed(2), +c.ty.toFixed(2), +c.tz.toFixed(2)); };
  const folded = mcam(0, -0.5, 0.25), half = mcam(0.5, -0.4, 0.28), unrolled = mcam(1, -0.3, 0.3);
  const macro = (extra) => ({ level: 'macro', fold: 0, pass: 0, camera: folded, lit: [], flow: { kind: 'none' }, ...extra });
  const micro = (name, extra) => ({ level: 'micro', micro: name, fold: 0, pass: 0, camera: LoopLayout.microCamera(name), lit: ['tensor'], flow: { kind: 'none' }, ...extra });
  return [
    { id: 'map', title: 'The map', kind: 'math', scene: macro({ lit: ['full'] }), terms: { full: ['full'], core: ['core'] } },
    { id: 'embed', title: 'Tokens to vectors', kind: 'math', scene: micro('embed'), terms: {} },
    { id: 'attn-res', title: 'Attention residuals', kind: 'math', scene: micro('depth-attn'), terms: { blocks: ['tensor.block'] } },
    { id: 'sliding-layer', title: 'A sliding-window layer', kind: 'math', scene: micro('sliding-layer'),
      terms: { q: ['tensor.q'], k: ['tensor.k'], v: ['tensor.v'], rel: ['tensor.rel'] } },
    { id: 'full-layer', title: 'A full-attention layer', kind: 'math', scene: micro('full-layer'), terms: {} },
    { id: 'moe', title: 'The mixture of experts', kind: 'math', scene: micro('moe', { lit: ['tensor', 'expert'] }), terms: { router: ['tensor.router'], expert: ['expert'] } },
    { id: 'prelude', title: 'The prelude', kind: 'math', scene: macro({ lit: ['prelude'], flow: { kind: 'token' } }), terms: { prelude: ['prelude'] } },
    { id: 'loop-full', title: 'The loop, in full', kind: 'math', scene: macro({ fold: 0.5, camera: half, lit: ['pass.2', 'pass.3', 'hop.return'] }), terms: {} },
    { id: 'loop-messages', title: 'Messages', kind: 'math', scene: micro('message-input'),
      terms: { sources: ['tensor.src'], queries: ['tensor.query'], messages: ['tensor.msg'] } },
    { id: 'loop-gates', title: 'The four gates', kind: 'math', scene: micro('message-gate'),
      terms: { gate: ['tensor.f', 'tensor.g'], forget: ['tensor.f'], mgate: ['tensor.g'] } },
    { id: 'loop-recorded', title: 'In the recorded run', kind: 'interlude', scene: macro({ fold: 1, camera: unrolled, lit: ['pass.2', 'pass.3'] }), terms: {} },
    { id: 'coda-head', title: 'The coda and the head', kind: 'math', scene: micro('head'), terms: {} },
    { id: 'bytes', title: 'Memory and bytes', kind: 'math', scene: macro({ fold: 1, camera: unrolled, lit: ['full'] }), terms: {} },
    { id: 'drafter', title: 'The drafter', kind: 'interlude', scene: macro({ fold: 1, camera: unrolled, lit: ['V5', 'V23', 'V41'] }), terms: {} },
    { id: 'vision', title: 'One sentence on vision', kind: 'interlude', scene: macro({ lit: [] }), terms: {} },
  ];
})();
