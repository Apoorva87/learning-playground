/* Beat data for Article A, "Why loop a transformer?". The prose lives in why-loop.html; this file says
   what the scene shows at each beat and which words in the text name which scene objects. */
window.LOOP_BEATS_A = (() => {
  const cam = (yaw, pitch, dist, tx = 0, ty = 0, tz = 0) => ({ yaw, pitch, dist, tx, ty, tz });
  /* Cameras are fitted to the layout, so a beat never frames its objects by guesswork. */
  const mcam = (fold, yaw, pitch) => { const c = LoopLayout.fitCamera(LoopLayout.macroObjects(fold), 1.3, { yaw, pitch }); return cam(yaw, pitch, +(c.dist * 1.08).toFixed(2), +c.tx.toFixed(2), +c.ty.toFixed(2), +c.tz.toFixed(2)); };
  const folded = mcam(0, -0.5, 0.25), side = mcam(0, -1.05, 0.2), half = mcam(0.5, -0.4, 0.28), unrolled = mcam(1, -0.3, 0.3);
  const macro = (extra) => ({ level: 'macro', fold: 0, pass: 0, camera: folded, lit: [], flow: { kind: 'none' }, ...extra });
  const micro = (name, extra, data) => ({ level: 'micro', micro: name, fold: 0, pass: 0, camera: LoopLayout.microCamera(name, 1.08, data), lit: [], flow: { kind: 'none' }, ...extra });
  return [
    { id: 'hook', title: 'Think before you write', kind: 'interlude',
      scene: macro({ lit: ['prelude', 'core', 'coda'], flow: { kind: 'token' } }),
      terms: { token: ['prelude', 'core', 'coda'], core: ['core'], visits: ['prelude', 'core', 'coda'] } },
    { id: 'serial-depth', title: 'Depth is a chain', kind: 'math',
      scene: macro({ lit: ['prelude', 'core', 'coda'], flow: { kind: 'token' } }),
      terms: { layer: ['V8'], chain: ['prelude', 'core', 'coda'] } },
    { id: 'two-depths', title: 'Two ways to add depth', kind: 'math',
      scene: macro({ lit: ['core', 'hop.return'] }),
      terms: { core: ['core'], loopback: ['hop.return'], tied: ['pass.2', 'pass.3'] } },
    { id: 'lineage', title: 'An old idea, rediscovered', kind: 'interlude',
      scene: macro({ camera: side, lit: ['core', 'hop.return'] }), terms: {} },
    { id: 'prelude-core-coda', title: 'Prelude, core, coda', kind: 'math',
      scene: macro({ lit: ['prelude', 'core', 'coda'], flow: { kind: 'token' } }),
      terms: { prelude: ['prelude'], core: ['core'], coda: ['coda'] } },
    { id: 'toy-hops', title: 'One hop per pass', kind: 'math',
      scene: micro('toy-hops', { lit: ['pair', 'ptr', 'tensor.attn'] }, { passes: 6 }),
      terms: { pairs: ['pair'], readout: ['ptr'], heatmap: ['tensor.attn'] } },
    { id: 'carry-state', title: 'Way one: carry the state', kind: 'math',
      scene: micro('two-ways', { lit: ['carry'] }),
      terms: {} },
    { id: 'restart-message', title: 'Way two: restart, and leave a note', kind: 'math',
      scene: micro('two-ways', { lit: ['inkling'] }),
      terms: { message: ['inkling.msg', 'inkling.gate'] } },
    { id: 'gate-math', title: 'The gate, with real numbers', kind: 'math',
      scene: micro('toy-gate', { lit: ['tensor.h', 'tensor.m', 'tensor.f', 'tensor.g', 'tensor.out'] }),
      terms: { forget: ['tensor.f'], mgate: ['tensor.g'] } },
    { id: 'stability', title: 'Why loops can blow up', kind: 'math',
      scene: micro('stability', { lit: ['bar.stable', 'bar.unstable', 'bar.axis'] }),
      terms: {} },
    { id: 'costs-stored', title: 'What a loop costs: storage', kind: 'math',
      scene: macro({ fold: 0, lit: ['pass.1', 'pass.2', 'pass.3'] }), terms: { passes: ['pass.2', 'pass.3'] } },
    { id: 'costs-compute', title: 'What a loop costs: compute', kind: 'math',
      scene: macro({ fold: 0.5, camera: half, lit: ['core'] }), terms: { row: ['core'] } },
    { id: 'costs-bandwidth', title: 'What a loop costs: weights read', kind: 'math',
      scene: macro({ fold: 1, camera: unrolled, lit: ['pass.1', 'pass.2', 'pass.3'] }), terms: {} },
    { id: 'costs-kv', title: 'What a loop costs: the cache', kind: 'math',
      scene: macro({ fold: 1, camera: unrolled, lit: ['full'] }), terms: {} },
    { id: 'costs-latency', title: 'What a loop costs: waiting', kind: 'math',
      scene: macro({ fold: 1, camera: unrolled, lit: ['prelude', 'core', 'coda'], flow: { kind: 'token' } }), terms: { token: ['prelude', 'core', 'coda'] } },
    { id: 'benefits', title: 'What the bill buys', kind: 'math',
      scene: macro({ fold: 1, camera: unrolled, lit: ['pass.3'] }), terms: {} },
    { id: 'monitorability', title: 'Think more, write less', kind: 'interlude',
      scene: macro({ lit: ['core', 'hop.return'] }), terms: {} },
    { id: 'bridge', title: 'Now open the model', kind: 'interlude',
      scene: macro({ lit: ['prelude', 'core', 'coda'] }), terms: {} },
  ];
})();
