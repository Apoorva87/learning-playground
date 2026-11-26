/* The live toy model on the page: a puzzle, a loop-count control, and the numbers behind them.
   It runs toy/toy.js (a float64 port of the PyTorch model) in the browser and feeds the scene's
   toy-hops and toy-gate micro scenes. window.LOOP.toy holds its state for the tests. */
(function () {
  'use strict';
  const NODES = 'ABCDEFGH', ORD = ['', '1st', '2nd', '3rd', '4th', '5th', '6th'];
  const pct = v => Math.round(v * 100) + '%';

  async function bind({ scene, root }) {
    const el = root.querySelector('#toy');
    if (!el) return null;
    const get = n => fetch(`assets/toy/${n}.json`).then(r => r.json());
    const [weights, acc, ev] = await Promise.all([get('toy-weights'), get('toy-accuracy'), get('toy-eval')]);
    const model = window.Toy.load(weights);
    const T = window.LOOP.toy = { ready: false, passes: 3, variant: el.querySelector('#toy-variant').value || 'inkling', seq: 0, tokens: null, argmax: [], gate: null, gatePass: 2 };

    function hops(tokens) {
      const f = {}; for (let i = 0; i < 8; i++) f[tokens[2 * i]] = tokens[2 * i + 1];
      const out = []; let c = tokens[16]; for (let k = 0; k < 6; k++) out.push(c = f[c]);
      return out;
    }

    function update() {
      const tokens = ev.tokens[T.seq], run = window.Toy.run(model, T.variant, tokens, 6), k = T.passes, truth = hops(tokens);
      T.tokens = tokens; T.argmax = run.argmax; T.truth = truth;
      T.gatePass = Math.max(k, 2);                                        // the gate only exists from the second pass
      const g = T.variant === 'inkling' ? run.trace[T.gatePass - 1].gate : null;
      T.gate = g ? { h: g.h[0], m: g.m[0], f: g.f[0], g: g.g[0], out: g.out[0] } : null;
      scene.setData('toy-hops', { passes: k, tokens, attn: run.trace.slice(0, k).map(t => t.attn[0]), argmax: run.argmax.slice(0, k) });
      scene.setData('toy-gate', { gate: T.gate });

      const pairs = Array.from({ length: 8 }, (_, i) => `${NODES[tokens[2 * i]]}→${NODES[tokens[2 * i + 1]]}`).join('  ');
      const said = NODES[run.argmax[k - 1]], want = NODES[truth[k - 1]];
      const verdict = said === want ? `the ${ORD[k]} hop` : `not the ${ORD[k]} hop, which is ${want}`;
      el.querySelector('#toy-readout').textContent =
        `Arrows: ${pairs}. Start at ${NODES[tokens[16]]}. Following them: ${NODES[tokens[16]]} → ${truth.slice(0, Math.max(4, k)).map(n => NODES[n]).join(' → ')}. ` +
        `After ${k} pass${k > 1 ? 'es' : ''} the ${T.variant === 'inkling' ? 'restart-and-message' : 'carry-the-state'} model reads out ${said}, ${verdict}.`;
      const gp = document.getElementById('toy-gate-pass'); if (gp) gp.textContent = T.variant === 'inkling' ? `pass ${T.gatePass}` : 'no gate in this variant';
      el.querySelectorAll('[data-toy-passes]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.toyPasses === k)));
      renderTable();
    }

    function renderTable() {
      const a = acc[T.variant], m = a.hop_matrix, trained = a.trained_passes;
      el.querySelector('#toy-table').innerHTML =
        `<table><caption>Share of 2,000 held-out puzzles where the readout after <i>p</i> passes is the <i>k</i>-th hop. Passes ${a.extrapolation_passes.join(' and ')} were never trained on.</caption>` +
        `<thead><tr><th scope="col">after <i>p</i> passes</th>${m[0].map((_, k) => `<th scope="col">hop ${k + 1}</th>`).join('')}</tr></thead><tbody>` +
        m.map((row, p) => `<tr class="${p + 1 > trained ? 'extra' : ''}"><th scope="row">${p + 1}${p + 1 > trained ? ' (never trained)' : ''}</th>${row.map((v, k) => `<td class="${p === k ? 'diag' : ''}">${pct(v)}</td>`).join('')}</tr>`).join('') +
        '</tbody></table>';
    }

    el.querySelectorAll('[data-toy-passes]').forEach(b => b.addEventListener('click', () => { T.passes = +b.dataset.toyPasses; update(); }));
    el.querySelector('#toy-variant').addEventListener('change', e => { T.variant = e.target.value; update(); });
    el.querySelector('#toy-new').addEventListener('click', () => { T.seq = (T.seq + 7) % ev.tokens.length; update(); });
    update();
    T.ready = true;
    return T;
  }
  window.LoopToy = { bind };
})();
