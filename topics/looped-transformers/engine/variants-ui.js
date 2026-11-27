/* The "how it changed" switches and tables. A button group [data-variants="family"] chooses a design; the scene redraws it
   and the table [data-variant-table="family"] highlights its row. Every figure comes from walkthrough/variants.js, computed from the config. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopVariantsUI = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const COLS = {
    kvHeads: 'key/value heads', kvProjParams: 'K and V weights per layer', cachePerTokenPerLayer: 'cache bytes per token per layer', cachePrompt: 'cache for the 5-token prompt (bytes)',
    cacheAtContext: 'cache at 32,768 tokens', decodeReadAtContext: 'cache read per decode step at 32,768', params: 'parameters', opsPerToken: 'operations per token (at the input)', opsPerTokenPerLayer: 'operations per layer (per decode step for ALiBi)',
    opsPerVector: 'operations per vector', opsPerLayerForTokens: 'operations per layer, 5 tokens', width: 'hidden width', activeParams: 'active parameters per token', flopsPerToken: 'operations per token', tables: 'tables',
  };
  const BYTES = new Set(['cacheAtContext', 'decodeReadAtContext']);
  const fmt = (col, v) => {
    if (v === null || v === undefined) return col === 'kvHeads' ? 'a latent' : '—';
    const s = v.toLocaleString('en-US');
    return BYTES.has(col) && v >= 2 ** 30 ? `${s} (${(v / 2 ** 30).toFixed(1)} GiB)` : s;
  };
  /* The table row that matches a chosen design. */
  const rowOf = (family, variant) => (family === 'norm' ? variant.replace(/^(post|pre)-/, '') : variant);

  function renderTable(el, table, selected) {
    el.innerHTML = '';
    const wrap = document.createElement('div'), t = document.createElement('table'), cap = document.createElement('caption');
    wrap.className = 'vtable-wrap'; cap.textContent = 'What it does to our model (computed from the config)';
    t.append(cap);
    const head = t.createTHead().insertRow(); head.append(Object.assign(document.createElement('th'), { scope: 'col', textContent: 'design' }));
    for (const c of table.columns) head.append(Object.assign(document.createElement('th'), { scope: 'col', textContent: COLS[c] || c }));
    const body = t.createTBody();
    for (const r of table.rows) {
      const tr = body.insertRow(); tr.dataset.kind = r.kind;
      tr.append(Object.assign(document.createElement('th'), { scope: 'row', textContent: r.label }));
      for (const c of table.columns) tr.insertCell().textContent = fmt(c, r.cells[c]);
      if (r.kind === selected) { tr.classList.add('picked'); tr.setAttribute('aria-current', 'true'); }
    }
    wrap.append(t);
    const note = document.createElement('p'); note.className = 'vnote'; note.textContent = table.note;
    el.append(wrap, note);
  }

  function bind({ scene, root, variants }) {
    const state = {};
    root.querySelectorAll('[data-variants]').forEach(group => {
      const family = group.dataset.variants, dflt = group.dataset.default, tableEl = root.querySelector(`[data-variant-table="${family}"]`);
      const model = family === 'embed' ? 'embedding' : family;
      state[model] = dflt;
      const apply = v => {
        state[model] = v;
        group.querySelectorAll('button[data-variant]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.variant === v)));
        if (tableEl) renderTable(tableEl, variants.table(model, variants.QWEN3), rowOf(model, v));
        scene.setData('variant', { ...state });
      };
      group.addEventListener('click', e => { const b = e.target.closest('button[data-variant]'); if (b) apply(b.dataset.variant); });
      group.querySelectorAll('button[data-variant]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.variant === dflt)));
      if (tableEl) renderTable(tableEl, variants.table(model, variants.QWEN3), rowOf(model, dflt));
    });
    scene.setData('variant', { ...state });
    return { state: () => ({ ...state }) };
  }
  return { bind, fmt, rowOf, COLS };
});
