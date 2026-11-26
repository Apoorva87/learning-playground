/* "In the recorded run" tables, built from assets/inkling-capture-summary.json so the page and the
   data cannot disagree. The builders are pure; bind() fetches the summary and fills every
   <div data-recorded="gate-ratio|overlap|messages|timing"> in the page. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopRecorded = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const ROLES = ['q', 'k', 'v', 'mlp'], LAYERS = Array.from({ length: 12 }, (_, i) => i + 6);
  const roleHead = () => [2, 3].flatMap(pass => ROLES.map(role => ({ pass, role, label: `p${pass} ${role}` })));

  /* ‖g ⊙ m‖ / ‖(1 − f) ⊙ h‖ at the last token: how much the message adds, relative to what is kept. */
  function gateRatio(S) {
    const head = roleHead(), rows = LAYERS.map(layer => ({
      layer, label: 'layer ' + layer,
      cells: head.map(h => { const d = S.gate_detail.find(x => x.layer === layer && x.pass === h.pass && x.role === h.role); return d.add_norm_last / d.keep_norm_last; }),
    }));
    const all = rows.flatMap(r => r.cells);
    return { caption: 'Message added over state kept, ‖g⊙m‖ / ‖(1−f)⊙h‖, at the last prompt token', head, rows, min: Math.min(...all), max: Math.max(...all) };
  }
  function overlap(S) {
    const head = ['passes 1 and 2', 'passes 2 and 3', 'all three', 'distinct experts, last token'].map(label => ({ label }));
    const rows = S.expert_overlap.map(o => ({ layer: o.layer, label: 'layer ' + o.layer, cells: [o.p1p2, o.p2p3, o.all3, o.union_last] }));
    return { caption: 'Experts shared between passes (of 8, averaged over prompt tokens)', head, rows, min: 0, max: 8 };
  }
  /* The weight each message puts on the previous pass's final state (the last of its two sources). */
  function messages(S) {
    const head = roleHead(), rows = LAYERS.map(layer => ({
      layer, label: 'layer ' + layer,
      cells: head.map(h => { const m = S.messages.find(x => x.layer === layer && x.before_visit === 18 + (h.pass - 2) * 12 + (layer - 6)); return m.w_last[h.role][m.n_sources - 1]; }),
    }));
    return { caption: 'Weight a message puts on the previous pass’s final state, at the last prompt token', head, rows, min: 0, max: 1 };
  }
  function timing(S) {
    const label = s => (s.stage === 'core' ? `core pass ${s.pass}` : s.stage);
    return { caption: 'Prefill time by stage, 43 tokens, CPU', head: [{ label: 'seconds' }], rows: S.stage_ms.map(s => ({ label: label(s), seconds: s.ms / 1000, cells: [s.ms / 1000] })), min: 0, max: Math.max(...S.stage_ms.map(s => s.ms / 1000)) };
  }

  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  /* Every number is printed; shading is a second cue, one hue, darker for larger. */
  function tableHtml(t, fmt) {
    const span = Math.max(t.max - t.min, 1e-9);
    return `<table class="heat"><caption>${esc(t.caption)}</caption><thead><tr><th scope="col"></th>${t.head.map(h => `<th scope="col">${esc(h.label)}</th>`).join('')}</tr></thead><tbody>` +
      t.rows.map(r => `<tr><th scope="row">${esc(r.label)}</th>${r.cells.map(v => `<td style="--heat:${(0.08 + 0.5 * (v - t.min) / span).toFixed(3)}">${fmt(v)}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
  }
  const FMT = { 'gate-ratio': v => v.toFixed(2), overlap: v => (Number.isInteger(v) ? String(v) : v.toFixed(1)), messages: v => v.toFixed(2), timing: v => v.toFixed(2) };

  async function bind(rootEl = document) {
    const holders = rootEl.querySelectorAll('[data-recorded]');
    if (!holders.length) return;
    const S = await fetch('assets/inkling-capture-summary.json').then(r => r.json());
    const build = { 'gate-ratio': gateRatio, overlap, messages, timing };
    holders.forEach(el => { const k = el.dataset.recorded; el.innerHTML = tableHtml(build[k](S), FMT[k]); });
  }
  return { gateRatio, overlap, messages, timing, tableHtml, bind };
});
