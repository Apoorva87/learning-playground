/* Small multiples as inline SVG. Each <figure class="chart" data-chart='{...}'> holds one
   single-series chart: line (value per pass) or col (count per pass). One shared 0-100 axis for the
   percentage panels, 2px lines, 8px markers with a surface ring, a few direct labels, a hover
   tooltip (also on keyboard focus), and a table view in the page for the same numbers. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const W = 250, H = 168, M = { l: 34, r: 14, t: 30, b: 30 };
  const el = (name, attrs = {}, text) => { const e = document.createElementNS(NS, name); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };

  function render(fig) {
    const spec = JSON.parse(fig.dataset.chart), n = spec.y.length, ymax = spec.ymax;
    const x = i => M.l + (W - M.l - M.r) * (spec.kind === 'col' ? (i + 0.5) / n : (n === 1 ? 0.5 : i / (n - 1))), y = v => H - M.b - (H - M.t - M.b) * v / ymax;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${spec.title}: ${spec.y.map((v, i) => `${spec.x[i]} pass${spec.x[i] > 1 ? 'es' : ''}, ${v}${spec.unit}`).join('; ')}` });
    svg.appendChild(el('text', { x: M.l, y: 16, class: 'ct' }, spec.title));
    for (const t of spec.ticks) {                                           // recessive hairline grid with y labels
      svg.appendChild(el('line', { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: 'grid' }));
      svg.appendChild(el('text', { x: M.l - 5, y: y(t) + 3.5, class: 'tick', 'text-anchor': 'end' }, t));
    }
    spec.x.forEach((v, i) => svg.appendChild(el('text', { x: x(i), y: H - M.b + 15, class: 'tick', 'text-anchor': 'middle' }, v)));
    svg.appendChild(el('text', { x: (M.l + W - M.r) / 2, y: H - 3, class: 'tick', 'text-anchor': 'middle' }, 'loop passes'));
    if (spec.kind === 'line') {
      svg.appendChild(el('polyline', { points: spec.y.map((v, i) => `${x(i)},${y(v)}`).join(' '), class: 'series', fill: 'none', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    } else {
      const bw = Math.min(24, (W - M.l - M.r) / n * 0.5);                 // bars: at most 24px, 4px rounded tip, square at the baseline
      spec.y.forEach((v, i) => svg.appendChild(el('path', { class: 'bar', d: `M${x(i) - bw / 2},${y(0)} V${y(v) + 4} a4,4 0 0 1 4,-4 h${bw - 8} a4,4 0 0 1 4,4 V${y(0)} Z` })));
    }
    const tip = fig.querySelector('.tip') || fig.appendChild(Object.assign(document.createElement('div'), { className: 'tip', hidden: true, role: 'status' }));
    const show = (i, ev) => {
      tip.hidden = false; tip.textContent = `${spec.x[i]} pass${spec.x[i] > 1 ? 'es' : ''}: ${spec.y[i]}${spec.unit}${spec.note ? ' · ' + spec.note[i] : ''}`;
      const r = fig.getBoundingClientRect(), p = svg.getBoundingClientRect(), sx = p.width / W;
      tip.style.left = Math.min(Math.max(x(i) * sx, 40), r.width - 40) + 'px'; tip.style.top = (y(spec.y[i]) * sx - 8) + 'px';
    };
    spec.y.forEach((v, i) => {
      if (spec.kind === 'line') { svg.appendChild(el('circle', { cx: x(i), cy: y(v), r: 4, class: 'dot' })); }
      const hit = el('circle', { cx: x(i), cy: spec.kind === 'line' ? y(v) : (y(v) + y(0)) / 2, r: 14, class: 'hit', tabindex: 0, 'aria-label': `${spec.x[i]} passes: ${v}${spec.unit}` });
      hit.addEventListener('mouseenter', e => show(i, e)); hit.addEventListener('focus', e => show(i, e));
      hit.addEventListener('mouseleave', () => { tip.hidden = true; }); hit.addEventListener('blur', () => { tip.hidden = true; });
      svg.appendChild(hit);
      if (spec.kind === 'col' || i === 0 || i === n - 1) svg.appendChild(el('text', { x: x(i), y: y(v) - 9, class: 'val', 'text-anchor': spec.kind === 'col' ? 'middle' : 'end' }, `${v}${spec.unit}`));   // first and last value only
    });
    fig.insertBefore(svg, fig.firstChild);
  }
  window.LoopCharts = { render, renderAll: root => (root || document).querySelectorAll('figure.chart[data-chart]').forEach(render) };
})();
