/* 2D canvas view. x = dependency step, y = lane, size = chosen cost, colour = stage. */
(() => {
  'use strict';
  const { D, MACHINES, fmt } = window.QG ? window.QG : {};
  const Q = window.QG;
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const S = { phase: 'decode', metric: 'flops', machine: 'h100', ovh: 0, crit: true, strip: 'flops',
    on: new Set(Q.D.stages.map((_, i) => i)), step: null, sel: -1, hover: -1, playing: false,
    cx: 0, sx: 10, W: 0, H: 0 };
  let g, a, scale, order, stripBins;

  const LANE0 = 1;                                   // lane centre of the ribbon
  const sy = () => Math.min(64, Math.max(14, S.sx * 1.7));
  const X = l => (l - S.cx) * S.sx + S.W / 2;
  const Y = lane => S.H / 2 + (lane - LANE0) * sy();
  const toLevel = px => (px - S.W / 2) / S.sx + S.cx;

  function recompute() {
    g = Q.load(S.phase);
    a = Q.analyse(g, S.machine, S.ovh);
    scale = Q.sizeScale(g, S.metric);
    order = Array.from({ length: g.n }, (_, i) => i).sort((p, q) => scale.norm[q] - scale.norm[p]);
    if (S.step === null || S.step > g.maxLevel) S.step = g.maxLevel;
    $('step').max = g.maxLevel; $('step').value = S.step;
    buildStrip(); summary(); stageTable(); drawKey(); draw(); drawStrip(); detail();
  }

  /* ---------- summary cards ---------- */
  function summary() {
    const L = g.maxLevel, onPathShare = a.pathFlops / a.totFlops;
    const pass = S.phase === 'decode' ? 'One cached decode step' : 'The 16-token prefill';
    $('lead').innerHTML = `${pass} is <b>${g.n.toLocaleString()} operations</b> in a dependency chain <b>${L.toLocaleString()} steps deep</b>. ` +
      `At most <b>${a.widest}</b> ops are ever ready at once, and <b>${fmt.pct(a.single / L)}</b> of the steps hold a single op. ` +
      `<b>${fmt.pct(onPathShare)}</b> of all FLOPs sit on the critical path. Layers cannot overlap, because each one needs the previous layer's output.`;
    const cards = [
      ['Chain depth', L.toLocaleString() + ' steps', `${g.n.toLocaleString()} ops · mean ${(g.n / L).toFixed(2)} per step · ${g.constantOps} weight-only ops left out`],
      ['Widest step', a.widest + ' ops', widestNote()],
      ['Graph-level parallelism', a.parallelism.toFixed(2) + '×', `serial time ${fmt.time(a.work)} ÷ critical path ${fmt.time(a.span)} (${a.machine.name.split(' (')[0]})`],
      ['FLOPs on critical path', fmt.pct(a.pathFlops / a.totFlops), `${fmt.flops(a.pathFlops)} of ${fmt.flops(a.totFlops)}`],
      ['Memory-bound time', fmt.pct(a.memBoundTime / a.work), `ops below ${a.ridge.toFixed(0)} FLOP/B ridge · ${fmt.pct(a.memBound / g.n)} of ops`],
    ];
    $('cards').innerHTML = cards.map(c => `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="s">${c[2]}</div></div>`).join('');
  }

  function widestNote() {
    const l = g.byLevel.findIndex(x => x.length === a.widest), ids = g.byLevel[l] || [];
    const names = [...new Set(ids.map(i => Q.D.stages[g.stage[i]][1].toLowerCase()))].join(', ');
    return `first at step ${l} (block ${g.block[ids[0]]}): ${names}`;
  }

  function stageTable() {
    const col = Q.colors();
    const tb = document.querySelector('#stageTable tbody');
    tb.innerHTML = Q.D.stages.map(([, name], i) => {
      const s = a.stages[i]; if (!s.ops) return '';
      return `<tr class="stage-row${S.on.has(i) ? '' : ' off'}" data-i="${i}"><td><span class="sw" style="background:${col[i]}"></span>${name}</td><td>${s.ops.toLocaleString()}</td>` +
        `<td>${fmt.pct(s.flops / a.totFlops)}</td><td>${fmt.pct(s.bytes / a.totBytes)}</td><td>${fmt.pct(s.time / a.work)}</td><td>${fmt.pct(s.pathTime / a.span)}</td></tr>`;
    }).join('');
  }

  function drawKey() {
    const lo = scale.lo, hi = scale.hi, m = S.metric;
    const val = v => m === 'flops' ? fmt.flops(v) : m === 'bytes' ? fmt.bytes(v) : fmt.ai(v) + ' FLOP/B';
    $('key').innerHTML = `<span>Colour = stage (table below)</span><span><i style="width:6px;height:6px"></i>${val(lo)}</span><span><i style="width:18px;height:18px"></i>${val(hi)}</span>` +
      `<span>Size = ${Q.SIZE_LABEL[m]}, log scale (smallest 2% and largest clipped to the ends)</span><span><i style="width:12px;height:12px;background:none;border:2px solid var(--ink)"></i>compute-bound on this machine</span>`;
  }

  /* ---------- overview strip ---------- */
  function buildStrip() {
    const W = Math.max(200, Math.floor($('strip').clientWidth)), bins = W, per = (g.maxLevel + 1) / bins;
    const stack = Q.D.stages.map(() => new Float64Array(bins));
    for (let i = 0; i < g.n; i++) {
      const b = Math.min(bins - 1, Math.floor(g.level[i] / per));
      stack[g.stage[i]][b] += S.strip === 'flops' ? g.flops[i] : S.strip === 'bytes' ? g.bytes[i] : 1;
    }
    let max = 0;
    for (let b = 0; b < bins; b++) { let t = 0; stack.forEach(s => (t += s[b])); max = Math.max(max, t / per); }
    stripBins = { stack, bins, per, max, W };
  }

  function sizeCanvas(c) {
    const r = c.getBoundingClientRect(), d = devicePixelRatio || 1;
    c.width = Math.round(r.width * d); c.height = Math.round(r.height * d);
    const x = c.getContext('2d'); x.setTransform(d, 0, 0, d, 0, 0);
    return [x, r.width, r.height];
  }

  function drawStrip() {
    const [x, W, H] = sizeCanvas($('strip')), col = Q.colors();
    if (stripBins.W !== Math.max(200, Math.floor(W))) buildStrip();
    const { stack, bins, per, max } = stripBins, pad = 18, h = H - pad - 4;
    x.clearRect(0, 0, W, H);
    // block ticks
    x.font = '10px system-ui'; x.fillStyle = css('--ink2'); x.textBaseline = 'top';
    g.blocks.forEach((b, i) => {
      if (b.hi < 0) return;
      const px = (b.lo / (g.maxLevel + 1)) * W;
      x.fillStyle = Q.D.layer_types[i] === 'F' ? css('--ink') : css('--grid'); x.fillRect(px, pad - 3, 1, 3);
      if (i % 8 === 0 || Q.D.layer_types[i] === 'F' && i % 4 === 3 && W > 900 && i % 8 === 3) { x.fillStyle = css('--ink2'); x.fillText(i, px + 2, 1); }
    });
    // stacked bars, one per bin. Height is sqrt(total) so small steps stay visible;
    // segments keep their true share of each bar.
    const bw = W / bins;
    for (let b = 0; b < bins; b++) {
      let tot = 0; for (let s = 0; s < stack.length; s++) tot += stack[s][b] / per;
      if (!tot) continue;
      const full = Math.max(1.2, Math.sqrt(tot / max) * h); let base = H - 4;
      for (let s = 0; s < stack.length; s++) {
        const v = stack[s][b] / per; if (!v) continue;
        const hh = (v / tot) * full;
        x.fillStyle = col[s]; x.fillRect(b * bw, base - hh, Math.ceil(bw), hh); base -= hh;
      }
    }
    // viewport box
    const l0 = toLevel(0), l1 = toLevel(S.W), p0 = (l0 / (g.maxLevel + 1)) * W, p1 = (l1 / (g.maxLevel + 1)) * W;
    x.strokeStyle = css('--ink'); x.lineWidth = 1.5; x.fillStyle = 'rgba(128,128,128,.12)';
    x.fillRect(p0, pad - 4, Math.max(3, p1 - p0), H - pad + 4); x.strokeRect(p0 + .5, pad - 3.5, Math.max(3, p1 - p0), H - pad + 3);
    const unit = S.strip === 'flops' ? fmt.flops(max) + ' per step' : S.strip === 'bytes' ? fmt.bytes(max) + ' per step' : max.toFixed(1) + ' ops per step';
    $('stripTitle').textContent = `Work at each dependency step, stacked by stage (bar height ∝ √value, tallest bin ≈ ${unit})`;
  }

  /* ---------- main graph ---------- */
  function radius(i) {
    return Math.min(42, Math.max(1.3, (0.1 + 0.4 * scale.norm[i]) * S.sx * 1.25));
  }
  const bound = i => g.ai[i] >= a.ridge;

  function draw() {
    const c = $('main'); const [x, W, H] = sizeCanvas(c); S.W = W; S.H = H;
    const col = Q.colors(), l0 = toLevel(-30), l1 = toLevel(W + 30);
    x.clearRect(0, 0, W, H);
    // block bands
    x.font = '11px system-ui'; x.textBaseline = 'top';
    g.blocks.forEach((b, i) => {
      if (b.hi < 0 || b.hi < l0 || b.lo > l1) return;
      const xa = X(b.lo - .5), xb = X(b.hi + .5);
      if (i % 2) { x.fillStyle = css('--band'); x.fillRect(xa, 0, xb - xa, H); }
      if (xb - xa > 34) { x.fillStyle = css('--ink2'); x.fillText(`block ${i} · ${Q.D.layer_types[i] === 'F' ? 'full attention' : 'recurrent'}`.slice(0, Math.floor((xb - xa) / 6.3)), xa + 4, 4); }
    });
    // time ruler
    x.fillStyle = css('--ink2'); x.textBaseline = 'bottom';
    const stepTick = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find(t => t * S.sx > 70) || 1000;
    for (let l = Math.ceil(Math.max(0, l0) / stepTick) * stepTick; l < l1; l += stepTick) {
      const t = a.levelStart[l]; x.fillStyle = css('--grid'); x.fillRect(X(l), H - 18, 1, 6);
      x.fillStyle = css('--ink2'); x.fillText(`step ${l}` + (isFinite(t) ? ` · ${fmt.time(t)}` : ''), X(l) + 3, H - 3);
    }
    const dim = i => !S.on.has(g.stage[i]) || g.level[i] > S.step;
    // edges
    const detailEdges = S.sx >= 2.2;
    if (detailEdges) {
      x.lineWidth = 1; x.strokeStyle = css('--edge'); x.beginPath();
      for (let e = 0; e < g.m; e++) {
        const p = g.edges[2 * e], q = g.edges[2 * e + 1];
        if (g.level[q] < l0 || g.level[p] > l1 || dim(q) || dim(p) || g.level[q] - g.level[p] > 60) continue;   // long-range fan-out (RoPE tables) shows on selection
        x.moveTo(X(g.level[p]), Y(g.lane[p])); x.lineTo(X(g.level[q]), Y(g.lane[q]));
      }
      x.stroke();
    }
    if (S.sel >= 0) {
      x.lineWidth = 2; x.strokeStyle = css('--accent'); x.beginPath();
      for (let k = g.pOff[S.sel]; k < g.pOff[S.sel + 1]; k++) { const p = g.par[k]; x.moveTo(X(g.level[p]), Y(g.lane[p])); x.lineTo(X(g.level[S.sel]), Y(g.lane[S.sel])); }
      for (let k = g.cOff[S.sel]; k < g.cOff[S.sel + 1]; k++) { const q = g.chi[k]; x.moveTo(X(g.level[S.sel]), Y(g.lane[S.sel])); x.lineTo(X(g.level[q]), Y(g.lane[q])); }
      x.stroke();
    }
    if (S.crit) {
      x.lineWidth = Math.max(1.5, Math.min(4, S.sx / 6)); x.strokeStyle = css('--path'); x.beginPath();
      for (let k = 1; k < a.path.length; k++) {
        const p = a.path[k - 1], q = a.path[k];
        if (g.level[q] < l0 || g.level[p] > l1) continue;
        x.moveTo(X(g.level[p]), Y(g.lane[p])); x.lineTo(X(g.level[q]), Y(g.lane[q]));
      }
      x.stroke();
    }
    // nodes: big first so small ones stay visible on top
    for (const i of order) {
      const lv = g.level[i]; if (lv < l0 || lv > l1) continue;
      const cx = X(lv), cy = Y(g.lane[i]), r = radius(i), d = dim(i);
      x.globalAlpha = d ? 0.13 : 1; x.fillStyle = col[g.stage[i]];
      x.beginPath(); x.arc(cx, cy, r, 0, 6.2832); x.fill();
      if (!d && r > 2.5) { x.lineWidth = 1; x.strokeStyle = css('--surface'); x.stroke(); }
      if (!d && bound(i) && r > 3) { x.lineWidth = 2.2; x.strokeStyle = css('--ink'); x.beginPath(); x.arc(cx, cy, r + 2, 0, 6.2832); x.stroke(); }
      x.globalAlpha = 1;
    }
    // wavefront
    if (S.step < g.maxLevel) {
      x.fillStyle = 'rgba(128,128,128,.10)'; x.fillRect(X(S.step + .5), 0, W, H);
      x.fillStyle = css('--ink'); x.fillRect(X(S.step + .5) - 1, 0, 2, H);
      for (const i of g.byLevel[S.step] || []) { if (!S.on.has(g.stage[i])) continue; x.lineWidth = 2.5; x.strokeStyle = css('--ink'); x.beginPath(); x.arc(X(g.level[i]), Y(g.lane[i]), radius(i) + 4, 0, 6.2832); x.stroke(); }
    }
    if (S.sel >= 0) { x.lineWidth = 3; x.strokeStyle = css('--accent'); x.beginPath(); x.arc(X(g.level[S.sel]), Y(g.lane[S.sel]), radius(S.sel) + 5, 0, 6.2832); x.stroke(); }
    stepReadout();
  }

  function stepReadout() {
    const ids = g.byLevel[S.step] || [];
    let f = 0, b = 0; ids.forEach(i => { f += g.flops[i]; b += g.bytes[i]; });
    const t = a.levelStart[S.step];
    $('stepOut').textContent = S.step >= g.maxLevel && !S.playing && S.step === g.maxLevel
      ? `Showing all ${g.maxLevel.toLocaleString()} steps. Drag the slider or press play to replay the dataflow step by step.`
      : `Step ${S.step.toLocaleString()} of ${g.maxLevel.toLocaleString()}` + (isFinite(t) ? ` · modeled time ${fmt.time(t)}` : '') +
        ` · ${ids.length} op${ids.length === 1 ? '' : 's'} can run at this step (${fmt.flops(f)}, ${fmt.bytes(b)}) · ${ids.length > 1 ? 'parallel' : 'serial'}`;
  }

  /* ---------- selection / tooltip ---------- */
  function pick(mx, my) {
    let best = -1, bd = 1e9;
    const l0 = toLevel(mx - 40), l1 = toLevel(mx + 40);
    for (let l = Math.max(0, Math.floor(l0)); l <= Math.min(g.maxLevel, Math.ceil(l1)); l++) for (const i of g.byLevel[l]) {
      if (!S.on.has(g.stage[i]) || g.level[i] > S.step) continue;
      const d = Math.hypot(X(g.level[i]) - mx, Y(g.lane[i]) - my) - radius(i);
      if (d < bd) { bd = d; best = i; }
    }
    return bd < 7 ? best : -1;
  }

  function describe(i) {
    const nm = Q.D.stages[g.stage[i]][1], d = Q.describe(g, a, i);
    return { nm, d, rows: [
      ['Where', (g.module[i] || 'model') + (g.block[i] >= 0 ? '' : '')],
      ['Op', g.type[i] + ' · ' + g.shape[i]],
      ['Stage', nm + (g.block[i] >= 0 ? ` · block ${g.block[i]}` : '')],
      ['FLOPs', fmt.flops(g.flops[i])],
      ['Bytes', `${fmt.bytes(g.bytes[i])} (in ${fmt.bytes(g.inB[i])}, out ${fmt.bytes(g.outB[i])}${g.wB[i] ? ', weights ' + fmt.bytes(g.wB[i]) : ''})`],
      ['FLOPs / byte', `${fmt.ai(g.ai[i])} · ${d.bound} on this machine`],
      ['Modeled time', `${fmt.time(a.t[i])} starting at ${fmt.time(a.start[i])}`],
      ['Step', `${g.level[i]} · ${d.par.length} input${d.par.length === 1 ? '' : 's'}, ${d.chi.length} consumer${d.chi.length === 1 ? '' : 's'}` + (a.onPath[i] ? ' · on the critical path' : '')],
    ] };
  }

  function showTip(i, ev) {
    const t = $('tip');
    if (i < 0) { t.style.display = 'none'; return; }
    const D = describe(i);
    t.innerHTML = `<b>${g.module[i] || g.type[i]}</b><br>${g.type[i]} · ${fmt.flops(g.flops[i])} · ${fmt.bytes(g.bytes[i])} · ${fmt.ai(g.ai[i])} FLOP/B<br>${D.nm}${g.block[i] >= 0 ? ' · block ' + g.block[i] : ''} · step ${g.level[i]}`;
    t.style.display = 'block'; t.style.left = Math.min(innerWidth - 330, ev.clientX + 14) + 'px'; t.style.top = (ev.clientY + 14) + 'px';
  }

  function detail() {
    const box = $('detail');
    if (S.sel < 0) { box.innerHTML = '<h3>Selected op</h3><p style="color:var(--ink2)">Click a node, or press Enter on the graph, to see its shapes, cost and neighbours.</p>'; return; }
    const D = describe(S.sel), link = i => `<button data-go="${i}">${g.module[i].split('.').slice(-2).join('.') || g.type[i]} ${g.type[i]}</button>`;
    box.innerHTML = `<h3><span class="sw" style="background:${Q.colors()[g.stage[S.sel]]}"></span>${D.nm}</h3><dl class="kv">${D.rows.map(r => `<dt>${r[0]}</dt><dd>${r[1]}</dd>`).join('')}</dl>` +
      `<div class="links"><b style="font-size:12px;color:var(--ink2)">INPUTS</b> ${D.d.par.slice(0, 8).map(link).join('') || '—'}</div><div class="links"><b style="font-size:12px;color:var(--ink2)">CONSUMERS</b> ${D.d.chi.slice(0, 8).map(link).join('') || '—'}</div>`;
  }

  function goTo(i) { S.sel = i; S.cx = g.level[i]; draw(); drawStrip(); detail(); }

  /* ---------- interaction ---------- */
  function fit(block) {
    const b = g.blocks[block];
    S.sx = Math.max(2, Math.min(60, S.W / (b.hi - b.lo + 6))); S.cx = (b.lo + b.hi) / 2;
  }
  const clampView = () => { S.cx = Math.max(0, Math.min(g.maxLevel, S.cx)); };

  function bind() {
    const c = $('main'); let drag = null;
    c.addEventListener('pointerdown', e => { c.setPointerCapture(e.pointerId); drag = { x: e.clientX, cx: S.cx, moved: false }; c.classList.add('drag'); });
    c.addEventListener('pointermove', e => {
      const r = c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      if (drag) { const dx = e.clientX - drag.x; if (Math.abs(dx) > 3) drag.moved = true; S.cx = drag.cx - dx / S.sx; clampView(); draw(); drawStrip(); return; }
      const h = pick(mx, my); if (h !== S.hover) { S.hover = h; } showTip(h, e);
    });
    c.addEventListener('pointerup', e => {
      const r = c.getBoundingClientRect(); c.classList.remove('drag');
      if (drag && !drag.moved) { S.sel = pick(e.clientX - r.left, e.clientY - r.top); draw(); detail(); }
      drag = null;
    });
    c.addEventListener('pointerleave', () => showTip(-1));
    c.addEventListener('wheel', e => {
      e.preventDefault(); const r = c.getBoundingClientRect(), mx = e.clientX - r.left, before = toLevel(mx);
      S.sx = Math.max(.25, Math.min(140, S.sx * Math.exp(-e.deltaY * 0.0015)));
      S.cx = before - (mx - S.W / 2) / S.sx; clampView(); draw(); drawStrip();
    }, { passive: false });
    c.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight') { S.cx += 20 / S.sx * 3; } else if (e.key === 'ArrowLeft') { S.cx -= 20 / S.sx * 3; }
      else if (e.key === '+' || e.key === '=') S.sx *= 1.2; else if (e.key === '-') S.sx /= 1.2;
      else if (e.key === 'Enter') { const m = Math.round(S.cx), ids = g.byLevel[m] || []; if (ids.length) { S.sel = ids[0]; detail(); } }
      else return;
      e.preventDefault(); clampView(); draw(); drawStrip();
    });
    const strip = $('strip'); let sd = false;
    const jump = e => { const r = strip.getBoundingClientRect(); S.cx = ((e.clientX - r.left) / r.width) * (g.maxLevel + 1); clampView(); draw(); drawStrip(); };
    strip.addEventListener('pointerdown', e => { sd = true; strip.setPointerCapture(e.pointerId); jump(e); });
    strip.addEventListener('pointermove', e => { if (sd) jump(e); });
    strip.addEventListener('pointerup', () => (sd = false));

    const seg = (id, key) => $(id).addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      [...$(id).children].forEach(x => x.setAttribute('aria-pressed', x === b));
      S[key] = b.dataset.v; if (key === 'phase') { S.sel = -1; S.step = null; } recompute(); if (key === 'phase') { fit(3); draw(); drawStrip(); }
    });
    seg('phase', 'phase'); seg('metric', 'metric');
    $('zoom').addEventListener('click', e => { const b = e.target.closest('button'); if (b) view(b.dataset.v); });
    $('machine').innerHTML = Object.entries(MACHINES).map(([k, m]) => `<option value="${k}">${m.name}</option>`).join('');
    $('machine').onchange = e => { S.machine = e.target.value; recompute(); };
    $('ovh').onchange = e => { S.ovh = Math.max(0, +e.target.value || 0); recompute(); };
    $('crit').onchange = e => { S.crit = e.target.checked; draw(); };
    $('stripMetric').onchange = e => { S.strip = e.target.value; buildStrip(); drawStrip(); };
    $('stageTable').addEventListener('click', e => {
      const tr = e.target.closest('tr.stage-row'); if (!tr) return;
      const i = +tr.dataset.i, all = Q.D.stages.length;
      if (S.on.size === all) { S.on = new Set([i]); } else if (S.on.has(i) && S.on.size === 1) { S.on = new Set(Q.D.stages.map((_, k) => k)); } else if (S.on.has(i)) S.on.delete(i); else S.on.add(i);
      stageTable(); draw();
    });
    $('detail').addEventListener('click', e => { const b = e.target.closest('[data-go]'); if (b) goTo(+b.dataset.go); });
    $('step').oninput = e => { S.step = +e.target.value; S.playing = false; $('play').textContent = '▶ Play the dataflow'; draw(); };
    $('reset').onclick = () => { S.step = g.maxLevel; $('step').value = S.step; S.playing = false; $('play').textContent = '▶ Play the dataflow'; draw(); };
    $('play').onclick = () => {
      S.playing = !S.playing; $('play').textContent = S.playing ? '❚❚ Pause' : '▶ Play the dataflow';
      if (S.playing) { if (S.step >= g.maxLevel) S.step = Math.max(1, Math.round(S.cx) - 4); tick(); }
    };
    addEventListener('resize', () => { draw(); drawStrip(); });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { stageTable(); draw(); drawStrip(); });
  }
  let last = 0;
  function tick(ts = 0) {
    if (!S.playing) return;
    if (ts - last > 60) { last = ts; S.step++; if (S.step > g.maxLevel) { S.playing = false; S.step = g.maxLevel; $('play').textContent = '▶ Play the dataflow'; }
      const x = X(S.step); if (x > S.W * .8 || x < 0) S.cx = S.step - (S.W * .3) / S.sx;
      $('step').value = S.step; draw(); drawStrip(); }
    requestAnimationFrame(tick);
  }

  function fitRange(lo, hi) { S.sx = Math.max(.2, Math.min(80, S.W / (hi - lo + 8))); S.cx = (lo + hi) / 2; }
  function view(name) {
    if (name === 'all') fitRange(0, g.maxLevel);
    else if (name === 'four') fitRange(g.blocks[0].lo, g.blocks[3].hi);
    else if (name === 'b0') fit(0); else fit(3);
    draw(); drawStrip();
  }

  function start() {
    bind();
    const h = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
    if (h.phase === 'prefill') { S.phase = 'prefill'; }
    if (['flops', 'bytes', 'ai'].includes(h.metric)) S.metric = h.metric;
    if (MACHINES[h.machine]) S.machine = h.machine;
    if (h.ovh) S.ovh = +h.ovh;
    for (const [id, key] of [['phase', 'phase'], ['metric', 'metric']]) [...$(id).children].forEach(b => b.setAttribute('aria-pressed', b.dataset.v === S[key]));
    $('machine').value = S.machine; $('ovh').value = S.ovh;
    const c = $('main'); const r = c.getBoundingClientRect(); S.W = r.width; S.H = r.height;
    recompute(); fit(3);
    if (h.view) view(h.view);
    if (h.sel) { S.sel = +h.sel; S.cx = g.level[S.sel]; detail(); }
    if (h.step) { S.step = +h.step; $('step').value = S.step; }
    if (h.dark) document.documentElement.dataset.theme = 'dark';
    stageTable(); draw(); drawStrip();
  }
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', start); else start();
})();
