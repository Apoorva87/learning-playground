/* Turns a SceneState (from state.js) into a frame and hands it to a renderer. Labels are
   DOM elements positioned from the same projection, so they stay readable and selectable.
   Drawing happens on demand (show, resize, theme change), not in an animation loop. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./layout.js'), require('./gl.js'), require('./canvas2d.js'));
  else root.LoopScene = factory(root.LoopLayout, root.LoopGL, root.LoopCanvas2D);
})(typeof self !== 'undefined' ? self : this, function (Layout, GL, C2D) {
  'use strict';
  const REF_ASPECT = 1.3, DIM = 0.24;
  const TINTS = ['prelude', 'core', 'coda', 'tensor', 'msg', 'gate', 'hot', 'neutral'];

  const hex = s => { s = s.trim().replace('#', ''); if (s.length === 3) s = [...s].map(c => c + c).join(''); return [0, 2, 4].map(i => parseInt(s.slice(i, i + 2), 16) / 255); };
  function readPalette(el) {
    const cs = getComputedStyle(el), get = n => hex(cs.getPropertyValue(n) || '#888888');
    const pal = { bg: get('--stage-bg'), ink: get('--stage-ink') };
    for (const t of TINTS) pal[t] = get('--c-' + t);
    return pal;
  }

  function create({ canvas, labelLayer, beats, renderer = 'auto' }) {
    let sceneData = {}, canvasEl = canvas, impl = null, palette = null, state = null, frame = null, hover = new Set(), markerPos = null;
    const labelPool = [];

    function makeRenderer(kind) {
      let r = null;
      if (kind !== '2d') r = GL.create(canvasEl);
      if (!r) r = C2D.create(canvasEl);
      return r;
    }
    function swapTo2D() {                 // a canvas that held a WebGL context can never hold a 2D one
      const fresh = canvasEl.cloneNode(false);
      canvasEl.replaceWith(fresh); canvasEl = fresh;
      impl = C2D.create(canvasEl); fit(); redraw();
    }
    function wire() {
      canvasEl.addEventListener('webglcontextlost', e => { e.preventDefault(); swapTo2D(); wire(); });
    }
    impl = makeRenderer(renderer === 'auto' ? 'gl' : renderer);
    wire();

    function aspect() { return Math.max(canvasEl.clientWidth, 1) / Math.max(canvasEl.clientHeight, 1); }
    function fit() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      impl.resize(Math.max(Math.round(canvasEl.clientWidth * dpr), 1), Math.max(Math.round(canvasEl.clientHeight * dpr), 1));
    }

    /* Objects of every active layer, with the beat's lit set applied as dimming. */
    function collect(s) {
      const lits = Object.entries(s.lit).map(([name, a]) => ({ name, a, ids: new Set(Layout.idsFor(name)) }));
      for (const id of hover) lits.push({ name: id, a: 1, ids: new Set([id]) });
      const litOf = id => { let best = 0; for (const l of lits) if (l.a > best && (l.ids.has(id) || id === l.name || id.startsWith(l.name + '.'))) best = l.a; return best; };
      const any = lits.length > 0;
      const out = [];
      for (const layer of s.layers) {
        const objs = layer.level === 'micro' ? Layout.microObjects(layer.micro, sceneData[layer.micro] || (/^walk-/.test(layer.micro) ? sceneData.walk : undefined)) : Layout.macroObjects(s.fold);
        for (const o of objs) {
          const l = litOf(o.id), factor = any ? DIM + (1 - DIM) * l : 0.9;
          let alpha = o.alpha * layer.alpha * factor;
          if (o.v !== undefined) alpha *= 0.18 + 0.82 * Math.min(Math.abs(o.v), 1);        // signed values: brightness shows size
          out.push({ ...o, lit: l, alpha, tint: o.v !== undefined && o.v < 0 ? 'hot' : o.tint });
        }
      }
      const m = marker(s, out);
      if (m) out.push(m);
      return out;
    }

    /* The token marker rides along the visits in order: V0 -> V47. */
    function marker(s, objs) {
      markerPos = null;
      if (s.flow.kind !== 'token' || !s.layers.some(l => l.level === 'macro')) return null;
      const visits = objs.filter(o => /^V\d+$/.test(o.id));
      if (visits.length < 2) return null;
      const f = s.flow.t * (visits.length - 1), i = Math.min(Math.floor(f), visits.length - 2), k = f - i, a = visits[i], b = visits[i + 1];
      const p = ['x', 'y', 'z'].map(c => a[c] + (b[c] - a[c]) * k);
      markerPos = p;
      return { id: 'marker', kind: 'marker', x: p[0], y: p[1], z: p[2] + 0.9, w: 0.5, h: 0.5, d: 0.5, tint: 'hot', alpha: 1, lit: 1, full: true };
    }

    function effectiveCamera(s, objs) {
      const c = { ...s.camera }, a = aspect();
      const solid = objs.filter(o => o.kind !== 'marker' && o.alpha > 0.05);
      if (solid.length) {
        const k = Layout.fitCamera(solid, a, c).dist / Layout.fitCamera(solid, REF_ASPECT, c).dist;
        c.dist *= Math.min(Math.max(k, 0.6), 8);
        // Whatever the beat asked for, never crop: if the solid objects' bounding box would leave the frame, back off.
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (const o of solid) for (const p of Layout.cornersOf(o)) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
        let worst = 0;
        for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) { const [sx, sy] = Layout.project(c, a, [x, y, z]); worst = Math.max(worst, Math.abs(sx), Math.abs(sy)); }
        if (worst > 0.97) c.dist *= worst / 0.97;
      }
      return c;
    }

    function buildFrame() {
      const objs = collect(state), camera = effectiveCamera(state, objs), a = aspect();
      const depth = o => Layout.project(camera, a, [o.x, o.y, o.z])[2];
      const drawn = objs.filter(o => o.alpha > 0.01).map(o => ({ ...o, d0: depth(o) })).sort((p, q) => q.d0 - p.d0);
      for (const o of drawn) {
        o.rgb = palette[o.tint] || palette.neutral;
        o.edge = o.full ? 0.9 : 0.35;
      }
      return { objects: drawn, camera, aspect: a, bg: palette.bg, ink: palette.ink };
    }

    /* Wide tensor frames are labelled above their top-left corner; boxes are labelled at their centre.
       Long labels wrap at 180px, and a label is skipped if it would overlap one already placed. */
    function placeLabels() {
      const shown = [], seen = new Set(), taken = [], W = canvasEl.clientWidth;
      if (Object.keys(state.lit).length) {
        for (const o of [...frame.objects].reverse()) {            // nearest first, so a layer is labelled by its front visit
          if (!o.label || o.lit < 0.5 || seen.has(o.label) || o.kind === 'marker') continue;
          const above = o.kind === 'tensor' && o.w > 1.2;
          const p = above ? [o.x - o.w / 2, o.y + o.h / 2, o.z + o.d / 2] : [o.x, o.y, o.z + o.d / 2];
          const [sx, sy] = Layout.project(frame.camera, frame.aspect, p);
          const text = o.label.length * 5.9 + 12, w = Math.min(text, 180), h = Math.ceil(text / 180) * 13 + 4;
          let px = (sx * 0.5 + 0.5) * W; const py = (0.5 - sy * 0.5) * canvasEl.clientHeight;
          px = above ? Math.min(Math.max(px, 4), W - w - 4) : Math.min(Math.max(px, w / 2 + 4), W - w / 2 - 4);   // keep the label on the canvas
          const cx = above ? px + w / 2 : px, cy = above ? py - h * 0.58 : py;
          if (taken.some(t => Math.abs(t[0] - cx) < (t[2] + w) / 2 + 4 && Math.abs(t[1] - cy) < (t[3] + h) / 2 + 2)) continue;
          seen.add(o.label); taken.push([cx, cy, w, h]); shown.push([o, px, py, above]);
          if (shown.length >= 14) break;
        }
      }
      shown.forEach(([o, px, py, above], i) => {
        const el = labelPool[i] || (labelPool[i] = labelLayer.appendChild(Object.assign(document.createElement('span'), { className: 'lbl' })));
        el.textContent = o.label; el.hidden = false;
        el.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px) translate(${above ? '0, -105%' : '-50%, -50%'})`;
      });
      for (let i = shown.length; i < labelPool.length; i++) labelPool[i].hidden = true;
    }

    /* CSS-pixel position of an object's centre on the canvas, or null if it is not drawn. */
    function screenOf(id) {
      const o = frame && frame.objects.find(o => o.id === id); if (!o) return null;
      const [sx, sy] = Layout.project(frame.camera, frame.aspect, [o.x, o.y, o.z]);
      return [(sx * 0.5 + 0.5) * canvasEl.clientWidth, (0.5 - sy * 0.5) * canvasEl.clientHeight];
    }
    /* The id of the nearest visible object under a CSS-pixel point on the canvas. */
    function pick(px, py) {
      if (!frame) return null;
      let best = null;
      for (const o of frame.objects) {
        if (o.alpha < 0.12 || o.kind === 'marker') continue;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const p of Layout.cornersOf(o)) {
          const [sx, sy] = Layout.project(frame.camera, frame.aspect, p), cx = (sx * 0.5 + 0.5) * canvasEl.clientWidth, cy = (0.5 - sy * 0.5) * canvasEl.clientHeight;
          x0 = Math.min(x0, cx); x1 = Math.max(x1, cx); y0 = Math.min(y0, cy); y1 = Math.max(y1, cy);
        }
        if (px >= x0 && px <= x1 && py >= y0 && py <= y1 && (!best || o.d0 < best.d0)) best = o;
      }
      return best && best.id;
    }

    function redraw() {
      if (!state) return;
      palette = readPalette(document.documentElement);
      frame = buildFrame();
      impl.draw(frame);
      placeLabels();
    }
    function show(s) { state = s; redraw(); }

    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { fit(); redraw(); }) : null;
    if (ro) ro.observe(canvasEl.parentElement);
    const mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    if (mq && mq.addEventListener) mq.addEventListener('change', redraw);
    fit();

    return {
      show, redraw, screenOf, pick,
      setData(name, value) { sceneData[name] = value; redraw(); }, canvas: () => canvasEl,
      highlight(ids) { hover = new Set(ids.flatMap(Layout.idsFor)); redraw(); },
      getState() {
        return {
          ...state, lit: Object.keys(state.lit), litAlpha: state.lit, level: state.layers[state.layers.length - 1].level,
          highlight: [...hover], renderer: impl.kind,
          coverage: () => { impl.draw(frame); return impl.coverage(palette.bg); },
          marker: () => markerPos && markerPos.map(v => +v.toFixed(4)),
          bounds: () => {
            let maxX = 0, maxY = 0;
            for (const o of frame.objects) {
              if (o.alpha <= 0.05 || o.kind === 'marker') continue;
              for (const p of Layout.cornersOf(o)) { const [sx, sy] = Layout.project(frame.camera, frame.aspect, p); maxX = Math.max(maxX, Math.abs(sx)); maxY = Math.max(maxY, Math.abs(sy)); }
            }
            return { maxX, maxY };
          },
        };
      },
    };
  }
  return { create };
});
