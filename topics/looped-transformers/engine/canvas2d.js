/* Canvas 2D fallback with the same interface as gl.js: it projects every box and draws its
   screen rectangle, so a browser without WebGL still shows the same scene. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./layout.js'));
  else root.LoopCanvas2D = factory(root.LoopLayout);
})(typeof self !== 'undefined' ? self : this, function (Layout) {
  'use strict';
  const css = (c, a) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;
  function create(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    let last = null;
    function draw(frame) {
      last = frame || last; if (!last) return;
      const { objects, camera, aspect, bg, ink } = last, W = canvas.width, H = canvas.height;
      ctx.fillStyle = css(bg, 1); ctx.fillRect(0, 0, W, H);
      for (const o of objects) {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const p of Layout.cornersOf(o)) {
          const [sx, sy] = Layout.project(camera, aspect, p);
          x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
        }
        const px = v => (v * 0.5 + 0.5) * W, py = v => (0.5 - v * 0.5) * H;
        ctx.fillStyle = css(o.rgb, o.alpha); ctx.fillRect(px(x0), py(y1), px(x1) - px(x0), py(y0) - py(y1));
        ctx.strokeStyle = css(ink, o.alpha * o.edge); ctx.lineWidth = Math.max(1, W / 600);
        ctx.strokeRect(px(x0), py(y1), px(x1) - px(x0), py(y0) - py(y1));
      }
    }
    return {
      kind: '2d',
      resize(w, h) { canvas.width = w; canvas.height = h; },
      draw,
      coverage(bg) {
        const w = canvas.width, h = canvas.height, px = ctx.getImageData(0, 0, w, h).data;
        const b = bg.map(v => Math.round(v * 255)); let n = 0;
        for (let i = 0; i < px.length; i += 4) if (Math.abs(px[i] - b[0]) + Math.abs(px[i + 1] - b[1]) + Math.abs(px[i + 2] - b[2]) > 24) n++;
        return n / (w * h);
      },
      destroy() {},
    };
  }
  return { create };
});
