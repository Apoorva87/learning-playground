/* Geometry for the scene. Pure: no DOM, no GL.
   The model has 24 physical layers and 48 layer visits per token. The macro layout draws all
   48 visits and morphs between two arrangements with one number, `fold`:
     fold 0  the stack as stored: 24 layers in a column, the three passes over the core piled
             behind each other, and a return arrow from layer 17 back to layer 6;
     fold 1  the stack as executed: 48 visits in a row, in order. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopLayout = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const N_LAYERS = 24, CORE_START = 6, CORE_END = 18, PASSES = 3;
  const CORE_LEN = CORE_END - CORE_START;                      // 12
  const N_VISITS = CORE_START + CORE_LEN * PASSES + (N_LAYERS - CORE_END); // 48
  const FULL_LAYERS = [5, 11, 17, 23];                         // the complement of local_layer_ids
  const FOV = 40 * Math.PI / 180;

  const lerp = (a, b, t) => a + (b - a) * t;

  /* Visit v (0..47) runs physical layer physicalOf(v). */
  function physicalOf(v) {
    if (v < CORE_START) return v;
    if (v < CORE_START + CORE_LEN * PASSES) return CORE_START + (v - CORE_START) % CORE_LEN;
    return v - CORE_LEN * (PASSES - 1);
  }
  function passOf(v) {
    if (v < CORE_START || v >= CORE_START + CORE_LEN * PASSES) return 0;
    return Math.floor((v - CORE_START) / CORE_LEN) + 1;
  }
  const regionOf = v => (v < CORE_START ? 'prelude' : passOf(v) ? 'core' : 'coda');

  /* Expand a name used in a beat's `lit` list into object ids. Names that are not shorthand
     pass through, so `msg.q.8` and `tensor.h` stay as they are. */
  function idsFor(name) {
    const all = Array.from({ length: N_VISITS }, (_, v) => v);
    let m;
    if ((m = /^L(\d+)$/.exec(name))) return all.filter(v => physicalOf(v) === +m[1]).map(v => 'V' + v);
    if ((m = /^pass\.(\d)$/.exec(name))) return all.filter(v => passOf(v) === +m[1]).map(v => 'V' + v);
    if (name === 'prelude' || name === 'coda') return all.filter(v => regionOf(v) === name).map(v => 'V' + v);
    if (name === 'core') return all.filter(v => regionOf(v) === 'core').map(v => 'V' + v);
    if (name === 'full') return all.filter(v => FULL_LAYERS.includes(physicalOf(v))).map(v => 'V' + v);
    const hit = microIds().filter(id => id === name || id.startsWith(name + '.'));   // tensor.h, pair, ptr, bar.stable: the name and everything under it
    if (hit.length) return hit;
    return [name];
  }

  const ROW_Y = 0.7, COL_STEP = 0.62, PASS_Z = 0.45;
  function macroObjects(fold) {
    const f = Math.min(Math.max(fold, 0), 1);
    const out = [];
    for (let v = 0; v < N_VISITS; v++) {
      const phys = physicalOf(v), pass = passOf(v);
      out.push({
        id: 'V' + v, kind: pass > 1 ? 'ghost' : 'visit',
        x: lerp(0, (v - (N_VISITS - 1) / 2) * COL_STEP, f),
        y: lerp((phys - (N_LAYERS - 1) / 2) * ROW_Y, 0, f),
        z: -(pass > 1 ? pass - 1 : 0) * PASS_Z * (1 - f),
        w: lerp(1.6, 0.5, f), h: lerp(0.5, 2.4, f), d: 1.2,
        tint: regionOf(v), pass, full: FULL_LAYERS.includes(phys),
        alpha: pass > 1 ? lerp(0.35, 1, f) : 1,
        label: 'layer ' + phys + (pass ? ', pass ' + pass : ''),
      });
    }
    out.push({
      id: 'hop.return', kind: 'arrow',
      x: 1.25, y: 0, z: 0, w: 0.07, h: (CORE_LEN - 1) * ROW_Y + 0.5, d: 0.07,
      tint: 'core', pass: 0, full: false, alpha: 1 - f, label: 'back to layer 6',
    });
    return out;
  }

  const MICRO = {};                                            // scenes register themselves from micro.js
  const microObjects = (name, data) => (MICRO[name] ? MICRO[name](data) : []);
  let microCache = null;
  const microIds = () => (microCache && microCache.n === Object.keys(MICRO).length ? microCache.ids
    : (microCache = { n: Object.keys(MICRO).length, ids: [...new Set(Object.keys(MICRO).flatMap(n => microObjects(n).map(o => o.id)))] }).ids);

  /* ---------- camera ---------- */
  function basis(cam) {
    const cp = Math.cos(cam.pitch), eye = [
      cam.tx + cam.dist * cp * Math.sin(cam.yaw), cam.ty + cam.dist * Math.sin(cam.pitch), cam.tz + cam.dist * cp * Math.cos(cam.yaw)];
    const f = norm([cam.tx - eye[0], cam.ty - eye[1], cam.tz - eye[2]]);
    const r = norm(cross(f, [0, 1, 0]));
    return { eye, f, r, u: cross(r, f) };
  }
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  /* World point to [screen x, screen y, depth]; screen coordinates run -1..1. */
  function project(cam, aspect, p) {
    const b = basis(cam), d = [p[0] - b.eye[0], p[1] - b.eye[1], p[2] - b.eye[2]];
    const z = dot(d, b.f), t = Math.tan(FOV / 2);
    return [dot(d, b.r) / (z * t * aspect), dot(d, b.u) / (z * t), z];
  }

  const cornersOf = o => {
    const c = [];
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) c.push([o.x + sx * o.w / 2, o.y + sy * o.h / 2, o.z + sz * o.d / 2]);
    return c;
  };

  /* A camera that frames every object, for a canvas of the given width/height ratio. */
  function fitCamera(objects, aspect, base) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const o of objects) for (const p of cornersOf(o)) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
    const pts = [];                                            // the bounding box's corners are enough: projection keeps convex shapes convex
    for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) pts.push([x, y, z]);
    const cam = { yaw: -0.5, pitch: 0.25, ...(base || {}), tx: (lo[0] + hi[0]) / 2, ty: (lo[1] + hi[1]) / 2, tz: (lo[2] + hi[2]) / 2, dist: 1 };
    const fits = dist => {
      const c = { ...cam, dist };
      return pts.every(p => { const [sx, sy, z] = project(c, aspect, p); return z > 0.1 && Math.abs(sx) <= 0.96 && Math.abs(sy) <= 0.96; });
    };
    let a = 0.5, b = 800;
    for (let i = 0; i < 48; i++) { const m = (a + b) / 2; if (fits(m)) b = m; else a = m; }
    cam.dist = b;
    return cam;
  }

  /* A front-on camera that frames a micro scene at the reference aspect (the scene refits it for other aspects). */
  function microCamera(name, margin = 1.08, data) {
    const cam = fitCamera(microObjects(name, data), 1.3, { yaw: 0, pitch: 0 });
    return { yaw: 0, pitch: 0, dist: +(cam.dist * margin).toFixed(3), tx: +cam.tx.toFixed(3), ty: +cam.ty.toFixed(3), tz: +cam.tz.toFixed(3) };
  }

  return { N_LAYERS, N_VISITS, CORE_START, CORE_END, PASSES, FULL_LAYERS, FOV, MICRO,
    physicalOf, passOf, regionOf, idsFor, macroObjects, microObjects, fitCamera, microCamera, project, cornersOf };
});
