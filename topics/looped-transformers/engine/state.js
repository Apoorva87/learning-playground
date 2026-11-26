/* The one source of truth for the looped-transformer articles.
   Everything here is pure: no DOM, no clock. Scroll, Step and Play all write the same
   (beat, progress) pair into a store, and the scene is a function of that pair. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopState = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* The first 35% of a beat's progress moves the scene from the previous beat's pose to this
     beat's pose. The rest is this beat's own animation (the flow), which runs from 0 to 1. */
  const ARRIVE = 0.35;

  const clamp01 = x => (x > 0 ? (x < 1 ? x : 1) : 0);          // NaN falls through to 0
  const ease = x => { x = clamp01(x); return x * x * (3 - 2 * x); };
  const lerp = (a, b, t) => a + (b - a) * t;

  /* tops: each section's top edge relative to the viewport, ascending. anchorY: the viewport
     line that decides the current beat. docEnd: the bottom edge of the last section. */
  function beatFromScroll(tops, anchorY, docEnd) {
    let beat = -1;
    for (let i = 0; i < tops.length; i++) if (tops[i] <= anchorY) beat = i;
    if (beat < 0) return { beat: 0, progress: 0 };
    const top = tops[beat];
    const bottom = beat + 1 < tops.length ? tops[beat + 1] : docEnd;
    const height = bottom - top;
    return { beat, progress: height > 0 ? clamp01((anchorY - top) / height) : (anchorY >= bottom ? 1 : 0) };
  }

  function createStore(nBeats) {
    let state = { beat: 0, progress: 0 };
    const listeners = new Set();
    const normal = s => ({
      beat: Math.min(Math.max(Math.round(Number.isFinite(s.beat) ? s.beat : 0), 0), Math.max(nBeats - 1, 0)),
      progress: clamp01(s.progress),
    });
    return {
      get: () => ({ ...state }),
      set(next, source) {
        const n = normal(next);
        if (n.beat === state.beat && n.progress === state.progress) return;
        state = n;
        for (const fn of [...listeners]) fn({ ...state }, source);
      },
      subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    };
  }

  const lerpObject = (a, b, t) => {
    const out = {};
    for (const k of Object.keys(b)) out[k] = typeof b[k] === 'number' && typeof a[k] === 'number' ? lerp(a[k], b[k], t) : b[k];
    return out;
  };
  const layerOf = s => (s.level === 'micro' ? { level: 'micro', micro: s.micro } : { level: 'macro' });
  const sameLayer = (a, b) => a.level === b.level && a.micro === b.micro;

  /* The scene at (beat, progress). Numbers lerp from the previous beat's spec to this beat's;
     discrete choices switch to this beat's value as soon as progress is above 0. */
  function sceneAt(beats, beat, progress, reduced) {
    const cur = beats[beat].scene;
    const prev = (beat > 0 ? beats[beat - 1] : beats[beat]).scene;
    const p = clamp01(progress);
    const e = reduced ? (p > 0 ? 1 : 0) : ease(p / ARRIVE);

    const lit = {};
    for (const id of prev.lit) lit[id] = 1 - e;
    for (const id of cur.lit) lit[id] = (lit[id] === undefined ? 0 : lit[id]) + e;
    for (const id of Object.keys(lit)) {
      lit[id] = Math.min(lit[id], 1);
      if (lit[id] <= 0) delete lit[id];
    }

    const a = layerOf(prev), b = layerOf(cur);
    let layers;
    if (sameLayer(a, b) || e >= 1) layers = [{ ...b, alpha: 1 }];
    else if (e <= 0) layers = [{ ...a, alpha: 1 }];
    else layers = [{ ...a, alpha: 1 - e }, { ...b, alpha: e }];

    const arriving = p === 0 || (!reduced && p < ARRIVE);
    const flow = arriving
      ? { kind: prev.flow.kind, t: prev.flow.kind === 'none' ? 0 : 1 }
      : { kind: cur.flow.kind, t: clamp01((p - ARRIVE) / (1 - ARRIVE)) };
    if (reduced && p > 0) flow.kind = cur.flow.kind;

    return {
      beatId: beats[beat].id,
      fold: lerp(prev.fold, cur.fold, e),
      pass: e >= 1 || e > 0 ? cur.pass : prev.pass,
      camera: lerpObject(prev.camera, cur.camera, e),
      lit, layers, flow,
      toy: cur.toy,
    };
  }

  return { ARRIVE, clamp01, ease, beatFromScroll, createStore, sceneAt };
});
