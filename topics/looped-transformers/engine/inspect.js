/* Hover and click detail for the scene. Hover (or arrow keys on the focused scene) shows a tooltip; a click, tap or Enter
   pins the object in the inspector under the scene, with the arithmetic behind its value. Objects carry info {title, lines}. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopInspect = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- pure helpers ---------- */
  /* The id after `current` in `ids` (wrapping); with nothing focused, the first (forward) or last (backward). */
  function step(ids, current, dir) {
    if (!ids.length) return null;
    const i = ids.indexOf(current);
    if (i < 0) return dir > 0 ? ids[0] : ids[ids.length - 1];
    return ids[(i + dir + ids.length) % ids.length];
  }
  /* Top-left of a tooltip of `size` next to a pointer, flipped to the other side near the right and bottom edges, kept inside `box`. */
  function place(pointer, size, box, gap = 14) {
    let x = pointer.x + gap, y = pointer.y + gap;
    if (x + size.w > box.w) x = pointer.x - gap - size.w;
    if (y + size.h > box.h) y = pointer.y - gap - size.h;
    return { x: Math.max(0, Math.min(x, box.w - size.w)), y: Math.max(0, Math.min(y, box.h - size.h)) };
  }
  /* A tooltip position beside an object, never over it: below, then above, then to the right, then to the left; null if none fits. */
  function placeAround(rect, size, box, gap = 8) {
    const clampX = x => Math.max(0, Math.min(x, box.w - size.w)), clampY = y => Math.max(0, Math.min(y, box.h - size.h));
    const cx = (rect.x0 + rect.x1) / 2 - size.w / 2, cy = (rect.y0 + rect.y1) / 2 - size.h / 2;
    const tries = [{ x: clampX(cx), y: rect.y1 + gap }, { x: clampX(cx), y: rect.y0 - gap - size.h }, { x: rect.x1 + gap, y: clampY(cy) }, { x: rect.x0 - gap - size.w, y: clampY(cy) }];
    const inside = p => p.x >= 0 && p.y >= 0 && p.x + size.w <= box.w && p.y + size.h <= box.h;
    const over = p => p.x < rect.x1 && p.x + size.w > rect.x0 && p.y < rect.y1 && p.y + size.h > rect.y0;
    return tries.find(p => inside(p) && !over(p)) || null;
  }
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const html = info => `<strong>${esc(info.title)}</strong>` + info.lines.map(l => `<span>${esc(l)}</span>`).join('');

  /* ---------- the DOM part ---------- */
  function bind({ scene, store, stage }) {
    const box = scene.canvas().parentElement;                    // the stage box outlives a canvas swapped for the 2D fallback
    const tip = Object.assign(document.createElement('div'), { id: 'tip', hidden: true });
    tip.setAttribute('role', 'presentation');
    const ring = Object.assign(document.createElement('div'), { id: 'ring', hidden: true });
    const live = Object.assign(document.createElement('div'), { id: 'inspect-live', className: 'sr' });   // what keyboard and screen-reader users hear as they move
    live.setAttribute('aria-live', 'polite');
    const ins = Object.assign(document.createElement('div'), { id: 'inspector', className: 'inspector', hidden: true });
    ins.setAttribute('aria-live', 'polite');
    const hint = Object.assign(document.createElement('p'), { className: 'hint', textContent: 'Hover or tap anything in the scene for its exact value. Click to pin it here. Keyboard: focus the scene, then arrow keys and Enter.' });
    const body = Object.assign(document.createElement('div'), { className: 'pin' });
    const clear = Object.assign(document.createElement('button'), { type: 'button', className: 'unpin', textContent: 'Clear', hidden: true });
    ins.append(hint, body, clear);
    box.append(tip, ring, live);
    box.after(ins);
    const canvasOf = () => scene.canvas();
    let pinned = null, focused = null, hovered = null, tipFor = null, tipAt = null, lastHtml = '';   // object ids; where the tip was anchored

    const infoOf = id => { const r = id && scene.rectOf(id); return r ? r.info : null; };
    const offset = () => { const r = canvasOf().getBoundingClientRect(), b = box.getBoundingClientRect(); return { x: r.left - b.left, y: r.top - b.top }; };

    function hideTip() { tip.hidden = true; ring.hidden = true; hovered = null; tipFor = null; }
    /* Show the tooltip for an object: at the pointer when there is one, otherwise beside the object. Returns false when there is no room beside it. */
    function showTip(id, at) {
      const info = infoOf(id); if (!info) { hideTip(); return false; }
      tip.innerHTML = html(info); tip.hidden = false; tipFor = id; tipAt = at || null;
      const r = scene.rectOf(id), o = offset(), size = { w: tip.offsetWidth, h: tip.offsetHeight }, b = { w: box.clientWidth, h: box.clientHeight };
      ring.style.cssText = `left:${(r.x0 + o.x).toFixed(1)}px;top:${(r.y0 + o.y).toFixed(1)}px;width:${Math.max(4, r.x1 - r.x0).toFixed(1)}px;height:${Math.max(4, r.y1 - r.y0).toFixed(1)}px`;
      ring.hidden = false;
      const p = at ? place(at, size, b, 14) : placeAround({ x0: r.x0 + o.x, y0: r.y0 + o.y, x1: r.x1 + o.x, y1: r.y1 + o.y }, size, b);
      if (!p) { tip.hidden = true; return false; }
      tip.style.left = p.x.toFixed(1) + 'px'; tip.style.top = p.y.toFixed(1) + 'px';
      return true;
    }
    function render() {
      const info = infoOf(pinned), h = info ? html(info) : '';
      if (h !== lastHtml) { lastHtml = h; body.innerHTML = h; }                       // an unchanged pin is left alone, so a screen reader does not hear it again
      clear.hidden = !info; ins.classList.toggle('has-pin', !!info);
    }
    function unpin() { pinned = null; render(); }
    function pin(id) { pinned = id; render(); }
    function reset() { hideTip(); focused = null; pinned = null; render(); }
    const announce = id => { const i = infoOf(id); live.textContent = i ? `${i.title}. ${i.lines.join('. ')}` : ''; };

    /* mouse and touch */
    const local = e => { const r = canvasOf().getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    box.addEventListener('pointermove', e => {
      if (e.pointerType === 'touch') return;
      const p = local(e), h = scene.hit(p.x, p.y), o = offset();
      if (!h) return hideTip();
      focused = h.id; hovered = h.id;
      showTip(h.id, { x: p.x + o.x, y: p.y + o.y });
    });
    box.addEventListener('pointerleave', hideTip);
    box.addEventListener('click', e => {
      if (e.target.closest && e.target.closest('#tip')) return;
      const p = local(e), h = scene.hit(p.x, p.y);
      if (h) { pin(h.id); focused = h.id; hideTip(); } else unpin();                 // a tap pins; the pinned text is the detail, so no tooltip is left behind
    });
    clear.addEventListener('click', () => { unpin(); canvasOf().focus(); });

    /* keyboard */
    box.addEventListener('keydown', e => {
      if (e.target !== canvasOf()) return;
      const list = scene.infoList().map(r => r.id);
      let next = null;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = step(list, focused, +1);
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = step(list, focused, -1);
      else if (e.key === 'Home') next = list[0] || null;
      else if (e.key === 'End') next = list[list.length - 1] || null;
      else if ((e.key === 'Enter' || e.key === ' ') && focused) { e.preventDefault(); e.stopPropagation(); pin(focused); announce(focused); return; }
      else if (e.key === 'Escape') { hideTip(); focused = null; unpin(); live.textContent = ''; return; }
      else return;
      e.preventDefault(); e.stopPropagation();                                        // these keys belong to the scene now, not to the article's own scrolling
      if (next) { focused = next; announce(next); if (!showTip(next)) pin(next); }   // no room for a tooltip beside it: show the detail in the inspector instead
    });
    box.addEventListener('focusout', e => { if (e.target === canvasOf() && hovered === null) hideTip(); });

    /* A scene with no detail (Articles A and B so far) shows no hint and is not a tab stop. A new beat starts clean; a redraw of the same beat keeps the pin, refreshes its text and the tooltip, and drops either if the object is gone. */
    let beat = store.get().beat;
    store.subscribe(() => { const b = store.get().beat; if (b !== beat) { beat = b; reset(); } });
    scene.onFrame(() => {
      const c = canvasOf(), has = scene.infoList().length > 0;
      ins.hidden = !has;
      if (has && !c.hasAttribute('tabindex')) { c.tabIndex = 0; c.setAttribute('aria-describedby', 'inspector'); c.setAttribute('role', 'application'); c.setAttribute('aria-roledescription', 'interactive diagram'); }
      if (!has && c.hasAttribute('tabindex')) { c.removeAttribute('tabindex'); c.removeAttribute('aria-describedby'); c.setAttribute('role', 'img'); c.removeAttribute('aria-roledescription'); }
      if (pinned && !infoOf(pinned)) unpin(); else if (pinned) render();
      if (focused && !infoOf(focused)) { focused = null; hideTip(); }
      else if (tipFor && !tip.hidden) { if (!showTip(tipFor, tipAt)) { hideTip(); } }
    });
    return { pin, unpin, reset };
  }
  return { step, place, placeAround, html, bind };
});
