/* Links words in the text to objects in the scene, in both directions.
   A term is <span class="term" data-term="gate">; the beat's `terms` map says which scene
   objects it names. Hover or focus a term: its objects light up. Hover an object: every term
   that names it, in the beat on screen, gets .term-active. */
(function () {
  'use strict';
  function bind({ root, beats, scene, store, canvas }) {
    const Layout = window.LoopLayout;
    const byId = Object.fromEntries(beats.map(b => [b.id, b]));
    const idsOf = (termEl) => {
      const sec = termEl.closest('section.beat'); const b = sec && byId[sec.dataset.beat];
      return b && b.terms && b.terms[termEl.dataset.term] ? [...new Set(b.terms[termEl.dataset.term].flatMap(Layout.idsFor))] : [];
    };
    const terms = [...root.querySelectorAll('.term[data-term]')];
    terms.forEach(t => { if (!t.hasAttribute('tabindex')) t.tabIndex = 0; });

    const on = e => { const t = e.target.closest && e.target.closest('.term[data-term]'); if (t) scene.highlight(idsOf(t)); };
    const off = e => { if (e.target.closest && e.target.closest('.term[data-term]')) scene.highlight([]); };
    root.addEventListener('mouseover', on); root.addEventListener('focusin', on);
    root.addEventListener('mouseout', off); root.addEventListener('focusout', off);

    let active = null;
    function mark(id) {
      const here = beats[store.get().beat].id;                 // only the words of the beat on screen
      terms.forEach(t => t.classList.toggle('term-active', id !== null && t.closest('section.beat').dataset.beat === here && idsOf(t).includes(id)));
    }
    const box = canvas().parentElement;                        // the stage box outlives a canvas swapped for the 2D fallback
    box.addEventListener('pointermove', e => {
      const r = canvas().getBoundingClientRect();
      const id = scene.pick(e.clientX - r.left, e.clientY - r.top);
      if (id !== active) { active = id; mark(id); }
    });
    box.addEventListener('pointerleave', () => { active = null; mark(null); });
    return { idsOf };
  }
  window.LoopTerms = { bind };
})();
