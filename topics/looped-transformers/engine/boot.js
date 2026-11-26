/* Wires one article page: store -> scene. Drivers, terms and math attach to the same store.
   window.LOOP is the hook the browser tests use. */
(function () {
  'use strict';
  const S = window.LoopState, params = new URLSearchParams(location.search);
  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* KaTeX comes from a pinned CDN. Without it the TeX source stays readable. */
  function renderMath() {
    const go = () => {
      if (typeof window.renderMathInElement !== 'function') return;
      for (const el of document.querySelectorAll('.math, .story, .intuition'))
        window.renderMathInElement(el, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
      if (window.LOOP && window.LOOP.drivers) window.LOOP.drivers.resync();
    };
    if (document.readyState === 'complete') go(); else addEventListener('load', go);
  }

  function start({ beats }) {
    const sections = [...document.querySelectorAll('section.beat')];
    if (sections.length !== beats.length || sections.some((s, i) => s.dataset.beat !== beats[i].id))
      throw new Error('Sections and beats differ: ' + sections.map(s => s.dataset.beat).join(',') + ' vs ' + beats.map(b => b.id).join(','));
    const config = { test: params.has('test'), tweenMs: params.has('test') ? 150 : 2200, dwellMs: params.has('test') ? 300 : 2000, renderer: params.get('renderer') || 'auto' };
    const store = S.createStore(beats.length);
    const scene = window.LoopScene.create({ canvas: document.getElementById('scene'), labelLayer: document.getElementById('labels'), beats, renderer: config.renderer });
    const reduced = () => reducedQuery.matches;
    const render = () => {
      const s = store.get(); scene.show(S.sceneAt(beats, s.beat, s.progress, reduced()));
      scene.canvas().setAttribute('aria-label', `3D scene for “${beats[s.beat].title}”`);       // screen readers get the beat the scene is on
    };
    store.subscribe(render);
    reducedQuery.addEventListener && reducedQuery.addEventListener('change', render);
    render();

    const LOOP = window.LOOP = {
      ready: true, beats, sections, config, store, scene: () => scene.getState(), sceneApi: scene,
      beatCount: beats.length, reduced,
      state: () => ({ ...store.get(), mode: LOOP.drivers ? LOOP.drivers.mode : 'scroll' }),
    };
    LOOP.drivers = window.LoopDrivers.create({ store, sections, beats, config, stage: document.getElementById('stage') });
    window.LoopTerms.bind({ root: document.getElementById('article'), beats, scene, store, canvas: scene.canvas });
    renderMath();
    if (window.LoopRecorded) window.LoopRecorded.bind(document.getElementById('article'));
    if (window.LoopCharts) window.LoopCharts.renderAll();
    if (window.LoopToy) window.LoopToy.bind({ scene, root: document.getElementById('article') });
    document.dispatchEvent(new CustomEvent('loop:ready'));
    return LOOP;
  }
  window.LoopArticle = { start };
})();
