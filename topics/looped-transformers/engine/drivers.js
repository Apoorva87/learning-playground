/* Three ways to move through an article, all writing the same (beat, progress) pair.
     Scroll  the scroll position is the state (default).
     Step    Next/Prev and the arrow keys; the article scrolls to the beat and the scene tweens.
     Play    Step on a timer, at reading pace.
   Text and scene cannot drift because there is only one state. If the reader scrolls by hand
   while Step or Play is running, control returns to Scroll mode and the state is read back
   from where they scrolled to. */
(function () {
  'use strict';
  const S = window.LoopState;
  const SCROLL_KEYS = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' ']);
  const LAND = 0.4;                          // how far into a beat's section we land: the pose has arrived

  function create({ store, sections, beats, config, stage }) {
    const $ = id => document.getElementById(id);
    const radios = { scroll: $('mode-scroll'), step: $('mode-step'), play: $('mode-play') };
    const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const narrow = window.matchMedia('(max-width: 1040px)');
    const words = sections.map(s => (s.textContent.match(/\S+/g) || []).length);

    const d = { mode: 'scroll', speed: parseFloat(($('speed') || {}).value) || 1 };
    let interacted = false;                                   // set once the reader scrolls, taps or presses keys by hand
    const isControl = t => !!(t && t.closest && t.closest('#controls, .controls, button, select, input, textarea, a, label, summary'));
    let runId = 0, programmaticUntil = 0, frame = 0, scrollJob = 0, lastSetY = -1;

    /* The viewport line that decides which beat is current. On a phone the scene is pinned
       above the text, so the line sits in the visible part below it. */
    d.anchorY = () => {
      if (narrow.matches && stage) { const b = stage.getBoundingClientRect().bottom; return b + 0.3 * (innerHeight - b); }
      return innerHeight * 0.4;
    };
    const geometry = () => ({
      tops: sections.map(s => s.getBoundingClientRect().top),
      end: sections[sections.length - 1].getBoundingClientRect().bottom,
    });
    const fromScroll = () => { const g = geometry(); return S.beatFromScroll(g.tops, d.anchorY(), g.end); };

    /* ---------- scrolling the article ---------- */
    const targetY = (i, frac = LAND) => {
      const r = sections[i].getBoundingClientRect();
      return Math.max(0, scrollY + r.top + frac * r.height - d.anchorY());
    };
    function scrollTo(y, ms) {
      cancelAnimationFrame(scrollJob);
      programmaticUntil = performance.now() + ms + 40;
      if (reduced() || ms <= 0) { window.scrollTo(0, y); lastSetY = scrollY; return; }
      const y0 = scrollY, t0 = performance.now();
      const tick = now => {
        const k = S.ease((now - t0) / ms);
        window.scrollTo(0, y0 + (y - y0) * k); lastSetY = scrollY;
        if (now - t0 < ms) scrollJob = requestAnimationFrame(tick);
      };
      scrollJob = requestAnimationFrame(tick);
    }

    /* ---------- tweens ---------- */
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    function tween(beat, from, to, ms, id) {
      return new Promise(resolve => {
        if (reduced() || ms <= 0) { if (id === runId) store.set({ beat, progress: to }, 'tween'); return resolve(); }
        const t0 = performance.now();
        const tick = now => {
          if (id !== runId) return resolve();
          const k = Math.min((now - t0) / ms, 1);
          store.set({ beat, progress: from + (to - from) * k }, 'tween');
          if (k < 1) requestAnimationFrame(tick); else resolve();
        };
        requestAnimationFrame(tick);
      });
    }
    const tweenMs = () => config.tweenMs / d.speed;

    /* Show beat i: scroll to it and let the scene arrive. */
    async function stepTo(i, id) {
      i = Math.min(Math.max(i, 0), beats.length - 1);
      scrollTo(targetY(i), 0);                                  // a cut, in the same task as the state change, so text and scene never disagree
      store.set({ beat: i, progress: 0 }, 'step');
      await tween(i, 0, 1, tweenMs(), id);
    }

    /* ---------- mode switching ---------- */
    function cancel() { runId++; cancelAnimationFrame(scrollJob); }
    function setMode(mode) {
      if (!radios[mode]) return;
      cancel();
      d.mode = mode;
      for (const [k, el] of Object.entries(radios)) if (el) el.checked = k === mode;
      if (mode === 'scroll') store.set(fromScroll(), 'scroll');
      if (mode === 'play') runPlay(runId);
    }
    async function runPlay(id) {
      const last = beats.length - 1;
      while (id === runId) {
        const { beat, progress } = store.get();
        if (progress < 1) await tween(beat, progress, 1, tweenMs() * (1 - progress), id);
        if (id !== runId) return;
        if (beat >= last) { setMode('step'); return; }          // the end: stay where we are
        await sleep((config.test ? config.dwellMs : Math.min(Math.max(config.dwellMs, words[beat] / 6 * 1000), 25000)) / d.speed);   // about 360 words a minute, between 2 and 25 seconds, scaled by the speed control
        if (id !== runId) return;
        await stepTo(beat + 1, id);
      }
    }
    d.setMode = setMode;
    d.play = () => setMode('play');
    d.pause = () => { if (d.mode === 'play') setMode('step'); };
    d.goTo = (id, instant = true) => {
      const i = beats.findIndex(b => b.id === id); if (i < 0) return;
      cancel(); setMode('scroll');
      scrollTo(targetY(i), instant ? 0 : 400);
      store.set({ beat: i, progress: LAND }, 'goto');
    };
    d.next = () => move(1);
    d.prev = () => move(-1);
    function move(dir) {
      const i = Math.min(Math.max(store.get().beat + dir, 0), beats.length - 1);
      if (d.mode === 'scroll') { scrollTo(targetY(i), 400); return; }
      if (d.mode === 'play') d.pause();
      runId++; const id = runId; stepTo(i, id);
    }
    d.resync = () => { if (d.mode === 'scroll') store.set(fromScroll(), 'scroll'); };

    /* ---------- events ---------- */
    addEventListener('scroll', () => {
      const ours = performance.now() < programmaticUntil || Math.abs(scrollY - lastSetY) < 2;
      if (!ours) interacted = true;
      if (d.mode !== 'scroll') {
        if (ours) return;                                       // our own scrolling, even when a slow frame delays its event
        setMode('scroll'); return;                              // the reader took over
      }
      if (frame) return;
      frame = requestAnimationFrame(() => { frame = 0; store.set(fromScroll(), 'scroll'); });
    }, { passive: true });
    addEventListener('wheel', () => { interacted = true; if (d.mode !== 'scroll') setMode('scroll'); }, { passive: true });
    addEventListener('touchstart', e => { if (isControl(e.target)) return; interacted = true; if (d.mode !== 'scroll') setMode('scroll'); }, { passive: true });
    /* A click anywhere in the page except the controls takes over from Play. */
    addEventListener('pointerdown', e => { if (d.mode === 'play' && !(e.target.closest && e.target.closest('#controls, .controls'))) setMode('scroll'); });
    addEventListener('resize', () => d.resync());
    addEventListener('keydown', e => {
      if (e.target.closest && e.target.closest('input,select,textarea,[contenteditable]')) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); d.next(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); d.prev(); }
      else if (SCROLL_KEYS.has(e.key) && !isControl(e.target)) { interacted = true; if (d.mode !== 'scroll') setMode('scroll'); }
    });
    for (const [mode, el] of Object.entries(radios)) el && el.addEventListener('change', () => el.checked && setMode(mode));
    $('next') && $('next').addEventListener('click', d.next);
    $('prev') && $('prev').addEventListener('click', d.prev);
    $('speed') && $('speed').addEventListener('change', e => { d.speed = parseFloat(e.target.value) || 1; });

    /* ---------- reflect the state in the page ---------- */
    const title = $('beat-title');
    store.subscribe(({ beat }) => {
      sections.forEach((s, i) => s.setAttribute('aria-current', i === beat ? 'true' : 'false'));
      if (title) title.textContent = beats[beat].title;
      const hash = '#b-' + beats[beat].id;
      if (location.hash !== hash) history.replaceState(null, '', hash);
    });
    addEventListener('hashchange', () => {
      const m = /^#b-(.+)$/.exec(location.hash);
      if (m && m[1] !== beats[store.get().beat].id) d.goTo(m[1]);
    });
    const m = /^#b-(.+)$/.exec(location.hash);
    if (m) d.goTo(m[1]); else store.set(fromScroll(), 'scroll');
    addEventListener('load', () => { if (m && d.mode === 'scroll' && !interacted) d.goTo(m[1]); else d.resync(); });   // late layout shifts re-seat a deep link, but never move a reader who has scrolled on
    return d;
  }
  window.LoopDrivers = { create, LAND };
})();
