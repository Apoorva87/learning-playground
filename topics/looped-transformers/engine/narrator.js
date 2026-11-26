/* Voice narration with the browser's built-in speech (speechSynthesis; on a Mac this is the same
   set of voices as the system's). It reads a beat sentence by sentence, says each display equation
   from its hand-written data-speak text, lights up the term and sentence being spoken, and when it
   drives a beat (Step or Play) lets the scene's progress follow the voice.
   Controls: #narrate, #voice, #rate, #narrate-pause, #voice-note. */
(function () {
  'use strict';
  const Sp = window.LoopSpeech;
  const KEY = { voice: 'loop.voice', rate: 'loop.rate' };
  const PREFERRED = ['Samantha', 'Ava', 'Allison', 'Daniel', 'Karen', 'Moira', 'Serena'];
  const remember = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode: fine */ } };
  const recall = k => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  function create({ store, beats, sections, scene, drivers, getTerms, config }) {
    const $ = id => document.getElementById(id);
    const ui = { on: $('narrate'), voice: $('voice'), rate: $('rate'), pause: $('narrate-pause'), note: $('voice-note') };
    const synth = window.speechSynthesis, supported = !!(synth && window.SpeechSynthesisUtterance);
    const n = { enabled: false, speaking: false, supported };
    let token = 0, idleTimer = 0, speakingBeat = -1, highlighted = false, paused = false;
    const ARRIVE = window.LoopState.ARRIVE;

    /* ---------- the script of a beat: sentences, each with its DOM range and the terms it names ---------- */
    function blocksOf(section) {
      return [...section.querySelectorAll('h2, p, li')].filter(b => !b.closest('.toy, .viz, figure, table, details, .cap, [data-recorded], .howto'));
    }
    function segmentsOf(block) {
      const segs = [];
      const pushText = node => {
        const text = node.nodeValue, re = /\\\(([\s\S]*?)\\\)|\$\$([\s\S]*?)\$\$/g;       // raw TeX, when KaTeX has not run
        let last = 0, m;
        while ((m = re.exec(text))) {
          if (m.index > last) segs.push({ text: text.slice(last, m.index), node, start: last, end: m.index });
          segs.push({ text: ' ' + (m[2] ? '' : Sp.texToSpeech(m[1])) + ' ', node, start: m.index, end: m.index + m[0].length });
          last = m.index + m[0].length;
        }
        if (last < text.length) segs.push({ text: text.slice(last), node, start: last, end: text.length });
      };
      (function walk(parent) {
        for (const c of parent.childNodes) {
          if (c.nodeType === 3) pushText(c);
          else if (c.nodeType === 1) {
            if (c.classList.contains('eq')) segs.push({ text: ' ' + (c.dataset.speak || '') + ' ', el: c });
            else if (c.classList.contains('katex')) { const a = c.querySelector('annotation'); segs.push({ text: ' ' + Sp.texToSpeech(a ? a.textContent : '') + ' ', el: c }); }
            else if (!/^(SCRIPT|STYLE)$/.test(c.tagName)) walk(c);
          }
        }
      })(block);
      return segs;
    }
    function sentencesOf(block) {
      const segs = segmentsOf(block); let full = '', at = 0;
      for (const s of segs) { s.from = at; at += s.text.length; full += s.text; }
      const out = [];
      for (const r of Sp.splitSentences(full)) {
        const text = full.slice(r.start, r.end).replace(/\s+/g, ' ').trim();
        if (!/[A-Za-z0-9]/.test(text)) continue;
        const a = segs.find(s => s.from + s.text.length > r.start && s.text.trim()), b = [...segs].reverse().find(s => s.from < r.end && s.text.trim());
        if (!a || !b) continue;
        const range = document.createRange();
        if (a.node) range.setStart(a.node, Math.min(a.end, a.start + Math.max(0, r.start - a.from))); else range.setStartBefore(a.el);
        if (b.node) range.setEnd(b.node, Math.min(b.end, b.start + Math.max(0, r.end - b.from))); else range.setEndAfter(b.el);
        out.push({ text, range, block });
      }
      return out;
    }
    function scriptOf(i) { return blocksOf(sections[i]).flatMap(sentencesOf); }
    n.scriptOf = scriptOf;

    /* ---------- voices ---------- */
    function voices() { return synth ? synth.getVoices() : []; }
    function fillVoices() {
      if (!supported) return;
      const all = voices(), keep = ui.voice.value || recall(KEY.voice) || '';
      const en = all.filter(v => /^en/i.test(v.lang)), rest = all.filter(v => !/^en/i.test(v.lang));
      ui.voice.innerHTML = '<option value="">System default</option>' + [...en, ...rest].map(v => `<option value="${v.name.replace(/"/g, '&quot;')}">${v.name} (${v.lang})</option>`).join('');
      let pick = [...ui.voice.options].some(o => o.value === keep) ? keep : '';
      if (!pick && !recall(KEY.voice)) pick = PREFERRED.find(p => en.some(v => v.name === p)) || '';
      ui.voice.value = pick;
    }
    const chosen = () => voices().find(v => v.name === ui.voice.value) || null;

    /* ---------- speaking ---------- */
    function clearHighlights() {
      if (window.CSS && CSS.highlights) CSS.highlights.delete('speaking');
      document.querySelectorAll('.speaking').forEach(e => e.classList.remove('speaking'));
      if (highlighted) { scene.highlight([]); highlighted = false; }
    }
    function mark(sentence) {
      clearHighlights();
      if (window.CSS && CSS.highlights && window.Highlight) CSS.highlights.set('speaking', new Highlight(sentence.range)); else sentence.block.classList.add('speaking');
      const terms = getTerms ? getTerms() : null;
      if (terms) {
        const els = [...sentence.block.querySelectorAll('.term[data-term]')].filter(t => sentence.range.intersectsNode(t));
        const ids = [...new Set(els.flatMap(t => terms.idsOf(t)))];
        if (ids.length) { scene.highlight(ids); highlighted = true; }
      }
    }
    function follow(sentence) {                                 // keep the spoken sentence on screen while Step or Play drives
      const r = sentence.range.getBoundingClientRect();
      if (r.bottom > innerHeight * 0.88 || r.top < drivers.anchorY() * 0.4) drivers.scrollBy(r.top - drivers.anchorY() * 0.7);
    }

    /* Speak beat i to the end. Resolves true when finished, false when stopped. drive: move the scene's progress with the voice. */
    function speakBeat(i, { drive = false, alive = () => true } = {}) {
      if (!supported) return Promise.resolve(false);
      stop(true);
      const my = ++token, script = scriptOf(i), total = script.reduce((a, s) => a + s.text.length, 0) || 1;
      speakingBeat = i; n.speaking = true; paused = false; refreshButton();
      const set = frac => { if (drive && alive()) store.set({ beat: i, progress: ARRIVE + (1 - ARRIVE) * Math.min(frac, 1) }, 'narrate'); };
      return new Promise(resolve => {
        let k = 0, done = 0;
        const finish = ok => { if (my === token) { n.speaking = false; speakingBeat = -1; clearHighlights(); refreshButton(); } resolve(ok); };
        const next = () => {
          if (my !== token || !alive()) return finish(false);
          if (k >= script.length) { set(1); return finish(true); }
          const s = script[k++], u = new SpeechSynthesisUtterance(s.text), before = done;
          const v = chosen(); if (v) { u.voice = v; u.lang = v.lang; }
          u.rate = parseFloat(ui.rate.value) || 1;
          u.onstart = () => { if (my !== token) return; mark(s); if (drive) follow(s); set(before / total); };
          u.onboundary = e => { if (my === token && typeof e.charIndex === 'number') set((before + e.charIndex) / total); };
          u.onend = () => { done += s.text.length; if (my === token) next(); };
          u.onerror = () => { if (my === token) finish(false); };
          synth.speak(u);
        };
        next();
      });
    }
    function stop(keepIdle) {
      token++; clearTimeout(idleTimer);
      if (supported && (n.speaking || synth.speaking)) synth.cancel();
      n.speaking = false; speakingBeat = -1; paused = false; clearHighlights(); refreshButton();
    }
    n.speakBeat = speakBeat; n.stop = () => stop();

    /* ---------- Scroll mode: read the beat the reader has settled on ---------- */
    function scheduleIdle() {
      clearTimeout(idleTimer);
      if (!n.enabled || drivers.mode !== 'scroll') return;
      idleTimer = setTimeout(() => { if (n.enabled && drivers.mode === 'scroll' && !n.speaking) speakBeat(store.get().beat); }, config.test ? 250 : 800);
    }
    store.subscribe((st, src) => {
      if (!n.enabled || drivers.mode !== 'scroll' || src === 'narrate') return;
      if (n.speaking && st.beat !== speakingBeat) stop();
      scheduleIdle();
    });

    /* ---------- controls ---------- */
    function refreshButton() { if (ui.pause) ui.pause.textContent = paused ? 'Resume' : n.speaking ? 'Pause' : 'Read beat'; }
    n.setEnabled = on => {
      n.enabled = !!on && supported;
      if (ui.on) ui.on.checked = n.enabled;
      if (!n.enabled) stop(); else if (drivers.mode === 'scroll') scheduleIdle(); else if (drivers.mode === 'play') drivers.setMode('play');
    };
    if (!supported) {
      for (const el of [ui.on, ui.voice, ui.rate, ui.pause]) if (el) el.disabled = true;
      if (ui.note) ui.note.textContent = 'Voice narration is not available in this browser.';
      return n;
    }
    fillVoices();
    if (synth.addEventListener) synth.addEventListener('voiceschanged', fillVoices); else synth.onvoiceschanged = fillVoices;
    const savedRate = recall(KEY.rate); if (savedRate && [...ui.rate.options].some(o => o.value === savedRate)) ui.rate.value = savedRate;
    ui.voice.addEventListener('change', () => { remember(KEY.voice, ui.voice.value); });
    ui.rate.addEventListener('change', () => remember(KEY.rate, ui.rate.value));
    ui.on.addEventListener('change', () => n.setEnabled(ui.on.checked));
    ui.pause.addEventListener('click', () => {
      if (n.speaking) { if (paused) { synth.resume(); paused = false; } else { synth.pause(); paused = true; } refreshButton(); }
      else { if (!n.enabled) n.setEnabled(true); speakBeat(store.get().beat); }
    });
    refreshButton();
    return n;
  }
  window.LoopNarrator = { create };
})();
