/* A stand-in for speechSynthesis, injected before the page loads. It records what is spoken, plays
   utterances one after another like a real engine, and fires start, boundary and end events. */
module.exports = () => {
  const S = window.__speech = { spoken: [], ended: 0, cancels: 0, paused: false, active: null, timers: [] };
  const voices = [
    { name: 'Samantha', lang: 'en-US', localService: true, default: true }, { name: 'Thomas', lang: 'fr-FR', localService: true },
    { name: 'Daniel', lang: 'en-GB', localService: true }, { name: 'Zarvox', lang: 'en-US', localService: true },
  ];
  window.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; this.voice = null; this.rate = 1; this.pitch = 1; this.lang = ''; } };
  const queue = [];
  const next = () => {
    if (S.active || !queue.length || S.paused) return;
    const u = queue.shift(); S.active = u;
    S.spoken.push({ text: u.text, voice: u.voice && u.voice.name, rate: u.rate });
    const dur = Math.max(60, u.text.length * 2);
    S.timers.push(setTimeout(() => u.onstart && u.onstart({}), 5));
    for (let c = 8; c < u.text.length; c += 24) S.timers.push(setTimeout(() => u.onboundary && u.onboundary({ name: 'word', charIndex: c }), 5 + (dur * c) / u.text.length));
    S.timers.push(setTimeout(() => { S.active = null; S.ended++; u.onend && u.onend({}); next(); }, dur));
  };
  Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
    getVoices: () => voices, onvoiceschanged: null, addEventListener() {}, removeEventListener() {},
    speak(u) { queue.push(u); next(); },
    cancel() { S.timers.forEach(clearTimeout); S.timers = []; queue.length = 0; if (S.active) { const u = S.active; S.active = null; u.onerror && u.onerror({ error: 'canceled' }); } S.cancels++; },
    pause() { S.paused = true; }, resume() { S.paused = false; next(); },
    get speaking() { return !!S.active || queue.length > 0; }, get paused() { return S.paused; },
  } });
};
