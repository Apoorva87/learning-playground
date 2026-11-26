/* Pure helpers for narration: say a TeX fragment in words, and cut text into sentences small
   enough for speechSynthesis (long utterances stop early in some browsers). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopSpeech = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const WORDS = { in: 'in', times: 'times', cdot: 'dot', odot: 'element-wise times', approx: 'approximately', le: 'at most', ge: 'at least',
    to: 'to', rightarrow: 'to', leftarrow: 'gets', sum: 'the sum of', tanh: 'tanh', ln: 'log', max: 'max', cup: 'union', bmod: 'mod',
    Longrightarrow: 'so', leq: 'at most', geq: 'at least', ell: 'ell', pi: 'pi', mid: 'given' };
  const GREEK = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'sigma', 'tau', 'rho', 'theta', 'lambda', 'mu', 'phi', 'omega'];

  /* A best-effort reading of a short TeX fragment. Display equations carry hand-written text instead. */
  function texToSpeech(tex) {
    let s = String(tex);
    s = s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, ' ');
    s = s.replace(/\\mathbb\{R\}\^\{\[([^\]]*)\]\}/g, (_, d) => ' R to the shape ' + d.replace(/\\[,;! ]/g, ' ').split(',').map(x => x.trim()).filter(Boolean).join(' by ') + ' ');
    s = s.replace(/\\(?:text|mathrm|operatorname|mathcal|mathbb|mathbf|textbf)\{([^{}]*)\}/g, ' $1 ');
    s = s.replace(/\\(?:big|Big|bigg|Bigg|left|right|displaystyle|tfrac|dfrac)(?![a-zA-Z])/g, ' ');
    s = s.replace(/\\(?:quad|qquad|,|;|!|:| )/g, ' ');
    for (let i = 0; i < 3; i++) s = s.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, ' $1 over $2 ');
    s = s.replace(/\\sqrt\{([^{}]*)\}/g, ' root $1 ');
    s = s.replace(/\^\{([^{}]*)\}/g, ' to the $1 ').replace(/\^(\w)/g, ' to the $1 ');
    s = s.replace(/_\{([^{}]*)\}/g, ' sub $1 ').replace(/_(\w|\\ell)/g, ' sub $1 ');
    s = s.replace(/\\([a-zA-Z]+)/g, (_, w) => ' ' + (WORDS[w] || (GREEK.includes(w) ? w : w)) + ' ');
    s = s.replace(/\+/g, ' plus ').replace(/(?<=\s|\w|\})-(?=\s*[\w{])/g, ' minus ').replace(/=/g, ' equals ').replace(/</g, ' less than ').replace(/>/g, ' greater than ').replace(/\//g, ' over ');
    s = s.replace(/[{}$|\\&]/g, ' ').replace(/[[\]]/g, ' ').replace(/\s+/g, ' ');
    return s.replace(/ ([,.;:)])/g, '$1').trim();
  }

  /* Offsets [start, end) of sentence-sized pieces that cover the text exactly, in order. */
  function splitSentences(text, max = 220) {
    const marks = [0];
    const re = /[.!?]["”)’]?(?=\s+["“(\\]?[A-Z0-9])/g;
    let m;
    while ((m = re.exec(text))) {
      const before = text.slice(Math.max(0, m.index - 4), m.index + 1);
      if (/(?:e\.g|i\.e|vs|etc|cf|Fig|No)\.$/.test(before) || /\b[A-Z]\.$/.test(text.slice(Math.max(0, m.index - 2), m.index + 1))) continue;
      let end = m.index + m[0].length;
      while (/\s/.test(text[end] || '')) end++;
      marks.push(end);
    }
    marks.push(text.length);
    const out = [];
    for (let i = 0; i + 1 < marks.length; i++) {
      let a = marks[i], b = marks[i + 1];
      if (b <= a) continue;
      while (b - a > max) {                                     // cut at the last comma or semicolon before the limit
        let cut = Math.max(text.lastIndexOf(', ', a + max), text.lastIndexOf('; ', a + max));
        if (cut <= a + 20) cut = text.lastIndexOf(' ', a + max);
        if (cut <= a + 20) cut = a + max;
        cut += (text[cut] === ',' || text[cut] === ';') ? 2 : 1;
        out.push({ start: a, end: cut }); a = cut;
      }
      out.push({ start: a, end: b });
    }
    return out;
  }
  return { texToSpeech, splitSentences };
});
