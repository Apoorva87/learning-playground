/* Shared model for the 2D and 3D views: unpack the graph, apply a machine model,
   find the critical path, and encode node size. Everything here is derived from
   data.js (shape-based counts). Nothing is measured. */
(() => {
  'use strict';
  const D = window.QWEN_GRAPH;

  // Nominal vendor peak numbers, used only to place each op on a roofline.
  const MACHINES = {
    h100: { name: 'H100 SXM (spec-sheet peak)', flops: 989e12, bw: 3.35e12 },
    m4: { name: 'Laptop GPU class (546 GB/s, ~16 TFLOP/s assumed)', flops: 16e12, bw: 546e9 },
  };

  // Stage colours: fixed order = fixed palette slot (validated, see README).
  const PALETTE = {
    light: ['#257de0', '#f47230', '#00b894', '#f2a600', '#ee7fac', '#008a17', '#6a4ce2', '#8a8a8a'],
    dark: ['#338ced', '#e76624', '#00a888', '#d28e05', '#e24d8b', '#0aac23', '#9887ef', '#8a8a8a'],
  };
  const isDark = () => {
    const t = document.documentElement.dataset.theme;
    if (t) return t === 'dark';
    return matchMedia('(prefers-color-scheme: dark)').matches;
  };
  const colors = () => (isDark() ? PALETTE.dark : PALETTE.light);

  const fmt = {
    flops(v) {
      if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TFLOP';
      if (v >= 1e9) return (v / 1e9).toFixed(2) + ' GFLOP';
      if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MFLOP';
      if (v >= 1e3) return (v / 1e3).toFixed(1) + ' kFLOP';
      return Math.round(v) + ' FLOP';
    },
    bytes(v) {
      if (v >= 1e9) return (v / 1e9).toFixed(2) + ' GB';
      if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MB';
      if (v >= 1e3) return (v / 1e3).toFixed(1) + ' kB';
      return Math.round(v) + ' B';
    },
    time(s) {
      if (s >= 1) return s.toFixed(2) + ' s';
      if (s >= 1e-3) return (s * 1e3).toFixed(2) + ' ms';
      return (s * 1e6).toFixed(1) + ' µs';
    },
    ai(v) { return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2); },
    pct(v) { return (v * 100 >= 10 ? (v * 100).toFixed(0) : (v * 100).toFixed(1)) + '%'; },
  };

  const cache = {};
  function load(phase) {
    if (cache[phase]) return cache[phase];
    const P = D.phases[phase], c = P.cols, n = P.n;
    const g = {
      phase, n, label: P.labels, stage: Int32Array.from(c.stage), block: Int32Array.from(c.block),
      level: Int32Array.from(c.level), lane: Int32Array.from(c.lane),
      flops: Float64Array.from(c.flops), inB: Float64Array.from(c.in_b),
      outB: Float64Array.from(c.out_b), wB: Float64Array.from(c.w_b),
      type: c.type.map(i => P.types[i]), module: c.mod.map(i => P.mods[i]),
      shape: c.shape.map(i => P.shapes[i]), edges: P.edges, constantOps: P.constant_ops,
    };
    g.bytes = new Float64Array(n); g.ai = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      g.bytes[i] = g.inB[i] + g.outB[i] + g.wB[i];
      g.ai[i] = g.bytes[i] ? g.flops[i] / g.bytes[i] : 0;
    }
    // CSR adjacency
    const m = g.edges.length / 2;
    g.m = m;
    g.pOff = new Int32Array(n + 1); g.cOff = new Int32Array(n + 1);
    for (let e = 0; e < m; e++) { g.pOff[g.edges[2 * e + 1] + 1]++; g.cOff[g.edges[2 * e] + 1]++; }
    for (let i = 0; i < n; i++) { g.pOff[i + 1] += g.pOff[i]; g.cOff[i + 1] += g.cOff[i]; }
    g.par = new Int32Array(m); g.chi = new Int32Array(m);
    const pf = g.pOff.slice(0, n), cf = g.cOff.slice(0, n);
    for (let e = 0; e < m; e++) {
      const a = g.edges[2 * e], b = g.edges[2 * e + 1];
      g.par[pf[b]++] = a; g.chi[cf[a]++] = b;
    }
    g.maxLevel = 0; for (let i = 0; i < n; i++) g.maxLevel = Math.max(g.maxLevel, g.level[i]);
    // per-level node lists and block level ranges
    g.byLevel = Array.from({ length: g.maxLevel + 2 }, () => []);
    for (let i = 0; i < n; i++) g.byLevel[g.level[i]].push(i);
    g.blocks = Array.from({ length: 64 }, () => ({ lo: 1e9, hi: -1 }));
    for (let i = 0; i < n; i++) if (g.block[i] >= 0) {
      const b = g.blocks[g.block[i]]; b.lo = Math.min(b.lo, g.level[i]); b.hi = Math.max(b.hi, g.level[i]);
    }
    return (cache[phase] = g);
  }

  /* Machine model: each op takes max(compute time, memory time) + launch overhead.
     Schedule every op as early as its inputs allow (infinite hardware, graph level). */
  function analyse(g, machine, overheadUs) {
    const M = MACHINES[machine], n = g.n, ridge = M.flops / M.bw;
    const t = new Float64Array(n), start = new Float64Array(n), finish = new Float64Array(n), from = new Int32Array(n).fill(-1);
    let work = 0, span = 0, last = 0;
    for (let i = 0; i < n; i++) {           // indices are already topological
      t[i] = Math.max(g.flops[i] / M.flops, g.bytes[i] / M.bw) + overheadUs * 1e-6;
      let s = 0, f = -1;
      for (let k = g.pOff[i]; k < g.pOff[i + 1]; k++) { const p = g.par[k]; if (finish[p] > s || f < 0) { s = Math.max(s, finish[p]); f = p; } }
      start[i] = s; finish[i] = s + t[i]; from[i] = f; work += t[i];
      if (finish[i] > span) { span = finish[i]; last = i; }
    }
    const path = []; for (let v = last; v >= 0; v = from[v]) path.push(v); path.reverse();
    const onPath = new Uint8Array(n); path.forEach(v => (onPath[v] = 1));
    let pathFlops = 0, pathBytes = 0, totFlops = 0, totBytes = 0, memBound = 0, memBoundTime = 0;
    const stages = D.stages.map(() => ({ ops: 0, flops: 0, bytes: 0, time: 0, pathTime: 0 }));
    for (let i = 0; i < n; i++) {
      totFlops += g.flops[i]; totBytes += g.bytes[i];
      if (onPath[i]) { pathFlops += g.flops[i]; pathBytes += g.bytes[i]; }
      const bound = g.ai[i] < ridge;
      if (bound) { memBound++; memBoundTime += t[i]; }
      const s = stages[g.stage[i]]; s.ops++; s.flops += g.flops[i]; s.bytes += g.bytes[i]; s.time += t[i]; if (onPath[i]) s.pathTime += t[i];
    }
    let widest = 0; for (const l of g.byLevel) widest = Math.max(widest, l.length);
    let single = 0; for (let l = 1; l <= g.maxLevel; l++) if (g.byLevel[l].length === 1) single++;
    // time at which each level first starts (for the readout)
    const levelStart = new Float64Array(g.maxLevel + 2).fill(Infinity);
    for (let i = 0; i < n; i++) levelStart[g.level[i]] = Math.min(levelStart[g.level[i]], start[i]);
    return { machine: M, ridge, t, start, finish, work, span, path, onPath, pathFlops, pathBytes, totFlops, totBytes,
      memBound, memBoundTime, stages, widest, single, levelStart, parallelism: work / span };
  }

  /* Node size: log scale between the 2nd and 100th percentile of non-zero values so the
     one huge matmul and the thousands of tiny elementwise ops both stay readable. */
  const norms = {};
  function sizeScale(g, metric) {
    const key = g.phase + metric;
    if (norms[key]) return norms[key];
    const src = metric === 'flops' ? g.flops : metric === 'bytes' ? g.bytes : g.ai;
    const vals = Array.from(src).filter(v => v > 0).sort((a, b) => a - b);
    const lo = Math.log10(vals[Math.floor(vals.length * 0.02)] || 1), hi = Math.log10(vals[vals.length - 1] || 10);
    const out = new Float32Array(g.n);
    for (let i = 0; i < g.n; i++) out[i] = src[i] > 0 ? Math.min(1, Math.max(0, (Math.log10(src[i]) - lo) / (hi - lo || 1))) : 0;
    return (norms[key] = { norm: out, lo: Math.pow(10, lo), hi: Math.pow(10, hi), src });
  }

  const SIZE_LABEL = { flops: 'FLOPs', bytes: 'bytes moved', ai: 'FLOPs per byte' };
  const valueText = (g, i, metric) => metric === 'flops' ? fmt.flops(g.flops[i]) : metric === 'bytes' ? fmt.bytes(g.bytes[i]) : fmt.ai(g.ai[i]) + ' FLOP/B';

  function describe(g, a, i) {
    const par = [], chi = [];
    for (let k = g.pOff[i]; k < g.pOff[i + 1]; k++) par.push(g.par[k]);
    for (let k = g.cOff[i]; k < g.cOff[i + 1]; k++) chi.push(g.chi[k]);
    return { par, chi, bound: g.ai[i] < a.ridge ? 'memory-bound' : 'compute-bound' };
  }

  window.QG = { D, MACHINES, colors, isDark, fmt, load, analyse, sizeScale, SIZE_LABEL, valueText, describe };
})();
