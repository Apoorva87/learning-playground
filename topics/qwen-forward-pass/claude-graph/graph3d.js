/* WebGL2 view of the same graph. The 3,400-step chain is wound into a helix, one turn
   per decoder block: angle = step inside the block, height = block number (time runs
   upward), radius = lane. A block's end sits directly above its start, so the whole pass
   is one continuous climb and the same stage lines up at the same angle on every turn.
   Spheres are instanced billboards. */
(() => {
  'use strict';
  const Q = window.QG, { fmt, MACHINES } = Q;
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const S = { phase: 'decode', metric: 'flops', machine: 'h100', crit: true, on: new Set(Q.D.stages.map((_, i) => i)),
    step: 0, playing: false, follow: true, sel: -1, yaw: -0.6, pitch: 0.3, dist: 36, ty: 9, auto: false };
  let g, a, scale, gl, prog = {}, buf = {}, pos, W = 1, H = 1, mvp = new Float32Array(16), view = new Float32Array(16), proj = new Float32Array(16);

  const TURN = 1.05, R0 = 3.6, DR = 0.34, BASE = 0.38;

  /* ---------- tiny mat4 ---------- */
  const mul = (A, B) => { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += A[k * 4 + r] * B[c * 4 + k]; o[c * 4 + r] = s; } return o; };
  const persp = (fov, asp, n, f) => { const t = 1 / Math.tan(fov / 2), o = new Float32Array(16); o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; };
  const look = (e, c, u) => {
    let z = [e[0] - c[0], e[1] - c[1], e[2] - c[2]], l = Math.hypot(...z); z = z.map(v => v / l);
    let x = [u[1] * z[2] - u[2] * z[1], u[2] * z[0] - u[0] * z[2], u[0] * z[1] - u[1] * z[0]]; l = Math.hypot(...x); x = x.map(v => v / l);
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -(x[0] * e[0] + x[1] * e[1] + x[2] * e[2]), -(y[0] * e[0] + y[1] * e[1] + y[2] * e[2]), -(z[0] * e[0] + z[1] * e[1] + z[2] * e[2]), 1]);
  };

  /* ---------- GL setup ---------- */
  const VS_N = `#version 300 es
in vec2 aCorner; in vec4 aP; in vec4 aM;
uniform mat4 uV, uP; uniform vec3 uPal[8]; uniform int uMask; uniform float uLevel, uBase; uniform vec3 uSelPos;
out vec2 vC; out vec3 vCol; out float vA; out float vRing; out float vSel;
void main(){
  int st = int(aM.x + .5); bool on = ((uMask >> st) & 1) == 1; bool seen = aM.y <= uLevel;
  float k = on ? (seen ? 1. : .3) : .22;
  float r = uBase * (.1 + .4 * aP.w) * (on && seen ? 1. : .45);
  vec4 vp = uV * vec4(aP.xyz, 1.); vp.xy += aCorner * r;
  gl_Position = uP * vp; vC = aCorner; vCol = uPal[st]; vA = k * (on && !seen ? .55 : 1.);
  vRing = aM.z + 2. * aM.w * step(.5, aM.w);
  vSel = (distance(aP.xyz, uSelPos) < .001) ? 1. : 0.;
}`;
  const FS_N = `#version 300 es
precision highp float; in vec2 vC; in vec3 vCol; in float vA; in float vRing; in float vSel; out vec4 o;
uniform vec3 uBg, uInk, uPath;
void main(){
  float d = dot(vC, vC); if (d > 1.) discard;
  vec3 n = vec3(vC, sqrt(1. - d)); float l = .5 + .5 * dot(n, normalize(vec3(-.4, .6, .7)));
  vec3 c = vCol * (.62 + .5 * l);
  if (vRing > .5 && vRing < 1.5 && d > .66) c = uPath;
  if (vRing > 1.5 && d > .7) c = uInk;
  if (vSel > .5 && d > .55) c = vec3(.2, .55, 1.);
  o = vec4(mix(uBg, c, vA), 1.);
}`;
  const VS_L = `#version 300 es
in vec4 aP; uniform mat4 uV, uP; uniform float uLevel; out float vS;
void main(){ gl_Position = uP * uV * vec4(aP.xyz, 1.); vS = aP.w <= uLevel ? 1. : .0; }`;
  const FS_L = `#version 300 es
precision highp float; in float vS; uniform vec4 uCol; uniform vec3 uBg; out vec4 o;
void main(){ o = vec4(mix(uBg, uCol.rgb, uCol.a * (.25 + .75 * vS)), 1.); }`;

  function compile(vs, fs) {
    const mk = (t, s) => { const h = gl.createShader(t); gl.shaderSource(h, s); gl.compileShader(h); if (!gl.getShaderParameter(h, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(h)); return h; };
    const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
  const cssRgb = v => { const m = v.match(/#([0-9a-f]{6})/i); if (m) return hex('#' + m[1]); const n = (v.match(/[\d.]+/g) || [0, 0, 0]).map(Number); return n.slice(0, 3).map(x => x / 255); };

  function layout() {
    const blk = g.blocks, first = blk[0].lo, last = blk[63].hi, avg = (last - first + 1) / 64;
    pos = new Float32Array(g.n * 3);
    for (let i = 0; i < g.n; i++) {
      let u;                                                 // turns completed, 0 = start of block 0
      if (g.block[i] >= 0) { const bb = blk[g.block[i]]; u = g.block[i] + (g.level[i] - bb.lo) / (bb.hi - bb.lo + 1); }
      else if (g.level[i] < first) u = (g.level[i] - first) / avg; else u = 64 + (g.level[i] - last - 1) / avg;
      const th = (u % 1 + 1) % 1 * Math.PI * 2 * 0.96, R = R0 + (g.lane[i] - 1) * DR;
      pos[3 * i] = R * Math.cos(th); pos[3 * i + 1] = u * TURN; pos[3 * i + 2] = R * Math.sin(th);
    }
  }

  function upload() {
    const n = g.n, inst = new Float32Array(n * 8);
    for (let i = 0; i < n; i++) {
      inst.set([pos[3 * i], pos[3 * i + 1], pos[3 * i + 2], scale.norm[i], g.stage[i], g.level[i], a.onPath[i] && S.crit ? 1 : 0, g.ai[i] >= a.ridge ? 1 : 0], i * 8);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.inst); gl.bufferData(gl.ARRAY_BUFFER, inst, gl.DYNAMIC_DRAW);
    const lines = [], cross = [], path = [];
    for (let e = 0; e < g.m; e++) {
      const p = g.edges[2 * e], q = g.edges[2 * e + 1];
      if (g.level[q] - g.level[p] > 60) continue;
      const seg = [pos[3 * p], pos[3 * p + 1], pos[3 * p + 2], g.level[p], pos[3 * q], pos[3 * q + 1], pos[3 * q + 2], g.level[q]];
      lines.push(...seg);
    }
    for (let k = 1; k < a.path.length; k++) {
      const p = a.path[k - 1], q = a.path[k]; if (g.level[q] - g.level[p] > 60) continue;
      path.push(pos[3 * p], pos[3 * p + 1], pos[3 * p + 2], g.level[p], pos[3 * q], pos[3 * q + 1], pos[3 * q + 2], g.level[q]);
    }
    buf.nLines = lines.length / 4; buf.nCross = cross.length / 4; buf.nPath = path.length / 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.lines); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.cross); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(cross), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.path); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(path), gl.STATIC_DRAW);
    buf.n = n;
  }

  function initGL() {
    const c = $('gl'); gl = c.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    prog.n = compile(VS_N, FS_N); prog.l = compile(VS_L, FS_L);
    buf.corner = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf.corner); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    buf.inst = gl.createBuffer(); buf.lines = gl.createBuffer(); buf.cross = gl.createBuffer(); buf.path = gl.createBuffer();
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
  }

  /* ---------- frame ---------- */
  function resize() {
    const c = $('gl'), r = c.getBoundingClientRect(), d = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height; c.width = Math.round(W * d); c.height = Math.round(H * d);
    const o = $('label'); o.width = c.width; o.height = c.height; o.getContext('2d').setTransform(d, 0, 0, d, 0, 0);
  }

  function matrices() {
    const e = [S.dist * Math.cos(S.pitch) * Math.sin(S.yaw), S.ty + S.dist * Math.sin(S.pitch), S.dist * Math.cos(S.pitch) * Math.cos(S.yaw)];
    view = look(e, [0, S.ty, 0], [0, 1, 0]); proj = persp(0.9, W / H, 0.2, 600); mvp = mul(proj, view);
  }

  function frame() {
    if (!gl) return;
    if (S.auto) S.yaw += 0.004;
    matrices();
    const c = $('gl'), bg = cssRgb(css('--surface')), ink = cssRgb(css('--ink')), path = cssRgb(css('--path'));
    gl.viewport(0, 0, c.width, c.height); gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const level = S.step;
    // lines
    gl.useProgram(prog.l);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog.l, 'uV'), false, view); gl.uniformMatrix4fv(gl.getUniformLocation(prog.l, 'uP'), false, proj);
    gl.uniform1f(gl.getUniformLocation(prog.l, 'uLevel'), level); gl.uniform3fv(gl.getUniformLocation(prog.l, 'uBg'), bg);
    const lp = gl.getAttribLocation(prog.l, 'aP');
    const drawLines = (b, n, col) => { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 4, gl.FLOAT, false, 16, 0); gl.vertexAttribDivisor(lp, 0); gl.uniform4fv(gl.getUniformLocation(prog.l, 'uCol'), col); gl.drawArrays(gl.LINES, 0, n); };
    const ec = cssRgb(css('--ink2')); drawLines(buf.lines, buf.nLines, [ec[0], ec[1], ec[2], .42]);
    if (S.crit) drawLines(buf.path, buf.nPath, [path[0], path[1], path[2], 1]);
    // nodes
    gl.useProgram(prog.n);
    const U = n => gl.getUniformLocation(prog.n, n);
    gl.uniformMatrix4fv(U('uV'), false, view); gl.uniformMatrix4fv(U('uP'), false, proj);
    const pal = Q.colors().flatMap(hex); gl.uniform3fv(U('uPal[0]'), new Float32Array(pal));
    let mask = 0; S.on.forEach(i => (mask |= 1 << i)); gl.uniform1i(U('uMask'), mask);
    gl.uniform1f(U('uLevel'), level); gl.uniform1f(U('uBase'), BASE);
    gl.uniform3fv(U('uBg'), bg); gl.uniform3fv(U('uInk'), ink); gl.uniform3fv(U('uPath'), path);
    const sp = S.sel >= 0 ? [pos[3 * S.sel], pos[3 * S.sel + 1], pos[3 * S.sel + 2]] : [1e9, 1e9, 1e9]; gl.uniform3fv(U('uSelPos'), sp);
    const ac = gl.getAttribLocation(prog.n, 'aCorner'), ap = gl.getAttribLocation(prog.n, 'aP'), am = gl.getAttribLocation(prog.n, 'aM');
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.corner); gl.enableVertexAttribArray(ac); gl.vertexAttribPointer(ac, 2, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(ac, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.inst);
    gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(ap, 1);
    gl.enableVertexAttribArray(am); gl.vertexAttribPointer(am, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(am, 1);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, buf.n);
    overlay();
  }

  const project = (i, out) => {
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12], cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13], cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw <= 0.01) return null;
    out[0] = (cx / cw * .5 + .5) * W; out[1] = (1 - (cy / cw * .5 + .5)) * H; out[2] = cw; return out;
  };

  function overlay() {
    const c = $('label'), x = c.getContext('2d'); x.clearRect(0, 0, W, H);
    x.font = '11px system-ui'; x.fillStyle = css('--ink2'); x.textBaseline = 'middle';
    const tmp = [0, 0, 0], blk = g.blocks;
    for (let b = 0; b < 64; b++) {
      if (blk[b].hi < 0) continue; const first = g.byLevel[blk[b].lo][0]; if (first == null) continue;
      const every = S.dist > 70 ? 8 : S.dist > 30 ? 4 : 1;
      if (b % every && !(S.dist < 30 && Q.D.layer_types[b] === 'F')) continue;
      const p = project(first, tmp); if (!p || p[0] < -40 || p[0] > W || p[1] < 0 || p[1] > H) continue;
      x.fillStyle = Q.D.layer_types[b] === 'F' ? css('--ink') : css('--ink2');
      x.fillText(`${b}${Q.D.layer_types[b]}`, p[0] + 6, p[1]);
    }
  }

  /* ---------- picking, UI ---------- */
  function pick(mx, my) {
    let best = -1, bd = 1e9; const t = [0, 0, 0];
    for (let i = 0; i < g.n; i++) {
      if (!S.on.has(g.stage[i]) || g.level[i] > S.step) continue;
      const p = project(i, t); if (!p) continue;
      const rpx = BASE * (.1 + .4 * scale.norm[i]) / p[2] * H / (2 * Math.tan(.45)) * 1.0;
      const d = Math.hypot(p[0] - mx, p[1] - my) - Math.max(rpx, 2);
      if (d < bd || (d === bd && p[2] < 0)) { bd = d; best = i; }
    }
    return bd < 6 ? best : -1;
  }
  function tip(i, ev) {
    const t = $('tip'); if (i < 0) { t.style.display = 'none'; return; }
    t.innerHTML = `<b>${g.module[i] || g.type[i]}</b><br>${g.type[i]} · ${fmt.flops(g.flops[i])} · ${fmt.bytes(g.bytes[i])} · ${fmt.ai(g.ai[i])} FLOP/B<br>${Q.D.stages[g.stage[i]][1]}${g.block[i] >= 0 ? ' · block ' + g.block[i] : ''} · step ${g.level[i]}`;
    t.style.display = 'block'; t.style.left = Math.min(innerWidth - 330, ev.clientX + 14) + 'px'; t.style.top = ev.clientY + 14 + 'px';
  }
  function detail() {
    const b = $('detail'); if (S.sel < 0) { b.innerHTML = '<h3>Selected op</h3><p style="color:var(--ink2)">Click a sphere to inspect it.</p>'; return; }
    const i = S.sel, d = Q.describe(g, a, i), col = Q.colors()[g.stage[i]];
    b.innerHTML = `<h3><span class="sw" style="background:${col}"></span>${Q.D.stages[g.stage[i]][1]}</h3><dl class="kv"><dt>Where</dt><dd>${g.module[i] || 'model'}</dd><dt>Op</dt><dd>${g.type[i]} · ${g.shape[i]}</dd>` +
      `<dt>FLOPs</dt><dd>${fmt.flops(g.flops[i])}</dd><dt>Bytes</dt><dd>${fmt.bytes(g.bytes[i])}</dd><dt>FLOPs / byte</dt><dd>${fmt.ai(g.ai[i])} · ${d.bound}</dd>` +
      `<dt>Modeled time</dt><dd>${fmt.time(a.t[i])} from ${fmt.time(a.start[i])}</dd><dt>Step</dt><dd>${g.level[i]}${g.block[i] >= 0 ? ' · block ' + g.block[i] : ''}${a.onPath[i] ? ' · critical path' : ''}</dd></dl>`;
  }

  function legend() {
    const col = Q.colors();
    $('legend').innerHTML = Q.D.stages.map(([, nm], i) => `<button class="chip${S.on.has(i) ? '' : ' off'}" data-i="${i}"><span class="sw" style="background:${col[i]}"></span>${nm}</button>`).join('');
  }
  function stats() {
    $('lead').innerHTML = `The ${S.phase === 'decode' ? 'single decode step' : '16-token prefill'} climbs a coil with <b>one turn per decoder block</b>, block 0 at the bottom. Time runs upward: ${g.n.toLocaleString()} ops, ${g.maxLevel.toLocaleString()} dependency steps, never more than <b>${a.widest}</b> side by side. ` +
      `Every fourth turn (labelled <b>F</b>) is full attention; the others are recurrent. Node size = ${Q.SIZE_LABEL[S.metric]}.`;
    $('readout').textContent = `${fmt.pct(a.pathFlops / a.totFlops)} of FLOPs on the critical path (red) · graph-level parallelism ${a.parallelism.toFixed(2)}× · ${S.step >= g.maxLevel ? 'all steps shown' : 'step ' + S.step + ' of ' + g.maxLevel}`;
  }

  function recompute() {
    g = Q.load(S.phase); a = Q.analyse(g, S.machine, 0); scale = Q.sizeScale(g, S.metric);
    layout(); upload(); if (S.step === 0 || S.step > g.maxLevel) S.step = g.maxLevel;
    $('step').max = g.maxLevel; $('step').value = S.step; legend(); stats(); detail();
  }

  let last = 0;
  function loop(ts) {
    if (S.playing && ts - last > 16) {
      last = ts; S.step = Math.min(g.maxLevel, S.step + 6); $('step').value = S.step;
      if (S.follow) { const ids = g.byLevel[S.step] || []; const i = ids[0]; if (i != null) S.ty += (pos[3 * i + 1] - S.ty) * .08; }
      if (S.step >= g.maxLevel) { S.playing = false; $('play').textContent = '▶ Play the pass'; }
      stats();
    }
    frame(); requestAnimationFrame(loop);
  }

  function bind() {
    const c = $('gl'); let drag = null;
    c.addEventListener('contextmenu', e => e.preventDefault());
    c.addEventListener('pointerdown', e => { c.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, yaw: S.yaw, pitch: S.pitch, ty: S.ty, pan: e.shiftKey || e.button === 2, moved: false }; });
    c.addEventListener('pointermove', e => {
      const r = c.getBoundingClientRect();
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.hypot(dx, dy) > 3) drag.moved = true;
        if (drag.pan) { S.ty = drag.ty + dy * S.dist * .0018; }
        else { S.yaw = drag.yaw - dx * .006; S.pitch = Math.max(-1.45, Math.min(1.45, drag.pitch + dy * .006)); }
        return;
      }
      tip(pick(e.clientX - r.left, e.clientY - r.top), e);
    });
    c.addEventListener('pointerup', e => { const r = c.getBoundingClientRect(); if (drag && !drag.moved) { S.sel = pick(e.clientX - r.left, e.clientY - r.top); detail(); } drag = null; });
    c.addEventListener('pointerleave', () => tip(-1));
    c.addEventListener('wheel', e => { e.preventDefault(); S.dist = Math.max(2, Math.min(200, S.dist * Math.exp(e.deltaY * .0012))); }, { passive: false });
    c.addEventListener('keydown', e => { const k = e.key; if (k === 'ArrowLeft') S.yaw -= .08; else if (k === 'ArrowRight') S.yaw += .08; else if (k === 'ArrowUp') S.ty += TURN * 2; else if (k === 'ArrowDown') S.ty -= TURN * 2; else if (k === '+' || k === '=') S.dist *= .9; else if (k === '-') S.dist /= .9; else return; e.preventDefault(); });
    const seg = (id, key) => $(id).addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; [...$(id).children].forEach(x => x.setAttribute('aria-pressed', x === b)); S[key] = b.dataset.v; if (key === 'phase') { S.step = 0; S.sel = -1; } recompute(); });
    seg('phase', 'phase'); seg('metric', 'metric');
    $('machine').innerHTML = Object.entries(MACHINES).map(([k, m]) => `<option value="${k}">${m.name}</option>`).join('');
    $('machine').onchange = e => { S.machine = e.target.value; recompute(); };
    $('crit').onchange = e => { S.crit = e.target.checked; upload(); };
    $('legend').addEventListener('click', e => { const b = e.target.closest('.chip'); if (!b) return; const i = +b.dataset.i, all = Q.D.stages.length;
      if (S.on.size === all) S.on = new Set([i]); else if (S.on.has(i) && S.on.size === 1) S.on = new Set(Q.D.stages.map((_, k) => k)); else if (S.on.has(i)) S.on.delete(i); else S.on.add(i); legend(); });
    $('views').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; const v = b.dataset.v;
      if (v === 'iso') Object.assign(S, { yaw: -0.6, pitch: 0.3, dist: 36, ty: 9 });
      if (v === 'whole') Object.assign(S, { yaw: -0.6, pitch: 0.25, dist: 80, ty: 34 });
      if (v === 'side') Object.assign(S, { yaw: 0, pitch: 0.02, dist: 120, ty: 34 });
      if (v === 'top') Object.assign(S, { yaw: 0, pitch: 1.5, dist: 75, ty: 34 });
      if (v === 'block') Object.assign(S, { yaw: -0.5, pitch: 0.32, dist: 11, ty: 3.5 * TURN });
      if (v === 'spin') { S.auto = !S.auto; b.setAttribute('aria-pressed', S.auto); } });
    $('step').oninput = e => { S.step = +e.target.value; S.playing = false; $('play').textContent = '▶ Play the pass'; stats(); };
    $('play').onclick = () => { S.playing = !S.playing; $('play').textContent = S.playing ? '❚❚ Pause' : '▶ Play the pass'; if (S.playing && S.step >= g.maxLevel) { S.step = 1; } };
    $('showall').onclick = () => { S.step = g.maxLevel; $('step').value = S.step; S.playing = false; stats(); };
    addEventListener('resize', resize);
  }

  function start() {
    try { initGL(); } catch (err) { $('fallback').hidden = false; $('fallback').textContent = err.message + ' Use the 2D graph instead.'; return; }
    bind(); resize(); recompute();
    const h = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
    if (h.phase === 'prefill') { S.phase = 'prefill'; [...$('phase').children].forEach(b => b.setAttribute('aria-pressed', b.dataset.v === 'prefill')); recompute(); }
    if (['flops', 'bytes', 'ai'].includes(h.metric)) { S.metric = h.metric; [...$('metric').children].forEach(b => b.setAttribute('aria-pressed', b.dataset.v === h.metric)); recompute(); }
    if (h.view) document.querySelector(`#views [data-v=${h.view}]`)?.click();
    if (h.yaw) S.yaw = +h.yaw; if (h.pitch) S.pitch = +h.pitch; if (h.dist) S.dist = +h.dist; if (h.ty) S.ty = +h.ty * TURN;
    if (h.step) { S.step = +h.step; $('step').value = S.step; stats(); }
    if (h.sel) { S.sel = +h.sel; detail(); }
    if (h.dark) document.documentElement.dataset.theme = 'dark';
    requestAnimationFrame(loop);
  }
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', start); else start();
})();
