/* WebGL2 renderer: instanced boxes, drawn back to front with alpha blending, plus an
   outline pass. It knows nothing about transformers. scene.js hands it a ready frame:
   { objects (sorted far to near), camera, aspect, bg, ink }. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LoopGL = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const FOV = 40 * Math.PI / 180;

  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
  function viewProj(cam, aspect) {
    const cp = Math.cos(cam.pitch);
    const e = [cam.tx + cam.dist * cp * Math.sin(cam.yaw), cam.ty + cam.dist * Math.sin(cam.pitch), cam.tz + cam.dist * cp * Math.cos(cam.yaw)];
    const f = norm([cam.tx - e[0], cam.ty - e[1], cam.tz - e[2]]), r = norm(cross(f, [0, 1, 0])), u = cross(r, f);
    const V = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0,
      -(r[0] * e[0] + r[1] * e[1] + r[2] * e[2]), -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]), f[0] * e[0] + f[1] * e[1] + f[2] * e[2], 1];
    const t = 1 / Math.tan(FOV / 2), n = 0.1, fa = 400, P = new Array(16).fill(0);
    P[0] = t / aspect; P[5] = t; P[10] = (fa + n) / (n - fa); P[11] = -1; P[14] = 2 * fa * n / (n - fa);
    return { V: new Float32Array(V), P: new Float32Array(P) };
  }

  const CUBE = (() => {                      // 36 vertices: position + normal
    const v = [], faces = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
      [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
    for (const [n, a, b] of faces) {
      const c = (s, t) => [0, 1, 2].map(i => n[i] * .5 + a[i] * s * .5 + b[i] * t * .5);
      for (const [s, t] of [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]]) v.push(...c(s, t), ...n);
    }
    return new Float32Array(v);
  })();
  const EDGES = (() => {
    const c = [], p = [-.5, .5];
    for (const x of p) for (const y of p) { c.push(x, y, -.5, x, y, .5, x, -.5, y, x, .5, y, -.5, x, y, .5, x, y); }
    return new Float32Array(c);
  })();

  const VS = `#version 300 es
in vec3 aPos; in vec3 aNor; in vec4 aC; in vec4 aS; in vec4 aCol;
uniform mat4 uV, uP; out vec3 vN; out vec4 vCol;
void main(){ vec3 w = aC.xyz + aPos * aS.xyz; gl_Position = uP * uV * vec4(w, 1.); vN = aNor; vCol = vec4(aCol.rgb, aC.w); }`;
  const FS = `#version 300 es
precision highp float; in vec3 vN; in vec4 vCol; out vec4 o; uniform vec3 uLight;
void main(){ float l = .62 + .38 * max(dot(normalize(vN), normalize(uLight)), 0.) ; o = vec4(vCol.rgb * l, vCol.a); }`;
  const VS_E = `#version 300 es
in vec3 aPos; in vec4 aC; in vec4 aS; in vec4 aCol; uniform mat4 uV, uP; out vec4 vCol;
void main(){ gl_Position = uP * uV * vec4(aC.xyz + aPos * aS.xyz, 1.); vCol = vec4(aCol.rgb, aC.w * aS.w); }`;
  const FS_E = `#version 300 es
precision highp float; in vec4 vCol; out vec4 o; void main(){ o = vCol; }`;

  function compile(gl, vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  /* Returns null when WebGL2 is unavailable, so the caller can fall back. */
  function create(canvas) {
    let gl;
    try { gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true }); } catch (e) { gl = null; }
    if (!gl) return null;
    let boxes, edges;
    try { boxes = compile(gl, VS, FS); edges = compile(gl, VS_E, FS_E); } catch (e) { return null; }

    const mkVao = (prog, mesh, stride, withNormal) => {
      const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
      const mb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, mb); gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);
      const aPos = gl.getAttribLocation(prog, 'aPos'); gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, stride * 4, 0);
      if (withNormal) { const aNor = gl.getAttribLocation(prog, 'aNor'); gl.enableVertexAttribArray(aNor); gl.vertexAttribPointer(aNor, 3, gl.FLOAT, false, stride * 4, 12); }
      const ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ib);
      for (const [name, off] of [['aC', 0], ['aS', 16], ['aCol', 32]]) {
        const a = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(a);
        gl.vertexAttribPointer(a, 4, gl.FLOAT, false, 48, off); gl.vertexAttribDivisor(a, 1);
      }
      return { vao, ib };
    };
    const B = mkVao(boxes, CUBE, 6, true), E = mkVao(edges, EDGES, 3, false);
    let size = { w: 1, h: 1 }, last = null;

    function draw(frame) {
      last = frame || last; if (!last) return;
      const { objects, camera, aspect, bg, ink } = last;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
      if (!objects.length) return;
      const data = new Float32Array(objects.length * 12);
      objects.forEach((o, i) => {
        data.set([o.x, o.y, o.z, o.alpha, o.w, o.h, o.d, o.edge, o.rgb[0], o.rgb[1], o.rgb[2], 0], i * 12);
      });
      const edgeData = new Float32Array(objects.length * 12);
      objects.forEach((o, i) => edgeData.set([o.x, o.y, o.z, o.alpha, o.w, o.h, o.d, o.edge, ink[0], ink[1], ink[2], 0], i * 12));
      const { V, P } = viewProj(camera, aspect);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.disable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);
      gl.useProgram(boxes);
      gl.uniformMatrix4fv(gl.getUniformLocation(boxes, 'uV'), false, V); gl.uniformMatrix4fv(gl.getUniformLocation(boxes, 'uP'), false, P);
      gl.uniform3f(gl.getUniformLocation(boxes, 'uLight'), -.4, .8, .6);
      gl.bindVertexArray(B.vao); gl.bindBuffer(gl.ARRAY_BUFFER, B.ib); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, objects.length);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(edges);
      gl.uniformMatrix4fv(gl.getUniformLocation(edges, 'uV'), false, V); gl.uniformMatrix4fv(gl.getUniformLocation(edges, 'uP'), false, P);
      gl.bindVertexArray(E.vao); gl.bindBuffer(gl.ARRAY_BUFFER, E.ib); gl.bufferData(gl.ARRAY_BUFFER, edgeData, gl.DYNAMIC_DRAW);
      gl.drawArraysInstanced(gl.LINES, 0, 24, objects.length);
    }
    return {
      kind: 'gl',
      resize(w, h) { size = { w, h }; canvas.width = w; canvas.height = h; },
      draw,
      coverage(bg) {                                           // fraction of pixels that differ from the background
        const w = canvas.width, h = canvas.height, px = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
        const b = bg.map(v => Math.round(v * 255)); let n = 0;
        for (let i = 0; i < px.length; i += 4) if (Math.abs(px[i] - b[0]) + Math.abs(px[i + 1] - b[1]) + Math.abs(px[i + 2] - b[2]) > 24) n++;
        return n / (w * h);
      },
      destroy() { const x = gl.getExtension('WEBGL_lose_context'); if (x) x.loseContext(); },
    };
  }
  return { create, viewProj, FOV };
});
