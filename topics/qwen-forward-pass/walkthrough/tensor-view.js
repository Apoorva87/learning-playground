/* An orthographic tensor diagram. Cell counts and lengths are compressed for
   display. Only the printed dimensions describe the full tensor. */
(() => {
  'use strict';

  const background = '#f7fafb';
  const palettes = {
    input:  ['#b7d9f5', '#74a8d0', '#d9ecfb', '#557f9e'],
    work:   ['#a2c9ea', '#659bc6', '#d5e9f9', '#517492'],
    output: ['#8bbce7', '#517fa9', '#c8e3f8', '#466f92'],
    weight: ['#e9c47e', '#bc8f46', '#f5dfaf', '#9c783c'],
    state:  ['#85cbbb', '#468f85', '#c1e8dc', '#397c73']
  };
  const home = {yaw: .42, pitch: .36, zoom: 1};
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const sum = (a, b) => a.map((n, i) => n + b[i]);
  const diff = (a, b) => a.map((n, i) => n - b[i]);
  const mix = (a, b, f) => a.map((n, i) => n + (b[i] - n) * f);
  const dot = (a, b) => a.reduce((n, v, i) => n + v * b[i], 0);
  const average = points => points[0].map((_, i) => points.reduce((v, p) => v + p[i], 0) / points.length);
  const path = (points, color) => ({points, color, fill: false});
  const polygon = (points, color) => ({points, color, fill: true});

  function cameraBasis({yaw, pitch}) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return {right: [cy, 0, -sy], up: [-sy * sp, cp, -cy * sp], near: [sy * cp, sp, cy * cp]};
  }

  function bounds(points) {
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    return {left: Math.min(...xs), right: Math.max(...xs), top: Math.min(...ys), bottom: Math.max(...ys)};
  }

  function makeFace(a, b, c, d, normal, hCells, vCells, fill, ink) {
    const strokes = [path([a, b, c, d, a], ink)];
    for (let i = 1; i < hCells; i++) {
      strokes.push(path([mix(a, b, i / hCells), mix(d, c, i / hCells)], ink));
    }
    for (let i = 1; i < vCells; i++) {
      strokes.push(path([mix(a, d, i / vCells), mix(b, c, i / vCells)], ink));
    }
    return {surface: polygon([a, b, c, d], fill), bands: [], strokes, trim: [], normal, center: mix(a, c, .5)};
  }

  // Position is the last axis of a probability tensor, and the penultimate
  // axis of the key/value cache. Heat describes one inspected head and query.
  function markPositions(face, node, count) {
    const [a, b, topRight, topLeft] = face.surface.points;
    const rowAxis = node.role === 'state';
    const heat = node.heat?.length === count ? node.heat : null;
    // The captured probability row belongs to the last query in the call.
    const queryCount = rowAxis ? 1 : node.shape.at(-2) || 1;
    face.positions = {rowAxis, queryCount, count};
    const c = mix(b, topRight, 1 / queryCount), d = mix(a, topLeft, 1 / queryCount);
    const band = (lo, hi, z) => {
      const quad = rowAxis
        ? [mix(d, a, lo), mix(c, b, lo), mix(c, b, hi), mix(d, a, hi)]
        : [mix(a, b, lo), mix(a, b, hi), mix(d, c, hi), mix(d, c, lo)];
      return quad.map(p => sum(p, [0, 0, z]));
    };
    if (heat) heat.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const color = mix([220, 237, 245], [20, 75, 137], clamp(value, 0, 1));
      const hex = '#' + color.map(x => Math.round(x).toString(16).padStart(2, '0')).join('');
      face.bands.push(polygon(band(index / count, (index + 1) / count, .003), hex));
    });
    if (heat) face.strokes.forEach(stroke => {
      stroke.points = stroke.points.map(p => sum(p, [0, 0, .006]));
    });
    if (Number.isInteger(node.activeCell) && node.activeCell >= 0 && node.activeCell < count) {
      const box = band(node.activeCell / count, (node.activeCell + 1) / count, .010);
      // Four short ribbons keep the selected cell visible on high-DPI WebGL.
      for (let i = 0; i < 4; i++) {
        const p = box[i], q = box[(i + 1) % 4];
        const [dx, dy] = diff(q, p), len = Math.hypot(dx, dy);
        const n = [-dy * .018 / len, dx * .018 / len, 0];
        face.trim.push(polygon([sum(p, n), sum(q, n), diff(q, n), diff(p, n)], '#18394c'));
      }
    }
  }

  // Last dimension runs across the front face, the preceding one runs up.
  // Earlier non-singleton axes form the stack. Every dimension is printed.
  function makeBlock(node, portrait) {
    const dims = node.shape;
    const widthDim = dims.at(-1) || 1, heightDim = dims.at(-2) || 1;
    const depthDim = dims.slice(0, -2).reduce((a, b) => a * b, 1);
    const width = 1.08 + Math.min(1.5, Math.log2(widthDim) * .12);
    const height = .4 + Math.min(1.46, Math.log2(heightDim) * .14);
    const slices = Math.min(3, depthDim);
    const depth = depthDim === 1 ? .34 : .52 + Math.min(.66, Math.log2(depthDim) * .12);
    const [x, y] = portrait ? [node.row * 3.85, -node.col * 3.35] : [node.col * 4.05, -node.row * 3.05];
    const [l, r, lo, hi] = [x - width / 2, x + width / 2, y - height / 2, y + height / 2];
    const palette = palettes[node.role] || palettes.work;
    const cells = n => n <= 12 ? n : Math.min(17, Math.round(5 + Math.log2(n) * .7));
    const faces = [], corners = [];
    const slab = (depth - (slices - 1) * .085) / slices;
    for (let i = 0; i < slices; i++) {
      const back = i * (slab + .085), front = back + slab;
      const a = [l, lo, front], b = [r, lo, front], c = [r, hi, front], d = [l, hi, front];
      const e = [l, lo, back], f = [r, lo, back], g = [r, hi, back], h = [l, hi, back];
      corners.push(a, b, c, d, e, f, g, h);
      const posCount = node.role === 'state' ? heightDim : widthDim;
      const showPositions = posCount <= 64 && i === slices - 1 && (node.heat || Number.isInteger(node.activeCell));
      const frontFace = makeFace(a, b, c, d, [0, 0, 1],
        showPositions && node.role !== 'state' ? posCount : cells(widthDim),
        showPositions && heightDim <= 64 ? heightDim : cells(heightDim), palette[0], palette[3]);
      if (showPositions) markPositions(frontFace, node, posCount);
      faces.push(frontFace,
        makeFace(e, f, g, h, [0, 0, -1], cells(widthDim), cells(heightDim), palette[1], palette[3]),
        makeFace(b, f, g, c, [1, 0, 0], 1, cells(heightDim), palette[1], palette[3]),
        makeFace(a, e, h, d, [-1, 0, 0], 1, cells(heightDim), palette[1], palette[3]),
        makeFace(d, c, g, h, [0, 1, 0], cells(widthDim), 1, palette[2], palette[3]),
        makeFace(a, b, f, e, [0, -1, 0], cells(widthDim), 1, palette[1], palette[3])
      );
    }
    const foot = corners.slice(0, 4).map(([px, py]) => [px + .10, py - .15, -.58]);
    return {node, x, y, width, height, depth, faces, corners, foot};
  }

  function makeArrow(a, b, active, portrait) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const across = portrait ? !dy : !!dx;
    const axis = across ? 0 : 1;
    const dir = Math.sign(across ? dx : dy);
    if (!dir) return [];
    const start = [a.x, a.y, .12], end = [b.x, b.y, .12];
    const extent = block => (across ? block.width : block.height) * .5;
    start[axis] += dir * (extent(a) + .07);
    end[axis] -= dir * (extent(b) + .10);
    const span = Math.abs(end[axis] - start[axis]);
    const tangent = Math.max(.26, span * .45);
    const first = start.slice(), last = end.slice();
    first[axis] += dir * tangent;
    last[axis] -= dir * tangent;
    const base = end.slice();
    base[axis] -= dir * Math.min(.25, span * .25);
    const ink = active ? '#376378' : '#8da4ae';
    const out = [];
    let prev = start;
    for (let i = 1; i <= 18; i++) {
      const t = i / 18;
      const p = mix(mix(mix(start, first, t), mix(first, last, t), t), mix(mix(first, last, t), mix(last, base, t), t), t);
      const [vx, vy] = diff(p, prev);
      const length = Math.hypot(vx, vy) || 1;
      const thickness = active ? .026 : .018;
      const offset = [-vy / length * thickness, vx / length * thickness, 0];
      out.push(polygon([sum(prev, offset), sum(p, offset), diff(p, offset), diff(prev, offset)], ink));
      prev = p;
    }
    const lip = across ? [0, .11, 0] : [.11, 0, 0];
    out.push(polygon([sum(base, lip), end, diff(base, lip)], ink));
    return out;
  }

  function sceneGeometry(scene, portrait) {
    const blocks = scene.nodes.map(node => makeBlock(node, portrait));
    const byId = new Map(blocks.map(b => [b.node.id, b]));
    const board = [], underlay = [], selection = [];
    // The board is behind the tensor geometry in the same coordinate system.
    for (let n = -52; n <= 52; n++) {
      const t = n * .8;
      const ink = n % 5 === 0 ? '#dce5e8' : '#e9eff1';
      board.push(path([[t, -42, -.68], [t, 42, -.68]], ink), path([[-42, t, -.68], [42, t, -.68]], ink));
    }
    blocks.forEach(b => underlay.push(polygon(b.foot, '#e1e8e9')));
    scene.edges.forEach(([from, to]) => {
      if (byId.has(from) && byId.has(to)) {
        underlay.push(...makeArrow(byId.get(from), byId.get(to), scene.selected === from || scene.selected === to, portrait));
      }
    });
    const active = byId.get(scene.selected);
    if (active) {
      const ink = '#24485c', margin = .075, length = .19, z = active.depth + .012;
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
        const x = active.x + sx * (active.width / 2 + margin);
        const y = active.y + sy * (active.height / 2 + margin);
        const x1 = x - sx * length, y1 = y - sy * length;
        const w = .024;
        selection.push(polygon([[x, y - w, z], [x1, y - w, z], [x1, y + w, z], [x, y + w, z]], ink));
        selection.push(polygon([[x - w, y, z], [x - w, y1, z], [x + w, y1, z], [x + w, y, z]], ink));
      }
    }
    return {blocks, board, underlay, selection};
  }

  function createGpu(canvas) {
    const gl = canvas.getContext('webgl', {alpha: false, antialias: true, depth: true});
    if (!gl || gl.isContextLost()) return null;
    const shaders = [];
    const program = gl.createProgram();
    const buffer = gl.createBuffer();
    try {
      const compile = (type, source) => {
        const shader = gl.createShader(type);
        shaders.push(shader);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
        gl.attachShader(program, shader);
      };
      compile(gl.VERTEX_SHADER, [
        'attribute vec3 point, ink;', 'uniform mat4 camera;', 'varying lowp vec3 color;',
        'void main() { gl_Position = camera * vec4(point, 1.0); color = ink; }'
      ].join('\n'));
      compile(gl.FRAGMENT_SHADER, 'precision mediump float; varying lowp vec3 color; void main() { gl_FragColor = vec4(color, 1.0); }');
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    } catch {
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      return null;
    } finally {
      shaders.forEach(shader => gl.deleteShader(shader));
    }

    const colors = new Map();
    const rgb = hex => {
      if (!colors.has(hex)) colors.set(hex, [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255));
      return colors.get(hex);
    };
    const camera = gl.getUniformLocation(program, 'camera');
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    for (const [attribute, offset] of [['point', 0], ['ink', 12]]) {
      const index = gl.getAttribLocation(program, attribute);
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, 3, gl.FLOAT, false, 24, offset);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clearColor(...rgb(background), 1);
    let triangles = 0, lines = 0;
    return {
      upload(model) {
        const solid = [], wire = [];
        function add(primitive) {
          const {points, color, fill} = primitive;
          const shade = rgb(color);
          for (let i = 1; i < points.length; i++) {
            if (fill && i + 1 < points.length) solid.push(...points[0], ...shade, ...points[i], ...shade, ...points[i + 1], ...shade);
            if (!fill) wire.push(...points[i - 1], ...shade, ...points[i], ...shade);
          }
        }
        model.board.forEach(add);
        model.underlay.forEach(add);
        for (const b of model.blocks) for (const face of b.faces) {
          add(face.surface);
          face.bands.forEach(add);
          face.strokes.forEach(add);
          face.trim.forEach(add);
        }
        model.selection.forEach(add);
        triangles = solid.length / 6;
        lines = wire.length / 6;
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(solid.concat(wire)), gl.STATIC_DRAW);
      },
      draw(matrix) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.uniformMatrix4fv(camera, false, matrix);
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(1, 1);
        gl.drawArrays(gl.TRIANGLES, 0, triangles);
        gl.disable(gl.POLYGON_OFFSET_FILL);
        gl.drawArrays(gl.LINES, triangles, lines);
      },
      dispose() { gl.deleteBuffer(buffer); gl.deleteProgram(program); }
    };
  }

  function drawFlat(ctx, model, frame, dpr) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, frame.width, frame.height);
    ctx.lineWidth = 1 / dpr;
    function paint(primitive) {
      ctx.beginPath();
      primitive.points.forEach((p, i) => {
        const [x, y] = frame.project(p);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      if (primitive.fill) { ctx.closePath(); ctx.fillStyle = primitive.color; ctx.fill(); }
      else { ctx.strokeStyle = primitive.color; ctx.stroke(); }
    }
    model.board.forEach(paint);
    model.underlay.forEach(paint);
    const faces = model.blocks.flatMap(b => b.faces).filter(f => dot(f.normal, frame.basis.near) > 0);
    faces.sort((a, b) => dot(a.center, frame.basis.near) - dot(b.center, frame.basis.near));
    faces.forEach(face => {
      paint(face.surface); face.bands.forEach(paint); face.strokes.forEach(paint); face.trim.forEach(paint);
    });
    model.selection.forEach(paint);
  }

  function makeFrame(model, camera, width, height, labelWidth) {
    const basis = cameraBasis(camera);
    const reference = cameraBasis(home);
    const points = model.blocks.flatMap(b => b.corners);
    if (!points.length) points.push([0, 0, 0]);
    const center = average(points);
    // Reuse the home scale as the camera turns so that scale stays stable.
    const homeXY = p => [dot(p, reference.right), dot(p, reference.up)];
    const limit = bounds(points.map(homeXY));
    const scale = Math.min(112, (width - Math.min(64, labelWidth * .5)) / Math.max(1, limit.right - limit.left),
      (height - 82) / Math.max(1, limit.bottom - limit.top)) * camera.zoom;
    const mid = [(limit.left + limit.right) / 2, (limit.top + limit.bottom) / 2];
    const tx = width / 2 - scale * (mid[0] - dot(center, reference.right) + dot(center, basis.right));
    const ty = height / 2 - 17 + scale * (mid[1] - dot(center, reference.up) + dot(center, basis.up));
    const project = p => [tx + dot(p, basis.right) * scale, ty - dot(p, basis.up) * scale];
    // Positive view depth is nearer to the viewer, hence the negative z row.
    const matrix = new Float32Array(16);
    for (let i = 0; i < 3; i++) {
      matrix[i * 4] = 2 * scale / width * basis.right[i];
      matrix[i * 4 + 1] = 2 * scale / height * basis.up[i];
      matrix[i * 4 + 2] = -basis.near[i] / 90;
    }
    matrix.set([2 * tx / width - 1, 1 - 2 * ty / height, 0, 1], 12);
    return {matrix, basis, project, width, height};
  }

  function hitFace(x, y, face, frame) {
    const [origin, right, , top] = face.surface.points;
    const a = frame.project(origin);
    const [ux, uy] = diff(frame.project(right), a), [vx, vy] = diff(frame.project(top), a);
    const determinant = ux * vy - uy * vx;
    if (Math.abs(determinant) < .0001) return null;
    // Invert the two projected edges of the face. Coordinates stay anchored
    // to the tensor's axes independently of orbit, zoom, or portrait layout.
    const dx = x - a[0], dy = y - a[1];
    const u = (dx * vy - dy * vx) / determinant, v = (ux * dy - uy * dx) / determinant;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    const world = origin.map((n, i) => n + u * (right[i] - n) + v * (top[i] - n));
    const hit = {depth: dot(world, frame.basis.near)};
    const positions = face.positions;
    if (positions && v <= 1 / positions.queryCount) {
      const fraction = positions.rowAxis ? 1 - v : u;
      hit.cell = Math.min(positions.count - 1, Math.floor(fraction * positions.count));
    }
    return hit;
  }

  window.createQwenTensorView = function createQwenTensorView(container, onSelect) {
    container.classList.add('qwen-tensor-view');
    const canvas = document.createElement('canvas'), flat = document.createElement('canvas');
    const labels = document.createElement('div');
    canvas.className = flat.className = 'qtv-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    flat.setAttribute('aria-hidden', 'true');
    labels.className = 'qtv-labels';
    labels.setAttribute('role', 'group');
    labels.setAttribute('aria-label', 'Tensors in this stage');
    container.append(canvas, flat, labels);

    let camera = {...home}, scene = {nodes: [], edges: [], selected: null};
    let gpu = createGpu(canvas), ctx = null, model = null, frame = null, portrait = null;
    let drag = null, tick = 0;
    const buttons = new Map();

    function setMode() {
      canvas.hidden = !gpu;
      flat.hidden = !!gpu;
      if (!gpu) ctx = flat.getContext('2d');
      container.dataset.renderer = gpu ? 'webgl' : 'canvas2d';
    }
    setMode();

    function draw() {
      tick = 0;
      const width = container.clientWidth, height = container.clientHeight;
      if (width < 1 || height < 1) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const surface = gpu ? canvas : flat;
      const w = Math.round(width * dpr), h = Math.round(height * dpr);
      if (surface.width !== w || surface.height !== h) { surface.width = w; surface.height = h; }
      const vertical = width < 540 && height > width * 1.03;
      if (!model || portrait !== vertical) {
        portrait = vertical;
        model = sceneGeometry(scene, portrait);
        if (gpu) gpu.upload(model);
      }
      const columns = new Set(scene.nodes.map(n => portrait ? n.row : n.col)).size || 1;
      const labelWidth = clamp(width / columns - 9, 65, 166);
      container.style.setProperty('--qtv-label-width', labelWidth + 'px');
      frame = makeFrame(model, camera, width, height, labelWidth);
      if (gpu) gpu.draw(frame.matrix);
      else if (ctx) drawFlat(ctx, model, frame, dpr);
      for (const b of model.blocks) {
        const box = bounds(b.corners.map(frame.project)), button = buttons.get(b.node.id);
        const halfLabel = button.offsetWidth / 2;
        const middle = (box.right + box.left) / 2;
        // Keep labels readable at the edges of the fitted view. On zoom, move
        // them with their geometry, so off-screen tensors do not leave labels.
        const x = camera.zoom <= 1 ? clamp(middle, halfLabel + 2, width - halfLabel - 2) : middle;
        button.style.left = x + 'px';
        button.style.top = (box.bottom + 4) + 'px';
      }
    }
    function invalidate() { if (!tick) tick = requestAnimationFrame(draw); }
    function setScene(next) {
      scene = next;
      const ids = new Set(scene.nodes.map(n => n.id));
      for (const [id, button] of buttons) {
        if (!ids.has(id)) { button.remove(); buttons.delete(id); }
      }
      scene.nodes.forEach(node => {
        let button = buttons.get(node.id);
        if (!button) {
          button = document.createElement('button');
          button.type = 'button';
          button.className = 'qtv-label';
          const label = document.createElement('span'), shape = document.createElement('span');
          shape.className = 'qtv-shape';
          button.append(label, shape);
          button.addEventListener('click', () => onSelect(node.id));
          buttons.set(node.id, button);
          labels.append(button);
        }
        const dimensions = node.shape.join(' × ');
        button.firstChild.textContent = node.label;
        button.lastChild.textContent = dimensions;
        button.setAttribute('aria-label', node.label + (dimensions ? ', shape ' + dimensions : ''));
        button.setAttribute('aria-pressed', String(node.id === scene.selected));
      });
      model = null;
      invalidate();
    }

    function pick(x, y) {
      if (!frame || !model) return;
      let best = null;
      for (const block of model.blocks) for (const face of block.faces) {
        if (dot(face.normal, frame.basis.near) <= 0) continue;
        const hit = hitFace(x, y, face, frame);
        if (hit && (!best || hit.depth > best.depth)) {
          best = {...hit, id: block.node.id};
        }
      }
      if (best && best.cell !== undefined) onSelect(best.id, best.cell);
      else if (best) onSelect(best.id);
    }

    const events = new AbortController(), listen = {signal: events.signal};
    container.addEventListener('pointerdown', event => {
      if (event.button !== 0 || !event.isPrimary || event.target.closest('button')) return;
      drag = {id: event.pointerId, x: event.clientX, y: event.clientY, yaw: camera.yaw, pitch: camera.pitch, moved: false};
      container.setPointerCapture(event.pointerId);
    }, listen);
    container.addEventListener('pointermove', event => {
      if (!drag || event.pointerId !== drag.id) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (!drag.moved) return;
      container.setAttribute('data-dragging', '');
      camera.yaw = clamp(drag.yaw + dx * .007, -1.15, 1.15);
      camera.pitch = clamp(drag.pitch + dy * .007, -.70, 1.05);
      invalidate();
    }, listen);
    container.addEventListener('pointerup', event => {
      if (!drag || event.pointerId !== drag.id) return;
      const box = container.getBoundingClientRect();
      if (!drag.moved) pick(event.clientX - box.left - container.clientLeft, event.clientY - box.top - container.clientTop);
      drag = null;
      container.removeAttribute('data-dragging');
    }, listen);
    container.addEventListener('lostpointercapture', () => {
      drag = null;
      container.removeAttribute('data-dragging');
    }, listen);
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      gpu = null;
      setMode();
      invalidate();
    }, listen);
    canvas.addEventListener('webglcontextrestored', () => {
      gpu = createGpu(canvas);
      model = null;
      setMode();
      invalidate();
    }, listen);
    const observer = new ResizeObserver(invalidate);
    observer.observe(container);
    invalidate();
    return {
      setScene,
      reset() { camera = {...home}; invalidate(); },
      zoom(factor) {
        if (Number.isFinite(factor) && factor > 0) camera.zoom = clamp(camera.zoom * factor, .55, 2.6);
        invalidate();
      },
      destroy() {
        events.abort();
        observer.disconnect();
        cancelAnimationFrame(tick);
        if (gpu) gpu.dispose();
        canvas.remove(); flat.remove(); labels.remove();
        container.classList.remove('qwen-tensor-view');
        container.removeAttribute('data-renderer');
        container.removeAttribute('data-dragging');
        container.style.removeProperty('--qtv-label-width');
      }
    };
  };
})();
