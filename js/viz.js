/*
 * viz.js — everything that draws: nucleus images, weight maps, evidence overlays, the network diagram,
 * and the SVG charts. Colours come from the CSS tokens so light and dark themes both work.
 */
window.Viz = (function () {
  'use strict';

  // ------------------------------------------------------------------ colours
  let C = null;
  function hexToRgb(h) {
    h = h.trim();
    if (h.startsWith('#')) {
      if (h.length === 4) h = '#' + h[1] + h[1] + h[2] + h[2] + h[3] + h[3];
      return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
    }
    const m = h.match(/\d+(\.\d+)?/g);
    return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
  }
  function refreshTheme() {
    const cs = getComputedStyle(document.documentElement);
    const get = k => cs.getPropertyValue(k).trim();
    C = {
      ground: get('--ground'), surface: get('--surface'), surface2: get('--surface-2'), surface3: get('--surface-3'),
      ink: get('--ink'), ink2: get('--ink-2'), ink3: get('--ink-3'), line: get('--line'), lineStrong: get('--line-strong'),
      accent: get('--accent'), accentSoft: get('--accent-soft'), accentInk: get('--accent-ink'),
      regular: get('--regular'), irregular: get('--irregular'), good: get('--good'), bad: get('--bad'), mapZero: get('--map-zero'),
    };
    C.rgb = {};
    for (const k of Object.keys(C)) if (typeof C[k] === 'string') C.rgb[k] = hexToRgb(C[k]);
    return C;
  }
  function colors() { return C || refreshTheme(); }

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  function mixRgb(a, b, t) { return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]; }
  function rgbStr(c, a) { return a === undefined ? `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})` : `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`; }
  function luminance(c) { return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255; }

  // diverging: blue (negative) — neutral — orange (positive); t in [-1, 1]
  function divergingRgb(t) {
    const c = colors().rgb;
    t = clamp(t, -1, 1);
    const k = Math.pow(Math.abs(t), 0.85);
    return t < 0 ? mixRgb(c.mapZero, c.regular, k) : mixRgb(c.mapZero, c.irregular, k);
  }
  function diverging(t, a) { return rgbStr(divergingRgb(t), a); }
  function sequentialRgb(t) { const c = colors().rgb; return mixRgb(c.surface2, c.accent, clamp(t, 0, 1)); }

  // H&E look-up: brightness 0..1 -> pale eosin background to deep hematoxylin
  const HE_STOPS = [[0, [38, 20, 88]], [0.38, [88, 56, 148]], [0.62, [160, 122, 186]], [0.8, [226, 196, 218]], [0.92, [244, 226, 236]], [1, [250, 240, 245]]];
  const HE_LUT = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const t = v / 255;
    let i = 0; while (i < HE_STOPS.length - 2 && t > HE_STOPS[i + 1][0]) i++;
    const [t0, c0] = HE_STOPS[i], [t1, c1] = HE_STOPS[i + 1];
    const k = clamp((t - t0) / (t1 - t0), 0, 1);
    const c = mixRgb(c0, c1, k);
    HE_LUT[v * 3] = c[0]; HE_LUT[v * 3 + 1] = c[1]; HE_LUT[v * 3 + 2] = c[2];
  }
  function pixelRgb(v, tint) { return tint ? [HE_LUT[v * 3], HE_LUT[v * 3 + 1], HE_LUT[v * 3 + 2]] : [v, v, v]; }

  // ------------------------------------------------------------------ image renderers
  const scratch = document.createElement('canvas');
  function imageToCanvas(px, size, tint) {
    // returns a size x size canvas holding the image (cached scratch, so draw it immediately)
    scratch.width = size; scratch.height = size;
    const ctx = scratch.getContext('2d');
    const im = ctx.createImageData(size, size);
    for (let i = 0; i < px.length; i++) {
      const c = pixelRgb(px[i], tint);
      im.data[i * 4] = c[0]; im.data[i * 4 + 1] = c[1]; im.data[i * 4 + 2] = c[2]; im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return scratch;
  }
  function renderThumb(canvas, px, size, tint) {
    if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageToCanvas(px, size, tint), 0, 0);
  }
  // a patient card: one bar per parameter, up = above the reference range, down = below, in the class-free diverging scale
  function renderFingerprint(canvas, dev, big) {
    const n = dev.length, W = big ? 288 : 48, H = big ? 288 : 48;
    const dpr = big ? (window.devicePixelRatio || 1) : 1;
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const c = colors();
    ctx.fillStyle = c.surface2; ctx.fillRect(0, 0, W, H);
    const pad = big ? 18 : 3, bw = (W - 2 * pad) / n, mid = H / 2, amp = (H / 2 - pad) / 3;
    ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, mid); ctx.lineTo(W - pad, mid); ctx.stroke();
    if (big) { ctx.setLineDash([3, 4]); ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(pad, mid - amp); ctx.lineTo(W - pad, mid - amp); ctx.moveTo(pad, mid + amp); ctx.lineTo(W - pad, mid + amp); ctx.stroke(); ctx.setLineDash([]); }
    for (let i = 0; i < n; i++) {
      const d = Math.max(-3, Math.min(3, dev[i]));
      const h = d * amp;
      ctx.fillStyle = diverging(d / 3);
      const x = pad + i * bw + bw * 0.15, w = bw * 0.7;
      if (Math.abs(h) < 1) ctx.fillRect(x, mid - 0.5, w, 1); else ctx.fillRect(x, h > 0 ? mid - h : mid, w, Math.abs(h));
    }
  }
  const UNIT_COLORS = ['#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948', '#2a78d6', '#eb6834'];
  const unitColor = j => UNIT_COLORS[j % UNIT_COLORS.length];
  function sequential(t, a) { return rgbStr(sequentialRgb(t), a); }

  function setupBig(canvas, size) {
    const dpr = window.devicePixelRatio || 1;
    const css = 288;
    const need = Math.round(css * dpr);
    if (canvas.width !== need) { canvas.width = need; canvas.height = need; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, need, need);
    const k = need / size;
    ctx.setTransform(k, 0, 0, k, 0, 0); // now 1 unit = 1 image pixel
    ctx.imageSmoothingEnabled = false;
    return ctx;
  }
  function renderBigImage(canvas, px, size, tint) {
    const ctx = setupBig(canvas, size);
    ctx.drawImage(imageToCanvas(px, size, tint), 0, 0, size, size);
  }
  // evidence: per-pixel signed contribution s (Float64Array size*size); orange pushes to irregular, blue to regular
  function renderEvidence(canvas, px, size, sal, tint) {
    const ctx = setupBig(canvas, size);
    ctx.drawImage(imageToCanvas(px, size, tint), 0, 0, size, size);
    let max = 1e-9;
    for (let i = 0; i < sal.length; i++) max = Math.max(max, Math.abs(sal[i]));
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const t = sal[y * size + x] / max;
      if (Math.abs(t) < 0.03) continue;
      ctx.fillStyle = diverging(t, 0.15 + 0.75 * Math.pow(Math.abs(t), 0.7));
      ctx.fillRect(x, y, 1, 1);
    }
    return max;
  }
  // measurements: membrane contour, convex hull, and the deep region used for darkness/texture
  function renderMeasurement(canvas, px, size, m, tint) {
    const ctx = setupBig(canvas, size);
    const c = colors();
    ctx.drawImage(imageToCanvas(px, size, tint), 0, 0, size, size);
    ctx.fillStyle = rgbStr(c.rgb.accent, 0.22);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (m.deep[y * size + x]) ctx.fillRect(x, y, 1, 1);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (m.hull.length > 2) {
      ctx.setLineDash([0.5, 0.35]); ctx.lineWidth = 0.22; ctx.strokeStyle = tint ? '#ffffff' : c.ink;
      ctx.beginPath(); m.hull.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 0.3; ctx.strokeStyle = c.accent;
    ctx.beginPath();
    for (const [a, b] of m.contour) { ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
    ctx.stroke();
    // equivalent smooth ellipse, dotted
    ctx.save();
    ctx.translate(m.centroid[0], m.centroid[1]); ctx.rotate(m.ellipse.angle);
    ctx.setLineDash([0.25, 0.4]); ctx.lineWidth = 0.18; ctx.strokeStyle = c.irregular;
    ctx.beginPath(); ctx.ellipse(0, 0, m.ellipse.a, m.ellipse.b, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }

  // ------------------------------------------------------------------ network diagram
  const NET_W = 800, NET_H = 440;
  const EDGE_FLOOR = 1.5;                 // |w| at which a connection is drawn at full thickness (weights start near ±0.3)
  const TILE_FLOOR = { map: 0.05, square: 0.08, filter: 0.3 }; // colour scale floors for weight maps, so they emerge from grey
  function fmtSigned(v, d) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d); }
  function fmtNum(v, d) { return v < 0 ? '−' + Math.abs(v).toFixed(d) : v.toFixed(d); }
  const fmtInt = n => n.toLocaleString();

  // small cached canvases for weight maps, filters and feature maps
  const tileCache = new Map();
  function tileCanvas(key, arr, offset, w, h, opt) {
    let cv = tileCache.get(key);
    if (!cv || cv.width !== w || cv.height !== h) { cv = document.createElement('canvas'); cv.width = w; cv.height = h; tileCache.set(key, cv); }
    const ctx = cv.getContext('2d');
    const im = ctx.createImageData(w, h);
    let max = opt && opt.max != null ? opt.max : 1e-9;
    if (!(opt && opt.max != null)) for (let i = 0; i < w * h; i++) max = Math.max(max, Math.abs(arr[offset + i]));
    if (opt && opt.floor) max = Math.max(max, opt.floor);
    max = Math.max(max, 1e-9);
    const limit = opt && opt.maskBefore != null ? opt.maskBefore : w * h;
    for (let i = 0; i < w * h; i++) {
      const v = arr[offset + i] / max;
      const c = opt && opt.mode === 'sequential' ? sequentialRgb(Math.max(0, v)) : divergingRgb(v);
      im.data[i * 4] = c[0]; im.data[i * 4 + 1] = c[1]; im.data[i * 4 + 2] = c[2]; im.data[i * 4 + 3] = i < limit ? 255 : 0;
    }
    ctx.putImageData(im, 0, 0);
    return { canvas: cv, max };
  }

  // A convolutional network's first dense layer reads the K pooled maps as one input. For the diagram they are stacked
  // into a single map (2 columns of maps up to 4 filters, 4 beyond), and so are the weights over them, so the dense
  // stage reads exactly like the pixel recipes: input map × weight map = product map, summed, through the activation.
  function compositeDims(net) { const K = net.conv.K, po = net.po, cols = K <= 4 ? 2 : 4, rowsK = Math.ceil(K / cols); return { K, po, cols, rowsK, cw: cols * po, ch: rowsK * po }; }
  function toComposite(net, arr, off) {
    const { K, po, cols, cw, ch } = compositeDims(net), out = new Float64Array(cw * ch);
    for (let k = 0; k < K; k++) { const bx = (k % cols) * po, by = Math.floor(k / cols) * po; for (let r = 0; r < po; r++) for (let c = 0; c < po; c++) out[(by + r) * cw + bx + c] = arr[off + k * po * po + r * po + c]; }
    return out;
  }
  // the first dense layer's input as a map (the pixels, or the stacked pooled maps once the case has run), or null
  function firstInput(m) {
    const net = m.net;
    if (!net.conv) return m.x ? { vec: m.x, w: m.size, h: m.size, D: net.D } : null;
    const fw = m.fw; if (!fw || !fw.conv) return null;
    const d = compositeDims(net); return { vec: toComposite(net, fw.conv.v, 0), w: d.cw, h: d.ch, D: net.featureCount };
  }
  // unit j's weights over that input (j = null: the output's, without a hidden layer): { arr, off, w, h, D }
  function firstWeights(m, j) {
    const net = m.net, D = net.sizes[0], arr = j == null ? net.Wo : net.W[0], off = j == null ? 0 : j * D;
    if (!net.conv) return { arr, off, w: m.size, h: m.size, D };
    const d = compositeDims(net); return { arr: toComposite(net, arr, off), off: 0, w: d.cw, h: d.ch, D };
  }
  // which cell of such a map a point is over, and its name: a pixel, or a cell of one pooled map
  function cellAt(n, W, x, y) {
    const th = n.h || n.size;
    const i = Math.max(0, Math.min(W.w - 1, Math.floor((x - (n.x - n.size / 2)) / n.size * W.w))), j = Math.max(0, Math.min(W.h - 1, Math.floor((y - (n.y - th / 2)) / th * W.h)));
    return { i, j, k: j * W.w + i };
  }
  function cellName(m, i, j) {
    if (!m.net.conv) return `pixel (${i}, ${j})`;
    const { po, cols } = compositeDims(m.net); return `pooled map ${Math.floor(j / po) * cols + Math.floor(i / po) + 1}, col ${i % po}, row ${j % po}`;
  }
  function drawCompositeDividers(ctx, c, net, x0, y0, w, h) { // thin lines between the stacked maps
    const { cols, rowsK } = compositeDims(net);
    ctx.strokeStyle = c.surface; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 1; i < cols; i++) { ctx.moveTo(x0 + i * w / cols, y0); ctx.lineTo(x0 + i * w / cols, y0 + h); }
    for (let i = 1; i < rowsK; i++) { ctx.moveTo(x0, y0 + i * h / rowsK); ctx.lineTo(x0 + w, y0 + i * h / rowsK); }
    ctx.stroke();
  }

  function spread(n, yc, step) { return Array.from({ length: n }, (_, i) => yc + (i - (n - 1) / 2) * step); }

  /*
   * Layout: columns from left to right depending on the architecture.
   *   measurements: inputs -> [units] -> [units] -> output
   *   pixels:       image -> squares (first-layer weight maps) | map (no hidden) -> [units] -> output
   *   pixels+conv:  image -> filters -> feature maps -> pooled -> [units] -> [units] -> output
   */
  function layoutNetwork(m) {
    const net = m.net, hidden = net.hidden, conv = net.conv;
    const L = { W: NET_W, H: NET_H, nodes: [], edges: [], bands: [], captions: [], ops: [], mode: m.mode };
    const yc = NET_H / 2 + 8;
    const add = n => { L.nodes.push(n); return n; };
    const output = add({ kind: 'output', x: 722, y: yc, r: 26 });
    const unitColumns = []; // arrays of unit nodes per dense hidden layer
    const prev = m.prev || null;
    const prevW = (l, i) => (prev && prev.W[l] ? prev.W[l][i] : null), prevWo = i => (prev ? prev.Wo[i] : null);
    const link = (from, to, w, layer, meta) => L.edges.push(Object.assign({ x1: from.x + (from.r || from.size / 2), y1: from.y, x2: to.x - (to.r || to.size / 2), y2: to.y, w, layer }, meta));

    if (m.mode === 'features') {
      const D = net.D;
      const wide = D > 8 ? 40 : 0; // room for long parameter names
      const xs = hidden.length === 0 ? [150 + wide] : hidden.length === 1 ? [150 + wide, 440 + wide / 2] : [140 + wide, 400 + wide / 2, 585];
      const inputs = spread(D, yc, Math.min(58, (NET_H - 90) / Math.max(1, D - 1))).map((y, i) => add({ kind: 'input', i, x: xs[0], y, r: D > 8 ? 13 : 16 }));
      L.captions.push({ x: xs[0], text: m.inputCaption || `INPUT · ${D} MEASUREMENTS` });
      let prevCol = inputs;
      hidden.forEach((h, l) => {
        const col = spread(h, yc, Math.min(46, (NET_H - 110) / Math.max(1, h - 1))).map((y, j) => add({ kind: 'unit', l, j, x: xs[l + 1], y, r: 17 }));
        unitColumns.push(col);
        L.captions.push({ x: xs[l + 1], text: `HIDDEN ${hidden.length > 1 ? l + 1 : ''} · ${h} ${m.activationLabel.toUpperCase()}` });
        prevCol.forEach((a, i) => col.forEach((b, j) => link(a, b, net.W[l][j * net.sizes[l] + i], l, { wi: j * net.sizes[l] + i, fi: i, ti: j, prev: prevW(l, j * net.sizes[l] + i), fromName: l === 0 ? m.featureNames[i] : `unit ${l}.${i + 1}`, fromIdx: l === 0 ? i : null, toName: `unit ${hidden.length > 1 ? (l + 1) + '.' : ''}${j + 1}` })));
        prevCol = col;
      });
      if (!hidden.length) L.captions.push({ x: 440, text: 'NO HIDDEN LAYER' });
      prevCol.forEach((a, i) => link(a, output, net.Wo[i], 'out', { wi: i, fi: i, prev: prevWo(i), fromName: hidden.length ? `unit ${hidden.length > 1 ? hidden.length + '.' : ''}${i + 1}` : m.featureNames[i], fromIdx: hidden.length ? null : i, toName: 'output' }));
    } else if (!conv) {
      const img = add({ kind: 'image', x: 108, y: yc, w: 124, h: 124 });
      L.captions.push({ x: 108, text: 'INPUT · 1,024 PIXELS' });
      if (!hidden.length) {
        img.x = 96; img.w = 112; img.h = 112;
        const map = add({ kind: 'map', x: 340, y: yc, size: 120 });
        const prod = add({ kind: 'product', x: 545, y: yc, size: 120 });
        L.captions[L.captions.length - 1].x = 96;
        L.bands.push({ pts: [[img.x + img.w / 2, img.y - img.h / 2 + 4], [map.x - map.size / 2, map.y - map.size / 2], [map.x - map.size / 2, map.y + map.size / 2], [img.x + img.w / 2, img.y + img.h / 2 - 4]], label: '1,024 weights' });
        L.captions.push({ x: 340, text: 'WEIGHTS · ONE PER PIXEL' });
        L.captions.push({ x: 545, text: 'WEIGHT × PIXEL' });
        // the row reads image × weights = weight × pixel: the image arrives along the band, so the × sits where the band meets the weight map
        L.ops.push({ x: map.x - map.size / 2 - 12, y: map.y, text: '×' }, { x: (map.x + map.size / 2 + prod.x - prod.size / 2) / 2, y: map.y, text: '=' });
        L.edges.push({ x1: prod.x + prod.size / 2, y1: prod.y, x2: output.x - output.r, y2: output.y, w: 1, layer: 'sum', label: 'Σ → z' });
      } else {
        const xW = hidden.length === 1 ? 330 : 312, xP = xW + 92, xs = [xW, 590];
        const h1 = hidden[0];
        const captioned = (NET_H - 100) / h1 - 18 >= 48; // room for a map of useful size plus its ±max caption below
        const gap = captioned ? 18 : 8;
        const size = Math.min(66, (NET_H - 100) / h1 - gap);
        const ys = spread(h1, yc, size + gap);
        const squares = ys.map((y, j) => add({ kind: 'square', j, x: xW, y, size, captioned }));
        const prods = ys.map((y, j) => add({ kind: 'uprod', j, x: xP, y, size }));
        unitColumns.push(squares);
        L.captions.push({ x: (xW + xP) / 2, text: `HIDDEN ${hidden.length > 1 ? '1' : ''} · ${h1} ${m.activationLabel.toUpperCase()}` });
        L.footnotes = [{ x: xW, text: 'weights' }, { x: xP, text: 'weight × pixel' }, { x: xP + Math.max(size / 2 + 62, 88), text: `Σ → ${m.activationLabel}` }];
        for (const s of squares) {
          L.bands.push({ pts: [[img.x + img.w / 2, img.y - img.h / 2 + 6], [s.x - s.size / 2, s.y - s.size / 2], [s.x - s.size / 2, s.y + s.size / 2], [img.x + img.w / 2, img.y + img.h / 2 - 6]], j: s.j });
          // each row reads image × weights = weight × pixel: the image arrives along the band, so the × sits where the band meets the weight map
          L.ops.push({ x: s.x - s.size / 2 - 12, y: s.y, text: '×' }, { x: (xW + xP) / 2, y: s.y, text: '=' });
        }
        L.bandLabel = { x: (img.x + img.w / 2 + xW - size / 2) / 2, text: '1,024 weights per unit · each map scaled to its own max' };
        const badgeX = xP + size / 2 + 16; // the activation badge sits after the product map: Σ weight × pixel + bias, through the activation
        L.badgeX = badgeX;
        let prevCol = prods;
        if (hidden.length > 1) {
          const col = spread(hidden[1], yc, Math.min(46, (NET_H - 110) / Math.max(1, hidden[1] - 1))).map((y, j) => add({ kind: 'unit', l: 1, j, x: xs[1], y, r: 17 }));
          unitColumns.push(col);
          L.captions.push({ x: xs[1], text: `HIDDEN 2 · ${hidden[1]} ${m.activationLabel.toUpperCase()}` });
          prevCol.forEach((a, i) => col.forEach((b, j) => L.edges.push({ x1: badgeX + 12, y1: a.y, x2: b.x - b.r, y2: b.y, w: net.W[1][j * net.sizes[1] + i], wi: j * net.sizes[1] + i, fi: i, ti: j, prev: prevW(1, j * net.sizes[1] + i), layer: 1, fromName: `unit 1.${i + 1}`, toName: `unit 2.${j + 1}` })));
          prevCol = col;
        }
        prevCol.forEach((a, i) => L.edges.push({ x1: a.r ? a.x + a.r : badgeX + 12, y1: a.y, x2: output.x - output.r, y2: output.y, w: net.Wo[i], wi: i, fi: i, prev: prevWo(i), layer: 'out', fromName: `unit ${hidden.length > 1 ? '2.' : ''}${i + 1}`, toName: 'output' }));
      }
    } else {
      const K = conv.K;
      const img = add({ kind: 'image', x: 80, y: yc, w: 100, h: 100 });
      L.captions.push({ x: 30, text: 'INPUT · 1,024 PX', align: 'left' });
      const rowStep = (NET_H - 100) / K;
      // tiles grow when there are few filters (four rows leave room for 48/64/40 px tiles); the columns follow the tile widths
      const fs = Math.min(48, rowStep - 6), ms = Math.min(64, rowStep - 4), ps = Math.min(40, rowStep - 10.5);
      const xF = 228, xM = ms > 50 ? xF + fs / 2 + 26 + ms / 2 : 298, xP = ms > 50 ? xM + ms / 2 + 26 + ps / 2 : 366;
      const ys = spread(K, yc, rowStep);
      const filters = ys.map((y, k) => add({ kind: 'filter', k, x: xF, y, size: fs }));
      const fmaps = ys.map((y, k) => add({ kind: 'fmap', k, x: xM, y, size: ms }));
      const pooled = ys.map((y, k) => add({ kind: 'pooled', k, x: xP, y, size: ps }));
      L.captions.push({ x: (xF + xP) / 2 + 8, text: `CONV · ${K} FILTERS ${conv.f}×${conv.f} · POOL ${conv.pool}×${conv.pool}` }); // the short form: the dense captions need the room
      L.bands.push({ pts: [[img.x + img.w / 2, img.y - img.h / 2], [xF - fs / 2 - 4, ys[0] - fs / 2], [xF - fs / 2 - 4, ys[K - 1] + fs / 2], [img.x + img.w / 2, img.y + img.h / 2]], label: '' });
      L.bandLabel = { x: (img.x + img.w / 2 + xF - fs / 2) / 2, y: 44, text: 'filters slide over the image' };
      L.footnotes = [{ x: xF, text: 'filters' }, { x: xM, text: 'feature maps' }, { x: xP, text: 'pooled' }];
      L.poolCaption = { x: xP, text: `each cell = largest of a ${conv.pool}×${conv.pool} block` };
      // the dense stage reads like the pixel recipes: the pooled maps, stacked into one map, × each unit's weight map
      // (its weights over the pooled cells, stacked the same way) = a product map, summed into the unit's badge
      const F = net.featureCount, comp = compositeDims(net);
      const poolBox = { x1: xP + ps / 2, yTop: ys[0] - ps / 2, yBot: ys[K - 1] + ps / 2 };
      const two = hidden.length > 1, twMax = two ? 48 : 64, cgap = two ? 22 : 26;
      const stacked = `stacked ${comp.rowsK}×${comp.cols}`;
      if (!hidden.length) {
        const tw = twMax, th = Math.round(tw * comp.ch / comp.cw);
        const xS = poolBox.x1 + cgap + tw / 2, xU = xS + tw / 2 + cgap + tw / 2;
        const map = add({ kind: 'map', x: xS, y: yc, size: tw, h: th });
        const prod = add({ kind: 'product', x: xU, y: yc, size: tw, h: th });
        L.bands.push({ pts: [[poolBox.x1, poolBox.yTop], [map.x - tw / 2, map.y - th / 2], [map.x - tw / 2, map.y + th / 2], [poolBox.x1, poolBox.yBot]], role: 'pool' });
        L.bandLabel2 = { x: (poolBox.x1 + xS - tw / 2) / 2, y: 44, text: `${fmtInt(F)} weights · ${stacked}` };
        L.captions.push({ x: (xS + xU) / 2, text: 'WEIGHT × POOLED' });
        L.footnotes.push({ x: xS, text: 'weights' }, { x: xU, text: 'products' });
        L.ops.push({ x: map.x - tw / 2 - 12, y: map.y, text: '×' }, { x: (xS + xU) / 2, y: map.y, text: '=' });
        L.edges.push({ x1: prod.x + tw / 2, y1: prod.y, x2: output.x - output.r, y2: output.y, w: 1, layer: 'sum', label: 'Σ → z' });
      } else {
        const h1 = hidden[0], rowAvail = (NET_H - 100) / h1, thMax = Math.round(twMax * comp.ch / comp.cw);
        const captioned = rowAvail - 18 >= thMax + 2, gap = captioned ? 18 : 8; // room for the ±max caption below each map
        const th = Math.min(thMax, rowAvail - gap), tw = Math.round(th * comp.cw / comp.ch);
        const xS = poolBox.x1 + cgap + tw / 2, xU = xS + tw / 2 + cgap + tw / 2;
        const ysU = h1 === K ? ys : spread(h1, yc, th + gap); // one unit per filter: the rows line up with the pooled maps
        const squares = ysU.map((y, j) => add({ kind: 'square', j, x: xS, y, size: tw, h: th, captioned }));
        const prods = ysU.map((y, j) => add({ kind: 'uprod', j, x: xU, y, size: tw, h: th }));
        unitColumns.push(squares);
        L.captions.push({ x: (xS + xU) / 2, text: `HIDDEN ${two ? '1' : ''} · ${h1} ${m.activationLabel.toUpperCase()}` });
        L.footnotes.push({ x: xS, text: 'weights' }, { x: xU, text: tw >= 56 ? 'weight × pooled' : 'products' });
        for (const sq of squares) {
          L.bands.push({ pts: [[poolBox.x1, poolBox.yTop], [sq.x - tw / 2, sq.y - th / 2], [sq.x - tw / 2, sq.y + th / 2], [poolBox.x1, poolBox.yBot]], j: sq.j, role: 'pool' });
          L.ops.push({ x: sq.x - tw / 2 - 12, y: sq.y, text: '×' }, { x: (xS + xU) / 2, y: sq.y, text: '=' });
        }
        L.bandLabel2 = { x: (poolBox.x1 + xS - tw / 2) / 2, y: 44, text: `${fmtInt(F)} weights per unit · ${stacked}` };
        const badgeX = xU + tw / 2 + 16; // the activation badge sits after the product map, as in the pixel layout
        L.badgeX = badgeX;
        let prevCol = prods;
        if (two) {
          const x2 = 625;
          const col = spread(hidden[1], yc, Math.min(46, (NET_H - 110) / Math.max(1, hidden[1] - 1))).map((y, j) => add({ kind: 'unit', l: 1, j, x: x2, y, r: 17 }));
          unitColumns.push(col);
          L.captions.push({ x: x2, text: `HIDDEN 2 · ${hidden[1]} ${m.activationLabel.toUpperCase()}` });
          prevCol.forEach((a, i) => col.forEach((b, j) => L.edges.push({ x1: badgeX + 12, y1: a.y, x2: b.x - b.r, y2: b.y, w: net.W[1][j * net.sizes[1] + i], wi: j * net.sizes[1] + i, fi: i, ti: j, prev: prevW(1, j * net.sizes[1] + i), layer: 1, fromName: `unit 1.${i + 1}`, toName: `unit 2.${j + 1}` })));
          prevCol = col;
        }
        prevCol.forEach((a, i) => L.edges.push({ x1: a.r ? a.x + a.r : badgeX + 12, y1: a.y, x2: output.x - output.r, y2: output.y, w: net.Wo[i], wi: i, fi: i, prev: prevWo(i), layer: 'out', fromName: `unit ${two ? '2.' : ''}${i + 1}`, toName: 'output' }));
      }
    }
    L.captions.push({ x: output.x, text: 'OUTPUT' });
    L.output = output; L.unitColumns = unitColumns;
    if (m.contrast) { // the foundation model's encoder: no output, the code units feed the pair panel on the right
      L.nodes = L.nodes.filter(n => n.kind !== 'output'); L.edges = L.edges.filter(e => e.layer !== 'out'); L.output = null;
      L.captions = L.captions.filter(cap => cap.text !== 'OUTPUT');
      for (const cap of L.captions) if (cap.text.startsWith('HIDDEN')) { cap.text = `CODE · ${net.hidden[0]} NUMBERS`; cap.x += 22; }
      L.captions.push({ x: 695, text: 'THE PAIR' });
      L.pair = { x: 604, w: 186 };
    }
    return L;
  }

  function fitCanvas(canvas, W, H) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || W;
    const cssH = cssW * H / W;
    const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    const ctx = canvas.getContext('2d');
    const k = bw / W;
    ctx.setTransform(k, 0, 0, k, 0, 0);
    return ctx;
  }

  function circleNode(ctx, x, y, r, fill, text, font, ring) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = ring ? 2 : 1.5; ctx.strokeStyle = ring ? colors().ink : colors().lineStrong; ctx.stroke();
    if (text != null) {
      const rgb = typeof fill === 'string' && fill.startsWith('rgb') ? hexToRgb(fill) : hexToRgb(fill);
      ctx.fillStyle = luminance(rgb) < 0.5 ? '#ffffff' : colors().ink;
      ctx.font = font || `500 11px "IBM Plex Mono", ui-monospace, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y + 0.5);
    }
  }
  function unitFill(a, layerActs, signed) {
    if (a == null) return colors().surface2;
    if (signed) return diverging(Math.max(-1, Math.min(1, a)));
    let max = 1; for (let i = 0; i < layerActs.length; i++) max = Math.max(max, Math.abs(layerActs[i]));
    return rgbStr(sequentialRgb(Math.abs(a) / max));
  }

  /*
   * m = { net, mode, x (standardized input or null), fw (forward result or null), featureNames, specimen, size, tint,
   *       stage (0 input only, 1 + hidden, 2 + output; default 2), hover, activation, activationLabel, positiveName }
   */
  function drawNetwork(canvas, m) {
    const ctx = fitCanvas(canvas, NET_W, NET_H);
    const c = colors();
    const L = layoutNetwork(m);
    canvas._layout = L;
    const stage = m.stage == null ? 2 : m.stage;
    const net = m.net, fw = m.fw;
    const signed = ACT_SIGNED(m.activation);
    ctx.clearRect(0, 0, NET_W, NET_H);
    ctx.fillStyle = c.surface; ctx.fillRect(0, 0, NET_W, NET_H);
    const labelFont = `500 12px "IBM Plex Sans", system-ui, sans-serif`;
    const monoFont = `500 11px "IBM Plex Mono", ui-monospace, monospace`;
    const capFont = `600 11px "IBM Plex Sans", system-ui, sans-serif`;
    const hovered = m.hover ? m.hover.ref : null;

    // captions
    ctx.font = capFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const cap of L.captions) { ctx.textAlign = cap.align || 'center'; ctx.fillText(cap.text, cap.x, 10); }
    ctx.textAlign = 'center';
    if (L.footnotes) { ctx.font = monoFont; for (const f of L.footnotes) ctx.fillText(f.text, f.x, 25); }
    if (L.poolCaption) { ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(L.poolCaption.text, L.poolCaption.x, NET_H - 16); ctx.textBaseline = 'top'; }

    // bands
    for (const b of L.bands) {
      ctx.beginPath(); b.pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
      ctx.fillStyle = rgbStr(c.rgb.accent, 0.07); ctx.fill();
      ctx.strokeStyle = rgbStr(c.rgb.accent, 0.35); ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    if (L.bandLabel && !(m.anim && net.conv)) ctx.fillText(L.bandLabel.text, L.bandLabel.x, L.bandLabel.y || NET_H - 16); // the walk-through puts the preprocessing row there
    if (L.bandLabel2) ctx.fillText(L.bandLabel2.text, L.bandLabel2.x, L.bandLabel2.y || NET_H - 16);
    if (L.bands.length && L.bands[0].label) ctx.fillText(L.bands[0].label, L.bandLabel ? L.bandLabel.x : 300, NET_H - 16);
    // operator glyphs: image × weights = weight × pixel
    ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2;
    for (const op of L.ops) ctx.fillText(op.text, op.x, op.y);

    // edges, weak first so strong ones sit on top. Thickness is |w| against a fixed floor (EDGE_FLOOR), so connections
    // visibly grow from thin to thick during training instead of being re-normalised every frame.
    const maxAbs = {}, maxDelta = {};
    for (const e of L.edges) {
      maxAbs[e.layer] = Math.max(maxAbs[e.layer] || 1e-9, Math.abs(e.w));
      e.delta = e.prev == null ? 0 : Math.abs(e.w - e.prev);
      maxDelta[e.layer] = Math.max(maxDelta[e.layer] || 1e-12, e.delta);
    }
    const norm = layer => Math.max(EDGE_FLOOR, maxAbs[layer] || 0);
    const edges = L.edges.slice().sort((a, b) => Math.abs(a.w) / norm(a.layer) - Math.abs(b.w) / norm(b.layer));
    for (const e of edges) {
      const rel = Math.min(1, Math.abs(e.w) / norm(e.layer));
      const hov = hovered === e;
      let width;
      if (e.layer === 'sum') { ctx.strokeStyle = c.lineStrong; ctx.lineWidth = width = 2; }
      else {
        const rgb = e.w < 0 ? c.rgb.regular : c.rgb.irregular;
        ctx.strokeStyle = rgbStr(rgb, hov ? 1 : 0.15 + 0.85 * rel);
        ctx.lineWidth = width = (0.6 + 5.5 * Math.pow(rel, 0.9)) * (hov ? 1.4 : 1);
      }
      ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke();
      // spark: a bright core on the connections the last training step moved most
      if (e.layer !== 'sum' && e.delta > 0 && maxDelta[e.layer] > 1e-9) {
        const drel = e.delta / maxDelta[e.layer];
        if (drel > 0.2) {
          ctx.strokeStyle = rgbStr(c.rgb.ink, 0.15 + 0.6 * drel); ctx.lineWidth = Math.max(0.8, width * 0.3);
          ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke();
        }
      }
      if (e.label) { ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(e.label, (e.x1 + e.x2) / 2, e.y1 - 6); }
    }

    // nodes
    const pending = c.surface2;
    let fmapMax = 1e-9;
    if (fw && fw.conv) for (let i = 0; i < fw.conv.act.length; i++) fmapMax = Math.max(fmapMax, fw.conv.act[i]);
    for (const n of L.nodes) {
      const isHov = hovered === n;
      if (n.kind === 'input') {
        const z = m.x ? m.x[n.i] : null;
        circleNode(ctx, n.x, n.y, n.r, z == null ? pending : diverging(Math.max(-1, Math.min(1, z / 2.5))), z == null ? '·' : fmtSigned(z, 1), null, isHov);
        ctx.font = labelFont; ctx.fillStyle = c.ink2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(m.featureNames[n.i], n.x - n.r - 10, n.y);
      } else if (n.kind === 'image') {
        const x0 = n.x - n.w / 2, y0 = n.y - n.h / 2;
        ctx.fillStyle = c.surface2; ctx.fillRect(x0 - 4, y0 - 4, n.w + 8, n.h + 8);
        if (m.specimen) { ctx.imageSmoothingEnabled = false; ctx.drawImage(imageToCanvas(m.specimen.px, m.size, m.tint), x0, y0, n.w, n.h); }
        else { ctx.font = labelFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('select a nucleus', n.x, n.y); }
        ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(x0 - 4, y0 - 4, n.w + 8, n.h + 8);
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(m.specimen ? `${m.size}×${m.size} ink, mean removed` : '', n.x, y0 + n.h + 12);
      } else if (n.kind === 'map' || n.kind === 'square' || n.kind === 'filter' || n.kind === 'product' || n.kind === 'uprod') {
        let tile = null;
        const tw = n.size, th = n.h || n.size, inp = n.kind === 'product' || n.kind === 'uprod' ? firstInput(m) : null;
        const shown = m.reveal != null ? m.reveal >= 1 : stage >= 1;
        if (n.kind === 'map') { const W = firstWeights(m, null); tile = tileCanvas('out', W.arr, W.off, W.w, W.h, { floor: TILE_FLOOR.map }); }
        else if (n.kind === 'square') { const W = firstWeights(m, n.j); tile = tileCanvas('h' + n.j, W.arr, W.off, W.w, W.h, { floor: TILE_FLOOR.square }); }
        else if (n.kind === 'filter') tile = tileCanvas('f' + n.k, net.Wc, n.k * net.conv.f * net.conv.f, net.conv.f, net.conv.f, { floor: TILE_FLOOR.filter });
        else if (n.kind === 'product' && inp && shown) { const W = firstWeights(m, null), prod = new Float64Array(inp.D); let sum = 0; for (let i = 0; i < inp.D; i++) { prod[i] = W.arr[W.off + i] * inp.vec[i]; sum += prod[i]; } tile = tileCanvas('prod', prod, 0, inp.w, inp.h); tile.sum = sum; }
        else if (n.kind === 'uprod' && inp && shown) { const W = firstWeights(m, n.j), prod = new Float64Array(inp.D); for (let i = 0; i < inp.D; i++) prod[i] = W.arr[W.off + i] * inp.vec[i]; tile = tileCanvas('uprod' + n.j, prod, 0, inp.w, inp.h); }
        ctx.imageSmoothingEnabled = false;
        if (tile) { ctx.drawImage(tile.canvas, n.x - tw / 2, n.y - th / 2, tw, th); if (net.conv && n.kind !== 'filter') drawCompositeDividers(ctx, c, net, n.x - tw / 2, n.y - th / 2, tw, th); }
        else { ctx.fillStyle = pending; ctx.fillRect(n.x - tw / 2, n.y - th / 2, tw, th); }
        ctx.strokeStyle = isHov ? c.ink : c.lineStrong; ctx.lineWidth = isHov ? 2 : 1;
        ctx.strokeRect(n.x - tw / 2, n.y - th / 2, tw, th);
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const learning = m.lesson && (m.lesson.phase === 'gradient' || m.lesson.phase === 'update'); // the gradient tile's caption takes this spot
        if ((n.kind === 'map' || (n.kind === 'square' && n.captioned)) && !learning) ctx.fillText(`±${tile.max.toFixed(2)}`, n.x, n.y + th / 2 + 3);
        if (n.kind === 'product') ctx.fillText(tile ? (tw >= 100 ? `Σ = ${fmtSigned(tile.sum, 2)}  (bias ${fmtSigned(net.bo, 2)})` : `Σ ${fmtSigned(tile.sum, 2)} + b ${fmtSigned(net.bo, 2)}`) : (m.reveal != null ? '' : 'select a nucleus'), n.x, n.y + th / 2 + 3);
        if (n.kind === 'uprod') { // activation badge: sum of the products plus the bias, through the activation
          const a = fw && (m.reveal != null ? m.reveal >= 1 : stage >= 1) ? fw.a[1][n.j] : null;
          circleNode(ctx, n.x + n.size / 2 + 16, n.y, 11, unitFill(a, fw ? fw.a[1] : [], signed), a == null ? null : (Math.abs(a) >= 10 ? a.toFixed(0) : a.toFixed(1)), `500 10px "IBM Plex Mono", ui-monospace, monospace`);
        }
      } else if (n.kind === 'fmap' || n.kind === 'pooled') {
        const side = n.kind === 'fmap' ? net.co : net.po;
        const anim = m.anim;
        const show = fw && fw.conv && (stage >= 1 || anim);
        ctx.fillStyle = pending; ctx.fillRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        if (show) {
          const arr = n.kind === 'fmap' ? fw.conv.act : fw.conv.v;
          let maskBefore = null;
          if (anim) {
            if (n.kind === 'fmap') maskBefore = anim.phase === 'prep' ? 0 : anim.phase === 'scan1' ? (n.k === 0 ? anim.pos : 0) : anim.phase === 'scan' ? (n.k === 0 ? side * side : anim.pos) : side * side;
            else maskBefore = anim.phase === 'pool1' ? (n.k === 0 ? anim.posP : 0) : anim.phase === 'pool' ? (n.k === 0 ? side * side : anim.posP) : (anim.phase === 'prep' || anim.phase === 'scan1' || anim.phase === 'scan') ? 0 : side * side;
          }
          const tile = tileCanvas(n.kind + n.k, arr, n.k * side * side, side, side, { mode: 'sequential', max: fmapMax, maskBefore });
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(tile.canvas, n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        }
        ctx.strokeStyle = isHov ? c.ink : c.lineStrong; ctx.lineWidth = isHov ? 2 : 1;
        ctx.strokeRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        // highlights: the cell being written during the scan, the block being pooled, or the hovered position
        const cs = n.size / side;
        if (n.kind === 'fmap' && anim && anim.pos > 0 && ((anim.phase === 'scan1' && n.k === 0) || (anim.phase === 'scan' && n.k > 0))) {
          const p = anim.pos - 1, py = Math.floor(p / side), px = p % side;
          ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.strokeRect(n.x - n.size / 2 + px * cs - 1, n.y - n.size / 2 + py * cs - 1, cs + 2, cs + 2);
        }
        if (anim && (anim.phase === 'pool1' || anim.phase === 'pool') && anim.posP > 0 && ((anim.phase === 'pool1' && n.k === 0) || (anim.phase === 'pool' && n.k > 0))) {
          const p = anim.posP - 1, py = Math.floor(p / net.po), px = p % net.po, bs = net.conv.pool;
          ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5;
          if (n.kind === 'fmap') ctx.strokeRect(n.x - n.size / 2 + px * bs * cs, n.y - n.size / 2 + py * bs * cs, bs * cs, bs * cs);
          else ctx.strokeRect(n.x - n.size / 2 + px * cs - 1, n.y - n.size / 2 + py * cs - 1, cs + 2, cs + 2);
        }
        // hover linkage: a feature-map pixel marks the pooled cell it feeds; a pooled cell marks its source block
        if (!anim && m.hover && m.hover.ref && m.hover.i != null && m.hover.ref.k === n.k) {
          const bs = net.conv.pool;
          ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5;
          if (m.hover.ref.kind === 'fmap' && n.kind === 'pooled') { const px = Math.floor(m.hover.i / bs), py = Math.floor(m.hover.j / bs); ctx.strokeRect(n.x - n.size / 2 + px * cs - 1, n.y - n.size / 2 + py * cs - 1, cs + 2, cs + 2); }
          if (m.hover.ref.kind === 'pooled' && n.kind === 'fmap') ctx.strokeRect(n.x - n.size / 2 + m.hover.i * bs * cs, n.y - n.size / 2 + m.hover.j * bs * cs, bs * cs, bs * cs);
          if (m.hover.ref.kind === 'pooled' && n.kind === 'pooled') ctx.strokeRect(n.x - n.size / 2 + m.hover.i * cs - 1, n.y - n.size / 2 + m.hover.j * cs - 1, cs + 2, cs + 2);
        }
        if (n.kind === 'fmap' && hovered === n && m.hover && m.hover.i != null) {
          ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.strokeRect(n.x - n.size / 2 + m.hover.i * cs - 1, n.y - n.size / 2 + m.hover.j * cs - 1, cs + 2, cs + 2);
        }
        // small arrows between filter -> map -> pooled
        const prevX = n.kind === 'fmap' ? n.x - n.size / 2 - 8 : n.x - n.size / 2 - 6;
        ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(prevX - 8, n.y); ctx.lineTo(prevX, n.y); ctx.lineTo(prevX - 3, n.y - 3); ctx.moveTo(prevX, n.y); ctx.lineTo(prevX - 3, n.y + 3); ctx.stroke();
      } else if (n.kind === 'unit') {
        const acts = fw && (m.reveal != null ? m.reveal >= n.l + 1 : stage >= 1) ? fw.a[n.l + 1] : null;
        const a = acts ? acts[n.j] : null;
        circleNode(ctx, n.x, n.y, n.r, unitFill(a, acts || [], signed), a == null ? '·' : fmtNum(a, 2), null, isHov);
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        if (L.unitColumns[n.l].length <= 8 && !(m.lesson && m.lesson.phase !== 'forward' && m.lesson.phase !== 'loss')) ctx.fillText(`b ${fmtSigned(net.b[n.l][n.j], 2)}`, n.x, n.y + n.r + 4);
      } else if (n.kind === 'output' && m.scoreLabel) { // a score, not a probability: the attention scorer's output
        const z = fw && stage >= 2 ? fw.z : null;
        circleNode(ctx, n.x, n.y, n.r, z == null ? pending : diverging(Math.tanh(z / 2)), z == null ? '?' : fmtSigned(z, 1), `600 14px "IBM Plex Mono", ui-monospace, monospace`, isHov);
        ctx.font = labelFont; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(m.scoreLabel, n.x, n.y + n.r + 6);
        ctx.font = monoFont; ctx.fillStyle = c.ink3;
        ctx.fillText(`b ${fmtSigned(net.bo, 2)}`, n.x, n.y + n.r + 24);
        if (m.scoreNote) String(m.scoreNote).split("\n").forEach((line, k) => { const w = ctx.measureText(line).width; ctx.fillText(line, Math.min(n.x, NET_W - 6 - w / 2), n.y + n.r + 40 + 16 * k); }); // kept inside the canvas
      } else if (n.kind === 'output') {
        const p = fw && (m.reveal != null ? m.reveal >= m.hops : stage >= 2) ? fw.p : null;
        circleNode(ctx, n.x, n.y, n.r, p == null ? pending : diverging((p - 0.5) * 2), p == null ? '?' : p.toFixed(2), `600 14px "IBM Plex Mono", ui-monospace, monospace`, isHov);
        ctx.font = labelFont; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(`P(${m.positiveName})`, n.x, n.y + n.r + 6);
        ctx.font = monoFont; ctx.fillStyle = c.ink3;
        ctx.fillText(`b ${fmtSigned(net.bo, 2)}`, n.x, n.y + n.r + 24);
        if (p != null) {
          ctx.font = `700 13px "Bricolage Grotesque", "IBM Plex Sans", system-ui, sans-serif`;
          ctx.fillStyle = p >= 0.5 ? c.irregular : c.regular; ctx.textBaseline = 'bottom';
          ctx.fillText((p >= 0.5 ? m.positiveName : m.negativeName).toUpperCase(), n.x, n.y - n.r - 8);
        }
      }
    }
    if (m.lesson) drawLesson(ctx, L, m);
    else if (m.sweep) drawTestSweep(ctx, L, m);
    else if (m.truth) drawTruthLabel(ctx, colors(), L.output, m.truth.y, m.truth.name, m.truth.mark);
    // convolution walk-through: the preprocessing row, the sliding window on the difference image with the arithmetic of
    // the current position, the pooling of one block, each unit's weight maps and products beside the pooled maps, the
    // wipes to the output, then the call and the truth. Hovering a first-layer unit shows its maps in the frozen diagram.
    if (net.conv && m.x) {
      const img = L.nodes.find(n => n.kind === 'image'), anim = m.anim;
      if (anim && img) { if (anim.phase === 'prep') drawPrep(ctx, L, c, m, anim.t); else drawPrepDone(ctx, L, c, m); }
      let spot = null;
      if (m.anim && (m.anim.phase === 'scan1' || m.anim.phase === 'scan') && m.anim.pos > 0) { const p = m.anim.pos - 1; spot = { k: m.anim.showFilter, oy: Math.floor(p / net.co), ox: p % net.co }; }
      else if (!m.anim && m.hover && m.hover.ref && m.hover.ref.kind === 'fmap' && m.hover.i != null) spot = { k: m.hover.ref.k, oy: m.hover.j, ox: m.hover.i };
      if (spot && img) {
        drawWindow(ctx, img, m.size, spot.oy, spot.ox, net.conv.f);
        drawConvPanel(ctx, m, spot.k, spot.oy, spot.ox, { x: 24, y: img.y + img.h / 2 + 40 });
      } else if (img) {
        let cellSpot = null;
        if (m.anim && (m.anim.phase === 'pool1' || m.anim.phase === 'pool') && m.anim.posP > 0) { const p = m.anim.posP - 1; cellSpot = { k: m.anim.phase === 'pool1' ? 0 : 1, py: Math.floor(p / net.po), px: p % net.po }; }
        else if (!m.anim && m.hover && m.hover.ref && m.hover.ref.kind === 'pooled' && m.hover.i != null) cellSpot = { k: m.hover.ref.k, py: m.hover.j, px: m.hover.i };
        if (cellSpot && fw && fw.conv) drawPoolPanel(ctx, m, cellSpot.k, cellSpot.py, cellSpot.px, { x: 24, y: img.y + img.h / 2 + 40 }, fmapMax);
      }
      if (m.lesson && anim && anim.phase === 'lesson') drawLessonConv(ctx, L, c, m); // the blame in the pooled maps and on the feature maps, the filters' gradients
      if (anim && anim.phase === 'units') drawImageHop(ctx, L, c, m, anim.t); // the stacked pooled maps × each unit's weight map, as in the pixel recipes
      if (anim && anim.wipes) for (const key of anim.wipes) { const t = key === anim.key ? anim.t : 1; if (key === 'sum') drawSumEdge(ctx, L, c, m, t); else drawHopWipe(ctx, L, c, m, key, t); } // finished hops stay drawn, the current one grows
      if (anim && anim.phase === 'reveal') drawTruthLabel(ctx, c, L.output, anim.y, anim.truthName, anim.correct ? '✓' : '✗');
      if (anim && anim.banner) drawBanner(ctx, c, L, anim.banner, anim.dir || 0);
    }
    if (m.contrast && L.pair) drawPairPanel(ctx, L, c, m);
    return L;
  }
  function ACT_SIGNED(name) { return name === 'tanh' || name === 'linear'; }

  // ---- the foundation model's pretraining
  // The pair panel beside the encoder: the code of the nucleus in the diagram (view 1), the other view of the same
  // nucleus with its code, their cosine similarity (to be pulled up), a few other nuclei of the batch with theirs (to be
  // pushed down), and the loss this view contributes. m.contrast = { view2Px, code1, code2, cos12, others: [{ px, cos }],
  // loss, share, view2Label }
  function codeStrip(ctx, c, x, y, w, h, code, scale) {
    const n = code.length, bw = w / n, mid = y + h / 2;
    let mx = scale || 1e-9; if (!scale) for (const v of code) mx = Math.max(mx, Math.abs(v));
    ctx.fillStyle = c.surface2; ctx.fillRect(x, y, w, h);
    for (let j = 0; j < n; j++) { const v = code[j] / mx, bh = Math.abs(v) * (h / 2 - 2); ctx.fillStyle = diverging(v); ctx.fillRect(x + j * bw + 1, v >= 0 ? mid - bh : mid, bw - 2, bh); }
    ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, mid + 0.5); ctx.lineTo(x + w, mid + 0.5); ctx.stroke();
    ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
  }
  function drawPairPanel(ctx, L, c, m) {
    const p = m.contrast, x0 = L.pair.x, w = L.pair.w, mono9 = `500 9px "IBM Plex Mono", ui-monospace, monospace`, mono11 = `500 11px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.imageSmoothingEnabled = false;
    ctx.font = mono9; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    // view 1: the nucleus in the diagram; its code is the badges, drawn again as a strip
    ctx.fillStyle = c.ink3; ctx.fillText('view 1 · in the diagram', x0, 36);
    ctx.fillStyle = c.ink2; ctx.fillText(p.view1Label, x0, 47);
    codeStrip(ctx, c, x0, 60, w, 28, p.code1);
    // view 2: the same nucleus seen differently, through the same encoder
    ctx.fillStyle = c.ink3; ctx.fillText('view 2 · the same nucleus, again', x0, 98);
    ctx.fillStyle = c.ink2; ctx.fillText(p.view2Label, x0, 109);
    ctx.drawImage(imageToCanvas(p.view2Px, m.size, m.tint), x0, 122, 44, 44);
    ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, 121.5, 45, 45);
    codeStrip(ctx, c, x0 + 54, 130, w - 54, 28, p.code2);
    // their similarity: what the loss pulls up
    ctx.font = mono11; ctx.fillStyle = c.ink; ctx.fillText(`cosine ${p.cos12.toFixed(2)}`, x0, 176);
    ctx.font = mono9; ctx.fillStyle = c.irregular; ctx.fillText('the same nucleus: pull together ↑', x0, 190);
    // others in the batch: what the loss pushes down (the three whose codes sit nearest)
    ctx.fillStyle = c.ink3; ctx.fillText('others in the batch: push apart ↓', x0, 212);
    let impostor = false;
    p.others.forEach((o, k) => {
      const ox = x0 + k * 62;
      ctx.drawImage(imageToCanvas(o.px, m.size, m.tint), ox, 224, 44, 44);
      ctx.strokeStyle = c.lineStrong; ctx.strokeRect(ox - 0.5, 223.5, 45, 45);
      const nearer = o.cos > p.cos12; impostor = impostor || nearer;
      ctx.fillStyle = nearer ? c.bad : c.ink2; ctx.textAlign = 'center'; ctx.fillText(o.cos.toFixed(2), ox + 22, 272);
    });
    ctx.textAlign = 'left';
    if (impostor) { ctx.fillStyle = c.bad; ctx.fillText('red: nearer than the pair itself', x0, 285); }
    // the loss for view 1: the pair's share of the softmax over every other view of the batch
    ctx.fillStyle = c.ink3; ctx.fillText('softmax over the batch, at τ:', x0, 304);
    ctx.fillStyle = c.ink2; ctx.fillText(`the pair gets ${Math.round(p.share * 100)}% of it`, x0, 317);
    ctx.font = mono11; ctx.fillStyle = c.ink; ctx.fillText(`loss = −log(share) = ${p.loss.toFixed(2)}`, x0, 332);
    ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.fillText('no label anywhere in this', x0, 350);
  }
  // The similarity matrix of a batch: every view against every other, thumbnails along the edges, the pairs ringed.
  // opt = { thumbs: [px…], sim: N×N cosines, size, tint }; view 2k and 2k+1 are the same nucleus.
  function drawSimilarityMatrix(canvas, opt) {
    const N = opt.thumbs.length, cell = 20, m0 = 26, W = m0 + N * cell + 2, H = W;
    const ctx = fitCanvas(canvas, W, H), c = colors();
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    for (let i = 0; i < N; i++) {
      const img = imageToCanvas(opt.thumbs[i], opt.size, opt.tint);
      ctx.drawImage(img, m0 + i * cell + 1, 2, cell - 2, cell - 2); ctx.drawImage(img, 2, m0 + i * cell + 1, cell - 2, cell - 2);
      if (i % 2 === 0) { ctx.strokeStyle = c.accent; ctx.lineWidth = 1; ctx.strokeRect(m0 + i * cell + 0.5, 0.5, 2 * cell - 1, cell + 1); ctx.strokeRect(0.5, m0 + i * cell + 0.5, cell + 1, 2 * cell - 1); }
    }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = m0 + j * cell, y = m0 + i * cell;
      if (i === j) { ctx.fillStyle = c.surface2; ctx.fillRect(x, y, cell - 1, cell - 1); continue; }
      ctx.fillStyle = diverging(Math.max(-1, Math.min(1, opt.sim[i][j]))); ctx.fillRect(x, y, cell - 1, cell - 1);
      if ((i ^ 1) === j) { ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5; ctx.strokeRect(x + 1, y + 1, cell - 3, cell - 3); }
    }
  }
  // ---- the game the encoder plays: spot the same nucleus
  // One view (the query) against every other view of the batch as a candidate for "the same nucleus", ranked by the
  // encoder's vote: a softmax over the cosines of the codes. opt = { query: px, queryLabel, cands: [{ px, p, partner }]
  // best first, share, loss, size, tint }
  function drawLineup(canvas, opt) {
    const W = 800, per = 20, slot = 35, thumb = 28, barH = 34, rowH = thumb + 4 + barH + 16, rows = Math.ceil(opt.cands.length / per), H = 30 + rows * rowH + 34;
    const ctx = fitCanvas(canvas, W, H), c = colors();
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    const mono9 = `500 9px "IBM Plex Mono", ui-monospace, monospace`, mono11 = `500 11px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.font = mono9; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    // the query, at the left
    const qx = 10, qy = 30;
    ctx.drawImage(imageToCanvas(opt.query, opt.size, opt.tint), qx, qy, 60, 60);
    ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5; ctx.strokeRect(qx - 0.5, qy - 0.5, 61, 61);
    ctx.fillStyle = c.ink; ctx.fillText('this view:', qx, qy + 66);
    const lab = opt.queryLabel.split(' · '); let ty = qy + 77;
    ctx.fillStyle = c.ink3; for (const t of lab) { ctx.fillText(t, qx, ty); ty += 11; }
    ty += 8; ctx.fillStyle = c.ink2; ['which of the', `${opt.cands.length} candidates`, 'is the same', 'nucleus?'].forEach(t => { ctx.fillText(t, qx, ty); ty += 11; });
    ctx.fillStyle = c.irregular; ctx.fillText('answer: orange', qx, ty + 6);
    // the candidates: the encoder's vote, best first
    const x0 = 92;
    ctx.fillStyle = c.ink3; ctx.fillText('the encoder’s vote, best first · each bar is the candidate’s share of 100%', x0, 10);
    let mx = 1e-9; for (const d of opt.cands) mx = Math.max(mx, d.p);
    opt.cands.forEach((d, i) => {
      const r = Math.floor(i / per), k = i % per, x = x0 + k * slot, y = 30 + r * rowH;
      ctx.drawImage(imageToCanvas(d.px, opt.size, opt.tint), x, y, thumb, thumb);
      ctx.strokeStyle = d.partner ? c.irregular : c.lineStrong; ctx.lineWidth = d.partner ? 2.5 : 1; ctx.strokeRect(x - 0.5, y - 0.5, thumb + 1, thumb + 1);
      const bh = Math.max(1, Math.round(d.p / mx * barH)), wrongPick = i === 0 && !d.partner;
      ctx.fillStyle = d.partner ? c.irregular : wrongPick ? c.bad : rgbStr(c.rgb.ink3, 0.45);
      ctx.fillRect(x + 5, y + thumb + 4 + barH - bh, thumb - 10, bh);
      ctx.font = mono9; ctx.fillStyle = d.partner ? c.irregular : wrongPick ? c.bad : c.ink3; ctx.textAlign = 'center';
      ctx.fillText(d.p >= 0.0995 ? `${Math.round(d.p * 100)}%` : `${(d.p * 100).toFixed(1)}%`, x + thumb / 2, y + thumb + 4 + barH + 3);
      ctx.textAlign = 'left';
    });
    // its first pick, right or wrong
    const first = opt.cands[0];
    ctx.font = mono11; ctx.fillStyle = first.partner ? c.good : c.bad; ctx.textAlign = 'right';
    ctx.fillText(first.partner ? '✓ first pick: the partner' : '✗ first pick: another nucleus', W - 8, 8);
    ctx.textAlign = 'left';
    // the loss for this view
    const yb = 30 + rows * rowH + 2, pc = v => (v >= 0.0995 ? `${Math.round(v * 100)}%` : `${(v * 100).toFixed(1)}%`);
    ctx.font = mono11; ctx.fillStyle = c.ink;
    ctx.fillText(`the answer gets ${pc(opt.share)} of the vote → loss = −log(share) = ${opt.loss.toFixed(2)}`, x0, yb);
    ctx.font = mono9; ctx.fillStyle = c.ink3;
    ctx.fillText(`at chance every candidate gets ${(100 / opt.cands.length).toFixed(1)}% (loss ${Math.log(opt.cands.length).toFixed(2)}) · a sure answer gets 100% (loss 0) · every view plays once per batch`, x0, yb + 16);
  }
  // What the code keeps and what it ignores: one nucleus in its 8 orientations from both labs, each view's code as a
  // strip (one scale per row), for the encoder at the start and now, and another nucleus for contrast.
  // opt = { rows: [{ label, note, note2, views: [{ px, code }] }], size, tint }
  function drawViews(canvas, opt) {
    const W = 800, x0 = 150, slot = 40, thumb = 30, strip = 14, rowH = thumb + 3 + strip + 10, H = 26 + opt.rows.length * rowH + 2;
    const ctx = fitCanvas(canvas, W, H), c = colors();
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    const mono9 = `500 9px "IBM Plex Mono", ui-monospace, monospace`, mono11 = `500 11px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.font = mono9; ctx.textBaseline = 'top'; ctx.fillStyle = c.ink3; ctx.textAlign = 'center';
    ctx.fillText('our lab’s scan · 8 orientations', x0 + 4 * slot - 5, 2); ctx.fillText('the other lab’s scan · 8 orientations', x0 + 12 * slot - 5, 2);
    const ori = ['0°', '90°', '180°', '270°', 'mirror', 'm+90°', 'm+180°', 'm+270°'];
    for (let k = 0; k < 16; k++) ctx.fillText(ori[k % 8], x0 + k * slot + thumb / 2, 13);
    opt.rows.forEach((row, r) => {
      const y = 26 + r * rowH;
      ctx.textAlign = 'left'; ctx.font = mono11; ctx.fillStyle = c.ink; ctx.fillText(row.label, 8, y + 4);
      ctx.font = mono9; ctx.fillStyle = c.ink3; if (row.note) ctx.fillText(row.note, 8, y + 21); if (row.note2) ctx.fillText(row.note2, 8, y + 32);
      let mx = 1e-9; for (const v of row.views) for (const q of v.code) mx = Math.max(mx, Math.abs(q));
      row.views.forEach((v, k) => {
        const x = x0 + k * slot;
        ctx.drawImage(imageToCanvas(v.px, opt.size, opt.tint), x, y, thumb, thumb);
        ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, thumb + 1, thumb + 1);
        codeStrip(ctx, c, x, y + thumb + 3, thumb, strip, v.code, mx);
      });
    });
  }
  // ---- a slide of nuclei and its attention
  // The slide as a grid, every nucleus framed by its attention weight (relative to the largest), the weight written
  // under it, a truth dot when revealed (orange atypical, blue bland), the hovered nucleus outlined.
  // opt = { nuclei: [{ px, a, pos }], cols, size, tint, reveal, hover }
  function drawSlide(canvas, opt) {
    const cols = opt.cols || 5, n = opt.nuclei.length, rows = Math.ceil(n / cols), cell = 64, gap = 10, pad = 6, rowH = cell + 14 + gap;
    const W = pad * 2 + cols * cell + (cols - 1) * gap, H = pad * 2 + rows * rowH - gap;
    const ctx = fitCanvas(canvas, W, H), c = colors();
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    let mx = 1e-9; for (const q of opt.nuclei) mx = Math.max(mx, q.a);
    opt.nuclei.forEach((q, i) => {
      const x = pad + (i % cols) * (cell + gap), y = pad + Math.floor(i / cols) * rowH, rel = q.a / mx;
      ctx.drawImage(imageToCanvas(q.px, opt.size, opt.tint), x, y, cell, cell);
      const lw = 1 + 5 * rel; // the attention frame: thicker and stronger with the weight
      ctx.strokeStyle = rgbStr(c.rgb.irregular, 0.15 + 0.85 * rel); ctx.lineWidth = lw; ctx.strokeRect(x + lw / 2, y + lw / 2, cell - lw, cell - lw);
      if (opt.hover === i) { ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.strokeRect(x - 2, y - 2, cell + 4, cell + 4); }
      ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = rel > 0.5 ? c.irregular : c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(`${q.a >= 0.095 ? Math.round(q.a * 100) : (q.a * 100).toFixed(1)}%`, x + cell / 2, y + cell + 2);
      if (opt.reveal) { ctx.fillStyle = q.pos ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(x + 9, y + 9, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = c.surface; ctx.lineWidth = 1.5; ctx.stroke(); }
    });
    if (opt.links) { // who the hovered nucleus listens to: a line to each, thicker with the weight
      const { from, weights } = opt.links, centre = k => [pad + (k % cols) * (cell + gap) + cell / 2, pad + Math.floor(k / cols) * rowH + cell / 2];
      let mw = 1e-9; weights.forEach((w, j) => { if (j !== from) mw = Math.max(mw, w); });
      const [x0, y0] = centre(from);
      weights.forEach((w, j) => { if (j === from || w < 0.04 * mw) return; const rel = w / mw, [x1, y1] = centre(j); ctx.strokeStyle = rgbStr(c.rgb.accent, 0.3 + 0.7 * rel); ctx.lineWidth = 1 + 6 * rel; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); });
      ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(x0, y0, 6, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = c.surface; ctx.lineWidth = 2; ctx.stroke();
    }
    canvas._slideLayout = { cols, cell, gap, pad, rowH, W, n };
  }
  // ---- who looks at whom: the context layer's attention between the nuclei of a slide, rows asking, columns answering
  // opt = { A (n × n, each row adds up to 1), thumbs, size, tint, hover (the asking nucleus outlined), pair ({ i, j }
  // outlined), pos, reveal }
  function drawAttentionMap(canvas, opt) {
    const N = opt.A.length, cell = 16, m0 = 24, W = m0 + N * cell + 2, H = W;
    const ctx = fitCanvas(canvas, W, H), c = colors();
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, H); ctx.imageSmoothingEnabled = false;
    let mx = 1e-9; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) mx = Math.max(mx, opt.A[i][j]);
    for (let i = 0; i < N; i++) {
      const img = imageToCanvas(opt.thumbs[i], opt.size, opt.tint);
      ctx.drawImage(img, m0 + i * cell + 1, 2, cell - 2, cell - 2); ctx.drawImage(img, 2, m0 + i * cell + 1, cell - 2, cell - 2);
      if (opt.reveal) { ctx.fillStyle = opt.pos[i] ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(m0 + i * cell + 4, 5, 2.5, 0, 2 * Math.PI); ctx.fill(); ctx.beginPath(); ctx.arc(5, m0 + i * cell + 4, 2.5, 0, 2 * Math.PI); ctx.fill(); }
    }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = m0 + j * cell, y = m0 + i * cell;
      if (i === j) { ctx.fillStyle = c.surface2; ctx.fillRect(x, y, cell - 1, cell - 1); continue; }
      ctx.fillStyle = sequential(opt.A[i][j] / mx); ctx.fillRect(x, y, cell - 1, cell - 1);
    }
    if (opt.hover != null) { ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5; ctx.strokeRect(m0 - 0.5, m0 + opt.hover * cell - 0.5, N * cell, cell); }
    if (opt.pair && opt.pair.i !== opt.pair.j) { ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.strokeRect(m0 + opt.pair.j * cell - 1, m0 + opt.pair.i * cell - 1, cell + 1, cell + 1); }
    canvas._attLayout = { N, cell, m0, W };
  }
  function hitAttentionMap(canvas, px, py) {
    const L = canvas._attLayout; if (!L) return null;
    const x = px * L.W / canvas.clientWidth, y = py * L.W / canvas.clientWidth, j = Math.floor((x - L.m0) / L.cell), i = Math.floor((y - L.m0) / L.cell);
    return i < 0 || j < 0 || i >= L.N || j >= L.N ? null : { i, j };
  }
  // which nucleus of a drawn slide is under a point (canvas-relative CSS pixels), or null
  function hitSlide(canvas, px, py) {
    const L = canvas._slideLayout; if (!L) return null;
    const x = px * L.W / canvas.clientWidth, y = py * L.W / canvas.clientWidth;
    const col = Math.floor((x - L.pad) / (L.cell + L.gap)), row = Math.floor((y - L.pad) / L.rowH);
    if (col < 0 || col >= L.cols || row < 0) return null;
    const i = row * L.cols + col; if (i >= L.n) return null;
    const cx = L.pad + col * (L.cell + L.gap), cy = L.pad + row * L.rowH;
    return x >= cx && x <= cx + L.cell && y >= cy && y <= cy + L.cell ? i : null;
  }
  // The nuclei ranked by attention: a thumbnail and a bar each (the bar is the nucleus's share of the slide's
  // attention, scaled to the largest), a truth dot when revealed, the hovered one outlined.
  // opt = { items: [{ px, value, pos, index }] best first, size, tint, reveal, hover, title }
  function drawRanked(canvas, opt) {
    const n = opt.items.length, W = 800, x0 = 10, slot = Math.floor((W - 2 * x0) / n), thumb = Math.min(32, slot - 6), barH = 40, H = 24 + thumb + 4 + barH + 16;
    const ctx = fitCanvas(canvas, W, H), c = colors();
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;
    const mono9 = `500 9px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    if (opt.title) ctx.fillText(opt.title, x0, 6);
    let mx = 1e-9; for (const d of opt.items) mx = Math.max(mx, d.value);
    opt.items.forEach((d, k) => {
      const x = x0 + k * slot + Math.floor((slot - thumb) / 2), y = 24, rel = d.value / mx;
      ctx.drawImage(imageToCanvas(d.px, opt.size, opt.tint), x, y, thumb, thumb);
      ctx.strokeStyle = opt.hover === d.index ? c.ink : c.lineStrong; ctx.lineWidth = opt.hover === d.index ? 2 : 1; ctx.strokeRect(x - 0.5, y - 0.5, thumb + 1, thumb + 1);
      if (opt.reveal) { ctx.fillStyle = d.pos ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(x + 5, y + 5, 4, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = c.surface; ctx.lineWidth = 1; ctx.stroke(); }
      const bh = Math.max(1, Math.round(rel * barH));
      ctx.fillStyle = rgbStr(c.rgb.irregular, 0.25 + 0.75 * rel); ctx.fillRect(x + 3, y + thumb + 4 + barH - bh, thumb - 6, bh);
      ctx.font = mono9; ctx.fillStyle = rel > 0.5 ? c.irregular : c.ink3; ctx.textAlign = 'center';
      ctx.fillText(d.value >= 0.095 ? `${Math.round(d.value * 100)}%` : `${(d.value * 100).toFixed(1)}%`, x + thumb / 2, y + thumb + 4 + barH + 3);
      ctx.textAlign = 'left';
    });
  }
  // a slide as a small tile for the trays: its nuclei in a grid of tiny cells
  function renderSlideThumb(canvas, pxs, cols, size, tint) {
    const cell = 12, rows = Math.ceil(pxs.length / cols), W = cols * cell, H = rows * cell;
    if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
    const ctx = canvas.getContext('2d'); ctx.imageSmoothingEnabled = true;
    pxs.forEach((px, i) => ctx.drawImage(imageToCanvas(px, size, tint), (i % cols) * cell, Math.floor(i / cols) * cell, cell, cell));
  }
  // ---- the whole slide model, unrolled on one canvas: every nucleus of the slide through the same scorer, the softmax
  // over the slide, the weighted average and the single layer, drawn like the other network diagrams.
  // m = { nuclei: [{ px, h, s, a, pos }], D, size, tint, reveal, scorer (Net, or null for a plain average), head (Net),
  //       fw (the slide model's forward result: s, a, z, head, fws), shown (the nucleus whose numbers the scorer shows),
  //       hover ({ kind, i, d } or null), walk ({ stage: 'score'|'softmax'|'sum'|'head', k, t } or null),
  //       positiveName, negativeName }
  const UN_W = 800, UN_H = 570, UN_STAGES = ['score', 'softmax', 'sum', 'head'];
  function layoutUnrolled(n, D) {
    const pitch = 24, top = 50, bottom = top + n * pitch, mid = (top + bottom) / 2;
    return { n, D, pitch, top, bottom, mid, rowY: i => top + pitch / 2 + i * pitch,
      xIdx: 24, xThumb: 30, thumb: 20, xCode: 56, codeW: 47, codeH: 18, xRowEnd: 598,
      box: { x: 180, y: mid - 122, w: 150, h: 244 }, xIn: 205, xUnit: 255, xOut: 310, inPitch: 24,
      xScore: 372, rScore: 9, xScoreTxt: 386, band: { x: 424, w: 28 }, xBar: 462, barMax: 90, xPct: 590, xLines: 596,
      sum: { x: 632, r: 14 }, xSummary: 700, rSummary: 11, summaryY: d => mid + (d - (D - 1) / 2) * 24, out: { x: 776, r: 16 } };
  }
  function roundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
  }
  const pctText = a => (a >= 0.095 ? `${Math.round(a * 100)}%` : `${(a * 100).toFixed(1)}%`);
  function drawSlideNetwork(canvas, m) {
    const n = m.nuclei.length, D = m.D, L = layoutUnrolled(n, D), c = colors(), ctx = fitCanvas(canvas, UN_W, UN_H);
    canvas._unrolled = L;
    const capFont = `600 11px "IBM Plex Sans", system-ui, sans-serif`, mono = `500 11px "IBM Plex Mono", ui-monospace, monospace`;
    const mono9 = `500 9px "IBM Plex Mono", ui-monospace, monospace`, mono8 = `500 8px "IBM Plex Mono", ui-monospace, monospace`;
    const att = !!m.scorer, fw = m.fw, w = m.walk, at = w ? UN_STAGES.indexOf(w.stage) : 4; // 4: everything on screen
    const scoredCount = !w || at > 0 ? n : w.k;                       // how many nuclei the scorer has done
    const softmaxT = !w || at > 1 ? 1 : (at === 1 ? w.t : 0);        // the shares growing
    const sumT = !w || at > 2 ? 1 : (at === 2 ? w.t : 0);            // the weighted lines and the summary
    const headT = !w ? 1 : (at === 3 ? w.t : 0);                     // the single layer and the call
    const hov = m.hover && m.hover.kind === 'nucleus' ? m.hover.i : null;
    const cur = w && at === 0 ? Math.max(0, w.k - 1) : null;         // the nucleus being scored right now
    const shown = cur != null ? cur : (hov != null ? hov : m.shown); // whose numbers the scorer displays
    const hl = hov != null ? hov : cur;                              // the row drawn on top
    const hovKind = m.hover ? m.hover.kind : null;
    const A = att ? m.scorer.hidden[0] : 0, unitPitch = A <= 4 ? 40 : 200 / A;
    ctx.clearRect(0, 0, UN_W, UN_H); ctx.fillStyle = c.surface; ctx.fillRect(0, 0, UN_W, UN_H);
    ctx.imageSmoothingEnabled = false;
    let maxA = 1e-9, maxH = 1e-9; for (const q of m.nuclei) { maxA = Math.max(maxA, q.a); for (const v of q.h) maxH = Math.max(maxH, Math.abs(v)); }

    // captions; the stage the walk-through is on lights up
    const caps = [
      { text: `THE SLIDE · ${n} NUCLEI`, x: 12, align: 'left', stage: -1 },
      { text: att ? `SCORER · USED ${n}×` : 'NO SCORER', x: L.box.x + L.box.w / 2, stage: 0 },
      att ? { text: 'SCORES', x: L.xScore + 20, stage: 0 } : null,
      { text: 'SHARES', x: 526, stage: 1 },
      { text: 'WEIGHTED AVERAGE', x: 666, stage: 2 },
      { text: 'CALL', x: L.out.x, stage: 3 },
    ].filter(Boolean);
    ctx.font = capFont; ctx.textBaseline = 'top';
    for (const cap of caps) { ctx.fillStyle = w && at === cap.stage ? c.accent : c.ink3; ctx.textAlign = cap.align || 'center'; ctx.fillText(cap.text, cap.x, 10); }
    ctx.font = mono; ctx.fillStyle = c.ink3; ctx.textAlign = 'center';
    ctx.fillText(att ? `${D} → ${A} tanh → 1 score` : 'every nucleus weighs the same', L.box.x + L.box.w / 2, 26);
    ctx.fillText(att ? 'of the attention · add up to 100%' : `of the attention · 1/${n} each`, 526, 26);
    ctx.textAlign = 'right'; ctx.fillText('single layer', 794, 26);
    if (m.tokenNote) { ctx.textAlign = 'left'; ctx.fillText(m.tokenNote, 12, 26); }

    // row highlights: the hovered nucleus (or the one being scored), and faintly the one whose numbers the scorer shows
    for (let i = 0; i < n; i++) {
      const strong = i === hl, faint = !strong && hl == null && !w && i === shown; if (!strong && !faint) continue;
      ctx.fillStyle = rgbStr(c.rgb.accent, strong ? 0.12 : 0.05); roundedRect(ctx, 6, L.rowY(i) - L.pitch / 2 + 1, L.xRowEnd - 6, L.pitch - 2, 6); ctx.fill();
    }
    // lines into the scorer and out of it: thin, the highlighted nucleus on top in ink
    const line = (x1, y1, x2, y2, strong) => { ctx.strokeStyle = strong ? c.ink : rgbStr(c.rgb.ink3, 0.28); ctx.lineWidth = strong ? 2 : 1; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); };
    const order = []; for (let i = 0; i < n; i++) if (i !== hl) order.push(i); if (hl != null) order.push(hl);
    for (const i of order) {
      if (i >= scoredCount) continue;
      const y = L.rowY(i), strong = i === hl;
      line(L.xCode + L.codeW + 4, y, L.box.x, L.mid, strong);
      if (att) line(L.xOut + 11, L.mid, L.xScore - L.rScore, y, strong); else line(L.box.x + L.box.w, L.mid, L.band.x, y, strong);
    }
    // the rows: index, thumbnail (framed by its share once the softmax has run), truth dot, code
    for (let i = 0; i < n; i++) {
      const y = L.rowY(i), q = m.nuclei[i], rel = q.a / maxA;
      ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(i + 1), L.xIdx, y);
      ctx.drawImage(imageToCanvas(q.px, m.size, m.tint), L.xThumb, y - L.thumb / 2, L.thumb, L.thumb);
      if (softmaxT >= 1 && att) { const lw = 1 + 2 * rel; ctx.strokeStyle = rgbStr(c.rgb.irregular, 0.15 + 0.85 * rel); ctx.lineWidth = lw; ctx.strokeRect(L.xThumb + lw / 2, y - L.thumb / 2 + lw / 2, L.thumb - lw, L.thumb - lw); }
      else { ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(L.xThumb - 0.5, y - L.thumb / 2 - 0.5, L.thumb + 1, L.thumb + 1); }
      if (i === hl) { ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.strokeRect(L.xThumb - 2, y - L.thumb / 2 - 2, L.thumb + 4, L.thumb + 4); }
      if (m.reveal) { ctx.fillStyle = q.pos ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(L.xThumb + 4, y - L.thumb / 2 + 4, 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = c.surface; ctx.lineWidth = 1; ctx.stroke(); }
      codeStrip(ctx, c, L.xCode, y - L.codeH / 2, L.codeW, L.codeH, q.h, maxH);
    }
    // the scorer, once: the box, the little network with the shown nucleus's numbers, or the plain-average note
    ctx.fillStyle = rgbStr(c.rgb.accent, w && at === 0 ? 0.11 : 0.05); roundedRect(ctx, L.box.x, L.box.y, L.box.w, L.box.h, 10); ctx.fill();
    ctx.strokeStyle = rgbStr(c.rgb.accent, 0.35); ctx.lineWidth = 1; if (!att) ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(att ? `same weights for all ${n}` : 'plain average', L.box.x + L.box.w / 2, L.box.y + 7);
    if (att) {
      const sf = fw.fws ? fw.fws[shown] : m.scorer.forward(m.nuclei[shown].h);
      const yIn = d => L.mid + (d - (D - 1) / 2) * L.inPitch, yU = j => L.mid + (j - (A - 1) / 2) * unitPitch;
      const W0 = m.scorer.W[0], Wo = m.scorer.Wo; let m0 = 0, m1 = 0; for (const v of W0) m0 = Math.max(m0, Math.abs(v)); for (const v of Wo) m1 = Math.max(m1, Math.abs(v));
      const n0 = Math.max(EDGE_FLOOR, m0), n1 = Math.max(EDGE_FLOOR, m1), es = [];
      for (let j = 0; j < A; j++) {
        for (let d = 0; d < D; d++) es.push({ w: W0[j * D + d], rel: Math.min(1, Math.abs(W0[j * D + d]) / n0), x1: L.xIn + 7, y1: yIn(d), x2: L.xUnit - 9, y2: yU(j) });
        es.push({ w: Wo[j], rel: Math.min(1, Math.abs(Wo[j]) / n1), x1: L.xUnit + 9, y1: yU(j), x2: L.xOut - 11, y2: L.mid });
      }
      es.sort((a, b) => a.rel - b.rel);
      for (const e of es) { ctx.strokeStyle = rgbStr(e.w < 0 ? c.rgb.regular : c.rgb.irregular, 0.15 + 0.85 * e.rel); ctx.lineWidth = 0.4 + 3 * e.rel; ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke(); }
      for (let d = 0; d < D; d++) circleNode(ctx, L.xIn, yIn(d), 7, diverging(clamp(sf.a[0][d] / 2.5, -1, 1)), null);
      for (let j = 0; j < A; j++) circleNode(ctx, L.xUnit, yU(j), 9, diverging(clamp(sf.a[1][j], -1, 1)), null);
      circleNode(ctx, L.xOut, L.mid, 11, diverging(Math.tanh(sf.z / 2)), fmtSigned(sf.z, 1), mono8, hovKind === 'scorer');
      ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(`showing nucleus ${shown + 1}`, L.box.x + L.box.w / 2, L.box.y + L.box.h - 7);
    } else {
      ctx.font = mono; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('no scorer', L.box.x + L.box.w / 2, L.mid - 8); ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.fillText(`every nucleus 1/${n}`, L.box.x + L.box.w / 2, L.mid + 10);
    }
    // the scores
    if (att) for (let i = 0; i < n; i++) {
      const y = L.rowY(i), q = m.nuclei[i], on = i < scoredCount;
      circleNode(ctx, L.xScore, y, L.rScore, on ? diverging(Math.tanh(q.s / 2)) : c.surface2, null, null, i === hl);
      if (on) { ctx.font = mono9; ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(fmtSigned(q.s, 2), L.xScoreTxt, y); }
    }
    // the softmax band across all the rows: a score only means something next to the other nineteen
    ctx.fillStyle = rgbStr(c.rgb.accent, 0.07 + (w && at === 1 ? 0.14 * Math.sin(Math.PI * w.t) : 0) + (hovKind === 'softmax' ? 0.06 : 0));
    roundedRect(ctx, L.band.x, L.top - 6, L.band.w, L.bottom - L.top + 12, 8); ctx.fill(); ctx.strokeStyle = rgbStr(c.rgb.accent, 0.35); ctx.lineWidth = 1; ctx.stroke();
    ctx.save(); ctx.translate(L.band.x + L.band.w / 2, L.mid); ctx.rotate(-Math.PI / 2); ctx.font = mono; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(att ? `softmax over the ${n} scores` : 'no scorer: equal shares', 0, 0); ctx.restore();
    // the shares
    for (let i = 0; i < n; i++) {
      const y = L.rowY(i), q = m.nuclei[i], rel = q.a / maxA, len = L.barMax * rel * softmaxT;
      if (len > 0) { ctx.fillStyle = rgbStr(c.rgb.irregular, 0.25 + 0.75 * rel); ctx.fillRect(L.xBar, y - 7, Math.max(1, len), 14); }
      if (softmaxT >= 1) { ctx.font = mono9; ctx.fillStyle = rel > 0.5 ? c.irregular : c.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(pctText(q.a), L.xPct, y); }
    }
    // the weighted lines into the sum, the biggest shares first during the walk-through, the highlighted one on top
    const ranked = m.nuclei.map((q, i) => ({ i, rel: q.a / maxA })).sort((a, b) => b.rel - a.rel);
    const visible = new Set(ranked.slice(0, sumT <= 0 ? 0 : (w && at === 2 ? Math.max(1, Math.ceil(w.t * n)) : n)).map(r => r.i));
    const wScale = Math.min(1, maxA * 2.2); // twenty equal shares (a plain average) stay thin; a share near half the slide is thick
    const sumLine = r => { const strong = r.i === hl; ctx.strokeStyle = strong ? c.ink : rgbStr(c.rgb.accent, 0.12 + 0.88 * r.rel); ctx.lineWidth = (0.6 + 5 * r.rel * wScale) * (strong ? 1.3 : 1); ctx.beginPath(); ctx.moveTo(L.xLines, L.rowY(r.i)); ctx.lineTo(L.sum.x - L.sum.r, L.mid); ctx.stroke(); };
    for (const r of ranked.slice().reverse()) if (visible.has(r.i) && r.i !== hl) sumLine(r);
    if (hl != null && visible.has(hl)) sumLine(ranked.find(r => r.i === hl));
    circleNode(ctx, L.sum.x, L.mid, L.sum.r, sumT > 0 ? c.accentSoft : c.surface2, 'Σ', `600 13px "IBM Plex Sans", system-ui, sans-serif`, hovKind === 'sum');
    ctx.font = mono9; { const lbl = 'share × code', lw = ctx.measureText(lbl).width; ctx.fillStyle = c.surface; ctx.fillRect(L.sum.x - lw / 2 - 3, L.mid + L.sum.r + 4, lw + 6, 13); } // legible over the lines
    ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('share × code', L.sum.x, L.mid + L.sum.r + 6);
    // the summary, and the single layer on it
    if (sumT > 0) for (let d = 0; d < D; d++) { ctx.strokeStyle = rgbStr(c.rgb.ink3, 0.35); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L.sum.x + L.sum.r, L.mid); ctx.lineTo(L.xSummary - L.rSummary, L.summaryY(d)); ctx.stroke(); }
    if (headT > 0) {
      let mh = 0; for (const v of m.head.Wo) mh = Math.max(mh, Math.abs(v)); const nh = Math.max(EDGE_FLOOR, mh);
      const hes = Array.from({ length: D }, (_, d) => ({ d, w: m.head.Wo[d], rel: Math.min(1, Math.abs(m.head.Wo[d]) / nh) })).sort((a, b) => a.rel - b.rel);
      for (const e of hes) { ctx.strokeStyle = rgbStr(e.w < 0 ? c.rgb.regular : c.rgb.irregular, (0.15 + 0.85 * e.rel) * headT); ctx.lineWidth = 0.6 + 5.5 * Math.pow(e.rel, 0.9); ctx.beginPath(); ctx.moveTo(L.xSummary + L.rSummary, L.summaryY(e.d)); ctx.lineTo(L.out.x - L.out.r, L.mid); ctx.stroke(); }
    }
    for (let d = 0; d < D; d++) circleNode(ctx, L.xSummary, L.summaryY(d), L.rSummary, sumT >= 1 ? diverging(clamp(fw.z[d] / 2.5, -1, 1)) : c.surface2, sumT >= 1 ? fmtSigned(fw.z[d], 1) : null, mono8, hovKind === 'summary' && m.hover.d === d);
    const p = fw.head.p, call = p >= 0.5;
    circleNode(ctx, L.out.x, L.mid, L.out.r, headT >= 1 ? diverging((p - 0.5) * 2) : c.surface2, headT >= 1 ? p.toFixed(2) : '?', `600 12px "IBM Plex Mono", ui-monospace, monospace`, hovKind === 'output');
    ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText(`P(${m.positiveName}) = ${headT >= 1 ? p.toFixed(2) : '?'}`, 794, UN_H - 36);
    if (headT >= 1) { ctx.font = capFont; ctx.fillStyle = call ? c.irregular : c.regular; ctx.fillText(String(call ? m.positiveName : m.negativeName).toUpperCase(), 794, UN_H - 22); }
    // the footer: what is going on
    let foot;
    if (w) foot = ({ score: `scoring nucleus ${Math.min(n, Math.max(1, w.k))} of ${n} with the same scorer`, softmax: `softmax: the ${n} scores become shares that add up to 100%`, sum: 'adding up the codes, each weighted by its share', head: 'the single layer reads the summary and makes the call' })[w.stage];
    else if (hov != null) foot = `nucleus ${hov + 1}: its code → its score → its share → its part of the summary`;
    else foot = att ? `hover a nucleus to follow it through the model · the scorer shows nucleus ${shown + 1}, the most attention` : 'plain average: nothing can make one nucleus count more than another';
    ctx.font = mono9; ctx.fillStyle = c.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(foot, 12, UN_H - 22);
  }
  // what is under a point of the unrolled diagram (canvas-relative CSS pixels): a nucleus row, the scorer, the softmax,
  // the sum, a summary number or the output, with a tooltip text for everything but the nucleus (the app knows its truth)
  function hitSlideNetwork(canvas, px, py, m) {
    const L = canvas._unrolled; if (!L) return null;
    const x = px * UN_W / canvas.clientWidth, y = py * UN_W / canvas.clientWidth, fw = m.fw, n = m.nuclei.length;
    if (Math.hypot(x - L.out.x, y - L.mid) <= L.out.r + 4) return { kind: 'output', text: `P(${m.positiveName}) = ${fw.head.p.toFixed(3)} · score z = ${fmtSigned(fw.head.z, 2)} · bias ${fmtSigned(m.head.bo, 3)}` };
    for (let d = 0; d < m.D; d++) if (Math.hypot(x - L.xSummary, y - L.summaryY(d)) <= L.rSummary + 4) return { kind: 'summary', d, text: `summary ${d + 1} = ${fmtSigned(fw.z[d], 2)} (Σ share × code ${d + 1}) · weight to the output ${fmtSigned(m.head.Wo[d], 3)}` };
    if (Math.hypot(x - L.sum.x, y - L.mid) <= L.sum.r + 4) return { kind: 'sum', text: `Σ share × code over the ${n} nuclei: the slide’s summary, ${m.D} numbers` };
    if (x >= L.box.x && x <= L.box.x + L.box.w && y >= L.box.y && y <= L.box.y + L.box.h) return { kind: 'scorer', text: m.scorer ? `the scorer: ${m.D} → ${m.scorer.hidden[0]} tanh → 1 score, the same weights for every nucleus · showing nucleus ${m.shown + 1}` : 'plain average: no scorer, every nucleus weighs the same' };
    if (x >= L.band.x - 4 && x <= L.band.x + L.band.w + 4 && y >= L.top - 6 && y <= L.bottom + 6) return { kind: 'softmax', text: m.scorer ? `softmax: share = e^score ÷ Σ e^score over the ${n} nuclei, so the shares add up to 100% and a score only counts relative to the others` : `every nucleus gets 1/${n}` };
    if ((x >= 6 && x <= L.xCode + L.codeW + 4) || (x >= L.xScore - L.rScore - 4 && x <= L.xRowEnd)) { const i = Math.floor((y - L.top) / L.pitch); if (y >= L.top && i >= 0 && i < n) return { kind: 'nucleus', i }; }
    return null;
  }
  // Lines over epochs for several named series. opt = { hist: [{ epoch, … }], keys: [{ key, color }], maxEpoch, pct,
  // baseline (a dashed reference level, with baselineLabel) }
  function drawSeries(svg, opt) {
    const W = 400, H = 170, ml = 40, mr = 16, mt = 10, mb = 22, pw = W - ml - mr, ph = H - mt - mb;
    const hist = opt.hist || [], maxX = Math.max(10, opt.maxEpoch || 0, hist.length ? hist[hist.length - 1].epoch : 0);
    let maxY = opt.pct ? 1 : 1e-9;
    if (!opt.pct) { for (const h of hist) for (const k of opt.keys) if (h[k.key] != null) maxY = Math.max(maxY, h[k.key]); maxY = Math.ceil(maxY * 2) / 2 || 1; }
    const sx = e => ml + e / maxX * pw, sy = v => mt + (1 - v / maxY) * ph;
    let g = '<g class="grid">';
    const ystep = opt.pct ? 0.25 : niceStep(maxY, 4);
    for (let v = 0; v <= maxY + 1e-9; v += ystep) g += `<line x1="${ml}" x2="${W - mr}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"/><text x="${ml - 5}" y="${(sy(v) + 3.5).toFixed(1)}" text-anchor="end">${opt.pct ? Math.round(v * 100) + '%' : v.toFixed(ystep < 1 ? 1 : 0)}</text>`;
    g += '</g>';
    const xstep = niceStep(maxX, 5);
    for (let e = 0; e <= maxX + 1e-9; e += xstep) g += `<text x="${sx(e).toFixed(1)}" y="${H - 6}" text-anchor="middle">${e}</text>`;
    g += `<line class="axis" x1="${ml}" x2="${W - mr}" y1="${mt + ph}" y2="${mt + ph}"/>`;
    const base = opt.baseline != null ? opt.baseline : (opt.pct ? 0.5 : null);
    if (base != null) g += `<line class="chance" x1="${ml}" x2="${W - mr}" y1="${sy(base).toFixed(1)}" y2="${sy(base).toFixed(1)}"/>` + (opt.baselineLabel ? `<text x="${W - mr}" y="${(sy(base) - 4).toFixed(1)}" text-anchor="end">${esc(opt.baselineLabel)}</text>` : '');
    for (const k of opt.keys) {
      const pts = hist.filter(h => h[k.key] != null);
      if (!pts.length) continue;
      g += `<path style="fill:none;stroke:${k.color};stroke-width:2;stroke-linejoin:round;stroke-linecap:round${k.dash ? ';stroke-dasharray:5 4' : ''}" d="${pts.map((h, i) => `${i ? 'L' : 'M'}${sx(h.epoch).toFixed(1)} ${sy(clamp(h[k.key], 0, maxY)).toFixed(1)}`).join(' ')}"/>`;
      const last = pts[pts.length - 1];
      g += `<circle style="fill:${k.color}" r="3" cx="${sx(last.epoch).toFixed(1)}" cy="${sy(clamp(last[k.key], 0, maxY)).toFixed(1)}"/>`;
    }
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = g;
  }

  // ---- shared by the lesson and the test walk-through
  // a banner along the bottom of the diagram naming the step, with an arrow for the passes (dir +1 forward, −1 back)
  function drawBanner(ctx, c, L, text, dir) {
    const y = L.bandLabel || L.poolCaption || L.bands.some(b => b.label) ? NET_H - 34 : NET_H - 16; // above the band label of the pixel layouts, else along the free bottom edge
    ctx.font = `600 11px "IBM Plex Sans", system-ui, sans-serif`; ctx.fillStyle = c.accent; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, 400, y);
    const gap = ctx.measureText(text).width / 2 + 12, x0 = 150, x1 = 650;
    if (dir && 400 - gap > x0 + 20) {
      ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(400 - gap, y); ctx.moveTo(400 + gap, y); ctx.lineTo(x1, y); ctx.stroke();
      const hx = dir > 0 ? x1 : x0, sg = dir > 0 ? -1 : 1;
      ctx.beginPath(); ctx.moveTo(hx, y); ctx.lineTo(hx + sg * 8, y - 4.5); ctx.lineTo(hx + sg * 8, y + 4.5); ctx.closePath(); ctx.fill();
    }
  }
  // which connections make up hop k of a forward sweep: the bands (pixels into the first maps), a dense layer, or the output
  function hopKeyFor(m, nL, k) { return m.mode === 'pixels' ? (k === 0 ? 'band' : (nL && k < nL ? k : (nL ? 'out' : 'sum'))) : (k < nL ? k : 'out'); }
  function drawHopDots(ctx, L, c, key, t) {
    ctx.fillStyle = c.accent;
    const dot = (x, y) => { ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill(); };
    if (key === 'band') { for (const b of L.bands) { const sx = (b.pts[0][0] + b.pts[3][0]) / 2, sy = (b.pts[0][1] + b.pts[3][1]) / 2, ex = (b.pts[1][0] + b.pts[2][0]) / 2, ey = (b.pts[1][1] + b.pts[2][1]) / 2; dot(sx + (ex - sx) * t, sy + (ey - sy) * t); } return; }
    for (const e of L.edges) { if (e.layer !== key) continue; dot(e.x1 + (e.x2 - e.x1) * t, e.y1 + (e.y2 - e.y1) * t); }
  }
  // the value at the start of a connection for the case on screen: an input, a hidden activation, or the top layer
  function sourceValue(m, e) {
    const fw = m.fw; if (e.fi == null) return null;
    if (e.layer === 'out') return fw ? fw.a[fw.a.length - 1][e.fi] : null;
    if (e.layer === 0) return m.x ? m.x[e.fi] : null;
    return fw ? fw.a[e.layer][e.fi] : null;
  }
  // ---- the forward sweep plan: how a forward pass is paced, shared by the lesson and the test walk-through.
  // One segment per hop: 'image' is the choreographed pixel hop (overlay, product, sum, ReLU), 'sum' the Σ → z edge of
  // the single-layer pixel network, a number a dense layer, 'out' the output; 'final' is a pause with everything shown.
  function sweepPlan(net, mode, opts) {
    const nL = net.hidden.length, segs = [];
    if (mode === 'pixels' && !net.conv) {
      const rows = nL ? net.hidden[0] : 1;
      if (!(opts && opts.prep === false)) segs.push({ key: 'prep', ms: 4000 }); // this nucleus − the mean nucleus = what the network sees
      segs.push({ key: 'image', ms: 12000 + (rows > 1 ? 6000 : 0) }); // the first unit slowly (12 s), the rest together (6 s)
      if (!nL) segs.push({ key: 'sum', ms: 1200 });
      else for (let k = 1; k <= nL; k++) segs.push({ key: k < nL ? k : 'out', ms: 1200 });
      segs.push({ key: 'final', ms: 800 });
    } else {
      const hops = nL + 1, ms = 2600 / (hops + 1);
      for (let k = 0; k < hops; k++) segs.push({ key: k < nL ? k : 'out', ms });
      segs.push({ key: 'final', ms });
    }
    const offset = segs[0].key === 'prep' ? 1 : 0; // leading segments that are not hops of the network
    return { segs, total: segs.reduce((q, g) => q + g.ms, 0), hops: segs.length - 1 - offset, offset };
  }
  // The forward pass of a convolutional network, as phases: the preprocessing, filter 1 scanning slowly then the others,
  // map 1 pooled block by block then the others, the pooled maps × each unit's weight map, the dense hops, a final pause.
  function convPlan(net, withPrep) {
    const nL = net.hidden.length, multi = net.conv.K > 1, units = nL ? net.hidden[0] : 1;
    const T = []; if (withPrep) T.push(['prep', 4000]);
    T.push(['scan1', 12000], ['scan', multi ? 7000 : 0], ['pool1', 4000], ['pool', multi ? 2500 : 0], ['units', 12000 + (units > 1 ? 6000 : 0)]);
    const hopKeys = []; for (let l = 1; l < nL; l++) hopKeys.push(l); hopKeys.push(nL ? 'out' : 'sum');
    for (const key of hopKeys) T.push(['hop', 1200, key]);
    T.push(['final', 800]);
    return { T, hopKeys, hops: 1 + hopKeys.length, total: T.reduce((a, p) => a + p[1], 0) };
  }
  // the segment of a plan at a fraction of its total, with the progress within it
  function convAt(plan, frac) {
    const t = Math.max(0, Math.min(0.9999, frac)) * plan.total; let acc = 0;
    for (const seg of plan.T) { if (seg[1] > 0 && t < acc + seg[1]) return { seg, frac: (t - acc) / seg[1] }; acc += seg[1]; }
    const last = plan.T[plan.T.length - 1]; return { seg: last, frac: 1 };
  }
  // the drawing state of one moment of that pass (m.anim); banners name what is happening
  function convAnim(net, plan, seg, frac, opts) {
    const nL = net.hidden.length, total = net.co * net.co, cells = net.po * net.po, [phase, , arg] = seg, again = opts && opts.again;
    const anim = { phase, hops: plan.hops, reveal: 0, banner: null, dir: 0 };
    if (phase === 'prep') anim.t = frac;
    else if (phase === 'scan1' || phase === 'scan') { anim.pos = Math.min(total, Math.floor(frac * total)); anim.showFilter = phase === 'scan1' ? 0 : 1; }
    else if (phase === 'pool1' || phase === 'pool') anim.posP = Math.min(cells, Math.floor(frac * cells));
    else if (phase === 'units') { anim.t = frac; anim.banner = nL ? 'dense layer: pooled maps × each unit’s weights, summed, through ReLU' : 'output: pooled maps × weights, summed, plus the bias'; anim.dir = 1; }
    else if (phase === 'hop') { const i = plan.hopKeys.indexOf(arg); anim.key = arg; anim.t = frac; anim.wipes = plan.hopKeys.slice(0, i + 1); anim.reveal = 1 + i; anim.banner = arg === 'sum' ? 'output: z through the sigmoid' : `${arg === 'out' ? 'output' : 'hidden layer ' + (arg + 1)}: each connection carries weight × value`; anim.dir = 1; }
    else { anim.wipes = plan.hopKeys; anim.reveal = plan.hops; }
    if (phase === 'final') anim.banner = again ? 'forward pass again, with the new weights' : (opts && opts.finalBanner) || 'forward pass done: the weights are frozen, nothing is learned here';
    if (again && anim.banner && phase !== 'final') anim.banner += ' · with the new weights';
    return anim;
  }
  // where a sweep is at a fraction of its plan: the segment, the progress within it, and the hops that have arrived
  function sweepState(plan, frac) {
    const t = Math.max(0, Math.min(0.9999, frac)) * plan.total;
    let acc = 0;
    for (let i = 0; i < plan.segs.length; i++) { const g = plan.segs[i]; if (t < acc + g.ms) return { index: i, t: (t - acc) / g.ms, reveal: Math.max(0, Math.min(plan.hops, i - plan.offset)), key: g.key }; acc += g.ms; }
    return { index: plan.segs.length - 1, t: 1, reveal: plan.hops, key: 'final' };
  }
  // The forward sweep. On dense connections a wipe grows from source to target with thickness = |weight × value at the
  // source| and the colour of its sign, so the sum arriving at each node can be read off the diagram. The pixel hop is
  // choreographed: the nucleus is laid over each weight map, the product map forms and is set aside, a scan line sums it
  // while the unit counts, then the bias and the ReLU (or the sigmoid at the output) finish the unit.
  function drawForwardSweep(ctx, L, c, m, frac, plan) {
    plan = plan || sweepPlan(m.net, m.mode);
    const st = sweepState(plan, frac);
    for (let i = 0; i <= Math.min(st.index, plan.segs.length - 2); i++) {
      const key = plan.segs[i].key, cur = i === st.index, t = cur ? st.t : 1;
      if (key === 'prep') { if (cur) drawPrep(ctx, L, c, m, t); else drawPrepDone(ctx, L, c, m); }
      else if (key === 'image') { if (cur) drawImageHop(ctx, L, c, m, t); } // once done, the diagram itself shows the products and the units
      else if (key === 'sum') drawSumEdge(ctx, L, c, m, t);
      else drawHopWipe(ctx, L, c, m, key, t);
    }
  }
  // The preprocessing shown before the pixel hop, in the free space above the input node: this nucleus − the mean
  // training nucleus = the difference the network sees (orange: more ink than average, blue: less). The difference then
  // drops into the input node, and it is the copy that is laid over the weight maps. The row and the difference stay on
  // screen for the rest of the walk-through (drawPrepDone), so every later step can be read against what the network sees.
  function prepRowAt(L, img, m) {
    const size = m.net.conv ? 36 : 44, gap = m.net.conv ? 36 : 30, x = 16, legendPx = m.net.conv ? 9 : 10; // the convolution layout has less room above its image
    return { size, gap, at: { x, y: m.net.conv ? 52 : img.y - img.h / 2 - 96 }, xs: [x, x + size + gap, x + 2 * (size + gap)], legendPx };
  }
  function drawPrepRow(ctx, L, c, m, t) { // the three tiles, fading in one after another; t = 1 shows the finished row
    const img = L.nodes.find(n => n.kind === 'image'); if (!img || !m.x || !m.specimen || !m.inputMean) return null;
    const S = m.size, { size, gap, at, xs, legendPx } = prepRowAt(L, img, m), lh = legendPx + 3;
    const pxMean = new Uint8ClampedArray(S * S); for (let i = 0; i < S * S; i++) pxMean[i] = Math.round(255 * (1 - m.inputMean[i]));
    const diff = tileCanvas('xin', m.x, 0, S, S).canvas;
    const fade = (a, b) => Math.max(0, Math.min(1, (t - a) / (b - a)));
    const label = (text, x) => { ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(text, x + size / 2, at.y - 4); };
    const glyph = (text, x) => { ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, x, at.y + size / 2); };
    const frame = x => { ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, at.y - 0.5, size + 1, size + 1); };
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = fade(0, 0.12);
    ctx.drawImage(imageToCanvas(m.inputPx || m.specimen.px, S, m.tint), xs[0], at.y, size, size); frame(xs[0]); label(m.inputPx ? 'normalised' : 'this nucleus', xs[0]);
    ctx.globalAlpha = fade(0.2, 0.36);
    glyph('−', xs[0] + size + gap / 2);
    ctx.drawImage(imageToCanvas(pxMean, S, m.tint), xs[1], at.y, size, size); frame(xs[1]); label('mean nucleus', xs[1]);
    ctx.globalAlpha = fade(0.44, 0.6);
    glyph('=', xs[1] + size + gap / 2);
    ctx.drawImage(diff, xs[2], at.y, size, size); frame(xs[2]); label('difference', xs[2]);
    ctx.font = `500 ${legendPx}px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('what the network sees', at.x, at.y + size + 6);
    ctx.fillStyle = c.ink3; ctx.fillText('orange: more ink than average', at.x, at.y + size + 6 + lh); ctx.fillText('blue: less ink than average', at.x, at.y + size + 6 + 2 * lh);
    ctx.globalAlpha = 1;
    return { img, diff, size, at, xs };
  }
  function drawPrep(ctx, L, c, m, t) {
    const row = drawPrepRow(ctx, L, c, m, t); if (!row) return;
    const { img, diff, size, at, xs } = row;
    const ease = u => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
    if (t > 0.72) { // the difference drops into the input node and takes the place of the raw nucleus
      const e = ease((t - 0.72) / 0.28);
      const cx = xs[2] + size / 2 + (img.x - xs[2] - size / 2) * e, cy = at.y + size / 2 + (img.y - at.y - size / 2) * e, sz = size + (img.w - size) * e;
      ctx.drawImage(diff, cx - sz / 2, cy - sz / 2, sz, sz); ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(cx - sz / 2, cy - sz / 2, sz, sz);
    }
  }
  function drawPrepDone(ctx, L, c, m) { // after the preprocessing step: the row stays up and the input node shows what the network sees
    const img = L.nodes.find(n => n.kind === 'image'); if (!img || !m.x) return;
    drawPrepRow(ctx, L, c, m, 1);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tileCanvas('xin', m.x, 0, m.size, m.size).canvas, img.x - img.w / 2, img.y - img.h / 2, img.w, img.h);
    ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(img.x - img.w / 2, img.y - img.h / 2, img.w, img.h);
  }
  const IMAGE_STAGES = [['overlay', 0.175], ['wipe', 0.4], ['aside', 0.525], ['sum', 0.825], ['relu', 1]];
  function imageStage(u) { let lo = 0; for (const [name, end] of IMAGE_STAGES) { if (u < end) return { name, v: (u - lo) / (end - lo) }; lo = end; } return { name: 'done', v: 1 }; }
  function drawImageHop(ctx, L, c, m, t) {
    const net = m.net, nL = net.hidden.length, fw = m.fw, inp = firstInput(m);
    if (!inp || !fw || !m.specimen) return;
    const x = inp.vec, D = inp.D, S = inp.w, SH = inp.h; // the input map: the pixels, or the pooled maps stacked into one
    const img = L.nodes.find(n => n.kind === 'image');
    const pooledNodes = net.conv ? L.nodes.filter(q => q.kind === 'pooled') : null, comp = net.conv ? compositeDims(net) : null;
    let fmapMax = 1e-9; if (net.conv) for (let i = 0; i < fw.conv.act.length; i++) fmapMax = Math.max(fmapMax, fw.conv.act[i]);
    const inTile = () => (net.conv ? tileCanvas('pooledin', x, 0, S, SH, { mode: 'sequential', max: fmapMax }) : tileCanvas('xin', x, 0, S, S)).canvas;
    const rows = nL ? L.unitColumns[0] : [L.nodes.find(n => n.kind === 'map')];
    const n = rows.length;
    const T1 = 12000, T2 = n > 1 ? 6000 : 0, tm = t * (T1 + T2);
    const monoS = `500 10px "IBM Plex Mono", ui-monospace, monospace`;
    const badgeFont = `500 10px "IBM Plex Mono", ui-monospace, monospace`;
    const fmtA = v => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));
    rows.forEach((w, j) => {
      if (!w) return;
      const st = j === 0 ? (tm < T1 ? imageStage(tm / T1) : { name: 'done', v: 1 }) : (tm < T1 ? { name: 'wait', v: 0 } : imageStage((tm - T1) / T2));
      if (st.name === 'wait') return;
      const pn = nL ? L.nodes.find(q => q.kind === 'uprod' && q.j === j) : L.nodes.find(q => q.kind === 'product');
      const Wj = firstWeights(m, nL ? j : null), Wrow = Wj.arr, off = Wj.off;
      const prod = new Float64Array(D); let pos = 0, neg = 0;
      for (let i = 0; i < D; i++) { const q = Wrow[off + i] * x[i]; prod[i] = q; if (q > 0) pos += q; else neg += q; }
      const bias = nL ? net.b[0][j] : net.bo, z = pos + neg + bias, act = nL ? fw.a[1][j] : null;
      const size = w.size, sizeH = w.h || w.size, badgeX = L.badgeX;
      const tile = () => tileCanvas('sweepprod' + j, prod, 0, S, SH).canvas;
      const frameTile = (x0, y0, tw, th) => { if (net.conv) drawCompositeDividers(ctx, c, net, x0, y0, tw, th); ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, tw, th); };
      ctx.imageSmoothingEnabled = false;
      if (st.name === 'overlay') {
        const e = st.v * st.v * (3 - 2 * st.v);
        if (net.conv) { // the pooled maps gather over the weight map, each into its own slot of the stack, half transparent
          pooledNodes.forEach((pq, k) => {
            const sw = size / comp.cols, sh = sizeH / comp.rowsK;
            const x1 = w.x - size / 2 + (k % comp.cols) * sw + sw / 2, y1 = w.y - sizeH / 2 + Math.floor(k / comp.cols) * sh + sh / 2;
            const cx = pq.x + (x1 - pq.x) * e, cy = pq.y + (y1 - pq.y) * e, cw = pq.size + (sw - pq.size) * e, chh = pq.size + (sh - pq.size) * e;
            ctx.globalAlpha = 1 - 0.5 * e;
            ctx.drawImage(tileCanvas('gather' + k, fw.conv.v, k * net.po * net.po, net.po, net.po, { mode: 'sequential', max: fmapMax }).canvas, cx - cw / 2, cy - chh / 2, cw, chh);
            ctx.globalAlpha = 1; ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(cx - cw / 2, cy - chh / 2, cw, chh);
          });
          return;
        }
        // the nucleus slides along the band and settles over the weight map, half transparent
        const cx = img.x + (w.x - img.x) * e, cy = img.y + (w.y - img.y) * e, sz = img.w + (size - img.w) * e;
        ctx.globalAlpha = 1 - 0.5 * e;
        ctx.drawImage(inTile(), cx - sz / 2, cy - sz / 2, sz, sz);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(cx - sz / 2, cy - sz / 2, sz, sz);
        return;
      }
      if (st.name === 'wipe') { // the product map wipes in over the overlay, cell = input × weight
        ctx.globalAlpha = 0.5 * (1 - st.v);
        ctx.drawImage(inTile(), w.x - size / 2, w.y - sizeH / 2, size, sizeH);
        ctx.globalAlpha = 1;
        ctx.save(); ctx.beginPath(); ctx.rect(w.x - size / 2, w.y - sizeH / 2, size, sizeH * st.v); ctx.clip();
        ctx.drawImage(tile(), w.x - size / 2, w.y - sizeH / 2, size, sizeH);
        ctx.restore();
        frameTile(w.x - size / 2, w.y - sizeH / 2, size, sizeH);
        return;
      }
      // from here on the product map exists; it slides to its slot, then is summed
      const e = st.name === 'aside' ? st.v * st.v * (3 - 2 * st.v) : 1;
      const px = w.x + (pn.x - w.x) * e, py = w.y + (pn.y - w.y) * e;
      ctx.drawImage(tile(), px - size / 2, py - sizeH / 2, size, sizeH);
      frameTile(px - size / 2, py - sizeH / 2, size, sizeH);
      if (st.name === 'aside') return;
      // the sum: a scan line sweeps the map, the unit counts, bars show the positive and negative parts
      const rowsDone = st.name === 'sum' ? Math.floor(st.v * SH) : SH;
      let sum = 0, posNow = 0, negNow = 0;
      for (let i = 0; i < rowsDone * S; i++) { const q = prod[i]; sum += q; if (q > 0) posNow += q; else negNow += q; }
      if (st.name === 'sum') { const sy = pn.y - sizeH / 2 + st.v * sizeH; ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(pn.x - size / 2, sy); ctx.lineTo(pn.x + size / 2, sy); ctx.stroke(); }
      const half = size / 2 - 2, maxAbs = Math.max(pos, -neg, 1e-9), by = pn.y + sizeH / 2 + 4;
      ctx.fillStyle = c.irregular; ctx.fillRect(pn.x, by, posNow / maxAbs * half, 4);
      ctx.fillStyle = c.regular; ctx.fillRect(pn.x + negNow / maxAbs * half, by, -negNow / maxAbs * half, 4);
      ctx.fillStyle = c.ink; ctx.fillRect(pn.x + (posNow + negNow) / maxAbs * half - 1, by - 2, 2, 8);
      if (nL) {
        const showAct = st.name === 'done' || (st.name === 'relu' && st.v >= 0.5);
        const label = st.name === 'sum' ? fmtA(sum) : st.name === 'relu' && st.v < 0.5 ? fmtA(z) : fmtA(act);
        circleNode(ctx, badgeX, pn.y, 11, showAct ? unitFill(act, fw.a[1], ACT_SIGNED(m.activation)) : c.surface2, label, badgeFont);
        const roomy = size >= 56 && (rows.length < 2 || Math.abs(rows[1].y - rows[0].y) >= 56);
        if (roomy) {
          ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3;
          const above = !!net.conv, cx = above ? badgeX + 8 : badgeX - 8, cy = above ? pn.y - 15 : pn.y + 15;
          ctx.textAlign = above ? 'center' : 'left'; ctx.textBaseline = above ? 'bottom' : 'top';
          if (st.name === 'sum') ctx.fillText('Σ so far', cx, cy);
          else if (st.name === 'relu' && st.v < 0.5) ctx.fillText(`Σ + b ${fmtSigned(bias, 2)}`, cx, cy);
          else if (st.name === 'relu' || st.name === 'done') drawReluGlyph(ctx, c, badgeX, pn.y - 24, z);
        }
      } else {
        ctx.font = monoS; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const line = st.name === 'sum' ? `Σ so far ${fmtSigned(sum, 2)}` : `Σ = ${fmtSigned(pos + neg, 2)}  + bias ${fmtSigned(bias, 2)}  = z ${fmtSigned(z, 2)}`;
        ctx.fillText(line, pn.x, by + 8);
      }
    });
  }
  // the Σ → z edge of the single-layer pixel network: a wipe in the colour of z, its value, and the sigmoid with the point plotted
  function drawSumEdge(ctx, L, c, m, t) {
    const e = L.edges.find(q => q.layer === 'sum'); if (!e || !m.fw) return;
    const z = m.fw.z;
    ctx.lineCap = 'round'; ctx.strokeStyle = rgbStr(z >= 0 ? c.rgb.irregular : c.rgb.regular, 0.9); ctx.lineWidth = 2 + 8 * Math.min(1, Math.abs(z) / 4);
    ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x1 + (e.x2 - e.x1) * t, e.y1 + (e.y2 - e.y1) * t); ctx.stroke(); ctx.lineCap = 'butt';
    ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`z = ${fmtSigned(z, 2)}`, (e.x1 + e.x2) / 2, e.y1 - 20);
    if (t > 0.4) { drawSigmoidGlyph(ctx, c, L.output.x - 62, L.output.y - 58, z); ctx.fillStyle = c.ink3; ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillText('sigmoid', L.output.x - 62, L.output.y - 68); }
  }
  // small activation glyphs with the point plotted: the ReLU hinge and the sigmoid S
  function drawReluGlyph(ctx, c, cx, cy, z) {
    const w = 28, h = 14, x0 = cx - w / 2, y0 = cy + h / 2;
    ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + w, y0); ctx.stroke();
    ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + w / 2, y0); ctx.lineTo(x0 + w, y0 - h); ctx.stroke();
    const zc = Math.max(-1, Math.min(1, z / 3)), px = x0 + w / 2 + zc * w / 2, py = y0 - Math.max(0, zc) * h;
    ctx.fillStyle = z > 0 ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
  }
  function drawSigmoidGlyph(ctx, c, cx, cy, z) {
    const w = 28, h = 14, x0 = cx - w / 2, y0 = cy + h / 2;
    ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + w, y0); ctx.stroke();
    ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i <= w; i++) { const zz = (i / w - 0.5) * 8, yy = y0 - h / (1 + Math.exp(-zz)); i ? ctx.lineTo(x0 + i, yy) : ctx.moveTo(x0 + i, yy); }
    ctx.stroke();
    const zc = Math.max(-4, Math.min(4, z)), px = x0 + (zc / 8 + 0.5) * w, py = y0 - h / (1 + Math.exp(-zc));
    ctx.fillStyle = z > 0 ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
  }
  function drawHopWipe(ctx, L, c, m, key, t) {
    const items = [];
    let max = 1e-9;
    for (const e of L.edges) { if (e.layer !== key) continue; const v = sourceValue(m, e); if (v == null) continue; const p = e.w * v; items.push({ e, p }); max = Math.max(max, Math.abs(p)); }
    ctx.lineCap = 'round';
    for (const { e, p } of items) {
      ctx.strokeStyle = rgbStr(p >= 0 ? c.rgb.irregular : c.rgb.regular, 0.9);
      ctx.lineWidth = 1.5 + 9 * Math.abs(p) / max;
      ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x1 + (e.x2 - e.x1) * t, e.y1 + (e.y2 - e.y1) * t); ctx.stroke();
    }
    ctx.lineCap = 'butt';
  }
  function drawTruthLabel(ctx, c, out, y, name, mark) {
    ctx.font = `600 11px "IBM Plex Sans", system-ui, sans-serif`; ctx.fillStyle = y ? c.irregular : c.regular; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`truth: ${name}${mark ? ' ' + mark : ''}`, out.x, out.y + out.r + 42);
  }
  // The test walk-through: one held-out case through the frozen weights, then the call at the threshold, then the truth.
  // m.sweep = { phase: forward | call | reveal, frac, hops, reveal, p, called, threshold, truthName, y, correct }
  function drawTestSweep(ctx, L, m) {
    const sw = m.sweep, c = colors(), nL = m.net.hidden.length;
    drawForwardSweep(ctx, L, c, m, sw.phase === 'forward' ? sw.frac : 1);
    if (sw.phase === 'forward') {
      drawBanner(ctx, c, L, 'forward pass: each connection carries weight × value', 1);
    } else if (sw.phase === 'call') {
      drawBanner(ctx, c, L, `P(${m.positiveName}) = ${sw.p.toFixed(2)} ${sw.p >= sw.threshold ? '≥' : '<'} ${sw.threshold.toFixed(2)}, so the call is ${sw.called}`, 0);
    } else {
      drawTruthLabel(ctx, c, L.output, sw.y, sw.truthName, sw.correct ? '✓' : '✗');
      drawBanner(ctx, c, L, `truth: ${sw.truthName} · ${sw.correct ? 'correct' : 'wrong'}`, 0);
    }
  }

  // The lesson walk-through, one training case in six steps (five without a hidden layer):
  //   forward → loss → blame (backward pass) → gradient → update → check
  // m.lesson = { phase, frac (0..1 within the phase), hops, reveal, error, loss, y, truthName, pBefore, pAfter,
  //   delta[l][j] (blame of each hidden unit), gW, gWo, lr, fwBefore }
  function drawLesson(ctx, L, m) {
    drawLessonSteps(ctx, L, m);
    const ph = m.lesson.phase; // the preprocessing row stays up after the forward pass; here it goes on top of the step's overlays
    if (m.mode === 'pixels' && !m.net.conv && (ph === 'blame' || ph === 'gradient' || ph === 'update')) drawPrepDone(ctx, L, colors(), m);
  }
  function drawLessonSteps(ctx, L, m) {
    const les = m.lesson, c = colors(), net = m.net, out = L.output;
    const monoS = `500 10px "IBM Plex Mono", ui-monospace, monospace`;
    const ease = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    const up = les.error < 0;                       // the score must go up (orange) or come down (blue)
    const pushCol = up ? c.irregular : c.regular, pushRgb = up ? c.rgb.irregular : c.rgb.regular;
    const truthCol = les.y ? c.irregular : c.regular;
    const ph = les.phase, conv = !!net.conv, nL = net.hidden.length;
    const plan = conv ? null : sweepPlan(net, m.mode, { prep: ph !== 'check' }), cplan = conv ? convPlan(net, ph !== 'check') : null;
    const hops = conv ? cplan.hops : plan.hops;
    const prepped = m.mode === 'pixels' && !conv && ph === 'check'; // the check replays the sweep with the preprocessing already done
    const banner = (text, dir) => drawBanner(ctx, c, L, text, dir);
    ctx.font = `600 11px "IBM Plex Sans", system-ui, sans-serif`; ctx.fillStyle = truthCol; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`${les.truthWord || 'truth'}: ${les.truthName}`, out.x, out.y + out.r + 42);

    // 1 · forward pass (and 6 · check): values flow hop by hop (a convolutional network's pass is drawn from m.anim)
    const s0 = conv ? 1 - 800 / cplan.total : 1 - plan.segs[plan.segs.length - 1].ms / plan.total;
    if (prepped) drawPrepDone(ctx, L, c, m);
    if (ph === 'forward' || ph === 'check') {
      if (!conv) { drawForwardSweep(ctx, L, c, m, les.frac, plan); banner(ph === 'forward' ? 'forward pass: each connection carries weight × value' : 'forward pass again, with the new weights', 1); }
      if (ph === 'forward') return;
    } else if (ph === 'loss' && !conv) drawForwardSweep(ctx, L, c, m, 1, plan);
    // the backward steps of a convolutional network run in sub-stages: the dense part first, then the pooling and the filters
    const q = les.frac, denseBlame = conv && ph === 'blame' ? Math.min(1, q / CONV_STAGES.blameDense) : 1, denseGrad = conv && ph === 'gradient' ? Math.min(1, q / CONV_STAGES.gradDense) : 1;
    const convBlameBanner = () => q < CONV_STAGES.blameDense
      ? (nL ? 'backward pass: each connection carries error × weight back' : 'backward pass: the error comes back along Σ → z to the weight map')
      : q < CONV_STAGES.blameDots ? `backward pass: each pooled cell’s blame = ${nL ? 'Σ unit blame × its weight' : 'error × its weight'}` : 'backward pass: through the pooling, onto the position that won each block';

    // 2 · loss: the call against the truth on a 0..1 scale beside the output; the gap is the error
    const bx = out.x + out.r + 20, top = out.y - 46, bot = out.y + 46, yOf = v => bot - v * (bot - top);
    ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx, top); ctx.lineTo(bx, bot); ctx.stroke();
    ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('1', bx + 6, top); ctx.fillText('0', bx + 6, bot);
    const ty = yOf(les.y);
    let pv = les.pBefore;
    if (ph === 'check' && les.pAfter != null) pv = les.frac <= s0 ? les.pBefore : les.pBefore + (les.pAfter - les.pBefore) * ease(Math.min(1, (les.frac - s0) / (1 - s0)));
    ctx.strokeStyle = rgbStr(pushRgb, 0.55); ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(bx, yOf(pv)); ctx.lineTo(bx, ty); ctx.stroke();
    ctx.strokeStyle = truthCol; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(bx - 7, ty); ctx.lineTo(bx + 7, ty); ctx.stroke();
    ctx.beginPath(); ctx.arc(bx, yOf(pv), 4.5, 0, Math.PI * 2); ctx.fillStyle = c.ink; ctx.fill();
    ctx.font = monoS; ctx.fillStyle = pushCol; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`error = p − y = ${fmtSigned(les.error, 2)} ${up ? '↑' : '↓'}`, out.x, out.y + out.r + 58);
    ctx.fillStyle = c.ink3; ctx.fillText(`loss ${les.loss.toFixed(2)}`, out.x, out.y + out.r + 74);
    drawLossGlyph(ctx, c, bx, top - 30, les.y, pv); // the loss against the call, with the point plotted: its slope there is the error
    if (ph === 'check' && les.pAfter != null && les.reveal >= hops) { const dp = les.pBefore.toFixed(2) === les.pAfter.toFixed(2) ? 3 : 2; ctx.fillStyle = c.ink2; ctx.fillText(`${les.pBefore.toFixed(dp)} → ${les.pAfter.toFixed(dp)}`, out.x, out.y + out.r + 90); }
    if (ph === 'loss') { banner('loss: how wrong was the call?', 0); return; }
    if (ph === 'check') return; // the replay shows the forward pass alone: the blame and the gradients are behind it

    // 3 · backward pass: the error flows back along each connection as error × weight, a wipe from the output end with
    // thickness = its size (blue: the unit should come down, orange: go up), one layer at a time; the activation then
    // gates what arrived (ReLU passes it only if the unit was on) and every unit shows its blame with the arithmetic
    if (nL) {
      const prog = ph === 'blame' ? denseBlame * nL : nL, fwB = les.fwBefore;
      for (let k = 0; k < nL; k++) {
        const l = nL - 1 - k, t = Math.min(1, prog - k);
        if (t <= 0) continue;
        const key = l === nL - 1 ? 'out' : l + 1;  // the connections that leave layer l
        const flowOf = e => (key === 'out' ? les.error : les.delta[l + 1][e.ti]) * e.w;
        if (ph === 'blame') {
          let max = 1e-9; for (const e of L.edges) if (e.layer === key) max = Math.max(max, Math.abs(flowOf(e)));
          const wt = Math.min(1, t / 0.7);
          ctx.lineCap = 'round';
          for (const e of L.edges) {
            if (e.layer !== key) continue;
            const f = flowOf(e);
            ctx.strokeStyle = rgbStr(f > 0 ? c.rgb.regular : c.rgb.irregular, 0.9); ctx.lineWidth = 1.5 + 9 * Math.abs(f) / max;
            ctx.beginPath(); ctx.moveTo(e.x2, e.y2); ctx.lineTo(e.x2 + (e.x1 - e.x2) * wt, e.y2 + (e.y1 - e.y2) * wt); ctx.stroke();
          }
          ctx.lineCap = 'butt';
          if (t < 0.7) continue;
        }
        L.unitColumns[l].forEach((u, j) => {
          const d = les.delta[l][j];
          if (ph !== 'blame' && u.kind === 'square' && out.x - out.r - L.badgeX < 130) return; // on short connections the gradient labels take the pills' room
          // what arrived (error × weight, or Σ blame × weight from the layer above) and the gate the activation applied
          let arrived = 0;
          if (key === 'out') arrived = les.error * net.Wo[j];
          else { const n1 = net.sizes[l + 1]; L.unitColumns[l + 1].forEach((_, q) => { arrived += les.delta[l + 1][q] * net.W[l + 1][q * n1 + j]; }); }
          const on = fwB.pre[l][j] > 0, gate = arrived !== 0 ? d / arrived : 0;
          // the full arithmetic where there is room: not on cramped rows, and not on a last layer that sits next to the
          // output's text when there is a layer before it (the tooltip carries the arithmetic there)
          // a next column hard by the badges (the convolution's two-layer layout) takes the pills' room: the short form goes under the product map
          const crowded = u.kind === 'square' && nL > 1 && L.unitColumns[1].length && L.unitColumns[1][0].x - L.badgeX < 80;
          const roomy = (u.kind !== 'square' || u.captioned) && !(key === 'out' && nL > 1) && !crowded;
          const lead = key === 'out' ? `${fmtSigned(les.error, 2)} × ${fmtSigned(net.Wo[j], 2)}` : `Σ blame × w = ${fmtSigned(arrived, 2)}`;
          let text;
          if (!roomy) text = d === 0 ? (m.activation === 'relu' && !on ? 'off → 0' : 'no blame') : `blame ${fmtSigned(d, 2)}`;
          else if (m.activation === 'relu') text = on ? `${lead}${key === 'out' ? ` = ${fmtSigned(d, 2)}` : ''}` : `${key === 'out' ? lead : `Σ = ${fmtSigned(arrived, 2)}`} · off → 0`;
          else text = `${lead} × slope ${gate.toFixed(2)} = ${fmtSigned(d, 2)}`;
          const under = crowded ? L.nodes.find(n => n.kind === 'uprod' && n.j === u.j) : null;
          let px = under ? under.x : u.kind === 'square' ? (roomy ? L.badgeX - 10 : L.badgeX) : u.x;
          let py = under ? u.y + (u.h || u.size) / 2 + 10 : u.kind === 'square' ? (roomy ? u.y + (u.h || u.size) / 2 + 10 : u.y + 20) : u.y + u.r + 7;
          if (u.kind === 'square' && py + 8 > NET_H - 40) { py = u.y + 20; px = L.badgeX; } // the bottom row would sit on the banner
          ctx.font = `600 9.5px "IBM Plex Mono", ui-monospace, monospace`;
          const half = ctx.measureText(text).width / 2 + 5, maxRight = out.x - 70; // clear of the truth, error and loss lines under the output
          if (px + half > maxRight) px = maxRight - half;
          pill(ctx, px, py, text, d === 0 ? c.ink3 : (d > 0 ? c.regular : c.irregular), c);
        });
      }
      if (ph === 'blame') { banner(conv ? convBlameBanner() : 'backward pass: each connection carries error × weight back', -1); return; }
    }
    if (ph === 'blame') { banner(convBlameBanner(), -1); return; } // a convolutional network without a hidden layer still has the pooling to go back through

    // 4 · gradients: every weight's gradient = blame at its end × activity at its start. Connections glow in the colour of
    // the step the weight will take (orange up, blue down); on pixels a scaled copy of the image slides into each weight map.
    // 5 · update: the weights themselves move (the app morphs them); the glows and copies fade out.
    const fr = conv && ph === 'gradient' ? denseGrad : Math.min(1, les.frac);
    const fade = ph === 'update' ? Math.max(0, 1 - fr / 0.4) : Math.min(1, fr * 2);
    if (fade > 0) {
      const dwOf = e => { if (e.wi == null || e.layer === 'sum') return 0; const arr = e.layer === 'out' ? les.gWo : les.gW[e.layer]; return -les.lr * arr[e.wi]; };
      const maxDw = {};
      for (const e of L.edges) { const a = Math.abs(dwOf(e)); if (a > (maxDw[e.layer] || 0)) maxDw[e.layer] = a; }
      ctx.lineCap = 'round';
      for (const e of L.edges) {
        const dw = dwOf(e); if (!dw) continue;
        const rel = Math.abs(dw) / maxDw[e.layer];
        ctx.strokeStyle = rgbStr(dw > 0 ? c.rgb.irregular : c.rgb.regular, 0.9 * fade);
        ctx.lineWidth = 2 + 8 * rel;
        ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke();
      }
      ctx.lineCap = 'butt';
      const inp = m.mode === 'pixels' ? firstInput(m) : null;
      if (inp) {
        // a weight map's gradient is blame × its input map (the image, or the stacked pooled maps), so it comes back from
        // the blame side: a copy of the input scaled by the blame sets out from the badge (from the output's edge without
        // a hidden layer) and slides back over the product map onto the weight map, where the update folds it in.
        // Without a hidden layer the error first comes back along Σ → z.
        const targets = nL
          ? L.unitColumns[0].map((u, j) => ({ node: u, from: L.badgeX, scalar: -les.lr * les.delta[0][j] }))
          : [{ node: L.nodes.find(n => n.kind === 'map'), from: L.nodes.find(n => n.kind === 'product').x, scalar: -les.lr * les.error }];
        const D = inp.D, xv = inp.vec, what = conv ? 'pooled' : 'image';
        let common = 1e-9, xmax = 1e-9;
        for (const t of targets) common = Math.max(common, Math.abs(t.scalar));
        for (let i = 0; i < D; i++) xmax = Math.max(xmax, Math.abs(xv[i]));
        if (!nL && ph === 'gradient') {
          const e = L.edges.find(q => q.layer === 'sum');
          if (e) { const wt = Math.min(1, fr / 0.3); ctx.lineCap = 'round'; ctx.strokeStyle = rgbStr(les.error > 0 ? c.rgb.regular : c.rgb.irregular, 0.9); ctx.lineWidth = 2 + 8 * Math.min(1, Math.abs(les.error)); ctx.beginPath(); ctx.moveTo(e.x2, e.y2); ctx.lineTo(e.x2 + (e.x1 - e.x2) * wt, e.y2 + (e.y1 - e.y2) * wt); ctx.stroke(); ctx.lineCap = 'butt'; }
        }
        const move = ph === 'update' ? 1 : ease(Math.max(0, Math.min(1, (fr - 0.3) / 0.7)));
        ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        targets.forEach((t, j) => {
          const n = t.node; if (!n) return;
          const W = n.size, H = n.h || n.size, captioned = W >= 40;
          ctx.globalAlpha = fade;
          if (t.scalar === 0) { if (captioned) { ctx.fillStyle = c.ink3; ctx.fillText(nL ? 'off · no change' : 'error 0 · no change', n.x, n.y + H / 2 + 3); } ctx.globalAlpha = 1; return; }
          const s0 = Math.max(20, Math.min(44, W - 12)), size = s0 + (W - s0) * move, sizeH = size * H / W, cx = t.from + (n.x - t.from) * move;
          const arr = new Float64Array(D); for (let i = 0; i < D; i++) arr[i] = t.scalar * xv[i];
          const tile = tileCanvas('lesson' + j, arr, 0, inp.w, inp.h, { max: Math.max(common * xmax, 0.3 * (nL ? TILE_FLOOR.square : TILE_FLOOR.map)) });
          ctx.imageSmoothingEnabled = false;
          ctx.globalAlpha = fade * (1 - 0.35 * move); // settles over the weight map, part transparent, until the update folds it in
          ctx.drawImage(tile.canvas, cx - size / 2, n.y - sizeH / 2, size, sizeH);
          if (conv) drawCompositeDividers(ctx, c, net, cx - size / 2, n.y - sizeH / 2, size, sizeH);
          ctx.globalAlpha = fade;
          ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(cx - size / 2, n.y - sizeH / 2, size, sizeH);
          if (captioned) { ctx.fillStyle = c.ink2; ctx.fillText(`${fmtSigned(t.scalar, Math.abs(t.scalar) < 0.001 ? 4 : 3)} × ${what}`, n.x, n.y + H / 2 + 3 + (!nL && W < 100 ? 12 : 0)); }
          ctx.globalAlpha = 1;
        });
      }
    }
    // the numbers on the connections of small layers: blame at the end × activity at the start = gradient, then the
    // weight before → after
    const counts = {}; for (const e of L.edges) counts[e.layer] = (counts[e.layer] || 0) + 1;
    const labelAlpha = ph === 'update' ? Math.min(1, fr / 0.4) : Math.min(1, fr * 2);
    ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.lineJoin = 'round';
    for (const e of L.edges) {
      if (e.wi == null || e.layer === 'sum' || counts[e.layer] > 16 || Math.abs(e.x2 - e.x1) < 60) continue; // no room on the short connections of the convolution's two-layer layout
      const grad = e.layer === 'out' ? les.gWo[e.wi] : les.gW[e.layer][e.wi]; if (!grad) continue;
      let text;
      if (ph === 'update') { const w0 = e.prev != null ? e.prev : e.w, w1 = w0 - les.lr * grad, dp = fmtSigned(w0, 2) === fmtSigned(w1, 2) ? 3 : 2; text = `w ${fmtSigned(w0, dp)} → ${fmtSigned(w1, dp)}`; }
      else { const blame = e.layer === 'out' ? les.error : les.delta[e.layer][e.ti], act = sourceValue(m, e); text = `${fmtSigned(blame, 2)} × ${fmtNum(act, 2)} = ${fmtSigned(grad, 2)}`; }
      const lx = e.x1 + (e.x2 - e.x1) * 0.4, ly = e.y1 + (e.y2 - e.y1) * 0.4 - 5;
      ctx.globalAlpha = labelAlpha; ctx.strokeStyle = c.surface; ctx.lineWidth = 3; ctx.strokeText(text, lx, ly); ctx.fillStyle = c.ink2; ctx.fillText(text, lx, ly); ctx.globalAlpha = 1;
    }
    if (ph === 'gradient') banner(conv && q >= CONV_STAGES.gradDense ? 'gradient of a filter = Σ over its positions of the blame there × the image patch under it' : nL ? 'gradient of every weight = blame at its end × activity at its start' : 'gradient of every weight = error × its input', 0);
    if (ph === 'update') banner(`update: every weight steps against its gradient, w ← w − ${les.lr} × gradient`, 0);
  }
  // The backward steps of a convolutional network, after the dense part, by fractions of the lesson's step:
  //   blame step: the dense part (wipes and pills) until blameDense; then the blame of each pooled cell (Σ unit blame ×
  //   weight) wipes back along the bands and is drawn over the pooled maps; from blameDots it goes back through the
  //   pooling to the one position per block that won the max (dots on the dimmed feature maps; ReLU lets it through only
  //   where the map was on, a hollow dot where it was off).
  //   gradient step: the dense part until gradDense; then a filter's gradient, Σ over those positions of the blame there ×
  //   the 5×5 image patch under it, accumulates position by position for filter 1 (the window on the image, the
  //   arithmetic under it) until gradOthers, then all at once for the others; its step settles over each filter as
  //   Δ until the update folds it in.
  const CONV_STAGES = { blameDense: 0.4, blameDots: 0.7, gradDense: 0.25, gradOthers: 0.8 };
  function drawLessonConv(ctx, L, c, m) {
    const les = m.lesson, net = m.net, ph = les.phase, q = les.frac, nL = net.hidden.length;
    if (!les.dPooled || !les.dConv || !(ph === 'blame' || ph === 'gradient' || ph === 'update')) return;
    const fwB = les.fwBefore, po = net.po, co = net.co, f = net.conv.f, cells = po * po, S = m.size, x = m.x;
    const fade = ph === 'update' ? Math.max(0, 1 - q / 0.4) : 1;
    if (fade <= 0 || !fwB || !fwB.conv) return;
    const pooled = L.nodes.filter(n => n.kind === 'pooled'), fmaps = L.nodes.filter(n => n.kind === 'fmap'), filters = L.nodes.filter(n => n.kind === 'filter');
    const img = L.nodes.find(n => n.kind === 'image');
    const clamp01 = v => Math.max(0, Math.min(1, v));
    const mono9 = `500 9px "IBM Plex Mono", ui-monospace, monospace`;
    // the blame wipes back to the pooled cells along the bands, in the colour and strength of each unit's blame (of the
    // error, without a hidden layer, where it first comes back along Σ → z)
    if (ph === 'blame') {
      if (!nL) {
        const e = L.edges.find(e => e.layer === 'sum');
        if (e) { const wt = Math.min(1, q / 0.3); ctx.lineCap = 'round'; ctx.strokeStyle = rgbStr(les.error > 0 ? c.rgb.regular : c.rgb.irregular, 0.9); ctx.lineWidth = 2 + 8 * Math.min(1, Math.abs(les.error)); ctx.beginPath(); ctx.moveTo(e.x2, e.y2); ctx.lineTo(e.x2 + (e.x1 - e.x2) * wt, e.y2 + (e.y1 - e.y2) * wt); ctx.stroke(); ctx.lineCap = 'butt'; }
      }
      const blameOf = b => (b.j != null && les.delta && les.delta[0] ? les.delta[0][b.j] : les.error);
      const bands = L.bands.filter(b => b.role === 'pool'), wt = clamp01((q - CONV_STAGES.blameDense) / 0.2);
      let bmax = 1e-9; for (const b of bands) bmax = Math.max(bmax, Math.abs(blameOf(b)));
      if (wt > 0) for (const b of bands) {
        const d = blameOf(b); if (!d) continue;
        const x1 = b.pts[0][0], x2 = b.pts[1][0], xw = x2 - (x2 - x1) * wt;
        ctx.save(); ctx.beginPath(); ctx.rect(xw, 0, x2 - xw, NET_H); ctx.clip();
        ctx.beginPath(); b.pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
        ctx.fillStyle = rgbStr(d > 0 ? c.rgb.regular : c.rgb.irregular, 0.12 + 0.3 * Math.abs(d) / bmax); ctx.fill();
        ctx.restore();
      }
    }
    const aPool = ph === 'blame' ? clamp01((q - CONV_STAGES.blameDense) / 0.15) : 1;   // the pooled blame appears
    const aDots = ph === 'blame' ? clamp01((q - CONV_STAGES.blameDots) / 0.15) : 1;    // then lands on the feature maps
    if (aPool <= 0) return;
    let dmax = 1e-9; for (let i = 0; i < les.dPooled.length; i++) dmax = Math.max(dmax, Math.abs(les.dPooled[i]));
    ctx.imageSmoothingEnabled = false;
    // the blame of each pooled cell, over the pooled maps
    ctx.globalAlpha = fade * aPool;
    pooled.forEach(n => {
      const tile = tileCanvas('bpool' + n.k, les.dPooled, n.k * cells, po, po, { max: dmax });
      ctx.drawImage(tile.canvas, n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
      ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
    });
    const labelled = fmaps.length && fmaps[0].y - fmaps[0].size / 2 >= 62; // the column labels need room under the caption row
    ctx.font = mono9; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    if (labelled) ctx.fillText('blame', pooled[0].x, pooled[0].y - pooled[0].size / 2 - 3);
    // the blame lands on the one position per block that won the max: dots on the dimmed feature maps
    if (aDots > 0) {
      ctx.globalAlpha = fade * aDots;
      fmaps.forEach(n => {
        const cs = n.size / co, ds = Math.max(cs + 1, 3), x0 = n.x - n.size / 2, y0 = n.y - n.size / 2;
        ctx.fillStyle = rgbStr(c.rgb.surface, 0.6); ctx.fillRect(x0, y0, n.size, n.size);
        for (let j = 0; j < cells; j++) {
          const i = fwB.conv.arg[n.k * cells + j]; if (i < 0) continue;
          const li = i - n.k * co * co, oy = Math.floor(li / co), ox = li % co, d = les.dConv[i];
          const px = x0 + (ox + 0.5) * cs - ds / 2, py = y0 + (oy + 0.5) * cs - ds / 2;
          if (d) { ctx.fillStyle = diverging(d / dmax); ctx.fillRect(px, py, ds, ds); }
          else { ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, py + 0.5, ds - 1, ds - 1); } // the map was off there: ReLU passes nothing
        }
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, n.size, n.size);
      });
      ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      if (labelled) ctx.fillText('where it lands', fmaps[0].x, fmaps[0].y - fmaps[0].size / 2 - 3);
    }
    ctx.globalAlpha = 1;
    if (ph === 'blame' || !x || !les.gWc) return;
    // the filters' gradients: Σ over the dotted positions of the blame there × the image patch under the filter, position
    // by position for filter 1, all at once for the others; the step (−lr × the sum) settles over each filter
    const per = fmaps.map(n => { const list = []; for (let j = 0; j < cells; j++) { const i = fwB.conv.arg[n.k * cells + j]; if (i >= 0 && les.dConv[i]) list.push(i); } return list; });
    const q1 = ph === 'update' ? 1 : clamp01((q - CONV_STAGES.gradDense) / (CONV_STAGES.gradOthers - CONV_STAGES.gradDense));
    const qr = ph === 'update' ? 1 : clamp01((q - CONV_STAGES.gradOthers) / (1 - CONV_STAGES.gradOthers));
    let gmax = 1e-9; for (let i = 0; i < les.gWc.length; i++) gmax = Math.max(gmax, Math.abs(les.gWc[i]));
    const scale = Math.max(les.lr * gmax, 0.3 * TILE_FLOOR.filter);
    const captioned = filters.length < 2 || filters[1].y - filters[0].y - filters[0].size >= 14; // room for a caption under each filter
    filters.forEach((n, k) => {
      const list = per[k], prog = k === 0 ? q1 : qr, nDone = Math.min(list.length, Math.ceil(prog * list.length));
      const alpha = fade * Math.min(1, prog * 3);
      if (alpha <= 0) return;
      ctx.globalAlpha = alpha;
      ctx.font = mono9; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      if (!list.length) { if (captioned) { ctx.fillStyle = c.ink3; ctx.fillText('no blame · no change', n.x, n.y + n.size / 2 + 3); } ctx.globalAlpha = 1; return; }
      const gacc = new Float64Array(f * f); // Σ so far of blame × patch, the gradient of the filter
      for (let t = 0; t < nDone; t++) { const i = list[t], li = i - k * co * co, oy = Math.floor(li / co), ox = li % co, d = les.dConv[i]; for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) gacc[dy * f + dx] += d * x[(oy + dy) * S + ox + dx]; }
      const step = Float64Array.from(gacc, v => -les.lr * v);
      const tile = tileCanvas('dfilt' + k, step, 0, f, f, { max: scale });
      ctx.globalAlpha = alpha * 0.65;
      ctx.drawImage(tile.canvas, n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
      if (captioned) { ctx.fillStyle = c.ink2; ctx.fillText(`Δ filter ${k + 1}${nDone < list.length ? ` · ${nDone}/${list.length}` : ''}`, n.x, n.y + n.size / 2 + 3); }
      // filter 1 position by position: the window on the image, the dot on the map, the arithmetic under the image
      if (k === 0 && ph === 'gradient' && nDone > 0 && q1 < 1 && img) {
        const i = list[nDone - 1], oy = Math.floor(i / co), ox = i % co, fm = fmaps[0], cs = fm.size / co;
        ctx.globalAlpha = fade; // the window and the panel are legible from the first position
        drawWindow(ctx, img, S, oy, ox, f);
        ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.strokeRect(fm.x - fm.size / 2 + ox * cs - 1.5, fm.y - fm.size / 2 + oy * cs - 1.5, cs + 3, cs + 3);
        drawGradPanel(ctx, c, m, les, oy, ox, nDone, list.length, gacc, gmax, { x: 24, y: img.y + img.h / 2 + 40 });
      }
      ctx.globalAlpha = 1;
    });
  }
  // the panel under the image while filter 1's gradient accumulates: the patch under the filter at the current position,
  // × the blame there, and the sum so far (the geometry of the convolution panel, so the two read alike)
  function drawGradPanel(ctx, c, m, les, oy, ox, nDone, nAll, gacc, gmax, at) {
    const net = m.net, f = net.conv.f, S = m.size, x = m.x, co = net.co, cell = 9, gap = 14, gw = f * cell;
    const d = les.dConv[oy * co + ox];
    const term = [], patch = []; let maxT = 1e-9, maxP = 1e-9;
    for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) { const p = x[(oy + dy) * S + ox + dx]; patch.push(p); term.push(d * p); maxP = Math.max(maxP, Math.abs(p)); maxT = Math.max(maxT, Math.abs(d * p)); }
    const grids = [['image patch', patch, maxP], ['× blame', term, maxT], ['Σ so far', Array.from(gacc), Math.max(gmax, 1e-9)]];
    ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    grids.forEach(([label, vals, mx], g) => {
      const gx = at.x + g * (gw + gap);
      ctx.fillStyle = c.ink3; ctx.fillText(label, gx + gw / 2, at.y - 3);
      for (let i = 0; i < f * f; i++) { ctx.fillStyle = diverging(vals[i] / mx); ctx.fillRect(gx + (i % f) * cell, at.y + Math.floor(i / f) * cell, cell - 1, cell - 1); }
      ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(gx - 0.5, at.y - 0.5, gw, gw);
      if (g < 2) { ctx.fillStyle = c.ink2; ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'middle'; ctx.fillText(g === 0 ? '→' : '+', gx + gw + gap / 2, at.y + gw / 2); ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; }
    });
    ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`blame ${fmtSigned(d, Math.abs(d) < 0.01 ? 4 : 3)} at col ${ox}, row ${oy}`, at.x, at.y + gw + 6);
    ctx.fillStyle = c.ink3;
    ctx.fillText(`Δ = −${les.lr} × Σ · ${nDone} of ${nAll}`, at.x, at.y + gw + 19);
  }
  // the loss against the call: −log p for a positive case, −log(1 − p) for a negative one, with the point plotted
  function drawLossGlyph(ctx, c, cx, cy, y, p) {
    const w = 36, h = 20, x0 = cx - w / 2, y0 = cy + h / 2, lossOf = q => (y ? -Math.log(q) : -Math.log(1 - q)), top = lossOf(0.03);
    ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + w, y0); ctx.stroke();
    ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i <= w; i++) { const q = 0.03 + 0.94 * i / w, yy = y0 - h * Math.min(1, lossOf(q) / top); i ? ctx.lineTo(x0 + i, yy) : ctx.moveTo(x0 + i, yy); }
    ctx.stroke();
    const q = Math.max(0.03, Math.min(0.97, p)), px = x0 + (q - 0.03) / 0.94 * w, py = y0 - h * Math.min(1, lossOf(q) / top);
    ctx.fillStyle = y ? c.irregular : c.regular; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
    ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(y ? '−log p' : '−log(1−p)', cx, cy - h / 2 - 3);
    ctx.textBaseline = 'top'; ctx.fillText('0', x0, y0 + 2); ctx.fillText('p', cx, y0 + 2); ctx.fillText('1', x0 + w, y0 + 2);
  }
  function pill(ctx, x, y, text, col, c) {
    ctx.font = `600 9.5px "IBM Plex Mono", ui-monospace, monospace`;
    const w = ctx.measureText(text).width + 10, h = 14, r = 4, x0 = x - w / 2, y0 = y - h / 2;
    ctx.beginPath(); ctx.moveTo(x0 + r, y0); ctx.lineTo(x0 + w - r, y0); ctx.quadraticCurveTo(x0 + w, y0, x0 + w, y0 + r); ctx.lineTo(x0 + w, y0 + h - r); ctx.quadraticCurveTo(x0 + w, y0 + h, x0 + w - r, y0 + h); ctx.lineTo(x0 + r, y0 + h); ctx.quadraticCurveTo(x0, y0 + h, x0, y0 + h - r); ctx.lineTo(x0, y0 + r); ctx.quadraticCurveTo(x0, y0, x0 + r, y0); ctx.closePath();
    ctx.fillStyle = c.surface; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, x, y + 0.5);
  }

  // one position of one filter, spelled out: image patch × filter = products, summed, through ReLU
  function drawConvPanel(ctx, m, k, oy, ox, at) {
    const c = colors(), net = m.net, f = net.conv.f, size = m.size, x = m.x;
    if (!x) return;
    const cell = 9, gap = 12, gw = f * cell;
    const patch = [], filt = [], prod = [];
    let maxP = 1e-9, maxW = TILE_FLOOR.filter, maxQ = 1e-9, sum = net.bc[k];
    for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) {
      const p = x[(oy + dy) * size + ox + dx], w = net.Wc[(k * f + dy) * f + dx], q = p * w;
      patch.push(p); filt.push(w); prod.push(q); sum += q;
      maxP = Math.max(maxP, Math.abs(p)); maxW = Math.max(maxW, Math.abs(w)); maxQ = Math.max(maxQ, Math.abs(q));
    }
    const act = sum > 0 ? sum : 0;
    const grids = [['image patch', patch, maxP], [`filter ${k + 1}`, filt, maxW], ['products', prod, maxQ]];
    ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    grids.forEach(([label, vals, mx], g) => {
      const gx = at.x + g * (gw + gap);
      ctx.fillStyle = c.ink3; ctx.fillText(label, gx + gw / 2, at.y - 3);
      for (let i = 0; i < f * f; i++) { ctx.fillStyle = diverging(vals[i] / mx); ctx.fillRect(gx + (i % f) * cell, at.y + Math.floor(i / f) * cell, cell - 1, cell - 1); }
      ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(gx - 0.5, at.y - 0.5, gw, gw);
      if (g < 2) { ctx.fillStyle = c.ink2; ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'middle'; ctx.fillText(g === 0 ? '×' : '=', gx + gw + gap / 2, at.y + gw / 2); ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; }
    });
    ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`Σ + bias ${fmtSigned(net.bc[k], 2)} = ${fmtSigned(sum, 2)}`, at.x, at.y + gw + 6);
    ctx.fillText(`ReLU → ${act.toFixed(2)}`, at.x, at.y + gw + 19);
    ctx.fillStyle = c.ink3;
    ctx.fillText(`→ map ${k + 1}, col ${ox}, row ${oy}`, at.x, at.y + gw + 32);
  }
  function drawPoolPanel(ctx, m, k, py, px, at, fmapMax) {
    const c = colors(), net = m.net, fw = m.fw;
    if (!fw || !fw.conv) return;
    const bs = net.conv.pool, co = net.co, cell = 12, gw = bs * cell;
    let best = -Infinity, bi = 0;
    const vals = [];
    for (let dy = 0; dy < bs; dy++) for (let dx = 0; dx < bs; dx++) { const v = fw.conv.act[(k * co + py * bs + dy) * co + px * bs + dx]; vals.push(v); if (v > best) { best = v; bi = vals.length - 1; } }
    ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    ctx.fillStyle = c.ink3; ctx.textAlign = 'left'; ctx.fillText(`${bs}×${bs} block of map ${k + 1}`, at.x, at.y - 3); ctx.textAlign = 'center';
    for (let i = 0; i < vals.length; i++) { ctx.fillStyle = sequential(Math.max(0, vals[i]) / Math.max(1e-9, fmapMax)); ctx.fillRect(at.x + (i % bs) * cell, at.y + Math.floor(i / bs) * cell, cell - 1, cell - 1); }
    ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(at.x - 0.5, at.y - 0.5, gw, gw);
    ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.strokeRect(at.x + (bi % bs) * cell - 1, at.y + Math.floor(bi / bs) * cell - 1, cell + 1, cell + 1);
    // arrow and the pooled cell
    const ax = at.x + gw + 10;
    ctx.fillStyle = c.ink2; ctx.font = `500 11px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'middle';
    ctx.fillText('max →', ax + 18, at.y + gw / 2);
    const rx = ax + 44, rs = 24;
    ctx.fillStyle = sequential(Math.max(0, best) / Math.max(1e-9, fmapMax)); ctx.fillRect(rx, at.y + gw / 2 - rs / 2, rs, rs);
    ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.strokeRect(rx, at.y + gw / 2 - rs / 2, rs, rs);
    ctx.fillStyle = c.ink3; ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    ctx.fillText('pooled', rx + rs / 2, at.y + gw / 2 - rs / 2 - 3);
    ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`largest of ${bs * bs} values = ${best.toFixed(2)}`, at.x, at.y + gw + 6);
    ctx.fillStyle = c.ink3;
    ctx.fillText(`→ pooled map ${k + 1}, col ${px}, row ${py}`, at.x, at.y + gw + 19);
    ctx.fillText('the other 15 values are dropped', at.x, at.y + gw + 32);
  }
  function drawWindow(ctx, imgNode, size, oy, ox, f) {
    const c = colors();
    const px = imgNode.w / size, x0 = imgNode.x - imgNode.w / 2 + ox * px, y0 = imgNode.y - imgNode.h / 2 + oy * px;
    ctx.fillStyle = rgbStr(c.rgb.accent, 0.18); ctx.fillRect(x0, y0, f * px, f * px);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 4; ctx.strokeRect(x0, y0, f * px, f * px);
    ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.strokeRect(x0, y0, f * px, f * px);
  }

  // the blame arithmetic of hidden unit j of layer l during a lesson, for the tooltips
  function blameNote(m, l, j) {
    const les = m.lesson; if (!les || !les.delta || les.phase === 'forward' || les.phase === 'loss') return '';
    const net = m.net, nL = net.hidden.length, d = les.delta[l][j], on = les.fwBefore.pre[l][j] > 0;
    let arrived = 0;
    if (l === nL - 1) arrived = les.error * net.Wo[j];
    else { const n1 = net.sizes[l + 1]; for (let q = 0; q < net.hidden[l + 1]; q++) arrived += les.delta[l + 1][q] * net.W[l + 1][q * n1 + j]; }
    const lead = l === nL - 1 ? `error ${fmtSigned(les.error, 2)} × w ${fmtSigned(net.Wo[j], 2)}` : `Σ blame × w = ${fmtSigned(arrived, 2)}`;
    if (m.activation === 'relu') return on ? ` · blame: ${lead} = ${fmtSigned(d, 2)}` : ` · blame: ${lead}, unit off → 0`;
    return ` · blame: ${lead} × slope ${(arrived ? d / arrived : 0).toFixed(2)} = ${fmtSigned(d, 2)}`;
  }
  // hit-test in logical coordinates; returns { kind, ref, text } or null
  function hitNetwork(canvas, px, py, m) {
    const L = canvas._layout; if (!L) return null;
    const x = px * NET_W / canvas.clientWidth, y = py * NET_W / canvas.clientWidth;
    const net = m.net, fw = m.fw;
    const inBox = (n, pad) => Math.abs(x - n.x) <= n.size / 2 + pad && Math.abs(y - n.y) <= (n.h || n.size) / 2 + pad;
    for (const n of L.nodes) {
      if (n.kind === 'output' && Math.hypot(x - n.x, y - n.y) <= n.r + 4) return { kind: 'node', ref: n, text: m.scoreLabel ? (fw ? `${m.scoreLabel} = ${fmtSigned(fw.z, 2)} · bias ${fmtSigned(net.bo, 3)}` : `${m.scoreLabel} · bias ${fmtSigned(net.bo, 3)}`) : fw ? `score z = ${fmtSigned(fw.z, 2)} → P(${m.positiveName}) = ${fw.p.toFixed(3)} · bias ${fmtSigned(net.bo, 3)}` : `output · bias ${fmtSigned(net.bo, 3)}` };
      if (n.kind === 'unit' && Math.hypot(x - n.x, y - n.y) <= n.r + 4) {
        const a = fw ? ` · activation ${fmtNum(fw.a[n.l + 1][n.j], 3)}` : '';
        const out = n.l === net.hidden.length - 1 ? ` · to output ${fmtSigned(net.Wo[n.j], 3)}` : '';
        return { kind: 'node', ref: n, text: `hidden unit ${net.hidden.length > 1 ? (n.l + 1) + '.' : ''}${n.j + 1}${a} · bias ${fmtSigned(net.b[n.l][n.j], 3)}${out}${blameNote(m, n.l, n.j)}` };
      }
      if (n.kind === 'input' && Math.hypot(x - n.x, y - n.y) <= n.r + 4) return { kind: 'node', ref: n, text: `${m.featureNames[n.i]}${m.x ? ` · standardized value ${fmtSigned(m.x[n.i], 2)}` : ''}` };
      if (n.kind === 'square' && inBox(n, 2)) {
        const W = firstWeights(m, n.j); let mm = 0; for (let i = 0; i < W.D; i++) mm = Math.max(mm, Math.abs(W.arr[W.off + i]));
        const { i, j, k } = cellAt(n, W, x, y), w = W.arr[W.off + k];
        const lessonTxt = m.lesson && (m.lesson.phase === 'gradient' || m.lesson.phase === 'update' || m.lesson.phase === 'check') ? ` · this lesson adds ${fmtSigned(-m.lesson.lr * m.lesson.delta[0][n.j], 3)} × image to this map` : '';
        return { kind: 'node', ref: n, text: `${m.contrast ? 'code number' : 'hidden unit'} ${n.j + 1} weights: max |w| ${mm.toFixed(3)} · ${cellName(m, i, j)} weight ${fmtSigned(w, 3)}${lessonTxt}` };
      }
      if (n.kind === 'uprod' && (inBox(n, 2) || Math.hypot(x - n.x - n.size / 2 - 16, y - n.y) <= 13)) {
        const W = firstWeights(m, n.j), inp = firstInput(m), what = net.conv ? 'pooled' : 'pixel';
        let sum = 0; if (inp) for (let i = 0; i < W.D; i++) sum += W.arr[W.off + i] * inp.vec[i];
        const a = fw ? fw.a[1][n.j] : null;
        const out = net.hidden.length === 1 && !m.contrast ? ` · to output ${fmtSigned(net.Wo[n.j], 3)}` : '';
        const pix = inBox(n, 0) && inp ? (() => { const { i, j, k } = cellAt(n, W, x, y); return ` · ${cellName(m, i, j)}: ${fmtSigned(W.arr[W.off + k], 3)} × ${fmtSigned(inp.vec[k], 2)} = ${fmtSigned(W.arr[W.off + k] * inp.vec[k], 3)}`; })() : '';
        return { kind: 'node', ref: n, text: `${m.contrast ? 'code number' : 'unit'} ${n.j + 1}: Σ weight × ${what} ${inp ? fmtSigned(sum, 2) : '?'} + bias ${fmtSigned(net.b[0][n.j], 2)}${a != null ? ` → ${m.activationLabel} → ${fmtNum(a, 2)}` : ''}${out}${pix}${blameNote(m, 0, n.j)}` };
      }
      if (n.kind === 'map' && inBox(n, 0)) {
        const W = firstWeights(m, null), inp = firstInput(m), { i, j, k } = cellAt(n, W, x, y), w = W.arr[W.off + k];
        const lessonTxt = m.lesson && (m.lesson.phase === 'gradient' || m.lesson.phase === 'update' || m.lesson.phase === 'check') ? ` · this lesson adds ${fmtSigned(-m.lesson.lr * m.lesson.error, 3)} × image to the map` : '';
        return { kind: 'node', ref: n, text: `${cellName(m, i, j)} weight ${fmtSigned(w, 3)}${inp ? ` × input ${fmtSigned(inp.vec[k], 2)}` : ''}${lessonTxt}` };
      }
      if (n.kind === 'product' && inBox(n, 0) && firstInput(m)) {
        const W = firstWeights(m, null), inp = firstInput(m), { i, j, k } = cellAt(n, W, x, y);
        return { kind: 'node', ref: n, text: `${cellName(m, i, j)}: weight ${fmtSigned(W.arr[W.off + k], 3)} × input ${fmtSigned(inp.vec[k], 2)} = ${fmtSigned(W.arr[W.off + k] * inp.vec[k], 3)}` };
      }
      if (n.kind === 'filter' && inBox(n, 2)) {
        let mm = 0; const f = net.conv.f; for (let i = 0; i < f * f; i++) mm = Math.max(mm, Math.abs(net.Wc[n.k * f * f + i]));
        return { kind: 'node', ref: n, text: `filter ${n.k + 1}: ${f}×${f} weights, max |w| ${mm.toFixed(3)} · bias ${fmtSigned(net.bc[n.k], 3)}` };
      }
      if (n.kind === 'fmap' && inBox(n, 2)) {
        const i = Math.max(0, Math.min(net.co - 1, Math.floor((x - (n.x - n.size / 2)) / n.size * net.co))), j = Math.max(0, Math.min(net.co - 1, Math.floor((y - (n.y - n.size / 2)) / n.size * net.co)));
        if (!fw || !fw.conv) return { kind: 'node', ref: n, i, j, text: `feature map ${n.k + 1}: where filter ${n.k + 1} fires (${net.co}×${net.co}) · the panel shows the arithmetic at column ${i}, row ${j}` };
        return { kind: 'node', ref: n, i, j, text: `feature map ${n.k + 1} at column ${i}, row ${j}: ${fmtNum(fw.conv.act[(n.k * net.co + j) * net.co + i], 2)} · see the arithmetic under the image` };
      }
      if (n.kind === 'pooled' && inBox(n, 2)) {
        const i = Math.max(0, Math.min(net.po - 1, Math.floor((x - (n.x - n.size / 2)) / n.size * net.po))), j = Math.max(0, Math.min(net.po - 1, Math.floor((y - (n.y - n.size / 2)) / n.size * net.po)));
        const v = fw && fw.conv ? ` = ${fmtNum(fw.conv.v[(n.k * net.po + j) * net.po + i], 2)}` : '';
        return { kind: 'node', ref: n, i, j, text: `pooled map ${n.k + 1}, col ${i}, row ${j}${v}: the largest of the ${net.conv.pool}×${net.conv.pool} block outlined on feature map ${n.k + 1}` };
      }
    }
    let best = null, bestD = 6;
    for (const e of L.edges) {
      if (e.layer === 'sum') continue;
      const dx = e.x2 - e.x1, dy = e.y2 - e.y1, len2 = dx * dx + dy * dy;
      const t = Math.max(0, Math.min(1, ((x - e.x1) * dx + (y - e.y1) * dy) / len2));
      const d = Math.hypot(x - (e.x1 + t * dx), y - (e.y1 + t * dy));
      if (d < bestD) { bestD = d; best = e; }
    }
    if (best) {
      const sv = sourceValue(m, best);
      const contrib = sv != null ? ` × ${best.fromIdx != null ? 'input' : 'value'} ${fmtSigned(sv, 2)} = ${fmtSigned(best.w * sv, 3)}` : '';
      const moved = best.prev != null && Math.abs(best.w - best.prev) > 1e-9 ? ` · last step ${fmtSigned(best.w - best.prev, 4)}` : '';
      let lessonTxt = '';
      const les = m.lesson;
      if (les && best.wi != null && (les.phase === 'gradient' || les.phase === 'update' || les.phase === 'check')) {
        const arr = best.layer === 'out' ? les.gWo : les.gW[best.layer], fwB = les.fwBefore;
        const inp = best.layer === 'out' ? fwB.a[fwB.a.length - 1][best.fi] : best.layer === 0 ? m.x[best.fi] : fwB.a[best.layer][best.fi];
        const blame = best.layer === 'out' ? les.error : les.delta[best.layer][best.ti];
        lessonTxt = ` · this lesson: Δw = −${les.lr} × ${best.layer === 'out' ? 'error' : 'blame'} ${fmtSigned(blame, 3)} × input ${fmtSigned(inp, 2)} = ${fmtSigned(-les.lr * arr[best.wi], 4)}`;
      }
      return { kind: 'edge', ref: best, text: `weight ${best.fromName} → ${best.toName}: ${fmtSigned(best.w, 3)}${contrib}${moved}${lessonTxt}` };
    }
    return null;
  }

  // ------------------------------------------------------------------ SVG helpers
  const SVG_NS = 'http://www.w3.org/2000/svg';
  function niceStep(range, n) {
    const raw = range / Math.max(1, n);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const r = raw / p;
    return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p;
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

  /*
   * curves: history [{epoch, loss, acc, testLoss, testAcc}], key 'loss' | 'acc', showTest, maxEpoch
   */
  function drawCurves(svg, opt) {
    const W = 400, H = 170, ml = 40, mr = 16, mt = 10, mb = 22;
    const pw = W - ml - mr, ph = H - mt - mb;
    const hist = opt.history || [];
    const isAcc = opt.key === 'acc';
    const trainKey = isAcc ? 'acc' : 'loss', testKey = isAcc ? 'testAcc' : 'testLoss';
    const maxX = Math.max(10, opt.maxEpoch || 0, hist.length ? hist[hist.length - 1].epoch : 0);
    let maxY = 1;
    if (!isAcc) { for (const h of hist) { maxY = Math.max(maxY, h.loss, opt.showTest ? h.testLoss : 0); } maxY = Math.ceil(maxY * 2) / 2; }
    const sx = e => ml + e / maxX * pw, sy = v => mt + (1 - v / maxY) * ph;
    let g = '';
    const ystep = isAcc ? 0.25 : niceStep(maxY, 4);
    g += '<g class="grid">';
    for (let v = 0; v <= maxY + 1e-9; v += ystep) {
      g += `<line x1="${ml}" x2="${W - mr}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"/>`;
      g += `<text x="${ml - 5}" y="${(sy(v) + 3.5).toFixed(1)}" text-anchor="end">${isAcc ? Math.round(v * 100) + '%' : v.toFixed(ystep < 1 ? 1 : 0)}</text>`;
    }
    g += '</g>';
    const xstep = niceStep(maxX, 5);
    for (let e = 0; e <= maxX + 1e-9; e += xstep) g += `<text x="${sx(e).toFixed(1)}" y="${H - 6}" text-anchor="middle">${e}</text>`;
    g += `<line class="axis" x1="${ml}" x2="${W - mr}" y1="${mt + ph}" y2="${mt + ph}"/>`;
    if (isAcc) g += `<line class="chance" x1="${ml}" x2="${W - mr}" y1="${sy(0.5)}" y2="${sy(0.5)}"/>`;
    // the honest ceiling (training accuracy above it means memorised mislabels), labelled under the line at the right, and
    // the epoch of the lowest test loss, labelled on the free side of its line: the top of the loss chart, the bottom of
    // the accuracy chart, where the curves are not
    if (isAcc && opt.ceiling != null) g += `<line class="ceiling" x1="${ml}" x2="${W - mr}" y1="${sy(opt.ceiling).toFixed(1)}" y2="${sy(opt.ceiling).toFixed(1)}"/><text class="marker ceiling" x="${W - mr}" y="${(sy(opt.ceiling) + 11).toFixed(1)}" text-anchor="end">honest ceiling ${Math.round(opt.ceiling * 100)}%</text>`;
    if (opt.stopAt != null && opt.showTest) { const x = sx(opt.stopAt), left = x > ml + pw / 2; g += `<line class="stop" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${mt}" y2="${mt + ph}"/><text class="marker stop" x="${(x + (left ? -4 : 4)).toFixed(1)}" y="${isAcc ? mt + ph - 5 : mt + 10}" text-anchor="${left ? 'end' : 'start'}">lowest test loss · epoch ${opt.stopAt}</text>`; }
    const path = key => hist.map((h, i) => `${i ? 'L' : 'M'}${sx(h.epoch).toFixed(1)} ${sy(clamp(h[key], 0, maxY)).toFixed(1)}`).join(' ');
    if (hist.length) {
      if (opt.showTest) g += `<path class="test" d="${path(testKey)}"/>`;
      g += `<path class="train" d="${path(trainKey)}"/>`;
      const last = hist[hist.length - 1];
      if (opt.showTest) g += `<circle class="end test" r="3.5" cx="${sx(last.epoch).toFixed(1)}" cy="${sy(clamp(last[testKey], 0, maxY)).toFixed(1)}"/>`;
      g += `<circle class="end" r="3.5" cx="${sx(last.epoch).toFixed(1)}" cy="${sy(clamp(last[trainKey], 0, maxY)).toFixed(1)}"/>`;
    }
    g += '<g class="hover"></g>';
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = g;
    svg._curve = { hist, sx, sy, maxX, maxY, isAcc, trainKey, testKey, showTest: opt.showTest, ml, mr, W, H, mt, ph };
    if (!svg._hooked) {
      svg._hooked = true;
      svg.addEventListener('mousemove', ev => {
        const d = svg._curve; if (!d || !d.hist.length) return;
        const r = svg.getBoundingClientRect();
        const x = (ev.clientX - r.left) / r.width * d.W;
        const e = clamp(Math.round((x - d.ml) / (d.W - d.ml - d.mr) * d.maxX), 0, d.maxX);
        let best = d.hist[0];
        for (const h of d.hist) if (Math.abs(h.epoch - e) < Math.abs(best.epoch - e)) best = h;
        const hx = d.sx(best.epoch);
        const fmt = v => (d.isAcc ? Math.round(v * 100) + '%' : v.toFixed(3));
        const text = `epoch ${best.epoch} · train ${fmt(best[d.trainKey])}${d.showTest ? ` · test ${fmt(best[d.testKey])}` : ''}`;
        const tw = text.length * 6.3 + 12;
        const tx = clamp(hx - tw / 2, 2, d.W - tw - 2);
        svg.querySelector('.hover').innerHTML =
          `<line class="cross" x1="${hx.toFixed(1)}" x2="${hx.toFixed(1)}" y1="${d.mt}" y2="${d.mt + d.ph}"/>` +
          `<circle class="end" r="3.5" cx="${hx.toFixed(1)}" cy="${d.sy(clamp(best[d.trainKey], 0, d.maxY)).toFixed(1)}"/>` +
          `<rect class="tipbox" x="${tx.toFixed(1)}" y="${d.mt - 2}" width="${tw.toFixed(1)}" height="16" rx="4"/>` +
          `<text class="tiptext" x="${(tx + tw / 2).toFixed(1)}" y="${d.mt + 9.5}" text-anchor="middle">${esc(text)}</text>`;
      });
      svg.addEventListener('mouseleave', () => { const h = svg.querySelector('.hover'); if (h) h.innerHTML = ''; });
    }
  }

  /*
   * scatter: points [{id, name, x, y, cls ('regular'|'irregular'|'unknown'), split, selected}], xLabel, yLabel, onSelect(id)
   */
  function drawScatter(svg, opt) {
    const W = 720, H = 340, ml = 52, mr = 14, mt = 10, mb = 36;
    const pw = W - ml - mr, ph = H - mt - mb;
    const pts = opt.points;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const pad = (a, b) => { const d = (b - a) || 1; return [a - d * 0.06, b + d * 0.06]; };
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
    const sx = v => ml + (v - x0) / (x1 - x0) * pw, sy = v => mt + (1 - (v - y0) / (y1 - y0)) * ph;
    let g = '<g class="grid">';
    const xstep = niceStep(x1 - x0, 5), ystep = niceStep(y1 - y0, 5);
    const dec = s => Math.max(0, -Math.floor(Math.log10(s)));
    for (let v = Math.ceil(x0 / xstep) * xstep; v <= x1 + 1e-9; v += xstep) {
      g += `<line x1="${sx(v).toFixed(1)}" x2="${sx(v).toFixed(1)}" y1="${mt}" y2="${mt + ph}"/>`;
      g += `<text x="${sx(v).toFixed(1)}" y="${mt + ph + 14}" text-anchor="middle">${v.toFixed(dec(xstep))}</text>`;
    }
    for (let v = Math.ceil(y0 / ystep) * ystep; v <= y1 + 1e-9; v += ystep) {
      g += `<line x1="${ml}" x2="${W - mr}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"/>`;
      g += `<text x="${ml - 5}" y="${(sy(v) + 3.5).toFixed(1)}" text-anchor="end">${v.toFixed(dec(ystep))}</text>`;
    }
    g += '</g>';
    g += `<text class="axlabel" x="${ml + pw / 2}" y="${H - 4}" text-anchor="middle">${esc(opt.xLabel)}</text>`;
    g += `<text class="axlabel" transform="translate(11 ${mt + ph / 2}) rotate(-90)" text-anchor="middle">${esc(opt.yLabel)}</text>`;
    if (opt.segments) for (const s of opt.segments) g += `<line class="seg" x1="${sx(s.x1).toFixed(1)}" y1="${sy(s.y1).toFixed(1)}" x2="${sx(s.x2).toFixed(1)}" y2="${sy(s.y2).toFixed(1)}"/>`;
    const ordered = pts.slice().sort((a, b) => (a.selected ? 1 : 0) - (b.selected ? 1 : 0));
    for (const p of ordered) {
      const cls = `pt ${p.cls}${p.split === 'test' ? ' test' : ''}${p.selected ? ' selected' : ''}${p.small ? ' small' : ''}`;
      g += `<circle class="${cls}"${p.color ? ` style="fill:${p.color}"` : ''} data-id="${p.id}" r="${p.selected ? 6 : p.small ? 3 : 4.5}" cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}"><title>${esc(p.name)} · ${esc(opt.xLabel)} ${p.x.toFixed(3)} · ${esc(opt.yLabel)} ${p.y.toFixed(3)}${p.clsName ? ' · ' + p.clsName : ''}</title></circle>`;
    }
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = g;
    if (!svg._hooked) {
      svg._hooked = true;
      svg.addEventListener('click', ev => {
        const t = ev.target.closest('circle[data-id]');
        if (t && opt.onSelect) opt.onSelect(Number(t.getAttribute('data-id')));
      });
    }
  }

  return { refreshTheme, colors, diverging, divergingRgb, sequential, unitColor, renderThumb, renderFingerprint, renderBigImage, renderEvidence, renderMeasurement, drawNetwork, hitNetwork, drawCurves, drawScatter, drawSeries, drawSimilarityMatrix, drawLineup, drawViews, drawSlide, hitSlide, drawRanked, renderSlideThumb, drawSlideNetwork, hitSlideNetwork, drawAttentionMap, hitAttentionMap, sweepPlan, sweepState, convPlan, convAt, convAnim, CONV_STAGES, pixelRgb, fmtSigned, fmtNum, NET_W, NET_H };
})();
