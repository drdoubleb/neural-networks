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
      L.captions.push({ x: xs[0], text: `INPUT · ${D} MEASUREMENTS` });
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
      // with two dense layers the caption row is crowded, so the conv caption drops to its short form
      L.captions.push({ x: (xF + xP) / 2 + 8, text: hidden.length > 1 ? `CONV · ${K} FILTERS ${conv.f}×${conv.f} · POOL ${conv.pool}×${conv.pool}` : `CONV · ${K} FILTERS ${conv.f}×${conv.f} · RELU · MAX-POOL ${conv.pool}×${conv.pool}` });
      L.bands.push({ pts: [[img.x + img.w / 2, img.y - img.h / 2], [xF - fs / 2 - 4, ys[0] - fs / 2], [xF - fs / 2 - 4, ys[K - 1] + fs / 2], [img.x + img.w / 2, img.y + img.h / 2]], label: '' });
      L.bandLabel = { x: (img.x + img.w / 2 + xF - fs / 2) / 2, y: 44, text: 'each filter slides over the image' };
      L.footnotes = [{ x: xF, text: 'filters' }, { x: xM, text: 'feature maps' }, { x: xP, text: 'pooled' }];
      L.poolCaption = { x: xP, text: `each cell = largest of a ${conv.pool}×${conv.pool} block` };
      const F = net.featureCount;
      const xs = hidden.length === 0 ? [] : hidden.length === 1 ? [530] : [510, 630];
      const poolBox = { x1: xP + ps / 2, yTop: ys[0] - ps / 2, yBot: ys[K - 1] + ps / 2 };
      let prevCol = null;
      hidden.forEach((h, l) => {
        const col = spread(h, yc, Math.min(46, (NET_H - 110) / Math.max(1, h - 1))).map((y, j) => add({ kind: 'unit', l, j, x: xs[l], y, r: 17 }));
        unitColumns.push(col);
        L.captions.push({ x: xs[l], text: `HIDDEN ${hidden.length > 1 ? l + 1 : ''} · ${h} ${m.activationLabel.toUpperCase()}` });
        if (l === 0) { for (const u of col) L.bands.push({ pts: [[poolBox.x1, poolBox.yTop], [u.x - u.r, u.y - u.r], [u.x - u.r, u.y + u.r], [poolBox.x1, poolBox.yBot]], unit: u }); L.bandLabel2 = { x: (poolBox.x1 + xs[0]) / 2, y: 44, text: `${fmtInt(F)} weights per unit` }; }
        else prevCol.forEach((a, i) => col.forEach((b, j) => link(a, b, net.W[l][j * net.sizes[l] + i], l, { wi: j * net.sizes[l] + i, fi: i, ti: j, prev: prevW(l, j * net.sizes[l] + i), fromName: `unit ${l}.${i + 1}`, toName: `unit ${l + 1}.${j + 1}` })));
        prevCol = col;
      });
      if (!hidden.length) { L.bands.push({ pts: [[poolBox.x1, poolBox.yTop], [output.x - output.r, output.y - output.r], [output.x - output.r, output.y + output.r], [poolBox.x1, poolBox.yBot]] }); L.bandLabel2 = { x: (poolBox.x1 + output.x) / 2, y: 44, text: `${fmtInt(F)} weights` }; }
      else prevCol.forEach((a, i) => link(a, output, net.Wo[i], 'out', { wi: i, fi: i, prev: prevWo(i), fromName: `unit ${hidden.length > 1 ? hidden.length + '.' : ''}${i + 1}`, toName: 'output' }));
    }
    L.captions.push({ x: output.x, text: 'OUTPUT' });
    L.output = output; L.unitColumns = unitColumns;
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
    if (L.bandLabel) ctx.fillText(L.bandLabel.text, L.bandLabel.x, L.bandLabel.y || NET_H - 16);
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
        if (n.kind === 'map') tile = tileCanvas('out', net.Wo, 0, m.size, m.size, { floor: TILE_FLOOR.map });
        else if (n.kind === 'square') tile = tileCanvas('h' + n.j, net.W[0], n.j * net.sizes[0], m.size, m.size, { floor: TILE_FLOOR.square });
        else if (n.kind === 'filter') tile = tileCanvas('f' + n.k, net.Wc, n.k * net.conv.f * net.conv.f, net.conv.f, net.conv.f, { floor: TILE_FLOOR.filter });
        else if (n.kind === 'product' && m.x && (m.reveal != null ? m.reveal >= 1 : stage >= 1)) { const prod = new Float64Array(net.D); let sum = 0; for (let i = 0; i < net.D; i++) { prod[i] = net.Wo[i] * m.x[i]; sum += prod[i]; } tile = tileCanvas('prod', prod, 0, m.size, m.size); tile.sum = sum; }
        else if (n.kind === 'uprod' && m.x && (m.reveal != null ? m.reveal >= 1 : stage >= 1)) { const D = net.D, off = n.j * D, prod = new Float64Array(D); for (let i = 0; i < D; i++) prod[i] = net.W[0][off + i] * m.x[i]; tile = tileCanvas('uprod' + n.j, prod, 0, m.size, m.size); }
        ctx.imageSmoothingEnabled = false;
        if (tile) ctx.drawImage(tile.canvas, n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        else { ctx.fillStyle = pending; ctx.fillRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size); }
        ctx.strokeStyle = isHov ? c.ink : c.lineStrong; ctx.lineWidth = isHov ? 2 : 1;
        ctx.strokeRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        if (n.kind === 'map' || (n.kind === 'square' && n.captioned)) ctx.fillText(`±${tile.max.toFixed(2)}`, n.x, n.y + n.size / 2 + 3);
        if (n.kind === 'product') ctx.fillText(tile ? `Σ = ${fmtSigned(tile.sum, 2)}  (bias ${fmtSigned(net.bo, 2)})` : (m.reveal != null ? '' : 'select a nucleus'), n.x, n.y + n.size / 2 + 3);
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
            if (n.kind === 'fmap') maskBefore = anim.phase === 'scan1' ? (n.k === 0 ? anim.pos : 0) : anim.phase === 'scan' ? (n.k === 0 ? side * side : anim.pos) : side * side;
            else maskBefore = anim.phase === 'pool1' ? (n.k === 0 ? anim.posP : 0) : anim.phase === 'pool' ? (n.k === 0 ? side * side : anim.posP) : (anim.phase === 'scan1' || anim.phase === 'scan') ? 0 : side * side;
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
    // convolution walk-through: the sliding window on the image and the arithmetic of the current position
    if (net.conv && m.x) {
      const img = L.nodes.find(n => n.kind === 'image');
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
    }
    return L;
  }
  function ACT_SIGNED(name) { return name === 'tanh'; }

  // ---- shared by the lesson and the test walk-through
  // a banner along the bottom of the diagram naming the step, with an arrow for the passes (dir +1 forward, −1 back)
  function drawBanner(ctx, c, L, text, dir) {
    const y = L.bandLabel || L.poolCaption || L.bands.some(b => b.label) ? NET_H - 34 : NET_H - 16; // above the band label of the pixel layouts, else along the free bottom edge
    ctx.font = `600 11px "IBM Plex Sans", system-ui, sans-serif`; ctx.fillStyle = c.accent; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, 400, y);
    if (dir) {
      const gap = ctx.measureText(text).width / 2 + 12, x0 = 150, x1 = 650;
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
  function prepRowAt(L, img) { const size = 44, gap = 30, x = 16; return { size, gap, at: { x, y: img.y - img.h / 2 - 96 }, xs: [x, x + size + gap, x + 2 * (size + gap)] }; }
  function drawPrepRow(ctx, L, c, m, t) { // the three tiles, fading in one after another; t = 1 shows the finished row
    const img = L.nodes.find(n => n.kind === 'image'); if (!img || !m.x || !m.specimen || !m.inputMean) return null;
    const S = m.size, { size, gap, at, xs } = prepRowAt(L, img);
    const pxMean = new Uint8ClampedArray(S * S); for (let i = 0; i < S * S; i++) pxMean[i] = Math.round(255 * (1 - m.inputMean[i]));
    const diff = tileCanvas('xin', m.x, 0, S, S).canvas;
    const fade = (a, b) => Math.max(0, Math.min(1, (t - a) / (b - a)));
    const label = (text, x) => { ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(text, x + size / 2, at.y - 4); };
    const glyph = (text, x) => { ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, x, at.y + size / 2); };
    const frame = x => { ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, at.y - 0.5, size + 1, size + 1); };
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = fade(0, 0.12);
    ctx.drawImage(imageToCanvas(m.specimen.px, S, m.tint), xs[0], at.y, size, size); frame(xs[0]); label('this nucleus', xs[0]);
    ctx.globalAlpha = fade(0.2, 0.36);
    glyph('−', xs[0] + size + gap / 2);
    ctx.drawImage(imageToCanvas(pxMean, S, m.tint), xs[1], at.y, size, size); frame(xs[1]); label('mean nucleus', xs[1]);
    ctx.globalAlpha = fade(0.44, 0.6);
    glyph('=', xs[1] + size + gap / 2);
    ctx.drawImage(diff, xs[2], at.y, size, size); frame(xs[2]); label('difference', xs[2]);
    ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('what the network sees', at.x, at.y + size + 6);
    ctx.fillStyle = c.ink3; ctx.fillText('orange: more ink than average', at.x, at.y + size + 19); ctx.fillText('blue: less ink than average', at.x, at.y + size + 32);
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
    const net = m.net, nL = net.hidden.length, x = m.x, fw = m.fw, S = m.size;
    if (!x || !fw || !m.specimen) return;
    const img = L.nodes.find(n => n.kind === 'image');
    const rows = nL ? L.unitColumns[0] : [L.nodes.find(n => n.kind === 'map')];
    const n = rows.length, D = net.D;
    const T1 = 12000, T2 = n > 1 ? 6000 : 0, tm = t * (T1 + T2);
    const monoS = `500 10px "IBM Plex Mono", ui-monospace, monospace`;
    const badgeFont = `500 10px "IBM Plex Mono", ui-monospace, monospace`;
    const fmtA = v => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));
    rows.forEach((w, j) => {
      if (!w) return;
      const st = j === 0 ? (tm < T1 ? imageStage(tm / T1) : { name: 'done', v: 1 }) : (tm < T1 ? { name: 'wait', v: 0 } : imageStage((tm - T1) / T2));
      if (st.name === 'wait') return;
      const pn = nL ? L.nodes.find(q => q.kind === 'uprod' && q.j === j) : L.nodes.find(q => q.kind === 'product');
      const off = nL ? j * D : 0, Wrow = nL ? net.W[0] : net.Wo;
      const prod = new Float64Array(D); let pos = 0, neg = 0;
      for (let i = 0; i < D; i++) { const q = Wrow[off + i] * x[i]; prod[i] = q; if (q > 0) pos += q; else neg += q; }
      const bias = nL ? net.b[0][j] : net.bo, z = pos + neg + bias, act = nL ? fw.a[1][j] : null;
      const size = w.size, badgeX = L.badgeX;
      const tile = () => tileCanvas('sweepprod' + j, prod, 0, S, S).canvas;
      ctx.imageSmoothingEnabled = false;
      if (st.name === 'overlay') { // the nucleus slides along the band and settles over the weight map, half transparent
        const e = st.v * st.v * (3 - 2 * st.v);
        const cx = img.x + (w.x - img.x) * e, cy = img.y + (w.y - img.y) * e, sz = img.w + (size - img.w) * e;
        ctx.globalAlpha = 1 - 0.5 * e;
        ctx.drawImage(tileCanvas('xin', x, 0, S, S).canvas, cx - sz / 2, cy - sz / 2, sz, sz);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(cx - sz / 2, cy - sz / 2, sz, sz);
        return;
      }
      if (st.name === 'wipe') { // the product map wipes in over the overlay, cell = pixel × weight
        ctx.globalAlpha = 0.5 * (1 - st.v);
        ctx.drawImage(tileCanvas('xin', x, 0, S, S).canvas, w.x - size / 2, w.y - size / 2, size, size);
        ctx.globalAlpha = 1;
        ctx.save(); ctx.beginPath(); ctx.rect(w.x - size / 2, w.y - size / 2, size, size * st.v); ctx.clip();
        ctx.drawImage(tile(), w.x - size / 2, w.y - size / 2, size, size);
        ctx.restore();
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(w.x - size / 2, w.y - size / 2, size, size);
        if (j === 0) { const r0 = Math.min(S - 4, 4 * Math.floor(st.v * (S / 4))), c0 = S / 2 - 2; drawWindow(ctx, img, S, r0, c0, 4); drawPixelPanel(ctx, c, m, Wrow, off, r0, c0, { x: 24, y: img.y + img.h / 2 + 40 }); }
        return;
      }
      // from here on the product map exists; it slides to its slot, then is summed
      const e = st.name === 'aside' ? st.v * st.v * (3 - 2 * st.v) : 1;
      const px = w.x + (pn.x - w.x) * e, py = w.y + (pn.y - w.y) * e;
      ctx.drawImage(tile(), px - size / 2, py - size / 2, size, size);
      ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(px - size / 2, py - size / 2, size, size);
      if (st.name === 'aside') return;
      // the sum: a scan line sweeps the map, the unit counts, bars show the positive and negative parts
      const rowsDone = st.name === 'sum' ? Math.floor(st.v * S) : S;
      let sum = 0, posNow = 0, negNow = 0;
      for (let i = 0; i < rowsDone * S; i++) { const q = prod[i]; sum += q; if (q > 0) posNow += q; else negNow += q; }
      if (st.name === 'sum') { const sy = pn.y - size / 2 + st.v * size; ctx.strokeStyle = c.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(pn.x - size / 2, sy); ctx.lineTo(pn.x + size / 2, sy); ctx.stroke(); }
      const half = size / 2 - 2, maxAbs = Math.max(pos, -neg, 1e-9), by = pn.y + size / 2 + 4;
      ctx.fillStyle = c.irregular; ctx.fillRect(pn.x, by, posNow / maxAbs * half, 4);
      ctx.fillStyle = c.regular; ctx.fillRect(pn.x + negNow / maxAbs * half, by, -negNow / maxAbs * half, 4);
      ctx.fillStyle = c.ink; ctx.fillRect(pn.x + (posNow + negNow) / maxAbs * half - 1, by - 2, 2, 8);
      if (nL) {
        const showAct = st.name === 'done' || (st.name === 'relu' && st.v >= 0.5);
        const label = st.name === 'sum' ? fmtA(sum) : st.name === 'relu' && st.v < 0.5 ? fmtA(z) : fmtA(act);
        circleNode(ctx, badgeX, pn.y, 11, showAct ? unitFill(act, fw.a[1], ACT_SIGNED(m.activation)) : c.surface2, label, badgeFont);
        if (size >= 56) {
          ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
          if (st.name === 'sum') ctx.fillText('Σ so far', badgeX - 8, pn.y + 15);
          else if (st.name === 'relu' && st.v < 0.5) ctx.fillText(`Σ + b ${fmtSigned(bias, 2)}`, badgeX - 8, pn.y + 15);
          else if (st.name === 'relu' || st.name === 'done') drawReluGlyph(ctx, c, badgeX, pn.y - 24, z);
        }
      } else {
        ctx.font = monoS; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const line = st.name === 'sum' ? `Σ so far ${fmtSigned(sum, 2)}` : `Σ = ${fmtSigned(pos + neg, 2)}  + bias ${fmtSigned(bias, 2)}  = z ${fmtSigned(z, 2)}`;
        ctx.fillText(line, pn.x, by + 8);
      }
    });
  }
  // the magnifier under the image: one 4×4 block of the multiplication, cell by cell, with the numbers of its largest cell
  function drawPixelPanel(ctx, c, m, Wrow, off, r0, c0, at) {
    const S = m.size, x = m.x, cell = 9, gap = 12, gw = 4 * cell;
    const pix = [], wts = [], prods = [];
    let maxP = 1e-9, maxW = TILE_FLOOR.square, maxQ = 1e-9, big = 0;
    for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) {
      const i = (r0 + dy) * S + c0 + dx, p = x[i], w = Wrow[off + i], q = p * w;
      pix.push(p); wts.push(w); prods.push(q);
      maxP = Math.max(maxP, Math.abs(p)); maxW = Math.max(maxW, Math.abs(w)); maxQ = Math.max(maxQ, Math.abs(q));
      if (Math.abs(q) > Math.abs(prods[big])) big = prods.length - 1;
    }
    const grids = [['pixels', pix, maxP], ['weights', wts, maxW], ['products', prods, maxQ]];
    ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; ctx.textAlign = 'center';
    grids.forEach(([label, vals, mx], g) => {
      const gx = at.x + g * (gw + gap);
      ctx.fillStyle = c.ink3; ctx.fillText(label, gx + gw / 2, at.y - 3);
      for (let i = 0; i < 16; i++) { ctx.fillStyle = diverging(vals[i] / mx); ctx.fillRect(gx + (i % 4) * cell, at.y + Math.floor(i / 4) * cell, cell - 1, cell - 1); }
      ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(gx - 0.5, at.y - 0.5, gw, gw);
      ctx.strokeStyle = c.ink; ctx.strokeRect(gx + (big % 4) * cell - 0.5, at.y + Math.floor(big / 4) * cell - 0.5, cell, cell);
      if (g < 2) { ctx.fillStyle = c.ink2; ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'middle'; ctx.fillText(g === 0 ? '×' : '=', gx + gw + gap / 2, at.y + gw / 2); ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`; ctx.textBaseline = 'bottom'; }
    });
    ctx.fillStyle = c.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`${fmtSigned(pix[big], 2)} × ${fmtSigned(wts[big], 3)} = ${fmtSigned(prods[big], 3)}`, at.x, at.y + gw + 6);
    ctx.fillStyle = c.ink3;
    ctx.fillText(`4×4 of the ${S}×${S} cells, rows ${r0}–${r0 + 3}`, at.x, at.y + gw + 19);
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
    const ph = les.phase, plan = sweepPlan(net, m.mode, { prep: ph !== 'check' }), nL = net.hidden.length, hops = plan.hops;
    const prepped = m.mode === 'pixels' && !net.conv && ph === 'check'; // the check replays the sweep with the preprocessing already done
    const banner = (text, dir) => drawBanner(ctx, c, L, text, dir);
    ctx.font = `600 11px "IBM Plex Sans", system-ui, sans-serif`; ctx.fillStyle = truthCol; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(`truth: ${les.truthName}`, out.x, out.y + out.r + 42);

    // 1 · forward pass (and 6 · check): values flow hop by hop
    const s0 = 1 - plan.segs[plan.segs.length - 1].ms / plan.total;
    if (prepped) drawPrepDone(ctx, L, c, m);
    if (ph === 'forward' || ph === 'check') {
      drawForwardSweep(ctx, L, c, m, les.frac, plan);
      banner(ph === 'forward' ? 'forward pass: each connection carries weight × value' : 'forward pass again, with the new weights', 1);
      if (ph === 'forward') return;
    } else if (ph === 'loss') drawForwardSweep(ctx, L, c, m, 1, plan);

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
    ctx.fillText(`error ${fmtSigned(les.error, 2)} ${up ? '↑' : '↓'}`, out.x, out.y + out.r + 58);
    ctx.fillStyle = c.ink3; ctx.fillText(`loss ${les.loss.toFixed(2)}`, out.x, out.y + out.r + 74);
    if (ph === 'check' && les.pAfter != null && les.reveal >= hops) { const dp = les.pBefore.toFixed(2) === les.pAfter.toFixed(2) ? 3 : 2; ctx.fillStyle = c.ink2; ctx.fillText(`${les.pBefore.toFixed(dp)} → ${les.pAfter.toFixed(dp)}`, out.x, out.y + out.r + 90); }
    if (ph === 'loss') { banner('loss: how wrong was the call?', 0); return; }

    // 3 · backward pass: blame flows back one layer at a time; every unit then shows its share, a unit that was off gets none
    if (nL) {
      const prog = ph === 'blame' ? les.frac * nL : nL;
      for (let k = 0; k < nL; k++) {
        const l = nL - 1 - k, t = Math.min(1, prog - k);
        if (t <= 0) continue;
        const key = l === nL - 1 ? 'out' : l + 1;  // the connections that leave layer l
        if (t < 1) {
          ctx.fillStyle = c.accent;
          for (const e of L.edges) { if (e.layer !== key) continue; const x = e.x2 + (e.x1 - e.x2) * t, y = e.y2 + (e.y1 - e.y2) * t; ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill(); }
        } else {
          L.unitColumns[l].forEach((u, j) => {
            const d = les.delta[l][j];
            const px = u.kind === 'square' ? L.badgeX : u.x, py = u.kind === 'square' ? u.y + 20 : u.y + u.r + 7;
            pill(ctx, px, py, d === 0 ? 'no blame' : `blame ${fmtSigned(d, 2)}`, d === 0 ? c.ink3 : (d > 0 ? c.regular : c.irregular), c);
          });
        }
      }
      if (ph === 'blame') { banner('backward pass: blame', -1); return; }
    }
    if (ph === 'check') return;

    // 4 · gradients: every weight's gradient = blame at its end × activity at its start. Connections glow in the colour of
    // the step the weight will take (orange up, blue down); on pixels a scaled copy of the image slides into each weight map.
    // 5 · update: the weights themselves move (the app morphs them); the glows and copies fade out.
    const fr = Math.min(1, les.frac);
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
      if (m.mode === 'pixels' && m.x && !net.conv) {
        const img = L.nodes.find(n => n.kind === 'image');
        const targets = nL ? L.unitColumns[0].map((u, j) => ({ node: u, scalar: -les.lr * les.delta[0][j] })) : [{ node: L.nodes.find(n => n.kind === 'map'), scalar: -les.lr * les.error }];
        const D = m.x.length;
        let common = 1e-9, xmax = 1e-9;
        for (const t of targets) common = Math.max(common, Math.abs(t.scalar));
        for (let i = 0; i < D; i++) xmax = Math.max(xmax, Math.abs(m.x[i]));
        ctx.globalAlpha = fade;
        targets.forEach((t, j) => {
          const n = t.node; if (!n || !img) return;
          const size = nL ? Math.max(20, Math.min(44, n.size - 12)) : 64;
          const x0 = img.x + img.w / 2 + 10 + size / 2, x1 = n.x - n.size / 2 - 22 - size / 2; // lands short of the × glyph
          if (t.scalar === 0) { // nothing arrives: an empty slot where the gradient tile would land, in the caption style of the other rows
            ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink3; ctx.textAlign = 'center';
            ctx.setLineDash([3, 3]); ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.strokeRect(x1 - size / 2 + 0.5, n.y - size / 2 + 0.5, size - 1, size - 1); ctx.setLineDash([]);
            ctx.textBaseline = 'middle'; ctx.fillText(nL ? 'off' : 'error 0', x1, n.y);
            if (size >= 40) { ctx.textBaseline = 'top'; ctx.fillText('no change', x1, n.y + size / 2 + 3); }
            return;
          }
          const cx = ph === 'update' ? x1 : x0 + (x1 - x0) * ease(fr);
          const arr = new Float64Array(D); for (let i = 0; i < D; i++) arr[i] = t.scalar * m.x[i];
          const tile = tileCanvas('lesson' + j, arr, 0, m.size, m.size, { max: Math.max(common * xmax, 0.3 * (nL ? TILE_FLOOR.square : TILE_FLOOR.map)) });
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(tile.canvas, cx - size / 2, n.y - size / 2, size, size);
          ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.strokeRect(cx - size / 2, n.y - size / 2, size, size);
          if (size >= 40) { ctx.font = `500 9px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`${fmtSigned(t.scalar, Math.abs(t.scalar) < 0.001 ? 4 : 3)} × image`, cx, n.y + size / 2 + 3); }
        });
        ctx.globalAlpha = 1;
      }
    }
    if (ph === 'gradient') banner(nL ? 'gradient of every weight = blame at its end × activity at its start' : 'gradient of every weight = error × its input', 0);
    if (ph === 'update') banner(`update: every weight steps against its gradient, w ← w − ${les.lr} × gradient`, 0);
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

  // hit-test in logical coordinates; returns { kind, ref, text } or null
  function hitNetwork(canvas, px, py, m) {
    const L = canvas._layout; if (!L) return null;
    const x = px * NET_W / canvas.clientWidth, y = py * NET_W / canvas.clientWidth;
    const net = m.net, fw = m.fw;
    const inBox = (n, pad) => Math.abs(x - n.x) <= n.size / 2 + pad && Math.abs(y - n.y) <= n.size / 2 + pad;
    for (const n of L.nodes) {
      if (n.kind === 'output' && Math.hypot(x - n.x, y - n.y) <= n.r + 4) return { kind: 'node', ref: n, text: fw ? `score z = ${fmtSigned(fw.z, 2)} → P(${m.positiveName}) = ${fw.p.toFixed(3)} · bias ${fmtSigned(net.bo, 3)}` : `output · bias ${fmtSigned(net.bo, 3)}` };
      if (n.kind === 'unit' && Math.hypot(x - n.x, y - n.y) <= n.r + 4) {
        const a = fw ? ` · activation ${fmtNum(fw.a[n.l + 1][n.j], 3)}` : '';
        const out = n.l === net.hidden.length - 1 ? ` · to output ${fmtSigned(net.Wo[n.j], 3)}` : '';
        return { kind: 'node', ref: n, text: `hidden unit ${net.hidden.length > 1 ? (n.l + 1) + '.' : ''}${n.j + 1}${a} · bias ${fmtSigned(net.b[n.l][n.j], 3)}${out}` };
      }
      if (n.kind === 'input' && Math.hypot(x - n.x, y - n.y) <= n.r + 4) return { kind: 'node', ref: n, text: `${m.featureNames[n.i]}${m.x ? ` · standardized value ${fmtSigned(m.x[n.i], 2)}` : ''}` };
      if (n.kind === 'square' && inBox(n, 2)) {
        let mm = 0; for (let i = 0; i < net.sizes[0]; i++) mm = Math.max(mm, Math.abs(net.W[0][n.j * net.sizes[0] + i]));
        const i = Math.floor((x - (n.x - n.size / 2)) / n.size * m.size), j = Math.floor((y - (n.y - n.size / 2)) / n.size * m.size);
        const w = net.W[0][n.j * net.sizes[0] + Math.max(0, Math.min(net.D - 1, j * m.size + i))];
        const lessonTxt = m.lesson && (m.lesson.phase === 'gradient' || m.lesson.phase === 'update' || m.lesson.phase === 'check') ? ` · this lesson adds ${fmtSigned(-m.lesson.lr * m.lesson.delta[0][n.j], 3)} × image to this map` : '';
        return { kind: 'node', ref: n, text: `hidden unit ${n.j + 1} weights: max |w| ${mm.toFixed(3)} · pixel (${i}, ${j}) weight ${fmtSigned(w, 3)}${lessonTxt}` };
      }
      if (n.kind === 'uprod' && (inBox(n, 2) || Math.hypot(x - n.x - n.size / 2 - 16, y - n.y) <= 13)) {
        const D = net.D, off = n.j * D;
        let sum = 0; if (m.x) for (let i = 0; i < D; i++) sum += net.W[0][off + i] * m.x[i];
        const a = fw ? fw.a[1][n.j] : null;
        const out = net.hidden.length === 1 ? ` · to output ${fmtSigned(net.Wo[n.j], 3)}` : '';
        const pix = inBox(n, 0) && m.x ? (() => { const i = Math.floor((x - (n.x - n.size / 2)) / n.size * m.size), j = Math.floor((y - (n.y - n.size / 2)) / n.size * m.size), k = j * m.size + i; return ` · pixel (${i}, ${j}): ${fmtSigned(net.W[0][off + k], 3)} × ${fmtSigned(m.x[k], 2)} = ${fmtSigned(net.W[0][off + k] * m.x[k], 3)}`; })() : '';
        return { kind: 'node', ref: n, text: `unit ${n.j + 1}: Σ weight × pixel ${m.x ? fmtSigned(sum, 2) : '?'} + bias ${fmtSigned(net.b[0][n.j], 2)}${a != null ? ` → ${m.activationLabel} → ${fmtNum(a, 2)}` : ''}${out}${pix}` };
      }
      if (n.kind === 'map' && inBox(n, 0)) {
        const i = Math.min(m.size - 1, Math.floor((x - (n.x - n.size / 2)) / n.size * m.size)), j = Math.min(m.size - 1, Math.floor((y - (n.y - n.size / 2)) / n.size * m.size));
        const w = net.Wo[j * m.size + i];
        const lessonTxt = m.lesson && (m.lesson.phase === 'gradient' || m.lesson.phase === 'update' || m.lesson.phase === 'check') ? ` · this lesson adds ${fmtSigned(-m.lesson.lr * m.lesson.error, 3)} × image to the map` : '';
        return { kind: 'node', ref: n, text: `pixel (${i}, ${j}) weight ${fmtSigned(w, 3)}${m.x ? ` × input ${fmtSigned(m.x[j * m.size + i], 2)}` : ''}${lessonTxt}` };
      }
      if (n.kind === 'product' && inBox(n, 0) && m.x) {
        const i = Math.min(m.size - 1, Math.floor((x - (n.x - n.size / 2)) / n.size * m.size)), j = Math.min(m.size - 1, Math.floor((y - (n.y - n.size / 2)) / n.size * m.size));
        const k = j * m.size + i;
        return { kind: 'node', ref: n, text: `pixel (${i}, ${j}): weight ${fmtSigned(net.Wo[k], 3)} × input ${fmtSigned(m.x[k], 2)} = ${fmtSigned(net.Wo[k] * m.x[k], 3)}` };
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
    const ordered = pts.slice().sort((a, b) => (a.selected ? 1 : 0) - (b.selected ? 1 : 0));
    for (const p of ordered) {
      const cls = `pt ${p.cls}${p.split === 'test' ? ' test' : ''}${p.selected ? ' selected' : ''}`;
      g += `<circle class="${cls}" data-id="${p.id}" r="${p.selected ? 6 : 4.5}" cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}"><title>${esc(p.name)} · ${esc(opt.xLabel)} ${p.x.toFixed(3)} · ${esc(opt.yLabel)} ${p.y.toFixed(3)}${p.clsName ? ' · ' + p.clsName : ''}</title></circle>`;
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

  return { refreshTheme, colors, diverging, divergingRgb, sequential, unitColor, renderThumb, renderFingerprint, renderBigImage, renderEvidence, renderMeasurement, drawNetwork, hitNetwork, drawCurves, drawScatter, sweepPlan, sweepState, pixelRgb, fmtSigned, fmtNum, NET_W, NET_H };
})();
