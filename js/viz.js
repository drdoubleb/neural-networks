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
        else if (n.kind === 'product' && m.x && stage >= 1) { const prod = new Float64Array(net.D); let sum = 0; for (let i = 0; i < net.D; i++) { prod[i] = net.Wo[i] * m.x[i]; sum += prod[i]; } tile = tileCanvas('prod', prod, 0, m.size, m.size); tile.sum = sum; }
        else if (n.kind === 'uprod' && m.x && stage >= 1) { const D = net.D, off = n.j * D, prod = new Float64Array(D); for (let i = 0; i < D; i++) prod[i] = net.W[0][off + i] * m.x[i]; tile = tileCanvas('uprod' + n.j, prod, 0, m.size, m.size); }
        ctx.imageSmoothingEnabled = false;
        if (tile) ctx.drawImage(tile.canvas, n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        else { ctx.fillStyle = pending; ctx.fillRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size); }
        ctx.strokeStyle = isHov ? c.ink : c.lineStrong; ctx.lineWidth = isHov ? 2 : 1;
        ctx.strokeRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        if (n.kind === 'map' || (n.kind === 'square' && n.captioned)) ctx.fillText(`±${tile.max.toFixed(2)}`, n.x, n.y + n.size / 2 + 3);
        if (n.kind === 'product') ctx.fillText(tile ? `Σ = ${fmtSigned(tile.sum, 2)}  (bias ${fmtSigned(net.bo, 2)})` : 'select a nucleus', n.x, n.y + n.size / 2 + 3);
        if (n.kind === 'uprod') { // activation badge: sum of the products plus the bias, through the activation
          const a = fw && stage >= 1 ? fw.a[1][n.j] : null;
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
        const acts = fw && stage >= 1 ? fw.a[n.l + 1] : null;
        const a = acts ? acts[n.j] : null;
        circleNode(ctx, n.x, n.y, n.r, unitFill(a, acts || [], signed), a == null ? '·' : fmtNum(a, 2), null, isHov);
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        if (L.unitColumns[n.l].length <= 8 && !(m.lesson && m.lesson.phase !== 'forward' && m.lesson.phase !== 'error')) ctx.fillText(`b ${fmtSigned(net.b[n.l][n.j], 2)}`, n.x, n.y + n.r + 4);
      } else if (n.kind === 'output') {
        const p = fw && stage >= 2 ? fw.p : null;
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

  // The overlay uses the frozen forward pass and gradients. The ordinary graph still shows weights/activations.
  // Only update glows encode parameter changes; backprop arrows encode weighted downstream gradients.
  function drawLesson(ctx, L, m) {
    const les = m.lesson, c = colors(), net = m.net, out = L.output;
    const backward = les.phase === 'blame', updating = les.phase === 'nudge';
    const after = les.phase === 'after', phaseIndex = les.index;
    const nL = net.hidden.length;
    const arrow = (e, reverse, alpha, moving) => {
      const ax = reverse ? e.x2 : e.x1, ay = reverse ? e.y2 : e.y1;
      const bx = reverse ? e.x1 : e.x2, by = reverse ? e.y1 : e.y2;
      const angle = Math.atan2(by - ay, bx - ax), length = Math.hypot(bx - ax, by - ay);
      if (length < 1) return;
      ctx.strokeStyle = rgbStr(c.rgb.accent, alpha); ctx.fillStyle = rgbStr(c.rgb.accent, alpha); ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      const tx = ax + (bx - ax) * 0.8, ty = ay + (by - ay) * 0.8;
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx - 9 * Math.cos(angle - 0.45), ty - 9 * Math.sin(angle - 0.45)); ctx.lineTo(tx - 9 * Math.cos(angle + 0.45), ty - 9 * Math.sin(angle + 0.45)); ctx.closePath(); ctx.fill();
      if (moving && les.playing && les.frac < 1) { const t = (les.frac * 2) % 1; ctx.beginPath(); ctx.arc(ax + (bx - ax) * t, ay + (by - ay) * t, 4, 0, Math.PI * 2); ctx.fill(); }
    };
    if (les.phase === 'forward') {
      for (const e of L.edges) arrow(e, false, 0.45, true);
    }
    if (backward) {
      const key = les.layer === nL - 1 ? 'out' : les.layer + 1;
      const paths = L.edges.filter(e => e.layer === key);
      const contribution = e => e.w * (key === 'out' ? les.error : les.delta[key][e.ti]);
      const max = Math.max(1e-12, ...paths.map(e => Math.abs(contribution(e))));
      for (const e of paths) { const rel = Math.abs(contribution(e)) / max; arrow(e, true, rel ? 0.35 + 0.65 * rel : 0.12, rel > 0); }
    }
    if (updating) {
      const change = e => {
        if (e.wi == null || e.layer === 'sum') return 0;
        const w = e.layer === 'out' ? les.before.Wo[e.wi] : les.before.W[e.layer][e.wi];
        const g = e.layer === 'out' ? les.gWo[e.wi] : les.gW[e.layer][e.wi];
        return -les.lr * (g + les.l2 * w);
      };
      const max = Math.max(1e-12, ...L.edges.map(e => Math.abs(change(e))));
      for (const e of L.edges) {
        const dw = change(e); if (dw === 0) continue;
        ctx.strokeStyle = rgbStr(dw > 0 ? c.rgb.irregular : c.rgb.regular, 0.75);
        ctx.lineWidth = 2 + 6 * Math.abs(dw) / max;
        ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke();
      }
    }
    // Ring the destination and selected incoming connection so table selection stays connected to the graph.
    const [ls, js] = les.target.split(':'), layer = ls === 'out' ? 'out' : +ls, j = +js;
    if (['blame', 'gradient', 'nudge'].includes(les.phase)) {
      const node = layer === 'out' ? out : L.unitColumns[layer][j];
      if (node) {
        ctx.strokeStyle = c.accent; ctx.lineWidth = 3;
        if (node.kind === 'square') ctx.strokeRect(node.x - node.size / 2 - 4, node.y - node.size / 2 - 4, node.size + 8, node.size + 8);
        else { ctx.beginPath(); ctx.arc(node.x, node.y, node.r + 5, 0, Math.PI * 2); ctx.stroke(); }
      }
      if (!backward) for (const e of L.edges) {
        if (e.layer === layer && e.fi === les.input && (layer === 'out' || e.ti === j)) {
          ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke(); ctx.setLineDash([]);
        }
      }
    }
    ctx.font = '600 11px "IBM Plex Sans", system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = c.ink2;
    ctx.fillText(`truth: ${les.truthName} (${les.y})`, out.x, out.y + out.r + 42);
    if (phaseIndex > 0 && !after) pill(ctx, out.x, out.y + out.r + 66, `δ = p − y = ${window.Backprop.signed(les.error)}`, c.accent, c);
    if (after) {
      ctx.fillText(`p: ${window.Backprop.num(les.pBefore)} → ${window.Backprop.num(les.pAfter)}`, out.x, out.y + out.r + 60);
      return;
    }
    if (backward || les.phase === 'gradient' || updating) {
      L.unitColumns.forEach((col, l) => {
        if (backward && l < les.layer) return;
        col.forEach((u, k) => {
          const d = les.delta[l][k], blocked = les.res.g.slopes[l][k] === 0;
          const px = u.kind === 'square' ? L.badgeX : u.x, py = u.kind === 'square' ? u.y + 20 : u.y + u.r + 9;
          pill(ctx, px, py, blocked ? 'slope 0 → δ 0' : `δ ${window.Backprop.signed(d)}`, blocked ? c.ink3 : c.accent, c);
        });
      });
    }
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
        const lessonTxt = m.lesson ? ' · click to inspect this pixel’s gradient and update below' : '';
        return { kind: 'node', ref: n, pixelIndex: Math.max(0, Math.min(net.D - 1, j * m.size + i)), text: `hidden unit ${n.j + 1} weights: max |w| ${mm.toFixed(3)} · pixel (${i}, ${j}) weight ${fmtSigned(w, 3)}${lessonTxt}` };
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
        const lessonTxt = m.lesson ? ' · click to inspect this pixel’s gradient and update below' : '';
        return { kind: 'node', ref: n, pixelIndex: j * m.size + i, text: `pixel (${i}, ${j}) weight ${fmtSigned(w, 3)}${m.x ? ` × input ${fmtSigned(m.x[j * m.size + i], 2)}` : ''}${lessonTxt}` };
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
      const contrib = best.fromIdx != null && m.x ? ` × input ${fmtSigned(m.x[best.fromIdx], 2)} = ${fmtSigned(best.w * m.x[best.fromIdx], 3)}` : '';
      const moved = best.prev != null && Math.abs(best.w - best.prev) > 1e-9 ? ` · last step ${fmtSigned(best.w - best.prev, 4)}` : '';
      let lessonTxt = '';
      const les = m.lesson;
      if (les && best.wi != null && (les.phase === 'nudge' || les.phase === 'after')) {
        const target = best.layer === 'out' ? 'out:0' : `${best.layer}:${best.ti}`;
        const step = window.Backprop.parameter(les.res, target, best.fi, les.lr, les.l2);
        lessonTxt = ` · saved update: data gradient ${window.Backprop.signed(step.gradient)} + decay ${window.Backprop.signed(step.decayGradient)}; Δw = ${window.Backprop.signed(step.change)}; ${window.Backprop.signed(step.before)} → ${window.Backprop.signed(step.after)}`;
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

  return { refreshTheme, colors, diverging, divergingRgb, sequential, unitColor, renderThumb, renderFingerprint, renderBigImage, renderEvidence, renderMeasurement, drawNetwork, hitNetwork, drawCurves, drawScatter, pixelRgb, fmtSigned, fmtNum, NET_W, NET_H };
})();
