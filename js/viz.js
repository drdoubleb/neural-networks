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
  // weight map: size*size weights drawn with the diverging scale, normalised to max |w|
  const mapCache = new Map();
  function weightMapCanvas(key, w, size, offset) {
    let cv = mapCache.get(key);
    if (!cv) { cv = document.createElement('canvas'); cv.width = size; cv.height = size; mapCache.set(key, cv); }
    const ctx = cv.getContext('2d');
    const im = ctx.createImageData(size, size);
    let max = 1e-9;
    for (let i = 0; i < size * size; i++) max = Math.max(max, Math.abs(w[offset + i]));
    for (let i = 0; i < size * size; i++) {
      const c = divergingRgb(w[offset + i] / max);
      im.data[i * 4] = c[0]; im.data[i * 4 + 1] = c[1]; im.data[i * 4 + 2] = c[2]; im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return { canvas: cv, max };
  }

  // ------------------------------------------------------------------ network diagram
  const NET_W = 760, NET_H = 440;
  function fmtSigned(v, d) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d); }
  function fmtNum(v, d) { return v < 0 ? '−' + Math.abs(v).toFixed(d) : v.toFixed(d); }

  function layoutNetwork(m) {
    const net = m.net, D = net.D, Hn = net.H;
    const L = { W: NET_W, H: NET_H, inputs: [], hidden: [], edges: [], bands: [], singleMap: null, inputBlock: null, mode: m.mode };
    const xOut = 668, xHid = 430;
    L.output = { x: xOut, y: NET_H / 2, r: 26 };
    if (m.mode === 'features') {
      const xIn = 150;
      const sp = Math.min(60, (NET_H - 80) / Math.max(1, D - 1));
      for (let i = 0; i < D; i++) L.inputs.push({ kind: 'input', x: xIn, y: NET_H / 2 + (i - (D - 1) / 2) * sp, r: 16, i });
      if (Hn > 0) {
        const hs = Math.min(64, (NET_H - 80) / Math.max(1, Hn - 1));
        for (let j = 0; j < Hn; j++) L.hidden.push({ kind: 'hidden', x: xHid, y: NET_H / 2 + (j - (Hn - 1) / 2) * hs, r: 18, j });
        for (let j = 0; j < Hn; j++) for (let i = 0; i < D; i++) {
          const a = L.inputs[i], b = L.hidden[j];
          L.edges.push({ x1: a.x + a.r, y1: a.y, x2: b.x - b.r, y2: b.y, w: net.W1[j * D + i], layer: 1, i, j });
        }
        for (let j = 0; j < Hn; j++) { const b = L.hidden[j]; L.edges.push({ x1: b.x + b.r, y1: b.y, x2: L.output.x - L.output.r, y2: L.output.y, w: net.W2[j], layer: 2, j }); }
      } else {
        for (let i = 0; i < D; i++) { const a = L.inputs[i]; L.edges.push({ x1: a.x + a.r, y1: a.y, x2: L.output.x - L.output.r, y2: L.output.y, w: net.W2[i], layer: 2, i }); }
      }
    } else {
      L.inputBlock = { x: 46, y: NET_H / 2 - 62, w: 124, h: 124 };
      if (Hn > 0) {
        const s = Math.min(66, (NET_H - 50) / Hn - 8);
        const step = s + 8;
        for (let j = 0; j < Hn; j++) {
          const y = NET_H / 2 + (j - (Hn - 1) / 2) * step;
          L.hidden.push({ kind: 'hidden', x: xHid, y, size: s, r: s / 2, j });
          const b = L.inputBlock;
          L.bands.push({ pts: [[b.x + b.w, b.y + 6], [xHid - s / 2, y - s / 2], [xHid - s / 2, y + s / 2], [b.x + b.w, b.y + b.h - 6]], j });
          L.edges.push({ x1: xHid + s / 2 + 12, y1: y, x2: L.output.x - L.output.r, y2: L.output.y, w: net.W2[j], layer: 2, j });
        }
      } else {
        const s = 132;
        L.singleMap = { kind: 'map', x: xHid, y: NET_H / 2, size: s, r: s / 2 };
        const b = L.inputBlock;
        L.bands.push({ pts: [[b.x + b.w, b.y + 4], [xHid - s / 2, NET_H / 2 - s / 2], [xHid - s / 2, NET_H / 2 + s / 2], [b.x + b.w, b.y + b.h - 4]], j: -1 });
        L.edges.push({ x1: xHid + s / 2, y1: NET_H / 2, x2: L.output.x - L.output.r, y2: L.output.y, w: 1, layer: 0 });
      }
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

  function nodeFillText(ctx, x, y, r, fill, text, font) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = colors().lineStrong; ctx.stroke();
    if (text != null) {
      const rgb = typeof fill === 'string' && fill.startsWith('rgb') ? hexToRgb(fill) : colors().rgb.surface;
      ctx.fillStyle = luminance(rgb) < 0.5 ? '#ffffff' : colors().ink;
      ctx.font = font || `500 11px "IBM Plex Mono", ui-monospace, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y + 0.5);
    }
  }

  /*
   * m = { net, mode, x (standardized input or null), fw (forward result or null), featureNames, specimen, size, tint,
   *       stage (0 input only, 1 + hidden, 2 + output; default 2), hover {x,y} in logical coords or null, activationLabel }
   */
  function drawNetwork(canvas, m) {
    const ctx = fitCanvas(canvas, NET_W, NET_H);
    const c = colors();
    const L = layoutNetwork(m);
    canvas._layout = L;
    const stage = m.stage == null ? 2 : m.stage;
    const net = m.net, fw = m.fw;
    ctx.clearRect(0, 0, NET_W, NET_H);
    ctx.fillStyle = c.surface; ctx.fillRect(0, 0, NET_W, NET_H);
    const labelFont = `500 12px "IBM Plex Sans", system-ui, sans-serif`;
    const monoFont = `500 11px "IBM Plex Mono", ui-monospace, monospace`;
    const capFont = `600 12px "IBM Plex Sans", system-ui, sans-serif`;

    // layer captions
    ctx.font = capFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const inX = m.mode === 'features' ? 150 : L.inputBlock.x + L.inputBlock.w / 2;
    ctx.fillText(m.mode === 'features' ? 'INPUT · 6 MEASUREMENTS' : 'INPUT · 1,024 PIXELS', inX, 10);
    if (net.H > 0) ctx.fillText(`HIDDEN · ${net.H} UNIT${net.H > 1 ? 'S' : ''} · ${(m.activationLabel || '').toUpperCase()}`, 430, 10);
    else if (m.mode === 'pixels') ctx.fillText('WEIGHTS · ONE PER PIXEL', 430, 10);
    else ctx.fillText('NO HIDDEN LAYER', 430, 10);
    ctx.fillText('OUTPUT', L.output.x, 10);

    // bands (pixel mode)
    for (const b of L.bands) {
      ctx.beginPath(); b.pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
      ctx.fillStyle = rgbStr(c.rgb.accent, 0.07); ctx.fill();
      ctx.strokeStyle = rgbStr(c.rgb.accent, 0.35); ctx.lineWidth = 1; ctx.stroke();
    }
    if (L.bands.length) {
      ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const b0 = L.bands[Math.floor(L.bands.length / 2)];
      const mx = (b0.pts[0][0] + b0.pts[1][0]) / 2;
      ctx.fillText(net.H > 0 ? '1,024 weights per unit · each map scaled to its own max |w|' : '1,024 weights', mx, NET_H - 22);
    }

    // edges, weak first so strong ones sit on top
    const maxAbs = {};
    for (const e of L.edges) maxAbs[e.layer] = Math.max(maxAbs[e.layer] || 1e-9, Math.abs(e.w));
    const edges = L.edges.slice().sort((a, b) => Math.abs(a.w) / maxAbs[a.layer] - Math.abs(b.w) / maxAbs[b.layer]);
    for (const e of edges) {
      const rel = Math.abs(e.w) / maxAbs[e.layer];
      const hovered = m.hover && m.hover.kind === 'edge' && m.hover.ref === e;
      if (e.layer === 0) { ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 2; }
      else {
        const rgb = e.w < 0 ? c.rgb.regular : c.rgb.irregular;
        ctx.strokeStyle = rgbStr(rgb, hovered ? 1 : 0.18 + 0.82 * rel);
        ctx.lineWidth = (0.6 + 5 * Math.pow(rel, 0.9)) * (hovered ? 1.4 : 1);
      }
      ctx.beginPath(); ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2); ctx.stroke();
    }
    if (L.singleMap) { // the sum symbol on the map -> output edge
      const e = L.edges[0]; ctx.font = `500 13px "IBM Plex Mono", ui-monospace, monospace`; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText('Σ weight × pixel', (e.x1 + e.x2) / 2, e.y1 - 6);
    }

    // input layer
    const pending = c.surface2;
    if (m.mode === 'features') {
      for (const n of L.inputs) {
        const z = m.x ? m.x[n.i] : null;
        const fill = z == null ? pending : diverging(clamp(z / 2.5, -1, 1));
        nodeFillText(ctx, n.x, n.y, n.r, fill, z == null ? '·' : fmtSigned(z, 1));
        ctx.font = labelFont; ctx.fillStyle = c.ink2; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(m.featureNames[n.i], n.x - n.r - 10, n.y);
      }
    } else {
      const b = L.inputBlock;
      ctx.fillStyle = c.surface2; ctx.fillRect(b.x - 4, b.y - 4, b.w + 8, b.h + 8);
      if (m.specimen) {
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(imageToCanvas(m.specimen.px, m.size, m.tint), b.x, b.y, b.w, b.h);
      } else {
        ctx.font = labelFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('select a nucleus', b.x + b.w / 2, b.y + b.h / 2);
      }
      ctx.strokeStyle = c.lineStrong; ctx.lineWidth = 1; ctx.strokeRect(b.x - 4, b.y - 4, b.w + 8, b.h + 8);
      ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(m.specimen ? `${m.size}×${m.size} ink values, mean removed` : '', b.x + b.w / 2, b.y + b.h + 12);
    }

    // hidden layer
    const hoveredNode = m.hover && m.hover.kind === 'node' ? m.hover.ref : null;
    if (m.mode === 'features') {
      for (const n of L.hidden) {
        const a = fw && stage >= 1 ? fw.h[n.j] : null;
        let fill = pending, txt = '·';
        if (a != null) {
          const signed = m.activation === 'tanh';
          const hmax = Math.max(1, ...Array.from(fw.h).map(Math.abs));
          fill = signed ? diverging(a) : rgbStr(sequentialRgb(Math.abs(a) / hmax));
          txt = fmtNum(a, 2);
        }
        nodeFillText(ctx, n.x, n.y, n.r, fill, txt);
        if (hoveredNode === n) { ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 3, 0, Math.PI * 2); ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.stroke(); }
        ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(`b ${fmtSigned(net.b1[n.j], 2)}`, n.x, n.y + n.r + 4);
      }
    } else {
      const squares = L.singleMap ? [L.singleMap] : L.hidden;
      for (const n of squares) {
        const isMap = n.kind === 'map';
        const { canvas: mc, max } = weightMapCanvas(isMap ? 'out' : 'h' + n.j, isMap ? net.W2 : net.W1, m.size, isMap ? 0 : n.j * net.D);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(mc, n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        ctx.strokeStyle = hoveredNode === n ? c.ink : c.lineStrong; ctx.lineWidth = hoveredNode === n ? 2 : 1;
        ctx.strokeRect(n.x - n.size / 2, n.y - n.size / 2, n.size, n.size);
        if (isMap || n.size >= 56) { ctx.font = monoFont; ctx.fillStyle = c.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`±${max.toFixed(2)}`, n.x, n.y + n.size / 2 + 3); }
        if (!isMap) {
          const a = fw && stage >= 1 ? fw.h[n.j] : null;
          const hmax = a != null ? Math.max(1, ...Array.from(fw.h).map(Math.abs)) : 1;
          const fill = a == null ? pending : (m.activation === 'tanh' ? diverging(a) : rgbStr(sequentialRgb(Math.abs(a) / hmax)));
          nodeFillText(ctx, n.x + n.size / 2 + 2, n.y, 11, fill, null);
          if (a != null && n.size >= 40) {
            ctx.font = `500 10px "IBM Plex Mono", ui-monospace, monospace`;
            ctx.fillStyle = luminance(hexToRgb(fill)) < 0.5 ? '#fff' : c.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(Math.abs(a) >= 10 ? a.toFixed(0) : a.toFixed(1), n.x + n.size / 2 + 2, n.y + 0.5);
          }
        }
      }
    }

    // output
    const o = L.output;
    const p = fw && stage >= 2 ? fw.p : null;
    const ofill = p == null ? pending : diverging((p - 0.5) * 2);
    nodeFillText(ctx, o.x, o.y, o.r, ofill, p == null ? '?' : p.toFixed(2), `600 14px "IBM Plex Mono", ui-monospace, monospace`);
    if (hoveredNode === o) { ctx.beginPath(); ctx.arc(o.x, o.y, o.r + 3, 0, Math.PI * 2); ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.stroke(); }
    ctx.font = labelFont; ctx.fillStyle = c.ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('P(irregular)', o.x, o.y + o.r + 6);
    ctx.font = monoFont; ctx.fillStyle = c.ink3;
    ctx.fillText(`b ${fmtSigned(net.b2, 2)}`, o.x, o.y + o.r + 24);
    if (p != null) {
      const call = p >= 0.5 ? 'IRREGULAR' : 'REGULAR';
      ctx.font = `700 13px "Bricolage Grotesque", "IBM Plex Sans", system-ui, sans-serif`;
      ctx.fillStyle = p >= 0.5 ? c.irregular : c.regular; ctx.textBaseline = 'bottom';
      ctx.fillText(call, o.x, o.y - o.r - 8);
    }
    return L;
  }

  // hit-test in logical coordinates; returns {kind:'node'|'edge', ref, text} or null
  function hitNetwork(canvas, px, py, m) {
    const L = canvas._layout; if (!L) return null;
    const x = px * NET_W / canvas.clientWidth, y = py * NET_W / canvas.clientWidth;
    const net = m.net, fw = m.fw;
    const near = (n, r) => Math.hypot(x - n.x, y - n.y) <= r;
    if (near(L.output, L.output.r + 4)) return { kind: 'node', ref: L.output, text: fw ? `score z = ${fmtSigned(fw.z, 2)} → P(irregular) = ${fw.p.toFixed(3)} · bias ${fmtSigned(net.b2, 3)}` : `output · bias ${fmtSigned(net.b2, 3)}` };
    for (const n of L.hidden) {
      const r = n.size ? n.size / 2 + 14 : n.r + 4;
      if (n.size ? (Math.abs(x - n.x - 6) <= n.size / 2 + 8 && Math.abs(y - n.y) <= n.size / 2 + 2) : near(n, r)) {
        const a = fw ? ` · activation ${fmtNum(fw.h[n.j], 3)}` : '';
        const mx = n.size ? (() => { let mm = 0; for (let i = 0; i < net.D; i++) mm = Math.max(mm, Math.abs(net.W1[n.j * net.D + i])); return ` · max |w| ${mm.toFixed(3)}`; })() : '';
      return { kind: 'node', ref: n, text: `hidden unit ${n.j + 1}${a}${mx} · bias ${fmtSigned(net.b1[n.j], 3)} · to output ${fmtSigned(net.W2[n.j], 3)}` };
      }
    }
    if (L.singleMap && Math.abs(x - L.singleMap.x) <= L.singleMap.size / 2 && Math.abs(y - L.singleMap.y) <= L.singleMap.size / 2) {
      const i = Math.floor((x - (L.singleMap.x - L.singleMap.size / 2)) / L.singleMap.size * m.size), j = Math.floor((y - (L.singleMap.y - L.singleMap.size / 2)) / L.singleMap.size * m.size);
      const w = net.W2[j * m.size + i];
      return { kind: 'node', ref: L.singleMap, text: `pixel (${i}, ${j}) weight ${fmtSigned(w, 3)}${m.x ? ` × input ${fmtSigned(m.x[j * m.size + i], 2)}` : ''}` };
    }
    for (const n of L.inputs) if (near(n, n.r + 4)) {
      return { kind: 'node', ref: n, text: `${m.featureNames[n.i]}${m.x ? ` · standardized value ${fmtSigned(m.x[n.i], 2)}` : ''}` };
    }
    let best = null, bestD = 6;
    for (const e of L.edges) {
      if (e.layer === 0) continue;
      const dx = e.x2 - e.x1, dy = e.y2 - e.y1, len2 = dx * dx + dy * dy;
      const t = clamp(((x - e.x1) * dx + (y - e.y1) * dy) / len2, 0, 1);
      const d = Math.hypot(x - (e.x1 + t * dx), y - (e.y1 + t * dy));
      if (d < bestD) { bestD = d; best = e; }
    }
    if (best) {
      const from = best.layer === 1 ? m.featureNames[best.i] : (best.j != null ? `unit ${best.j + 1}` : m.featureNames[best.i]);
      const to = best.layer === 1 ? `unit ${best.j + 1}` : 'output';
      return { kind: 'edge', ref: best, text: `weight ${from} → ${to}: ${fmtSigned(best.w, 3)}` };
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
      g += `<circle class="${cls}" data-id="${p.id}" r="${p.selected ? 6 : 4.5}" cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}"><title>${esc(p.name)} · ${esc(opt.xLabel)} ${p.x.toFixed(3)} · ${esc(opt.yLabel)} ${p.y.toFixed(3)}${p.cls !== 'unknown' ? ' · ' + p.cls : ''}</title></circle>`;
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

  return { refreshTheme, colors, diverging, divergingRgb, renderThumb, renderBigImage, renderEvidence, renderMeasurement, drawNetwork, hitNetwork, drawCurves, drawScatter, pixelRgb, fmtSigned, fmtNum, NET_W, NET_H };
})();
