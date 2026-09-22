/*
 * features.js — measures a 32x32 nucleus image the way a morphometry pipeline would.
 * Works in the browser (window.NucleusFeatures) and in Node (module.exports).
 *
 * All measurements are made from the pixels only; nothing from the generator leaks in.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NucleusFeatures = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FEATURES = [
    { key: 'area',       name: 'Area',              unit: 'px²', fmt: v => v.toFixed(0),
      desc: 'Pixels enclosed by the nuclear membrane. Bigger nuclei score higher.' },
    { key: 'elongation', name: 'Elongation',        unit: '',    fmt: v => v.toFixed(2),
      desc: 'Long axis ÷ short axis. 1.00 is round; 1.40 is a stretched oval.' },
    { key: 'darkness',   name: 'Darkness',          unit: '',    fmt: v => v.toFixed(2),
      desc: 'Mean ink deep inside the nucleus (hyperchromasia). 0 = white, 1 = black.' },
    { key: 'texture',    name: 'Texture',           unit: '',    fmt: v => v.toFixed(3),
      desc: 'Spread of ink deep inside the nucleus (chromatin clumping).' },
    { key: 'solidity',   name: 'Solidity',          unit: '',    fmt: v => v.toFixed(3),
      desc: 'Area ÷ area of the convex hull. 1.000 means no notches or clefts.' },
    { key: 'roughness',  name: 'Contour roughness', unit: '',    fmt: v => v.toFixed(3),
      desc: 'Perimeter ÷ perimeter of the smooth ellipse with the same area and elongation. 1.000 is perfectly smooth.' },
  ];

  function decodeBase64(b64) {
    if (typeof atob === 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
    const s = atob(b64);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  // ink = 1 - brightness, so that "more ink" = darker = more nucleus
  function toInk(px) {
    const ink = new Float32Array(px.length);
    for (let i = 0; i < px.length; i++) ink[i] = 1 - px[i] / 255;
    return ink;
  }

  function otsuThreshold(values) {
    const bins = 64, hist = new Float64Array(bins);
    for (let i = 0; i < values.length; i++) hist[Math.min(bins - 1, Math.floor(values[i] * bins))]++;
    const total = values.length;
    let sum = 0;
    for (let b = 0; b < bins; b++) sum += b * hist[b];
    let sumB = 0, wB = 0, best = 0, thr = bins / 2;
    for (let b = 0; b < bins; b++) {
      wB += hist[b]; if (wB === 0) continue;
      const wF = total - wB; if (wF === 0) break;
      sumB += b * hist[b];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; thr = b + 1; }
    }
    return thr / bins;
  }

  function largestComponent(mask, w, h) {
    const lab = new Int32Array(w * h).fill(-1);
    let best = -1, bestSize = 0, n = 0;
    const stack = [];
    for (let s = 0; s < w * h; s++) {
      if (!mask[s] || lab[s] >= 0) continue;
      let size = 0; stack.push(s); lab[s] = n;
      while (stack.length) {
        const i = stack.pop(); size++;
        const x = i % w, y = (i / w) | 0;
        const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
        for (const j of nb) if (j >= 0 && mask[j] && lab[j] < 0) { lab[j] = n; stack.push(j); }
      }
      if (size > bestSize) { bestSize = size; best = n; }
      n++;
    }
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = lab[i] === best ? 1 : 0;
    return out;
  }

  function fillHoles(comp, w, h) {
    const outside = new Uint8Array(w * h);
    const stack = [];
    for (let x = 0; x < w; x++) stack.push(x, (h - 1) * w + x);
    for (let y = 0; y < h; y++) stack.push(y * w, y * w + w - 1);
    while (stack.length) {
      const i = stack.pop();
      if (comp[i] || outside[i]) continue;
      outside[i] = 1;
      const x = i % w, y = (i / w) | 0;
      if (x > 0) stack.push(i - 1); if (x < w - 1) stack.push(i + 1);
      if (y > 0) stack.push(i - w); if (y < h - 1) stack.push(i + w);
    }
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = outside[i] ? 0 : 1;
    return out;
  }

  function erode(mask, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let all = true;
      for (let dy = -1; dy <= 1 && all; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (!(xx >= 0 && yy >= 0 && xx < w && yy < h && mask[yy * w + xx])) { all = false; break; }
      }
      out[y * w + x] = all ? 1 : 0;
    }
    return out;
  }

  // Marching squares on field f at level lvl. Segments are oriented consistently (the "inside" is always on
  // the same side), so the shoelace formula over all segments gives the enclosed sub-pixel area.
  const MS_CASES = [[], [[3, 2]], [[2, 1]], [[3, 1]], [[0, 1]], [[0, 3], [2, 1]], [[0, 2]], [[0, 3]],
                    [[0, 3]], [[0, 2]], [[0, 1], [3, 2]], [[0, 1]], [[3, 1]], [[2, 1]], [[3, 2]], []];
  function isoContour(f, w, h, lvl) {
    const segs = [];
    let length = 0, area2 = 0;
    const lerp = (a, b) => { const d = b - a; if (Math.abs(d) < 1e-9) return 0.5; return Math.max(0, Math.min(1, (lvl - a) / d)); };
    for (let y = 0; y < h - 1; y++) {
      for (let x = 0; x < w - 1; x++) {
        const tl = f[y * w + x], tr = f[y * w + x + 1], br = f[(y + 1) * w + x + 1], bl = f[(y + 1) * w + x];
        const c = (tl >= lvl ? 8 : 0) | (tr >= lvl ? 4 : 0) | (br >= lvl ? 2 : 0) | (bl >= lvl ? 1 : 0);
        if (c === 0 || c === 15) continue;
        const gx = ((tr - tl) + (br - bl)) / 2, gy = ((bl - tl) + (br - tr)) / 2; // points toward "inside"
        const pt = e => {
          switch (e) {
            case 0: return [x + 0.5 + lerp(tl, tr), y + 0.5];
            case 1: return [x + 1.5, y + 0.5 + lerp(tr, br)];
            case 2: return [x + 0.5 + lerp(bl, br), y + 1.5];
            default: return [x + 0.5, y + 0.5 + lerp(tl, bl)];
          }
        };
        for (const [e1, e2] of MS_CASES[c]) {
          let a = pt(e1), b = pt(e2);
          const dx = b[0] - a[0], dy = b[1] - a[1];
          if (dy * gx - dx * gy < 0) { const t = a; a = b; b = t; } // keep the inside on one fixed side
          length += Math.hypot(dx, dy);
          area2 += a[0] * b[1] - b[0] * a[1];
          segs.push([a, b]);
        }
      }
    }
    return { length, area: Math.abs(area2) / 2, segs };
  }

  function convexHull(points) {
    const pts = points.slice().sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    if (pts.length < 3) return pts;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }
  function polygonArea(poly) {
    let a = 0;
    for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; a += p[0] * q[1] - q[0] * p[1]; }
    return Math.abs(a) / 2;
  }

  function measure(px, size) {
    const w = size, h = size;
    const ink = toInk(px);
    const thr = otsuThreshold(ink);
    const mask0 = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) mask0[i] = ink[i] >= thr ? 1 : 0;
    const comp = fillHoles(largestComponent(mask0, w, h), w, h);
    const interior = erode(comp, w, h);        // 1 px in from the membrane
    const deep = erode(interior, w, h);        // 2 px in: used for darkness & texture so the membrane cannot leak shape
    const halo = new Uint8Array(w * h);        // just outside the membrane
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (comp[i]) continue;
      for (let dy = -1; dy <= 1 && !halo[i]; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h && comp[yy * w + xx]) { halo[i] = 1; break; }
      }
    }
    // field for the sub-pixel contour: real ink at the membrane, clamped above the level deep inside, zero far outside
    const f = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      if (interior[i]) f[i] = Math.max(ink[i], thr + 0.05);
      else if (comp[i] || halo[i]) f[i] = ink[i];
      else f[i] = 0;
    }
    const contour = isoContour(f, w, h, thr);
    const area = contour.area;

    // centroid & second moments from the pixel mask (for orientation / elongation)
    let count = 0, sx = 0, sy = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (comp[y * w + x]) { count++; sx += x + 0.5; sy += y + 0.5; }
    const cx = sx / count, cy = sy / count;
    let mxx = 0, myy = 0, mxy = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (comp[y * w + x]) {
      const dx = x + 0.5 - cx, dy = y + 0.5 - cy; mxx += dx * dx; myy += dy * dy; mxy += dx * dy;
    }
    mxx = mxx / count + 1 / 12; myy = myy / count + 1 / 12; mxy /= count;
    const tr = mxx + myy, det = mxx * myy - mxy * mxy;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc, l2 = Math.max(1e-6, tr / 2 - disc);
    const elongation = Math.sqrt(l1 / l2);
    const angle = 0.5 * Math.atan2(2 * mxy, mxx - myy);
    // the smooth ellipse with the same area and elongation
    const ea = Math.sqrt(area * elongation / Math.PI), eb = Math.sqrt(area / (Math.PI * elongation));
    const ellipsePerimeter = Math.PI * (3 * (ea + eb) - Math.sqrt((3 * ea + eb) * (ea + 3 * eb)));
    const roughness = contour.length / ellipsePerimeter;

    // convex hull of the sub-pixel contour
    const pts = [];
    for (const s of contour.segs) pts.push(s[0], s[1]);
    const hull = convexHull(pts);
    const hullArea = Math.max(area, polygonArea(hull));
    const solidity = area / hullArea;

    let n = 0, s = 0, s2 = 0;
    for (let i = 0; i < w * h; i++) if (deep[i]) { n++; s += ink[i]; s2 += ink[i] * ink[i]; }
    const darkness = n ? s / n : 0;
    const texture = n ? Math.sqrt(Math.max(0, s2 / n - darkness * darkness)) : 0;

    return {
      features: { area, elongation, darkness, texture, solidity, roughness },
      vector: [area, elongation, darkness, texture, solidity, roughness],
      threshold: thr, mask: comp, deep, contour: contour.segs, perimeter: contour.length,
      hull, centroid: [cx, cy], ellipse: { a: ea, b: eb, angle },
    };
  }

  return { FEATURES, decodeBase64, toInk, measure, otsuThreshold };
});
