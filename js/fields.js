/*
 * fields.js — the tissue fields of the invasion question, shared by the page and the Node tools: a field ships as a
 * PNG without its pixel grain, the grain comes back from the field's seed (so Node and the page see the same pixels),
 * and every nucleus's crop is the 32 × 32 window around its centre, which is what the frozen encoder reads.
 * Works in the browser (window.NucleusFields) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NucleusFields = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const GRAIN = 0.024; // the per-pixel grain, in grey levels 0..1: between a crop's background noise and its nuclear grain
  function mulberry32(seed) { let a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gaussianFrom(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  // the grain back on a grainless field: the same numbers wherever it runs
  function withGrain(px, seed, sigma) {
    const rng = mulberry32(seed), out = new Uint8Array(px.length), s = sigma == null ? GRAIN : sigma;
    for (let i = 0; i < px.length; i++) out[i] = Math.max(0, Math.min(255, Math.round(px[i] + gaussianFrom(rng) * s * 255)));
    return out;
  }
  // the crop of one nucleus: size × size grey levels around (x, y); outside the field, the field's pale edge value
  function crop(px, w, h, x, y, size, fill) {
    const out = new Uint8Array(size * size), cx = Math.round(x) - size / 2, cy = Math.round(y) - size / 2, f = fill == null ? 236 : fill;
    for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) { const X = cx + i, Y = cy + j; out[j * size + i] = X < 0 || Y < 0 || X >= w || Y >= h ? f : px[Y * w + X]; }
    return out;
  }
  // the crop of one nucleus with the field around it masked away, as a nuclear segmentation would leave it: pixels the
  // segmentation gives to another nucleus, or to no nucleus, become the pale background of a lone-nucleus crop, with
  // the same grain, so that the encoder sees the nucleus alone (index: the nucleus's index in its field, 0-based)
  const MASK_FILL = 232;
  function cropMasked(px, seg, w, h, x, y, size, index, seed) {
    const out = crop(px, w, h, x, y, size, MASK_FILL), lab = crop(seg, w, h, x, y, size, 0), rng = mulberry32((seed + 7919 * (index + 1)) >>> 0), want = index + 1;
    for (let i = 0; i < out.length; i++) if (lab[i] !== want) out[i] = Math.max(0, Math.min(255, Math.round(MASK_FILL + gaussianFrom(rng) * GRAIN * 255)));
    return out;
  }
  // the membrane's height at x from its samples every 8 px
  function membraneAt(field, x) { const s = field.membrane, k = Math.max(0, Math.min(s.length - 2, Math.floor(x / 8))), t = Math.max(0, Math.min(1, x / 8 - k)); return s[k] + (s[k + 1] - s[k]) * t; }
  // a field's PNG as grey levels, in Node: inflate, then undo the row filters
  function decodePNGNode(b64) {
    const zlib = require('zlib'), buf = Buffer.from(b64, 'base64');
    let p = 8, w = 0, h = 0; const idat = [];
    while (p < buf.length) { const len = buf.readUInt32BE(p), type = buf.toString('ascii', p + 4, p + 8); if (type === 'IHDR') { w = buf.readUInt32BE(p + 8); h = buf.readUInt32BE(p + 12); } if (type === 'IDAT') idat.push(buf.subarray(p + 8, p + 8 + len)); p += 12 + len; }
    const raw = zlib.inflateSync(Buffer.concat(idat)), px = new Uint8Array(w * h), paeth = (a, b, c) => { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); return pa <= pb && pa <= pc ? a : pb <= pc ? b : c; };
    for (let y = 0; y < h; y++) { // the five row filters of a greyscale PNG, undone
      const type = raw[y * (w + 1)], o = y * (w + 1) + 1;
      for (let x = 0; x < w; x++) { const a = x ? px[y * w + x - 1] : 0, b = y ? px[(y - 1) * w + x] : 0, c = x && y ? px[(y - 1) * w + x - 1] : 0; px[y * w + x] = (raw[o + x] + (type === 0 ? 0 : type === 1 ? a : type === 2 ? b : type === 3 ? (a + b) >> 1 : paeth(a, b, c))) & 255; }
    }
    return { w, h, px };
  }
  // every nucleus of the given fields as the model will see it, in Node: its crop as ink (1 dark … 0 pale), with the
  // field around it or masked to the nucleus alone, its position, and its truth (kind, atypical, below the membrane,
  // which nest); F is window.LECTURE_FIELDS
  function nucleiOf(F, fields, { masked = false } = {}) {
    const K = F.meta.nucleus.reduce((o, k, i) => Object.assign(o, { [k]: i }), {}), out = [];
    for (const f of fields) {
      const d = decodePNGNode(f.png), px = withGrain(d.px, f.grainSeed, F.meta.grain), seg = masked ? decodePNGNode(f.seg).px : null;
      f.nuclei.forEach((n, index) => { const c = masked ? cropMasked(px, seg, d.w, d.h, n[K.x], n[K.y], F.meta.size, index, f.grainSeed) : crop(px, d.w, d.h, n[K.x], n[K.y], F.meta.size), ink = new Float32Array(c.length); for (let i = 0; i < c.length; i++) ink[i] = 1 - c[i] / 255; out.push({ field: f, index, x: n[K.x], y: n[K.y], ink, kind: F.meta.kinds[n[K.kind]], atypical: n[K.atypical], below: n[K.below], nest: n[K.nest], pattern: f.pattern }); });
    }
    return out;
  }
  // a greyscale PNG, in Node, with the row filter chosen per row (None, Sub, Up, Average or Paeth, whichever leaves
    // the smallest residuals): the fields are smooth, so the residuals compress far better than raw rows
  function encodePNGNode(width, height, px) {
    const zlib = require('zlib');
    const raw = Buffer.alloc((width + 1) * height), paeth = (a, b, c) => { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); return pa <= pb && pa <= pc ? a : pb <= pc ? b : c; };
    const line = new Uint8Array(width), best = new Uint8Array(width);
    for (let y = 0; y < height; y++) {
      let bestType = 0, bestSum = Infinity;
      for (let type = 0; type < 5; type++) {
        let sum = 0;
        for (let x = 0; x < width; x++) {
          const v = px[y * width + x], a = x ? px[y * width + x - 1] : 0, b = y ? px[(y - 1) * width + x] : 0, c = x && y ? px[(y - 1) * width + x - 1] : 0;
          const r = (v - (type === 0 ? 0 : type === 1 ? a : type === 2 ? b : type === 3 ? (a + b) >> 1 : paeth(a, b, c))) & 255;
          line[x] = r; sum += r < 128 ? r : 256 - r;
        }
        if (sum < bestSum) { bestSum = sum; bestType = type; best.set(line); }
      }
      raw[y * (width + 1)] = bestType; raw.set(best, y * (width + 1) + 1);
    }
    const crcTable = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; crcTable.push(c >>> 0); }
    const crc = buf => { let c = 0xFFFFFFFF; for (const b of buf) c = crcTable[(c ^ b) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
    const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type, 'ascii'), data]), c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
    const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = 0;
    return Buffer.concat([Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
  }
  // a field's PNG as grey levels, in the browser (through an Image and a canvas): a promise
  function decodePNGBrowser(b64, w, h) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => { const cv = document.createElement('canvas'); cv.width = w; cv.height = h; const ctx = cv.getContext('2d', { willReadFrequently: true }); ctx.drawImage(img, 0, 0); const d = ctx.getImageData(0, 0, w, h).data, px = new Uint8Array(w * h); for (let i = 0; i < px.length; i++) px[i] = d[i * 4]; resolve(px); };
      img.onerror = () => reject(new Error('fields.js: the field image did not decode'));
      img.src = 'data:image/png;base64,' + b64;
    });
  }
  return { GRAIN, MASK_FILL, mulberry32, gaussianFrom, withGrain, crop, cropMasked, membraneAt, nucleiOf, encodePNGNode, decodePNGNode, decodePNGBrowser };
});
