#!/usr/bin/env node
/*
 * generate_nuclei.js — builds the synthetic nucleus datasets used by the demo, one per lecture question.
 *
 * Zero dependencies (Node >= 16). Run:  node tools/generate_nuclei.js
 *
 * For each task (data/enlargement, data/irregularity) it writes:
 *   nuclei_data.js        all 100 images (base64 grayscale) + labels + split, loaded by index.html
 *   nuclei.json           manifest: labels, split, and the generator parameters of every nucleus
 *   images/train/*.png    80 training nuclei  (40 + 40)
 *   images/test/*.png     20 test nuclei      (10 + 10)
 *   contact_sheet.png     all 100 at 4x, class-coloured frames, for slides
 *
 * Design of the data (this matters for the lecture):
 *   - Every nucleus is a dark shape on a pale background, 32x32 pixels, 8-bit grayscale.
 *   - Task "irregularity": size, elongation, rotation, darkness, chromatin texture, nucleolus presence and
 *     position jitter are drawn from the SAME distributions for both classes. The only systematic difference
 *     between "regular" and "irregular" is the contour: smooth ellipses versus lobulated, notched/blebbed or
 *     finely jagged outlines.
 *   - Task "enlargement": the classes differ in size and darkness (bland = small and pale, enlarged = large
 *     and hyperchromatic), and now the contour is the decoy: half of each class is irregular.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

// ----------------------------------------------------------------------------- config
const SIZE = 32;                // image side in pixels
const SS = 4;                   // supersampling factor for anti-aliased contours
const N_PER_CLASS = 50;         // 50 regular + 50 irregular = 100 nuclei
const N_TEST_PER_CLASS = 10;    // 10 + 10 = 20 test, the remaining 80 train
const DATA_ROOT = path.join(__dirname, '..', 'data');

const TASKS = [
  {
    id: 'enlargement', order: 1, kind: 'image', seed: 20260923,
    subtypes: [{ key: 'bland-smooth', name: 'Bland · smooth contour', positive: 0 }, { key: 'bland-irregular', name: 'Bland · irregular contour', positive: 0 }, { key: 'enlarged-smooth', name: 'Enlarged · smooth contour', positive: 1 }, { key: 'enlarged-irregular', name: 'Enlarged · irregular contour', positive: 1 }],
    title: 'Enlarged and hyperchromatic?',
    short: 'Enlargement',
    classes: [{ key: 'bland', name: 'Bland' }, { key: 'enlarged', name: 'Enlarged' }],
    blurb: 'Half the nuclei are bland: small and pale. Half are enlarged and hyperchromatic: bigger and darker, the first thing residents learn to look for. Everything else is a decoy, and this time that includes the contour: half of each class has an irregular outline.',
    decoys: 'contour shape, elongation, rotation, chromatin texture, nucleolus and position',
    signal: 'size and darkness',
  },
  {
    id: 'irregularity', order: 2, kind: 'image', seed: 20260922,
    subtypes: [{ key: 'smooth', name: 'Smooth ellipse', positive: 0 }, { key: 'lobulated', name: 'Lobulated', positive: 1 }, { key: 'notched', name: 'Notched / blebbed', positive: 1 }, { key: 'jagged', name: 'Finely jagged', positive: 1 }],
    title: 'Irregular contour?',
    short: 'Irregularity',
    classes: [{ key: 'regular', name: 'Regular' }, { key: 'irregular', name: 'Irregular' }],
    blurb: 'Half the nuclei have a smooth, elliptical membrane; half are irregular: lobulated, notched or finely jagged. Size, elongation, rotation, hyperchromasia, chromatin texture, a nucleolus and position in the crop are drawn from the same distribution in both groups. The only honest way to score well is to look at the outline.',
    decoys: 'size, elongation, rotation, darkness, chromatin texture, nucleolus and position',
    signal: 'the contour',
  },
];

// ----------------------------------------------------------------------------- PRNG
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let rand = mulberry32(1); // re-seeded per task below
const uniform = (lo, hi) => lo + (hi - lo) * rand();
const randint = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1)); // inclusive
function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ----------------------------------------------------------------------------- smooth noise (chromatin texture)
function makeValueNoise(cells) {
  // a cells x cells lattice of random values, bilinearly interpolated over [0,1)^2
  const g = new Float64Array((cells + 1) * (cells + 1));
  for (let i = 0; i < g.length; i++) g[i] = rand() * 2 - 1;
  return function (u, v) { // u,v in [0,1)
    const x = u * cells, y = v * cells;
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const at = (i, j) => g[Math.min(j, cells) * (cells + 1) + Math.min(i, cells)];
    const a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1);
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy;
  };
}

// ----------------------------------------------------------------------------- contour models
// The contour is an ellipse (semi-axes a, b, rotation phi) whose radius is multiplied by (1 + m(theta)).
// m(theta) is a sum of low-order harmonics plus optional localised notches (negative) or blebs (positive).
function sampleShapeParams(label, task) {
  const p = {
    a: task.id === 'enlargement' ? (label ? uniform(10.0, 12.0) : uniform(7.5, 9.5)) : uniform(8.5, 11.0), // semi-major axis, px
    aspect: uniform(1.0, 1.4),                  // major/minor
    phi: uniform(0, Math.PI),                   // rotation
    cx: SIZE / 2 + uniform(-1.0, 1.0),          // centre jitter
    cy: SIZE / 2 + uniform(-1.0, 1.0),
    harmonics: [],                              // {k, amp, phase}
    bumps: [],                                  // {theta0, depth (signed), width}
    style: 'smooth',
  };
  p.b = p.a / p.aspect;
  // which contour family? in the irregularity task it IS the label; in the enlargement task it is a coin flip
  const contourClass = task.id === 'irregularity' ? label : (rand() < 0.5 ? 0 : 1);
  p.contourClass = contourClass;
  if (contourClass === 0) {
    // regular: essentially a perfect ellipse, at most a faint egg-shape
    p.style = 'smooth';
    const k = randint(2, 3);
    p.harmonics.push({ k, amp: uniform(0, 0.02), phase: uniform(0, 2 * Math.PI) });
  } else {
    const style = ['lobulated', 'notched', 'jagged'][randint(0, 2)];
    p.style = style;
    if (style === 'lobulated') {
      const n = randint(3, 4);
      const ks = shuffle([3, 4, 5, 6, 7, 8]).slice(0, n);
      for (const k of ks) p.harmonics.push({ k, amp: uniform(0.05, 0.11), phase: uniform(0, 2 * Math.PI) });
    } else if (style === 'notched') {
      const n = randint(1, 2);
      for (let i = 0; i < n; i++) {
        const sign = rand() < 0.65 ? -1 : 1;   // mostly notches (clefts), sometimes blebs
        p.bumps.push({ theta0: uniform(0, 2 * Math.PI), depth: sign * uniform(0.28, 0.42), width: uniform(0.22, 0.36) });
      }
      p.harmonics.push({ k: randint(3, 5), amp: uniform(0.02, 0.04), phase: uniform(0, 2 * Math.PI) });
    } else { // jagged
      const n = randint(4, 5);
      const ks = shuffle([6, 7, 8, 9, 10, 11]).slice(0, n);
      for (const k of ks) p.harmonics.push({ k, amp: uniform(0.035, 0.06), phase: uniform(0, 2 * Math.PI) });
    }
  }
  return p;
}
function contourModulation(p, theta) {
  let m = 0;
  for (const h of p.harmonics) m += h.amp * Math.cos(h.k * theta + h.phase);
  for (const b of p.bumps) {
    let d = theta - b.theta0;
    d = Math.atan2(Math.sin(d), Math.cos(d)); // wrap to [-pi, pi]
    m += b.depth * Math.exp(-(d * d) / (2 * b.width * b.width));
  }
  return m;
}
function contourRms(p) {
  let s = 0; const n = 720;
  for (let i = 0; i < n; i++) { const m = contourModulation(p, (i / n) * 2 * Math.PI); s += m * m; }
  return Math.sqrt(s / n);
}

// ----------------------------------------------------------------------------- appearance (shared by both classes)
function sampleAppearance(label, task) {
  return {
    bg: uniform(0.88, 0.94),                 // background intensity (pale eosin)
    bgNoise: 0.02,
    nucleus: task.id === 'enlargement' ? (label ? uniform(0.22, 0.36) : uniform(0.40, 0.52)) : uniform(0.30, 0.46), // mean nuclear intensity
    textureAmp: uniform(0.04, 0.09),         // chromatin clumping amplitude
    grain: 0.018,                            // per-pixel grain
    rim: uniform(0.03, 0.08),                // slightly darker nuclear membrane / margination
    nucleolus: rand() < 0.35 ? { r: uniform(1.0, 1.6), rho: uniform(0.0, 0.5), ang: uniform(0, 2 * Math.PI), dark: uniform(0.10, 0.18) } : null,
    edgeSoftness: uniform(0.35, 0.6),        // px of extra blur at the membrane
  };
}

// ----------------------------------------------------------------------------- rasteriser
function renderNucleus(shape, look) {
  const W = SIZE * SS;
  const cov = new Float64Array(SIZE * SIZE);   // coverage (0..1) of the nucleus per pixel
  const rhoAcc = new Float64Array(SIZE * SIZE); // mean normalised radius per pixel (for rim shading)
  const cosP = Math.cos(shape.phi), sinP = Math.sin(shape.phi);
  for (let v = 0; v < W; v++) {
    for (let u = 0; u < W; u++) {
      const px = (u + 0.5) / SS, py = (v + 0.5) / SS;
      const dx = px - shape.cx, dy = py - shape.cy;
      const xr = dx * cosP + dy * sinP, yr = -dx * sinP + dy * cosP;
      const ex = xr / shape.a, ey = yr / shape.b;
      const rho = Math.sqrt(ex * ex + ey * ey);
      const theta = Math.atan2(ey, ex);
      const boundary = 1 + contourModulation(shape, theta);
      const r = rho / boundary; // 0 at centre, 1 at the membrane
      // soft membrane: treat |r-1| within edgeSoftness/a as partial
      const soft = look.edgeSoftness / shape.a;
      let inside;
      if (r <= 1 - soft) inside = 1; else if (r >= 1 + soft) inside = 0; else inside = 0.5 - 0.5 * Math.sin(((r - 1) / soft) * Math.PI / 2);
      const idx = Math.floor(v / SS) * SIZE + Math.floor(u / SS);
      cov[idx] += inside / (SS * SS);
      rhoAcc[idx] += Math.min(r, 1.2) / (SS * SS);
    }
  }
  const tex = makeValueNoise(7);
  const img = new Uint8Array(SIZE * SIZE);
  const nucCos = Math.cos(shape.phi), nucSin = Math.sin(shape.phi);
  let nx = 0, ny = 0;
  if (look.nucleolus) {
    // place nucleolus in the ellipse frame
    const ex = look.nucleolus.rho * Math.cos(look.nucleolus.ang) * shape.a;
    const ey = look.nucleolus.rho * Math.sin(look.nucleolus.ang) * shape.b;
    nx = shape.cx + ex * nucCos - ey * nucSin;
    ny = shape.cy + ex * nucSin + ey * nucCos;
  }
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const i = y * SIZE + x;
      const c = cov[i];
      const bgv = look.bg + gaussian() * look.bgNoise;
      let nv = look.nucleus + look.textureAmp * tex(x / SIZE, y / SIZE);
      const rr = rhoAcc[i];
      if (rr > 0.72) nv -= look.rim * Math.min(1, (rr - 0.72) / 0.28); // darker toward the membrane
      if (look.nucleolus) {
        const d = Math.hypot(x + 0.5 - nx, y + 0.5 - ny);
        if (d < look.nucleolus.r + 0.7) nv -= look.nucleolus.dark * Math.max(0, Math.min(1, look.nucleolus.r + 0.7 - d));
      }
      let val = c * nv + (1 - c) * bgv + gaussian() * look.grain;
      val = Math.max(0, Math.min(1, val));
      img[i] = Math.round(val * 255);
    }
  }
  return img;
}

// ----------------------------------------------------------------------------- PNG encoder (no deps)
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td), 0);
  return Buffer.concat([len, td, crc]);
}
function encodePNG(width, height, channels, pixels) {
  const stride = width * channels;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0; // filter: none
    Buffer.from(pixels.buffer, pixels.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = channels === 1 ? 0 : 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// ----------------------------------------------------------------------------- build the set
function makeOne(label, task) {
  // resample until the contour is unambiguously smooth or unambiguously irregular
  for (let tries = 0; tries < 50; tries++) {
    const shape = sampleShapeParams(label, task);
    const rms = contourRms(shape);
    if (shape.contourClass === 0 && rms > 0.02) continue;
    if (shape.contourClass === 1 && rms < 0.065) continue;
    const look = sampleAppearance(label, task);
    return { shape, look, rms };
  }
  throw new Error('could not sample a nucleus');
}

for (const task of TASKS) buildTask(task);

function buildTask(task) {
const OUT = path.join(DATA_ROOT, task.id);
rand = mulberry32(task.seed);
const negatives = [], positives = [];
for (let i = 0; i < N_PER_CLASS; i++) negatives.push({ label: 0, ...makeOne(0, task) });
for (let i = 0; i < N_PER_CLASS; i++) positives.push({ label: 1, ...makeOne(1, task) });
shuffle(negatives); shuffle(positives);
const test = shuffle([...negatives.slice(0, N_TEST_PER_CLASS), ...positives.slice(0, N_TEST_PER_CLASS)]);
const train = shuffle([...negatives.slice(N_TEST_PER_CLASS), ...positives.slice(N_TEST_PER_CLASS)]);
const all = [...train.map(n => ({ ...n, split: 'train' })), ...test.map(n => ({ ...n, split: 'test' }))];

fs.mkdirSync(path.join(OUT, 'images', 'train'), { recursive: true });
fs.mkdirSync(path.join(OUT, 'images', 'test'), { recursive: true });
for (const f of fs.readdirSync(path.join(OUT, 'images', 'train'))) fs.unlinkSync(path.join(OUT, 'images', 'train', f));
for (const f of fs.readdirSync(path.join(OUT, 'images', 'test'))) fs.unlinkSync(path.join(OUT, 'images', 'test', f));

const records = [];
let trainIdx = 0, testIdx = 0;
for (let i = 0; i < all.length; i++) {
  const n = all[i];
  const img = renderNucleus(n.shape, n.look);
  const k = n.split === 'train' ? ++trainIdx : ++testIdx;
  const file = `${n.split}_${String(k).padStart(2, '0')}_${task.classes[n.label].key}.png`;
  fs.writeFileSync(path.join(OUT, 'images', n.split, file), encodePNG(SIZE, SIZE, 1, img));
  records.push({
    id: i,
    name: `${n.split === 'train' ? 'T' : 'X'}${String(k).padStart(2, '0')}`,
    split: n.split,
    label: n.label,
    className: task.classes[n.label].key,
    subtype: task.id === 'irregularity' ? n.shape.style : `${n.label ? 'enlarged' : 'bland'}-${n.shape.contourClass ? 'irregular' : 'smooth'}`,
    file: `images/${n.split}/${file}`,
    px: Buffer.from(img).toString('base64'),
    generator: {
      style: n.shape.style, contourIrregular: n.shape.contourClass === 1,
      semiMajor: +n.shape.a.toFixed(2), semiMinor: +n.shape.b.toFixed(2),
      rotationDeg: +(n.shape.phi * 180 / Math.PI).toFixed(1),
      centre: [+n.shape.cx.toFixed(2), +n.shape.cy.toFixed(2)],
      contourRms: +n.rms.toFixed(3),
      harmonics: n.shape.harmonics.map(h => ({ k: h.k, amp: +h.amp.toFixed(3) })),
      bumps: n.shape.bumps.map(b => ({ depth: +b.depth.toFixed(2), width: +b.width.toFixed(2) })),
      meanIntensity: +n.look.nucleus.toFixed(2),
      textureAmp: +n.look.textureAmp.toFixed(3),
      nucleolus: !!n.look.nucleolus,
    },
  });
}

// contact sheet: 10 x 10, 4x scale, 2px frame in class colour (blue = regular, orange = irregular),
// train first (rows 1-8) then test (rows 9-10)
(function contactSheet() {
  const scale = 4, cell = SIZE * scale, gap = 6, cols = 10;
  const rows = Math.ceil(records.length / cols);
  const W = cols * cell + (cols + 1) * gap, H = rows * cell + (rows + 1) * gap;
  const rgb = new Uint8Array(W * H * 3);
  rgb.fill(250);
  const frame = { 0: [42, 120, 214], 1: [235, 104, 52] };
  records.forEach((r, idx) => {
    const c = idx % cols, rr = Math.floor(idx / cols);
    const x0 = gap + c * (cell + gap), y0 = gap + rr * (cell + gap);
    const px = Buffer.from(r.px, 'base64');
    for (let y = -2; y < cell + 2; y++) for (let x = -2; x < cell + 2; x++) {
      const X = x0 + x, Y = y0 + y; if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
      const o = (Y * W + X) * 3;
      if (x < 0 || y < 0 || x >= cell || y >= cell) { const f = frame[r.label]; rgb[o] = f[0]; rgb[o + 1] = f[1]; rgb[o + 2] = f[2]; continue; }
      const v = px[Math.floor(y / scale) * SIZE + Math.floor(x / scale)];
      rgb[o] = v; rgb[o + 1] = v; rgb[o + 2] = v;
    }
  });
  fs.writeFileSync(path.join(OUT, 'contact_sheet.png'), encodePNG(W, H, 3, rgb));
})();

const meta = {
  generated: new Date().toISOString().slice(0, 10),
  seed: task.seed, size: SIZE, count: records.length,
  train: records.filter(r => r.split === 'train').length,
  test: records.filter(r => r.split === 'test').length,
  task: { id: task.id, order: task.order, kind: task.kind, title: task.title, short: task.short, classes: task.classes, blurb: task.blurb, decoys: task.decoys, signal: task.signal, subtypes: task.subtypes, specimenNoun: 'nucleus' },
  classes: task.classes.map(c => c.key),
};
fs.writeFileSync(path.join(OUT, 'nuclei.json'), JSON.stringify({ meta, nuclei: records.map(({ px, ...rest }) => rest) }, null, 1));
const js = `// Generated by tools/generate_nuclei.js — do not edit by hand.\n` +
  `// Task "${task.id}": ${meta.count} synthetic nuclei, ${SIZE}x${SIZE} 8-bit grayscale, base64 of the raw pixel rows (0 = black, 255 = white).\n` +
  `window.LECTURE_TASKS = window.LECTURE_TASKS || {};\n` +
  `window.LECTURE_TASKS[${JSON.stringify(task.id)}] = ${JSON.stringify({ meta, nuclei: records.map(({ generator, ...rest }) => rest) })};\n`;
fs.writeFileSync(path.join(OUT, 'nuclei_data.js'), js);

const styles = {};
for (const r of records) styles[r.generator.style] = (styles[r.generator.style] || 0) + 1;
console.log(`[${task.id}] wrote ${records.length} nuclei (${meta.train} train / ${meta.test} test) to ${OUT}`);
console.log('  contour styles:', styles);
}
