#!/usr/bin/env node
/*
 * generate_slides.js — builds the slides of the 5 · Slides stage: a pool of fresh nuclei drawn from the atypia
 * generator (bland, and atypical in one or more ways), and slides of 20 nuclei with ONE label each: "no atypical cells"
 * (20 bland nuclei) or "atypical cells present" (2 to 4 atypical nuclei among bland ones). Nobody labels the nuclei of
 * a slide; the pool keeps the truth so that the page can reveal, afterwards, where the attention landed.
 *   node tools/generate_slides.js
 * writes data/slides/slides_data.js (the pool as base64 images, the slide definitions) and a contact sheet of the
 * first eight training slides (atypical nuclei framed dark).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const gen = require('./generate_nuclei.js');

const SEED = 20260929, SIZE = gen.SIZE, PER_SLIDE = 20, N_BLAND = 160, N_ATYPICAL = 80, TRAIN_SLIDES = 60, TEST_SLIDES = 20;
const OUT = path.join(__dirname, '..', 'data', 'slides');
const task = gen.TASKS.find(t => t.id === 'atypia');

// ---- the pool: every 4th nucleus of each class is held out for the test slides
gen.setSeed(SEED);
const pool = [];
const render = (label, i) => {
  const n = gen.makeOne(label, task, i);
  const tex = gen.makeValueNoise(n.look.textureCells || 7);
  const img = gen.renderNucleus(n.shape, n.look, tex, gen.gaussian);
  return { label, subtype: n.traits ? n.traits.subtype : 'bland', px: Buffer.from(img).toString('base64') };
};
for (let i = 0; i < N_BLAND; i++) pool.push(Object.assign(render(0, i), { split: i % 4 === 3 ? 'test' : 'train' }));
for (let j = 0; j < N_ATYPICAL; j++) pool.push(Object.assign(render(1, j), { split: j % 4 === 3 ? 'test' : 'train' })); // j < 24: one trait each, then combinations
pool.forEach((p, i) => { p.id = i; });

// ---- the slides
const rng = gen.mulberry32(SEED + 1);
const pick = (arr, k) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, k); };
function makeSlide(label, split) {
  const bland = pool.filter(p => p.split === split && !p.label), atyp = pool.filter(p => p.split === split && p.label);
  const k = label ? 2 + Math.floor(rng() * 3) : 0;
  const nuclei = pick([...pick(atyp, k), ...pick(bland, PER_SLIDE - k)], PER_SLIDE).map(p => [p.id, Math.floor(rng() * 8)]); // [pool index, orientation]
  return { label, nuclei };
}
const slides = split => { const n = split === 'train' ? TRAIN_SLIDES : TEST_SLIDES, out = []; for (let i = 0; i < n; i++) out.push(makeSlide(i % 2, split)); return pick(out, n); };
const train = slides('train').map((s, i) => Object.assign({ id: i, name: `S${String(i + 1).padStart(2, '0')}`, split: 'train' }, s));
const test = slides('test').map((s, i) => Object.assign({ id: TRAIN_SLIDES + i, name: `X${String(i + 1).padStart(2, '0')}`, split: 'test' }, s));

const meta = {
  seed: SEED, size: SIZE, perSlide: PER_SLIDE, question: 'Atypical cells on the slide?', short: 'Slides',
  classes: [{ key: 'clean', name: 'No atypical cells' }, { key: 'atypical', name: 'Atypical cells present' }],
  blurb: 'Every slide is 20 nuclei from a pool of fresh ones, and carries one label: no atypical cells (20 bland nuclei), or atypical cells present (2 to 4 atypical nuclei among bland ones, atypical in any of the four ways of the atypia question). Nobody labels the nuclei; the pool remembers which are atypical so the page can reveal, afterwards, where the attention landed.',
  pool: pool.length, train: train.length, test: test.length,
};
fs.mkdirSync(OUT, { recursive: true });
const js = `window.LECTURE_SLIDES = ${JSON.stringify({ meta, pool: pool.map(({ split, ...p }) => p), train, test })};\n`;
fs.writeFileSync(path.join(OUT, 'slides_data.js'), js);

// ---- a contact sheet of the first eight training slides: 5 × 4 nuclei each, atypical ones framed dark
{
  const cols = 5, rows = 4, gap = 3, cell = SIZE + gap, sw = cols * cell + gap, sh = rows * cell + gap, per = 4, pad = 12;
  const W = per * sw + (per + 1) * pad, H = 2 * sh + 3 * pad, pix = new Uint8Array(W * H).fill(255);
  train.slice(0, 8).forEach((s, si) => {
    const ox = pad + (si % per) * (sw + pad), oy = pad + Math.floor(si / per) * (sh + pad);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) pix[(oy + y) * W + ox + x] = 225;
    s.nuclei.forEach(([id, t], k) => {
      const px = Buffer.from(pool[id].px, 'base64'), cx = ox + gap + (k % cols) * cell, cy = oy + gap + Math.floor(k / cols) * cell;
      const img = t ? dihedral(px, t) : px;
      for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) pix[(cy + y) * W + cx + x] = img[y * SIZE + x];
      if (pool[id].label) for (let y = -1; y <= SIZE; y++) for (let x = -1; x <= SIZE; x++) if (y === -1 || y === SIZE || x === -1 || x === SIZE) pix[(cy + y) * W + cx + x] = 40;
    });
  });
  fs.writeFileSync(path.join(OUT, 'contact_sheet.png'), gen.encodePNG(W, H, 1, pix));
}
function dihedral(px, t) { // the 8 symmetries of a square image, as js/dataset.js applies them
  const out = new Uint8Array(px.length), rot = t & 3, flip = t >> 2;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) { let sx = x, sy = y; if (flip) sx = SIZE - 1 - sx; for (let r = 0; r < rot; r++) { const nx = SIZE - 1 - sy; sy = sx; sx = nx; } out[y * SIZE + x] = px[sy * SIZE + sx]; }
  return out;
}
const kPos = train.filter(s => s.label).map(s => s.nuclei.filter(([id]) => pool[id].label).length);
console.log(`wrote ${train.length} training and ${test.length} test slides of ${PER_SLIDE} nuclei from a pool of ${pool.length} (${N_BLAND} bland, ${N_ATYPICAL} atypical; ${pool.filter(p => p.split === 'test').length} held out for the test slides) to ${OUT} (${(js.length / 1024).toFixed(0)} KB)`);
console.log(`positive training slides carry ${Math.min(...kPos)} to ${Math.max(...kPos)} atypical nuclei (mean ${(kPos.reduce((a, b) => a + b, 0) / kPos.length).toFixed(1)})`);
