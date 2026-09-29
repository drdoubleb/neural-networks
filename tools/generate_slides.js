#!/usr/bin/env node
/*
 * generate_slides.js — builds the slides of the 5 · Slides stage: a pool of fresh nuclei drawn from the atypia
 * generator (bland, and atypical in one or more ways), and two questions asked of slides of 20 nuclei with ONE label
 * each. "Atypical cells on the slide?": "no atypical cells" (20 bland nuclei) or "atypical cells present" (2 to 4
 * atypical nuclei among bland ones). "A focus of atypical cells?": every slide holds exactly four atypical nuclei
 * among sixteen bland ones, on a positive slide as a 2 × 2 block on the 5 × 4 grid (a focus), on a negative slide
 * scattered (no two adjacent, not even diagonally), so the nuclei are the same in both classes and only their
 * arrangement differs. Nobody labels the nuclei of a slide; the pool keeps the truth so that the page can reveal,
 * afterwards, where the attention landed.
 *   node tools/generate_slides.js
 * writes data/slides/slides_data.js (the pool as base64 images, both questions' slide definitions) and a contact sheet
 * per question of the first eight training slides (atypical nuclei framed dark).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const gen = require('./generate_nuclei.js');

const SEED = 20260929, SIZE = gen.SIZE, PER_SLIDE = 20, COLS = 5, N_BLAND = 160, N_ATYPICAL = 80, TRAIN_SLIDES = 60, TEST_SLIDES = 20, FOCUS_K = 4, FOCUS_TRAIN = 200, FOCUS_TEST = 40;
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
  seed: SEED, size: SIZE, perSlide: PER_SLIDE, cols: COLS, question: 'Atypical cells on the slide?', short: 'Atypical cells',
  classes: [{ key: 'clean', name: 'No atypical cells' }, { key: 'atypical', name: 'Atypical cells present' }],
  blurb: 'Every slide is 20 nuclei from a pool of fresh ones, and carries one label: no atypical cells (20 bland nuclei), or atypical cells present (2 to 4 atypical nuclei among bland ones, atypical in any of the four ways of the atypia question). Nobody labels the nuclei; the pool remembers which are atypical so the page can reveal, afterwards, where the attention landed.',
  pool: pool.length, train: train.length, test: test.length,
};

// ---- the second question: a focus. Four atypical nuclei on every slide; a 2 × 2 block on the 5 × 4 grid, or scattered.
const rngF = gen.mulberry32(SEED + 2);
const pickF = (arr, k) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rngF() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, k); };
const cell = k => [k % COLS, Math.floor(k / COLS)];
const adjacent4 = (a, b) => { const [ax, ay] = cell(a), [bx, by] = cell(b); return Math.abs(ax - bx) + Math.abs(ay - by) === 1; };
const near8 = (a, b) => { const [ax, ay] = cell(a), [bx, by] = cell(b); return Math.max(Math.abs(ax - bx), Math.abs(ay - by)) <= 1; };
const groups = { focus: [], scattered: [] }, ROWS = PER_SLIDE / COLS;
for (let k = 0; k < PER_SLIDE; k++) if (k % COLS < COLS - 1 && Math.floor(k / COLS) < ROWS - 1) groups.focus.push([k, k + 1, k + COLS, k + COLS + 1]); // a 2 × 2 block
for (let a = 0; a < PER_SLIDE; a++) for (let b = a + 1; b < PER_SLIDE; b++) for (let c = b + 1; c < PER_SLIDE; c++) for (let d = c + 1; d < PER_SLIDE; d++) {
  const cells = [a, b, c, d]; let ok = true;
  for (let i = 0; i < 4 && ok; i++) for (let j = i + 1; j < 4; j++) if (near8(cells[i], cells[j])) { ok = false; break; }
  if (ok) groups.scattered.push(cells); // no two even diagonally adjacent
}
void adjacent4;
function makeFocusSlide(label, split) {
  const bland = pool.filter(p => p.split === split && !p.label), atyp = pool.filter(p => p.split === split && p.label);
  const slots = pickF(label ? groups.focus : groups.scattered, 1)[0], at = pickF(atyp, FOCUS_K), bl = pickF(bland, PER_SLIDE - FOCUS_K);
  const nuclei = []; let ia = 0, ib = 0;
  for (let k = 0; k < PER_SLIDE; k++) nuclei.push([(slots.includes(k) ? at[ia++] : bl[ib++]).id, Math.floor(rngF() * 8)]); // [pool index, orientation]; the slot is the grid cell
  return { label, nuclei };
}
const focusSlides = split => { const n = split === "train" ? FOCUS_TRAIN : FOCUS_TEST, out = []; for (let i = 0; i < n; i++) out.push(makeFocusSlide(i % 2, split)); return pickF(out, n); };
const focusTrain = focusSlides('train').map((s, i) => Object.assign({ id: i, name: `F${String(i + 1).padStart(3, "0")}`, split: 'train' }, s));
const focusTest = focusSlides('test').map((s, i) => Object.assign({ id: TRAIN_SLIDES + i, name: `Y${String(i + 1).padStart(2, '0')}`, split: 'test' }, s));
const focusMeta = {
  seed: SEED, size: SIZE, perSlide: PER_SLIDE, cols: COLS, question: 'A focus of atypical cells?', short: 'A focus', positions: true,
  classes: [{ key: 'scattered', name: 'Scattered atypical cells' }, { key: 'focus', name: 'A focus of atypical cells' }],
  blurb: 'Every slide holds exactly four atypical nuclei among sixteen bland ones, from the same pool. On a positive slide the four sit together as a 2 × 2 block, a focus; on a negative slide they are scattered, no two adjacent. The nuclei are the same in both classes and only their arrangement differs, so nothing about any single nucleus, and nothing about the bag of them, tells the classes apart: a model has to let the nuclei look at their neighbours.',
  pool: pool.length, train: focusTrain.length, test: focusTest.length,
};

fs.mkdirSync(OUT, { recursive: true });
const js = `window.LECTURE_SLIDES = ${JSON.stringify({ pool: pool.map(({ split, ...p }) => p), questions: { atypia: { meta, train, test }, focus: { meta: focusMeta, train: focusTrain, test: focusTest } } })};\n`;
fs.writeFileSync(path.join(OUT, 'slides_data.js'), js);

// ---- a contact sheet per question of the first eight training slides: 5 × 4 nuclei each, atypical ones framed dark
for (const [file, set] of [['contact_sheet.png', train], ['contact_sheet_focus.png', focusTrain]]) {
  const cols = 5, rows = 4, gap = 3, cell = SIZE + gap, sw = cols * cell + gap, sh = rows * cell + gap, per = 4, pad = 12;
  const W = per * sw + (per + 1) * pad, H = 2 * sh + 3 * pad, pix = new Uint8Array(W * H).fill(255);
  set.slice(0, 8).forEach((s, si) => {
    const ox = pad + (si % per) * (sw + pad), oy = pad + Math.floor(si / per) * (sh + pad);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) pix[(oy + y) * W + ox + x] = 225;
    s.nuclei.forEach(([id, t], k) => {
      const px = Buffer.from(pool[id].px, 'base64'), cx = ox + gap + (k % cols) * cell, cy = oy + gap + Math.floor(k / cols) * cell;
      const img = t ? dihedral(px, t) : px;
      for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) pix[(cy + y) * W + cx + x] = img[y * SIZE + x];
      if (pool[id].label) for (let y = -1; y <= SIZE; y++) for (let x = -1; x <= SIZE; x++) if (y === -1 || y === SIZE || x === -1 || x === SIZE) pix[(cy + y) * W + cx + x] = 40;
    });
  });
  fs.writeFileSync(path.join(OUT, file), gen.encodePNG(W, H, 1, pix));
}
function dihedral(px, t) { // the 8 symmetries of a square image, as js/dataset.js applies them
  const out = new Uint8Array(px.length), rot = t & 3, flip = t >> 2;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) { let sx = x, sy = y; if (flip) sx = SIZE - 1 - sx; for (let r = 0; r < rot; r++) { const nx = SIZE - 1 - sy; sy = sx; sx = nx; } out[y * SIZE + x] = px[sy * SIZE + sx]; }
  return out;
}
const kPos = train.filter(s => s.label).map(s => s.nuclei.filter(([id]) => pool[id].label).length);
console.log(`wrote ${train.length} training and ${test.length} test slides of ${PER_SLIDE} nuclei from a pool of ${pool.length} (${N_BLAND} bland, ${N_ATYPICAL} atypical; ${pool.filter(p => p.split === 'test').length} held out for the test slides) to ${OUT} (${(js.length / 1024).toFixed(0)} KB)`);
console.log(`positive training slides carry ${Math.min(...kPos)} to ${Math.max(...kPos)} atypical nuclei (mean ${(kPos.reduce((a, b) => a + b, 0) / kPos.length).toFixed(1)})`);
console.log(`focus question: ${groups.focus.length} ways to place a 2 × 2 focus and ${groups.scattered.length} ways to scatter four on the ${COLS} × ${PER_SLIDE / COLS} grid; ${focusTrain.length} training and ${focusTest.length} test slides, ${focusTrain.filter(s => s.label).length} + ${focusTest.filter(s => s.label).length} with a focus`);
