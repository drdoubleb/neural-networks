#!/usr/bin/env node
/*
 * pretrain_backbone.js — the foundation encoder shipped with the page (data/foundation/backbones.js), pretrained on a
 * separate, larger set of generated nuclei the page never shows, and the measurement behind the choice.
 *   node tools/pretrain_backbone.js              pretrain the shipped backbone(s) and write data/foundation/
 *   node tools/pretrain_backbone.js --measure [--seed 2]   the ablation: pretraining set × encoder size × labelled cases
 * Required as a module (tools/check_training.js), it runs nothing and exports pretrain, pageSet, generateSet and probe.
 * The pretraining set draws N nuclei per question from the page's own generator with another seed (half of each
 * class, so the mix matches the page's; the labels are then thrown away), each scanned at both labs. The rule is the
 * one of the 4 · Foundation stage: batches of 20 nuclei, two random views each (flips, rotations, either lab's scan),
 * the contrastive loss at τ = 0.2, learning rate 0.05. What a code is worth is measured as on the page: a single layer
 * trained on the frozen code with n labelled training cases of a question (n/2 per class, a seeded random subset),
 * scored on that question's 20 test nuclei.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const gen = require('./generate_nuclei.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');
const { Net, Contrastive, mulberry32, fitStandardizer } = require('../js/nn.js');

const ROOT = path.join(__dirname, '..'), SIZE = gen.SIZE, OUT = path.join(ROOT, 'data', 'foundation');
const measure = process.argv.includes('--measure'), seedArg = process.argv.includes('--seed') ? +process.argv[process.argv.indexOf('--seed') + 1] : 1;
const window = {};
for (const f of ['atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js']) new Function('window', fs.readFileSync(path.join(ROOT, 'data', f), 'utf8'))(window);
const tasks = {};
for (const [id, raw] of Object.entries(window.LECTURE_TASKS)) tasks[id] = DS.prepare(raw);
const TASK_IDS = ['atypia', 'enlargement', 'irregularity'];
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';

// ---- pretraining sets: [{ task, label, px, pxB, inkA, inkB }]
function generateSet(perTask, seed) {
  const set = [];
  for (const task of gen.TASKS) {
    gen.setSeed(seed + task.order * 1000003);
    for (let i = 0; i < perTask; i++) {
      const label = i % 2, n = gen.makeOne(label, task, i);
      const tex = gen.makeValueNoise(n.look.textureCells || 7);
      const img = gen.renderNucleus(n.shape, n.look, tex, gen.gaussian);
      const rngB = mulberry32((seed + 7919 * (set.length + 1)) >>> 0), lookB = gen.otherLabLook(n.look, rngB);
      const imgB = gen.renderNucleus(n.shape, lookB, tex, () => gen.gaussianFrom(rngB));
      const px = Uint8ClampedArray.from(img), pxB = Uint8ClampedArray.from(imgB);
      set.push({ task: task.id, label, px, pxB, inkA: NF.toInk(px), inkB: NF.toInk(pxB) });
    }
  }
  return set;
}
function pageSet(perTask) { // the first training nuclei of every question ([34, 33, 33] is the page's own set)
  const set = [];
  TASK_IDS.forEach((id, k) => tasks[id].train.slice(0, perTask[k]).forEach(s => set.push({ task: id, label: s.label, px: s.variants.A.px, pxB: s.variants.B.px, inkA: s.variants.A.ink, inkB: s.variants.B.ink })));
  return set;
}
let generatedSet = null;
const GEN_PER_TASK = 700, GEN_SEED = 20260928;
function generated() {
  if (!generatedSet) { const t0 = Date.now(); generatedSet = generateSet(GEN_PER_TASK, GEN_SEED); console.log(`generated ${generatedSet.length} nuclei in ${((Date.now() - t0) / 1000).toFixed(0)} s`); }
  return generatedSet;
}

// ---- pretraining, exactly as the page does it
function pretrain(set, { K, code, epochs, seed = 1, labs = true, log = false }) {
  const std = fitStandardizer(set.map(e => e.inkA), { perDimScale: false });
  const cl = new Contrastive({ imageSize: SIZE, conv: { K, f: 5, pool: 4 }, code, tau: 0.2, seed });
  const rng = mulberry32(seed * 31 + 7), order = set.map((_, i) => i), batch = 20, L = labs ? ['A', 'B'] : ['A'];
  const view = e => { const lab = L[Math.floor(rng() * L.length)], t = Math.floor(rng() * 8), ink = lab === 'B' ? e.inkB : e.inkA; return std.apply(t ? DS.dihedral(ink, SIZE, t) : ink); };
  const t0 = Date.now();
  for (let ep = 1; ep <= epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let loss = 0, nb = 0;
    for (let b0 = 0; b0 < order.length; b0 += batch) { const idx = order.slice(b0, b0 + batch); if (idx.length < 4) continue; loss += cl.step(idx.map(i => [view(set[i]), view(set[i])]), 0.05, 0).loss; nb++; }
    if (log && (ep === 1 || ep % Math.max(1, Math.round(epochs / 4)) === 0 || ep === epochs)) console.log(`    epoch ${String(ep).padStart(3)}: batch loss ${(loss / nb).toFixed(2)}  (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  }
  return { cl, std, msPerEpoch: (Date.now() - t0) / epochs };
}

// ---- what a code is worth: a single layer on it with n labelled cases, scored on the 20 test nuclei of a question
function probe(cl, std, id, n, subsetSeed, testLab = 'A') {
  const ds = tasks[id], rng = mulberry32(1000 + subsetSeed * 17 + n);
  const pick = (arr, k) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, k); };
  const tr = [...pick(ds.train.filter(s => s.label), n / 2), ...pick(ds.train.filter(s => !s.label), n / 2)];
  const code = (s, lab) => cl.encode(std.apply(s.variants[lab].ink));
  const rows = tr.map(s => code(s, 'A')), labels = tr.map(s => s.label), st = fitStandardizer(rows, { perDimScale: true });
  const net = new Net({ inputSize: cl.code, hidden: [], activation: 'relu', seed: 1 }), rnd = mulberry32(1), idx = rows.map((_, i) => i);
  for (let ep = 0; ep < 150; ep++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < idx.length; b += 8) { const bb = idx.slice(b, b + 8); net.trainBatch(bb.map(i => st.apply(rows[i])), bb.map(i => labels[i]), 0.1, 0); }
  }
  return net.evaluate(ds.test.map(s => st.apply(code(s, testLab))), ds.test.map(s => s.label)).accuracy;
}
const NS = [4, 10, 20, 40, 80], SUBSETS = [1, 2, 3];
function worth(cl, std, testLab = 'A') { return NS.map(n => ({ n, tasks: TASK_IDS.map(id => SUBSETS.reduce((a, s) => a + probe(cl, std, id, n, s, testLab), 0) / SUBSETS.length) })); }
const rowText = r => `    n = ${String(r.n).padStart(2)}:  ` + TASK_IDS.map((id, k) => `${id} ${pc(r.tasks[k])}`).join('  ') + `   mean ${pc(r.tasks.reduce((a, v) => a + v, 0) / 3)}`;

// ---- the candidates
const CONFIGS = [
  { id: 'page100-k4', title: 'the page’s 100 nuclei · 4 filters, code of 8 · 100 epochs', note: 'the encoder 4 · Foundation pretrains with seed 1, saved', set: () => pageSet([34, 33, 33]), K: 4, code: 8, epochs: 100 },
  { id: 'page240-k4', title: 'the page’s 240 training nuclei · 4 filters, code of 8 · 200 epochs', set: () => pageSet([80, 80, 80]), K: 4, code: 8, epochs: 200 },
  { id: 'page240-k8', title: 'the page’s 240 training nuclei · 8 filters, code of 16 · 200 epochs', note: 'a bigger encoder on more nuclei', set: () => pageSet([80, 80, 80]), K: 8, code: 16, epochs: 200 },
  { id: 'gen2100-k4', title: '2,100 generated nuclei · 4 filters, code of 8 · 60 epochs', set: generated, K: 4, code: 8, epochs: 60 },
  { id: 'gen2100-k8', title: '2,100 generated nuclei · 8 filters, code of 16 · 60 epochs', set: generated, K: 8, code: 16, epochs: 60 },
];
const SHIP = ['page100-k4', 'page240-k8']; // what the page ships: the stage's own encoder saved, and the bigger one that leads with 40 labelled cases and more (see the README)

if (require.main !== module) module.exports = { pretrain, pageSet, generateSet, probe, tasks, TASK_IDS, CONFIGS };
else if (measure) {
  console.log('What a code is worth by pretraining set, encoder size and labelled cases: a single layer on the frozen code,\nn labelled training cases per question (n/2 per class, mean of 3 random subsets), scored on the 20 test nuclei.\n');
  for (const c of CONFIGS) {
    console.log(`${c.id}: ${c.title}`);
    const set = c.set(), { cl, std, msPerEpoch } = pretrain(set, { K: c.K, code: c.code, epochs: c.epochs, seed: seedArg, log: true });
    console.log(`    ${msPerEpoch.toFixed(0)} ms per epoch · ${cl.parameterCount().toLocaleString()} weights · pretraining seed ${seedArg}`);
    for (const r of worth(cl, std)) console.log(rowText(r));
    const b = worth(cl, std, 'B').find(r => r.n === 20);
    console.log(`    n = 20, scored on the other lab’s scans: ` + TASK_IDS.map((id, k) => `${id} ${pc(b.tasks[k])}`).join('  ') + `   mean ${pc(b.tasks.reduce((a, v) => a + v, 0) / 3)}\n`);
  }
} else {
  fs.mkdirSync(OUT, { recursive: true });
  const backbones = [];
  for (const id of SHIP) {
    const c = CONFIGS.find(x => x.id === id);
    console.log(`pretraining ${c.id}: ${c.title}`);
    const set = c.set(), { cl, std } = pretrain(set, { K: c.K, code: c.code, epochs: c.epochs, log: true });
    const w = worth(cl, std), wB = worth(cl, std, 'B');
    for (const r of w) console.log(rowText(r));
    backbones.push(cl.toJSON(std, { id: c.id, title: c.title, note: c.note || '', nuclei: set.length, epochs: c.epochs, generated: c.set === generated,
      worth: Object.fromEntries(w.map(r => [r.n, r.tasks.map(v => +v.toFixed(3))])), worthOtherLab: Object.fromEntries(wB.map(r => [r.n, r.tasks.map(v => +v.toFixed(3))])) }));
  }
  const js = `// The foundation encoders shipped with the page, pretrained by tools/pretrain_backbone.js on generated nuclei the page\n// never shows (see that script). Each entry is a Contrastive encoder's weights plus its input standardiser.\nwindow.FOUNDATION_BACKBONES = ${JSON.stringify(backbones)};\n`;
  fs.writeFileSync(path.join(OUT, 'backbones.js'), js);
  console.log(`wrote ${path.join(OUT, 'backbones.js')} (${(js.length / 1024).toFixed(0)} KB)`);
  // a sample of the generated pretraining set, 100 of its nuclei at 2×, for the README (only when a shipped backbone used it)
  if (!SHIP.some(id => CONFIGS.find(x => x.id === id).set === generated)) return;
  const set = generated(), step = Math.floor(set.length / 100), scale = 2, cell = SIZE * scale, gap = 4, cols = 10, W = cols * cell + (cols + 1) * gap, H = W;
  const pix = new Uint8Array(W * H).fill(235);
  for (let k = 0; k < 100; k++) {
    const e = set[k * step], cx = gap + (k % cols) * (cell + gap), cy = gap + Math.floor(k / cols) * (cell + gap);
    for (let y = 0; y < cell; y++) for (let x = 0; x < cell; x++) pix[(cy + y) * W + cx + x] = e.px[Math.floor(y / scale) * SIZE + Math.floor(x / scale)];
  }
  fs.writeFileSync(path.join(OUT, 'contact_sheet_sample.png'), gen.encodePNG(W, H, 1, pix));
  console.log('wrote the sample contact sheet');
}
