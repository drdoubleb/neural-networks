#!/usr/bin/env node
/*
 * check_training.js — trains the demo's ten lecture recipes in Node and prints their accuracies, then measures what
 * the other lab's scans (weaker stain) do to each image recipe, with and without stain normalisation, and what the
 * shortcut trap (positives from the other lab) does. Run:  node tools/check_training.js [--seeds 5]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { Net, Contrastive, mulberry32, gradientCheck } = require('../js/nn.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');

const nSeeds = process.argv.includes('--seeds') ? +process.argv[process.argv.indexOf('--seeds') + 1] : 3;
const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null; // 'labelled': just the labelled-cases table
const window = {};
for (const f of ['leukemia/patients_data.js', 'atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const tasks = {};
for (const [id, raw] of Object.entries(window.LECTURE_TASKS)) tasks[id] = { raw, ds: DS.prepare(raw) };

const gc = gradientCheck();
console.log(`gradient check (conv + two dense layers): ${gc.checked} gradients compared, worst relative error ${gc.worst.toExponential(1)} ${gc.worst < 1e-5 ? 'OK' : 'BAD'}`);

for (const [id, t] of Object.entries(tasks)) {
  console.log(`\nTask "${t.raw.meta.task.title}" — class means on the training set (effect size d):`);
  t.ds.featureDefs.forEach((f, i) => {
    const grp = lab => t.ds.train.filter(s => s.label === lab).map(s => s.features[i]);
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const sd = a => { const mu = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - mu) ** 2, 0) / a.length); };
    const r = grp(0), q = grp(1);
    const d = Math.abs(mean(q) - mean(r)) / Math.sqrt((sd(r) ** 2 + sd(q) ** 2) / 2);
    console.log(`  ${f.name.padEnd(22)} ${t.raw.meta.task.classes[0].name.toLowerCase().padEnd(13)} ${mean(r).toFixed(3).padStart(8)}   ${t.raw.meta.task.classes[1].name.toLowerCase().padEnd(13)} ${mean(q).toFixed(3).padStart(8)}   d = ${d.toFixed(2)}`);
  });
}

const RECIPES = [
  { n: '①', label: 'Leukemia · blood count · single layer',                       task: 'leukemia',    mode: 'features', hidden: [],     conv: null,               lr: 0.05, epochs: 60,  augment: false, l2: 0 },
  { n: '②', label: 'Leukemia · blood count · 3 ReLU units',                       task: 'leukemia',    mode: 'features', hidden: [3],    conv: null,               lr: 0.05, epochs: 150, augment: false, l2: 0 },
  { n: '③', label: 'Atypia · measurements · single layer',                       task: 'atypia',       mode: 'features', hidden: [],     conv: null,               lr: 0.1,  epochs: 60, augment: false, l2: 0 },
  { n: '④', label: 'Atypia · measurements · 8 + 8 ReLU · 25% of the training labels wrong', task: 'atypia', mode: 'features', hidden: [8, 8], conv: null, lr: 0.1, epochs: 300, augment: false, l2: 0, labelNoise: 0.25 },
  { n: '⑤', label: 'Enlargement · pixels · single layer',                        task: 'enlargement',  mode: 'pixels',   hidden: [],     conv: null,               lr: 0.02, epochs: 30, augment: false, l2: 0 },
  { n: '⑥', label: 'Irregularity · pixels · single layer',                       task: 'irregularity', mode: 'pixels',   hidden: [],     conv: null,               lr: 0.01, epochs: 60, augment: false, l2: 0 },
  { n: '⑦', label: 'Irregularity · pixels · 4 ReLU + augmentation',              task: 'irregularity', mode: 'pixels',   hidden: [4],    conv: null,               lr: 0.02, epochs: 60, augment: true,  l2: 0.01 },
  { n: '⑧', label: 'Irregularity · pixels · 4 + 4 ReLU + augmentation',          task: 'irregularity', mode: 'pixels',   hidden: [4, 4], conv: null,               lr: 0.02, epochs: 60, augment: true,  l2: 0.01 },
  { n: '⑨', label: 'Irregularity · 4 + 4 ReLU · the shortcut (irregular from the other lab)', task: 'irregularity', mode: 'pixels', hidden: [4, 4], conv: null, lr: 0.02, epochs: 60, augment: true, l2: 0.01, trainLab: 'byClass', testLab: 'byClass' },
  { n: '⑩', label: 'Irregularity · pixels · conv 4@5×5 + 4 ReLU + augmentation', task: 'irregularity', mode: 'pixels',   hidden: [4],    conv: { K: 4, f: 5, pool: 4 }, lr: 0.02, epochs: 30, augment: true, l2: 0 },
  { n: '⑫', label: 'Irregularity · the foundation code (shipped) · single layer · 10 labelled', task: 'irregularity', mode: 'code', hidden: [], conv: null, lr: 0.1, epochs: 60, augment: false, l2: 0, labelled: 10, encoder: () => shippedEncoder() },
];
// the foundation encoders behind the code input: the shipped ones (the first is the encoder 4 · Foundation pretrains
// with seed 1, saved; the second a bigger one on the page's 240 training nuclei)
const shipped = {};
function shippedEncoder(k = 0) { if (!shipped[k]) { const cl = Contrastive.fromJSON(window.FOUNDATION_BACKBONES[k]); shipped[k] = { encode: ink => cl.encode(cl.std.apply(ink)), cl }; } return shipped[k]; }
// the convolution trained on the shortcut, measured for the README but no longer a recipe
const CONV_TRAP = { n: '·', label: 'the convolution of ⑩ trained on the shortcut', task: 'irregularity', mode: 'pixels', hidden: [4], conv: { K: 4, f: 5, pool: 4 }, lr: 0.02, epochs: 30, augment: true, l2: 0, trainLab: 'byClass', testLab: 'byClass' };
// trains a recipe (its training cases from opts.trainLab, stain normalisation opts.normalize) and scores the test set
// as scanned under each of opts.testLabs, refitting nothing: the standardizer only ever sees the training rows
function run(r, seed, opts = {}) {
  const ds = tasks[r.task].ds, trainLab = opts.trainLab || r.trainLab || 'ours', normalize = opts.normalize || 'off';
  const testLabs = opts.testLabs || [r.testLab || 'ours'];
  DS.assignLabs(ds, trainLab, trainLab);
  const encoder = r.mode === 'code' ? r.encoder() : null, labelled = opts.labelled != null ? opts.labelled : (r.labelled || 0);
  const inp = DS.buildInputs(ds, r.mode, { augment: r.augment, normalize, labelNoise: r.labelNoise || 0, encoder, labelled });
  const net = new Net({ inputSize: inp.inputSize, imageSize: ds.size, conv: r.conv, hidden: r.hidden, activation: 'relu', seed });
  const rnd = mulberry32(seed); const idx = inp.trainX.map((_, i) => i); const batch = 8;
  const t0 = Date.now(); const curve = [];
  for (let e = 1; e <= r.epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let s = 0; s < idx.length; s += batch) { const b = idx.slice(s, s + batch); net.trainBatch(b.map(i => inp.trainX[i]), b.map(i => inp.trainY[i]), r.lr, r.l2); }
    if (opts.curve) { const te = net.evaluate(ds.test.map(s => inp.xOf(s)), ds.test.map(s => s.label)); curve.push({ e, teLoss: te.loss, teAcc: te.accuracy, trAcc: net.evaluate(ds.train.map(s => inp.xOf(s)), ds.train.map(inp.labelOf)).accuracy }); }
  }
  const ms = (Date.now() - t0) / r.epochs;
  const train = net.evaluate(inp.trainSet.map(s => inp.xOf(s)), inp.trainSet.map(inp.labelOf)).accuracy; // against the labels the network was given, on the labelled cases
  const tests = {};
  for (const mode of testLabs) { DS.assignLabs(ds, trainLab, mode); const ti = DS.buildInputs(ds, r.mode, { augment: r.augment, normalize, encoder, labelled }); tests[mode] = net.evaluate(ds.test.map(s => ti.xOf(s)), ds.test.map(s => s.label)).accuracy; }
  DS.assignLabs(ds, 'ours', 'ours');
  return { train, test: tests[testLabs[0]], tests, params: net.parameterCount(), ms, curve };
}
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';
const mean = (rs, f) => rs.reduce((a, x) => a + f(x), 0) / rs.length;
function everything() {
console.log(`\nLecture recipes (batch 8, ReLU, mean of ${nSeeds} seeds):`);
for (const r of RECIPES) {
  const rs = Array.from({ length: nSeeds }, (_, i) => run(r, i + 1));
  console.log(`  ${r.n} ${r.label.padEnd(66)} params ${String(rs[0].params).padStart(6)}   train ${pc(mean(rs, x => x.train))}   test ${pc(mean(rs, x => x.test))} (${rs.map(x => Math.round(x.test * 100)).join('/')})   ${rs[0].ms.toFixed(0)} ms/epoch`);
}

// the other lab: every image recipe trained at our lab, scored on the test nuclei as scanned at our lab and at the
// other lab; then retrained with the pixels stain-normalised (per lab: each lab's typical levels matched to ours;
// per image: each scan by its own levels), which only applies to pixel inputs
console.log(`\nThe other lab (weaker stain): test accuracy at our lab · at the other lab, mean of ${nSeeds} seeds`);
for (const r of RECIPES.filter(x => x.task !== 'leukemia' && !x.trainLab && !x.labelNoise)) {
  const cell = normalize => { const rs = Array.from({ length: nSeeds }, (_, i) => run(r, i + 1, { normalize, testLabs: ['ours', 'other'] })); return `${pc(mean(rs, x => x.tests.ours))} · ${pc(mean(rs, x => x.tests.other))}`; };
  const parts = [`as is ${cell('off')}`];
  if (r.mode === 'pixels') parts.push(`normalised per lab ${cell('lab')}`, `per image ${cell('image')}`);
  else if (r.mode === 'code') parts.push(`normalised per lab ${cell('lab')}`, `per image ${cell('image')}`);
  else parts.push('(the measurements are taken from the raw scan: no normalisation)');
  console.log(`  ${r.n} ${r.label.padEnd(66)} ${parts.join('   ')}`);
}

// the shortcut: recipe ⑨ (and the convolution), a network trained with the irregular nuclei from the other lab and the regular ones from ours
for (const trap of [...RECIPES.filter(x => x.trainLab === 'byClass'), CONV_TRAP]) {
  console.log(`\nThe shortcut (${trap.n} ${trap.label}), mean of ${nSeeds} seeds: test accuracy by where the test nuclei come from`);
  const labsRow = (opts, tag) => { const rs = Array.from({ length: nSeeds }, (_, i) => run(trap, i + 1, opts)); console.log(`  ${tag.padEnd(52)} ${opts.testLabs.map(t => `${t.padEnd(7)} ${pc(mean(rs, x => x.tests[t]))}`).join('   ')}`); };
  labsRow({ trainLab: 'byClass', testLabs: ['byClass', 'ours', 'other', 'mixed'] }, 'trained split by class');
  labsRow({ trainLab: 'byClass', normalize: 'lab', testLabs: ['byClass', 'ours', 'other', 'mixed'] }, 'trained split by class, normalised per lab');
  labsRow({ trainLab: 'mixed', testLabs: ['byClass', 'ours', 'other', 'mixed'] }, 'trained on both labs mixed at random');
}

// label noise: recipe ④'s test curve, which rises and then falls as the network memorises the mislabelled cases
const noisy = RECIPES.find(x => x.n === '④');
console.log(`\nLabel noise (${noisy.label}), ${nSeeds} seeds: the test accuracy over training (against the true labels)`);
{
  const rs = Array.from({ length: nSeeds }, (_, i) => run(noisy, i + 1, { curve: true }));
  const at = e => mean(rs, x => x.curve[e - 1].teAcc), lossAt = e => mean(rs, x => x.curve[e - 1].teLoss), trAt = e => mean(rs, x => x.curve[e - 1].trAcc);
  const peaks = rs.map(x => { let b = x.curve[0]; for (const h of x.curve) if (h.teAcc > b.teAcc) b = h; return b; });
  console.log(`  test accuracy at epoch 10 ${pc(at(10))} · 25 ${pc(at(25))} · 50 ${pc(at(50))} · 100 ${pc(at(100))} · 200 ${pc(at(200))} · 300 ${pc(at(300))}   (peak ${pc(mean(peaks, p => p.teAcc))} at epochs ${peaks.map(p => p.e).join('/')})`);
  console.log(`  test loss     at epoch 10 ${lossAt(10).toFixed(2)} · 25 ${lossAt(25).toFixed(2)} · 50 ${lossAt(50).toFixed(2)} · 100 ${lossAt(100).toFixed(2)} · 200 ${lossAt(200).toFixed(2)} · 300 ${lossAt(300).toFixed(2)}`);
  console.log(`  train accuracy (against the given labels; 75% is the honest ceiling) at epoch 10 ${pc(trAt(10))} · 50 ${pc(trAt(50))} · 100 ${pc(trAt(100))} · 300 ${pc(trAt(300))}`);
}

}
// how many labelled cases? Irregularity with a growing number of labelled training cases (the first n/2 of each class),
// by input: the measurements, the pixels, and the foundation code from the two shipped encoders
function labelledTable() {
console.log(`\nHow many labelled cases? Irregularity, test accuracy at our lab, mean of ${nSeeds} seeds, 60 epochs`);
{
  const base = { task: 'irregularity', hidden: [], conv: null, epochs: 60, augment: false, l2: 0 };
  const COLS = [
    { short: 'measurements', r: { ...base, mode: 'features', lr: 0.1 } },
    { short: 'pixels', r: { ...base, mode: 'pixels', lr: 0.01 } },
    { short: 'pixels 4 ReLU + aug', r: { ...base, mode: 'pixels', hidden: [4], lr: 0.02, augment: true, l2: 0.01 } },
    { short: 'code · 100 nuclei, 4 filters', r: { ...base, mode: 'code', lr: 0.1, encoder: () => shippedEncoder(0) } },
    { short: 'code · 240 nuclei, 8 filters', r: { ...base, mode: 'code', lr: 0.1, encoder: () => shippedEncoder(1) } },
  ];
  console.log('  labelled   ' + COLS.map(c => c.short.padStart(28)).join(''));
  for (const n of [4, 10, 20, 40, 80]) {
    const cells = COLS.map(c => { const rs = Array.from({ length: nSeeds }, (_, i) => run(c.r, i + 1, { labelled: n })); return pc(mean(rs, x => x.test)); });
    console.log(`  ${String(n).padStart(8)}   ` + cells.map(v => v.padStart(28)).join(''));
  }
}
}
if (only === 'labelled') labelledTable(); else { everything(); labelledTable(); }
