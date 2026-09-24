#!/usr/bin/env node
/*
 * check_training.js — trains the demo's eight lecture recipes in Node and prints their accuracies,
 * so you can verify the datasets are learnable without a browser.  node tools/check_training.js [--seeds 5]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { Net, mulberry32, gradientCheck } = require('../js/nn.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');

const nSeeds = process.argv.includes('--seeds') ? +process.argv[process.argv.indexOf('--seeds') + 1] : 3;
const window = {};
for (const f of ['leukaemia/patients_data.js', 'atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
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
  { n: '①', label: 'Leukaemia · blood count · single layer',                       task: 'leukaemia',    mode: 'features', hidden: [],     conv: null,               lr: 0.05, epochs: 60,  augment: false, l2: 0 },
  { n: '②', label: 'Leukaemia · blood count · 3 ReLU units',                       task: 'leukaemia',    mode: 'features', hidden: [3],    conv: null,               lr: 0.05, epochs: 150, augment: false, l2: 0 },
  { n: '③', label: 'Atypia · measurements · single layer',                       task: 'atypia',       mode: 'features', hidden: [],     conv: null,               lr: 0.1,  epochs: 60, augment: false, l2: 0 },
  { n: '④', label: 'Enlargement · pixels · single layer',                        task: 'enlargement',  mode: 'pixels',   hidden: [],     conv: null,               lr: 0.02, epochs: 30, augment: false, l2: 0 },
  { n: '⑤', label: 'Irregularity · pixels · single layer',                       task: 'irregularity', mode: 'pixels',   hidden: [],     conv: null,               lr: 0.01, epochs: 60, augment: false, l2: 0 },
  { n: '⑥', label: 'Irregularity · pixels · 4 ReLU + augmentation',              task: 'irregularity', mode: 'pixels',   hidden: [4],    conv: null,               lr: 0.02, epochs: 60, augment: true,  l2: 0.01 },
  { n: '⑦', label: 'Irregularity · pixels · 4 + 4 ReLU + augmentation',          task: 'irregularity', mode: 'pixels',   hidden: [4, 4], conv: null,               lr: 0.02, epochs: 60, augment: true,  l2: 0.01 },
  { n: '⑧', label: 'Irregularity · pixels · conv 4@5×5 + 4 ReLU + augmentation', task: 'irregularity', mode: 'pixels',   hidden: [4],    conv: { K: 4, f: 5, pool: 4 }, lr: 0.02, epochs: 30, augment: true, l2: 0 },
];
function run(r, seed) {
  const ds = tasks[r.task].ds;
  const inp = DS.buildInputs(ds, r.mode, { augment: r.augment });
  const net = new Net({ inputSize: inp.inputSize, imageSize: ds.size, conv: r.conv, hidden: r.hidden, activation: 'relu', seed });
  const rnd = mulberry32(seed); const idx = inp.trainX.map((_, i) => i); const batch = 8;
  const t0 = Date.now();
  for (let e = 0; e < r.epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let s = 0; s < idx.length; s += batch) { const b = idx.slice(s, s + batch); net.trainBatch(b.map(i => inp.trainX[i]), b.map(i => inp.trainY[i]), r.lr, r.l2); }
  }
  return {
    train: net.evaluate(ds.train.map(s => inp.xOf(s)), ds.train.map(s => s.label)).accuracy,
    test: net.evaluate(ds.test.map(s => inp.xOf(s)), ds.test.map(s => s.label)).accuracy,
    params: net.parameterCount(), ms: (Date.now() - t0) / r.epochs,
  };
}
console.log(`\nLecture recipes (batch 8, ReLU, mean of ${nSeeds} seeds):`);
for (const r of RECIPES) {
  const rs = Array.from({ length: nSeeds }, (_, i) => run(r, i + 1));
  const m = k => Math.round(rs.reduce((a, x) => a + x[k], 0) / rs.length * 100);
  console.log(`  ${r.n} ${r.label.padEnd(62)} params ${String(rs[0].params).padStart(6)}   train ${String(m('train')).padStart(3)}%   test ${String(m('test')).padStart(3)}% (${rs.map(x => Math.round(x.test * 100)).join('/')})   ${rs[0].ms.toFixed(0)} ms/epoch`);
}
