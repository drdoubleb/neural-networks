#!/usr/bin/env node
/*
 * check_training.js — trains the demo network on the generated dataset in Node and prints accuracies,
 * so you can verify the data is learnable (and see the pixels-vs-measurements gap) without a browser.
 *   node tools/check_training.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { TinyNet, mulberry32 } = require('../js/nn.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');

const window = {};
new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', 'nuclei_data.js'), 'utf8'))(window);
const ds = DS.prepare(window.NUCLEI_DATA);

console.log('\nPer-feature class means (train set):');
NF.FEATURES.forEach((f, i) => {
  const grp = lab => ds.train.filter(t => t.label === lab).map(t => t.features[i]);
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = a => { const mu = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - mu) ** 2, 0) / a.length); };
  const r = grp(0), q = grp(1);
  const d = Math.abs(mean(q) - mean(r)) / Math.sqrt((sd(r) ** 2 + sd(q) ** 2) / 2);
  console.log(`  ${f.name.padEnd(18)} regular ${mean(r).toFixed(3).padStart(8)} ± ${sd(r).toFixed(3)}   irregular ${mean(q).toFixed(3).padStart(8)} ± ${sd(q).toFixed(3)}   effect size d = ${d.toFixed(2)}`);
});

function run(mode, { hidden, activation = 'sigmoid', lr, batch = 8, epochs = 60, seed = 1, augment = false, l2 = 0 }) {
  const inp = DS.buildInputs(ds, mode, { augment });
  const X = inp.trainX, Y = inp.trainY;
  const Xe = ds.train.map(s => inp.xOf(s)), Ye = ds.train.map(s => s.label);
  const Xt = ds.test.map(s => inp.xOf(s)), Yt = ds.test.map(s => s.label);
  const net = new TinyNet({ inputSize: inp.inputSize, hidden, activation, seed });
  const rnd = mulberry32(seed);
  const idx = X.map((_, i) => i);
  let last;
  for (let e = 0; e < epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let s = 0; s < idx.length; s += batch) {
      const b = idx.slice(s, s + batch);
      net.trainBatch(b.map(i => X[i]), b.map(i => Y[i]), lr, l2);
    }
    last = { train: net.evaluate(Xe, Ye), test: net.evaluate(Xt, Yt) };
  }
  last.wrongTest = ds.test.filter((s, k) => (last.test.probs[k] >= 0.5 ? 1 : 0) !== s.label).map(s => `${s.name}(${s.className[0]}, p=${last.test.probs[ds.test.indexOf(s)].toFixed(2)})`);
  return last;
}

console.log('\nTraining runs (60 epochs, batch 8, mean of 3 seeds):');
const configs = [
  ['features', { hidden: 0, lr: 0.1 }],
  ['features', { hidden: 4, lr: 0.1 }],
  ['pixels', { hidden: 0, lr: 0.01 }],
  ['pixels', { hidden: 0, lr: 0.01, l2: 0.05 }],
  ['pixels', { hidden: 0, lr: 0.01, augment: true }],
  ['pixels', { hidden: 0, lr: 0.01, augment: true, l2: 0.05 }],
  ['pixels', { hidden: 4, lr: 0.02, augment: true }],
  ['pixels', { hidden: 4, lr: 0.02, augment: true, l2: 0.05 }],
  ['pixels', { hidden: 8, lr: 0.02, activation: 'relu', augment: true, l2: 0.05 }],
];
for (const [mode, cfg] of configs) {
  const accs = [];
  for (const seed of [1, 2, 3]) accs.push(run(mode, { ...cfg, seed }));
  const avg = k => (accs.reduce((a, r) => a + r[k].accuracy, 0) / accs.length * 100).toFixed(0);
  const avgL = k => (accs.reduce((a, r) => a + r[k].loss, 0) / accs.length).toFixed(3);
  const flags = `${cfg.augment ? ' +augment' : ''}${cfg.l2 ? ' +l2=' + cfg.l2 : ''}`;
  console.log(`  ${mode.padEnd(9)} hidden=${cfg.hidden} act=${(cfg.activation || 'sigmoid').padEnd(7)} lr=${String(cfg.lr).padEnd(5)}${flags.padEnd(20)} train ${avg('train').padStart(3)}% (loss ${avgL('train')})   test ${avg('test').padStart(3)}% (loss ${avgL('test')})   wrong on seed 1: ${accs[0].wrongTest.join(' ') || 'none'}`);
}
