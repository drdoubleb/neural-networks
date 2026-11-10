#!/usr/bin/env node
/*
 * check_finetune.js — the code as an input with the encoder unfrozen: recipe ⑬'s single layer on the shipped encoder's
 * code (irregularity, 100-nuclei encoder, learning rate 0.1, batch 8, 60 epochs), trained with the encoder frozen and
 * with the gradient let through into the encoder at a fraction of the head's learning rate, for every labelled-cases
 * count; test accuracy at our lab and at the other lab (what fine-tuning on one lab's scans does to the stain
 * invariance the pretraining bought). Run:  node tools/check_finetune.js [--seeds 5] [--epochs 60]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { Net, Contrastive, mulberry32 } = require('../js/nn.js');
const DS = require('../js/dataset.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const nSeeds = +arg('--seeds', 5), epochs = +arg('--epochs', 60), backboneIx = +arg('--backbone', 0);
const window = {};
for (const f of ['irregularity/nuclei_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const ds = DS.prepare(window.LECTURE_TASKS.irregularity);
const shipped = Contrastive.fromJSON(window.FOUNDATION_BACKBONES[backboneIx]);
const encoder = { encode: ink => shipped.encode(shipped.std.apply(ink)) };
const LR = 0.1, BATCH = 8, L2 = 0;
// one run: the head on the code, the encoder a fresh copy of the shipped one, trained at rate × LR (0 = frozen)
function run(labelled, rate, seed) {
  DS.assignLabs(ds, 'ours', 'ours');
  const inp = DS.buildInputs(ds, 'code', { encoder, labelled }); // the code standardiser, fitted on the frozen encoder's codes of every training nucleus
  const enc = shipped.clone(), net = new Net({ inputSize: inp.inputSize, hidden: [], activation: 'relu', seed });
  const inkA = s => shipped.std.apply(s.variants.A.ink), inkB = s => shipped.std.apply(s.variants.B.ink);
  const live = x => inp.std.apply(enc.encode(x));
  const xs = (s, lab) => (rate > 0 ? live(lab === 'B' ? inkB(s) : inkA(s)) : inp.xFor(s, lab));
  const train = inp.trainSet, ys = train.map(inp.labelOf), rnd = mulberry32(seed), idx = train.map((_, i) => i);
  const scale = inp.std.scale;
  for (let e = 1; e <= epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b0 = 0; b0 < idx.length; b0 += BATCH) {
      const b = idx.slice(b0, b0 + BATCH);
      if (rate === 0) { net.trainBatch(b.map(i => inp.xOf(train[i])), b.map(i => ys[i]), LR, L2); continue; }
      const gC = net.newGradient(), gE = enc.enc.newGradient();
      for (const i of b) {
        const s = train[i], xi = inkA(s), fwE = enc.enc.forward(xi), x = inp.std.apply(fwE.a[1]), fwC = net.forward(x);
        const dIn = net.backDense(fwC, fwC.p - ys[i], gC), dCode = new Float64Array(dIn.length); for (let d = 0; d < dIn.length; d++) dCode[d] = dIn[d] / scale[d];
        const d0 = enc.enc.backHidden(fwE, dCode, gE); enc.enc.convAccumulate(xi, fwE, d0, gE);
      }
      net.applyGradient(gC, b.length, LR, L2); enc.enc.applyGradient(gE, b.length, LR * rate, L2);
    }
  }
  const acc = (set, lab) => net.evaluate(set.map(s => xs(s, lab)), set.map(s => s.label)).accuracy;
  let drift = 0, n0 = 0; for (let i = 0; i < enc.enc.Wc.length; i++) { drift += (enc.enc.Wc[i] - shipped.enc.Wc[i]) ** 2; n0 += shipped.enc.Wc[i] ** 2; }
  return { train: acc(train, 'A'), testA: acc(ds.test, 'A'), testB: acc(ds.test, 'B'), drift: Math.sqrt(drift / n0) };
}
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';
const RATES = [0, 0.1, 0.3, 1], COUNTS = [4, 10, 20, 40, 80];
console.log(`irregularity · the code of the shipped ${shipped.describe()} · single layer, lr ${LR}, batch ${BATCH}, ${epochs} epochs · mean of ${nSeeds} seeds`);
console.log(`columns per encoder rate: train · test at our lab · test at the other lab · relative drift of the filters`);
for (const labelled of COUNTS) {
  const cells = RATES.map(rate => { const rs = Array.from({ length: nSeeds }, (_, i) => run(labelled, rate, i + 1)), m = k => rs.reduce((a, r) => a + r[k], 0) / rs.length; return `${rate === 0 ? 'frozen' : `×${rate}`}: ${pc(m('train'))} ${pc(m('testA'))} ${pc(m('testB'))} ${rate ? m('drift').toFixed(2) : '   –'}`; });
  console.log(`  ${String(labelled).padStart(2)} labelled   ${cells.join('   ')}`);
}
