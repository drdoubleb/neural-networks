#!/usr/bin/env node
/*
 * check_slides.js — trains the attention model of the 5 · Slides stage in Node, as the page does: every nucleus of a
 * slide becomes the code of a frozen foundation encoder, a small scorer weights the nuclei by a softmax over the
 * slide, their weighted average goes to a single layer, and only the slide's label is ever used. It reports the
 * slide accuracy on the training and test slides and where the attention lands: the share of a positive slide's
 * attention that falls on its atypical nuclei (uniform weights give their share of the slide, about 15%). Then the
 * same with attention off, a plain average of the nuclei.
 *   node tools/check_slides.js [--seeds 3] [--epochs 100] [--units 4] [--lr 0.05] [--batch 1] [--big]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { AttentionMIL, Contrastive, mulberry32, fitStandardizer, gradientCheckMIL } = require('../js/nn.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');
const arg = (k, d) => (process.argv.includes(k) ? +process.argv[process.argv.indexOf(k) + 1] : d);
const nSeeds = arg('--seeds', 3), epochs = arg('--epochs', 100), units = arg('--units', 4), lr = arg('--lr', 0.05), batch = arg('--batch', 1), big = process.argv.includes('--big');
const window = {};
for (const f of ['slides/slides_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const gc = gradientCheckMIL();
console.log(`Attention gradient check: worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} weights ${gc.worst < 1e-4 ? '(ok)' : '(FAILED)'}`);

const SL = window.LECTURE_SLIDES, size = SL.meta.size, cl = Contrastive.fromJSON(window.FOUNDATION_BACKBONES[big ? 1 : 0]);
console.log(`encoder: ${cl.meta.title} · ${SL.train.length} training and ${SL.test.length} test slides of ${SL.meta.perSlide} nuclei`);
// every nucleus of every slide, in its orientation, through the frozen encoder
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';
const ink = SL.pool.map(p => NF.toInk(NF.decodeBase64(p.px)));
const codeOf = (id, t) => cl.encode(cl.std.apply(t ? DS.dihedral(ink[id], size, t) : ink[id]));
const raw = s => s.nuclei.map(([id, t]) => codeOf(id, t));
const trainRaw = SL.train.map(raw), testRaw = SL.test.map(raw);
const std = fitStandardizer([].concat(...trainRaw), { perDimScale: true }); // on every nucleus of the training slides
const mk = (s, rows) => ({ H: rows.map(r => std.apply(r)), y: s.label, pos: s.nuclei.map(([id]) => !!SL.pool[id].label) });
const train = SL.train.map((s, i) => mk(s, trainRaw[i])), test = SL.test.map((s, i) => mk(s, testRaw[i]));
const share = train.filter(s => s.y).reduce((a, s) => a + s.pos.filter(Boolean).length / s.pos.length, 0) / train.filter(s => s.y).length;
console.log(`uniform attention would give the atypical nuclei ${pc(share)} of a positive slide's attention\n`);

function run(attention, seed) {
  const mil = new AttentionMIL({ inputSize: cl.code, attentionUnits: units, attention, seed });
  const rng = mulberry32(seed * 31 + 7), order = train.map((_, i) => i), curve = [];
  const rec = e => { const tr = mil.evaluate(train), te = mil.evaluate(test); curve.push({ e, tr, te }); };
  rec(0);
  for (let e = 1; e <= epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let b = 0; b < order.length; b += batch) mil.trainBatch(order.slice(b, b + batch).map(i => train[i]), lr, 0);
    rec(e);
  }
  return { mil, curve };
}
const mean = (arr, f) => arr.reduce((a, x) => a + f(x), 0) / arr.length;
for (const attention of [true, false]) {
  console.log(`${attention ? 'Attention' : 'Plain average'} (${units} attention units, lr ${lr}, batch ${batch} slide${batch > 1 ? 's' : ''}, ${epochs} epochs, mean of ${nSeeds} seeds)`);
  const runs = Array.from({ length: nSeeds }, (_, i) => run(attention, i + 1));
  console.log('  epoch   train acc   test acc   test loss   attention on the atypical nuclei (train · test)');
  for (const e of [0, 5, 10, 20, 40, 60, 80, 100, 150, 200, 300].filter(v => v <= epochs)) {
    const at = runs.map(r => r.curve[e]);
    console.log(`  ${String(e).padStart(5)}   ${pc(mean(at, c => c.tr.accuracy)).padStart(9)}   ${pc(mean(at, c => c.te.accuracy)).padStart(8)}   ${mean(at, c => c.te.loss).toFixed(2).padStart(9)}   ${pc(mean(at, c => c.tr.culpritMass))} · ${pc(mean(at, c => c.te.culpritMass))}`);
  }
  console.log(`  per seed at the end: test ${runs.map(r => Math.round(r.curve[epochs].te.accuracy * 100)).join('/')}% · attention on the atypical nuclei ${runs.map(r => Math.round(r.curve[epochs].te.culpritMass * 100)).join('/')}%\n`);
}
