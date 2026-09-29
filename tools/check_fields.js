#!/usr/bin/env node
/*
 * check_fields.js — what the frozen encoders make of the nuclei of the tissue fields, before any slide model is built
 * on them. Every nucleus of a field is cut out as the 32 × 32 crop around its centre, which is what the encoder will
 * read; unlike the nuclei it was pretrained on, a crop from a field holds the edges of neighbours, the membrane, or
 * stroma. Three single-layer probes on the code, trained on the training fields' nuclei and scored on the test fields':
 *   atypical vs bland   does the atypia signal survive in a crop with neighbours? (also per kind of nucleus, and the
 *                       same probe trained on the slides' pool of lone nuclei, to see whether it transfers)
 *   below vs above      how much of a nucleus's location leaks into its code through what surrounds it in the crop
 *   pattern from a bag  not here: that is the slide model's job, in the next step
 *   node tools/check_fields.js [--seeds 3] [--epochs 40] [--masked]
 * --masked: every crop masked to its nucleus by the field's segmentation, what a segment-then-encode pipeline sees
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { Net, Contrastive, mulberry32, fitStandardizer } = require('../js/nn.js');
const NF = require('../js/features.js');
const FL = require('../js/fields.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const nSeeds = +arg('--seeds', 3), epochs = +arg('--epochs', 40), masked = process.argv.includes('--masked'); // --masked: every crop masked to its nucleus by the segmentation
const window = {};
for (const f of ['fields/fields_data.js', 'slides/slides_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const F = window.LECTURE_FIELDS, meta = F.meta, SL = window.LECTURE_SLIDES, size = meta.size, pc = v => String(Math.round(v * 100)).padStart(3) + '%';

const train = FL.nucleiOf(F, F.train, { masked }), test = FL.nucleiOf(F, F.test, { masked }); // every nucleus as its crop, with its truth
const poolInk = SL.pool.map(p => NF.toInk(NF.decodeBase64(p.px))), poolTrain = SL.pool.map((p, i) => ({ ink: poolInk[i], atypical: p.label, i })).filter((_, i) => i % 4 !== 3), poolTest = SL.pool.map((p, i) => ({ ink: poolInk[i], atypical: p.label, i })).filter((_, i) => i % 4 === 3);
console.log(`${F.train.length} training and ${F.test.length} test fields · ${train.length} and ${test.length} nuclei · crops ${masked ? 'masked to the nucleus by the segmentation' : 'with the field around the nucleus'} · kinds: ${meta.kinds.map(k => `${k} ${test.filter(n => n.kind === k).length}`).join(', ')} (test)\n`);

// a single layer on the code, trained a few epochs on standardised codes, scored on held-out nuclei
function probe(trainSet, testSet, target, seed) {
  const std = fitStandardizer(trainSet.map(n => n.code), { perDimScale: true }), X = trainSet.map(n => std.apply(n.code)), Y = trainSet.map(n => target(n)), D = X[0].length;
  const net = new Net({ inputSize: D, hidden: [], seed }), rng = mulberry32(seed * 7 + 1), order = X.map((_, i) => i);
  for (let e = 0; e < epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let b = 0; b < order.length; b += 32) { const idx = order.slice(b, b + 32); net.trainBatch(idx.map(i => X[i]), idx.map(i => Y[i]), 0.05, 1e-4); }
  }
  const calls = testSet.map(n => (net.predict(std.apply(n.code)) >= 0.5 ? 1 : 0));
  return { calls, acc: calls.filter((c, i) => c === target(testSet[i])).length / testSet.length };
}
const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
for (const [k, bb] of window.FOUNDATION_BACKBONES.entries()) {
  const cl = Contrastive.fromJSON(bb), enc = ink => cl.encode(cl.std.apply(ink));
  for (const set of [train, test, poolTrain, poolTest]) for (const n of set) n.code = enc(n.ink);
  console.log(`=== encoder ${k}: ${cl.meta.title} (code of ${cl.code} numbers)`);
  // atypical vs bland, from field crops
  const runs = Array.from({ length: nSeeds }, (_, s) => probe(train, test, n => n.atypical, s + 1));
  const byKind = meta.kinds.map(kind => { const idx = test.map((n, i) => (n.kind === kind ? i : -1)).filter(i => i >= 0); return `${kind} ${pc(mean(runs.map(r => idx.filter(i => r.calls[i] === test[i].atypical).length / idx.length)))}`; });
  console.log(`  atypical vs bland, trained on the fields' nuclei:            ${pc(mean(runs.map(r => r.acc)))} of the test fields' nuclei (${byKind.join(' · ')})`);
  const epiOnly = test.filter(n => n.kind !== 'stroma'), runsEpi = Array.from({ length: nSeeds }, (_, s) => probe(train.filter(n => n.kind !== 'stroma'), epiOnly, n => n.atypical, s + 1));
  console.log(`  the same without the stromal spindle cells:                  ${pc(mean(runsEpi.map(r => r.acc)))} (${pc(mean(runsEpi.map(r => r.calls.filter((c, i) => c === 1 && epiOnly[i].atypical).length / epiOnly.filter(n => n.atypical).length)))} of the atypical, ${pc(mean(runsEpi.map(r => r.calls.filter((c, i) => c === 0 && !epiOnly[i].atypical).length / epiOnly.filter(n => !n.atypical).length)))} of the bland)`);
  // does a probe trained on lone nuclei transfer to crops with neighbours?
  const runsPool = Array.from({ length: nSeeds }, (_, s) => probe(poolTrain, poolTest, n => n.atypical, s + 1)), runsTransfer = Array.from({ length: nSeeds }, (_, s) => probe(poolTrain, epiOnly, n => n.atypical, s + 1));
  console.log(`  trained on the slides' pool of lone nuclei:                  ${pc(mean(runsPool.map(r => r.acc)))} of the pool's held-out nuclei · ${pc(mean(runsTransfer.map(r => r.acc)))} of the fields' nuclei (no stroma): what a lone-nucleus probe makes of crops with neighbours`);
  // location from the code alone: what the surroundings in the crop give away
  const runsBelow = Array.from({ length: nSeeds }, (_, s) => probe(train.filter(n => n.kind !== 'stroma'), epiOnly, n => n.below, s + 1)), baseBelow = Math.max(mean(epiOnly.map(n => n.below)), 1 - mean(epiOnly.map(n => n.below)));
  console.log(`  below vs above the membrane, from the code alone (no stroma): ${pc(mean(runsBelow.map(r => r.acc)))} (always guessing the majority: ${pc(baseBelow)}): what the crop's surroundings give away about location\n`);
}
