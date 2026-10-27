#!/usr/bin/env node
/*
 * check_fields_cnn.js — the invasion question end to end: a convolutional encoder over the field at half resolution
 * (levels of two 3 × 3 convolutions with 2 × 2 pooling, two more convolutions at the bottom), so that the field
 * becomes a grid of cells, each a token of a few numbers; then either a plain average of the cells (a convolutional
 * network with global average pooling, lesson 11) or the fields' attention heads over them (lesson 17). Nothing but
 * the pixels and the field's two labels train it: no nucleus is segmented, no code pretrained, no coordinate fed, and
 * the training fields are mirrored and turned upside down at random, so that no model can read depth from the lumen
 * being at the top (on a real slide it would not be). The model itself is NN.FieldNet (js/nn.js), the one the page
 * trains; this script trains it in Node for the lessons' "Load the trained model" buttons and for the measurements.
 *   node tools/check_fields_cnn.js [--levels 3] [--channels 8] [--pooling average|attention] [--orient 4] [--epochs 60] [--seed 1]
 *                                  [--lr 0.002] [--lr-head 0.02] [--batch 4] [--out file.json] [--ship deep|shallow] [--check]
 * --orient 2 keeps the mirror only (the ablation of the upside-down flip); --ship writes the model, its history and
 * its mimic table into data/fields/fields_cnn_weights.js under that key, for the page.
 */
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const NN = require(path.join(ROOT, 'js', 'nn.js')), FL = require(path.join(ROOT, 'js', 'fields.js'));
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d), num = (k, d) => +arg(k, d);
const cfg = { channels: num('--channels', 8), levels: num('--levels', 3), pooling: arg('--pooling', 'average'), orient: num('--orient', 4), epochs: num('--epochs', 60), seed: num('--seed', 1), lr: num('--lr', 0.002), lrHead: num('--lr-head', 0.02), batch: num('--batch', 4), out: arg('--out', null), ship: arg('--ship', null), check: process.argv.includes('--check') };
if (!['average', 'attention'].includes(cfg.pooling)) { console.error(`--pooling must be average or attention, not ${cfg.pooling}`); process.exit(2); }
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(ROOT, 'data', 'fields', 'fields_data.js')).LECTURE_FIELDS, W0 = D.meta.w, H0 = D.meta.h, W = W0 >> 1, H = H0 >> 1;
const cisOf = f => (['cis', 'cisvbn', 'inv'].includes(f.pattern) ? 1 : 0), pc = v => (v == null ? '–' : `${Math.round(v * 100)}%`);

// ---- a finite-difference check of the gradient through the whole model on a small field, with the step disabled
if (cfg.check) {
  const h = 16, w = 16, rnd = NN.mulberry32(5), x = Float64Array.from({ length: h * w }, () => rnd()), y = [1, 0];
  const model = new NN.FieldNet({ channels: cfg.channels, levels: cfg.levels, pooling: cfg.pooling, seed: cfg.seed, mean: 0.4, lr: 0, lrHead: 0, decay: 0, decayHead: 0, clipEnc: 0 });
  const lossOf = () => model.forward(x, h, w).outs.reduce((a, o, k) => a + NN.bce(o.p, y[k]), 0);
  const { g, loss } = model.trainBatch([{ x, y }], h, w); if (Math.abs(loss - lossOf()) > 1e-9) { console.log(`the batch's loss ${loss} differs from the forward's ${lossOf()}`); process.exit(1); }
  const params = model.enc.layers(), eps = 1e-5; let worst = 0, checked = 0;
  for (let li = 0; li < params.length; li++) for (const key of ['W', 'b']) { const p = params[li][key], gp = g[li][key]; for (let t = 0; t < 12; t++) { const i = Math.floor(rnd() * p.length), o = p[i]; p[i] = o + eps; const lp = lossOf(); p[i] = o - eps; const lm = lossOf(); p[i] = o; const numeric = (lp - lm) / (2 * eps); if (Math.abs(numeric) > 1e-7) { checked++; worst = Math.max(worst, Math.abs(numeric - gp[i]) / (Math.abs(numeric) + Math.abs(gp[i]))); } } }
  const fw = model.forward(x, h, w), sums = fw.outs.map((o, k) => fw.evidence[k].reduce((a, v) => a + v, 0) + model.heads[k].bo), evErr = Math.max(...sums.map((s, k) => Math.abs(s - fw.outs[k].logit)));
  console.log(`gradient check through the encoder (${cfg.pooling}, ${cfg.levels} levels): worst relative error ${worst.toExponential(2)} over ${checked} weights · the cells' evidence adds up to the score within ${evErr.toExponential(1)}`); process.exit(worst < 1e-4 && evErr < 1e-9 ? 0 : 1);
}

// ---- the fields at half resolution (the ink), the labels, and the cells' truth on the model's grid
const t0 = Date.now(), model0 = new NN.FieldNet({ channels: cfg.channels, levels: cfg.levels, pooling: cfg.pooling, seed: cfg.seed }), { gh: GH, gw: GW } = model0.grid(H, W);
const prep = f => { const px = FL.withGrain(FL.decodePNGNode(f.png).px, f.grainSeed, D.meta.grain), nuclei = f.nuclei.map(arr => Object.fromEntries(D.meta.nucleus.map((k, i) => [k, arr[i]])));
  return { name: f.name, split: f.split, pattern: f.pattern, x: FL.halfInk(px, W0, H0), y: [cisOf(f), f.label], pos: FL.cellTruth(nuclei, W0, H0, GH, GW) }; };
const train = D.train.map(prep), test = D.test.map(prep); let mean = 0; for (const f of train) for (const v of f.x) mean += v; mean /= train.length * W * H; mean = +mean.toFixed(5);
const patterns = D.meta.patterns || [...new Set(test.map(f => f.pattern))].map(key => ({ key, name: key }));
console.log(`fields ${train.length} training, ${test.length} test · ${W} × ${H} px at half resolution → ${GW} × ${GH} cells · prepared in ${((Date.now() - t0) / 1000).toFixed(1)} s · mean ink ${mean}`);

// ---- the model, and its training: batches of fields, augmented, with the test measures after every epoch
const model = new NN.FieldNet({ channels: cfg.channels, levels: cfg.levels, pooling: cfg.pooling, seed: cfg.seed, mean, lr: cfg.lr, lrHead: cfg.lrHead });
console.log(`${model.describe(H, W)} · ${model.parameterCount()} parameters · orientations ${cfg.orient >= 4 ? 'mirror and upside down' : 'mirror'} · batch ${cfg.batch} · lr ${cfg.lr} (encoder, Adam) ${cfg.lrHead} (heads) · seed ${cfg.seed}`);
const rng = NN.mulberry32(cfg.seed * 31 + 7), order = train.map((_, i) => i), hist = [];
const mimicOf = ev => patterns.map(pat => { const idx = test.map((s, i) => (s.pattern === pat.key ? i : -1)).filter(i => i >= 0); return { key: pat.key, cis: idx.filter(i => ev.outputs[0].probs[i] >= 0.5).length / idx.length, inv: idx.filter(i => ev.outputs[1].probs[i] >= 0.5).length / idx.length }; });
{ const ev = model.evaluate(test, H, W), o = ev.outputs; hist.push({ epoch: 0, loss: null, testLoss: +(o[0].loss + o[1].loss).toFixed(4), acc: null, testAcc: +o[1].accuracy.toFixed(3), accCis: null, testAccCis: +o[0].accuracy.toFixed(3), mass: null, testMass: o[1].culpritMass == null ? null : +o[1].culpritMass.toFixed(3), massCis: null, testMassCis: o[0].culpritMass == null ? null : +o[0].culpritMass.toFixed(3), rows: mimicOf(ev), seconds: 0 }); } // the untrained model on the test fields, as the page records it
for (let epoch = 1; epoch <= cfg.epochs; epoch++) {
  const te = Date.now(); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  let loss = 0, steps = 0; const correct = [0, 0];
  for (let s = 0; s < order.length; s += cfg.batch) {
    const batch = order.slice(s, s + cfg.batch).map(i => train[i]), r = model.trainBatch(batch.map(f => ({ x: FL.augmentField(f.x, W, H, rng, { upsideDown: cfg.orient >= 4, fill: mean }).x, y: f.y })), H, W);
    loss += r.loss; steps++; r.probs.forEach((p, i) => p.forEach((q, k) => { if ((q >= 0.5 ? 1 : 0) === batch[i].y[k]) correct[k]++; }));
  }
  const ev = model.evaluate(test, H, W), o = ev.outputs, rows = mimicOf(ev);
  const rec = { epoch, loss: +(loss / steps).toFixed(4), testLoss: +((o[0].loss + o[1].loss)).toFixed(4), acc: +(correct[1] / train.length).toFixed(3), testAcc: +o[1].accuracy.toFixed(3), accCis: +(correct[0] / train.length).toFixed(3), testAccCis: +o[0].accuracy.toFixed(3), mass: null, testMass: o[1].culpritMass == null ? null : +o[1].culpritMass.toFixed(3), massCis: null, testMassCis: o[0].culpritMass == null ? null : +o[0].culpritMass.toFixed(3), rows, seconds: +((Date.now() - te) / 1000).toFixed(0) };
  hist.push(rec);
  console.log(`epoch ${String(epoch).padStart(2)} · loss ${rec.loss.toFixed(3)} · train (augmented) CIS ${pc(rec.accCis)} inv ${pc(rec.acc)} · test CIS ${pc(rec.testAccCis)} inv ${pc(rec.testAcc)} · inv head on atypical-below cells ${pc(rec.testMass)} · ${rows.map(r => `${r.key} ${pc(r.inv)}`).join(' ')} invasive · ${rec.seconds} s`);
}
const last = hist[hist.length - 1];
console.log(`\nmimic table at the end (test fields, ${test.length / patterns.length} per pattern):`);
console.log(`  ${'pattern'.padEnd(10)} called CIS   called invasive`);
for (const r of last.rows) console.log(`  ${r.key.padEnd(10)} ${pc(r.cis).padStart(10)}   ${pc(r.inv).padStart(15)}`);
const saved = model.toJSON({ cfg, hist, trainedAt: new Date().toISOString() });
if (cfg.out) fs.writeFileSync(cfg.out, JSON.stringify(saved), 'utf8');
if (cfg.ship) { // merge into the page's shipped file: window.FIELDS_CNN = { deep, shallow }
  const file = path.join(ROOT, 'data', 'fields', 'fields_cnn_weights.js'), cur = fs.existsSync(file) ? loadJs(file).FIELDS_CNN || {} : {}; cur[cfg.ship] = saved;
  fs.writeFileSync(file, `// The fields end to end, trained in Node by tools/check_fields_cnn.js: 'deep' is the convolutional network with global average pooling (lesson 11), 'shallow' the shallow encoder under attention (lesson 17). Each carries its history and its mimic table.\nwindow.FIELDS_CNN = ${JSON.stringify(cur)};\n`, 'utf8');
  console.log(`shipped as ${cfg.ship} in ${path.relative(ROOT, file)} (${(fs.statSync(file).size / 1024).toFixed(0)} kB)`);
}
