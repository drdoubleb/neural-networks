#!/usr/bin/env node
/*
 * probe_fields_cnn.js — an end-to-end field model saved by check_fields_cnn.js (--out, or a key of the shipped file)
 * re-evaluated from its weights: accuracy on both sets, the mimic table, the heads' shares on the cells holding atypical
 * nuclei, how spread the attention is (the effective number of cells, 1 / Σ a²), the best epoch on the curve, and the
 * invasion head's map over the cells of the first invasive and the first CIS-into-nests test field (its attention, or
 * its evidence for a model with a plain average).
 *   node tools/probe_fields_cnn.js run.json [more.json ...]
 *   node tools/probe_fields_cnn.js data/fields/fields_cnn_weights.js:deep data/fields/fields_cnn_weights.js:shallow
 */
'use strict';
const fs = require('fs'), path = require('path'); const ROOT = path.join(__dirname, '..');
const NN = require(path.join(ROOT, 'js', 'nn.js')), FL = require(path.join(ROOT, 'js', 'fields.js'));
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(ROOT, 'data', 'fields', 'fields_data.js')).LECTURE_FIELDS, W0 = D.meta.w, H0 = D.meta.h, W = W0 >> 1, H = H0 >> 1;
const cisOf = f => (['cis', 'cisvbn', 'inv'].includes(f.pattern) ? 1 : 0), pc = v => (v == null ? '–' : `${Math.round(v * 100)}%`);
const fields = D.train.concat(D.test).map(f => ({ name: f.name, split: f.split, pattern: f.pattern, x: FL.halfInk(FL.withGrain(FL.decodePNGNode(f.png).px, f.grainSeed, D.meta.grain), W0, H0), y: [cisOf(f), f.label], nuclei: f.nuclei.map(arr => Object.fromEntries(D.meta.nucleus.map((k, i) => [k, arr[i]]))) }));
const readModel = spec => { const [file, key] = spec.split(':'); if (!fs.existsSync(file)) return null; const R = key ? loadJs(file).FIELDS_CNN[key] : JSON.parse(fs.readFileSync(file, 'utf8')); return R && R.kind === 'fieldnet' ? R : null; };
for (const spec of process.argv.slice(2)) {
  const R = readModel(spec); if (!R) { console.log(`${spec}: not found, or not a model of check_fields_cnn.js`); continue; }
  const model = NN.FieldNet.fromJSON(R), { gh: GH, gw: GW } = model.grid(H, W), samples = fields.map(f => Object.assign({ pos: FL.cellTruth(f.nuclei, W0, H0, GH, GW) }, f));
  const tr = samples.filter(f => f.split === 'train'), te = samples.filter(f => f.split === 'test'), evR = model.evaluate(tr, H, W), evT = model.evaluate(te, H, W), o = evT.outputs;
  console.log(`${spec}: ${model.describe(H, W)} · ${model.parameterCount()} parameters · seed ${R.seed} · ${R.hist ? R.hist[R.hist.length - 1].epoch : '?'} epochs · training CIS ${pc(evR.outputs[0].accuracy)} inv ${pc(evR.outputs[1].accuracy)} · test CIS ${pc(o[0].accuracy)} inv ${pc(o[1].accuracy)} · CIS head on the cells with atypical nuclei ${pc(o[0].culpritMass)} · invasion head on the cells with atypical nuclei below the membrane ${pc(o[1].culpritMass)}`);
  const eff = k => o[k].weights.reduce((s, a) => s + 1 / a.reduce((q, v) => q + v * v, 0), 0) / o[k].weights.length; console.log(`  attention spread, effective number of cells of ${GW * GH}: CIS head ${eff(0).toFixed(1)} · invasion head ${eff(1).toFixed(1)}${model.pooling === 'average' ? ' (a plain average: uniform by construction)' : ''}`);
  console.log(`  ${'pattern'.padEnd(18)} called CIS   called invasive`);
  for (const pat of D.meta.patterns) { const idx = te.map((s, i) => (s.pattern === pat.key ? i : -1)).filter(i => i >= 0); console.log(`  ${pat.short.padEnd(18)} ${pc(idx.filter(i => o[0].probs[i] >= 0.5).length / idx.length).padStart(10)}   ${pc(idx.filter(i => o[1].probs[i] >= 0.5).length / idx.length).padStart(15)}`); }
  if (R.hist) { const best = R.hist.reduce((a, h) => (h.testAcc > a.testAcc ? h : a), R.hist[0]); console.log(`  best test invasion accuracy on the curve: ${pc(best.testAcc)} at epoch ${best.epoch}; last ${pc(R.hist[R.hist.length - 1].testAcc)}`); }
  // the invasion head's map over the first invasive and the first mimic test field: the attention, or the evidence for a plain average
  for (const key of ['inv', 'cisvbn']) { const i = te.findIndex(s => s.pattern === key), s = te[i], fw = model.forward(s.x, H, W), att = model.pooling === 'attention', a = att ? fw.outs[1].a : fw.evidence[1]; let amax = 1e-9; for (const v of a) amax = Math.max(amax, Math.abs(v)); const rows = [];
    for (let r = 0; r < GH; r++) { let line = ''; for (let c = 0; c < GW; c++) { const k = r * GW + c, v = a[k] / amax; line += (s.pos[1][k] ? '[' : ' ') + (att ? (v > 0.66 ? '█' : v > 0.33 ? '▓' : v > 0.1 ? '░' : '·') : (v > 0.5 ? '█' : v > 0.15 ? '▓' : v < -0.5 ? '▄' : v < -0.15 ? '▁' : '·')) + (s.pos[1][k] ? ']' : ' '); } rows.push('    ' + line); }
    console.log(`  invasion head on ${s.name} (${key}, called ${fw.outs[1].p >= 0.5 ? 'invasive' : 'not invasive'}, P ${fw.outs[1].p.toFixed(2)}; ${att ? 'its attention' : 'its evidence, █ ▓ towards invasive, ▄ ▁ against'}; [ ] marks cells with an atypical nucleus below the membrane):\n${rows.join('\n')}`); }
}
