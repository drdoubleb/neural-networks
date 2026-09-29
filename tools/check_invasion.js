#!/usr/bin/env node
/*
 * check_invasion.js — the field model in Node, before the page has it: every nucleus of a field becomes the code of a
 * frozen foundation encoder (plus its position when the model is given one), a stack of self-attention layers lets the
 * nuclei look at each other (each head with a learned distance cost), and two attention heads over the same tokens
 * answer the two questions a report asks of the field, "carcinoma in situ?" and "invasion?", each with its own scorer,
 * weighted average and single layer. Only the field's two labels train it.
 * Four models, so that the mimic table shows which cue each one lacks:
 *   bag    the codes alone, no positions, no context                cytology only
 *   pos    the codes with their positions, no context               cytology + location
 *   ctx    the codes without positions, but the context stack       cytology + arrangement (distances live in the attention)
 *   full   positions and the context stack                          all three cues
 * For each: accuracy of both flags on the test fields, the share of each head's attention on the nuclei it should
 * weigh (the CIS head: the atypical nuclei; the invasion head: the atypical nuclei below the membrane), and the mimic
 * table: how often the test fields of each pattern are called CIS and called invasive.
 *   node tools/check_invasion.js [--models bag,pos,ctx,full] [--layers 2] [--heads 1] [--seeds 3] [--epochs 150]
 *                                [--lr 0.02] [--decay 0.001] [--units 4] [--dk 8] [--ffn 8] [--cost 1.5] [--big] [--curve]
 *                                [--crop nucleus|surroundings]
 * --crop nucleus (the default) masks every crop to its nucleus with the field's segmentation, so the code carries
 * cytology alone and location and arrangement have to come from the positions and the context; surroundings leaves
 * the field around the nucleus in the crop, which leaks both.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { AttentionMIL, Contrastive, mulberry32, fitStandardizer, gradientCheckField } = require('../js/nn.js');
const FL = require('../js/fields.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const models = arg('--models', 'bag,pos,ctx,full').split(','), layers = num('--layers', 2), heads = num('--heads', 1), nSeeds = num('--seeds', 3), epochs = num('--epochs', 150);
const lr = num('--lr', 0.02), decay = num('--decay', 0.001), units = num('--units', 4), dk = num('--dk', 8), ffn = num('--ffn', 8), costInit = num('--cost', 1.5), big = process.argv.includes('--big'), curve = process.argv.includes('--curve'), cropMode = arg('--crop', 'nucleus');
const UNIT = 16; // distances in nucleus diameters, so a cost per unit compares with the slides' cost per cell
const window = {};
for (const f of ['fields/fields_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const F = window.LECTURE_FIELDS, meta = F.meta, cl = Contrastive.fromJSON(window.FOUNDATION_BACKBONES[big ? 1 : 0]), enc = ink => cl.encode(cl.std.apply(ink));
const pc = v => String(Math.round(v * 100)).padStart(3) + '%', mean = (arr, f) => arr.reduce((a, x) => a + (f ? f(x) : x), 0) / arr.length;
{ const gc = gradientCheckField(); console.log(`Field model gradient check (2 layers × 2 heads, 2 outputs): worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} weights ${gc.worst < 1e-4 ? '(ok)' : '(FAILED)'}`); }

// ---- the fields as the model sees them: standardised codes (on the training fields' nuclei), positions in [-1, 1],
// distances in nucleus diameters, the two labels, and the two masks the attention is judged against
const FLAGS = ['CIS', 'invasion'], cisOf = f => (['cis', 'cisvbn', 'inv'].includes(f.pattern) ? 1 : 0);
function prepare(withPos) {
  const codesOf = fields => { const out = new Map(); for (const n of FL.nucleiOf(F, fields, { masked: cropMode === 'nucleus' })) { if (!out.has(n.field)) out.set(n.field, []); out.get(n.field).push(Object.assign(n, { code: enc(n.ink) })); } return out; };
  const trainMap = codesOf(F.train), testMap = codesOf(F.test);
  const std = fitStandardizer([].concat(...[...trainMap.values()].map(ns => ns.map(n => n.code))), { perDimScale: true });
  const mk = (f, ns) => ({
    H: ns.map(n => { const c = std.apply(n.code); return withPos ? Float64Array.from([...c, n.x / meta.w * 2 - 1, n.y / meta.h * 2 - 1]) : c; }),
    dist: ns.map(a => Float64Array.from(ns, b => Math.hypot(a.x - b.x, a.y - b.y) / UNIT)),
    y: [cisOf(f), f.label], pos: [ns.map(n => !!n.atypical), ns.map(n => !!(n.atypical && n.below))], pattern: f.pattern, name: f.name });
  return { train: [...trainMap].map(([f, ns]) => mk(f, ns)), test: [...testMap].map(([f, ns]) => mk(f, ns)), D: cl.code + (withPos ? 2 : 0) };
}
const MODELS = {
  bag: { name: 'bag of codes: no positions, no context', pos: false, ctx: false },
  pos: { name: 'codes with positions, no context', pos: true, ctx: false },
  ctx: { name: 'context without positions (distances inside the attention)', pos: false, ctx: true },
  full: { name: 'positions and context', pos: true, ctx: true },
};
function run(P, M, seed) {
  const mil = new AttentionMIL({ inputSize: P.D, attentionUnits: units, outputs: 2, context: M.ctx ? { dk, ffn, heads, layers, distanceBias: true, excludeSelf: true, costInit } : null, seed });
  const rng = mulberry32(seed * 31 + 7), order = P.train.map((_, i) => i), hist = [];
  const rec = e => { const tr = mil.evaluate(P.train), te = mil.evaluate(P.test); hist.push({ e, tr, te }); };
  rec(0);
  for (let e = 1; e <= epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (const i of order) mil.trainBatch([P.train[i]], lr, decay);
    if (curve ? (e % 10 === 0 || e === epochs) : e === epochs) rec(e);
  }
  return { mil, hist };
}
console.log(`encoder: ${cl.meta.title} · crops ${cropMode === 'nucleus' ? 'masked to the nucleus by the segmentation' : 'with the field around the nucleus'} · ${F.train.length} training and ${F.test.length} test fields · ${meta.patterns.map(p => p.key).join(', ')}\n`);
const prepared = {};
for (const key of models) {
  const M = MODELS[key]; if (!M) { console.log(`unknown model ${key}`); continue; }
  const P = prepared[M.pos] || (prepared[M.pos] = prepare(M.pos)), t0 = Date.now();
  const runs = Array.from({ length: nSeeds }, (_, i) => run(P, M, i + 1)), last = runs.map(r => r.hist[r.hist.length - 1]), mil = runs[0].mil;
  console.log(`=== ${key}: ${M.name}\n    ${mil.describe()} · ${mil.parameterCount()} parameters · each nucleus: its code${M.pos ? ' + its position' : ''} (${P.D} numbers) · lr ${lr}, decay ${decay}, ${epochs} epochs, mean of ${nSeeds} seeds, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  if (curve) { console.log('    epoch   train CIS · invasion   test CIS · invasion   test loss'); for (const h of runs[0].hist) { const at = runs.map(r => r.hist.find(x => x.e === h.e)); console.log(`    ${String(h.e).padStart(5)}   ${pc(mean(at, x => x.tr.outputs[0].accuracy))} · ${pc(mean(at, x => x.tr.outputs[1].accuracy))}       ${pc(mean(at, x => x.te.outputs[0].accuracy))} · ${pc(mean(at, x => x.te.outputs[1].accuracy))}      ${mean(at, x => x.te.outputs[0].loss + x.te.outputs[1].loss).toFixed(2)}`); } }
  FLAGS.forEach((flag, k) => console.log(`    ${flag.padEnd(9)} test accuracy ${pc(mean(last, x => x.te.outputs[k].accuracy))} (${last.map(x => Math.round(x.te.outputs[k].accuracy * 100)).join('/')}) · training ${pc(mean(last, x => x.tr.outputs[k].accuracy))} · its head's attention on the ${k ? 'atypical nuclei below the membrane' : 'atypical nuclei'} of a positive test field ${pc(mean(last, x => x.te.outputs[k].culpritMass))}`));
  if (M.ctx) console.log(`    learned distance cost per nucleus diameter: ${runs.map(r => r.mil.layers.map((c, l) => `layer ${l + 1} ${Array.from(c.beta, b => Math.log1p(Math.exp(b)).toFixed(2)).join('/')}`).join(', ')).join(' · ')}`);
  // the mimic table: the test fields of each pattern, how often called CIS and called invasive
  console.log(`    ${'pattern'.padEnd(46)} called CIS   called invasive   (of ${P.test.filter(s => s.pattern === meta.patterns[0].key).length} test fields × ${nSeeds} seeds)`);
  for (const pat of meta.patterns) {
    const idx = P.test.map((s, i) => (s.pattern === pat.key ? i : -1)).filter(i => i >= 0);
    const rate = k => mean(last, x => mean(idx, i => (x.te.outputs[k].probs[i] >= 0.5 ? 1 : 0)));
    console.log(`    ${pat.name.padEnd(46)} ${pc(rate(0))}         ${pc(rate(1))}           truth: CIS ${['cis', 'cisvbn', 'inv'].includes(pat.key) ? 'yes' : 'no '} · invasive ${pat.label ? 'yes' : 'no'}`);
  }
  console.log('');
}
