#!/usr/bin/env node
/*
 * check_slides.js — trains the slide models of the 5 · Slides stage in Node, as the page does: every nucleus of a slide
 * becomes the code of a frozen foundation encoder (plus its grid position when the question needs it), optionally a
 * context layer lets the nuclei look at each other (one layer of self-attention), a small scorer weights the nuclei by
 * a softmax over the slide, their weighted average goes to a single layer, and only the slide's label is ever used.
 * Two questions: "atypical cells on the slide?" and "a focus of atypical cells?" (three atypical nuclei on every slide,
 * adjacent or scattered, so only the arrangement differs). It reports the slide accuracy on the training and test
 * slides and where the attention lands: the share of a positive slide's attention that falls on its atypical nuclei.
 *   node tools/check_slides.js [--question atypia|focus|both] [--context none|attn|dist|all] [--pooling attention|plain|both]
 *                              [--seeds 3] [--epochs 100] [--units 4] [--lr 0.05] [--batch 1] [--big]
 * --context: none (the model as shipped first), attn (one self-attention layer), dist (the same with a learned
 * distance cost in the match). --pooling plain replaces the attention pooling by a plain average.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { AttentionMIL, Contrastive, mulberry32, fitStandardizer, gradientCheckMIL, gradientCheckContext } = require('../js/nn.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const nSeeds = num('--seeds', 3), epochs = num('--epochs', 100), units = num('--units', 4), lr = num('--lr', 0.05), batch = num('--batch', 1), big = process.argv.includes('--big');
const questions = arg('--question', 'both') === 'both' ? ['atypia', 'focus'] : [arg('--question')];
const contexts = arg('--context', 'all') === 'all' ? ['none', 'attn', 'dist'] : arg('--context').split(',');
const decay = num('--decay', 0), costInit = num('--cost', 0), dk = num('--dk', 8), ffn = num('--ffn', 8), noPos = process.argv.includes('--nopos'), summary = process.argv.includes('--summary');
const poolings = arg('--pooling', 'attention') === 'both' ? [true, false] : [arg('--pooling', 'attention') !== 'plain'];
const window = {};
for (const f of ['slides/slides_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
for (const [name, gc] of [['Attention pooling', gradientCheckMIL()], ['Context layer', gradientCheckContext()]]) console.log(`${name} gradient check: worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} weights ${gc.worst < 1e-4 ? '(ok)' : '(FAILED)'}`);

const SL = window.LECTURE_SLIDES, size = SL.questions.atypia.meta.size, cl = Contrastive.fromJSON(window.FOUNDATION_BACKBONES[big ? 1 : 0]);
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';
const ink = SL.pool.map(p => NF.toInk(NF.decodeBase64(p.px)));
const codeOf = (id, t) => cl.encode(cl.std.apply(t ? DS.dihedral(ink[id], size, t) : ink[id]));
const mean = (arr, f) => arr.reduce((a, x) => a + f(x), 0) / arr.length;

// the slides of a question as the model sees them: standardised codes, the grid position when the question needs it,
// and the distances between the nuclei for the distance cost
function prepare(q) {
  const Q = SL.questions[q], meta = Q.meta, cols = meta.cols || 5, rows = meta.perSlide / cols, withPos = !!meta.positions && !noPos;
  const posOf = k => [((k % cols) - (cols - 1) / 2) / ((cols - 1) / 2), (Math.floor(k / cols) - (rows - 1) / 2) / ((rows - 1) / 2)]; // in [-1, 1]
  const cellOf = k => [k % cols, Math.floor(k / cols)];
  const raw = s => s.nuclei.map(([id, t]) => codeOf(id, t));
  const trainRaw = Q.train.map(raw), testRaw = Q.test.map(raw);
  const std = fitStandardizer([].concat(...trainRaw), { perDimScale: true }); // on every nucleus of the training slides
  const dist = Array.from({ length: meta.perSlide }, (_, i) => Float64Array.from({ length: meta.perSlide }, (_, j) => { const [ax, ay] = cellOf(i), [bx, by] = cellOf(j); return Math.hypot(ax - bx, ay - by); })); // in cells
  const mk = (s, rowsRaw) => ({ H: rowsRaw.map((r, k) => { const c = std.apply(r); return withPos ? Float64Array.from([...c, ...posOf(k)]) : c; }), y: s.label, pos: s.nuclei.map(([id]) => !!SL.pool[id].label), dist });
  const train = Q.train.map((s, i) => mk(s, trainRaw[i])), test = Q.test.map((s, i) => mk(s, testRaw[i]));
  const share = mean(train.filter(s => s.y), s => s.pos.filter(Boolean).length / s.pos.length);
  return { meta, train, test, share, D: cl.code + (withPos ? 2 : 0), withPos };
}
function run(P, attention, context, seed) {
  const mil = new AttentionMIL({ inputSize: P.D, attentionUnits: units, attention, context: context === 'none' ? null : { dk, ffn, distanceBias: context.startsWith('dist'), excludeSelf: context.endsWith('X'), costInit }, seed });
  const rng = mulberry32(seed * 31 + 7), order = P.train.map((_, i) => i), curve = [];
  const rec = e => { const tr = mil.evaluate(P.train), te = mil.evaluate(P.test); curve.push({ e, tr, te }); };
  rec(0);
  for (let e = 1; e <= epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let b = 0; b < order.length; b += batch) mil.trainBatch(order.slice(b, b + batch).map(i => P.train[i]), lr, decay);
    rec(e);
  }
  return { mil, curve };
}
console.log(`encoder: ${cl.meta.title}\n`);
for (const q of questions) {
  const P = prepare(q);
  console.log(`=== ${P.meta.question} · ${P.train.length} training and ${P.test.length} test slides of ${P.meta.perSlide} nuclei · each nucleus: its code${P.withPos ? ' + its grid position' : ''} (${P.D} numbers)`);
  console.log(`uniform attention would give the atypical nuclei ${pc(P.share)} of a positive slide's attention\n`);
  for (const attention of poolings) for (const context of contexts) {
    const label = `${context === 'none' ? 'no context' : `context: one self-attention layer${context.startsWith('dist') ? ' with a distance cost' : ''}${context.endsWith('X') ? ', over the other nuclei' : ''}`} · ${attention ? 'attention pooling' : 'plain average'}`;
    const runs = Array.from({ length: nSeeds }, (_, i) => run(P, attention, context, i + 1));
    if (summary) { const last = runs.map(r => r.curve[epochs]); console.log(`${label.padEnd(90)} train ${pc(mean(last, c => c.tr.accuracy))} · test ${pc(mean(last, c => c.te.accuracy))} (${runs.map(r => Math.round(r.curve[epochs].te.accuracy * 100)).join('/')}) · mass ${pc(mean(last, c => c.te.culpritMass))} · ${runs[0].mil.parameterCount()} params${process.argv.slice(2).join(' ').replace(/--question \w+ |--context [\w,]+ |--summary/g, '')}`); continue; }
    console.log(`${label} (${runs[0].mil.parameterCount()} parameters, lr ${lr}, batch ${batch}, ${epochs} epochs, mean of ${nSeeds} seeds)`);
    console.log('  epoch   train acc   test acc   test loss   attention on the atypical nuclei (train · test)');
    for (const e of [0, 5, 10, 20, 40, 60, 80, 100, 150, 200, 300].filter(v => v <= epochs)) {
      const at = runs.map(r => r.curve[e]);
      console.log(`  ${String(e).padStart(5)}   ${pc(mean(at, c => c.tr.accuracy)).padStart(9)}   ${pc(mean(at, c => c.te.accuracy)).padStart(8)}   ${mean(at, c => c.te.loss).toFixed(2).padStart(9)}   ${pc(mean(at, c => c.tr.culpritMass))} · ${pc(mean(at, c => c.te.culpritMass))}`);
    }
    const dc = runs.map(r => r.mil.context && r.mil.context.distanceBias ? Math.log1p(Math.exp(r.mil.context.beta)).toFixed(2) : null).filter(Boolean);
    console.log(`  per seed at the end: test ${runs.map(r => Math.round(r.curve[epochs].te.accuracy * 100)).join('/')}% · attention on the atypical nuclei ${runs.map(r => Math.round(r.curve[epochs].te.culpritMass * 100)).join('/')}%${dc.length ? ` · learned distance cost ${dc.join('/')} per cell` : ''}\n`);
  }
}
