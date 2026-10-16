#!/usr/bin/env node
/*
 * check_segment.js — trains the segmentation network of the fields in Node, as the page will: every pixel of a
 * bladder strip gets a class (the lumen above the surface, the epithelium, the stroma, or cells below the membrane
 * in nests, cords or shed singly), from masks the field data carries in its own geometry (the membrane and the nest
 * outlines, js/fields.js masksOf). The network is a tiny fully convolutional one (nn.js SegNet): 3 × 3 convolutions
 * at the pixels' own scale and, for context, 0, 1 or 2 levels of pooling below them with upsampling and a join on
 * the way back (a U-Net in miniature), trained on random crops with the per-pixel cross-entropy. It reports, per
 * epoch, the loss on the crops and on test fields, the pixel accuracy, and the intersection over union per class on
 * the test fields at full size; after training, the IoU by pattern, and what the network gets wrong where.
 *   node tools/check_segment.js [--levels 1] [--channels 8] [--epochs 20] [--crops 400] [--crop 64] [--lr 0.003]
 *                               [--batch 4] [--decay 0.0001] [--clip 1] [--seed 1] [--weighted] [--eval 10]
 *                               [--curve] [--save FILE] [--load FILE] [--sheet FILE]
 * --weighted weighs every pixel's loss by the inverse of its class's share, so that the rare nests count; --eval N
 * scores N test fields after every epoch (all 50 at the end); --save writes the trained network with its history as
 * window.SEG_NET; --load takes the measures on a saved network; --sheet writes a PNG of test fields as the image,
 * the truth and the prediction, in the page's mask colours.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { SegNet, mulberry32, gradientCheckSeg } = require('../js/nn.js');
const FL = require('../js/fields.js');
const gen = require('./generate_nuclei.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const has = k => process.argv.includes(k);
const cfg = { levels: num('--levels', 1), channels: num('--channels', 8), epochs: num('--epochs', 20), crops: num('--crops', 400), crop: num('--crop', 64), lr: num('--lr', 0.003), batch: num('--batch', 4), decay: num('--decay', 0.0001), clip: num('--clip', 1), seed: num('--seed', 1), evalN: num('--eval', 10) };
const weighted = has('--weighted'), curve = has('--curve'), save = arg('--save', null), load = arg('--load', null), sheet = arg('--sheet', null);
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(__dirname, '..', 'data', 'fields', 'fields_data.js')).LECTURE_FIELDS, W = D.meta.w, H = D.meta.h, K = FL.SEG_CLASSES.length;
const pct = v => `${Math.round(v * 100)}%`;
// every field as the page shows it (the grain added back from its seed), as ink in 0..1, with its mask
const prep = f => { const raw = FL.decodePNGNode(f.png).px, px = FL.withGrain(raw, f.grainSeed, D.meta.grain), x = new Float64Array(px.length); for (let i = 0; i < px.length; i++) x[i] = 1 - px[i] / 255; return { id: f.id, name: f.name, split: f.split, pattern: f.pattern, label: f.label, px, x, y: FL.masksOf(f, W, H) }; };
const t0 = Date.now(), train = D.train.map(prep), test = D.test.map(prep);
const share = new Float64Array(K); for (const f of train) for (const v of f.y) share[v]++; const total = share.reduce((a, b) => a + b, 0); for (let k = 0; k < K; k++) share[k] /= total;
let meanInk = 0; for (const f of train) for (const v of f.x) meanInk += v; meanInk /= train.length * W * H;
console.log(`fields: ${train.length} training, ${test.length} test, ${W} × ${H} · masks built in ${((Date.now() - t0) / 1000).toFixed(1)} s · class shares ${FL.SEG_CLASSES.map((c, k) => `${c} ${pct(share[k])}`).join(', ')} · mean ink ${meanInk.toFixed(3)}`);
const gc = gradientCheckSeg({ levels: Math.min(2, cfg.levels), H: 16, W: 16 });
console.log(`Segmentation gradient check (${Math.min(2, cfg.levels)} level${cfg.levels === 1 ? '' : 's'}): worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} gradients`);
const LOADED = load ? loadJs(load).SEG_NET : null;
const classWeights = weighted ? Array.from(share, s => 1 / Math.sqrt(s) / K) : null;
const net = LOADED ? SegNet.fromJSON(LOADED) : new SegNet({ channels: cfg.channels, levels: cfg.levels, classes: K, seed: cfg.seed, clip: cfg.clip, classWeights, norm: { mean: +meanInk.toFixed(4) } });
console.log(`network: ${net.describe()} · ${net.parameterCount()} parameters · adam lr ${cfg.lr}, batches of ${cfg.batch} crops of ${cfg.crop} × ${cfg.crop}, ${cfg.crops} crops per epoch, decay ${cfg.decay}, clip ${cfg.clip}${weighted ? ' · pixels weighted by 1/√(class share)' : ''}`);
const rng = mulberry32(cfg.seed * 7919 + 3), hist = [];
// a random crop of a random training field: the image and the mask, cut at the same place
const cropOf = () => { const f = train[Math.floor(rng() * train.length)], s = cfg.crop, x0 = Math.floor(rng() * (W - s + 1)), y0 = Math.floor(rng() * (H - s + 1)), x = new Float64Array(s * s), y = new Uint8Array(s * s); for (let r = 0; r < s; r++) { x.set(f.x.subarray((y0 + r) * W + x0, (y0 + r) * W + x0 + s), r * s); y.set(f.y.subarray((y0 + r) * W + x0, (y0 + r) * W + x0 + s), r * s); } return { x, y, H: s, W: s }; };
// the measures on whole fields: loss, pixel accuracy, IoU per class and its mean, and by pattern
const measure = docs => { const per = new Float64Array(K), perN = new Float64Array(K), byPat = {}; let loss = 0, acc = 0, meanIoU = 0, nMean = 0; const preds = [];
  for (const f of docs) { const p = net.predict(f.x, H, W), l = net.loss(p.fw, f.y), io = FL.iou(p.labels, f.y, K); loss += l.loss; acc += l.accuracy; if (io.mean != null) { meanIoU += io.mean; nMean++; } io.per.forEach((v, k) => { if (v != null) { per[k] += v; perN[k]++; } }); const b = byPat[f.pattern] || (byPat[f.pattern] = { n: 0, mean: 0, nest: 0, nestN: 0 }); b.n++; b.mean += io.mean || 0; if (io.per[3] != null) { b.nest += io.per[3]; b.nestN++; } preds.push(p.labels); }
  return { n: docs.length, loss: loss / docs.length, acc: acc / docs.length, meanIoU: nMean ? meanIoU / nMean : null, per: Array.from(per, (v, k) => (perN[k] ? v / perN[k] : null)), byPat, preds }; };
for (let epoch = 1; epoch <= (LOADED ? 0 : cfg.epochs); epoch++) {
  const t1 = Date.now(); let loss = 0, nb = 0;
  for (let b = 0; b < cfg.crops; b += cfg.batch) { const batch = []; for (let i = 0; i < cfg.batch; i++) batch.push(cropOf()); loss += net.trainBatch(batch, cfg.lr, cfg.decay); nb++; }
  const m = measure(test.slice(0, epoch === cfg.epochs ? test.length : cfg.evalN));
  hist.push({ epoch, loss: +(loss / nb).toFixed(4), testLoss: +m.loss.toFixed(4), acc: +m.acc.toFixed(4), meanIoU: m.meanIoU == null ? null : +m.meanIoU.toFixed(4), iou: m.per.map(v => (v == null ? null : +v.toFixed(4))), n: m.n });
  if (curve || epoch === cfg.epochs || epoch % 5 === 0 || epoch === 1) console.log(`epoch ${String(epoch).padStart(3)} · crop loss ${(loss / nb).toFixed(3)} · test loss ${m.loss.toFixed(3)} · pixels right ${pct(m.acc)} · mean IoU ${m.meanIoU == null ? '–' : pct(m.meanIoU)} (${FL.SEG_CLASSES.map((c, k) => `${c} ${m.per[k] == null ? '–' : pct(m.per[k])}`).join(', ')}) on ${m.n} test fields · ${((Date.now() - t1) / 1000).toFixed(1)} s`);
}
const final = measure(test);
console.log(`\ntest fields (${final.n}): loss ${final.loss.toFixed(3)} · pixels right ${pct(final.acc)} · mean IoU ${pct(final.meanIoU)} · ${FL.SEG_CLASSES.map((c, k) => `${c} ${pct(final.per[k])}`).join(' · ')}`);
console.log(`  by pattern, mean IoU and the nest IoU: ${Object.entries(final.byPat).map(([p, b]) => `${p} ${pct(b.mean / b.n)} / ${b.nestN ? pct(b.nest / b.nestN) : '–'}`).join(' · ')}`);
// what goes wrong where: the confusion between classes, as a share of each true class
const conf = Array.from({ length: K }, () => new Float64Array(K)); test.forEach((f, i) => { const p = final.preds[i]; for (let j = 0; j < f.y.length; j++) conf[f.y[j]][p[j]]++; });
console.log(`  of each true class, where the pixels went: ${FL.SEG_CLASSES.map((c, t) => { const row = conf[t], s = row.reduce((a, b) => a + b, 0); return `${c} → ${FL.SEG_CLASSES.map((d, p) => `${d[0]} ${pct(row[p] / s)}`).join(' ')}`; }).join(' · ')}`);
if (sheet) { // ten test fields: the image, the truth, the prediction, side by side, in the page's mask colours
  const COL = [[255, 255, 255], [222, 170, 200], [240, 232, 220], [60, 170, 110]], pick = test.slice(0, 10), cell = 2, gap = 4, SW = 3 * (W * cell + gap) + gap, SH = pick.length * (H * cell + gap) + gap, rgb = new Uint8Array(SW * SH * 3).fill(255);
  const paint = (row, col, fn) => { const ox = gap + col * (W * cell + gap), oy = gap + row * (H * cell + gap); for (let y = 0; y < H * cell; y++) for (let x = 0; x < W * cell; x++) { const [r, g, b] = fn(Math.floor(y / cell) * W + Math.floor(x / cell)); const o = ((oy + y) * SW + ox + x) * 3; rgb[o] = r; rgb[o + 1] = g; rgb[o + 2] = b; } };
  pick.forEach((f, i) => { const p = final.preds[i]; paint(i, 0, j => [f.px[j], f.px[j], f.px[j]]); paint(i, 1, j => COL[f.y[j]].map((c, q) => Math.round(0.55 * c + 0.45 * f.px[j]))); paint(i, 2, j => { const c = COL[p[j]], wrong = p[j] !== f.y[j]; return c.map(v => Math.round((0.55 * v + 0.45 * f.px[j]) * (wrong ? 0.45 : 1))); }); });
  fs.writeFileSync(sheet, gen.encodePNG(SW, SH, 3, rgb)); console.log(`sheet: ${sheet} (${pick.map(f => f.name).join(', ')}: the image, the truth, the prediction with wrong pixels darkened)`);
}
if (save) { const out = net.toJSON({ name: 'the segmentation network', trained: `${cfg.epochs} epochs of ${cfg.crops} crops of ${cfg.crop} × ${cfg.crop} from ${train.length} fields`, classNames: FL.SEG_CLASSES, shares: Array.from(share, v => +v.toFixed(4)), testLoss: +final.loss.toFixed(4), testAccuracy: +final.acc.toFixed(4), meanIoU: +final.meanIoU.toFixed(4), iou: final.per.map(v => +v.toFixed(4)), byPattern: Object.fromEntries(Object.entries(final.byPat).map(([p, b]) => [p, { meanIoU: +(b.mean / b.n).toFixed(4), nestIoU: b.nestN ? +(b.nest / b.nestN).toFixed(4) : null }])), hist, config: cfg }); fs.writeFileSync(save, `window.SEG_NET = ${JSON.stringify(out)};\n`); console.log(`saved ${save} (${(fs.statSync(save).size / 1024).toFixed(0)} KB)`); }
