#!/usr/bin/env node
/*
 * check_describe.js — trains the Describe model in Node, as the page will: every nucleus's image goes through the
 * foundation encoder that ships with the page (frozen), its code of 8 numbers is projected into the embedding space
 * as the first token, and the language model of the Reports question writes the description word by word, trained
 * on the next word alone. It reports, per epoch, the loss on the training and test nuclei and the test loss by
 * section (the image token predicts the first word; the description; the impression), where the description looks
 * (the share of its attention on the image token), and, from the test nuclei written from their image alone, how
 * many of the four traits come out right; after training, each trait's accuracy on the test nuclei, the impression,
 * what the model writes with no image (a code of zeros, and the mean code), the same nuclei as scanned at the other
 * lab, and the seven other views (flips and rotations) of each test nucleus.
 *   node tools/check_describe.js [--dim 48] [--layers 2] [--heads 2] [--dk 12] [--ffn 48] [--cost 0.05,0.003]
 *                                [--positions 32] [--epochs 40] [--lr 0.005] [--batch 8] [--decay 0.0001] [--clip 1]
 *                                [--seed 1] [--curve] [--ground 80] [--ground-every 1] [--backbone ID]
 *                                [--no-code] [--shuffle-codes] [--skip-probes] [--save FILE] [--load FILE]
 * --no-code trains with the image token as a plain word (no code projected): what the words alone give, the base
 * rates; --shuffle-codes gives every training nucleus another nucleus's code: the same, by another route; --save
 * writes the trained model with its vocabulary, its history and its measures as window.DESCRIBE_LM; --load takes
 * the measures on a saved model.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { LanguageModel, Contrastive, mulberry32, gradientCheckLM, wordTrails } = require('../js/nn.js');
const DSC = require('../js/describe.js');
const NF = require('../js/features.js');
const DS = require('../js/dataset.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const has = k => process.argv.includes(k);
const cfg = { dim: num('--dim', 48), layers: num('--layers', 2), heads: num('--heads', 2), dk: num('--dk', 12), ffn: num('--ffn', 48), cost: String(arg('--cost', '0.05,0.003')).split(',').map(Number), positions: num('--positions', 32), epochs: num('--epochs', 40), lr: num('--lr', 0.005), batch: num('--batch', 8), decay: num('--decay', 0.0001), clip: num('--clip', 1), seed: num('--seed', 1), ground: num('--ground', 80), groundEvery: num('--ground-every', 1) };
const curve = has('--curve'), noCode = has('--no-code'), shuffleCodes = has('--shuffle-codes'), skipProbes = has('--skip-probes'), save = arg('--save', null), load = arg('--load', null);
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(__dirname, '..', 'data', 'describe', 'describe_data.js')).LECTURE_DESCRIBE;
const backbones = loadJs(path.join(__dirname, '..', 'data', 'foundation', 'backbones.js')).FOUNDATION_BACKBONES;
const backbone = backbones.find(b => b.id === arg('--backbone', backbones[0].id)) || backbones[0];
const cl = Contrastive.fromJSON(backbone), SIZE = D.meta.size;
const pct = v => `${Math.round(v * 100)}%`;
// the code of a scan: pixels → ink → the backbone's standardiser → the encoder; and of its other views
const inkOf = px64 => NF.toInk(NF.decodeBase64(px64));
const codeOf = ink => cl.encode(cl.std.apply(ink));
const LOADED = load ? loadJs(load).DESCRIBE_LM : null;
const vocab = LOADED ? DSC.vocabFrom(LOADED.words) : DSC.buildVocab(D.nuclei.filter(n => n.split === 'train').map(n => n.text));
const prep = n => { const tokens = DSC.encode(vocab, n.text), words = DSC.decode(vocab, tokens), ink = inkOf(n.px); return { id: n.id, name: n.name, split: n.split, traits: n.traits, text: n.text, tokens, words, sections: DSC.sectionsOf(words), ink, code: noCode ? null : codeOf(ink), codeB: n.pxB ? codeOf(inkOf(n.pxB)) : null }; };
const train = D.nuclei.filter(n => n.split === 'train').map(prep), test = D.nuclei.filter(n => n.split === 'test').map(prep);
if (shuffleCodes) { const r = mulberry32(cfg.seed * 31 + 7), codes = train.map(d => d.code); for (let i = codes.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [codes[i], codes[j]] = [codes[j], codes[i]]; } train.forEach((d, i) => { d.code = codes[i]; }); }
const gc = gradientCheckLM({ positions: 4, codeDim: 3 });
console.log(`Language model gradient check with an image code: worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} gradients`);
console.log(`corpus: ${train.length} training nuclei, ${test.length} test · vocabulary ${vocab.size} · ${(train.reduce((a, d) => a + d.tokens.length, 0) / train.length).toFixed(1)} tokens per description · the image token first, then the description, then the impression`);
console.log(`encoder: ${backbone.id} (${cl.describe()}), frozen · codes ${noCode ? 'not given (the image token is a plain word)' : shuffleCodes ? 'shuffled across the training nuclei' : 'from the scans'}`);
const lm = LOADED ? LanguageModel.fromJSON(LOADED) : new LanguageModel({ vocabSize: vocab.size, dim: cfg.dim, layers: cfg.layers, heads: cfg.heads, dk: cfg.dk, ffn: cfg.ffn, costInit: cfg.cost.length === cfg.heads ? cfg.cost : cfg.cost[0], positions: cfg.positions, clip: cfg.clip, optimizer: 'adam', seed: cfg.seed, codeDim: noCode ? 0 : cl.code });
console.log(`model: dim ${lm.D}, ${lm.layers.length} layers × ${lm.H} heads (dk ${lm.dk}, ffn ${lm.F}), ${lm.P} learned positions, code of ${lm.C || 'none'} · adam lr ${cfg.lr}, batch ${cfg.batch}, decay ${cfg.decay}, clip ${cfg.clip}`);
console.log(`\n== ${LOADED ? `loaded ${load} (${LOADED.trained})` : `seed ${cfg.seed}`} · ${lm.parameterCount()} parameters · ${lm.describe()}`);
const rng = mulberry32(cfg.seed * 977 + 1), order = train.map((_, i) => i), hist = [], snaps = [{ epoch: 0, E: Float32Array.from(lm.E), out: Float32Array.from(lm.Wout) }];
// the description written from the image alone, greedily, and what it says about the traits
const write = (d, code, temperature = 0) => { const g = lm.generate([vocab.img], { maxTokens: 40, temperature, rng, stop: vocab.end, code: code === undefined ? d.code : code }); return DSC.decode(vocab, [vocab.img].concat(g.tokens)); };
const measure = (docs, codeOfDoc) => { const n = docs.length, right = Object.fromEntries(DSC.TRAITS.map(k => [k, 0])), said = Object.fromEntries(DSC.TRAITS.map(k => [k, { yes: 0, no: 0, unsaid: 0 }])); let all = 0, impression = 0; for (const d of docs) { const words = write(d, codeOfDoc ? codeOfDoc(d) : undefined), r = DSC.traitsRight(d.traits, words); if (r.all) all++; if (r.impressionRight) impression++; for (const k of DSC.TRAITS) { if (r.right[k]) right[k]++; said[k][r.said[k] === true ? 'yes' : r.said[k] === false ? 'no' : 'unsaid']++; } } return { n, all: all / n, impression: impression / n, traits: Object.fromEntries(DSC.TRAITS.map(k => [k, right[k] / n])), said }; };
const bySection = (docs, per) => { const sum = {}, cnt = {}; docs.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const s = d.sections[i + 1]; sum[s] = (sum[s] || 0) + per[k][i]; cnt[s] = (cnt[s] || 0) + 1; } }); return Object.fromEntries(DSC.SECTIONS.map(s => [s, cnt[s] ? sum[s] / cnt[s] : null])); };
const evidence = docs => { let s = 0, n = 0; for (const d of docs) { const v = DSC.imageShare(lm.forward(d.tokens, d.code), d.sections); if (v != null) { s += v; n++; } } return n ? s / n : null; };
for (let epoch = 1; epoch <= (LOADED ? 0 : cfg.epochs); epoch++) {
  const t0 = Date.now(); let loss = 0, nb = 0;
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  for (let b = 0; b < order.length; b += cfg.batch) { loss += lm.trainBatch(order.slice(b, b + cfg.batch).map(i => train[i]), cfg.lr, cfg.decay); nb++; }
  const ev = lm.evaluate(test), sec = bySection(test, ev.per), ev2 = evidence(test.slice(0, 40));
  let ground = null, m = null; if (cfg.ground && (epoch % cfg.groundEvery === 0 || epoch === cfg.epochs)) { m = measure(test.slice(0, cfg.ground)); ground = m.all; }
  hist.push(Object.assign({ epoch, loss: +(loss / nb).toFixed(4), testLoss: +ev.loss.toFixed(4), acc: +ev.accuracy.toFixed(4), ground, evidence: ev2 == null ? null : +ev2.toFixed(4), traitsRight: m ? Object.fromEntries(DSC.TRAITS.map(k => [k, +m.traits[k].toFixed(4)])) : null }, Object.fromEntries(DSC.SECTIONS.map(s => [s, sec[s] == null ? null : +sec[s].toFixed(4)]))));
  snaps.push({ epoch, E: Float32Array.from(lm.E), out: Float32Array.from(lm.Wout) });
  if (curve || epoch === cfg.epochs || epoch % 5 === 0 || epoch === 1) console.log(`epoch ${String(epoch).padStart(3)} · train loss ${(loss / nb).toFixed(3)} · test loss ${ev.loss.toFixed(3)} · next word right ${pct(ev.accuracy)} · description on the image ${ev2 == null ? '–' : pct(ev2)} · by section: ${DSC.SECTIONS.map(s => `${s} ${sec[s] == null ? '–' : sec[s].toFixed(2)}`).join(', ')}${m ? ` · from the image: all four traits right ${pct(m.all)} (${DSC.TRAITS.map(k => `${k} ${pct(m.traits[k])}`).join(', ')}), impression ${pct(m.impression)}` : ''} · ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
let measures = null;
if (!skipProbes) {
  const t1 = Date.now();
  const own = measure(test);
  console.log(`\ntest nuclei, written from the image alone: all four traits right ${pct(own.all)} · ${DSC.TRAITS.map(k => `${k} ${pct(own.traits[k])}`).join(' · ')} · impression right ${pct(own.impression)}`);
  console.log(`  what it said, per trait (yes / no / unsaid): ${DSC.TRAITS.map(k => `${k} ${own.said[k].yes}/${own.said[k].no}/${own.said[k].unsaid}`).join(' · ')} · truth on: ${DSC.TRAITS.map(k => `${k} ${test.filter(d => d.traits[k]).length}`).join(', ')}`);
  for (const d of test.slice(0, 6)) console.log(`  ${d.name} ${DSC.TRAITS.map(k => (d.traits[k] ? k[0].toUpperCase() : '·')).join('')}: wrote "${DSC.detokenize(write(d).slice(1).filter(w => w !== DSC.END))}" · truth "${d.text}"`);
  const zeros = new Float64Array(cl.code), mean = new Float64Array(cl.code); for (const d of train) if (d.code) for (let c = 0; c < cl.code; c++) mean[c] += d.code[c] / train.length;
  const base = { all: 0, traits: Object.fromEntries(DSC.TRAITS.map(k => [k, 0])) }; // the base rate of each trait's commoner state on the test nuclei, what a description that ignores the image can reach at best
  for (const k of DSC.TRAITS) { const on = test.filter(d => d.traits[k]).length; base.traits[k] = Math.max(on, test.length - on) / test.length; }
  if (lm.C) {
    const noImage = measure(test, () => zeros), meanImage = measure(test, () => mean);
    console.log(`\nno image (a code of zeros): writes "${DSC.detokenize(write(test[0], zeros).slice(1).filter(w => w !== DSC.END))}" · traits right ${DSC.TRAITS.map(k => `${k} ${pct(noImage.traits[k])}`).join(', ')} · all four ${pct(noImage.all)}`);
    console.log(`the mean code: writes "${DSC.detokenize(write(test[0], mean).slice(1).filter(w => w !== DSC.END))}" · traits right ${DSC.TRAITS.map(k => `${k} ${pct(meanImage.traits[k])}`).join(', ')} · all four ${pct(meanImage.all)} · (the commoner state of each trait on the test nuclei: ${DSC.TRAITS.map(k => `${k} ${pct(base.traits[k])}`).join(', ')})`);
    const other = measure(test, d => d.codeB); let agree = 0, darkFlip = 0;
    for (const d of test) { const a = DSC.traitsOf(write(d)), b = DSC.traitsOf(write(d, d.codeB)); if (DSC.TRAITS.every(k => a[k] === b[k])) agree++; if (a.dark === true && b.dark === false) darkFlip++; }
    console.log(`the other lab's scans (weaker stain): traits right ${DSC.TRAITS.map(k => `${k} ${pct(other.traits[k])}`).join(', ')} · all four ${pct(other.all)} · the same four traits as from our lab's scan for ${pct(agree / test.length)} of the nuclei · said hyperchromatic at our lab but not at the other for ${darkFlip} of ${test.filter(d => d.traits.dark).length} hyperchromatic nuclei`);
    let viewsSame = 0, viewsN = 0, viewsRight = 0; const viewTraits = Object.fromEntries(DSC.TRAITS.map(k => [k, 0]));
    for (const d of test) { const upright = DSC.traitsOf(write(d)); for (let t = 1; t < 8; t++) { const code = codeOf(DS.dihedral(d.ink, SIZE, t)), said = DSC.traitsOf(write(d, code)), r = DSC.traitsRight(d.traits, write(d, code)); viewsN++; if (DSC.TRAITS.every(k => said[k] === upright[k])) viewsSame++; if (r.all) viewsRight++; for (const k of DSC.TRAITS) if (r.right[k]) viewTraits[k]++; } }
    console.log(`the seven other views of each test nucleus (flips and rotations): the same four traits as upright for ${pct(viewsSame / viewsN)} of the views · all four right ${pct(viewsRight / viewsN)} · ${DSC.TRAITS.map(k => `${k} ${pct(viewTraits[k] / viewsN)}`).join(', ')}`);
    const samples = test.slice(0, 3).map(d => `${d.name}: ${[0, 1, 2].map(() => `"${DSC.detokenize(write(d, undefined, 1).slice(1).filter(w => w !== DSC.END))}"`).join(' / ')}`);
    console.log(`at temperature 1, three draws each:\n  ${samples.join('\n  ')}`);
    measures = { all: +own.all.toFixed(4), impression: +own.impression.toFixed(4), traits: Object.fromEntries(DSC.TRAITS.map(k => [k, +own.traits[k].toFixed(4)])), said: own.said, base: Object.fromEntries(DSC.TRAITS.map(k => [k, +base.traits[k].toFixed(4)])), noImage: { all: +noImage.all.toFixed(4), traits: Object.fromEntries(DSC.TRAITS.map(k => [k, +noImage.traits[k].toFixed(4)])), text: DSC.detokenize(write(test[0], zeros).slice(1).filter(w => w !== DSC.END)) }, meanCode: { all: +meanImage.all.toFixed(4), traits: Object.fromEntries(DSC.TRAITS.map(k => [k, +meanImage.traits[k].toFixed(4)])), text: DSC.detokenize(write(test[0], mean).slice(1).filter(w => w !== DSC.END)) }, otherLab: { all: +other.all.toFixed(4), traits: Object.fromEntries(DSC.TRAITS.map(k => [k, +other.traits[k].toFixed(4)])), same: +(agree / test.length).toFixed(4), darkDropped: darkFlip, darkN: test.filter(d => d.traits.dark).length }, views: { same: +(viewsSame / viewsN).toFixed(4), all: +(viewsRight / viewsN).toFixed(4), traits: Object.fromEntries(DSC.TRAITS.map(k => [k, +(viewTraits[k] / viewsN).toFixed(4)])) } };
  } else measures = { all: +own.all.toFixed(4), impression: +own.impression.toFixed(4), traits: Object.fromEntries(DSC.TRAITS.map(k => [k, +own.traits[k].toFixed(4)])), said: own.said, base: Object.fromEntries(DSC.TRAITS.map(k => [k, +base.traits[k].toFixed(4)])) };
  console.log(`(the probes took ${((Date.now() - t1) / 1000).toFixed(0)} s)`);
}
if (save) { const ev = lm.evaluate(test), out = lm.toJSON({ name: 'the describe model', trained: `${cfg.epochs} epochs on ${train.length} nuclei`, words: vocab.words, backbone: backbone.id, testLoss: +ev.loss.toFixed(4), testAccuracy: +ev.accuracy.toFixed(4), groundN: cfg.ground || null, hist, measures }); out.trails = wordTrails(snaps, out.E, out.Wout, lm.V, lm.D); fs.writeFileSync(save, `window.DESCRIBE_LM = ${JSON.stringify(out)};\n`); console.log(`saved ${save} (${(fs.statSync(save).size / 1024).toFixed(0)} KB)`); }
