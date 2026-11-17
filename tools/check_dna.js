#!/usr/bin/env node
/*
 * check_dna.js — the page's next-token model on the synthetic genome: single bases as tokens, windows of the chromosome
 * as documents, the first three quarters to train on and the last quarter held out. Per epoch it reports the loss (bits per
 * base) on the held-out windows and by the kind of base predicted (exon, intron, intergenic, CpG island, repeat…).
 * After training it takes three measures of what the model learned without a label:
 *   the codon rhythm    the held-out exons' loss by position in the codon (a model that found the reading frame is
 *                       least surprised by the third base, where the preferred codon decides)
 *   the variant score   for every planted variant, log P(alt window) − log P(ref window) over the variant and the 32
 *                       bases after it: a zero-shot score of harm, as Evo 2 scores a variant; summarised per kind and
 *                       as how often a harmful kind (frameshift, splice, TATA) scores below a neutral one (synonymous,
 *                       intergenic)
 *   the probe           a single layer on the frozen model's mean vector of a 64-base window, trained on a few
 *                       labelled training windows, told exon from not, CpG island from not, repeat from not, scored on
 *                       held-out windows, against the same probe on the untrained model
 *   node tools/check_dna.js [--epochs 20] [--window 128] [--stride 64] [--dim 32] [--layers 2] [--heads 2] [--dk 8] [--ffn 32]
 *                           [--cost 0.05,0.003] [--lr 0.005] [--batch 4] [--decay 0.0001] [--seed 1] [--kmer 1] [--save FILE] [--load FILE] [--curve]
 * --kmer K: the tokens are K-mers read in step (as DNABERT's and the Nucleotide Transformer's), the vocabulary 4^K plus the specials
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { LanguageModel, Net, mulberry32, fitStandardizer } = require('../js/nn.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d), num = (k, d) => +arg(k, d), has = k => process.argv.includes(k);
const cfg = { epochs: num('--epochs', 20), window: num('--window', 128), stride: num('--stride', 64), dim: num('--dim', 32), layers: num('--layers', 2), heads: num('--heads', 2), dk: num('--dk', 8), ffn: num('--ffn', 32), cost: String(arg('--cost', '0.05,0.003')).split(',').map(Number), lr: num('--lr', 0.005), batch: num('--batch', 4), decay: num('--decay', 0.0001), seed: num('--seed', 1), tail: num('--tail', 32), kmer: num('--kmer', 1) };
const save = arg('--save', null), load = arg('--load', null), curve = has('--curve');
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const G = loadJs(process.env.GENOME_FILE || path.join(__dirname, '..', 'data', 'dna', 'genome_data.js')).LECTURE_GENOME, seq = G.seq, lab = G.lab, N = seq.length, trainEnd = G.meta.trainEnd, KINDS = G.meta.kinds;
// ---- the tokens: the three specials the page's models share, then the four bases; a window is the start token and its bases, no end token (the chromosome goes on)
const K = cfg.kmer, kmers = (() => { let out = ['']; for (let i = 0; i < K; i++) out = [].concat(...out.map(p => ['A', 'C', 'G', 'T'].map(b => p + b))); return out; })();
const WORDS = ['⟨s⟩', '⟨/s⟩', '⟨?⟩'].concat(kmers), index = new Map(WORDS.map((w, i) => [w, i]));
const encode = s => { const t = [0]; for (let i = 0; i + K <= s.length; i += K) t.push(index.get(s.slice(i, i + K))); return t; }; // position i of the window predicts token i + 1: the bases [iK, iK + K)
const labOfWindow = l => { const out = []; for (let i = 0; i + K <= l.length; i += K) out.push(l[i]); return out; }; // a token's kind: its first base's
const windows = (from, to, stride) => { const out = []; for (let s = from; s + cfg.window <= to; s += stride) out.push({ start: s, tokens: encode(seq.slice(s, s + cfg.window)), lab: labOfWindow(lab.slice(s, s + cfg.window)) }); return out; };
const train = windows(0, trainEnd, cfg.stride), test = windows(trainEnd, N, cfg.window);
const LN2 = Math.LN2, bits = v => v / LN2 / K, pc = v => `${Math.round(100 * v)}%`; // bits per base: a token's loss is spread over its K bases
// the loss on a set of windows, overall and by the kind of the base predicted (position i of a window predicts base i)
function lossByKind(lm, docs) {
  const ev = lm.evaluate(docs), sum = {}, n = {};
  docs.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const kind = d.lab[i]; sum[kind] = (sum[kind] || 0) + ev.per[k][i]; n[kind] = (n[kind] || 0) + 1; } });
  const byKind = {}; for (const k of Object.keys(sum)) byKind[k] = bits(sum[k] / n[k]);
  return { loss: bits(ev.loss), accuracy: ev.accuracy, byKind };
}
// the codon offset of a base in an exon (0, 1, 2), the frame carried across a gene's exons; null elsewhere
const frameOf = (() => { const off = new Int8Array(N).fill(-1); for (const g of G.genes) { let f = 0; for (const [s, e] of g.exons) { for (let p = s; p < e; p++) off[p] = (p - s + f) % 3; f = (f + (e - s)) % 3; } } return p => (off[p] < 0 ? null : off[p]); })();
function codonRhythm(lm, docs) { if (K !== 1) return [null, null, null]; const ev = lm.evaluate(docs), s = [0, 0, 0], n = [0, 0, 0]; docs.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const f = frameOf(d.start + i); if (f == null) continue; s[f] += ev.per[k][i]; n[f]++; } }); return s.map((v, i) => (n[i] ? bits(v / n[i]) : null)); }
// the variant score: the window of 96 bases before the variant to 32 after, as the reference and with the change; the
// sum of log-probabilities from the variant on, alt minus ref, in bits (negative: the change made the sequence less likely)
function variantScore(lm, v) {
  const a = Math.max(0, v.pos - (cfg.window - K - cfg.tail) - ((v.pos - (cfg.window - K - cfg.tail)) % K + K) % K), b = Math.min(N, v.pos + 1 + cfg.tail), ref = seq.slice(a, b), alt = (ref.slice(0, v.pos - a) + v.alt + ref.slice(v.pos - a + 1) + seq.slice(b, b + 1)).slice(0, ref.length); // a lost base: the window reaches one base further, so both cover the same bases after the variant
  const pr = lm.evaluate([{ tokens: encode(ref) }]).per[0], pa = lm.evaluate([{ tokens: encode(alt) }]).per[0], i0 = Math.floor((v.pos - a) / K); let sr = 0, sa = 0; for (let i = i0; i < pr.length; i++) { sr += pr[i]; sa += pa[i]; }
  return { score: bits(sr - sa) * K, local: bits(pr[i0] - pa[i0]) * K }; // a score is a whole log-probability difference, not per base
}
function variantTable(lm) {
  const rows = G.variants.map(v => Object.assign({}, v, variantScore(lm, v))), by = {};
  for (const r of rows) (by[r.cls] = by[r.cls] || []).push(r);
  const mean = xs => xs.reduce((a, x) => a + x, 0) / xs.length, summary = {};
  for (const [cls, xs] of Object.entries(by)) summary[cls] = { n: xs.length, mean: mean(xs.map(x => x.score)), local: mean(xs.map(x => x.local)), median: xs.map(x => x.score).sort((p, q) => p - q)[Math.floor(xs.length / 2)] };
  const harmful = rows.filter(r => ['frameshift', 'splice', 'tata'].includes(r.cls)), neutral = rows.filter(r => ['synonymous', 'intergenic'].includes(r.cls)); let below = 0; for (const h of harmful) for (const u of neutral) if (h.score < u.score) below++; else if (h.score === u.score) below += 0.5;
  return { rows, summary, auc: below / (harmful.length * neutral.length) };
}
// the probe: the frozen model's mean vector over a 64-base window, a single layer per question, n labelled training windows per class
function features(lm, docs) { return docs.map(d => { const fw = lm.forward(d.tokens), D = lm.D, m = new Float64Array(D); for (const t of fw.T) for (let j = 0; j < D; j++) m[j] += t[j] / fw.T.length; return m; }); }
function probes(lm, nPer) {
  const W = 66, mk = (from, to) => { const out = []; for (let s = from; s + W <= to; s += W) { const l = lab.slice(s, s + W), c = {}; for (const ch of l) c[ch] = (c[ch] || 0) + 1; const top = Object.keys(c).sort((p, q) => c[q] - c[p])[0]; if (c[top] >= W * 0.6) out.push({ tokens: encode(seq.slice(s, s + W)), kind: top }); } return out; };
  const tr = mk(0, trainEnd), te = mk(trainEnd, N), Ftr = features(lm, tr), Fte = features(lm, te), out = {};
  for (const [name, kind] of [['exon', 'e'], ['island', 'c'], ['repeat', 'r'], ['intron', 'n']]) {
    const pos = tr.map((d, i) => i).filter(i => tr[i].kind === kind), neg = tr.map((d, i) => i).filter(i => tr[i].kind !== kind);
    const idx = [...pos.slice(0, nPer), ...neg.filter((_, k) => k % Math.ceil(neg.length / nPer) === 0).slice(0, nPer)], X = idx.map(i => Ftr[i]), y = idx.map(i => (tr[i].kind === kind ? 1 : 0));
    if (!pos.length || !te.some(d => d.kind === kind)) { out[name] = null; continue; }
    const st = fitStandardizer(X, { perDimScale: true }), net = new Net({ inputSize: lm.D, hidden: [], activation: 'relu', seed: 1 }), rnd = mulberry32(3), order = X.map((_, i) => i);
    for (let ep = 0; ep < 200; ep++) { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } for (let b = 0; b < order.length; b += 8) { const bb = order.slice(b, b + 8); net.trainBatch(bb.map(i => st.apply(X[i])), bb.map(i => y[i]), 0.1, 0.001); } }
    // scored on the held-out windows, balanced: every window of the kind against as many of the rest
    const tp = te.map((d, i) => i).filter(i => te[i].kind === kind), tn = te.map((d, i) => i).filter(i => te[i].kind !== kind), take = Math.min(tp.length, tn.length), ids = [...tp.slice(0, take), ...tn.slice(0, take)];
    out[name] = { acc: net.evaluate(ids.map(i => st.apply(Fte[i])), ids.map(i => (te[i].kind === kind ? 1 : 0))).accuracy, nTrain: idx.length, nTest: ids.length };
  }
  return out;
}
const mkModel = seed => new LanguageModel({ vocabSize: WORDS.length, dim: cfg.dim, layers: cfg.layers, heads: cfg.heads, dk: cfg.dk, ffn: cfg.ffn, costInit: cfg.cost.length === 1 ? cfg.cost[0] : cfg.cost, positions: Math.ceil(cfg.window / K) + 1, clip: 1, optimizer: 'adam', seed });
console.log(`genome: ${N} bases, ${train.length} training windows of ${cfg.window} bases (stride ${cfg.stride}), ${test.length} held-out windows · tokens: ${K === 1 ? 'single bases' : `${K}-mers in step, ${kmers.length} of them`} · GC ${G.meta.gc}`);
let lm = load ? LanguageModel.fromJSON(loadJs(load).DNA_LM) : mkModel(cfg.seed);
console.log(`model: ${lm.parameterCount()} parameters · ${lm.describe()} · adam lr ${cfg.lr}, batch ${cfg.batch}, decay ${cfg.decay}`);
const untrained = mkModel(cfg.seed), kindOrder = ['i', 'c', 't', '5', 'e', 'n', '3', 'r', 'm'];
const fmtKinds = bk => kindOrder.filter(k => bk[k] != null).map(k => `${KINDS[k]} ${bk[k].toFixed(2)}`).join(' · ');
const e0 = lossByKind(lm, test); console.log(`epoch   0 · held-out ${e0.loss.toFixed(3)} bits/base (2.000 is a coin toss over four) · ${fmtKinds(e0.byKind)}`);
// the record of an epoch, in the shape the page keeps for its own training: the losses in bits per base, the kinds as keys, the variant measure as `ground`, a sequence written at temperature 0.8 as `sample`
const sampleRng = mulberry32(cfg.seed * 31 + 7), sampleOf = lm => lm.generate([0], { maxTokens: Math.ceil(120 / K), temperature: 0.8, rng: sampleRng }).tokens.map(t => (WORDS[t].length === K && t > 2 ? WORDS[t] : 'N')).join('');
const meansOf = rows => { const by = {}; for (const r of rows) (by[r.cls] = by[r.cls] || []).push(r.score); const out = {}; for (const [k, xs] of Object.entries(by)) out[k] = +(xs.reduce((a, x) => a + x, 0) / xs.length).toFixed(3); return out; };
const record = (epoch, loss, ev, vt) => Object.assign({ epoch, loss, testLoss: ev.loss, acc: ev.accuracy, ground: vt ? vt.auc : null, means: vt ? meansOf(vt.rows) : null, sample: epoch ? sampleOf(lm) : null }, ev.byKind);
const rng = mulberry32(cfg.seed * 977 + 1), order = train.map((_, i) => i), hist = [record(0, null, e0, null)];
for (let epoch = 1; epoch <= (load ? 0 : cfg.epochs); epoch++) {
  const t0 = Date.now(); let loss = 0, nb = 0;
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  for (let b = 0; b < order.length; b += cfg.batch) { loss += lm.trainBatch(order.slice(b, b + cfg.batch).map(i => train[i]), cfg.lr, cfg.decay); nb++; }
  const ev = lossByKind(lm, test), vt = variantTable(lm);
  hist.push(record(epoch, bits(loss / nb), ev, vt));
  if (curve || epoch === 1 || epoch % 5 === 0 || epoch === cfg.epochs) console.log(`epoch ${String(epoch).padStart(3)} · train ${bits(loss / nb).toFixed(3)} · held-out ${ev.loss.toFixed(3)} bits/base · next base right ${pc(ev.accuracy)} · harmful below neutral ${pc(vt.auc)} · ${fmtKinds(ev.byKind)} · ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
const rhythm = codonRhythm(lm, test), vt = variantTable(lm);
if (K === 1) console.log(`\ncodon rhythm, held-out exons: position 1 ${rhythm[0].toFixed(2)} · 2 ${rhythm[1].toFixed(2)} · 3 ${rhythm[2].toFixed(2)} bits (untrained: ${codonRhythm(untrained, test).map(v => v.toFixed(2)).join(' · ')})`);
console.log(`variant score (log2 P(alt) − log2 P(ref) over the variant and the ${cfg.tail} bases after it; more negative = less likely):`);
for (const cls of ['synonymous', 'intergenic', 'missense', 'frameshift', 'splice', 'tata']) { const s = vt.summary[cls]; if (s) console.log(`  ${cls.padEnd(11)} n ${s.n} · mean ${s.mean.toFixed(2)} · median ${s.median.toFixed(2)} · at the base alone ${s.local.toFixed(2)}`); }
console.log(`  a harmful variant (frameshift, splice, TATA) scores below a neutral one (synonymous, intergenic) ${pc(vt.auc)} of the time (untrained model: ${pc(variantTable(untrained).auc)})`);
for (const nPer of [10, 40]) { const p = probes(lm, nPer), p0 = probes(untrained, nPer); console.log(`probe, ${nPer} labelled windows per class: ${Object.keys(p).map(k => `${k} ${p[k] ? pc(p[k].acc) : '–'} (untrained ${p0[k] ? pc(p0[k].acc) : '–'})`).join(' · ')}`); }
if (save) {
  const out = lm.toJSON({ name: 'the DNA model', trained: `${cfg.epochs} epochs on ${train.length} windows of ${cfg.window} bases, seed ${cfg.seed}`, testLoss: `${hist[hist.length - 1].testLoss.toFixed(3)} bits per base`, words: WORDS, window: cfg.window, stride: cfg.stride, tail: cfg.tail, hist, rhythm, variants: vt.rows.map(r => ({ pos: r.pos, cls: r.cls, score: +r.score.toFixed(3), local: +r.local.toFixed(3) })), auc: vt.auc, probe: probes(lm, 10) });
  fs.writeFileSync(save, `window.DNA_LM = ${JSON.stringify(out)};\n`); console.log(`saved ${save} (${(fs.statSync(save).size / 1024).toFixed(0)} kB)`);
}
