#!/usr/bin/env node
/*
 * check_dna_annotate.js — the DNA model put to work (the Annotate question, recipe 26): a kind for every base of the
 * held-out quarter of the chromosome (exon, intron, CpG island, repeat, everything else), read from the vectors the
 * frozen next-base model holds, by a single layer trained on the labels of a few genes. What the real DNA language
 * models are used for: a small head on frozen embeddings (Evo 2's exon classifier), or a segmentation head fine-tuned
 * with the model (SegmentNT on the Nucleotide Transformer). The supervised family (Enformer, Borzoi, AlphaGenome) skips
 * the label-free stage and trains on labels directly, which is the --e2e scratch contrast here.
 *   (default)   the table by labelled genes (1, 2, 3, 5, 10, 16): the same single layer on the shipped model's vectors,
 *               on an untrained model's, and on the bases themselves (one-hot, the base and 8 either side): the bases
 *               right, the intersection over union per kind and their mean, the exons found and the false ones
 *   --e2e scratch|finetune --genes K [--updates 60000] [--seed 2] [--append FILE]
 *               the same network trained end to end on the labels alone, from fresh weights (scratch; its rate 0.003,
 *               the better of the two tried, 0.01 the other) or from the pretrained ones at a tenth of the head's rate
 *               (finetune): every position predicts the kind of the base it read
 *               8 bases before, so that it has read past it, and the probabilities are averaged over 8 bases either
 *               side; the held-out quarter scored a dozen times on the way. --updates is the number of windows trained
 *               on, the same for every count of genes (60,000 is the pretraining's 60 epochs)
 *   --by-epoch N   the recipe's pretraining run again (seed 2, as tools/check_dna.js and the page train it), and the
 *               single layer with one labelled gene after every epoch: what the vectors are worth as the model learns
 *   --table FILE [--epochs-file FILE] --save data/dna/annotate_table.js
 *               the table the page ships: the single-layer rows measured now, the end-to-end rows from the runs
 *               appended to FILE (the mean of each run's last four scores, then of the seeds), and the rows a
 *               --by-epoch run saved
 */
'use strict';
const fs = require('fs');
const path = require('path');
const NN = require('../js/nn.js');
const DNA = require('../js/dna.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d), has = k => process.argv.includes(k);
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const ROOT = path.join(__dirname, '..'), G = loadJs(path.join(ROOT, 'data', 'dna', 'genome_data.js')).LECTURE_GENOME, W = loadJs(path.join(ROOT, 'data', 'dna', 'dna_weights.js')).DNA_LM;
const N = G.seq.length, held = G.meta.trainEnd, vocab = DNA.vocabOf(1), truth = DNA.tagTruth(G), K = DNA.TAGS.length, COUNTS = [1, 2, 3, 5, 10, 16];
const pc = v => (v == null ? '–' : `${Math.round(100 * v)}%`), names = DNA.TAGS.map(t => DNA.TAG_NAMES[t]);
const fmt = r => `bases right ${pc(r.acc)} · mean IoU ${pc(r.miou)} · ${names.map((c, k) => `${c} ${pc(r.iou[k])}`).join(' · ')} · exons found ${r.found}/${r.total}, false ${r.falseExons}`;
const config = seed => ({ vocabSize: vocab.size, dim: W.dim, layers: W.layers, heads: W.heads, dk: W.dk, ffn: W.ffn, costInit: [0.05, 0.003], positions: W.positions, clip: 1, optimizer: 'adam', seed });
const slim = r => ({ acc: +r.acc.toFixed(4), miou: +r.miou.toFixed(4), iou: r.iou.map(v => (v == null ? null : +v.toFixed(4))), found: r.found, total: r.total, falseExons: r.falseExons });
// the bases themselves as the vector of a base: one-hot, the base and r either side (no model at all)
function rawVectors(r) { const D = 4 * (2 * r + 1), V = DNA.newVectors(N, D), idx = { A: 0, C: 1, G: 2, T: 3 }; for (let p = 0; p < N; p++) { for (let o = -r; o <= r; o++) { const q = p + o; if (q >= 0 && q < N) V.X[p * D + (o + r) * 4 + idx[G.seq[q]]] = 1; } V.done[p] = 1; } return V; }
function rawAnnotate(V, genes) { const end = DNA.labelledEnd(G, genes), Xl = V.X.subarray(0, end * V.D), head = DNA.newHead(V.D, K, 1); DNA.headStandardise(head, Xl, end); for (let ep = 0; ep < 30; ep++) DNA.headEpoch(head, Xl, truth, end); const P = DNA.headProbs(head, V.X.subarray(held * V.D), N - held); return DNA.tagScore(DNA.paintKinds(P, N - held, K), truth, G, held); }
function probeRows() { // the single layer on the shipped model's vectors and on an untrained model's, per count of labelled genes
  const lm = NN.LanguageModel.fromJSON(W), lm0 = new NN.LanguageModel(config(2)), V = DNA.newVectors(N, lm.D), V0 = DNA.newVectors(N, lm0.D), rows = { frozen: [], untrained: [] };
  for (const k of COUNTS) { rows.frozen.push(DNA.annotate(lm, G, vocab, { genes: k, vectors: V })); rows.untrained.push(DNA.annotate(lm0, G, vocab, { genes: k, vectors: V0 })); }
  return rows;
}

if (has('--e2e')) {
  const regime = arg('--e2e', 'scratch'), genes = +arg('--genes', 16), updates = +arg('--updates', 60000), seed = +arg('--seed', 2), delta = 8, rp = 8, win = 64, stride = 32, batch = 4, l2 = 1e-4, lr = 0.01, lrBody = +arg('--lrbody', regime === 'finetune' ? 0.001 : 0.003);
  const lm = regime === 'finetune' ? NN.LanguageModel.fromJSON(W) : new NN.LanguageModel(config(seed)), D = lm.D, end = DNA.labelledEnd(G, genes), starts = []; for (let s = 0; s + win <= end; s += stride) starts.push(s);
  const rnd = NN.mulberry32(seed * 977 + 3), Wh = Float64Array.from({ length: K * D }, () => (rnd() * 2 - 1) * 0.1), bh = new Float64Array(K), ad = { t: 0, mW: new Float64Array(K * D), vW: new Float64Array(K * D), mb: new Float64Array(K), vb: new Float64Array(K) };
  const each = (g, f) => { for (const a of [g.E, g.Pos].filter(Boolean)) f(a); for (const gl of g.layers) for (const key of Object.keys(gl)) f(gl[key]); };
  const softmax = (h, z) => { let mx = -Infinity; for (let k = 0; k < K; k++) { let v = bh[k]; const o = k * D; for (let d = 0; d < D; d++) v += Wh[o + d] * h[d]; z[k] = v; if (v > mx) mx = v; } let Z = 0; for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); Z += z[k]; } for (let k = 0; k < K; k++) z[k] /= Z; return z; };
  function step(ss) { // one step on a batch of windows: the kinds' cross-entropy at every position, back through the head, the stack and the embeddings
    const g = lm.newGradient(), gW = new Float64Array(K * D), gb = new Float64Array(K), z = new Float64Array(K); let loss = 0;
    for (const s of ss) {
      const tokens = DNA.encode(vocab, G.seq.slice(s, s + win)), fw = lm.forward(tokens), n = tokens.length, dT = fw.T.map(() => new Float64Array(D)), m = n - 1 - delta;
      for (let i = 1 + delta; i < n; i++) { const t = truth[s + i - 1 - delta], h = fw.T[i]; softmax(h, z); loss -= Math.log(z[t] + 1e-12) / m; for (let k = 0; k < K; k++) { const dl = (z[k] - (k === t ? 1 : 0)) / m; gb[k] += dl; const o = k * D; for (let d = 0; d < D; d++) { gW[o + d] += dl * h[d]; dT[i][d] += dl * Wh[o + d]; } } }
      let d = dT; for (let l = lm.layers.length - 1; l >= 0; l--) d = lm.layers[l].backward(fw.ctxs[l], d, g.layers[l]);
      for (let i = 0; i < n; i++) { const off = tokens[i] * D, po = lm.Pos && i < lm.P ? i * D : -1; for (let e = 0; e < D; e++) { g.E[off + e] += d[i][e]; if (po >= 0) g.Pos[po + e] += d[i][e]; } }
    }
    const kd = ss.length; let n2 = 0; each(g, a => { for (const x of a) n2 += x * x; }); const gn = Math.sqrt(n2) / kd; if (gn > 1) each(g, a => { for (let i = 0; i < a.length; i++) a[i] /= gn; }); // clipped as the language model clips
    lm.adamStep(g, kd, lrBody, l2);
    ad.t++; const c1 = 1 - Math.pow(0.9, ad.t), c2 = 1 - Math.pow(0.999, ad.t);
    for (let i = 0; i < Wh.length; i++) { const gi = gW[i] / kd + l2 * Wh[i]; ad.mW[i] = 0.9 * ad.mW[i] + 0.1 * gi; ad.vW[i] = 0.999 * ad.vW[i] + 0.001 * gi * gi; Wh[i] -= lr * (ad.mW[i] / c1) / (Math.sqrt(ad.vW[i] / c2) + 1e-8); }
    for (let k = 0; k < K; k++) { const gi = gb[k] / kd; ad.mb[k] = 0.9 * ad.mb[k] + 0.1 * gi; ad.vb[k] = 0.999 * ad.vb[k] + 0.001 * gi * gi; bh[k] -= lr * (ad.mb[k] / c1) / (Math.sqrt(ad.vb[k] / c2) + 1e-8); }
    return loss / kd;
  }
  function evaluate() { // the held-out quarter: every base from the vector `delta` bases on, the probabilities averaged over ±rp bases
    const V = DNA.newVectors(N, D); for (const s of DNA.vectorStarts(V, held, N, win, stride)) DNA.readVectors(lm, G, vocab, s, V, win, stride);
    const n = N - held, P = new Float32Array(n * K), z = new Float64Array(K), h = new Float64Array(D);
    for (let p = held; p < N; p++) { const q = Math.min(N - 1, p + delta); for (let d = 0; d < D; d++) h[d] = V.X[q * D + d]; softmax(h, z); for (let k = 0; k < K; k++) P[(p - held) * K + k] = z[k]; }
    const Ps = new Float32Array(n * K); for (let k = 0; k < K; k++) { let acc = 0, lo = 0, hi = -1; for (let p = 0; p < n; p++) { const a = Math.max(0, p - rp), b = Math.min(n - 1, p + rp); while (hi < b) { hi++; acc += P[hi * K + k]; } while (lo < a) { acc -= P[lo * K + k]; lo++; } Ps[p * K + k] = acc / (b - a + 1); } }
    return DNA.tagScore(DNA.paintKinds(Ps, n, K), truth, G, held);
  }
  const epochs = Math.max(1, Math.ceil(updates / starts.length)), every = Math.max(1, Math.round(epochs / 12)), order = starts.slice(), r2 = NN.mulberry32(seed * 31 + 1), traj = [];
  console.log(`${regime} · ${genes} labelled gene${genes === 1 ? '' : 's'} (${end.toLocaleString()} bases, ${starts.length} windows) · ${epochs} epochs (${(epochs * starts.length).toLocaleString()} windows) · rates: head ${lr}, network ${lrBody} · seed ${seed}`);
  for (let ep = 1; ep <= epochs; ep++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r2() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let loss = 0, nb = 0; for (let b = 0; b < order.length; b += batch) { loss += step(order.slice(b, b + batch)); nb++; }
    if (ep % every === 0 || ep === epochs) { const r = evaluate(); traj.push(Object.assign({ epoch: ep, loss: +(loss / nb).toFixed(4) }, slim(r))); console.log(`epoch ${String(ep).padStart(4)} · training loss ${(loss / nb).toFixed(3)} · ${fmt(r)}`); }
  }
  const tail = traj.slice(-4), mean = f => tail.reduce((a, r) => a + f(r), 0) / tail.length, best = traj.reduce((a, r) => (r.miou > a.miou ? r : a));
  const last = { acc: mean(r => r.acc), miou: mean(r => r.miou), iou: names.map((_, k) => mean(r => r.iou[k] || 0)), found: mean(r => r.found), total: tail[0].total, falseExons: mean(r => r.falseExons) };
  console.log(`\nthe last four scores, averaged: bases right ${pc(last.acc)} · mean IoU ${pc(last.miou)} · ${names.map((c, k) => `${c} ${pc(last.iou[k])}`).join(' · ')} · exons found ${last.found.toFixed(1)}/${last.total}, false ${last.falseExons.toFixed(1)}`);
  console.log(`the best score on the way (chosen with the held-out quarter in hand, which flatters it): epoch ${best.epoch} · mean IoU ${pc(best.miou)} · exon ${pc(best.iou[0])}`);
  if (arg('--append', null)) fs.appendFileSync(arg('--append'), JSON.stringify({ regime, genes, seed, updates: epochs * starts.length, epochs, lrBody, last, best, traj }) + '\n');
} else if (has('--by-epoch')) {
  const epochs = +arg('--by-epoch', 30), seed = +arg('--seed', 2), lm = new NN.LanguageModel(config(seed)), wins = DNA.windows(G, { window: 64, stride: 32, k: 1 }), train = wins.train.map(w => DNA.prep(vocab, w, 'train')), test = wins.test.map(w => DNA.prep(vocab, w, 'test'));
  const rng = NN.mulberry32(seed * 977 + 1), order = train.map((_, i) => i), rows = [];
  console.log(`the pretraining run of the recipe again (seed ${seed}, ${train.length} windows), the single layer with one labelled gene after every epoch`);
  for (let ep = 0; ep <= epochs; ep++) {
    if (ep) { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } for (let b = 0; b < order.length; b += 4) lm.trainBatch(order.slice(b, b + 4).map(i => train[i]), 0.01, 0.0001); }
    const ev = DNA.lossByKind(lm, test, 1), r = DNA.annotate(lm, G, vocab, { genes: 1 }); rows.push({ epoch: ep, exonLoss: ev.byKind.e, r });
    console.log(`epoch ${String(ep).padStart(2)} · held-out ${ev.loss.toFixed(3)} bits per base, exons ${ev.byKind.e.toFixed(2)} · ${fmt(r)}`);
  }
  if (arg('--save', null)) { fs.writeFileSync(arg('--save'), JSON.stringify(rows.map(x => Object.assign({ epoch: x.epoch, exonLoss: +x.exonLoss.toFixed(3) }, slim(x.r))))); console.log(`saved ${arg('--save')}`); }
} else {
  const t0 = Date.now(), rows = probeRows(), raw = rawVectors(8);
  console.log(`the single layer on the frozen model's vectors (${W.trained}), by labelled genes; the held-out quarter, ${(N - held).toLocaleString()} bases, ${rows.frozen[0].total} exons\n`);
  COUNTS.forEach((k, i) => { console.log(`${String(k).padStart(2)} gene${k === 1 ? ' ' : 's'} (${String(DNA.labelledEnd(G, k).toLocaleString()).padStart(6)} bases) · the trained model   · ${fmt(rows.frozen[i])}`); console.log(`${' '.repeat(26)}an untrained model  · ${fmt(rows.untrained[i])}`); console.log(`${' '.repeat(26)}the bases themselves · ${fmt(rawAnnotate(raw, k))}`); });
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
  if (has('--table')) {
    const file = arg('--table', null), runs = file && fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [], avg = xs => xs.reduce((a, x) => a + x, 0) / xs.length;
    const e2e = regime => COUNTS.map(k => { const rs = runs.filter(r => r.regime === regime && r.genes === k); if (!rs.length) return null; return { acc: +avg(rs.map(r => r.last.acc)).toFixed(4), miou: +avg(rs.map(r => r.last.miou)).toFixed(4), iou: names.map((_, j) => +avg(rs.map(r => r.last.iou[j])).toFixed(4)), found: +avg(rs.map(r => r.last.found)).toFixed(1), total: rs[0].last.total, falseExons: +avg(rs.map(r => r.last.falseExons)).toFixed(1), best: +avg(rs.map(r => r.best.miou)).toFixed(4), bestExon: +avg(rs.map(r => r.best.iou[0])).toFixed(4), seeds: rs.length, updates: rs[0].updates }; });
    const out = { counts: COUNTS, bases: COUNTS.map(k => DNA.labelledEnd(G, k)), kinds: DNA.TAGS, model: W.trained, frozen: rows.frozen.map(slim), untrained: rows.untrained.map(slim), scratch: e2e('scratch'), finetune: e2e('finetune'), byEpoch: arg('--epochs-file', null) ? JSON.parse(fs.readFileSync(arg('--epochs-file'), 'utf8')) : null };
    for (const regime of ['scratch', 'finetune']) if (out[regime].some(Boolean)) { console.log(`\n${regime === 'scratch' ? 'the same network from scratch on the labels' : 'the pretrained network fine-tuned on the labels'} (the last four scores of each run, the seeds averaged):`); COUNTS.forEach((k, i) => { const r = out[regime][i]; if (r) console.log(`${String(k).padStart(2)} gene${k === 1 ? ' ' : 's'} · bases right ${pc(r.acc)} · mean IoU ${pc(r.miou)} · ${names.map((c, j) => `${c} ${pc(r.iou[j])}`).join(' · ')} · exons found ${r.found}/${r.total}, false ${r.falseExons} · best on the way: mean IoU ${pc(r.best)}, exon ${pc(r.bestExon)} · ${r.seeds} seed${r.seeds === 1 ? '' : 's'}`); }); }
    if (arg('--save', null)) { fs.writeFileSync(arg('--save'), `window.DNA_ANNOTATE = ${JSON.stringify(out)};\n`); console.log(`\nsaved ${arg('--save')} (${(fs.statSync(arg('--save')).size / 1024).toFixed(1)} kB)`); }
  }
}
