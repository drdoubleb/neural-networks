/*
 * dna.js — the DNA question: the page's next-token model on a synthetic genome. The tokens are single bases, or k-mers
 * read in step (as DNABERT's and the Nucleotide Transformer's were); the documents are windows of the chromosome, the
 * first three quarters to train on and the last quarter held out; every base carries the truth of what it is (exon,
 * intron, CpG island, repeat…), which the model never sees and the page shows beside what the model learned. Also the
 * measures of the Test step: the loss base by base along the held-out stretch, the zero-shot variant score (the
 * sequence with the change against the sequence without), and a probe on the frozen model's vectors. And the model
 * put to work (the Annotate question): the vector the frozen model holds at every base, a single layer on it trained on
 * the labels of a few genes, the held-out stretch painted a kind per base and scored. Works in the browser
 * (window.NucleusDNA) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./nn.js'));
  else root.NucleusDNA = factory(root.TinyNet);
})(typeof self !== 'undefined' ? self : this, function (NN) {
  'use strict';
  const BASES = ['A', 'C', 'G', 'T'], SPECIALS = ['⟨s⟩', '⟨/s⟩', '⟨?⟩'];
  const KINDS = ['i', 'c', 't', '5', 'e', 'n', '3', 'r', 'm']; // intergenic, CpG island, TATA box, 5' UTR, exon, intron, 3' UTR, repeat, microsatellite
  const KIND_NAMES = { i: 'intergenic', c: 'CpG island', t: 'TATA box', 5: '5′ UTR', e: 'exon', n: 'intron', 3: '3′ UTR', r: 'repeat', m: 'microsatellite' };
  const KIND_COLORS = { i: '#9a9aa6', c: '#0e9f6e', t: '#d4a017', 5: '#7aa6d6', e: '#4a3aa7', n: '#b7b0e6', 3: '#7aa6d6', r: '#eb6834', m: '#c2185b' };
  const LN2 = Math.LN2;
  // the vocabulary for k-mers in step: the three specials the page's models share, then the 4^k k-mers
  function vocabOf(k) {
    let kmers = ['']; for (let i = 0; i < k; i++) kmers = [].concat(...kmers.map(p => BASES.map(b => p + b)));
    const words = SPECIALS.concat(kmers), index = new Map(words.map((w, i) => [w, i]));
    return { words, index, start: 0, end: 1, unk: 2, size: words.length, k };
  }
  // a stretch of sequence as tokens: the start token, then its k-mers in step (a remainder shorter than k is dropped); no end token, the chromosome goes on
  function encode(vocab, s) { const k = vocab.k, t = [vocab.start]; for (let i = 0; i + k <= s.length; i += k) t.push(vocab.index.has(s.slice(i, i + k)) ? vocab.index.get(s.slice(i, i + k)) : vocab.unk); return t; }
  const decode = (vocab, ids) => ids.map(i => vocab.words[i]);
  // the kind of every token: the start token takes the first base's, a k-mer its first base's
  function kindsOf(lab, k) { const out = [lab[0] || 'i']; for (let i = 0; i + k <= lab.length; i += k) out.push(lab[i]); return out; }
  // the windows of the chromosome as documents: training windows every `stride` bases over the first three quarters,
  // held-out windows in step over the last quarter; a window's kind is the kind of most of its bases
  function windows(G, { window = 128, stride = 64, k = 1 } = {}) {
    const N = G.seq.length, trainEnd = G.meta.trainEnd, mk = (from, to, st, split) => { const out = []; for (let s = from; s + window <= to; s += st) { const l = G.lab.slice(s, s + window), c = {}; for (const ch of l) c[ch] = (c[ch] || 0) + 1; const kind = Object.keys(c).sort((p, q) => c[q] - c[p])[0]; out.push({ id: `${split === 'train' ? 'W' : 'H'}${String(out.length + 1).padStart(3, '0')}`, name: `${s.toLocaleString()}–${(s + window).toLocaleString()}`, start: s, end: s + window, text: G.seq.slice(s, s + window), lab: l, kind, split, share: c[kind] / window }); } return out; };
    return { train: mk(0, trainEnd, stride, 'train'), test: mk(trainEnd, N, window, 'test') };
  }
  // a prepared document: tokens, the words they stand for, and a kind per token (the sections the page colours by)
  function prep(vocab, w, split) { const tokens = encode(vocab, w.text), words = decode(vocab, tokens); return Object.assign({}, w, { split, tokens, words, sections: kindsOf(w.lab, vocab.k), cls: w.kind }); }
  // the loss of a set of windows, overall and by the kind of the token predicted, in bits per base
  function lossByKind(lm, docs, k) {
    const ev = lm.evaluate(docs), sum = {}, n = {};
    docs.forEach((d, j) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const kind = d.sections[i + 1]; sum[kind] = (sum[kind] || 0) + ev.per[j][i]; n[kind] = (n[kind] || 0) + 1; } });
    const byKind = {}; for (const key of Object.keys(sum)) byKind[key] = sum[key] / n[key] / LN2 / k;
    return { loss: ev.loss / LN2 / k, accuracy: ev.accuracy, byKind, per: ev.per };
  }
  // the loss base by base along a stretch, in bits: the windows in step, every token's loss spread over its k bases;
  // the first base of a window is predicted from the start token alone
  function surprise(lm, G, vocab, from, to, window) {
    const k = vocab.k, out = new Float32Array(to - from).fill(NaN);
    for (let s = from; s + k <= to; s += window) { const text = G.seq.slice(s, Math.min(to, s + window)), tokens = encode(vocab, text), fw = lm.forward(tokens), l = lm.loss(fw); for (let i = 0; i + 1 < tokens.length; i++) { const v = l.per[i] / LN2 / k; for (let q = 0; q < k; q++) out[s - from + i * k + q] = v; } }
    return out;
  }
  // the variant score: the window of bases before the variant and `tail` after it, as the reference and with the
  // change; log2 P(alt window) − log2 P(ref window) summed from the variant's token on (negative: the change made the
  // sequence less likely), and the same at the variant's token alone
  function variantScore(lm, G, vocab, v, { window = 128, tail = 32 } = {}) {
    const k = vocab.k, N = G.seq.length, left = window - k - tail; let a = Math.max(0, v.pos - left); a -= ((v.pos - a) % k + k) % k; const b = Math.min(N, v.pos + 1 + tail);
    const ref = G.seq.slice(a, b), alt = (ref.slice(0, v.pos - a) + v.alt + ref.slice(v.pos - a + 1) + G.seq.slice(b, b + 1)).slice(0, ref.length); // a lost base: the window reaches one base further, so both cover the same bases after the variant
    const pr = lm.loss(lm.forward(encode(vocab, ref))).per, pa = lm.loss(lm.forward(encode(vocab, alt))).per, i0 = Math.floor((v.pos - a) / k); let sr = 0, sa = 0; for (let i = i0; i < pr.length; i++) { sr += pr[i]; sa += pa[i]; }
    return { score: (sr - sa) / LN2, local: (pr[i0] - pa[i0]) / LN2, from: a, to: b, i0, pr, pa };
  }
  const HARMFUL = ['frameshift', 'splice', 'tata'], NEUTRAL = ['synonymous', 'intergenic'], CLASSES = ['synonymous', 'intergenic', 'missense', 'frameshift', 'splice', 'tata'];
  const CLASS_NAMES = { synonymous: 'synonymous', intergenic: 'intergenic', missense: 'missense', frameshift: 'frameshift', splice: 'splice site', tata: 'TATA box' };
  // every planted variant scored, the summary per kind, and how often a harmful kind scores below a neutral one
  function variantTable(lm, G, vocab, opts) {
    const rows = G.variants.map(v => { const r = variantScore(lm, G, vocab, v, opts); return Object.assign({}, v, { score: r.score, local: r.local }); }), by = {};
    for (const r of rows) (by[r.cls] = by[r.cls] || []).push(r);
    const summary = {}; for (const cls of CLASSES) { const xs = by[cls] || []; if (!xs.length) continue; const sc = xs.map(x => x.score).sort((p, q) => p - q); summary[cls] = { n: xs.length, mean: sc.reduce((a, x) => a + x, 0) / sc.length, median: sc[Math.floor(sc.length / 2)], local: xs.reduce((a, x) => a + x.local, 0) / xs.length }; }
    const harmful = rows.filter(r => HARMFUL.includes(r.cls)), neutral = rows.filter(r => NEUTRAL.includes(r.cls)); let below = 0; for (const h of harmful) for (const u of neutral) below += h.score < u.score ? 1 : h.score === u.score ? 0.5 : 0;
    return { rows, summary, auc: harmful.length && neutral.length ? below / (harmful.length * neutral.length) : null };
  }
  // the probe: the frozen model's mean final vector over a short window, a single layer per question (exon or not, CpG
  // island or not, repeat or not, intron or not) trained on nPer labelled training windows per side and scored on the
  // held-out windows, balanced
  function probeWindows(G, vocab, W) { const mk = (from, to) => { const out = []; for (let s = from; s + W <= to; s += W) { const l = G.lab.slice(s, s + W), c = {}; for (const ch of l) c[ch] = (c[ch] || 0) + 1; const top = Object.keys(c).sort((p, q) => c[q] - c[p])[0]; if (c[top] >= W * 0.6) out.push({ tokens: encode(vocab, G.seq.slice(s, s + W)), kind: top, start: s }); } return out; }; return { train: mk(0, G.meta.trainEnd), test: mk(G.meta.trainEnd, G.seq.length) }; }
  function features(lm, docs) { return docs.map(d => { const fw = lm.forward(d.tokens), D = lm.D, m = new Float64Array(D); for (const t of fw.T) for (let j = 0; j < D; j++) m[j] += t[j] / fw.T.length; return m; }); }
  function probes(lm, G, vocab, { nPer = 10, W = 66, questions = [['exon', 'e'], ['island', 'c'], ['repeat', 'r'], ['intron', 'n']] } = {}) {
    const { train: tr, test: te } = probeWindows(G, vocab, W), Ftr = features(lm, tr), Fte = features(lm, te), out = {};
    for (const [name, kind] of questions) {
      const pos = tr.map((d, i) => i).filter(i => tr[i].kind === kind), neg = tr.map((d, i) => i).filter(i => tr[i].kind !== kind), tp = te.map((d, i) => i).filter(i => te[i].kind === kind), tn = te.map((d, i) => i).filter(i => te[i].kind !== kind);
      if (pos.length < 2 || !tp.length) { out[name] = null; continue; }
      const step = Math.max(1, Math.ceil(neg.length / nPer)), idx = [...pos.slice(0, nPer), ...neg.filter((_, j) => j % step === 0).slice(0, nPer)], X = idx.map(i => Ftr[i]), y = idx.map(i => (tr[i].kind === kind ? 1 : 0));
      const st = NN.fitStandardizer(X, { perDimScale: true }), net = new NN.Net({ inputSize: lm.D, hidden: [], activation: 'relu', seed: 1 }), rnd = NN.mulberry32(3), order = X.map((_, i) => i);
      for (let ep = 0; ep < 200; ep++) { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } for (let b = 0; b < order.length; b += 8) { const bb = order.slice(b, b + 8); net.trainBatch(bb.map(i => st.apply(X[i])), bb.map(i => y[i]), 0.1, 0.001); } }
      const take = Math.min(tp.length, tn.length), ids = [...tp.slice(0, take), ...tn.slice(0, take)];
      out[name] = { acc: net.evaluate(ids.map(i => st.apply(Fte[i])), ids.map(i => (te[i].kind === kind ? 1 : 0))).accuracy, nTrain: idx.length, nTest: ids.length, nPos: pos.length };
    }
    return out;
  }
  // ---- the model put to work: a kind for every base, read from the vectors the frozen model holds (the gene finder of the Annotate question)
  // the kinds it paints: exon, intron, CpG island, repeat, and everything else (the open sequence, the UTRs, the TATA box, the microsatellites)
  const TAGS = ['e', 'n', 'c', 'r', 'o'], TAG_NAMES = { e: 'exon', n: 'intron', c: 'CpG island', r: 'repeat', o: 'everything else' }, TAG_COLORS = { e: KIND_COLORS.e, n: KIND_COLORS.n, c: KIND_COLORS.c, r: KIND_COLORS.r, o: KIND_COLORS.i };
  const tagTruth = G => Int8Array.from(G.lab, ch => { const k = TAGS.indexOf(ch); return k < 0 ? TAGS.length - 1 : k; });
  // the labels come as an annotated stretch: from the first base of the chromosome to the start of the gene after the
  // k-th, so k genes with the open sequence and the repeats that lie on the way; only genes that lie whole in the training stretch count
  const labelGenes = G => G.genes.filter(g => g.end <= G.meta.trainEnd).length;
  function labelledEnd(G, k) { const g = G.genes[Math.max(1, Math.min(k, labelGenes(G)))]; return g ? Math.min(g.start, G.meta.trainEnd) : G.meta.trainEnd; }
  // the vector the model holds after reading a base: the chromosome is read in windows every `stride` bases and every
  // base is taken from the one window in which it has at least window − stride bases before it (the first window gives
  // all of its bases, the last of the chromosome whatever is left), so that a base's vector does not depend on what else is read
  function newVectors(N, D) { return { X: new Float32Array(N * D), done: new Uint8Array(N), N, D }; }
  function vectorSource(N, p, win, stride) { const tail = Math.floor((N - win) / stride) * stride + win; return p < win ? 0 : p < tail ? stride * Math.floor((p - win + stride) / stride) : N - win; }
  function vectorStarts(V, from, to, win = 64, stride = 32) { const out = []; let last = -1; for (let p = Math.max(0, from); p < Math.min(V.N, to); p++) { if (V.done[p]) continue; const s = vectorSource(V.N, p, win, stride); if (s !== last) { out.push(s); last = s; } } return out; }
  function readVectors(lm, G, vocab, s, V, win = 64, stride = 32) { const fw = lm.forward(encode(vocab, G.seq.slice(s, s + win))), D = V.D; for (let j = 0; j < win; j++) { const p = s + j; if (p >= V.N || V.done[p] || vectorSource(V.N, p, win, stride) !== s) continue; const t = fw.T[j + 1]; for (let d = 0; d < D; d++) V.X[p * D + d] = t[d]; V.done[p] = 1; } }
  // what the head is given for a base: the mean of the vectors over the r bases either side of it (clipped at the ends of the chromosome); [from, to) of them
  function meanVectors(V, from, to, r = 16) { const { X, D, N } = V, out = new Float32Array((to - from) * D), acc = new Float64Array(D); let lo = Math.max(0, from - r), hi = lo - 1; for (let p = from; p < to; p++) { const a = Math.max(0, p - r), b = Math.min(N - 1, p + r); while (hi < b) { hi++; for (let d = 0; d < D; d++) acc[d] += X[hi * D + d]; } while (lo < a) { for (let d = 0; d < D; d++) acc[d] -= X[lo * D + d]; lo++; } const n = b - a + 1, o = (p - from) * D; for (let d = 0; d < D; d++) out[o + d] = acc[d] / n; } return out; }
  // the head: a single layer with a softmax over the kinds, on the vectors standardised by the labelled bases' mean and spread; Adam on minibatches
  function newHead(D, K, seed = 1) { const z = n => new Float64Array(n); return { D, K, W: z(K * D), b: z(K), mu: z(D), sd: z(D).fill(1), t: 0, mW: z(K * D), vW: z(K * D), mb: z(K), vb: z(K), rnd: NN.mulberry32(seed * 131 + 7), order: null, epoch: 0 }; }
  function headStandardise(head, X, n) { const D = head.D, mu = head.mu.fill(0), sd = head.sd.fill(0); for (let p = 0; p < n; p++) for (let d = 0; d < D; d++) mu[d] += X[p * D + d]; for (let d = 0; d < D; d++) mu[d] /= n; for (let p = 0; p < n; p++) for (let d = 0; d < D; d++) { const v = X[p * D + d] - mu[d]; sd[d] += v * v; } for (let d = 0; d < D; d++) sd[d] = Math.sqrt(sd[d] / n) || 1; }
  function headEpoch(head, X, y, n, { lr = 0.01, l2 = 1e-4, batch = 64 } = {}) { // one pass over the n labelled bases in a fresh order; returns the mean loss on the way
    const { D, K, W, b, mu, sd } = head, x = new Float64Array(D), pr = new Float64Array(K), gW = new Float64Array(K * D), gb = new Float64Array(K);
    if (!head.order || head.order.length !== n) head.order = Array.from({ length: n }, (_, i) => i);
    const order = head.order; for (let i = n - 1; i > 0; i--) { const j = Math.floor(head.rnd() * (i + 1)), q = order[i]; order[i] = order[j]; order[j] = q; }
    let loss = 0;
    for (let s = 0; s < n; s += batch) {
      gW.fill(0); gb.fill(0); const e = Math.min(n, s + batch), nb = e - s;
      for (let q = s; q < e; q++) { const p = order[q]; for (let d = 0; d < D; d++) x[d] = (X[p * D + d] - mu[d]) / sd[d]; let mx = -Infinity; for (let k = 0; k < K; k++) { let z = b[k]; const o = k * D; for (let d = 0; d < D; d++) z += W[o + d] * x[d]; pr[k] = z; if (z > mx) mx = z; } let Z = 0; for (let k = 0; k < K; k++) { pr[k] = Math.exp(pr[k] - mx); Z += pr[k]; } loss -= Math.log(pr[y[p]] / Z + 1e-12); for (let k = 0; k < K; k++) { const dl = pr[k] / Z - (k === y[p] ? 1 : 0); gb[k] += dl; const o = k * D; for (let d = 0; d < D; d++) gW[o + d] += dl * x[d]; } }
      head.t++; const c1 = 1 - Math.pow(0.9, head.t), c2 = 1 - Math.pow(0.999, head.t);
      for (let i = 0; i < W.length; i++) { const g = gW[i] / nb + l2 * W[i]; head.mW[i] = 0.9 * head.mW[i] + 0.1 * g; head.vW[i] = 0.999 * head.vW[i] + 0.001 * g * g; W[i] -= lr * (head.mW[i] / c1) / (Math.sqrt(head.vW[i] / c2) + 1e-8); }
      for (let k = 0; k < K; k++) { const g = gb[k] / nb; head.mb[k] = 0.9 * head.mb[k] + 0.1 * g; head.vb[k] = 0.999 * head.vb[k] + 0.001 * g * g; b[k] -= lr * (head.mb[k] / c1) / (Math.sqrt(head.vb[k] / c2) + 1e-8); }
    }
    head.epoch++; return loss / n;
  }
  function headProbs(head, X, n) { const { D, K, W, b, mu, sd } = head, P = new Float32Array(n * K), x = new Float64Array(D), z = new Float64Array(K); for (let p = 0; p < n; p++) { for (let d = 0; d < D; d++) x[d] = (X[p * D + d] - mu[d]) / sd[d]; let mx = -Infinity; for (let k = 0; k < K; k++) { let v = b[k]; const o = k * D; for (let d = 0; d < D; d++) v += W[o + d] * x[d]; z[k] = v; if (v > mx) mx = v; } let Z = 0; for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); Z += z[k]; } for (let k = 0; k < K; k++) P[p * K + k] = z[k] / Z; } return P; }
  function paintKinds(P, n, K) { const out = new Int8Array(n); for (let p = 0; p < n; p++) { let best = 0; for (let k = 1; k < K; k++) if (P[p * K + k] > P[p * K + best]) best = k; out[p] = best; } return out; }
  // a painting of [from, from + n) against the truth: the bases right, the intersection over union of every kind and
  // their mean, and the exons: found when more than half of an exon's bases are painted exon, false when a painted
  // run of 20 bases or more lies less than half inside a true exon
  function tagScore(pred, truth, G, from) {
    const n = pred.length, K = TAGS.length, inter = new Float64Array(K), uni = new Float64Array(K), count = new Float64Array(K); let right = 0;
    for (let p = 0; p < n; p++) { const t = truth[from + p], q = pred[p]; count[t]++; if (t === q) { right++; inter[t]++; uni[t]++; } else { uni[t]++; uni[q]++; } }
    const iou = Array.from(inter, (v, k) => (uni[k] ? v / uni[k] : null)), recall = Array.from(inter, (v, k) => (count[k] ? v / count[k] : null)), have = iou.filter(v => v != null), exons = [];
    for (const g of G.genes) g.exons.forEach(([s, e], i) => { if (s < from || e > from + n) return; let c = 0; for (let p = s; p < e; p++) if (pred[p - from] === 0) c++; exons.push({ gene: g.id, n: i + 1, start: s, end: e, covered: c / (e - s) }); });
    const falseRuns = []; for (let p = 0; p < n;) { if (pred[p] !== 0) { p++; continue; } let q = p, c = 0; while (q < n && pred[q] === 0) { if (truth[from + q] === 0) c++; q++; } if (q - p >= 20 && c < (q - p) / 2) falseRuns.push([from + p, from + q]); p = q; }
    return { acc: right / n, iou, recall, miou: have.length ? have.reduce((a, v) => a + v, 0) / have.length : null, exons, found: exons.filter(x => x.covered > 0.5).length, total: exons.length, falseExons: falseRuns.length, falseRuns };
  }
  // the whole measure for a model as it stands, in one call (Node; the page does the same in steps): the vectors of
  // the labelled stretch and of the held-out quarter, the head trained on the labelled bases, the held-out quarter painted and scored
  function annotate(lm, G, vocab, { genes = 1, epochs = 30, radius = 16, window = 64, stride = 32, seed = 1, vectors = null } = {}) {
    const N = G.seq.length, held = G.meta.trainEnd, end = labelledEnd(G, genes), truth = tagTruth(G), V = vectors || newVectors(N, lm.D);
    for (const [a, b] of [[0, end + radius], [held - radius, N]]) for (const s of vectorStarts(V, a, b, window, stride)) readVectors(lm, G, vocab, s, V, window, stride);
    const Xl = meanVectors(V, 0, end, radius), head = newHead(lm.D, TAGS.length, seed); headStandardise(head, Xl, end); let loss = null; for (let ep = 0; ep < epochs; ep++) loss = headEpoch(head, Xl, truth, end);
    const Xh = meanVectors(V, held, N, radius), P = headProbs(head, Xh, N - held), pred = paintKinds(P, N - held, TAGS.length);
    return Object.assign(tagScore(pred, truth, G, held), { head, pred, probs: P, loss, labelled: end, vectors: V });
  }
  // the motifs the generator planted, for marking what the model writes
  const MOTIFS = [{ re: /TATAAA/g, name: 'TATA box', kind: 't' }, { re: /GTAAGT/g, name: 'splice donor', kind: 'n' }, { re: /AATAAA/g, name: 'polyadenylation signal', kind: '3' }, { re: /(?:CA){6,}|(?:TG){6,}|A{10,}|(?:GATA){4,}|(?:AAT){5,}/g, name: 'microsatellite', kind: 'm' }, { re: /ATG/g, name: 'start codon', kind: 'e' }];
  return { BASES, SPECIALS, KINDS, KIND_NAMES, KIND_COLORS, CLASSES, CLASS_NAMES, HARMFUL, NEUTRAL, MOTIFS, vocabOf, encode, decode, kindsOf, windows, prep, lossByKind, surprise, variantScore, variantTable, probeWindows, features, probes, TAGS, TAG_NAMES, TAG_COLORS, tagTruth, labelGenes, labelledEnd, newVectors, vectorSource, vectorStarts, readVectors, meanVectors, newHead, headStandardise, headEpoch, headProbs, paintKinds, tagScore, annotate };
});
