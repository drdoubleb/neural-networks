#!/usr/bin/env node
/*
 * check_distill.js — can the miniature foundation model be pretrained as a teacher and a student instead of by instance
 * discrimination? The experiment behind the question. Same 100 unlabelled nuclei, same encoder (4 filters 5×5, pool
 * 4×4, a linear layer to a code of 8), same views (flips, rotations, the other lab's scan), same probes as
 * tools/check_foundation.js; but the rule is self-distillation in the manner of DINO: a student network and a teacher
 * that is a slow moving average of the student, a small head from the code to K prototypes, and the student's softmax
 * over the prototypes for one view trained to match the teacher's for the other view, no negatives anywhere; the
 * teacher's output is centred (a running mean subtracted) and sharpened (a lower temperature) so that the two cannot
 * agree on one prototype for everything. The contrastive rule runs beside it on the same seeds and views.
 *   node tools/check_distill.js [--epochs 100] [--seeds 2] [--grid small|full|round2|long|byol|knn] [--probe teacher|student] [--knn]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { Net, Contrastive, mulberry32, fitStandardizer } = require('../js/nn.js');
const DS = require('../js/dataset.js');

const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const epochs = +arg('--epochs', 100), nSeeds = +arg('--seeds', 2), grid = arg('--grid', 'small'), probeOf = arg('--probe', 'teacher'), knn = process.argv.includes('--knn');
const window = {};
for (const f of ['atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const tasks = {}; for (const [id, raw] of Object.entries(window.LECTURE_TASKS)) tasks[id] = DS.prepare(raw);
const FM = { tasks: ['atypia', 'enlargement', 'irregularity'], counts: [34, 33, 33], heldPer: 20, heldBatch: 20, batch: 20, lr: 0.05, K: 4, code: 8, tau: 0.2, probePerClass: 10, probeEpochs: 150, probeLr: 0.1 };
const set = [], held = [];
FM.tasks.forEach((id, k) => { const tr = tasks[id].train; tr.slice(0, FM.counts[k]).forEach(s => set.push({ s, task: id })); tr.slice(FM.counts[k], FM.counts[k] + FM.heldPer).forEach(s => held.push({ s, task: id })); });
const size = tasks.atypia.size, D = size * size;
const std = fitStandardizer(set.map(e => e.s.variants.A.ink), { perDimScale: false });
const view = (e, lab, t) => std.apply(t ? DS.dihedral(e.s.variants[lab].ink, size, t) : e.s.variants[lab].ink);
const randomView = (e, rng) => { const lab = ['A', 'B'][Math.floor(rng() * 2)], t = Math.floor(rng() * 8); return view(e, lab, t); };
const heldRng = mulberry32(99), heldPairs = held.map(e => [randomView(e, heldRng), randomView(e, heldRng)]);

// ---------------------------------------------------------------- the teacher and the student
class Distill {
  constructor({ imageSize, conv = { K: 4, f: 5, pool: 4 }, code = 8, K = 16, tauS = 0.1, tauT = 0.04, momentum = 0.99, centerMomentum = 0.9, hidden = 0, norm = true, seed = 1 }) {
    this.norm = norm; this.size = imageSize; this.code = code; this.K = K; this.tauS = tauS; this.tauT = tauT; this.m = momentum; this.cm = centerMomentum; this.hidden = hidden; this.seed = seed; this.steps = 0;
    this.student = new Net({ inputSize: imageSize * imageSize, imageSize, conv, hidden: [code], activation: 'linear', seed });
    this.teacher = new Net({ inputSize: imageSize * imageSize, imageSize, conv, hidden: [code], activation: 'linear', seed });
    const rnd = mulberry32(seed * 7919 + 11), u = s => (rnd() * 2 - 1) * s;
    // the head: code → (hidden, relu →) K prototype logits; the teacher's head is its moving average too
    this.H = hidden; const inH = hidden || code;
    this.W1 = hidden ? Float64Array.from({ length: hidden * code }, () => u(Math.sqrt(6 / (code + hidden)))) : null; this.b1 = hidden ? new Float64Array(hidden) : null;
    this.Wh = Float64Array.from({ length: K * inH }, () => u(Math.sqrt(6 / (inH + K)))); this.bh = new Float64Array(K);
    this.center = new Float64Array(K);
    this.syncTeacher(1); // the teacher starts as the student
  }
  get enc() { return probeOf === 'student' ? this.student : this.teacher; }
  encode(x) { return this.enc.forward(x).a[1]; }
  describe() { const c = this.student.conv; return `${c.K} filters ${c.f}×${c.f} + pool ${c.pool}×${c.pool} → code of ${this.code} → ${this.H ? this.H + ' → ' : ''}${this.K} prototypes${this.norm ? ' (unit code · unit prototypes)' : ''}`; }
  params() { return [['Wc', this.student.Wc, this.teacher.Wc], ['bc', this.student.bc, this.teacher.bc], ['W', this.student.W[0], this.teacher.W[0]], ['b', this.student.b[0], this.teacher.b[0]]]; }
  syncTeacher(rate) { // teacher ← (1 − rate) teacher + rate student, the head included
    for (const [, s, t] of this.params()) for (let i = 0; i < s.length; i++) t[i] += rate * (s[i] - t[i]);
    if (!this.WhT) { this.WhT = this.Wh.slice(); this.bhT = this.bh.slice(); this.W1T = this.W1 ? this.W1.slice() : null; this.b1T = this.b1 ? this.b1.slice() : null; }
    else { for (let i = 0; i < this.Wh.length; i++) this.WhT[i] += rate * (this.Wh[i] - this.WhT[i]); for (let i = 0; i < this.bh.length; i++) this.bhT[i] += rate * (this.bh[i] - this.bhT[i]); if (this.W1) { for (let i = 0; i < this.W1.length; i++) this.W1T[i] += rate * (this.W1[i] - this.W1T[i]); for (let i = 0; i < this.b1.length; i++) this.b1T[i] += rate * (this.b1[i] - this.b1T[i]); } }
  }
  head(code, W1, b1, Wh, bh) { // returns the logits and what the gradient needs: the hidden activation, its unit vector and the norms
    let h = code; if (this.H) { h = new Float64Array(this.H); for (let j = 0; j < this.H; j++) { let s = b1[j]; for (let d = 0; d < this.code; d++) s += W1[j * this.code + d] * code[d]; h[j] = s > 0 ? s : 0; } }
    const inH = h.length; let u = h, nh = 1;
    if (this.norm) { nh = 0; for (let d = 0; d < inH; d++) nh += h[d] * h[d]; nh = Math.sqrt(nh) + 1e-8; u = new Float64Array(inH); for (let d = 0; d < inH; d++) u[d] = h[d] / nh; }
    const l = new Float64Array(this.K), nw = new Float64Array(this.K);
    for (let k = 0; k < this.K; k++) { // normalised: the cosine between the unit code and the unit prototype (no bias), as DINO's weight-normed last layer; else a plain linear layer
      const off = k * inH; let n = 1; if (this.norm) { n = 0; for (let d = 0; d < inH; d++) n += Wh[off + d] * Wh[off + d]; n = Math.sqrt(n) + 1e-8; }
      let s = this.norm ? 0 : bh[k]; for (let d = 0; d < inH; d++) s += Wh[off + d] / n * u[d]; l[k] = s; nw[k] = n;
    }
    return { l, h, u, nh, nw };
  }
  static softmax(l, tau, center) { const K = l.length, z = new Float64Array(K); let mx = -Infinity; for (let k = 0; k < K; k++) { z[k] = (l[k] - (center ? center[k] : 0)) / tau; if (z[k] > mx) mx = z[k]; } let Z = 0; for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); Z += z[k]; } for (let k = 0; k < K; k++) z[k] /= Z; return z; }
  // one step on a batch of pairs: the loss, the gradients and, when lr is given, the update of the student and the teacher's drift
  step(pairs, lr) {
    const N = pairs.length * 2, inH = this.H || this.code, g = this.student.newGradient(), gWh = new Float64Array(this.Wh.length), gbh = new Float64Array(this.K), gW1 = this.W1 ? new Float64Array(this.W1.length) : null, gb1 = this.b1 ? new Float64Array(this.H) : null;
    const views = []; for (const [x1, x2] of pairs) views.push(x1, x2);
    const sfw = views.map(x => this.student.forward(x)), sh = sfw.map(fw => this.head(fw.a[1], this.W1, this.b1, this.Wh, this.bh));
    const th = views.map(x => this.head(this.teacher.forward(x).a[1], this.W1T, this.b1T, this.WhT, this.bhT));
    const Pt = th.map(o => Distill.softmax(o.l, this.tauT, this.center)), Ps = sh.map(o => Distill.softmax(o.l, this.tauS, null));
    let loss = 0; const batchMean = new Float64Array(this.K), usage = new Float64Array(this.K); let sharp = 0;
    for (let i = 0; i < N; i++) { const target = Pt[i ^ 1], p = Ps[i]; for (let k = 0; k < this.K; k++) loss -= target[k] * Math.log(p[k] + 1e-12) / N; for (let k = 0; k < this.K; k++) { batchMean[k] += th[i].l[k] / N; usage[k] += Pt[i][k] / N; } let e = 0; for (let k = 0; k < this.K; k++) if (Pt[i][k] > 1e-12) e -= Pt[i][k] * Math.log(Pt[i][k]); sharp += e / N; }
    let usageEntropy = 0; for (let k = 0; k < this.K; k++) if (usage[k] > 1e-12) usageEntropy -= usage[k] * Math.log(usage[k]);
    for (let i = 0; i < N; i++) { // d loss / d student logits = (p − target) / tauS / N, back through the head and the encoder
      const target = Pt[i ^ 1], p = Ps[i], dl = new Float64Array(this.K); for (let k = 0; k < this.K; k++) dl[k] = (p[k] - target[k]) / this.tauS / N;
      const { h: hvec, u, nh, nw, l } = sh[i], du = new Float64Array(inH);
      for (let k = 0; k < this.K; k++) { const off = k * inH;
        if (this.norm) { const n = nw[k], lk = l[k]; for (let d = 0; d < inH; d++) { const wn = this.Wh[off + d] / n; gWh[off + d] += dl[k] * (u[d] - wn * lk) / n; du[d] += dl[k] * wn; } } // through w/|w|
        else { gbh[k] += dl[k]; for (let d = 0; d < inH; d++) { gWh[off + d] += dl[k] * u[d]; du[d] += dl[k] * this.Wh[off + d]; } } }
      let dh = du; if (this.norm) { let dot = 0; for (let d = 0; d < inH; d++) dot += u[d] * du[d]; dh = new Float64Array(inH); for (let d = 0; d < inH; d++) dh[d] = (du[d] - u[d] * dot) / nh; } // through h/|h|
      let dcode = dh;
      if (this.H) { dcode = new Float64Array(this.code); const code = sfw[i].a[1]; for (let j = 0; j < this.H; j++) { if (hvec[j] <= 0) continue; gb1[j] += dh[j]; for (let d = 0; d < this.code; d++) { gW1[j * this.code + d] += dh[j] * code[d]; dcode[d] += dh[j] * this.W1[j * this.code + d]; } } }
      const d0 = this.student.backHidden(sfw[i], dcode, g); this.student.convAccumulate(views[i], sfw[i], d0, g);
    }
    if (lr != null) {
      this.student.applyGradient(g, 1, lr, 0);
      for (let i = 0; i < this.Wh.length; i++) this.Wh[i] -= lr * gWh[i]; for (let k = 0; k < this.K; k++) this.bh[k] -= lr * gbh[k];
      if (this.H) { for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * gW1[i]; for (let j = 0; j < this.H; j++) this.b1[j] -= lr * gb1[j]; }
      this.syncTeacher(1 - this.m);
      for (let k = 0; k < this.K; k++) this.center[k] = this.cm * this.center[k] + (1 - this.cm) * batchMean[k];
      this.steps++;
    }
    let hits = 0; const codes = views.map(x => this.teacher.forward(x).a[1]); // the partner test on the teacher's code, as the page scores the contrastive one
    for (let i = 0; i < N; i++) { let best = -1, bs = -Infinity; for (let j = 0; j < N; j++) if (j !== i) { const s = Contrastive.cosine(codes[i], codes[j]); if (s > bs) { bs = s; best = j; } } if (best === (i ^ 1)) hits++; }
    return { loss, pairAcc: hits / N, usageEntropy, sharp, g, gWh, gbh, gW1, gb1 };
  }
  evaluate(pairs) { return this.step(pairs, null); }
}

// ---------------------------------------------------------------- the other teacher–student rule: predict the teacher's code (BYOL)
// No prototypes: the student's code goes through a small predictor and must land, as a direction, on the teacher's code
// of the other view; the teacher is again the slow average of the student. Nothing pushes different nuclei apart; the
// predictor and the teacher's slowness are what keep every code from becoming the same.
class Byol {
  constructor({ imageSize, conv = { K: 4, f: 5, pool: 4 }, code = 8, hidden = 16, momentum = 0.99, seed = 1 }) {
    this.size = imageSize; this.code = code; this.H = hidden; this.m = momentum; this.seed = seed; this.steps = 0;
    this.student = new Net({ inputSize: imageSize * imageSize, imageSize, conv, hidden: [code], activation: 'linear', seed });
    this.teacher = new Net({ inputSize: imageSize * imageSize, imageSize, conv, hidden: [code], activation: 'linear', seed });
    const rnd = mulberry32(seed * 7919 + 11), u = s => (rnd() * 2 - 1) * s, inH = hidden || code;
    this.W1 = hidden ? Float64Array.from({ length: hidden * code }, () => u(Math.sqrt(6 / (code + hidden)))) : null; this.b1 = hidden ? new Float64Array(hidden) : null;
    this.W2 = Float64Array.from({ length: code * inH }, () => u(Math.sqrt(6 / (inH + code)))); this.b2 = new Float64Array(code);
    this.syncTeacher(1);
  }
  get enc() { return probeOf === 'student' ? this.student : this.teacher; }
  encode(x) { return this.enc.forward(x).a[1]; }
  describe() { const c = this.student.conv; return `${c.K} filters ${c.f}×${c.f} + pool ${c.pool}×${c.pool} → code of ${this.code} → predictor ${this.H ? this.H + ' → ' : ''}${this.code}, aimed at the teacher's code`; }
  params() { return [['Wc', this.student.Wc, this.teacher.Wc], ['bc', this.student.bc, this.teacher.bc], ['W', this.student.W[0], this.teacher.W[0]], ['b', this.student.b[0], this.teacher.b[0]]]; }
  syncTeacher(rate) { for (const [, s, t] of this.params()) for (let i = 0; i < s.length; i++) t[i] += rate * (s[i] - t[i]); }
  predict(code) {
    let h = code; if (this.H) { h = new Float64Array(this.H); for (let j = 0; j < this.H; j++) { let v = this.b1[j]; for (let d = 0; d < this.code; d++) v += this.W1[j * this.code + d] * code[d]; h[j] = v > 0 ? v : 0; } }
    const inH = h.length, q = new Float64Array(this.code); for (let k = 0; k < this.code; k++) { let v = this.b2[k]; for (let d = 0; d < inH; d++) v += this.W2[k * inH + d] * h[d]; q[k] = v; }
    return { q, h };
  }
  step(pairs, lr) {
    const N = pairs.length * 2, C = this.code, inH = this.H || C, g = this.student.newGradient(), gW2 = new Float64Array(this.W2.length), gb2 = new Float64Array(C), gW1 = this.W1 ? new Float64Array(this.W1.length) : null, gb1 = this.b1 ? new Float64Array(this.H) : null;
    const views = []; for (const [x1, x2] of pairs) views.push(x1, x2);
    const sfw = views.map(x => this.student.forward(x)), tcodes = views.map(x => this.teacher.forward(x).a[1]), ut = tcodes.map(z => Contrastive.unit(z).u);
    let loss = 0;
    for (let i = 0; i < N; i++) {
      const code = sfw[i].a[1], { q, h } = this.predict(code), { u: uq, n: nq } = Contrastive.unit(q), target = ut[i ^ 1];
      let cos = 0; for (let k = 0; k < C; k++) cos += uq[k] * target[k]; loss += (2 - 2 * cos) / N;
      const du = new Float64Array(C); for (let k = 0; k < C; k++) du[k] = -2 * target[k] / N; // d loss / d unit prediction
      let dot = 0; for (let k = 0; k < C; k++) dot += uq[k] * du[k]; const dq = new Float64Array(C); for (let k = 0; k < C; k++) dq[k] = (du[k] - uq[k] * dot) / nq; // through q/|q|
      const dh = new Float64Array(inH); for (let k = 0; k < C; k++) { gb2[k] += dq[k]; for (let d = 0; d < inH; d++) { gW2[k * inH + d] += dq[k] * h[d]; dh[d] += dq[k] * this.W2[k * inH + d]; } }
      let dcode = dh;
      if (this.H) { dcode = new Float64Array(C); for (let j = 0; j < this.H; j++) { if (h[j] <= 0) continue; gb1[j] += dh[j]; for (let d = 0; d < C; d++) { gW1[j * C + d] += dh[j] * code[d]; dcode[d] += dh[j] * this.W1[j * C + d]; } } }
      const d0 = this.student.backHidden(sfw[i], dcode, g); this.student.convAccumulate(views[i], sfw[i], d0, g);
    }
    if (lr != null) {
      this.student.applyGradient(g, 1, lr, 0);
      for (let i = 0; i < this.W2.length; i++) this.W2[i] -= lr * gW2[i]; for (let k = 0; k < C; k++) this.b2[k] -= lr * gb2[k];
      if (this.H) { for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * gW1[i]; for (let j = 0; j < this.H; j++) this.b1[j] -= lr * gb1[j]; }
      this.syncTeacher(1 - this.m); this.steps++;
    }
    // the partner test on the teacher's code, and how spread the unit codes are (0 would be every view at one point: collapse)
    let hits = 0; for (let i = 0; i < N; i++) { let best = -1, bs = -Infinity; for (let j = 0; j < N; j++) if (j !== i) { let s = 0; for (let k = 0; k < C; k++) s += ut[i][k] * ut[j][k]; if (s > bs) { bs = s; best = j; } } if (best === (i ^ 1)) hits++; }
    const mean = new Float64Array(C); for (const u of ut) for (let k = 0; k < C; k++) mean[k] += u[k] / N; let spread = 0; for (const u of ut) for (let k = 0; k < C; k++) spread += (u[k] - mean[k]) ** 2 / N; spread = Math.sqrt(spread / C);
    return { loss, pairAcc: hits / N, spread, g, gW1, gb1, gW2, gb2 };
  }
  evaluate(pairs) { return this.step(pairs, null); }
}

// ---------------------------------------------------------------- the gradient, checked by finite differences
function gradientCheck(norm) {
  const d = new Distill({ imageSize: size, code: FM.code, K: 8, hidden: 6, norm, seed: 3 }), rng = mulberry32(5);
  const pairs = [0, 1, 2].map(i => [randomView(set[i], rng), randomView(set[i], rng)]);
  const r = d.step(pairs, null), eps = 1e-5; let worst = 0, checked = 0;
  const probes = [['Wc', d.student.Wc, r.g.gWc], ['W', d.student.W[0], r.g.gW[0]], ['b', d.student.b[0], r.g.gb[0]], ['Wh', d.Wh, r.gWh], ['W1', d.W1, r.gW1], ['b1', d.b1, r.gb1]]; if (!norm) probes.push(['bh', d.bh, r.gbh]);
  for (const [, arr, grad] of probes) for (let t = 0; t < 4; t++) { const i = Math.floor(rng() * arr.length), v = arr[i]; arr[i] = v + eps; const lp = d.step(pairs, null).loss; arr[i] = v - eps; const lm = d.step(pairs, null).loss; arr[i] = v; const num = (lp - lm) / (2 * eps), an = grad[i], rel = Math.abs(num - an) / Math.max(1e-6, Math.abs(num) + Math.abs(an)); worst = Math.max(worst, rel); checked++; }
  return { worst, checked };
}
function gradientCheckByol() {
  const d = new Byol({ imageSize: size, code: FM.code, hidden: 6, seed: 3 }), rng = mulberry32(5);
  const pairs = [0, 1, 2].map(i => [randomView(set[i], rng), randomView(set[i], rng)]);
  const r = d.step(pairs, null), eps = 1e-5; let worst = 0, checked = 0;
  const probes = [['Wc', d.student.Wc, r.g.gWc], ['W', d.student.W[0], r.g.gW[0]], ['b', d.student.b[0], r.g.gb[0]], ['W2', d.W2, r.gW2], ['b2', d.b2, r.gb2], ['W1', d.W1, r.gW1], ['b1', d.b1, r.gb1]];
  for (const [, arr, grad] of probes) for (let t = 0; t < 4; t++) { const i = Math.floor(rng() * arr.length), v = arr[i]; arr[i] = v + eps; const lp = d.step(pairs, null).loss; arr[i] = v - eps; const lm = d.step(pairs, null).loss; arr[i] = v; const num = (lp - lm) / (2 * eps), an = grad[i], rel = Math.abs(num - an) / Math.max(1e-6, Math.abs(num) + Math.abs(an)); worst = Math.max(worst, rel); checked++; }
  return { worst, checked };
}

// ---------------------------------------------------------------- the probes, as on the page
function heldEval(model) { let loss = 0, hit = 0, nb = 0; for (let b = 0; b < heldPairs.length; b += FM.heldBatch) { const r = model.evaluate(heldPairs.slice(b, b + FM.heldBatch)); loss += r.loss; hit += r.pairAcc; nb++; } return { loss: loss / nb, pairAcc: hit / nb }; }
function probe(model, id, testLab = 'A') {
  const ds = tasks[id], tr = [...ds.train.filter(s => s.label).slice(0, FM.probePerClass), ...ds.train.filter(s => !s.label).slice(0, FM.probePerClass)];
  const code = (s, lab) => model.encode(std.apply(s.variants[lab].ink));
  const rows = tr.map(s => code(s, 'A')), labels = tr.map(s => s.label), st = fitStandardizer(rows, { perDimScale: true });
  const net = new Net({ inputSize: FM.code, hidden: [], activation: 'relu', seed: 1 }), rnd = mulberry32(1), idx = rows.map((_, i) => i);
  for (let ep = 0; ep < FM.probeEpochs; ep++) { for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } for (let b = 0; b < idx.length; b += 8) { const bb = idx.slice(b, b + 8); net.trainBatch(bb.map(i => st.apply(rows[i])), bb.map(i => labels[i]), FM.probeLr, 0); } }
  const scored = [...ds.test, ...held.filter(e => e.task === id).map(e => e.s)];
  const Xs = scored.map(s => st.apply(code(s, testLab))), acc = net.evaluate(Xs, scored.map(s => s.label)).accuracy;
  if (!knn) return acc;
  const Xt = rows.map(r => st.apply(r)); let hit = 0; // 5 nearest labelled cases by cosine of the standardised code, majority vote
  Xs.forEach((x, i) => { const near = Xt.map((t, j) => ({ s: Contrastive.cosine(x, t), y: labels[j] })).sort((a, b) => b.s - a.s).slice(0, 5); const votes = near.reduce((a, o) => a + (o.y ? 1 : -1), 0); if ((votes > 0 ? 1 : 0) === scored[i].label) hit++; });
  return { acc, knn: hit / Xs.length };
}
function train(model, seed, lr, log) {
  const rng = mulberry32(seed * 31 + 7), order = set.map((_, i) => i), marks = [];
  for (let e = 1; e <= epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let loss = 0, nb = 0, ue = 0, sh = 0, sp = 0;
    for (let b0 = 0; b0 < order.length; b0 += FM.batch) { const pairs = order.slice(b0, b0 + FM.batch).map(i => [randomView(set[i], rng), randomView(set[i], rng)]); const r = model.step(pairs, lr); loss += r.loss; ue += r.usageEntropy || 0; sh += r.sharp || 0; sp += r.spread || 0; nb++; }
    if (e === epochs || e % 25 === 0) { const h = heldEval(model), pr = FM.tasks.map(id => probe(model, id)), acc = pr.map(r => (knn ? r.acc : r)), kn = knn ? pr.map(r => r.knn) : null; marks.push({ epoch: e, loss: loss / nb, held: h, acc, knn: kn, usage: ue / nb, sharp: sh / nb, spread: sp / nb }); if (log) log(marks[marks.length - 1]); }
  }
  return marks;
}
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';
for (const norm of [true, false]) { const gc = gradientCheck(norm); console.log(`distillation gradient check (${norm ? 'normalised' : 'plain'} head): worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} weights ${gc.worst < 1e-4 ? '(ok)' : '(FAILED)'}`); }
{ const gc = gradientCheckByol(); console.log(`BYOL gradient check: worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} weights ${gc.worst < 1e-4 ? '(ok)' : '(FAILED)'}`); }
const configs = grid === 'full' ? [
  { name: 'contrastive', kind: 'cl' },
  { name: 'K16 τt.04 m.99', K: 16, tauT: 0.04, momentum: 0.99 }, { name: 'K16 τt.07 m.99', K: 16, tauT: 0.07, momentum: 0.99 }, { name: 'K32 τt.04 m.99', K: 32, tauT: 0.04, momentum: 0.99 },
  { name: 'K16 τt.04 m.9', K: 16, tauT: 0.04, momentum: 0.9 }, { name: 'K16 τt.04 m.99 lr.1', K: 16, tauT: 0.04, momentum: 0.99, lr: 0.1 }, { name: 'K16 τt.04 m.99 mlp16', K: 16, tauT: 0.04, momentum: 0.99, hidden: 16 },
  { name: 'K8 τt.04 m.99', K: 8, tauT: 0.04, momentum: 0.99 }, { name: 'K32 τt.07 m.99 mlp16', K: 32, tauT: 0.07, momentum: 0.99, hidden: 16 }, { name: 'K16 τt.04 m.99 τs.2', K: 16, tauT: 0.04, momentum: 0.99, tauS: 0.2 },
] : grid === 'knn' ? [{ name: 'contrastive', kind: 'cl' }, { name: 'K32 τt.04 m.99', K: 32, tauT: 0.04, momentum: 0.99 }, { name: 'byol linear m.99', kind: 'byol', hidden: 0, momentum: 0.99 }] : grid === 'byol' ? [
  { name: 'contrastive', kind: 'cl' },
  { name: 'byol mlp16 m.99', kind: 'byol', hidden: 16, momentum: 0.99 }, { name: 'byol mlp16 m.9', kind: 'byol', hidden: 16, momentum: 0.9 }, { name: 'byol linear m.99', kind: 'byol', hidden: 0, momentum: 0.99 },
  { name: 'byol mlp16 m.99 lr.1', kind: 'byol', hidden: 16, momentum: 0.99, lr: 0.1 }, { name: 'byol mlp32 m.99', kind: 'byol', hidden: 32, momentum: 0.99 },
] : grid === 'long' ? [{ name: 'contrastive', kind: 'cl' }, { name: 'K32 τt.04 m.99', K: 32, tauT: 0.04, momentum: 0.99 }, { name: 'K64 τt.04 m.99 lr.1', K: 64, tauT: 0.04, momentum: 0.99, lr: 0.1 }] : grid === 'round2' ? [
  { name: 'contrastive', kind: 'cl' },
  { name: 'K32 τt.04 m.99 lr.1', K: 32, tauT: 0.04, momentum: 0.99, lr: 0.1 }, { name: 'K64 τt.04 m.99', K: 64, tauT: 0.04, momentum: 0.99 }, { name: 'K64 τt.04 m.99 lr.1', K: 64, tauT: 0.04, momentum: 0.99, lr: 0.1 },
  { name: 'K64 τt.07 m.99 lr.1', K: 64, tauT: 0.07, momentum: 0.99, lr: 0.1 }, { name: 'K100 τt.04 m.99 lr.1', K: 100, tauT: 0.04, momentum: 0.99, lr: 0.1 }, { name: 'K64 τt.04 m.9 lr.1', K: 64, tauT: 0.04, momentum: 0.9, lr: 0.1 },
] : [{ name: 'contrastive', kind: 'cl' }, { name: 'K16 τt.04 m.99', K: 16, tauT: 0.04, momentum: 0.99 }, { name: 'K16 τt.04 m.99 mlp16', K: 16, tauT: 0.04, momentum: 0.99, hidden: 16 }, { name: 'K32 τt.07 m.9', K: 32, tauT: 0.07, momentum: 0.9 }];
console.log(`\n${epochs} epochs · seeds 1..${nSeeds} · probe on the ${probeOf}'s code · columns: batch loss · held-out partner found · atypia / enlargement / irregularity probes (mean) · then prototype usage entropy of ln K and mean sharpness (distillation) or the spread of the unit codes (BYOL)`);
for (const c of configs) {
  const rows = [];
  for (let seed = 1; seed <= nSeeds; seed++) {
    const t0 = Date.now();
    const model = c.kind === 'cl' ? new Contrastive({ imageSize: size, conv: { K: FM.K, f: 5, pool: 4 }, code: FM.code, tau: FM.tau, seed }) : c.kind === 'byol' ? new Byol({ imageSize: size, code: FM.code, hidden: c.hidden, momentum: c.momentum, seed }) : new Distill({ imageSize: size, code: FM.code, K: c.K, tauT: c.tauT, tauS: c.tauS || 0.1, momentum: c.momentum, hidden: c.hidden || 0, seed });
    const marks = train(model, seed, c.lr || FM.lr), last = marks[marks.length - 1], mean = last.acc.reduce((a, b) => a + b, 0) / 3;
    rows.push({ seed, last, mean, ms: (Date.now() - t0) / epochs });
    console.log(`  ${c.name.padEnd(22)} seed ${seed}: loss ${last.loss.toFixed(3)} · partner ${pc(last.held.pairAcc)} · ${last.acc.map(pc).join(' /')} (${pc(mean)})${knn ? ` · 5-NN ${last.knn.map(pc).join(' /')}` : ''}${c.kind === 'cl' ? '' : c.kind === 'byol' ? ` · spread ${last.spread.toFixed(3)}` : ` · usage ${(last.usage / Math.log(c.K)).toFixed(2)} · sharp ${last.sharp.toFixed(2)}`} · ${rows[rows.length - 1].ms.toFixed(0)} ms/epoch`);
    if (c.kind !== 'cl') for (const m of marks.slice(0, -1)) console.log(`      epoch ${String(m.epoch).padStart(3)}: loss ${m.loss.toFixed(3)} · partner ${pc(m.held.pairAcc)} · ${m.acc.map(pc).join(' /')}${c.kind === 'byol' ? ` · spread ${m.spread.toFixed(3)}` : ` · usage ${(m.usage / Math.log(c.K)).toFixed(2)} · sharp ${m.sharp.toFixed(2)}`}`);
  }
  const m = rows.reduce((a, r) => a + r.mean, 0) / rows.length, p = rows.reduce((a, r) => a + r.last.held.pairAcc, 0) / rows.length;
  console.log(`  ${c.name.padEnd(22)} mean of ${rows.length} seed${rows.length === 1 ? '' : 's'}: probes ${pc(m)} · partner ${pc(p)}\n`);
}
