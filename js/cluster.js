/*
 * cluster.js — the Cluster question: k-means on the training cases, with no label anywhere, and a two-component map
 * to draw them on. Works in the browser (window.NucleusCluster) and in Node (module.exports), with no dependencies.
 *
 * k-means as Lloyd wrote it: k centres, every case joins its nearest centre, every centre moves to the mean of its
 * cases, repeat until nobody moves. The centres start by k-means++ (the first a random case, each next one a case
 * drawn with probability proportional to its squared distance from the centres so far), and since a single start
 * can settle in a poor local optimum, `start` tries several seedings and keeps the tightest result, as the usual
 * implementations do; the stepping then replays the winning start so that the lecture watches the run that ends best.
 * `startRandom` is the plain start, k random cases, for watching the rounds and the local optima they end in.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NucleusCluster = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function dist2(a, b) { let d = 0; for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; d += t * t; } return d; }
  function nearest(x, C) { let b = 0, bd = Infinity; for (let j = 0; j < C.length; j++) { const d = dist2(x, C[j]); if (d < bd) { bd = d; b = j; } } return b; }
  function seedCentres(X, k, rng) { // k-means++; returns the centres and which cases they are
    const n = X.length, idx = [Math.floor(rng() * n)], C = [Array.from(X[idx[0]])];
    while (C.length < k) {
      const d = X.map(x => dist2(x, C[nearest(x, C)])), tot = d.reduce((a, b) => a + b, 0);
      let r = rng() * tot, i = 0; while (i < n - 1 && (r -= d[i]) > 0) i++;
      idx.push(i); C.push(Array.from(X[i]));
    }
    return { C, idx };
  }
  function assign(X, C) { return X.map(x => nearest(x, C)); }
  function means(X, A, k, C) { // a centre with no case keeps its place
    const D = X[0].length, M = [], N = new Array(k).fill(0);
    for (let j = 0; j < k; j++) M.push(new Array(D).fill(0));
    for (let i = 0; i < X.length; i++) { const j = A[i]; N[j]++; const m = M[j], x = X[i]; for (let d = 0; d < D; d++) m[d] += x[d]; }
    return M.map((m, j) => (N[j] ? m.map(v => v / N[j]) : Array.from(C[j])));
  }
  function inertia(X, C, A) { let s = 0; for (let i = 0; i < X.length; i++) s += dist2(X[i], C[A[i]]); return s; }
  // one Lloyd iteration on a state {X, k, C, A, it, moved, converged}; returns the state
  function step(st) { // a round: every centre to the mean of its cases, then every case to its nearest centre
    if (st.converged) return st;
    st.C = means(st.X, st.A, st.k, st.C);
    const A = assign(st.X, st.C); let moved = 0; for (let i = 0; i < A.length; i++) if (A[i] !== st.A[i]) moved++;
    st.A = A; st.it++; st.moved = moved; st.converged = moved === 0; st.inertia = inertia(st.X, st.C, st.A);
    return st;
  }
  function run(st, maxIt = 200) { while (!st.converged && st.it < maxIt) step(st); return st; }
  // a fresh state from a seed: the tightest of `starts` k-means++ seedings, rewound to its first centres
  function start(X, k, seed, starts = 10) {
    let best = null;
    for (let s = 0; s < starts; s++) {
      const seeded = seedCentres(X, k, mulberry32((seed * 1000003 + s * 7919 + k) | 0)), C = seeded.C.map(c => Array.from(c));
      const st = run({ X, k, C, A: assign(X, C), it: 0, moved: 0, converged: false, inertia: 0 });
      if (!best || st.inertia < best.inertia - 1e-9) best = { inertia: st.inertia, C0: seeded.C, idx: seeded.idx, it: st.it };
    }
    const st = { X, k, C: best.C0.map(c => Array.from(c)), A: assign(X, best.C0), it: 0, moved: 0, converged: false, inertia: 0, starts, seeds: best.idx, final: best.inertia, finalIt: best.it };
    st.inertia = inertia(X, st.C, st.A);
    return st;
  }
  // the plain start: k distinct random cases as the first centres, one run, so that the rounds are there to watch and
  // another seed ends elsewhere (the salt makes seed 1 a long run to the optimum on the page's blood counts)
  function startRandom(X, k, seed) {
    const rng = mulberry32((seed * 1000003 + k * 7919 + 44) | 0), idx = [];
    while (idx.length < k) { const i = Math.floor(rng() * X.length); if (!idx.includes(i)) idx.push(i); }
    const C = idx.map(i => Array.from(X[i])), st = { X, k, C, A: assign(X, C), it: 0, moved: 0, converged: false, inertia: 0, starts: 1, seeds: idx };
    st.inertia = inertia(X, C, st.A);
    return st;
  }
  // the silhouette: for every case, how much nearer its own cluster is than the nearest other, (b − a) / max(a, b),
  // averaged; 1 is tight and apart, 0 is on the border, negative is in the wrong cluster
  function silhouette(X, A, k) {
    const n = X.length, members = Array.from({ length: k }, () => []); A.forEach((j, i) => members[j].push(i));
    let tot = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      const own = members[A[i]]; if (own.length <= 1) continue;
      let a = 0; for (const o of own) if (o !== i) a += Math.sqrt(dist2(X[i], X[o])); a /= own.length - 1;
      let b = Infinity; for (let j = 0; j < k; j++) { if (j === A[i] || !members[j].length) continue; let s = 0; for (const o of members[j]) s += Math.sqrt(dist2(X[i], X[o])); s /= members[j].length; if (s < b) b = s; }
      if (b === Infinity) continue; tot += (b - a) / Math.max(a, b); cnt++;
    }
    return cnt ? tot / cnt : 0;
  }
  // what each k gives, each from the tightest of ten k-means++ starts: the within-cluster distance and the silhouette
  function sweep(X, ks, seed = 1) { return ks.map(k => { const st = run(start(X, k, seed)); return { k, inertia: st.inertia, silhouette: silhouette(X, st.A, k) }; }); }
  // the clusters against a labelling (the hidden subtypes, or the class): counts, each cluster's majority, the purity
  function contingency(A, labels, k) {
    const rows = []; let agree = 0;
    for (let j = 0; j < k; j++) {
      const counts = new Map(); let n = 0;
      for (let i = 0; i < A.length; i++) if (A[i] === j) { n++; counts.set(labels[i], (counts.get(labels[i]) || 0) + 1); }
      let majority = null, top = 0; for (const [l, c] of counts) if (c > top) { top = c; majority = l; }
      agree += top; rows.push({ counts, n, majority, purity: n ? top / n : 0 });
    }
    return { rows, purity: A.length ? agree / A.length : 0 };
  }
  // the two directions of greatest spread, from the n × n Gram matrix (cheap when the cases are few and the pixels many)
  function pca2(X) {
    const n = X.length, D = X[0].length, mean = new Array(D).fill(0);
    for (const x of X) for (let d = 0; d < D; d++) mean[d] += x[d] / n;
    const Xc = X.map(x => x.map((v, d) => v - mean[d]));
    const G = Xc.map(a => Xc.map(b => { let s = 0; for (let d = 0; d < D; d++) s += a[d] * b[d]; return s; }));
    let trace = 0; for (let i = 0; i < n; i++) trace += G[i][i];
    const axes = [], values = [];
    for (let c = 0; c < 2; c++) {
      let v = Array.from({ length: n }, (_, i) => Math.sin(i * 0.7 + c)), lambda = 0; // a fixed start, so the map is the same on every visit
      for (let it = 0; it < 300; it++) {
        const w = G.map(row => { let s = 0; for (let i = 0; i < n; i++) s += row[i] * v[i]; return s; });
        for (const a of axes) { let p = 0; for (let i = 0; i < n; i++) p += w[i] * a[i]; for (let i = 0; i < n; i++) w[i] -= p * a[i]; } // keep it orthogonal to the first
        const norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1; const nv = w.map(x => x / norm);
        let diff = 0; for (let i = 0; i < n; i++) diff += Math.abs(nv[i] - v[i]); v = nv; lambda = norm; if (diff < 1e-9) break;
      }
      axes.push(v); values.push(lambda);
    }
    const dirs = axes.map(v => { const u = new Array(D).fill(0); for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) u[d] += v[i] * Xc[i][d]; const norm = Math.sqrt(u.reduce((s, x) => s + x * x, 0)) || 1; return u.map(x => x / norm); });
    const project = x => dirs.map(u => { let s = 0; for (let d = 0; d < D; d++) s += (x[d] - mean[d]) * u[d]; return s; });
    return { mean, dirs, explained: values.map(l => (trace ? l / trace : 0)), project };
  }
  return { dist2, nearest, assign, means, inertia, step, run, start, startRandom, silhouette, sweep, contingency, pca2 };
});
