#!/usr/bin/env node
/*
 * check_fields_cnn.js — the invasion question end to end, a feasibility experiment: a convolutional encoder over the field at half resolution, three
 * levels of two 3 × 3 convolutions with 2 × 2 pooling and two more convolutions at the bottom, so that the field becomes
 * an 11 × 8 grid of cells (one per 16 px of field, about a nucleus each); every cell is a token (its channels plus its
 * position), and the fields' attention stack sits on top unchanged: two context layers with distance costs, two
 * heads (CIS, invasion). Only the field's two labels train it; no nucleus is ever segmented, no code pretrained.
 *   node tools/check_fields_cnn.js [--levels 3] [--channels 8] [--context 1] [--pos 1] [--epochs 60] [--seed 1] [--lr 0.003] [--lr-att 0.02] [--batch 1] [--norm 1] [--out file.json] [--check]
 * With --norm 1 every cell's token is standardised over its channels (a layer norm without gain), as the frozen codes the
 * attention stack was built for are standardised: it keeps the tokens' scale from drifting as the encoder learns.
 */
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const NN = require(path.join(ROOT, 'js', 'nn.js')), FL = require(path.join(ROOT, 'js', 'fields.js'));
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d), num = (k, d) => +arg(k, d);
const cfg = { channels: num('--channels', 8), context: num('--context', 1), pos: num('--pos', 1), epochs: num('--epochs', 60), seed: num('--seed', 1), lr: num('--lr', 0.003), lrAtt: num('--lr-att', 0.02), batch: num('--batch', 1), norm: num('--norm', 1), levels: num('--levels', 3), decay: 1e-4, decayAtt: 0.001, clip: 1, out: arg('--out', null), check: process.argv.includes('--check') };
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(ROOT, 'data', 'fields', 'fields_data.js')).LECTURE_FIELDS, W0 = D.meta.w, H0 = D.meta.h, W = W0 >> 1, H = H0 >> 1, UNIT = 16;
const cisOf = f => (['cis', 'cisvbn', 'inv'].includes(f.pattern) ? 1 : 0), pc = v => `${Math.round(v * 100)}%`;
const conv3 = NN.SegNet.prototype.conv3, conv3Back = NN.SegNet.prototype.conv3Back, pool2 = NN.SegNet.prototype.pool2;

// ---- the encoder: levels of (conv, conv, pool) and a last pair of convolutions on the coarsest map
class ConvTokens {
  constructor({ channels = 8, levels = 3, seed = 1, norm = true }) {
    this.C = channels; this.L = levels; this.norm = !!norm; const rnd = NN.mulberry32(seed * 104729 + 11), mk = (n, fanIn) => { const s = Math.sqrt(6 / fanIn); return Float64Array.from({ length: n }, () => (rnd() * 2 - 1) * s); };
    const conv = (cin, cout) => ({ W: mk(cout * cin * 9, cin * 9), b: new Float64Array(cout), cin, cout });
    this.enc = []; for (let l = 0; l <= levels; l++) this.enc.push([conv(l === 0 ? 1 : channels, channels), conv(channels, channels)]);
    this.adam = null; this.steps = 0;
  }
  layers() { const out = []; for (const pair of this.enc) out.push(pair[0], pair[1]); return out; }
  parameterCount() { return this.layers().reduce((a, c) => a + c.W.length + c.b.length, 0); }
  newGradient() { return this.layers().map(c => ({ W: new Float64Array(c.W.length), b: new Float64Array(c.b.length) })); }
  forward(x, h, w) { // x: the half-resolution ink minus its mean; returns the levels and the tokens (one per cell, C numbers)
    const C = this.C, levels = []; let cur = x, cin = 1;
    for (let l = 0; l <= this.L; l++) { const c1 = conv3(cur, this.enc[l][0], h, w, true), c2 = conv3(c1.act, this.enc[l][1], h, w, true), lev = { x: cur, cin, h, w, c1, c2 }; levels.push(lev); if (l < this.L) { const p = pool2(c2.act, C, h, w); lev.pool = p; cur = p.v; h = p.h; w = p.w; cin = C; } }
    const top = levels[this.L], n = top.h * top.w, raw = Array.from({ length: n }, (_, i) => Float64Array.from({ length: C }, (_, c) => top.c2.act[c * n + i]));
    if (!this.norm) return { levels, tokens: raw, h: top.h, w: top.w };
    const sig = new Float64Array(n), tokens = raw.map((a, i) => { let mu = 0; for (const v of a) mu += v; mu /= C; let va = 0; for (const v of a) va += (v - mu) * (v - mu); const s = Math.sqrt(va / C + 1e-6); sig[i] = s; return Float64Array.from(a, v => (v - mu) / s); });
    return { levels, tokens, raw, sig, h: top.h, w: top.w };
  }
  backward(fw, dTokens, g) { // dTokens: n × C gradients at the tokens; accumulates the weight gradients into g
    const C = this.C, top = fw.levels[this.L], n = top.h * top.w; let d = new Float64Array(C * n);
    if (this.norm) dTokens = dTokens.map((dy, i) => { const y = fw.tokens[i], s = fw.sig[i]; let mdy = 0, mdyy = 0; for (let c = 0; c < C; c++) { mdy += dy[c]; mdyy += dy[c] * y[c]; } mdy /= C; mdyy /= C; return Float64Array.from(dy, (v, c) => (v - mdy - y[c] * mdyy) / s); }); // back through the standardisation
    for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) d[c * n + i] = dTokens[i][c];
    for (let l = this.L; l >= 0; l--) { const lev = fw.levels[l], m = lev.h * lev.w; let dAct = new Float64Array(C * m); if (l === this.L) dAct.set(d); else { const p = lev.pool; for (let j = 0; j < d.length; j++) dAct[p.arg[j]] += d[j]; }
      const d2 = new Float64Array(C * m); for (let i = 0; i < d2.length; i++) d2[i] = lev.c2.pre[i] > 0 ? dAct[i] : 0; const d1a = conv3Back(lev.c1.act, this.enc[l][1], lev.h, lev.w, d2, g[2 * l + 1]);
      const d1 = new Float64Array(C * m); for (let i = 0; i < d1.length; i++) d1[i] = lev.c1.pre[i] > 0 ? d1a[i] : 0; d = conv3Back(lev.x, this.enc[l][0], lev.h, lev.w, d1, g[2 * l]); }
  }
  apply(g, k, lr, l2, clip) { // Adam, the step's mean gradient clipped to a norm
    let n2 = 0; for (const gl of g) { for (const v of gl.W) n2 += v * v; for (const v of gl.b) n2 += v * v; } const norm = Math.sqrt(n2) / k; this.lastGradientNorm = norm;
    if (clip > 0 && norm > clip) { const f = clip / norm; for (const gl of g) { for (let i = 0; i < gl.W.length; i++) gl.W[i] *= f; for (let i = 0; i < gl.b.length; i++) gl.b[i] *= f; } }
    const st = this.adam || (this.adam = { t: 0, m: this.newGradient(), v: this.newGradient() }), b1 = 0.9, b2 = 0.999; st.t++; const c1 = 1 - Math.pow(b1, st.t), c2 = 1 - Math.pow(b2, st.t);
    const step = (p, gp, m, v, decay) => { for (let i = 0; i < p.length; i++) { const gi = gp[i] / k + (decay ? l2 * p[i] : 0); m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi; p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); } };
    this.layers().forEach((c, i) => { step(c.W, g[i].W, st.m[i].W, st.v[i].W, 1); step(c.b, g[i].b, st.m[i].b, st.v[i].b, 0); }); this.steps++;
  }
  toJSON() { const r = v => Array.from(v, q => +q.toPrecision(5)); return { channels: this.C, levels: this.L, norm: this.norm, steps: this.steps, enc: this.enc.map(pair => pair.map(c => ({ W: r(c.W), b: r(c.b), cin: c.cin, cout: c.cout }))) }; }
}

// ---- the fields at half resolution (2 × 2 mean of the ink), the cell grid's positions and distances, the labels and the cells' truth
const grid = (gh, gw) => { const xy = [], cx = []; for (let i = 0; i < gh; i++) for (let j = 0; j < gw; j++) { xy.push([(j + 0.5) / gw * 2 - 1, (i + 0.5) / gh * 2 - 1]); cx.push([(j + 0.5) * W0 / gw, (i + 0.5) * H0 / gh]); } const dist = cx.map(a => Float64Array.from(cx, b => Math.hypot(a[0] - b[0], a[1] - b[1]) / UNIT)); return { xy, dist }; };
const GH = H >> cfg.levels, GW = W >> cfg.levels, G = grid(GH, GW); // one pooling per level: 64 × 88 → 8 × 11 with three, 16 × 22 with two
const prep = f => { const raw = FL.decodePNGNode(f.png).px, px = FL.withGrain(raw, f.grainSeed, D.meta.grain), x = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) { let s = 0; for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) s += 1 - px[(2 * y + dy) * W0 + 2 * xx + dx] / 255; x[y * W + xx] = s / 4; }
  const cellOf = n => Math.min(GH - 1, Math.floor(n.y / H0 * GH)) * GW + Math.min(GW - 1, Math.floor(n.x / W0 * GW)), posA = new Array(GH * GW).fill(false), posB = new Array(GH * GW).fill(false);
  for (const n of f.nuclei.map(arr => Object.fromEntries(D.meta.nucleus.map((k, i) => [k, arr[i]])))) { if (n.atypical) posA[cellOf(n)] = true; if (n.atypical && n.below) posB[cellOf(n)] = true; }
  return { name: f.name, split: f.split, pattern: f.pattern, x, y: [cisOf(f), f.label], pos: [posA, posB] }; };
const t0 = Date.now(), train = D.train.map(prep), test = D.test.map(prep); let meanInk = 0; for (const f of train) for (const v of f.x) meanInk += v; meanInk /= train.length * W * H;
for (const f of train.concat(test)) for (let i = 0; i < f.x.length; i++) f.x[i] -= meanInk;
const patterns = D.meta.patterns || [...new Set(test.map(f => f.pattern))].map(key => ({ key, name: key }));
console.log(`fields ${train.length} training, ${test.length} test · ${W} × ${H} px at half resolution → ${GW} × ${GH} cells · prepared in ${((Date.now() - t0) / 1000).toFixed(1)} s · mean ink ${meanInk.toFixed(3)}`);

// ---- the model: the encoder and the attention stack over its tokens
const DT = cfg.channels + (cfg.pos ? 2 : 0);
const enc = new ConvTokens({ channels: cfg.channels, levels: cfg.levels, seed: cfg.seed, norm: !!cfg.norm });
const att = new NN.AttentionMIL({ inputSize: DT, attentionUnits: 4, outputs: 2, clip: 20, context: cfg.context ? { dk: 8, ffn: 8, heads: 1, layers: 2, distanceBias: true, excludeSelf: true, costInit: 1.5 } : null, seed: cfg.seed });
const tokensOf = (fw, mirror) => fw.tokens.map((t, i) => { const [px, py] = G.xy[i]; return cfg.pos ? Float64Array.from([...t, mirror ? -px : px, py]) : t; });
const rng = NN.mulberry32(cfg.seed * 31 + 7), gauss = () => FL.gaussianFrom(rng);
const augment = f => { // a mirror half the time, and a shift of up to 4 px sideways and 2 px up or down, the edge filled with the pale corner value
  const mirror = rng() < 0.5, sx = Math.round(gauss() * 2), sy = Math.round(gauss() * 1), x = new Float64Array(W * H), fill = -meanInk;
  for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) { let X = xx - sx, Y = y - sy; if (mirror) X = W - 1 - X; x[y * W + xx] = X < 0 || X >= W || Y < 0 || Y >= H ? fill : f.x[Y * W + X]; }
  return { x, mirror };
};
const forward = (x, mirror) => { const fw = enc.forward(x, H, W); return { fw, H: tokensOf(fw, mirror) }; };
const trainStep = fields => { // one step on a batch of fields: the attention stack averages its gradient over them, the encoder likewise
  const runs = fields.map(f => { const a = augment(f), r = forward(a.x, a.mirror); return { fw: r.fw, slide: { H: r.H, dist: G.dist, xy: G.xy, y: f.y } }; }), collect = [], g = enc.newGradient();
  const loss = att.trainBatch(runs.map(r => r.slide), cfg.lrAtt, cfg.decayAtt, collect);
  runs.forEach((r, i) => enc.backward(r.fw, collect[i].map(d => d.subarray(0, cfg.channels)), g)); enc.apply(g, runs.length, cfg.lr, cfg.decay, cfg.clip); return loss; };
const evaluate = set => { const slides = set.map(f => ({ H: forward(f.x, false).H, dist: G.dist, xy: G.xy, y: f.y, pos: f.pos, pattern: f.pattern })); const ev = att.evaluate(slides); return { ev, slides }; };

// ---- a finite-difference check of the gradient through the encoder on a small field
if (cfg.check) {
  const h = 16, w = 16, rnd = NN.mulberry32(5), x = Float64Array.from({ length: h * w }, () => rnd() - 0.5), Gs = grid(h >> cfg.levels, w >> cfg.levels), y = [1, 0];
  const lossOf = () => { const fw = enc.forward(x, h, w), toks = fw.tokens.map((t, i) => (cfg.pos ? Float64Array.from([...t, Gs.xy[i][0], Gs.xy[i][1]]) : t)), o = att.forward(toks, Gs.dist, Gs.xy); return o.outs.reduce((a, out, k) => a - (y[k] ? Math.log(out.p + 1e-12) : Math.log(1 - out.p + 1e-12)), 0); };
  const fw = enc.forward(x, h, w), toks = fw.tokens.map((t, i) => (cfg.pos ? Float64Array.from([...t, Gs.xy[i][0], Gs.xy[i][1]]) : t)), collect = [], g = enc.newGradient();
  att.trainBatch([{ H: toks, dist: Gs.dist, xy: Gs.xy, y }], 0, 0, collect); enc.backward(fw, collect[0].map(d => d.subarray(0, cfg.channels)), g);
  const params = enc.layers(), eps = 1e-5; let worst = 0, checked = 0;
  for (let li = 0; li < params.length; li++) for (const key of ['W', 'b']) { const p = params[li][key], gp = g[li][key]; for (let t = 0; t < 12; t++) { const i = Math.floor(rnd() * p.length), o = p[i]; p[i] = o + eps; const lp = lossOf(); p[i] = o - eps; const lm = lossOf(); p[i] = o; const numeric = (lp - lm) / (2 * eps); if (Math.abs(numeric) > 1e-7) { checked++; worst = Math.max(worst, Math.abs(numeric - gp[i]) / (Math.abs(numeric) + Math.abs(gp[i]))); } } }
  console.log(`gradient check through the encoder: worst relative error ${worst.toExponential(2)} over ${checked} weights`); process.exit(worst < 1e-4 ? 0 : 1);
}

// ---- training, one field per step, with the test measures after every epoch
console.log(`encoder ${enc.parameterCount()} parameters (${cfg.channels} channels, ${cfg.levels} levels, ${GW * GH} tokens of ${DT}) · attention ${att.parameterCount ? att.parameterCount() : '?'} · context ${cfg.context ? 'on' : 'off'} · positions ${cfg.pos ? 'on' : 'off'} · tokens ${cfg.norm ? 'standardised' : 'raw'} · batch ${cfg.batch} · lr ${cfg.lr} (encoder, Adam) ${cfg.lrAtt} (attention) · seed ${cfg.seed}`);
const order = train.map((_, i) => i), hist = [];
for (let epoch = 1; epoch <= cfg.epochs; epoch++) {
  const te = Date.now(); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  let loss = 0, steps = 0; for (let s = 0; s < order.length; s += cfg.batch) { loss += trainStep(order.slice(s, s + cfg.batch).map(i => train[i])); steps++; } loss /= steps;
  const { ev } = evaluate(test), tr = evaluate(train).ev, o = ev.outputs, rows = patterns.map(pat => { const idx = test.map((s, i) => (s.pattern === pat.key ? i : -1)).filter(i => i >= 0); return { key: pat.key, cis: idx.filter(i => o[0].probs[i] >= 0.5).length / idx.length, inv: idx.filter(i => o[1].probs[i] >= 0.5).length / idx.length }; });
  const rec = { epoch, loss: +loss.toFixed(4), trainCis: +tr.outputs[0].accuracy.toFixed(3), trainInv: +tr.outputs[1].accuracy.toFixed(3), testCis: +o[0].accuracy.toFixed(3), testInv: +o[1].accuracy.toFixed(3), massInv: o[1].culpritMass == null ? null : +o[1].culpritMass.toFixed(3), massCis: o[0].culpritMass == null ? null : +o[0].culpritMass.toFixed(3), rows, seconds: +((Date.now() - te) / 1000).toFixed(0) };
  hist.push(rec);
  console.log(`epoch ${String(epoch).padStart(2)} · loss ${rec.loss.toFixed(3)} · train CIS ${pc(rec.trainCis)} inv ${pc(rec.trainInv)} · test CIS ${pc(rec.testCis)} inv ${pc(rec.testInv)} · inv head on atypical-below cells ${rec.massInv == null ? '–' : pc(rec.massInv)} · ${rows.map(r => `${r.key} ${pc(r.inv)}`).join(' ')} invasive · ${rec.seconds} s`);
}
const last = hist[hist.length - 1];
console.log(`\nmimic table at the end (test fields, ${test.length / patterns.length} per pattern):`);
console.log(`  ${'pattern'.padEnd(10)} called CIS   called invasive`);
for (const r of last.rows) console.log(`  ${r.key.padEnd(10)} ${pc(r.cis).padStart(10)}   ${pc(r.inv).padStart(15)}`);
if (cfg.out) fs.writeFileSync(cfg.out, JSON.stringify({ cfg, hist, encoder: enc.toJSON(), attention: att.toJSON ? att.toJSON() : null }), 'utf8');
