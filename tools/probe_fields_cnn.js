#!/usr/bin/env node
/*
 * probe_fields_cnn.js — a model saved by check_fields_cnn.js (--out) re-evaluated from its weights: accuracy, the mimic
 * table, how spread the heads' attention is (the effective number of cells, 1 / Σ a²), the heads' shares on the cells
 * holding atypical nuclei, and the invasion head's map over the cells of the first invasive and the first CIS-into-nests
 * test field.   node tools/probe_fields_cnn.js run.json [more.json ...]
 */
'use strict';
const fs = require('fs'), path = require('path'); const ROOT = path.join(__dirname, '..');
const NN = require(path.join(ROOT, 'js', 'nn.js')), FL = require(path.join(ROOT, 'js', 'fields.js'));
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(ROOT, 'data', 'fields', 'fields_data.js')).LECTURE_FIELDS, W0 = D.meta.w, H0 = D.meta.h, W = W0 >> 1, H = H0 >> 1, UNIT = 16; let GH = H >> 3, GW = W >> 3;
const cisOf = f => (['cis', 'cisvbn', 'inv'].includes(f.pattern) ? 1 : 0), pc = v => `${Math.round(v * 100)}%`;
const conv3 = NN.SegNet.prototype.conv3, pool2 = NN.SegNet.prototype.pool2;
const grid = (gh, gw) => { GH = gh; GW = gw; const xy = [], cx = []; for (let i = 0; i < GH; i++) for (let j = 0; j < GW; j++) { xy.push([(j + 0.5) / GW * 2 - 1, (i + 0.5) / GH * 2 - 1]); cx.push([(j + 0.5) * W0 / GW, (i + 0.5) * H0 / GH]); } return { xy, dist: cx.map(a => Float64Array.from(cx, b => Math.hypot(a[0] - b[0], a[1] - b[1]) / UNIT)) }; };
const prep = f => { const raw = FL.decodePNGNode(f.png).px, px = FL.withGrain(raw, f.grainSeed, D.meta.grain), x = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) { let s = 0; for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) s += 1 - px[(2 * y + dy) * W0 + 2 * xx + dx] / 255; x[y * W + xx] = s / 4; }
  const cellOf = n => Math.min(GH - 1, Math.floor(n.y / H0 * GH)) * GW + Math.min(GW - 1, Math.floor(n.x / W0 * GW)), posA = new Array(GH * GW).fill(false), posB = new Array(GH * GW).fill(false), nestCell = new Array(GH * GW).fill(false);
  for (const n of f.nuclei.map(arr => Object.fromEntries(D.meta.nucleus.map((k, i) => [k, arr[i]])))) { if (n.atypical) posA[cellOf(n)] = true; if (n.atypical && n.below) posB[cellOf(n)] = true; if (n.below) nestCell[cellOf(n)] = true; }
  return { name: f.name, split: f.split, pattern: f.pattern, x, y: [cisOf(f), f.label], pos: [posA, posB], nestCell }; };
let train, test, builtFor = -1; const build = levels => { if (builtFor === levels) return; grid(H >> levels, W >> levels); train = D.train.map(prep); test = D.test.map(prep); let m = 0; for (const f of train) for (const v of f.x) m += v; m /= train.length * W * H; for (const f of train.concat(test)) for (let i = 0; i < f.x.length; i++) f.x[i] -= m; builtFor = levels; };
const encForward = (enc, x) => { const C = enc.channels; let cur = x, h = H, w = W; for (let l = 0; l <= enc.levels; l++) { const c1 = conv3(cur, enc.enc[l][0], h, w, true), c2 = conv3(c1.act, enc.enc[l][1], h, w, true); if (l < enc.levels) { const p = pool2(c2.act, C, h, w); cur = p.v; h = p.h; w = p.w; } else cur = c2.act; } const n = h * w, raw = Array.from({ length: n }, (_, i) => Float64Array.from({ length: C }, (_, c) => cur[c * n + i])); if (!enc.norm) return raw; return raw.map(v => { let mu = 0; for (const q of v) mu += q; mu /= C; let va = 0; for (const q of v) va += (q - mu) * (q - mu); const s = Math.sqrt(va / C + 1e-6); return Float64Array.from(v, q => (q - mu) / s); }); }; // standardised tokens when the model was trained with them
for (const file of process.argv.slice(2)) {
  const run = path.basename(file, '.json'); if (!fs.existsSync(file)) { console.log(`${file}: not found`); continue; }
  const R = JSON.parse(fs.readFileSync(file, 'utf8')), enc = R.encoder; for (const pair of enc.enc) for (const c of pair) { c.W = Float64Array.from(c.W); c.b = Float64Array.from(c.b); }
  const att = NN.AttentionMIL.fromJSON(R.attention), pos = R.cfg.pos; build(enc.levels); const G = grid(H >> enc.levels, W >> enc.levels);
  const slidesOf = set => set.map(f => ({ H: encForward(enc, f.x).map((t, i) => (pos ? Float64Array.from([...t, G.xy[i][0], G.xy[i][1]]) : t)), dist: G.dist, xy: G.xy, y: f.y, pos: f.pos, pattern: f.pattern, name: f.name, nestCell: f.nestCell }));
  const te = slidesOf(test), tr = slidesOf(train), evT = att.evaluate(te), evR = att.evaluate(tr), o = evT.outputs;
  console.log(`run ${run}: ${R.cfg.channels} channels, context ${R.cfg.context ? 'on' : 'off'}, seed ${R.cfg.seed}, ${R.hist.length} epochs · training CIS ${pc(evR.outputs[0].accuracy)} inv ${pc(evR.outputs[1].accuracy)} · test CIS ${pc(o[0].accuracy)} inv ${pc(o[1].accuracy)} · CIS head on atypical cells ${pc(o[0].culpritMass)} · invasion head on atypical-below cells ${pc(o[1].culpritMass)}`);
  const eff = k => o[k].weights.reduce((s, a) => s + 1 / a.reduce((q, v) => q + v * v, 0), 0) / o[k].weights.length; console.log(`  attention spread, effective number of cells of ${GW * GH}: CIS head ${eff(0).toFixed(1)} · invasion head ${eff(1).toFixed(1)}`);
  const patterns = D.meta.patterns; console.log(`  ${'pattern'.padEnd(18)} called CIS   called invasive`);
  for (const pat of patterns) { const idx = te.map((s, i) => (s.pattern === pat.key ? i : -1)).filter(i => i >= 0); console.log(`  ${pat.short.padEnd(18)} ${pc(idx.filter(i => o[0].probs[i] >= 0.5).length / idx.length).padStart(10)}   ${pc(idx.filter(i => o[1].probs[i] >= 0.5).length / idx.length).padStart(15)}`); }
  // the best epoch on the curve, and the invasion head's map over the cells of the first invasive and the first mimic test field
  const best = R.hist.reduce((a, h) => (h.testInv > a.testInv ? h : a), R.hist[0]); console.log(`  best test invasion accuracy on the curve: ${pc(best.testInv)} at epoch ${best.epoch}; last ${pc(R.hist[R.hist.length - 1].testInv)}`);
  for (const key of ['inv', 'cisvbn']) { const i = te.findIndex(s => s.pattern === key), s = te[i], a = o[1].weights[i]; const amax = Math.max(...a); let rows = [];
    for (let r = 0; r < GH; r++) { let line = ''; for (let c = 0; c < GW; c++) { const k = r * GW + c, v = a[k] / amax; line += (s.pos[1][k] ? '[' : ' ') + (v > 0.66 ? '█' : v > 0.33 ? '▓' : v > 0.1 ? '░' : '·') + (s.pos[1][k] ? ']' : ' '); } rows.push('    ' + line); }
    console.log(`  invasion head on ${s.name} (${key}, called ${o[1].probs[i] >= 0.5 ? 'invasive' : 'not invasive'}, P ${o[1].probs[i].toFixed(2)}; [ ] marks cells with an atypical nucleus below the membrane):\n${rows.join('\n')}`); }
}
