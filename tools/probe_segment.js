#!/usr/bin/env node
/*
 * probe_segment.js — what the shipped segmentation network (data/fields/seg_weights.js) reads when it calls a pixel a
 * nest. The lesson on the page is that a nucleus in a nest is drawn like a nucleus in the epithelium, so the class has
 * to come from the window around the pixel; this measures what is in that window and what the network does with it:
 *   1 the geometry: the epithelium's thickness and depth, the nests' sizes, every nest pixel's depth below the
 *     membrane (the band is in the window only when that is within 17 to 20 px), its distance to the lumen and
 *     sideways to the nest's edge, how many lie near the field's edge (where the zero padding leaks position), and
 *     whether the y coordinate alone could call them;
 *   2 the shipped network's nest recall on the test fields by depth below the membrane;
 *   3 the band and the lumen painted over with the field's own stroma rows: the recall of the pixels that could see it;
 *   4 transplants: the centre of a big nest pasted into the band, and a patch of the band pasted into plain stroma.
 *   node tools/probe_segment.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const FL = require('../js/fields.js');
const NN = require('../js/nn.js');
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const DATA = path.join(__dirname, '..', 'data', 'fields');
const D = loadJs(path.join(DATA, 'fields_data.js')).LECTURE_FIELDS, W = D.meta.w, H = D.meta.h, cls = FL.SEG_CLASSES;
const net = NN.SegNet.fromJSON(loadJs(path.join(DATA, 'seg_weights.js')).SEG_NET);
const prep = f => { const raw = FL.decodePNGNode(f.png).px, px = FL.withGrain(raw, f.grainSeed, D.meta.grain), x = new Float64Array(px.length); for (let i = 0; i < px.length; i++) x[i] = 1 - px[i] / 255; return { f, x, y: FL.masksOf(f, W, H), m: Array.from({ length: W }, (_, xx) => FL.membraneAt(f, xx + 0.5)) }; };
const all = [].concat(D.train, D.test).map(prep), tests = all.filter(F => F.f.split === 'test' && F.y.some(v => v === 3));
const mx = a => a.reduce((p, v) => (v > p ? v : p), -Infinity), mn = a => a.reduce((p, v) => (v < p ? v : p), Infinity);
const pctl = (a, p) => { const s = Float64Array.from(a).sort(); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const pc = v => `${Math.round(v * 100)}%`, pc1 = v => `${(100 * v).toFixed(1)}%`;
const NEST = 3, EPI = 1, STROMA = 2, REACH = 20; // the window reaches 17 to 20 px up from a pixel with two levels of context
// the connected nest components of a field: pixel count and bounding box
const comps = F => { const seen = new Uint8Array(W * H), out = []; for (let i = 0; i < W * H; i++) { if (F.y[i] !== NEST || seen[i]) continue; const stack = [i]; seen[i] = 1; let n = 0, x0 = W, x1 = 0, y0 = H, y1 = 0; while (stack.length) { const j = stack.pop(); n++; const jx = j % W, jy = (j - jx) / W; x0 = Math.min(x0, jx); x1 = Math.max(x1, jx); y0 = Math.min(y0, jy); y1 = Math.max(y1, jy); for (const k of [j - 1, j + 1, j - W, j + W]) { if (k < 0 || k >= W * H || Math.abs((k % W) - jx) > 1) continue; if (F.y[k] === NEST && !seen[k]) { seen[k] = 1; stack.push(k); } } } out.push({ n, x0, x1, y0, y1 }); } return out; };

// 1 the geometry
const surf = all.map(F => F.f.surface), mdep = all.map(F => F.m.reduce((a, b) => a + b, 0) / W);
console.log(`fields ${all.length} of ${W} × ${H} px · epithelium thickness min ${mn(surf).toFixed(0)}, median ${pctl(surf, 0.5).toFixed(0)}, max ${mx(surf).toFixed(0)} px · membrane depth (mean per field) min ${mn(mdep).toFixed(0)}, median ${pctl(mdep, 0.5).toFixed(0)}, max ${mx(mdep).toFixed(0)} px`);
const cc = [], depth = [], toLumen = [], side = []; let nestPx = 0, nearEdge = 0, aboveEpi = 0, withNest = 0;
const yEpiMax = mx(all.map(F => mx(F.m)));
for (const F of all) { const c = comps(F); if (c.length) withNest++; cc.push(...c);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = y * W + x; if (F.y[i] !== NEST) continue; nestPx++;
    const d = y + 0.5 - F.m[x]; depth.push(d); toLumen.push(d + F.f.surface); if (y + 0.5 < yEpiMax) aboveEpi++;
    let l = 0; while (x - l - 1 >= 0 && F.y[i - l - 1] === NEST) l++; let r = 0; while (x + r + 1 < W && F.y[i + r + 1] === NEST) r++; side.push(Math.min(l, r) + 1);
    if (x < REACH || x >= W - REACH || y >= H - REACH) nearEdge++; } }
const within = (a, k) => pc(a.filter(d => d <= k).length / a.length);
console.log(`nests: ${cc.length} on ${withNest} fields · width median ${pctl(cc.map(c => c.x1 - c.x0 + 1), 0.5)} px (10th to 90th percentile ${pctl(cc.map(c => c.x1 - c.x0 + 1), 0.1)} to ${pctl(cc.map(c => c.x1 - c.x0 + 1), 0.9)}), height median ${pctl(cc.map(c => c.y1 - c.y0 + 1), 0.5)} px (${pctl(cc.map(c => c.y1 - c.y0 + 1), 0.1)} to ${pctl(cc.map(c => c.y1 - c.y0 + 1), 0.9)})`);
console.log(`nest pixels ${nestPx} · depth below the membrane: median ${pctl(depth, 0.5).toFixed(0)}, 90th percentile ${pctl(depth, 0.9).toFixed(0)}, max ${mx(depth).toFixed(0)} px · within 2 px ${within(depth, 2)}, 7 px ${within(depth, 7)}, 17 px ${within(depth, 17)}, 20 px ${within(depth, 20)} (the band in the window) · the lumen within 20 px ${within(toLumen, 20)}`);
console.log(`  sideways to the nest's edge: median ${pctl(side, 0.5)} px, within 17 px ${within(side, 17)} · within ${REACH} px of the left, right or bottom edge ${pc(nearEdge / nestPx)} · the epithelium reaches y ${yEpiMax.toFixed(0)} at most over the fields, and ${pc(1 - aboveEpi / nestPx)} of nest pixels lie deeper (y alone would call them)`);

// 2 nest recall by depth, on the test fields with a nest
const bands = [['within 20 px', d => d <= 20], ['20 to 40 px', d => d > 20 && d <= 40], ['beyond 40 px', d => d > 40]], tally = bands.map(() => [0, 0]);
for (const F of tests) { const p = net.predict(F.x, H, W).labels; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = y * W + x; if (F.y[i] !== NEST) continue; const d = y + 0.5 - F.m[x]; bands.forEach(([, f], b) => { if (f(d)) { tally[b][0]++; if (p[i] === NEST) tally[b][1]++; } }); } }
console.log(`shipped network, ${tests.length} test fields with a nest · nest recall by depth below the membrane: ${bands.map(([name], b) => `${name} ${pc1(tally[b][1] / tally[b][0])} of ${tally[b][0]} px`).join(' · ')}`);

// 3 the band and the lumen painted over with the field's own stroma rows (tiled from below the deepest membrane)
const paintOver = F => { const x = Float64Array.from(F.x), srcLo = Math.min(H - 8, Math.ceil(mx(F.m)) + 2); for (let xx = 0; xx < W; xx++) { const bottom = Math.floor(F.m[xx]); for (let y = 0; y < bottom; y++) x[y * W + xx] = F.x[(srcLo + (y % (H - srcLo))) * W + xx]; } return x; };
let seeBand = [0, 0, 0];
for (const F of tests) { const p0 = net.predict(F.x, H, W).labels, p1 = net.predict(paintOver(F), H, W).labels; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = y * W + x; if (F.y[i] !== NEST || y + 0.5 - F.m[x] > REACH) continue; seeBand[0]++; if (p0[i] === NEST) seeBand[1]++; if (p1[i] === NEST) seeBand[2]++; } }
console.log(`band and lumen painted over with stroma: nest recall on the ${seeBand[0]} pixels within ${REACH} px of the membrane ${pc1(seeBand[1] / seeBand[0])} as scanned → ${pc1(seeBand[2] / seeBand[0])} painted over`);

// 4 transplants: a P × P patch, the calls read on its core (a margin of `core` left all round)
const P = 48, core = 12, A = { n: 0, as: [0, 0, 0, 0], src: [0, 0, 0, 0], fields: 0 }, B = { n: 0, as: [0, 0, 0, 0], src: [0, 0, 0, 0], fields: 0 };
const paste = (F, sx0, sy0, tx0, ty0) => { const x = Float64Array.from(F.x); for (let dy = 0; dy < P; dy++) for (let dx = 0; dx < P; dx++) x[(ty0 + dy) * W + tx0 + dx] = F.x[(sy0 + dy) * W + sx0 + dx]; return x; };
const readCore = (T, F, p0, p1, sx0, sy0, tx0, ty0, want) => { T.fields++; for (let dy = core; dy < P - core; dy++) for (let dx = core; dx < P - core; dx++) { const si = (sy0 + dy) * W + sx0 + dx; if (F.y[si] !== want) continue; T.n++; T.src[p0[si]]++; T.as[p1[(ty0 + dy) * W + tx0 + dx]]++; } };
for (const F of tests) {
  const bandDeep = tx0 => { for (let xx = tx0; xx < tx0 + P; xx++) if (F.m[xx] < P + 1.5) return false; return true; }, p0 = net.predict(F.x, H, W).labels;
  const big = comps(F).filter(c => c.x1 - c.x0 + 1 >= P + 4 && c.y1 - c.y0 + 1 >= P + 4);
  if (big.length && bandDeep(64)) { const c = big[0], sx0 = Math.round((c.x0 + c.x1) / 2 - P / 2), sy0 = Math.round((c.y0 + c.y1) / 2 - P / 2); if (sx0 >= 0 && sy0 >= 0 && sx0 + P <= W && sy0 + P <= H) readCore(A, F, p0, net.predict(paste(F, sx0, sy0, 64, 1), H, W).labels, sx0, sy0, 64, 1, NEST); }
  { const sx0 = 10, sy0 = 1, tx0 = W - P - 4, ty0 = H - P - 2; let ok = bandDeep(sx0); for (let dy = 0; dy < P && ok; dy++) for (let dx = 0; dx < P; dx++) if (F.y[(ty0 + dy) * W + tx0 + dx] !== STROMA) { ok = false; break; } if (ok) readCore(B, F, p0, net.predict(paste(F, sx0, sy0, tx0, ty0), H, W).labels, sx0, sy0, tx0, ty0, EPI); }
}
const calls = (T, k) => `${cls.map((c, j) => `${c} ${pc(k[j] / T.n)}`).join(', ')}`;
console.log(`transplant: the centre of a big nest (${P} × ${P} px) pasted into the epithelial band, ${A.fields} fields, ${A.n} core nest pixels · in place ${calls(A, A.src)} · inside the band ${calls(A, A.as)}`);
console.log(`transplant: a ${P} × ${P} px patch of the band pasted into plain stroma, ${B.fields} fields, ${B.n} core band pixels · in place ${calls(B, B.src)} · in the stroma ${calls(B, B.as)}`);
