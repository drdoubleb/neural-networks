#!/usr/bin/env node
/*
 * generate_fields.js — draws the tissue fields of the invasion question: a strip of bladder, urothelium on top of a
 * wavy basement membrane, stroma with spindle cells beneath, and nests of urothelial cells below the membrane. Five
 * patterns, one label (invasive or not), so that invasion is the conjunction of three cues and every incomplete
 * combination has a real mimic:
 *   vbn     normal urothelium with von Brunn nests   bland cells, below the membrane, in round smooth nests       not invasive
 *   ip      inverted papilloma                       bland cells, below the membrane, in anastomosing cords       not invasive
 *   cis     carcinoma in situ                        atypical cells, confined above the membrane                  not invasive
 *   cisvbn  CIS extending into von Brunn nests       atypical cells, below the membrane, in round smooth nests    not invasive
 *   inv     invasive carcinoma                       atypical cells, below the membrane, in angulated nests that mostly
 *                                                    grow down from the epithelium; on some fields single cells shed
 *                                                    into the stroma                                              INVASIVE
 * The nuclei come from the atypia generator (tools/generate_nuclei.js), so their cytology is what the encoder learned
 * on; every nucleus of a field is a token for the model, the stromal spindle cells included, and the field keeps the
 * truth about each (atypical or bland, above or below the membrane, which nest) for the page to reveal afterwards.
 *   node tools/generate_fields.js
 * writes data/fields/fields_data.js (every field as a PNG without its pixel grain, which the loader adds back from the
 * field's seed, in steps of four grey levels that the grain hides, its nuclear segmentation as a second PNG, plus the
 * membrane, the nuclei and the nests) and two contact sheets in H&E colour, the fields as they are and with the truth
 * drawn over them.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const gen = require('./generate_nuclei.js');
const FL = require('../js/fields.js');

const SEED = 20260930, W = 176, H = 128, SIZE = gen.SIZE, PER_PATTERN_TRAIN = 40, PER_PATTERN_TEST = 10, SHEET_PER_PATTERN = 4;
const OUT = path.join(__dirname, '..', 'data', 'fields');
const task = gen.TASKS.find(t => t.id === 'atypia');
const PATTERNS = [
  { key: 'vbn', name: 'Normal urothelium with von Brunn nests', short: 'von Brunn nests', label: 0, atypical: false, below: 'round',
    cues: 'bland cells · below the membrane · round smooth nests', blurb: 'Bland urothelium, and bland cells below the basement membrane in round, smooth nests: von Brunn nests, a normal finding in the bladder. Cells below the membrane are not invasion.' },
  { key: 'ip', name: 'Inverted papilloma', short: 'Inverted papilloma', label: 0, atypical: false, below: 'cords',
    cues: 'bland cells · below the membrane · anastomosing cords', blurb: 'Bland urothelium, and bland cells growing down below the membrane in irregular, anastomosing cords. The architecture is complex, but the cells are bland: not invasion.' },
  { key: 'cis', name: 'Carcinoma in situ', short: 'CIS', label: 0, atypical: true, below: 'round-sometimes',
    cues: 'atypical cells · above the membrane only', blurb: 'Atypical cells through the whole thickness of the urothelium, confined above the basement membrane; on some fields normal von Brunn nests lie below it. Atypia alone is not invasion.' },
  { key: 'cisvbn', name: 'Carcinoma in situ extending into von Brunn nests', short: 'CIS into nests', label: 0, atypical: true, below: 'round',
    cues: 'atypical cells · below the membrane · round smooth nests', blurb: 'Atypical cells above the membrane and below it, but the cells below fill von Brunn nests that keep their round, smooth outline. Atypical cells below the membrane are still not invasion when the architecture is that of a pre-existing nest.' },
  { key: 'inv', name: 'Invasive carcinoma', short: 'Invasive', label: 1, atypical: true, below: 'jagged',
    cues: 'atypical cells · below the membrane · angulated nests growing down from the epithelium, sometimes single cells', blurb: 'Atypical cells above the membrane and below it, in angulated nests, tongues and branches that mostly grow down from the epithelium; on some fields single cells are shed into the stroma. Cytology, location and architecture together: the only invasive pattern.' },
];

// ---- random numbers: one stream for the fields, the nucleus generator's own for the nuclei
let rand = gen.mulberry32(SEED);
const U = (lo, hi) => lo + (hi - lo) * rand(), RI = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1)), TAU = 2 * Math.PI;
gen.setSeed(SEED + 1);
let atypCounter = 0, blandCounter = 0;
// a nucleus of the atypia generator: its shape (ellipse axes, contour harmonics and bumps), its look and a chromatin
// texture; atypical nuclei cycle through the generator's single-trait and combined kinds
function sampleNucleus(label) {
  const n = gen.makeOne(label, task, label ? (atypCounter++ % 60) : blandCounter++), tex = gen.makeValueNoise(n.look.textureCells || 7);
  return { a: n.shape.a, b: n.shape.b, phi: n.shape.phi, harmonics: n.shape.harmonics, bumps: n.shape.bumps, value: n.look.nucleus, textureAmp: n.look.textureAmp, rim: n.look.rim, edgeSoftness: n.look.edgeSoftness, tex, label, subtype: n.traits.subtype };
}
// a stromal spindle cell: thin, pale, lying along the surface
function sampleSpindle() {
  const tex = gen.makeValueNoise(7);
  return { a: U(5.5, 7.5), b: U(1.7, 2.4), phi: U(-0.55, 0.55) + (rand() < 0.5 ? 0 : Math.PI), harmonics: [{ k: 2, amp: U(0, 0.02), phase: U(0, TAU) }], bumps: [], value: U(0.46, 0.56), textureAmp: 0.04, rim: 0.03, edgeSoftness: 0.45, tex, label: 0, subtype: 'stroma' };
}
// periodic value noise: a cx × cy lattice of random values, smoothly interpolated, wrapping around so that the whole
// field is covered (the nucleus generator's lattice covers only one crop)
function makeNoise(cx, cy) {
  const g = new Float64Array(cx * cy); for (let i = 0; i < g.length; i++) g[i] = rand() * 2 - 1;
  return (u, v) => { const x = ((u % 1) + 1) % 1 * cx, y = ((v % 1) + 1) % 1 * cy, x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy), at = (i, j) => g[(j % cy) * cx + (i % cx)];
    return (at(x0, y0) * (1 - sx) + at(x0 + 1, y0) * sx) * (1 - sy) + (at(x0, y0 + 1) * (1 - sx) + at(x0 + 1, y0 + 1) * sx) * sy; };
}
function hull(points) { // the convex hull, counter-clockwise (Andrew's monotone chain)
  const P = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  if (P.length < 3) return P;
  const lower = []; for (const q of P) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  const upper = []; for (const q of P.slice().reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
function contourModulation(p, theta) { // as tools/generate_nuclei.js draws it
  let m = 0;
  for (const h of p.harmonics) m += h.amp * Math.cos(h.k * theta + h.phase);
  for (const b of p.bumps) { let d = theta - b.theta0; d = Math.atan2(Math.sin(d), Math.cos(d)); m += b.depth * Math.exp(-(d * d) / (2 * b.width * b.width)); }
  return m;
}

// ---- geometry helpers
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function segDist(p, a, b) { const vx = b[0] - a[0], vy = b[1] - a[1], L2 = vx * vx + vy * vy, t = L2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2)) : 0; return Math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy); }
function polyDist(p, poly) { let d = Infinity; for (let i = 0; i < poly.length - 1; i++) d = Math.min(d, segDist(p, poly[i], poly[i + 1])); return d; }
function inPolygon(p, poly) { let inside = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) inside = !inside; } return inside; }
// signed distance to a nest outline: negative inside. round: an ellipse with a faint harmonic; jagged: a polygon;
// cords: a band of half-width w/2 around a polyline
function nestSigned(nest, p) {
  if (nest.kind === 'cords') return Math.min(...nest.paths.map(path => polyDist(p, path))) - nest.halfWidth;
  if (nest.kind === 'round') {
    const dx = p[0] - nest.cx, dy = p[1] - nest.cy, c = Math.cos(nest.rot), s = Math.sin(nest.rot), xr = dx * c + dy * s, yr = -dx * s + dy * c, ex = xr / nest.rx, ey = yr / nest.ry;
    const rho = Math.hypot(ex, ey), theta = Math.atan2(ey, ex), boundary = 1 + nest.amp * Math.cos(nest.k * theta + nest.phase);
    return (rho - boundary) * Math.min(nest.rx, nest.ry);
  }
  let best = Infinity; for (const poly of nest.outlines) { const d = polyDist(p, poly), sd = inPolygon(p, poly) ? -d : d; if (sd < best) best = sd; } return best; // jagged: the union of its closed ribbons
}
const membraneOf = f => x => f.ym0 + f.A1 * Math.sin(TAU * x / f.L1 + f.p1) + f.A2 * Math.sin(TAU * x / f.L2 + f.p2);

// ---- one field: the membrane, the epithelium, what lies below, the stroma
function makeField(P, id) {
  const f = { id, pattern: P.key, label: P.label, ym0: U(46, 58), A1: U(1.5, 3.5), L1: U(90, 170), p1: U(0, TAU), A2: U(0.5, 1.5), L2: U(28, 55), p2: U(0, TAU), nuclei: [], nests: [] };
  const ym = membraneOf(f), rows = rand() < 0.5 ? 2 : 3, pitch = U(15, 17), spacing = U(17, 20);
  f.rows = rows; f.thickness = 9 + (rows - 1) * pitch + 12; // the surface is this far above the membrane
  const ys = x => ym(x) - f.thickness;
  // the urothelium: rows of nuclei on the membrane, staggered, bland or atypical through the whole thickness
  for (let r = 0; r < rows; r++) {
    for (let x = U(3, 9) + (r % 2) * spacing / 2; x < W - 3; x += spacing + U(-1.5, 1.5)) {
      const y = ym(x) - 9 - r * pitch + U(-1.2, 1.2);
      f.nuclei.push(Object.assign(sampleNucleus(P.atypical ? 1 : 0), { x, y, kind: 'epi', below: false, group: null }));
    }
  }
  const taken = () => f.nuclei.map(n => [n.x, n.y]);
  const clear = (p, min) => taken().every(q => dist(p, q) >= min);
  const belowMembrane = (x, margin) => ym(x) + margin;
  // nests below the membrane, placed where they do not overlap
  const nestFits = (cx, cy, R) => cy - R >= belowMembrane(cx, 5) && cy + R <= H - 3 && cx - R >= 3 && cx + R <= W - 3 && f.nests.every(n => dist([cx, cy], [n.cx, n.cy]) >= R + n.R + 8);
  function placeNest(R) { for (let t = 0; t < 40; t++) { const cx = U(R + 4, W - R - 4), cy = U(belowMembrane(cx, 5) + R, H - 3 - R); if (cy - R >= belowMembrane(cx, 5) && nestFits(cx, cy, R)) return [cx, cy]; } return null; }
  const roundNest = (atypical, gid) => {
    const R = U(19, 25), at = placeNest(R); if (!at) return false;
    const nest = { kind: 'round', cx: at[0], cy: at[1], R, rx: R, ry: R * U(0.85, 1), rot: U(0, Math.PI), amp: U(0, 0.025), k: RI(2, 3), phase: U(0, TAU), group: gid };
    f.nests.push(nest);
    const r = R - 8.5, k = Math.min(8, Math.floor(TAU * r / 13.5)), a0 = U(0, TAU); // nuclei on a ring, and one in the middle of a big nest
    for (let i = 0; i < k; i++) { const ang = a0 + i * TAU / k + U(-0.12, 0.12), rr = r + U(-1, 1); f.nuclei.push(Object.assign(sampleNucleus(atypical ? 1 : 0), { x: nest.cx + rr * Math.cos(ang), y: nest.cy + rr * Math.sin(ang) * nest.ry / nest.rx, kind: 'nest', below: true, group: gid })); }
    if (r >= 13) f.nuclei.push(Object.assign(sampleNucleus(atypical ? 1 : 0), { x: nest.cx + U(-1.5, 1.5), y: nest.cy + U(-1.5, 1.5), kind: 'nest', below: true, group: gid }));
    return true;
  };
  // invasive: a chain of atypical nuclei that grows down from the underside of the epithelium (or, less often, lies free
  // in the stroma), turning as it goes, with a side branch and stretches two cells wide; its outline is a ribbon hugging
  // the nuclei, kinked where the chain turns and pointed at the tip, so the angulation is the nuclei themselves
  // pressing on the border
  const radiusOf = nn => (nn.a + nn.b) / 2;
  const freeAt = (q, own) => q[0] >= 10 && q[0] <= W - 10 && q[1] <= H - 8 && q[1] >= ym(q[0]) + 6 && clear(q, 11) && own.every(o => dist(q, [o.x, o.y]) >= 11) && f.nests.every(o => nestSigned(o, q) < -1e9 || nestSigned(o, q) >= 7);
  const growChain = (start, dir, count, own) => { // nuclei touching one another along a turning path; blocked, it tries other turns
    const nodes = [start];
    while (nodes.length < count) {
      const prev = nodes[nodes.length - 1], nn = sampleNucleus(1), r = radiusOf(nn); let placed = false;
      for (let t = 0; t < 6 && !placed; t++) {
        const d = Math.max(-1.4, Math.min(1.4, dir + U(-0.6, 0.6))), step = prev.r + r - U(0.5, 2), q = [prev.x + step * Math.sin(d), prev.y + step * Math.cos(d)];
        if (!freeAt(q, own.concat(nodes.slice(0, -1)))) continue;
        nodes.push(Object.assign(nn, { x: q[0], y: q[1], r })); dir = d; placed = true;
      }
      if (!placed) break;
    }
    return nodes;
  };
  const tangent = (nodes, i) => { const a = nodes[Math.max(0, i - 1)], b = nodes[Math.min(nodes.length - 1, i + 1)], d = dist([a.x, a.y], [b.x, b.y]) || 1; return [(b.x - a.x) / d, (b.y - a.y) / d]; };
  const ribbon = (nodes, headLen) => { // an offset point either side of every nucleus, wider beside a twin, a point at the tip
    const Lp = [], Rp = [];
    for (let i = 0; i < nodes.length; i++) {
      const p = nodes[i], [tx, ty] = tangent(nodes, i), nx = -ty, ny = tx, w = p.twin ? p.twin.r * 2 + 1 : 0, offL = p.r + U(0.8, 2.2) + (p.twin && p.twin.side === 1 ? w : 0), offR = p.r + U(0.8, 2.2) + (p.twin && p.twin.side === -1 ? w : 0);
      Lp.push([p.x + nx * offL, p.y + ny * offL]); Rp.push([p.x - nx * offR, p.y - ny * offR]);
    }
    const [t0x, t0y] = tangent(nodes, 0), [t1x, t1y] = tangent(nodes, nodes.length - 1), first = nodes[0], last = nodes[nodes.length - 1];
    const head = [first.x - t0x * (first.r + headLen), first.y - t0y * (first.r + headLen)], tip = [last.x + t1x * (last.r + 4), last.y + t1y * (last.r + 4)];
    return [head, ...Lp, tip, ...Rp.reverse(), head];
  };
  const invasiveNest = (gid, connected, count, branch) => {
    for (let attempt = 0; attempt < 8; attempt++) {
      let root = null, dir;
      if (connected) { const n0 = sampleNucleus(1), r0 = radiusOf(n0), x = U(22, W - 22), y = ym(x) + r0 + 1; if (freeAt([x, y], [])) { root = Object.assign(n0, { x, y, r: r0 }); dir = U(-0.8, 0.8); } }
      else { const n0 = sampleNucleus(1), r0 = radiusOf(n0), x = U(16, W - 16), y = U(0, H); if (y > ym(x) + 22 && y < H - 30 && freeAt([x, y], [])) { root = Object.assign(n0, { x, y, r: r0 }); dir = U(-1.2, 1.2); } }
      if (!root) continue;
      const main = growChain(root, dir, count, []);
      if (main.length < 4) continue;
      const chains = [main];
      if (branch && main.length >= 5) { const k = RI(1, main.length - 3), [tx, ty] = tangent(main, k), base = Math.atan2(tx, ty), side = rand() < 0.5 ? -1 : 1, b = growChain(main[k], base + side * U(0.9, 1.4), 1 + RI(2, 4), main); if (b.length >= 3) { chains.push(b); main[k].branch = true; } }
      const all = [].concat(...chains.map((c, i) => (i ? c.slice(1) : c))), twins = [];
      for (const c of chains) for (let i = 1; i < c.length - 1; i++) { // a second nucleus beside some links, two cells wide there
        const p = c[i]; if (p.branch || p.twin || rand() >= 0.45) continue;
        const [tx, ty] = tangent(c, i), side = rand() < 0.5 ? -1 : 1, nn = sampleNucleus(1), r = radiusOf(nn), off = (p.r + r - 1) * side, q = [p.x - ty * off, p.y + tx * off];
        if (!freeAt(q, all.filter(o => o !== p).concat(twins))) continue;
        twins.push(Object.assign(nn, { x: q[0], y: q[1], r })); p.twin = { side, r };
      }
      const outlines = chains.map((c, i) => ribbon(c, i ? 0 : connected ? 6 : 2)), every = all.concat(twins);
      f.nests.push({ kind: 'jagged', cx: every.reduce((a, o) => a + o.x, 0) / every.length, cy: every.reduce((a, o) => a + o.y, 0) / every.length, R: Math.max(...[].concat(...outlines).map(q => dist(q, [root.x, root.y]))), outlines, connected, group: gid });
      for (const o of every) { const { r, twin, branch: br, ...rest } = o; void r; void twin; void br; f.nuclei.push(Object.assign(rest, { kind: 'nest', below: true, group: gid })); }
      return true;
    }
    return false;
  };
  const cords = () => { // inverted papilloma: cords of bland cells growing down from the membrane, some joined
    const nCords = RI(2, 3), paths = [], halfWidth = U(11, 13);
    for (let c = 0; c < nCords; c++) {
      const xs = U(24, W - 24), start = [xs, ym(xs) + 1], p = [start]; let th = U(-0.55, 0.55), x = xs, y = start[1];
      for (let s = 0; s < 6; s++) { th += U(-0.35, 0.35); th = Math.max(-1.0, Math.min(1.0, th)); const st = U(12, 16); x += st * Math.sin(th); y += st * Math.cos(th); if (y > H - 10 || x < 12 || x > W - 12) break; p.push([x, y]); }
      if (p.length >= 2) paths.push(p);
    }
    const base = paths.slice(); // bridges only between the cords themselves, never between bridges
    for (let a = 0; a < base.length; a++) for (let b = a + 1; b < base.length; b++) { // a bridge where two cords run close: anastomosis
      let best = null; for (const pa of base[a].slice(1)) for (const pb of base[b].slice(1)) { const d = dist(pa, pb); if (d >= 24 && d <= 60 && (!best || d < best.d)) best = { pa, pb, d }; }
      if (best && rand() < 0.8) paths.push([best.pa, [(best.pa[0] + best.pb[0]) / 2 + U(-6, 6), (best.pa[1] + best.pb[1]) / 2 + U(-6, 6)], best.pb]);
    }
    const nest = { kind: 'cords', cx: 0, cy: 0, R: 0, paths, halfWidth, group: 0 }; f.nests.push(nest);
    for (const p of paths) for (let i = 0; i < p.length - 1; i++) { // two staggered rows of nuclei along every cord
      const a = p[i], b = p[i + 1], L = dist(a, b), ux = (b[0] - a[0]) / L, uy = (b[1] - a[1]) / L, nx = -uy, ny = ux;
      for (let t = i === 0 ? 8 : 2; t < L - 1; t += 13) for (const side of [-1, 1]) { const off = side * (halfWidth - 6.5) + U(-1, 1), q = [a[0] + ux * (t + side * 3) + nx * off, a[1] + uy * (t + side * 3) + ny * off]; if (q[1] > ym(q[0]) + 6 && q[1] < H - 4 && q[0] > 4 && q[0] < W - 4 && clear(q, 11.5)) f.nuclei.push(Object.assign(sampleNucleus(0), { x: q[0], y: q[1], kind: 'cord', below: true, group: 0 })); }
    }
    return true;
  };
  if (P.below === 'round' || (P.below === 'round-sometimes' && rand() < 0.5)) { const n = RI(1, 2); for (let i = 0; i < n; i++) roundNest(P.key === 'cisvbn', i); }
  else if (P.below === 'cords') cords();
  else if (P.below === 'jagged') { // one nest, or two; most grow from the underside of the epithelium
    const n = RI(1, 2); let placed = 0;
    for (let i = 0; i < n; i++) if (invasiveNest(placed, i === 0 ? rand() < 0.7 : rand() < 0.4, i === 0 ? RI(6, 10) : RI(4, 6), i === 0 && rand() < 0.6)) placed++;
    if (!placed && !invasiveNest(0, true, RI(6, 10), true)) throw new Error(`field ${id}: no invasive nest could be placed`);
    if (rand() < 0.5) for (let s = 0, tries = 0; s < RI(1, 3) && tries < 60; tries++) { // single cells shed into the stroma, on half the fields
      const p = [U(12, W - 12), U(0, H)]; if (p[1] < ym(p[0]) + 16 || p[1] > H - 8 || !clear(p, 17) || f.nests.some(n => nestSigned(n, p) < 9)) continue;
      f.nests.push({ kind: 'round', cx: p[0], cy: p[1], R: 10, rx: 10, ry: 9, rot: U(0, Math.PI), amp: 0.06, k: 3, phase: U(0, TAU), group: 10 + s, single: true });
      f.nuclei.push(Object.assign(sampleNucleus(1), { x: p[0], y: p[1], kind: 'single', below: true, group: 10 + s })); s++;
    }
  }
  // the stroma: spindle cells wherever nothing else is
  for (let s = 0, tries = 0; s < RI(6, 9) && tries < 200; tries++) {
    const p = [U(6, W - 6), U(0, H - 5)]; if (p[1] < ym(p[0]) + 8 || !clear(p, 13) || f.nests.some(n => nestSigned(n, p) < 7)) continue;
    f.nuclei.push(Object.assign(sampleSpindle(), { x: p[0], y: p[1], kind: 'stroma', below: true, group: null })); s++;
  }
  if (f.nuclei.length > 80) throw new Error(`field ${id}: ${f.nuclei.length} nuclei`);
  f.nuclei.forEach(n => { n.x = Math.round(n.x * 10) / 10; n.y = Math.round(n.y * 10) / 10; });
  return f;
}

// ---- rendering: the tissue as grey levels (0 dark … 1 pale), without the pixel grain
const VAL = { lumen: 0.955, cyto: 0.905, stroma: 0.83, membrane: 0.17, outline: 0.11 }, QUANTUM = 4;
function renderField(f) {
  const ym = membraneOf(f), ys = x => ym(x) - f.thickness, img = new Float64Array(W * H);
  const fib = makeNoise(5, 16), fib2 = makeNoise(3, 3), cyt = makeNoise(10, 8), lum = makeNoise(4, 3);
  const soft = d => 0.5 - 0.5 * Math.tanh(d / 0.9); // 1 well below a boundary, 0 above, soft across ~2 px
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const mem = ym(x), sur = ys(x), u = x / W, v = y / H;
    const stroma = VAL.stroma + 0.035 * fib(u, v) + 0.02 * fib2(u, v); // fibres run along the surface
    const cyto = VAL.cyto + 0.02 * cyt(u, v), lumen = VAL.lumen + 0.01 * lum(u, v);
    const inEpi = soft(sur - y) * soft(y - mem), inLumen = soft(sur - y) * (1 - soft(y - mem)) > 0.5 ? 1 : soft(y - sur);
    let val = inEpi * cyto + (1 - inEpi) * (y < mem ? lumen : stroma);
    if (y < sur + 2) val = (1 - inLumen) * val + inLumen * lumen;
    img[y * W + x] = val;
  }
  // nests: cytoplasm inside the outline, a fine darker outline
  for (const nest of f.nests) {
    const pad = nest.kind === 'cords' ? nest.halfWidth + 3 : nest.kind === 'jagged' ? 4 : nest.R + 6, pts = nest.kind === 'cords' ? [].concat(...nest.paths) : nest.kind === 'jagged' ? [].concat(...nest.outlines) : [[nest.cx - nest.R, nest.cy - nest.R], [nest.cx + nest.R, nest.cy + nest.R]];
    const x0 = Math.max(0, Math.floor(Math.min(...pts.map(p => p[0])) - pad)), x1 = Math.min(W - 1, Math.ceil(Math.max(...pts.map(p => p[0])) + pad)), y0 = Math.max(0, Math.floor(Math.min(...pts.map(p => p[1])) - pad)), y1 = Math.min(H - 1, Math.ceil(Math.max(...pts.map(p => p[1])) + pad));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const d = nestSigned(nest, [x + 0.5, y + 0.5]), c = soft(d), i = y * W + x;
      const cyto = VAL.cyto + 0.02 * cyt(x / W, y / H);
      img[i] = c * cyto + (1 - c) * img[i] - VAL.outline * Math.exp(-(d * d) / (2 * 0.75 * 0.75));
    }
  }
  // the basement membrane: a thin dark line
  for (let x = 0; x < W; x++) { const mem = ym(x); for (let y = Math.max(0, Math.floor(mem - 4)); y <= Math.min(H - 1, Math.ceil(mem + 4)); y++) { const d = y + 0.5 - mem; img[y * W + x] -= VAL.membrane * Math.exp(-(d * d) / (2 * 0.8 * 0.8)); } }
  // the nuclei: stromal cells first, so the epithelium and the nests lie over them
  const order = f.nuclei.slice().sort((a, b) => (a.kind === 'stroma' ? 0 : 1) - (b.kind === 'stroma' ? 0 : 1)), seg = new Uint8Array(W * H);
  for (const n of order) stampNucleus(img, n, seg, f.nuclei.indexOf(n) + 1);
  const px = new Uint8Array(W * H); for (let i = 0; i < px.length; i++) px[i] = Math.min(255, Math.round(Math.max(0, Math.min(1, img[i])) * 255 / QUANTUM) * QUANTUM); // in steps the grain hides, for a smaller file
  return { px, seg };
}
function stampNucleus(img, n, seg, index) { // the nucleus's coverage from its contour, supersampled, as tools/generate_nuclei.js renders a crop; seg: the segmentation, 1 + the nucleus's index where it covers more than half a pixel
  const R = Math.ceil(Math.max(n.a, n.b) * 1.5 + 2), SS = 3, cosP = Math.cos(n.phi), sinP = Math.sin(n.phi), soft = n.edgeSoftness / n.a;
  const x0 = Math.max(0, Math.floor(n.x - R)), x1 = Math.min(W - 1, Math.ceil(n.x + R)), y0 = Math.max(0, Math.floor(n.y - R)), y1 = Math.min(H - 1, Math.ceil(n.y + R));
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    let cov = 0, rho = 0;
    for (let v = 0; v < SS; v++) for (let u = 0; u < SS; u++) {
      const dx = x + (u + 0.5) / SS - n.x, dy = y + (v + 0.5) / SS - n.y, xr = dx * cosP + dy * sinP, yr = -dx * sinP + dy * cosP, ex = xr / n.a, ey = yr / n.b;
      const r = Math.hypot(ex, ey) / (1 + contourModulation(n, Math.atan2(ey, ex)));
      cov += r <= 1 - soft ? 1 : r >= 1 + soft ? 0 : 0.5 - 0.5 * Math.sin(((r - 1) / soft) * Math.PI / 2); rho += Math.min(r, 1.2);
    }
    cov /= SS * SS; rho /= SS * SS;
    if (cov <= 0.002) continue;
    let nv = n.value + n.textureAmp * n.tex(Math.max(0, Math.min(0.999, (x - n.x) / SIZE + 0.5)), Math.max(0, Math.min(0.999, (y - n.y) / SIZE + 0.5)));
    if (rho > 0.72) nv -= n.rim * Math.min(1, (rho - 0.72) / 0.28);
    const i = y * W + x; img[i] = cov * nv + (1 - cov) * img[i];
    if (seg && cov >= 0.5) seg[i] = index;
  }
}

// ---- the fields
const fields = [];
let id = 0;
for (const split of ['train', 'test']) {
  const per = split === 'train' ? PER_PATTERN_TRAIN : PER_PATTERN_TEST, batch = [];
  for (const P of PATTERNS) for (let i = 0; i < per; i++) batch.push(makeField(P, id++));
  for (let i = batch.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [batch[i], batch[j]] = [batch[j], batch[i]]; }
  batch.forEach((f, i) => { f.split = split; f.name = `${split === 'train' ? 'B' : 'Z'}${String(i + 1).padStart(3, '0')}`; });
  fields.push(...batch);
}
const t0 = Date.now();
const rendered = fields.map((f, i) => { if (i % 50 === 0) process.stderr.write(`rendering field ${i + 1} of ${fields.length}\n`); return Object.assign({ f }, renderField(f)); });
process.stderr.write(`rendered ${fields.length} fields in ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);

const KINDS = ['epi', 'nest', 'cord', 'single', 'stroma'], SUBTYPES = ['bland', 'enlarged', 'hyperchromatic', 'irregular', 'coarse', 'combined', 'stroma'];
// ---- the data file: a PNG per field without its grain (the loader adds it back from grainSeed), the segmentation as
// a second PNG (1 + the index of the nucleus covering each pixel, 0 elsewhere, what a nuclear segmentation would give),
// the membrane sampled every 8 px, the nuclei with their truth, the nest outlines for drawing
fs.mkdirSync(OUT, { recursive: true });
const records = rendered.map(({ f, px, seg }) => {
  const ym = membraneOf(f);
  return { id: f.id, name: f.name, split: f.split, pattern: f.pattern, label: f.label, grainSeed: SEED * 7 + f.id, rows: f.rows,
    membrane: Array.from({ length: W / 8 + 1 }, (_, i) => Math.round(ym(Math.min(W, i * 8)) * 10) / 10), surface: Math.round(f.thickness * 10) / 10,
    nuclei: f.nuclei.map(n => [n.x, n.y, KINDS.indexOf(n.kind), n.label, SUBTYPES.indexOf(n.subtype), n.below ? 1 : 0, n.group == null ? -1 : n.group]), // [x, y, kind, atypical, subtype, below, nest]
    nests: f.nests.map(n => n.kind === 'cords' ? { kind: 'cords', halfWidth: Math.round(n.halfWidth * 10) / 10, paths: n.paths.map(p => p.map(q => [Math.round(q[0] * 10) / 10, Math.round(q[1] * 10) / 10])) } : n.kind === 'jagged' ? { kind: 'jagged', connected: !!n.connected, outlines: n.outlines.map(poly => poly.map(q => [Math.round(q[0] * 10) / 10, Math.round(q[1] * 10) / 10])) } : { kind: n.single ? 'single' : 'round', cx: Math.round(n.cx * 10) / 10, cy: Math.round(n.cy * 10) / 10, rx: Math.round(n.rx * 10) / 10, ry: Math.round(n.ry * 10) / 10, rot: Math.round(n.rot * 100) / 100, amp: n.amp, k: n.k, phase: Math.round(n.phase * 100) / 100 }),
    png: FL.encodePNGNode(W, H, px).toString('base64'), seg: FL.encodePNGNode(W, H, seg).toString('base64') };
});
const meta = { seed: SEED, w: W, h: H, size: SIZE, grain: FL.GRAIN, quantum: QUANTUM, kinds: KINDS, subtypes: SUBTYPES, nucleus: ['x', 'y', 'kind', 'atypical', 'subtype', 'below', 'nest'], question: 'Invasion?', short: 'Invasion', positions: true,
  classes: [{ key: 'notInvasive', name: 'Not invasive' }, { key: 'invasive', name: 'Invasive carcinoma' }],
  patterns: PATTERNS.map(({ key, name, short, label, cues, blurb }) => ({ key, name, short, label, cues, blurb })),
  blurb: 'A strip of bladder: urothelium on its basement membrane, stroma beneath. Every field is one of five patterns and carries one label, invasive or not. Invasion is atypical cells below the membrane in angulated nests: every pattern that lacks one of the three cues is a mimic, and a real one.',
  train: records.filter(r => r.split === 'train').length, test: records.filter(r => r.split === 'test').length };
const js = `window.LECTURE_FIELDS = ${JSON.stringify({ meta, train: records.filter(r => r.split === 'train'), test: records.filter(r => r.split === 'test') })};\n`;
fs.writeFileSync(path.join(OUT, 'fields_data.js'), js);

// ---- contact sheets in H&E colour: the first fields of each pattern, at 2×, as they are and with the truth over them
const HE = [[0, [38, 20, 88]], [0.38, [88, 56, 148]], [0.62, [160, 122, 186]], [0.8, [226, 196, 218]], [0.92, [244, 226, 236]], [1, [250, 240, 245]]];
function he(v) { const t = v / 255; let i = 0; while (i < HE.length - 2 && t > HE[i + 1][0]) i++; const [t0, c0] = HE[i], [t1, c1] = HE[i + 1], k = Math.max(0, Math.min(1, (t - t0) / (t1 - t0))); return [0, 1, 2].map(j => Math.round(c0[j] + (c1[j] - c0[j]) * k)); }
function contactSheet(file, truth) {
  const scale = 2, cw = W * scale, ch = H * scale, gap = 10, per = SHEET_PER_PATTERN, SW = per * cw + (per + 1) * gap, SH = PATTERNS.length * ch + (PATTERNS.length + 1) * gap, rgb = new Uint8Array(SW * SH * 3).fill(250);
  PATTERNS.forEach((P, row) => rendered.filter(r => r.f.pattern === P.key && r.f.split === 'train').slice(0, per).forEach(({ f, px }, col) => {
    const ox = gap + col * (cw + gap), oy = gap + row * (ch + gap), grained = FL.withGrain(px, SEED * 7 + f.id, FL.GRAIN);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) { const c = he(grained[Math.floor(y / scale) * W + Math.floor(x / scale)]), o = ((oy + y) * SW + ox + x) * 3; rgb[o] = c[0]; rgb[o + 1] = c[1]; rgb[o + 2] = c[2]; }
    if (!truth) return;
    const put = (x, y, c) => { if (x < 0 || y < 0 || x >= cw || y >= ch) return; const o = ((oy + y) * SW + ox + x) * 3; rgb[o] = c[0]; rgb[o + 1] = c[1]; rgb[o + 2] = c[2]; };
    const ym = membraneOf(f);
    for (let x = 0; x < cw; x++) put(x, Math.round(ym(x / scale) * scale), [42, 120, 214]); // the membrane in blue
    for (const nest of f.nests) { // nest outlines in green
      const pts = nest.kind === 'cords' ? [].concat(...nest.paths) : nest.kind === 'jagged' ? [].concat(...nest.outlines) : [[nest.cx - nest.R, nest.cy - nest.R], [nest.cx + nest.R, nest.cy + nest.R]], pad = (nest.kind === 'cords' ? nest.halfWidth : nest.kind === 'jagged' ? 0 : nest.R) + 4;
      const bx0 = Math.max(0, Math.floor((Math.min(...pts.map(p => p[0])) - pad) * scale)), bx1 = Math.min(cw - 1, Math.ceil((Math.max(...pts.map(p => p[0])) + pad) * scale)), by0 = Math.max(0, Math.floor((Math.min(...pts.map(p => p[1])) - pad) * scale)), by1 = Math.min(ch - 1, Math.ceil((Math.max(...pts.map(p => p[1])) + pad) * scale));
      for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) { const d = Math.abs(nestSigned(nest, [x / scale, y / scale])); if (d < 0.6) put(x, y, [40, 160, 90]); }
    }
    for (const n of f.nuclei) { const r = (Math.max(n.a, n.b) + 2.5) * scale, c = n.label ? [235, 104, 52] : n.kind === 'stroma' ? [120, 120, 120] : [42, 120, 214]; for (let a = 0; a < 90; a++) put(Math.round(n.x * scale + r * Math.cos(a / 90 * TAU)), Math.round(n.y * scale + r * Math.sin(a / 90 * TAU)), c); }
  }));
  fs.writeFileSync(path.join(OUT, file), gen.encodePNG(SW, SH, 3, rgb));
}
contactSheet('contact_sheet_fields.png', false);
contactSheet('contact_sheet_fields_truth.png', true);

// ---- what was drawn
const count = (arr, f) => arr.filter(f).length, mean = (arr, f) => arr.reduce((a, x) => a + f(x), 0) / arr.length;
console.log(`wrote ${meta.train} training and ${meta.test} test fields of ${W} × ${H} to ${OUT} (${(js.length / 1024).toFixed(0)} KB)`);
for (const P of PATTERNS) {
  const fs_ = fields.filter(f => f.pattern === P.key);
  console.log(`  ${P.key.padEnd(7)} ${String(fs_.length).padStart(3)} fields · nuclei ${Math.min(...fs_.map(f => f.nuclei.length))}–${Math.max(...fs_.map(f => f.nuclei.length))} (mean ${mean(fs_, f => f.nuclei.length).toFixed(1)}) · atypical ${mean(fs_, f => count(f.nuclei, n => n.label)).toFixed(1)} · below the membrane ${mean(fs_, f => count(f.nuclei, n => n.below)).toFixed(1)} (atypical below ${mean(fs_, f => count(f.nuclei, n => n.below && n.label)).toFixed(1)}) · nests ${mean(fs_, f => f.nests.filter(n => !n.single).length).toFixed(1)} · single cells on ${count(fs_, f => f.nests.some(n => n.single))}${P.key === 'inv' ? ` · nests from the epithelium ${count([].concat(...fs_.map(f => f.nests.filter(n => n.kind === 'jagged'))), n => n.connected)} of ${[].concat(...fs_.map(f => f.nests.filter(n => n.kind === 'jagged'))).length}` : ''}`);
}
