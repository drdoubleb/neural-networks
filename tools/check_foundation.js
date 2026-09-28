/*
 * check_foundation.js — measures the miniature foundation model of the 4 · Foundation stage, exactly as the page
 * pretrains it: 100 unlabelled nuclei (34 / 33 / 33 from the three image questions' training sets), a convolutional
 * encoder (4 filters 5×5, pool 4×4, a linear layer to a code of 8) trained by instance discrimination on batches of
 * 16 nuclei with two random views each (flips, rotations, and the other lab's scan). After every epoch it reports the
 * contrastive loss on the batches and on 60 held-out nuclei, and what the frozen code is worth: a single layer trained
 * on it with 20 labelled cases per question, scored on 40 nuclei it never saw (the question's 20 test nuclei and its
 * 20 held-out training nuclei).
 *   node tools/check_foundation.js [--seeds 3] [--epochs 100] [--no-labs]
 */
const fs = require('fs');
const path = require('path');
const { Net, Contrastive, mulberry32, fitStandardizer, gradientCheckCL } = require('../js/nn.js');
const DS = require('../js/dataset.js');

const arg = (k, d) => (process.argv.includes(k) ? +process.argv[process.argv.indexOf(k) + 1] : d);
const nSeeds = arg('--seeds', 3), epochs = arg('--epochs', 100), labs = !process.argv.includes('--no-labs');
const window = {};
for (const f of ['atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const tasks = {};
for (const [id, raw] of Object.entries(window.LECTURE_TASKS)) tasks[id] = DS.prepare(raw);

const gc = gradientCheckCL();
console.log(`Contrastive gradient check: worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} weights ${gc.worst < 1e-4 ? '(ok)' : '(FAILED)'}`);

// the same constants as app.js
const FM = { tasks: ['atypia', 'enlargement', 'irregularity'], counts: [34, 33, 33], heldPer: 20, heldBatch: 15, batch: 16, lr: 0.05, K: 4, code: 8, tau: 0.2, probePerClass: 10, probeEpochs: 150, probeLr: 0.1 };
const set = [], held = [];
FM.tasks.forEach((id, k) => { const tr = tasks[id].train; tr.slice(0, FM.counts[k]).forEach(s => set.push({ s, task: id })); tr.slice(FM.counts[k], FM.counts[k] + FM.heldPer).forEach(s => held.push({ s, task: id })); });
const size = tasks.atypia.size;
const std = fitStandardizer(set.map(e => e.s.variants.A.ink), { perDimScale: false });
const view = (e, lab, t) => std.apply(t ? DS.dihedral(e.s.variants[lab].ink, size, t) : e.s.variants[lab].ink);
const randomView = (e, rng) => { const L = labs ? ['A', 'B'] : ['A']; const lab = L[Math.floor(rng() * L.length)], t = Math.floor(rng() * 8); return view(e, lab, t); };
const heldRng = mulberry32(99), heldPairs = held.map(e => [randomView(e, heldRng), randomView(e, heldRng)]), heldBatches = [];
for (let b = 0; b < heldPairs.length; b += FM.heldBatch) heldBatches.push(heldPairs.slice(b, b + FM.heldBatch));

function heldEval(cl) { let loss = 0, hit = 0; for (const b of heldBatches) { const r = cl.evaluate(b); loss += r.loss / heldBatches.length; hit += r.pairAcc / heldBatches.length; } return { loss, pairAcc: hit }; }
function probe(cl, id, testLab = 'A') {
  const ds = tasks[id], tr = [...ds.train.filter(s => s.label).slice(0, FM.probePerClass), ...ds.train.filter(s => !s.label).slice(0, FM.probePerClass)];
  const code = (s, lab) => cl.encode(std.apply(s.variants[lab].ink));
  const rows = tr.map(s => code(s, 'A')), labels = tr.map(s => s.label), st = fitStandardizer(rows, { perDimScale: true });
  const net = new Net({ inputSize: FM.code, hidden: [], activation: 'relu', seed: 1 }), rnd = mulberry32(1), idx = rows.map((_, i) => i);
  for (let ep = 0; ep < FM.probeEpochs; ep++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let b = 0; b < idx.length; b += 8) { const bb = idx.slice(b, b + 8); net.trainBatch(bb.map(i => st.apply(rows[i])), bb.map(i => labels[i]), FM.probeLr, 0); }
  }
  const scored = [...ds.test, ...held.filter(e => e.task === id).map(e => e.s)]; // 40 nuclei the code and the probe never saw
  return net.evaluate(scored.map(s => st.apply(code(s, testLab))), scored.map(s => s.label)).accuracy;
}
const pc = v => String(Math.round(v * 100)).padStart(3) + '%';
const runs = [];
for (let seed = 1; seed <= nSeeds; seed++) {
  const cl = new Contrastive({ imageSize: size, conv: { K: FM.K, f: 5, pool: 4 }, code: FM.code, tau: FM.tau, seed });
  const rng = mulberry32(seed * 31 + 7), order = set.map((_, i) => i), hist = [];
  const record = (epoch, loss) => { const h = heldEval(cl); hist.push(Object.assign({ epoch, loss, heldLoss: h.loss, pairAcc: h.pairAcc }, Object.fromEntries(FM.tasks.map(id => [id, probe(cl, id)])))); };
  record(0, null);
  const t0 = Date.now();
  for (let e = 1; e <= epochs; e++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let loss = 0, nb = 0;
    for (let b0 = 0; b0 < order.length; b0 += FM.batch) { const pairs = order.slice(b0, b0 + FM.batch).map(i => [randomView(set[i], rng), randomView(set[i], rng)]); loss += cl.step(pairs, FM.lr, 0).loss; nb++; }
    record(e, loss / nb);
  }
  const ms = (Date.now() - t0) / epochs;
  const atB = FM.tasks.map(id => probe(cl, id, 'B'));
  runs.push({ hist, atB });
  console.log(`\nseed ${seed} · ${ms.toFixed(0)} ms per epoch including the measurements${labs ? '' : ' · views without the other lab'}`);
  console.log('  epoch  batches  held-out  pair-hit   ' + FM.tasks.map(id => id.padStart(12)).join(''));
  for (const h of hist) if (h.epoch === 0 || h.epoch % 10 === 0 || h.epoch === epochs) console.log(`  ${String(h.epoch).padStart(5)}  ${h.loss == null ? '      –' : h.loss.toFixed(2).padStart(7)}  ${h.heldLoss.toFixed(2).padStart(8)}  ${pc(h.pairAcc).padStart(8)}   ` + FM.tasks.map(id => pc(h[id]).padStart(12)).join(''));
  console.log(`  the same single layer scored on the other lab's scans of the test nuclei: ${FM.tasks.map((id, k) => `${id} ${pc(atB[k]).trim()}`).join(' · ')}`);
}
const mean = f => runs.reduce((a, r) => a + f(r), 0) / runs.length;
const at = e => runs.map(r => r.hist.find(h => h.epoch === e));
console.log(`\nMean of ${nSeeds} seeds:`);
for (const e of [0, 10, 25, 50, epochs].filter((v, i, a) => v <= epochs && a.indexOf(v) === i)) {
  const hs = at(e);
  console.log(`  epoch ${String(e).padStart(3)}: held-out loss ${(hs.reduce((a, h) => a + h.heldLoss, 0) / hs.length).toFixed(2)} · pair-hit ${pc(hs.reduce((a, h) => a + h.pairAcc, 0) / hs.length)} · a single layer on the code: ` + FM.tasks.map(id => `${id} ${pc(hs.reduce((a, h) => a + h[id], 0) / hs.length).trim()}`).join(' · '));
}
console.log(`  at the other lab after ${epochs} epochs: ` + FM.tasks.map((id, k) => `${id} ${pc(mean(r => r.atB[k])).trim()}`).join(' · '));
