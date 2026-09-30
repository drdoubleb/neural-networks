#!/usr/bin/env node
/*
 * check_reports.js — trains the report language model in Node, as the page will: every report is a run of word
 * tokens, the model predicts each next token, and what it learned is read off in the ways the lecture needs. It
 * reports, per epoch, the loss on the training and test reports and the test loss by section (the boilerplate of
 * the gross falls first, the diagnosis last); after training, the diagnosis it writes for the test reports when
 * given everything up to "DIAGNOSIS:" and when given the findings block alone (writing the whole report itself),
 * the same for the held-out combination the training set never held (invasion under a normal surface), what it
 * writes with no findings at all (the base rates of the queue), and where the diagnosis tokens' attention lands: on
 * the findings block, on its decisive lines, on the distractor, on the microscopic description.
 *   node tools/check_reports.js [--dim 32] [--layers 2] [--heads 2] [--dk 8] [--ffn 32] [--cost 0.1] [--positions 0]
 *                               [--epochs 15] [--lr 0.003] [--opt adam|sgd] [--batch 8] [--decay 0.0001] [--clip 1]
 *                               [--seeds 1] [--seed 1] [--docs 800] [--no-hedge] [--report-only] [--curve] [--skip-gen] [--samples 20]
 * --report-only keeps the loss to the report (the block is given, never predicted); --no-hedge drops the reports
 * signed out as suspicious for invasion from the training set; --docs trains on the first N training reports;
 * --save FILE writes the last seed's trained model with its vocabulary as a script (window.REPORT_LM = {...}), the
 * model the page ships.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { LanguageModel, mulberry32, gradientCheckLM } = require('../js/nn.js');
const R = require('../js/reports.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const has = k => process.argv.includes(k);
const cfg = { dim: num('--dim', 32), layers: num('--layers', 2), heads: num('--heads', 2), dk: num('--dk', 8), ffn: num('--ffn', 32), cost: String(arg('--cost', '0.1')).split(',').map(Number), positions: num('--positions', 0), epochs: num('--epochs', 15), lr: num('--lr', 0.003), opt: arg('--opt', 'adam'), batch: num('--batch', 8), decay: num('--decay', 0.0001), clip: num('--clip', 1), seeds: num('--seeds', 1), docs: num('--docs', 800), samples: num('--samples', 20), temperature: num('--temperature', 0), ground: num('--ground', 0) };
const noHedge = has('--no-hedge'), reportOnly = has('--report-only'), curve = has('--curve'), skipGen = has('--skip-gen'), save = arg('--save', null);
const window = {}; new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', 'reports', 'reports_data.js'), 'utf8'))(window);
const DATA = window.LECTURE_REPORTS;
const gc = gradientCheckLM(); console.log(`Language model gradient check: worst relative error ${gc.worst.toExponential(2)} over ${gc.checked} gradients`);
const pct = v => `${(100 * v).toFixed(0)}%`;

let trainDocs = DATA.train.slice(0, cfg.docs); if (noHedge) trainDocs = trainDocs.filter(r => r.case.cls !== 'suspicious');
const vocab = R.buildVocab(DATA.train.map(r => r.text)); // the vocabulary of the full training set, so that a smaller corpus shares it
const prep = r => { const tokens = R.encode(vocab, r.text), words = R.decode(vocab, tokens), { sections, lines } = R.sectionsOf(words); const weights = reportOnly ? Float64Array.from({ length: tokens.length - 1 }, (_, i) => (sections[i + 1] === 'findings' ? 0 : 1)) : null; return { r, tokens, words, sections, lines, weights, dx: R.diagnosisOf(words), dxSection: R.diagnosisSectionOf(words) }; };
const train = trainDocs.map(prep), test = DATA.test.map(prep), held = DATA.held.map(prep);
const meanLen = train.reduce((a, d) => a + d.tokens.length, 0) / train.length;
console.log(`corpus: ${train.length} training reports (${trainDocs.filter(r => r.case.cls === 'suspicious').length} suspicious), ${test.length} test, ${held.length} held out · vocabulary ${vocab.size} · ${meanLen.toFixed(0)} tokens per report · loss on ${reportOnly ? 'the report only' : 'every token'}`);
console.log(`model: dim ${cfg.dim}, ${cfg.layers} layers × ${cfg.heads} heads (dk ${cfg.dk}, ffn ${cfg.ffn}), distance cost init ${cfg.cost.join('/')}${cfg.cost.length === 1 ? ' (halving per head)' : ' per head'}${cfg.positions ? `, ${cfg.positions} learned positions` : ''} · ${cfg.opt} lr ${cfg.lr}, batch ${cfg.batch}, decay ${cfg.decay}, clip ${cfg.clip}`);

const bySection = (docs, per) => { const sum = {}, n = {}; docs.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const s = d.sections[i + 1]; sum[s] = (sum[s] || 0) + per[k][i]; n[s] = (n[s] || 0) + 1; } }); const out = {}; for (const s of R.SECTIONS) out[s] = n[s] ? sum[s] / n[s] : null; return out; };
const generated = (lm, prefix, maxTokens, temperature, rng) => { const g = lm.generate(prefix, { maxTokens, temperature, rng, stop: vocab.end }); return R.decode(vocab, prefix.concat(g.tokens)); };
const wellFormed = words => { let last = -1; for (const h of ['SPECIMEN', 'CLINICAL', 'GROSS', 'MICROSCOPIC', 'DIAGNOSIS']) { const i = words.indexOf(h); if (i < 0 || i < last || words.indexOf(h, i + 1) >= 0) return false; last = i; } return true; };
const tally = (rows, label) => { const c = {}; for (const r of rows) c[r] = (c[r] || 0) + 1; console.log(`  ${label}: ` + Object.entries(c).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} ${n}`).join(' · ')); };

const seed0 = num('--seed', 1); // the first seed; --seeds N runs N seeds from it
for (let seed = seed0; seed < seed0 + cfg.seeds; seed++) {
  const lm = new LanguageModel({ vocabSize: vocab.size, dim: cfg.dim, layers: cfg.layers, heads: cfg.heads, dk: cfg.dk, ffn: cfg.ffn, costInit: cfg.cost.length === 1 ? cfg.cost[0] : cfg.cost, positions: cfg.positions, clip: cfg.clip, optimizer: cfg.opt, seed });
  const rng = mulberry32(seed * 977 + 1), order = train.map((_, i) => i);
  console.log(`\n== seed ${seed} · ${lm.parameterCount()} parameters · ${lm.describe()}`);
  for (let epoch = 1; epoch <= cfg.epochs; epoch++) {
    const t0 = Date.now(); let loss = 0, nb = 0;
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let b = 0; b < order.length; b += cfg.batch) { loss += lm.trainBatch(order.slice(b, b + cfg.batch).map(i => train[i]), cfg.lr, cfg.decay); nb++; }
    const ev = lm.evaluate(test), sec = bySection(test, ev.per), costs = lm.layers.map(c => Array.from(c.beta, b => Math.log1p(Math.exp(b)).toFixed(2)).join('/')).join(' ');
    let ground = ''; // grounding: the diagnosis written from the block alone, on the first --ground test reports, every fifth epoch
    if (cfg.ground && (epoch % 5 === 0 || epoch === cfg.epochs)) { let ok = 0; for (const d of test.slice(0, cfg.ground)) { const b = generated(lm, d.tokens.slice(0, d.words.indexOf('SPECIMEN')), 220, 0, rng); if (R.classOf(R.diagnosisOf(b)) === R.classOf(d.dx)) ok++; } ground = ` · from the block: class right ${pct(ok / cfg.ground)}`; }
    if (curve || epoch === cfg.epochs || epoch % 5 === 0 || epoch === 1) console.log(`epoch ${String(epoch).padStart(3)} · train loss ${(loss / nb).toFixed(3)} · test loss ${ev.loss.toFixed(3)} · next token right ${pct(ev.accuracy)} · by section: ${R.SECTIONS.map(s => `${s} ${sec[s] == null ? '–' : sec[s].toFixed(2)}`).join(', ')} · cost/token ${costs}${ground} · ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
  if (save) { const ev = lm.evaluate(test), out = lm.toJSON({ name: 'the report model', trained: `${cfg.epochs} epochs on ${train.length} reports`, words: vocab.words, testLoss: +ev.loss.toFixed(4), testAccuracy: +ev.accuracy.toFixed(4) }); fs.writeFileSync(save, `window.REPORT_LM = ${JSON.stringify(out)};\n`); console.log(`saved ${save} (${(fs.statSync(save).size / 1024).toFixed(0)} KB)`); }
  if (skipGen) continue;
  const t1 = Date.now();
  // (a) the diagnosis given everything up to "DIAGNOSIS:"; (b) the whole report given the findings block alone
  const dxTest = (docs, label) => {
    const rowsA = [], rowsB = [], confA = [], confB = []; let okA = 0, okMpA = 0, okB = 0, okMpB = 0, formed = 0, okDxA = 0, okDxB = 0;
    for (const d of docs) {
      const iDx = d.words.indexOf('DIAGNOSIS'), a = generated(lm, d.tokens.slice(0, iDx + 2), 40, cfg.temperature, rng), dxA = R.diagnosisOf(a), secA = R.diagnosisSectionOf(a);
      if (dxA === d.dx) okDxA++; if (secA === d.dxSection) okMpA++; if (R.classOf(dxA) === R.classOf(d.dx)) okA++; confA.push(`${R.classOf(d.dx)}→${R.classOf(dxA)}`);
      const iSp = d.words.indexOf('SPECIMEN'), b = generated(lm, d.tokens.slice(0, iSp), 220, cfg.temperature, rng), dxB = R.diagnosisOf(b), secB = R.diagnosisSectionOf(b);
      if (wellFormed(b)) formed++; if (dxB === d.dx) okDxB++; if (secB === d.dxSection) okMpB++; if (R.classOf(dxB) === R.classOf(d.dx)) okB++; confB.push(`${R.classOf(d.dx)}→${R.classOf(dxB)}`);
      rowsA.push({ d, dxA, secA }); rowsB.push({ d, dxB, secB, text: R.detokenize(b) });
    }
    const n = docs.length;
    console.log(`${label} · given everything up to DIAGNOSIS: diagnosis sentence exact ${pct(okDxA / n)}, class right ${pct(okA / n)}, whole diagnosis line exact ${pct(okMpA / n)}`);
    console.log(`${label} · given the findings block alone: report well formed ${pct(formed / n)}, diagnosis sentence exact ${pct(okDxB / n)}, class right ${pct(okB / n)}, whole diagnosis line exact ${pct(okMpB / n)}`);
    const wrongA = confA.filter(c => c.split('→')[0] !== c.split('→')[1]), wrongB = confB.filter(c => c.split('→')[0] !== c.split('→')[1]);
    if (wrongA.length) tally(wrongA, 'class errors, with the prefix'); if (wrongB.length) tally(wrongB, 'class errors, from the block');
    return { rowsA, rowsB };
  };
  dxTest(test, 'test reports');
  const heldRes = dxTest(held, 'held out (invasion under a normal surface)');
  heldRes.rowsB.slice(0, 3).forEach(({ d, dxB, secB }) => console.log(`  held ${d.r.name}: wrote "${secB || '(no diagnosis)'}" · truth "${d.dxSection}"`));
  // (c) nothing given at all: from the start token the model writes a whole case, findings block included; the most
  // probable one is the modal report, and reports drawn at temperature 1 follow the base rates it learned
  const bare = [vocab.start], greedy = generated(lm, bare, 260, 0, rng);
  console.log(`nothing given, most probable report: "${R.diagnosisSectionOf(greedy) || '(no diagnosis)'}"`);
  const drawn = []; for (let s = 0; s < cfg.samples; s++) drawn.push(R.classOf(R.diagnosisOf(generated(lm, bare, 260, 1, rng))));
  tally(drawn, `nothing given, ${cfg.samples} reports drawn at temperature 1, by class`);
  const base = {}; for (const d of train) { const c = R.classOf(d.dx); base[c] = (base[c] || 0) + 1; }
  console.log('  training base rates: ' + Object.entries(base).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} ${pct(n / train.length)}`).join(' · '));
  // (d) where the diagnosis tokens look: their attention, per layer (heads averaged), by section and by block line
  const share = lm.layers.map(() => ({ bySec: {}, byLine: {}, n: 0 }));
  for (const d of test) {
    const fw = lm.forward(d.tokens), iDx = d.words.indexOf('DIAGNOSIS'); let end = iDx + 2; while (end < d.words.length && d.words[end] !== '.') end++;
    for (let i = iDx + 2; i <= end && i < d.words.length; i++) lm.layers.forEach((c, l) => { const S = share[l]; S.n++; for (let j = 0; j <= i; j++) { let a = 0; for (let h = 0; h < c.H; h++) a += fw.ctxs[l].Ah[h][i][j]; a /= c.H; const sec = d.sections[j], line = d.lines[j]; S.bySec[sec] = (S.bySec[sec] || 0) + a; if (line) S.byLine[line] = (S.byLine[line] || 0) + a; } });
  }
  share.forEach((S, l) => console.log(`attention of the diagnosis tokens, layer ${l + 1}: ${R.SECTIONS.map(s => `${s} ${pct((S.bySec[s] || 0) / S.n)}`).join(', ')} · block lines: ${Object.entries(S.byLine).map(([k, v]) => `${k} ${pct(v / S.n)}`).join(', ')}`));
  console.log(`(generation and attention: ${((Date.now() - t1) / 1000).toFixed(0)} s)`);
}
