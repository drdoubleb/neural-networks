#!/usr/bin/env node
/*
 * check_chat.js — trains the report model's recipe on the transcripts of the chat question and takes its measures:
 * the loss by role per epoch (system, user, findings, question, assistant, tool); the answers right on test
 * transcripts, exactly and by their first word, by style and by question; the agent loop (does the model write
 * findings() first, and answer right once the findings come back); the held-out phrasings of the instructions; and
 * what the model says to questions and system lines it never trained on.
 *   node tools/check_chat.js [--dim 48] [--layers 2] [--heads 2] [--dk 12] [--ffn 48] [--cost 0.05,0.003] [--positions 200]
 *                            [--epochs 30] [--lr 0.005] [--opt adam] [--batch 8] [--decay 0.0001] [--clip 1] [--seed 1]
 *                            [--curve] [--ground 24] [--ground-every 1] [--skip-gen] [--save FILE] [--load FILE]
 * --ground N measures, every --ground-every epochs, the answers right on the first N test transcripts; --save writes
 * the trained model with its vocabulary, its history and its measures as window.CHAT_LM; --load takes the measures on
 * a saved model.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { LanguageModel, mulberry32 } = require('../js/nn.js');
const R = require('../js/reports.js');
const C = require('../js/chat.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const has = k => process.argv.includes(k);
const cfg = { dim: num('--dim', 48), layers: num('--layers', 2), heads: num('--heads', 2), dk: num('--dk', 12), ffn: num('--ffn', 48), cost: String(arg('--cost', '0.05,0.003')).split(',').map(Number), positions: num('--positions', 200), epochs: num('--epochs', 30), lr: num('--lr', 0.005), opt: arg('--opt', 'adam'), batch: num('--batch', 8), decay: num('--decay', 0.0001), clip: num('--clip', 1), seed: num('--seed', 1), ground: num('--ground', 24), groundEvery: num('--ground-every', 1) };
const curve = has('--curve'), skipGen = has('--skip-gen'), save = arg('--save', null), load = arg('--load', null);
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = loadJs(path.join(__dirname, '..', 'data', 'chat', 'chat_data.js')).LECTURE_CHAT, LOADED = load ? loadJs(load).CHAT_LM : null;
const vocab = LOADED ? { words: LOADED.words, index: new Map(LOADED.words.map((w, i) => [w, i])), start: 0, end: 1, unk: 2, size: LOADED.words.length } : R.buildVocab(D.train.map(t => t.text));
const prep = t => { const tokens = R.encode(vocab, t.text), words = R.decode(vocab, tokens); return Object.assign({}, t, { tokens, words, roles: C.rolesOf(words) }); };
const train = D.train.map(prep), test = D.test.map(prep), reph = D.rephrased.map(prep), NL = vocab.index.get('\n');
const pct = v => `${Math.round(100 * v)}%`;
console.log(`corpus: ${train.length} training transcripts, ${test.length} test, ${reph.length} with the held-out phrasings · vocabulary ${vocab.size} · ${(train.reduce((s, d) => s + d.tokens.length, 0) / train.length).toFixed(0)} tokens per transcript`);
console.log(`model: dim ${cfg.dim}, ${cfg.layers} layers × ${cfg.heads} heads (dk ${cfg.dk}, ffn ${cfg.ffn}), distance cost init ${cfg.cost.join('/')} per head, ${cfg.positions} learned positions · ${cfg.opt} lr ${cfg.lr}, batch ${cfg.batch}, decay ${cfg.decay}, clip ${cfg.clip}`);
const lm = LOADED ? LanguageModel.fromJSON(LOADED) : new LanguageModel({ vocabSize: vocab.size, dim: cfg.dim, layers: cfg.layers, heads: cfg.heads, dk: cfg.dk, ffn: cfg.ffn, costInit: cfg.cost.length === 1 ? cfg.cost[0] : cfg.cost, positions: cfg.positions, clip: cfg.clip, optimizer: cfg.opt, seed: cfg.seed });
const rng = mulberry32(cfg.seed * 977 + 1), order = train.map((_, i) => i), hist = [];
// the loss by role: per[i] is the loss of token i + 1, so the role is that token's
const byRole = (docs, per) => { const sum = {}, n = {}; docs.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const r = d.roles[i + 1]; sum[r] = (sum[r] || 0) + per[k][i]; n[r] = (n[r] || 0) + 1; } }); const out = {}; for (const r of C.ROLES) out[r] = n[r] ? sum[r] / n[r] : null; return out; };
// writing: the prefix continued greedily to the end token or a newline, as the page's chat does
const write = (prefix, max) => { const st = lm.genState(prefix), out = []; while (out.length < max) { const { token } = lm.genNext(st, { temperature: 0 }); if (token === vocab.end || token === NL) break; out.push(token); } return R.detokenize(R.decode(vocab, out)); };
const header = (words, which) => { let i = -1; for (let k = 0; k + 1 < words.length; k++) if (words[k] === 'ASSISTANT' && words[k + 1] === ':') { i = k; if (which === 'first') break; } return i; };
const answerPrefix = d => d.tokens.slice(0, header(d.words, 'last') + 2); // up to the last "ASSISTANT:" (the agent's second turn, the findings back)
const measure = docs => { let exact = 0, lead = 0; for (const d of docs) { const a = write(answerPrefix(d), 60); if (C.same(a, d.answer)) exact++; if (C.classOf(a) === C.classOf(d.answer)) lead++; } return { n: docs.length, exact, lead }; };
const agentLoop = d => { const call = write(d.tokens.slice(0, header(d.words, 'first') + 2), 20); if (!C.callOf(call)) return { called: false, right: false }; return { called: true, right: C.same(write(answerPrefix(d), 60), d.answer) }; };
const fmt = m => `${m.exact}/${m.n} exact, ${m.lead}/${m.n} first word`;
console.log(`\n== ${LOADED ? `loaded ${load} (${LOADED.trained})` : `seed ${cfg.seed}`} · ${lm.parameterCount()} parameters · ${lm.describe()}`);
for (let epoch = 1; epoch <= (LOADED ? 0 : cfg.epochs); epoch++) {
  const t0 = Date.now(); let loss = 0, nb = 0;
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  for (let b = 0; b < order.length; b += cfg.batch) { loss += lm.trainBatch(order.slice(b, b + cfg.batch).map(i => train[i]), cfg.lr, cfg.decay); nb++; }
  const ev = lm.evaluate(test), roles = byRole(test, ev.per);
  let ground = null; if (cfg.ground && (epoch % cfg.groundEvery === 0 || epoch === cfg.epochs)) { const m = measure(test.slice(0, cfg.ground)); ground = m.exact / m.n; }
  hist.push(Object.assign({ epoch, loss: +(loss / nb).toFixed(4), testLoss: +ev.loss.toFixed(4), acc: +ev.accuracy.toFixed(4), ground }, Object.fromEntries(C.ROLES.map(r => [r, roles[r] == null ? null : +roles[r].toFixed(4)]))));
  if (curve || epoch === cfg.epochs || epoch % 5 === 0 || epoch === 1) console.log(`epoch ${String(epoch).padStart(3)} · train loss ${(loss / nb).toFixed(3)} · test loss ${ev.loss.toFixed(3)} · next token right ${pct(ev.accuracy)} · by role: ${C.ROLES.map(r => `${r} ${roles[r] == null ? '–' : roles[r].toFixed(2)}`).join(', ')}${ground == null ? '' : ` · answers right ${pct(ground)} of ${cfg.ground}`} · ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
let measures = null;
if (!skipGen) {
  const t1 = Date.now(), byStyle = {}, byQ = {}, byStyleReph = {};
  for (const sk of Object.keys(D.meta.styles)) { byStyle[sk] = measure(test.filter(d => d.style === sk)); byStyleReph[sk] = measure(reph.filter(d => d.style === sk)); }
  for (const q of D.meta.questions) byQ[q.key] = measure(test.filter(d => d.style !== 'agent' && d.q === q.q));
  let called = 0, right = 0, nA = 0; for (const d of test.filter(d => d.style === 'agent')) { nA++; const r = agentLoop(d); if (r.called) called++; if (r.right) right++; }
  console.log(`\nanswers on the test transcripts (the prefix up to the last ASSISTANT:): ${Object.keys(byStyle).map(sk => `${D.meta.styles[sk].name} ${fmt(byStyle[sk])}`).join(' · ')}`);
  console.log(`with the held-out phrasing of the instruction: ${Object.keys(byStyleReph).map(sk => `${D.meta.styles[sk].name} ${fmt(byStyleReph[sk])}`).join(' · ')}`);
  console.log(`by question (brief and full): ${D.meta.questions.map(q => `${q.q.replace(/\?$/, '')} ${byQ[q.key].exact}/${byQ[q.key].n}`).join(' · ')}`);
  console.log(`the agent loop: writes findings() first for ${called}/${nA}, then answers exactly for ${right}/${nA}`);
  // what it says to what it never trained on, for a benign and an invasive case
  const ask = (sysLine, block, q) => write(R.encode(vocab, block ? `SYSTEM: ${D.meta.persona} ${sysLine}\nUSER: Here are the findings.\n${block}\n${q}\nASSISTANT:` : `SYSTEM: ${D.meta.persona} ${sysLine}\nUSER: ${q}\nASSISTANT:`).slice(0, -1), 60);
  const blockOf = d => d.text.split('\n').filter((_, i) => i >= 2 && i < 10).join('\n');
  for (const d of [test.find(d => d.style === 'brief' && /^Benign urothelium\./.test(d.dx)), test.find(d => d.style === 'brief' && /invasive into lamina/.test(d.dx))].filter(Boolean)) {
    const block = blockOf(d); console.log(`\n${d.case} · ${d.dx}`);
    for (const q of D.meta.probes.questions) console.log(`   never asked · “${q}” → ${ask(D.meta.styles.brief.lines[0], block, q)}`);
    for (const l of D.meta.probes.lines) console.log(`   never seen line · “${l}” · Is there invasion? → ${ask(l, block, 'Is there invasion?')}`);
    console.log(`   held-out phrasing · “${D.meta.styles.full.held}” · Is there invasion? → ${ask(D.meta.styles.full.held, block, 'Is there invasion?')}`);
  }
  console.log(`\nmeasures took ${((Date.now() - t1) / 1000).toFixed(0)} s`);
  measures = { byStyle, byStyleReph, byQ, agent: { n: nA, called, right } };
}
if (save) { const ev = lm.evaluate(test), out = lm.toJSON({ name: 'the chat model', trained: `${cfg.epochs} epochs on ${train.length} transcripts`, words: vocab.words, testLoss: +ev.loss.toFixed(4), testAccuracy: +ev.accuracy.toFixed(4), groundN: cfg.ground || null, hist, measures }); fs.writeFileSync(save, `window.CHAT_LM = ${JSON.stringify(out)};\n`); console.log(`saved ${save} (${(fs.statSync(save).size / 1024).toFixed(0)} KB) with ${hist.length} epochs of history`); }
