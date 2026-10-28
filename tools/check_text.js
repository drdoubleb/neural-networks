#!/usr/bin/env node
/*
 * check_text.js — trains the page's next-word model on a text of one's own, as the Your-text question does: the text
 * is cut into documents (a poem each when the text marks its titles, else a paragraph; long ones into windows of 200
 * tokens), one document in ten is held out, and the model predicts every next token. Per epoch it reports the loss and
 * the next-word accuracy on the test documents and on the first training documents (the gap between the two is
 * memorisation), and what the model writes when given the title line of the same held-out poem. After training it
 * measures recall: given a title line, how much of the poem the model reproduces word for word, for training poems and
 * for held-out ones, and writes a few poems from titles of one's own at a temperature.
 *   node tools/check_text.js [--text data/text/mountain_interval.js] [--global LECTURE_TEXT] [--window 200]
 *                            [--dim 48] [--layers 2] [--heads 2] [--dk 12] [--ffn 48] [--cost 0.05,0.003] [--positions 200]
 *                            [--epochs 30] [--lr 0.005] [--opt adam] [--batch 4] [--decay 0.0001] [--clip 1] [--seed 1]
 *                            [--ground 20] [--sample-max 60] [--curve] [--skip-gen] [--temperature 0.8] [--save FILE] [--load FILE]
 *                            [--global-out TEXT_LM] [--keep-best] [--patience 0] [--trail-snaps 8]
 * --ground N measures the next-word accuracy on the first N training documents every epoch; --save writes the trained
 * model with its vocabulary, its history (the samples of every epoch included) and its word paths as window.TEXT_LM
 * (--global-out names the global); --keep-best saves the model of the epoch with the lowest held-out loss instead of
 * the last, its history cut there (early stopping), and --patience K stops the run once K epochs have passed without
 * a new lowest; --trail-snaps caps the word paths at N snapshots; --load takes the measures on a saved model instead
 * of training.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { LanguageModel, mulberry32, wordTrails } = require('../js/nn.js');
const T = require('../js/text.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const num = (k, d) => +arg(k, d);
const has = k => process.argv.includes(k);
const cfg = { dim: num('--dim', 48), layers: num('--layers', 2), heads: num('--heads', 2), dk: num('--dk', 12), ffn: num('--ffn', 48), cost: String(arg('--cost', '0.05,0.003')).split(',').map(Number), positions: num('--positions', 200), epochs: num('--epochs', 30), lr: num('--lr', 0.005), opt: arg('--opt', 'adam'), batch: num('--batch', 4), decay: num('--decay', 0.0001), clip: num('--clip', 1), seed: num('--seed', 1), ground: num('--ground', 20), sampleMax: num('--sample-max', 60), temperature: num('--temperature', 0.8), window: num('--window', 200) };
const curve = has('--curve'), skipGen = has('--skip-gen'), save = arg('--save', null), load = arg('--load', null), globalOut = arg('--global-out', 'TEXT_LM'), keepBest = has('--keep-best'), patience = num('--patience', 0), trailSnaps = num('--trail-snaps', 8);
const loadJs = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const textFile = arg('--text', path.join(__dirname, '..', 'data', 'text', 'mountain_interval.js')), TEXT = loadJs(textFile)[arg('--global', 'LECTURE_TEXT')];
if (!TEXT || !TEXT.text) { console.error(`no text in ${textFile}`); process.exit(1); }
const LOADED = load ? loadJs(load)[globalOut] : null;
const docs = T.documentsOf(TEXT.text, { maxTokens: cfg.window }), sp = T.splits(docs);
const vocab = LOADED ? { words: LOADED.words, index: new Map(LOADED.words.map((w, i) => [w, i])), start: 0, end: 1, unk: 2, size: LOADED.words.length } : T.vocabOf(docs);
const prep = (d, split) => { const tokens = T.encode(vocab, d.text), words = T.decode(vocab, tokens); return Object.assign({}, d, { split, tokens, words, sections: T.sectionsOf(words) }); };
const train = sp.train.map(d => prep(d, 'train')), test = sp.test.map(d => prep(d, 'test'));
const pct = v => `${Math.round(100 * v)}%`;
const tokensOf = ds => ds.reduce((s, d) => s + d.tokens.length - 2, 0);
console.log(`text: ${TEXT.meta ? `${TEXT.meta.title} (${TEXT.meta.author}, ${TEXT.meta.year})` : textFile} · ${docs.length} documents (${docs.filter(d => d.part === 1).length} ${docs.some(d => d.title) ? 'poems' : 'paragraphs'}, windows of at most ${cfg.window} tokens) · ${train.length} training, ${test.length} test · ${tokensOf(train)} training tokens, ${tokensOf(test)} test · vocabulary ${vocab.size}`);
console.log(`model: dim ${cfg.dim}, ${cfg.layers} layers × ${cfg.heads} heads (dk ${cfg.dk}, ffn ${cfg.ffn}), distance cost init ${cfg.cost.join('/')} per head, ${cfg.positions} learned positions · ${cfg.opt} lr ${cfg.lr}, batch ${cfg.batch}, decay ${cfg.decay}, clip ${cfg.clip}`);
let lm = LOADED ? LanguageModel.fromJSON(LOADED) : new LanguageModel({ vocabSize: vocab.size, dim: cfg.dim, layers: cfg.layers, heads: cfg.heads, dk: cfg.dk, ffn: cfg.ffn, costInit: cfg.cost.length === 1 ? cfg.cost[0] : cfg.cost, positions: cfg.positions, clip: cfg.clip, optimizer: cfg.opt, seed: cfg.seed });
const rng = mulberry32(cfg.seed * 977 + 1), order = train.map((_, i) => i), hist = [], snaps = [{ epoch: 0, E: Float32Array.from(lm.E), out: Float32Array.from(lm.Wout) }]; // the tables after every epoch, for the word paths of the page's map
let best = null; // the epoch with the lowest held-out loss, and its weights when --keep-best asks for them
// the documents the model continues after every epoch, each by its title line: the first held-out poem given whole, and
// the first training poem (what comes back of the second word for word is memory; nothing of the first can)
const titled = d => d.title && d.part === 1, sampleDocs = [...new Set([test.find(titled) || test[0], train.find(titled) || train[0]].filter(Boolean))];
const detok = tokens => T.detokenize(T.decode(vocab, tokens).filter(w => w !== T.START && w !== T.END));
const write = (prefix, max, temperature) => { const st = lm.genState(prefix), out = []; while (out.length < max) { const { token } = lm.genNext(st, { temperature, rng }); if (token === vocab.end) break; out.push(token); } return out; }; // the tokens written after the prefix
const sampleOf = () => sampleDocs.map(d => { const prefix = T.prefixOf(d); return { id: d.id, name: d.name, split: d.split, given: detok(prefix), written: detok(write(prefix, cfg.sampleMax, 0)) }; });
const recall = d => { const prefix = T.prefixOf(d), truth = d.tokens.slice(prefix.length), out = write(prefix, truth.length, 0); let same = 0; for (let i = 0; i < truth.length && i < out.length && out[i] === truth[i]; i++) same++; let any = 0; for (let i = 0; i < truth.length && i < out.length; i++) if (out[i] === truth[i]) any++; return { run: same / truth.length, any: any / truth.length, n: truth.length }; }; // how far the model reproduces the document word for word from its title line, and how many words it gets right in place
const byToken = (ds, per) => { let s = 0, n = 0; ds.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { s += per[k][i]; n++; } }); return n ? s / n : null; };
console.log(`\n== ${LOADED ? `loaded ${load} (${LOADED.trained})` : `seed ${cfg.seed}`} · ${lm.parameterCount()} parameters · ${lm.describe()} · samples: ${sampleDocs.map(d => `“${d.name}” (${d.split})`).join(', ')}`);
for (let epoch = 1; epoch <= (LOADED ? 0 : cfg.epochs); epoch++) {
  const t0 = Date.now(); let loss = 0, nb = 0;
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  for (let b = 0; b < order.length; b += cfg.batch) { loss += lm.trainBatch(order.slice(b, b + cfg.batch).map(i => train[i]), cfg.lr, cfg.decay); nb++; }
  const ev = lm.evaluate(test), evT = cfg.ground ? lm.evaluate(train.slice(0, cfg.ground)) : null, samples = sampleOf(), text = byToken(test, ev.per);
  hist.push({ epoch, loss: +(loss / nb).toFixed(4), testLoss: +ev.loss.toFixed(4), acc: +ev.accuracy.toFixed(4), ground: evT ? +evT.accuracy.toFixed(4) : null, groundN: evT ? cfg.ground : null, evidence: null, text: text == null ? null : +text.toFixed(4), samples });
  snaps.push({ epoch, E: Float32Array.from(lm.E), out: Float32Array.from(lm.Wout) });
  const newBest = !best || ev.loss < best.loss; if (newBest) best = { epoch, loss: ev.loss, json: keepBest ? lm.toJSON({}) : null };
  if (patience && epoch - best.epoch >= patience) { console.log(`stopping after epoch ${epoch}: no lower held-out loss for ${patience} epochs (lowest ${best.loss.toFixed(3)} at epoch ${best.epoch})`); cfg.epochs = epoch; }
  if (curve || epoch === cfg.epochs || epoch % 5 === 0 || epoch === 1) console.log(`epoch ${String(epoch).padStart(3)} · train loss ${(loss / nb).toFixed(3)} · test loss ${ev.loss.toFixed(3)} · next word right ${pct(ev.accuracy)} test${evT ? `, ${pct(evT.accuracy)} training` : ''} · ${((Date.now() - t0) / 1000).toFixed(1)} s${newBest ? ' · lowest held-out loss so far' : ''}${samples.map(s => `\n   ${s.split}: “${s.given.replace(/\n/g, ' / ')}” → ${s.written.replace(/\n/g, ' / ')}`).join('')}`);
}
// early stopping: the model of the best epoch stands in for the last, with the history up to it
const kept = keepBest && best && best.json && best.epoch < hist.length ? best.epoch : hist.length;
if (keepBest && best && best.json && best.epoch < hist.length) { lm = LanguageModel.fromJSON(best.json); console.log(`kept the model of epoch ${best.epoch}, the lowest held-out loss (${best.loss.toFixed(3)}); the ${hist.length - best.epoch} epochs after it are dropped from the file`); }
if (save) {
  const ev = lm.evaluate(test), ran = hist.length, histOut = hist.slice(0, kept), snapsOut = snaps.slice(0, kept + 1);
  const out = lm.toJSON({ name: 'the text model', trained: `${kept} epochs on ${train.length} documents of ${TEXT.meta ? TEXT.meta.title : 'the text'}${kept < ran ? `, stopped where the held-out loss was lowest (${ran} epochs run)` : ''}`, source: TEXT.meta ? { title: TEXT.meta.title, author: TEXT.meta.author, year: TEXT.meta.year } : null, words: vocab.words, testLoss: +ev.loss.toFixed(4), testAccuracy: +ev.accuracy.toFixed(4), groundN: cfg.ground || null, samples: sampleDocs.map(d => d.id), epochsRun: ran, hist: histOut });
  const t = wordTrails(snapsOut, out.E, out.Wout, lm.V, lm.D, e => e % 10 === 0); // the paths, thinned to --trail-snaps snapshots (the first and the last kept) and rounded, since a large vocabulary makes them most of the file
  if (t.epochs.length > trailSnaps) { const n = t.epochs.length, keep = new Set(Array.from({ length: trailSnaps }, (_, k) => Math.round(k * (n - 1) / (trailSnaps - 1)))); const idx = t.epochs.map((_, k) => k).filter(k => keep.has(k)); t.epochs = idx.map(k => t.epochs[k]); for (const s of ['in', 'out']) t[s].xy = idx.map(k => t[s].xy[k]); }
  for (const s of ['in', 'out']) t[s].xy = t[s].xy.map(a => a.map(v => +v.toFixed(2)));
  out.trails = t;
  fs.writeFileSync(save, `window.${globalOut} = ${JSON.stringify(out)};\n`); console.log(`saved ${save} (${(fs.statSync(save).size / 1024).toFixed(0)} KB) with ${histOut.length} epochs of history and ${t.epochs.length} snapshots of the word paths`);
}
if (skipGen) process.exit(0);
// recall from the title line: training poems against held-out ones
const t1 = Date.now(), rTrain = train.filter(titled).slice(0, 12).map(d => ({ d, r: recall(d) })), rTest = test.filter(titled).slice(0, 12).map(d => ({ d, r: recall(d) }));
const mean = (xs, f) => (xs.length ? xs.reduce((s, x) => s + f(x), 0) / xs.length : null);
if (rTrain.length) console.log(`\nrecall from the title line, ${rTrain.length} training poems: ${pct(mean(rTrain, x => x.r.run))} of the words reproduced in an unbroken run, ${pct(mean(rTrain, x => x.r.any))} right in place · ${rTrain.slice(0, 4).map(x => `${x.d.name} ${pct(x.r.run)}`).join(', ')}`);
if (rTest.length) console.log(`recall from the title line, ${rTest.length} held-out poems: ${pct(mean(rTest, x => x.r.run))} in an unbroken run, ${pct(mean(rTest, x => x.r.any))} right in place · ${rTest.slice(0, 4).map(x => `${x.d.name} ${pct(x.r.run)}`).join(', ')}`);
// poems from titles of one's own, drawn at the temperature
for (const title of ['The Road Not Taken', 'A Winter Evening', 'The Old Barn']) { const prefix = T.encode(vocab, title + '\n').slice(0, -1); console.log(`\n“${title}” at temperature ${cfg.temperature}:\n${detok(write(prefix, 70, cfg.temperature)).split('\n').map(l => '   ' + l).join('\n')}`); }
console.log(`(recall and writing: ${((Date.now() - t1) / 1000).toFixed(0)} s)`);
