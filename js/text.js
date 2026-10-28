/*
 * text.js — the Your-text question: any text as documents for the next-word model. The tokenizer is the reports'
 * (words, numbers, punctuation and the line break as tokens). A text is cut into documents of at most a window of
 * tokens: a poem per document when the text marks its titles the way Project Gutenberg does (_Title_ on a line of
 * its own), else a paragraph per document, long ones cut at line ends; one document in ten is held out as the test
 * set. Works in the browser (window.NucleusText) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./reports.js'));
  else root.NucleusText = factory(root.NucleusReports);
})(typeof self !== 'undefined' ? self : this, function (R) {
  'use strict';
  const SECTIONS = ['text'];
  const TITLE_RE = /^_([^_]+)_$/;
  // the text as the tokenizer wants it: one kind of line break, straight quotes, plain letters, no indentation, at
  // most one blank line in a row (the italics marks are kept here: a title is recognised by them, and dropped after)
  function normalize(text) {
    return String(text || '').replace(/\r\n?/g, '\n').replace(/[‘’‚‛]/g, "'").replace(/[“”„‟]/g, '"').replace(/ /g, ' ').replace(/[❦❧]/g, '')
      .replace(/[à-ä]/g, 'a').replace(/[è-ë]/g, 'e').replace(/[ì-ï]/g, 'i').replace(/[ò-ö]/g, 'o').replace(/[ù-ü]/g, 'u').replace(/æ/g, 'ae').replace(/[À-Ä]/g, 'A').replace(/[È-Ë]/g, 'E').replace(/[Ò-Ö]/g, 'O').replace(/ç/g, 'c')
      .split('\n').map(l => l.replace(/\s+$/, '').replace(/^\s+/, '')).join('\n').replace(/\n{3,}/g, '\n\n').replace(/^\n+|\n+$/g, '');
  }
  const countTokens = text => R.tokenize(text).length, unmark = text => text.replace(/_([^_\n]+)_/g, '$1').replace(/_/g, ''); // italics marks inside a line are dropped
  // a long document cut into windows of at most `max` tokens (with the start and end tokens), at line ends
  function windows(title, body, max) {
    const lines = body.split('\n'), out = []; let cur = [], n = 0, part = 0;
    const flush = () => { if (!cur.length) return; part++; out.push({ title, text: (title && part === 1 ? title + '\n' : '') + cur.join('\n'), part }); cur = []; n = 0; };
    const budget = max - 2 - (title ? countTokens(title) + 1 : 0);
    for (const line of lines) { const k = countTokens(line) + 1; if (cur.length && n + k > budget) flush(); cur.push(line); n += k; }
    flush(); return out;
  }
  // the documents of a text: poems by their _Title_ lines when the text has them, paragraphs otherwise
  function documentsOf(raw, { maxTokens = 200 } = {}) {
    const lines = normalize(raw).split('\n'), titled = lines.some(l => TITLE_RE.test(l));
    const docs = [], push = (title, body) => { const b = unmark(body).replace(/^\n+|\n+$/g, ''); if (!b && !title) return; for (const w of windows(title, b || title, maxTokens)) docs.push(w); };
    if (titled) { let title = null, body = []; for (const l of lines) { const m = l.match(TITLE_RE); if (m) { if (title != null || body.join('').trim()) push(title, body.join('\n')); title = unmark(m[1].trim()); body = []; } else if (title != null || l.trim()) body.push(l); } push(title, body.join('\n')); }
    else { let body = []; const flush = () => { if (body.join('').trim()) push(null, body.join('\n')); body = []; }; for (const l of lines) { if (!l.trim()) flush(); else body.push(l); } flush(); }
    // an all-caps line on its own without a title (a book's heading) is not a document
    const kept = docs.filter(d => !(d.title == null && /^[A-Z][A-Z'" ,\-.!?]{3,}$/.test(d.text.trim())));
    return kept.map((d, i) => Object.assign({ id: `T${String(i + 1).padStart(3, '0')}`, name: d.title ? (d.part > 1 ? `${d.title} (${d.part})` : d.title) : `¶ ${i + 1}` }, d));
  }
  // one document in ten is the test set, the tenth of each ten; a short text keeps its last document for the test
  function splits(docs) {
    if (docs.length < 2) return { train: docs.slice(), test: docs.slice(), held: [] };
    const test = docs.filter((_, i) => (docs.length >= 10 ? i % 10 === 9 : i === docs.length - 1));
    return { train: docs.filter(d => !test.includes(d)), test, held: [] };
  }
  // the vocabulary is the whole text's, held-out documents included, so that a held-out poem can be given to the model
  // by its title line; a word that appears only in held-out documents keeps the random row it started with, since no
  // training document ever moves it, and the test loss counts that honestly
  const vocabOf = docs => R.buildVocab(docs.map(d => d.text));
  const sectionsOf = words => words.map(() => 'text');
  // the prefix the model continues from after every epoch and on the Test step: the title line of a poem, else the
  // first few words. d: a prepared document with tokens and words
  function prefixOf(d) {
    if (d.title) { const k = d.words.indexOf('\n'); if (k > 0) return d.tokens.slice(0, k + 1); }
    return d.tokens.slice(0, Math.min(d.tokens.length - 1, 8));
  }
  function stats(text) { const docs = documentsOf(text), sp = splits(docs), toks = [].concat(...docs.map(d => R.tokenize(d.text))); return { documents: docs.length, train: sp.train.length, test: sp.test.length, tokens: toks.length, words: toks.filter(w => /^[A-Za-z]/.test(w)).length, lines: toks.filter(w => w === '\n').length, vocabulary: new Set(toks).size, titled: docs.some(d => d.title) }; }
  return { SECTIONS, normalize, documentsOf, splits, vocabOf, sectionsOf, prefixOf, stats, buildVocab: R.buildVocab, encode: R.encode, decode: R.decode, detokenize: R.detokenize, tokenize: R.tokenize, START: R.START, END: R.END, UNK: R.UNK, NL: R.NL };
});
