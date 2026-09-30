/*
 * reports.js — the reports of the language-model question as tokens: a word-level tokenizer (words, numbers,
 * punctuation and the newline each a token), a vocabulary built from the training reports with three special tokens
 * (start, end, unknown), encoding and decoding, the text put back together from tokens, and which section of a
 * report every token belongs to (the findings block, then specimen, clinical, gross, microscopic, diagnosis).
 * Works in the browser (window.NucleusReports) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NucleusReports = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const RE = /\n|[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?|[^\sA-Za-z\d]/g;
  const START = '⟨s⟩', END = '⟨/s⟩', UNK = '⟨?⟩', NL = '\n';
  const SECTIONS = ['findings', 'specimen', 'clinical', 'gross', 'microscopic', 'diagnosis'];
  const HEADERS = { FINDINGS: 'findings', SPECIMEN: 'specimen', CLINICAL: 'clinical', GROSS: 'gross', MICROSCOPIC: 'microscopic', DIAGNOSIS: 'diagnosis' };
  const FINDING_LINES = { surface: 'surface', nests: 'nests', atypia: 'atypia in nests', nest: 'contours', stromal: 'stromal reaction', muscularis: 'muscularis propria', inflammation: 'inflammation' }; // a block line by its first word
  function tokenize(text) { return text.match(RE) || []; }
  // the vocabulary: the specials first, then every token of the texts, the most frequent first
  function buildVocab(texts) {
    const counts = new Map();
    for (const t of texts) for (const w of tokenize(t)) counts.set(w, (counts.get(w) || 0) + 1);
    const words = [START, END, UNK].concat([...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(e => e[0]));
    const index = new Map(words.map((w, i) => [w, i]));
    return { words, index, counts, start: 0, end: 1, unk: 2, size: words.length };
  }
  function encode(vocab, text) { return [vocab.start].concat(tokenize(text).map(w => (vocab.index.has(w) ? vocab.index.get(w) : vocab.unk)), [vocab.end]); }
  function decode(vocab, ids) { return ids.map(i => vocab.words[i]); }
  // the text of a run of tokens: a space between words, none before punctuation, the newline as itself, the specials dropped
  function detokenize(words) {
    let out = '';
    for (const w of words) {
      if (w === START || w === END) continue;
      if (w === NL) { out += NL; continue; }
      if (out && !out.endsWith(NL) && !/^[,.:;)]$/.test(w) && !out.endsWith('(')) out += ' ';
      out += w;
    }
    return out;
  }
  // the section of every token (a header token opens its section; the start token counts with the findings, the
  // end token with the diagnosis) and, inside the findings block, which line the token is on
  function sectionsOf(words) {
    let sec = 'findings', line = null, atLineStart = true; const secs = [], lines = [];
    for (const w of words) {
      if (HEADERS[w]) { sec = HEADERS[w]; line = null; }
      if (sec === 'findings') { if (atLineStart && FINDING_LINES[w]) line = FINDING_LINES[w]; atLineStart = w === NL; if (w === NL) line = null; }
      secs.push(sec); lines.push(sec === 'findings' ? line : null);
    }
    return { sections: secs, lines };
  }
  // the diagnosis sentence of a text or a run of words: what follows "DIAGNOSIS:" up to the first full stop
  function diagnosisOf(words) {
    const i = words.indexOf('DIAGNOSIS'); if (i < 0) return null;
    const out = []; for (let k = i + 1; k < words.length; k++) { const w = words[k]; if (w === ':' && !out.length) continue; if (w === END || w === NL) break; out.push(w); if (w === '.') break; }
    return out.length ? detokenize(out) : null;
  }
  // the whole diagnosis section, to the end
  function diagnosisSectionOf(words) {
    const i = words.indexOf('DIAGNOSIS'); if (i < 0) return null;
    const out = []; for (let k = i + 1; k < words.length; k++) { const w = words[k]; if (w === ':' && !out.length) continue; if (w === END || w === NL) break; out.push(w); }
    return out.length ? detokenize(out) : null;
  }
  // a diagnosis sentence's class, for tallies
  function classOf(dx) {
    if (!dx) return 'none';
    if (/suspicious/.test(dx)) return 'suspicious';
    if (/invasive into muscularis/.test(dx)) return 'invasive-mp';
    if (/invasive/.test(dx)) return 'invasive';
    if (/in situ/.test(dx)) return 'cis';
    if (/reactive/.test(dx)) return 'reactive';
    if (/^Denuded/.test(dx)) return 'denuded';
    if (/^Benign/.test(dx)) return 'benign';
    return 'other';
  }
  return { tokenize, buildVocab, encode, decode, detokenize, sectionsOf, diagnosisOf, diagnosisSectionOf, classOf, START, END, UNK, NL, SECTIONS, HEADERS };
});
