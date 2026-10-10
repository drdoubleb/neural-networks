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
  // the findings block as a form: the seven lines, their keys and their states, in the order of the block
  const FINDINGS = [
    { key: 'surface', line: 'surface urothelium', states: ['normal', 'reactive', 'atypia', 'denuded'] },
    { key: 'nests', line: 'nests below basement membrane', states: ['absent', 'present'] },
    { key: 'atypia', line: 'atypia in nests', states: ['none', 'absent', 'present'] },
    { key: 'contours', line: 'nest contours', states: ['none', 'rounded', 'irregular'] },
    { key: 'stromal', line: 'stromal reaction', states: ['none', 'desmoplasia'] },
    { key: 'mp', line: 'muscularis propria', states: ['absent', 'present', 'involved'] },
    { key: 'inflammation', line: 'inflammation', states: ['none', 'mild', 'marked'] },
  ];
  // the block's text from a findings object { surface, nests, atypia, contours, stromal, mp, inflammation }
  function blockOf(f) { return 'FINDINGS\n' + FINDINGS.map(l => `${l.line}: ${f[l.key]}`).join('\n'); }
  // the findings read back from a block's tokens (the first 40 of a report), or null
  function findingsOf(words) {
    const f = {}; let i = 0;
    for (const l of FINDINGS) { const at = words.indexOf(l.line.split(' ')[0], i); if (at < 0) return null; const colon = words.indexOf(':', at); if (colon < 0) return null; f[l.key] = words[colon + 1]; i = colon + 1; if (!l.states.includes(f[l.key])) return null; }
    return f;
  }
  // the rule the generator followed, for any findings: the diagnosis sentence and the muscularis propria line
  function ruleOf(f) {
    const at = f.nests === 'present' && f.atypia === 'present', inv = at && f.contours === 'irregular' && f.stromal === 'desmoplasia', cisn = at && f.contours === 'rounded' && f.stromal === 'none', disc = at && !inv && !cisn;
    let dx;
    if (disc) dx = 'Urothelial carcinoma in situ with foci suspicious for invasion.';
    else if (inv) dx = `Urothelial carcinoma, invasive into ${f.mp === 'involved' ? 'muscularis propria' : 'lamina propria'}${f.surface === 'atypia' ? ', with associated carcinoma in situ' : ''}.`;
    else if (cisn || f.surface === 'atypia') dx = 'Urothelial carcinoma in situ.';
    else if (f.surface === 'reactive') dx = 'Benign urothelium with reactive changes.';
    else if (f.surface === 'denuded') dx = 'Denuded urothelium, no diagnostic abnormality in the material present.';
    else dx = 'Benign urothelium.';
    const mp = f.mp === 'absent' ? 'Muscularis propria not identified.' : f.mp === 'present' ? 'Muscularis propria present, not involved.' : 'Muscularis propria present and involved.';
    return { dx, mp, text: `${dx} ${mp}` };
  }
  const CLASS_NAMES = { benign: 'benign', reactive: 'reactive', denuded: 'denuded', cis: 'CIS', suspicious: 'suspicious', invasive: 'invasive', 'invasive-mp': 'invasive, MP', other: 'other', none: 'none' };
  // a byte-pair tokenizer, the way a real model's tokenizer is built, for the page's demonstration: start from single
  // letters with a mark for the end of a word, merge the most frequent adjacent pair over the training words, repeat
  // n times; a word, seen or not, is then written in the pieces the merges made
  const BPE_EOW = '·';
  function bpeLearn(texts, n) {
    const freq = new Map(); for (const t of texts) for (const w of tokenize(t)) if (/^[A-Za-z]/.test(w)) freq.set(w, (freq.get(w) || 0) + 1);
    const words = [...freq.entries()].map(([w, f]) => ({ syms: [...w, BPE_EOW], f })), merges = [];
    for (let m = 0; m < n; m++) {
      const pairs = new Map(); for (const { syms, f } of words) for (let i = 0; i + 1 < syms.length; i++) { const k = syms[i] + '\u0000' + syms[i + 1]; pairs.set(k, (pairs.get(k) || 0) + f); }
      let best = null, bf = 0; for (const [k, f] of pairs) if (f > bf) { bf = f; best = k; }
      if (!best) break;
      const [a, b] = best.split('\u0000'); merges.push([a, b]);
      for (const w of words) { const s = w.syms; for (let i = 0; i + 1 < s.length; i++) if (s[i] === a && s[i + 1] === b) s.splice(i, 2, a + b); }
    }
    return { merges, freq };
  }
  function bpeEncode(word, bpe) { const s = [...word, BPE_EOW]; for (const [a, b] of bpe.merges) for (let i = 0; i + 1 < s.length; i++) if (s[i] === a && s[i + 1] === b) s.splice(i, 2, a + b); return s; }
  function bpeVocab(bpe) { const count = new Map(); for (const [w, f] of bpe.freq) for (const p of bpeEncode(w, bpe)) count.set(p, (count.get(p) || 0) + f); return [...count.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)); } // the pieces in use over the training words, most frequent first
  return { tokenize, buildVocab, encode, decode, detokenize, sectionsOf, diagnosisOf, diagnosisSectionOf, classOf, blockOf, findingsOf, ruleOf, bpeLearn, bpeEncode, bpeVocab, BPE_EOW, FINDINGS, CLASS_NAMES, START, END, UNK, NL, SECTIONS, HEADERS };
});
