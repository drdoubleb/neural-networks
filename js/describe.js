/*
 * describe.js — the Describe question (from pixels to words): a nucleus's description as text, and the text as
 * tokens. Every description names the four traits the atypia generator can give a nucleus (enlarged, hyperchromatic,
 * irregular in outline, coarse in chromatin), each in one of a few phrasings and in one of a few sentence shapes, and
 * ends with an impression (bland, or atypical when any trait is on). The vocabulary has four special tokens: start,
 * end, unknown and the image, which stands first in every document and carries the nucleus's code instead of a word.
 * Works in the browser (window.NucleusDescribe) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./reports.js'));
  else root.NucleusDescribe = factory(root.NucleusReports);
})(typeof self !== 'undefined' ? self : this, function (R) {
  'use strict';
  const START = '⟨s⟩', END = '⟨/s⟩', UNK = '⟨?⟩', IMG = '⟨img⟩';
  const TRAITS = ['enlarged', 'dark', 'irregular', 'coarse'];
  const TRAIT_NAMES = { enlarged: 'enlarged', dark: 'hyperchromatic', irregular: 'irregular outline', coarse: 'coarse chromatin' };
  // the words a description uses for each trait, on and off; a phrase may be more than one token
  const WORDS = {
    enlarged: { yes: ['enlarged', 'large', 'big'], no: ['normal-sized', 'small', 'modest'] },
    dark: { yes: ['hyperchromatic', 'dark', 'densely stained'], no: ['pale', 'lightly stained', 'faint'] },
    irregular: { yes: ['irregular', 'notched', 'lobulated', 'angulated'], no: ['smooth', 'regular', 'rounded'] },
    coarse: { yes: ['coarse', 'clumped', 'coarsely granular'], no: ['fine', 'finely granular', 'even'] },
  };
  // the sentence shapes; w holds the phrase chosen for each trait
  const an = w => (/^[aeiou]/i.test(w) ? 'an' : 'a'), An = w => (/^[aeiou]/i.test(w) ? 'An' : 'A'); // the article before a phrase
  const TEMPLATES = [
    w => `${An(w.enlarged)} ${w.enlarged}, ${w.dark} nucleus with ${an(w.irregular)} ${w.irregular} contour and ${w.coarse} chromatin.`,
    w => `Nucleus ${w.enlarged} and ${w.dark}; the outline is ${w.irregular} and the chromatin ${w.coarse}.`,
    w => `The nucleus is ${w.enlarged} and ${w.dark}, with ${w.coarse} chromatin and ${an(w.irregular)} ${w.irregular} outline.`,
    w => `Outline ${w.irregular}, chromatin ${w.coarse}; the nucleus is ${w.enlarged} and ${w.dark}.`,
  ];
  const SECTIONS = ['image', 'description', 'impression'];
  const isAtypical = traits => TRAITS.some(t => !!traits[t]);
  function subtypeOf(traits) { const on = TRAITS.filter(t => traits[t]); return on.length === 0 ? 'bland' : on.length === 1 ? on[0] : 'combined'; }
  // a description of a nucleus from its traits; pick(n) chooses the phrasing (an index below n)
  function describe(traits, pick) {
    const w = {}; for (const t of TRAITS) { const list = traits[t] ? WORDS[t].yes : WORDS[t].no; w[t] = list[pick(list.length)]; }
    return `${TEMPLATES[pick(TEMPLATES.length)](w)} Impression: ${isAtypical(traits) ? 'atypical' : 'bland'}.`;
  }
  // the vocabulary: the four specials, then every word of the training descriptions by frequency
  function buildVocab(texts) {
    const count = new Map(); for (const t of texts) for (const w of R.tokenize(t)) count.set(w, (count.get(w) || 0) + 1);
    const words = [START, END, UNK, IMG].concat([...count.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(e => e[0]));
    return { words, index: new Map(words.map((w, i) => [w, i])), start: 0, end: 1, unk: 2, img: 3, size: words.length };
  }
  function vocabFrom(words) { return { words, index: new Map(words.map((w, i) => [w, i])), start: 0, end: 1, unk: 2, img: 3, size: words.length }; }
  // a document: the image token first (its code goes in beside the tokens), the description, the end
  function encode(vocab, text) { return [vocab.img].concat(R.tokenize(text).map(w => (vocab.index.has(w) ? vocab.index.get(w) : vocab.unk)), [vocab.end]); }
  const decode = (vocab, ids) => ids.map(i => vocab.words[i]);
  // the section of every token: the image, the description, the impression (from the word Impression to the end)
  function sectionsOf(words) { let sec = 'description'; return words.map((w, i) => { if (i === 0) return 'image'; if (w === 'Impression') sec = 'impression'; return sec; }); }
  // a phrase's tokens occur in the words, in order
  function has(words, phrase) { const p = R.tokenize(phrase); outer: for (let i = 0; i + p.length <= words.length; i++) { for (let k = 0; k < p.length; k++) if (words[i + k] !== p[k]) continue outer; return true; } return false; }
  // what a written description says about each trait: true, false, or null when it says nothing or contradicts itself
  function traitsOf(words) { const t = {}; for (const k of TRAITS) { const yes = WORDS[k].yes.some(p => has(words, p)), no = WORDS[k].no.some(p => has(words, p)); t[k] = yes && !no ? true : no && !yes ? false : null; } return t; }
  function impressionOf(words) { const i = words.indexOf('Impression'); if (i < 0) return null; const w = words[i + 1] === ':' ? words[i + 2] : words[i + 1]; return w === 'bland' || w === 'atypical' ? w : null; }
  // the description's traits against the truth: which were right, and whether all four were
  function traitsRight(truth, words) { const said = traitsOf(words), right = {}; let all = true; for (const k of TRAITS) { right[k] = said[k] === !!truth[k]; if (!right[k]) all = false; } return { said, right, all, impression: impressionOf(words), impressionRight: impressionOf(words) === (isAtypical(truth) ? 'atypical' : 'bland') }; }
  // the share of the description tokens' attention (the last layer, heads averaged) that lands on the image token
  function imageShare(fw, sections) { const L = fw.ctxs.length; if (!L) return null; const c = fw.ctxs[L - 1]; let s = 0, n = 0; for (let i = 1; i < fw.tokens.length - 1; i++) { if (sections && sections[i] === 'image') continue; let a = 0; for (let h = 0; h < c.Ah.length; h++) a += c.Ah[h][i][0] / c.Ah.length; s += a; n++; } return n ? s / n : null; }
  return { START, END, UNK, IMG, TRAITS, TRAIT_NAMES, WORDS, TEMPLATES, SECTIONS, isAtypical, subtypeOf, describe, buildVocab, vocabFrom, encode, decode, detokenize: R.detokenize, tokenize: R.tokenize, sectionsOf, has, traitsOf, impressionOf, traitsRight, imageShare };
});
