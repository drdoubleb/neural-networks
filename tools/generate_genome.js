#!/usr/bin/env node
/*
 * generate_genome.js — draws the synthetic genome of the DNA question: one chromosome of about 44,000 bases with the
 * kinds of structure a sequence model can learn, every base labelled with the truth of what it is, so that the page can
 * show what the model was surprised by and what it was not.
 *   intergenic   random bases at 40% GC with CpG depleted (after a C, a G comes four times less often than it would)
 *   CpG island   a GC-rich stretch (70% GC, CG pairs enriched) in front of every gene
 *   TATA box     TATAAA, 30 bases before the start of transcription
 *   5' UTR       a short untranslated stretch before the ATG
 *   exon         codons of the organism's own code, purine · any · pyrimidine (the RNY pattern of the comma-free and
 *                primordial-code hypotheses: a frame a reader can find from a few bases, where the real code's cannot be
 *                found without counting): 16 codons for 8 amino acids, the third base synonymous, codon usage strongly
 *                uneven (so that the frame shows from a few codons); ATG first, one of the
 *                real stops TAA, TAG, TGA last, both of which break the pattern; the exons of a gene split at codon
 *                boundaries
 *   intron       cuts an exon after the AG of a codon (…AG|GTAAGT, the real donor consensus) and runs GTAAGT … AT-rich
 *                body … a pyrimidine tract … CAG, the codon finishing after it: the splice sites every gene finder knows
 *   3' UTR       a short untranslated stretch ending with the polyadenylation signal AATAAA
 *   repeat       copies of one 150-base element scattered through the chromosome, each 3% mutated, about a twelfth of it
 *   microsat     short tandem repeats, (CA)n, (A)n, (GATA)n
 * The last quarter of the chromosome is held out: the model trains on windows of the first three quarters. Planted variants,
 * fifteen per kind in the held-out stretch where the stretch allows, give the Test step its zero-shot question: a
 * synonymous change, a missense change, a frameshift (one base lost mid-exon), a splice-site change, a TATA box
 * change, a change in the open intergenic sequence. Writes data/dna/genome_data.js.
 *   node tools/generate_genome.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const SEED = +(process.env.GENOME_SEED || 20261115), TARGET = +(process.env.GENOME_LENGTH || 44000), HELD = 0.25, GAP = (process.env.GENOME_GAP || '150,400').split(',').map(Number), GENE_SHARE = +(process.env.GENOME_GENES || 0.55), REPEAT_SHARE = +(process.env.GENOME_REPEATS || 0.5), ELEMENT_LEN = +(process.env.GENOME_ELEMENT || 80), MUTATION = +(process.env.GENOME_MUT || 0.03), ISLAND_GC = +(process.env.GENOME_ISLAND || 0.8), TANDEM = +(process.env.GENOME_TANDEM || 4), EXONS = (process.env.GENOME_EXONS || '3,6').split(',').map(Number), CODONS_PER = (process.env.GENOME_CODONS || '20,50').split(',').map(Number), BIAS = process.env.GENOME_BIAS || 'strong'; // GENOME_*: another draw, length, intergenic gap range, the shares of gene and repeat loci, the repeat element's length and mutation, the islands' GC, the most copies of the element at one locus, the exons per gene and codons per exon, the codon bias (strong, mild), for experiments
// ---- random numbers
let state = SEED >>> 0;
const rand = () => { state |= 0; state = (state + 0x6D2B79F5) | 0; let t = Math.imul(state ^ (state >>> 15), 1 | state); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const U = (lo, hi) => lo + (hi - lo) * rand(), RI = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1)), pick = a => a[Math.floor(rand() * a.length)];
const weighted = (items, ws) => { let r = rand() * ws.reduce((a, b) => a + b, 0); for (let i = 0; i < items.length; i++) { r -= ws[i]; if (r <= 0) return items[i]; } return items[items.length - 1]; };
// ---- the organism's code: purine · any · pyrimidine; (first, second) base = the amino acid, the third base free
const PURINES = ['A', 'G'], PYRIMIDINES = ['C', 'T'], ANY = ['A', 'C', 'G', 'T'];
const CODE = {}; { let k = 0; for (const r of PURINES) for (const n of ANY) { k++; for (const y of PYRIMIDINES) CODE[r + n + y] = `aa${k}`; } for (const st of ['TAA', 'TAG', 'TGA']) CODE[st] = '*'; }
const CODONS = Object.keys(CODE).filter(c => CODE[c] !== '*'), STOPS = ['TAA', 'TAG', 'TGA'];
// codon usage: uneven between amino acids, as in any genome, here strongly (the commonest amino acid is a codon in
// four), so that a small model can find the frame from a few codons: 3.8 bits per codon in frame, 1.3 per base,
// against 2 for a coin toss
const AA_USE = BIAS === 'mild' ? [3, 2, 1, 3, 2, 1, 3, 2] : [8, 3, 5, 2, 6, 1, 4, 3], THIRD = { C: 1, T: 1 }; // the third base is used evenly, so that a synonymous change is neutral by construction
const AA_WEIGHTS = CODONS.map(c => AA_USE[+CODE[c].slice(2) - 1] * THIRD[c[2]]);
const codonFor = () => weighted(CODONS, AA_WEIGHTS);
// ---- the kinds of sequence
const BASES = ['A', 'C', 'G', 'T'];
function randomBases(n, gc, cpgDepletion = 1) { // i.i.d. at a GC content, with G after C thinned (or, above 1, enriched) by the factor
  let out = '', prev = '';
  for (let i = 0; i < n; i++) { let ws = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]; if (prev === 'C') ws = [ws[0], ws[1], ws[2] * cpgDepletion, ws[3]]; const b = weighted(BASES, ws); out += b; prev = b; }
  return out;
}
function exonCodons(n, first, last) { // n codons: ATG first when first, a stop last when last
  let s = ''; for (let k = 0; k < n; k++) { if (first && k === 0) s += 'ATG'; else if (last && k === n - 1) s += pick(STOPS); else s += codonFor(); } return s;
}
// where the introns cut the coding sequence: after the AG of a codon, so that every exon ends …AG and the intron opens
// GTAAGT, the consensus of the real splice donor (the codon finishes after the intron, a phase-2 intron); the cuts
// aimed at even spacing, at least 45 bases apart, as many as the sequence allows
function cutPoints(cds, n) {
  const cands = []; for (let i = 3; i + 3 < cds.length - 3; i += 3) if (cds[i] === 'A' && cds[i + 1] === 'G') cands.push(i + 2);
  const cuts = []; let last = 0;
  for (let k = 1; k <= n; k++) { const target = Math.round(cds.length * k / (n + 1)), pool = cands.filter(c => c >= last + 45 && c <= cds.length - 45); if (!pool.length) break; const c = pool.reduce((best, x) => (Math.abs(x - target) < Math.abs(best - target) ? x : best)); cuts.push(c); last = c; }
  return cuts;
}
function intron() { const body = randomBases(RI(60, 160), 0.3, 0.3), tract = Array.from({ length: RI(10, 16) }, () => pick(['C', 'T', 'T', 'C', 'T'])).join(''); return 'GTAAGT' + body + tract + 'CAG'; }
const ELEMENT = randomBases(ELEMENT_LEN, 0.55, 0.6); // the repeat element, drawn once
function repeatCopy() { let s = ''; for (const b of ELEMENT) s += rand() < MUTATION ? pick(BASES.filter(x => x !== b)) : b; return s; }
function microsat() { const unit = pick(['CA', 'A', 'GATA', 'TG', 'AAT']), n = Math.max(6, Math.round(RI(20, 40) / unit.length)); return unit.repeat(n); }
// ---- the chromosome, piece by piece, every piece with its label
const pieces = []; // { seq, lab, kind, meta }
const add = (seq, lab, meta) => { pieces.push({ seq, lab, meta: meta || null }); return seq.length; };
let genes = [], repeats = [], microsats = [], geneCount = 0;
function gene(name) {
  const g = { id: name, start: null, island: null, tata: null, tss: null, utr5: null, exons: [], introns: [], utr3: null, end: null };
  const at = () => pieces.reduce((a, p) => a + p.seq.length, 0);
  g.start = at();
  g.island = [at(), at() + add(randomBases(RI(250, 350), ISLAND_GC, 1.6), 'c')]; // GC-rich, and CG pairs not depleted but enriched, as the real ones are against the rest of a vertebrate genome
  add(randomBases(RI(20, 40), 0.45, 0.25), 'i');
  g.tata = [at(), at() + add('TATAAA', 't')];
  add(randomBases(25, 0.45, 0.25), 'i');
  g.tss = at();
  g.utr5 = [at(), at() + add(randomBases(RI(30, 60), 0.5, 0.6), '5')];
  const nExons = RI(EXONS[0], EXONS[1]), cds = exonCodons(Array.from({ length: nExons }, () => RI(CODONS_PER[0], CODONS_PER[1])).reduce((a, b) => a + b, 0), true, true), cuts = cutPoints(cds, nExons - 1);
  let from = 0;
  cuts.concat([cds.length]).forEach((to, e) => {
    const s = at(); add(cds.slice(from, to), 'e'); g.exons.push([s, at()]); from = to;
    if (to < cds.length) { const s2 = at(); add(intron(), 'n'); g.introns.push([s2, at()]); }
  });
  const u3 = randomBases(RI(50, 110), 0.42, 0.4) + 'AATAAA' + randomBases(RI(10, 20), 0.42, 0.4);
  g.utr3 = [at(), at() + add(u3, '3')];
  g.end = at(); genes.push(g);
}
function intergenic(n) { // open sequence, with a microsatellite dropped in now and then
  let left = n;
  while (left > 0) {
    if (left > 120 && rand() < 0.35) { const k = RI(40, Math.min(left - 60, 400)); add(randomBases(k, 0.4, 0.25), 'i'); left -= k; const m = microsat(); const at = pieces.reduce((a, p) => a + p.seq.length, 0); add(m, 'm'); microsats.push([at, at + m.length, m.slice(0, 4)]); left -= m.length; }
    else { add(randomBases(left, 0.4, 0.25), 'i'); left = 0; }
  }
}
function repeatElement() { const n = RI(1, TANDEM); for (let k = 0; k < n; k++) { const at = pieces.reduce((a, p) => a + p.seq.length, 0), s = repeatCopy(); add(s, 'r'); repeats.push([at, at + s.length]); } } // a locus holds one to a few copies back to back, as the real families' insertions cluster
// the layout: genes and repeats in a loose alternation, intergenic sequence between
let total = 0;
while (total < TARGET) {
  intergenic(RI(GAP[0], GAP[1]));
  const r = rand();
  if (r < GENE_SHARE) gene(`G${String(++geneCount).padStart(2, '0')}`); else if (r < GENE_SHARE + REPEAT_SHARE) repeatElement(); else { intergenic(RI(100, 300)); }
  total = pieces.reduce((a, p) => a + p.seq.length, 0);
}
intergenic(RI(200, 400));
const seq = pieces.map(p => p.seq).join(''), lab = pieces.map(p => p.lab.repeat(p.seq.length)).join('');
const N = seq.length, trainEnd = Math.round(N * (1 - HELD));
// ---- the planted variants, in the held-out stretch where it allows, else anywhere
const variants = [];
const codonAt = pos => { // the codon a base belongs to, within an exon: its start, and the base's offset in it
  for (const g of genes) { let frame = 0; for (const [s, e] of g.exons) { if (pos >= s && pos < e) { const off = (pos - s + frame) % 3, cs = pos - off; return { gene: g.id, start: cs, offset: off }; } frame = (frame + (e - s)) % 3; } }
  return null;
};
const inHeld = pos => pos >= trainEnd;
function plant(cls, candidates, n, note) {
  const held = candidates.filter(c => inHeld(c.pos)), rest = candidates.filter(c => !inHeld(c.pos)), chosen = [];
  for (const pool of [held, rest]) { const idx = pool.map((_, i) => i); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } for (const i of idx) { if (chosen.length >= n) break; chosen.push(pool[i]); } }
  for (const c of chosen) variants.push(Object.assign({ cls, note: typeof note === 'function' ? note(c) : note, held: inHeld(c.pos) }, c));
}
const cdsSites = []; for (let p = 0; p < N; p++) if (lab[p] === 'e') cdsSites.push(p);
const syn = [], mis = [], fsh = [];
for (const p of cdsSites) {
  const c = codonAt(p); if (!c) continue; const codon = seq.slice(c.start, c.start + 3), aa = CODE[codon]; if (!aa || aa === '*' || codon === 'ATG') continue;
  for (const alt of BASES) { if (alt === seq[p]) continue; const mut = codon.slice(0, c.offset) + alt + codon.slice(c.offset + 1), aa2 = CODE[mut], v = { pos: p, ref: seq[p], alt, codon, mutCodon: mut, aa, aa2, gene: c.gene };
    if (!aa2 || aa2 === '*') continue; // a change that leaves the code (no such codon) is not planted: the organism's reader would stall there, and the lesson is about changes within it
    if (aa2 === aa) syn.push(v); else mis.push(v); }
  if (c.offset === 1) fsh.push({ pos: p, ref: seq[p], alt: '', codon, gene: c.gene }); // the middle base lost: every codon after it reads out of frame
}
plant('synonymous', syn, 15, v => `${v.codon} → ${v.mutCodon}: the same amino acid, the third base is free`);
plant('missense', mis, 15, v => `${v.codon} → ${v.mutCodon}: amino acid ${v.aa.slice(2)} → ${v.aa2.slice(2)}`);
plant('frameshift', fsh, 15, v => `one base of ${v.codon} lost: the frame shifts for the rest of the gene`);
const splice = []; for (const g of genes) for (const [s, e] of g.introns) { splice.push({ pos: s, ref: seq[s], alt: 'C', gene: g.id, site: 'donor GT' }, { pos: s + 1, ref: seq[s + 1], alt: 'A', gene: g.id, site: 'donor GT' }, { pos: e - 2, ref: seq[e - 2], alt: 'T', gene: g.id, site: 'acceptor AG' }, { pos: e - 1, ref: seq[e - 1], alt: 'C', gene: g.id, site: 'acceptor AG' }); }
plant('splice', splice, 15, v => `${v.site} broken`);
const tata = []; for (const g of genes) for (let k = 0; k < 6; k++) { const p = g.tata[0] + k; tata.push({ pos: p, ref: seq[p], alt: seq[p] === 'A' ? 'C' : 'G', gene: g.id }); }
plant('tata', tata, 15, 'TATA box broken');
const inter = []; for (let p = 0; p < N; p++) if (lab[p] === 'i' && p % 7 === 0) inter.push({ pos: p, ref: seq[p], alt: pick(BASES.filter(b => b !== seq[p])) });
plant('intergenic', inter, 15, 'open sequence');
variants.sort((a, b) => a.pos - b.pos);
// ---- out
const KINDS = { i: 'intergenic', c: 'CpG island', t: 'TATA box', 5: "5' UTR", e: 'exon', n: 'intron', 3: "3' UTR", r: 'repeat', m: 'microsatellite' };
const gcOf = s => { let g = 0; for (const b of s) if (b === 'C' || b === 'G') g++; return g / s.length; };
const counts = {}; for (const ch of lab) counts[ch] = (counts[ch] || 0) + 1;
const meta = { seed: SEED, length: N, trainEnd, held: N - trainEnd, gc: +gcOf(seq).toFixed(3), kinds: KINDS, counts, genes: genes.length, repeats: repeats.length, microsats: microsats.length, element: ELEMENT, code: CODE,
  codonUse: Object.fromEntries(CODONS.map((c, i) => [c, AA_WEIGHTS[i]])),
  rules: ['intergenic: 40% GC, CpG depleted four-fold', `CpG island: 250–350 bases at ${Math.round(ISLAND_GC * 100)}% GC, CG pairs enriched, before every gene`, 'TATA box 30 bases before the start of transcription', 'exons: codons purine · any · pyrimidine, 16 of them for 8 amino acids, the third base synonymous; the amino acids used unevenly, the commonest a codon in four; ATG first, TAA, TAG or TGA last', 'introns cut an exon after the AG of a codon and run GTAAGT … AT-rich … pyrimidine tract … CAG, the codon finishing after them', "3' UTR ends with AATAAA", `repeat: one ${ELEMENT_LEN}-base element, copies ${Math.round(MUTATION * 100)}% mutated`, 'microsatellites: (CA)n, (A)n, (GATA)n, (TG)n, (AAT)n'] };
const out = { meta, seq, lab, genes, repeats, microsats, variants };
const file = process.env.GENOME_OUT || path.join(__dirname, '..', 'data', 'dna', 'genome_data.js');
fs.writeFileSync(file, `window.LECTURE_GENOME = ${JSON.stringify(out)};\n`);
console.log(`${N} bases (${trainEnd} train, ${N - trainEnd} held out) · GC ${meta.gc} · ${genes.length} genes (${genes.filter(g => g.start >= trainEnd).length} in the held-out stretch) · ${repeats.length} repeat copies · ${microsats.length} microsatellites`);
console.log('bases by kind: ' + Object.entries(counts).map(([k, v]) => `${KINDS[k]} ${v}`).join(' · '));
console.log('variants: ' + Object.entries(variants.reduce((a, v) => (a[v.cls] = (a[v.cls] || 0) + 1, a), {})).map(([k, v]) => `${k} ${v} (${variants.filter(x => x.cls === k && x.held).length} held out)`).join(' · '));
console.log(`wrote ${file} (${(fs.statSync(file).size / 1024).toFixed(0)} kB)`);
