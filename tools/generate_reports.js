#!/usr/bin/env node
/*
 * generate_reports.js — the corpus of the small language model: synthetic bladder biopsy reports, each written from
 * a hidden case. A case is a handful of findings (the surface urothelium: normal, reactive atypia, atypia or denuded;
 * nests below the basement membrane: absent, benign or atypical, and for atypical nests their contours and whether
 * there is desmoplasia; muscularis propria: not identified, present or involved; inflammation, which bears on
 * nothing), a diagnosis that follows from the findings by a fixed rule, and cosmetic details (site, procedure,
 * clinical history, the gross). Each document is the findings block, as an image analyser might report it, followed
 * by the report proper: specimen, clinical, gross, microscopic and, last, the diagnosis. The microscopic section says
 * only what the findings block holds, in one of several phrasings, so that a model writing the report from the block
 * never has to invent a fact. The rule and the base rates are the lecture's: a benign majority, a small share of
 * discordant cases signed out as "suspicious for invasion", and one real combination, invasion under a normal surface,
 * kept out of training altogether to ask whether the model learned the findings or the templates.
 *   node tools/generate_reports.js
 * writes data/reports/reports_data.js (train, test and held-out reports with their hidden cases) and
 * data/reports/sample_reports.md (a sheet of reports of every class, for vetting).
 */
'use strict';
const fs = require('fs');
const path = require('path');

const SEED = 20260930, N_TRAIN = 800, N_TEST = 100, N_HELD = 8;
const OUT = path.join(__dirname, '..', 'data', 'reports');

function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
let rng = mulberry32(SEED);
const pick = arr => arr[Math.floor(rng() * arr.length)];
const chance = p => rng() < p;
const weighted = pairs => { let r = rng(); for (const [v, p] of pairs) { if ((r -= p) < 0) return v; } return pairs[pairs.length - 1][0]; };

// ---- the classes of the sign-out queue, with their base rates; each fixes the surface and the nests
// surface: normal | reactive | atypia | denuded · nests: absent | benign | atypical (+ contours rounded | irregular, desmoplasia)
const CLASSES = [
  { key: 'benign', share: 0.33, name: 'Benign urothelium', draw: () => ({ surface: 'normal', nests: chance(0.4) ? 'benign' : 'absent' }) },
  { key: 'reactive', share: 0.12, name: 'Benign urothelium with reactive changes', draw: () => ({ surface: 'reactive', nests: chance(0.3) ? 'benign' : 'absent' }) },
  { key: 'denuded', share: 0.05, name: 'Denuded urothelium', draw: () => ({ surface: 'denuded', nests: chance(0.3) ? 'benign' : 'absent' }) },
  { key: 'cis', share: 0.15, name: 'Carcinoma in situ', draw: () => ({ surface: 'atypia', nests: chance(0.3) ? 'benign' : 'absent' }) },
  { key: 'cisnests', share: 0.10, name: 'Carcinoma in situ involving von Brunn nests', draw: () => ({ surface: chance(0.85) ? 'atypia' : 'denuded', nests: 'atypical', contours: 'rounded', desmoplasia: false }) },
  { key: 'invasive', share: 0.20, name: 'Invasive urothelial carcinoma', draw: () => ({ surface: chance(0.8) ? 'atypia' : 'denuded', nests: 'atypical', contours: 'irregular', desmoplasia: true }) },
  { key: 'suspicious', share: 0.05, name: 'Carcinoma in situ, suspicious for invasion', draw: () => ({ surface: 'atypia', nests: 'atypical', contours: chance(0.5) ? 'rounded' : 'irregular', desmoplasia: null }) }, // desmoplasia set discordant below
];
const HELD = { key: 'invasive-quiet-surface', name: 'Invasive urothelial carcinoma under a normal surface', draw: () => ({ surface: 'normal', nests: 'atypical', contours: 'irregular', desmoplasia: true }) }; // never trained on

function makeCase(cls) {
  const c = Object.assign({ cls: cls.key }, cls.draw());
  if (c.desmoplasia === null) c.desmoplasia = c.contours === 'rounded'; // discordant: rounded with desmoplasia, or irregular without
  if (c.nests !== 'atypical') { c.contours = c.nests === 'benign' ? 'rounded' : null; c.desmoplasia = c.nests === 'benign' ? false : null; }
  const invasive = c.nests === 'atypical' && c.contours === 'irregular' && c.desmoplasia;
  c.mp = invasive ? weighted([['ni', 0.4], ['present', 0.45], ['involved', 0.15]]) : weighted([['ni', 0.5], ['present', 0.5]]);
  c.inflammation = weighted([['none', 0.4], ['mild', 0.4], ['marked', 0.2]]); // bears on nothing
  c.site = pick(['dome', 'trigone', 'right lateral wall', 'left lateral wall', 'posterior wall', 'anterior wall', 'bladder neck']);
  c.procedure = pick(['cold cup biopsy', 'cold cup biopsy', 'biopsy']);
  c.clinical = pick(['Hematuria.', 'Gross hematuria.', 'Microscopic hematuria.', 'Surveillance, history of urothelial carcinoma.', 'Surveillance following intravesical therapy.', 'Abnormal urine cytology.', 'Erythematous mucosa at cystoscopy.', 'Irritative voiding symptoms.']);
  c.fragments = 1 + Math.floor(rng() * 6);
  const a = 1 + Math.floor(rng() * 4), b = a + 1 + Math.floor(rng() * 4); c.sizes = c.fragments === 1 ? [b] : [a, b]; // tenths of a cm
  c.cassettes = c.fragments >= 5 && chance(0.3) ? 2 : 1;
  return c;
}

// ---- the diagnosis, a fixed rule on the findings
function diagnose(c) {
  const at = c.nests === 'atypical', inv = at && c.contours === 'irregular' && c.desmoplasia, cisn = at && c.contours === 'rounded' && !c.desmoplasia, disc = at && !inv && !cisn;
  let dx;
  if (disc) dx = 'Urothelial carcinoma in situ with foci suspicious for invasion.';
  else if (inv) dx = `Urothelial carcinoma, invasive into ${c.mp === 'involved' ? 'muscularis propria' : 'lamina propria'}${c.surface === 'atypia' ? ', with associated carcinoma in situ' : ''}.`;
  else if (cisn || c.surface === 'atypia') dx = 'Urothelial carcinoma in situ.';
  else if (c.surface === 'reactive') dx = 'Benign urothelium with reactive changes.';
  else if (c.surface === 'denuded') dx = 'Denuded urothelium, no diagnostic abnormality in the material present.';
  else dx = 'Benign urothelium.';
  const mp = c.mp === 'ni' ? 'Muscularis propria not identified.' : c.mp === 'present' ? 'Muscularis propria present, not involved.' : 'Muscularis propria present and involved.';
  return { dx, mp, text: `${dx} ${mp}` };
}

// ---- the findings block: what an image analyser would report, one line per finding, every line always there with a
// one-word value, so that every value sits at the same place in every report (the three nest lines read none when
// there are no nests)
function findings(c) {
  const present = c.nests !== 'absent';
  return [
    `surface urothelium: ${c.surface}`,
    `nests below basement membrane: ${present ? 'present' : 'absent'}`,
    `atypia in nests: ${!present ? 'none' : c.nests === 'atypical' ? 'present' : 'absent'}`,
    `nest contours: ${present ? c.contours : 'none'}`,
    `stromal reaction: ${present && c.desmoplasia ? 'desmoplasia' : 'none'}`,
    `muscularis propria: ${{ ni: 'absent', present: 'present', involved: 'involved' }[c.mp]}`,
    `inflammation: ${c.inflammation}`,
  ];
}

// ---- the prose: several phrasings per finding, the microscopic section says nothing the block does not hold
const P = {
  surface: {
    normal: [
      'The urothelium is of normal thickness, with orderly maturation and no cytologic atypia.',
      'The surface urothelium shows preserved polarity and an intact umbrella cell layer, without atypia.',
      'Intact urothelium with normal maturation is present; no atypia is identified.',
      'The urothelium is unremarkable, with maintained polarity and no nuclear atypia.',
    ],
    reactive: [
      'The urothelium shows reactive atypia, with uniformly enlarged nuclei, prominent nucleoli and fine chromatin, and preserved maturation.',
      'Reactive urothelial atypia is present: the nuclei are enlarged with prominent nucleoli, but polarity is maintained and there is no hyperchromasia or pleomorphism.',
      'The surface urothelium shows reactive changes, with nuclear enlargement and prominent nucleoli but preserved maturation and an intact umbrella cell layer.',
      'Reactive atypia of the urothelium is noted, with vesicular nuclei, prominent nucleoli and maintained polarity; there is no loss of maturation.',
    ],
    atypia: [
      'The urothelium shows full-thickness atypia, with enlarged, hyperchromatic, pleomorphic nuclei, loss of polarity and scattered mitoses, consistent with carcinoma in situ.',
      'Flat urothelium with marked cytologic atypia is present: enlarged hyperchromatic nuclei, irregular nuclear membranes and loss of maturation, diagnostic of carcinoma in situ.',
      'The surface urothelium is replaced by cells with large, hyperchromatic nuclei, coarse chromatin and loss of polarity; mitoses are present. The findings are those of carcinoma in situ.',
      'Full-thickness urothelial atypia with hyperchromasia, pleomorphism and disordered maturation is present, consistent with carcinoma in situ.',
    ],
    denuded: [
      'The surface urothelium is denuded, with only scattered residual urothelial cells clinging to the surface.',
      'The urothelium is largely denuded; no intact surface epithelium is present for evaluation.',
      'The surface is denuded, and only lamina propria with a few detached urothelial cells is present.',
    ],
  },
  nests: {
    absent: ['No urothelial nests are present in the lamina propria.', 'The lamina propria contains no von Brunn nests or other urothelial nests.', '', ''], // absence often goes unmentioned
    benign: [
      'Von Brunn nests are present in the lamina propria, rounded and without cytologic atypia.',
      'Rounded nests of cytologically bland urothelium, von Brunn nests, are present beneath the surface.',
      'Small rounded nests of bland urothelium lie within the lamina propria, consistent with von Brunn nests; there is no stromal reaction.',
      'Von Brunn nests with smooth contours and bland cytology are present in the lamina propria.',
    ],
    cisnests: [ // atypical, rounded, no desmoplasia
      'Nests in the lamina propria are involved by the same atypical cells; the nests are rounded, with smooth contours and a lobular arrangement, in keeping with carcinoma in situ extending into von Brunn nests. There is no stromal reaction.',
      'The atypical cells extend into von Brunn nests, which retain smooth, rounded contours; the surrounding stroma shows no desmoplasia or retraction.',
      'Rounded nests of atypical urothelial cells with smooth borders are present in the lamina propria, consistent with involvement of von Brunn nests by carcinoma in situ. No desmoplastic response is seen.',
      'Von Brunn nests are involved by atypical cells but keep their rounded outlines and lobular arrangement, without a stromal reaction; this is carcinoma in situ within nests rather than invasion.',
    ],
    invasive: [ // atypical, irregular, desmoplasia
      'Irregular, angulated nests and cords of atypical urothelial cells infiltrate the lamina propria, with a desmoplastic stromal response and retraction artifact.',
      'Nests of atypical cells with jagged, irregular contours extend into the lamina propria and are surrounded by desmoplastic stroma.',
      'The lamina propria contains infiltrating nests and single atypical cells with irregular contours, with desmoplasia around them.',
      'Atypical urothelial cells invade the lamina propria as irregular nests and cords; there is a desmoplastic stromal reaction.',
    ],
    roundedDesmo: [ // discordant: rounded contours with desmoplasia
      'Nests of atypical urothelial cells in the lamina propria retain rounded contours, but a desmoplastic stromal reaction surrounds some of them, raising the possibility of early invasion.',
      'Atypical cells fill rounded nests in the lamina propria; the nests keep smooth contours, yet there is desmoplasia around several of them, and early invasion cannot be excluded.',
    ],
    irregularNoDesmo: [ // discordant: irregular contours without desmoplasia
      'Some nests of atypical cells in the lamina propria have irregular contours, without a stromal reaction; the distinction between carcinoma in situ involving von Brunn nests and early invasion is difficult on this material.',
      'Nests of atypical urothelial cells in the lamina propria show focally irregular contours, but no desmoplasia or retraction is present; early invasion is suspected but not established.',
    ],
  },
  mp: {
    ni: ['Muscularis propria is not identified.', 'No muscularis propria is present in the biopsy.', 'The biopsy does not include muscularis propria.'],
    presentBenign: ['Muscularis propria is present.', 'Bundles of muscularis propria are present.', 'Muscularis propria is present and unremarkable.'],
    presentTumor: ['Muscularis propria is present and uninvolved.', 'Bundles of muscularis propria are present and free of tumor.', 'Muscularis propria is present; it is not involved.'],
    involved: ['Nests of tumor extend between bundles of muscularis propria.', 'Muscularis propria is present and is infiltrated by tumor.', 'Tumor invades the muscularis propria.'],
  },
  inflammation: {
    none: ['', '', 'There is no significant inflammation.'],
    mild: ['There is mild chronic inflammation in the lamina propria.', 'A mild chronic inflammatory infiltrate is present in the lamina propria.', 'Mild chronic inflammation is noted.'],
    marked: ['The lamina propria shows marked chronic inflammation.', 'A dense chronic inflammatory infiltrate is present in the lamina propria.', 'There is marked inflammation of the lamina propria.'],
  },
};
const NUM = ['one', 'two', 'three', 'four', 'five', 'six'];
function microscopic(c) {
  const tumour = c.nests === 'atypical' || c.surface === 'atypia';
  const nestKey = c.nests === 'atypical' ? (c.contours === 'rounded' ? (c.desmoplasia ? 'roundedDesmo' : 'cisnests') : (c.desmoplasia ? 'invasive' : 'irregularNoDesmo')) : c.nests;
  const s = [pick(P.surface[c.surface]), pick(P.nests[nestKey]), pick(P.mp[c.mp === 'present' ? (tumour ? 'presentTumor' : 'presentBenign') : c.mp])];
  const infl = pick(P.inflammation[c.inflammation]);
  if (infl) s.splice(chance(0.3) ? 1 : 3, 0, infl); // inflammation after the surface or at the end
  return s.filter(Boolean).join(' ');
}
function gross(c) {
  const n = c.fragments, size = c.sizes.length === 1 ? `${(c.sizes[0] / 10).toFixed(1)} cm` : `${(c.sizes[0] / 10).toFixed(1)} to ${(c.sizes[1] / 10).toFixed(1)} cm`;
  return `${NUM[n - 1][0].toUpperCase()}${NUM[n - 1].slice(1)} tan-pink tissue fragment${n === 1 ? '' : 's'}, ${size}, entirely submitted in ${NUM[c.cassettes - 1]} cassette${c.cassettes === 1 ? '' : 's'}.`;
}
function render(c) {
  const d = diagnose(c), sections = {
    findings: findings(c),
    specimen: `Bladder, ${c.site}, ${c.procedure}.`,
    clinical: c.clinical,
    gross: gross(c),
    microscopic: microscopic(c),
    diagnosis: d.text,
  };
  const text = `FINDINGS\n${sections.findings.join('\n')}\nSPECIMEN: ${sections.specimen}\nCLINICAL: ${sections.clinical}\nGROSS: ${sections.gross}\nMICROSCOPIC: ${sections.microscopic}\nDIAGNOSIS: ${sections.diagnosis}`;
  return { sections, dx: d.dx, text };
}

// ---- the corpus: counts per class from the base rates, shuffled; the held-out combination apart
function allocate(n) {
  const counts = CLASSES.map(k => Math.floor(k.share * n)); let left = n - counts.reduce((a, b) => a + b, 0);
  for (let i = 0; left > 0; i = (i + 1) % CLASSES.length, left--) counts[i]++;
  const keys = []; CLASSES.forEach((k, i) => { for (let j = 0; j < counts[i]; j++) keys.push(k); });
  for (let i = keys.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [keys[i], keys[j]] = [keys[j], keys[i]]; }
  return keys;
}
function makeSet(prefix, split, classes) {
  return classes.map((cls, i) => { const c = makeCase(cls), r = render(c), name = `${prefix}${String(i + 1).padStart(3, '0')}`; return { id: name, name, split, case: c, dx: r.dx, text: r.text }; }); // the sections are the lines of the text
}
const train = makeSet('R', 'train', allocate(N_TRAIN)), test = makeSet('T', 'test', allocate(N_TEST)), held = makeSet('H', 'held', Array.from({ length: N_HELD }, () => HELD));

// ---- what the corpus looks like, for the console and the meta
const tokenize = text => text.match(/[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?|[^\sA-Za-z\d]/g) || [];
const all = [...train, ...test, ...held], lens = all.map(r => tokenize(r.text).length);
const vocab = new Set(), vocabLower = new Set(); for (const r of all) for (const t of tokenize(r.text)) { vocab.add(t); vocabLower.add(t.toLowerCase()); }
const dxCounts = {}; for (const r of train) dxCounts[r.dx] = (dxCounts[r.dx] || 0) + 1;
const clsCounts = {}; for (const r of train) clsCounts[r.case.cls] = (clsCounts[r.case.cls] || 0) + 1;
console.log(`train ${train.length} · test ${test.length} · held ${held.length} · tokens per report ${Math.min(...lens)}–${Math.max(...lens)} (mean ${(lens.reduce((a, b) => a + b, 0) / lens.length).toFixed(0)}) · vocabulary ${vocab.size} cased, ${vocabLower.size} lowercased`);
console.log('training classes:', Object.entries(clsCounts).map(([k, n]) => `${k} ${n}`).join(' · '));
console.log('training diagnoses:'); for (const [dx, n] of Object.entries(dxCounts).sort((a, b) => b[1] - a[1])) console.log(`  ${String(n).padStart(4)}  ${dx}`);

const meta = {
  question: 'Reports: write the report from the findings?',
  blurb: 'Every document is a findings block, as an image analyser might report it, followed by the report: specimen, clinical history, gross, microscopic description and, last, the diagnosis. The diagnosis follows from the findings by a fixed rule; the microscopic section says only what the block holds, in one of several phrasings.',
  findings: [
    { key: 'surface', line: 'surface urothelium', states: ['normal', 'reactive', 'atypia', 'denuded'] },
    { key: 'nests', line: 'nests below basement membrane', states: ['absent', 'present'] },
    { key: 'atypiaInNests', line: 'atypia in nests', states: ['none', 'absent', 'present'], note: 'none when there are no nests' },
    { key: 'contours', line: 'nest contours', states: ['none', 'rounded', 'irregular'], note: 'none when there are no nests' },
    { key: 'desmoplasia', line: 'stromal reaction', states: ['none', 'desmoplasia'] },
    { key: 'mp', line: 'muscularis propria', states: ['absent', 'present', 'involved'] },
    { key: 'inflammation', line: 'inflammation', states: ['none', 'mild', 'marked'], note: 'bears on nothing' },
  ],
  classes: CLASSES.map(({ key, share, name }) => ({ key, share, name })).concat([{ key: HELD.key, share: 0, name: HELD.name, held: true }]),
  rule: [
    'surface normal, nests absent or benign → Benign urothelium.',
    'surface reactive atypia, nests absent or benign → Benign urothelium with reactive changes.',
    'surface denuded, nests absent or benign → Denuded urothelium, no diagnostic abnormality in the material present.',
    'surface atypia, nests absent or benign → Urothelial carcinoma in situ.',
    'atypical nests, rounded, no desmoplasia → Urothelial carcinoma in situ.',
    'atypical nests, irregular, desmoplasia → Urothelial carcinoma, invasive into lamina propria (into muscularis propria when involved), with associated carcinoma in situ when the surface shows atypia.',
    'atypical nests, contours and desmoplasia discordant → Urothelial carcinoma in situ with foci suspicious for invasion.',
    'then: Muscularis propria not identified · present, not involved · present and involved.',
  ],
  sections: ['findings', 'specimen', 'clinical', 'gross', 'microscopic', 'diagnosis'],
  counts: { train: train.length, test: test.length, held: held.length },
  tokens: { min: Math.min(...lens), max: Math.max(...lens), mean: Math.round(lens.reduce((a, b) => a + b, 0) / lens.length), vocabulary: vocab.size, vocabularyLowercased: vocabLower.size },
  seed: SEED,
};
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'reports_data.js'), `window.LECTURE_REPORTS = ${JSON.stringify({ meta, train, test, held })};\n`);

// ---- the sheet for vetting: reports of every class, the hidden case above each
const wanted = { benign: 3, reactive: 2, denuded: 2, cis: 3, cisnests: 3, invasive: 4, suspicious: 2 }, sheet = [];
for (const [cls, k] of Object.entries(wanted)) { const rs = train.filter(r => r.case.cls === cls); const picked = cls === 'invasive' ? [rs.find(r => r.case.surface === 'denuded'), rs.find(r => r.case.mp === 'involved'), ...rs.filter(r => r.case.surface === 'atypia' && r.case.mp !== 'involved')] : rs; sheet.push(...picked.filter(Boolean).slice(0, k)); }
sheet.push(...held.slice(0, 2));
const caseLine = c => `surface ${c.surface} · nests ${c.nests}${c.nests === 'atypical' ? ` (${c.contours}, ${c.desmoplasia ? 'desmoplasia' : 'no desmoplasia'})` : ''} · muscularis propria ${{ ni: 'not identified', present: 'present, not involved', involved: 'involved' }[c.mp]} · inflammation ${c.inflammation}`;
const md = [`# Sample reports\n\n${sheet.length} generated reports, of every class, with the hidden case above each one. Regenerate with \`node tools/generate_reports.js\`. Corpus: ${train.length} training, ${test.length} test and ${held.length} held-out reports; ${meta.tokens.min}–${meta.tokens.max} tokens each (mean ${meta.tokens.mean}); vocabulary ${vocab.size} tokens.\n`];
for (const r of sheet) md.push(`## ${r.name} · ${CLASSES.concat(HELD).find(k => k.key === r.case.cls).name}${r.split === 'held' ? ' · held out of training' : ''}\n\n*${caseLine(r.case)}*\n\n\`\`\`\n${r.text}\n\`\`\`\n`);
fs.writeFileSync(path.join(OUT, 'sample_reports.md'), md.join('\n'));
console.log(`wrote ${path.join(OUT, 'reports_data.js')} and sample_reports.md (${sheet.length} reports)`);
