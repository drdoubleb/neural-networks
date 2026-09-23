#!/usr/bin/env node
/*
 * generate_cbc.js — builds the synthetic complete-blood-count dataset for the "Leukaemia?" question.
 *
 * Zero dependencies (Node >= 16). Run:  node tools/generate_cbc.js
 *
 * Writes data/leukaemia/patients_data.js (loaded by index.html), patients.json and patients.csv.
 *
 * 200 synthetic patients, 160 training / 40 test. Each patient is drawn from one of seven hidden subtypes:
 *   leukaemia:   acute leukaemia (a third present aleukaemic, with a low count), CML, CLL
 *   not:         normal, bacterial infection (neutrophilia + left shift), viral lymphocytosis, iron-deficiency anaemia
 * The network only ever sees the label (leukaemia yes/no); the subtype is kept so the page can show which
 * hidden units respond to which kind of blood. Distributions are plausible teaching values, not population data.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const SEED = 20260923;
const N = 200, TEST_FRACTION = 0.2;
const OUT = path.join(__dirname, '..', 'data', 'leukaemia');

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
const G = () => { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const LN = (median, sigma) => median * Math.exp(sigma * G());                       // log-normal
const NR = (mu, sd, lo, hi) => Math.min(hi, Math.max(lo, mu + sd * G()));           // clipped normal

// the parameters the network sees, with adult reference ranges used for the report flags and fingerprints
const FEATURES = [
  { key: 'wbc',   name: 'WBC',        unit: '×10⁹/L', low: 4.0, high: 11.0, log: true,  decimals: 1, desc: 'White cell count. Raised in infection and in leukaemias with circulating cells; can be low in acute leukaemia.' },
  { key: 'hb',    name: 'Hb',         unit: 'g/dL',   low: 12.0, high: 16.0, log: false, decimals: 1, desc: 'Haemoglobin. Falls when the marrow is replaced or iron is short.' },
  { key: 'plt',   name: 'Platelets',  unit: '×10⁹/L', low: 150, high: 400,  log: true,  decimals: 0, desc: 'Platelet count. Low when the marrow is replaced; reactive rises in infection and iron deficiency; often high in CML.' },
  { key: 'mcv',   name: 'MCV',        unit: 'fL',     low: 80,  high: 100,  log: false, decimals: 0, desc: 'Mean red-cell volume. Low in iron deficiency, mildly high in some acute leukaemias.' },
  { key: 'neut',  name: 'Neutrophils', unit: '%',     low: 40,  high: 75,   log: false, decimals: 0, desc: 'Neutrophil fraction of the differential.' },
  { key: 'lymph', name: 'Lymphocytes', unit: '%',     low: 20,  high: 45,   log: false, decimals: 0, desc: 'Lymphocyte fraction. Very high in CLL, moderately high in viral infection.' },
  { key: 'eos',   name: 'Eosinophils', unit: '%',     low: 0,   high: 6,    log: false, decimals: 1, desc: 'Eosinophil fraction. Raised in CML, suppressed in acute bacterial infection.' },
  { key: 'baso',  name: 'Basophils',  unit: '%',      low: 0,   high: 1,    log: false, decimals: 1, desc: 'Basophil fraction. Basophilia is a hallmark of CML.' },
  { key: 'ig',    name: 'Immature granulocytes', unit: '%', low: 0, high: 0.5, log: false, decimals: 1, desc: 'Myelocytes, metamyelocytes and promyelocytes: a left shift. Marked in CML, mild in bacterial infection.' },
  { key: 'blast', name: 'Blasts',     unit: '%',      low: 0,   high: 0,    log: false, decimals: 0, desc: 'Blasts on the smear. Any blast is abnormal; high in acute leukaemia with a raised count, few or none when aleukaemic.' },
];
const SUBTYPES = [
  { key: 'normal',    name: 'Normal',                  leuk: 0, w: 0.20, gen: () => ({ wbc: LN(6.5, 0.22), hb: NR(14, 1.2, 11.5, 17), plt: LN(250, 0.22), mcv: NR(90, 4, 80, 100), neut: NR(58, 8, 35, 80), lymph: NR(30, 7, 15, 50), eos: NR(2.5, 1.2, 0, 7), baso: NR(0.5, 0.3, 0, 1.5), ig: NR(0.2, 0.15, 0, 0.8), blast: 0 }) },
  { key: 'bacterial', name: 'Bacterial infection',     leuk: 0, w: 0.12, gen: () => { const septic = rand() < 0.25; return { wbc: LN(18, 0.4), hb: NR(12.5, 1.5, 8.5, 16), plt: septic ? LN(90, 0.4) : LN(320, 0.3), mcv: NR(89, 4, 80, 100), neut: NR(84, 5, 70, 94), lymph: NR(9, 4, 2, 20), eos: NR(0.5, 0.5, 0, 2), baso: NR(0.3, 0.3, 0, 1.2), ig: NR(4, 3, 0.5, 15), blast: 0 }; } },
  { key: 'viral',     name: 'Viral lymphocytosis',     leuk: 0, w: 0.10, gen: () => ({ wbc: LN(15, 0.38), hb: NR(13.5, 1.1, 11, 16), plt: LN(160, 0.3), mcv: NR(90, 4, 80, 100), neut: NR(24, 8, 8, 45), lymph: NR(66, 10, 45, 88), eos: NR(2, 1.2, 0, 6), baso: NR(0.5, 0.3, 0, 1.5), ig: NR(0.5, 0.4, 0, 2), blast: 0 }) },
  { key: 'ida',       name: 'Iron-deficiency anaemia', leuk: 0, w: 0.08, gen: () => ({ wbc: LN(6.8, 0.22), hb: NR(9, 1.4, 5.5, 11.5), plt: LN(420, 0.25), mcv: NR(70, 5, 58, 79), neut: NR(58, 8, 35, 80), lymph: NR(30, 7, 15, 50), eos: NR(2.5, 1.2, 0, 7), baso: NR(0.5, 0.3, 0, 1.5), ig: NR(0.2, 0.15, 0, 0.8), blast: 0 }) },
  { key: 'acute',     name: 'Acute leukaemia',         leuk: 1, w: 0.20, gen: () => { const high = rand() < 0.6; return { wbc: high ? LN(40, 0.7) : LN(3, 0.4), hb: NR(9, 1.6, 5.5, 12.5), plt: high ? LN(45, 0.6) : LN(70, 0.6), mcv: NR(94, 5, 82, 108), neut: NR(18, 9, 2, 45), lymph: NR(22, 9, 5, 50), eos: NR(0.5, 0.5, 0, 2), baso: NR(0.3, 0.3, 0, 1.2), ig: NR(1.5, 1.2, 0, 5), blast: high ? NR(45, 22, 8, 95) : NR(4, 4, 0, 15) }; } },
  { key: 'cml',       name: 'CML',                     leuk: 1, w: 0.15, gen: () => ({ wbc: LN(120, 0.5), hb: NR(11, 1.5, 7.5, 14), plt: LN(420, 0.5), mcv: NR(88, 4, 78, 98), neut: NR(45, 8, 25, 65), lymph: NR(6, 3, 1, 15), eos: NR(4, 2, 0.5, 10), baso: NR(7, 3, 2, 18), ig: NR(22, 7, 8, 40), blast: NR(2, 1.5, 0, 6) }) },
  { key: 'cll',       name: 'CLL',                     leuk: 1, w: 0.15, gen: () => ({ wbc: LN(32, 0.6), hb: NR(12.5, 1.8, 8, 16), plt: LN(180, 0.35), mcv: NR(90, 4, 80, 100), neut: NR(14, 6, 3, 30), lymph: NR(80, 9, 55, 96), eos: NR(0.5, 0.5, 0, 2), baso: NR(0.3, 0.3, 0, 1.2), ig: NR(0.3, 0.3, 0, 1.5), blast: 0 }) },
];

function patient() {
  let r = rand(), sub = SUBTYPES[SUBTYPES.length - 1];
  for (const s of SUBTYPES) { r -= s.w; if (r <= 0) { sub = s; break; } }
  const c = sub.gen();
  const total = c.neut + c.lymph + c.eos + c.baso + c.ig + c.blast + 6; // the ~6% of monocytes are not reported
  for (const k of ['neut', 'lymph', 'eos', 'baso', 'ig', 'blast']) c[k] = c[k] / total * 100;
  const values = FEATURES.map(f => +c[f.key].toFixed(f.decimals === 0 ? 0 : 2));
  return { subtype: sub.key, label: sub.leuk, values };
}

const all = Array.from({ length: N }, patient);
const nTest = Math.round(N * TEST_FRACTION);
const records = all.map((p, i) => {
  const split = i < nTest ? 'test' : 'train';
  const k = split === 'test' ? i + 1 : i - nTest + 1;
  return { id: i, name: `${split === 'train' ? 'T' : 'X'}${String(k).padStart(split === 'train' ? 3 : 2, '0')}`, split, label: p.label, className: p.label ? 'leukaemia' : 'no', subtype: p.subtype, values: p.values };
});
// put the training cases first so ids run train then test, like the nuclei sets
records.sort((a, b) => (a.split === b.split ? 0 : a.split === 'train' ? -1 : 1));
records.forEach((r, i) => { r.id = i; });

const task = {
  id: 'leukaemia', order: 0, kind: 'tabular',
  title: 'Leukaemia?', short: 'Leukaemia',
  classes: [{ key: 'no', name: 'No leukaemia' }, { key: 'leukaemia', name: 'Leukaemia' }],
  blurb: 'Two hundred synthetic complete blood counts. Half are leukaemias: acute leukaemia (a third of them presenting with a low count), CML and CLL. The other half are the blood counts that fool a simple rule: normal, bacterial infection with neutrophilia and a left shift, viral lymphocytosis, and iron-deficiency anaemia with a reactive thrombocytosis. The network is told only leukaemia yes or no; the subtype is kept so we can see which hidden units respond to which kind of blood.',
  decoys: 'a raised white count on its own, cytopenias on their own',
  signal: 'the pattern across the count and the differential',
  features: FEATURES,
  subtypes: SUBTYPES.map(s => ({ key: s.key, name: s.name, positive: s.leuk })),
  specimenNoun: 'patient',
};
const meta = { generated: new Date().toISOString().slice(0, 10), seed: SEED, size: 0, count: records.length, train: records.filter(r => r.split === 'train').length, test: records.filter(r => r.split === 'test').length, task, classes: task.classes.map(c => c.key) };

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'patients.json'), JSON.stringify({ meta, patients: records }, null, 1));
fs.writeFileSync(path.join(OUT, 'patients.csv'), ['id,name,split,label,subtype,' + FEATURES.map(f => f.key).join(','), ...records.map(r => [r.id, r.name, r.split, r.label, r.subtype, ...r.values].join(','))].join('\n') + '\n');
fs.writeFileSync(path.join(OUT, 'patients_data.js'),
  `// Generated by tools/generate_cbc.js — do not edit by hand.\n// Task "leukaemia": ${records.length} synthetic complete blood counts (${meta.train} train / ${meta.test} test).\n` +
  `window.LECTURE_TASKS = window.LECTURE_TASKS || {};\nwindow.LECTURE_TASKS["leukaemia"] = ${JSON.stringify({ meta, nuclei: records })};\n`);
const counts = {}; for (const r of records) counts[r.subtype] = (counts[r.subtype] || 0) + 1;
console.log(`[leukaemia] wrote ${records.length} patients (${meta.train} train / ${meta.test} test) to ${OUT}`);
console.log('  subtypes:', counts, '| leukaemia:', records.filter(r => r.label).length);
