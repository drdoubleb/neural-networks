/*
 * dataset.js — turns a generated task (nuclei images or blood counts) into ready-to-train specimens, and builds
 * the input representations the demo can feed the network:
 *   'features' : the task's measurements (six nuclear morphometrics, or the ten CBC parameters)
 *   'pixels'   : all 1,024 ink values (mean image subtracted) — image tasks only
 * Every nucleus was scanned at two labs (px: our lab; pxB: the other lab, whose stain is weaker). A source mode says
 * which lab each set's cases come from (assignLabs), and the pixels can be stain-normalised before the network sees them.
 * Works in the browser (window.NucleusDataset) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(root.NucleusFeatures || require('./features.js'), root.TinyNet || require('./nn.js'));
  else root.NucleusDataset = factory(root.NucleusFeatures, root.TinyNet);
})(typeof self !== 'undefined' ? self : this, function (NF, NN) {
  'use strict';

  // the 8 symmetries of a square image: t = rot (0..3) + 4 * flip
  function dihedral(px, size, t) {
    if (t === 0) return px;
    const out = new px.constructor(px.length);
    const rot = t & 3, flip = t >> 2;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      let sx = x, sy = y;
      if (flip) sx = size - 1 - sx;
      for (let r = 0; r < rot; r++) { const nx = size - 1 - sy; sy = sx; sx = nx; }
      out[y * size + x] = px[sy * size + sx];
    }
    return out;
  }

  // how far a value sits outside its reference range, in half-ranges (0 = mid-range, ±1 = at the limit), clipped to ±3
  function deviation(def, v) {
    if (def.log) {
      const lo = Math.log(Math.max(def.low, 1e-3)), hi = Math.log(def.high), mid = (lo + hi) / 2, half = Math.max(1e-6, (hi - lo) / 2);
      return Math.max(-3, Math.min(3, (Math.log(Math.max(v, 1e-3)) - mid) / half));
    }
    const mid = (def.low + def.high) / 2, half = (def.high - def.low) / 2 || 1;
    return Math.max(-3, Math.min(3, (v - mid) / half));
  }

  // ---- the two labs. A source mode says where a set's cases come from:
  //   ours    every case as scanned at our lab
  //   other   every case as scanned at the other lab (the same nuclei, weaker stain)
  //   mixed   both labs, mixed at random: every other case of each class from the other lab
  //   byClass both labs, split by class: the positives from the other lab, the negatives from ours (the shortcut trap)
  const SOURCE_MODES = ['ours', 'other', 'mixed', 'byClass'];
  function labUnder(mode, s) { return mode === 'other' ? 'B' : mode === 'mixed' ? s.mixLab : mode === 'byClass' ? (s.label ? 'B' : 'A') : 'A'; }
  // the stain levels of one scan, for the normalisation: the background from the outer frame of pixels, the nucleus
  // from the pixels above Otsu's threshold (no label involved)
  function stainLevels(ink, size) {
    let bg = 0, nb = 0;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (x < 2 || y < 2 || x >= size - 2 || y >= size - 2) { bg += ink[y * size + x]; nb++; }
    bg /= nb;
    const thr = NF.otsuThreshold(ink); let nuc = 0, nn = 0;
    for (let i = 0; i < ink.length; i++) if (ink[i] >= thr) { nuc += ink[i]; nn++; }
    return { bg, nuc: nn ? nuc / nn : bg + 0.5 };
  }
  // a scan rescaled so that stain levels (its lab's typical ones, or its own) land on our lab's typical levels: the toy
  // version of matching a slide's colour statistics to a reference slide before the network sees it
  function normalizeInk(ink, from, ref) {
    const k = (ref.nuc - ref.bg) / Math.max(1e-3, from.nuc - from.bg), out = new Float32Array(ink.length);
    for (let i = 0; i < ink.length; i++) out[i] = ref.bg + (ink[i] - from.bg) * k;
    return out;
  }
  const current = v => (v ? { px: v.px, ink: v.ink, measurement: v.measurement, features: v.features } : {});

  function prepare(raw) {
    const task = raw.meta.task;
    const kind = task.kind || 'image';
    const size = raw.meta.size || 0;
    const featureDefs = kind === 'tabular' ? task.features.map(f => Object.assign({}, f, { fmt: v => v.toFixed(f.decimals) })) : NF.FEATURES;
    const specimens = raw.nuclei.map(n => {
      if (kind === 'tabular') {
        return { id: n.id, name: n.name, split: n.split, label: n.label, className: n.className, subtype: n.subtype,
          px: null, ink: null, size: 0, measurement: null, features: n.values.slice(), deviation: n.values.map((v, i) => deviation(featureDefs[i], v)) };
      }
      const variants = {}; // the scans of this nucleus, one per lab, each measured on its own
      for (const [lab, b64] of [['A', n.px], ['B', n.pxB]]) {
        if (!b64) continue;
        const px = NF.decodeBase64(b64), m = NF.measure(px, size), ink = NF.toInk(px);
        variants[lab] = { px, ink, measurement: m, features: m.vector, levels: stainLevels(ink, size) };
      }
      return Object.assign({ id: n.id, name: n.name, split: n.split, label: n.label, className: n.className, subtype: n.subtype,
        size, deviation: null, variants, lab: 'A', mixLab: 'A' }, current(variants.A));
    });
    const labs = kind === 'image' && specimens.every(s => s.variants.B);
    let labStats = null;
    if (labs) {
      for (const split of ['train', 'test']) for (const label of [0, 1]) specimens.filter(s => s.split === split && s.label === label).forEach((s, k) => { s.mixLab = k % 2 ? 'B' : 'A'; });
      labStats = {}; // each lab's typical stain levels over all of its scans
      for (const lab of ['A', 'B']) { let bg = 0, nuc = 0; for (const s of specimens) { bg += s.variants[lab].levels.bg; nuc += s.variants[lab].levels.nuc; } labStats[lab] = { bg: bg / specimens.length, nuc: nuc / specimens.length }; }
    }
    return {
      kind, size, task, featureDefs,
      specimens,
      train: specimens.filter(s => s.split === 'train'),
      test: specimens.filter(s => s.split === 'test'),
      labs, labStats, sources: { train: 'ours', test: 'ours' },
    };
  }
  // gives every case the scan of the lab its set's source mode assigns it (px, ink, measurement and features follow)
  function assignLabs(ds, trainMode, testMode) {
    ds.sources = { train: ds.labs && SOURCE_MODES.includes(trainMode) ? trainMode : 'ours', test: ds.labs && SOURCE_MODES.includes(testMode) ? testMode : 'ours' };
    if (!ds.labs) return ds;
    for (const s of ds.specimens) {
      const lab = labUnder(s.split === 'train' ? ds.sources.train : ds.sources.test, s);
      s.lab = s.variants[lab] ? lab : 'A';
      Object.assign(s, current(s.variants[s.lab]));
    }
    return ds;
  }

  // Builds standardized inputs. The standardizer is fitted on the training set only.
  //   augment:   also include the 7 other flips/rotations of every training image (pixels mode only)
  //   exclude:   set of feature keys the network is not allowed to see (features mode)
  //   normalize: 'off', or stain-normalise the pixels first, 'lab' (each lab's typical levels matched to our lab's) or
  //              'image' (each scan by its own levels)
  function buildInputs(ds, mode, { augment = false, exclude = null, normalize = 'off' } = {}) {
    const columns = mode === 'features' ? ds.featureDefs.map((f, i) => i).filter(i => !exclude || !exclude.has(ds.featureDefs[i].key)) : null;
    const variant = (s, lab) => (s.variants && (s.variants[lab] || s.variants.A)) || s;
    const inkFor = (s, lab) => { const v = variant(s, lab); if (normalize === 'off' || !ds.labStats || !v.levels) return v.ink; return normalizeInk(v.ink, normalize === 'lab' ? ds.labStats[lab] || ds.labStats.A : v.levels, ds.labStats.A); };
    const rawFor = (s, lab) => (mode === 'pixels' ? inkFor(s, lab) : columns.map(i => { const f = variant(s, lab).features; return ds.featureDefs[i].log ? Math.log(Math.max(f[i], 1e-3)) : f[i]; }));
    const rawOf = s => rawFor(s, s.lab || 'A');
    let trainRows = ds.train.map(rawOf), trainLabels = ds.train.map(s => s.label), trainOwner = ds.train.map(s => s.id);
    if (mode === 'pixels' && augment) {
      const rows = [], labels = [], owner = [];
      ds.train.forEach(s => { const raw = rawOf(s); for (let t = 0; t < 8; t++) { rows.push(t === 0 ? raw : dihedral(raw, ds.size, t)); labels.push(s.label); owner.push(s.id); } });
      trainRows = rows; trainLabels = labels; trainOwner = owner;
    }
    const std = NN.fitStandardizer(trainRows, { perDimScale: mode !== 'pixels' });
    const byId = new Map();
    for (const s of ds.specimens) byId.set(s.id, std.apply(rawOf(s)));
    return {
      mode, std, columns, normalize,
      inputSize: trainRows[0].length,
      trainX: trainRows.map(r => std.apply(r)), trainY: trainLabels, trainOwner,
      xOf: s => byId.get(s.id),
      rawOf,                                            // what the network is given before standardisation (the normalised ink, on pixels)
      xFor: (s, lab) => std.apply(rawFor(s, lab)),      // the same case as scanned at a given lab
      featureNames: mode === 'features' ? columns.map(i => ds.featureDefs[i].name) : null,
    };
  }

  return { prepare, assignLabs, buildInputs, dihedral, deviation, SOURCE_MODES, labUnder, stainLevels, normalizeInk };
});
