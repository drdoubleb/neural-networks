/*
 * dataset.js — turns a generated task (nuclei images or blood counts) into ready-to-train specimens, and builds
 * the input representations the demo can feed the network:
 *   'features' : the task's measurements (six nuclear morphometrics, or the ten CBC parameters)
 *   'pixels'   : all 1,024 ink values (mean image subtracted) — image tasks only
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
      const px = NF.decodeBase64(n.px);
      const m = NF.measure(px, size);
      return { id: n.id, name: n.name, split: n.split, label: n.label, className: n.className, subtype: n.subtype,
        px, ink: NF.toInk(px), size, measurement: m, features: m.vector, deviation: null };
    });
    return {
      kind, size, task, featureDefs,
      specimens,
      train: specimens.filter(s => s.split === 'train'),
      test: specimens.filter(s => s.split === 'test'),
    };
  }

  // Builds standardized inputs. The standardizer is fitted on the training set only.
  //   augment: also include the 7 other flips/rotations of every training image (pixels mode only)
  //   exclude: set of feature keys the network is not allowed to see (features mode)
  function buildInputs(ds, mode, { augment = false, exclude = null } = {}) {
    const columns = mode === 'features' ? ds.featureDefs.map((f, i) => i).filter(i => !exclude || !exclude.has(ds.featureDefs[i].key)) : null;
    const rawOf = s => (mode === 'pixels' ? s.ink : columns.map(i => (ds.featureDefs[i].log ? Math.log(Math.max(s.features[i], 1e-3)) : s.features[i])));
    let trainRows = ds.train.map(rawOf), trainLabels = ds.train.map(s => s.label), trainOwner = ds.train.map(s => s.id);
    if (mode === 'pixels' && augment) {
      const rows = [], labels = [], owner = [];
      ds.train.forEach(s => { for (let t = 0; t < 8; t++) { rows.push(t === 0 ? s.ink : dihedral(s.ink, ds.size, t)); labels.push(s.label); owner.push(s.id); } });
      trainRows = rows; trainLabels = labels; trainOwner = owner;
    }
    const std = NN.fitStandardizer(trainRows, { perDimScale: mode !== 'pixels' });
    const byId = new Map();
    for (const s of ds.specimens) byId.set(s.id, std.apply(rawOf(s)));
    return {
      mode, std, columns,
      inputSize: trainRows[0].length,
      trainX: trainRows.map(r => std.apply(r)), trainY: trainLabels, trainOwner,
      xOf: s => byId.get(s.id),
      featureNames: mode === 'features' ? columns.map(i => ds.featureDefs[i].name) : null,
    };
  }

  return { prepare, buildInputs, dihedral, deviation };
});
