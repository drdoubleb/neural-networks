/*
 * dataset.js — turns the embedded NUCLEI_DATA into ready-to-train specimens, and builds the two input
 * representations the demo can feed the network:
 *   'features' : 6 morphometric measurements per nucleus (see features.js)
 *   'pixels'   : all 1,024 ink values (mean image subtracted)
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

  function prepare(raw) {
    const size = raw.meta.size;
    const specimens = raw.nuclei.map(n => {
      const px = NF.decodeBase64(n.px);
      const m = NF.measure(px, size);
      return {
        id: n.id, name: n.name, split: n.split, label: n.label, className: n.className,
        px, ink: NF.toInk(px), size,
        measurement: m, features: m.vector,
      };
    });
    return {
      size, specimens,
      train: specimens.filter(s => s.split === 'train'),
      test: specimens.filter(s => s.split === 'test'),
    };
  }

  // Builds standardized inputs. The standardizer is fitted on the training set only.
  // augment: also include the 7 other flips/rotations of every training image (pixels mode only).
  function buildInputs(ds, mode, { augment = false } = {}) {
    const rawOf = s => (mode === 'pixels' ? s.ink : s.features);
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
      mode, std,
      inputSize: trainRows[0].length,
      trainX: trainRows.map(r => std.apply(r)), trainY: trainLabels, trainOwner,
      xOf: s => byId.get(s.id),
      featureNames: mode === 'features' ? NF.FEATURES.map(f => f.name) : null,
    };
  }

  return { prepare, buildInputs, dihedral };
});
