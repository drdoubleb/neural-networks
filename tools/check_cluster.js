#!/usr/bin/env node
/*
 * check_cluster.js — the Cluster question's numbers in Node: k-means (js/cluster.js) over every dataset and input the
 * page offers, with the purity of the clusters against the hidden subtypes and against the class, and the accuracy a
 * naming by majority class reaches on the held-out cases. Run:  node tools/check_cluster.js [--k 2,4,7]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { Contrastive } = require('../js/nn.js');
const DS = require('../js/dataset.js');
const K = require('../js/cluster.js');

const ks = process.argv.includes('--k') ? process.argv[process.argv.indexOf('--k') + 1].split(',').map(Number) : [2, 3, 4, 5, 6, 7, 8];
const window = {};
for (const f of ['leukemia/patients_data.js', 'atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js', 'foundation/backbones.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const bb = window.FOUNDATION_BACKBONES[0], enc = Contrastive.fromJSON(bb), encoder = { encode: ink => enc.encode(enc.std.apply(ink)) };
const pc = v => `${Math.round(v * 100)}%`;

for (const [id, raw] of Object.entries(window.LECTURE_TASKS)) {
  const ds = DS.prepare(raw), sub = Object.fromEntries((ds.task.subtypes || []).map(t => [t.key, t.name]));
  const counts = {}; for (const s of ds.train) counts[sub[s.subtype] || s.subtype] = (counts[sub[s.subtype] || s.subtype] || 0) + 1;
  console.log(`\n${ds.task.title} · ${ds.train.length} training / ${ds.test.length} held-out · kinds: ${Object.entries(counts).map(([n, c]) => `${n} ${c}`).join(', ')}`);
  for (const mode of (ds.kind === 'tabular' ? ['features'] : ['features', 'pixels', 'code'])) {
    const inp = DS.buildInputs(ds, mode, { encoder: mode === 'code' ? encoder : null });
    const X = ds.train.map(s => Array.from(inp.xOf(s))), Xt = ds.test.map(s => Array.from(inp.xOf(s)));
    const line = ks.map(k => {
      const st = K.run(K.start(X, k, 1)); // the page's own start: seed 1, the tightest of ten
      const subs = K.contingency(st.A, ds.train.map(s => s.subtype), k), cls = K.contingency(st.A, ds.train.map(s => s.label), k);
      const names = cls.rows.map(r => (r.majority == null ? 1 : r.majority)), At = Xt.map(x => K.nearest(x, st.C));
      const held = ds.test.filter((s, i) => names[At[i]] === s.label).length / ds.test.length;
      return `k${k} ${pc(subs.purity)}/${pc(cls.purity)}/${pc(held)}`;
    });
    console.log(`  ${mode.padEnd(8)} (${X[0].length} numbers)  subtype purity / class purity / named → held-out:  ${line.join(' · ')}`);
  }
}
