#!/usr/bin/env node
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const NN = require('../js/nn.js');
const BP = require('../js/backprop.js');
const DS = require('../js/dataset.js');
const close = (a, b, label, tol = 1e-8) => assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(a) + Math.abs(b)), `${label}: ${a} vs ${b}`);
let comparisons = 0;
function check(net, x, y, lr, l2, name) {
  const before = JSON.stringify(net), res = net.lesson(x, y);
  assert.equal(JSON.stringify(net), before, 'Preview must not mutate parameters or step count');
  const targets = ['out:0']; net.hidden.forEach((h, l) => { for (let j = 0; j < h; j++) targets.push(`${l}:${j}`); });
  const selected = [];
  for (const target of targets) {
    const t = BP.targetInfo(res, target);
    const indices = new Set([0, Math.floor(t.input.length / 2), t.input.length - 1, BP.bestInput(res, target, lr, l2), 'bias']);
    for (const i of indices) {
      const expected = BP.parameter(res, target, i, lr, l2);
      const isOut = t.l === 'out', isBias = i === 'bias';
      const arr = isBias ? isOut ? null : net.b[t.l] : isOut ? net.Wo : net.W[t.l];
      const idx = isBias ? t.j : isOut ? i : t.j * t.input.length + i;
      const get = () => arr ? arr[idx] : net.bo;
      const set = v => { if (arr) arr[idx] = v; else net.bo = v; };
      const old = get(), eps = 1e-6;
      set(old + eps); const high = NN.bceFromLogit(net.forward(x).z, y);
      set(old - eps); const low = NN.bceFromLogit(net.forward(x).z, y);
      set(old);
      close(expected.gradient, (high - low) / (2 * eps), `${name} ${target} ${i} finite difference`, 2e-6);
      close(expected.gradient, expected.input * expected.delta, 'input × delta');
      selected.push({ expected, get }); comparisons++;
    }
    if (t.l !== 'out') {
      let sum = 0;
      if (t.l === net.hidden.length - 1) sum = res.before.Wo[t.j] * res.error;
      else for (let k = 0; k < net.hidden[t.l + 1]; k++) sum += res.before.W[t.l + 1][k * net.hidden[t.l] + t.j] * res.g.delta[t.l + 1][k];
      close(sum, res.g.upstream[t.l][t.j], 'Sum all downstream paths');
      close(t.delta, sum * res.g.slopes[t.l][t.j], 'Apply local activation derivative');
    }
  }
  net.applyGradient(res.g, 1, lr, l2);
  for (const { expected, get } of selected) close(get(), expected.after, 'Displayed parameter update matches optimizer', 1e-12);
  assert.equal(net.steps, 1);
  assert.equal(res.before.bo, JSON.parse(before).bo, 'Saved parameters survive update');
  return res;
}
for (const activation of ['relu', 'sigmoid', 'tanh']) for (const hidden of [[], [3], [3, 2]]) for (const y of [0, 1]) {
  check(new NN.Net({ inputSize: 4, hidden, activation, seed: 3 }), Float64Array.from([-1.4, 0, 0.2, 1.3]), y, 0.1, 0.07, `${activation}/${hidden}/${y}`);
}
// Off ReLU: zero data gradients, nonzero decay; biases must never decay.
const off = new NN.Net({ inputSize: 2, hidden: [1], activation: 'relu' });
off.W[0].set([0.5, -0.5]); off.b[0][0] = -2; off.Wo[0] = 0.7;
const offRes = check(off, Float64Array.from([0, 0]), 1, 0.1, 0.1, 'off ReLU');
close(offRes.g.delta[0][0], 0, 'Inactive delta'); assert.equal(offRes.g.slopes[0][0], 0);
close(off.W[0][0], 0.495, 'Inactive weight still decays'); close(off.b[0][0], -2, 'Inactive bias unchanged');
for (const z of [-1000, -100, 0, 100, 1000]) for (const y of [0, 1]) {
  assert.ok(Number.isFinite(NN.bceFromLogit(z, y)) && NN.bceFromLogit(z, y) >= 0);
  const eps = 1e-3, numerical = (NN.bceFromLogit(z + eps, y) - NN.bceFromLogit(z - eps, y)) / (2 * eps);
  close(numerical, NN.sigmoid(z) - y, 'Stable loss has displayed derivative', 1e-7);
}
const window = {};
for (const f of ['leukaemia/patients_data.js', 'atypia/nuclei_data.js', 'enlargement/nuclei_data.js', 'irregularity/nuclei_data.js']) new Function('window', fs.readFileSync(path.join(__dirname, '..', 'data', f), 'utf8'))(window);
const recipes = [['leukaemia', 'features', [], .05, 0], ['leukaemia', 'features', [3], .05, 0], ['atypia', 'features', [], .1, 0], ['enlargement', 'pixels', [], .02, 0], ['irregularity', 'pixels', [], .01, 0], ['irregularity', 'pixels', [4], .02, .01], ['irregularity', 'pixels', [4, 4], .02, .01]];
recipes.forEach(([task, mode, hidden, lr, l2], ri) => {
  const ds = DS.prepare(window.LECTURE_TASKS[task]), inputs = DS.buildInputs(ds, mode, { augment: ri >= 5 });
  for (const y of [0, 1]) {
    const sample = ds.train.find(s => s.label === y);
    check(new NN.Net({ inputSize: inputs.inputSize, hidden, activation: 'relu', seed: 1 }), inputs.xOf(sample), y, lr, l2, `recipe ${ri + 1}`);
  }
  const p = BP.phases(hidden);
  assert.deepEqual(p.filter(s => s.phase === 'blame').map(s => s.layer), hidden.map((_, i) => i).reverse());
});
// A high learning rate can worsen same-case loss. The UI must report the actual result.
const over = new NN.Net({ inputSize: 2, hidden: [2], activation: 'tanh', seed: 3 });
const ox = Float64Array.from([-2, 3]), o = over.lesson(ox, 1); over.applyGradient(o.g, 1, 100, 0.2);
assert.ok(NN.bceFromLogit(over.forward(ox).z, 1) > o.loss);
console.log(`PASS: ${comparisons} finite-difference and optimizer comparisons; all 7 dense recipes, both labels, 0–2 hidden layers, all activations, signed/zero inputs, decay, biases, stable loss, and overshoot.`);
