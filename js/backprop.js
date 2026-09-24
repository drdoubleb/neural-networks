/* Exact, inspectable views of a single dense-network learning step. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./nn.js'));
  else root.Backprop = factory(root.TinyNet);
})(typeof self !== 'undefined' ? self : this, function (NN) {
  'use strict';
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const num = v => !Number.isFinite(v) ? String(v) : v === 0 ? '0' : Math.abs(v) < 0.0001 ? v.toExponential(2) : v.toFixed(4);
  const signed = v => (v > 0 ? '+' : '') + num(v);
  const mono = v => `<span class="bp-number">${v}</span>`;
  const unitName = (l, j) => l === 'out' ? 'Output' : `Hidden ${l + 1} · unit ${j + 1}`;
  function phases(hidden) {
    const p = [{ phase: 'forward', title: 'Predict' }, { phase: 'error', title: 'Measure loss' }];
    for (let l = hidden.length - 1; l >= 0; l--) p.push({ phase: 'blame', layer: l, title: `Back through H${l + 1}` });
    return [...p, { phase: 'gradient', title: 'Weight gradients' }, { phase: 'nudge', title: 'Update weights' }, { phase: 'after', title: 'Predict again' }];
  }
  function targetInfo(res, target) {
    const [ls, js] = target.split(':');
    const l = ls === 'out' ? 'out' : +ls, j = +js || 0;
    const { fw, g, before } = res;
    const input = l === 'out' ? fw.a[fw.a.length - 1] : fw.a[l];
    const offset = l === 'out' ? 0 : j * input.length;
    const weights = l === 'out' ? before.Wo : before.W[l].subarray(offset, offset + input.length);
    const gradients = l === 'out' ? g.gWo : g.gW[l].subarray(offset, offset + input.length);
    return { l, j, input, weights, gradients, delta: l === 'out' ? res.error : g.delta[l][j], bias: l === 'out' ? before.bo : before.b[l][j], name: unitName(l, j) };
  }
  function parameter(res, target, i, lr, l2) {
    const t = targetInfo(res, target), bias = i === 'bias';
    return { ...NN.parameterStep(bias ? t.bias : t.weights[i], bias ? t.delta : t.gradients[i], lr, l2, bias), input: bias ? 1 : t.input[i], delta: t.delta, bias };
  }
  function bestInput(res, target, lr, l2) {
    const t = targetInfo(res, target); let best = 0, size = -1;
    t.weights.forEach((w, i) => { const d = Math.abs(parameter(res, target, i, lr, l2).change); if (d > size) { best = i; size = d; } });
    return best;
  }
  function inputName(t, i, c) {
    const first = t.l === 0 || (t.l === 'out' && !c.net.hidden.length);
    return first ? c.mode === 'pixels' ? `Pixel (${i % c.size + 1}, ${Math.floor(i / c.size) + 1})` : c.featureNames[i] : `Hidden ${t.l === 'out' ? c.net.hidden.length : t.l} · unit ${i + 1}`;
  }
  function targetOptions(c) {
    let opts = '<option value="out:0">Output</option>';
    c.net.hidden.forEach((h, l) => { for (let j = 0; j < h; j++) opts += `<option value="${l}:${j}">${unitName(l, j)}</option>`; });
    return opts;
  }
  function metrics(les, c) {
    const r = les.res, after = les.phase === 'after';
    return `<div class="bp-metrics"><div><span>Known label y</span><b>${les.y} · ${esc(c.truthName)}</b></div><div><span>P(${esc(c.positiveName)})</span><b>${num(r.fw.p)}${after ? ` → ${num(les.pAfter)}` : ''}</b></div><div><span>Case loss · cross-entropy</span><b>${les.phase === 'forward' ? 'Next step' : `${num(r.loss)}${after ? ` → ${num(les.lossAfter)}` : ''}`}</b></div></div>`;
  }
  function explanation(les, c) {
    const { res: r, phase, layer } = les;
    if (phase === 'forward') return `<h3>Start with this ${c.mode === 'pixels' ? 'nucleus' : 'case'} and the current weights</h3><p>Each unit multiplies its inputs by its weights and adds a bias. The output score becomes a probability through the sigmoid.</p><div class="bp-equation">z = Σ(w × input) + b = ${mono(signed(r.fw.z))} → sigmoid(z) = ${mono(num(r.fw.p))}</div><p class="small">${c.mode === 'pixels' ? 'The network sees standardized pixel values: a negative value means below the training-set mean at that position. It does not mean a negative brightness.' : 'Inputs are standardized using the training set. A negative input means below its training mean, and reverses the sign of input × weight.'} This forward pass uses the saved weights from before the update.</p>`;
    if (phase === 'error') return `<h3>How costly was the prediction, and which way should the score move?</h3><p>Loss is a nonnegative penalty. The output gradient is a signed sensitivity: how loss changes if we increase the score z.</p><div class="bp-equation">L = −[y ln(p) + (1 − y) ln(1 − p)] = ${mono(num(r.loss))}</div><div class="bp-equation">δ<sub>out</sub> = ∂L/∂z = p − y = ${mono(num(r.fw.p))} − ${les.y} = ${mono(signed(r.error))}</div><p>${r.error > 0 ? 'Positive gradient → decreasing the score would reduce this case’s loss.' : r.error < 0 ? 'Negative gradient → increasing the score would reduce this case’s loss.' : 'The output gradient is zero at the displayed numerical precision.'} For sigmoid + binary cross-entropy, p − y already includes the sigmoid derivative; do not multiply it by p(1 − p) again.</p>`;
    if (phase === 'blame') {
      const slope = c.net.activation === 'relu' ? 'ReLU: 1 when z > 0; 0 when z ≤ 0 (including the chosen derivative at zero)' : c.net.activation === 'tanh' ? 'Tanh: 1 − a²' : 'Sigmoid: a(1 − a)';
      const rows = Array.from(r.g.delta[layer], (d, j) => `<tr class="${les.target === `${layer}:${j}` ? 'is-selected' : ''}"><th><button type="button" data-bp-unit="${layer}:${j}">H${layer + 1}.${j + 1}</button></th><td>${num(r.fw.pre[layer][j])}</td><td>${num(r.fw.a[layer + 1][j])}</td><td>${signed(r.g.upstream[layer][j])}</td><td>${num(r.g.slopes[layer][j])}</td><td><b>${signed(d)}</b>${r.g.slopes[layer][j] === 0 ? ' · blocked' : ''}</td></tr>`).join('');
      return `<h3>Back through hidden layer ${layer + 1}</h3><p>For each unit, <b>sum</b> the downstream gradients multiplied by the outgoing weights, then multiply by the local activation derivative. These are sensitivities, not percentages of blame: they can reverse sign, cancel, or grow.</p><div class="bp-equation">δ<sub>unit</sub> = (Σ w<sub>outgoing</sub> × δ<sub>downstream</sub>) × f′(z)</div><p class="small">${slope}. All factors use the weights from the original forward pass.</p><div class="bp-table-wrap"><table class="bp-table"><caption>Every unit in this layer. Select a row to inspect its outgoing paths.</caption><thead><tr><th>Unit</th><th>Score z</th><th>Value a</th><th>Weighted sum ∂L/∂a</th><th>Local slope f′</th><th>Result δ = ∂L/∂z</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }
    if (phase === 'gradient') return `<h3>Turn each unit’s gradient into gradients for its incoming weights</h3><p>A weight multiplies one input, so its data gradient is <b>that input × the destination unit’s δ</b>. The bias has an implicit input of 1, so its gradient is δ. Backpropagation computes these gradients; the optimizer applies them in the next step.</p><div class="bp-equation">∂L/∂w = input × δ &nbsp; · &nbsp; ∂L/∂b = δ</div><p class="small">Zero input gives zero data gradient for that weight. A negative input reverses the sign. Computing gradients does not change the weights.</p>`;
    if (phase === 'nudge') return `<h3>${les.applied ? 'Review the update that was applied' : 'Preview one gradient-descent update'}</h3><p>Subtract learning rate × gradient. A <b>positive gradient lowers the weight</b>; a <b>negative gradient raises it</b>. Weight decay adds λw to each weight’s gradient; biases are not decayed.</p><div class="bp-equation">w<sub>new</sub> = w<sub>old</sub> − η(∂L/∂w + λw<sub>old</sub>) &nbsp; · &nbsp; b<sub>new</sub> = b<sub>old</sub> − ηδ</div><p class="small">η = ${les.lr} · λ = ${les.l2}. This is <b>one original training case</b>, batch size 1, with no random augmentation. Normal training averages the data gradients over its batch before adding decay. An inactive ReLU has no data gradient, but its weights can still move through decay.</p>`;
    const change = les.lossAfter - r.loss;
    return `<h3>Run the same case through the updated network</h3><p>${Math.abs(change) < 1e-12 ? 'This case’s loss is unchanged at numerical precision.' : change < 0 ? 'This case’s loss decreased.' : 'This case’s loss increased. A large learning rate can overshoot; weight decay can also trade off this case’s loss against smaller weights.'} This measured result is from <b>one update</b>. It does not establish performance on unseen cases.</p><div class="bp-equation">Case loss ${mono(num(r.loss))} → ${mono(num(les.lossAfter))} &nbsp; · &nbsp; training-set loss ${mono(num(les.trainLossBefore))} → ${mono(num(les.trainLossAfter))}</div><p class="small">Every parameter used its pre-update gradient. Back and the step buttons replay the saved calculation without training again. The regularization penalty is excluded from the loss values above.</p>`;
  }
  function chainDetail(les, c, t) {
    if (t.l === 'out') return `<p class="small">Output gradient δ = p − y = ${signed(les.res.error)}.</p>`;
    const r = les.res, l = t.l, j = t.j, last = l === c.net.hidden.length - 1;
    const terms = last ? [{ name: 'Output', w: r.before.Wo[j], d: r.error }] : Array.from(r.g.delta[l + 1], (d, k) => ({ name: `H${l + 2}.${k + 1}`, w: r.before.W[l + 1][k * c.net.hidden[l] + j], d }));
    return `<div class="bp-chain"><b>Trace ${esc(t.name)} to the loss</b><div class="bp-table-wrap"><table class="bp-table"><thead><tr><th>Downstream destination</th><th>Outgoing weight</th><th>Downstream δ</th><th>Product</th></tr></thead><tbody>${terms.map(q => `<tr><th>${q.name}</th><td>${signed(q.w)}</td><td>${signed(q.d)}</td><td>${signed(q.w * q.d)}</td></tr>`).join('')}</tbody></table></div><div class="bp-equation">Sum ${mono(signed(r.g.upstream[l][j]))} × local slope ${mono(num(r.g.slopes[l][j]))} = δ ${mono(signed(t.delta))}</div></div>`;
  }
  function inspector(les, c) {
    const t = targetInfo(les.res, les.target), n = t.weights.length;
    les.input = Math.min(n - 1, Math.max(0, les.input));
    const first = t.l === 0 || (t.l === 'out' && !c.net.hidden.length), pixels = first && c.mode === 'pixels';
    const p = parameter(les.res, les.target, les.input, les.lr, les.l2), b = parameter(les.res, les.target, 'bias', les.lr, les.l2);
    const controls = `<div class="bp-inspect-controls"><label>Inspect destination <select id="bp-target">${targetOptions(c)}</select></label><label>${pixels ? 'Pixel index (1–1,024)' : 'Incoming connection'} ${pixels ? `<input id="bp-input" type="number" min="1" max="${n}" value="${les.input + 1}">` : `<select id="bp-input">${Array.from(t.input, (_, i) => `<option value="${i + 1}">${esc(inputName(t, i, c))}</option>`).join('')}</select>`}</label></div>`;
    if (les.phase === 'blame') return controls + chainDetail(les, c, t);
    const color = p.change > 0 ? 'bp-up' : p.change < 0 ? 'bp-down' : '';
    const values = `<div class="bp-factors"><div><span>Input · ${esc(inputName(t, les.input, c))}</span><b>${signed(p.input)}</b></div><i>×</i><div><span>Destination gradient δ</span><b>${signed(p.delta)}</b></div><i>=</i><div><span>Data gradient ∂L/∂w</span><b>${signed(p.gradient)}</b></div></div>`;
    const update = les.phase !== 'gradient' ? `<div class="bp-table-wrap"><table class="bp-table"><caption>Exact update for the selected weight and its destination’s bias</caption><thead><tr><th>Parameter</th><th>Before</th><th>Data gradient</th><th>Decay λw</th><th>Total gradient</th><th>Change −ηg</th><th>${les.applied ? 'After update' : 'Proposed after'}</th></tr></thead><tbody>${[[`Weight · ${inputName(t, les.input, c)}`, p], ['Bias · input = 1', b]].map(([label, v]) => `<tr><th>${esc(label)}</th><td>${signed(v.before)}</td><td>${signed(v.gradient)}</td><td>${signed(v.decayGradient)}</td><td>${signed(v.totalGradient)}</td><td class="${v.change > 0 ? 'bp-up' : v.change < 0 ? 'bp-down' : ''}">${signed(v.change)}</td><td>${signed(v.after)}</td></tr>`).join('')}</tbody></table></div><p class="bp-equation">${mono(signed(p.before))} − ${les.lr} × (${mono(signed(p.gradient))} + ${mono(signed(p.decayGradient))}) = ${mono(signed(p.after))}</p><p class="small ${color}">Selected weight ${p.change > 0 ? 'increases ↑' : p.change < 0 ? 'decreases ↓' : 'does not change'} by ${signed(p.change)}. Orange = an increase; blue = a decrease. These colors show the update, not the sign of the old weight.</p>` : `<p class="bp-equation">Bias data gradient = 1 × δ = ${mono(signed(b.gradient))}</p>`;
    const all = pixels ? `<div class="bp-pixels" id="bp-pixels"></div>` : `<details class="bp-all"><summary>Compare all ${n} incoming weights</summary><div class="bp-table-wrap"><table class="bp-table"><thead><tr><th>Input</th><th>Input value</th><th>Data gradient</th>${les.phase !== 'gradient' ? '<th>Weight change</th>' : ''}</tr></thead><tbody>${Array.from(t.input, (x, i) => { const q = parameter(les.res, les.target, i, les.lr, les.l2); return `<tr><th><button type="button" data-bp-input="${i}">${esc(inputName(t, i, c))}</button></th><td>${signed(x)}</td><td>${signed(q.gradient)}</td>${les.phase !== 'gradient' ? `<td>${signed(q.change)}</td>` : ''}</tr>`; }).join('')}</tbody></table></div></details>`;
    return controls + values + update + all;
  }
  function drawMaps(les, c) {
    const box = document.getElementById('bp-pixels'); if (!box) return;
    const t = targetInfo(les.res, les.target), n = t.input.length;
    const data = new Float64Array(n), decay = new Float64Array(n), total = new Float64Array(n), after = new Float64Array(n);
    for (let i = 0; i < n; i++) { const p = parameter(les.res, les.target, i, les.lr, les.l2); data[i] = -les.lr * p.gradient; decay[i] = -les.lr * p.decayGradient; total[i] = p.change; after[i] = p.after; }
    const max = a => Math.max(0, ...a.map(Math.abs));
    const deltaScale = Math.max(max(data), max(decay), max(total)), weightScale = Math.max(max(t.weights), max(after));
    const maps = les.phase === 'gradient' ? [['Standardized input', t.input, max(t.input)], ['Data gradient = input × δ', t.gradients, max(t.gradients)]] : [['Standardized input', t.input, max(t.input)], ['Weight before', t.weights, weightScale], ['Data change −η × input × δ', data, deltaScale], ['Decay change −ηλw', decay, deltaScale], ['Total change = data + decay', total, deltaScale], [les.applied ? 'Weight after' : 'Proposed weight after', after, weightScale]];
    box.innerHTML = `<p class="small">Each square is the same pixel position across all maps. Click a pixel, or use arrow keys on a map, to inspect its arithmetic. ${les.phase === 'gradient' ? 'Input and gradient use separately labeled scales.' : 'All change maps share one scale; before/after weights share another.'} Values near zero are pale. These are signed parameter calculations, not a saliency map.</p><div class="bp-map-grid">${maps.map(([name, a, scale], k) => `<figure><figcaption>${name}</figcaption><canvas width="192" height="192" tabindex="0" data-bp-map="${k}" aria-label="${name}; select a pixel with arrow keys"></canvas><span class="small">${signed(a[les.input])} at selected pixel<br>scale ±${num(scale)}</span></figure>`).join('')}</div>`;
    const style = getComputedStyle(document.documentElement), get = key => style.getPropertyValue(key).trim();
    maps.forEach(([, a, scale], k) => {
      const canvas = box.querySelector(`[data-bp-map="${k}"]`), ctx = canvas.getContext('2d'), cell = canvas.width / c.size;
      ctx.fillStyle = get('--surface'); ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < n; i++) { ctx.globalAlpha = scale ? Math.abs(a[i]) / scale : 0; ctx.fillStyle = get(a[i] >= 0 ? '--irregular' : '--regular'); ctx.fillRect(i % c.size * cell, Math.floor(i / c.size) * cell, cell, cell); }
      ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = get('--ink'); ctx.strokeRect(les.input % c.size * cell, Math.floor(les.input / c.size) * cell, cell, cell);
    });
  }
  function render(les, c) {
    const el = document.getElementById('lesson-details');
    el.hidden = !les; if (!les) return;
    el.innerHTML = metrics(les, c) + `<div class="bp-explanation">${explanation(les, c)}</div>` + (['blame', 'gradient', 'nudge', 'after'].includes(les.phase) ? `<div class="bp-inspector">${inspector(les, c)}</div>` : '');
    const target = document.getElementById('bp-target'), inp = document.getElementById('bp-input');
    if (target) target.value = les.target;
    if (inp) inp.value = les.input + 1;
    drawMaps(les, c);
  }
  function summary(les, c) {
    const r = les.res, t = targetInfo(r, les.target), p = parameter(r, les.target, les.input, les.lr, les.l2);
    if (les.phase === 'forward') return `Predict: the inputs travel forward through weighted sums and activations. Score ${signed(r.fw.z)} → P(${c.positiveName}) = ${num(r.fw.p)}. The known label is ${les.y} (${c.truthName}).`;
    if (les.phase === 'error') return `Loss = ${num(r.loss)}. Output gradient δ = p − y = ${num(r.fw.p)} − ${les.y} = ${signed(r.error)}. ${r.error > 0 ? 'Reducing' : 'Increasing'} the score would reduce this case’s loss.`;
    if (les.phase === 'blame') return t.l === 'out' ? `Output δ = ${signed(r.error)}. Select a hidden unit below to see how the chain rule carries this gradient backward.` : `${t.name}: sum of weighted downstream gradients ${signed(r.g.upstream[t.l][t.j])} × activation slope ${num(r.g.slopes[t.l][t.j])} = δ ${signed(t.delta)}. Follow the violet arrows from right to left; inspect every path below.`;
    if (les.phase === 'gradient') return `${inputName(t, les.input, c)} → ${t.name}: input ${signed(p.input)} × destination δ ${signed(p.delta)} = weight gradient ${signed(p.gradient)}. A bias uses input 1. Inspect another connection or pixel below.`;
    if (les.phase === 'nudge') return `${t.name}, selected weight: ${signed(p.before)} − ${les.lr} × (${signed(p.gradient)} data + ${signed(p.decayGradient)} decay) = ${signed(p.after)}. ${les.applied ? 'This update was already applied.' : 'Press Apply one update when ready; all parameters change together.'}`;
    return `One update applied: P(${c.positiveName}) ${num(les.pBefore)} → ${num(les.pAfter)}; case loss ${num(r.loss)} → ${num(les.lossAfter)}. Back and Replay inspect the saved steps without training again.`;
  }
  return { phases, targetInfo, parameter, bestInput, render, summary, num, signed };
});
