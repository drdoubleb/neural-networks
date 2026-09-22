/*
 * app.js — wires the dataset, the network and the drawings into the three-stage bench.
 */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const NF = window.NucleusFeatures, DS = window.NucleusDataset, NN = window.TinyNet, Viz = window.Viz;
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ state
  const S = {
    ds: null, size: 32,
    mode: 'features', hidden: 0, activation: 'sigmoid', lr: 0.1, batch: 8, epochs: 60, speed: 4, seed: 1, augment: false, l2: 0, peek: false,
    inputs: null, inputCache: new Map(), net: null,
    epoch: 0, ptr: 0, order: [], history: [], running: false, debt: 0, lastTime: 0, lastBatch: new Set(),
    trainEval: null, testEval: null,
    selected: null, view: 'image', tint: true, revealTest: false, stage: 'data',
    test: { results: new Map(), next: 0, threshold: 0.5, animating: false, revealed: new Set() },
    hover: { train: null, test: null },
  };
  const thumbs = { data: new Map(), train: new Map(), test: new Map() }; // id -> element

  const RECIPES = {
    1: { mode: 'features', hidden: 0, activation: 'sigmoid', lr: 0.1, batch: 8, epochs: 60, augment: false, l2: 0, peek: false, speed: 4 },
    2: { mode: 'features', hidden: 4, activation: 'tanh', lr: 0.1, batch: 8, epochs: 60, augment: false, l2: 0, peek: false, speed: 4 },
    3: { mode: 'pixels', hidden: 0, activation: 'sigmoid', lr: 0.01, batch: 8, epochs: 60, augment: false, l2: 0, peek: true, speed: 4 },
    4: { mode: 'pixels', hidden: 8, activation: 'relu', lr: 0.02, batch: 8, epochs: 60, augment: true, l2: 0.02, peek: true, speed: 6 },
  };

  // ------------------------------------------------------------------ helpers
  const fmtP = p => p.toFixed(2);
  const pct = v => Math.round(v * 100) + '%';
  function lrFromSlider(v) { return +(Math.pow(10, -3 + 3.5 * v / 100)).toPrecision(2); }
  function sliderFromLr(lr) { return Math.round((Math.log10(lr) + 3) / 3.5 * 100); }
  function speedFromSlider(v) { return +(Math.pow(10, -0.6 + 2.3 * v / 100)).toPrecision(2); } // 0.25 .. 50 epochs/s
  function sliderFromSpeed(s) { return Math.round((Math.log10(s) + 0.6) / 2.3 * 100); }
  function classOf(label) { return label ? 'irregular' : 'regular'; }
  function className(label) { return label ? 'Irregular' : 'Regular'; }
  function truthKnown(s) { return s.split === 'train' || S.revealTest || S.test.revealed.has(s.id); }
  function isClassified(s) { return s.split === 'train' || S.test.results.has(s.id); }

  // ------------------------------------------------------------------ model lifecycle
  function inputsFor(mode, augment) {
    const key = mode + (mode === 'pixels' && augment ? '+aug' : '');
    if (!S.inputCache.has(key)) S.inputCache.set(key, DS.buildInputs(S.ds, mode, { augment }));
    return S.inputCache.get(key);
  }
  function resetModel(reason) {
    stopTraining();
    S.inputs = inputsFor(S.mode, S.augment);
    S.net = new NN.TinyNet({ inputSize: S.inputs.inputSize, hidden: S.hidden, activation: S.activation, seed: S.seed });
    S.epoch = 0; S.ptr = 0; S.debt = 0; S.history = []; S.lastBatch = new Set();
    S.rng = NN.mulberry32(S.seed * 31 + 7);
    S.order = S.inputs.trainX.map((_, i) => i);
    clearTestResults(false);
    recordEpoch();
    renderTraining(true);
    if (reason) note(reason);
  }
  function shuffleOrder() {
    const o = S.order;
    for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(S.rng() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; }
  }
  function evaluateAll() {
    const tr = S.ds.train, te = S.ds.test;
    S.trainEval = S.net.evaluate(tr.map(s => S.inputs.xOf(s)), tr.map(s => s.label));
    S.testEval = S.net.evaluate(te.map(s => S.inputs.xOf(s)), te.map(s => s.label));
  }
  function recordEpoch() {
    evaluateAll();
    S.history.push({ epoch: S.epoch, loss: S.trainEval.loss, acc: S.trainEval.accuracy, testLoss: S.testEval.loss, testAcc: S.testEval.accuracy });
  }
  function trainStep() {
    const X = S.inputs.trainX, Y = S.inputs.trainY, n = X.length;
    if (S.ptr === 0) shuffleOrder();
    const end = Math.min(n, S.ptr + S.batch);
    const idx = S.order.slice(S.ptr, end);
    S.net.trainBatch(idx.map(i => X[i]), idx.map(i => Y[i]), S.lr, S.l2);
    S.lastBatch = new Set(idx.map(i => S.inputs.trainOwner[i]));
    S.ptr = end >= n ? 0 : end;
    if (S.ptr === 0) { S.epoch++; recordEpoch(); }
    if (S.test.results.size) clearTestResults(true);
  }
  function batchesPerEpoch() { return Math.ceil(S.inputs.trainX.length / S.batch); }

  function startTraining() {
    if (S.epoch >= S.epochs) { note(`Already at ${S.epochs} epochs. Raise the epoch count or reset to train again.`); return; }
    S.running = true; S.lastTime = performance.now(); S.debt = 0;
    $('btn-train').textContent = '⏸ Pause';
    $('btn-train').classList.add('is-running');
    requestAnimationFrame(tick);
  }
  function stopTraining(msg) {
    S.running = false;
    const b = $('btn-train'); if (b) { b.textContent = '▶ Train'; b.classList.remove('is-running'); }
    if (msg) note(msg);
  }
  function tick(now) {
    if (!S.running) return;
    const dt = Math.min(0.1, (now - S.lastTime) / 1000); S.lastTime = now;
    const bpe = batchesPerEpoch();
    S.debt += dt * S.speed * bpe;
    const t0 = performance.now(); let did = false;
    while (S.debt >= 1 && performance.now() - t0 < 16) {
      trainStep(); S.debt -= 1; did = true;
      if (S.ptr === 0 && S.epoch >= S.epochs) { stopTraining(`Finished ${S.epochs} epochs. Train accuracy ${pct(S.trainEval.accuracy)}. Head to 3 · Test.`); break; }
    }
    if (S.debt > bpe) S.debt = bpe;
    if (did) renderTraining();
    if (S.running) requestAnimationFrame(tick);
  }
  function stepBatch() { stopTraining(); trainStep(); renderTraining(); if (S.ptr === 0) note(`Epoch ${S.epoch} complete.`); }
  function stepEpoch() { stopTraining(); do { trainStep(); } while (S.ptr !== 0); renderTraining(); note(`Epoch ${S.epoch} complete.`); }

  // ------------------------------------------------------------------ test phase
  function clearTestResults(notify) {
    S.test.results.clear(); S.test.next = 0; S.test.revealed.clear(); S.test.animating = false;
    if (notify) $('test-note').textContent = 'The model changed, so the test results were cleared. Classify again to score the new weights.';
    renderTestPanel();
  }
  function classifyNext(quiet) {
    if (S.test.animating || S.test.next >= S.ds.test.length) return false;
    const s = S.ds.test[S.test.next++];
    const fw = S.net.forward(S.inputs.xOf(s));
    S.test.results.set(s.id, { p: fw.p });
    $('test-note').textContent = '';
    selectSpecimen(s, { silent: true });
    if (quiet || reducedMotion) { S.test.revealed.add(s.id); renderTestPanel(); renderInspector(); renderTestGraph(); return true; }
    S.test.animating = true;
    const stages = [0, 1, 2];
    let k = 0;
    const run = () => {
      renderTestGraph(stages[k]);
      k++;
      if (k < stages.length) setTimeout(run, 260);
      else setTimeout(() => { S.test.revealed.add(s.id); S.test.animating = false; renderTestPanel(); renderInspector(); renderTestGraph(); }, 420);
    };
    renderTestPanel(); renderInspector();
    run();
    return true;
  }
  function classifyAll() {
    if (S.test.animating) return;
    const go = () => { if (S.test.next < S.ds.test.length) { classifyNext(true); setTimeout(go, reducedMotion ? 0 : 90); } };
    go();
  }
  function testStats() {
    const thr = S.test.threshold;
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const s of S.ds.test) {
      const r = S.test.results.get(s.id); if (!r || !S.test.revealed.has(s.id)) continue;
      const call = r.p >= thr ? 1 : 0;
      if (call && s.label) tp++; else if (!call && !s.label) tn++; else if (call && !s.label) fp++; else fn++;
    }
    const n = tp + tn + fp + fn;
    return { tp, tn, fp, fn, n, acc: n ? (tp + tn) / n : null, sens: tp + fn ? tp / (tp + fn) : null, spec: tn + fp ? tn / (tn + fp) : null };
  }

  // ------------------------------------------------------------------ rendering: trays
  function makeThumb(s, map, named) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'thumb'; b.dataset.id = s.id;
    b.setAttribute('aria-label', `Specimen ${s.name}`);
    const cv = document.createElement('canvas'); cv.width = s.size; cv.height = s.size;
    b.appendChild(cv);
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = '✗'; b.appendChild(badge);
    if (named) { const nm = document.createElement('span'); nm.className = 'name'; nm.textContent = s.name; b.appendChild(nm); }
    b.addEventListener('click', () => selectSpecimen(s));
    Viz.renderThumb(cv, s.px, s.size, S.tint);
    map.set(s.id, b);
    return b;
  }
  function buildTrays() {
    const dt = $('data-train-tray'), dx = $('data-test-tray'), tt = $('train-tray'), xt = $('test-tray');
    for (const s of S.ds.train) { dt.appendChild(makeThumb(s, thumbs.data, true)); tt.appendChild(makeThumb(s, thumbs.train, false)); }
    for (const s of S.ds.test) { dx.appendChild(makeThumb(s, thumbs.data, true)); xt.appendChild(makeThumb(s, thumbs.test, true)); }
  }
  function repaintThumbs() {
    for (const map of Object.values(thumbs)) for (const [id, el] of map) {
      const s = S.ds.specimens[id];
      Viz.renderThumb(el.querySelector('canvas'), s.px, s.size, S.tint);
    }
  }
  function setThumbState(el, { call, truth, wrong, right, unknown, inBatch, selected, q, title }) {
    el.className = 'thumb' + (call != null ? ` call-${call}` : '') + (truth != null ? ` truth-${truth}` : '') + (wrong ? ' wrong' : '') + (right ? ' right' : '') + (unknown ? ' unknown' : '') + (inBatch ? ' in-batch' : '') + (selected ? ' selected' : '');
    const badge = el.querySelector('.badge');
    badge.className = 'badge' + (q ? ' q' : '');
    badge.textContent = q ? '?' : (right ? '✓' : '✗');
    if (title) el.title = title;
  }
  function renderDataTrays() {
    for (const s of S.ds.specimens) {
      const el = thumbs.data.get(s.id); if (!el) continue;
      const known = truthKnown(s);
      setThumbState(el, { truth: known ? s.label : null, unknown: !known, selected: S.selected && S.selected.id === s.id,
        title: `${s.name} · ${s.split === 'train' ? 'training' : 'test'} set${known ? ' · ' + className(s.label) : ''}` });
    }
  }
  function renderTrainTray() {
    const probs = S.trainEval.probs;
    S.ds.train.forEach((s, k) => {
      const el = thumbs.train.get(s.id);
      const p = probs[k], call = p >= 0.5 ? 1 : 0;
      setThumbState(el, { call, wrong: call !== s.label, inBatch: S.lastBatch.has(s.id), selected: S.selected && S.selected.id === s.id,
        title: `${s.name} · truth ${className(s.label)} · call ${className(call)} (P ${fmtP(p)})` });
    });
  }
  function renderTestTray() {
    const thr = S.test.threshold;
    for (const s of S.ds.test) {
      const el = thumbs.test.get(s.id);
      const r = S.test.results.get(s.id), shown = r && S.test.revealed.has(s.id);
      if (!shown) setThumbState(el, { unknown: true, q: true, selected: S.selected && S.selected.id === s.id, title: `${s.name} · not classified yet` });
      else {
        const call = r.p >= thr ? 1 : 0;
        setThumbState(el, { call, wrong: call !== s.label, right: call === s.label, selected: S.selected && S.selected.id === s.id,
          title: `${s.name} · call ${className(call)} (P ${fmtP(r.p)}) · truth ${className(s.label)}` });
      }
    }
  }

  // ------------------------------------------------------------------ rendering: train panel
  function note(msg) { $('note').textContent = msg || ''; }
  function renderStatus() {
    const bpe = batchesPerEpoch();
    const bi = S.ptr === 0 ? bpe : Math.ceil(S.ptr / S.batch);
    const h = S.history[S.history.length - 1];
    $('status').innerHTML =
      `<span>epoch <b>${S.epoch}</b> / ${S.epochs}</span>` +
      `<span>batch <b>${S.ptr === 0 ? '–' : bi}</b> / ${bpe}</span>` +
      `<span>loss <b>${S.trainEval.loss.toFixed(3)}</b></span>` +
      `<span>train accuracy <b>${pct(S.trainEval.accuracy)}</b></span>` +
      (S.peek ? `<span>test accuracy <b>${pct(S.testEval.accuracy)}</b> (peeking)</span>` : '') +
      `<span>parameters <b>${S.net.parameterCount().toLocaleString()}</b></span>` +
      `<span>training images <b>${S.inputs.trainX.length}</b>${S.augment && S.mode === 'pixels' ? ' (80 × 8 orientations)' : ''}</span>`;
    void h;
  }
  function graphModel(spec, stage, hover) {
    const s = spec || null;
    let x = null, fw = null;
    if (s) {
      x = S.inputs.xOf(s);
      const allowed = isClassified(s) || s.split === 'train';
      fw = S.net.forward(x);
      if (!allowed) { fw = null; stage = 0; }
    }
    return { net: S.net, mode: S.mode, x, fw, featureNames: NF.FEATURES.map(f => f.name), specimen: s, size: S.size, tint: S.tint, stage, hover, activation: S.activation, activationLabel: NN.ACTIVATIONS[S.activation].label };
  }
  function renderTrainGraph() {
    const cv = $('net-canvas');
    const m = graphModel(S.selected && S.selected.split === 'train' ? S.selected : (S.selected || null), 2, S.hover.train);
    cv._model = m;
    Viz.drawNetwork(cv, m);
  }
  function renderTestGraph(stage) {
    const cv = $('net-canvas-test');
    const s = S.selected;
    let st = 2;
    if (s && s.split === 'test') st = S.test.revealed.has(s.id) ? 2 : (stage == null ? 0 : stage);
    if (s && s.split === 'test' && S.test.results.has(s.id) && stage != null) st = stage;
    const m = graphModel(s, st, S.hover.test);
    if (s && s.split === 'test' && S.test.results.has(s.id) && !S.test.revealed.has(s.id)) { m.fw = S.net.forward(m.x); }
    cv._model = m;
    Viz.drawNetwork(cv, m);
  }
  function renderCharts() {
    Viz.drawCurves($('chart-loss'), { history: S.history, key: 'loss', showTest: S.peek, maxEpoch: S.epochs });
    Viz.drawCurves($('chart-acc'), { history: S.history, key: 'acc', showTest: S.peek, maxEpoch: S.epochs });
    $('loss-now').textContent = S.trainEval.loss.toFixed(3);
    $('acc-now').textContent = pct(S.trainEval.accuracy);
    $('curves-legend-test').hidden = !S.peek;
  }
  function renderTraining(full) {
    if (!S.trainEval || full) evaluateAll();
    else evaluateAll();
    renderStatus();
    if (S.stage === 'train') { renderTrainTray(); renderTrainGraph(); renderCharts(); }
    if (S.stage === 'test') { renderTestPanel(); renderTestGraph(); }
    renderInspector();
  }

  // ------------------------------------------------------------------ rendering: test panel
  function renderTestPanel() {
    renderTestTray();
    const st = testStats();
    $('stat-n').textContent = `${st.n} / ${S.ds.test.length}`;
    $('stat-acc').textContent = st.acc == null ? '–' : pct(st.acc);
    $('stat-sens').textContent = st.sens == null ? '–' : pct(st.sens);
    $('stat-spec').textContent = st.spec == null ? '–' : pct(st.spec);
    $('stat-acc-sub').textContent = st.n ? `${st.tp + st.tn} of ${st.n} correct` : 'no calls yet';
    $('stat-sens-sub').textContent = st.tp + st.fn ? `${st.tp} of ${st.tp + st.fn} irregular caught` : 'no irregular seen yet';
    $('stat-spec-sub').textContent = st.tn + st.fp ? `${st.tn} of ${st.tn + st.fp} regular cleared` : 'no regular seen yet';
    const cell = (v, kind) => `<div class="cell ${v ? kind : 'empty'}">${v}</div>`;
    $('confusion').innerHTML =
      `<div></div><div class="hd">called regular</div><div class="hd">called irregular</div>` +
      `<div class="rh">truth regular</div>${cell(st.tn, 'hit')}${cell(st.fp, 'miss')}` +
      `<div class="rh">truth irregular</div>${cell(st.fn, 'miss')}${cell(st.tp, 'hit')}`;
    $('btn-classify-next').disabled = S.test.next >= S.ds.test.length;
    $('btn-classify-all').disabled = S.test.next >= S.ds.test.length;
    $('btn-classify-next').textContent = S.test.next >= S.ds.test.length ? 'All 20 classified' : `Classify next (${S.test.next + 1} of ${S.ds.test.length})`;
    $('threshold-val').textContent = S.test.threshold.toFixed(2);
    const untrained = S.epoch === 0 && S.net.steps === 0;
    $('test-warning').hidden = !untrained;
  }

  // ------------------------------------------------------------------ rendering: inspector
  function selectSpecimen(s, opts) {
    S.selected = s;
    renderDataTrays();
    if (S.stage === 'train') { renderTrainTray(); renderTrainGraph(); }
    if (S.stage === 'test') { renderTestTray(); if (!(opts && opts.silent)) renderTestGraph(); }
    if (S.stage === 'data') renderScatter();
    renderInspector();
  }
  function renderInspector() {
    const s = S.selected;
    const cv = $('spec-view');
    if (!s) return;
    $('spec-name').textContent = `Specimen ${s.name}`;
    const known = truthKnown(s);
    $('spec-chips').innerHTML =
      `<span class="chip plain">${s.split === 'train' ? 'Training set' : 'Test set · held out'}</span>` +
      (known ? `<span class="chip ${classOf(s.label)}">truth: ${className(s.label)}</span>` : `<span class="chip plain">truth hidden</span>`);
    document.querySelectorAll('.views button').forEach(b => b.classList.toggle('is-active', b.dataset.view === S.view));
    const m = s.measurement;
    const x = S.inputs.xOf(s);
    const classified = isClassified(s) && (s.split === 'train' || S.test.revealed.has(s.id));
    const untrainedHere = S.stage === 'data' && S.net.steps === 0;
    const fw = classified && !untrainedHere ? S.net.forward(x) : null;
    let caption = '';
    if (S.view === 'image') { Viz.renderBigImage(cv, s.px, s.size, S.tint); caption = `${s.size} × ${s.size} pixels, 8-bit grayscale${S.tint ? ', shown with an H&E tint' : ''}.`; }
    else if (S.view === 'measure') { Viz.renderMeasurement(cv, s.px, s.size, m, S.tint); caption = 'Violet: membrane found by thresholding · white dashes: convex hull · orange dots: smooth ellipse with the same area · shaded: where darkness and texture are read.'; }
    else {
      if (!fw) { Viz.renderBigImage(cv, s.px, s.size, S.tint); caption = untrainedHere ? 'Evidence appears once the network has trained (stage 2).' : 'Evidence appears once the network has classified this nucleus.'; }
      else if (S.mode === 'pixels') {
        const g = S.net.inputGradient(x, fw);
        const sal = new Float64Array(g.length);
        for (let i = 0; i < g.length; i++) sal[i] = g[i] * x[i];
        Viz.renderEvidence(cv, s.px, s.size, sal, S.tint);
        caption = 'Orange pixels push the score toward Irregular, blue toward Regular (weight × input at each pixel).';
      } else { Viz.renderMeasurement(cv, s.px, s.size, m, S.tint); caption = 'In measurement mode the evidence is per measurement — see the “push” column below.'; }
    }
    $('spec-caption').textContent = caption;

    // verdict
    const v = $('verdict');
    if (!fw) {
      v.querySelector('.p').innerHTML = untrainedHere ? '<small class="lbl">network call</small><b class="soft">not trained yet</b><small>see stage 2</small>' : (s.split === 'test' ? '<small class="lbl">network call</small><b class="soft">pending</b><small>press Classify next</small>' : '<small class="lbl">network call</small><b class="soft">–</b>');
      $('verdict-call').innerHTML = '';
      $('pbar-marker').style.left = '50%';
      $('evidence-sum').textContent = '';
    } else {
      const thr = s.split === 'test' ? S.test.threshold : 0.5;
      const call = fw.p >= thr ? 1 : 0;
      v.querySelector('.p').innerHTML = `<small class="lbl">P(irregular)</small><b>${fw.p.toFixed(3)}</b><small>score z = ${Viz.fmtSigned(fw.z, 2)}</small>`;
      let chips = `<span class="chip ${classOf(call)}">call: ${className(call)}</span>`;
      if (known) chips += call === s.label ? `<span class="chip good">✓ agrees with truth</span>` : `<span class="chip bad">✗ truth is ${className(s.label)}</span>`;
      $('verdict-call').innerHTML = chips;
      $('pbar-marker').style.left = (fw.p * 100).toFixed(1) + '%';
      $('pbar-thr').style.left = (thr * 100).toFixed(1) + '%';
      const g = S.net.inputGradient(x, fw);
      let sum = 0; for (let i = 0; i < g.length; i++) sum += g[i] * x[i];
      const exact = S.net.H === 0;
      $('evidence-sum').innerHTML = `evidence Σ(weight × input) ${exact ? '=' : '≈'} ${Viz.fmtSigned(sum, 2)} &nbsp;·&nbsp; bias ${Viz.fmtSigned(S.net.b2, 2)} &nbsp;→&nbsp; z ${Viz.fmtSigned(fw.z, 2)} &nbsp;→&nbsp; P = 1 / (1 + e<sup>−z</sup>) = ${fw.p.toFixed(3)}${exact ? '' : '<br><span class="muted">(with a hidden layer the per-input evidence is a linear approximation)</span>'}`;
    }

    // measurement table
    const tb = $('feat-table');
    const featMode = S.mode === 'features';
    const g = fw && featMode ? S.net.inputGradient(x, fw) : null;
    let maxPush = 1e-9;
    const pushes = NF.FEATURES.map((f, i) => (g ? g[i] * x[i] : 0));
    for (const p of pushes) maxPush = Math.max(maxPush, Math.abs(p));
    const zf = featMode ? x : inputsFor('features', false).xOf(s);
    tb.innerHTML = NF.FEATURES.map((f, i) => {
      const val = m.vector[i];
      const z = zf[i];
      const push = pushes[i];
      const w = Math.min(50, Math.abs(push) / maxPush * 50);
      const bar = featMode ? `<div class="bar" title="push ${Viz.fmtSigned(push, 2)}">${g ? `<i class="${push >= 0 ? 'pos' : 'neg'}" style="width:${w.toFixed(1)}%"></i>` : ''}</div>` : '';
      return `<tr><td class="name" title="${f.desc.replace(/"/g, '&quot;')}">${f.name}</td><td class="n">${f.fmt(val)}${f.unit ? ' ' + f.unit : ''}</td><td class="n">${Viz.fmtSigned(z, 1)}</td>${featMode ? `<td>${bar}</td>` : ''}</tr>`;
    }).join('');
    $('feat-push-head').hidden = !featMode;
    $('feat-note').textContent = featMode ? 'push = weight × standardized value: how far this measurement moves the score (orange → Irregular, blue → Regular).' : 'In pixel mode the network never sees these measurements — they are here for you, the human.';
  }

  // ------------------------------------------------------------------ data panel
  function renderScatter() {
    const xi = +$('scatter-x').value, yi = +$('scatter-y').value;
    const pts = S.ds.specimens.filter(s => s.split === 'train' || S.revealTest).map(s => ({
      id: s.id, name: s.name, x: s.features[xi], y: s.features[yi], split: s.split,
      cls: truthKnown(s) ? classOf(s.label) : 'unknown', selected: S.selected && S.selected.id === s.id,
    }));
    Viz.drawScatter($('scatter'), { points: pts, xLabel: NF.FEATURES[xi].name, yLabel: NF.FEATURES[yi].name, onSelect: id => selectSpecimen(S.ds.specimens[id]) });
  }

  // ------------------------------------------------------------------ controls
  function syncControls() {
    document.querySelectorAll('#mode-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.mode === S.mode));
    $('hidden').value = S.hidden; $('hidden-val').textContent = S.hidden === 0 ? 'none' : S.hidden;
    $('activation').value = S.activation; $('activation').disabled = S.hidden === 0;
    $('lr').value = sliderFromLr(S.lr); $('lr-val').textContent = S.lr;
    $('batch').value = S.batch;
    $('epochs').value = S.epochs; $('epochs-val').textContent = S.epochs;
    $('speed').value = sliderFromSpeed(S.speed); $('speed-val').textContent = `${S.speed} epochs/s`;
    $('seed').value = S.seed;
    $('augment').checked = S.augment; $('augment').disabled = S.mode !== 'pixels';
    $('augment-wrap').classList.toggle('muted', S.mode !== 'pixels');
    $('l2').value = S.l2; $('l2-val').textContent = S.l2 === 0 ? 'off' : S.l2.toFixed(2);
    $('peek').checked = S.peek;
    $('threshold').value = S.test.threshold;
  }
  function bindControls() {
    document.querySelectorAll('#mode-seg button').forEach(b => b.addEventListener('click', () => {
      if (S.mode === b.dataset.mode) return;
      S.mode = b.dataset.mode;
      if (S.mode === 'pixels' && S.lr > 0.05) S.lr = 0.01;
      if (S.mode === 'features' && S.lr < 0.05) S.lr = 0.1;
      syncControls(); resetModel(`Input changed to ${S.mode === 'pixels' ? 'raw pixels' : 'measurements'} — fresh random weights.`);
    }));
    $('hidden').addEventListener('input', () => { S.hidden = +$('hidden').value; syncControls(); resetModel(`Hidden layer set to ${S.hidden || 'none'} — fresh random weights.`); });
    $('activation').addEventListener('change', () => { S.activation = $('activation').value; resetModel('Activation changed — fresh random weights.'); });
    $('lr').addEventListener('input', () => { S.lr = lrFromSlider(+$('lr').value); $('lr-val').textContent = S.lr; });
    $('batch').addEventListener('change', () => { S.batch = +$('batch').value; S.ptr = 0; renderStatus(); });
    $('epochs').addEventListener('input', () => { S.epochs = +$('epochs').value; $('epochs-val').textContent = S.epochs; renderStatus(); if (S.stage === 'train') renderCharts(); });
    $('speed').addEventListener('input', () => { S.speed = speedFromSlider(+$('speed').value); $('speed-val').textContent = `${S.speed} epochs/s`; });
    $('seed').addEventListener('change', () => { S.seed = Math.max(1, Math.floor(+$('seed').value || 1)); syncControls(); resetModel(`Seed ${S.seed} — fresh random weights.`); });
    $('seed-random').addEventListener('click', () => { S.seed = 1 + Math.floor(Math.random() * 9999); syncControls(); resetModel(`Seed ${S.seed} — fresh random weights.`); });
    $('augment').addEventListener('change', () => { S.augment = $('augment').checked; resetModel(S.augment ? 'Training set augmented with flips and rotations (80 × 8 = 640 views) — fresh random weights.' : 'Augmentation off — fresh random weights.'); });
    $('l2').addEventListener('input', () => { S.l2 = +$('l2').value; $('l2-val').textContent = S.l2 === 0 ? 'off' : S.l2.toFixed(2); });
    $('peek').addEventListener('change', () => { S.peek = $('peek').checked; renderStatus(); if (S.stage === 'train') renderCharts(); });
    $('btn-train').addEventListener('click', () => (S.running ? stopTraining('Paused.') : startTraining()));
    $('btn-step-batch').addEventListener('click', stepBatch);
    $('btn-step-epoch').addEventListener('click', stepEpoch);
    $('btn-reset').addEventListener('click', () => resetModel('Weights re-initialised from the seed.'));
    document.querySelectorAll('.recipe').forEach(b => b.addEventListener('click', () => {
      Object.assign(S, RECIPES[b.dataset.recipe]);
      syncControls(); resetModel(`Recipe loaded: ${b.textContent.trim()}. Press Train.`);
    }));
    document.querySelectorAll('.views button').forEach(b => b.addEventListener('click', () => { S.view = b.dataset.view; renderInspector(); }));
    $('reveal-test').addEventListener('change', () => { S.revealTest = $('reveal-test').checked; renderDataTrays(); renderScatter(); renderInspector(); });
    $('scatter-x').addEventListener('change', renderScatter);
    $('scatter-y').addEventListener('change', renderScatter);
    $('btn-classify-next').addEventListener('click', () => classifyNext(false));
    $('btn-classify-all').addEventListener('click', classifyAll);
    $('btn-test-reset').addEventListener('click', () => { clearTestResults(false); $('test-note').textContent = 'Test results cleared.'; renderInspector(); renderTestGraph(); });
    $('threshold').addEventListener('input', () => { S.test.threshold = +$('threshold').value; renderTestPanel(); renderInspector(); });
    $('tint').addEventListener('change', () => { S.tint = $('tint').checked; repaintThumbs(); renderInspector(); if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
    $('theme-toggle').addEventListener('click', () => {
      const root = document.documentElement;
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('nucleus-net-theme', root.dataset.theme); } catch (e) { /* ignore */ }
      onThemeChange();
    });
    document.querySelectorAll('.stage').forEach(b => b.addEventListener('click', () => showStage(b.dataset.stage)));
    // hover tooltips on the network diagrams
    for (const [key, cvId, tipId] of [['train', 'net-canvas', 'net-tip'], ['test', 'net-canvas-test', 'net-tip-test']]) {
      const cv = $(cvId), tip = $(tipId);
      cv.addEventListener('mousemove', ev => {
        const m = cv._model; if (!m) return;
        const r = cv.getBoundingClientRect();
        const hit = Viz.hitNetwork(cv, ev.clientX - r.left, ev.clientY - r.top, m);
        const prev = S.hover[key];
        S.hover[key] = hit;
        if (hit) { tip.hidden = false; tip.textContent = hit.text; tip.style.left = (ev.clientX - r.left) + 'px'; tip.style.top = (ev.clientY - r.top) + 'px'; }
        else tip.hidden = true;
        if ((prev && prev.ref) !== (hit && hit.ref)) { key === 'train' ? renderTrainGraph() : renderTestGraph(); }
      });
      cv.addEventListener('mouseleave', () => { S.hover[key] = null; tip.hidden = true; key === 'train' ? renderTrainGraph() : renderTestGraph(); });
    }
    document.addEventListener('keydown', ev => {
      if (ev.target.matches('input, select, textarea, button')) return;
      if (ev.key === ' ' && S.stage === 'train') { ev.preventDefault(); S.running ? stopTraining('Paused.') : startTraining(); }
      else if ((ev.key === 'n' || ev.key === 'N') && S.stage === 'test') classifyNext(false);
      else if (ev.key === '1') showStage('data'); else if (ev.key === '2') showStage('train'); else if (ev.key === '3') showStage('test');
    });
    window.addEventListener('resize', () => { if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
  }
  function onThemeChange() {
    Viz.refreshTheme();
    const dark = getComputedStyle(document.documentElement).colorScheme.includes('dark');
    $('theme-toggle').textContent = dark ? '☀ Light' : '☾ Dark';
    if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph();
    renderInspector();
  }
  function showStage(name) {
    S.stage = name;
    document.querySelectorAll('.stage').forEach(b => b.classList.toggle('is-active', b.dataset.stage === name));
    $('panel-data').hidden = name !== 'data'; $('panel-train').hidden = name !== 'train'; $('panel-test').hidden = name !== 'test';
    if (name === 'data') { renderDataTrays(); renderScatter(); }
    if (name === 'train') { if (S.selected && S.selected.split !== 'train') S.selected = S.ds.train[0]; renderTraining(); }
    if (name === 'test') { stopTraining(); renderTestPanel(); if (!S.selected || S.selected.split !== 'test') { S.selected = S.ds.test[Math.max(0, S.test.next - 1)]; } renderTestGraph(); renderInspector(); renderDataTrays(); }
    try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------------ boot
  function init() {
    try { const t = localStorage.getItem('nucleus-net-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
    Viz.refreshTheme();
    S.ds = DS.prepare(window.NUCLEI_DATA);
    S.size = S.ds.size;
    $('meta-line').textContent = `${S.ds.specimens.length} synthetic nuclei · ${S.ds.train.length} training / ${S.ds.test.length} test · ${S.size} × ${S.size} px · generated with seed ${window.NUCLEI_DATA.meta.seed}`;
    const opts = NF.FEATURES.map((f, i) => `<option value="${i}">${f.name}</option>`).join('');
    $('scatter-x').innerHTML = opts; $('scatter-y').innerHTML = opts;
    $('scatter-x').value = 5; $('scatter-y').value = 4;
    $('feature-list').innerHTML = NF.FEATURES.map(f => `<li><strong>${f.name}</strong> — ${f.desc}</li>`).join('');
    buildTrays();
    bindControls();
    syncControls();
    S.selected = S.ds.train[0];
    resetModel();
    renderDataTrays(); renderScatter(); renderInspector();
    onThemeChange();
    const hash = (location.hash || '').replace('#', '');
    showStage(['data', 'train', 'test'].includes(hash) ? hash : 'data');
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', onThemeChange);
  }
  document.addEventListener('DOMContentLoaded', init);
})();
