/*
 * app.js — wires the datasets, the network and the drawings into the four-stage bench (specimens, train, test, and the
 * foundation model's pretraining).
 */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const NF = window.NucleusFeatures, DS = window.NucleusDataset, NN = window.TinyNet, Viz = window.Viz;
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ state
  const S = {
    tasks: null, taskId: 'leukemia', task: null, ds: null, kind: 'tabular', size: 0, featureDefs: [],
    mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 60, speed: 6, seed: 1, augment: false, l2: 0, peek: false,
    trainLab: 'ours', testLab: 'ours', normalize: 'off', showLab: false, // where each set's cases come from (our lab, the other lab, both), and whether the pixels are stain-normalised first
    labelNoise: 0, showFlipped: false, // the share of training cases given the wrong label, and whether they are marked
    labelled: 0, backbone: 'page', shipped: {}, // how many training cases carry a label (0 = all), and which foundation encoder feeds the code input ('page', or a shipped backbone's id)
    animSpeed: 1, // playback speed of the walk-throughs (the lesson and Classify next): 1 = the normal pace
    excluded: new Set(),
    inputs: null, inputCache: new Map(), net: null,
    epoch: 0, ptr: 0, order: [], history: [], running: false, debt: 0, lastTime: 0, lastBatch: new Set(),
    trainEval: null, testEval: null, profileAt: 0, prevW: null, applyingRecipe: false,
    selected: null, view: 'image', tint: true, revealTest: false, stage: 'data', trayColor: 'call',
    test: { results: new Map(), next: 0, threshold: 0.5, animating: false, revealed: new Set(), prevalence: 0.01, scores: {} },
    hover: { train: null, test: null },
    lesson: null, lastLesson: null, lessonCursor: 0,
    // 4 · Foundation: the miniature foundation model's pretraining (its set, encoder and run live here)
    fm: { set: null, cl: null, labs: true, epochs: 100, speed: 4, seed: 1, epoch: 0, ptr: 0, order: [], hist: [], running: false, debt: 0, lastTime: 0, lastBatch: null, selected: null, color: 'none', x: 0, y: 1, showPairs: true, hover: null },
  };
  const thumbs = { data: new Map(), train: new Map(), test: new Map() }; // id -> element

  const RECIPE_LABELS = ['', '① Leukemia · blood count · single layer', '② Leukemia · blood count · 3 ReLU units', '③ Atypia · measurements · single layer', '④ Atypia · measurements · 8 + 8 ReLU · a quarter of the training labels wrong: overfitting', '⑤ Enlargement · pixels · single layer', '⑥ Irregularity · pixels · single layer', '⑦ Irregularity · pixels · 4 ReLU + augmentation', '⑧ Irregularity · pixels · 4 + 4 ReLU + augmentation · then try the other lab', '⑨ Irregularity · pixels · 4 + 4 ReLU · the shortcut: irregular nuclei scanned at another lab', '⑩ Irregularity · pixels · convolution + 4 ReLU + augmentation', '⑪ Foundation · pretrain a code on 100 unlabelled nuclei, then see what it is worth', '⑫ Irregularity · the foundation code · single layer · 10 labelled cases'];
  const LAB_SETTINGS = { trainLab: 'ours', testLab: 'ours', normalize: 'off', labelNoise: 0, labelled: 0, seed: 1 }; // every recipe starts from our lab's scans, unnormalised, with every label as it is, from seed 1 unless it says otherwise
  const RECIPES = {
    1: { task: 'leukemia',    mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 60,  augment: false, l2: 0,    peek: false, speed: 6 },
    2: { task: 'leukemia',    mode: 'features', h1: 3, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 150, augment: false, l2: 0,    peek: false, speed: 10 },
    3: { task: 'atypia',       mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.1,  batch: 8, epochs: 60,  augment: false, l2: 0,    peek: false, speed: 4 },
    4: { task: 'atypia',       mode: 'features', h1: 8, h2: 8, convK: 0, activation: 'relu', lr: 0.1,  batch: 8, epochs: 300, augment: false, l2: 0,    peek: true,  speed: 10, labelNoise: 0.25 },
    5: { task: 'enlargement',  mode: 'pixels',   h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 30,  augment: false, l2: 0,    peek: true,  speed: 4 },
    6: { task: 'irregularity', mode: 'pixels',   h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.01, batch: 8, epochs: 60,  augment: false, l2: 0,    peek: true,  speed: 4 },
    7: { task: 'irregularity', mode: 'pixels',   h1: 4, h2: 0, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 60,  augment: true,  l2: 0.01, peek: true,  speed: 6 },
    8: { task: 'irregularity', mode: 'pixels',   h1: 4, h2: 4, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 60,  augment: true,  l2: 0.01, peek: true,  speed: 6 },
    9: { task: 'irregularity', mode: 'pixels',   h1: 4, h2: 4, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 60,  augment: true,  l2: 0.01, peek: true,  speed: 6, trainLab: 'byClass', testLab: 'byClass' },
    10: { task: 'irregularity', mode: 'pixels',  h1: 4, h2: 0, convK: 4, activation: 'relu', lr: 0.02, batch: 8, epochs: 30,  augment: true,  l2: 0,    peek: true,  speed: 4 },
    12: { task: 'irregularity', mode: 'code',    h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.1,  batch: 8, epochs: 60,  augment: false, l2: 0,    peek: true,  speed: 4, labelled: 10, seed: 8 }, // seed 8: the untrained network starts near chance (a random weighting of the code already classifies, right or wrong by the seed)
  };
  // the source modes, in the order of the selects; the class-split one names the positive class
  const SOURCE_LABELS = () => ({ ours: 'our lab', other: 'the other lab (weaker stain)', mixed: 'both labs, mixed at random', byClass: `split by class: ${posName()} from the other lab` });

  // ------------------------------------------------------------------ helpers
  const fmtP = p => p.toFixed(2);
  const pct = v => Math.round(v * 100) + '%';
  function lrFromSlider(v) { return +(Math.pow(10, -3 + 3.5 * v / 100)).toPrecision(2); }
  function sliderFromLr(lr) { return Math.round((Math.log10(lr) + 3) / 3.5 * 100); }
  function speedFromSlider(v) { return +(Math.pow(10, -0.6 + 2.3 * v / 100)).toPrecision(2); } // 0.25 .. 50 epochs/s
  function sliderFromSpeed(s) { return Math.round((Math.log10(s) + 0.6) / 2.3 * 100); }
  function prevFromSlider(v) { return Math.pow(10, -4 + 3.7 * v / 100); }                       // 1 in 10,000 .. 1 in 2
  function sliderFromPrev(p) { return Math.round((Math.log10(p) + 4) / 3.7 * 100); }
  const classOf = label => (label ? 'pos' : 'neg');
  const className = label => S.task.classes[label].name;
  const posName = () => S.task.classes[1].name, negName = () => S.task.classes[0].name;
  const noun = (n) => { const w = S.task.specimenNoun || 'specimen'; return n === 1 ? w : (w === 'nucleus' ? 'nuclei' : w + 's'); };
  function truthKnown(s) { return s.split === 'train' || S.revealTest || S.test.revealed.has(s.id); }
  function isClassified(s) { return s.split === 'train' || S.test.results.has(s.id); }
  function subtypeName(key) { const st = (S.task.subtypes || []).find(t => t.key === key); return st ? st.name : key; }
  const esc = t => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

  // ------------------------------------------------------------------ task + model lifecycle
  function loadTask(id) {
    S.taskId = id;
    const raw = S.tasks[id];
    S.task = raw.meta.task;
    S.ds = DS.prepare(raw);
    DS.assignLabs(S.ds, S.trainLab, S.testLab);
    S.kind = S.ds.kind; S.size = S.ds.size; S.featureDefs = S.ds.featureDefs;
    S.inputCache.clear();
    S.excluded = new Set();
    S.selected = S.ds.train[0];
    if (S.kind === 'tabular') { S.mode = 'features'; S.convK = 0; S.augment = false; }
    document.body.dataset.kind = S.kind;
    buildTrays();
    $('task-title').textContent = S.task.title;
    $('task-blurb').textContent = S.task.blurb;
    $('meta-line').textContent = `${S.ds.specimens.length} synthetic ${noun(2)} · ${S.ds.train.length} training / ${S.ds.test.length} test${S.size ? ` · ${S.size} × ${S.size} px` : ''} · seed ${raw.meta.seed}`;
    $('question-select').value = id;
    document.querySelectorAll('[data-cls="0"]').forEach(el => { el.textContent = negName(); });
    document.querySelectorAll('[data-cls="1"]').forEach(el => { el.textContent = posName(); });
    document.querySelectorAll('[data-cls-called="0"]').forEach(el => { el.textContent = `called ${negName()}`; });
    document.querySelectorAll('[data-cls-called="1"]').forEach(el => { el.textContent = `called ${posName()}`; });
    document.querySelectorAll('[data-noun]').forEach(el => { el.textContent = noun(2); });
    $('train-count').textContent = `${S.ds.train.length}`; $('test-count').textContent = `${S.ds.test.length}`;
    $('stat-n-sub').textContent = `held-out ${noun(2)}`;
    $('train-count-2').textContent = `${S.ds.train.length} ${noun(2)}`; $('test-count-2').textContent = `${S.ds.test.length} ${noun(2)}`;
    $('feature-heading').textContent = S.kind === 'tabular' ? `The ${S.featureDefs.length} parameters the network can see` : 'The six measurements';
    $('feature-list').innerHTML = S.featureDefs.map(f => `<li><strong>${esc(f.name)}</strong>${f.unit ? ` (${esc(f.unit)})` : ''}${f.low != null ? ` · reference ${f.low}–${f.high}` : ''} — ${esc(f.desc)}</li>`).join('');
    const opts = S.featureDefs.map((f, i) => `<option value="${i}">${esc(f.name)}</option>`).join('');
    $('scatter-x').innerHTML = opts; $('scatter-y').innerHTML = opts;
    const ax = { leukemia: [0, 5], atypia: [0, 2], enlargement: [0, 2], irregularity: [5, 4] }[id] || [0, 1];
    $('scatter-x').value = ax[0]; $('scatter-y').value = ax[1];
    $('scatter-hint').textContent = {
      leukemia: 'WBC against lymphocytes shows why no single parameter works: CLL and viral lymphocytosis overlap, and acute leukemia sits at both ends of the WBC axis. Try platelets against hemoglobin, or basophils against immature granulocytes.',
      atypia: 'Area against darkness catches the enlarged and the hyperchromatic nuclei, solidity against contour roughness the irregular ones, texture the coarse chromatin. No single pair catches every atypical nucleus, which is why the network gets all six measurements.',
      enlargement: 'Area against darkness separates the classes with a straight line; solidity against contour roughness is now the decoy pair.',
      irregularity: 'Try the decoys, area against darkness, then solidity against contour roughness. A single straight line separates the classes on the second pair; that is what a one-layer network has to find.',
    }[id] || '';
    $('mode-pixels').disabled = S.kind === 'tabular';
    $('mode-pixels').title = S.kind === 'tabular' ? 'Blood counts have no pixels' : '';
    renderSourceOptions();
    renderInputPicker();
  }
  function inputsFor(mode, augment, excluded) {
    const normalize = mode !== 'features' ? S.normalize : 'off', enc = mode === 'code' ? codeEncoder() : null;
    const key = mode + (mode === 'pixels' && augment ? '+aug' : '') + (mode === 'features' && excluded.size ? '-' + [...excluded].sort().join(',') : '') + (enc ? '|' + enc.key : '') + `|${S.ds.sources.train}/${S.ds.sources.test}|${normalize}|noise${S.labelNoise}|labelled${S.labelled}`;
    if (!S.inputCache.has(key)) {
      if (S.inputCache.size > 40) S.inputCache.clear();
      const inp = DS.buildInputs(S.ds, mode, { augment, exclude: excluded, normalize, labelNoise: S.labelNoise, encoder: enc, labelled: S.labelled });
      inp.encoderKey = enc ? enc.key : null;
      S.inputCache.set(key, inp);
    }
    return S.inputCache.get(key);
  }
  // ---- the foundation encoder behind the code input: the one shipped with the page, or the one pretrained in 4 · Foundation
  function backbones() { return window.FOUNDATION_BACKBONES || []; }
  function shippedEncoder(id) { const b = backbones().find(x => x.id === id); if (!b) return null; if (!S.shipped[id]) S.shipped[id] = NN.Contrastive.fromJSON(b); return S.shipped[id]; }
  function defaultBackbone() { return backbones().length ? backbones()[0].id : 'page'; }
  function codeEncoder() {
    const shipped = S.backbone !== 'page' ? shippedEncoder(S.backbone) : null;
    if (shipped) return { cl: shipped, std: shipped.std, encode: ink => shipped.encode(shipped.std.apply(ink)), key: `shipped:${shipped.meta.id}`, trained: true,
      describe: `the encoder shipped with the page (${shipped.meta.nuclei.toLocaleString()} nuclei, ${shipped.describe()})` };
    if (!S.fm.set) fmBuild();
    if (!S.fm.cl) fmReset();
    const cl = S.fm.cl, std = S.fm.std;
    return { cl, std, encode: ink => cl.encode(std.apply(ink)), key: `page:${S.fm.gen}:${cl.steps}`, trained: cl.steps > 0,
      describe: `the encoder pretrained here${cl.steps ? ` (${S.fm.epoch} epoch${S.fm.epoch === 1 ? '' : 's'}, ${cl.describe()})` : ' (not pretrained yet: a random code)'}` };
  }
  function renderBackboneOptions() {
    $('backbone').innerHTML = backbones().map(b => `<option value="${esc(b.id)}">shipped: ${esc(b.nuclei.toLocaleString())} nuclei, ${esc(b.conv.K)} filters, code of ${esc(b.code)}${b.note ? ` (${esc(b.note)})` : ''}</option>`).join('') + '<option value="page">pretrained in 4 · Foundation</option>';
    S.backbone = defaultBackbone();
  }
  // the label a training case carries (wrong for the mislabelled ones when label noise is on), and whether it is wrong
  const trainLabel = s => (s.split === 'train' && S.inputs && S.inputs.labelOf ? S.inputs.labelOf(s) : s.label);
  const isFlipped = s => !!(S.inputs && S.inputs.flipped && S.inputs.flipped.has(s.id));
  const labelWord = s => (S.labelNoise > 0 && s.split === 'train' ? 'label' : 'truth');
  // ---- the two labs: which scans each set uses, and the pixels' stain normalisation
  function applySources() { DS.assignLabs(S.ds, S.trainLab, S.testLab); repaintThumbs(); }
  function renderSourceOptions() {
    const L = SOURCE_LABELS();
    for (const id of ['source-train', 'source-train-2', 'source-test', 'source-test-2']) $(id).innerHTML = DS.SOURCE_MODES.map(k => `<option value="${k}">${esc(L[k])}`).join('');
  }
  function sourceText() {
    const tr = S.ds.sources.train, te = S.ds.sources.test, pos = posName().toLowerCase(), neg = negName().toLowerCase();
    const train = {
      ours: 'The training set is our lab’s scans.',
      other: 'The training set is the other lab’s scans: the same nuclei, paler and with less contrast.',
      mixed: 'The training set mixes the labs at random: half of each class from each lab, the proper way to collect cases from two sites, and the way to make a network indifferent to the stain.',
      byClass: `Every ${pos} training nucleus comes from the other lab and every ${neg} one from ours: the stain is a perfect shortcut to the label, easier to learn than the contour. Would you have noticed?`,
    }[tr];
    const test = {
      ours: 'The test set is our lab’s scans.',
      other: 'The test set is the other lab’s scans of the same 20 held-out nuclei: what happens when the network meets a stain it never saw.',
      mixed: 'The test set mixes the labs at random.',
      byClass: `The test set is split by class the same way, so it carries the same shortcut: a network that reads the stain will look perfect on it.`,
    }[te];
    return `${train} ${test}${S.normalize !== 'off' && S.mode !== 'features' ? ` Stain normalisation is on (${S.normalize === 'lab' ? 'each lab’s typical background and nucleus levels, measured from its scans without any label, are matched to ours' : 'each scan is rescaled by its own background and nucleus levels'}), so the network sees comparable images.` : ''}`;
  }
  function renderSources() {
    const labs = !!S.ds.labs;
    $('sources-card').hidden = !labs; $('source-train-wrap').hidden = !labs; $('normalize-wrap').hidden = !(labs && S.mode !== 'features'); $('source-test-wrap').hidden = !labs;
    if (!labs) return;
    for (const id of ['source-train', 'source-train-2']) $(id).value = S.ds.sources.train;
    for (const id of ['source-test', 'source-test-2']) $(id).value = S.ds.sources.test;
    $('normalize').value = S.normalize; $('show-lab').checked = S.showLab;
    $('sources-text').textContent = sourceText();
  }
  function renderNoiseControls() {
    const on = S.labelNoise > 0;
    $('noise').value = Math.round(S.labelNoise * 100); $('noise-val').textContent = on ? `${Math.round(S.labelNoise * 100)}% of the labels wrong` : 'off';
    for (const id of ['show-flipped-wrap', 'show-flipped-wrap-2']) $(id).hidden = !on;
    for (const id of ['show-flipped', 'show-flipped-2']) $(id).checked = S.showFlipped;
    $('train-frame-word').textContent = on ? 'frame = the label the network is given' : 'frame = ground truth';
  }
  function setTrainSource(v) {
    S.trainLab = v; applySources(); syncControls();
    resetModel(`Training cases now from ${SOURCE_LABELS()[v]} — fresh random weights.`);
    renderDataTrays(); renderScatter(); renderInspector();
  }
  function setTestSource(v) { // the weights stay: only the test nuclei's scans change, so the test results start over
    S.testLab = v; applySources();
    S.inputs = inputsFor(S.mode, S.augment, S.excluded);
    clearTestResults(false, true); evaluateAll(); syncControls(); renderStatus();
    $('test-note').textContent = { ours: 'The test nuclei are now our lab’s scans. Classify them again.', other: 'The test nuclei are now the other lab’s scans of the same 20 nuclei: paler, less contrast. Classify them again and compare.', mixed: 'The test nuclei now come from both labs, mixed at random. Classify them again.', byClass: `The test nuclei are now split by class: ${posName().toLowerCase()} from the other lab, ${negName().toLowerCase()} from ours. Classify them again.` }[v];
    renderDataTrays(); renderScatter(); renderInspector(); if (S.stage === 'train') renderTraining(true); if (S.stage === 'test') { renderTestPanel(); renderTestGraph(); }
  }
  function resetModel(reason) {
    stopTraining();
    S.inputs = inputsFor(S.mode, S.augment, S.excluded);
    const conv = S.kind === 'image' && S.mode === 'pixels' && S.convK > 0 ? { K: S.convK, f: 5, pool: 4 } : null;
    const hidden = S.h1 > 0 ? (S.h2 > 0 ? [S.h1, S.h2] : [S.h1]) : [];
    S.net = new NN.Net({ inputSize: S.inputs.inputSize, imageSize: S.size, conv, hidden, activation: S.activation, seed: S.seed });
    S.epoch = 0; S.ptr = 0; S.debt = 0; S.history = []; S.lastBatch = new Set(); S.prevW = null;
    S.lesson = null; S.lastLesson = null; S.lessonCursor = 0; renderLessonLine();
    if (!S.applyingRecipe) $('recipe-select').value = '';
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
    const tr = S.ds.train, te = S.ds.test, lab = S.inputs.labelled;
    S.trainEval = S.net.evaluate(tr.map(s => S.inputs.xOf(s)), tr.map(trainLabel), 0.5, true); // against the labels the network was given
    if (lab.size < tr.length) { // the training scores count only the cases the network sees; the rest still get a call, for the tray
      const sub = tr.filter(s => lab.has(s.id)), ev = S.net.evaluate(sub.map(s => S.inputs.xOf(s)), sub.map(trainLabel));
      S.trainEval.accuracy = ev.accuracy; S.trainEval.loss = ev.loss;
    }
    S.testEval = S.net.evaluate(te.map(s => S.inputs.xOf(s)), te.map(s => s.label));
  }
  function recordEpoch() {
    evaluateAll();
    S.history.push({ epoch: S.epoch, loss: S.trainEval.loss, acc: S.trainEval.accuracy, testLoss: S.testEval.loss, testAcc: S.testEval.accuracy });
  }
  function trainStep() {
    if (S.lesson || S.lastLesson) endLesson();
    const X = S.inputs.trainX, Y = S.inputs.trainY, n = X.length;
    if (S.ptr === 0) shuffleOrder();
    const end = Math.min(n, S.ptr + S.batch);
    const idx = S.order.slice(S.ptr, end);
    S.prevW = { W: S.net.W.map(w => Float64Array.from(w)), Wo: Float64Array.from(S.net.Wo) };
    S.net.trainBatch(idx.map(i => X[i]), idx.map(i => Y[i]), S.lr, S.l2);
    S.lastBatch = new Set(idx.map(i => S.inputs.trainOwner[i]));
    S.ptr = end >= n ? 0 : end;
    let ended = false;
    if (S.ptr === 0) { S.epoch++; recordEpoch(); ended = true; }
    if (S.test.results.size) clearTestResults(true);
    return ended;
  }
  function batchesPerEpoch() { return Math.ceil(S.inputs.trainX.length / S.batch); }

  function startTraining() {
    if (S.lesson) endLesson();
    if (S.epoch >= S.epochs) { note(`Already at ${S.epochs} epochs. Raise the epoch count or reset to train again.`); return; }
    S.running = true; S.lastTime = performance.now(); S.debt = 0;
    $('btn-train').textContent = '⏸ Pause';
    requestAnimationFrame(tick);
  }
  function stopTraining(msg) {
    S.running = false;
    const b = $('btn-train'); if (b) b.textContent = '▶ Complete training';
    if (msg) note(msg);
  }
  function tick(now) {
    if (!S.running) return;
    const dt = Math.min(0.1, (now - S.lastTime) / 1000); S.lastTime = now;
    const bpe = batchesPerEpoch();
    S.debt += dt * S.speed * bpe;
    const t0 = performance.now(); let did = false, ended = false;
    while (S.debt >= 1 && performance.now() - t0 < 16) {
      ended = trainStep() || ended; S.debt -= 1; did = true;
      if (S.ptr === 0 && S.epoch >= S.epochs) { stopTraining(`Finished ${S.epochs} epochs. Train accuracy ${pct(S.trainEval.accuracy)}${S.peek ? `, test accuracy ${pct(S.testEval.accuracy)}` : ''}. Head to 3 · Test.`); break; }
    }
    if (S.debt > bpe) S.debt = bpe;
    if (did) renderTraining(ended || S.kind === 'tabular');
    if (S.running) requestAnimationFrame(tick);
  }
  // ------------------------------------------------------------------ playback: pause, step and slow the walk-throughs
  // A player paces a phased animation on its own clock, so a lesson or a test walk-through plays at the normal pace
  // untouched but can be paused, stepped back and forth, slowed or sped up, and skipped. phases = [[name, ms, arg], …];
  // frame(now) draws the current state and, while playing, schedules the next frame through the player.
  function makePlayer(phases, frame) {
    const starts = []; let total = 0; for (const p of phases) { starts.push(total); total += p[1]; }
    const live = i => phases[i][1] > 0;
    const P = { phases, starts, total, t: 0, playing: true, last: null, pending: false, frame };
    P.tick = now => { if (P.playing && P.last != null) P.t += Math.max(0, now - P.last) * S.animSpeed; P.last = now; };
    P.at = () => { // the phase the clock is in and its progress, or null once the animation has run out
      if (P.t >= total) return null;
      let i = 0; while (i < phases.length - 1 && P.t >= starts[i] + phases[i][1]) i++;
      const ms = phases[i][1]; return { i, phase: phases[i], frac: ms ? Math.min(1, (P.t - starts[i]) / ms) : 1, start: starts[i] };
    };
    P.schedule = () => { if (P.pending) return; P.pending = true; requestAnimationFrame(now => { P.pending = false; P.frame(now); }); };
    P.render = () => P.frame(performance.now());
    P.pause = () => { P.playing = false; P.render(); };
    P.resume = () => { if (P.playing) return; P.playing = true; P.last = null; P.schedule(); };
    P.toggle = () => (P.playing ? P.pause() : P.resume());
    P.skip = () => { P.t = total; P.playing = true; P.last = null; P.schedule(); };
    // stepping: while playing, jump to the start of the next (or of this) step and carry on; while paused, show the
    // steps at their end, one at a time, so each finished picture can be looked at
    P.next = () => {
      const cur = P.at(); if (!cur) return;
      let n = cur.i + 1; while (n < phases.length && !live(n)) n++;
      if (P.playing) { P.t = n < phases.length ? starts[n] : total; P.last = null; }
      else { const end = cur.start + cur.phase[1] - 1; P.t = P.t < end - 30 ? end : (n < phases.length ? starts[n] + phases[n][1] - 1 : total); }
      if (P.t >= total) { P.playing = true; P.last = null; P.schedule(); } else if (!P.playing) P.render();
    };
    P.prev = () => {
      const cur = P.at(), i = cur ? cur.i : phases.length;
      let p = i - 1; while (p >= 0 && !live(p)) p--;
      if (P.playing) { P.t = cur && P.t - cur.start > 1000 ? cur.start : (p >= 0 ? starts[p] : 0); P.last = null; }
      else { P.t = p >= 0 ? starts[p] + phases[p][1] - 1 : 0; P.render(); }
    };
    return P;
  }
  function setAnimSpeed(v) { S.animSpeed = v; try { localStorage.setItem('nn-anim-speed', String(v)); } catch (e) { /* ignore */ } syncSpeedControls(); }
  function syncSpeedControls() { for (const key of ['lesson', 'test']) { const r = $(`${key}-speed`), l = $(`${key}-speed-val`); if (r) r.value = S.animSpeed; if (l) l.textContent = `${S.animSpeed}×`; } }
  function renderPlayerControls(key, P) { const box = $(`${key}-player`); if (!box) return; box.hidden = !P; if (P) $(`${key}-pause`).textContent = P.playing ? '⏸ Pause' : '▶ Play'; }
  function activePlayer() { return (S.stage === 'train' ? (S.lesson && S.lesson.player) : S.stage === 'test' ? (S.test.animating && S.test.player) : null) || null; }

  // ------------------------------------------------------------------ the lesson: one case teaches the network
  // A walk-through of one gradient step on a training case in six steps (five without a hidden layer):
  //   1 forward pass   values flow from the inputs to the output, hop by hop
  //   2 loss           call against truth; the error is the slope of the loss at the output
  //   3 backward pass  blame flows back to every hidden unit (skipped without hidden units)
  //   4 gradients      every weight's gradient = blame at its end × activity at its start
  //   5 update         the weights actually move: w ← w − learning rate × gradient (drawn as a morph)
  //   6 check          the same case runs forward again with the new weights
  // The step is real training: a batch of one at the current learning rate.
  const LESSON_MS = { forward: 2600, loss: 2600, blame: 2600, blame2: 3400, gradient: 3000, update: 2600, check: 2600,
    convBlame: 4000, convBlame0: 5200, convGradient: 9000 }; // a convolutional network's extra: the pooled blame and the un-pooling, the filters' gradients
  const LESSON_TITLES = { forward: 'Forward pass', loss: 'Loss', blame: 'Backward pass', gradient: 'Gradients', update: 'Update', check: 'Check' };
  function lessonPhases() {
    const L = S.net.hidden.length, conv = !!S.net.conv;
    const plan = conv ? Viz.convPlan(S.net, true) : Viz.sweepPlan(S.net, S.mode), replay = conv ? Viz.convPlan(S.net, false) : Viz.sweepPlan(S.net, S.mode, { prep: false });
    const ph = [['forward', plan.total], ['loss', LESSON_MS.loss]];
    if (L) ph.push(['blame', (L > 1 ? LESSON_MS.blame2 : LESSON_MS.blame) + (conv ? LESSON_MS.convBlame : 0)]);
    else if (conv) ph.push(['blame', LESSON_MS.convBlame0]); // without a hidden layer the error still goes back through the pooling
    ph.push(['gradient', LESSON_MS.gradient + (conv ? LESSON_MS.convGradient : 0)], ['update', LESSON_MS.update], ['check', S.mode === 'pixels' ? Math.round(replay.total / 3) : replay.total]); // the check replays a pixel sweep at triple speed, the subtraction already done
    return ph;
  }
  function canTeach() { return !!(S.net && !S.running && !S.lesson); }
  // the case the next lesson will use: a training case the user picked, else the one after the last lesson
  function nextLessonCase() {
    const tr = S.ds.train, lab = S.inputs.labelled;
    if (S.selected && S.selected.split === 'train' && lab.has(S.selected.id) && !(S.lastLesson && S.lastLesson.s === S.selected)) return S.selected;
    for (let k = 0; k < tr.length; k++) { const c = tr[(S.lessonCursor + k) % tr.length]; if (lab.has(c.id)) return c; } // the next labelled case
    return tr[S.lessonCursor % tr.length];
  }
  function updateTeachButton() {
    const b = $('btn-teach'); if (!b || !S.net) return;
    b.disabled = !canTeach() && !S.lesson;
    b.textContent = S.lesson ? 'Skip ▸' : 'Teach next case';
    b.title = `Teach ${nextLessonCase().name} one step: forward pass, loss, backward pass, gradients, update, check (T)`;
  }
  function teachNext() {
    if (S.lesson) { S.lesson.player.skip(); return; }
    if (!canTeach()) return;
    startLesson(nextLessonCase());
  }
  function snapParams() {
    const n = S.net, p = { W: n.W.map(w => Float64Array.from(w)), b: n.b.map(b => Float64Array.from(b)), Wo: Float64Array.from(n.Wo), bo: n.bo };
    if (n.conv) { p.Wc = Float64Array.from(n.Wc); p.bc = Float64Array.from(n.bc); }
    return p;
  }
  function setParams(p) {
    p.W.forEach((w, l) => S.net.W[l].set(w)); p.b.forEach((b, l) => S.net.b[l].set(b)); S.net.Wo.set(p.Wo); S.net.bo = p.bo;
    if (p.Wc) { S.net.Wc.set(p.Wc); S.net.bc.set(p.bc); }
  }
  function lerpParams(a, b, t) {
    const mix = (x, y) => { const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] + (y[i] - x[i]) * t; return o; };
    const p = { W: a.W.map((w, l) => mix(w, b.W[l])), b: a.b.map((v, l) => mix(v, b.b[l])), Wo: mix(a.Wo, b.Wo), bo: a.bo + (b.bo - a.bo) * t };
    if (a.Wc) { p.Wc = mix(a.Wc, b.Wc); p.bc = mix(a.bc, b.bc); }
    return p;
  }
  function startLesson(s) {
    if (!s || s.split !== 'train' || !S.net || !S.inputs.labelled.has(s.id)) return;
    stopTraining();
    if (S.selected !== s) selectSpecimen(s);
    const x = S.inputs.xOf(s), y = trainLabel(s); // the network learns the label it is given, right or wrong
    const res = S.net.lesson(x, y);
    const hops = S.net.conv ? Viz.convPlan(S.net, true).hops : Viz.sweepPlan(S.net, S.mode).hops;
    S.lastLesson = null;
    const player = makePlayer(lessonPhases(), lessonFrame);
    if (reducedMotion) player.t = player.total;
    S.lesson = { s, x, y, res, hops, phases: player.phases, player, phase: 'forward', frac: 0, pBefore: res.fw.p, pAfter: null, morph: null };
    updateTeachButton(); renderLessonLine();
    player.schedule();
  }
  // the update itself: applied once, then drawn as a morph from the old weights to the new ones
  function applyLesson() {
    const les = S.lesson; if (les.morph) return;
    const before = snapParams();
    S.prevW = { W: before.W.map(w => Float64Array.from(w)), Wo: Float64Array.from(before.Wo) };
    S.net.applyGradient(les.res.g, 1, S.lr, S.l2);
    les.morph = { before, after: snapParams() };
    les.pAfter = S.net.forward(les.x).p;
  }
  function settleLesson() { // leave the network on the new weights and refresh what depends on them
    const les = S.lesson;
    if (!les.morph) applyLesson();
    setParams(les.morph.after);
    if (S.test.results.size) clearTestResults(true);
    evaluateAll(); renderStatus();
  }
  const easeInOut = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  function lessonFrame(now) {
    const les = S.lesson; if (!les) return;
    const P = les.player; P.tick(now);
    const cur = P.at();
    if (!cur) { finishLesson(); return; }
    const phase = cur.phase[0], frac = cur.frac;
    // the weights follow the step (the player can step back): the old ones before the update, morphing during it, the new ones from the check on
    if (phase === 'update') { applyLesson(); setParams(lerpParams(les.morph.before, les.morph.after, easeInOut(frac))); }
    else if (phase === 'check') { if (!les.settled) { settleLesson(); les.settled = true; } else setParams(les.morph.after); }
    else if (les.morph) setParams(les.morph.before);
    les.phase = phase; les.frac = frac;
    renderTrainGraph(); renderLessonLine();
    if (P.playing) P.schedule();
  }
  function finishLesson() {
    const les = S.lesson; if (!les) return;
    if (!les.settled) settleLesson();
    const tr = S.ds.train;
    S.lessonCursor = (tr.indexOf(les.s) + 1) % tr.length;
    S.lastLesson = { s: les.s, pBefore: les.pBefore, pAfter: les.pAfter };
    S.lesson = null;
    renderTraining(true); renderLessonLine(); updateTeachButton();
  }
  function endLesson() { // a reset or a normal training step supersedes the lesson: settle it and clear the strip
    if (S.lesson) { if (!S.lesson.settled) settleLesson(); S.lesson = null; }
    S.lastLesson = null;
    renderLessonLine(); updateTeachButton();
  }
  // where the forward pass of a convolutional lesson is: the segment of the convolution walk-through and the progress in it
  function lessonConvAt() { const les = S.lesson; return Viz.convAt(Viz.convPlan(S.net, les.phase !== 'check'), les.frac); }
  function lessonView() {
    const les = S.lesson; if (!les) return null;
    const g = les.res.g, hops = les.hops, sweeping = les.phase === 'forward' || les.phase === 'check';
    let reveal = hops, anim = null; // how many hops of the forward sweep have arrived (everything is visible outside the sweeps)
    if (S.net.conv) { // the convolution's forward pass is drawn from an anim, as in the test walk-through; the other steps carry a marker for the conv overlays
      const plan = Viz.convPlan(S.net, les.phase !== 'check');
      if (sweeping) { const at = Viz.convAt(plan, les.frac); anim = Viz.convAnim(S.net, plan, at.seg, at.frac, { again: les.phase === 'check', finalBanner: 'forward pass done' }); reveal = anim.reveal; }
      else anim = { phase: 'lesson', hops, reveal: hops, wipes: les.phase === 'loss' ? plan.hopKeys : null };
    } else if (sweeping) reveal = Viz.sweepState(Viz.sweepPlan(S.net, S.mode, { prep: les.phase !== 'check' }), les.frac).reveal;
    return { phase: les.phase, frac: les.frac, hops, reveal, anim, error: les.res.error, loss: les.res.loss, y: les.y, truthName: className(les.y), pBefore: les.pBefore, pAfter: les.pAfter,
      truthWord: labelWord(les.s), delta: g.delta, gW: g.gW, gWo: g.gWo, gWc: g.gWc, dPooled: g.dPooled, dConv: g.dConv, lr: S.lr, fwBefore: les.res.fw };
  }
  // what the forward sweep shows, for the strip
  function forwardText() {
    const hidden = S.net.hidden.length;
    if (S.mode === 'pixels' && !S.net.conv) {
      const prep = `${S.normalize !== 'off' ? 'After the stain normalisation, the' : 'First the'} mean training ${noun(1)} is subtracted from this one: the network sees the difference (orange = more ink than average, blue = less), which is large at the membrane and near zero in the centre. `;
      return prep + (hidden
        ? `That difference is laid over each weight map and multiplied cell by cell into a product map. A scan line sums the map, orange cells against blue, the bias is added, and ReLU keeps the positive part: that is the unit's value. The units' values × their weights then flow to the output.`
        : `That difference is laid over the weight map and multiplied cell by cell into the product map. A scan line sums the map, orange cells against blue, the bias is added to give the score z, and the sigmoid turns z into a probability.`);
    }
    return 'Each connection carries its weight × the value at its start (thick = large, orange positive, blue negative) and each node adds up what arrives, hop by hop to the output.';
  }
  function fmtPair(a, b) { const d = a.toFixed(2) === b.toFixed(2) ? 3 : 2; return `${a.toFixed(d)} → ${b.toFixed(d)}`; }
  function renderLessonLine() {
    const box = $('lesson'); if (!box) return;
    const les = S.lesson, last = S.lastLesson;
    if (!les && !last) { box.hidden = true; $('legend-lesson').hidden = true; return; }
    box.hidden = false; $('legend-lesson').hidden = !les;
    let html;
    if (les) {
      const n = les.phases.findIndex(p => p[0] === les.phase) + 1, total = les.phases.length;
      const hidden = S.net.hidden.length, pixels = S.mode === 'pixels', conv = !!S.net.conv, q = les.frac, ST = Viz.CONV_STAGES;
      const p = les.pBefore.toFixed(2), e = Viz.fmtSigned(les.res.error, 2), call = les.pBefore >= 0.5 ? posName() : negName();
      const sure = Math.abs(les.res.error) < 0.02, allSilent = hidden > 0 && les.res.g.delta.every(d => d.every(v => v === 0));
      const pair = les.pAfter == null ? `${p} → …` : fmtPair(les.pBefore, les.pAfter);
      const verdict = `The network calls <b>${p}</b> (${esc(call)}); the ${isFlipped(les.s) ? `label says <b>${esc(className(les.y))}</b>, though the truth is ${esc(className(les.s.label))}: this case is mislabelled, and the network learns the label it is given` : `${labelWord(les.s)} is <b>${esc(className(les.y))}</b>`}.`;
      // a convolutional network's forward pass and check are narrated stage by stage, as in the test walk-through
      const sub = conv && (les.phase === 'forward' || les.phase === 'check') ? (() => { const at = lessonConvAt(); return convPhaseInfo(les.s, at.seg[0], { key: at.seg[2] }, true); })() : null;
      const pool = conv ? S.net.conv.pool : 0;
      const convBlame = q < ST.blameDense
        ? (hidden ? null : 'How much is each pooled cell to blame? The error goes back along Σ → z to the weight map: each pooled cell’s blame is the error × its weight on that cell.')
        : q < ST.blameDots
        ? (hidden ? 'Then the blame goes back through the weight maps to the pooled cells: a pooled cell’s blame is Σ over the units of the unit’s blame × its weight on that cell. It wipes back along the bands and is drawn over the pooled maps (blue: the cell should come down, orange: go up).'
                  : 'The error × each weight is the blame of the pooled cell under it, drawn over the pooled maps (blue: the cell should come down, orange: go up).')
        : `Then back through the pooling. Each pooled cell kept only the largest value of its ${pool}×${pool} block, so all of its blame lands on that one position of the feature map and none on the rest: the dots. ReLU passes it only where the map was on; a hollow dot was off and gets none.`;
      const convGradient = q < ST.gradDense ? null
        : 'The filters last. A filter was used at every position of the image, so its gradient adds up over every position that got blame: the blame there × the 5×5 image patch under the filter. Filter 1 goes position by position (the window on the image, the arithmetic under it: patch → × blame → Σ so far), the other filters all at once. The step, −learning rate × Σ, settles over each filter until the update folds it in.';
      const texts = {
        forward: sub ? `${sub.text}${sub.phase === 'final' ? ' ' + verdict : ''}` : `${forwardText()} ${verdict}`,
        loss: `How wrong was it? Cross-entropy loss <b>${les.res.loss.toFixed(2)}</b>. Its slope at the output is the error, call − truth = ${p} − ${les.y} = <b>${e}</b>: ${les.res.error > 0 ? 'too high, so the score must come down' : 'too low, so the score must go up'}.`,
        blame: (conv && convBlame) || (hidden > 1
          ? 'How much is each hidden unit to blame? The error flows back one layer at a time along each connection as blame × weight (thick = large; blue: the unit should come down, orange: go up). A unit adds up what arrives, and ReLU passes it on only if the unit was on: an off unit’s blame is 0. Each pill shows the arithmetic.'
          : 'How much is each hidden unit to blame? The error flows back along each connection as error × weight (thick = large; blue: the unit should come down, orange: go up). ReLU passes it on only if the unit was on, so a unit that was off gets blame 0. Each pill shows the arithmetic.'),
        gradient: allSilent
          ? `Every hidden unit was off for this case, so every blame is zero and no weight map${conv ? ' and no filter' : ''} has a gradient: only the output bias does. A ReLU unit that is off cannot learn from a case.`
          : (conv && convGradient) || (hidden
          ? `For every connection, gradient = blame at its end × activity at its start; the labels spell it out. The glow shows which way the weight should move (orange up, blue down) and how steeply.${conv ? ' A unit’s weight map reads the stacked pooled maps, so its gradient is that map scaled by the unit’s blame: a copy comes back from the blame side onto the weight map.' : pixels ? ` On pixels, a weight map’s gradient is the difference image itself (what the network sees) scaled by the unit’s blame: a copy comes back from the blame side onto the weight map.` : ''}`
          : `For every connection, gradient = error × its input: big inputs, big gradients; the labels spell it out. Orange = the weight should rise, blue = fall.${conv ? ' The weight map reads the stacked pooled maps, so its gradient is that map scaled by the error, which comes back along Σ → z onto the map.' : pixels ? ` On pixels, the gradient of the whole weight map is the difference image itself (what the network sees) scaled by the error, which comes back along Σ → z onto the map.` : ''}`),
        update: `Every weight takes one small step against its gradient: w ← w − learning rate × gradient, with learning rate ${S.lr}. The labels show each weight before → after; watch the connections${conv ? ', the weight maps and the filters' : pixels ? ' and weight maps' : ''} change.`,
        check: sure
          ? `The same case runs forward again, only to show what the step did (training itself moves on to the next case). It was already right and sure, so the step was tiny: <b>${pair}</b>.`
          : `The same case runs forward again with the new ${conv ? 'filters and ' : ''}weights, only to show what the step did: <b>${pair}</b>. This replay is not part of training, which moves straight on to the next case: one case, one small step, repeated for every case, many times over.`,
      };
      html = `<span class="step">Step ${n} of ${total} · ${LESSON_TITLES[les.phase]}${sub ? ' · ' + sub.title : ''}</span> <span>${texts[les.phase]}</span> <span class="muted small">N or a click on the diagram skips ahead · space pauses · ← → step</span>`;
    } else {
      html = `<span class="step">✓ ${esc(last.s.name)}</span> <span>(${labelWord(last.s)} ${esc(className(trainLabel(last.s)))}) ${fmtPair(last.pBefore, last.pAfter)}. Next up: <b>${esc(nextLessonCase().name)}</b>.</span>`;
    }
    $('lesson-text').innerHTML = html;
    renderPlayerControls('lesson', les ? les.player : null);
  }

  function stepBatch() { stopTraining(); trainStep(); renderTraining(true); if (S.ptr === 0) note(`Epoch ${S.epoch} complete.`); }
  function stepEpoch() { stopTraining(); do { trainStep(); } while (S.ptr !== 0); renderTraining(true); note(`Epoch ${S.epoch} complete.`); }

  // ------------------------------------------------------------------ test phase
  function clearTestResults(notify, keepScores) {
    S.test.results.clear(); S.test.next = 0; S.test.revealed.clear(); S.test.animating = false; S.test.player = null;
    if (!keepScores) S.test.scores = {};
    renderTestLine(null);
    if (notify) $('test-note').textContent = 'The model changed, so the test results were cleared. Classify again to score the new weights.';
    renderTestPanel();
  }
  function classifyNext(quiet) {
    if (S.test.animating) { if (S.test.player) S.test.player.skip(); return false; }
    if (S.test.next >= S.ds.test.length) return false;
    const s = S.ds.test[S.test.next++];
    const fw = S.net.forward(S.inputs.xOf(s));
    S.test.results.set(s.id, { p: fw.p });
    $('test-note').textContent = '';
    selectSpecimen(s, { silent: true });
    if (quiet || reducedMotion) { S.test.revealed.add(s.id); renderTestPanel(); renderInspector(); renderTestGraph(); return true; }
    S.test.animating = true;
    renderTestPanel(); renderInspector();
    if (S.net.conv) animateConvClassify(s); else animateDenseClassify(s);
    return true;
  }
  function classifyAll() {
    const go = () => { if (S.test.animating) { if (S.test.player) S.test.player.skip(); setTimeout(go, 50); return; } if (S.test.next < S.ds.test.length) { classifyNext(true); setTimeout(go, reducedMotion ? 0 : 60); } };
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
  function paintThumb(cv, s) { if (s.px) Viz.renderThumb(cv, s.px, s.size, S.tint); else Viz.renderFingerprint(cv, s.deviation, false); }
  function makeThumb(s, map, named) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'thumb'; b.dataset.id = s.id;
    b.setAttribute('aria-label', `${S.task.specimenNoun || 'Specimen'} ${s.name}`);
    const cv = document.createElement('canvas'); cv.width = s.size || 48; cv.height = s.size || 48;
    b.appendChild(cv);
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = '✗'; b.appendChild(badge);
    const lab = document.createElement('span'); lab.className = 'lab'; lab.textContent = 'B'; lab.title = 'scanned at the other lab'; b.appendChild(lab);
    const flip = document.createElement('span'); flip.className = 'flip'; flip.textContent = '✎'; flip.title = 'mislabelled'; b.appendChild(flip);
    if (named) { const nm = document.createElement('span'); nm.className = 'name'; nm.textContent = s.name; b.appendChild(nm); }
    b.addEventListener('click', () => selectSpecimen(s));
    paintThumb(cv, s);
    map.set(s.id, b);
    return b;
  }
  function buildTrays() {
    const dt = $('data-train-tray'), dx = $('data-test-tray'), tt = $('train-tray'), xt = $('test-tray');
    for (const el of [dt, dx, tt, xt]) el.innerHTML = '';
    for (const m of Object.values(thumbs)) m.clear();
    for (const s of S.ds.train) { dt.appendChild(makeThumb(s, thumbs.data, true)); tt.appendChild(makeThumb(s, thumbs.train, false)); }
    for (const s of S.ds.test) { dx.appendChild(makeThumb(s, thumbs.data, true)); xt.appendChild(makeThumb(s, thumbs.test, true)); }
  }
  function repaintThumbs() {
    for (const map of Object.values(thumbs)) for (const [id, el] of map) paintThumb(el.querySelector('canvas'), S.ds.specimens[id]);
  }
  function setThumbState(el, { call, truth, wrong, right, unknown, inBatch, selected, q, title, unit, lab, flipped, unlabelled }) {
    el.className = 'thumb' + (call != null ? ` call-${call}` : '') + (truth != null ? ` truth-${truth}` : '') + (wrong ? ' wrong' : '') + (right ? ' right' : '') + (unknown ? ' unknown' : '') + (inBatch ? ' in-batch' : '') + (selected ? ' selected' : '') + (lab === 'B' ? ' lab-b' : '') + (S.showLab ? ' show-lab' : '') + (flipped ? ' flipped' : '') + (S.showFlipped ? ' show-flip' : '') + (unlabelled ? ' unlabelled' : '');
    el.style.borderColor = unit != null ? Viz.unitColor(unit) : '';
    const badge = el.querySelector('.badge');
    badge.className = 'badge' + (q ? ' q' : '');
    badge.textContent = q ? '?' : (right ? '✓' : '✗');
    if (title) el.title = title;
  }
  function renderDataTrays() {
    for (const s of S.ds.specimens) {
      const el = thumbs.data.get(s.id); if (!el) continue;
      const known = truthKnown(s);
      const withheld = s.split === 'train' && !S.inputs.labelled.has(s.id);
      setThumbState(el, { truth: known ? trainLabel(s) : null, unknown: !known, selected: S.selected && S.selected.id === s.id, lab: s.lab, flipped: isFlipped(s), unlabelled: withheld,
        title: `${s.name} · ${s.split === 'train' ? 'training' : 'test'} set${withheld ? ' · label withheld: the network never sees this case' : ''}${known ? ` · ${labelWord(s)} ${className(trainLabel(s))}` : ''}${known && isFlipped(s) && S.showFlipped ? ` (mislabelled: truth ${className(s.label)})` : ''}${known && s.subtype ? ' · ' + subtypeName(s.subtype) : ''}${s.lab === 'B' ? ' · scanned at the other lab' : ''}` });
    }
  }
  function topUnit(acts) { let j = 0; for (let i = 1; i < acts.length; i++) if (acts[i] > acts[j]) j = i; return acts[j] > 0 ? j : null; }
  function renderTrainTray() {
    const probs = S.trainEval.probs, acts = S.trainEval.acts;
    const byUnit = S.trayColor === 'unit' && acts;
    S.ds.train.forEach((s, k) => {
      const el = thumbs.train.get(s.id);
      const p = probs[k], call = p >= 0.5 ? 1 : 0, unl = !S.inputs.labelled.has(s.id);
      const u = byUnit && !unl ? topUnit(acts[k]) : null;
      const y = trainLabel(s);
      setThumbState(el, { call: unl ? null : call, wrong: !unl && call !== y, unlabelled: unl, inBatch: S.lastBatch.has(s.id), selected: S.selected && S.selected.id === s.id, unit: u, lab: s.lab, flipped: !unl && isFlipped(s),
        title: unl ? `${s.name} · no label given: the network never trains on it · it would call ${className(call)} (P ${fmtP(p)})${s.lab === 'B' ? ' · scanned at the other lab' : ''}` : `${s.name} · ${labelWord(s)} ${className(y)}${isFlipped(s) && S.showFlipped ? ` (mislabelled: truth ${className(s.label)})` : ''}${s.subtype ? ' (' + subtypeName(s.subtype) + ')' : ''} · call ${className(call)} (P ${fmtP(p)})${byUnit ? ` · most active: ${u == null ? 'no unit' : 'unit ' + (u + 1)}` : ''}${s.lab === 'B' ? ' · scanned at the other lab' : ''}` });
    });
    $('tray-legend-call').hidden = !!byUnit;
    $('tray-legend-unit').hidden = !byUnit;
    $('tray-legend-flip').hidden = !(S.labelNoise > 0 && S.showFlipped);
    $('tray-disagree-word').textContent = S.labelNoise > 0 ? 'disagrees with the label it was given' : 'disagrees with truth';
    if (byUnit) $('tray-legend-unit').innerHTML = S.net.hidden[0] ? Array.from({ length: S.net.hidden[0] }, (_, j) => `<span><span class="udot" style="background:${Viz.unitColor(j)}"></span>unit ${j + 1}</span>`).join('') + '<span class="muted">grey = no unit active</span>' : '';
  }
  function renderTestTray() {
    const thr = S.test.threshold;
    for (const s of S.ds.test) {
      const el = thumbs.test.get(s.id);
      const r = S.test.results.get(s.id), shown = r && S.test.revealed.has(s.id);
      if (!shown) setThumbState(el, { unknown: true, q: true, selected: S.selected && S.selected.id === s.id, lab: s.lab, title: `${s.name} · not classified yet${s.lab === 'B' ? ' · scanned at the other lab' : ''}` });
      else {
        const call = r.p >= thr ? 1 : 0;
        setThumbState(el, { call, wrong: call !== s.label, right: call === s.label, selected: S.selected && S.selected.id === s.id, lab: s.lab,
          title: `${s.name} · call ${className(call)} (P ${fmtP(r.p)}) · truth ${className(s.label)}${s.subtype ? ' (' + subtypeName(s.subtype) + ')' : ''}${s.lab === 'B' ? ' · scanned at the other lab' : ''}` });
      }
    }
  }

  // ------------------------------------------------------------------ rendering: train panel
  function note(msg) { $('note').textContent = msg || ''; }
  function renderStatus() {
    const bpe = batchesPerEpoch();
    const bi = S.ptr === 0 ? bpe : Math.ceil(S.ptr / S.batch);
    const inputDesc = S.mode === 'features' ? `${S.inputs.inputSize} ${S.kind === 'tabular' ? 'parameters' : 'measurements'}` : S.mode === 'code' ? `the code: ${S.inputs.inputSize} numbers from ${esc(codeEncoder().describe)}` : '1,024 pixels';
    const nTrain = S.inputs.trainSet.length, sub = nTrain < S.ds.train.length, flippedSeen = [...S.inputs.flipped].filter(id => S.inputs.labelled.has(id)).length;
    $('status').innerHTML =
      `<span>architecture <b>${inputDesc} → ${S.net.describe()} → output</b></span>` +
      `<span>parameters <b>${S.net.parameterCount().toLocaleString()}</b></span>` +
      `<span>training cases <b>${nTrain}</b>${sub ? ` labelled of ${S.ds.train.length}` : ''}${S.augment && S.mode === 'pixels' ? ` (× 8 orientations = ${S.inputs.trainX.length})` : ''}${S.ds.sources.train !== 'ours' ? ` · from ${esc(SOURCE_LABELS()[S.ds.sources.train])}` : ''}${S.normalize !== 'off' && S.mode !== 'features' ? ' · stain normalised' : ''}${S.labelNoise > 0 ? ` · <b>${flippedSeen}</b> mislabelled` : ''}</span>` +
      `<span>epoch <b>${S.epoch}</b> / ${S.epochs}</span>` +
      `<span>batch <b>${S.ptr === 0 ? '–' : bi}</b> / ${bpe}</span>` +
      `<span>loss <b>${S.trainEval.loss.toFixed(3)}</b></span>` +
      `<span>train accuracy <b>${pct(S.trainEval.accuracy)}</b></span>` +
      (S.peek ? `<span>test accuracy <b>${pct(S.testEval.accuracy)}</b> (peeking)</span>` : '');
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
    return { net: S.net, mode: S.mode === 'code' ? 'features' : S.mode, inputCaption: S.mode === 'code' ? `INPUT · THE CODE · ${S.inputs.inputSize} NUMBERS` : null, x, fw, prev: S.prevW, featureNames: S.inputs.featureNames || [], specimen: s, size: S.size, tint: S.tint, stage, hover,
      inputMean: S.mode === 'pixels' && S.inputs.std ? S.inputs.std.mean : null,
      inputPx: s && S.mode === 'pixels' && S.inputs.normalize !== 'off' ? inkToPx(S.inputs.rawOf(s)) : null, // the scan after the stain normalisation
      activation: S.activation, activationLabel: NN.ACTIVATIONS[S.activation].label, positiveName: posName(), negativeName: negName() };
  }
  function renderTrainGraph() {
    const cv = $('net-canvas');
    const m = graphModel(S.selected || null, 2, S.hover.train);
    if (S.lesson) { const lv = lessonView(); m.lesson = lv; m.reveal = lv.reveal; m.hops = lv.hops; m.anim = lv.anim; if (lv.phase !== 'check') m.fw = S.lesson.res.fw; }
    cv._model = m;
    Viz.drawNetwork(cv, m);
  }
  function renderTestGraph(stage, anim, sweep) {
    const cv = $('net-canvas-test');
    const s = S.selected;
    let st = 2;
    if (s && s.split === 'test') st = S.test.revealed.has(s.id) ? 2 : (stage == null ? 0 : stage);
    if (s && s.split === 'test' && S.test.results.has(s.id) && stage != null) st = stage;
    const m = graphModel(s, st, S.hover.test);
    if (s && s.split === 'test' && S.test.results.has(s.id) && !S.test.revealed.has(s.id)) { m.fw = S.net.forward(m.x); }
    m.anim = anim || null;
    m.sweep = sweep || null;
    if (sweep) { m.reveal = sweep.reveal; m.hops = sweep.hops; }
    else if (anim && anim.reveal != null) { m.reveal = anim.reveal; m.hops = anim.hops; } // the convolution walk-through reveals hop by hop too
    if (!sweep && !anim && s && s.split === 'test' && S.test.revealed.has(s.id)) { const r = S.test.results.get(s.id); m.truth = { name: className(s.label), y: s.label, mark: r ? ((r.p >= S.test.threshold ? 1 : 0) === s.label ? '✓' : '✗') : '' }; }
    cv._model = m;
    Viz.drawNetwork(cv, m);
  }
  // Classify-next walk-through for dense networks, about 4 s: a forward pass through the frozen weights, hop by hop,
  // then the call at the threshold, then the truth
  function animateDenseClassify(s) {
    const plan = Viz.sweepPlan(S.net, S.mode), hops = plan.hops;
    const T = [['forward', plan.total], ['call', 900], ['reveal', 900]];
    const x = S.inputs.xOf(s), p = S.net.forward(x).p, called = p >= S.test.threshold ? 1 : 0;
    let shown = false;
    const P = makePlayer(T, null); S.test.player = P;
    const finish = () => { S.test.player = null; S.test.revealed.add(s.id); S.test.animating = false; renderTestPanel(); renderInspector(); renderTestGraph(); renderTestLine(s, p, called, 'done'); };
    const frame = now => {
      if (!S.test.animating || S.selected !== s || S.test.player !== P) return;
      P.tick(now); const cur = P.at();
      if (!cur) { finish(); return; }
      const phase = cur.phase[0], frac = cur.frac;
      const reveal = phase === 'forward' ? Viz.sweepState(plan, frac).reveal : hops;
      if (phase === 'reveal' && !shown) { shown = true; S.test.revealed.add(s.id); renderTestPanel(); renderInspector(); }
      renderTestGraph(phase === 'forward' ? 0 : 2, null, { phase, frac, hops, reveal, p, called: className(called), threshold: S.test.threshold, truthName: className(s.label), y: s.label, correct: called === s.label });
      renderTestLine(s, p, called, phase);
      if (P.playing) P.schedule();
    };
    P.frame = frame;
    P.schedule();
  }
  // the strip above the test diagram: what is happening to the case going through
  function renderTestLine(s, p, called, phase, info) {
    const box = $('test-lesson'); if (!box) return;
    if (!s) { box.hidden = true; return; }
    box.hidden = false;
    const inputDesc = S.mode === 'pixels' ? '1,024 pixels' : `${S.inputs.inputSize} ${S.kind === 'tabular' ? 'parameters' : 'measurements'}`;
    const thr = S.test.threshold.toFixed(2), pp = p.toFixed(2), calledName = className(called), truth = className(s.label), correct = called === s.label;
    let html;
    const ci = S.net.conv ? convPhaseInfo(s, phase, info) : null;
    if (ci) html = `<span class="step">${ci.title}</span> <span>${ci.text}</span>`;
    else if (phase === 'forward') html = `<span class="step">Forward pass</span> <span><b>${esc(s.name)}</b>: its ${inputDesc} flow through the frozen weights. ${forwardText()} Nothing is learned here.</span>`;
    else if (phase === 'call') html = `<span class="step">Call</span> <span>P(${esc(posName())}) = <b>${pp}</b>, which is ${p >= S.test.threshold ? 'at or above' : 'below'} the threshold of ${thr}, so the network calls <b>${esc(calledName)}</b>.</span>`;
    else html = `<span class="step">${correct ? '✓' : '✗'} ${esc(s.name)}</span> <span>called <b>${esc(calledName)}</b> (${pp}) · truth <b>${esc(truth)}</b>${correct ? '' : (called ? ' · a false positive' : ' · a false negative')}.</span> <span class="muted small">N classifies the next case</span>`;
    $('test-lesson-text').innerHTML = html;
    renderPlayerControls('test', S.test.animating ? S.test.player : null);
  }
  // what each stage of a convolutional network's forward pass shows, for the strips: { phase, title, text }, or null
  function convPhaseInfo(s, phase, info, lesson) {
    const K = S.net.conv.K, pool = S.net.conv.pool, nL = S.net.hidden.length, F = S.net.featureCount;
    const hop = info && info.key, cols = K <= 4 ? 2 : 4, stack = `${Math.ceil(K / cols)}×${cols}`;
    const texts = {
      prep: ['Preprocessing', `<b>${esc(s.name)}</b>: ${S.normalize !== 'off' ? 'after the stain normalisation, ' : ''}the mean training ${noun(1)} is subtracted from this one. The network sees the difference (orange = more ink than average, blue = less), large at the membrane and near zero in the centre.`],
      scan1: ['Convolution', 'Filter 1 slides over the difference image. At each position its 5×5 weights multiply the 5×5 values under them, the products are summed, and ReLU keeps the positive part: one cell of feature map 1. The arithmetic of the current position is shown under the image.'],
      scan: ['Convolution', `The other ${K - 1} filters sweep the ${noun(1)} the same way, each producing its own feature map.`],
      pool1: ['Max-pooling', `Each ${pool}×${pool} block of feature map 1 keeps only its largest value, so the map shrinks and the exact position inside a block no longer matters.`],
      pool: ['Max-pooling', 'The other maps are pooled the same way.'],
      units: nL
        ? ['Dense layer', `The ${K} pooled maps are stacked ${stack} into one map and laid over each hidden unit’s weight map (one weight per pooled cell, ${F}, stacked the same way), multiplied cell by cell into a product map. A scan line sums the map, orange cells against blue, the bias is added, and ReLU keeps the positive part: the unit’s value. Unit 1 slowly, then the rest together.`]
        : ['Output', `The ${K} pooled maps are stacked ${stack} into one map and laid over the output’s weight map (one weight per pooled cell, ${F}), multiplied cell by cell into a product map. A scan line sums the map, orange cells against blue; the sum plus the bias is z.`],
      hop: hop === 'out'
        ? ['Output', `The units’ values × their weights flow to the output, each connection as thick as the product it carries. Their sum plus the bias is z, and the sigmoid turns z into P(${esc(posName())}).`]
        : hop === 'sum'
        ? ['Output', `z goes through the sigmoid: P(${esc(posName())}) = 1 / (1 + e<sup>−z</sup>).`]
        : [`Hidden layer ${(hop || 0) + 1}`, `Each unit sums weight × value over the layer before, plus its bias, through ${NN.ACTIVATIONS[S.activation].label}.`],
      final: ['Forward pass done', lesson ? 'Every value is on the diagram.' : 'Every value is on the diagram. The weights are frozen: nothing is learned here.'],
    };
    const t = texts[phase]; return t ? { phase, title: t[0], text: t[1] } : null;
  }
  // Classify-next walk-through for convolutional networks, about 45 s in all:
  //   0. the mean nucleus is subtracted: the network sees the difference (4 s)
  //   1. filter 1 alone scans the whole image slowly, with its arithmetic spelled out (~12 s)
  //   2. the remaining filters scan together at a quicker pace (~7 s)
  //   3. map 1 is pooled slowly, block by block, with the block and its maximum shown (~4 s)
  //   4. the remaining maps are pooled together (~2.5 s)
  //   5. the pooled maps, stacked into one map, are laid over each unit's weight map and summed, exactly like the
  //      pixel hop of the dense recipes: unit 1 slowly (12 s), the rest together (6 s)
  //   6. the remaining dense hops wipe to the output (1.2 s each), then the call at the threshold, then the truth
  function animateConvClassify(s) {
    const net = S.net, plan = Viz.convPlan(net, true), hops = plan.hops; // the pooled-maps hop, the dense hops between hidden layers, the output
    const T = plan.T.concat([['call', 900], ['reveal', 900]]);
    const p = S.test.results.get(s.id).p, called = p >= S.test.threshold ? 1 : 0, thr = S.test.threshold;
    let shown = false;
    const P = makePlayer(T, null); S.test.player = P;
    const finish = () => { S.test.player = null; S.test.revealed.add(s.id); S.test.animating = false; renderTestPanel(); renderInspector(); renderTestGraph(); renderTestLine(s, p, called, 'done'); };
    const frame = now => {
      if (!S.test.animating || S.selected !== s || S.test.player !== P) return;
      P.tick(now); const cur = P.at();
      if (!cur) { finish(); return; }
      const phase = cur.phase[0], frac = cur.frac;
      const anim = phase === 'call' || phase === 'reveal' ? { phase, hops, reveal: hops, wipes: plan.hopKeys, banner: null, dir: 0 } : Viz.convAnim(net, plan, cur.phase, frac);
      if (phase === 'call') anim.banner = `P(${posName()}) = ${p.toFixed(2)} ${p >= thr ? '≥' : '<'} ${thr.toFixed(2)}, so the call is ${className(called)}`;
      if (phase === 'reveal') {
        Object.assign(anim, { y: s.label, truthName: className(s.label), correct: called === s.label, banner: `truth: ${className(s.label)} · ${called === s.label ? 'correct' : 'wrong'}` });
        if (!shown) { shown = true; S.test.revealed.add(s.id); renderTestPanel(); renderInspector(); }
      }
      renderTestGraph(0, anim);
      renderTestLine(s, p, called, phase, anim);
      if (P.playing) P.schedule();
    };
    P.frame = frame;
    P.schedule();
  }
  function renderCharts() {
    let stopAt = null; // the epoch of the lowest test loss so far: where early stopping on a held-out set would have stopped
    if (S.peek && S.history.length > 3) { let best = S.history[0]; for (const h of S.history) if (h.testLoss < best.testLoss) best = h; if (best.epoch > 0 && best.epoch < S.history[S.history.length - 1].epoch) stopAt = best.epoch; }
    const ceiling = S.labelNoise > 0 ? 1 - S.labelNoise : null;
    Viz.drawCurves($('chart-loss'), { history: S.history, key: 'loss', showTest: S.peek, maxEpoch: S.epochs, stopAt });
    Viz.drawCurves($('chart-acc'), { history: S.history, key: 'acc', showTest: S.peek, maxEpoch: S.epochs, stopAt, ceiling });
    $('curves-legend-ceiling').hidden = ceiling == null; $('curves-legend-stop').hidden = stopAt == null;
    $('loss-now').textContent = S.trainEval.loss.toFixed(3);
    $('acc-now').textContent = pct(S.trainEval.accuracy);
    $('curves-legend-test').hidden = !S.peek;
  }
  // the single-layer network as a weighted checklist
  function renderScorecard() {
    const card = $('scorecard');
    const show = S.net.hidden.length === 0 && !S.net.conv && S.mode !== 'pixels';
    card.hidden = !show; if (!show) return;
    const names = S.inputs.featureNames;
    const rows = names.map((n, i) => ({ n, w: S.net.Wo[i] })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const max = Math.max(1e-9, ...rows.map(r => Math.abs(r.w)));
    $('scorecard-body').innerHTML = rows.map(r => `<tr><td>${esc(r.n)}</td><td class="n" style="color:${r.w >= 0 ? 'var(--irregular)' : 'var(--regular)'}">${Viz.fmtSigned(r.w, 2)}</td><td><div class="bar"><i class="${r.w >= 0 ? 'pos' : 'neg'}" style="width:${(Math.abs(r.w) / max * 50).toFixed(1)}%"></i></div></td></tr>`).join('');
    $('scorecard-formula').innerHTML = `score = Σ weight × standardized value + bias (${Viz.fmtSigned(S.net.bo, 2)}) &nbsp;→&nbsp; P(${esc(posName())}) = 1 / (1 + e<sup>−score</sup>)`;
  }
  // which kinds of case make each first-layer unit fire
  function renderProfile(force) {
    const card = $('unit-profile');
    const H = S.net.hidden.length ? S.net.hidden[0] : 0;
    const show = H > 0 && S.trainEval && S.trainEval.acts;
    card.hidden = !show; if (!show) return;
    const now = performance.now();
    if (!force && now - S.profileAt < 400) return;
    S.profileAt = now;
    const subtypes = (S.task.subtypes || []).slice();
    const acts = S.trainEval.acts;
    const rows = subtypes.map(st => { const sums = new Float64Array(H); let n = 0; S.ds.train.forEach((s, k) => { if (s.subtype === st.key && S.inputs.labelled.has(s.id)) { n++; for (let j = 0; j < H; j++) sums[j] += acts[k][j]; } }); return { st, n, mean: Array.from(sums, v => (n ? v / n : 0)) }; }).filter(r => r.n > 0);
    const colMax = Array.from({ length: H }, (_, j) => Math.max(1e-9, ...rows.map(r => Math.abs(r.mean[j]))));
    const signed = S.activation === 'tanh';
    let html = `<table class="profile"><thead><tr><th class="row">kind of ${esc(S.task.specimenNoun || 'case')}</th>` + Array.from({ length: H }, (_, j) => `<th><span class="udot" style="background:${Viz.unitColor(j)}"></span>unit ${j + 1}</th>`).join('') + '</tr></thead><tbody>';
    for (const r of rows) {
      html += `<tr><td class="row"><span class="dot" style="background:${r.st.positive ? 'var(--irregular)' : 'var(--regular)'}"></span>${esc(r.st.name)}<span class="n">n=${r.n}</span></td>` +
        r.mean.map((v, j) => { const t = Math.abs(v) / colMax[j]; const bg = signed ? Viz.diverging(v / colMax[j], 0.25 + 0.75 * t) : Viz.sequential(t, 0.15 + 0.85 * t); return `<td style="background:${bg};color:${t > 0.6 ? '#fff' : 'var(--ink)'}">${v.toFixed(2)}</td>`; }).join('') + '</tr>';
    }
    if (S.net.hidden.length === 1) html += `<tr><td class="row">weight to output</td>` + Array.from({ length: H }, (_, j) => `<td class="out" style="color:${S.net.Wo[j] >= 0 ? 'var(--irregular)' : 'var(--regular)'}">${Viz.fmtSigned(S.net.Wo[j], 2)}</td>`).join('') + '</tr>';
    if (S.mode !== 'pixels') {
      const names = S.inputs.featureNames, D = S.net.sizes[0];
      html += `<tr><td class="row">responds most to</td>` + Array.from({ length: H }, (_, j) => {
        const ws = names.map((n, i) => ({ n, w: S.net.W[0][j * D + i] })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 3);
        return `<td class="top">${ws.map(w => `${w.w >= 0 ? '↑' : '↓'} ${esc(w.n)}`).join('<br>')}</td>`;
      }).join('') + '</tr>';
    }
    html += '</tbody></table>';
    $('unit-profile-body').innerHTML = html;
    $('unit-profile-note').textContent = signed
      ? 'Mean activation of each first-layer unit for each kind of training case (tanh: orange positive, blue negative). Nothing told the network these kinds exist; it only saw the label.'
      : 'Mean activation of each first-layer unit for each kind of training case, each column scaled to its own maximum. Nothing told the network these kinds exist; it only saw the label. Rows with a blue dot are negatives, orange are positives.';
  }
  function renderTraining(fresh) {
    if (fresh || !S.trainEval) evaluateAll();
    renderStatus();
    if (S.stage === 'train') { renderTrainTray(); renderTrainGraph(); renderCharts(); renderScorecard(); renderProfile(fresh === true && !S.running); }
    if (S.stage === 'test') { renderTestPanel(); renderTestGraph(); }
    renderInspector();
    updateTeachButton();
  }

  function inkToPx(ink) { const px = new Uint8ClampedArray(ink.length); for (let i = 0; i < ink.length; i++) px[i] = Math.round(255 * (1 - ink[i])); return px; }
  // ------------------------------------------------------------------ rendering: test panel
  function renderLabScores(st) {
    const box = $('lab-scores'); if (!S.ds.labs) { box.hidden = true; return; }
    if (st.n === S.ds.test.length) S.test.scores[S.ds.sources.test] = { acc: st.acc, sens: st.sens, spec: st.spec };
    const keys = Object.keys(S.test.scores); box.hidden = !keys.length; if (!keys.length) return;
    const L = SOURCE_LABELS(), f = v => (v == null ? '–' : pct(v));
    box.innerHTML = '<span class="hd">test cases from</span><span class="hd n">accuracy</span><span class="hd n">sensitivity</span><span class="hd n">specificity</span>' +
      DS.SOURCE_MODES.filter(k => S.test.scores[k]).map(k => { const r = S.test.scores[k], cur = k === S.ds.sources.test ? ' current' : ''; return `<span class="row${cur}">${esc(L[k])}</span><span class="row n${cur}">${f(r.acc)}</span><span class="row n${cur}">${f(r.sens)}</span><span class="row n${cur}">${f(r.spec)}</span>`; }).join('');
  }
  function renderTestPanel() {
    renderTestTray();
    const st = testStats();
    renderLabScores(st);
    $('stat-n').textContent = `${st.n} / ${S.ds.test.length}`;
    $('stat-n-sub').textContent = `held-out ${noun(2)}${S.ds.labs && S.ds.sources.test !== 'ours' ? ` · ${SOURCE_LABELS()[S.ds.sources.test].split(':')[0]}` : ''}`;
    $('stat-acc').textContent = st.acc == null ? '–' : pct(st.acc);
    $('stat-sens').textContent = st.sens == null ? '–' : pct(st.sens);
    $('stat-spec').textContent = st.spec == null ? '–' : pct(st.spec);
    $('stat-acc-sub').textContent = st.n ? `${st.tp + st.tn} of ${st.n} correct` : 'no calls yet';
    $('stat-sens-sub').textContent = st.tp + st.fn ? `${st.tp} of ${st.tp + st.fn} ${posName().toLowerCase()} caught` : `no ${posName().toLowerCase()} seen yet`;
    $('stat-spec-sub').textContent = st.tn + st.fp ? `${st.tn} of ${st.tn + st.fp} ${negName().toLowerCase()} cleared` : `no ${negName().toLowerCase()} seen yet`;
    const cell = (v, kind) => `<div class="cell ${v ? kind : 'empty'}">${v}</div>`;
    $('confusion').innerHTML =
      `<div></div><div class="hd">called ${esc(negName())}</div><div class="hd">called ${esc(posName())}</div>` +
      `<div class="rh">truth ${esc(negName())}</div>${cell(st.tn, 'hit')}${cell(st.fp, 'miss')}` +
      `<div class="rh">truth ${esc(posName())}</div>${cell(st.fn, 'miss')}${cell(st.tp, 'hit')}`;
    $('btn-classify-next').disabled = S.test.next >= S.ds.test.length;
    $('btn-classify-all').disabled = S.test.next >= S.ds.test.length;
    $('btn-classify-next').textContent = S.test.next >= S.ds.test.length ? `All ${S.ds.test.length} classified` : `Classify next (${S.test.next + 1} of ${S.ds.test.length})`;
    $('btn-classify-all').textContent = `Classify all ${S.ds.test.length}`;
    $('threshold-val').textContent = S.test.threshold.toFixed(2);
    $('test-warning').hidden = !(S.epoch === 0 && S.net.steps === 0);
    // prevalence -> predictive values
    const p = S.test.prevalence;
    $('prev-val').textContent = `1 in ${Math.round(1 / p).toLocaleString()}`;
    if (st.sens == null || st.spec == null) { $('ppv').textContent = '–'; $('npv').textContent = '–'; $('prev-text').textContent = `Classify some ${noun(2)} of both kinds first; the predictive values use the sensitivity and specificity measured above.`; }
    else {
      const ppv = st.sens * p / (st.sens * p + (1 - st.spec) * (1 - p)), npv = st.spec * (1 - p) / (st.spec * (1 - p) + (1 - st.sens) * p);
      $('ppv').textContent = pct(ppv); $('npv').textContent = pct(npv);
      const per = 100000;
      const tp = st.sens * p * per, fp = (1 - st.spec) * (1 - p) * per;
      $('prev-text').textContent = `In ${per.toLocaleString()} ${noun(2)} with this prevalence the network would call ${Math.round(tp + fp).toLocaleString()} “${posName()}”, of which ${Math.round(tp).toLocaleString()} truly are. The test set was half positive, so its accuracy says nothing about this.`;
    }
  }

  // ------------------------------------------------------------------ rendering: inspector
  function selectSpecimen(s, opts) {
    S.selected = s;
    renderDataTrays();
    if (S.stage === 'train') { renderTrainTray(); renderTrainGraph(); }
    if (S.stage === 'test') { renderTestTray(); if (!(opts && opts.silent)) renderTestGraph(); }
    if (S.stage === 'data') renderScatter();
    renderInspector();
    updateTeachButton();
  }
  function renderInspector() {
    const s = S.selected;
    const cv = $('spec-view');
    if (!s) return;
    const tabular = S.kind === 'tabular';
    $('spec-name').textContent = `${tabular ? 'Patient' : 'Specimen'} ${s.name}`;
    const known = truthKnown(s);
    $('spec-chips').innerHTML =
      `<span class="chip plain">${s.split === 'train' ? (S.inputs.labelled.has(s.id) ? 'Training set' : 'Training set · label withheld') : 'Test set · held out'}</span>` +
      (known ? `<span class="chip ${classOf(trainLabel(s))}">${labelWord(s)}: ${esc(className(trainLabel(s)))}</span>` : `<span class="chip plain">truth hidden</span>`) +
      (known && isFlipped(s) && S.showFlipped ? `<span class="chip bad">mislabelled · truth: ${esc(className(s.label))}</span>` : '') +
      (known && s.subtype && tabular ? `<span class="chip plain">${esc(subtypeName(s.subtype))}</span>` : '') +
      (s.lab === 'B' ? '<span class="chip plain">scanned at the other lab</span>' : '');
    document.querySelector('.views').hidden = tabular;
    document.querySelectorAll('.views button').forEach(b => b.classList.toggle('is-active', b.dataset.view === S.view));
    const m = s.measurement;
    const x = S.inputs.xOf(s);
    const classified = isClassified(s) && (s.split === 'train' || S.test.revealed.has(s.id));
    const untrainedHere = S.stage === 'data' && S.net.steps === 0;
    const fw = classified && !untrainedHere ? S.net.forward(x) : null;
    let caption = '';
    if (tabular) { Viz.renderFingerprint(cv, s.deviation, true); caption = 'Blood-count fingerprint: one bar per parameter in report order, up = above the reference range, down = below, dotted lines = the limits of the range.'; }
    else if (S.view === 'image') { Viz.renderBigImage(cv, s.px, s.size, S.tint); caption = `${s.size} × ${s.size} pixels, 8-bit grayscale${S.tint ? ', shown with an H&E tint' : ''}.`; }
    else if (S.view === 'measure') { Viz.renderMeasurement(cv, s.px, s.size, m, S.tint); caption = 'Violet: membrane found by thresholding · white dashes: convex hull · orange dots: smooth ellipse with the same area · shaded: where darkness and texture are read.'; }
    else {
      if (!fw) { Viz.renderBigImage(cv, s.px, s.size, S.tint); caption = untrainedHere ? 'Evidence appears once the network has trained (stage 2).' : 'Evidence appears once the network has classified this nucleus.'; }
      else if (S.mode === 'pixels') {
        const g = S.net.inputGradient(x, fw);
        const sal = new Float64Array(g.length);
        for (let i = 0; i < g.length; i++) sal[i] = g[i] * x[i];
        Viz.renderEvidence(cv, s.px, s.size, sal, S.tint);
        caption = `Orange pixels push the score toward ${posName()}, blue toward ${negName()} (weight × input at each pixel).`;
      } else if (S.mode === 'code') { // the classifier's evidence on the code, carried back through the frozen encoder to the pixels
        const enc = codeEncoder(), gCode = S.net.inputGradient(x, fw), dRaw = new Float64Array(gCode.length);
        for (let i = 0; i < gCode.length; i++) dRaw[i] = gCode[i] / S.inputs.std.scale[i];
        const xEnc = enc.std.apply(S.inputs.inkOf(s)), gPix = enc.cl.inputGradient(xEnc, dRaw), sal = new Float64Array(gPix.length);
        for (let i = 0; i < gPix.length; i++) sal[i] = gPix[i] * xEnc[i];
        Viz.renderEvidence(cv, s.px, s.size, sal, S.tint);
        caption = `Orange pixels push the score toward ${posName()}, blue toward ${negName()}: the evidence on the code, carried back through the frozen encoder to the pixels (a linear approximation).`;
      } else { Viz.renderMeasurement(cv, s.px, s.size, m, S.tint); caption = 'In measurement mode the evidence is per measurement — see the “push” column below.'; }
    }
    $('spec-caption').textContent = caption;
    renderLabCard(s, fw);

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
      v.querySelector('.p').innerHTML = `<small class="lbl">P(${esc(posName())})</small><b>${fw.p.toFixed(3)}</b><small>score z = ${Viz.fmtSigned(fw.z, 2)}</small>`;
      let chips = `<span class="chip ${classOf(call)}">call: ${esc(className(call))}</span>`;
      if (known) chips += call === trainLabel(s) ? `<span class="chip good">✓ agrees with ${labelWord(s) === 'label' ? 'the label' : 'truth'}</span>` : `<span class="chip bad">✗ ${labelWord(s)} is ${esc(className(trainLabel(s)))}</span>`;
      $('verdict-call').innerHTML = chips;
      $('pbar-marker').style.left = (fw.p * 100).toFixed(1) + '%';
      $('pbar-thr').style.left = (thr * 100).toFixed(1) + '%';
      const g = S.net.inputGradient(x, fw);
      let sum = 0; for (let i = 0; i < g.length; i++) sum += g[i] * x[i];
      const exact = !S.net.conv && S.net.hidden.length === 0;
      $('evidence-sum').innerHTML = `evidence Σ(weight × input) ${exact ? '=' : '≈'} ${Viz.fmtSigned(sum, 2)} &nbsp;·&nbsp; bias ${Viz.fmtSigned(S.net.bo, 2)} &nbsp;→&nbsp; z ${Viz.fmtSigned(fw.z, 2)} &nbsp;→&nbsp; P = 1 / (1 + e<sup>−z</sup>) = ${fw.p.toFixed(3)}${exact ? '' : '<br><span class="muted">(with hidden layers or a convolution the per-input evidence is a linear approximation)</span>'}`;
    }

    // measurement / blood-count table
    const featMode = S.mode === 'features';
    const g = fw && featMode ? S.net.inputGradient(x, fw) : null;
    const pushByFeature = new Array(S.featureDefs.length).fill(null);
    if (g) S.inputs.columns.forEach((fi, ci) => { pushByFeature[fi] = g[ci] * x[ci]; });
    let maxPush = 1e-9; for (const p of pushByFeature) if (p != null) maxPush = Math.max(maxPush, Math.abs(p));
    const zInputs = featMode ? S.inputs : inputsFor('features', false, new Set());
    const zx = zInputs.xOf(s);
    const zByFeature = new Array(S.featureDefs.length).fill(null);
    zInputs.columns.forEach((fi, ci) => { zByFeature[fi] = zx[ci]; });
    $('feat-card-title').textContent = tabular ? 'Blood count' : 'Measurements';
    $('feat-head').innerHTML = tabular
      ? `<th>parameter</th><th style="text-align:right">value</th><th class="ref">reference</th><th>flag</th><th style="text-align:right" title="standardized: training-set standard deviations from the training-set mean (log scale for counts)">z</th>${featMode ? '<th>push</th>' : ''}`
      : `<th>measurement</th><th style="text-align:right">value</th><th style="text-align:right" title="standardized: how many training-set standard deviations from the training-set mean">z</th>${featMode ? '<th>push</th>' : ''}`;
    $('feat-table').innerHTML = S.featureDefs.map((f, i) => {
      const val = s.features[i];
      const off = featMode && S.excluded.has(f.key);
      const push = pushByFeature[i];
      const w = push == null ? 0 : Math.min(50, Math.abs(push) / maxPush * 50);
      const bar = featMode ? `<td>${off ? '<span class="muted small">withheld</span>' : `<div class="bar" title="push ${push == null ? '' : Viz.fmtSigned(push, 2)}">${push != null ? `<i class="${push >= 0 ? 'pos' : 'neg'}" style="width:${w.toFixed(1)}%"></i>` : ''}</div>`}</td>` : '';
      const z = zByFeature[i];
      const zTxt = z == null ? '' : Viz.fmtSigned(z, 1);
      if (tabular) {
        const flag = f.high === f.low ? (val > f.high ? 'H' : '') : (val > f.high ? 'H' : val < f.low ? 'L' : '');
        return `<tr class="${off ? 'off' : ''}"><td class="name" title="${esc(f.desc)}">${esc(f.name)}</td><td class="n">${f.fmt(val)} <span class="muted">${esc(f.unit)}</span></td><td class="ref">${f.low === f.high ? f.low : `${f.low}–${f.high}`}</td><td class="flag ${flag}">${flag}</td><td class="n">${zTxt}</td>${bar}</tr>`;
      }
      return `<tr class="${off ? 'off' : ''}"><td class="name" title="${esc(f.desc)}">${esc(f.name)}</td><td class="n">${f.fmt(val)}${f.unit ? ' ' + f.unit : ''}</td><td class="n">${zTxt}</td>${bar}</tr>`;
    }).join('');
    $('feat-note').textContent = featMode
      ? `push = weight × standardized value: how far this ${tabular ? 'parameter' : 'measurement'} moves the score (orange → ${posName()}, blue → ${negName()}).${S.excluded.size ? ' Withheld inputs are not given to the network.' : ''}`
      : `In ${S.mode === 'code' ? 'code' : 'pixel'} mode the network never sees these measurements — they are here for you, the human.`;
    renderCodeCard(s, x, fw);
  }
  // the code the encoder gives this case: its numbers, standardized, and each one's push on the score
  function renderCodeCard(s, x, fw) {
    const card = $('code-card'); card.hidden = S.mode !== 'code'; if (card.hidden) return;
    const raw = S.inputs.rawOf(s), g = fw ? S.net.inputGradient(x, fw) : null;
    let maxPush = 1e-9; if (g) for (let i = 0; i < g.length; i++) maxPush = Math.max(maxPush, Math.abs(g[i] * x[i]));
    $('code-table').innerHTML = Array.from(raw, (v, i) => {
      const push = g ? g[i] * x[i] : null, w = push == null ? 0 : Math.min(50, Math.abs(push) / maxPush * 50);
      return `<tr><td class="name">code ${i + 1}</td><td class="n">${Viz.fmtSigned(v, 2)}</td><td class="n">${Viz.fmtSigned(x[i], 1)}</td><td><div class="bar" title="push ${push == null ? '' : Viz.fmtSigned(push, 2)}">${push != null ? `<i class="${push >= 0 ? 'pos' : 'neg'}" style="width:${w.toFixed(1)}%"></i>` : ''}</div></td></tr>`;
    }).join('');
    $('code-note').textContent = `${raw.length} numbers from ${codeEncoder().describe}, standardized on the labelled training cases; push = weight × standardized value (orange → ${posName()}, blue → ${negName()}).`;
  }

  // the same nucleus as scanned at both labs, with the network's call for each scan once it may be shown
  function renderLabCard(s, fw) {
    const card = $('lab-card'); card.hidden = !(S.ds.labs && s.variants && s.variants.B); if (card.hidden) return;
    const thr = s.split === 'test' ? S.test.threshold : 0.5;
    for (const [lab, id] of [['A', 'lab-a'], ['B', 'lab-b']]) {
      const el = $(id), v = s.variants[lab];
      Viz.renderBigImage(el.querySelector('canvas'), v.px, s.size, S.tint);
      el.classList.toggle('current', s.lab === lab);
      const p = el.querySelector('.p');
      if (!fw) { p.textContent = ''; p.className = 'p'; continue; }
      const q = S.net.forward(S.inputs.xFor(s, lab)).p, call = q >= thr ? 1 : 0;
      p.textContent = `P ${q.toFixed(2)} · ${className(call)}`; p.className = `p ${classOf(call)}`;
    }
    $('lab-note').textContent = fw
      ? `Same shape, same chromatin, weaker stain${S.mode === 'features' ? '; the measurements are taken from each scan' : S.normalize !== 'off' ? '; both scans are stain-normalised before the network sees them' : ''}. The framed scan is the one in the ${s.split === 'train' ? 'training' : 'test'} set.`
      : 'Same shape, same chromatin, weaker stain. The network’s call for each scan appears once it has classified this nucleus.';
  }

  // ------------------------------------------------------------------ data panel
  function renderScatter() {
    const xi = +$('scatter-x').value, yi = +$('scatter-y').value;
    const fx = S.featureDefs[xi], fy = S.featureDefs[yi];
    const tx = v => (fx.log ? Math.log10(Math.max(v, 1e-3)) : v), ty = v => (fy.log ? Math.log10(Math.max(v, 1e-3)) : v);
    const pts = S.ds.specimens.filter(s => s.split === 'train' || S.revealTest).map(s => ({
      id: s.id, name: s.name, x: tx(s.features[xi]), y: ty(s.features[yi]), split: s.split,
      cls: truthKnown(s) ? classOf(trainLabel(s)) : 'unknown', clsName: truthKnown(s) ? className(trainLabel(s)) + (isFlipped(s) && S.showFlipped ? ' · mislabelled' : '') + (s.subtype ? ' · ' + subtypeName(s.subtype) : '') : '', selected: S.selected && S.selected.id === s.id,
    }));
    Viz.drawScatter($('scatter'), { points: pts, xLabel: (fx.log ? 'log₁₀ ' : '') + fx.name, yLabel: (fy.log ? 'log₁₀ ' : '') + fy.name, onSelect: id => selectSpecimen(S.ds.specimens[id]) });
  }

  // ------------------------------------------------------------------ controls
  function renderInputPicker() {
    $('input-picker').innerHTML = S.featureDefs.map(f => `<label class="${S.excluded.has(f.key) ? 'off' : ''}"><input type="checkbox" data-key="${esc(f.key)}" ${S.excluded.has(f.key) ? '' : 'checked'}> ${esc(f.name)}</label>`).join('');
    $('input-picker').querySelectorAll('input').forEach(cb => cb.addEventListener('change', () => {
      const key = cb.dataset.key;
      if (!cb.checked && S.featureDefs.length - S.excluded.size <= 2) { cb.checked = true; note('Keep at least two inputs.'); return; }
      if (cb.checked) S.excluded.delete(key); else S.excluded.add(key);
      const name = S.featureDefs.find(f => f.key === key).name;
      renderInputPicker();
      resetModel(cb.checked ? `${name} restored — fresh random weights.` : `${name} withheld from the network — fresh random weights.`);
    }));
  }
  function applyVisibility() {
    const tabular = S.kind === 'tabular', pixels = S.mode === 'pixels', code = S.mode === 'code';
    $('mode-wrap').hidden = tabular;
    $('conv-wrap').hidden = !pixels;
    $('augment-wrap').hidden = !pixels;
    $('picker-wrap').hidden = pixels || code;
    $('code-wrap').hidden = !code;
    $('tint-wrap').hidden = tabular;
    $('prevalence-card').hidden = !tabular;
    $('legend-conv').hidden = !(pixels && S.convK > 0);
    renderSources();
  }
  function syncControls() {
    applyVisibility();
    document.querySelectorAll('#mode-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.mode === S.mode));
    $('hidden').value = S.h1; $('hidden-val').textContent = S.h1 === 0 ? 'none' : S.h1;
    $('hidden2').value = S.h2; $('hidden2-val').textContent = S.h1 === 0 ? '–' : (S.h2 === 0 ? 'none' : S.h2); $('hidden2').disabled = S.h1 === 0;
    $('conv').value = S.mode === 'pixels' ? S.convK : 0; $('conv').disabled = S.mode !== 'pixels';
    $('activation').value = S.activation; $('activation').disabled = S.h1 === 0;
    $('lr').value = sliderFromLr(S.lr); $('lr-val').textContent = S.lr;
    $('batch').value = S.batch;
    $('epochs').value = S.epochs; $('epochs-val').textContent = S.epochs;
    $('speed').value = sliderFromSpeed(S.speed); $('speed-val').textContent = `${S.speed} epochs/s`;
    $('seed').value = S.seed;
    $('augment').checked = S.augment; $('augment').disabled = S.mode !== 'pixels';
    $('augment-wrap').classList.toggle('muted', S.mode !== 'pixels');
    $('l2').value = S.l2; $('l2-val').textContent = S.l2 === 0 ? 'off' : S.l2.toFixed(2);
    $('peek').checked = S.peek;
    $('labelled').value = String(S.labelled); $('labelled').querySelector('option[value="0"]').textContent = `all ${S.ds.train.length}`;
    if (S.backbone !== 'page' && !backbones().some(b => b.id === S.backbone)) S.backbone = defaultBackbone();
    $('backbone').value = S.backbone;
    renderNoiseControls();
    $('threshold').value = S.test.threshold;
    $('prev').value = sliderFromPrev(S.test.prevalence);
    document.querySelectorAll('#tray-color-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.color === S.trayColor));
    renderInputPicker();
  }
  function bindControls() {
    $('question-select').addEventListener('change', () => {
      const id = $('question-select').value;
      if (S.taskId === id) return;
      loadTask(id);
      if (S.kind === 'image' && S.mode === 'features' && S.lr < 0.05) S.lr = 0.1;
      syncControls();
      resetModel(`Question changed to “${S.task.title}” — new ${noun(2)}, fresh random weights.`);
      renderDataTrays(); renderScatter(); renderInspector();
    });
    document.querySelectorAll('#mode-seg button').forEach(b => b.addEventListener('click', () => {
      if (S.mode === b.dataset.mode || b.disabled) return;
      S.mode = b.dataset.mode;
      if (S.mode === 'pixels' && S.lr > 0.05) S.lr = 0.01;
      if (S.mode !== 'pixels' && S.lr < 0.05) S.lr = 0.1;
      syncControls(); resetModel(`Input changed to ${S.mode === 'pixels' ? 'raw pixels' : S.mode === 'code' ? `the code from ${codeEncoder().describe}` : (S.kind === 'tabular' ? 'the blood count' : 'measurements')} — fresh random weights.`);
    }));
    $('hidden').addEventListener('input', () => { S.h1 = +$('hidden').value; syncControls(); resetModel(`Hidden layer 1 set to ${S.h1 || 'none'} — fresh random weights.`); });
    $('hidden2').addEventListener('input', () => { S.h2 = +$('hidden2').value; syncControls(); resetModel(`Hidden layer 2 set to ${S.h2 || 'none'} — fresh random weights.`); });
    $('conv').addEventListener('change', () => { S.convK = +$('conv').value; resetModel(S.convK ? `Convolutional layer with ${S.convK} filters — fresh random weights.` : 'Convolution removed — fresh random weights.'); });
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
    $('labelled').addEventListener('change', () => { S.labelled = +$('labelled').value; resetModel(S.labelled ? `${S.labelled} labelled training cases: the other ${S.ds.train.length - S.labelled} stay in the tray without a label, and the network never sees them — fresh random weights.` : 'Every training case labelled again — fresh random weights.'); renderDataTrays(); });
    $('backbone').addEventListener('change', () => { S.backbone = $('backbone').value; resetModel(`The code now comes from ${codeEncoder().describe} — fresh random weights.`); });
    for (const id of ['source-train', 'source-train-2']) $(id).addEventListener('change', ev => setTrainSource(ev.target.value));
    for (const id of ['source-test', 'source-test-2']) $(id).addEventListener('change', ev => setTestSource(ev.target.value));
    $('normalize').addEventListener('change', () => { S.normalize = $('normalize').value; syncControls(); resetModel(S.normalize === 'off' ? 'Stain normalisation off — fresh random weights.' : `Stain normalisation ${S.normalize === 'lab' ? 'per lab' : 'per image'} — fresh random weights.`); });
    $('show-lab').addEventListener('change', () => { S.showLab = $('show-lab').checked; renderDataTrays(); if (S.stage === 'train') renderTrainTray(); if (S.stage === 'test') renderTestTray(); });
    $('noise').addEventListener('input', () => { const v = +$('noise').value; $('noise-val').textContent = v ? `${v}% of the labels wrong` : 'off'; });
    $('noise').addEventListener('change', () => {
      S.labelNoise = +$('noise').value / 100; syncControls();
      resetModel(S.labelNoise ? `${Math.round(S.labelNoise * 100)}% of the training labels flipped (${S.inputs.flipped.size} of ${S.ds.train.length}), as a second pathologist might have called them — fresh random weights.` : 'Every training label right again — fresh random weights.');
      renderDataTrays(); renderScatter(); renderInspector();
    });
    for (const id of ['show-flipped', 'show-flipped-2']) $(id).addEventListener('change', ev => { S.showFlipped = ev.target.checked; syncControls(); renderDataTrays(); renderScatter(); renderInspector(); if (S.stage === 'train') renderTrainTray(); });
    $('btn-train').addEventListener('click', () => (S.running ? stopTraining('Paused.') : startTraining()));
    $('btn-step-batch').addEventListener('click', stepBatch);
    $('btn-step-epoch').addEventListener('click', stepEpoch);
    $('btn-reset').addEventListener('click', () => resetModel('Weights re-initialised from the seed.'));
    $('btn-teach').addEventListener('click', teachNext);
    $('recipe-select').addEventListener('change', () => {
      const k = $('recipe-select').value; if (!k) return;
      if (k === '11') { // the foundation model has its own stage and controls
        showStage('foundation');
        S.fm.labs = true; S.fm.epochs = 100; S.fm.speed = 4;
        fmReset(`Recipe ${RECIPE_LABELS[11]}: 100 nuclei from all three questions, no labels. Press Pretrain and watch the right-hand curve: a single layer on the code gets better at every question, though the code was never told what any of them asks.`);
        return;
      }
      const { task: taskId, ...settings } = RECIPES[k];
      const switchTask = taskId !== S.taskId;
      if (switchTask) loadTask(taskId);
      if (settings.mode === 'code') S.backbone = S.fm.cl && S.fm.epoch > 0 ? 'page' : defaultBackbone(); // the encoder you pretrained, if you did; else the shipped copy of it
      Object.assign(S, LAB_SETTINGS, settings);
      applySources();
      S.excluded = new Set();
      S.applyingRecipe = true;
      syncControls();
      resetModel(`Recipe ${RECIPE_LABELS[k]}${switchTask ? ` — question switched to “${S.task.title}”` : ''}. Press Train.`);
      S.applyingRecipe = false;
      $('recipe-select').value = k;
      renderDataTrays(); renderScatter(); renderInspector();
      if (S.stage !== 'train') showStage('train');
      if (k === '8') note(`Recipe ${RECIPE_LABELS[8]}: after training, go to 3 · Test, classify all, then switch “Test cases from” to the other lab and classify again: this network falls to chance on the paler scans. Stain normalisation per lab (Advanced settings) repairs it after retraining.`);
      if (k === '9') note(`Recipe ${RECIPE_LABELS[9]}: every irregular training nucleus was scanned at the other lab. Train, test on the matching test set, then switch the test cases to our lab. Watch the first-layer weight maps: they turn into plain interior templates instead of contour detectors, and the Evidence view weighs the inside of the nucleus.`);
      if (k === '12') note(`Recipe ${RECIPE_LABELS[12]}: the network sees 10 labelled nuclei only, each as ${S.inputs.inputSize} numbers from ${codeEncoder().describe}. Train, then compare: set “Labelled cases” to 10 on recipe ⑥ (pixels), or switch this input to measurements with the same 10 cases; with 40 labelled cases the code pulls further ahead. Before training, seed 8 starts near chance; roll the dice and the untrained network can start anywhere from 20% to 85%, because every code number already carries information and a random weighting of them is already a classifier, right or wrong.`);
      if (k === '4') note(`Recipe ${RECIPE_LABELS[4]}: ${S.inputs.flipped.size} of the ${S.ds.train.length} training labels are wrong. Train with the test set peeking: the test curve peaks early and then falls while training accuracy climbs past the honest ceiling, as the network memorises the mislabelled cases. Tick “Mark the mislabelled cases” to watch it happen.`);
    });
    document.querySelectorAll('#tray-color-seg button').forEach(b => b.addEventListener('click', () => { S.trayColor = b.dataset.color; syncControls(); renderTrainTray(); }));
    document.querySelectorAll('.views button').forEach(b => b.addEventListener('click', () => { S.view = b.dataset.view; renderInspector(); }));
    $('reveal-test').addEventListener('change', () => { S.revealTest = $('reveal-test').checked; renderDataTrays(); renderScatter(); renderInspector(); });
    $('scatter-x').addEventListener('change', renderScatter);
    $('scatter-y').addEventListener('change', renderScatter);
    $('btn-classify-next').addEventListener('click', () => classifyNext(false));
    $('btn-classify-all').addEventListener('click', classifyAll);
    $('btn-test-reset').addEventListener('click', () => { clearTestResults(false); $('test-note').textContent = 'Test results cleared.'; renderInspector(); renderTestGraph(); });
    $('threshold').addEventListener('input', () => { S.test.threshold = +$('threshold').value; renderTestPanel(); renderInspector(); });
    $('prev').addEventListener('input', () => { S.test.prevalence = prevFromSlider(+$('prev').value); renderTestPanel(); });
    $('tint').addEventListener('change', () => { S.tint = $('tint').checked; repaintThumbs(); fmRepaint(); renderInspector(); if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
    $('theme-toggle').addEventListener('click', () => {
      const root = document.documentElement;
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('nucleus-net-theme', root.dataset.theme); } catch (e) { /* ignore */ }
      onThemeChange();
    });
    document.querySelectorAll('.stage').forEach(b => b.addEventListener('click', () => showStage(b.dataset.stage)));
    for (const [key, cvId, tipId] of [['train', 'net-canvas', 'net-tip'], ['test', 'net-canvas-test', 'net-tip-test']]) {
      const cv = $(cvId), tip = $(tipId);
      cv.addEventListener('mousemove', ev => {
        const m = cv._model; if (!m) return;
        if (key === 'test' && S.test.animating) return;
        const r = cv.getBoundingClientRect();
        const hit = Viz.hitNetwork(cv, ev.clientX - r.left, ev.clientY - r.top, m);
        const prev = S.hover[key];
        S.hover[key] = hit;
        if (hit) { tip.hidden = false; tip.textContent = hit.text; tip.style.left = (ev.clientX - r.left) + 'px'; tip.style.top = (ev.clientY - r.top) + 'px'; }
        else tip.hidden = true;
        if ((prev && prev.ref) !== (hit && hit.ref) || (hit && hit.ref && (hit.ref.kind === 'map' || hit.ref.kind === 'fmap' || hit.ref.kind === 'pooled' || hit.ref.kind === 'product' || hit.ref.kind === 'uprod' || hit.ref.kind === 'square'))) { key === 'train' ? renderTrainGraph() : renderTestGraph(); }
      });
      cv.addEventListener('mouseleave', () => { S.hover[key] = null; tip.hidden = true; key === 'train' ? renderTrainGraph() : renderTestGraph(); });
      if (key === 'test') cv.addEventListener('click', () => { if (S.test.animating && S.test.player) S.test.player.skip(); });
      if (key === 'train') cv.addEventListener('click', () => { if (S.lesson) S.lesson.player.skip(); });
    }
    document.addEventListener('keydown', ev => {
      if (ev.target.matches('input, select, textarea')) return;
      const P = activePlayer(); // a running walk-through takes space and the arrow keys, even from a focused button
      if (P && ev.key === ' ') { ev.preventDefault(); P.toggle(); return; }
      if (P && ev.key === 'ArrowRight') { ev.preventDefault(); P.next(); return; }
      if (P && ev.key === 'ArrowLeft') { ev.preventDefault(); P.prev(); return; }
      if (ev.target.matches('button') && ev.key === ' ') return;
      if (ev.key === ' ' && S.stage === 'train') { ev.preventDefault(); S.running ? stopTraining('Paused.') : startTraining(); }
      else if (ev.key === ' ' && S.stage === 'foundation') { ev.preventDefault(); S.fm.running ? fmStop('Paused.') : fmStart(); }
      else if ((ev.key === 'n' || ev.key === 'N') && S.stage === 'test') classifyNext(false);
      else if ((ev.key === 'n' || ev.key === 'N' || ev.key === 't' || ev.key === 'T') && S.stage === 'train') teachNext();
      else if (ev.key === '1') showStage('data'); else if (ev.key === '2') showStage('train'); else if (ev.key === '3') showStage('test'); else if (ev.key === '4') showStage('foundation');
    });
    window.addEventListener('resize', () => { if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); if (S.stage === 'foundation' && S.fm.cl) { fmRenderLineup(); fmRenderGraph(); fmRenderViews(); fmRenderBatch(); } });
    // playback controls in both strips: previous step, pause/play, next step, speed
    for (const key of ['lesson', 'test']) {
      const P = () => (key === 'lesson' ? (S.lesson && S.lesson.player) : (S.test.animating && S.test.player)) || null;
      $(`${key}-prev`).addEventListener('click', ev => { const p = P(); if (p) p.prev(); ev.currentTarget.blur(); });
      $(`${key}-next`).addEventListener('click', ev => { const p = P(); if (p) p.next(); ev.currentTarget.blur(); });
      $(`${key}-pause`).addEventListener('click', ev => { const p = P(); if (p) p.toggle(); ev.currentTarget.blur(); });
      $(`${key}-speed`).addEventListener('input', ev => setAnimSpeed(+ev.target.value));
    }
    try { const v = parseFloat(localStorage.getItem('nn-anim-speed')); if (v >= 0.25 && v <= 3) S.animSpeed = v; } catch (e) { /* ignore */ }
    syncSpeedControls();
  }
  function onThemeChange() {
    Viz.refreshTheme();
    const dark = getComputedStyle(document.documentElement).colorScheme.includes('dark');
    $('theme-toggle').textContent = dark ? '☀ Light' : '☾ Dark';
    repaintThumbs();
    if (S.stage === 'train') { renderTrainGraph(); renderProfile(true); } if (S.stage === 'test') renderTestGraph();
    fmRepaint();
    renderInspector();
  }
  function showStage(name) {
    S.stage = name;
    document.querySelectorAll('.stage').forEach(b => b.classList.toggle('is-active', b.dataset.stage === name));
    $('panel-data').hidden = name !== 'data'; $('panel-train').hidden = name !== 'train'; $('panel-test').hidden = name !== 'test'; $('panel-foundation').hidden = name !== 'foundation';
    document.querySelector('.inspector').hidden = name === 'foundation'; document.querySelector('.bench').classList.toggle('no-inspector', name === 'foundation'); // the foundation stage has no specimen inspector
    if (name !== 'foundation') fmStop();
    if (name === 'foundation') { stopTraining(); fmEnter(); }
    if (name === 'data') { renderDataTrays(); renderScatter(); renderInspector(); }
    if (name === 'train') {
      if (S.selected && S.selected.split !== 'train') S.selected = S.ds.train[0];
      if (S.mode === 'code' && S.inputs && S.inputs.encoderKey !== codeEncoder().key) resetModel('The code follows the encoder pretrained in 4 · Foundation, which has changed — fresh random weights on the new code.');
      renderTraining(true);
    }
    if (name === 'test') {
      stopTraining(); renderTestPanel();
      if (!S.selected || S.selected.split !== 'test') { S.selected = S.ds.test[Math.max(0, S.test.next - 1)]; }
      renderTestGraph(); renderInspector(); renderDataTrays();
      if (S.net.conv && !$('test-note').textContent) $('test-note').textContent = 'Classify next walks through the convolution (about 50 s): the mean nucleus is subtracted, filter 1 scans the difference slowly with its arithmetic shown, the other filters follow together, map 1 is pooled block by block and the other maps follow, then the pooled maps are stacked and laid over each hidden unit’s weight map and summed, and the units feed the output. Press N or click the diagram to skip ahead; space pauses, ← → step.';
    }
    try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------------ 4 · Foundation: a miniature foundation model, pretrained live
  // 100 nuclei from the three image questions' training sets (34 / 33 / 33) and no labels. The encoder (4 filters 5×5,
  // pool 4×4, a linear layer to a code of 8) is pretrained by instance discrimination: a batch takes 20 nuclei and two
  // random views of each (a flip or rotation, and the other lab's scan when allowed); the loss pulls the two views of a
  // nucleus together and pushes every other view of the batch away. After every epoch the code is measured on the side:
  // the contrastive loss on 60 held-out nuclei, and what a single layer trained on the frozen code with 20 labelled cases
  // per question scores on 40 nuclei it never saw.
  const FM = { tasks: ['atypia', 'enlargement', 'irregularity'], counts: [34, 33, 33], heldPer: 20, heldBatch: 20, batch: 20, lr: 0.05, K: 4, code: 8, tau: 0.2,
    probePerClass: 10, probeEpochs: 150, probeLr: 0.1, colors: { atypia: '#7c3aed', enlargement: '#0e9f6e', irregularity: '#d97706' } };
  const ORIENT = ['as scanned', '90°', '180°', '270°', 'mirrored', 'mirror+90°', 'mirror+180°', 'mirror+270°'];
  const ORIENT_LONG = ['as scanned', 'rotated 90°', 'rotated 180°', 'rotated 270°', 'mirrored', 'mirrored, rotated 90°', 'mirrored, rotated 180°', 'mirrored, rotated 270°'];
  const fmViewLabel = v => `${ORIENT[v.t]}${v.lab === 'B' ? ' · other lab' : ''}`;
  const fmViewLabelLong = v => `${ORIENT_LONG[v.t]}${v.lab === 'B' ? ', the other lab’s scan' : ''}`;
  const dot = (a, b) => { let s = 0; for (let d = 0; d < a.length; d++) s += a[d] * b[d]; return s; };
  function fmNote(msg) { $('fm-note').textContent = msg || ''; }
  function fmBatchesPerEpoch() { return Math.ceil(S.fm.set.length / FM.batch); }
  // a view of a nucleus: one lab's scan in one of the 8 orientations, standardised like every input to the encoder
  function fmMakeView(e, lab, t) {
    const F = S.fm, v = e.s.variants[lab] || e.s.variants.A;
    return { lab, t, px: t ? DS.dihedral(v.px, F.size, t) : v.px, x: F.std.apply(t ? DS.dihedral(v.ink, F.size, t) : v.ink) };
  }
  function fmRandomView(e, rng) { const labs = S.fm.labs ? ['A', 'B'] : ['A']; const lab = labs[Math.floor(rng() * labs.length)], t = Math.floor(rng() * 8); return fmMakeView(e, lab, t); }
  function fmFixedViews(e) { return [fmMakeView(e, 'A', 0), fmMakeView(e, S.fm.labs ? 'B' : 'A', 1)]; } // the pair the map shows, and a nucleus not yet batched
  function fmBuild() {
    const F = S.fm;
    F.dsets = {}; F.set = []; F.held = []; F.size = 0;
    FM.tasks.forEach((id, k) => {
      const ds = DS.prepare(S.tasks[id]); F.dsets[id] = ds; F.size = ds.size;
      ds.train.slice(0, FM.counts[k]).forEach(s => F.set.push({ s, task: id, i: F.set.length }));
      ds.train.slice(FM.counts[k], FM.counts[k] + FM.heldPer).forEach(s => F.held.push({ s, task: id }));
    });
    F.std = NN.fitStandardizer(F.set.map(e => e.s.variants.A.ink), { perDimScale: false }); // one scale for every pixel, fitted on our lab's scans of the 100
    const tray = $('fm-tray'); tray.innerHTML = '';
    F.thumbs = F.set.map(e => { const b = fmThumb(e); tray.appendChild(b); return b; });
    const opts = Array.from({ length: FM.code }, (_, j) => `<option value="${j}">code number ${j + 1}</option>`).join('');
    $('fm-x').innerHTML = opts; $('fm-y').innerHTML = opts;
    $('fm-worth-legend').innerHTML = FM.tasks.map(id => `<span><span class="line" style="border-top-color:${FM.colors[id]}"></span>${esc(F.dsets[id].task.title)}</span>`).join('');
    fmBuildHeld();
    fmSyncControls();
  }
  // the held-out nuclei get two views each, drawn once from a fixed seed, in batches of 20 pairs so that their loss reads
  // on the same scale as the training batches' (at chance, −log of one over the other views in the batch)
  function fmBuildHeld() {
    const F = S.fm, rng = NN.mulberry32(99), pairs = F.held.map(e => [fmRandomView(e, rng).x, fmRandomView(e, rng).x]);
    F.heldBatches = []; for (let b = 0; b < pairs.length; b += FM.heldBatch) F.heldBatches.push(pairs.slice(b, b + FM.heldBatch));
  }
  function fmHeldEval() { const F = S.fm; let loss = 0, hit = 0; for (const b of F.heldBatches) { const r = F.cl.evaluate(b); loss += r.loss / F.heldBatches.length; hit += r.pairAcc / F.heldBatches.length; } return { loss, pairAcc: hit }; }
  // what the frozen code is worth: a single layer trained on it with 20 labelled training cases of a question (10 per
  // class, our lab's scans as they are), scored on 40 nuclei it never saw: the question's 20 test nuclei and its 20
  // held-out training nuclei
  function fmProbes() {
    const F = S.fm, out = {};
    for (const id of FM.tasks) {
      const ds = F.dsets[id], tr = [...ds.train.filter(s => s.label).slice(0, FM.probePerClass), ...ds.train.filter(s => !s.label).slice(0, FM.probePerClass)];
      const code = s => F.cl.encode(F.std.apply(s.variants.A.ink));
      const rows = tr.map(code), labels = tr.map(s => s.label), st = NN.fitStandardizer(rows, { perDimScale: true });
      const net = new NN.Net({ inputSize: FM.code, hidden: [], activation: 'relu', seed: 1 }), rnd = NN.mulberry32(1), idx = rows.map((_, i) => i);
      for (let ep = 0; ep < FM.probeEpochs; ep++) {
        for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
        for (let b = 0; b < idx.length; b += 8) { const bb = idx.slice(b, b + 8); net.trainBatch(bb.map(i => st.apply(rows[i])), bb.map(i => labels[i]), FM.probeLr, 0); }
      }
      const scored = [...ds.test, ...F.held.filter(e => e.task === id).map(e => e.s)];
      out[id] = net.evaluate(scored.map(s => st.apply(code(s))), scored.map(s => s.label)).accuracy;
    }
    return out;
  }
  function fmThumb(e) {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'thumb'; b.dataset.id = e.i;
    const cv = document.createElement('canvas'); cv.width = cv.height = S.fm.size; b.appendChild(cv);
    Viz.renderThumb(cv, e.s.variants.A.px, S.fm.size, S.tint);
    b.addEventListener('click', () => fmSelect(e.i));
    return b;
  }
  function fmRepaint() { const F = S.fm; if (!F.thumbs) return; F.set.forEach((e, i) => Viz.renderThumb(F.thumbs[i].querySelector('canvas'), e.s.variants.A.px, F.size, S.tint)); if (F.cl && S.stage === 'foundation') fmRender(true); }
  function fmReset(reason) {
    const F = S.fm; fmStop();
    if (!F.set) fmBuild();
    F.cl = new NN.Contrastive({ imageSize: F.size, conv: { K: FM.K, f: 5, pool: 4 }, code: FM.code, tau: FM.tau, seed: F.seed });
    F.cl0 = new NN.Contrastive({ imageSize: F.size, conv: { K: FM.K, f: 5, pool: 4 }, code: FM.code, tau: FM.tau, seed: F.seed }); // the same encoder as it started, kept for comparison
    F.gen = (F.gen || 0) + 1; // which encoder the code input was built from
    F.rng = NN.mulberry32(F.seed * 31 + 7); F.order = F.set.map((_, i) => i);
    F.epoch = 0; F.ptr = 0; F.debt = 0; F.hist = []; F.lastBatch = null; F.epochLoss = 0; F.epochBatches = 0; F.codes = null; F.batchCache = null; F.selected = null;
    for (const e of F.set) { e.fixed = fmFixedViews(e); e.views = e.fixed; }
    F.defaultBatch = F.set.slice(0, FM.batch).map(e => ({ e, v1: e.views[0], v2: e.views[1] }));
    fmRecordEpoch(null);
    fmSyncControls(); fmRender(true);
    if (reason) fmNote(reason);
  }
  function fmShuffle() { const o = S.fm.order, rng = S.fm.rng; for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } }
  function fmRecordEpoch(loss) {
    const F = S.fm, held = fmHeldEval(), probes = fmProbes();
    F.hist.push(Object.assign({ epoch: F.epoch, loss, heldLoss: held.loss, pairAcc: held.pairAcc }, probes));
  }
  // one gradient step: the next 20 nuclei of the epoch's order, two fresh random views each
  function fmStep() {
    const F = S.fm, n = F.set.length;
    if (F.ptr === 0) fmShuffle();
    const end = Math.min(n, F.ptr + FM.batch), idx = F.order.slice(F.ptr, end);
    const views = idx.map(i => { const e = F.set[i], v1 = fmRandomView(e, F.rng), v2 = fmRandomView(e, F.rng); e.views = [v1, v2]; return { e, v1, v2 }; });
    const res = F.cl.step(views.map(v => [v.v1.x, v.v2.x]), FM.lr, 0);
    F.lastBatch = { views, res, ids: new Set(idx), index: Math.ceil(end / FM.batch) };
    F.epochLoss += res.loss; F.epochBatches++;
    F.ptr = end >= n ? 0 : end;
    let ended = false;
    if (F.ptr === 0) { F.epoch++; fmRecordEpoch(F.epochLoss / F.epochBatches); F.epochLoss = 0; F.epochBatches = 0; ended = true; }
    return ended;
  }
  function fmStart() {
    const F = S.fm;
    if (F.epoch >= F.epochs) { fmNote(`Already at ${F.epochs} epochs. Raise the epoch count, or reset to pretrain again.`); return; }
    F.running = true; F.lastTime = performance.now(); F.debt = 0; F.lastRender = 0; fmSyncButtons();
    requestAnimationFrame(fmTick);
  }
  function fmStop(msg) { const F = S.fm; F.running = false; fmSyncButtons(); if (msg) fmNote(msg); }
  function fmFinish() {
    const F = S.fm, last = F.hist[F.hist.length - 1];
    fmStop(`Finished ${F.epochs} epochs without a label. Held-out loss ${last.heldLoss.toFixed(2)}, partner found first for ${pct(last.pairAcc)} of the held-out views. A single layer on the frozen code, with 20 labelled cases per question, now scores ${FM.tasks.map(id => `${pct(last[id])} on “${F.dsets[id].task.title}”`).join(', ')}.`);
  }
  function fmTick(now) {
    const F = S.fm; if (!F.running) return;
    const dt = Math.min(0.1, (now - F.lastTime) / 1000); F.lastTime = now;
    const bpe = fmBatchesPerEpoch();
    F.debt += dt * F.speed * bpe;
    const t0 = performance.now(); let did = false, ended = false;
    while (F.debt >= 1 && performance.now() - t0 < 16) {
      ended = fmStep() || ended; F.debt -= 1; did = true;
      if (F.ptr === 0 && F.epoch >= F.epochs) { fmFinish(); break; }
    }
    if (F.debt > bpe) F.debt = bpe;
    if (did && (ended || now - F.lastRender >= 80)) { fmRender(ended); F.lastRender = now; } // the live panels redraw about 12 times a second; every epoch end redraws everything
    if (F.running) requestAnimationFrame(fmTick);
  }
  function fmStepBatch() { const F = S.fm; fmStop(); const ended = fmStep(); fmRender(true); fmNote(ended ? `Epoch ${F.epoch} complete.` : `One gradient step on ${F.lastBatch.views.length} nuclei, two views each: batch ${F.lastBatch.index} of ${fmBatchesPerEpoch()}.`); }
  function fmStepEpoch() { const F = S.fm; fmStop(); do { fmStep(); } while (F.ptr !== 0); fmRender(true); fmNote(`Epoch ${F.epoch} complete.`); }
  function fmLabsChanged() {
    const F = S.fm; if (!F.set) return;
    for (const e of F.set) { const untouched = e.views === e.fixed; e.fixed = fmFixedViews(e); if (untouched) e.views = e.fixed; }
    F.defaultBatch = F.set.slice(0, FM.batch).map(e => ({ e, v1: e.views[0], v2: e.views[1] }));
    fmBuildHeld(); F.codes = null; F.batchCache = null;
    if (F.cl) { fmRender(true); fmNote(F.labs ? 'The other lab’s scans now count as views of the same nucleus: from the next batch on, the encoder is asked to see past the stain too. The held-out pairs follow.' : 'Only flips and rotations count as views now: the encoder is no longer asked to see past the stain. The held-out pairs follow.'); }
  }
  function fmSelect(i) { const F = S.fm; F.selected = i == null || i === F.selected ? null : i; fmRenderLineup(); fmRenderGraph(); fmRenderViews(); fmRenderTray(); fmRenderMap(); fmRenderSelected(); }
  function fmFocus() { const F = S.fm; return F.selected != null ? F.set[F.selected] : F.lastBatch ? F.lastBatch.views[0].e : F.set[0]; } // the nucleus in the diagram
  function fmSyncButtons() { const F = S.fm; $('fm-train').textContent = F.running ? '⏸ Pause' : F.epoch > 0 ? '▶ Continue' : '▶ Pretrain'; }
  function fmSyncControls() {
    const F = S.fm;
    $('fm-labs').checked = F.labs;
    $('fm-epochs').value = F.epochs; $('fm-epochs-val').textContent = F.epochs;
    $('fm-speed').value = sliderFromSpeed(F.speed); $('fm-speed-val').textContent = `${F.speed} epochs/s`;
    $('fm-seed').value = F.seed;
    $('fm-color').value = F.color; $('fm-x').value = F.x; $('fm-y').value = F.y; $('fm-pairs').checked = F.showPairs;
    fmSyncButtons();
  }
  // ---- rendering
  // the reference batch through the current encoder: the last batch, or the first 20 nuclei before any step
  function fmBatchView() {
    const F = S.fm, views = F.lastBatch ? F.lastBatch.views : F.defaultBatch;
    if (F.batchCache && F.batchCache.steps === F.cl.steps && F.batchCache.views === views) return F.batchCache;
    const codes = [], units = [];
    for (const v of views) for (const x of [v.v1.x, v.v2.x]) { const c = F.cl.encode(x); codes.push(c); units.push(NN.Contrastive.unit(c).u); }
    const N = units.length, sim = Array.from({ length: N }, () => new Float64Array(N));
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) sim[i][j] = sim[j][i] = dot(units[i], units[j]);
    F.batchCache = { steps: F.cl.steps, views, codes, units, sim };
    return F.batchCache;
  }
  // the codes of every nucleus's fixed pair of views (for the map and the nearest neighbours), cached per encoder state
  function fmCodes() {
    const F = S.fm;
    if (F.codes && F.codes.steps === F.cl.steps) return F.codes;
    F.codes = { steps: F.cl.steps, canon: F.set.map(e => F.cl.encode(e.fixed[0].x)), alt: F.set.map(e => F.cl.encode(e.fixed[1].x)) };
    return F.codes;
  }
  // the game for one view: every other view of the batch is a candidate for "the same nucleus", and the encoder's vote
  // is a softmax over the cosines of the codes at τ (the partner included), best first
  function fmLineup() {
    const F = S.fm, B = fmBatchView(), e = fmFocus(), k = B.views.findIndex(v => v.e === e), [v1, v2] = e.views;
    const fw = F.cl.enc.forward(v1.x);
    const code1 = k >= 0 ? B.codes[2 * k] : fw.a[1], code2 = k >= 0 ? B.codes[2 * k + 1] : F.cl.encode(v2.x);
    const u1 = NN.Contrastive.unit(code1).u, u2 = NN.Contrastive.unit(code2).u, tau = F.cl.tau, cos12 = dot(u1, u2);
    const cands = [{ px: v2.px, cos: cos12, partner: true, e }];
    const field = k >= 0 ? B.views : B.views.slice(0, -1); // a nucleus picked from outside the batch takes the last one's seat, so the line-up stays the same size
    field.forEach((v, j) => { if (v.e === e) return; cands.push({ px: v.v1.px, cos: dot(u1, B.units[2 * j]), partner: false, e: v.e }, { px: v.v2.px, cos: dot(u1, B.units[2 * j + 1]), partner: false, e: v.e }); });
    let Z = 0; for (const d of cands) { d.w = Math.exp(d.cos / tau); Z += d.w; }
    for (const d of cands) d.p = d.w / Z;
    cands.sort((x, y) => y.p - x.p);
    const share = cands.find(d => d.partner).p;
    return { e, v1, v2, fw, code1, code2, cos12, cands, share, loss: -Math.log(share), inBatch: k >= 0 };
  }
  function fmGraphModel() {
    const F = S.fm, L = fmLineup(), seen = new Set(), others = [];
    for (const d of L.cands) { if (d.partner || seen.has(d.e)) continue; seen.add(d.e); others.push({ px: d.px, cos: d.cos }); if (others.length === 3) break; } // the hardest negatives: the nearest other nuclei by code
    return { net: F.cl.enc, mode: 'pixels', x: L.v1.x, fw: L.fw, specimen: { px: L.v1.px }, size: F.size, tint: S.tint, stage: 2, inputMean: F.std.mean, hover: F.hover,
      activation: 'linear', activationLabel: NN.ACTIVATIONS.linear.label, featureNames: [], positiveName: '', negativeName: '',
      contrast: { view1Label: fmViewLabel(L.v1), view2Label: fmViewLabel(L.v2), view2Px: L.v2.px, code1: L.code1, code2: L.code2, cos12: L.cos12, others, loss: L.loss, share: L.share, inBatch: L.inBatch } };
  }
  function fmRenderLineup() {
    const F = S.fm, L = fmLineup();
    Viz.drawLineup($('fm-lineup'), { query: L.v1.px, queryLabel: fmViewLabel(L.v1), cands: L.cands.map(d => ({ px: d.px, p: d.p, partner: d.partner })), share: L.share, loss: L.loss, size: F.size, tint: S.tint });
  }
  // the same nucleus in every orientation from both labs: its codes at the start and now, and another nucleus (the
  // batch's nearest by code) for contrast
  function fmRenderViews() {
    const F = S.fm, L = fmLineup(), e = L.e, near = L.cands.find(d => !d.partner), other = near ? near.e : F.set.find(o => o !== e);
    const views = o => { const out = []; for (const lab of ['A', 'B']) for (let t = 0; t < 8; t++) out.push(fmMakeView(o, lab, t)); return out; };
    const row = (o, cl, label, ref) => {
      const vs = views(o), codes = vs.map(v => cl.encode(v.x)), units = codes.map(cd => NN.Contrastive.unit(cd).u);
      let sum = 0, n = 0;
      if (ref) { for (const p of units) for (const q of ref) { sum += dot(p, q); n++; } }
      else for (let i = 0; i < units.length; i++) for (let j = i + 1; j < units.length; j++) { sum += dot(units[i], units[j]); n++; }
      return { label, units, mean: sum / n, views: vs.map((v, i) => ({ px: v.px, code: codes[i] })) };
    };
    const r0 = row(e, F.cl0, 'at the start'), r1 = row(e, F.cl, 'now'), r2 = row(other, F.cl, 'another nucleus, now', r1.units);
    r0.note = `mean cosine ${r0.mean.toFixed(2)}`; r0.note2 = 'among its 16 codes';
    r1.note = `mean cosine ${r1.mean.toFixed(2)}`; r1.note2 = 'among its 16 codes';
    r2.note = `mean cosine ${r2.mean.toFixed(2)}`; r2.note2 = 'to the codes above';
    Viz.drawViews($('fm-views'), { rows: [r0, r1, r2], size: F.size, tint: S.tint });
  }
  function fmRenderGraph() { const cv = $('fm-canvas'), m = fmGraphModel(); cv._model = m; Viz.drawNetwork(cv, m); }
  function fmRenderBatch() {
    const F = S.fm, B = fmBatchView(), n = Math.min(8, B.views.length), thumbs = [];
    for (let k = 0; k < n; k++) thumbs.push(B.views[k].v1.px, B.views[k].v2.px);
    Viz.drawSimilarityMatrix($('fm-batch'), { thumbs, sim: B.sim, size: F.size, tint: S.tint });
  }
  function fmRenderStatus() {
    const F = S.fm, last = F.hist[F.hist.length - 1], bpe = fmBatchesPerEpoch();
    $('fm-status').innerHTML =
      `<span>encoder <b>${F.size} × ${F.size} px → ${esc(F.cl.describe())}</b></span>` +
      `<span>parameters <b>${F.cl.parameterCount().toLocaleString()}</b></span>` +
      `<span>nuclei <b>${F.set.length}</b>, no labels · views: flips &amp; rotations${F.labs ? ' + the other lab’s scans' : ''}</span>` +
      `<span>epoch <b>${F.epoch}</b> / ${F.epochs}</span>` +
      `<span>batch <b>${F.lastBatch ? F.lastBatch.index : '–'}</b> / ${bpe}</span>` +
      (F.lastBatch ? `<span>last batch loss <b>${F.lastBatch.res.loss.toFixed(2)}</b> · partner found <b>${pct(F.lastBatch.res.pairAcc)}</b></span>` : '') +
      (last ? `<span>held-out loss <b>${last.heldLoss.toFixed(2)}</b> · partner found <b>${pct(last.pairAcc)}</b></span>` : '');
  }
  function fmRenderCurves() {
    const F = S.fm, c = Viz.colors(), testCol = getComputedStyle(document.documentElement).getPropertyValue('--test-series').trim();
    Viz.drawSeries($('fm-loss'), { hist: F.hist, keys: [{ key: 'loss', color: c.accent }, { key: 'heldLoss', color: testCol, dash: true }], maxEpoch: F.epochs });
    Viz.drawSeries($('fm-worth'), { hist: F.hist, keys: FM.tasks.map(id => ({ key: id, color: FM.colors[id] })), maxEpoch: F.epochs, pct: true });
    const last = F.hist[F.hist.length - 1];
    $('fm-loss-now').textContent = last ? `${last.loss != null ? `batches ${last.loss.toFixed(2)} · ` : ''}held-out ${last.heldLoss.toFixed(2)} · partner found ${pct(last.pairAcc)}` : '';
    $('fm-worth-now').textContent = last ? FM.tasks.map(id => pct(last[id])).join(' / ') : '';
  }
  function fmRenderMap() {
    const F = S.fm, C = fmCodes(), c = Viz.colors(), pts = [], segs = [];
    F.set.forEach((e, i) => {
      const a = C.canon[i], b = C.alt[i], task = F.dsets[e.task].task;
      const color = F.color === 'question' ? FM.colors[e.task] : F.color === 'none' ? c.ink3 : null, cls = F.color === 'label' ? classOf(e.s.label) : '';
      const name = `${task.title} · ${e.s.name}`, clsName = F.color === 'label' ? task.classes[e.s.label].name : F.color === 'question' ? `from “${task.title}”` : '';
      pts.push({ id: i, x: a[F.x], y: a[F.y], cls, color, name, clsName, selected: F.selected === i });
      if (F.showPairs) { pts.push({ id: i, x: b[F.x], y: b[F.y], cls, color, name: `${name} · second view (${fmViewLabelLong(e.fixed[1])})`, clsName, small: true }); segs.push({ x1: a[F.x], y1: a[F.y], x2: b[F.x], y2: b[F.y] }); }
    });
    Viz.drawScatter($('fm-map'), { points: pts, segments: segs, xLabel: `code number ${F.x + 1}`, yLabel: `code number ${F.y + 1}`, onSelect: id => fmSelect(id) });
  }
  function fmRenderTray() {
    const F = S.fm;
    F.set.forEach((e, i) => {
      const el = F.thumbs[i], task = F.dsets[e.task].task;
      el.className = 'thumb' + (F.lastBatch && F.lastBatch.ids.has(i) ? ' in-batch' : '') + (F.selected === i ? ' selected' : '') + (F.color === 'label' ? ` truth-${e.s.label}` : F.color === 'question' ? ` task-${e.task} show-q` : '');
      el.title = `${task.title} · ${e.s.name}${F.color === 'label' ? ` · ${task.classes[e.s.label].name}` : ''}`;
    });
  }
  function fmRenderSelected() {
    const F = S.fm, e = fmFocus(), i = e.i, C = fmCodes(), task = F.dsets[e.task].task, [v1, v2] = e.views;
    const code1 = F.cl.encode(v1.x), code2 = F.cl.encode(v2.x), cos = NN.Contrastive.cosine(code1, code2);
    const u = NN.Contrastive.unit(C.canon[i]).u; let best = -1, bestS = -Infinity;
    F.set.forEach((o, j) => { if (j === i) return; const s = dot(u, NN.Contrastive.unit(C.canon[j]).u); if (s > bestS) { bestS = s; best = j; } });
    const bars = code => { let mx = 1e-9; for (const v of code) mx = Math.max(mx, Math.abs(v)); return `<span class="code" title="the code: ${Array.from(code).map(v => v.toFixed(2)).join(', ')}">${Array.from(code).map(v => `<i class="${v < 0 ? 'neg' : 'pos'}" style="height:${Math.max(2, Math.round(Math.abs(v) / mx * 100))}%"></i>`).join('')}</span>`; };
    const view = (v, label, cls) => `<div class="v${cls}"><canvas width="${F.size}" height="${F.size}"></canvas><div class="lbl">${label}</div><div class="n">${esc(fmViewLabelLong(v))}</div></div>`;
    const nb = F.set[best], nbTask = F.dsets[nb.task].task;
    $('fm-selected').innerHTML =
      `<div class="who"><b>${esc(task.title)} · ${esc(e.s.name)}</b>${F.selected == null ? ' · the first nucleus of the last batch' : ''}${F.color === 'label' ? ` · ${esc(task.classes[e.s.label].name)} (a label the model never saw)` : ''}</div>` +
      view(v1, 'view 1', ' current') + `<div class="v"><div class="lbl">its code</div>${bars(code1)}</div>` +
      view(v2, 'view 2', '') + `<div class="v"><div class="lbl">its code</div>${bars(code2)}</div>` +
      `<div class="txt"><div>cosine of the two codes <b>${cos.toFixed(2)}</b></div>` +
      `<div class="nb">nearest other nucleus by code <canvas width="${F.size}" height="${F.size}" title="${esc(nbTask.title)} · ${esc(nb.s.name)}"></canvas> <b>${bestS.toFixed(2)}</b> · ${esc(nbTask.title)} · ${esc(nb.s.name)}${F.color === 'label' ? ` · ${esc(nbTask.classes[nb.s.label].name)}` : ''}</div>` +
      (F.selected != null ? '<div><button type="button" class="btn" id="fm-unselect">back to the batch</button></div>' : '') + '</div>';
    const cvs = $('fm-selected').querySelectorAll('canvas');
    Viz.renderThumb(cvs[0], v1.px, F.size, S.tint); Viz.renderThumb(cvs[1], v2.px, F.size, S.tint); Viz.renderThumb(cvs[2], nb.s.variants.A.px, F.size, S.tint);
  }
  function fmRender(full) {
    const F = S.fm; if (!F.cl) return;
    fmRenderStatus(); fmRenderLineup(); fmRenderGraph(); fmRenderBatch();
    if (full) { fmRenderViews(); fmRenderCurves(); fmRenderMap(); fmRenderTray(); fmRenderSelected(); }
    fmSyncButtons();
  }
  function fmEnter() {
    const F = S.fm;
    if (!F.set) fmBuild();
    if (!F.cl) fmReset('A fresh random encoder. Step a batch to watch one gradient step on 20 nuclei, or press Pretrain and watch the code become worth something.');
    else fmRender(true);
  }
  function bindFoundation() {
    $('fm-labs').addEventListener('change', () => { S.fm.labs = $('fm-labs').checked; fmLabsChanged(); });
    $('fm-epochs').addEventListener('input', () => { S.fm.epochs = +$('fm-epochs').value; $('fm-epochs-val').textContent = S.fm.epochs; if (S.fm.cl) fmRenderCurves(); });
    $('fm-speed').addEventListener('input', () => { S.fm.speed = speedFromSlider(+$('fm-speed').value); $('fm-speed-val').textContent = `${S.fm.speed} epochs/s`; });
    $('fm-seed').addEventListener('change', () => { S.fm.seed = Math.max(1, Math.round(+$('fm-seed').value) || 1); fmReset(`Encoder re-initialised from seed ${S.fm.seed}.`); });
    $('fm-step-batch').addEventListener('click', fmStepBatch);
    $('fm-step-epoch').addEventListener('click', fmStepEpoch);
    $('fm-train').addEventListener('click', () => (S.fm.running ? fmStop('Paused.') : fmStart()));
    $('fm-reset').addEventListener('click', () => fmReset('Encoder re-initialised from the seed.'));
    $('fm-color').addEventListener('change', () => { S.fm.color = $('fm-color').value; if (S.fm.cl) { fmRenderTray(); fmRenderMap(); fmRenderSelected(); } });
    for (const id of ['fm-x', 'fm-y']) $(id).addEventListener('change', () => { S.fm.x = +$('fm-x').value; S.fm.y = +$('fm-y').value; if (S.fm.cl) fmRenderMap(); });
    $('fm-pairs').addEventListener('change', () => { S.fm.showPairs = $('fm-pairs').checked; if (S.fm.cl) fmRenderMap(); });
    $('fm-selected').addEventListener('click', ev => { if (ev.target.closest('#fm-unselect')) fmSelect(null); });
    const cv = $('fm-canvas'), tip = $('fm-tip');
    cv.addEventListener('mousemove', ev => {
      const m = cv._model; if (!m) return;
      const r = cv.getBoundingClientRect();
      const hit = Viz.hitNetwork(cv, ev.clientX - r.left, ev.clientY - r.top, m);
      const prev = S.fm.hover;
      S.fm.hover = hit;
      if (hit) { tip.hidden = false; tip.textContent = hit.text; tip.style.left = (ev.clientX - r.left) + 'px'; tip.style.top = (ev.clientY - r.top) + 'px'; }
      else tip.hidden = true;
      if ((prev && prev.ref) !== (hit && hit.ref) || (hit && hit.ref && (hit.ref.kind === 'fmap' || hit.ref.kind === 'pooled' || hit.ref.kind === 'uprod' || hit.ref.kind === 'square'))) fmRenderGraph();
    });
    cv.addEventListener('mouseleave', () => { S.fm.hover = null; tip.hidden = true; fmRenderGraph(); });
  }

  // ------------------------------------------------------------------ boot
  function init() {
    try { const t = localStorage.getItem('nucleus-net-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
    Viz.refreshTheme();
    S.tasks = window.LECTURE_TASKS;
    $('question-select').innerHTML = Object.values(S.tasks).sort((a, b) => a.meta.task.order - b.meta.task.order).map((t, i) => `<option value="${t.meta.task.id}">${i + 1} · ${esc(t.meta.task.title)}</option>`).join('');
    $('recipe-select').innerHTML = '<option value="">choose a step…</option>' + RECIPE_LABELS.map((l, i) => (i ? `<option value="${i}">${esc(l)}</option>` : '')).join('');
    bindControls(); bindFoundation(); renderBackboneOptions();
    loadTask(S.taskId);
    syncControls();
    resetModel();
    renderDataTrays(); renderScatter(); renderInspector();
    onThemeChange();
    const hash = (location.hash || '').replace('#', '');
    showStage(['data', 'train', 'test', 'foundation'].includes(hash) ? hash : 'data');
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', onThemeChange);
  }
  document.addEventListener('DOMContentLoaded', init);
})();
