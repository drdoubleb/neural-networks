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
    tasks: null, taskId: 'leukemia', task: null, world: 'classic', questionId: 'leukemia', // world: which kind of question is selected (classic = a blood count or a nucleus question) ds: null, kind: 'tabular', size: 0, featureDefs: [],
    mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 60, speed: 6, seed: 1, augment: false, l2: 0, peek: false,
    trainLab: 'ours', testLab: 'ours', normalize: 'off', showLab: false, // where each set's cases come from (our lab, the other lab, both), and whether the pixels are stain-normalised first
    labelNoise: 0, showFlipped: false, // the share of training cases given the wrong label, and whether they are marked
    labelled: 0, backbone: 'page', shipped: {}, // how many training cases carry a label (0 = all), and which foundation encoder feeds the code input ('page', or a shipped backbone's id)
    foundationShown: false, // the code input and the labelled-cases control of the nucleus questions appear once the foundation question, a slides question or recipe ⑪/⑫ introduces them, and stay for the session
    // the Slides questions: attention over slides of nuclei with one label each
    sl: { built: false, model: null, question: 'atypia', context: false, hoverAtt: null, trial: { next: 0, results: new Map() }, dataSelected: null, testSelected: null, trayFor: {}, attention: true, units: 4, lr: 0.02, epochs: 60, speed: 2, seed: 1, epoch: 0, ptr: 0, order: [], hist: [], running: false, debt: 0, lastTime: 0, lastRender: 0, lastSlide: null, selected: null, hoverNucleus: null, hoverScorer: null, hoverHead: null, hoverUnrolled: null, walk: null, walkNucleus: null, reveal: false, encKey: null, pinned: null, attView: 'both', linksAll: false, linksMin: 0.1, hoverDecide: null, dwalk: null },
    // the Fields question: is it invasive? two attention heads over the nuclei of a field, with positions and context
    rp: { world: 'reports', built: false, model: null, dim: 48, positions: true, epochs: 10, lr: 0.005, seed: 1, epoch: 0, ptr: 0, order: [], hist: [], running: false, stopAt: null, lastRender: 0, lastBatch: [], trainLoss: null, selected: null, hover: null, pinned: null, layer: 1, head: -1, reveal: false, dataSelected: null, trayFor: {}, cases: 'test', trial: { next: 0, results: new Map() }, testSelected: null, temperature: 0, pace: 8, form: null, formCase: null, noblock: false, writing: null, written: null, truthShown: false, batchWriting: false, measuring: null, shipped: false, fw: null, fwFor: null, arcs: true, hoverNet: null, hoverAtt: null, pcaAxes: null, wordSec: null, netView: 'graph', netWalk: null, hops: true, hoverHeads: null, map: { mode: 'tsne', source: 'out', Y: null, iter: 0, todo: 0, running: false, forModel: null } },
    ch: { world: 'chat', built: false, model: null, dim: 48, positions: true, epochs: 10, lr: 0.005, seed: 1, epoch: 0, ptr: 0, order: [], hist: [], running: false, stopAt: null, lastRender: 0, lastBatch: [], trainLoss: null, selected: null, hover: null, pinned: null, layer: 1, head: -1, reveal: false, dataSelected: null, trayFor: {}, cases: 'test', trial: { next: 0, results: new Map() }, testSelected: null, temperature: 0, pace: 8, form: null, formCase: null, noblock: false, writing: null, written: null, truthShown: false, batchWriting: false, measuring: null, shipped: false, fw: null, fwFor: null, arcs: true, hoverNet: null, hoverAtt: null, pcaAxes: null, wordSec: null, netView: 'graph', netWalk: null, hops: true, hoverHeads: null, map: { mode: 'tsne', source: 'out', Y: null, iter: 0, todo: 0, running: false, forModel: null } },
    fd: { built: false, model: null, pos: true, ctx: true, crop: 'nucleus', units: 4, lr: 0.02, decay: 0.001, clip: 20, epochs: 60, speed: 8, seed: 1, epoch: 0, ptr: 0, order: [], hist: [], running: false, debt: 0, lastTime: 0, lastRender: 0, lastField: null, selected: null, hoverNucleus: null, pinned: null, reveal: false, linksAll: false, linksMin: 0.1, layer: 0, trial: { next: 0, results: new Map() }, dataSelected: null, dataHover: null, testSelected: null, trayFor: {}, encKey: null, progress: '', head: 1, attView: 'both', frozen: null, hoverAtt: null, hoverDecide: null, hoverUnrolled: null, hoverScorer: null, hoverHead: null, walk: null, dwalk: null, walkNucleus: null },
    animSpeed: 1, // playback speed of the walk-throughs (the lesson and Classify next): 1 = the normal pace
    excluded: new Set(),
    inputs: null, inputCache: new Map(), net: null,
    epoch: 0, ptr: 0, order: [], history: [], running: false, debt: 0, lastTime: 0, lastBatch: new Set(),
    trainEval: null, testEval: null, profileAt: 0, prevW: null, applyingRecipe: false,
    selected: null, view: 'image', tint: true, revealTest: false, stage: 'data', trayColor: 'call',
    test: { results: new Map(), next: 0, threshold: 0.5, animating: false, revealed: new Set(), prevalence: 0.01, scores: {} },
    hover: { train: null, test: null },
    lesson: null, lastLesson: null, lessonCursor: 0,
    // the Foundation question: the miniature foundation model's pretraining (its set, encoder and run live here)
    fm: { set: null, cl: null, labs: true, epochs: 100, speed: 4, seed: 1, epoch: 0, ptr: 0, order: [], hist: [], running: false, debt: 0, lastTime: 0, lastBatch: null, selected: null, color: 'none', x: 0, y: 1, showPairs: true, hover: null },
  };
  const thumbs = { data: new Map(), train: new Map(), test: new Map() }; // id -> element

  const RECIPE_LABELS = ['', '① Leukemia · blood count · single layer', '② Leukemia · blood count · 3 ReLU units', '③ Atypia · measurements · single layer', '④ Atypia · measurements · 8 + 8 ReLU · a quarter of the training labels wrong: overfitting', '⑤ Enlargement · pixels · single layer', '⑥ Irregularity · pixels · single layer', '⑦ Irregularity · pixels · 4 ReLU + augmentation', '⑧ Irregularity · pixels · 4 + 4 ReLU + augmentation · then try the other lab', '⑨ Irregularity · pixels · 4 + 4 ReLU · the shortcut: irregular nuclei scanned at another lab', '⑩ Irregularity · pixels · convolution + 4 ReLU + augmentation', '⑪ Foundation · pretrain a code on 100 unlabelled nuclei, then see what it is worth', '⑫ Irregularity · the foundation code · single layer · 10 labelled cases', '⑬ Slides · one label for 20 nuclei · attention finds the atypical ones', '⑭ Focus · four atypical cells together or scattered · the nuclei look at each other', '⑮ Fields · is it invasive? · cytology, location and arrangement · two heads, one per question', '⑯ Reports · a small language model writes the report from the findings · fluency first, grounding last', '⑰ Chat · the next-word model on transcripts · ask it about a case'];
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
  // the questions that are not a dataset of their own: the foundation model, and the slides
  const WORLD_QUESTIONS = [
    { id: 'foundation', title: 'Foundation: a code from 100 unlabelled nuclei', world: 'foundation' },
    { id: 'slides-atypia', title: 'Slides: atypical cells?', world: 'slides', q: 'atypia' },
    { id: 'slides-focus', title: 'Slides: a focus of atypical cells?', world: 'slides', q: 'focus' },
    { id: 'fields', title: 'Fields: is it invasive?', world: 'fields' },
    { id: 'reports', title: 'Reports: write the report from the findings?', world: 'reports' },
    { id: 'chat', title: 'Chat: ask the model about a case?', world: 'chat' },
  ];
  const onClassic = step => S.world === 'classic' && S.stage === step;
  function selectQuestion(id) {
    const wq = WORLD_QUESTIONS.find(q => q.id === id);
    if (!wq) {
      const switched = S.taskId !== id;
      S.world = 'classic'; S.questionId = id;
      if (switched) {
        loadTask(id);
        if (S.kind === 'image' && S.mode === 'features' && S.lr < 0.05) S.lr = 0.1;
        syncControls();
        resetModel(`Question changed to “${S.task.title}” — new ${noun(2)}, fresh random weights.`);
        renderDataTrays(); renderScatter(); renderInspector();
      }
    } else {
      S.world = wq.world; S.questionId = id;
      if (wq.world === 'slides') S.sl.question = wq.q;
      revealFoundation(); // both use the foundation encoder's code
    }
    $('question-select').value = id;
    showStage(S.stage);
  }
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
  // ---- the foundation encoder behind the code input: the one shipped with the page, or the one pretrained in the Foundation question
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
    const opts = backbones().map(b => `<option value="${esc(b.id)}">shipped: ${esc(b.nuclei.toLocaleString())} nuclei, ${esc(b.conv.K)} filters, code of ${esc(b.code)}${b.note ? ` (${esc(b.note)})` : ''}</option>`).join('') + '<option value="page">pretrained in the Foundation question</option>';
    $('backbone').innerHTML = opts; $('sl-backbone').innerHTML = opts; $('fd-backbone').innerHTML = opts;
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
    renderDataTrays(); renderScatter(); renderInspector(); if (onClassic('train')) renderTraining(true); if (onClassic('test')) { renderTestPanel(); renderTestGraph(); }
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
  function activePlayer() { return (onClassic('train') ? (S.lesson && S.lesson.player) : onClassic('test') ? (S.test.animating && S.test.player) : null) || null; }

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
    if (onClassic('train')) { renderTrainTray(); renderTrainGraph(); renderCharts(); renderScorecard(); renderProfile(fresh === true && !S.running); }
    if (onClassic('test')) { renderTestPanel(); renderTestGraph(); }
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
    if (onClassic('train')) { renderTrainTray(); renderTrainGraph(); }
    if (onClassic('test')) { renderTestTray(); if (!(opts && opts.silent)) renderTestGraph(); }
    if (onClassic('data')) renderScatter();
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
    const untrainedHere = onClassic('data') && S.net.steps === 0;
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
    $('mode-code').hidden = !S.foundationShown; $('labelled-wrap').hidden = !S.foundationShown;
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
    $('question-select').addEventListener('change', () => selectQuestion($('question-select').value));
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
    $('epochs').addEventListener('input', () => { S.epochs = +$('epochs').value; $('epochs-val').textContent = S.epochs; renderStatus(); if (onClassic('train')) renderCharts(); });
    $('speed').addEventListener('input', () => { S.speed = speedFromSlider(+$('speed').value); $('speed-val').textContent = `${S.speed} epochs/s`; });
    $('seed').addEventListener('change', () => { S.seed = Math.max(1, Math.floor(+$('seed').value || 1)); syncControls(); resetModel(`Seed ${S.seed} — fresh random weights.`); });
    $('seed-random').addEventListener('click', () => { S.seed = 1 + Math.floor(Math.random() * 9999); syncControls(); resetModel(`Seed ${S.seed} — fresh random weights.`); });
    $('augment').addEventListener('change', () => { S.augment = $('augment').checked; resetModel(S.augment ? 'Training set augmented with flips and rotations (80 × 8 = 640 views) — fresh random weights.' : 'Augmentation off — fresh random weights.'); });
    $('l2').addEventListener('input', () => { S.l2 = +$('l2').value; $('l2-val').textContent = S.l2 === 0 ? 'off' : S.l2.toFixed(2); });
    $('peek').addEventListener('change', () => { S.peek = $('peek').checked; renderStatus(); if (onClassic('train')) renderCharts(); });
    $('labelled').addEventListener('change', () => { S.labelled = +$('labelled').value; resetModel(S.labelled ? `${S.labelled} labelled training cases: the other ${S.ds.train.length - S.labelled} stay in the tray without a label, and the network never sees them — fresh random weights.` : 'Every training case labelled again — fresh random weights.'); renderDataTrays(); });
    $('backbone').addEventListener('change', () => { S.backbone = $('backbone').value; resetModel(`The code now comes from ${codeEncoder().describe} — fresh random weights.`); });
    for (const id of ['source-train', 'source-train-2']) $(id).addEventListener('change', ev => setTrainSource(ev.target.value));
    for (const id of ['source-test', 'source-test-2']) $(id).addEventListener('change', ev => setTestSource(ev.target.value));
    $('normalize').addEventListener('change', () => { S.normalize = $('normalize').value; syncControls(); resetModel(S.normalize === 'off' ? 'Stain normalisation off — fresh random weights.' : `Stain normalisation ${S.normalize === 'lab' ? 'per lab' : 'per image'} — fresh random weights.`); });
    $('show-lab').addEventListener('change', () => { S.showLab = $('show-lab').checked; renderDataTrays(); if (onClassic('train')) renderTrainTray(); if (onClassic('test')) renderTestTray(); });
    $('noise').addEventListener('input', () => { const v = +$('noise').value; $('noise-val').textContent = v ? `${v}% of the labels wrong` : 'off'; });
    $('noise').addEventListener('change', () => {
      S.labelNoise = +$('noise').value / 100; syncControls();
      resetModel(S.labelNoise ? `${Math.round(S.labelNoise * 100)}% of the training labels flipped (${S.inputs.flipped.size} of ${S.ds.train.length}), as a second pathologist might have called them — fresh random weights.` : 'Every training label right again — fresh random weights.');
      renderDataTrays(); renderScatter(); renderInspector();
    });
    for (const id of ['show-flipped', 'show-flipped-2']) $(id).addEventListener('change', ev => { S.showFlipped = ev.target.checked; syncControls(); renderDataTrays(); renderScatter(); renderInspector(); if (onClassic('train')) renderTrainTray(); });
    $('btn-train').addEventListener('click', () => (S.running ? stopTraining('Paused.') : startTraining()));
    $('btn-step-batch').addEventListener('click', stepBatch);
    $('btn-step-epoch').addEventListener('click', stepEpoch);
    $('btn-reset').addEventListener('click', () => resetModel('Weights re-initialised from the seed.'));
    $('btn-teach').addEventListener('click', teachNext);
    $('recipe-select').addEventListener('change', () => {
      const k = $('recipe-select').value; if (!k) return;
      if (k === '17') { // the chat question: the same model on transcripts
        S.ch.dim = 48; S.ch.positions = true; S.ch.epochs = 10; S.ch.lr = 0.005; S.ch.seed = 1; S.ch.reveal = false;
        selectQuestion('chat'); showStage('train');
        rpReset(`Recipe ${RECIPE_LABELS[17]}: 2,400 transcripts, the next word alone. Press Train and watch the loss by role: the system line and the findings cost almost nothing within an epoch or two, the question stays high since nothing predicts which one comes, and the assistant line, the only one that matters, falls last; the third chart asks the model for the answer of 8 test transcripts after every epoch. Or load the trained model and go to 3 · Test to ask it yourself, in the trained phrasings and in your own.`);
        return;
      }
      if (k === '16') { // the reports question: the small language model
        S.rp.dim = 48; S.rp.positions = true; S.rp.epochs = 10; S.rp.lr = 0.005; S.rp.seed = 1; S.rp.reveal = false;
        selectQuestion('reports'); showStage('train');
        rpReset(`Recipe ${RECIPE_LABELS[16]}: 800 reports, no rule ever given, only the next word. Press Train and watch the loss by section: the gross falls first, the microscopic description next, and the diagnosis reads as certain from the fifth epoch while the third chart, the diagnosis written from the findings alone, is still climbing. Load the trained model when you have seen enough, then go to 3 · Test and flip a finding.`); return;
      }
      if (k === '15') { // the fields question: the full model
        S.fd.pos = true; S.fd.ctx = true; S.fd.crop = 'nucleus'; S.fd.epochs = 60; S.fd.speed = 8; S.fd.seed = 1; S.fd.reveal = false;
        selectQuestion('fields'); showStage('train');
        fdReset(`Recipe ${RECIPE_LABELS[15]}: 400 fields of bladder, five patterns, two labels each and no nucleus ever labelled. Press Train and watch the mimic table: cytology alone gets the first three rows, positions add the membrane, and the context is what tells a round nest from an angulated one. Then switch a cue off and train again.`); return;
      }
      if (k === '14') { // the focus question, with the context layer
        S.sl.context = true; S.sl.attention = true; S.sl.units = 4; S.sl.epochs = 150; S.sl.speed = 2; S.sl.seed = 1; S.sl.reveal = false;
        selectQuestion('slides-focus'); showStage('train');
        slReset(`Recipe ${RECIPE_LABELS[14]}: every slide holds four atypical nuclei, a 2 × 2 block or scattered, so the bag of nuclei is the same in both classes and the model of recipe ⑬ stays at chance. Context is on: one layer of self-attention lets each nucleus read its neighbours before the scorer sees it. Press Train, then hover a nucleus to see whom it listens to; switch context off and train again to compare.`); return;
      }
      if (k === '13') { // the slides question, without context
        S.sl.context = false; S.sl.attention = true; S.sl.units = 4; S.sl.epochs = 60; S.sl.speed = 2; S.sl.seed = 1; S.sl.reveal = false;
        selectQuestion('slides-atypia'); showStage('train');
        slReset(`Recipe ${RECIPE_LABELS[13]}: 60 slides of 20 nuclei, one label each, no nucleus ever labelled. Press Train and watch the right-hand curve: the share of attention landing on the atypical nuclei climbs from the uniform ${pct(S.sl.share)} towards 90%. Then tick “Reveal” on any slide, and try the plain average for comparison.`);
        return;
      }
      if (k === '11') { // the foundation question
        S.fm.labs = true; S.fm.epochs = 100; S.fm.speed = 4;
        selectQuestion('foundation'); showStage('train');
        fmReset(`Recipe ${RECIPE_LABELS[11]}: 100 nuclei from all three questions, no labels. Press Pretrain and watch the right-hand curve: a single layer on the code gets better at every question, though the code was never told what any of them asks.`);
        return;
      }
      const { task: taskId, ...settings } = RECIPES[k];
      const switchTask = taskId !== S.taskId;
      if (switchTask) loadTask(taskId);
      S.world = 'classic'; S.questionId = taskId; $('question-select').value = taskId;
      if (settings.mode === 'code') { revealFoundation(); S.backbone = S.fm.cl && S.fm.epoch > 0 ? 'page' : defaultBackbone(); } // the encoder you pretrained, if you did; else the shipped copy of it
      Object.assign(S, LAB_SETTINGS, settings);
      applySources();
      S.excluded = new Set();
      S.applyingRecipe = true;
      syncControls();
      resetModel(`Recipe ${RECIPE_LABELS[k]}${switchTask ? ` — question switched to “${S.task.title}”` : ''}. Press Train.`);
      S.applyingRecipe = false;
      $('recipe-select').value = k;
      renderDataTrays(); renderScatter(); renderInspector();
      showStage('train');
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
    $('tint').addEventListener('change', () => { S.tint = $('tint').checked; repaintThumbs(); fmRepaint(); slRepaint(); renderInspector(); if (onClassic('train')) renderTrainGraph(); if (onClassic('test')) renderTestGraph(); });
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
      if (ev.key === ' ' && S.stage === 'train') { ev.preventDefault(); if (S.world === 'classic') S.running ? stopTraining('Paused.') : startTraining(); else if (S.world === 'foundation') S.fm.running ? fmStop('Paused.') : fmStart(); else if (S.world === 'fields') S.fd.running ? fdStop('Paused.') : fdStart(); else if (S.world === 'reports' || S.world === 'chat') { const Lx = LM(); if (Lx.running && Lx.stopAt == null) rpStop('Paused.'); else rpStart(); } else S.sl.running ? slStop('Paused.') : slStart(); }
      else if ((ev.key === 'n' || ev.key === 'N') && S.stage === 'test') { if (S.world === 'classic') classifyNext(false); else if (S.world === 'slides') slClassifyNext(); else if (S.world === 'fields') fdClassifyNext(); else if (S.world === 'reports') rpWriteNext(); else if (S.world === 'chat') chAskNext(); }
      else if ((ev.key === 'n' || ev.key === 'N' || ev.key === 't' || ev.key === 'T') && onClassic('train')) teachNext();
      else if (ev.key === '1') showStage('data'); else if (ev.key === '2') showStage('train'); else if (ev.key === '3') showStage('test');
    });
    window.addEventListener('resize', () => { if (onClassic('train')) renderTrainGraph(); if (onClassic('test')) renderTestGraph(); if (S.world === 'foundation' && S.fm.cl) { fmRenderLineup(); fmRenderGraph(); fmRenderViews(); fmRenderBatch(); } if (S.world === 'slides') slRenderAll(); if (S.world === 'fields') fdRenderAll(); });
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
    if (onClassic('train')) { renderTrainGraph(); renderProfile(true); } if (onClassic('test')) renderTestGraph();
    fmRepaint(); slRepaint(); fdRepaint();
    renderInspector();
  }
  // the lecture introduces the foundation model late, so its stage and controls stay out of the way until then
  function revealFoundation() {
    if (!S.foundationShown) { S.foundationShown = true; try { sessionStorage.setItem('nucleus-net-foundation', '1'); } catch (e) { /* ignore */ } }
    applyVisibility();
  }
  const PANELS = { 'classic:data': 'panel-data', 'classic:train': 'panel-train', 'classic:test': 'panel-test', 'foundation:data': 'panel-fm-data', 'foundation:train': 'panel-foundation', 'foundation:test': 'panel-fm-test', 'slides:data': 'panel-sl-data', 'slides:train': 'panel-slides', 'slides:test': 'panel-sl-test', 'fields:data': 'panel-fd-data', 'fields:train': 'panel-fields', 'fields:test': 'panel-fd-test', 'reports:data': 'panel-rp-data', 'reports:train': 'panel-reports', 'reports:test': 'panel-rp-test', 'chat:data': 'panel-ch-data', 'chat:train': 'panel-reports', 'chat:test': 'panel-ch-test' };
  function showStage(name) {
    if (!['data', 'train', 'test'].includes(name)) name = 'data';
    S.stage = name;
    const w = S.world, show = PANELS[`${w}:${name}`];
    document.querySelectorAll('.stage').forEach(b => b.classList.toggle('is-active', b.dataset.stage === name));
    for (const id of Object.values(PANELS)) $(id).hidden = id !== show;
    const wide = w !== 'classic';
    document.querySelector('.inspector').hidden = wide; document.querySelector('.bench').classList.toggle('no-inspector', wide); // only the blood counts and the nuclei have a specimen inspector
    if (!(w === 'foundation' && name === 'train')) fmStop();
    if (!(w === 'slides' && name === 'train')) slStop();
    if (!(w === 'fields' && name === 'train')) fdStop();
    if (!((w === 'reports' || w === 'chat') && name === 'train')) lmStopAll();
    if (!(w === 'reports' && name === 'test')) { rpWriteStop(); S.rp.batchWriting = false; }
    if (!(w === 'chat' && name === 'test')) { chWriteStop(); S.ch.batchWriting = false; }
    if (w !== 'classic' || name === 'test') stopTraining();
    if (w === 'classic') {
      if (name === 'data') { renderDataTrays(); renderScatter(); renderInspector(); }
      if (name === 'train') {
        if (S.selected && S.selected.split !== 'train') S.selected = S.ds.train[0];
        if (S.mode === 'code' && S.inputs && S.inputs.encoderKey !== codeEncoder().key) resetModel('The code follows the encoder pretrained in the Foundation question, which has changed — fresh random weights on the new code.');
        renderTraining(true);
      }
      if (name === 'test') {
        renderTestPanel();
        if (!S.selected || S.selected.split !== 'test') { S.selected = S.ds.test[Math.max(0, S.test.next - 1)]; }
        renderTestGraph(); renderInspector(); renderDataTrays();
        if (S.net.conv && !$('test-note').textContent) $('test-note').textContent = 'Classify next walks through the convolution (about 50 s): the mean nucleus is subtracted, filter 1 scans the difference slowly with its arithmetic shown, the other filters follow together, map 1 is pooled block by block and the other maps follow, then the pooled maps are stacked and laid over each hidden unit’s weight map and summed, and the units feed the output. Press N or click the diagram to skip ahead; space pauses, ← → step.';
      }
      } else if (w === 'foundation') { if (name === 'data') fmEnterData(); else if (name === 'train') fmEnter(); else fmEnterTest(); }
    else if (w === 'slides') { if (name === 'data') slEnterData(); else if (name === 'train') slEnter(); else slEnterTest(); }
    else if (w === 'reports') { if (name === 'data') rpEnterData(); else if (name === 'train') rpEnter(); else rpEnterTest(); }
    else if (w === 'chat') { if (name === 'data') chEnterData(); else if (name === 'train') rpEnter(); else chEnterTest(); }
    else { if (name === 'data') fdEnterData(); else if (name === 'train') fdEnter(); else fdEnterTest(); }
    try { history.replaceState(null, '', '#' + (w === 'classic' ? name : `${S.questionId}/${name}`)); } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------------ the Foundation question: a miniature foundation model, pretrained live
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
    if (full) { fmRenderViews(); fmRenderCurves(); fmRenderMap(); fmRenderTray(); fmRenderSelected(); fmRenderWorth(); }
    fmSyncButtons();
  }
  function fmEnterData() { fmEnter(); }
  function fmEnterTest() { fmEnter(); fmRenderWorth(); }
  // what the code is worth: the probe accuracies per question, before pretraining and now
  function fmRenderWorth() {
    const F = S.fm; if (!F.hist || !F.hist.length) return;
    const first = F.hist[0], last = F.hist[F.hist.length - 1];
    $('fm-worth-table').innerHTML = `<div class="profile-wrap"><table class="profile"><thead><tr><th class="row">question</th><th>a random code (epoch 0)</th><th>after ${F.epoch} epoch${F.epoch === 1 ? '' : 's'}</th></tr></thead><tbody>` +
      FM.tasks.map(id => `<tr><td class="row"><span class="udot" style="background:${FM.colors[id]}"></span>${esc(S.tasks[id].meta.task.title)}</td><td>${pct(first[id])}</td><td><b>${pct(last[id])}</b></td></tr>`).join('') + '</tbody></table></div>';
    $('fm-worth-note').textContent = F.epoch === 0
      ? 'Nothing has been pretrained yet: this is what a single layer gets from a random code. Go to 2 · Train, press Pretrain, and come back.'
      : `After ${F.epoch} epoch${F.epoch === 1 ? '' : 's'} of pretraining on 100 unlabelled nuclei, a single layer on the frozen code, with 20 labelled cases per question, scores this on nuclei it never saw. The code was never told what any question asks.`;
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

  // ------------------------------------------------------------------ the Slides questions: attention over a slide of nuclei
  // A slide is 20 nuclei with one label. Every nucleus becomes its code from the frozen foundation encoder (the one
  // the code input uses), an AttentionMIL scores the nuclei, a softmax over the slide turns the scores into weights,
  // the weighted average of the codes is the slide's summary and a single layer makes the call. Only the slide's label
  // trains it; the pool's truth about the nuclei is only ever used to show, afterwards, where the attention landed.
  const SLC = { lr: 0.02, batch: 1, decay: 0.001, context: { dk: 8, ffn: 8, distanceBias: true, excludeSelf: true, costInit: 1.5 } }; // decay only with the context layer
  function slNote(msg) { $(slOnTest() ? 'sl-test-note' : 'sl-note').textContent = msg || ''; } // the note of the step on screen
  function slData() { const D = window.LECTURE_SLIDES; return D && D.questions ? D : null; }
  function slBuild() { // the pool once; each question's slides on first use; the stage switched to the current question
    const L = S.sl, D = slData(); if (!D) return false;
    if (!L.pool) {
      const meta0 = D.questions.atypia.meta; L.size = meta0.size; L.cols = meta0.cols || 5; L.perSlide = meta0.perSlide;
      L.pool = D.pool.map(p => { const px = NF.decodeBase64(p.px); return { id: p.id, label: p.label, subtype: p.subtype, px, ink: NF.toInk(px) }; });
      const cellOf = k => [k % L.cols, Math.floor(k / L.cols)];
      L.dist = Array.from({ length: L.perSlide }, (_, i) => Float64Array.from({ length: L.perSlide }, (_, j) => { const [ax, ay] = cellOf(i), [bx, by] = cellOf(j); return Math.hypot(ax - bx, ay - by); })); // grid distances in cells, for the context layer
      L.sets = {};
    }
    const q = L.question;
    if (!L.sets[q]) {
      const Q = D.questions[q], mk = s => ({ id: s.id, name: s.name, split: s.split, label: s.label, y: s.label ? 1 : 0, dist: L.dist, nuclei: s.nuclei.map(([id, t]) => { const p = L.pool[id]; return { id, t, label: p.label, subtype: p.subtype, px: t ? DS.dihedral(p.px, L.size, t) : p.px, ink: t ? DS.dihedral(p.ink, L.size, t) : p.ink }; }) });
      const train = Q.train.map(mk), test = Q.test.map(mk), all = [...train, ...test], pos = train.filter(s => s.label);
      L.sets[q] = { meta: Q.meta, train, test, all, byId: new Map(all.map(s => [s.id, s])), share: pos.reduce((a, s) => a + s.nuclei.filter(n => n.label).length / s.nuclei.length, 0) / pos.length }; // share: what uniform attention gives the atypical nuclei
    }
    if (L.loaded !== q) {
      const set = L.sets[q]; L.meta = set.meta; L.train = set.train; L.test = set.test; L.all = set.all; L.byId = set.byId; L.share = set.share; L.loaded = q;
      L.selected = null; L.lastSlide = null; L.hoverNucleus = null; L.hoverAtt = null; L.hoverDecide = null; L.pinned = null; L.encKey = null; L.trayFor = {}; L.dataSelected = null; L.testSelected = null; L.trial = { next: 0, results: new Map() };
      for (const [id, tray] of [['sl-train-tray', L.train], ['sl-test-tray', L.test]]) { const el = $(id); el.innerHTML = ''; for (const s of tray) el.appendChild(slThumb(s)); }
      document.querySelectorAll('[data-slcls="0"]').forEach(el => { el.textContent = L.meta.classes[0].name; });
      document.querySelectorAll('[data-slcls="1"]').forEach(el => { el.textContent = L.meta.classes[1].name; });
      $('sl-train-label').textContent = `Training slides (${L.train.length})`; $('sl-test-label').textContent = `Test slides (${L.test.length}) · never trained on, scored as it goes`;
      L.thumbs = new Map(L.all.map(s => [s.id, document.querySelector(`.thumb.slide[data-id="${s.id}"]`)]));
      $('sl-blurb').innerHTML = `<strong>${esc(L.meta.question)}</strong> ${esc(L.meta.blurb)} ${q === 'focus'
        ? 'The model of the first question, each nucleus scored on its own, stays at chance here by construction. Switch <em>context</em> on: one layer of self-attention lets every nucleus read the others before the scorer sees it, and the model learns how far to look. Train, then hover a nucleus to see whom it listens to.'
        : 'Each nucleus becomes its <strong>code</strong> from the frozen foundation encoder; a small <strong>attention</strong> network gives every nucleus a score, a softmax over the slide turns the scores into weights that add up to one, the nuclei’s codes are averaged with those weights into one summary of the slide, and a single layer makes the call. The only teacher is the slide’s label, yet the attention learns which nuclei matter: tick <em>reveal</em> to see where it lands. Switch to a plain average and compare.'}`;
    }
    L.built = true;
    return true;
  }
  function slThumb(s, onClick) {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'thumb slide'; b.dataset.id = s.id; b.title = s.name;
    const cv = document.createElement('canvas'); b.appendChild(cv);
    Viz.renderSlideThumb(cv, s.nuclei.map(n => n.px), S.sl.cols, S.sl.size, S.tint);
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = '✗'; b.appendChild(badge);
    b.addEventListener('click', () => (onClick || slSelect)(s.id));
    return b;
  }
  function slRepaint() { const L = S.sl; if (!L.built) return; for (const map of [L.thumbs, L.dataThumbs, L.testThumbs]) if (map) for (const [id, el] of map) { const s = L.byId.get(id); if (s) Viz.renderSlideThumb(el.querySelector('canvas'), s.nuclei.map(n => n.px), L.cols, L.size, S.tint); } slRenderAll(); }
  // every nucleus of every slide through the frozen encoder, then standardised on the training slides' nuclei
  function slEncode() {
    const L = S.sl, enc = codeEncoder();
    if (L.encKey === enc.key) return;
    for (const s of L.all) for (const n of s.nuclei) if (n.codeKey !== enc.key) { n.code = enc.encode(n.ink); n.codeKey = enc.key; }
    L.std = NN.fitStandardizer([].concat(...L.train.map(s => s.nuclei.map(n => n.code))), { perDimScale: true });
    for (const s of L.all) { for (const n of s.nuclei) n.h = L.std.apply(n.code); s.H = s.nuclei.map(n => n.h); s.pos = s.nuclei.map(n => !!n.label); }
    L.encKey = enc.key; L.D = enc.cl.code; L.encDescribe = enc.describe;
  }
  function slReset(reason) {
    const L = S.sl; slStop();
    if (!slBuild()) return;
    slEncode();
    L.model = new NN.AttentionMIL({ inputSize: L.D, attentionUnits: L.units, attention: L.attention, context: L.context ? SLC.context : null, seed: L.seed });
    L.rng = NN.mulberry32(L.seed * 31 + 7); L.order = L.train.map((_, i) => i);
    L.epoch = 0; L.ptr = 0; L.debt = 0; L.hist = []; L.lastSlide = null; L.hoverNucleus = null; L.hoverAtt = null; L.hoverDecide = null;
    L.trial = { next: 0, results: new Map() }; L.testSelected = null; L.modelQuestion = L.loaded; // a new model: the test step starts over
    slRecordEpoch();
    slSyncControls(); slRenderAll();
    if (reason) slNote(reason);
  }
  function slRecordEpoch() {
    const L = S.sl; L.trainEval = L.model.evaluate(L.train); L.testEval = L.model.evaluate(L.test);
    L.hist.push({ epoch: L.epoch, loss: L.trainEval.loss, acc: L.trainEval.accuracy, testLoss: L.testEval.loss, testAcc: L.testEval.accuracy, mass: L.trainEval.culpritMass, testMass: L.testEval.culpritMass });
  }
  function slStep() { // one gradient step on the next training slide of the epoch's order
    const L = S.sl, n = L.train.length;
    if (L.ptr === 0) { const o = L.order; for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(L.rng() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } }
    const idx = L.order.slice(L.ptr, L.ptr + SLC.batch);
    L.model.trainBatch(idx.map(i => L.train[i]), L.lr, L.context ? SLC.decay : 0);
    L.lastSlide = L.train[idx[idx.length - 1]].id;
    L.ptr += idx.length;
    let ended = false;
    if (L.ptr >= n) { L.ptr = 0; L.epoch++; slRecordEpoch(); ended = true; }
    return ended;
  }
  function slStart() {
    const L = S.sl; if (L.walk) slWalkStop();
    if (L.epoch >= L.epochs) { slNote(`Already at ${L.epochs} epochs. Raise the epoch count, or reset to train again.`); return; }
    L.running = true; L.lastTime = performance.now(); L.debt = 0; L.lastRender = 0; slSyncButtons();
    requestAnimationFrame(slTick);
  }
  function slStop(msg) { const L = S.sl; L.running = false; if (L.walk) slWalkStop(); if (L.dwalk) slDecideWalkStop(); slSyncButtons(); if (msg) slNote(msg); }
  function slFinish() {
    const L = S.sl, last = L.hist[L.hist.length - 1];
    slStop(`Finished ${L.epochs} epochs on the slides' labels alone. Slide accuracy ${pct(last.acc)} on the training slides, ${pct(last.testAcc)} on the test slides${L.attention ? `; ${pct(last.testMass)} of a positive test slide's attention now lands on its atypical nuclei (uniform: ${pct(L.share)}). Tick “Reveal” and look.` : '. A plain average cannot say where it looked.'}`);
  }
  function slTick(now) {
    const L = S.sl; if (!L.running) return;
    const dt = Math.min(0.1, (now - L.lastTime) / 1000); L.lastTime = now;
    const spe = Math.ceil(L.train.length / SLC.batch);
    L.debt += dt * L.speed * spe;
    const t0 = performance.now(); let did = false, ended = false;
    while (L.debt >= 1 && performance.now() - t0 < 16) {
      ended = slStep() || ended; L.debt -= 1; did = true;
      if (L.ptr === 0 && L.epoch >= L.epochs) { slFinish(); break; }
    }
    if (L.debt > spe) L.debt = spe;
    if (did && (ended || now - L.lastRender >= 80)) { slRender(ended); L.lastRender = now; }
    if (L.running) requestAnimationFrame(slTick);
  }
  function slStepSlide() { const L = S.sl; slStop(); const ended = slStep(); slRender(true); slNote(ended ? `Epoch ${L.epoch} complete.` : `One gradient step on slide ${L.byId.get(L.lastSlide).name} (${L.meta.classes[L.byId.get(L.lastSlide).label].name}): step ${L.ptr} of ${L.train.length}.`); }
  function slStepEpoch() { const L = S.sl; slStop(); do { slStep(); } while (L.ptr !== 0); slRender(true); slNote(`Epoch ${L.epoch} complete.`); }
  function slSelect(id) { const L = S.sl; L.selected = id === L.selected ? null : id; L.hoverNucleus = null; slWalkStop(); slDecideWalkStop(); slRenderFocus(); slRenderSummary(); slRenderTrays(); }
  const slOnTest = () => S.world === 'slides' && S.stage === 'test';
  function slTestResult() { const L = S.sl; return L.testSelected != null ? L.trial.results.get(L.testSelected) : null; } // the call already made for the slide under test, if any
  function slFocus() { const L = S.sl; if (slOnTest()) return (L.testSelected != null && L.byId.get(L.testSelected)) || L.test[Math.min(L.trial.next, L.test.length - 1)]; return (L.selected != null && L.byId.get(L.selected)) || (L.lastSlide != null && L.byId.get(L.lastSlide)) || L.train[0]; } // the slide on screen
  // the diagram cards live in the Train step and move to the Test step with the slide under test (once it is classified)
  function slPlaceCards() {
    const cards = $('sl-cards'), onTest = slOnTest(), before = $(onTest ? 'sl-test-results-card' : 'sl-curves-card');
    if (cards.nextElementSibling !== before) before.parentNode.insertBefore(cards, before);
    cards.hidden = onTest && !slTestResult();
  }
  function slNeighbour(step) { const L = S.sl, cur = slFocus(), i = L.all.indexOf(cur); slSelect(L.all[(i + step + L.all.length) % L.all.length].id); }
  function slSyncButtons() { const L = S.sl; $('sl-train').textContent = L.running ? '⏸ Pause' : L.epoch > 0 ? '▶ Continue' : '▶ Train'; }
  function slSyncControls() {
    const L = S.sl;
    document.querySelectorAll('#sl-pool-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.att === '1') === L.attention));
    document.querySelectorAll('#sl-context-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.ctx === '1') === L.context));
    $('sl-context-card').hidden = !L.context; $('sl-decide-card').hidden = !L.context; $('sl-links-ctl').hidden = !L.context;
    document.querySelectorAll('#sl-attview-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.view === L.attView));
    document.querySelectorAll('#sl-links-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.links === '1') === L.linksAll));
    $('sl-links-min').value = Math.round(L.linksMin * 100); $('sl-links-min-val').textContent = pct(L.linksMin); $('sl-links-min').disabled = !L.linksAll;
    $('sl-units').value = L.units; $('sl-units').disabled = !L.attention;
    $('sl-epochs').value = L.epochs; $('sl-epochs-val').textContent = L.epochs;
    $('sl-speed').value = sliderFromSpeed(L.speed); $('sl-speed-val').textContent = `${L.speed} epochs/s`;
    $('sl-seed').value = L.seed; for (const id of ['sl-reveal', 'sl-data-reveal', 'sl-test-reveal']) $(id).checked = L.reveal;
    $('sl-backbone').value = S.backbone;
    $('sl-scorer-card').hidden = !L.attention;
    slSyncButtons();
  }
  // ---- rendering
  function slRenderStatus() {
    const L = S.sl, last = L.hist[L.hist.length - 1], m = L.model, cls = L.meta.classes;
    $('sl-status').innerHTML =
      `<span>architecture <b>${L.meta.perSlide} nuclei → code (${L.D} numbers each, from ${esc(L.encDescribe)}) → ${esc(m.describe())} → P</b></span>` +
      `<span>parameters <b>${m.parameterCount()}</b></span>` +
      `<span>slides <b>${L.train.length}</b> training (${L.train.filter(s => s.label).length} positive) · <b>${L.test.length}</b> test</span>` +
      `<span>epoch <b>${L.epoch}</b> / ${L.epochs}</span><span>slide <b>${L.ptr === 0 ? '–' : L.ptr}</b> / ${L.train.length}</span>` +
      `<span>loss <b>${last.loss.toFixed(3)}</b></span><span>slide accuracy <b>${pct(last.acc)}</b> training · <b>${pct(last.testAcc)}</b> test (peeking)</span>` +
      (L.attention ? `<span>attention on the atypical nuclei <b>${pct(last.mass)}</b> training · <b>${pct(last.testMass)}</b> test (uniform: ${pct(L.share)})</span>` : `<span>every nucleus weighs <b>${pct(1 / L.meta.perSlide)}</b> (plain average)</span>`);
    void cls;
  }
  function slFocusForward() { const L = S.sl, s = slFocus(), r = slOnTest() ? slTestResult() : null; return { s, fw: r ? r.fw : L.model.forward(s.H, s.dist) }; } // on the Test step, the call as it was made
  // the nucleus the cards follow: the hovered one, else the pinned one (a click on the slide or the map), else the one with the most attention
  function slShown(fw) { const L = S.sl; if (L.hoverNucleus != null) return L.hoverNucleus; if (L.pinned != null && L.pinned < fw.a.length) return L.pinned; let j = 0; for (let i = 1; i < fw.a.length; i++) if (fw.a[i] > fw.a[j]) j = i; return j; }
  function slShownWord() { const L = S.sl; return L.hoverNucleus != null ? '' : L.pinned != null ? ', pinned,' : ', the one with the most attention,'; }
  // the context layer's attention as the map shows it: what the layer uses, or what the match alone or the distance alone would give
  function slAttentionView(fw) { const L = S.sl, ex = L.model.context.explain(fw.ctx); return L.attView === 'match' ? ex.matchOnly : L.attView === 'distance' ? ex.distOnly : fw.ctx.A; }
  function slDrawSlide(id, s, fw) { // one viewer: the slide with its attention frames and, with context, the lines of the nucleus under the cursor; plain before a call
    const L = S.sl, n = s.nuclei.length, view = fw && L.context && fw.ctx ? slAttentionView(fw) : null, ok = i => (i != null && i >= 0 && i < n ? i : null);
    const hover = ok(L.hoverNucleus != null ? L.hoverNucleus : L.walkNucleus), pinned = ok(L.pinned), from = ok(L.hoverNucleus != null ? L.hoverNucleus : L.dwalk ? L.dwalk.i : L.pinned); // whose listening the lines show
    Viz.drawSlide($(id), { nuclei: s.nuclei.map((q, i) => ({ px: q.px, a: fw ? fw.a[i] : 0, pos: !!q.label })), cols: L.cols, size: L.size, tint: S.tint, reveal: fw ? L.reveal : false, plain: !fw, hover, pinned: view ? pinned : null,
      links: view && !L.linksAll && from != null ? { from, weights: view[from] } : null, allLinks: view && L.linksAll ? { A: view, min: L.linksMin, hover: from } : null });
    return view;
  }
  function slRenderSlide() {
    const L = S.sl;
    if (slOnTest()) { const r = slTestResult(); slDrawSlide('sl-test-canvas', slFocus(), r ? r.fw : null); return; } // the viewer of the Test step: the call as it was made, or the next slide plain
    const { s, fw } = slFocusForward(), cls = L.meta.classes, view = slDrawSlide('sl-canvas', s, fw);
    if (view) $('sl-links-note').textContent = (L.linksAll
      ? `Every link above ${pct(L.linksMin)}${L.linksMin > 0 ? ' at once' : ''}: a link is wide at the end that listens, its width that nucleus’s share for the other, and comes to a point at an end that does not listen back. Hover a nucleus to lift its links out; click it to keep it.`
      : 'Lines from the hovered nucleus to the nuclei it listens to, thicker with the share. Click a nucleus to keep it as the one the cards follow; switch to every nucleus to see all the links at once.')
      + (L.attView === 'both' ? '' : L.attView === 'match' ? ' Shown from the match alone, as if distance cost nothing.' : ' Shown from the distance alone, as if every nucleus looked the same.');
    const call = fw.p >= 0.5 ? 1 : 0, k = s.nuclei.filter(n => n.label).length;
    $('sl-slide-title').textContent = `Slide ${s.name} · ${s.split === 'train' ? 'training' : 'test'} · label: ${cls[s.label].name}${L.reveal ? ` (${k} atypical nucle${k === 1 ? 'us' : 'i'})` : ''}`;
    $('sl-call').innerHTML = `The model says <b>P(${esc(cls[1].name)}) = ${fw.p.toFixed(2)}</b> → <span class="${call === s.label ? 'good-text' : 'bad-text'}">${esc(cls[call].name)} ${call === s.label ? '✓' : '✗'}</span>${L.selected == null && L.lastSlide != null ? ' · the last slide trained on' : ''}. ${L.attention ? 'Frames and percentages are the attention weights.' : 'Plain average: every nucleus weighs the same.'}`;
  }
  function slRenderRank() {
    const L = S.sl, onTest = slOnTest();
    if (onTest && !slTestResult()) { const s0 = slFocus(); Viz.drawRanked($('sl-test-rank'), { items: s0.nuclei.map((n, i) => ({ px: n.px, value: 0, pos: false, index: i })), size: L.size, tint: S.tint, reveal: false, hover: null, title: 'not classified yet' }); return; }
    const { s, fw } = slFocusForward();
    const items = s.nuclei.map((n, i) => ({ px: n.px, value: fw.a[i], pos: !!n.label, index: i })).sort((a, b) => b.value - a.value);
    Viz.drawRanked($(onTest ? 'sl-test-rank' : 'sl-rank'), { items, size: L.size, tint: S.tint, reveal: L.reveal, hover: L.hoverNucleus, title: L.attention ? 'the 20 nuclei ranked by attention · bars: each one’s share of the slide’s attention' : 'plain average: every nucleus gets the same share' });
  }
  function slScorerModel() { // the attention network for one nucleus: the hovered one, or the one with the most attention
    const L = S.sl, { s, fw } = slFocusForward();
    const j = slShown(fw), n = s.nuclei[j], tok = fw.T[j], sf = fw.fws ? fw.fws[j] : L.model.scorer.forward(tok);
    return { j, n, m: { net: L.model.scorer, mode: 'features', inputCaption: `INPUT · THIS NUCLEUS’S CODE${L.context ? ', AFTER CONTEXT' : ''} · ${L.D} NUMBERS`, featureNames: Array.from({ length: L.D }, (_, i) => `code ${i + 1}`), x: tok, fw: sf, stage: 2, hover: L.hoverScorer,
      activation: 'tanh', activationLabel: 'Tanh', positiveName: 'more attention', negativeName: 'less attention', scoreLabel: 'attention score', scoreNote: `softmax over the slide\n→ ${pct(fw.a[j])} of the attention`, specimen: null, size: L.size, tint: S.tint } };
  }
  function slRenderScorer() {
    const L = S.sl; if (!L.attention) return;
    const { j, n, m } = slScorerModel(), cv = $('sl-scorer'); cv._model = m; Viz.drawNetwork(cv, m);
    $('sl-scorer-note').textContent = `${L.hoverNucleus == null && L.pinned == null ? 'The nucleus with the most attention on this slide' : `Nucleus ${j + 1} of this slide${L.hoverNucleus == null ? ', pinned' : ''}`}${L.reveal ? ` (${n.label ? 'atypical: ' + subtypeWord(n.subtype) : 'bland'})` : ''}. Every nucleus of the slide goes through this same little network; the softmax then compares the 20 scores, so a score only means something relative to the others on the slide.`;
  }
  function subtypeWord(key) { return { enlarged: 'enlarged', hyperchromatic: 'hyperchromatic', irregular: 'irregular contour', coarse: 'coarse chromatin', combined: 'two or more traits', bland: 'bland' }[key] || key; }
  function slRenderSummary() {
    const L = S.sl, { s, fw } = slFocusForward(), cls = L.meta.classes, D = L.D;
    const plain = new Float64Array(D); for (const h of fw.T) for (let d = 0; d < D; d++) plain[d] += h[d] / fw.T.length;
    const bars = code => { let mx = 1e-9; for (const v of code) mx = Math.max(mx, Math.abs(v)); return `<span class="code" title="${Array.from(code).map(v => v.toFixed(2)).join(', ')}">${Array.from(code).map(v => `<i class="${v < 0 ? 'neg' : 'pos'}" style="height:${Math.max(2, Math.round(Math.abs(v) / mx * 100))}%"></i>`).join('')}</span>`; };
    $('sl-summary').innerHTML = `<div class="v"><div class="lbl">${L.attention ? 'weighted average' : 'plain average'}</div>${bars(fw.z)}<div class="n">the summary</div></div>` + (L.attention ? `<div class="v"><div class="lbl">plain average</div>${bars(plain)}<div class="n">for comparison</div></div>` : '') +
      `<div class="txt"><div>Σ weight × code over the 20 nuclei gives ${D} numbers, the slide’s summary${L.attention ? '; with attention it leans towards the nuclei that weigh most, where a plain average lets 17 bland nuclei drown 3 atypical ones' : ''}.</div><div>Then a single layer on the summary: <b>P(${esc(cls[1].name)}) = ${fw.p.toFixed(2)}</b>.</div></div>`;
    const cv = $('sl-head'), m = { net: L.model.head, mode: 'features', inputCaption: `INPUT · THE SLIDE’S SUMMARY · ${D} NUMBERS`, featureNames: Array.from({ length: D }, (_, i) => `summary ${i + 1}`), x: fw.z, fw: fw.head, stage: 2, hover: L.hoverHead, activation: 'relu', activationLabel: 'ReLU', positiveName: cls[1].name, negativeName: cls[0].name, specimen: null, size: L.size, tint: S.tint };
    cv._model = m; Viz.drawNetwork(cv, m);
  }
  function slNucleusTip(i) { const L = S.sl, { s, fw } = slFocusForward(), n = s.nuclei[i]; return `nucleus ${i + 1} · attention ${pct(fw.a[i])}${L.attention ? ` (score ${Viz.fmtSigned(fw.s[i], 2)})` : ''}${L.reveal ? ` · ${n.label ? 'atypical: ' + subtypeWord(n.subtype) : 'bland'}` : ''}`; }
  // the whole model for the slide on screen, unrolled: every nucleus through the scorer, the softmax, the sum, the call
  function slRenderUnrolled() {
    const L = S.sl, { s, fw } = slFocusForward(), cls = L.meta.classes, n = s.nuclei.length;
    const m = { nuclei: s.nuclei.map((q, i) => ({ px: q.px, h: fw.T[i], s: fw.s[i], a: fw.a[i], pos: !!q.label })), D: L.D, tokenNote: L.context ? 'code, after context' : null, size: L.size, tint: S.tint, reveal: L.reveal,
      scorer: L.attention ? L.model.scorer : null, head: L.model.head, fw, shown: slShown(fw), hover: L.hoverUnrolled, walk: L.walk, positiveName: cls[1].name, negativeName: cls[0].name };
    const cv = $('sl-unrolled'); cv._model = m; Viz.drawSlideNetwork(cv, m);
    $('sl-unrolled-note').textContent = L.attention
      ? `Left to right: the ${n} nuclei of this slide, each as its code of ${L.D} numbers; one scorer, the same weights for all of them, gives each a score; the softmax compares the ${n} scores and turns them into shares that add up to 100%; the codes are added up, each weighted by its share, into the ${L.D}-number summary; a single layer on the summary makes the call. Training sends the label’s error back along the same path, through the softmax into the scorer, which is how the scorer learns where to look without a single nucleus label. Hover a nucleus to follow it, or press Walk through.`
      : `Plain average: there is no scorer, so every nucleus gets the same share, 1/${n}; the codes are averaged into the ${L.D}-number summary and a single layer makes the call. Nothing can make one nucleus count more than another.`;
  }
  // the walk-through: the slide on screen goes through the model step by step, one nucleus at a time
  const SL_WALK = { score: 110, softmax: 900, sum: 1000, head: 800, pause: 300 };
  function slWalkStart() {
    const L = S.sl; if (!L.model) return; slStop(); slDecideWalkStop();
    L.walk = { stage: L.attention ? 'score' : 'softmax', k: 0, t: 0, t0: performance.now() }; L.hoverUnrolled = null;
    slSyncWalk(); requestAnimationFrame(slWalkTick);
  }
  function slWalkStop() { const L = S.sl; if (!L.walk) return; L.walk = null; L.walkNucleus = null; slSyncWalk(); if (L.model) { slRenderSlide(); slRenderUnrolled(); } }
  function slSyncWalk() { $('sl-walk').textContent = S.sl.walk ? '■ Stop' : '▶ Walk through'; }
  function slWalkTick(now) {
    const L = S.sl, w = L.walk; if (!w) return;
    const el = Math.max(0, now - w.t0), n = slFocus().nuclei.length, next = stage => { w.stage = stage; w.t = 0; w.t0 = now; };
    if (w.stage === 'score') { w.k = Math.min(n, Math.floor(el / SL_WALK.score) + 1); L.walkNucleus = w.k - 1; if (el >= n * SL_WALK.score + SL_WALK.pause) { L.walkNucleus = null; next('softmax'); } }
    else if (w.stage === 'softmax') { w.t = Math.min(1, el / SL_WALK.softmax); if (el >= SL_WALK.softmax + SL_WALK.pause) next('sum'); }
    else if (w.stage === 'sum') { w.t = Math.min(1, el / SL_WALK.sum); if (el >= SL_WALK.sum + SL_WALK.pause) next('head'); }
    else if (w.stage === 'head') {
      w.t = Math.min(1, el / SL_WALK.head);
      if (el >= SL_WALK.head + SL_WALK.pause) {
        const { s, fw } = slFocusForward(), cls = L.meta.classes, call = fw.p >= 0.5 ? 1 : 0; slWalkStop();
        slNote(`Slide ${s.name} through the model: ${L.attention ? 'every nucleus scored by the same scorer, softmax, weighted sum' : 'plain average'}, then the single layer: P(${cls[1].name}) = ${fw.p.toFixed(2)} → ${cls[call].name} ${call === s.label ? '✓' : '✗'}.`);
        return;
      }
    }
    slRenderSlide(); slRenderUnrolled();
    requestAnimationFrame(slWalkTick);
  }
  // who looks at whom: the context layer's attention (rows asking, columns answering), the hovered nucleus's token
  // before and after context, and whom it listens to most
  function slRenderContext() {
    const L = S.sl; if (!L.context || !L.model || !L.model.context) return;
    const { s, fw } = slFocusForward(), A = slAttentionView(fw), n = s.nuclei.length, cost = fw.ctx.cost, j = L.dwalk ? L.dwalk.i : slShown(fw);
    const pair = L.hoverAtt || (L.hoverDecide && L.hoverDecide.j != null ? { i: j, j: L.hoverDecide.j } : null);
    Viz.drawAttentionMap($('sl-attmap'), { A, thumbs: s.nuclei.map(q => q.px), size: L.size, tint: S.tint, hover: j, pair, pos: s.pos, reveal: L.reveal });
    const costWord = cost > 2 ? 'the layer listens almost only to immediate neighbours' : cost > 0.8 ? 'near nuclei count more, but far ones still count' : 'distance hardly matters to it';
    $('sl-attmap-note').textContent = L.attView === 'match'
      ? `Match only: the shares the softmax would give from query · key alone, as if distance cost nothing. This is the part of the decision that reads what a nucleus looks like: the nuclei whose keys fit a query get the listening, wherever they sit on the slide. Switch back to see what the distance cost (${cost.toFixed(2)} per cell) does to it.`
      : L.attView === 'distance'
        ? `Distance only: the shares from the learned distance cost alone, as if every nucleus looked the same; each cell of distance divides a nucleus’s weight by ${Math.exp(cost).toFixed(1)}, so ${costWord}. This is the part of the decision that reads where a nucleus is. The layer uses both: match minus distance cost, then the softmax.`
        : `Each row is one nucleus asking, each column one answering: how much of its listening goes to each of the other ${n - 1} (a row adds up to 100%; a nucleus does not listen to itself, its own code stays through the residual). The learned distance cost is ${cost.toFixed(2)} per cell: every extra cell of distance divides a nucleus’s weight by ${Math.exp(cost).toFixed(1)}, so ${costWord}. Match only and distance only show the two parts of the decision on their own.`;
    const row = Array.from(A[j], (w, k) => ({ k, w })).filter(e => e.k !== j).sort((a, b) => b.w - a.w), top = row.slice(0, 3);
    const bars = code => { let mx = 1e-9; for (const v of code) mx = Math.max(mx, Math.abs(v)); return `<span class="code" title="${Array.from(code).map(v => v.toFixed(2)).join(', ')}">${Array.from(code).map(v => `<i class="${v < 0 ? 'neg' : 'pos'}" style="height:${Math.max(2, Math.round(Math.abs(v) / mx * 100))}%"></i>`).join('')}</span>`; };
    const where = k => (s.dist[j][k] <= 1 ? 'a neighbour' : s.dist[j][k] < 1.5 ? 'diagonal' : `${s.dist[j][k].toFixed(1)} cells away`);
    $('sl-context-focus').innerHTML = `<div class="v"><div class="lbl">nucleus ${j + 1} · code</div>${bars(s.H[j])}<div class="n">before context</div></div><div class="v"><div class="lbl">after context</div>${bars(fw.T[j])}<div class="n">what the scorer sees</div></div>` +
      `<div class="txt"><div>Nucleus ${j + 1}${slShownWord()} listens most to ${top.map(e => `nucleus ${e.k + 1} (${pct(e.w)}, ${where(e.k)})`).join(', ')}${L.attView === 'both' ? '' : ` (from the ${L.attView} alone)`}.</div><div>What it hears is added to its own code, so the scorer can weigh “atypical, with atypical neighbours” rather than “atypical” alone.</div></div>`;
  }
  // how the shown nucleus decides where to look: the context layer's attention for it, taken apart
  function slRenderDecide() {
    const L = S.sl; if (!L.context || !L.model || !L.model.context) return;
    const { s, fw } = slFocusForward(), ctx = fw.ctx, cl = L.model.context, ex = cl.explain(ctx), n = s.nuclei.length, i = L.dwalk ? L.dwalk.i : slShown(fw), cost = ctx.cost;
    const m = { asker: i, nuclei: s.nuclei.map(q => ({ px: q.px, pos: !!q.label })), X: ctx.X, Q: ctx.Q, K: ctx.K, V: ctx.V, cost, match: ex.match[i], costs: ex.cost[i], shares: ctx.A[i], C: ctx.C[i], heard: Float64Array.from(ctx.Xp[i], (v, d) => v - ctx.X[i][d]), after: ctx.Xp[i], final: ctx.Y[i],
      D: L.D, dk: cl.dk, ffn: cl.F, size: L.size, tint: S.tint, reveal: L.reveal, hover: L.hoverDecide, walk: L.dwalk };
    const cv = $('sl-decide'); cv._model = m; Viz.drawDecision(cv, m);
    $('sl-decide-title').textContent = `How nucleus ${i + 1} decides where to look`;
    const row = Array.from(ctx.A[i], (w, k) => ({ k, w })).filter(e => e.k !== i).sort((a, b) => b.w - a.w), best = row[0], bestMatch = row.slice().sort((a, b) => ex.match[i][b.k] - ex.match[i][a.k])[0];
    $('sl-decide-note').textContent = `Nucleus ${i + 1}${slShownWord()} turns its code into a query; each of the other ${n - 1} turns its code into a key and a value, with the same three weight maps. Query × key, ${cl.dk} products added up, is the match: how well that nucleus fits what nucleus ${i + 1} is looking for (best match: nucleus ${bestMatch.k + 1}, ${Viz.fmtSigned(ex.match[i][bestMatch.k], 2)}). The learned distance cost, ${cost.toFixed(2)} per cell, comes off, and the softmax over the ${n - 1} scores gives shares that add up to 100% (most to nucleus ${best.k + 1}, ${pct(best.w)}, ${s.dist[i][best.k] <= 1 ? 'a neighbour' : s.dist[i][best.k] < 1.5 ? 'diagonal' : `${s.dist[i][best.k].toFixed(1)} cells away`}). The values, each weighted by its share, add up to the message, ${cl.dk} numbers; projected back to ${L.D}, it is added to nucleus ${i + 1}’s own code, and a small feed-forward is added after that. Nothing else changes: the scorer sees the same nucleus, plus what it heard. Hover a row to read its numbers; press Walk through to watch the decision happen.`;
  }
  function slDecideRowTip(m, j) { const L = S.sl, s = slFocus(), d = s.dist[m.asker][j], where = d <= 1 ? 'a neighbour' : d < 1.5 ? 'diagonal' : `${d.toFixed(1)} cells away`; return `nucleus ${j + 1} · ${where} · match ${Viz.fmtSigned(m.match[j], 2)} − ${m.costs[j].toFixed(2)} = ${Viz.fmtSigned(m.match[j] - m.costs[j], 2)} → ${pct(m.shares[j])} of what nucleus ${m.asker + 1} hears${L.reveal ? ` · ${s.nuclei[j].label ? 'atypical: ' + subtypeWord(s.nuclei[j].subtype) : 'bland'}` : ''}`; }
  const DE_WALK = { query: 700, keys: 40, match: 110, distance: 900, softmax: 900, message: 1000, add: 1000, pause: 300 };
  function slDecideWalkStart() {
    const L = S.sl; if (!L.model || !L.context) return; slStop(); slWalkStop();
    const { fw } = slFocusForward();
    L.dwalk = { stage: 'query', k: 0, t: 0, t0: performance.now(), i: slShown(fw) }; L.hoverDecide = null; L.walkNucleus = L.dwalk.i;
    slSyncDecideWalk(); slRenderSlide(); slRenderContext(); requestAnimationFrame(slDecideTick);
  }
  function slDecideWalkStop() { const L = S.sl; if (!L.dwalk) return; L.dwalk = null; L.walkNucleus = null; slSyncDecideWalk(); if (L.model) { slRenderSlide(); slRenderContext(); slRenderDecide(); } }
  function slSyncDecideWalk() { $('sl-decide-walk').textContent = S.sl.dwalk ? '■ Stop' : '▶ Walk through'; }
  function slDecideTick(now) {
    const L = S.sl, w = L.dwalk; if (!w) return;
    const el = Math.max(0, now - w.t0), n = slFocus().nuclei.length, next = st => { w.stage = st; w.k = 0; w.t = 0; w.t0 = now; };
    const timed = (dur, after) => { w.t = Math.min(1, el / dur); if (el >= dur + DE_WALK.pause) next(after); };
    if (w.stage === 'query') timed(DE_WALK.query, 'keys');
    else if (w.stage === 'keys') { w.k = Math.min(n, Math.floor(el / DE_WALK.keys) + 1); if (el >= n * DE_WALK.keys + DE_WALK.pause) next('match'); }
    else if (w.stage === 'match') { w.k = Math.min(n, Math.floor(el / DE_WALK.match) + 1); if (el >= n * DE_WALK.match + DE_WALK.pause) next('distance'); }
    else if (w.stage === 'distance') timed(DE_WALK.distance, 'softmax');
    else if (w.stage === 'softmax') timed(DE_WALK.softmax, 'message');
    else if (w.stage === 'message') timed(DE_WALK.message, 'add');
    else if (w.stage === 'add') {
      w.t = Math.min(1, el / DE_WALK.add);
      if (el >= DE_WALK.add + DE_WALK.pause) {
        const i = w.i, { fw } = slFocusForward(), row = Array.from(fw.ctx.A[i], (v, k) => ({ k, v })).filter(e => e.k !== i).sort((a, b) => b.v - a.v).slice(0, 2);
        slDecideWalkStop(); slNote(`Nucleus ${i + 1} through the context layer: query against every key, minus the distance cost, softmax; it listens most to ${row.map(e => `nucleus ${e.k + 1} (${pct(e.v)})`).join(' and ')}, and what it hears is added to its own code before the scorer sees it.`);
        return;
      }
    }
    slRenderDecide();
    requestAnimationFrame(slDecideTick);
  }
  function slRenderFocus() { slRenderSlide(); slRenderRank(); if (slOnTest() && !slTestResult()) return; slRenderScorer(); slRenderUnrolled(); slRenderContext(); slRenderDecide(); }
  function slRenderCurves() {
    const L = S.sl, c = Viz.colors(), testCol = getComputedStyle(document.documentElement).getPropertyValue('--test-series').trim(), last = L.hist[L.hist.length - 1];
    Viz.drawCurves($('sl-loss'), { history: L.hist, key: 'loss', showTest: true, maxEpoch: L.epochs });
    Viz.drawCurves($('sl-acc'), { history: L.hist, key: 'acc', showTest: true, maxEpoch: L.epochs });
    Viz.drawSeries($('sl-mass'), { hist: L.hist, keys: [{ key: 'mass', color: c.accent }, { key: 'testMass', color: testCol, dash: true }], maxEpoch: L.epochs, pct: true, baseline: L.share, baselineLabel: 'uniform' });
    $('sl-loss-now').textContent = `${last.loss.toFixed(3)} · test ${last.testLoss.toFixed(3)}`; $('sl-acc-now').textContent = `${pct(last.acc)} · test ${pct(last.testAcc)}`; $('sl-mass-now').textContent = `${pct(last.mass)} · test ${pct(last.testMass)}`;
  }
  function slRenderTrays() {
    const L = S.sl, cls = L.meta.classes, focus = slFocus();
    [['train', L.train, L.trainEval], ['test', L.test, L.testEval]].forEach(([split, slides, ev]) => slides.forEach((s, k) => {
      const el = L.thumbs.get(s.id), p = ev.probs[k], call = p >= 0.5 ? 1 : 0;
      el.className = `thumb slide call-${call}${call !== s.label ? ' wrong' : ''}${focus === s ? ' selected' : ''}${L.lastSlide === s.id && split === 'train' ? ' in-batch' : ''}`;
      el.title = `${s.name} · ${split === 'train' ? 'training' : 'test'} slide · label ${cls[s.label].name} · call ${cls[call].name} (P ${p.toFixed(2)})`;
    }));
  }
  function slRender(full) {
    const L = S.sl; if (!L.model) return;
    slRenderStatus(); slRenderFocus(); slRenderSummary();
    if (full) { slRenderCurves(); slRenderTrays(); }
    slSyncButtons();
  }
  function slEnter() {
    const L = S.sl; if (!slData()) { slNote('The slides did not load.'); return; }
    if (!slBuild()) return;
    slPlaceCards();
    if (!L.model || L.modelQuestion !== L.loaded) slReset(L.model ? 'A new question — fresh random weights.' : 'Untrained: the attention weights are near uniform and the calls are guesses. Step a slide to watch one gradient step, or press Train and watch the right-hand curve climb.');
    else { slEncode(); slRender(true); }
  }
  // ---- the slides' other two steps: Specimens (the slides with their labels) and Test (the held-out slides, one at a time)
  function slSetReveal(v) { S.sl.reveal = v; for (const id of ['sl-reveal', 'sl-data-reveal', 'sl-test-reveal']) $(id).checked = v; slRenderAll(); }
  function slRenderAll() { const L = S.sl; if (S.world !== 'slides' || !L.built) return; if (S.stage === 'data') slRenderData(); else if (S.stage === 'test') slRenderTest(); else if (L.model) slRender(true); }
  function slTrays(kind, onClick) { // a panel's trays, rebuilt when the question changes
    const L = S.sl; if (L.trayFor[kind] === L.loaded) return; L.trayFor[kind] = L.loaded;
    const map = new Map(), fill = (id, tray) => { const el = $(id); el.innerHTML = ''; for (const s of tray) { const b = slThumb(s, onClick); el.appendChild(b); map.set(s.id, b); } };
    if (kind === 'data') { fill('sl-data-train-tray', L.train); fill('sl-data-test-tray', L.test); L.dataThumbs = map; $('sl-data-train-label').textContent = `Training slides (${L.train.length})`; $('sl-data-test-label').textContent = `Test slides (${L.test.length}) · held out`; }
    else { fill('sl-test-results', L.test); L.testThumbs = map; $('sl-test-count').textContent = `${L.test.length} slides`; }
  }
  function slEnterData() { const L = S.sl; if (!slData() || !slBuild()) return; slTrays('data', slDataSelect); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id; slRenderData(); }
  function slDataSelect(id) { S.sl.dataSelected = id; slRenderData(); }
  function slDataNeighbour(step) { const L = S.sl, i = L.all.findIndex(s => s.id === L.dataSelected); slDataSelect(L.all[(i + step + L.all.length) % L.all.length].id); }
  function slRenderData() {
    const L = S.sl; slTrays('data', slDataSelect); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id;
    const s = L.byId.get(L.dataSelected) || L.train[0], cls = L.meta.classes, k = s.nuclei.filter(n => n.label).length;
    $('sl-data-title').textContent = L.meta.question; $('sl-data-sub').textContent = `${L.train.length} training and ${L.test.length} test slides of ${L.meta.perSlide} nuclei · one label per slide`;
    $('sl-data-blurb').textContent = L.meta.blurb;
    Viz.drawSlide($('sl-data-canvas'), { nuclei: s.nuclei.map(n => ({ px: n.px, a: 0, pos: !!n.label })), cols: L.cols, size: L.size, tint: S.tint, reveal: L.reveal, plain: true });
    $('sl-data-slide-title').textContent = `Slide ${s.name} · ${s.split === 'train' ? 'training' : 'test'} · label: ${cls[s.label].name}${L.reveal ? ` (${k} atypical nucle${k === 1 ? 'us' : 'i'})` : ''}`;
    $('sl-data-note').textContent = `${L.meta.perSlide} nuclei from the pool, each in a random orientation, and one label for the whole slide. ${L.reveal ? 'The dots mark the atypical nuclei; the model never sees them.' : 'Nobody labels the nuclei: tick Reveal to see which are atypical.'} Click a slide below, or step through them.`;
    for (const [id, el] of L.dataThumbs) { const sl = L.byId.get(id); el.classList.toggle('selected', id === L.dataSelected); el.classList.toggle('call-1', !!sl.label); el.classList.toggle('call-0', !sl.label); }
  }
  function slEnterTest() { const L = S.sl; if (!slData() || !slBuild()) return; if (!L.model || L.modelQuestion !== L.loaded) slReset(); slTrays('test', slTestSelect); slRenderTest(); }
  function slTestClear() { const L = S.sl; L.trial = { next: 0, results: new Map() }; L.testSelected = null; slRenderTest(); }
  function slClassifyNext(quiet) {
    const L = S.sl, T = L.trial; if (!L.model || T.next >= L.test.length) return false;
    const s = L.test[T.next++], fw = L.model.forward(s.H, s.dist);
    T.results.set(s.id, { p: fw.p, call: fw.p >= 0.5 ? 1 : 0, a: fw.a, fw }); L.testSelected = s.id;
    if (!quiet) slRenderTest();
    return true;
  }
  function slClassifyAll() { const L = S.sl; if (!L.model) return; const go = () => { if (slClassifyNext(false) && S.world === 'slides' && S.stage === 'test') setTimeout(go, reducedMotion ? 0 : 120); }; go(); }
  function slTestSelect(id) { const L = S.sl; if (!L.trial.results.has(id)) return; L.testSelected = id; slRenderTest(); }
  function slTestStats() { const L = S.sl; let tp = 0, tn = 0, fp = 0, fn = 0; for (const [id, r] of L.trial.results) { const y = L.byId.get(id).label; if (r.call && y) tp++; else if (!r.call && !y) tn++; else if (r.call && !y) fp++; else fn++; } const n = tp + tn + fp + fn; return { n, tp, tn, fp, fn, acc: n ? (tp + tn) / n : null, sens: tp + fn ? tp / (tp + fn) : null, spec: tn + fp ? tn / (tn + fp) : null }; }
  function slRenderTest() {
    const L = S.sl; slTrays('test', slTestSelect);
    const T = L.trial, cls = L.meta.classes, st = slTestStats(), pos = cls[1].name, neg = cls[0].name;
    $('sl-stat-n').textContent = `${st.n} / ${L.test.length}`;
    $('sl-stat-acc').textContent = st.acc == null ? '–' : pct(st.acc); $('sl-stat-acc-sub').textContent = st.n ? `${st.tp + st.tn} of ${st.n} correct` : 'no calls yet';
    $('sl-stat-sens').textContent = st.sens == null ? '–' : pct(st.sens); $('sl-stat-sens-sub').textContent = st.tp + st.fn ? `${st.tp} of ${st.tp + st.fn} caught` : 'none seen yet';
    $('sl-stat-spec').textContent = st.spec == null ? '–' : pct(st.spec); $('sl-stat-spec-sub').textContent = st.tn + st.fp ? `${st.tn} of ${st.tn + st.fp} cleared` : 'none seen yet';
    const cell = (v, kind) => `<div class="cell ${v ? kind : 'empty'}">${v}</div>`;
    $('sl-confusion').innerHTML = `<div></div><div class="hd">called ${esc(neg)}</div><div class="hd">called ${esc(pos)}</div><div class="rh">truth ${esc(neg)}</div>${cell(st.tn, 'hit')}${cell(st.fp, 'miss')}<div class="rh">truth ${esc(pos)}</div>${cell(st.fn, 'miss')}${cell(st.tp, 'hit')}`;
    const done = T.next >= L.test.length;
    $('sl-classify-next').disabled = done; $('sl-classify-all').disabled = done;
    $('sl-classify-next').textContent = done ? `All ${L.test.length} classified` : `Classify next slide (${T.next + 1} of ${L.test.length})`;
    $('sl-classify-all').textContent = `Classify all ${L.test.length}`;
    $('sl-test-warning').hidden = !(L.model && L.model.steps === 0);
    document.querySelectorAll('[data-slcalled="0"]').forEach(el => { el.textContent = `called ${neg}`; }); document.querySelectorAll('[data-slcalled="1"]').forEach(el => { el.textContent = `called ${pos}`; });
    const s = L.testSelected != null ? L.byId.get(L.testSelected) : null, r = s ? T.results.get(s.id) : null;
    slPlaceCards(); slRenderFocus(); // the viewer and the ranking, and the cards once the slide is called
    if (s && r) {
      const k = s.nuclei.filter(n => n.label).length;
      slRenderSummary();
      $('sl-test-slide-title').textContent = `Slide ${s.name} · test · truth: ${cls[s.label].name}${L.reveal ? ` (${k} atypical nucle${k === 1 ? 'us' : 'i'})` : ''}`;
      $('sl-test-call').innerHTML = `The model says <b>P(${esc(pos)}) = ${r.p.toFixed(2)}</b> → <span class="${r.call === s.label ? 'good-text' : 'bad-text'}">${esc(cls[r.call].name)} ${r.call === s.label ? '✓' : '✗'}</span>. ${L.attention ? 'Frames and percentages are the attention weights; the weights never change here.' : 'Plain average: every nucleus weighs the same.'}${L.context ? ' Hover a nucleus for whom it listens to; click it to keep it.' : ''}`;
      $('sl-test-rank-note').textContent = (L.attention ? 'With Reveal ticked the dots say which nuclei were atypical: a call for the right reason lands the attention on them.' : 'A plain average has no ranking to show: every nucleus counts the same.') + ' The cards below take this call apart, as they did while training; hover a nucleus on the slide to follow it through them.';
    } else {
      const s0 = L.test[Math.min(T.next, L.test.length - 1)];
      $('sl-test-slide-title').textContent = `Slide ${s0.name} · test · next up`;
      $('sl-test-call').textContent = 'Press Classify next slide: the model reads the twenty nuclei and makes one call, and the truth is shown after.';
      $('sl-test-rank-note').textContent = '';
    }
    for (const [id, el] of L.testThumbs) { const rr = T.results.get(id), sl = L.byId.get(id), right = !!rr && rr.call === sl.label; el.classList.toggle('selected', id === L.testSelected); el.classList.toggle('call-0', !!rr && rr.call === 0); el.classList.toggle('call-1', !!rr && rr.call === 1); el.classList.toggle('wrong', !!rr && !right); el.classList.toggle('right', right); el.querySelector('.badge').textContent = right ? '✓' : '✗'; }
  }
  // the lecture introduces the slides late: the stage stays hidden until recipe ⑬, key 5 or #slides brings it in
  function bindSlides() {
    document.querySelectorAll('#sl-pool-seg button').forEach(b => b.addEventListener('click', () => { const att = b.dataset.att === '1'; if (att === S.sl.attention) return; S.sl.attention = att; slReset(att ? 'Attention on: a scorer weighs the nuclei — fresh random weights.' : 'Plain average: every nucleus weighs the same and the scorer is switched off — fresh random weights.'); }));
    $('sl-units').addEventListener('change', () => { S.sl.units = +$('sl-units').value; slReset(`${S.sl.units} attention units — fresh random weights.`); });
    $('sl-backbone').addEventListener('change', () => { S.backbone = $('sl-backbone').value; slReset(`The code now comes from ${codeEncoder().describe} — fresh random weights.`); });
    $('sl-epochs').addEventListener('input', () => { S.sl.epochs = +$('sl-epochs').value; $('sl-epochs-val').textContent = S.sl.epochs; if (S.sl.model) { slRenderStatus(); slRenderCurves(); } });
    $('sl-speed').addEventListener('input', () => { S.sl.speed = speedFromSlider(+$('sl-speed').value); $('sl-speed-val').textContent = `${S.sl.speed} epochs/s`; });
    $('sl-seed').addEventListener('change', () => { S.sl.seed = Math.max(1, Math.round(+$('sl-seed').value) || 1); slReset(`Seed ${S.sl.seed} — fresh random weights.`); });
    for (const id of ['sl-reveal', 'sl-data-reveal', 'sl-test-reveal']) $(id).addEventListener('change', ev => slSetReveal(ev.target.checked));
    $('sl-data-prev').addEventListener('click', () => slDataNeighbour(-1)); $('sl-data-next').addEventListener('click', () => slDataNeighbour(1));
    $('sl-classify-next').addEventListener('click', () => slClassifyNext()); $('sl-classify-all').addEventListener('click', slClassifyAll); $('sl-test-clear').addEventListener('click', slTestClear);
    $('sl-step-slide').addEventListener('click', slStepSlide);
    $('sl-step-epoch').addEventListener('click', slStepEpoch);
    $('sl-train').addEventListener('click', () => (S.sl.running ? slStop('Paused.') : slStart()));
    $('sl-reset').addEventListener('click', () => slReset('Weights re-initialised from the seed.'));
    $('sl-prev').addEventListener('click', () => slNeighbour(-1));
    $('sl-next').addEventListener('click', () => slNeighbour(1));
    const pin = i => { const L = S.sl; if (i == null || !L.context) return; L.pinned = L.pinned === i ? null : i; slDecideWalkStop(); slRenderFocus(); slNote(L.pinned == null ? 'Unpinned: the cards follow the hovered nucleus, else the one with the most attention.' : `Nucleus ${i + 1} pinned: the cards follow it until you click it again.`); };
    for (const [cvId, tipId] of [['sl-canvas', 'sl-tip'], ['sl-test-canvas', 'sl-test-tip']]) { // the Train step's viewer and the Test step's: hovering follows the nucleus everywhere, a click keeps it
      const cv = $(cvId), tip = $(tipId);
      cv.addEventListener('mousemove', ev => {
        const L = S.sl; if (!L.model || (slOnTest() && !slTestResult())) return;
        const r = cv.getBoundingClientRect(), i = Viz.hitSlide(cv, ev.clientX - r.left, ev.clientY - r.top);
        if (i !== L.hoverNucleus) { L.hoverNucleus = i; slRenderFocus(); }
        if (i != null) { tip.hidden = false; tip.textContent = slNucleusTip(i); tip.style.left = (ev.clientX - r.left) + 'px'; tip.style.top = (ev.clientY - r.top) + 'px'; }
        else tip.hidden = true;
      });
      cv.addEventListener('mouseleave', () => { const L = S.sl; tip.hidden = true; if (L.hoverNucleus != null) { L.hoverNucleus = null; if (L.model) { slRenderFocus(); } } });
      cv.addEventListener('click', ev => { const L = S.sl; if (!L.model || (slOnTest() && !slTestResult())) return; const r = cv.getBoundingClientRect(); pin(Viz.hitSlide(cv, ev.clientX - r.left, ev.clientY - r.top)); });
    }
    document.querySelectorAll('#sl-attview-seg button').forEach(b => b.addEventListener('click', () => { S.sl.attView = b.dataset.view; slSyncControls(); if (S.sl.model) { slRenderSlide(); slRenderContext(); } }));
    document.querySelectorAll('#sl-links-seg button').forEach(b => b.addEventListener('click', () => { S.sl.linksAll = b.dataset.links === '1'; slSyncControls(); if (S.sl.model) slRenderSlide(); }));
    $('sl-links-min').addEventListener('input', () => { S.sl.linksMin = +$('sl-links-min').value / 100; $('sl-links-min-val').textContent = pct(S.sl.linksMin); if (S.sl.model) slRenderSlide(); });
    document.querySelectorAll('#sl-context-seg button').forEach(b => b.addEventListener('click', () => { const on = b.dataset.ctx === '1'; if (on === S.sl.context) return; S.sl.context = on; slReset(on ? 'Context on: one layer of self-attention lets every nucleus read the others before the pooling — fresh random weights.' : 'Context off: every nucleus is scored on its own — fresh random weights.'); }));
    const ca = $('sl-attmap'), ta = $('sl-attmap-tip');
    ca.addEventListener('mousemove', ev => {
      const L = S.sl; if (!L.model || !L.context || L.walk) return;
      const r = ca.getBoundingClientRect(), hit = Viz.hitAttentionMap(ca, ev.clientX - r.left, ev.clientY - r.top);
      const prevI = L.hoverNucleus, prevKey = L.hoverAtt ? `${L.hoverAtt.i}:${L.hoverAtt.j}` : '';
      L.hoverAtt = hit; if (hit) L.hoverNucleus = hit.i;
      if (hit && hit.i !== prevI) { slRenderFocus(); }
      else if ((hit ? `${hit.i}:${hit.j}` : '') !== prevKey) slRenderContext();
      if (hit) {
        const { s, fw } = slFocusForward(), c = fw.ctx, ex = L.model.context.explain(c), view = slAttentionView(fw);
        ta.hidden = false;
        ta.textContent = hit.i === hit.j ? `nucleus ${hit.i + 1} does not listen to itself` : `nucleus ${hit.i + 1} listens to nucleus ${hit.j + 1}: ${pct(view[hit.i][hit.j])}${L.attView === 'both' ? '' : ` from the ${L.attView} alone (${pct(c.A[hit.i][hit.j])} in fact)`} · match ${Viz.fmtSigned(ex.match[hit.i][hit.j], 2)} − distance ${s.dist[hit.i][hit.j].toFixed(1)} × ${c.cost.toFixed(2)} · click to pin the row`;
        const half = ta.offsetWidth / 2 + 4; ta.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; ta.style.top = (ev.clientY - r.top) + 'px';
      } else ta.hidden = true;
    });
    ca.addEventListener('mouseleave', () => { const L = S.sl; ta.hidden = true; if (L.hoverAtt || L.hoverNucleus != null) { L.hoverAtt = null; L.hoverNucleus = null; if (L.model) { slRenderFocus(); } } });
    ca.addEventListener('click', ev => { const L = S.sl; if (!L.model || !L.context) return; const r = ca.getBoundingClientRect(), hit = Viz.hitAttentionMap(ca, ev.clientX - r.left, ev.clientY - r.top); if (hit) pin(hit.i); });
    // the decision card: hovering a row outlines the pair on the map; the numbers explain themselves
    const cd = $('sl-decide'), td = $('sl-decide-tip'), rowOf = h => (h && h.j != null ? h.j : null), keyOf = h => (h ? `${h.kind}:${h.j == null ? '' : h.j}:${h.d == null ? '' : h.d}` : '');
    cd.addEventListener('mousemove', ev => {
      const L = S.sl, m = cd._model; if (!m || L.dwalk) return;
      const r = cd.getBoundingClientRect(), hit = Viz.hitDecision(cd, ev.clientX - r.left, ev.clientY - r.top, m), prev = L.hoverDecide;
      L.hoverDecide = hit;
      if (keyOf(prev) !== keyOf(hit)) { slRenderDecide(); if (rowOf(prev) !== rowOf(hit)) slRenderContext(); }
      if (hit) { td.hidden = false; td.textContent = hit.kind === 'asker' ? slNucleusTip(m.asker) : hit.kind === 'row' ? slDecideRowTip(m, hit.j) : hit.text; const half = td.offsetWidth / 2 + 4; td.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; td.style.top = (ev.clientY - r.top) + 'px'; }
      else td.hidden = true;
    });
    cd.addEventListener('mouseleave', () => { const L = S.sl; td.hidden = true; if (L.hoverDecide) { L.hoverDecide = null; if (L.model) { slRenderDecide(); slRenderContext(); } } });
    $('sl-decide-walk').addEventListener('click', () => (S.sl.dwalk ? slDecideWalkStop() : slDecideWalkStart()));
    // the unrolled diagram: hovering a nucleus row follows it everywhere; the other parts explain themselves
    const cu = $('sl-unrolled'), tu = $('sl-unrolled-tip'), hitKey = h => (h ? `${h.kind}:${h.i == null ? '' : h.i}:${h.d == null ? '' : h.d}` : '');
    cu.addEventListener('mousemove', ev => {
      const L = S.sl, m = cu._model; if (!m || L.walk) return;
      const r = cu.getBoundingClientRect(), hit = Viz.hitSlideNetwork(cu, ev.clientX - r.left, ev.clientY - r.top, m), prev = L.hoverUnrolled;
      const ni = hit && hit.kind === 'nucleus' ? hit.i : null;
      L.hoverUnrolled = hit;
      if (ni !== L.hoverNucleus) { L.hoverNucleus = ni; slRenderFocus(); }
      else if (hitKey(prev) !== hitKey(hit)) slRenderUnrolled();
      if (hit) { tu.hidden = false; tu.textContent = hit.kind === 'nucleus' ? slNucleusTip(hit.i) : hit.text; const half = tu.offsetWidth / 2 + 4; tu.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; tu.style.top = (ev.clientY - r.top) + 'px'; } // kept inside the (scrollable) wrapper else tu.hidden = true;
    });
    cu.addEventListener('mouseleave', () => { const L = S.sl; tu.hidden = true; const had = L.hoverNucleus != null || L.hoverUnrolled; L.hoverUnrolled = null; L.hoverNucleus = null; if (had && L.model) { slRenderFocus(); } });
    $('sl-walk').addEventListener('click', () => (S.sl.walk ? slWalkStop() : slWalkStart()));
    for (const [cvId, tipId, key, render] of [['sl-scorer', 'sl-scorer-tip', 'hoverScorer', slRenderScorer], ['sl-head', 'sl-head-tip', 'hoverHead', slRenderSummary]]) {
      const c2 = $(cvId), t2 = $(tipId);
      c2.addEventListener('mousemove', ev => {
        const m = c2._model; if (!m) return;
        const r = c2.getBoundingClientRect(), hit = Viz.hitNetwork(c2, ev.clientX - r.left, ev.clientY - r.top, m), prev = S.sl[key];
        S.sl[key] = hit;
        if (hit) { t2.hidden = false; t2.textContent = hit.text; t2.style.left = (ev.clientX - r.left) + 'px'; t2.style.top = (ev.clientY - r.top) + 'px'; } else t2.hidden = true;
        if ((prev && prev.ref) !== (hit && hit.ref)) render();
      });
      c2.addEventListener('mouseleave', () => { S.sl[key] = null; t2.hidden = true; if (S.sl.model) render(); });
    }
  }

  // ------------------------------------------------------------------ the Fields question: is it invasive?
  // 450 strips of bladder (400 training, 50 held out), five patterns, two labels each: carcinoma in situ? invasion?
  // Every nucleus of a field is a token (the frozen encoder's code for its crop, masked to the nucleus by the field's
  // segmentation, plus its position), two layers of self-attention let the nuclei look at each other, and two attention
  // heads over the same tokens answer the two questions. The fields (5.7 MB) load on demand, the first time the
  // question is opened, from the page's data folder or, failing that, from the published page.
  const FD = { unit: 16, dk: 8, ffn: 8, layers: 2, heads: 1, costInit: 1.5, batch: 1, jitter: 0.015, cisPatterns: ['cis', 'cisvbn', 'inv'], names: ['CIS', 'invasion'],
    sources: ['data/fields/fields_data.js', 'https://cdn.jsdelivr.net/gh/drdoubleb/neural-networks@main/data/fields/fields_data.js', 'https://drdoubleb.github.io/neural-networks/data/fields/fields_data.js'] }; // the page's own folder, then a mirror of the repository, then the published page
  const NFL = window.NucleusFields;
  function fdNote(msg) { $(fdOnTest() ? 'fd-test-note' : 'fd-note').textContent = msg || ''; } // the note of the step on screen
  function fdData() { const D = window.LECTURE_FIELDS; return D && D.meta ? D : null; }
  const fdCisOf = f => (FD.cisPatterns.includes(f.pattern) ? 1 : 0);
  const fdPattern = f => S.fd.meta.patterns.find(p => p.key === f.pattern);
  // the fields' file, fetched once, from the first source that answers
  function fdLoad() {
    const L = S.fd; if (fdData()) return Promise.resolve(fdData());
    if (L.loading) return L.loading;
    L.loading = new Promise((resolve, reject) => {
      const tryUrl = k => { if (k >= FD.sources.length) { reject(new Error('no source answered')); return; } const el = document.createElement('script'); el.src = FD.sources[k]; el.onload = () => (fdData() ? resolve(fdData()) : tryUrl(k + 1)); el.onerror = () => tryUrl(k + 1); document.head.appendChild(el); };
      tryUrl(0);
    });
    return L.loading;
  }
  const fdYield = () => new Promise(r => setTimeout(r, 0));
  // the fields as the page keeps them: the pixels with their grain, the segmentation, every nucleus with its truth,
  // the distances between the nuclei in nucleus diameters; the thumbnails and the pattern table
  async function fdBuild(progress) {
    const L = S.fd; if (L.built) return true;
    const D = await fdLoad(), meta = D.meta, K = meta.nucleus.reduce((o, k, i) => Object.assign(o, { [k]: i }), {});
    L.meta = meta; L.w = meta.w; L.h = meta.h; L.size = meta.size;
    const all = D.train.concat(D.test);
    progress(`Decoding the ${all.length} fields…`);
    const decoded = await Promise.all(all.map(f => Promise.all([NFL.decodePNGBrowser(f.png, meta.w, meta.h), NFL.decodePNGBrowser(f.seg, meta.w, meta.h)])));
    const mk = (f, i) => {
      const [raw, seg] = decoded[i], px = NFL.withGrain(raw, f.grainSeed, meta.grain);
      const nuclei = f.nuclei.map(n => ({ x: n[K.x], y: n[K.y], kind: meta.kinds[n[K.kind]], atypical: !!n[K.atypical], subtype: meta.subtypes[n[K.subtype]], below: !!n[K.below], nest: n[K.nest] }));
      const dist = nuclei.map(a => Float64Array.from(nuclei, b => Math.hypot(a.x - b.x, a.y - b.y) / FD.unit)), xy = nuclei.map(n => [n.x / meta.w * 2 - 1, n.y / meta.h * 2 - 1]);
      return { id: f.id, name: f.name, split: f.split, pattern: f.pattern, label: f.label, cis: fdCisOf(f), px, seg, grainSeed: f.grainSeed, membrane: f.membrane, nests: f.nests, nuclei, dist, xy, y: [fdCisOf(f), f.label], pos: [nuclei.map(n => n.atypical), nuclei.map(n => n.atypical && n.below)], codes: {}, crops: {} };
    };
    L.all = all.map(mk); L.train = L.all.filter(s => s.split === 'train'); L.test = L.all.filter(s => s.split === 'test'); L.byId = new Map(L.all.map(s => [s.id, s]));
    L.nNuclei = L.all.reduce((a, s) => a + s.nuclei.length, 0);
    L.share = [0, 1].map(k => { const pos = L.train.filter(s => s.y[k]); return pos.reduce((a, s) => a + s.pos[k].filter(Boolean).length / s.nuclei.length, 0) / pos.length; }); // what uniform attention gives each head's nuclei
    L.thumbs = new Map(); L.trayFor = {};
    for (const [id, tray] of [['fd-train-tray', L.train], ['fd-test-tray', L.test]]) { const el = $(id); el.innerHTML = ''; for (const s of tray) { const b = fdThumb(s, fdSelect); el.appendChild(b); L.thumbs.set(s.id, b); } }
    $('fd-train-label').textContent = `Training fields (${L.train.length}: ${meta.patterns.map(p => `${L.train.filter(s => s.pattern === p.key).length} ${p.short.toLowerCase()}`).join(', ')})`;
    $('fd-test-label').textContent = `Test fields (${L.test.length}) · never trained on, scored as it goes`;
    $('fd-blurb').innerHTML = `<strong>${esc(meta.question)}</strong> ${esc(meta.blurb)} Every nucleus becomes its <strong>code</strong> from the frozen foundation encoder, for a crop masked to the nucleus by the field’s segmentation (cytology and nothing else), plus its <strong>position</strong>; two layers of self-attention let the nuclei look at each other, each with a learned cost per nucleus diameter of distance; then <strong>two attention heads</strong> over the same nuclei, one per question, each with its own scorer, weighted average and single layer. Only the field’s two labels train it. Switch a cue off and watch the mimic table: which pattern fools a model that lacks it.`;
    L.built = true;
    return true;
  }
  function fdThumb(s, onClick) {
    const L = S.fd, b = document.createElement('button'); b.type = 'button'; b.className = 'thumb field'; b.dataset.id = s.id; b.title = `${s.name} · ${fdPattern(s).name}`;
    const cv = document.createElement('canvas'); b.appendChild(cv); Viz.renderFieldThumb(cv, s.px, L.w, L.h, S.tint, s.tiles || (s.tiles = {}));
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = '✗'; b.appendChild(badge);
    b.addEventListener('click', () => onClick(s.id));
    return b;
  }
  function fdRepaint() { const L = S.fd; if (!L.built) return; for (const map of [L.thumbs, L.dataThumbs, L.testThumbs]) if (map) for (const [id, el] of map) { const s = L.byId.get(id); if (s) Viz.renderFieldThumb(el.querySelector('canvas'), s.px, L.w, L.h, S.tint, s.tiles || (s.tiles = {})); } fdRenderAll(); }
  // every nucleus of every field through the frozen encoder, for the crop mode in force, in chunks so the page keeps
  // breathing; then standardised on the training fields' nuclei
  async function fdEncode(progress) {
    const L = S.fd, enc = codeEncoder(), key = `${L.crop}|${enc.key}`;
    if (L.encKey === key) return;
    if (L.encoding && L.encoding.key === key) return L.encoding.promise;
    const run = (async () => {
      const t0 = performance.now(); let done = 0;
      for (let i = 0; i < L.all.length; i++) {
        const s = L.all[i];
        if (!s.codes[key]) {
          const crops = s.nuclei.map((n, j) => (L.crop === 'nucleus' ? NFL.cropMasked(s.px, s.seg, L.w, L.h, n.x, n.y, L.size, j, s.grainSeed) : NFL.crop(s.px, L.w, L.h, n.x, n.y, L.size)));
          s.codes[key] = crops.map(c => { const ink = new Float32Array(c.length); for (let q = 0; q < c.length; q++) ink[q] = 1 - c[q] / 255; return enc.encode(ink); });
          s.crops[L.crop] = crops; // the crops as the model saw them, for the diagrams
        }
        done += s.nuclei.length;
        if (i % 12 === 11) { progress(`Encoding the nuclei${L.crop === 'nucleus' ? ', each masked to itself' : ', each with its surroundings'}: ${done.toLocaleString()} of ${L.nNuclei.toLocaleString()}…`); await fdYield(); }
        if (L.encoding && L.encoding.key !== key) throw new Error('superseded'); // the crop or the encoder changed meanwhile
      }
      L.std = NN.fitStandardizer([].concat(...L.train.map(s => s.codes[key])), { perDimScale: true });
      for (const s of L.all) { s.Hcode = s.codes[key].map(c => L.std.apply(c)); s.nuclei.forEach((n, j) => { n.px = s.crops[L.crop][j]; }); }
      L.encKey = key; L.encDescribe = enc.describe; L.codeD = enc.cl.code;
      progress(`Encoded ${L.nNuclei.toLocaleString()} nuclei in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
    })();
    L.encoding = { key, promise: run };
    try { await run; } finally { if (L.encoding && L.encoding.key === key) L.encoding = null; }
  }
  function fdTokens() { // the tokens the model sees: the standardised code, with the position when the model is given one
    const L = S.fd; if (L.tokKey === `${L.encKey}|${L.pos}`) return;
    for (const s of L.all) s.H = L.pos ? s.Hcode.map((c, i) => Float64Array.from([...c, s.xy[i][0], s.xy[i][1]])) : s.Hcode;
    L.D = L.codeD + (L.pos ? 2 : 0); L.tokKey = `${L.encKey}|${L.pos}`;
  }
  // everything the question needs, once: the file, the fields, the codes; the callers render when it resolves
  function fdReady() {
    const L = S.fd; if (L.built && L.encKey === `${L.crop}|${codeEncoder().key}`) return Promise.resolve(true);
    if (L.readying) return L.readying;
    const progress = msg => { L.progress = msg; for (const id of ['fd-note', 'fd-data-note', 'fd-test-note']) { const el = $(id); if (el) el.textContent = msg; } };
    L.readying = (async () => {
      try {
        if (!fdData()) progress('Loading the fields (5.7 MB)…');
        await fdBuild(progress); await fdEncode(progress);
        progress(''); // the notes are the panels' own again
        return true;
      } catch (e) { if (e.message !== 'superseded') progress(`The fields did not load: ${e.message}`); return false; }
      finally { L.readying = null; }
    })();
    return L.readying;
  }
  function fdModelConfig() { const L = S.fd; return { inputSize: L.D, attentionUnits: L.units, outputs: 2, clip: L.clip, context: L.ctx ? { dk: FD.dk, ffn: FD.ffn, heads: FD.heads, layers: FD.layers, distanceBias: true, excludeSelf: true, costInit: FD.costInit } : null, seed: L.seed }; }
  function fdReset(reason) {
    const L = S.fd; fdStop();
    if (!L.built || L.encKey !== `${L.crop}|${codeEncoder().key}`) { fdReady().then(ok => { if (ok && S.world === 'fields') fdReset(reason); }); return; }
    fdTokens();
    L.model = new NN.AttentionMIL(fdModelConfig());
    L.rng = NN.mulberry32(L.seed * 31 + 7); L.order = L.train.map((_, i) => i);
    L.epoch = 0; L.ptr = 0; L.debt = 0; L.hist = []; L.lastField = null; L.hoverNucleus = null; L.pinned = null; L.frozen = null; L.hoverAtt = null; L.hoverDecide = null; L.hoverUnrolled = null;
    L.trial = { next: 0, results: new Map() }; L.testSelected = null; L.modelKey = fdModelKey();
    fdRecordEpoch();
    fdSyncControls(); fdRenderAll();
    if (reason) fdNote(reason);
  }
  function fdModelKey() { const L = S.fd; return `${L.encKey}|${L.pos}|${L.ctx}`; }
  function fdRecordEpoch() {
    const L = S.fd; L.trainEval = L.model.evaluate(L.train); L.testEval = L.model.evaluate(L.test);
    const tr = L.trainEval.outputs, te = L.testEval.outputs;
    L.hist.push({ epoch: L.epoch, loss: tr[0].loss + tr[1].loss, testLoss: te[0].loss + te[1].loss, acc: tr[1].accuracy, testAcc: te[1].accuracy, accCis: tr[0].accuracy, testAccCis: te[0].accuracy, mass: tr[1].culpritMass, testMass: te[1].culpritMass, massCis: tr[0].culpritMass, testMassCis: te[0].culpritMass });
  }
  // a training field seen in a mirror half the time, its positions jittered by about a pixel (only with positions)
  function fdAugmented(sl) {
    const L = S.fd, D = L.D, rng = L.rng, flip = rng() < 0.5 ? -1 : 1, g = () => NFL.gaussianFrom(rng);
    const H = sl.H.map(h => { const o = Float64Array.from(h); o[D - 2] = flip * h[D - 2] + FD.jitter * g(); o[D - 1] = h[D - 1] + FD.jitter * g(); return o; });
    return Object.assign({}, sl, { H });
  }
  function fdStep() { // one gradient step on the next training field of the epoch's order
    const L = S.fd, n = L.train.length;
    if (L.ptr === 0) { const o = L.order; for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(L.rng() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } }
    const i = L.order[L.ptr], s = L.train[i];
    L.model.trainBatch([L.pos ? fdAugmented(s) : s], L.lr, L.decay);
    L.lastField = s.id; L.ptr++;
    let ended = false;
    if (L.ptr >= n) { L.ptr = 0; L.epoch++; fdRecordEpoch(); ended = true; }
    return ended;
  }
  function fdStart() {
    const L = S.fd; if (!L.model) return;
    if (L.epoch >= L.epochs) { fdNote(`Already at ${L.epochs} epochs. Raise the epoch count, or reset to train again.`); return; }
    L.running = true; L.lastTime = performance.now(); L.debt = 0; L.lastRender = 0; fdSyncButtons();
    requestAnimationFrame(fdTick);
  }
  function fdStop(msg) { const L = S.fd; L.running = false; if (L.walk) fdWalkStop(); if (L.dwalk) fdDecideWalkStop(); fdSyncButtons(); if (msg) fdNote(msg); }
  function fdFinish() {
    const L = S.fd, last = L.hist[L.hist.length - 1], rates = fdMimicRates(L.testEval);
    fdStop(`Finished ${L.epochs} epochs on the fields' two labels alone. On the test fields: CIS right ${pct(last.testAccCis)}, invasion right ${pct(last.testAcc)}; ${pct(rates.inv[1])} of the invasive fields found, ${pct(rates.cisvbn[1])} of the CIS-into-nests fields called invasive. The invasion head puts ${pct(last.testMass)} of its attention on the atypical nuclei below the membrane of an invasive test field (uniform: ${pct(L.share[1])}).`);
  }
  function fdTick(now) {
    const L = S.fd; if (!L.running) return;
    const dt = Math.min(0.1, (now - L.lastTime) / 1000); L.lastTime = now;
    const spe = L.train.length;
    L.debt += dt * L.speed * spe;
    const t0 = performance.now(); let did = false, ended = false;
    while (L.debt >= 1 && performance.now() - t0 < 24) {
      ended = fdStep() || ended; L.debt -= 1; did = true;
      if (L.ptr === 0 && L.epoch >= L.epochs) { fdFinish(); break; }
      if (ended) break; // an epoch's evaluation is work enough for one frame
    }
    if (L.debt > spe) L.debt = spe;
    if (did && (ended || now - L.lastRender >= 120)) { try { fdRender(ended); } catch (e) { console.error(e); } L.lastRender = now; }
    if (L.running) requestAnimationFrame(fdTick);
  }
  function fdStepField() { const L = S.fd; if (!L.model) return; fdStop(); const ended = fdStep(); fdRender(true); const s = L.byId.get(L.lastField); fdNote(ended ? `Epoch ${L.epoch} complete.` : `One gradient step on field ${s.name} (${fdPattern(s).name}): step ${L.ptr} of ${L.train.length}.`); }
  function fdStepEpoch() { const L = S.fd; if (!L.model) return; fdStop(); do { fdStep(); } while (L.ptr !== 0); fdRender(true); fdNote(`Epoch ${L.epoch} complete.`); }
  function fdSelect(id) { const L = S.fd; L.selected = id === L.selected ? null : id; L.hoverNucleus = null; L.pinned = null; fdRenderFocus(); fdRenderTrays(); }
  const fdOnTest = () => S.world === 'fields' && S.stage === 'test';
  function fdTestResult() { const L = S.fd; return L.testSelected != null ? L.trial.results.get(L.testSelected) : null; } // the call already made for the field under test, if any
  function fdFocus() { const L = S.fd; if (fdOnTest()) return (L.testSelected != null && L.byId.get(L.testSelected)) || L.test[Math.min(L.trial.next, L.test.length - 1)]; return (L.selected != null && L.byId.get(L.selected)) || (L.frozen != null && L.byId.get(L.frozen)) || (L.lastField != null && L.byId.get(L.lastField)) || L.train[0]; }
  // the diagram cards live in the Train step and move to the Test step with the field under test (once it is classified)
  function fdPlaceCards() {
    const cards = $('fd-cards'), onTest = fdOnTest(), before = $(onTest ? 'fd-test-mimic-card' : 'fd-mimic-card');
    if (cards.nextElementSibling !== before) before.parentNode.insertBefore(cards, before);
    cards.hidden = onTest && !fdTestResult();
  }
  // while training runs, the field on screen is the last one trained on and changes with every step; a cursor over a
  // viewer or a diagram keeps the field it found there until it leaves, so what it points at stays put
  function fdFreeze(on) { const L = S.fd; if (on) { if (L.frozen == null && L.selected == null && L.built) L.frozen = fdFocus().id; } else if (L.frozen != null) { L.frozen = null; if (L.model && L.running) fdRenderFocus(); } }
  function fdNeighbour(step) { const L = S.fd, cur = fdFocus(), i = L.all.indexOf(cur); fdSelect(L.all[(i + step + L.all.length) % L.all.length].id); }
  function fdSyncButtons() { const L = S.fd; $('fd-train').textContent = L.running ? '⏸ Pause' : L.epoch > 0 ? '▶ Continue' : '▶ Train'; }
  function fdSyncControls() {
    const L = S.fd;
    document.querySelectorAll('#fd-pos-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.pos === '1') === L.pos));
    document.querySelectorAll('#fd-ctx-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.ctx === '1') === L.ctx));
    document.querySelectorAll('#fd-crop-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.crop === L.crop));
    document.querySelectorAll('#fd-links-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.links === '1') === L.linksAll));
    document.querySelectorAll('#fd-layer-seg button, #fd-ctx-layer-seg button').forEach(b => b.classList.toggle('is-active', +b.dataset.layer === L.layer));
    document.querySelectorAll('#fd-head-seg button').forEach(b => b.classList.toggle('is-active', +b.dataset.head === L.head));
    document.querySelectorAll('#fd-attview-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.view === L.attView));
    $('fd-context-card').hidden = !L.ctx; $('fd-decide-card').hidden = !L.ctx;
    $('fd-links-ctl').hidden = !L.ctx; $('fd-links-min').value = Math.round(L.linksMin * 100); $('fd-links-min-val').textContent = pct(L.linksMin); $('fd-links-min').disabled = !L.linksAll;
    $('fd-epochs').value = L.epochs; $('fd-epochs-val').textContent = L.epochs;
    $('fd-speed').value = sliderFromSpeed(L.speed); $('fd-speed-val').textContent = `${L.speed} epochs/s`;
    $('fd-seed').value = L.seed; for (const id of ['fd-reveal', 'fd-data-reveal', 'fd-test-reveal']) $(id).checked = L.reveal;
    $('fd-backbone').value = S.backbone;
    fdSyncButtons();
  }
  // ---- rendering
  const fdModelWord = () => { const L = S.fd; return L.pos && L.ctx ? 'positions and context' : L.pos ? 'codes with positions, no context' : L.ctx ? 'context without positions' : 'a bag of codes: no positions, no context'; };
  function fdRenderStatus() {
    const L = S.fd, last = L.hist[L.hist.length - 1], m = L.model;
    $('fd-status').innerHTML =
      `<span>architecture <b>every nucleus → code (${L.codeD} numbers, from ${esc(L.encDescribe)})${L.pos ? ' + its position' : ''} → ${esc(m.describe())}</b></span>` +
      `<span>parameters <b>${m.parameterCount()}</b></span><span>model <b>${fdModelWord()}</b> · crops <b>${L.crop === 'nucleus' ? 'masked to the nucleus' : 'with the surroundings'}</b></span>` +
      `<span>fields <b>${L.train.length}</b> training · <b>${L.test.length}</b> test</span><span>epoch <b>${L.epoch}</b> / ${L.epochs}</span><span>field <b>${L.ptr === 0 ? '–' : L.ptr}</b> / ${L.train.length}</span>` +
      `<span>loss <b>${last.loss.toFixed(3)}</b></span><span>CIS right <b>${pct(last.accCis)}</b> training · <b>${pct(last.testAccCis)}</b> test</span><span>invasion right <b>${pct(last.acc)}</b> training · <b>${pct(last.testAcc)}</b> test (peeking)</span>` +
      `<span>attention on the culprits <b>${pct(last.massCis)}</b> CIS head · <b>${pct(last.mass)}</b> invasion head (test fields; uniform: ${pct(L.share[0])} · ${pct(L.share[1])})</span>`;
  }
  function fdFocusForward() { const L = S.fd, s = fdFocus(), r = fdOnTest() ? fdTestResult() : null; return { s, fw: r ? r.fw : L.model.forward(s.H, s.dist, s.xy) }; } // on the Test step, the call as it was made
  function fdShown(fw, k) { const L = S.fd, a = fw.outs[k].a; if (L.hoverNucleus != null && L.hoverNucleus < a.length) return L.hoverNucleus; if (L.pinned != null && L.pinned < a.length) return L.pinned; let j = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[j]) j = i; return j; }
  function fdContextView(fw) { const L = S.fd; return L.ctx && fw && fw.ctxs && fw.ctxs.length ? fdAttentionView(fw) : null; }
  function fdCallText(s, o, k) { const yes = k ? 'invasive' : 'CIS', no = k ? 'not invasive' : 'no CIS', call = o.p >= 0.5 ? 1 : 0, truth = s.y[k]; return `The ${FD.names[k]} head says <b>P(${yes}) = ${o.p.toFixed(2)}</b> → <span class="${call === truth ? 'good-text' : 'bad-text'}">${call ? yes : no} ${call === truth ? '✓' : '✗'}</span> · truth: ${truth ? yes : no}.`; }
  function fdDrawPair(ids, s, fw) { // the two viewers of one field, one per head
    const L = S.fd, n = s.nuclei.length, view = fdContextView(fw), ok = i => (i != null && i >= 0 && i < n ? i : null), hover = ok(L.hoverNucleus != null ? L.hoverNucleus : L.walkNucleus), pinned = ok(L.pinned), from = ok(L.hoverNucleus != null ? L.hoverNucleus : L.dwalk ? L.dwalk.i : L.pinned);
    ids.forEach((id, k) => Viz.drawField($(id), { px: s.px, w: L.w, h: L.h, scale: 2, tint: S.tint, nuclei: s.nuclei.map((q, i) => ({ x: q.x, y: q.y, a: fw ? fw.outs[k].a[i] : 0, pos: s.pos[k][i] })), reveal: L.reveal, membrane: s.membrane, nests: s.nests, plain: !fw,
      hover, pinned: view ? pinned : null, links: view && !L.linksAll && from != null ? { from, weights: view[from] } : null, allLinks: view && L.linksAll ? { A: view, min: L.linksMin, hover: from } : null }));
  }
  function fdRenderField() {
    const L = S.fd;
    if (fdOnTest()) { const r = fdTestResult(), s = fdFocus(); fdDrawPair(['fd-test-canvas-cis', 'fd-test-canvas-inv'], s, r ? r.fw : null); return; } // the viewers of the Test step: the call as it was made, or the next field plain
    const { s, fw } = fdFocusForward(), P = fdPattern(s);
    fdDrawPair(['fd-canvas-cis', 'fd-canvas-inv'], s, fw);
    $('fd-field-title').textContent = `Field ${s.name} · ${s.split === 'train' ? 'training' : 'test'} · ${P.name}`;
    $('fd-call-cis').innerHTML = fdCallText(s, fw.outs[0], 0); $('fd-call-inv').innerHTML = fdCallText(s, fw.outs[1], 1);
    $('fd-field-note').textContent = `Rings are the attention weights, thicker and stronger with the weight; the two heads read the same nuclei and weigh them differently. ${L.reveal ? `Dots mark the nuclei each head is judged against: the atypical nuclei for the CIS head, the atypical nuclei below the membrane for the invasion head; the blue line is the basement membrane and the green outlines the nests.` : 'Tick Reveal to see the membrane, the nests and the atypical nuclei.'} Hover a nucleus for its two weights${L.ctx ? ' and whom it listens to' : ''}${L.selected == null && L.lastField != null ? ' · the last field trained on' : ''}.`;
    if (L.ctx) $('fd-links-note').textContent = (L.linksAll ? `Every link of the ${L.layer ? 'second' : 'first'} context layer above ${pct(L.linksMin)}: a link is wide at the end that listens. Hover a nucleus to lift its links out; click it to keep it.` : `Lines from the hovered nucleus to the nuclei it listens to in the ${L.layer ? 'second' : 'first'} context layer, thicker with the share. Click a nucleus to keep it as the one the lines follow.`) + (L.model.layers.length ? ` The learned distance cost is ${L.model.layers.map((c, l) => `${Math.log1p(Math.exp(c.beta[0])).toFixed(2)} in layer ${l + 1}`).join(', ')} per nucleus diameter: every extra diameter divides a nucleus’s weight by ${Math.exp(Math.log1p(Math.exp(L.model.layers[Math.min(L.layer, L.model.layers.length - 1)].beta[0]))).toFixed(1)}.` : '');
  }
  function fdNucleusTip(s, fw, i) { const L = S.fd, n = s.nuclei[i]; if (!n) return ''; const where = n.kind === 'stroma' ? 'stromal cell' : n.below ? 'below the membrane' : 'in the epithelium'; return `nucleus ${i + 1} · ${where} · CIS head ${pct(fw.outs[0].a[i])} · invasion head ${pct(fw.outs[1].a[i])}${L.reveal ? ` · ${n.kind === 'stroma' ? 'stroma' : n.atypical ? `atypical (${subtypeWord(n.subtype)})` : 'bland'}` : ''}`; }
  // the mimic table: the test fields of each pattern, how often called CIS and called invasive
  function fdMimicRates(ev, ids) {
    const L = S.fd, out = {};
    for (const P of L.meta.patterns) {
      const idx = L.test.map((s, i) => (s.pattern === P.key && (!ids || ids.has(s.id)) ? i : -1)).filter(i => i >= 0);
      out[P.key] = [0, 1].map(k => (idx.length ? idx.filter(i => ev.outputs[k].probs[i] >= 0.5).length / idx.length : null)); out[P.key].n = idx.length;
    }
    return out;
  }
  function fdRenderMimic(el, ev, ids, focusKey) {
    const L = S.fd, rates = fdMimicRates(ev, ids), cell = (r, truth) => (r == null ? '<td class="num">–</td>' : `<td class="num ${Math.round(r * 100) === (truth ? 100 : 0) ? 'good' : (truth ? r < 0.5 : r >= 0.5) ? 'bad' : ''}">${pct(r)}</td>`);
    el.innerHTML = `<table><thead><tr><th>pattern</th><th>cues</th><th class="num">fields</th><th class="num">called CIS</th><th class="num">called invasive</th><th>truth</th></tr></thead><tbody>` +
      L.meta.patterns.map(P => { const r = rates[P.key], cis = FD.cisPatterns.includes(P.key) ? 1 : 0; return `<tr${P.key === focusKey ? ' class="current"' : ''}><td>${esc(P.name)}</td><td>${esc(P.cues)}</td><td class="num">${r.n}</td>${cell(r[0], cis)}${cell(r[1], P.label)}<td class="truth">CIS ${cis ? 'yes' : 'no'} · invasive ${P.label ? 'yes' : 'no'}</td></tr>`; }).join('') + '</tbody></table>';
  }
  function fdRenderMimicCard() {
    const L = S.fd; fdRenderMimic($('fd-mimic'), L.testEval, null, fdFocus().pattern);
    const r = fdMimicRates(L.testEval);
    $('fd-mimic-note').textContent = `Every model gets the top two rows right by cytology alone. The last three rows are the point: ${r.cisvbn[1] == null ? '' : `${pct(r.cisvbn[1])} of the CIS-into-von-Brunn-nests fields are called invasive, atypical cells below the membrane in the one arrangement that is not invasion, and ${pct(r.inv[1])} of the invasive fields are found. `}A model without positions cannot tell above from below; one without context cannot tell a round nest from an angulated one; the full model reads all three cues.`;
  }
  function fdRenderCurves() {
    const L = S.fd, c = Viz.colors(), testCol = getComputedStyle(document.documentElement).getPropertyValue('--test-series').trim(), last = L.hist[L.hist.length - 1];
    Viz.drawCurves($('fd-loss'), { history: L.hist, key: 'loss', showTest: true, maxEpoch: L.epochs });
    Viz.drawCurves($('fd-acc'), { history: L.hist, key: 'acc', showTest: true, maxEpoch: L.epochs });
    Viz.drawSeries($('fd-mass'), { hist: L.hist, keys: [{ key: 'mass', color: c.accent }, { key: 'testMass', color: testCol, dash: true }], maxEpoch: L.epochs, pct: true, baseline: L.share[1], baselineLabel: 'uniform' });
    $('fd-loss-now').textContent = `${last.loss.toFixed(3)} · test ${last.testLoss.toFixed(3)}`; $('fd-acc-now').textContent = `${pct(last.acc)} · test ${pct(last.testAcc)}`; $('fd-mass-now').textContent = `${pct(last.mass)} · test ${pct(last.testMass)}`;
    $('fd-curves-note').textContent = `The loss is the sum of the two questions’ cross-entropies. CIS is right on ${pct(last.testAccCis)} of the test fields (the codes alone carry cytology); invasion is the hard one. The test curves peek at the held-out fields as the model trains, which is how one sees it overfit: 400 fields are few, and the test loss turns up after some 40 epochs while the training loss keeps falling.`;
  }
  function fdRenderTrays() {
    const L = S.fd, focus = fdFocus();
    [['train', L.train, L.trainEval], ['test', L.test, L.testEval]].forEach(([split, fields, ev]) => fields.forEach((s, k) => {
      const el = L.thumbs.get(s.id), pc = ev.outputs[0].probs[k], pi = ev.outputs[1].probs[k], callC = pc >= 0.5 ? 1 : 0, callI = pi >= 0.5 ? 1 : 0, wrong = callC !== s.y[0] || callI !== s.y[1];
      el.className = `thumb field call-${callI}${wrong ? ' wrong' : ''}${focus === s ? ' selected' : ''}${L.lastField === s.id && split === 'train' ? ' in-batch' : ''}`;
      el.title = `${s.name} · ${split === 'train' ? 'training' : 'test'} · ${fdPattern(s).name} · CIS ${callC ? 'yes' : 'no'} (P ${pc.toFixed(2)}) ${callC === s.y[0] ? '✓' : '✗'} · invasive ${callI ? 'yes' : 'no'} (P ${pi.toFixed(2)}) ${callI === s.y[1] ? '✓' : '✗'}`;
    }));
  }
  function fdRenderCards() { fdRenderUnrolled(); fdRenderScorer(); fdRenderSummary(); fdRenderContext(); fdRenderDecide(); }
  function fdRenderFocus(light) { fdRenderField(); if (fdOnTest()) { if (fdTestResult()) fdRenderCards(); return; } fdRenderUnrolled(); if (light) return; fdRenderScorer(); fdRenderSummary(); fdRenderContext(); fdRenderDecide(); }
  function fdRender(full) {
    const L = S.fd; if (!L.model) return;
    fdRenderStatus(); fdRenderFocus(!full && L.running);
    if (full) { fdRenderMimicCard(); fdRenderCurves(); fdRenderTrays(); }
    fdSyncButtons();
  }
  function fdRenderAll() { const L = S.fd; if (S.world !== 'fields' || !L.built) return; if (S.stage === 'data') fdRenderData(); else if (S.stage === 'test') fdRenderTest(); else if (L.model) fdRender(true); }
  function fdEnter() {
    const L = S.fd;
    fdReady().then(ok => {
      if (!ok || S.world !== 'fields' || S.stage !== 'train') return;
      fdPlaceCards();
      if (!L.model || L.modelKey !== fdModelKey()) fdReset(L.model ? 'The input changed — fresh random weights.' : 'Untrained: the attention weights are near uniform and both calls are guesses. Step a field to watch one gradient step, or press Train and watch the mimic table and the curves.');
      else { fdTokens(); fdRender(true); }
    });
    if (!L.built) fdNote(L.progress || 'Loading the fields…');
  }
  // ---- the other two steps: Specimens (the fields with their patterns and labels) and Test (the held-out fields, one at a time)
  function fdSetReveal(v) { S.fd.reveal = v; for (const id of ['fd-reveal', 'fd-data-reveal', 'fd-test-reveal']) $(id).checked = v; fdRenderAll(); }
  function fdTrays(kind, onClick) {
    const L = S.fd; if (L.trayFor[kind]) return; L.trayFor[kind] = true;
    const map = new Map(), fill = (id, tray) => { const el = $(id); el.innerHTML = ''; for (const s of tray) { const b = fdThumb(s, onClick); el.appendChild(b); map.set(s.id, b); } };
    if (kind === 'data') { fill('fd-data-train-tray', L.train); fill('fd-data-test-tray', L.test); L.dataThumbs = map; $('fd-data-train-label').textContent = `Training fields (${L.train.length})`; $('fd-data-test-label').textContent = `Test fields (${L.test.length}) · held out`; }
    else { fill('fd-test-results', L.test); L.testThumbs = map; $('fd-test-count').textContent = `${L.test.length} fields`; }
  }
  function fdEnterData() { const L = S.fd; fdReady().then(ok => { if (ok && S.world === 'fields' && S.stage === 'data') fdRenderData(); }); if (!L.built) $('fd-data-note').textContent = L.progress || 'Loading the fields…'; }
  function fdDataSelect(id) { S.fd.dataSelected = id; fdRenderData(); }
  function fdDataNeighbour(step) { const L = S.fd, i = L.all.findIndex(s => s.id === L.dataSelected); fdDataSelect(L.all[(i + step + L.all.length) % L.all.length].id); }
  function fdRenderData() {
    const L = S.fd; if (!L.built) return; fdTrays('data', fdDataSelect); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id;
    const s = L.byId.get(L.dataSelected) || L.train[0], P = fdPattern(s), meta = L.meta, nBelow = s.nuclei.filter(n => n.atypical && n.below).length, nAt = s.nuclei.filter(n => n.atypical).length;
    $('fd-data-title').textContent = `Fields of bladder: ${meta.question.toLowerCase().replace('?', '')}?`; $('fd-data-sub').textContent = `${L.train.length} training and ${L.test.length} test fields of ${meta.w} × ${meta.h} px · five patterns · two labels per field`;
    $('fd-data-blurb').textContent = `${meta.blurb} Every nucleus of a field, the spindle cells of the stroma included, is a token for the model; nobody labels the nuclei, and the model never sees the membrane, the nests or which nuclei are atypical, only the field’s two labels: carcinoma in situ, yes or no; invasive, yes or no.`;
    $('fd-data-note').textContent = '';
    fdRenderDataCanvas();
    $('fd-data-field-title').textContent = `Field ${s.name} · ${s.split === 'train' ? 'training' : 'test'} · ${P.name}`;
    $('fd-data-caption').innerHTML = `<b>${esc(P.name)}</b> · ${esc(P.cues)}. ${esc(P.blurb)} Labels: <b>CIS ${s.y[0] ? 'yes' : 'no'}</b> · <b>invasive ${s.y[1] ? 'yes' : 'no'}</b>. ${s.nuclei.length} nuclei${L.reveal ? `, ${nAt} atypical, ${nBelow} of them below the membrane` : ''}. Click a field below, or step through them.`;
    $('fd-patterns').innerHTML = `<table><thead><tr><th>pattern</th><th>cytology</th><th>location</th><th>architecture</th><th>CIS</th><th>invasive</th></tr></thead><tbody>` + meta.patterns.map(Q => { const cues = Q.cues.split(' · '); return `<tr${Q.key === s.pattern ? ' class="current"' : ''}><td>${esc(Q.name)}</td><td>${esc(cues[0])}</td><td>${esc(cues[1] || '')}</td><td>${esc(cues[2] || '')}</td><td>${FD.cisPatterns.includes(Q.key) ? 'yes' : 'no'}</td><td>${Q.label ? '<b>yes</b>' : 'no'}</td></tr>`; }).join('') + '</tbody></table>';
    for (const [id, el] of L.dataThumbs) { const f = L.byId.get(id); el.classList.toggle('selected', id === L.dataSelected); el.classList.toggle('call-1', !!f.label); el.classList.toggle('call-0', !f.label); }
  }
  function fdRenderDataCanvas() { const L = S.fd, s = L.byId.get(L.dataSelected) || L.train[0]; Viz.drawField($('fd-data-canvas'), { px: s.px, w: L.w, h: L.h, scale: 3, tint: S.tint, nuclei: s.nuclei.map(n => ({ x: n.x, y: n.y, a: 0, pos: n.atypical })), reveal: L.reveal, membrane: s.membrane, nests: s.nests, plain: true, hover: L.dataHover }); }
  function fdEnterTest() { const L = S.fd; fdReady().then(ok => { if (!ok || S.world !== 'fields' || S.stage !== 'test') return; if (!L.model || L.modelKey !== fdModelKey()) fdReset(); else fdTokens(); fdTrays('test', fdTestSelect); fdRenderTest(); }); if (!L.built) $('fd-test-note').textContent = L.progress || 'Loading the fields…'; }
  function fdTestClear() { const L = S.fd; L.trial = { next: 0, results: new Map() }; L.testSelected = null; fdRenderTest(); }
  function fdClassifyNext(quiet) {
    const L = S.fd, T = L.trial; if (!L.model || T.next >= L.test.length) return false;
    const s = L.test[T.next++], fw = L.model.forward(s.H, s.dist, s.xy);
    T.results.set(s.id, { p: fw.outs.map(o => o.p), call: fw.outs.map(o => (o.p >= 0.5 ? 1 : 0)), a: fw.outs.map(o => o.a), fw }); L.testSelected = s.id;
    if (!quiet) fdRenderTest();
    return true;
  }
  function fdClassifyAll() { const L = S.fd; if (!L.model) return; const go = () => { if (fdClassifyNext(false) && S.world === 'fields' && S.stage === 'test') setTimeout(go, reducedMotion ? 0 : 120); }; go(); }
  function fdTestSelect(id) { const L = S.fd; if (!L.trial.results.has(id)) return; L.testSelected = id; fdRenderTest(); }
  function fdTestStats() {
    const L = S.fd; let n = 0, cisRight = 0, invRight = 0, tp = 0, nInv = 0, mimicClear = 0, nMimic = 0;
    for (const [id, r] of L.trial.results) { const s = L.byId.get(id); n++; if (r.call[0] === s.y[0]) cisRight++; if (r.call[1] === s.y[1]) invRight++; if (s.y[1]) { nInv++; if (r.call[1]) tp++; } if (s.pattern === 'cisvbn') { nMimic++; if (!r.call[1]) mimicClear++; } }
    return { n, cisRight, invRight, tp, nInv, mimicClear, nMimic };
  }
  function fdRenderTest() {
    const L = S.fd; if (!L.built || !L.model) return; fdTrays('test', fdTestSelect);
    const T = L.trial, st = fdTestStats();
    $('fd-stat-n').textContent = `${st.n} / ${L.test.length}`;
    $('fd-stat-cis').textContent = st.n ? pct(st.cisRight / st.n) : '–'; $('fd-stat-cis-sub').textContent = st.n ? `${st.cisRight} of ${st.n} right` : 'no calls yet';
    $('fd-stat-inv').textContent = st.n ? pct(st.invRight / st.n) : '–'; $('fd-stat-inv-sub').textContent = st.n ? `${st.invRight} of ${st.n} right` : 'no calls yet';
    $('fd-stat-sens').textContent = st.nInv ? pct(st.tp / st.nInv) : '–'; $('fd-stat-sens-sub').textContent = st.nInv ? `${st.tp} of ${st.nInv} invasive fields` : 'none seen yet';
    $('fd-stat-mimic').textContent = st.nMimic ? pct(st.mimicClear / st.nMimic) : '–'; $('fd-stat-mimic-sub').textContent = st.nMimic ? `${st.mimicClear} of ${st.nMimic} CIS-into-nests fields not called invasive` : 'no CIS-into-nests field yet';
    const done = T.next >= L.test.length;
    $('fd-classify-next').disabled = done; $('fd-classify-all').disabled = done;
    $('fd-classify-next').textContent = done ? `All ${L.test.length} classified` : `Classify next field (${T.next + 1} of ${L.test.length})`;
    $('fd-classify-all').textContent = `Classify all ${L.test.length}`;
    $('fd-test-warning').hidden = !(L.model && L.model.steps === 0);
    const ev = { outputs: [0, 1].map(k => ({ probs: L.test.map(s => { const r = T.results.get(s.id); return r ? r.p[k] : null; }) })) };
    fdRenderMimic($('fd-test-mimic'), ev, new Set(T.results.keys()), L.testSelected != null ? L.byId.get(L.testSelected).pattern : null);
    const s = L.testSelected != null ? L.byId.get(L.testSelected) : null, r = s ? T.results.get(s.id) : null;
    fdPlaceCards();
    if (s && r) {
      fdDrawPair(['fd-test-canvas-cis', 'fd-test-canvas-inv'], s, r.fw);
      fdRenderCards();
      $('fd-test-field-title').textContent = `Field ${s.name} · test · ${fdPattern(s).name}`;
      $('fd-test-call-cis').innerHTML = fdCallText(s, r.fw.outs[0], 0); $('fd-test-call-inv').innerHTML = fdCallText(s, r.fw.outs[1], 1);
      $('fd-test-field-note').textContent = `${fdPattern(s).blurb} ${L.reveal ? 'The dots mark the nuclei each head is judged against; a call for the right reason lands the attention on them.' : 'Tick Reveal to see the membrane, the nests and the atypical nuclei.'} Hover a nucleus for its two weights; the cards below take this call apart, as they did while training.`;
    } else {
      const s0 = L.test[Math.min(T.next, L.test.length - 1)];
      fdDrawPair(['fd-test-canvas-cis', 'fd-test-canvas-inv'], s0, null);
      $('fd-test-field-title').textContent = `Field ${s0.name} · test · next up`;
      $('fd-test-call-cis').textContent = 'Press Classify next field: the model reads every nucleus and makes both calls, and the truth is shown after.'; $('fd-test-call-inv').textContent = '';
      $('fd-test-field-note').textContent = '';
    }
    for (const [id, el] of L.testThumbs) { const rr = T.results.get(id), f = L.byId.get(id), right = !!rr && rr.call[0] === f.y[0] && rr.call[1] === f.y[1]; el.classList.toggle('selected', id === L.testSelected); el.classList.toggle('call-0', !!rr && rr.call[1] === 0); el.classList.toggle('call-1', !!rr && rr.call[1] === 1); el.classList.toggle('wrong', !!rr && !right); el.classList.toggle('right', right); el.querySelector('.badge').textContent = right ? '✓' : '✗'; }
  }
  // ---- the model diagrams: the whole model for one head, unrolled; that head's scorer and single layer; the context
  // layers' attention map with the hovered nucleus's token before and after; one nucleus's decision taken apart
  const fdHeadName = k => (k ? 'invasion' : 'CIS');
  function fdShownWord() { const L = S.fd; return L.hoverNucleus != null ? '' : L.pinned != null ? ', pinned,' : `, the one with the most of the ${fdHeadName(L.head)} head’s attention,`; }
  const fdBars = code => { let mx = 1e-9; for (const v of code) mx = Math.max(mx, Math.abs(v)); return `<span class="code" title="${Array.from(code).map(v => v.toFixed(2)).join(', ')}">${Array.from(code).map(v => `<i class="${v < 0 ? 'neg' : 'pos'}" style="height:${Math.max(2, Math.round(Math.abs(v) / mx * 100))}%"></i>`).join('')}</span>`; };
  function fdRenderUnrolled() {
    const L = S.fd, { s, fw } = fdFocusForward(), k = L.head, o = fw.outs[k], n = s.nuclei.length, yes = k ? 'invasive' : 'CIS', no = k ? 'not invasive' : 'no CIS';
    const m = { setName: 'THE FIELD', nuclei: s.nuclei.map((q, i) => ({ px: q.px, h: fw.T[i], s: o.s[i], a: o.a[i], pos: s.pos[k][i] })), D: L.D, tokenNote: `${L.pos ? 'code + position' : 'code'}${L.ctx ? ', after context' : ''}`, size: L.size, tint: S.tint, reveal: L.reveal,
      scorer: L.model.scorers[k], head: L.model.heads[k], fw: { s: o.s, a: o.a, z: o.z, head: o.head, fws: o.fws }, shown: fdShown(fw, k), hover: L.hoverUnrolled, walk: L.walk, positiveName: yes, negativeName: no };
    const cv = $('fd-unrolled'); cv._model = m; Viz.drawSlideNetwork(cv, m);
    $('fd-unrolled-note').textContent = `The ${fdHeadName(k)} head, left to right: the ${n} nuclei of this field, each as its token of ${L.D} numbers (${m.tokenNote}); one scorer, the same weights for all of them, gives each a score; the softmax compares the ${n} scores and turns them into shares that add up to 100%; the tokens are added up, each weighted by its share, into the ${L.D}-number summary; a single layer on the summary makes the call. The other head does the same over the same tokens with its own scorer and single layer. Hover a nucleus to follow it through the model.`;
  }
  // the walk-through: the field on screen goes through one head step by step, one nucleus at a time
  function fdWalkStart() {
    const L = S.fd; if (!L.model) return; fdStop(); fdDecideWalkStop();
    L.walk = { stage: 'score', k: 0, t: 0, t0: performance.now() }; L.hoverUnrolled = null;
    fdSyncWalk(); requestAnimationFrame(fdWalkTick);
  }
  function fdWalkStop() { const L = S.fd; if (!L.walk) return; L.walk = null; L.walkNucleus = null; fdSyncWalk(); if (L.model) { fdRenderField(); fdRenderUnrolled(); } }
  function fdSyncWalk() { $('fd-walk').textContent = S.fd.walk ? '■ Stop' : '▶ Walk through'; }
  function fdWalkTick(now) {
    const L = S.fd, w = L.walk; if (!w) return;
    const el = Math.max(0, now - w.t0), n = fdFocus().nuclei.length, per = Math.min(SL_WALK.score, 2400 / n), next = stage => { w.stage = stage; w.t = 0; w.t0 = now; };
    if (w.stage === 'score') { w.k = Math.min(n, Math.floor(el / per) + 1); L.walkNucleus = w.k - 1; if (el >= n * per + SL_WALK.pause) { L.walkNucleus = null; next('softmax'); } }
    else if (w.stage === 'softmax') { w.t = Math.min(1, el / SL_WALK.softmax); if (el >= SL_WALK.softmax + SL_WALK.pause) next('sum'); }
    else if (w.stage === 'sum') { w.t = Math.min(1, el / SL_WALK.sum); if (el >= SL_WALK.sum + SL_WALK.pause) next('head'); }
    else if (w.stage === 'head') {
      w.t = Math.min(1, el / SL_WALK.head);
      if (el >= SL_WALK.head + SL_WALK.pause) {
        const { s, fw } = fdFocusForward(), k = L.head, o = fw.outs[k], call = o.p >= 0.5 ? 1 : 0, yes = k ? 'invasive' : 'CIS', no = k ? 'not invasive' : 'no CIS'; fdWalkStop();
        fdNote(`Field ${s.name} through the ${fdHeadName(k)} head: every nucleus scored by the same scorer, softmax, weighted sum, then the single layer: P(${yes}) = ${o.p.toFixed(2)} → ${call ? yes : no} ${call === s.y[k] ? '✓' : '✗'}.`);
        return;
      }
    }
    fdRenderField(); fdRenderUnrolled();
    requestAnimationFrame(fdWalkTick);
  }
  function fdScorerModel() { // the attention network of the chosen head for one nucleus: the hovered one, else the pinned one, else the one with the most attention
    const L = S.fd, { s, fw } = fdFocusForward(), k = L.head, o = fw.outs[k], j = fdShown(fw, k), n = s.nuclei[j], tok = fw.T[j], sf = o.fws ? o.fws[j] : L.model.scorers[k].forward(tok);
    return { j, n, m: { net: L.model.scorers[k], mode: 'features', inputCaption: `INPUT · THIS NUCLEUS’S TOKEN${L.ctx ? ', AFTER CONTEXT' : ''} · ${L.D} NUMBERS`, featureNames: Array.from({ length: L.D }, (_, i) => (L.pos && i >= L.codeD ? (i === L.codeD ? 'position x' : 'position y') : `code ${i + 1}`)), x: tok, fw: sf, stage: 2, hover: L.hoverScorer,
      activation: 'tanh', activationLabel: 'Tanh', positiveName: 'more attention', negativeName: 'less attention', scoreLabel: 'attention score', scoreNote: `softmax over the field\n→ ${pct(o.a[j])} of the ${fdHeadName(k)} head’s attention`, specimen: null, size: L.size, tint: S.tint } };
  }
  function fdRenderScorer() {
    const L = S.fd, { j, n, m } = fdScorerModel(), cv = $('fd-scorer'); cv._model = m; Viz.drawNetwork(cv, m);
    $('fd-scorer-title').textContent = `How a nucleus is scored by the ${fdHeadName(L.head)} head`;
    $('fd-scorer-note').textContent = `${L.hoverNucleus == null && L.pinned == null ? `The nucleus with the most of the ${fdHeadName(L.head)} head’s attention on this field` : `Nucleus ${j + 1} of this field${L.hoverNucleus == null ? ', pinned' : ''}`}${L.reveal ? ` (${n.kind === 'stroma' ? 'a stromal cell' : n.atypical ? 'atypical: ' + subtypeWord(n.subtype) : 'bland'}, ${n.kind === 'stroma' ? 'in the stroma' : n.below ? 'below the membrane' : 'in the epithelium'})` : ''}. Every nucleus of the field goes through this same little network; the softmax then compares the ${fdFocus().nuclei.length} scores, so a score only counts relative to the others. The CIS head’s scorer learns to fire on atypia wherever it sits; the invasion head’s on atypia below the membrane and, through the context, on atypia arranged like an invasive nest.`;
  }
  function fdRenderSummary() {
    const L = S.fd, { s, fw } = fdFocusForward(), k = L.head, o = fw.outs[k], D = L.D, yes = k ? 'invasive' : 'CIS', no = k ? 'not invasive' : 'no CIS';
    const plain = new Float64Array(D); for (const h of fw.T) for (let d = 0; d < D; d++) plain[d] += h[d] / fw.T.length;
    $('fd-summary-title').textContent = `The field’s summary for the ${fdHeadName(k)} head, and its call`;
    $('fd-summary').innerHTML = `<div class="v"><div class="lbl">weighted average</div>${fdBars(o.z)}<div class="n">the summary</div></div><div class="v"><div class="lbl">plain average</div>${fdBars(plain)}<div class="n">for comparison</div></div>` +
      `<div class="txt"><div>Σ weight × token over the ${s.nuclei.length} nuclei gives ${D} numbers, the field’s summary for this head; it leans towards the nuclei that weigh most, where a plain average lets the bland majority drown them.</div><div>Then a single layer on the summary: <b>P(${yes}) = ${o.p.toFixed(2)}</b>.</div></div>`;
    const cv = $('fd-head'), m = { net: L.model.heads[k], mode: 'features', inputCaption: `INPUT · THE FIELD’S SUMMARY · ${D} NUMBERS`, featureNames: Array.from({ length: D }, (_, i) => `summary ${i + 1}`), x: o.z, fw: o.head, stage: 2, hover: L.hoverHead, activation: 'relu', activationLabel: 'ReLU', positiveName: yes, negativeName: no, specimen: null, size: L.size, tint: S.tint };
    cv._model = m; Viz.drawNetwork(cv, m);
  }
  // the context: the chosen layer's attention as the map shows it (what the layer uses, or the match alone, or the distance alone)
  function fdLayerFw(fw) { const L = S.fd; return fw.ctxs[Math.min(L.layer, fw.ctxs.length - 1)]; }
  function fdLayer() { const L = S.fd; return L.model.layers[Math.min(L.layer, L.model.layers.length - 1)]; }
  function fdAttentionView(fw) { const L = S.fd, c = fdLayerFw(fw); if (L.attView === 'both') return c.A; const ex = fdLayer().explain(c); return L.attView === 'match' ? ex.matchOnly : ex.distOnly; }
  function fdRenderContext() {
    const L = S.fd; if (!L.ctx || !L.model || !L.model.layers.length) return;
    const { s, fw } = fdFocusForward(), c = fdLayerFw(fw), A = fdAttentionView(fw), n = s.nuclei.length, cost = c.cost, j = L.dwalk ? L.dwalk.i : fdShown(fw, L.head), last = L.layer + 1 >= L.model.layers.length;
    const pair = L.hoverAtt || (L.hoverDecide && L.hoverDecide.j != null ? { i: j, j: L.hoverDecide.j } : null);
    Viz.drawAttentionMap($('fd-attmap'), { A, thumbs: s.nuclei.map(q => q.px), size: L.size, tint: S.tint, hover: j, pair, pos: s.pos[0], reveal: L.reveal });
    const per = Math.exp(cost).toFixed(1), costWord = cost > 2 ? 'a nucleus listens almost only to the nuclei touching it' : cost > 0.8 ? 'near nuclei count more, but far ones still count' : 'distance hardly matters to it';
    $('fd-attmap-note').textContent = L.attView === 'match'
      ? `Layer ${L.layer + 1}, match only: the shares the softmax would give from query · key alone, as if distance cost nothing. This is the part of the decision that reads what a nucleus looks like: the nuclei whose keys fit a query get the listening, wherever they sit in the field. Switch back to see what the distance cost (${cost.toFixed(2)} per nucleus diameter) does to it.`
      : L.attView === 'distance'
        ? `Layer ${L.layer + 1}, distance only: the shares from the learned distance cost alone, as if every nucleus looked the same; each nucleus diameter of distance divides a nucleus’s weight by ${per}, so ${costWord}. This is the part of the decision that reads where a nucleus is. The layer uses both: match minus distance cost, then the softmax.`
        : `Layer ${L.layer + 1}. Each row is one nucleus asking, each column one answering: how much of its listening goes to each of the other ${n - 1} (a row adds up to 100%; a nucleus does not listen to itself, its own token stays through the residual). The learned distance cost is ${cost.toFixed(2)} per nucleus diameter: every extra diameter divides a nucleus’s weight by ${per}, so ${costWord}.`;
    const row = Array.from(A[j], (w, k) => ({ k, w })).filter(e => e.k !== j).sort((a, b) => b.w - a.w), top = row.slice(0, 3);
    const where = k => { const d = s.dist[j][k]; return d <= 1.3 ? 'touching' : `${d.toFixed(1)} diameters away`; };
    $('fd-context-focus').innerHTML = `<div class="v"><div class="lbl">nucleus ${j + 1} · token</div>${fdBars(c.X[j])}<div class="n">${L.layer ? 'into layer 2' : 'before context'}</div></div><div class="v"><div class="lbl">after layer ${L.layer + 1}</div>${fdBars(c.Y[j])}<div class="n">${last ? 'what the heads see' : 'into the next layer'}</div></div>` +
      `<div class="txt"><div>Nucleus ${j + 1}${fdShownWord()} listens most to ${top.map(e => `nucleus ${e.k + 1} (${pct(e.w)}, ${where(e.k)})`).join(', ')}${L.attView === 'both' ? '' : ` (from the ${L.attView} alone)`}.</div><div>What it hears is added to its own token, so the heads can weigh “atypical, below the membrane, packed with atypical neighbours on every side” rather than “atypical” alone.</div></div>`;
  }
  function fdRenderDecide() {
    const L = S.fd; if (!L.ctx || !L.model || !L.model.layers.length) return;
    const { s, fw } = fdFocusForward(), ctx = fdLayerFw(fw), cl = fdLayer(), ex = cl.explain(ctx), n = s.nuclei.length, i = L.dwalk ? L.dwalk.i : fdShown(fw, L.head), cost = ctx.cost;
    const m = { asker: i, nuclei: s.nuclei.map(q => ({ px: q.px, pos: q.atypical })), X: ctx.X, Q: ctx.Q, K: ctx.K, V: ctx.V, cost, match: ex.match[i], costs: ex.cost[i], shares: ctx.A[i], C: ctx.C[i], heard: Float64Array.from(ctx.Xp[i], (v, d) => v - ctx.X[i][d]), after: ctx.Xp[i], final: ctx.Y[i],
      D: L.D, dk: cl.dk, ffn: cl.F, size: L.size, tint: S.tint, reveal: L.reveal, hover: L.hoverDecide, walk: L.dwalk };
    const cv = $('fd-decide'); cv._model = m; Viz.drawDecision(cv, m);
    $('fd-decide-title').textContent = `How nucleus ${i + 1} decides where to look, in layer ${L.layer + 1}`;
    const row = Array.from(ctx.A[i], (w, k) => ({ k, w })).filter(e => e.k !== i).sort((a, b) => b.w - a.w), best = row[0], bestMatch = row.slice().sort((a, b) => ex.match[i][b.k] - ex.match[i][a.k])[0];
    $('fd-decide-note').textContent = `Nucleus ${i + 1}${fdShownWord()} turns its token into a query; each of the other ${n - 1} turns its token into a key and a value, with the same three weight maps. Query × key, ${cl.dk} products added up, is the match: how well that nucleus fits what nucleus ${i + 1} is looking for (best match: nucleus ${bestMatch.k + 1}, ${Viz.fmtSigned(ex.match[i][bestMatch.k], 2)}). The learned distance cost, ${cost.toFixed(2)} per nucleus diameter, is taken off; the softmax over the ${n - 1} turns the results into shares (most to nucleus ${best.k + 1}, ${pct(best.w)}); the values are added up with those shares into the message it hears, which is added to its own token and passed through the feed-forward step.`;
  }
  function fdDecideRowTip(m, j) { const L = S.fd, s = fdFocus(), d = s.dist[m.asker][j], where = d <= 1.3 ? 'touching' : `${d.toFixed(1)} diameters away`, q = s.nuclei[j]; return `nucleus ${j + 1} · ${where} · match ${Viz.fmtSigned(m.match[j], 2)} − ${m.costs[j].toFixed(2)} = ${Viz.fmtSigned(m.match[j] - m.costs[j], 2)} → ${pct(m.shares[j])} of what nucleus ${m.asker + 1} hears${L.reveal ? ` · ${q.kind === 'stroma' ? 'stroma' : q.atypical ? 'atypical' : 'bland'}${q.kind === 'stroma' ? '' : q.below ? ', below the membrane' : ', in the epithelium'}` : ''}`; }
  function fdDecideWalkStart() {
    const L = S.fd; if (!L.model || !L.ctx) return; fdStop(); fdWalkStop();
    const { fw } = fdFocusForward();
    L.dwalk = { stage: 'query', k: 0, t: 0, t0: performance.now(), i: fdShown(fw, L.head) }; L.hoverDecide = null; L.walkNucleus = L.dwalk.i;
    fdSyncDecideWalk(); fdRenderField(); fdRenderContext(); requestAnimationFrame(fdDecideTick);
  }
  function fdDecideWalkStop() { const L = S.fd; if (!L.dwalk) return; L.dwalk = null; L.walkNucleus = null; fdSyncDecideWalk(); if (L.model) { fdRenderField(); fdRenderContext(); fdRenderDecide(); } }
  function fdSyncDecideWalk() { $('fd-decide-walk').textContent = S.fd.dwalk ? '■ Stop' : '▶ Walk through'; }
  function fdDecideTick(now) {
    const L = S.fd, w = L.dwalk; if (!w) return;
    const el = Math.max(0, now - w.t0), n = fdFocus().nuclei.length, perKey = Math.min(DE_WALK.keys, 1200 / n), perMatch = Math.min(DE_WALK.match, 2400 / n), next = st => { w.stage = st; w.k = 0; w.t = 0; w.t0 = now; };
    const timed = (dur, after) => { w.t = Math.min(1, el / dur); if (el >= dur + DE_WALK.pause) next(after); };
    if (w.stage === 'query') timed(DE_WALK.query, 'keys');
    else if (w.stage === 'keys') { w.k = Math.min(n, Math.floor(el / perKey) + 1); if (el >= n * perKey + DE_WALK.pause) next('match'); }
    else if (w.stage === 'match') { w.k = Math.min(n, Math.floor(el / perMatch) + 1); if (el >= n * perMatch + DE_WALK.pause) next('distance'); }
    else if (w.stage === 'distance') timed(DE_WALK.distance, 'softmax');
    else if (w.stage === 'softmax') timed(DE_WALK.softmax, 'message');
    else if (w.stage === 'message') timed(DE_WALK.message, 'add');
    else if (w.stage === 'add') {
      w.t = Math.min(1, el / DE_WALK.add);
      if (el >= DE_WALK.add + DE_WALK.pause) {
        const i = w.i, { fw } = fdFocusForward(), row = Array.from(fdLayerFw(fw).A[i], (v, k) => ({ k, v })).filter(e => e.k !== i).sort((a, b) => b.v - a.v).slice(0, 2);
        fdDecideWalkStop(); fdNote(`Nucleus ${i + 1} through layer ${L.layer + 1}: query against every key, minus the distance cost, softmax; it listens most to ${row.map(e => `nucleus ${e.k + 1} (${pct(e.v)})`).join(' and ')}, and what it hears is added to its own token before the heads see it.`);
        return;
      }
    }
    fdRenderDecide();
    requestAnimationFrame(fdDecideTick);
  }
  function bindFieldCards() {
    const L = S.fd;
    document.querySelectorAll('#fd-head-seg button').forEach(b => b.addEventListener('click', () => { L.head = +b.dataset.head; fdWalkStop(); fdSyncControls(); if (L.model) fdRenderFocus(); }));
    document.querySelectorAll('#fd-ctx-layer-seg button').forEach(b => b.addEventListener('click', () => { L.layer = +b.dataset.layer; fdDecideWalkStop(); fdSyncControls(); if (L.model) fdRenderFocus(); }));
    document.querySelectorAll('#fd-attview-seg button').forEach(b => b.addEventListener('click', () => { L.attView = b.dataset.view; fdSyncControls(); if (L.model) { fdRenderField(); fdRenderContext(); } }));
    const pin = i => { if (i == null || !L.ctx) return; L.pinned = L.pinned === i ? null : i; fdDecideWalkStop(); fdRenderFocus(); fdNote(L.pinned == null ? 'Unpinned: the cards follow the hovered nucleus, else the one with the most attention.' : `Nucleus ${i + 1} pinned: the cards follow it until you click it again.`); };
    const ca = $('fd-attmap'), ta = $('fd-attmap-tip');
    ca.addEventListener('mouseenter', () => fdFreeze(true));
    ca.addEventListener('mousemove', ev => {
      if (!L.model || !L.ctx || L.walk) return;
      const r = ca.getBoundingClientRect(), hit = Viz.hitAttentionMap(ca, ev.clientX - r.left, ev.clientY - r.top);
      const prevI = L.hoverNucleus, prevKey = L.hoverAtt ? `${L.hoverAtt.i}:${L.hoverAtt.j}` : '';
      L.hoverAtt = hit; if (hit) L.hoverNucleus = hit.i;
      if (hit && hit.i !== prevI) fdRenderFocus();
      else if ((hit ? `${hit.i}:${hit.j}` : '') !== prevKey) fdRenderContext();
      if (hit) {
        const { s, fw } = fdFocusForward(), c = fdLayerFw(fw), ex = fdLayer().explain(c), view = fdAttentionView(fw);
        ta.hidden = false;
        ta.textContent = hit.i === hit.j ? `nucleus ${hit.i + 1} does not listen to itself` : `nucleus ${hit.i + 1} listens to nucleus ${hit.j + 1}: ${pct(view[hit.i][hit.j])}${L.attView === 'both' ? '' : ` from the ${L.attView} alone (${pct(c.A[hit.i][hit.j])} in fact)`} · match ${Viz.fmtSigned(ex.match[hit.i][hit.j], 2)} − distance ${s.dist[hit.i][hit.j].toFixed(1)} × ${c.cost.toFixed(2)} · click to pin the row`;
        const half = ta.offsetWidth / 2 + 4; ta.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; ta.style.top = (ev.clientY - r.top) + 'px';
      } else ta.hidden = true;
    });
    ca.addEventListener('mouseleave', () => { ta.hidden = true; fdFreeze(false); if (L.hoverAtt || L.hoverNucleus != null) { L.hoverAtt = null; L.hoverNucleus = null; if (L.model) fdRenderFocus(); } });
    ca.addEventListener('click', ev => { if (!L.model || !L.ctx) return; const r = ca.getBoundingClientRect(), hit = Viz.hitAttentionMap(ca, ev.clientX - r.left, ev.clientY - r.top); if (hit) pin(hit.i); });
    // the decision card: hovering a row outlines the pair on the map; the numbers explain themselves
    const cd = $('fd-decide'), td = $('fd-decide-tip'), rowOf = h => (h && h.j != null ? h.j : null), keyOf = h => (h ? `${h.kind}:${h.j == null ? '' : h.j}:${h.d == null ? '' : h.d}` : '');
    cd.addEventListener('mouseenter', () => fdFreeze(true));
    cd.addEventListener('mousemove', ev => {
      const m = cd._model; if (!m || L.dwalk) return;
      const r = cd.getBoundingClientRect(), hit = Viz.hitDecision(cd, ev.clientX - r.left, ev.clientY - r.top, m), prev = L.hoverDecide;
      L.hoverDecide = hit;
      if (keyOf(prev) !== keyOf(hit)) { fdRenderDecide(); if (rowOf(prev) !== rowOf(hit)) fdRenderContext(); }
      if (hit) { td.hidden = false; td.textContent = hit.kind === 'asker' ? fdNucleusTip(fdFocus(), fdFocusForward().fw, m.asker) : hit.kind === 'row' ? fdDecideRowTip(m, hit.j) : hit.text; const half = td.offsetWidth / 2 + 4; td.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; td.style.top = (ev.clientY - r.top) + 'px'; }
      else td.hidden = true;
    });
    cd.addEventListener('mouseleave', () => { td.hidden = true; fdFreeze(false); if (L.hoverDecide) { L.hoverDecide = null; if (L.model) { fdRenderDecide(); fdRenderContext(); } } });
    $('fd-decide-walk').addEventListener('click', () => (L.dwalk ? fdDecideWalkStop() : fdDecideWalkStart()));
    // the unrolled diagram: hovering a nucleus row follows it everywhere; the other parts explain themselves
    const cu = $('fd-unrolled'), tu = $('fd-unrolled-tip'), hitKey = h => (h ? `${h.kind}:${h.i == null ? '' : h.i}:${h.d == null ? '' : h.d}` : '');
    cu.addEventListener('mouseenter', () => fdFreeze(true));
    cu.addEventListener('mousemove', ev => {
      const m = cu._model; if (!m || L.walk) return;
      const r = cu.getBoundingClientRect(), hit = Viz.hitSlideNetwork(cu, ev.clientX - r.left, ev.clientY - r.top, m), prev = L.hoverUnrolled;
      const ni = hit && hit.kind === 'nucleus' ? hit.i : null;
      L.hoverUnrolled = hit;
      if (ni !== L.hoverNucleus) { L.hoverNucleus = ni; fdRenderFocus(); }
      else if (hitKey(prev) !== hitKey(hit)) fdRenderUnrolled();
      if (hit) { tu.hidden = false; tu.textContent = hit.kind === 'nucleus' ? fdNucleusTip(fdFocus(), fdFocusForward().fw, hit.i) : hit.text; const half = tu.offsetWidth / 2 + 4; tu.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; tu.style.top = (ev.clientY - r.top) + 'px'; }
      else tu.hidden = true;
    });
    cu.addEventListener('mouseleave', () => { tu.hidden = true; fdFreeze(false); const had = L.hoverNucleus != null || L.hoverUnrolled; L.hoverUnrolled = null; L.hoverNucleus = null; if (had && L.model) fdRenderFocus(); });
    $('fd-walk').addEventListener('click', () => (L.walk ? fdWalkStop() : fdWalkStart()));
    for (const [cvId, tipId, key, render] of [['fd-scorer', 'fd-scorer-tip', 'hoverScorer', fdRenderScorer], ['fd-head', 'fd-head-tip', 'hoverHead', fdRenderSummary]]) {
      const c2 = $(cvId), t2 = $(tipId);
      c2.addEventListener('mouseenter', () => fdFreeze(true));
      c2.addEventListener('mousemove', ev => {
        const m = c2._model; if (!m) return;
        const r = c2.getBoundingClientRect(), hit = Viz.hitNetwork(c2, ev.clientX - r.left, ev.clientY - r.top, m), prev = L[key];
        L[key] = hit;
        if (hit) { t2.hidden = false; t2.textContent = hit.text; t2.style.left = (ev.clientX - r.left) + 'px'; t2.style.top = (ev.clientY - r.top) + 'px'; } else t2.hidden = true;
        if ((prev && prev.ref) !== (hit && hit.ref)) render();
      });
      c2.addEventListener('mouseleave', () => { L[key] = null; t2.hidden = true; fdFreeze(false); if (L.model) render(); });
    }
  }
  function bindFields() {
    const L = S.fd; bindFieldCards();
    document.querySelectorAll('#fd-pos-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.pos === '1'; if (v === L.pos) return; L.pos = v; fdReset(v ? 'Every nucleus now carries its position with its code — fresh random weights.' : 'The codes alone: the model no longer knows where a nucleus sits — fresh random weights.'); }));
    document.querySelectorAll('#fd-ctx-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.ctx === '1'; if (v === L.ctx) return; L.ctx = v; fdReset(v ? 'Context on: two layers of self-attention let every nucleus read the others before the heads see it — fresh random weights.' : 'Context off: every nucleus is scored on its own — fresh random weights.'); }));
    document.querySelectorAll('#fd-crop-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.crop; if (v === L.crop) return; L.crop = v; fdSyncControls(); fdReset(v === 'nucleus' ? 'Crops masked to the nucleus: the code carries cytology and nothing else — fresh random weights.' : 'Crops with their surroundings: the field around the nucleus reaches the code, and with it its location and the outline of its nest — fresh random weights.'); }));
    $('fd-backbone').addEventListener('change', () => { S.backbone = $('fd-backbone').value; fdReset(`The code now comes from ${codeEncoder().describe} — fresh random weights.`); });
    $('fd-epochs').addEventListener('input', () => { L.epochs = +$('fd-epochs').value; $('fd-epochs-val').textContent = L.epochs; if (L.model) { fdRenderStatus(); fdRenderCurves(); } });
    $('fd-speed').addEventListener('input', () => { L.speed = speedFromSlider(+$('fd-speed').value); $('fd-speed-val').textContent = `${L.speed} epochs/s`; });
    $('fd-seed').addEventListener('change', () => { L.seed = Math.max(1, Math.round(+$('fd-seed').value) || 1); fdReset(`Seed ${L.seed} — fresh random weights.`); });
    for (const id of ['fd-reveal', 'fd-data-reveal', 'fd-test-reveal']) $(id).addEventListener('change', ev => fdSetReveal(ev.target.checked));
    $('fd-data-prev').addEventListener('click', () => fdDataNeighbour(-1)); $('fd-data-next').addEventListener('click', () => fdDataNeighbour(1));
    $('fd-classify-next').addEventListener('click', () => fdClassifyNext()); $('fd-classify-all').addEventListener('click', fdClassifyAll); $('fd-test-clear').addEventListener('click', fdTestClear);
    $('fd-step-field').addEventListener('click', fdStepField); $('fd-step-epoch').addEventListener('click', fdStepEpoch);
    $('fd-train').addEventListener('click', () => (L.running ? fdStop('Paused.') : fdStart()));
    $('fd-reset').addEventListener('click', () => fdReset('Weights re-initialised from the seed.'));
    $('fd-prev').addEventListener('click', () => fdNeighbour(-1)); $('fd-next').addEventListener('click', () => fdNeighbour(1));
    document.querySelectorAll('#fd-links-seg button').forEach(b => b.addEventListener('click', () => { L.linksAll = b.dataset.links === '1'; fdSyncControls(); if (L.model) fdRenderField(); }));
    document.querySelectorAll('#fd-layer-seg button').forEach(b => b.addEventListener('click', () => { L.layer = +b.dataset.layer; fdDecideWalkStop(); fdSyncControls(); if (L.model) fdRenderFocus(); }));
    $('fd-links-min').addEventListener('input', () => { L.linksMin = +$('fd-links-min').value / 100; $('fd-links-min-val').textContent = pct(L.linksMin); if (L.model) fdRenderField(); });
    // hovering either viewer follows the nucleus on both; a click keeps it
    for (const [cvId, tipId] of [['fd-canvas-cis', 'fd-tip-cis'], ['fd-canvas-inv', 'fd-tip-inv'], ['fd-test-canvas-cis', 'fd-test-tip-cis'], ['fd-test-canvas-inv', 'fd-test-tip-inv']]) {
      const cv = $(cvId), tip = $(tipId);
      cv.addEventListener('mouseenter', () => fdFreeze(true));
      cv.addEventListener('mousemove', ev => {
        if (!L.model || (fdOnTest() && !fdTestResult())) return;
        const r = cv.getBoundingClientRect(), i = Viz.hitField(cv, ev.clientX - r.left, ev.clientY - r.top);
        if (i !== L.hoverNucleus) { L.hoverNucleus = i; fdRenderFocus(); }
        if (i != null) { const { s, fw } = fdFocusForward(); tip.hidden = false; tip.textContent = fdNucleusTip(s, fw, i); const half = tip.offsetWidth / 2 + 4; tip.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; tip.style.top = (ev.clientY - r.top) + 'px'; }
        else tip.hidden = true;
      });
      cv.addEventListener('mouseleave', () => { tip.hidden = true; fdFreeze(false); if (L.hoverNucleus != null) { L.hoverNucleus = null; if (L.model) fdRenderFocus(); } });
      cv.addEventListener('click', ev => { if (!L.model || !L.ctx) return; const r = cv.getBoundingClientRect(), i = Viz.hitField(cv, ev.clientX - r.left, ev.clientY - r.top); if (i == null) return; L.pinned = L.pinned === i ? null : i; fdRenderFocus(); fdNote(L.pinned == null ? 'Unpinned: the lines follow the hovered nucleus.' : `Nucleus ${i + 1} pinned: the lines follow it until you click it again.`); });
    }
    const dc = $('fd-data-canvas'), dt = $('fd-data-tip');
    dc.addEventListener('mousemove', ev => {
      if (!L.built) return; const s = L.byId.get(L.dataSelected); if (!s) return;
      const r = dc.getBoundingClientRect(), i = Viz.hitField(dc, ev.clientX - r.left, ev.clientY - r.top);
      if (i !== L.dataHover) { L.dataHover = i; fdRenderDataCanvas(); }
      if (i != null) { const n = s.nuclei[i]; dt.hidden = false; dt.textContent = `nucleus ${i + 1} · ${n.kind === 'stroma' ? 'stromal spindle cell' : n.below ? 'below the membrane' : 'in the epithelium'}${L.reveal ? ` · ${n.kind === 'stroma' ? '' : n.atypical ? `atypical (${subtypeWord(n.subtype)})` : 'bland'}` : ''}`; const half = dt.offsetWidth / 2 + 4; dt.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; dt.style.top = (ev.clientY - r.top) + 'px'; }
      else dt.hidden = true;
    });
    dc.addEventListener('mouseleave', () => { dt.hidden = true; if (L.dataHover != null) { L.dataHover = null; if (L.built) fdRenderDataCanvas(); } });
  }

  // ------------------------------------------------------------------ the Reports question: a small language model
  // 908 synthetic bladder biopsy reports, each a findings block, as an image analyser would report it, followed by
  // the report in a sign-out's order with the diagnosis last. Every word is a token; the model predicts each next word
  // from the words before it, and nothing else ever teaches it. The page trains it live in batches of four reports
  // (an epoch is about half a minute), measures its loss by section on test reports and, every epoch, the diagnosis
  // it writes from the findings alone on a few of them; the same model after 30 epochs in Node ships with the page.
  const RP = { batch: 4, layers: 2, heads: 2, costs: [0.05, 0.003], positions: 200, clip: 1, decay: 0.0001, groundN: 8, evalN: 40, frameMs: 70, maxWrite: 200,
    colors: { findings: '#7c3aed', specimen: '#8a8fa8', clinical: '#b0a37a', gross: '#d4a017', microscopic: '#0e9f6e', diagnosis: '#eb6834' } };
  const RPR = window.NucleusReports;
  const CHR = window.NucleusChat;
  const CH = { colors: { system: '#8a8fa8', user: '#b0a37a', findings: '#7c3aed', question: '#d4a017', assistant: '#eb6834', tool: '#0e9f6e' } };
  // The two language-model worlds, the reports and the chat, share the Train step and its cards. Each has its own
  // state, data, words and model, and a descriptor of what differs: the sections (the roles, for a transcript) and
  // their colours, how a document is prepared, the word the diagrams follow by default, the grounding measure taken
  // after every epoch, the Test step's view of what the model wrote, and the words of the notes.
  const LM_WORLDS = {
    reports: {
      key: 'reports', title: 'Reports: a small language model', subtitle: 'every word predicted from the words before it · the only teacher is the next word', testNote: 'rp-test-note',
      state: () => S.rp, data: () => rpData(), shipped: () => window.REPORT_LM,
      sections: RPR.SECTIONS, colors: RP.colors, headers: RPR.HEADERS, sectionsOf: words => RPR.sectionsOf(words).sections,
      unit: 'reports', doc: 'report', heldLabel: 'never trained on', groundLabel: 'diagnosis right from the findings', groundSub: 'test reports, the description written by the model', groundChart: 'Diagnosis right, the description written by the model', groundVerb: 'Measuring the diagnosis from the findings', groundColor: RP.colors.diagnosis,
      prep: (L, r, split) => { const tokens = RPR.encode(L.vocab, r.text), words = RPR.decode(L.vocab, tokens), { sections, lines } = RPR.sectionsOf(words); return { id: r.id, name: r.name, split, case: r.case, dx: r.dx, text: r.text, tokens, words, sections, lines, cls: RPR.classOf(r.dx), dxLine: RPR.diagnosisSectionOf(words), findings: RPR.findingsOf(words) }; },
      groundPrefix: d => d.tokens.slice(0, d.words.indexOf('MICROSCOPIC')), groundStop: () => false, groundRight: (d, words) => RPR.classOf(RPR.diagnosisOf(words)) === d.cls,
      defaultWord: words => { const i = words.indexOf('DIAGNOSIS'); return i >= 0 && i + 2 < words.length ? i + 2 : words.length - 1; }, defaultWordName: 'the first word of the diagnosis',
      docTitle: r => `Report ${r.name} · ${r.split === 'train' ? 'training' : r.split === 'test' ? 'test' : 'never trained on'} · ${RPR.CLASS_NAMES[r.cls]}`,
      thumbClass: r => rpClassGroup(r.cls), thumbTitle: r => `${r.name} · ${r.dx}`, thumbLabel: r => RPR.CLASS_NAMES[r.cls], traysTitle: 'The reports',
      traysLegend: '<span><span class="swatch ring"></span>benign, reactive or denuded</span><span><span class="swatch ring pos"></span>carcinoma in situ, or suspicious</span><span><span class="swatch ring inv"></span>invasive</span><span><span class="swatch batch"></span>the last batch trained on</span>',
      labelled: () => RP_LABELLED, mapExample: 'the states of a findings line, the numbers, the words of the diagnosis',
      untrainedNote: 'Untrained: every next word is a guess over the whole vocabulary, so the report on screen is red all over. Step a batch to watch one gradient step, press Train and watch the loss by section, or load the trained model.',
      loadedNote: 'It writes from the findings; go to 3 · Test, or keep training it here.',
      curvesNote: 'The loss is the surprise per word, in nats, on test reports the model never trains on. The specimen, clinical and gross lines sit on a floor set by the draws: nothing predicts which wall or how many fragments. The microscopic line falls as the phrasings are learned, the diagnosis line, measured with the pathologist’s description in front of the model, falls fastest of all; but the third chart, the diagnosis the model writes when it has to write the description itself, climbs long after the loss has flattened: fluency comes first, grounding last, and the loss cannot tell them apart.',
      attentionNote: n => `Rows ask, columns answer: every row is one word's shares over the words before it, scaled to its largest share, with the sections as bands along the edges and thin lines at their boundaries. The near head hugs the diagonal, reading the words just before; the far head's columns stand in the findings block, where the diagnosis is decided. The bars give each head's attention by distance back, averaged over the ${n} words; the tables, where the words of each section look on average. Hover a cell for its pair of words; the outlined row is the word the diagram follows.`,
      testView: L => { const res = L.written; if (!res || L.writing) return null; return { words: res.words, tokens: res.tokens, fw: res.fw, textEl: $('rp-test-text'), pre: 'rp-test-', chosen: true, steps: res.steps, nextEl: 'rp-test-nextword' }; },
      renderData: () => rpRenderData(), renderTest: () => rpRenderTest(),
    },
    chat: {
      key: 'chat', title: 'Chat: the same model on transcripts', subtitle: 'the next-word model trained on transcripts: a system line, the user’s turn, the assistant’s · it learns to continue them', testNote: 'ch-test-note',
      state: () => S.ch, data: () => (window.LECTURE_CHAT && window.LECTURE_CHAT.meta ? window.LECTURE_CHAT : null), shipped: () => window.CHAT_LM,
      sections: CHR ? CHR.ROLES : [], colors: CH.colors, headers: CHR ? CHR.HEADERS : {}, sectionsOf: words => CHR.rolesOf(words),
      unit: 'transcripts', doc: 'transcript', trayScroll: true, heldLabel: 'held-out phrasings', groundLabel: 'answers right', groundSub: 'test transcripts, the answer written by the model', groundChart: 'Answers right, written by the model', groundVerb: 'Measuring the answers', groundColor: CH.colors.assistant,
      prep: (L, r, split) => { const tokens = RPR.encode(L.vocab, r.text), words = RPR.decode(L.vocab, tokens); return { id: r.id, name: r.name, split, case: r.case, dx: r.dx, style: r.style, line: r.line, q: r.q, qi: r.qi, answer: r.answer, text: r.text, tokens, words, sections: CHR.rolesOf(words), cls: r.style }; },
      groundPrefix: d => d.tokens.slice(0, CHR.firstAnswerIndex(d.words)), groundStop: (token, L) => token === L.nl, groundRight: (d, words) => { const a = CHR.lastAssistant(words); return !!a && CHR.same(a.text, d.answer); },
      defaultWord: words => CHR.firstAnswerIndex(words), defaultWordName: 'the first word of the answer',
      docTitle: r => `Transcript ${r.name} · ${r.split === 'train' ? 'training' : r.split === 'test' ? 'test' : 'held-out phrasing'} · ${r.style === 'full' ? 'full sentence' : r.style} · ${r.q}`,
      thumbClass: r => `ch-${r.style}`, thumbTitle: r => `${r.name} · ${r.q} → ${r.answer}`, thumbLabel: r => (r.style === 'full' ? 'full' : r.style), traysTitle: 'The transcripts',
      traysLegend: '<span><span class="swatch ring ch-brief"></span>brief</span><span><span class="swatch ring ch-full"></span>full sentence</span><span><span class="swatch ring ch-agent"></span>agent</span><span><span class="swatch batch"></span>the last batch trained on</span>',
      labelled: () => CH_LABELLED, mapExample: 'yes and no, the states of a findings line, the words of the questions and of the answers',
      untrainedNote: 'Untrained: every next word is a guess over the whole vocabulary, so the transcript on screen is red all over. Step a batch to watch one gradient step, press Train and watch the loss by role, or load the trained model.',
      loadedNote: 'It answers from the findings; go to 3 · Test to ask it, or keep training it here.',
      curvesNote: 'The loss is the surprise per word, in nats, on test transcripts the model never trains on. The system line and the findings soon cost almost nothing: they repeat. The question line stays high, since nothing predicts which question comes. The assistant line is what matters, and the third chart measures it the hard way: the answer the model writes, exactly right or not.',
      attentionNote: n => `Rows ask, columns answer: every row is one word's shares over the words before it, scaled to its largest share, with the roles as bands along the edges and thin lines at their boundaries. The near head hugs the diagonal, reading the words just before; the far head's columns stand in the findings block and in the question, where the answer is decided. The bars give each head's attention by distance back, averaged over the ${n} words; the tables, where the words of each role look on average. Hover a cell for its pair of words; the outlined row is the word the diagram follows.`,
      testView: L => (L.written && !L.writing && L.chat && L.chat.view ? L.chat.view : null),
      renderData: () => chRenderData(), renderTest: () => chRenderTest(),
    },
  };
  const LMW = () => LM_WORLDS[S.world === 'chat' ? 'chat' : 'reports']; // the language-model world on screen
  const LM = () => LMW().state();                                          // its state
  const WDof = L => LM_WORLDS[L.world];                                    // the descriptor of a state, for work that runs on while another world is on screen
  function lmStopAll() { for (const k of Object.keys(LM_WORLDS)) { const Lx = LM_WORLDS[k].state(); Lx.running = false; Lx.stopAt = null; } rpNetWalkStop(); }
  const rpOnTest = () => (S.world === 'reports' || S.world === 'chat') && S.stage === 'test';
  function rpNote(msg) { $(rpOnTest() ? LMW().testNote : 'rp-note').textContent = msg || ''; } // the note of the step on screen
  function rpData() { const D = window.LECTURE_REPORTS; return D && D.meta ? D : null; }
  const rpClassGroup = cls => (cls === 'invasive' || cls === 'invasive-mp' ? 'rp-inv' : cls === 'cis' || cls === 'suspicious' ? 'rp-cis' : 'rp-benign');
  function rpBuild() {
    const L = LM(), WD = LMW(); if (L.built) return true; const D = WD.data(); if (!D) return false;
    const W = WD.shipped();
    L.meta = D.meta;
    L.vocab = W && W.words ? { words: W.words, index: new Map(W.words.map((w, i) => [w, i])), start: 0, end: 1, unk: 2, size: W.words.length } : RPR.buildVocab(D.train.map(r => r.text)); // the shipped model's vocabulary, so that its weights fit
    const prep = (r, split) => WD.prep(L, r, split);
    L.train = D.train.map(r => prep(r, 'train')); L.test = D.test.map(r => prep(r, 'test')); L.held = (D.held || D.rephrased || []).map(r => prep(r, 'held'));
    L.all = [...L.train, ...L.test, ...L.held]; L.byId = new Map(L.all.map(r => [r.id, r]));
    L.nl = L.vocab.index.get('\n');
    const secCount = L.vocab.words.map(() => ({})); for (const r of L.train) r.tokens.forEach((t, k) => { const sec = r.sections[k]; secCount[t][sec] = (secCount[t][sec] || 0) + 1; });
    L.wordSec = secCount.map(cnt => { let best = null, bv = 0; for (const sec of WD.sections) if ((cnt[sec] || 0) > bv) { bv = cnt[sec]; best = sec; } return best; }); // the section each word mostly appears in
    L.built = true;
    return true;
  }
  function rpConfig() { const L = LM(); return { vocabSize: L.vocab.size, dim: L.dim, layers: RP.layers, heads: RP.heads, dk: L.dim >= 48 ? 12 : 8, ffn: L.dim, costInit: RP.costs, positions: L.positions ? RP.positions : 0, clip: RP.clip, optimizer: 'adam', seed: L.seed }; }
  function rpReset(reason) {
    const L = LM(); rpStop(); if (!rpBuild()) return;
    L.model = new NN.LanguageModel(rpConfig()); L.shipped = false;
    L.rng = NN.mulberry32(L.seed * 977 + 1); L.order = L.train.map((_, i) => i);
    L.epoch = 0; L.ptr = 0; L.hist = []; L.lastBatch = []; L.trainLoss = null; L.lastRender = 0; L.measuring = null; L.fw = null; L.fwFor = null; L.finishedAt = null;
    L.hover = null; L.pinned = null;
    L.trial = { next: 0, results: new Map() }; L.written = null; L.writing = null; L.formCase = null; L.truthShown = false; // a new model: the test step starts over
    rpRecordEpoch(false);
    rpSyncControls(); rpRenderAll();
    if (reason) rpNote(reason);
  }
  // the model trained for 30 epochs in Node, the weights that ship with the page
  function rpLoadShipped() {
    const L = LM(), WD = LMW(), W = WD.shipped(); if (!W) { rpNote('The trained model did not load with the page.'); return; } rpStop(); if (!rpBuild()) return;
    L.model = NN.LanguageModel.fromJSON(W); L.shipped = true; L.dim = W.dim; L.positions = W.positions > 0;
    L.rng = NN.mulberry32(L.seed * 977 + 1); L.order = L.train.map((_, i) => i);
    const hist = Array.isArray(W.hist) && W.hist.length ? W.hist.map(h => Object.assign({}, h)) : null; // the history of its training in Node, epoch by epoch, when the file carries it
    L.epoch = hist ? hist.length : 30; L.epochs = Math.max(L.epochs, L.epoch); L.ptr = 0; L.hist = hist || []; L.lastBatch = []; L.trainLoss = hist ? hist[hist.length - 1].loss : null; L.measuring = null; L.fw = null; L.fwFor = null; L.finishedAt = null;
    L.trial = { next: 0, results: new Map() }; L.written = null; L.writing = null; L.formCase = null; L.truthShown = false;
    if (!hist) rpRecordEpoch(true);
    rpSyncControls(); rpRenderAll();
    rpNote(`Loaded the trained model: ${W.trained}, test loss ${W.testLoss}${hist ? ', and the curves of its training' : ''}. ${WD.loadedNote}`);
  }
  // the test loss by section on the first reports of the test set, and, from epoch 1, the diagnosis written from the
  // findings alone on a few of them (measured in the animation loop, a few words per frame, so the page stays alive)
  function rpRecordEpoch(withGround) {
    const L = LM(), docs = L.test.slice(0, RP.evalN), ev = L.model.evaluate(docs), rec = { epoch: L.epoch, loss: L.trainLoss, testLoss: ev.loss, acc: ev.accuracy, ground: null };
    const sum = {}, n = {}; docs.forEach((d, k) => { for (let i = 0; i + 1 < d.tokens.length; i++) { const s = d.sections[i + 1]; sum[s] = (sum[s] || 0) + ev.per[k][i]; n[s] = (n[s] || 0) + 1; } });
    for (const s of WDof(L).sections) rec[s] = n[s] ? sum[s] / n[s] : null;
    L.hist.push(rec);
    if (withGround) { L.measuring = { docs: L.test.slice(0, RP.groundN), i: 0, st: null, ok: 0, rec }; if (!L.running) requestAnimationFrame(now => rpTick(now, L)); }
  }
  function rpMeasureChunk(deadline, L) { // a few words of the grounding measure per frame: the model writes the description and the diagnosis of a test report from its findings and its requisition, with its cache of the words so far
    const WD = WDof(L), M = L.measuring; if (!M) return;
    while (performance.now() < deadline) {
      const d = M.docs[M.i]; if (!d) break;
      if (!M.st) M.st = L.model.genState(WD.groundPrefix(d)); // the reports: the findings, the specimen, the history and the gross given; the chat: the transcript up to the assistant's last turn
      const { token } = L.model.genNext(M.st, { temperature: 0 });
      if (token === L.vocab.end || WD.groundStop(token, L) || M.st.tokens.length >= RP.maxWrite) {
        if (WD.groundRight(d, RPR.decode(L.vocab, M.st.tokens))) M.ok++;
        M.i++; M.st = null;
      }
    }
    if (M.i >= M.docs.length) { M.rec.ground = M.ok / M.docs.length; L.measuring = null; if (L === LM()) { rpRenderCurves(); rpRenderStatus(); if (!L.running && L.finishedAt === M.rec.epoch) rpFinishNote(); } }
  }
  function rpStep(Lx) { // one gradient step on the next batch of the epoch's order; true when the epoch ended
    const L = Lx || LM(), n = L.train.length;
    if (L.ptr === 0) { const o = L.order; for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(L.rng() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } }
    const idx = L.order.slice(L.ptr, L.ptr + RP.batch), loss = L.model.trainBatch(idx.map(i => L.train[i]), L.lr, RP.decay);
    L.trainLoss = L.trainLoss == null ? loss : 0.95 * L.trainLoss + 0.05 * loss; // a running mean over the recent batches
    L.lastBatch = idx.map(i => L.train[i].id); L.ptr += idx.length; L.fw = null;
    if (L.ptr >= n) { L.ptr = 0; L.epoch++; rpRecordEpoch(true); return true; }
    return false;
  }
  function rpStart(stopAt) {
    const L = LM(); if (!L.model) return;
    if (L.epoch >= L.epochs && stopAt == null) { rpNote(`Already at ${L.epochs} epochs. Raise the epoch count, or reset to train again.`); return; }
    L.running = true; L.stopAt = stopAt == null ? null : stopAt; L.lastRender = 0; rpSyncButtons();
    requestAnimationFrame(now => rpTick(now, L));
  }
  function rpStop(msg, Lx) { const L = Lx || LM(); L.running = false; L.stopAt = null; if (L === LM()) { rpSyncButtons(); if (msg) rpNote(msg); } }
  function rpTick(now, L) { // the training loop of one world's state; it draws only while that world is on screen
    const deadline = now + RP.frameMs;
    try {
      if (L.measuring) rpMeasureChunk(deadline, L); // the grounding measure of the last epoch first
      else if (L.running) {
        let ended = false;
        do { ended = rpStep(L); } while (!ended && L.running && performance.now() < deadline);
        if (ended) { if (L.stopAt != null && L.epoch >= L.stopAt) rpStop(`Epoch ${L.epoch} complete.`, L); else if (L.epoch >= L.epochs) rpFinish(L); }
        if (L === LM() && (ended || now - L.lastRender >= 200)) { rpRender(ended); L.lastRender = now; }
      }
    } catch (e) { console.error(e); L.running = false; if (L === LM()) rpSyncButtons(); }
    if (L.running || L.measuring) requestAnimationFrame(now2 => rpTick(now2, L));
  }
  function rpFinish(Lx) { const L = Lx || LM(), WD = WDof(L); L.finishedAt = L.epoch; rpStop(`Finished ${rpEpochs(L.epochs)} on the next word alone. ${WD.groundVerb} on ${RP.groundN} test ${WD.unit}…`, L); if (!L.measuring && L === LM()) rpFinishNote(); }
  const rpEpochs = n => `${n} epoch${n === 1 ? '' : 's'}`;
  function rpFinishNote() { // the run's last word, with the shipped model for comparison
    const L = LM(), WD = LMW(), last = L.hist[L.hist.length - 1], W = WD.shipped(), shipped = W && Array.isArray(W.hist) && W.hist.length ? W.hist[W.hist.length - 1] : null, same = W && L.dim === W.dim && L.positions === (W.positions > 0);
    const cmp = shipped && shipped.ground != null ? (L.shipped ? ` The trained model that ships with the page, which you continued, had ${pct(shipped.ground)} after its ${W.hist.length} epochs.` : ` The trained model that ships with the page, ${same ? 'the same recipe' : `${W.dim} numbers per token`} after ${W.hist.length} epochs, gets ${pct(shipped.ground)}${last.ground != null && last.ground < shipped.ground ? ': fluent first, grounded later. Load it to compare, or raise the epochs and keep training' : ''}.`) : '';
    rpNote(`Finished ${rpEpochs(L.epochs)} on the next word alone. Test loss ${last.testLoss.toFixed(3)}, ${pct(last.acc)} of next words right, and ${WD.groundLabel} for ${last.ground == null ? '–' : pct(last.ground)} of ${RP.groundN} test ${WD.unit}.${cmp} Go to 3 · Test to ${WD.key === 'chat' ? 'ask it' : 'make it write'}.`);
  }
  function rpStepBatch() { const L = LM(); if (!L.model) return; rpStop(); const ended = rpStep(); rpRender(true); rpNote(ended ? `Epoch ${L.epoch} complete.` : `One gradient step on ${L.lastBatch.length} ${LMW().unit} (${L.lastBatch.join(', ')}): batch ${Math.ceil(L.ptr / RP.batch)} of ${Math.ceil(L.train.length / RP.batch)}.`); }
  function rpStepEpoch() { const L = LM(); if (!L.model) return; if (L.running) { rpStop(); return; } rpStart(L.epoch + 1); }
  function rpSelect(id) { const L = LM(); L.selected = id === L.selected ? null : id; L.hover = null; L.pinned = null; L.fw = null; rpRenderFocus(); rpRenderTrays(); }
  function rpFocus() { const L = LM(); return (L.selected != null && L.byId.get(L.selected)) || (L.lastBatch.length && L.byId.get(L.lastBatch[L.lastBatch.length - 1])) || L.train[0]; } // the report on screen
  function rpNeighbour(step) { const L = LM(), cur = rpFocus(), i = L.all.indexOf(cur); rpSelect(L.all[(i + step + L.all.length) % L.all.length].id); }
  function rpSyncButtons() { const L = LM(); $('rp-train').textContent = L.running && L.stopAt == null ? '⏸ Pause' : L.epoch > 0 ? '▶ Continue' : '▶ Train'; $('rp-step-epoch').textContent = L.running && L.stopAt != null ? '■ Stop' : 'Step epoch'; }
  const rpLrFromSlider = v => +Math.pow(10, -3 + 1.3 * v / 100).toPrecision(2), rpSliderFromLr = lr => Math.round((Math.log10(lr) + 3) / 1.3 * 100); // 0.001 .. 0.02
  function rpSyncControls() {
    const L = LM();
    document.querySelectorAll('#rp-dim-seg button').forEach(b => b.classList.toggle('is-active', +b.dataset.dim === L.dim));
    document.querySelectorAll('#rp-pos-seg button').forEach(b => b.classList.toggle('is-active', (b.dataset.pos === '1') === L.positions));
    document.querySelectorAll('.rp-layer-seg button').forEach(b => b.classList.toggle('is-active', +b.dataset.layer === L.layer));
    document.querySelectorAll('.rp-head-seg button').forEach(b => b.classList.toggle('is-active', +b.dataset.head === L.head));
    document.querySelectorAll('.rp-arcs-check').forEach(cb => { cb.checked = L.arcs; });
    document.querySelectorAll('.rp-hops-check').forEach(cb => { cb.checked = L.hops; });
    document.querySelectorAll('.rp-netview-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.view === L.netView));
    document.querySelectorAll('#rp-map-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.map === L.map.mode)); document.querySelectorAll('#rp-map-src-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.src === L.map.source));
    document.querySelectorAll('#rp-cases-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.cases === L.cases));
    $('rp-epochs').value = L.epochs; $('rp-epochs-val').textContent = L.epochs;
    $('rp-lr').value = rpSliderFromLr(L.lr); $('rp-lr-val').textContent = L.lr;
    $('rp-seed').value = L.seed; $('rp-load').disabled = !LMW().shipped();
    $('rp-temp').value = Math.round(L.temperature * 100); $('rp-temp-val').textContent = L.temperature.toFixed(2);
    $('rp-pace').value = L.pace; $('rp-pace-val').textContent = `${L.pace} words/s`;
    $('rp-noblock').checked = L.noblock; $('rp-data-reveal').checked = L.reveal;
    rpSyncButtons();
  }
  // ---- rendering the Train step
  function rpRenderStatus() {
    const L = LM(), WD = LMW(), last = L.hist[L.hist.length - 1], m = L.model, W = WD.shipped();
    $('rp-status').innerHTML =
      `<span>architecture <b>${esc(m.describe())}</b></span><span>parameters <b>${m.parameterCount()}</b></span>` +
      `<span>${WD.unit} <b>${L.train.length}</b> training · <b>${L.test.length}</b> test · <b>${L.held.length}</b> ${WD.heldLabel}</span>` +
      `<span>epoch <b>${L.epoch}</b> / ${L.epochs}${L.shipped ? ' (the trained model that ships with the page)' : ''}</span><span>batch <b>${L.ptr === 0 ? '–' : Math.ceil(L.ptr / RP.batch)}</b> / ${Math.ceil(L.train.length / RP.batch)}</span>` +
      `<span>loss <b>${L.trainLoss == null ? '–' : L.trainLoss.toFixed(3)}</b> training · <b>${last ? last.testLoss.toFixed(3) : '–'}</b> test</span><span>next word right <b>${last ? pct(last.acc) : '–'}</b> test</span>` +
      `<span>${WD.groundLabel} <b>${L.measuring ? 'measuring…' : last && last.ground != null ? pct(last.ground) : '–'}</b> (${L.shipped && W && W.groundN ? W.groundN : RP.groundN} ${WD.groundSub})</span>`;
  }
  function rpForward(r) { const L = LM(); if (L.fw && L.fwFor === r.id) return L.fw; L.fw = L.model.forward(r.tokens); L.fwFor = r.id; return L.fw; }
  // the words of a report as spans: each coloured by the probability the model gave it, given the words before
  function rpTextHtml(words, probsOf, opts) {
    const WD = LMW(), secs = WD.sectionsOf(words); let html = '', lineStart = true;
    words.forEach((w, i) => {
      if (w === RPR.START) return;
      if (w === RPR.END) { html += `<span class="tok muted" data-i="${i}">⟨end⟩</span>`; return; }
      if (w === '\n') { html += '<br>'; lineStart = true; return; }
      const p = probsOf ? probsOf(i) : null, hd = WD.headers[w] != null, cls = ['tok', `sec-${secs[i]}`, hd ? 'hd' : '', opts && opts.cur === i ? 'cur' : ''].filter(Boolean).join(' ');
      let style = '';
      if (p != null) style = p >= 0.35 ? ` style="background:rgba(12,163,12,${(0.1 + 0.4 * p).toFixed(2)})"` : ` style="background:rgba(208,59,59,${(0.12 + 0.5 * (0.35 - p) / 0.35).toFixed(2)})"`;
      const space = lineStart || /^[,.:;)]$/.test(w) ? '' : ' ';
      html += `${space}<span class="${cls}" data-i="${i}"${style}${p != null ? ` title="${w === '\n' ? '' : esc(w)} · P = ${p.toFixed(3)}"` : ''}>${esc(w)}</span>`; lineStart = false;
    });
    return html;
  }
  function rpNextWordHtml(probs, actual, chosen, title) { // the model's candidates for the next word: the most probable eight, and the actual or chosen one
    const L = LM(), V = probs.length, idx = Array.from({ length: V }, (_, v) => v).sort((a, b) => probs[b] - probs[a]).slice(0, 8);
    if (actual != null && !idx.includes(actual)) idx.push(actual); if (chosen != null && !idx.includes(chosen)) idx.push(chosen);
    const mx = probs[idx[0]] || 1e-9, name = v => (v === L.vocab.end ? '⟨end⟩' : v === L.nl ? '↵' : L.vocab.words[v]);
    return `<div class="title">${esc(title)}</div>` + idx.map(v => `<div class="row${v === actual ? ' actual' : ''}${v === chosen ? ' chosen' : ''}"><span class="w">${esc(name(v))}${v === actual ? ' ✓' : v === chosen ? ' ←' : ''}</span><span class="b"><i style="width:${(100 * probs[v] / mx).toFixed(1)}%"></i></span><span class="p">${pct(probs[v])}</span></div>`).join('');
  }
  function rpAttentionRow(fw, i) { return rpAttentionRowAt(fw, LM().layer, i); } // whom token i reads, in the chosen layer, the chosen head or both averaged
  function rpAttentionRowAt(fw, layer, i) {
    const L = LM(), c = fw.ctxs[Math.max(0, Math.min(layer, fw.ctxs.length - 1))], row = new Float64Array(i + 1);
    const heads = L.head < 0 ? c.Ah.map((_, h) => h) : [Math.min(L.head, c.Ah.length - 1)];
    for (const h of heads) for (let j = 0; j <= i; j++) row[j] += c.Ah[h][i][j] / heads.length;
    return row;
  }
  function rpHighlight(el, fw, i, opts) { // light up the words the model read, standing at the word before word i, to choose it, and draw the lines to them; soft: the word followed by default, not hovered
    const L = LM(), o = opts || {}, spans = el.querySelectorAll('.tok'); spans.forEach(s => { s.classList.remove('reads', 'reads2', 'hover', 'cur', 'pinned'); s.style.boxShadow = ''; });
    const q = i == null ? null : i - 1;
    if (q == null || q < 0 || !fw || q >= fw.tokens.length) { rpArcs(el, fw, null); return; }
    const row = rpAttentionRow(fw, q); let mx = 1e-9; for (let j = 0; j <= q; j++) mx = Math.max(mx, row[j]);
    spans.forEach(s => { const j = +s.dataset.i; if (j === L.pinned) s.classList.add('pinned'); if (j === i) s.classList.add(o.soft ? 'cur' : 'hover'); else if (j <= q && row[j] > 0.02 * mx) { s.classList.add('reads'); s.style.boxShadow = `inset 0 -${(2 + 6 * row[j] / mx).toFixed(1)}px 0 rgba(74,58,167,${(0.15 + 0.7 * row[j] / mx).toFixed(2)})`; } });
    rpArcs(el, fw, i, row, q);
  }
  function rpRenderText() {
    const L = LM(), r = rpFocus(), fw = rpForward(r), probsOf = i => (i > 0 ? fw.probs[i - 1][r.tokens[i]] : null);
    $('rp-report-title').textContent = `${LMW().docTitle(r)}${L.selected == null && L.lastBatch.length ? ' · in the last batch trained on' : ''}`;
    const el = $('rp-text'); el.innerHTML = rpTextHtml(r.words, probsOf, null);
    let s = 0, n = 0, right = 0; for (let i = 1; i < r.tokens.length; i++) { const p = probsOf(i); s += -Math.log(p + 1e-7); n++; const pr = fw.probs[i - 1]; let best = 0; for (let v = 1; v < pr.length; v++) if (pr[v] > pr[best]) best = v; if (best === r.tokens[i]) right++; }
    $('rp-text-note').textContent = `Each word is coloured by the probability the model gave it, given the words before: green when it expected it, red when it did not (loss ${(s / n).toFixed(2)} per word on this ${LMW().doc}, ${pct(right / n)} of its words the most probable). Hover a word: the bars show what the model expected there, and the words it read to expect it light up, with lines to them, in the layer and head chosen above; click to keep a word. The diagrams below follow the same word.`;
    rpFollowWord(false); rpRenderAttention(rpView(false), false);
  }
  function rpRenderNextWord(i) {
    const L = LM(), r = rpFocus(), fw = rpForward(r), el = $('rp-nextword');
    if (i == null || i < 1) { const k = rpDefaultWord(r.words); el.innerHTML = rpNextWordHtml(fw.probs[k - 1], r.tokens[k], null, `before “${rpWordLabel(r.words[k])}”, ${k === r.words.length - 1 ? 'the last word' : LMW().defaultWordName}: what the model expected`); $('rp-next-note').textContent = `The model writes a ${LMW().doc} by drawing the next word from bars like these, over and over: at every step it can only say what usually comes next. Hover any word of the ${LMW().doc} for its bars.`; return; }
    el.innerHTML = rpNextWordHtml(fw.probs[i - 1], r.tokens[i], null, `before “${r.words[i] === '\n' ? '↵' : r.words[i]}”: what the model expected`);
    $('rp-next-note').textContent = `The eight most probable next words after “${r.words.slice(Math.max(1, i - 4), i).map(w => (w === '\n' ? '↵' : w)).join(' ')}”, and the actual one (✓). The lit words on the left, with the lines to them, are the ones the model read, standing at “${rpWordLabel(r.words[i - 1])}”, to expect this word, in layer ${L.layer + 1}${L.head < 0 ? ', both heads' : L.head ? ', the far head' : ', the near head'}; the deeper the underline and the thicker the line, the more it listened.`;
  }
  function rpRenderCurves() {
    const L = LM(), WD = LMW(), keys = WD.sections.map(s => ({ key: s, color: WD.colors[s] })), last = L.hist[L.hist.length - 1];
    Viz.drawSeries($('rp-loss'), { hist: L.hist, keys, maxEpoch: L.epochs });
    Viz.drawSeries($('rp-acc'), { hist: L.hist, keys: [{ key: 'acc', color: Viz.colors().accent }], maxEpoch: L.epochs, pct: true, baseline: null });
    Viz.drawSeries($('rp-ground'), { hist: L.hist, keys: [{ key: 'ground', color: WD.groundColor }], maxEpoch: L.epochs, pct: true, baseline: null }); $('rp-ground-title').textContent = WD.groundChart;
    $('rp-legend').innerHTML = WD.sections.map(s => `<span><span class="line" style="border-top-color:${WD.colors[s]}"></span>${s}</span>`).join('');
    $('rp-loss-now').textContent = last ? `test ${last.testLoss.toFixed(3)}` : ''; $('rp-acc-now').textContent = last ? pct(last.acc) : ''; $('rp-ground-now').textContent = last && last.ground != null ? pct(last.ground) : L.measuring ? 'measuring…' : '';
    $('rp-curves-note').textContent = WD.curvesNote;
  }
  function rpThumb(r, onClick) {
    const WD = LMW(), b = document.createElement('button'); b.type = 'button'; b.className = `thumb report ${WD.thumbClass(r)}`; b.dataset.id = r.id; b.title = WD.thumbTitle(r);
    b.innerHTML = `<span class="nm">${esc(r.name)}</span><span class="dx">${esc(WD.thumbLabel(r))}</span>`;
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = '✗'; b.appendChild(badge);
    b.addEventListener('click', () => (onClick || rpSelect)(r.id));
    return b;
  }
  function rpTrays(kind, onClick) { // a panel's trays, built once
    const L = LM(); if (L.trayFor[kind]) return; L.trayFor[kind] = true;
    const map = new Map(), fill = (id, tray) => { const el = $(id); el.innerHTML = ''; for (const r of tray) { const b = rpThumb(r, onClick); el.appendChild(b); map.set(r.id, b); } };
    if (kind === 'train') { const WD = LMW(); fill('rp-train-tray', L.train); fill('rp-test-tray', L.test); for (const id of ['rp-train-tray', 'rp-test-tray']) $(id).classList.toggle('scroll', !!WD.trayScroll); L.thumbs = map; $('rp-train-label').textContent = `Training ${WD.unit} (${L.train.length})`; $('rp-test-label').textContent = `Test ${WD.unit} (${L.test.length}) · never trained on`; $('rp-trays-title').textContent = WD.traysTitle; $('rp-trays-legend').innerHTML = WD.traysLegend; }
    else if (kind === 'data') { fill('rp-data-train-tray', L.train); fill('rp-data-test-tray', L.test); fill('rp-data-held-tray', L.held); L.dataThumbs = map; $('rp-data-train-label').textContent = `Training reports (${L.train.length})`; $('rp-data-test-label').textContent = `Test reports (${L.test.length}) · held out`; }
    else { fill('rp-test-results', L.cases === 'held' ? L.held : L.test); L.testThumbs = map; $('rp-test-count').textContent = `${(L.cases === 'held' ? L.held : L.test).length} cases`; }
  }
  function rpRenderTrays() { const L = LM(), focus = rpFocus(), batch = new Set(L.lastBatch); rpTrays('train'); for (const [id, el] of L.thumbs) { el.classList.toggle('selected', focus.id === id); el.classList.toggle('in-batch', batch.has(id)); } }
  function rpRenderFocus() { rpRenderText(); }
  function rpRender(full) { const L = LM(); if (!L.model) return; rpRenderStatus(); rpRenderFocus(); if (full) { rpRenderCurves(); rpRenderTrays(); rpRenderWordMap(); } rpSyncButtons(); }
  function rpRenderAll() { const L = LM(); if (!(S.world === 'reports' || S.world === 'chat') || !L.built) return; if (S.stage === 'data') LMW().renderData(); else if (S.stage === 'test') LMW().renderTest(); else if (L.model) rpRender(true); }
  function rpEnter() {
    const L = LM(); if (!rpData()) { rpNote('The reports did not load.'); return; }
    if (!rpBuild()) return;
    const WD = LMW(); $('rp-panel-title').textContent = WD.title; $('rp-panel-sub').textContent = WD.subtitle; L.trayFor.train = false; // the trays are this world's
    L.hover = null; L.pinned = null; L.hoverNet = null; L.hoverAtt = null; rpSyncControls();
    if (!L.model) rpReset(WD.untrainedNote);
    else rpRender(true);
  }
  // ---- Specimens: the reports with their hidden cases and the rule
  function rpSetReveal(v) { S.rp.reveal = v; $('rp-data-reveal').checked = v; rpRenderAll(); }
  function rpEnterData() { const L = S.rp; if (!rpData() || !rpBuild()) return; rpTrays('data', rpDataSelect); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id; rpRenderData(); }
  function rpDataSelect(id) { S.rp.dataSelected = id; rpRenderData(); }
  function rpDataNeighbour(step) { const L = S.rp, i = L.all.findIndex(r => r.id === L.dataSelected); rpDataSelect(L.all[(i + step + L.all.length) % L.all.length].id); }
  const rpCaseWords = { surface: { normal: 'normal', reactive: 'reactive atypia', atypia: 'atypia (carcinoma in situ)', denuded: 'denuded' }, nests: { absent: 'none', benign: 'benign (von Brunn nests)', atypical: 'atypical' }, mp: { ni: 'not identified', present: 'present, not involved', involved: 'involved' } };
  function rpRenderData() {
    const L = S.rp; rpTrays('data', rpDataSelect); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id;
    const r = L.byId.get(L.dataSelected) || L.train[0], c = r.case, M = L.meta;
    $('rp-data-title').textContent = M.question; $('rp-data-sub').textContent = `${L.train.length} training, ${L.test.length} test and ${L.held.length} never-trained-on reports · every word a token · vocabulary of ${L.vocab.size}`;
    $('rp-data-blurb').textContent = M.blurb;
    $('rp-data-report-title').textContent = `Report ${r.name} · ${r.split === 'train' ? 'training' : r.split === 'test' ? 'test' : 'never trained on: a combination the training set never held'}${L.reveal ? ` · ${RPR.CLASS_NAMES[r.cls]}` : ''}`;
    $('rp-data-text').innerHTML = rpTextHtml(r.words, null, null);
    $('rp-data-case').innerHTML = L.reveal
      ? `<div class="row"><span class="k">surface urothelium</span><span class="v">${rpCaseWords.surface[c.surface]}</span></div><div class="row"><span class="k">nests below the membrane</span><span class="v">${rpCaseWords.nests[c.nests]}${c.nests === 'atypical' ? `, ${c.contours}, ${c.desmoplasia ? 'desmoplasia' : 'no desmoplasia'}` : ''}</span></div><div class="row"><span class="k">muscularis propria</span><span class="v">${rpCaseWords.mp[c.mp]}</span></div><div class="row"><span class="k">inflammation</span><span class="v">${c.inflammation} (bears on nothing)</span></div><div class="dx">The rule gives: <b>${esc(r.dxLine)}</b></div>`
      : '<div class="row"><span class="k">the hidden case</span><span class="v">tick Reveal</span></div>';
    $('rp-data-note').textContent = `A findings block of seven lines, as an image analyser would report it, then the report: specimen, clinical history, gross, microscopic description and, last, the diagnosis, which follows from the findings by a fixed rule. The microscopic description says only what the block holds, in one of several phrasings. ${L.reveal ? 'The hidden case above is what the generator drew; the model never sees it, only the words.' : 'Tick Reveal to see the hidden case behind this report, and the rule.'} Click a report below, or step through them.`;
    $('rp-rule-card').hidden = !L.reveal;
    if (L.reveal) $('rp-rule').innerHTML = `<table><thead><tr><th>surface urothelium</th><th>nests below the membrane</th><th>diagnosis</th><th class="num">share</th></tr></thead><tbody>${[
      ['normal', 'absent, or present without atypia', 'Benign urothelium', '33%'], ['reactive atypia', 'absent, or present without atypia', 'Benign urothelium with reactive changes', '12%'], ['denuded', 'absent, or present without atypia', 'Denuded urothelium, no diagnostic abnormality in the material present', '5%'],
      ['atypia', 'absent, or present without atypia', 'Urothelial carcinoma in situ', '15%'], ['atypia or denuded', 'atypical, rounded, no desmoplasia', 'Urothelial carcinoma in situ (involving von Brunn nests, in the description)', '10%'],
      ['atypia or denuded', 'atypical, irregular, desmoplasia', 'Urothelial carcinoma, invasive into lamina propria (into muscularis propria when involved), with associated carcinoma in situ when the surface shows atypia', '20%'],
      ['atypia', 'atypical, contours and desmoplasia discordant', 'Urothelial carcinoma in situ with foci suspicious for invasion', '5%'], ['normal', 'atypical, irregular, desmoplasia', 'Urothelial carcinoma, invasive into lamina propria', 'never trained on']].map(row => `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td><td class="num">${row[3]}</td></tr>`).join('')}</tbody></table><p class="small muted" style="margin-top:8px">A line on the muscularis propria closes every diagnosis. Inflammation bears on nothing. The last row is real and rare, and the training set never holds it: whether the model still calls invasion when it meets it says whether it learned the findings or the templates.</p>`;
    for (const [id, el] of L.dataThumbs) el.classList.toggle('selected', id === L.dataSelected);
  }
  // ---- Test: the findings go in, the model writes the report
  function rpCaseSet() { const L = S.rp; return L.cases === 'held' ? L.held : L.test; }
  function rpEnterTest() { const L = S.rp; if (!rpData() || !rpBuild()) return; if (!L.model) rpReset(); L.hover = null; L.pinned = null; L.hoverNet = null; L.hoverAtt = null; rpTrays('test', rpTestSelect); rpBuildForm(); if (!L.form) rpLoadCase(rpCaseSet()[Math.min(L.trial.next, rpCaseSet().length - 1)]); rpRenderTest(); }
  function rpBuildForm() {
    const L = S.rp, el = $('rp-form'); if (el.childElementCount) return;
    el.innerHTML = RPR.FINDINGS.map(l => `<div class="row" data-key="${l.key}"><label for="rp-f-${l.key}">${esc(l.line)}:</label><select id="rp-f-${l.key}">${l.states.map(s => `<option value="${s}">${s}</option>`).join('')}</select></div>`).join('');
    for (const l of RPR.FINDINGS) $(`rp-f-${l.key}`).addEventListener('change', () => { rpReadForm(); L.formCase = null; L.written = null; L.truthShown = false; if (L.writing) rpWriteStop(); rpRenderTest(); });
  }
  function rpReadForm() { const L = S.rp, f = {}; for (const l of RPR.FINDINGS) f[l.key] = $(`rp-f-${l.key}`).value; if (f.nests === 'absent') { f.atypia = 'none'; f.contours = 'none'; f.stromal = 'none'; } L.form = f; rpSyncForm(); }
  function rpSyncForm() { const L = S.rp, f = L.form; if (!f) return; for (const l of RPR.FINDINGS) { const sel = $(`rp-f-${l.key}`); sel.value = f[l.key]; const off = f.nests === 'absent' && ['atypia', 'contours', 'stromal'].includes(l.key); sel.disabled = off || L.noblock; sel.parentElement.classList.toggle('off', off || L.noblock); } }
  function rpLoadCase(r) { const L = S.rp; L.form = Object.assign({}, r.findings); L.formCase = r.id; L.reqCase = r.id; L.written = null; L.truthShown = false; L.testSelected = r.id; rpSyncForm(); }
  function rpRequisition() { const L = S.rp, r = L.reqCase != null ? L.byId.get(L.reqCase) : null; if (!r) return []; const iSp = r.words.indexOf('SPECIMEN'), iMi = r.words.indexOf('MICROSCOPIC'); return iSp >= 0 && iMi > iSp ? r.tokens.slice(iSp, iMi) : []; } // the case's specimen, clinical history and gross, up to the newline before the microscopic description
  function rpTestSelect(id) { const L = S.rp, r = L.byId.get(id); if (!r) return; if (L.writing) rpWriteStop(); rpLoadCase(r); const res = L.trial.results.get(id); if (res) { L.written = res; L.truthShown = true; } rpRenderTest(); }
  function rpNextCase() { const L = S.rp, set = rpCaseSet(); if (L.writing) rpWriteStop(); let i = L.trial.next; if (i >= set.length) { rpNote(`All ${set.length} cases written. Clear to start over, or edit the findings and write again.`); return null; } rpLoadCase(set[i]); rpRenderTest(); return set[i]; }
  function rpPrefix() { const L = S.rp; if (L.noblock) return [L.vocab.start]; const enc = RPR.encode(L.vocab, RPR.blockOf(L.form)); return enc.slice(0, -1).concat([L.nl], rpRequisition()); } // the block from the form, then the case's requisition and gross: the model writes the microscopic description and the diagnosis
  function rpWrite() { // the model writes the report for the findings shown, a few words per frame at the chosen pace
    const L = S.rp; if (!L.model || !L.form) return; if (L.writing) { rpWriteStop(); return; }
    const prefix = rpPrefix(), st = L.model.genState(prefix); L.writing = { prefix, st, tokens: st.tokens, steps: [], t0: performance.now(), count: 0, docId: L.noblock ? null : L.formCase }; L.written = null; L.truthShown = false; L.hover = null; L.pinned = null;
    $('rp-write').textContent = '■ Stop'; rpRenderTest(); requestAnimationFrame(rpWriteTick);
  }
  function rpWriteStop() { const L = S.rp; if (!L.writing) return; L.writing = null; $('rp-write').textContent = 'Write the report'; }
  function rpWriteTick(now) {
    const L = S.rp, W = L.writing; if (!W) return;
    try {
      const due = Math.min(6, Math.floor((now - W.t0) / 1000 * L.pace) - W.count); let done = false;
      for (let k = 0; k < due && !done; k++) {
        const at = W.st.tokens.length, { token, probs } = L.model.genNext(W.st, { temperature: L.temperature }); W.steps.push({ at, token, probs }); W.count++;
        if (token === L.vocab.end || W.tokens.length >= RP.maxWrite) done = true;
      }
      if (due > 0 || done) rpRenderWriting();
      if (done) { rpWriteFinish(); return; }
    } catch (e) { console.error(e); rpWriteStop(); return; }
    requestAnimationFrame(rpWriteTick);
  }
  function rpWriteFinish() {
    const L = S.rp, W = L.writing; if (!W) return;
    const words = RPR.decode(L.vocab, W.tokens), dx = RPR.diagnosisOf(words), dxLine = RPR.diagnosisSectionOf(words), fw = L.model.forward(W.tokens);
    const doc = W.docId ? L.byId.get(W.docId) : null, rule = L.noblock ? null : RPR.ruleOf(L.form), truthDx = doc ? doc.dx : rule ? rule.dx : null, truthLine = doc ? doc.dxLine : rule ? rule.text : null;
    const res = { tokens: W.tokens, words, steps: W.steps, fw, dx, dxLine, truthDx, truthLine, right: truthDx != null && RPR.classOf(dx) === RPR.classOf(truthDx), exact: truthLine != null && dxLine === truthLine, docId: W.docId, noblock: L.noblock, findings: L.noblock ? RPR.findingsOf(words) : Object.assign({}, L.form) };
    L.writing = null; $('rp-write').textContent = 'Write the report'; L.written = res; L.truthShown = true;
    if (doc) { L.trial.results.set(doc.id, res); const set = rpCaseSet(), i = set.indexOf(doc); if (i === L.trial.next) L.trial.next = i + 1; }
    rpRenderTest();
  }
  function rpWriteNext() { const L = S.rp; if (L.writing || L.batchWriting) return; const r = rpNextCase(); if (r) rpWrite(); }
  function rpWriteAll() { // every remaining case, one per frame, no animation
    const L = S.rp; if (!L.model || L.writing || L.batchWriting) return; L.batchWriting = true; $('rp-write-all').textContent = '■ Stop';
    const set = rpCaseSet(), step = () => {
      if (!L.batchWriting) return;
      const r = set[L.trial.next]; if (!r) { L.batchWriting = false; $('rp-write-all').textContent = 'Write all'; rpRenderTest(); rpNote(`All ${set.length} cases written.`); return; }
      L.noblock = false; rpLoadCase(r); const prefix = rpPrefix(), st = L.model.genState(prefix), steps = [];
      while (st.tokens.length < RP.maxWrite) { const at = st.tokens.length, { token, probs } = L.model.genNext(st, { temperature: L.temperature }); steps.push({ at, token, probs }); if (token === L.vocab.end) break; }
      L.writing = { prefix, st, tokens: st.tokens, steps, docId: r.id }; rpWriteFinish();
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function rpTestClear() { const L = S.rp; if (L.writing) rpWriteStop(); L.batchWriting = false; $('rp-write-all').textContent = 'Write all'; L.trial = { next: 0, results: new Map() }; L.written = null; L.truthShown = false; rpLoadCase(rpCaseSet()[0]); rpRenderTest(); }
  function rpSetCases(v) { const L = S.rp; if (v === L.cases) return; if (L.writing) rpWriteStop(); L.batchWriting = false; L.cases = v; L.trayFor.test = false; L.trial = { next: 0, results: new Map() }; L.written = null; L.truthShown = false; rpTrays('test', rpTestSelect); rpLoadCase(rpCaseSet()[0]); rpSyncControls(); rpRenderTest(); }
  function rpTestStats() { const L = S.rp; let n = 0, right = 0, exact = 0; for (const [, r] of L.trial.results) { n++; if (r.right) right++; if (r.exact) exact++; } return { n, right, exact }; }
  function rpRenderWriting() { // the report as it grows, and the bars for the word being chosen
    const L = S.rp, W = L.writing, words = RPR.decode(L.vocab, W.tokens), stepAt = new Map(W.steps.map(s => [s.at, s]));
    const probsOf = i => { const s = stepAt.get(i); return s ? s.probs[s.token] : null; };
    $('rp-test-text').innerHTML = rpTextHtml(words, probsOf, null) + '<span class="cursor"></span>';
    const last = W.steps[W.steps.length - 1];
    if (last) { $('rp-test-nextword').innerHTML = rpNextWordHtml(last.probs, null, last.token, `word ${W.steps.length}: what the model chose from`); }
    $('rp-test-call').textContent = `Writing… ${W.steps.length} words so far${L.temperature > 0 ? `, drawn at temperature ${L.temperature.toFixed(2)}` : ', the most probable word each time'}.`;
  }
  function rpRenderTest() {
    const L = S.rp; if (!L.built || !L.model) return; rpTrays('test', rpTestSelect); rpSyncForm();
    const set = rpCaseSet(), T = L.trial, st = rpTestStats();
    $('rp-stat-n').textContent = `${st.n} / ${set.length}`;
    $('rp-stat-dx').textContent = st.n ? pct(st.right / st.n) : '–'; $('rp-stat-dx-sub').textContent = st.n ? `${st.right} of ${st.n} the right class` : 'no case written yet';
    $('rp-stat-exact').textContent = st.n ? pct(st.exact / st.n) : '–'; $('rp-stat-exact-sub').textContent = st.n ? `${st.exact} of ${st.n} word for word` : '';
    const done = T.next >= set.length;
    $('rp-next-case').disabled = done; $('rp-write-all').disabled = done && !L.batchWriting;
    $('rp-next-case').textContent = done ? `All ${set.length} cases loaded` : `Next case (${T.next + 1} of ${set.length})`;
    $('rp-test-warning').hidden = !(L.model && L.model.steps === 0 && !L.shipped);
    const doc = L.formCase ? L.byId.get(L.formCase) : null, showCards = !!L.written && !L.writing;
    $('rp-test-net-card').hidden = !showCards; $('rp-test-heads-card').hidden = !showCards; $('rp-test-att-card').hidden = !showCards;
    $('rp-test-title').textContent = L.noblock ? 'Nothing given: the model writes a whole case from nothing' : doc ? `Case ${doc.name} · ${doc.split === 'held' ? 'never trained on' : 'test'} · the findings and the requisition go in, the description and the diagnosis come out` : 'Findings of your own · the findings and the requisition go in, the description and the diagnosis come out';
    if (L.writing) { rpRenderWriting(); }
    else if (L.written) {
      const res = L.written, el = $('rp-test-text'), stepAt = new Map(res.steps.map(s => [s.at, s])), probsOf = i => { const s = stepAt.get(i); return s ? s.probs[s.token] : null; };
      el.innerHTML = rpTextHtml(res.words, probsOf, null);
      $('rp-test-call').innerHTML = res.dx ? `The model wrote: <b>${esc(res.dxLine || res.dx)}</b>${res.truthDx != null ? ` → <span class="${res.right ? 'good-text' : 'bad-text'}">${res.right ? 'the right diagnosis' : 'the wrong diagnosis'} ${res.right ? '✓' : '✗'}${res.right && !res.exact ? ', not word for word' : ''}</span>` : ''} · ${res.steps.length} words${L.temperature > 0 ? `, drawn at temperature ${L.temperature.toFixed(2)}` : ', the most probable word each time'}. Hover a word for what the model chose from and the words it read.` : 'The model wrote no diagnosis line at all.';
      $('rp-truth').hidden = !L.truthShown;
      if (L.truthShown) $('rp-truth').innerHTML = res.noblock
        ? `<div>Nothing was given, so there is no truth: the model drew a whole case from what it learned, findings and requisition included${res.findings ? `, and for the findings it drew the rule would give: <b>${esc(RPR.ruleOf(res.findings).text)}</b>` : ''} Draw again, and again: at temperature 1 the diagnoses follow the base rates of the queue it was trained on.</div>`
        : `<div>The rule says: <b>${esc(res.truthLine)}</b>${doc && !L.noblock ? ` · the generator’s own report for this case reads: <b>${esc(doc.dxLine)}</b>` : ''}</div>${doc ? '' : '<div class="small muted">Findings of your own: the rule’s answer is what a pathologist would sign out for them; the model has only ever seen the training reports.</div>'}`;
      rpRenderEvidence(res); rpFollowWord(true); rpRenderAttention(rpView(true), false);
    } else {
      $('rp-test-text').innerHTML = L.noblock ? '<span class="muted">Press Write the report: from the start token alone the model writes findings and all.</span>' : rpTextHtml(RPR.decode(L.vocab, rpPrefix()), null, null) + '<span class="muted"> … press Write the report.</span>';
      $('rp-test-nextword').innerHTML = ''; $('rp-test-call').textContent = L.noblock ? '' : 'The findings, the specimen, the clinical history and the gross are the prompt, as they would come with the case; the model continues from the newline after the gross, one word at a time, until it writes the end token.';
      $('rp-truth').hidden = true; $('rp-evidence-card').hidden = true;
    }
    for (const [id, el] of L.testThumbs) { const rr = T.results.get(id); el.classList.toggle('selected', id === L.testSelected); el.classList.toggle('wrong', !!rr && !rr.right); el.classList.toggle('right', !!rr && rr.right); el.querySelector('.badge').textContent = rr && rr.right ? '✓' : '✗'; }
  }
  function rpRenderEvidence(res) { // where the diagnosis words looked: the second layer's attention, heads averaged, by block line and by section
    const L = S.rp, words = res.words, iDx = words.indexOf('DIAGNOSIS'); if (iDx < 0 || !res.fw.ctxs.length) { $('rp-evidence-card').hidden = true; return; }
    let end = iDx + 2; while (end < words.length && words[end] !== '.') end++;
    const { sections, lines } = RPR.sectionsOf(words), c = res.fw.ctxs[res.fw.ctxs.length - 1], bySec = {}, byLine = {}; let n = 0;
    for (let i = iDx + 2; i <= end && i < words.length; i++) { n++; for (let j = 0; j <= i; j++) { let a = 0; for (let h = 0; h < c.Ah.length; h++) a += c.Ah[h][i][j]; a /= c.Ah.length; bySec[sections[j]] = (bySec[sections[j]] || 0) + a; if (lines[j]) byLine[lines[j]] = (byLine[lines[j]] || 0) + a; } }
    if (!n) { $('rp-evidence-card').hidden = true; return; }
    const bars = (title, rows) => `<div><div class="title">${title}</div><div class="rp-next">${rows.map(([k, v]) => `<div class="row"><span class="w">${esc(k)}</span><span class="b"><i style="width:${(100 * Math.min(1, v / n) / 0.5).toFixed(1)}%"></i></span><span class="p">${pct(v / n)}</span></div>`).join('')}</div></div>`;
    $('rp-evidence-card').hidden = false;
    $('rp-evidence').innerHTML = bars('by line of the findings', RPR.FINDINGS.map(l => { const key = { surface: 'surface', nests: 'nests', atypia: 'atypia in nests', contours: 'contours', stromal: 'stromal reaction', mp: 'muscularis propria', inflammation: 'inflammation' }[l.key]; return [l.line, byLine[key] || 0]; })) + bars('by section of the report', RPR.SECTIONS.map(s => [s === 'diagnosis' ? 'the diagnosis itself' : s, bySec[s] || 0]));
  }
  // ---- the model's diagrams: how a word was chosen, who reads whom across the report, the words as the model sees them
  const RP_HEAD_COLORS = ['#4a3aa7', '#0891b2'];
  const RP_BINS = [[0, 0, 'itself'], [1, 1, '1 back'], [2, 2, '2'], [3, 4, '3–4'], [5, 8, '5–8'], [9, 16, '9–16'], [17, 32, '17–32'], [33, 64, '33–64'], [65, 128, '65–128'], [129, 1e9, '129+']];
  const RP_LABELLED = new Set(['FINDINGS', 'SPECIMEN', 'CLINICAL', 'GROSS', 'MICROSCOPIC', 'DIAGNOSIS', 'surface', 'urothelium', 'normal', 'reactive', 'atypia', 'denuded', 'nests', 'basement', 'membrane', 'absent', 'present', 'none', 'contours', 'rounded', 'irregular', 'stromal', 'desmoplasia', 'muscularis', 'propria', 'involved', 'inflammation', 'mild', 'marked', 'carcinoma', 'situ', 'invasive', 'lamina', 'Benign', 'Denuded', 'suspicious', 'foci', 'associated', 'Urothelial', 'Bladder', 'biopsy', 'Hematuria', 'cm', 'fragments', 'The', 'is', 'and', 'with', 'of', ',', '.', ':', '\n', RPR.END, RPR.START, '1', '2', '0']);
  const rpHeadName = (h, H) => (H === 2 ? (h === 0 ? 'near head' : 'far head') : `head ${h + 1}`);
  const rpWordLabel = w => (w === RPR.END ? '⟨end⟩' : w === RPR.START ? '⟨start⟩' : w === '\n' ? '↵' : w);
  function rpDefaultWord(words) { return LMW().defaultWord(words); } // the word the panels follow when none is hovered or kept: the reports' first word of the diagnosis, the chat's first word of the answer
  function rpView(onTest) { // what the diagrams draw from: the report on screen, or the report the model wrote
    const L = LM(); if (!L.model || !L.built) return null;
    if (onTest) return LMW().testView(L);
    const r = rpFocus(); return { words: r.words, tokens: r.tokens, fw: rpForward(r), textEl: $('rp-text'), pre: 'rp-', chosen: false, steps: null, nextEl: null };
  }
  function rpFocusWord(view) { const L = LM(), i = L.hover != null ? L.hover : L.pinned != null ? L.pinned : rpDefaultWord(view.words); return Math.max(1, Math.min(view.words.length - 1, i)); }
  function rpFollowWord(onTest) { // every panel follows one word: hovered, else kept, else the first word of the diagnosis
    const L = LM(), view = rpView(onTest); if (!view) return;
    const i = rpFocusWord(view), soft = L.hover == null && L.pinned == null;
    rpHighlight(view.textEl, view.fw, i, { soft });
    if (onTest && view.nextEl) { const s = view.steps.find(x => x.at === i); $(view.nextEl).innerHTML = s ? rpNextWordHtml(s.probs, null, s.token, `word ${view.steps.indexOf(s) + 1} it wrote, “${rpWordLabel(view.words[i])}”: what the model chose from`) : rpNextWordHtml(view.fw.probs[i - 1], view.tokens[i], null, `before “${rpWordLabel(view.words[i])}”, given in the prompt: what the model expected`); }
    else rpRenderNextWord(soft ? null : i);
    rpRenderNet(view); rpRenderHeads(view); rpRenderAttention(view, true);
  }
  // lines over the report, from the word followed to the words the model read for it
  function rpArcs(el, fw, i, row, q) {
    const L = LM(); let svg = el.querySelector('svg.rp-arcs');
    if (!svg) { svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('class', 'rp-arcs'); el.appendChild(svg); }
    if (i == null || !row || !L.arcs) { svg.innerHTML = ''; return; }
    const box = el.getBoundingClientRect(), W = el.clientWidth, Hh = el.clientHeight; svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`); svg.style.width = `${W}px`; svg.style.height = `${Hh}px`;
    const at = j => { const s = el.querySelector(`.tok[data-i="${j}"]`); if (!s) return null; const r = s.getBoundingClientRect(); return { x: r.left - box.left + r.width / 2, top: r.top - box.top, bottom: r.bottom - box.top }; };
    const to = at(i); if (!to) { svg.innerHTML = ''; return; }
    let mx = 1e-9; for (let j = 0; j <= q; j++) if (j !== i) mx = Math.max(mx, row[j]);
    const items = []; for (let j = 0; j <= q; j++) if (j !== i && row[j] >= 0.08 * mx) items.push({ j, rel: row[j] / mx });
    items.sort((a, b) => a.rel - b.rel);
    const line = (a, b, cls, w, o) => { const same = Math.abs(a.top - b.top) < 4, d = same ? `M${a.x.toFixed(1)} ${a.top.toFixed(1)} Q${((a.x + b.x) / 2).toFixed(1)} ${(a.top - 10 - Math.abs(b.x - a.x) * 0.12).toFixed(1)} ${b.x.toFixed(1)} ${b.top.toFixed(1)}` : `M${a.x.toFixed(1)} ${a.top.toFixed(1)} L${b.x.toFixed(1)} ${b.bottom.toFixed(1)}`; return `<path${cls ? ` class="${cls}"` : ''} d="${d}" style="stroke-width:${w.toFixed(1)};stroke-opacity:${o.toFixed(2)}"/>`; };
    let paths = items.slice(-12).map(({ j, rel }) => { const fr = at(j); return fr ? line(to, fr, '', 0.8 + 3 * rel, 0.25 + 0.65 * rel) : ''; }).join('');
    // two hops: from the words read most, what they had themselves read a layer earlier, dashed
    if (L.hops && fw && Math.min(L.layer, fw.ctxs.length - 1) > 0) {
      for (const { j, rel } of items.slice(-4)) {
        const row2 = rpAttentionRowAt(fw, Math.min(L.layer, fw.ctxs.length - 1) - 1, j); let m2 = 1e-9; for (let k = 0; k < j; k++) m2 = Math.max(m2, row2[k]);
        const second = []; for (let k = 0; k < j; k++) if (row2[k] >= 0.25 * m2) second.push({ k, rel2: row2[k] / m2 }); second.sort((a, b) => b.rel2 - a.rel2);
        const fr = at(j); if (!fr) continue;
        for (const { k, rel2 } of second.slice(0, 3)) { const to2 = at(k); if (!to2) continue; paths += line(fr, to2, 'hop2', 0.6 + 2.4 * rel * rel2, 0.3 + 0.5 * rel * rel2); const sp = el.querySelector(`.tok[data-i="${k}"]`); if (sp && k !== i && !sp.classList.contains('reads')) sp.classList.add('reads2'); }
      }
    }
    svg.innerHTML = paths;
  }
  function rpMarkPair(el, i, j) { el.querySelectorAll('.tok.att-q, .tok.att-k').forEach(s => s.classList.remove('att-q', 'att-k')); if (i != null) { const a = el.querySelector(`.tok[data-i="${i}"]`); if (a) a.classList.add('att-q'); } if (j != null) { const b = el.querySelector(`.tok[data-i="${j}"]`); if (b) b.classList.add('att-k'); } }
  // how the word was chosen: the numbers of position q = i − 1 on their way through the model, prepared for the drawing
  function rpRenderNet(view) {
    const L = LM(), m = L.model, i = rpFocusWord(view), q = i - 1, fw = view.fw, D = m.D, dk = m.dk, H = m.H, canvas = $(`${view.pre}net`), chosen = !!view.chosen && !!view.steps && view.steps.some(x => x.at === i); if (!canvas || q < 0 || q >= fw.tokens.length) return; // chosen: the model wrote this word; a word of the prompt was given, and the diagram says what the model expected there
    const tq = view.tokens[q], emb = m.E.subarray(tq * D, (tq + 1) * D), pos = m.Pos && q < m.P ? m.Pos.subarray(q * D, (q + 1) * D) : null;
    const lens = x => { const z = new Float64Array(m.V); let mx = -Infinity; for (let v = 0; v < m.V; v++) { let s = m.bout[v]; const off = v * D; for (let d = 0; d < D; d++) s += m.Wout[off + d] * x[d]; z[v] = s; mx = Math.max(mx, s); } let Z = 0; for (let v = 0; v < m.V; v++) { z[v] = Math.exp(z[v] - mx); Z += z[v]; } const idx = Array.from(z.keys()).sort((a, b) => z[b] - z[a]).slice(0, 3); return idx.map(v => ({ word: rpWordLabel(L.vocab.words[v]), p: z[v] / Z })); };
    const layers = fw.ctxs.map((c, l) => {
      const lay = m.layers[l], HM = H * dk, heard = new Float64Array(D); for (let d = 0; d < D; d++) { let s = lay.bo[d]; const off = d * HM; for (let e = 0; e < HM; e++) s += lay.Wo[off + e] * c.C[q][e]; heard[d] = s; }
      const heads = Array.from({ length: H }, (_, h) => { const row = c.Ah[h][q], reads = []; for (let j = 0; j <= q; j++) reads.push({ j, a: row[j] }); reads.sort((a, b) => b.a - a.a); return { name: rpHeadName(h, H), cost: c.costs[h], query: c.Q[q].subarray(h * dk, (h + 1) * dk), message: c.C[q].subarray(h * dk, (h + 1) * dk), reads: reads.slice(0, 5).filter(r => r.a > 0.005).map(r => { const rd = { j: r.j, a: r.a, word: rpWordLabel(view.words[r.j]), dist: q - r.j }; if (l > 0 && r.j > 0) rd.prior = rpTopReadsAvg(fw, l - 1, r.j, 3).map(p => `${rpWordLabel(view.words[p.j])} ${pct(p.a)}`).join(', '); return rd; }) }; });
      return { heads, heard, hidden: c.U[q], W: { Wq: lay.Wq, Wo: lay.Wo, W1: lay.W1, W2: lay.W2 }, after: c.Xp[q], out: c.Y[q], lens: lens(c.Y[q]) };
    });
    const probs = fw.probs[q], target = view.tokens[i], idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 6); if (!idx.includes(target)) idx.push(target);
    const model = { i, q, n: view.words.length, V: m.V, D, dk, word: rpWordLabel(view.words[i]), qword: rpWordLabel(view.words[q]), arch: `${m.V} words → vector of ${D}${m.P ? ' + position' : ''} → ${m.layers.length} × (${H} heads of ${dk} + feed-forward ${m.F}) → softmax over ${m.V} words`, emb, pos, x0: fw.X[q], lens0: lens(fw.X[q]), layers, out: idx.map(v => ({ v, word: rpWordLabel(L.vocab.words[v]), p: probs[v], target: v === target })), Wout: m.Wout, chosen, walk: L.netWalk && L.netWalk.pre === view.pre ? { at: L.netWalk.at, t: L.netWalk.t } : null, hover: L.hoverNet && L.hoverNet.pre === view.pre ? L.hoverNet : null };
    canvas._model = model; if (L.netView === 'graph') Viz.drawTokenGraph(canvas, model); else Viz.drawTokenNetwork(canvas, model);
    const title = $(`${view.pre}net-title`); if (title) title.textContent = `How “${model.word}” was ${chosen ? 'chosen' : 'expected'}${view.chosen && !chosen ? ' (given in the prompt)' : ''}`;
    const note = $(`${view.pre}net-note`); if (note) note.textContent = L.netView === 'graph'
      ? `The network for the word hovered or kept on the ${LMW().doc} (${LMW().defaultWordName} when none is). Every circle is one of the numbers the model holds for this word at that step, red positive and blue negative, scaled within its column. Every line is a learned weight of the network, the strongest drawn, red positive and blue negative: the weights are the same for every word of every report, and they are what training changes. The purple lines are the attention, this word's shares of the words it read, computed from its queries and the other words' keys for this very report, so they change from word to word. Read left to right: the input, then for each layer the queries, the words read, the messages, what the layer hears, the add, the feed-forward, the add, and last the output layer's scores for the most probable next words. Walk through lights the columns one by one; hover a circle for its number, a read word to find it on the report.`
      : `The same pass as strips of numbers. Standing at “${model.qword}”, the model adds that word's vector to its position vector; in each layer the near head, with its steep distance cost, reads the last few words, and the far head, with a shallow one, can reach back to the findings; the heads' messages are projected and added to the vector, never swapped for it, and the feed-forward is added too; the output layer scores all ${m.V} words. The "read off" lines show what the output layer would say at each stage: the prediction takes shape layer by layer.`;
  }
  // who reads whom across the whole report, in the chosen layer: one matrix per head, the attention by distance, where the words of each section look
  function rpRenderAttention(view, light) {
    const L = LM(), m = L.model, fw = view.fw, n = view.words.length, l = Math.min(L.layer, fw.ctxs.length - 1), c = fw.ctxs[l], H = m.H, pre = view.pre, WD = LMW(), sections = WD.sectionsOf(view.words), q = rpFocusWord(view) - 1;
    rpEnsureAttBody(pre, H);
    for (let h = 0; h < H; h++) { const hv = L.hoverAtt && L.hoverAtt.pre === pre && L.hoverAtt.h === h ? L.hoverAtt : null; Viz.drawTokenAttention($(`${pre}att-${h}`), { A: c.Ah[h], n, sections, colors: WD.colors, mark: q, hover: hv }); $(`${pre}att-${h}-title`).textContent = `${rpHeadName(h, H)}, layer ${l + 1} · cost ${c.costs[h].toFixed(3)} per word of distance`; }
    if (light) return;
    const series = Array.from({ length: H }, (_, h) => { const vals = RP_BINS.map(() => 0); for (let i = 0; i < n; i++) { const row = c.Ah[h][i]; for (let j = 0; j <= i; j++) { const d = i - j; let b = 0; while (d > RP_BINS[b][1]) b++; vals[b] += row[j] / n; } } return { name: rpHeadName(h, H), color: RP_HEAD_COLORS[h % RP_HEAD_COLORS.length], values: vals }; });
    Viz.drawGroupedBars($(`${pre}att-dist`), { labels: RP_BINS.map(b => b[2]), series, pct: true, xLabel: 'words back from the one asking' });
    $(`${pre}att-dist-legend`).innerHTML = series.map(s => `<span><span class="swatch" style="background:${s.color}"></span>${s.name}</span>`).join('');
    for (let h = 0; h < H; h++) { $(`${pre}sec-${h}`).innerHTML = rpSectionTable(rpSectionMatrix(c.Ah[h], sections, n)); $(`${pre}sec-${h}-title`).textContent = `${rpHeadName(h, H)} · where a word of each section looks`; }
    const note = $(`${pre}att-note`); if (note) note.textContent = WD.attentionNote(n);
  }
  function rpSectionMatrix(A, sections, n) { const S6 = LMW().sections, idx = new Map(S6.map((s, k) => [s, k])), M = S6.map(() => S6.map(() => 0)), cnt = S6.map(() => 0); for (let i = 0; i < n; i++) { const r = idx.get(sections[i]); cnt[r]++; const row = A[i]; for (let j = 0; j <= i; j++) M[r][idx.get(sections[j])] += row[j]; } return { M: M.map((row, r) => row.map(v => (cnt[r] ? v / cnt[r] : 0))), cnt }; }
  function rpSectionTable(sm) { const S6 = LMW().sections; return `<tr><th class="row">a word in …</th>${S6.map(s => `<th>reads ${s}</th>`).join('')}</tr>` + S6.map((s, r) => `<tr><th class="row">${s} <span class="n">(${sm.cnt[r]})</span></th>${S6.map((t, k) => { const v = sm.M[r][k]; return `<td style="background:${Viz.sequential(Math.sqrt(v), 0.8)}${v > 0.5 ? ';color:#fff' : ''}">${v >= 0.005 ? pct(v) : '·'}</td>`; }).join('')}</tr>`).join(''); }
  function rpAttBodyHtml(pre, H) {
    return `<div class="rp-att">${Array.from({ length: H }, (_, h) => `<div><div class="rp-ttl" id="${pre}att-${h}-title">${rpHeadName(h, H)}</div><div class="net-wrap"><canvas id="${pre}att-${h}" width="320" height="320" aria-label="${rpHeadName(h, H)}: the attention between the words of the report, rows ask, columns answer"></canvas><div class="tip" id="${pre}att-${h}-tip" hidden></div></div></div>`).join('')}<div class="chart"><div class="title"><span>How far back each head reads</span></div><svg id="${pre}att-dist" role="img" aria-label="Each head's attention by distance back, averaged over the words of the report"></svg><div class="legend" id="${pre}att-dist-legend" style="margin-top:6px"></div></div></div><div class="rp-sec-grid">${Array.from({ length: H }, (_, h) => `<div><div class="rp-ttl" id="${pre}sec-${h}-title">${rpHeadName(h, H)}</div><table class="rp-sec" id="${pre}sec-${h}"></table></div>`).join('')}</div>`;
  }
  function rpEnsureAttBody(pre, H) { // the matrices, the chart and the tables for H heads, built once, their canvases wired
    const body = $(`${pre}att-body`); if (!body || +body.dataset.heads === H) return; body.dataset.heads = H; body.innerHTML = rpAttBodyHtml(pre, H);
    const L = LM(), onTest = pre !== 'rp-';
    for (let h = 0; h < H; h++) rpBindTip($(`${pre}att-${h}`), $(`${pre}att-${h}-tip`), (cv, x, y) => { const hit = Viz.hitTokenAttention(cv, x, y), view = rpView(onTest); if (!hit || !view) return null; const l = Math.min(L.layer, view.fw.ctxs.length - 1), a = view.fw.ctxs[l].Ah[h][hit.i][hit.j]; hit.text = `“${rpWordLabel(view.words[hit.i])}” (word ${hit.i}) reads “${rpWordLabel(view.words[hit.j])}” (word ${hit.j}, ${hit.i - hit.j} back): ${(a * 100).toFixed(1)}%`; return hit; }, hit => { L.hoverAtt = hit ? { pre, h, i: hit.i, j: hit.j } : null; const view = rpView(onTest); if (!view) return; rpRenderAttention(view, true); rpMarkPair(view.textEl, hit ? hit.i : null, hit ? hit.j : null); });
  }
  function rpBindTip(canvas, tip, hit, onChange) { // a canvas with a tooltip: hit(canvas, x, y) → { …, text } or null; onChange when what is under the mouse changes
    const keyOf = h => (h ? `${h.kind || ''}:${h.l == null ? '' : h.l}:${h.h == null ? '' : h.h}:${h.k == null ? '' : h.k}:${h.d == null ? '' : h.d}:${h.i == null ? '' : h.i}:${h.j == null ? '' : h.j}` : '');
    canvas.addEventListener('mousemove', ev => { const r = canvas.getBoundingClientRect(), h = hit(canvas, ev.clientX - r.left, ev.clientY - r.top), key = keyOf(h); if (key !== (canvas._hitKey || '')) { canvas._hitKey = key; onChange(h); } if (h && h.text) { tip.hidden = false; tip.textContent = h.text; const half = tip.offsetWidth / 2 + 4; tip.style.left = Math.max(half, Math.min(r.width - half, ev.clientX - r.left)) + 'px'; tip.style.top = (ev.clientY - r.top) + 'px'; } else tip.hidden = true; });
    canvas.addEventListener('mouseleave', () => { tip.hidden = true; if (canvas._hitKey) { canvas._hitKey = ''; onChange(null); } });
  }
  // the words as the model sees them: the two directions of the D that spread the V word vectors most (power iteration
  // on their covariance, the second with the first taken out); the signs follow the previous axes, so that the map does
  // not flip between renders
  function rpPca2(E, V, D, prev) {
    const mean = new Float64Array(D); for (let v = 0; v < V; v++) for (let d = 0; d < D; d++) mean[d] += E[v * D + d] / V;
    const C = new Float64Array(D * D), x = new Float64Array(D);
    for (let v = 0; v < V; v++) { for (let d = 0; d < D; d++) x[d] = E[v * D + d] - mean[d]; for (let a = 0; a < D; a++) { const xa = x[a] / V; if (!xa) continue; const off = a * D; for (let b = 0; b < D; b++) C[off + b] += xa * x[b]; } }
    const axis = (deflate, seed) => { let u = Float64Array.from({ length: D }, (_, d) => Math.cos(d * 1.7 + seed)); for (let it = 0; it < 80; it++) { const w = new Float64Array(D); for (let a = 0; a < D; a++) { let s = 0; const off = a * D; for (let b = 0; b < D; b++) s += C[off + b] * u[b]; w[a] = s; } if (deflate) { let dot = 0; for (let d = 0; d < D; d++) dot += w[d] * deflate[d]; for (let d = 0; d < D; d++) w[d] -= dot * deflate[d]; } let nrm = 0; for (const v of w) nrm += v * v; nrm = Math.sqrt(nrm) || 1; u = Float64Array.from(w, v => v / nrm); } return u; };
    const u1 = axis(null, 0), u2 = axis(u1, 1);
    for (const [u, p] of [[u1, prev && prev[0]], [u2, prev && prev[1]]]) if (p) { let dot = 0; for (let d = 0; d < D; d++) dot += u[d] * p[d]; if (dot < 0) for (let d = 0; d < D; d++) u[d] = -u[d]; }
    const lam = u => { let s = 0; for (let a = 0; a < D; a++) { let t = 0; const off = a * D; for (let b = 0; b < D; b++) t += C[off + b] * u[b]; s += u[a] * t; } return s; };
    let trace = 0; for (let d = 0; d < D; d++) trace += C[d * D + d];
    const xs = new Float64Array(V), ys = new Float64Array(V); for (let v = 0; v < V; v++) { let sx = 0, sy = 0; for (let d = 0; d < D; d++) { const e = E[v * D + d] - mean[d]; sx += e * u1[d]; sy += e * u2[d]; } xs[v] = sx; ys[v] = sy; }
    return { x: xs, y: ys, axes: [u1, u2], share: trace > 0 ? (lam(u1) + lam(u2)) / trace : 0 };
  }
  // ---- how each head decides where to look, from the followed word's position, in the chosen layer
  function rpRenderHeads(view) {
    const L = LM(), m = L.model, fw = view.fw, i = rpFocusWord(view), q = i - 1, l = Math.min(L.layer, fw.ctxs.length - 1), c = fw.ctxs[l], dk = m.dk, sc = 1 / Math.sqrt(dk), n = q + 1, canvas = $(`${view.pre}heads`); if (!canvas || q < 0) return;
    const sections = LMW().sectionsOf(view.words), words = view.words.slice(0, n).map(rpWordLabel);
    const heads = Array.from({ length: m.H }, (_, h) => {
      const o = h * dk, cost = c.costs[h], match = new Float64Array(n), costs = new Float64Array(n), score = new Float64Array(n), share = Float64Array.from(c.Ah[h][q].subarray(0, n)), matchOnly = new Float64Array(n); let mx = -Infinity;
      for (let j = 0; j < n; j++) { let v = 0; for (let d = 0; d < dk; d++) v += c.Q[q][o + d] * c.K[j][o + d]; match[j] = v * sc; costs[j] = cost * (q - j); score[j] = match[j] - costs[j]; mx = Math.max(mx, match[j]); }
      let Z = 0; for (let j = 0; j < n; j++) { matchOnly[j] = Math.exp(match[j] - mx); Z += matchOnly[j]; } for (let j = 0; j < n; j++) matchOnly[j] /= Z;
      const at50 = cost * 50;
      return { name: rpHeadName(h, m.H), cost, match, costs, score, share, matchOnly, note: `${at50.toFixed(1)} nats off a word 50 back, so ${at50 > 2 ? 'only the words just before can win' : 'a strong match anywhere can win'}` };
    });
    const model = { q, qword: rpWordLabel(view.words[q]), word: rpWordLabel(view.words[i]), words, sections: sections.slice(0, n), colors: LMW().colors, heads, hover: L.hoverHeads && L.hoverHeads.pre === view.pre ? L.hoverHeads : null };
    canvas._model = model; Viz.drawHeadRows(canvas, model);
    const title = $(`${view.pre}heads-title`); if (title) title.textContent = `How each head decides where to look, from “${model.qword}” · layer ${l + 1}`;
    const note = $(`${view.pre}heads-note`); if (note) note.textContent = `A head is one question put to every word so far. It has its own query map (what the asking word looks for), key map (how each earlier word presents itself) and value map (what a word says when read). Query · key is the match; the head subtracts its learned cost × the distance back; the softmax over the words so far turns the scores into shares that add up to 100%, and the values, weighted by their shares, add up to the head's message. The heads of a layer ask at the same time and their messages are joined and added to the asking word's vector. The two heads here differ in one learned number, the cost per word: the near head's is steep, so a good match a dozen words back has already lost several nats and only the words just before can win; the far head's is nearly flat, so its matches count wherever they stand, back to the findings block. The grey bars behind the shares are what the match alone would give: the difference is the cost at work.`;
  }
  // the network lit column by column: a walk through the pass for the word followed
  const RP_WALK_MS = 900;
  function rpNetWalkStart(pre) { const L = LM(); if (L.netWalk) { rpNetWalkStop(); return; } L.netView = 'graph'; rpSyncControls(); L.netWalk = { pre, at: 0, t: 0, t0: performance.now() }; const b = $(`${pre}net-walk`); if (b) b.textContent = '■ Stop'; requestAnimationFrame(rpNetWalkTick); }
  function rpNetWalkStop() { const L = LM(), wk = L.netWalk; if (!wk) return; L.netWalk = null; const b = $(`${wk.pre}net-walk`); if (b) b.textContent = '▶ Walk through'; const view = rpView(wk.pre === 'rp-test-'); if (view) rpRenderNet(view); }
  function rpNetWalkTick(now) { const L = LM(), wk = L.netWalk; if (!wk) return; const view = rpView(wk.pre !== 'rp-'), canvas = $(`${wk.pre}net`); if (!view || !canvas) { rpNetWalkStop(); return; } const el = (now - wk.t0) / RP_WALK_MS, nCols = canvas._tokenGraph ? canvas._tokenGraph.cols.length : 16; wk.at = Math.floor(el); wk.t = el - wk.at; if (wk.at > nCols) { rpNetWalkStop(); return; } rpRenderNet(view); requestAnimationFrame(rpNetWalkTick); }
  // the words a position read most, in one layer, both heads averaged, itself left out
  function rpTopReadsAvg(fw, layer, i, k) { const c = fw.ctxs[layer], row = new Float64Array(i + 1); for (let h = 0; h < c.Ah.length; h++) for (let j = 0; j <= i; j++) row[j] += c.Ah[h][i][j] / c.Ah.length; const idx = []; for (let j = 0; j < i; j++) idx.push(j); idx.sort((a, b) => row[b] - row[a]); return idx.slice(0, k).map(j => ({ j, a: row[j] })); }
  // ---- the word map laid out by t-SNE: Gaussian affinities in the model's space (each word's width set to a perplexity
  // of 20, on cosine distances), the layout started from the PCA map and pushed a few iterations per frame, so that the
  // page stays alive; after a training step it carries on from where it was
  const RP_MAP_PERPLEXITY = 20, RP_MAP_ITERS = 400;
  function rpMapVectors() { const L = LM(), m = L.model; return L.map.source === 'out' ? m.Wout : m.E; } // the word as an input (its embedding) or as a prediction (its row of the output layer)
  function rpMapAffinities(E, V, D, perp) {
    const N = new Float64Array(V * D); for (let v = 0; v < V; v++) { let s = 0; for (let d = 0; d < D; d++) s += E[v * D + d] ** 2; s = Math.sqrt(s) || 1; for (let d = 0; d < D; d++) N[v * D + d] = E[v * D + d] / s; } // unit vectors: cosine distances
    const P = new Float64Array(V * V), d2 = new Float64Array(V), logU = Math.log(perp), p = new Float64Array(V);
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) { let dot = 0; for (let d = 0; d < D; d++) dot += N[i * D + d] * N[j * D + d]; d2[j] = 2 - 2 * dot; }
      let lo = 0, hi = Infinity, beta = 1;
      for (let it = 0; it < 60; it++) {
        let Z = 0; for (let j = 0; j < V; j++) { p[j] = j === i ? 0 : Math.exp(-d2[j] * beta); Z += p[j]; }
        let H = 0; for (let j = 0; j < V; j++) if (p[j] > 0) { p[j] /= Z; H -= p[j] * Math.log(p[j]); }
        const diff = H - logU; if (Math.abs(diff) < 1e-4) break;
        if (diff > 0) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
      }
      for (let j = 0; j < V; j++) P[i * V + j] = p[j];
    }
    const Sy = new Float64Array(V * V); for (let i = 0; i < V; i++) for (let j = 0; j < V; j++) Sy[i * V + j] = (P[i * V + j] + P[j * V + i]) / (2 * V);
    return Sy;
  }
  function rpMapStep(M, iters) { // gradient steps on the layout, with the usual early exaggeration, momentum and gains
    const V = M.V, Y = M.Y, P = M.P, dY = M.dY, g = M.gains, grad = M.grad, num = M.num, eta = 100;
    for (let it = 0; it < iters; it++) {
      const ex = M.iter < 100 ? 4 : 1, mom = M.iter < 250 ? 0.5 : 0.8;
      let Z = 0; for (let i = 0; i < V; i++) for (let j = i + 1; j < V; j++) { const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], q = 1 / (1 + dx * dx + dy * dy); num[i * V + j] = q; num[j * V + i] = q; Z += 2 * q; }
      grad.fill(0);
      for (let i = 0; i < V; i++) { let gx = 0, gy = 0; const yi = Y[2 * i], yi1 = Y[2 * i + 1]; for (let j = 0; j < V; j++) { if (j === i) continue; const q = num[i * V + j], f = (ex * P[i * V + j] - q / Z) * q; gx += f * (yi - Y[2 * j]); gy += f * (yi1 - Y[2 * j + 1]); } grad[2 * i] = 4 * gx; grad[2 * i + 1] = 4 * gy; }
      let cx = 0, cy = 0;
      for (let k = 0; k < 2 * V; k++) { g[k] = Math.max(0.01, (grad[k] > 0) !== (dY[k] > 0) ? g[k] + 0.2 : g[k] * 0.8); dY[k] = mom * dY[k] - eta * g[k] * grad[k]; Y[k] += dY[k]; if (k & 1) cy += Y[k]; else cx += Y[k]; }
      cx /= V; cy /= V; for (let i = 0; i < V; i++) { Y[2 * i] -= cx; Y[2 * i + 1] -= cy; }
      M.iter++;
    }
  }
  function rpMapRestart(full) { // the layout from the PCA map (full) or carried on from where it was, with fresh affinities
    const L = LM(), m = L.model, M = L.map, V = m.V, D = m.D; if (!m || M.mode !== 'tsne') return;
    const E = rpMapVectors(); full = full || M.forModel !== m || M.V !== V || !M.Y; M.forModel = m; M.V = V;
    M.P = rpMapAffinities(E, V, D, RP_MAP_PERPLEXITY);
    if (full) { const pca = rpPca2(E, V, D, null); let s2 = 0; for (let v = 0; v < V; v++) s2 += pca.x[v] ** 2 + pca.y[v] ** 2; const sc = Math.sqrt(s2 / (2 * V)) || 1; M.Y = new Float64Array(2 * V); for (let v = 0; v < V; v++) { M.Y[2 * v] = pca.x[v] / sc * 1e-2; M.Y[2 * v + 1] = pca.y[v] / sc * 1e-2; } M.dY = new Float64Array(2 * V); M.gains = new Float64Array(2 * V).fill(1); M.grad = new Float64Array(2 * V); M.num = new Float64Array(V * V); M.iter = 0; M.todo = RP_MAP_ITERS; }
    else { M.todo = Math.max(M.todo, 80); if (M.iter < 100) M.iter = 100; }
    if (!M.running) { M.running = true; requestAnimationFrame(() => rpMapTick(L)); }
  }
  function rpMapTick(L) { // the map loop of one world's state; it draws only while that world is on screen
    const M = L.map; if (!M.running) return;
    if (M.mode !== 'tsne' || M.todo <= 0 || !L.model) { M.running = false; if (M.mode === 'tsne' && L.model && L === LM()) rpDrawWordMap(); return; }
    const t0 = performance.now(); while (M.todo > 0 && performance.now() - t0 < 40) { rpMapStep(M, 4); M.todo -= 4; }
    if (L === LM()) rpDrawWordMap(); requestAnimationFrame(() => rpMapTick(L));
  }
  function rpRenderWordMap() {
    const L = LM(), m = L.model, M = L.map; if (!$('rp-map') || !m || !L.wordSec) return;
    if (M.mode === 'tsne') { rpMapRestart(false); return; } // the frames draw it
    const E = rpMapVectors(), pca = rpPca2(E, m.V, m.D, L.pcaAxes); L.pcaAxes = pca.axes; M.pcaShare = pca.share; M.pcaX = pca.x; M.pcaY = pca.y;
    rpDrawWordMap();
  }
  function rpDrawWordMap() {
    const L = LM(), WD = LMW(), m = L.model, M = L.map, tsne = M.mode === 'tsne', xs = v => (tsne ? M.Y[2 * v] : M.pcaX[v]), ys = v => (tsne ? M.Y[2 * v + 1] : M.pcaY[v]); if (!(tsne ? M.Y : M.pcaX)) return;
    const pts = L.vocab.words.map((w, v) => ({ id: v, word: rpWordLabel(w), note: L.wordSec[v] ? `mostly in the ${L.wordSec[v]}` : 'never in the training reports', x: xs(v), y: ys(v), color: L.wordSec[v] ? WD.colors[L.wordSec[v]] : Viz.colors().ink3, label: WD.labelled().has(w) }));
    Viz.drawWordMap($('rp-map'), tsne ? { points: pts, W: 640, H: 400, axes: false, xLabel: `t-SNE${M.todo > 0 ? ` · settling, ${M.todo} steps to go` : ''}` } : { points: pts, W: 640, H: 400 });
    $('rp-map-legend').innerHTML = WD.sections.map(s => `<span><span class="swatch" style="background:${WD.colors[s]}"></span>mostly in the ${s}</span>`).join('');
    const what = M.source === 'out' ? `its row of the output layer, the ${m.D} numbers the model scores it with as the next word` : `its vector at the input, ${m.D} learned numbers`;
    $('rp-map-note').textContent = tsne
      ? `Every word is ${what}; t-SNE lays the ${m.V} words out so that each keeps its neighbours from the ${m.D}-dimensional space (cosine distances, perplexity ${RP_MAP_PERPLEXITY}), starting from the two-direction map; clusters are real neighbourhoods, the distances between clusters mean little, and the layout carries on from where it was after each training step. Each word is coloured by the section it mostly appears in on the training reports. Hover a dot for its word.`
      : `Every word is ${what}; the map is the two directions that spread the ${m.V} vectors most (${pct(M.pcaShare || 0)} of their spread), each word coloured by the section it mostly appears in on the training reports. Untrained, the vectors are random and the map a cloud; as the model trains, words used alike move together: ${WD.mapExample}. Hover a dot for its word.`;
  }
  // ------------------------------------------------------------------ the chat question: Specimens and Test (the Train step is the shared language-model lab)
  const CH_LABELLED = new Set(['SYSTEM', 'USER', 'ASSISTANT', 'TOOL', 'FINDINGS', 'Yes', 'No', 'Suspicious', 'findings', '(', ')', 'briefly', 'brief', 'full', 'sentence', 'reason', 'diagnosis', 'invasion', 'situ', 'carcinoma', 'present', 'absent', 'involved', 'rounded', 'irregular', 'normal', 'reactive', 'atypia', 'denuded', 'mild', 'marked', 'none', 'Benign', 'Denuded', 'Urothelial', 'muscularis', 'propria', 'inflammation', 'contours', 'surface', 'nests', 'question', 'Here', 'Is', 'What', '?', '.', ':', '\n', RPR.END, RPR.START]);
  // ------------------------------------------------------------------ the chat question: the transcripts, and asking the model
  // The chat world shares the Train step and the model's diagrams with the reports (LM_WORLDS above); here are its own two
  // steps. Specimens shows the transcripts with the role of every token, the explainer and the phrasings; Test is the chat
  // box: a system line (a trained phrasing, the held-out one, a line never seen, or one's own words), a case, a question
  // (one of the eight, one never asked, or one's own), and the model continues the transcript from ASSISTANT: on. In the
  // agent style the page is the program of the loop: when the assistant's line is findings(), it pastes the findings back
  // as a TOOL turn and lets the model go on.
  const CH_MAX_LINE = 40; // words of one assistant line at most
  const chStyleName = s => (CHR.STYLES[s] ? CHR.STYLES[s].name : s);
  const chSplitName = r => (r.split === 'train' ? 'training' : r.split === 'test' ? 'test' : 'held-out phrasing');
  const chLineName = r => (r.line === 'held' ? 'the phrasing held out of training' : `phrasing ${r.line + 1} of ${CHR.STYLES[r.style].lines.length}`);
  function chTrays(kind) { // the chat panels' trays, built once per set
    const L = S.ch, key = `ch${kind}`; if (L.trayFor[key]) return; L.trayFor[key] = true;
    const map = new Map(), fill = (id, tray, onClick) => { const el = $(id); el.innerHTML = ''; for (const r of tray) { const b = rpThumb(r, onClick); el.appendChild(b); map.set(r.id, b); } };
    if (kind === 'data') { fill('ch-data-train-tray', L.train, id => chDataSelect(id)); fill('ch-data-test-tray', L.test, id => chDataSelect(id)); fill('ch-data-held-tray', L.held, id => chDataSelect(id)); L.dataThumbs = map; $('ch-data-train-label').textContent = `Training transcripts (${L.train.length})`; $('ch-data-test-label').textContent = `Test transcripts (${L.test.length}) · held out, on the trained phrasings`; $('ch-data-held-label').textContent = `Held-out phrasings (${L.held.length}) · the same cases and questions, the fourth phrasing of every instruction`; }
    else { const set = chCaseSet(); fill('ch-test-results', set, chTestSelect); L.testThumbs = map; $('ch-test-count').textContent = `${set.length} transcripts`; }
  }
  // ---- Specimens: the transcripts, the roles, the explainer's table of phrasings
  function chEnterData() { const L = S.ch; if (!LMW().data() || !rpBuild()) return; chTrays('data'); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id; chRenderPhrasings(); chRenderData(); }
  function chDataSelect(id, scroll) { const L = S.ch; L.dataSelected = id; chRenderData(); if (scroll) { const el = L.dataThumbs && L.dataThumbs.get(id); if (el) el.scrollIntoView({ block: 'nearest' }); } }
  function chDataNeighbour(step) { const L = S.ch, i = L.all.findIndex(r => r.id === L.dataSelected); chDataSelect(L.all[(i + step + L.all.length) % L.all.length].id, true); }
  function chRenderData() {
    const L = S.ch; chTrays('data'); if (L.dataSelected == null || !L.byId.get(L.dataSelected)) L.dataSelected = L.train[0].id;
    const r = L.byId.get(L.dataSelected) || L.train[0], M = L.meta;
    $('ch-data-title').textContent = M.question; $('ch-data-sub').textContent = `${L.train.length} training, ${L.test.length} test and ${L.held.length} held-out-phrasing transcripts · every word a token · vocabulary of ${L.vocab.size} · the tokenizer of the reports`;
    $('ch-data-blurb').textContent = M.blurb;
    $('ch-data-transcript-title').textContent = `Transcript ${r.name} · ${chSplitName(r)} · ${chStyleName(r.style)} · “${r.q}”`;
    $('ch-data-text').innerHTML = rpTextHtml(r.words, null, null);
    $('ch-data-roles').innerHTML = CHR.ROLES.map(k => `<span><span class="swatch" style="background:${CH.colors[k]}"></span>${k}</span>`).join('');
    const line = r.line === 'held' ? CHR.STYLES[r.style].held : CHR.STYLES[r.style].lines[r.line];
    $('ch-data-case').innerHTML = `<div class="row"><span class="k">style</span><span class="v">${chStyleName(r.style)}${r.style === 'brief' ? ': a word or two' : r.style === 'full' ? ': a sentence with the reason' : ': the question alone, findings() first'}</span></div><div class="row"><span class="k">instruction</span><span class="v">${esc(line)} (${chLineName(r)})</span></div><div class="row"><span class="k">question</span><span class="v">${esc(r.q)}</span></div>`
      + (L.reveal ? `<div class="row"><span class="k">the case</span><span class="v">${esc(r.case)} of the report corpus · ${esc(r.dx)}</span></div><div class="dx">The rule answers: <b>${esc(r.answer)}</b></div>` : '<div class="row"><span class="k">the rule’s answer</span><span class="v">tick Reveal</span></div>');
    $('ch-data-note').textContent = `A transcript: the system line (the persona and an instruction), the user’s turn and the assistant’s, every word a token and every token given a role by where it stands. The model trains on all of it, the next word alone, and on the Test step continues a transcript from ASSISTANT: on. ${L.reveal ? 'The rule’s answer is what the generator wrote for the assistant; the model never sees the rule, only the words.' : 'Tick Reveal to see the case and the rule’s answer.'} Click a transcript below, or step through them.`;
    for (const [id, el] of L.dataThumbs) el.classList.toggle('selected', id === L.dataSelected);
  }
  function chRenderPhrasings() {
    const el = $('ch-phrasings'); if (el.childElementCount) return;
    el.innerHTML = `<table><thead><tr><th>style</th><th>the answer</th><th>trained phrasings of the instruction</th><th>held out of training</th></tr></thead><tbody>${Object.values(CHR.STYLES).map(st => `<tr><td>${esc(st.name)}</td><td>${st.key === 'brief' ? 'a word or two' : st.key === 'full' ? 'a full sentence with the reason' : 'findings() first, then a word or two'}</td><td>${st.lines.map(l => `“${esc(l)}”`).join('<br>')}</td><td>“${esc(st.held)}”</td></tr>`).join('')}</tbody></table><p class="small muted" style="margin-top:8px">Every system line starts with the persona, “${esc(CHR.PERSONA)}”. The eight questions: ${CHR.QUESTIONS.map(q => `“${esc(q.q)}”`).join(', ')}. Kept for the Test step, never trained on: the questions ${CHR.PROBES.questions.map(q => `“${esc(q)}”`).join(', ')}, and the system lines ${CHR.PROBES.lines.map(q => `“${esc(q)}”`).join(', ')}.</p>`;
  }
  // ---- Test: the chat box, the agent loop, the probes
  const chCaseSet = () => (S.ch.cases === 'held' ? S.ch.held : S.ch.test);
  function chChat() { const L = S.ch; if (!L.chat) L.chat = { style: 'brief', line: 0, lineText: '', caseId: null, qi: 0, qText: '', docId: null, view: null }; return L.chat; }
  function chCases() { const L = S.ch; if (!L.caseIds) { L.blocks = new Map(); for (const r of L.all) if (!L.blocks.has(r.case)) L.blocks.set(r.case, CHR.blockOfText(r.text)); L.caseIds = [...new Set(L.test.map(r => r.case))]; } return L.caseIds; } // the test cases, the same for both sets
  function chEnterTest() { const L = S.ch; if (!LMW().data() || !rpBuild()) return; if (!L.model) rpReset(); L.hover = null; L.pinned = null; L.hoverNet = null; L.hoverAtt = null; chTrays('test'); if (!chChat().caseId) chLoadDoc(chCaseSet()[Math.min(L.trial.next, chCaseSet().length - 1)]); chRenderTest(); }
  function chLoadDoc(r) { // a transcript as written: its style, its instruction, its case and its question
    const L = S.ch, C = chChat(); if (L.writing) chWriteStop(); C.style = r.style; C.line = r.line; C.caseId = r.case; C.qi = r.qi; C.docId = r.id; C.view = null; L.written = null; L.truthShown = false; L.testSelected = r.id;
  }
  function chDetach() { const L = S.ch, C = chChat(); if (L.writing) chWriteStop(); C.docId = null; C.view = null; L.testSelected = null; L.written = null; L.truthShown = false; } // a prompt of one's own: not a transcript, not tallied
  function chTestSelect(id) { const L = S.ch, r = L.byId.get(id); if (!r) return; chLoadDoc(r); const res = L.trial.results.get(id); if (res) { L.written = res; chChat().view = res.view; L.truthShown = true; } chRenderTest(); }
  function chNextDoc() { const L = S.ch, set = chCaseSet(), i = L.trial.next; if (i >= set.length) { rpNote(`All ${set.length} transcripts asked. Clear to start over, or make a prompt of your own.`); return null; } chLoadDoc(set[i]); chRenderTest(); return set[i]; }
  function chAskNext() { const L = S.ch; if (L.writing || L.batchWriting) return; const r = chNextDoc(); if (r) chAsk(); }
  const chLineText = () => { const C = chChat(), st = CHR.STYLES[C.style]; return C.line === 'own' ? C.lineText.trim() : C.line === 'held' ? st.held : typeof C.line === 'string' ? CHR.PROBES.lines[+C.line.slice(1)] : st.lines[C.line]; };
  const chQuestionText = () => { const C = chChat(); return C.qi === 'own' ? C.qText.trim() : typeof C.qi === 'string' ? CHR.PROBES.questions[+C.qi.slice(1)] : CHR.QUESTIONS[C.qi].q; };
  const chLineKind = () => { const C = chChat(); return C.line === 'own' ? 'own' : C.line === 'held' ? 'held' : typeof C.line === 'string' ? 'probe' : 'trained'; };
  const chQuestionKind = () => { const C = chChat(); return C.qi === 'own' ? 'own' : typeof C.qi === 'string' ? 'probe' : 'trained'; };
  function chBlock() { chCases(); return S.ch.blocks.get(chChat().caseId) || ''; }
  function chPromptText() { const C = chChat(); return CHR.promptOf(C.style, chLineText(), chBlock(), chQuestionText()); }
  function chPrefix() { return RPR.encode(S.ch.vocab, chPromptText()).slice(0, -1); } // the start token and the prompt, without the end token
  function chUnknown() { const L = S.ch; return RPR.tokenize(chPromptText()).filter(w => !L.vocab.index.has(w)); } // the words of the prompt the model has no token for
  function chSyncControls() {
    const L = S.ch, C = chChat(), st = CHR.STYLES[C.style], lineSel = $('ch-line'), qSel = $('ch-question'), opt = (v, t) => `<option value="${v}">${esc(t)}</option>`;
    const lines = `<optgroup label="trained phrasings">${st.lines.map((l, i) => opt(i, l)).join('')}</optgroup><optgroup label="held out of training">${opt('held', st.held)}</optgroup><optgroup label="never seen as a system line">${CHR.PROBES.lines.map((l, i) => opt('p' + i, l)).join('')}</optgroup><optgroup label="your own">${opt('own', 'your own words…')}</optgroup>`;
    if (lineSel.dataset.style !== C.style) { lineSel.innerHTML = lines; lineSel.dataset.style = C.style; }
    if (![...lineSel.options].some(o => o.value === String(C.line))) C.line = 0;
    lineSel.value = String(C.line); $('ch-line-text').hidden = C.line !== 'own'; if ($('ch-line-text').value !== C.lineText) $('ch-line-text').value = C.lineText;
    if (!qSel.childElementCount) qSel.innerHTML = `<optgroup label="the eight questions">${CHR.QUESTIONS.map((q, i) => opt(i, q.q)).join('')}</optgroup><optgroup label="never asked in training">${CHR.PROBES.questions.map((q, i) => opt('p' + i, q)).join('')}</optgroup><optgroup label="your own">${opt('own', 'your own words…')}</optgroup>`;
    qSel.value = String(C.qi); $('ch-question-text').hidden = C.qi !== 'own'; if ($('ch-question-text').value !== C.qText) $('ch-question-text').value = C.qText;
    document.querySelectorAll('#ch-style-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.style === C.style));
    document.querySelectorAll('#ch-cases-seg button').forEach(b => b.classList.toggle('is-active', b.dataset.cases === L.cases));
    $('ch-temp').value = Math.round(L.temperature * 100); $('ch-temp-val').textContent = L.temperature.toFixed(2); $('ch-pace').value = L.pace; $('ch-pace-val').textContent = `${L.pace} words/s`;
  }
  function chAsk() { // the model continues the transcript from ASSISTANT: on, a few words per frame at the chosen pace
    const L = S.ch, C = chChat(); if (!L.model) return; if (L.writing) { chWriteStop(); return; }
    const prefix = chPrefix(), st = L.model.genState(prefix);
    L.writing = { prefix, st, tokens: st.tokens, steps: [], events: [], t0: performance.now(), count: 0, lineLen: 0, tooled: false, docId: C.docId, block: chBlock(), style: C.style, qi: C.qi, lineKind: chLineKind(), qKind: chQuestionKind(), unknown: chUnknown() };
    L.written = null; C.view = null; L.truthShown = false; L.hover = null; L.pinned = null;
    $('ch-ask').textContent = '■ Stop'; chRenderTest(); requestAnimationFrame(chWriteTick);
  }
  function chWriteStop() { const L = S.ch; if (!L.writing) return; L.writing = null; $('ch-ask').textContent = 'Ask'; }
  function chGenStep(W) { // one more token of the assistant's turn; true when the transcript is finished. The tool call is the loop's one event: the page runs it and pastes the findings back
    const L = S.ch, m = L.model, at = W.st.tokens.length, { token, probs } = m.genNext(W.st, { temperature: L.temperature }); W.steps.push({ at, token, probs }); W.lineLen++;
    const end = token === L.vocab.end, nl = token === L.nl;
    if (!end && !nl && W.lineLen < CH_MAX_LINE) return false;
    const words = RPR.decode(L.vocab, W.st.tokens), lines = CHR.assistantLines(words), last = lines[lines.length - 1];
    if (nl && last && CHR.callOf(last.text) && !W.tooled) {
      W.tooled = true; W.events.push({ kind: 'call', at });
      const paste = RPR.encode(L.vocab, `TOOL:\n${W.block}\nASSISTANT:`).slice(1, -1); W.pasted = { from: W.st.tokens.length, to: W.st.tokens.length + paste.length }; for (const t of paste) m.genPush(W.st, t);
      W.events.push({ kind: 'paste', at: W.pasted.from }); W.lineLen = 0; return false;
    }
    return true;
  }
  function chWriteTick(now) {
    const L = S.ch, W = L.writing; if (!W) return;
    try {
      const due = Math.min(6, Math.floor((now - W.t0) / 1000 * L.pace) - W.count); let done = false;
      for (let k = 0; k < due && !done; k++) { done = chGenStep(W); W.count++; }
      if (due > 0 || done) chRenderWriting();
      if (done) { chWriteFinish(); return; }
    } catch (e) { console.error(e); chWriteStop(); return; }
    requestAnimationFrame(chWriteTick);
  }
  function chResult(W) { // what the model answered, against the rule's answer where there is one
    const L = S.ch, words = RPR.decode(L.vocab, W.tokens), fw = L.model.forward(W.tokens), lines = CHR.assistantLines(words), last = lines[lines.length - 1], called = lines.some(l => CHR.callOf(l.text));
    const answer = last ? last.text : '', f = RPR.findingsOf(RPR.tokenize(W.block)), truth = typeof W.qi === 'number' ? CHR.answerOf(f, W.qi, W.style) : null;
    return { tokens: W.tokens, words, steps: W.steps, fw, answer, truth, exact: truth != null && CHR.same(answer, truth), first: truth != null && CHR.classOf(answer) === CHR.classOf(truth), called, tooled: W.tooled, pasted: W.pasted || null, style: W.style, answered: CHR.styleOf(words), docId: W.docId, qi: W.qi, lineKind: W.lineKind, qKind: W.qKind, unknown: W.unknown, events: W.events, temperature: L.temperature };
  }
  function chWriteFinish() {
    const L = S.ch, W = L.writing; if (!W) return; const res = chResult(W);
    L.writing = null; $('ch-ask').textContent = 'Ask'; L.written = res; L.truthShown = true;
    res.view = { words: res.words, tokens: res.tokens, fw: res.fw, textEl: $('ch-test-text'), pre: 'ch-test-', chosen: true, steps: res.steps, nextEl: 'ch-test-nextword' }; chChat().view = res.view;
    if (res.docId) { const doc = L.byId.get(res.docId); L.trial.results.set(res.docId, res); const set = chCaseSet(), i = set.indexOf(doc); if (i === L.trial.next) L.trial.next = i + 1; }
    chRenderTest();
  }
  function chAskAll() { // every remaining transcript of the set, one per frame, no animation
    const L = S.ch; if (!L.model || L.writing || L.batchWriting) return; L.batchWriting = true; $('ch-ask-all').textContent = '■ Stop';
    const set = chCaseSet(), step = () => {
      if (!L.batchWriting) return;
      const r = set[L.trial.next]; if (!r) { L.batchWriting = false; $('ch-ask-all').textContent = 'Ask all'; chRenderTest(); rpNote(`All ${set.length} transcripts asked.`); return; }
      chLoadDoc(r); const prefix = chPrefix(), st = L.model.genState(prefix), W = { prefix, st, tokens: st.tokens, steps: [], events: [], lineLen: 0, tooled: false, docId: r.id, block: chBlock(), style: r.style, qi: r.qi, lineKind: chLineKind(), qKind: chQuestionKind(), unknown: [] };
      let n = 0; while (!chGenStep(W) && ++n < 3 * CH_MAX_LINE) { /* until the transcript ends */ }
      L.writing = W; chWriteFinish();
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function chTestClear() { const L = S.ch; if (L.writing) chWriteStop(); L.batchWriting = false; $('ch-ask-all').textContent = 'Ask all'; L.trial = { next: 0, results: new Map() }; chLoadDoc(chCaseSet()[0]); rpNote(''); chRenderTest(); }
  function chSetCases(v) { const L = S.ch; if (v === L.cases) return; if (L.writing) chWriteStop(); L.batchWriting = false; L.cases = v; L.trayFor.chtest = false; L.trial = { next: 0, results: new Map() }; chTrays('test'); chLoadDoc(chCaseSet()[0]); rpNote(''); chRenderTest(); }
  function chCaseStep(d) { const ids = chCases(), C = chChat(), i = ids.indexOf(C.caseId); C.caseId = ids[(i + d + ids.length) % ids.length]; chDetach(); chRenderTest(); }
  function chTestStats() { const L = S.ch; let n = 0, exact = 0, first = 0, called = 0, agents = 0; for (const [, r] of L.trial.results) { n++; if (r.exact) exact++; if (r.first) first++; if (r.style === 'agent') { agents++; if (r.called) called++; } } return { n, exact, first, called, agents }; }
  function chCaseRows(block) { return (block || '').split('\n').slice(1).map(l => { const k = l.indexOf(':'); return `<div class="row"><span class="k">${esc(l.slice(0, k))}</span><span class="v">${esc(l.slice(k + 1).trim())}</span></div>`; }).join(''); }
  function chRenderWriting() { // the transcript as it grows, and the bars for the word being chosen
    const L = S.ch, W = L.writing, words = RPR.decode(L.vocab, W.tokens), stepAt = new Map(W.steps.map(s => [s.at, s])), probsOf = i => { const s = stepAt.get(i); return s ? s.probs[s.token] : null; };
    $('ch-test-text').innerHTML = rpTextHtml(words, probsOf, null) + '<span class="cursor"></span>';
    const last = W.steps[W.steps.length - 1]; if (last) $('ch-test-nextword').innerHTML = rpNextWordHtml(last.probs, null, last.token, `word ${W.steps.length}: what the model chose from`);
    $('ch-test-call').textContent = `Writing… ${W.steps.length} words so far${L.temperature > 0 ? `, drawn at temperature ${L.temperature.toFixed(2)}` : ', the most probable word each time'}.`;
    chRenderSteps(W.events, true);
  }
  function chRenderSteps(events, writing) { // the agent loop as it happens: the call, the paste, the answer
    const el = $('ch-steps'); if (!events || !events.length) { el.hidden = true; el.innerHTML = ''; return; }
    const pasted = events.some(e => e.kind === 'paste'), items = ['The assistant wrote <b>findings()</b>: a tool call, as text, and ended its line.'];
    if (pasted) items.push('The page, the program of the loop, saw the call, ran it, pasted the analyser’s findings for the case into the transcript as a <b>TOOL</b> turn, wrote <b>ASSISTANT:</b> and handed the transcript back.', writing ? 'The assistant continues: the answer, from the findings it was handed.' : 'The assistant answered from the findings it was handed.');
    el.hidden = false; el.innerHTML = items.map((t, i) => `<li${i === items.length - 1 && writing ? ' class="now"' : ''}>${t}</li>`).join('');
  }
  function chRenderTest() {
    const L = S.ch; if (!L.built || !L.model) return; chTrays('test'); chSyncControls(); const C = chChat();
    const set = chCaseSet(), T = L.trial, st = chTestStats(), heldSet = L.cases === 'held';
    $('ch-stat-n').textContent = `${st.n} / ${set.length}`; $('ch-stat-n-sub').textContent = heldSet ? 'transcripts on the held-out phrasings' : 'test transcripts, as written';
    $('ch-stat-exact').textContent = st.n ? pct(st.exact / st.n) : '–'; $('ch-stat-exact-sub').textContent = st.n ? `${st.exact} of ${st.n} the rule’s words${st.agents ? ` · findings() written in ${st.called} of ${st.agents} agent transcripts` : ''}` : 'no transcript asked yet';
    $('ch-stat-first').textContent = st.n ? pct(st.first / st.n) : '–'; $('ch-stat-first-sub').textContent = st.n ? `${st.first} of ${st.n}: yes, no, or the diagnosis’ first word` : '';
    const done = T.next >= set.length;
    $('ch-next-case').disabled = done; $('ch-ask-all').disabled = done && !L.batchWriting; $('ch-next-case').textContent = done ? `All ${set.length} asked` : `Next transcript (${T.next + 1} of ${set.length})`;
    $('ch-test-warning').hidden = !(L.model.steps === 0 && !L.shipped);
    const doc = C.docId ? L.byId.get(C.docId) : null, showCards = !!L.written && !L.writing;
    $('ch-test-net-card').hidden = !showCards; $('ch-test-heads-card').hidden = !showCards; $('ch-test-att-card').hidden = !showCards;
    $('ch-test-title').textContent = doc ? `Transcript ${doc.name} · ${heldSet ? 'held-out phrasing' : 'test'} · ${chStyleName(doc.style)} · case ${doc.case}` : `A prompt of your own · ${chStyleName(C.style)} · case ${C.caseId}`;
    $('ch-case-name').textContent = `Case ${C.caseId}`; $('ch-case-block').innerHTML = chCaseRows(chBlock());
    const unk = chUnknown(), lk = chLineKind(), qk = chQuestionKind();
    $('ch-prompt-note').textContent = (doc ? `Transcript ${doc.name} as written: the ${chStyleName(doc.style)} style, ${chLineName(doc)}, the case’s findings and its question. Press Ask, or change anything above for a prompt of your own.` : `A prompt of your own: ${lk === 'trained' ? 'a trained phrasing of the instruction' : lk === 'held' ? 'the phrasing of the instruction held out of training' : lk === 'probe' ? 'a system line the model never saw' : 'an instruction in your own words'}, ${qk === 'trained' ? 'one of the eight questions' : qk === 'probe' ? 'a question never asked in training' : 'a question in your own words'}. The answer is compared with the rule’s where there is one, and not tallied.`) + (unk.length ? ` ${unk.length} word${unk.length === 1 ? '' : 's'} of the prompt the model has no token for (${unk.map(w => `“${w}”`).join(', ')}) go${unk.length === 1 ? 'es' : ''} in as ⟨unk⟩.` : '');
    if (L.writing) { chRenderWriting(); $('ch-truth').hidden = true; }
    else if (L.written) {
      const res = L.written, stepAt = new Map(res.steps.map(s => [s.at, s])), probsOf = i => { const s = stepAt.get(i); return s ? s.probs[s.token] : null; };
      $('ch-test-text').innerHTML = rpTextHtml(res.words, probsOf, null);
      const verdict = res.truth == null ? '' : res.exact ? ' → <span class="good-text">the rule’s answer, word for word ✓</span>' : res.first ? ' → <span class="good-text">the right first word ✓</span>, not the rule’s words' : ' → <span class="bad-text">not the rule’s answer ✗</span>';
      const styleNote = res.lineKind === 'trained' ? '' : ` It answered in the ${chStyleName(res.answered)} style${res.answered === res.style ? ', as the instruction asked' : `, not the ${chStyleName(res.style)} style the instruction asked for`}${res.lineKind === 'held' ? ': the phrasing was held out of training' : res.lineKind === 'probe' ? ': the model never saw this system line' : ': the instruction was in your words'}.`;
      const agentNote = res.style === 'agent' && !res.called ? ' It answered without calling findings(): it never saw the findings, so this is a guess from the base rates.' : res.called && !res.tooled ? ' It wrote findings() and ended the transcript, so no answer followed.' : '';
      $('ch-test-call').innerHTML = `The model answered: <b>${esc(res.answer || '(nothing)')}</b>${verdict}${styleNote}${agentNote} · ${res.steps.length} words${res.temperature > 0 ? `, drawn at temperature ${res.temperature.toFixed(2)}` : ', the most probable word each time'}. Hover a word for what the model chose from and the words it read.`;
      chRenderSteps(res.events, false);
      $('ch-truth').hidden = !L.truthShown;
      if (L.truthShown) $('ch-truth').innerHTML = res.truth != null ? `<div>The rule says: <b>${esc(res.truth)}</b>${doc ? ' · the transcript’s own answer, which the model never saw for this case.' : ' · the rule’s answer for these findings, in the style asked for.'}</div>` : `<div>${res.qKind === 'probe' ? 'This question was never asked in training, so there is no rule’s answer: whatever the model wrote is the most probable continuation of a pattern it learned from the eight questions, fluent and unfounded.' : 'A question in your own words has no rule’s answer: whatever the model wrote is the most probable continuation of what it learned from the eight questions.'}</div>`;
      if (!L.batchWriting) { rpFollowWord(true); rpRenderAttention(rpView(true), false); }
    } else {
      $('ch-test-text').innerHTML = rpTextHtml(RPR.decode(L.vocab, chPrefix()), null, null) + '<span class="muted"> … press Ask.</span>';
      $('ch-test-nextword').innerHTML = ''; $('ch-test-call').textContent = C.style === 'agent' ? 'The system line and the question are the prompt; the model continues from ASSISTANT: one word at a time. If it writes findings(), the page pastes the findings back as a TOOL turn and lets it go on.' : 'The system line, the findings and the question are the prompt, as a user would type them; the model continues from ASSISTANT: one word at a time, to the end of the line.';
      $('ch-truth').hidden = true; chRenderSteps(null);
    }
    for (const [id, el] of L.testThumbs) { const rr = T.results.get(id); el.classList.toggle('selected', id === L.testSelected); el.classList.toggle('wrong', !!rr && !rr.exact); el.classList.toggle('right', !!rr && rr.exact); el.querySelector('.badge').textContent = rr && rr.exact ? '✓' : '✗'; }
  }
  function bindChat() {
    const L = S.ch;
    $('ch-data-prev').addEventListener('click', () => chDataNeighbour(-1)); $('ch-data-next').addEventListener('click', () => chDataNeighbour(1));
    $('ch-data-reveal').addEventListener('change', ev => { L.reveal = ev.target.checked; if (S.world === 'chat' && S.stage === 'data') chRenderData(); });
    $('ch-ask').addEventListener('click', chAsk); $('ch-next-case').addEventListener('click', () => chNextDoc()); $('ch-ask-all').addEventListener('click', () => (L.batchWriting ? (L.batchWriting = false, $('ch-ask-all').textContent = 'Ask all') : chAskAll())); $('ch-test-clear').addEventListener('click', chTestClear);
    document.querySelectorAll('#ch-cases-seg button').forEach(b => b.addEventListener('click', () => chSetCases(b.dataset.cases)));
    $('ch-temp').addEventListener('input', () => { L.temperature = +$('ch-temp').value / 100; $('ch-temp-val').textContent = L.temperature.toFixed(2); });
    $('ch-pace').addEventListener('input', () => { L.pace = +$('ch-pace').value; $('ch-pace-val').textContent = `${L.pace} words/s`; if (L.writing) { L.writing.t0 = performance.now(); L.writing.count = 0; } });
    const own = () => { chDetach(); chRenderTest(); };
    document.querySelectorAll('#ch-style-seg button').forEach(b => b.addEventListener('click', () => { const C = chChat(); if (b.dataset.style === C.style) return; C.style = b.dataset.style; own(); }));
    $('ch-line').addEventListener('change', () => { const v = $('ch-line').value, C = chChat(); C.line = /^\d+$/.test(v) ? +v : v; own(); if (C.line === 'own') $('ch-line-text').focus(); });
    $('ch-line-text').addEventListener('input', () => { chChat().lineText = $('ch-line-text').value; own(); });
    $('ch-question').addEventListener('change', () => { const v = $('ch-question').value, C = chChat(); C.qi = /^\d+$/.test(v) ? +v : v; own(); if (C.qi === 'own') $('ch-question-text').focus(); });
    $('ch-question-text').addEventListener('input', () => { chChat().qText = $('ch-question-text').value; own(); });
    for (const id of ['ch-line-text', 'ch-question-text']) $(id).addEventListener('keydown', ev => { if (ev.key === 'Enter') { ev.preventDefault(); chAsk(); } });
    $('ch-case-prev').addEventListener('click', () => chCaseStep(-1)); $('ch-case-next').addEventListener('click', () => chCaseStep(1));
  }
  function bindReports() {
    const L = new Proxy({}, { get: (_, k) => LM()[k], set: (_, k, v) => { LM()[k] = v; return true; } }); // the state of the world on screen, looked up at every event
    document.querySelectorAll('#rp-dim-seg button').forEach(b => b.addEventListener('click', () => { const v = +b.dataset.dim; if (v === L.dim) return; L.dim = v; rpReset(`${v} numbers per token — fresh random weights.`); }));
    document.querySelectorAll('#rp-pos-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.pos === '1'; if (v === L.positions) return; L.positions = v; rpReset(v ? 'Learned position vectors: every word also carries where it is — fresh random weights.' : 'No position vectors: the layers know how far back a word is only through their distance cost — fresh random weights.'); }));
    $('rp-epochs').addEventListener('input', () => { L.epochs = +$('rp-epochs').value; $('rp-epochs-val').textContent = L.epochs; if (L.model) { rpRenderStatus(); rpRenderCurves(); } });
    $('rp-lr').addEventListener('input', () => { L.lr = rpLrFromSlider(+$('rp-lr').value); $('rp-lr-val').textContent = L.lr; });
    $('rp-seed').addEventListener('change', () => { L.seed = Math.max(1, Math.round(+$('rp-seed').value) || 1); rpReset(`Seed ${L.seed} — fresh random weights.`); });
    $('rp-step-report').addEventListener('click', rpStepBatch); $('rp-step-epoch').addEventListener('click', rpStepEpoch);
    $('rp-train').addEventListener('click', () => (L.running && L.stopAt == null ? rpStop('Paused.') : rpStart()));
    $('rp-reset').addEventListener('click', () => rpReset('Weights re-initialised from the seed.'));
    $('rp-load').addEventListener('click', rpLoadShipped);
    $('rp-prev').addEventListener('click', () => rpNeighbour(-1)); $('rp-next').addEventListener('click', () => rpNeighbour(1));
    $('rp-data-prev').addEventListener('click', () => rpDataNeighbour(-1)); $('rp-data-next').addEventListener('click', () => rpDataNeighbour(1));
    $('rp-data-reveal').addEventListener('change', ev => rpSetReveal(ev.target.checked));
    document.querySelectorAll('.rp-layer-seg button').forEach(b => b.addEventListener('click', () => { L.layer = +b.dataset.layer; rpSyncControls(); if (L.model) { if (rpOnTest()) LMW().renderTest(); else rpRenderFocus(); } }));
    document.querySelectorAll('.rp-head-seg button').forEach(b => b.addEventListener('click', () => { L.head = +b.dataset.head; rpSyncControls(); if (L.model) { if (rpOnTest()) LMW().renderTest(); else rpRenderFocus(); } }));
    // hovering a word on either report follows it: what the model expected there, and whom it read
    for (const [id, onTest] of [['rp-text', false], ['rp-test-text', true], ['ch-test-text', true]]) {
      const el = $(id), tokAt = ev => { const t = ev.target.closest('.tok'); return t && el.contains(t) ? +t.dataset.i : null; };
      el.addEventListener('mousemove', ev => { const i = tokAt(ev); if (i === L.hover) return; L.hover = i; if (L.model) rpFollowWord(onTest); });
      el.addEventListener('mouseleave', () => { if (L.hover == null) return; L.hover = null; if (L.model) rpFollowWord(onTest); });
      el.addEventListener('click', ev => { const i = tokAt(ev); if (i == null || !L.model) return; L.pinned = L.pinned === i ? null : i; rpFollowWord(onTest); });
    }
    // the diagrams: a tooltip on every canvas; a read word hovered on the diagram is marked on the report
    for (const pre of ['rp-', 'rp-test-', 'ch-test-']) { const cv = $(`${pre}net`); if (!cv) continue; rpBindTip(cv, $(`${pre}net-tip`), (c, x, y) => (c._model ? (L.netView === 'graph' ? Viz.hitTokenGraph : Viz.hitTokenNetwork)(c, x, y, c._model) : null), h => { L.hoverNet = h ? Object.assign({ pre }, h) : null; const view = rpView(pre !== 'rp-'); if (!view) return; rpRenderNet(view); rpMarkPair(view.textEl, null, h && h.kind === 'read' ? h.j : null); }); }
    document.querySelectorAll('.rp-arcs-check').forEach(cb => cb.addEventListener('change', () => { L.arcs = cb.checked; rpSyncControls(); if (L.model) rpFollowWord(rpOnTest()); }));
    document.querySelectorAll('.rp-hops-check').forEach(cb => cb.addEventListener('change', () => { L.hops = cb.checked; rpSyncControls(); if (L.model) rpFollowWord(rpOnTest()); }));
    document.querySelectorAll('.rp-netview-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.view; if (v === L.netView) return; rpNetWalkStop(); L.netView = v; L.hoverNet = null; rpSyncControls(); const view = rpView(rpOnTest()); if (view) rpRenderNet(view); }));
    for (const pre of ['rp-', 'rp-test-', 'ch-test-']) { const b = $(`${pre}net-walk`); if (b) b.addEventListener('click', () => rpNetWalkStart(pre)); }
    for (const pre of ['rp-', 'rp-test-', 'ch-test-']) { const ch = $(`${pre}heads`); if (!ch) continue; rpBindTip(ch, $(`${pre}heads-tip`), (c, x, y) => (c._model ? Viz.hitHeadRows(c, x, y, c._model) : null), h => { L.hoverHeads = h ? Object.assign({ pre }, h) : null; const view = rpView(pre !== 'rp-'); if (!view) return; rpRenderHeads(view); rpMarkPair(view.textEl, null, h ? h.j : null); }); }
    document.querySelectorAll('#rp-map-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.map; if (v === L.map.mode) return; L.map.mode = v; rpSyncControls(); if (!L.model) return; if (v === 'tsne') rpMapRestart(true); else { L.map.running = false; rpRenderWordMap(); } }));
    document.querySelectorAll('#rp-map-src-seg button').forEach(b => b.addEventListener('click', () => { const v = b.dataset.src; if (v === L.map.source) return; L.map.source = v; L.pcaAxes = null; rpSyncControls(); if (!L.model) return; if (L.map.mode === 'tsne') rpMapRestart(true); else rpRenderWordMap(); }));
    // the Test step
    $('rp-write').addEventListener('click', rpWrite); $('rp-next-case').addEventListener('click', () => rpNextCase()); $('rp-write-all').addEventListener('click', () => (L.batchWriting ? (L.batchWriting = false, $('rp-write-all').textContent = 'Write all') : rpWriteAll())); $('rp-test-clear').addEventListener('click', rpTestClear);
    document.querySelectorAll('#rp-cases-seg button').forEach(b => b.addEventListener('click', () => rpSetCases(b.dataset.cases)));
    $('rp-temp').addEventListener('input', () => { L.temperature = +$('rp-temp').value / 100; $('rp-temp-val').textContent = L.temperature.toFixed(2); });
    $('rp-pace').addEventListener('input', () => { L.pace = +$('rp-pace').value; $('rp-pace-val').textContent = `${L.pace} words/s`; if (L.writing) { L.writing.t0 = performance.now(); L.writing.count = 0; } });
    $('rp-noblock').addEventListener('change', ev => { L.noblock = ev.target.checked; if (L.writing) rpWriteStop(); L.written = null; L.truthShown = false; rpSyncForm(); rpRenderTest(); });
  }

  // ------------------------------------------------------------------ boot
  function init() {
    try { const t = localStorage.getItem('nucleus-net-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
    try { if (sessionStorage.getItem('nucleus-net-foundation')) S.foundationShown = true; } catch (e) { /* ignore */ }
    Viz.refreshTheme();
    S.tasks = window.LECTURE_TASKS;
    const questions = [...Object.values(S.tasks).sort((a, b) => a.meta.task.order - b.meta.task.order).map(t => ({ id: t.meta.task.id, title: t.meta.task.title })), ...WORLD_QUESTIONS];
    $('question-select').innerHTML = questions.map((q, i) => `<option value="${q.id}">${i + 1} · ${esc(q.title)}</option>`).join('');
    $('recipe-select').innerHTML = '<option value="">choose a step…</option>' + RECIPE_LABELS.map((l, i) => (i ? `<option value="${i}">${esc(l)}</option>` : '')).join('');
    bindControls(); bindFoundation(); bindSlides(); bindFields(); bindReports(); bindChat(); renderBackboneOptions();
    loadTask(S.taskId);
    if (S.foundationShown) revealFoundation();
    syncControls();
    resetModel();
    renderDataTrays(); renderScatter(); renderInspector();
    onThemeChange();
    const hash = (location.hash || '').replace('#', ''), steps = ['data', 'train', 'test'];
    let qid = null, step = 'data';
    if (hash.includes('/')) [qid, step] = hash.split('/'); else if (steps.includes(hash)) step = hash; else if (hash) { qid = hash; step = 'train'; }
    if (qid === 'slides') qid = 'slides-atypia';
    if (qid && (WORLD_QUESTIONS.some(q => q.id === qid) || S.tasks[qid])) selectQuestion(qid);
    showStage(steps.includes(step) ? step : 'data');
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { if (onClassic('train')) renderTrainGraph(); if (onClassic('test')) renderTestGraph(); });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', onThemeChange);
  }
  document.addEventListener('DOMContentLoaded', init);
})();
