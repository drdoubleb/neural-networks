/*
 * app.js — wires the datasets, the network and the drawings into the three-stage bench.
 */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const NF = window.NucleusFeatures, DS = window.NucleusDataset, NN = window.TinyNet, Viz = window.Viz;
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ state
  const S = {
    tasks: null, taskId: 'leukaemia', task: null, ds: null, kind: 'tabular', size: 0, featureDefs: [],
    mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 60, speed: 6, seed: 1, augment: false, l2: 0, peek: false,
    animSpeed: 1, // playback speed of the walk-throughs (the lesson and Classify next): 1 = the normal pace
    excluded: new Set(),
    inputs: null, inputCache: new Map(), net: null,
    epoch: 0, ptr: 0, order: [], history: [], running: false, debt: 0, lastTime: 0, lastBatch: new Set(),
    trainEval: null, testEval: null, profileAt: 0, prevW: null, applyingRecipe: false,
    selected: null, view: 'image', tint: true, revealTest: false, stage: 'data', trayColor: 'call',
    test: { results: new Map(), next: 0, threshold: 0.5, animating: false, revealed: new Set(), prevalence: 0.01 },
    hover: { train: null, test: null },
    lesson: null, lastLesson: null, lessonCursor: 0,
  };
  const thumbs = { data: new Map(), train: new Map(), test: new Map() }; // id -> element

  const RECIPE_LABELS = ['', '① Leukaemia · blood count · single layer', '② Leukaemia · blood count · 3 ReLU units', '③ Atypia · measurements · single layer', '④ Enlargement · pixels · single layer', '⑤ Irregularity · pixels · single layer', '⑥ Irregularity · pixels · 4 ReLU + augmentation', '⑦ Irregularity · pixels · 4 + 4 ReLU + augmentation', '⑧ Irregularity · pixels · convolution + 4 ReLU + augmentation'];
  const RECIPES = {
    1: { task: 'leukaemia',    mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 60,  augment: false, l2: 0,    peek: false, speed: 6 },
    2: { task: 'leukaemia',    mode: 'features', h1: 3, h2: 0, convK: 0, activation: 'relu', lr: 0.05, batch: 8, epochs: 150, augment: false, l2: 0,    peek: false, speed: 10 },
    3: { task: 'atypia',       mode: 'features', h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.1,  batch: 8, epochs: 60,  augment: false, l2: 0,    peek: false, speed: 4 },
    4: { task: 'enlargement',  mode: 'pixels',   h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 30,  augment: false, l2: 0,    peek: true,  speed: 4 },
    5: { task: 'irregularity', mode: 'pixels',   h1: 0, h2: 0, convK: 0, activation: 'relu', lr: 0.01, batch: 8, epochs: 60,  augment: false, l2: 0,    peek: true,  speed: 4 },
    6: { task: 'irregularity', mode: 'pixels',   h1: 4, h2: 0, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 60,  augment: true,  l2: 0.01, peek: true,  speed: 6 },
    7: { task: 'irregularity', mode: 'pixels',   h1: 4, h2: 4, convK: 0, activation: 'relu', lr: 0.02, batch: 8, epochs: 60,  augment: true,  l2: 0.01, peek: true,  speed: 6 },
    8: { task: 'irregularity', mode: 'pixels',   h1: 4, h2: 0, convK: 4, activation: 'relu', lr: 0.02, batch: 8, epochs: 30,  augment: true,  l2: 0,    peek: true,  speed: 4 },
  };

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
    const ax = { leukaemia: [0, 5], atypia: [0, 2], enlargement: [0, 2], irregularity: [5, 4] }[id] || [0, 1];
    $('scatter-x').value = ax[0]; $('scatter-y').value = ax[1];
    $('scatter-hint').textContent = {
      leukaemia: 'WBC against lymphocytes shows why no single parameter works: CLL and viral lymphocytosis overlap, and acute leukaemia sits at both ends of the WBC axis. Try platelets against haemoglobin, or basophils against immature granulocytes.',
      atypia: 'Area against darkness catches the enlarged and the hyperchromatic nuclei, solidity against contour roughness the irregular ones, texture the coarse chromatin. No single pair catches every atypical nucleus, which is why the network gets all six measurements.',
      enlargement: 'Area against darkness separates the classes with a straight line; solidity against contour roughness is now the decoy pair.',
      irregularity: 'Try the decoys, area against darkness, then solidity against contour roughness. A single straight line separates the classes on the second pair; that is what a one-layer network has to find.',
    }[id] || '';
    $('mode-pixels').disabled = S.kind === 'tabular';
    $('mode-pixels').title = S.kind === 'tabular' ? 'Blood counts have no pixels' : '';
    renderInputPicker();
  }
  function inputsFor(mode, augment, excluded) {
    const key = mode + (mode === 'pixels' && augment ? '+aug' : '') + (mode === 'features' && excluded.size ? '-' + [...excluded].sort().join(',') : '');
    if (!S.inputCache.has(key)) S.inputCache.set(key, DS.buildInputs(S.ds, mode, { augment, exclude: excluded }));
    return S.inputCache.get(key);
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
    const tr = S.ds.train, te = S.ds.test;
    S.trainEval = S.net.evaluate(tr.map(s => S.inputs.xOf(s)), tr.map(s => s.label), 0.5, true);
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
  const LESSON_MS = { forward: 2600, loss: 2600, blame: 2600, blame2: 3400, gradient: 3000, update: 2600, check: 2600 };
  const LESSON_TITLES = { forward: 'Forward pass', loss: 'Loss', blame: 'Backward pass', gradient: 'Gradients', update: 'Update', check: 'Check' };
  function lessonPhases() {
    const L = S.net.hidden.length, plan = Viz.sweepPlan(S.net, S.mode), replay = Viz.sweepPlan(S.net, S.mode, { prep: false });
    const ph = [['forward', plan.total], ['loss', LESSON_MS.loss]];
    if (L) ph.push(['blame', L > 1 ? LESSON_MS.blame2 : LESSON_MS.blame]);
    ph.push(['gradient', LESSON_MS.gradient], ['update', LESSON_MS.update], ['check', S.mode === 'pixels' ? Math.round(replay.total / 3) : replay.total]); // the check replays a pixel sweep at triple speed, the subtraction already done
    return ph;
  }
  function canTeach() { return !!(S.net && !S.running && !S.lesson && !S.net.conv); }
  // the case the next lesson will use: a training case the user picked, else the one after the last lesson
  function nextLessonCase() {
    const tr = S.ds.train;
    if (S.selected && S.selected.split === 'train' && !(S.lastLesson && S.lastLesson.s === S.selected)) return S.selected;
    return tr[S.lessonCursor % tr.length];
  }
  function updateTeachButton() {
    const b = $('btn-teach'); if (!b || !S.net) return;
    b.disabled = !canTeach() && !S.lesson;
    b.textContent = S.lesson ? 'Skip ▸' : 'Teach next case';
    b.title = S.net.conv ? 'The walk-through covers dense networks: switch the convolution off'
      : `Teach ${nextLessonCase().name} one step: forward pass, loss, backward pass, gradients, update, check (T)`;
  }
  function teachNext() {
    if (S.lesson) { S.lesson.player.skip(); return; }
    if (!canTeach()) return;
    startLesson(nextLessonCase());
  }
  function snapParams() { return { W: S.net.W.map(w => Float64Array.from(w)), b: S.net.b.map(b => Float64Array.from(b)), Wo: Float64Array.from(S.net.Wo), bo: S.net.bo }; }
  function setParams(p) { p.W.forEach((w, l) => S.net.W[l].set(w)); p.b.forEach((b, l) => S.net.b[l].set(b)); S.net.Wo.set(p.Wo); S.net.bo = p.bo; }
  function lerpParams(a, b, t) {
    const mix = (x, y) => { const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] + (y[i] - x[i]) * t; return o; };
    return { W: a.W.map((w, l) => mix(w, b.W[l])), b: a.b.map((v, l) => mix(v, b.b[l])), Wo: mix(a.Wo, b.Wo), bo: a.bo + (b.bo - a.bo) * t };
  }
  function startLesson(s) {
    if (!s || s.split !== 'train' || !S.net || S.net.conv) return;
    stopTraining();
    if (S.selected !== s) selectSpecimen(s);
    const x = S.inputs.xOf(s), y = s.label;
    const res = S.net.lesson(x, y);
    const hops = Viz.sweepPlan(S.net, S.mode).hops;
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
  function lessonView() {
    const les = S.lesson; if (!les) return null;
    const g = les.res.g, hops = les.hops;
    let reveal = hops; // how many hops of the forward sweep have arrived (everything is visible outside the sweeps)
    if (les.phase === 'forward' || les.phase === 'check') reveal = Viz.sweepState(Viz.sweepPlan(S.net, S.mode, { prep: les.phase !== 'check' }), les.frac).reveal;
    return { phase: les.phase, frac: les.frac, hops, reveal, error: les.res.error, loss: les.res.loss, y: les.y, truthName: className(les.y), pBefore: les.pBefore, pAfter: les.pAfter, delta: g.delta, gW: g.gW, gWo: g.gWo, lr: S.lr, fwBefore: les.res.fw };
  }
  // what the forward sweep shows, for the strip
  function forwardText() {
    const hidden = S.net.hidden.length;
    if (S.mode === 'pixels' && !S.net.conv) {
      const prep = `First the mean training ${noun(1)} is subtracted from this one: the network sees the difference (orange = more ink than average, blue = less), which is large at the membrane and near zero in the centre. `;
      return prep + (hidden
        ? `That difference is laid over each weight map and multiplied cell by cell into a product map (the magnifier shows one 4×4 block with its numbers). A scan line sums the map, orange cells against blue, the bias is added, and ReLU keeps the positive part: that is the unit's value. The units' values × their weights then flow to the output.`
        : `That difference is laid over the weight map and multiplied cell by cell into the product map (the magnifier shows one 4×4 block with its numbers). A scan line sums the map, orange cells against blue, the bias is added to give the score z, and the sigmoid turns z into a probability.`);
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
      const hidden = S.net.hidden.length, pixels = S.mode === 'pixels';
      const p = les.pBefore.toFixed(2), e = Viz.fmtSigned(les.res.error, 2), call = les.pBefore >= 0.5 ? posName() : negName();
      const sure = Math.abs(les.res.error) < 0.02, allSilent = hidden > 0 && les.res.g.delta.every(d => d.every(v => v === 0));
      const pair = les.pAfter == null ? `${p} → …` : fmtPair(les.pBefore, les.pAfter);
      const texts = {
        forward: `${forwardText()} The network calls <b>${p}</b> (${esc(call)}); the truth is <b>${esc(className(les.y))}</b>.`,
        loss: `How wrong was it? Cross-entropy loss <b>${les.res.loss.toFixed(2)}</b>. Its slope at the output is the error, call − truth = ${p} − ${les.y} = <b>${e}</b>: ${les.res.error > 0 ? 'too high, so the score must come down' : 'too low, so the score must go up'}.`,
        blame: hidden > 1
          ? 'How much is each hidden unit to blame? The error flows back one layer at a time along each connection as blame × weight (thick = large; blue: the unit should come down, orange: go up). A unit adds up what arrives, and ReLU passes it on only if the unit was on: an off unit’s blame is 0. Each pill shows the arithmetic.'
          : 'How much is each hidden unit to blame? The error flows back along each connection as error × weight (thick = large; blue: the unit should come down, orange: go up). ReLU passes it on only if the unit was on, so a unit that was off gets blame 0. Each pill shows the arithmetic.',
        gradient: allSilent
          ? 'Every hidden unit was off for this case, so every blame is zero and no weight map has a gradient: only the output bias does. A ReLU unit that is off cannot learn from a case.'
          : hidden
          ? `For every connection, gradient = blame at its end × activity at its start; the labels spell it out. The glow shows which way the weight should move (orange up, blue down) and how steeply.${pixels ? ` On pixels, a weight map’s gradient is the difference image itself (what the network sees), scaled by the unit’s blame.` : ''}`
          : `For every connection, gradient = error × its input: big inputs, big gradients; the labels spell it out. Orange = the weight should rise, blue = fall.${pixels ? ` On pixels, the gradient of the whole weight map is the difference image itself (what the network sees), scaled by the error.` : ''}`,
        update: `Every weight takes one small step against its gradient: w ← w − learning rate × gradient, with learning rate ${S.lr}. The labels show each weight before → after; watch the connections${pixels ? ' and weight maps' : ''} change.`,
        check: sure
          ? `The same case runs forward again. It was already right and sure, so the step was tiny: <b>${pair}</b>.`
          : `The same case runs forward again with the new weights: <b>${pair}</b>. One case, one small step; training repeats this for every case, many times over.`,
      };
      html = `<span class="step">Step ${n} of ${total} · ${LESSON_TITLES[les.phase]}</span> <span>${texts[les.phase]}</span> <span class="muted small">N or a click on the diagram skips ahead · space pauses · ← → step</span>`;
    } else {
      html = `<span class="step">✓ ${esc(last.s.name)}</span> <span>(truth ${esc(className(last.s.label))}) ${fmtPair(last.pBefore, last.pAfter)}. Next up: <b>${esc(nextLessonCase().name)}</b>.</span>`;
    }
    $('lesson-text').innerHTML = html;
    renderPlayerControls('lesson', les ? les.player : null);
  }

  function stepBatch() { stopTraining(); trainStep(); renderTraining(true); if (S.ptr === 0) note(`Epoch ${S.epoch} complete.`); }
  function stepEpoch() { stopTraining(); do { trainStep(); } while (S.ptr !== 0); renderTraining(true); note(`Epoch ${S.epoch} complete.`); }

  // ------------------------------------------------------------------ test phase
  function clearTestResults(notify) {
    S.test.results.clear(); S.test.next = 0; S.test.revealed.clear(); S.test.animating = false; S.test.player = null;
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
  function setThumbState(el, { call, truth, wrong, right, unknown, inBatch, selected, q, title, unit }) {
    el.className = 'thumb' + (call != null ? ` call-${call}` : '') + (truth != null ? ` truth-${truth}` : '') + (wrong ? ' wrong' : '') + (right ? ' right' : '') + (unknown ? ' unknown' : '') + (inBatch ? ' in-batch' : '') + (selected ? ' selected' : '');
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
      setThumbState(el, { truth: known ? s.label : null, unknown: !known, selected: S.selected && S.selected.id === s.id,
        title: `${s.name} · ${s.split === 'train' ? 'training' : 'test'} set${known ? ' · ' + className(s.label) : ''}${known && s.subtype ? ' · ' + subtypeName(s.subtype) : ''}` });
    }
  }
  function topUnit(acts) { let j = 0; for (let i = 1; i < acts.length; i++) if (acts[i] > acts[j]) j = i; return acts[j] > 0 ? j : null; }
  function renderTrainTray() {
    const probs = S.trainEval.probs, acts = S.trainEval.acts;
    const byUnit = S.trayColor === 'unit' && acts;
    S.ds.train.forEach((s, k) => {
      const el = thumbs.train.get(s.id);
      const p = probs[k], call = p >= 0.5 ? 1 : 0;
      const u = byUnit ? topUnit(acts[k]) : null;
      setThumbState(el, { call, wrong: call !== s.label, inBatch: S.lastBatch.has(s.id), selected: S.selected && S.selected.id === s.id, unit: u,
        title: `${s.name} · truth ${className(s.label)}${s.subtype ? ' (' + subtypeName(s.subtype) + ')' : ''} · call ${className(call)} (P ${fmtP(p)})${byUnit ? ` · most active: ${u == null ? 'no unit' : 'unit ' + (u + 1)}` : ''}` });
    });
    $('tray-legend-call').hidden = !!byUnit;
    $('tray-legend-unit').hidden = !byUnit;
    if (byUnit) $('tray-legend-unit').innerHTML = S.net.hidden[0] ? Array.from({ length: S.net.hidden[0] }, (_, j) => `<span><span class="udot" style="background:${Viz.unitColor(j)}"></span>unit ${j + 1}</span>`).join('') + '<span class="muted">grey = no unit active</span>' : '';
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
          title: `${s.name} · call ${className(call)} (P ${fmtP(r.p)}) · truth ${className(s.label)}${s.subtype ? ' (' + subtypeName(s.subtype) + ')' : ''}` });
      }
    }
  }

  // ------------------------------------------------------------------ rendering: train panel
  function note(msg) { $('note').textContent = msg || ''; }
  function renderStatus() {
    const bpe = batchesPerEpoch();
    const bi = S.ptr === 0 ? bpe : Math.ceil(S.ptr / S.batch);
    const inputDesc = S.mode === 'features' ? `${S.inputs.inputSize} ${S.kind === 'tabular' ? 'parameters' : 'measurements'}` : '1,024 pixels';
    $('status').innerHTML =
      `<span>architecture <b>${inputDesc} → ${S.net.describe()} → output</b></span>` +
      `<span>parameters <b>${S.net.parameterCount().toLocaleString()}</b></span>` +
      `<span>training cases <b>${S.inputs.trainX.length}</b>${S.augment && S.mode === 'pixels' ? ' (80 × 8 orientations)' : ''}</span>` +
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
    return { net: S.net, mode: S.mode, x, fw, prev: S.prevW, featureNames: S.inputs.featureNames || [], specimen: s, size: S.size, tint: S.tint, stage, hover,
      inputMean: S.mode === 'pixels' && S.inputs.std ? S.inputs.std.mean : null,
      activation: S.activation, activationLabel: NN.ACTIVATIONS[S.activation].label, positiveName: posName(), negativeName: negName() };
  }
  function renderTrainGraph() {
    const cv = $('net-canvas');
    const m = graphModel(S.selected || null, 2, S.hover.train);
    if (S.lesson) { const lv = lessonView(); m.lesson = lv; m.reveal = lv.reveal; m.hops = lv.hops; if (lv.phase !== 'check') m.fw = S.lesson.res.fw; }
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
    const K = S.net.conv ? S.net.conv.K : 0, pool = S.net.conv ? S.net.conv.pool : 0, nL = S.net.hidden.length, F = S.net.featureCount;
    const hop = info && info.key, cols = K <= 4 ? 2 : 4, stack = S.net.conv ? `${Math.ceil(K / cols)}×${cols}` : '';
    const convTexts = {
      prep: `<span class="step">Preprocessing</span> <span><b>${esc(s.name)}</b>: the mean training ${noun(1)} is subtracted from this one. The network sees the difference (orange = more ink than average, blue = less), large at the membrane and near zero in the centre.</span>`,
      scan1: `<span class="step">Convolution</span> <span>Filter 1 slides over the difference image. At each position its 5×5 weights multiply the 5×5 values under them, the products are summed, and ReLU keeps the positive part: one cell of feature map 1. The arithmetic of the current position is shown under the image.</span>`,
      scan: `<span class="step">Convolution</span> <span>The other ${K - 1} filters sweep the ${noun(1)} the same way, each producing its own feature map.</span>`,
      pool1: `<span class="step">Max-pooling</span> <span>Each ${pool}×${pool} block of feature map 1 keeps only its largest value, so the map shrinks and the exact position inside a block no longer matters.</span>`,
      pool: `<span class="step">Max-pooling</span> <span>The other maps are pooled the same way.</span>`,
      units: nL
        ? `<span class="step">Dense layer</span> <span>The ${K} pooled maps are stacked ${stack} into one map and laid over each hidden unit’s weight map (one weight per pooled cell, ${F}, stacked the same way), multiplied cell by cell into a product map. A scan line sums the map, orange cells against blue, the bias is added, and ReLU keeps the positive part: the unit’s value. Unit 1 slowly, then the rest together.</span>`
        : `<span class="step">Output</span> <span>The ${K} pooled maps are stacked ${stack} into one map and laid over the output’s weight map (one weight per pooled cell, ${F}), multiplied cell by cell into a product map. A scan line sums the map, orange cells against blue; the sum plus the bias is z.</span>`,
      hop: hop === 'out'
        ? `<span class="step">Output</span> <span>The units’ values × their weights flow to the output, each connection as thick as the product it carries. Their sum plus the bias is z, and the sigmoid turns z into P(${esc(posName())}).</span>`
        : hop === 'sum'
        ? `<span class="step">Output</span> <span>z goes through the sigmoid: P(${esc(posName())}) = 1 / (1 + e<sup>−z</sup>).</span>`
        : `<span class="step">Hidden layer ${(hop || 0) + 1}</span> <span>Each unit sums weight × value over the layer before, plus its bias, through ${NN.ACTIVATIONS[S.activation].label}.</span>`,
      final: `<span class="step">Forward pass done</span> <span>Every value is on the diagram. The weights are frozen: nothing is learned here.</span>`,
    };
    if (convTexts[phase]) html = convTexts[phase];
    else if (phase === 'forward') html = `<span class="step">Forward pass</span> <span><b>${esc(s.name)}</b>: its ${inputDesc} flow through the frozen weights. ${forwardText()} Nothing is learned here.</span>`;
    else if (phase === 'call') html = `<span class="step">Call</span> <span>P(${esc(posName())}) = <b>${pp}</b>, which is ${p >= S.test.threshold ? 'at or above' : 'below'} the threshold of ${thr}, so the network calls <b>${esc(calledName)}</b>.</span>`;
    else html = `<span class="step">${correct ? '✓' : '✗'} ${esc(s.name)}</span> <span>called <b>${esc(calledName)}</b> (${pp}) · truth <b>${esc(truth)}</b>${correct ? '' : (called ? ' · a false positive' : ' · a false negative')}.</span> <span class="muted small">N classifies the next case</span>`;
    $('test-lesson-text').innerHTML = html;
    renderPlayerControls('test', S.test.animating ? S.test.player : null);
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
    const net = S.net, nL = net.hidden.length, total = net.co * net.co, cells = net.po * net.po, multi = net.conv.K > 1, units = nL ? net.hidden[0] : 1;
    const T = [['prep', 4000], ['scan1', 12000], ['scan', multi ? 7000 : 0], ['pool1', 4000], ['pool', multi ? 2500 : 0], ['units', 12000 + (units > 1 ? 6000 : 0)]];
    const hopKeys = []; for (let l = 1; l < nL; l++) hopKeys.push(l); hopKeys.push(nL ? 'out' : 'sum');
    for (const key of hopKeys) T.push(['hop', 1200, key]);
    T.push(['final', 800], ['call', 900], ['reveal', 900]);
    const hops = 1 + hopKeys.length; // the pooled-maps hop, the dense hops between hidden layers, the output
    const p = S.test.results.get(s.id).p, called = p >= S.test.threshold ? 1 : 0, thr = S.test.threshold;
    let shown = false;
    const P = makePlayer(T, null); S.test.player = P;
    const finish = () => { S.test.player = null; S.test.revealed.add(s.id); S.test.animating = false; renderTestPanel(); renderInspector(); renderTestGraph(); renderTestLine(s, p, called, 'done'); };
    const frame = now => {
      if (!S.test.animating || S.selected !== s || S.test.player !== P) return;
      P.tick(now); const cur = P.at();
      if (!cur) { finish(); return; }
      const [phase, , arg] = cur.phase, frac = cur.frac;
      const anim = { phase, hops, reveal: 0, banner: null, dir: 0 };
      if (phase === 'prep') anim.t = frac;
      else if (phase === 'scan1' || phase === 'scan') { anim.pos = Math.min(total, Math.floor(frac * total)); anim.showFilter = phase === 'scan1' ? 0 : 1; }
      else if (phase === 'pool1' || phase === 'pool') anim.posP = Math.min(cells, Math.floor(frac * cells));
      else if (phase === 'units') { anim.t = frac; anim.banner = nL ? 'dense layer: pooled maps × each unit’s weights, summed, through ReLU' : 'output: pooled maps × weights, summed, plus the bias'; anim.dir = 1; }
      else if (phase === 'hop') { const i = hopKeys.indexOf(arg); anim.key = arg; anim.t = frac; anim.wipes = hopKeys.slice(0, i + 1); anim.reveal = 1 + i; anim.banner = arg === 'sum' ? 'output: z through the sigmoid' : `${arg === 'out' ? 'output' : 'hidden layer ' + (arg + 1)}: each connection carries weight × value`; anim.dir = 1; }
      else { anim.wipes = hopKeys; anim.reveal = hops; }
      if (phase === 'final') anim.banner = 'forward pass done: the weights are frozen, nothing is learned here';
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
    Viz.drawCurves($('chart-loss'), { history: S.history, key: 'loss', showTest: S.peek, maxEpoch: S.epochs });
    Viz.drawCurves($('chart-acc'), { history: S.history, key: 'acc', showTest: S.peek, maxEpoch: S.epochs });
    $('loss-now').textContent = S.trainEval.loss.toFixed(3);
    $('acc-now').textContent = pct(S.trainEval.accuracy);
    $('curves-legend-test').hidden = !S.peek;
  }
  // the single-layer network as a weighted checklist
  function renderScorecard() {
    const card = $('scorecard');
    const show = S.net.hidden.length === 0 && !S.net.conv && S.mode === 'features';
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
    const rows = subtypes.map(st => { const sums = new Float64Array(H); let n = 0; S.ds.train.forEach((s, k) => { if (s.subtype === st.key) { n++; for (let j = 0; j < H; j++) sums[j] += acts[k][j]; } }); return { st, n, mean: Array.from(sums, v => (n ? v / n : 0)) }; }).filter(r => r.n > 0);
    const colMax = Array.from({ length: H }, (_, j) => Math.max(1e-9, ...rows.map(r => Math.abs(r.mean[j]))));
    const signed = S.activation === 'tanh';
    let html = `<table class="profile"><thead><tr><th class="row">kind of ${esc(S.task.specimenNoun || 'case')}</th>` + Array.from({ length: H }, (_, j) => `<th><span class="udot" style="background:${Viz.unitColor(j)}"></span>unit ${j + 1}</th>`).join('') + '</tr></thead><tbody>';
    for (const r of rows) {
      html += `<tr><td class="row"><span class="dot" style="background:${r.st.positive ? 'var(--irregular)' : 'var(--regular)'}"></span>${esc(r.st.name)}<span class="n">n=${r.n}</span></td>` +
        r.mean.map((v, j) => { const t = Math.abs(v) / colMax[j]; const bg = signed ? Viz.diverging(v / colMax[j], 0.25 + 0.75 * t) : Viz.sequential(t, 0.15 + 0.85 * t); return `<td style="background:${bg};color:${t > 0.6 ? '#fff' : 'var(--ink)'}">${v.toFixed(2)}</td>`; }).join('') + '</tr>';
    }
    if (S.net.hidden.length === 1) html += `<tr><td class="row">weight to output</td>` + Array.from({ length: H }, (_, j) => `<td class="out" style="color:${S.net.Wo[j] >= 0 ? 'var(--irregular)' : 'var(--regular)'}">${Viz.fmtSigned(S.net.Wo[j], 2)}</td>`).join('') + '</tr>';
    if (S.mode === 'features') {
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

  // ------------------------------------------------------------------ rendering: test panel
  function renderTestPanel() {
    renderTestTray();
    const st = testStats();
    $('stat-n').textContent = `${st.n} / ${S.ds.test.length}`;
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
      `<span class="chip plain">${s.split === 'train' ? 'Training set' : 'Test set · held out'}</span>` +
      (known ? `<span class="chip ${classOf(s.label)}">truth: ${esc(className(s.label))}</span>` : `<span class="chip plain">truth hidden</span>`) +
      (known && s.subtype && tabular ? `<span class="chip plain">${esc(subtypeName(s.subtype))}</span>` : '');
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
      v.querySelector('.p').innerHTML = `<small class="lbl">P(${esc(posName())})</small><b>${fw.p.toFixed(3)}</b><small>score z = ${Viz.fmtSigned(fw.z, 2)}</small>`;
      let chips = `<span class="chip ${classOf(call)}">call: ${esc(className(call))}</span>`;
      if (known) chips += call === s.label ? `<span class="chip good">✓ agrees with truth</span>` : `<span class="chip bad">✗ truth is ${esc(className(s.label))}</span>`;
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
      : 'In pixel mode the network never sees these measurements — they are here for you, the human.';
  }

  // ------------------------------------------------------------------ data panel
  function renderScatter() {
    const xi = +$('scatter-x').value, yi = +$('scatter-y').value;
    const fx = S.featureDefs[xi], fy = S.featureDefs[yi];
    const tx = v => (fx.log ? Math.log10(Math.max(v, 1e-3)) : v), ty = v => (fy.log ? Math.log10(Math.max(v, 1e-3)) : v);
    const pts = S.ds.specimens.filter(s => s.split === 'train' || S.revealTest).map(s => ({
      id: s.id, name: s.name, x: tx(s.features[xi]), y: ty(s.features[yi]), split: s.split,
      cls: truthKnown(s) ? classOf(s.label) : 'unknown', clsName: truthKnown(s) ? className(s.label) + (s.subtype ? ' · ' + subtypeName(s.subtype) : '') : '', selected: S.selected && S.selected.id === s.id,
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
    const tabular = S.kind === 'tabular', pixels = S.mode === 'pixels';
    $('mode-wrap').hidden = tabular;
    $('conv-wrap').hidden = !pixels;
    $('augment-wrap').hidden = !pixels;
    $('picker-wrap').hidden = pixels;
    $('tint-wrap').hidden = tabular;
    $('prevalence-card').hidden = !tabular;
    $('legend-conv').hidden = !(pixels && S.convK > 0);
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
      if (S.mode === 'features' && S.lr < 0.05) S.lr = 0.1;
      syncControls(); resetModel(`Input changed to ${S.mode === 'pixels' ? 'raw pixels' : (S.kind === 'tabular' ? 'the blood count' : 'measurements')} — fresh random weights.`);
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
    $('btn-train').addEventListener('click', () => (S.running ? stopTraining('Paused.') : startTraining()));
    $('btn-step-batch').addEventListener('click', stepBatch);
    $('btn-step-epoch').addEventListener('click', stepEpoch);
    $('btn-reset').addEventListener('click', () => resetModel('Weights re-initialised from the seed.'));
    $('btn-teach').addEventListener('click', teachNext);
    $('recipe-select').addEventListener('change', () => {
      const k = $('recipe-select').value; if (!k) return;
      const { task: taskId, ...settings } = RECIPES[k];
      const switchTask = taskId !== S.taskId;
      if (switchTask) loadTask(taskId);
      Object.assign(S, settings);
      S.excluded = new Set();
      S.applyingRecipe = true;
      syncControls();
      resetModel(`Recipe ${RECIPE_LABELS[k]}${switchTask ? ` — question switched to “${S.task.title}”` : ''}. Press Train.`);
      S.applyingRecipe = false;
      $('recipe-select').value = k;
      if (switchTask) { renderDataTrays(); renderScatter(); renderInspector(); }
      if (S.stage !== 'train') showStage('train');
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
    $('tint').addEventListener('change', () => { S.tint = $('tint').checked; repaintThumbs(); renderInspector(); if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
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
      else if ((ev.key === 'n' || ev.key === 'N') && S.stage === 'test') classifyNext(false);
      else if ((ev.key === 'n' || ev.key === 'N' || ev.key === 't' || ev.key === 'T') && S.stage === 'train') teachNext();
      else if (ev.key === '1') showStage('data'); else if (ev.key === '2') showStage('train'); else if (ev.key === '3') showStage('test');
    });
    window.addEventListener('resize', () => { if (S.stage === 'train') renderTrainGraph(); if (S.stage === 'test') renderTestGraph(); });
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
    renderInspector();
  }
  function showStage(name) {
    S.stage = name;
    document.querySelectorAll('.stage').forEach(b => b.classList.toggle('is-active', b.dataset.stage === name));
    $('panel-data').hidden = name !== 'data'; $('panel-train').hidden = name !== 'train'; $('panel-test').hidden = name !== 'test';
    if (name === 'data') { renderDataTrays(); renderScatter(); renderInspector(); }
    if (name === 'train') { if (S.selected && S.selected.split !== 'train') S.selected = S.ds.train[0]; renderTraining(true); }
    if (name === 'test') {
      stopTraining(); renderTestPanel();
      if (!S.selected || S.selected.split !== 'test') { S.selected = S.ds.test[Math.max(0, S.test.next - 1)]; }
      renderTestGraph(); renderInspector(); renderDataTrays();
      if (S.net.conv && !$('test-note').textContent) $('test-note').textContent = 'Classify next walks through the convolution (about 50 s): the mean nucleus is subtracted, filter 1 scans the difference slowly with its arithmetic shown, the other filters follow together, map 1 is pooled block by block and the other maps follow, then the pooled maps are stacked and laid over each hidden unit’s weight map and summed, and the units feed the output. Press N or click the diagram to skip ahead; space pauses, ← → step.';
    }
    try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------------ boot
  function init() {
    try { const t = localStorage.getItem('nucleus-net-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
    Viz.refreshTheme();
    S.tasks = window.LECTURE_TASKS;
    $('question-select').innerHTML = Object.values(S.tasks).sort((a, b) => a.meta.task.order - b.meta.task.order).map((t, i) => `<option value="${t.meta.task.id}">${i + 1} · ${esc(t.meta.task.title)}</option>`).join('');
    $('recipe-select').innerHTML = '<option value="">choose a step…</option>' + RECIPE_LABELS.map((l, i) => (i ? `<option value="${i}">${esc(l)}</option>` : '')).join('');
    bindControls();
    loadTask(S.taskId);
    syncControls();
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
