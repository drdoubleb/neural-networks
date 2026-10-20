#!/usr/bin/env node
/*
 * export_models.js — trains every lesson's model on the page itself, in a headless browser, and writes the models
 * that the page's "Load the trained model" buttons bring back: data/trained/trained_models.js (window.TRAINED_MODELS).
 * The page does the training with its own code, data, seeds and curves, so a loaded model is exactly what pressing
 * Train would have given with the recipe's settings, with the curves of that run. Covered: the classic recipes
 * (1–10 and 12, by recipe number), the 4 · Foundation encoder (seed 1, 100 epochs), the slide models (both
 * questions, with and without the context layer, attention pooling or a plain average) and the field models (crops
 * masked to the nucleus, with and without positions and the context layers); the language models and the
 * segmentation network ship from their own trainers.
 *   node tools/export_models.js [--only classic|foundation|slides|fields] [--chromium /path/to/chromium]
 * Needs playwright-core (npm install playwright-core, or NODE_PATH pointing at a node_modules that has it) and a
 * Chromium: --chromium, $CHROMIUM, or Playwright's own install.
 * With --only, the other parts of the file are kept as they are.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');
const ROOT = path.join(__dirname, '..'), OUT = path.join(ROOT, 'data', 'trained'), FILE = path.join(OUT, 'trained_models.js');
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const only = arg('--only', null), exe = arg('--chromium', process.env.CHROMIUM || '/opt/pw-browsers/chromium');
const pc = v => `${Math.round(v * 100)}%`;
const roundHist = hist => hist.map(h => { const o = {}; for (const [k, v] of Object.entries(h)) o[k] = typeof v === 'number' ? +v.toFixed(4) : v; return o; });
const existing = () => { if (!fs.existsSync(FILE)) return null; const w = {}; new Function('window', fs.readFileSync(FILE, 'utf8'))(w); return w.TRAINED_MODELS || null; };

(async () => {
  const out = Object.assign({ classic: {}, foundation: null, slides: {}, fields: {} }, only ? existing() || {} : {});
  const browser = await chromium.launch({ executablePath: exe, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.goto('file://' + path.join(ROOT, 'index.html') + '#leukemia/train', { waitUntil: 'load' });
  await page.waitForFunction(() => window.NUCLEUS_NET && window.NUCLEUS_NET.state.net, null, { timeout: 120000 });
  const run = (fn, a) => page.evaluate(fn, a);
  const until = (fn, timeout = 1800000) => page.waitForFunction(fn, null, { timeout, polling: 250 });
  const want = part => !only || only === part;
  const t00 = Date.now(), stamp = () => `${((Date.now() - t00) / 1000).toFixed(0)} s`;

  if (want('classic')) for (const k of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]) {
    await page.selectOption('#recipe-select', String(k));
    await until(() => { const S = window.NUCLEUS_NET.state; return !S.applyingRecipe && S.net && S.epoch === 0 && !S.running; });
    await run(() => { window.NUCLEUS_NET.state.speed = 1e6; });
    await page.click('#btn-train');
    await until(() => { const S = window.NUCLEUS_NET.state; return !S.running && S.epoch >= S.epochs; });
    const e = await run(k => {
      const S = window.NUCLEUS_NET.state, settings = {}; for (const key of ['task', 'mode', 'h1', 'h2', 'convK', 'activation', 'lr', 'batch', 'epochs', 'augment', 'l2', 'peek', 'labelNoise', 'trainLab', 'testLab', 'normalize', 'labelled', 'seed']) settings[key] = S[key];
      const last = S.history[S.history.length - 1];
      return { recipe: k, settings, seed: S.seed, epochs: S.epochs, model: S.net.toJSON(), hist: S.history, trainAcc: last.acc, testAcc: last.testAcc, trained: `recipe ${k} trained on this page, seed ${S.seed}, ${S.epochs} epochs` };
    }, k);
    e.hist = roundHist(e.hist); out.classic[k] = e;
    console.log(`classic recipe ${String(k).padStart(2)}: ${e.epochs} epochs · train ${pc(e.trainAcc)} · test ${pc(e.testAcc)} · ${e.model.W.reduce((a, w) => a + w.length, 0) + e.model.Wo.length + (e.model.Wc ? e.model.Wc.length : 0)} weights · ${stamp()}`);
  }

  if (want('foundation')) {
    await page.selectOption('#question-select', 'foundation'); await page.click('.stage[data-stage="train"]');
    await until(() => { const F = window.NUCLEUS_NET.state.fm; return F.cl && F.set && !F.running; });
    await page.click('#fm-reset'); await page.waitForTimeout(300);
    await run(() => { const F = window.NUCLEUS_NET.state.fm; F.speed = 1e6; });
    await page.click('#fm-train');
    await until(() => { const F = window.NUCLEUS_NET.state.fm; return !F.running && F.epoch >= F.epochs; });
    const e = await run(() => {
      const F = window.NUCLEUS_NET.state.fm, last = F.hist[F.hist.length - 1];
      return { seed: F.seed, epochs: F.epochs, labs: F.labs, model: F.cl.toJSON(F.std, { id: 'page-foundation', title: 'the 4 · Foundation encoder, pretrained on this page' }), hist: F.hist, pairAcc: last.pairAcc, trained: `pretrained on this page: the stage’s 100 nuclei, ${F.labs ? 'both labs’ scans' : 'our lab’s scans'}, seed ${F.seed}, ${F.epochs} epochs` };
    });
    e.hist = roundHist(e.hist); out.foundation = e;
    console.log(`foundation: ${e.epochs} epochs · held-out pair accuracy ${pc(e.pairAcc)} · probes ${JSON.stringify(e.hist[e.hist.length - 1])} · ${stamp()}`);
  }

  if (want('slides')) for (const qid of ['slides-atypia', 'slides-focus']) for (const att of [1, 0]) for (const ctx of [0, 1]) {
    await page.selectOption('#question-select', qid); await page.click('.stage[data-stage="train"]');
    await until(() => { const L = window.NUCLEUS_NET.state.sl; return L.built && L.model && !L.running; });
    await page.click(`#sl-pool-seg button[data-att="${att}"]`); await page.click(`#sl-context-seg button[data-ctx="${ctx}"]`); await page.waitForTimeout(300);
    await run(() => { window.NUCLEUS_NET.state.sl.speed = 1e6; });
    await page.click('#sl-reset'); await page.waitForTimeout(300);
    await page.click('#sl-train');
    await until(() => { const L = window.NUCLEUS_NET.state.sl; return !L.running && L.epoch >= L.epochs; });
    const e = await run(() => {
      const L = window.NUCLEUS_NET.state.sl, last = L.hist[L.hist.length - 1];
      return { key: window.NUCLEUS_NET.keys.slides(), question: L.question, context: L.context, attention: L.attention, units: L.units, seed: L.seed, epochs: L.epochs, lr: L.lr, model: L.model.toJSON(), hist: L.hist, testAcc: last.testAcc, testMass: last.testMass, trained: `${L.question === 'focus' ? 'the focus question' : 'the atypia question'}, ${L.context ? 'with' : 'without'} the context layer, ${L.attention ? 'attention pooling' : 'a plain average'}, trained on this page with seed ${L.seed} for ${L.epochs} epochs` };
    });
    e.hist = roundHist(e.hist); out.slides[e.key] = e;
    console.log(`slides ${e.key}: ${e.epochs} epochs · test ${pc(e.testAcc)} · attention on the atypical nuclei ${e.testMass == null ? '–' : pc(e.testMass)} · ${stamp()}`);
  }

  if (want('fields')) {
    await page.selectOption('#question-select', 'fields'); await page.click('.stage[data-stage="train"]');
    await until(() => { const L = window.NUCLEUS_NET.state.fd; return L.built && L.model && !L.running; }, 900000);
    for (const pos of [1, 0]) for (const ctx of [1, 0]) {
      await page.click(`#fd-pos-seg button[data-pos="${pos}"]`); await page.click(`#fd-ctx-seg button[data-ctx="${ctx}"]`); await page.waitForTimeout(500);
      await until(() => { const L = window.NUCLEUS_NET.state.fd; return L.model && !L.running; });
      await run(() => { window.NUCLEUS_NET.state.fd.speed = 1e6; });
      await page.click('#fd-reset'); await page.waitForTimeout(500);
      await page.click('#fd-train');
      await until(() => { const L = window.NUCLEUS_NET.state.fd; return !L.running && L.epoch >= L.epochs; }, 3600000);
      const e = await run(() => {
        const L = window.NUCLEUS_NET.state.fd, last = L.hist[L.hist.length - 1];
        return { key: window.NUCLEUS_NET.keys.fields(), crop: L.crop, pos: L.pos, ctx: L.ctx, units: L.units, seed: L.seed, epochs: L.epochs, lr: L.lr, decay: L.decay, clip: L.clip, model: L.model.toJSON(), hist: L.hist, testAcc: last.testAcc, testAccCis: last.testAccCis, trained: `crops ${L.crop === 'nucleus' ? 'masked to the nucleus' : 'with the field around the nucleus'}, ${L.pos ? 'with' : 'without'} positions, ${L.ctx ? 'with' : 'without'} the context layers, trained on this page with seed ${L.seed} for ${L.epochs} epochs` };
      });
      e.hist = roundHist(e.hist); out.fields[e.key] = e;
      console.log(`fields ${e.key}: ${e.epochs} epochs · test invasion ${pc(e.testAcc)} · CIS ${pc(e.testAccCis)} · ${stamp()}`);
    }
  }
  await browser.close();
  fs.mkdirSync(OUT, { recursive: true });
  const js = `// The trained models that the page's "Load the trained model" buttons bring back, one per lesson, trained on the page\n// itself by tools/export_models.js (a headless browser running this very page), so that each is exactly what pressing\n// Train gives with the recipe's settings and seed, with the curves of that run. classic: by recipe number; foundation:\n// the 4 · Foundation encoder; slides and fields: by the key of their settings and encoder.\nwindow.TRAINED_MODELS = ${JSON.stringify(out)};\n`;
  fs.writeFileSync(FILE, js);
  console.log(`wrote ${FILE} (${(js.length / 1024).toFixed(0)} KB) · classic ${Object.keys(out.classic).length} · foundation ${out.foundation ? 1 : 0} · slides ${Object.keys(out.slides).length} · fields ${Object.keys(out.fields).length}`);
})().catch(e => { console.error('FAILED', e); process.exit(1); });
