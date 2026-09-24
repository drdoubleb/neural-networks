#!/usr/bin/env node
'use strict';
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
const fs = require('node:fs');
const root = path.join(__dirname, '..');
const shots = process.env.QA_OUTPUT_DIR;
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(path.join(root, 'index.html')).href + '#train');
    const steps = () => page.evaluate(() => window.qaNet.steps);
    const capture = () => page.evaluate(() => { window.qaNet = document.getElementById('net-canvas')._model.net; return JSON.stringify(window.qaNet); });
    const jump = title => page.locator('.bp-step').filter({ hasText: title }).click();
    for (let recipe = 1; recipe <= 7; recipe++) {
      await page.selectOption('#recipe-select', String(recipe));
      const original = await capture();
      await page.click('#btn-teach');
      assert.equal((await page.locator('.bp-step[aria-current="step"]').innerText()).replace(/\s/g, ''), '1Predict');
      assert.equal(await steps(), 0);
      assert.equal(await page.locator('#btn-lesson-play').innerText(), 'Play steps', 'Starts paused');
      await jump('Measure loss');
      assert.match(await page.locator('#lesson-details').innerText(), /p − y/);
      const hidden = recipe === 7 ? 2 : recipe === 2 || recipe === 6 ? 1 : 0;
      for (let layer = hidden; layer >= 1; layer--) {
        await jump(`Back through H${layer}`);
        assert.equal(await page.locator('.bp-explanation tbody tr').count(), recipe === 2 ? 3 : 4);
        assert.doesNotMatch(await page.locator('#lesson-details').innerText(), /NaN|undefined/);
        await page.locator('[data-bp-unit]').last().click();
      }
      await jump('Weight gradients');
      assert.match(await page.locator('#lesson-details').innerText(), /Bias data gradient/);
      if (recipe >= 4) {
        if (hidden) await page.selectOption('#bp-target', '0:0');
        const map = page.locator('[data-bp-map="0"]');
        await map.click({ position: { x: 80, y: 80 } });
        const input = +(await page.locator('#bp-input').inputValue());
        await page.locator('[data-bp-map="0"]').press('ArrowRight');
        assert.equal(+(await page.locator('#bp-input').inputValue()), input + 1);
      } else { await page.selectOption('#bp-input', '2'); }
      await jump('Update weights');
      assert.equal(await steps(), 0);
      assert.equal(await page.evaluate(() => JSON.stringify(window.qaNet)), original, 'Navigation never trains');
      if (recipe >= 4) assert.equal(await page.locator('[data-bp-map]').count(), 6);
      // Capture and verify the selected weight's predicted update against the real optimizer.
      const expected = await page.evaluate(() => {
        const les = document.getElementById('net-canvas')._model.lesson;
        const t = Backprop.targetInfo(les.res, les.target);
        return { l: t.l, j: t.j, i: les.input, n: t.input.length, step: Backprop.parameter(les.res, les.target, les.input, les.lr, les.l2), lr: les.lr };
      });
      if (recipe === 7) {
        // Changing a global setting cannot silently change the captured update.
        await page.locator('#advanced summary').click();
        await page.locator('#lr').fill('100'); await page.locator('#lr').dispatchEvent('input');
        await page.locator('#advanced summary').click();
      }
      await page.click('#btn-lesson-apply');
      assert.equal(await steps(), 1);
      const actual = await page.evaluate(q => q.l === 'out' ? window.qaNet.Wo[q.i] : window.qaNet.W[q.l][q.j * q.n + q.i], expected);
      assert.ok(Math.abs(actual - expected.step.after) < 1e-12);
      assert.match(await page.locator('#lesson-details').innerText(), /training-set loss/);
      await page.click('#btn-lesson-back');
      assert.ok(await page.locator('#btn-lesson-apply').isDisabled());
      await page.click('#btn-lesson-forward'); assert.equal(await steps(), 1);
      await page.click('#btn-lesson-close');
      assert.ok(await page.locator('#btn-teach-again').isVisible());
      if (recipe === 1) { await page.click('#btn-teach-again'); assert.equal(await steps(), 11); }
      console.log(`PASS recipe ${recipe}: navigation, gradients, selection, exact one update, replay`);
    }
    // Cancel, case selection, stage change, and model reset must not apply pending updates.
    await page.selectOption('#recipe-select', '2');
    await capture(); await page.click('#btn-teach'); await jump('Update weights'); await page.click('#btn-lesson-close'); assert.equal(await steps(), 0);
    await page.click('#btn-teach');
    await page.locator('#train-tray .thumb').nth(1).click();
    assert.ok(await page.locator('#lesson').isHidden()); assert.equal(await steps(), 0);
    await page.click('#btn-teach'); await page.locator('.stage[data-stage="test"]').click();
    assert.equal(await steps(), 0); assert.ok(await page.locator('#btn-teach').isDisabled());
    await page.locator('.stage[data-stage="train"]').click();
    await page.click('#btn-teach'); await page.click('#btn-reset'); assert.equal(await steps(), 0); assert.ok(await page.locator('#lesson').isHidden());
    await page.selectOption('#recipe-select', '8'); assert.ok(await page.locator('#btn-teach').isDisabled());
    // Autoplay and reduced-motion still stop before applying. No timed sleep in the test.
    await page.selectOption('#recipe-select', '1'); await capture();
    await page.clock.install(); await page.click('#btn-teach'); await page.selectOption('#lesson-playback','2');
    await page.click('#btn-lesson-play'); await page.clock.runFor(11000);
    assert.ok(await page.locator('#btn-lesson-apply').isVisible()); assert.equal(await steps(), 0);
    assert.equal(await page.locator('#btn-lesson-play').innerText(), 'Replay steps');
    await page.click('#btn-lesson-close');
    // Visual QA includes hidden layers, image updates, narrow/mobile and dark theme.
    if (shots) fs.mkdirSync(shots, { recursive: true });
    await page.selectOption('#recipe-select', '7'); await page.click('#btn-teach'); await jump('Back through H1');
    if (shots) { await page.locator('#lesson').scrollIntoViewIfNeeded(); await page.screenshot({ path: path.join(shots, 'hidden-desktop.png'), fullPage: true }); }
    await jump('Update weights');
    if (shots) { await page.locator('#lesson-details').scrollIntoViewIfNeeded(); await page.screenshot({ path: path.join(shots, 'pixels-desktop.png') }); }
    await page.click('#theme-toggle');
    if (shots) await page.screenshot({ path: path.join(shots, 'pixels-dark.png') });
    for (const width of [390, 768]) {
      await page.setViewportSize({ width, height: 844 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No page overflow at ${width}`);
      await page.locator('#lesson-details').scrollIntoViewIfNeeded();
      if (shots) await page.screenshot({ path: path.join(shots, `pixels-${width}.png`) });
    }
    assert.deepEqual(errors, []);
    const reduced = await browser.newPage({ reducedMotion: 'reduce', viewport: { width: 1200, height: 900 } });
    reduced.on('pageerror', e => errors.push(e.message));
    await reduced.goto(pathToFileURL(path.join(root, 'dist/nucleus-net.html')).href + '#train');
    await reduced.click('#btn-teach');
    assert.equal(await reduced.locator('#btn-lesson-play').innerText(), 'Play steps');
    await reduced.locator('.bp-step').filter({ hasText: 'Update weights' }).click();
    assert.equal(await reduced.evaluate(() => document.getElementById('net-canvas')._model.net.steps), 0);
    await reduced.click('#btn-lesson-apply');
    assert.equal(await reduced.evaluate(() => document.getElementById('net-canvas')._model.net.steps), 1);
    assert.deepEqual(errors, []);
    console.log('PASS: cancel/case/stage/reset isolation; CNN boundary; autoplay; dark/mobile layouts; reduced-motion portable build; no browser errors.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
