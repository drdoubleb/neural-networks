# Nucleus Net

An interactive, single-page teaching tool for an *AI for pathologists* lecture: residents build a tiny neural
network, watch it train on 80 synthetic nuclei — every node and weight on screen — and then run it on 20 nuclei it has
never seen. No installation, no server: open `index.html` in a browser.

![All 100 synthetic nuclei; blue frames are regular, orange are irregular](data/contact_sheet.png)

## Running it

- **Locally:** double-click `index.html` (everything is plain HTML/CSS/JS; it works from a `file://` URL).
- **One file to email or drop on a USB stick:** `dist/nucleus-net.html` has the stylesheet, scripts and all 100 images
  inlined. Rebuild it with `node tools/build_single_file.js`.
- **GitHub Pages:** in the repository settings, Pages → Source → *Deploy from a branch* → `main` / root. The page is
  then served at `https://<user>.github.io/<repo>/`.

Keys during a lecture: `1` `2` `3` switch stages, `space` trains/pauses, `N` classifies the next test nucleus.

## The three stages

1. **Specimens.** All 100 nuclei with their ground truth (test labels hidden by default; a lecturer's checkbox
   reveals them). Click a nucleus to inspect it: the image, the six morphometric measurements made on it, and an
   overlay showing exactly where each measurement comes from. A scatter plot shows how separable any two measurements are.
2. **Train.** Choose what the network sees (six measurements, or all 1,024 raw pixels), add 0–8 hidden units with a
   sigmoid / tanh / ReLU activation, and set the learning rate, batch size, epochs, seed, weight decay and
   augmentation. Press *Train* (or step one batch / one epoch at a time). The diagram redraws every frame: connection
   thickness and colour show each weight, node fill shows the value flowing through for the selected nucleus, and in pixel
   mode each hidden unit's 1,024 weights are drawn as a 32 × 32 map. The training tray shows the network's current call
   for every nucleus (frame colour) and flags disagreements with the truth; the last mini-batch is ringed. Learning curves
   plot loss and accuracy per epoch, optionally with the test set "peeking" to show over-fitting.
3. **Test.** The weights are frozen. The 20 held-out nuclei are classified one at a time (values animate through the
   network, then the truth is revealed) or all at once. Accuracy, sensitivity, specificity and a confusion matrix
   accumulate; a decision-threshold slider shows the sensitivity/specificity trade-off.

The **inspector** on the right follows the selected nucleus through all three stages. Its *Evidence* view shows what the
network is weighing: in pixel mode a per-pixel overlay (orange pixels push toward *Irregular*, blue toward *Regular*),
in measurement mode a *push* bar per measurement (weight × standardized value), plus the arithmetic from evidence to
score to probability.

## Suggested lecture flow (the "recipe" buttons)

| Recipe | What it shows |
|---|---|
| ① Measurements, single layer | Seven parameters. The weights on *solidity* and *contour roughness* grow, the decoys stay near zero. ~95 % on the test set. |
| ② Measurements, 4 tanh units | The same problem with a hidden layer — the full node-and-edge picture of a multi-layer network. |
| ③ Pixels, single layer | 1,025 parameters on 80 images: training accuracy races to 100 % while the (peeking) test curve stays near chance. Memorisation, not learning. The weight map shows why no single template can capture "irregular". |
| ④ Pixels, 8 ReLU units + augmentation + decay | Hidden units act as learned feature detectors, flips/rotations turn 80 images into 640 views, weight decay tames the fit. Typically 80–90 % on the test set. |

`node tools/check_training.js` reproduces these numbers in Node without a browser.

## The data

`tools/generate_nuclei.js` (no dependencies) draws the 100 nuclei from a fixed seed and writes:

- `data/images/train/*.png`, `data/images/test/*.png` — 32 × 32 8-bit grayscale PNGs (80 + 20, class in the file name)
- `data/nuclei_data.js` — the same pixels, base64-encoded, loaded by the page
- `data/nuclei.json` — labels, split and the generator parameters of every nucleus
- `data/contact_sheet.png` — all 100 at 4×, for slides

Design of the set: each nucleus is a dark ellipse on a pale, noisy background. Size, elongation, rotation, darkness,
chromatin texture, an optional nucleolus and position jitter are drawn from the **same** distributions for both classes.
The only systematic difference is the contour: *regular* nuclei are smooth ellipses; *irregular* nuclei are lobulated
(low-order harmonics), notched or blebbed (localised clefts/bumps), or finely jagged (higher harmonics). The split is
stratified: 40 + 40 train, 10 + 10 test.

The six measurements are computed from the pixels in the browser (`js/features.js`): Otsu threshold → largest component
→ sub-pixel contour by marching squares. *Area*, *elongation* (from second moments), *darkness* and *texture* (mean and
spread of ink two pixels in from the membrane), *solidity* (area ÷ convex-hull area) and *contour roughness* (perimeter ÷
perimeter of the smooth ellipse with the same area and elongation). Standardization uses training-set statistics only.

## Code map

```
index.html              the page
css/style.css           tokens (light + dark) and components
js/features.js          measurements from pixels (browser + Node)
js/nn.js                the network: forward pass, hand-written backprop, mini-batch SGD, weight decay (browser + Node)
js/dataset.js           decoding, augmentation, standardized inputs (browser + Node)
js/viz.js               canvas + SVG drawing: images, weight maps, evidence overlays, network diagram, charts
js/app.js               state, training loop, the three stages, the inspector
tools/generate_nuclei.js   make the dataset
tools/check_training.js    train in Node and print accuracies
tools/build_single_file.js bundle everything into dist/nucleus-net.html
```
