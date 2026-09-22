# Nucleus Net

An interactive, single-page teaching tool for an *AI for pathologists* lecture. Residents build a small neural network,
watch it train on synthetic nuclei with every node and weight on screen, then run it on nuclei it has never seen. The
page takes the network from a single layer on raw pixels to a convolutional network, with the reasons for each step
visible along the way. No installation, no server: open `index.html` in a browser.

![Contour irregularity task: all 100 nuclei, blue frames regular, orange irregular](data/irregularity/contact_sheet.png)

## Running it

- **Locally:** double-click `index.html` (plain HTML/CSS/JS; it works from a `file://` URL).
- **One file to email or drop on a USB stick:** `dist/nucleus-net.html` has the stylesheet, scripts and all 200 images
  inlined. Rebuild it with `node tools/build_single_file.js`.
- **GitHub Pages:** repository settings → Pages → *Deploy from a branch* → the default branch, root folder.

Keys during a lecture: `1` `2` `3` switch stages, `space` trains/pauses, `N` classifies the next test nucleus.

## Two questions, one page

The Specimens stage has a switch between two questions, each with its own 100 nuclei (80 training, 20 held-out test,
stratified by class):

| Question | Positive class | What separates the classes | Decoys drawn identically in both classes |
|---|---|---|---|
| **Enlarged and hyperchromatic?** | Enlarged (large, dark) vs Bland (small, pale) | size and darkness | contour shape (half of each class is irregular), elongation, rotation, texture, nucleolus, position |
| **Irregular contour?** | Irregular (lobulated, notched, jagged) vs Regular (smooth ellipse) | the contour only | size, elongation, rotation, darkness, texture, nucleolus, position |

## The three stages

1. **Specimens.** All 100 nuclei with their ground truth (test labels hidden until a lecturer's checkbox reveals them).
   Click a nucleus to inspect it: the image, the six morphometric measurements made on it, and an overlay showing exactly
   where each measurement comes from. A scatter plot shows how separable any two measurements are.
2. **Train.** Choose the input (six measurements, or all 1,024 raw pixels), an optional convolutional layer (4 or 8
   filters of 5 × 5, ReLU, 4 × 4 max-pooling), zero to two dense hidden layers with a ReLU / sigmoid / tanh activation, and
   the learning rate, batch size, epochs, seed, weight decay and flip/rotation augmentation. Press *Train*, or step one
   batch or one epoch at a time. The diagram redraws every frame: connection thickness and colour show each weight, node
   fill shows the value flowing through for the selected nucleus, first-layer weights on pixels are drawn as 32 × 32 maps,
   and with a convolution you see the learned filters, their feature maps lighting up on the nucleus, and the pooled
   maps. The training tray shows the current call for every nucleus and flags disagreements with the truth; learning
   curves plot loss and accuracy per epoch, with the test set "peeking" to show over-fitting.
3. **Test.** The weights are frozen. The 20 held-out nuclei are classified one at a time (values animate through the
   network, then the truth is revealed) or all at once. Accuracy, sensitivity, specificity and a confusion matrix
   accumulate; a decision-threshold slider shows the sensitivity/specificity trade-off.

The **inspector** on the right follows the selected nucleus through all three stages. Its *Evidence* view shows what the
network is weighing: on pixels, a per-pixel overlay (orange pushes toward the positive class, blue away from it), computed
by back-propagating the score to the input, so it works through hidden layers and the convolution too; on measurements, a
*push* bar per measurement.

## The lecture arc: the six recipe buttons

Numbers are test-set accuracy, mean of three seeds, reproducible with `node tools/check_training.js`.

| Recipe | Parameters | Test accuracy | What it shows |
|---|---|---|---|
| ① Enlargement · pixels · single layer | 1,025 | 100% | A basic network works when the answer is a whole-image template: the weight map becomes a nucleus-shaped stencil. |
| ② Irregularity · pixels · single layer | 1,025 | ~60% | Same network, new question: training accuracy hits 100% while the test curve stays at chance. Memorisation. No single template can capture "a bump somewhere on the outline". |
| ③ Irregularity · measurements · single layer | 7 | ~95% | Hand-made features rescue it: the weights land on solidity and contour roughness, the decoys stay near zero. |
| ④ Irregularity · pixels · 8 ReLU + augmentation | 8,209 | ~90% | Hidden units as learned feature detectors; flips and rotations turn 80 images into 640 views (roughly worth 8× more real data). |
| ⑤ Irregularity · pixels · 8 + 8 ReLU + augmentation | 8,281 | ~88% | A second layer is "deep learning" but buys only a few points here: depth is not the missing ingredient. |
| ⑥ Irregularity · pixels · convolution + 8 ReLU + augmentation | 3,361 | ~93% | The same 5 × 5 filter slides over the whole image, so a bend in the membrane is detected wherever it is. Fewer parameters, far better generalisation. |

## The data

`tools/generate_nuclei.js` (no dependencies) draws both datasets from fixed seeds and writes, per task, into
`data/enlargement/` and `data/irregularity/`:

- `images/train/*.png`, `images/test/*.png` — 32 × 32 8-bit grayscale PNGs (80 + 20, class in the file name)
- `nuclei_data.js` — the same pixels, base64-encoded, loaded by the page
- `nuclei.json` — labels, split and the generator parameters of every nucleus
- `contact_sheet.png` — all 100 at 4×, for slides

Each nucleus is a dark ellipse on a pale, noisy background whose radius is modulated by low-order harmonics (lobulation),
localised clefts or blebs, or higher harmonics (jagged outlines). Chromatin texture, an optional nucleolus, a slightly
darker membrane and position jitter are added to every nucleus.

The six measurements are computed from the pixels in the browser (`js/features.js`): Otsu threshold → largest component
→ sub-pixel contour by marching squares. *Area*, *elongation* (second moments), *darkness* and *texture* (mean and spread
of ink two pixels in from the membrane), *solidity* (area ÷ convex-hull area) and *contour roughness* (perimeter ÷
perimeter of the smooth ellipse with the same area and elongation). Standardization uses training-set statistics only.

## Code map

```
index.html                 the page
css/style.css              tokens (light + dark) and components
js/features.js             measurements from pixels (browser + Node)
js/nn.js                   the network: optional convolution, 0–2 dense layers, hand-written backprop, weight decay (browser + Node)
js/dataset.js              decoding, flip/rotation augmentation, standardized inputs (browser + Node)
js/viz.js                  canvas + SVG drawing: images, weight maps, filters, feature maps, evidence overlays, network diagram, charts
js/app.js                  state, task switch, training loop, the three stages, the inspector
tools/generate_nuclei.js   make both datasets
tools/check_training.js    gradient check + the six recipes in Node
tools/build_single_file.js bundle everything into dist/nucleus-net.html
```
