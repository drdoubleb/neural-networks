# Nucleus Net

An interactive, single-page teaching tool for an *AI for pathologists* lecture. Residents build a small neural network,
watch it train with every node and weight on screen, then run it on cases it has never seen. The page starts with a
clinical-pathology question (a blood count: leukaemia or not?) where the whole network fits on one screen, moves to
nuclei (first as six hand-made measurements, then as raw pixels), and takes the network from a single layer on raw
pixels to a convolutional network, with the reasons for each step visible along the way. No installation, no server: open `index.html` in a browser.

![Contour irregularity task: all 100 nuclei, blue frames regular, orange irregular](data/irregularity/contact_sheet.png)

## Running it

- **Locally:** double-click `index.html` (plain HTML/CSS/JS; it works from a `file://` URL).
- **One file to email or drop on a USB stick:** `dist/nucleus-net.html` has the stylesheet, scripts and all the cases
  inlined. Rebuild it with `node tools/build_single_file.js`.
- **GitHub Pages:** repository settings → Pages → *Deploy from a branch* → the default branch, root folder.

Keys during a lecture: `1` `2` `3` switch stages, `space` trains/pauses, `T` or `N` teaches the next case one step
(Train) and `N` classifies the next test case (Test).

The masthead has two dropdowns: the **question** (which dataset) and the **lecture recipe** (a one-click preset for
each step of the arc, which also switches the question if needed). The Train stage shows only the essential controls
(input, convolution, hidden units, and the training buttons in the order a resident should try them: Teach next case,
Step batch, Step epoch, Complete training, Reset); everything else lives under *Advanced settings*. Controls that
do not apply to the current question are hidden.

## Four questions, one page

The **question** dropdown switches between four datasets, each with its own synthetic cases (80/20 split):

| Question | Cases | Positive class | What separates the classes | Decoys |
|---|---|---|---|---|
| **Leukaemia?** | 200 blood counts (160 / 40) | Leukaemia (acute, CML, CLL) vs No leukaemia (normal, bacterial infection, viral lymphocytosis, iron deficiency) | the pattern across ten CBC parameters | a raised white count on its own, cytopenias on their own |
| **Atypical nucleus?** | 100 nuclei (80 / 20) | Atypical (enlarged, hyperchromatic, irregular contour or coarse chromatin, alone or combined; six nuclei carry each trait on its own) vs Bland | size, darkness, contour and texture, in any combination | elongation, rotation, nucleolus, position |
| **Enlarged and hyperchromatic?** | 100 nuclei (80 / 20) | Enlarged (large, dark) vs Bland (small, pale) | size and darkness | contour shape (half of each class is irregular), elongation, rotation, texture, nucleolus, position |
| **Irregular contour?** | 100 nuclei (80 / 20) | Irregular (lobulated, notched, jagged) vs Regular (smooth ellipse) | the contour only | size, elongation, rotation, darkness, texture, nucleolus, position |

Every case also carries a hidden **subtype** (the kind of blood count, the traits of an atypical nucleus, or the contour
style). The network never sees it; the page uses it to show which hidden units respond to which kind of case.

## The three stages

1. **Specimens.** All cases with their ground truth (test labels hidden until a lecturer's checkbox reveals them).
   Blood counts appear as fingerprint cards (one bar per parameter, up = above the reference range); nuclei as images.
   Click a case to inspect it: for a blood count a mock report with reference ranges and H/L flags, for a nucleus the
   image, its six morphometric measurements and an overlay showing exactly where each comes from. A scatter plot shows
   how separable any two parameters are. For the nucleus questions, a *Where the cases come from* card says which lab's
   scans each set uses (see *Another lab, and the shortcut* below).
2. **Train.** Choose the input (the measurements or blood count, or all 1,024 raw pixels), tick which inputs the network
   may use (withhold blasts and watch it lean on cytopenias), an optional convolutional layer (4 or 8 filters of 5 × 5,
   ReLU, 4 × 4 max-pooling), zero to two dense hidden layers with a ReLU / sigmoid / tanh activation, and the learning
   rate, batch size, epochs, seed, weight decay and flip/rotation augmentation. Teach one case, step one batch or one
   epoch, or press *Complete training*. The diagram redraws every frame: connection thickness and colour show each weight, node fill shows
   the value flowing through for the selected case, first-layer weights on pixels are drawn as 32 × 32 maps, and with a
   convolution you see the learned filters, their feature maps lighting up on the nucleus, the pooled maps, and each
   hidden unit's weights over the pooled cells drawn as one stacked map, with its product map once a case has run.
   Connections are drawn against a fixed scale so they visibly grow as the network learns, and a bright core marks the
   ones the last training step moved most. A single-layer network on pixels also shows the **weight × pixel** product
   map whose sum is the score, and every pixel-fed hidden unit shows its own weight × pixel map beside its weight map.
   Each row reads **image × weights = weight × pixel** (the image arrives along the band, pixel by pixel), with the
   activation badge after it (sum plus bias, through the activation). Hovering a feature-map pixel of a convolutional network shows the arithmetic of that
   position under the image: the 5 × 5 patch, the filter, their products, the sum and the ReLU. A
   single-layer network is also shown as a **weighted checklist** (every weight, largest first). **Teach next case** shows
   back-propagation on one training case as a six-step walk-through (five without a hidden layer), each step named in a
   strip above the diagram: (1) the forward pass, each dense connection wiping from source to target with thickness =
   weight × the value at its start and the colour of its sign, so each node's sum can be read off, hop by hop to the output
   (on pixels the mean training nucleus is first subtracted from this one, so the network sees the difference, which is
   large at the membrane and near zero in the centre, and the three tiles stay up for the rest of the walk-through; that
   difference is laid over each weight map, the product map forms cell by cell and is set aside, a scan line
   sums it with orange cells against blue, and the bias and the ReLU, or the sigmoid at the output, finish the unit;
   the first unit runs slowly, the rest together); (2) the loss, the call against
   the truth on a scale beside the output, with the loss curve above it and the error (p − y) as its slope; (3) the
   backward pass, the error wiping back along each connection as error × weight (thick = large, blue: the unit should
   come down, orange: go up), one layer at a time, and a pill on each unit spelling out what arrived and what ReLU let
   through (a unit that was switched off gets none); (4) the gradients, blame at one end of a connection × activity at
   the other, drawn as a glow on each connection with the arithmetic written on the connections of small layers and, on
   pixels, as a copy of the nucleus scaled by the unit's blame that comes back from the blame side onto each weight map;
   (5) the update, the connections and maps visibly moving to w − learning rate × gradient, each labelled connection
   showing its weight before → after; (6) the check, the same case running forward again with the new weights, a replay
   to show what the step did rather than a part of training, which moves straight on to the next case. The step is real
   training, a batch of one at the current learning rate. Each press teaches the next case in the training set (or the case you
   picked in the tray; with label noise on, the lesson teaches the label the case was given and says so when it is wrong). For a convolutional network the forward pass and the check are the convolution walk-through
   (preprocessing, the filters scanning, the pooling, the pooled maps × each unit's weight map, the dense hops), and the
   backward steps go on past the dense layer: in the backward pass the blame of each pooled cell (Σ unit blame × its
   weight) wipes back along the bands and is drawn over the pooled maps, then goes back through the pooling onto the one
   position per block that won the max (dots on the feature maps; ReLU passes it only where the map was on, a hollow dot
   got none); in the gradient step each unit's weight map receives a copy of the stacked pooled maps scaled by its blame,
   and then the filters, whose gradient adds up over every position that got blame (the blame there × the 5 × 5 image
   patch under the filter): filter 1 accumulates position by position, with the window on the image and the arithmetic
   under it (patch → × blame → Σ so far), the other filters all at once, and the step settles over each filter as a Δ
   until the update folds it in; in the update the filters visibly move too. Every walk-through (the lesson and
   *Classify next*) plays at its normal pace unless you touch the controls in its strip: pause/play, previous and next
   step, and a speed slider (space pauses, ← → step, N skips ahead); while paused, the step buttons show each step at
   its end. With hidden units, a
   heatmap shows **what each unit responds to**: its mean activation for each hidden subtype, its weight to the output
   and the inputs it weighs most, and the training tray can be coloured by the most active unit instead of by the call.
   Learning curves plot loss and accuracy per epoch, with the test set "peeking" to show over-fitting; when peeking, a
   dotted line marks the epoch of the lowest test loss so far (where early stopping on a held-out set would have
   stopped), and with label noise on, a dashed line marks the honest ceiling of training accuracy. *Label noise* in the
   advanced settings flips a share of the training labels, as a second pathologist might have called them; the trays and
   the inspector then show the label each case was given, and a checkbox marks the mislabelled ones.
3. **Test.** The weights are frozen. *Classify next* runs one held-out case as a walk-through: a forward pass through
   the frozen weights, each connection wiping with the product it carries while a strip names what is happening, then the call at the threshold,
   then the truth with a ✓ or ✗ on the diagram; *Classify all* scores the rest at once. A *Test cases from* switch swaps the
   held-out nuclei for the other lab's scans of the same nuclei (or a mix of both labs), and a table keeps the scores of
   every test set run since the weights last changed. For a convolutional network, *Classify next* walks through the
   convolution in about 45 seconds: the mean training nucleus is subtracted first (the three tiles stay up), filter 1
   alone scans the difference image slowly with the arithmetic of each position shown, the remaining filters sweep
   together at a quicker pace, then feature map 1 is pooled block by block (each 4 × 4 block shown with its largest value
   and the single pooled cell it becomes), the other maps follow together, then the dense layer runs exactly like the
   pixel hop: the pooled maps gather into one stacked map over hidden unit 1's weight map (its weights over the pooled
   cells, stacked the same way and drawn beside the pooled column at all times), the product map wipes in and slides
   aside, a scan line sums it while the unit counts, the bias and the ReLU finish it, and the other units follow
   together; the units' values then wipe to the output and the call is made. Press N or click the diagram to skip ahead.
   In any convolutional diagram, hovering a pooled cell outlines its source block, hovering a feature-map pixel marks the
   pooled cell it feeds, and hovering a cell of a stacked weight or product map names the pooled map and cell it belongs to. Accuracy, sensitivity, specificity and a confusion matrix
   accumulate; a decision-threshold slider shows the sensitivity/specificity trade-off, and a **prevalence** slider turns
   them into positive and negative predictive values for a realistic population.

The **inspector** on the right follows the selected nucleus through all three stages. Its *Evidence* view shows what the
network is weighing: on pixels, a per-pixel overlay (orange pushes toward the positive class, blue away from it), computed
by back-propagating the score to the input, so it works through hidden layers and the convolution too; on measurements, a
*push* bar per measurement. For a nucleus it also shows the same nucleus as scanned at both labs, with the network's
call for each scan.

## The lecture arc: the ten recipe buttons

Numbers are test-set accuracy, mean of three seeds, reproducible with `node tools/check_training.js`.

| Recipe | Parameters | Test accuracy | What it shows |
|---|---|---|---|
| ① Leukaemia · blood count · single layer | 11 | ~90% | The whole network on one screen. A perceptron is a weighted checklist: negative on neutrophil fraction, platelets and haemoglobin, positive on basophils, immature granulocytes, WBC and blasts. |
| ② Leukaemia · blood count · 3 ReLU units | 37 | ~95% | Hidden units organise themselves: one becomes an acute-leukaemia detector, one a CML detector, one a "looks healthy" detector with a negative weight; CLL is folded into a neighbour. Nobody told the network these types exist. |
| ③ Atypia · measurements · single layer | 7 | ~95% | One approach to images: measure what a pathologist would name (area, darkness, chromatin texture, solidity, contour roughness) and let a single layer weigh the numbers. The checklist reads like a grading scheme: every trait measurement earns a weight, because each trait appears on its own in some nuclei, and elongation, the decoy, stays near zero. |
| ④ Enlargement · pixels · single layer | 1,025 | 100% | The other approach: raw pixels. A basic network works on pixels when the answer is a whole-image template: the weight map becomes a nucleus-shaped stencil. |
| ⑤ Irregularity · pixels · single layer | 1,025 | ~60% | Same network, new question: training accuracy hits 100% while the test curve stays at chance. Memorisation. No single template can capture "a bump somewhere on the outline". Switching the input to measurements rescues it (~95%, the weights land on solidity and contour roughness), but someone had to invent those measurements. |
| ⑥ Irregularity · pixels · 4 ReLU + augmentation | 4,105 | ~77% | Hidden units as learned feature detectors; flips and rotations turn 80 images into 640 views (roughly worth 8× more real data). |
| ⑦ Irregularity · pixels · 4 + 4 ReLU + augmentation | 4,125 | ~82% | A second layer is "deep learning" but buys only a few points here: depth is not the missing ingredient. |
| ⑧ Irregularity · pixels · convolution + 4 ReLU + augmentation | 897 | ~88% | The same 5 × 5 filter slides over the whole image, so a bend in the membrane is detected wherever it is. Fewer parameters, far better generalisation. |
| ⑨ Irregularity · convolution · the shortcut | 897 | 100% on a test set with the same flaw, ~58% at our lab | The network of ⑧ trained on a badly collected set: every irregular nucleus was scanned at another lab with a weaker stain. It learns the stain instead of the contour, looks perfect on a test set split the same way, and fails on our lab's scans. See *Another lab, and the shortcut*. |
| ⑩ Atypia · measurements · 8 + 8 ReLU · a quarter of the training labels wrong | 137 | ~97% early, ~83% after 300 epochs | Overfitting in its textbook shape. With 20 of the 80 training labels flipped, the network first learns the rule (the test curve peaks early), then memorises the mislabelled cases: training accuracy climbs past the honest ceiling of 75% while the test curve falls. See *Label noise and overfitting*. |

The pixel recipes use four hidden units and four filters so that every weight map, filter and feature map stays legible on
a laptop screen. The sliders go to eight; over eight seeds, eight units score about 81% on ⑥ (four: 79%), 87% on ⑦
(four: 81%) and 94% on ⑧ (four: 93%).

## Another lab, and the shortcut

Every nucleus comes as two scans: ours, and the same nucleus as scanned at **another lab whose stain is weaker** (a paler
nucleus, less chromatin and membrane contrast, a touch paler background; the same shape, the same chromatin pattern, the
same optics). The *Where the cases come from* card in the Specimens stage says which scans each set uses: our lab, the
other lab, both labs mixed at random (half of each class from each lab), or both labs split by class (the positives from
the other lab, the negatives from ours). The Test stage repeats the test-set choice next to the threshold and keeps a
table of the scores of every test set run since the weights last changed; the inspector shows the selected nucleus as
scanned at both labs with the network's call for each; a checkbox marks the cases scanned at the other lab with a **B**.
Two teaching points come out of this:

- **Lab differences.** Train any image recipe at our lab, then switch the test cases to the other lab and classify
  again. The dense pixel networks collapse to chance: they learned templates of absolute ink, and a paler nucleus turns
  the difference image blue across its whole interior (the inspector's pair of scans shows the call flipping on the same
  nucleus). The measurements recipe loses twenty points because darkness and texture are read off the raw scan. The
  convolution barely drops, because its filters read edges. The remedy is **stain normalisation** (Advanced settings):
  *per lab*, each lab's typical background and nucleus levels, measured from its scans without any label, are matched to
  ours, the toy version of matching a slide's colour statistics to a reference slide; *per image*, each scan is rescaled
  by its own levels, which also throws away hyperchromasia as a feature. Normalised, every pixel recipe scores as well at
  the other lab as at home. The walk-throughs then show the normalised scan in the preprocessing row.
- **Shortcut learning (recipe ⑨).** The convolutional network of ⑧, trained on a set where every irregular nucleus was
  scanned at the other lab and every regular one at ours. Training accuracy hits 100% and so does the test set when it
  carries the same split, because the stain is a perfect shortcut to the label and easier to learn than the contour: the
  *Evidence* overlay shows the network weighing the interior of the nucleus rather than its outline. Switch the test cases
  to our lab and the sensitivity collapses (the irregular nuclei are well stained, so they are called regular); switch
  them to the other lab and the specificity collapses. This is the pen-mark and hospital-watermark story of the medical
  AI literature. Two things defuse it: stain normalisation per lab, and collecting both classes from both labs (training
  cases from *both labs, mixed at random*).

Test accuracy, mean of three seeds (`node tools/check_training.js`):

| Recipe, trained at our lab | at our lab | at the other lab | normalised per lab: our lab / the other lab |
|---|---|---|---|
| ③ Atypia · measurements · single layer | 95% | 75% | the measurements are read off the raw scan |
| ④ Enlargement · pixels · single layer | 100% | 92% | 100% / 100% |
| ⑤ Irregularity · pixels · single layer | 60% | 55% | 60% / 57% |
| ⑥ Irregularity · pixels · 4 ReLU + augmentation | 77% | 50% | 77% / 80% |
| ⑦ Irregularity · pixels · 4 + 4 ReLU + augmentation | 82% | 52% | 82% / 82% |
| ⑧ Irregularity · pixels · convolution + 4 ReLU + augmentation | 88% | 85% | 88% / 92% |

| ⑨ The convolution of ⑧, trained on… | test split by class | test at our lab | test at the other lab | test mixed |
|---|---|---|---|---|
| irregular from the other lab, regular from ours | 100% | 58% | 57% | 57% |
| the same, stain normalised per lab | 93% | 90% | 92% | 88% |
| both labs, mixed at random | 90% | 92% | 90% | 92% |

## Label noise and overfitting

None of the clean recipes overfits in the textbook way: the pixel networks that memorise never generalised in the first
place, so their test curve never had a peak to fall from, and the measurement and blood-count networks are too small,
with too clean a signal, to overfit at all. The test *loss* does rise once a pixel network without weight decay trains
on, from 0.49 at its minimum around epoch 10 to 0.9 by epoch 300 on recipe ⑥, while the test accuracy stays flat: the
network makes no more errors, it only grows more confident in the errors it makes.

Label noise brings the classic curve out, and it is a pathology-native reason for it: labels are a consensus with
disagreement in them. Recipe ⑩ trains the atypia measurements with 8 + 8 units and a quarter of the training labels
flipped (a fixed set of 20, so the lecture is reproducible whatever the seed). The network first learns the rule from
the majority of right labels, so the test accuracy, scored against the true labels, peaks early; then it memorises the
mislabelled cases one by one, training accuracy climbs past the **honest ceiling** of 75%, and the test curve falls while
the test loss rises. Tick *Mark the mislabelled cases* and watch the ✗ badges on them disappear as they are memorised.
The dotted **lowest test loss** marker shows where early stopping would have stopped. Weight decay, fewer units, or
stopping early all soften it; the labels stay wrong.

Mean of three seeds (`node tools/check_training.js`): test accuracy peaks at 97% (epochs 2 to 14) and ends at
83% after 300 epochs; the test loss goes from 0.31 at its minimum to 0.48; training accuracy ends at 88%,
above the 75% ceiling.

## The data

`tools/generate_cbc.js` (no dependencies) draws the 200 blood counts from a fixed seed into `data/leukaemia/`
(`patients_data.js` for the page, `patients.json` and `patients.csv` for you). Each patient comes from one of seven
subtypes with plausible teaching distributions, not population data: acute leukaemia (40% of them aleukaemic, with a low
count and few blasts), CML (very high count, basophilia, marked left shift), CLL (lymphocytosis), normal, bacterial
infection (neutrophilia, left shift, a quarter septic with low platelets), viral lymphocytosis and iron-deficiency
anaemia (microcytosis, reactive thrombocytosis). Counts are log-transformed before standardization.

`tools/generate_nuclei.js` (no dependencies) draws the three nucleus datasets from fixed seeds and writes, per task, into
`data/atypia/`, `data/enlargement/` and `data/irregularity/`:

- `images/train/*.png`, `images/test/*.png` — 32 × 32 8-bit grayscale PNGs (80 + 20, class in the file name)
- `images/other-lab/train/*.png`, `images/other-lab/test/*.png` — the same nuclei as scanned at the other lab (weaker stain)
- `nuclei_data.js` — both scans of every nucleus, base64-encoded, loaded by the page
- `nuclei.json` — labels, split and the generator parameters of every nucleus
- `contact_sheet.png`, `contact_sheet_other_lab.png` — all 100 at 4×, at each lab, for slides

Each nucleus is a dark ellipse on a pale, noisy background whose radius is modulated by low-order harmonics (lobulation),
localised clefts or blebs, or higher harmonics (jagged outlines). Chromatin texture, an optional nucleolus, a slightly
darker membrane and position jitter are added to every nucleus. In the atypia set, 24 of the 50 atypical nuclei carry
exactly one trait (six enlarged, six hyperchromatic, six irregular, six coarse) and 26 carry two to four. The other lab's
scan of a nucleus keeps its shape and its chromatin pattern and renders it with a weaker stain: the nucleus about 0.13
paler, chromatin clumps and the membrane rim at 60% contrast, the nucleolus fainter, the background a touch paler, with
its own pixel grain; our lab's images are unchanged by it.

The six measurements are computed from the pixels in the browser (`js/features.js`): Otsu threshold → largest component
→ sub-pixel contour by marching squares. *Area*, *elongation* (second moments), *darkness* and *texture* (mean and spread
of ink two pixels in from the membrane), *solidity* (area ÷ convex-hull area) and *contour roughness* (perimeter ÷
perimeter of the smooth ellipse with the same area and elongation). Standardization uses training-set statistics only.

## Code map

```
index.html                 the page
css/style.css              tokens (light + dark) and components
js/features.js             measurements from pixels (browser + Node)
js/nn.js                   the network: optional convolution, 0–2 dense layers, hand-written backprop, weight decay, one-case lessons (browser + Node)
js/dataset.js              decoding, blood-count fingerprints, the two labs' scans and source modes, stain normalisation, label noise, flip/rotation augmentation, standardized inputs, withheld inputs (browser + Node)
js/viz.js                  canvas + SVG drawing: images, fingerprints, weight maps, filters, feature maps, evidence overlays, network diagram, charts
js/app.js                  state, task switch, training loop, the three stages, the inspector, unit heatmap, prevalence
tools/generate_cbc.js      make the blood-count dataset
tools/generate_nuclei.js   make the nucleus datasets, each nucleus scanned at both labs
tools/check_training.js    gradient check + the eight recipes in Node
tools/build_single_file.js bundle everything into dist/nucleus-net.html
```
