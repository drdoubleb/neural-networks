# Nucleus Net

An interactive, single-page teaching tool for an *AI for pathologists* lecture. Residents build a small neural network,
watch it train with every node and weight on screen, then run it on cases it has never seen. The page starts with a
clinical-pathology question (a blood count: leukemia or not?) where the whole network fits on one screen, moves to
nuclei (first as six hand-made measurements, then as raw pixels), and takes the network from a single layer on raw
pixels to a convolutional network, with the reasons for each step visible along the way, and ends by pretraining a
miniature foundation model on 100 unlabelled nuclei, live. No installation, no server: open `index.html` in a browser.

![Contour irregularity task: all 100 nuclei, blue frames regular, orange irregular](data/irregularity/contact_sheet.png)

## Running it

- **Locally:** double-click `index.html` (plain HTML/CSS/JS; it works from a `file://` URL).
- **One file to email or drop on a USB stick:** `dist/nucleus-net.html` has the stylesheet, scripts and all the cases
  inlined. Rebuild it with `node tools/build_single_file.js`.
- **GitHub Pages:** repository settings → Pages → *Deploy from a branch* → the default branch, root folder.
- **The numbers in this file:** `node tools/check_training.js` (the recipes, the other lab, the shortcut, label noise,
  the labelled-cases table), `node tools/check_foundation.js` (the foundation model's pretraining) and
  `node tools/pretrain_backbone.js --measure` (the shipped encoder's ablation) reproduce them.

Keys during a lecture: `1` `2` `3` `4` switch stages, `space` trains/pauses, `T` or `N` teaches the next case one step
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
| **Leukemia?** | 200 blood counts (160 / 40) | Leukemia (acute, CML, CLL) vs No leukemia (normal, bacterial infection, viral lymphocytosis, iron deficiency) | the pattern across ten CBC parameters | a raised white count on its own, cytopenias on their own |
| **Atypical nucleus?** | 100 nuclei (80 / 20) | Atypical (enlarged, hyperchromatic, irregular contour or coarse chromatin, alone or combined; six nuclei carry each trait on its own) vs Bland | size, darkness, contour and texture, in any combination | elongation, rotation, nucleolus, position |
| **Enlarged and hyperchromatic?** | 100 nuclei (80 / 20) | Enlarged (large, dark) vs Bland (small, pale) | size and darkness | contour shape (half of each class is irregular), elongation, rotation, texture, nucleolus, position |
| **Irregular contour?** | 100 nuclei (80 / 20) | Irregular (lobulated, notched, jagged) vs Regular (smooth ellipse) | the contour only | size, elongation, rotation, darkness, texture, nucleolus, position |

Every case also carries a hidden **subtype** (the kind of blood count, the traits of an atypical nucleus, or the contour
style). The network never sees it; the page uses it to show which hidden units respond to which kind of case.

## The four stages

1. **Specimens.** All cases with their ground truth (test labels hidden until a lecturer's checkbox reveals them).
   Blood counts appear as fingerprint cards (one bar per parameter, up = above the reference range); nuclei as images.
   Click a case to inspect it: for a blood count a mock report with reference ranges and H/L flags, for a nucleus the
   image, its six morphometric measurements and an overlay showing exactly where each comes from. A scatter plot shows
   how separable any two parameters are. For the nucleus questions, a *Where the cases come from* card says which lab's
   scans each set uses (see *Another lab, and the shortcut* below).
2. **Train.** Choose the input (the measurements or blood count, all 1,024 raw pixels, or the **code** from a foundation
   encoder: the one pretrained in stage 4 or the one shipped with the page), tick which inputs the network
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
   the inspector then show the label each case was given, and a checkbox marks the mislabelled ones. A *Labelled
   cases* control (4, 10, 20, 40 or all) withholds the labels of the rest of the training set: those cases stay in the
   tray, dimmed and dashed, the network never sees them and the training scores skip them, which is how the value of a
   pretrained code is shown (see *The code as an input*). With the code as input the inspector adds a card with the
   code's numbers and each one's push on the score, and its *Evidence* view carries that push back through the frozen
   encoder to the pixels.
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
4. **Foundation.** A miniature foundation model, pretrained live on 100 nuclei from all three image questions (34 / 33 /
   33 of their training sets) without a single label: a convolutional encoder (4 filters of 5 × 5, pool 4 × 4, a linear
   layer to a **code** of 8 numbers) trained by instance discrimination. Every batch takes 20 nuclei and two random
   *views* of each (a flip or rotation, and the other lab's scan of it when the checkbox allows); the loss pulls the two
   views of a nucleus together and pushes every other view of the batch away. The stage shows the rule as the game it
   is, *spot the same nucleus*: a line-up puts one view against every other view of the batch as a candidate, best
   first, with the encoder's vote for each (a softmax over the cosines of the codes) and the answer framed in orange;
   the encoder diagram shows the same pair live (the codes of both views as strips, their cosine, the three nearest
   other nuclei of the batch and the pair's share of the vote, which is the loss); a strip shows what the code keeps
   and ignores (one nucleus in all 8 orientations from both labs with the code of each view, at the start and now,
   then another nucleus); a similarity matrix shows the batch, every view against every other, with the pairs ringed;
   the tray of 100 can be coloured by question or by the answer to each nucleus's own question (which the model never
   saw); an embedding map plots any two code numbers for every nucleus and its second view. Two curves run
   per epoch: the contrastive loss on the batches and on 60 held-out nuclei, and *what the code is worth*: a single
   layer trained on the frozen code with 20 labelled cases per question, scored on 40 nuclei it never saw. See *A
   foundation model, in miniature*. The encoder then serves as the *code* input of stage 2 (see *The code as an input*).

The **inspector** on the right follows the selected nucleus through the first three stages. Its *Evidence* view shows what the
network is weighing: on pixels, a per-pixel overlay (orange pushes toward the positive class, blue away from it), computed
by back-propagating the score to the input, so it works through hidden layers and the convolution too; on measurements, a
*push* bar per measurement. For a nucleus it also shows the same nucleus as scanned at both labs, with the network's
call for each scan.

## The lecture arc: the twelve recipe buttons

Numbers are test-set accuracy, mean of three seeds, reproducible with `node tools/check_training.js`.

| Recipe | Parameters | Test accuracy | What it shows |
|---|---|---|---|
| ① Leukemia · blood count · single layer | 11 | ~90% | The whole network on one screen. A perceptron is a weighted checklist: negative on neutrophil fraction, platelets and hemoglobin, positive on basophils, immature granulocytes, WBC and blasts. |
| ② Leukemia · blood count · 3 ReLU units | 37 | ~95% | Hidden units organise themselves: one becomes an acute-leukemia detector, one a CML detector, one a "looks healthy" detector with a negative weight; CLL is folded into a neighbour. Nobody told the network these types exist. |
| ③ Atypia · measurements · single layer | 7 | ~95% | One approach to images: measure what a pathologist would name (area, darkness, chromatin texture, solidity, contour roughness) and let a single layer weigh the numbers. The checklist reads like a grading scheme: every trait measurement earns a weight, because each trait appears on its own in some nuclei, and elongation, the decoy, stays near zero. |
| ④ Atypia · measurements · 8 + 8 ReLU · a quarter of the training labels wrong | 137 | ~97% early, ~83% after 300 epochs | Overfitting in its textbook shape. With 20 of the 80 training labels flipped, the network first learns the rule (the test curve peaks early), then memorises the mislabelled cases: training accuracy climbs past the honest ceiling of 75% while the test curve falls. See *Label noise and overfitting*. |
| ⑤ Enlargement · pixels · single layer | 1,025 | 100% | The other approach: raw pixels. A basic network works on pixels when the answer is a whole-image template: the weight map becomes a nucleus-shaped stencil. |
| ⑥ Irregularity · pixels · single layer | 1,025 | ~60% | Same network, new question: training accuracy hits 100% while the test curve stays at chance. Memorisation. No single template can capture "a bump somewhere on the outline". Switching the input to measurements rescues it (~95%, the weights land on solidity and contour roughness), but someone had to invent those measurements. |
| ⑦ Irregularity · pixels · 4 ReLU + augmentation | 4,105 | ~77% | Hidden units as learned feature detectors; flips and rotations turn 80 images into 640 views (roughly worth 8× more real data). |
| ⑧ Irregularity · pixels · 4 + 4 ReLU + augmentation · then try the other lab | 4,125 | ~82% | A second layer is "deep learning" but buys only a few points here: depth is not the missing ingredient. The recipe to show the other lab on: switch the test cases to the other lab and it falls to chance, and stain normalisation repairs it (see *Another lab, and the shortcut*). |
| ⑨ Irregularity · pixels · 4 + 4 ReLU · the shortcut | 4,125 | 97% on a test set with the same flaw, ~55% at our lab | The network of ⑧ trained on a badly collected set: every irregular nucleus was scanned at another lab with a weaker stain. It learns the stain instead of the contour: the first-layer weight maps turn into plain interior templates, the *Evidence* view weighs the inside of the nucleus, the test set looks perfect when it is split the same way, and our lab's scans are called regular. See *Another lab, and the shortcut*. |
| ⑩ Irregularity · pixels · convolution + 4 ReLU + augmentation | 897 | ~88% | The same 5 × 5 filter slides over the whole image, so a bend in the membrane is detected wherever it is. Fewer parameters, far better generalisation. |
| ⑪ Foundation · pretrain a code on 100 unlabelled nuclei | 1,680 | a single layer on the code, 20 labelled cases per question: ~81% atypia, ~97% enlargement, ~83% irregularity | The foundation-model idea in miniature: pretrain once without labels, then every question is a small model on top of the code. The right-hand curve shows the code becoming worth more for every question as pretraining runs, though it was never told what any question asks. See *A foundation model, in miniature*. |
| ⑫ Irregularity · the foundation code · single layer · 10 labelled cases | 9 | ~82% with 10 labelled cases (pixels with the same 10: ~60%) | The payoff of the pretraining: the network sees 10 labelled nuclei only, each as the 8 numbers of the code from the encoder of ⑪ (or its shipped copy), and a single layer on them beats any pixel network trained on all 80. Move the *Labelled cases* control on recipe ⑥ to compare, and switch to the bigger shipped encoder at 40 cases. See *The code as an input*. |

The pixel recipes use four hidden units and four filters so that every weight map, filter and feature map stays legible on
a laptop screen. The sliders go to eight; over eight seeds, eight units score about 81% on ⑦ (four: 79%), 87% on ⑧
(four: 81%) and 94% on ⑩ (four: 93%).

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
- **Shortcut learning (recipe ⑨).** The 4 + 4 network of ⑧, trained on a set where every irregular nucleus was scanned
  at the other lab and every regular one at ours. Training accuracy hits 100% and so does the test set when it carries
  the same split, because the stain is a perfect shortcut to the label and easier to learn than the contour: the
  first-layer weight maps turn into plain interior templates instead of contour detectors, and the *Evidence* overlay
  shows the network weighing the interior of the nucleus rather than its outline (the convolution of ⑩ falls for it just
  the same; its numbers are in the second table below). Switch the test cases
  to our lab and the sensitivity collapses (the irregular nuclei are well stained, so they are called regular); switch
  them to the other lab and the specificity collapses. This is the pen-mark and hospital-watermark story of the medical
  AI literature. Two things defuse it: stain normalisation per lab, and collecting both classes from both labs (training
  cases from *both labs, mixed at random*).

Test accuracy, mean of three seeds (`node tools/check_training.js`):

| Recipe, trained at our lab | at our lab | at the other lab | normalised per lab: our lab / the other lab |
|---|---|---|---|
| ③ Atypia · measurements · single layer | 95% | 75% | the measurements are read off the raw scan |
| ⑤ Enlargement · pixels · single layer | 100% | 92% | 100% / 100% |
| ⑥ Irregularity · pixels · single layer | 60% | 55% | 60% / 57% |
| ⑦ Irregularity · pixels · 4 ReLU + augmentation | 77% | 50% | 77% / 80% |
| ⑧ Irregularity · pixels · 4 + 4 ReLU + augmentation | 82% | 52% | 82% / 82% |
| ⑩ Irregularity · pixels · convolution + 4 ReLU + augmentation | 88% | 85% | 88% / 92% |

| ⑨ The 4 + 4 network of ⑧, trained on… | test split by class | test at our lab | test at the other lab | test mixed |
|---|---|---|---|---|
| irregular from the other lab, regular from ours | 97% | 55% | 52% | 52% |
| the same, stain normalised per lab | 78% | 77% | 82% | 82% |
| both labs, mixed at random | 68% | 75% | 58% | 73% |

| The convolution of ⑩, trained the same way (no longer a recipe) | test split by class | test at our lab | test at the other lab | test mixed |
|---|---|---|---|---|
| irregular from the other lab, regular from ours | 100% | 58% | 57% | 57% |
| the same, stain normalised per lab | 93% | 90% | 92% | 88% |
| both labs, mixed at random | 90% | 92% | 90% | 92% |

## Label noise and overfitting

None of the clean recipes overfits in the textbook way: the pixel networks that memorise never generalised in the first
place, so their test curve never had a peak to fall from, and the measurement and blood-count networks are too small,
with too clean a signal, to overfit at all. The test *loss* does rise once a pixel network without weight decay trains
on, from 0.49 at its minimum around epoch 10 to 0.9 by epoch 300 on recipe ⑦, while the test accuracy stays flat: the
network makes no more errors, it only grows more confident in the errors it makes.

Label noise brings the classic curve out, and it is a pathology-native reason for it: labels are a consensus with
disagreement in them. Recipe ④ trains the atypia measurements with 8 + 8 units and a quarter of the training labels
flipped (a fixed set of 20, so the lecture is reproducible whatever the seed). The network first learns the rule from
the majority of right labels, so the test accuracy, scored against the true labels, peaks early; then it memorises the
mislabelled cases one by one, training accuracy climbs past the **honest ceiling** of 75%, and the test curve falls while
the test loss rises. Tick *Mark the mislabelled cases* and watch the ✗ badges on them disappear as they are memorised.
The dotted **lowest test loss** marker shows where early stopping would have stopped. Weight decay, fewer units, or
stopping early all soften it; the labels stay wrong.

Mean of three seeds (`node tools/check_training.js`): test accuracy peaks at 97% (epochs 2 to 14) and ends at
83% after 300 epochs; the test loss goes from 0.31 at its minimum to 0.48; training accuracy ends at 88%,
above the 75% ceiling.

## A foundation model, in miniature

Recipe ⑪ and the fourth stage. The idea of a foundation model is that the expensive part, learning what images are
made of, is done once, on many unlabelled images, and every later question is a small model on top of the resulting
**code**. The miniature keeps every part visible: 100 nuclei from the three image questions' training sets (34 / 33 /
33, our lab's scans, never a label), an encoder of 1,680 weights (4 filters of 5 × 5, pool 4 × 4, a linear layer to a
code of 8), and a pretraining rule with no labels in it, instance discrimination, which the page presents as the game
it is, **spot the same nucleus**: take a nucleus and make two views of it (a flip or rotation, or the other lab's scan);
the encoder turns each view into 8 numbers; among all the other views in the batch it must find the partner by
comparing the numbers; the weights are nudged so the partner scores higher next time. Nothing rebuilds the image. A
batch takes 20 nuclei with two random views each; for each view the loss is minus the log of its partner's share of a
softmax over every other view of the batch at temperature 0.2 (so it starts near −log(1/39) ≈ 3.7), and the gradient
runs back through the normalisation, the code layer and the convolution exactly as in the classifiers (learning rate
0.05, no weight decay, 100 epochs by default).

The pretext task matters. An autoencoder (the same encoder plus a linear decoder, trained to reconstruct the image) was
tried first and rejected: its code captures size and darkness but leaves contour irregularity at chance however long it
trains, because a smooth blob reconstructs almost as well as a lobulated one. The contrastive rule has to keep each
nucleus recognisable across flips, rotations and stains, and the contour is part of what does that. The class is still
in `js/nn.js` (`AutoEncoder`) for comparison.

What the code is worth is measured on the side after every epoch and never fed back: a single layer is trained on the
frozen code with 20 labelled training cases of a question (10 per class) and scored on 40 nuclei it never saw (the
question's 20 test nuclei and its 20 held-out training nuclei). Mean of three seeds (`node tools/check_foundation.js`):

| Epoch | Held-out loss | Partner found | Atypia | Enlargement | Irregularity |
|---|---|---|---|---|---|
| 0 (random encoder) | 3.44 | 15% | 74% | 80% | 60% |
| 10 | 2.31 | 39% | 84% | 98% | 61% |
| 25 | 1.87 | 63% | 82% | 98% | 79% |
| 50 | 1.62 | 70% | 85% | 98% | 80% |
| 100 | 1.40 | 81% | 81% | 97% | 83% |

*Partner found* is the share of held-out views whose nearest other view, by cosine, is their own partner. Enlargement is
readable even from a random code (size is total ink, which any filter passes on), atypia is mostly size and darkness
too, and contour irregularity is what the pretraining earns: from chance to about 80% with 20 labelled cases, where
recipe ⑦ needed all 80 labelled cases and their 640 augmented views to reach 77%. Scored on the other lab's scans of the
same nuclei, the single layer keeps 83% / 89% / 79%: the code was made to see past the stain. Untick *The other lab's
scans count as views* and it no longer is. The measurements cost about as much as the pretraining itself (some 80 ms
per epoch for both in Node); at the default speed of 4 epochs per second the 100 epochs take about half a minute.

## The code as an input, and how many labelled cases it needs

Recipe ⑫ and the *Labelled cases* control close the loop on why anyone pretrains. In stage 2 the third input, **the
code**, feeds the network the encoder's 8 (or 16) numbers for each nucleus instead of its pixels or measurements; the
*Code from* control picks the encoder: the one pretrained in stage 4, or one of two shipped with the page
(`data/foundation/backbones.js`, written by `tools/pretrain_backbone.js`): a saved copy of exactly what stage 4
makes with seed 1 (100 nuclei, 4 filters, a code of 8, 100 epochs), so the recipe works before anyone has run ⑪, and a
bigger encoder (the page's 240 training nuclei, 8 filters, a code of 16, 200 epochs). Everything else works as for the
measurements: the diagram shows the code numbers as inputs, the checklist lists their weights, the inspector adds a
card with each number's value and push, and its *Evidence* view carries the push back through the frozen encoder to
the pixels. The *Labelled cases* control (4, 10, 20, 40 or all) keeps the first n/2 of each class in the training order
as labelled and withholds the labels of the rest: they stay in the tray, dimmed and dashed, the network never trains
on them and the training scores skip them, though their images still serve to standardise the inputs, as an
unlabelled archive would.

Irregular contour, test accuracy at our lab, mean of five seeds (`node tools/check_training.js --only labelled --seeds 5`),
a single layer for 60 epochs except the augmented column (4 ReLU units, the recipe ⑦ network):

| Labelled cases | Measurements | Pixels | Pixels · 4 ReLU + augmentation | The code · 100 nuclei, 4 filters | The code · 240 nuclei, 8 filters |
|---|---|---|---|---|---|
| 4 | 93% | 53% | 49% | 78% | 56% |
| 10 | 84% | 60% | 64% | 82% | 58% |
| 20 | 91% | 69% | 74% | 71% | 73% |
| 40 | 95% | 66% | 59% | 80% | 85% |
| 80 (all) | 95% | 60% | 80% | 80% | 90% |

Three things to read off it. Pixels stay at chance however many labels there are, and the best pixel network of the
lecture needs all 80 cases and their augmented views to reach 80%. The code of the stage-4 encoder gets there with 4
to 10 labelled cases: pretraining on 100 unlabelled nuclei bought what 70 labels could not. And the hand-made
measurements win at every count, because solidity and contour roughness were designed for exactly this question; the
code is a general-purpose set of numbers that nobody had to design, and most questions in pathology come without such
measurements. With 20 test nuclei and one fixed draw of labelled cases the numbers are noisy: another draw of 10
labelled cases moves the code's score by ten points either way (over random draws it averages 73 to 75%), which is
why the recipe keeps its draw fixed.

**Does a bigger encoder help?** `node tools/pretrain_backbone.js --measure` pretrains five encoders and probes each
one the same way (a single layer on the frozen code, n labelled cases per question as random draws, scored on the 20
test nuclei); mean over the three questions and two pretraining seeds:

| Pretraining set · encoder | 4 | 10 | 20 | 40 | 80 | Other lab, 20 |
|---|---|---|---|---|---|---|
| The page's 100 nuclei · 4 filters, code of 8 · 100 epochs (stage 4; shipped) | 69% | 80% | 86% | 88% | 88% | 84% |
| The page's 240 training nuclei · 4 filters, code of 8 · 200 epochs | 63% | 75% | 83% | 86% | 86% | 83% |
| The page's 240 training nuclei · 8 filters, code of 16 · 200 epochs (shipped) | 66% | 75% | 89% | 91% | 93% | 87% |
| 2,100 generated nuclei · 4 filters, code of 8 · 60 epochs | 62% | 74% | 87% | 87% | 86% | 86% |
| 2,100 generated nuclei · 8 filters, code of 16 · 60 epochs | 64% | 74% | 86% | 88% | 92% | 85% |

Not much, and not at all where it would matter most. With 4 to 20 labelled cases every encoder lands within a few
points of the stage's own 100-nucleus one, which is the best of them at 4 and 10. Only with 40 to 80 labelled cases do
the 8-filter encoders pull ahead (91 to 93% against 88%), and the one pretrained on the page's 240 nuclei is the best
of those, which is why it ships as the second option. Twenty-one times more generated nuclei bought nothing: a hundred
nuclei already span everything this generator can draw, and longer pretraining makes the code better at telling
individuals apart rather than at the traits the questions ask about. In this miniature the scaling story of the real
foundation models does not reproduce, and the page says so rather than pretending otherwise.

## The data

`tools/generate_cbc.js` (no dependencies) draws the 200 blood counts from a fixed seed into `data/leukemia/`
(`patients_data.js` for the page, `patients.json` and `patients.csv` for you). Each patient comes from one of seven
subtypes with plausible teaching distributions, not population data: acute leukemia (40% of them aleukemic, with a low
count and few blasts), CML (very high count, basophilia, marked left shift), CLL (lymphocytosis), normal, bacterial
infection (neutrophilia, left shift, a quarter septic with low platelets), viral lymphocytosis and iron-deficiency
anemia (microcytosis, reactive thrombocytosis). Counts are log-transformed before standardization.

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
js/nn.js                   the network: optional convolution, 0–2 dense layers, hand-written backprop, weight decay, one-case lessons; the contrastive encoder (with weights that ship as JSON) and an autoencoder for the foundation model (browser + Node)
js/dataset.js              decoding, blood-count fingerprints, the two labs' scans and source modes, stain normalisation, label noise, flip/rotation augmentation, standardized inputs, withheld inputs, the code input, labelled-case subsets (browser + Node)
js/viz.js                  canvas + SVG drawing: images, fingerprints, weight maps, filters, feature maps, evidence overlays, network diagram, charts, the encoder's pair panel and similarity matrix
js/app.js                  state, task switch, training loop, the four stages (with the foundation model's pretraining loop and probes), the code input and its encoders, the inspector, unit heatmap, prevalence
data/foundation/backbones.js the foundation encoder shipped with the page: its weights and input standardiser, written by tools/pretrain_backbone.js
tools/generate_cbc.js      make the blood-count dataset
tools/generate_nuclei.js   make the nucleus datasets, each nucleus scanned at both labs (also a module for the pretraining script)
tools/pretrain_backbone.js pretrain the shipped encoder, and the ablation behind the choice (pretraining set × encoder size × labelled cases)
tools/check_training.js    gradient check + the ten recipes, the other lab, the shortcut and the label-noise curve in Node
tools/check_foundation.js  the foundation model's pretraining and what its code is worth, in Node
tools/build_single_file.js bundle everything into dist/nucleus-net.html
```
