/*
 * glossary.js — one definition per recurring term, shown wherever the term appears in the page's prose.
 *
 * The terms are underlined where they first appear in a paragraph or a note (one underline per term per paragraph),
 * and hovering one shows its definition in a small tooltip; the Glossary button in the masthead opens the whole list.
 * The page rewrites most of its notes as a model trains, so a MutationObserver re-marks any paragraph whose text
 * changed. The underlines can be turned off (the choice is kept in this browser); the list stays.
 */
window.Glossary = (function () {
  'use strict';
  // t: the term as listed; also: other forms that mean the same (plurals, the phrases the page uses); d: the definition
  const TERMS = [
    { t: 'loss', also: ['losses', 'test loss', 'training loss', 'train loss'], d: 'The number training pushes down: how far the network’s answer is from the right one, averaged over the cases of a batch (minus the log of the probability it gave the right answer). Zero would be perfect. The test loss is the same number on cases it never trained on.' },
    { t: 'cross-entropy', d: 'The loss used for probabilities: minus the log of the probability the network gave the right answer. Confident and wrong costs a lot; hedged and wrong costs less.' },
    { t: 'nat', also: ['nats'], d: 'The unit of the loss: a loss of one nat means the right answer was given a probability of 1/e, about 37%. Two nats, about 14%.' },
    { t: 'epoch', also: ['epochs'], d: 'One pass through every training case, in batches. The curves have one point per epoch.' },
    { t: 'batch', also: ['batches'], d: 'The handful of cases whose errors are averaged for one step of training. Smaller batches mean noisier, more frequent steps.' },
    { t: 'gradient', also: ['gradients'], d: 'For every weight, how much the loss would change if that weight moved a little: the direction to step in. Backpropagation computes all of them from one pass, by the chain rule.' },
    { t: 'backpropagation', also: ['backprop'], d: 'Working the error backward through the layers to get every weight’s gradient from one forward pass: the chain rule, applied layer by layer.' },
    { t: 'learning rate', d: 'How big a step training takes along the gradient. Too small and it crawls; too large and it overshoots and wanders.' },
    { t: 'weight decay', d: 'A small pull of every weight toward zero at each step, so that no weight grows large without earning it. A form of regularisation.' },
    { t: 'gradient clipping', also: ['clip', 'clipping', 'clipped'], d: 'Scaling a gradient down when its size exceeds a limit, so that one bad batch cannot throw the weights.' },
    { t: 'Adam', d: 'The optimiser the language models train with: every weight gets its own step size, set from the running size of its gradients, so rare words’ rows move as readily as common ones’.' },
    { t: 'softmax', d: 'Turns a row of scores into probabilities that add up to 100%: each score is exponentiated and divided by the sum. Large gaps between scores become near-certainties.' },
    { t: 'weights', also: ['weight', 'parameters', 'parameter count'], d: 'The numbers the network learns: a multiplier on every connection, plus a bias per unit. The parameter count is how many there are.' },
    { t: 'bias', also: ['biases'], d: 'The one number added to a unit’s weighted sum before the activation: its threshold, learned like any other weight.' },
    { t: 'activation', also: ['activations'], d: 'The function a unit applies to its weighted sum (ReLU, tanh, sigmoid). Without one, stacked layers would collapse into a single linear map.' },
    { t: 'ReLU', d: 'Rectified linear unit: the activation that passes positive sums through unchanged and sets negative ones to zero.' },
    { t: 'tanh', d: 'The activation that squashes a sum smoothly to between −1 and 1.' },
    { t: 'sigmoid', d: 'The squash to between 0 and 1, used at the output for a yes-or-no probability.' },
    { t: 'unit', also: ['units', 'neuron', 'neurons'], d: 'One cell of a layer: a weighted sum of its inputs, a bias, an activation. The network diagrams draw one circle per unit.' },
    { t: 'hidden layer', also: ['hidden layers', 'hidden units'], d: 'A layer between the input and the output. Its units are not told what to represent; they find whatever features make the output’s job easier.' },
    { t: 'dense', also: ['fully connected'], d: 'A layer where every unit sees every value of the layer before it.' },
    { t: 'convolution', also: ['convolutional', 'convolutions'], d: 'A small filter slid over the image with the same weights at every position, so that a feature is found wherever it lies. Its output is a feature map.' },
    { t: 'filter', also: ['filters', 'kernel'], d: 'The small grid of weights a convolution slides over the image. Each filter looks for one pattern (an edge, a dark blob) and writes a feature map.' },
    { t: 'feature map', also: ['feature maps'], d: 'The image of a filter’s responses: bright where the filter’s pattern is present.' },
    { t: 'pooling', also: ['max-pool', 'max-pooling'], d: 'Shrinking a feature map by keeping the maximum over small windows, so that what matters is whether a feature is present, not exactly where.' },
    { t: 'overfitting', also: ['overfit', 'overfits', 'overfitted'], d: 'Learning the training cases themselves, their noise and their quirks, rather than the rule behind them: the training loss keeps falling while the test loss turns up.' },
    { t: 'regularisation', also: ['regularise', 'regularised', 'regulariser'], d: 'Anything that holds a network back from fitting the training cases too closely: weight decay, augmentation, a smaller network, stopping early.' },
    { t: 'augmentation', also: ['augment', 'augmented'], d: 'Making more training cases from the ones there are, by changes that keep the label: here the seven other flips and rotations of every image.' },
    { t: 'early stopping', d: 'Stopping training at the epoch where the loss on held-out cases is lowest, before the network goes on to fit the training cases’ noise. Done properly, those cases are a validation set split from the training data, not the test set, which is kept untouched for the final score; this page has only the two sets, so its test set stands in.' },
    { t: 'label noise', d: 'Labels that are wrong in the training set. A network learns them too, up to a point, and its ceiling drops.' },
    { t: 'test set', also: ['test cases', 'test reports', 'test transcripts', 'test nuclei'], d: 'Cases set aside before training and never trained on, to measure what the network learned rather than what it memorised.' },
    { t: 'held-out', also: ['held out', 'held-out cases', 'held-out reports'], d: 'Cases, or a whole combination of findings, kept out of training on purpose, to ask what the network does with something it has never seen.' },
    { t: 'accuracy', d: 'The share of cases called right. It hides how confident the calls were; the loss does not.' },
    { t: 'calibration', also: ['calibrated', 'calibrate'], d: 'Whether the probabilities mean what they say: of the calls made at 80% confidence, about 80% should be right.' },
    { t: 'deferral', also: ['defer', 'deferred', 'defers'], d: 'Declining to call a case when the network’s confidence is below a threshold, so that it goes to a person instead; coverage is the share still called.' },
    { t: 'threshold', also: ['thresholds'], d: 'The cut on a probability above which the call is positive. Moving it trades sensitivity for specificity.' },
    { t: 'seed', also: ['random seed', 'seeds'], d: 'The number that starts the random generator. The same seed gives the same starting weights and the same shuffles, so a run can be repeated exactly.' },
    { t: 'stain normalisation', also: ['stain-normalised', 'normalisation'], d: 'Mapping one lab’s ink levels onto another’s before the network sees the image, so that a difference in staining is not learned as a difference in biology.' },
    { t: 'encoder', also: ['encoders'], d: 'The part of a network that turns an input, a nucleus, into a short vector: its code.' },
    { t: 'code', also: ['codes'], unless: /(?:and|or|source|write|writing|of)\s$/i, d: 'The vector a network makes of an input: a short description in numbers that the next part reads. The foundation question trains an encoder for it without labels, then reuses it.' },
    { t: 'contrastive', d: 'Training an encoder with no labels: two views of the same nucleus (a flip, a rotation) must get the same code, and different nuclei different codes.' },
    { t: 'foundation model', also: ['foundation'], d: 'A network trained on unlabelled data for a general task, whose code for an image is then reused for many questions with few labels each.' },
    { t: 'backbone', also: ['backbones'], d: 'A pretrained encoder reused under a new, small head.' },
    { t: 'pretraining', also: ['pretrain', 'pretrained'], d: 'Training on a large plain corpus, or on unlabelled images, before the task that matters. Most of what a language model knows comes from predicting the next token this way.' },
    { t: 'attention pooling', also: ['multiple instance'], d: 'Deciding about a whole slide from its nuclei: each nucleus’s code is weighted by how much it matters, with the weights learned, and the weighted sum is what the call is made from.' },
    { t: 'token', also: ['tokens'], d: 'The unit a language model reads and writes: here a word or a punctuation mark, looked up in a fixed vocabulary. The real models use sub-word pieces.' },
    { t: 'vocabulary', d: 'The list of tokens the model knows, each with its number. A word outside it becomes the unknown token ⟨?⟩.' },
    { t: 'sub-word', also: ['sub-words', 'byte-pair', 'byte pair'], d: 'Splitting words into pieces learned from the corpus, so that common words are one piece, rare ones several, and a word never seen can still be written.' },
    { t: 'embedding', also: ['embeddings', 'lookup table'], d: 'The table with one learned vector per token. A token’s row is what the model starts from, and tokens used alike end up with rows alike.' },
    { t: 'position vector', also: ['position vectors', 'learned positions'], d: 'A learned vector per position in the text, added to the token’s row, so that the model can tell the first word from the fortieth.' },
    { t: 'attention', d: 'A step where each token takes a weighted look at the tokens before it: the weights, shares that add up to 100%, say whom it reads, and what it reads is added to its own vector.' },
    { t: 'query map', also: ['query maps', 'query', 'queries'], d: 'In attention, what a token asks for: a vector it is mapped to, compared with every earlier token’s key.' },
    { t: 'key map', also: ['key maps', 'keys'], d: 'In attention, how a token presents itself to later tokens: a vector it is mapped to. Query · key is the match that sets the attention share.' },
    { t: 'value map', also: ['value maps', 'value vectors'], d: 'In attention, what a token says when read: a vector it is mapped to, added to the reader in proportion to the attention share.' },
    { t: 'head', also: ['heads', 'near head', 'far head'], d: 'One attention question put to every earlier token, with its own query, key and value maps. A layer has several heads asking at once; here the near head and the far head differ in their distance cost.' },
    { t: 'distance cost', also: ['distance costs'], d: 'A learned penalty per word of distance back, subtracted from an attention match. Steep, and a head reads only the words just before; flat, and it reads anywhere.' },
    { t: 'context layer', also: ['context layers'], d: 'Attention followed by a small feed-forward network, applied to every token: the block that lets a token’s vector depend on the tokens around it. Stacked, these are the layers of a transformer.' },
    { t: 'feed-forward', d: 'The small dense network applied to each token’s vector on its own after attention, the same weights at every position.' },
    { t: 'causal', d: 'Reading only the tokens before, never after, so that the model can write one token at a time.' },
    { t: 'output layer', d: 'The last layer. In the language models, a row of weights per token in the vocabulary, scoring each as the next word; the scores go through the softmax.' },
    { t: 'temperature', d: 'A divisor on the scores before the softmax: low, and the largest score takes nearly all the probability; high, and the probabilities flatten. When the model writes, 0 always takes the most probable next word and 1 draws words in proportion to their probabilities.' },
    { t: 'grounding', also: ['grounded'], d: 'Whether what the model writes follows from what it was given, here the findings, rather than from what is usual.' },
    { t: 'instruction tuning', also: ['instruction-tuned'], d: 'Further training on transcripts of questions and good answers, so that the model answers rather than continues.' },
    { t: 'preference tuning', d: 'Further training on pairs of answers where one was preferred, pushing the model toward the preferred kind.' },
    { t: 'system prompt', also: ['system line'], d: 'The instructions placed before the conversation, which the model reads like any other text: its persona, its rules, the tools it may call.' },
    { t: 'agent', also: ['agents', 'agent loop'], d: 'A model in a loop with a program: it can write a tool call, the program runs it and pastes the result into the transcript, and the model reads it before answering.' },
    { t: 't-SNE', d: 'A layout of high-dimensional points in two dimensions that keeps neighbours together. Clusters are real neighbourhoods; the distances between clusters mean little.' },
    { t: 'two directions', also: ['principal components', 'PCA'], d: 'The two directions along which a set of vectors varies most, used as the map’s axes. A linear projection, so the paths through training stay comparable.' },
    { t: 'perplexity', d: 'For t-SNE, the effective number of neighbours each point keeps. For a language model, e to the loss: the number of equally likely choices it is effectively picking from at each token.' },
    { t: 'cosine', also: ['cosine similarity', 'cosine distance'], d: 'A similarity between two vectors from the angle between them alone: 1 when they point the same way, 0 at right angles, whatever their lengths.' },
  ];
  const INDEX = new Map(); for (const e of TERMS) for (const f of [e.t, ...(e.also || [])]) INDEX.set(f.toLowerCase(), e);
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const RE = new RegExp('(?<![\\w\\u2019-])(' + [...INDEX.keys()].sort((a, b) => b.length - a.length).map(esc).join('|') + ')(?![\\w-])', 'gi');
  const PROSE = 'p, li, dd, .note, .lesson, .ch-explain, .explain, figcaption'; // where the terms are marked
  const SKIP = 'code, kbd, button, label, select, option, textarea, input, svg, canvas, a, h1, h2, h3, h4, .mono, .tok, .term, .legend, .no-gloss, .actions-inline, .ctl, .seg, .tip, #gloss-dialog';
  let on = true, observer = null, pending = new Set(), scheduled = false, pinned = null;
  function markElement(el) {
    if (!on || !el.isConnected) return; const text = el.textContent, prev = el.__gloss; if (prev && prev.text === text && (prev.first ? el.contains(prev.first) : true)) return; // our own spans leave the text as it was; a rewrite with the same text drops them, and the first span's absence says so
    unwrap(el);
    const seen = new Set(), nodes = []; const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); let n; while ((n = walker.nextNode())) nodes.push(n);
    for (const node of nodes) {
      const parent = node.parentElement; if (!parent || parent.closest(SKIP)) continue; const s = node.nodeValue; if (!s || s.length < 3) continue;
      const parts = []; let last = 0; RE.lastIndex = 0; let m;
      while ((m = RE.exec(s))) { const e = INDEX.get(m[1].toLowerCase()); if (!e || seen.has(e.t)) continue; if (e.unless && e.unless.test(s.slice(Math.max(0, m.index - 12), m.index))) continue; seen.add(e.t); parts.push(s.slice(last, m.index), { text: m[1], e }); last = m.index + m[1].length; }
      if (!parts.length) continue; parts.push(s.slice(last));
      const frag = document.createDocumentFragment();
      for (const p of parts) { if (typeof p === 'string') { if (p) frag.appendChild(document.createTextNode(p)); } else { const span = document.createElement('span'); span.className = 'term'; span.dataset.term = p.e.t; span.textContent = p.text; frag.appendChild(span); } }
      node.parentNode.replaceChild(frag, node);
    }
    el.__gloss = { text: el.textContent, first: el.querySelector('.term') };
  }
  function unwrap(el) { for (const span of el.querySelectorAll('.term')) span.replaceWith(document.createTextNode(span.textContent)); el.normalize(); }
  function markAll() { for (const el of document.querySelectorAll(PROSE)) markElement(el); }
  function unmarkAll() { for (const el of document.querySelectorAll(PROSE)) { unwrap(el); el.__gloss = null; } hide(); }
  function flush() { scheduled = false; const todo = [...pending]; pending.clear(); for (const el of todo) markElement(el); } // marking adds spans, so our own records come back next: markElement sees the text unchanged and skips them
  function onMutations(records) {
    if (!on) return;
    for (const r of records) {
      const target = r.target.nodeType === 3 ? r.target.parentElement : r.target; if (!target) continue;
      const host = target.closest ? target.closest(PROSE) : null; if (host) pending.add(host);
      if (r.type === 'childList') for (const node of r.addedNodes) { if (node.nodeType !== 1) continue; if (node.matches(PROSE)) pending.add(node); for (const el of node.querySelectorAll(PROSE)) pending.add(el); }
    }
    if (pending.size) flush(); // at once, not on the next frame: a note the page rewrites every frame would otherwise be unmarked every other frame
  }
  // the tooltip: one floating box, placed under the term (above it when there is no room below)
  let tip = null;
  function show(span) {
    const e = INDEX.get(span.dataset.term.toLowerCase()) || TERMS.find(x => x.t === span.dataset.term); if (!e) return;
    if (!tip) { tip = document.createElement('div'); tip.id = 'gloss-tip'; tip.setAttribute('role', 'tooltip'); document.body.appendChild(tip); }
    tip.innerHTML = `<b>${e.t}</b> ${e.d}`; tip.style.display = 'block';
    const r = span.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    let left = Math.min(Math.max(8, r.left), vw - w - 8), top = r.bottom + 6; if (top + h > vh - 8 && r.top - h - 6 > 8) top = r.top - h - 6;
    tip.style.left = `${left}px`; tip.style.top = `${top}px`;
  }
  function hide() { if (tip) tip.style.display = 'none'; }
  // the whole list, in a dialog
  function renderList() {
    const dl = document.getElementById('gloss-list'); if (!dl) return;
    dl.innerHTML = TERMS.slice().sort((a, b) => a.t.localeCompare(b.t, 'en', { sensitivity: 'base' })).map(e => `<dt>${e.t}${e.also && e.also.length ? `<span class="also">${e.also.join(', ')}</span>` : ''}</dt><dd>${e.d}</dd>`).join('');
  }
  function setOn(v) {
    on = !!v; try { localStorage.setItem('nucleus-net-gloss', on ? '1' : '0'); } catch (err) { /* ignore */ }
    const box = document.getElementById('gloss-underline'); if (box) box.checked = on;
    if (on) { markAll(); if (!observer) { observer = new MutationObserver(onMutations); observer.observe(document.body, { childList: true, characterData: true, subtree: true }); } }
    else { if (observer) { observer.disconnect(); observer = null; } pending.clear(); unmarkAll(); }
  }
  function init() {
    try { const v = localStorage.getItem('nucleus-net-gloss'); if (v === '0') on = false; } catch (err) { /* ignore */ }
    renderList();
    const dialog = document.getElementById('gloss-dialog'), open = document.getElementById('gloss-open'), box = document.getElementById('gloss-underline');
    if (open && dialog) open.addEventListener('click', () => { hide(); dialog.showModal(); });
    if (box) { box.checked = on; box.addEventListener('change', () => setOn(box.checked)); }
    document.addEventListener('mouseover', ev => { const t = ev.target.closest && ev.target.closest('.term'); if (t) show(t); });
    document.addEventListener('mouseout', ev => { const t = ev.target.closest && ev.target.closest('.term'); if (t && t !== pinned) hide(); });
    document.addEventListener('click', ev => { const t = ev.target.closest && ev.target.closest('.term'); if (t && t !== pinned) { pinned = t; show(t); } else { pinned = null; hide(); } }); // a tap pins the tooltip; a tap elsewhere lets it go
    document.addEventListener('scroll', () => { pinned = null; hide(); }, true);
    setOn(on);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
  return { TERMS, markAll, setOn, isOn: () => on, markElement };
})();
