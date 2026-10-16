/*
 * nn.js — a small neural network with hand-written backpropagation.
 *
 *   input  ->  [optional convolution: K filters f×f, ReLU, max-pool p×p]  ->  [0, 1 or 2 dense hidden layers]  ->  sigmoid output
 *
 * The output is P(positive class). Trained with mini-batch gradient descent on binary cross-entropy,
 * with optional weight decay. Two miniature foundation models reuse the same pieces, an encoder being a Net without
 * its output (the convolution, then a dense layer to a short code), both trained without labels: Contrastive makes two
 * views of the same nucleus land on the same code, AutoEncoder rebuilds the pixels from the code. AttentionMIL is
 * attention over a slide of nuclei with one label, with ContextLayer, one layer of self-attention, letting the nuclei
 * look at each other; LanguageModel stacks causal context layers over the words of a report and predicts the next.
 * Works in the browser (window.TinyNet) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TinyNet = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const ACTIVATIONS = {
    sigmoid: { label: 'Sigmoid', f: z => 1 / (1 + Math.exp(-z)), df: (z, a) => a * (1 - a), signed: false },
    tanh:    { label: 'Tanh',    f: z => Math.tanh(z),           df: (z, a) => 1 - a * a,   signed: true },
    relu:    { label: 'ReLU',    f: z => (z > 0 ? z : 0),        df: (z, a) => (z > 0 ? 1 : 0), signed: false },
    linear:  { label: 'Linear',  f: z => z,                      df: () => 1,                signed: true }, // for an autoencoder's code
  };
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const EPS = 1e-7;
  const bce = (p, y) => -(y * Math.log(p + EPS) + (1 - y) * Math.log(1 - p + EPS));

  class Net {
    /*
     * inputSize   number of inputs (6 measurements, or 1024 pixels)
     * imageSize   side of the image when the input is pixels (needed for convolution)
     * conv        null, or { K: filters, f: filter side, pool: pooling side }
     * hidden      [] | [h1] | [h1, h2] dense hidden layer sizes
     * activation  'sigmoid' | 'tanh' | 'relu' for the dense hidden layers (the convolution always uses ReLU)
     */
    constructor({ inputSize, imageSize = 0, conv = null, hidden = [], activation = 'relu', seed = 1 }) {
      this.D = inputSize; this.size = imageSize;
      this.conv = conv && conv.K > 0 ? { K: conv.K, f: conv.f || 5, pool: conv.pool || 4 } : null;
      this.hidden = (hidden || []).filter(h => h > 0);
      this.activation = activation; this.seed = seed;
      this.init();
    }
    init() {
      const rnd = mulberry32(this.seed * 7919 + 17);
      const u = s => (rnd() * 2 - 1) * s;
      let d0 = this.D;
      if (this.conv) {
        const { K, f, pool } = this.conv;
        this.co = this.size - f + 1; this.po = Math.floor(this.co / pool);
        this.Wc = new Float64Array(K * f * f); const sc = 0.05; // small, so the learned filters dominate what is drawn
        for (let i = 0; i < this.Wc.length; i++) this.Wc[i] = u(sc);
        this.bc = new Float64Array(K);
        d0 = K * this.po * this.po;
      }
      this.sizes = [d0, ...this.hidden];
      this.smallNet = !this.conv && this.D < 256;
      this.W = []; this.b = [];
      for (let l = 0; l < this.hidden.length; l++) {
        // layers fed by pixels (or pooled maps) start small, so the learned structure shows through in the weight maps
        // pixel-fed layers start very small so the learned maps show through; small measurement nets start small so
        // the drawn connections visibly grow; deeper layers of pixel nets keep the usual scale so training is not slowed
        const fanIn = this.sizes[l], fanOut = this.sizes[l + 1], s = fanIn >= 256 ? 0.02 : (this.smallNet ? 0.3 : Math.sqrt(6 / (fanIn + fanOut)));
        const W = new Float64Array(fanOut * fanIn); for (let i = 0; i < W.length; i++) W[i] = u(s);
        this.W.push(W); this.b.push(new Float64Array(fanOut));
      }
      const last = this.sizes[this.sizes.length - 1], so = last >= 256 ? 0.01 : (this.smallNet ? 0.3 : Math.sqrt(6 / (last + 1)));
      this.Wo = new Float64Array(last); for (let i = 0; i < last; i++) this.Wo[i] = u(so);
      this.bo = 0;
      this.steps = 0;
    }
    get H() { return this.hidden.length ? this.hidden[0] : 0; }           // first hidden layer width (0 = none)
    get featureCount() { return this.sizes[0]; }                          // inputs to the first dense layer

    convForward(x) {
      const { K, f, pool } = this.conv, size = this.size, co = this.co, po = this.po, Wc = this.Wc;
      const pre = new Float64Array(K * co * co), act = new Float64Array(K * co * co);
      for (let k = 0; k < K; k++) for (let oy = 0; oy < co; oy++) for (let ox = 0; ox < co; ox++) {
        let s = this.bc[k];
        if (f === 5) for (let dy = 0; dy < 5; dy++) { const row = (oy + dy) * size + ox, wrow = (k * 5 + dy) * 5; s += Wc[wrow] * x[row] + Wc[wrow + 1] * x[row + 1] + Wc[wrow + 2] * x[row + 2] + Wc[wrow + 3] * x[row + 3] + Wc[wrow + 4] * x[row + 4]; }
        else for (let dy = 0; dy < f; dy++) { const row = (oy + dy) * size + ox, wrow = (k * f + dy) * f; for (let dx = 0; dx < f; dx++) s += Wc[wrow + dx] * x[row + dx]; }
        const i = (k * co + oy) * co + ox; pre[i] = s; act[i] = s > 0 ? s : 0;
      }
      const F = K * po * po, v = new Float64Array(F), arg = new Int32Array(F);
      for (let k = 0; k < K; k++) for (let py = 0; py < po; py++) for (let px = 0; px < po; px++) {
        let best = -Infinity, bi = -1;
        for (let dy = 0; dy < pool; dy++) for (let dx = 0; dx < pool; dx++) { const i = (k * co + py * pool + dy) * co + px * pool + dx; if (act[i] > best) { best = act[i]; bi = i; } }
        const j = (k * po + py) * po + px; v[j] = best; arg[j] = bi;
      }
      return { pre, act, v, arg };
    }
    forward(x) {
      const act = ACTIVATIONS[this.activation];
      const conv = this.conv ? this.convForward(x) : null;
      const a = [conv ? conv.v : x], pre = [];
      for (let l = 0; l < this.hidden.length; l++) {
        const fanIn = this.sizes[l], fanOut = this.sizes[l + 1], W = this.W[l], inp = a[l];
        const p = new Float64Array(fanOut), o = new Float64Array(fanOut);
        for (let j = 0; j < fanOut; j++) { let s = this.b[l][j]; const off = j * fanIn; for (let i = 0; i < fanIn; i++) s += W[off + i] * inp[i]; p[j] = s; o[j] = act.f(s); }
        pre.push(p); a.push(o);
      }
      const top = a[a.length - 1];
      let z = this.bo; for (let i = 0; i < top.length; i++) z += this.Wo[i] * top[i];
      // h / pre kept for the first hidden layer for backwards compatibility with the diagrams
      return { conv, a, pre, h: a.length > 1 ? a[1] : null, z, p: sigmoid(z) };
    }
    predict(x) { return this.forward(x).p; }

    // backpropagate d(loss)/dz = dz through the output weights and the dense layers; returns d(loss)/d(a0)
    // (a0 = pooled conv features or the raw input)
    backDense(fw, dz, g) {
      const top = fw.a[fw.a.length - 1];
      const d = new Float64Array(top.length);
      if (g) g.gbo += dz;
      for (let i = 0; i < top.length; i++) { if (g) g.gWo[i] += dz * top[i]; d[i] = dz * this.Wo[i]; }
      return this.backHidden(fw, d, g);
    }
    // backpropagate d(loss)/d(top) through the dense hidden layers, top being the last hidden layer's activations (or
    // a0 itself when there is none); fills g.gW, g.gb and g.delta when g is given; returns d(loss)/d(a0)
    backHidden(fw, dTop, g) {
      const act = ACTIVATIONS[this.activation];
      let d = dTop;
      for (let l = this.hidden.length - 1; l >= 0; l--) {
        const fanIn = this.sizes[l], fanOut = this.sizes[l + 1], inp = fw.a[l], W = this.W[l];
        const dn = new Float64Array(fanIn);
        for (let j = 0; j < fanOut; j++) {
          const dj = d[j] * act.df(fw.pre[l][j], fw.a[l + 1][j]);
          if (g && g.delta) g.delta[l][j] = dj;
          if (dj === 0) continue;
          if (g) g.gb[l][j] += dj;
          const off = j * fanIn;
          if (g) for (let i = 0; i < fanIn; i++) { g.gW[l][off + i] += dj * inp[i]; dn[i] += dj * W[off + i]; }
          else for (let i = 0; i < fanIn; i++) dn[i] += dj * W[off + i];
        }
        d = dn;
      }
      return d;
    }
    // route d(loss)/d(pooled) back through pooling + ReLU to the conv pre-activations
    backPool(fw, d0) {
      const { K } = this.conv, co = this.co;
      const dconv = new Float64Array(K * co * co);
      for (let j = 0; j < d0.length; j++) { const i = fw.conv.arg[j]; if (i >= 0 && fw.conv.pre[i] > 0) dconv[i] += d0[j]; }
      return dconv;
    }

    // an empty gradient accumulator, one slot per parameter
    newGradient() {
      const g = { gW: this.W.map(W => new Float64Array(W.length)), gb: this.b.map(b => new Float64Array(b.length)), gWo: new Float64Array(this.Wo.length), gbo: 0 };
      if (this.conv) { g.gWc = new Float64Array(this.Wc.length); g.gbc = new Float64Array(this.bc.length); }
      return g;
    }
    // the convolution's gradients from d(loss)/d(pooled values), added into g.gWc and g.gbc
    convAccumulate(x, fw, d0, g) {
      const { K, f } = this.conv, size = this.size, co = this.co;
      const dconv = this.backPool(fw, d0);
      for (let kk = 0; kk < K; kk++) for (let oy = 0; oy < co; oy++) for (let ox = 0; ox < co; ox++) {
        const gg = dconv[(kk * co + oy) * co + ox]; if (gg === 0) continue;
        g.gbc[kk] += gg;
        const gWc = g.gWc;
        if (f === 5) for (let dy = 0; dy < 5; dy++) { const row = (oy + dy) * size + ox, wrow = (kk * 5 + dy) * 5; gWc[wrow] += gg * x[row]; gWc[wrow + 1] += gg * x[row + 1]; gWc[wrow + 2] += gg * x[row + 2]; gWc[wrow + 3] += gg * x[row + 3]; gWc[wrow + 4] += gg * x[row + 4]; }
        else for (let dy = 0; dy < f; dy++) { const row = (oy + dy) * size + ox, wrow = (kk * f + dy) * f; for (let dx = 0; dx < f; dx++) gWc[wrow + dx] += gg * x[row + dx]; }
      }
    }
    // forward + backward for one case, adding its gradient into g; returns the case's loss
    accumulate(x, y, g, fw) {
      fw = fw || this.forward(x);
      const loss = bce(fw.p, y);
      const d0 = this.backDense(fw, fw.p - y, g);
      if (this.conv) this.convAccumulate(x, fw, d0, g);
      return loss;
    }
    // gradient-descent update from an accumulated gradient over n cases (l2 = weight decay)
    applyGradient(g, n, lr, l2 = 0) {
      const s = lr / n, decay = 1 - lr * l2;
      if (this.conv) { for (let i = 0; i < this.Wc.length; i++) this.Wc[i] = this.Wc[i] * decay - s * g.gWc[i]; for (let k = 0; k < this.bc.length; k++) this.bc[k] -= s * g.gbc[k]; }
      for (let l = 0; l < this.W.length; l++) { const W = this.W[l]; for (let i = 0; i < W.length; i++) W[i] = W[i] * decay - s * g.gW[l][i]; for (let j = 0; j < this.b[l].length; j++) this.b[l][j] -= s * g.gb[l][j]; }
      for (let i = 0; i < this.Wo.length; i++) this.Wo[i] = this.Wo[i] * decay - s * g.gWo[i];
      this.bo -= s * g.gbo;
      this.steps++;
    }
    // one gradient-descent step on a mini-batch; returns the mean loss before the update
    trainBatch(xs, ys, lr, l2 = 0) {
      const n = xs.length, g = this.newGradient();
      let loss = 0;
      for (let k = 0; k < n; k++) loss += this.accumulate(xs[k], ys[k], g);
      this.applyGradient(g, n, lr, l2);
      return loss / n;
    }
    // one case's lesson, not yet applied: its forward pass, error (call − truth), the blame each hidden unit receives
    // (g.delta[l][j] = d loss / d pre-activation) and the gradient of every weight. Apply with applyGradient(g, 1, lr, l2).
    lesson(x, y) {
      const g = this.newGradient(); g.delta = this.hidden.map(h => new Float64Array(h));
      const fw = this.forward(x);
      const loss = this.accumulate(x, y, g, fw);
      if (this.conv) { const d0 = this.backDense(fw, fw.p - y, null); g.dPooled = d0; g.dConv = this.backPool(fw, d0); } // for the walk-through: the blame at the pooled cells and where it lands on the feature maps
      return { fw, error: fw.p - y, loss, g };
    }

    // d(score z)/d(input): how much each input nudges the score toward the positive class
    inputGradient(x, fw) {
      fw = fw || this.forward(x);
      return this.inputGradientFrom(fw, this.backDense(fw, 1, null));
    }
    // the same from d(anything)/d(a0), a0 being the pooled convolution features or the raw input: back through the
    // pooling and the filters to the pixels
    inputGradientFrom(fw, d0) {
      if (!this.conv) return d0;
      const { K, f } = this.conv, size = this.size, co = this.co;
      const dconv = this.backPool(fw, d0);
      const gx = new Float64Array(this.D);
      for (let k = 0; k < K; k++) for (let oy = 0; oy < co; oy++) for (let ox = 0; ox < co; ox++) {
        const gg = dconv[(k * co + oy) * co + ox]; if (gg === 0) continue;
        for (let dy = 0; dy < f; dy++) { const row = (oy + dy) * size + ox, wrow = (k * f + dy) * f; for (let dx = 0; dx < f; dx++) gx[row + dx] += gg * this.Wc[wrow + dx]; }
      }
      return gx;
    }

    // withActs: also return each sample's first-hidden-layer activations (null when there is no hidden layer)
    evaluate(xs, ys, threshold = 0.5, withActs = false) {
      let loss = 0, correct = 0;
      const probs = new Float64Array(xs.length);
      const acts = withActs && this.hidden.length ? new Array(xs.length) : null;
      for (let k = 0; k < xs.length; k++) {
        const fw = this.forward(xs[k]);
        const p = fw.p;
        probs[k] = p; loss += bce(p, ys[k]);
        if (acts) acts[k] = fw.a[1];
        if ((p >= threshold ? 1 : 0) === ys[k]) correct++;
      }
      return { loss: loss / xs.length, accuracy: correct / xs.length, probs, acts };
    }
    parameterCount() {
      let n = this.Wo.length + 1;
      for (let l = 0; l < this.W.length; l++) n += this.W[l].length + this.b[l].length;
      if (this.conv) n += this.Wc.length + this.bc.length;
      return n;
    }
    describe() {
      const parts = [];
      if (this.conv) parts.push(`${this.conv.K} filters ${this.conv.f}×${this.conv.f} + pool ${this.conv.pool}×${this.conv.pool}`);
      if (this.hidden.length) parts.push(`${this.hidden.join(' + ')} ${ACTIVATIONS[this.activation].label} units`);
      return parts.length ? parts.join(' → ') : 'single layer';
    }
  }

  // A convolutional autoencoder, the miniature foundation model. The encoder is a Net without its output: the
  // convolution, then a dense layer to a short code (tanh, so every code value lies in −1..1). The decoder rebuilds the
  // pixels from the code as Σ code_j × basis image_j + a bias image. Trained on the squared error of the rebuild, so no
  // label is ever involved; the code is the embedding the downstream tasks can use.
  class AutoEncoder {
    constructor({ imageSize, conv = { K: 4, f: 5, pool: 4 }, code = 8, seed = 1 }) {
      this.size = imageSize; this.D = imageSize * imageSize; this.code = code; this.seed = seed;
      this.enc = new Net({ inputSize: this.D, imageSize, conv, hidden: [code], activation: 'tanh', seed });
      const rnd = mulberry32(seed * 104729 + 3), u = s => (rnd() * 2 - 1) * s;
      this.Wd = new Float64Array(this.D * code); for (let i = 0; i < this.Wd.length; i++) this.Wd[i] = u(0.05);
      this.bd = new Float64Array(this.D);
      this.steps = 0;
    }
    get conv() { return this.enc.conv; }
    encode(x) { return this.enc.forward(x).a[1]; }
    // basis image j: the decoder's weights from code unit j, as an image
    basis(j) { const b = new Float64Array(this.D); for (let i = 0; i < this.D; i++) b[i] = this.Wd[i * this.code + j]; return b; }
    forward(x) {
      const fw = this.enc.forward(x), code = fw.a[1], recon = new Float64Array(this.D), C = this.code;
      let loss = 0;
      for (let i = 0; i < this.D; i++) { let s = this.bd[i]; const off = i * C; for (let j = 0; j < C; j++) s += this.Wd[off + j] * code[j]; recon[i] = s; const e = s - x[i]; loss += e * e; }
      return { enc: fw, code, recon, loss: loss / this.D };
    }
    newGradient() { const g = this.enc.newGradient(); g.delta = [new Float64Array(this.code)]; g.gWd = new Float64Array(this.Wd.length); g.gbd = new Float64Array(this.D); return g; }
    // forward + backward for one nucleus, adding its gradient into g; returns the squared error of the rebuild
    accumulate(x, g, fw) {
      fw = fw || this.forward(x);
      const { code, recon } = fw, D = this.D, C = this.code, dCode = new Float64Array(C);
      for (let i = 0; i < D; i++) {
        const d = 2 * (recon[i] - x[i]) / D, off = i * C;
        g.gbd[i] += d;
        for (let j = 0; j < C; j++) { g.gWd[off + j] += d * code[j]; dCode[j] += d * this.Wd[off + j]; }
      }
      const d0 = this.enc.backHidden(fw.enc, dCode, g);
      if (this.enc.conv) this.enc.convAccumulate(x, fw.enc, d0, g);
      return fw.loss;
    }
    applyGradient(g, n, lr, l2 = 0) {
      this.enc.applyGradient(g, n, lr, l2); // the encoder's unused output weights receive a zero gradient
      const s = lr / n, decay = 1 - lr * l2;
      for (let i = 0; i < this.Wd.length; i++) this.Wd[i] = this.Wd[i] * decay - s * g.gWd[i];
      for (let i = 0; i < this.D; i++) this.bd[i] -= s * g.gbd[i];
      this.steps++;
    }
    trainBatch(xs, lr, l2 = 0) {
      const g = this.newGradient(); let loss = 0;
      for (const x of xs) loss += this.accumulate(x, g);
      this.applyGradient(g, xs.length, lr, l2);
      return loss / xs.length;
    }
    // mean squared error of the rebuild over a set, and every nucleus's code
    evaluate(xs) { let loss = 0; const codes = new Array(xs.length); for (let k = 0; k < xs.length; k++) { const fw = this.forward(xs[k]); loss += fw.loss; codes[k] = fw.code; } return { loss: loss / xs.length, codes }; }
    parameterCount() { return this.enc.parameterCount() - this.enc.Wo.length - 1 + this.Wd.length + this.bd.length; }
    describe() { const c = this.enc.conv; return `${c.K} filters ${c.f}×${c.f} + pool ${c.pool}×${c.pool} → code of ${this.code} (tanh) → ${this.D.toLocaleString()} pixels rebuilt`; }
  }

  // Contrastive pretraining, the miniature foundation model of the page. The encoder is a Net without its output: the
  // convolution, then a dense linear layer to a short code. It is trained so that two views of the same nucleus (a flip
  // or rotation, or the other lab's scan of it) land close together in the code, on the unit sphere, while every other
  // nucleus in the batch lands far away: instance discrimination with the normalised-temperature cross-entropy loss.
  // No label is ever involved; the code is the embedding the downstream tasks can use.
  class Contrastive {
    constructor({ imageSize, conv = { K: 4, f: 5, pool: 4 }, code = 8, tau = 0.2, seed = 1 }) {
      this.size = imageSize; this.D = imageSize * imageSize; this.code = code; this.tau = tau; this.seed = seed;
      this.enc = new Net({ inputSize: this.D, imageSize, conv, hidden: [code], activation: 'linear', seed });
      this.steps = 0;
    }
    get conv() { return this.enc.conv; }
    encode(x) { return this.enc.forward(x).a[1]; }
    static unit(z) { let n = 0; for (const q of z) n += q * q; n = Math.sqrt(n) + 1e-8; const u = new Float64Array(z.length); for (let d = 0; d < z.length; d++) u[d] = z[d] / n; return { u, n }; }
    // cosine similarity of two codes
    static cosine(a, b) { const ua = Contrastive.unit(a).u, ub = Contrastive.unit(b).u; let s = 0; for (let d = 0; d < ua.length; d++) s += ua[d] * ub[d]; return s; }
    // A batch of pairs [[x1, x2], …] (view 2k and 2k+1 belong to nucleus k). Returns the loss, how often each view's
    // nearest other view is its own pair, the cosine similarity matrix and the codes; when lr is given, takes one
    // gradient step. Loss = mean over views of −log( e^(s_pair/τ) / Σ_(other views) e^(s/τ) ).
    step(pairs, lr, l2 = 0) {
      const N = pairs.length * 2, C = this.code, tau = this.tau, xs = [], fws = [], u = [], norm = [];
      for (const [x1, x2] of pairs) for (const x of [x1, x2]) { const fw = this.enc.forward(x); const { u: uu, n } = Contrastive.unit(fw.a[1]); xs.push(x); fws.push(fw); u.push(uu); norm.push(n); }
      const sim = Array.from({ length: N }, () => new Float64Array(N));
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { let s = 0; for (let d = 0; d < C; d++) s += u[i][d] * u[j][d]; sim[i][j] = sim[j][i] = s; }
      const g = lr != null ? this.enc.newGradient() : null;
      const du = g ? Array.from({ length: N }, () => new Float64Array(C)) : null;
      let loss = 0, hits = 0;
      for (let i = 0; i < N; i++) {
        const pos = i ^ 1; let mx = -Infinity, best = -1, bestS = -Infinity;
        for (let j = 0; j < N; j++) if (j !== i) { mx = Math.max(mx, sim[i][j] / tau); if (sim[i][j] > bestS) { bestS = sim[i][j]; best = j; } }
        if (best === pos) hits++;
        let Z = 0; const p = new Float64Array(N);
        for (let j = 0; j < N; j++) if (j !== i) { p[j] = Math.exp(sim[i][j] / tau - mx); Z += p[j]; }
        loss += -Math.log(p[pos] / Z) / N;
        if (g) for (let j = 0; j < N; j++) if (j !== i) { const coef = (p[j] / Z - (j === pos ? 1 : 0)) / tau / N; for (let d = 0; d < C; d++) { du[i][d] += coef * u[j][d]; du[j][d] += coef * u[i][d]; } }
      }
      if (g) {
        for (let i = 0; i < N; i++) { // through the normalisation: dz = (du − u (u·du)) / |z|
          let dot = 0; for (let d = 0; d < C; d++) dot += u[i][d] * du[i][d];
          const dz = new Float64Array(C); for (let d = 0; d < C; d++) dz[d] = (du[i][d] - u[i][d] * dot) / norm[i];
          const d0 = this.enc.backHidden(fws[i], dz, g);
          if (this.enc.conv) this.enc.convAccumulate(xs[i], fws[i], d0, g);
        }
        this.enc.applyGradient(g, 1, lr, l2); this.steps++;
      }
      return { loss, pairAcc: hits / N, sim, codes: fws.map(fw => fw.a[1]), units: u };
    }
    evaluate(pairs) { return this.step(pairs, null); }
    parameterCount() { return this.enc.parameterCount() - this.enc.Wo.length - 1; }
    describe() { const c = this.enc.conv; return `${c.K} filters ${c.f}×${c.f} + pool ${c.pool}×${c.pool} → code of ${this.code}`; }
    // d(anything)/d(pixels) given d(anything)/d(code): the evidence of a classifier on the code, carried through the encoder
    inputGradient(x, dCode) { const fw = this.enc.forward(x); return this.enc.inputGradientFrom(fw, this.enc.backHidden(fw, dCode, null)); }
    // the encoder's weights and its input standardiser as plain arrays, so a pretrained backbone can ship with the page
    toJSON(std, meta) {
      const r = v => Array.from(v, q => +q.toPrecision(5));
      return Object.assign({}, meta || {}, { size: this.size, conv: this.enc.conv, code: this.code, tau: this.tau, steps: this.steps, std: { mean: r(std.mean), scale: +std.scale[0].toPrecision(6) }, Wc: r(this.enc.Wc), bc: r(this.enc.bc), W: r(this.enc.W[0]), b: r(this.enc.b[0]) });
    }
    static fromJSON(o) {
      const cl = new Contrastive({ imageSize: o.size, conv: o.conv, code: o.code, tau: o.tau, seed: 1 });
      cl.enc.Wc.set(o.Wc); cl.enc.bc.set(o.bc); cl.enc.W[0].set(o.W); cl.enc.b[0].set(o.b); cl.steps = o.steps || 0;
      cl.std = standardizerFrom(o.std.mean, o.std.scale); cl.meta = o;
      return cl;
    }
  }

  // ---------------------------------------------------------------------------- context: the nuclei look at each other
  // One layer of self-attention over the slide, the mechanism of a transformer. Every nucleus turns its vector into a
  // query (what it is looking for), a key (what it offers) and a value (what it passes on). Query · key over all the
  // nuclei, through a softmax, says how much each one listens to each other; the weighted values are projected and
  // added to its own vector (a residual), then a small feed-forward, added again. With a distance bias, the match is
  // discounted by a learned amount per unit of distance, so the layer can prefer near neighbours.
  const softplus = x => (x > 30 ? x : Math.log1p(Math.exp(x)));
  class ContextLayer {
    constructor({ inputSize, dk = 8, ffn = 8, heads = 1, distanceBias = false, excludeSelf = false, causal = false, costInit = 0, relative = false, seed = 1 }) {
      this.D = inputSize; this.dk = dk; this.H = heads; this.F = ffn; this.distanceBias = distanceBias; this.excludeSelf = excludeSelf; this.causal = causal; this.relative = relative; this.seed = seed; // causal: a token reads itself and the tokens before it, never the ones after (a language model)
      const rnd = mulberry32(seed * 104729 + 3), u = s => (rnd() * 2 - 1) * s, mk = (n, s) => Float64Array.from({ length: n }, () => u(s)), D = inputSize, HD = heads * dk, HM = heads * (dk + (relative ? 2 : 0));
      this.Wq = mk(HD * D, 0.3); this.bq = new Float64Array(HD); this.Wk = mk(HD * D, 0.3); this.bk = new Float64Array(HD);
      this.Wv = mk(HD * D, 0.3); this.bv = new Float64Array(HD); this.Wo = mk(D * HM, 0.1); this.bo = new Float64Array(D); // small: the layer starts close to doing nothing
      this.W1 = mk(ffn * D, 0.3); this.b1 = new Float64Array(ffn); this.W2 = mk(D * ffn, 0.1); this.b2 = new Float64Array(D);
      this.beta = Float64Array.from({ length: heads }, (_, h) => { const c = Array.isArray(costInit) ? costInit[h] : costInit; return c > 0 ? Math.log(Math.expm1(c)) : -8; }); // softplus(beta) is each head's cost per unit of distance, when distanceBias is on (one value, or one per head; 0 starts the cost at nothing, softplus(−8), and lets it grow)
    }
    // X: the slide's tokens (rows of D numbers); dist: n × n distances between them (only with the distance bias).
    // With several heads, every head has its own query, key and value maps and its own distance cost, and their
    // messages are concatenated before the output projection. S, A and cost are the first head's (what a single-head
    // layer shows); Sh, Ah and costs hold every head's. With relative on, every head's message also carries where the
    // nuclei it listened to lie, relative to the listener (the weighted mean offset, from xy: n × 2 positions).
    forward(X, dist, xy) {
      const n = X.length, D = this.D, dk = this.dk, H = this.H, HD = H * dk, F = this.F, sc = 1 / Math.sqrt(dk), mask = this.excludeSelf && n > 1, causal = this.causal; // a nucleus reads the others, not itself (its own vector is the residual); a token reads itself and what came before
      const rel = this.relative && !!xy, mw = dk + (rel ? 2 : 0), HM = H * mw;
      const costs = Array.from(this.beta, b => (this.distanceBias ? softplus(b) : 0));
      const lin = (W, bias, out) => X.map(x => { const o = new Float64Array(out); for (let j = 0; j < out; j++) { let s = bias[j]; const off = j * D; for (let i = 0; i < D; i++) s += W[off + i] * x[i]; o[j] = s; } return o; });
      const Q = lin(this.Wq, this.bq, HD), K = lin(this.Wk, this.bk, HD), V = lin(this.Wv, this.bv, HD), Sh = [], Ah = [];
      for (let h = 0; h < H; h++) {
        const S = [], A = [], b = costs[h], o = h * dk;
        for (let i = 0; i < n; i++) {
          const si = new Float64Array(n), ai = new Float64Array(n), last = causal ? i : n - 1; let mx = -Infinity; // causal: the tokens after i stay at zero
          for (let j = 0; j <= last; j++) { if (mask && j === i) { si[j] = -Infinity; continue; } let v = 0; for (let d = 0; d < dk; d++) v += Q[i][o + d] * K[j][o + d]; si[j] = v * sc - (b ? b * dist[i][j] : 0); mx = Math.max(mx, si[j]); }
          for (let j = last + 1; j < n; j++) si[j] = -Infinity;
          let Z = 0; for (let j = 0; j <= last; j++) { ai[j] = Math.exp(si[j] - mx); Z += ai[j]; }
          for (let j = 0; j <= last; j++) ai[j] /= Z;
          S.push(si); A.push(ai);
        }
        Sh.push(S); Ah.push(A);
      }
      const C = X.map((_, i) => { const c = new Float64Array(HM); for (let h = 0; h < H; h++) { const ai = Ah[h][i], o = h * dk, om = h * mw; for (let j = 0; j < n; j++) if (ai[j]) { for (let d = 0; d < dk; d++) c[om + d] += ai[j] * V[j][o + d]; if (rel) { c[om + dk] += ai[j] * (xy[j][0] - xy[i][0]); c[om + dk + 1] += ai[j] * (xy[j][1] - xy[i][1]); } } } return c; });
      const Xp = X.map((x, i) => { const o = Float64Array.from(x); for (let d = 0; d < D; d++) { let s = this.bo[d]; const off = d * HM; for (let e = 0; e < HM; e++) s += this.Wo[off + e] * C[i][e]; o[d] += s; } return o; });
      const U = Xp.map(xp => { const u = new Float64Array(F); for (let f = 0; f < F; f++) { let s = this.b1[f]; const off = f * D; for (let d = 0; d < D; d++) s += this.W1[off + d] * xp[d]; u[f] = Math.tanh(s); } return u; });
      const Y = Xp.map((xp, i) => { const y = Float64Array.from(xp); for (let d = 0; d < D; d++) { let s = this.b2[d]; const off = d * F; for (let f = 0; f < F; f++) s += this.W2[off + f] * U[i][f]; y[d] += s; } return y; });
      return { X, dist, xy: rel ? xy : null, Q, K, V, S: Sh[0], A: Ah[0], cost: costs[0], Sh, Ah, costs, C, Xp, U, Y };
    }
    // One forward pass taken apart, for showing, for head h: match[i][j] = query_i · key_j / √dk, cost[i][j] = the
    // learned cost × the distance, and the shares the softmax would give from the match alone or from the distance
    // alone (each row over the other nuclei, like the real attention); both = the shares actually used. Nothing here
    // changes the layer.
    explain(fw, h = 0) {
      const { Q, K, dist } = fw, cost = fw.costs[h], n = Q.length, dk = this.dk, o = h * dk, sc = 1 / Math.sqrt(dk), mask = this.excludeSelf && n > 1, hidden = (i, j) => (mask && j === i) || (this.causal && j > i);
      const soft = row => { let mx = -Infinity; for (const v of row) mx = Math.max(mx, v); const e = Float64Array.from(row, v => Math.exp(v - mx)); let Z = 0; for (const v of e) Z += v; for (let j = 0; j < n; j++) e[j] /= Z; return e; };
      const match = [], costs = [], matchOnly = [], distOnly = [];
      for (let i = 0; i < n; i++) {
        const m = new Float64Array(n), ct = new Float64Array(n);
        for (let j = 0; j < n; j++) { let v = 0; for (let d = 0; d < dk; d++) v += Q[i][o + d] * K[j][o + d]; m[j] = v * sc; ct[j] = cost && dist ? cost * dist[i][j] : 0; }
        match.push(m); costs.push(ct);
        matchOnly.push(soft(Float64Array.from(m, (v, j) => (hidden(i, j) ? -Infinity : v))));
        distOnly.push(soft(Float64Array.from(ct, (v, j) => (hidden(i, j) ? -Infinity : -v))));
      }
      return { match, cost: costs, matchOnly, distOnly, both: fw.Ah[h] };
    }
    newGradient() { const z = a => new Float64Array(a.length); return { Wq: z(this.Wq), bq: z(this.bq), Wk: z(this.Wk), bk: z(this.bk), Wv: z(this.Wv), bv: z(this.bv), Wo: z(this.Wo), bo: z(this.bo), W1: z(this.W1), b1: z(this.b1), W2: z(this.W2), b2: z(this.b2), beta: new Float64Array(this.H) }; }
    // d(loss)/d(Y) in, the layer's gradients accumulated into g, d(loss)/d(X) out
    backward(fw, dY, g) {
      const { X, dist, xy, Q, K, V, Ah, C, Xp, U } = fw, n = X.length, D = this.D, dk = this.dk, H = this.H, HD = H * dk, F = this.F, sc = 1 / Math.sqrt(dk), rel = !!xy, mw = dk + (rel ? 2 : 0), HM = H * mw, causal = this.causal;
      const dXp = dY.map(v => Float64Array.from(v));
      for (let i = 0; i < n; i++) { // the feed-forward, then its residual
        const dU = new Float64Array(F);
        for (let d = 0; d < D; d++) { const gd = dY[i][d]; if (!gd) continue; g.b2[d] += gd; const off = d * F; for (let f = 0; f < F; f++) { g.W2[off + f] += gd * U[i][f]; dU[f] += gd * this.W2[off + f]; } }
        for (let f = 0; f < F; f++) { const dp = dU[f] * (1 - U[i][f] * U[i][f]); if (!dp) continue; g.b1[f] += dp; const off = f * D; for (let d = 0; d < D; d++) { g.W1[off + d] += dp * Xp[i][d]; dXp[i][d] += dp * this.W1[off + d]; } }
      }
      const dX = dXp.map(v => Float64Array.from(v)), dC = [], dQ = X.map(() => new Float64Array(HD)), dK = X.map(() => new Float64Array(HD)), dV = X.map(() => new Float64Array(HD));
      for (let i = 0; i < n; i++) { // the output projection
        const dc = new Float64Array(HM);
        for (let d = 0; d < D; d++) { const gd = dXp[i][d]; if (!gd) continue; g.bo[d] += gd; const off = d * HM; for (let e = 0; e < HM; e++) { g.Wo[off + e] += gd * C[i][e]; dc[e] += gd * this.Wo[off + e]; } }
        dC.push(dc);
      }
      const dbeta = new Float64Array(H);
      for (let h = 0; h < H; h++) { // every head: the weighted values and the softmax over the slide
        const A = Ah[h], o = h * dk, om = h * mw;
        for (let i = 0; i < n; i++) {
          const dA = new Float64Array(n), last = causal ? i : n - 1; let dot = 0; // causal: the tokens after i carry no attention
          for (let j = 0; j <= last; j++) { let v = 0; for (let e = 0; e < dk; e++) { v += dC[i][om + e] * V[j][o + e]; dV[j][o + e] += A[i][j] * dC[i][om + e]; } if (rel) v += dC[i][om + dk] * (xy[j][0] - xy[i][0]) + dC[i][om + dk + 1] * (xy[j][1] - xy[i][1]); dA[j] = v; dot += A[i][j] * v; }
          for (let j = 0; j <= last; j++) { const dS = A[i][j] * (dA[j] - dot); if (!dS) continue; for (let e = 0; e < dk; e++) { dQ[i][o + e] += dS * sc * K[j][o + e]; dK[j][o + e] += dS * sc * Q[i][o + e]; } if (this.distanceBias) dbeta[h] -= dS * dist[i][j]; }
        }
      }
      if (this.distanceBias) for (let h = 0; h < H; h++) g.beta[h] += dbeta[h] * sigmoid(this.beta[h]); // through the softplus
      const back = (dOut, W, gW, gb) => { for (let i = 0; i < n; i++) for (let j = 0; j < HD; j++) { const gd = dOut[i][j]; if (!gd) continue; gb[j] += gd; const off = j * D; for (let d = 0; d < D; d++) { gW[off + d] += gd * X[i][d]; dX[i][d] += gd * W[off + d]; } } };
      back(dQ, this.Wq, g.Wq, g.bq); back(dK, this.Wk, g.Wk, g.bk); back(dV, this.Wv, g.Wv, g.bv);
      return dX;
    }
    // one token at position i, with the layer's cache of the keys and values of the tokens before it (a language model
    // writing): its output vector and each head's attention over the tokens so far; its key and value join the cache
    forwardOne(x, cache, i) {
      const D = this.D, dk = this.dk, H = this.H, HD = H * dk, F = this.F, sc = 1 / Math.sqrt(dk), mask = this.excludeSelf;
      const costs = Array.from(this.beta, b => (this.distanceBias ? softplus(b) : 0));
      const lin = (W, bias, out) => { const o = new Float64Array(out); for (let j = 0; j < out; j++) { let s = bias[j]; const off = j * D; for (let d = 0; d < D; d++) s += W[off + d] * x[d]; o[j] = s; } return o; };
      const q = lin(this.Wq, this.bq, HD), k = lin(this.Wk, this.bk, HD), v = lin(this.Wv, this.bv, HD);
      cache.K.push(k); cache.V.push(v);
      const n = cache.K.length, A = [], c = new Float64Array(HD);
      for (let h = 0; h < H; h++) {
        const o = h * dk, b = costs[h], si = new Float64Array(n); let mx = -Infinity;
        for (let j = 0; j < n; j++) { if (mask && j === i) { si[j] = -Infinity; continue; } let sv = 0; const kj = cache.K[j]; for (let d = 0; d < dk; d++) sv += q[o + d] * kj[o + d]; si[j] = sv * sc - (b ? b * (i - j) : 0); mx = Math.max(mx, si[j]); }
        let Z = 0; for (let j = 0; j < n; j++) { si[j] = Math.exp(si[j] - mx); Z += si[j]; }
        for (let j = 0; j < n; j++) { si[j] /= Z; if (si[j]) { const vj = cache.V[j]; for (let d = 0; d < dk; d++) c[o + d] += si[j] * vj[o + d]; } }
        A.push(si);
      }
      const xp = Float64Array.from(x); for (let d = 0; d < D; d++) { let sv = this.bo[d]; const off = d * HD; for (let e = 0; e < HD; e++) sv += this.Wo[off + e] * c[e]; xp[d] += sv; }
      const u = new Float64Array(F); for (let f = 0; f < F; f++) { let sv = this.b1[f]; const off = f * D; for (let d = 0; d < D; d++) sv += this.W1[off + d] * xp[d]; u[f] = Math.tanh(sv); }
      const y = Float64Array.from(xp); for (let d = 0; d < D; d++) { let sv = this.b2[d]; const off = d * F; for (let f = 0; f < F; f++) sv += this.W2[off + f] * u[f]; y[d] += sv; }
      return { y, A };
    }
    applyGradient(g, k, lr, l2 = 0) {
      const step = (p, gp, decay) => { for (let i = 0; i < p.length; i++) p[i] -= lr * (gp[i] / k + (decay ? l2 * p[i] : 0)); };
      step(this.Wq, g.Wq, 1); step(this.bq, g.bq); step(this.Wk, g.Wk, 1); step(this.bk, g.bk); step(this.Wv, g.Wv, 1); step(this.bv, g.bv);
      step(this.Wo, g.Wo, 1); step(this.bo, g.bo); step(this.W1, g.W1, 1); step(this.b1, g.b1); step(this.W2, g.W2, 1); step(this.b2, g.b2);
      if (this.distanceBias) for (let h = 0; h < this.H; h++) this.beta[h] -= lr * g.beta[h] / k;
    }
    parameterCount() { const HD = this.H * this.dk, HM = this.H * (this.dk + (this.relative ? 2 : 0)); return 3 * (HD * this.D + HD) + this.D * HM + this.D + this.F * this.D + this.F + this.D * this.F + this.D + (this.distanceBias ? this.H : 0); }
    describe() { return `self-attention over the ${this.causal ? 'tokens so far' : `${this.excludeSelf ? 'other ' : ''}nuclei`} (${this.H > 1 ? `${this.H} heads, each ` : ''}query · key of ${this.dk}${this.distanceBias ? ' − a learned distance cost' : ''}, values${this.relative ? ' and where they lie, relative,' : ''} added back) + ${this.F} tanh feed-forward`; }
  }

  // ---------------------------------------------------------------------------- a small language model
  // Tokens (ids into a vocabulary) → embeddings (a learned vector per word, plus a learned vector per position when
  // positions > 0) → L causal context layers (every token reads itself and the tokens before it; each head has a
  // learned cost per token of distance back, so the layers know how far back a token is without position vectors)
  // → a linear layer to a score per word → softmax → P(next token). Trained to predict every next token of a document
  // (cross-entropy, each position weighted, so that the loss can be kept to the report and off a prompt), with weight
  // decay on the matrices, a clip on the gradient norm, and plain gradient descent or Adam.
  class LanguageModel {
    constructor({ vocabSize, dim = 32, layers = 2, heads = 2, dk = 8, ffn = 32, costInit = 0.1, positions = 0, clip = 1, optimizer = 'adam', seed = 1, codeDim = 0 }) {
      this.V = vocabSize; this.D = dim; this.H = heads; this.dk = dk; this.F = ffn; this.P = positions; this.clip = clip; this.optimizer = optimizer; this.seed = seed; this.C = codeDim;
      const rnd = mulberry32(seed * 7919 + 5), u = s => (rnd() * 2 - 1) * s, mk = (n, s) => Float64Array.from({ length: n }, () => u(s));
      this.E = mk(vocabSize * dim, 0.5); this.Pos = positions ? mk(positions * dim, 0.2) : null;
      const costs = Array.isArray(costInit) ? costInit : Array.from({ length: heads }, (_, h) => costInit / Math.pow(2, h)); // like ALiBi: a slope per head, halving from head to head, or one per head given
      this.costs = costs; this.layers = Array.from({ length: layers }, (_, l) => new ContextLayer({ inputSize: dim, dk, ffn, heads, distanceBias: costs.some(c => c > 0), excludeSelf: false, causal: true, costInit: costs, seed: seed + 2 + l })); // no cost at all when every head starts at 0: the layers then know nothing of distance
      this.Wout = mk(vocabSize * dim, 0.3); this.bout = new Float64Array(vocabSize);
      this.Pin = codeDim ? mk(dim * codeDim, 0.3) : null; this.bin = codeDim ? new Float64Array(dim) : null; // an image's code, projected into the embedding space, can stand as the first token (from pixels to words)
      this.steps = 0; this.adam = null; this._dist = {};
    }
    distances(n) { let d = this._dist[n]; if (!d) { d = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => Math.abs(i - j))); this._dist[n] = d; } return d; } // tokens apart
    embed(tokens, code) { const D = this.D; return tokens.map((t, i) => { const x = new Float64Array(D), po = this.Pos && i < this.P ? i * D : -1; if (i === 0 && code && this.Pin) { const C = this.C; for (let d = 0; d < D; d++) { let s = this.bin[d]; const o = d * C; for (let c = 0; c < C; c++) s += this.Pin[o + c] * code[c]; x[d] = s + (po >= 0 ? this.Pos[po + d] : 0); } return x; } const off = t * D; for (let d = 0; d < D; d++) x[d] = this.E[off + d] + (po >= 0 ? this.Pos[po + d] : 0); return x; }); } // the first token can be an image: its code projected, in place of a word's row
    // tokens: ids. Returns the embeddings (X), every layer's forward (ctxs), the final vectors (T) and, per position,
    // the scores and the probabilities of the next token.
    forward(tokens, code) {
      const D = this.D, V = this.V, X = this.embed(tokens, code), dist = this.distances(tokens.length), ctxs = []; let T = X;
      for (const layer of this.layers) { const c = layer.forward(T, dist); ctxs.push(c); T = c.Y; }
      const logits = T.map(t => { const z = new Float64Array(V); for (let v = 0; v < V; v++) { let s = this.bout[v]; const off = v * D; for (let d = 0; d < D; d++) s += this.Wout[off + d] * t[d]; z[v] = s; } return z; });
      const probs = logits.map(z => { let mx = -Infinity; for (const v of z) mx = Math.max(mx, v); const p = Float64Array.from(z, v => Math.exp(v - mx)); let Z = 0; for (const v of p) Z += v; for (let v = 0; v < V; v++) p[v] /= Z; return p; });
      return { tokens, code: code || null, X, dist, ctxs, T, logits, probs };
    }
    // the loss of a document: the mean cross-entropy of its next tokens; weights[i] weighs the prediction made at
    // position i (of token i + 1), default 1; also every position's own loss
    loss(fw, weights) { const n = fw.tokens.length, per = new Float64Array(Math.max(0, n - 1)); let s = 0, w = 0; for (let i = 0; i + 1 < n; i++) { per[i] = -Math.log(fw.probs[i][fw.tokens[i + 1]] + EPS); const wi = weights ? weights[i] : 1; if (wi) { s += wi * per[i]; w += wi; } } return { loss: w ? s / w : 0, per }; }
    newGradient() { const z = a => new Float64Array(a.length); return { E: z(this.E), Pos: this.Pos ? z(this.Pos) : null, Wout: z(this.Wout), bout: z(this.bout), Pin: this.Pin ? z(this.Pin) : null, bin: this.bin ? z(this.bin) : null, layers: this.layers.map(c => c.newGradient()) }; }
    // one step on a batch of documents [tokens] or [{ tokens, weights }]; returns their mean loss before the step
    trainBatch(docs, lr, l2 = 0) {
      const g = this.newGradient(), D = this.D, V = this.V; let loss = 0;
      for (const doc of docs) {
        const tokens = doc.tokens || doc, weights = doc.weights || null, n = tokens.length; if (n < 2) continue;
        let wsum = 0; for (let i = 0; i + 1 < n; i++) wsum += weights ? weights[i] : 1; if (!wsum) continue;
        const fw = this.forward(tokens, doc.code || null); loss += this.loss(fw, weights).loss;
        const dT = fw.T.map(() => new Float64Array(D));
        for (let i = 0; i + 1 < n; i++) { // the softmax over the vocabulary, into the output layer and the final vectors
          const wi = (weights ? weights[i] : 1) / wsum; if (!wi) continue;
          const p = fw.probs[i], t = tokens[i + 1], h = fw.T[i], dh = dT[i];
          for (let v = 0; v < V; v++) { const dl = wi * (p[v] - (v === t ? 1 : 0)); if (!dl) continue; g.bout[v] += dl; const off = v * D; for (let d = 0; d < D; d++) { g.Wout[off + d] += dl * h[d]; dh[d] += dl * this.Wout[off + d]; } }
        }
        let d = dT; for (let l = this.layers.length - 1; l >= 0; l--) d = this.layers[l].backward(fw.ctxs[l], d, g.layers[l]); // back through the stack
        for (let i = 0; i < n; i++) { const po = this.Pos && i < this.P ? i * D : -1; if (i === 0 && fw.code) { const C = this.C; for (let e = 0; e < D; e++) { g.bin[e] += d[0][e]; const o = e * C; for (let c = 0; c < C; c++) g.Pin[o + c] += d[0][e] * fw.code[c]; if (po >= 0) g.Pos[po + e] += d[0][e]; } continue; } const off = tokens[i] * D; for (let e = 0; e < D; e++) { g.E[off + e] += d[i][e]; if (po >= 0) g.Pos[po + e] += d[i][e]; } } // into the embeddings, or into the projection for an image first
      }
      const k = docs.length, own = [g.E, g.Wout, g.bout].concat(g.Pos ? [g.Pos] : [], g.Pin ? [g.Pin, g.bin] : []);
      let n2 = 0; for (const a of own) for (const x of a) n2 += x * x; for (const gl of g.layers) n2 += gradientNorm2(gl);
      this.lastGradientNorm = Math.sqrt(n2) / k;
      if (this.clip > 0 && this.lastGradientNorm > this.clip) { const f = this.clip / this.lastGradientNorm; for (const a of own) for (let i = 0; i < a.length; i++) a[i] *= f; for (const gl of g.layers) scaleGradient(gl, f); }
      if (this.optimizer === 'adam') this.adamStep(g, k, lr, l2);
      else { const step = (p, gp, decay) => { for (let i = 0; i < p.length; i++) p[i] -= lr * (gp[i] / k + (decay ? l2 * p[i] : 0)); }; step(this.E, g.E, 1); if (this.Pos) step(this.Pos, g.Pos, 1); step(this.Wout, g.Wout, 1); step(this.bout, g.bout); if (this.Pin) { step(this.Pin, g.Pin, 1); step(this.bin, g.bin); } this.layers.forEach((c, l) => c.applyGradient(g.layers[l], k, lr, l2)); }
      this.steps++;
      return loss / k;
    }
    // Adam: every weight gets its own step size, from running means of its gradient and of its square
    adamStep(g, k, lr, l2) {
      const st = this.adam || (this.adam = { t: 0, m: this.newGradient(), v: this.newGradient() }), b1 = 0.9, b2 = 0.999; st.t++;
      const c1 = 1 - Math.pow(b1, st.t), c2 = 1 - Math.pow(b2, st.t);
      const step = (p, gp, m, v, decay) => { for (let i = 0; i < p.length; i++) { const gi = gp[i] / k + (decay ? l2 * p[i] : 0); m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi; p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); } };
      step(this.E, g.E, st.m.E, st.v.E, 1); if (this.Pos) step(this.Pos, g.Pos, st.m.Pos, st.v.Pos, 1); step(this.Wout, g.Wout, st.m.Wout, st.v.Wout, 1); step(this.bout, g.bout, st.m.bout, st.v.bout, 0); if (this.Pin) { step(this.Pin, g.Pin, st.m.Pin, st.v.Pin, 1); step(this.bin, g.bin, st.m.bin, st.v.bin, 0); }
      this.layers.forEach((c, l) => { const gl = g.layers[l], ml = st.m.layers[l], vl = st.v.layers[l]; for (const key of Object.keys(gl)) step(c[key], gl[key], ml[key], vl[key], key[0] === 'W' ? 1 : 0); });
    }
    // docs: [tokens] or [{ tokens, weights }]; the mean loss, the next-token accuracy, and every document's per-position losses
    evaluate(docs) {
      let loss = 0, correct = 0, count = 0; const per = [];
      for (const doc of docs) { const tokens = doc.tokens || doc, fw = this.forward(tokens, doc.code || null), l = this.loss(fw, doc.weights || null); loss += l.loss; per.push(l.per); for (let i = 0; i + 1 < tokens.length; i++) { const p = fw.probs[i]; let best = 0; for (let v = 1; v < this.V; v++) if (p[v] > p[best]) best = v; if (best === tokens[i + 1]) correct++; count++; } }
      return { loss: docs.length ? loss / docs.length : 0, accuracy: count ? correct / count : 0, per };
    }
    // writing, one token at a time: a state holds every layer's keys and values for the tokens so far, so that each
    // new token costs one token's work rather than a pass over the whole text. genState runs a prefix through; genPush
    // adds one token and leaves the next token's probabilities on the state; genNext chooses the next token, the most
    // probable (temperature 0) or a draw from the probabilities sharpened or flattened by the temperature, and pushes it
    genState(prefix, code) { const st = { tokens: [], code: code || null, caches: this.layers.map(() => ({ K: [], V: [] })), logits: null, probs: null, A: null }; for (const t of prefix) this.genPush(st, t); return st; }
    genPush(st, token) {
      const D = this.D, V = this.V, i = st.tokens.length, x = new Float64Array(D), off = token * D, po = this.Pos && i < this.P ? i * D : -1;
      if (i === 0 && st.code && this.Pin) { const C = this.C; for (let d = 0; d < D; d++) { let s = this.bin[d]; const o = d * C; for (let c = 0; c < C; c++) s += this.Pin[o + c] * st.code[c]; x[d] = s + (po >= 0 ? this.Pos[po + d] : 0); } }
      else for (let d = 0; d < D; d++) x[d] = this.E[off + d] + (po >= 0 ? this.Pos[po + d] : 0);
      let h = x; const A = [];
      this.layers.forEach((c, l) => { const o = c.forwardOne(h, st.caches[l], i); h = o.y; A.push(o.A); });
      const z = new Float64Array(V); for (let v = 0; v < V; v++) { let sv = this.bout[v]; const o2 = v * D; for (let d = 0; d < D; d++) sv += this.Wout[o2 + d] * h[d]; z[v] = sv; }
      let mx = -Infinity; for (const v of z) mx = Math.max(mx, v); const p = Float64Array.from(z, v => Math.exp(v - mx)); let Z = 0; for (const v of p) Z += v; for (let v = 0; v < V; v++) p[v] /= Z;
      st.tokens.push(token); st.logits = z; st.probs = p; st.A = A;
      return p;
    }
    genNext(st, { temperature = 0, rng = Math.random } = {}) {
      const p = st.probs, V = this.V; let next;
      if (temperature <= 0) { next = 0; for (let v = 1; v < V; v++) if (p[v] > p[next]) next = v; }
      else { const z = st.logits; let mx = -Infinity; for (const v of z) mx = Math.max(mx, v / temperature); const e = Float64Array.from(z, v => Math.exp(v / temperature - mx)); let Z = 0; for (const v of e) Z += v; let r = rng() * Z; next = V - 1; for (let v = 0; v < V; v++) { r -= e[v]; if (r <= 0) { next = v; break; } } }
      this.genPush(st, next);
      return { token: next, probs: p };
    }
    // continue a prefix until the stop token or maxTokens: the new tokens and, per step, the probabilities chosen from
    generate(prefix, { maxTokens = 80, temperature = 0, rng = Math.random, stop = null, code = null } = {}) {
      const st = this.genState(prefix, code), steps = [];
      while (steps.length < maxTokens) { const { token, probs } = this.genNext(st, { temperature, rng }); steps.push({ token, p: probs }); if (token === stop) break; }
      return { tokens: st.tokens.slice(prefix.length), steps };
    }
    parameterCount() { return this.E.length + (this.Pos ? this.Pos.length : 0) + this.Wout.length + this.bout.length + (this.Pin ? this.Pin.length + this.bin.length : 0) + this.layers.reduce((a, c) => a + c.parameterCount(), 0); }
    describe() { return `${this.C ? `an image code of ${this.C} → projected as the first token · ` : ''}${this.V} words → embedding of ${this.D}${this.P ? ' + position' : ''} → ${this.layers.length ? `${this.layers.length} × ${this.layers[0].describe()}` : 'no context layer'} → single layer → softmax over the ${this.V} words`; }
    // the weights as plain numbers, to ship a trained model, and a model back from them
    toJSON(meta) {
      const r = v => Array.from(v, q => +q.toPrecision(5));
      return Object.assign({}, meta || {}, { vocabSize: this.V, dim: this.D, layers: this.layers.length, heads: this.H, dk: this.dk, ffn: this.F, positions: this.P, costs: this.costs, clip: this.clip, optimizer: this.optimizer, steps: this.steps,
        E: r(this.E), Pos: this.Pos ? r(this.Pos) : null, Wout: r(this.Wout), bout: r(this.bout), codeDim: this.C, Pin: this.Pin ? r(this.Pin) : null, bin: this.bin ? r(this.bin) : null, layerWeights: this.layers.map(c => ({ Wq: r(c.Wq), bq: r(c.bq), Wk: r(c.Wk), bk: r(c.bk), Wv: r(c.Wv), bv: r(c.bv), Wo: r(c.Wo), bo: r(c.bo), W1: r(c.W1), b1: r(c.b1), W2: r(c.W2), b2: r(c.b2), beta: r(c.beta) })) });
    }
    static fromJSON(o) {
      const lm = new LanguageModel({ vocabSize: o.vocabSize, dim: o.dim, layers: o.layers, heads: o.heads, dk: o.dk, ffn: o.ffn, costInit: o.costs, positions: o.positions, clip: o.clip, optimizer: o.optimizer, seed: 1, codeDim: o.codeDim || 0 });
      lm.E.set(o.E); if (o.Pos && lm.Pos) lm.Pos.set(o.Pos); lm.Wout.set(o.Wout); lm.bout.set(o.bout); if (o.Pin && lm.Pin) { lm.Pin.set(o.Pin); lm.bin.set(o.bin); }
      o.layerWeights.forEach((w, l) => { const c = lm.layers[l]; for (const k of Object.keys(w)) c[k].set(w[k]); });
      lm.steps = o.steps || 0; lm.meta = o;
      return lm;
    }
  }

  // ---------------------------------------------------------------------------- attention over a slide
  // A slide is a set of instances (the codes of its nuclei, with their positions when the question needs them) with ONE
  // label, or several: one context stack (0, 1 or more layers of self-attention) lets the instances look at each other,
  // then, for every output, a small scorer (D → A tanh → score) scores every instance, a softmax over the slide turns
  // the scores into weights that sum to 1, the instances are averaged with those weights into one summary, and a single
  // layer classifies the summary. Trained on the slide's label(s) only, each scorer learns where to look. With attention
  // off the weights are uniform: a plain average. scorer, head, context and the top-level fields of forward() and
  // evaluate() are the first output's (and the first layer's), so a one-output model reads as before.
  // the squared norm of a gradient accumulator, and the accumulator scaled: every Float64Array in it, in an array of
  // them, or a bare number (the dense net's output bias); a net's 'delta' slots are not weights and are left alone
  function gradientNorm2(g) { let s = 0; for (const key of Object.keys(g)) { if (key === 'delta') continue; const v = g[key]; if (v instanceof Float64Array) { for (const x of v) s += x * x; } else if (Array.isArray(v)) { for (const a of v) for (const x of a) s += x * x; } else if (typeof v === 'number') s += v * v; } return s; }
  function scaleGradient(g, f) { for (const key of Object.keys(g)) { if (key === 'delta') continue; const v = g[key]; if (v instanceof Float64Array) { for (let i = 0; i < v.length; i++) v[i] *= f; } else if (Array.isArray(v)) { for (const a of v) for (let i = 0; i < a.length; i++) a[i] *= f; } else if (typeof v === 'number') g[key] = v * f; } }
  class AttentionMIL {
    constructor({ inputSize, attentionUnits = 4, attention = true, context = null, outputs = 1, clip = 0, seed = 1 }) {
      this.D = inputSize; this.A = attentionUnits; this.attention = attention; this.seed = seed; this.K = outputs; this.clip = clip; // clip > 0: a step's mean gradient is scaled down to this norm when it is larger (a guard against the rare exploding step)
      const nl = context ? (context.layers || 1) : 0;
      this.layers = Array.from({ length: nl }, (_, l) => new ContextLayer({ inputSize, dk: context.dk || 8, ffn: context.ffn == null ? 8 : context.ffn, heads: context.heads || 1, distanceBias: !!context.distanceBias, excludeSelf: !!context.excludeSelf, costInit: context.costInit || 0, relative: !!context.relative, seed: seed + 2 + l }));
      this.context = this.layers[0] || null;
      this.scorers = Array.from({ length: outputs }, (_, k) => new Net({ inputSize, hidden: [attentionUnits], activation: 'tanh', seed: seed + 10 * k }));
      this.heads = Array.from({ length: outputs }, (_, k) => new Net({ inputSize, hidden: [], activation: 'relu', seed: seed + 1 + 10 * k }));
      this.scorer = this.scorers[0]; this.head = this.heads[0];
      this.steps = 0;
    }
    // H: the slide's instances; dist: their distances (with a distance bias); xy: their positions (with relative
    // messages). Returns the context (ctx: the first
    // layer's forward, ctxs: every layer's), the tokens after context (T), and per output (outs[k]) the scores, the
    // weights, the summary and the call; the first output's are also on the result itself.
    forward(H, dist, xy) {
      const ctxs = []; let T = H;
      for (const layer of this.layers) { const c = layer.forward(T, dist, xy); ctxs.push(c); T = c.Y; }
      const n = T.length, D = this.D;
      const outs = this.scorers.map((scorer, k) => {
        const fws = this.attention ? T.map(t => scorer.forward(t)) : null, s = new Float64Array(n), a = new Float64Array(n);
        if (this.attention) {
          let mx = -Infinity; for (let i = 0; i < n; i++) { s[i] = fws[i].z; mx = Math.max(mx, s[i]); }
          let Z = 0; for (let i = 0; i < n; i++) { a[i] = Math.exp(s[i] - mx); Z += a[i]; }
          for (let i = 0; i < n; i++) a[i] /= Z;
        } else a.fill(1 / n);
        const z = new Float64Array(D);
        for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) z[d] += a[i] * T[i][d];
        const head = this.heads[k].forward(z);
        return { s, a, z, fws, head, p: head.p, logit: head.z };
      });
      const o = outs[0];
      return { s: o.s, a: o.a, z: o.z, fws: o.fws, head: o.head, p: o.p, logit: o.logit, ctx: ctxs[0] || null, ctxs, T, outs };
    }
    // one gradient step on a batch of slides [{ H, y, dist }] (y a label, or one per output); returns their mean loss
    // before the step (summed over the outputs)
    trainBatch(slides, lr, l2 = 0) {
      const gss = this.scorers.map(s => s.newGradient()), ghs = this.heads.map(h => h.newGradient()), gcs = this.layers.map(c => c.newGradient()); let loss = 0;
      for (const sl of slides) {
        const fw = this.forward(sl.H, sl.dist, sl.xy), T = fw.T, n = T.length, ys = Array.isArray(sl.y) ? sl.y : [sl.y];
        const dT = gcs.length ? T.map(() => new Float64Array(this.D)) : null; // d loss / d instance, only needed with a context layer
        fw.outs.forEach((o, k) => {
          const y = ys[k], dl = o.p - y; loss += bce(o.p, y);
          const dz = this.heads[k].backDense(o.head, dl, ghs[k]); // d loss / d summary, the head's gradient collected on the way
          if (dT) for (let i = 0; i < n; i++) for (let d = 0; d < this.D; d++) dT[i][d] += o.a[i] * dz[d]; // through the weighted average
          if (this.attention) { // through the weighted average and the softmax to every score, then through the scorer
            const da = new Float64Array(n); let dot = 0;
            for (let i = 0; i < n; i++) { let v = 0; for (let d = 0; d < this.D; d++) v += dz[d] * T[i][d]; da[i] = v; dot += o.a[i] * v; }
            for (let i = 0; i < n; i++) { const ds = o.a[i] * (da[i] - dot); if (ds !== 0) { const dIn = this.scorers[k].backDense(o.fws[i], ds, gss[k]); if (dT) for (let d = 0; d < this.D; d++) dT[i][d] += dIn[d]; } }
          }
        });
        if (dT) { let d = dT; for (let l = this.layers.length - 1; l >= 0; l--) d = this.layers[l].backward(fw.ctxs[l], d, gcs[l]); } // back through the stack
      }
      const k = slides.length, all = ghs.concat(this.attention ? gss : [], gcs);
      this.lastGradientNorm = Math.sqrt(all.reduce((a, g) => a + gradientNorm2(g), 0)) / k; // the norm of the step's mean gradient, over every weight
      if (this.clip > 0 && this.lastGradientNorm > this.clip) for (const g of all) scaleGradient(g, this.clip / this.lastGradientNorm);
      this.heads.forEach((h, i) => h.applyGradient(ghs[i], k, lr, l2)); if (this.attention) this.scorers.forEach((s, i) => s.applyGradient(gss[i], k, lr, l2)); this.layers.forEach((c, i) => c.applyGradient(gcs[i], k, lr, l2));
      this.steps++;
      return loss / k;
    }
    // slides: [{ H, y, pos, dist }] with pos[i] true for the instances that are truly positive (never used to train),
    // or one such mask per output; returns per output (outputs[k], the first also on the result itself) the loss, the
    // accuracy, every slide's call and weights, and the share of a positive slide's attention that falls on its positive
    // instances (uniform weights give their share of the slide)
    evaluate(slides, threshold = 0.5) {
      const acc = this.scorers.map(() => ({ loss: 0, correct: 0, mass: 0, nPos: 0, probs: [], weights: [] }));
      for (const sl of slides) {
        const fw = this.forward(sl.H, sl.dist, sl.xy), ys = Array.isArray(sl.y) ? sl.y : [sl.y], poss = !sl.pos ? null : Array.isArray(sl.pos[0]) ? sl.pos : [sl.pos];
        fw.outs.forEach((o, k) => {
          const y = ys[k], A = acc[k], pos = poss ? poss[Math.min(k, poss.length - 1)] : null;
          A.probs.push(o.p); A.weights.push(o.a); A.loss += bce(o.p, y);
          if ((o.p >= threshold ? 1 : 0) === y) A.correct++;
          if (y && pos) { let m = 0; for (let i = 0; i < sl.H.length; i++) if (pos[i]) m += o.a[i]; A.mass += m; A.nPos++; }
        });
      }
      const outputs = acc.map(A => ({ loss: A.loss / slides.length, accuracy: A.correct / slides.length, probs: A.probs, weights: A.weights, culpritMass: A.nPos ? A.mass / A.nPos : null }));
      return Object.assign({}, outputs[0], { outputs });
    }
    parameterCount() { return this.heads.reduce((a, h) => a + h.parameterCount(), 0) + (this.attention ? this.scorers.reduce((a, s) => a + s.parameterCount(), 0) : 0) + this.layers.reduce((a, c) => a + c.parameterCount(), 0); }
    describe() { return `${this.layers.length ? `context: ${this.layers.length > 1 ? `${this.layers.length} layers of ` : ''}${this.layers[0].describe()} → ` : ''}${this.attention ? `attention (${this.D} → ${this.A} tanh → score) over the slide → weighted average → single layer` : `plain average over the slide → single layer`}${this.K > 1 ? `, × ${this.K} outputs` : ''}`; }
  }

  // z-scoring helpers. Fit on the training set only, apply to everything.
  function fitStandardizer(rows, { perDimScale = true } = {}) {
    const D = rows[0].length, n = rows.length;
    const mean = new Float64Array(D), scale = new Float64Array(D);
    for (const r of rows) for (let i = 0; i < D; i++) mean[i] += r[i] / n;
    if (perDimScale) {
      for (const r of rows) for (let i = 0; i < D; i++) { const d = r[i] - mean[i]; scale[i] += d * d / n; }
      for (let i = 0; i < D; i++) scale[i] = Math.sqrt(scale[i]) || 1;
    } else {
      let v = 0;
      for (const r of rows) for (let i = 0; i < D; i++) { const d = r[i] - mean[i]; v += d * d / (n * D); }
      scale.fill(Math.sqrt(v) || 1);
    }
    return {
      mean, scale,
      apply(r) { const out = new Float64Array(D); for (let i = 0; i < D; i++) out[i] = (r[i] - mean[i]) / scale[i]; return out; },
    };
  }

  // a standardiser from stored statistics (one scale for every dimension, or one per dimension)
  function standardizerFrom(mean, scale) {
    const D = mean.length, m = Float64Array.from(mean), sc = typeof scale === 'number' ? new Float64Array(D).fill(scale) : Float64Array.from(scale);
    return { mean: m, scale: sc, apply(r) { const out = new Float64Array(D); for (let i = 0; i < D; i++) out[i] = (r[i] - m[i]) / sc[i]; return out; } };
  }

  // finite-difference check of the analytic gradients on a tiny random instance; returns the worst relative error
  function gradientCheck(opts) {
    const net = new Net(Object.assign({ inputSize: 64, imageSize: 8, conv: { K: 2, f: 3, pool: 2 }, hidden: [3, 2], activation: 'sigmoid', seed: 3 }, opts || {}));
    const rnd = mulberry32(11);
    const x = new Float64Array(net.D); for (let i = 0; i < x.length; i++) x[i] = rnd() * 2 - 1;
    const y = 1, eps = 1e-6;
    const lossAt = () => bce(net.forward(x).p, y);
    // analytic gradient via one lr=1 step on a batch of one (new = old - grad), then restore
    const params = [...(net.conv ? [net.Wc, net.bc] : []), ...net.W, ...net.b, net.Wo];
    const before = params.map(p => Float64Array.from(p)); const bo = net.bo;
    net.trainBatch([x], [y], 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j]));
    params.forEach((p, i) => p.set(before[i])); net.bo = bo; net.steps = 0;
    let worst = 0, checked = 0;
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) {
      const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o;
      const num = (lp - lm) / (2 * eps);
      if (Math.abs(num) > 1e-5) { checked++; worst = Math.max(worst, Math.abs(num - analytic[pi][i]) / (Math.abs(num) + Math.abs(analytic[pi][i]))); }
    } });
    return { worst, checked };
  }

  // the same check for the autoencoder's gradients, on a tiny instance
  function gradientCheckAE() {
    const ae = new AutoEncoder({ imageSize: 8, conv: { K: 2, f: 3, pool: 2 }, code: 3, seed: 5 });
    const rnd = mulberry32(13), eps = 1e-6;
    const x = new Float64Array(ae.D); for (let i = 0; i < x.length; i++) x[i] = rnd() * 2 - 1;
    const lossAt = () => ae.forward(x).loss;
    const params = [ae.enc.Wc, ae.enc.bc, ae.enc.W[0], ae.enc.b[0], ae.Wd, ae.bd];
    const before = params.map(p => Float64Array.from(p));
    ae.trainBatch([x], 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j]));
    params.forEach((p, i) => p.set(before[i]));
    let worst = 0, checked = 0;
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) {
      const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o;
      const num = (lp - lm) / (2 * eps);
      if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - analytic[pi][i]) / (Math.abs(num) + Math.abs(analytic[pi][i]))); }
    } });
    return { worst, checked };
  }

  // and for the contrastive loss, on a tiny instance with three pairs
  function gradientCheckCL() {
    const cl = new Contrastive({ imageSize: 8, conv: { K: 2, f: 3, pool: 2 }, code: 3, tau: 0.3, seed: 9 });
    const rnd = mulberry32(17), eps = 1e-6;
    const mk = () => { const x = new Float64Array(cl.D); for (let i = 0; i < x.length; i++) x[i] = rnd() * 2 - 1; return x; };
    const pairs = [[mk(), mk()], [mk(), mk()], [mk(), mk()]];
    const lossAt = () => cl.evaluate(pairs).loss;
    const params = [cl.enc.Wc, cl.enc.bc, cl.enc.W[0], cl.enc.b[0]];
    const before = params.map(p => Float64Array.from(p));
    cl.step(pairs, 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j]));
    params.forEach((p, i) => p.set(before[i]));
    let worst = 0, checked = 0;
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) {
      const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o;
      const num = (lp - lm) / (2 * eps);
      if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - analytic[pi][i]) / (Math.abs(num) + Math.abs(analytic[pi][i]))); }
    } });
    return { worst, checked };
  }

  // and for the attention model, on two small slides
  function gradientCheckMIL() {
    const mil = new AttentionMIL({ inputSize: 5, attentionUnits: 3, seed: 4 });
    const rnd = mulberry32(21), eps = 1e-6, mk = () => Float64Array.from({ length: 5 }, () => rnd() * 2 - 1);
    const slides = [{ H: [mk(), mk(), mk(), mk()], y: 1 }, { H: [mk(), mk(), mk()], y: 0 }];
    const lossAt = () => slides.reduce((a, sl) => a + bce(mil.forward(sl.H).p, sl.y), 0) / slides.length;
    const params = [mil.scorer.W[0], mil.scorer.b[0], mil.scorer.Wo, mil.head.Wo];
    const before = params.map(p => Float64Array.from(p)), bo = [mil.scorer.bo, mil.head.bo];
    mil.trainBatch(slides, 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j]));
    params.forEach((p, i) => p.set(before[i])); mil.scorer.bo = bo[0]; mil.head.bo = bo[1];
    let worst = 0, checked = 0;
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) {
      const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o;
      const num = (lp - lm) / (2 * eps);
      if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - analytic[pi][i]) / (Math.abs(num) + Math.abs(analytic[pi][i]))); }
    } });
    return { worst, checked };
  }

  function gradientCheckContext() {
    const mil = new AttentionMIL({ inputSize: 5, attentionUnits: 3, context: { dk: 3, ffn: 3, distanceBias: true, excludeSelf: true, costInit: 0.8 }, seed: 5 });
    const rnd = mulberry32(22), eps = 1e-6, mk = () => Float64Array.from({ length: 5 }, () => rnd() * 2 - 1);
    const distOf = n => { const d = Array.from({ length: n }, () => new Float64Array(n)); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = rnd() * 2; d[i][j] = v; d[j][i] = v; } return d; };
    const slides = [{ H: [mk(), mk(), mk(), mk()], dist: distOf(4), y: 1 }, { H: [mk(), mk(), mk()], dist: distOf(3), y: 0 }];
    const c = mil.context; c.beta[0] = 0.4;
    const lossAt = () => slides.reduce((a, sl) => a + bce(mil.forward(sl.H, sl.dist).p, sl.y), 0) / slides.length;
    const params = [c.Wq, c.bq, c.Wk, c.bk, c.Wv, c.bv, c.Wo, c.bo, c.W1, c.b1, c.W2, c.b2, mil.scorer.W[0], mil.scorer.b[0], mil.scorer.Wo, mil.head.Wo];
    const before = params.map(p => Float64Array.from(p)), scalars = [mil.scorer.bo, mil.head.bo], beta0 = Float64Array.from(c.beta);
    mil.trainBatch(slides, 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j])), aBeta = beta0.map((v, h) => v - c.beta[h]);
    params.forEach((p, i) => p.set(before[i])); mil.scorer.bo = scalars[0]; mil.head.bo = scalars[1]; c.beta.set(beta0);
    let worst = 0, checked = 0;
    const cmp = (num, an) => { if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - an) / (Math.abs(num) + Math.abs(an))); } };
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) { const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o; cmp((lp - lm) / (2 * eps), analytic[pi][i]); } });
    for (let h = 0; h < c.H; h++) { const o = c.beta[h]; c.beta[h] = o + eps; const lp = lossAt(); c.beta[h] = o - eps; const lm = lossAt(); c.beta[h] = o; cmp((lp - lm) / (2 * eps), aBeta[h]); }
    return { worst, checked };
  }
  // the field model: two layers of two heads, two outputs, every parameter against finite differences
  function gradientCheckField({ relative = false } = {}) {
    const mil = new AttentionMIL({ inputSize: 5, attentionUnits: 3, outputs: 2, context: { dk: 3, ffn: 3, heads: 2, layers: 2, distanceBias: true, excludeSelf: true, costInit: 0.8, relative }, seed: 6 });
    const rnd = mulberry32(23), eps = 1e-5, mk = () => Float64Array.from({ length: 5 }, () => rnd() * 2 - 1); // eps: the smallest gradients here are ~1e-6, where 1e-6 steps leave 1e-4 of round-off
    const distOf = n => { const d = Array.from({ length: n }, () => new Float64Array(n)); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = rnd() * 2; d[i][j] = v; d[j][i] = v; } return d; };
    const xyOf = n => (relative ? Array.from({ length: n }, () => [rnd() * 2 - 1, rnd() * 2 - 1]) : null);
    const slides = [{ H: [mk(), mk(), mk(), mk()], dist: distOf(4), xy: xyOf(4), y: [1, 0] }, { H: [mk(), mk(), mk()], dist: distOf(3), xy: xyOf(3), y: [0, 1] }, { H: [mk(), mk(), mk(), mk(), mk()], dist: distOf(5), xy: xyOf(5), y: [1, 1] }];
    mil.layers[0].beta.set([0.4, -0.2]); mil.layers[1].beta.set([0.1, 0.6]);
    const lossAt = () => slides.reduce((a, sl) => { const fw = mil.forward(sl.H, sl.dist, sl.xy); return a + fw.outs.reduce((b, o, k) => b + bce(o.p, sl.y[k]), 0); }, 0) / slides.length;
    const params = [].concat(...mil.layers.map(c => [c.Wq, c.bq, c.Wk, c.bk, c.Wv, c.bv, c.Wo, c.bo, c.W1, c.b1, c.W2, c.b2]), ...mil.scorers.map(s => [s.W[0], s.b[0], s.Wo]), mil.heads.map(h => h.Wo));
    const before = params.map(p => Float64Array.from(p)), sbo = mil.scorers.map(s => s.bo), hbo = mil.heads.map(h => h.bo), betas = mil.layers.map(c => Float64Array.from(c.beta));
    mil.trainBatch(slides, 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j])), aBeta = mil.layers.map((c, l) => betas[l].map((v, h) => v - c.beta[h]));
    params.forEach((p, i) => p.set(before[i])); mil.scorers.forEach((s, i) => { s.bo = sbo[i]; }); mil.heads.forEach((h, i) => { h.bo = hbo[i]; }); mil.layers.forEach((c, l) => c.beta.set(betas[l]));
    let worst = 0, checked = 0;
    const cmp = (num, an) => { if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - an) / (Math.abs(num) + Math.abs(an))); } };
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) { const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o; cmp((lp - lm) / (2 * eps), analytic[pi][i]); } });
    mil.layers.forEach((c, l) => { for (let h = 0; h < c.H; h++) { const o = c.beta[h]; c.beta[h] = o + eps; const lp = lossAt(); c.beta[h] = o - eps; const lm = lossAt(); c.beta[h] = o; cmp((lp - lm) / (2 * eps), aBeta[l][h]); } });
    return { worst, checked };
  }
  // the language model: embeddings, positions, both causal layers and the output layer, against finite differences
  function gradientCheckLM({ positions = 4, codeDim = 0 } = {}) {
    const lm = new LanguageModel({ vocabSize: 6, dim: 4, layers: 2, heads: 2, dk: 3, ffn: 3, costInit: 0.3, positions, clip: 0, optimizer: 'sgd', seed: 7, codeDim });
    lm.layers[0].beta.set([0.4, -0.2]); lm.layers[1].beta.set([0.1, 0.6]);
    const docs = [{ tokens: [0, 3, 1, 4, 2], weights: [1, 0.5, 1, 1], code: codeDim ? Float64Array.from({ length: codeDim }, (_, c) => 0.7 - 0.4 * c) : null }, { tokens: [5, 2, 2, 1], code: codeDim ? Float64Array.from({ length: codeDim }, (_, c) => -0.3 + 0.5 * c) : null }], eps = 1e-5;
    const lossAt = () => docs.reduce((a, d) => a + lm.loss(lm.forward(d.tokens, d.code), d.weights || null).loss, 0) / docs.length;
    const params = [lm.E, lm.Wout, lm.bout].concat(lm.Pos ? [lm.Pos] : [], lm.Pin ? [lm.Pin, lm.bin] : [], ...lm.layers.map(c => [c.Wq, c.bq, c.Wk, c.bk, c.Wv, c.bv, c.Wo, c.bo, c.W1, c.b1, c.W2, c.b2]));
    const before = params.map(p => Float64Array.from(p)), betas = lm.layers.map(c => Float64Array.from(c.beta));
    lm.trainBatch(docs, 1);
    const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j])), aBeta = lm.layers.map((c, l) => betas[l].map((v, h) => v - c.beta[h]));
    params.forEach((p, i) => p.set(before[i])); lm.layers.forEach((c, l) => c.beta.set(betas[l]));
    let worst = 0, checked = 0;
    const cmp = (num, an) => { if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - an) / (Math.abs(num) + Math.abs(an))); } };
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) { const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm2 = lossAt(); p[i] = o; cmp((lp - lm2) / (2 * eps), analytic[pi][i]); } });
    lm.layers.forEach((c, l) => { for (let h = 0; h < c.H; h++) { const o = c.beta[h]; c.beta[h] = o + eps; const lp = lossAt(); c.beta[h] = o - eps; const lm2 = lossAt(); c.beta[h] = o; cmp((lp - lm2) / (2 * eps), aBeta[l][h]); } });
    return { worst, checked };
  }
  // ---------------------------------------------------------------------------- segmentation: a class for every pixel
  // A tiny fully convolutional network: 3 × 3 convolutions with 'same' padding and ReLU at the image's own scale,
  // and, for context, levels below it: max-pooling 2 × 2, two convolutions on the coarser map, nearest-neighbour
  // upsampling and a join with the finer features (a U-Net in miniature). A 1 × 1 convolution scores every pixel for
  // every class and the softmax over the classes, with the cross-entropy against the mask, trains it. The same
  // weights work on a crop and on a whole field: nothing in the network knows how big its input is. Tensors are
  // Float64Arrays laid out [channel][row][column].
  class SegNet {
    constructor({ channels = 8, levels = 1, classes = 4, seed = 1, optimizer = 'adam', clip = 1, classWeights = null, norm = null }) {
      this.C = channels; this.L = levels; this.K = classes; this.seed = seed; this.optimizer = optimizer; this.clip = clip; this.classWeights = classWeights ? Float64Array.from(classWeights) : null; this.norm = norm || { mean: 0.5 };
      const rnd = mulberry32(seed * 104729 + 11), mk = (n, fanIn) => { const s = Math.sqrt(6 / fanIn); return Float64Array.from({ length: n }, () => (rnd() * 2 - 1) * s); }; // He-uniform
      const conv = (cin, cout) => ({ W: mk(cout * cin * 9, cin * 9), b: new Float64Array(cout), cin, cout });
      this.enc = []; for (let l = 0; l <= levels; l++) this.enc.push([conv(l === 0 ? 1 : channels, channels), conv(channels, channels)]); // two convolutions per level, the first level on the image
      this.dec = []; for (let l = levels - 1; l >= 0; l--) this.dec.push(conv(2 * channels, channels)); // one convolution per level on the way up, on the upsampled map joined to that level's features
      this.head = { W: mk(classes * channels, channels), b: new Float64Array(classes), cin: channels, cout: classes };
      this.steps = 0; this.adam = null;
    }
    layers() { const out = []; for (const pair of this.enc) out.push(pair[0], pair[1]); for (const d of this.dec) out.push(d); out.push(this.head); return out; }
    parameterCount() { return this.layers().reduce((a, c) => a + c.W.length + c.b.length, 0); }
    receptiveField() { let r = 1; for (let l = 0; l <= this.L; l++) r += 4 * Math.pow(2, l); for (let l = this.L - 1; l >= 0; l--) r += 2 * Math.pow(2, l); return r; } // every 3 × 3 convolution adds 2 at its level's scale
    describe() { return `${this.L ? `${this.L} level${this.L > 1 ? 's' : ''} of context (pool, convolve, upsample, join)` : 'no context below the pixels'} · ${this.C} channels · 3 × 3 convolutions, ReLU · receptive field ${this.receptiveField()} px · ${this.K} classes per pixel`; }
    // the 3 × 3 convolution with 'same' zero padding: pre-activations, and the activations after ReLU when asked
    conv3(x, layer, H, W, relu) {
      const { W: Wt, b, cin, cout } = layer, pre = new Float64Array(cout * H * W);
      for (let o = 0; o < cout; o++) {
        const po = o * H * W;
        for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) pre[po + y * W + xx] = b[o];
        for (let c = 0; c < cin; c++) { const wo = (o * cin + c) * 9, xc = c * H * W;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const wv = Wt[wo + (dy + 1) * 3 + dx + 1]; if (!wv) continue;
            const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
            for (let y = y0; y < y1; y++) { const prow = po + y * W, xrow = xc + (y + dy) * W + dx; for (let xx = x0; xx < x1; xx++) pre[prow + xx] += wv * x[xrow + xx]; }
          } }
      }
      if (!relu) return { pre, act: pre };
      const act = new Float64Array(pre.length); for (let i = 0; i < pre.length; i++) act[i] = pre[i] > 0 ? pre[i] : 0; return { pre, act };
    }
    // the backward of conv3: the gradient of the weights and the bias into g, and the gradient at the input
    conv3Back(x, layer, H, W, dOut, g) {
      const { W: Wt, cin, cout } = layer, dX = new Float64Array(cin * H * W);
      for (let o = 0; o < cout; o++) {
        const po = o * H * W; let sb = 0; for (let i = 0; i < H * W; i++) sb += dOut[po + i]; g.b[o] += sb;
        for (let c = 0; c < cin; c++) { const wo = (o * cin + c) * 9, xc = c * H * W;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const wi = wo + (dy + 1) * 3 + dx + 1, wv = Wt[wi]; let sw = 0;
            const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
            for (let y = y0; y < y1; y++) { const prow = po + y * W, xrow = xc + (y + dy) * W + dx; for (let xx = x0; xx < x1; xx++) { const d = dOut[prow + xx]; sw += d * x[xrow + xx]; dX[xrow + xx] += wv * d; } }
            g.W[wi] += sw;
          } }
      }
      return dX;
    }
    pool2(x, C, H, W) { const h = H >> 1, w = W >> 1, v = new Float64Array(C * h * w), arg = new Int32Array(C * h * w); for (let c = 0; c < C; c++) for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) { let best = -Infinity, bi = -1; for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) { const i = (c * H + 2 * y + dy) * W + 2 * xx + dx; if (x[i] > best) { best = x[i]; bi = i; } } const j = (c * h + y) * w + xx; v[j] = best; arg[j] = bi; } return { v, arg, h, w }; }
    up2(v, C, h, w, H, W) { const out = new Float64Array(C * H * W); for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) { const ys = Math.min(h - 1, y >> 1); for (let xx = 0; xx < W; xx++) out[(c * H + y) * W + xx] = v[(c * h + ys) * w + Math.min(w - 1, xx >> 1)]; } return out; }
    up2Back(d, C, h, w, H, W) { const out = new Float64Array(C * h * w); for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) { const ys = Math.min(h - 1, y >> 1); for (let xx = 0; xx < W; xx++) out[(c * h + ys) * w + Math.min(w - 1, xx >> 1)] += d[(c * H + y) * W + xx]; } return out; }
    // x: the image as one channel of H × W numbers (ink, 0 to 1); returns every level's activations and the probabilities per pixel
    forward(x, H, W) {
      const C = this.C, K = this.K, levels = [], input = new Float64Array(x.length); for (let i = 0; i < x.length; i++) input[i] = x[i] - this.norm.mean;
      let cur = input, h = H, w = W, cin = 1;
      for (let l = 0; l <= this.L; l++) { const c1 = this.conv3(cur, this.enc[l][0], h, w, true), c2 = this.conv3(c1.act, this.enc[l][1], h, w, true); const lev = { x: cur, cin, h, w, c1, c2 }; levels.push(lev); if (l < this.L) { const p = this.pool2(c2.act, C, h, w); lev.pool = p; cur = p.v; h = p.h; w = p.w; cin = C; } }
      let top = levels[this.L].c2.act; const decs = [];
      for (let k = 0; k < this.dec.length; k++) { const l = this.L - 1 - k, lev = levels[l], up = this.up2(top, C, levels[l + 1].h, levels[l + 1].w, lev.h, lev.w), cat = new Float64Array(2 * C * lev.h * lev.w); cat.set(lev.c2.act, 0); cat.set(up, C * lev.h * lev.w); const d = this.conv3(cat, this.dec[k], lev.h, lev.w, true); decs.push({ l, up, cat, d }); top = d.act; }
      const N = H * W, logits = new Float64Array(K * N), probs = new Float64Array(K * N), hd = this.head;
      for (let k = 0; k < K; k++) { const ko = k * N; for (let i = 0; i < N; i++) { let s = hd.b[k]; for (let c = 0; c < C; c++) s += hd.W[k * C + c] * top[c * N + i]; logits[ko + i] = s; } }
      for (let i = 0; i < N; i++) { let mx = -Infinity; for (let k = 0; k < K; k++) mx = Math.max(mx, logits[k * N + i]); let Z = 0; for (let k = 0; k < K; k++) { const e = Math.exp(logits[k * N + i] - mx); probs[k * N + i] = e; Z += e; } for (let k = 0; k < K; k++) probs[k * N + i] /= Z; }
      return { H, W, levels, decs, top, logits, probs };
    }
    // the mean cross-entropy over the pixels (weighted by class when the network has class weights) and the pixel accuracy
    loss(fw, y) { const K = this.K, N = fw.H * fw.W, cw = this.classWeights; let s = 0, wsum = 0, right = 0; for (let i = 0; i < N; i++) { const t = y[i], wi = cw ? cw[t] : 1; s += -wi * Math.log(fw.probs[t * N + i] + EPS); wsum += wi; let best = 0; for (let k = 1; k < K; k++) if (fw.probs[k * N + i] > fw.probs[best * N + i]) best = k; if (best === t) right++; } return { loss: wsum ? s / wsum : 0, accuracy: right / N }; }
    predict(x, H, W) { const fw = this.forward(x, H, W), N = H * W, labels = new Uint8Array(N); for (let i = 0; i < N; i++) { let best = 0; for (let k = 1; k < this.K; k++) if (fw.probs[k * N + i] > fw.probs[best * N + i]) best = k; labels[i] = best; } return { labels, probs: fw.probs, fw }; }
    newGradient() { return this.layers().map(c => ({ W: new Float64Array(c.W.length), b: new Float64Array(c.b.length) })); }
    // one step on a batch of samples [{ x, y, H, W }]: the mean loss before the step
    trainBatch(samples, lr, l2 = 0) {
      const g = this.newGradient(), C = this.C, K = this.K, cw = this.classWeights; let loss = 0;
      for (const s of samples) {
        const fw = this.forward(s.x, s.H, s.W), N = s.H * s.W; loss += this.loss(fw, s.y).loss;
        let wsum = 0; for (let i = 0; i < N; i++) wsum += cw ? cw[s.y[i]] : 1;
        const dL = new Float64Array(K * N); for (let i = 0; i < N; i++) { const t = s.y[i], wi = (cw ? cw[t] : 1) / wsum; for (let k = 0; k < K; k++) dL[k * N + i] = wi * (fw.probs[k * N + i] - (k === t ? 1 : 0)); }
        const gi = this.layers().length - 1, gh = g[gi], hd = this.head, dTop = new Float64Array(C * N);
        for (let k = 0; k < K; k++) { let sb = 0; for (let i = 0; i < N; i++) { const d = dL[k * N + i]; sb += d; for (let c = 0; c < C; c++) { gh.W[k * C + c] += d * fw.top[c * N + i]; dTop[c * N + i] += hd.W[k * C + c] * d; } } gh.b[k] += sb; }
        let d = dTop; // back down the decoder
        for (let k = fw.decs.length - 1; k >= 0; k--) { const dc = fw.decs[k], lev = fw.levels[dc.l], n = lev.h * lev.w; const dPre = new Float64Array(d.length); for (let i = 0; i < d.length; i++) dPre[i] = dc.d.pre[i] > 0 ? d[i] : 0; const dCat = this.conv3Back(dc.cat, this.dec[k], lev.h, lev.w, dPre, g[2 * (this.L + 1) + k]); lev.dSkip = dCat.subarray(0, C * n); d = this.up2Back(dCat.subarray(C * n), C, fw.levels[dc.l + 1].h, fw.levels[dc.l + 1].w, lev.h, lev.w); }
        for (let l = this.L; l >= 0; l--) { const lev = fw.levels[l], n = lev.h * lev.w; let dAct = new Float64Array(C * n); if (l === this.L) dAct.set(d); else { const p = lev.pool; for (let j = 0; j < d.length; j++) dAct[p.arg[j]] += d[j]; } if (lev.dSkip) for (let i = 0; i < dAct.length; i++) dAct[i] += lev.dSkip[i];
          const d2 = new Float64Array(C * n); for (let i = 0; i < d2.length; i++) d2[i] = lev.c2.pre[i] > 0 ? dAct[i] : 0; const d1a = this.conv3Back(lev.c1.act, this.enc[l][1], lev.h, lev.w, d2, g[2 * l + 1]);
          const d1 = new Float64Array(C * n); for (let i = 0; i < d1.length; i++) d1[i] = lev.c1.pre[i] > 0 ? d1a[i] : 0; d = this.conv3Back(lev.x, this.enc[l][0], lev.h, lev.w, d1, g[2 * l]); }
      }
      const k = samples.length; let n2 = 0; for (const gl of g) { for (const v of gl.W) n2 += v * v; for (const v of gl.b) n2 += v * v; } this.lastGradientNorm = Math.sqrt(n2) / k;
      if (this.clip > 0 && this.lastGradientNorm > this.clip) { const f = this.clip / this.lastGradientNorm; for (const gl of g) { for (let i = 0; i < gl.W.length; i++) gl.W[i] *= f; for (let i = 0; i < gl.b.length; i++) gl.b[i] *= f; } }
      const layers = this.layers();
      if (this.optimizer === 'adam') {
        const st = this.adam || (this.adam = { t: 0, m: this.newGradient(), v: this.newGradient() }), b1 = 0.9, b2 = 0.999; st.t++; const c1 = 1 - Math.pow(b1, st.t), c2 = 1 - Math.pow(b2, st.t);
        const step = (p, gp, m, v, decay) => { for (let i = 0; i < p.length; i++) { const gi = gp[i] / k + (decay ? l2 * p[i] : 0); m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi; p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); } };
        layers.forEach((c, i) => { step(c.W, g[i].W, st.m[i].W, st.v[i].W, 1); step(c.b, g[i].b, st.m[i].b, st.v[i].b, 0); });
      } else layers.forEach((c, i) => { for (let j = 0; j < c.W.length; j++) c.W[j] -= lr * (g[i].W[j] / k + l2 * c.W[j]); for (let j = 0; j < c.b.length; j++) c.b[j] -= lr * g[i].b[j] / k; });
      this.steps++;
      return loss / k;
    }
    toJSON(meta) { const r = v => Array.from(v, q => +q.toPrecision(5)); return Object.assign({}, meta || {}, { channels: this.C, levels: this.L, classes: this.K, clip: this.clip, optimizer: this.optimizer, classWeights: this.classWeights ? Array.from(this.classWeights) : null, norm: this.norm, steps: this.steps, enc: this.enc.map(pair => pair.map(c => ({ W: r(c.W), b: r(c.b) }))), dec: this.dec.map(c => ({ W: r(c.W), b: r(c.b) })), head: { W: r(this.head.W), b: r(this.head.b) } }); }
    static fromJSON(o) { const net = new SegNet({ channels: o.channels, levels: o.levels, classes: o.classes, clip: o.clip, optimizer: o.optimizer, classWeights: o.classWeights, norm: o.norm, seed: 1 }); o.enc.forEach((pair, l) => pair.forEach((c, i) => { net.enc[l][i].W.set(c.W); net.enc[l][i].b.set(c.b); })); o.dec.forEach((c, i) => { net.dec[i].W.set(c.W); net.dec[i].b.set(c.b); }); net.head.W.set(o.head.W); net.head.b.set(o.head.b); net.steps = o.steps || 0; return net; }
  }
  function gradientCheckSeg({ levels = 1, channels = 2, classes = 3, H = 8, W = 8, weighted = false } = {}) {
    const net = new SegNet({ channels, levels, classes, seed: 5, optimizer: 'sgd', clip: 0, classWeights: weighted ? [1, 2, 0.5] : null, norm: { mean: 0.3 } }), rnd = mulberry32(99);
    const samples = [0, 1].map(() => ({ x: Float64Array.from({ length: H * W }, () => rnd()), y: Uint8Array.from({ length: H * W }, () => Math.floor(rnd() * classes)), H, W })), eps = 1e-5;
    const lossAt = () => samples.reduce((a, s) => a + net.loss(net.forward(s.x, s.H, s.W), s.y).loss, 0) / samples.length;
    const params = []; for (const c of net.layers()) params.push(c.W, c.b);
    const before = params.map(p => Float64Array.from(p)); net.trainBatch(samples, 1); const analytic = params.map((p, i) => before[i].map((v, j) => v - p[j])); params.forEach((p, i) => p.set(before[i]));
    let worst = 0, checked = 0; const cmp = (num, an) => { if (Math.abs(num) > 1e-6) { checked++; worst = Math.max(worst, Math.abs(num - an) / (Math.abs(num) + Math.abs(an))); } };
    params.forEach((p, pi) => { for (let i = 0; i < p.length; i++) { const o = p[i]; p[i] = o + eps; const lp = lossAt(); p[i] = o - eps; const lm = lossAt(); p[i] = o; cmp((lp - lm) / (2 * eps), analytic[pi][i]); } });
    return { worst, checked };
  }
  // ---- the two directions of a table of V rows of D numbers that spread the rows most (the first two principal
  // components, by power iteration on the covariance), with the rows projected on them, centred on the table's mean;
  // prev, two axes found earlier, fixes the signs so that a map drawn epoch after epoch does not flip
  function pca2(E, V, D, prev) {
    const mean = new Float64Array(D); for (let v = 0; v < V; v++) for (let d = 0; d < D; d++) mean[d] += E[v * D + d] / V;
    const C = new Float64Array(D * D), x = new Float64Array(D);
    for (let v = 0; v < V; v++) { for (let d = 0; d < D; d++) x[d] = E[v * D + d] - mean[d]; for (let a = 0; a < D; a++) { const xa = x[a] / V; if (!xa) continue; const off = a * D; for (let b = 0; b < D; b++) C[off + b] += xa * x[b]; } }
    const axis = (deflate, seed) => { let u = Float64Array.from({ length: D }, (_, d) => Math.cos(d * 1.7 + seed)); for (let it = 0; it < 80; it++) { const w = new Float64Array(D); for (let a = 0; a < D; a++) { let s = 0; const off = a * D; for (let b = 0; b < D; b++) s += C[off + b] * u[b]; w[a] = s; } if (deflate) { let dot = 0; for (let d = 0; d < D; d++) dot += w[d] * deflate[d]; for (let d = 0; d < D; d++) w[d] -= dot * deflate[d]; } let nrm = 0; for (const v of w) nrm += v * v; nrm = Math.sqrt(nrm) || 1; u = Float64Array.from(w, v => v / nrm); } return u; };
    const u1 = axis(null, 0), u2 = axis(u1, 1);
    for (const [u, p] of [[u1, prev && prev[0]], [u2, prev && prev[1]]]) if (p) { let dot = 0; for (let d = 0; d < D; d++) dot += u[d] * p[d]; if (dot < 0) for (let d = 0; d < D; d++) u[d] = -u[d]; }
    const lam = u => { let s = 0; for (let a = 0; a < D; a++) { let t = 0; const off = a * D; for (let b = 0; b < D; b++) t += C[off + b] * u[b]; s += u[a] * t; } return s; };
    let trace = 0; for (let d = 0; d < D; d++) trace += C[d * D + d];
    const xy = project2(E, V, D, [u1, u2]), xs = new Float64Array(V), ys = new Float64Array(V); for (let v = 0; v < V; v++) { xs[v] = xy[2 * v]; ys[v] = xy[2 * v + 1]; }
    return { x: xs, y: ys, axes: [u1, u2], share: trace > 0 ? (lam(u1) + lam(u2)) / trace : 0 };
  }
  // the rows of a table, centred on its mean, on two given axes: x and y interleaved, 2V numbers
  function project2(E, V, D, axes) {
    const mean = new Float64Array(D); for (let v = 0; v < V; v++) for (let d = 0; d < D; d++) mean[d] += E[v * D + d] / V;
    const xy = new Float32Array(2 * V), [u1, u2] = axes;
    for (let v = 0; v < V; v++) { let sx = 0, sy = 0; for (let d = 0; d < D; d++) { const e = E[v * D + d] - mean[d]; sx += e * u1[d]; sy += e * u2[d]; } xy[2 * v] = sx; xy[2 * v + 1] = sy; }
    return xy;
  }
  // the words' way through training, for the page's word map: snapshots of the embedding table (E) and of the output
  // table (Wout), one per epoch from epoch 0, each centred and projected on the two directions of the final tables,
  // so that every step of a word's path lies on the axes the finished map is drawn with; the epochs kept are the
  // first six, then every one the filter keeps, and the last
  function wordTrails(snaps, finalE, finalOut, V, D, keepEpoch) {
    const last = snaps[snaps.length - 1].epoch, keep = snaps.filter(s => s.epoch <= 6 || s.epoch === last || (keepEpoch ? keepEpoch(s.epoch) : s.epoch % 5 === 0 || s.epoch === 8 || s.epoch === 12));
    const t = { epochs: keep.map(s => s.epoch) };
    for (const [key, final, field] of [['in', finalE, 'E'], ['out', finalOut, 'out']]) { const axes = pca2(final, V, D, null).axes; t[key] = { axes: axes.map(a => Array.from(a, v => +v.toFixed(5))), xy: keep.map(s => Array.from(project2(s[field], V, D, axes), v => +v.toFixed(3))) }; }
    return t;
  }
  return { Net, TinyNet: Net, AutoEncoder, Contrastive, AttentionMIL, ContextLayer, LanguageModel, ACTIVATIONS, mulberry32, pca2, project2, wordTrails, SegNet, gradientCheckSeg, fitStandardizer, standardizerFrom, bce, sigmoid, gradientCheck, gradientCheckAE, gradientCheckCL, gradientCheckMIL, gradientCheckContext, gradientCheckField, gradientCheckLM };
});
