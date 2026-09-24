/*
 * nn.js — a small neural network with hand-written backpropagation.
 *
 *   input  ->  [optional convolution: K filters f×f, ReLU, max-pool p×p]  ->  [0, 1 or 2 dense hidden layers]  ->  sigmoid output
 *
 * The output is P(positive class). Trained with mini-batch gradient descent on binary cross-entropy,
 * with optional weight decay. Works in the browser (window.TinyNet) and in Node (module.exports).
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
  };
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  // Evaluate training loss from the logit: no epsilon changes the objective whose derivative is p − y.
  const bce = (p, y) => (y === 0 ? 0 : -y * Math.log(p)) + (y === 1 ? 0 : -(1 - y) * Math.log1p(-p));
  const bceFromLogit = (z, y) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
  function parameterStep(before, gradient, lr, l2 = 0, bias = false) {
    const decayGradient = bias ? 0 : l2 * before;
    const after = before * (1 - lr * (bias ? 0 : l2)) - lr * gradient;
    return { before, gradient, decayGradient, totalGradient: gradient + decayGradient, after, change: after - before };
  }

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

    // backpropagate d(loss)/dz = dz through the dense layers; returns d(loss)/d(a0) (a0 = pooled conv features or the raw input)
    backDense(fw, dz, g) {
      const act = ACTIVATIONS[this.activation];
      const top = fw.a[fw.a.length - 1];
      let d = new Float64Array(top.length);
      if (g) g.gbo += dz;
      for (let i = 0; i < top.length; i++) { if (g) g.gWo[i] += dz * top[i]; d[i] = dz * this.Wo[i]; }
      for (let l = this.hidden.length - 1; l >= 0; l--) {
        const fanIn = this.sizes[l], fanOut = this.sizes[l + 1], inp = fw.a[l], W = this.W[l];
        const dn = new Float64Array(fanIn);
        for (let j = 0; j < fanOut; j++) {
          const slope = act.df(fw.pre[l][j], fw.a[l + 1][j]);
          const dj = d[j] * slope;
          if (g && g.upstream) { g.upstream[l][j] = d[j]; g.slopes[l][j] = slope; }
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
    // forward + backward for one case, adding its gradient into g; returns the case's loss
    accumulate(x, y, g, fw) {
      fw = fw || this.forward(x);
      const loss = bceFromLogit(fw.z, y);
      const d0 = this.backDense(fw, fw.p - y, g);
      if (this.conv) {
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
    // A read-only snapshot of one case. All chain-rule factors use the SAME pre-update parameters.
    // upstream = dL/da; slopes = da/dz; delta = dL/dz. Apply g once with applyGradient(g, 1, lr, l2).
    lesson(x, y) {
      const g = this.newGradient(); g.delta = this.hidden.map(h => new Float64Array(h));
      g.upstream = this.hidden.map(h => new Float64Array(h));
      g.slopes = this.hidden.map(h => new Float64Array(h));
      const fw = this.forward(x);
      const loss = this.accumulate(x, y, g, fw);
      const before = { W: this.W.map(w => w.slice()), b: this.b.map(b => b.slice()), Wo: this.Wo.slice(), bo: this.bo };
      return { fw, error: fw.p - y, loss, g, before };
    }

    // d(score z)/d(input): how much each input nudges the score toward the positive class
    inputGradient(x, fw) {
      fw = fw || this.forward(x);
      const d0 = this.backDense(fw, 1, null);
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
        probs[k] = p; loss += bceFromLogit(fw.z, ys[k]);
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

  // finite-difference check of the analytic gradients on a tiny random instance; returns the worst relative error
  function gradientCheck(opts) {
    const net = new Net(Object.assign({ inputSize: 64, imageSize: 8, conv: { K: 2, f: 3, pool: 2 }, hidden: [3, 2], activation: 'sigmoid', seed: 3 }, opts || {}));
    const rnd = mulberry32(11);
    const x = new Float64Array(net.D); for (let i = 0; i < x.length; i++) x[i] = rnd() * 2 - 1;
    const y = 1, eps = 1e-6;
    const lossAt = () => bceFromLogit(net.forward(x).z, y);
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

  return { Net, TinyNet: Net, ACTIVATIONS, mulberry32, fitStandardizer, bce, bceFromLogit, parameterStep, sigmoid, gradientCheck };
});
