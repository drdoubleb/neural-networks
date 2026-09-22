/*
 * nn.js — a tiny fully-connected network with hand-written backpropagation.
 *   inputs (D) -> optional hidden layer (H units, sigmoid/tanh/ReLU) -> 1 output (sigmoid) = P(irregular)
 * Trained with plain mini-batch gradient descent on binary cross-entropy.
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
    sigmoid: { label: 'Sigmoid', f: z => 1 / (1 + Math.exp(-z)), df: (z, a) => a * (1 - a), range: [0, 1] },
    tanh:    { label: 'Tanh',    f: z => Math.tanh(z),           df: (z, a) => 1 - a * a,   range: [-1, 1] },
    relu:    { label: 'ReLU',    f: z => (z > 0 ? z : 0),        df: (z, a) => (z > 0 ? 1 : 0), range: [0, Infinity] },
  };
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const EPS = 1e-7;
  const bce = (p, y) => -(y * Math.log(p + EPS) + (1 - y) * Math.log(1 - p + EPS));

  class TinyNet {
    constructor({ inputSize, hidden = 0, activation = 'sigmoid', seed = 1 }) {
      this.D = inputSize;
      this.H = hidden;
      this.activation = activation;
      this.seed = seed;
      this.init();
    }
    init() {
      const rnd = mulberry32(this.seed * 7919 + 17);
      const u = s => (rnd() * 2 - 1) * s;
      const D = this.D, H = this.H;
      if (H > 0) {
        const s1 = Math.sqrt(6 / (D + H));
        this.W1 = new Float64Array(H * D); for (let i = 0; i < H * D; i++) this.W1[i] = u(s1);
        this.b1 = new Float64Array(H);
        const s2 = Math.sqrt(6 / (H + 1));
        this.W2 = new Float64Array(H); for (let j = 0; j < H; j++) this.W2[j] = u(s2);
      } else {
        this.W1 = null; this.b1 = null;
        const s = Math.sqrt(6 / (D + 1));
        this.W2 = new Float64Array(D); for (let i = 0; i < D; i++) this.W2[i] = u(s);
      }
      this.b2 = 0;
      this.steps = 0;
    }
    forward(x) {
      const D = this.D, H = this.H;
      let z = this.b2;
      let pre = null, h = null;
      if (H > 0) {
        const act = ACTIVATIONS[this.activation];
        pre = new Float64Array(H); h = new Float64Array(H);
        for (let j = 0; j < H; j++) {
          let s = this.b1[j]; const off = j * D;
          for (let i = 0; i < D; i++) s += this.W1[off + i] * x[i];
          pre[j] = s; h[j] = act.f(s);
          z += this.W2[j] * h[j];
        }
      } else {
        for (let i = 0; i < D; i++) z += this.W2[i] * x[i];
      }
      const p = sigmoid(z);
      return { pre, h, z, p };
    }
    predict(x) { return this.forward(x).p; }

    // one gradient-descent step on a mini-batch (l2 = weight decay strength); returns the mean loss before the update
    trainBatch(xs, ys, lr, l2 = 0) {
      const D = this.D, H = this.H, n = xs.length;
      let loss = 0;
      if (H > 0) {
        const act = ACTIVATIONS[this.activation];
        const gW1 = new Float64Array(H * D), gb1 = new Float64Array(H), gW2 = new Float64Array(H);
        let gb2 = 0;
        for (let k = 0; k < n; k++) {
          const x = xs[k], y = ys[k];
          const fw = this.forward(x);
          loss += bce(fw.p, y);
          const dz = fw.p - y;                    // dLoss/dz for sigmoid + cross-entropy
          gb2 += dz;
          for (let j = 0; j < H; j++) {
            gW2[j] += dz * fw.h[j];
            const dpre = dz * this.W2[j] * act.df(fw.pre[j], fw.h[j]);
            gb1[j] += dpre;
            const off = j * D;
            for (let i = 0; i < D; i++) gW1[off + i] += dpre * x[i];
          }
        }
        const s = lr / n, decay = 1 - lr * l2;
        for (let i = 0; i < H * D; i++) this.W1[i] = this.W1[i] * decay - s * gW1[i];
        for (let j = 0; j < H; j++) { this.b1[j] -= s * gb1[j]; this.W2[j] = this.W2[j] * decay - s * gW2[j]; }
        this.b2 -= s * gb2;
      } else {
        const gW = new Float64Array(D);
        let gb = 0;
        for (let k = 0; k < n; k++) {
          const x = xs[k], y = ys[k];
          const fw = this.forward(x);
          loss += bce(fw.p, y);
          const dz = fw.p - y;
          gb += dz;
          for (let i = 0; i < D; i++) gW[i] += dz * x[i];
        }
        const s = lr / n, decay = 1 - lr * l2;
        for (let i = 0; i < D; i++) this.W2[i] = this.W2[i] * decay - s * gW[i];
        this.b2 -= s * gb;
      }
      this.steps++;
      return loss / n;
    }

    // d(score z)/d(input) — how much each input nudges the score toward "irregular"
    inputGradient(x, fw) {
      const D = this.D, H = this.H;
      const g = new Float64Array(D);
      if (H > 0) {
        fw = fw || this.forward(x);
        const act = ACTIVATIONS[this.activation];
        for (let j = 0; j < H; j++) {
          const c = this.W2[j] * act.df(fw.pre[j], fw.h[j]);
          if (c === 0) continue;
          const off = j * D;
          for (let i = 0; i < D; i++) g[i] += c * this.W1[off + i];
        }
      } else {
        g.set(this.W2);
      }
      return g;
    }

    evaluate(xs, ys, threshold = 0.5) {
      let loss = 0, correct = 0;
      const probs = new Float64Array(xs.length);
      for (let k = 0; k < xs.length; k++) {
        const p = this.predict(xs[k]);
        probs[k] = p;
        loss += bce(p, ys[k]);
        if ((p >= threshold ? 1 : 0) === ys[k]) correct++;
      }
      return { loss: loss / xs.length, accuracy: correct / xs.length, probs };
    }
    parameterCount() { return this.H > 0 ? this.H * this.D + this.H + this.H + 1 : this.D + 1; }
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

  return { TinyNet, ACTIVATIONS, mulberry32, fitStandardizer, bce, sigmoid };
});
