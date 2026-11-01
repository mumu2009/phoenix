/* One gradient step on an MLP whose output column can be opened.

   Closed softmax has one logit per class it was given. A new file has
   no column, so it cannot be the argmax. openClass() appends a zero
   column. step() is one update of cross-entropy, step size lr, on
   h = leaky(U x), z = W h. leaky(p) = p when p >= 0, else leak * p.
   The derivative at 0 is 1. This does not admit a meme and does not
   replace the word index.
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace phoenix {
namespace memetic {

struct OpenColumnMlp {
  int hidden{0};
  int vocab{0};
  float leak{0.01f};
  std::vector<float> U;
  std::vector<float> W;

  int classes() const {
    if (hidden <= 0) return 0;
    return static_cast<int>(W.size() / static_cast<size_t>(hidden));
  }

  void init(int h, int v, int c, float leaky = 0.01f) {
    hidden = h;
    vocab = v;
    leak = leaky;
    U.assign(static_cast<size_t>(h * v), 0.0f);
    W.assign(static_cast<size_t>(c * h), 0.0f);
  }

  void openClass() {
    W.insert(W.end(), static_cast<size_t>(hidden), 0.0f);
  }

  float act(float pre) const { return pre >= 0.0f ? pre : leak * pre; }
  float actGrad(float pre) const { return pre >= 0.0f ? 1.0f : leak; }

  void forward(const std::vector<float> &x, std::vector<float> &pre,
               std::vector<float> &h, std::vector<float> &z) const {
    const int width = classes();
    pre.assign(static_cast<size_t>(hidden), 0.0f);
    h.assign(static_cast<size_t>(hidden), 0.0f);
    z.assign(static_cast<size_t>(width), 0.0f);
    const int words = std::min(vocab, static_cast<int>(x.size()));
    for (int i = 0; i < hidden; ++i) {
      float sum = 0.0f;
      for (int v = 0; v < words; ++v) {
        sum += U[static_cast<size_t>(i * vocab + v)] * x[static_cast<size_t>(v)];
      }
      pre[static_cast<size_t>(i)] = sum;
      h[static_cast<size_t>(i)] = act(sum);
    }
    for (int c = 0; c < width; ++c) {
      float sum = 0.0f;
      for (int i = 0; i < hidden; ++i) {
        sum += W[static_cast<size_t>(c * hidden + i)] * h[static_cast<size_t>(i)];
      }
      z[static_cast<size_t>(c)] = sum;
    }
  }

  int argmaxStrict(const std::vector<float> &x) const {
    std::vector<float> pre, h, z;
    forward(x, pre, h, z);
    if (z.empty()) return -1;
    int best = 0;
    int ties = 1;
    for (int c = 1; c < static_cast<int>(z.size()); ++c) {
      if (z[static_cast<size_t>(c)] > z[static_cast<size_t>(best)] + 0.0f) {
        best = c;
        ties = 1;
      } else if (z[static_cast<size_t>(c)] == z[static_cast<size_t>(best)]) {
        ++ties;
      }
    }
    return ties == 1 ? best : -1;
  }

  void step(const std::vector<float> &x, int target, float lr) {
    if (target < 0 || target >= classes()) return;
    std::vector<float> pre, h, z;
    forward(x, pre, h, z);
    const int width = classes();
    float peak = z[0];
    for (float value : z) {
      if (value > peak) peak = value;
    }
    float sum = 0.0f;
    std::vector<float> soft(static_cast<size_t>(width), 0.0f);
    for (int c = 0; c < width; ++c) {
      soft[static_cast<size_t>(c)] = std::exp(z[static_cast<size_t>(c)] - peak);
      sum += soft[static_cast<size_t>(c)];
    }
    if (!(sum > 0.0f)) return;
    std::vector<float> dz(static_cast<size_t>(width), 0.0f);
    for (int c = 0; c < width; ++c) {
      soft[static_cast<size_t>(c)] /= sum;
      dz[static_cast<size_t>(c)] =
          soft[static_cast<size_t>(c)] - (c == target ? 1.0f : 0.0f);
    }
    std::vector<float> dh(static_cast<size_t>(hidden), 0.0f);
    for (int i = 0; i < hidden; ++i) {
      float grad = 0.0f;
      for (int c = 0; c < width; ++c) {
        grad += W[static_cast<size_t>(c * hidden + i)] * dz[static_cast<size_t>(c)];
      }
      dh[static_cast<size_t>(i)] = grad;
    }
    for (int c = 0; c < width; ++c) {
      for (int i = 0; i < hidden; ++i) {
        W[static_cast<size_t>(c * hidden + i)] -=
            lr * dz[static_cast<size_t>(c)] * h[static_cast<size_t>(i)];
      }
    }
    const int words = std::min(vocab, static_cast<int>(x.size()));
    for (int i = 0; i < hidden; ++i) {
      const float grad = dh[static_cast<size_t>(i)] * actGrad(pre[static_cast<size_t>(i)]);
      for (int v = 0; v < words; ++v) {
        U[static_cast<size_t>(i * vocab + v)] -=
            lr * grad * x[static_cast<size_t>(v)];
      }
    }
  }
};

}  // namespace memetic
}  // namespace phoenix
