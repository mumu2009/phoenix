/* open_column_router.hpp - a file router whose output column can open.

   open_column_step.hpp proves one thing: a closed softmax cannot give a
   file it never saw a logit, and appending a zero column plus one
   cross-entropy step lets a new word bag point at the new column while
   the old bags keep pointing at their old files. That header is not a
   router. This one is.

   Architecture (one hidden layer, one trained output layer):
     h     = phi(U x)          U is FROZEN. Each known word maps to one
                               hidden unit (a binary hash projection), so
                               h is the union of the hidden units of the
                               words in the bag. This keeps the first
                               layer fixed and cheap: a new word only
                               extends U by one zero column and one 1.
     z     = W h               W is the trained layer.
     route = strict argmax z with a positive margin, else no file.

   Training is one associative write per file, not an epoch: addFile()
   opens the file's column and adds 1 to the output weights of the
   hidden units its bag activates (W += y h^T with y one-hot). That is
   the same one-step outer-product rule as the witnessed gloss.

   Why not the cross-entropy step here: with one class the softmax is
   already 1, so the gradient is zero and the first file's column can
   never leave zero. With two classes the step would also push the older
   columns' weights down. The Hebbian write has neither problem, and it
   is still one step.

   Honesty note: this router memorizes the bags it is given. It does not
   beat the index and it is not an admission test. It is used only when
   the lexical index, the side table and the conditional model all
   returned no file, and it returns a path - never the file body. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open_column_step.hpp"

namespace phoenix {
namespace memetic {

struct OpenColumnRoute {
  int index{-1};
  float margin{0.0f};
  float best{0.0f};
  int knownTokens{0};
};

class OpenColumnRouter {
 public:
  void init(int hidden = 64, float leak = 0.01f) {
    hidden_ = hidden > 0 ? hidden : 64;
    leak_ = leak;
    mlp_.init(hidden_, 0, 0, leak_);
    vocab_.clear();
    ids_.clear();
    classIds_.clear();
    bags_.clear();
    used_ = 0;
    background_ = false;
    growCapacity(64);
  }

  /* Open the null/background class (class 0).  A one-class softmax has a
     zero gradient; with a background class the target probability is
     below 1 even for the first file, so one CE step trains that column.
     The background has no positive samples: every step pushes its logit
     down, and an unseen bag (all logits 0) stays a tie -> no file. */
  void enableBackground() {
    if (background_)
      return;
    mlp_.openClass();
    classIds_.push_back(std::string());
    trainBags_.push_back({});
    background_ = true;
  }

  bool hasBackground() const { return background_; }

  int classCount() const { return mlp_.classes(); }
  int vocabSize() const { return used_; }
  int hidden() const { return hidden_; }
  const std::vector<float> &rawWeights() const { return mlp_.W; }

  std::string fileId(int index) const {
    if (index < 0 || index >= static_cast<int>(classIds_.size()))
      return std::string();
    return classIds_[index];
  }

  /* Add one file. The bag is the set of tokens. Returns the new class
     index (or -1 when the bag has no known token). */
  int addFile(const std::string &id, const std::vector<std::string> &tokens,
              float lr = 1.0f) {
    if (id.empty())
      return -1;
    std::vector<int> columns;
    std::unordered_set<int> seen;
    for (const auto &w : tokens) {
      if (w.size() < 3)
        continue;
      const int v = intern(w);
      if (v < 0)
        continue;
      if (!seen.insert(v).second)
        continue;
      columns.push_back(v);
    }
    if (columns.empty())
      return -1;
    std::vector<float> x(static_cast<size_t>(mlp_.vocab), 0.0f);
    for (int v : columns)
      x[static_cast<size_t>(v)] = 1.0f;
    mlp_.openClass();
    const int cls = mlp_.classes() - 1;
    writeOutput(x, cls, lr);
    std::vector<int> bag = columns;
    std::sort(bag.begin(), bag.end());
    bag.erase(std::unique(bag.begin(), bag.end()), bag.end());
    trainBags_.push_back(bag);
    ids_.push_back(id);
    classIds_.push_back(id);
    bags_.push_back(x);
    return cls;
  }

  /* One cross-entropy step on the whole MLP (the header's step updates
     both the hidden layer U and the output W).  Requires the background
     class so even the first file has a non-zero gradient.  One step per
     file; no epoch. */
  int addFileCe(const std::string &id,
                const std::vector<std::string> &tokens, float lr = 1.0f) {
    if (id.empty())
      return -1;
    enableBackground();
    std::vector<int> columns;
    std::unordered_set<int> seen;
    for (const auto &w : tokens) {
      if (w.size() < 3)
        continue;
      const int v = intern(w);
      if (v < 0)
        continue;
      if (!seen.insert(v).second)
        continue;
      columns.push_back(v);
    }
    if (columns.empty())
      return -1;
    std::vector<float> x(static_cast<size_t>(mlp_.vocab), 0.0f);
    for (int v : columns)
      x[static_cast<size_t>(v)] = 1.0f;
    mlp_.openClass();
    const int cls = mlp_.classes() - 1;
    mlp_.step(x, cls, lr);
    std::vector<int> bag = columns;
    std::sort(bag.begin(), bag.end());
    bag.erase(std::unique(bag.begin(), bag.end()), bag.end());
    trainBags_.push_back(bag);
    ids_.push_back(id);
    classIds_.push_back(id);
    bags_.push_back(x);
    return cls;
  }

  /* One associative write: for every hidden unit the bag activates, add
     lr to the file's output weight. No epoch, no other column touched. */
  void writeOutput(const std::vector<float> &x, int target, float lr = 1.0f) {
    const int width = mlp_.classes();
    if (target < 0 || target >= width || width <= 0)
      return;
    std::vector<float> pre, h, z;
    mlp_.forward(x, pre, h, z);
    for (int i = 0; i < mlp_.hidden; ++i) {
      if (h[static_cast<size_t>(i)] == 0.0f)
        continue;
      mlp_.W[static_cast<size_t>(target * mlp_.hidden + i)] +=
          lr * h[static_cast<size_t>(i)];
    }
  }

  OpenColumnRoute route(const std::vector<std::string> &tokens) const {
    OpenColumnRoute out;
    if (mlp_.classes() == 0 || mlp_.vocab == 0)
      return out;
    std::vector<float> x(static_cast<size_t>(mlp_.vocab), 0.0f);
    std::unordered_set<int> seen;
    for (const auto &w : tokens) {
      if (w.size() < 3)
        continue;
      auto it = vocab_.find(w);
      if (it == vocab_.end())
        continue;
      if (!seen.insert(it->second).second)
        continue;
      x[static_cast<size_t>(it->second)] = 1.0f;
    }
    out.knownTokens = static_cast<int>(seen.size());
    if (out.knownTokens == 0)
      return out;
    /* Structural tie: if the query's known words exactly equal the
       training bags of two or more files, the evidence is identical.
       CE's softmax scales a column's step by the class count, so the
       scores need not be exactly equal; the router still refuses. */
    std::vector<int> queryBag;
    for (int v : seen)
      queryBag.push_back(v);
    std::sort(queryBag.begin(), queryBag.end());
    int bagMatches = 0;
    for (const auto &bag : trainBags_) {
      if (!bag.empty() && bag == queryBag) {
        ++bagMatches;
        if (bagMatches >= 2)
          return out;
      }
    }
    std::vector<float> pre, h, z;
    mlp_.forward(x, pre, h, z);
    if (z.empty())
      return out;
    int best = 0;
    int ties = 1;
    for (int c = 1; c < static_cast<int>(z.size()); ++c) {
      if (z[static_cast<size_t>(c)] > z[static_cast<size_t>(best)] + 1e-9f) {
        best = c;
        ties = 1;
      } else if (std::fabs(z[static_cast<size_t>(c)] -
                           z[static_cast<size_t>(best)]) <= 1e-9f) {
        ++ties;
      }
    }
    if (ties != 1)
      return out;
    float second = 0.0f;
    bool haveSecond = false;
    for (int c = 0; c < static_cast<int>(z.size()); ++c) {
      if (c == best)
        continue;
      if (!haveSecond || z[static_cast<size_t>(c)] > second) {
        second = z[static_cast<size_t>(c)];
        haveSecond = true;
      }
    }
    out.best = z[static_cast<size_t>(best)];
    out.margin = haveSecond ? out.best - second : out.best;
    /* A CE step that trains the hidden layer can leave two identical
       bags a hair apart. Near-equal columns are a tie, not a coin
       flip: the router only names a file when the margin is clear. */
    constexpr float kRouteMinMargin = 1e-3f;
    if (!(out.best > 1e-9f) || !(out.margin > kRouteMinMargin))
      return out;
    if (best < static_cast<int>(classIds_.size()) &&
        classIds_[static_cast<size_t>(best)].empty())
      return out; /* background/null class is not a file */
    out.index = best;
    return out;
  }

 private:
  int intern(const std::string &w) {
    auto it = vocab_.find(w);
    if (it != vocab_.end())
      return it->second;
    if (used_ >= mlp_.vocab)
      growCapacity(std::max(64, mlp_.vocab * 2));
    const int v = used_++;
    vocab_[w] = v;
    /* One hidden unit per vocabulary slot, assigned by index rather than
       by hash: when the width is at least the vocabulary size this is
       injective, so two files' distinctive words never share a unit.
       The caller sizes the router from the union of its bags. */
    const int unit = v % mlp_.hidden;
    mlp_.U[static_cast<size_t>(unit * mlp_.vocab + v)] = 1.0f;
    return v;
  }

  /* Grow the frozen embedding in doubling steps. Rebuilding keeps U
     frozen: only zero columns are appended. */
  void growCapacity(int newVocab) {
    if (newVocab <= mlp_.vocab)
      return;
    const int oldVocab = mlp_.vocab;
    std::vector<float> grown(static_cast<size_t>(mlp_.hidden) *
                                 static_cast<size_t>(newVocab),
                             0.0f);
    for (int i = 0; i < mlp_.hidden; ++i) {
      for (int v = 0; v < oldVocab; ++v) {
        grown[static_cast<size_t>(i * newVocab + v)] =
            mlp_.U[static_cast<size_t>(i * oldVocab + v)];
      }
    }
    mlp_.U.swap(grown);
    mlp_.vocab = newVocab;
  }

  OpenColumnMlp mlp_;
  int hidden_{64};
  float leak_{0.01f};
  int used_{0};
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> ids_;
  std::vector<std::string> classIds_; /* class -> file id; "" = background */
  std::vector<std::vector<int>> trainBags_;
  std::vector<std::vector<float>> bags_;
  bool background_{false};
};

}  // namespace memetic
}  // namespace phoenix
