/* meme_tensor.cpp - KVM word-space tensors + RoPE sentence encode */

#include "meme_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace phoenix {
namespace secamp {
namespace {

void l2Normalize(std::vector<float> &v) {
  double n2 = 0.0;
  for (float x : v)
    n2 += static_cast<double>(x) * static_cast<double>(x);
  if (n2 <= 1e-18)
    return;
  const float inv = static_cast<float>(1.0 / std::sqrt(n2));
  for (float &x : v)
    x *= inv;
}

int dfOf(const MemeTensorSpace &space, const std::string &w) {
  auto it = space.df.find(w);
  if (it == space.df.end() || it->second <= 0)
    return 0;
  return it->second;
}

void takeTopNeighbors(std::vector<TensorNeighbor> &hits, int cap) {
  std::sort(hits.begin(), hits.end(),
            [](const TensorNeighbor &a, const TensorNeighbor &b) {
              if (a.cosine != b.cosine)
                return a.cosine > b.cosine;
              if (a.kind != b.kind)
                return a.kind < b.kind;
              return a.id < b.id;
            });
  if (cap > 0 && static_cast<int>(hits.size()) > cap)
    hits.resize(static_cast<size_t>(cap));
}

} // namespace

std::vector<float> hashedWordVec(const std::string &word, int dim) {
  const int d = dim > 0 ? dim : kMemeTensorDim;
  std::vector<float> out(static_cast<size_t>(d), 0.0f);
  std::uint32_t h = 2166136261u;
  for (unsigned char c : word) {
    h ^= c;
    h *= 16777619u;
  }
  for (int i = 0; i < d; ++i) {
    h ^= static_cast<std::uint32_t>(i + 1) * 0x9e3779b9u;
    h *= 16777619u;
    out[static_cast<size_t>(i)] =
        static_cast<float>(static_cast<int>(h % 20001u) - 10000) / 10000.0f;
  }
  l2Normalize(out);
  return out;
}

void applyRope(std::vector<float> &vec, int position, double base) {
  if (vec.size() < 2 || position <= 0)
    return;
  if (base < 2.0)
    base = 10000.0;
  const int pairs = static_cast<int>(vec.size()) / 2;
  const double pos = static_cast<double>(position);
  for (int i = 0; i < pairs; ++i) {
    const double expn =
        (2.0 * static_cast<double>(i)) / static_cast<double>(pairs * 2);
    const double theta = pos / std::pow(base, expn);
    const float c = static_cast<float>(std::cos(theta));
    const float s = static_cast<float>(std::sin(theta));
    const float x = vec[static_cast<size_t>(2 * i)];
    const float y = vec[static_cast<size_t>(2 * i + 1)];
    vec[static_cast<size_t>(2 * i)] = x * c - y * s;
    vec[static_cast<size_t>(2 * i + 1)] = x * s + y * c;
  }
}

std::vector<float> encodeSentenceRope(const MemeTensorSpace &space,
                                      const std::vector<std::string> &tokens);

double tensorCosine(const std::vector<float> &a, const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  if (n == 0)
    return 0.0;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double x = static_cast<double>(a[i]);
    const double y = static_cast<double>(b[i]);
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na <= 1e-18 || nb <= 1e-18)
    return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

MemeTensorSpace buildMemeTensorSpace(
    const std::vector<std::string> &ids,
    const std::vector<std::vector<std::string>> &wordsPerId,
    const std::vector<std::vector<double>> &weightPerId) {
  MemeTensorSpace space;
  space.dim = kMemeTensorDim;
  const size_t n = std::min(ids.size(), wordsPerId.size());
  for (size_t i = 0; i < n; ++i) {
    std::unordered_map<std::string, int> seen;
    for (const auto &w : wordsPerId[i]) {
      if (w.empty() || seen[w]++)
        continue;
      space.df[w]++;
    }
  }
  for (const auto &kv : space.df)
    space.wordVecs[kv.first] = hashedWordVec(kv.first, space.dim);
  space.memes.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (ids[i].empty())
      continue;
    MemeTensor row;
    row.id = ids[i];
    row.bag.assign(static_cast<size_t>(space.dim), 0.0f);
    std::unordered_map<std::string, int> seen;
    std::vector<std::string> order;
    const bool haveW = i < weightPerId.size() &&
                       weightPerId[i].size() == wordsPerId[i].size();
    for (size_t j = 0; j < wordsPerId[i].size(); ++j) {
      const auto &w = wordsPerId[i][j];
      if (w.empty())
        continue;
      order.push_back(w);
      if (seen[w]++)
        continue;
      const int df = dfOf(space, w);
      if (df <= 0)
        continue;
      const auto wv = hashedWordVec(w, space.dim);
      const float scale =
          haveW ? static_cast<float>(std::max(0.0, weightPerId[i][j]))
                : 1.0f / static_cast<float>(df);
      if (scale <= 0.0f)
        continue;
      for (int k = 0; k < space.dim; ++k)
        row.bag[static_cast<size_t>(k)] += scale * wv[static_cast<size_t>(k)];
    }
    l2Normalize(row.bag);
    row.words = order;
    if (haveW) {
      std::vector<float> ordered(static_cast<size_t>(space.dim), 0.0f);
      int pos = 0;
      std::unordered_map<std::string, int> used;
      for (size_t j = 0; j < wordsPerId[i].size(); ++j) {
        const auto &w = wordsPerId[i][j];
        if (w.empty() || used[w]++)
          continue;
        ++pos;
        if (dfOf(space, w) <= 0)
          continue;
        auto wv = hashedWordVec(w, space.dim);
        applyRope(wv, pos);
        const float scale = static_cast<float>(std::max(0.0, weightPerId[i][j]));
        if (scale <= 0.0f)
          continue;
        for (int k = 0; k < space.dim; ++k)
          ordered[static_cast<size_t>(k)] += scale * wv[static_cast<size_t>(k)];
      }
      l2Normalize(ordered);
      row.ordered = std::move(ordered);
    } else {
      row.ordered = encodeSentenceRope(space, order);
    }
    space.index[row.id] = space.memes.size();
    space.memes.push_back(std::move(row));
  }
  return space;
}

std::vector<float> encodeBag(const MemeTensorSpace &space,
                             const std::vector<std::string> &tokens) {
  std::vector<float> out(static_cast<size_t>(space.dim), 0.0f);
  std::unordered_map<std::string, int> tf;
  for (const auto &w : tokens) {
    if (!w.empty())
      tf[w]++;
  }
  for (const auto &kv : tf) {
    const int df = dfOf(space, kv.first);
    if (df <= 0)
      continue;
    const auto wv = hashedWordVec(kv.first, space.dim);
    const float scale =
        static_cast<float>(kv.second) / static_cast<float>(df);
    for (int k = 0; k < space.dim; ++k)
      out[static_cast<size_t>(k)] += scale * wv[static_cast<size_t>(k)];
  }
  l2Normalize(out);
  return out;
}

std::vector<float> encodeSentenceRope(const MemeTensorSpace &space,
                                      const std::vector<std::string> &tokens) {
  std::vector<float> out(static_cast<size_t>(space.dim), 0.0f);
  int pos = 0;
  for (const auto &w : tokens) {
    if (w.empty())
      continue;
    ++pos;
    const int df = dfOf(space, w);
    if (df <= 0)
      continue;
    auto wv = hashedWordVec(w, space.dim);
    applyRope(wv, pos);
    const float scale = 1.0f / static_cast<float>(df);
    for (int k = 0; k < space.dim; ++k)
      out[static_cast<size_t>(k)] += scale * wv[static_cast<size_t>(k)];
  }
  l2Normalize(out);
  return out;
}

TensorQuery queryMemeTensor(const MemeTensorSpace &space,
                            const std::vector<std::string> &tokens,
                            const std::string &memeId) {
  TensorQuery q;
  if (space.memes.empty() || tokens.empty())
    return q;
  const auto sent = encodeSentenceRope(space, tokens);
  const auto bag = encodeBag(space, tokens);
  std::vector<std::string> rev(tokens.rbegin(), tokens.rend());
  const auto sentRev = encodeSentenceRope(space, rev);

  std::vector<TensorNeighbor> memeRank;
  memeRank.reserve(space.memes.size());
  std::vector<TensorNeighbor> mixed;
  mixed.reserve(space.memes.size() + space.wordVecs.size());
  double bestSent = -2.0;
  for (const auto &m : space.memes) {
    TensorNeighbor hit;
    hit.id = m.id;
    hit.kind = "meme";
    hit.cosine = tensorCosine(bag, m.bag);
    memeRank.push_back(hit);
    mixed.push_back(hit);
    const double sc = tensorCosine(sent, m.ordered);
    if (sc > bestSent) {
      bestSent = sc;
      q.nearestSentence = m.id;
    }
  }
  std::unordered_map<std::string, int> local;
  for (const auto &w : tokens) {
    if (w.empty() || local[w]++ || !space.wordVecs.count(w))
      continue;
    TensorNeighbor hit;
    hit.id = w;
    hit.kind = "word";
    hit.cosine = tensorCosine(bag, space.wordVecs.at(w));
    mixed.push_back(hit);
    if (q.nearestWord.empty() || hit.cosine > tensorCosine(bag, space.wordVecs.at(q.nearestWord)))
      q.nearestWord = w;
  }
  takeTopNeighbors(memeRank, static_cast<int>(memeRank.size()));
  takeTopNeighbors(mixed, kTensorNeighborCap);
  q.nearest = std::move(mixed);
  auto it = space.index.find(memeId);
  if (it != space.index.end()) {
    const auto &row = space.memes[it->second];
    q.bagCos = tensorCosine(bag, row.bag);
    q.semanticCos = tensorCosine(sent, row.ordered);
    q.reversedCos = tensorCosine(sentRev, row.ordered);
    q.orderLift = q.semanticCos / std::max(std::fabs(q.reversedCos), 1e-12);
    for (size_t i = 0; i < memeRank.size(); ++i) {
      if (memeRank[i].id == memeId) {
        q.tensorRank = static_cast<int>(i);
        break;
      }
    }
  }
  return q;
}

std::vector<TensorNeighbor> tensorNeighborsOf(const MemeTensorSpace &space,
                                              const std::string &memeId,
                                              int cap) {
  std::vector<TensorNeighbor> out;
  auto it = space.index.find(memeId);
  if (it == space.index.end())
    return out;
  const auto &self = space.memes[it->second].bag;
  for (const auto &m : space.memes) {
    if (m.id == memeId)
      continue;
    TensorNeighbor hit;
    hit.id = m.id;
    hit.kind = "meme";
    hit.cosine = tensorCosine(self, m.bag);
    out.push_back(hit);
  }
  takeTopNeighbors(out, cap);
  return out;
}

std::vector<TensorNeighbor> nearestWordsOf(const MemeTensorSpace &space,
                                           const std::string &memeId,
                                           int cap) {
  std::vector<TensorNeighbor> out;
  auto it = space.index.find(memeId);
  if (it == space.index.end())
    return out;
  const auto &row = space.memes[it->second];
  std::unordered_map<std::string, int> seen;
  out.reserve(row.words.size());
  for (const auto &w : row.words) {
    if (w.empty() || seen[w]++)
      continue;
    auto wit = space.wordVecs.find(w);
    if (wit == space.wordVecs.end())
      continue;
    TensorNeighbor hit;
    hit.id = w;
    hit.kind = "word";
    hit.cosine = tensorCosine(row.bag, wit->second);
    out.push_back(hit);
  }
  takeTopNeighbors(out, cap);
  return out;
}

} // namespace secamp
} // namespace phoenix
