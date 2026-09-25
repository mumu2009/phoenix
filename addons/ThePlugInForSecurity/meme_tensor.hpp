/* meme_tensor.hpp - KVM word-space tensors + RoPE sentence encode

   GNN find/detect stays official. This is an extra geometry on the same
   meme↔word map: each meme is an ordered (word, α) point in a hashed
   word space (α = tf/df, L1-normalized when weights are supplied);
   a sentence is the same points rotated by token index (RoPE).
   Does not call llama. Not a construct/deploy surface. */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phoenix {
namespace secamp {

inline constexpr int kMemeTensorDim = 32;
inline constexpr int kTensorNeighborCap = 5;

struct MemeTensor {
  std::string id;
  std::vector<std::string> words;
  std::vector<float> bag;
  std::vector<float> ordered;
};

struct TensorNeighbor {
  std::string id;
  std::string kind; /* meme | word | sentence */
  double cosine{0.0};
};

struct MemeTensorSpace {
  int dim{kMemeTensorDim};
  std::unordered_map<std::string, int> df;
  std::unordered_map<std::string, std::vector<float>> wordVecs;
  std::vector<MemeTensor> memes;
  std::unordered_map<std::string, size_t> index;
};

struct TensorQuery {
  double semanticCos{0.0};
  double bagCos{0.0};
  double reversedCos{0.0};
  double orderLift{0.0};
  int tensorRank{-1};
  std::string nearestWord;
  std::string nearestSentence;
  std::vector<TensorNeighbor> nearest;
};

std::vector<float> hashedWordVec(const std::string &word, int dim);
void applyRope(std::vector<float> &vec, int position, double base = 10000.0);
double tensorCosine(const std::vector<float> &a, const std::vector<float> &b);

MemeTensorSpace buildMemeTensorSpace(
    const std::vector<std::string> &ids,
    const std::vector<std::vector<std::string>> &wordsPerId,
    const std::vector<std::vector<double>> &weightPerId = {});

std::vector<float> encodeBag(const MemeTensorSpace &space,
                             const std::vector<std::string> &tokens);
std::vector<float> encodeSentenceRope(const MemeTensorSpace &space,
                                      const std::vector<std::string> &tokens);

TensorQuery queryMemeTensor(const MemeTensorSpace &space,
                            const std::vector<std::string> &tokens,
                            const std::string &memeId);

std::vector<TensorNeighbor> tensorNeighborsOf(const MemeTensorSpace &space,
                                              const std::string &memeId,
                                              int cap = kTensorNeighborCap);
std::vector<TensorNeighbor> nearestWordsOf(const MemeTensorSpace &space,
                                           const std::string &memeId,
                                           int cap = kTensorNeighborCap);

} // namespace secamp
} // namespace phoenix
