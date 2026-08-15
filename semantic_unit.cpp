/* semantic_unit.cpp - Multimodal semantic unit implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#include "semantic_unit.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

#include "async_task_system.hpp"

namespace phoenix {
namespace multimodal {

namespace {

/* Deterministic projection matrix cache keyed by (sourceDim, targetDim, seed, nonZeros). */
struct ProjectionKey {
  size_t sourceDim;
  size_t targetDim;
  unsigned int seed;
  size_t nonZeros;
  bool operator==(const ProjectionKey &o) const noexcept {
    return sourceDim == o.sourceDim && targetDim == o.targetDim &&
           seed == o.seed && nonZeros == o.nonZeros;
  }
};

struct ProjectionKeyHash {
  std::size_t operator()(const ProjectionKey &k) const noexcept {
    return std::hash<std::size_t>{}(k.sourceDim) ^
           (std::hash<std::size_t>{}(k.targetDim) << 1) ^
           (std::hash<unsigned int>{}(k.seed) << 2) ^
           (std::hash<std::size_t>{}(k.nonZeros) << 3);
  }
};

std::mutex gProjectionCacheMu;
std::unordered_map<ProjectionKey, Eigen::MatrixXf, ProjectionKeyHash> gProjectionCache;
std::unordered_map<ProjectionKey, Eigen::SparseMatrix<float>, ProjectionKeyHash> gSparseProjectionCache;

const Eigen::MatrixXf &getProjectionMatrix(size_t sourceDim,
                                           size_t targetDim,
                                           unsigned int seed) {
  ProjectionKey key{sourceDim, targetDim, seed, 0};
  {
    std::lock_guard<std::mutex> lock(gProjectionCacheMu);
    auto it = gProjectionCache.find(key);
    if (it != gProjectionCache.end()) {
      return it->second;
    }
  }

  // Generation is deterministic given (sourceDim, targetDim, seed), so it is
  // safe to build the matrix outside the lock; if two threads race on the
  // same brand-new key they simply compute the same matrix twice and only
  // one copy is kept in the cache.
  std::mt19937 rng(static_cast<unsigned int>(sourceDim + targetDim * 1315423911u + seed));
  std::normal_distribution<float> dist(0.0f,
                                       std::sqrt(2.0f / static_cast<float>(sourceDim + targetDim)));
  Eigen::MatrixXf mat(static_cast<Eigen::Index>(targetDim),
                      static_cast<Eigen::Index>(sourceDim));
  for (Eigen::Index r = 0; r < mat.rows(); ++r) {
    for (Eigen::Index c = 0; c < mat.cols(); ++c) {
      mat(r, c) = dist(rng);
    }
  }

  std::lock_guard<std::mutex> lock(gProjectionCacheMu);
  auto it = gProjectionCache.find(key);
  if (it != gProjectionCache.end()) {
    return it->second;  // another thread already inserted the same matrix
  }
  auto &ref = gProjectionCache[key];
  ref = std::move(mat);
  return ref;
}

const Eigen::SparseMatrix<float> &getSparseProjectionMatrix(size_t sourceDim,
                                                            size_t targetDim,
                                                            size_t nonZeros,
                                                            unsigned int seed) {
  ProjectionKey key{sourceDim, targetDim, seed, nonZeros};
  {
    std::lock_guard<std::mutex> lock(gProjectionCacheMu);
    auto it = gSparseProjectionCache.find(key);
    if (it != gSparseProjectionCache.end()) {
      return it->second;
    }
  }

  // Sparse Johnson-Lindenstrauss projection (Achlioptas-style): each source
  // dimension is projected to `nonZeros` distinct target rows, each row value
  // is +/- 1/sqrt(nonZeros).  This preserves the L2 norm of a one-hot vector
  // exactly and preserves pairwise distances with high probability for larger
  // unit vectors, while reducing the per-projection cost from O(source*target)
  // to O(source*nonZeros).
  const float scale = 1.0f / std::sqrt(static_cast<float>(nonZeros));
  std::mt19937 rng(static_cast<unsigned int>(sourceDim + targetDim * 1315423911u +
                                            seed + nonZeros * 2654435761u));
  std::vector<int> rows(targetDim);
  std::iota(rows.begin(), rows.end(), 0);
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.reserve(sourceDim * nonZeros);
  for (size_t c = 0; c < sourceDim; ++c) {
    std::shuffle(rows.begin(), rows.end(), rng);
    for (size_t i = 0; i < nonZeros; ++i) {
      const int r = rows[i];
      const float value = (rng() & 1u) ? scale : -scale;
      triplets.emplace_back(r, static_cast<int>(c), value);
    }
  }

  Eigen::SparseMatrix<float> mat(static_cast<Eigen::Index>(targetDim),
                                 static_cast<Eigen::Index>(sourceDim));
  mat.setFromTriplets(triplets.begin(), triplets.end());

  std::lock_guard<std::mutex> lock(gProjectionCacheMu);
  auto it = gSparseProjectionCache.find(key);
  if (it != gSparseProjectionCache.end()) {
    return it->second;
  }
  auto &ref = gSparseProjectionCache[key];
  ref = std::move(mat);
  return ref;
}

} // namespace

std::string generateSemanticId(const std::string &content, uint64_t seed) {
  std::hash<std::string> hasher;
  uint64_t v = hasher(content) ^ seed;
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << v;
  return oss.str();
}

std::string modalityToString(Modality m) {
  switch (m) {
    case Modality::Text: return "text";
    case Modality::Image: return "image";
    case Modality::Audio: return "audio";
    case Modality::Video: return "video";
    case Modality::Sensor: return "sensor";
    case Modality::Structured: return "structured";
  }
  return "text";
}

Modality modalityFromString(const std::string &s) {
  std::string lowered;
  lowered.reserve(s.size());
  for (char ch : s) {
    lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  if (lowered == "image" || lowered == "img") return Modality::Image;
  if (lowered == "audio" || lowered == "sound") return Modality::Audio;
  if (lowered == "video") return Modality::Video;
  if (lowered == "sensor") return Modality::Sensor;
  if (lowered == "structured" || lowered == "json" || lowered == "table") {
    return Modality::Structured;
  }
  return Modality::Text;
}

nlohmann::json SemanticUnit::toJson() const {
  nlohmann::json j;
  j["id"] = id;
  j["modality"] = modalityToString(modality);
  j["semanticVector"] = semanticVector;
  j["unitQuerySequence"] = unitQuerySequence;
  j["content"] = content;
  j["confidence"] = confidence;
  j["timestampMs"] = timestampMs;
  j["metadata"] = metadata;
  j["associationIds"] = associationIds;
  j["modalWeights"] = modalWeights;
  return j;
}

SemanticUnit SemanticUnit::fromJson(const nlohmann::json &j) {
  SemanticUnit u;
  if (!j.is_object()) {
    return u;
  }
  if (j.contains("modality") && j["modality"].is_string()) {
    u.modality = modalityFromString(j["modality"].get<std::string>());
  }
  if (j.contains("semanticVector") && j["semanticVector"].is_array()) {
    u.semanticVector = j["semanticVector"].get<std::vector<float>>();
  }
  if (j.contains("content") && j["content"].is_string()) {
    u.content = j["content"].get<std::string>();
  }
  if (j.contains("confidence") && j["confidence"].is_number()) {
    u.confidence = j["confidence"].get<float>();
  }
  if (j.contains("timestampMs") && j["timestampMs"].is_number()) {
    u.timestampMs = j["timestampMs"].get<uint64_t>();
  }
  if (j.contains("metadata") && j["metadata"].is_object()) {
    u.metadata = j["metadata"].get<std::map<std::string, std::string>>();
  }
  if (j.contains("associationIds") && j["associationIds"].is_array()) {
    u.associationIds = j["associationIds"].get<std::vector<std::string>>();
  }
  if (j.contains("id") && j["id"].is_string()) {
    u.id = j["id"].get<std::string>();
  } else {
    u.id = generateSemanticId();
  }
  if (j.contains("modalWeights") && j["modalWeights"].is_object()) {
    for (auto it = j["modalWeights"].begin(); it != j["modalWeights"].end(); ++it) {
      if (it.value().is_number()) {
        u.modalWeights[it.key()] = it.value().get<float>();
      }
    }
  }
  return u;
}

SemanticUnit::SemanticUnit() : id(generateSemanticId()) {}

std::vector<float> normalizeVector(const std::vector<float> &v) {
  double sum2 = 0.0;
  for (float x : v) {
    sum2 += static_cast<double>(x) * static_cast<double>(x);
  }
  if (sum2 == 0.0) {
    return v;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(sum2));
  std::vector<float> out;
  out.reserve(v.size());
  for (float x : v) {
    out.push_back(x * inv);
  }
  return out;
}

float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0f;
  }
  const size_t n = a.size();
  double dot = 0.0;
  double sum2A = 0.0;
  double sum2B = 0.0;
  for (size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    sum2A += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    sum2B += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  const double denom = std::sqrt(sum2A * sum2B);
  if (denom == 0.0) {
    return 0.0f;
  }
  return static_cast<float>(dot / denom);
}

std::vector<float> projectToDimensionSparse(const std::vector<float> &v,
                                            size_t targetDim,
                                            size_t nonZerosPerColumn,
                                            unsigned int seed) {
  if (targetDim == 0 || targetDim == v.size() || v.empty()) {
    return v;
  }
  if (nonZerosPerColumn == 0) {
    nonZerosPerColumn = std::max<size_t>(3, targetDim / 3);
  }
  nonZerosPerColumn = std::min(nonZerosPerColumn, targetDim);
  if (nonZerosPerColumn == 0) {
    return v;
  }

  const Eigen::Index sourceDim = static_cast<Eigen::Index>(v.size());
  const Eigen::Index outDim = static_cast<Eigen::Index>(targetDim);
  const Eigen::SparseMatrix<float> &mat =
      getSparseProjectionMatrix(v.size(), targetDim, nonZerosPerColumn, seed);

  Eigen::VectorXf in(sourceDim);
  for (Eigen::Index i = 0; i < sourceDim; ++i) {
    in(i) = v[static_cast<size_t>(i)];
  }

  Eigen::VectorXf out = mat * in;
  std::vector<float> result;
  result.reserve(targetDim);
  for (Eigen::Index i = 0; i < outDim; ++i) {
    result.push_back(out(i));
  }
  return result;
}

std::vector<float> projectToDimension(const std::vector<float> &v,
                                      size_t targetDim,
                                      unsigned int seed) {
  if (targetDim == 0 || targetDim == v.size() || v.empty()) {
    return v;
  }

  // For large source*target projections use the sparse JL transform to reduce
  // the per-call cost from O(source*target) to O(source*nonZeros), while
  // keeping the dense path for small dimensions where it is faster and where
  // existing unit tests depend on the exact dense distribution.
  constexpr size_t kSparseJlThreshold = 4096;
  if (v.size() * targetDim > kSparseJlThreshold) {
    return projectToDimensionSparse(v, targetDim, 0, seed);
  }

  const Eigen::Index sourceDim = static_cast<Eigen::Index>(v.size());
  const Eigen::Index outDim = static_cast<Eigen::Index>(targetDim);
  const Eigen::MatrixXf &mat = getProjectionMatrix(v.size(), targetDim, seed);

  Eigen::VectorXf in(sourceDim);
  for (Eigen::Index i = 0; i < sourceDim; ++i) {
    in(i) = v[static_cast<size_t>(i)];
  }

  Eigen::VectorXf out = mat * in;
  std::vector<float> result;
  result.reserve(targetDim);
  for (Eigen::Index i = 0; i < outDim; ++i) {
    result.push_back(out(i));
  }
  return result;
}

namespace {

size_t chooseTargetDim(const SemanticUnit &a, const SemanticUnit &b, size_t targetDim) {
  if (targetDim != 0) {
    return targetDim;
  }
  return std::max(a.semanticVector.size(), b.semanticVector.size());
}

SemanticUnit fuseImpl(const SemanticUnit &a,
                      const SemanticUnit &b,
                      size_t targetDim,
                      bool multiply) {
  const size_t dim = chooseTargetDim(a, b, targetDim);
  std::vector<float> va = projectToDimension(a.semanticVector, dim);
  std::vector<float> vb = projectToDimension(b.semanticVector, dim);

  SemanticUnit out;
  out.modality = a.confidence >= b.confidence ? a.modality : b.modality;
  out.confidence = (a.confidence + b.confidence) * 0.5f;
  out.timestampMs = std::max(a.timestampMs, b.timestampMs);

  for (const auto &kv : a.modalWeights) out.modalWeights[kv.first] += kv.second;
  for (const auto &kv : b.modalWeights) out.modalWeights[kv.first] += kv.second;
  if (!out.modalWeights.empty()) {
    for (auto &kv : out.modalWeights) kv.second *= 0.5f;
  } else {
    out.modalWeights[modalityToString(out.modality)] = 1.0f;
  }

  out.semanticVector.resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    if (multiply) {
      out.semanticVector[i] = va[i] * vb[i];
    } else {
      out.semanticVector[i] = va[i] + vb[i];
    }
  }
  out.semanticVector = normalizeVector(out.semanticVector);

  if (!a.content.empty() && !b.content.empty()) {
    out.content = a.content + " | " + b.content;
  } else {
    out.content = a.content.empty() ? b.content : a.content;
  }

  for (const auto &kv : a.metadata) out.metadata[kv.first] = kv.second;
  for (const auto &kv : b.metadata) out.metadata[kv.first] = kv.second;
  out.metadata["fusion"] = multiply ? "multiply" : "add";
  return out;
}

} // namespace

SemanticUnit fuseAdd(const SemanticUnit &a,
                     const SemanticUnit &b,
                     size_t targetDim) {
  return fuseImpl(a, b, targetDim, false);
}

SemanticUnit fuseMultiply(const SemanticUnit &a,
                        const SemanticUnit &b,
                        size_t targetDim) {
  return fuseImpl(a, b, targetDim, true);
}

static SemanticUnit fuseAttention(const SemanticUnit &query,
                                  const std::vector<SemanticUnit> &units,
                                  size_t dim,
                                  unsigned int seed) {
  if (units.empty()) {
    return query;
  }

  const size_t targetDim = dim != 0
                               ? dim
                               : std::max(query.semanticVector.size(),
                                          std::max_element(
                                              units.begin(), units.end(),
                                              [](const SemanticUnit &x, const SemanticUnit &y) {
                                                return x.semanticVector.size() < y.semanticVector.size();
                                              })->semanticVector.size());

  std::vector<float> q = projectToDimension(query.semanticVector, targetDim, seed);
  q = normalizeVector(q);

  // Memoize each unit's projected-and-normalised vector so the aggregation
  // loop below reuses it instead of re-projecting.  This halves the dominant
  // O(K * D_in * D_out) projection cost on the hot path (a "JIT-like" single
  // materialisation per unit).
  std::vector<float> weights(units.size());
  std::vector<std::vector<float>> projected(units.size());

  // Per-unit projection and scoring is embarrassingly parallel.
  auto projectAndScore = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      std::vector<float> uv =
          normalizeVector(projectToDimension(units[i].semanticVector, targetDim, seed));
      weights[i] = cosineSimilarity(q, uv);
      projected[i] = std::move(uv);
    }
  };

  const size_t workerCount = std::max<size_t>(1, std::thread::hardware_concurrency());
  const size_t chunkCount = std::min(units.size(), workerCount);

  if (units.size() > 8 && chunkCount > 1) {
    auto &ats = phoenix::v7::AsyncTaskSystem::global();
    ats.start();

    std::vector<std::future<void>> futures;
    futures.reserve(chunkCount);

    for (size_t c = 0; c < chunkCount; ++c) {
      const size_t start = c * units.size() / chunkCount;
      const size_t end = (c + 1) * units.size() / chunkCount;
      std::future<void> f = ats.submitWithFuture(
          phoenix::v7::TaskModule::Encoder,
          phoenix::v7::TaskPriority::High,
          [&, start, end]() { projectAndScore(start, end); },
          "fuseAttention_chunk");
      futures.push_back(std::move(f));
    }

    bool parallelOk = true;
    for (auto &f : futures) {
      try {
        f.get();
      } catch (...) {
        parallelOk = false;
      }
    }

    if (!parallelOk) {
      projectAndScore(0, units.size());
    }
  } else {
    projectAndScore(0, units.size());
  }

  float maxScore = -std::numeric_limits<float>::infinity();
  for (float w : weights) {
    maxScore = std::max(maxScore, w);
  }

  /* Numerically stable softmax. */
  double sumExp = 0.0;
  for (float w : weights) {
    sumExp += std::exp(static_cast<double>(w - maxScore));
  }
  for (float &w : weights) {
    w = static_cast<float>(std::exp(static_cast<double>(w - maxScore)) / sumExp);
  }

  SemanticUnit out;
  out.semanticVector.resize(targetDim, 0.0f);
  size_t bestIdx = 0;
  float bestWeight = weights[0];
  for (size_t i = 0; i < units.size(); ++i) {
    if (weights[i] > bestWeight) {
      bestWeight = weights[i];
      bestIdx = i;
    }
    const auto &uv = projected[i];
    for (size_t d = 0; d < targetDim; ++d) {
      out.semanticVector[d] += uv[d] * weights[i];
    }
  }
  out.semanticVector = normalizeVector(out.semanticVector);
  out.modality = units[bestIdx].modality;
  out.confidence = bestWeight;
  out.timestampMs = query.timestampMs;
  out.content = units[bestIdx].content;
  for (const auto &u : units) {
    for (const auto &kv : u.modalWeights) out.modalWeights[kv.first] += kv.second;
  }
  if (!out.modalWeights.empty()) {
    for (auto &kv : out.modalWeights) kv.second /= static_cast<float>(units.size());
  }
  out.metadata["fusion"] = "attention";
  return out;
}

SemanticUnit fuseAttention(const SemanticUnit &query,
                           const std::vector<SemanticUnit> &units,
                           size_t targetDim) {
  return fuseAttention(query, units, targetDim, 0x61727468U);
}

SemanticMemory::SemanticMemory(size_t reserveCount) {
  units_.reserve(reserveCount);
}

std::vector<SemanticUnit> SemanticMemory::search(const SemanticUnit &query,
                                               size_t topK) const {
  auto scored = retrieve(query, topK);
  std::vector<SemanticUnit> out;
  out.reserve(scored.size());
  for (auto &p : scored) out.push_back(std::move(p.first));
  return out;
}

SemanticUnit SemanticMemory::fuseUnits(const std::vector<SemanticUnit> &units,
                                     size_t targetDim) const {
  if (units.empty()) {
    return SemanticUnit{};
  }
  SemanticUnit result = units.front();
  for (size_t i = 1; i < units.size(); ++i) {
    result = fuseAdd(result, units[i], targetDim);
  }
  return result;
}

void SemanticMemory::addUnit(const SemanticUnit &unit) {
  std::lock_guard<std::mutex> lock(mu_);
  SemanticUnit copy = unit;
  if (copy.timestampMs == 0) {
    copy.timestampMs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }
  units_.push_back(std::move(copy));
}

std::vector<std::pair<SemanticUnit, float>> SemanticMemory::retrieve(
    const SemanticUnit &query,
    size_t topK) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::pair<SemanticUnit, float>> scored;
  scored.reserve(units_.size());
  for (const auto &u : units_) {
    float sim = cosineSimilarity(query.semanticVector, u.semanticVector);
    scored.emplace_back(u, sim);
  }
  // Partial sort returns the top-K (descending) in O(N log K) instead of the
  // previous full O(N log N) sort — an asymptotic win whenever topK << N.
  const size_t k = std::min(topK, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k),
                    scored.end(),
                    [](const auto &a, const auto &b) { return a.second > b.second; });
  scored.resize(k);
  return scored;
}

SemanticUnit SemanticMemory::fuseAll(size_t targetDim) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (units_.empty()) {
    return SemanticUnit{};
  }
  if (units_.size() == 1) {
    return units_.front();
  }
  SemanticUnit summary = units_.front();
  for (size_t i = 1; i < units_.size(); ++i) {
    summary = fuseAdd(summary, units_[i], targetDim);
  }
  return summary;
}

void SemanticMemory::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  units_.clear();
}

nlohmann::json SemanticMemory::toJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  nlohmann::json j = nlohmann::json::array();
  for (const auto &u : units_) {
    j.push_back(u.toJson());
  }
  return j;
}

void SemanticMemory::fromJson(const nlohmann::json &j) {
  std::lock_guard<std::mutex> lock(mu_);
  units_.clear();
  if (!j.is_array()) {
    return;
  }
  for (const auto &item : j) {
    units_.push_back(SemanticUnit::fromJson(item));
  }
}

size_t SemanticMemory::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return units_.size();
}

} // namespace multimodal
} // namespace phoenix
