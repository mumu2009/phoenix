/* semantic_unit.hpp - Multimodal semantic unit representation and fusion
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

#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace phoenix {
namespace multimodal {

/**
 * @brief Supported modalities for a SemanticUnit.
 */
enum class Modality {
  Text,       /**< Text or token sequence */
  Image,      /**< Image or visual patch */
  Audio,      /**< Audio waveform or spectrogram */
  Video,      /**< Video clip or frame sequence */
  Sensor,     /**< Structured sensor data */
  Structured  /**< JSON/tabular data */
};

/** Convert a modality to a human-readable string. */
std::string modalityToString(Modality m);

/** Parse a modality from a string (case-insensitive). */
Modality modalityFromString(const std::string &s);

/** Generate a deterministic 16-character hex semantic unit id. */
std::string generateSemanticId(const std::string &content = "", uint64_t seed = 0x617274687572ULL);

/**
 * @brief The basic storage unit for v7.0 true-multimodal connection.
 *
 * A SemanticUnit is a modality-agnostic container with a semantic vector,
 * optional raw content, confidence, timestamp, metadata, and association IDs.
 */
struct SemanticUnit {
  std::string id;                                     /**< Stable correlation id */
  Modality modality{Modality::Text};
  std::vector<float> semanticVector;                  /**< Unified embedding vector (pooled unit query) */
  std::vector<std::vector<float>> unitQuerySequence;  /**< Optional per-token unit queries for media enc/dec */
  std::string content;                              /**< Optional payload or summary */
  float confidence{0.0f};                             /**< Confidence in [0, 1] */
  uint64_t timestampMs{0};                            /**< UTC milliseconds */
  std::map<std::string, std::string> metadata;        /**< Additional metadata */
  std::vector<std::string> associationIds;            /**< IDs of related units */
  std::map<std::string, float> modalWeights;         /**< Per-modality fusion weights */

  SemanticUnit();

  /** Serialize to JSON. */
  nlohmann::json toJson() const;

  /** Deserialize from JSON. */
  static SemanticUnit fromJson(const nlohmann::json &j);
};

/**
 * @brief Normalize a vector to unit length.
 * @param v Input vector.
 * @return Normalized copy, or zero vector if length is zero.
 */
std::vector<float> normalizeVector(const std::vector<float> &v);

/**
 * @brief Cosine similarity between two vectors.
 * @return Value in [-1, 1], or 0 if either vector is zero.
 */
float cosineSimilarity(const std::vector<float> &a,
                         const std::vector<float> &b);

/**
 * @brief Project a vector to a target dimension using a deterministic random matrix.
 *
 * The projection matrix is seeded by the source/target dimensions so the same
 * (source, target) pair always yields the same mapping.  This allows vectors
 * from different modalities to be fused in a unified embedding space without
 * retraining the base LLM.
 *
 * @param v Input vector.
 * @param targetDim Desired output dimension.  If 0, the input dimension is kept.
 * @param seed Additional seed for reproducibility.
 * @return Projected vector.
 */
std::vector<float> projectToDimension(const std::vector<float> &v,
                                      size_t targetDim,
                                      unsigned int seed = 0x61727468U);

/**
 * @brief Project a vector to a target dimension using a sparse Johnson-
 *        Lindenstrauss (Achlioptas-style) random projection.
 *
 * This is the implementation of the algorithm.md 17.6.3 recommendation:
 * for large source*target it reduces the cost from O(source*target) to
 * O(source*nonZeros) while preserving distances with high probability.
 * Each source dimension is mapped to nonZerosPerColumn distinct target rows
 * with +/- 1/sqrt(nonZerosPerColumn) values.
 *
 * @param v Input vector.
 * @param targetDim Desired output dimension.  If 0, the input dimension is kept.
 * @param nonZerosPerColumn Number of non-zero entries per column.  If 0, a
 *        default of max(3, targetDim/3) is used.
 * @param seed Additional seed for reproducibility.
 * @return Projected vector.
 */
std::vector<float> projectToDimensionSparse(const std::vector<float> &v,
                                            size_t targetDim,
                                            size_t nonZerosPerColumn = 0,
                                            unsigned int seed = 0x61727468U);

/**
 * @brief Element-wise addition of two semantic units after projection.
 */
SemanticUnit fuseAdd(const SemanticUnit &a,
                     const SemanticUnit &b,
                     size_t targetDim = 0);

/**
 * @brief Element-wise (Hadamard) multiplication after projection.
 */
SemanticUnit fuseMultiply(const SemanticUnit &a,
                          const SemanticUnit &b,
                          size_t targetDim = 0);

/**
 * @brief Attention-weighted fusion of a query against a pool of units.
 *
 * Computes cosine similarity between the query and each unit, applies softmax,
 * and returns the weighted sum.  The returned unit's modality is the modality
 * of the highest-weight input unit.
 */
SemanticUnit fuseAttention(const SemanticUnit &query,
                           const std::vector<SemanticUnit> &units,
                           size_t targetDim = 0);

/**
 * @brief Lightweight in-memory store for SemanticUnits.
 *
 * Provides add, retrieve-by-similarity, and attention-fusion helpers.  This is
 * intended as an adapter layer for the existing context and memory systems.
 */
class SemanticMemory {
 public:
  explicit SemanticMemory(size_t reserveCount = 0);

  /** Add a unit to memory. */
  void addUnit(const SemanticUnit &unit);

  /**
   * @brief Retrieve the top-K units most similar to a query unit.
   */
  std::vector<std::pair<SemanticUnit, float>> retrieve(
      const SemanticUnit &query,
      size_t topK = 5) const;

  /** Fuse all stored units into a single semantic summary. */
  SemanticUnit fuseAll(size_t targetDim = 0) const;

  /** Clear all units. */
  void clear();

  /** Serialize to JSON. */
  nlohmann::json toJson() const;

  /** Deserialize from JSON. */
  void fromJson(const nlohmann::json &j);

  /** Total number of stored units. */
  size_t size() const;

  /** Convenience: retrieve top-K units (without similarity scores). */
  std::vector<SemanticUnit> search(const SemanticUnit &query,
                                   size_t topK = 5) const;

  /** Fuse a vector of units into one semantic unit. */
  SemanticUnit fuseUnits(const std::vector<SemanticUnit> &units,
                         size_t targetDim = 0) const;

 private:
  mutable std::mutex mu_;
  std::vector<SemanticUnit> units_;
};

} // namespace multimodal
} // namespace phoenix
