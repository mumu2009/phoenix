/* concept_matrix.hpp - Unified sparse concept matrix (text-first, multimodal stubs)
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include "semantic_unit.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phoenix {
namespace conceptmatrix {

struct ConceptMatrixConfig {
  size_t rows{4096};
  size_t cols{4096};
  size_t conceptDim{128};
  float minConfidence{0.05f};
  float sparsityThreshold{0.01f};
  uint32_t positionSeed{0x434F4E43U};  // 'CONC'
};

/**
 * @brief Sparse concept matrix with deterministic concept -> position mapping.
 *
 * Text, image and audio concepts are stored in the same matrix.  In v7.4 only
 * text encoding is functional; image/audio are stubs that reserve the position
 * pipeline so cross-modal unification can be filled in later.
 */
class ConceptMatrix {
 public:
  explicit ConceptMatrix(const ConceptMatrixConfig &config = {});

  // Deterministic hash of a concept id to a matrix cell.
  std::pair<size_t, size_t> conceptToPosition(const std::string &conceptId) const;

  // Place or fuse a SemanticUnit at its deterministic position.
  void activate(const std::string &conceptId,
                const phoenix::multimodal::SemanticUnit &unit,
                float activation = 1.0f);

  // Encode modalities into the matrix.  Text is real; image/audio are stubs.
  std::vector<std::string> encodeText(const std::string &text);
  std::vector<std::string> encodeImage(const std::vector<uint8_t> &bytes);
  std::vector<std::string> encodeAudio(const std::vector<float> &samples);

  // Run GNN-style reasoning: propagate activation to nearby similar units.
  void propagate(size_t steps = 2);

  // Retrieve top-k active concepts as (conceptId, confidence).
  std::vector<std::pair<std::string, float>> topConcepts(size_t k,
                                                         float minConfidence) const;

  // Decode active concepts into a short text context string.
  std::string toContextString(size_t k = 64, float minConfidence = 0.1f) const;

  void clear();
  size_t activeCount() const;

 private:
  ConceptMatrixConfig config_;
  std::unordered_map<uint64_t, phoenix::multimodal::SemanticUnit> units_;
  mutable std::mutex mutex_;

  uint64_t positionHash(const std::string &conceptId) const;
  uint64_t key(size_t row, size_t col) const { return row * config_.cols + col; }

  std::vector<std::string> tokenize_(const std::string &text) const;
  std::vector<float> embedText_(const std::string &text) const;
};

}  // namespace conceptmatrix
}  // namespace phoenix
