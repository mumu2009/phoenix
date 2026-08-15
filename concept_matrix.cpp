/* concept_matrix.cpp - Unified sparse concept matrix implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "concept_matrix.hpp"

#include "async_task_system.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <sstream>
#include <thread>
#include <vector>

namespace phoenix {
namespace conceptmatrix {

namespace {

// FNV-1a 64-bit hash.
uint64_t fnv1a64(const std::string &s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) {
    h ^= static_cast<uint64_t>(c);
    h *= 0x100000001b3ULL;
  }
  return h;
}

std::string normalize(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isalnum(c))
      out.push_back(static_cast<char>(std::tolower(c)));
    else if (!out.empty() && out.back() != ' ')
      out.push_back(' ');
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

}  // namespace

ConceptMatrix::ConceptMatrix(const ConceptMatrixConfig &config) : config_(config) {}

std::pair<size_t, size_t> ConceptMatrix::conceptToPosition(
    const std::string &conceptId) const {
  uint64_t h = fnv1a64(conceptId + std::to_string(config_.positionSeed));
  size_t row = h % config_.rows;
  size_t col = (h >> 32) % config_.cols;
  return {row, col};
}

uint64_t ConceptMatrix::positionHash(const std::string &conceptId) const {
  auto [row, col] = conceptToPosition(conceptId);
  return key(row, col);
}

void ConceptMatrix::activate(const std::string &conceptId,
                             const phoenix::multimodal::SemanticUnit &unit,
                             float activation) {
  uint64_t k = positionHash(conceptId);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = units_.find(k);
  if (it != units_.end()) {
    // Simple average of semantic vectors to avoid expensive attention fusion.
    if (!it->second.semanticVector.empty() && !unit.semanticVector.empty()) {
      size_t n = std::min(it->second.semanticVector.size(), unit.semanticVector.size());
      for (size_t i = 0; i < n; ++i) {
        it->second.semanticVector[i] =
            (it->second.semanticVector[i] * it->second.confidence +
             unit.semanticVector[i] * activation) /
            (it->second.confidence + activation + 1e-6f);
      }
    }
    it->second.confidence =
        std::min(1.0f, it->second.confidence + activation * 0.1f);
  } else {
    phoenix::multimodal::SemanticUnit u = unit;
    u.confidence = activation;
    units_[k] = u;
  }
}

std::vector<std::string> ConceptMatrix::tokenize_(const std::string &text) const {
  std::vector<std::string> out;
  std::string s = normalize(text);
  std::istringstream iss(s);
  std::string w;
  while (iss >> w) {
    if (!w.empty()) out.push_back(w);
  }
  // Bigrams / trigrams as composite concepts.
  for (size_t i = 0; i + 1 < out.size(); ++i) out.push_back(out[i] + "_" + out[i + 1]);
  for (size_t i = 0; i + 2 < out.size(); ++i)
    out.push_back(out[i] + "_" + out[i + 1] + "_" + out[i + 2]);
  return out;
}

std::vector<float> ConceptMatrix::embedText_(const std::string &text) const {
  std::vector<float> vec(config_.conceptDim, 0.0f);
  uint64_t h = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    h = h * 31 + static_cast<unsigned char>(text[i]);
    vec[i % config_.conceptDim] +=
        static_cast<float>(static_cast<int8_t>(h & 0xFF)) / 128.0f;
  }
  float norm = 0.0f;
  for (float v : vec) norm += v * v;
  if (norm > 0.0f) {
    norm = std::sqrt(norm);
    for (float &v : vec) v /= norm;
  }
  return vec;
}

std::vector<std::string> ConceptMatrix::encodeText(const std::string &text) {
  auto tokens = tokenize_(text);
std::vector<std::string> ids;
  ids.reserve(tokens.size());
  for (const auto &tok : tokens) {
    std::string conceptId = "txt:" + tok;
    phoenix::multimodal::SemanticUnit u;
    u.id = conceptId;
    u.modality = phoenix::multimodal::Modality::Text;
    u.content = tok;
u.semanticVector = embedText_(tok);
    u.confidence = 1.0f;
    activate(conceptId, u, 1.0f);
    ids.push_back(conceptId);
  }
  return ids;
}

std::vector<std::string> ConceptMatrix::encodeImage(const std::vector<uint8_t> &bytes) {
  // v7.4 stub: reserve a deterministic image concept position.
  std::string conceptId = "img:stub";
  phoenix::multimodal::SemanticUnit u;
  u.id = conceptId;
  u.modality = phoenix::multimodal::Modality::Image;
  u.content = "<image>";
  u.semanticVector = embedText_("image");
  activate(conceptId, u, 0.5f);
  return {conceptId};
}

std::vector<std::string> ConceptMatrix::encodeAudio(const std::vector<float> &samples) {
  // v7.4 stub: reserve a deterministic audio concept position.
  std::string conceptId = "aud:stub";
  phoenix::multimodal::SemanticUnit u;
  u.id = conceptId;
  u.modality = phoenix::multimodal::Modality::Audio;
  u.content = "<audio>";
  u.semanticVector = embedText_("audio");
  activate(conceptId, u, 0.5f);
  return {conceptId};
}

void ConceptMatrix::propagate(size_t steps) {
  if (steps == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t s = 0; s < steps; ++s) {
    std::unordered_map<uint64_t, float> boosts;

    using UnitPtr =
        const std::pair<const uint64_t, phoenix::multimodal::SemanticUnit> *;
    std::vector<UnitPtr> allUnits;
    allUnits.reserve(units_.size());
    for (const auto &kv : units_) {
      allUnits.push_back(&kv);
    }

    // Gather boosts for a contiguous range of units into `out`.
    // Each unit's 24-neighbour scan is independent and only reads `units_`.
    // `begin` and `end` are pointers into the `allUnits` array, not into the
    // unordered_map (which is not contiguous), so we dereference once.
    auto gatherBoosts = [this](const UnitPtr *begin, const UnitPtr *end,
                               std::unordered_map<uint64_t, float> &out) {
      const auto &unitsC = this->units_;
      for (const UnitPtr *p = begin; p != end; ++p) {
        const auto &unit = (*p)->second;
        if (unit.confidence < this->config_.sparsityThreshold) continue;
        auto [row, col] = this->conceptToPosition(unit.id);
        for (int dr = -2; dr <= 2; ++dr) {
          for (int dc = -2; dc <= 2; ++dc) {
            if (dr == 0 && dc == 0) continue;
            size_t nr = (row + dr + this->config_.rows) % this->config_.rows;
            size_t nc = (col + dc + this->config_.cols) % this->config_.cols;
            uint64_t nk = this->key(nr, nc);
            auto it = unitsC.find(nk);
            if (it == unitsC.end()) continue;
            float sim = phoenix::multimodal::cosineSimilarity(
                unit.semanticVector, it->second.semanticVector);
            if (sim > 0.2f) {
              out[nk] += unit.confidence * sim * 0.25f;
            }
          }
        }
      }
    };

    const bool tryParallel = allUnits.size() > 16;
    bool usedParallel = false;

    const size_t workerCount =
        std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t chunkCount = std::min(allUnits.size(), workerCount);

    if (tryParallel && chunkCount > 1) {
      auto &ats = phoenix::v7::AsyncTaskSystem::global();
      ats.start();

      std::vector<std::unordered_map<uint64_t, float>> chunkBoosts(chunkCount);
      std::vector<std::future<void>> futures;
      futures.reserve(chunkCount);

      for (size_t c = 0; c < chunkCount; ++c) {
        const size_t start = c * allUnits.size() / chunkCount;
        const size_t end = (c + 1) * allUnits.size() / chunkCount;
        auto *local = &chunkBoosts[c];
        std::future<void> f = ats.submitWithFuture(
            phoenix::v7::TaskModule::Encoder,
            phoenix::v7::TaskPriority::High,
            [this, gatherBoosts, &allUnits, local, start, end]() {
              gatherBoosts(&allUnits[start], &allUnits[end], *local);
            },
            "propagate_chunk");
        futures.push_back(std::move(f));
      }

      bool ok = true;
      for (auto &f : futures) {
        try {
          f.get();
        } catch (...) {
          ok = false;
        }
      }

      if (ok) {
        usedParallel = true;
        for (const auto &local : chunkBoosts) {
          for (const auto &kv : local) {
            boosts[kv.first] += kv.second;
          }
        }
      }
    }

    if (!usedParallel) {
      gatherBoosts(allUnits.data(), allUnits.data() + allUnits.size(),
                   boosts);
    }

    // Apply the accumulated boosts under the matrix mutex, sequential.
    for (const auto &[k, delta] : boosts) {
      auto it = units_.find(k);
      if (it != units_.end()) {
        it->second.confidence = std::min(1.0f, it->second.confidence + delta);
      }
    }
  }
}

std::vector<std::pair<std::string, float>> ConceptMatrix::topConcepts(
    size_t k, float minConfidence) const {
  std::vector<std::pair<std::string, float>> out;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    out.reserve(units_.size());
    for (const auto &[_, unit] : units_) {
      if (unit.confidence >= minConfidence) {
        out.emplace_back(unit.id, unit.confidence);
      }
    }
  }
  std::sort(out.begin(), out.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  if (out.size() > k) out.resize(k);
  return out;
}

std::string ConceptMatrix::toContextString(size_t k, float minConfidence) const {
  auto concepts = topConcepts(k, minConfidence);
  if (concepts.empty()) return "";
  std::string out = "Concept matrix context:\n";
  for (const auto &[id, conf] : concepts) {
    std::string label = id;
    auto colon = id.find(':');
    if (colon != std::string::npos) label = id.substr(colon + 1);
    out += "- " + label + " (" + std::to_string(static_cast<int>(conf * 100)) + "%)\n";
  }
  return out;
}

void ConceptMatrix::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  units_.clear();
}

size_t ConceptMatrix::activeCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return units_.size();
}

}  // namespace conceptmatrix
}  // namespace phoenix
