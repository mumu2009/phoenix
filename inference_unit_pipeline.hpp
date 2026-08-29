/* inference_unit_pipeline.hpp - unified text/unit iteration workflow

   Single-instance path (every module except infer is optional):
     I/O enc (paragraph in, unit-query sequence out)
       -> preprocess (memory / emotion / ...)
       -> gnn
       -> infer
       -> I/O dec (unit-query sequence in, paragraph text out)
   Paragraph I/O is an async boundary (enc/dec/infer can overlap later).
   It does NOT mean-pool a paragraph into one vector. Enc still emits a
   unit-query sequence. After enc, N-gram merge (2-3 adjacent token
   rows -> one equivalent vector) is the same protocol step for every
   module that speaks unit query: enc output, preprocess, gnn, infer,
   dec. Causal and RAG both go through it — they are the same unit
   stream type. Pre-built graph-node rows are already units (not token
   n-grams) and are not merged again.
   Two infer streams (mainstream RAG / FiD-style, not one causal pipe):
     causal: pin + packed recent draft (only this is decoded)
     rag:    memory + GNN + dropped-head summary unit-query sequences
   RAG is cross-attended into causal embeddings as residual emphasis.
   Last resume units stay unmerged so generate continues the document,
   never the memory. RAG rows never occupy sequence positions.
   Internal 8B pair: infer -> dec -> enc -> infer.
   Infer never sees memory as prompt text. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "inference_backend_router.hpp"

namespace phoenix {
namespace inference {

inline std::vector<float> poolUnitRows(
    const std::vector<std::vector<float>> &rows) {
  if (rows.empty()) return {};
  const size_t dim = rows[0].size();
  if (dim == 0) return {};
  std::vector<float> out(dim, 0.f);
  size_t n = 0;
  for (const auto &r : rows) {
    if (r.size() != dim) continue;
    for (size_t i = 0; i < dim; ++i) out[i] += r[i];
    ++n;
  }
  if (n == 0) return {};
  const float inv = 1.f / static_cast<float>(n);
  for (float &v : out) v *= inv;
  return out;
}

inline bool moduleArmed(bool enabled, float weight) {
  return enabled && weight > 0.f;
}

inline nlohmann::json unitRowsToJson(
    const std::vector<std::vector<float>> &rows);

inline std::string joinUnitQueryTexts(const std::vector<std::string> &parts) {
  std::string out;
  for (const auto &p : parts) {
    if (p.empty()) continue;
    if (!out.empty()) out.push_back('\n');
    out += p;
  }
  return out;
}

/** I/O batch = one paragraph. Enc of that paragraph is still a unit-query
    sequence (one row per token), not one pooled vector. */
inline std::string trimIoParagraph(std::string s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return {};
  const auto b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

inline std::vector<std::string> splitIoParagraphs(const std::string &text) {
  std::string norm;
  norm.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\r') {
      if (i + 1 < text.size() && text[i + 1] == '\n') continue;
      norm.push_back('\n');
    } else {
      norm.push_back(text[i]);
    }
  }
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos < norm.size()) {
    const size_t eol = norm.find("\n\n", pos);
    const std::string raw = eol == std::string::npos
                                ? norm.substr(pos)
                                : norm.substr(pos, eol - pos);
    const std::string para = trimIoParagraph(raw);
    if (!para.empty()) out.push_back(para);
    if (eol == std::string::npos) break;
    pos = eol + 2;
  }
  return out;
}

/** Memory / GNN I/O as seen by infer: unit-query packets, not prompt text.
    Fair across text/audio/video — infer only concatenates E-space rows.
    rows: already encoded (LLaVA / Qwen2-Audio / llama enc).
    content: payload for the matching enc when rows are empty.
    audio/video/image without rows are skipped (must not be tokenized). */
struct UnitQueryIO {
  std::string modality{"text"};
  std::vector<std::vector<float>> rows;
  std::string content;
};

/** One packet = one paragraph. Split multi-paragraph blobs first. */
inline UnitQueryIO unitQueryFromText(const std::string &text,
                                     const std::string &modality = "text") {
  UnitQueryIO p;
  p.modality = modality.empty() ? "text" : modality;
  p.content = trimIoParagraph(text);
  return p;
}

inline UnitQueryIO unitQueryFromRows(std::vector<std::vector<float>> rows,
                                     const std::string &modality = "unit") {
  UnitQueryIO p;
  p.modality = modality.empty() ? "unit" : modality;
  p.rows = std::move(rows);
  return p;
}

inline bool unitQueryEmpty(const UnitQueryIO &p) {
  return p.rows.empty() && p.content.empty();
}

inline nlohmann::json unitQueryIOToJson(const UnitQueryIO &p) {
  nlohmann::json j = nlohmann::json::object();
  j["modality"] = p.modality.empty() ? "text" : p.modality;
  if (!p.rows.empty())
    j["rows"] = unitRowsToJson(p.rows);
  if (!p.content.empty())
    j["content"] = p.content;
  return j;
}

inline nlohmann::json unitQueryListToJson(
    const std::vector<UnitQueryIO> &items) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &p : items) {
    if (unitQueryEmpty(p)) continue;
    arr.push_back(unitQueryIOToJson(p));
  }
  return arr;
}

inline void appendUnitQuery(std::vector<UnitQueryIO> &dst, UnitQueryIO p) {
  if (!unitQueryEmpty(p))
    dst.push_back(std::move(p));
}

/** I/O enc payload: one UnitQueryIO per paragraph. */
inline void appendUnitQueriesFromParagraphs(std::vector<UnitQueryIO> &dst,
                                            const std::string &text,
                                            const std::string &modality = "text") {
  const auto paras = splitIoParagraphs(text);
  if (paras.empty()) {
    const std::string t = trimIoParagraph(text);
    if (!t.empty())
      appendUnitQuery(dst, unitQueryFromText(t, modality));
    return;
  }
  for (const auto &p : paras)
    appendUnitQuery(dst, unitQueryFromText(p, modality));
}

inline std::vector<std::vector<float>> concatUnitRows(
    const std::vector<std::vector<float>> &left,
    const std::vector<std::vector<float>> &right) {
  std::vector<std::vector<float>> out;
  out.reserve(left.size() + right.size());
  out.insert(out.end(), left.begin(), left.end());
  out.insert(out.end(), right.begin(), right.end());
  return out;
}

/** Cross-attend one causal query onto RAG keys/values (embedding space). */
inline std::vector<float> ragAttend(
    const std::vector<float> &query,
    const std::vector<std::vector<float>> &keys) {
  std::vector<float> out(query.size(), 0.f);
  if (keys.empty() || query.empty()) return out;
  const float inv = 1.f / std::sqrt(static_cast<float>(query.size()));
  std::vector<float> score(keys.size(), 0.f);
  float mx = -1e30f;
  for (size_t j = 0; j < keys.size(); ++j) {
    double s = 0.0;
    const size_t n = std::min(query.size(), keys[j].size());
    for (size_t d = 0; d < n; ++d)
      s += static_cast<double>(query[d]) * static_cast<double>(keys[j][d]);
    score[j] = static_cast<float>(s) * inv;
    if (score[j] > mx) mx = score[j];
  }
  float sum = 0.f;
  for (float &s : score) {
    s = std::exp(s - mx);
    sum += s;
  }
  if (sum <= 0.f) return out;
  for (size_t j = 0; j < keys.size(); ++j) {
    const float w = score[j] / sum;
    const size_t n = std::min(out.size(), keys[j].size());
    for (size_t d = 0; d < n; ++d)
      out[d] += w * keys[j][d];
  }
  return out;
}

/** Adjacent token rows -> one equivalent unit (mean of the window).
    ngram 0/1 = no-op. ngram 2 or 3. Last protectLast rows stay raw so
    the continuation tip is still one unit per token. */
inline void mergeNgramUnits(std::vector<std::vector<float>> &rows, int ngram,
                            int protectLast = 0) {
  if (ngram < 2 || rows.size() < 2) return;
  ngram = std::min(ngram, 3);
  const int n = static_cast<int>(rows.size());
  const int prot = std::max(0, std::min(protectLast, n));
  const int head = n - prot;
  if (head < ngram) return;
  std::vector<std::vector<float>> out;
  out.reserve(static_cast<size_t>(head / ngram + prot + 1));
  int i = 0;
  while (i < head) {
    int take = ngram;
    if (i + take > head) take = head - i;
    if (take <= 0) break;
    if (take == 1) {
      out.push_back(std::move(rows[static_cast<size_t>(i)]));
      ++i;
      continue;
    }
    const size_t dim = rows[static_cast<size_t>(i)].size();
    std::vector<float> acc(dim, 0.f);
    int got = 0;
    for (int k = 0; k < take; ++k) {
      const auto &r = rows[static_cast<size_t>(i + k)];
      const size_t nd = std::min(dim, r.size());
      for (size_t d = 0; d < nd; ++d)
        acc[d] += r[d];
      ++got;
    }
    if (got > 0) {
      const float inv = 1.f / static_cast<float>(got);
      for (float &v : acc) v *= inv;
    }
    out.push_back(std::move(acc));
    i += take;
  }
  for (int j = head; j < n; ++j)
    out.push_back(std::move(rows[static_cast<size_t>(j)]));
  rows.swap(out);
}

inline int ngramMergeFactor(int ngram) {
  return ngram >= 2 ? std::min(ngram, 3) : 1;
}

/** How many trailing causal units stay unmixed.
    Empty / tiny resume (first tick): 0 — assignment RAG must reach the
    last position, or the first sampled token ignores the task.
    Long resume: only a short continuation tip, never the whole prefix. */
inline int ragProtectUnits(int nUnits, int resumeUnits, int tip = 16) {
  if (nUnits <= 0) return 0;
  if (resumeUnits < 24) return 0;
  const int t = std::max(1, tip);
  return std::min(t, std::max(0, nUnits / 4));
}

/** Residual RAG mix. Last protectLast causal rows stay clean (continuation). */
inline void ragMixIntoCausal(std::vector<std::vector<float>> &causal,
                             const std::vector<std::vector<float>> &rag,
                             float beta, int protectLast) {
  if (causal.empty() || rag.empty() || beta <= 0.f) return;
  const int n = static_cast<int>(causal.size());
  const int prot = std::max(0, std::min(protectLast, n));
  const int last = n - prot;
  for (int i = 0; i < last; ++i) {
    const auto add = ragAttend(causal[static_cast<size_t>(i)], rag);
    auto &row = causal[static_cast<size_t>(i)];
    const size_t dmax = std::min(row.size(), add.size());
    for (size_t d = 0; d < dmax; ++d)
      row[d] += beta * add[d];
  }
}

/** Legacy name: two-stream mix, not concat onto the causal tail. */
inline std::vector<std::vector<float>> modulateEncHidden(
    const std::vector<std::vector<float>> &encHidden,
    const std::vector<std::vector<float>> &memoryUnits,
    const std::vector<std::vector<float>> &gnnUnits, float memoryWeight,
    float gnnWeight) {
  auto out = encHidden;
  if (moduleArmed(true, memoryWeight) && !memoryUnits.empty())
    ragMixIntoCausal(out, memoryUnits, memoryWeight * 0.18f, 48);
  if (moduleArmed(true, gnnWeight) && !gnnUnits.empty())
    ragMixIntoCausal(out, gnnUnits, gnnWeight * 0.12f, 48);
  return out;
}

inline std::vector<std::vector<float>> jsonToUnitRows(const nlohmann::json &arr) {
  std::vector<std::vector<float>> out;
  if (!arr.is_array()) return out;
  for (const auto &row : arr) {
    if (!row.is_array()) continue;
    std::vector<float> v;
    for (const auto &x : row) {
      if (x.is_number()) v.push_back(static_cast<float>(x.get<double>()));
    }
    if (!v.empty()) out.push_back(std::move(v));
  }
  return out;
}

inline nlohmann::json unitRowsToJson(
    const std::vector<std::vector<float>> &rows) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &r : rows) arr.push_back(r);
  return arr;
}

/** Stable slot for chat / mission isolation on llama-server parallel slots. */
inline int slotFromInstanceKey(const std::string &key, int maxSlots) {
  maxSlots = std::max(1, maxSlots);
  std::hash<std::string> h;
  const size_t bucket = h(key) % static_cast<size_t>(maxSlots);
  return static_cast<int>(bucket);
}

inline int chatSlotId(const std::string &sessionId, int maxSlots) {
  return slotFromInstanceKey(std::string("chat:") + sessionId, maxSlots);
}

inline int missionSlotId(const std::string &scope, int maxSlots) {
  return slotFromInstanceKey(std::string("mission:") + scope, maxSlots);
}

struct UnitPipelineConfig {
  std::string loopMode{"unit"};
  std::string feedbackMode{"dec_enc"};
  bool autoFeedbackFallback{true};
  bool memoryEnabled{true};
  bool gnnEnabled{true};
  float memoryWeight{1.0f};
  float gnnWeight{1.0f};
  int ngramMerge{2};
  int ngramProtectLast{16};
};

inline UnitPipelineConfig pipelineConfigFromJson(const nlohmann::json &opts) {
  UnitPipelineConfig cfg;
  if (!opts.is_object()) return cfg;
  if (opts.contains("loop_mode") && opts["loop_mode"].is_string())
    cfg.loopMode = opts["loop_mode"].get<std::string>();
  if (opts.contains("feedback_mode") && opts["feedback_mode"].is_string())
    cfg.feedbackMode = opts["feedback_mode"].get<std::string>();
  if (opts.contains("autoFeedbackFallback") &&
      opts["autoFeedbackFallback"].is_boolean())
    cfg.autoFeedbackFallback = opts["autoFeedbackFallback"].get<bool>();
  if (opts.contains("memoryEnabled") && opts["memoryEnabled"].is_boolean())
    cfg.memoryEnabled = opts["memoryEnabled"].get<bool>();
  if (opts.contains("gnnEnabled") && opts["gnnEnabled"].is_boolean())
    cfg.gnnEnabled = opts["gnnEnabled"].get<bool>();
  if (opts.contains("memoryWeight") && opts["memoryWeight"].is_number())
    cfg.memoryWeight = static_cast<float>(opts["memoryWeight"].get<double>());
  if (opts.contains("gnnWeight") && opts["gnnWeight"].is_number())
    cfg.gnnWeight = static_cast<float>(opts["gnnWeight"].get<double>());
  if (opts.contains("ngram_merge") && opts["ngram_merge"].is_number_integer())
    cfg.ngramMerge = opts["ngram_merge"].get<int>();
  if (opts.contains("ngramMerge") && opts["ngramMerge"].is_number_integer())
    cfg.ngramMerge = opts["ngramMerge"].get<int>();
  if (opts.contains("ngram_protect_last") &&
      opts["ngram_protect_last"].is_number_integer())
    cfg.ngramProtectLast = opts["ngram_protect_last"].get<int>();
  if (cfg.memoryWeight <= 0.f) cfg.memoryEnabled = false;
  if (cfg.gnnWeight <= 0.f) cfg.gnnEnabled = false;
  if (cfg.ngramMerge < 0) cfg.ngramMerge = 0;
  if (cfg.ngramMerge > 3) cfg.ngramMerge = 3;
  if (cfg.ngramProtectLast < 0) cfg.ngramProtectLast = 0;
  if (cfg.ngramProtectLast > 128) cfg.ngramProtectLast = 128;
  return cfg;
}

}  // namespace inference
}  // namespace phoenix
