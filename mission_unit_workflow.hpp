/* mission_unit_workflow.hpp - memory / GNN / emotion matrix for mission ticks

   Designed path (not the text-token shortcut):
     recall CCM + experience as UnitQueryIO
     graph the draft -> node embeddings projected to llama n_embd
     pressure/emotion scale sampling AND unit weights
     after a write: deposit the new span so the next tick can recall it
   Infer must see E-space rows. Content is only the enc payload. */
#pragma once

#include "cross_context_memory.hpp"
#include "inference_backend_router.hpp"
#include "inference_unit_pipeline.hpp"
#include "mission_experience.hpp"
#include "mission_reply_parse.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace mission {

inline std::vector<std::string> splitEmotionVocabTokens(const std::string &text,
                                                       size_t maxTok = 256) {
  std::vector<std::string> tokens;
  std::string cur;
  for (unsigned char ch : text) {
    if (std::isalnum(ch) || ch == '\'') {
      cur.push_back(static_cast<char>(std::tolower(ch)));
    } else if (!cur.empty()) {
      tokens.push_back(cur);
      cur.clear();
      if (tokens.size() >= maxTok)
        return tokens;
    }
  }
  if (!cur.empty() && tokens.size() < maxTok)
    tokens.push_back(cur);
  return tokens;
}

inline std::string buildMemoryRecallQuery(const std::string &goal,
                                          const std::string &draft,
                                          size_t goalChars = 800,
                                          size_t draftChars = 400) {
  std::string q;
  if (!goal.empty()) {
    q = goal.size() > goalChars ? goal.substr(0, goalChars) : goal;
  }
  if (!draft.empty()) {
    const std::string d =
        draft.size() > draftChars ? draft.substr(draft.size() - draftChars)
                                  : draft;
    if (!q.empty()) q.push_back('\n');
    q += d;
  }
  return q;
}

inline std::string draftExperienceSummary(const std::string &draft,
                                          size_t maxChars = 400) {
  std::string sum;
  size_t sp = 0;
  int secs = 0;
  while (sp < draft.size() && secs < 6) {
    const auto eol = draft.find('\n', sp);
    const std::string line = draft.substr(
        sp, eol == std::string::npos ? std::string::npos : eol - sp);
    const auto b = line.find_first_not_of(" \t\r");
    if (b != std::string::npos &&
        (line[b] == '#' || (line.size() > b + 1 && line[b] == '*' &&
                            line[b + 1] == '*'))) {
      if (!sum.empty()) sum += "; ";
      sum += trimCopy(line.substr(b));
      ++secs;
    }
    if (eol == std::string::npos) break;
    sp = eol + 1;
  }
  if (sum.size() < 80 && !draft.empty()) {
    if (!sum.empty()) sum += " | ";
    sum += draft.substr(0, std::min(draft.size(), size_t{200}));
  }
  if (sum.size() > maxChars) sum.resize(maxChars);
  return sum;
}

inline std::vector<std::vector<float>>
projectRowsToLlamaDim(const std::vector<std::vector<float>> &rows,
                      int targetDim) {
  std::vector<std::vector<float>> out;
  out.reserve(rows.size());
  for (const auto &r : rows) {
    if (targetDim > 0 && static_cast<int>(r.size()) == targetDim)
      out.push_back(r);
    else
      out.push_back(phoenix::inference::projectUnitQuery(r, targetDim));
  }
  return out;
}

/** Token-salad / control-token dumps must not re-enter CCM or the window. */
inline bool looksLikeCollapsedProse(const std::string &s) {
  const std::string t = trimCopy(s);
  if (t.size() < 40) return false;
  if (t.find("<|") != std::string::npos) return true;
  if (t.find("://acks") != std::string::npos) return true;
  int letters = 0;
  int punct = 0;
  for (unsigned char c : t) {
    if (std::isalpha(c))
      ++letters;
    else if (!std::isspace(c) && !std::isdigit(c))
      ++punct;
  }
  if (t.size() >= 80 && letters * 2 < static_cast<int>(t.size()))
    return true;
  if (t.size() >= 120 && punct * 3 > static_cast<int>(t.size()))
    return true;
  return false;
}

inline bool draftTailLooksCollapsed(const std::string &draft,
                                    size_t tailChars = 800) {
  if (draft.empty()) return false;
  const std::string tail = draft.size() > tailChars
                               ? draft.substr(draft.size() - tailChars)
                               : draft;
  return looksLikeCollapsedProse(tail);
}

/** Recall CCM + experience. Prefer stored E-space rows; text is enc payload. */
inline void appendRecalledMemory(
    std::vector<phoenix::inference::UnitQueryIO> &dst,
    const std::string &ccmPath, const std::string &expPath,
    const std::string &goal, const std::string &draft, size_t kCcm = 3,
    size_t kExp = 3) {
  const std::string query = buildMemoryRecallQuery(goal, draft);
  if (query.empty()) return;
  for (const auto &e : phoenix::memory::ccmRecall(ccmPath, query, kCcm)) {
    if (looksLikeCollapsedProse(e.text))
      continue;
    if (!e.unitQuery.empty()) {
      if (e.text.empty())
        continue;
      phoenix::inference::appendUnitQuery(
          dst, phoenix::inference::unitQueryFromRows(e.unitQuery, e.modality));
      continue;
    }
    if (!e.text.empty())
      phoenix::inference::appendUnitQueriesFromParagraphs(dst, e.text, "text");
  }
  for (const auto &e : experienceTop(expPath, goal, kExp)) {
    if (e.summary.empty() || looksLikeCollapsedProse(e.summary))
      continue;
    phoenix::inference::appendUnitQueriesFromParagraphs(dst, e.summary, "text");
  }
}

inline void depositMissionProgress(const std::string &ccmPath,
                                   const std::string &expPath,
                                   const std::string &sourceTag,
                                   const std::string &goal,
                                   const std::string &draft,
                                   const std::string &newSpan) {
  if (looksLikeCollapsedProse(newSpan) || draftTailLooksCollapsed(draft))
    return;
  const std::string sum = draftExperienceSummary(draft);
  if (!sum.empty() && !looksLikeCollapsedProse(sum))
    experienceAdd(expPath, goal, sum);
  std::string remember = trimCopy(newSpan);
  if (remember.size() > 480) remember.resize(480);
  if (remember.empty()) remember = sum;
  if (!remember.empty() && !looksLikeCollapsedProse(remember))
    phoenix::memory::ccmRemember(ccmPath, sourceTag, remember);
}

/** Merge emotion tensor into mission sampling. Values are absolute
    (emotion_system.hpp), then blended with the already-set mission base. */
inline void applyEmotionSampling(nlohmann::json &opts,
                                 const nlohmann::json &emotionOpts,
                                 const nlohmann::json &emotionTensor,
                                 float pressure) {
  auto blend = [](double base, double emo, double w) {
    return base * (1.0 - w) + emo * w;
  };
  double arousal = 0.0;
  if (emotionTensor.is_object() && emotionTensor.contains("arousal") &&
      emotionTensor["arousal"].is_number())
    arousal = emotionTensor["arousal"].get<double>();
  const double w = std::min(0.85, 0.25 + 0.6 * std::fabs(static_cast<double>(pressure)) +
                                      0.2 * std::fabs(arousal));
  if (emotionOpts.is_object()) {
    if (emotionOpts.contains("temperature") &&
        emotionOpts["temperature"].is_number()) {
      const double base = opts.value("temperature", 0.35);
      opts["temperature"] =
          blend(base, emotionOpts["temperature"].get<double>(), w);
    }
    static const char *keys[] = {"top_p", "presence_penalty",
                                 "frequency_penalty", "top_k", "min_p",
                                 "seed"};
    for (const char *key : keys) {
      if (!emotionOpts.contains(key) || !emotionOpts[key].is_number())
        continue;
      if (std::string(key) == "seed") {
        opts[key] = emotionOpts[key];
        continue;
      }
      if (opts.contains(key) && opts[key].is_number())
        opts[key] = blend(opts[key].get<double>(),
                          emotionOpts[key].get<double>(), w);
      else
        opts[key] = emotionOpts[key];
    }
  }
  const float memW = static_cast<float>(opts.value("memoryWeight", 1.0));
  const float gnnW = static_cast<float>(opts.value("gnnWeight", 1.0));
  const float scale = static_cast<float>(
      1.0 + 0.45 * std::min(1.0, static_cast<double>(pressure)) +
      0.20 * std::min(1.0, std::fabs(arousal)));
  if (memW > 0.f) opts["memoryWeight"] = memW * scale;
  if (gnnW > 0.f) opts["gnnWeight"] = gnnW * scale;
  auto scaleRows = [&](const char *key, float w) {
    if (!opts.contains(key) || !opts[key].is_array() || w == 1.f)
      return;
    for (auto &pkt : opts[key]) {
      if (!pkt.is_object() || !pkt.contains("rows") || !pkt["rows"].is_array())
        continue;
      for (auto &row : pkt["rows"]) {
        if (!row.is_array()) continue;
        for (auto &x : row) {
          if (x.is_number())
            x = x.get<double>() * static_cast<double>(w);
        }
      }
    }
  };
  scaleRows("context_units", memW * scale);
  scaleRows("gnn_units", gnnW * scale);
}

}  // namespace mission
}  // namespace phoenix
