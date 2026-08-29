/* context_window_pack.hpp - sliding-window packing with pinned summaries

   Two streams (do not concatenate onto one causal blob):
     causal: recent full text (sliding) — continuation chain
     rag:    pinned extractive summary of the dropped head + optional GNN text

   N-gram merge (2-3 token rows -> one unit) multiplies how much recent
   text fits in the same KV slots. Summaries stay RAG and do not consume
   causal positions.

   Modes:
     - "summary"          : mostly summary (RAG) + a small recent window
     - "full_and_summary" : pinned summary of dropped head + as much recent
                            full text as fits in the causal budget
*/
#pragma once

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace context {

struct PackOptions {
  std::string mode{"full_and_summary"}; /* summary | full_and_summary */
  bool includeGnnSummary{false};
  int ctxTokens{4096};          /* 4096 or 16384 typical */
  int replyReserveTokens{512};  /* leave room for model output */
  int summaryBudgetTokens{512}; /* pinned summary slot (RAG) */
  int gnnBudgetTokens{256};     /* pinned GNN slot when enabled (RAG) */
  int overheadTokens{256};      /* pin / template overhead on causal */
  int ngramMerge{2};            /* 0/1 = off; 2 or 3 = unit compaction */
};

struct PackResult {
  std::string packed;         /* legacy concat; do not put on causal tail */
  std::string recentFull;     /* causal short-term / resume source */
  std::string pinnedSummary;  /* RAG: dropped-head extractive */
  std::string gnnPinned;      /* RAG: GNN text if any */
  size_t estimatedTokens{0};
  size_t fullCharsUsed{0};
  size_t droppedChars{0};
  size_t causalTokenBudget{0};
  bool usedSummary{false};
  bool usedGnn{false};
};

/** Rough token estimate (mixed EN/ZH): ~3 chars/token. */
inline size_t estimateTokens(const std::string &s) {
  return s.empty() ? 0 : (s.size() + 2) / 3;
}

inline std::string takeTailChars(const std::string &s, size_t maxChars) {
  if (s.size() <= maxChars) return s;
  return s.substr(s.size() - maxChars);
}

inline std::string takeHeadChars(const std::string &s, size_t maxChars) {
  if (s.size() <= maxChars) return s;
  return s.substr(0, maxChars);
}

/**
 * Extractive summary: head snapshot + middle ellipsis + recent tail.
 * Avoids calling a second LLM during the mission tick (RDK latency).
 */
inline std::string extractiveSummary(const std::string &full, size_t budgetTokens) {
  if (full.empty() || budgetTokens < 32) return std::string();
  const size_t budgetChars = budgetTokens * 3;
  if (full.size() <= budgetChars) return full;
  const size_t head = budgetChars / 3;
  const size_t tail = budgetChars - head - 48;
  std::ostringstream ss;
  ss << takeHeadChars(full, head)
     << "\n...\n[earlier content omitted; " << (full.size() - head - tail)
     << " chars]\n...\n"
     << takeTailChars(full, tail);
  return ss.str();
}

/** Truncate to an approximate token budget (prefer keeping the end). */
inline std::string fitTokensTail(const std::string &s, size_t budgetTokens) {
  if (budgetTokens == 0) return std::string();
  return takeTailChars(s, budgetTokens * 3);
}

inline std::string fitTokensHead(const std::string &s, size_t budgetTokens) {
  if (budgetTokens == 0) return std::string();
  return takeHeadChars(s, budgetTokens * 3);
}

/**
 * Split full text (+ optional GNN summary) into causal recent + RAG pins.
 * Causal budget uses n-gram compaction: 4096 slots * ngram=2 ≈ 8k tokens
 * of recent draft. Pinned summaries never occupy those slots.
 */
inline PackResult packContext(const std::string &fullText,
                              const std::string &gnnSummary,
                              const PackOptions &opt) {
  PackResult out;
  const int ngram = (opt.ngramMerge >= 2) ? std::min(opt.ngramMerge, 3) : 1;
  const int usableSlots =
      std::max(256, opt.ctxTokens - opt.replyReserveTokens - opt.overheadTokens);
  const size_t causalBudget =
      static_cast<size_t>(usableSlots) * static_cast<size_t>(ngram);
  out.causalTokenBudget = causalBudget;
  std::ostringstream body;

  if (opt.includeGnnSummary && !gnnSummary.empty()) {
    out.gnnPinned = fitTokensHead(
        gnnSummary, static_cast<size_t>(std::max(32, opt.gnnBudgetTokens)));
    if (!out.gnnPinned.empty()) {
      body << "=== GNN summary (pinned) ===\n" << out.gnnPinned << "\n\n";
      out.usedGnn = true;
    }
  }

  const bool wantSummary =
      (opt.mode == "summary" || opt.mode == "full_and_summary");
  size_t recentBudget = causalBudget;
  if (opt.mode == "summary")
    recentBudget = std::max(static_cast<size_t>(64), causalBudget / 3);

  if (!fullText.empty() && recentBudget > 0) {
    out.recentFull = fitTokensTail(fullText, recentBudget);
    out.fullCharsUsed = out.recentFull.size();
    if (fullText.size() > out.recentFull.size())
      out.droppedChars = fullText.size() - out.recentFull.size();
    body << "=== Recent full text (sliding window) ===\n"
         << out.recentFull << "\n";
  }

  size_t summaryBudget = static_cast<size_t>(std::max(64, opt.summaryBudgetTokens));
  if (opt.mode == "summary")
    summaryBudget = std::max(summaryBudget, causalBudget * 2 / 3);
  if (wantSummary && out.droppedChars > 0) {
    const std::string droppedHead = takeHeadChars(fullText, out.droppedChars);
    out.pinnedSummary = extractiveSummary(droppedHead, summaryBudget);
    if (!out.pinnedSummary.empty()) {
      body << "=== Context summary (pinned, global view) ===\n"
           << out.pinnedSummary << "\n\n";
      out.usedSummary = true;
    }
  } else if (wantSummary && !fullText.empty() && opt.mode == "summary") {
    out.pinnedSummary = extractiveSummary(fullText, summaryBudget);
    if (!out.pinnedSummary.empty()) {
      body << "=== Context summary (pinned, global view) ===\n"
           << out.pinnedSummary << "\n\n";
      out.usedSummary = true;
    }
  }

  out.packed = body.str();
  out.estimatedTokens = estimateTokens(out.recentFull);
  return out;
}

inline PackOptions optionsFromJson(const nlohmann::json &j, const PackOptions &defaults) {
  PackOptions o = defaults;
  if (!j.is_object()) return o;
  if (j.contains("contextPack") && j["contextPack"].is_string()) {
    const std::string m = j["contextPack"].get<std::string>();
    if (m == "summary" || m == "full_and_summary") o.mode = m;
  }
  if (j.contains("includeGnnSummary") && j["includeGnnSummary"].is_boolean())
    o.includeGnnSummary = j["includeGnnSummary"].get<bool>();
  if (j.contains("ctxTokens") && j["ctxTokens"].is_number_integer()) {
    int c = j["ctxTokens"].get<int>();
    if (c < 2048) c = 2048;
    if (c > 32768) c = 32768;
    o.ctxTokens = c;
  }
  if (j.contains("ctxSize") && j["ctxSize"].is_number_integer()) {
    int c = j["ctxSize"].get<int>();
    if (c < 2048) c = 2048;
    if (c > 32768) c = 32768;
    o.ctxTokens = c;
  }
  if (j.contains("ngramMerge") && j["ngramMerge"].is_number_integer())
    o.ngramMerge = j["ngramMerge"].get<int>();
  if (j.contains("ngram_merge") && j["ngram_merge"].is_number_integer())
    o.ngramMerge = j["ngram_merge"].get<int>();
  if (o.ngramMerge < 0) o.ngramMerge = 0;
  if (o.ngramMerge > 3) o.ngramMerge = 3;
  /* Scale pinned/sliding budgets with ctx: 16k keeps summary + recent full. */
  if (o.ctxTokens >= 12000) {
    o.summaryBudgetTokens = std::max(o.summaryBudgetTokens, 1536);
    o.gnnBudgetTokens = std::max(o.gnnBudgetTokens, 384);
    o.replyReserveTokens = std::max(o.replyReserveTokens, 768);
    o.overheadTokens = std::max(o.overheadTokens, 384);
  } else if (o.ctxTokens >= 6000) {
    o.summaryBudgetTokens = std::max(o.summaryBudgetTokens, 768);
    o.gnnBudgetTokens = std::max(o.gnnBudgetTokens, 320);
  }
  return o;
}

}  // namespace context
}  // namespace phoenix
