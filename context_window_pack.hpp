/* context_window_pack.hpp - sliding-window prompt packing with pinned summaries
   Copyright (C) 2026 079 Project

   Packs long mission/chat context into a fixed token budget:
     [optional GNN summary | pinned] [text summary | pinned] [recent full | sliding]

   Modes:
     - "summary"          : mostly summary + a small recent window
     - "full_and_summary" : pinned summary of dropped head + as much recent full
                            text as fits in the remaining budget
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
  int summaryBudgetTokens{512}; /* pinned summary slot */
  int gnnBudgetTokens{256};     /* pinned GNN slot when enabled */
  int overheadTokens{256};      /* goal/tools/instructions overhead */
};

struct PackResult {
  std::string packed;
  size_t estimatedTokens{0};
  size_t fullCharsUsed{0};
  size_t droppedChars{0};
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
 * Pack full text (+ optional GNN summary) into a sliding window under ctx.
 *
 * Layout (fixed prefix, sliding suffix):
 *   === GNN summary (pinned) ===
 *   === Context summary (pinned) ===
 *   === Recent full text (sliding) ===
 */
inline PackResult packContext(const std::string &fullText,
                              const std::string &gnnSummary,
                              const PackOptions &opt) {
  PackResult out;
  const int usable = std::max(256, opt.ctxTokens - opt.replyReserveTokens - opt.overheadTokens);
  size_t remaining = static_cast<size_t>(usable);
  std::ostringstream body;

  std::string gnn;
  if (opt.includeGnnSummary && !gnnSummary.empty()) {
    gnn = fitTokensHead(gnnSummary, static_cast<size_t>(std::max(32, opt.gnnBudgetTokens)));
    if (estimateTokens(gnn) > remaining / 2)
      gnn = fitTokensHead(gnn, remaining / 2);
    if (!gnn.empty()) {
      body << "=== GNN summary (pinned) ===\n" << gnn << "\n\n";
      remaining -= std::min(remaining, estimateTokens(gnn) + 12);
      out.usedGnn = true;
    }
  }

  const bool wantSummary = (opt.mode == "summary" || opt.mode == "full_and_summary");
  size_t summaryBudget = static_cast<size_t>(std::max(64, opt.summaryBudgetTokens));
  if (opt.mode == "summary") {
    /* summary mode: give most of the remaining budget to the summary */
    summaryBudget = std::max(summaryBudget, remaining * 2 / 3);
  } else {
    summaryBudget = std::min(summaryBudget, remaining / 3);
  }

  std::string summary;
  if (wantSummary && !fullText.empty()) {
    summary = extractiveSummary(fullText, summaryBudget);
    if (!summary.empty()) {
      body << "=== Context summary (pinned, global view) ===\n" << summary << "\n\n";
      remaining -= std::min(remaining, estimateTokens(summary) + 16);
      out.usedSummary = true;
    }
  }

  if (opt.mode == "full_and_summary" || opt.mode == "summary") {
    /* Sliding recent full text in the leftover budget.
       summary mode keeps a smaller recent window so the model still sees
       the latest concrete wording. */
    size_t recentBudget = remaining;
    if (opt.mode == "summary")
      recentBudget = std::min(remaining, remaining / 3 + 64);
    if (recentBudget > 32 && !fullText.empty()) {
      const std::string recent = fitTokensTail(fullText, recentBudget);
      body << "=== Recent full text (sliding window) ===\n" << recent << "\n";
      out.fullCharsUsed = recent.size();
      if (fullText.size() > recent.size())
        out.droppedChars = fullText.size() - recent.size();
      remaining -= std::min(remaining, estimateTokens(recent) + 12);
    }
  } else {
    /* unknown mode: fall back to tail-only */
    const std::string recent = fitTokensTail(fullText, remaining);
    body << recent;
    out.fullCharsUsed = recent.size();
  }

  out.packed = body.str();
  out.estimatedTokens = estimateTokens(out.packed);
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
