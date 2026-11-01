/* memetic_existence.hpp - Official-path harmless meme existence lab

   Memes are graph units on the official MemeBarrier / KVM / GNN path
   (wikitext ingest via the same clustering as
   RuntimeState::buildMemeSequenceFromTokens). A sample is a meme id,
   not a carrier sentence and not an invented nonce.

   The graph stores an ordered (word, α) index, not a replicator.
   α(w|m) = tf(w,m)/df(w) changes when the collection changes, so the
   same copied sentence does not have a stable weight set. α locates
   a carrier. A meme, when the word is kept, is a later copy whose
   contiguous token fidelity clears Eigen's threshold. That test is
   judgeCopy in replica_trace.hpp. Old snapshots with no TF load as
   tf=1, so α on those snapshots is collection 1/df.
   Query seeds allocate each word as a distribution over its memes,
   conditioned on how much of the rest of the query already matches
   that meme, after collection-IDF scales the word's total budget.
   Product mapping then keeps the typical set (IPR floor) so a
   hub-word spray does not open the walk window. present is GNN
   diffusion + lift (plus the existing stop-only content check).
   No extra present gates.

   Existence serial (the real test):
     screen memes on the frozen instrument graph
     -> composeCarrierRag: realize an ordered sentence carrier from
        the meme's typical (word, α) source sentences, not a KVM
        bag join and not the first N wiki units; among presenting
        sentences pick the one where the target holds the largest
        share of present-meme mass (rank-0 preferred). Extra
        sentences may stay as mutation slots only if they do not
        steal present-mass from the target.
     -> detectActivatedMemes lists every meme a sentence lights
        (isotope sentence→memes); RagCarrier.sources lists sentences
        that light the target (isotope meme→sentences). present of
        a named id is still GNN+lift
     -> empty-session raw /completion of the carrier text only
        (no chat template, no system/task/persona wrapper)
     -> detectMemeInText on the same frozen graph (do not ingest the
        model output into the instrument; do not write the meme id back)
   Wipe-reingest asks whether a fresh graph grew the same index, not
   whether a replicator was copied. Bag-hash identity is neither.
     carrierOfMeme remains the KVM word bag (diagnostic / fallback).

   expressActivated can name a payload when the cluster is active. That
   is a lab decoder, not llama-server, and is not an existence proof.
   Two-AI contagion is not part of this lab. */
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "replica_trace.hpp"

namespace phoenix {
namespace memetic {

/* Harmless distinctive payload: not an instruction, URL, or shell. */
inline constexpr const char *kHarmlessNonce = "qx7celadon47";

struct Params {
  int ngramMin{3};
  int ngramMax{14};
  int minOverlap{2};
  int maxMemeWords{100};
  int diffusionRounds{5};
};

/* Seed increment: keep the word (including stopwords), scale by how many
   memes already map it. Used as α(w|m)=tf/df and as the fallback when
   a word has no query-conditioned competitors. */
inline double mappingDegreeSeed(double tokenWeight, int df) {
  if (df <= 0 || tokenWeight <= 0.0)
    return 0.0;
  return tokenWeight / static_cast<double>(df);
}

inline double mappingWordAlpha(int tf, int df) {
  if (tf <= 0 || df <= 0)
    return 0.0;
  return static_cast<double>(tf) / static_cast<double>(df);
}

/* Collection IDF in (0,1]: unique words keep full budget, hub words
   deposit little. Not a present gate — it only scales q(w). */
inline double mappingIdfScale(int df, int memeCount) {
  if (df <= 1 || memeCount <= 1)
    return 1.0;
  const double n = static_cast<double>(std::max(memeCount, df));
  const double d = static_cast<double>(df);
  const double den = std::log(1.0 + n);
  if (den <= 1e-12)
    return 1.0;
  return std::log(1.0 + n / d) / den;
}

struct MappingWordHit {
  std::string memeId;
  double alpha{0.0};
};

/* memeId -> (source word or ng:span -> mass). Isotope label, not a gate. */
using MappingMassFrom =
    std::unordered_map<std::string, std::unordered_map<std::string, double>>;

inline std::string mappingNewQueryId() {
  static std::atomic<std::uint64_t> seq{1};
  const auto n = seq.fetch_add(1, std::memory_order_relaxed);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "q%llu",
                static_cast<unsigned long long>(n));
  return buf;
}

inline std::string mappingTokenId(const std::string &queryId, int idx) {
  return queryId + ".t" + std::to_string(idx);
}

inline std::string mappingSeedId(const std::string &queryId, int idx) {
  return queryId + ".s" + std::to_string(idx);
}

inline std::string mappingFormatFrom(const std::string &memeId,
                                     const MappingMassFrom &from,
                                     int cap = 6) {
  auto it = from.find(memeId);
  if (it == from.end() || it->second.empty())
    return "via=walk";
  std::vector<std::pair<std::string, double>> rows(it->second.begin(),
                                                   it->second.end());
  std::sort(rows.begin(), rows.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  std::string out;
  const int n = std::min(cap, static_cast<int>(rows.size()));
  for (int i = 0; i < n; ++i) {
    if (i)
      out += ",";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", rows[static_cast<size_t>(i)].second);
    out += rows[static_cast<size_t>(i)].first;
    out += ":";
    out += buf;
  }
  return out;
}

/* First GNN layer: word → meme. A word's mass is a distribution over
   memes that contain it, weighted by α(w|m)·(ε + support_rest[m]).
   support is Σ q(w')α(w'|m) over the query. Unique names pull shared
   words toward their meme; a lone hub word stays spread by α.
   q(w) is first scaled by collection IDF so high-df tokens deposit
   little total mass. Optional `from` records isotope mass per
   (meme, source word). Optional `memeCount` is the collection size
   for IDF (falls back to |hits|). */
inline void mappingAddConditionedSeeds(
    const std::unordered_map<std::string, double> &queryMass,
    const std::unordered_map<std::string, std::vector<MappingWordHit>> &hits,
    std::unordered_map<std::string, double> *seeds,
    MappingMassFrom *from = nullptr, int memeCount = 0) {
  if (!seeds)
    return;
  int nColl = memeCount;
  if (nColl <= 0) {
    std::unordered_set<std::string> uniq;
    for (const auto &row : hits) {
      for (const auto &h : row.second) {
        if (!h.memeId.empty())
          uniq.insert(h.memeId);
      }
    }
    nColl = static_cast<int>(uniq.size());
  }
  std::unordered_map<std::string, double> qEff;
  for (const auto &wq : queryMass) {
    if (wq.second <= 0.0)
      continue;
    int df = 0;
    auto hitIt = hits.find(wq.first);
    if (hitIt != hits.end())
      df = static_cast<int>(hitIt->second.size());
    const double scaled = wq.second * mappingIdfScale(df, nColl);
    if (scaled > 0.0)
      qEff[wq.first] = scaled;
  }
  auto add = [&](const std::string &mid, const std::string &word, double m) {
    if (mid.empty() || m <= 0.0)
      return;
    (*seeds)[mid] += m;
    if (from && !word.empty())
      (*from)[mid][word] += m;
  };
  std::unordered_map<std::string, double> support;
  for (const auto &wq : qEff) {
    auto it = hits.find(wq.first);
    if (it == hits.end())
      continue;
    for (const auto &h : it->second) {
      if (h.memeId.empty() || h.alpha <= 0.0)
        continue;
      support[h.memeId] += wq.second * h.alpha;
    }
  }
  const double eps = 1e-6;
  for (const auto &wq : qEff) {
    auto it = hits.find(wq.first);
    if (it == hits.end() || it->second.empty())
      continue;
    double z = 0.0;
    std::vector<double> score;
    score.reserve(it->second.size());
    for (const auto &h : it->second) {
      double rest = 0.0;
      auto sit = support.find(h.memeId);
      if (sit != support.end())
        rest = sit->second - wq.second * h.alpha;
      if (rest < 0.0)
        rest = 0.0;
      const double s = std::max(0.0, h.alpha) * (eps + rest);
      score.push_back(s);
      z += s;
    }
    if (z <= 1e-18) {
      z = 0.0;
      for (const auto &h : it->second)
        z += std::max(0.0, h.alpha);
      if (z <= 1e-18)
        continue;
      for (const auto &h : it->second)
        add(h.memeId, wq.first, wq.second * (std::max(0.0, h.alpha) / z));
      continue;
    }
    for (size_t i = 0; i < it->second.size(); ++i)
      add(it->second[i].memeId, wq.first, wq.second * (score[i] / z));
  }
}

/* Typical set of a seed distribution: drop the thin tail below
   total / (ipr + √ipr). Peaked queries keep the mass-bearing memes;
   a hub-word spray does not open a radius-2 window on the whole
   graph. Not a present gate — walk geometry only. Always keeps the
   current max so a query is never emptied. */
inline void mappingConcentrateSeeds(
    std::unordered_map<std::string, double> *seeds,
    std::vector<std::string> *order = nullptr,
    MappingMassFrom *from = nullptr) {
  if (!seeds || seeds->empty())
    return;
  double total = 0.0;
  std::string best;
  double bestMass = -1.0;
  for (const auto &kv : *seeds) {
    if (kv.second > 0.0)
      total += kv.second;
    if (kv.second > bestMass) {
      bestMass = kv.second;
      best = kv.first;
    }
  }
  if (total <= 0.0 || best.empty())
    return;
  double sumSq = 0.0;
  for (const auto &kv : *seeds) {
    if (kv.second <= 0.0)
      continue;
    const double p = kv.second / total;
    sumSq += p * p;
  }
  if (sumSq <= 1e-18)
    return;
  const double ipr = 1.0 / sumSq;
  const double floor =
      total / (ipr + std::sqrt(std::max(1.0, ipr)));
  std::vector<std::string> drop;
  drop.reserve(seeds->size());
  for (const auto &kv : *seeds) {
    if (kv.first == best)
      continue;
    if (kv.second < floor)
      drop.push_back(kv.first);
  }
  for (const auto &id : drop) {
    seeds->erase(id);
    if (from)
      from->erase(id);
  }
  if (order) {
    std::vector<std::string> kept;
    kept.reserve(order->size());
    for (const auto &id : *order) {
      if (seeds->count(id))
        kept.push_back(id);
    }
    *order = std::move(kept);
  }
}

/* One index entry. id addresses a carrier. text is its word order.
   alpha inside the scorer is tf/df on this collection. Matching this
   index does not make the entry a meme. */
struct OrderedMeme {
  std::string id;
  std::string text;
};

struct OrderedMemeScore {
  std::string id;
  double mass{0.0};
  double lift{0.0};
  double orderLift{1.0};
  int overlap{0};
  bool present{false};
};

/* Declared here, defined after kGnnPresentLift. Present requires the
   existing lift cut (mass over the high-df prior) and Kendall tau > 0
   on the meme's word order. A tie or a reversal is not present. */
inline std::vector<OrderedMemeScore>
scoreOrderedMemes(const std::string &query,
                  const std::vector<OrderedMeme> &memes);

/* Latent tool desire. The model is not shown these niches. This is an
   index: lift over the high-df prior at least 2, overlap at least 2,
   and Kendall tau > 0. It is not judgeCopy, and a hit is not a meme.
   A single anchor word, or the same words in the wrong order, does not
   open a call. Empty name means no tool. */
struct LatentTool {
  std::string name;
  double share{0.0};
  double ipr{0.0};
  int hits{0};
};

inline LatentTool inferLatentTool(const std::string &text) {
  LatentTool none;
  if (text.size() < 8 || text.size() > 600)
    return none;
  int rawTokens = 0;
  {
    std::string cur;
    auto flush = [&]() {
      if (!cur.empty())
        ++rawTokens;
      cur.clear();
    };
    for (unsigned char c : text) {
      if (std::isalnum(c))
        cur.push_back(static_cast<char>(std::tolower(c)));
      else
        flush();
    }
    flush();
  }
  if (rawTokens < 2 || rawTokens > 40)
    return none;
  /* Digits and operators become ordered tokens. "0.0.0 b 1.0." has
     numbers and no operator, so it does not match number-operator-number. */
  std::string norm;
  std::string num;
  auto flushNum = [&]() {
    bool digit = false;
    for (unsigned char c : num) {
      if (std::isdigit(c))
        digit = true;
    }
    if (digit)
      norm += " number ";
    num.clear();
  };
  for (unsigned char c : text) {
    if (std::isdigit(c) || c == '.') {
      num.push_back(static_cast<char>(c));
      continue;
    }
    flushNum();
    if (c == '+' || c == '*' || c == '/' || c == '=')
      norm += " operator ";
    else if (std::isalpha(c))
      norm.push_back(static_cast<char>(std::tolower(c)));
    else
      norm.push_back(' ');
  }
  flushNum();
  const std::vector<OrderedMeme> memes = {
      {"math", "derivative squared polynomial equation factorial matrix"},
      {"math-expr", "number operator number"},
      {"search", "according wikipedia citation bibliography reference"},
      {"replicate", "delegate subgoal helper summon"},
  };
  const auto scored = scoreOrderedMemes(norm, memes);
  std::string tool;
  const OrderedMemeScore *best = nullptr;
  for (const auto &s : scored) {
    if (!s.present)
      continue;
    const std::string name = s.id == "math-expr" ? "math" : s.id;
    if (!tool.empty() && name != tool)
      return none;
    tool = name;
    if (!best || s.lift > best->lift)
      best = &s;
  }
  if (!best)
    return none;
  LatentTool got;
  got.name = tool;
  got.share = best->lift;
  got.ipr = best->orderLift;
  got.hits = best->overlap;
  return got;
}

/* Exclusive carrier for a source edit. The score-formula words stay in
   the goal prefix; they are a hub spray and must not be the text the
   model continues. The carrier is one file span, cut on a line boundary.
   Empty when the path is unsafe or the span is missing. */
inline std::string exclusiveEditCarrier(const std::string &path,
                                       const std::string &span) {
  if (path.empty() || span.empty())
    return std::string();
  if (path.find("..") != std::string::npos)
    return std::string();
  if (path.find('\n') != std::string::npos ||
      path.find('\r') != std::string::npos)
    return std::string();
  std::string body = span;
  if (body.size() > 1400) {
    body.resize(1400);
    const auto nl = body.rfind('\n');
    if (nl != std::string::npos && nl > 200)
      body.resize(nl);
  }
  return "FILE " + path + "\n" + body +
         "\nEdit only the file named above. One diff, or FILE, OLD, and NEW "
         "for one exact span of that text. The measure is a number, not a "
         "program.\n";
}

/* Span of a chosen file that actually carries the description.
   The file is already selected. This only places the 1400-byte
   window. A window starts at byte 0 or after a newline, then ends
   on a newline once it passes 200 bytes, and never passes 1400.
   Score is the number of distinct description tokens, lowercase
   alphanumeric runs of length at least 3, that occur as whole
   tokens in the window. The earliest maximum wins. Fewer than 2
   shared tokens returns empty: the tail and a processMessage
   anchor are not substitutes. */
inline std::string latentCarrierSpan(const std::string &description,
                                    const std::string &body) {
  std::vector<std::string> need;
  {
    std::string cur;
    std::unordered_set<std::string> seen;
    auto flush = [&]() {
      if (cur.size() >= 3 && seen.insert(cur).second)
        need.push_back(cur);
      cur.clear();
    };
    for (unsigned char c : description) {
      if (std::isalnum(c))
        cur.push_back(static_cast<char>(std::tolower(c)));
      else
        flush();
    }
    flush();
  }
  if (body.empty() || need.size() < 2)
    return std::string();
  auto wordsIn = [](const std::string &text) {
    std::unordered_set<std::string> have;
    std::string cur;
    auto flush = [&]() {
      if (cur.size() >= 3)
        have.insert(cur);
      cur.clear();
    };
    for (unsigned char c : text) {
      if (std::isalnum(c))
        cur.push_back(static_cast<char>(std::tolower(c)));
      else
        flush();
    }
    flush();
    return have;
  };
  std::string best;
  int bestScore = 0;
  auto consider = [&](std::size_t begin) {
    if (begin >= body.size())
      return;
    std::string window = body.substr(begin, 1400);
    if (window.size() == 1400) {
      const auto nl = window.rfind('\n');
      if (nl != std::string::npos && nl > 200)
        window.resize(nl);
    }
    const auto have = wordsIn(window);
    int score = 0;
    for (const auto &w : need)
      if (have.count(w))
        ++score;
    if (score > bestScore) {
      bestScore = score;
      best = std::move(window);
    }
  };
  consider(0);
  for (std::size_t i = 0; i < body.size(); ++i) {
    if (body[i] == '\n')
      consider(i + 1);
  }
  if (bestScore < 2)
    return std::string();
  return best;
}

/* Repository explore. A path is present only when the goal lifts it
   over the collection's high-df prior and the goal's word order agrees
   with the path. A word that sits in many paths cannot open one file.
   Empty means the tool looked and did not find one ordered carrier. */
inline std::vector<std::string> rankExplorePaths(
    const std::string &goal, const std::vector<std::string> &paths, int keep) {
  std::vector<std::string> none;
  if (goal.empty() || paths.empty() || keep < 1)
    return none;
  std::vector<OrderedMeme> memes;
  memes.reserve(paths.size());
  for (const auto &path : paths)
    memes.push_back(OrderedMeme{path, path});
  const auto scored = scoreOrderedMemes(goal, memes);
  std::vector<const OrderedMemeScore *> present;
  for (const auto &s : scored) {
    if (s.present)
      present.push_back(&s);
  }
  if (present.empty())
    return none;
  if (keep == 1 && present.size() != 1)
    return none;
  std::sort(present.begin(), present.end(),
            [](const OrderedMemeScore *a, const OrderedMemeScore *b) {
              return a->mass * a->orderLift > b->mass * b->orderLift;
            });
  std::vector<std::string> out;
  for (const auto *s : present) {
    out.push_back(s->id);
    if (static_cast<int>(out.size()) >= keep)
      break;
  }
  return out;
}

inline constexpr double kGnnPresentPeakFrac = 0.05;
inline constexpr double kGnnPresentLift = 2.0;

/* One latent edit from a base description. path is empty when no file
   is eligible or the top two masses tie. presentCount is how many files
   were eligible. The description is the context; file profiles are the
   items. Eligible means overlap >= 2, lift >= kGnnPresentLift, and
   mass > 0. Forward word order is not required: a conception is not a
   copy of the source, so Kendall tau is not a veto. The pick is the
   unique maximum mass. A filename does not have to occur in the
   description. This does not copy a file. */
struct LatentEdit {
  std::string path;
  double mass{0.0};
  double orderLift{0.0};
  int overlap{0};
  int presentCount{0};
};

inline LatentEdit rankLatentEdit(const std::string &description,
                                 const std::vector<OrderedMeme> &files,
                                 const std::vector<int> &fecundity = {}) {
  LatentEdit out;
  const auto scored = scoreOrderedMemes(description, files);
  const OrderedMemeScore *best = nullptr;
  const OrderedMemeScore *second = nullptr;
  for (const auto &s : scored) {
    if (!(s.mass > 0.0 && s.lift >= kGnnPresentLift && s.overlap >= 2))
      continue;
    ++out.presentCount;
    if (!best || s.mass > best->mass) {
      second = best;
      best = &s;
    } else if (!second || s.mass > second->mass) {
      second = &s;
    }
  }
  if (!best)
    return out;
  if (second && !(best->mass > second->mass)) {
    /* Mass does not see fecundity: an exclusive word's share is 1
       however often it is repeated. A tie is broken only by a unique
       strictly positive fecundity. Equal counts, or a missing vector,
       still open no file. */
    if (fecundity.size() == scored.size()) {
      int topF = 0;
      const OrderedMemeScore *pick = nullptr;
      int ties = 0;
      for (size_t i = 0; i < scored.size(); ++i) {
        const auto &s = scored[i];
        if (!(s.mass > 0.0 && s.lift >= kGnnPresentLift && s.overlap >= 2))
          continue;
        if (best->mass > s.mass || s.mass > best->mass)
          continue;
        const int f = fecundity[i];
        if (pick == nullptr || f > topF) {
          pick = &s;
          topF = f;
          ties = 1;
        } else if (f == topF) {
          ++ties;
        }
      }
      if (ties == 1 && pick != nullptr && topF > 0) {
        out.path = pick->id;
        out.mass = pick->mass;
        out.orderLift = pick->orderLift;
        out.overlap = pick->overlap;
        return out;
      }
    }
    return out;
  }
  out.path = best->id;
  out.mass = best->mass;
  out.orderLift = best->orderLift;
  out.overlap = best->overlap;
  return out;
}

/* One witnessed gloss. A short description that already selected a
   file contributes its content words, except words that already sit
   in some candidate file. The union is one insertion. A later text
   made only of those words can then pass the same ranker, which
   still has no new output column. More than 40 content tokens does
   not write, and the same word is not written twice. The cap is 32
   words per file. */
inline std::vector<std::string> witnessedAliases(
    const std::string &description,
    const std::vector<OrderedMeme> &files, const std::string &chosenId,
    const std::vector<std::string> &already) {
  auto tokenize = [](const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
      if (cur.size() >= 3)
        out.push_back(cur);
      cur.clear();
    };
    for (unsigned char c : s) {
      if (std::isalnum(c))
        cur.push_back(static_cast<char>(std::tolower(c)));
      else
        flush();
    }
    flush();
    return out;
  };
  std::vector<std::string> out = already;
  const auto desc = tokenize(description);
  if (desc.size() < 2 || desc.size() > 40)
    return out;
  std::unordered_set<std::string> inAny;
  for (const auto &file : files)
    for (const auto &w : tokenize(file.text))
      inAny.insert(w);
  std::unordered_set<std::string> have(out.begin(), out.end());
  for (const auto &w : desc) {
    if (static_cast<int>(out.size()) >= 32)
      break;
    if (have.count(w) || inAny.count(w))
      continue;
    if (chosenId.empty())
      continue;
    out.push_back(w);
    have.insert(w);
  }
  return out;
}

inline std::vector<std::string> glossTokens(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    if (cur.size() >= 3)
      out.push_back(cur);
    cur.clear();
  };
  for (unsigned char c : text) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else
      flush();
  }
  flush();
  return out;
}

inline bool condFunctionWord(const std::string &w) {
  return w == "the" || w == "and" || w == "for" || w == "that" || w == "this" ||
         w == "with" || w == "from" || w == "are" || w == "was" || w == "were" ||
         w == "been" || w == "have" || w == "not" || w == "but" || w == "you" ||
         w == "your" || w == "into" || w == "about" || w == "would" ||
         w == "could" || w == "should" || w == "there" || w == "their" ||
         w == "which" || w == "when" || w == "where" || w == "what" ||
         w == "will" || w == "shall" || w == "than" || w == "then" ||
         w == "also" || w == "only" || w == "such" || w == "some" ||
         w == "any" || w == "all" || w == "can" || w == "may";
}

/* Two descriptions are the same hit when their content-token sets are
   almost the same. A repeat of one sentence is not a second witness:
   the promotion rule below needs two distinct observations. */
inline double condDescriptionSimilarity(const std::string &a,
                                        const std::string &b) {
  std::unordered_set<std::string> sa;
  std::unordered_set<std::string> sb;
  for (const auto &w : glossTokens(a))
    if (!condFunctionWord(w))
      sa.insert(w);
  for (const auto &w : glossTokens(b))
    if (!condFunctionWord(w))
      sb.insert(w);
  if (sa.empty() && sb.empty())
    return 1.0;
  int inter = 0;
  for (const auto &w : sa)
    if (sb.count(w))
      ++inter;
  const int uni = static_cast<int>(sa.size() + sb.size()) - inter;
  if (uni <= 0)
    return 1.0;
  return static_cast<double>(inter) / static_cast<double>(uni);
}

inline bool condNearDuplicate(const std::string &a, const std::string &b) {
  return condDescriptionSimilarity(a, b) >= 0.9;
}

/* Recent-hit window: a description is independent only when it is not a
   near duplicate of ANY of the last few hits. Checking only the last hit
   would let A, B, A count A twice. */
inline bool condRecentIndependent(const std::vector<std::string> &recent,
                                  const std::string &current) {
  for (const auto &prev : recent) {
    if (condNearDuplicate(prev, current))
      return false;
  }
  return true;
}

/* A conditional hit is not yet an index term. The first short
   description only parks its new content words. Function words are
   not parked. Words already in a candidate file are not parked.
   A later description that shares at least two parked words promotes
   that intersection. Calendar paraphrases usually share no two
   content words, so a second hit with a smaller intersection instead
   promotes up to four parked words and four new words. That is one
   insertion, W += y x^T on those words, not an extra training epoch.

   Two observations must be distinct: repeating one description is not
   a second witness (condNearDuplicate). The mixed branch requires at
   least two words from each observation, otherwise it only parks the
   new words: a promotion that leaves one side with a single word
   cannot route that side later, so it would be an update the local
   student cannot use. */
inline int confirmCondWords(const std::string &description,
                            const std::vector<OrderedMeme> &files,
                            std::vector<std::string> &active,
                            std::vector<std::string> &pending,
                            const std::string *lastDescription = nullptr) {
  if (lastDescription != nullptr && !lastDescription->empty() &&
      condNearDuplicate(*lastDescription, description))
    return 0;
  const auto eligible = witnessedAliases(description, files, "cond", active);
  std::unordered_set<std::string> already(active.begin(), active.end());
  std::vector<std::string> fresh;
  for (const auto &w : eligible) {
    if (already.count(w) || condFunctionWord(w))
      continue;
    fresh.push_back(w);
  }
  std::unordered_set<std::string> parked(pending.begin(), pending.end());
  std::vector<std::string> shared;
  for (const auto &w : fresh) {
    if (parked.count(w))
      shared.push_back(w);
  }
  auto dropMoved = [&](const std::unordered_set<std::string> &moved) {
    std::vector<std::string> kept;
    for (const auto &w : pending) {
      if (!moved.count(w))
        kept.push_back(w);
    }
    pending.swap(kept);
  };
  if (shared.size() >= 2 &&
      static_cast<int>(active.size()) + 2 <= 32) {
    std::unordered_set<std::string> moved;
    for (const auto &w : shared) {
      if (static_cast<int>(active.size()) >= 32)
        break;
      active.push_back(w);
      moved.insert(w);
    }
    dropMoved(moved);
    return static_cast<int>(moved.size());
  }
  if (!pending.empty() && !fresh.empty() &&
      static_cast<int>(active.size()) < 32) {
    /* At least two words from each observation, or this is not yet a
       usable two-sided witness. */
    const int room = std::min(32 - static_cast<int>(active.size()), 8);
    const int fromParkWanted = std::min(4, static_cast<int>(pending.size()));
    const int fromFreshWanted = std::min(4, static_cast<int>(fresh.size()));
    if (fromParkWanted >= 2 && fromFreshWanted >= 2 &&
        fromParkWanted + fromFreshWanted <= std::max(room, 4)) {
      std::unordered_set<std::string> moved;
      int fromPark = 0;
      for (const auto &w : pending) {
        if (fromPark == 4 || static_cast<int>(active.size()) >= 32)
          break;
        active.push_back(w);
        moved.insert(w);
        ++fromPark;
      }
      int fromFresh = 0;
      std::unordered_set<std::string> act(active.begin(), active.end());
      for (const auto &w : fresh) {
        if (fromFresh == 4 || static_cast<int>(active.size()) >= 32)
          break;
        if (!act.insert(w).second)
          continue;
        active.push_back(w);
        ++fromFresh;
      }
      dropMoved(moved);
      return fromPark + fromFresh;
    }
    /* Not enough on both sides: keep parking, do not publish. */
    std::unordered_set<std::string> placed;
    for (const auto &w : fresh) {
      if (static_cast<int>(pending.size()) >= 32)
        break;
      if (placed.insert(w).second)
        pending.push_back(w);
    }
    return 0;
  }
  for (const auto &w : fresh) {
    if (static_cast<int>(pending.size()) >= 32)
      break;
    if (parked.insert(w).second)
      pending.push_back(w);
  }
  return 0;
}

inline void appendNeighborPairs(const std::string &description,
                                const std::unordered_set<std::string> &justAdded,
                                std::vector<std::string> &active);

/* Words that already sit in two or more candidate texts cannot be
   exclusive aliases: tf/df splits them and the lexical rank ties.
   The side table is a linear score, score_f = sum_w z_{f,w} x_w,
   with z in {0,1}. The first teacher hit only parks those shared
   words. The second hit writes z=1 for the same intersection rule
   used by confirmCondWords, or up to four parked and four new words
   when the intersection is smaller. That write is one saturating
   step. It does not enter the lexical profile and it does not admit
   a meme. A later query needs a unique score of at least 2. */
inline int confirmBlockedLinks(const std::string &description,
                               const std::vector<OrderedMeme> &files,
                               std::vector<std::string> &active,
                               std::vector<std::string> &pending) {
  std::unordered_map<std::string, int> df;
  for (const auto &file : files) {
    std::unordered_set<std::string> seen;
    for (const auto &w : glossTokens(file.text)) {
      if (seen.insert(w).second)
        df[w] += 1;
    }
  }
  std::unordered_set<std::string> already(active.begin(), active.end());
  std::vector<std::string> fresh;
  std::unordered_set<std::string> freshSeen;
  for (const auto &w : glossTokens(description)) {
    if (condFunctionWord(w) || already.count(w) || df[w] < 2)
      continue;
    if (freshSeen.insert(w).second)
      fresh.push_back(w);
  }
  std::unordered_set<std::string> parked(pending.begin(), pending.end());
  std::vector<std::string> shared;
  for (const auto &w : fresh) {
    if (parked.count(w))
      shared.push_back(w);
  }
  auto dropMoved = [&](const std::unordered_set<std::string> &moved) {
    std::vector<std::string> kept;
    for (const auto &w : pending) {
      if (!moved.count(w))
        kept.push_back(w);
    }
    pending.swap(kept);
  };
  if (shared.size() >= 2 &&
      static_cast<int>(active.size()) + 2 <= 32) {
    std::unordered_set<std::string> moved;
    for (const auto &w : shared) {
      if (static_cast<int>(active.size()) >= 32)
        break;
      active.push_back(w);
      moved.insert(w);
    }
    dropMoved(moved);
    appendNeighborPairs(description, moved, active);
    return static_cast<int>(moved.size());
  }
  if (!pending.empty() && !fresh.empty() &&
      static_cast<int>(active.size()) < 32) {
    std::unordered_set<std::string> moved;
    int fromPark = 0;
    for (const auto &w : pending) {
      if (fromPark == 4 || static_cast<int>(active.size()) >= 32)
        break;
      active.push_back(w);
      moved.insert(w);
      ++fromPark;
    }
    int fromFresh = 0;
    std::unordered_set<std::string> act(active.begin(), active.end());
    for (const auto &w : fresh) {
      if (fromFresh == 4 || static_cast<int>(active.size()) >= 32)
        break;
      if (!act.insert(w).second)
        continue;
      active.push_back(w);
      ++fromFresh;
    }
    dropMoved(moved);
    appendNeighborPairs(description, moved, active);
    return fromPark + fromFresh;
  }
  for (const auto &w : fresh) {
    if (static_cast<int>(pending.size()) >= 32)
      break;
    if (parked.insert(w).second)
      pending.push_back(w);
  }
  return 0;
}

/* A neighbor pair is two content tokens with nothing between them.
   The stored key is "left+right". It is a replicator span of length 2,
   not an extra unigram. Pairs live in the same link vector.

   Longer contiguous spans of the witnessed words (length 3 and 4) are
   stored as "a>b>c" / "a>b>c>d". They are the conjunctive hidden units
   of HiddenLinkNet: a unit fires only when the whole ordered span is
   present, which a sum of unigram indicators cannot express. */
inline void appendNeighborPairs(const std::string &description,
                                const std::unordered_set<std::string> &justAdded,
                                std::vector<std::string> &active) {
  const auto tokens = glossTokens(description);
  std::unordered_set<std::string> have(active.begin(), active.end());
  auto spansAllowed = [&](size_t begin, size_t len) {
    for (size_t k = 0; k < len; ++k) {
      const std::string &w = tokens[begin + k];
      if (condFunctionWord(w) || !justAdded.count(w))
        return false;
    }
    return true;
  };
  for (size_t i = 1; i < tokens.size(); ++i) {
    if (static_cast<int>(active.size()) >= 32)
      break;
    const std::string &left = tokens[i - 1];
    const std::string &right = tokens[i];
    if (condFunctionWord(left) || condFunctionWord(right))
      continue;
    if (!justAdded.count(left) || !justAdded.count(right))
      continue;
    const std::string key = left + "+" + right;
    if (have.insert(key).second && static_cast<int>(active.size()) < 32)
      active.push_back(key);
  }
  for (size_t len = 3; len <= 4; ++len) {
    for (size_t begin = 0; begin + len <= tokens.size(); ++begin) {
      if (static_cast<int>(active.size()) >= 32)
        break;
      if (!spansAllowed(begin, len))
        continue;
      std::string key = tokens[begin];
      for (size_t k = 1; k < len; ++k) {
        key.push_back('>');
        key += tokens[begin + k];
      }
      if (have.insert(key).second)
        active.push_back(key);
    }
  }
}

/* HiddenLinkNet score. The hidden units are the conjunctive keys stored
   in the row: length-2 pairs (left+right) and ordered spans of length
   3..4 (a>b>c). The output layer is linear with weight 1 on every unit
   the row wrote, so the score is the number of matched units. This is a
   one-hidden-layer network with fixed binary hidden units and a trained
   linear readout (the write is the one-step update). It is used only
   after the unigram and pair stages tie; a unique unigram or pair score
   never reaches it. */
inline int hiddenLinkScore(const std::string &description,
                           const std::vector<std::string> &links) {
  std::unordered_set<std::string> units(links.begin(), links.end());
  std::unordered_set<std::string> seen;
  const auto tokens = glossTokens(description);
  int n = 0;
  for (size_t i = 1; i < tokens.size(); ++i) {
    if (condFunctionWord(tokens[i - 1]) || condFunctionWord(tokens[i]))
      continue;
    const std::string key = tokens[i - 1] + "+" + tokens[i];
    if (units.count(key) && seen.insert(key).second)
      ++n;
  }
  for (size_t len = 3; len <= 4; ++len) {
    for (size_t begin = 0; begin + len <= tokens.size(); ++begin) {
      bool ok = true;
      for (size_t k = 0; k < len; ++k) {
        if (condFunctionWord(tokens[begin + k])) {
          ok = false;
          break;
        }
      }
      if (!ok)
        continue;
      std::string key = tokens[begin];
      for (size_t k = 1; k < len; ++k) {
        key.push_back('>');
        key += tokens[begin + k];
      }
      if (units.count(key) && seen.insert(key).second)
        ++n;
    }
  }
  return n;
}

inline int blockedPairScore(const std::string &description,
                            const std::vector<std::string> &links) {
  std::unordered_set<std::string> pairs;
  for (const auto &link : links) {
    if (link.find('+') != std::string::npos)
      pairs.insert(link);
  }
  if (pairs.empty())
    return 0;
  const auto tokens = glossTokens(description);
  std::unordered_set<std::string> seen;
  int n = 0;
  for (size_t i = 1; i < tokens.size(); ++i) {
    if (condFunctionWord(tokens[i - 1]) || condFunctionWord(tokens[i]))
      continue;
    const std::string key = tokens[i - 1] + "+" + tokens[i];
    if (pairs.count(key) && seen.insert(key).second)
      ++n;
  }
  return n;
}

inline int blockedLinkScore(const std::string &description,
                            const std::vector<std::string> &links) {
  std::unordered_set<std::string> have(links.begin(), links.end());
  std::unordered_set<std::string> seen;
  int n = 0;
  for (const auto &w : glossTokens(description)) {
    if (condFunctionWord(w))
      continue;
    if (have.count(w) && seen.insert(w).second)
      ++n;
  }
  return n;
}

inline std::string rankBlockedLinks(
    const std::string &description,
    const std::vector<std::pair<std::string, std::vector<std::string>>> &rows,
    const std::vector<int> &fecundity = {}) {
  int best = 0;
  for (const auto &row : rows) {
    const int score = blockedLinkScore(description, row.second);
    if (score > best)
      best = score;
  }
  if (best < 2)
    return std::string();
  std::vector<size_t> tied;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (blockedLinkScore(description, rows[i].second) == best)
      tied.push_back(i);
  }
  if (tied.size() == 1)
    return rows[tied[0]].first;
  if (fecundity.size() == rows.size()) {
    int top = 0;
    size_t pick = 0;
    int winners = 0;
    for (size_t index : tied) {
      const int f = fecundity[index];
      if (winners == 0 || f > top) {
        top = f;
        pick = index;
        winners = 1;
      } else if (f == top) {
        ++winners;
      }
    }
    if (winners == 1 && top > 0)
      return rows[pick].first;
  }
  /* Unigram scores tied and fecundity did not separate them.
     Neighbor pairs are the length-2 replicator. A unique pair score
     of at least 2 picks that row. A unique unigram score never
     reaches this branch. */
  int pairBest = 0;
  for (size_t index : tied) {
    const int score = blockedPairScore(description, rows[index].second);
    if (score > pairBest)
      pairBest = score;
  }
  if (pairBest < 2)
    return std::string();
  size_t pairPick = 0;
  int pairWinners = 0;
  for (size_t index : tied) {
    if (blockedPairScore(description, rows[index].second) != pairBest)
      continue;
    pairPick = index;
    ++pairWinners;
  }
  if (pairWinners == 1)
    return rows[pairPick].first;
  /* Pair scores are tied too. The conjunctive hidden layer adds the
     longer ordered spans (length 3 and 4). It can only add matches, so
     a row that has no span beyond its pairs cannot win by accident. */
  int hiddenBest = 0;
  for (size_t index : tied) {
    const int score = hiddenLinkScore(description, rows[index].second);
    if (score > hiddenBest)
      hiddenBest = score;
  }
  if (hiddenBest <= pairBest)
    return std::string();
  size_t hiddenPick = 0;
  int hiddenWinners = 0;
  for (size_t index : tied) {
    if (hiddenLinkScore(description, rows[index].second) != hiddenBest)
      continue;
    hiddenPick = index;
    ++hiddenWinners;
  }
  if (hiddenWinners == 1)
    return rows[hiddenPick].first;
  return std::string();
}

/* One-step bilinear memory. The first layer is a frozen one-hot
   embedding of words. The second layer is W. A witnessed pair (x, y)
   trains it by W += y x^T, which is one gradient step on -y^T W x
   with step size 1. There is no epoch loop and no new output column.
   With a single update and no repeated words,
   score(query, file) = |query ∩ x| * |file ∩ y|. */
inline double bilinearWitnessScore(
    const std::vector<std::string> &query, const std::vector<std::string> &file,
    const std::vector<std::string> &witnessQuery,
    const std::vector<std::string> &witnessFile) {
  std::unordered_set<std::string> qx(query.begin(), query.end());
  std::unordered_set<std::string> fy(file.begin(), file.end());
  int nq = 0;
  int nf = 0;
  std::unordered_set<std::string> seen;
  for (const auto &w : witnessQuery)
    if (qx.count(w) && seen.insert(w).second)
      ++nq;
  seen.clear();
  for (const auto &w : witnessFile)
    if (fy.count(w) && seen.insert(w).second)
      ++nf;
  return static_cast<double>(nq) * static_cast<double>(nf);
}

/* A gloss already written onto one file. tf is stored so the bilinear
   weight can be shown. For a word that hits only one file, the index
   mass does not depend on tf: the hit list has one alpha and the
   normalized share is 1. Repeating the word is not a stronger index. */
struct GlossTerm {
  std::string word;
  int tf{1};
};

struct GlossLineage {
  LineageState state{};
  std::vector<std::string> carriers;
};

inline int glossTokenOverlap(const std::string &text,
                            const std::vector<GlossTerm> &terms) {
  std::unordered_set<std::string> have;
  for (const auto &t : terms)
    if (!t.word.empty())
      have.insert(t.word);
  std::string cur;
  std::unordered_set<std::string> seen;
  int n = 0;
  auto flush = [&]() {
    if (cur.size() >= 3 && have.count(cur) && seen.insert(cur).second)
      ++n;
    cur.clear();
  };
  for (unsigned char c : text) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else
      flush();
  }
  flush();
  return n;
}

inline bool glossCarried(const std::string &description,
                         const std::string &rawBody,
                         const std::vector<GlossTerm> &terms) {
  return glossTokenOverlap(description, terms) >= 2 &&
         glossTokenOverlap(rawBody, terms) == 0;
}

inline std::string glossParent(const std::vector<GlossTerm> &terms) {
  std::string out;
  for (const auto &t : terms) {
    if (t.word.empty())
      continue;
    if (!out.empty())
      out.push_back(' ');
    out += t.word;
  }
  return out;
}

/* A later text that copies the gloss is a generation. One copy does
   not admit a meme. The index may already route on the inserted
   words; admission is a separate count. indexWithoutCopy ends the
   streak and does not increase fecundity. */
inline void reinforceGloss(std::vector<GlossTerm> &terms, GlossLineage &line,
                           const std::string &description) {
  if (terms.empty() || description.empty())
    return;
  const auto judged = judgeCopy(glossParent(terms), description);
  /* The stored list is the source identity. A string already stored is
     the same source. Once eight strings are stored, further unseen
     strings are not new sources: counting them without storing them
     would count the same echo twice on the next call. The cost is that
     a ninth distinct copy does not raise fecundity. */
  bool seen = false;
  for (const auto &stored : line.carriers) {
    if (stored == description) {
      seen = true;
      break;
    }
  }
  const bool distinct = !seen && line.carriers.size() < 8;
  lineageObserve(line.state, judged, distinct);
  if (!(judged.replicator && distinct))
    return;
  line.carriers.push_back(description);
  std::unordered_set<std::string> offspring;
  std::string cur;
  auto flush = [&]() {
    if (cur.size() >= 3)
      offspring.insert(cur);
    cur.clear();
  };
  for (unsigned char c : description) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else
      flush();
  }
  flush();
  for (auto &t : terms) {
    if (t.tf < 4 && offspring.count(t.word))
      t.tf += 1;
  }
}

/* Collection-frequent on this graph. memeCount/4 only ever caught "the"
   on the 497-node wikitext instrument; of/and/to still looked like content. */
inline int mappingHighDfCut(int memeCount) {
  if (memeCount <= 0)
    return 3;
  int cut = 3;
  int s = 1;
  while (s * s < memeCount)
    ++s;
  if (s > cut)
    cut = s;
  return cut;
}

inline bool mappingHighDf(int df, int memeCount) {
  return df > mappingHighDfCut(memeCount);
}

inline bool mappingMergeStopWord(const std::string &w) {
  return w == "a" || w == "an" || w == "the" || w == "of" || w == "to" ||
         w == "and" || w == "in" || w == "on" || w == "for" || w == "is" ||
         w == "was" || w == "with" || w == "as" || w == "by" || w == "at" ||
         w == "from" || w == "that" || w == "this" || w == "it" || w == "be" ||
         w == "or" || w == "are" || w == "were" || w == "been" || w == "have" ||
         w == "has" || w == "had" || w == "not" || w == "but" || w == "they" ||
         w == "their" || w == "which" || w == "its" || w == "he" || w == "she" ||
         w == "we" || w == "you" || w == "his" || w == "her" || w == "them" ||
         w == "than" || w == "then" || w == "also" || w == "into" || w == "over";
}

/* n-gram merge: require a majority of incoming words and of incoming
   content words. Two shared stopwords must not glue unrelated articles.
   Same-article sliding windows may still grow a bag; screen drops
   anything larger than one ngramMax window so those bags are not samples. */
inline bool mappingMergeAllowed(int overlap, int incomingUniq, int existingSize,
                                int minOverlap, int incomingContent,
                                int overlapContent) {
  if (overlap < minOverlap || incomingUniq <= 0 || incomingContent <= 0)
    return false;
  const int need = std::max(minOverlap, (incomingUniq + 1) / 2);
  if (overlap < need)
    return false;
  const int contentNeed = std::max(1, (incomingContent + 1) / 2);
  if (overlapContent < contentNeed)
    return false;
  const int newContent = std::max(0, incomingContent - overlapContent);
  if (newContent >= 1 && incomingContent > 0 &&
      overlapContent * 5 < incomingContent * 3)
    return false;
  if (existingSize >= 12 && overlapContent < 3 &&
      overlapContent * 4 < existingSize)
    return false;
  return true;
}

/* Query-time n-grams in mapWordsToMemes must not grow a prior meme.
   Only bind when the incoming window is already a content subset. */
inline bool mappingQueryMergeAllowed(int overlap, int incomingUniq,
                                     int existingSize, int minOverlap,
                                     int incomingContent, int overlapContent) {
  if (!mappingMergeAllowed(overlap, incomingUniq, existingSize, minOverlap,
                           incomingContent, overlapContent))
    return false;
  return incomingContent > 0 && overlapContent >= incomingContent;
}

/* Stop-only queries must not count as present, even if mid-df stops
   leak past the collection-frequency prior on a small graph. */
/* All existing memes a token window already sits inside. Query face is
   lookup-only (content subset). Ingest face uses the merge gate but the
   caller decides whether to bind. Not a present gate — a sentence may
   light many memes; do not keep only the single best overlap. */
template <class MemesOfWord, class WordsOfMeme>
inline std::vector<std::string> mappingCollectExistingMemes(
    const std::vector<std::string> &uniq, MemesOfWord memesOfWord,
    WordsOfMeme wordsOfMeme, int minOverlap, int maxWordSet, bool queryFace) {
  std::vector<std::string> hits;
  if (uniq.size() <= 1)
    return hits;
  std::unordered_map<std::string, int> counts;
  for (const auto &w : uniq) {
    for (const auto &mid : memesOfWord(w))
      counts[mid]++;
  }
  int incomingContent = 0;
  for (const auto &w : uniq) {
    if (!mappingMergeStopWord(w))
      ++incomingContent;
  }
  std::vector<std::pair<int, std::string>> ranked;
  ranked.reserve(counts.size());
  for (const auto &kv : counts)
    ranked.push_back({kv.second, kv.first});
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<int, std::string> &a,
               const std::pair<int, std::string> &b) {
              if (a.first != b.first)
                return a.first > b.first;
              return a.second < b.second;
            });
  for (const auto &kv : ranked) {
    if (kv.first < minOverlap)
      continue;
    const auto existing = wordsOfMeme(kv.second);
    std::unordered_set<std::string> have(existing.begin(), existing.end());
    int overlapContent = 0;
    for (const auto &w : uniq) {
      if (mappingMergeStopWord(w))
        continue;
      if (have.count(w))
        ++overlapContent;
    }
    int next = static_cast<int>(have.size());
    for (const auto &w : uniq)
      if (have.insert(w).second)
        ++next;
    if (next > maxWordSet)
      continue;
    if (queryFace) {
      if (!mappingQueryMergeAllowed(kv.first, static_cast<int>(uniq.size()),
                                    static_cast<int>(existing.size()),
                                    minOverlap, incomingContent, overlapContent))
        continue;
    } else if (!mappingMergeAllowed(
                   kv.first, static_cast<int>(uniq.size()),
                   static_cast<int>(existing.size()), minOverlap,
                   incomingContent, overlapContent)) {
      continue;
    }
    hits.push_back(kv.second);
  }
  return hits;
}

inline bool mappingQueryHasContentForMeme(
    const std::vector<std::string> &tokens,
    const std::vector<std::string> &memeWords) {
  if (tokens.empty() || memeWords.empty())
    return false;
  std::unordered_set<std::string> have(memeWords.begin(), memeWords.end());
  for (const auto &t : tokens) {
    if (t.empty() || mappingMergeStopWord(t))
      continue;
    if (have.count(t))
      return true;
  }
  return false;
}

/* α(w|m)=tf/df in bind order. Missing tfs load as 1 (old WORD lines). */
inline std::vector<std::pair<std::string, double>>
mappingMemeWeights(const std::vector<std::string> &words,
                   const std::vector<int> &tfs,
                   const std::unordered_map<std::string, int> &df) {
  std::vector<std::pair<std::string, double>> out;
  out.reserve(words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    if (words[i].empty())
      continue;
    int d = 1;
    auto it = df.find(words[i]);
    if (it != df.end() && it->second > 0)
      d = it->second;
    const int tf = (i < tfs.size() && tfs[i] > 0) ? tfs[i] : 1;
    out.push_back({words[i], mappingWordAlpha(tf, d)});
  }
  return out;
}

/* Share of a meme bag that sits on content words, scaled by collection
   IDF. Stop-heavy phrase nodes score near 0; rare names score near 1.
   Used to pick 隔代 samples, not as a present gate. */
inline double mappingContentSpec(
    const std::vector<std::pair<std::string, double>> &weights,
    const std::unordered_map<std::string, int> &df, int memeCount) {
  double num = 0.0;
  double den = 0.0;
  for (const auto &wp : weights) {
    if (wp.first.empty() || wp.second <= 0.0)
      continue;
    den += wp.second;
    if (mappingMergeStopWord(wp.first) || wp.first.size() < 3)
      continue;
    int d = 1;
    auto it = df.find(wp.first);
    if (it != df.end() && it->second > 0)
      d = it->second;
    num += wp.second * mappingIdfScale(d, memeCount);
  }
  if (den <= 1e-12)
    return 0.0;
  return num / den;
}

/* Typical content (word, α) of a meme: drop stops / short tokens, then
   IPR-concentrate. Identity for wipe-reingest: a new graph recovered
   the same meme if it grew a node that covers the typical set, even
   when clustering minted a new meme_p_* hash. Not a present gate. */
struct AlphaIdentity {
  bool same{false};
  double cosine{0.0};
  double massCover{0.0};
  int typicalNeed{0};
  int typicalHit{0};
};

inline std::unordered_map<std::string, double> mappingTypicalAlpha(
    const std::vector<std::pair<std::string, double>> &weights) {
  std::unordered_map<std::string, double> seeds;
  for (const auto &wp : weights) {
    if (wp.first.empty() || wp.second <= 0.0)
      continue;
    if (mappingMergeStopWord(wp.first) || wp.first.size() < 4)
      continue;
    seeds[wp.first] = wp.second;
  }
  mappingConcentrateSeeds(&seeds);
  return seeds;
}

/* Identity words of a meme: typical set minus short/high-df hubs.
   Compose must hit these so "green" cannot retrieve an unrelated
   Green. Not a present gate. */
inline std::vector<std::string> mappingRareTypical(
    const std::vector<std::pair<std::string, double>> &weights,
    const std::unordered_map<std::string, int> &df, int memeCount) {
  const auto typ = mappingTypicalAlpha(weights);
  std::vector<std::pair<double, std::string>> scored;
  for (const auto &kv : typ) {
    if (kv.first.size() < 5)
      continue;
    int d = 1;
    auto it = df.find(kv.first);
    if (it != df.end() && it->second > 0)
      d = it->second;
    const double idf = mappingIdfScale(d, memeCount);
    if (kv.first.size() < 6 && d > 2)
      continue;
    if (kv.first.size() < 7 && idf < 0.55)
      continue;
    const double bonus = kv.first.size() >= 7 ? 1.25 : 1.0;
    scored.push_back({idf * bonus * kv.second, kv.first});
  }
  std::sort(scored.begin(), scored.end(),
            [](const std::pair<double, std::string> &a,
               const std::pair<double, std::string> &b) {
              if (a.first != b.first)
                return a.first > b.first;
              return a.second < b.second;
            });
  std::vector<std::string> out;
  const int take = std::min(6, static_cast<int>(scored.size()));
  out.reserve(static_cast<size_t>(take));
  for (int i = 0; i < take; ++i)
    out.push_back(scored[static_cast<size_t>(i)].second);
  return out;
}

inline AlphaIdentity mappingAlphaIdentity(
    const std::vector<std::pair<std::string, double>> &frozen,
    const std::vector<std::pair<std::string, double>> &live) {
  AlphaIdentity out;
  const auto typ = mappingTypicalAlpha(frozen);
  out.typicalNeed = static_cast<int>(typ.size());
  if (typ.empty())
    return out;
  std::unordered_map<std::string, double> liveA;
  for (const auto &wp : live) {
    if (!wp.first.empty() && wp.second > 0.0)
      liveA[wp.first] = wp.second;
  }
  double frozenMass = 0.0;
  double hitMass = 0.0;
  double dot = 0.0;
  double nf = 0.0;
  double nl = 0.0;
  for (const auto &kv : typ) {
    frozenMass += kv.second;
    nf += kv.second * kv.second;
    auto it = liveA.find(kv.first);
    if (it == liveA.end())
      continue;
    ++out.typicalHit;
    hitMass += kv.second;
    dot += kv.second * it->second;
    nl += it->second * it->second;
  }
  if (frozenMass > 1e-12)
    out.massCover = hitMass / frozenMass;
  if (nf > 1e-18 && nl > 1e-18)
    out.cosine = dot / (std::sqrt(nf) * std::sqrt(nl));
  const int needHit = std::max(1, (out.typicalNeed + 1) / 2);
  out.same = out.typicalHit >= needHit && out.massCover >= 0.5;
  return out;
}

namespace {

inline std::vector<std::string> orderedMemeTokens(const std::string &s) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    if (cur.size() >= 3)
      out.push_back(cur);
    cur.clear();
  };
  for (unsigned char c : s) {
    if (std::isalnum(c))
      cur.push_back(static_cast<char>(std::tolower(c)));
    else
      flush();
  }
  flush();
  return out;
}

inline bool orderedMemeHighDf(int df, int memeCount) {
  if (memeCount <= 0)
    return false;
  int side = 1;
  while (side * side < memeCount && side < 100000)
    ++side;
  return df > std::max(3, side);
}

/* Kendall tau on the query tokens that sit in the meme.
   Positions are successive slots of that word in the meme.
   tau = (n_concordant - n_discordant) / (n_concordant + n_discordant).
   Ties do not enter the denominator. 0 when there is no ordered pair.
   Kendall, M. G. (1938). Biometrika 30:81-93. */
inline double orderedMemeOrderLift(const std::vector<std::string> &query,
                                   const std::vector<std::string> &meme) {
  if (query.size() < 2 || meme.size() < 2)
    return 0.0;
  std::unordered_map<std::string, std::vector<int>> at;
  for (int i = 0; i < static_cast<int>(meme.size()); ++i)
    at[meme[static_cast<size_t>(i)]].push_back(i);
  std::unordered_map<std::string, int> used;
  std::vector<int> pos;
  pos.reserve(query.size());
  for (const auto &w : query) {
    auto it = at.find(w);
    if (it == at.end() || it->second.empty())
      continue;
    int u = used[w]++;
    if (u >= static_cast<int>(it->second.size()))
      u = static_cast<int>(it->second.size()) - 1;
    pos.push_back(it->second[static_cast<size_t>(u)]);
  }
  int concordant = 0;
  int discordant = 0;
  for (size_t i = 0; i < pos.size(); ++i) {
    for (size_t j = i + 1; j < pos.size(); ++j) {
      if (pos[i] < pos[j])
        ++concordant;
      else if (pos[i] > pos[j])
        ++discordant;
    }
  }
  if (concordant + discordant == 0)
    return 0.0;
  return static_cast<double>(concordant - discordant) /
         static_cast<double>(concordant + discordant);
}

} // namespace

inline std::vector<OrderedMemeScore>
scoreOrderedMemes(const std::string &query,
                  const std::vector<OrderedMeme> &memes) {
  std::vector<OrderedMemeScore> out;
  if (query.empty() || memes.empty())
    return out;
  const auto qTokens = orderedMemeTokens(query);
  if (qTokens.size() < 2)
    return out;
  std::unordered_map<std::string, double> queryMass;
  for (const auto &w : qTokens)
    queryMass[w] += 1.0;
  std::vector<std::vector<std::string>> memeToks;
  memeToks.reserve(memes.size());
  std::unordered_map<std::string, int> df;
  for (const auto &m : memes) {
    auto tok = orderedMemeTokens(m.text);
    std::unordered_set<std::string> seen;
    for (const auto &w : tok) {
      if (seen.insert(w).second)
        df[w] += 1;
    }
    memeToks.push_back(std::move(tok));
  }
  const int n = static_cast<int>(memes.size());
  std::unordered_map<std::string, std::vector<MappingWordHit>> hits;
  for (size_t i = 0; i < memes.size(); ++i) {
    std::unordered_map<std::string, int> tf;
    for (const auto &w : memeToks[i])
      tf[w] += 1;
    for (const auto &kv : tf) {
      const int dfi = df[kv.first];
      if (dfi <= 0)
        continue;
      hits[kv.first].push_back(MappingWordHit{
          memes[i].id, static_cast<double>(kv.second) / static_cast<double>(dfi)});
    }
  }
  std::unordered_map<std::string, double> seeds;
  mappingAddConditionedSeeds(queryMass, hits, &seeds, nullptr, n);
  std::unordered_map<std::string, double> nullMass;
  for (const auto &kv : queryMass) {
    if (orderedMemeHighDf(df[kv.first], n))
      nullMass[kv.first] = kv.second;
  }
  std::unordered_map<std::string, double> nullSeeds;
  if (!nullMass.empty())
    mappingAddConditionedSeeds(nullMass, hits, &nullSeeds, nullptr, n);
  out.reserve(memes.size());
  for (size_t i = 0; i < memes.size(); ++i) {
    OrderedMemeScore s;
    s.id = memes[i].id;
    auto it = seeds.find(s.id);
    s.mass = it == seeds.end() ? 0.0 : it->second;
    const double nullM = nullSeeds.count(s.id) ? nullSeeds[s.id] : 0.0;
    s.lift = s.mass / std::max(nullM, 1e-6);
    s.orderLift = orderedMemeOrderLift(qTokens, memeToks[i]);
    std::unordered_set<std::string> inMeme(memeToks[i].begin(), memeToks[i].end());
    std::unordered_set<std::string> seenQ;
    for (const auto &w : qTokens) {
      if (inMeme.count(w) && seenQ.insert(w).second)
        ++s.overlap;
    }
    s.present = s.mass > 0.0 && s.lift >= kGnnPresentLift &&
                s.orderLift > 0.0 && s.overlap >= 2;
    out.push_back(std::move(s));
  }
  return out;
}

/* Word↔meme bindings used by official n-gram clustering. */
class MemeClusterStore {
public:
  virtual ~MemeClusterStore() = default;
  virtual std::vector<std::string> memesOfWord(const std::string &word) const = 0;
  virtual std::vector<std::string> wordsOfMeme(const std::string &memeId) const = 0;
  virtual void bind(const std::string &word, const std::string &memeId) = 0;
  virtual void ensureNode(const std::string &memeId) = 0;
  virtual int wordTf(const std::string &memeId,
                     const std::string &word) const {
    if (word.empty() || memeId.empty())
      return 0;
    const auto words = wordsOfMeme(memeId);
    return std::find(words.begin(), words.end(), word) != words.end() ? 1 : 0;
  }
};

/* In-memory KVM + undirected consecutive edges for the existence lab. */
class InMemoryStore : public MemeClusterStore {
public:
  std::vector<std::string> memesOfWord(const std::string &word) const override;
  std::vector<std::string> wordsOfMeme(const std::string &memeId) const override;
  void bind(const std::string &word, const std::string &memeId) override;
  int wordTf(const std::string &memeId, const std::string &word) const override;
  void bindTf(const std::string &word, const std::string &memeId, int tf);
  /* Ordered (word, α=tf/df) pairs. */
  std::vector<std::pair<std::string, double>>
  weightsOfMeme(const std::string &memeId) const;
  void ensureNode(const std::string &memeId) override;
  void linkConsecutive(const std::string &from, const std::string &to,
                       double weight = 1.0);
  std::vector<std::pair<std::string, double>>
  neighborsOf(const std::string &memeId) const;
  std::vector<std::string> nodeIds() const { return nodes_; }
  void clear();
  bool empty() const;
  size_t nodeCount() const { return nodes_.size(); }
  void saveText(const std::string &path) const;
  bool loadText(const std::string &path);

private:
  std::vector<std::string> nodes_;
  std::unordered_map<std::string, std::vector<std::string>> wordToMemes_;
  std::unordered_map<std::string, std::vector<std::string>> memeToWords_;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> wordTf_;
  std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>
      adj_;
};

std::string sha1Hex(const std::string &s);

/* Same clustering rules as RuntimeState::buildMemeSequenceFromTokens. */
std::vector<std::string> buildMemeSequence(MemeClusterStore &store,
                                           const std::vector<std::string> &tokens,
                                           const Params &p);

/* ingestDocument: sequence + consecutive undirected links. */
std::vector<std::string> ingestDocumentLike(InMemoryStore &store,
                                            const std::vector<std::string> &tokens,
                                            const Params &p);

struct ExistenceResult {
  bool queryMentionsNonce{false};
  bool kvmReconstructedNonce{false};
  bool diffusionRankedNonceMeme{false};
  bool distractorReconstructedNonce{false};
  bool distractorDiffusionRankedNonce{false};
  double cueNonceScore{0.0};
  double cueMaxScore{0.0};
  double distractorNonceScore{0.0};
  int nonceMemeCount{0};
  int cueSeedCount{0};
  int hopsCueToNonce{-1};
  std::vector<std::string> cueReconstructedWords;
  std::vector<std::pair<std::string, double>> cueRanked;
};

/* Construct one harmless unit on the official path and try to recall it. */
ExistenceResult runHarmlessExistence(const Params &p = {});

std::vector<std::string> tokenizeAscii(const std::string &text);
bool textExpressesPayload(const std::string &text, const std::string &payload);
bool textExpressesHarmlessMeme(const std::string &text);

/* Propagation judge: exact token is not enough. A unit may be split,
   carried only by the host sentence, drift, then return. */
struct ExpressionJudgement {
  bool exactToken{false};
  bool reconstructed{false};
  bool carrierPresent{false};
  bool unitExpressed{false};
  bool latent{false};
  bool drifted{false};
};

std::vector<std::string> payloadFragments(const std::string &payload);
ExpressionJudgement judgeExpression(const std::string &text,
                                    const std::string &payload,
                                    const std::vector<std::string> &carrierTokens);
bool unitPropagates(const std::vector<ExpressionJudgement> &rounds);

/* Lab decoder only. If the payload word is in the active cluster it is
   pushed first. Do not use this as the generation surface for existence. */
std::string expressActivated(const InMemoryStore &store,
                             const std::vector<std::string> &stimulus,
                             const Params &p, int surfaceId,
                             const std::string &payload);

/* Honest GNN note for organ-on llama prompts: ranked words, no payload force. */
std::string graphNoteActivated(const InMemoryStore &store,
                               const std::vector<std::string> &stimulus,
                               const Params &p);

inline constexpr int kMinStabilityRounds = 6;

struct SerialCase {
  std::string id;
  std::string payload;
  std::vector<std::string> inputTokens;
  std::vector<std::string> distractorTokens;
  Params params{};
  int generations{kMinStabilityRounds};
  bool expectExpress{true};
  std::string forbiddenPayload;
  bool interfereFirst{false};
};

struct SerialRound {
  std::string text;
  bool expresses{false};
  ExpressionJudgement judge{};
  bool emptyStoreBeforeIngest{false};
  bool decoderEchoOnEmpty{false};
  bool heldOutResidue{false};
};

struct SerialReplicationResult {
  std::string id;
  std::string payload;
  std::string input;
  std::vector<SerialRound> rounds;
  std::string distractorOutput;
  bool distractorExpressesMeme{false};
  bool leakedForbidden{false};
  bool residueDetected{false};
  bool expectExpress{true};
  bool passed{false};
  bool propagationPassed{false};
  bool returnedAfterGap{false};
  int expressedRounds{0};
  int unitRounds{0};
  int latentRounds{0};
  int driftRounds{0};
};

struct SerialBatteryReport {
  std::vector<SerialReplicationResult> groups;
  int positivePassed{0};
  int positiveTotal{0};
  int negativePassed{0};
  int negativeTotal{0};
};

/*
 * Decoder-echo battery (expressActivated). Not the existence proof.
 * The real serial is: carrierOfMeme -> empty-session llama ->
 * detectMemeInText on the frozen instrument.
 */
SerialReplicationResult runSerialReplication(const Params &p = {});
SerialReplicationResult runSerialReplicationCase(const SerialCase &c);
SerialBatteryReport runSerialReplicationBattery();

struct ScreenedMeme {
  std::string id;
  std::vector<std::string> words;
  std::string carrier;
  int degree{0};
  double barrierScore{0.0};
};

struct MemeDetection {
  bool present{false};
  double memeScore{0.0};
  double peakScore{0.0};
  int rank{-1};
  double nullScore{0.0};
  double lift{0.0};
  /* Extra geometry. Official present is still GNN+lift only. */
  double semanticCos{0.0};
  double bagCos{0.0};
  double reversedCos{0.0};
  double orderLift{0.0};
  int tensorRank{-1};
  std::string nearestWord;
  std::string nearestSentence;
  /* Isotope label for this query → this meme. Not used as present. */
  std::string queryId;
  std::string trace;
};

/* KVM words of one meme id. Bag join; diagnostic / fallback, not the RAG carrier. */
std::string carrierOfMeme(const InMemoryStore &store, const std::string &memeId);

/* Ordered carrier from the meme tensor. Text is a sentence (corpus path
   or RoPE/bigram decode), never a KVM bag join. */
struct RagCarrier {
  std::string text;
  std::vector<int> unitIndex;
  std::vector<std::string> units;
  double score{0.0};
  int wordHit{0};
  double bagCos{0.0};
  double semanticCos{0.0};
  double orderLift{0.0};
  std::string method;
  /* Target share of present-meme mass on the chosen sentence. */
  double exclusivity{0.0};
  int targetRank{-1};
  int otherPresent{0};
  double typicalCover{0.0};
  /* Isotope meme→sentences: corpus units that officially present this id. */
  std::vector<std::string> sources;
};

struct MemeActivation {
  std::string id;
  double score{0.0};
  double lift{0.0};
  int rank{-1};
  bool present{false};
  std::string trace;
};

/* requireExclusive: only a sentence where the target is rank-0 or holds
   >= half of present-meme mass may be the carrier. No exclusive source
   => empty carrier (do not emit a mixed present sentence). Serial
   generation uses this; the host soak may pass false to keep a
   diagnostic fallback. */
RagCarrier composeCarrierRag(const InMemoryStore &store,
                             const std::string &memeId,
                             const std::string &corpusText, int maxUnits,
                             int maxSents = 2, bool requireExclusive = false);

/* Words with df > sqrt(n). Language prior for present-lift; not deleted from queries. */
std::vector<std::string> highDfPriorTokens(const InMemoryStore &store);

/* Query the frozen instrument. Does not ingest `text` and does not bind new words. */
MemeDetection detectMemeInText(const InMemoryStore &instrument,
                               const std::string &text,
                               const std::string &memeId, const Params &p = {},
                               bool withTensor = true);

/* Isotope id-track: every meme this sentence lights, not one chunk→one
   meme. Official present of a named id is still GNN+lift+content. */
std::vector<MemeActivation> detectActivatedMemes(
    const InMemoryStore &instrument, const std::string &text,
    const Params &p = {}, int maxKeep = 16);

/* Degree heuristic only. Not readme 统计. Official screen is analyzeGraph
   most/least in tools/meme_barrier_instrument.cpp. */
std::vector<ScreenedMeme> screenMemesFromBarrier(const InMemoryStore &store,
                                                 int maxKeep = 12);

int ingestCorpusText(InMemoryStore &store, const std::string &text, int maxUnits,
                     const Params &p = {});
int ingestCorpusFile(InMemoryStore &store, const std::string &path, int maxUnits,
                     const Params &p = {});

/* Paragraph/sentence unit split shared by ingestion and carrier compose;
   exported so lab instruments can slice a corpus by unit offset. */
std::vector<std::string> splitCorpusUnits(const std::string &text);

} // namespace memetic
} // namespace phoenix
