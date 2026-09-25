/* memetic_existence.hpp - Official-path harmless meme existence lab

   Memes are graph units on the official MemeBarrier / KVM / GNN path
   (wikitext ingest via the same clustering as
   RuntimeState::buildMemeSequenceFromTokens). A sample is a meme id,
   not a carrier sentence and not an invented nonce.

   A meme is an ordered (word, α) set on the official graph, not a
   bare lexicon. α(w|m) = tf(w,m)/df(w) from ingest counts (old
   snapshots with no TF load as tf=1, so α is collection 1/df).
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
     Wipe-reingest asks whether a fresh graph grew the same ordered
     (word, α) typical set, not whether the bag-hash id is identical.
     carrierOfMeme remains the KVM word bag (diagnostic / fallback).

   expressActivated can name a payload when the cluster is active. That
   is a lab decoder, not llama-server, and is not an existence proof.
   Two-AI contagion is not part of this lab. */
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

inline constexpr double kGnnPresentPeakFrac = 0.05;
inline constexpr double kGnnPresentLift = 2.0;

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

} // namespace memetic
} // namespace phoenix
