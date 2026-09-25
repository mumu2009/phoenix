/* memetic_existence.cpp - Official-path harmless meme existence lab */

#include "memetic_existence.hpp"

#include "addons/ThePlugInForSecurity/meme_tensor.hpp"
#include "graph_diffusion_summarizer.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <queue>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace phoenix {
namespace memetic {
namespace {

void pushUnique(std::vector<std::string> &xs, const std::string &x) {
  if (std::find(xs.begin(), xs.end(), x) == xs.end())
    xs.push_back(x);
}

std::string trimCopy(const std::string &s) {
  auto start = s.find_first_not_of(" \n\r\t");
  if (start == std::string::npos)
    return std::string();
  auto end = s.find_last_not_of(" \n\r\t");
  return s.substr(start, end - start + 1);
}

bool wordsContain(const std::vector<std::string> &words, const std::string &needle) {
  return std::find(words.begin(), words.end(), needle) != words.end();
}

std::vector<std::string> reconstructWords(const InMemoryStore &store,
                                          const std::vector<std::string> &queryTokens) {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (const auto &w : queryTokens) {
    for (const auto &mid : store.memesOfWord(w)) {
      for (const auto &mw : store.wordsOfMeme(mid)) {
        if (seen.insert(mw).second)
          out.push_back(mw);
      }
    }
  }
  return out;
}

std::vector<std::pair<std::string, double>>
diffuseFromQuery(const InMemoryStore &store,
                 const std::vector<std::string> &queryTokens, const Params &p,
                 bool highDfOnly = false, MappingMassFrom *from = nullptr) {
  const auto ids = store.nodeIds();
  if (ids.empty())
    return {};
  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < ids.size(); ++i)
    index[ids[i]] = i;

  /* Keep every token, including stopwords. Allocate each word over its
     memes using query-conditioned α, then diffuse. */
  std::vector<double> seeds(ids.size(), 0.0);
  std::unordered_map<std::string, int> tf;
  for (const auto &w : queryTokens)
    tf[w]++;
  const int memeCount = static_cast<int>(ids.size());
  std::unordered_map<std::string, double> queryMass;
  std::unordered_map<std::string, std::vector<MappingWordHit>> hits;
  for (const auto &kv : tf) {
    const auto mids = store.memesOfWord(kv.first);
    if (mids.empty())
      continue;
    const int df = static_cast<int>(mids.size());
    if (highDfOnly && !mappingHighDf(df, memeCount))
      continue;
    queryMass[kv.first] = static_cast<double>(kv.second);
    auto &row = hits[kv.first];
    row.reserve(mids.size());
    for (const auto &mid : mids) {
      const int tfm = store.wordTf(mid, kv.first);
      if (tfm <= 0 || !index.count(mid))
        continue;
      row.push_back({mid, mappingWordAlpha(tfm, df)});
    }
  }
  std::unordered_map<std::string, double> allocated;
  mappingAddConditionedSeeds(queryMass, hits, &allocated, from, memeCount);
  for (const auto &kv : allocated) {
    auto it = index.find(kv.first);
    if (it != index.end())
      seeds[it->second] += kv.second;
  }

  std::vector<std::vector<std::tuple<size_t, double, int>>> adjacency(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    for (const auto &nb : store.neighborsOf(ids[i])) {
      auto jt = index.find(nb.first);
      if (jt == index.end())
        continue;
      adjacency[i].push_back({jt->second, nb.second, 0});
    }
  }

  phoenix::graph::GraphDiffusionSummarizer summarizer;
  auto summary = summarizer.summarize(ids, adjacency, seeds, p.diffusionRounds,
                                      0.85, ids.size());
  return summary.rankedNodes;
}

double maxNonceScore(const InMemoryStore &store,
                     const std::vector<std::pair<std::string, double>> &ranked,
                     const std::string &nonce) {
  double best = 0.0;
  for (const auto &row : ranked) {
    if (row.second > best && wordsContain(store.wordsOfMeme(row.first), nonce))
      best = row.second;
  }
  return best;
}

double maxScore(const std::vector<std::pair<std::string, double>> &ranked) {
  double best = 0.0;
  for (const auto &row : ranked) {
    if (row.second > best)
      best = row.second;
  }
  return best;
}

bool diffusionActivatedNonce(double nonceScore, double peakScore) {
  if (nonceScore <= 1e-6 || peakScore <= 1e-6)
    return false;
  return nonceScore >= 0.05 * peakScore;
}

int bfsHops(const InMemoryStore &store, const std::vector<std::string> &fromIds,
            const std::unordered_set<std::string> &targets) {
  if (targets.empty() || fromIds.empty())
    return -1;
  std::queue<std::pair<std::string, int>> q;
  std::unordered_set<std::string> seen;
  for (const auto &id : fromIds) {
    if (id.empty() || !seen.insert(id).second)
      continue;
    if (targets.count(id))
      return 0;
    q.push({id, 0});
  }
  while (!q.empty()) {
    auto cur = q.front();
    q.pop();
    for (const auto &nb : store.neighborsOf(cur.first)) {
      if (!seen.insert(nb.first).second)
        continue;
      if (targets.count(nb.first))
        return cur.second + 1;
      q.push({nb.first, cur.second + 1});
    }
  }
  return -1;
}

} // namespace

std::vector<std::string>
InMemoryStore::memesOfWord(const std::string &word) const {
  auto it = wordToMemes_.find(word);
  if (it == wordToMemes_.end())
    return {};
  return it->second;
}

void InMemoryStore::bind(const std::string &word, const std::string &memeId) {
  if (word.empty() || memeId.empty())
    return;
  pushUnique(wordToMemes_[word], memeId);
  pushUnique(memeToWords_[memeId], word);
  wordTf_[memeId][word] += 1;
  ensureNode(memeId);
}

void InMemoryStore::bindTf(const std::string &word, const std::string &memeId,
                           int tf) {
  if (word.empty() || memeId.empty())
    return;
  pushUnique(wordToMemes_[word], memeId);
  pushUnique(memeToWords_[memeId], word);
  wordTf_[memeId][word] = std::max(1, tf);
  ensureNode(memeId);
}

int InMemoryStore::wordTf(const std::string &memeId,
                          const std::string &word) const {
  auto it = wordTf_.find(memeId);
  if (it == wordTf_.end())
    return 0;
  auto jt = it->second.find(word);
  if (jt == it->second.end() || jt->second <= 0)
    return 0;
  return jt->second;
}

std::vector<std::pair<std::string, double>>
InMemoryStore::weightsOfMeme(const std::string &memeId) const {
  const auto words = wordsOfMeme(memeId);
  std::vector<int> tfs;
  std::unordered_map<std::string, int> df;
  tfs.reserve(words.size());
  for (const auto &w : words) {
    tfs.push_back(std::max(1, wordTf(memeId, w)));
    df[w] = static_cast<int>(memesOfWord(w).size());
  }
  return mappingMemeWeights(words, tfs, df);
}

void InMemoryStore::ensureNode(const std::string &memeId) {
  if (memeId.empty())
    return;
  if (std::find(nodes_.begin(), nodes_.end(), memeId) == nodes_.end())
    nodes_.push_back(memeId);
}

std::vector<std::string>
InMemoryStore::wordsOfMeme(const std::string &memeId) const {
  auto it = memeToWords_.find(memeId);
  if (it == memeToWords_.end())
    return {};
  return it->second;
}

void InMemoryStore::linkConsecutive(const std::string &from, const std::string &to,
                                    double weight) {
  if (from.empty() || to.empty() || from == to)
    return;
  ensureNode(from);
  ensureNode(to);
  auto add = [&](const std::string &a, const std::string &b) {
    auto &row = adj_[a];
    for (auto &e : row) {
      if (e.first == b) {
        e.second += weight;
        return;
      }
    }
    row.push_back({b, weight});
  };
  add(from, to);
  add(to, from);
}

std::vector<std::pair<std::string, double>>
InMemoryStore::neighborsOf(const std::string &memeId) const {
  auto it = adj_.find(memeId);
  if (it == adj_.end())
    return {};
  return it->second;
}

void InMemoryStore::clear() {
  nodes_.clear();
  wordToMemes_.clear();
  memeToWords_.clear();
  wordTf_.clear();
  adj_.clear();
}

bool InMemoryStore::empty() const {
  return nodes_.empty() && wordToMemes_.empty() && memeToWords_.empty() &&
         adj_.empty();
}

void InMemoryStore::saveText(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  for (const auto &id : nodes_)
    out << "NODE " << id << "\n";
  for (const auto &kv : memeToWords_) {
    for (const auto &w : kv.second)
      out << "WORD " << kv.first << " " << w << " "
          << std::max(1, wordTf(kv.first, w)) << "\n";
  }
  for (const auto &kv : adj_) {
    for (const auto &e : kv.second) {
      if (kv.first < e.first)
        out << "EDGE " << kv.first << " " << e.first << " " << e.second << "\n";
    }
  }
}

bool InMemoryStore::loadText(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  clear();
  std::string kind;
  while (in >> kind) {
    if (kind == "NODE") {
      std::string id;
      in >> id;
      ensureNode(id);
    } else if (kind == "WORD") {
      std::string id, w;
      in >> id >> w;
      int tf = 1;
      const int peek = in.peek();
      if (peek == ' ' || peek == '\t') {
        in >> std::ws;
        const int digit = in.peek();
        if (digit >= '0' && digit <= '9')
          in >> tf;
      }
      bindTf(w, id, tf);
    } else if (kind == "EDGE") {
      std::string a, b;
      double wt = 1.0;
      in >> a >> b >> wt;
      linkConsecutive(a, b, wt);
    } else {
      std::string skip;
      std::getline(in, skip);
    }
  }
  return !empty();
}

std::string sha1Hex(const std::string &s) {
  auto rol = [](uint32_t v, int n) -> uint32_t {
    return (v << n) | (v >> (32 - n));
  };
  uint32_t h0 = 0x67452301u;
  uint32_t h1 = 0xEFCDAB89u;
  uint32_t h2 = 0x98BADCFEu;
  uint32_t h3 = 0x10325476u;
  uint32_t h4 = 0xC3D2E1F0u;
  std::vector<uint8_t> msg(s.begin(), s.end());
  const uint64_t bitLen = static_cast<uint64_t>(msg.size()) * 8ull;
  msg.push_back(0x80);
  while ((msg.size() % 64) != 56)
    msg.push_back(0);
  for (int i = 7; i >= 0; --i)
    msg.push_back(static_cast<uint8_t>((bitLen >> (i * 8)) & 0xffu));
  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(msg[off + 4 * i]) << 24) |
             (static_cast<uint32_t>(msg[off + 4 * i + 1]) << 16) |
             (static_cast<uint32_t>(msg[off + 4 * i + 2]) << 8) |
             static_cast<uint32_t>(msg[off + 4 * i + 3]);
    }
    for (int i = 16; i < 80; ++i)
      w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int i = 0; i < 80; ++i) {
      uint32_t f = 0, k = 0;
      if (i < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5A827999u;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1u;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDCu;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6u;
      }
      const uint32_t temp = rol(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rol(b, 30);
      b = a;
      a = temp;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }
  char buf[41];
  std::snprintf(buf, sizeof(buf), "%08x%08x%08x%08x%08x", h0, h1, h2, h3, h4);
  return std::string(buf);
}

std::vector<std::string> buildMemeSequence(MemeClusterStore &store,
                                           const std::vector<std::string> &tokens,
                                           const Params &p) {
  std::vector<std::string> list;
  for (const auto &t : tokens) {
    std::string s = trimCopy(t);
    if (!s.empty())
      list.push_back(s);
  }
  const int nMin = std::max(2, p.ngramMin);
  const int nMax = std::max(nMin, p.ngramMax);
  const int minOverlap = std::max(1, p.minOverlap);
  const int maxWordSet = std::max(4, p.maxMemeWords);

  std::vector<std::string> extras;
  auto resolveOrCreate =
      [&](const std::vector<std::string> &tokenSet) -> std::string {
    extras.clear();
    std::vector<std::string> uniq;
    std::unordered_set<std::string> seen;
    for (const auto &t : tokenSet) {
      if (t.empty())
        continue;
      if (seen.insert(t).second)
        uniq.push_back(t);
    }
    if (uniq.size() <= 1)
      return uniq.empty() ? "" : ("meme_" + sha1Hex(uniq[0]));

    std::unordered_map<std::string, int> counts;
    for (const auto &w : uniq) {
      for (const auto &mid : store.memesOfWord(w))
        counts[mid]++;
    }
    const auto hits = mappingCollectExistingMemes(
        uniq,
        [&](const std::string &w) { return store.memesOfWord(w); },
        [&](const std::string &id) { return store.wordsOfMeme(id); },
        minOverlap, maxWordSet, false);
    extras = hits;
    std::string best = hits.empty() ? std::string() : hits.front();
    int bestOverlap = 0;
    if (!best.empty()) {
      auto it = counts.find(best);
      if (it != counts.end())
        bestOverlap = it->second;
    }
    if (!best.empty() && bestOverlap >= minOverlap) {
      const auto existing = store.wordsOfMeme(best);
      std::unordered_set<std::string> have(existing.begin(), existing.end());
      int incomingContent = 0;
      int overlapContent = 0;
      for (const auto &w : uniq) {
        if (mappingMergeStopWord(w))
          continue;
        ++incomingContent;
        if (have.count(w))
          ++overlapContent;
      }
      int next = static_cast<int>(have.size());
      for (const auto &w : uniq)
        if (have.insert(w).second)
          ++next;
      if (next <= maxWordSet &&
          mappingMergeAllowed(bestOverlap, static_cast<int>(uniq.size()),
                              static_cast<int>(existing.size()), minOverlap,
                              incomingContent, overlapContent)) {
        for (const auto &w : uniq)
          store.bind(w, best);
        store.ensureNode(best);
        return best;
      }
    }
    std::vector<std::string> sorted = uniq;
    std::sort(sorted.begin(), sorted.end());
    std::string join;
    for (size_t i = 0; i < sorted.size(); ++i) {
      if (i)
        join += "|";
      join += sorted[i];
    }
    const std::string memeId = "meme_p_" + sha1Hex(join);
    store.ensureNode(memeId);
    for (const auto &w : sorted)
      store.bind(w, memeId);
    return memeId;
  };

  std::vector<std::string> seq;
  for (size_t i = 0; i < list.size(); ++i) {
    std::string picked;
    for (int n = nMax; n >= nMin; --n) {
      if (i + static_cast<size_t>(n) > list.size())
        continue;
      picked = resolveOrCreate(std::vector<std::string>(
          list.begin() + static_cast<std::ptrdiff_t>(i),
          list.begin() + static_cast<std::ptrdiff_t>(i + static_cast<size_t>(n))));
      if (!picked.empty())
        break;
    }
    if (picked.empty()) {
      const auto &w = list[i];
      if (!w.empty()) {
        picked = "meme_" + sha1Hex(w);
        store.ensureNode(picked);
        store.bind(w, picked);
      }
    }
    auto push = [&](const std::string &id) {
      if (!id.empty() && (seq.empty() || seq.back() != id))
        seq.push_back(id);
    };
    push(picked);
    for (const auto &id : extras)
      push(id);
  }
  return seq;
}

std::vector<std::string> ingestDocumentLike(InMemoryStore &store,
                                            const std::vector<std::string> &tokens,
                                            const Params &p) {
  auto memeIds = buildMemeSequence(store, tokens, p);
  for (size_t i = 0; i + 1 < memeIds.size(); ++i)
    store.linkConsecutive(memeIds[i], memeIds[i + 1], 1.0);
  return memeIds;
}

ExistenceResult runHarmlessExistence(const Params &p) {
  ExistenceResult out;
  const std::string nonce = kHarmlessNonce;

  /* Cue tokens occupy the first window; nonce sits past ngramMax so no
     official n-gram bag contains both. Recall must travel consecutive edges. */
  const std::vector<std::string> fact = {
      "north",  "dock",      "ledger", "records", "station", "paints",
      "night",  "signals",   "using",  "unique",  "named",   "locally",
      "after",  "harvest",   "season", "code",    nonce,     "never",
      "cargo",  "marks"};
  const std::vector<std::string> cue = {"north", "dock", "ledger"};
  const std::vector<std::string> distractor = {"lunar", "greenhouse", "harvest",
                                               "schedule", "alpha9"};
  const std::vector<std::string> distractorCue = {"lunar", "greenhouse",
                                                 "schedule"};

  out.queryMentionsNonce = wordsContain(cue, nonce);

  InMemoryStore store;
  ingestDocumentLike(store, fact, p);
  ingestDocumentLike(store, distractor, p);

  std::unordered_set<std::string> nonceMemes;
  for (const auto &id : store.nodeIds()) {
    if (wordsContain(store.wordsOfMeme(id), nonce))
      nonceMemes.insert(id);
  }
  out.nonceMemeCount = static_cast<int>(nonceMemes.size());
  std::vector<std::string> cueSeeds;
  for (const auto &w : cue) {
    for (const auto &mid : store.memesOfWord(w))
      cueSeeds.push_back(mid);
  }
  out.cueSeedCount = static_cast<int>(cueSeeds.size());
  out.hopsCueToNonce = bfsHops(store, cueSeeds, nonceMemes);

  out.cueReconstructedWords = reconstructWords(store, cue);
  out.kvmReconstructedNonce = wordsContain(out.cueReconstructedWords, nonce);
  out.cueRanked = diffuseFromQuery(store, cue, p);
  out.cueNonceScore = maxNonceScore(store, out.cueRanked, nonce);
  out.cueMaxScore = maxScore(out.cueRanked);
  out.diffusionRankedNonceMeme =
      diffusionActivatedNonce(out.cueNonceScore, out.cueMaxScore);

  const auto distWords = reconstructWords(store, distractorCue);
  out.distractorReconstructedNonce = wordsContain(distWords, nonce);
  const auto distRanked = diffuseFromQuery(store, distractorCue, p);
  out.distractorNonceScore = maxNonceScore(store, distRanked, nonce);
  out.distractorDiffusionRankedNonce =
      diffusionActivatedNonce(out.distractorNonceScore, maxScore(distRanked));
  return out;
}

std::vector<std::string> tokenizeAscii(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    if (cur.empty())
      return;
    out.push_back(cur);
    cur.clear();
  };
  for (unsigned char c : text) {
    if (std::isalnum(c) || c == '_' || c == '-') {
      cur.push_back(static_cast<char>(std::tolower(c)));
    } else {
      flush();
    }
  }
  flush();
  return out;
}

bool textExpressesPayload(const std::string &text, const std::string &payload) {
  if (payload.empty())
    return false;
  return wordsContain(tokenizeAscii(text), payload);
}

bool textExpressesHarmlessMeme(const std::string &text) {
  return textExpressesPayload(text, kHarmlessNonce);
}

std::vector<std::string> payloadFragments(const std::string &payload) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    if (cur.size() >= 2)
      out.push_back(cur);
    cur.clear();
  };
  bool letter = false;
  bool have = false;
  for (unsigned char c : payload) {
    const bool isLet = std::isalpha(c) != 0;
    const bool isDig = std::isdigit(c) != 0;
    if (!isLet && !isDig) {
      flush();
      have = false;
      continue;
    }
    const char lc = static_cast<char>(std::tolower(c));
    if (!have) {
      letter = isLet;
      cur.push_back(lc);
      have = true;
      continue;
    }
    if (isLet == letter)
      cur.push_back(lc);
    else {
      flush();
      letter = isLet;
      cur.push_back(lc);
    }
  }
  flush();
  return out;
}

namespace {

std::string compactAlnum(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    if (std::isalnum(c))
      out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

} // namespace

ExpressionJudgement judgeExpression(const std::string &text,
                                    const std::string &payload,
                                    const std::vector<std::string> &carrierTokens) {
  ExpressionJudgement j;
  if (payload.empty())
    return j;
  const auto tokens = tokenizeAscii(text);
  j.exactToken = wordsContain(tokens, payload);
  const std::string compactText = compactAlnum(text);
  const std::string compactPay = compactAlnum(payload);
  const bool contiguous =
      !compactPay.empty() && compactText.find(compactPay) != std::string::npos;
  const auto frags = payloadFragments(payload);
  bool allFrags = !frags.empty();
  for (const auto &f : frags) {
    bool hit = wordsContain(tokens, f);
    if (!hit) {
      for (const auto &t : tokens) {
        if (t.find(f) != std::string::npos) {
          hit = true;
          break;
        }
      }
    }
    if (!hit) {
      allFrags = false;
      break;
    }
  }
  j.reconstructed = !j.exactToken && (contiguous || allFrags);
  j.unitExpressed = j.exactToken || j.reconstructed;

  int carrierHits = 0;
  int carrierNeed = 0;
  for (const auto &c : carrierTokens) {
    if (c.empty() || c == payload)
      continue;
    carrierNeed++;
    if (wordsContain(tokens, c))
      carrierHits++;
  }
  const int need = carrierNeed <= 2 ? 1 : 2;
  j.carrierPresent = carrierHits >= need;
  j.latent = !j.unitExpressed && j.carrierPresent;
  j.drifted = !j.unitExpressed && !j.carrierPresent;
  return j;
}

bool unitPropagates(const std::vector<ExpressionJudgement> &rounds) {
  if (rounds.size() < 2)
    return false;
  bool sawUnit = false;
  bool sawGap = false;
  for (size_t i = 0; i < rounds.size(); ++i) {
    if (rounds[i].unitExpressed) {
      if (i >= 1)
        return true;
      sawUnit = true;
    } else if (sawUnit) {
      sawGap = true;
    }
  }
  (void)sawGap;
  return false;
}

std::string joinTokens(const std::vector<std::string> &tokens) {
  std::string out;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i)
      out += " ";
    out += tokens[i];
  }
  return out;
}

std::string expressActivated(const InMemoryStore &store,
                             const std::vector<std::string> &stimulus,
                             const Params &p, int surfaceId,
                             const std::string &payload) {
  if (store.empty() || stimulus.empty())
    return {};
  const auto ranked = diffuseFromQuery(store, stimulus, p);
  const double peak = maxScore(ranked);
  if (peak <= 1e-6)
    return {};

  std::unordered_map<std::string, double> wordScore;
  bool payloadActive = false;
  for (const auto &row : ranked) {
    if (row.second < 0.05 * peak)
      continue;
    for (const auto &w : store.wordsOfMeme(row.first)) {
      if (w.empty())
        continue;
      wordScore[w] = std::max(wordScore[w], row.second);
      if (!payload.empty() && w == payload)
        payloadActive = true;
    }
  }
  if (wordScore.empty())
    return {};

  std::vector<std::string> picked;
  if (payloadActive)
    picked.push_back(payload);
  std::vector<std::pair<double, std::string>> ordered;
  for (const auto &kv : wordScore) {
    if (!payload.empty() && kv.first == payload)
      continue;
    ordered.push_back({kv.second, kv.first});
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
    if (a.first != b.first)
      return a.first > b.first;
    return a.second < b.second;
  });
  for (const auto &row : ordered) {
    picked.push_back(row.second);
    if (picked.size() >= 6)
      break;
  }

  std::string body;
  const int surface = surfaceId >= 0 ? surfaceId % 3 : 0;
  if (surface == 1) {
    for (size_t i = 0; i < picked.size(); ++i) {
      if (i)
        body += ", ";
      body += picked[i];
    }
    return "record " + body;
  }
  if (surface == 2) {
    for (size_t i = 0; i < picked.size(); ++i) {
      if (i)
        body += " / ";
      body += picked[i];
    }
    return "station mark " + body;
  }
  for (size_t i = 0; i < picked.size(); ++i) {
    if (i)
      body += " ";
    body += picked[i];
  }
  return "noted " + body;
}

std::string graphNoteActivated(const InMemoryStore &store,
                               const std::vector<std::string> &stimulus,
                               const Params &p) {
  if (store.empty() || stimulus.empty())
    return {};
  const auto ranked = diffuseFromQuery(store, stimulus, p);
  const double peak = maxScore(ranked);
  if (peak <= 1e-6)
    return {};
  std::vector<std::pair<double, std::string>> ordered;
  std::unordered_set<std::string> seen;
  for (const auto &row : ranked) {
    if (row.second < 0.05 * peak)
      continue;
    for (const auto &w : store.wordsOfMeme(row.first)) {
      if (w.empty() || !seen.insert(w).second)
        continue;
      ordered.push_back({row.second, w});
    }
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
    if (a.first != b.first)
      return a.first > b.first;
    return a.second < b.second;
  });
  std::string body = "graph";
  size_t n = 0;
  for (const auto &row : ordered) {
    body += " ";
    body += row.second;
    if (++n >= 8)
      break;
  }
  return n == 0 ? std::string() : body;
}

namespace {

bool isStopWord(const std::string &w) {
  static const std::unordered_set<std::string> k = {
      "a",    "an",   "the",  "of",   "to",    "and",  "in",   "on",   "for",
      "is",   "was",  "with", "as",   "by",    "at",   "from", "that", "this",
      "it",   "be",   "or",   "are",  "were",  "been", "have", "has",  "had",
      "not",  "but",  "they", "their","which", "its",  "he",   "she",  "we",
      "you",  "his",  "her",  "them", "than",  "then", "also", "into", "over"};
  return k.count(w) > 0;
}

bool wordLooksHostile(const std::string &w) {
  static const std::unordered_set<std::string> k = {
      "http", "https", "www",   "ignore", "password", "weapon", "bomb",
      "kill", "suicide", "exploit", "malware", "chmod", "sudo"};
  return k.count(w) > 0;
}

bool memeLooksHarmless(const std::vector<std::string> &words) {
  int content = 0;
  for (const auto &w : words) {
    if (w.empty() || wordLooksHostile(w))
      return false;
    if (!isStopWord(w) && w.size() >= 4)
      content++;
  }
  return content >= 2 && !words.empty();
}

double jaccard(const std::vector<std::string> &a,
               const std::vector<std::string> &b) {
  std::unordered_set<std::string> sa(a.begin(), a.end());
  int inter = 0;
  for (const auto &w : b) {
    if (sa.count(w))
      inter++;
  }
  const int uni = static_cast<int>(sa.size() + b.size() - inter);
  if (uni <= 0)
    return 0.0;
  return static_cast<double>(inter) / static_cast<double>(uni);
}

std::vector<std::string> splitCorpusUnits(const std::string &text) {
  std::vector<std::string> units;
  std::string para;
  auto flushPara = [&]() {
    const std::string t = trimCopy(para);
    para.clear();
    if (t.size() < 40)
      return;
    std::string sent;
    for (size_t i = 0; i < t.size(); ++i) {
      sent.push_back(t[i]);
      if ((t[i] == '.' || t[i] == '!' || t[i] == '?') && sent.size() >= 40) {
        units.push_back(trimCopy(sent));
        sent.clear();
      }
    }
    const std::string tail = trimCopy(sent);
    if (tail.size() >= 40)
      units.push_back(tail);
  };
  for (char c : text) {
    if (c == '\n') {
      para.push_back(' ');
      if (para.size() >= 2 && para[para.size() - 2] == ' ')
        flushPara();
    } else {
      para.push_back(c);
    }
  }
  flushPara();
  return units;
}

phoenix::secamp::MemeTensorSpace tensorSpaceOf(const InMemoryStore &store) {
  const auto ids = store.nodeIds();
  std::vector<std::vector<std::string>> words;
  std::vector<std::vector<double>> alpha;
  words.reserve(ids.size());
  alpha.reserve(ids.size());
  for (const auto &id : ids) {
    const auto wts = store.weightsOfMeme(id);
    std::vector<std::string> ws;
    std::vector<double> a;
    double z = 0.0;
    ws.reserve(wts.size());
    a.reserve(wts.size());
    for (const auto &p : wts)
      z += p.second;
    for (const auto &p : wts) {
      ws.push_back(p.first);
      a.push_back(z > 1e-18 ? p.second / z : 0.0);
    }
    words.push_back(std::move(ws));
    alpha.push_back(std::move(a));
  }
  return phoenix::secamp::buildMemeTensorSpace(ids, words, alpha);
}

} // namespace

std::vector<std::string> highDfPriorTokens(const InMemoryStore &store) {
  std::unordered_map<std::string, int> df;
  for (const auto &id : store.nodeIds()) {
    std::unordered_set<std::string> seen;
    for (const auto &w : store.wordsOfMeme(id)) {
      if (w.empty() || !seen.insert(w).second)
        continue;
      df[w]++;
    }
  }
  const int cut = mappingHighDfCut(static_cast<int>(store.nodeCount()));
  std::vector<std::string> out;
  out.reserve(df.size());
  for (const auto &kv : df) {
    if (kv.second > cut)
      out.push_back(kv.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string carrierOfMeme(const InMemoryStore &store, const std::string &memeId) {
  const auto words = store.wordsOfMeme(memeId);
  std::vector<std::string> content;
  for (const auto &w : words) {
    if (!isStopWord(w) && w.size() >= 3)
      content.push_back(w);
    if (content.size() >= 12)
      break;
  }
  if (content.size() >= 2)
    return joinTokens(content);
  return joinTokens(words);
}

RagCarrier composeCarrierRag(const InMemoryStore &store,
                             const std::string &memeId,
                             const std::string &corpusText, int maxUnits,
                             int maxSents, bool requireExclusive) {
  RagCarrier out;
  if (memeId.empty() || store.empty() || corpusText.empty())
    return out;
  const auto memeWords = store.wordsOfMeme(memeId);
  if (memeWords.empty())
    return out;

  const auto space = tensorSpaceOf(store);
  if (!space.index.count(memeId))
    return out;

  const auto weights = store.weightsOfMeme(memeId);
  std::unordered_map<std::string, double> alpha;
  double contentZ = 0.0;
  for (const auto &wp : weights) {
    alpha[wp.first] = wp.second;
    if (!wp.first.empty() && !isStopWord(wp.first) && wp.first.size() >= 3)
      contentZ += wp.second;
  }
  std::unordered_set<std::string> memeSet(memeWords.begin(), memeWords.end());
  std::unordered_set<std::string> content;
  for (const auto &w : memeWords) {
    if (!w.empty() && !isStopWord(w) && w.size() >= 3)
      content.insert(w);
  }

  const auto units = splitCorpusUnits(corpusText);
  const int cap = std::max(1, maxUnits);
  const int take = std::max(1, maxSents);
  std::unordered_map<std::string, int> df;
  for (const auto &id : store.nodeIds()) {
    std::unordered_set<std::string> seen;
    for (const auto &w : store.wordsOfMeme(id)) {
      if (w.empty() || !seen.insert(w).second)
        continue;
      df[w]++;
    }
  }
  const int nColl = std::max(1, static_cast<int>(store.nodeCount()));
  const auto typical = mappingTypicalAlpha(weights);
  std::vector<std::string> typicalWords;
  typicalWords.reserve(typical.size());
  for (const auto &kv : typical)
    typicalWords.push_back(kv.first);
  if (typicalWords.empty()) {
    for (const auto &w : memeWords) {
      if (!w.empty() && !isStopWord(w) && w.size() >= 4)
        typicalWords.push_back(w);
    }
  }
  const auto rareWords = mappingRareTypical(weights, df, nColl);
  const int needTyp =
      typicalWords.empty()
          ? 0
          : std::max(1, (static_cast<int>(typicalWords.size()) + 1) / 2);
  const int needRare =
      rareWords.empty()
          ? 0
          : (static_cast<int>(rareWords.size()) <= 3
                 ? static_cast<int>(rareWords.size())
                 : std::max(2, static_cast<int>(rareWords.size()) - 1));
  const int scanCap = std::max(cap * 32, 2048);

  auto typicalHitsIn = [&](const std::vector<std::string> &tokens) -> int {
    if (typicalWords.empty())
      return 0;
    std::unordered_set<std::string> have(tokens.begin(), tokens.end());
    int n = 0;
    for (const auto &w : typicalWords)
      if (have.count(w))
        ++n;
    return n;
  };
  auto rareHitsIn = [&](const std::vector<std::string> &tokens) -> int {
    if (rareWords.empty())
      return 0;
    std::unordered_set<std::string> have(tokens.begin(), tokens.end());
    int n = 0;
    for (const auto &w : rareWords)
      if (have.count(w))
        ++n;
    return n;
  };

  struct Sent {
    int idx{0};
    std::string text;
    std::vector<std::string> tokens;
    double bagCos{0.0};
    double semanticCos{0.0};
    double orderLift{0.0};
    double score{0.0};
    int wordHit{0};
    int typHit{0};
    double typCover{0.0};
  };
  std::vector<Sent> sents;
  int scannedLong = 0;
  auto consider = [&](int i, const std::string &unit,
                      const std::vector<std::string> &tokens, int typHit) {
    std::unordered_set<std::string> seen;
    int wordHit = 0;
    double overlapMass = 0.0;
    for (const auto &w : tokens) {
      if (!memeSet.count(w) || !seen.insert(w).second)
        continue;
      ++wordHit;
      auto ait = alpha.find(w);
      if (ait != alpha.end() && !isStopWord(w))
        overlapMass += ait->second;
    }
    if (wordHit < 2)
      return;
    const auto tq = phoenix::secamp::queryMemeTensor(space, tokens, memeId);
    Sent s;
    s.idx = i;
    s.text = unit;
    s.tokens = tokens;
    s.bagCos = tq.bagCos;
    s.semanticCos = tq.semanticCos;
    s.orderLift = tq.orderLift;
    s.wordHit = wordHit;
    s.typHit = typHit;
    s.typCover =
        typicalWords.empty()
            ? 0.0
            : static_cast<double>(typHit) /
                  static_cast<double>(typicalWords.size());
    const double cov =
        contentZ > 1e-12
            ? overlapMass / contentZ
            : static_cast<double>(wordHit) /
                  static_cast<double>(std::max<size_t>(1, content.size()));
    s.score = 0.50 * tq.bagCos + 0.35 * tq.semanticCos +
              0.15 * std::min(1.0, cov) + 0.20 * s.typCover;
    sents.push_back(std::move(s));
  };

  for (int i = 0; i < static_cast<int>(units.size()); ++i) {
    const auto &unit = units[static_cast<size_t>(i)];
    int letters = 0;
    for (unsigned char c : unit) {
      if (std::isalpha(c))
        ++letters;
    }
    if (letters < 48)
      continue;
    const auto tokens = tokenizeAscii(unit);
    if (static_cast<int>(tokens.size()) < 10)
      continue;
    ++scannedLong;
    const int typHit = typicalHitsIn(tokens);
    const int rareHit = rareHitsIn(tokens);
    const bool typOk =
        needRare > 0 ? rareHit >= needRare : (needTyp <= 0 || typHit >= needTyp);
    if (typOk)
      consider(i, unit, tokens, typHit);
    if (static_cast<int>(sents.size()) >= cap)
      break;
    const int stopScan = needRare > 0 ? scanCap * 4 : scanCap;
    if (scannedLong >= stopScan && static_cast<int>(sents.size()) >= 1)
      break;
  }
  if (sents.empty() && needRare <= 0) {
    scannedLong = 0;
    for (int i = 0; i < static_cast<int>(units.size()); ++i) {
      const auto &unit = units[static_cast<size_t>(i)];
      int letters = 0;
      for (unsigned char c : unit) {
        if (std::isalpha(c))
          ++letters;
      }
      if (letters < 48)
        continue;
      const auto tokens = tokenizeAscii(unit);
      if (static_cast<int>(tokens.size()) < 10)
        continue;
      ++scannedLong;
      consider(i, unit, tokens, typicalHitsIn(tokens));
      if (static_cast<int>(sents.size()) >= cap || scannedLong >= cap)
        break;
    }
  }
  if (sents.empty())
    return out;
  std::sort(sents.begin(), sents.end(), [](const Sent &a, const Sent &b) {
    if (a.typCover != b.typCover)
      return a.typCover > b.typCover;
    if (a.score != b.score)
      return a.score > b.score;
    return a.idx < b.idx;
  });
  const int pool = std::min(48, static_cast<int>(sents.size()));
  double bestPick = -1.0;
  double bestExcl = -1.0;
  double bestLift = 0.0;
  int bestI = 0;
  int bestRank = -1;
  int bestOthers = 0;
  auto exclusivityOf = [&](const std::string &text, double *liftOut, int *rankOut,
                           int *othersOut) -> double {
    const auto acts = detectActivatedMemes(store, text, {}, 16);
    double target = 0.0;
    double presentMass = 0.0;
    int others = 0;
    int trank = -1;
    bool present = false;
    double tlift = 0.0;
    for (const auto &a : acts) {
      if (!a.present)
        continue;
      presentMass += a.score;
      if (a.id == memeId) {
        present = true;
        target = a.score;
        trank = a.rank;
        tlift = a.lift;
      } else {
        ++others;
      }
    }
    if (liftOut)
      *liftOut = tlift;
    if (rankOut)
      *rankOut = trank;
    if (othersOut)
      *othersOut = others;
    if (!present || presentMass <= 1e-12)
      return -1.0;
    return target / presentMass;
  };
  auto pickScore = [](double excl, int trank, int others, double typCover,
                      int wordHit, double lift) {
    double s = excl * 4.0;
    if (trank == 0)
      s += 3.0;
    else if (trank >= 0 && trank <= 2)
      s += 0.4;
    if (excl >= 0.5)
      s += 2.0;
    if (excl >= 0.8)
      s += 1.0;
    s += 1.5 * typCover;
    s -= 0.5 * static_cast<double>(others);
    s += 0.02 * static_cast<double>(wordHit);
    s += 0.001 * std::min(lift, 100.0);
    return s;
  };
  struct Eval {
    int i{0};
    double excl{0.0};
    double lift{0.0};
    int trank{-1};
    int others{0};
    bool exclusive{false};
  };
  std::vector<Eval> evals;
  bool haveExclusive = false;
  for (int i = 0; i < pool; ++i) {
    Eval e;
    e.i = i;
    e.excl = exclusivityOf(sents[static_cast<size_t>(i)].text, &e.lift,
                           &e.trank, &e.others);
    if (e.excl < 0.0)
      continue;
    e.exclusive = (e.trank == 0 || e.excl >= 0.5);
    if (e.exclusive)
      haveExclusive = true;
    if (static_cast<int>(out.sources.size()) < 8)
      out.sources.push_back(sents[static_cast<size_t>(i)].text);
    evals.push_back(e);
  }
  for (const auto &e : evals) {
    if (haveExclusive && !e.exclusive)
      continue;
    if (requireExclusive && !e.exclusive)
      continue;
    const double rank = pickScore(e.excl, e.trank, e.others,
                                  sents[static_cast<size_t>(e.i)].typCover,
                                  sents[static_cast<size_t>(e.i)].wordHit,
                                  e.lift);
    if (rank > bestPick) {
      bestPick = rank;
      bestExcl = e.excl;
      bestLift = e.lift;
      bestI = e.i;
      bestRank = e.trank;
      bestOthers = e.others;
    }
  }
  if (bestPick < 0.0 && !requireExclusive) {
    for (int i = pool; i < static_cast<int>(sents.size()); ++i) {
      double lift = 0.0;
      int trank = -1;
      int others = 0;
      const double excl = exclusivityOf(sents[static_cast<size_t>(i)].text, &lift,
                                        &trank, &others);
      if (excl < 0.0)
        continue;
      if (static_cast<int>(out.sources.size()) < 8)
        out.sources.push_back(sents[static_cast<size_t>(i)].text);
      bestPick = pickScore(excl, trank, others,
                           sents[static_cast<size_t>(i)].typCover,
                           sents[static_cast<size_t>(i)].wordHit, lift);
      bestExcl = excl;
      bestLift = lift;
      bestI = i;
      bestRank = trank;
      bestOthers = others;
      break;
    }
  }
  if (bestPick < 0.0 && needRare > 0 && !requireExclusive) {
    const int oldN = static_cast<int>(sents.size());
    int fbScan = 0;
    for (int i = 0; i < static_cast<int>(units.size()); ++i) {
      const auto &unit = units[static_cast<size_t>(i)];
      int letters = 0;
      for (unsigned char c : unit) {
        if (std::isalpha(c))
          ++letters;
      }
      if (letters < 48)
        continue;
      const auto tokens = tokenizeAscii(unit);
      if (static_cast<int>(tokens.size()) < 10)
        continue;
      ++fbScan;
      const int typHit = typicalHitsIn(tokens);
      if (needTyp > 0 && typHit < needTyp)
        continue;
      consider(i, unit, tokens, typHit);
      if (static_cast<int>(sents.size()) - oldN >= cap || fbScan >= cap)
        break;
    }
    for (int i = oldN; i < static_cast<int>(sents.size()); ++i) {
      double lift = 0.0;
      int trank = -1;
      int others = 0;
      const double excl = exclusivityOf(sents[static_cast<size_t>(i)].text, &lift,
                                        &trank, &others);
      if (excl < 0.0)
        continue;
      const double rank = pickScore(excl, trank, others,
                                    sents[static_cast<size_t>(i)].typCover,
                                    sents[static_cast<size_t>(i)].wordHit, lift);
      if (rank > bestPick) {
        bestPick = rank;
        bestExcl = excl;
        bestLift = lift;
        bestI = i;
        bestRank = trank;
        bestOthers = others;
      }
    }
  }
  if (bestPick < 0.0 || sents.empty())
    return out;
  const Sent &best = sents[static_cast<size_t>(bestI)];

  out.text = best.text;
  out.unitIndex = {best.idx};
  out.units = {best.text};
  out.score = best.score;
  out.wordHit = best.wordHit;
  out.bagCos = best.bagCos;
  out.semanticCos = best.semanticCos;
  out.orderLift = best.orderLift;
  out.method = "tensor-sentence";
  out.exclusivity = bestExcl > 0.0 ? bestExcl : 0.0;
  out.targetRank = bestRank;
  out.otherPresent = bestOthers;
  out.typicalCover = best.typCover;

  if (take > 1 && bestPick >= 0.0 && bestOthers == 0 && !requireExclusive) {
    int addI = -1;
    double addLift = bestLift;
    const int partnerCap = std::min(24, static_cast<int>(sents.size()));
    for (int i = 0; i < partnerCap; ++i) {
      if (i == bestI)
        continue;
      std::string cat = best.text + " " + sents[static_cast<size_t>(i)].text;
      double lift = 0.0;
      int trank = -1;
      int others = 0;
      const double excl = exclusivityOf(cat, &lift, &trank, &others);
      if (excl < 0.0 || excl + 1e-9 < bestExcl)
        continue;
      if (others != 0)
        continue;
      if (bestRank >= 0 && (trank < 0 || trank > bestRank))
        continue;
      if (lift <= addLift)
        continue;
      addLift = lift;
      addI = i;
    }
    if (addI >= 0) {
      out.text += " ";
      out.text += sents[static_cast<size_t>(addI)].text;
      out.unitIndex.push_back(sents[static_cast<size_t>(addI)].idx);
      out.units.push_back(sents[static_cast<size_t>(addI)].text);
      out.wordHit += sents[static_cast<size_t>(addI)].wordHit;
      const auto tok = tokenizeAscii(out.text);
      const auto tq = phoenix::secamp::queryMemeTensor(space, tok, memeId);
      out.bagCos = tq.bagCos;
      out.semanticCos = tq.semanticCos;
      out.orderLift = tq.orderLift;
      out.method = "tensor-sentences";
    }
  }
  return out;
}

MemeDetection detectMemeInText(const InMemoryStore &instrument,
                               const std::string &text,
                               const std::string &memeId, const Params &p,
                               bool withTensor) {
  MemeDetection d;
  if (memeId.empty() || instrument.empty())
    return d;
  const auto tokens = tokenizeAscii(text);
  d.queryId = mappingNewQueryId();
  MappingMassFrom from;
  const auto ranked = diffuseFromQuery(instrument, tokens, p, false, &from);
  d.trace = d.queryId + "|" + memeId + "<-" + mappingFormatFrom(memeId, from);
  /* Lift vs the graph's high-df vocabulary, not vs high-df words inside
     this query. A stop-only query is a subset of that prior on large
     graphs; on small graphs mid-df stops are blocked by content evidence. */
  const auto prior = highDfPriorTokens(instrument);
  const auto nullRanked =
      prior.empty() ? diffuseFromQuery(instrument, tokens, p, true)
                    : diffuseFromQuery(instrument, prior, p, false);
  d.peakScore = maxScore(ranked);
  for (size_t i = 0; i < ranked.size(); ++i) {
    if (ranked[i].first != memeId)
      continue;
    d.memeScore = ranked[i].second;
    d.rank = static_cast<int>(i);
    break;
  }
  for (const auto &row : nullRanked) {
    if (row.first == memeId) {
      d.nullScore = row.second;
      break;
    }
  }
  d.lift = d.memeScore / std::max(d.nullScore, 1e-12);
  d.present = d.memeScore > 1e-6 && d.peakScore > 1e-6 &&
              d.memeScore >= kGnnPresentPeakFrac * d.peakScore &&
              d.lift >= kGnnPresentLift &&
              mappingQueryHasContentForMeme(tokens,
                                            instrument.wordsOfMeme(memeId));
  if (!withTensor)
    return d;
  {
    const auto space = tensorSpaceOf(instrument);
    const auto tq = phoenix::secamp::queryMemeTensor(space, tokens, memeId);
    d.semanticCos = tq.semanticCos;
    d.bagCos = tq.bagCos;
    d.reversedCos = tq.reversedCos;
    d.orderLift = tq.orderLift;
    d.tensorRank = tq.tensorRank;
    d.nearestWord = tq.nearestWord;
    d.nearestSentence = tq.nearestSentence;
  }
  return d;
}

std::vector<MemeActivation> detectActivatedMemes(const InMemoryStore &instrument,
                                                 const std::string &text,
                                                 const Params &p, int maxKeep) {
  std::vector<MemeActivation> out;
  if (instrument.empty() || text.empty())
    return out;
  const auto tokens = tokenizeAscii(text);
  const std::string qid = mappingNewQueryId();
  MappingMassFrom from;
  const auto ranked = diffuseFromQuery(instrument, tokens, p, false, &from);
  const auto prior = highDfPriorTokens(instrument);
  const auto nullRanked =
      prior.empty() ? diffuseFromQuery(instrument, tokens, p, true)
                    : diffuseFromQuery(instrument, prior, p, false);
  std::unordered_map<std::string, double> nullOf;
  for (const auto &row : nullRanked)
    nullOf[row.first] = row.second;
  const double peak = maxScore(ranked);
  const int cap = std::max(1, maxKeep);
  for (size_t i = 0; i < ranked.size() && static_cast<int>(out.size()) < cap; ++i) {
    MemeActivation a;
    a.id = ranked[i].first;
    a.score = ranked[i].second;
    a.rank = static_cast<int>(i);
    a.lift = a.score / std::max(nullOf[a.id], 1e-12);
    a.present = a.score > 1e-6 && peak > 1e-6 &&
                a.score >= kGnnPresentPeakFrac * peak &&
                a.lift >= kGnnPresentLift &&
                mappingQueryHasContentForMeme(tokens,
                                              instrument.wordsOfMeme(a.id));
    a.trace = qid + "|" + a.id + "<-" + mappingFormatFrom(a.id, from);
    out.push_back(std::move(a));
  }
  return out;
}

std::vector<ScreenedMeme> screenMemesFromBarrier(const InMemoryStore &store,
                                                 int maxKeep) {
  std::vector<ScreenedMeme> rich;
  std::vector<ScreenedMeme> thin;
  double sum = 0.0;
  double sum2 = 0.0;
  int n = 0;
  for (const auto &id : store.nodeIds()) {
    const int deg = static_cast<int>(store.neighborsOf(id).size());
    sum += deg;
    sum2 += static_cast<double>(deg) * deg;
    n++;
  }
  const double mean = n > 0 ? sum / n : 0.0;
  const double var = n > 0 ? sum2 / n - mean * mean : 0.0;
  const double stdv = var > 1e-9 ? std::sqrt(var) : 1.0;
  for (const auto &id : store.nodeIds()) {
    const auto words = store.wordsOfMeme(id);
    int content = 0;
    bool hostile = false;
    for (const auto &w : words) {
      if (wordLooksHostile(w))
        hostile = true;
      if (!isStopWord(w) && w.size() >= 4)
        content++;
    }
    if (hostile || content < 1)
      continue;
    if (static_cast<int>(words.size()) > Params{}.ngramMax)
      continue;
    ScreenedMeme row;
    row.id = id;
    row.words = words;
    row.carrier = carrierOfMeme(store, id);
    row.degree = static_cast<int>(store.neighborsOf(id).size());
    row.barrierScore = (static_cast<double>(row.degree) - mean) / stdv;
    if (content >= 2)
      rich.push_back(std::move(row));
    else
      thin.push_back(std::move(row));
  }
  auto byScore = [](const ScreenedMeme &a, const ScreenedMeme &b) {
    if (a.barrierScore != b.barrierScore)
      return a.barrierScore > b.barrierScore;
    return a.id < b.id;
  };
  std::sort(rich.begin(), rich.end(), byScore);
  std::sort(thin.begin(), thin.end(), byScore);
  std::vector<ScreenedMeme> kept;
  const int cap = std::max(1, maxKeep);
  auto take = [&](const std::vector<ScreenedMeme> &src) {
    for (const auto &row : src) {
      if (static_cast<int>(kept.size()) >= cap)
        return;
      bool near = false;
      for (const auto &k : kept) {
        if (jaccard(row.words, k.words) >= 0.75) {
          near = true;
          break;
        }
      }
      if (!near)
        kept.push_back(row);
    }
  };
  take(rich);
  take(thin);
  return kept;
}

int ingestCorpusText(InMemoryStore &store, const std::string &text, int maxUnits,
                     const Params &p) {
  const auto units = splitCorpusUnits(text);
  int n = 0;
  const int cap = std::max(1, maxUnits);
  for (const auto &u : units) {
    const auto tokens = tokenizeAscii(u);
    if (tokens.size() < 6)
      continue;
    ingestDocumentLike(store, tokens, p);
    if (++n >= cap)
      break;
  }
  return n;
}

int ingestCorpusFile(InMemoryStore &store, const std::string &path, int maxUnits,
                     const Params &p) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return 0;
  std::ostringstream raw;
  raw << in.rdbuf();
  return ingestCorpusText(store, raw.str(), maxUnits, p);
}

namespace {

bool anyLeaked(const SerialReplicationResult &r, const std::string &forbidden) {
  if (forbidden.empty())
    return false;
  if (textExpressesPayload(r.distractorOutput, forbidden))
    return true;
  for (const auto &rnd : r.rounds) {
    if (textExpressesPayload(rnd.text, forbidden))
      return true;
  }
  return false;
}

bool heldOutFindsPayload(const InMemoryStore &store,
                         const std::vector<std::string> &held,
                         const std::string &payload, const Params &p) {
  if (held.empty() || payload.empty())
    return false;
  bool heldBound = false;
  for (const auto &w : held) {
    if (!store.memesOfWord(w).empty())
      heldBound = true;
  }
  if (!heldBound)
    return false;
  if (wordsContain(reconstructWords(store, held), payload))
    return true;
  const auto ranked = diffuseFromQuery(store, held, p);
  return diffusionActivatedNonce(maxNonceScore(store, ranked, payload),
                                 maxScore(ranked));
}

void markSerialPassed(SerialReplicationResult &r, int generations) {
  if (r.residueDetected || r.leakedForbidden || r.distractorExpressesMeme) {
    r.passed = false;
    return;
  }
  if (static_cast<int>(r.rounds.size()) < generations) {
    r.passed = false;
    return;
  }
  for (const auto &rnd : r.rounds) {
    if (!rnd.emptyStoreBeforeIngest || rnd.decoderEchoOnEmpty ||
        rnd.heldOutResidue) {
      r.passed = false;
      return;
    }
  }
  if (!r.expectExpress) {
    r.passed = r.expressedRounds == 0;
    return;
  }
  r.passed = r.expressedRounds == generations;
}

} // namespace

SerialReplicationResult runSerialReplicationCase(const SerialCase &c) {
  SerialReplicationResult out;
  out.id = c.id;
  out.payload = c.payload;
  out.expectExpress = c.expectExpress;
  out.input = joinTokens(c.inputTokens);
  const int gens = std::max(kMinStabilityRounds, c.generations);
  std::string current = out.input;

  for (int r = 0; r < gens; ++r) {
    SerialRound rnd;
    const auto incoming = tokenizeAscii(current);
    if (incoming.empty()) {
      out.residueDetected = true;
      break;
    }

    InMemoryStore probe;
    rnd.emptyStoreBeforeIngest = probe.empty() && probe.nodeCount() == 0;
    const std::string echoed =
        expressActivated(probe, incoming, c.params, r, c.payload);
    rnd.decoderEchoOnEmpty = textExpressesPayload(echoed, c.payload) ||
                             !echoed.empty();
    if (rnd.decoderEchoOnEmpty)
      out.residueDetected = true;

    InMemoryStore live;
    if (r == 0 && c.interfereFirst && !c.distractorTokens.empty())
      ingestDocumentLike(live, c.distractorTokens, c.params);
    ingestDocumentLike(live, incoming, c.params);
    rnd.text = expressActivated(live, incoming, c.params, r, c.payload);
    rnd.judge = judgeExpression(rnd.text, c.payload, c.inputTokens);
    rnd.expresses = rnd.judge.exactToken;
    if (rnd.expresses)
      out.expressedRounds++;
    if (rnd.judge.unitExpressed)
      out.unitRounds++;
    if (rnd.judge.latent)
      out.latentRounds++;
    if (rnd.judge.drifted)
      out.driftRounds++;

    std::vector<std::string> held;
    for (const auto &t : c.inputTokens) {
      if (t == c.payload)
        continue;
      if (!wordsContain(incoming, t))
        held.push_back(t);
    }
    rnd.heldOutResidue = heldOutFindsPayload(live, held, c.payload, c.params);
    if (rnd.heldOutResidue)
      out.residueDetected = true;

    out.rounds.push_back(rnd);
    if (rnd.text.empty())
      break;
    current = rnd.text;
  }

  InMemoryStore distStore;
  auto distractor = c.distractorTokens;
  if (distractor.empty())
    distractor = {"lunar", "greenhouse", "harvest", "schedule"};
  ingestDocumentLike(distStore, distractor, c.params);
  out.distractorOutput =
      expressActivated(distStore, distractor, c.params, 0, c.payload);
  out.distractorExpressesMeme =
      textExpressesPayload(out.distractorOutput, c.payload);
  out.leakedForbidden = anyLeaked(out, c.forbiddenPayload);
  markSerialPassed(out, gens);
  std::vector<ExpressionJudgement> judges;
  judges.reserve(out.rounds.size());
  bool seenUnit = false;
  bool gap = false;
  for (const auto &rnd : out.rounds) {
    judges.push_back(rnd.judge);
    if (rnd.judge.unitExpressed) {
      if (seenUnit && gap)
        out.returnedAfterGap = true;
      seenUnit = true;
    } else if (seenUnit) {
      gap = true;
    }
  }
  if (c.expectExpress)
    out.propagationPassed = unitPropagates(judges) && !out.leakedForbidden;
  else
    out.propagationPassed = out.unitRounds == 0 && !out.leakedForbidden;
  return out;
}

SerialReplicationResult runSerialReplication(const Params &p) {
  SerialCase c;
  c.id = "baseline";
  c.payload = kHarmlessNonce;
  c.inputTokens = {"north", "dock", "ceremonial", "pigment", "code",
                   kHarmlessNonce};
  c.params = p;
  return runSerialReplicationCase(c);
}

SerialBatteryReport runSerialReplicationBattery() {
  SerialBatteryReport report;
  Params tight;
  tight.ngramMin = 2;
  tight.ngramMax = 5;
  Params noMerge;
  noMerge.minOverlap = 99;

  const int kRounds = kMinStabilityRounds;
  const std::vector<SerialCase> cases = {
      {"G1-baseline-dock",
       kHarmlessNonce,
       {"north", "dock", "ceremonial", "pigment", "code", kHarmlessNonce},
       {"lunar", "greenhouse", "harvest", "schedule"},
       {},
       kRounds,
       true,
       "ambern9k2",
       false},
      {"G2-orchard-amber",
       "ambern9k2",
       {"west", "orchard", "harvest", "label", "ambern9k2"},
       {"polar", "survey", "beacon", "grid"},
       {},
       kRounds,
       true,
       kHarmlessNonce,
       false},
      {"G3-river-teal",
       "tealridge08",
       {"south", "river", "survey", "marker", "tealridge08"},
       {"bakery", "inventory", "shelf", "count"},
       {},
       kRounds,
       true,
       kHarmlessNonce,
       false},
      {"G4-mineral-slate",
       "slateq4m",
       {"east", "quarry", "sample", "ticket", "slateq4m"},
       {"clinic", "waiting", "room", "list"},
       {},
       kRounds,
       true,
       kHarmlessNonce,
       false},
      {"G5-reversed-order",
       kHarmlessNonce,
       {kHarmlessNonce, "code", "pigment", "ceremonial", "dock", "north"},
       {"lunar", "greenhouse", "harvest", "schedule"},
       {},
       kRounds,
       true,
       "tealridge08",
       false},
      {"G6-payload-medial",
       kHarmlessNonce,
       {"north", "dock", kHarmlessNonce, "ceremonial", "pigment", "code"},
       {"museum", "catalog", "entry", "index"},
       {},
       kRounds,
       true,
       "slateq4m",
       false},
      {"G7-nomerge-overlap",
       kHarmlessNonce,
       {"north", "dock", "ceremonial", "pigment", "code", kHarmlessNonce},
       {"lunar", "greenhouse", "harvest", "schedule"},
       noMerge,
       kRounds,
       true,
       "",
       false},
      {"G8-tight-ngrams",
       "ambern9k2",
       {"west", "orchard", "harvest", "label", "ambern9k2"},
       {"polar", "survey", "beacon", "grid"},
       tight,
       kRounds,
       true,
       "",
       false},
      {"G9-interfere-then-serial",
       "tealridge08",
       {"south", "river", "survey", "marker", "tealridge08"},
       {"bakery", "inventory", "shelf", "count"},
       {},
       kRounds,
       true,
       kHarmlessNonce,
       true},
      {"G10-control-no-payload",
       kHarmlessNonce,
       {"north", "dock", "ceremonial", "pigment", "code"},
       {"lunar", "greenhouse", "harvest", "schedule"},
       {},
       kRounds,
       false,
       "",
       false},
      {"G11-control-sibling-token",
       "qx7celadon46",
       {"north", "dock", "ceremonial", "pigment", "code", "qx7celadon46"},
       {"lunar", "greenhouse", "harvest", "schedule"},
       {},
       kRounds,
       true,
       kHarmlessNonce,
       false},
      {"G12-control-unrelated-input",
       kHarmlessNonce,
       {"bakery", "inventory", "shelf", "count", "alpha9"},
       {"polar", "survey", "beacon", "grid"},
       {},
       kRounds,
       false,
       "",
       false},
  };

  for (const auto &c : cases) {
    auto row = runSerialReplicationCase(c);
    if (c.expectExpress) {
      report.positiveTotal++;
      if (row.passed)
        report.positivePassed++;
    } else {
      report.negativeTotal++;
      if (row.passed)
        report.negativePassed++;
    }
    report.groups.push_back(std::move(row));
  }
  return report;
}

} // namespace memetic
} // namespace phoenix
