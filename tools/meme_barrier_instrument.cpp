/* Official readme 统计/识别 on a frozen instrument. Not llama. No writeback. */
#include "memetic_existence.hpp"
#include "addons/ThePlugInForSecurity/security_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using phoenix::memetic::InMemoryStore;
using phoenix::memetic::detectMemeInText;
using phoenix::memetic::ingestCorpusFile;
using phoenix::memetic::ingestCorpusText;
using phoenix::memetic::mappingMergeStopWord;
using phoenix::secamp::DiscreteGraph;
using phoenix::secamp::GraphEdge;
using phoenix::secamp::analyzeGraph;

static DiscreteGraph graphFromStore(const InMemoryStore &store) {
  DiscreteGraph g;
  const auto ids = store.nodeIds();
  g.ids = ids;
  g.layers.assign(ids.size(), "meme");
  g.mapped.resize(ids.size());
  std::unordered_map<std::string, int> idx;
  for (int i = 0; i < static_cast<int>(ids.size()); ++i)
    idx[ids[static_cast<size_t>(i)]] = i;
  for (size_t i = 0; i < ids.size(); ++i)
    g.mapped[i] = store.wordsOfMeme(ids[i]);
  for (size_t i = 0; i < ids.size(); ++i) {
    for (const auto &nb : store.neighborsOf(ids[i])) {
      auto it = idx.find(nb.first);
      if (it == idx.end() || it->second <= static_cast<int>(i))
        continue;
      GraphEdge e;
      e.from = static_cast<int>(i);
      e.to = it->second;
      e.weight = nb.second;
      g.edges.push_back(e);
    }
  }
  return g;
}

static bool isCatalogToken(const std::string &w) {
  int digits = 0;
  int letters = 0;
  for (unsigned char c : w) {
    if (c >= '0' && c <= '9')
      ++digits;
    else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
      ++letters;
  }
  return digits >= 2 && digits >= letters;
}

static bool isCatalogBag(const std::vector<std::string> &words) {
  int cat = 0;
  for (const auto &w : words) {
    if (isCatalogToken(w))
      ++cat;
  }
  return cat >= 2;
}

static bool isPhraseMeme(const std::string &id,
                         const std::vector<std::string> &words) {
  if (id.rfind("meme_p_", 0) == 0)
    return words.size() >= 2;
  int content = 0;
  for (const auto &w : words) {
    if (w.size() >= 3)
      ++content;
  }
  return content >= 3;
}

static std::string fullCarrier(const InMemoryStore &store,
                               const std::string &memeId) {
  const auto words = store.wordsOfMeme(memeId);
  std::string out;
  for (const auto &w : words) {
    if (!out.empty())
      out += " ";
    out += w;
  }
  return out;
}

static void jsonEscape(const std::string &s) {
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      std::cout << '\\';
    if (c == '\n')
      std::cout << "\\n";
    else if (c == '\r')
      std::cout << "\\r";
    else
      std::cout << static_cast<char>(c);
  }
}

/* Carrier-health statistics on the continuation manifold: unigram entropy,
   top-token dominance, longest single-token run. A verbatim-repeat carrier
   sits next to the token-collapse basin ("aalborg aalborg ..." ate two
   memes at round 5 of the dialogue serial); a collapsed output must not be
   reingested. Statistics only, not a present gate. */
struct TextStats {
  int tokens{0};
  double entropy{0.0};
  double top1Frac{0.0};
  int maxRun{0};
  bool collapsed{false};
};

static TextStats textStatsOf(const std::string &text) {
  TextStats st;
  std::vector<std::string> toks;
  std::string cur;
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z')
      cur += static_cast<char>(c - 'A' + 'a');
    else if (c >= 'a' && c <= 'z')
      cur += static_cast<char>(c);
    else if (!cur.empty()) {
      toks.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty())
    toks.push_back(cur);
  st.tokens = static_cast<int>(toks.size());
  if (toks.empty())
    return st;
  std::unordered_map<std::string, int> freq;
  for (const auto &t : toks)
    ++freq[t];
  double h = 0.0;
  int top = 0;
  for (const auto &kv : freq) {
    const double p = static_cast<double>(kv.second) / toks.size();
    h -= p * std::log2(p);
    top = std::max(top, kv.second);
  }
  st.entropy = h;
  st.top1Frac = static_cast<double>(top) / toks.size();
  int run = 1;
  for (size_t i = 1; i < toks.size(); ++i) {
    if (toks[i] == toks[i - 1]) {
      ++run;
      st.maxRun = std::max(st.maxRun, run);
    } else {
      run = 1;
    }
  }
  if (st.maxRun < 1)
    st.maxRun = 1;
  st.collapsed = st.maxRun >= 8 || st.top1Frac >= 0.5;
  return st;
}

static void printWordArray(const std::vector<std::string> &ws) {
  std::cout << "[";
  for (size_t i = 0; i < ws.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << "\"";
    jsonEscape(ws[i]);
    std::cout << "\"";
  }
  std::cout << "]";
}

static void printDetect(const InMemoryStore &store, const std::string &text,
                        const std::string &memeId) {
  const auto d = detectMemeInText(store, text, memeId, {});
  std::cout << "present=" << (d.present ? "1" : "0")
            << " activation=" << d.memeScore << " peak=" << d.peakScore
            << " rank=" << d.rank << " lift=" << d.lift
            << " null=" << d.nullScore
            << " semantic=" << d.semanticCos << " bag=" << d.bagCos
            << " rev=" << d.reversedCos << " order=" << d.orderLift
            << " trank=" << d.tensorRank
            << " nword=" << d.nearestWord
            << " nsent=" << d.nearestSentence
            << " qid=" << d.queryId
            << " trace=" << d.trace;
  const auto acts = phoenix::memetic::detectActivatedMemes(store, text, {}, 8);
  std::cout << " activated=";
  for (size_t i = 0; i < acts.size(); ++i) {
    if (i)
      std::cout << ",";
    std::cout << acts[i].id << ":" << (acts[i].present ? "1" : "0") << ":"
              << acts[i].lift << ":" << acts[i].rank;
  }
  std::cout << "\n";
}

static std::string readStdin() {
  std::ostringstream raw;
  raw << std::cin.rdbuf();
  return raw.str();
}

int main(int argc, char **argv) {
  std::string cmd = argc > 1 ? argv[1] : "screen";
  std::string corpus;
  std::string snap;
  std::string memeId;
  int maxUnits = 256;
  int takeN = 3;
  bool requireExclusive = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--corpus" && i + 1 < argc)
      corpus = argv[++i];
    else if (a == "--snap" && i + 1 < argc)
      snap = argv[++i];
    else if (a == "--meme" && i + 1 < argc)
      memeId = argv[++i];
    else if (a == "--units" && i + 1 < argc)
      maxUnits = std::atoi(argv[++i]);
    else if (a == "--take" && i + 1 < argc)
      takeN = std::atoi(argv[++i]);
    else if (a == "--require-exclusive")
      requireExclusive = true;
  }
  if (takeN < 1)
    takeN = 3;

  InMemoryStore store;
  int ingested = 0;
  if (!snap.empty() && store.loadText(snap)) {
    ingested = -1;
  } else {
    if (corpus.empty()) {
      std::cerr << "need --corpus or --snap\n";
      return 2;
    }
    ingested = ingestCorpusFile(store, corpus, maxUnits, {});
    if (!snap.empty() && !store.empty())
      store.saveText(snap);
  }
  if (store.empty()) {
    std::cerr << "empty instrument ingested=" << ingested << "\n";
    return 1;
  }
  std::string rawCorpus;
  if (!corpus.empty()) {
    std::ifstream in(corpus, std::ios::binary);
    if (in) {
      std::ostringstream ss;
      ss << in.rdbuf();
      rawCorpus = ss.str();
    }
  }

  if (cmd == "detect") {
    if (memeId.empty()) {
      std::cerr << "detect needs --meme\n";
      return 2;
    }
    printDetect(store, readStdin(), memeId);
    return 0;
  }

  if (cmd == "compose") {
    if (memeId.empty()) {
      std::cerr << "compose needs --meme\n";
      return 2;
    }
    if (corpus.empty()) {
      std::cerr << "compose needs --corpus\n";
      return 2;
    }
    std::ifstream in(corpus, std::ios::binary);
    if (!in) {
      std::cerr << "cannot read corpus\n";
      return 2;
    }
    std::ostringstream raw;
    raw << in.rdbuf();
    const auto rag = phoenix::memetic::composeCarrierRag(store, memeId, raw.str(),
                                                         maxUnits, 1, requireExclusive);
    const auto det = detectMemeInText(store, rag.text, memeId, {});
    const TextStats carrierStats = textStatsOf(rag.text);
    const auto typSet = phoenix::memetic::mappingTypicalAlpha(
        store.weightsOfMeme(memeId));
    const int typicalNeed = static_cast<int>(typSet.size());
    const double redundancy = std::min(typicalNeed, 3) / 3.0;
    const double entropyNorm = std::min(1.0, carrierStats.entropy / 8.0);
    std::cout << "{\"id\": \"";
    jsonEscape(memeId);
    std::cout << "\", \"score\": " << rag.score << ", \"wordHit\": " << rag.wordHit
              << ", \"units\": " << rag.unitIndex.size()
              << ", \"method\": \"";
    jsonEscape(rag.method);
    std::cout << "\", \"bag\": " << rag.bagCos << ", \"semantic\": " << rag.semanticCos
              << ", \"order\": " << rag.orderLift
              << ", \"exclusivity\": " << rag.exclusivity
              << ", \"targetRank\": " << rag.targetRank
              << ", \"otherPresent\": " << rag.otherPresent
              << ", \"typicalCover\": " << rag.typicalCover
              << ", \"typicalNeed\": " << typicalNeed
              << ", \"carrierEntropy\": " << carrierStats.entropy
              << ", \"carrierTop1Frac\": " << carrierStats.top1Frac
              << ", \"carrierMaxRun\": " << carrierStats.maxRun
              << ", \"carrierCollapsed\": " << (carrierStats.collapsed ? "true" : "false")
              << ", \"fieldScore\": " << redundancy * entropyNorm * rag.typicalCover
              << ", \"sourceN\": " << rag.sources.size()
              << ", \"present\": " << (det.present ? "true" : "false")
              << ", \"activation\": " << det.memeScore
              << ", \"lift\": " << det.lift << ", \"rank\": " << det.rank
              << ", \"text\": \"";
    jsonEscape(rag.text);
    std::cout << "\"}\n";
    return rag.text.empty() ? 4 : 0;
  }

  if (cmd == "wipe-ingest") {
    if (memeId.empty()) {
      std::cerr << "wipe-ingest needs --meme\n";
      return 2;
    }
    const std::string text = readStdin();
    const auto frozenWords = store.wordsOfMeme(memeId);
    const auto frozenW = store.weightsOfMeme(memeId);
    const auto ingestDet = detectMemeInText(store, text, memeId, {});
    const auto frozenActs =
        phoenix::memetic::detectActivatedMemes(store, text, {}, 8);
    int targetActRank = ingestDet.rank;
    int otherFrozenPresent = 0;
    for (const auto &a : frozenActs) {
      if (a.present && a.id != memeId)
        ++otherFrozenPresent;
    }
    const bool targetDominant =
        ingestDet.present &&
        (ingestDet.rank == 0 ||
         (ingestDet.peakScore > 1e-12 &&
          ingestDet.memeScore >= 0.5 * ingestDet.peakScore));
    InMemoryStore live;
    const bool emptyBefore = live.empty() && live.nodeCount() == 0;
    const int liveUnits = ingestCorpusText(live, text, maxUnits, {});
    std::unordered_set<std::string> frozenContent;
    for (const auto &w : frozenWords) {
      if (!w.empty() && !mappingMergeStopWord(w))
        frozenContent.insert(w);
    }
    int overlapContent = 0;
    std::string bestId;
    int bestOverlap = -1;
    int bestSize = 1 << 30;
    phoenix::memetic::AlphaIdentity bestAlpha;
    for (const auto &id : live.nodeIds()) {
      int ov = 0;
      const auto ws = live.wordsOfMeme(id);
      for (const auto &w : ws) {
        if (frozenContent.count(w))
          ++ov;
      }
      const auto idn = phoenix::memetic::mappingAlphaIdentity(
          frozenW, live.weightsOfMeme(id));
      const bool betterAlpha =
          idn.typicalHit > bestAlpha.typicalHit ||
          (idn.typicalHit == bestAlpha.typicalHit &&
           idn.massCover > bestAlpha.massCover);
      const bool betterOv =
          ov > bestOverlap ||
          (ov == bestOverlap && ov > 0 &&
           static_cast<int>(ws.size()) < bestSize);
      if (betterAlpha || (idn.typicalHit == bestAlpha.typicalHit && betterOv)) {
        bestAlpha = idn;
        bestOverlap = ov;
        bestSize = static_cast<int>(ws.size());
        bestId = id;
        overlapContent = ov;
      }
    }
    phoenix::memetic::RagCarrier rag;
    phoenix::memetic::MemeDetection reDet{};
    if (!bestId.empty() && (bestAlpha.typicalHit > 0 || overlapContent > 0)) {
      rag = phoenix::memetic::composeCarrierRag(live, bestId, text, maxUnits, 2);
      reDet = detectMemeInText(store, rag.text, memeId, {});
    }
    /* Word-level drift tracking: which typical words the reingested graph
       kept (hit) vs dropped (miss), plus collapse health of the ingested
       text. Feeds the manifold-field analysis of decay rounds. */
    const TextStats inStats = textStatsOf(text);
    const auto typFrozen = phoenix::memetic::mappingTypicalAlpha(frozenW);
    std::vector<std::string> hitWords;
    std::vector<std::string> missWords;
    if (!bestId.empty()) {
      std::unordered_set<std::string> liveW;
      for (const auto &wp : live.weightsOfMeme(bestId)) {
        if (!wp.first.empty() && wp.second > 0.0)
          liveW.insert(wp.first);
      }
      for (const auto &kv : typFrozen) {
        if (liveW.count(kv.first))
          hitWords.push_back(kv.first);
        else
          missWords.push_back(kv.first);
      }
    } else {
      for (const auto &kv : typFrozen)
        missWords.push_back(kv.first);
    }
    std::cout << "{\"emptyBefore\": " << (emptyBefore ? "true" : "false")
              << ", \"liveUnits\": " << liveUnits
              << ", \"liveNodes\": " << live.nodeCount()
              << ", \"frozenContent\": " << frozenContent.size()
              << ", \"overlapContent\": " << overlapContent
              << ", \"typicalNeed\": " << bestAlpha.typicalNeed
              << ", \"typicalHit\": " << bestAlpha.typicalHit
              << ", \"massCover\": " << bestAlpha.massCover
              << ", \"alphaCos\": " << bestAlpha.cosine
              << ", \"alphaSame\": " << (bestAlpha.same ? "true" : "false")
              << ", \"typicalFrozen\": ";
    printWordArray([&] {
      std::vector<std::string> ws;
      ws.reserve(typFrozen.size());
      for (const auto &kv : typFrozen)
        ws.push_back(kv.first);
      std::sort(ws.begin(), ws.end());
      return ws;
    }());
    std::cout << ", \"typicalHitWords\": ";
    printWordArray(hitWords);
    std::cout << ", \"typicalMissWords\": ";
    printWordArray(missWords);
    std::cout << ", \"outEntropy\": " << inStats.entropy
              << ", \"outTop1Frac\": " << inStats.top1Frac
              << ", \"outMaxRun\": " << inStats.maxRun
              << ", \"collapsed\": " << (inStats.collapsed ? "true" : "false")
              << ", \"decayWarn\": "
              << ((liveUnits <= 1 || inStats.collapsed) ? "true" : "false")
              << ", \"targetDominant\": " << (targetDominant ? "true" : "false")
              << ", \"targetActRank\": " << targetActRank
              << ", \"otherFrozenPresent\": " << otherFrozenPresent
              << ", \"bestLiveId\": \"";
    jsonEscape(bestId);
    std::cout << "\", \"ingestPresent\": "
              << (ingestDet.present ? "true" : "false")
              << ", \"ingestLift\": " << ingestDet.lift
              << ", \"ingestRank\": " << ingestDet.rank
              << ", \"recomposePresent\": " << (reDet.present ? "true" : "false")
              << ", \"recomposeLift\": " << reDet.lift
              << ", \"recomposeRank\": " << reDet.rank
              << ", \"recomposeMethod\": \"";
    jsonEscape(rag.method);
    std::cout << "\", \"recomposeText\": \"";
    jsonEscape(rag.text);
    std::cout << "\"}\n";
    return 0;
  }

  const auto g = graphFromStore(store);
  const auto report = analyzeGraph(g);
  if (!report.ok) {
    std::cerr << "analyzeGraph failed " << report.error << "\n";
    return 3;
  }

  struct Row {
    std::string id;
    std::string pole;
    int rankMost{0};
    int rankLeast{0};
    double significance{0};
    double contentSpec{0};
    int contentN{0};
    int typicalNeed{0};
    double carrierEntropy{0.0};
    double typicalCover{0.0};
    double fieldScore{0.0};
    std::vector<std::string> words;
    std::string carrier;
  };
  const int nColl = static_cast<int>(store.nodeCount());
  std::unordered_map<std::string, int> df;
  for (const auto &id : store.nodeIds()) {
    for (const auto &w : store.wordsOfMeme(id)) {
      if (df.count(w))
        continue;
      df[w] = static_cast<int>(store.memesOfWord(w).size());
    }
  }
  std::vector<const phoenix::secamp::NodeInfluence *> phrase;
  std::unordered_map<std::string, double> specOf;
  std::unordered_map<std::string, int> contentNOf;
  std::vector<double> specs;
  std::vector<int> contentNs;
  for (const auto &n : report.nodes) {
    const auto words = store.wordsOfMeme(n.id);
    if (!isPhraseMeme(n.id, words) || isCatalogBag(words) ||
        static_cast<int>(words.size()) > phoenix::memetic::Params{}.ngramMax)
      continue;
    phrase.push_back(&n);
    const auto wts = store.weightsOfMeme(n.id);
    const double spec = phoenix::memetic::mappingContentSpec(wts, df, nColl);
    int contentN = 0;
    for (const auto &w : words) {
      if (!w.empty() && !phoenix::memetic::mappingMergeStopWord(w) &&
          w.size() >= 4)
        ++contentN;
    }
    specOf[n.id] = spec;
    contentNOf[n.id] = contentN;
    specs.push_back(spec);
    contentNs.push_back(contentN);
  }
  /* Typical set = contentSpec at/above median (drop stop bags) and
     contentN strictly above median (drop 3-gram rare-name bags when
     the collection median is 2). Not a present gate. */
  double specMed = 0.0;
  int nMed = 0;
  if (!specs.empty()) {
    std::sort(specs.begin(), specs.end());
    specMed = specs[specs.size() / 2];
  }
  if (!contentNs.empty()) {
    std::sort(contentNs.begin(), contentNs.end());
    nMed = contentNs[contentNs.size() / 2];
  }
  const int need = std::max(4, 2 * takeN);
  auto fillEligible =
      [&](bool strictN) {
        std::vector<const phoenix::secamp::NodeInfluence *> out;
        for (const auto *n : phrase) {
          const double spec = specOf[n->id];
          const int cn = contentNOf[n->id];
          if (spec + 1e-15 < specMed)
            continue;
          if (strictN ? (cn <= nMed) : (cn < nMed))
            continue;
          out.push_back(n);
        }
        return out;
      };
  std::vector<const phoenix::secamp::NodeInfluence *> eligible =
      fillEligible(true);
  if (static_cast<int>(eligible.size()) < need)
    eligible = fillEligible(false);
  if (eligible.empty())
    eligible = phrase;
  std::sort(eligible.begin(), eligible.end(),
            [](const phoenix::secamp::NodeInfluence *a,
               const phoenix::secamp::NodeInfluence *b) {
              return a->significance > b->significance;
            });

  std::vector<Row> rows;
  std::unordered_set<std::string> seen;
  auto add = [&](const phoenix::secamp::NodeInfluence *n, const char *pole) {
    if (!n || !seen.insert(n->id).second)
      return;
    Row row;
    row.id = n->id;
    row.pole = pole;
    row.words = store.wordsOfMeme(n->id);
    row.carrier = fullCarrier(store, n->id);
    row.rankMost = n->rankMost;
    row.rankLeast = n->rankLeast;
    row.significance = n->significance;
    auto sit = specOf.find(n->id);
    row.contentSpec = sit == specOf.end() ? 0.0 : sit->second;
    auto nit = contentNOf.find(n->id);
    row.contentN = nit == contentNOf.end() ? 0 : nit->second;
    rows.push_back(std::move(row));
  };
  /* Prefer memes that already hold an exclusive carrier sentence in the
     corpus (target rank-0 or >= half of present-meme mass). Fall back to
     significance poles only when no exclusive meme exists. Cache the probe
     carriers: their entropy and typical-cover feed the field score. */
  std::vector<const phoenix::secamp::NodeInfluence *> exclusive;
  std::unordered_map<std::string, double> carrierEntropyOf;
  std::unordered_map<std::string, double> typicalCoverOf;
  for (const auto *n : eligible) {
    const auto rag = phoenix::memetic::composeCarrierRag(store, n->id, rawCorpus,
                                                         maxUnits, 1, true);
    if (!rag.text.empty()) {
      carrierEntropyOf[n->id] = textStatsOf(rag.text).entropy;
      typicalCoverOf[n->id] = rag.typicalCover;
      if (rag.targetRank == 0 || rag.exclusivity >= 0.5)
        exclusive.push_back(n);
    }
  }
  const auto &pool = exclusive.empty() ? eligible : exclusive;
  const int take = std::min(takeN, static_cast<int>(pool.size()) / 2);
  for (int i = 0; i < take; ++i)
    add(pool[static_cast<size_t>(i)], "most");
  for (int i = 0; i < take; ++i)
    add(pool[pool.size() - 1 - static_cast<size_t>(i)], "least");
  /* Manifold-field prior per screened meme: redundancy (typical-set size,
     degenerate singletons are saddles) x carrier entropy norm (verbatim
     repeaters sit next to the collapse basin) x typical cover. Verified
     against the dialogue serial: spearman(S, survival rounds) = 0.90. */
  for (auto &row : rows) {
    const auto typ = phoenix::memetic::mappingTypicalAlpha(
        store.weightsOfMeme(row.id));
    row.typicalNeed = static_cast<int>(typ.size());
    auto eit = carrierEntropyOf.find(row.id);
    row.carrierEntropy =
        eit == carrierEntropyOf.end() ? 0.0 : eit->second;
    auto cit = typicalCoverOf.find(row.id);
    row.typicalCover = cit == typicalCoverOf.end() ? 0.0 : cit->second;
    const double redundancy = std::min(row.typicalNeed, 3) / 3.0;
    const double entropyNorm = std::min(1.0, row.carrierEntropy / 8.0);
    row.fieldScore = redundancy * entropyNorm * row.typicalCover;
  }

  std::cout << "{\n  \"method\": \"analyzeGraph resolvent energy; phrase memes; content-typical\",\n"
            << "  \"source\": \"addons/ThePlugInForSecurity/readme.md 统计+识别\",\n"
            << "  \"ingestedUnits\": " << ingested
            << ",\n  \"nodes\": " << store.nodeCount()
            << ",\n  \"analyzed\": " << report.nodeCount
            << ",\n  \"phraseMemes\": " << phrase.size()
            << ",\n  \"eligibleMemes\": " << eligible.size()
            << ",\n  \"exclusiveMemes\": " << exclusive.size()
            << ",\n  \"contentSpecMedian\": " << specMed
            << ",\n  \"contentNMedian\": " << nMed
            << ",\n  \"clipped\": false"
            << ",\n  \"memes\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto self = detectMemeInText(store, rows[i].carrier, rows[i].id, {});
    std::cout << "    {\"id\": \"" << rows[i].id << "\", \"pole\": \""
              << rows[i].pole << "\", \"rankMost\": " << rows[i].rankMost
              << ", \"rankLeast\": " << rows[i].rankLeast
              << ", \"significance\": " << rows[i].significance
              << ", \"contentSpec\": " << rows[i].contentSpec
              << ", \"contentN\": " << rows[i].contentN
              << ", \"typicalNeed\": " << rows[i].typicalNeed
              << ", \"carrierEntropy\": " << rows[i].carrierEntropy
              << ", \"typicalCover\": " << rows[i].typicalCover
              << ", \"fieldScore\": " << rows[i].fieldScore
              << ", \"wordCount\": " << rows[i].words.size()
              << ", \"selfPresent\": " << (self.present ? "true" : "false")
              << ", \"selfActivation\": " << self.memeScore
              << ", \"selfPeak\": " << self.peakScore
              << ", \"selfRank\": " << self.rank
              << ", \"carrier\": \"";
    jsonEscape(rows[i].carrier);
    std::cout << "\"}";
    if (i + 1 < rows.size())
      std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "  ]\n}\n";
  return 0;
}
