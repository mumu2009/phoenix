/* Official readme 统计/识别 on a frozen instrument. Not llama. No writeback. */
#include "memetic_existence.hpp"
/* security_core.hpp 在 PHOENIX_INSTRUMENT_LEAN 下也是安全的（重依赖
   已在头文件内裁掉）；精益构建只裁掉本文件的 defense-test 子命令。 */
#include "addons/ThePlugInForSecurity/security_core.hpp"

#include <algorithm>
#include <chrono>
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
  std::string poolMode;
  std::string targetsFile;
  std::string memesFile;
  int maxUnits = 256;
  int takeN = 3;
  int skipUnits = 0;
  bool requireExclusive = false;
  bool relaxFilter = false;
  double defenseQuantile = 0.85;
  std::string quantilesArg;
  std::string negativesFile;
  bool flagAll = false;
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
    else if (a == "--pool" && i + 1 < argc)
      poolMode = argv[++i];
    else if (a == "--skip" && i + 1 < argc)
      skipUnits = std::atoi(argv[++i]);
    else if (a == "--relax")
      relaxFilter = true;
    else if (a == "--require-exclusive")
      requireExclusive = true;
    else if (a == "--targets" && i + 1 < argc)
      targetsFile = argv[++i];
    else if (a == "--memes-file" && i + 1 < argc)
      memesFile = argv[++i];
    else if (a == "--quantile" && i + 1 < argc)
      defenseQuantile = std::atof(argv[++i]);
    else if (a == "--quantiles" && i + 1 < argc)
      quantilesArg = argv[++i];
    else if (a == "--negatives" && i + 1 < argc)
      negativesFile = argv[++i];
    else if (a == "--flag-all")
      flagAll = true;
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
    if (skipUnits > 0) {
      /* Second-instrument slicing: skip the first K corpus units so a new
         frozen snap can be built from a disjoint slice of the same corpus
         (the 761-node reference snap holds units 0..255 of wikitext-103). */
      std::ifstream in(corpus, std::ios::binary);
      if (!in) {
        std::cerr << "cannot read corpus\n";
        return 2;
      }
      std::ostringstream raw;
      raw << in.rdbuf();
      const auto units =
          phoenix::memetic::splitCorpusUnits(raw.str());
      std::string sliced;
      for (size_t i = static_cast<size_t>(skipUnits); i < units.size(); ++i) {
        sliced += units[i];
        sliced += "\n\n";
      }
      ingested = phoenix::memetic::ingestCorpusText(store, sliced, maxUnits,
                                                    {});
    } else {
      ingested = ingestCorpusFile(store, corpus, maxUnits, {});
    }
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
    if (skipUnits > 0 && !rawCorpus.empty()) {
      /* 片区间化（2026-09-30 性能）：screen/compose 的载体搜索只扫本片
         [skip, skip+units) 而非全语料。模因由该片 ingest 而来，最佳载
         体几乎必然在片内；全库扫描是 O(pool x 全库) 的浪费（dialogue
         片实测 screen 筛选 >20 分钟）。ingest 行为不变（上方建图分支
         内部本就 cap maxUnits）。 */
      const auto units = phoenix::memetic::splitCorpusUnits(rawCorpus);
      std::string sliced;
      const size_t end =
          std::min(units.size(),
                   static_cast<size_t>(skipUnits) +
                       static_cast<size_t>(std::max(1, maxUnits)));
      for (size_t i = static_cast<size_t>(skipUnits); i < end; ++i) {
        sliced += units[i];
        sliced += "\n\n";
      }
      if (!sliced.empty())
        rawCorpus = std::move(sliced);
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
    /* 用 main 的 rawCorpus（--skip>0 时已片区间化），不再自读全文 */
    const auto rag = phoenix::memetic::composeCarrierRag(store, memeId, rawCorpus,
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

  /* compose-batch：一次进程对整片候选批量 compose（2026-09-30）。
     倒排缓存同进程生效：N 个模因 = 1 次建索引 + N 次命中查询，
     对比逐次 CLI（每次冷启动全扫）提速约 N 倍。
     输入 --memes-file：每行一个 meme id。输出：
     {"results": {"<id>": {<compose 同 schema>, "composeExclusive": bool}}} */
  if (cmd == "compose-batch") {
    if (memesFile.empty()) {
      std::cerr << "compose-batch needs --memes-file\n";
      return 2;
    }
    if (corpus.empty()) {
      std::cerr << "compose-batch needs --corpus\n";
      return 2;
    }
    std::ifstream min(memesFile);
    if (!min) {
      std::cerr << "cannot read memes-file\n";
      return 2;
    }
    std::vector<std::string> ids;
    std::string line;
    while (std::getline(min, line)) {
      const auto b = line.find_first_not_of(" \t\r\n");
      if (b == std::string::npos)
        continue;
      const auto e = line.find_last_not_of(" \t\r\n");
      ids.push_back(line.substr(b, e - b + 1));
    }
    std::cout << "{\"results\": {\n";
    bool first = true;
    for (const auto &mid : ids) {
      auto rag = phoenix::memetic::composeCarrierRag(store, mid, rawCorpus,
                                                     maxUnits, 1, true);
      bool excl = true;
      if (rag.text.empty()) {
        rag = phoenix::memetic::composeCarrierRag(store, mid, rawCorpus,
                                                  maxUnits, 1, false);
        excl = false;
      }
      if (rag.text.empty())
        continue; /* 无载体：不出现在 results（调用方视为 no-carrier） */
      const auto det = detectMemeInText(store, rag.text, mid, {});
      const TextStats carrierStats = textStatsOf(rag.text);
      const auto typSet = phoenix::memetic::mappingTypicalAlpha(
          store.weightsOfMeme(mid));
      const int typicalNeed = static_cast<int>(typSet.size());
      const double redundancy = std::min(typicalNeed, 3) / 3.0;
      const double entropyNorm = std::min(1.0, carrierStats.entropy / 8.0);
      if (!first)
        std::cout << ",\n";
      first = false;
      std::cout << "  \"";
      jsonEscape(mid);
      std::cout << "\": {\"id\": \"";
      jsonEscape(mid);
      std::cout << "\", \"score\": " << rag.score
                << ", \"wordHit\": " << rag.wordHit
                << ", \"units\": " << rag.unitIndex.size()
                << ", \"method\": \"";
      jsonEscape(rag.method);
      std::cout << "\", \"bag\": " << rag.bagCos
                << ", \"semantic\": " << rag.semanticCos
                << ", \"order\": " << rag.orderLift
                << ", \"exclusivity\": " << rag.exclusivity
                << ", \"targetRank\": " << rag.targetRank
                << ", \"otherPresent\": " << rag.otherPresent
                << ", \"typicalCover\": " << rag.typicalCover
                << ", \"typicalNeed\": " << typicalNeed
                << ", \"carrierEntropy\": " << carrierStats.entropy
                << ", \"carrierTop1Frac\": " << carrierStats.top1Frac
                << ", \"carrierMaxRun\": " << carrierStats.maxRun
                << ", \"carrierCollapsed\": "
                << (carrierStats.collapsed ? "true" : "false")
                << ", \"fieldScore\": "
                << redundancy * entropyNorm * rag.typicalCover
                << ", \"sourceN\": " << rag.sources.size()
                << ", \"present\": " << (det.present ? "true" : "false")
                << ", \"activation\": " << det.memeScore
                << ", \"lift\": " << det.lift << ", \"rank\": " << det.rank
                << ", \"composeExclusive\": " << (excl ? "true" : "false")
                << ", \"text\": \"";
      jsonEscape(rag.text);
      std::cout << "\"}";
    }
    std::cout << "\n}}\n";
    return 0;
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

#ifndef PHOENIX_INSTRUMENT_LEAN
  if (cmd == "defense-test") {
    /* Defense-layer reality check on the frozen instrument: flag target memes
       anomalous, enable the security plugin, then inspect each target's own
       carrier sentence. Emits where the interception chain breaks per target
       (significance cut / GNN activation+lift gate / flag gate), plus an
       unflagged pass and a clean-text pass as controls. Not a present gate. */
    if (targetsFile.empty()) {
      std::cerr << "defense-test needs --targets <json>\n";
      return 2;
    }
    std::ifstream tin(targetsFile, std::ios::binary);
    if (!tin) {
      std::cerr << "cannot read targets " << targetsFile << "\n";
      return 2;
    }
    std::ostringstream tss;
    tss << tin.rdbuf();
    const json tj = json::parse(tss.str());
    struct Target {
      std::string id;
      std::string carrier;
    };
    std::vector<Target> targets;
    for (const auto &row : tj)
      targets.push_back(Target{row.value("id", ""), row.value("carrier", "")});

    auto &obs = phoenix::secamp::SecurityObservatory::instance();
    /* Quantile list: --quantiles "0.85,0.3,..." overrides the default. */
    std::vector<double> quantiles;
    if (!quantilesArg.empty()) {
      std::stringstream qs(quantilesArg);
      std::string tok;
      while (std::getline(qs, tok, ','))
        quantiles.push_back(std::atof(tok.c_str()));
    } else {
      quantiles = {0.85, 0.5, 0.0};
    }
    /* Negatives: clean sentences that must never be blocked. */
    std::vector<std::string> negatives;
    if (!negativesFile.empty()) {
      std::ifstream nin(negativesFile, std::ios::binary);
      if (nin) {
        std::ostringstream nss;
        nss << nin.rdbuf();
        const json nj = json::parse(nss.str());
        for (const auto &s : nj)
          negatives.push_back(s.get<std::string>());
      }
    }
    negatives.push_back(
        "The committee published its annual report on municipal water "
        "usage .");

    DiscreteGraph dg = graphFromStore(store);
    if (flagAll) {
      /* Stress test: flag every phrase meme, measuring how much of the graph
         each cut actually covers and whether clean text stays safe when the
         isolation policy is maximally aggressive. */
      for (const auto &id : dg.ids) {
        const auto words = store.wordsOfMeme(id);
        if (isPhraseMeme(id, words))
          dg.anomalous.insert(id);
      }
    } else {
      for (const auto &t : targets)
        dg.anomalous.insert(t.id);
    }
    obs.resetForTests();
    const auto rep = obs.ingest(dg);
    obs.setPluginEnabled(true);
    double maxSig = 0.0;
    std::unordered_map<std::string, double> sigOf;
    for (const auto &n : rep.nodes) {
      sigOf[n.id] = n.significance;
      if (n.significance > maxSig)
        maxSig = n.significance;
    }
    json out;
    out["snap"] = snap;
    out["targetCount"] = targets.size();
    out["negativeCount"] = negatives.size();
    out["nodes"] = dg.ids.size();
    out["flaggedNodes"] = dg.anomalous.size();
    out["flagAll"] = flagAll;
    out["maxSig"] = maxSig;
    json sweeps = json::array();
    for (double q : quantiles) {
      obs.setHighImpactQuantile(q);
      const double cut = maxSig * q;
      json sweep;
      sweep["quantile"] = q;
      sweep["cut"] = cut;
      int nodesAboveCut = 0;
      int flaggedAboveCut = 0;
      for (const auto &n : rep.nodes) {
        if (n.significance >= cut) {
          ++nodesAboveCut;
          if (dg.anomalous.count(n.id))
            ++flaggedAboveCut;
        }
      }
      sweep["nodesAboveCut"] = nodesAboveCut;
      sweep["flaggedAboveCut"] = flaggedAboveCut;
      json rows = json::array();
      int blockedTargets = 0;
      for (const auto &t : targets) {
        const auto dec = obs.inspectText(t.carrier);
        const auto det = detectMemeInText(store, t.carrier, t.id, {});
        if (dec.blocked)
          ++blockedTargets;
        json row;
        row["id"] = t.id;
        row["sig"] = sigOf.count(t.id) ? sigOf[t.id] : 0.0;
        row["passCut"] = sigOf.count(t.id) != 0 && sigOf[t.id] >= cut;
        row["detectPresent"] = det.present;
        row["detectLift"] = det.lift;
        row["blocked"] = dec.blocked;
        row["reason"] = dec.reason;
        rows.push_back(std::move(row));
      }
      sweep["blockedTargets"] = blockedTargets;
      sweep["results"] = std::move(rows);
      int blockedNeg = 0;
      for (const auto &s : negatives)
        if (obs.inspectText(s).blocked)
          ++blockedNeg;
      sweep["blockedNegatives"] = blockedNeg;
      sweeps.push_back(std::move(sweep));
    }
    out["sweeps"] = std::move(sweeps);
    /* Unflagged control at the most permissive cut: nothing may block. */
    {
      obs.resetForTests();
      DiscreteGraph plain = graphFromStore(store);
      obs.ingest(plain);
      obs.setPluginEnabled(true);
      obs.setHighImpactQuantile(0.0);
      int blockedN = 0;
      for (const auto &t : targets)
        if (obs.inspectText(t.carrier).blocked)
          ++blockedN;
      out["unflaggedControl"] = {{"blockedTargets", blockedN},
                                 {"targetCount", targets.size()}};
    }
    std::cout << out.dump(2) << "\n";
    return 0;
  }
#endif /* PHOENIX_INSTRUMENT_LEAN */

  fprintf(stderr, "[screen-profile] store loaded, nodes=%d\n",
          (int)store.nodeCount());
  const auto tProf0 = std::chrono::steady_clock::now();
  const auto g = graphFromStore(store);
  const auto report = analyzeGraph(g);
  fprintf(stderr, "[screen-profile] analyzeGraph %.1fs\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        tProf0).count());
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
    /* 图连接特征（2026-09-30 综合约束体系）：dfMax=词袋最大 df（枢纽搭接，
       单特征 AUC 0.661 最强）；deg=模因间边数；top1Frac=最大 α 权重占比。
       combo_v2 = z(wordCount)-z(carrierEntropy)+z(log1p(dfMax)) AUC 0.693。 */
    int dfMax{0};
    int deg{0};
    double top1Frac{0.0};
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
  if (relaxFilter) {
    /* Batch hunting beyond the median-filtered pool: keep a mild quality
       floor (half the content-spec median, at least one content word) so
       pure stop-bags stay out, but admit the rest of the phrase memes.
       The field prior and the serial posterior do the real ranking. */
    std::vector<const phoenix::secamp::NodeInfluence *> wide;
    for (const auto *n : phrase) {
      const double spec = specOf[n->id];
      const int cn = contentNOf[n->id];
      if (spec + 1e-15 < 0.5 * specMed || cn < 1)
        continue;
      wide.push_back(n);
    }
    if (!wide.empty())
      eligible = std::move(wide);
  }
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
    /* 图连接特征：dfMax / deg / top1Frac（store 现成接口，O(词数)） */
    row.deg = static_cast<int>(store.neighborsOf(n->id).size());
    {
      const auto walpha = store.weightsOfMeme(n->id);
      double sum = 0.0, mx = 0.0;
      for (const auto &wa : walpha) {
        sum += wa.second;
        if (wa.second > mx)
          mx = wa.second;
      }
      row.top1Frac = sum > 0.0 ? mx / sum : 0.0;
      int dfm = 0;
      for (const auto &w : row.words) {
        const int d = static_cast<int>(store.memesOfWord(w).size());
        if (d > dfm)
          dfm = d;
      }
      row.dfMax = dfm;
    }
    rows.push_back(std::move(row));
  };
  /* Prefer memes that already hold an exclusive carrier sentence in the
     corpus (target rank-0 or >= half of present-meme mass). Fall back to
     significance poles only when no exclusive meme exists. Cache the probe
     carriers: their entropy and typical-cover feed the field score.

     性能（2026-09-30）：composeCarrierRag 每次调用扫整个 rawCorpus，
     对全 eligible 池（数千模因 × 数百 MB 语料）是 O(pool x corpus)。
     --pool eligible 模式下 exclusive 列表根本不被使用，fieldScore 也
     只查最终选中行——故 eligible 模式只对选中行 compose（O(take)），
     默认模式保持全池扫描（需要 exclusive 全集）。行为等价。 */
  std::vector<const phoenix::secamp::NodeInfluence *> exclusive;
  std::unordered_map<std::string, double> carrierEntropyOf;
  std::unordered_map<std::string, double> typicalCoverOf;
  auto probeCarrier = [&](const phoenix::secamp::NodeInfluence *n) {
    auto rag = phoenix::memetic::composeCarrierRag(store, n->id, rawCorpus,
                                                   maxUnits, 1, true);
    if (rag.text.empty()) {
      /* Prior-only probe: a non-exclusive carrier still feeds the field
         score (entropy x typical cover); the exclusive pool test below
         keeps using the rank/exclusivity thresholds either way. */
      rag = phoenix::memetic::composeCarrierRag(store, n->id, rawCorpus,
                                                maxUnits, 1, false);
    }
    if (!rag.text.empty()) {
      carrierEntropyOf[n->id] = textStatsOf(rag.text).entropy;
      typicalCoverOf[n->id] = rag.typicalCover;
      return rag.targetRank == 0 || rag.exclusivity >= 0.5;
    }
    return false;
  };
  /* --pool eligible screens the whole typical-set-filtered pool (the
     exclusive pool is small and may already be exhausted by earlier
     batches); default keeps the exclusive preference. */
  if (poolMode != "eligible") {
    for (const auto *n : eligible) {
      if (probeCarrier(n))
        exclusive.push_back(n);
    }
  }
  const auto &pool = (poolMode == "eligible")
                         ? eligible
                         : (exclusive.empty() ? eligible : exclusive);
  const int take = std::min(takeN, static_cast<int>(pool.size()) / 2);
  /* eligible 模式完全不 compose（2026-09-30）：carrierEntropy/
     typicalCover/fieldScore 由 exporter 的 compose-batch 实测补填。
     screen 只留图内快字段，筛选从 O(pool x 语料) 降为 O(图)。 */
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

  fprintf(stderr, "[screen-profile] rows built=%zu, start output\n",
          rows.size());
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
              << ", \"dfMax\": " << rows[i].dfMax
              << ", \"deg\": " << rows[i].deg
              << ", \"top1Frac\": " << rows[i].top1Frac
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
