/* Host soak of the official RuntimeState GNN path.
   ingestDocument -> buildMemeSequence + consecutive edges
   mapWordsToMemes query-time n-gram resolveExisting (lookup only)
   detect = GNN query-conditioned α + lift + content evidence
   Does not call llama. Does not start phoenix_main. */
#include "memetic_existence.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using phoenix::memetic::ExistenceResult;
using phoenix::memetic::InMemoryStore;
using phoenix::memetic::Params;
using phoenix::memetic::carrierOfMeme;
using phoenix::memetic::composeCarrierRag;
using phoenix::memetic::detectMemeInText;
using phoenix::memetic::ingestCorpusText;
using phoenix::memetic::ingestDocumentLike;
using phoenix::memetic::kGnnPresentLift;
using phoenix::memetic::mappingMergeAllowed;
using phoenix::memetic::mappingQueryMergeAllowed;
using phoenix::memetic::mappingMergeStopWord;
using phoenix::memetic::mappingQueryHasContentForMeme;
using phoenix::memetic::runHarmlessExistence;
using phoenix::memetic::screenMemesFromBarrier;
using phoenix::memetic::tokenizeAscii;

static std::string slurp(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return {};
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

static bool officialRecallOk(const ExistenceResult &r, std::string *why) {
  if (r.queryMentionsNonce) {
    *why = "cue mentions nonce";
    return false;
  }
  if (!(r.kvmReconstructedNonce || r.diffusionRankedNonceMeme)) {
    *why = "cue neither kvm nor diffusion recalled nonce";
    return false;
  }
  if (r.distractorReconstructedNonce || r.distractorDiffusionRankedNonce) {
    *why = "distractor saw nonce";
    return false;
  }
  return true;
}

static bool twoTopicNoGlue(std::string *why) {
  InMemoryStore store;
  const std::string corpus =
      "The south river survey marker stands beside the tidal marsh. "
      "The bakery inventory shelf count lists flour bags and yeast jars.";
  if (ingestCorpusText(store, corpus, 8) < 2) {
    *why = "two-topic ingest produced fewer than 2 units";
    return false;
  }
  for (const auto &id : store.nodeIds()) {
    const auto words = store.wordsOfMeme(id);
    bool river = false;
    bool bakery = false;
    for (const auto &w : words) {
      if (w == "river" || w == "marsh")
        river = true;
      if (w == "bakery" || w == "yeast")
        bakery = true;
    }
    if (river && bakery) {
      *why = "river/marsh glued to bakery/yeast on " + id;
      return false;
    }
  }
  return true;
}

/* Same merge gate as RuntimeState::mapWordsToMemesOrdered resolveOrCreate. */
static std::string queryResolveOrCreate(InMemoryStore &store,
                                        const std::vector<std::string> &tokenSet,
                                        const Params &p) {
  std::vector<std::string> uniq;
  std::unordered_set<std::string> seen;
  for (const auto &t : tokenSet) {
    if (t.empty())
      continue;
    if (seen.insert(t).second)
      uniq.push_back(t);
  }
  if (uniq.size() <= 1)
    return uniq.empty() ? "" : ("meme_" + uniq[0]);
  std::unordered_map<std::string, int> counts;
  for (const auto &w : uniq) {
    for (const auto &mid : store.memesOfWord(w))
      counts[mid]++;
  }
  std::string best;
  int bestOverlap = 0;
  for (const auto &kv : counts) {
    if (kv.second > bestOverlap) {
      bestOverlap = kv.second;
      best = kv.first;
    }
  }
  const int minOverlap = std::max(1, p.minOverlap);
  const int maxWordSet = std::max(4, p.maxMemeWords);
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
        mappingQueryMergeAllowed(bestOverlap, static_cast<int>(uniq.size()),
                                 static_cast<int>(existing.size()), minOverlap,
                                 incomingContent, overlapContent)) {
      for (const auto &w : uniq)
        store.bind(w, best);
      store.ensureNode(best);
      return best;
    }
  }
  /* Query face is lookup-only: do not mint meme_p_q_* during resolve. */
  return "";
}

static bool queryTimeMergeNoGlue(std::string *why) {
  InMemoryStore store;
  ingestDocumentLike(store, {"south", "river", "survey", "marker", "marsh"}, {});
  ingestDocumentLike(store, {"bakery", "flour", "yeast", "inventory", "shelf"},
                     {});
  const auto priorIds = store.nodeIds();
  std::unordered_set<std::string> prior(priorIds.begin(), priorIds.end());
  const auto mixed = tokenizeAscii(
      "the south river survey marker and the bakery flour yeast inventory");
  Params p;
  const int nMin = std::max(2, p.ngramMin);
  const int nMax = std::max(nMin, p.ngramMax);
  for (size_t i = 0; i < mixed.size(); ++i) {
    for (int n = nMin; n <= nMax; ++n) {
      if (i + static_cast<size_t>(n) > mixed.size())
        break;
      queryResolveOrCreate(
          store,
          std::vector<std::string>(mixed.begin() + static_cast<std::ptrdiff_t>(i),
                                   mixed.begin() + static_cast<std::ptrdiff_t>(
                                                       i + static_cast<size_t>(n))),
          p);
    }
  }
  for (const auto &id : prior) {
    const auto words = store.wordsOfMeme(id);
    bool river = false;
    bool bakery = false;
    for (const auto &w : words) {
      if (w == "river" || w == "marsh")
        river = true;
      if (w == "bakery" || w == "yeast")
        bakery = true;
    }
    if (river && bakery) {
      std::string bag;
      for (const auto &w : words) {
        if (!bag.empty())
          bag += " ";
        bag += w;
      }
      *why = "query-time merge expanded a prior meme across topics id=" + id +
             " words=" + bag;
      return false;
    }
  }
  return true;
}

static bool seedBudgetOk(std::string *why) {
  if (phoenix::memetic::mappingIdfScale(1, 761) != 1.0) {
    *why = "idf unique word is not 1";
    return false;
  }
  if (phoenix::memetic::mappingIdfScale(400, 761) >= 0.25) {
    *why = "idf hub word not shrunk";
    return false;
  }
  using phoenix::memetic::MappingWordHit;
  std::unordered_map<std::string, double> q{{"used", 1.0}, {"sakimoto", 1.0}};
  std::unordered_map<std::string, std::vector<MappingWordHit>> hits;
  hits["used"] = {{"meme_p_a", 0.1}, {"meme_p_b", 0.1}, {"meme_p_c", 0.1}};
  hits["sakimoto"] = {{"meme_p_a", 1.0}};
  std::unordered_map<std::string, double> seeds;
  phoenix::memetic::mappingAddConditionedSeeds(q, hits, &seeds, nullptr, 761);
  if (!(seeds["meme_p_a"] > seeds["meme_p_b"] &&
        seeds["meme_p_a"] > seeds["meme_p_c"])) {
    *why = "conditioned seed lost named-meme pull after idf";
    return false;
  }
  std::unordered_map<std::string, double> spray;
  spray["meme_peak"] = 1.9;
  for (int i = 0; i < 200; ++i)
    spray["meme_hub_" + std::to_string(i)] = 0.002;
  phoenix::memetic::mappingConcentrateSeeds(&spray);
  if (!spray.count("meme_peak") || spray.size() >= 8) {
    *why = "concentrate did not drop thin hub tail";
    return false;
  }
  std::unordered_map<std::string, int> df{{"the", 400}, {"was", 200},
                                         {"sakimoto", 2}, {"valkyria", 3}};
  const auto stops = phoenix::memetic::mappingMemeWeights(
      {"the", "was"}, {1, 1}, df);
  const auto names = phoenix::memetic::mappingMemeWeights(
      {"sakimoto", "valkyria"}, {1, 1}, df);
  if (phoenix::memetic::mappingContentSpec(names, df, 761) <=
      phoenix::memetic::mappingContentSpec(stops, df, 761)) {
    *why = "content spec lost rare-name preference";
    return false;
  }
  std::unordered_map<std::string, int> dfId{
      {"the", 400},      {"opening", 80}, {"theme", 90},
      {"faylan", 2},     {"tomoshibi", 2}, {"akari", 2},
      {"kalafina", 3},   {"singer", 40}};
  const auto frozen = phoenix::memetic::mappingMemeWeights(
      {"opening", "theme", "akari", "tomoshibi", "faylan", "the"},
      {1, 1, 1, 1, 1, 1}, dfId);
  const auto sameNames = phoenix::memetic::mappingMemeWeights(
      {"faylan", "sang", "akari", "tomoshibi"}, {1, 1, 1, 1}, dfId);
  const auto genreOnly = phoenix::memetic::mappingMemeWeights(
      {"opening", "theme", "the", "singer"}, {1, 1, 1, 1}, dfId);
  if (!phoenix::memetic::mappingAlphaIdentity(frozen, sameNames).same) {
    *why = "alpha identity missed same typical names";
    return false;
  }
  if (phoenix::memetic::mappingAlphaIdentity(frozen, genreOnly).same) {
    *why = "alpha identity accepted genre-only leftover";
    return false;
  }
  auto memesOf = [](const std::string &w) -> std::vector<std::string> {
    if (w == "naval" || w == "duties")
      return {"meme_p_a", "meme_p_b"};
    if (w == "dunnington")
      return {"meme_p_a"};
    if (w == "ponchartrain")
      return {"meme_p_b"};
    return {};
  };
  auto wordsOf = [](const std::string &id) -> std::vector<std::string> {
    if (id == "meme_p_a")
      return {"dunnington", "naval", "duties"};
    if (id == "meme_p_b")
      return {"ponchartrain", "naval", "duties"};
    return {};
  };
  const auto multi = phoenix::memetic::mappingCollectExistingMemes(
      {"naval", "duties"}, memesOf, wordsOf, 2, 20, true);
  if (multi.size() < 2) {
    *why = "n-gram lookup kept only one existing meme";
    return false;
  }
  return true;
}

static bool stopHubOk(std::string *why) {
  InMemoryStore store;
  const char *ids[] = {"meme_p_hub", "meme_p_b", "meme_p_c",
                       "meme_p_d", "meme_p_e", "meme_p_f"};
  const char *uniq[] = {"river", "bakery", "quarry",
                        "orchard", "harbor", "depot"};
  for (int i = 0; i < 6; ++i) {
    store.bind("the", ids[i]);
    store.bind("of", ids[i]);
    store.bind("and", ids[i]);
    store.bind(uniq[i], ids[i]);
  }
  store.bind("from", ids[0]);
  store.bind("with", ids[0]);
  const auto own = detectMemeInText(store, "river marker survey", ids[0]);
  if (!own.present || own.lift < kGnnPresentLift) {
    *why = "content query missed hub";
    return false;
  }
  const auto stops = detectMemeInText(
      store,
      "the of and to a in on for is was with as by at from that this it be or are",
      ids[0]);
  if (stops.present) {
    *why = "stop query presented hub";
    return false;
  }
  if (mappingQueryHasContentForMeme({"the", "from", "with"},
                                    {"the", "from", "with", "river"})) {
    *why = "stops counted as content evidence";
    return false;
  }
  return true;
}

static bool corpusIdentifyOk(const std::string &corpus, int units,
                             std::string *why, int *nodes, int *kept) {
  InMemoryStore store;
  const int n = ingestCorpusText(store, corpus, units);
  if (n < 2) {
    *why = "corpus ingest < 2";
    return false;
  }
  *nodes = static_cast<int>(store.nodeCount());
  const auto screened = screenMemesFromBarrier(store, 6);
  *kept = static_cast<int>(screened.size());
  if (screened.empty()) {
    *why = "screen empty after corpus ingest";
    return false;
  }
  const std::string stops =
      "the of and to a in on for is was with as by at from that this it be or are";
  const std::string unrelated =
      "quantum chess tournament pairing sheet omega7 bakery inventory flour yeast";
  int checked = 0;
  for (const auto &row : screened) {
    if (static_cast<int>(row.words.size()) > Params{}.ngramMax)
      continue;
    const auto own = detectMemeInText(store, row.carrier, row.id);
    const auto st = detectMemeInText(store, stops, row.id);
    const auto un = detectMemeInText(store, unrelated, row.id);
    ++checked;
    if (!own.present || st.present || un.present) {
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    "identify fail id=%s own=%d stop=%d un=%d lift=%.3f",
                    row.id.c_str(), own.present ? 1 : 0, st.present ? 1 : 0,
                    un.present ? 1 : 0, own.lift);
      *why = buf;
      return false;
    }
  }
  if (checked < 1) {
    *why = "no phrase meme within ngramMax after screen";
    return false;
  }

  std::string composeId;
  for (const auto &row : screened) {
    for (const auto &w : row.words) {
      if (!mappingMergeStopWord(w) && w.size() >= 4) {
        composeId = row.id;
        break;
      }
    }
    if (!composeId.empty())
      break;
  }
  if (composeId.empty())
    composeId = screened.front().id;
  const auto rag = composeCarrierRag(store, composeId, corpus, units, 1);
  if (rag.text.empty() || rag.method.find("tensor") == std::string::npos) {
    *why = "compose empty or not tensor-sentence";
    return false;
  }
  if (rag.text == carrierOfMeme(store, composeId)) {
    *why = "compose returned the word bag";
    return false;
  }
  if (!detectMemeInText(store, rag.text, composeId).present) {
    *why = "composed sentence does not present meme";
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  std::string corpusPath;
  double hours = 1.0;
  int units = 48;
  int sleepSec = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--hours" && i + 1 < argc)
      hours = std::atof(argv[++i]);
    else if (a == "--corpus" && i + 1 < argc)
      corpusPath = argv[++i];
    else if (a == "--units" && i + 1 < argc)
      units = std::atoi(argv[++i]);
    else if (a == "--sleep" && i + 1 < argc)
      sleepSec = std::atoi(argv[++i]);
    else if (a == "--once")
      hours = 0.0;
  }
  if (corpusPath.empty()) {
    std::fprintf(stderr, "usage: host_main_gnn_flow --corpus FILE [--hours 1]\n");
    return 2;
  }
  const std::string corpus = slurp(corpusPath);
  if (corpus.size() < 80) {
    std::fprintf(stderr, "corpus too small: %s\n", corpusPath.c_str());
    return 2;
  }

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(static_cast<int>(hours * 3600000.0));
  const auto started = std::chrono::steady_clock::now();
  int cycle = 0;
  int fails = 0;
  std::printf("main-flow soak start hours=%.2f units=%d corpus=%s\n", hours,
              units, corpusPath.c_str());
  std::fflush(stdout);

  auto runOnce = [&]() -> bool {
    std::string why;
    std::printf("stage official\n");
    std::fflush(stdout);
    const auto official = runHarmlessExistence();
    if (!officialRecallOk(official, &why)) {
      std::printf("FAIL official-recall %s kvm=%d diff=%d hops=%d\n", why.c_str(),
                  official.kvmReconstructedNonce ? 1 : 0,
                  official.diffusionRankedNonceMeme ? 1 : 0,
                  official.hopsCueToNonce);
      return false;
    }
    std::printf("stage high-overlap\n");
    std::fflush(stdout);
    Params tight;
    tight.minOverlap = 99;
    const auto high = runHarmlessExistence(tight);
    if (high.kvmReconstructedNonce) {
      std::printf("FAIL high-overlap kvm reconstructed nonce without merge\n");
      return false;
    }
    if (!high.diffusionRankedNonceMeme) {
      std::printf("FAIL high-overlap diffusion missed nonce\n");
      return false;
    }
    std::printf("stage two-topic\n");
    std::fflush(stdout);
    if (!twoTopicNoGlue(&why)) {
      std::printf("FAIL two-topic %s\n", why.c_str());
      return false;
    }
    std::printf("stage query-merge\n");
    std::fflush(stdout);
    if (!queryTimeMergeNoGlue(&why)) {
      std::printf("FAIL query-merge %s\n", why.c_str());
      return false;
    }
    std::printf("stage seed-budget\n");
    std::fflush(stdout);
    if (!seedBudgetOk(&why)) {
      std::printf("FAIL seed-budget %s\n", why.c_str());
      return false;
    }
    std::printf("stage stop-hub\n");
    std::fflush(stdout);
    if (!stopHubOk(&why)) {
      std::printf("FAIL stop-hub %s\n", why.c_str());
      return false;
    }
    std::printf("stage corpus-identify\n");
    std::fflush(stdout);
    int nodes = 0;
    int kept = 0;
    if (!corpusIdentifyOk(corpus, units, &why, &nodes, &kept)) {
      std::printf("FAIL corpus-identify %s nodes=%d kept=%d\n", why.c_str(),
                  nodes, kept);
      return false;
    }
    std::printf("ok official kvm=%d diff=%d hops=%d nodes=%d screened=%d\n",
                official.kvmReconstructedNonce ? 1 : 0,
                official.diffusionRankedNonceMeme ? 1 : 0,
                official.hopsCueToNonce, nodes, kept);
    return true;
  };

  do {
    ++cycle;
    const bool ok = runOnce();
    const auto elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count() /
                         60.0;
    if (!ok) {
      ++fails;
      std::printf("cycle=%d ok=0 fails=%d elapsedMin=%.1f STOP\n", cycle, fails,
                  elapsed);
      std::fflush(stdout);
      return 3;
    }
    std::printf("cycle=%d ok=1 fails=0 elapsedMin=%.1f\n", cycle, elapsed);
    std::fflush(stdout);
    if (hours <= 0.0)
      break;
    if (std::chrono::steady_clock::now() >= deadline)
      break;
    if (sleepSec > 0)
      std::this_thread::sleep_for(std::chrono::seconds(sleepSec));
  } while (std::chrono::steady_clock::now() < deadline);

  std::printf("Done cycles=%d fails=%d\n", cycle, fails);
  std::fflush(stdout);
  return fails > 0 ? 3 : 0;
}
