/* meme_semantic_pipeline.cpp - Semantic dynamic slicing + multi-meme
   annealing pipeline (security-module components on the screening side).

   slice:  corpus -> GNN ingest per unit (official buildMemeSequence via
           ingestDocumentLike; the graph is grown once, read-only after)
           -> shared-meme gap flow -> SemanticSlicer -> manifest of
           (skipUnits, maxUnits) windows at semantic seams + optional
           per-slice corpus text files. Feeds the existing frozen
           instrument unchanged.

   anneal: frozen snap(s) -> minimal meme pool (2..softWords; bigger bags
           are not samples) -> organic merge pairs (method 1, the GNN's
           own mappingMergeAllowed gate) -> MemeAnnealer (method 2,
           simulated annealing with a hard word cap so no giant meme
           appears) -> ranked groups json.

   export: materialize top combined groups: compose each member's carrier
           sentence on the BASE snap (composeCarrierRag, query-face), join,
           ingest into a COPY of the base snap (the GNN organically mints
           the combined node - method 1 executing method 2's plan), find
           the node whose typical set matches the merged set (official
           mappingAlphaIdentity), write snap v2 + a screen-shaped json so
           tools/manifold_export_candidates.py exports remote candidates
           unchanged. Singleton groups (minimal units worth testing as-is)
           are emitted against the base snap as pole "minimal".

   The GNN core (memetic_existence) is only called, never modified. */

#include "../addons/ThePlugInForSecurity/meme_annealer.hpp"
#include "../addons/ThePlugInForSecurity/meme_semantic_slicer.hpp"
#include "../memetic_existence.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;
namespace mem = phoenix::memetic;
namespace sec = phoenix::secamp;

namespace {

std::string readAll(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool writeAll(const std::string &path, const std::string &data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    return false;
  out << data;
  return static_cast<bool>(out);
}

/* ---------------- slice ---------------- */

int cmdSlice(int argc, char **argv) {
  std::string corpus, outFile, slicesDir, snapOut;
  int startUnit = 0;
  int unitCap = 0; /* 0 = to corpus end */
  sec::SemanticSliceConfig cfg;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--corpus")
      corpus = next("--corpus");
    else if (a == "--out")
      outFile = next("--out");
    else if (a == "--slices-dir")
      slicesDir = next("--slices-dir");
    else if (a == "--snap-out")
      snapOut = next("--snap-out");
    else if (a == "--start-unit")
      startUnit = std::stoi(next("--start-unit"));
    else if (a == "--unit-cap")
      unitCap = std::stoi(next("--unit-cap"));
    else if (a == "--min-units")
      cfg.minUnits = std::stoi(next("--min-units"));
    else if (a == "--max-units")
      cfg.maxUnits = std::stoi(next("--max-units"));
    else if (a == "--target-units")
      cfg.targetUnits = std::stoi(next("--target-units"));
    else if (a == "--lookahead")
      cfg.lookahead = std::stoi(next("--lookahead"));
    else {
      std::cerr << "unknown arg " << a << "\n";
      return 2;
    }
  }
  if (corpus.empty() || outFile.empty()) {
    std::cerr << "slice needs --corpus and --out\n";
    return 2;
  }
  const std::string raw = readAll(corpus);
  if (raw.empty()) {
    std::cerr << "cannot read corpus\n";
    return 2;
  }
  const auto allUnits = mem::splitCorpusUnits(raw);
  const int total = static_cast<int>(allUnits.size());
  const int begin = std::max(0, std::min(startUnit, total));
  const int end =
      unitCap > 0 ? std::min(total, begin + unitCap) : total;
  if (end <= begin) {
    std::cerr << "empty unit range\n";
    return 1;
  }

  /* One shared store for the region: cross-unit organic merging is the
     semantic glue the slicer reads. GNN is only called, never modified. */
  mem::InMemoryStore store;
  const mem::Params p;
  std::vector<sec::SliceUnitFootprint> units;
  units.reserve(static_cast<size_t>(end - begin));
  int ingested = 0;
  for (int u = begin; u < end; ++u) {
    sec::SliceUnitFootprint fp;
    fp.unitIndex = u;
    const auto tokens = mem::tokenizeAscii(allUnits[static_cast<size_t>(u)]);
    if (tokens.size() >= 6) {
      fp.memeSeq = mem::ingestDocumentLike(store, tokens, p);
      ++ingested;
    }
    units.push_back(std::move(fp));
  }

  const auto flows = sec::SemanticSlicer::gapFlows(units, cfg.lookahead);
  auto slices = sec::SemanticSlicer::slice(units, cfg);
  /* The slicer works on local vector indices; translate back to absolute
     corpus-unit indices via the footprints. */
  for (auto &s : slices) {
    s.beginUnit = units[static_cast<size_t>(s.beginUnit)].unitIndex;
    s.endUnit = s.endUnit < static_cast<int>(units.size())
                    ? units[static_cast<size_t>(s.endUnit)].unitIndex
                    : end;
  }

  double flowMean = 0.0;
  double flowMin = flows.empty() ? 0.0 : flows[0];
  for (const double f : flows) {
    flowMean += f;
    flowMin = std::min(flowMin, f);
  }
  if (!flows.empty())
    flowMean /= static_cast<double>(flows.size());

  json rows = json::array();
  for (const auto &s : slices) {
    rows.push_back(json{{"beginUnit", s.beginUnit},
                        {"endUnit", s.endUnit},
                        {"units", s.endUnit - s.beginUnit},
                        {"seamAfter", s.seamAfter},
                        {"memeRefs", s.memeRefs},
                        {"uniqueMemes", s.uniqueMemes}});
  }
  json doc{
      {"kind", "meme semantic slices v1"},
      {"corpus", corpus},
      {"startUnit", begin},
      {"unitCap", unitCap},
      {"unitsInRange", end - begin},
      {"ingestedUnits", ingested},
      {"nodes", store.nodeCount()},
      {"config",
       json{{"minUnits", cfg.minUnits},
            {"maxUnits", cfg.maxUnits},
            {"targetUnits", cfg.targetUnits},
            {"lookahead", cfg.lookahead}}},
      {"flowMean", flowMean},
      {"flowMin", flowMin},
      {"slices", rows}};
  if (!writeAll(outFile, doc.dump(2))) {
    std::cerr << "cannot write " << outFile << "\n";
    return 2;
  }

  if (!slicesDir.empty()) {
    /* Per-slice corpus text: units joined by blank lines, same shape the
       instrument expects. */
    const size_t slash = slicesDir.find_last_of("/\\");
    if (slash != std::string::npos) {
      /* best-effort single-level mkdir (parent must exist) */
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      std::string text;
      for (int u = slices[i].beginUnit; u < slices[i].endUnit; ++u) {
        text += allUnits[static_cast<size_t>(u)];
        text += "\n\n";
      }
      char name[64];
      std::snprintf(name, sizeof(name), "slice_%04d_%04d.txt",
                    slices[i].beginUnit, slices[i].endUnit);
      if (!writeAll(slicesDir + "/" + name, text)) {
        std::cerr << "cannot write slice file in " << slicesDir << "\n";
        return 2;
      }
    }
  }
  if (!snapOut.empty())
    store.saveText(snapOut);

  std::cout << "slices=" << slices.size() << " units=" << (end - begin)
            << " ingested=" << ingested << " nodes=" << store.nodeCount()
            << " flowMean=" << flowMean << " flowMin=" << flowMin << "\n";
  for (const auto &s : slices)
    std::cout << "  [" << s.beginUnit << "," << s.endUnit << ") n="
              << (s.endUnit - s.beginUnit) << " seam=" << s.seamAfter
              << " uniqMemes=" << s.uniqueMemes << "\n";
  return 0;
}

/* ---------------- anneal ---------------- */

struct SnapPool {
  std::vector<sec::AnnealMeme> pool;
  std::vector<sec::AnnealEdge> edges;
  std::unordered_map<std::string, int> ix; /* snap node id -> pool index */
};

SnapPool poolFromSnap(const mem::InMemoryStore &store, int softWords) {
  SnapPool sp;
  std::unordered_map<std::string, int> df;
  for (const auto &id : store.nodeIds())
    for (const auto &w : store.wordsOfMeme(id))
      if (!w.empty())
        df[w]++;
  for (const auto &id : store.nodeIds()) {
    const auto words = store.wordsOfMeme(id);
    if (words.size() < 2 || static_cast<int>(words.size()) > softWords)
      continue;
    sec::AnnealMeme m;
    m.id = id;
    m.weights = store.weightsOfMeme(id);
    m.degree = static_cast<int>(store.neighborsOf(id).size());
    if (m.weights.size() < 2)
      continue;
    sp.ix[id] = static_cast<int>(sp.pool.size());
    sp.pool.push_back(std::move(m));
  }
  for (const auto &id : store.nodeIds()) {
    auto ai = sp.ix.find(id);
    if (ai == sp.ix.end())
      continue;
    for (const auto &nb : store.neighborsOf(id)) {
      auto bi = sp.ix.find(nb.first);
      if (bi == sp.ix.end() || bi->second <= ai->second)
        continue;
      sp.edges.push_back(sec::AnnealEdge{ai->second, bi->second, nb.second});
    }
  }
  return sp;
}

int cmdAnneal(int argc, char **argv) {
  std::vector<std::string> snaps;
  std::string outFile;
  sec::AnnealConfig cfg;
  int minOverlap = 2;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--snap")
      snaps.push_back(next("--snap"));
    else if (a == "--out")
      outFile = next("--out");
    else if (a == "--seed")
      cfg.seed = static_cast<std::uint64_t>(
          std::stoull(next("--seed")));
    else if (a == "--steps")
      cfg.steps = std::stoi(next("--steps"));
    else if (a == "--max-words")
      cfg.maxMemeWords = std::stoi(next("--max-words"));
    else if (a == "--soft-words")
      cfg.softWords = std::stoi(next("--soft-words"));
    else if (a == "--size-penalty")
      cfg.sizePenalty = std::stod(next("--size-penalty"));
    else if (a == "--cohesion-weight")
      cfg.cohesionWeight = std::stod(next("--cohesion-weight"));
    else if (a == "--t0")
      cfg.t0 = std::stod(next("--t0"));
    else if (a == "--cooling")
      cfg.cooling = std::stod(next("--cooling"));
    else if (a == "--min-overlap")
      minOverlap = std::stoi(next("--min-overlap"));
    else {
      std::cerr << "unknown arg " << a << "\n";
      return 2;
    }
  }
  if (snaps.empty() || outFile.empty()) {
    std::cerr << "anneal needs --snap and --out\n";
    return 2;
  }

  /* Merge pools across snaps; ids are content-addressed (meme_p_sha1), so
     the same bag on two snaps is the same meme - dedupe by id. */
  SnapPool merged;
  for (const auto &path : snaps) {
    mem::InMemoryStore store;
    if (!store.loadText(path)) {
      std::cerr << "cannot load snap " << path << "\n";
      return 2;
    }
    auto sp = poolFromSnap(store, cfg.softWords);
    std::unordered_map<int, int> remap;
    for (size_t i = 0; i < sp.pool.size(); ++i) {
      auto it = merged.ix.find(sp.pool[i].id);
      if (it == merged.ix.end()) {
        remap[static_cast<int>(i)] = static_cast<int>(merged.pool.size());
        merged.ix[sp.pool[i].id] = static_cast<int>(merged.pool.size());
        merged.pool.push_back(std::move(sp.pool[i]));
      } else {
        remap[static_cast<int>(i)] = it->second;
        merged.pool[static_cast<size_t>(it->second)].degree +=
            sp.pool[i].degree;
      }
    }
    for (const auto &e : sp.edges)
      merged.edges.push_back(
          sec::AnnealEdge{remap[e.a], remap[e.b], e.w});
  }

  const auto organic = sec::MemeAnnealer::organicMergePairs(
      merged.pool, minOverlap, cfg.maxMemeWords);
  const auto res = sec::MemeAnnealer::run(merged.pool, merged.edges, cfg);

  json organicRows = json::array();
  const size_t oTake = std::min<size_t>(64, organic.size());
  for (size_t i = 0; i < oTake; ++i) {
    organicRows.push_back(json{
        {"a", merged.pool[static_cast<size_t>(organic[i].first)].id},
        {"b", merged.pool[static_cast<size_t>(organic[i].second)].id}});
  }
  json groups = json::array();
  for (const auto &g : res.groups) {
    json members = json::array();
    for (const int mi : g.members)
      members.push_back(merged.pool[static_cast<size_t>(mi)].id);
    json mergedW = json::array();
    for (const auto &wp : g.merged)
      mergedW.push_back(json{{"w", wp.first}, {"a", wp.second}});
    groups.push_back(json{{"members", members},
                          {"wordCount", g.wordCount},
                          {"fitness", g.fitness},
                          {"cohesion", g.cohesion},
                          {"typical", g.typical},
                          {"merged", mergedW}});
  }
  json doc{{"kind", "meme anneal v1"},
           {"snaps", snaps},
           {"poolSize", merged.pool.size()},
           {"edgeCount", merged.edges.size()},
           {"config",
            json{{"maxMemeWords", cfg.maxMemeWords},
                 {"softWords", cfg.softWords},
                 {"sizePenalty", cfg.sizePenalty},
                 {"cohesionWeight", cfg.cohesionWeight},
                 {"t0", cfg.t0},
                 {"cooling", cfg.cooling},
                 {"steps", cfg.steps},
                 {"seed", cfg.seed},
                 {"minOverlap", minOverlap}}},
           {"organicMergePairCount", organic.size()},
           {"organicMergePairs", organicRows},
           {"initialEnergy", res.initialEnergy},
           {"energy", res.energy},
           {"stepsDone", res.stepsDone},
           {"groups", groups}};
  if (!writeAll(outFile, doc.dump(2))) {
    std::cerr << "cannot write " << outFile << "\n";
    return 2;
  }
  std::cout << "pool=" << merged.pool.size() << " edges=" << merged.edges.size()
            << " organicPairs=" << organic.size()
            << " E0=" << res.initialEnergy << " E=" << res.energy
            << " groups=" << res.groups.size() << "\n";
  return 0;
}

/* ---------------- export ---------------- */

int cmdExport(int argc, char **argv) {
  std::string snapPath, annealPath, corpusPath, outSnap, outScreen;
  int units = 256;
  int groupCap = 64;
  int singletonCap = 200;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--snap")
      snapPath = next("--snap");
    else if (a == "--anneal")
      annealPath = next("--anneal");
    else if (a == "--corpus")
      corpusPath = next("--corpus");
    else if (a == "--units")
      units = std::stoi(next("--units"));
    else if (a == "--out-snap")
      outSnap = next("--out-snap");
    else if (a == "--out-screen")
      outScreen = next("--out-screen");
    else if (a == "--groups")
      groupCap = std::stoi(next("--groups"));
    else if (a == "--singletons")
      singletonCap = std::stoi(next("--singletons"));
    else {
      std::cerr << "unknown arg " << a << "\n";
      return 2;
    }
  }
  if (snapPath.empty() || annealPath.empty() || corpusPath.empty() ||
      outSnap.empty() || outScreen.empty()) {
    std::cerr << "export needs --snap --anneal --corpus --out-snap "
                 "--out-screen\n";
    return 2;
  }
  mem::InMemoryStore base;
  if (!base.loadText(snapPath)) {
    std::cerr << "cannot load snap\n";
    return 2;
  }
  const std::string corpus = readAll(corpusPath);
  if (corpus.empty()) {
    std::cerr << "cannot read corpus\n";
    return 2;
  }
  json anneal = json::parse(readAll(annealPath));
  const auto groupsJ = anneal["groups"].get<std::vector<json>>();

  /* snap v2 = base + organically ingested combined carriers. The base
     file on disk stays frozen. */
  mem::InMemoryStore v2 = base;
  const mem::Params p;

  json screenRows = json::array();
  int materialized = 0;
  int failed = 0;
  int failedNoCarrier = 0;
  int failedShort = 0;
  int failedNoNode = 0;
  int emittedSingletons = 0;

  for (const auto &g : groupsJ) {
    const auto members = g["members"].get<std::vector<std::string>>();
    const int wordCount = g.value("wordCount", 0);
    if (members.size() < 2) {
      /* Singleton: a minimal unit worth testing as-is, judged on the
         pristine base snap. */
      if (emittedSingletons < singletonCap) {
        const std::string &id = members[0];
        const auto words = base.wordsOfMeme(id);
        if (words.size() >= 2) {
          std::string bag;
          for (const auto &w : base.wordsOfMeme(id)) {
            if (!bag.empty())
              bag += " ";
            bag += w;
          }
          screenRows.push_back(json{{"id", id},
                                    {"pole", "minimal"},
                                    {"wordCount", wordCount},
                                    {"carrier", bag},
                                    {"fitness", g.value("fitness", 0.0)},
                                    {"snap", "base"}});
          ++emittedSingletons;
        }
      }
      continue;
    }
    if (materialized >= groupCap)
      continue;

    /* Carrier: one presenting sentence per member (query-face compose on
       the frozen base), joined so junction windows mix both word sets. */
    std::string carrier;
    int memberCarriers = 0;
    for (const auto &mid : members) {
      if (base.wordsOfMeme(mid).empty())
        continue;
      auto rc = mem::composeCarrierRag(base, mid, corpus, units, 1, false);
      if (rc.text.empty())
        continue;
      if (!carrier.empty())
        carrier += " ";
      carrier += rc.text;
      ++memberCarriers;
    }
    if (memberCarriers < 2) {
      ++failed;
      ++failedNoCarrier;
      continue;
    }
    const auto tokens = mem::tokenizeAscii(carrier);
    if (tokens.size() < 6) {
      ++failed;
      ++failedShort;
      continue;
    }

    /* The GNN organically integrates the combined window (method 1). */
    const auto seq = mem::ingestDocumentLike(v2, tokens, p);

    /* Find the node whose typical set matches the merged set. Size-capped:
       an over-cap node (e.g. an old bag that swallowed the carrier
       windows) is exactly the giant meme this pipeline exists to avoid. */
    const int acceptCap = anneal.value("config", json::object())
                              .value("maxMemeWords", 28);
    std::vector<std::pair<std::string, double>> mergedW;
    for (const auto &wp : g["merged"].get<std::vector<json>>())
      mergedW.push_back(
          {wp.value("w", std::string()), wp.value("a", 0.0)});
    std::string bestId;
    mem::AlphaIdentity bestIdn;
    double bestScore = -1.0;
    auto considerNode = [&](const std::string &nid) {
      if (static_cast<int>(v2.wordsOfMeme(nid).size()) > acceptCap)
        return;
      const auto idn =
          mem::mappingAlphaIdentity(mergedW, v2.weightsOfMeme(nid));
      const double score = idn.typicalHit * 10.0 + idn.massCover +
                           idn.cosine;
      if (idn.same && score > bestScore) {
        bestScore = score;
        bestId = nid;
        bestIdn = idn;
      }
    };
    for (const auto &nid : seq)
      considerNode(nid);
    if (bestId.empty()) {
      /* Fall back to any size-capped node on the v2 graph. */
      for (const auto &nid : v2.nodeIds())
        considerNode(nid);
    }
    if (bestId.empty()) {
      ++failed;
      ++failedNoNode;
      continue;
    }
    std::string bag;
    for (const auto &w : v2.wordsOfMeme(bestId)) {
      if (!bag.empty())
        bag += " ";
      bag += w;
    }
    screenRows.push_back(json{{"id", bestId},
                              {"pole", "anneal"},
                              {"wordCount",
                               static_cast<int>(v2.wordsOfMeme(bestId).size())},
                              {"carrier", bag},
                              {"fitness", g.value("fitness", 0.0)},
                              {"cohesion", g.value("cohesion", 0.0)},
                              {"typicalHit", bestIdn.typicalHit},
                              {"typicalNeed", bestIdn.typicalNeed},
                              {"massCover", bestIdn.massCover},
                              {"members", members},
                              {"snap", "v2"}});
    ++materialized;
  }

  v2.saveText(outSnap);
  json screen{{"kind", "meme semantic anneal screen v1"},
              {"baseSnap", snapPath},
              {"snapV2", outSnap},
              {"materializedCombined", materialized},
              {"failedCombined", failed},
              {"failedNoCarrier", failedNoCarrier},
              {"failedShort", failedShort},
              {"failedNoNode", failedNoNode},
              {"singletons", emittedSingletons},
              {"memes", screenRows}};
  if (!writeAll(outScreen, screen.dump(2))) {
    std::cerr << "cannot write " << outScreen << "\n";
    return 2;
  }
  std::cout << "materialized=" << materialized << " failed=" << failed
            << " (noCarrier=" << failedNoCarrier
            << " short=" << failedShort << " noNode=" << failedNoNode
            << ") singletons=" << emittedSingletons
            << " v2nodes=" << v2.nodeCount() << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: meme_semantic_pipeline <slice|anneal|export> ...\n";
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "slice")
    return cmdSlice(argc, argv);
  if (cmd == "anneal")
    return cmdAnneal(argc, argv);
  if (cmd == "export")
    return cmdExport(argc, argv);
  std::cerr << "unknown command " << cmd << "\n";
  return 2;
}
