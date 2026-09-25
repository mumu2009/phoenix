/* Host gates for KVM tensor + RoPE order. No llama. */
#include "addons/ThePlugInForSecurity/meme_tensor.hpp"
#include "addons/ThePlugInForSecurity/security_core.hpp"
#include "memetic_existence.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using phoenix::secamp::DiscreteGraph;
using phoenix::secamp::GraphEdge;
using phoenix::secamp::analyzeGraph;
using phoenix::secamp::applyRope;
using phoenix::secamp::buildMemeTensorSpace;
using phoenix::secamp::encodeSentenceRope;
using phoenix::secamp::hashedWordVec;
using phoenix::secamp::queryMemeTensor;
using phoenix::secamp::tensorCosine;
using phoenix::secamp::tensorNeighborsOf;

static int gFails = 0;

static void expect(bool cond, const char *msg) {
  if (cond)
    return;
  ++gFails;
  std::fprintf(stderr, "FAIL %s\n", msg);
}

int main() {
  auto a = hashedWordVec("river", 32);
  auto b = hashedWordVec("river", 32);
  expect(tensorCosine(a, b) > 0.99, "hash vec deterministic");
  auto c = hashedWordVec("bakery", 32);
  expect(tensorCosine(a, c) < 0.99, "different words differ");

  auto rope0 = a;
  applyRope(rope0, 0);
  expect(tensorCosine(a, rope0) > 0.99, "pos0 is identity");
  auto rope1 = a;
  applyRope(rope1, 3);
  expect(tensorCosine(a, rope1) < 0.99, "rope moves the vector");

  const std::vector<std::string> ids = {"meme_p_a", "meme_p_b", "meme_p_c"};
  const std::vector<std::vector<std::string>> words = {
      {"south", "river", "survey"},
      {"bakery", "flour", "yeast"},
      {"the", "of", "and"}};
  const auto space = buildMemeTensorSpace(ids, words);
  expect(space.memes.size() == 3, "3 memes");
  const std::vector<std::string> sent = {"south", "river", "survey"};
  const auto q = queryMemeTensor(space, sent, "meme_p_a");
  expect(q.tensorRank == 0, "own sentence nearest");
  expect(q.orderLift >= 1.0, "bind order lift >= 1");
  expect(q.nearestSentence == "meme_p_a", "nearest sentence is own meme");
  expect(q.nearestWord == "south" || q.nearestWord == "river" ||
             q.nearestWord == "survey",
         "nearest word in own bag");
  const std::vector<std::string> fwd = {"south", "river", "survey"};
  const std::vector<std::string> rev = {"survey", "river", "south"};
  const auto vf = encodeSentenceRope(space, fwd);
  const auto vr = encodeSentenceRope(space, rev);
  expect(tensorCosine(vf, vr) < 0.999, "rope sentence order sensitive");

  DiscreteGraph g;
  g.ids = {"hub", "p1", "p2", "p3"};
  g.layers = {"meme", "meme", "meme", "meme"};
  g.mapped = {{"hubword"}, {"w1"}, {"w2"}, {"w3"}};
  g.edges = {GraphEdge{0, 1, 1.0}, GraphEdge{0, 2, 1.0}, GraphEdge{0, 3, 1.0}};
  const auto report = analyzeGraph(g);
  expect(report.ok, "analyzeGraph ok");
  bool hubNear = false;
  for (const auto &n : report.nodes) {
    if (n.id == "hub" && !n.neighborIds.empty() && !n.tensorNeighbors.empty() &&
        !n.nearestWords.empty())
      hubNear = true;
  }
  expect(hubNear, "hub has graph + tensor neighbors");
  const auto tn = tensorNeighborsOf(space, "meme_p_a", 2);
  expect(!tn.empty(), "tensorNeighborsOf");

  phoenix::memetic::InMemoryStore store;
  const std::string corpus =
      "The south river survey marker stands beside the tidal marsh after "
      "harvest season and records the channel depth for local boats. "
      "A lunar greenhouse harvest schedule lists polar survey beacon grid "
      "times for the winter crew. The bakery inventory shelf count lists "
      "flour bags and yeast jars for the morning shift only.";
  expect(phoenix::memetic::ingestCorpusText(store, corpus, 8) >= 2, "ingest compose corpus");
  const auto screened = phoenix::memetic::screenMemesFromBarrier(store, 6);
  expect(!screened.empty(), "screened for compose");
  if (!screened.empty()) {
    const auto rag = phoenix::memetic::composeCarrierRag(store, screened.front().id, corpus, 8, 1);
    expect(!rag.text.empty(), "composed text");
    expect(rag.text.find(' ') != std::string::npos, "composed has spaces");
    expect(rag.method.find("tensor") != std::string::npos, "tensor method");
    expect(rag.text != phoenix::memetic::carrierOfMeme(store, screened.front().id),
           "composed is not the word bag");
    expect(phoenix::memetic::detectMemeInText(store, rag.text, screened.front().id).present,
           "composed presents meme");
  }

  phoenix::memetic::InMemoryStore splitStore;
  const std::string splitCorpus =
      "The south river survey marker stands beside the tidal marsh. "
      "The bakery inventory shelf count lists flour bags and yeast jars.";
  expect(phoenix::memetic::ingestCorpusText(splitStore, splitCorpus, 8) >= 2,
         "ingest two topics");
  bool glued = false;
  for (const auto &id : splitStore.nodeIds()) {
    const auto words = splitStore.wordsOfMeme(id);
    bool river = false;
    bool bakery = false;
    for (const auto &w : words) {
      if (w == "river" || w == "marsh")
        river = true;
      if (w == "bakery" || w == "yeast")
        bakery = true;
    }
    if (river && bakery)
      glued = true;
  }
  expect(!glued, "unrelated articles do not share one meme");

  phoenix::memetic::InMemoryStore longStore;
  const std::string longTopic =
      "Valkyria Chronicles II director Takeshi Ozawa and composer Hitoshi "
      "Sakimoto returned with designer Raita Honjou. Stories are told through "
      "comic book like panels and animated portraits of characters speaking. "
      "Each squad member has unique potentials and skills to learn in battle. "
      "The expanded edition arrived that November after western critics praised "
      "sales in Japan.";
  expect(phoenix::memetic::ingestCorpusText(longStore, longTopic, 8) >= 1,
         "ingest one long topic");
  int screenedFat = 0;
  for (const auto &row : phoenix::memetic::screenMemesFromBarrier(longStore, 8)) {
    if (static_cast<int>(row.words.size()) > phoenix::memetic::Params{}.ngramMax)
      ++screenedFat;
  }
  expect(screenedFat == 0, "screen skips bags past ngramMax");

  phoenix::memetic::InMemoryStore priorStore;
  const char *hubIds[] = {"meme_p_hub", "meme_p_b", "meme_p_c",
                          "meme_p_d", "meme_p_e", "meme_p_f"};
  const char *uniq[] = {"river", "bakery", "quarry",
                        "orchard", "harbor", "depot"};
  for (int i = 0; i < 6; ++i) {
    priorStore.bind("the", hubIds[i]);
    priorStore.bind("of", hubIds[i]);
    priorStore.bind("and", hubIds[i]);
    priorStore.bind(uniq[i], hubIds[i]);
  }
  priorStore.bind("from", hubIds[0]);
  priorStore.bind("with", hubIds[0]);
  expect(phoenix::memetic::mappingMergeStopWord("from"), "from is language stop");
  const auto stopHit = phoenix::memetic::detectMemeInText(
      priorStore,
      "the of and to a in on for is was with as by at from that this it be or are",
      hubIds[0]);
  expect(!stopHit.present, "stop query has no content evidence");
  expect(phoenix::memetic::mappingQueryHasContentForMeme(
             {"river", "the"}, {"the", "of", "and", "river"}),
         "content token counts");
  expect(!phoenix::memetic::mappingQueryHasContentForMeme(
             {"the", "from", "with"}, {"the", "from", "with", "river"}),
         "stops alone do not count");

  phoenix::memetic::InMemoryStore tfStore;
  tfStore.bind("river", "meme_p_w");
  tfStore.bind("river", "meme_p_w");
  tfStore.bind("used", "meme_p_w");
  tfStore.bind("used", "meme_p_x");
  tfStore.bind("used", "meme_p_y");
  expect(tfStore.wordTf("meme_p_w", "river") == 2, "rebind increments tf");
  const auto wts = tfStore.weightsOfMeme("meme_p_w");
  double aRiver = 0.0;
  double aUsed = 0.0;
  for (const auto &wp : wts) {
    if (wp.first == "river")
      aRiver = wp.second;
    if (wp.first == "used")
      aUsed = wp.second;
  }
  expect(aRiver > aUsed, "meme-conditional alpha ranks characteristic word");
  {
    using phoenix::memetic::MappingWordHit;
    std::unordered_map<std::string, double> q{{"used", 1.0}, {"river", 1.0}};
    std::unordered_map<std::string, std::vector<MappingWordHit>> hits;
    hits["used"] = {{"meme_p_w", aUsed}, {"meme_p_x", 1.0 / 3.0},
                    {"meme_p_y", 1.0 / 3.0}};
    hits["river"] = {{"meme_p_w", aRiver}};
    std::unordered_map<std::string, double> seeds;
    phoenix::memetic::MappingMassFrom from;
    phoenix::memetic::mappingAddConditionedSeeds(q, hits, &seeds, &from);
    expect(seeds["meme_p_w"] > seeds["meme_p_x"],
           "query-conditioned seed prefers the named meme");
    expect(from["meme_p_w"].count("river") == 1,
           "isotope records the named source word");
    expect(!phoenix::memetic::mappingNewQueryId().empty(), "query id");
  }

  const std::string snapPath = "build/tmp_meme_tf_roundtrip.txt";
  tfStore.saveText(snapPath);
  phoenix::memetic::InMemoryStore loaded;
  expect(loaded.loadText(snapPath), "load weighted snap");
  expect(loaded.wordTf("meme_p_w", "river") == 2, "saved tf roundtrip");
  {
    std::ofstream oldSnap(snapPath, std::ios::binary);
    oldSnap << "NODE meme_p_old\nWORD meme_p_old marsh\n";
  }
  phoenix::memetic::InMemoryStore oldLoaded;
  expect(oldLoaded.loadText(snapPath), "load old WORD without tf");
  expect(oldLoaded.wordTf("meme_p_old", "marsh") == 1, "old snap tf defaults to 1");

  if (gFails > 0) {
    std::fprintf(stderr, "%d fails\n", gFails);
    return 1;
  }
  std::printf("ok semantic=%.3f order=%.3f trank=%d tnear=%zu nword=%s nsent=%s\n",
              q.semanticCos, q.orderLift, q.tensorRank, tn.size(),
              q.nearestWord.c_str(), q.nearestSentence.c_str());
  return 0;
}
