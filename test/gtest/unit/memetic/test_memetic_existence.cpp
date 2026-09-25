#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "graph_diffusion_summarizer.hpp"
#include "memetic_existence.hpp"

using phoenix::memetic::ExistenceResult;
using phoenix::memetic::InMemoryStore;
using phoenix::memetic::Params;
using phoenix::memetic::SerialReplicationResult;
using phoenix::memetic::carrierOfMeme;
using phoenix::memetic::composeCarrierRag;
using phoenix::memetic::detectMemeInText;
using phoenix::memetic::expressActivated;
using phoenix::memetic::graphNoteActivated;
using phoenix::memetic::ingestCorpusText;
using phoenix::memetic::ingestDocumentLike;
using phoenix::memetic::screenMemesFromBarrier;
using phoenix::memetic::judgeExpression;
using phoenix::memetic::kHarmlessNonce;
using phoenix::memetic::payloadFragments;
using phoenix::memetic::unitPropagates;
using phoenix::memetic::kMinStabilityRounds;
using phoenix::memetic::runHarmlessExistence;
using phoenix::memetic::runSerialReplication;
using phoenix::memetic::runSerialReplicationBattery;
using phoenix::memetic::textExpressesHarmlessMeme;

namespace {

bool hasNonce(const std::vector<std::string> &words) {
  return std::find(words.begin(), words.end(), std::string(kHarmlessNonce)) !=
         words.end();
}

} // namespace

TEST(MemeticExistenceTest, OfficialDefaultsReconstructNonceFromCueNotDistractor) {
  const ExistenceResult r = runHarmlessExistence();
  EXPECT_FALSE(r.queryMentionsNonce);
  EXPECT_FALSE(hasNonce({"north", "dock", "ledger"}));
  /* Official n-gram + minOverlap=2 merges sliding windows into one cluster.
     Reconstructing the nonce from cue words is recall of that cluster, not a
     side-table write. An unrelated query must not see it. */
  EXPECT_TRUE(r.kvmReconstructedNonce || r.diffusionRankedNonceMeme);
  EXPECT_FALSE(r.distractorReconstructedNonce);
  EXPECT_FALSE(r.distractorDiffusionRankedNonce);
}

TEST(MemeticExistenceTest, HighOverlapThresholdNeedsGraphDiffusionNotSameBag) {
  Params p;
  p.minOverlap = 99;
  const ExistenceResult r = runHarmlessExistence(p);
  EXPECT_FALSE(r.queryMentionsNonce);
  EXPECT_FALSE(r.kvmReconstructedNonce)
      << "cue window must not contain the nonce when clusters cannot merge";
  EXPECT_TRUE(r.diffusionRankedNonceMeme)
      << "official GraphDiffusionSummarizer must activate the nonce-bearing meme";
  EXPECT_FALSE(r.distractorReconstructedNonce);
  EXPECT_FALSE(r.distractorDiffusionRankedNonce);
}

TEST(MemeticExistenceTest, GraphDiffusionWalksToNeighbor) {
  std::vector<std::string> ids{"a", "b", "c"};
  std::vector<std::vector<std::tuple<size_t, double, int>>> adj(3);
  adj[0].push_back({1, 1.0, 0});
  adj[1].push_back({0, 1.0, 0});
  adj[1].push_back({2, 1.0, 0});
  adj[2].push_back({1, 1.0, 0});
  phoenix::graph::GraphDiffusionSummarizer summarizer;
  auto out = summarizer.summarize(ids, adj, {1.0, 0.0, 0.0}, 5, 0.85, 3);
  ASSERT_EQ(out.rankedNodes.size(), 3u);
  double scoreB = 0.0;
  double scoreA = 0.0;
  for (const auto &row : out.rankedNodes) {
    if (row.first == "a")
      scoreA = row.second;
    if (row.first == "b")
      scoreB = row.second;
  }
  EXPECT_GT(scoreA, 0.0);
  EXPECT_GT(scoreB, 0.0);
}

TEST(MemeticExistenceTest, Sha1MatchesFips180VectorAbc) {
  EXPECT_EQ(phoenix::memetic::sha1Hex("abc"),
            "a9993e364706816aba3e25717850c26c9cd0d89d");
}

TEST(MemeticExistenceTest, SerialReplicationSurvivesWipeWithoutTwoAgents) {
  const SerialReplicationResult r = runSerialReplication();
  EXPECT_TRUE(textExpressesHarmlessMeme(r.input));
  EXPECT_FALSE(r.residueDetected);
  ASSERT_EQ(static_cast<int>(r.rounds.size()), kMinStabilityRounds);
  EXPECT_EQ(r.expressedRounds, kMinStabilityRounds);
  for (const auto &rnd : r.rounds) {
    EXPECT_TRUE(rnd.emptyStoreBeforeIngest);
    EXPECT_FALSE(rnd.decoderEchoOnEmpty);
    EXPECT_FALSE(rnd.heldOutResidue);
    EXPECT_TRUE(rnd.expresses);
    EXPECT_FALSE(rnd.text.empty());
  }
  EXPECT_FALSE(r.distractorExpressesMeme);
}

TEST(MemeticExistenceTest, WipeLeavesNoGraphOrDecoderResidue) {
  InMemoryStore store;
  ingestDocumentLike(store, {"north", "dock", "ceremonial", "pigment", "code",
                             kHarmlessNonce},
                     {});
  ASSERT_FALSE(store.empty());
  store.clear();
  EXPECT_TRUE(store.empty());
  EXPECT_EQ(store.nodeCount(), 0u);
  EXPECT_TRUE(store.memesOfWord(kHarmlessNonce).empty());
  const std::string echoed = expressActivated(
      store, {"north", "dock", kHarmlessNonce}, {}, 0, kHarmlessNonce);
  EXPECT_TRUE(echoed.empty());
  EXPECT_FALSE(textExpressesHarmlessMeme(echoed));

  InMemoryStore fresh;
  const std::string ghost = expressActivated(
      fresh, {"north", "dock", "ceremonial", kHarmlessNonce}, {}, 0,
      kHarmlessNonce);
  EXPECT_TRUE(ghost.empty());
}

TEST(MemeticExistenceTest, SerialBatteryNotAccidentalAcrossIndependentGroups) {
  const auto bat = runSerialReplicationBattery();
  EXPECT_GE(bat.groups.size(), 8u);
  EXPECT_LE(bat.groups.size(), 12u);
  EXPECT_GE(bat.positiveTotal, 6);
  EXPECT_GE(bat.negativeTotal, 2);
  EXPECT_EQ(bat.positivePassed, bat.positiveTotal);
  EXPECT_EQ(bat.negativePassed, bat.negativeTotal);
  for (const auto &g : bat.groups) {
    EXPECT_EQ(static_cast<int>(g.rounds.size()), kMinStabilityRounds) << g.id;
    EXPECT_FALSE(g.residueDetected) << g.id;
    EXPECT_TRUE(g.passed) << g.id << " payload=" << g.payload
                          << " in=" << g.input
                          << " expressed=" << g.expressedRounds
                          << " leak=" << g.leakedForbidden
                          << " distE=" << g.distractorExpressesMeme;
  }
}

TEST(MemeticExistenceTest, JudgeAllowsSplitNonceAndReturnAfterDrift) {
  const auto fr = payloadFragments(kHarmlessNonce);
  EXPECT_FALSE(fr.empty());
  const auto split = judgeExpression(
      "QX7 is a celadon glaze lot 47 at the north dock", kHarmlessNonce,
      {"north", "dock", "ceremonial", "pigment", "code", kHarmlessNonce});
  EXPECT_FALSE(split.exactToken);
  EXPECT_TRUE(split.reconstructed);
  EXPECT_TRUE(split.unitExpressed);
  EXPECT_TRUE(split.carrierPresent);
  EXPECT_FALSE(split.drifted);

  const auto drifted = judgeExpression(
      "I would rather discuss bakery inventory.", kHarmlessNonce,
      {"north", "dock", "ceremonial", "pigment", "code", kHarmlessNonce});
  EXPECT_TRUE(drifted.drifted);
  EXPECT_FALSE(drifted.unitExpressed);

  const auto latent = judgeExpression(
      "the north dock ceremonial pigment code is unlisted", kHarmlessNonce,
      {"north", "dock", "ceremonial", "pigment", "code", kHarmlessNonce});
  EXPECT_TRUE(latent.latent);
  EXPECT_FALSE(latent.unitExpressed);

  using phoenix::memetic::ExpressionJudgement;
  ExpressionJudgement a{};
  a.unitExpressed = true;
  ExpressionJudgement b{};
  b.drifted = true;
  ExpressionJudgement c{};
  c.unitExpressed = true;
  EXPECT_FALSE(unitPropagates({a, b, b, b, b, b}));
  EXPECT_TRUE(unitPropagates({a, b, c, b, b, b}));
}

TEST(MemeticExistenceTest, GraphNoteDoesNotForcePayloadFirst) {
  InMemoryStore store;
  ingestDocumentLike(store, {"north", "dock", "ceremonial", "pigment", "code",
                             kHarmlessNonce},
                     {});
  const auto note = graphNoteActivated(
      store, {"north", "dock", "ceremonial", "pigment", "code", kHarmlessNonce},
      {});
  EXPECT_FALSE(note.empty());
  EXPECT_EQ(note.find("noted "), std::string::npos);
  EXPECT_EQ(note.rfind("graph", 0), 0u);
}

TEST(MemeticExistenceTest, BarrierScreenTracksMemeIdNotCarrierSentence) {
  InMemoryStore store;
  const std::string corpus =
      "The south river survey marker stands beside the tidal marsh after "
      "harvest season and records the channel depth for local boats. "
      "A lunar greenhouse harvest schedule lists polar survey beacon grid "
      "times for the winter crew. The bakery inventory shelf count lists "
      "flour bags and yeast jars for the morning shift only.";
  ASSERT_GE(ingestCorpusText(store, corpus, 8), 2);
  const size_t nodesBefore = store.nodeCount();
  const auto screened = screenMemesFromBarrier(store, 6);
  ASSERT_FALSE(screened.empty());
  const auto sample = screened.front();
  EXPECT_FALSE(sample.id.empty());
  EXPECT_GE(sample.words.size(), 2u);
  EXPECT_EQ(sample.carrier, carrierOfMeme(store, sample.id));
  EXPECT_EQ(sample.carrier.find("north dock ceremonial"), std::string::npos);

  const auto hit = detectMemeInText(store, sample.carrier, sample.id);
  EXPECT_TRUE(hit.present);
  EXPECT_EQ(store.nodeCount(), nodesBefore);

  const auto miss = detectMemeInText(
      store, "quantum chess tournament pairing sheet omega7", sample.id);
  EXPECT_FALSE(miss.present);
  EXPECT_EQ(store.nodeCount(), nodesBefore);
}

TEST(MemeticExistenceTest, ComposeRetrievesSourceSentenceNotBag) {
  InMemoryStore store;
  const std::string corpus =
      "The south river survey marker stands beside the tidal marsh after "
      "harvest season and records the channel depth for local boats. "
      "A lunar greenhouse harvest schedule lists polar survey beacon grid "
      "times for the winter crew. The bakery inventory shelf count lists "
      "flour bags and yeast jars for the morning shift only.";
  ASSERT_GE(ingestCorpusText(store, corpus, 8), 2);
  const auto screened = screenMemesFromBarrier(store, 6);
  ASSERT_FALSE(screened.empty());
  const auto rag = composeCarrierRag(store, screened.front().id, corpus, 8, 2);
  EXPECT_FALSE(rag.text.empty());
  EXPECT_NE(rag.text.find(' '), std::string::npos);
  EXPECT_NE(rag.method.find("tensor"), std::string::npos);
  EXPECT_NE(rag.text, carrierOfMeme(store, screened.front().id));
  EXPECT_TRUE(detectMemeInText(store, rag.text, screened.front().id).present);
  EXPECT_EQ(rag.text.find("north dock ceremonial"), std::string::npos);
}

TEST(MemeticExistenceTest, ComposeKeepsTargetExclusive) {
  InMemoryStore store;
  store.ensureNode("meme_p_target");
  store.ensureNode("meme_p_other");
  for (const char *w : {"tealridge08", "river", "survey", "marker", "channel"})
    store.bind(w, "meme_p_target");
  for (const char *w : {"bakery", "festival", "harbor", "cakes", "lunar"})
    store.bind(w, "meme_p_other");
  for (int i = 0; i < 4; ++i) {
    const std::string id = "meme_p_decoy" + std::to_string(i);
    store.ensureNode(id);
    store.bind(std::string("decoy") + char('w' + i) + "word", id);
  }
  const std::string exclusive =
      "Tealridge08 river survey marker records the south channel depth after "
      "harvest time.";
  const std::string mixed =
      "Tealridge08 bakery festival cakes at the harbor lunar greenhouse after "
      "harvest season were sold beside the river survey marker channel.";
  const std::string corpus = exclusive + " " + mixed;
  const auto rag = composeCarrierRag(store, "meme_p_target", corpus, 8, 2);
  EXPECT_FALSE(rag.text.empty());
  EXPECT_NE(rag.text.find("Tealridge08"), std::string::npos);
  EXPECT_EQ(rag.text.find("lunar"), std::string::npos);
  EXPECT_LE(rag.otherPresent, 0);
  EXPECT_LE(static_cast<int>(rag.unitIndex.size()), 1);
  EXPECT_FALSE(rag.sources.empty());
  EXPECT_TRUE(detectMemeInText(store, rag.text, "meme_p_target").present);
}

TEST(MemeticExistenceTest, ExpressActivatedIsNotTheExistenceSurface) {
  InMemoryStore store;
  ingestDocumentLike(store, {"south", "river", "survey", "marker", "tealridge08"},
                     {});
  const std::string forced = expressActivated(
      store, {"south", "river", "survey", "marker", "tealridge08"}, {}, 0,
      "tealridge08");
  EXPECT_NE(forced.find("tealridge08"), std::string::npos);
  EXPECT_NE(forced.find("noted "), std::string::npos);
  const std::string carrier = carrierOfMeme(store, store.nodeIds().front());
  EXPECT_EQ(carrier.find("noted "), std::string::npos);
}

TEST(MemeticExistenceTest, MappingDegreeSeedIsInverseDf) {
  EXPECT_DOUBLE_EQ(phoenix::memetic::mappingDegreeSeed(1.0, 1), 1.0);
  EXPECT_DOUBLE_EQ(phoenix::memetic::mappingDegreeSeed(1.0, 400), 1.0 / 400.0);
  EXPECT_DOUBLE_EQ(phoenix::memetic::mappingDegreeSeed(2.0, 0), 0.0);
  EXPECT_EQ(phoenix::memetic::mappingHighDfCut(497), 23);
  EXPECT_TRUE(phoenix::memetic::mappingHighDf(200, 497));
  EXPECT_TRUE(phoenix::memetic::mappingHighDf(24, 497));
  EXPECT_FALSE(phoenix::memetic::mappingHighDf(23, 497));
  EXPECT_FALSE(phoenix::memetic::mappingHighDf(1, 497));
}

TEST(MemeticExistenceTest, ConditionedSeedPullsSharedWordToNamedMeme) {
  using phoenix::memetic::MappingWordHit;
  using phoenix::memetic::mappingAddConditionedSeeds;
  std::unordered_map<std::string, double> q{{"used", 1.0}, {"sakimoto", 1.0}};
  std::unordered_map<std::string, std::vector<MappingWordHit>> hits;
  hits["used"] = {{"meme_p_a", 0.1}, {"meme_p_b", 0.1}, {"meme_p_c", 0.1}};
  hits["sakimoto"] = {{"meme_p_a", 1.0}};
  std::unordered_map<std::string, double> seeds;
  phoenix::memetic::MappingMassFrom from;
  mappingAddConditionedSeeds(q, hits, &seeds, &from, 761);
  EXPECT_GT(seeds["meme_p_a"], seeds["meme_p_b"]);
  EXPECT_GT(seeds["meme_p_a"], seeds["meme_p_c"]);
  EXPECT_FALSE(from["meme_p_a"].empty());
  EXPECT_NE(phoenix::memetic::mappingFormatFrom("meme_p_a", from).find("sakimoto"),
            std::string::npos);
}

TEST(MemeticExistenceTest, IdfScaleShrinksHubWordBudget) {
  EXPECT_DOUBLE_EQ(phoenix::memetic::mappingIdfScale(1, 761), 1.0);
  EXPECT_LT(phoenix::memetic::mappingIdfScale(400, 761), 0.25);
  EXPECT_GT(phoenix::memetic::mappingIdfScale(2, 761),
            phoenix::memetic::mappingIdfScale(400, 761));
}

TEST(MemeticExistenceTest, CollectExistingKeepsAllSupersetMemes) {
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
  const auto hits = phoenix::memetic::mappingCollectExistingMemes(
      {"naval", "duties"}, memesOf, wordsOf, 2, 20, true);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_NE(std::find(hits.begin(), hits.end(), "meme_p_a"), hits.end());
  EXPECT_NE(std::find(hits.begin(), hits.end(), "meme_p_b"), hits.end());
}

TEST(MemeticExistenceTest, RareTypicalDropsShortHubKeepsName) {
  std::unordered_map<std::string, int> df{{"green", 40},
                                         {"captain", 8},
                                         {"lockman", 1},
                                         {"the", 400}};
  const auto w = phoenix::memetic::mappingMemeWeights(
      {"green", "captain", "lockman", "the"}, {1, 1, 1, 1}, df);
  const auto rare = phoenix::memetic::mappingRareTypical(w, df, 761);
  ASSERT_FALSE(rare.empty());
  EXPECT_NE(std::find(rare.begin(), rare.end(), "lockman"), rare.end());
  EXPECT_EQ(std::find(rare.begin(), rare.end(), "green"), rare.end());
}

TEST(MemeticExistenceTest, AlphaIdentityMatchesTypicalNamesNotGenre) {
  std::unordered_map<std::string, int> df{{"the", 400},
                                         {"opening", 80},
                                         {"theme", 90},
                                         {"faylan", 2},
                                         {"tomoshibi", 2},
                                         {"akari", 2},
                                         {"singer", 40}};
  const auto frozen = phoenix::memetic::mappingMemeWeights(
      {"opening", "theme", "akari", "tomoshibi", "faylan", "the"},
      {1, 1, 1, 1, 1, 1}, df);
  const auto same = phoenix::memetic::mappingMemeWeights(
      {"faylan", "sang", "akari", "tomoshibi"}, {1, 1, 1, 1}, df);
  const auto genre = phoenix::memetic::mappingMemeWeights(
      {"opening", "theme", "the", "singer"}, {1, 1, 1, 1}, df);
  EXPECT_TRUE(phoenix::memetic::mappingAlphaIdentity(frozen, same).same);
  EXPECT_FALSE(phoenix::memetic::mappingAlphaIdentity(frozen, genre).same);
}

TEST(MemeticExistenceTest, ContentSpecPrefersRareNamesOverStops) {
  std::unordered_map<std::string, int> df{{"the", 400}, {"was", 200},
                                         {"sakimoto", 2}, {"valkyria", 3}};
  const auto stops = phoenix::memetic::mappingMemeWeights(
      {"the", "was"}, {1, 1}, df);
  const auto names = phoenix::memetic::mappingMemeWeights(
      {"sakimoto", "valkyria"}, {1, 1}, df);
  EXPECT_GT(phoenix::memetic::mappingContentSpec(names, df, 761),
            phoenix::memetic::mappingContentSpec(stops, df, 761));
  EXPECT_LT(phoenix::memetic::mappingContentSpec(stops, df, 761), 0.2);
}

TEST(MemeticExistenceTest, ConcentrateDropsThinHubTail) {
  std::unordered_map<std::string, double> seeds;
  seeds["meme_peak"] = 1.9;
  for (int i = 0; i < 200; ++i)
    seeds["meme_hub_" + std::to_string(i)] = 0.002;
  std::vector<std::string> order;
  order.push_back("meme_peak");
  for (int i = 0; i < 200; ++i)
    order.push_back("meme_hub_" + std::to_string(i));
  phoenix::memetic::mappingConcentrateSeeds(&seeds, &order);
  EXPECT_TRUE(seeds.count("meme_peak"));
  EXPECT_LT(seeds.size(), 8u);
  EXPECT_EQ(order.size(), seeds.size());
}

TEST(MemeticExistenceTest, DetectRejectsStopwordOnlyHubLighting) {
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
  ASSERT_EQ(store.nodeCount(), 6u);
  ASSERT_FALSE(phoenix::memetic::highDfPriorTokens(store).empty());
  const auto own = detectMemeInText(store, "river marker survey", ids[0]);
  EXPECT_TRUE(own.present);
  EXPECT_GE(own.lift, phoenix::memetic::kGnnPresentLift);
  EXPECT_GE(own.orderLift, 1.0);
  EXPECT_FALSE(own.nearestWord.empty());
  EXPECT_EQ(own.nearestSentence, ids[0]);
  const auto stops = detectMemeInText(
      store, "the of and to a in on for is was with as by at from that this",
      ids[0]);
  EXPECT_FALSE(stops.present);
  const auto miss = detectMemeInText(
      store, "quantum chess tournament pairing sheet omega7", ids[0]);
  EXPECT_FALSE(miss.present);
}

TEST(MemeticExistenceTest, HarmlessNonceIsNotAnInstruction) {
  const std::string n(kHarmlessNonce);
  EXPECT_EQ(n.find("http"), std::string::npos);
  EXPECT_EQ(n.find("ignore"), std::string::npos);
  EXPECT_EQ(n.find("rm "), std::string::npos);
  EXPECT_EQ(n.find("#!/"), std::string::npos);
  EXPECT_EQ(n.find(';'), std::string::npos);
  EXPECT_EQ(n.find('('), std::string::npos);
  EXPECT_TRUE(n.find("celadon") != std::string::npos);
}
