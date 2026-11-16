/* v10.2 expert summoning: scored selection instead of "first row wins".

   Pinned behaviour:
     - a matching role is required (a non-programming endpoint is never
       summonsed for a source edit);
     - a proven endpoint outranks an unproven one (Laplace prior 0.5);
     - an endpoint with consecutive failures is deprioritised, and two
       failures outweigh four in-flight calls;
     - load spreads across equally good endpoints;
     - the just-used endpoint gets a small sticky bonus (no thrash). */
#include <gtest/gtest.h>

#include "summon_registry.hpp"

using phoenix::summon::EndpointRanker;
using phoenix::summon::EndpointStats;
using phoenix::summon::ModelEndpoint;

namespace {

ModelEndpoint makeEndpoint(const std::string &name, bool programming) {
  ModelEndpoint ep;
  ep.name = name;
  ep.kind = "llamacpp";
  ep.url = "http://127.0.0.1:8093";
  if (programming) ep.roles.push_back("programming");
  return ep;
}

}  // namespace

TEST(SummonSelect, Sample1_RequiresMatchingRole) {
  std::vector<ModelEndpoint> eps{makeEndpoint("chat-only", false),
                                 makeEndpoint("coder", true)};
  EndpointRanker ranker;
  const int idx = ranker.pick(eps, nullptr, "programming", 1000);
  ASSERT_EQ(idx, 1);
  EXPECT_EQ(eps[static_cast<size_t>(idx)].name, "coder");
  /* no programming endpoint at all -> no summon */
  std::vector<ModelEndpoint> noRole{makeEndpoint("chat-only", false)};
  EXPECT_EQ(ranker.pick(noRole, nullptr, "programming", 1000), -1);
}

TEST(SummonSelect, Sample2_ProvenEndpointOutranksUnproven) {
  std::vector<ModelEndpoint> eps{makeEndpoint("fresh", true),
                                 makeEndpoint("proven", true)};
  auto stats = [](const std::string &name) {
    EndpointStats s;
    if (name == "proven") s.successes = 9;
    return s;
  };
  EndpointRanker ranker;
  EXPECT_EQ(eps[static_cast<size_t>(ranker.pick(eps, stats, "programming", 1000))].name,
            "proven");
}

TEST(SummonSelect, Sample3_ConsecutiveFailuresOutweighLoad) {
  std::vector<ModelEndpoint> eps{makeEndpoint("broken", true),
                                 makeEndpoint("busy", true)};
  auto stats = [](const std::string &name) {
    EndpointStats s;
    if (name == "broken") {
      s.consecutiveFailures = 2;
      s.failures = 2;
    } else {
      s.inFlight = 4; /* the busiest still-allowed load */
      s.successes = 4;
    }
    return s;
  };
  EndpointRanker ranker;
  EXPECT_EQ(
      eps[static_cast<size_t>(ranker.pick(eps, stats, "programming", 1000))].name,
      "busy");
}

TEST(SummonSelect, Sample4_LoadSpreadsBetweenEqualEndpoints) {
  std::vector<ModelEndpoint> eps{makeEndpoint("e0", true),
                                 makeEndpoint("e1", true)};
  auto stats = [](const std::string &name) {
    EndpointStats s;
    if (name == "e0") s.inFlight = 2;
    return s;
  };
  EndpointRanker ranker;
  EXPECT_EQ(eps[static_cast<size_t>(ranker.pick(eps, stats, "programming", 1000))].name,
            "e1");
}

TEST(SummonSelect, Sample5_StickyBonusKeepsTheJustUsedEndpoint) {
  std::vector<ModelEndpoint> eps{makeEndpoint("e0", true),
                                 makeEndpoint("e1", true)};
  auto stats = [](const std::string &name) {
    EndpointStats s;
    if (name == "e0") s.lastUsedMs = 10000;
    return s;
  };
  EndpointRanker ranker;
  /* 10.5 s after the last use: still inside the sticky window */
  EXPECT_EQ(
      eps[static_cast<size_t>(ranker.pick(eps, stats, "programming", 20500))].name,
      "e0");
  /* far outside the window the score is identical again; the earlier row
     wins the tie, which is deterministic and therefore safe to assert. */
  EXPECT_EQ(
      eps[static_cast<size_t>(ranker.pick(eps, stats, "programming", 200000))].name,
      "e0");
}

TEST(SummonSelect, Sample6_UnlocatedEndpointIsNeverPicked) {
  ModelEndpoint ghost;
  ghost.name = "ghost";
  ghost.kind = "llamacpp";
  ghost.roles.push_back("programming"); /* no url -> not located */
  std::vector<ModelEndpoint> eps{ghost, makeEndpoint("real", true)};
  EndpointRanker ranker;
  EXPECT_EQ(
      eps[static_cast<size_t>(ranker.pick(eps, nullptr, "programming", 1000))].name,
      "real");
}
