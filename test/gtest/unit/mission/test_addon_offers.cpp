#include <gtest/gtest.h>

#include "addon.hpp"
#include "addons/math_addon.hpp"
#include "addons/search_addon.hpp"

using json = nlohmann::json;

namespace {

std::string stationGoal() {
  return "Task Name\n"
         "Design the Surface Station Operations Software\n\n"
         "It is 2035. A Mars science station has successfully landed.\n"
         "Due to communications delays of 8-40 minutes, operate 1000 sols.\n"
         "The station must schedule energy, recover from faults, and "
         "plan scientific observations without a real-time uplink.\n\n"
         "Chapter 1: Mission and Requirements Analysis\n";
}

} // namespace

TEST(AddonOffers, SearchSelfSelectsOnMissionGoalMathStaysSilent) {
  auto mgr = addon::createDefaultAddons();
  ASSERT_TRUE(mgr);
  json sit{{"phase", "mission-deliberate"},
           {"goal", stationGoal()},
           {"draft", ""},
           {"allowWeb", false}};
  const auto offers = mgr->collectOffers(sit, 0.35f, 6);
  bool sawSearch = false;
  bool sawMath = false;
  bool sawComputer = false;
  for (const auto &o : offers) {
    if (o.type == "search") {
      sawSearch = true;
      EXPECT_GE(o.score, 0.35f);
      EXPECT_TRUE(o.result.handled);
      EXPECT_FALSE(o.result.reply.empty());
      EXPECT_TRUE(o.result.units.is_array());
      EXPECT_FALSE(o.result.units.empty());
      EXPECT_EQ(o.result.units[0].value("modality", std::string()), "text");
      EXPECT_FALSE(o.result.units[0].value("content", std::string()).empty());
    }
    if (o.type == "math") sawMath = true;
    if (o.type == "computer") sawComputer = true;
  }
  EXPECT_TRUE(sawSearch);
  EXPECT_FALSE(sawMath);
  EXPECT_FALSE(sawComputer);
}

TEST(AddonOffers, MathSelfSelectsOnlyWhenAsked) {
  auto math = addon::builtins::createMathAddon("math");
  ASSERT_TRUE(math);
  json quiet{{"phase", "mission-deliberate"},
             {"goal", stationGoal()},
             {"text", ""}};
  EXPECT_LT(math->consider(quiet), 0.35f);
  json ask{{"phase", "mission-deliberate"},
           {"goal", stationGoal()},
           {"text", "please calculate 12*8 for the power budget"}};
  EXPECT_GE(math->consider(ask), 0.8f);
  const auto res = math->contribute(ask);
  EXPECT_TRUE(res.handled);
  EXPECT_NE(res.reply.find("96"), std::string::npos);
}

TEST(AddonOffers, HotUnplugDropsSearchFromOffers) {
  auto mgr = addon::createDefaultAddons();
  std::string err;
  ASSERT_TRUE(mgr->removeAddon("search", &err)) << err;
  json sit{{"phase", "mission-deliberate"},
           {"goal", stationGoal()},
           {"allowWeb", false}};
  const auto offers = mgr->collectOffers(sit, 0.35f, 6);
  for (const auto &o : offers) {
    EXPECT_NE(o.type, "search");
  }
  ASSERT_TRUE(mgr->addBuiltin("search", "search", &err)) << err;
  const auto again = mgr->collectOffers(sit, 0.35f, 6);
  bool saw = false;
  for (const auto &o : again) {
    if (o.type == "search") saw = true;
  }
  EXPECT_TRUE(saw);
}

TEST(AddonOffers, ListMarksSelfOffer) {
  auto mgr = addon::createDefaultAddons();
  const json listed = mgr->listAddons();
  ASSERT_TRUE(listed.is_array());
  ASSERT_FALSE(listed.empty());
  EXPECT_TRUE(listed[0].value("selfOffer", false));
  EXPECT_GE(listed[0].value("apiVersion", 0), 2);
}
