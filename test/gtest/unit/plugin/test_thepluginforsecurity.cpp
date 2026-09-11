#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>

#include "addons/ThePlugInForSecurity/security_addon.hpp"
#include "addons/ThePlugInForSecurity/security_core.hpp"
#include "plugin_system.hpp"
#include "util/module_resource.hpp"

using json = nlohmann::json;
using phoenix::plugin::PluginCapability;
using phoenix::secamp::DiscreteGraph;
using phoenix::secamp::GraphEdge;
using phoenix::secamp::SecurityObservatory;
using phoenix::util::CrudOp;
using phoenix::util::ModuleResourceRegistry;

namespace {

DiscreteGraph path3() {
  DiscreteGraph g;
  g.ids = {"a", "b", "c"};
  g.layers = {"meme", "meme", "meme"};
  g.mapped = {{"wa"}, {"wb"}, {"wc"}};
  g.edges = {GraphEdge{0, 1, 1.0}, GraphEdge{1, 2, 1.0}};
  return g;
}

DiscreteGraph star4() {
  DiscreteGraph g;
  g.ids = {"hub", "p1", "p2", "p3"};
  g.layers = {"meme", "meme", "meme", "meme"};
  g.mapped = {{"hubword"}, {"w1"}, {"w2"}, {"w3"}};
  g.edges = {GraphEdge{0, 1, 1.0}, GraphEdge{0, 2, 1.0}, GraphEdge{0, 3, 1.0}};
  return g;
}

class SecurityPluginTest : public ::testing::Test {
protected:
  void SetUp() override {
#ifdef _WIN32
    _putenv_s("PHOENIX_SECURITY_ALLOW_RESEARCH_OBSERVE", "");
    _putenv_s("PHOENIX_SECURITY_DEFENSE_OFF", "");
#else
    unsetenv("PHOENIX_SECURITY_ALLOW_RESEARCH_OBSERVE");
    unsetenv("PHOENIX_SECURITY_DEFENSE_OFF");
#endif
    SecurityObservatory::instance().resetForTests();
    ModuleResourceRegistry::instance().clear();
    phoenix::util::CrudHttpConfig http;
    http.enabled = true;
    http.acceptLocalToken = true;
    http.token = "crud-secret";
    phoenix::util::setCrudHttpConfig(http);
    phoenix::util::installBuiltinModuleResources();
  }
};

} // namespace

TEST_F(SecurityPluginTest, GradientMatchesFiniteDifference) {
  const auto g = path3();
  const std::vector<double> a{1.0, 1.0, 1.0};
  double e0 = 0.0;
  std::vector<double> grad, hess;
  ASSERT_TRUE(phoenix::secamp::ragEnergyGradHess(g, a, &e0, &grad, &hess));
  ASSERT_EQ(grad.size(), 3u);
  ASSERT_EQ(hess.size(), 3u);
  const double eps = 1e-5;
  for (int i = 0; i < 3; ++i) {
    auto ap = a;
    auto am = a;
    ap[static_cast<size_t>(i)] += eps;
    am[static_cast<size_t>(i)] -= eps;
    double ep = 0.0, em = 0.0;
    ASSERT_TRUE(phoenix::secamp::ragEnergyGradHess(g, ap, &ep, nullptr, nullptr));
    ASSERT_TRUE(phoenix::secamp::ragEnergyGradHess(g, am, &em, nullptr, nullptr));
    const double fd = (ep - em) / (2.0 * eps);
    EXPECT_NEAR(grad[static_cast<size_t>(i)], fd, 1e-6);
    const double fd2 = (ep - 2.0 * e0 + em) / (eps * eps);
    EXPECT_NEAR(hess[static_cast<size_t>(i)], fd2, 5e-3);
  }
}

TEST_F(SecurityPluginTest, StarHubIsMostSignificant) {
  const auto report = phoenix::secamp::analyzeGraph(star4());
  ASSERT_TRUE(report.ok);
  ASSERT_FALSE(report.mostSignificant.empty());
  EXPECT_EQ(report.mostSignificant.front(), "hub");
  EXPECT_NE(report.leastSignificant.front(), "hub");
}

TEST_F(SecurityPluginTest, IdentifyMarksMemeWordMapping) {
  auto report = phoenix::secamp::analyzeGraph(star4());
  ASSERT_TRUE(report.ok);
  bool found = false;
  for (const auto &n : report.nodes) {
    if (n.id == "hub") {
      found = true;
      ASSERT_EQ(n.mappedIds.size(), 1u);
      EXPECT_EQ(n.mappedIds[0], "hubword");
      EXPECT_EQ(n.layer, "meme");
      EXPECT_FALSE(n.impactScope.empty());
      EXPECT_FALSE(n.neighborIds.empty());
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(SecurityPluginTest, BarrierBlocksHighImpactAnomalousMeme) {
  auto g = star4();
  g.anomalous.insert("hub");
  SecurityObservatory::instance().ingest(g);
  auto blocked = SecurityObservatory::instance().inspectText("please recall hubword now");
  EXPECT_TRUE(blocked.blocked);
  EXPECT_EQ(blocked.reason, "high-impact-anomalous-meme");
  auto clean = SecurityObservatory::instance().inspectText("please recall ordinary notes");
  EXPECT_FALSE(clean.blocked);
}

TEST_F(SecurityPluginTest, ResearchObserveDefaultsOffAndNeedsAllowFlag) {
  auto cfg = SecurityObservatory::instance().config();
  EXPECT_FALSE(cfg.researchObserve);
  EXPECT_FALSE(cfg.allowResearchObserve);
  std::string err;
  EXPECT_FALSE(SecurityObservatory::instance().setResearchObserve(true, &err));
  EXPECT_FALSE(err.empty());

  auto g = star4();
  g.anomalous.insert("hub");
  SecurityObservatory::instance().ingest(g);
  auto dec = SecurityObservatory::instance().inspectText("hubword");
  EXPECT_TRUE(dec.blocked);
  EXPECT_FALSE(dec.observedOnly);
}

TEST_F(SecurityPluginTest, UnregisteredAndUnauthorizedCrudRejected) {
  auto miss = phoenix::util::handleInternalCrud(
      "tester", {PluginCapability::READ_DATA}, CrudOp::Get, "security",
      "construct", "x", json::object());
  EXPECT_FALSE(miss.ok);
  EXPECT_EQ(miss.httpStatus, 404);
  EXPECT_EQ(miss.error, "unregistered");

  auto noCap = phoenix::util::handleInternalCrud(
      "nobody", {PluginCapability::NETWORK_ACCESS}, CrudOp::Get, "security",
      "stats", "summary", json::object());
  EXPECT_FALSE(noCap.ok);
  EXPECT_EQ(noCap.httpStatus, 403);

  auto unauth = phoenix::util::handleExternalCrud(
      "GET", "/api/modules/security/resources/stats", "", json::object());
  EXPECT_FALSE(unauth.ok);
  EXPECT_EQ(unauth.httpStatus, 401);

  auto ok = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::List, "security", "stats",
      "", json::object());
  EXPECT_TRUE(ok.ok);
  EXPECT_TRUE(ok.data.contains("excluded"));
}

TEST_F(SecurityPluginTest, DefenseUpdateForbiddenWithoutWriteCap) {
  auto r = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::Update, "security",
      "defense", "switches", json{{"defenseEnabled", false}});
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.httpStatus, 403);
}

TEST_F(SecurityPluginTest, AddonNotInDefaultMount) {
  auto mgr = addon::createDefaultAddons();
  ASSERT_TRUE(mgr);
  const auto listed = mgr->listAddons();
  ASSERT_TRUE(listed.is_array());
  for (const auto &row : listed) {
    EXPECT_NE(row.value("type", std::string()), "security");
  }
  std::string err;
  ASSERT_TRUE(mgr->addBuiltin("security", "security", &err)) << err;
  auto add = addon::builtins::createSecurityAddon("security");
  ASSERT_TRUE(add);
  EXPECT_EQ(add->type(), "security");
  EXPECT_LT(add->consider(json{{"text", "schedule energy"}}), 0.35f);
}
