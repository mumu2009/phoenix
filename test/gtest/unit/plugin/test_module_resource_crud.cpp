#include <gtest/gtest.h>

#include "plugin_system.hpp"
#include "util/module_resource.hpp"

using json = nlohmann::json;
using phoenix::plugin::PluginCapability;
using phoenix::plugin::PluginManager;
using phoenix::plugin::PluginMetadata;
using phoenix::plugin::PluginState;
using phoenix::util::CrudCall;
using phoenix::util::CrudChannel;
using phoenix::util::CrudOp;
using phoenix::util::CrudReply;
using phoenix::util::ModuleResourceRegistry;
using phoenix::util::ResourceAcl;
using phoenix::util::ResourceSpec;

namespace {

class DummyPlugin : public phoenix::plugin::Plugin {
public:
  explicit DummyPlugin(std::string name, std::vector<PluginCapability> caps)
      : name_(std::move(name)), caps_(std::move(caps)) {}

  bool onLoad(const PluginMetadata &) override { return true; }
  bool onInit() override { return true; }
  bool onStart() override { return true; }
  bool onStop() override { return true; }
  bool onUnload() override { return true; }
  bool hasCapability(PluginCapability cap) const override {
    for (auto c : caps_) {
      if (c == cap || c == PluginCapability::FULL_ACCESS)
        return true;
    }
    return false;
  }
  std::vector<PluginCapability> getCapabilities() const override { return caps_; }
  std::string name() const override { return name_; }
  std::string version() const override { return "1.0.0"; }
  PluginMetadata getMetadata() const override {
    PluginMetadata m;
    m.name = name_;
    m.version = "1.0.0";
    m.capabilities = caps_;
    return m;
  }
  PluginState getState() const override { return PluginState::RUNNING; }

private:
  std::string name_;
  std::vector<PluginCapability> caps_;
};

class ModuleResourceCrud : public ::testing::Test {
protected:
  void SetUp() override {
    phoenix::util::sessionMetaStore().clear();
    phoenix::util::pluginCatalogStore().clear();
    phoenix::util::addonCatalogStore().clear();
    ModuleResourceRegistry::instance().clear();
    phoenix::util::setPublicConfig(json{{"gatewayHost", "127.0.0.1"},
                                        {"gatewayPort", "5080"}});
    phoenix::util::CrudHttpConfig http;
    http.enabled = true;
    http.acceptLocalToken = true;
    http.token = "crud-secret";
    phoenix::util::setCrudHttpConfig(http);
    phoenix::util::installBuiltinModuleResources();
  }
};

} // namespace

TEST_F(ModuleResourceCrud, BuiltinsAreRegistered) {
  auto &reg = ModuleResourceRegistry::instance();
  EXPECT_TRUE(reg.isRegistered("session", "meta"));
  EXPECT_TRUE(reg.isRegistered("plugin", "catalog"));
  EXPECT_TRUE(reg.isRegistered("addon", "mounted"));
  EXPECT_TRUE(reg.isRegistered("config", "public"));
  EXPECT_TRUE(reg.isRegistered("module_mount", "factories"));
}

TEST_F(ModuleResourceCrud, UnregisteredRejected) {
  auto r = phoenix::util::handleInternalCrud(
      "tester", {PluginCapability::READ_DATA}, CrudOp::Get, "nope", "ghost",
      "1", json::object());
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.httpStatus, 404);
  EXPECT_EQ(r.error, "unregistered");
}

TEST_F(ModuleResourceCrud, InternalSessionMetaRoundTrip) {
  json body{{"id", "s1"}, {"label", "demo"}};
  auto created = phoenix::util::handleInternalCrud(
      "annotator", {PluginCapability::WRITE_DATA}, CrudOp::Create, "session",
      "meta", "s1", body);
  EXPECT_TRUE(created.ok);
  EXPECT_EQ(created.httpStatus, 201);
  EXPECT_EQ(created.data.value("label", std::string()), "demo");

  auto got = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::Get, "session", "meta",
      "s1", json::object());
  EXPECT_TRUE(got.ok);
  EXPECT_EQ(got.data.value("id", std::string()), "s1");

  auto listed = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::List, "session", "meta",
      "", json::object());
  EXPECT_TRUE(listed.ok);
  ASSERT_TRUE(listed.data["items"].is_array());
  EXPECT_GE(listed.data["items"].size(), 1u);
}

TEST_F(ModuleResourceCrud, InternalReadDeniedWithoutCapability) {
  phoenix::util::handleInternalCrud("annotator", {PluginCapability::WRITE_DATA},
                                    CrudOp::Create, "session", "meta", "s2",
                                    json{{"label", "x"}});
  auto r = phoenix::util::handleInternalCrud(
      "no-read", {PluginCapability::NETWORK_ACCESS}, CrudOp::Get, "session",
      "meta", "s2", json::object());
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.httpStatus, 403);
  EXPECT_EQ(r.error, "forbidden");
  EXPECT_EQ(r.data.value("reason", std::string()), "capability_denied");
}

TEST_F(ModuleResourceCrud, InternalWriteDeniedOnReadOnlyResource) {
  auto r = phoenix::util::handleInternalCrud(
      "writer", {PluginCapability::WRITE_DATA}, CrudOp::Update, "config",
      "public", "gatewayHost", json{{"value", "0.0.0.0"}});
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.httpStatus, 405);
  EXPECT_EQ(r.error, "method_not_allowed");
}

TEST_F(ModuleResourceCrud, ActorAllowListRejects) {
  ResourceAcl acl;
  acl.allowInternal = true;
  acl.allowInternalWrite = false;
  acl.allowedActors = {"trusted"};
  ModuleResourceRegistry::instance().registerResource(ResourceSpec{
      "lab", "cell", "allowlist demo", acl, [](const CrudCall &call) {
        return CrudReply{true, 200, "", json{{"id", call.id}}};
      }});
  auto bad = phoenix::util::handleInternalCrud(
      "stranger", {PluginCapability::READ_DATA}, CrudOp::Get, "lab", "cell",
      "c1", json::object());
  EXPECT_FALSE(bad.ok);
  EXPECT_EQ(bad.data.value("reason", std::string()), "actor_denied");
  auto good = phoenix::util::handleInternalCrud(
      "trusted", {PluginCapability::READ_DATA}, CrudOp::Get, "lab", "cell",
      "c1", json::object());
  EXPECT_TRUE(good.ok);
}

TEST_F(ModuleResourceCrud, ExternalHandlerAuthAndPath) {
  auto unauth = phoenix::util::handleExternalCrud(
      "GET", "/api/modules/config/resources/public", "", json::object());
  EXPECT_FALSE(unauth.ok);
  EXPECT_EQ(unauth.httpStatus, 401);
  EXPECT_EQ(unauth.error, "unauthorized");

  auto listed = phoenix::util::handleExternalCrud(
      "GET", "/api/modules/config/resources/public", "Bearer local-dev",
      json::object());
  EXPECT_TRUE(listed.ok);
  EXPECT_EQ(listed.httpStatus, 200);
  ASSERT_TRUE(listed.data["items"].is_array());

  auto one = phoenix::util::handleExternalCrud(
      "GET", "/api/modules/config/resources/public/gatewayHost",
      "Bearer crud-secret", json::object());
  EXPECT_TRUE(one.ok);
  EXPECT_EQ(one.data.value("id", std::string()), "gatewayHost");

  auto badPath = phoenix::util::handleExternalCrud(
      "GET", "/api/not-modules/x", "Bearer local-dev", json::object());
  EXPECT_FALSE(badPath.ok);
  EXPECT_EQ(badPath.error, "unregistered");
}

TEST_F(ModuleResourceCrud, ExternalWriteOnSessionRejected) {
  auto r = phoenix::util::handleExternalCrud(
      "POST", "/api/modules/session/resources/meta/s9", "Bearer local-dev",
      json{{"label", "nope"}});
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.httpStatus, 405);
}

TEST_F(ModuleResourceCrud, PluginManagerInternalCrudUsesCaps) {
  PluginManager::Config cfg;
  cfg.autoLoad = false;
  cfg.hotReload = false;
  cfg.maxPlugins = 8;
  PluginManager mgr(cfg);
  auto reader = std::make_shared<DummyPlugin>(
      "reader", std::vector<PluginCapability>{PluginCapability::READ_DATA});
  auto writer = std::make_shared<DummyPlugin>(
      "writer", std::vector<PluginCapability>{PluginCapability::WRITE_DATA});
  ASSERT_TRUE(mgr.registerPlugin(reader));
  ASSERT_TRUE(mgr.registerPlugin(writer));

  auto w = mgr.crudWrite("writer", "session", "meta", "from-mgr",
                         json{{"label", "via-manager"}}, true);
  EXPECT_TRUE(w.success);
  auto g = mgr.crudGet("reader", "session", "meta", "from-mgr");
  EXPECT_TRUE(g.success);
  EXPECT_EQ(g.data.value("label", std::string()), "via-manager");
  auto denied = mgr.crudWrite("reader", "session", "meta", "from-mgr",
                              json{{"label", "x"}}, false);
  EXPECT_FALSE(denied.success);

  auto cat = mgr.crudList("reader", "plugin", "catalog");
  EXPECT_TRUE(cat.success);
}

TEST_F(ModuleResourceCrud, PublicConfigAndMountFactoriesReadable) {
  auto cfg = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::Get, "config", "public",
      "gatewayPort", json::object());
  EXPECT_TRUE(cfg.ok);
  EXPECT_EQ(cfg.data.value("value", std::string()), "5080");

  auto fac = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::List, "module_mount",
      "factories", "", json::object());
  EXPECT_TRUE(fac.ok);
  ASSERT_TRUE(fac.data["items"].is_array());
  EXPECT_FALSE(fac.data["items"].empty());
}

TEST_F(ModuleResourceCrud, AddonPublishVisibleToCrud) {
  phoenix::util::addonCatalogStore().put(
      "math", json{{"name", "math"}, {"type", "math"}, {"source", "builtin"}});
  auto r = phoenix::util::handleInternalCrud(
      "reader", {PluginCapability::READ_DATA}, CrudOp::Get, "addon", "mounted",
      "math", json::object());
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.data.value("type", std::string()), "math");
}
