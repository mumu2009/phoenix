/* test_cli_json_addon.cpp - Whitelist gate + JSON output handling. */

#include <gtest/gtest.h>
#include "addons/cli_json_addon.hpp"

using addon::builtins::clearCliJsonRegistry;
using addon::builtins::getCliJsonRegistry;
using addon::builtins::runCliJsonCommand;
using addon::builtins::setCliJsonRegistry;
using addon::json;

namespace {
class CliJsonRegistryTest : public ::testing::Test {
protected:
  void SetUp() override {
    json registry = json::array();
    json echo;
    echo["name"] = "echo";
#ifdef _WIN32
    echo["command"] = "cmd.exe";
    echo["args"] = json::array({"/c", "echo"});
#else
    echo["command"] = "/bin/echo";
#endif
    echo["timeoutMs"] = 8000;
    echo["json"] = false;
    registry.push_back(echo);
    std::string err;
    ASSERT_TRUE(setCliJsonRegistry(registry, &err)) << err;
  }
  void TearDown() override { clearCliJsonRegistry(); }
};
}  // namespace

TEST_F(CliJsonRegistryTest, WhitelistRejectsUnknownTool) {
  json r = runCliJsonCommand("not-whitelisted", {}, json::object());
  EXPECT_FALSE(r.value("ok", true));
  EXPECT_NE(r.value("error", ""), "");
}

TEST_F(CliJsonRegistryTest, WhitelistedCommandRuns) {
  json r = runCliJsonCommand("echo", {"cli-json-smoke"}, json::object());
  ASSERT_TRUE(r.value("ok", false)) << r.dump();
  EXPECT_NE(r.value("reply", "").find("cli-json-smoke"), std::string::npos);
}

TEST_F(CliJsonRegistryTest, RegistryRoundTrip) {
  json reg = getCliJsonRegistry();
  ASSERT_TRUE(reg.is_array());
  EXPECT_EQ(reg.size(), 1u);
  EXPECT_EQ(reg[0].value("name", ""), "echo");
}
