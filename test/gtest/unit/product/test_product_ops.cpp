#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>

#include "product_ops.hpp"

using namespace phoenix::product;

TEST(ProductAuth, HumanMessagesAndLoginPaths)
{
    EXPECT_NE(humanAuthError("Invalid credentials"), "Invalid credentials");
    EXPECT_TRUE(humanAuthError("email not verified").find("邮箱") != std::string::npos);

    auto bad = decideLogin(false, false, false, true);
    EXPECT_FALSE(bad.ok);
    EXPECT_EQ(bad.status, 401);
    EXPECT_FALSE(bad.message.empty());

    auto unverified = decideLogin(true, true, false, true);
    EXPECT_FALSE(unverified.ok);
    EXPECT_EQ(unverified.status, 403);
    EXPECT_EQ(unverified.error, "email not verified");

    auto ok = decideLogin(true, true, false, false);
    EXPECT_TRUE(ok.ok);

    auto regOff = decideRegister(false, "");
    EXPECT_EQ(regOff.status, 403);
    auto regDup = decideRegister(true, "username exists");
    EXPECT_EQ(regDup.status, 400);
    EXPECT_TRUE(regDup.message.find("用户名") != std::string::npos);

    auto boot = decideBootstrap(true, "");
    EXPECT_EQ(boot.status, 409);

    EXPECT_TRUE(roleAllows("admin", "admin"));
    EXPECT_FALSE(roleAllows("user", "admin"));
    auto setRole = decideSetRole("user", "admin");
    EXPECT_FALSE(setRole.ok);
    auto setRoleBad = decideSetRole("admin", "root");
    EXPECT_EQ(setRoleBad.error, "role not allowed");
    EXPECT_TRUE(decideSetRole("admin", "user").ok);
}

TEST(ProductModules, ForbidLlamaAndUnknown)
{
    LogicalModuleBoard board;
    auto listed = board.listJson();
    ASSERT_TRUE(listed.is_array());
    bool sawLlama = false;
    for (const auto &row : listed) {
        if (row.value("id", "") == "inference-llama") {
            sawLlama = true;
            EXPECT_TRUE(row.value("readonly", false));
        }
    }
    EXPECT_TRUE(sawLlama);

    EXPECT_TRUE(moduleIdForbidden("llama-server"));
    EXPECT_TRUE(moduleIdForbidden("supervisor"));
    EXPECT_FALSE(moduleIdForbidden("chat-ui"));

    auto denied = board.setEnabled("inference-llama", false);
    EXPECT_FALSE(denied.ok);
    EXPECT_EQ(denied.status, 403);

    auto unknown = board.setEnabled("not-a-module", false);
    EXPECT_EQ(unknown.status, 404);

    auto ok = board.setEnabled("search", false);
    EXPECT_TRUE(ok.ok);
    EXPECT_FALSE(board.enabled("search"));
}

TEST(ProductMonitorDb, SnapshotAndBackup)
{
    std::string host;
    int port = 0;
    EXPECT_TRUE(parseHostPort("http://127.0.0.1:8082/v1", host, port));
    EXPECT_EQ(host, "127.0.0.1");
    EXPECT_EQ(port, 8082);
    EXPECT_FALSE(parseHostPort("not-a-url", host, port));

    EndpointFact llama{"inference-llama", "推理", "127.0.0.1", 8082, 4242, true, "tcp-port"};
    auto mon = buildMonitorJson({llama}, 8ull * 1024ull * 1024ull, {{"total", 3}}, 99);
    EXPECT_TRUE(mon["ok"].get<bool>());
    EXPECT_EQ(mon["pid"], 99);
    EXPECT_TRUE(mon["endpoints"][0]["healthHttpForbidden"].get<bool>());
    EXPECT_EQ(mon["endpoints"][0]["probe"], "tcp-port");

    RequestMeter meter;
    meter.record(200, 10.0);
    meter.record(500, 30.0);
    auto snap = meter.snapshot();
    EXPECT_EQ(snap["total"], 2);
    EXPECT_EQ(snap["errors"], 1);

    auto tmp = std::filesystem::temp_directory_path() / "phoenix_product_ops_db";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp);
    auto db = tmp / "ai_store.sqlite";
    {
        std::ofstream out(db);
        out << "sqlite-placeholder";
    }
    auto info = inspectDatabase(db, tmp / "lmdb", tmp / "backups");
    EXPECT_TRUE(info.exists);
    auto infoJson = databaseInfoJson(info);
    EXPECT_TRUE(infoJson["healthy"].get<bool>());

    auto dest = suggestedBackupPath(db, tmp / "backups", 123);
    auto copied = copyDatabaseBackup(db, dest);
    EXPECT_TRUE(copied.ok);
    EXPECT_TRUE(std::filesystem::exists(dest));

    auto missing = copyDatabaseBackup(tmp / "nope.sqlite", dest);
    EXPECT_FALSE(missing.ok);
    EXPECT_EQ(missing.error, "db missing");
    std::filesystem::remove_all(tmp, ec);
}
