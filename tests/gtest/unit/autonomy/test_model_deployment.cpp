/* test_model_deployment.cpp - GTest for v7.0 model deployment topology
   Copyright (C) 2026 079 Project */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "model_deployment.hpp"

using json = nlohmann::json;

TEST(ModelDeployment, PlacementToString) {
  EXPECT_EQ(phoenix::deployment::placementToString(
                phoenix::deployment::ModelPlacement::Local),
            "local");
  EXPECT_EQ(phoenix::deployment::placementToString(
                phoenix::deployment::ModelPlacement::Remote),
            "remote");
  EXPECT_EQ(phoenix::deployment::placementToString(
                phoenix::deployment::ModelPlacement::Auto),
            "auto");
  EXPECT_EQ(phoenix::deployment::placementToString(
                phoenix::deployment::ModelPlacement::ServerClient),
            "server-client");
}

TEST(ModelDeployment, ParsePlacement) {
  using phoenix::deployment::ModelPlacement;
  EXPECT_EQ(phoenix::deployment::parsePlacement("local"), ModelPlacement::Local);
  EXPECT_EQ(phoenix::deployment::parsePlacement("LOCAL"), ModelPlacement::Local);
  EXPECT_EQ(phoenix::deployment::parsePlacement("remote"), ModelPlacement::Remote);
  EXPECT_EQ(phoenix::deployment::parsePlacement("REMOTE"), ModelPlacement::Remote);
  EXPECT_EQ(phoenix::deployment::parsePlacement("edge"), ModelPlacement::Remote);
  EXPECT_EQ(phoenix::deployment::parsePlacement("on-device"), ModelPlacement::Remote);
  EXPECT_EQ(phoenix::deployment::parsePlacement("auto"), ModelPlacement::Auto);
  EXPECT_EQ(phoenix::deployment::parsePlacement("AUTO"), ModelPlacement::Auto);
  EXPECT_EQ(phoenix::deployment::parsePlacement("automatic"), ModelPlacement::Auto);
  EXPECT_EQ(phoenix::deployment::parsePlacement("server-client"), ModelPlacement::ServerClient);
  EXPECT_EQ(phoenix::deployment::parsePlacement("ServerClient"), ModelPlacement::ServerClient);
  EXPECT_EQ(phoenix::deployment::parsePlacement("web"), ModelPlacement::ServerClient);
  EXPECT_EQ(phoenix::deployment::parsePlacement(""), ModelPlacement::Local);
  EXPECT_EQ(phoenix::deployment::parsePlacement("garbage"), ModelPlacement::Local);
}

TEST(ModelDeployment, ServerClientRecordJsonRoundTrip) {
  phoenix::deployment::ModelDeploymentRecord rec;
  rec.placement = phoenix::deployment::ModelPlacement::ServerClient;

  json j = rec.toJson();
  EXPECT_EQ(j["placement"], "server-client");

  phoenix::deployment::ModelDeploymentRecord restored;
  restored.fromJson(j);
  EXPECT_EQ(restored.placement, rec.placement);
}

TEST(ModelDeployment, RemoteEndpointJsonRoundTrip) {
  phoenix::deployment::RemoteEndpoint ep;
  ep.url = "http://192.168.1.10:5000/infer";
  ep.method = "http-json";
  ep.modelName = "jepa_vith14";
  ep.authToken = "token-123";
  ep.timeoutMs = 12345;
  ep.headers["X-Custom"] = "value";

  json j = ep.toJson();
  EXPECT_EQ(j["url"], ep.url);
  EXPECT_EQ(j["method"], "http-json");

  phoenix::deployment::RemoteEndpoint restored;
  restored.fromJson(j);
  EXPECT_EQ(restored.url, ep.url);
  EXPECT_EQ(restored.method, ep.method);
  EXPECT_EQ(restored.modelName, ep.modelName);
  EXPECT_EQ(restored.authToken, ep.authToken);
  EXPECT_EQ(restored.timeoutMs, ep.timeoutMs);
  ASSERT_EQ(restored.headers.size(), 1u);
  EXPECT_EQ(restored.headers.at("X-Custom"), "value");
}

TEST(ModelDeployment, ModelDeploymentRecordJsonRoundTrip) {
  phoenix::deployment::ModelDeploymentRecord rec;
  rec.placement = phoenix::deployment::ModelPlacement::Remote;
  rec.remote.url = "http://host:8080";
  rec.remote.method = "ollama";

  json j = rec.toJson();
  EXPECT_EQ(j["placement"], "remote");

  phoenix::deployment::ModelDeploymentRecord restored;
  restored.fromJson(j);
  EXPECT_EQ(restored.placement, rec.placement);
  EXPECT_EQ(restored.remote.url, rec.remote.url);
  EXPECT_EQ(restored.remote.method, rec.remote.method);
}

TEST(ModelDeployment, AutoRecordJsonRoundTrip) {
  phoenix::deployment::ModelDeploymentRecord rec;
  rec.placement = phoenix::deployment::ModelPlacement::Auto;
  rec.remote.url = "http://host:5000/infer";
  rec.remote.method = "http-json";

  json j = rec.toJson();
  EXPECT_EQ(j["placement"], "auto");

  phoenix::deployment::ModelDeploymentRecord restored;
  restored.fromJson(j);
  EXPECT_EQ(restored.placement, rec.placement);
  EXPECT_EQ(restored.remote.url, rec.remote.url);
  EXPECT_EQ(restored.remote.method, rec.remote.method);
}

TEST(ModelDeployment, ConfigSingletonReset) {
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();
  EXPECT_EQ(cfg.llm().placement, phoenix::deployment::ModelPlacement::Local);
  EXPECT_EQ(cfg.vision().placement, phoenix::deployment::ModelPlacement::Local);
  EXPECT_EQ(cfg.speech().placement, phoenix::deployment::ModelPlacement::Local);
}

TEST(ModelDeployment, ConfigLoadFromArgs) {
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();

  std::map<std::string, std::string> args;
  args["vision-placement"] = "remote";
  args["vision-remote-url"] = "http://192.168.1.11:5000/infer";
  args["vision-remote-method"] = "http-json";
  args["speech-placement"] = "edge";
  args["speech-remote-url"] = "http://192.168.1.12:5001/infer";

  cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

  EXPECT_EQ(cfg.vision().placement, phoenix::deployment::ModelPlacement::Remote);
  EXPECT_EQ(cfg.vision().remote.url, "http://192.168.1.11:5000/infer");
  EXPECT_EQ(cfg.vision().remote.method, "http-json");

  EXPECT_EQ(cfg.speech().placement, phoenix::deployment::ModelPlacement::Remote);
  EXPECT_EQ(cfg.speech().remote.url, "http://192.168.1.12:5001/infer");

  EXPECT_EQ(cfg.llm().placement, phoenix::deployment::ModelPlacement::Local);
  cfg.reset();
}

TEST(ModelDeployment, ConfigLoadFromArgsAuto) {
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();

  std::map<std::string, std::string> args;
  args["llm-placement"] = "auto";
  args["llm-remote-url"] = "http://192.168.1.10:11434";
  args["llm-remote-method"] = "ollama";
  args["llm-remote-model"] = "llama3.1:8b";

  cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

  EXPECT_EQ(cfg.llm().placement, phoenix::deployment::ModelPlacement::Auto);
  EXPECT_EQ(cfg.llm().remote.url, "http://192.168.1.10:11434");
  EXPECT_EQ(cfg.llm().remote.method, "ollama");
  EXPECT_EQ(cfg.llm().remote.modelName, "llama3.1:8b");

  json j = cfg.toJson();
  EXPECT_EQ(j["llm"]["placement"], "auto");
  EXPECT_EQ(j["llm"]["remote"]["url"], "http://192.168.1.10:11434");

  cfg.reset();
}

TEST(ModelDeployment, Base64RoundTrip) {
  const std::vector<uint8_t> input = {0x00, 0x01, 0x02, 0xff, 0x7f};
  std::string encoded =
      phoenix::deployment::RemoteModelClient::base64Encode(input);
  EXPECT_FALSE(encoded.empty());
  auto decoded =
      phoenix::deployment::RemoteModelClient::base64Decode(encoded);
  EXPECT_EQ(decoded, input);
}

TEST(ModelDeployment, Base64Empty) {
  auto encoded =
      phoenix::deployment::RemoteModelClient::base64Encode({});
  EXPECT_TRUE(encoded.empty());
  auto decoded =
      phoenix::deployment::RemoteModelClient::base64Decode(encoded);
  EXPECT_TRUE(decoded.empty());
}

TEST(ModelDeployment, Base64Padding1) {
  // Input length not multiple of 3, exercise single-pad branch.
  const std::vector<uint8_t> input = {'a', 'b'};
  std::string encoded =
      phoenix::deployment::RemoteModelClient::base64Encode(input);
  EXPECT_FALSE(encoded.empty());
  auto decoded =
      phoenix::deployment::RemoteModelClient::base64Decode(encoded);
  EXPECT_EQ(decoded, input);
}

TEST(ModelDeployment, Base64Padding2) {
  // Input length % 3 == 1, exercise double-pad branch.
  const std::vector<uint8_t> input = {'a'};
  std::string encoded =
      phoenix::deployment::RemoteModelClient::base64Encode(input);
  EXPECT_FALSE(encoded.empty());
  auto decoded =
      phoenix::deployment::RemoteModelClient::base64Decode(encoded);
  EXPECT_EQ(decoded, input);
}

TEST(ModelDeployment, Base64DecodeIgnoresWhitespace) {
  const std::vector<uint8_t> input = {0x00, 0x01, 0x02};
  std::string encoded =
      phoenix::deployment::RemoteModelClient::base64Encode(input);
  encoded.insert(2, " \n\t");
  auto decoded =
      phoenix::deployment::RemoteModelClient::base64Decode(encoded);
  EXPECT_EQ(decoded, input);
}

TEST(ModelDeployment, Base64DecodeInvalidReturnsEmpty) {
  auto decoded =
      phoenix::deployment::RemoteModelClient::base64Decode("!@#$");
  EXPECT_TRUE(decoded.empty());
}

TEST(ModelDeployment, ConfigToJsonRoundTrip) {
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();
  std::map<std::string, std::string> args;
  args["llm-placement"] = "remote";
  args["llm-remote-url"] = "http://llm:11434";
  args["llm-remote-method"] = "ollama";
  args["llm-remote-model"] = "llama3.1:8b";
  cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

  json j = cfg.toJson();
  EXPECT_EQ(j["llm"]["placement"], "remote");

  cfg.fromJson(j);
  EXPECT_EQ(cfg.llm().placement, phoenix::deployment::ModelPlacement::Remote);
  EXPECT_EQ(cfg.llm().remote.url, "http://llm:11434");
  EXPECT_EQ(cfg.llm().remote.method, "ollama");
  EXPECT_EQ(cfg.llm().remote.modelName, "llama3.1:8b");

  cfg.reset();
}

TEST(ModelDeployment, LocalBackendToStringAndParse) {
  using phoenix::deployment::LocalBackendType;
  EXPECT_EQ(phoenix::deployment::localBackendTypeToString(LocalBackendType::Cpu), "cpu");
  EXPECT_EQ(phoenix::deployment::localBackendTypeToString(LocalBackendType::Gpu), "gpu");
  EXPECT_EQ(phoenix::deployment::localBackendTypeToString(LocalBackendType::Bpu), "bpu");
  EXPECT_EQ(phoenix::deployment::localBackendTypeToString(LocalBackendType::Js), "js");
  EXPECT_EQ(phoenix::deployment::localBackendTypeToString(LocalBackendType::Auto), "auto");

  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("cpu"), LocalBackendType::Cpu);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("GPU"), LocalBackendType::Gpu);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("bpu"), LocalBackendType::Bpu);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("x5"), LocalBackendType::Bpu);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("js"), LocalBackendType::Js);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("browser"), LocalBackendType::Js);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("onnx"), LocalBackendType::Cpu);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType(""), LocalBackendType::Auto);
  EXPECT_EQ(phoenix::deployment::parseLocalBackendType("garbage"), LocalBackendType::Auto);
}

TEST(ModelDeployment, RecordRoundTripWithLocalBackend) {
  phoenix::deployment::ModelDeploymentRecord rec;
  rec.placement = phoenix::deployment::ModelPlacement::Local;
  rec.localBackend = phoenix::deployment::LocalBackendType::Bpu;

  json j = rec.toJson();
  EXPECT_EQ(j["placement"], "local");
  EXPECT_EQ(j["localBackend"], "bpu");

  phoenix::deployment::ModelDeploymentRecord restored;
  restored.fromJson(j);
  EXPECT_EQ(restored.placement, rec.placement);
  EXPECT_EQ(restored.localBackend, rec.localBackend);
}

TEST(ModelDeployment, ConfigLoadFromArgsLocalBackend) {
  auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
  cfg.reset();

  std::map<std::string, std::string> args;
  args["vision-placement"] = "local";
  args["vision-local-backend"] = "bpu";
  args["speech-placement"] = "local";
  args["speech-local-backend"] = "onnx";

  cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

  EXPECT_EQ(cfg.vision().placement, phoenix::deployment::ModelPlacement::Local);
  EXPECT_EQ(cfg.vision().localBackend, phoenix::deployment::LocalBackendType::Bpu);
  EXPECT_EQ(cfg.speech().placement, phoenix::deployment::ModelPlacement::Local);
  EXPECT_EQ(cfg.speech().localBackend, phoenix::deployment::LocalBackendType::Cpu);

  json j = cfg.toJson();
  EXPECT_EQ(j["vision"]["localBackend"], "bpu");
  EXPECT_EQ(j["speech"]["localBackend"], "cpu");

  cfg.reset();
}

TEST(ModelDeployment, RemoteClientCallEmptyUrl) {
  phoenix::deployment::RemoteEndpoint ep;
  auto result = phoenix::deployment::RemoteModelClient::call(ep, json{});
  EXPECT_FALSE(result.value("ok", true));
  EXPECT_EQ(result.value("error", ""), "remote endpoint URL is empty");
}
