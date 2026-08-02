/* model_deployment.cpp - Runtime model placement topology for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#include "model_deployment.hpp"
#include "phoenix_config.hpp"

#include <drogon/HttpClient.h>
#include <trantor/utils/Logger.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>

namespace phoenix {
namespace deployment {

namespace fs = std::filesystem;

std::string placementToString(ModelPlacement p) {
  if (p == ModelPlacement::Remote) return "remote";
  if (p == ModelPlacement::Auto) return "auto";
  if (p == ModelPlacement::ServerClient) return "server-client";
  return "local";
}

ModelPlacement parsePlacement(const std::string &s) {
  std::string t;
  t.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c)))
      t.push_back(static_cast<char>(std::tolower(c)));
  }
  if (t == "remote" || t == "edge" || t == "ondevice" || t == "on_device" ||
      t == "device" || t == "external" || t == "offboard") {
    return ModelPlacement::Remote;
  }
  if (t == "auto" || t == "automatic" || t == "default" || t == "best" ||
      t == "adaptive") {
    return ModelPlacement::Auto;
  }
  if (t == "serverclient" || t == "server_client" || t == "server-client" ||
      t == "web" || t == "browser" || t == "client") {
    return ModelPlacement::ServerClient;
  }
  return ModelPlacement::Local;
}

namespace {

std::string trimCopy(const std::string &s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

std::string lowerCopy(const std::string &s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return t;
}

/**
 * @brief Split a URL into a base (scheme://host:port) and a path.
 *
 * Handles trailing paths like http://host:5000/infer correctly so the
 * drogon HttpClient is created with only the base and the request path
 * includes the configured endpoint path.
 */
std::pair<std::string, std::string> splitHttpUrl(const std::string &url) {
  std::string base = url;
  std::string basePath;
  size_t schemeEnd = url.find("://");
  if (schemeEnd != std::string::npos) {
    size_t hostStart = schemeEnd + 3;
    size_t pathStart = url.find('/', hostStart);
    if (pathStart != std::string::npos) {
      base = url.substr(0, pathStart);
      basePath = url.substr(pathStart);
    }
  }
  if (!basePath.empty() && basePath.back() == '/')
    basePath.pop_back();
  return {base, basePath};
}

}  // namespace

nlohmann::json RemoteEndpoint::toJson() const {
  nlohmann::json j;
  j["url"] = url;
  j["method"] = method;
  j["modelName"] = modelName;
  j["authToken"] = authToken;
  j["timeoutMs"] = timeoutMs;
  j["headers"] = nlohmann::json::object();
  for (const auto &kv : headers) j["headers"][kv.first] = kv.second;
  return j;
}

void RemoteEndpoint::fromJson(const nlohmann::json &j) {
  if (!j.is_object()) return;
  url = j.value("url", url);
  method = lowerCopy(j.value("method", method));
  modelName = j.value("modelName", modelName);
  authToken = j.value("authToken", authToken);
  if (j.contains("timeoutMs") && j["timeoutMs"].is_number())
    timeoutMs = std::max(1, j["timeoutMs"].get<int>());
  if (j.contains("headers") && j["headers"].is_object()) {
    for (auto it = j["headers"].begin(); it != j["headers"].end(); ++it)
      headers[it.key()] = it.value().get<std::string>();
  }
}

nlohmann::json ModelDeploymentRecord::toJson() const {
  nlohmann::json j;
  j["placement"] = placementToString(placement);
  j["remote"] = remote.toJson();
  return j;
}

void ModelDeploymentRecord::fromJson(const nlohmann::json &j) {
  if (!j.is_object()) return;
  placement = parsePlacement(j.value("placement", "local"));
  if (j.contains("remote") && j["remote"].is_object())
    remote.fromJson(j["remote"]);
}

ModelDeploymentConfig &ModelDeploymentConfig::instance() {
  static ModelDeploymentConfig inst;
  return inst;
}

void ModelDeploymentConfig::reset() {
  std::lock_guard<std::mutex> lock(mu_);
  llm_ = ModelDeploymentRecord{};
  vision_ = ModelDeploymentRecord{};
  speech_ = ModelDeploymentRecord{};
}

nlohmann::json ModelDeploymentConfig::toJson() const {
  std::lock_guard<std::mutex> lock(mu_);
  nlohmann::json j;
  j["llm"] = llm_.toJson();
  j["vision"] = vision_.toJson();
  j["speech"] = speech_.toJson();
  return j;
}

void ModelDeploymentConfig::fromJson(const nlohmann::json &j) {
  if (!j.is_object()) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (j.contains("llm")) llm_.fromJson(j["llm"]);
  if (j.contains("vision")) vision_.fromJson(j["vision"]);
  if (j.contains("speech")) speech_.fromJson(j["speech"]);
}

namespace {

static std::string dotPathForModelEnv(const std::string &env) {
  if (env == "AI_MODEL_DEPLOYMENT_CONFIG") return "model_deployment.configPath";
  if (env == "AI_LLM_PLACEMENT") return "model_deployment.llm.placement";
  if (env == "AI_LLM_REMOTE_URL") return "model_deployment.llm.remoteUrl";
  if (env == "AI_LLM_REMOTE_METHOD") return "model_deployment.llm.remoteMethod";
  if (env == "AI_LLM_REMOTE_MODEL") return "model_deployment.llm.remoteModel";
  if (env == "AI_LLM_REMOTE_TOKEN") return "model_deployment.llm.remoteToken";
  if (env == "AI_LLM_REMOTE_TIMEOUT_MS") return "model_deployment.llm.remoteTimeoutMs";
  if (env == "AI_VISION_PLACEMENT") return "model_deployment.vision.placement";
  if (env == "AI_VISION_REMOTE_URL") return "model_deployment.vision.remoteUrl";
  if (env == "AI_VISION_REMOTE_METHOD") return "model_deployment.vision.remoteMethod";
  if (env == "AI_VISION_REMOTE_MODEL") return "model_deployment.vision.remoteModel";
  if (env == "AI_VISION_REMOTE_TOKEN") return "model_deployment.vision.remoteToken";
  if (env == "AI_VISION_REMOTE_TIMEOUT_MS") return "model_deployment.vision.remoteTimeoutMs";
  if (env == "AI_SPEECH_PLACEMENT") return "model_deployment.speech.placement";
  if (env == "AI_SPEECH_REMOTE_URL") return "model_deployment.speech.remoteUrl";
  if (env == "AI_SPEECH_REMOTE_METHOD") return "model_deployment.speech.remoteMethod";
  if (env == "AI_SPEECH_REMOTE_MODEL") return "model_deployment.speech.remoteModel";
  if (env == "AI_SPEECH_REMOTE_TOKEN") return "model_deployment.speech.remoteToken";
  if (env == "AI_SPEECH_REMOTE_TIMEOUT_MS") return "model_deployment.speech.remoteTimeoutMs";
  return "";
}

std::string argOrEnv(
    const std::map<std::string, std::string> &args,
    const std::string &argKey,
    const std::string &envKey,
    const std::string &defaultValue) {
  auto it = args.find(argKey);
  if (it != args.end() && !it->second.empty()) return it->second;
  return phoenix::resolveConfigAsString(dotPathForModelEnv(envKey), defaultValue,
                                        envKey.c_str());
}

void applyRecordFromArgs(
    ModelDeploymentRecord &record,
    const std::map<std::string, std::string> &args,
    const std::string &prefix,
    const std::string &envPrefix) {
  record.placement = parsePlacement(argOrEnv(
      args, prefix + "-placement", envPrefix + "_PLACEMENT", "local"));
  record.remote.url = argOrEnv(
      args, prefix + "-remote-url", envPrefix + "_REMOTE_URL", "");
  record.remote.method = lowerCopy(argOrEnv(
      args, prefix + "-remote-method", envPrefix + "_REMOTE_METHOD", "http-json"));
  record.remote.modelName = argOrEnv(
      args, prefix + "-remote-model", envPrefix + "_REMOTE_MODEL", "");
  record.remote.authToken = argOrEnv(
      args, prefix + "-remote-token", envPrefix + "_REMOTE_TOKEN", "");
  std::string timeoutRaw = argOrEnv(
      args, prefix + "-remote-timeout-ms", envPrefix + "_REMOTE_TIMEOUT_MS", "");
  if (!timeoutRaw.empty()) {
    try {
      record.remote.timeoutMs = std::max(1, std::stoi(timeoutRaw));
    } catch (...) {
    }
  }
}

}  // namespace

void ModelDeploymentConfig::load(const std::map<std::string, std::string> &args,
                                 const std::string &rootDir,
                                 const std::string &configPath) {
  reset();

  std::string cfgPath = configPath;
  if (cfgPath.empty())
    cfgPath = argOrEnv(args, "model-deployment-config",
                       "AI_MODEL_DEPLOYMENT_CONFIG", "");

  // 1. Optional JSON config file.
  if (!cfgPath.empty()) {
    fs::path p = cfgPath;
    if (p.is_relative()) p = fs::path(rootDir) / p;
    p = fs::absolute(p);
    std::error_code ec;
    if (fs::exists(p, ec) && fs::is_regular_file(p, ec)) {
      std::ifstream in(p, std::ios::binary);
      if (in) {
        std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        auto j = nlohmann::json::parse(text, nullptr, false);
        if (!j.is_discarded()) fromJson(j);
      }
    }
  }

  // 2. Args/env override.
  {
    std::lock_guard<std::mutex> lock(mu_);
    applyRecordFromArgs(llm_, args, "llm", "AI_LLM");
    applyRecordFromArgs(vision_, args, "vision", "AI_VISION");
    applyRecordFromArgs(speech_, args, "speech", "AI_SPEECH");
  }
}

namespace {

constexpr std::array<char, 64> kBase64Chars = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

int base64Index(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  if (c == '=') return 0;
  return -1;
}

}  // namespace

std::string RemoteModelClient::base64Encode(const std::vector<uint8_t> &data) {
  std::string out;
  out.reserve(((data.size() + 2) / 3) * 4);
  size_t i = 0;
  while (i + 2 < data.size()) {
    uint32_t b = (static_cast<uint32_t>(data[i]) << 16) |
                 (static_cast<uint32_t>(data[i + 1]) << 8) |
                 static_cast<uint32_t>(data[i + 2]);
    out.push_back(kBase64Chars[(b >> 18) & 0x3f]);
    out.push_back(kBase64Chars[(b >> 12) & 0x3f]);
    out.push_back(kBase64Chars[(b >> 6) & 0x3f]);
    out.push_back(kBase64Chars[b & 0x3f]);
    i += 3;
  }
  if (i == data.size() - 1) {
    uint32_t b = static_cast<uint32_t>(data[i]) << 16;
    out.push_back(kBase64Chars[(b >> 18) & 0x3f]);
    out.push_back(kBase64Chars[(b >> 12) & 0x3f]);
    out.push_back('=');
    out.push_back('=');
  } else if (i == data.size() - 2) {
    uint32_t b = (static_cast<uint32_t>(data[i]) << 16) |
                 (static_cast<uint32_t>(data[i + 1]) << 8);
    out.push_back(kBase64Chars[(b >> 18) & 0x3f]);
    out.push_back(kBase64Chars[(b >> 12) & 0x3f]);
    out.push_back(kBase64Chars[(b >> 6) & 0x3f]);
    out.push_back('=');
  }
  return out;
}

std::vector<uint8_t> RemoteModelClient::base64Decode(const std::string &encoded) {
  std::vector<uint8_t> out;
  if (encoded.empty()) return out;
  out.reserve((encoded.size() * 3) / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (char ch : encoded) {
    if (std::isspace(static_cast<unsigned char>(ch))) continue;
    int idx = base64Index(ch);
    if (idx < 0) return {};  // malformed
    if (ch == '=') break;
    acc = (acc << 6) | static_cast<uint32_t>(idx);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>((acc >> bits) & 0xff));
    }
  }
  return out;
}

nlohmann::json RemoteModelClient::call(const RemoteEndpoint &endpoint,
                                       const nlohmann::json &payload,
                                       const std::string &path) {
  nlohmann::json error = {{"ok", false},
                          {"error", "remote model call failed"},
                          {"url", endpoint.url},
                          {"method", endpoint.method}};
  if (endpoint.url.empty()) {
    error["error"] = "remote endpoint URL is empty";
    return error;
  }
  try {
    auto [base, basePath] = splitHttpUrl(endpoint.url);
    std::string reqPath = basePath;
    if (!path.empty()) {
      if (!reqPath.empty() && reqPath.back() == '/' && path.front() == '/')
        reqPath.pop_back();
      else if (!reqPath.empty() && reqPath.back() != '/' && path.front() != '/')
        reqPath.push_back('/');
      reqPath += path;
    }
    if (reqPath.empty()) reqPath = "/";

    auto client = drogon::HttpClient::newHttpClient(base);
    if (!client) {
      error["error"] = "drogon HttpClient creation failed";
      return error;
    }
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->setPath(reqPath);
    req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    if (!endpoint.authToken.empty())
      req->addHeader("Authorization", "Bearer " + endpoint.authToken);
    for (const auto &kv : endpoint.headers)
      req->addHeader(kv.first, kv.second);
    req->setBody(payload.dump());

    std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
    auto future = promise.get_future();
    client->sendRequest(req, [&promise](drogon::ReqResult result,
                                        const drogon::HttpResponsePtr &resp) {
      try {
        promise.set_value({result, resp});
      } catch (...) {
      }
    });
    if (future.wait_for(std::chrono::milliseconds(
            std::max(1, endpoint.timeoutMs))) != std::future_status::ready) {
      error["error"] = "remote model request timeout";
      error["timeoutMs"] = endpoint.timeoutMs;
      return error;
    }
    auto pair = future.get();
    if (pair.first != drogon::ReqResult::Ok || !pair.second) {
      error["error"] = "remote model HTTP request failed";
      return error;
    }
    auto resp = pair.second;
    if (resp->statusCode() < 200 || resp->statusCode() >= 300) {
      error["error"] = "remote model returned bad status";
      error["status"] = resp->statusCode();
      return error;
    }
    auto body = resp->getBody();
    auto j = nlohmann::json::parse(body, nullptr, false);
    if (j.is_discarded()) {
      error["error"] = "remote model returned non-JSON body";
      return error;
    }
    return j;
  } catch (const std::exception &e) {
    error["error"] = std::string("remote model exception: ") + e.what();
    return error;
  } catch (...) {
    error["error"] = "remote model unknown exception";
    return error;
  }
}

}  // namespace deployment
}  // namespace phoenix
