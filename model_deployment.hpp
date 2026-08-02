/* model_deployment.hpp - Runtime model placement topology for Phoenix v7.0 "Arthur"
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version. */

#pragma once

#include <nlohmann/json.hpp>
#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace phoenix {
namespace deployment {

/**
 * @brief Where a model should be executed.
 *
 * - Local: the model is executed inside the Phoenix process (or on the same
 *   host with a local IPC/HTTP server).
 * - Remote: the model is executed on a separate host and called over the
 *   network via an HTTP/JSON endpoint.
 * - Auto: let the model factory decide.  Typically remote is chosen when a
 *   remote URL is configured, otherwise a local backend is used.
 * - ServerClient: the client (e.g. a web browser) runs the pre-processing and
 *   model, and sends pre-computed concept vectors to the Phoenix backend.  The
 *   backend only consumes the concept and never runs the model itself.
 */
enum class ModelPlacement {
  Local = 0,
  Remote = 1,
  Auto = 2,
  ServerClient = 3,
};

/**
 * @brief Local backend used when a model is placed on this host.
 *
 * - Cpu:    x86_64 / general-purpose CPU; loads and runs the ONNX model
 *           (directly via ONNX Runtime or through the local HTTP runner).
 * - Gpu:    local GPU / CUDA (uses the same ONNX path with GPU provider).
 * - Bpu:    aarch64 / RDK X5 BPU; loads the compiled Horizon .bin.
 * - Js:     browser / client-side JS runner (server-client mode stub).
 * - Auto:   let the factory choose from the build / runtime environment.
 */
enum class LocalBackendType {
  Auto = 0,
  Cpu = 1,
  Gpu = 2,
  Bpu = 3,
  Js = 4,
};

/**
 * @brief Convert a ModelPlacement value to a human-readable string.
 *
 * @param p  placement value.
 * @return   "local", "remote", "auto", or "server-client".
 */
std::string placementToString(ModelPlacement p);

/**
 * @brief Convert a LocalBackendType value to a human-readable string.
 */
std::string localBackendTypeToString(LocalBackendType b);

/**
 * @brief Parse a placement string.
 *
 * Accepts "local", "remote", "auto", "on-device", "edge" and a few common
 * aliases.
 *
 * @param s  input string; leading/trailing whitespace is ignored.
 * @return   corresponding ModelPlacement, or ModelPlacement::Local on failure.
 */
ModelPlacement parsePlacement(const std::string &s);

/**
 * @brief Parse a local backend string.
 *
 * Accepts "cpu", "gpu", "bpu", "js", "auto" and common aliases.
 *
 * @param s  input string; leading/trailing whitespace is ignored.
 * @return   corresponding LocalBackendType, or LocalBackendType::Auto on failure.
 */
LocalBackendType parseLocalBackendType(const std::string &s);

/**
 * @brief Description of a remote model endpoint.
 *
 * The endpoint carries everything Phoenix needs to reach a model that lives on
 * another machine (e.g. an RDK X5 running a vision BPU bridge, a desktop GPU
 * running Ollama, or a dedicated speech inference box).
 */
struct RemoteEndpoint {
  /** @brief Base URL of the remote service, e.g. http://192.168.1.10:5000. */
  std::string url;

  /**
   * @brief Calling convention / protocol.
   *
   * For LLMs: "ollama", "llamacpp", "bitnet", "openai".
   * For vision/speech: "http-json" (generic base64-in / embedding-out),
   *                    "bpu-bridge" (RDK X5 BPU bridge protocol).
   */
  std::string method;

  /**
   * @brief Optional model name.
   *
   * Used by LLM backends that require a model selector (Ollama, OpenAI).
   */
  std::string modelName;

  /** @brief Optional bearer token or API key. */
  std::string authToken;

  /** @brief Request timeout in milliseconds. */
  int timeoutMs = 30000;

  /** @brief Extra HTTP headers to send with every request. */
  std::map<std::string, std::string> headers;

  /**
   * @brief Serialize to JSON.
   *
   * The JSON preserves all fields and is used for configuration snapshots.
   */
  nlohmann::json toJson() const;

  /**
   * @brief Restore from JSON.
   *
   * Missing fields keep their default values; malformed strings are ignored.
   */
  void fromJson(const nlohmann::json &j);
};

/**
 * @brief Per-model type deployment record.
 *
 * Combines placement (local/remote) and the remote endpoint details used when
 * the model is not executed inside the Phoenix process.
 */
struct ModelDeploymentRecord {
  ModelPlacement placement = ModelPlacement::Local;
  LocalBackendType localBackend = LocalBackendType::Auto;
  RemoteEndpoint remote;

  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json &j);
};

/**
 * @brief Central runtime model deployment configuration.
 *
 * The singleton is populated at startup from command-line arguments, environment
 * variables, or an optional JSON configuration file.  Downstream model factories
 * and chat backends query it to decide whether to run locally or dispatch to an
 * edge device.
 */
class ModelDeploymentConfig {
 public:
  /** @brief Singleton accessor. */
  static ModelDeploymentConfig &instance();

  /**
   * @brief Load configuration from a JSON file, args and env.
   *
   * Later sources override earlier sources in the following order:
   *   1. Optional JSON file (model-deployment-config)
   *   2. Environment variables (AI_*_PLACEMENT, AI_*_REMOTE_URL, ...)
   *   3. Command-line arguments (--llm-placement, --vision-remote-url, ...)
   *
   * @param args        parsed command-line map (key -> value, no leading dashes).
   * @param rootDir     project root used to resolve relative config file paths.
   * @param configPath  explicit config file path (may be empty).
   */
  void load(const std::map<std::string, std::string> &args,
            const std::string &rootDir,
            const std::string &configPath = "");

  /** @brief Reset to the default all-local topology. */
  void reset();

  /** @brief Serialize the entire topology to JSON. */
  nlohmann::json toJson() const;

  /** @brief Restore the entire topology from JSON. */
  void fromJson(const nlohmann::json &j);

  /** @brief Access the LLM deployment record. */
  const ModelDeploymentRecord &llm() const { return llm_; }

  /** @brief Access the vision (image) deployment record. */
  const ModelDeploymentRecord &vision() const { return vision_; }

  /** @brief Access the speech (audio) deployment record. */
  const ModelDeploymentRecord &speech() const { return speech_; }

  /**
   * @brief Override LLM record after initial load.
   *
   * Used by Config validation when an explicit local backend is selected.
   */
  void setLlm(const ModelDeploymentRecord &r) { llm_ = r; }

 private:
  ModelDeploymentConfig() = default;

  ModelDeploymentRecord llm_;
  ModelDeploymentRecord vision_;
  ModelDeploymentRecord speech_;
  mutable std::mutex mu_;
};

/**
 * @brief HTTP client for remote model endpoints.
 *
 * Wraps drogon::HttpClient with a synchronous std::future/promise interface.
 * The remote service must accept a JSON POST body and return a JSON response.
 */
class RemoteModelClient {
 public:
  /**
   * @brief Send a generic JSON request to a remote endpoint.
   *
   * @param endpoint  target service description.
   * @param payload   request body.
   * @param path      optional URL path appended to endpoint.url.
   * @return          JSON response body, or {"ok":false,"error":...} on failure.
   */
  static nlohmann::json call(const RemoteEndpoint &endpoint,
                             const nlohmann::json &payload,
                             const std::string &path = "");

  /**
   * @brief Encode raw bytes as base64.
   *
   * @param data  raw input bytes.
   * @return    base64 encoded string.
   */
  static std::string base64Encode(const std::vector<uint8_t> &data);

  /**
   * @brief Decode a base64 string.
   *
   * @param encoded  base64 string.
   * @return         decoded bytes; empty on failure.
   */
  static std::vector<uint8_t> base64Decode(const std::string &encoded);
};

}  // namespace deployment
}  // namespace phoenix
