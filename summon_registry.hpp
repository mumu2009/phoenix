/* summon_registry.hpp - Registry of summonable external models

   The base model (e.g. llama3.1-8b on the main llama-server) can summon
   helper boxes that run on OTHER models: a bigger MoE served by ollama, or
   a single-expert llama-server instance named `${model}_${expertId}`
   (see llama_server_mods/force_moe_expert.patch).

   HARD CONSTRAINT (user rule): the framework adapts to the model, never the
   other way round.  Nothing here teaches the base model how to call tools
   (no prompt text, no fine-tuning): the summon decision is made by the EFE
   planner picking a registered `summon.<name>` action, and each endpoint
   adapter translates Phoenix's prompt into the target backend's API shape
   (ollama /api/chat, llamacpp /completion, OpenAI /v1/chat/completions).

   Config (config/phoenix.json):
     "summon": {
       "enabled": false,
       "models": [
         {"name": "qwen3-30b",      "kind": "ollama",
          "url": "http://127.0.0.1:11434", "model": "qwen3:30b",
          "timeoutMs": 120000, "maxTokens": 256,
          "description": "big MoE whole model"},
         {"name": "qwen3-30b_e3",   "kind": "llamacpp",
          "url": "http://127.0.0.1:8093", "model": "",
          "timeoutMs": 120000, "maxTokens": 128,
          "description": "MoE expert 3 (forced single expert)"}
       ],
       "moe": {
         "name": "moe", "kind": "moe",
         "modelPath": "", "url": "", "port": 0,
         "k": 1, "pinnedExpert": -1
       }
     }

   Single MoE (kind "moe") is one registry row. Canonical keys, with aliases
   the loader also accepts: modelPath (path, modelFile, gguf), url, port,
   k (K, activeExperts), pinnedExpert (forceExpert, expertId). Defaults when
   a key is absent: port 0, k 1 (clamped to 1..64), pinnedExpert -1 (not
   pinned). An empty url plus port>0 is recorded as http://127.0.0.1:<port>.
   summon.moe may omit name (default "moe") and kind. Registration never
   starts a process. Concrete port, K and modelPath are filled by the
   runtime config, not by this header.

   An empty model name on a child box means "base model" and never touches
   this registry.  Disabled (default) means no model field is accepted at
   all - summon is an opt-in experiment.
*/
#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

namespace phoenix {
namespace summon {

/** @brief One summonable external model endpoint. */
struct ModelEndpoint {
  std::string name;                 /*!< registry key, e.g. "qwen3-30b_e3". */
  std::string kind{"llamacpp"};     /*!< "ollama" | "llamacpp" | "openai" | "tiny-expert" | "moe". */
  std::string url;                  /*!< base url, e.g. http://127.0.0.1:8093. */
  std::string model;                /*!< backend-side model id (ollama/openai). */
  int timeoutMs{120000};            /*!< bounded per-call timeout. */
  int maxTokens{256};               /*!< default generation cap. */
  std::string description;          /*!< human/planner-facing capability text. */
  std::string weights;              /*!< local file for kind "tiny-expert". */
  std::string modelPath;            /*!< local GGUF for kind "moe". */
  int port{0};                      /*!< endpoint port; 0 means unset. */
  int k{1};                         /*!< active expert count for kind "moe". */
  int pinnedExpert{-1};             /*!< forced expert id; -1 means not pinned. */

  /** Fill MoE defaults. Does not open a socket or start a process. */
  void applyDefaults() {
    if (kind != "moe") return;
    if (k < 1) k = 1;
    if (k > 64) k = 64;
    if (pinnedExpert < -1) pinnedExpert = -1;
    if (pinnedExpert > 1024) pinnedExpert = 1024;
    if (port < 0 || port > 65535) port = 0;
    if (url.empty() && port > 0)
      url = std::string("http://127.0.0.1:") + std::to_string(port);
  }

  bool located() const {
    if (kind == "tiny-expert") return !weights.empty();
    if (kind == "moe")
      return !modelPath.empty() || !url.empty() || port > 0;
    return !url.empty();
  }

  nlohmann::json toJson() const {
    return {{"name", name},         {"kind", kind},
            {"url", url},           {"model", model},
            {"timeoutMs", timeoutMs}, {"maxTokens", maxTokens},
            {"description", description},
            {"weights", weights},
            {"modelPath", modelPath},
            {"port", port},
            {"k", k},
            {"pinnedExpert", pinnedExpert}};
  }

  static int pickInt(const nlohmann::json &j, int fallback,
                     const char *a, const char *b = nullptr,
                     const char *c = nullptr) {
    const char *keys[3] = {a, b, c};
    for (const char *key : keys) {
      if (key == nullptr) continue;
      if (j.contains(key) && j[key].is_number()) return j[key].get<int>();
    }
    return fallback;
  }

  static ModelEndpoint fromJson(const nlohmann::json &j) {
    ModelEndpoint ep;
    if (!j.is_object()) return ep;
    if (j.contains("name") && j["name"].is_string())
      ep.name = j["name"].get<std::string>();
    if (j.contains("kind") && j["kind"].is_string())
      ep.kind = j["kind"].get<std::string>();
    if (j.contains("url") && j["url"].is_string())
      ep.url = j["url"].get<std::string>();
    if (j.contains("model") && j["model"].is_string())
      ep.model = j["model"].get<std::string>();
    if (j.contains("timeoutMs") && j["timeoutMs"].is_number())
      ep.timeoutMs = j["timeoutMs"].get<int>();
    if (j.contains("maxTokens") && j["maxTokens"].is_number())
      ep.maxTokens = j["maxTokens"].get<int>();
    if (j.contains("description") && j["description"].is_string())
      ep.description = j["description"].get<std::string>();
    if (j.contains("weights") && j["weights"].is_string())
      ep.weights = j["weights"].get<std::string>();
    if (j.contains("modelPath") && j["modelPath"].is_string())
      ep.modelPath = j["modelPath"].get<std::string>();
    else if (j.contains("path") && j["path"].is_string())
      ep.modelPath = j["path"].get<std::string>();
    else if (j.contains("modelFile") && j["modelFile"].is_string())
      ep.modelPath = j["modelFile"].get<std::string>();
    else if (j.contains("gguf") && j["gguf"].is_string())
      ep.modelPath = j["gguf"].get<std::string>();
    ep.port = pickInt(j, ep.port, "port");
    ep.k = pickInt(j, ep.k, "k", "K", "activeExperts");
    ep.pinnedExpert = pickInt(j, ep.pinnedExpert, "pinnedExpert",
                              "forceExpert", "expertId");
    return ep;
  }
};

/** @brief Process-wide summon registry (config-driven, thread-safe). */
class SummonRegistry {
 public:
  static SummonRegistry &instance() {
    static SummonRegistry reg;
    return reg;
  }

  /** Replace the whole registry from config.  Invalid entries (empty name,
      empty url, unknown kind) are skipped and reported in the return value. */
  nlohmann::json loadFromConfig(bool enabled, const nlohmann::json &models) {
    std::lock_guard<std::mutex> lock(mu_);
    enabled_ = enabled;
    endpoints_.clear();
    scopeOverrides_.clear();
    std::vector<std::string> skipped;
    if (models.is_array()) {
      for (const auto &item : models) {
        ModelEndpoint ep = ModelEndpoint::fromJson(item);
        ep.applyDefaults();
        const bool knownKind = ep.kind == "ollama" || ep.kind == "llamacpp" ||
                               ep.kind == "openai" || ep.kind == "tiny-expert" ||
                               ep.kind == "moe";
        if (ep.name.empty() || !knownKind || !ep.located()) {
          skipped.push_back(ep.name.empty() ? std::string("<unnamed>")
                                            : ep.name);
          continue;
        }
        if (ep.timeoutMs < 1000) ep.timeoutMs = 1000;
        if (ep.maxTokens < 1) ep.maxTokens = 1;
        endpoints_.push_back(std::move(ep));
      }
    }
    nlohmann::json names = nlohmann::json::array();
    for (const auto &ep : endpoints_) names.push_back(ep.name);
    return nlohmann::json{{"enabled", enabled_},
                          {"loaded", names},
                          {"skipped", skipped}};
  }

  bool enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return enabled_;
  }

  /** Find an endpoint by name (nullptr when absent or disabled). */
  const ModelEndpoint *find(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return nullptr;
    for (const auto &ep : endpoints_) {
      if (ep.name == name) return &ep;
    }
    return nullptr;
  }

  std::vector<ModelEndpoint> list() const {
    std::lock_guard<std::mutex> lock(mu_);
    return endpoints_;
  }

  /* scope -> model override: set when a helper box is spawned with a model
     field; the gateway deliberator resolves its workspace scope
     ("<missionId>/children/<boxId>") through this table.  Empty model name
     is never stored (base model needs no override). */
  void setScopeModel(const std::string &scope, const std::string &model) {
    if (scope.empty() || model.empty()) return;
    std::lock_guard<std::mutex> lock(mu_);
    scopeOverrides_[scope] = model;
  }

  std::string modelForScope(const std::string &scope) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) return std::string();
    const auto it = scopeOverrides_.find(scope);
    return it == scopeOverrides_.end() ? std::string() : it->second;
  }

  void clearScopeModel(const std::string &scope) {
    std::lock_guard<std::mutex> lock(mu_);
    scopeOverrides_.erase(scope);
  }

  /** Insert or replace one endpoint without dropping the others.
      kind "tiny-expert" needs a weights path and may omit url.
      kind "moe" needs a model path, url, or port. Nothing is spawned. */
  nlohmann::json upsert(const ModelEndpoint &ep) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) {
      return nlohmann::json{{"ok", false}, {"reason", "disabled"},
                            {"started", false}};
    }
    ModelEndpoint copy = ep;
    copy.applyDefaults();
    const bool tiny = copy.kind == "tiny-expert";
    const bool moe = copy.kind == "moe";
    const bool known = tiny || moe || copy.kind == "ollama" ||
                       copy.kind == "llamacpp" || copy.kind == "openai";
    if (copy.name.empty() || !known) {
      return nlohmann::json{{"ok", false}, {"reason", "invalid"},
                            {"started", false}};
    }
    if (tiny && copy.weights.empty()) {
      return nlohmann::json{{"ok", false}, {"reason", "weights"},
                            {"started", false}};
    }
    if (moe && !copy.located()) {
      return nlohmann::json{{"ok", false}, {"reason", "location"},
                            {"started", false}};
    }
    if (!tiny && !moe && copy.url.empty()) {
      return nlohmann::json{{"ok", false}, {"reason", "url"},
                            {"started", false}};
    }
    if (copy.timeoutMs < 1000) copy.timeoutMs = 1000;
    if (copy.maxTokens < 1) copy.maxTokens = 1;
    for (auto &slot : endpoints_) {
      if (slot.name == copy.name) {
        slot = std::move(copy);
        return nlohmann::json{{"ok", true}, {"updated", true},
                              {"started", false}};
      }
    }
    endpoints_.push_back(std::move(copy));
    return nlohmann::json{{"ok", true}, {"updated", false},
                          {"started", false}};
  }

  /** Insert or replace the single MoE spec. Does not clear other endpoints
      and does not start a process. Missing name becomes "moe"; missing kind
      is forced to "moe". */
  nlohmann::json upsertMoe(const nlohmann::json &spec) {
    if (!spec.is_object() || spec.empty()) {
      return nlohmann::json{{"ok", false}, {"reason", "invalid"},
                            {"started", false}};
    }
    ModelEndpoint ep = ModelEndpoint::fromJson(spec);
    ep.kind = "moe";
    if (ep.name.empty()) ep.name = "moe";
    return upsert(ep);
  }

  nlohmann::json toJson() const {
    std::lock_guard<std::mutex> lock(mu_);
    nlohmann::json eps = nlohmann::json::array();
    for (const auto &ep : endpoints_) eps.push_back(ep.toJson());
    nlohmann::json ov = nlohmann::json::object();
    for (const auto &kv : scopeOverrides_) ov[kv.first] = kv.second;
    return nlohmann::json{{"enabled", enabled_},
                          {"models", eps},
                          {"scopeOverrides", ov}};
  }

 private:
  SummonRegistry() = default;
  mutable std::mutex mu_;
  bool enabled_{false};
  std::vector<ModelEndpoint> endpoints_;
  std::unordered_map<std::string, std::string> scopeOverrides_;
};

}  // namespace summon
}  // namespace phoenix
