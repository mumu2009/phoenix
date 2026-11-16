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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "phoenix_config.hpp"

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
  std::vector<std::string> roles;   /*!< tool-side tags, e.g. "programming". The base model is not shown these. */

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
            {"pinnedExpert", pinnedExpert},
            {"roles", roles}};
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
    if (j.contains("roles") && j["roles"].is_array()) {
      for (const auto &role : j["roles"]) {
        if (role.is_string() && !role.get<std::string>().empty())
          ep.roles.push_back(role.get<std::string>());
      }
    }
    return ep;
  }

  /** True when a tool may treat this endpoint as the programming specialist.
      Matching is on the configured role tag only. Description text is not
      scanned, and nothing here is written into a prompt. */
  bool servesProgramming() const {
    for (const auto &role : roles) {
      if (role == "programming" || role == "code") return true;
    }
    return false;
  }
};

/* ---------------------------------------------------------------------------
   Scored summoning.

   selectForSourceEdit() used to return the FIRST endpoint tagged
   "programming": with several expert servers registered (e0..e3) the first
   row took every call, a failing expert kept being summoned (no health
   signal) and a busy expert was preferred over an idle one.

     score = roleBonus + reliability*RELIABILITY_WEIGHT
           - IN_FLIGHT_PENALTY*inFlight
           - FAILURE_PENALTY*min(consecutiveFailures, FAILURE_CAP)
           + STICKY_BONUS (last used < 60 s ago)

   Reliability is Laplace-smoothed, so an unused expert starts at 0.5:
   exploration is possible but never preferred over a proven endpoint.
   Two consecutive failures cost more than four in-flight calls, which is
   what "stop summoning the broken one" has to mean in practice.
   The policy is pure (no I/O, no locks) so it is unit tested directly. */
struct EndpointStats {
  int inFlight{0};
  int successes{0};
  int failures{0};
  int consecutiveFailures{0};
  double lastLatencyMs{0.0};
  int64_t lastUsedMs{0};
};

class EndpointRanker {
 public:
  static constexpr double ROLE_BONUS = 2.0;
  static constexpr double RELIABILITY_WEIGHT = 1.5;
  static constexpr double IN_FLIGHT_PENALTY = 0.5;
  static constexpr double FAILURE_PENALTY = 1.5;
  static constexpr int FAILURE_CAP = 4;
  static constexpr double STICKY_BONUS = 0.1;
  static constexpr int64_t STICKY_WINDOW_MS = 60000;

  static double reliability(const EndpointStats &s) {
    const double ok = static_cast<double>(std::max(0, s.successes));
    const double bad = static_cast<double>(std::max(0, s.failures));
    return (ok + 1.0) / (ok + bad + 2.0);
  }

  double score(const ModelEndpoint &ep, const EndpointStats &st,
               const std::string &domain, int64_t nowMs) const {
    double sc = 0.0;
    if (!domain.empty()) {
      for (const auto &role : ep.roles) {
        if (role == domain) {
          sc += ROLE_BONUS;
          break;
        }
      }
    } else {
      sc += ROLE_BONUS;
    }
    sc += reliability(st) * RELIABILITY_WEIGHT;
    sc -= IN_FLIGHT_PENALTY * static_cast<double>(std::max(0, st.inFlight));
    sc -= FAILURE_PENALTY * static_cast<double>(
                                 std::min(std::max(0, st.consecutiveFailures),
                                          FAILURE_CAP));
    if (st.lastUsedMs > 0 && nowMs > st.lastUsedMs &&
        nowMs - st.lastUsedMs < STICKY_WINDOW_MS)
      sc += STICKY_BONUS;
    return sc;
  }

  /* Exploration width.  pick()/argmax always summons the same endpoint once
     it holds the best score, so a plateau of equally good experts never gets
     tried - the classic greedy local optimum.  pickSoftmax samples from
     softmax(score / T): width is preserved, and the caller anneals T down so
     the policy converges late instead of never.  T <= 0 falls back to argmax
     (identical behaviour to pick()). */
  int pickSoftmax(
      const std::vector<ModelEndpoint> &endpoints,
      const std::function<EndpointStats(const std::string &)> &statsFor,
      const std::string &domain, int64_t nowMs, double temperature,
      std::mt19937 &rng) const {
    if (!(temperature > 0.0)) return pick(endpoints, statsFor, domain, nowMs);
    std::vector<int> idx;
    std::vector<double> sc;
    double best = 0.0;
    for (size_t i = 0; i < endpoints.size(); ++i) {
      const auto &ep = endpoints[i];
      if (!ep.located()) continue;
      if (!domain.empty()) {
        bool matched = false;
        for (const auto &role : ep.roles)
          if (role == domain) matched = true;
        if (!matched) continue;
      }
      EndpointStats st;
      if (statsFor) st = statsFor(ep.name);
      const double s = score(ep, st, domain, nowMs);
      if (idx.empty() || s > best) best = s;
      idx.push_back(static_cast<int>(i));
      sc.push_back(s);
    }
    if (idx.empty()) return -1;
    double total = 0.0;
    std::vector<double> w(sc.size(), 0.0);
    for (size_t k = 0; k < sc.size(); ++k) {
      w[k] = std::exp((sc[k] - best) / temperature);
      total += w[k];
    }
    if (!(total > 0.0)) return idx.front();
    std::uniform_real_distribution<double> uni(0.0, total);
    const double r = uni(rng);
    double acc = 0.0;
    for (size_t k = 0; k < w.size(); ++k) {
      acc += w[k];
      if (r <= acc) return idx[k];
    }
    return idx.back();
  }

  int pick(const std::vector<ModelEndpoint> &endpoints,
           const std::function<EndpointStats(const std::string &)> &statsFor,
           const std::string &domain, int64_t nowMs) const {
    int best = -1;
    double bestScore = 0.0;
    for (size_t i = 0; i < endpoints.size(); ++i) {
      const auto &ep = endpoints[i];
      if (!ep.located()) continue;
      if (!domain.empty()) {
        bool matched = false;
        for (const auto &role : ep.roles)
          if (role == domain) matched = true;
        if (!matched) continue;
      }
      EndpointStats st;
      if (statsFor) st = statsFor(ep.name);
      const double sc = score(ep, st, domain, nowMs);
      if (best < 0 || sc > bestScore) {
        best = static_cast<int>(i);
        bestScore = sc;
      }
    }
    return best;
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
        if (ep.kind == "moe" && item.is_object() && !item.contains("roles") &&
            ep.roles.empty())
          ep.roles.push_back("programming");
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

  /** Tool-side pick for a source-edit situation. Empty name means no
      endpoint tagged "programming" is registered. Does not start a process
      and does not produce prompt text.
      v10.2: scored (see EndpointRanker) instead of "first row wins". */
  ModelEndpoint selectForSourceEdit() const {
    std::lock_guard<std::mutex> lock(mu_);
    ModelEndpoint none;
    if (!enabled_) return none;
    const std::unordered_map<std::string, EndpointStats> snapshot = stats_;
    EndpointRanker ranker;
    const int64_t nowMs = nowMsSinceEpoch();
    auto statsLookup = [&snapshot](const std::string &n) {
      auto it = snapshot.find(n);
      return it == snapshot.end() ? EndpointStats{} : it->second;
    };
    /* Width control: summon.selectTemperature > 0 samples softmax(score/T) and
       anneals T per call (summon.selectAnneal, default 0.995, floor 0.05).
       Default 0 keeps the deterministic argmax - no silent behaviour change. */
    const double t0 = phoenix::cfgOr<double>("summon.selectTemperature", 0.0);
    int idx = -1;
    if (t0 > 0.0) {
      const double decay = std::max(0.5, std::min(1.0,
          phoenix::cfgOr<double>("summon.selectAnneal", 0.995)));
      const double temp =
          std::max(0.05, t0 * std::pow(decay, static_cast<double>(selectCalls_)));
      ++selectCalls_;
      idx = ranker.pickSoftmax(endpoints_, statsLookup, "programming", nowMs,
                               temp, rng_);
    } else {
      idx = ranker.pick(endpoints_, statsLookup, "programming", nowMs);
    }
    if (idx < 0) return none;
    return endpoints_[static_cast<size_t>(idx)];
  }

  /* ---- summon outcome accounting (feeds EndpointRanker) ---- */

  static int64_t nowMsSinceEpoch() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  /** A summon call is starting: counts against in-flight and stamps use. */
  void noteSummonStart(const std::string &name) {
    if (name.empty()) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto &st = stats_[name];
    st.inFlight += 1;
    st.lastUsedMs = nowMsSinceEpoch();
  }

  /** A summon call finished: updates reliability and clears one in-flight. */
  void noteSummonOutcome(const std::string &name, bool ok,
                         double latencyMs = 0.0) {
    if (name.empty()) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto &st = stats_[name];
    if (st.inFlight > 0) st.inFlight -= 1;
    st.lastUsedMs = nowMsSinceEpoch();
    if (latencyMs > 0.0) st.lastLatencyMs = latencyMs;
    if (ok) {
      st.successes += 1;
      st.consecutiveFailures = 0;
    } else {
      st.failures += 1;
      st.consecutiveFailures += 1;
    }
  }

  EndpointStats statsFor(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = stats_.find(name);
    return it == stats_.end() ? EndpointStats{} : it->second;
  }

  /* Health snapshot for /api/summon and tests. */
  nlohmann::json outcomes() const {
    std::lock_guard<std::mutex> lock(mu_);
    nlohmann::json out = nlohmann::json::object();
    for (const auto &kv : stats_) {
      out[kv.first] = nlohmann::json{{"inFlight", kv.second.inFlight},
                                     {"successes", kv.second.successes},
                                     {"failures", kv.second.failures},
                                     {"consecutiveFailures",
                                      kv.second.consecutiveFailures},
                                     {"lastLatencyMs", kv.second.lastLatencyMs},
                                     {"lastUsedMs", kv.second.lastUsedMs}};
    }
    return out;
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
    if (!spec.contains("roles") && ep.roles.empty())
      ep.roles.push_back("programming");
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
  std::unordered_map<std::string, EndpointStats> stats_;
  /* softmax annealing state (see pickSoftmax) */
  mutable uint64_t selectCalls_{0};
  mutable std::mt19937 rng_{0x5eed1234u};
};

}  // namespace summon
}  // namespace phoenix
