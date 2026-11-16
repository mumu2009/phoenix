/* phoenix_config.hpp
   Single source of truth for Phoenix runtime parameters that were previously
   hard-coded.  The authoritative values live in config/phoenix.json; this
   header only exposes them.  There are no fallbacks in production code. */

#ifndef PHOENIX_CONFIG_HPP
#define PHOENIX_CONFIG_HPP

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace phoenix {

class PhoenixConfig {
 public:
  static PhoenixConfig& instance() {
    static PhoenixConfig inst;
    return inst;
  }

  void load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("Phoenix config not found: " + path.string());
    }
    std::ifstream f(path);
    if (!f.good()) {
      throw std::runtime_error("Cannot open Phoenix config: " + path.string());
    }
    f >> base_;
    basePath_ = path;
    baseMtime_ = mtimeOf(path);
    loaded_ = true;
    rebuild();
  }

  /* Deep-merge one JSON object over the loaded config.  Research profiles
     (config/research/<name>.json) are the only caller: a parameter
     combination overlays config/phoenix.json without touching the base
     file or any production code path.  Object leaves merge recursively;
     everything else is replaced. */
  void loadOverlay(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.good()) {
      std::cerr << "[config] research profile missing: " << path.string()
                << " (base config unchanged)" << std::endl;
      return;
    }
    nlohmann::json ov;
    try {
      f >> ov;
    } catch (const std::exception& e) {
      std::cerr << "[config] research profile unreadable: " << path.string()
                << ": " << e.what() << " (base config unchanged)" << std::endl;
      return;
    }
    profile_ = std::move(ov);
    profilePath_ = path;
    profileMtime_ = mtimeOf(path);
    rebuild();
  }

  /* Deep-merge an inline JSON object over the loaded config.  The feature
     tab mechanism (featureTabs.active -> featureTabs.tabs.<name>) uses this:
     a named tab is a parameter combination held inside config/phoenix.json,
     selected by one key instead of a separate file + env var. */
  void applyOverlay(const nlohmann::json& ov) {
    if (!ov.is_object()) {
      std::cerr << "[config] feature tab overlay is not an object (ignored)"
                << std::endl;
      return;
    }
    tabOverlay_ = ov;
    rebuild();
  }

  /* ---- v10.2 hot reload ------------------------------------------------
     cfgOr() used to read a snapshot taken once at startup, so every
     hyper-parameter needed a process restart.  The live config is now a
     copy-on-write shared_ptr: readers take the atomic pointer (no lock on the
     hot path), and reload() swaps in a freshly merged object.  A failed parse
     keeps the previous snapshot, so a half-written file cannot break a
     running system. */
  bool reloadNow() {
    if (basePath_.empty()) return false;
    nlohmann::json base;
    try {
      std::ifstream f(basePath_);
      if (!f.good()) return false;
      f >> base;
    } catch (...) {
      return false;
    }
    nlohmann::json prof = profile_;
    if (!profilePath_.empty()) {
      try {
        std::ifstream f(profilePath_);
        if (f.good()) {
          nlohmann::json o;
          f >> o;
          prof = std::move(o);
        }
      } catch (...) {
        /* keep the previous profile on a bad edit */
      }
    }
    base_ = std::move(base);
    profile_ = std::move(prof);
    baseMtime_ = mtimeOf(basePath_);
    profileMtime_ = mtimeOf(profilePath_);
    rebuild();
    return true;
  }

  /* Cheap check used by the watcher thread: returns true when a watched file
     changed and the reload succeeded. */
  bool reloadIfChanged() {
    const auto bm = mtimeOf(basePath_);
    const auto pm = mtimeOf(profilePath_);
    if (bm == baseMtime_ && pm == profileMtime_) return false;
    return reloadNow();
  }

  void setReloadHook(std::function<void()> hook) {
    reloadHook_ = std::move(hook);
  }

  /* Called by reloadNow() users after the swap so startup-cached subsystems
     can re-read their parameters. */
  void fireReloadHook() {
    if (reloadHook_) {
      try {
        reloadHook_();
      } catch (...) {
      }
    }
  }

  bool loaded() const { return loaded_; }

  const nlohmann::json& raw() const { return *snap_.load(); }

  template <typename T>
  T get(const std::string& dotPath) const {
    const auto snap = snap_.load();
    const nlohmann::json* p = getPathIn(*snap, dotPath);
    if (!p) {
      throw std::runtime_error("Missing config key: " + dotPath);
    }
    try {
      return p->get<T>();
    } catch (...) {
      throw std::runtime_error("Invalid type for config key: " + dotPath);
    }
  }

  template <typename T>
  T getOr(const std::string& dotPath, const T& fallback) const {
    const auto snap = snap_.load();
    const nlohmann::json* p = getPathIn(*snap, dotPath);
    if (!p) {
      return fallback;
    }
    try {
      return p->get<T>();
    } catch (...) {
      return fallback;
    }
  }

  const nlohmann::json* getJsonPtr(const std::string& dotPath) const {
    const auto snap = snap_.load();
    return getPathIn(*snap, dotPath);
  }

 private:
  PhoenixConfig() = default;

  static void mergeInto(nlohmann::json& base, const nlohmann::json& ov) {
    if (!ov.is_object() || !base.is_object()) {
      base = ov;
      return;
    }
    for (auto it = ov.begin(); it != ov.end(); ++it) {
      if (it.value().is_object() && base.contains(it.key()) &&
          base[it.key()].is_object()) {
        mergeInto(base[it.key()], it.value());
      } else {
        base[it.key()] = it.value();
      }
    }
  }

  static const nlohmann::json* getPathIn(const nlohmann::json& root,
                                         const std::string& dotPath) {
    const nlohmann::json* cur = &root;
    std::size_t start = 0;
    while (start <= dotPath.size()) {
      std::size_t dot = dotPath.find('.', start);
      std::string key = dotPath.substr(start, dot - start);
      if (!cur->is_object() || !cur->contains(key)) {
        return nullptr;
      }
      cur = &(*cur)[key];
      if (dot == std::string::npos) {
        break;
      }
      start = dot + 1;
    }
    return cur;
  }

  static std::filesystem::file_time_type mtimeOf(const std::filesystem::path& p) {
    std::error_code ec;
    if (p.empty()) return std::filesystem::file_time_type{};
    auto t = std::filesystem::last_write_time(p, ec);
    return ec ? std::filesystem::file_time_type{} : t;
  }

  void rebuild() {
    nlohmann::json merged = base_;
    mergeInto(merged, profile_);
    mergeInto(merged, tabOverlay_);
    if (!merged.is_object()) merged = nlohmann::json::object();
    snap_.store(std::make_shared<const nlohmann::json>(std::move(merged)));
  }

  std::atomic<std::shared_ptr<const nlohmann::json>> snap_{
      std::make_shared<const nlohmann::json>(nlohmann::json::object())};
  nlohmann::json base_ = nlohmann::json::object();
  nlohmann::json profile_ = nlohmann::json::object();
  nlohmann::json tabOverlay_ = nlohmann::json::object();
  std::filesystem::path basePath_;
  std::filesystem::path profilePath_;
  std::filesystem::file_time_type baseMtime_{};
  std::filesystem::file_time_type profileMtime_{};
  std::function<void()> reloadHook_;
  bool loaded_{false};
};

template <typename T>
inline T cfg(const std::string& dotPath) {
  return PhoenixConfig::instance().get<T>(dotPath);
}

template <typename T>
inline T cfgOr(const std::string& dotPath, const T& fallback) {
  return PhoenixConfig::instance().getOr<T>(dotPath, fallback);
}

// Parse an environment-variable string into a typed value.
// Supports std::string, bool, integral and floating-point scalars, and
// comma/space/semicolon-separated float vectors.  Returns fallback on failure.
template <typename T>
inline T parseEnvString(const std::string& s, const T& fallback) {
  if constexpr (std::is_same_v<T, std::string>) {
    return s;
  } else if constexpr (std::is_same_v<T, bool>) {
    std::string v = s;
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
    if (v == "0" || v == "false" || v == "off" || v == "no") return false;
    return fallback;
  } else if constexpr (std::is_same_v<T, std::vector<float>>) {
    std::string work = s;
    for (char& c : work) {
      if (c == ',' || c == ';' || c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    std::stringstream tokenizer(work);
    std::vector<std::string> tokens;
    std::string tok;
    while (tokenizer >> tok) {
      tokens.push_back(tok);
    }
    std::vector<float> out;
    out.reserve(tokens.size());
    for (const auto& t : tokens) {
      try {
        out.push_back(std::stof(t));
      } catch (...) {
        return fallback;
      }
    }
    return out;
  } else if constexpr (std::is_integral_v<T>) {
    try {
      return static_cast<T>(std::stoll(s));
    } catch (...) {
      return fallback;
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    try {
      return static_cast<T>(std::stod(s));
    } catch (...) {
      return fallback;
    }
  } else {
    return fallback;
  }
}

// Resolve a typed config value: environment variables have highest priority,
// then the value at dotPath in config/phoenix.json, then the supplied fallback.
template <typename T>
inline T resolveConfig(const std::string& dotPath, const T& fallback) {
  return cfgOr<T>(dotPath, fallback);
}

template <typename T, typename... EnvNames>
inline T resolveConfig(const std::string& dotPath, const T& fallback,
                       const char* first, EnvNames... rest) {
  const char* raw = std::getenv(first);
  if (raw && *raw) {
    try {
      return parseEnvString<T>(std::string(raw), fallback);
    } catch (...) {
      return fallback;
    }
  }
  if constexpr (sizeof...(rest) > 0) {
    return resolveConfig<T>(dotPath, fallback, rest...);
  }
  return cfgOr<T>(dotPath, fallback);
}

// Resolve a config value as a string.  This overload is useful for callers
// (e.g. CLI arg/env parsers) that need a scalar represented as text and may
// encounter numeric or boolean JSON values at dotPath.
inline std::string resolveConfigAsString(const std::string& dotPath,
                                         const std::string& fallback) {
  const nlohmann::json* p = PhoenixConfig::instance().getJsonPtr(dotPath);
  if (!p || p->is_null()) return fallback;
  if (p->is_string()) return p->get<std::string>();
  if (p->is_boolean()) return p->get<bool>() ? "true" : "false";
  if (p->is_number()) return p->dump();
  // Arrays/objects are returned as compact JSON for advanced callers.
  return p->dump();
}

template <typename... EnvNames>
inline std::string resolveConfigAsString(const std::string& dotPath,
                                         const std::string& fallback,
                                         const char* first, EnvNames... rest) {
  const char* raw = std::getenv(first);
  if (raw && *raw) {
    try {
      return parseEnvString<std::string>(std::string(raw), fallback);
    } catch (...) {
      return fallback;
    }
  }
  if constexpr (sizeof...(rest) > 0) {
    return resolveConfigAsString(dotPath, fallback, rest...);
  }
  return resolveConfigAsString(dotPath, fallback);
}

}  // namespace phoenix

#endif
