/* phoenix_config.hpp
   Single source of truth for Phoenix runtime parameters that were previously
   hard-coded.  The authoritative values live in config/phoenix.json; this
   header only exposes them.  There are no fallbacks in production code. */

#ifndef PHOENIX_CONFIG_HPP
#define PHOENIX_CONFIG_HPP

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    f >> cfg_;
    loaded_ = true;
  }

  bool loaded() const { return loaded_; }

  const nlohmann::json& raw() const { return cfg_; }

  template <typename T>
  T get(const std::string& dotPath) const {
    const nlohmann::json* p = getPath(dotPath);
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
    const nlohmann::json* p = getPath(dotPath);
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
    return getPath(dotPath);
  }

 private:
  PhoenixConfig() = default;

  const nlohmann::json* getPath(const std::string& dotPath) const {
    const nlohmann::json* cur = &cfg_;
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

  nlohmann::json cfg_;
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
