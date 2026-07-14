/* phoenix_config.hpp
   Single source of truth for Phoenix runtime parameters that were previously
   hard-coded.  The authoritative values live in config/phoenix.json; this
   header only exposes them.  There are no fallbacks in production code. */

#ifndef PHOENIX_CONFIG_HPP
#define PHOENIX_CONFIG_HPP

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

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

}  // namespace phoenix

#endif
