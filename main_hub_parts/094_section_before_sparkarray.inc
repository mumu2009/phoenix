// Auto-generated from main.cpp
// Segment: section_before_SparkArray
// Original lines: 11389-11389

// v8.0 SparkArray scopes: optional + multi-select (user decides which
// functional domains the ensemble applies to, or disables it entirely).
//   []  = legacy (per-feature flags decide)
//   list = whitelist ("all" matches everything; unlisted = off)
//   ["none"] = disabled everywhere
// Config errors fall back to legacy (fail-open, never break chat).
inline bool sparkArrayScopeEnabled(const std::string &scope) {
  try {
    nlohmann::json scopes =
        phoenix::cfgOr<nlohmann::json>("spark.scopes", nlohmann::json::array());
    if (!scopes.is_array() || scopes.empty()) return true;
    for (const auto &s : scopes) {
      if (s.is_string() &&
          (s.get<std::string>() == scope || s.get<std::string>() == "all"))
        return true;
    }
    return false;
  } catch (...) {
    return true;
  }
}


