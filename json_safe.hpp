#ifndef JSON_SAFE_HPP
#define JSON_SAFE_HPP

#include <nlohmann/json.hpp>

// Safely extract a value from a JSON object.  Returns fallback if the key is
// missing, the JSON value is null, the container is not an object, or the type
// conversion fails.
template <typename T>
inline T safeJsonValue(const nlohmann::json &j, const std::string &key, const T &fallback) {
    if (!j.is_object()) {
        return fallback;
    }
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return fallback;
    }
    try {
        return it->get<T>();
    } catch (...) {
        return fallback;
    }
}

// Helper to safely check whether a JSON value contains a given key.
// Returns false when the value is not an object.
inline bool safeJsonContains(const nlohmann::json &j, const std::string &key) {
    return j.is_object() && j.contains(key);
}

#endif // JSON_SAFE_HPP
