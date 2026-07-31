/* runtime_tuned_config.hpp - Runtime overrides for auto-tuned parameters
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#ifndef PHOENIX_RUNTIME_TUNED_CONFIG_HPP
#define PHOENIX_RUNTIME_TUNED_CONFIG_HPP

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <shared_mutex>
#include <string>

namespace phoenix {
namespace tuned {

/* Thread-safe holder for runtime tuned JSON overrides. */
inline std::shared_mutex gTunedConfigMutex;
inline nlohmann::json gTunedConfig;
inline bool gTunedConfigLoaded{false};

/**
 * @brief Load tuned runtime parameters from a JSON file.
 * @param path Filesystem path to the tuned config.
 * @return true if the file was parsed successfully, false otherwise.
 */
inline bool loadTunedConfig(const std::filesystem::path &path) {
  if (!std::filesystem::exists(path)) {
    return false;
  }
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (...) {
    return false;
  }
  std::unique_lock<std::shared_mutex> lock(gTunedConfigMutex);
  gTunedConfig = std::move(doc);
  gTunedConfigLoaded = true;
  return true;
}

/**
 * @brief Retrieve a tuned value by dot-separated path with a fallback.
 * @tparam T Expected return type.
 * @param dotPath Dot-separated key path (e.g. "context.maxTokens").
 * @param fallback Value to return if the key is missing or has the wrong type.
 * @return The configured value or fallback.
 */
template <typename T>
inline T value(const std::string &dotPath, const T &fallback) {
  std::shared_lock<std::shared_mutex> lock(gTunedConfigMutex);
  if (!gTunedConfigLoaded || !gTunedConfig.is_object()) {
    return fallback;
  }
  const nlohmann::json *cur = &gTunedConfig;
  std::size_t start = 0;
  while (start <= dotPath.size()) {
    std::size_t dot = dotPath.find('.', start);
    std::string key = dotPath.substr(start, dot - start);
    if (!cur->is_object() || !cur->contains(key)) {
      return fallback;
    }
    cur = &(*cur)[key];
    if (dot == std::string::npos) {
      break;
    }
    start = dot + 1;
  }
  try {
    return cur->get<T>();
  } catch (...) {
    return fallback;
  }
}

} // namespace tuned
} // namespace phoenix

#endif // PHOENIX_RUNTIME_TUNED_CONFIG_HPP
