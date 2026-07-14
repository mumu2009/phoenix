/* component_config.hpp - Component configuration parsing and management
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

#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace component_config {

using json = nlohmann::json;
using OverrideMap = std::map<std::string, std::string>;

/* Trim whitespace from both ends of a string */
inline std::string trimCopy(const std::string &value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(start, end - start);
}

/* Convert string to lowercase */
inline std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

/* Normalize configuration key to a standard format.
   Converts separators to dots/hyphens and removes consecutive separators. */
inline std::string normalizeKey(const std::string &raw) {
    std::string out;
    out.reserve(raw.size());
    char previous = '\0';
    for (unsigned char ch : raw) {
        if (std::isalnum(ch) != 0) {
            out.push_back(static_cast<char>(std::tolower(ch)));
            previous = out.back();
            continue;
        }
        char mapped = '\0';
        if (ch == '.' || ch == '/' || ch == '\\' || ch == ':') {
            mapped = '.';
        } else if (ch == '-' || ch == '_' || std::isspace(ch) != 0) {
            mapped = '-';
        }
        if (mapped == '\0') {
            continue;
        }
        if (out.empty() || previous == mapped) {
            continue;
        }
        out.push_back(mapped);
        previous = mapped;
    }
    while (!out.empty() && (out.back() == '.' || out.back() == '-')) {
        out.pop_back();
    }
    return out;
}

/* Parse boolean string value */
inline bool parseBoolString(const std::string &raw, bool &value) {
    const std::string lowered = lowerCopy(trimCopy(raw));
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on" || lowered == "enabled") {
        value = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off" || lowered == "disabled") {
        value = false;
        return true;
    }
    return false;
}

/* Merge override maps, with incoming values taking precedence */
inline void mergeOverrides(OverrideMap &target, const OverrideMap &incoming) {
    for (const auto &entry : incoming) {
        target[entry.first] = entry.second;
    }
}

/* Parse CLI specification string into override map.
   Supports comma and semicolon separators, key=value format. */
inline OverrideMap parseCliSpec(const std::string &spec) {
    OverrideMap out;
    std::string normalized = spec;
    for (char &ch : normalized) {
        if (ch == ';' || ch == '\n' || ch == '\r') {
            ch = ',';
        }
    }
    std::stringstream stream(normalized);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trimCopy(item);
        if (item.empty()) {
            continue;
        }
        const auto pos = item.find('=');
        if (pos == std::string::npos) {
            out[normalizeKey(item)] = "true";
            continue;
        }
        const std::string key = normalizeKey(item.substr(0, pos));
        const std::string value = trimCopy(item.substr(pos + 1));
        if (!key.empty()) {
            out[key] = value.empty() ? std::string("true") : value;
        }
    }
    return out;
}

/* Flatten JSON value to dot-separated key-value map */
inline void flattenJsonValue(const json &value, const std::string &prefix, OverrideMap &out) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            std::string next = normalizeKey(it.key());
            if (next.empty()) {
                continue;
            }
            if (!prefix.empty()) {
                next = prefix + "." + next;
            }
            flattenJsonValue(it.value(), next, out);
        }
        return;
    }
    if (prefix.empty()) {
        return;
    }
    if (value.is_boolean()) {
        out[prefix] = value.get<bool>() ? "true" : "false";
    } else if (value.is_string()) {
        out[prefix] = value.get<std::string>();
    } else if (value.is_number_integer()) {
        out[prefix] = std::to_string(value.get<long long>());
    } else if (value.is_number_unsigned()) {
        out[prefix] = std::to_string(value.get<unsigned long long>());
    } else if (value.is_number_float()) {
        std::ostringstream oss;
        oss << value.get<double>();
        out[prefix] = oss.str();
    } else if (!value.is_null()) {
        out[prefix] = value.dump();
    }
}

/* Parse JSON text into override map */
inline OverrideMap parseJsonText(const std::string &raw) {
    OverrideMap out;
    json doc = json::parse(raw, nullptr, false);
    if (doc.is_discarded()) {
        throw std::runtime_error("invalid component JSON");
    }
    const json &root = (doc.is_object() && doc.contains("components") && doc["components"].is_object()) ? doc["components"] : doc;
    flattenJsonValue(root, std::string(), out);
    return out;
}

/* Parse XML attributes string into key-value map */
inline std::map<std::string, std::string> parseXmlAttributes(const std::string &attrs) {
    std::map<std::string, std::string> out;
    const std::regex attrRegex("([A-Za-z0-9_.:-]+)\\s*=\\s*(\"([^\"]*)\"|'([^']*)')", std::regex::icase);
    auto begin = std::sregex_iterator(attrs.begin(), attrs.end(), attrRegex);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const auto &match = *it;
        const std::string key = normalizeKey(match[1].str());
        const std::string value = match[3].matched ? match[3].str() : match[4].str();
        if (!key.empty()) {
            out[key] = value;
        }
    }
    return out;
}

/* Parse XML text into override map.
   Extracts component/option tags and their attributes. */
inline OverrideMap parseXmlText(const std::string &raw) {
    OverrideMap out;
    const std::regex tagRegex(R"(<\s*(component|option)\b([^>]*)/?>)", std::regex::icase);
    auto begin = std::sregex_iterator(raw.begin(), raw.end(), tagRegex);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const auto attrs = parseXmlAttributes((*it)[2].str());
        std::string key;
        auto pathIt = attrs.find("path");
        if (pathIt != attrs.end()) {
            key = normalizeKey(pathIt->second);
        } else {
            auto keyIt = attrs.find("key");
            if (keyIt != attrs.end()) {
                key = normalizeKey(keyIt->second);
            } else {
                auto nameIt = attrs.find("name");
                if (nameIt != attrs.end()) {
                    key = normalizeKey(nameIt->second);
                }
            }
        }
        if (key.empty()) {
            continue;
        }
        std::string value;
        auto valueIt = attrs.find("value");
        if (valueIt != attrs.end()) {
            value = valueIt->second;
        } else {
            auto enabledIt = attrs.find("enabled");
            if (enabledIt != attrs.end()) {
                value = enabledIt->second;
            } else {
                auto profileIt = attrs.find("profile");
                if (profileIt != attrs.end()) {
                    value = profileIt->second;
                } else {
                    auto modeIt = attrs.find("mode");
                    if (modeIt != attrs.end()) {
                        value = modeIt->second;
                    }
                }
            }
        }
        out[key] = value.empty() ? std::string("true") : value;
    }
    return out;
}

/* Load component configuration from file.
   Auto-detects format from extension or uses format hint. */
inline OverrideMap loadFromFile(const std::filesystem::path &path, const std::string &formatHint = std::string()) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open component config: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::string format = lowerCopy(trimCopy(formatHint));
    if (format.empty()) {
        format = lowerCopy(path.extension().string());
        if (!format.empty() && format[0] == '.') {
            format.erase(format.begin());
        }
    }
    if (format == "xml") {
        return parseXmlText(buffer.str());
    }
    return parseJsonText(buffer.str());
}

/* Convert text string to appropriate JSON scalar type */
inline json scalarFromText(const std::string &raw) {
    const std::string value = trimCopy(raw);
    bool boolValue = false;
    if (parseBoolString(value, boolValue)) {
        return boolValue;
    }
    if (!value.empty()) {
        char *endPtr = nullptr;
        const double parsed = std::strtod(value.c_str(), &endPtr);
        if (endPtr != value.c_str() && endPtr != nullptr && *endPtr == '\0') {
            if (value.find('.') == std::string::npos && value.find('e') == std::string::npos && value.find('E') == std::string::npos) {
                return static_cast<long long>(parsed);
            }
            return parsed;
        }
    }
    return value;
}

/* Split dot-separated path into segments */
inline std::vector<std::string> splitPath(const std::string &key) {
    std::vector<std::string> out;
    std::stringstream stream(key);
    std::string item;
    while (std::getline(stream, item, '.')) {
        item = normalizeKey(item);
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

/* Build nested JSON from flat override map */
inline json buildNestedJson(const OverrideMap &overrides) {
    json out = json::object();
    for (const auto &entry : overrides) {
        const auto path = splitPath(entry.first);
        if (path.empty()) {
            continue;
        }
        json *cursor = &out;
        for (std::size_t index = 0; index + 1 < path.size(); ++index) {
            const auto &segment = path[index];
            if (!cursor->contains(segment) || !(*cursor)[segment].is_object()) {
                (*cursor)[segment] = json::object();
            }
            cursor = &((*cursor)[segment]);
        }
        (*cursor)[path.back()] = scalarFromText(entry.second);
    }
    return out;
}

} // namespace component_config