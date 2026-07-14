/* physics_world.hpp - Physics world and Bullet3 runtime integration
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

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace physics_world {

using json = nlohmann::json;
namespace fs = std::filesystem;

/* Get bundled earth heightfield URI */
inline std::string bundledEarthHeightfieldUri() {
    return "static/earth_maps/china_relief_heightfield.json";
}

/* Get preferred earth map format */
inline std::string preferredEarthMapFormat() {
    return "heightfield";
}

/* Get preferred earth coordinate frame */
inline std::string preferredEarthCoordinateFrame() {
    return "wgs84-local-enu";
}

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

/* Truncate text to specified length with ellipsis */
inline std::string truncateText(const std::string &value, std::size_t limit) {
    if (value.size() <= limit) {
        return value;
    }
    if (limit < 4) {
        return value.substr(0, limit);
    }
    return value.substr(0, limit - 3) + "...";
}

/* Detect source type from URI */
inline std::string detectSourceType(const std::string &sourceUri) {
    const std::string lowered = lowerCopy(trimCopy(sourceUri));
    if (lowered.empty()) {
        return "unspecified";
    }
    if (lowered.rfind("http://", 0) == 0 || lowered.rfind("https://", 0) == 0) {
        return "remote-stream";
    }
    if (lowered.rfind("s3://", 0) == 0 || lowered.rfind("gs://", 0) == 0) {
        return "object-store";
    }
    return "local-file";
}

/* Normalize earth map import request */
inline json normalizeEarthMapImportRequest(const json &rawRequest) {
    json request = json::object();
    if (rawRequest.is_object()) {
        request = rawRequest;
    } else if (rawRequest.is_string()) {
        request["sourceUri"] = rawRequest.get<std::string>();
    }

    const bool enabled = request.value("enabled", false) ||
                         (request.contains("sourceUri") && request["sourceUri"].is_string() && !trimCopy(request["sourceUri"].get<std::string>()).empty());
    std::string sourceUri = trimCopy(request.value("sourceUri", request.value("uri", std::string())));
    const std::string format = lowerCopy(trimCopy(request.value("format", preferredEarthMapFormat())));
    const std::string coordinateFrame = trimCopy(request.value("coordinateFrame", preferredEarthCoordinateFrame()));
    const std::string regionLabel = trimCopy(request.value("regionLabel", std::string("global-earth")));
    const int lod = std::max(0, request.value("lod", 6));
    const double metersPerCell = std::max(1.0, request.value("metersPerCell", 750.0));

    if (enabled && sourceUri.empty() && format == "heightfield") {
        sourceUri = bundledEarthHeightfieldUri();
    }

    json geoBounds = json{{"latMin", -90.0}, {"latMax", 90.0}, {"lonMin", -180.0}, {"lonMax", 180.0}};
    if (request.contains("geoBounds") && request["geoBounds"].is_object()) {
        const auto &rawBounds = request["geoBounds"];
        geoBounds["latMin"] = rawBounds.value("latMin", geoBounds["latMin"].get<double>());
        geoBounds["latMax"] = rawBounds.value("latMax", geoBounds["latMax"].get<double>());
        geoBounds["lonMin"] = rawBounds.value("lonMin", geoBounds["lonMin"].get<double>());
        geoBounds["lonMax"] = rawBounds.value("lonMax", geoBounds["lonMax"].get<double>());
    }

    json importPlan = json::array({
        "normalize geodetic coordinates into a physics-friendly frame",
        "load or derive a Bullet-compatible heightfield proxy",
        "segment the globe into LOD-managed training regions",
        "bind rigid bodies to local tangent frames before simulation"
    });

    const std::string summary = enabled
        ? truncateText("earth map import prepared from " + (sourceUri.empty() ? std::string("an unspecified source") : sourceUri) +
                           " using format " + format + " in " + coordinateFrame +
                           " with lod " + std::to_string(lod),
                       220)
        : std::string("earth map import disabled");

    return json{{"enabled", enabled},
                {"sourceUri", sourceUri},
                {"sourceType", detectSourceType(sourceUri)},
                {"format", format.empty() ? preferredEarthMapFormat() : format},
                {"coordinateFrame", coordinateFrame.empty() ? preferredEarthCoordinateFrame() : coordinateFrame},
                {"regionLabel", regionLabel.empty() ? std::string("global-earth") : regionLabel},
                {"lod", lod},
                {"metersPerCell", metersPerCell},
                {"geoBounds", geoBounds},
                {"importPlan", importPlan},
                {"status", enabled ? (sourceUri.empty() ? std::string("awaiting-source") : std::string("configured")) : std::string("disabled")},
                {"summary", summary}};
}

/* Get MinGW library candidate paths for Bullet3 */
inline std::vector<fs::path> bullet3MinGwLibCandidates(const fs::path &root) {
    return {
        root / "build_cmake" / "lib",
        root / "build_cmake" / "lib" / "Release",
        root / "build" / "lib",
        root / "build" / "lib" / "Release",
        root / "lib"
    };
}

/* Get MSVC library candidate paths for Bullet3 */
inline std::vector<fs::path> bullet3MsvcLibCandidates(const fs::path &root) {
    return {
        root / "build_cmake" / "lib" / "Release",
        root / "build" / "lib" / "Release",
        root / "lib"
    };
}

/* Check if directory has MinGW Bullet3 libraries */
inline bool hasBullet3MinGwLibs(const fs::path &dir) {
    return fs::exists(dir / "libBulletDynamics.a") &&
           fs::exists(dir / "libBulletCollision.a") &&
           fs::exists(dir / "libLinearMath.a");
}

/* Check if directory has MSVC Bullet3 libraries */
inline bool hasBullet3MsvcLibs(const fs::path &dir) {
    return fs::exists(dir / "BulletDynamics.lib") &&
           fs::exists(dir / "BulletCollision.lib") &&
           fs::exists(dir / "LinearMath.lib");
}

/* Inspect Bullet3 runtime configuration */
inline json inspectBullet3Runtime(const fs::path &root) {
    const bool repoPresent = fs::exists(root / "README.md") && fs::exists(root / "src");
    const bool headersPresent = fs::exists(root / "src" / "btBulletDynamicsCommon.h") &&
                                fs::exists(root / "src" / "BulletCollision") &&
                                fs::exists(root / "src" / "LinearMath");
    const bool importersPresent = fs::exists(root / "examples" / "Importers");
    const bool dataAssetsPresent = fs::exists(root / "data");

    fs::path minGwLibDir;
    for (const auto &candidate : bullet3MinGwLibCandidates(root)) {
        if (hasBullet3MinGwLibs(candidate)) {
            minGwLibDir = candidate;
            break;
        }
    }

    fs::path msvcLibDir;
    for (const auto &candidate : bullet3MsvcLibCandidates(root)) {
        if (hasBullet3MsvcLibs(candidate)) {
            msvcLibDir = candidate;
            break;
        }
    }

    std::string runtimeMode = "unavailable";
    if (repoPresent && headersPresent) {
        runtimeMode = "repo-only";
    }
    if (!minGwLibDir.empty()) {
        runtimeMode = "native-ready-mingw";
    } else if (!msvcLibDir.empty()) {
        runtimeMode = "native-ready-msvc-artifacts";
    }

#if defined(AI_HAVE_BULLET3_NATIVE)
    if (repoPresent && headersPresent) {
        runtimeMode = "native-embedded-source";
    }
#endif

    return json{{"root", root.string()},
                {"repoPresent", repoPresent},
                {"headersPresent", headersPresent},
                {"importersPresent", importersPresent},
                {"dataAssetsPresent", dataAssetsPresent},
                {"minGwLibDir", minGwLibDir.empty() ? std::string() : minGwLibDir.string()},
                {"msvcLibDir", msvcLibDir.empty() ? std::string() : msvcLibDir.string()},
                {"preferredEarthFormat", preferredEarthMapFormat()},
                {"preferredCoordinateFrame", preferredEarthCoordinateFrame()},
                {"bundledEarthHeightfieldUri", bundledEarthHeightfieldUri()},
#if defined(AI_HAVE_BULLET3_NATIVE)
                {"nativeCompiled", true},
#else
                {"nativeCompiled", false},
#endif
                {"runtimeMode", runtimeMode},
                {"summary", repoPresent
                                ? truncateText("bullet3 repo detected; runtime mode is " + runtimeMode, 180)
                                : std::string("bullet3 repo not found")}};
}

} // namespace physics_world