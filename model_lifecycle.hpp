/* model_lifecycle.hpp - Model lifecycle management
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

#include <mutex>
#include <string>
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace model_lifecycle {

using json = nlohmann::json;

/* Model lifecycle manager for deployment, compression, and updates */
class ModelLifecycleManager {
public:
    ModelLifecycleManager();

    /* Get current status */
    json status() const;
    /* Generate compression plan */
    json compressPlan(const json &payload);
    /* Explain model output */
    json explainOutput(const json &payload);
    /* Deploy model to target */
    json deployTarget(const json &payload);
    /* Apply online update */
    json applyOnlineUpdate(const json &payload);

private:
    /* Load manifest from disk if exists */
    void loadManifestIfExists();
    /* Persist manifest to disk */
    void persistManifest() const;

    mutable std::mutex mu_;      /* Mutex for thread safety */
    std::string activeTarget_;   /* Active deployment target */
    std::string activeVersion_;  /* Active model version */
    json compression_;           /* Compression configuration */
    json explainability_;        /* Explainability configuration */
    json deployment_;            /* Deployment configuration */
    json onlineUpdate_;          /* Online update configuration */
    json servingCluster_;        /* Serving cluster configuration */
    uint64_t updateSeq_;         /* Update sequence number */
    std::vector<json> events_;   /* Event log */
};

} // namespace model_lifecycle
