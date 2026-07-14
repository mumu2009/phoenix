/* model_lifecycle.hpp - Model lifecycle management */

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
