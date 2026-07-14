/* module_mount.hpp - Module factory registry for pluggable components
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

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

class ControllerPoolBase;
class RuntimeBase;
class ShardManager;
class RotationManager;
class Database079;
struct Config;

/* Redis synchronizer interface for distributed state sync */
class IRedisSynchronizer {
public:
    virtual ~IRedisSynchronizer() = default;
    virtual void start() = 0;                                    /* Start synchronizer */
    virtual void publish(const nlohmann::json &snapshot) = 0;   /* Publish snapshot */
    virtual const std::string &channel() const = 0;             /* Get channel name */
};

/* Study engine interface for document ingestion */
class IStudyEngine {
public:
    virtual ~IStudyEngine() = default;
    virtual void start() = 0;                                    /* Start study engine */
    virtual void enqueueDocument(const nlohmann::json &doc) = 0; /* Enqueue document */
    virtual nlohmann::json status() const = 0;                  /* Get status */
    virtual bool running() const = 0;                            /* Check if running */
};

/* Snapshot manager interface for state persistence */
class ISnapshotManager {
public:
    virtual ~ISnapshotManager() = default;
    virtual std::vector<std::string> list() const = 0;          /* List snapshots */
    virtual std::filesystem::path create(const std::string &name = "auto") = 0; /* Create snapshot */
    virtual nlohmann::json restore(const std::string &id) = 0;   /* Restore snapshot */
    virtual void remove(const std::string &id) = 0;             /* Remove snapshot */
};

/* Persona forest averager interface for persona selection */
class IPersonaForestAverager {
public:
    virtual ~IPersonaForestAverager() = default;
    virtual int maxHistory() const = 0;                          /* Get max history */
    virtual std::optional<nlohmann::json> pick(const nlohmann::json &payload,
                                               const nlohmann::json &layerResults,
                                               const std::vector<float> &requestEmbedding,
                                               const std::vector<nlohmann::json> &history) = 0; /* Pick persona */
};

/* Spark array interface for layer dispatch */
class ISparkArray {
public:
    virtual ~ISparkArray() = default;
    virtual nlohmann::json getLayers() const = 0;              /* Get layers */
    virtual void updateLayers(const nlohmann::json &layers) = 0; /* Update layers */
    virtual nlohmann::json dispatch(const nlohmann::json &payload, const nlohmann::json &options = nlohmann::json::object()) = 0; /* Dispatch */
    virtual nlohmann::json dispatchBig(const nlohmann::json &payload, const nlohmann::json &options) = 0; /* Dispatch big */
    virtual const std::vector<nlohmann::json> &history() const = 0; /* Get history */
};

/* Reinforcement learner interface */
class IReinforcementLearner {
public:
    virtual ~IReinforcementLearner() = default;
    virtual nlohmann::json learn(int cycles) = 0;               /* Learn for cycles */
    virtual nlohmann::json latest() const = 0;                  /* Get latest results */
    virtual nlohmann::json refreshTests(const std::filesystem::path &testsDir) = 0; /* Refresh tests */
    virtual nlohmann::json setTestsDir(const std::filesystem::path &testsDir) = 0; /* Set tests dir */
};

/* Adversarial learner interface */
class IAdversarialLearner {
public:
    virtual ~IAdversarialLearner() = default;
    virtual nlohmann::json attackAndDefend(const nlohmann::json &samples) = 0; /* Attack and defend */
    virtual nlohmann::json latest() const = 0;                  /* Get latest results */
};

/* GNN genetic algorithm learner interface */
class IGnnGaLearner {
public:
    virtual ~IGnnGaLearner() = default;
    virtual nlohmann::json evolve(int generations,
                                  int population,
                                  const std::vector<std::string> &samples,
                                  double residualWeight,
                                  double mutationRate,
                                  double mutationScale) = 0; /* Evolve */
    virtual nlohmann::json latest() const = 0;                  /* Get latest results */
};

/* Gateway server interface */
class IGatewayServer {
public:
    virtual ~IGatewayServer() = default;
    virtual void listen() = 0;                                   /* Start listening */
    virtual void warmupLearning() = 0;                          /* Warmup learning */
    virtual void bootstrapTransformerFromCorpus(int maxDocs = 48) = 0; /* Bootstrap transformer */
};

namespace module_mount {

/* Factory function types */
using RedisSynchronizerFactory = std::function<std::shared_ptr<IRedisSynchronizer>(std::shared_ptr<ControllerPoolBase>, const std::string &, const std::string &)>;
using StudyEngineFactory = std::function<std::shared_ptr<IStudyEngine>(std::shared_ptr<ControllerPoolBase>, std::shared_ptr<IRedisSynchronizer>)>;
using SnapshotManagerFactory = std::function<std::shared_ptr<ISnapshotManager>(std::shared_ptr<RuntimeBase>, const std::filesystem::path &)>;
using PersonaForestAveragerFactory = std::function<std::shared_ptr<IPersonaForestAverager>(const nlohmann::json &)>;
using SparkArrayFactory = std::function<std::shared_ptr<ISparkArray>(std::shared_ptr<ControllerPoolBase>, std::shared_ptr<ShardManager>, const nlohmann::json &)>;
using ReinforcementLearnerFactory = std::function<std::shared_ptr<IReinforcementLearner>(std::shared_ptr<ControllerPoolBase>, const std::filesystem::path &)>;
using AdversarialLearnerFactory = std::function<std::shared_ptr<IAdversarialLearner>(std::shared_ptr<ControllerPoolBase>)>;
using GnnGaLearnerFactory = std::function<std::shared_ptr<IGnnGaLearner>(std::shared_ptr<ControllerPoolBase>, const std::filesystem::path &)>;
using GatewayServerFactory = std::function<std::shared_ptr<IGatewayServer>(std::shared_ptr<ControllerPoolBase>,
                                                                           std::shared_ptr<ISnapshotManager>,
                                                                           std::shared_ptr<IRedisSynchronizer>,
                                                                           std::shared_ptr<RotationManager>,
                                                                           std::shared_ptr<IStudyEngine>,
                                                                           std::shared_ptr<Database079>,
                                                                           const Config &)>;

/* Module registry for factory functions */
class ModuleRegistry {
public:
    static ModuleRegistry &instance() {
        static ModuleRegistry inst;
        return inst;
    }

    /* Set factory functions */
    void setRedisSynchronizerFactory(RedisSynchronizerFactory f) { std::lock_guard<std::mutex> lk(mu_); redisSyncFactory_ = std::move(f); }
    void setStudyEngineFactory(StudyEngineFactory f) { std::lock_guard<std::mutex> lk(mu_); studyFactory_ = std::move(f); }
    void setSnapshotManagerFactory(SnapshotManagerFactory f) { std::lock_guard<std::mutex> lk(mu_); snapshotFactory_ = std::move(f); }
    void setPersonaForestAveragerFactory(PersonaForestAveragerFactory f) { std::lock_guard<std::mutex> lk(mu_); personaFactory_ = std::move(f); }
    void setSparkArrayFactory(SparkArrayFactory f) { std::lock_guard<std::mutex> lk(mu_); sparkFactory_ = std::move(f); }
    void setReinforcementLearnerFactory(ReinforcementLearnerFactory f) { std::lock_guard<std::mutex> lk(mu_); rlFactory_ = std::move(f); }
    void setAdversarialLearnerFactory(AdversarialLearnerFactory f) { std::lock_guard<std::mutex> lk(mu_); advFactory_ = std::move(f); }
    void setGnnGaLearnerFactory(GnnGaLearnerFactory f) { std::lock_guard<std::mutex> lk(mu_); gnnGaFactory_ = std::move(f); }
    void setGatewayServerFactory(GatewayServerFactory f) { std::lock_guard<std::mutex> lk(mu_); gatewayFactory_ = std::move(f); }

    /* Get factory functions */
    RedisSynchronizerFactory redisSynchronizerFactory() const { std::lock_guard<std::mutex> lk(mu_); return redisSyncFactory_; }
    StudyEngineFactory studyEngineFactory() const { std::lock_guard<std::mutex> lk(mu_); return studyFactory_; }
    SnapshotManagerFactory snapshotManagerFactory() const { std::lock_guard<std::mutex> lk(mu_); return snapshotFactory_; }
    PersonaForestAveragerFactory personaForestAveragerFactory() const { std::lock_guard<std::mutex> lk(mu_); return personaFactory_; }
    SparkArrayFactory sparkArrayFactory() const { std::lock_guard<std::mutex> lk(mu_); return sparkFactory_; }
    ReinforcementLearnerFactory reinforcementLearnerFactory() const { std::lock_guard<std::mutex> lk(mu_); return rlFactory_; }
    AdversarialLearnerFactory adversarialLearnerFactory() const { std::lock_guard<std::mutex> lk(mu_); return advFactory_; }
    GnnGaLearnerFactory gnnGaLearnerFactory() const { std::lock_guard<std::mutex> lk(mu_); return gnnGaFactory_; }
    GatewayServerFactory gatewayServerFactory() const { std::lock_guard<std::mutex> lk(mu_); return gatewayFactory_; }

private:
    ModuleRegistry() = default;

    mutable std::mutex mu_;              /* Mutex for thread safety */
    RedisSynchronizerFactory redisSyncFactory_; /* Redis synchronizer factory */
    StudyEngineFactory studyFactory_;   /* Study engine factory */
    SnapshotManagerFactory snapshotFactory_; /* Snapshot manager factory */
    PersonaForestAveragerFactory personaFactory_; /* Persona forest averager factory */
    SparkArrayFactory sparkFactory_;     /* Spark array factory */
    ReinforcementLearnerFactory rlFactory_; /* Reinforcement learner factory */
    AdversarialLearnerFactory advFactory_; /* Adversarial learner factory */
    GnnGaLearnerFactory gnnGaFactory_;  /* GNN GA learner factory */
    GatewayServerFactory gatewayFactory_; /* Gateway server factory */
};

/* Inline registration functions */
inline void registerRedisSynchronizerFactory(const RedisSynchronizerFactory &f) { ModuleRegistry::instance().setRedisSynchronizerFactory(f); }
inline void registerStudyEngineFactory(const StudyEngineFactory &f) { ModuleRegistry::instance().setStudyEngineFactory(f); }
inline void registerSnapshotManagerFactory(const SnapshotManagerFactory &f) { ModuleRegistry::instance().setSnapshotManagerFactory(f); }
inline void registerPersonaForestAveragerFactory(const PersonaForestAveragerFactory &f) { ModuleRegistry::instance().setPersonaForestAveragerFactory(f); }
inline void registerSparkArrayFactory(const SparkArrayFactory &f) { ModuleRegistry::instance().setSparkArrayFactory(f); }
inline void registerReinforcementLearnerFactory(const ReinforcementLearnerFactory &f) { ModuleRegistry::instance().setReinforcementLearnerFactory(f); }
inline void registerAdversarialLearnerFactory(const AdversarialLearnerFactory &f) { ModuleRegistry::instance().setAdversarialLearnerFactory(f); }
inline void registerGnnGaLearnerFactory(const GnnGaLearnerFactory &f) { ModuleRegistry::instance().setGnnGaLearnerFactory(f); }
inline void registerGatewayServerFactory(const GatewayServerFactory &f) { ModuleRegistry::instance().setGatewayServerFactory(f); }

/* Convenience create* helpers used by unit tests.
   Returns nullptr if the factory has not been registered (e.g. in pure unit test
   environments where the implementation TU is not linked). */
inline std::shared_ptr<IReinforcementLearner>
createReinforcementLearner(std::shared_ptr<ControllerPoolBase> pool,
                           const std::filesystem::path &testsDir = {}) {
    auto f = ModuleRegistry::instance().reinforcementLearnerFactory();
    return f ? f(std::move(pool), testsDir) : nullptr;
}

inline std::shared_ptr<IAdversarialLearner>
createAdversarialLearner(std::shared_ptr<ControllerPoolBase> pool) {
    auto f = ModuleRegistry::instance().adversarialLearnerFactory();
    return f ? f(std::move(pool)) : nullptr;
}

inline std::shared_ptr<IGnnGaLearner>
createGnnGaLearner(std::shared_ptr<ControllerPoolBase> pool,
                   const std::filesystem::path &testsDir = {}) {
    auto f = ModuleRegistry::instance().gnnGaLearnerFactory();
    return f ? f(std::move(pool), testsDir) : nullptr;
}

/* Backward-compat alias: test files used IGnnGALearner (uppercase GA).
   The canonical name is IGnnGaLearner. */
using IGnnGALearner = IGnnGaLearner;
inline std::shared_ptr<IGnnGALearner>
createGnnGALearner(std::shared_ptr<ControllerPoolBase> pool,
                   const std::filesystem::path &testsDir = {}) {
    return createGnnGaLearner(std::move(pool), testsDir);
}

} // namespace module_mount

/* json convenience alias for test files that use bare 'json' */
using json = nlohmann::json;
