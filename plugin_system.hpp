/* plugin_system.hpp - Plugin system for extensibility and dynamic loading
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

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <chrono>

namespace phoenix {
namespace plugin {

/* Plugin lifecycle states */
enum class PluginState {
    UNLOADED,    /* Plugin not loaded */
    LOADING,     /* Plugin loading in progress */
    LOADED,      /* Plugin loaded but not initialized */
    INITIALIZING, /* Plugin initializing */
    READY,       /* Plugin ready to start */
    RUNNING,     /* Plugin running */
    PAUSED,      /* Plugin paused */
    STOPPING,    /* Plugin stopping */
    ERROR,       /* Plugin in error state */
    UNLOADING    /* Plugin unloading in progress */
};

/* Plugin capability flags */
enum class PluginCapability {
    /* Data operations */
    READ_DATA,          /* Read data access */
    WRITE_DATA,         /* Write data access */
    DELETE_DATA,        /* Delete data access */

    /* System operations */
    MODIFY_CONFIG,      /* Modify system configuration */
    ACCESS_DATABASE,    /* Access database */
    ACCESS_FILESYSTEM,  /* Access filesystem */
    NETWORK_ACCESS,     /* Network access */

    /* Process operations */
    INTERCEPT_REQUEST,  /* Intercept incoming requests */
    MODIFY_REQUEST,     /* Modify incoming requests */
    INTERCEPT_RESPONSE, /* Intercept outgoing responses */
    MODIFY_RESPONSE,    /* Modify outgoing responses */

    /* LLM operations */
    MODIFY_PROMPT,      /* Modify LLM prompts */
    MODIFY_CONTEXT,     /* Modify LLM context */
    ADJUST_WEIGHTS,     /* Adjust LLM weights */

    /* Emotion operations */
    READ_EMOTION,       /* Read emotion state */
    WRITE_EMOTION,      /* Write emotion state */
    MODIFY_EMOTION,     /* Modify emotion state */

    /* Full system access */
    FULL_ACCESS         /* Full system access */
};

/* Plugin metadata */
struct PluginMetadata {
    std::string name;                                    /* Plugin name */
    std::string version;                                 /* Plugin version */
    std::string author;                                  /* Plugin author */
    std::string description;                             /* Plugin description */
    std::vector<std::string> dependencies;               /* Plugin dependencies */
    std::vector<PluginCapability> capabilities;          /* Plugin capabilities */
    std::map<std::string, std::string> config;          /* Plugin configuration */
    std::chrono::system_clock::time_point loadTime;     /* Load timestamp */
};

/* Plugin context for operations */
struct PluginContext {
    std::string sessionId;                              /* Session identifier */
    std::string requestId;                               /* Request identifier */
    std::string userId;                                  /* User identifier */
    nlohmann::json requestData;                          /* Request data */
    nlohmann::json responseData;                         /* Response data */
    std::map<std::string, std::string> headers;          /* HTTP headers */
    std::chrono::system_clock::time_point timestamp;    /* Operation timestamp */

    /* System state */
    bool isRequest{true};                                /* Is request (vs response) */
    std::string endpoint;                                /* API endpoint */
    std::string method;                                  /* HTTP method */
};

/* Plugin operation result */
struct PluginResult {
    bool success{false};                                 /* Operation success */
    std::string errorMessage;                            /* Error message if any */
    nlohmann::json data;                                 /* Result data */
    bool shouldContinue{true};                           /* Whether to continue processing chain */
    bool modifiedRequest{false};                         /* Whether request was modified */
    bool modifiedResponse{false};                        /* Whether response was modified */
};

/* Base plugin interface */
class Plugin {
public:
    virtual ~Plugin() = default;

    /* Lifecycle methods */
    virtual bool onLoad(const PluginMetadata& metadata) = 0; /* Called when plugin is loaded */
    virtual bool onInit() = 0;                              /* Called when plugin is initialized */
    virtual bool onStart() = 0;                             /* Called when plugin is started */
    virtual bool onStop() = 0;                              /* Called when plugin is stopped */
    virtual bool onUnload() = 0;                            /* Called when plugin is unloaded */

    /* Capability checks */
    virtual bool hasCapability(PluginCapability cap) const = 0; /* Check if plugin has capability */
    virtual std::vector<PluginCapability> getCapabilities() const = 0; /* Get all capabilities */

    /* Request/Response interception */
    virtual PluginResult onRequest(const PluginContext& context) {
        return PluginResult{true, "", {}, true, false, false};
    }

    virtual PluginResult onResponse(const PluginContext& context) {
        return PluginResult{true, "", {}, true, false, false};
    }

    /* Data operations */
    virtual PluginResult onReadData(const std::string& key) {
        return PluginResult{false, "not supported", {}, false, false, false};
    }

    virtual PluginResult onWriteData(const std::string& key, const nlohmann::json& value) {
        return PluginResult{false, "not supported", {}, false, false, false};
    }

    virtual PluginResult onDeleteData(const std::string& key) {
        return PluginResult{false, "not supported", {}, false, false, false};
    }

    /* Configuration */
    virtual PluginResult onConfigChange(const std::string& key, const nlohmann::json& value) {
        return PluginResult{false, "not supported", {}, false, false, false};
    }

    /* LLM operations */
    virtual PluginResult onPromptModify(const std::string& prompt) {
        return PluginResult{true, "", nlohmann::json{{"prompt", prompt}}, true, false, false};
    }

    virtual PluginResult onContextModify(const nlohmann::json& context) {
        return PluginResult{true, "", nlohmann::json{{"context", context}}, true, false, false};
    }

    /* Emotion operations */
    virtual PluginResult onEmotionRead(const std::string& sessionId) {
        return PluginResult{false, "not supported", {}, false, false, false};
    }

    virtual PluginResult onEmotionWrite(const std::string& sessionId, const nlohmann::json& emotion) {
        return PluginResult{false, "not supported", {}, false, false, false};
    }

    /* Health check */
    virtual bool isHealthy() const {
        return true;
    }

    /* Get plugin name */
    virtual std::string name() const = 0;

    /* Get plugin version */
    virtual std::string version() const = 0;

    /* Get plugin metadata */
    virtual PluginMetadata getMetadata() const = 0;

    /* Get current state */
    virtual PluginState getState() const = 0;
};

/* Plugin manager for lifecycle and orchestration */
class PluginManager {
public:
    struct Config {
        std::string pluginDirectory{"plugins"}; /* Plugin directory path */
        bool autoLoad{true};                    /* Auto-load plugins on startup */
        bool hotReload{true};                    /* Enable hot reload */
        int maxPlugins{100};                    /* Maximum number of plugins */
        std::chrono::milliseconds healthCheckInterval{5000}; /* Health check interval */
    };

    explicit PluginManager(const Config& config);
    ~PluginManager();

    /* Plugin lifecycle */
    bool loadPlugin(const std::string& pluginPath); /* Load plugin from path */
    bool unloadPlugin(const std::string& pluginName); /* Unload plugin by name */
    bool reloadPlugin(const std::string& pluginName); /* Reload plugin by name */

    /* Plugin discovery */
    std::vector<std::string> discoverPlugins() const; /* Discover available plugins */
    std::vector<std::string> getLoadedPlugins() const; /* Get loaded plugin names */

    /* Plugin execution */
    PluginResult executeRequestChain(const PluginContext& context); /* Execute request chain */
    PluginResult executeResponseChain(const PluginContext& context); /* Execute response chain */

    /* Capability-based operations */
    std::vector<std::string> getPluginsWithCapability(PluginCapability cap) const; /* Get plugins with capability */

    /* Data operations via plugins */
    PluginResult readData(const std::string& key); /* Read data via plugins */
    PluginResult writeData(const std::string& key, const nlohmann::json& value); /* Write data via plugins */
    PluginResult deleteData(const std::string& key); /* Delete data via plugins */

    /* Configuration operations */
    PluginResult setPluginConfig(const std::string& pluginName,
                                const std::string& key,
                                const nlohmann::json& value); /* Set plugin configuration */

    /* Health monitoring */
    std::map<std::string, bool> getPluginHealth() const; /* Get plugin health status */
    void startHealthCheck(); /* Start health check thread */
    void stopHealthCheck(); /* Stop health check thread */

    /* Plugin registration (for dynamic plugins) */
    bool registerPlugin(std::shared_ptr<Plugin> plugin); /* Register plugin instance */
    bool unregisterPlugin(const std::string& pluginName); /* Unregister plugin by name */

    /* Get plugin instance */
    std::shared_ptr<Plugin> getPlugin(const std::string& pluginName) const; /* Get plugin by name */

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* Plugin registry for factory pattern */
class PluginRegistry {
public:
    using PluginFactory = std::function<std::shared_ptr<Plugin>()>;

    static PluginRegistry& instance(); /* Get singleton instance */

    void registerFactory(const std::string& pluginType, PluginFactory factory); /* Register plugin factory */
    std::shared_ptr<Plugin> createPlugin(const std::string& pluginType); /* Create plugin from factory */
    std::vector<std::string> getRegisteredTypes() const; /* Get registered plugin types */

private:
    PluginRegistry() = default;
    std::map<std::string, PluginFactory> factories_; /* Registered factories */
    mutable std::mutex mutex_; /* Mutex for thread safety */
};

/* Plugin configuration manager */
class PluginConfigManager {
public:
    explicit PluginConfigManager(const std::string& configPath);

    bool loadConfig(); /* Load configuration from file */
    bool saveConfig(); /* Save configuration to file */

    nlohmann::json getPluginConfig(const std::string& pluginName) const; /* Get plugin configuration */
    bool setPluginConfig(const std::string& pluginName, const nlohmann::json& config); /* Set plugin configuration */

    std::vector<std::string> getConfiguredPlugins() const; /* Get configured plugin names */

private:
    bool loadConfigUnlocked(); /* Load configuration while mutex is already held */
    bool saveConfigUnlocked(); /* Save configuration while mutex is already held */

    std::string configPath_; /* Configuration file path */
    nlohmann::json config_;  /* Configuration data */
    mutable std::mutex mutex_; /* Mutex for thread safety */
};

/* Plugin execution context with timing and metrics */
struct PluginExecutionMetrics {
    std::string pluginName;                       /* Plugin name */
    std::chrono::microseconds executionTime;      /* Execution time */
    bool success;                                 /* Execution success */
    std::string errorMessage;                    /* Error message if any */
    size_t memoryUsed;                            /* Memory used in bytes */

    nlohmann::json toJson() const {
        return {
            {"pluginName", pluginName},
            {"executionTimeUs", executionTime.count()},
            {"success", success},
            {"errorMessage", errorMessage},
            {"memoryUsed", memoryUsed}
        };
    }
};

/* Plugin execution profiler */
class PluginProfiler {
public:
    void startProfiling(const std::string& pluginName); /* Start profiling plugin */
    void endProfiling(const std::string& pluginName, bool success, const std::string& error = ""); /* End profiling plugin */

    std::vector<PluginExecutionMetrics> getMetrics() const; /* Get all metrics */
    void clearMetrics(); /* Clear all metrics */

    nlohmann::json getMetricsJson() const; /* Get metrics as JSON */

private:
    std::map<std::string, PluginExecutionMetrics> metrics_; /* Plugin metrics */
    std::map<std::string, std::chrono::system_clock::time_point> startTimes_; /* Start times */
    mutable std::mutex mutex_; /* Mutex for thread safety */
};

/* Helper macro for plugin registration */
#define REGISTER_PLUGIN(PluginClass, PluginType) \
    namespace { \
        struct PluginClass##Registrar { \
            PluginClass##Registrar() { \
                phoenix::plugin::PluginRegistry::instance().registerFactory( \
                    PluginType, \
                    []() -> std::shared_ptr<phoenix::plugin::Plugin> { \
                        return std::make_shared<PluginClass>(); \
                    } \
                ); \
            } \
        }; \
        static PluginClass##Registrar g_##PluginClass##Registrar; \
    }

} // namespace plugin
} // namespace phoenix
