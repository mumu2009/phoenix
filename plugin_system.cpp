/* plugin_system.cpp - Plugin system implementation
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

#include "plugin_system.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <algorithm>
#include <iostream>

namespace phoenix {
namespace plugin {

/* Plugin manager implementation */
struct PluginManager::Impl {
    Config config;
    std::map<std::string, std::shared_ptr<Plugin>> plugins;
    std::map<std::string, PluginState> pluginStates;
    std::map<std::string, PluginMetadata> pluginMetadata;
    PluginProfiler profiler;
    std::unique_ptr<PluginConfigManager> configManager;
    std::thread healthCheckThread;
    std::atomic<bool> healthCheckRunning{false};
    mutable std::mutex mutex;
    
    Impl(const Config& cfg) : config(cfg) {
        configManager = std::make_unique<PluginConfigManager>(
            config.pluginDirectory + "/plugin_config.json"
        );
    }
};

PluginManager::PluginManager(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

PluginManager::~PluginManager() {
    stopHealthCheck();
    
    // Unload all plugins
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto& pair : impl_->plugins) {
        try {
            pair.second->onStop();
            pair.second->onUnload();
        } catch (const std::exception& e) {
            std::cerr << "[PluginManager] Error unloading plugin " << pair.first 
                      << ": " << e.what() << std::endl;
        }
    }
}

bool PluginManager::loadPlugin(const std::string& pluginPath) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    // Extract plugin name from path
    std::filesystem::path path(pluginPath);
    std::string pluginName = path.stem().string();
    
    // Check if already loaded
    if (impl_->plugins.find(pluginName) != impl_->plugins.end()) {
        std::cerr << "[PluginManager] Plugin already loaded: " << pluginName << std::endl;
        return false;
    }
    
    // Load plugin configuration
    nlohmann::json pluginConfig = impl_->configManager->getPluginConfig(pluginName);
    
    // Create plugin metadata
    PluginMetadata metadata;
    metadata.name = pluginName;
    metadata.version = pluginConfig.value("version", "1.0.0");
    metadata.author = pluginConfig.value("author", "unknown");
    metadata.description = pluginConfig.value("description", "");
    metadata.config = pluginConfig;
    metadata.loadTime = std::chrono::system_clock::now();
    
    // Set plugin state
    impl_->pluginStates[pluginName] = PluginState::LOADING;
    
    // Create plugin instance using registry
    std::string pluginType = pluginConfig.value("type", pluginName);
    auto plugin = PluginRegistry::instance().createPlugin(pluginType);
    
    if (!plugin) {
        std::cerr << "[PluginManager] Failed to create plugin: " << pluginName << std::endl;
        impl_->pluginStates[pluginName] = PluginState::ERROR;
        return false;
    }
    
    // Load plugin
    impl_->pluginStates[pluginName] = PluginState::LOADED;
    if (!plugin->onLoad(metadata)) {
        std::cerr << "[PluginManager] Plugin onLoad failed: " << pluginName << std::endl;
        impl_->pluginStates[pluginName] = PluginState::ERROR;
        return false;
    }
    
    // Initialize plugin
    impl_->pluginStates[pluginName] = PluginState::INITIALIZING;
    if (!plugin->onInit()) {
        std::cerr << "[PluginManager] Plugin onInit failed: " << pluginName << std::endl;
        impl_->pluginStates[pluginName] = PluginState::ERROR;
        return false;
    }
    
    // Start plugin
    impl_->pluginStates[pluginName] = PluginState::READY;
    if (!plugin->onStart()) {
        std::cerr << "[PluginManager] Plugin onStart failed: " << pluginName << std::endl;
        impl_->pluginStates[pluginName] = PluginState::ERROR;
        return false;
    }
    
    // Plugin is now running
    impl_->pluginStates[pluginName] = PluginState::RUNNING;
    impl_->plugins[pluginName] = plugin;
    impl_->pluginMetadata[pluginName] = metadata;
    
    std::cout << "[PluginManager] Plugin loaded successfully: " << pluginName << std::endl;
    return true;
}

bool PluginManager::unloadPlugin(const std::string& pluginName) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    auto it = impl_->plugins.find(pluginName);
    if (it == impl_->plugins.end()) {
        std::cerr << "[PluginManager] Plugin not found: " << pluginName << std::endl;
        return false;
    }
    
    impl_->pluginStates[pluginName] = PluginState::STOPPING;
    
    try {
        if (!it->second->onStop()) {
            std::cerr << "[PluginManager] Plugin onStop failed: " << pluginName << std::endl;
        }
        
        impl_->pluginStates[pluginName] = PluginState::UNLOADING;
        if (!it->second->onUnload()) {
            std::cerr << "[PluginManager] Plugin onUnload failed: " << pluginName << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PluginManager] Error unloading plugin " << pluginName 
                  << ": " << e.what() << std::endl;
    }
    
    impl_->plugins.erase(it);
    impl_->pluginStates.erase(pluginName);
    impl_->pluginMetadata.erase(pluginName);
    
    std::cout << "[PluginManager] Plugin unloaded: " << pluginName << std::endl;
    return true;
}

bool PluginManager::reloadPlugin(const std::string& pluginName) {
    // Find plugin path
    std::string pluginPath = impl_->config.pluginDirectory + "/" + pluginName + ".json";
    
    // Unload if loaded
    if (impl_->plugins.find(pluginName) != impl_->plugins.end()) {
        unloadPlugin(pluginName);
    }
    
    // Reload
    return loadPlugin(pluginPath);
}

std::vector<std::string> PluginManager::discoverPlugins() const {
    std::vector<std::string> plugins;
    
    try {
        std::filesystem::path pluginDir(impl_->config.pluginDirectory);
        if (!std::filesystem::exists(pluginDir)) {
            return plugins;
        }
        
        for (const auto& entry : std::filesystem::directory_iterator(pluginDir)) {
            if (entry.path().extension() == ".json") {
                plugins.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[PluginManager] Error discovering plugins: " << e.what() << std::endl;
    }
    
    return plugins;
}

std::vector<std::string> PluginManager::getLoadedPlugins() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    std::vector<std::string> pluginNames;
    for (const auto& pair : impl_->plugins) {
        pluginNames.push_back(pair.first);
    }
    
    return pluginNames;
}

PluginResult PluginManager::executeRequestChain(const PluginContext& context) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    PluginResult finalResult{true, "", {}, true, false, false};
    
    // Execute plugins in order (could be priority-based in future)
    for (auto& pair : impl_->plugins) {
        const auto& pluginName = pair.first;
        auto& plugin = pair.second;
        
        if (impl_->pluginStates[pluginName] != PluginState::RUNNING) {
            continue;
        }
        
        // Check if plugin has request interception capability
        if (!plugin->hasCapability(PluginCapability::INTERCEPT_REQUEST)) {
            continue;
        }
        
        impl_->profiler.startProfiling(pluginName);
        
        try {
            PluginResult result = plugin->onRequest(context);
            
            impl_->profiler.endProfiling(pluginName, result.success, result.errorMessage);
            
            if (!result.success) {
                std::cerr << "[PluginManager] Plugin " << pluginName << " failed: " 
                          << result.errorMessage << std::endl;
                continue;
            }
            
            // Merge results
            if (result.modifiedRequest) {
                finalResult.modifiedRequest = true;
            }
            
            // Stop chain if plugin requests it
            if (!result.shouldContinue) {
                finalResult.shouldContinue = false;
                break;
            }
            
        } catch (const std::exception& e) {
            impl_->profiler.endProfiling(pluginName, false, e.what());
            std::cerr << "[PluginManager] Exception in plugin " << pluginName 
                      << ": " << e.what() << std::endl;
        }
    }
    
    return finalResult;
}

PluginResult PluginManager::executeResponseChain(const PluginContext& context) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    PluginResult finalResult{true, "", {}, true, false, false};
    
    // Execute plugins in reverse order for response processing
    std::vector<std::string> pluginNames;
    for (const auto& pair : impl_->plugins) {
        pluginNames.push_back(pair.first);
    }
    std::reverse(pluginNames.begin(), pluginNames.end());
    
    for (const auto& pluginName : pluginNames) {
        auto plugin = impl_->plugins[pluginName];
        
        if (impl_->pluginStates[pluginName] != PluginState::RUNNING) {
            continue;
        }
        
        // Check if plugin has response interception capability
        if (!plugin->hasCapability(PluginCapability::INTERCEPT_RESPONSE)) {
            continue;
        }
        
        impl_->profiler.startProfiling(pluginName);
        
        try {
            PluginResult result = plugin->onResponse(context);
            
            impl_->profiler.endProfiling(pluginName, result.success, result.errorMessage);
            
            if (!result.success) {
                std::cerr << "[PluginManager] Plugin " << pluginName << " failed: " 
                          << result.errorMessage << std::endl;
                continue;
            }
            
            // Merge results
            if (result.modifiedResponse) {
                finalResult.modifiedResponse = true;
            }
            
            // Stop chain if plugin requests it
            if (!result.shouldContinue) {
                finalResult.shouldContinue = false;
                break;
            }
            
        } catch (const std::exception& e) {
            impl_->profiler.endProfiling(pluginName, false, e.what());
            std::cerr << "[PluginManager] Exception in plugin " << pluginName 
                      << ": " << e.what() << std::endl;
        }
    }
    
    return finalResult;
}

std::vector<std::string> PluginManager::getPluginsWithCapability(PluginCapability cap) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    std::vector<std::string> pluginNames;
    for (const auto& pair : impl_->plugins) {
        if (pair.second->hasCapability(cap)) {
            pluginNames.push_back(pair.first);
        }
    }
    
    return pluginNames;
}

PluginResult PluginManager::readData(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    // Find first plugin with READ_DATA capability
    for (auto& pair : impl_->plugins) {
        if (pair.second->hasCapability(PluginCapability::READ_DATA)) {
            impl_->profiler.startProfiling(pair.first);
            auto result = pair.second->onReadData(key);
            impl_->profiler.endProfiling(pair.first, result.success, result.errorMessage);
            
            if (result.success) {
                return result;
            }
        }
    }
    
    return PluginResult{false, "No plugin with READ_DATA capability found", {}, false, false, false};
}

PluginResult PluginManager::writeData(const std::string& key, const nlohmann::json& value) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    // Find first plugin with WRITE_DATA capability
    for (auto& pair : impl_->plugins) {
        if (pair.second->hasCapability(PluginCapability::WRITE_DATA)) {
            impl_->profiler.startProfiling(pair.first);
            auto result = pair.second->onWriteData(key, value);
            impl_->profiler.endProfiling(pair.first, result.success, result.errorMessage);
            
            if (result.success) {
                return result;
            }
        }
    }
    
    return PluginResult{false, "No plugin with WRITE_DATA capability found", {}, false, false, false};
}

PluginResult PluginManager::deleteData(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    // Find first plugin with DELETE_DATA capability
    for (auto& pair : impl_->plugins) {
        if (pair.second->hasCapability(PluginCapability::DELETE_DATA)) {
            impl_->profiler.startProfiling(pair.first);
            auto result = pair.second->onDeleteData(key);
            impl_->profiler.endProfiling(pair.first, result.success, result.errorMessage);
            
            if (result.success) {
                return result;
            }
        }
    }
    
    return PluginResult{false, "No plugin with DELETE_DATA capability found", {}, false, false, false};
}

PluginResult PluginManager::setPluginConfig(const std::string& pluginName,
                                           const std::string& key,
                                           const nlohmann::json& value) {
    bool success = impl_->configManager->setPluginConfig(pluginName, nlohmann::json{{key, value}});
    PluginResult result;
    result.success = success;
    if (!success) {
        result.errorMessage = "Failed to set plugin config";
    }
    return result;
}

std::map<std::string, bool> PluginManager::getPluginHealth() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    std::map<std::string, bool> health;
    for (const auto& pair : impl_->plugins) {
        health[pair.first] = pair.second->isHealthy();
    }
    
    return health;
}

void PluginManager::startHealthCheck() {
    if (impl_->healthCheckRunning) {
        return;
    }
    
    impl_->healthCheckRunning = true;
    impl_->healthCheckThread = std::thread([this]() {
        while (impl_->healthCheckRunning) {
            std::this_thread::sleep_for(impl_->config.healthCheckInterval);
            
            std::lock_guard<std::mutex> lock(impl_->mutex);
            for (auto& pair : impl_->plugins) {
                if (!pair.second->isHealthy()) {
                    std::cerr << "[PluginManager] Plugin " << pair.first << " is unhealthy" << std::endl;
                    // Could implement auto-restart here
                }
            }
        }
    });
}

void PluginManager::stopHealthCheck() {
    impl_->healthCheckRunning = false;
    if (impl_->healthCheckThread.joinable()) {
        impl_->healthCheckThread.join();
    }
}

bool PluginManager::registerPlugin(std::shared_ptr<Plugin> plugin) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    if (!plugin) {
        std::cerr << "[PluginManager] Cannot register null plugin" << std::endl;
        return false;
    }
    
    auto metadata = plugin->getMetadata();
    std::string pluginName = metadata.name;
    
    bool alreadyRegistered = impl_->plugins.find(pluginName) != impl_->plugins.end();
    if (!alreadyRegistered) {
        if (impl_->config.maxPlugins >= 0 && impl_->plugins.size() >= static_cast<size_t>(impl_->config.maxPlugins)) {
            std::cerr << "[PluginManager] Maximum plugin count (" << impl_->config.maxPlugins << ") reached, cannot register " << pluginName << std::endl;
            return false;
        }
    }
    
    impl_->plugins[pluginName] = plugin;
    impl_->pluginMetadata[pluginName] = metadata;
    impl_->pluginStates[pluginName] = PluginState::LOADED;
    
    return true;
}

bool PluginManager::unregisterPlugin(const std::string& pluginName) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    auto it = impl_->plugins.find(pluginName);
    if (it == impl_->plugins.end()) {
        return false;
    }
    
    impl_->plugins.erase(it);
    impl_->pluginMetadata.erase(pluginName);
    impl_->pluginStates.erase(pluginName);
    
    return true;
}

std::shared_ptr<Plugin> PluginManager::getPlugin(const std::string& pluginName) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    
    auto it = impl_->plugins.find(pluginName);
    if (it != impl_->plugins.end()) {
        return it->second;
    }
    
    return nullptr;
}

// Plugin registry implementation
PluginRegistry& PluginRegistry::instance() {
    static PluginRegistry instance;
    return instance;
}

void PluginRegistry::registerFactory(const std::string& pluginType, PluginFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[pluginType] = factory;
}

std::shared_ptr<Plugin> PluginRegistry::createPlugin(const std::string& pluginType) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = factories_.find(pluginType);
    if (it != factories_.end()) {
        return it->second();
    }
    
    return nullptr;
}

std::vector<std::string> PluginRegistry::getRegisteredTypes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> types;
    for (const auto& pair : factories_) {
        types.push_back(pair.first);
    }
    
    return types;
}

// Plugin config manager implementation
PluginConfigManager::PluginConfigManager(const std::string& configPath)
    : configPath_(configPath) {
    loadConfig();
}

bool PluginConfigManager::loadConfig() {
    std::lock_guard<std::mutex> lock(mutex_);
    return loadConfigUnlocked();
}

bool PluginConfigManager::loadConfigUnlocked() {
    try {
        std::ifstream file(configPath_);
        if (file.is_open()) {
            file >> config_;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PluginConfigManager] Error loading config: " << e.what() << std::endl;
    }

    // Initialize empty config
    config_ = nlohmann::json::object();
    return false;
}

bool PluginConfigManager::saveConfig() {
    std::lock_guard<std::mutex> lock(mutex_);
    return saveConfigUnlocked();
}

bool PluginConfigManager::saveConfigUnlocked() {
    try {
        std::ofstream file(configPath_);
        if (file.is_open()) {
            file << config_.dump(2);
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PluginConfigManager] Error saving config: " << e.what() << std::endl;
    }

    return false;
}

nlohmann::json PluginConfigManager::getPluginConfig(const std::string& pluginName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (config_.contains(pluginName)) {
        return config_[pluginName];
    }
    
    return nlohmann::json::object();
}

bool PluginConfigManager::setPluginConfig(const std::string& pluginName, const nlohmann::json& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    config_[pluginName] = config;
    return saveConfigUnlocked();
}

std::vector<std::string> PluginConfigManager::getConfiguredPlugins() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> pluginNames;
    for (auto it = config_.begin(); it != config_.end(); ++it) {
        pluginNames.push_back(it.key());
    }
    
    return pluginNames;
}

// Plugin profiler implementation
void PluginProfiler::startProfiling(const std::string& pluginName) {
    std::lock_guard<std::mutex> lock(mutex_);
    startTimes_[pluginName] = std::chrono::system_clock::now();
}

void PluginProfiler::endProfiling(const std::string& pluginName, bool success, const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = startTimes_.find(pluginName);
    if (it == startTimes_.end()) {
        return;
    }
    
    auto endTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - it->second);
    
    PluginExecutionMetrics metric;
    metric.pluginName = pluginName;
    metric.executionTime = duration;
    metric.success = success;
    metric.errorMessage = error;
    metric.memoryUsed = 0; // Could implement memory tracking
    
    metrics_[pluginName] = metric;
    startTimes_.erase(it);
}

std::vector<PluginExecutionMetrics> PluginProfiler::getMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<PluginExecutionMetrics> result;
    for (const auto& pair : metrics_) {
        result.push_back(pair.second);
    }
    
    return result;
}

void PluginProfiler::clearMetrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.clear();
    startTimes_.clear();
}

nlohmann::json PluginProfiler::getMetricsJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    nlohmann::json result = nlohmann::json::array();
    for (const auto& pair : metrics_) {
        result.push_back(pair.second.toJson());
    }
    
    return result;
}

} // namespace plugin
} // namespace phoenix
