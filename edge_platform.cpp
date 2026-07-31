/* edge_platform.cpp - Edge platform implementation
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

#include "edge_platform.hpp"
#include "rdk_x5_bpu.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef __linux__
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/gpio.h>
#include <linux/i2c-dev.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace edge_platform {

namespace {

int64_t nowMs() {
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string trimCopy(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

template <typename T>
T safeJsonValue(const json &j, const std::string &key, const T &fallback) {
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

std::string upperCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

fs::path resolvePath(const fs::path &candidate, const fs::path &baseDir) {
    if (candidate.empty()) {
        return {};
    }
    if (candidate.is_absolute()) {
        return candidate;
    }
    if (!baseDir.empty()) {
        return baseDir / candidate;
    }
    return candidate;
}

std::string pathText(const fs::path &path) {
    return path.generic_string();
}

bool pathLooksAvailable(const std::string &path);

json stringListToJson(const std::vector<std::string> &values);

std::string normalizePlatformTaskKind(const std::string &raw, const std::string &fallback) {
    const std::string lowered = lowerCopy(trimCopy(raw));
    return lowered.empty() ? fallback : lowered;
}

std::string platformTaskOperation(const json &payload, const std::string &fallback) {
    return normalizePlatformTaskKind(
        safeJsonValue(payload, "operation", safeJsonValue(payload, "opType", safeJsonValue(payload, "kind", fallback))),
        fallback);
}

json buildPeripheralQueueView(const PlatformManager::Metrics &metrics, const RuntimeConfig &config) {
    return json{{"backlog", metrics.computeBacklog},
                {"capacity", config.maxPeripheralInflight},
                {"schedulingEnabled", config.peripheralSchedulingEnabled}};
}

json buildPeripheralPlanSummary(const PlatformManager::TopologyCache &topology,
                                const PlatformManager::Metrics &metrics,
                                const RuntimeConfig &config,
                                const json &payload,
                                const std::string &taskClass,
                                const std::string &defaultOperation) {
    const std::string operation = platformTaskOperation(payload, defaultOperation);
    const int latencyBudgetMs = std::max(0, safeJsonValue(payload, "latencyBudgetMs", safeJsonValue(payload, "deadlineMs", 80)));
    const bool schedulingEnabled = config.enabled && config.peripheralSchedulingEnabled;
    const bool topologyReady = topology.loaded;
    std::vector<std::string> reasons;

    if (!config.enabled) {
        reasons.push_back("platform disabled");
    }
    if (!config.peripheralSchedulingEnabled) {
        reasons.push_back("peripheral scheduling disabled");
    }
    if (!topology.loaded) {
        reasons.push_back("topology unavailable; using static fallback envelope");
    }
    if (reasons.empty()) {
        reasons.push_back("task scheduled against edge platform control plane");
    }

    return json{{"requestSummary",
                 json{{"taskClass", taskClass},
                      {"operation", operation},
                      {"latencyBudgetMs", latencyBudgetMs}}},
                {"route",
                 json{{"mode", schedulingEnabled ? "scheduled" : "passthrough"},
                      {"category", taskClass},
                      {"controller", schedulingEnabled ? "edge-platform-control-plane" : "host-cpu"},
                      {"topologyReady", topologyReady},
                      {"reason", stringListToJson(reasons)}}},
                {"queue", buildPeripheralQueueView(metrics, config)},
                {"topologyDigest",
                 json{{"loaded", topology.loaded},
                      {"signalCount", topology.signalBindings.size()},
                      {"interfaceCount", topology.interfaces.size()},
                      {"connectors", stringListToJson(topology.connectors)}}}};
}

json buildPeripheralDispatchResult(PlatformManager::Metrics &metrics,
                                   const RuntimeConfig &config,
                                   const json &plan,
                                   const std::string &taskClass,
                                   const json &payload) {
    const bool accepted = config.enabled &&
                          config.peripheralSchedulingEnabled &&
                          metrics.computeBacklog < static_cast<double>(std::max(1, config.maxPeripheralInflight));
    const uint64_t dispatchId = metrics.nextDispatchId++;
    if (accepted) {
        metrics.computeBacklog += 1.0;
    } else {
        metrics.rejectedDispatches += 1;
    }

    json event{{"dispatchId", dispatchId},
               {"kind", taskClass},
               {"accepted", accepted},
               {"operation", plan["requestSummary"].value("operation", platformTaskOperation(payload, taskClass))},
               {"createdAtMs", nowMs()}};

    json response{{"ok", accepted},
                  {"result",
                   json{{"dispatchId", dispatchId},
                        {"accepted", accepted},
                        {"taskClass", taskClass},
                        {"plan", plan},
                        {"queue", buildPeripheralQueueView(metrics, config)}}}};

    if (!accepted) {
        response["error"] = config.enabled && config.peripheralSchedulingEnabled
                                 ? "peripheral queue saturated"
                                 : "peripheral scheduling disabled";
        event["error"] = response["error"];
    }

    return json{{"event", event}, {"response", response}};
}

void pushUnique(std::vector<std::string> &target, const std::string &value) {
    if (value.empty()) {
        return;
    }
    if (std::find(target.begin(), target.end(), value) == target.end()) {
        target.push_back(value);
    }
}

bool containsAny(const std::string &haystack, std::initializer_list<const char *> needles) {
    for (const char *needle : needles) {
        if (haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool isNetlistJson(const fs::path &path) {
    const std::string filename = lowerCopy(path.filename().string());
    return path.has_extension() && lowerCopy(path.extension().string()) == ".json" && filename.rfind("eext_netlist", 0) == 0;
}

std::string normalizeBackend(const std::string &raw) {
    const std::string lowered = lowerCopy(trimCopy(raw));
    if (lowered == "cpu" || lowered == "npu" || lowered == "hybrid") {
        return lowered;
    }
    return "auto";
}

int resolveWeightBlockId(const json &payload) {
    return std::max(0, safeJsonValue(payload, "weightBlockId", safeJsonValue(payload, "weightBlock", 0)));
}

std::size_t resolveWeightBytes(const json &payload, int tensorBytes) {
    const int declared = std::max(0, safeJsonValue(payload, "weightBytes", safeJsonValue(payload, "weightsBytes", 0)));
    if (declared > 0) {
        return static_cast<std::size_t>(declared);
    }
    return static_cast<std::size_t>(std::max(4096, tensorBytes));
}

double updateEwma(double prev, double sample, double alpha) {
    if (sample <= 0.0) {
        return prev;
    }
    if (prev <= 0.0) {
        return sample;
    }
    const double clampedAlpha = std::max(0.01, std::min(0.99, alpha));
    return prev * (1.0 - clampedAlpha) + sample * clampedAlpha;
}

fs::path defaultGerberConnectorMap(const RuntimeConfig &config) {
    if (!config.gerberConnectorMap.empty()) {
        return resolvePath(config.gerberConnectorMap, config.baseDir);
    }
    if (!config.baseDir.empty()) {
        return config.baseDir / "catastrophe" / "gerber_catastrophe1_20260417_connector_map.json";
    }
    return fs::path("catastrophe") / "gerber_catastrophe1_20260417_connector_map.json";
}

std::string preferredLinuxGpioHardwareDriver(const RuntimeConfig &config) {
#ifdef __linux__
    const fs::path gpioChipPath = resolvePath(config.gpioChipDevice, config.baseDir);
    const fs::path sysfsRoot = resolvePath(config.npuGpioSysfsRoot, config.baseDir);
    const bool gpioChipReady = pathLooksAvailable(pathText(gpioChipPath));
    const bool sysfsReady = pathLooksAvailable(pathText(sysfsRoot));
    if (config.preferGpioChip) {
        if (gpioChipReady) {
            return "linux-gpiochip";
        }
        if (sysfsReady) {
            return "linux-sysfs-gpio";
        }
    } else {
        if (sysfsReady) {
            return "linux-sysfs-gpio";
        }
        if (gpioChipReady) {
            return "linux-gpiochip";
        }
    }
    return "unavailable";
#else
    (void)config;
    return "unavailable";
#endif
}

std::string preferredAsyncGpioExecutionDriver(const RuntimeConfig &config) {
#ifdef __linux__
    if (!config.npuAsyncGpioExecuteLocally) {
        return "mock-gpio-async";
    }
    const std::string hardwareDriver = preferredLinuxGpioHardwareDriver(config);
    return hardwareDriver == "unavailable" ? "mock-gpio-async" : hardwareDriver;
#else
    (void)config;
    return "mock-gpio-async";
#endif
}

bool isLocalLinuxGpioDriver(const std::string &driver) {
    return driver == "linux-gpiochip" || driver == "linux-sysfs-gpio";
}

std::string runtimeLabel() {
#ifdef __linux__
    return "linux";
#else
    return "host";
#endif
}

double clampUnit(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

double clampPositiveUnit(double value) {
    return std::max(0.0, std::min(1.0, value));
}

int clampDurationMs(int value, int fallback) {
    return std::max(50, std::min(10000, value > 0 ? value : fallback));
}

std::string joinPreview(const std::vector<std::string> &values, std::size_t maxItems) {
    std::ostringstream out;
    const std::size_t limit = std::min(maxItems, values.size());
    for (std::size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << values[i];
    }
    if (values.size() > limit) {
        out << " ...";
    }
    return out.str();
}

bool pathLooksAvailable(const std::string &path) {
    if (path.empty()) {
        return false;
    }
    if (std::all_of(path.begin(), path.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
        return true;
    }
    std::error_code ec;
    return fs::exists(path, ec);
}


std::vector<std::string> sortedPhysicalConnectorKeys(const PlatformManager::TopologyCache &topology) {
    std::vector<std::string> keys;
    keys.reserve(topology.physicalConnectors.size());
    for (const auto &entry : topology.physicalConnectors) {
        keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

json physicalConnectorMapToJson(const PlatformManager::TopologyCache &topology) {
    json out = json::object();
    for (const auto &key : sortedPhysicalConnectorKeys(topology)) {
        out[key] = topology.physicalConnectors.at(key).toJson();
    }
    return out;
}

bool parseSpiFrames(const json &payload, std::vector<std::vector<uint8_t>> &frames, std::string &error) {
    auto parseOne = [&](const std::string &raw) -> std::vector<uint8_t> {
        std::string compact;
        compact.reserve(raw.size());
        for (char ch : raw) {
            if (std::isxdigit(static_cast<unsigned char>(ch)) != 0) {
                compact.push_back(ch);
            }
        }
        if (compact.empty()) {
            return {};
        }
        if ((compact.size() % 2) != 0) {
            compact.insert(compact.begin(), '0');
        }
        std::vector<uint8_t> bytes;
        bytes.reserve(compact.size() / 2);
        for (std::size_t i = 0; i < compact.size(); i += 2) {
            const std::string token = compact.substr(i, 2);
            char *end = nullptr;
            const auto value = std::strtoul(token.c_str(), &end, 16);
            if (end == nullptr || *end != '\0') {
                bytes.clear();
                return bytes;
            }
            bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
        }
        return bytes;
    };

    frames.clear();
    if (payload.is_object() && payload.contains("transportFrames") && payload["transportFrames"].is_array()) {
        for (const auto &item : payload["transportFrames"]) {
            if (!item.is_string()) {
                continue;
            }
            auto bytes = parseOne(item.get<std::string>());
            if (bytes.empty()) {
                error = "invalid transportFrames hex payload";
                return false;
            }
            frames.push_back(std::move(bytes));
        }
    } else if (payload.is_object() && payload.contains("spiTxHex") && payload["spiTxHex"].is_string()) {
        auto bytes = parseOne(payload["spiTxHex"].get<std::string>());
        if (bytes.empty()) {
            error = "invalid spiTxHex payload";
            return false;
        }
        frames.push_back(std::move(bytes));
    }
    return true;
}

bool hasSignalBinding(const PlatformManager::TopologyCache &topology, const std::string &signal) {
    return topology.signalBindings.find(signal) != topology.signalBindings.end();
}

std::vector<std::string> collectPresentSignals(const PlatformManager::TopologyCache &topology,
                                               std::initializer_list<const char *> signals) {
    std::vector<std::string> out;
    for (const char *signal : signals) {
        if (signal != nullptr && hasSignalBinding(topology, signal)) {
            out.emplace_back(signal);
        }
    }
    return out;
}

std::string firstPresentSignal(const PlatformManager::TopologyCache &topology,
                               std::initializer_list<const char *> signals) {
    for (const char *signal : signals) {
        if (signal != nullptr && hasSignalBinding(topology, signal)) {
            return signal;
        }
    }
    return {};
}

std::vector<std::string> collectWeightConfigBusSignals(const PlatformManager::TopologyCache &topology) {
    return collectPresentSignals(topology,
                                 {"GPIO2_WEIGHT_CFG_I2C_SDA", "GPIO3_WEIGHT_CFG_I2C_SCL"});
}

std::string firstWeightConfigIrqSignal(const PlatformManager::TopologyCache &topology) {
    return firstPresentSignal(topology, {"GPIO19_WEIGHT_CFG_INT"});
}

std::string resolveWeightControllerFaultMode(const json &payload) {
    std::string mode;
    if (payload.is_object() && payload.contains("faultInjection") && payload["faultInjection"].is_object()) {
        const json &faultInjection = payload["faultInjection"];
        if (faultInjection.contains("weightControllerFaultMode") && faultInjection["weightControllerFaultMode"].is_string()) {
            mode = faultInjection["weightControllerFaultMode"].get<std::string>();
        } else if (faultInjection.contains("weightController") && faultInjection["weightController"].is_string()) {
            mode = faultInjection["weightController"].get<std::string>();
        } else if (faultInjection.contains("mcu") && faultInjection["mcu"].is_string()) {
            mode = faultInjection["mcu"].get<std::string>();
        }
    }
    if (mode.empty() && payload.is_object() && payload.contains("weightControllerFaultMode") && payload["weightControllerFaultMode"].is_string()) {
        mode = payload["weightControllerFaultMode"].get<std::string>();
    }
    if (mode.empty() && payload.is_object() && payload.contains("mcuFaultMode") && payload["mcuFaultMode"].is_string()) {
        mode = payload["mcuFaultMode"].get<std::string>();
    }
    mode = lowerCopy(trimCopy(mode));
    if (mode.empty() || mode == "none" || mode == "ok" || mode == "healthy") {
        return "none";
    }
    return mode;
}

bool weightControllerFaulted(const std::string &mode) {
    return !mode.empty() && mode != "none";
}

bool hasDirectAsyncBoundary(const PlatformManager::TopologyCache &topology) {
    return collectPresentSignals(topology,
                                 {"GPIO10_DAC_IN0", "GPIO11_DAC_IN1", "GPIO8_DAC_IN2", "GPIO7_DAC_IN3",
                                  "GPIO17_DAC_IN4", "GPIO18_DAC_IN5", "GPIO27_DAC_IN6", "GPIO22_DAC_IN7",
                                  "GPIO23_DAC_IN8", "GPIO24_DAC_IN9", "GPIO9_DAC_IN10", "GPIO25_DAC_IN11"}).size() == 12 &&
           collectPresentSignals(topology,
                                 {"GPIO5_CARRA_CMP", "GPIO12_CARRB_CMP", "GPIO13_CARRC_CMP", "GPIO16_CARRD_CMP",
                                  "GPIO20_CARRE_CMP", "GPIO21_CARRF_CMP", "GPIO14_CARRG_CMP", "GPIO15_CARRH_CMP"}).size() == 8 &&
           hasSignalBinding(topology, "GPIO6_NPU_SAMPLE_SYNC");
}

int clampProtocolValue(int value, int minimum, int maximum) {
    return std::max(minimum, std::min(maximum, value));
}

int signalBitMask(std::size_t signalCount) {
    if (signalCount == 0) {
        return 0;
    }
    if (signalCount >= 16) {
        return 0xFFFF;
    }
    return (1 << static_cast<int>(signalCount)) - 1;
}

void appendLe16(std::vector<uint8_t> &bytes, int value) {
    const uint16_t packed = static_cast<uint16_t>(clampProtocolValue(value, 0, 65535));
    bytes.push_back(static_cast<uint8_t>(packed & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((packed >> 8) & 0xFFu));
}

std::vector<uint8_t> buildProtocolFrame(uint8_t opcode,
                                        uint8_t requestTag,
                                        uint8_t flags,
                                        const std::vector<uint8_t> &payload) {
    std::vector<uint8_t> frame;
    frame.reserve(payload.size() + 8);
    frame.push_back(0xA5u);
    frame.push_back(0x5Au);
    frame.push_back(0x01u);
    frame.push_back(opcode);
    frame.push_back(requestTag);
    frame.push_back(flags);
    frame.push_back(static_cast<uint8_t>(std::min<std::size_t>(payload.size(), 255)));
    frame.insert(frame.end(), payload.begin(), payload.end());
    uint8_t checksum = 0u;
    for (uint8_t byte : frame) {
        checksum ^= byte;
    }
    frame.push_back(checksum);
    return frame;
}

int gpioLineFromSignal(const std::string &signal) {
    const std::string trimmed = trimCopy(signal);
    if (trimmed.size() < 5 || trimmed.rfind("GPIO", 0) != 0) {
        return -1;
    }
    int value = 0;
    bool sawDigit = false;
    for (std::size_t index = 4; index < trimmed.size(); ++index) {
        const unsigned char ch = static_cast<unsigned char>(trimmed[index]);
        if (std::isdigit(ch) == 0) {
            break;
        }
        value = (value * 10) + static_cast<int>(trimmed[index] - '0');
        sawDigit = true;
    }
    return sawDigit ? value : -1;
}

int jsonProtocolInt(const json &node, const char *key, int fallback) {
    if (!node.is_object() || key == nullptr || !node.contains(key)) {
        return fallback;
    }
    const json &value = node.at(key);
    if (value.is_number_integer()) {
        return value.get<int>();
    }
    if (value.is_number_unsigned()) {
        return static_cast<int>(value.get<unsigned int>());
    }
    return fallback;
}

json buildSignalActions(const std::vector<std::string> &signals, int value) {
    json actions = json::array();
    for (std::size_t index = 0; index < signals.size(); ++index) {
        const int level = (value >> static_cast<int>(index)) & 0x01;
        actions.push_back(json{{"signal", signals[index]},
                               {"line", gpioLineFromSignal(signals[index])},
                               {"bitIndex", static_cast<int>(index)},
                               {"level", level}});
    }
    return actions;
}

json buildReadSignals(const std::vector<std::string> &signals) {
    json readSignals = json::array();
    for (std::size_t index = 0; index < signals.size(); ++index) {
        readSignals.push_back(json{{"signal", signals[index]},
                                   {"line", gpioLineFromSignal(signals[index])},
                                   {"bitIndex", static_cast<int>(index)}});
    }
    return readSignals;
}

json buildPulseActions(const std::string &signal, int pulseUs) {
    if (signal.empty()) {
        return json::array();
    }
    return json::array({json{{"signal", signal},
                             {"line", gpioLineFromSignal(signal)},
                             {"highUs", clampProtocolValue(pulseUs, 0, 1000)},
                             {"lowUs", 0}}});
}

std::string bytesToHex(const std::vector<uint8_t> &bytes);
json stringListToJson(const std::vector<std::string> &values);

json buildNpuProtocolEnvelope(const PlatformManager::TopologyCache &topology,
                              const RuntimeConfig &config,
                              const json &payload,
                              const std::string &mode,
                              const std::string &transport,
                              int shardCount,
                              int preferredBatch,
                              int tokens,
                              int tensorBytes,
                              bool streaming,
                              bool postProcess) {
    const auto txDataSignals = collectPresentSignals(topology,
                                                     {"GPIO10_DAC_IN0", "GPIO11_DAC_IN1", "GPIO8_DAC_IN2", "GPIO7_DAC_IN3",
                                                      "GPIO17_DAC_IN4", "GPIO18_DAC_IN5", "GPIO27_DAC_IN6", "GPIO22_DAC_IN7",
                                                      "GPIO23_DAC_IN8", "GPIO24_DAC_IN9", "GPIO9_DAC_IN10", "GPIO25_DAC_IN11"});
    const std::vector<std::string> txSelectSignals;
    const std::vector<std::string> rxSelectSignals;
    const auto rxDataSignals = collectPresentSignals(topology,
                                                     {"GPIO5_CARRA_CMP", "GPIO12_CARRB_CMP", "GPIO13_CARRC_CMP", "GPIO16_CARRD_CMP", "GPIO20_CARRE_CMP", "GPIO21_CARRF_CMP",
                                                      "GPIO14_CARRG_CMP", "GPIO15_CARRH_CMP",
                                                      "GPIO4_SLICE_SUM0_CMP", "GPIO26_SLICE_SUM1_CMP"});
    const std::string sampleSyncSignal = firstPresentSignal(topology, {"GPIO6_NPU_SAMPLE_SYNC"});
    const auto weightCfgBusSignals = collectWeightConfigBusSignals(topology);
    const std::string weightCfgSdaSignal = firstPresentSignal(topology, {"GPIO2_WEIGHT_CFG_I2C_SDA"});
    const std::string weightCfgSclSignal = firstPresentSignal(topology, {"GPIO3_WEIGHT_CFG_I2C_SCL"});
    const std::string irqSignal = firstWeightConfigIrqSignal(topology);
    const bool directAsync = hasDirectAsyncBoundary(topology);
    const bool rawTransportProvided = (payload.is_object() && payload.contains("transportFrames") && payload["transportFrames"].is_array()) ||
                                      (payload.is_object() && payload.contains("spiTxHex") && payload["spiTxHex"].is_string());
    const bool autoBuildFrames = safeJsonValue(payload, "autoBuildTransportFrames", true) && !rawTransportProvided;
    const int requestId = clampProtocolValue(safeJsonValue(payload, "requestId", safeJsonValue(payload, "dispatchTag", 1)), 1, 65535);
    const int weightBlockId = clampProtocolValue(safeJsonValue(payload, "weightBlockId", safeJsonValue(payload, "weightBlock", 0)), 0, 255);
    // 硬件常量——来自网表 eext_netlist.json 实际信号绑定
    // 比较器：8路载波（CARRA~CARRH，GPIO5/12/13/16/20/21/14/15）+ 2路slice（GPIO4/26）
    const int hardwareComparatorCount = 10;
    // DAC输入：12路（GPIO10/11/8/7/17/18/27/22/23/24/9/25）
    const int dacChannelCount = 12;
    // Pi Zero 2W GPIO实测速度（来源：quickfixsurrey.ca benchmark，Saleae 100MS/s验证）：
    //   C + libgpiod v2.1 : ~720 kHz（受系统调用开销限制）
    //   C + mmap 寄存器直写: ~24 MHz（BCM2837B0 ARM核直接写GPSET/GPCLR寄存器）
    // SPI0 路径（BCM2835 SPI核心250MHz，最小分频2）：理论上限125MHz，npuSpiSpeedHz=120MHz为保守设置
    // tau绕过机制：在directAsync模式下sampleWindowUs=0，Pi不等待RC充电，
    //   由板载GD32F427 MCU管理NPU_SAMPLE_SYNC脉冲时序，RC时间常数(101ns)由MCU侧控制。
    //   Pi侧只负责以最快速度驱动DAC线和读回比较器结果，流水线由overlapSlots实现。
    const int maxPipelineDepth = hardwareComparatorCount; // 全部10路比较器流水线
    const int queueDepth = clampProtocolValue(safeJsonValue(payload, "queueDepth", std::min(config.maxComputeInflight, directAsync ? maxPipelineDepth : 1)), 1, std::max(1, config.maxComputeInflight));
    const int pipelineDepth = clampProtocolValue(safeJsonValue(payload, "pipelineDepth", directAsync ? std::min(queueDepth, maxPipelineDepth) : 1), 1, queueDepth);
    // 读回通道：全部10路比较器并行读回
    const int defaultReadChannels = static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(hardwareComparatorCount, rxDataSignals.size())));
    const int readChannels = clampProtocolValue(safeJsonValue(payload, "readChannels", directAsync ? defaultReadChannels : 1), 1, hardwareComparatorCount);
    // sampleWindowUs=0（directAsync）：Pi不插入软件等待，RC时序完全由MCU NPU_SAMPLE_SYNC控制
    // sampleWindowUs>0（同步模式）：Pi自己等待窗口后读回，需至少1µs
    const int minSampleWindowUs = directAsync ? 0 : 1;
    const int sampleWindowUs = clampProtocolValue(safeJsonValue(payload, "sampleWindowUs", safeJsonValue(payload, "integrationWindowUs", 0)), minSampleWindowUs, 60000);
    const int timeoutUs = clampProtocolValue(safeJsonValue(payload, "readTimeoutUs", std::max(1, sampleWindowUs * 4)), 1, 65535);
    // DAC并行字数：默认驱动全部12路DAC线（每个GPIO对应一路模拟输入权重）
    const int dacParallelWords = dacChannelCount;
    const int inputWords = clampProtocolValue(
        safeJsonValue(payload, "inputVectorWords",
                      std::max(dacParallelWords, std::min(255, tokens > 0 ? (tokens + 15) / 16 : std::max(1, tensorBytes / 4096)))),
        1,
        255);
    const int outputWords = clampProtocolValue(safeJsonValue(payload, "outputVectorWords", std::max(1, readChannels)), 1, 255);
    const bool streamWeights = safeJsonValue(payload, "streamWeights", false);
    const int overlapSlots = directAsync ? pipelineDepth : 1;
    const std::string executionModel = directAsync ? "async-overlapped" : "ordered-sync";
    const std::string queueDiscipline = directAsync ? "tagged-overlap-readback" : "in-order-readback";
    const bool separateTxRx = directAsync && !txDataSignals.empty() && !rxDataSignals.empty();
    const json requestedWeightConfig = (payload.is_object() && payload.contains("weightConfig") && payload["weightConfig"].is_object())
                                           ? payload["weightConfig"]
                                           : json::object();
    const json requestedWeightTargets = (requestedWeightConfig.contains("targets") && requestedWeightConfig["targets"].is_array())
                                            ? requestedWeightConfig["targets"]
                                            : ((payload.is_object() && payload.contains("weightTargets") && payload["weightTargets"].is_array())
                                                   ? payload["weightTargets"]
                                                   : json::array());
    const bool weightControlAvailable = weightCfgBusSignals.size() == 2;
    const std::string weightControllerFaultMode = resolveWeightControllerFaultMode(payload);
    const bool weightControllerHealthy = !weightControllerFaulted(weightControllerFaultMode);
    const bool weightConfigRequested = !requestedWeightConfig.empty() || !requestedWeightTargets.empty();
    const json requestedCycles = (payload.is_object() && payload.contains("inputCycles") && payload["inputCycles"].is_array())
                                     ? payload["inputCycles"]
                                     : ((payload.is_object() && payload.contains("inputWindows") && payload["inputWindows"].is_array())
                                            ? payload["inputWindows"]
                                            : json::array());
    const bool legacyWindowAliasUsed = payload.is_object() && !payload.contains("inputCycles") && payload.contains("inputWindows") && payload["inputWindows"].is_array();
    const int cycleCount = clampProtocolValue(
        !requestedCycles.empty() ? static_cast<int>(requestedCycles.size()) : shardCount,
        1,
        255);

    json transportFrames = json::array();
    json framePlan = json::array();
    json cyclePlan = json::array();
    json gpioPlan = json::array();
    json weightTransactionPlan = json::array();
    int order = 1;
    int gpioOrder = 1;
    auto addFrame = [&](const std::string &stage,
                        const std::string &opcodeName,
                        int requestTag,
                        int queueSlot,
                        const std::vector<uint8_t> &frame,
                        const std::string &overlapHint) {
        const std::string hex = bytesToHex(frame);
        transportFrames.push_back(hex);
        framePlan.push_back(json{{"order", order++},
                                 {"stage", stage},
                                 {"opcode", opcodeName},
                                 {"requestTag", requestTag},
                                 {"queueSlot", queueSlot},
                                 {"hex", hex},
                                 {"overlapHint", overlapHint}});
    };
    auto appendActions = [](json &target, const json &source) {
        if (!source.is_array()) {
            return;
        }
        for (const auto &item : source) {
            target.push_back(item);
        }
    };
    auto addGpioStage = [&](const std::string &stage,
                            int requestTag,
                            int queueSlot,
                            int windowIndex,
                            const json &setActions,
                            const json &pulseActions,
                            const json &readSignals,
                            int settleUs,
                            int mockReadbackValue,
                            const std::string &overlapHint) {
        json step{{"order", gpioOrder++},
                  {"stage", stage},
                  {"requestTag", requestTag},
                  {"queueSlot", queueSlot},
                  {"windowIndex", windowIndex},
                  {"setActions", setActions},
                  {"pulseActions", pulseActions},
                  {"readSignals", readSignals},
                  {"settleUs", std::max(0, settleUs)},
                  {"overlapHint", overlapHint}};
        if (mockReadbackValue >= 0) {
            step["mockReadbackValue"] = mockReadbackValue;
        }
        gpioPlan.push_back(step);
    };

    for (std::size_t index = 0; index < requestedWeightTargets.size(); ++index) {
        const auto &target = requestedWeightTargets[index];
        if (!target.is_object()) {
            continue;
        }
        int targetIndex = clampProtocolValue(target.value("targetIndex", target.value("index", 0)), 0, 127);
        int row = clampProtocolValue(target.value("row", targetIndex / 16), 0, 7);
        int col = clampProtocolValue(target.value("col", targetIndex % 16), 0, 15);
        targetIndex = row * 16 + col;
        json item{{"order", static_cast<int>(index + 1)},
                  {"targetIndex", targetIndex},
                  {"row", row},
                  {"col", col}};
        if (target.contains("value")) {
            item["value"] = target["value"];
        }
        if (target.contains("code")) {
            item["code"] = target["code"];
        }
        if (target.contains("label") && target["label"].is_string()) {
            item["label"] = target["label"].get<std::string>();
        }
        weightTransactionPlan.push_back(item);
    }

    for (int shard = 0; shard < cycleCount; ++shard) {
        const json requestedCycle = (shard < static_cast<int>(requestedCycles.size()) && requestedCycles[shard].is_object())
                                        ? requestedCycles[shard]
                                         : json::object();
        const int queueSlot = overlapSlots > 0 ? (shard % overlapSlots) : 0;
        const int requestTag = (requestId + shard) & 0xFF;
        const int windowWeightBlockId = clampProtocolValue(jsonProtocolInt(requestedCycle, "weightBlockId", weightBlockId), 0, 255);
        const int windowSampleWindowUs = clampProtocolValue(jsonProtocolInt(requestedCycle, "sampleWindowUs", sampleWindowUs), minSampleWindowUs, 60000);
        const int windowReadChannels = clampProtocolValue(jsonProtocolInt(requestedCycle, "readChannels", readChannels), 1, 16);
        const int windowInputWords = clampProtocolValue(jsonProtocolInt(requestedCycle, "inputWords", inputWords), 1, 255);
        const int windowOutputWords = clampProtocolValue(jsonProtocolInt(requestedCycle, "outputWords", outputWords), 1, 255);
        // txDataValue：默认驱动全部12路DAC线（signalBitMask(12)=0x0FFF）
        // shard & 0x07 是历史遗留3位掩码，已替换为全通道掩码
        // 调用方可在inputCycles[i].txDataValue中按需指定特定权重向量
        const int txDataValue = clampProtocolValue(
            jsonProtocolInt(requestedCycle, "txDataValue", jsonProtocolInt(requestedCycle, "txData", signalBitMask(txDataSignals.size()))),
            0,
            signalBitMask(txDataSignals.size()));
        const int txSelectValue = clampProtocolValue(
            jsonProtocolInt(requestedCycle, "txSelectValue", jsonProtocolInt(requestedCycle, "txSelect", shard & 0x0F)),
            0,
            signalBitMask(txSelectSignals.size()));
        const int rxSelectValue = clampProtocolValue(
            jsonProtocolInt(requestedCycle, "rxSelectValue", jsonProtocolInt(requestedCycle, "rxSelect", queueSlot & 0x07)),
            0,
            signalBitMask(rxSelectSignals.size()));
        const int mockReadbackValue = clampProtocolValue(
            jsonProtocolInt(requestedCycle, "mockReadbackValue", jsonProtocolInt(requestedCycle, "mockReadbackBits", requestTag & 0x03)),
            0,
            signalBitMask(rxDataSignals.size()));

        cyclePlan.push_back(json{{"cycleIndex", shard},
                                 {"windowIndex", shard},
                                 {"requestTag", requestTag},
                                 {"queueSlot", queueSlot},
                                 {"weightBlockId", windowWeightBlockId},
                                 {"sampleWindowUs", windowSampleWindowUs},
                                 {"readChannels", windowReadChannels},
                                 {"inputWords", windowInputWords},
                                 {"outputWords", windowOutputWords},
                                 {"txDataValue", txDataValue},
                                 {"txSelectValue", txSelectValue},
                                 {"rxSelectValue", rxSelectValue},
                                 {"mockReadbackValue", mockReadbackValue}});
    }

    if (autoBuildFrames) {
        std::vector<uint8_t> configurePayload;
        configurePayload.push_back(static_cast<uint8_t>(queueDepth));
        configurePayload.push_back(static_cast<uint8_t>(pipelineDepth));
        configurePayload.push_back(static_cast<uint8_t>(preferredBatch));
        configurePayload.push_back(static_cast<uint8_t>(cycleCount));
        appendLe16(configurePayload, sampleWindowUs);
        appendLe16(configurePayload, timeoutUs);
        configurePayload.push_back(static_cast<uint8_t>(readChannels));
        configurePayload.push_back(static_cast<uint8_t>(inputWords));
        configurePayload.push_back(static_cast<uint8_t>(outputWords));
        uint8_t configureFlags = 0u;
        if (directAsync) {
            configureFlags |= 0x01u;
        }
        if (streaming) {
            configureFlags |= 0x02u;
        }
        if (postProcess) {
            configureFlags |= 0x04u;
        }
        if (streamWeights) {
            configureFlags |= 0x08u;
        }
        addFrame("configure-pipeline",
                 "CONFIGURE_ASYNC_PIPELINE",
                 requestId & 0xFF,
                 0,
                 buildProtocolFrame(0x10u, static_cast<uint8_t>(requestId & 0xFF), configureFlags, configurePayload),
                 directAsync ? "fill async pipeline before first readback" : "sequential execution only");

        int nextResultToDrain = 0;
        for (const auto &cycle : cyclePlan) {
            const int shard = cycle.value("cycleIndex", cycle.value("windowIndex", 0));
            const int queueSlot = cycle.value("queueSlot", 0);
            const int requestTag = cycle.value("requestTag", requestId & 0xFF);
            const int windowWeightBlockId = cycle.value("weightBlockId", weightBlockId);
            const int windowInputWords = cycle.value("inputWords", inputWords);
            const int windowReadChannels = cycle.value("readChannels", readChannels);
            const int windowOutputWords = cycle.value("outputWords", outputWords);
            const int windowSampleWindowUs = cycle.value("sampleWindowUs", sampleWindowUs);
            const int txDataValue = cycle.value("txDataValue", 0);
            const int txSelectValue = cycle.value("txSelectValue", 0);
            const int rxSelectValue = cycle.value("rxSelectValue", 0);
            const int mockReadbackValue = cycle.value("mockReadbackValue", 0);

            std::vector<uint8_t> enqueuePayload;
            enqueuePayload.push_back(static_cast<uint8_t>(queueSlot));
            enqueuePayload.push_back(static_cast<uint8_t>(windowWeightBlockId & 0xFF));
            enqueuePayload.push_back(static_cast<uint8_t>(windowInputWords));
            enqueuePayload.push_back(static_cast<uint8_t>(preferredBatch));
            enqueuePayload.push_back(static_cast<uint8_t>(windowReadChannels));
            enqueuePayload.push_back(static_cast<uint8_t>(streamWeights ? 1 : 0));
            appendLe16(enqueuePayload, txDataValue & signalBitMask(txDataSignals.size()));
            enqueuePayload.push_back(static_cast<uint8_t>(txSelectValue & signalBitMask(txSelectSignals.size())));
            uint8_t enqueueFlags = static_cast<uint8_t>(queueSlot & 0x03u);
            if (directAsync) {
                enqueueFlags |= 0x10u;
            }
            if (!streamWeights) {
                enqueueFlags |= 0x20u;
            }
            addFrame("enqueue-input-cycle",
                     "ENQUEUE_INPUT_CYCLE",
                     requestTag,
                     queueSlot,
                     buildProtocolFrame(0x20u, static_cast<uint8_t>(requestTag), enqueueFlags, enqueuePayload),
                     directAsync ? "host may keep toggling DAC write lines one GPIO timestep at a time while prior ADC readback remains in flight" : "wait for previous result before the next GPIO input cycle");

            json enqueueActions = json::array();
            appendActions(enqueueActions, buildSignalActions(txDataSignals, txDataValue));
            appendActions(enqueueActions, buildSignalActions(txSelectSignals, txSelectValue));
            addGpioStage("enqueue-input-cycle",
                         requestTag,
                         queueSlot,
                         shard,
                         enqueueActions,
                         json::array(),
                         json::array(),
                         0,
                         -1,
                         directAsync ? "drive the DAC data bus and selector lines for this GPIO input cycle" : "prepare the single in-order GPIO input cycle");

            std::vector<uint8_t> samplePayload;
            samplePayload.push_back(static_cast<uint8_t>(queueSlot));
            appendLe16(samplePayload, windowSampleWindowUs);
            samplePayload.push_back(static_cast<uint8_t>(windowReadChannels));
            samplePayload.push_back(static_cast<uint8_t>(windowOutputWords));
            samplePayload.push_back(static_cast<uint8_t>(txDataSignals.size()));
            samplePayload.push_back(static_cast<uint8_t>(rxDataSignals.size()));
            samplePayload.push_back(static_cast<uint8_t>(rxSelectValue & 0x07u));
            addFrame("commit-sample-cycle",
                     "COMMIT_SAMPLE_CYCLE",
                     requestTag,
                     queueSlot,
                     buildProtocolFrame(0x30u,
                                        static_cast<uint8_t>(requestTag),
                                        directAsync ? 0x01u : 0x00u,
                                        samplePayload),
                     directAsync ? "sample sync fires for this GPIO timestep while the receive side keeps a separate address domain" : "sample is serialized with readback for this GPIO timestep");
            addGpioStage("commit-sample-cycle",
                         requestTag,
                         queueSlot,
                         shard,
                         json::array(),
                         buildPulseActions(sampleSyncSignal, config.npuGpioPulseUs),
                         json::array(),
                         windowSampleWindowUs,
                         -1,
                         directAsync ? "pulse the sample sync for the current GPIO input cycle while the readback domain remains independent" : "pulse sample sync for the current GPIO input cycle");

            const bool resultReadyToDrain = directAsync ? (shard + 1 >= overlapSlots) : true;
            if (resultReadyToDrain) {
                const int completed = directAsync ? nextResultToDrain : shard;
                const json &readWindow = cyclePlan.at(completed);
                const int readSlot = readWindow.value("queueSlot", overlapSlots > 0 ? (completed % overlapSlots) : 0);
                const int readTag = readWindow.value("requestTag", (requestId + completed) & 0xFF);
                const int readOutputWords = readWindow.value("outputWords", outputWords);
                const int readRxSelectValue = readWindow.value("rxSelectValue", 0);
                const int readMockReadbackValue = readWindow.value("mockReadbackValue", 0);
                std::vector<uint8_t> readPayload;
                readPayload.push_back(static_cast<uint8_t>(readSlot));
                readPayload.push_back(static_cast<uint8_t>(readOutputWords));
                appendLe16(readPayload, timeoutUs);
                readPayload.push_back(static_cast<uint8_t>(rxSelectSignals.size()));
                readPayload.push_back(static_cast<uint8_t>(rxDataSignals.size()));
                readPayload.push_back(static_cast<uint8_t>(readRxSelectValue & signalBitMask(rxSelectSignals.size())));
                addFrame("dequeue-result-cycle",
                         "DEQUEUE_RESULT_CYCLE",
                         readTag,
                         readSlot,
                         buildProtocolFrame(0x40u,
                                            static_cast<uint8_t>(readTag),
                                            directAsync ? 0x02u : 0x00u,
                                            readPayload),
                         directAsync ? "readback drains an older tag while a newer GPIO input cycle may already be integrating" : "current result must be drained before the next GPIO input cycle");
                addGpioStage("dequeue-result-cycle",
                             readTag,
                             readSlot,
                             completed,
                             buildSignalActions(rxSelectSignals, readRxSelectValue),
                             json::array(),
                             buildReadSignals(rxDataSignals),
                             1,
                             readMockReadbackValue,
                             directAsync ? "select an older ADC bank and read back its comparator result for an earlier GPIO input cycle" : "read back the current GPIO input cycle result");
                if (directAsync) {
                    nextResultToDrain += 1;
                }
            }
        }

        while (nextResultToDrain < cycleCount) {
            const json &readWindow = cyclePlan.at(nextResultToDrain);
            const int readSlot = readWindow.value("queueSlot", overlapSlots > 0 ? (nextResultToDrain % overlapSlots) : 0);
            const int readTag = readWindow.value("requestTag", (requestId + nextResultToDrain) & 0xFF);
            const int readOutputWords = readWindow.value("outputWords", outputWords);
            const int readRxSelectValue = readWindow.value("rxSelectValue", 0);
            const int readMockReadbackValue = readWindow.value("mockReadbackValue", 0);
            std::vector<uint8_t> readPayload;
            readPayload.push_back(static_cast<uint8_t>(readSlot));
            readPayload.push_back(static_cast<uint8_t>(readOutputWords));
            appendLe16(readPayload, timeoutUs);
            readPayload.push_back(static_cast<uint8_t>(rxSelectSignals.size()));
            readPayload.push_back(static_cast<uint8_t>(rxDataSignals.size()));
            readPayload.push_back(static_cast<uint8_t>(readRxSelectValue & signalBitMask(rxSelectSignals.size())));
            addFrame("drain-tail-result-cycle",
                     "DEQUEUE_RESULT_CYCLE",
                     readTag,
                     readSlot,
                     buildProtocolFrame(0x40u,
                                        static_cast<uint8_t>(readTag),
                                        directAsync ? 0x02u : 0x00u,
                                        readPayload),
                     "drain remaining queued results after the final GPIO input cycle is committed");
            addGpioStage("drain-tail-result-cycle",
                         readTag,
                         readSlot,
                         nextResultToDrain,
                         buildSignalActions(rxSelectSignals, readRxSelectValue),
                         json::array(),
                         buildReadSignals(rxDataSignals),
                         1,
                         readMockReadbackValue,
                         "drain the remaining queued comparator results for earlier GPIO input cycles");
            nextResultToDrain += 1;
        }
    }

    std::string effectiveWeightControlMode = "idle-sideband-available";
    if (!weightControlAvailable) {
        effectiveWeightControlMode = "sideband-incomplete";
    }
    if (weightConfigRequested) {
        if (!weightControllerHealthy) {
            effectiveWeightControlMode = "bypass-retain-current-weights";
        } else if (weightControlAvailable) {
            effectiveWeightControlMode = "i2c-u30-matrix-update";
        } else {
            effectiveWeightControlMode = "unavailable-retain-current-weights";
        }
    } else if (!weightControllerHealthy) {
        effectiveWeightControlMode = "faulted-idle";
    }

    const json signalsJson{{"txData", stringListToJson(txDataSignals)},
                           {"txSelect", stringListToJson(txSelectSignals)},
                           {"rxSelect", stringListToJson(rxSelectSignals)},
                           {"rxData", stringListToJson(rxDataSignals)},
                           {"sampleSync", sampleSyncSignal},
                           {"irq", irqSignal},
                           {"weightCfgI2c", stringListToJson(weightCfgBusSignals)},
                           {"weightCfgInt", irqSignal}};
    json stagesJson = json::array({json{{"name", "enqueue-input-cycle"},
                                         {"hostRole", "drive the DAC boundary lines for the next GPIO input timestep"},
                                         {"overlapAllowed", directAsync},
                                         {"requiresWindowValues", true}},
                                     json{{"name", "commit-sample-cycle"},
                                         {"hostRole", "pulse sample sync after the current GPIO input timestep is armed"},
                                         {"overlapAllowed", directAsync},
                                         {"requiresWindowValues", false}},
                                     json{{"name", "dequeue-result-cycle"},
                                         {"hostRole", "sample the comparator readback GPIO set associated with one GPIO input timestep"},
                                         {"overlapAllowed", directAsync},
                                         {"requiresWindowValues", true}}});
    if (weightControlAvailable || weightConfigRequested || !weightControllerHealthy) {
        stagesJson.push_back(json{{"name", "apply-weight-config"},
                                  {"hostRole",
                                   weightControllerHealthy
                                       ? "optionally update the onboard GD32 weight matrix over GPIO2/GPIO3 I2C before compute windows are launched"
                                       : "skip the GD32 weight update, retain the current analog weights, and keep the Pi compute hot path independent"},
                                  {"overlapAllowed", false},
                                  {"requiresWindowValues", false}});
    }
    const json throughputHintsJson{{"recommendedInFlight", overlapSlots},
                                   {"estimatedOverlapGain", directAsync ? overlapSlots : 1},
                                   {"streamWeights", streamWeights},
                                   {"preferredBatch", preferredBatch},
                                   {"shardCount", cycleCount},
                                   {"inputCycleCount", cycleCount},
                                   {"inputCycleSignalWidth", txDataSignals.size()},
                                   {"ioCycleGranularity", "per-gpio-timestep"}};

    json weightControlJson{{"available", weightControlAvailable},
                           {"bus", "i2c"},
                           {"signals", json{{"sda", weightCfgSdaSignal}, {"scl", weightCfgSclSignal}, {"irq", irqSignal}}},
                           {"matrix", json{{"rows", 8}, {"cols", 16}, {"totalTargets", 128}}},
                           {"weightBlockId", weightBlockId},
                           {"requested", weightConfigRequested},
                           {"requestedTargetCount", static_cast<int>(weightTransactionPlan.size())},
                           {"transactionPlan", weightTransactionPlan},
                           {"readyInterrupt", !irqSignal.empty()},
                           {"controllerHealthy", weightControllerHealthy},
                           {"faultMode", weightControllerFaultMode},
                           {"degraded", !weightControllerHealthy},
                           {"effectiveMode", effectiveWeightControlMode},
                           {"hotPathIndependent", true},
                           {"fallbackPath", "pi-default-compute-hot-path"}};
    if (weightConfigRequested && !weightControlAvailable) {
        weightControlJson["warning"] = "weight configuration was requested but GPIO2/GPIO3 I2C is incomplete in the discovered topology";
    } else if (weightConfigRequested && !weightControllerHealthy) {
        weightControlJson["warning"] = "weight configuration was requested while the onboard weight controller is fault-injected; the request is bypassed and current weights are retained";
    }

    return json{{"name", "phoenix-gpio-async-npu-v1"},
                {"enabled", mode != "cpu" && mode != "rejected"},
                {"executionModel", executionModel},
                {"queueDiscipline", queueDiscipline},
                {"separateTxRx", separateTxRx},
                {"separateAddressDomains", directAsync},
                {"autoBuiltFrames", autoBuildFrames},
                {"transport", transport},
                {"queueDepth", queueDepth},
                {"pipelineDepth", pipelineDepth},
                {"readChannels", readChannels},
                {"sampleWindowUs", sampleWindowUs},
                {"timeoutUs", timeoutUs},
                {"weightBlockId", weightBlockId},
                {"ioCycleGranularity", "per-gpio-timestep"},
                {"legacyWindowAliasUsed", legacyWindowAliasUsed},
                {"cyclePlan", cyclePlan},
                {"windowPlan", cyclePlan},
                {"gpioPlan", gpioPlan},
                {"signals", signalsJson},
                {"weightControl", weightControlJson},
                {"stages", stagesJson},
                {"throughputHints", throughputHintsJson},
                {"transportFrames", transportFrames},
                {"framePlan", framePlan}};
}

std::string bytesToHex(const std::vector<uint8_t> &bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) {
        out << std::setw(2) << static_cast<int>(byte);
    }
    return out.str();
}

#ifdef __linux__
struct GpioChipExecutionContext {
    std::unordered_map<int, int> outputHandles;
    std::unordered_map<int, int> inputHandles;
    std::vector<int> ownedHandles;

    ~GpioChipExecutionContext() {
        for (int handle : ownedHandles) {
            if (handle >= 0) {
                ::close(handle);
            }
        }
    }

    GpioChipExecutionContext(const GpioChipExecutionContext &) = delete;
    GpioChipExecutionContext &operator=(const GpioChipExecutionContext &) = delete;
    GpioChipExecutionContext() = default;
};

bool requestGpioChipLineHandle(const RuntimeConfig &config,
                               int line,
                               bool output,
                               int initialLevel,
                               int &handle,
                               std::string &error) {
    if (line < 0) {
        error = "invalid gpio line";
        return false;
    }
    const fs::path gpioChipPath = resolvePath(config.gpioChipDevice, config.baseDir);
    const int chipFd = ::open(gpioChipPath.c_str(), O_RDONLY | O_CLOEXEC);
    if (chipFd < 0) {
        error = "failed to open gpio chip " + pathText(gpioChipPath) + ": " + std::string(std::strerror(errno));
        return false;
    }

    struct gpiohandle_request request {
    };
    request.lineoffsets[0] = static_cast<unsigned int>(line);
    request.lines = 1;
    request.flags = output ? GPIOHANDLE_REQUEST_OUTPUT : GPIOHANDLE_REQUEST_INPUT;
    request.default_values[0] = initialLevel != 0 ? 1 : 0;
    std::snprintf(request.consumer_label, sizeof(request.consumer_label), "phoenix-edge");
    if (ioctl(chipFd, GPIO_GET_LINEHANDLE_IOCTL, &request) < 0) {
        error = "failed to request gpio line " + std::to_string(line) + " from " + pathText(gpioChipPath) + ": " + std::string(std::strerror(errno));
        ::close(chipFd);
        return false;
    }
    ::close(chipFd);
    handle = request.fd;
    return true;
}

bool ensureGpioChipHandle(const RuntimeConfig &config,
                          GpioChipExecutionContext &context,
                          int line,
                          bool output,
                          int initialLevel,
                          int &handle,
                          std::string &error) {
    auto &handles = output ? context.outputHandles : context.inputHandles;
    const auto it = handles.find(line);
    if (it != handles.end()) {
        handle = it->second;
        return true;
    }

    int openedHandle = -1;
    if (!requestGpioChipLineHandle(config, line, output, initialLevel, openedHandle, error)) {
        return false;
    }
    context.ownedHandles.push_back(openedHandle);
    handles.emplace(line, openedHandle);
    handle = openedHandle;
    return true;
}

bool writeGpioChipHandleValue(int handle, int level, std::string &error) {
    struct gpiohandle_data data {
    };
    data.values[0] = level != 0 ? 1 : 0;
    if (ioctl(handle, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &data) < 0) {
        error = "failed to set gpiochip line value: " + std::string(std::strerror(errno));
        return false;
    }
    return true;
}

bool readGpioChipHandleValue(int handle, int &level, std::string &error) {
    struct gpiohandle_data data {
    };
    if (ioctl(handle, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &data) < 0) {
        error = "failed to read gpiochip line value: " + std::string(std::strerror(errno));
        return false;
    }
    level = data.values[0] != 0 ? 1 : 0;
    return true;
}

bool writeGpioChipLine(const RuntimeConfig &config,
                       GpioChipExecutionContext &context,
                       int line,
                       int level,
                       std::string &error) {
    int handle = -1;
    if (!ensureGpioChipHandle(config, context, line, true, level, handle, error)) {
        return false;
    }
    return writeGpioChipHandleValue(handle, level, error);
}

bool readGpioChipLine(const RuntimeConfig &config,
                      GpioChipExecutionContext &context,
                      int line,
                      int &level,
                      std::string &error) {
    int handle = -1;
    if (!ensureGpioChipHandle(config, context, line, false, 0, handle, error)) {
        return false;
    }
    return readGpioChipHandleValue(handle, level, error);
}

bool writeGpioChipLineOnce(const RuntimeConfig &config, int line, bool high, std::string &error) {
    GpioChipExecutionContext context;
    return writeGpioChipLine(config, context, line, high ? 1 : 0, error);
}

bool readGpioChipLineOnce(const RuntimeConfig &config, int line, bool &high, std::string &error) {
    GpioChipExecutionContext context;
    int level = 0;
    if (!readGpioChipLine(config, context, line, level, error)) {
        return false;
    }
    high = level != 0;
    return true;
}

bool writeTextFile(const fs::path &path, const std::string &text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << text;
    return static_cast<bool>(out);
}

bool ensureSysfsGpioDirection(const RuntimeConfig &config, int line, const std::string &direction, std::string &error) {
    if (line < 0) {
        error = "invalid gpio line";
        return false;
    }
    const fs::path sysfsRoot = resolvePath(config.npuGpioSysfsRoot, config.baseDir);
    const fs::path gpioDir = sysfsRoot / ("gpio" + std::to_string(line));
    if (!fs::exists(gpioDir)) {
        if (!writeTextFile(sysfsRoot / "export", std::to_string(line))) {
            error = "failed to export gpio" + std::to_string(line);
            return false;
        }
    }
    if (!writeTextFile(gpioDir / "direction", direction)) {
        error = "failed to set gpio" + std::to_string(line) + " direction to " + direction;
        return false;
    }
    return true;
}

bool writeSysfsGpioLine(const RuntimeConfig &config, int line, int level, std::string &error) {
    if (!ensureSysfsGpioDirection(config, line, "out", error)) {
        return false;
    }
    const fs::path valuePath = resolvePath(config.npuGpioSysfsRoot, config.baseDir) / ("gpio" + std::to_string(line)) / "value";
    if (!writeTextFile(valuePath, level != 0 ? "1" : "0")) {
        error = "failed to write gpio" + std::to_string(line) + " value";
        return false;
    }
    return true;
}

bool readSysfsGpioLine(const RuntimeConfig &config, int line, int &level, std::string &error) {
    if (!ensureSysfsGpioDirection(config, line, "in", error)) {
        return false;
    }
    const fs::path valuePath = resolvePath(config.npuGpioSysfsRoot, config.baseDir) / ("gpio" + std::to_string(line)) / "value";
    std::ifstream in(valuePath, std::ios::binary);
    if (!in) {
        error = "failed to read gpio" + std::to_string(line) + " value";
        return false;
    }
    char value = '0';
    in >> value;
    level = value == '0' ? 0 : 1;
    return true;
}
#endif

bool executeAsyncGpioPlanWithDriver(const RuntimeConfig &config,
                                    const json &protocol,
                                    const std::string &driver,
                                    json &execution,
                                    std::string &error) {
    if (!protocol.is_object() || !protocol.contains("gpioPlan") || !protocol["gpioPlan"].is_array()) {
        error = "protocol missing gpioPlan";
        return false;
    }

    const bool useSysfs = driver == "linux-sysfs-gpio";
    const bool useGpioChip = driver == "linux-gpiochip";
#ifndef __linux__
    if (useSysfs || useGpioChip) {
        error = "linux gpio drivers are unavailable on this target";
        return false;
    }
#endif

#ifdef __linux__
    GpioChipExecutionContext gpioChipContext;
#endif

    json trace = json::array();
    json readbacks = json::array();
    std::map<std::string, int> signalState;
    const json transportFrames = (protocol.contains("transportFrames") && protocol["transportFrames"].is_array())
                                     ? protocol["transportFrames"]
                                     : json::array();

    auto recordTrace = [&](const std::string &stage,
                           const std::string &kind,
                           int requestTag,
                           int queueSlot,
                           const std::string &signal,
                           int line,
                           int level) {
        trace.push_back(json{{"stage", stage},
                             {"kind", kind},
                             {"requestTag", requestTag},
                             {"queueSlot", queueSlot},
                             {"signal", signal},
                             {"line", line},
                             {"level", level}});
    };

    for (const auto &step : protocol["gpioPlan"]) {
        if (!step.is_object()) {
            continue;
        }
        const std::string stage = step.value("stage", std::string("unknown"));
        const int requestTag = step.value("requestTag", 0);
        const int queueSlot = step.value("queueSlot", 0);

        if (step.contains("setActions") && step["setActions"].is_array()) {
            for (const auto &action : step["setActions"]) {
                const std::string signal = action.value("signal", std::string());
                const int line = action.value("line", -1);
                const int level = action.value("level", 0) != 0 ? 1 : 0;
                if (useGpioChip) {
#ifdef __linux__
                    if (!writeGpioChipLine(config, gpioChipContext, line, level, error)) {
                        return false;
                    }
#endif
                } else if (useSysfs) {
#ifdef __linux__
                    if (!writeSysfsGpioLine(config, line, level, error)) {
                        return false;
                    }
#endif
                }
                signalState[signal] = level;
                recordTrace(stage, "set", requestTag, queueSlot, signal, line, level);
            }
        }

        if (step.contains("pulseActions") && step["pulseActions"].is_array()) {
            for (const auto &pulse : step["pulseActions"]) {
                const std::string signal = pulse.value("signal", std::string());
                const int line = pulse.value("line", -1);
                const int highUs = std::max(0, pulse.value("highUs", config.npuGpioPulseUs));
                const int lowUs = std::max(0, pulse.value("lowUs", 0));
                if (useGpioChip) {
#ifdef __linux__
                    if (!writeGpioChipLine(config, gpioChipContext, line, 1, error)) {
                        return false;
                    }
#endif
                } else if (useSysfs) {
#ifdef __linux__
                    if (!writeSysfsGpioLine(config, line, 1, error)) {
                        return false;
                    }
#endif
                }
                signalState[signal] = 1;
                recordTrace(stage, "pulse-high", requestTag, queueSlot, signal, line, 1);
                if (highUs > 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(highUs));
                }
                if (useGpioChip) {
#ifdef __linux__
                    if (!writeGpioChipLine(config, gpioChipContext, line, 0, error)) {
                        return false;
                    }
#endif
                } else if (useSysfs) {
#ifdef __linux__
                    if (!writeSysfsGpioLine(config, line, 0, error)) {
                        return false;
                    }
#endif
                }
                signalState[signal] = 0;
                recordTrace(stage, "pulse-low", requestTag, queueSlot, signal, line, 0);
                if (lowUs > 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(lowUs));
                }
            }
        }

        const int settleUs = std::max(0, step.value("settleUs", 0));
        if (settleUs > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(settleUs));
        }

        if (step.contains("readSignals") && step["readSignals"].is_array() && !step["readSignals"].empty()) {
            json samples = json::array();
            int combined = 0;
            const int mockReadbackValue = clampProtocolValue(step.value("mockReadbackValue", 0), 0, 255);
            for (const auto &readSignal : step["readSignals"]) {
                const std::string signal = readSignal.value("signal", std::string());
                const int line = readSignal.value("line", -1);
                const int bitIndex = std::max(0, readSignal.value("bitIndex", 0));
                int level = (mockReadbackValue >> bitIndex) & 0x01;
                if (useGpioChip) {
#ifdef __linux__
                    if (!readGpioChipLine(config, gpioChipContext, line, level, error)) {
                        return false;
                    }
#endif
                } else if (useSysfs) {
#ifdef __linux__
                    if (!readSysfsGpioLine(config, line, level, error)) {
                        return false;
                    }
#endif
                }
                signalState[signal] = level;
                combined |= (level & 0x01) << bitIndex;
                samples.push_back(json{{"signal", signal},
                                       {"line", line},
                                       {"bitIndex", bitIndex},
                                       {"level", level}});
                recordTrace(stage, "read", requestTag, queueSlot, signal, line, level);
            }
            readbacks.push_back(json{{"stage", stage},
                                     {"requestTag", requestTag},
                                     {"queueSlot", queueSlot},
                                     {"combinedValue", combined},
                                     {"samples", samples}});
        }
    }

    execution = json{{"executed", true},
                     {"driver", driver},
                     {"frameCount", transportFrames.size()},
                     {"stepCount", protocol["gpioPlan"].size()},
                     {"simulated", !(useGpioChip || useSysfs)},
                     {"trace", trace},
                     {"readbacks", readbacks}};
    return true;
}

bool executeAsyncGpioPlan(const RuntimeConfig &config,
                          const json &protocol,
                          json &execution,
                          std::string &error) {
    const std::string preferredDriver = preferredAsyncGpioExecutionDriver(config);
#ifdef __linux__
    if (preferredDriver == "linux-gpiochip") {
        json preferredExecution = json::object();
        std::string preferredError;
        if (executeAsyncGpioPlanWithDriver(config, protocol, preferredDriver, preferredExecution, preferredError)) {
            execution = std::move(preferredExecution);
            return true;
        }

        const fs::path sysfsRoot = resolvePath(config.npuGpioSysfsRoot, config.baseDir);
        if (pathLooksAvailable(pathText(sysfsRoot))) {
            json fallbackExecution = json::object();
            std::string fallbackError;
            if (executeAsyncGpioPlanWithDriver(config, protocol, "linux-sysfs-gpio", fallbackExecution, fallbackError)) {
                fallbackExecution["fallbackFrom"] = preferredDriver;
                fallbackExecution["fallbackReason"] = preferredError;
                execution = std::move(fallbackExecution);
                return true;
            }
            error = preferredError + "; linux-sysfs-gpio fallback failed: " + fallbackError;
            return false;
        }

        error = preferredError;
        return false;
    }
#endif
    return executeAsyncGpioPlanWithDriver(config, protocol, preferredDriver, execution, error);
}

#ifdef __linux__
bool writeLinuxGpioValueAuto(const RuntimeConfig &config, int line, bool high, std::string &driverUsed, std::string &error) {
    driverUsed = preferredLinuxGpioHardwareDriver(config);
    if (driverUsed == "linux-gpiochip") {
        std::string gpioChipError;
        if (writeGpioChipLineOnce(config, line, high, gpioChipError)) {
            return true;
        }
        const fs::path sysfsRoot = resolvePath(config.npuGpioSysfsRoot, config.baseDir);
        if (pathLooksAvailable(pathText(sysfsRoot))) {
            driverUsed = "linux-sysfs-gpio";
            std::string sysfsError;
            if (writeSysfsGpioLine(config, line, high ? 1 : 0, sysfsError)) {
                return true;
            }
            error = gpioChipError + "; linux-sysfs-gpio fallback failed: " + sysfsError;
            return false;
        }
        error = gpioChipError;
        return false;
    }
    if (driverUsed == "linux-sysfs-gpio") {
        return writeSysfsGpioLine(config, line, high ? 1 : 0, error);
    }
    error = "no linux gpio driver available";
    return false;
}

bool readLinuxGpioValueAuto(const RuntimeConfig &config, int line, bool &high, std::string &driverUsed, std::string &error) {
    driverUsed = preferredLinuxGpioHardwareDriver(config);
    if (driverUsed == "linux-gpiochip") {
        std::string gpioChipError;
        if (readGpioChipLineOnce(config, line, high, gpioChipError)) {
            return true;
        }
        const fs::path sysfsRoot = resolvePath(config.npuGpioSysfsRoot, config.baseDir);
        if (pathLooksAvailable(pathText(sysfsRoot))) {
            driverUsed = "linux-sysfs-gpio";
            int level = 0;
            std::string sysfsError;
            if (readSysfsGpioLine(config, line, level, sysfsError)) {
                high = level != 0;
                return true;
            }
            error = gpioChipError + "; linux-sysfs-gpio fallback failed: " + sysfsError;
            return false;
        }
        error = gpioChipError;
        return false;
    }
    if (driverUsed == "linux-sysfs-gpio") {
        int level = 0;
        if (!readSysfsGpioLine(config, line, level, error)) {
            return false;
        }
        high = level != 0;
        return true;
    }
    error = "no linux gpio driver available";
    return false;
}
#endif

double positiveDuty(double value) {
    return std::max(0.0, std::min(1.0, value));
}

std::string jsonString(const json &node, const std::string &key) {
    if (!node.is_object() || !node.contains(key) || !node[key].is_string()) {
        return {};
    }
    return trimCopy(node[key].get<std::string>());
}

bool jsonBool(const json &node, const std::string &key, bool fallback = false) {
    if (!node.is_object() || !node.contains(key) || !node[key].is_boolean()) {
        return fallback;
    }
    return node[key].get<bool>();
}

int jsonInt(const json &node, const std::string &key, int fallback) {
    if (!node.is_object() || !node.contains(key) || !node[key].is_number_integer()) {
        return fallback;
    }
    return node[key].get<int>();
}

double jsonDouble(const json &node, const std::string &key, double fallback) {
    if (!node.is_object() || !node.contains(key) || !node[key].is_number()) {
        return fallback;
    }
    return node[key].get<double>();
}

bool isGroundSignal(const std::string &signalUpper);

void mergeGerberConnectorMap(PlatformManager::TopologyCache &topology, const RuntimeConfig &config) {
    topology.gerberConnectorMap = defaultGerberConnectorMap(config);
    if (topology.gerberConnectorMap.empty()) {
        return;
    }

    std::ifstream input(topology.gerberConnectorMap, std::ios::binary);
    if (!input) {
        topology.gerberWarnings.push_back("failed to open gerber connector map: " + pathText(topology.gerberConnectorMap));
        return;
    }

    json doc = json::parse(input, nullptr, false);
    if (doc.is_discarded() || !doc.is_object() || !doc.contains("connectors") || !doc["connectors"].is_object()) {
        topology.gerberWarnings.push_back("invalid gerber connector map json: " + pathText(topology.gerberConnectorMap));
        return;
    }

    for (auto connectorIt = doc["connectors"].begin(); connectorIt != doc["connectors"].end(); ++connectorIt) {
        if (!connectorIt.value().is_object()) {
            continue;
        }
        PlatformManager::ConnectorSummary summary;
        summary.id = connectorIt.key();
        summary.role = jsonString(connectorIt.value(), "role");

        std::vector<std::string> mismatchPins;
        std::vector<std::string> pollutedGroundPins;
        const json pins = connectorIt.value().contains("pins") && connectorIt.value()["pins"].is_object()
                              ? connectorIt.value()["pins"]
                              : json::object();
        for (auto pinIt = pins.begin(); pinIt != pins.end(); ++pinIt) {
            if (!pinIt.value().is_object()) {
                continue;
            }

            PlatformManager::PhysicalPinSummary pinSummary;
            pinSummary.connectorId = summary.id;
            pinSummary.pinNumber = pinIt.key();
            pinSummary.logicalNet = jsonString(pinIt.value(), "logicalNet");
            pinSummary.boardNet = jsonString(pinIt.value(), "boardNet");
            pinSummary.exactMatch = pinIt.value().value("exactMatch",
                                                        !pinSummary.logicalNet.empty() && pinSummary.logicalNet == pinSummary.boardNet);
            summary.pins[pinSummary.pinNumber] = pinSummary;

            const std::string logicalUpper = upperCopy(pinSummary.logicalNet);
            const std::string boardUpper = upperCopy(pinSummary.boardNet);
            if (!pinSummary.logicalNet.empty() && !pinSummary.boardNet.empty() && pinSummary.logicalNet != pinSummary.boardNet) {
                mismatchPins.push_back(summary.id + "." + pinSummary.pinNumber);
                if (boardUpper == "BAT_3S_POS" && isGroundSignal(logicalUpper)) {
                    pollutedGroundPins.push_back(summary.id + "." + pinSummary.pinNumber);
                }
            }
        }

        if (!mismatchPins.empty()) {
            topology.gerberWarnings.push_back("gerber connector map mismatches logical nets on " + summary.id + ": " + joinPreview(mismatchPins, 8));
        }
        if (!pollutedGroundPins.empty()) {
            topology.gerberWarnings.push_back("gerber connector map marks expected ground pins as BAT_3S_POS on " + summary.id + ": " + joinPreview(pollutedGroundPins, 8));
        }
        topology.physicalConnectors[summary.id] = std::move(summary);
    }

    topology.gerberLoaded = !topology.physicalConnectors.empty();
}

json buildHardwareStatus(const RuntimeConfig &config) {
    const std::string preferredGpioDriver = preferredLinuxGpioHardwareDriver(config);
    const std::string preferredAsyncDriver = preferredAsyncGpioExecutionDriver(config);
    json hardware{{"runtime", runtimeLabel()},
                  {"npuSpiDevice", config.npuSpiDevice},
                  {"npuSpiSpeedHz", config.npuSpiSpeedHz},
                  {"npuSpiMode", config.npuSpiMode},
                  {"gpioChipDevice", config.gpioChipDevice},
                  {"preferGpioChip", config.preferGpioChip}};
    hardware["npuSpiReady"] = pathLooksAvailable(config.npuSpiDevice);
#ifdef __linux__
    hardware["gpioSysfsReady"] = pathLooksAvailable(pathText(resolvePath(config.npuGpioSysfsRoot, config.baseDir)));
    hardware["gpioChipReady"] = pathLooksAvailable(pathText(resolvePath(config.gpioChipDevice, config.baseDir)));
    hardware["preferredGpioDriver"] = preferredGpioDriver;
    hardware["preferredAsyncGpioDriver"] = preferredAsyncDriver;
    hardware["asyncGpioLocalExecutionReady"] = isLocalLinuxGpioDriver(preferredAsyncDriver);
#else
    hardware["gpioSysfsReady"] = false;
    hardware["gpioChipReady"] = false;
    hardware["preferredGpioDriver"] = preferredGpioDriver;
    hardware["preferredAsyncGpioDriver"] = preferredAsyncDriver;
    hardware["asyncGpioLocalExecutionReady"] = false;
#endif
    return hardware;
}

#ifdef __linux__

bool writeTextFile(const fs::path &path, const std::string &value, std::string &error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "failed to open file for write: " + pathText(path);
        return false;
    }
    out << value;
    if (!out) {
        error = "failed to write file: " + pathText(path);
        return false;
    }
    return true;
}

bool readTextFile(const fs::path &path, std::string &value, std::string &error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open file for read: " + pathText(path);
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    value = trimCopy(buffer.str());
    return true;
}

bool ensureSysfsGpio(int pin, const std::string &direction, std::string &error) {
    const fs::path gpioRoot = fs::path("/sys/class/gpio") / ("gpio" + std::to_string(pin));
    std::error_code ec;
    if (!fs::exists(gpioRoot, ec)) {
        if (!writeTextFile("/sys/class/gpio/export", std::to_string(pin), error)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return writeTextFile(gpioRoot / "direction", direction, error);
}

bool writeSysfsGpioValue(int pin, bool high, std::string &error) {
    if (!ensureSysfsGpio(pin, "out", error)) {
        return false;
    }
    return writeTextFile(fs::path("/sys/class/gpio") / ("gpio" + std::to_string(pin)) / "value", high ? "1" : "0", error);
}

bool readSysfsGpioValue(int pin, bool &high, std::string &error) {
    if (!ensureSysfsGpio(pin, "in", error)) {
        return false;
    }
    std::string value;
    if (!readTextFile(fs::path("/sys/class/gpio") / ("gpio" + std::to_string(pin)) / "value", value, error)) {
        return false;
    }
    high = !value.empty() && value[0] == '1';
    return true;
}

bool performSpiTransfer(const RuntimeConfig &config,
                        const std::vector<std::vector<uint8_t>> &frames,
                        std::vector<std::vector<uint8_t>> &responses,
                        std::string &error) {
    responses.clear();
    const int fd = ::open(config.npuSpiDevice.c_str(), O_RDWR);
    if (fd < 0) {
        error = "failed to open spi device " + config.npuSpiDevice + ": " + std::string(std::strerror(errno));
        return false;
    }
    auto closeFd = [&]() { ::close(fd); };

    uint8_t mode = static_cast<uint8_t>(config.npuSpiMode & 0x3);
    uint8_t bitsPerWord = 8;
    uint32_t speed = static_cast<uint32_t>(std::max(100000, config.npuSpiSpeedHz));
    if (ioctl(fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bitsPerWord) < 0 ||
        ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        error = "failed to configure spi device: " + std::string(std::strerror(errno));
        closeFd();
        return false;
    }

    for (const auto &frame : frames) {
        std::vector<uint8_t> rx(frame.size(), 0);
        struct spi_ioc_transfer transfer {
        };
        transfer.tx_buf = reinterpret_cast<unsigned long>(frame.data());
        transfer.rx_buf = reinterpret_cast<unsigned long>(rx.data());
        transfer.len = static_cast<uint32_t>(frame.size());
        transfer.speed_hz = speed;
        transfer.bits_per_word = bitsPerWord;
        if (ioctl(fd, SPI_IOC_MESSAGE(1), &transfer) < 0) {
            error = "spi transfer failed: " + std::string(std::strerror(errno));
            closeFd();
            return false;
        }
        responses.push_back(std::move(rx));
    }

    closeFd();
    return true;
}

#endif

bool isPowerSignal(const std::string &signalUpper) {
    if (signalUpper == "3V3" || signalUpper == "5V" || signalUpper == "VIN" || signalUpper == "VBUS" || signalUpper == "VCC" || signalUpper == "VDD") {
        return true;
    }
    if (signalUpper.rfind("V", 0) == 0 && signalUpper.size() > 1 && std::isdigit(static_cast<unsigned char>(signalUpper[1])) != 0) {
        return true;
    }
    if (signalUpper.rfind("BAT", 0) == 0 || signalUpper.find("_POS") != std::string::npos) {
        return true;
    }
    return false;
}

bool isGroundSignal(const std::string &signalUpper) {
    return signalUpper == "GND" || signalUpper == "AGND" || signalUpper == "DGND" || signalUpper.find("_GND") != std::string::npos;
}

bool componentLooksConnector(const std::string &designatorUpper, const std::string &labelLower) {
    if (!designatorUpper.empty() && designatorUpper[0] == 'J') {
        return true;
    }
    return containsAny(labelLower, {"header", "expansion", "mezzanine", "terminal", "input", "module", "boundary", "service", "backbone"});
}

std::string bindingLabel(const PlatformManager::SignalBinding &binding) {
    std::ostringstream out;
    out << (binding.designator.empty() ? binding.componentId : binding.designator);
    if (!binding.pinNumber.empty()) {
        out << ':' << binding.pinNumber;
    }
    return out.str();
}

void addBinding(PlatformManager::TopologyCache &topology,
                const std::string &id,
                const std::string &type,
                const PlatformManager::SignalBinding &binding) {
    auto &iface = topology.interfaces[id];
    iface.id = id;
    iface.type = type;
    iface.available = true;
    pushUnique(iface.signals, binding.signal);
    pushUnique(iface.bindings, bindingLabel(binding));
}

std::vector<fs::path> discoverNetlistFiles(const RuntimeConfig &config) {
    std::vector<fs::path> results;
    const fs::path baseDir = config.baseDir.empty() ? fs::current_path() : config.baseDir;
    if (!config.netlistFiles.empty()) {
        for (const auto &file : config.netlistFiles) {
            const auto resolved = resolvePath(file, baseDir);
            std::error_code ec;
            if (fs::exists(resolved, ec) && fs::is_regular_file(resolved, ec)) {
                results.push_back(fs::absolute(resolved));
            }
        }
    } else {
        fs::path root = resolvePath(config.netlistRoot.empty() ? fs::path("catastrophe") : config.netlistRoot, baseDir);
        std::error_code ec;
        if (fs::exists(root, ec) && fs::is_regular_file(root, ec) && isNetlistJson(root)) {
            results.push_back(fs::absolute(root));
        } else if (fs::exists(root, ec) && fs::is_directory(root, ec)) {
            const fs::path unified = root / "eext_netlist.json";
            std::error_code unifiedEc;
            if (fs::exists(unified, unifiedEc) && fs::is_regular_file(unified, unifiedEc) && isNetlistJson(unified)) {
                results.push_back(fs::absolute(unified));
            }
            for (const auto &entry : fs::directory_iterator(root, ec)) {
                if (ec) {
                    break;
                }
                if (!entry.is_regular_file()) {
                    continue;
                }
                if (isNetlistJson(entry.path())) {
                    if (!results.empty() && lowerCopy(results.front().filename().string()) == "eext_netlist.json") {
                        continue;
                    }
                    results.push_back(fs::absolute(entry.path()));
                }
            }
        }
    }

    std::sort(results.begin(), results.end(), [](const fs::path &left, const fs::path &right) {
        return lowerCopy(left.filename().string()) < lowerCopy(right.filename().string());
    });
    results.erase(std::unique(results.begin(), results.end()), results.end());
    return results;
}

json fileListToJson(const std::vector<fs::path> &paths) {
    json out = json::array();
    for (const auto &path : paths) {
        out.push_back(pathText(path));
    }
    return out;
}

json stringListToJson(const std::vector<std::string> &values) {
    json out = json::array();
    for (const auto &value : values) {
        out.push_back(value);
    }
    return out;
}

std::vector<std::string> sortedInterfaceKeys(const PlatformManager::TopologyCache &topology) {
    std::vector<std::string> keys;
    keys.reserve(topology.interfaces.size());
    for (const auto &entry : topology.interfaces) {
        keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

bool interfaceAvailable(const PlatformManager::TopologyCache &topology, const std::string &id) {
    const auto it = topology.interfaces.find(id);
    return it != topology.interfaces.end() && it->second.available;
}

json interfaceMapToJson(const PlatformManager::TopologyCache &topology) {
    json out = json::object();
    for (const auto &key : sortedInterfaceKeys(topology)) {
        out[key] = topology.interfaces.at(key).toJson();
    }
    return out;
}

void classifySignal(PlatformManager::TopologyCache &topology, const PlatformManager::SignalBinding &binding) {
    const std::string signalUpper = upperCopy(binding.signal);
    const bool npuBoundary = containsAny(signalUpper, {"DAC_", "NPU_SAMPLE_SYNC", "CARR", "SLICE_SUM", "_CMP"});
    const bool weightConfigSideband = containsAny(signalUpper, {"WEIGHT_CFG"});
    const bool npuSideband = npuBoundary || weightConfigSideband;

    if (isPowerSignal(signalUpper)) {
        addBinding(topology, "power", "power", binding);
    }
    if (isGroundSignal(signalUpper)) {
        addBinding(topology, "ground", "ground", binding);
    }
    if (npuBoundary) {
        addBinding(topology, "npu-boundary", "gpio-npu", binding);
    }
    if (weightConfigSideband) {
        addBinding(topology, "weight-config", "i2c-sideband", binding);
    }
    if (npuSideband) {
        pushUnique(topology.npuControlSignals, binding.signal);
        if (containsAny(signalUpper, {"DAC_IN", "NPU_SAMPLE_SYNC", "_CMP"})) {
            pushUnique(topology.npuTransportSignals, binding.signal);
        }
    }
    if (!npuBoundary && containsAny(signalUpper, {"MOSI", "MISO", "SCLK", "_CS", "CE0", "CE1"})) {
        addBinding(topology, "spi0", "spi", binding);
    }
    if (signalUpper.rfind("GPIO", 0) == 0) {
        addBinding(topology, "gpio-bank", "gpio", binding);
    }
}

std::string jsonText(const json &doc) {
    return doc.dump(2);
}

} // namespace

json PlatformManager::InterfaceSummary::toJson() const {
    return json{{"id", id},
                {"type", type},
                {"available", available},
                {"signalCount", signals.size()},
                {"signals", stringListToJson(signals)},
                {"bindings", stringListToJson(bindings)}};
}

json PlatformManager::PhysicalPinSummary::toJson() const {
    return json{{"connectorId", connectorId},
                {"pinNumber", pinNumber},
                {"logicalNet", logicalNet},
                {"boardNet", boardNet},
                {"exactMatch", exactMatch}};
}

json PlatformManager::ConnectorSummary::toJson() const {
    json pinsJson = json::object();
    for (const auto &entry : pins) {
        pinsJson[entry.first] = entry.second.toJson();
    }
    return json{{"id", id}, {"role", role}, {"pinCount", pins.size()}, {"pins", pinsJson}};
}

PlatformManager::PlatformManager() {
    metrics_.lastDecayAtMs = nowMs();
}

void PlatformManager::ensureNpuLanesLocked() {
    const int laneCount = std::max(1, config_.npuUnitCount);
    if (static_cast<int>(npuLanes_.size()) == laneCount) {
        return;
    }
    npuLanes_.clear();
    npuLanes_.reserve(static_cast<std::size_t>(laneCount));
    for (int lane = 0; lane < laneCount; ++lane) {
        NpuLaneState state;
        state.laneId = lane;
        npuLanes_.push_back(state);
    }
}

json PlatformManager::buildFleetScheduleLocked(int shardCount, int preferredBatch) const {
    const int laneCount = std::max(1, config_.npuUnitCount);
    const int lanes = std::max(1, std::min(laneCount, static_cast<int>(npuLanes_.size())));
    const int actualShards = std::max(1, shardCount);
    std::vector<int> virtualInflight(lanes, 0);
    for (int lane = 0; lane < lanes; ++lane) {
        virtualInflight[lane] = std::max(0, npuLanes_[lane].inflight);
    }

    json assignments = json::array();
    for (int shard = 0; shard < actualShards; ++shard) {
        int bestLane = 0;
        double bestScore = std::numeric_limits<double>::max();
        for (int lane = 0; lane < lanes; ++lane) {
            const double latencyPenalty = npuLanes_[lane].ewmaLatencyMs > 0.0 ? npuLanes_[lane].ewmaLatencyMs / 50.0 : 0.0;
            const double assignPenalty = static_cast<double>(virtualInflight[lane]) * 1.5;
            const double score = assignPenalty + latencyPenalty;
            if (score < bestScore) {
                bestScore = score;
                bestLane = lane;
            }
        }
        virtualInflight[bestLane] += 1;
        assignments.push_back(json{{"shard", shard}, {"lane", bestLane}, {"batch", std::max(1, preferredBatch)}});
    }

    int activeLanes = 0;
    int idleLanes = 0;
    for (int lane = 0; lane < lanes; ++lane) {
        if (npuLanes_[lane].inflight > 0 || npuLanes_[lane].assigned > 0) {
            activeLanes += 1;
        } else {
            idleLanes += 1;
        }
    }
    const double utilization = lanes <= 0 ? 0.0 : static_cast<double>(activeLanes) / static_cast<double>(lanes);

    return json{{"laneCount", lanes},
                {"assignments", assignments},
                {"activeLanes", activeLanes},
                {"idleLanes", idleLanes},
                {"utilization", utilization},
                {"planner", "weighted-least-inflight-v1"}};
}

json PlatformManager::buildWeightVirtualizationViewLocked(const json &payload) const {
    const int tensorBytes = std::max(0, safeJsonValue(payload, "tensorBytes", safeJsonValue(payload, "bytes", safeJsonValue(payload, "activationBytes", 0))));
    const int weightBlockId = resolveWeightBlockId(payload);
    const std::size_t weightBytes = resolveWeightBytes(payload, tensorBytes);
    const auto found = weightResidency_.find(weightBlockId);
    const bool hot = found != weightResidency_.end() && found->second.hot;
    const uint64_t totalHits = found != weightResidency_.end() ? found->second.totalHits : 0;
    const bool presentOnSd = found != weightResidency_.end() ? found->second.presentOnSd : true;
    const bool streamWeights = payload.is_object() && payload.contains("streamWeights") ? safeJsonValue(payload, "streamWeights", false) : !hot;

    const int advertisedMb = std::max(1024, config_.npuAdvertisedMemoryMb);
    const int requiredMb = static_cast<int>((weightBytes + (1024 * 1024 - 1)) / (1024 * 1024));
    const bool virtualized = config_.npuVirtualMemoryEnabled;
    const bool memoryAdmitted = virtualized || (requiredMb <= advertisedMb);

    return json{{"enabled", virtualized},
                {"advertisedMemoryMb", advertisedMb},
                {"requiredWeightMb", requiredMb},
                {"admitted", memoryAdmitted},
                {"weightBlockId", weightBlockId},
                {"weightBytes", static_cast<uint64_t>(weightBytes)},
                {"hot", hot},
                {"totalHits", totalHits},
                {"presentOnSd", presentOnSd},
                {"sdcardRoot", pathText(resolvePath(config_.npuSdcardWeightsRoot, config_.baseDir))},
                {"streamWeights", streamWeights},
                {"policy", hot ? "hot-resident" : "sdcard-on-demand"}};
}

void PlatformManager::updateNpuWeightResidencyLocked(const json &result, bool accepted) {
    if (!accepted) {
        return;
    }
    const json vm = result.value("virtualMemory", json::object());
    const int weightBlockId = std::max(0, vm.value("weightBlockId", 0));
    const std::size_t weightBytes = static_cast<std::size_t>(vm.value("weightBytes", static_cast<uint64_t>(0)));

    auto &entry = weightResidency_[weightBlockId];
    entry.weightBlockId = weightBlockId;
    entry.weightBytes = weightBytes;
    entry.presentOnSd = true;
    entry.totalHits += 1;
    entry.lastAccessMs = nowMs();
    const bool wasHot = entry.hot;
    if (entry.hot) {
        entry.hotHits += 1;
        metrics_.weightHotHits += 1;
    } else {
        metrics_.weightColdLoads += 1;
    }
    if (!entry.hot && static_cast<int>(entry.totalHits) >= std::max(1, config_.npuHotPromoteHits)) {
        entry.hot = true;
        entry.promotedAtMs = nowMs();
        metrics_.weightPromotions += 1;
    }

    int hotCount = 0;
    for (const auto &item : weightResidency_) {
        if (item.second.hot) {
            hotCount += 1;
        }
    }
    const int hotLimit = std::max(1, config_.npuHotWeightsLimit);
    if (hotCount > hotLimit) {
        int demoteId = -1;
        int64_t oldestAccess = std::numeric_limits<int64_t>::max();
        for (const auto &item : weightResidency_) {
            if (!item.second.hot || item.first == weightBlockId) {
                continue;
            }
            if (item.second.lastAccessMs < oldestAccess) {
                oldestAccess = item.second.lastAccessMs;
                demoteId = item.first;
            }
        }
        if (demoteId >= 0) {
            weightResidency_[demoteId].hot = false;
        }
    }

    if (wasHot) {
        metrics_.weightHotHits += 0;
    }
}

void PlatformManager::updateFleetProbeLocked(const json &result, bool accepted, const json &execution) {
    if (!accepted || !result.contains("fleet") || !result["fleet"].is_object()) {
        return;
    }
    ensureNpuLanesLocked();
    const json assignments = result["fleet"].value("assignments", json::array());
    if (!assignments.is_array() || assignments.empty()) {
        return;
    }

    double observedLatencyMs = execution.value("totalElapsedMs", 0.0);
    if (observedLatencyMs <= 0.0) {
        const json protocol = result.value("protocol", json::object());
        const int windowUs = protocol.value("sampleWindowUs", 0);
        const int cycleCount = static_cast<int>(protocol.value("cyclePlan", protocol.value("windowPlan", json::array())).size());
        if (windowUs > 0 && cycleCount > 0) {
            observedLatencyMs = static_cast<double>(windowUs) * static_cast<double>(cycleCount) / 1000.0;
        }
    }
    if (observedLatencyMs <= 0.0) {
        observedLatencyMs = 1.0;
    }
    const double perLaneLatency = observedLatencyMs / std::max(1.0, static_cast<double>(assignments.size()));

    for (const auto &item : assignments) {
        if (!item.is_object()) {
            continue;
        }
        const int lane = std::max(0, item.value("lane", 0));
        if (lane >= static_cast<int>(npuLanes_.size())) {
            continue;
        }
        auto &state = npuLanes_[lane];
        state.assigned += 1;
        state.completed += 1;
        state.ewmaLatencyMs = updateEwma(state.ewmaLatencyMs, perLaneLatency, 0.2);
        state.lastAssignedAtMs = nowMs();
        state.inflight = std::max(0, state.inflight);
        metrics_.fleetAssignments += 1;
    }

    double efficiencyAccum = 0.0;
    int efficiencyCount = 0;
    json laneScores = json::array();
    for (const auto &lane : npuLanes_) {
        if (lane.assigned == 0) {
            continue;
        }
        const double latencyScore = lane.ewmaLatencyMs <= 0.0 ? 1.0 : std::min(1.0, 8.0 / lane.ewmaLatencyMs);
        efficiencyAccum += latencyScore;
        efficiencyCount += 1;
        laneScores.push_back(json{{"lane", lane.laneId},
                                  {"assigned", lane.assigned},
                                  {"completed", lane.completed},
                                  {"ewmaLatencyMs", lane.ewmaLatencyMs},
                                  {"score", latencyScore}});
    }
    const double globalScore = efficiencyCount > 0 ? (efficiencyAccum / static_cast<double>(efficiencyCount)) : 1.0;
    metrics_.accelEfficiencyScore = globalScore;
    (void)laneScores;
}

void PlatformManager::reconfigure(const RuntimeConfig &config) {
    std::lock_guard<std::mutex> lock(mu_);
    config_ = config;
    config_.preferredComputeBackend = normalizeBackend(config_.preferredComputeBackend);
    config_.maxComputeInflight = std::max(1, config_.maxComputeInflight);
    config_.maxPeripheralInflight = std::max(1, config_.maxPeripheralInflight);
    config_.npuSpiSpeedHz = std::max(100000, config_.npuSpiSpeedHz);
    config_.npuSpiMode = std::max(0, std::min(3, config_.npuSpiMode));
    config_.npuGpioPulseUs = std::max(0, std::min(1000, config_.npuGpioPulseUs));
    config_.npuAdvertisedMemoryMb = std::max(1024, config_.npuAdvertisedMemoryMb);
    config_.npuHotWeightsLimit = std::max(1, config_.npuHotWeightsLimit);
    config_.npuHotPromoteHits = std::max(1, config_.npuHotPromoteHits);
    config_.npuUnitCount = std::max(1, config_.npuUnitCount);
    config_.npuEfficiencyAnomalyThreshold = std::max(0.05, std::min(1.0, config_.npuEfficiencyAnomalyThreshold));
    if (config_.baseDir.empty()) {
        config_.baseDir = fs::current_path();
    }
    if (config_.netlistRoot.empty() && config_.netlistFiles.empty()) {
        config_.netlistRoot = config_.baseDir / "catastrophe";
    }
    if (config_.gerberConnectorMap.empty()) {
        config_.gerberConnectorMap = defaultGerberConnectorMap(config_);
    } else {
        config_.gerberConnectorMap = resolvePath(config_.gerberConnectorMap, config_.baseDir);
    }
    if (config_.npuSpiDevice.empty()) {
        config_.npuSpiDevice = "/dev/spidev0.0";
    }
    if (config_.npuGpioSysfsRoot.empty()) {
        config_.npuGpioSysfsRoot = "/sys/class/gpio";
    }
    if (config_.gpioChipDevice.empty()) {
        config_.gpioChipDevice = "/dev/gpiochip0";
    }
    if (config_.npuSdcardWeightsRoot.empty()) {
        config_.npuSdcardWeightsRoot = fs::path("runtime_store") / "sdcard_weights";
    }
    ensureNpuLanesLocked();
}

PlatformManager::TopologyCache PlatformManager::buildTopologyLocked() const {
    TopologyCache topology;
    topology.root = resolvePath(config_.netlistRoot.empty() ? fs::path("catastrophe") : config_.netlistRoot, config_.baseDir);
    topology.files = discoverNetlistFiles(config_);
    if (topology.files.empty()) {
        topology.warnings.push_back("no eext netlist json files were discovered under the configured catastrophe root");
        return topology;
    }

    for (const auto &file : topology.files) {
        std::ifstream input(file, std::ios::binary);
        if (!input) {
            topology.warnings.push_back("failed to open netlist: " + pathText(file));
            continue;
        }

        json doc = json::parse(input, nullptr, false);
        if (doc.is_discarded() || !doc.is_object()) {
            topology.warnings.push_back("invalid netlist json: " + pathText(file));
            continue;
        }

        for (auto componentIt = doc.begin(); componentIt != doc.end(); ++componentIt) {
            if (!componentIt.value().is_object()) {
                continue;
            }

            const json &component = componentIt.value();
            const json props = component.contains("props") && component["props"].is_object() ? component["props"] : json::object();
            const json pins = component.contains("pins") && component["pins"].is_object() ? component["pins"] : json::object();
            if (pins.empty()) {
                continue;
            }

            topology.componentCount += 1;
            const std::string designator = trimCopy(props.value("Designator", componentIt.key()));
            const std::string value = trimCopy(props.value("value", props.value("device_name", std::string())));
            const std::string designatorUpper = upperCopy(designator);
            const std::string labelLower = lowerCopy(value);
            if (componentLooksConnector(designatorUpper, labelLower)) {
                pushUnique(topology.connectors, designator + (value.empty() ? std::string() : (" " + value)));
            }

            for (auto pinIt = pins.begin(); pinIt != pins.end(); ++pinIt) {
                if (!pinIt.value().is_string()) {
                    continue;
                }
                const std::string signal = trimCopy(pinIt.value().get<std::string>());
                if (signal.empty()) {
                    continue;
                }
                SignalBinding binding{componentIt.key(), designator, value, pinIt.key(), signal};
                topology.signalBindings[signal].push_back(binding);
                classifySignal(topology, binding);
            }
        }
    }

    mergeGerberConnectorMap(topology, config_);

    for (const auto &signal : collectPresentSignals(topology,
                                                    {"GPIO10_DAC_IN0", "GPIO11_DAC_IN1", "GPIO8_DAC_IN2", "GPIO7_DAC_IN3",
                                                     "GPIO17_DAC_IN4", "GPIO18_DAC_IN5", "GPIO27_DAC_IN6", "GPIO22_DAC_IN7",
                                                     "GPIO23_DAC_IN8", "GPIO24_DAC_IN9", "GPIO9_DAC_IN10", "GPIO25_DAC_IN11",
                                                     "GPIO6_NPU_SAMPLE_SYNC",
                                                     "GPIO5_CARRA_CMP", "GPIO12_CARRB_CMP", "GPIO13_CARRC_CMP", "GPIO16_CARRD_CMP", "GPIO20_CARRE_CMP", "GPIO21_CARRF_CMP",
                                                     "GPIO14_CARRG_CMP", "GPIO15_CARRH_CMP",
                                                     "GPIO4_SLICE_SUM0_CMP", "GPIO26_SLICE_SUM1_CMP"})) {
        pushUnique(topology.npuTransportSignals, signal);
    }
    if (interfaceAvailable(topology, "spi0")) {
        const auto &spiSignals = topology.interfaces["spi0"].signals;
        for (const auto &signal : spiSignals) {
            const std::string signalUpper = upperCopy(signal);
            if (containsAny(signalUpper, {"MOSI", "MISO", "SCLK", "_CS", "CE0", "CE1"})) {
                pushUnique(topology.npuTransportSignals, signal);
            }
        }
    }

    topology.npuAvailable = (interfaceAvailable(topology, "spi0") || hasDirectAsyncBoundary(topology)) && !topology.npuControlSignals.empty();
    topology.loaded = !topology.signalBindings.empty();

    if (!topology.loaded) {
        topology.warnings.push_back("netlist files were discovered but no signal bindings were parsed");
    }
    if (!topology.npuAvailable && !topology.npuControlSignals.empty()) {
        topology.warnings.push_back("npu boundary signals exist but neither a direct gpio boundary nor a fallback spi plane was discovered");
    }
    if (topology.npuAvailable) {
        if (collectPresentSignals(topology,
                                  {"GPIO10_DAC_IN0", "GPIO11_DAC_IN1", "GPIO8_DAC_IN2", "GPIO7_DAC_IN3",
                                   "GPIO17_DAC_IN4", "GPIO18_DAC_IN5", "GPIO27_DAC_IN6", "GPIO22_DAC_IN7",
                                   "GPIO23_DAC_IN8", "GPIO24_DAC_IN9", "GPIO9_DAC_IN10", "GPIO25_DAC_IN11"}).size() != 12) {
            topology.warnings.push_back("npu write boundary is missing one or more direct DAC input lines");
        }
        if (!hasSignalBinding(topology, "GPIO6_NPU_SAMPLE_SYNC")) {
            topology.warnings.push_back("npu boundary is missing GPIO6_NPU_SAMPLE_SYNC");
        }
        if (collectPresentSignals(topology,
                                  {"GPIO5_CARRA_CMP", "GPIO12_CARRB_CMP", "GPIO13_CARRC_CMP", "GPIO16_CARRD_CMP",
                                   "GPIO20_CARRE_CMP", "GPIO21_CARRF_CMP", "GPIO14_CARRG_CMP", "GPIO15_CARRH_CMP"}).size() < 8) {
            topology.warnings.push_back("npu read boundary is missing one or more fast comparator outputs");
        }
    }
    const auto weightCfgBusSignals = collectWeightConfigBusSignals(topology);
    const std::string weightCfgIrqSignal = firstWeightConfigIrqSignal(topology);
    if (!weightCfgBusSignals.empty() || !weightCfgIrqSignal.empty()) {
        if (weightCfgBusSignals.size() != 2) {
            topology.warnings.push_back("weight configuration sideband is missing one or more GPIO2/GPIO3 I2C bus lines");
        }
        if (weightCfgIrqSignal.empty()) {
            topology.warnings.push_back("weight configuration sideband is missing GPIO19_WEIGHT_CFG_INT");
        }
    }
    for (const auto &warning : topology.gerberWarnings) {
        topology.warnings.push_back(warning);
    }
    return topology;
}

void PlatformManager::decayBacklogLocked(int64_t now) {
    if (metrics_.lastDecayAtMs <= 0) {
        metrics_.lastDecayAtMs = now;
        return;
    }
    const double elapsedSec = std::max(0.0, static_cast<double>(now - metrics_.lastDecayAtMs) / 1000.0);
    metrics_.lastDecayAtMs = now;
    metrics_.computeBacklog = std::max(0.0, metrics_.computeBacklog - elapsedSec * 1.5);
}

void PlatformManager::pushHistoryLocked(const json &event) {
    metrics_.recentDispatches.push_front(event);
    while (metrics_.recentDispatches.size() > 24) {
        metrics_.recentDispatches.pop_back();
    }
}

json PlatformManager::buildStatusLocked() const {
    json recent = json::array();
    for (const auto &entry : metrics_.recentDispatches) {
        recent.push_back(entry);
    }

    int hotWeightCount = 0;
    json weightTable = json::array();
    for (const auto &item : weightResidency_) {
        const auto &entry = item.second;
        if (entry.hot) {
            hotWeightCount += 1;
        }
        weightTable.push_back(json{{"weightBlockId", entry.weightBlockId},
                                   {"hot", entry.hot},
                                   {"totalHits", entry.totalHits},
                                   {"hotHits", entry.hotHits},
                                   {"weightBytes", static_cast<uint64_t>(entry.weightBytes)},
                                   {"presentOnSd", entry.presentOnSd},
                                   {"lastAccessMs", entry.lastAccessMs}});
    }

    int activeLanes = 0;
    json laneProbe = json::array();
    for (const auto &lane : npuLanes_) {
        if (lane.inflight > 0 || lane.assigned > 0) {
            activeLanes += 1;
        }
        laneProbe.push_back(json{{"lane", lane.laneId},
                                 {"inflight", lane.inflight},
                                 {"assigned", lane.assigned},
                                 {"completed", lane.completed},
                                 {"ewmaLatencyMs", lane.ewmaLatencyMs}});
    }
    const int totalLanes = std::max(1, static_cast<int>(npuLanes_.size()));
    const double laneUtilization = static_cast<double>(activeLanes) / static_cast<double>(totalLanes);
    const bool efficiencyAnomaly = config_.npuEfficiencyProbeEnabled && metrics_.accelEfficiencyScore < config_.npuEfficiencyAnomalyThreshold;

    return json{{"ok", true},
                {"enabled", config_.enabled},
                {"result",
                 json{{"config",
                       json{{"enabled", config_.enabled},
                            {"npuEnabled", config_.npuEnabled},
                            {"peripheralSchedulingEnabled", config_.peripheralSchedulingEnabled},
                            {"preferredComputeBackend", config_.preferredComputeBackend},
                            {"maxComputeInflight", config_.maxComputeInflight},
                            {"maxPeripheralInflight", config_.maxPeripheralInflight},
                            {"npuAsyncGpioExecuteLocally", config_.npuAsyncGpioExecuteLocally},
                            {"npuGpioSysfsRoot", config_.npuGpioSysfsRoot},
                            {"gpioChipDevice", config_.gpioChipDevice},
                            {"preferGpioChip", config_.preferGpioChip},
                            {"npuGpioPulseUs", config_.npuGpioPulseUs},
                            {"npuVirtualMemoryEnabled", config_.npuVirtualMemoryEnabled},
                            {"npuAdvertisedMemoryMb", config_.npuAdvertisedMemoryMb},
                            {"npuSdcardWeightsRoot", pathText(resolvePath(config_.npuSdcardWeightsRoot, config_.baseDir))},
                            {"npuHotWeightsLimit", config_.npuHotWeightsLimit},
                            {"npuHotPromoteHits", config_.npuHotPromoteHits},
                            {"npuUnitCount", config_.npuUnitCount},
                            {"npuEfficiencyProbeEnabled", config_.npuEfficiencyProbeEnabled},
                            {"npuEfficiencyAnomalyThreshold", config_.npuEfficiencyAnomalyThreshold},
                            {"gerberConnectorMap", pathText(config_.gerberConnectorMap)},
                            {"netlistRoot", pathText(resolvePath(config_.netlistRoot.empty() ? fs::path("catastrophe") : config_.netlistRoot, config_.baseDir))},
                            {"netlistFiles", fileListToJson(config_.netlistFiles)},
                            {"hardware", buildHardwareStatus(config_)}}},
                      {"topology",
                       json{{"loaded", topology_.loaded},
                            {"root", pathText(topology_.root)},
                            {"files", fileListToJson(topology_.files)},
                            {"componentCount", topology_.componentCount},
                            {"signalCount", topology_.signalBindings.size()},
                            {"connectors", stringListToJson(topology_.connectors)},
                            {"interfaces", interfaceMapToJson(topology_)},
                            {"npu",
                             json{{"available", topology_.npuAvailable && config_.npuEnabled},
                                  {"controlSignals", stringListToJson(topology_.npuControlSignals)},
                                  {"transportSignals", stringListToJson(topology_.npuTransportSignals)},
                                                                    {"transport", hasDirectAsyncBoundary(topology_) ? "gpio-async" : (interfaceAvailable(topology_, "spi0") ? "spi0" : "unavailable")},
                                                                    {"asyncBoundaryAvailable", hasDirectAsyncBoundary(topology_)}}},
                                {"gerber",
                                 json{{"loaded", topology_.gerberLoaded},
                                    {"path", pathText(topology_.gerberConnectorMap)},
                                    {"warnings", stringListToJson(topology_.gerberWarnings)},
                                    {"connectors", physicalConnectorMapToJson(topology_)}}},
                            {"warnings", stringListToJson(topology_.warnings)}}},
                      {"metrics",
                       json{{"refreshCount", metrics_.refreshCount},
                            {"computePlanned", metrics_.computePlanned},
                            {"computeDispatched", metrics_.computeDispatched},
                            {"routedCpu", metrics_.routedCpu},
                            {"routedNpu", metrics_.routedNpu},
                            {"routedHybrid", metrics_.routedHybrid},
                            {"rejectedDispatches", metrics_.rejectedDispatches},
                            {"weightHotHits", metrics_.weightHotHits},
                            {"weightColdLoads", metrics_.weightColdLoads},
                            {"weightPromotions", metrics_.weightPromotions},
                            {"fleetAssignments", metrics_.fleetAssignments},
                            {"computeBacklog", metrics_.computeBacklog},
                                                        {"effectiveComputeCapacity", config_.maxComputeInflight},
                                                        {"effectivePeripheralCapacity", config_.maxPeripheralInflight},
                                                        {"accelEfficiencyScore", metrics_.accelEfficiencyScore}}},
                      {"virtualMemory",
                       json{{"enabled", config_.npuVirtualMemoryEnabled},
                            {"advertisedMemoryMb", config_.npuAdvertisedMemoryMb},
                            {"hotWeightCount", hotWeightCount},
                            {"trackedWeightBlocks", weightResidency_.size()},
                            {"weights", weightTable}}},
                      {"fleet",
                       json{{"laneCount", totalLanes},
                            {"activeLanes", activeLanes},
                            {"idleLanes", std::max(0, totalLanes - activeLanes)},
                            {"utilization", laneUtilization},
                            {"planner", "weighted-least-inflight-v1"},
                            {"lanes", laneProbe}}},
                      {"efficiencyProbe",
                       json{{"model", "mini-ewma-latency-v1"},
                            {"enabled", config_.npuEfficiencyProbeEnabled},
                            {"score", metrics_.accelEfficiencyScore},
                            {"anomalyThreshold", config_.npuEfficiencyAnomalyThreshold},
                            {"anomalyDetected", efficiencyAnomaly}}},
                      {"recentDispatches", recent}}}};
}

json PlatformManager::status() {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    return buildStatusLocked();
}

json PlatformManager::refreshTopology() {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    topology_ = buildTopologyLocked();
    metrics_.refreshCount += 1;
    return buildStatusLocked();
}

json PlatformManager::applyPatch(const json &patch) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    json applied = json::object();
    json warnings = json::array();
    bool refresh = false;

    if (patch.contains("enabled") && patch["enabled"].is_boolean()) {
        config_.enabled = patch["enabled"].get<bool>();
        applied["enabled"] = config_.enabled;
    }
    if (patch.contains("npuEnabled") && patch["npuEnabled"].is_boolean()) {
        config_.npuEnabled = patch["npuEnabled"].get<bool>();
        applied["npuEnabled"] = config_.npuEnabled;
    }
    if (patch.contains("preferredComputeBackend") && patch["preferredComputeBackend"].is_string()) {
        config_.preferredComputeBackend = normalizeBackend(patch["preferredComputeBackend"].get<std::string>());
        applied["preferredComputeBackend"] = config_.preferredComputeBackend;
    }
    if (patch.contains("maxComputeInflight") && patch["maxComputeInflight"].is_number_integer()) {
        config_.maxComputeInflight = std::max(1, patch["maxComputeInflight"].get<int>());
        applied["maxComputeInflight"] = config_.maxComputeInflight;
    }
    if (patch.contains("peripheralSchedulingEnabled") && patch["peripheralSchedulingEnabled"].is_boolean()) {
        config_.peripheralSchedulingEnabled = patch["peripheralSchedulingEnabled"].get<bool>();
        applied["peripheralSchedulingEnabled"] = config_.peripheralSchedulingEnabled;
    }
    if (patch.contains("maxPeripheralInflight") && patch["maxPeripheralInflight"].is_number_integer()) {
        config_.maxPeripheralInflight = std::max(1, patch["maxPeripheralInflight"].get<int>());
        applied["maxPeripheralInflight"] = config_.maxPeripheralInflight;
    }
    if (patch.contains("gerberConnectorMap") && patch["gerberConnectorMap"].is_string()) {
        config_.gerberConnectorMap = patch["gerberConnectorMap"].get<std::string>();
        applied["gerberConnectorMap"] = pathText(config_.gerberConnectorMap);
        refresh = true;
    }
    if (patch.contains("npuSpiDevice") && patch["npuSpiDevice"].is_string()) {
        config_.npuSpiDevice = patch["npuSpiDevice"].get<std::string>();
        applied["npuSpiDevice"] = config_.npuSpiDevice;
    }
    if (patch.contains("npuSpiSpeedHz") && patch["npuSpiSpeedHz"].is_number_integer()) {
        config_.npuSpiSpeedHz = std::max(100000, patch["npuSpiSpeedHz"].get<int>());
        applied["npuSpiSpeedHz"] = config_.npuSpiSpeedHz;
    }
    if (patch.contains("npuSpiMode") && patch["npuSpiMode"].is_number_integer()) {
        config_.npuSpiMode = std::max(0, std::min(3, patch["npuSpiMode"].get<int>()));
        applied["npuSpiMode"] = config_.npuSpiMode;
    }
    if (patch.contains("npuAsyncGpioExecuteLocally") && patch["npuAsyncGpioExecuteLocally"].is_boolean()) {
        config_.npuAsyncGpioExecuteLocally = patch["npuAsyncGpioExecuteLocally"].get<bool>();
        applied["npuAsyncGpioExecuteLocally"] = config_.npuAsyncGpioExecuteLocally;
    }
    if (patch.contains("npuGpioSysfsRoot") && patch["npuGpioSysfsRoot"].is_string()) {
        config_.npuGpioSysfsRoot = patch["npuGpioSysfsRoot"].get<std::string>();
        applied["npuGpioSysfsRoot"] = config_.npuGpioSysfsRoot;
    }
    if (patch.contains("gpioChipDevice") && patch["gpioChipDevice"].is_string()) {
        config_.gpioChipDevice = patch["gpioChipDevice"].get<std::string>();
        applied["gpioChipDevice"] = config_.gpioChipDevice;
    }
    if (patch.contains("preferGpioChip") && patch["preferGpioChip"].is_boolean()) {
        config_.preferGpioChip = patch["preferGpioChip"].get<bool>();
        applied["preferGpioChip"] = config_.preferGpioChip;
    }
    if (patch.contains("npuGpioPulseUs") && patch["npuGpioPulseUs"].is_number_integer()) {
        config_.npuGpioPulseUs = std::max(0, std::min(1000, patch["npuGpioPulseUs"].get<int>()));
        applied["npuGpioPulseUs"] = config_.npuGpioPulseUs;
    }
    if (patch.contains("npuVirtualMemoryEnabled") && patch["npuVirtualMemoryEnabled"].is_boolean()) {
        config_.npuVirtualMemoryEnabled = patch["npuVirtualMemoryEnabled"].get<bool>();
        applied["npuVirtualMemoryEnabled"] = config_.npuVirtualMemoryEnabled;
    }
    if (patch.contains("npuAdvertisedMemoryMb") && patch["npuAdvertisedMemoryMb"].is_number_integer()) {
        config_.npuAdvertisedMemoryMb = std::max(1024, patch["npuAdvertisedMemoryMb"].get<int>());
        applied["npuAdvertisedMemoryMb"] = config_.npuAdvertisedMemoryMb;
    }
    if (patch.contains("npuSdcardWeightsRoot") && patch["npuSdcardWeightsRoot"].is_string()) {
        config_.npuSdcardWeightsRoot = fs::path(patch["npuSdcardWeightsRoot"].get<std::string>());
        applied["npuSdcardWeightsRoot"] = pathText(config_.npuSdcardWeightsRoot);
    }
    if (patch.contains("npuHotWeightsLimit") && patch["npuHotWeightsLimit"].is_number_integer()) {
        config_.npuHotWeightsLimit = std::max(1, patch["npuHotWeightsLimit"].get<int>());
        applied["npuHotWeightsLimit"] = config_.npuHotWeightsLimit;
    }
    if (patch.contains("npuHotPromoteHits") && patch["npuHotPromoteHits"].is_number_integer()) {
        config_.npuHotPromoteHits = std::max(1, patch["npuHotPromoteHits"].get<int>());
        applied["npuHotPromoteHits"] = config_.npuHotPromoteHits;
    }
    if (patch.contains("npuUnitCount") && patch["npuUnitCount"].is_number_integer()) {
        config_.npuUnitCount = std::max(1, patch["npuUnitCount"].get<int>());
        ensureNpuLanesLocked();
        applied["npuUnitCount"] = config_.npuUnitCount;
    }
    if (patch.contains("npuEfficiencyProbeEnabled") && patch["npuEfficiencyProbeEnabled"].is_boolean()) {
        config_.npuEfficiencyProbeEnabled = patch["npuEfficiencyProbeEnabled"].get<bool>();
        applied["npuEfficiencyProbeEnabled"] = config_.npuEfficiencyProbeEnabled;
    }
    if (patch.contains("npuEfficiencyAnomalyThreshold") && patch["npuEfficiencyAnomalyThreshold"].is_number()) {
        config_.npuEfficiencyAnomalyThreshold = std::max(0.05, std::min(1.0, patch["npuEfficiencyAnomalyThreshold"].get<double>()));
        applied["npuEfficiencyAnomalyThreshold"] = config_.npuEfficiencyAnomalyThreshold;
    }
    if (patch.contains("netlistRoot") && patch["netlistRoot"].is_string()) {
        config_.netlistRoot = patch["netlistRoot"].get<std::string>();
        applied["netlistRoot"] = pathText(config_.netlistRoot);
        refresh = true;
    }
    if (patch.contains("netlistFiles") && patch["netlistFiles"].is_array()) {
        config_.netlistFiles.clear();
        for (const auto &item : patch["netlistFiles"]) {
            if (item.is_string()) {
                config_.netlistFiles.push_back(item.get<std::string>());
            }
        }
        applied["netlistFiles"] = fileListToJson(config_.netlistFiles);
        refresh = true;
    }
    if (patch.contains("refresh") && patch["refresh"].is_boolean()) {
        refresh = patch["refresh"].get<bool>() || refresh;
        applied["refresh"] = refresh;
    }

    if (refresh || !topology_.loaded) {
        topology_ = buildTopologyLocked();
        metrics_.refreshCount += 1;
    }
    for (const auto &warning : topology_.warnings) {
        warnings.push_back(warning);
    }
    return json{{"ok", true}, {"result", json{{"applied", applied}, {"warnings", warnings}, {"status", buildStatusLocked()["result"]}}}};
}

json PlatformManager::buildComputePlanLocked(const json &payload) {
    ensureNpuLanesLocked();
    const std::string op = lowerCopy(trimCopy(safeJsonValue(payload, "operation", safeJsonValue(payload, "opType", safeJsonValue(payload, "kind", std::string("chat-inference"))))));
    const int tokens = std::max(0, safeJsonValue(payload, "tokens", safeJsonValue(payload, "promptTokens", safeJsonValue(payload, "sequenceLength", 0))));
    const int tensorBytes = std::max(0, safeJsonValue(payload, "tensorBytes", safeJsonValue(payload, "bytes", safeJsonValue(payload, "activationBytes", 0))));
    const int latencyBudgetMs = std::max(0, safeJsonValue(payload, "latencyBudgetMs", 120));
    const bool controlHeavy = safeJsonValue(payload, "controlHeavy", false) || containsAny(op, {"tool", "routing", "json", "control", "planner"});
    const bool postProcess = safeJsonValue(payload, "postProcessOnCpu", safeJsonValue(payload, "needsPostProcess", false));
    const bool streaming = safeJsonValue(payload, "streaming", false);
    const bool allowCpu = safeJsonValue(payload, "allowCpu", true);
    const bool horizonDnnRequested = rdk_x5_bpu::requested(payload);
    const bool horizonDnnAvailable = horizonDnnRequested && rdk_x5_bpu::available();
    const bool allowNpu = safeJsonValue(payload, "allowNpu", true) && config_.enabled && config_.npuEnabled && (topology_.npuAvailable || horizonDnnAvailable);
    const bool rawTransportProvided = (payload.is_object() && payload.contains("transportFrames") && payload["transportFrames"].is_array()) ||
                                      (payload.is_object() && payload.contains("spiTxHex") && payload["spiTxHex"].is_string());
    const bool directAsyncBoundary = hasDirectAsyncBoundary(topology_);
    const bool weightControlAvailable = collectWeightConfigBusSignals(topology_).size() == 2;
    const std::string weightControllerFaultMode = resolveWeightControllerFaultMode(payload);
    const std::string preferredTransport = lowerCopy(trimCopy(safeJsonValue(payload, "preferredTransport", safeJsonValue(payload, "transportHint", std::string("auto")))));
    const json virtualMemory = buildWeightVirtualizationViewLocked(payload);
    const bool memoryAdmitted = virtualMemory.value("admitted", true);
    std::string preferred = config_.preferredComputeBackend;
    if (payload.is_object() && payload.contains("preferredBackend") && payload["preferredBackend"].is_string()) {
        preferred = normalizeBackend(payload["preferredBackend"].get<std::string>());
    }

    // 矩阵维度感知优化：检测32×16矩阵适配性
    const int matrixRows = safeJsonValue(payload, "matrixRows", 0);
    const int matrixCols = safeJsonValue(payload, "matrixCols", 0);
    const bool isMatrixCompatible = (matrixRows > 0 && matrixCols > 0) ?
                                    (matrixRows % 32 == 0 && matrixCols % 16 == 0) :
                                    (tensorBytes % 512 == 0); // 512交叉点对齐
    const int matrixBlockCount = isMatrixCompatible ?
                                  ((matrixRows > 0 && matrixCols > 0) ? (matrixRows / 32) * (matrixCols / 16) : (tensorBytes / 512)) :
                                  0;

    // 二值网络检测优化
    const bool isBinaryNetwork = safeJsonValue(payload, "isBinaryNetwork", false) ||
                                 safeJsonValue(payload, "weightPrecision", std::string("")) == "binary" ||
                                 safeJsonValue(payload, "weightPrecision", std::string("")) == "ternary" ||
                                 containsAny(op, {"binary", "ternary", "bnn", "xnor"});

    const bool dense = containsAny(op, {"matmul", "attention", "conv", "embedding", "tensor", "analog", "npu", "vector", "decode"}) || tensorBytes >= 65536 || tokens >= 128;
    std::string mode = "cpu";
    std::vector<std::string> reasons;

    if (!config_.enabled) {
        mode = "cpu";
        reasons.push_back("platform disabled; fall back to local CPU path");
    } else if (horizonDnnRequested && !horizonDnnAvailable) {
        mode = allowCpu ? "cpu" : "rejected";
        reasons.push_back("RDK X5 hbDNN model was requested but the BPU runtime is unavailable");
    } else if (horizonDnnAvailable) {
        mode = postProcess || streaming ? "hybrid" : "npu";
        reasons.push_back("explicit RDK X5 hbDNN model uses the native BPU runtime");
    } else if (!memoryAdmitted && !config_.npuVirtualMemoryEnabled) {
        mode = allowCpu ? "cpu" : "rejected";
        reasons.push_back("insufficient physical memory and virtual memory management disabled");
    } else if (!allowNpu) {
        mode = allowCpu ? "cpu" : "rejected";
        reasons.push_back("npu route unavailable or not allowed for this request");
    } else if (preferred == "cpu") {
        mode = "cpu";
        reasons.push_back("preferred backend forced to cpu");
    } else if (preferred == "npu") {
        mode = controlHeavy && postProcess ? "hybrid" : "npu";
        reasons.push_back("preferred backend forced to npu");
    } else if (preferred == "hybrid") {
        mode = "hybrid";
        reasons.push_back("preferred backend forced to hybrid");
    } else if (isBinaryNetwork && isMatrixCompatible && directAsyncBoundary) {
        // 优化：二值网络 + 矩阵兼容 + GPIO异步边界 → 优先NPU
        mode = "npu";
        reasons.push_back("binary/ternary network with 32x16 matrix compatible dimensions optimized for analog NPU");
    } else if (isMatrixCompatible && matrixBlockCount > 0 && directAsyncBoundary) {
        // 优化：矩阵维度匹配 → 优先NPU
        mode = "npu";
        reasons.push_back("tensor dimensions align with 32x16 hardware matrix blocks (" + std::to_string(matrixBlockCount) + " blocks)");
    } else if (controlHeavy && tokens < 96 && tensorBytes < 32768 && latencyBudgetMs <= 40) {
        mode = allowCpu ? "cpu" : "rejected";
        reasons.push_back("request is control heavy and latency sensitive");
    } else if (dense && (postProcess || streaming || tokens > 512)) {
        mode = "hybrid";
        reasons.push_back("dense tensor work benefits from npu offload with cpu post processing");
    } else if (dense) {
        mode = "npu";
        reasons.push_back("dense tensor work matches gpio-attached npu path");
    } else if (tokens > 1024) {
        mode = "hybrid";
        reasons.push_back("long context should be chunked across npu and cpu");
    } else {
        mode = allowCpu ? "cpu" : "npu";
        reasons.push_back("request remains cheaper on the raspberry pi host cpu");
    }

    if (mode == "rejected") {
        reasons.push_back("request disabled both cpu and npu execution paths");
    }
    if (config_.npuVirtualMemoryEnabled) {
        reasons.push_back("virtual memory manager keeps cold weights on sdcard and reserves hot weights in memory");
    }
    if (weightControllerFaulted(weightControllerFaultMode)) {
        reasons.push_back("weight controller fault injection is active; retain current matrix weights and keep the raspberry pi compute hot path independent");
    }

    std::string transport = "local-cpu";
    if (mode != "cpu") {
        if (horizonDnnAvailable) {
            transport = "horizon-dnn";
            reasons.push_back("using the RDK X5 libdnn runtime and CMA-backed BPU buffers");
        } else if ((preferredTransport == "spi0" || rawTransportProvided) && interfaceAvailable(topology_, "spi0")) {
            transport = "spi0";
            reasons.push_back("using spi0 because raw transport frames were supplied or spi0 was explicitly requested");
        } else if ((preferredTransport == "gpio-async" || (preferredTransport == "auto" && directAsyncBoundary)) && directAsyncBoundary) {
            transport = "gpio-async";
            reasons.push_back("using direct gpio async boundary because the current catastrophe netlist exports dedicated DAC write lanes and comparator readback lanes");
        } else if (interfaceAvailable(topology_, "spi0")) {
            transport = "spi0";
            reasons.push_back("falling back to spi0 transport because direct gpio async signals are incomplete");
        } else if (directAsyncBoundary) {
            transport = "gpio-async";
            reasons.push_back("using gpio async boundary because no spi transport plane was discovered");
        } else {
            transport = "unavailable";
        }
    }
    const bool hardwareReady = mode == "cpu" ? true : (transport == "horizon-dnn" ? horizonDnnAvailable : (transport == "spi0" ? pathLooksAvailable(config_.npuSpiDevice) : directAsyncBoundary));
    const bool localAsyncGpioRequested = safeJsonValue(payload, "executeGpioAsync", config_.npuAsyncGpioExecuteLocally) ||
                              (payload.is_object() && payload.contains("inputCycles") && payload["inputCycles"].is_array()) ||
                              (payload.is_object() && payload.contains("inputWindows") && payload["inputWindows"].is_array());
    const std::string localAsyncDriver = preferredAsyncGpioExecutionDriver(config_);
    const std::string driver = mode == "cpu"
                                   ? "host-cpu"
                                   : (transport == "horizon-dnn"
                                          ? "horizon-hbdnn"
                                          : (transport == "spi0"
                                                 ? (hardwareReady ? "linux-spidev" : "scheduled-envelope")
                                                 : (transport == "gpio-async"
                                                        ? (localAsyncGpioRequested ? localAsyncDriver
                                                                                   : "async-gpio-envelope")
                                                        : "scheduled-envelope")));
    const int shardCount = std::max(1, std::min(8, std::max(tokens / 256, tensorBytes / 262144) + ((mode == "hybrid") ? 1 : 0)));
    const int preferredBatch = std::max(1, std::min(16, safeJsonValue(payload, "batch", dense ? 4 : 1)));
    const json fleetPlan = buildFleetScheduleLocked(shardCount, preferredBatch);
    const bool streamWeights = virtualMemory.value("streamWeights", !virtualMemory.value("hot", false));

    json result{{"requestSummary",
                 json{{"operation", op},
                      {"tokens", tokens},
                      {"tensorBytes", tensorBytes},
                      {"latencyBudgetMs", latencyBudgetMs},
                      {"controlHeavy", controlHeavy},
                      {"postProcessOnCpu", postProcess},
                      {"streaming", streaming}}},
                {"route",
                 json{{"mode", mode},
                      {"transport", transport},
                      {"driver", driver},
                      {"hardwareReady", hardwareReady},
                      {"npuAvailable", topology_.npuAvailable && config_.npuEnabled},
                        {"weightControlAvailable", weightControlAvailable},
                        {"weightControlFaultMode", weightControllerFaultMode},
                        {"weightControlDegraded", weightControllerFaulted(weightControllerFaultMode)},
                      {"preferredBackend", preferred},
                      {"asyncBoundaryAvailable", directAsyncBoundary},
                      {"sidebandSignals", stringListToJson(topology_.npuControlSignals)},
                      {"reason", stringListToJson(reasons)}}},
                {"chunking", json{{"shardCount", shardCount}, {"preferredBatch", preferredBatch}}},
                {"virtualMemory", virtualMemory},
                {"fleet", fleetPlan},
                {"queue",
                 json{{"computeBacklog", metrics_.computeBacklog},
                                                {"computeCapacity", config_.maxComputeInflight}}},
                {"topologyDigest",
                 json{{"loaded", topology_.loaded},
                      {"fileCount", topology_.files.size()},
                      {"signalCount", topology_.signalBindings.size()},
                      {"transportSignals", stringListToJson(topology_.npuTransportSignals)}}},
                {"efficiencyProbe",
                 json{{"model", "mini-ewma-latency-v1"},
                      {"enabled", config_.npuEfficiencyProbeEnabled},
                      {"score", metrics_.accelEfficiencyScore},
                      {"anomalyThreshold", config_.npuEfficiencyAnomalyThreshold},
                      {"anomalyDetected", config_.npuEfficiencyProbeEnabled && metrics_.accelEfficiencyScore < config_.npuEfficiencyAnomalyThreshold}}}};
    if (mode != "cpu" && mode != "rejected" && transport != "horizon-dnn") {
        json protocolPayload = payload.is_object() ? payload : json::object();
        if (!protocolPayload.contains("streamWeights")) {
            protocolPayload["streamWeights"] = streamWeights;
        }
        if (!protocolPayload.contains("weightBlockId")) {
            protocolPayload["weightBlockId"] = virtualMemory.value("weightBlockId", 0);
        }
        json protocol = buildNpuProtocolEnvelope(topology_,
                                                 config_,
                                                 protocolPayload,
                                                 mode,
                                                 transport,
                                                 shardCount,
                                                 preferredBatch,
                                                 tokens,
                                                 tensorBytes,
                                                 streaming,
                                                 postProcess);
        result["route"]["executionModel"] = protocol.value("executionModel", std::string("ordered-sync"));
        result["route"]["transportProtocol"] = protocol.value("name", std::string("phoenix-gpio-async-npu-v1"));
        result["protocol"] = protocol;
    }
    return result;
}

json PlatformManager::planCompute(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    metrics_.computePlanned += 1;
    return json{{"ok", true}, {"result", buildComputePlanLocked(payload)}};
}

json PlatformManager::planMobility(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    metrics_.computePlanned += 1;
    return json{{"ok", true}, {"result", buildPeripheralPlanSummary(topology_, metrics_, config_, payload, "mobility", "mobility")}};
}

json PlatformManager::dispatchCompute(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    json plan = buildComputePlanLocked(payload);
    const std::string mode = plan["route"].value("mode", std::string("cpu"));
    const bool accepted = mode != "rejected" && metrics_.computeBacklog < static_cast<double>(config_.maxComputeInflight);
    const uint64_t dispatchId = metrics_.nextDispatchId++;

    if (accepted) {
        metrics_.computeDispatched += 1;
        metrics_.computeBacklog += mode == "hybrid" ? 1.5 : 1.0;
        if (mode == "cpu") {
            metrics_.routedCpu += 1;
        } else if (mode == "npu") {
            metrics_.routedNpu += 1;
        } else if (mode == "hybrid") {
            metrics_.routedHybrid += 1;
        }
    } else {
        metrics_.rejectedDispatches += 1;
    }

    json event{{"dispatchId", dispatchId},
               {"kind", "compute"},
               {"accepted", accepted},
               {"mode", mode},
               {"transport", plan["route"].value("transport", std::string("local-cpu"))},
               {"operation", plan["requestSummary"].value("operation", std::string("chat-inference"))},
               {"createdAtMs", nowMs()}};
    pushHistoryLocked(event);

    json execution = json::object();
    std::vector<std::vector<uint8_t>> frames;
    std::string frameError;
    bool autoBuiltFrames = false;
    const bool wantsLocalAsyncGpio = accepted &&
                                     mode != "cpu" &&
                                     plan["route"].value("transport", std::string()) == "gpio-async" &&
                                     (safeJsonValue(payload, "executeGpioAsync", config_.npuAsyncGpioExecuteLocally) ||
                                      (payload.is_object() && payload.contains("inputCycles") && payload["inputCycles"].is_array()) ||
                                      (payload.is_object() && payload.contains("inputWindows") && payload["inputWindows"].is_array()));
    if (accepted && rdk_x5_bpu::requested(payload)) {
        execution = rdk_x5_bpu::execute(payload);
    } else if (wantsLocalAsyncGpio && plan.contains("protocol") && plan["protocol"].is_object() &&
        plan["protocol"].contains("gpioPlan") && plan["protocol"]["gpioPlan"].is_array()) {
        std::string gpioError;
        if (executeAsyncGpioPlan(config_, plan["protocol"], execution, gpioError)) {
            autoBuiltFrames = plan["protocol"].value("autoBuiltFrames", false);
            execution["autoBuiltFrames"] = autoBuiltFrames;
        } else {
            execution = json{{"executed", false},
                             {"scheduled", false},
                             {"driver", preferredAsyncGpioExecutionDriver(config_)},
                             {"error", gpioError}};
        }
    } else if (accepted && mode != "cpu" && parseSpiFrames(payload, frames, frameError) && !frames.empty()) {
#ifdef __linux__
        if (plan["route"].value("transport", std::string()) == "spi0") {
            std::vector<std::vector<uint8_t>> responses;
            std::string hardwareError;
            if (performSpiTransfer(config_, frames, responses, hardwareError)) {
                json responseFrames = json::array();
                for (const auto &response : responses) {
                    responseFrames.push_back(bytesToHex(response));
                }
                execution = json{{"executed", true},
                                 {"frameCount", frames.size()},
                                 {"responses", responseFrames},
                                 {"driver", plan["route"].value("driver", std::string("linux-spidev"))}};
            } else {
                execution = json{{"executed", false},
                                 {"driver", plan["route"].value("driver", std::string("linux-spidev"))},
                                 {"error", hardwareError}};
            }
        } else {
            execution = json{{"executed", false},
                             {"scheduled", true},
                             {"frameCount", frames.size()},
                             {"driver", plan["route"].value("driver", std::string("async-gpio-envelope"))},
                             {"note", "async gpio frames prepared for a board-side controller; host-side direct gpio execution is not available in this process"}};
        }
#else
        execution = json{{"executed", false},
                         {"scheduled", true},
                         {"driver", plan["route"].value("driver", std::string("scheduled-envelope"))},
                         {"frameCount", frames.size()},
                         {"note", "transport frames were prepared but direct hardware execution is only available on linux targets"}};
#endif
    } else if (accepted && mode != "cpu" && plan.contains("protocol") && plan["protocol"].is_object() &&
               plan["protocol"].contains("transportFrames") && plan["protocol"]["transportFrames"].is_array() &&
               safeJsonValue(payload, "autoBuildTransportFrames", true)) {
        json framePayload{{"transportFrames", plan["protocol"]["transportFrames"]}};
        if (parseSpiFrames(framePayload, frames, frameError) && !frames.empty()) {
            autoBuiltFrames = true;
#ifdef __linux__
            if (plan["route"].value("transport", std::string()) == "spi0") {
                std::vector<std::vector<uint8_t>> responses;
                std::string hardwareError;
                if (performSpiTransfer(config_, frames, responses, hardwareError)) {
                    json responseFrames = json::array();
                    for (const auto &response : responses) {
                        responseFrames.push_back(bytesToHex(response));
                    }
                    execution = json{{"executed", true},
                                     {"frameCount", frames.size()},
                                     {"responses", responseFrames},
                                     {"driver", plan["route"].value("driver", std::string("linux-spidev"))},
                                     {"autoBuiltFrames", true}};
                } else {
                    execution = json{{"executed", false},
                                     {"driver", plan["route"].value("driver", std::string("linux-spidev"))},
                                     {"error", hardwareError},
                                     {"autoBuiltFrames", true}};
                }
            } else {
                execution = json{{"executed", false},
                                 {"scheduled", true},
                                 {"frameCount", frames.size()},
                                 {"driver", plan["route"].value("driver", std::string("async-gpio-envelope"))},
                                 {"autoBuiltFrames", true},
                                 {"note", "async gpio protocol frames were auto-built and scheduled for a board-side controller"}};
            }
#else
            execution = json{{"executed", false},
                             {"scheduled", true},
                             {"driver", plan["route"].value("driver", std::string("scheduled-envelope"))},
                             {"frameCount", frames.size()},
                             {"autoBuiltFrames", true},
                             {"note", "async gpio protocol frames were auto-built; hardware execution is only available on linux targets"}};
#endif
        }
    } else if (!frameError.empty()) {
        execution = json{{"executed", false}, {"driver", plan["route"].value("driver", std::string("scheduled-envelope"))}, {"error", frameError}};
    }

    json result = plan;
    const std::string dispatchStatus = !accepted
                                           ? "rejected"
                                           : (execution.value("executed", false)
                                                  ? "executed"
                                                  : (execution.value("scheduled", false) ? "scheduled" : "failed"));
    result["dispatch"] = json{{"dispatchId", dispatchId},
                               {"accepted", accepted},
                               {"driver", plan["route"].value("driver", std::string(mode == "cpu" ? "host-cpu" : "scheduled-envelope"))},
                               {"status", dispatchStatus},
                               {"autoBuiltFrames", autoBuiltFrames},
                               {"computeBacklog", metrics_.computeBacklog}};
    if (!execution.empty()) {
        result["execution"] = execution;
    }

    updateNpuWeightResidencyLocked(result, accepted && mode != "cpu");
    updateFleetProbeLocked(result, accepted && mode != "cpu", execution);
    result["virtualMemory"]["hotHitCounter"] = metrics_.weightHotHits;
    result["virtualMemory"]["coldLoadCounter"] = metrics_.weightColdLoads;
    result["efficiencyProbe"] = json{{"model", "mini-ewma-latency-v1"},
                                      {"enabled", config_.npuEfficiencyProbeEnabled},
                                      {"score", metrics_.accelEfficiencyScore},
                                      {"anomalyThreshold", config_.npuEfficiencyAnomalyThreshold},
                                      {"anomalyDetected", config_.npuEfficiencyProbeEnabled && metrics_.accelEfficiencyScore < config_.npuEfficiencyAnomalyThreshold}};
    return json{{"ok", accepted}, {"accepted", accepted}, {"result", result}};
}

json PlatformManager::dispatchPeripheral(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    json plan = buildPeripheralPlanSummary(topology_, metrics_, config_, payload, "peripheral", "peripheral-io");
    json wrapped = buildPeripheralDispatchResult(metrics_, config_, plan, "peripheral", payload);
    pushHistoryLocked(wrapped["event"]);
    return wrapped["response"];
}

json PlatformManager::dispatchMobility(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    json plan = buildPeripheralPlanSummary(topology_, metrics_, config_, payload, "mobility", "mobility");
    json wrapped = buildPeripheralDispatchResult(metrics_, config_, plan, "mobility", payload);
    pushHistoryLocked(wrapped["event"]);
    return wrapped["response"];
}

json PlatformManager::runSelfTest(const json &payload) {
    std::lock_guard<std::mutex> lock(mu_);
    decayBacklogLocked(nowMs());
    const bool refreshTopologyNow = safeJsonValue(payload, "refreshTopology", false) || !topology_.loaded;
    if (refreshTopologyNow) {
        topology_ = buildTopologyLocked();
        metrics_.refreshCount += 1;
    }

    const json computePlan = buildComputePlanLocked(payload);
    const json peripheralPlan = buildPeripheralPlanSummary(topology_, metrics_, config_, payload, "peripheral", "self-test");
    const json mobilityPlan = buildPeripheralPlanSummary(topology_, metrics_, config_, payload, "mobility", "self-test");
    const bool ok = config_.enabled && topology_.loaded;

    return json{{"ok", ok},
                {"result",
                 json{{"summary",
                       json{{"platformEnabled", config_.enabled},
                            {"topologyLoaded", topology_.loaded},
                            {"npuAvailable", topology_.npuAvailable && config_.npuEnabled},
                            {"peripheralSchedulingEnabled", config_.peripheralSchedulingEnabled},
                            {"hardware", buildHardwareStatus(config_)}}},
                      {"computePlan", computePlan},
                      {"peripheralPlan", peripheralPlan},
                      {"mobilityPlan", mobilityPlan},
                      {"status", buildStatusLocked()["result"]}}},
                {"error", ok ? std::string() : std::string("edge platform self-test requires a loaded topology")}};
}

} // namespace edge_platform