/* rpi_zero2w_capability_probe.cpp - Raspberry Pi Zero 2W capability probe tool
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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <linux/spi/spidev.h>
#if defined(__aarch64__) || defined(__arm__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#endif

namespace {

constexpr double kPiBoardWindowBits = 256.0;
constexpr double kRecommendedSafetyFactor = 0.80;
constexpr double kCorrectnessMismatchCeiling = 0.20;
constexpr double kPeakAccuracyFloorPercent = 80.0;
constexpr int kDefaultBenchmarkMs = 800;
constexpr int kDefaultGpioMs = 180;
constexpr int kDefaultSpiMs = 120;
constexpr int kDefaultPeakBenchmarkMs = 450;
constexpr int kDefaultPeakCalibrationStride = 128;
constexpr int kDefaultPeakTieThreshold = 8;
constexpr int kMinimumPeakComparisons = 60;
constexpr int kDefaultBoardRows = 8;
constexpr int kDefaultBoardCols = 16;
constexpr int kMinimumActiveFeedbackPins = 2;
constexpr int kDefaultCalSum0CsPin = 18;
constexpr int kDefaultCalSum1CsPin = 19;
constexpr const char *kDefaultPeakTracePrefix = "peak_accuracy_capture";
constexpr std::array<int, 6> kDefaultBoardFeedbackPins{{5, 12, 13, 16, 20, 21}};
constexpr std::array<int, 6> kPage16CarrierFeedbackPins{{5, 12, 13, 16, 20, 21}};
constexpr std::array<int, 4> kDefaultBoardTxDataPins{{10, 11, 8, 7}};
constexpr std::array<int, 0> kDefaultBoardSelectPins{};
constexpr std::array<const char *, 6> kPage16CarrierNames{{"A", "B", "C", "D", "E", "F"}};
constexpr std::array<unsigned int, 7> kPeakWindowGapUsCandidates{{0U, 1U, 2U, 4U, 8U, 16U, 32U}};
constexpr std::array<unsigned int, 8> kPeakSettleSpinCandidates{{0U, 4U, 16U, 48U, 128U, 256U, 512U, 1024U}};
constexpr std::array<unsigned int, 13> kDefaultSpiSweepHz{{500000U,
                                                            1000000U,
                                                            2000000U,
                                                            4000000U,
                                                            8000000U,
                                                            12000000U,
                                                            16000000U,
                                                            24000000U,
                                                            32000000U,
                                                            48000000U,
                                                            64000000U,
                                                            96000000U,
                                                            120000000U}};

volatile std::uint64_t g_integer_sink = 0;
volatile double g_float_sink = 0.0;
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
volatile float g_neon_sink = 0.0f;
#endif
volatile std::uint64_t g_memory_sink = 0;

template <typename T>
T clampValue(T value, T low, T high) {
    return std::max(low, std::min(high, value));
}

bool fileExists(const std::string &path) {
#ifdef __linux__
    struct stat st {
    };
    return ::stat(path.c_str(), &st) == 0;
#else
    (void)path;
    return false;
#endif
}

std::string trimCopy(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n\0", 0);
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n\0");
    return value.substr(begin, end - begin + 1);
}

std::optional<std::string> readTextFile(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::optional<std::string> readFirstNonEmptyLine(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }
    std::string line;
    while (std::getline(in, line)) {
        line = trimCopy(line);
        if (!line.empty()) {
            return line;
        }
    }
    return std::nullopt;
}

std::vector<std::string> splitCommaList(const std::string &text) {
    std::vector<std::string> out;
    std::string token;
    for (char ch : text) {
        if (ch == ',' || ch == ';' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!token.empty()) {
                out.push_back(token);
                token.clear();
            }
            continue;
        }
        token.push_back(ch);
    }
    if (!token.empty()) {
        out.push_back(token);
    }
    return out;
}

std::vector<int> parsePinList(const std::string &text) {
    std::vector<int> pins;
    for (const std::string &token : splitCommaList(text)) {
        char *end = nullptr;
        const long value = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0') {
            continue;
        }
        pins.push_back(static_cast<int>(value));
    }
    return pins;
}

std::string joinInts(const std::vector<int> &values) {
    std::ostringstream out;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        out << values[index];
    }
    return out.str();
}

std::string jsonEscape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20U) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec;
            } else {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}

std::uint64_t nowNs() {
    using clock = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count());
}

std::uint32_t bitMaskForCount(std::size_t count) {
    if (count == 0U) {
        return 0U;
    }
    if (count >= 32U) {
        return 0xFFFFFFFFU;
    }
    return (1U << static_cast<unsigned int>(count)) - 1U;
}

std::uint32_t grayCode(std::uint32_t value) {
    return value ^ (value >> 1U);
}

double nsToSeconds(std::uint64_t ns) {
    return static_cast<double>(ns) / 1000000000.0;
}

std::string formatHz(double hz) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(hz >= 1000000.0 ? 2 : 0);
    if (hz >= 1000000.0) {
        out << (hz / 1000000.0) << " MHz";
    } else if (hz >= 1000.0) {
        out << (hz / 1000.0) << " kHz";
    } else {
        out << hz << " Hz";
    }
    return out.str();
}

std::string formatBytes(std::uint64_t bytes) {
    static constexpr const char *kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << ' ' << kUnits[unit];
    return out.str();
}

std::string formatThroughput(double bytesPerSecond) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (bytesPerSecond / (1024.0 * 1024.0)) << " MiB/s";
    return out.str();
}

std::string maskToBitString(std::uint8_t mask, std::size_t width) {
    std::string bits;
    bits.reserve(width);
    for (std::size_t index = 0; index < width; ++index) {
        bits.push_back(((mask >> static_cast<unsigned int>(index)) & 0x01U) != 0U ? '1' : '0');
    }
    return bits;
}

void spinDelay(unsigned int iterations) {
    for (unsigned int index = 0; index < iterations; ++index) {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#else
        std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }
}

void pinCurrentThreadToCpu(unsigned int cpuIndex) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpuIndex, &set);
    (void)::sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cpuIndex;
#endif
}

struct Options {
    int benchmarkMs{kDefaultBenchmarkMs};
    int gpioMs{kDefaultGpioMs};
    int spiMs{kDefaultSpiMs};
    int peakBenchmarkMs{kDefaultPeakBenchmarkMs};
    int peakCalibrationStride{kDefaultPeakCalibrationStride};
    int peakTieThreshold{kDefaultPeakTieThreshold};
    int memoryMiB{64};
    int gpioPin{6};
    int calSum0CsPin{kDefaultCalSum0CsPin};
    int calSum1CsPin{kDefaultCalSum1CsPin};
    int boardRows{kDefaultBoardRows};
    int boardCols{kDefaultBoardCols};
    bool forcePage16CalibrationSpi{false};
    bool skipCpu{false};
    bool skipMemory{false};
    bool skipGpio{false};
    bool skipSpi{false};
    bool spiLoopback{false};
    std::string spiDevice{"/dev/spidev0.0"};
    unsigned int spiMaxHz{120000000U};
    std::string jsonOut;
    std::string peakTracePrefix{kDefaultPeakTracePrefix};
    std::vector<int> feedbackPins;
};

bool writeReportFile(const std::string &path, const std::string &content, std::string &error);

void printUsage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "This program auto-runs Raspberry Pi Zero 2W host and board-side capability probes.\n\n"
        << "Options:\n"
        << "  --benchmark-ms N        Per CPU benchmark duration in ms (default 800)\n"
        << "  --gpio-ms N             GPIO stress duration in ms (default 180)\n"
        << "  --spi-ms N              Per SPI sweep point duration in ms (default 120)\n"
        << "  --peak-benchmark-ms N   Peak page16 capture duration in ms (default 450)\n"
        << "  --peak-calibration-stride N  Sample SUM0/SUM1 every N windows (default 128)\n"
        << "  --peak-tie-threshold N  Exclude |SUM0-SUM1| <= N ADC counts from scoring (default 8)\n"
        << "  --memory-mib N          Memory benchmark working set in MiB (default 64)\n"
        << "  --gpio-pin BCM          GPIO output pin to stress (default 6 / SAMPLE_SYNC)\n"
        << "  --cal-sum0-cs-pin BCM   BCM pin for page16 SUM0 calibration ADC CS (default 18)\n"
        << "  --cal-sum1-cs-pin BCM   BCM pin for page16 SUM1 calibration ADC CS (default 19)\n"
        << "  --force-page16-calibration-spi  Use /dev/spidev0.0 + manual CS for page16 slow ADCs even though the current repo eext_netlist keeps CAL_SPI/CAL_ADC_SUM* isolated from Pi J1\n"
        << "  --board-rows N          Full-board output rows for equivalent NPU compute (default 8)\n"
        << "  --board-cols N          Full-board output cols for equivalent NPU compute (default 16)\n"
        << "  --feedback-pins LIST    Optional BCM input pins, comma-separated\n"
        << "  --spi-device PATH       SPI device path (default /dev/spidev0.0)\n"
        << "  --spi-max-hz N          Max SPI sweep target in Hz (default 120000000)\n"
        << "  --spi-loopback          Require RX == TX during SPI sweep\n"
        << "  --skip-cpu              Skip scalar/NEON benchmark\n"
        << "  --skip-memory           Skip memory bandwidth benchmark\n"
        << "  --skip-gpio             Skip GPIO edge-rate probe\n"
        << "  --skip-spi              Skip SPI sweep\n"
        << "  --json-out PATH         Write machine-readable JSON report\n"
        << "  --peak-trace-prefix P   Write selected peak traces to P.out.csv and P.samples.csv (use - to disable)\n"
        << "  --help                  Show this help\n";
}

bool parseArgs(int argc, char **argv, Options &options, std::string &error) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto requireValue = [&](const char *label) -> std::optional<std::string> {
            if (index + 1 >= argc) {
                error = std::string("missing value for ") + label;
                return std::nullopt;
            }
            ++index;
            return std::string(argv[index]);
        };
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--benchmark-ms") {
            const auto value = requireValue("--benchmark-ms");
            if (!value) {
                return false;
            }
            options.benchmarkMs = clampValue(std::atoi(value->c_str()), 50, 10000);
            continue;
        }
        if (arg == "--gpio-ms") {
            const auto value = requireValue("--gpio-ms");
            if (!value) {
                return false;
            }
            options.gpioMs = clampValue(std::atoi(value->c_str()), 20, 5000);
            continue;
        }
        if (arg == "--spi-ms") {
            const auto value = requireValue("--spi-ms");
            if (!value) {
                return false;
            }
            options.spiMs = clampValue(std::atoi(value->c_str()), 20, 5000);
            continue;
        }
        if (arg == "--peak-benchmark-ms") {
            const auto value = requireValue("--peak-benchmark-ms");
            if (!value) {
                return false;
            }
            options.peakBenchmarkMs = clampValue(std::atoi(value->c_str()), 100, 5000);
            continue;
        }
        if (arg == "--peak-calibration-stride") {
            const auto value = requireValue("--peak-calibration-stride");
            if (!value) {
                return false;
            }
            options.peakCalibrationStride = clampValue(std::atoi(value->c_str()), 1, 8192);
            continue;
        }
        if (arg == "--peak-tie-threshold") {
            const auto value = requireValue("--peak-tie-threshold");
            if (!value) {
                return false;
            }
            options.peakTieThreshold = clampValue(std::atoi(value->c_str()), 0, 256);
            continue;
        }
        if (arg == "--memory-mib") {
            const auto value = requireValue("--memory-mib");
            if (!value) {
                return false;
            }
            options.memoryMiB = clampValue(std::atoi(value->c_str()), 8, 256);
            continue;
        }
        if (arg == "--gpio-pin") {
            const auto value = requireValue("--gpio-pin");
            if (!value) {
                return false;
            }
            options.gpioPin = clampValue(std::atoi(value->c_str()), 0, 27);
            continue;
        }
        if (arg == "--cal-sum0-cs-pin") {
            const auto value = requireValue("--cal-sum0-cs-pin");
            if (!value) {
                return false;
            }
            options.calSum0CsPin = clampValue(std::atoi(value->c_str()), 0, 27);
            continue;
        }
        if (arg == "--cal-sum1-cs-pin") {
            const auto value = requireValue("--cal-sum1-cs-pin");
            if (!value) {
                return false;
            }
            options.calSum1CsPin = clampValue(std::atoi(value->c_str()), 0, 27);
            continue;
        }
        if (arg == "--force-page16-calibration-spi") {
            options.forcePage16CalibrationSpi = true;
            continue;
        }
        if (arg == "--board-rows") {
            const auto value = requireValue("--board-rows");
            if (!value) {
                return false;
            }
            options.boardRows = clampValue(std::atoi(value->c_str()), 1, 512);
            continue;
        }
        if (arg == "--board-cols") {
            const auto value = requireValue("--board-cols");
            if (!value) {
                return false;
            }
            options.boardCols = clampValue(std::atoi(value->c_str()), 1, 512);
            continue;
        }
        if (arg == "--feedback-pins") {
            const auto value = requireValue("--feedback-pins");
            if (!value) {
                return false;
            }
            options.feedbackPins = parsePinList(*value);
            continue;
        }
        if (arg == "--spi-device") {
            const auto value = requireValue("--spi-device");
            if (!value) {
                return false;
            }
            options.spiDevice = *value;
            continue;
        }
        if (arg == "--spi-max-hz") {
            const auto value = requireValue("--spi-max-hz");
            if (!value) {
                return false;
            }
            options.spiMaxHz = static_cast<unsigned int>(clampValue(std::atoll(value->c_str()), 100000LL, 200000000LL));
            continue;
        }
        if (arg == "--json-out") {
            const auto value = requireValue("--json-out");
            if (!value) {
                return false;
            }
            options.jsonOut = *value;
            continue;
        }
        if (arg == "--peak-trace-prefix") {
            const auto value = requireValue("--peak-trace-prefix");
            if (!value) {
                return false;
            }
            options.peakTracePrefix = *value == "-" ? std::string() : *value;
            continue;
        }
        if (arg == "--skip-cpu") {
            options.skipCpu = true;
            continue;
        }
        if (arg == "--skip-memory") {
            options.skipMemory = true;
            continue;
        }
        if (arg == "--skip-gpio") {
            options.skipGpio = true;
            continue;
        }
        if (arg == "--skip-spi") {
            options.skipSpi = true;
            continue;
        }
        if (arg == "--spi-loopback") {
            options.spiLoopback = true;
            continue;
        }
        error = "unknown argument: " + arg;
        return false;
    }
    if (options.feedbackPins.empty()) {
        options.feedbackPins.assign(kDefaultBoardFeedbackPins.begin(), kDefaultBoardFeedbackPins.end());
    }
    options.feedbackPins.erase(std::remove(options.feedbackPins.begin(), options.feedbackPins.end(), options.gpioPin), options.feedbackPins.end());
    std::sort(options.feedbackPins.begin(), options.feedbackPins.end());
    options.feedbackPins.erase(std::unique(options.feedbackPins.begin(), options.feedbackPins.end()), options.feedbackPins.end());
    return true;
}

struct SystemSnapshot {
    std::string model;
    std::string kernel;
    std::string machine;
    std::string osRelease;
    long pageSize{0};
    int configuredCores{0};
    int onlineCores{0};
    std::uint64_t totalMemoryBytes{0};
    std::uint64_t freeMemoryBytes{0};
    std::uint64_t availableMemoryBytes{0};
    int thermalMilliC{-1};
    std::vector<int> currentCpuFreqKHz;
    std::map<std::string, bool> hwcaps;
};

SystemSnapshot collectSystemSnapshot() {
    SystemSnapshot snapshot;
#ifdef __linux__
    if (const auto model = readTextFile("/proc/device-tree/model")) {
        snapshot.model = trimCopy(*model);
    }
    if (snapshot.model.empty()) {
        if (const auto model = readTextFile("/sys/firmware/devicetree/base/model")) {
            snapshot.model = trimCopy(*model);
        }
    }
    struct utsname uts {
    };
    if (::uname(&uts) == 0) {
        snapshot.kernel = uts.release;
        snapshot.machine = uts.machine;
        snapshot.osRelease = uts.sysname;
    }
    snapshot.pageSize = ::sysconf(_SC_PAGESIZE);
    snapshot.configuredCores = static_cast<int>(::sysconf(_SC_NPROCESSORS_CONF));
    snapshot.onlineCores = static_cast<int>(::sysconf(_SC_NPROCESSORS_ONLN));

    struct sysinfo info {
    };
    if (::sysinfo(&info) == 0) {
        snapshot.totalMemoryBytes = static_cast<std::uint64_t>(info.totalram) * static_cast<std::uint64_t>(info.mem_unit);
        snapshot.freeMemoryBytes = static_cast<std::uint64_t>(info.freeram) * static_cast<std::uint64_t>(info.mem_unit);
    }
    if (const auto meminfo = readTextFile("/proc/meminfo")) {
        std::istringstream in(*meminfo);
        std::string line;
        while (std::getline(in, line)) {
            const auto colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            const std::string key = trimCopy(line.substr(0, colon));
            const std::string rest = trimCopy(line.substr(colon + 1));
            if (key == "MemAvailable") {
                std::istringstream parser(rest);
                std::uint64_t kib = 0;
                parser >> kib;
                snapshot.availableMemoryBytes = kib * 1024ULL;
            }
        }
    }
    if (const auto thermal = readFirstNonEmptyLine("/sys/class/thermal/thermal_zone0/temp")) {
        snapshot.thermalMilliC = std::atoi(thermal->c_str());
    }
    for (int cpu = 0; cpu < snapshot.onlineCores; ++cpu) {
        std::ostringstream path;
        path << "/sys/devices/system/cpu/cpu" << cpu << "/cpufreq/scaling_cur_freq";
        if (const auto freq = readFirstNonEmptyLine(path.str())) {
            snapshot.currentCpuFreqKHz.push_back(std::atoi(freq->c_str()));
        }
    }

#if defined(__aarch64__) || defined(__arm__)
    const unsigned long hwcap = ::getauxval(AT_HWCAP);
    const unsigned long hwcap2 = ::getauxval(AT_HWCAP2);
#ifdef HWCAP_NEON
    snapshot.hwcaps["neon"] = (hwcap & HWCAP_NEON) != 0UL;
#endif
#ifdef HWCAP_ASIMD
    snapshot.hwcaps["asimd"] = (hwcap & HWCAP_ASIMD) != 0UL;
#endif
#ifdef HWCAP_AES
    snapshot.hwcaps["aes"] = (hwcap & HWCAP_AES) != 0UL;
#endif
#ifdef HWCAP_CRC32
    snapshot.hwcaps["crc32"] = (hwcap & HWCAP_CRC32) != 0UL;
#endif
#ifdef HWCAP_SHA1
    snapshot.hwcaps["sha1"] = (hwcap & HWCAP_SHA1) != 0UL;
#endif
#ifdef HWCAP_SHA2
    snapshot.hwcaps["sha2"] = (hwcap & HWCAP_SHA2) != 0UL;
#endif
#ifdef HWCAP_ATOMICS
    snapshot.hwcaps["atomics"] = (hwcap & HWCAP_ATOMICS) != 0UL;
#endif
#ifdef HWCAP_PMULL
    snapshot.hwcaps["pmull"] = (hwcap & HWCAP_PMULL) != 0UL;
#endif
    (void)hwcap2;
#endif
#endif
    return snapshot;
}

enum class BenchmarkKind {
    Integer,
    Float,
    Neon,
};

struct BenchmarkRun {
    std::string name;
    bool attempted{false};
    bool available{false};
    int threads{0};
    double seconds{0.0};
    double gigaOpsPerSec{0.0};
    std::uint64_t operationCount{0};
    std::string note;
};

struct ScalingPoint {
    int threads{0};
    double gigaOpsPerSec{0.0};
    double speedup{0.0};
    double efficiency{0.0};
};

struct ComputeSummary {
    BenchmarkRun integerSingle;
    BenchmarkRun floatSingle;
    BenchmarkRun neonSingle;
    std::vector<ScalingPoint> scaling;
    int recommendedThreads{1};
    std::string preferredKernel{"scalar-int"};
};

std::uint64_t runIntegerKernel(std::uint64_t durationNs) {
    std::uint64_t stateA = 0x123456789abcdef0ULL;
    std::uint64_t stateB = 0xfedcba9876543210ULL;
    std::uint64_t operations = 0;
    const std::uint64_t stopAt = nowNs() + durationNs;
    while (nowNs() < stopAt) {
        for (int iteration = 0; iteration < 512; ++iteration) {
            stateA = stateA * 2862933555777941757ULL + 3037000493ULL;
            stateB ^= (stateA >> 13);
            stateB += (stateA << 7) ^ 0x9e3779b97f4a7c15ULL;
            stateA ^= (stateB >> 17);
            operations += 8;
        }
    }
    g_integer_sink ^= (stateA ^ stateB);
    return operations;
}

std::uint64_t runFloatKernel(std::uint64_t durationNs) {
    double a = 1.1;
    double b = 1.3;
    double c = 1.7;
    double d = 2.1;
    std::uint64_t operations = 0;
    const std::uint64_t stopAt = nowNs() + durationNs;
    while (nowNs() < stopAt) {
        for (int iteration = 0; iteration < 512; ++iteration) {
            a = a * b + c;
            b = b * c + d;
            c = c * d + a;
            d = d * a + b;
            operations += 8;
        }
    }
    g_float_sink += (a + b + c + d);
    return operations;
}

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
std::uint64_t runNeonKernel(std::uint64_t durationNs) {
    float32x4_t a = vdupq_n_f32(1.0f);
    float32x4_t b = vdupq_n_f32(1.125f);
    float32x4_t c = vdupq_n_f32(1.25f);
    float32x4_t d = vdupq_n_f32(1.5f);
    std::uint64_t operations = 0;
    const std::uint64_t stopAt = nowNs() + durationNs;
    while (nowNs() < stopAt) {
        for (int iteration = 0; iteration < 512; ++iteration) {
            a = vmlaq_f32(a, b, c);
            b = vmlaq_f32(b, c, d);
            c = vmlaq_f32(c, d, a);
            d = vmlaq_f32(d, a, b);
            operations += 4ULL * 4ULL * 2ULL * 4ULL;
        }
    }
    alignas(16) float lane[4];
    vst1q_f32(lane, vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    g_neon_sink += lane[0] + lane[1] + lane[2] + lane[3];
    return operations;
}
#endif

BenchmarkRun runBenchmark(BenchmarkKind kind, int threads, int durationMs, int onlineCores, bool neonAvailable) {
    BenchmarkRun run;
    run.attempted = true;
    run.threads = threads;
    switch (kind) {
    case BenchmarkKind::Integer:
        run.name = "scalar-int";
        break;
    case BenchmarkKind::Float:
        run.name = "scalar-fp";
        break;
    case BenchmarkKind::Neon:
        run.name = "neon-fp32";
        break;
    }
    if (kind == BenchmarkKind::Neon && !neonAvailable) {
        run.note = "NEON/ASIMD not reported by this runtime";
        return run;
    }

    std::vector<std::thread> workers;
    std::vector<std::uint64_t> perThreadOps(static_cast<std::size_t>(threads), 0ULL);
    std::atomic<bool> go{false};
    const std::uint64_t durationNs = static_cast<std::uint64_t>(durationMs) * 1000000ULL;

    for (int threadIndex = 0; threadIndex < threads; ++threadIndex) {
        workers.emplace_back([&, threadIndex]() {
            if (onlineCores > 0) {
                pinCurrentThreadToCpu(static_cast<unsigned int>(threadIndex % onlineCores));
            }
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            switch (kind) {
            case BenchmarkKind::Integer:
                perThreadOps[static_cast<std::size_t>(threadIndex)] = runIntegerKernel(durationNs);
                break;
            case BenchmarkKind::Float:
                perThreadOps[static_cast<std::size_t>(threadIndex)] = runFloatKernel(durationNs);
                break;
            case BenchmarkKind::Neon:
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
                perThreadOps[static_cast<std::size_t>(threadIndex)] = runNeonKernel(durationNs);
#else
                perThreadOps[static_cast<std::size_t>(threadIndex)] = 0ULL;
#endif
                break;
            }
        });
    }

    const std::uint64_t start = nowNs();
    go.store(true, std::memory_order_release);
    for (std::thread &worker : workers) {
        worker.join();
    }
    const std::uint64_t elapsed = std::max<std::uint64_t>(1ULL, nowNs() - start);
    run.operationCount = std::accumulate(perThreadOps.begin(), perThreadOps.end(), 0ULL);
    run.seconds = nsToSeconds(elapsed);
    run.gigaOpsPerSec = (static_cast<double>(run.operationCount) / run.seconds) / 1000000000.0;
    run.available = run.operationCount > 0;
    return run;
}

ComputeSummary runComputeSummary(const SystemSnapshot &snapshot, const Options &options) {
    ComputeSummary summary;
    const bool neonAvailable = snapshot.hwcaps.count("asimd") != 0 ? snapshot.hwcaps.at("asimd") : (snapshot.hwcaps.count("neon") != 0 && snapshot.hwcaps.at("neon"));
    const int onlineCores = std::max(1, snapshot.onlineCores);

    summary.integerSingle = runBenchmark(BenchmarkKind::Integer, 1, options.benchmarkMs, onlineCores, neonAvailable);
    summary.floatSingle = runBenchmark(BenchmarkKind::Float, 1, options.benchmarkMs, onlineCores, neonAvailable);
    summary.neonSingle = runBenchmark(BenchmarkKind::Neon, 1, options.benchmarkMs, onlineCores, neonAvailable);

    const BenchmarkRun &scalingBase = summary.neonSingle.available ? summary.neonSingle : (summary.floatSingle.available ? summary.floatSingle : summary.integerSingle);
    summary.preferredKernel = scalingBase.name;
    const BenchmarkKind scalingKind = scalingBase.name == "neon-fp32" ? BenchmarkKind::Neon : (scalingBase.name == "scalar-fp" ? BenchmarkKind::Float : BenchmarkKind::Integer);

    double bestThroughput = 0.0;
    int bestThreads = 1;
    for (int threads = 1; threads <= onlineCores; ++threads) {
        const BenchmarkRun run = runBenchmark(scalingKind, threads, options.benchmarkMs, onlineCores, neonAvailable);
        if (!run.available) {
            continue;
        }
        const double base = std::max(1e-9, scalingBase.gigaOpsPerSec);
        const double speedup = run.gigaOpsPerSec / base;
        const double efficiency = speedup / static_cast<double>(threads);
        summary.scaling.push_back(ScalingPoint{threads, run.gigaOpsPerSec, speedup, efficiency});
        if (run.gigaOpsPerSec > bestThroughput) {
            bestThroughput = run.gigaOpsPerSec;
            bestThreads = threads;
        }
    }
    summary.recommendedThreads = bestThreads;
    return summary;
}

struct MemorySummary {
    bool attempted{false};
    bool available{false};
    std::size_t workingSetBytes{0};
    double readBytesPerSec{0.0};
    double copyBytesPerSec{0.0};
    std::string note;
};

MemorySummary runMemorySummary(const SystemSnapshot &snapshot, const Options &options) {
    MemorySummary summary;
    summary.attempted = true;
    const std::uint64_t capByAvailability = snapshot.availableMemoryBytes > 0 ? snapshot.availableMemoryBytes / 4ULL : 64ULL * 1024ULL * 1024ULL;
    const std::size_t workingSet = static_cast<std::size_t>(clampValue<std::uint64_t>(static_cast<std::uint64_t>(options.memoryMiB) * 1024ULL * 1024ULL, 8ULL * 1024ULL * 1024ULL, std::max<std::uint64_t>(8ULL * 1024ULL * 1024ULL, capByAvailability)));
    summary.workingSetBytes = workingSet;
    std::vector<std::uint8_t> src(workingSet, 0x5a);
    std::vector<std::uint8_t> dst(workingSet, 0x00);
    if (src.empty() || dst.empty()) {
        summary.note = "failed to allocate memory benchmark buffers";
        return summary;
    }

    const std::uint64_t readStart = nowNs();
    std::uint64_t bytesRead = 0;
    while (nowNs() - readStart < static_cast<std::uint64_t>(options.benchmarkMs / 2) * 1000000ULL) {
        std::uint64_t checksum = 0;
        for (std::size_t index = 0; index < src.size(); index += 64) {
            checksum += src[index];
        }
        g_memory_sink ^= checksum;
        bytesRead += src.size();
    }
    const std::uint64_t readElapsed = std::max<std::uint64_t>(1ULL, nowNs() - readStart);
    summary.readBytesPerSec = static_cast<double>(bytesRead) / nsToSeconds(readElapsed);

    const std::uint64_t copyStart = nowNs();
    std::uint64_t bytesCopied = 0;
    while (nowNs() - copyStart < static_cast<std::uint64_t>(options.benchmarkMs / 2) * 1000000ULL) {
        std::memcpy(dst.data(), src.data(), src.size());
        bytesCopied += src.size();
    }
    const std::uint64_t copyElapsed = std::max<std::uint64_t>(1ULL, nowNs() - copyStart);
    summary.copyBytesPerSec = static_cast<double>(bytesCopied) / nsToSeconds(copyElapsed);
    summary.available = true;
    return summary;
}

#ifdef __linux__
class GpioMmio {
public:
    ~GpioMmio() {
        close();
    }

    bool open(const std::string &model, std::string &deviceUsed, std::string &error) {
        if (regs_ != nullptr) {
            deviceUsed = devicePath_;
            return true;
        }

        if (mapDevice("/dev/gpiomem", 0, deviceUsed, error)) {
            devicePath_ = "/dev/gpiomem";
            return true;
        }

        off_t gpioBase = 0x3F200000;
        if (model.find("Raspberry Pi 4") != std::string::npos || model.find("Raspberry Pi 400") != std::string::npos || model.find("Raspberry Pi 5") != std::string::npos) {
            gpioBase = 0xFE200000;
        } else if (model.find("Raspberry Pi Zero") == std::string::npos && model.find("Raspberry Pi 2") == std::string::npos && model.find("Raspberry Pi 3") == std::string::npos) {
            gpioBase = 0x20200000;
        }
        std::string fallbackError;
        if (mapDevice("/dev/mem", gpioBase, deviceUsed, fallbackError)) {
            devicePath_ = "/dev/mem";
            return true;
        }
        error += error.empty() ? fallbackError : std::string("; ") + fallbackError;
        return false;
    }

    void close() {
        if (regs_ != nullptr) {
            ::munmap(const_cast<std::uint32_t *>(regs_), kMapLength);
            regs_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        devicePath_.clear();
    }

    bool configureInput(int pin, std::string &error) {
        return setFunction(pin, 0U, error);
    }

    bool configureOutput(int pin, std::string &error) {
        return setFunction(pin, 1U, error);
    }

    bool configureAlt0(int pin, std::string &error) {
        return setFunction(pin, 4U, error);
    }

    void write(int pin, bool high) {
        if (regs_ == nullptr) {
            return;
        }
        const std::uint32_t mask = 1U << (pin % 32);
        if (high) {
            regs_[7 + (pin / 32)] = mask;
        } else {
            regs_[10 + (pin / 32)] = mask;
        }
    }

    bool read(int pin) const {
        if (regs_ == nullptr) {
            return false;
        }
        const std::uint32_t value = regs_[13 + (pin / 32)];
        return ((value >> (pin % 32)) & 0x1U) != 0U;
    }

private:
    static constexpr std::size_t kMapLength = 4096;

    bool mapDevice(const char *path, off_t offset, std::string &deviceUsed, std::string &error) {
        const int openFlags = O_RDWR | O_SYNC | O_CLOEXEC;
        fd_ = ::open(path, openFlags);
        if (fd_ < 0) {
            error = std::string("failed to open ") + path + ": " + std::strerror(errno);
            return false;
        }
        void *mapped = ::mmap(nullptr, kMapLength, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, offset);
        if (mapped == MAP_FAILED) {
            error = std::string("failed to mmap ") + path + ": " + std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        regs_ = static_cast<volatile std::uint32_t *>(mapped);
        deviceUsed = offset == 0 ? path : (std::string(path) + "+0x" + toHex(static_cast<std::uint32_t>(offset)));
        return true;
    }

    bool setFunction(int pin, std::uint32_t function, std::string &error) {
        if (pin < 0 || pin > 53) {
            error = "invalid GPIO pin";
            return false;
        }
        if (regs_ == nullptr) {
            error = "GPIO MMIO is unavailable";
            return false;
        }
        volatile std::uint32_t &reg = regs_[pin / 10];
        const unsigned shift = static_cast<unsigned>((pin % 10) * 3);
        std::uint32_t value = reg;
        value &= ~(0x7U << shift);
        value |= ((function & 0x7U) << shift);
        reg = value;
        return true;
    }

    static std::string toHex(std::uint32_t value) {
        std::ostringstream out;
        out << std::hex << std::uppercase << value << std::dec;
        return out.str();
    }

    int fd_{-1};
    volatile std::uint32_t *regs_{nullptr};
    std::string devicePath_;
};
#endif

struct FeedbackObservation {
    int pin{-1};
    std::uint64_t highSamples{0};
    std::uint64_t lowSamples{0};
    std::uint64_t observedTransitions{0};
    bool lastLevel{false};
    bool initialized{false};
};

struct GpioSummary {
    bool attempted{false};
    bool available{false};
    int pin{-1};
    std::vector<int> feedbackPins;
    std::string device;
    double openLoopToggleHz{0.0};
    double validatedToggleHz{0.0};
    double recommendedToggleHz{0.0};
    double mismatchRatio{0.0};
    std::uint64_t sampleCount{0};
    std::uint64_t mismatchCount{0};
    std::vector<FeedbackObservation> feedback;
    double protocolWindowRateHz{0.0};
    double protocolLineOpsPerSec{0.0};
    double protocolOpsPerWindow{0.0};
    std::uint64_t protocolWindows{0};
    std::uint64_t protocolValidationSamples{0};
    std::uint64_t protocolValidationMismatches{0};
    double protocolValidationMismatchRatio{0.0};
    std::vector<FeedbackObservation> protocolFeedback;
    std::string note;
    std::string error;
};

GpioSummary runGpioSummary(const SystemSnapshot &snapshot, const Options &options) {
    GpioSummary summary;
    summary.attempted = true;
    summary.pin = options.gpioPin;
    summary.feedbackPins = options.feedbackPins;
#ifdef __linux__
    GpioMmio mmio;
    std::string deviceUsed;
    std::string error;
    if (!mmio.open(snapshot.model, deviceUsed, error)) {
        summary.error = error;
        return summary;
    }
    summary.device = deviceUsed;
    if (!mmio.configureOutput(options.gpioPin, error)) {
        summary.error = error;
        return summary;
    }
    auto appendNote = [&](const std::string &fragment) {
        if (fragment.empty()) {
            return;
        }
        if (!summary.note.empty()) {
            summary.note += "; ";
        }
        summary.note += fragment;
    };
    auto makeFeedbackObservations = [&](const std::vector<int> &pins) {
        std::vector<FeedbackObservation> observations;
        observations.reserve(pins.size());
        for (int pin : pins) {
            observations.push_back(FeedbackObservation{pin});
        }
        return observations;
    };
    std::vector<FeedbackObservation> feedback;
    for (int pin : options.feedbackPins) {
        if (pin < 0) {
            continue;
        }
        std::string configureError;
        if (!mmio.configureInput(pin, configureError)) {
            appendNote("feedback GPIO" + std::to_string(pin) + " setup failed: " + configureError);
            continue;
        }
        feedback.push_back(FeedbackObservation{pin});
    }
    summary.feedback = feedback;

    mmio.write(options.gpioPin, false);
    const std::uint64_t openLoopStart = nowNs();
    std::uint64_t openLoopTransitions = 0;
    while (nowNs() - openLoopStart < static_cast<std::uint64_t>(options.gpioMs) * 1000000ULL) {
        for (int batch = 0; batch < 4096; ++batch) {
            mmio.write(options.gpioPin, true);
            mmio.write(options.gpioPin, false);
        }
        openLoopTransitions += 8192ULL;
    }
    const std::uint64_t openLoopElapsed = std::max<std::uint64_t>(1ULL, nowNs() - openLoopStart);
    summary.openLoopToggleHz = (static_cast<double>(openLoopTransitions) * 0.5) / nsToSeconds(openLoopElapsed);

    const std::uint64_t validatedStart = nowNs();
    std::uint64_t validatedTransitions = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t sampleCount = 0;
    while (nowNs() - validatedStart < static_cast<std::uint64_t>(options.gpioMs) * 1000000ULL) {
        for (int batch = 0; batch < 1024; ++batch) {
            mmio.write(options.gpioPin, true);
            if (!mmio.read(options.gpioPin)) {
                ++mismatches;
            }
            ++sampleCount;
            for (FeedbackObservation &observation : summary.feedback) {
                const bool level = mmio.read(observation.pin);
                if (level) {
                    ++observation.highSamples;
                } else {
                    ++observation.lowSamples;
                }
                if (observation.initialized && observation.lastLevel != level) {
                    ++observation.observedTransitions;
                }
                observation.lastLevel = level;
                observation.initialized = true;
            }

            mmio.write(options.gpioPin, false);
            if (mmio.read(options.gpioPin)) {
                ++mismatches;
            }
            ++sampleCount;
            for (FeedbackObservation &observation : summary.feedback) {
                const bool level = mmio.read(observation.pin);
                if (level) {
                    ++observation.highSamples;
                } else {
                    ++observation.lowSamples;
                }
                if (observation.initialized && observation.lastLevel != level) {
                    ++observation.observedTransitions;
                }
                observation.lastLevel = level;
                observation.initialized = true;
            }
        }
        validatedTransitions += 2048ULL;
    }
    const std::uint64_t validatedElapsed = std::max<std::uint64_t>(1ULL, nowNs() - validatedStart);
    summary.validatedToggleHz = (static_cast<double>(validatedTransitions) * 0.5) / nsToSeconds(validatedElapsed);
    summary.sampleCount = sampleCount;
    summary.mismatchCount = mismatches;
    summary.recommendedToggleHz = summary.validatedToggleHz * kRecommendedSafetyFactor;
    summary.available = true;
    if (summary.sampleCount > 0 && summary.mismatchCount > 0) {
        summary.mismatchRatio = static_cast<double>(summary.mismatchCount) / static_cast<double>(summary.sampleCount);
        appendNote("self-readback mismatch ratio=" + std::to_string(summary.mismatchRatio));
        summary.recommendedToggleHz = summary.validatedToggleHz * std::max(0.35, 1.0 - summary.mismatchRatio * 20.0);
    }
    mmio.write(options.gpioPin, false);

    std::vector<int> protocolOutputPins;
    protocolOutputPins.reserve(kDefaultBoardTxDataPins.size() + kDefaultBoardSelectPins.size() + 1U);
    protocolOutputPins.insert(protocolOutputPins.end(), kDefaultBoardTxDataPins.begin(), kDefaultBoardTxDataPins.end());
    protocolOutputPins.insert(protocolOutputPins.end(), kDefaultBoardSelectPins.begin(), kDefaultBoardSelectPins.end());
    protocolOutputPins.push_back(options.gpioPin);
    std::sort(protocolOutputPins.begin(), protocolOutputPins.end());
    protocolOutputPins.erase(std::unique(protocolOutputPins.begin(), protocolOutputPins.end()), protocolOutputPins.end());

    std::vector<int> overlappingPins;
    for (int pin : protocolOutputPins) {
        if (std::find(options.feedbackPins.begin(), options.feedbackPins.end(), pin) != options.feedbackPins.end()) {
            overlappingPins.push_back(pin);
        }
    }

    bool protocolReady = overlappingPins.empty();
    if (!protocolReady) {
        appendNote("protocol benchmark skipped because feedback pins overlap the GPIO output domain: " + joinInts(overlappingPins));
    }
    for (int pin : protocolOutputPins) {
        if (!protocolReady || pin == options.gpioPin) {
            continue;
        }
        std::string configureError;
        if (!mmio.configureOutput(pin, configureError)) {
            appendNote("protocol GPIO" + std::to_string(pin) + " setup failed: " + configureError);
            protocolReady = false;
        }
    }

    if (protocolReady) {
        summary.protocolFeedback = makeFeedbackObservations(options.feedbackPins);
        const std::uint32_t txDataMask = bitMaskForCount(kDefaultBoardTxDataPins.size());
        const std::uint32_t selectMask = bitMaskForCount(kDefaultBoardSelectPins.size());
        std::uint64_t protocolOps = 0;
        std::uint32_t sequence = 1U;
        const std::uint64_t protocolStart = nowNs();
        while (nowNs() - protocolStart < static_cast<std::uint64_t>(options.gpioMs) * 1000000ULL) {
            for (int batch = 0; batch < 256; ++batch) {
                const std::uint32_t txDataValue = grayCode(sequence) & txDataMask;
                const std::uint32_t txSelectValue = grayCode(sequence * 3U + 1U) & selectMask;
                const std::uint32_t rxSelectValue = grayCode(sequence * 5U + 3U) & selectMask;

                for (std::size_t index = 0; index < kDefaultBoardTxDataPins.size(); ++index) {
                    mmio.write(kDefaultBoardTxDataPins[index], ((txDataValue >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
                    ++protocolOps;
                }
                for (std::size_t index = 0; index < kDefaultBoardSelectPins.size(); ++index) {
                    mmio.write(kDefaultBoardSelectPins[index], ((txSelectValue >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
                    ++protocolOps;
                }
                mmio.write(options.gpioPin, true);
                ++protocolOps;
                mmio.write(options.gpioPin, false);
                ++protocolOps;
                for (std::size_t index = 0; index < kDefaultBoardSelectPins.size(); ++index) {
                    mmio.write(kDefaultBoardSelectPins[index], ((rxSelectValue >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
                    ++protocolOps;
                }
                for (FeedbackObservation &observation : summary.protocolFeedback) {
                    const bool level = mmio.read(observation.pin);
                    if (level) {
                        ++observation.highSamples;
                    } else {
                        ++observation.lowSamples;
                    }
                    if (observation.initialized && observation.lastLevel != level) {
                        ++observation.observedTransitions;
                    }
                    observation.lastLevel = level;
                    observation.initialized = true;
                    ++protocolOps;
                }
                ++summary.protocolWindows;
                ++sequence;
            }
        }
        const std::uint64_t protocolElapsed = std::max<std::uint64_t>(1ULL, nowNs() - protocolStart);
        if (summary.protocolWindows > 0) {
            summary.protocolWindowRateHz = static_cast<double>(summary.protocolWindows) / nsToSeconds(protocolElapsed);
            summary.protocolLineOpsPerSec = static_cast<double>(protocolOps) / nsToSeconds(protocolElapsed);
            summary.protocolOpsPerWindow = static_cast<double>(protocolOps) / static_cast<double>(summary.protocolWindows);
        }

        const int validationMs = clampValue(options.gpioMs / 3, 40, 250);
        sequence = 1U;
        const std::uint64_t validationStart = nowNs();
        while (nowNs() - validationStart < static_cast<std::uint64_t>(validationMs) * 1000000ULL) {
            for (int batch = 0; batch < 128; ++batch) {
                const std::uint32_t txDataValue = grayCode(sequence) & txDataMask;
                const std::uint32_t txSelectValue = grayCode(sequence * 7U + 1U) & selectMask;
                const std::uint32_t rxSelectValue = grayCode(sequence * 11U + 3U) & selectMask;

                for (std::size_t index = 0; index < kDefaultBoardTxDataPins.size(); ++index) {
                    mmio.write(kDefaultBoardTxDataPins[index], ((txDataValue >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
                }
                for (std::size_t index = 0; index < kDefaultBoardSelectPins.size(); ++index) {
                    mmio.write(kDefaultBoardSelectPins[index], ((txSelectValue >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
                }
                mmio.write(options.gpioPin, true);
                if (!mmio.read(options.gpioPin)) {
                    ++summary.protocolValidationMismatches;
                }
                ++summary.protocolValidationSamples;
                mmio.write(options.gpioPin, false);
                if (mmio.read(options.gpioPin)) {
                    ++summary.protocolValidationMismatches;
                }
                ++summary.protocolValidationSamples;
                for (std::size_t index = 0; index < kDefaultBoardSelectPins.size(); ++index) {
                    mmio.write(kDefaultBoardSelectPins[index], ((rxSelectValue >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
                }
                for (int pin : options.feedbackPins) {
                    (void)mmio.read(pin);
                }
                ++sequence;
            }
        }
        if (summary.protocolValidationSamples > 0) {
            summary.protocolValidationMismatchRatio = static_cast<double>(summary.protocolValidationMismatches) /
                                                     static_cast<double>(summary.protocolValidationSamples);
            appendNote("protocol validation mismatch ratio=" + std::to_string(summary.protocolValidationMismatchRatio));
        }
    }
    mmio.write(options.gpioPin, false);
#else
    (void)snapshot;
    (void)options;
    summary.error = "GPIO probe requires Linux on Raspberry Pi";
#endif
    return summary;
}

struct SpiPoint {
    unsigned int requestedHz{0};
    unsigned int actualHz{0};
    double throughputBytesPerSec{0.0};
    bool passed{false};
    std::string note;
};

struct SpiSummary {
    bool attempted{false};
    bool available{false};
    std::string device;
    unsigned int bestStableHz{0};
    double bestThroughputBytesPerSec{0.0};
    std::vector<SpiPoint> points;
    std::string note;
    std::string error;
};

SpiSummary runSpiSummary(const Options &options) {
    SpiSummary summary;
    summary.attempted = true;
    summary.device = options.spiDevice;
#ifdef __linux__
    if (!fileExists(options.spiDevice)) {
        summary.error = "SPI device is unavailable";
        return summary;
    }
    const int fd = ::open(options.spiDevice.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        summary.error = std::string("failed to open ") + options.spiDevice + ": " + std::strerror(errno);
        return summary;
    }
    std::uint8_t mode = 0;
    std::uint8_t bitsPerWord = 8;
    if (ioctl(fd, SPI_IOC_WR_MODE, &mode) < 0 || ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bitsPerWord) < 0) {
        summary.error = std::string("failed to configure SPI mode: ") + std::strerror(errno);
        ::close(fd);
        return summary;
    }

    std::vector<std::uint8_t> tx(256);
    std::vector<std::uint8_t> rx(256, 0);
    for (std::size_t index = 0; index < tx.size(); ++index) {
        tx[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);
    }

    for (unsigned int requestedHz : kDefaultSpiSweepHz) {
        if (requestedHz > options.spiMaxHz) {
            continue;
        }
        unsigned int speed = requestedHz;
        SpiPoint point;
        point.requestedHz = requestedHz;
        if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
            point.note = std::string("set speed failed: ") + std::strerror(errno);
            summary.points.push_back(point);
            continue;
        }
        unsigned int actualHz = 0;
        (void)ioctl(fd, SPI_IOC_RD_MAX_SPEED_HZ, &actualHz);
        if (actualHz == 0U) {
            actualHz = requestedHz;
        }
        point.actualHz = actualHz;

        const std::uint64_t start = nowNs();
        std::uint64_t bytes = 0;
        bool failed = false;
        std::string note;
        while (nowNs() - start < static_cast<std::uint64_t>(options.spiMs) * 1000000ULL) {
            struct spi_ioc_transfer transfer {
            };
            transfer.tx_buf = reinterpret_cast<unsigned long>(tx.data());
            transfer.rx_buf = reinterpret_cast<unsigned long>(rx.data());
            transfer.len = static_cast<std::uint32_t>(tx.size());
            transfer.speed_hz = actualHz;
            transfer.bits_per_word = bitsPerWord;
            if (ioctl(fd, SPI_IOC_MESSAGE(1), &transfer) < 0) {
                failed = true;
                note = std::string("transfer failed: ") + std::strerror(errno);
                break;
            }
            if (options.spiLoopback && rx != tx) {
                failed = true;
                note = "loopback mismatch";
                break;
            }
            bytes += tx.size();
        }
        const std::uint64_t elapsed = std::max<std::uint64_t>(1ULL, nowNs() - start);
        point.throughputBytesPerSec = static_cast<double>(bytes) / nsToSeconds(elapsed);
        point.passed = !failed && bytes > 0;
        point.note = point.passed ? (options.spiLoopback ? "driver+loopback ok" : "driver transport ok") : note;
        summary.points.push_back(point);
        if (point.passed) {
            summary.available = true;
            if (point.actualHz >= summary.bestStableHz) {
                summary.bestStableHz = point.actualHz;
                summary.bestThroughputBytesPerSec = point.throughputBytesPerSec;
            }
        }
    }
    ::close(fd);
    if (!summary.available && summary.error.empty()) {
        summary.error = "no SPI sweep point succeeded";
    }
    if (!options.spiLoopback) {
        summary.note = "without --spi-loopback this is controller-side stability, not board-end echo validation";
    }
#else
    (void)options;
    summary.error = "SPI probe requires Linux";
#endif
    return summary;
}

struct PeakOutRecord {
    std::uint32_t sequence{0};
    std::uint32_t txDataValue{0};
    std::uint32_t txSelectValue{0};
    std::uint32_t rxSelectValue{0};
    std::uint8_t outMask{0};
};

struct PeakCalibrationRecord {
    std::uint32_t sequence{0};
    std::uint32_t txDataValue{0};
    std::uint32_t txSelectValue{0};
    std::uint32_t rxSelectValue{0};
    std::uint8_t outMask{0};
    std::array<std::uint16_t, 6> sum0{{0, 0, 0, 0, 0, 0}};
    std::array<std::uint16_t, 6> sum1{{0, 0, 0, 0, 0, 0}};
};

struct PeakTrialSummary {
    unsigned int settleSpinIterations{0};
    unsigned int windowGapUs{0};
    double windowRateHz{0.0};
    double equivalent8BitGmacs{0.0};
    double equivalent8BitGops{0.0};
    double elapsedSeconds{0.0};
    std::uint64_t windows{0};
    std::uint64_t determinateComparisons{0};
    std::uint64_t indeterminateComparisons{0};
    std::uint64_t equalAdcPairs{0};
    std::uint64_t nonEqualAdcPairs{0};
    std::uint64_t bothZeroAdcPairs{0};
    std::uint16_t minObservedSum0{std::numeric_limits<std::uint16_t>::max()};
    std::uint16_t maxObservedSum0{0};
    std::uint16_t minObservedSum1{std::numeric_limits<std::uint16_t>::max()};
    std::uint16_t maxObservedSum1{0};
    double accuracyPercent{0.0};
    double positivePolarityAccuracyPercent{0.0};
    double negativePolarityAccuracyPercent{0.0};
    bool outHighWhenSum0GeSum1{true};
    bool passed{false};
    std::string note;
    std::string error;
};

struct PeakBenchmarkSummary {
    bool attempted{false};
    bool available{false};
    bool passed{false};
    int calibrationStride{0};
    int tieThresholdCounts{0};
    int sum0CsPin{-1};
    int sum1CsPin{-1};
    std::string spiDevice;
    unsigned int selectedSettleSpinIterations{0};
    unsigned int selectedWindowGapUs{0};
    double selectedWindowRateHz{0.0};
    double selectedEquivalent8BitGmacs{0.0};
    double selectedEquivalent8BitGops{0.0};
    double selectedAccuracyPercent{0.0};
    double selectedElapsedSeconds{0.0};
    std::uint64_t selectedWindows{0};
    std::uint64_t selectedComparisons{0};
    std::uint64_t selectedIndeterminateComparisons{0};
    std::string selectedPolarity;
    std::vector<PeakTrialSummary> trials;
    std::string outTracePath;
    std::string calibrationTracePath;
    std::uint64_t recordedOutWindows{0};
    std::uint64_t recordedCalibrationSamples{0};
    std::string note;
    std::string error;
};

struct PeakTrialCapture {
    PeakTrialSummary summary;
    std::vector<PeakOutRecord> outRecords;
    std::vector<PeakCalibrationRecord> calibrationRecords;
};

struct PeakAccuracyScore {
    std::uint64_t determinate{0};
    std::uint64_t indeterminate{0};
    std::uint64_t positiveMatches{0};
    std::uint64_t negativeMatches{0};
};

PeakAccuracyScore scorePeakCalibrationRecords(const std::vector<PeakCalibrationRecord> &records, int tieThreshold) {
    PeakAccuracyScore score;
    for (const PeakCalibrationRecord &record : records) {
        for (std::size_t index = 0; index < kPage16CarrierFeedbackPins.size(); ++index) {
            const int delta = static_cast<int>(record.sum0[index]) - static_cast<int>(record.sum1[index]);
            if (std::abs(delta) <= tieThreshold) {
                ++score.indeterminate;
                continue;
            }
            const bool outHigh = ((record.outMask >> static_cast<unsigned int>(index)) & 0x01U) != 0U;
            const bool positiveExpected = delta > 0;
            if (outHigh == positiveExpected) {
                ++score.positiveMatches;
            } else {
                ++score.negativeMatches;
            }
            ++score.determinate;
        }
    }
    return score;
}

void accumulatePeakCalibrationStats(PeakTrialSummary &summary, const PeakCalibrationRecord &record) {
    for (std::size_t index = 0; index < kPage16CarrierFeedbackPins.size(); ++index) {
        const std::uint16_t sum0 = record.sum0[index];
        const std::uint16_t sum1 = record.sum1[index];
        summary.minObservedSum0 = std::min(summary.minObservedSum0, sum0);
        summary.maxObservedSum0 = std::max(summary.maxObservedSum0, sum0);
        summary.minObservedSum1 = std::min(summary.minObservedSum1, sum1);
        summary.maxObservedSum1 = std::max(summary.maxObservedSum1, sum1);
        if (sum0 == sum1) {
            ++summary.equalAdcPairs;
        } else {
            ++summary.nonEqualAdcPairs;
        }
        if (sum0 == 0U && sum1 == 0U) {
            ++summary.bothZeroAdcPairs;
        }
    }
}

std::string buildPeakOutTraceCsv(const std::vector<PeakOutRecord> &records) {
    std::ostringstream out;
    out << "sequence,txDataValue,txSelectValue,rxSelectValue,outMask,outBits\n";
    for (const PeakOutRecord &record : records) {
        out << record.sequence << ','
            << record.txDataValue << ','
            << record.txSelectValue << ','
            << record.rxSelectValue << ','
            << static_cast<unsigned int>(record.outMask) << ','
            << maskToBitString(record.outMask, kPage16CarrierFeedbackPins.size()) << '\n';
    }
    return out.str();
}

std::string buildPeakCalibrationTraceCsv(const std::vector<PeakCalibrationRecord> &records) {
    std::ostringstream out;
    out << "sequence,txDataValue,txSelectValue,rxSelectValue,outMask,outBits";
    for (const char *carrier : kPage16CarrierNames) {
        out << ",sum0_" << carrier;
    }
    for (const char *carrier : kPage16CarrierNames) {
        out << ",sum1_" << carrier;
    }
    out << '\n';
    for (const PeakCalibrationRecord &record : records) {
        out << record.sequence << ','
            << record.txDataValue << ','
            << record.txSelectValue << ','
            << record.rxSelectValue << ','
            << static_cast<unsigned int>(record.outMask) << ','
            << maskToBitString(record.outMask, kPage16CarrierFeedbackPins.size());
        for (std::uint16_t value : record.sum0) {
            out << ',' << value;
        }
        for (std::uint16_t value : record.sum1) {
            out << ',' << value;
        }
        out << '\n';
    }
    return out.str();
}

#ifdef __linux__
template <std::size_t N>
void writePinsFromValue(GpioMmio &mmio, const std::array<int, N> &pins, std::uint32_t value) {
    for (std::size_t index = 0; index < pins.size(); ++index) {
        mmio.write(pins[index], ((value >> static_cast<unsigned int>(index)) & 0x01U) != 0U);
    }
}

template <std::size_t N>
std::uint8_t readPinsToMask(const GpioMmio &mmio, const std::array<int, N> &pins) {
    std::uint8_t mask = 0;
    for (std::size_t index = 0; index < pins.size(); ++index) {
        if (mmio.read(pins[index])) {
            mask |= static_cast<std::uint8_t>(1U << static_cast<unsigned int>(index));
        }
    }
    return mask;
}

class ManualChipSelectSpi {
public:
    ~ManualChipSelectSpi() {
        close();
    }

    bool open(const std::string &device, std::string &error) {
        if (!fileExists(device)) {
            error = "SPI device is unavailable";
            return false;
        }
        fd_ = ::open(device.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0) {
            error = std::string("failed to open ") + device + ": " + std::strerror(errno);
            return false;
        }
        std::uint8_t mode = SPI_NO_CS;
        if (ioctl(fd_, SPI_IOC_WR_MODE, &mode) < 0 || ioctl(fd_, SPI_IOC_WR_BITS_PER_WORD, &bitsPerWord_) < 0) {
            error = std::string("failed to configure manual-CS SPI mode: ") + std::strerror(errno);
            close();
            return false;
        }
        return true;
    }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool readChannels(GpioMmio &mmio,
                      int csPin,
                      std::array<std::uint16_t, 6> &values,
                      unsigned int speedHz,
                      std::string &error) {
        for (std::size_t channel = 0; channel < values.size(); ++channel) {
            if (!readSingleChannel(mmio, csPin, static_cast<int>(channel), speedHz, values[channel], error)) {
                return false;
            }
        }
        return true;
    }

private:
    bool readSingleChannel(GpioMmio &mmio,
                           int csPin,
                           int channel,
                           unsigned int speedHz,
                           std::uint16_t &value,
                           std::string &error) {
        std::uint8_t tx[3]{static_cast<std::uint8_t>(0x06U | static_cast<unsigned int>((channel & 0x4) >> 2)),
                           static_cast<std::uint8_t>((channel & 0x3) << 6),
                           0U};
        std::uint8_t rx[3]{0U, 0U, 0U};
        struct spi_ioc_transfer transfer {
        };
        transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
        transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
        transfer.len = 3U;
        transfer.speed_hz = speedHz;
        transfer.bits_per_word = bitsPerWord_;

        mmio.write(csPin, false);
        const int status = ioctl(fd_, SPI_IOC_MESSAGE(1), &transfer);
        mmio.write(csPin, true);
        if (status < 0) {
            error = std::string("SPI calibration transfer failed: ") + std::strerror(errno);
            return false;
        }
        value = static_cast<std::uint16_t>(((rx[1] & 0x0FU) << 8U) | rx[2]);
        return true;
    }

    int fd_{-1};
    std::uint8_t bitsPerWord_{8};
};

bool configurePeakBenchmarkPins(GpioMmio &mmio, const Options &options, std::string &error) {
    for (int pin : kDefaultBoardTxDataPins) {
        if (!mmio.configureOutput(pin, error)) {
            return false;
        }
    }
    for (int pin : kDefaultBoardSelectPins) {
        if (!mmio.configureOutput(pin, error)) {
            return false;
        }
    }
    for (int pin : kPage16CarrierFeedbackPins) {
        if (!mmio.configureInput(pin, error)) {
            return false;
        }
    }
    if (!mmio.configureOutput(options.gpioPin, error) ||
        !mmio.configureOutput(options.calSum0CsPin, error) ||
        !mmio.configureOutput(options.calSum1CsPin, error) ||
        !mmio.configureAlt0(9, error)) {
        return false;
    }
    mmio.write(options.gpioPin, false);
    mmio.write(options.calSum0CsPin, true);
    mmio.write(options.calSum1CsPin, true);
    return true;
}

bool samplePage16Calibration(ManualChipSelectSpi &spi,
                             GpioMmio &mmio,
                             const Options &options,
                             std::uint32_t txDataValue,
                             std::array<std::uint16_t, 6> &sum0,
                             std::array<std::uint16_t, 6> &sum1,
                             std::string &error) {
    if (!mmio.configureAlt0(10, error) || !mmio.configureAlt0(11, error)) {
        return false;
    }
    const unsigned int speedHz = std::max(100000U, std::min(options.spiMaxHz, 1000000U));
    const bool sampled = spi.readChannels(mmio, options.calSum0CsPin, sum0, speedHz, error) &&
                         spi.readChannels(mmio, options.calSum1CsPin, sum1, speedHz, error);

    std::string restoreError;
    const bool restoredMosi = mmio.configureOutput(10, restoreError);
    const bool restoredSclk = mmio.configureOutput(11, restoreError);
    if (restoredMosi) {
        mmio.write(10, (txDataValue & 0x01U) != 0U);
    }
    if (restoredSclk) {
        mmio.write(11, ((txDataValue >> 1U) & 0x01U) != 0U);
    }
    if ((!restoredMosi || !restoredSclk) && error.empty()) {
        error = restoreError;
    }
    return sampled && restoredMosi && restoredSclk;
}
#endif

PeakTrialCapture runPeakTrialCapture(const SystemSnapshot &snapshot,
                                     const Options &options,
                                     unsigned int settleSpinIterations,
                                     unsigned int windowGapUs,
                                     bool recordAllWindows) {
    PeakTrialCapture capture;
    capture.summary.settleSpinIterations = settleSpinIterations;
    capture.summary.windowGapUs = windowGapUs;
#ifdef __linux__
    GpioMmio mmio;
    std::string deviceUsed;
    std::string error;
    if (!mmio.open(snapshot.model, deviceUsed, error)) {
        capture.summary.error = error;
        return capture;
    }

    ManualChipSelectSpi spi;
    if (!spi.open(options.spiDevice, error)) {
        capture.summary.error = error;
        return capture;
    }
    if (!configurePeakBenchmarkPins(mmio, options, error)) {
        capture.summary.error = error;
        return capture;
    }

    const std::uint32_t txDataMask = bitMaskForCount(kDefaultBoardTxDataPins.size());
    const std::uint32_t selectMask = bitMaskForCount(kDefaultBoardSelectPins.size());
    const std::uint64_t start = nowNs();
    std::uint32_t sequence = 1U;
    while (nowNs() - start < static_cast<std::uint64_t>(options.peakBenchmarkMs) * 1000000ULL) {
        const std::uint32_t txDataValue = grayCode(sequence) & txDataMask;
        const std::uint32_t txSelectValue = grayCode(sequence * 3U + 1U) & selectMask;
        const std::uint32_t rxSelectValue = grayCode(sequence * 5U + 3U) & selectMask;

        writePinsFromValue(mmio, kDefaultBoardTxDataPins, txDataValue);
        writePinsFromValue(mmio, kDefaultBoardSelectPins, txSelectValue);
        mmio.write(options.gpioPin, true);
        mmio.write(options.gpioPin, false);
        spinDelay(settleSpinIterations);
        writePinsFromValue(mmio, kDefaultBoardSelectPins, rxSelectValue);

        const std::uint8_t outMask = readPinsToMask(mmio, kPage16CarrierFeedbackPins);
        ++capture.summary.windows;
        if (recordAllWindows) {
            capture.outRecords.push_back(PeakOutRecord{sequence, txDataValue, txSelectValue, rxSelectValue, outMask});
        }

        if (options.peakCalibrationStride > 0 &&
            (capture.summary.windows % static_cast<std::uint64_t>(options.peakCalibrationStride)) == 0ULL) {
            PeakCalibrationRecord record;
            record.sequence = sequence;
            record.txDataValue = txDataValue;
            record.txSelectValue = txSelectValue;
            record.rxSelectValue = rxSelectValue;
            record.outMask = outMask;
            if (!samplePage16Calibration(spi, mmio, options, txDataValue, record.sum0, record.sum1, error)) {
                capture.summary.error = error;
                return capture;
            }
            writePinsFromValue(mmio, kDefaultBoardSelectPins, rxSelectValue);
            accumulatePeakCalibrationStats(capture.summary, record);
            capture.calibrationRecords.push_back(record);
        }
        if (windowGapUs > 0U) {
            std::this_thread::sleep_for(std::chrono::microseconds(windowGapUs));
        }
        ++sequence;
    }
    capture.summary.elapsedSeconds = nsToSeconds(std::max<std::uint64_t>(1ULL, nowNs() - start));
    if (capture.summary.elapsedSeconds > 0.0) {
        capture.summary.windowRateHz = static_cast<double>(capture.summary.windows) / capture.summary.elapsedSeconds;
    }
    capture.summary.equivalent8BitGmacs =
        (capture.summary.windowRateHz * static_cast<double>(options.boardRows) * static_cast<double>(options.boardCols)) / 1000000000.0;
    capture.summary.equivalent8BitGops = capture.summary.equivalent8BitGmacs * 2.0;

    const PeakAccuracyScore score = scorePeakCalibrationRecords(capture.calibrationRecords, options.peakTieThreshold);
    capture.summary.determinateComparisons = score.determinate;
    capture.summary.indeterminateComparisons = score.indeterminate;
    if (score.determinate > 0) {
        capture.summary.positivePolarityAccuracyPercent =
            (static_cast<double>(score.positiveMatches) * 100.0) / static_cast<double>(score.determinate);
        capture.summary.negativePolarityAccuracyPercent =
            (static_cast<double>(score.negativeMatches) * 100.0) / static_cast<double>(score.determinate);
        capture.summary.outHighWhenSum0GeSum1 = score.positiveMatches >= score.negativeMatches;
        capture.summary.accuracyPercent = std::max(capture.summary.positivePolarityAccuracyPercent,
                                                   capture.summary.negativePolarityAccuracyPercent);
    }
    capture.summary.passed = capture.summary.error.empty() &&
                             capture.summary.windows > 0 &&
                             capture.summary.determinateComparisons >= kMinimumPeakComparisons &&
                             capture.summary.accuracyPercent >= kPeakAccuracyFloorPercent;
    if (capture.summary.minObservedSum0 == std::numeric_limits<std::uint16_t>::max()) {
        capture.summary.minObservedSum0 = 0;
    }
    if (capture.summary.minObservedSum1 == std::numeric_limits<std::uint16_t>::max()) {
        capture.summary.minObservedSum1 = 0;
    }
    std::ostringstream note;
    note << "A=SUM0, B=SUM1 from page16 slow ADCs; calibration samples=" << capture.calibrationRecords.size();
    if (capture.summary.determinateComparisons > 0) {
        note << ", polarity="
             << (capture.summary.outHighWhenSum0GeSum1 ? "Out=1 when SUM0>SUM1" : "Out=1 when SUM1>SUM0");
    } else {
        note << ", no determinate SUM0/SUM1 separation exceeded tie threshold";
    }
    note << ", adc equal/non-equal/both-zero=" << capture.summary.equalAdcPairs
         << '/' << capture.summary.nonEqualAdcPairs
         << '/' << capture.summary.bothZeroAdcPairs
         << ", sum0 range=" << capture.summary.minObservedSum0 << ".." << capture.summary.maxObservedSum0
         << ", sum1 range=" << capture.summary.minObservedSum1 << ".." << capture.summary.maxObservedSum1;
    capture.summary.note = note.str();
#else
    (void)snapshot;
    (void)options;
    (void)settleSpinIterations;
    (void)windowGapUs;
    (void)recordAllWindows;
    capture.summary.error = "Peak accuracy benchmark requires Linux on Raspberry Pi";
#endif
    return capture;
}

PeakBenchmarkSummary runPeakAccuracyBenchmark(const SystemSnapshot &snapshot,
                                             const Options &options,
                                             const GpioSummary &gpio,
                                             const SpiSummary &spi) {
    PeakBenchmarkSummary summary;
    summary.attempted = true;
    summary.calibrationStride = options.peakCalibrationStride;
    summary.tieThresholdCounts = options.peakTieThreshold;
    summary.sum0CsPin = options.calSum0CsPin;
    summary.sum1CsPin = options.calSum1CsPin;
    summary.spiDevice = options.spiDevice;

    if (!options.forcePage16CalibrationSpi) {
        summary.attempted = false;
        summary.note = "current numbered eext_netlist_16.json keeps U160/U161 on CAL_SPI_* and CAL_ADC_SUM*_CS nets that are not wired to Pi J1 SPI0 or GPIO18/GPIO19, so page16 slow-ADC scoring is disabled by default; use --force-page16-calibration-spi only for a custom board that intentionally differs from the repo netlist";
        return summary;
    }

    if (options.skipGpio || options.skipSpi) {
        summary.attempted = false;
        summary.note = "requires both GPIO fast path and SPI0 calibration access";
        return summary;
    }
    if (!gpio.available) {
        summary.error = gpio.error.empty() ? "GPIO fast path is unavailable" : gpio.error;
        return summary;
    }
    if (!spi.available) {
        summary.error = spi.error.empty() ? "SPI calibration path is unavailable" : spi.error;
        return summary;
    }
    if (options.calSum0CsPin == options.calSum1CsPin) {
        summary.error = "SUM0 and SUM1 ADC chip-select pins must be different";
        return summary;
    }

    std::vector<int> forbiddenPins(kDefaultBoardTxDataPins.begin(), kDefaultBoardTxDataPins.end());
    forbiddenPins.insert(forbiddenPins.end(), kDefaultBoardSelectPins.begin(), kDefaultBoardSelectPins.end());
    forbiddenPins.push_back(options.gpioPin);
    forbiddenPins.push_back(options.calSum0CsPin);
    forbiddenPins.push_back(options.calSum1CsPin);
    std::sort(forbiddenPins.begin(), forbiddenPins.end());
    forbiddenPins.erase(std::unique(forbiddenPins.begin(), forbiddenPins.end()), forbiddenPins.end());
    for (int pin : kPage16CarrierFeedbackPins) {
        if (std::find(forbiddenPins.begin(), forbiddenPins.end(), pin) != forbiddenPins.end()) {
            summary.error = "page16 feedback pins overlap the active protocol domain";
            return summary;
        }
    }

    PeakTrialSummary bestObserved;
    bool haveBestObserved = false;
    for (unsigned int windowGapUs : kPeakWindowGapUsCandidates) {
        for (unsigned int settleSpinIterations : kPeakSettleSpinCandidates) {
            PeakTrialCapture discovery = runPeakTrialCapture(snapshot, options, settleSpinIterations, windowGapUs, false);
            summary.trials.push_back(discovery.summary);
            summary.available = true;
            if (!discovery.summary.error.empty()) {
                summary.error = discovery.summary.error;
                return summary;
            }
            if (!haveBestObserved || discovery.summary.accuracyPercent > bestObserved.accuracyPercent ||
                (discovery.summary.accuracyPercent == bestObserved.accuracyPercent &&
                 discovery.summary.equivalent8BitGmacs > bestObserved.equivalent8BitGmacs)) {
                bestObserved = discovery.summary;
                haveBestObserved = true;
            }
            if (!discovery.summary.passed) {
                continue;
            }

            PeakTrialCapture finalCapture = runPeakTrialCapture(snapshot, options, settleSpinIterations, windowGapUs, true);
            if (!finalCapture.summary.error.empty()) {
                summary.error = finalCapture.summary.error;
                return summary;
            }
            if (!finalCapture.summary.passed) {
                summary.trials.back().note += "; selected-candidate full capture fell below 80% accuracy after Out recording";
                if (!haveBestObserved || finalCapture.summary.accuracyPercent > bestObserved.accuracyPercent ||
                    (finalCapture.summary.accuracyPercent == bestObserved.accuracyPercent &&
                     finalCapture.summary.equivalent8BitGmacs > bestObserved.equivalent8BitGmacs)) {
                    bestObserved = finalCapture.summary;
                    haveBestObserved = true;
                }
                continue;
            }

            summary.passed = true;
            summary.selectedSettleSpinIterations = finalCapture.summary.settleSpinIterations;
            summary.selectedWindowGapUs = finalCapture.summary.windowGapUs;
            summary.selectedWindowRateHz = finalCapture.summary.windowRateHz;
            summary.selectedEquivalent8BitGmacs = finalCapture.summary.equivalent8BitGmacs;
            summary.selectedEquivalent8BitGops = finalCapture.summary.equivalent8BitGops;
            summary.selectedAccuracyPercent = finalCapture.summary.accuracyPercent;
            summary.selectedElapsedSeconds = finalCapture.summary.elapsedSeconds;
            summary.selectedWindows = finalCapture.summary.windows;
            summary.selectedComparisons = finalCapture.summary.determinateComparisons;
            summary.selectedIndeterminateComparisons = finalCapture.summary.indeterminateComparisons;
            summary.selectedPolarity = finalCapture.summary.outHighWhenSum0GeSum1 ? "Out=1 when SUM0>SUM1" : "Out=1 when SUM1>SUM0";
            summary.recordedOutWindows = finalCapture.outRecords.size();
            summary.recordedCalibrationSamples = finalCapture.calibrationRecords.size();
            if (!options.peakTracePrefix.empty()) {
                summary.outTracePath = options.peakTracePrefix + ".out.csv";
                summary.calibrationTracePath = options.peakTracePrefix + ".samples.csv";
                std::string writeError;
                if (!writeReportFile(summary.outTracePath, buildPeakOutTraceCsv(finalCapture.outRecords), writeError) ||
                    !writeReportFile(summary.calibrationTracePath, buildPeakCalibrationTraceCsv(finalCapture.calibrationRecords), writeError)) {
                    summary.error = writeError;
                    summary.passed = false;
                    return summary;
                }
            }
            summary.note = "records every page16 fast comparator Out window, samples SUM0/SUM1 opportunistically through the slow calibration ADCs, sweeps slower inter-window gaps when needed, and scores polarity + accuracy on the Pi CPU after capture";
            return summary;
        }
    }

    if (haveBestObserved) {
        std::ostringstream note;
        note << "no settle-spin candidate reached " << kPeakAccuracyFloorPercent << "% accuracy; best observed was "
             << std::fixed << std::setprecision(2) << bestObserved.accuracyPercent << "% at "
             << bestObserved.equivalent8BitGmacs << " GMAC/s with gap=" << bestObserved.windowGapUs
             << "us, spins=" << bestObserved.settleSpinIterations;
        summary.note = note.str();
    } else {
        summary.note = "peak accuracy benchmark did not collect any usable calibration samples";
    }
    if (!options.peakTracePrefix.empty() && haveBestObserved) {
        PeakTrialCapture diagnosticCapture = runPeakTrialCapture(snapshot,
                                                                 options,
                                                                 bestObserved.settleSpinIterations,
                                                                 bestObserved.windowGapUs,
                                                                 true);
        if (!diagnosticCapture.summary.error.empty()) {
            if (summary.error.empty()) {
                summary.error = diagnosticCapture.summary.error;
            }
            return summary;
        }
        summary.outTracePath = options.peakTracePrefix + ".best_effort.out.csv";
        summary.calibrationTracePath = options.peakTracePrefix + ".best_effort.samples.csv";
        summary.recordedOutWindows = diagnosticCapture.outRecords.size();
        summary.recordedCalibrationSamples = diagnosticCapture.calibrationRecords.size();
        std::string writeError;
        if (!writeReportFile(summary.outTracePath, buildPeakOutTraceCsv(diagnosticCapture.outRecords), writeError) ||
            !writeReportFile(summary.calibrationTracePath, buildPeakCalibrationTraceCsv(diagnosticCapture.calibrationRecords), writeError)) {
            summary.error = writeError;
            return summary;
        }
        if (!summary.note.empty()) {
            summary.note += "; wrote best-effort failure traces for the highest-throughput candidate";
        }
    }
    return summary;
}

struct CorrectnessCheck {
    std::string name;
    bool passed{false};
    std::string note;
};

struct CorrectnessSummary {
    bool attempted{false};
    bool available{false};
    bool passed{false};
    double protocolWindowRateHz{0.0};
    double selfReadbackMismatchRatio{0.0};
    int activeFeedbackPins{0};
    int totalFeedbackPins{0};
    std::vector<CorrectnessCheck> checks;
    std::string note;
    std::string error;
};

CorrectnessSummary buildCorrectnessSummary(const Options &options, const GpioSummary &gpio) {
    CorrectnessSummary summary;
    summary.attempted = gpio.attempted;
    summary.available = gpio.available;
    summary.protocolWindowRateHz = gpio.protocolWindowRateHz;
    summary.selfReadbackMismatchRatio = gpio.protocolValidationMismatchRatio;
    summary.totalFeedbackPins = static_cast<int>(gpio.protocolFeedback.size());
    if (!gpio.attempted) {
        summary.note = "GPIO probe was skipped";
        return summary;
    }
    if (!gpio.available) {
        summary.error = gpio.error;
        return summary;
    }

    std::vector<int> protocolOutputPins;
    protocolOutputPins.insert(protocolOutputPins.end(), kDefaultBoardTxDataPins.begin(), kDefaultBoardTxDataPins.end());
    protocolOutputPins.insert(protocolOutputPins.end(), kDefaultBoardSelectPins.begin(), kDefaultBoardSelectPins.end());
    protocolOutputPins.push_back(options.gpioPin);
    std::sort(protocolOutputPins.begin(), protocolOutputPins.end());
    protocolOutputPins.erase(std::unique(protocolOutputPins.begin(), protocolOutputPins.end()), protocolOutputPins.end());

    std::vector<int> overlaps;
    for (int pin : options.feedbackPins) {
        if (std::find(protocolOutputPins.begin(), protocolOutputPins.end(), pin) != protocolOutputPins.end()) {
            overlaps.push_back(pin);
        }
    }

    summary.checks.push_back(CorrectnessCheck{"pin-domain-separation",
                                              overlaps.empty(),
                                              overlaps.empty() ? "feedback inputs are separate from the GPIO output domain"
                                                               : ("overlapping BCM lines: " + joinInts(overlaps))});

    const bool protocolRatePassed = gpio.protocolWindowRateHz > 0.0 && gpio.protocolOpsPerWindow > 0.0;
    summary.checks.push_back(CorrectnessCheck{"protocol-window-execution",
                                              protocolRatePassed,
                                              protocolRatePassed ? ("window rate=" + formatHz(gpio.protocolWindowRateHz))
                                                                 : "protocol-shaped GPIO window benchmark did not complete"});

    std::ostringstream mismatchNote;
    mismatchNote << "mismatch ratio=" << std::fixed << std::setprecision(6) << gpio.protocolValidationMismatchRatio
                 << " from " << gpio.protocolValidationSamples << " sample-sync reads";
    summary.checks.push_back(CorrectnessCheck{"sample-sync-self-readback",
                                              gpio.protocolValidationSamples > 0 &&
                                                  gpio.protocolValidationMismatchRatio <= kCorrectnessMismatchCeiling,
                                              mismatchNote.str()});

    const std::uint64_t activeTransitionThreshold = gpio.protocolWindows > 0 ? std::max<std::uint64_t>(1ULL, gpio.protocolWindows / 256ULL) : 1ULL;
    for (const FeedbackObservation &observation : gpio.protocolFeedback) {
        const bool active = observation.observedTransitions >= activeTransitionThreshold &&
                            observation.highSamples > 0 && observation.lowSamples > 0;
        if (active) {
            ++summary.activeFeedbackPins;
        }
    }
    const int requiredActivePins = std::max(kMinimumActiveFeedbackPins, summary.totalFeedbackPins > 0 ? (summary.totalFeedbackPins + 2) / 3 : 0);
    std::ostringstream feedbackNote;
    feedbackNote << "active feedback pins=" << summary.activeFeedbackPins << " / " << summary.totalFeedbackPins
                 << ", threshold=" << requiredActivePins;
    summary.checks.push_back(CorrectnessCheck{"feedback-electrical-activity",
                                              summary.activeFeedbackPins >= requiredActivePins,
                                              feedbackNote.str()});

    summary.passed = std::all_of(summary.checks.begin(), summary.checks.end(), [](const CorrectnessCheck &check) {
        return check.passed;
    });
    summary.note = "validates the netlist-grounded hot path only: page9 DAC write lines on BCM10/11/8/7, sample-sync on BCM6, and live page16 fast-comparator feedback on BCM5/12/13/16/20/21; current numbered netlists expose no Pi-visible numerical oracle, so this does not prove analog numerical correctness against a golden tensor";
    return summary;
}

struct BoardEstimate {
    bool available{false};
    std::string transportModel{"gpio-page9-dac-plus-page16-fast-comparator"};
    int matrixRows{kDefaultBoardRows};
    int matrixCols{kDefaultBoardCols};
    double macsPerWindow{0.0};
    double equivalent8BitOpsPerWindow{0.0};
    double estimatedWindowRateHz{0.0};
    double gpioOpsPerWindow{0.0};
    double gpioLineOpsPerSec{0.0};
    double estimatedVectorOutputsPerSec{0.0};
    double estimatedComparatorDecisionsPerSec{0.0};
    double estimatedEquivalent8BitGmacs{0.0};
    double estimatedEquivalent8BitGops{0.0};
    std::string limitingFactor;
    std::string note;
    std::string alternateTransportNote;
};

BoardEstimate estimateBoardBudget(const Options &options, const GpioSummary &gpio, const SpiSummary &spi, const CorrectnessSummary &correctness) {
    BoardEstimate estimate;
    estimate.matrixRows = options.boardRows;
    estimate.matrixCols = options.boardCols;
    estimate.macsPerWindow = static_cast<double>(estimate.matrixRows) * static_cast<double>(estimate.matrixCols);
    estimate.equivalent8BitOpsPerWindow = estimate.macsPerWindow * 2.0;
    if (!(gpio.available && gpio.protocolWindowRateHz > 0.0)) {
        estimate.note = "need a valid gpio-async protocol window benchmark to estimate whole-board GPIO NPU throughput";
        if (spi.available) {
            estimate.alternateTransportNote = "spidev sweep peaked at " + formatThroughput(spi.bestThroughputBytesPerSec) +
                                              ", but that alternate path is not used in the whole-board GPIO async estimate";
        }
        return estimate;
    }
    estimate.available = true;
    estimate.estimatedWindowRateHz = gpio.protocolWindowRateHz;
    estimate.gpioOpsPerWindow = gpio.protocolOpsPerWindow;
    estimate.gpioLineOpsPerSec = gpio.protocolLineOpsPerSec;
    estimate.estimatedVectorOutputsPerSec = estimate.estimatedWindowRateHz * static_cast<double>(estimate.matrixCols);
    estimate.estimatedComparatorDecisionsPerSec = estimate.estimatedWindowRateHz * 6.0;
    estimate.estimatedEquivalent8BitGmacs = (estimate.estimatedWindowRateHz * estimate.macsPerWindow) / 1000000000.0;
    estimate.estimatedEquivalent8BitGops = (estimate.estimatedWindowRateHz * estimate.equivalent8BitOpsPerWindow) / 1000000000.0;
    estimate.limitingFactor = "GPIO gpio-async protocol line ops";
    estimate.note = "whole-board equivalent assumes one committed window exercises the current numbered-netlist hot path: page9 DAC write lines on BCM10/11/8/7, one BCM6 sample-sync pulse, and six page16 fast-comparator reads; GPIO22 and BCM17/27/25 are excluded because the routed numbered netlist does not place them on the whole-board compute path";
    if (correctness.available && !correctness.passed) {
        estimate.note += "; correctness validation did not fully pass, so treat the equivalent GOPS as provisional";
    }
    if (spi.available) {
        estimate.alternateTransportNote = "spidev sweep peaked at " + formatThroughput(spi.bestThroughputBytesPerSec) +
                                          ", but the gpio-async whole-board estimate intentionally ignores SPI because the protocol executes as per-line GPIO on this board";
    }
    return estimate;
}

void printSystemSection(const SystemSnapshot &snapshot) {
    std::cout << "== System ==\n";
    std::cout << "Model: " << (snapshot.model.empty() ? "unknown" : snapshot.model) << '\n';
    std::cout << "Kernel: " << snapshot.osRelease << ' ' << snapshot.kernel << " / " << snapshot.machine << '\n';
    std::cout << "Cores: online=" << snapshot.onlineCores << ", configured=" << snapshot.configuredCores << '\n';
    std::cout << "Page size: " << snapshot.pageSize << " bytes\n";
    std::cout << "Memory: total=" << formatBytes(snapshot.totalMemoryBytes)
              << ", free=" << formatBytes(snapshot.freeMemoryBytes)
              << ", available=" << formatBytes(snapshot.availableMemoryBytes) << '\n';
    if (snapshot.thermalMilliC >= 0) {
        std::cout << "Thermal: " << std::fixed << std::setprecision(1) << (snapshot.thermalMilliC / 1000.0) << " C\n";
    }
    if (!snapshot.currentCpuFreqKHz.empty()) {
        std::cout << "CPU freq: ";
        for (std::size_t index = 0; index < snapshot.currentCpuFreqKHz.size(); ++index) {
            if (index != 0) {
                std::cout << ", ";
            }
            std::cout << snapshot.currentCpuFreqKHz[index] / 1000.0 << " MHz";
        }
        std::cout << '\n';
    }
    if (!snapshot.hwcaps.empty()) {
        std::cout << "HWCAP: ";
        bool first = true;
        for (const auto &entry : snapshot.hwcaps) {
            if (!entry.second) {
                continue;
            }
            if (!first) {
                std::cout << ", ";
            }
            std::cout << entry.first;
            first = false;
        }
        if (first) {
            std::cout << "none reported";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void printComputeSection(const ComputeSummary &summary) {
    std::cout << "== Compute ==\n";
    const auto printRun = [](const BenchmarkRun &run) {
        std::cout << run.name << ": ";
        if (!run.available) {
            std::cout << "unavailable";
            if (!run.note.empty()) {
                std::cout << " (" << run.note << ')';
            }
            std::cout << '\n';
            return;
        }
        std::cout << std::fixed << std::setprecision(2) << run.gigaOpsPerSec << " GOPS"
                  << " in " << run.seconds << " s";
        if (!run.note.empty()) {
            std::cout << " (" << run.note << ')';
        }
        std::cout << '\n';
    };
    printRun(summary.integerSingle);
    printRun(summary.floatSingle);
    printRun(summary.neonSingle);
    std::cout << "Recommended threads: " << summary.recommendedThreads << " using " << summary.preferredKernel << '\n';
    if (!summary.scaling.empty()) {
        std::cout << "Scaling:" << '\n';
        for (const ScalingPoint &point : summary.scaling) {
            std::cout << "  " << point.threads << " thread(s): "
                      << std::fixed << std::setprecision(2) << point.gigaOpsPerSec << " GOPS, speedup "
                      << point.speedup << ", efficiency " << (point.efficiency * 100.0) << "%\n";
        }
    }
    std::cout << '\n';
}

void printMemorySection(const MemorySummary &summary) {
    std::cout << "== Memory ==\n";
    if (!summary.available) {
        std::cout << "Unavailable";
        if (!summary.note.empty()) {
            std::cout << ": " << summary.note;
        }
        std::cout << "\n\n";
        return;
    }
    std::cout << "Working set: " << formatBytes(summary.workingSetBytes) << '\n';
    std::cout << "Read bandwidth: " << formatThroughput(summary.readBytesPerSec) << '\n';
    std::cout << "Copy bandwidth: " << formatThroughput(summary.copyBytesPerSec) << "\n\n";
}

void printGpioSection(const GpioSummary &summary) {
    std::cout << "== GPIO ==\n";
    if (!summary.available) {
        std::cout << "Unavailable";
        if (!summary.error.empty()) {
            std::cout << ": " << summary.error;
        }
        std::cout << "\n\n";
        return;
    }
    std::cout << "Device: " << summary.device << '\n';
    std::cout << "Output pin: BCM" << summary.pin << '\n';
    std::cout << "Observed feedback pins: " << (summary.feedbackPins.empty() ? std::string("none") : joinInts(summary.feedbackPins)) << '\n';
    std::cout << "Open-loop toggle ceiling: " << formatHz(summary.openLoopToggleHz) << '\n';
    std::cout << "Closed-loop validated toggle ceiling: " << formatHz(summary.validatedToggleHz) << '\n';
    std::cout << "Recommended stable toggle rate: " << formatHz(summary.recommendedToggleHz) << '\n';
    std::cout << "Self-readback mismatches: " << summary.mismatchCount << " / " << summary.sampleCount << '\n';
    if (summary.protocolWindowRateHz > 0.0) {
        std::cout << "Protocol-shaped window rate: " << formatHz(summary.protocolWindowRateHz) << '\n';
        std::cout << "Protocol GPIO ops/window: " << std::fixed << std::setprecision(2) << summary.protocolOpsPerWindow << '\n';
        std::cout << "Protocol GPIO line ops/s: " << std::fixed << std::setprecision(2) << (summary.protocolLineOpsPerSec / 1000000.0) << " Mops/s\n";
        std::cout << "Protocol validation mismatches: " << summary.protocolValidationMismatches << " / " << summary.protocolValidationSamples << '\n';
    }
    for (const FeedbackObservation &observation : summary.feedback) {
        const std::uint64_t total = observation.highSamples + observation.lowSamples;
        std::cout << "  BCM" << observation.pin << ": transitions=" << observation.observedTransitions;
        if (total > 0) {
            const double highRatio = static_cast<double>(observation.highSamples) / static_cast<double>(total);
            std::cout << ", high-ratio=" << std::fixed << std::setprecision(3) << highRatio;
        }
        std::cout << '\n';
    }
    if (!summary.note.empty()) {
        std::cout << "Note: " << summary.note << '\n';
    }
    std::cout << '\n';
}

void printCorrectnessSection(const CorrectnessSummary &summary) {
    std::cout << "== Correctness Validation ==\n";
    if (!summary.attempted) {
        std::cout << "Unavailable: GPIO probe was skipped\n\n";
        return;
    }
    if (!summary.available) {
        std::cout << "Unavailable";
        if (!summary.error.empty()) {
            std::cout << ": " << summary.error;
        }
        std::cout << "\n\n";
        return;
    }
    std::cout << "Protocol window rate: " << formatHz(summary.protocolWindowRateHz) << '\n';
    std::cout << "Sample-sync mismatch ratio: " << std::fixed << std::setprecision(6) << summary.selfReadbackMismatchRatio << '\n';
    std::cout << "Active feedback pins: " << summary.activeFeedbackPins << " / " << summary.totalFeedbackPins << '\n';
    std::cout << "Overall result: " << (summary.passed ? "pass" : "review required") << '\n';
    for (const CorrectnessCheck &check : summary.checks) {
        std::cout << "  " << check.name << ": " << (check.passed ? "ok" : "fail");
        if (!check.note.empty()) {
            std::cout << " (" << check.note << ')';
        }
        std::cout << '\n';
    }
    if (!summary.note.empty()) {
        std::cout << "Note: " << summary.note << '\n';
    }
    std::cout << '\n';
}

void printSpiSection(const SpiSummary &summary) {
    std::cout << "== SPI ==\n";
    if (!summary.available) {
        std::cout << "Unavailable";
        if (!summary.error.empty()) {
            std::cout << ": " << summary.error;
        }
        std::cout << "\n\n";
        return;
    }
    std::cout << "Device: " << summary.device << '\n';
    std::cout << "Best stable speed: " << formatHz(static_cast<double>(summary.bestStableHz)) << '\n';
    std::cout << "Best throughput: " << formatThroughput(summary.bestThroughputBytesPerSec) << '\n';
    for (const SpiPoint &point : summary.points) {
        std::cout << "  " << formatHz(static_cast<double>(point.requestedHz))
                  << " -> actual " << formatHz(static_cast<double>(point.actualHz))
                  << ", " << (point.passed ? "ok" : "fail")
                  << ", " << formatThroughput(point.throughputBytesPerSec);
        if (!point.note.empty()) {
            std::cout << " (" << point.note << ')';
        }
        std::cout << '\n';
    }
    if (!summary.note.empty()) {
        std::cout << "Note: " << summary.note << '\n';
    }
    std::cout << '\n';
}

void printPeakAccuracySection(const PeakBenchmarkSummary &summary) {
    std::cout << "== Peak Accuracy Benchmark ==\n";
    if (!summary.attempted) {
        std::cout << "Unavailable";
        if (!summary.note.empty()) {
            std::cout << ": " << summary.note;
        }
        std::cout << "\n\n";
        return;
    }
    if (!summary.available) {
        std::cout << "Unavailable";
        if (!summary.error.empty()) {
            std::cout << ": " << summary.error;
        } else if (!summary.note.empty()) {
            std::cout << ": " << summary.note;
        }
        std::cout << "\n\n";
        return;
    }

    std::cout << "Calibration stride: every " << summary.calibrationStride << " windows\n";
    std::cout << "Tie-zone threshold: +/-" << summary.tieThresholdCounts << " ADC counts\n";
    std::cout << "Calibration ADC CS: BCM" << summary.sum0CsPin << " / BCM" << summary.sum1CsPin << '\n';
    std::cout << "Overall result: " << (summary.passed ? "pass" : "review required") << '\n';
    if (summary.passed) {
        std::cout << "Selected settle spins: " << summary.selectedSettleSpinIterations << '\n';
        std::cout << "Selected inter-window gap: " << summary.selectedWindowGapUs << " us\n";
        std::cout << "Selected polarity: " << summary.selectedPolarity << '\n';
        std::cout << "Recorded Out windows: " << summary.recordedOutWindows << '\n';
        std::cout << "Recorded calibration samples: " << summary.recordedCalibrationSamples << '\n';
        std::cout << "Peak accuracy: " << std::fixed << std::setprecision(2) << summary.selectedAccuracyPercent << "%\n";
        std::cout << "Peak window rate: " << formatHz(summary.selectedWindowRateHz) << '\n';
        std::cout << "Peak equivalent throughput: " << std::fixed << std::setprecision(3)
                  << summary.selectedEquivalent8BitGops << " GOPS (" << summary.selectedEquivalent8BitGmacs << " GMAC/s)\n";
        std::cout << "Elapsed capture time: " << std::fixed << std::setprecision(3) << summary.selectedElapsedSeconds << " s\n";
        std::cout << "Scored comparisons: " << summary.selectedComparisons
                  << " determinate, " << summary.selectedIndeterminateComparisons << " tie-zone excluded\n";
        if (!summary.outTracePath.empty()) {
            std::cout << "Out trace: " << summary.outTracePath << '\n';
        }
        if (!summary.calibrationTracePath.empty()) {
            std::cout << "Calibration trace: " << summary.calibrationTracePath << '\n';
        }
    }
    if (!summary.trials.empty()) {
        std::cout << "Trials:" << '\n';
        for (const PeakTrialSummary &trial : summary.trials) {
            std::cout << "  spins=" << trial.settleSpinIterations
                      << ", gapUs=" << trial.windowGapUs
                      << ", rate=" << formatHz(trial.windowRateHz)
                      << ", accuracy=" << std::fixed << std::setprecision(2) << trial.accuracyPercent << "%"
                      << ", throughput=" << std::fixed << std::setprecision(3) << trial.equivalent8BitGmacs << " GMAC/s"
                      << ", comparisons=" << trial.determinateComparisons
                      << ", " << (trial.passed ? "pass" : "fail");
            if (!trial.note.empty()) {
                std::cout << " (" << trial.note << ')';
            }
            if (!trial.error.empty()) {
                std::cout << " [" << trial.error << ']';
            }
            std::cout << '\n';
        }
    }
    if (!summary.note.empty()) {
        std::cout << "Note: " << summary.note << '\n';
    }
    if (!summary.error.empty()) {
        std::cout << "Error: " << summary.error << '\n';
    }
    std::cout << '\n';
}

void printBudgetSection(const BoardEstimate &estimate) {
    std::cout << "== Board NPU 8-bit Equivalent ==\n";
    if (!estimate.available) {
        std::cout << "Unavailable";
        if (!estimate.note.empty()) {
            std::cout << ": " << estimate.note;
        }
        if (!estimate.alternateTransportNote.empty()) {
            std::cout << "\nAlternate path: " << estimate.alternateTransportNote;
        }
        std::cout << "\n\n";
        return;
    }
    std::cout << "Transport model: " << estimate.transportModel << '\n';
    std::cout << "Board matrix: " << estimate.matrixRows << " x " << estimate.matrixCols << '\n';
    std::cout << "Protocol window rate: " << formatHz(estimate.estimatedWindowRateHz) << '\n';
    std::cout << "GPIO ops/window: " << std::fixed << std::setprecision(2) << estimate.gpioOpsPerWindow << '\n';
    std::cout << "GPIO line ops/s: " << std::fixed << std::setprecision(2) << (estimate.gpioLineOpsPerSec / 1000000.0) << " Mops/s\n";
    std::cout << "Equivalent 8-bit MACs/window: " << estimate.macsPerWindow << '\n';
    std::cout << "Equivalent 8-bit ops/window: " << estimate.equivalent8BitOpsPerWindow << '\n';
    std::cout << "Equivalent 8-bit throughput: " << std::fixed << std::setprecision(3) << estimate.estimatedEquivalent8BitGops
              << " GOPS (" << estimate.estimatedEquivalent8BitGmacs << " GMAC/s)\n";
    std::cout << "Estimated vector outputs/s: " << estimate.estimatedVectorOutputsPerSec << '\n';
    std::cout << "Estimated fast comparator decisions/s: " << estimate.estimatedComparatorDecisionsPerSec << '\n';
    std::cout << "Limiting factor: " << estimate.limitingFactor << '\n';
    std::cout << "Note: " << estimate.note << '\n';
    if (!estimate.alternateTransportNote.empty()) {
        std::cout << "Alternate path: " << estimate.alternateTransportNote << '\n';
    }
    std::cout << '\n';
}

std::string buildJsonReport(const Options &options,
                            const SystemSnapshot &snapshot,
                            const ComputeSummary &compute,
                            const MemorySummary &memory,
                            const GpioSummary &gpio,
                            const SpiSummary &spi,
                            const CorrectnessSummary &correctness,
                            const PeakBenchmarkSummary &peak,
                            const BoardEstimate &budget) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"program\": \"rpi_zero2w_capability_probe\",\n";
    out << "  \"gpioPin\": " << options.gpioPin << ",\n";
    out << "  \"boardRows\": " << options.boardRows << ",\n";
    out << "  \"boardCols\": " << options.boardCols << ",\n";
    out << "  \"feedbackPins\": [";
    for (std::size_t index = 0; index < options.feedbackPins.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << options.feedbackPins[index];
    }
    out << "],\n";
    out << "  \"system\": {\n";
    out << "    \"model\": \"" << jsonEscape(snapshot.model) << "\",\n";
    out << "    \"kernel\": \"" << jsonEscape(snapshot.kernel) << "\",\n";
    out << "    \"machine\": \"" << jsonEscape(snapshot.machine) << "\",\n";
    out << "    \"configuredCores\": " << snapshot.configuredCores << ",\n";
    out << "    \"onlineCores\": " << snapshot.onlineCores << ",\n";
    out << "    \"pageSize\": " << snapshot.pageSize << ",\n";
    out << "    \"totalMemoryBytes\": " << snapshot.totalMemoryBytes << ",\n";
    out << "    \"availableMemoryBytes\": " << snapshot.availableMemoryBytes << ",\n";
    out << "    \"thermalMilliC\": " << snapshot.thermalMilliC << "\n";
    out << "  },\n";
    out << "  \"compute\": {\n";
    out << "    \"preferredKernel\": \"" << jsonEscape(compute.preferredKernel) << "\",\n";
    out << "    \"recommendedThreads\": " << compute.recommendedThreads << ",\n";
    out << "    \"integerSingleGops\": " << compute.integerSingle.gigaOpsPerSec << ",\n";
    out << "    \"floatSingleGops\": " << compute.floatSingle.gigaOpsPerSec << ",\n";
    out << "    \"neonSingleGops\": " << compute.neonSingle.gigaOpsPerSec << ",\n";
    out << "    \"scaling\": [\n";
    for (std::size_t index = 0; index < compute.scaling.size(); ++index) {
        const ScalingPoint &point = compute.scaling[index];
        out << "      {\"threads\": " << point.threads
            << ", \"gops\": " << point.gigaOpsPerSec
            << ", \"speedup\": " << point.speedup
            << ", \"efficiency\": " << point.efficiency << "}";
        out << (index + 1 == compute.scaling.size() ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"memory\": {\n";
    out << "    \"available\": " << (memory.available ? "true" : "false") << ",\n";
    out << "    \"workingSetBytes\": " << memory.workingSetBytes << ",\n";
    out << "    \"readBytesPerSec\": " << memory.readBytesPerSec << ",\n";
    out << "    \"copyBytesPerSec\": " << memory.copyBytesPerSec << "\n";
    out << "  },\n";
    out << "  \"gpio\": {\n";
    out << "    \"available\": " << (gpio.available ? "true" : "false") << ",\n";
    out << "    \"device\": \"" << jsonEscape(gpio.device) << "\",\n";
    out << "    \"openLoopToggleHz\": " << gpio.openLoopToggleHz << ",\n";
    out << "    \"validatedToggleHz\": " << gpio.validatedToggleHz << ",\n";
    out << "    \"recommendedToggleHz\": " << gpio.recommendedToggleHz << ",\n";
    out << "    \"mismatchRatio\": " << gpio.mismatchRatio << ",\n";
    out << "    \"sampleCount\": " << gpio.sampleCount << ",\n";
    out << "    \"mismatchCount\": " << gpio.mismatchCount << ",\n";
    out << "    \"protocolWindowRateHz\": " << gpio.protocolWindowRateHz << ",\n";
    out << "    \"protocolLineOpsPerSec\": " << gpio.protocolLineOpsPerSec << ",\n";
    out << "    \"protocolOpsPerWindow\": " << gpio.protocolOpsPerWindow << ",\n";
    out << "    \"protocolWindows\": " << gpio.protocolWindows << ",\n";
    out << "    \"protocolValidationSamples\": " << gpio.protocolValidationSamples << ",\n";
    out << "    \"protocolValidationMismatches\": " << gpio.protocolValidationMismatches << ",\n";
    out << "    \"protocolValidationMismatchRatio\": " << gpio.protocolValidationMismatchRatio << ",\n";
    out << "    \"note\": \"" << jsonEscape(gpio.note) << "\",\n";
    out << "    \"feedback\": [\n";
    for (std::size_t index = 0; index < gpio.feedback.size(); ++index) {
        const FeedbackObservation &observation = gpio.feedback[index];
        out << "      {\"pin\": " << observation.pin
            << ", \"highSamples\": " << observation.highSamples
            << ", \"lowSamples\": " << observation.lowSamples
            << ", \"observedTransitions\": " << observation.observedTransitions << "}";
        out << (index + 1 == gpio.feedback.size() ? "\n" : ",\n");
    }
    out << "    ],\n";
    out << "    \"protocolFeedback\": [\n";
    for (std::size_t index = 0; index < gpio.protocolFeedback.size(); ++index) {
        const FeedbackObservation &observation = gpio.protocolFeedback[index];
        out << "      {\"pin\": " << observation.pin
            << ", \"highSamples\": " << observation.highSamples
            << ", \"lowSamples\": " << observation.lowSamples
            << ", \"observedTransitions\": " << observation.observedTransitions << "}";
        out << (index + 1 == gpio.protocolFeedback.size() ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"spi\": {\n";
    out << "    \"available\": " << (spi.available ? "true" : "false") << ",\n";
    out << "    \"device\": \"" << jsonEscape(spi.device) << "\",\n";
    out << "    \"bestStableHz\": " << spi.bestStableHz << ",\n";
    out << "    \"bestThroughputBytesPerSec\": " << spi.bestThroughputBytesPerSec << ",\n";
    out << "    \"note\": \"" << jsonEscape(spi.note) << "\"\n";
    out << "  },\n";
    out << "  \"correctness\": {\n";
    out << "    \"attempted\": " << (correctness.attempted ? "true" : "false") << ",\n";
    out << "    \"available\": " << (correctness.available ? "true" : "false") << ",\n";
    out << "    \"passed\": " << (correctness.passed ? "true" : "false") << ",\n";
    out << "    \"protocolWindowRateHz\": " << correctness.protocolWindowRateHz << ",\n";
    out << "    \"selfReadbackMismatchRatio\": " << correctness.selfReadbackMismatchRatio << ",\n";
    out << "    \"activeFeedbackPins\": " << correctness.activeFeedbackPins << ",\n";
    out << "    \"totalFeedbackPins\": " << correctness.totalFeedbackPins << ",\n";
    out << "    \"note\": \"" << jsonEscape(correctness.note) << "\",\n";
    out << "    \"error\": \"" << jsonEscape(correctness.error) << "\",\n";
    out << "    \"checks\": [\n";
    for (std::size_t index = 0; index < correctness.checks.size(); ++index) {
        const CorrectnessCheck &check = correctness.checks[index];
        out << "      {\"name\": \"" << jsonEscape(check.name)
            << "\", \"passed\": " << (check.passed ? "true" : "false")
            << ", \"note\": \"" << jsonEscape(check.note) << "\"}";
        out << (index + 1 == correctness.checks.size() ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"peakAccuracy\": {\n";
    out << "    \"attempted\": " << (peak.attempted ? "true" : "false") << ",\n";
    out << "    \"available\": " << (peak.available ? "true" : "false") << ",\n";
    out << "    \"passed\": " << (peak.passed ? "true" : "false") << ",\n";
    out << "    \"calibrationStride\": " << peak.calibrationStride << ",\n";
    out << "    \"tieThresholdCounts\": " << peak.tieThresholdCounts << ",\n";
    out << "    \"sum0CsPin\": " << peak.sum0CsPin << ",\n";
    out << "    \"sum1CsPin\": " << peak.sum1CsPin << ",\n";
    out << "    \"spiDevice\": \"" << jsonEscape(peak.spiDevice) << "\",\n";
    out << "    \"selectedSettleSpinIterations\": " << peak.selectedSettleSpinIterations << ",\n";
    out << "    \"selectedWindowGapUs\": " << peak.selectedWindowGapUs << ",\n";
    out << "    \"selectedWindowRateHz\": " << peak.selectedWindowRateHz << ",\n";
    out << "    \"selectedEquivalent8BitGmacs\": " << peak.selectedEquivalent8BitGmacs << ",\n";
    out << "    \"selectedEquivalent8BitGops\": " << peak.selectedEquivalent8BitGops << ",\n";
    out << "    \"selectedAccuracyPercent\": " << peak.selectedAccuracyPercent << ",\n";
    out << "    \"selectedElapsedSeconds\": " << peak.selectedElapsedSeconds << ",\n";
    out << "    \"selectedWindows\": " << peak.selectedWindows << ",\n";
    out << "    \"selectedComparisons\": " << peak.selectedComparisons << ",\n";
    out << "    \"selectedIndeterminateComparisons\": " << peak.selectedIndeterminateComparisons << ",\n";
    out << "    \"selectedPolarity\": \"" << jsonEscape(peak.selectedPolarity) << "\",\n";
    out << "    \"recordedOutWindows\": " << peak.recordedOutWindows << ",\n";
    out << "    \"recordedCalibrationSamples\": " << peak.recordedCalibrationSamples << ",\n";
    out << "    \"outTracePath\": \"" << jsonEscape(peak.outTracePath) << "\",\n";
    out << "    \"calibrationTracePath\": \"" << jsonEscape(peak.calibrationTracePath) << "\",\n";
    out << "    \"note\": \"" << jsonEscape(peak.note) << "\",\n";
    out << "    \"error\": \"" << jsonEscape(peak.error) << "\",\n";
    out << "    \"trials\": [\n";
    for (std::size_t index = 0; index < peak.trials.size(); ++index) {
        const PeakTrialSummary &trial = peak.trials[index];
        out << "      {\"settleSpinIterations\": " << trial.settleSpinIterations
            << ", \"windowGapUs\": " << trial.windowGapUs
            << ", \"windowRateHz\": " << trial.windowRateHz
            << ", \"equivalent8BitGmacs\": " << trial.equivalent8BitGmacs
            << ", \"equivalent8BitGops\": " << trial.equivalent8BitGops
            << ", \"elapsedSeconds\": " << trial.elapsedSeconds
            << ", \"windows\": " << trial.windows
            << ", \"determinateComparisons\": " << trial.determinateComparisons
            << ", \"indeterminateComparisons\": " << trial.indeterminateComparisons
            << ", \"equalAdcPairs\": " << trial.equalAdcPairs
            << ", \"nonEqualAdcPairs\": " << trial.nonEqualAdcPairs
            << ", \"bothZeroAdcPairs\": " << trial.bothZeroAdcPairs
            << ", \"minObservedSum0\": " << trial.minObservedSum0
            << ", \"maxObservedSum0\": " << trial.maxObservedSum0
            << ", \"minObservedSum1\": " << trial.minObservedSum1
            << ", \"maxObservedSum1\": " << trial.maxObservedSum1
            << ", \"accuracyPercent\": " << trial.accuracyPercent
            << ", \"positivePolarityAccuracyPercent\": " << trial.positivePolarityAccuracyPercent
            << ", \"negativePolarityAccuracyPercent\": " << trial.negativePolarityAccuracyPercent
            << ", \"outHighWhenSum0GeSum1\": " << (trial.outHighWhenSum0GeSum1 ? "true" : "false")
            << ", \"passed\": " << (trial.passed ? "true" : "false")
            << ", \"note\": \"" << jsonEscape(trial.note)
            << "\", \"error\": \"" << jsonEscape(trial.error) << "\"}";
        out << (index + 1 == peak.trials.size() ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"boardBudget\": {\n";
    out << "    \"available\": " << (budget.available ? "true" : "false") << ",\n";
    out << "    \"transportModel\": \"" << jsonEscape(budget.transportModel) << "\",\n";
    out << "    \"matrixRows\": " << budget.matrixRows << ",\n";
    out << "    \"matrixCols\": " << budget.matrixCols << ",\n";
    out << "    \"macsPerWindow\": " << budget.macsPerWindow << ",\n";
    out << "    \"equivalent8BitOpsPerWindow\": " << budget.equivalent8BitOpsPerWindow << ",\n";
    out << "    \"estimatedWindowRateHz\": " << budget.estimatedWindowRateHz << ",\n";
    out << "    \"gpioOpsPerWindow\": " << budget.gpioOpsPerWindow << ",\n";
    out << "    \"gpioLineOpsPerSec\": " << budget.gpioLineOpsPerSec << ",\n";
    out << "    \"estimatedVectorOutputsPerSec\": " << budget.estimatedVectorOutputsPerSec << ",\n";
    out << "    \"estimatedComparatorDecisionsPerSec\": " << budget.estimatedComparatorDecisionsPerSec << ",\n";
    out << "    \"estimatedEquivalent8BitGmacs\": " << budget.estimatedEquivalent8BitGmacs << ",\n";
    out << "    \"estimatedEquivalent8BitGops\": " << budget.estimatedEquivalent8BitGops << ",\n";
    out << "    \"limitingFactor\": \"" << jsonEscape(budget.limitingFactor) << "\",\n";
    out << "    \"note\": \"" << jsonEscape(budget.note) << "\",\n";
    out << "    \"alternateTransportNote\": \"" << jsonEscape(budget.alternateTransportNote) << "\"\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

bool writeReportFile(const std::string &path, const std::string &content, std::string &error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "failed to open output file: " + path;
        return false;
    }
    out << content;
    if (!out) {
        error = "failed to write output file: " + path;
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    std::string parseError;
    if (!parseArgs(argc, argv, options, parseError)) {
        std::cerr << parseError << "\n\n";
        printUsage(argv[0]);
        return 1;
    }

    const SystemSnapshot snapshot = collectSystemSnapshot();

    ComputeSummary compute;
    if (!options.skipCpu) {
        compute = runComputeSummary(snapshot, options);
    }

    MemorySummary memory;
    if (!options.skipMemory) {
        memory = runMemorySummary(snapshot, options);
    }

    GpioSummary gpio;
    if (!options.skipGpio) {
        gpio = runGpioSummary(snapshot, options);
    }

    SpiSummary spi;
    if (!options.skipSpi) {
        spi = runSpiSummary(options);
    }

    const CorrectnessSummary correctness = buildCorrectnessSummary(options, gpio);
    const PeakBenchmarkSummary peak = runPeakAccuracyBenchmark(snapshot, options, gpio, spi);
    const BoardEstimate budget = estimateBoardBudget(options, gpio, spi, correctness);

    printSystemSection(snapshot);
    if (!options.skipCpu) {
        printComputeSection(compute);
    }
    if (!options.skipMemory) {
        printMemorySection(memory);
    }
    if (!options.skipGpio) {
        printGpioSection(gpio);
    }
    printCorrectnessSection(correctness);
    if (!options.skipSpi) {
        printSpiSection(spi);
    }
    printPeakAccuracySection(peak);
    printBudgetSection(budget);

    if (!options.jsonOut.empty()) {
        const std::string json = buildJsonReport(options, snapshot, compute, memory, gpio, spi, correctness, peak, budget);
        std::string writeError;
        if (!writeReportFile(options.jsonOut, json, writeError)) {
            std::cerr << writeError << '\n';
            return 2;
        }
        std::cout << "JSON report written to " << options.jsonOut << '\n';
    }

    return 0;
}